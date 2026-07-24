// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/LockFree
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#ifndef CONCURRENT_HASH_SET_H
#define CONCURRENT_HASH_SET_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <bit>
#include <vector>
#include "concurrent_deque.h"
#include <thread>
#include <mutex>
#include "spinlock.h"

struct empty_struct {};

template <typename T, typename... Args>
using DefaultConcurrentDeque = ConcurrentAppendDeque<T, 1024>;

// ===========================================================================
// ConcurrentResizableHashSet -- design overview (read this before the code).
//
// A closed-addressing (chained) hash set that grows by doubling, without ever
// stopping the world and without a global rehash pass. The two big ideas are
// (1) an append-only node arena addressed by index, and (2) lazy, cooperative,
// per-bucket splitting that copies (never moves) live nodes into new buckets.
//
// STORAGE
//   data_    : append-only arena of Node (a ConcurrentAppendDeque). A node is
//              never moved, never freed, and (apart from its tombstone bit)
//              never mutated once published. Nodes are addressed by their
//              arena index, NOT by pointer -- indices are stable because the
//              arena is segmented and blocks never move.
//   buckets_ : array of atomic bucket heads. buckets_[j] holds the arena index
//              of the first node of bucket j's singly linked chain (or a
//              sentinel). buckets_ only ever grows.
//   table_size_ : current number of buckets, always a power of two, and
//              MONOTONICALLY NON-DECREASING. This monotonicity is load-bearing
//              for the read paths (see contains()): a value read once is a
//              valid lower bound forever.
//
// INDEX ENCODING (see the sentinel constants below)
//   A "next" link / bucket head is a 64-bit word: the top bit (MARK_BIT) is a
//   per-node tombstone, the low 63 bits (PTR_MASK) are the arena index or a
//   sentinel. The tombstone lives on the node's OWN next_bucket_node_idx and
//   means "THIS node is logically deleted" (not "my successor is being
//   unlinked", which is the more familiar Harris convention). Real arena
//   indices are small, so they never collide with the EMPTY / UNINITIALIZED
//   sentinels, which sit at the top of the 63-bit range.
//
// LAZY SPLIT REHASH (the heart of the structure; see split_bucket())
//   On a resize from N to 2N buckets, the new buckets [N, 2N) are published as
//   UNINITIALIZED and the old buckets [0, N) are left untouched. A bucket j in
//   [N, 2N) is populated on first access, by ANY thread that touches it, by
//   copying the still-live nodes of its parent bucket (parent = j - N) whose
//   key now hashes to j under the wider mask. The parent chain is NOT modified:
//   after a split, a key that moved to j exists in BOTH the parent chain (a
//   now-unreachable stale copy) and bucket j. Duplicated copies are benign
//   because every reader recomputes its bucket from a freshly loaded
//   table_size_, so it only ever consults the chain that is authoritative for
//   the size it observed. Splitting is recursive: a parent that is itself still
//   UNINITIALIZED is split first, so the bucket tree is filled in on demand.
//
// WHY COPY INSTEAD OF UNLINK/MOVE
//   Because nodes are copied and never unlinked, a node index observed by any
//   traversing thread stays valid for the whole life of the set. That is what
//   lets the read path dereference arena indices with no hazard pointers, no
//   reference counts, and no reclamation protocol at all -- the memory is
//   simply never reclaimed until the whole set is destroyed. The only nodes
//   ever "wasted" are (a) stale parent copies after a split and (b) speculative
//   split subchains that lost their publishing CAS; neither is ever reachable
//   in a way that harms correctness.
//
// SYNCHRONIZATION CHANNELS (all correctness rides on these three)
//   1. buckets_[j] CAS/store is release; every load of buckets_[j] is acquire.
//      Publishing a node (or a split result) into a bucket head with release,
//      and reading the head with acquire, transfers everything the publisher
//      did first -- crucially the node's construction in the arena -- to the
//      reader. This is why a reader may dereference data_[idx].value safely.
//   2. table_size_ store is release; every load is acquire. A resize stores the
//      UNINITIALIZED bucket markers (relaxed) and THEN releases table_size_, so
//      any thread that acquires the new size is guaranteed to see the markers.
//   3. The arena's own size_ handoff (see ConcurrentAppendDeque): a reader that
//      indexes data_ relies on the deque's release-of-size / acquire-of-size
//      pair to see fully constructed nodes and block pointers.
//
// PROGRESS: this structure is lock-free on the pure read/traverse path, but it
// is NOT wait-free and not lock-free end to end: contains(), insert() and
// erase() all fall into split_bucket() when they meet an UNINITIALIZED bucket,
// and split_bucket() allocates through the arena's internal SpinLock; resize
// itself is serialized by resize_lock_.
//
// Template parameters:
//   T           : element (key) type; must be equality-comparable and hashable.
//   AllowDelete : when true, compiles erase() (tombstone deletion). When false,
//                 no node is ever marked, so the MARK_BIT checks are pure
//                 overhead-free no-ops and every chain is append-only.
//   Hash        : hash functor; must return the SAME hash for a key every call
//                 (the split math re-hashes keys under wider masks).
//   Container   : the append-only, index-stable arena template (see the
//                 requirements the two channels above impose on it).
// ===========================================================================
template <
    typename T,
    bool AllowDelete = false,
    typename Hash = std::hash<T>,
    template <typename, typename...> class Container = DefaultConcurrentDeque
>
class ConcurrentResizableHashSet {
public:
    struct Node {
        // The stored key. Written once at construction, then immutable -- readers
        // compare against it with no synchronization beyond the acquire load of
        // the index that reached this node (see synchronization channel 1).
        T value;
        // Encoded link to the next node in this bucket's chain: low 63 bits
        // (PTR_MASK) are the successor's arena index or the EMPTY sentinel; the
        // top bit (MARK_BIT) is THIS node's tombstone. Atomic because erase()
        // sets the tombstone via CAS while readers traverse concurrently, and
        // because it is the field that publishes/observes chain structure.
        std::atomic<size_t> next_bucket_node_idx;

        // Default ctor: an unlinked node whose successor is EMPTY (~MARK_BIT).
        // Rarely used -- the arena is filled via the (val, next) ctor below;
        // this exists only for the container's value-initialization path.
        Node() : value(), next_bucket_node_idx(~(1ULL << 63)) {}
        Node(const T& val, size_t next) : value(val), next_bucket_node_idx(next) {}
    };

private:
    // The dynamically resizable array of atomic bucket heads. 
    // Each index stores the offset of the first node in the bucket's linked list.
    Container<std::atomic<size_t>> buckets_;

    // The monotonically growing block allocator that stores all Node objects. 
    // Nodes are appended block-by-block and never destructed until the set is destroyed.
    Container<Node> data_;

    // The current logical size (number of buckets) of the hash table. Always a power of 2.
    std::atomic<size_t> table_size_;

    // A SpinLock used to serialize table resizes via the Double-Checked Locking Pattern.
    // Only one thread can expand the buckets_ array at a time.
    SpinLock resize_lock_;

    // Index-word encoding. The top bit is the tombstone; the low 63 bits are an
    // arena index or one of two sentinels placed at the very top of that range
    // (so they can never alias a real, small arena index):
    //   MARK_BIT      = 0x8000000000000000 : node's tombstone (top bit).
    //   EMPTY         = 0x7FFFFFFFFFFFFFFF : end-of-chain / "no node" sentinel.
    //   UNINITIALIZED = 0x7FFFFFFFFFFFFFFE : bucket exists but its lazy split
    //                                        has not run yet (see split_bucket).
    //   PTR_MASK      = 0x7FFFFFFFFFFFFFFF : strips the tombstone to recover the
    //                                        index (numerically equal to EMPTY).
    //
    // GOTCHA that dictates every traversal condition: tombstoning the LAST node
    // of a chain stores EMPTY | MARK_BIT == 0xFFFFFFFFFFFFFFFF (SIZE_MAX) into
    // its next link. A naive `while (curr != EMPTY)` would then keep going and
    // dereference SIZE_MAX. Every loop therefore tests `(curr & PTR_MASK) !=
    // EMPTY`, i.e. compares only the index bits, so a marked end-of-chain still
    // terminates traversal cleanly.
    static constexpr size_t MARK_BIT = 1ULL << 63;
    static constexpr size_t EMPTY = ~MARK_BIT;
    static constexpr size_t UNINITIALIZED = EMPTY - 1;
    static constexpr size_t PTR_MASK = ~MARK_BIT;

    // Thread-safe bump allocator. It pushes a new node to the data_ deque
    // and returns its contiguous index offset. 
    // ARCHITECTURE NOTE: We intentionally use the return value of data_.emplace_back() 
    // instead of a separate atomic node_counter_. If we used a separate node_counter_ 
    // and incremented it before emplace_back(), an std::bad_alloc thrown by the deque 
    // would permanently desynchronize the counter from the actual node count. 
    // Returning the size from the internal locked section gives us strict 
    // exception safety and negative overhead (by removing an atomic fetch_add).
    size_t alloc_node(const T& val, size_t next) {
        return data_.emplace_back(val, next);
    }

    // Cooperative lazy split: populate the UNINITIALIZED bucket `j` on first
    // access. Any thread (reader, inserter, eraser) that lands on an
    // UNINITIALIZED bucket runs this; the work is idempotent and at most one
    // thread's result is published, so concurrent callers are safe.
    //
    // Bucket-tree geometry: at table size 2N, bucket j in [N, 2N) was created by
    // the doubling that produced sizes 2N, and its parent is the bucket it split
    // off from. std::bit_floor(j) is the highest power of two <= j, which for
    // j in [N, 2N) is exactly N, so:
    //     parent = j - N        (the sibling bucket in the lower half)
    //     mask   = 2N - 1       (the bucket mask for the CURRENT, wider table)
    // A parent that is itself still UNINITIALIZED is split first (recursion),
    // so an arbitrarily deep chain of ancestors is materialized on demand.
    //
    // The split COPIES, it does not move: we walk the parent chain and, for each
    // still-live (unmarked) node whose key re-hashes to j under the wider mask,
    // prepend a fresh arena node to a private subchain. The parent chain is left
    // completely intact -- this is what keeps already-observed node indices
    // valid forever (see the class overview, "why copy instead of unlink").
    // Marked (tombstoned) nodes are skipped, so logically deleted keys are
    // physically dropped from the new bucket -- resize is the de-facto GC.
    void split_bucket(size_t j) {
        size_t parent = j - std::bit_floor(j);
        size_t parent_head = buckets_[parent].load(std::memory_order_acquire);
        if (parent_head == UNINITIALIZED) {
            split_bucket(parent);
            parent_head = buckets_[parent].load(std::memory_order_acquire);
        }

        size_t N = std::bit_floor(j);
        size_t mask = (N << 1) - 1;   // == 2N-1, the mask for the current table

        // Build the child subchain privately. It is invisible to every other
        // thread until (and unless) the publishing CAS below succeeds, so no
        // synchronization is needed while constructing it.
        size_t new_subchain_head = EMPTY;
        size_t curr = parent_head;

        while ((curr & PTR_MASK) != EMPTY) {   // traverse parent chain
            size_t actual_curr = curr & PTR_MASK;
            T val = data_[actual_curr].value;
            size_t next_raw = data_[actual_curr].next_bucket_node_idx.load(std::memory_order_acquire);
            if (!(next_raw & MARK_BIT)) {          // skip logically deleted nodes
                if ((Hash{}(val) & mask) == j) {   // key belongs to bucket j now
                    // Prepend a fresh copy to the child subchain.
                    new_subchain_head = alloc_node(val, new_subchain_head);
                }
            }
            curr = next_raw;
        } // walk parent chain

        // Publish with a single CAS: only if bucket j is still UNINITIALIZED do
        // we install our subchain (release, so a reader that acquires the head
        // sees every node we constructed). If it fails, another thread already
        // published its own split of j; our subchain was never linked anywhere
        // and is simply abandoned -- benign wasted arena space, never reachable.
        size_t expected = UNINITIALIZED;
        if (buckets_[j].compare_exchange_strong(expected, new_subchain_head, std::memory_order_release, std::memory_order_relaxed)) {
            // CAS succeeded: our subchain is now bucket j's authoritative head.
        } else {
            // CAS failed: subchain abandoned (unpublished, benign memory waste).
        }
    } // split_bucket()

public:
    // Initializes the hash set with the given capacity, rounded up to the
    // nearest power of two (minimum 4, since the bucket mask math assumes a
    // power-of-two table of at least a few slots). Every initial bucket starts
    // EMPTY (not UNINITIALIZED): the original buckets have no parent to split
    // from. table_size_ is released LAST so that any thread which later acquires
    // it is guaranteed to see the fully initialized bucket array.
    ConcurrentResizableHashSet(size_t initial_capacity = 4) {
        if (initial_capacity < 4) initial_capacity = 4;
        initial_capacity = std::bit_ceil(initial_capacity);
        buckets_.resize(initial_capacity);
        for (size_t i = 0; i < initial_capacity; ++i) {
            buckets_[i].store(EMPTY, std::memory_order_relaxed);
        }
        table_size_.store(initial_capacity, std::memory_order_release);
    }

    // Membership test. Lock-free on the fast path (a plain chain walk with no
    // atomic writes); it can, however, fall into split_bucket() -- which
    // allocates under the arena lock -- if it lands on an UNINITIALIZED bucket,
    // so it is not lock-free/wait-free in general (see the class overview).
    // Logically deleted nodes are skipped via the MARK_BIT test.
    //
    // The retry loop exists solely to defend against FALSE NEGATIVES under a
    // concurrent resize: if the table doubled while we were walking a bucket, a
    // key we were looking for may now live in a higher bucket we never examined.
    // Note the deliberate asymmetry between the two exits:
    //   - FOUND returns true immediately, WITHOUT re-reading table_size_. A live
    //     node matching the key is proof of membership at the moment we read its
    //     (unmarked) link, which is a valid linearization point regardless of any
    //     concurrent resize -- so there is nothing to revalidate.
    //   - NOT-FOUND must re-read table_size_. Only if the size is unchanged is a
    //     miss authoritative (we searched the one bucket that can hold the key);
    //     if it grew, we retry against the new geometry. Because table_size_ is
    //     monotone, this loop makes at most one extra pass per intervening
    //     doubling and always terminates.
    bool contains(const T& key) {
        size_t ts = table_size_.load(std::memory_order_acquire);
        while (true) {
            size_t j = Hash{}(key) & (ts - 1);
            size_t head = buckets_[j].load(std::memory_order_acquire);
            if (head == UNINITIALIZED) {
                split_bucket(j);
                head = buckets_[j].load(std::memory_order_acquire);
            }

            bool found = false;
            size_t curr = head;
            while ((curr & PTR_MASK) != EMPTY) {   // walk bucket j's chain
                size_t actual_curr = curr & PTR_MASK;
                // Load the successor link once: it is both the tombstone flag for
                // THIS node and the pointer to the next one, so a single acquire
                // load serves the mark check and the advance.
                size_t check_curr = data_[actual_curr].next_bucket_node_idx.load(std::memory_order_acquire);
                if (data_[actual_curr].value == key && !(check_curr & MARK_BIT)) {
                    found = true;
                    break;
                }
                curr = check_curr;
            } // walk chain
            if (found) return true;

            size_t new_ts = table_size_.load(std::memory_order_acquire);
            if (new_ts == ts) {
                return false;   // table stable: a miss is authoritative
            }
            ts = new_ts;        // table grew under us: retry in new geometry
        }
    } // contains()

    // Lock-free insertion. Prevents lost updates using a dynamic re-insertion fallback
    // if the table resizes beneath the thread during the CAS. Triggers DCLP resizing
    // if the data_.size() exceeds twice the current table size.
    bool insert(const T& key) {
        // ARCHITECTURE NOTE: High-Contention CAS Memory Leak Fix
        // We lazily allocate `new_node` outside the CAS retry loop. If we blindly allocated inside 
        // the loop on every iteration, a failed CAS under high contention would instantly abandon 
        // the node, causing a massive memory leak. By allocating once and reusing the node on CAS 
        // failure (mutating its next pointer), we achieve a zero-overhead fix that dramatically 
        // improves performance under contention.
        size_t new_node = EMPTY;
        while (true) {
            size_t ts = table_size_.load(std::memory_order_acquire);
            size_t j = Hash{}(key) & (ts - 1);
            size_t head = buckets_[j].load(std::memory_order_acquire);
            if (head == UNINITIALIZED) {
                split_bucket(j);
                continue; // Retry after split (re-read head, which is now published)
            }

            // Scan bucket j for the key. A tombstoned node counts as absent, so a
            // key that was erased and is being re-inserted is treated as new (we
            // prepend a fresh live node rather than trying to resurrect the mark).
            bool exists = false;
            size_t curr = head;
            while ((curr & PTR_MASK) != EMPTY) {   // walk bucket j's chain
                size_t actual_curr = curr & PTR_MASK;
                size_t check_curr = data_[actual_curr].next_bucket_node_idx.load(std::memory_order_acquire);
                if (data_[actual_curr].value == key && !(check_curr & MARK_BIT)) {
                    exists = true;
                    break;
                }
                curr = check_curr;
            } // walk chain
            if (exists) {
                /*
                 * ARCHITECTURE NOTE: Rare single-node memory leak
                 * If new_node != EMPTY, we allocated a node, failed our CAS, and upon retrying, 
                 * discovered another thread just inserted this exact key. We return false here, 
                 * permanently orphaning our pre-allocated node. Because we removed global free 
                 * lists to achieve zero-overhead, accepting this incredibly rare, single-node 
                 * leak on concurrent duplicate collisions is the correct architectural trade-off.
                 */
                return false;
            }

            // Prepare the node to prepend. On the first attempt we allocate it;
            // on a CAS-retry we REUSE the same still-private node (its CAS never
            // succeeded, so it was never published) and only repoint its next
            // link at the freshly observed head. The relaxed store is safe
            // precisely because the node is still thread-private -- the release
            // CAS below is what publishes both the link and the node.
            if (new_node == EMPTY) {
                new_node = alloc_node(key, head);
            } else {
                data_[new_node].next_bucket_node_idx.store(head, std::memory_order_relaxed);
            }

            // Publish: prepend by swinging the bucket head from `head` to our
            // node (release). Failure means a concurrent writer changed the head;
            // loop and retry, reusing new_node.
            if (buckets_[j].compare_exchange_strong(head, new_node, std::memory_order_release, std::memory_order_relaxed)) {
                // Post-publish geometry recheck. The CAS succeeded against the
                // table as it was at `ts`. If the table has since doubled and the
                // key's bucket MOVED (correct_j != j), a split of the new bucket
                // may have already snapshotted the parent chain WITHOUT our node,
                // which would strand the key in a bucket no future reader
                // consults. Re-inserting under the new size guarantees the key is
                // present in its correct new bucket. (See the report's flag on the
                // ARM/IRIW note below regarding the memory-ordering framing.)
                size_t check_ts = table_size_.load(std::memory_order_acquire);
                /* 
                 * ARCHITECTURE NOTE: IRIW Race on ARM
                 * We use acq_rel instead of seq_cst for maximum performance.
                 * On ARM architectures, because CAS and table_size_.store are independent 
                 * variables, they lack a total order (IRIW). A thread might see the new 
                 * table_size_ but miss this CAS, dropping the node.
                 * On x86 (TSO), LOCK CMPXCHG is a full barrier, physically preventing this IRIW 
                 * bug. As instructed, we accept the theoretical ARM edge-case in favor of 
                 * zero-overhead x86 performance.
                 */
                if (check_ts > ts) {
                    size_t correct_j = Hash{}(key) & (check_ts - 1);
                    if (correct_j != j) {
                        // Re-publish into the correct new bucket, then return TRUE
                        // unconditionally. We must NOT propagate the nested call's
                        // return value: it would find the copy WE just published
                        // (or a split copy of it) and report false, wrongly telling
                        // the caller the key pre-existed. Reaching here proves our
                        // CAS was the winning, first publication of this key, so
                        // exactly-one-winner semantics require true.
                        insert(key);
                        return true;
                    }
                }
                // Resize trigger, guarded by Double-Checked Locking. The unlocked
                // test `data_.size() > ts*2` is a hint; the decision is remade under
                // resize_lock_ against a fresh table_size_ so only ONE thread
                // doubles per epoch (current_ts == ts). NOTE: data_.size() is TOTAL
                // arena occupancy -- it counts tombstones, stale split copies and
                // abandoned subchains -- so this is an arena-consumption trigger,
                // not a live-load-factor trigger (see the walkthrough's load-factor
                // caveat). New buckets are marked UNINITIALIZED (relaxed) and then
                // table_size_ is released, so any thread that later acquires the new
                // size is guaranteed to observe those markers (channel 2).
                if (data_.size() > ts * 2) {
                    std::lock_guard lock(resize_lock_);
                    size_t current_ts = table_size_.load(std::memory_order_relaxed);
                    if (current_ts == ts) {
                        size_t new_ts = ts * 2;
                        buckets_.resize(new_ts);
                        for (size_t i = ts; i < new_ts; ++i) {
                            buckets_[i].store(UNINITIALIZED, std::memory_order_relaxed);
                        }
                        table_size_.store(new_ts, std::memory_order_release);
                    } // if still the same epoch under the lock
                } // if resize threshold crossed
                return true;
            } // if publishing CAS succeeded
        } // insert retry loop
    } // insert()

    // Tombstone deletion (compiled only when AllowDelete). Finds the live node
    // for `key` and logically deletes it with a single CAS that sets MARK_BIT on
    // its own next link, preserving the successor index. Thereafter contains()
    // skips it and the next split of this bucket physically drops it. Returns
    // true iff THIS call set the tombstone (exactly-one-eraser semantics); a key
    // that is absent or already tombstoned yields false.
    //
    // The retry loop mirrors contains(): re-read table_size_ on a miss to avoid
    // a false negative under concurrent growth. The interesting extra step is
    // the post-mark geometry recheck: if the table doubled and the key's bucket
    // moved, a concurrent split may have copied an UNMARKED copy of this key into
    // the new bucket BEFORE we set our mark. We would then have deleted a copy no
    // reader consults while a live copy survives -- a lost deletion. Re-erasing
    // under the new size idempotently marks that resurrected copy. (This is the
    // erase-side dual of insert()'s re-insert fallback.)
    bool erase(const T& key) requires AllowDelete {
        size_t ts = table_size_.load(std::memory_order_acquire);
        while (true) {
            size_t j = Hash{}(key) & (ts - 1);
            size_t head = buckets_[j].load(std::memory_order_acquire);
            if (head == UNINITIALIZED) {
                split_bucket(j);
                head = buckets_[j].load(std::memory_order_acquire);
            }

            bool erased = false;
            size_t curr = head;
            while ((curr & PTR_MASK) != EMPTY) {   // walk bucket j's chain
                size_t actual_curr = curr & PTR_MASK;
                size_t check_curr = data_[actual_curr].next_bucket_node_idx.load(std::memory_order_acquire);
                if (data_[actual_curr].value == key && !(check_curr & MARK_BIT)) {
                    // Set MARK_BIT while keeping the same successor. CAS on the
                    // observed link value: failure means the link changed under us
                    // (another eraser marked it, since a published node's link only
                    // ever changes by marking) -- so the key is already deleted and
                    // this call reports false.
                    if (data_[actual_curr].next_bucket_node_idx.compare_exchange_strong(check_curr, check_curr | MARK_BIT, std::memory_order_release, std::memory_order_relaxed)) {
                        size_t check_ts = table_size_.load(std::memory_order_acquire);
                        if (check_ts > ts) {
                            size_t correct_j = Hash{}(key) & (check_ts - 1);
                            if (correct_j != j) {
                                erase(key); // re-mark any resurrected copy in the new bucket
                            }
                        }
                        erased = true;
                    }
                    break;
                }
                curr = check_curr;
            } // walk chain
            if (erased) return true;

            size_t new_ts = table_size_.load(std::memory_order_acquire);
            if (new_ts == ts) {
                return false; // stable table, key absent or already tombstoned
            }
            ts = new_ts;      // table grew under us: retry in new geometry
        }
    } // erase()

    // Test-only accessor: total nodes ever allocated in the arena (live + dead).
    // Used by InsertContention_NoMemoryLeak to detect the CAS-retry leak, since a
    // leak inflates this count far above the number of distinct keys inserted.
    size_t get_internal_node_count() const {
        return data_.size();
    }
}; // class ConcurrentResizableHashSet

#endif // CONCURRENT_HASH_SET_H
