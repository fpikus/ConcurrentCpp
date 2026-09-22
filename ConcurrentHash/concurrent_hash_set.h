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
#include <cassert>
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
//              never moved, never freed, and (apart from the two state bits of
//              its link, see below) never mutated once published. Nodes are
//              addressed by their arena index, NOT by pointer -- indices are
//              stable because the arena is segmented and blocks never move.
//   buckets_ : array of atomic bucket heads. buckets_[j] holds the arena index
//              of the first node of bucket j's singly linked chain (or a
//              sentinel), plus the bucket's SEAL LEVEL. buckets_ only ever grows.
//   table_size_ : current number of buckets, always a power of two, and
//              MONOTONICALLY NON-DECREASING. This monotonicity is load-bearing
//              for the read paths (see contains()): a value read once is a
//              valid lower bound forever.
//
// WORD ENCODING (see the constants below for the exact bit layout)
//   Bucket heads and node links are 64-bit words whose low 57 bits (IDX_MASK)
//   are an arena index or a sentinel. The high bits differ by kind of word:
//     - a NODE LINK carries two state bits that describe THIS node (not its
//       successor, which is the more familiar Harris convention):
//         MARK_BIT   : tombstone, "this node is logically deleted";
//         FROZEN_BIT : "this node has been superseded by a copy in a child
//                       bucket; it no longer decides anything about its key".
//       The two bits are mutually exclusive and each is terminal: a link goes
//       live -> MARKED or live -> FROZEN, by one CAS, and never changes again.
//     - a BUCKET HEAD carries a 6-bit SEAL LEVEL: log2 of the largest table size
//       for which a child split has SEALED this bucket's chain (the seal comes
//       first; the snapshot follows, and a splitter may stall in between).
//   Real arena indices are small, so they never collide with the EMPTY /
//   UNINITIALIZED sentinels, which sit at the top of the 57-bit range.
//
// LAZY SPLIT REHASH (the heart of the structure; see split_bucket())
//   On a resize from N to 2N buckets, the new buckets [N, 2N) are published as
//   UNINITIALIZED and the old buckets [0, N) are left untouched. A bucket j in
//   [N, 2N) is populated on first access, by ANY thread that touches it, by
//   copying the still-live nodes of its parent bucket (parent = j - N) whose
//   key now hashes to j under the wider mask. The parent chain is NOT relinked:
//   after a split, a key that moved to j exists in BOTH the parent chain (a
//   stale, FROZEN copy) and bucket j. Splitting is recursive: a parent that is
//   itself still UNINITIALIZED is split first, so the bucket tree is filled in
//   on demand.
//
// WHY COPY INSTEAD OF UNLINK/MOVE
//   Because nodes are copied and never unlinked, a node index observed by any
//   traversing thread stays valid for the whole life of the set. That is what
//   lets the read path dereference arena indices with no hazard pointers, no
//   reference counts, and no reclamation protocol at all -- the memory is
//   simply never reclaimed until the whole set is destroyed. The only nodes
//   ever "wasted" are (a) stale parent copies after a split and (b) speculative
//   split subchains that lost their publishing CAS.
//
// STALE GEOMETRY: EVERY DECISION IS MADE ON ONE ATOMIC
//   An operation picks its bucket from a table_size_ it loaded earlier; the
//   table may double before the operation's deciding CAS. "Load table_size_,
//   CAS something else, then re-check table_size_" does NOT close that window:
//   the two atomics are independent, a CAS on a word nobody else wrote succeeds
//   no matter how stale the geometry is, and a post-hoc re-execution cannot
//   tell its own earlier effect from a competitor's. Instead, the split makes
//   itself visible ON THE SAME WORD each stale operation is about to CAS:
//     - insert() decides by a CAS on the BUCKET HEAD. Before a child split
//       snapshots a parent chain it SEALS the parent head (raises its level,
//       by CAS). Seal and publication are totally ordered on the head word: a
//       node published before the seal is in the snapshot; a stale publication
//       attempted after it fails, because its expected value has the old level.
//     - erase() decides by a CAS on the NODE LINK. Before a child split copies
//       a node it FREEZES that node's link (by CAS). Mark and freeze are totally
//       ordered on the link word: if the mark wins the splitter sees the
//       tombstone and drops the node; if the freeze wins a stale eraser's mark
//       CAS fails, and the eraser retries where the copy lives.
//   Consequently insert() and erase() have NO post-CAS geometry recheck and no
//   recursion, a successful publishing CAS is THE insertion of the key, and a
//   successful marking CAS is THE deletion of it: exactly one insert() returns
//   true per absent->present transition and exactly one erase() returns true
//   per present->absent transition.
//   INVARIANT (key authority, AllowDelete == true): for every key, at any
//   instant, at most one node reachable from a published bucket head is live
//   (neither MARKED nor FROZEN). The key is in the set iff such a node exists,
//   or a FROZEN node for it exists whose child bucket is not yet published (any
//   operation that needs the child publishes it first, see split_bucket()).
//   Authority passes from a node to its single published copy; a tombstone ends
//   the lineage, because MARKED nodes are never copied.
//   With AllowDelete == false the freeze is compiled out: a moved key then has
//   an unmarked node in the parent AND its copy in the child. Nothing is ever
//   tombstoned, so both answer "present" and the weaker invariant "every
//   reachable node for a key agrees" is all that is needed.
//
// SYNCHRONIZATION CHANNELS (all memory-ordering correctness rides on these)
//   1. buckets_[j] CAS/store is release; every load of buckets_[j] is acquire.
//      Publishing a node (or a split result) into a bucket head with release,
//      and reading the head with acquire, transfers everything the publisher
//      did first -- crucially the node's construction in the arena -- to the
//      reader. This is why a reader may dereference data_[idx].value safely.
//   2. table_size_ store is release; every load is acquire. A resize stores the
//      UNINITIALIZED bucket markers (relaxed) and THEN releases table_size_, so
//      any thread that acquires the new size is guaranteed to see the markers.
//   3. The arena. ConcurrentAppendDeque's rule is that a thread may index an
//      element only after it has learned, with an acquire, how many elements
//      exist: from size(), or from the return value of its own emplace_back().
//      No read path here calls data_.size(); instead the thread that appended
//      the node (and so knows its index) hands the index to readers through
//      channel 1 -- a release CAS on a bucket head or a release store into a
//      link, read with acquire -- which is the same handoff size() performs,
//      with the head or link word in the role of size_. operator[] then does
//      its own acquire load of the block directory. Likewise buckets_[j] is
//      indexed on the strength of channel 2 (table_size_ is released after
//      buckets_.resize()), never of buckets_.size().
//   4. A node link's state bits are set by a release CAS and read by acquire
//      loads: an operation that returns after erase() returned sees the
//      tombstone, and a thread that sees FROZEN or a raised seal level is
//      ordered after the splitter's acquire of the larger table_size_, so its
//      own next acquire of table_size_ returns the larger size (by read-read
//      coherence). For the seal this depends on RELEASE SEQUENCES: a stale
//      inserter usually reads a head value written by a LATER inserter's CAS,
//      not by the seal CAS itself, and it synchronizes with the sealer only
//      because every write to a published head is a read-modify-write, which
//      extends the sealer's release sequence. A plain store to a published head
//      would break this argument: never add one. (Links are safe as they are:
//      nothing is ever written to a link after MARK or FROZEN.)
//   Linearizability throughout this class is with respect to happens-before,
//   which is all the C++ memory model can express: "an erase() that returned
//   before this call began" means the return happens-before the call (e.g. the
//   caller synchronized), not wall-clock order.
//
// WEAK MEMORY (ARM, POWER): WHAT THIS DESIGN DOES AND DOES NOT RELY ON
//   The decisions above are atomic read-modify-writes of a single word, which
//   are totally ordered with every other write of that word on every
//   architecture. No correctness argument in this class has the store-buffering
//   (Dekker) shape "A: write x, read y  ||  B: write y, read x", which is the
//   shape the removed post-CAS rechecks had (x = head or link, y = table_size_)
//   and which acquire/release on two DIFFERENT atomics does not order.
//   What remains cross-variable is only the read-side chain of channel 4
//   (acquire word -> acquire table_size_). It is used for PROGRESS: a thread
//   that meets a seal level above its own, or a FROZEN node for its key,
//   reloads table_size_ and retries, and the chain says the reload returns a
//   larger size at once. Were that edge missing the thread would re-read until
//   the larger size became visible; it would never return a wrong answer.
//   A miss that is confirmed by an unchanged table_size_ (contains(), erase())
//   linearizes at the first table_size_ load; see contains().
//
// PROGRESS: this structure is lock-free on the pure read/traverse path, but it
// is NOT wait-free and not lock-free end to end: contains(), insert() and
// erase() all fall into split_bucket() when they meet an UNINITIALIZED bucket,
// and split_bucket() allocates through the arena's internal SpinLock, as does
// every insert() (alloc_node()); resize itself is serialized by resize_lock_.
//
// Template parameters:
//   T           : element (key) type; must be equality-comparable and hashable.
//   AllowDelete : when true, compiles erase() (tombstone deletion) and the
//                 freeze step of split_bucket(). When false, no node is ever
//                 marked or frozen, so the state-bit checks are overhead-free
//                 no-ops and every chain is append-only.
//   Hash        : hash functor; must return the SAME hash for a key every call
//                 (the split math re-hashes keys under wider masks).
//   Container   : the append-only, index-stable arena template (see the
//                 requirements the channels above impose on it).
// ===========================================================================
template <
    typename T,
    bool AllowDelete = false,
    typename Hash = std::hash<T>,
    template <typename, typename...> class Container = DefaultConcurrentDeque
>
class ConcurrentResizableHashSet {
private:
    // Word encoding. Both kinds of word keep an arena index, or one of two
    // sentinels placed at the very top of the index range (so they can never
    // alias a real, small arena index), in their low 57 bits:
    //
    //   bit      63      62     61..57     56..0
    //   LINK   MARK    FROZEN   zero       successor index | EMPTY
    //   HEAD   zero    [  seal level  ]    first-node index | EMPTY | UNINITIALIZED
    //                  (bits 62..57)
    //
    //   IDX_MASK      = 0x01FFFFFFFFFFFFFF : the index field of either word.
    //   EMPTY         = 0x01FFFFFFFFFFFFFF : end-of-chain / "no node" sentinel
    //                                        (numerically equal to IDX_MASK).
    //   UNINITIALIZED = 0x01FFFFFFFFFFFFFE : bucket exists but its lazy split
    //                                        has not run yet (see split_bucket).
    //                   The largest usable arena index is therefore 2^57 - 3.
    //   MARK_BIT      = 1 << 63 (links only): the node's tombstone. Set by
    //                   erase() with one CAS whose expected value is the LIVE
    //                   link (no MARK, no FROZEN).
    //   FROZEN_BIT    = 1 << 62 (links only): the node has been, or is about to
    //                   be, copied into a child bucket by split_bucket(); it can
    //                   never be marked afterwards. Set with one CAS whose
    //                   expected value is the LIVE link. MARK and FROZEN are
    //                   therefore mutually exclusive and both terminal.
    //   LEVEL_MASK    = 0x3F << 57 (heads only): the seal level, log2 of the
    //                   largest table size for which a child split has SEALED
    //                   this chain (the snapshot that follows the seal may not
    //                   have been taken yet); 0 = never sealed. Levels
    //                   only grow. Six bits hold any log2 of a 64-bit size. The
    //                   level bits of a head overlap FROZEN_BIT of a link, so a
    //                   head word is never stored into a link without masking
    //                   (see insert()) and vice versa.
    //
    // GOTCHA that dictates every traversal condition: the LAST node of a chain
    // holds EMPTY in its index field, and once it is tombstoned or frozen its
    // link is EMPTY | MARK_BIT or EMPTY | FROZEN_BIT. A naive
    // `while (curr != EMPTY)` would then keep going and dereference garbage.
    // Every loop therefore tests `(curr & IDX_MASK) != EMPTY`, i.e. compares
    // only the index bits, and every dereference uses `curr & IDX_MASK`.
    static constexpr size_t MARK_BIT = 1ULL << 63;
    static constexpr size_t FROZEN_BIT = 1ULL << 62;
    static constexpr unsigned LEVEL_SHIFT = 57;
    static constexpr size_t LEVEL_MASK = 0x3FULL << LEVEL_SHIFT;
    static constexpr size_t IDX_MASK = (1ULL << LEVEL_SHIFT) - 1;
    static constexpr size_t EMPTY = IDX_MASK;
    static constexpr size_t UNINITIALIZED = EMPTY - 1;

public:
    struct Node {
        // The stored key. Written once at construction, then immutable -- readers
        // compare against it with no synchronization beyond the acquire load of
        // the index that reached this node (see synchronization channel 1).
        T value;
        // Encoded link to the next node in this bucket's chain: the low 57 bits
        // (IDX_MASK) are the successor's arena index or the EMPTY sentinel; bit
        // 63 (MARK_BIT) is THIS node's tombstone and bit 62 (FROZEN_BIT) says
        // THIS node was superseded by a copy in a child bucket. Atomic because
        // erase() and split_bucket() set those bits via CAS while readers
        // traverse concurrently, and because it is the field that
        // publishes/observes chain structure. The index bits never change once
        // the node is published.
        std::atomic<size_t> next_bucket_node_idx;

        // Default ctor: an unlinked live node whose successor is EMPTY. Rarely
        // used -- the arena is filled via the (val, next) ctor below; this
        // exists only for the container's value-initialization path.
        Node() : value(), next_bucket_node_idx(EMPTY) {}
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


    // Seal level stored in the head word `head`.
    static constexpr size_t level_of(size_t head) { return (head & LEVEL_MASK) >> LEVEL_SHIFT; }
    // Seal level that corresponds to the table size `ts` (a power of two): a
    // thread working with table size ts may publish into a bucket only while
    // level_of(head) <= level_for(ts).
    static constexpr size_t level_for(size_t ts) { return static_cast<size_t>(std::countr_zero(ts)); }

    // Thread-safe bump allocator. It pushes a new node to the data_ deque
    // and returns its contiguous index offset.
    // ARCHITECTURE NOTE: We intentionally use the return value of data_.emplace_back()
    // instead of a separate atomic node_counter_. If we used a separate node_counter_
    // and incremented it before emplace_back(), an std::bad_alloc thrown by the deque
    // would permanently desynchronize the counter from the actual node count.
    // Returning the size from the internal locked section gives us strict
    // exception safety and negative overhead (by removing an atomic fetch_add).
    size_t alloc_node(const T& val, size_t next) {
        size_t idx = data_.emplace_back(val, next);
        // The index field is 57 bits wide and the two top values are sentinels
        // (see the constants): 2^57 - 3 is the largest index a link or head can
        // hold. Unreachable in practice; checked in debug builds only.
        assert(idx < UNINITIALIZED);
        return idx;
    } // alloc_node()

    // Cooperative lazy split: populate the UNINITIALIZED bucket `j` on first
    // access. Any thread (reader, inserter, eraser) that lands on an
    // UNINITIALIZED bucket runs this; the work is idempotent and at most one
    // thread's result is published, so concurrent callers are safe. On return
    // bucket j is published (by this thread or another).
    //
    // Bucket-tree geometry: at table size 2N, bucket j in [N, 2N) was created by
    // the doubling that produced sizes 2N, and its parent is the bucket it split
    // off from. std::bit_floor(j) is the highest power of two <= j, which for
    // j in [N, 2N) is exactly N, so:
    //     parent = j - N        (the sibling bucket in the lower half)
    //     mask   = 2N - 1       (the bucket mask for the table that created j)
    // A parent that is itself still UNINITIALIZED is split first (recursion),
    // so an arbitrarily deep chain of ancestors is materialized on demand.
    //
    // The split runs in three steps, each of which hands off through one word:
    //   1. SEAL the parent head to the level of table size 2N and take exactly
    //      the sealed head as the snapshot. From here on no thread working with
    //      a table size < 2N can publish into the parent (see insert()), so
    //      every node of a key that belongs to j is either in the snapshot or
    //      will be published directly into j (or a descendant of j). Nodes that
    //      ARE still published into the parent after the seal come from threads
    //      at table size >= 2N, hence hash to the parent under `mask`, not to j:
    //      every thread that splits j therefore selects the same set of nodes.
    //   2. Walk the snapshot. For each node whose key re-hashes to j: FREEZE its
    //      link, then copy it to a private subchain unless it turned out to be
    //      tombstoned. Freeze and tombstone are decided by CAS on the same link
    //      word, each from the live value, so exactly one of them happens, it is
    //      final, and every splitter of j makes the same copy/skip decision for
    //      the node. A node is copied only if it is FROZEN, and a FROZEN node can
    //      never be tombstoned: no erase() can succeed on a node that has, or
    //      will have, a copy. Tombstoned nodes are skipped, so logically deleted
    //      keys are physically dropped from the new bucket -- resize is the
    //      de-facto GC. Only nodes that MOVE to j are frozen; a node that stays
    //      in the parent under `mask` remains live and authoritative there, and
    //      may be frozen by a later doubling's child. A node is frozen at most
    //      once: if its key hashes to j = parent + N under 2N - 1, it cannot
    //      hash to parent + 2^m*N under 2^(m+1)*N - 1 for any m >= 1 (that
    //      would require bit N of the hash to be clear), so no other child of
    //      this parent ever selects it, and it is never copied forward again.
    //      (With AllowDelete == false there are no tombstones, nothing can race
    //      with the copy, and the freeze is compiled out.)
    //   3. Publish the subchain with one CAS on bucket j's head.
    // The parent chain is never relinked -- this is what keeps already-observed
    // node indices valid forever (see the class overview).
    void split_bucket(size_t j) {
        // Bucket 0 has no parent and is published EMPTY by the constructor, so
        // it can never be UNINITIALIZED and never reaches here. The guard
        // matters: bit_floor(0) == 0 would make `seal` 64, which a 6-bit level
        // cannot reach, and the seal loop below would not terminate.
        assert(j > 0);
        size_t parent = j - std::bit_floor(j);
        size_t N = std::bit_floor(j);
        size_t mask = (N << 1) - 1;   // == 2N-1, the mask for the table that created j
        size_t parent_head = buckets_[parent].load(std::memory_order_acquire);
        if ((parent_head & IDX_MASK) == UNINITIALIZED) {
            split_bucket(parent);
            // The parent is published now and a published head never returns to
            // UNINITIALIZED, so the seal below never seals an unsplit bucket.
            parent_head = buckets_[parent].load(std::memory_order_acquire);
        }

        // Step 1, SEAL. Raise the parent's level to that of table size 2N unless
        // it is already there (another splitter of j, or a split of a later
        // child of the same parent, got here first). Success is release: the
        // sealed head is what stale inserters acquire, and it orders them after
        // our caller's acquire of table_size_ >= 2N (channel 4). Failure is
        // acquire: the head we get back was released by an inserter (or another
        // sealer) and we are going to walk the chain it points to (channel 1).
        // Weak CAS: we are in a retry loop anyway.
        const size_t seal = level_for(N << 1);
        while (level_of(parent_head) < seal) {
            size_t sealed = (parent_head & ~LEVEL_MASK) | (seal << LEVEL_SHIFT);
            if (buckets_[parent].compare_exchange_weak(parent_head, sealed, std::memory_order_release, std::memory_order_acquire)) {
                parent_head = sealed;
            }
        } // seal loop

        // Steps 2 and 3. Build the child subchain privately. It is invisible to
        // every other thread until (and unless) the publishing CAS below
        // succeeds, so no synchronization is needed while constructing it.
        size_t new_subchain_head = EMPTY;
        size_t curr = parent_head;

        while ((curr & IDX_MASK) != EMPTY) {   // traverse parent chain
            size_t actual_curr = curr & IDX_MASK;
            T val = data_[actual_curr].value;
            size_t next_raw = data_[actual_curr].next_bucket_node_idx.load(std::memory_order_acquire);
            if (!(next_raw & MARK_BIT)) {          // skip logically deleted nodes
                if ((Hash{}(val) & mask) == j) {   // key belongs to bucket j now
                    if constexpr (AllowDelete) {
                        // FREEZE before copying. Expected value: the live link
                        // we just read. Strong CAS, so failure means the link
                        // really changed, and a published link changes only by
                        // gaining MARK (an eraser won: skip the node) or FROZEN
                        // (another splitter of j won: copy it, as that splitter
                        // does). On failure next_raw is the value that failed
                        // the comparison, i.e. one of those two final states.
                        // Success is release: a stale eraser that acquires the
                        // FROZEN link is thereby ordered after our caller's
                        // acquire of table_size_ >= 2N (channel 4). Failure is
                        // relaxed: both outcomes are decided by the returned
                        // bits alone, we need nothing else its writer did.
                        if (!(next_raw & FROZEN_BIT)) {
                            if (data_[actual_curr].next_bucket_node_idx.compare_exchange_strong(next_raw, next_raw | FROZEN_BIT, std::memory_order_release, std::memory_order_relaxed)) {
                                next_raw |= FROZEN_BIT;
                            }
                        } // if not frozen yet
                    } // if erase() exists
                    if (!(next_raw & MARK_BIT)) {
                        // Prepend a fresh live copy to the child subchain.
                        new_subchain_head = alloc_node(val, new_subchain_head);
                    }
                } // if key belongs to bucket j
            } // if not tombstoned
            curr = next_raw;
        } // walk parent chain

        // Publish with a single CAS: only if bucket j is still UNINITIALIZED do
        // we install our subchain (release, so a reader that acquires the head
        // sees every node we constructed). The child starts at seal level 0
        // (UNINITIALIZED carries none and new_subchain_head is a bare index or
        // EMPTY). If the CAS fails, another thread already published its own,
        // equivalent split of j; our subchain was never linked anywhere and is
        // simply abandoned -- benign wasted arena space, never reachable. The
        // nodes we froze stay frozen, which is correct: the winner copied them.
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
    // EMPTY (not UNINITIALIZED) at seal level 0: the original buckets have no
    // parent to split from. table_size_ is released LAST so that any thread
    // which later acquires it is guaranteed to see the fully initialized bucket
    // array.
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
    // Logically deleted nodes are skipped via the MARK_BIT test. The head's seal
    // level and a node's FROZEN bit are deliberately IGNORED here (they are only
    // masked off the index): contains() publishes nothing, so a stale geometry
    // cannot make it corrupt anything, and its answers stay linearizable:
    //   - FOUND returns true immediately, WITHOUT re-reading table_size_, for a
    //     matching node that is not tombstoned, FROZEN or not. If the node is
    //     live, the key is in the set now. If it is FROZEN, this chain is stale
    //     for the key. The freeze does NOT happen-before our first table_size_
    //     load: the freezer had acquired the larger size first, so by read-read
    //     coherence our load would have returned it. The freeze does happen-
    //     before our acquire load of the link. So the freeze is not ordered
    //     before the call and is ordered before its end: it can be placed inside
    //     this call, and at the instant of the freeze the node was live, i.e.
    //     the key was in the set -- that is our linearization point. An erase()
    //     of the key's current copy whose return happens-before this call
    //     cannot be missed this way: it worked in the larger geometry, so our
    //     first table_size_ load would have returned that geometry, not the
    //     stale one. (Happens-before, not wall clock: see the class overview.)
    //   - NOT-FOUND must re-read table_size_. Only if the size is unchanged is a
    //     miss authoritative (we searched the one bucket that can hold the key);
    //     if it grew, we retry against the new geometry. Because table_size_ is
    //     monotone, this loop makes at most one extra pass per intervening
    //     doubling and always terminates.
    bool contains(const T& key) {
        size_t ts = table_size_.load(std::memory_order_acquire);
        while (true) {
            size_t j = Hash{}(key) & (ts - 1);
            size_t head = buckets_[j].load(std::memory_order_acquire) & IDX_MASK;
            if (head == UNINITIALIZED) {
                split_bucket(j);
                head = buckets_[j].load(std::memory_order_acquire) & IDX_MASK;
            }

            bool found = false;
            size_t curr = head;
            while ((curr & IDX_MASK) != EMPTY) {   // walk bucket j's chain
                size_t actual_curr = curr & IDX_MASK;
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

    // Insertion. Returns true iff THIS call made the key a member: exactly one
    // insert() returns true per absent->present transition of the key, across
    // any number of concurrent resizes. Returns false if the key was present at
    // some instant during the call. Not lock-free: every new node is allocated
    // under the arena's SpinLock (alloc_node()), a bucket that is still
    // UNINITIALIZED is split first, and a successful insert may perform the
    // DCLP-guarded doubling when data_.size() exceeds twice the table size.
    //
    // The decision is the publishing CAS on the bucket head, and the geometry is
    // validated by that same CAS: its expected value includes the head's seal
    // level, which was checked against the table size this attempt works with.
    // See "STALE GEOMETRY" in the class overview and step 1 of split_bucket().
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
            if ((head & IDX_MASK) == UNINITIALIZED) {
                split_bucket(j);
                continue; // Retry after split (re-read head, which is now published)
            }
            // GEOMETRY CHECK. A seal level above ours means a child split for a
            // larger table has snapshotted (or is about to snapshot) this chain:
            // `ts` is stale and the key may no longer belong here, so neither a
            // "present" nor an "absent" verdict from this chain can be trusted.
            // Reload table_size_ and start over. Termination: the sealer acquired
            // the larger size before its release CAS of this head, and we
            // acquired the head, so the reload returns a size whose level is at
            // least the one we saw (channel 4); table_size_ is monotone and every
            // level in a head is the level of a size that was already stored.
            // A level <= ours is fine: it says only that keys which hash
            // elsewhere under OUR mask have been moved out.
            if (level_of(head) > level_for(ts)) continue;

            // Scan bucket j for the key. A tombstoned node counts as absent, so a
            // key that was erased and is being re-inserted is treated as new (we
            // prepend a fresh live node rather than trying to resurrect the mark).
            // FROZEN needs no test here: a node of OUR key in OUR bucket can be
            // frozen only by a split for a table size above `ts`, which sealed
            // this head first. We read a head that was not sealed that high, so
            // the node was still live when we read the head, and "present" was
            // the truth at that instant; an "absent" verdict is validated by the
            // publishing CAS below.
            bool exists = false;
            size_t curr = head & IDX_MASK;
            while ((curr & IDX_MASK) != EMPTY) {   // walk bucket j's chain
                size_t actual_curr = curr & IDX_MASK;
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
            // CAS below is what publishes both the link and the node. The link
            // gets the head's INDEX only: the head's level bits overlap the
            // link's FROZEN bit and must not leak into it.
            if (new_node == EMPTY) {
                new_node = alloc_node(key, head & IDX_MASK);
            } else {
                data_[new_node].next_bucket_node_idx.store(head & IDX_MASK, std::memory_order_relaxed);
            }

            // Publish: prepend by swinging the bucket head from `head` to our
            // node, keeping the bucket's seal level (release). The expected value
            // is the FULL word we validated above, index and level, so the CAS
            // fails if another writer prepended a node OR a splitter sealed the
            // bucket since we read it; either way loop and retry from the
            // table_size_ load, reusing new_node. Failure is relaxed: the
            // returned value is not used.
            if (buckets_[j].compare_exchange_strong(head, new_node | (head & LEVEL_MASK), std::memory_order_release, std::memory_order_relaxed)) {
                // No post-publish geometry recheck. The head we replaced carried a
                // level <= level_for(ts), so at the instant of this CAS no split
                // for a table larger than `ts` had snapshotted this chain: any
                // such split seals after us, takes its snapshot from the sealed
                // head, and therefore sees this node (or a successor that links
                // to it). The key cannot be stranded, no other thread can have
                // published it elsewhere without first copying this chain, and
                // this CAS is the one and only publication of the key: `true` is
                // exact. Nothing here depends on how this CAS is ordered relative
                // to the table_size_ store of a concurrent resize.
                //
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
    // skips it and no split ever copies it. Returns true iff THIS call set the
    // tombstone: exactly one erase() returns true per present->absent transition
    // of the key, across any number of concurrent resizes. A key that is absent
    // or already tombstoned yields false.
    //
    // The decision is the marking CAS, and the geometry is validated by that same
    // CAS: its expected value is the LIVE link (no MARK, no FROZEN). A split that
    // moves the key to a child bucket freezes the node before copying it (step 2
    // of split_bucket()), on this very word, so:
    //   - our CAS succeeded => the node was not frozen => no copy of it exists or
    //     ever will: it was the key's one authoritative node and the key is gone.
    //     There is nothing to re-check and nothing to re-erase.
    //   - the node is FROZEN (seen on the walk, or as the reason our CAS failed)
    //     => a copy exists or is being made; this chain is stale for the key.
    //     We reload table_size_ and retry where the copy lives. If its bucket is
    //     still UNINITIALIZED we publish it ourselves through split_bucket(),
    //     like every other operation, so we never wait for the splitter.
    //   - our CAS failed and the node is MARKED => another eraser won; false.
    // A strong CAS on a published link can fail for no other reason: the only
    // writers of a published link are this CAS and the freeze CAS, both from the
    // live value (insert()'s plain store targets a node that is still private).
    //
    // Retry bound. A FROZEN node of our key in our bucket was frozen for a table
    // size strictly above `ts` (under any size <= ts the key still hashes here),
    // and its freezer acquired that size before the release CAS whose result we
    // acquired, so the reload returns a larger size (channel 4): at most one
    // retry per doubling of the table, as for a plain miss. The loop does not
    // DEPEND on that for safety: should the reload still return `ts`, we walk
    // again and reload again until the larger size is visible; `false` is never
    // returned from a stale chain.
    bool erase(const T& key) requires AllowDelete {
        size_t ts = table_size_.load(std::memory_order_acquire);
        while (true) {
            size_t j = Hash{}(key) & (ts - 1);
            size_t head = buckets_[j].load(std::memory_order_acquire) & IDX_MASK;
            if (head == UNINITIALIZED) {
                split_bucket(j);
                head = buckets_[j].load(std::memory_order_acquire) & IDX_MASK;
            }

            // Set when this chain proved stale for the key (FROZEN node found):
            // a miss in it is then NOT authoritative even if table_size_ reads
            // unchanged.
            bool stale = false;
            size_t curr = head;
            while ((curr & IDX_MASK) != EMPTY) {   // walk bucket j's chain
                size_t actual_curr = curr & IDX_MASK;
                size_t check_curr = data_[actual_curr].next_bucket_node_idx.load(std::memory_order_acquire);
                if (data_[actual_curr].value == key && !(check_curr & MARK_BIT)) {
                    // Live (or frozen) node of our key. If live, set MARK_BIT while
                    // keeping the same successor. Success is release: operations
                    // that acquire this link afterwards see the key as deleted.
                    // Failure is acquire: if the reason is FROZEN we go on to
                    // reload table_size_, and the acquire orders that reload after
                    // the freezer's view of the table (see "Retry bound" above).
                    if (!(check_curr & FROZEN_BIT) &&
                        data_[actual_curr].next_bucket_node_idx.compare_exchange_strong(check_curr, check_curr | MARK_BIT, std::memory_order_release, std::memory_order_acquire)) {
                        return true;   // this CAS is THE deletion of the key
                    }
                    // Not marked by us; check_curr holds the link's current value.
                    // FROZEN: stale geometry, retry in the current one. MARKED:
                    // another eraser deleted the key; report false below unless the
                    // table grew (then the retry may find and delete a newer
                    // insertion of the key, which is equally linearizable).
                    stale = (check_curr & FROZEN_BIT) != 0;
                    break;
                } // if found an unmarked node of our key
                curr = check_curr;
            } // walk chain

            size_t new_ts = table_size_.load(std::memory_order_acquire);
            if (new_ts == ts && !stale) {
                return false; // stable table, key absent or already tombstoned
            }
            ts = new_ts;      // table grew under us: retry in new geometry
        } // while (true)
    } // erase()

    // Test-only accessor: total nodes ever allocated in the arena (live + dead).
    // Used by InsertContention_NoMemoryLeak to detect the CAS-retry leak, since a
    // leak inflates this count far above the number of distinct keys inserted.
    size_t get_internal_node_count() const {
        return data_.size();
    }
}; // class ConcurrentResizableHashSet

#endif // CONCURRENT_HASH_SET_H
