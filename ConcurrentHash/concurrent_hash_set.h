// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/ConcurrentCpp
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
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <bit>
#include <vector>
#include <memory>
#include "concurrent_deque.h"
#include <thread>
#include <mutex>
#include "spinlock.h"

struct empty_struct {};

namespace concurrent_hash_detail {
// Process-wide sequential thread numbering, shared by every instantiation of
// ConcurrentResizableHashSet: a thread's number is assigned on its first
// allocation and never changes. Sequential (not hashed) on purpose: threads
// started together get consecutive numbers, so `number mod shards` spreads a
// batch of up to `shards` threads over distinct arena shards with no
// collisions, however many earlier threads have come and gone.
inline std::atomic<unsigned> next_thread_number{0};
inline unsigned thread_number() {
    static thread_local unsigned number = next_thread_number.fetch_add(1, std::memory_order_relaxed);
    return number;
}
} // namespace concurrent_hash_detail

template <typename T, typename... Args>
using DefaultConcurrentDeque = ConcurrentAppendDeque<T, 1024>;

// ===========================================================================
// ConcurrentResizableHashSet -- design overview (read this before the code).
//
// A closed-addressing (chained) hash set that grows by doubling, without ever
// stopping the world and without a global rehash pass. The two big ideas are
// (1) an append-only node arena in which nothing ever moves, and (2) lazy, cooperative,
// per-bucket splitting that copies (never moves) live nodes into new buckets.
//
// STORAGE
//   arenas_  : the node arena, SHARDED: an array of append-only deques of Node
//              (ConcurrentAppendDeque), a thread always appending to the shard
//              its thread number selects (see alloc_node()). A node is never
//              moved, never freed, and (apart from the two state bits of its
//              link, see below) never mutated once published. Nodes are
//              addressed by POINTER: a deque is segmented and its blocks never
//              move, so a node's address is stable for the life of the set, and
//              nothing that reads the structure knows or cares which shard a
//              node lives in. The deque's index is used exactly once, to obtain
//              that address at allocation; no read path ever goes through a
//              deque. Sharding exists because one shared per-node atomic of any
//              kind (a lock, a counter) was measured to be the insert ceiling;
//              a shard's lock is contended only by the threads whose numbers
//              collide modulo the shard count.
//   node_count_ : arena occupancy, OVER-counted by up to 255 per touched shard
//              (each shard adds 256 when it appends a node whose index is a
//              multiple of 256, index 0 included); the resize hint.
//   buckets_ : array of atomic bucket heads. buckets_[j] holds the address of
//              the first node of bucket j's singly linked chain (or a
//              sentinel), plus the bucket's SEAL LEVEL. buckets_ only ever grows.
//   table_size_ : current number of buckets, always a power of two, and
//              MONOTONICALLY NON-DECREASING. This monotonicity is load-bearing
//              for the read paths (see contains()): a value read once is a
//              valid lower bound forever.
//
// WORD ENCODING (see the constants below for the exact bit layout)
//   Bucket heads and node links are 64-bit words holding a node address or a
//   sentinel, plus tag bits in the parts of the word a user-space address never
//   uses: the low three bits (nodes are 8-aligned) and the top six (bits 63..58
//   are zero on every supported platform). The tags differ by kind of word:
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
//   The EMPTY / UNINITIALIZED sentinels are the values 0 and 8: no node can
//   live at either address.
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
//   Because nodes are copied and never unlinked, a node address observed by any
//   traversing thread stays valid for the whole life of the set. That is what
//   lets the read path dereference node pointers with no hazard pointers, no
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
//      reader. This is why a reader may dereference node->value safely.
//   2. table_size_ store is release; every load is acquire. A resize stores the
//      UNINITIALIZED bucket markers (relaxed) and THEN releases table_size_, so
//      any thread that acquires the new size is guaranteed to see the markers.
//   3. The arena. ConcurrentAppendDeque's rule is that a thread may index an
//      element only after it has learned, with an acquire, how many elements
//      exist: from size(), or from the return value of its own emplace_back().
//      Only alloc_node() indexes the arena, with the index its own emplace_back()
//      returned, and it does so once, to take the node's address. From then on
//      the node is reached by pointer only: the thread that constructed it hands
//      the address to readers through channel 1 -- a release CAS on a bucket
//      head, read with acquire (a node reached through a link was published
//      by the head CAS of the node that links to it, or of a later prepend)
//      -- and that acquire is what makes the node's construction visible to
//      the reader.
//      The deque's directory is never consulted on a read path. buckets_[j], by
//      contrast, IS indexed, on the strength of channel 2 (table_size_ is
//      released after buckets_.resize()), never of buckets_.size().
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
// and split_bucket() allocates through its arena shard's internal SpinLock, as
// does every insert() (alloc_node()); with one thread per shard that lock is
// uncontended, but it is a lock. Resize itself is serialized by resize_lock_.
//
// Template parameters:
//   T           : element (key) type; must be equality-comparable and hashable.
//   AllowDelete : when true, compiles erase() (tombstone deletion) and the
//                 freeze step of split_bucket(). When false, no node is ever
//                 marked or frozen, so the state-bit checks are overhead-free
//                 no-ops and every chain is append-only.
//   Hash        : hash functor; must return the SAME hash for a key every call
//                 (the split math re-hashes keys under wider masks).
//   Container   : the append-only, address-stable arena template: elements
//                 never move once constructed (see channel 3).
// ===========================================================================
template <
    typename T,
    bool AllowDelete = false,
    typename Hash = std::hash<T>,
    template <typename, typename...> class Container = DefaultConcurrentDeque
>
class ConcurrentResizableHashSet {
private:
    // Word encoding. Both kinds of word keep a node address, or one of two
    // sentinel values no node can have as its address, in the bits a user-space
    // pointer occupies; the tag bits sit where such a pointer is always zero:
    //
    //   bit     63..58      57..3            2      1       0
    //   LINK    zero        successor address | EMPTY   zero  FROZEN  MARK
    //   HEAD    [ seal level ]  first-node address | EMPTY | UNINITIALIZED  zero
    //
    //   PTR_MASK      = ~(LEVEL_MASK | 7)  : the address field of either word.
    //   EMPTY         = 0                  : end-of-chain / "no node" sentinel
    //                                        (a null pointer).
    //   UNINITIALIZED = 8                  : bucket exists but its lazy split
    //                                        has not run yet (see split_bucket).
    //                   A node is a heap object of at least 16 bytes, so its
    //                   address is neither 0 nor 8; 8 is 8-aligned, so it
    //                   survives PTR_MASK and is compared through addr_of()
    //                   like an address.
    //   MARK_BIT      = 1 (links only): the node's tombstone. Set by erase()
    //                   with one CAS whose expected value is the LIVE link (no
    //                   MARK, no FROZEN).
    //   FROZEN_BIT    = 2 (links only): the node has been, or is about to be,
    //                   copied into a child bucket by split_bucket(); it can
    //                   never be marked afterwards. Set with one CAS whose
    //                   expected value is the LIVE link. MARK and FROZEN are
    //                   therefore mutually exclusive and both terminal.
    //   LEVEL_MASK    = 0x3F << 58 (heads only): the seal level, log2 of the
    //                   largest table size for which a child split has SEALED
    //                   this chain (the snapshot that follows the seal may not
    //                   have been taken yet); 0 = never sealed. Levels only
    //                   grow. Six bits hold any log2 of a 64-bit size. A head
    //                   word is never stored into a link without addr_of()
    //                   (see insert()), so a level never reaches a link.
    //
    // GOTCHA that dictates every traversal condition: the LAST node of a chain
    // holds EMPTY in its address field, and once it is tombstoned or frozen its
    // link is EMPTY | MARK_BIT or EMPTY | FROZEN_BIT, which is not EMPTY. A naive
    // `while (curr != EMPTY)` would then keep going and dereference garbage.
    // Every loop therefore tests `addr_of(curr) != EMPTY`, i.e. compares only
    // the address bits, and every dereference goes through node_of(curr).
    using word_t = std::uintptr_t;
    static_assert(sizeof(word_t) == 8, "the word encoding assumes 64-bit pointers");
    static constexpr word_t MARK_BIT = 1;
    static constexpr word_t FROZEN_BIT = 2;
    static constexpr unsigned LEVEL_SHIFT = 58;
    static constexpr word_t LEVEL_MASK = word_t{0x3F} << LEVEL_SHIFT;
    static constexpr word_t PTR_MASK = ~(LEVEL_MASK | word_t{7});
    static constexpr word_t EMPTY = 0;
    static constexpr word_t UNINITIALIZED = 8;
    // Largest arena shard count the constructor accepts (a power of two); more
    // shards than this serve no thread count that exists.
    static constexpr size_t MAX_ARENA_SHARDS = size_t{1} << 16;
    static_assert(PTR_MASK == 0x03FFFFFFFFFFFFF8, "bit diagram above and the masks disagree");
    static_assert(std::atomic<word_t>::is_always_lock_free, "head and link words must be lock-free atomics");

public:
    struct Node {
        // The stored key. Written once at construction, then immutable -- readers
        // compare against it with no synchronization beyond the acquire load of
        // the address that reached this node (see synchronization channel 1).
        T value;
        // Encoded link to the next node in this bucket's chain: the address
        // bits (PTR_MASK) are the successor's address or the EMPTY sentinel; bit
        // 0 (MARK_BIT) is THIS node's tombstone and bit 1 (FROZEN_BIT) says
        // THIS node was superseded by a copy in a child bucket. Atomic because
        // erase() and split_bucket() set those bits via CAS while readers
        // traverse concurrently, and because it is the field that
        // publishes/observes chain structure. The address bits never change
        // once the node is published.
        std::atomic<word_t> link;

        // Default ctor: an unlinked live node whose successor is EMPTY. Rarely
        // used -- the arena is filled via the (val, next) ctor below; this
        // exists only for the container's value-initialization path.
        Node() : value(), link(EMPTY) {}
        Node(const T& val, word_t next) : value(val), link(next) {}
    };
    static_assert(alignof(Node) >= 8, "the low three bits of a node address are the link's tag bits");

private:
    // The dynamically resizable array of atomic bucket heads.
    // Each entry holds the address of the first node in the bucket's chain,
    // plus the bucket's seal level (see the word encoding).
    Container<std::atomic<word_t>> buckets_;

    // The node arena: arena_mask_ + 1 (a power of two) append-only deques. A
    // thread appends to arenas_[thread_number & arena_mask_]. Nodes are appended
    // block-by-block within a shard and never destructed until the set is
    // destroyed. Fixed at construction; the array itself is never resized.
    std::unique_ptr<Container<Node>[]> arenas_;
    size_t arena_mask_;

    // The current logical size (number of buckets) of the hash table. Always a power of 2.
    std::atomic<size_t> table_size_;

    // A SpinLock used to serialize table resizes via the Double-Checked Locking Pattern.
    // Only one thread can expand the buckets_ array at a time.
    SpinLock resize_lock_;

    // Total nodes ever allocated, over-counted by up to 255 per touched shard:
    // a shard adds 256 when the index of a node it appends is a multiple of
    // 256, index 0 included, so a shard's first node counts as 256. The error
    // is one-sided on purpose: an over-count doubles the table early (memory:
    // a set touched by k shards grows to at least 128k buckets, small next to
    // the k node blocks the shards themselves hold), where an under-count
    // would let chains grow long before a doubling. The resize hint in
    // insert() reads it; the exact count is the sum of the shards' sizes
    // (get_internal_node_count()), which is too many acquire loads of
    // frequently written lines to do per insert. On its own cache line: it is
    // written from every shard, and table_size_ is read by every operation.
    alignas(64) std::atomic<size_t> node_count_{0};

    // Seal level stored in the head word `head`.
    static constexpr size_t level_of(word_t head) { return (head & LEVEL_MASK) >> LEVEL_SHIFT; }
    // Seal level that corresponds to the table size `ts` (a power of two): a
    // thread working with table size ts may publish into a bucket only while
    // level_of(head) <= level_for(ts).
    static constexpr size_t level_for(size_t ts) { return static_cast<size_t>(std::countr_zero(ts)); }
    // The address field of a head or link word: the node address, EMPTY or
    // UNINITIALIZED, with the tag bits (MARK/FROZEN of a link, level of a head)
    // cleared. This is what every traversal condition compares.
    static constexpr word_t addr_of(word_t w) { return w & PTR_MASK; }
    // The node a head or link word points to. Only valid when addr_of(w) is a
    // real address, i.e. neither EMPTY nor UNINITIALIZED.
    static Node* node_of(word_t w) { return reinterpret_cast<Node*>(addr_of(w)); }
    // The bare word for a node: its address, no tag bits, level 0.
    static word_t word_of(const Node* n) { return reinterpret_cast<word_t>(n); }

    // Thread-safe bump allocator. It pushes a new node to the calling thread's
    // arena shard and returns the node's address, which is stable for the life
    // of the set. The index emplace_back() returns is the one place a shard is
    // indexed: it is converted to the address here, used to keep node_count_
    // current, and never used again. If the append throws nothing has changed.
    Node* alloc_node(const T& val, word_t next) {
        Container<Node>& shard = arenas_[concurrent_hash_detail::thread_number() & arena_mask_];
        size_t idx = shard.emplace_back(val, next);
        if ((idx & 255) == 0) node_count_.fetch_add(256, std::memory_order_relaxed);
        Node* node = &shard[idx];   // the deque's index is used exactly here, once
        // The address must fit the pointer field of a word: no tag bits (the
        // node is 8-aligned, see the static_assert), no level bits (bits 63..58
        // of a user-space address are zero on every supported platform), and
        // not the UNINITIALIZED sentinel. Checked unconditionally, not by
        // assert(): a platform that tags heap pointers in the high bits (MTE,
        // HWASan) would otherwise fail silently or by livelock in an NDEBUG
        // build, and one AND per allocation is unmeasurable next to the
        // arena lock the allocation just took.
        if ((word_of(node) & ~PTR_MASK) != 0 || word_of(node) == UNINITIALIZED) {
            std::abort();   // the word encoding cannot represent this address
        }
        return node;
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
    // node addresses valid forever (see the class overview).
    void split_bucket(size_t j) {
        // Bucket 0 has no parent and is published EMPTY by the constructor, so
        // it can never be UNINITIALIZED and never reaches here. The guard
        // matters: bit_floor(0) == 0 would make `seal` 64, which a 6-bit level
        // cannot reach, and the seal loop below would not terminate.
        assert(j > 0);
        size_t parent = j - std::bit_floor(j);
        size_t N = std::bit_floor(j);
        size_t mask = (N << 1) - 1;   // == 2N-1, the mask for the table that created j
        word_t parent_head = buckets_[parent].load(std::memory_order_acquire);
        if (addr_of(parent_head) == UNINITIALIZED) {
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
            word_t sealed = (parent_head & ~LEVEL_MASK) | (word_t{seal} << LEVEL_SHIFT);
            if (buckets_[parent].compare_exchange_weak(parent_head, sealed, std::memory_order_release, std::memory_order_acquire)) {
                parent_head = sealed;
            }
        } // seal loop

        // Steps 2 and 3. Build the child subchain privately. It is invisible to
        // every other thread until (and unless) the publishing CAS below
        // succeeds, so no synchronization is needed while constructing it.
        word_t new_subchain_head = EMPTY;
        word_t curr = parent_head;

        while (addr_of(curr) != EMPTY) {   // traverse parent chain
            Node* node = node_of(curr);
            const T& val = node->value;   // immutable once published, never freed: a reference is safe
            word_t next_raw = node->link.load(std::memory_order_acquire);
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
                            if (node->link.compare_exchange_strong(next_raw, next_raw | FROZEN_BIT, std::memory_order_release, std::memory_order_relaxed)) {
                                next_raw |= FROZEN_BIT;
                            }
                        } // if not frozen yet
                    } // if erase() exists
                    if (!(next_raw & MARK_BIT)) {
                        // Prepend a fresh live copy to the child subchain.
                        new_subchain_head = word_of(alloc_node(val, new_subchain_head));
                    }
                } // if key belongs to bucket j
            } // if not tombstoned
            curr = next_raw;
        } // walk parent chain

        // Publish with a single CAS: only if bucket j is still UNINITIALIZED do
        // we install our subchain (release, so a reader that acquires the head
        // sees every node we constructed). The child starts at seal level 0
        // (UNINITIALIZED carries none and new_subchain_head is a bare address or
        // EMPTY). If the CAS fails, another thread already published its own,
        // equivalent split of j; our subchain was never linked anywhere and is
        // simply abandoned -- benign wasted arena space, never reachable. The
        // nodes we froze stay frozen, which is correct: the winner copied them.
        word_t expected = UNINITIALIZED;
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
    //
    // arena_shards: number of arena shards, rounded up to a power of two;
    // 0 (the default) means the hardware concurrency, rounded up. As many
    // shards as threads that allocate concurrently makes every shard lock
    // uncontended (see alloc_node()); fewer shards trade contention for
    // memory (an untouched shard costs one small object, a touched one at
    // least a block of nodes).
    ConcurrentResizableHashSet(size_t initial_capacity = 4, size_t arena_shards = 0) {
        if (arena_shards == 0) arena_shards = std::thread::hardware_concurrency();
        if (arena_shards == 0) arena_shards = 1;   // hardware_concurrency() may report 0
        // Cap before rounding: bit_ceil() of a value above 2^63 has no
        // representable result, and no machine has that many threads anyway.
        if (arena_shards > MAX_ARENA_SHARDS) arena_shards = MAX_ARENA_SHARDS;
        arena_shards = std::bit_ceil(arena_shards);
        arenas_ = std::make_unique<Container<Node>[]>(arena_shards);
        arena_mask_ = arena_shards - 1;
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
    // allocates under its arena shard's lock -- if it lands on an UNINITIALIZED bucket,
    // so it is not lock-free/wait-free in general (see the class overview).
    // Logically deleted nodes are skipped via the MARK_BIT test. The head's seal
    // level and a node's FROZEN bit are deliberately IGNORED here (they are only
    // masked off the address): contains() publishes nothing, so a stale geometry
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
            word_t head = addr_of(buckets_[j].load(std::memory_order_acquire));
            if (head == UNINITIALIZED) {
                split_bucket(j);
                head = addr_of(buckets_[j].load(std::memory_order_acquire));
            }

            bool found = false;
            word_t curr = head;
            while (addr_of(curr) != EMPTY) {   // walk bucket j's chain
                Node* node = node_of(curr);
                // Load the successor link once: it is both the tombstone flag for
                // THIS node and the pointer to the next one, so a single acquire
                // load serves the mark check and the advance.
                word_t check_curr = node->link.load(std::memory_order_acquire);
                if (node->value == key && !(check_curr & MARK_BIT)) {
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
    // under its arena shard's SpinLock (alloc_node()), a bucket that is still
    // UNINITIALIZED is split first, and a successful insert may perform the
    // DCLP-guarded doubling when the node count exceeds twice the table size.
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
        Node* new_node = nullptr;
        while (true) {
            size_t ts = table_size_.load(std::memory_order_acquire);
            size_t j = Hash{}(key) & (ts - 1);
            word_t head = buckets_[j].load(std::memory_order_acquire);
            if (addr_of(head) == UNINITIALIZED) {
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
            word_t curr = addr_of(head);
            while (addr_of(curr) != EMPTY) {   // walk bucket j's chain
                Node* node = node_of(curr);
                word_t check_curr = node->link.load(std::memory_order_acquire);
                if (node->value == key && !(check_curr & MARK_BIT)) {
                    exists = true;
                    break;
                }
                curr = check_curr;
            } // walk chain
            if (exists) {
                /*
                 * ARCHITECTURE NOTE: Rare single-node memory leak
                 * If new_node != nullptr, we allocated a node, failed our CAS, and upon retrying,
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
            // gets the head's ADDRESS only: a level is a head's business and a
            // link's tag bits are its own (they are at the other end of the
            // word, so nothing could alias, but the rule is the same).
            if (new_node == nullptr) {
                new_node = alloc_node(key, addr_of(head));
            } else {
                new_node->link.store(addr_of(head), std::memory_order_relaxed);
            }

            // Publish: prepend by swinging the bucket head from `head` to our
            // node, keeping the bucket's seal level (release). The expected value
            // is the FULL word we validated above, address and level, so the CAS
            // fails if another writer prepended a node OR a splitter sealed the
            // bucket since we read it; either way loop and retry from the
            // table_size_ load, reusing new_node. Failure is relaxed: the
            // returned value is not used.
            if (buckets_[j].compare_exchange_strong(head, word_of(new_node) | (head & LEVEL_MASK), std::memory_order_release, std::memory_order_relaxed)) {
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
                // test `node_count_ > ts*2` is a hint (relaxed, and exact only to
                // within 256 per shard); the decision is remade under resize_lock_
                // against a fresh table_size_ so only ONE thread doubles per epoch
                // (current_ts == ts). NOTE: node_count_ is TOTAL arena occupancy --
                // it counts tombstones, stale split copies and abandoned
                // subchains -- so this is an arena-consumption trigger,
                // not a live-load-factor trigger: a delete-heavy workload grows the
                // table although the live key count does not. New buckets are
                // marked UNINITIALIZED (relaxed) and then table_size_ is released,
                // so any thread that later acquires the new size is guaranteed to
                // observe those markers (channel 2).
                if (node_count_.load(std::memory_order_relaxed) > ts*2) {
                    std::lock_guard lock(resize_lock_);
                    size_t current_ts = table_size_.load(std::memory_order_relaxed);
                    if (current_ts == ts) {
                        size_t new_ts = ts*2;
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
    // its own next link, preserving the successor address. Thereafter contains()
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
            word_t head = addr_of(buckets_[j].load(std::memory_order_acquire));
            if (head == UNINITIALIZED) {
                split_bucket(j);
                head = addr_of(buckets_[j].load(std::memory_order_acquire));
            }

            // Set when this chain proved stale for the key (FROZEN node found):
            // a miss in it is then NOT authoritative even if table_size_ reads
            // unchanged.
            bool stale = false;
            word_t curr = head;
            while (addr_of(curr) != EMPTY) {   // walk bucket j's chain
                Node* node = node_of(curr);
                word_t check_curr = node->link.load(std::memory_order_acquire);
                if (node->value == key && !(check_curr & MARK_BIT)) {
                    // Live (or frozen) node of our key. If live, set MARK_BIT while
                    // keeping the same successor. Success is release: operations
                    // that acquire this link afterwards see the key as deleted.
                    // Failure is acquire: if the reason is FROZEN we go on to
                    // reload table_size_, and the acquire orders that reload after
                    // the freezer's view of the table (see "Retry bound" above).
                    if (!(check_curr & FROZEN_BIT) &&
                        node->link.compare_exchange_strong(check_curr, check_curr | MARK_BIT, std::memory_order_release, std::memory_order_acquire)) {
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

    // Test-only accessor: total nodes ever allocated in the arena (live + dead),
    // exact: the sum of the shards' sizes. Used by InsertContention_NoMemoryLeak
    // to detect the CAS-retry leak, since a leak inflates this count far above
    // the number of distinct keys inserted.
    size_t get_internal_node_count() const {
        size_t n = 0;
        for (size_t i = 0; i <= arena_mask_; ++i) n += arenas_[i].size();
        return n;
    }
    // Test-only accessor: the number of arena shards (a power of two).
    size_t get_internal_arena_shards() const { return arena_mask_ + 1; }
}; // class ConcurrentResizableHashSet

#endif // CONCURRENT_HASH_SET_H
