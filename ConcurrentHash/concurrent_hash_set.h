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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <bit>
#include <vector>
#include <memory>
#include "concurrent_deque.h"
#include <thread>
#include <mutex>
#include "spinlock.h"

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

// ===========================================================================
// ConcurrentResizableHashSet -- design overview (read this before the code).
//
// A closed-addressing (chained) hash set that grows by doubling, without ever
// stopping the world and without a global rehash pass. The two big ideas are
// (1) an append-only node arena in which nothing ever moves, and (2) lazy, cooperative,
// per-bucket splitting that copies (never moves) live nodes into new buckets.
//
// STORAGE
//   shards_  : the node arena, SHARDED: an array of Shard, each an append-only
//              deque of Node (ConcurrentAppendDeque) plus that shard's FREE
//              LIST and LIMBO LIST heads (see RECLAMATION below). A thread
//              always allocates from the shard its thread number selects (see
//              alloc_node()). A node is never moved and never returned to the
//              system; between two quiescent points (see reclaim()) it is,
//              apart from the two state bits of its link, never mutated once
//              published. Nodes are addressed by POINTER: a deque is segmented
//              and its blocks never move, so a node's address is stable for the
//              life of the set, and nothing that reads the structure knows or
//              cares which shard a node lives in. The deque's index is used
//              once per node, to obtain that address at allocation, and by
//              the diagnostic accounting sweep (get_internal_accounting()) at
//              a quiescent point; no read path ever goes through a deque.
//              Sharding exists because one shared per-node atomic of any kind
//              (a lock, a counter) is the insert ceiling; a shard's lock and
//              free-list head are contended only by the threads whose numbers
//              collide modulo the shard count.
//   node_count_ : arena occupancy, OVER-counted by up to 255 per touched shard
//              per allocation path (each shard adds 256 when it appends a node
//              whose index is a multiple of 256, index 0 included, and 256 on
//              every 256th free-list pop, the first pop after a reclaim()
//              included); the resize hint. reclaim() resets it to the live
//              count plus the credit the append path relies on (see there).
//              A reclaim() that throws leaves the previous value, which stays
//              an over-count but is no longer held to that bound (EXCEPTIONS).
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
//       The two bits are mutually exclusive and each is terminal for the
//       node's current life: a link goes live -> MARKED or live -> FROZEN, by
//       one CAS, and never changes again until reclaim() takes the node out of
//       its chain and a later allocation reuses it, at which point the link is
//       rewritten while the node is unreachable (see RECLAMATION).
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
//   Because nodes are copied and no concurrent operation ever unlinks one, a
//   node address observed by any traversing thread stays valid for as long as
//   that thread's operation runs. That is what lets the read path dereference
//   node pointers with no hazard pointers, no reference counts, and no
//   reclamation protocol at all -- a dead node's memory becomes reusable only
//   at a QUIESCENT POINT, when the caller guarantees that no operation is in
//   progress (reclaim(), below, puts it on a free list, from which later
//   allocations, by concurrent inserts and splits, take it), and is never
//   returned to the system before the set is destroyed. The nodes that go
//   dead between two quiescent points are (a) tombstoned nodes
//   (AllowDelete), (b) stale parent copies after a split, (c) speculative split
//   subchains that lost their publishing CAS or were abandoned by a throw
//   mid-build, (d) a node insert() prepared and never published (a retry found
//   the key published by another thread, or threw), and (e) a free-list node
//   whose value assignment threw. (a) and (b) stay reachable until reclaim()
//   unlinks them; (c), (d) and (e) were never published and are recorded on a
//   LIMBO list before the operation that made them returns or throws, because
//   nothing else could ever find them again.
//
// EXCEPTIONS: WHAT IS AND IS NOT HANDLED (the one place that says it all)
//   Exception behavior is mostly out of scope for this class, as it is for
//   the arena: concurrent_deque.h declares exception safety out of scope in
//   the comment on its resize(), and its reallocate_directory() can leak a
//   retired directory if the vector push after the directory swap throws.
//   Hash{} is expected not to throw; where it can, the limbo pushes below
//   apply to it as to any other throw. What CAN throw here is the user's
//   code -- T's copy constructor, copy assignment and operator==, and
//   Hash{} -- and allocation (bad_alloc): the deques' storage, the
//   constructor's shard array, and the vectors of get_internal_accounting();
//   nothing else in this class throws. An arena address the word encoding
//   cannot represent is not an exception either: alloc_node() calls
//   std::abort(), as the constructor does on an initial capacity above 2^63.
//   The one deliberate guarantee: NO ARENA SLOT IS LOST TO A THROW. Every
//   private node an operation holds when an exception passes through it is
//   pushed to limbo before the exception leaves, so get_internal_accounting()
//   stays consistent and reclaim() recycles the node. Per operation:
//   - T (template parameter): must be copy-assignable, because a node reused
//     after a reclaim() takes its new value by assignment; a throwing
//     assignment must leave the old object valid, as every standard type's
//     does.
//   - alloc_node(): if T's copy assignment throws on a free-list node, the
//     node (still holding its old, valid value) goes to the shard's limbo
//     list -- the limbo push writes the link the caller's `next` would have
//     taken -- and the exception propagates. If the append throws, nothing
//     observable in the arena has changed (the deque's own heap ownership on
//     that path is its business, see above; a copy constructor that throws
//     inside emplace_back() needs no handling either: the deque publishes
//     its new size only after the construction, so nothing was appended and
//     the next append reuses the slot).
//   - split_bucket(): if alloc_node() or Hash{} throws while the private
//     subchain is being built (step 2), the copies made so far are
//     referenced by nothing but that frame; the catch around the walk pushes
//     the partial subchain to limbo, whole (its tail is remembered for this
//     as for a lost CAS), and rethrows. The call does not publish j (a
//     parent split it ran first stays published): bucket j is still
//     UNINITIALIZED unless another splitter of j has published it meanwhile.
//     The nodes this call froze stay FROZEN; while j is unpublished they
//     remain authoritative for their keys (INVARIANT), and the next operation
//     that needs j splits it again from them, so the work is redone, not
//     lost; once a competitor publishes j, its copies take over, exactly as
//     after a lost publishing CAS.
//   - insert(): a throw before the publishing CAS -- from alloc_node(),
//     Hash{}, T's operator==, or a split_bucket() met on any attempt --
//     leaves the key as it was; the catch around the retry loop pushes the
//     pending node, if one was allocated on an earlier attempt, to limbo and
//     rethrows. new_node is nulled the moment its CAS publishes it, so that
//     catch never pushes a published node. A throw from the doubling AFTER
//     the publishing CAS succeeded leaves the key a member although the
//     caller sees an exception and no return value; nothing is said about the
//     state after a throw from inside buckets_.resize().
//   - contains(), erase(): hold no private node; they throw only what
//     Hash{}, T's operator== or a split_bucket() they fall into throws, and
//     change no membership when they do. erase() can throw only before its
//     marking CAS: a successful mark is followed by nothing but the return.
//   - reclaim(): throws only if Hash{}(key) throws. The set is then
//     consistent: every node unlinked so far is already on a free list, since
//     each node is pushed as it is freed, not collected first, and
//     node_count_ keeps its previous value, an over-count, so growth stays
//     early, never late. reclaim() may be called again.
//
// RECLAMATION: FREE LISTS, LIMBO LISTS, reclaim()
//   Every shard has two intrusive singly linked lists threaded through the
//   nodes' `link` words: a FREE LIST of nodes available for reuse and a LIMBO
//   LIST of unreachable nodes waiting for the next reclaim(). The next pointer
//   lives in the atomic link word and never in the T bytes: a thread that
//   loses a pop race may still read the node's link while the winner is
//   already assigning the node's value (see alloc_node()).
//   - POP (alloc_node()): a lock-free CAS on the calling thread's own shard's
//     free head. The head word packs the top node's address with a 9-bit POP
//     COUNTER (see the free-list constants), so the CAS that pops a node also
//     counts it, and only the winner's count lands: every 256th pop, the first
//     pop after a reclaim() included, adds 256 to node_count_, mirroring the
//     append path's batching. Empty list: append to the deque.
//   - PUSH onto a free list happens ONLY inside reclaim(), i.e. never
//     concurrently with a pop. That is the whole ABA and reclamation argument
//     for the free list: a head value (address, counter) that a popper has read
//     can be restored only by a push, so a CAS that finds the head unchanged
//     really did pop the node it read; and a popped node cannot be observed
//     through the free list by anyone else, because the address it carried is
//     never pushed again before the next quiescent point. No Harris-style
//     marking is needed for the same reason: there is no concurrent unlink.
//   - PUSH onto a limbo list is a Treiber push (CAS) on the pushing thread's
//     own shard; the list is push-only until reclaim() drains it, so it is
//     ABA-free too, and nothing but reclaim() and the diagnostic accounting
//     sweep ever read it.
//   - reclaim() (contract at its definition) walks every published bucket,
//     unlinks every dead node, deals the dead and the limbo nodes round-robin
//     over the shards' free lists, and resets the pop counters and
//     node_count_. It takes no lock: its precondition is that the caller has
//     made every earlier operation happen-before it and it happens-before every
//     later one, which is what "quiescent" means here. Under that precondition
//     it needs no atomic ordering at all (all of its atomic accesses are
//     relaxed): every write it makes is visible to every later operation by
//     happens-before, exactly as the constructor's writes are.
//   - PERIODS. reclaim() calls divide the set's life into PERIODS (the first
//     starts at construction, the last ends at destruction). The CLIENT's
//     view: the only thing contains(), insert() and erase() can observe
//     across a reclaim() is MEMBERSHIP -- a value in the set before it is in
//     the set after it, a value not in it still is not (the get_internal_*
//     diagnostics also see the arena, which reclaim() does change). The
//     client holds no references into the table (the API hands out none), so
//     nothing of the client's can outlive a period, and no ABA problem can
//     reach the client. The INTERNAL view: slots never move (a slot's address
//     is stable for the life of the set, see STORAGE), doublings are never
//     undone, published buckets stay published with their seal levels, and
//     kept links keep their tag bits.
//     What reclaim() changes is which node, holding which value, occupies a
//     slot; the guarantees that are period-bounded are exactly these: a slot
//     hosts at most one NODE LIFE per period (freed by one reclaim(),
//     popped and published at most once before the next), and a link's
//     MARK/FROZEN bits are terminal per node life.
//   - ADDRESS REUSE and the publishing CAS. insert()'s CAS on a bucket head and
//     erase()'s CAS on a link are safe against ABA because within one period
//     no node address is published twice: a node leaves a chain only through
//     reclaim(), and re-enters one only through a pop that follows a reclaim().
//     An address a thread observed in a head or link therefore names the same
//     node, in the same chain, for the whole of that thread's operation. No
//     CAS of any operation, and no pop of a free list, can span a reclaim():
//     nothing else runs on the table while it does. There is therefore no ABA
//     problem the client can get into, and none in the free lists.
//   - LIFETIME OF VALUES. A free node keeps its old, constructed T; reclaim()
//     destroys nothing. Reuse COPY-ASSIGNS the new value over the old one, so
//     the old value's resources are released at reuse (the assignment's
//     left-hand side), not at erase() and not at reclaim(); values that are
//     never reused are destroyed with the set. Callers must not rely on
//     destructor timing. Chosen over destroy-at-reclaim + construct-at-reuse
//     (the deque destroys every slot at the end, so a destroyed slot would have
//     to be re-constructed at once) and over raw-storage slots (saves only
//     destructor calls, costs a non-destroying container).
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
//   successful marking CAS is THE deletion of it: at most one insert() returns
//   true per absent->present transition -- exactly one, unless the call whose
//   CAS made the transition then exits by an exception (the doubling that
//   follows the CAS can throw, see EXCEPTIONS) -- and exactly one erase()
//   returns true per present->absent transition (nothing follows its marking
//   CAS but the return).
//   INVARIANT (key authority, AllowDelete == true): for every key, at any
//   instant, at most one node reachable from a published bucket head is live
//   (neither MARKED nor FROZEN). The key is in the set iff such a node exists,
//   or a FROZEN node for it exists whose child bucket is not yet published (any
//   operation that needs the child publishes it first, see split_bucket()).
//   Authority passes from a node to its single published copy; a tombstone ends
//   the lineage, because MARKED nodes are never copied.
//   With AllowDelete == false the freeze is compiled out: a moved key then has
//   an unmarked node in the parent AND its copy in the child (until reclaim()
//   unlinks the parent's node, see is_dead()). Nothing is ever tombstoned,
//   so both answer "present" and the weaker invariant "every reachable node
//   for a key agrees" is all that is needed.
//   LINEARIZABILITY of a miss, in two halves.
//   (i) A miss is linearized before every insert whose publishing CAS does
//       NOT happen-before the call: no mechanism is needed, and no test may
//       assert a hit for a key unless the insert's publishing CAS -- not
//       merely its invocation -- happens-before the call, through the
//       inserter's return or through any call that observed the key and
//       then synchronized with the caller.
//   (ii) For an insert whose publishing CAS DOES happen-before the call, a
//       miss is final only if the key was deleted. The call's walk runs at a
//       table size no smaller than the insert's (the inserter acquired its
//       size before its CAS, so by read-read coherence the call's load
//       returns that size or a larger one); no concurrent operation takes a
//       node out of a chain, reclaim() takes out only MARKED nodes and nodes
//       a split has copied into a published child (is_dead()), and a split
//       copies every node of the key that it does not find MARKED (step 2 of
//       split_bucket()). So the walk meets the inserted node, or its copy in
//       the call's bucket, unless that node or a copy of it was MARKED; and
//       if the key's node was MARKED, the deletion linearizes at the mark
//       CAS (the marking CAS is THE deletion, see above), and contains() and
//       erase() skip a MARKED node anyway.
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
//      On a concurrent path only alloc_node() indexes the arena, with the
//      index its own emplace_back() returned, and it does so once, to take
//      the node's address. From then on the node is reached by pointer only:
//      the thread that constructed it hands the address to readers through
//      channel 1 -- a release CAS on a bucket head, read with acquire (a node
//      reached through a link was published by the head CAS of the node that
//      links to it, or of a later prepend) -- and that acquire is what makes
//      the node's construction visible to the reader. A node popped from a
//      free list is handed over the same way: the popper assigns its value
//      and link while the node is reachable by nobody (see RECLAMATION), then
//      publishes it through channel 1. The only other indexing of a shard is
//      by the diagnostic accounting sweep, at a quiescent point, over
//      [0, size()).
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
//      by a CONCURRENT operation would break this argument: never add one.
//      reclaim()'s plain stores to heads (and to links) do not, because they
//      happen at a quiescent point: every seal before them already
//      happens-before every operation after them through the caller's
//      synchronization, so no later thread needs the release sequence the
//      store cuts short, and every seal after them heads a new release
//      sequence that only read-modify-writes extend, as before. (Links are
//      safe on the same two counts: between quiescent points nothing is
//      written to a link after MARK or FROZEN, and reclaim()'s link stores
//      are quiescent.)
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
//   shape a post-CAS recheck would have (x = head or link, y = table_size_)
//   and which acquire/release on two DIFFERENT atomics does not order.
//   What remains cross-variable is only the read-side chain of channel 4
//   (acquire word -> acquire table_size_). It is used for PROGRESS: a thread
//   that meets a seal level above its own, or a FROZEN node for its key,
//   reloads table_size_ and retries, and the chain says the reload returns a
//   larger size at once. Were that edge missing the thread would re-read until
//   the larger size became visible; it would never return a wrong answer.
//   A miss that is confirmed by an unchanged table_size_ (contains();
//   erase(), whose walk must also have found no FROZEN node of the key)
//   linearizes at a point inside the call; the two halves under STALE
//   GEOMETRY are the argument that such a point exists, and this comment does
//   not name it.
//
// PROGRESS: the hash's own protocol takes no lock. The walks, the splits,
// the publishing CAS and every retry loop are lock-free, and the pure
// read/traverse path writes nothing; none of it is wait-free. The class
// takes a lock in only two places, both on the allocation side, and only an
// operation that allocates can reach either, so an operation that allocates
// nothing is lock-free end to end:
//   (1) alloc_node(): a lock-free pop from the calling thread's arena shard's
//       free list when it has a node, else an append under that shard's
//       deque SpinLock; with one thread per shard that lock is uncontended,
//       but it is a lock. An insert() whose walk misses its key allocates
//       this way, and so does split_bucket() for its copies, so contains(),
//       insert() and erase() can all reach the lock by meeting an
//       UNINITIALIZED bucket.
//   (2) resize_lock_: taken by an insert() after its own publishing CAS,
//       when node_count_ exceeds twice the table size it inserted at. If
//       table_size_ is still that size, the holder grows buckets_ (an
//       allocation, under the bucket deque's own SpinLock, which after the
//       constructor only the holder of resize_lock_ takes) and marks the new
//       buckets UNINITIALIZED; otherwise another insert() has doubled first,
//       and the holder releases the lock having allocated nothing. Only
//       another insert() past its own threshold waits on it; no other
//       operation takes it.
// The locks belong to the allocation, not to the hash: an allocator that
// allocates blocks under a lock and hands nodes over lock-free leaves the
// lock with whichever thread allocates the block. reclaim() is not a
// concurrent operation at all (see its contract).
//
// Template parameters:
//   T           : element (key) type; must be copy-constructible (a new node
//                 is constructed from the key), copy-assignable (a node
//                 reused after a reclaim() takes its new value by
//                 assignment; see EXCEPTIONS for what a throwing assignment
//                 must satisfy), and equality-comparable as `value == key`,
//                 a stored T against a const T&, consistently with Hash:
//                 keys that compare equal must hash equal, since an
//                 operation looks for its key only in the bucket the key's
//                 own hash selects.
//   AllowDelete : when true, compiles erase() (tombstone deletion) and the
//                 freeze step of split_bucket(). When false, no node is ever
//                 marked or frozen; the MARK checks that remain (the walks,
//                 split_bucket(), reclaim()) are tests of a bit that is never
//                 set. Between quiescent points a published chain then
//                 changes only by prepends; reclaim() unlinks, for both
//                 values, the parent nodes that a published child supersedes
//                 (is_dead(), rule (2)).
//   Hash        : hash functor; must be default-constructible, with Hash{}(k)
//                 well-formed for a const T& k and returning a size_t, the
//                 SAME value for a key every call (the split math re-hashes
//                 keys under wider masks). No Hash object is stored: every
//                 hash is computed by a freshly constructed Hash{}, so a Hash
//                 with state only ever has its default state. Besides the key
//                 of every call, Hash{} is applied to stored values: by a
//                 split, to every node of the parent snapshot that is not
//                 MARKED (to select the nodes that move), and by reclaim(),
//                 to every reachable node that is not MARKED (to decide
//                 whether it is dead, see is_dead()).
// ===========================================================================
template <
    typename T,
    bool AllowDelete = false,
    typename Hash = std::hash<T>
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
    //                   therefore mutually exclusive and both terminal, until
    //                   reclaim() unlinks the node and a pop reuses it (a
    //                   free node's link is a bare address, no tag bits).
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
    // Elements per block of each deque: buckets_ and every Shard::nodes.
    // Larger blocks allocate less often (one block allocation, under the
    // deque's lock, per ARENA_BLOCK elements; the deque also reallocates its
    // block directory under that lock, a number of times logarithmic in the
    // block count); smaller ones leave less memory unused, since a deque's
    // storage comes in whole blocks: buckets_ is rounded up to whole blocks,
    // and every shard a thread has allocated from holds at least one.
    static constexpr size_t ARENA_BLOCK = 1024;
    static_assert(PTR_MASK == 0x03FFFFFFFFFFFFF8, "bit diagram above and the masks disagree");
    static_assert(std::atomic<word_t>::is_always_lock_free, "head and link words must be lock-free atomics");

    // FREE-LIST HEAD encoding (Shard::free_head). The word packs the address of
    // the top free node with the shard's POP COUNTER, so that the one CAS that
    // pops a node also counts the pop:
    //
    //   bit     63..55        54..0
    //           pop counter   top node address >> 3   (0 = empty list)
    //
    // The address occupies 58 bits at most and is 8-aligned (alloc_node()
    // checks both), so address >> 3 fits 55 bits and the top nine are the
    // counter. One pop adds FREE_POP_ONE; the carry out of bit 63 is lost,
    // which is the wrap of a 9-bit counter. Every 256th pop since the last
    // reclaim() adds 256 to node_count_: the pop is the 256th exactly when
    // the counter it replaces has its low eight bits clear (FREE_BATCH_MASK),
    // and reclaim() clears the counter, so the first pop after it counts.
    // The obvious alternative is not cheaper: with the counter in the low bits
    // (address << 5 | counter) the carry of the increment has to be masked off
    // the address, which costs the AND that this layout spends on the decode
    // shift. Encode: (address >> 3) | counter bits; decode: (word << 9) >> 6,
    // two shifts, no mask.
    static constexpr unsigned FREE_COUNT_SHIFT = 55;
    static constexpr word_t FREE_POP_ONE = word_t{1} << FREE_COUNT_SHIFT;
    static constexpr word_t FREE_COUNT_MASK = ~(FREE_POP_ONE - 1);       // bits 63..55
    static constexpr word_t FREE_BATCH_MASK = word_t{255} << FREE_COUNT_SHIFT;   // bits 62..55
    static constexpr word_t FREE_EMPTY = 0;
    static_assert((PTR_MASK >> 3) < FREE_POP_ONE, "an address >> 3 must fit below the pop counter");

public:
    struct Node {
        // The stored key. Written at construction, or by copy assignment when
        // a free node is reused, and immutable while the node is reachable --
        // readers compare against it with no synchronization beyond the
        // acquire load of the address that reached this node (see
        // synchronization channel 1). A free node keeps its old value until
        // reuse (see RECLAMATION).
        T value;
        // Encoded link to the next node in this bucket's chain: the address
        // bits (PTR_MASK) are the successor's address or the EMPTY sentinel; bit
        // 0 (MARK_BIT) is THIS node's tombstone and bit 1 (FROZEN_BIT) says
        // THIS node was superseded by a copy in a child bucket. Atomic because
        // erase() and split_bucket() set those bits via CAS while readers
        // traverse concurrently, and because it is the field that
        // publishes/observes chain structure. The address bits never change
        // while the node is published; between quiescent points a published
        // link changes only by gaining MARK or FROZEN. On a free or limbo list
        // the same word is the list's next pointer (a bare address).
        std::atomic<word_t> link;

        Node(const T& val, word_t next) : value(val), link(next) {}
    }; // struct Node
    static_assert(alignof(Node) >= 8, "the low three bits of a node address are the link's tag bits");

private:
    // The dynamically resizable array of atomic bucket heads.
    // Each entry holds the address of the first node in the bucket's chain,
    // plus the bucket's seal level (see the word encoding).
    ConcurrentAppendDeque<std::atomic<word_t>, ARENA_BLOCK> buckets_;

    // One arena shard: the append-only node deque and the heads of the shard's
    // free and limbo lists (see RECLAMATION in the class overview). The two
    // heads share a cache line of their own, apart from the deque's members
    // (which the deque's default build pads onto lines of their own): the
    // free head is written by every pop of the shard's threads, and with more
    // threads than shards a pop CAS and the deque's lock exchange would
    // otherwise fight over one line; the limbo head is written rarely (lost
    // splits, orphans, throwing assignments) and costs nothing on the free
    // head's line.
    struct Shard {
        // Nodes are appended block-by-block and never destructed until the set
        // is destroyed; a slot's address is stable for the life of the set.
        ConcurrentAppendDeque<Node, ARENA_BLOCK> nodes;
        // Free list: the encoded top node address and pop counter (see the
        // FREE-LIST HEAD encoding). Popped by alloc_node() with a relaxed CAS;
        // pushed and reset by reclaim() only.
        alignas(64) std::atomic<word_t> free_head{FREE_EMPTY};
        // Limbo list: the bare address of the top node, or EMPTY; the nodes
        // link through their `link` words. Pushed with a relaxed CAS by the
        // shard's threads, drained by reclaim(), read by the accounting sweep.
        std::atomic<word_t> limbo_head{EMPTY};
    }; // struct Shard

    // The node arena: arena_mask_ + 1 (a power of two) shards. A thread
    // allocates from shards_[thread_number & arena_mask_]. Fixed at
    // construction; the array itself is never resized.
    std::unique_ptr<Shard[]> shards_;
    size_t arena_mask_;

    // The current logical size (number of buckets) of the hash table. Always a power of 2.
    std::atomic<size_t> table_size_;

    // A SpinLock used to serialize table resizes via the Double-Checked Locking Pattern.
    // Only one thread can expand the buckets_ array at a time. Taken only by
    // an insert() after its publishing CAS, when the node count crosses the
    // doubling threshold; no other operation waits on it (see PROGRESS).
    SpinLock resize_lock_;

    // Arena occupancy, over-counted by up to 255 per touched shard and
    // allocation path: a shard adds 256 when the index of a node it appends
    // is a multiple of 256, index 0 included, so a shard's first node counts
    // as 256; and 256 on every 256th free-list pop since the last reclaim(),
    // the first pop included (see the FREE-LIST HEAD encoding). The error is
    // one-sided on purpose: an over-count doubles the table early (memory: a
    // set touched by k shards grows to at least 128k buckets, small next to
    // the k node blocks the shards themselves hold), where an under-count
    // would let chains grow long before a doubling. reclaim() resets it to
    // the live node count plus, per shard, the appends that remain until the
    // shard's next multiple of 256 (its "credit"), so the append path stays
    // one-sided after a reset: without the credit a shard of size 100 would
    // append 156 uncounted nodes before its next batch. A reclaim() that
    // throws leaves the count as it was: still an over-count, but no longer
    // held to the bound above (EXCEPTIONS). The resize hint in insert()
    // reads it; the exact count is the sum of the shards' sizes
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
    // The address field of a free-list head word (see the FREE-LIST HEAD
    // encoding): the top node's address, or FREE_EMPTY.
    static constexpr word_t free_addr_of(word_t head) { return (head << 9) >> 6; }
    // The free-list head word for top node address `addr` (a bare node
    // address or EMPTY) with the counter bits of `count_bits` (a head word
    // whose address field is ignored).
    static constexpr word_t free_word(word_t addr, word_t count_bits) { return (addr >> 3) | (count_bits & FREE_COUNT_MASK); }

    // The arena shard the calling thread allocates from and pushes limbo onto.
    Shard& my_shard() { return shards_[concurrent_hash_detail::thread_number() & arena_mask_]; }

    // Push the private chain first..last (linked through their link words,
    // nothing else references them) onto `shard`'s limbo list, where
    // reclaim() will find it. Treiber push; the list is push-only until a
    // quiescent point, so a CAS that finds the head unchanged is safe (no
    // ABA: nothing pops). Relaxed throughout: the pushers exchange nothing
    // but the head word itself (each links its own private tail), and the
    // only readers of the list, reclaim() and the accounting sweep, are
    // ordered after every push by the quiescence precondition. Weak CAS: a
    // retry loop anyway.
    static void push_limbo(Shard& shard, Node* first, Node* last) {
        word_t old_head = shard.limbo_head.load(std::memory_order_relaxed);
        do {
            last->link.store(old_head, std::memory_order_relaxed);
        } while (!shard.limbo_head.compare_exchange_weak(old_head, word_of(first), std::memory_order_relaxed, std::memory_order_relaxed));
    } // push_limbo()

    // Thread-safe node allocator: returns a node holding `val` with link
    // `next`, private to the caller until it publishes the node, at an
    // address that is stable for the life of the set. The node comes from
    // the calling thread's shard: from its free list when the list has a
    // node, else appended to its deque. Exceptions: see EXCEPTIONS in the
    // class overview.
    //
    // The pop is a lock-free CAS on the free head, and every ordering in it
    // is relaxed. Why that suffices (see RECLAMATION in the class overview):
    // the free list is written by reclaim() alone, at a quiescent point, so
    // everything reclaim() wrote (the head, every free node's link) is visible
    // to this thread by happens-before, not by any atomic ordering; and the
    // only concurrent writers of the head are other poppers of this shard,
    // from which nothing needs to be received. A popper that reads the head
    // (X, c) and then X's link may lose to another popper of X, who then
    // rewrites X's link (and value) for its own use; the loser's `successor`
    // is then garbage, but its CAS fails regardless, because the head is no
    // longer (X, c): X can only return to the head through a push, and pushes
    // happen only in reclaim(). A CAS that succeeds therefore read X's link
    // as reclaim() left it, i.e. the true successor. Weak CAS: we loop anyway.
    //
    // The successful CAS also increments the counter packed in the head word
    // (it replaces the whole word), and only the winner's increment lands, so
    // the count of pops since the last reclaim() is exact; the winner whose
    // pop opens a batch of 256 (the counter it replaced had its low eight
    // bits clear, the first pop after a reclaim() included) adds 256 to
    // node_count_: the same one-sided batching as the append path.
    //
    // The popped node is private from the moment the CAS succeeds (no chain
    // reaches it, and no other popper can obtain it: see above), so its value
    // is copy-assigned and its link stored relaxed, exactly as insert()
    // repoints a node it has not published yet; the publishing release CAS
    // on a bucket head is what hands both to readers (channel 1). If the
    // assignment throws, the catch below sends the node to limbo (see
    // EXCEPTIONS in the class overview). Note the loser of
    // the pop race may still read this node's link (relaxed, harmless: see
    // above) while the value is being assigned; that is why a list's next
    // pointer lives in the atomic link and never in the T bytes.
    Node* alloc_node(const T& val, word_t next) {
        Shard& shard = my_shard();
        word_t head = shard.free_head.load(std::memory_order_relaxed);
        while (free_addr_of(head) != FREE_EMPTY) {   // free-list pop
            Node* node = node_of(free_addr_of(head));
            word_t successor = node->link.load(std::memory_order_relaxed);
            if (shard.free_head.compare_exchange_weak(head, free_word(successor, head + FREE_POP_ONE), std::memory_order_relaxed, std::memory_order_relaxed)) {
                if ((head & FREE_BATCH_MASK) == 0) node_count_.fetch_add(256, std::memory_order_relaxed);
                try {
                    node->value = val;
                } catch (...) {
                    push_limbo(shard, node, node);
                    throw;
                }
                node->link.store(next, std::memory_order_relaxed);
                return node;
            } // if this thread popped the node
        } // free-list pop
        size_t idx = shard.nodes.emplace_back(val, next);
        if ((idx & 255) == 0) node_count_.fetch_add(256, std::memory_order_relaxed);
        Node* node = &shard.nodes[idx];   // the deque's index is used here, once, and by the accounting sweep
        // The address must fit the pointer field of a word: no tag bits (the
        // node is 8-aligned, see the static_assert), no level bits (bits 63..58
        // of a user-space address are zero on every supported platform), and
        // not the UNINITIALIZED sentinel. Checked unconditionally, not by
        // assert(): a platform that tags heap pointers in the high bits (MTE,
        // HWASan) would otherwise fail silently or by livelock in an NDEBUG
        // build, and one AND per allocation is unmeasurable next to the
        // arena lock the allocation just took. A node passes this check once,
        // when it is appended; every free-list node once passed it.
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
    //   3. Publish the subchain with one CAS on bucket j's head. A subchain
    //      that loses the CAS goes to the limbo list, whole.
    // Exceptions: see EXCEPTIONS in the class overview.
    // The parent chain is never relinked by a concurrent operation -- this is
    // what keeps already-observed node addresses valid for the whole of an
    // operation (see the class overview); only reclaim() unlinks, at a
    // quiescent point.
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
        } // if the parent is not split yet

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
        // The subchain grows by prepending, so its tail is the first node
        // allocated; the tail is remembered for the limbo push of a lost CAS
        // or of a throw (the catch below; see EXCEPTIONS in the class
        // overview).
        word_t new_subchain_head = EMPTY;
        Node* new_subchain_tail = nullptr;
        word_t curr = parent_head;

        try {
            while (addr_of(curr) != EMPTY) {   // traverse parent chain
                Node* node = node_of(curr);
                // Immutable while reachable, and reachable nodes are unlinked only
                // at a quiescent point, never during this call: a reference is safe.
                const T& val = node->value;
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
                            Node* copy = alloc_node(val, new_subchain_head);
                            if (new_subchain_tail == nullptr) new_subchain_tail = copy;
                            new_subchain_head = word_of(copy);
                        } // if still live: copy it
                    } // if key belongs to bucket j
                } // if not tombstoned
                curr = next_raw;
            } // walk parent chain
        } catch (...) {
            if (new_subchain_head != EMPTY) push_limbo(my_shard(), node_of(new_subchain_head), new_subchain_tail);
            throw;
        } // the partial subchain is accounted for before the exception leaves

        // Publish with a single CAS: only if bucket j is still UNINITIALIZED do
        // we install our subchain (release, so a reader that acquires the head
        // sees every node we constructed). The child starts at seal level 0
        // (UNINITIALIZED carries none and new_subchain_head is a bare address or
        // EMPTY). If the CAS fails, another thread already published its own,
        // equivalent split of j; our subchain was never linked anywhere, so
        // nothing but this thread could ever find it again: it goes, whole,
        // onto our shard's limbo list (one CAS, head to tail), for reclaim()
        // to recycle. The nodes we froze stay frozen, which is correct: the
        // winner copied them. Failure is relaxed: the value that won is not
        // used; our caller re-reads the head with acquire.
        word_t expected = UNINITIALIZED;
        if (buckets_[j].compare_exchange_strong(expected, new_subchain_head, std::memory_order_release, std::memory_order_relaxed)) {
            // CAS succeeded: our subchain is now bucket j's authoritative head.
        } else {
            // CAS failed: subchain abandoned (unpublished, memory waste until reclaim() collects it).
            if (new_subchain_head != EMPTY) push_limbo(my_shard(), node_of(new_subchain_head), new_subchain_tail);
        } // if another thread's split of j won
    } // split_bucket()

public:
    // Initializes the hash set with the given capacity, rounded up to the
    // nearest power of two (the bucket mask math needs only a power of two;
    // the minimum of 4 is a convenience). A capacity above 2^63 is a caller's
    // bug: std::bit_ceil() of it has no representable result, so the
    // constructor calls std::abort(), in every build, before rounding. The
    // buckets are allocated here, eagerly, so a capacity beyond memory throws
    // bad_alloc from the constructor. Every initial bucket starts EMPTY (not
    // UNINITIALIZED) at seal level 0: the original buckets have no parent to
    // split from. table_size_ is released LAST so that any thread which later
    // acquires it is guaranteed to see the fully initialized bucket array.
    //
    // arena_shards: number of arena shards, capped at MAX_ARENA_SHARDS (2^16)
    // and then rounded up to a power of two; 0 (the default) means the
    // hardware concurrency (1 if it is unknown), capped and rounded up the
    // same way. As many shards as threads that allocate concurrently makes
    // every shard lock uncontended (see alloc_node()); fewer shards trade
    // contention for memory (an untouched shard costs one small object, a
    // touched one at least a block of nodes).
    //
    // The set is neither copyable nor movable (its deques and atomics are
    // neither). Its destruction is quiescent, as reclaim() is: every call on
    // the set must happen-before the destructor, which destroys every value
    // the arena holds.
    ConcurrentResizableHashSet(size_t initial_capacity = 4, size_t arena_shards = 0) {
        if (arena_shards == 0) arena_shards = std::thread::hardware_concurrency();
        if (arena_shards == 0) arena_shards = 1;   // hardware_concurrency() may report 0
        // Cap before rounding: bit_ceil() of a value above 2^63 has no
        // representable result, and no machine has that many threads anyway.
        if (arena_shards > MAX_ARENA_SHARDS) arena_shards = MAX_ARENA_SHARDS;
        arena_shards = std::bit_ceil(arena_shards);
        shards_ = std::make_unique<Shard[]>(arena_shards);
        arena_mask_ = arena_shards - 1;
        if (initial_capacity < 4) initial_capacity = 4;
        if (initial_capacity > (size_t{1} << 63)) std::abort();   // a caller's bug: bit_ceil() cannot represent the result
        initial_capacity = std::bit_ceil(initial_capacity);
        buckets_.resize(initial_capacity);
        for (size_t i = 0; i < initial_capacity; ++i) {
            buckets_[i].store(EMPTY, std::memory_order_relaxed);
        }
        table_size_.store(initial_capacity, std::memory_order_release);
    } // ConcurrentResizableHashSet()

    // Membership test. Lock-free on the fast path (a plain chain walk with no
    // atomic writes); it can, however, fall into split_bucket() -- which
    // allocates, possibly under its arena shard's lock -- if it lands on an
    // UNINITIALIZED bucket, so it is not lock-free/wait-free in general (see
    // the class overview). For the same reason it is not const: the split
    // writes the set's internals (a seal, freezes when AllowDelete, a
    // publish, allocations), though never its membership.
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
    //     doubling and always terminates. The two halves of the miss argument
    //     under STALE GEOMETRY say when a miss is final.
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
        } // retry loop
    } // contains()

    // Insertion. Returns true iff THIS call made the key a member: at most one
    // insert() returns true per absent->present transition of the key, across
    // any number of concurrent resizes -- exactly one, unless the call that
    // made the transition throws after its publishing CAS (EXCEPTIONS).
    // Returns false only if the key was present at some instant during the
    // call. Not lock-free: every new node is allocated from its arena shard
    // (alloc_node(): a lock-free free-list pop, or an append under the
    // shard's SpinLock), a bucket that is still UNINITIALIZED is split first,
    // and a successful insert may perform the DCLP-guarded doubling when the
    // node count exceeds twice the table size.
    //
    // The decision is the publishing CAS on the bucket head, and the geometry is
    // validated by that same CAS: its expected value includes the head's seal
    // level, which was checked against the table size this attempt works with.
    // See "STALE GEOMETRY" in the class overview and step 1 of split_bucket().
    //
    // Exceptions: see EXCEPTIONS in the class overview.
    bool insert(const T& key) {
        // The node to prepend is allocated once, on the first attempt that
        // needs it, and reused across CAS retries (only its link is repointed):
        // allocating per attempt would consume a node per failed CAS under
        // contention. If a retry finds the key present, the node is unneeded;
        // it goes to the limbo list, since nothing else references it; so does
        // a pending node when a retry throws (the catch below; see EXCEPTIONS
        // in the class overview).
        Node* new_node = nullptr;
        try {
            while (true) {
                size_t ts = table_size_.load(std::memory_order_acquire);
                size_t j = Hash{}(key) & (ts - 1);
                word_t head = buckets_[j].load(std::memory_order_acquire);
                if (addr_of(head) == UNINITIALIZED) {
                    split_bucket(j);
                    continue;   // Retry after split (re-read head, which is now published)
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
                    // A non-null new_node means an earlier attempt allocated it,
                    // lost its CAS, and this retry found the key published by
                    // another thread. The node was never published; it is orphaned
                    // onto our shard's limbo list for reclaim() to recycle.
                    if (new_node != nullptr) push_limbo(my_shard(), new_node, new_node);
                    return false;
                } // if the key is present

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
                    new_node = nullptr;   // published: no longer ours to send to limbo
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
                    // (current_ts == ts). NOTE: node_count_ is arena occupancy since
                    // the last reclaim() (or ever, if none was called) -- it counts
                    // tombstones, stale split copies, lost subchains and orphans
                    // until a reclaim() recycles them -- so this is an
                    // arena-consumption trigger, not a live-load-factor trigger: a
                    // delete-heavy workload without reclaim() calls grows the table
                    // although the live key count does not. New buckets are
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
        } catch (...) {
            if (new_node != nullptr) push_limbo(my_shard(), new_node, new_node);
            throw;
        } // a pending, unpublished node is accounted for before the exception leaves
    } // insert()

    // Tombstone deletion (compiled only when AllowDelete). Finds the live node
    // for `key` and logically deletes it with a single CAS that sets MARK_BIT on
    // its own next link, preserving the successor address. Thereafter contains()
    // skips it and no split ever copies it. Returns true iff THIS call set the
    // tombstone: exactly one erase() returns true per present->absent transition
    // of the key, across any number of concurrent resizes. Returns false only
    // if the key was absent at some instant during the call.
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

    // ------------------------------------------------------------------------
    // Quiescent reclamation. reclaim() and the quiescent-only diagnostics
    // below (the accounting sweep and the counts taken from it; the other
    // diagnostics are race-free, see DIAGNOSTICS) share one precondition,
    // QUIESCENCE: no other call on this set is in progress while the function
    // runs, every earlier call happens-before it, and it happens-before every
    // later call (e.g. the caller joins or barriers the worker threads before
    // and starts or releases them after). Under that precondition every
    // atomic access here is relaxed: the values these functions read were
    // published to them, and the values reclaim() writes are published to
    // every later operation, by the caller's synchronization, exactly as the
    // constructor's writes are.
    // ------------------------------------------------------------------------

private:
    // Push `node` (unreachable, nothing else references it) onto `shard`'s
    // free list, keeping the head's counter bits. Quiescent (reclaim() only).
    static void push_free(Shard& shard, Node* node) {
        word_t head = shard.free_head.load(std::memory_order_relaxed);
        node->link.store(free_addr_of(head), std::memory_order_relaxed);
        shard.free_head.store(free_word(word_of(node), head), std::memory_order_relaxed);
    } // push_free()

    // The dead-node rule of reclaim(), for `node` with link value `link` in
    // the chain of published bucket `j`, at table size `ts`. Dead means: no
    // operation, present or future, can need this node, so it may leave the
    // chain. A node is dead iff
    //   (1) it is MARKED (a tombstone: contains() skips it, no split copies it,
    //       and an insert of the key prepends a fresh node); or
    //   (2) its key's NEXT CHILD bucket on its path out of j is published.
    // Rule (2), derived. Every node in chain j has a key whose hash h agrees
    // with j on j's own bits, h & own_mask == j where own_mask has bit_width(j)
    // bits (it was published into j at a table size whose mask includes those
    // bits, or copied into j by j's split under exactly that mask); the
    // assert() checks it. From j the key can move only when the table doubles
    // past the LOWEST bit of h above own_mask, call it b: at every doubling
    // below 2^b the key stays in j, at 2^(b+1) buckets it moves to child =
    // j | 2^b, and no other child of j ever selects it (a child j + 2^m with m
    // != b needs bit m of h set, which fails for m < b, and bit b clear, which
    // fails for m > b; see split_bucket() step 2). So "the key maps elsewhere
    // under the current mask" is exactly child < ts, and "the next child on
    // its path" is that one bucket. If h has no bit above own_mask the key
    // never leaves j.
    // Why child published is the right test, for both AllowDelete values: the
    // split that published child SEALED j first and snapshotted the sealed
    // head, and an insert of this key into j can only have run at a table
    // size <= 2^b, whose level is below the seal, so it is in the snapshot or
    // it failed. Every splitter of child therefore saw this node: with
    // AllowDelete it froze it and the winner copied it (or found it marked and
    // skipped it), without AllowDelete it copied it unconditionally. Either
    // way the copy in child (or, transitively, in a deeper published bucket)
    // is what every operation at the current table size consults, and this
    // node decides nothing any more. Conversely, if child is still
    // UNINITIALIZED this node is the key's only reachable node and is kept,
    // FROZEN or not: a FROZEN node whose child is unpublished is still
    // authoritative (class overview, INVARIANT; it arises when a split threw
    // between the freeze and the publication), and the next operation that
    // needs child will split it from this node. A deeper bucket on the path
    // cannot be published while child is not (split_bucket() publishes the
    // parent first), so testing the next child alone is sufficient.
    bool is_dead(const Node* node, word_t link, size_t j, size_t ts) const {
        if (link & MARK_BIT) return true;
        const size_t h = Hash{}(node->value);
        const size_t own_mask = (size_t{1} << std::bit_width(j)) - 1;
        assert((h & own_mask) == j && "a node's key does not hash to the chain it is in: Hash is not a function of the key");
        const size_t moving_bits = h & ~own_mask;
        if (moving_bits == 0) return false;   // the key never leaves bucket j
        const size_t child = j | (size_t{1} << std::countr_zero(moving_bits));
        return child < ts && addr_of(buckets_[child].load(std::memory_order_relaxed)) != UNINITIALIZED;
    } // is_dead()

public:
    // Quiescent reclamation: recycles every node the set no longer needs, for
    // reuse by later allocations. Returns the number of keys in the set.
    //
    // CONTRACT
    //   Precondition: QUIESCENCE (see above) -- the caller guarantees that no
    //   other call on this set is in progress, that every earlier call
    //   happens-before this one and that this one happens-before every later
    //   call. Nothing here checks it; a concurrent operation of any kind is a
    //   data race. The client holds no references into the table during or
    //   across the call (the API hands out none).
    //   Postconditions: MEMBERSHIP is unchanged, and it is the only thing
    //   contains(), insert() and erase() can observe across the call: a value
    //   in the set is still in it, a value not in it still is not (the
    //   get_internal_* diagnostics see the arena change). Internally the call
    //   ends one PERIOD and begins the next (see PERIODS in the class
    //   overview): slots keep their addresses, but which node and which value
    //   occupy a slot may change; a slot hosts at most one node life per
    //   period, and link tag bits are terminal per node life. The table never
    //   shrinks (a doubling is never undone, published buckets stay
    //   published, pending splits stay pending: a split moves work rather
    //   than removing it, so they are left to the concurrent phase); no
    //   memory is returned to the system (the arena only ever grows;
    //   reclaimed nodes are reused); every key in the set has exactly one
    //   reachable node afterwards (see below), no dead node is reachable, the
    //   limbo lists are empty, and the return value is the number of keys,
    //   exact; node_count_ is reset from it.
    //   Requirements on T: copy-assignable (a reused node takes its new value
    //   by assignment; the old value is not destroyed here -- see LIFETIME OF
    //   VALUES in the class overview; callers must not rely on when an erased
    //   value's resources are released: the assignment at reuse overwrites
    //   it, and its destructor runs only at the set's destruction).
    //   Exceptions: see EXCEPTIONS in the class overview.
    //   Compiles and works for both AllowDelete values.
    //
    // FLOW
    //   1. Clear every shard's pop counter, so the first pop after this call
    //      adds 256 to node_count_ (the counters' batching restarts with the
    //      count set in step 4).
    //   2. Drain every shard's limbo list (lost or abandoned subchains,
    //      orphans, nodes whose assignment threw: unreachable since birth)
    //      into the free lists.
    //   3. Walk every PUBLISHED bucket j < table_size_, skipping UNINITIALIZED
    //      ones, and unlink every node is_dead() says is dead, relinking
    //      through the predecessor's word: a head keeps its seal level, a
    //      link keeps its tag bits (a kept predecessor is never MARKED but may
    //      be FROZEN with an unpublished child). Every kept node is counted.
    //   4. node_count_ := kept + the append credit (see its comment).
    //   Nodes from steps 2 and 3 are DEALT ROUND-ROBIN over the shards' free
    //   lists, on top of whatever those lists still hold, in one policy for
    //   both sources: a shard whose threads lost many splits or erased many
    //   keys does not hoard its own garbage, and splicing a limbo list whole
    //   would save nothing (finding its tail is the same walk). Each node is
    //   pushed as it is freed, not collected first (see EXCEPTIONS in the
    //   class overview).
    //
    // WHY ONE REACHABLE NODE PER KEY AFTERWARDS. A key's nodes lie along its
    // bucket path j0 -> j1 -> ... (each the next child of the previous, see
    // is_dead()), and a published bucket holds at most one unmarked node of a
    // key (an insert after the bucket was sealed for the child fails; a
    // duplicate insert into the same chain is excluded by insert()'s scan and
    // CAS). Every unmarked node in a bucket whose next child is published is
    // dead, so exactly the node in the deepest published bucket of the path
    // survives. This holds for both AllowDelete values, and it is why the
    // return value is the key count, not merely a node count.
    size_t reclaim() {
        const size_t ts = table_size_.load(std::memory_order_relaxed);
        size_t deal = 0;   // round-robin cursor: the shard the next freed node goes to
        for (size_t s = 0; s <= arena_mask_; ++s) {   // step 1: clear the pop counters
            Shard& shard = shards_[s];
            shard.free_head.store(free_word(free_addr_of(shard.free_head.load(std::memory_order_relaxed)), 0), std::memory_order_relaxed);
        } // for each shard's pop counter
        for (size_t s = 0; s <= arena_mask_; ++s) {   // step 2: limbo lists
            Shard& shard = shards_[s];
            word_t curr = shard.limbo_head.load(std::memory_order_relaxed);
            shard.limbo_head.store(EMPTY, std::memory_order_relaxed);
            while (addr_of(curr) != EMPTY) {
                Node* node = node_of(curr);
                curr = node->link.load(std::memory_order_relaxed);   // before push_free() rewrites the link
                push_free(shards_[deal++ & arena_mask_], node);
            } // walk the limbo chain
        } // for each shard's limbo list
        size_t live = 0;
        for (size_t j = 0; j < ts; ++j) {   // step 3: published chains
            // `slot` is the word that points at the current node (the bucket
            // head, then the previous kept node's link), `slot_word` its
            // current value and `keep` the tag bits that survive a relink.
            std::atomic<word_t>* slot = &buckets_[j];
            word_t slot_word = slot->load(std::memory_order_relaxed);
            if (addr_of(slot_word) == UNINITIALIZED) continue;   // pending split: see the contract
            word_t keep = LEVEL_MASK;
            while (addr_of(slot_word) != EMPTY) {   // walk bucket j's chain
                Node* node = node_of(slot_word);
                const word_t link = node->link.load(std::memory_order_relaxed);
                if (is_dead(node, link, j, ts)) {
                    slot_word = (slot_word & keep) | addr_of(link);   // unlink: bypass the dead node
                    slot->store(slot_word, std::memory_order_relaxed);
                    push_free(shards_[deal++ & arena_mask_], node);
                } else {
                    ++live;
                    slot = &node->link;
                    slot_word = link;
                    keep = MARK_BIT | FROZEN_BIT;
                } // dead: unlinked; alive: becomes the predecessor
            } // walk chain
        } // for each bucket
        size_t credit = 0;   // step 4: see node_count_
        for (size_t s = 0; s <= arena_mask_; ++s) credit += (256 - (shards_[s].nodes.size() & 255)) & 255;
        node_count_.store(live + credit, std::memory_order_relaxed);
        return live;
    } // reclaim()

    // DIAGNOSTICS. The get_internal_* accessors below, and the struct one of
    // them returns, show the arena to the tests and the benchmark. They come
    // in two classes:
    //   - RACE-FREE, approximate while operations run:
    //     get_internal_node_count() and get_internal_arena_shards() read only
    //     atomics and values fixed at construction, and may be called
    //     concurrently with any operation; each shard size the node count
    //     reads is one that shard had, but the sum over the shards is not one
    //     instant's total. At a quiescent point it is exact;
    //     get_internal_arena_shards(), a value fixed at construction, is
    //     exact always.
    //   - QUIESCENT ONLY: get_internal_accounting() and the free and limbo
    //     counts taken from it, under the QUIESCENCE precondition above. The
    //     accounting sweep indexes buckets_ up to a relaxed-loaded
    //     table_size_, which is safe concurrently only below a table size the
    //     calling thread has acquired (channels 2 and 3): against a
    //     concurrent doubling it is a data race. (Its walks of chains and
    //     lists are not: it dereferences only nodes whose slots it collected
    //     below a shard size() it acquired, and reads nothing of a node but
    //     its atomic link.) Nor would its counts and `consistent` describe
    //     one instant while operations run.
    //
    // Diagnostic accessor: total nodes ever appended to the arena (reachable,
    // free and limbo alike): the sum of the shards' sizes, exact at a
    // quiescent point.
    size_t get_internal_node_count() const {
        size_t n = 0;
        for (size_t i = 0; i <= arena_mask_; ++i) n += shards_[i].nodes.size();
        return n;
    } // get_internal_node_count()
    // Diagnostic accessor: the number of arena shards (a power of two), as
    // the constructor fixed it.
    size_t get_internal_arena_shards() const { return arena_mask_ + 1; }
    // Diagnostic accessors, quiescent only: the number of nodes on all free
    // lists / on all limbo lists, as the accounting sweep below counts them.
    // A corrupted (cyclic) list does not hang: the sweep stops at the first
    // address it has already seen -- but the count returned here is then
    // merely the nodes walked before the stop (a cycle A->B->C->A counts 3);
    // the corruption itself is visible only in get_internal_accounting()'s
    // `consistent`, which tests assert alongside these counts.
    size_t get_internal_free_count() const { return get_internal_accounting().free_nodes; }
    size_t get_internal_limbo_count() const { return get_internal_accounting().limbo; }

    // Diagnostic accounting sweep, quiescent only (see DIAGNOSTICS above):
    // where every arena slot is. `consistent` is true iff every slot of every
    // shard is reachable from a published bucket head, on a free list, or on
    // a limbo list, exactly one of the three and exactly once, and no list or
    // chain contains an address that is not a slot. The counts are what was
    // found (a walk stops at the first address that is unknown or already
    // seen, so a cyclic list cannot hang it, and then `consistent` is false).
    // Expected true at every quiescent point, including after an exception on
    // any operation (see EXCEPTIONS in the class overview).
    struct InternalAccounting {
        size_t slots;       // sum of the shards' sizes (get_internal_node_count())
        size_t reachable;   // nodes reachable from published heads, live and dead alike
        size_t free_nodes;  // nodes on the free lists
        size_t limbo;       // nodes on the limbo lists
        bool consistent;    // reachable + free_nodes + limbo == slots, with no slot twice
    }; // struct InternalAccounting
    InternalAccounting get_internal_accounting() const {
        // Every slot's address, sorted, with a seen flag per slot. A shard is
        // indexed over [0, size()) with operator[], which the deque allows
        // after the acquire load that its size() is (channel 3); here the
        // quiescence precondition would allow it anyway.
        std::vector<word_t> slots;
        slots.reserve(get_internal_node_count());
        for (size_t s = 0; s <= arena_mask_; ++s) {
            const ConcurrentAppendDeque<Node, ARENA_BLOCK>& nodes = shards_[s].nodes;
            for (size_t i = 0, n = nodes.size(); i < n; ++i) slots.push_back(word_of(&nodes[i]));
        }
        std::sort(slots.begin(), slots.end());
        std::vector<unsigned char> seen(slots.size(), 0);
        InternalAccounting acc{slots.size(), 0, 0, 0, true};
        // Marks `addr` as seen; false (and the sweep inconsistent) if it is not
        // a slot or was seen before.
        auto visit = [&](word_t addr) -> bool {
            auto it = std::lower_bound(slots.begin(), slots.end(), addr);
            if (it == slots.end() || *it != addr || seen[it - slots.begin()]) {
                acc.consistent = false;
                return false;
            }
            seen[it - slots.begin()] = 1;
            return true;
        };
        const size_t ts = table_size_.load(std::memory_order_relaxed);
        for (size_t j = 0; j < ts; ++j) {   // published chains
            word_t curr = buckets_[j].load(std::memory_order_relaxed);
            if (addr_of(curr) == UNINITIALIZED) continue;
            for (; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++acc.reachable) curr = node_of(curr)->link.load(std::memory_order_relaxed);
        }
        for (size_t s = 0; s <= arena_mask_; ++s) {   // free and limbo lists
            word_t curr = free_addr_of(shards_[s].free_head.load(std::memory_order_relaxed));
            for (; curr != FREE_EMPTY && visit(curr); ++acc.free_nodes) curr = node_of(curr)->link.load(std::memory_order_relaxed);
            curr = shards_[s].limbo_head.load(std::memory_order_relaxed);
            for (; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++acc.limbo) curr = node_of(curr)->link.load(std::memory_order_relaxed);
        }
        if (acc.reachable + acc.free_nodes + acc.limbo != acc.slots) acc.consistent = false;
        return acc;
    } // get_internal_accounting()
}; // class ConcurrentResizableHashSet

#endif // CONCURRENT_HASH_SET_H
