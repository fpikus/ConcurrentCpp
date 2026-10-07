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
#ifndef CONCURRENT_HASH_SET_RCU_H
#define CONCURRENT_HASH_SET_RCU_H

#include <algorithm>
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

struct empty_struct_rcu {};

namespace concurrent_hash_rcu_detail {
// Process-wide sequential thread numbering, shared by every instantiation of
// ConcurrentResizableHashSetRCU: a thread's number is assigned on its first
// allocation and never changes. Sequential (not hashed) on purpose: threads
// started together get consecutive numbers, so `number mod shards` spreads a
// batch of up to `shards` threads over distinct arena shards with no
// collisions, however many earlier threads have come and gone.
inline std::atomic<unsigned> next_thread_number{0};
inline unsigned thread_number() {
    static thread_local unsigned number = next_thread_number.fetch_add(1, std::memory_order_relaxed);
    return number;
}
} // namespace concurrent_hash_rcu_detail

template <typename T, typename... Args>
using DefaultConcurrentDequeRCU = ConcurrentAppendDeque<T, 1024>;

// ===========================================================================
// ConcurrentResizableHashSetRCU -- design overview (read this before the code).
//
// A closed-addressing (chained) hash set that grows by doubling, without ever
// stopping the world and without a global rehash pass. The two big ideas are
// (1) an append-only node arena in which nothing ever moves, and (2) lazy, cooperative,
// per-bucket splitting that copies (never moves) live nodes into new buckets.
//
// STORAGE
//   shards_  : the node arena, SHARDED: an array of Shard, each an append-only
//              deque of Node (ConcurrentAppendDeque) plus that shard's FREE
//              LIST, LIMBO LIST and RETIRED LIST heads (see RECLAMATION
//              below). A thread always allocates from the shard its thread
//              number selects (see alloc_node()). A node is never moved and
//              never returned to the system; between two quiescent points
//              (see reclaim()) a published node's value is never written, its
//              link changes only by the one CAS that tags it and by unlink
//              CASes that bypass dead successors (see WORD ENCODING), and its
//              retire_link is written only when the node is retired. Nodes are
//              addressed by POINTER: a deque is segmented and its blocks never
//              move, so a node's address is stable for the life of the set,
//              and nothing that reads the structure knows or cares which shard
//              a node lives in. The deque's index is used once per node, to
//              obtain that address at allocation, and by the test-only sweeps
//              at a quiescent point; no read path ever goes through a deque.
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
//       one CAS. No concurrent operation writes a tagged link. Concurrently,
//       a published link's address bits change only by an unlink CAS on a
//       LIVE link, which bypasses a run of dead successors. reclaim()
//       (quiescent) may relink a kept FROZEN node past dead successors, and
//       rewrites the link of every node it frees, while that node is
//       unreachable (see RECLAMATION).
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
//   key now hashes to j under the wider mask. The copied nodes stay in the
//   parent chain, FROZEN, until a writer bypasses them (the split's own winner
//   first, right after it publishes j: see WHY COPY below), so for a while a
//   key that moved to j exists in BOTH the parent chain (a superseded, FROZEN
//   copy) and bucket j. Splitting is recursive: a parent that is itself still
//   UNINITIALIZED is split first, so the bucket tree is filled in on demand.
//
// WHY COPY, AND WHEN A DEAD NODE LEAVES ITS CHAIN
//   Nodes are copied, never moved: a node that a split supersedes stays where
//   it is, tagged FROZEN, and a node erase() deletes stays where it is, tagged
//   MARKED. A node address observed by any traversing thread therefore stays
//   a valid node for as long as that thread's operation runs, and memory is
//   reused only at a QUIESCENT POINT, when the caller guarantees that no
//   operation is in progress (reclaim(), below), never returned to the system
//   before the set is destroyed. That is what lets the read path dereference
//   node pointers with no hazard pointers, no reference counts, and no
//   reclamation protocol. The nodes that go dead between two quiescent points
//   are (a) tombstoned nodes (AllowDelete), (b) superseded parent copies after
//   a split, (c) speculative split subchains that lost their publishing CAS
//   or were abandoned by a throw mid-build, (d) a node insert() prepared and
//   never published (a retry found the key published by another thread, or
//   threw), and (e) a free-list node whose value assignment threw. (c), (d)
//   and (e) were never published and are recorded on a LIMBO list before the
//   operation that made them returns or throws, because nothing else could
//   ever find them again.
//   DEAD NODES. (a) and (b) are DEAD IN THEIR CHAIN: reachable, tagged, and
//   deciding nothing. A DEAD NODE is a tombstone, or a FROZEN node whose
//   child bucket is published; a FROZEN node whose child is not yet published
//   is not dead: it is still authoritative for its key (INVARIANT below). A
//   concurrent WRITER takes dead nodes out: it bypasses a maximal RUN of
//   consecutive dead nodes with one CAS on the live word before the run, its
//   PRED WORD (a bucket head, or the untagged link of a live node), and the
//   winner of that CAS RETIRES exactly the run onto its shard's retired list
//   (unlink_run(), push_retired()). Where this happens:
//   - the split winner's cleanup pass over the parent chain, right after it
//     publishes the child (the copies it superseded leave with the split that
//     superseded them, and the pass is lazy with respect to the doubling, as
//     splitting is; whichever operation started the split runs it, a
//     contains() included);
//   - erase(), right after its mark (an eager self-unlink: one attempt
//     through the predecessor it tracked, one restart from the head, then
//     give up);
//   - the walks of insert() and erase(), which attempt to bypass every dead
//     run they pass behind a live word, on the way to their key (a hit stops
//     a walk at its key, so a hit walk attempts the runs before the key and
//     only a MISS walks the whole chain).
//   contains()'s own walk writes nothing: a reader pays for no unlinking
//   except by starting a split.
//   STRAGGLERS. Every one of these is ONE CAS per run, never retried
//   (erase()'s restart is the one exception, bounded at two): a lost CAS is
//   another thread's completed step on the same word. When that step was a
//   peer's bypass of the same run (the word already holds the successor this
//   thread computed), nothing is left behind. Otherwise -- the word gained a
//   prepended node, a seal level, a tag of the predecessor, or a peer's
//   bypass of a run of a different extent -- the dead nodes still reachable
//   are a STRAGGLER, collected by the next writer walk that passes them (hit
//   or miss), the next cleanup pass over the chain (every doubling gives the
//   bucket a new child, and the split of ANY child of the bucket walks the
//   whole parent chain with no hit to stop it), or reclaim()'s chain walk,
//   which collects everything. Stragglers also arise from a stale view (a
//   walker whose table size does not yet cover the doubling a node was frozen
//   for cannot decide it), from an erase() that gave up (two lost attempts,
//   the first of which may have been skipped because the walk's own CAS on
//   the pred word was lost), from a split stalled or thrown between its
//   freeze and its publish, and from Hash{} throwing inside an unlinking site
//   (EXCEPTIONS). A dead run that no writer walks past again, in a bucket
//   none of whose new children is accessed, stays until reclaim(): its
//   lifetime is unbounded only when the table stops doubling or the bucket's
//   new children are never accessed.
//   POSTCONDITIONS. Unlinking is therefore best effort under contention: no
//   operation guarantees that a dead node has left its chain when it returns.
//   What an UNCONTENDED call does guarantee -- one during which no other
//   operation on the set runs, and that returns normally -- is this: a
//   split's cleanup leaves no dead node reachable in the parent chain behind
//   an untagged word (the bucket head or a live node's link); an insert() or
//   erase() walk leaves no dead run reachable behind the bucket head or a
//   live node it passed (a run at the head counts: the head is an untagged
//   word); an erase() leaves its tombstone unreachable, unless a tagged,
//   not-dead node lies between it and the last live word before it, which in
//   uncontended use arises only after an earlier split threw between its
//   freeze and its publish (behind such a node nothing can be bypassed until
//   its child is published and a later walk passes it).
//   NOT TAKEN. Two designs are deliberately NOT taken here: a contains() that
//   bypasses the dead runs it passes (it would make every reader a writer),
//   and a standalone collector call that walks published chains like a
//   writer miss walk without inserting anything; both would trade read-path
//   cost or an API call for fewer stragglers, and this class keeps the read
//   walk pure and reclaim() as the only collector of the remainder.
//   RETIREMENT. A retired node is not reused before reclaim(): the retired
//   lists only separate "dead and out of its chain" from "dead and still in
//   it", for reclaim() to drain without a chain walk (RECLAMATION).
//
// EXCEPTIONS: WHAT IS AND IS NOT HANDLED (the one place that says it all)
//   Exception behavior is mostly out of scope for this class, as it is for
//   the arena: concurrent_deque.h declares exception safety out of scope in
//   the comment on its resize(), and its reallocate_directory() can leak a
//   retired directory if the vector push after the directory swap throws.
//   Hash{} is expected not to throw; where it can, the limbo pushes below
//   apply to it as to any other throw. What CAN throw here is T's copy
//   constructor and copy assignment, Hash{}, and the allocations of the
//   deques (bad_alloc); nothing else in this class throws.
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
//     as for a lost CAS), and rethrows. Bucket j stays UNINITIALIZED, and the
//     nodes already FROZEN remain authoritative for their keys, which the
//     INVARIANT allows: the next operation that needs j splits it again from
//     them, so the work is redone, not lost. If Hash{} throws inside the
//     winner's cleanup pass (step 4), AFTER the publishing CAS (the case the
//     comments below call W4), bucket j is published and complete; the
//     exception leaves the dead nodes of the parent chain reachable, as a
//     lost CAS would, for a later walk, split or reclaim() to collect. The
//     caller of split_bucket() sees the exception, not the published bucket.
//   - insert(): a throw before the publishing CAS -- from alloc_node(),
//     Hash{}, or a split_bucket() met on any attempt -- leaves the key as it
//     was; the catch around the retry loop pushes the pending node, if one
//     was allocated on an earlier attempt, to limbo and rethrows. new_node is
//     nulled the moment its CAS publishes it, so that catch never pushes a
//     published node. A throw from the doubling AFTER the publishing CAS
//     succeeded leaves the key a member although the caller sees an
//     exception and no return value; nothing is said about the state after a
//     throw from inside buckets_.resize().
//   - contains(): holds no private node; it throws only what Hash{} or a
//     split_bucket() it falls into throws, and changes no membership when it
//     does (a split it won is published, cleanup unlinks included).
//   - erase(): holds no private node; it throws what Hash{} or a
//     split_bucket() throws. A throw BEFORE the mark (from the hash of the
//     key, from a split, or from the hash of a FROZEN node the walk passes,
//     the call site H5' of PINNED ORDERS, before contains()) leaves the key
//     as it was. Hash{} is also called inside the self-unlink that follows a
//     successful mark (H5; the case the comments below call P5): a throw
//     there leaves the key DELETED (the mark is the deletion) but the caller
//     sees an exception and no `true`, as insert() does on a throw from the
//     doubling after its publishing CAS. Any unlink a throwing erase() or
//     insert() walk had already won stands; it changed no membership.
//   - reclaim(): throws only if Hash{}(key) throws. The set is then
//     consistent: every node unlinked so far is already on a free list, since
//     each node is pushed as it is freed, not collected first, and
//     node_count_ keeps its previous value, an over-count, so growth stays
//     early, never late. reclaim() may be called again.
//
// RECLAMATION: FREE LISTS, LIMBO LISTS, RETIRED LISTS, reclaim()
//   Every shard has two intrusive singly linked lists threaded through the
//   nodes' `link` words: a FREE LIST of nodes available for reuse and a LIMBO
//   LIST of unreachable nodes waiting for the next reclaim(). The next pointer
//   lives in the atomic link word and never in the T bytes: a thread that
//   loses a pop race may still read the node's link while the winner is
//   already assigning the node's value (see alloc_node()).
//   A third list per shard, the RETIRED LIST, is threaded through a word of
//   its own, Node::retire_link: it holds the dead nodes that a concurrent
//   operation has unlinked from their chain (push_retired() is its one entry
//   point, called by the unlink winner with exactly the run it bypassed), and
//   such a node may still be traversed by an operation that reached it before
//   the unlink, so its `link` must stay what it was (EXIT). The lists are
//   LIFO and push-only between two reclaim() calls; each push is a RELEASE
//   CAS, so an ACQUIRE load of a retired head is a happens-before boundary
//   for the whole list below the value loaded (every push is a
//   read-modify-write extending the release sequence of the one before it).
//   reclaim() drains them in two phases, sequester_retired() and
//   reclaim_sequestered(), onto the free lists, where a drained node is a
//   free node like any other; their comments state the drain's own, weaker
//   precondition. A retired node is reused no earlier than a node reclaim()'s
//   chain walk unlinks: retirement changes WHEN a dead node stops being
//   walked, not when its memory comes back.
//   BEYOND QUIESCENCE. A reclamation that frees retired nodes WITHOUT
//   quiescence needs three things this class does not provide. First, a way
//   to know that no operation still holds a retired node's address: handles
//   with a generation counter, where a node is stamped at retirement (the
//   STAMP form: retire_link's nine spare bits can hold a stamp, compared
//   modulo 512, which bounds the number of generations that may be live at
//   once), or where a generation bump takes an acquire snapshot of every
//   retired head, so that every node below the snapshot was pushed, and
//   unlinked, before the bump (the SNAPSHOT form). Second, thread-safe free
//   lists: a drain that runs while alloc_node() pops pushes concurrently with
//   the pops, which the pop's ABA argument below excludes; two options, both
//   open: option A, use the free head's 9-bit pop count as a version tag,
//   which wraps at 512 pops; option B, one drain at a time, started under
//   the drain precondition, so that no popper can have seen an address the
//   drain resurrects. Third, an answer to the
//   unlink CAS's own ABA premise, that every expected value of an unlink CAS
//   (the pred word and the recorded run) is loaded and used within one such
//   handle. That premise needs a BRIDGE from a node's retirement to the head
//   loads of later operations: the snapshot form supplies it by construction
//   (every unlink below the snapshot happens-before the bump, and a later
//   handle acquires the bump), condition (2) of the drain precondition
//   (stated before sequester_retired()) supplies it for a drain, and
//   surrendering a handle alone does not; for the stamp form it is an open
//   question.
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
//     never pushed again before the next quiescent point. The free list needs
//     no Harris-style marking for the same reason: nothing unlinks from it
//     concurrently (the chains are another matter: WHY COPY above).
//   - PUSH onto a limbo list is a Treiber push (CAS) on the pushing thread's
//     own shard; the list is push-only until reclaim() drains it, so it is
//     ABA-free too, and nothing but reclaim() and the test-only sweeps ever
//     read it.
//   - reclaim() (contract at its definition) walks every published bucket,
//     unlinks every dead node, deals the dead, the retired and the limbo
//     nodes round-robin over the shards' free lists, and resets the pop
//     counters and node_count_. It takes no lock: its precondition is that
//     the caller has made every earlier operation happen-before it and it
//     happens-before every later one, which is what "quiescent" means here.
//     Under that precondition it needs no atomic ordering at all (all of its
//     atomic accesses are relaxed): every write it makes is visible to every
//     later operation by happens-before, exactly as the constructor's writes
//     are.
//   - PERIODS. reclaim() calls divide the set's life into PERIODS (the first
//     starts at construction, the last ends at destruction). The CLIENT's
//     view: the only thing the client can observe across a reclaim() is
//     MEMBERSHIP -- a value in the set before it is in the set after it, a
//     value not in it still is not. The client holds no references into the
//     table (the API hands out none), so nothing of the client's can outlive
//     a period, and no ABA problem can reach the client. The INTERNAL view:
//     slots never move (a slot's address is stable for the life of the set,
//     see STORAGE), doublings are never undone, published buckets stay
//     published with their seal levels, and kept links keep their tag bits.
//     What reclaim() changes is which node, holding which value, occupies a
//     slot; the guarantees that are period-bounded are exactly these: a slot
//     hosts at most one NODE LIFE per period (freed by one reclaim(),
//     popped and published at most once before the next), and a link's
//     MARK/FROZEN bits are terminal per node life.
//   - ADDRESS REUSE and the deciding CASes. insert()'s CAS on a bucket head,
//     erase()'s CAS on a link and the unlink CAS on either are safe against
//     ABA because within one period no node address is published twice: a
//     node leaves a chain through an unlink CAS or through reclaim(), and
//     re-enters one only through a pop that follows a reclaim(). An address a
//     thread observed in a head or link therefore names the same node, in
//     that chain or retired from it, for the whole of that thread's
//     operation, and an unlink CAS whose expected value still holds really is
//     bypassing the run it recorded. No CAS of any operation, and no pop of a
//     free list, can span a reclaim(): nothing else runs on the table while
//     it does. There is therefore no ABA problem the client can get into,
//     and none in the free lists.
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
//   With AllowDelete == false the invariant holds in the same form: the freeze
//   runs for both values (see split_bucket() step 2), so a moved key has a
//   FROZEN node in the parent and its live copy in the child. Nothing is ever
//   tombstoned, so the tag decides nothing about membership there (both nodes
//   answer "present"); it is set so that every dead node is tagged, which the
//   removal of dead nodes from their chains depends on.
//   ROOTS. The traversal roots are the published bucket heads [0, ts) for
//   the operation's acquired ts; the free, limbo and retired heads are not
//   roots: no operation walks a list from them.
//   UNREACHABILITY. Let u be the unlink CAS that bypasses node X. If u
//   happens-before the bucket-head load that begins an operation's walk, that
//   walk never loads X's address from any word. The words that ever held X's
//   address are of four kinds, and the walk reads none of them with X in it:
//   (1) u's own word: its post-u value is in u's modification order, and a
//       load that happens-after u returns that value or a later one.
//   (2) The links of the run-predecessors of X that u itself bypasses
//       (P -> D1 -> X: D1's link holds X): the walk never loads D1's address,
//       by this same argument applied to D1, whose address was in u's word.
//   (3) The link of a node D bypassed EARLIER, by an unlink u_D (D's
//       immutable link still holds X): the walk never loads D's address
//       because u_D happens-before u -- u's unlinker reached its pred word by
//       acquire loads along the chain from the head, and the word u_D
//       rewrote lies on that path, or was itself bypassed later by an unlink
//       whose word does, recursively; every later write to a published head
//       or link is a read-modify-write, which extends u_D's release sequence,
//       so the unlinker's acquire of that word synchronizes with u_D. The
//       walk's and the pred word's acquires are load-bearing here, not u's
//       release.
//   (4) Words that held X's address and were overwritten before u by a
//       read-modify-write on that same path (a prepend at the head, an
//       earlier bypass): u's unlinker read them at or after the overwrite, so
//       a load that happens-after u returns the overwritten value or a later
//       one (coherence).
//   There is no fifth kind: an address enters a published word only by a
//   prepend's publishing CAS, a split's publishing CAS (whose subchain was
//   linked privately), or an unlink CAS, all on the chain u's unlinker
//   walked.
//   EXIT. An operation whose head load does not happen-after u may hold X and
//   may read X's `link` and `value`, which no concurrent operation writes
//   after X is tagged. Following links from X reaches the current chain,
//   possibly after passing through nodes retired LATER than X, each of them
//   dead, tagged and exitable the same way: a tagged link is only ever
//   bypassed, by an unlink CAS that skips dead nodes forward, never
//   rewritten. This is why a retired node's `link` is never rewritten before
//   reclaim(), and why a tagged link is never a CAS target.
//   P9. A node FROZEN for child j is bypassed only after j's publishing CAS:
//   is_dead_now() requires the child head published, and the winner's
//   cleanup pass runs after its own publish. A splitter that LOST its publish
//   and whose snapshot of the parent was taken after a winner's cleanup
//   missed the copies the winner already bypassed; that is harmless only
//   because its publish CAS must fail, which it does with no happens-before
//   needed: two CASes from UNINITIALIZED on bucket j cannot both succeed,
//   since a successful CAS is a read-modify-write that reads the value
//   immediately before its own write in the head's modification order
//   ([atomics.order]), and after the first successful publish no later
//   value in that order is UNINITIALIZED -- and a publish of j did succeed,
//   since a copy can be bypassed only after it.
//   LINEARIZABILITY of a miss, with dead nodes leaving concurrently, in two
//   halves.
//   (i) A miss is linearized before every insert that does NOT happen-before
//       the call: no mechanism is needed, and no test may assert a hit for a
//       key whose insert does not happen-before the call.
//   (ii) For an insert that DOES happen-before the call, a miss is final only
//       if the key was deleted:
//       (a) if the key's node was MARKED and then bypassed, the deletion
//           linearizes at the mark CAS (the marking CAS is THE deletion, see
//           above), and contains() and erase() skip a MARKED node anyway.
//       In every other case the miss is never final:
//       (b) if the key's FROZEN parent copy was bypassed after a later split
//           published the copy's bucket, the reader's reload of table_size_
//           returns a strictly larger size (channel 5), so the walk retries
//           in the child, at most once per doubling.
//       (c) The copy's child is then PUBLISHED to the reader, never
//           UNINITIALIZED: the unlinker read the child head published before
//           its CAS, and the reader's acquire of the CAS's value orders the
//           reader's own child-head load after that read, so by read-read
//           coherence it returns the published value or a later one. If the
//           reload sent the reader to a deeper bucket on the key's path that
//           is still UNINITIALIZED, that is a first access, and the reader
//           splits it from the published child as any operation would; a
//           losing splitter is the P9 case.
//       The list is complete because no unlink bypasses a live node or a
//       FROZEN node whose child is unpublished (is_dead_now()): the only
//       nodes a walk can miss through a bypass are tombstones, (a), and
//       superseded copies, (b).
//       (d) The edge of (b) is the unlinker's OWN: it decided the copy dead
//           at a table size it had READ, which covers the child; that read is
//           sequenced before its release CAS, and the reader, which acquires
//           the CAS's value (or a later value of the word, through the
//           release sequence that read-modify-writes extend), has the read
//           happen-before its own reload, so by read-read coherence the
//           reload returns that size or a newer one, whatever the order of
//           the unlinker's load. The acquires of the FROZEN link and of the
//           child head inside is_dead_now() are not what carries this edge.
//           The acquires of the walk and the run walk carry something else:
//           an EARLIER unlinker's edge, which a walker at a stale size
//           inherits when it bypasses a node whose link that earlier unlinker
//           rewrote and a tag CAS then extended; for a MARKED link there is
//           no other route, and for a FROZEN link the child head's publisher
//           need not have acquired the earlier size either (channel 5).
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
//      by the test-only sweeps, at a quiescent point, over [0, size()).
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
//   5. UNLINKING (unlink_run()). The unlink CAS is release on success; the
//      head load that starts a writer walk, every link load of the walk and
//      every link load of the run walk are acquire (so is the child-head
//      load of is_dead_now(), by channel 1). This channel is SAFETY-bearing,
//      not progress-bearing: it is what makes half (ii)(b) of the miss
//      argument hold for a bypass of a FROZEN copy (a head bypass also rides
//      channel 4's release sequence on the head word). It carries two things.
//      FIRST, the deciding thread's OWN edge: an unlinker decides a FROZEN
//      node dead only at a table size it READ that covers the node's child
//      (is_dead_now()). That read is sequenced before the unlinker's release
//      CAS, and a reader that acquires the CAS's value -- or any later value
//      of that word, since every later write to a published head or link is
//      a read-modify-write extending the release sequence -- therefore has
//      the unlinker's read of table_size_ happen-before its own reload, and
//      by read-read coherence the reload returns that size or a newer one.
//      No order on the unlinker's table_size_ load is needed for this (its
//      acquire is channel 2's: it is what lets the unlinker index the new
//      buckets), and the acquire of the FROZEN link and the acquire of the
//      child head are redundant for it.
//      SECOND, EARLIER unlinkers' edges, through release sequences: when an
//      unlinker rewrote a live link (from F to E, say) and a mark or freeze
//      then tagged that link (a read-modify-write, so the tagged value is
//      still in the unlinker's release sequence), a walker at a STALE size
//      may bypass the tagged node and install E through its own pred word;
//      the reader that follows that bypass needs the EARLIER unlinker's size,
//      and the only route is the walker's acquire of the tagged link -- the
//      walk's load when the node starts a run, the run walk's load otherwise.
//      For a MARKED link there is no other route at all; for a FROZEN link
//      the child head's publisher need not have acquired the earlier size
//      either. The eraser has a third such route, the ACQUIRE FAILURE ORDER
//      of its mark CAS: when the mark fails on a live value that an unlinker
//      wrote (a new successor address), the retried mark and the self-unlink
//      that follows install that successor through the pred word, and the
//      failed CAS's read is the eraser's only acquire of the unlinker's write
//      (the run walk then loads the successor's own link, which the unlinker
//      did not write); with that failure order relaxed a reader following
//      the eraser's bypass would reload the old size. The same acquire of a
//      LIVE pred word that an earlier unlinker rewrote is what
//      UNREACHABILITY's u_D clause rests on. Every acquire of the walk and
//      the run walk is therefore load-bearing, for earlier unlinkers' edges,
//      whatever the walker's own size.
//      The retire push (push_retired()) is a release CAS so that an acquire
//      of a retired head orders everything before every push below it;
//      nothing in this class acquires one, and reclaim() needs no ordering:
//      the release is reserved for a reclamation that snapshots the retired
//      heads without quiescence (RECLAMATION).
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
//   What remains cross-variable is the read-side chain of channels 4 and 5
//   (acquire word -> acquire table_size_). For a thread that meets a seal
//   level above its own, or a FROZEN node for its key, it is used for
//   PROGRESS: the reload returns a larger size at once, and were the edge
//   missing the thread would re-read until the larger size became visible,
//   never returning a wrong answer. For a thread whose walk MISSED its key
//   because the key's FROZEN copy was bypassed concurrently, the same chain
//   is used for SAFETY: the reload must return a larger size, or the miss
//   would be final and wrong (half (ii)(b) of the miss argument, STALE
//   GEOMETRY). The chain is a happens-before chain on two different atomics
//   (the bypassed word, then table_size_), closed by read-read coherence:
//   the unlinker's READ of the larger size (or, through a release sequence,
//   an earlier unlinker's) happens-before the reader's reload, so the reload
//   cannot return an older value, whatever the order of that read. A miss
//   that is confirmed by an unchanged table_size_ (contains(), erase())
//   linearizes at the first table_size_ load; see contains() and the two
//   halves under STALE GEOMETRY.
//
// PROGRESS: this structure is lock-free on the pure read/traverse path, but it
// is NOT wait-free and not lock-free end to end: contains(), insert() and
// erase() all fall into split_bucket() when they meet an UNINITIALIZED bucket,
// and split_bucket() allocates through its arena shard, as does every
// insert() (alloc_node()): a lock-free pop from the shard's free list when it
// has one, else an append under the deque's internal SpinLock; with one thread
// per shard that lock is uncontended, but it is a lock. Resize itself is
// serialized by resize_lock_. reclaim() is not a concurrent operation at all
// (see its contract). Every loop that unlinking adds is lock-free in the
// sense that a failed CAS implies another thread's completed step: the mark
// and freeze retries on a live failure value (each failure is a completed
// unlink through the node), the retire push's CAS loop (each failure a
// completed push, or a spurious failure of its weak CAS, a plain retry), and
// erase()'s restart (one, after a lost CAS); the opportunistic unlink sites
// and the cleanup pass make ONE attempt per run and never retry, and
// erase()'s self-unlink makes at most two.
//
// Template parameters:
//   T           : element (key) type; must be equality-comparable, hashable,
//                 copy-constructible and copy-assignable (a node reused after
//                 a reclaim() takes its new value by assignment; see
//                 EXCEPTIONS for what a throwing assignment must satisfy).
//   AllowDelete : when true, compiles erase() (tombstone deletion). When
//                 false, no node is ever marked; the freeze step of
//                 split_bucket() runs for both values (every superseded
//                 parent copy is FROZEN), so a chain's links still gain a
//                 tag, and the MARK checks on the read paths are tests of a
//                 bit that is never set.
//   Hash        : hash functor; must return the SAME hash for a key every call
//                 (the split math re-hashes keys under wider masks). Besides
//                 the key of every call, Hash{} is applied to the stored
//                 values of FROZEN nodes that writer walks and cleanup passes
//                 meet (to find the node's child bucket): a cost per FROZEN
//                 node passed, and the place where a throwing Hash{} leaves
//                 a dead run behind (EXCEPTIONS).
//   Container   : the append-only, address-stable arena template: elements
//                 never move once constructed (see channel 3).
// ===========================================================================
template <
    typename T,
    bool AllowDelete = false,
    typename Hash = std::hash<T>,
    template <typename, typename...> class Container = DefaultConcurrentDequeRCU
>
class ConcurrentResizableHashSetRCU {
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
    //                   therefore mutually exclusive and both terminal for the
    //                   node's life: a tagged link is never written again by a
    //                   concurrent operation, and only reclaim() rewrites it,
    //                   when it frees the node for a pop to reuse (a free
    //                   node's link is a bare address, no tag bits).
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
        // publishes/observes chain structure. The address bits of a published
        // link change in exactly one way, by an unlink CAS on a LIVE link
        // that bypasses a run of dead successors (unlink_run()); a TAGGED link
        // (MARK or FROZEN) is written by no concurrent operation, and by
        // reclaim() only when it frees the node (a kept FROZEN node's link may
        // be relinked past dead successors there too), so a dead node can
        // always be EXITED through the successor it held when it died, until
        // the next quiescent point. On a free or limbo list the same word is
        // the list's next pointer (a bare address); a retired node keeps its
        // chain link intact (the retired lists use retire_link).
        std::atomic<word_t> link;
        // The RETIRED LIST's next pointer: a bare node address or EMPTY, tag
        // bits zero. A separate word, and not `link`, because a retired node
        // is still EXITED by concurrent operations that reached it before it
        // was unlinked: its `link` must keep the chain successor (see above).
        // Written only by push_retired() (the retiring thread, privately for
        // the run and then as the list's next pointer) and read only by the
        // quiescent reclaim_sequestered() and the test-only sweep; no
        // concurrent operation reads it, so its stores are relaxed and only
        // the retired head's CAS carries ordering. The nine bits an address
        // never uses (63..58 and 2..0) are zero here and RESERVED for a
        // retire-time stamp. LAST member on purpose: `value` and `link` keep
        // their relative placement, and the hot read path never touches this
        // word. A free or limbo node's retire_link is stale and never read.
        std::atomic<word_t> retire_link;

        // Default ctor: an unlinked live node whose successor is EMPTY. Rarely
        // used -- the arena is filled via the (val, next) ctor below; this
        // exists only for the container's value-initialization path.
        Node() : value(), link(EMPTY), retire_link(EMPTY) {}
        Node(const T& val, word_t next) : value(val), link(next), retire_link(EMPTY) {}
    };
    static_assert(alignof(Node) >= 8, "the low three bits of a node address are the link's tag bits");

private:
    // The dynamically resizable array of atomic bucket heads.
    // Each entry holds the address of the first node in the bucket's chain,
    // plus the bucket's seal level (see the word encoding).
    Container<std::atomic<word_t>> buckets_;

    // One arena shard: the append-only node deque and the heads of the shard's
    // free, limbo and retired lists (see RECLAMATION in the class overview),
    // on two cache lines of their own, apart from the deque's members (which
    // the deque's default build pads onto lines of their own):
    //   - the FREE LINE: free_head and limbo_head. The free head is written by
    //     every pop of the shard's threads, and with more threads than shards
    //     a pop CAS and the deque's lock exchange would otherwise fight over
    //     one line; the limbo head is written rarely (lost splits, orphans,
    //     throwing assignments) and costs nothing on the free head's line.
    //   - the RETIRED LINE: retired_head, the drain's private slot and the
    //     test-only counters. The retired head takes a release CAS from every
    //     operation that unlinks a dead run: the erase path, the writer walks
    //     of insert() and erase(), and the split winner's cleanup pass (which
    //     any operation may run, contains() included). On the free line that
    //     CAS and the pop CAS would alternate ownership of one line between an
    //     inserting and an erasing thread whose numbers collide modulo the
    //     shard count. The retire counters are bumped right before the push
    //     CAS, by the thread about to own the line; the split counters and
    //     the four loss counters are bumped by a thread that may not own it
    //     (a loss is a CAS that failed, with no push to follow), at sites
    //     that run far less often than pops. None of them is on the free
    //     line.
    struct Shard {
        // Nodes are appended block-by-block and never destructed until the set
        // is destroyed; a slot's address is stable for the life of the set.
        Container<Node> nodes;
        // Free list: the encoded top node address and pop counter (see the
        // FREE-LIST HEAD encoding). Popped by alloc_node() with a relaxed CAS;
        // pushed and reset by reclaim() only.
        alignas(64) std::atomic<word_t> free_head{FREE_EMPTY};
        // Limbo list: the bare address of the top node, or EMPTY; the nodes
        // link through their `link` words. Pushed with a relaxed CAS by the
        // shard's threads, drained by reclaim(), read by the test-only sweeps.
        std::atomic<word_t> limbo_head{EMPTY};
        // Retired list: the bare address of the top retired node, or EMPTY;
        // the nodes link through their `retire_link` words, never through
        // `link` (a retired node keeps its chain link: see Node). LIFO and
        // push-only between two sequesters: push_retired() pushes a run with
        // one RELEASE CAS on behalf of the thread that unlinked it (its own
        // shard, see there), sequester_retired() exchanges the head to EMPTY
        // at a quiescent point, and nothing pops. Push-only makes the CAS
        // ABA-free for the same reason the limbo push is. The release on every
        // push means an ACQUIRE load of this head is a happens-before boundary
        // for the whole list below the value loaded: the push whose value was
        // loaded is a read-modify-write and so lies in the release sequence
        // of every push below it (RECLAMATION in the class overview), and a
        // reader that acquires the head synchronizes with all of them and
        // sees every unlink that preceded any of those pushes.
        alignas(64) std::atomic<word_t> retired_head{EMPTY};
        // The drain's private slot: the retired list sequester_retired() took
        // from retired_head, held between the two drain phases for
        // reclaim_sequestered() to deal out. EMPTY outside a drain. Owned by
        // the drainer alone (a plain word: both phases run at a quiescent
        // point), it exists so that the sequester is one exchange per shard
        // and no allocation, and so that the drained set is fixed before any
        // node of it is dealt (see sequester_retired()).
        word_t sequestered{EMPTY};
        // Test-only counters, each bumped with a relaxed fetch_add by the
        // thread whose operation took the counted step (so they are exact
        // when read at a quiescent point, and a snapshot otherwise); summed
        // over the shards by get_internal_counters(). Always compiled: the
        // benchmark reads them in its NDEBUG build, and their cost is one
        // relaxed increment per counted event, on a line the thread already
        // owns for the retire counters (bumped just before the push CAS) and
        // on a line it may not own for the split and loss counters (see the
        // layout comment above).
        std::atomic<size_t> retire_runs{0};             // push_retired() calls: one per unlinked run
        std::atomic<size_t> retire_nodes{0};            // nodes those runs held in total
        // Lost one-shot unlink CASes, by site and by kind. A loss is a PEER
        // LOSS when the failure value already holds the successor this thread
        // computed: another thread bypassed the same run, and nothing is left
        // behind. Every other loss (the word gained a prepended node, a seal
        // level, a tag, or a peer's bypass of a run of a different extent) may
        // leave dead nodes reachable, so `failures - peer_losses` bounds the
        // stragglers a site produced; the total alone measures contention on
        // the word, not stragglers.
        std::atomic<size_t> cleanup_cas_failures{0};    // lost CASes of a split winner's cleanup pass (cleanup_parent()), all kinds
        std::atomic<size_t> cleanup_cas_peer_losses{0}; // of those, peer losses
        std::atomic<size_t> walk_cas_failures{0};       // lost CASes of insert()'s and erase()'s walks (step_over()), all kinds
        std::atomic<size_t> walk_cas_peer_losses{0};    // of those, peer losses
        std::atomic<size_t> split_attempts{0};          // split_bucket() calls that sealed a parent and walked it, recursive parent splits included
        std::atomic<size_t> splits_published{0};        // of those, the ones whose publishing CAS won
    }; // struct Shard

    // The node arena: arena_mask_ + 1 (a power of two) shards. A thread
    // allocates from shards_[thread_number & arena_mask_]. Fixed at
    // construction; the array itself is never resized.
    std::unique_ptr<Shard[]> shards_;
    size_t arena_mask_;

    // The current logical size (number of buckets) of the hash table. Always a power of 2.
    std::atomic<size_t> table_size_;

    // A SpinLock used to serialize table resizes via the Double-Checked Locking Pattern.
    // Only one thread can expand the buckets_ array at a time.
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
    // append 156 uncounted nodes before its next batch. The resize hint in
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
    // The address field of a free-list head word (see the FREE-LIST HEAD
    // encoding): the top node's address, or FREE_EMPTY.
    static constexpr word_t free_addr_of(word_t head) { return (head << 9) >> 6; }
    // The free-list head word for top node address `addr` (a bare node
    // address or EMPTY) with the counter bits of `count_bits` (a head word
    // whose address field is ignored).
    static constexpr word_t free_word(word_t addr, word_t count_bits) { return (addr >> 3) | (count_bits & FREE_COUNT_MASK); }

    // The arena shard the calling thread allocates from and pushes limbo onto.
    Shard& my_shard() { return shards_[concurrent_hash_rcu_detail::thread_number() & arena_mask_]; }

    // Push the private chain first..last (linked through their link words,
    // nothing else references them) onto `shard`'s limbo list, where
    // reclaim() will find it. Treiber push; the list is push-only until a
    // quiescent point, so a CAS that finds the head unchanged is safe (no
    // ABA: nothing pops). Relaxed throughout: the pushers exchange nothing
    // but the head word itself (each links its own private tail), and the
    // only readers of the list, reclaim() and the test-only sweeps, are
    // ordered after every push by the quiescence precondition. Weak CAS: a
    // retry loop anyway.
    static void push_limbo(Shard& shard, Node* first, Node* last) {
        word_t old_head = shard.limbo_head.load(std::memory_order_relaxed);
        do {
            last->link.store(old_head, std::memory_order_relaxed);
        } while (!shard.limbo_head.compare_exchange_weak(old_head, word_of(first), std::memory_order_relaxed, std::memory_order_relaxed));
    } // push_limbo()

    // Retire the run first..last of dead nodes that the calling operation has
    // just unlinked from its chain: push it, as one unit, onto `shard`'s
    // retired list, where reclaim() (through sequester_retired() and
    // reclaim_sequestered()) will find it. The ONE entry point to the retired
    // lists.
    // Preconditions: every node of the run is dead and TAGGED (MARKED or
    // FROZEN; asserted), the run is the exact set of nodes the caller's OWN
    // successful unlink CAS bypassed (a loser retires nothing: the winner
    // retires the same nodes, and a node pushed twice makes a cycle in the
    // list), the run's nodes are consecutive in the chain, first through
    // last, through their immutable tagged `link` words, and `shard` is the
    // caller's own shard (my_shard()). It must be the caller's own shard
    // because that is the only shard a node can be attributed to without a
    // search: a Node carries no shard identity (its address is stable, but
    // nothing maps it back to the deque it lives in), and the retired list is
    // only a holding bag until reclaim() deals its nodes round-robin over
    // every shard's free list, so the shard a node is retired to has no
    // effect on where it is reused; what it does affect is contention, and
    // the caller's own retired line is the one its thread already owns.
    // Steps: (1) link the run privately through the retire_link words,
    // walking the run's own chain links from first to last. This happens
    // AFTER the caller's unlink CAS won, never while recording the run: a
    // recorder that then loses its CAS would be writing the retire_link of a
    // node a concurrent winner may already have pushed, with a value from a
    // run of a different extent (a FROZEN node's child can publish between
    // two recorders' walks), and corrupt the list. The chain links are
    // reloaded relaxed: the caller loaded each one with acquire when it
    // recorded the run, or installed it by its own CAS (the first node's
    // link, for an eraser's own tombstone), a tagged link is never written
    // by a concurrent operation, and coherence returns the same value to
    // this thread. (2) The push_limbo() loop on retired_head, with RELEASE on
    // success: the push is the one point that orders "this node is
    // unreachable" before whatever later acquires the head (the quiescent
    // reclaim() needs no ordering, but a generation bump that snapshots the
    // heads with an acquire does: the SNAPSHOT form under RECLAMATION in the
    // class overview, and Shard::retired_head).
    // Failure is relaxed: the value that failed is only the next expected
    // value. Weak CAS: a retry loop anyway, and each failure is another
    // thread's completed push or a spurious failure (a plain retry), so the
    // loop is lock-free. The retire_link stores are relaxed: nothing reads
    // them concurrently. The counters are bumped before the CAS, on the line
    // the CAS is about to own.
    static void push_retired(Shard& shard, Node* first, Node* last) {
        size_t count = 1;
        for (Node* node = first; node != last; ++count) {   // step 1: link the run privately
            const word_t link = node->link.load(std::memory_order_relaxed);
            assert((link & (MARK_BIT | FROZEN_BIT)) != 0 && "retiring a node that is not tagged: it is not dead");
            assert(addr_of(link) != EMPTY && "the run ends before `last`: first..last is not a chain");
            Node* next = node_of(link);
            node->retire_link.store(word_of(next), std::memory_order_relaxed);
            node = next;
        } // link the run privately
        assert((last->link.load(std::memory_order_relaxed) & (MARK_BIT | FROZEN_BIT)) != 0 && "retiring a node that is not tagged: it is not dead");
        shard.retire_runs.fetch_add(1, std::memory_order_relaxed);
        shard.retire_nodes.fetch_add(count, std::memory_order_relaxed);
        word_t old_head = shard.retired_head.load(std::memory_order_relaxed);
        do {   // step 2: the push
            last->retire_link.store(old_head, std::memory_order_relaxed);
        } while (!shard.retired_head.compare_exchange_weak(old_head, word_of(first), std::memory_order_release, std::memory_order_relaxed));
    } // push_retired()

    // The NEXT CHILD of bucket j on the path of a key with hash `h`: the one
    // bucket the key moves to when the table doubles past j, or j itself if
    // the key never leaves j. Every node in chain j has a key whose hash
    // agrees with j on j's own bits, h & own_mask == j where own_mask has
    // bit_width(j) bits (it was published into j at a table size whose mask
    // includes those bits, or copied into j by j's split under exactly that
    // mask); the assert() checks it. From j the key can move only when the
    // table doubles past the LOWEST bit of h above own_mask, call it b: at
    // every doubling below 2^b the key stays in j, at 2^(b+1) buckets it
    // moves to child = j | 2^b, and no other child of j ever selects it (a
    // child j + 2^m with m != b needs bit m of h set, which fails for m < b,
    // and bit b clear, which fails for m > b; see split_bucket() step 2). So
    // "the key maps elsewhere under a table size ts" is exactly child < ts.
    static size_t next_child(size_t h, size_t j) {
        const size_t own_mask = (size_t{1} << std::bit_width(j)) - 1;
        assert((h & own_mask) == j && "a node's key does not hash to the chain it is in: Hash is not a function of the key");
        const size_t moving_bits = h & ~own_mask;
        if (moving_bits == 0) return j;   // the key never leaves bucket j
        return j | (size_t{1} << std::countr_zero(moving_bits));
    } // next_child()

    // CONCURRENT deadness: may a concurrent operation bypass `node` (in the
    // chain of bucket j, link value `link` as loaded by the caller, at the
    // table size `ts` the caller loaded from table_size_)? Dead iff MARKED,
    // or FROZEN with its next child (next_child()) below `ts` AND published
    // (an ACQUIRE load of the child head, channel 1: a caller that goes on
    // to read the child chain is ordered after its publication; the edge a
    // stale reader needs after a bypass is carried by the caller's READ of
    // `ts`, through read-read coherence, channel 5 -- the acquire on that
    // load is channel 2's, for indexing buckets_[child] at all). An untagged
    // node is never dead to a concurrent operation, whatever its child's
    // state: every node that reclaim()'s quiescent rule (is_dead()) calls
    // dead is tagged (a tombstone by its mark, a superseded copy by the
    // freeze that precedes every copy), and only a tagged node has an
    // immutable link, which the one-CAS bypass of a run depends on. A FROZEN
    // node whose child is unpublished (a splitter between its freeze and its
    // publish, stalled or thrown) is NOT dead: it is still the key's
    // authoritative node (INVARIANT), and bypassing it would lose the key
    // (P9). A FROZEN node whose child is not below `ts` cannot be decided by
    // this caller (its `ts` is stale for that doubling); it is not dead to
    // it. Concurrent-dead implies quiescent-dead. Calls Hash{} on every
    // FROZEN node it is asked about (the child index needs the hash), and
    // only there; MARKED nodes and live nodes are decided from the link
    // alone. A FROZEN node that ENDS a run (not dead to the run walk) is
    // asked about again by the walk that continues at it, so it is hashed
    // twice (PINNED ORDERS, H5').
    bool is_dead_now(const Node* node, word_t link, size_t j, size_t ts) const {
        if (link & MARK_BIT) return true;
        if (!(link & FROZEN_BIT)) return false;
        const size_t child = next_child(Hash{}(node->value), j);   // the Hash{} per FROZEN node of H5, H5', H6 (see PINNED ORDERS for the exact count)
        assert(child != j && "a FROZEN node's key never leaves its bucket: it could not have been copied out");
        return child < ts && addr_of(buckets_[child].load(std::memory_order_acquire)) != UNINITIALIZED;
    } // is_dead_now()

    // The state of a WRITER WALK over one chain: the pred word a run of dead
    // nodes would be bypassed through, and whether this thread may CAS it. A
    // walk starts with pred = the bucket head (its full word, level included)
    // and advances pred onto every LIVE node it passes (the node's untagged
    // link as loaded, which is the CAS expected value); a tagged node never
    // becomes pred, so between pred and the walk's current node there are
    // only tagged nodes the walk could not or did not bypass. pred_ok is
    // false after a CAS on pred lost: pred_word is then stale and a further
    // CAS on it would be wasted (the one-shot sites never retry); it becomes
    // true again at the next live node. head_word is the current head value
    // as this thread knows it: the value loaded, or the value its OWN
    // successful head unlink installed (the own-CAS rule, CAS VALUE RULES
    // before contains(); insert()'s publish CAS expects it, and keeps the
    // level it carries).
    struct WalkState {
        std::atomic<word_t>* pred;    // the word holding the address the walk follows next
        word_t pred_word;             // its value as loaded, or as this thread's own CAS installed it
        bool pred_ok;                 // no CAS on pred has been lost since pred_word was loaded/installed
        std::atomic<word_t>* head_slot;   // the bucket head this walk started from
        word_t head_word;             // the head's value as loaded, or as this thread's own CAS installed it

        // The state at the start of a walk from bucket head `slot`, whose
        // value `word` the caller has just loaded with acquire: the head is
        // the first pred, usable, and the head value as known.
        WalkState(std::atomic<word_t>* slot, word_t word)
            : pred(slot), pred_word(word), pred_ok(true), head_slot(slot), head_word(word) {}
    }; // struct WalkState

    // What unlink_run() reports: whether the CAS won, and the bare address
    // after the run (the first non-dead node, or EMPTY), where the caller's
    // walk continues whether or not the CAS won.
    struct Unlink {
        bool won;
        word_t succ;
    }; // struct Unlink

    // Which site calls the unlink primitive: decides which of the shard's
    // test-only loss counters a lost CAS is charged to (none for the eraser's
    // self-unlink, whose losses are bounded at two per call and whose
    // residue is the tombstone itself).
    enum class UnlinkSite { walk, cleanup, self_unlink };

    // THE UNLINK PRIMITIVE. Bypass the maximal run of consecutive dead nodes
    // that starts at `first` (link value `first_link` as loaded with acquire
    // by the caller, or as installed by the caller's own mark CAS) with ONE
    // strong CAS on the predecessor word w.pred, whose value w.pred_word
    // (loaded with acquire, or installed by this thread's own earlier unlink
    // CAS on that word -- coherence makes either the current value as this
    // thread knows it; untagged; address == first) is the expected value,
    // never a reload. Preconditions: w.pred_ok; `first`
    // is dead by is_dead_now() at the caller's `ts`.
    // The run walk follows the dead nodes' IMMUTABLE tagged links (acquire
    // loads: each is load-bearing for the edge of an EARLIER unlinker whose
    // value a tag CAS extended, channel 5 in the class overview) and stops at
    // the first node that is not dead, S, or at EMPTY; the new value is S's
    // address with the pred word's own tag bits (a head keeps its level; a
    // link pred is untagged).
    // The CAS is release on success (the channel-5 edge: a thread that
    // acquires the new value is ordered after this unlink, after this
    // thread's read of the table size at which it decided the run dead, and
    // after every earlier unlink whose edge this thread's walk acquired) and
    // acquire on failure (the pinned order; no site dereferences or follows
    // the failed value -- the loser only compares its address for the loss
    // counters, which needs no happens-before -- so the failure acquire
    // carries nothing in this class).
    // The WINNER retires EXACTLY the recorded run first..last and nothing
    // else: deadness is never re-evaluated after the CAS, because a FROZEN
    // node's child can publish between the run walk and the CAS, which would
    // make S dead and a re-evaluation would retire S while S is reachable. A
    // loser retires nothing (the winner retires the same nodes) and marks
    // pred stale; the loss is charged to the site's counters (Shard), as a
    // PEER LOSS when the failure value's address is S, the successor this
    // thread computed: the word was rewritten by a bypass of exactly this
    // run, so no dead node is left behind by the loss. The only writes: the
    // one CAS, the retirement and the counters.
    // Why one CAS on an untagged pred cannot lose a node or retire one twice
    // (the Harris argument): two concurrent bypasses can only overlap through
    // a node both treat the same way, and a tagged link is never a CAS
    // target, so a run is always cut out at a LIVE word whose value the
    // winner read; the loser's CAS fails because that word changed, and the
    // nodes of the loser's run are retired by whoever won the word they hang
    // from. Why the CAS cannot succeed on a stale view: within one period no
    // address is published twice (RECLAMATION), so the expected value names
    // the same node in the same chain for the whole operation.
    Unlink unlink_run(WalkState& w, Node* first, word_t first_link, size_t j, size_t ts, UnlinkSite site) {
        assert(w.pred_ok && addr_of(w.pred_word) == word_of(first) && "the pred word does not lead to the run");
        assert((w.pred_word & (MARK_BIT | FROZEN_BIT)) == 0 && "unlink CAS on a tagged word: only a LIVE link or a head is a CAS target");
        Node* last = first;
        word_t link = first_link;
        while (addr_of(link) != EMPTY) {   // record the run: extend it while the next node is dead
            Node* next = node_of(link);
            const word_t next_link = next->link.load(std::memory_order_acquire);
            if (!is_dead_now(next, next_link, j, ts)) break;
            last = next;
            link = next_link;
        } // record the run
        const word_t succ = addr_of(link);
        const word_t installed = (w.pred_word & ~PTR_MASK) | succ;
        word_t expected = w.pred_word;
        if (w.pred->compare_exchange_strong(expected, installed, std::memory_order_release, std::memory_order_acquire)) {
            w.pred_word = installed;                         // the own-CAS rule: the next expected value is what we installed
            if (w.pred == w.head_slot) w.head_word = installed;
            push_retired(my_shard(), first, last);           // exactly the recorded run
            return {true, succ};
        } // if the unlink won
        w.pred_ok = false;                                   // the word changed under us: one shot, never retried here
        if (site != UnlinkSite::self_unlink) {   // test-only counters: the loss, and whether a peer bypassed this very run
            Shard& shard = my_shard();
            const bool cleanup = site == UnlinkSite::cleanup;
            std::atomic<size_t>& failures = cleanup ? shard.cleanup_cas_failures : shard.walk_cas_failures;
            std::atomic<size_t>& peer_losses = cleanup ? shard.cleanup_cas_peer_losses : shard.walk_cas_peer_losses;
            failures.fetch_add(1, std::memory_order_relaxed);
            if (addr_of(expected) == succ) peer_losses.fetch_add(1, std::memory_order_relaxed);
        } // if a counted site
        return {false, succ};
    } // unlink_run()

    // One step of a writer walk past `node` (link value `link` as loaded with
    // acquire), the shared shape of insert()'s and erase()'s walks and of the
    // split winner's cleanup pass. A LIVE node becomes the new pred and is
    // passed. A tagged node is passed through its immutable link (EXIT), with
    // ONE unlink attempt on its run if, and only if, it is dead now AND pred
    // leads straight to it (w.pred_ok, and pred_word's address is this node:
    // a tagged node that is not dead between pred and here, e.g. FROZEN with
    // an unpublished child, makes pred useless for this run, and a lost CAS
    // on pred has made pred_word stale). Never retried: a lost CAS is another
    // thread's completed step on this word -- possibly a peer's bypass of
    // this very run, which leaves nothing behind -- and whatever dead nodes it
    // leaves are STRAGGLERS (WHY COPY in the class overview). Returns the
    // bare address where the walk continues: after the run in either case;
    // after a lost CAS the dead run may already be bypassed by someone else,
    // and the exit still lands in the current chain (EXIT in the class
    // overview). `site` names the caller for the loss counters.
    word_t step_over(WalkState& w, Node* node, word_t link, size_t j, size_t ts, UnlinkSite site) {
        if ((link & (MARK_BIT | FROZEN_BIT)) == 0) {   // live: the new pred
            w.pred = &node->link;
            w.pred_word = link;
            w.pred_ok = true;
            return addr_of(link);
        } // if live
        if (w.pred_ok && addr_of(w.pred_word) == word_of(node) && is_dead_now(node, link, j, ts)) {
            return unlink_run(w, node, link, j, ts, site).succ;
        } // if dead and bypassable
        return addr_of(link);   // tagged, not dead or not bypassable: exit through it; pred unchanged
    } // step_over()

    // The split winner's CLEANUP PASS over the parent chain, after its
    // publishing CAS (P9: a node FROZEN for j is bypassed only once j is
    // published). Called by split_bucket(j) with `parent` = j's parent; j and
    // 2N below are split_bucket()'s. One writer walk with one unlink attempt
    // per dead run (never retried; a lost CAS is charged to
    // cleanup_cas_failures, and to cleanup_cas_peer_losses when a peer -- a
    // concurrent writer walk, or the cleanup of another split of this parent
    // -- bypassed the same run, in which case nothing is left; otherwise the
    // run stays for a later walk, cleanup or reclaim()). Dead here, besides
    // any tombstone, is every node FROZEN for a PUBLISHED child of the
    // parent: the copies this split superseded (their child is j, just
    // published) and those of earlier splits of other children that are still
    // here; so the cleanup of ANY child's split collects the whole parent
    // chain's dead runs, with no hit to stop it.
    // The table size is acquired here, once: split_bucket()'s precondition
    // is that its caller acquired, on this thread, a table size above j, so
    // this load returns at least 2N (table_size_ is monotone and the two
    // loads are sequenced), hence j < ts and the FROZEN-for-j nodes are
    // decidable; nodes FROZEN for a later doubling than this `ts` covers are
    // left (not dead to this walk). The head is reloaded with acquire: nodes
    // prepended by threads at table size >= 2N since the seal are live and in
    // front of the snapshot. Calls Hash{} on every FROZEN node it passes
    // (H6). Exceptions: see EXCEPTIONS in the class overview (W4).
    void cleanup_parent(size_t parent) {
        const size_t ts = table_size_.load(std::memory_order_acquire);
        WalkState w(&buckets_[parent], buckets_[parent].load(std::memory_order_acquire));
        word_t curr = addr_of(w.pred_word);
        while (addr_of(curr) != EMPTY) {   // walk the parent chain
            Node* node = node_of(curr);
            const word_t link = node->link.load(std::memory_order_acquire);
            curr = step_over(w, node, link, parent, ts, UnlinkSite::cleanup);
        } // walk the parent chain
    } // cleanup_parent()

    // erase()'s EAGER SELF-UNLINK of the node `x` it has just marked (x's link
    // is `x_link`, the value the mark CAS installed: the own-CAS rule, no
    // reload). `w` is the walk state at x: pred is the last live word before
    // x, and between pred and x there are only tagged nodes the walk did not
    // bypass. At most TWO unlink CASes, then give up; the tombstone is then a
    // STRAGGLER (WHY COPY in the class overview), and the return value of
    // erase() does not depend on any of this.
    //   Attempt 1: if pred is usable (no lost CAS on it), the run bypassed
    //   starts at the node pred leads to, `first`: x itself, or a tagged node
    //   before x. If `first` is tagged and NOT dead (FROZEN with an
    //   unpublished child), x sits behind a node that is not a CAS target and
    //   no attempt can reach it: GIVE UP (exit 1, "tagged pred"). Otherwise
    //   one unlink_run() from `first`; it is the run walk's business how far
    //   the run extends (to x and the dead nodes behind it, usually; if a
    //   not-dead tagged node sits between `first` and x the run ends there,
    //   and x stays behind a tagged, not-dead node, which is exit 1's
    //   situation: a restart would only find it again, so a WON CAS ends the
    //   call whether or not x was in the run). The run walk hashes every
    //   FROZEN node it passes (H5).
    //   Restart: when attempt 1 was not made (a lost CAS had made pred stale)
    //   or was lost: re-walk bucket j from its head, ACQUIRE loads, locating
    //   x by identity and tracking pred as the walk does (live nodes only);
    //   the key is NOT re-hashed (x is in bucket j: the mark succeeded there).
    //   No opportunistic unlinks on this walk: the one CAS it may lead to is
    //   attempt 2. If x is no longer reachable, someone else bypassed it:
    //   done. Else attempt 2 exactly as attempt 1, and whatever its outcome,
    //   GIVE UP (exit 2, "CAS lost twice": two concurrent changes to one chain
    //   inside one erase, the first of them possibly the one that cost the
    //   walk its own CAS on the pred word, so that attempt 1 was skipped).
    //   Neither loss is counted: the residue of a give-up is the tombstone
    //   itself, visible in the accounting as a reachable dead node.
    // Exceptions: see EXCEPTIONS in the class overview (P5).
    void self_unlink(WalkState& w, Node* x, word_t x_link, size_t j, size_t ts) {
        bool restarted = false;
        while (true) {   // at most two rounds
            if (w.pred_ok) {
                Node* first = node_of(w.pred_word);   // never EMPTY: pred leads to x through tagged nodes at most
                const word_t first_link = first == x ? x_link : first->link.load(std::memory_order_acquire);
                if (first != x && !is_dead_now(first, first_link, j, ts)) return;   // exit 1: tagged pred, not dead: no CAS target leads to x
                if (unlink_run(w, first, first_link, j, ts, UnlinkSite::self_unlink).won) return;
            } // if pred is usable
            if (restarted) return;   // exit 2: the CAS was lost on both attempts
            restarted = true;
            w = WalkState(&buckets_[j], buckets_[j].load(std::memory_order_acquire));   // restart: same bucket, no re-hash
            word_t curr = addr_of(w.pred_word);
            while (addr_of(curr) != EMPTY && curr != word_of(x)) {   // locate x by identity, tracking the last live pred
                Node* node = node_of(curr);
                const word_t link = node->link.load(std::memory_order_acquire);
                if ((link & (MARK_BIT | FROZEN_BIT)) == 0) {
                    w.pred = &node->link;
                    w.pred_word = link;
                }
                curr = addr_of(link);
            } // locate x
            if (addr_of(curr) == EMPTY) return;   // x is already unreachable: bypassed by another operation
        } // at most two rounds
    } // self_unlink()

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
        Node* node = &shard.nodes[idx];   // the deque's index is used here, once, and by the test-only sweeps
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
    // Precondition: the caller has acquired, on this thread, a table size
    // above j (it indexed bucket j under that size, or it is this function
    // splitting j's parent, whose caller had); the cleanup pass (step 4)
    // relies on it to decide the nodes frozen for j.
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
    // The split runs in three steps, each of which hands off through one word,
    // and the winner then runs a fourth:
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
    //      The freeze runs for BOTH AllowDelete values. With AllowDelete ==
    //      false there are no tombstones and nothing races with the copy, so
    //      the tag decides nothing for the key; it is set anyway because a
    //      superseded parent copy must be TAGGED to be dead: the tag is what
    //      makes its link immutable, and an immutable link is what lets a run
    //      of dead nodes be bypassed by one CAS on an untagged pred word
    //      without a lost unlink or a double retirement (two bypasses through
    //      a node that one of them treats as live). Every dead node in a
    //      chain, tombstone or superseded copy, is therefore tagged.
    //   3. Publish the subchain with one CAS on bucket j's head. A subchain
    //      that loses the CAS goes to the limbo list, whole.
    //   4. The WINNER's cleanup pass (cleanup_parent()): one walk of the
    //      parent chain that bypasses, with one CAS per run and no retry,
    //      every run of dead nodes it meets -- the copies this split just
    //      superseded, foremost. Losers do nothing: the winner's pass covers
    //      the same nodes, and what the pass loses to contention is a
    //      STRAGGLER (WHY COPY in the class overview). The pass runs AFTER the
    //      publication (P9 in the class overview): until j is published, a
    //      node FROZEN for j is the key's only reachable node.
    // Exceptions: see EXCEPTIONS in the class overview.
    // The parent chain IS relinked by concurrent operations, by unlink CASes
    // on LIVE links and heads only (unlink_run()); an operation that holds a
    // node's address can still exit the node through its tagged, immutable
    // link (EXIT in the class overview), which is what keeps already-observed
    // node addresses usable for the whole of an operation.
    // Pinned orders: H3 (the snapshot walk's link load, Hash{}, freeze CAS)
    // and H6 (the cleanup pass's Hash{} calls) are this function's; see
    // PINNED ORDERS before contains().
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
        // acquire: the head we get back was released by an inserter, another
        // sealer or an unlink through the head, and we are going to walk the
        // chain it points to (channel 1). Weak CAS: we are in a retry loop
        // anyway.
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
        my_shard().split_attempts.fetch_add(1, std::memory_order_relaxed);   // test-only counter

        try {
            while (addr_of(curr) != EMPTY) {   // traverse parent chain
                Node* node = node_of(curr);
                // A reference is safe: a published node's value is never written,
                // and its memory is reused only at a quiescent point, whether or
                // not the node is bypassed during this call (EXIT).
                const T& val = node->value;
                word_t next_raw = node->link.load(std::memory_order_acquire);
                if (!(next_raw & MARK_BIT)) {          // skip logically deleted nodes
                    if ((Hash{}(val) & mask) == j) {   // key belongs to bucket j now (H3: hashed after the link load)
                        // FREEZE before copying, for both AllowDelete values (see
                        // step 2 above). Expected value: the live link loaded
                        // BEFORE the hash (H3), never a reload. Strong CAS, so
                        // failure means the link really changed, in one of three
                        // ways: it gained MARK (an eraser won: skip the node),
                        // it gained FROZEN (another splitter of j won: copy it,
                        // as that splitter does), or it is still LIVE with a new
                        // successor address (a concurrent unlink bypassed a dead
                        // run behind this node). The third is RETRIED with the
                        // failure value as the new expected value: the loop ends
                        // when the node is tagged (by us or by someone else), and
                        // every pass through it is another thread's completed
                        // unlink, so it is lock-free. Without the retry a live
                        // failure would fall through and copy an UNFROZEN node:
                        // two live nodes for one key, and a stale eraser could
                        // mark the parent and return true while the child keeps
                        // the key. The retried freeze installs the CURRENT
                        // successor with the tag, so a later run walk through
                        // this node finds whatever is live behind it. Success is
                        // release: a stale eraser that acquires the FROZEN link is
                        // thereby ordered after our caller's acquire of
                        // table_size_ >= 2N (channel 4). Failure is relaxed: every
                        // outcome is decided by the returned bits alone, we need
                        // nothing else its writer did.
                        while (!(next_raw & (MARK_BIT | FROZEN_BIT))) {   // freeze loop: retried while the failure value is live
                            if (node->link.compare_exchange_strong(next_raw, next_raw | FROZEN_BIT, std::memory_order_release, std::memory_order_relaxed)) {
                                next_raw |= FROZEN_BIT;
                            }
                        } // freeze loop
                        if (!(next_raw & MARK_BIT)) {
                            // Prepend a fresh live copy to the child subchain.
                            Node* copy = alloc_node(val, new_subchain_head);
                            if (new_subchain_tail == nullptr) new_subchain_tail = copy;
                            new_subchain_head = word_of(copy);
                        }
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
            my_shard().splits_published.fetch_add(1, std::memory_order_relaxed);   // test-only counter
            cleanup_parent(parent);   // step 4, after the publication (P9); H6 inside
        } else {
            // CAS failed: subchain abandoned (unpublished, memory waste until reclaim() collects it).
            if (new_subchain_head != EMPTY) push_limbo(my_shard(), node_of(new_subchain_head), new_subchain_tail);
        } // if another thread's split of j won
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
    ConcurrentResizableHashSetRCU(size_t initial_capacity = 4, size_t arena_shards = 0) {
        if (arena_shards == 0) arena_shards = std::thread::hardware_concurrency();
        if (arena_shards == 0) arena_shards = 1;   // hardware_concurrency() may report 0
        // Cap before rounding: bit_ceil() of a value above 2^63 has no
        // representable result, and no machine has that many threads anyway.
        if (arena_shards > MAX_ARENA_SHARDS) arena_shards = MAX_ARENA_SHARDS;
        arena_shards = std::bit_ceil(arena_shards);
        shards_ = std::make_unique<Shard[]>(arena_shards);
        arena_mask_ = arena_shards - 1;
        if (initial_capacity < 4) initial_capacity = 4;
        initial_capacity = std::bit_ceil(initial_capacity);
        buckets_.resize(initial_capacity);
        for (size_t i = 0; i < initial_capacity; ++i) {
            buckets_[i].store(EMPTY, std::memory_order_relaxed);
        }
        table_size_.store(initial_capacity, std::memory_order_release);
    }

    // ------------------------------------------------------------------------
    // PINNED ORDERS. The order in which an operation loads a word, hashes or
    // compares a key, and performs its deciding CAS is part of this class's
    // correctness argument AND of its test contract: a test can suspend a
    // thread inside Hash{} or operator== (a hook in the key type) and run a
    // concurrent operation in that window, so every hook test rests on the
    // exact call site it hooks, counted from the start of the operation. The
    // orders are named H1-H6 and marked at their sites. Changing an order
    // changes which window a test opens: name the affected tests.
    //   H1  contains(), insert(), erase(): Hash{}(key) is called between the
    //       table_size_ load and the bucket-head load, once per pass or
    //       attempt (every retry re-hashes, since it reloads table_size_).
    //   H2  erase(): for the matching node, the link load, then operator==,
    //       then the mark CAS, whose expected value is the link loaded BEFORE
    //       the comparison, never a reload.
    //   H3  split_bucket(): for every node of the snapshot, the link load,
    //       then Hash{}(val), then the freeze CAS whose expected value is
    //       that earlier load, never a reload.
    //   H4  contains(): a node's link is loaded before its key is compared,
    //       and the comparison precedes the next node's link load; nothing
    //       is hashed on the walk.
    //   H5' the run unlinks of insert()'s and erase()'s walks (step_over()),
    //       every Hash{} through is_dead_now(), which hashes FROZEN nodes
    //       only: for every TAGGED node passed with a usable pred, Hash{} on
    //       it if it is FROZEN (the deadness test); then, if it is dead, the
    //       run walk calls Hash{} on each further FROZEN node it reaches,
    //       INCLUDING the node that ends the run when that node is FROZEN
    //       and not dead; then the run's one CAS. The walk then continues at
    //       the run's end node and, if it is tagged and the pred is usable,
    //       tests it again: a FROZEN run-ending node is hashed TWICE, once by
    //       the run walk and once by the step that passes it. Hits stop the
    //       walk before the step (a matching node is never hashed by the
    //       walk). A tagged node passed with an unusable pred is not hashed.
    //   H5  erase()'s own unlink (self_unlink()), after a successful mark:
    //       attempt 1 hashes the node the tracked pred leads to if that node
    //       is FROZEN and not x (the tagged-pred test: a FROZEN node the walk
    //       already tested once is hashed a second time here), then the run
    //       walk hashes as in H5' (x itself is MARKED, never hashed); the
    //       restart re-reads the head of the SAME bucket j, does NOT re-hash
    //       the key, and its locating walk hashes nothing; attempt 2 hashes
    //       as attempt 1.
    //   H6  the split winner's cleanup pass (cleanup_parent()), after its
    //       publishing CAS: as H5', over the parent chain from its head; the
    //       copies this split froze are hashed here after their H3 hash.
    // CAS VALUE RULES. Every deciding CAS is one strong CAS on one word with
    // the FULL expected value as loaded (a head with its level, a link with
    // its tags), loaded before the compare or hash that precedes the CAS,
    // or, on a retry of the mark or freeze CAS, the failure value of the
    // previous attempt; after an operation's OWN successful CAS on a word, its next expected
    // value for that word is the value it installed (a head unlink keeps the
    // level; a publish after an own head unlink expects the unlinked head).
    // This last rule is the OWN-CAS RULE.
    // ------------------------------------------------------------------------

    // Membership test. Lock-free on the fast path (a plain chain walk with no
    // atomic writes); it can, however, fall into split_bucket() -- which
    // allocates, possibly under its arena shard's lock -- if it lands on an
    // UNINITIALIZED bucket, so it is not lock-free/wait-free in general (see
    // the class overview).
    // Logically deleted nodes are skipped via the MARK_BIT test. The head's seal
    // level and a node's FROZEN bit are deliberately IGNORED here (they are only
    // masked off the address): contains()'s walk writes nothing (the split it
    // may fall into is a writer path, cleanup included, but decides nothing
    // about this call's answer), so a stale geometry cannot make it corrupt
    // anything, and its answers stay linearizable:
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
    //     doubling and always terminates. The reload is SAFETY-bearing, not
    //     only a progress device: a writer may have bypassed the key's FROZEN
    //     copy during our walk, after a later split published the copy's
    //     bucket, and then this reload returns the larger size (channel 5);
    //     the two halves of the miss argument under STALE GEOMETRY say when a
    //     miss is final.
    bool contains(const T& key) {
        size_t ts = table_size_.load(std::memory_order_acquire);
        while (true) {
            size_t j = Hash{}(key) & (ts - 1);   // H1: hashed between the table_size_ load and the head load
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
                // load serves the mark check and the advance. H4: the link load
                // precedes the comparison, and the comparison precedes the next
                // node's link load; nothing is hashed on the walk.
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
    // from its arena shard (alloc_node(): a lock-free free-list pop, or an
    // append under the shard's SpinLock), a bucket that is still UNINITIALIZED
    // is split first, and a successful insert may perform the DCLP-guarded
    // doubling when the node count exceeds twice the table size.
    // Postcondition of an UNCONTENDED call that returns normally (no other
    // operation on the set runs meanwhile; POSTCONDITIONS in the class
    // overview): no dead run stays reachable behind the bucket head or a live
    // node the walk passed. Under contention the walk's unlinks are best
    // effort.
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
                size_t j = Hash{}(key) & (ts - 1);   // H1: hashed between the table_size_ load and the head load, once per attempt
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
                // publishing CAS below. That validation needs no ordering from
                // channel 5: if the key's node was frozen and bypassed during our
                // walk, its freezer had sealed this head above our level first,
                // and the seal is a write to the very word our CAS expects
                // unchanged, so the CAS fails and the attempt restarts from the
                // table_size_ load. The failed CAS carries no ordering (its
                // failure order is relaxed), so that first reload may still
                // return the old size; the retry then acquires the sealed head,
                // its geometry check sends it round again, and the second reload
                // returns the larger size (channel 4).
                // The walk is a WRITER WALK (step_over()): every dead run it
                // passes gets one unlink attempt through the last live word
                // before it, never retried (H5': the run walk hashes each FROZEN
                // node of the run before the run's CAS). The walk stops at the
                // key, so only a miss walks the whole chain. After an own head
                // unlink the publishing CAS below expects the value that unlink
                // installed (w.head_word; the own-CAS rule), level included.
                bool exists = false;
                WalkState w(&buckets_[j], head);
                word_t curr = addr_of(head);
                while (addr_of(curr) != EMPTY) {   // walk bucket j's chain
                    Node* node = node_of(curr);
                    word_t check_curr = node->link.load(std::memory_order_acquire);
                    if (node->value == key && !(check_curr & MARK_BIT)) {
                        exists = true;
                        break;
                    }
                    curr = step_over(w, node, check_curr, j, ts, UnlinkSite::walk);
                } // walk chain
                head = w.head_word;
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
                // fails if another writer prepended a node, a splitter sealed the
                // bucket, or an unlink bypassed a run at the head since we read it
                // (an unlink of our own is accounted for: `head` is then the value
                // we installed); in every case loop and retry from the table_size_
                // load, reusing new_node. Failure is relaxed: the returned value is
                // not used.
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
    // of the key, across any number of concurrent resizes. A key that is absent
    // or already tombstoned yields false. After a successful mark the call
    // makes a bounded effort to take its tombstone out of the chain at once
    // (self_unlink(): one CAS on the pred word tracked during the walk, one
    // restart from the head, then give up); the result is unaffected unless
    // Hash{} throws there (EXCEPTIONS, P5), and a tombstone that stays is a
    // STRAGGLER (WHY COPY in the class overview). The walk itself is a writer
    // walk (step_over()): every dead run it passes behind a live word gets one
    // unlink attempt. Postconditions of an UNCONTENDED call that returns
    // normally (no other operation on the set runs meanwhile; POSTCONDITIONS
    // in the class overview): no dead run stays reachable behind the bucket
    // head or a live node the walk passed, and the tombstone is unreachable
    // unless a tagged, not-dead node lies between it and the last live word
    // before it.
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
    //   - our CAS failed and the node is still LIVE => a concurrent unlink
    //     bypassed a dead run behind it and changed the successor address; the
    //     mark is RETRIED with the failure value as the new expected value.
    //     Each retry follows another thread's completed unlink, so the loop is
    //     lock-free, and it ends when the link is tagged. Without the retry a
    //     live failure would fall through to the reload and report `false` for
    //     a key that is present.
    // A strong CAS on a published link can fail for no other reason: the
    // writers of a published link are this CAS, the freeze CAS (both from the
    // live value) and the unlink CAS (from a live value, to a live value with a
    // new address); insert()'s plain store targets a node that is still private.
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
            size_t j = Hash{}(key) & (ts - 1);   // H1: hashed between the table_size_ load and the head load, once per pass
            word_t head = buckets_[j].load(std::memory_order_acquire);   // the FULL word: a head unlink's expected value keeps the level
            if (addr_of(head) == UNINITIALIZED) {
                split_bucket(j);
                head = buckets_[j].load(std::memory_order_acquire);
            }

            // Set when this chain proved stale for the key (FROZEN node found):
            // a miss in it is then NOT authoritative even if table_size_ reads
            // unchanged.
            bool stale = false;
            // The writer walk's state (H5': each dead run passed gets one unlink
            // attempt, its FROZEN nodes hashed before the CAS); at the key, w
            // holds the predecessor the self-unlink's first attempt goes through.
            WalkState w(&buckets_[j], head);
            word_t curr = addr_of(head);
            while (addr_of(curr) != EMPTY) {   // walk bucket j's chain
                Node* node = node_of(curr);
                word_t check_curr = node->link.load(std::memory_order_acquire);
                if (node->value == key && !(check_curr & MARK_BIT)) {   // H2: the comparison follows the link load
                    // Live (or frozen) node of our key. If live, set MARK_BIT while
                    // keeping the same successor. H2: the expected value is
                    // check_curr as loaded BEFORE the comparison, never a reload;
                    // a retry's expected value is the failure value. Success is
                    // release: operations that acquire this link afterwards see
                    // the key as deleted. Failure is acquire, load-bearing on both
                    // failure paths: if the reason is FROZEN we go on to reload
                    // table_size_, and the acquire orders that reload after the
                    // freezer's view of the table (see "Retry bound" above); if the
                    // reason is an unlink (a live value with a new successor
                    // address), the failure value is the new expected value, the
                    // loop goes round, and the self-unlink after the retried mark
                    // starts its run at that successor and installs it through the
                    // pred word -- this failed CAS's read is the eraser's only
                    // acquire of the unlinker's write, and a reader that follows
                    // the eraser's bypass needs the unlinker's edge (channel 5).
                    while (!(check_curr & (MARK_BIT | FROZEN_BIT))) {   // mark loop: retried while the failure value is live
                        if (node->link.compare_exchange_strong(check_curr, check_curr | MARK_BIT, std::memory_order_release, std::memory_order_acquire)) {
                            // This CAS is THE deletion of the key. H5: the self-unlink
                            // follows, with the tombstone's link as installed.
                            self_unlink(w, node, check_curr | MARK_BIT, j, ts);
                            return true;
                        } // if this CAS marked the node
                    } // mark loop
                    // Not marked by us; check_curr holds the link's current value.
                    // FROZEN: stale geometry, retry in the current one. MARKED:
                    // another eraser deleted the key; report false below unless the
                    // table grew (then the retry may find and delete a newer
                    // insertion of the key, which is equally linearizable).
                    stale = (check_curr & FROZEN_BIT) != 0;
                    break;
                } // if found an unmarked node of our key
                curr = step_over(w, node, check_curr, j, ts, UnlinkSite::walk);
            } // walk chain

            size_t new_ts = table_size_.load(std::memory_order_acquire);
            if (new_ts == ts && !stale) {
                return false; // stable table, key absent or already tombstoned
            }
            ts = new_ts;      // table grew under us: retry in new geometry
        } // while (true)
    } // erase()

    // ------------------------------------------------------------------------
    // Quiescent reclamation. reclaim() and the test-only sweeps below share
    // one precondition, QUIESCENCE: no other call on this set is in progress
    // while the function runs, every earlier call happens-before it, and it
    // happens-before every later call (e.g. the caller joins or barriers the
    // worker threads before and starts or releases them after). Under
    // that precondition every atomic access here is relaxed: the values these
    // functions read were published to them, and the values reclaim() writes
    // are published to every later operation, by the caller's synchronization,
    // exactly as the constructor's writes are.
    // ------------------------------------------------------------------------

private:
    // Push `node` (unreachable, nothing else references it) onto `shard`'s
    // free list, keeping the head's counter bits. Quiescent (reclaim() only).
    static void push_free(Shard& shard, Node* node) {
        word_t head = shard.free_head.load(std::memory_order_relaxed);
        node->link.store(free_addr_of(head), std::memory_order_relaxed);
        shard.free_head.store(free_word(word_of(node), head), std::memory_order_relaxed);
    } // push_free()

    // The DRAIN of the retired lists, in two phases, both private and both
    // called by reclaim() alone, under its quiescence. The phases are
    // separate because a drain has a weaker true precondition than
    // quiescence, and the split is what a caller that wants to exploit it
    // needs. The precondition, exact, in happens-before terms (a retired node
    // is one that an operation's unlink CAS u bypassed and push_retired()
    // then pushed):
    //   (1) every operation that could have loaded a retired node's address
    //       has completed, and its completion happens-before the drain's
    //       START;
    //   (2) every operation that starts during or after the drain has each
    //       retired node's unlink CAS u happen-before its first bucket-head
    //       load (a thread that never synchronized after u could otherwise,
    //       in the C++ model, still load a pre-u head value and walk into the
    //       node); "arrange happens-before" means delivering this, e.g. by
    //       having new operations start after acquiring something the
    //       drain's starter released after the retirers' completions;
    //   (3) the drained set is FIXED at the START, which is the point after
    //       the drainer has exchanged EVERY shard's retired head into its
    //       private slot: operations resumed under (2) start after that
    //       point, so every later retire push follows every exchange in
    //       modification order, and no node retired after the start is in
    //       the drained set. Without (3) a per-shard drain could take a node
    //       retired onto shard 1 after shard 0 was exchanged while a resumed
    //       operation still holds that node.
    // reclaim()'s quiescence implies all three. What a caller that resumes
    // operations after the START would ALSO need, and this class does not
    // provide, is thread-safe free lists: reclaim_sequestered() pushes onto
    // the free lists, and a push concurrent with alloc_node()'s pops breaks
    // the pop's ABA argument (RECLAMATION in the class overview: a head value
    // a popper read can be restored only by a push, and pushes happen only
    // inside reclaim()); so in this class the two phases run only inside
    // reclaim(), and a drain concurrent with operations is not supported.

    // Phase 1, SEQUESTER: take every shard's retired list out of its head and
    // into the shard's private slot (one exchange per shard, no walk, no
    // allocation). Relaxed, as every access in reclaim() is: under
    // quiescence every push already happens-before this call. The last
    // exchange is the drain's START point of condition (3).
    void sequester_retired() {
        for (size_t s = 0; s <= arena_mask_; ++s) {
            Shard& shard = shards_[s];
            assert(shard.sequestered == EMPTY && "a sequestered list was never dealt out: reclaim_sequestered() did not run");
            shard.sequestered = shard.retired_head.exchange(EMPTY, std::memory_order_relaxed);
        } // loop over the shards
    } // sequester_retired()

    // Phase 2, RECLAIM: deal every sequestered node onto the free lists,
    // round-robin over the shards with the caller's cursor `deal` (reclaim()
    // shares one cursor across its limbo, retired and chain-walk sources so
    // that the free lists come out equally long whatever the source: a
    // single-threaded prefill retires every victim onto one shard, and
    // without the dealing that shard's threads alone would pop while the
    // others append). This is the ONLY place a retired node's `link` is
    // rewritten (push_free() makes it the free list's next pointer); until
    // here the link keeps the chain successor the node had when it died, so
    // an operation that reached the node before its unlink can exit through
    // it (condition (1) says no such operation remains). The node's
    // retire_link is left as it is: stale, never read again. Checks (live in
    // the sanitizer builds): every retired node is tagged, since only dead
    // nodes are retired and every dead node in a chain is tagged (a live node
    // here means a run walk overshot its run, or a node was retired on a
    // lost CAS); and the dealt count never exceeds the arena's slot count (a
    // longer list is a cycle from a double retirement, which would otherwise
    // hang this loop).
    void reclaim_sequestered(size_t& deal) {
#ifndef NDEBUG
        const size_t slots = get_internal_node_count();   // the drain bound (an assert-only shard-size sum)
        size_t count = 0;                                 // nodes dealt so far, for the cycle check
#endif
        for (size_t s = 0; s <= arena_mask_; ++s) {
            Shard& shard = shards_[s];
            word_t curr = shard.sequestered;
            shard.sequestered = EMPTY;
            while (addr_of(curr) != EMPTY) {   // walk the sequestered list
                Node* node = node_of(curr);
                assert((node->link.load(std::memory_order_relaxed) & (MARK_BIT | FROZEN_BIT)) != 0 && "a retired node is not tagged: a live node was retired");
#ifndef NDEBUG
                ++count;
                assert(count <= slots && "more retired nodes than arena slots: a retired list has a cycle (a node retired twice)");
#endif
                curr = node->retire_link.load(std::memory_order_relaxed);   // before push_free() rewrites the link; retire_link stays stale
                push_free(shards_[deal++ & arena_mask_], node);
            } // walk the sequestered list
        } // loop over the shards
    } // reclaim_sequestered()

    // The dead-node rule of reclaim(), for `node` with link value `link` in
    // the chain of published bucket `j`, at table size `ts`. Dead means: no
    // operation, present or future, can need this node, so it may leave the
    // chain. A node is dead iff
    //   (1) it is MARKED (a tombstone: contains() skips it, no split copies it,
    //       and an insert of the key prepends a fresh node); or
    //   (2) its key's NEXT CHILD bucket on its path out of j is published.
    // Rule (2): the child is next_child() (its comment derives it); "the key
    // maps elsewhere under the current mask" is exactly child < ts.
    // Why child published is the right test, for both AllowDelete values: the
    // split that published child SEALED j first and snapshotted the sealed
    // head, and an insert of this key into j can only have run at a table
    // size <= 2^b, whose level is below the seal, so it is in the snapshot or
    // it failed. Every splitter of child therefore saw this node and froze it
    // (for both AllowDelete values), and the winner copied it, or found it
    // marked and skipped it. Either way the copy in child (or, transitively,
    // in a deeper published bucket) is what every operation at the current
    // table size consults, and this node decides nothing any more.
    // Conversely, if child is still UNINITIALIZED this node is the key's only
    // reachable node and is kept, FROZEN or not: a FROZEN node whose child is
    // unpublished is still authoritative (class overview, INVARIANT; it arises
    // when a split threw between the freeze and the publication), and the next
    // operation that needs child will split it from this node. A deeper bucket
    // on the path cannot be published while child is not (split_bucket()
    // publishes the parent first), so testing the next child alone is
    // sufficient.
    // Relation to the concurrent form (is_dead_now()): the same rule, minus
    // the tag test. Concurrent-dead implies quiescent-dead; a reachable
    // quiescent-dead node is tagged (C2 of the accounting sweep,
    // get_internal_accounting(), below), so at a quiescent point the two
    // agree on every reachable node.
    bool is_dead(const Node* node, word_t link, size_t j, size_t ts) const {
        if (link & MARK_BIT) return true;
        const size_t child = next_child(Hash{}(node->value), j);
        return child != j && child < ts && addr_of(buckets_[child].load(std::memory_order_relaxed)) != UNINITIALIZED;
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
    //   the client can observe across the call: a value in the set is still
    //   in it, a value not in it still is not. Internally the call ends one
    //   PERIOD and begins the next (see PERIODS in the class overview): slots
    //   keep their addresses, but which node and which value occupy a slot
    //   may change; a slot hosts at most one node life per period, and link
    //   tag bits are terminal per node life. The table never shrinks (a
    //   doubling is never undone, published buckets stay published, pending
    //   splits stay pending: a split moves work rather than removing it, so
    //   they are left to the concurrent phase); no memory is returned to the
    //   system (the arena only ever grows; reclaimed nodes are reused);
    //   every key in the set has exactly one reachable node afterwards (see
    //   below), no dead node is reachable, the limbo and retired lists are
    //   empty, and the return value is that count, exact; node_count_ is
    //   reset from it.
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
    //   2'. Drain every shard's retired list (dead nodes a concurrent
    //      operation unlinked from their chain; see RECLAMATION in the class
    //      overview) into the free lists, in the two phases
    //      sequester_retired() and reclaim_sequestered() (their comments
    //      state the drain's own precondition, which quiescence implies).
    //   3. Walk every PUBLISHED bucket j < table_size_, skipping UNINITIALIZED
    //      ones, and unlink every node is_dead() says is dead, relinking
    //      through the predecessor's word: a head keeps its seal level, a
    //      link keeps its tag bits (a kept predecessor is never MARKED but may
    //      be FROZEN with an unpublished child). Every kept node is counted.
    //   4. node_count_ := kept + the append credit (see its comment).
    //   Nodes from steps 2, 2' and 3 are DEALT ROUND-ROBIN over the shards'
    //   free lists with ONE cursor, on top of whatever those lists still
    //   hold, in one policy for all sources: a shard whose threads lost many
    //   splits or erased many keys does not hoard its own garbage, and
    //   splicing a limbo or retired list whole would save nothing (finding
    //   its tail is the same walk). Each node is pushed as it is freed, not
    //   collected first (see EXCEPTIONS in the class overview).
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
        }
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
        sequester_retired();          // step 2': the retired lists, same cursor
        reclaim_sequestered(deal);
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

    // Test-only accessor: total nodes ever appended to the arena (reachable,
    // free, limbo and retired alike), exact: the sum of the shards' sizes. Used by
    // InsertContention_NoMemoryLeak to detect the CAS-retry leak, since a leak
    // inflates this count far above the number of distinct keys inserted.
    // Quiescent in the sense above, as is every get_internal_* below.
    size_t get_internal_node_count() const {
        size_t n = 0;
        for (size_t i = 0; i <= arena_mask_; ++i) n += shards_[i].nodes.size();
        return n;
    }
    // Test-only accessor: the number of arena shards (a power of two).
    size_t get_internal_arena_shards() const { return arena_mask_ + 1; }
    // Test-only accessors: the number of nodes on all free lists / on all
    // limbo lists, as the accounting sweep below counts them. A corrupted
    // (cyclic) list does not hang: the sweep stops at the first address it
    // has already seen -- but the count returned here is then merely the
    // nodes walked before the stop (a cycle A->B->C->A counts 3); the
    // corruption itself is visible only in get_internal_accounting()'s
    // `consistent`, which tests assert alongside these counts.
    size_t get_internal_free_count() const { return get_internal_accounting().free_nodes; }
    size_t get_internal_limbo_count() const { return get_internal_accounting().limbo; }
    // Test-only accessor: the number of nodes on all retired lists, as the
    // accounting sweep counts them (the same caveats as the two above).
    size_t get_internal_retired_count() const { return get_internal_accounting().retired; }
    // Test-only accessor: the current table size (number of buckets), for
    // tests that construct an exact geometry and must know whether a doubling
    // happened. Quiescent, as the others.
    size_t get_internal_table_size() const { return table_size_.load(std::memory_order_relaxed); }
    // Test-only accessor: whether bucket j (j below the current table size)
    // is published, i.e. its lazy split has run; a test that walks the
    // buckets at a quiescent point skips the pending ones with it. Quiescent,
    // relaxed, as the others.
    bool get_internal_bucket_published(size_t j) const { return addr_of(buckets_[j].load(std::memory_order_relaxed)) != UNINITIALIZED; }

    // Test-only counters: the sums over the shards of Shard's counters (see
    // there for what each counts). Exact at a quiescent point; a snapshot of
    // monotone counters otherwise. Never reset.
    struct InternalCounters {
        size_t retire_runs;              // push_retired() calls (one per unlinked run)
        size_t retire_nodes;             // nodes those runs held in total
        size_t cleanup_cas_failures;     // lost one-shot CASes of a split winner's cleanup pass, all kinds
        size_t cleanup_cas_peer_losses;  // of those, losses to a peer that bypassed the same run (nothing left behind)
        size_t walk_cas_failures;        // lost one-shot CASes of insert()'s and erase()'s walks, all kinds
        size_t walk_cas_peer_losses;     // of those, peer losses
        size_t split_attempts;           // split_bucket() calls that walked a parent chain
        size_t splits_published;         // of those, the ones whose publishing CAS won
    }; // struct InternalCounters
    InternalCounters get_internal_counters() const {
        InternalCounters c{0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t s = 0; s <= arena_mask_; ++s) {
            const Shard& shard = shards_[s];
            c.retire_runs += shard.retire_runs.load(std::memory_order_relaxed);
            c.retire_nodes += shard.retire_nodes.load(std::memory_order_relaxed);
            c.cleanup_cas_failures += shard.cleanup_cas_failures.load(std::memory_order_relaxed);
            c.cleanup_cas_peer_losses += shard.cleanup_cas_peer_losses.load(std::memory_order_relaxed);
            c.walk_cas_failures += shard.walk_cas_failures.load(std::memory_order_relaxed);
            c.walk_cas_peer_losses += shard.walk_cas_peer_losses.load(std::memory_order_relaxed);
            c.split_attempts += shard.split_attempts.load(std::memory_order_relaxed);
            c.splits_published += shard.splits_published.load(std::memory_order_relaxed);
        } // loop over the shards
        return c;
    } // get_internal_counters()

    // Test-only accounting sweep: where every arena slot is. `consistent` is
    // true iff
    //   C1  every slot of every shard is reachable from a published bucket
    //       head, on a free list, on a limbo list, or on a retired list,
    //       exactly one of the four and exactly once, and no list or chain
    //       contains an address that is not a slot;
    //   C2  every reachable node that is dead by is_dead()'s quiescent rule
    //       is TAGGED (MARKED or FROZEN): a dead node with a live link would
    //       be a node a concurrent bypass can run through while another
    //       bypasses it (a lost unlink or a double retirement); the freeze
    //       runs for both AllowDelete values to keep this true;
    //   C3  every retired node is TAGGED (only dead nodes may be retired: a
    //       live one here means a run walk overshot its run, or a retirement
    //       on a lost CAS of a run that then changed), and its `link`'s
    //       address field is EMPTY or an arena slot (the chain link a retired
    //       node is exited through was not overwritten by the retirement).
    // The counts are what was found (a walk stops at the first address that
    // is unknown or already seen, so a cyclic list cannot hang it, and then
    // `consistent` is false; a node retired twice makes exactly such a cycle
    // in a retired list). `reachable_dead` is the number of reachable nodes
    // that is_dead() calls dead at this quiescent point: the dead nodes still
    // in their chains, which reclaim()'s chain walk would unlink; it calls
    // Hash{} on every reachable node that is not MARKED, as reclaim() does.
    // Expected consistent at every quiescent point, including after an
    // exception on any operation (see EXCEPTIONS in the class overview).
    struct InternalAccounting {
        size_t slots;           // sum of the shards' sizes (get_internal_node_count())
        size_t reachable;       // nodes reachable from published heads, live and dead alike
        size_t reachable_dead;  // of those, the ones is_dead() calls dead (quiescent rule)
        size_t free_nodes;      // nodes on the free lists
        size_t limbo;           // nodes on the limbo lists
        size_t retired;         // nodes on the retired lists
        bool consistent;        // C1 (reachable + free_nodes + limbo + retired == slots, no slot twice), C2 and C3
    }; // struct InternalAccounting
    InternalAccounting get_internal_accounting() const {
        // Every slot's address, sorted, with a seen flag per slot. A shard is
        // indexed over [0, size()) with operator[], which the deque allows
        // after the acquire load that its size() is (channel 3); here the
        // quiescence precondition would allow it anyway.
        std::vector<word_t> slots;
        slots.reserve(get_internal_node_count());
        for (size_t s = 0; s <= arena_mask_; ++s) {
            const Container<Node>& nodes = shards_[s].nodes;
            for (size_t i = 0, n = nodes.size(); i < n; ++i) slots.push_back(word_of(&nodes[i]));
        }
        std::sort(slots.begin(), slots.end());
        std::vector<unsigned char> seen(slots.size(), 0);
        InternalAccounting acc{slots.size(), 0, 0, 0, 0, 0, true};
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
        // True iff `addr` is EMPTY or a slot (seen or not): the C3 link check.
        auto is_empty_or_slot = [&](word_t addr) -> bool {
            return addr == EMPTY || std::binary_search(slots.begin(), slots.end(), addr);
        };
        const size_t ts = table_size_.load(std::memory_order_relaxed);
        for (size_t j = 0; j < ts; ++j) {   // published chains
            word_t curr = buckets_[j].load(std::memory_order_relaxed);
            if (addr_of(curr) == UNINITIALIZED) continue;
            for (; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++acc.reachable) {
                const Node* node = node_of(curr);
                const word_t link = node->link.load(std::memory_order_relaxed);
                if (is_dead(node, link, j, ts)) {
                    ++acc.reachable_dead;
                    if ((link & (MARK_BIT | FROZEN_BIT)) == 0) acc.consistent = false;   // C2: a dead node with a live link
                }
                curr = link;
            } // walk bucket j's chain
        } // loop over the published buckets
        for (size_t s = 0; s <= arena_mask_; ++s) {   // free, limbo and retired lists
            word_t curr = free_addr_of(shards_[s].free_head.load(std::memory_order_relaxed));
            for (; curr != FREE_EMPTY && visit(curr); ++acc.free_nodes) curr = node_of(curr)->link.load(std::memory_order_relaxed);
            curr = shards_[s].limbo_head.load(std::memory_order_relaxed);
            for (; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++acc.limbo) curr = node_of(curr)->link.load(std::memory_order_relaxed);
            curr = shards_[s].retired_head.load(std::memory_order_relaxed);
            for (; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++acc.retired) {
                const Node* node = node_of(curr);
                const word_t link = node->link.load(std::memory_order_relaxed);
                if ((link & (MARK_BIT | FROZEN_BIT)) == 0 || !is_empty_or_slot(addr_of(link))) acc.consistent = false;   // C3
                curr = node->retire_link.load(std::memory_order_relaxed);
            } // walk the retired list
        } // loop over the shards
        if (acc.reachable + acc.free_nodes + acc.limbo + acc.retired != acc.slots) acc.consistent = false;   // C1's sum
        return acc;
    } // get_internal_accounting()
}; // class ConcurrentResizableHashSetRCU

#endif // CONCURRENT_HASH_SET_RCU_H
