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
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <bit>
#include <ostream>
#include <utility>
#include <vector>
#include <memory>
#include "concurrent_deque.h"
#include <thread>
#include <mutex>
#include "spinlock.h"

// AddressSanitizer poisoning of free-list nodes (see ASAN POISONING under
// HANDLES AND GENERATIONS in the class overview): the value bytes of a node
// are poisoned while the node is on a free list -- by the free pass,
// before it pushes the node, and unpoisoned by the pop that wins the node,
// before it assigns the new value, and by the destructor -- so that a read
// of a freed value through a stale address is reported instead of returning
// a stale or a reused value. A diagnostic: no contract of the class promises
// it. The detection is in the NESTED form on purpose: GCC defines
// __SANITIZE_ADDRESS__ and has no __has_feature, and `#if defined(X) &&
// __has_feature(Y)` on one line is a GCC syntax error. Without ASan the
// macros are no-ops. The prefix is this header's own, so that a translation
// unit which includes another header with the same scheme sees no clash.
//
// ODR: the layout of the set's handle type depends on NDEBUG (a debug build
// adds a per-handle count of the operations in flight: HANDLES in the class
// overview), and a handle crosses translation units by value, so every
// translation unit of one program that includes this header must agree on
// NDEBUG.
#if defined(__SANITIZE_ADDRESS__)
#define CHS_RCU_HAVE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define CHS_RCU_HAVE_ASAN 1
#endif
#endif
#ifdef CHS_RCU_HAVE_ASAN
#include <sanitizer/asan_interface.h>
#define CHS_RCU_ASAN_POISON(addr, size) ASAN_POISON_MEMORY_REGION((addr), (size))
#define CHS_RCU_ASAN_UNPOISON(addr, size) ASAN_UNPOISON_MEMORY_REGION((addr), (size))
#else
#define CHS_RCU_ASAN_POISON(addr, size) ((void)(addr), (void)(size))
#define CHS_RCU_ASAN_UNPOISON(addr, size) ((void)(addr), (void)(size))
#endif
// CHS_RCU_UNLIKELY(c) (header-private, #undef'd at the end): the condition
// `c`, marked as usually false but taken regularly, with the builtin's
// default weight, under which GCC keeps the taken path in the caller's own
// text (not in its cold partition) and does not make it several times
// slower. With GCC and Clang (including clang-cl) this is
//   __builtin_expect(!!(c), 0);
// elsewhere the hint is dropped and the code is still correct, only without
// the layout benefit. The !! makes the builtin accept exactly what the bare
// condition accepts (a type with only an explicit operator bool would not
// convert to the builtin's `long` parameter). Used on the automatic
// cadence's trigger test in push_retired(), which is identically false with
// the cadence off and true at most once per crossing of a multiple of the
// cadence by a shard's retired-node count otherwise. The builtin rather than
// the standard [[unlikely]] attribute: the standard leaves the attribute's
// strength to the compiler, while the builtin states the edge's weight, and
// a compiler without it needs one edit here.
#if defined(__GNUC__) || defined(__clang__)
#define CHS_RCU_UNLIKELY(c) __builtin_expect(!!(c), 0)
#else
#define CHS_RCU_UNLIKELY(c) (c)
#endif

namespace concurrent_hash_rcu_detail {
// Process-wide sequential thread numbering, shared by every instantiation of
// ConcurrentResizableHashSetRCU: a thread gets its number at its first join or
// first shard use, whichever comes first -- a join being new_handle(),
// refresh()'s slow path or advance()'s walk, a shard use an allocation, a
// retire or limbo push, or a split's or an unlink's counters (or a direct call
// of this function) -- and the number never changes. Every per-shard choice of
// the set is `thread_number() & arena_mask_`: the arena shard a thread appends
// to and pops from first, and the per-shard live-handle count its join
// increments (THE RING in the class overview). Sequential (not hashed) on
// purpose: threads started together get consecutive numbers, so `number mod
// shards` spreads a batch of up to `shards` threads over distinct arena shards
// with no collisions, however many earlier threads have come and gone.
inline std::atomic<unsigned> next_thread_number{0};
inline unsigned thread_number() {
    static thread_local unsigned number = next_thread_number.fetch_add(1, std::memory_order_relaxed);
    return number;
}

// TEST SEAMS: the `Hooks` template parameter of ConcurrentResizableHashSetRCU
// (see TEST SEAMS under HANDLES AND GENERATIONS in the class overview). A
// Hooks type provides the three static functions `NoHooks` documents, each
// callable with no arguments, returning void and `noexcept`: the set calls
// them from `noexcept` members (the join, through `new_handle()` and
// `handle::refresh()`; the bump-and-free of `advance()`, `reclaim()` and the
// automatic trigger), and a hook action that throws terminates the program.
// The hooks take no arguments: a node or block pointer would expose a
// private type. The set calls a hook from no function other than the one
// named at each hook: never from the destructor, from an operation's walk
// itself, or from the diagnostic sweep; on a set constructed with a cadence
// the publish hook fires from the trigger, which runs INSIDE an operation
// (THE TRIGGER under HANDLES AND GENERATIONS in the class overview).
template <typename H>
concept SetHooks = requires {
    { H::in_join() } noexcept -> std::same_as<void>;
    { H::in_join_after_bump() } noexcept -> std::same_as<void>;
    { H::in_advance_after_publish() } noexcept -> std::same_as<void>;
};

// The default Hooks: every hook is an empty inline function, so the
// instantiation compiles to the same code as a set with no hooks at all.
// Lives in this namespace, not at global scope, so that a translation unit
// including another container's header with its own `NoHooks` sees no
// clash, and not nested in the class, since a nested type cannot be the
// default of the enclosing template's own parameter.
struct NoHooks {
    // The join: on EVERY join attempt -- `new_handle()`'s first attempt and
    // each retry, and `refresh()`'s slow path (never its fast path, nor
    // moves or handle destruction, nor the hook-free join of `advance()`'s
    // walk) -- after the attempt read the current generation and before it
    // counts the handle in it.
    static void in_join() noexcept {}
    // The join: on EVERY join attempt (as `in_join`), after the attempt
    // counted the handle in the generation it read and before it re-checks
    // that this generation is still current (a check that fails undoes the
    // count and retries).
    static void in_join_after_bump() noexcept {}
    // Once per PUBLISH, whichever call publishes -- `advance()`, `reclaim()`
    // or an automatic trigger -- after the new generation is published and
    // before the final free pass (and, in `advance()`, the walk); the
    // advancer flag is held, so a nested `advance()` or `reclaim()` from the
    // action returns `contended` and a nested operation's retire push is a
    // skipped trigger. Not called on a `nothing_retired`, `ring_full` or
    // `contended` outcome of anything, nor by a skipped trigger. Inside a
    // trigger it fires inside the operation that retired, under that
    // operation's handle, which the action must not refresh, move, swap or
    // destroy.
    static void in_advance_after_publish() noexcept {}
}; // NoHooks
} // namespace concurrent_hash_rcu_detail

// What one `advance()` or `reclaim()` call of a ConcurrentResizableHashSetRCU
// did (`reclaim()` is `advance()` without the walk: its three walk counts
// are 0 for every outcome). Not a template and not nested: equal across
// every instantiation, so tests typed over several sets compare against one
// literal and one `PrintTo` overload serves them all, and distinct from any
// other container's result type.
struct AdvanceResult {
    // Whether and how the call advanced the generation. Checked in this
    // order: `contended` at entry; then `nothing_retired`, which wins over
    // `ring_full`. "The walk ran" below is `advance()`'s; `reclaim()` never
    // walks.
    enum class Advance {
        nothing_retired, // every shard's retired AND limbo heads were found EMPTY (by loads, before any exchange): no generation slot consumed and nothing published; the free pass ran, and the walk ran under the still-current generation
        advanced,        // the retired and limbo lists became the closing generation's bag and a new generation is current (possibly after freeing bags to make room); the free pass and the walk ran, the walk under the new generation
        ring_full,       // every generation slot is still in use after the in-call free pass (the oldest bag is pinned by a handle, or transiently by a concurrent join attempt): no slot consumed, nothing published, NO walk; the free pass ran
        contended        // another advance(), reclaim() or automatic trigger holds the flag: this call did nothing and returned at once, with every count 0
    };
    size_t freed_nodes;     // nodes moved from freed bags to the free lists in this call: retired and limbo nodes alike
    size_t freed_bags;      // generations whose bags were freed in this call (oldest first)
    size_t walk_retired;    // dead nodes this call's walk unlinked and retired (into the current generation's retired lists: reusable at a LATER call); 0 for reclaim()
    size_t buckets_walked;  // published buckets this call's walk passed over (UNINITIALIZED ones are skipped and not counted); 0 for reclaim()
    size_t nodes_passed;    // nodes this call's walk passed, live and dead alike: every node it loaded, plus the interior of every dead run it unlinked; a run the walk cannot bypass (behind a lost pred, or behind a FROZEN node with an unpublished child) is walked node by node and counted in full; only a run whose one-shot unlink CAS the walk LOST counts its first node alone (reasoned: a loss needs a concurrent writer). An uncontended walk passes exactly the reachable nodes; 0 for reclaim()
    Advance advance;        // see above; `nothing_retired` and `ring_full` do NOT imply freed_nodes == 0 (the free pass still runs); `contended` implies every count 0
    bool operator==(const AdvanceResult&) const = default;
}; // AdvanceResult

// GoogleTest's printer for AdvanceResult, found by argument-dependent lookup;
// an assertion that fails prints the fields instead of the bytes.
inline void PrintTo(const AdvanceResult& r, std::ostream* os) {
    static constexpr const char* names[] = {"nothing_retired", "advanced", "ring_full", "contended"};
    *os << "{freed_nodes=" << r.freed_nodes << ", freed_bags=" << r.freed_bags
        << ", walk_retired=" << r.walk_retired << ", buckets_walked=" << r.buckets_walked
        << ", nodes_passed=" << r.nodes_passed << ", " << names[static_cast<size_t>(r.advance)] << "}";
} // PrintTo(AdvanceResult)

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
//              below). A thread appends to, and first pops from, the shard
//              its thread number selects; when that shard's free list is
//              empty, and unless a set-wide hint says no free list has a
//              node, it scans every shard's free list, its own first, before
//              it appends (the steal: see alloc_node()). A node is never moved and
//              never returned to the system; while a node is published its
//              value is never written, its link changes only by the one CAS
//              that tags it and by unlink CASes that bypass dead successors
//              (see WORD ENCODING), and its retire_link is written only when
//              the node is retired; its link is rewritten, and its value
//              reassigned, only once the node is freed, which happens only
//              when no handle can observe it (see RECLAMATION and HANDLES
//              AND GENERATIONS). Nodes are
//              addressed by POINTER: a deque is segmented and its blocks never
//              move, so a node's address is stable for the life of the set,
//              and nothing that reads the structure knows or cares which shard
//              a node lives in. The deque's index is used once per node, to
//              obtain that address at allocation, and by the diagnostic
//              accounting sweep (get_internal_accounting()) at a quiescent
//              point; no read path ever goes through a deque.
//              Sharding exists because one shared per-node atomic of any kind
//              (a lock, a counter) is the insert ceiling; a shard's lock is
//              contended only by the threads whose numbers collide modulo
//              the shard count, and its free-list head by those threads, by
//              the free pass's pushes and by threads of other shards that
//              scan it when their own free lists are empty and the steal's
//              hint is not EMPTY (the steal).
//   node_count_ : the resize hint: arena occupancy, OVER-counted by up to 255
//              per touched shard per allocation path (each shard adds 256
//              when it appends a node whose index is a multiple of 256, index
//              0 included, and 256 on every 256th pop from a free list, the
//              first pop included), minus every node a free pass has
//              freed. One-sided at every quiescent point;
//              while operations run it may lag the occupied slots by the
//              batches whose 256 is not yet added, and it never goes below
//              zero (the member's comment says why).
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
//       LIVE link, which bypasses a run of dead successors. The free pass
//       rewrites the link of every node it frees, while that node
//       is on no chain and observable by no handle (see RECLAMATION).
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
//   parent chain, FROZEN, until a writer or advance()'s walk bypasses them,
//   so for a while a key that moved to j exists in BOTH the parent
//   chain (a superseded, FROZEN copy) and bucket j. The split's own winner
//   tries to bypass them in its cleanup pass, right after it publishes j,
//   except for copies behind a FROZEN node the pass cannot decide dead (one
//   whose child is unpublished, or beyond the table size the pass loaded); a
//   writer walk that passes them after the publish may get there first, and
//   a try can lose (see WHY COPY below). Splitting is recursive: a parent
//   that is itself still UNINITIALIZED is split first, so the bucket tree is
//   filled in on demand.
//
// WHY COPY, AND WHEN A DEAD NODE LEAVES ITS CHAIN
//   Nodes are copied, never moved: a node that a split supersedes stays where
//   it is, tagged FROZEN, and a node erase() deletes stays where it is, tagged
//   MARKED. A node address observed by any traversing thread therefore stays
//   a valid node for as long as that thread's operation runs: a dead node's
//   memory becomes reusable only when no handle that could observe it is
//   live (a bump bags it and a free pass puts it on a free list, from
//   which later allocations take it: HANDLES AND GENERATIONS), and is never
//   returned to the system before the set is destroyed. That is what lets
//   the read path dereference node pointers with no hazard pointers and no
//   per-node reference counts: the one protocol is the handle's join and
//   leave, off the operation path. The nodes that go dead are (a)
//   tombstoned nodes (AllowDelete), (b) superseded
//   parent copies after a split, (c) speculative split subchains that lost
//   their publishing CAS or were abandoned by a throw mid-build, (d) a node
//   insert() prepared and never published (a retry found the key published
//   by another thread, or threw), and (e) a free-list node whose value
//   assignment threw. (c), (d) and (e) were never published and are recorded
//   on a LIMBO list before the operation that made them returns or throws,
//   because nothing else could ever find them again.
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
//     publishes the child. The pass is lazy with respect to the doubling, as
//     splitting is; whichever operation won the split's publish runs it, a
//     contains() included (and on a set constructed with a cadence that
//     cleanup's retire pushes can trigger a bump and a free pass, so a
//     contains() that enters an UNINITIALIZED bucket may reclaim: THE
//     TRIGGER under HANDLES AND GENERATIONS; a contains() that meets no
//     UNINITIALIZED bucket never does, so on a table that has not doubled
//     no lookup bumps or frees). The copies the split superseded get an unlink
//     attempt from that same split, unless a tagged, not-dead node stands
//     between them and the pass's last live word (STRAGGLERS: the one
//     exception, and the stale view); a writer walk that passes them after the publish may
//     bypass them first, and the pass's attempt then loses or never meets
//     them;
//   - erase(), right after its mark (an eager self-unlink: one attempt
//     through the predecessor it tracked, one restart from the head, then
//     give up);
//   - the walks of insert() and erase(), which attempt to bypass every dead
//     run they pass behind a live word, on the way to their key (a hit stops
//     a walk at its key, so a hit walk attempts the runs before the key and
//     only a MISS walks the whole chain).
//   contains()'s own walk writes nothing: a reader pays for no unlinking
//   except by winning a split.
//   STRAGGLERS. Every one of these is ONE CAS per run, never retried
//   (erase()'s restart is the one exception, bounded at two): a lost CAS is
//   another thread's completed step on the same word. When that step was a
//   peer's bypass of the same run (the word already holds the successor this
//   thread computed), nothing is left behind. Otherwise -- the word gained a
//   prepended node, a seal level, a tag of the predecessor, or a peer's
//   bypass of a run of a different extent -- the dead nodes still reachable
//   are a STRAGGLER, which gets another attempt from the next writer walk
//   that passes it (hit or miss) and from the next cleanup pass over the
//   chain (every doubling gives the bucket a new child, and the split of ANY
//   child of the bucket walks the whole parent chain with no hit to stop
//   it), and from advance()'s walk, which passes every published chain with
//   the same one attempt per run. The one exception is a dead run behind a
//   FROZEN node whose child is not published: a tagged node never becomes a
//   walk's pred, so no walk, cleanup pass or advance() can bypass that run,
//   however often it is passed, until the node's own child is published (by
//   the first access to that child, whose cleanup pass, or a later walk, can
//   then bypass the node and the run together). Stragglers also arise from a stale view (a
//   walker whose table size does not yet cover the doubling a node was
//   frozen for cannot decide it), from an erase() that gave up (two lost
//   attempts, the first of which may have been skipped because the walk's
//   own CAS on the pred word was lost), from a split stalled or thrown
//   between its freeze and its publish, and from Hash{} throwing inside an
//   unlinking site (EXCEPTIONS). A dead run that no writer walks past again,
//   in a bucket none of whose new children is accessed, stays until the next
//   advance()'s walk (reclaim() and the automatic trigger never walk): its
//   lifetime is bounded by how often the caller calls advance() (and by
//   that walk's one-shot CAS winning), except for a run
//   behind a FROZEN node whose child is not published, which stays until
//   that one child is accessed.
//   POSTCONDITIONS. Unlinking is therefore best effort under contention: no
//   operation guarantees that a dead node has left its chain when it returns.
//   What an UNCONTENDED call does guarantee -- one that every other call on
//   the set happens-before or happens-after (the memory model has no wall
//   clock: a call that merely finished earlier, without synchronizing with
//   this one, counts as concurrent), and that returns normally -- is this: a
//   split's cleanup leaves no dead node reachable in the parent chain behind
//   an untagged word (the bucket head or a live node's link); an insert() or
//   erase() walk leaves no dead run reachable behind the bucket head or a
//   live node it passed (a run at the head counts: the head is an untagged
//   word); an advance() whose walk runs leaves no dead run reachable behind
//   an untagged word in any published bucket, except a run behind a FROZEN
//   node whose child is not published (UNINITIALIZED buckets hold nothing
//   and are not visited); an erase() leaves its tombstone unreachable,
//   unless a tagged, not-dead node lies between it and the last live word
//   before it, which in
//   uncontended use arises only after an earlier split threw between its
//   freeze and its publish (behind such a node nothing can be bypassed until
//   its child is published and that child's split's cleanup pass, or a
//   later walk, passes it). reclaim() and the automatic trigger walk no
//   chain: they make no dead node unreachable and promise nothing here.
//   NOT TAKEN. One design is deliberately NOT taken here: a contains() that
//   bypasses the dead runs it passes (it would make every reader a writer);
//   the read walk stays pure. The collector of the remainder is advance()'s
//   walk: a standalone call that walks every published chain like a writer
//   miss walk without inserting anything, at the cost of an API call on
//   the caller's schedule.
//   RETIREMENT. A retired node is not reused before a bump has bagged it
//   and no handle that could observe it remains: the retired lists only
//   separate "dead and out of its chain" from "dead and still in it", for
//   the bump to take without a chain walk (RECLAMATION).
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
//   constructor's shard array, the generation ring's bag slots and scratch
//   and the per-shard count rows (in the constructor, before any handle
//   exists), and the vectors of get_internal_accounting(),
//   get_internal_ring_state(), get_internal_slot_counts() and
//   get_internal_retire_nodes_per_shard(); nothing else in this class
//   throws. new_handle(), refresh(), the handle's special
//   members and the join and leave are noexcept. An operation that throws
//   leaves its handle live and usable: the handle's debug in-flight count is
//   undone on the throwing path too (by RAII), and release builds touch
//   nothing of the handle. An arena address the word
//   encoding cannot represent is not an exception either: alloc_node()
//   calls std::abort(), as the constructor does on an initial capacity
//   above 2^63. A Hooks action that throws terminates (the hooks are
//   noexcept: TEST SEAMS). reclaim() and the automatic trigger throw
//   nothing: the bump, the free pass and the hook are noexcept, and nothing
//   in them hashes, compares, copies or allocates; an operation that throws
//   AFTER its trigger ran leaves the trigger's effects standing.
//   The one deliberate guarantee: NO ARENA SLOT IS LOST TO A THROW. Every
//   private node an operation holds when an exception passes through it is
//   pushed to limbo before the exception leaves, so get_internal_accounting()
//   stays consistent and a later bump and free pass recycle the node. Per
//   operation:
//   - T (template parameter): must be copy-assignable, because a node reused
//     from a free list takes its new value by assignment; a throwing
//     assignment must leave the old object valid, as every standard type's
//     does.
//   - alloc_node(): if T's copy assignment throws on a free-list node, the
//     node (still holding its old, valid value) goes to the limbo list of
//     the popping thread's own shard (for a node stolen from another
//     shard's free list too) -- the limbo push writes the link the caller's `next` would have
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
//     lost; once a competitor publishes j, its copies take over and the
//     nodes this call froze are DEAD NODES, exactly as after a lost
//     publishing CAS. If Hash{} throws inside the winner's cleanup pass
//     (step 4), AFTER the publishing CAS (the case the comments below call
//     W4), bucket j is published and complete; the exception leaves the dead
//     nodes of the parent chain reachable, as a lost CAS would, for a later
//     walk, split or advance() to collect. The caller of split_bucket() sees
//     the exception, not the published bucket.
//   - insert(): a throw before the publishing CAS -- from alloc_node(),
//     Hash{}, T's operator==, or a split_bucket() met on any attempt --
//     leaves the key as it was; the catch around the retry loop pushes the
//     pending node, if one was allocated on an earlier attempt, to limbo and
//     rethrows. new_node is nulled the moment its CAS publishes it, so that
//     catch never pushes a published node. A throw from the doubling AFTER
//     the publishing CAS succeeded leaves the key a member although the
//     caller sees an exception and no return value; nothing is said about the
//     state after a throw from inside buckets_.resize().
//   - contains(): holds no private node; it throws only what Hash{}, T's
//     operator== or a split_bucket() it falls into throws, and changes no
//     membership when it does (a split it won is published, cleanup unlinks
//     included).
//   - erase(): holds no private node; it throws what Hash{}, T's operator==
//     or a split_bucket() throws. A throw BEFORE the mark (from the hash of
//     the key, from a split, from a key comparison, or from the hash of a
//     FROZEN node the walk passes, the call site H5' of PINNED ORDERS, before
//     contains()) leaves the key as it was. Hash{} is also called inside the
//     self-unlink that follows a successful mark (H5; the case the comments
//     below call P5): a throw there leaves the key DELETED (the mark is the
//     deletion) but the caller sees an exception and no `true`, as insert()
//     does on a throw from the doubling after its publishing CAS. Any unlink
//     a throwing erase() or insert() walk had already won stands; it changed
//     no membership.
//   - advance(): throws only if Hash{}(key) throws, and only from its walk,
//     which hashes the FROZEN nodes it passes (H5'); the bump and the free
//     pass complete before the walk starts and cannot throw. BASIC
//     guarantee: membership is unchanged (the walk never changes it); every
//     unlink the walk won stands and its retirees are on the current
//     generation's retired lists; the dead runs not yet reached are
//     stragglers for the next call; the advancer flag and the walk's handle
//     are released by RAII guards, so the next advance() proceeds (it is
//     not `contended`, and nothing is pinned by the walk); the accounting is
//     consistent. The RESULT IS LOST: the caller learns nothing about what
//     the call freed or whether it consumed a slot
//     (get_internal_generation() tells). advance() may be called again.
//
// RECLAMATION: FREE LISTS, LIMBO LISTS, RETIRED LISTS, BAGS
//   Every shard has two intrusive singly linked lists threaded through the
//   nodes' `link` words: a FREE LIST of nodes available for reuse and a LIMBO
//   LIST of unreachable nodes waiting for the next bump (advance(), reclaim()
//   or a trigger). The next pointer lives in the atomic link word and never in
//   the T bytes: a thread that loses a pop race may still read the node's link
//   while the winner is already assigning the node's value (see alloc_node()).
//   A third list per shard, the RETIRED LIST, is threaded through a word of
//   its own, Node::retire_link: it holds the dead nodes that a concurrent
//   operation has unlinked from their chain (push_retired() is its one entry
//   point, called by the unlink winner with exactly the run it bypassed), and
//   such a node may still be traversed by an operation that reached it before
//   the unlink, so its `link` must stay what it was (EXIT). The limbo and
//   retired lists are LIFO and push-only between two bumps; each
//   push is a RELEASE CAS, so the bump's ACQUIRE exchange of a head is a
//   happens-before boundary for the whole list below the value it took
//   (every push is a read-modify-write extending the release sequence of
//   the one before it).
//   advance() moves nodes between the lists in three steps (reclaim() and
//   the automatic trigger run the first two; HANDLES AND GENERATIONS below
//   is the protocol; this paragraph is the data structure): the BUMP
//   exchanges every shard's retired list, and every limbo list that is not
//   empty, into the closing generation's BAG -- one
//   retired-sublist head and one limbo-sublist head per shard, kept in the
//   generation block, not in the shard, so that the bump's writes never
//   land on a line the operations contend for -- and publishes the next
//   generation; the FREE PASS pushes every node of every bag that no live
//   handle can observe onto the free lists, dealing the nodes round-robin
//   over the shards; the WALK unlinks the dead nodes still in their chains
//   with the writer walks' one-shot CAS and retires them. A retired or
//   limbo node is reused no earlier than the bump that bags it, and
//   only once no handle of its bag's generation or an older one is live:
//   retirement changes WHEN a dead node stops being walked, and the bag
//   decides when its memory comes back.
//   - POP (alloc_node()): a lock-free CAS on the calling thread's own shard's
//     free head, with ACQUIRE orders, concurrent with the free pass's pushes.
//     The head word packs the top node's address with a 9-bit POP COUNTER
//     (see the free-list constants), so the CAS that pops a node also counts
//     it, and only the winner's count lands: every 256th pop from a free
//     list, its first pop included, adds 256 to node_count_, mirroring the
//     append path's batching. Own list empty: unless a set-wide hint says
//     no free list has a node (one relaxed load), the same pop on every
//     shard's free list in turn (the STEAL: the free pass deals freed nodes
//     over every shard, so a node dealt to a shard no allocating thread maps
//     to would otherwise never be reused); every list empty: append to the
//     deque.
//   - PUSH onto a free list (push_free()) happens ONLY inside the free pass
//     (of advance(), reclaim() or an automatic trigger: ONE ADVANCER), one
//     flag holder at a time, with a RELEASE CAS that keeps the
//     counter bits, and runs concurrently with the pops. ABA on the pop is
//     excluded by the handle invariant together with the single advancer,
//     not by the counter (ABA WITHOUT VERSION TAGS below): a node another
//     popper took cannot return to the free list while this popper's handle
//     lives. The free list needs no Harris-style marking for the same
//     reason: nothing unlinks from it concurrently (the chains are another
//     matter: WHY COPY above).
//   - PUSH onto a limbo list is a Treiber push (release CAS) on the pushing
//     thread's own shard; the list is push-only until the bump exchanges it,
//     and the push is ABA-benign (the pusher only stores the head it read
//     into its own tail's link: Shard::limbo_head); nothing but the bump and
//     the diagnostic accounting sweep ever read it. THE LIMBO LIST below says why limbo
//     goes through the bag and why the push is release.
//   - PERIODS. What contains(), insert() and erase() can observe across an
//     advance() is MEMBERSHIP only -- a value in the set before it is in the
//     set after it, a value not in it still is not (the get_internal_*
//     diagnostics also see the arena, which advance() does change). The
//     client holds no references into the table (the API hands out none),
//     so nothing of the client's outlives a call, and no ABA problem can
//     reach the client. The INTERNAL view: slots never move (a slot's
//     address is stable for the life of the set, see STORAGE), doublings are
//     never undone, published buckets stay published with their seal
//     levels, and kept links keep their tag bits. What the free pass changes
//     is which node, holding which value, occupies a slot; a slot hosts one
//     NODE LIFE per session of every handle that can observe it (SESSIONS
//     below), and a link's MARK/FROZEN bits are terminal per node life.
//   - ADDRESS REUSE and the deciding CASes. insert()'s CAS on a bucket head,
//     erase()'s CAS on a link and the unlink CAS on either are safe against
//     ABA because no node address a handle can observe is published twice
//     during that handle's session: a node leaves a chain through an unlink
//     CAS or dies in it, is bagged at a later bump, is freed only when no
//     handle of its bag's generation or an older one is live, and re-enters
//     a chain only through a pop that happens-after that free. An address a
//     thread observed in a head or link therefore names the same node, in
//     that chain or retired from it, for the whole of that thread's
//     operation, and an unlink CAS whose expected value still holds really
//     is bypassing the run it recorded (ABA WITHOUT VERSION TAGS below is
//     the argument). There is therefore no ABA problem the client can get
//     into, and none in the free lists.
//   - LIFETIME OF VALUES. A free node keeps its old, constructed T; no
//     reclamation -- advance(), reclaim() or a trigger -- destroys
//     anything. Reuse COPY-ASSIGNS the new value over the old one, so the
//     old value's resources are released at reuse (the assignment's
//     left-hand side), not at erase() and not at any reclamation;
//     values that are never reused are destroyed with the set. Callers must
//     not rely on destructor timing. Reuse happens while other operations
//     run; it is safe because a node is freed only after every handle that
//     could observe it has left or refreshed, and it is popped by one thread
//     while on no chain (THE PER-NODE REUSE CHAIN below). A throwing
//     assignment routes the node to limbo, reused only after a later bump
//     and free pass. Chosen over destroy-at-free + construct-at-reuse (the
//     deque destroys every slot at the end, so a destroyed slot would have to
//     be re-constructed at once) and over raw-storage slots (saves only
//     destructor calls, costs a non-destroying container).
//
// HANDLES AND GENERATIONS: RECLAMATION BESIDE THE OPERATIONS
//   The protocol stated here is the one the handle, the generation ring and
//   advance() implement; it is the correctness argument of the class, and
//   every memory order of the generation protocol is chosen against it and
//   labeled at its site: DESIGNATED (a weaker order has a known failing
//   execution), REQUIRED (a weaker order breaks a happens-before chain the
//   argument needs) or EXPECTED-EQUIVALENT (a weaker order is believed
//   equivalent; the stated one is kept for the reason given).
//
//   HANDLES. Every operation (contains(), insert(), erase()) runs under a
//   `handle`, its first argument. new_handle() JOINS the set's current
//   generation; the handle's destructor LEAVES it; refresh() leaves the
//   handle's generation for the current one. A handle is a SESSION with one
//   generation: while it lives, no node that the handle could observe --
//   every node reachable from a published bucket head at or after its join,
//   and every node it can reach by EXIT through the immutable tagged links
//   of dead nodes -- is freed or reused. The set hands out no iterators and
//   no references, so a handle protects nothing between two calls: its only
//   idle effect is to pin its generation, and with it every node retired
//   since (the cumulative rule, below). A handle is therefore meant to be
//   long-lived and refreshed regularly. Thread rules: any number of
//   threads may run operations concurrently, each under its own handle or
//   sharing one; refresh(), moving from a handle, move assignment over it,
//   swap and destruction are exclusive with every other use of that handle
//   (debug builds count the operations in flight on a handle and assert
//   zero at each of these), and with nothing else: not with other handles,
//   with new_handle(), advance() or reclaim(). THE HAND-OFF RULE: a handle
//   may be created on one thread and used on another, and its session may
//   be ended or moved on another thread than the one that used it last --
//   destroyed, refreshed, assigned over, moved from or swapped -- only
//   after a SYNCHRONIZING hand-off between the threads (a mutex, a thread
//   join, a release store read by an acquire load, or any other
//   happens-before edge): every use of the handle, its creation included,
//   must happen-before the call that ends or moves its session, as its
//   creation must happen-before its first use elsewhere. Not overlapping
//   in time is not enough: the leave's release orders for the free pass
//   only the accesses that happen before it (THE PER-NODE REUSE CHAIN),
//   so with a relaxed hand-off the free pass can reuse a node a reader on the
//   other thread is still reading. The handle records the per-shard count its
//   join incremented, never a thread id, so a leave on any thread decrements
//   that count. An operation runs under a handle of ANY live generation, the
//   current one or one a bump has closed since the join, with its full
//   contract: the generation decides only which bags a free pass may
//   free while the handle lives, and only refresh() or a release changes
//   it. A temporary is a handle too: `set.contains(set.new_handle(), k)`
//   joins, and leaves at the end of the full expression. The precondition
//   violations debug builds diagnose, by assertion: an operation with an
//   empty handle or with a handle of another set; refresh() of an empty
//   handle; refresh(), a move, swap or destruction of a handle while an
//   operation runs under it; destruction of the set with a live handle or
//   during an advance(), a reclaim() or an automatic trigger; the accounting
//   sweep or a ring view (DIAGNOSTICS) during any of those three. Release
//   builds diagnose none of them, and there an
//   operation touches nothing of its handle, so an operation's cost does
//   not depend on its handle's age or on how many handles exist. Taking a
//   handle never waits for an advance(): the join is lock-free and retries
//   only after an advancer has
//   published a new generation. ODR: the handle's layout depends on NDEBUG
//   (the debug in-flight counter), so every translation unit of one program
//   that names the handle type must agree on NDEBUG.
//
//   THE RING. Generations are status blocks in a fixed ring of
//   kGenerations `GenerationBlock`s that are never deallocated or moved.
//   Each block has three PLAIN fields owned by the advancer -- `state`
//   (EMPTY, CURRENT, SEALED), `gen` (the generation number the slot
//   currently represents) and the BAG: one pair of slots per arena shard,
//   the shard's retired sublist head and limbo sublist head as exchanged at
//   the bump -- and a pointer to the owning set (how a handle finds its set
//   to refresh). Its live-handle count is SPLIT BY ARENA SHARD: S counts per
//   slot, S = the number of arena shards, `c[i][s]` being shard s's count of
//   ring slot i. They live outside the blocks, in one row per shard
//   (`rows_[s]`: shard s's count of every slot, then shard s's join
//   counters), so that a join and a leave write only lines of the joining
//   thread's shard and never a line every thread writes. A join increments
//   the count of the JOINING thread's shard (`thread_number() &
//   arena_mask_`); the handle records which count it incremented and its
//   leave decrements THAT count, from whatever thread runs the leave; the
//   free pass reads every count of a slot. The counts are atomic and
//   RMW-only (J2 below) and, like the blocks, never deallocated or moved
//   while the set lives. `current_` points at the CURRENT block; behind it
//   sits a contiguous run of SEALED blocks whose numbers decrease by one
//   per slot going back, down to the oldest at index `oldest_`; every other
//   slot is EMPTY. The ring is fixed because a joiner that has loaded
//   `current_` but not yet incremented its count may increment a count of a
//   block that was closed in between: the count must still exist for that
//   stale increment to land somewhere harmless. Footprint: kGenerations
//   blocks of 128 bytes each (a pair of 64-byte lines: see GenerationBlock)
//   plus two words per shard per block for the bags, plus one count row per
//   shard (sizeof(CountRow): kGenerations counts and two join counters,
//   rounded up to 128 bytes), plus the bump's scratch, one word per shard
//   (limbo_loaded_).
//
//   THE JOIN (join(), from new_handle() and refresh()'s slow path; the walk
//   joins the same way without the hooks): s = the calling thread's shard,
//   once per call; then per attempt: load `current_` -> B (relaxed); C =
//   B's count for shard s, `c[slot(B)][s]`, computed once per attempt from
//   the B just loaded; `C.fetch_add(1, seq_cst)`; re-verify with a seq_cst
//   load that `current_` is still B (pointer equality); if so the handle is
//   a MEMBER of B's generation and records (B, C); if not, undo with a
//   release `fetch_sub` ON C and retry. The join and the advancer's
//   "publish, then read every count" form a STORE-BUFFERING pair, one per
//   count: joiner "+1 on C, then load current_", advancer "store current_,
//   then load C"; with acquire/release both can miss (the joiner sees the
//   old block, the advancer sees 0 on C) and the bag is freed under the
//   handle. seq_cst on all FOUR operations -- the increment, the re-verify
//   load, the publish store and the advancer's load of each count in the
//   free pass -- puts them in one total order S. A member's re-verify reads
//   B, so it precedes the publish that closes B in S (else it would read
//   that publish or a later one), and the increment, sequenced before the
//   re-verify, precedes it too; every free-pass load of a count of B
//   follows that publish in S (FACT 2, below); so the free pass's load of C
//   follows C's +1 in S, and a seq_cst load that follows a seq_cst RMW of
//   the same object in S reads that RMW's value or a later one in C's
//   modification order (coherence-ordered-before agrees with S). By J2,
//   while the member has not left, every such value is >= 1; reading 0
//   needs the member's leave at or before the value read, and then (every
//   later modification being an RMW) the value read is in the release
//   sequence the leave heads, so the leave synchronizes with the load. The
//   S counts change nothing in this pair: each runs it on its own, and the
//   free pass loads EVERY count of the bag (ONE ADVANCER). The argument
//   rests on three facts. FACT 1: a member's count is fixed from its join
//   to its leave: a move or swap carries the count with the block, and
//   refresh() is a new join plus the leave of the old membership; nothing
//   recomputes a recorded count. FACT 2: every free-pass load of a count
//   of a closed block follows, in S, the publish that closed it: in the
//   flag holder that published, the load is sequenced after the publish;
//   in a later holder, the publish is sequenced before the earlier
//   holder's release of the flag, which the later holder's acquire
//   exchange reads (or a later exchange in its release sequence), which is
//   sequenced before the load -- sequenced-before, then synchronizes-with,
//   then sequenced-before, which makes the publish strongly happen before
//   the load, and two seq_cst operations so ordered are ordered so in S.
//   FACT 3 is J2. The publish is a STORE, not a read-modify-write: FACT 2's
//   step across two holders is the standard's rule for seq_cst operations
//   ordered by strongly-happens-before; a seq_cst RMW would be
//   expected-equivalent. Two invariants make a stale increment harmless.
//   J1: between its `fetch_add` and a successful re-verify a joiner
//   touches NO node, head, bag slot, plain block field or plain field of
//   the count rows (it has read only `current_` and the rows pointer, which
//   is constant after construction), so a count it leaves on a just-closed
//   or recycled block can delay that block's freeing but never is the
//   count that protects a dereference. The undo decrements the count the
//   attempt incremented, which is why the attempt holds C rather than
//   recomputing it: recomputing the SHARD would give the same index (a
//   thread's number and the set's shard count never change), but a slot
//   recomputed from a reloaded `current_` would put the -1 on another
//   block's count, which could then read 0 under that block's live member.
//   J2 (FACT 3), per count: every count is modified ONLY by
//   read-modify-writes -- +1 by joiners, -1 by a leave and by a join's undo
//   -- and is never stored: the advancer only reads it and does NOT reset
//   it when it recycles a slot (a reset would wipe a stale +1, and the
//   matching -1 would then drive the count negative, or to zero under a
//   live member). Every -1 follows its OWN +1 ON THE SAME COUNT in that
//   count's modification order (the undo is sequenced after its increment;
//   a leave happens after its join, so write-write coherence orders them),
//   so no count goes negative (debug-asserted at every `fetch_sub`), and at
//   any point of a count's order its value is at least the number of its
//   members that have not left. Pointer equality suffices across a ring
//   wrap: a joiner whose re-verify reads B after B was freed, recycled and
//   republished has synchronized with that LATER publish, joins the later
//   generation and, by J1, looked at nothing before; a joiner whose
//   re-verify fails after its +1 landed on a recycled block's count only
//   delays that generation's free pass until its undo. The leave is a
//   release `fetch_sub` on the recorded count, from whatever thread runs
//   it: it heads the per-node reuse chain (below), whose first link -- the
//   last access made under the handle, then the leave -- is a
//   happens-before edge, sequenced on one thread or ordered by the caller's
//   synchronizing hand-off when the handle is left on another (THE HAND-OFF
//   RULE under HANDLES).
//   refresh()'s fast path is a relaxed load of `current_` compared with the
//   handle's block: it changes no membership, so it has nothing to
//   synchronize with, and its only stale answer is "same", which keeps the
//   handle in a just-closed generation -- safe and conservative.
//
//   ONE ADVANCER. Three things reclaim, and they share one global try-flag,
//   `reclaiming_`: advance() (the bump, the free pass, the walk), reclaim()
//   (the bump and the free pass, no walk: advance() = reclaim() + walk) and, on
//   a set constructed with a cadence, the AUTOMATIC TRIGGER (THE TRIGGER below:
//   reclaim()'s work, run from inside an operation). Every one of them:
//   `exchange(true, acquire)` at entry (a call that finds the flag set returns
//   `contended` at once, every count 0; a trigger that finds it set does
//   nothing and counts itself skipped) and `store(false, release)` at exit, on
//   every path, by an RAII guard. Reclamation is CALLER-DRIVEN by default: on a
//   set without a cadence nothing is ever freed except by an advance() or
//   reclaim() that some thread calls, with or without a handle of its own, and
//   no operation and no handle's leave frees anything; on a cadenced set an
//   operation frees only through its trigger, hence only when it RETIRES (an
//   erase whose self-unlink won, a writer walk that bypassed a dead run, a
//   split's cleanup pass), and a handle's leave still frees nothing. The flag
//   is held for the whole call or trigger, so an advance() or reclaim() nested
//   inside it -- from the hook, or from a Hash{} the walk calls -- returns
//   `contended`, and a trigger nested inside it -- the walk's own retire
//   pushes, or an operation the hook runs -- is skipped; an advance() or
//   reclaim() nested in a Hash{} or operator== that an OPERATION calls is an
//   ordinary call, and the operation's own handle pins the generation it closes
//   (the trigger is this same case without the walk: THE TRIGGER below).
//   Between successive holders the flag's acquire/release pair is the ONLY
//   synchronization on the plain fields `state`, `gen`, the bag slots,
//   `oldest_`, the dealing cursor, the bump's scratch and the steal hint's
//   stamp counter, and it orders one holder's publish before a later holder's
//   count loads (the store-buffering argument across calls: FACT 2 of THE
//   JOIN). `gen` is also read outside the flag, through the publish that made
//   its block current: by a handle's own diagnostic (its successful re-verify
//   read that publish) and by get_internal_generation() (an acquire load of
//   `current_`). Inside, sequentially (A0-A5 and the free pass for every
//   holder; the walk for advance() alone):
//   (A0) load every shard's retired head and limbo head (relaxed; a push
//        racing a load is seen by the next call): all EMPTY -> the result is
//        `nothing_retired`, no exchange is made, no slot is consumed and
//        nothing is published; the free pass and the walk still run (below).
//   (A1) closing = the CURRENT block; next = the slot after it.
//   (A2) next not EMPTY (the ring has wrapped): run the free pass once and
//        look again; still not EMPTY -> `ring_full`: the oldest generation
//        is pinned, no slot is consumed, nothing is published, and the call
//        returns after the free pass below (its second, which frees only
//        what a leave since the first one unpinned) without walking. The
//        skipped walk is a cost choice, not an invariant: its retirees
//        could not be bagged until the pin goes away, so a refused call
//        does not pay a whole-table walk for nothing; a `nothing_retired`
//        call on a wrapped, pinned ring decides at A0, before this check,
//        and does walk, so retirees can wait on the retired lists of a
//        full ring all the same, and are bagged by the first call after
//        the pin leaves.
//   (A3) for every shard, retired_head.exchange(EMPTY, acquire) into the
//        closing block's retired slot for that shard, and, for every shard
//        whose limbo head (A0) read as non-EMPTY, limbo_head.exchange(EMPTY,
//        acquire) into its limbo slot. EVERY exchange precedes the publish:
//        a retire push that lands after its shard's exchange goes to the
//        next generation, which is conservative by construction and is what
//        makes the bag form sound without reading a generation number at
//        retire time.
//   (A4) plain writes: closing->state = SEALED, next->state = CURRENT,
//        next->gen = closing->gen + 1.
//   (A5) publish: `current_.store(next, seq_cst)`; the hook
//        in_advance_after_publish() fires here (TEST SEAMS).
//   Then the FREE PASS (free_oldest()): while ring_[oldest_] is SEALED and
//   EVERY one of its S per-shard counts reads 0 -- one seq_cst load per
//   count, in ascending shard order, the first count that is not 0 ending
//   the pass -- free its bag -- every retired sublist
//   through `retire_link` (read, never written here) and every limbo
//   sublist through `link`, each node's next word read BEFORE push_free()
//   rewrites `link` into the free list's next pointer -- dealing the nodes
//   round-robin over the shards' free lists with one cursor, poisoning each
//   node's value bytes under ASan, and subtracting the
//   count from node_count_; mark the slot EMPTY (its counts untouched, J2)
//   and advance `oldest_`. Bags are freed strictly oldest-first; a block is
//   never skipped; a slot is reusable only when EMPTY. The S loads of one
//   bag are not one snapshot, and need not be: after the publish that
//   closed the block no join of it succeeds (THE JOIN) and no membership
//   moves from one count to another (FACT 1), so its members only leave,
//   and a count read 0 had every member on it left, its leave acquired by
//   that load. For the same reason the order of the loads, or comparing
//   their sum with 0 (the counts are never negative), would do as well for
//   this protocol; ascending order with an early exit costs one load per
//   count up to the first pinned one. Then the WALK, under
//   a handle of advance()'s own (joined through the hook-free join path):
//   every published bucket, with the split winner's cleanup pass's shape --
//   head load with acquire, one unlink attempt per dead run, never retried,
//   UNINITIALIZED buckets skipped, no split and no allocation -- retiring
//   what it bypasses onto the shard's retired lists of the generation that
//   is current AFTER the bump (after the publish, when there is one: a
//   `nothing_retired` call publishes nothing). A walk retiree is therefore
//   freed no earlier than the NEXT bump (it is exchanged into a bag by the
//   next bump's A3 and freed by that bump's free pass at the earliest):
//   two clicks. A dead node still in a chain carries no generation and is
//   visible to handles of EVERY live generation, which is why the walk can
//   only make it unreachable with the one-shot unlink CAS every writer walk
//   makes, never with a plain-store relink, and why the walk never touches
//   a bagged node (a bagged node is unreachable from every head). Why
//   sequester (A3) before the walk: a walk before A3 would be safe too (its
//   retirees would land in the bag A3 closes, freed only after the
//   advancer's own handle leaves); the order chosen keeps one rule, "every
//   walk retiree is in the new generation". Advancers do not run
//   concurrently: two advancers writing one closing bag, an advance order
//   that wedges the ring after one wrap, or a free walk passing a block
//   still current are the hazards a second advancer brings, and refusal
//   costs nothing (`contended`, `ring_full` and `nothing_retired` consume no
//   slot; the caller calls again). A thread that HOLDS a handle and calls
//   advance() or reclaim() frees nothing at or after that handle's
//   generation, because its own handle pins it: refresh or release the
//   handle first (a trigger cannot: THE TRIGGER). A thread stalled while it
//   holds the flag -- inside advance(), reclaim() or a trigger -- stalls
//   all reclamation until it resumes (every call `contended`, every
//   trigger skipped), never another thread's operation.
//
//   THE TRIGGER (the automatic cadence; opt-in through Options). A set
//   constructed with `auto_reclaim_every` = R > 0 (rounded up to a power
//   of two; above 2^63 clamped to 2^63, which is "never") runs reclaim()'s
//   work by itself: every retire push (push_retired(): the winner of an
//   unlink CAS retiring a run onto its own shard's retired list) takes the
//   shard's retired-node count from old to old + n, n the run's length, and
//   if the half-open interval (old, old + n] holds a multiple of R, the push
//   TRIGGERS, exactly once however many multiples the run jumps: after the
//   push, on the retiring thread, inside the operation that retired, under
//   that operation's handle, it attempts the flag and, holding it, runs the
//   bump, the hook if it published and the final free pass (auto_reclaim(),
//   bump_and_free()); never the walk. The unit is nodes retired PER ARENA
//   SHARD: not runs, not set-wide, not per thread; the multiples are
//   absolute positions of the shard's counter, never reset, so a skipped
//   attempt is not retried and the next is R nodes on. What the cadence
//   guarantees is therefore a BOUND ON SPACING, not a count: at most one
//   attempt per R nodes retired on a shard, and nothing about how many
//   proceed (an attempt that meets the flag held -- an explicit call, another
//   thread's trigger, or this thread's own advance() whose walk is retiring
//   -- is skipped: `auto_skipped`; one that acquires the flag is
//   `auto_reclaims`, whatever its bump then decides). Under contention the
//   skipped attempts can far outnumber the proceeding ones; no ratio
//   between the two is promised, and a caller that needs the counts reads
//   the two counters. The crossing run's
//   fate follows the bump's outcome: `advanced`, the trigger's own exchange
//   took it (push then exchange on one thread: it is in the bag the trigger
//   closed); `nothing_retired`, a concurrent bump of any cause took it
//   between this thread's push and its flag acquisition; `ring_full` or
//   skipped, it stays on the retired list for the next bump of any cause.
//   Limbo pushes never count and never trigger. Under a cadence an
//   explicit advance() or reclaim() concurrent with operations may return
//   `contended` (an operating thread's trigger holds the flag); at a
//   quiescent point it never does, since a trigger lives only inside an
//   operation. A contains() can reach a trigger only by entering an
//   UNINITIALIZED bucket and winning its split, whose cleanup pass retires:
//   on a table that has not doubled no lookup ever bumps or frees.
//   WHY THE TRIGGER IS SAFE: it is the nested-call case above (an advance()
//   from a Hash{} or operator== that an operation calls, which also runs
//   inside an operation, with that operation's walk state live) minus the
//   walk, at a different site -- the end of push_retired(), just after the
//   push. Four claims, each of which the trigger's code keeps:
//   (alpha) the operation's handle is live and unmoved for the whole
//           trigger: refresh(), moves and destruction are excluded while
//           an operation is in flight (HANDLES; debug-asserted), and the
//           trigger has no handle in scope -- push_retired(), unlink_run()
//           and the walk state carry none -- so no refresh, leave or move
//           is reachable from it. Its +1 on the count its join
//           incremented -- generation g's count for the shard of the thread
//           that joined the handle or ran its last refresh() that returned
//           true (moved it), which need not be the
//           triggering thread's shard -- therefore stands (FACT 1), and the
//           trigger's own free pass reads it, since that pass loads EVERY
//           count of block g: the pass's seq_cst load of the handle's count
//           happens after the handle's join increment (sequenced after it
//           when the operating thread joined, ordered by the caller's
//           synchronizing hand-off when another thread joined the handle
//           or ran its last moving refresh(): THE HAND-OFF RULE under
//           HANDLES), so by write-read coherence (and J2: every -1 follows
//           its own +1 on the same count) it reads at least 1, whatever
//           shard the triggering thread maps to. The store-buffering
//           argument of THE JOIN is needed for OTHER threads' pins only.
//   (beta)  the free pass stops at the first pinned bag (oldest-first,
//           never skipping: direction (II)), hence at or before bag g (bag
//           g is SEALED with the handle's count >= 1, or still CURRENT):
//           a trigger frees only bags of generations BELOW its own
//           handle's.
//   (gamma) every node address the operation holds across the trigger --
//           the walk state's pred word and its value, the head word, the
//           run just pushed, the successor the walk continues from, the
//           eraser's tombstone, the inserter's still-private node -- is
//           either in no bag, or in a bag of generation >= g by direction
//           (I) (the thread obtained it under g through a chain entered at
//           a head loaded after the join: C1-C4), or is the just-pushed
//           run (delta); none is in a bag the trigger's free pass frees.
//           The values that serve as later CAS expected values (the pred
//           word, the head word) rest on ABA WITHOUT VERSION TAGS, whose
//           premise -- the handle's pin stays -- (alpha) keeps.
//   (delta) the run just pushed is exchanged by the SAME thread in the
//           trigger's A3 (push then exchange on one atomic, sequenced; an
//           RMW reads the last value in modification order), so it is in
//           the bag of the generation the trigger closes, G >= g, which the
//           thread's own handle pins; it is freed by a later free pass,
//           after every handle of generation <= G has refreshed or left.
//   Consequences. LATENCY: under the cadence a node retired by a thread is
//   freed no earlier than a bump AFTER the one that bagged it, and only once
//   every handle of that bag's generation, the retiring thread's included, has
//   refreshed or left: a thread that erases under a never-refreshed handle pins
//   its own garbage, every trigger it runs closes a bag it then pins, and after
//   kGenerations - 1 such bumps every further trigger finds `ring_full` until
//   that handle refreshes; the refresh advice of HANDLES stands. GROWTH UNDER
//   LAGGING HANDLES: once the ring has filled behind the oldest pin, every
//   proceeding trigger is `ring_full` until that pin leaves -- it bags nothing,
//   its free passes free nothing the pin holds, and the retired lists keep
//   growing -- and with every handle lagging the ring refills after each
//   release, so when the triggers far outnumber the refreshes that release the
//   oldest pin, almost every proceeding trigger is `ring_full`. No node is
//   freed meanwhile, so every allocation that finds no free node (an insert's
//   node, a split's copies) appends, and the arena grows without bound for as
//   long as the pins last (the trade of WHAT THIS DOES AND DOES NOT GUARANTEE;
//   the cadence does not change it). THE HOOK fires inside the operation (TEST
//   SEAMS). COST: a trigger adds WORK to the operation that retired -- the
//   loads and exchanges of the bump and the free pass of every unpinned bag,
//   bounded by the bagged nodes, unbounded in time only in that the first
//   trigger after a long stall ends frees everything the stall pinned -- never
//   a wait, no lock, no allocation, no Hash{} or operator== call, noexcept
//   (PROGRESS). What would break the claims, and is therefore not done: a
//   trigger that refreshes or releases the operation's handle before its free
//   pass (its own bag, holding nodes the walk still continues from, would be
//   freed under it: a use after free on the same thread); a free pass that
//   takes a count of 1 for "only my pin" -- the sum of the block's counts <= 1,
//   each count <= 1, or the triggering thread's own shard's count <= 1 -- or
//   that skips the triggering thread's own shard's count (the one pin may be
//   another thread's, on any count; the thread itself holds nodes of its own
//   generation's bag; and the operation's handle may sit on another shard's
//   count than the triggering thread's); a trigger placed before the push
//   (safe, since a push that lands after its shard's exchange goes to the next
//   generation, but the crossing run would then not be in the trigger's bag); a
//   trigger without the flag's acquire exchange (two holders on the plain ring
//   fields). The counting invariants, exact at a quiescent point and asserted
//   by the destructor in debug builds: on a cadence-off set no attempt;
//   `auto_reclaims + auto_skipped` <= the sum over shards of `retire_nodes`
//   divided by R (the intervals (old, old + n] of one shard's pushes partition
//   its count, each multiple lies in exactly one, and a push attempts at most
//   once); at R = 1 the sum equals the number of retired runs (every interval
//   holds a multiple).
//
//   THE HANDLE INVARIANT, in two directions. Write u_X for the unlink CAS
//   that bypassed a dead node X (the CAS on X's PRED WORD, the live word
//   before X's run: a bucket head or a live node's untagged link), and
//   push_X for the retire push that followed it, on the unlinker's own
//   shard's retired head; a node whose push landed in the bag of generation
//   G "is in bag G". Limbo nodes are bagged the same way (their push is
//   push_limbo()'s), and everything below holds for them with "the push"
//   for "the retire push" and "unreachable since birth" for "unlinked".
//   (I) A handle of generation g never observes a node in a bag of
//       generation < g.
//       The chain that heads it is the RETIRE PUSH's release, sequenced
//       after u_X (the unlink CAS's own release is channel 5's, which (I)
//       does not use): push_X (release RMW on the retired head) -> the
//       bump's exchange of that head (acquire: push_X is in the release
//       sequence headed by the push whose value the exchange read, since
//       every push is a read-modify-write) -> the publish of generation
//       G + 1 (seq_cst store, sequenced after the exchange in the same flag
//       holder, or after an earlier holder's exchange through the flag's
//       release/acquire) -> the joiner's seq_cst re-verify that read the
//       publish of g, g >= G + 1 -> every load the handle makes afterwards.
//       So for a handle of generation g, every exchange that bagged nodes
//       into a generation < g, and hence the unlink of every node in such a
//       bag, happens-before its join. It remains to show that a handle
//       whose join u_X happens-before never loads X's address from any
//       word. Four conditions of the chains carry this, and every operation
//       keeps them:
//       C1 (ENTRY)  every walk under a handle starts at a bucket head loaded
//                   with acquire after the handle's join, and reaches every
//                   further node only through the `link` of a node it
//                   reached from that head in the same walk; no walk
//                   resumes from an address retained from before the join
//                   (self_unlink()'s restart compares a retained address by
//                   identity and dereferences only through its own walk from
//                   the head; insert()'s pending node is private; the
//                   child-head load of is_dead_now() tests publication and
//                   is never followed; a split enters the parent chain, and
//                   recursively the grandparent's, through its head; the
//                   advance() walk enters every bucket through its head).
//       C2 (WORDS)  the address part of a published word changes only by a
//                   prepend (insert()'s publishing CAS, or a split's
//                   publishing CAS from UNINITIALIZED) installing a node
//                   that is in no chain, or by an unlink CAS on an UNTAGGED
//                   word installing the first node after a run every node
//                   of which the CAS's thread read as tagged, with acquire;
//                   a mark, a freeze and a seal keep the address; a tagged
//                   `link` is never written again until the node is freed;
//                   a node is linked into a chain once and never moves.
//       C3 (TAGS)   a tag is permanent for the node's life: once a link is
//                   MARKED or FROZEN, every later value of it in its
//                   modification order is tagged too, until the node is
//                   freed; and an unlink CAS expects an UNTAGGED word.
//       C4 (REUSE)  a node's address is reused only after its bag is freed
//                   under the cumulative rule, so the only X-address a
//                   handle can legitimately read is a NEW node at that
//                   address, which the handle never held before.
//       Let w_X be the word u_X wrote and R the run it bypassed. Suppose a
//       handle H whose join u_X happens-before reads a word w and obtains
//       the address of a node N of R; take the first such read. (1) w =
//       w_X: u_X happens-before the read, so by write-read coherence the
//       read returns u_X's value (the first node after R) or a later value
//       of w_X, and by C2 every later value keeps or advances the address
//       past R. (2) w is the link of a node Z before w_X's node L in chain
//       order: for w to hold N, the CAS that installed N into w bypassed a
//       run containing L (Z prepended with N as its successor is excluded
//       by C2: L lies between them and chains prepend only at the head);
//       that CAS's recorder read L's link as TAGGED and continued from the
//       address it held. u_X expected L's link untagged, so by C3 every
//       tagged value of that link lies after u_X in its modification
//       order; the read-modify-write that tagged it read the value just
//       before it in that order, at or after u_X's, and kept its address,
//       which by C2 is the first node after R or a node further on. So the
//       recorder's run walk went from L to a node past R, and the node it
//       installed lies past R as well, not N. The step uses only the
//       modification order of L's link: no acquire of the recorder, no
//       release of u_X, and no order of the tagging CAS (a freeze whose
//       expected value came from a relaxed failure tags L the same way).
//       (3) w is the link of a node of R:
//       then H reached a node of R by an earlier read, contradicting "first
//       such read". There is no other word: an address enters a published
//       word only by a prepend, a split's publish or an unlink CAS (C2).
//       Hence H never reaches R, in particular never X. This single
//       argument covers the two shapes that need naming:
//       EXIT ONTO LATER-RETIRED NODES. A handle of g holding a dead node X
//       (so X is unbagged or in a bag >= g) follows X's immutable link onto
//       its successor S, which may have been retired LATER than X, after a
//       bump, into bag g + 1 or beyond. By the argument, H can obtain S's
//       address from X's link only if u_S does not happen-before H's join,
//       so S is unbagged or in a bag >= g as well: freeing bag g + 1 before
//       bag g would be a use-after-free, and oldest-first cumulative freeing
//       excludes it.
//       THE STALLED UNLINKER (the u_D case). A dead predecessor D's unlink
//       u_D happens-before its successor X's unlink u_X (the unlinker of X
//       reached its pred word by acquire loads along the chain, through the
//       word u_D rewrote or through a word that bypassed it later), yet D's
//       push can land in a LATER bag than X's when D's unlinker stalls
//       between its CAS and its push: bags are not monotone in unlink
//       order. Direction (I) bounds every observable node's bag from BELOW
//       by g and does not care how late a node is bagged; the cumulative
//       rule then keeps the late one, which is in a bag >= g too. The
//       cross-word form of this case, where the word that could still lead
//       H to X is not w_X, is case (2) above: under C1-C4 it reduces to
//       the modification order of one word, L's link, and needs neither
//       the unlink CAS's release nor any acquire of the walks. (The pinned
//       orders of the unlink layer, channels 4 and 5, are for its own
//       arguments: the miss argument and the seal.)
//   (II) Contrapositive: a node in bag G is observable only by handles of
//        generation <= G. This is what the CUMULATIVE RULE needs: a handle
//        of generation g observes nodes whose unlink does not happen-before
//        its join, and those land in bags g, g + 1, ...; freeing bag G
//        therefore requires that EVERY per-shard count of EVERY generation
//        <= G was read 0 (by a seq_cst load after the publish that closed
//        that generation), not only G's own counts. Oldest-first freeing
//        from `oldest_` -- stop at the first block that is not SEALED or
//        at the first of its counts that is not 0 -- delivers it by
//        construction: bag G is examined only after every bag < G was
//        freed, each after all of its counts read 0, possibly by an
//        earlier flag holder (FACT 2 of THE JOIN covers any holder). For a
//        handle h of generation g <= G: when bag g was freed, the load of
//        h's own count read 0 only after h's leave, which that load
//        acquired (THE JOIN), and that free pass precedes the one that
//        frees bag G (in one holder, or through the flag).
//   Together: every node address a handle of generation g can observe -- in
//   a chain, by EXIT through a dead node's link, or in a free-list head --
//   is either not bagged or in a bag of generation >= g, and that bag
//   outlives the handle.
//
//   THE PER-NODE REUSE CHAIN. (I) and (II) say WHICH nodes a handle keeps; the
//   reuse of a freed node is a plain write of a plain `T` over bytes a reader
//   may have read, so the reader's last read must happen-before the popper's
//   write through a chain of synchronizations, every link of which is
//   load-bearing: the reader's last read of X's `value` or `link` ->
//   (happens-before: sequenced before on the reader's thread, or ordered by the
//   caller's synchronizing hand-off when the handle is destroyed, assigned
//   over, refreshed, moved from or swapped on another thread, as THE HAND-OFF
//   RULE under HANDLES requires) its handle's leave, a RELEASE `fetch_sub` on
//   the count the handle recorded at its join -> the free pass's seq_cst load
//   of THAT count that reads 0, or a later value in the release sequence the
//   decrement heads (the pass's loads of the block's other counts acquire other
//   handles' leaves, not this one) -> (sequenced before) the free pass's
//   rewrite of X's `link` and its push of X with a RELEASE CAS on a free head
//   -> the popper's ACQUIRE load of that head, or the acquire failure of its
//   CAS that returns a later value -> (sequenced before) the popper's
//   `node->value = val`. A relaxed leave, a relaxed count load, a relaxed free
//   push or a relaxed pop each break the chain at that link, and the result is
//   a data race on `value`, whatever the bag arithmetic says. The same chain
//   covers the free pass's own read of X's words. The walk's internal handle is
//   NOT a link of either chain: under the flag no bag is freed while the walk
//   runs (the free pass of the call precedes it, and no other advancer runs),
//   and every node the call freed had its unlink happen-before the walk's head
//   loads -- the closing bag's nodes through this holder's own exchanges, older
//   bags' through the previous holders' and the flag -- so by (I)'s argument
//   the walk never loads them. The handle is kept for uniformity: a walk, like
//   every other chain reader, runs under a handle, and its join costs nothing
//   on an operation path.
//
//   ABA WITHOUT VERSION TAGS. Two CASes assume "the value I expected still
//   names the node I read": the free-list POP (alloc_node()) and the UNLINK CAS
//   (and insert()'s and erase()'s deciding CASes with it). Both are excluded by
//   the handle invariant together with the single advancer; the free head's
//   9-bit counter plays no part (it batches node_count_, see the FREE-LIST HEAD
//   encoding) and no word carries a version tag. The premise: every pop runs
//   under the caller's live handle for the whole pop loop (alloc_node() is
//   called only from insert() and split_bucket(), both under the caller's
//   handle), and every unlink CAS's expected value (the pred word, and the run
//   recorded through acquire loads) is loaded and used within one operation
//   under one handle. THE POP. A popper A of generation g loads the head (N, c)
//   and reads N's `link` = M. For A's CAS to succeed wrongly, N must leave the
//   free list (another popper B pops it), be published, die, be retired, be
//   bagged in some G*, be freed and be pushed back, restoring the head's
//   address while A holds M -- a stale next. B's pop CAS read the value A's
//   load read, or a later one; B's push_N happens-before the exchange of bag
//   G*. If G* < g, that exchange happens-before A's join (the chain of (I)),
//   which happens before A's load of the head (A's operation runs under the
//   handle after its join: sequenced on one thread, or through the
//   synchronizing hand-off that gave A the handle: THE HAND-OFF RULE under
//   HANDLES), so B's pop (sequenced before B's push) happens- before A's load,
//   and by write-read coherence A's load reads B's pop's write or a later value
//   -- not the N-headed value B consumed. Contradiction; hence G* >= g, and
//   freeing bag G* needs every count of every generation <= G* read 0, g's
//   among them (II), while the count A's live handle recorded reads at least 1
//   until A leaves (THE JOIN). So N is not back on the free list while A's CAS
//   runs, and the CAS fails. The counter bits add nothing to this. THE UNLINK
//   CAS. An operation E of generation g read its pred word with X's address and
//   recorded the run X.. through acquire loads; for its CAS to succeed wrongly,
//   X must be unlinked by another thread, retired, bagged, freed, popped and
//   re-published under the same pred word with a different successor. X
//   observed by E is unbagged or in a bag >= g (I), and that bag is not freed
//   while E lives (II), so X is never reused while E can CAS on a word that
//   names it; the run E recorded is therefore the run its successful CAS
//   bypasses. The same argument covers insert()'s publishing CAS on a head and
//   erase()'s mark CAS on a link. WHY THE SINGLE ADVANCER IS PART OF THE
//   ANSWER: the chain of (I) runs through "every exchange precedes the publish"
//   within one flag holder and through the flag's release/acquire between
//   holders; two advancers would have exchanges of one call interleaved with
//   the publish of another, and a node could be bagged behind a publish a
//   handle has already acquired.
//
//   SESSIONS. The set has no quiescent moment and no period of its own; the
//   unit ADDRESS REUSE (under RECLAMATION) rests on is the handle's SESSION,
//   its join to its leave or refresh: within one session, no address the
//   handle can observe is published twice. A node leaves a chain by an
//   unlink CAS or dies in it; its address is bagged at a later bump, freed
//   only when no handle of its bag's generation or an older one is live,
//   and re-enters a chain only through a pop that happens-after that free;
//   a node the handle can observe is therefore not freed, and not
//   re-published, before the handle leaves. For the CLIENT: membership is
//   the only thing the operations observe across an advance(). A slot hosts
//   one node life per session of every handle that can observe it, and the
//   MARK/FROZEN bits stay terminal per node life. Reuse (the copy
//   assignment over the old value) happens while other operations run; it
//   is safe because the node is popped by one thread while on no chain,
//   after every handle that could observe it has left or refreshed (the
//   reuse chain). A throwing assignment routes the node to limbo, reused
//   only after a later bump and free pass.
//
//   THE LIMBO LIST under the bump. A limbo node was never published, but a
//   node popped from a free list whose assignment threw may still have its
//   `link` read by a popper that lost the pop race (alloc_node()), so limbo
//   goes through the bag like every retired node, never straight back onto
//   a free list (a stale pop CAS would then succeed by ABA). The push's
//   success order must be RELEASE, for two reasons: the free pass walks the
//   limbo sublist through `link`, and the exchange's acquire synchronizes
//   only with release pushes -- with a relaxed push the free pass reads a
//   node's `link` before the pusher's store of it is visible; and, second,
//   the pusher's store of the last node's `link` (the limbo next pointer)
//   and the free pass's later rewrite of that same `link` (the free-list
//   next pointer) are two writes to one word, which happens-before orders
//   in modification order only if the push is release and the exchange
//   acquire (write-write coherence). The first reason has a known failing
//   execution; the second is reasoned from the standard: a failure needs
//   the two writes to one word to land in modification order against
//   their execution order. The retire push's release is already in place
//   for the same reason (Shard::retired_head).
//
//   WHAT THIS DOES AND DOES NOT GUARANTEE. MADE UNREACHABLE (the unlink
//   layer) is best effort: one CAS per dead run, stragglers collected by
//   later walks, splits and advance()'s walk; nothing promises that a given
//   dead node ever leaves its chain. MADE REUSABLE (this section): a
//   retired node is freed when no handle of its bag's generation or an
//   older one is live, and never otherwise; a node freed is never one a
//   live handle can observe. There is NO bound on when: a handle holder
//   that stalls (preempted, suspended, or simply never refreshing) pins its
//   generation and every node retired since for as long as it stalls, the
//   set cannot tell a stalled holder from a slow one, and after
//   kGenerations - 1 bumps under such a pin every bump -- advance()'s,
//   reclaim()'s or a trigger's -- refuses with `ring_full` while retired
//   nodes pile up on the retired lists and every insert that finds no
//   free node grows the arena (GROWTH UNDER LAGGING HANDLES, under THE
//   TRIGGER). This is
//   the trade of epoch-based reclamation, accepted here; refresh(),
//   `ring_full` and the diagnostics (live-handle counts and bagged counts
//   per generation, the counts as sums over the shards or per shard) are
//   what a caller has to see a stall. Latency of the
//   explicit calls (under the cadence: THE TRIGGER): with a long-lived
//   handle refreshed before each advance() or reclaim(), a node retired
//   between two calls is freed TWO calls after its retirement (the first
//   closes its bag, which the refreshed handle still pins; the second frees
//   it); with every handle released before each call, by the NEXT call. A
//   straggler the walk of call k bypasses is freed by call k + 1 with every
//   handle released, and by call k + 2 under a handle refreshed before each
//   call (call k + 1 bags it into the generation that handle pins).
//
//   TEST SEAMS, outside the operational contract. Two kinds.
//   The `Hooks` template parameter names three static `noexcept` functions
//   the set calls at fixed positions of the join and of the bump-and-free
//   that advance(), reclaim() and the trigger share (see
//   NoHooks for each): tests force the interleavings the store-buffering
//   argument depends on -- the join before its increment, the increment
//   before its re-verify, operations between the publish and the free
//   pass -- deterministically; the default NoHooks compiles to nothing, and
//   no hook is in the destructor or in the sweep; on a cadence-off set no
//   hook is on an operation path either, while on a cadenced set the
//   publish hook fires from the trigger, inside the operation that retired
//   (THE TRIGGER). The hooks are `noexcept`, so a hook action that throws
//   terminates the program. A hook action may run complete operations,
//   advance() and reclaim() on the same set (a nested advance() or
//   reclaim() inside in_advance_after_publish() returns `contended`: the
//   same thread holds the flag; a nested operation's retire push there is
//   a skipped trigger) but must not refresh, move, swap or destroy the
//   handle the interrupted join is creating or refreshing, nor the handle
//   of the operation a trigger interrupted, nor call the diagnostic sweep (get_internal_accounting()
//   and everything derived from it), and inside in_advance_after_publish()
//   -- or a join hook nested in it -- nor get_internal_generation(),
//   get_internal_ring_state() or get_internal_bag_count(): each asserts the
//   advancer flag clear and aborts (get_internal_handle_generation() of a
//   live handle is fine anywhere).
//   The get_internal_* accessors (DIAGNOSTICS, before them) show the
//   arena, the ring and the counters, each under its own precondition:
//   RACE-FREE ones (counts, the table size, the counters) may run beside
//   anything and are exact at a quiescent point; RING VIEWS need no flag
//   holder in progress (no advance(), no reclaim(), and on a set
//   constructed with a cadence no operation); the QUIESCENT ones -- the
//   accounting sweep and everything derived from it -- need no operation,
//   no join attempt and no advance() or reclaim() in progress on any
//   thread, every other call ordered against
//   them by happens-before, live idle handles allowed.
//
//   ASAN POISONING. Under AddressSanitizer a node's VALUE bytes -- the
//   extent [&value, &link) by address difference, never sizeof(T), which
//   under ASan's 8-byte granularity would poison nothing for an int --
//   are poisoned by the free pass before the node is pushed onto a free
//   list, unpoisoned by the pop that wins the node before it assigns the
//   new value, and unpoisoned by the destructor before the arena destroys
//   the slots. `link` and `retire_link` stay readable: the free list links
//   through `link`, a losing popper reads it, and the sweep reads both. A
//   read of a freed value through a stale address is then an ASan report
//   instead of a silent wrong answer, and the accounting sweep checks that
//   every free node is poisoned exactly so (its C6). A no-op without ASan;
//   a diagnostic that no contract promises. The header detects ASan itself
//   (the nested test at the top of the file).
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
//   true per absent->present transition and at most one erase() returns true
//   per present->absent transition -- exactly one, unless the call whose CAS
//   made the transition then exits by an exception (EXCEPTIONS).
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
//   the node is freed, and why a tagged link is never a CAS target.
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
//   (i) A miss is linearized before every insert whose publishing CAS does
//       NOT happen-before the call: no mechanism is needed, and no test may
//       assert a hit for a key unless the insert's publishing CAS -- not
//       merely its invocation -- happens-before the call, through the
//       inserter's return or through any call that observed the key and
//       then synchronized with the caller.
//   (ii) For an insert whose publishing CAS DOES happen-before the call, a
//       miss is final only if the key was deleted:
//       (a) if the key's node was MARKED and then bypassed, the deletion
//           linearizes at the mark CAS (the marking CAS is THE deletion, see
//           above), and contains() and erase() skip a MARKED node anyway.
//       In every other case the miss is never final:
//       (b) if the key's FROZEN parent copy was bypassed after a later split
//           published the copy's bucket, the reader's reload of table_size_
//           returns a strictly larger size (channel 5), so the walk retries
//           in the child, at most once per doubling.
//       (c) Within (b): the copy's child is then PUBLISHED to the reader,
//           never UNINITIALIZED: the unlinker read the child head published
//           before its CAS, and the reader's acquire of the CAS's value
//           orders the reader's own child-head load after that read, so by
//           read-read coherence it returns the published value or a later
//           one. If the reload sent the reader to a deeper bucket on the
//           key's path that is still UNINITIALIZED, that is a first access,
//           and the reader splits it from the published child as any
//           operation would; a losing splitter is the P9 case.
//       The two cases are complete because no unlink bypasses a live node or
//       a FROZEN node whose child is unpublished (is_dead_now()): the only
//       nodes a walk can miss through a bypass are tombstones, (a), and
//       superseded copies, (b).
//       (d) Within (b): the edge (b) rides is the unlinker's OWN: it decided
//           the copy dead at a table size it had READ, which covers the child;
//           that read is sequenced before its release CAS, and the reader,
//           which acquires the CAS's value (or a later value of the word,
//           through the release sequence that read-modify-writes extend), has
//           the read happen-before its own reload, so by read-read coherence
//           the reload returns that size or a newer one, whatever the order
//           of the unlinker's load. The acquires of the FROZEN link and of the
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
//      by a CONCURRENT operation would break this argument: never add one;
//      no store to a published head exists in this class. The free pass's
//      relaxed store to a freed node's `link` is not one: nothing is written
//      to a link after MARK or FROZEN while the node is in a chain, and the
//      free pass writes the link of a node that is on no chain and that no
//      handle can observe (HANDLES AND GENERATIONS).
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
//      The retire push (push_retired()) is a release CAS so that the bump's
//      acquire exchange of a retired head orders everything before every
//      push below it, the unlink CASes foremost: that edge heads the chain of
//      direction (I) of THE HANDLE INVARIANT (HANDLES AND GENERATIONS), and
//      the unlink CAS's own release is not what carries it.
//   Linearizability throughout this class is with respect to happens-before,
//   which is all the C++ memory model can express: "an erase() that returned
//   before this call began" means the return happens-before the call (e.g. the
//   caller synchronized), not wall-clock order.
//
// WEAK MEMORY: WHAT THIS DESIGN DOES AND DOES NOT RELY ON
//   The decisions above are atomic read-modify-writes of a single word, which
//   are totally ordered with every other write of that word on every
//   architecture. No correctness argument of the OPERATIONS has the
//   store-buffering (Dekker) shape "A: write x, read y  ||  B: write y,
//   read x", which is the shape a post-CAS recheck would have (x = head or
//   link, y = table_size_) and which acquire/release on two DIFFERENT
//   atomics does not order. The one argument in this class that HAS that
//   shape is the handle join against the generation publish (THE JOIN under
//   HANDLES AND GENERATIONS: joiner "+1 on its shard's count, then load
//   current_" against advancer "store current_, then load every count", a
//   pair per count), and it is closed by seq_cst on all four operations
//   (the increment, the re-verify load, the publish store and the free
//   pass's load of each count), not by acquire/release. Store
//   buffering is the one reordering x86 itself performs, so the pair needs
//   its seq_cst on every architecture, x86 included. The directory's
//   store-buffering regression test (concurrent_hash_set_rcu_tso_test.C)
//   does NOT cover it: that test guards the operations' pair above (a
//   bucket-head CAS against a resize), and the hook tests (TEST SEAMS)
//   force interleavings of the join, never a reordering of its accesses.
//   The pair rests on the standard's single total order of seq_cst
//   operations, reasoned, not on an observed execution.
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
//   cannot return an older value, whatever the order of that read.
//   A miss that is confirmed by an unchanged table_size_ (contains();
//   erase(), whose walk must also have found no FROZEN node of the key)
//   linearizes at a point inside the call; the two halves under STALE
//   GEOMETRY are the argument that such a point exists, and this comment does
//   not name it.
//
// PROGRESS: the hash's own protocol takes no lock. The walks, the splits,
// the unlinks, the publishing CAS and every retry loop are lock-free, and
// the pure read/traverse path writes nothing; none of it is wait-free. On a
// set constructed with a cadence the retire path also runs the automatic
// trigger, after the retire push (THE TRIGGER under HANDLES AND
// GENERATIONS): added WORK on the
// retiring thread -- the bump's loads and exchanges and the free pass of
// every unpinned bag, bounded by the bagged nodes, non-blocking by refusal
// of the try-flag, lock-free, allocation-free and noexcept -- never a wait. The
// class takes a lock in only two places, both on the allocation side, and
// only an operation that allocates can reach either, so an operation that
// allocates nothing is lock-free end to end:
//   (1) alloc_node(): a lock-free pop from the calling thread's arena shard's
//       free list when it has a node, else, unless the steal's hint says no
//       free list has a node, a lock-free pop from the first shard's list,
//       own first, that has one (the steal), else an append under the own
//       shard's deque SpinLock; with one thread per shard that lock is uncontended,
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
// lock with whichever thread allocates the block. Every loop that unlinking
// adds is lock-free in the sense that a failed CAS implies another thread's
// completed step: the mark and freeze retries on a live failure value (each
// failure is a completed unlink through the node), the retire push's and
// the limbo push's CAS loops (each failure a completed push, or a spurious
// failure of its weak CAS, a plain retry), and erase()'s restart (one, after
// a lost CAS); the opportunistic unlink sites, the cleanup pass and
// advance()'s walk make ONE attempt per run and never retry, and erase()'s
// self-unlink makes at most two. The generation protocol adds no lock and
// no blocking wait either: the join retries only after an advancer
// published a new generation (a completed step of another thread) and
// never waits for an advance(); advance() and reclaim() are non-blocking
// by REFUSAL (a try-flag: a caller that finds it set returns `contended`
// at once, a trigger that finds it set is skipped, and nothing ever waits
// for the holder); the free pass's push loops are
// lock-free (each failure a completed pop of the same list, or a spurious
// failure of the weak CAS); the free-list pop, the steal's included, is a
// lock-free loop against concurrent pops and pushes (each failure a
// completed pop or push of the same list, or a spurious failure).
//
// Template parameters:
//   T           : element (key) type; must be copy-constructible (a new node
//                 is constructed from the key), copy-assignable (a node
//                 reused from a free list takes its new value by
//                 assignment; see EXCEPTIONS for what a throwing assignment
//                 must satisfy; no T is destroyed before the set's
//                 destructor: LIFETIME OF VALUES), destructible without
//                 throwing (the set's destructor, which is noexcept, destroys
//                 every stored T), and
//                 equality-comparable as `value == key`, a stored T against
//                 a const T&, consistently with Hash:
//                 keys that compare equal must hash equal, since an
//                 operation looks for its key only in the bucket the key's
//                 own hash selects.
//   AllowDelete : when true, compiles erase() (tombstone deletion). When
//                 false, no node is ever marked; the freeze step of
//                 split_bucket() runs for both values (every superseded
//                 parent copy is FROZEN), so a chain's links still gain a
//                 tag, and the MARK checks on the read paths are tests of a
//                 bit that is never set.
//   Hash        : hash functor; must be default-constructible, with Hash{}(k)
//                 well-formed for a const T& k and returning a size_t, the
//                 SAME value for a key every call (the split math re-hashes
//                 keys under wider masks). No Hash object is stored: every
//                 hash is computed by a freshly constructed Hash{}, so a Hash
//                 with state only ever has its default state. Besides the key
//                 of every call, Hash{} is applied to stored values: by a
//                 split, to every node of the parent snapshot that is not
//                 MARKED (to select the nodes that move); by writer walks,
//                 cleanup passes, erase()'s self-unlink and advance()'s
//                 walk, to the FROZEN nodes they meet (to find the node's
//                 child bucket); and by get_internal_accounting(), to every
//                 reachable node that is not MARKED. A walk or cleanup pass
//                 hashes a FROZEN node it passes at most once, at most twice
//                 if the node ends a dead run (the second hash follows only
//                 a won CAS), and not at all when its pred is unusable; the
//                 self-unlink adds its own (PINNED ORDERS, H5', H5 and H6).
//                 These hashes of FROZEN nodes are also where a throwing
//                 Hash{} leaves a dead run behind (EXCEPTIONS).
//   kGenerations: the number of generation slots in the ring (THE RING under
//                 HANDLES AND GENERATIONS), at least 2: one slot is always
//                 CURRENT, so kGenerations - 1 generations can be SEALED and
//                 pinned at once before a bump (advance()'s, reclaim()'s or
//                 a trigger's) refuses with `ring_full`.
//                 Fixed per instantiation; the ring is part of the set
//                 object.
//   Hooks       : the test seams (TEST SEAMS under HANDLES AND GENERATIONS),
//                 a type satisfying concurrent_hash_rcu_detail::SetHooks;
//                 the default, concurrent_hash_rcu_detail::NoHooks, compiles
//                 to nothing. Last on purpose: existing instantiations name
//                 at most T, AllowDelete and Hash.
// ===========================================================================
template <
    typename T,
    bool AllowDelete = false,
    typename Hash = std::hash<T>,
    size_t kGenerations = 64,
    concurrent_hash_rcu_detail::SetHooks Hooks = concurrent_hash_rcu_detail::NoHooks
>
class ConcurrentResizableHashSetRCU {
    static_assert(kGenerations >= 2, "the ring needs a CURRENT slot and at least one SEALED slot");
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
    //                   concurrent operation, and only the free pass
    //                   rewrites it, when it frees the node for a
    //                   pop to reuse (a free node's link is a bare address,
    //                   no tag bits).
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
    // which is the wrap of a 9-bit counter. Every 256th pop adds 256 to
    // node_count_: the pop is the 256th exactly when the counter it replaces
    // has its low eight bits clear (FREE_BATCH_MASK); the counter starts at
    // zero, so a shard's first pop counts, and it is never cleared (the free
    // pass's push_free() keeps it, and the counter bits are no part of the
    // pop's ABA argument: RECLAMATION in the class overview).
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
    // Bit 0 of steal_hint_ (THE HINT at steal_scan()): set when a scan of
    // the steal found every free list empty; the stamp is in bits 63..1.
    static constexpr word_t STEAL_HINT_EMPTY = 1;
    static_assert((PTR_MASK >> 3) < FREE_POP_ONE, "an address >> 3 must fit below the pop counter");

public:
    // The session type every operation runs under; defined after the
    // diagnostics at the end of the class, where every new member of the
    // generation protocol lives (see HANDLES AND GENERATIONS in the class
    // overview). Declared here so the operations below can name it.
    class handle;

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
        // (MARK or FROZEN) is written by no concurrent operation, and by the
        // free pass only when it frees the node, which happens
        // only once no handle can observe it, so a dead node can always be
        // EXITED through the successor it held when it died, for as long as
        // a handle that reached it lives. On a free or limbo list the same word is
        // the list's next pointer (a bare address); a retired node keeps its
        // chain link intact (the retired lists use retire_link).
        std::atomic<word_t> link;
        // The RETIRED LIST's next pointer: a bare node address or EMPTY, tag
        // bits zero. A separate word, and not `link`, because a retired node
        // is still EXITED by concurrent operations that reached it before it
        // was unlinked: its `link` must keep the chain successor (see above).
        // Written only by push_retired() (the retiring thread, privately for
        // the run and then as the list's next pointer) and read only by the
        // free pass, after a bump's acquire exchange of the
        // head, and by the accounting sweep; no concurrent operation reads
        // it, so its stores are relaxed and only the retired head's CAS
        // carries ordering. The nine bits an address never uses (63..58 and
        // 2..0) are zero here and unused by this class. LAST member on purpose: `value` and `link` keep
        // their relative placement, and the hot read path never touches this
        // word. A free or limbo node's retire_link is stale and never read.
        std::atomic<word_t> retire_link;

        Node(const T& val, word_t next) : value(val), link(next), retire_link(EMPTY) {}
    }; // struct Node
    static_assert(alignof(Node) >= 8, "the low three bits of a node address are the link's tag bits");

private:
    // The dynamically resizable array of atomic bucket heads.
    // Each entry holds the address of the first node in the bucket's chain,
    // plus the bucket's seal level (see the word encoding).
    ConcurrentAppendDeque<std::atomic<word_t>, ARENA_BLOCK> buckets_;

    // One arena shard: the append-only node deque, and the heads of the
    // shard's free, limbo and retired lists (see RECLAMATION in the class
    // overview) with the shard's diagnostic counters, on three cache lines of
    // their own, apart from the deque's members (which the deque's default
    // build pads onto lines of their own):
    //   - the FREE LINE: free_head and limbo_head. The free head is written by
    //     every pop of the shard's threads, by the pops of threads of other
    //     shards stealing (steal_node()) and by the free pass's pushes, and
    //     with more threads than shards a pop CAS and the deque's lock
    //     exchange would otherwise fight over one line; the limbo head is
    //     written by lost splits (a whole subchain per lost publish, which
    //     under contended doublings is not rare), orphans and throwing
    //     assignments, and is exchanged by the bump only when the bump's
    //     emptiness load found it non-EMPTY, so an idle limbo list costs the
    //     free line nothing.
    //   - the RETIRED LINE: retired_head, one unused placeholder word, and
    //     the two retire and four loss counters, which fill it. The retired head
    //     takes a release CAS from every operation that unlinks a dead run:
    //     the erase path, the writer walks of insert() and erase(), and the
    //     split winner's cleanup pass (which any operation may run,
    //     contains() included). On the free line that CAS and the pop CAS
    //     would alternate ownership of one line between an inserting and an
    //     erasing thread whose numbers collide modulo the shard count. The
    //     retire counters are bumped right before the push CAS, by the thread
    //     about to own the line; the loss counters are bumped by a thread
    //     that may not own it (a loss is a CAS that failed, with no push to
    //     follow).
    //   - the line after it: the two split counters, for which the retired
    //     line has no room, and the two loss counters of advance()'s walk;
    //     nothing else of the shard is on it, and each increment of one of
    //     them takes ownership of it.
    //   No counter is on the free line.
    struct Shard {
        // Nodes are appended block-by-block and never destructed until the set
        // is destroyed; a slot's address is stable for the life of the set.
        ConcurrentAppendDeque<Node, ARENA_BLOCK> nodes;
        // Free list: the encoded top node address and pop counter (see the
        // FREE-LIST HEAD encoding). Popped with an acquire CAS by
        // alloc_node() on the shard's own threads and by steal_node() on
        // threads of other shards, pushed by the free pass (of advance(),
        // reclaim() or an automatic trigger: one flag holder at a time) with a release CAS that
        // keeps the counter bits (push_free()); the counter is never reset.
        // The two run concurrently: the orders are THE PER-NODE REUSE CHAIN
        // in the class overview.
        alignas(64) std::atomic<word_t> free_head{FREE_EMPTY};
        // Limbo list: the bare address of the top node, or EMPTY; the nodes
        // link through their `link` words. Pushed with a RELEASE CAS by the
        // shard's threads (push_limbo()), exchanged to EMPTY with acquire by
        // a bump when its emptiness load found the list
        // non-EMPTY (into the closing generation's bag, where the free pass
        // walks it through `link`), read by the accounting sweep. Push-only
        // between two exchanges. The push CAS is ABA-benign: a pusher never
        // dereferences the head value it read, it only stores that value
        // into the `link` of its own private tail, so a CAS that succeeds
        // after the head changed and changed back (a node exchanged away,
        // freed, reused and pushed again in a later life) links the tail to
        // the node that IS the head at the CAS, which is all a push needs.
        std::atomic<word_t> limbo_head{EMPTY};
        // Retired list: the bare address of the top retired node, or EMPTY;
        // the nodes link through their `retire_link` words, never through
        // `link` (a retired node keeps its chain link: see Node). LIFO and
        // push-only between two bumps: push_retired() pushes a run with one
        // RELEASE CAS on behalf of the thread that unlinked it (its own
        // shard, see there), a bump (advance()'s, reclaim()'s or an
        // automatic trigger's) exchanges the head to
        // EMPTY with ACQUIRE into the closing generation's bag, and nothing
        // pops. The push CAS is ABA-benign for the same reason the limbo
        // push is (the pusher stores the head it read into its own tail's
        // `retire_link` and dereferences nothing through it). The release on every push means the exchange's
        // acquire is a happens-before boundary for the whole list below the
        // value it took: the push whose value was loaded is a
        // read-modify-write and so lies in the release sequence of every
        // push below it, and the advancer synchronizes with all of them and
        // sees every unlink that preceded any of those pushes (the chain of
        // direction (I) of THE HANDLE INVARIANT in the class overview).
        alignas(64) std::atomic<word_t> retired_head{EMPTY};
        // Unused: a placeholder word that keeps every counter below at its
        // offset, so that the six counters fill the rest of the retired line
        // and the split counters start the next line (the layout comment
        // above). Removing it moves split_attempts onto the retired line.
        word_t placeholder{EMPTY};
        // Diagnostic counters, each bumped with a relaxed fetch_add by the
        // thread whose operation took the counted step (so they are exact
        // when read at a quiescent point, and a snapshot otherwise); summed
        // over the shards by get_internal_counters(). Always compiled: the
        // benchmark reads them in its NDEBUG build. Their cost is a relaxed
        // fetch_add, which is still an atomic read-modify-write, per counter
        // per counted event: two per retired run (push_retired(), on the
        // retired line just before the push CAS takes it), one or two per
        // lost one-shot CAS of a walk or a cleanup pass (on the retired line,
        // which the losing thread may not own; the self-unlink's lost CASes
        // are not counted), one per split attempt and one per won publish
        // (on the line after the retired line); see the layout comment above.
        std::atomic<size_t> retire_runs{0};             // push_retired() calls: one per unlinked run
        std::atomic<size_t> retire_nodes{0};            // nodes those runs held in total; the automatic cadence's input (THE TRIGGER in the class overview): the value its fetch_add returns decides whether the push triggers
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
        std::atomic<size_t> split_attempts{0};          // split_bucket() calls that reached the walk of the sealed parent, recursive parent splits included
        std::atomic<size_t> splits_published{0};        // of those, the ones whose publishing CAS won
        // The loss counters of advance()'s walk (HANDLES AND GENERATIONS in
        // the class overview), the same two kinds as the walk and cleanup
        // counters above, on the same line as the split counters; bumped by
        // the advancing thread, which may not own the line.
        std::atomic<size_t> advance_cas_failures{0};    // lost one-shot CASes of advance()'s walk, all kinds
        std::atomic<size_t> advance_cas_peer_losses{0}; // of those, peer losses
    }; // struct Shard

    // The node arena: arena_mask_ + 1 (a power of two) shards. A thread
    // appends to, and first pops from, shards_[thread_number & arena_mask_];
    // it pops from another shard's free list only when its own is empty
    // (steal_node()). Fixed at
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

    // The resize hint: arena occupancy, over-counted by up to 255 per touched
    // shard's deque and per free list, minus every node a free pass has
    // freed. A shard adds 256 when the index of a node it
    // appends is a multiple of 256, index 0 included, so a shard's first
    // node counts as 256; and 256 on every 256th pop from a free list, the
    // list's first pop ever included (see the FREE-LIST HEAD encoding: the
    // counter is never reset), whichever thread pops -- one of the shard's
    // own or a stealer from another shard (alloc_node()): the batch belongs
    // to the list's counter, and the popper that opens a batch adds its
    // 256; the free pass subtracts the exact number of nodes it moves to the
    // free lists, one fetch_sub per bag. The error is one-sided on purpose:
    // an over-count doubles the table early (memory: a set touched by k
    // shards grows to at least 128k buckets, small next to the k node blocks
    // the shards themselves hold), where an under-count would let chains
    // grow long before a doubling. Why the subtraction keeps it one-sided AT
    // EVERY QUIESCENT POINT (every batch's increment landed): per deque the
    // appends added 256*ceil(appended/256) >= appended, per free list the
    // pops added 256*ceil(popped/256) >= popped, the frees subtracted exactly
    // freed, and the occupied slots number appended + popped - freed summed
    // over deques and lists (a slot is appended once, and every later life
    // of it is one pop, from whatever list the free pass dealt it to, after
    // one free); so the count is at least the occupied slots (the accounting
    // sweep asserts `node_count_ >= slots - free`). WHILE OPERATIONS RUN the
    // count can lag the occupied slots, by the batches whose opener has not
    // yet added its 256 (at most 256 each), but it never dips below zero
    // (below). No exact live count exists: this class has no quiescent
    // moment in which to take one. Without the subtraction appends plus pops
    // would over-count without bound under churn and the table would double
    // forever.
    // WHY IT NEVER DIPS BELOW ZERO, so that an unsigned word is safe. A
    // batch's +256 is added AFTER the batch-opening pop's CAS (alloc_node())
    // or append, so one might fear a node of that batch, taken by another
    // thread, being published, erased, bagged and freed -- its -1 landing
    // before the +256 that covers it, and wrapping the count. It cannot: the
    // opener holds a live handle of some generation g for the whole of its
    // allocating operation, the +256 included, and every node of its batch
    // is bagged in a generation >= g. A batch-mate's pop follows the
    // opener's CAS in that free head's modification order (or its append
    // follows the opener's under the shard's lock). Were the batch-mate's
    // node bagged in a generation below g, the batch-mate's pop (or append)
    // would happen-before that bag's exchange (the node is popped,
    // published, erased and retired before it is bagged), the exchange
    // before the publish of g, the publish before the opener's join and so
    // before the opener's CAS (or append); for the pop, write-write
    // coherence then puts the batch-mate's pop before the opener's CAS in
    // the free head's modification order, and for the append the lock's
    // order would be a happens-before cycle: either way a contradiction.
    // A bag of generation >= g is freed only after the opener's handle
    // leaves (THE HANDLE INVARIANT, direction (II)), and that leave happens
    // after the +256 (sequenced after it on the opener's thread, or ordered
    // by the caller's synchronizing hand-off when the opener's handle is
    // left on another thread: THE HAND-OFF RULE under HANDLES) and is
    // acquired by the free pass's load of the count the
    // opener's handle recorded (THE JOIN):
    // the +256 happens-before the -1, and coherence puts it first in the
    // count's modification order. Every free is thus preceded in that order
    // by the increment of the batch that covers its node's life, each batch
    // covers at most 256 lives, and the count stays at or above zero. The
    // argument never asks which shard the popper maps to, so a stolen pop
    // is covered as an own-shard pop is. The accounting sweep's conjunct C5
    // checks the quiescent bound, and a churn test under stress checks that
    // the table stays bounded.
    // The exact slot count is the sum of the shards' sizes
    // (get_internal_node_count()), which is too many acquire loads of
    // frequently written lines to do per insert. On its own cache line: it
    // is written from every shard, and table_size_ is read by every
    // operation.
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

    // The arena shard the calling thread appends to, pops from first and
    // pushes limbo onto (a stolen pop takes a node from another shard's free
    // list: steal_node()).
    Shard& my_shard() { return shards_[concurrent_hash_rcu_detail::thread_number() & arena_mask_]; }

    // Push the private chain first..last (linked through their link words,
    // nothing else references them) onto `shard`'s limbo list, where the
    // next bump will find it (THE LIMBO LIST under HANDLES AND
    // GENERATIONS in the class overview). Never counted toward the automatic
    // cadence, which counts retired nodes only. Treiber push, ABA-benign: the
    // pusher never dereferences the head value it read, it only stores it
    // into `last->link`, so a CAS that finds the head value again links the
    // tail to the node that is the head at the CAS. The head load and the failure order
    // are relaxed: the pushers exchange nothing but the head word itself
    // (each links its own private tail), and the failed value is only the
    // next expected value (EXPECTED-EQUIVALENT to acquire). Success is
    // RELEASE, REQUIRED for two reasons, neither of them a pusher: the
    // bump's acquire exchange synchronizes only with release pushes, and the
    // free pass then walks the sublist through `link` -- with a relaxed push
    // it can read a node's `link` before the pusher's store of it is
    // visible (an execution that fails exists); and the pusher's store of
    // `last->link` (the limbo next pointer) and the free pass's later
    // rewrite of that same word (the free-list next pointer) are two writes
    // to one atomic, which happens-before orders in modification order only
    // through this release and the exchange's acquire (write-write
    // coherence; that half is reasoned from the standard). Weak CAS:
    // a retry loop anyway. Counted (limbo_pushes, one per subchain) on the
    // generation protocol's counter line: a lost split pushes a whole
    // subchain and a hot growth pushes many, so the count is the measure of
    // how often the bump's limbo exchange is worth its line.
    void push_limbo(Shard& shard, Node* first, Node* last) {
        advance_counters_.limbo_pushes.fetch_add(1, std::memory_order_relaxed);
        word_t old_head = shard.limbo_head.load(std::memory_order_relaxed);
        do {
            last->link.store(old_head, std::memory_order_relaxed);
        } while (!shard.limbo_head.compare_exchange_weak(old_head, word_of(first), std::memory_order_release, std::memory_order_relaxed));
    } // push_limbo()

    // Retire the run first..last of dead nodes that the calling operation has
    // just unlinked from its chain: push it, as one unit, onto `shard`'s
    // retired list, where the next bump will find it (it exchanges
    // the list into the closing generation's bag, and the free pass frees
    // the bag once no handle of that generation or an older one is live:
    // HANDLES AND GENERATIONS in the class overview); then, on a set
    // constructed with a cadence, run the automatic trigger if this push
    // took the shard's retired-node count across a multiple of the cadence
    // (step 3; THE TRIGGER in the class overview). The ONE entry point to
    // the retired lists, and the one site of the trigger.
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
    // only a holding bag until the free pass deals its nodes round-robin
    // over every shard's free list, so the shard a node is retired to has no
    // effect on where it is reused; what it does affect is contention, and
    // the caller's own shard's retired line is written concurrently only by
    // the threads that share that shard (and by the bump's exchange, once
    // per bump of any cause).
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
    // success (REQUIRED, DESIGNATED: the bump's exchange acquires it, and it
    // is sequenced after the caller's unlink CAS, so it heads the chain of
    // direction (I) of THE HANDLE INVARIANT -- unlink, push, exchange,
    // publish, join -- that keeps a handle of a later generation from ever
    // loading a node of this run; with a relaxed push the advancer could
    // also read a stale `retire_link` of the run below the head it took).
    // Failure is relaxed: the value that failed is only the next expected
    // value. Weak CAS: a retry loop anyway, and each failure is another
    // thread's completed push or a spurious failure (a plain retry), so the
    // loop is lock-free. The retire_link stores are relaxed: nothing reads
    // them before the exchange's acquire (the free pass and the accounting
    // sweep are the only readers). The counters are bumped before the CAS,
    // on the line the CAS is about to own; the value `retire_nodes` held
    // before this run is kept for step 3. (3) The cadence test, AFTER the
    // push, on one line with its call: the half-open interval (old, old +
    // count] holds a multiple of the cadence iff the bits above the
    // cadence's low bits differ between old and old + count, which the mask
    // (~(R - 1), or 0 for a cadence-off set, where the test is identically
    // false and the feature costs the use of the fetch_add's returned value,
    // one XOR, one AND on a `const` member and one untaken branch) tests; a
    // run that jumps several multiples fires once. After the push, not before, so that a trigger
    // which publishes bags THIS run too: push then exchange on one thread
    // is sequenced, and the exchange reads the push or a later value. A
    // trigger that does not publish leaves the run to another bump:
    // `nothing_retired` means a concurrent bump of any cause took the list
    // between this push and the flag acquisition (the head was not EMPTY
    // after the push), so the run is in that bump's bag; `ring_full`, like a
    // skipped attempt, leaves it on the retired list for the next bump of any
    // cause. No
    // ordering is needed for the decision: a relaxed fetch_add's returned
    // value is exact (the counter's modification order totals every earlier
    // count), and the counter orders nothing. The trigger reads nothing of
    // the operation's handle, which pins everything this thread still holds
    // (THE TRIGGER, claims (alpha)-(delta)).
    void push_retired(Shard& shard, Node* first, Node* last) {
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
        const size_t old_nodes = shard.retire_nodes.fetch_add(count, std::memory_order_relaxed);   // the count before this run: the cadence test's input (step 3)
        word_t old_head = shard.retired_head.load(std::memory_order_relaxed);
        do {   // step 2: the push
            last->retire_link.store(old_head, std::memory_order_relaxed);
        } while (!shard.retired_head.compare_exchange_weak(old_head, word_of(first), std::memory_order_release, std::memory_order_relaxed));
        if (CHS_RCU_UNLIKELY(((old_nodes ^ (old_nodes + count)) & auto_reclaim_mask_) != 0)) auto_reclaim();   // step 3: the cadence (identically false with the cadence off)
    } // push_retired()

    // THE AUTOMATIC TRIGGER (THE TRIGGER under HANDLES AND GENERATIONS in
    // the class overview): the tail of push_retired() on a cadenced set,
    // reached when the push took the shard's retired-node count across a
    // multiple of the cadence. It is reclaim() with no caller to report to:
    // the try-flag or nothing (a trigger that finds the flag held -- an
    // explicit call on any thread, another thread's trigger, or this
    // thread's own advance() whose walk is retiring -- counts itself in
    // `auto_skipped` and returns: no clock read, no hook, no `contended`;
    // the crossing run stays on the retired list for the next bump of any
    // cause); else, under the guard,
    // `auto_reclaims` +1 whatever the bump decides, then the bump, the hook
    // if it published and the final free pass (bump_and_free(), timed on
    // bump_ns and free_ns of the retiring thread). It runs on the retiring
    // thread, inside the operation that retired, under that operation's
    // handle, which it never reads: no handle is in scope here, so no
    // refresh, move or release of it is reachable from the trigger, and its
    // free pass stops at the operation's own pin (THE TRIGGER, claims (alpha)
    // and (beta)). It cannot nest: a retire push inside its hook (a nested
    // operation from the action) finds the flag held and is skipped. Never
    // the walk. noexcept end to end: the bump, the free pass and the hook
    // are noexcept and the guard's destructor throws nothing. No new memory
    // order: the flag's acquire exchange and the guard's release store are
    // advance()'s, and the decision that reached here is arithmetic on the
    // value a relaxed fetch_add returned. Never inlined: the retire path
    // reaches it once per cadence at most, and keeping its body out of
    // push_retired() keeps the push inlined into unlink_run() and the
    // operation paths free of the trigger's body: only the cadence test and
    // a call (an inlined trigger drags the bump and the free pass into
    // every unlink site). Not marked cold: under GCC a cold callee puts its
    // call site in the caller's cold partition, and values the caller keeps
    // live across that site (a caller's loop counters) can then be homed in
    // memory; with the call in the caller's own text most of them stay in
    // registers.
    [[gnu::noinline]] void auto_reclaim() noexcept {
        if (reclaiming_.exchange(true, std::memory_order_acquire)) {
            advance_counters_.auto_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        } // if the flag is held: the attempt is skipped
        const AdvanceGuard guard(*this);
        advance_counters_.auto_reclaims.fetch_add(1, std::memory_order_relaxed);
        AdvanceResult result{0, 0, 0, 0, 0, AdvanceResult::Advance::nothing_retired};
        bump_and_free(result);
    } // auto_reclaim()

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
    // state: every node that the sweep's quiescent rule (is_dead()) calls
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
    // diagnostic loss counters a lost CAS is charged to (none for the eraser's
    // self-unlink, whose losses are bounded at two per call and whose
    // residue is the tombstone itself; `advance` is the walk of advance(),
    // the fourth site, with the same one-shot shape as the writer walks).
    enum class UnlinkSite { walk, cleanup, self_unlink, advance };

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
    // from. Why the CAS cannot succeed on a stale view: no address the
    // caller's handle can observe is published twice during the caller's
    // session (ABA WITHOUT VERSION TAGS in the class overview), so the
    // expected value names the same node in the same chain for the whole
    // operation.
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
        if (site != UnlinkSite::self_unlink) {   // diagnostic counters: the loss, and whether a peer bypassed this very run
            Shard& shard = my_shard();
            std::atomic<size_t>& failures = site == UnlinkSite::cleanup ? shard.cleanup_cas_failures : site == UnlinkSite::advance ? shard.advance_cas_failures : shard.walk_cas_failures;
            std::atomic<size_t>& peer_losses = site == UnlinkSite::cleanup ? shard.cleanup_cas_peer_losses : site == UnlinkSite::advance ? shard.advance_cas_peer_losses : shard.walk_cas_peer_losses;
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
    // run stays for a later walk, cleanup or advance()). Dead here, besides
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

    // THE STEAL, alloc_node()'s slow path, taken only when the calling
    // thread's own free list is empty: pop a node holding `val` with link
    // `next` from any shard's free list; nullptr when the set-wide hint says
    // no free list has a node, or when a scan finds every list empty, and
    // the caller appends. Why it exists: the free pass deals freed nodes
    // round-robin over EVERY shard's free list, while a thread pops first
    // from its own shard's, so without the steal the nodes dealt to shards
    // no allocating thread maps to would never be reused, and under churn
    // with fewer allocating threads than shards the arena would grow
    // without bound.
    // Two parts. steal_node() is the CHEAP TEST, inlined into the append
    // path: one relaxed load of the set-wide word steal_hint_ and a branch;
    // when the word says "nothing free anywhere" (STEAL_HINT_EMPTY set) the
    // call returns nullptr at once, so an append in a set where nothing has
    // been freed since the last empty scan costs that load and nothing
    // more. Otherwise steal_scan() (not inlined) scans the lists. The test
    // reads a word on a line of its own that only the free pass and a
    // failed scan write, and adds no read-modify-write to the own-shard pop
    // or to the append. The hint is ONLY a hint: skipping a steal is always
    // safe (the caller appends: memory, not correctness), and no
    // correctness argument rests on the word or on its orders.
    // Why a flag with a stamp and not a count of free nodes: an exact count
    // would need a decrement per pop, a read-modify-write on one set-wide
    // line for every allocation of every thread (the contention the shards
    // exist to avoid); the steal needs only to know whether a scan could
    // find anything, and that changes only when the free pass pushes
    // (nodes enter a free list only there) or when the lists run dry.
    Node* steal_node(const T& val, word_t next) {
        const word_t hint = steal_hint_.load(std::memory_order_relaxed);   // relaxed: a hint (see steal_scan() for what the scan acquires)
        if (hint & STEAL_HINT_EMPTY) return nullptr;
        return steal_scan(val, next);
    } // steal_node()

    // THE SCAN of the steal: called by steal_node() when the hint it loaded
    // was not EMPTY. Pops a node holding `val` with link `next` from the
    // first shard's free list that has one, trying the calling thread's own
    // shard and then own + 1, own + 2, ... (modulo the shard count), each
    // once; if every list is empty, marks the hint EMPTY and returns
    // nullptr (also at once, if the hint reloaded here is EMPTY already).
    // THE HINT. steal_hint_ holds a STAMP (bits 63..1) and the EMPTY flag
    // (bit 0). The free pass, after pushing a freed bag's nodes, stores a
    // new stamp with EMPTY clear (RELEASE: sequenced after its pushes). A
    // scan begins by reloading the hint with ACQUIRE, so if that load read
    // a pass's store, every push of that pass and of every earlier one
    // happens-before the scan's loads of the heads, which therefore see
    // those nodes or their later pops. (A reload, not an acquire fence
    // after steal_node()'s relaxed load: the ordering the scan needs is the
    // same, the reload may read a later value and return early, and GCC's
    // ThreadSanitizer does not support fences.) A scan that finds every
    // list empty sets EMPTY with a CAS from the value it reloaded (relaxed:
    // nothing acquires through it); a free pass that stored a new stamp in
    // the meantime makes that CAS fail, and a store that comes after the
    // CAS in the word's modification order clears EMPTY again. So EMPTY stays set only while every node pushed by every
    // pass whose stamp store precedes it was seen popped. What the hint can
    // miss, and the bound: nodes a pass has pushed but whose stamp store
    // has not landed yet (a scan in that window may find them or not, and
    // may set EMPTY), until that store, at the end of freeing that bag,
    // clears EMPTY; and appends that load the hint before a store reaches
    // them, which is ordinary staleness. No node is left unstealable past
    // the stamp store of the bag that freed it. The acquire reload is on
    // the scan path only; the test in steal_node() stays one relaxed load.
    // The pop on each list is alloc_node()'s own-shard pop, unchanged, on
    // another head; the memory-model argument does not ask which shard the
    // popper maps to. Several threads already pop one shard's list (thread
    // numbers collide modulo the shard count), and a stealer is one more:
    // the head load and the CAS failure ACQUIRE synchronize with the free
    // pass's RELEASE push before the top's `link` is read and the value
    // assigned (THE PER-NODE REUSE CHAIN), and the losing-popper and ABA
    // argument (ABA WITHOUT VERSION TAGS) rests only on the popper's live
    // handle for the whole pop loop, which the stealer holds as the caller
    // of alloc_node() (insert() or split_bucket(), under the caller's
    // handle). The batching of node_count_ belongs to the list: the
    // stealer's CAS advances THAT list's pop counter, and if the counter it
    // replaced opened a batch the stealer adds the 256, after its CAS, as
    // an own-shard opener does (node_count_'s comment re-derives the
    // quiescent bound and the no-underflow argument per list). The value
    // bytes are unpoisoned by the stealing winner before the assignment,
    // as by an own-shard winner. A throwing assignment sends the node to
    // the stealer's OWN shard's limbo list (a limbo list is only a holding
    // place until the next bump; which shard holds a node does not matter
    // to the bag). The own shard is scanned again, first, because its
    // emptiness was read before the hint, so without the acquire above it.
    // Bounded: one pass over the shards, and on each list only the pop
    // loop's own retries, each after another thread's completed pop or
    // push (or a spurious failure of the weak CAS), and one CAS on the
    // hint, never retried: no lock, lock-free in the header's sense. Cost,
    // when the hint is not EMPTY: the acquire reload of steal_hint_, one
    // acquire load of each shard's free head (its FREE LINE), the shard
    // array's base and mask read once into locals (the atomic accesses in
    // the loop would otherwise make the compiler reload them per shard),
    // and, when every list is empty, the relaxed CAS that sets EMPTY: the
    // only read-modify-write on the line every appender reads, attempted
    // by every thread that finishes an empty scan on the same stamp (a
    // failed CAS still takes the line). The loop body is a copy of
    // alloc_node()'s own-shard pop with the source shard varied, not a
    // shared helper, and it is not inlined into alloc_node(), so that the
    // own-shard pop -- the hot path -- keeps its own code and gains only
    // what a call on the slow path costs a caller (the arguments kept for
    // after the call).
    [[gnu::noinline]] Node* steal_scan(const T& val, word_t next) {
        word_t hint = steal_hint_.load(std::memory_order_acquire);   // acquire: the free pass's pushes before the scan (THE HINT above)
        if (hint & STEAL_HINT_EMPTY) return nullptr;
        Shard* const shards = shards_.get();
        const size_t mask = arena_mask_;
        const size_t own = concurrent_hash_rcu_detail::thread_number() & mask;
        for (size_t i = 0; i <= mask; ++i) {   // one pass over the shards, own first
            Shard& from = shards[(own + i) & mask];
            word_t head = from.free_head.load(std::memory_order_acquire);   // acquire: REQUIRED, as alloc_node()'s head load
            while (free_addr_of(head) != FREE_EMPTY) {   // free-list pop on that shard's list
                Node* node = node_of(free_addr_of(head));
                word_t successor = node->link.load(std::memory_order_relaxed);
                const word_t expected = head;
                // (acquire, acquire): as alloc_node()'s own pop (failure
                // REQUIRED, success EXPECTED-EQUIVALENT to relaxed).
                if (from.free_head.compare_exchange_weak(head, free_word(successor, head + FREE_POP_ONE), std::memory_order_acquire, std::memory_order_acquire)) {
                    if ((head & FREE_BATCH_MASK) == 0) node_count_.fetch_add(256, std::memory_order_relaxed);   // this list's batch, opened by this pop
                    unpoison_node(node);
                    try {
                        node->value = val;
                    } catch (...) {
                        push_limbo(my_shard(), node, node);
                        throw;
                    }
                    node->link.store(next, std::memory_order_relaxed);
                    return node;
                } // if this thread popped the node
                if ((head & FREE_COUNT_MASK) == (expected & FREE_COUNT_MASK)) advance_counters_.pops_raced_push.fetch_add(1, std::memory_order_relaxed);   // see alloc_node()
            } // free-list pop on that shard's list
        } // one pass over the shards, own first
        // Every list was empty: mark the hint EMPTY unless a free pass stored
        // a new stamp since `hint` was reloaded. relaxed, and one attempt: a
        // hint (a failure means a pass freed something, and EMPTY must stay
        // clear).
        steal_hint_.compare_exchange_strong(hint, hint | STEAL_HINT_EMPTY, std::memory_order_relaxed, std::memory_order_relaxed);
        return nullptr;
    } // steal_scan()

    // Thread-safe node allocator: returns a node holding `val` with link
    // `next`, private to the caller until it publishes the node, at an
    // address that is stable for the life of the set. The node comes from
    // the calling thread's shard's free list when the list has a node, else
    // -- unless the steal's hint is EMPTY, in which case no list is tried --
    // from the first shard's free list, own first, that has one
    // (steal_node()), else it is appended to the calling thread's shard's
    // deque. Exceptions: see EXCEPTIONS in the
    // class overview.
    //
    // The pop is a lock-free CAS on the free head, run CONCURRENTLY with the
    // free pass's pushes (push_free(), one advancer at a time) and with the
    // other poppers of this list (its shard's threads, and threads of other
    // shards stealing: steal_node()). The orders are the last links of THE
    // PER-NODE REUSE CHAIN in the class overview: the free pass rewrote the
    // top node's `link` into the free list's next pointer, poisoned its
    // value bytes and pushed it with a RELEASE CAS, after it read the leaves
    // of every handle that could observe the node; this thread's ACQUIRE
    // must synchronize with that push before it reads the top's `link` and
    // before it assigns the value. Head load ACQUIRE (REQUIRED: relaxed lets
    // this thread read the top's `link` before the pusher's store of it,
    // and install a stale successor as the head; relaxed would do only if
    // no push could run concurrently with a pop, and the free pass's pushes
    // do). CAS failure ACQUIRE (REQUIRED, same reason: the refreshed
    // head may be a later push's value, and the next pass reads that top's
    // `link`). CAS success ACQUIRE (EXPECTED-EQUIVALENT to relaxed: the
    // value consumed is the one the acquire load or the acquire failure
    // returned, so the success order re-acquires an edge this thread has;
    // kept in the (acquire, acquire) pair form: a failure order stronger
    // than the success order buys nothing here and is a form some tools
    // silently weaken). The relaxed load of the
    // top's `link` is carried by the head load's acquire (the pusher stored
    // the link before its release CAS). Why a losing popper is harmless:
    // a popper that read the head (X, c) and then X's link may lose to
    // another popper of X, who then rewrites X's link (and value) for its
    // own use; the loser's `successor` is then garbage, but its CAS fails
    // regardless, because the head is no longer (X, c): X returns to a
    // free list only through a push by the free pass, which needs X freed
    // again, which needs every handle of X's bag's generation or older gone
    // -- this thread's own handle included (ABA WITHOUT VERSION TAGS in the
    // class overview: the counter bits play no part). A CAS that succeeds
    // therefore read X's link as the free pass left it, i.e. the true
    // successor. Weak CAS: we loop anyway. A failure whose counter bits are
    // the expected ones lost to a PUSH (a pop would have advanced them), or
    // was spurious, or -- the counter has nine bits -- lost to a multiple of
    // 512 pops between this thread's read and its CAS; it is counted
    // (pops_raced_push) as the witness that pops and pushes really meet, an
    // approximate one.
    //
    // The successful CAS also increments the counter packed in the head word
    // (it replaces the whole word), and only the winner's increment lands, so
    // the count of pops is exact; the winner whose pop opens a batch of 256
    // (the counter it replaced had its low eight bits clear, the shard's
    // first pop included) adds 256 to node_count_ AFTER the CAS: the same
    // one-sided batching as the append path (why the delay cannot let a
    // concurrent free take the count below zero: see node_count_).
    //
    // The popped node is private from the moment the CAS succeeds (no chain
    // reaches it, and no other popper can obtain it: see above), so its value
    // is copy-assigned and its link stored relaxed, exactly as insert()
    // repoints a node it has not published yet; the publishing release CAS
    // on a bucket head is what hands both to readers (channel 1). Under ASan
    // the value bytes are poisoned while the node is free (ASAN POISONING in
    // the class overview) and are unpoisoned here, before the assignment. If
    // the assignment throws, the catch below sends the node to limbo (see
    // EXCEPTIONS in the class overview), unpoisoned: the free pass poisons
    // it again when it frees the bag the limbo node lands in. Note the loser
    // of the pop race may still read this node's link (relaxed, harmless:
    // see above) while the value is being assigned; that is why a list's
    // next pointer lives in the atomic link and never in the T bytes.
    Node* alloc_node(const T& val, word_t next) {
        Shard& shard = my_shard();
        word_t head = shard.free_head.load(std::memory_order_acquire);
        while (free_addr_of(head) != FREE_EMPTY) {   // free-list pop
            Node* node = node_of(free_addr_of(head));
            word_t successor = node->link.load(std::memory_order_relaxed);
            const word_t expected = head;
            if (shard.free_head.compare_exchange_weak(head, free_word(successor, head + FREE_POP_ONE), std::memory_order_acquire, std::memory_order_acquire)) {
                if ((head & FREE_BATCH_MASK) == 0) node_count_.fetch_add(256, std::memory_order_relaxed);
                unpoison_node(node);
                try {
                    node->value = val;
                } catch (...) {
                    push_limbo(shard, node, node);
                    throw;
                }
                node->link.store(next, std::memory_order_relaxed);
                return node;
            } // if this thread popped the node
            if ((head & FREE_COUNT_MASK) == (expected & FREE_COUNT_MASK)) advance_counters_.pops_raced_push.fetch_add(1, std::memory_order_relaxed);   // lost to a push (or a spurious failure): the witness that pops meet pushes
        } // free-list pop
        if (Node* stolen = steal_node(val, next)) return stolen;   // the slow path: another shard's free list
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
        } // if the parent is not split yet

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
        my_shard().split_attempts.fetch_add(1, std::memory_order_relaxed);   // diagnostic counter

        try {
            while (addr_of(curr) != EMPTY) {   // traverse parent chain
                Node* node = node_of(curr);
                // A reference is safe: a published node's value is never written,
                // and its memory is not reused while the caller's handle lives,
                // whether or not the node is bypassed during this call (EXIT;
                // THE HANDLE INVARIANT in the class overview).
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
        // onto our shard's limbo list (one CAS, head to tail), for the next
        // bump to bag and a later free pass to free. The nodes we froze stay
        // frozen, which is correct: the winner copied them. Failure is
        // relaxed: the value that won is not used; our caller re-reads the
        // head with acquire.
        word_t expected = UNINITIALIZED;
        if (buckets_[j].compare_exchange_strong(expected, new_subchain_head, std::memory_order_release, std::memory_order_relaxed)) {
            // CAS succeeded: our subchain is now bucket j's authoritative head.
            my_shard().splits_published.fetch_add(1, std::memory_order_relaxed);   // diagnostic counter
            cleanup_parent(parent);   // step 4, after the publication (P9); H6 inside
        } else {
            // CAS failed: subchain abandoned (unpublished, memory waste until a later free pass frees it).
            if (new_subchain_head != EMPTY) push_limbo(my_shard(), node_of(new_subchain_head), new_subchain_tail);
        } // if another thread's split of j won
    } // split_bucket()

public:
    // Construction, through an Options aggregate (designated initializers:
    // `Set::Options{.auto_reclaim_every = 8192}`) or through the two-parameter
    // form below, which is the same with the cadence off. Both `explicit`.
    // The Options form has no default argument: with one, `Set s;` would
    // be ambiguous against the two-parameter form's defaults.
    // Options is a parameter type only: the set keeps none, and the
    // effective values are reachable through the diagnostics
    // (get_internal_table_size() on a fresh set, get_internal_arena_shards(),
    // get_internal_auto_reclaim_every()).
    //
    // initial_capacity: the number of buckets, rounded up to the nearest
    // power of two (the bucket mask math needs only a power of two; the
    // minimum of 4 is a convenience). A capacity above 2^63 is a caller's
    // bug: std::bit_ceil() of it has no representable result, so the
    // constructor calls std::abort(), in every build, before rounding. The
    // buckets are allocated here, eagerly, so a capacity beyond memory throws
    // bad_alloc from the constructor. Every initial bucket starts EMPTY (not
    // UNINITIALIZED) at seal level 0: the original buckets have no parent to
    // split from.
    // table_size_ is released LAST so that any thread which later acquires
    // it is guaranteed to see the fully initialized bucket array.
    //
    // arena_shards: number of arena shards, capped at MAX_ARENA_SHARDS (2^16)
    // and then rounded up to a power of two; 0 (the default) means the
    // hardware concurrency (1 if it is unknown), capped and rounded up the
    // same way. As many shards as threads that allocate concurrently makes
    // every shard lock uncontended (see alloc_node()); fewer shards trade
    // contention for memory (an untouched shard costs one small object, a
    // touched one at least a block of nodes).
    //
    // auto_reclaim_every: the automatic cadence (THE TRIGGER under HANDLES
    // AND GENERATIONS in the class overview), in nodes retired per arena
    // shard; 0 (the default) means OFF, decided before any rounding; a value
    // above 2^63 is clamped, silently, to 2^63 (SIZE_MAX is a valid way to
    // say "never"); the rest is rounded up to a power of two. With the
    // cadence off the set behaves exactly as one constructed by the
    // two-parameter form: nothing is freed unless advance() or reclaim() is
    // called. The cadence allocates nothing and cannot fail.
    //
    // The set is neither copyable nor movable (its deques and atomics are
    // neither). Its destruction is quiescent: every call on the set must
    // happen-before the destructor, which destroys every value the arena
    // holds (see the destructor for its preconditions on handles).
    struct Options {
        size_t initial_capacity = 4;     // buckets, as the first parameter of the two-parameter constructor
        size_t arena_shards = 0;         // arena shards, as its second; 0 = the hardware concurrency
        size_t auto_reclaim_every = 0;   // the cadence: nodes retired per arena shard between two automatic reclaims; 0 = off
    }; // struct Options
    explicit ConcurrentResizableHashSetRCU(const Options& options) : auto_reclaim_mask_(auto_reclaim_mask(options.auto_reclaim_every)) {
        size_t initial_capacity = options.initial_capacity;
        size_t arena_shards = options.arena_shards;
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
        // The generation ring (HANDLES AND GENERATIONS in the class
        // overview): every block gets its back-pointer and its bag slots, two
        // words per shard, all EMPTY; slot 0 is CURRENT with generation 0,
        // every other slot EMPTY, `oldest_` = 0; the bump's scratch array of
        // one word per shard (limbo_loaded_). These are the allocations of
        // the ring, and they can throw bad_alloc here, before any handle or
        // operation exists. Before the table_size_ release below, which
        // stays the constructor's last store.
        for (GenerationBlock& block : ring_) {
            block.set = this;
            block.bag = std::make_unique<word_t[]>(2*arena_shards);
            std::fill_n(block.bag.get(), 2*arena_shards, EMPTY);
        } // loop over the ring
        ring_[0].state = GenerationState::CURRENT;
        limbo_loaded_ = std::make_unique<word_t[]>(arena_shards);
        // The per-shard count rows, every count 0 (value-initialized; an
        // over-aligned array new). Before the release below, like the ring.
        rows_ = std::make_unique<CountRow[]>(arena_shards);
        table_size_.store(initial_capacity, std::memory_order_release);
    } // ConcurrentResizableHashSetRCU(const Options&)
    // The two-parameter form: the same parameters and defaults as the
    // Options form, the cadence off. It delegates, so that the one
    // constructor body and the one `const` mask initializer exist; `explicit`
    // because a bare size is not a set (copy-initialization from a size or a
    // braced list is rejected).
    explicit ConcurrentResizableHashSetRCU(size_t initial_capacity = 4, size_t arena_shards = 0)
        : ConcurrentResizableHashSetRCU(Options{initial_capacity, arena_shards, 0}) {}

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
    // the class overview). For the same reason it is not const: the split
    // writes the set's internals (a seal, freezes, a publish, allocations,
    // the cleanup pass's unlinks and, on a set constructed with a cadence,
    // the bump and free pass of a trigger that a cleanup's retire push may
    // run: THE TRIGGER), though never its membership.
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
    // `h`: the caller's handle (HANDLES AND GENERATIONS in the class
    // overview): live, of this set, and not being refreshed, moved or
    // destroyed meanwhile; debug builds check the first two and count the
    // call in flight on the handle, release builds touch nothing of it.
    bool contains(const handle& h, const T& key) {
        [[maybe_unused]] const HandleUse use(*this, h);
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
    // Postcondition of an UNCONTENDED call that returns normally (every other
    // call on the set happens-before or happens-after it; POSTCONDITIONS in
    // the class overview): no dead run stays reachable behind the bucket head
    // or a live node the walk passed. Under contention the walk's unlinks are
    // best effort.
    //
    // The decision is the publishing CAS on the bucket head, and the geometry is
    // validated by that same CAS: its expected value includes the head's seal
    // level, which was checked against the table size this attempt works with.
    // See "STALE GEOMETRY" in the class overview and step 1 of split_bucket().
    //
    // Exceptions: see EXCEPTIONS in the class overview.
    // `h`: the caller's handle, as for contains(); the node this call pops or
    // appends and every word it CASes are used under it (ABA WITHOUT VERSION
    // TAGS in the class overview).
    bool insert(const handle& h, const T& key) {
        [[maybe_unused]] const HandleUse use(*this, h);
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
                    // onto our shard's limbo list for a bump to bag and a free pass to free.
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
                    // (current_ts == ts). NOTE: node_count_ is arena occupancy minus
                    // what the free pass has freed -- it counts
                    // tombstones, stale split copies, lost subchains and orphans
                    // until a bump bags them and a free pass frees them -- so this is an
                    // arena-consumption trigger, not a live-load-factor trigger: a
                    // delete-heavy workload without advance() or reclaim() calls
                    // (and without a cadence) grows the table
                    // although the live key count does not. The count never dips
                    // below zero, concurrent frees included (see node_count_), so
                    // the unsigned compare is safe. New buckets are
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
    // tombstone: at most one erase() returns true per present->absent
    // transition of the key, across any number of concurrent resizes --
    // exactly one, unless the call that set the tombstone throws after it
    // (EXCEPTIONS, P5). Returns false only if the key was absent at some
    // instant during the call. After a successful mark the call makes a
    // bounded effort to take its tombstone out of the chain at once
    // (self_unlink(): one CAS on the pred word tracked during the walk, one
    // restart from the head, then give up); the result is unaffected unless
    // Hash{} throws there (EXCEPTIONS, P5), and a tombstone that stays is a
    // STRAGGLER (WHY COPY in the class overview). The walk itself is a writer
    // walk (step_over()): every dead run it passes behind a live word gets one
    // unlink attempt. Postconditions of an UNCONTENDED call that returns
    // normally (every other call on the set happens-before or happens-after
    // it; POSTCONDITIONS in the class overview): no dead run stays reachable
    // behind the bucket head or a live node the walk passed, and the
    // tombstone is unreachable unless a tagged, not-dead node lies between it
    // and the last live word before it.
    // Cost of an uncontended successful erase() whose bucket is published
    // and whose walk meets no tagged node before the key: three CASes (the
    // mark, the self-unlink's bypass of the tombstone, and the retire push)
    // and the two relaxed fetch_adds of push_retired() on the shard's retire
    // counters, which are atomic read-modify-writes as the CASes are. A dead
    // run that the walk bypasses before the key adds one unlink CAS, one
    // retire push and the same two increments.
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
    // `h`: the caller's handle, as for contains(); the mark CAS and the
    // self-unlink's CASes are used under it (ABA WITHOUT VERSION TAGS in the
    // class overview).
    bool erase(const handle& h, const T& key) requires AllowDelete {
        [[maybe_unused]] const HandleUse use(*this, h);
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
    // The free lists' PUSH side, and the dead-node rule of the diagnostic
    // sweep. Everything else of reclamation -- the bump, the free pass that
    // calls push_free(), the walk: advance()'s steps, the first two shared
    // with reclaim() and the trigger -- is at the end of the class (HANDLES
    // AND GENERATIONS in the class overview); the trigger's entry,
    // auto_reclaim(), is beside push_retired().
    // ------------------------------------------------------------------------

private:
    // Push `node` onto `shard`'s free list, keeping the head's counter bits.
    // Called by the free pass alone (one flag holder at a time: advance(),
    // reclaim() or an automatic trigger), for a node that is on no chain and on no list and
    // that no live handle can observe (THE HANDLE INVARIANT in the class
    // overview): a bagged retired or limbo node whose bag's generation, and
    // every older one, has no live handle. Runs CONCURRENTLY with the pops of
    // alloc_node() on the same head; the CAS loop is lock-free (a failure is
    // a completed pop, or a spurious failure of the weak CAS, a plain retry;
    // no other pusher exists). Steps: (1) poison the node's value bytes under
    // ASan (ASAN POISONING in the class overview; a no-op otherwise), before
    // the loop, so the poison is published by the same release as the link;
    // (2) rewrite `link` into the free list's next pointer, the bare address
    // of the old top (the ONE place a retired node's `link` is rewritten:
    // until here it kept the chain successor the node had when it died, so
    // an operation that reached the node before its unlink could exit
    // through it, and the handle invariant says no such operation remains);
    // (3) the CAS. RELEASE on success (REQUIRED, DESIGNATED: the popper's
    // acquire of the head must see the link of step 2 and the poison before
    // it reads the one and assigns over the other; this is the free push's
    // link of THE PER-NODE REUSE CHAIN, and a relaxed push lets the popper
    // read a stale `link` and install garbage as the new head). Failure
    // relaxed (EXPECTED-EQUIVALENT to acquire: the failed value, a popper's,
    // is only the next expected value and the next top's address for step
    // 2; nothing the popper wrote is read). The head load is relaxed for the
    // same reason. The counter bits of the head are copied into the new
    // word, never cleared: the pop batching (node_count_) counts on them
    // only ever advancing. `retire_link` is left as it is: stale, never read
    // again until the node is retired anew. Each lost CAS is counted
    // (free_push_cas_failures), the witness that pushes meet pops.
    void push_free(Shard& shard, Node* node) {
        poison_node(node);
        word_t head = shard.free_head.load(std::memory_order_relaxed);
        while (true) {   // the push loop: every failure is a completed pop of this list, or spurious
            node->link.store(free_addr_of(head), std::memory_order_relaxed);   // published by the release CAS below
            if (shard.free_head.compare_exchange_weak(head, free_word(word_of(node), head), std::memory_order_release, std::memory_order_relaxed)) return;
            advance_counters_.free_push_cas_failures.fetch_add(1, std::memory_order_relaxed);
        } // the push loop
    } // push_free()

    // The dead-node rule of the diagnostic accounting sweep (its C2 check
    // and its `reachable_dead` count: get_internal_accounting()), for `node`
    // with link value `link` in the chain of published bucket `j`, at table
    // size `ts`, read at a QUIESCENT point. Dead means: no operation, present
    // or future, can need this node, so it may leave the chain. A node is
    // dead iff
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
    // Relation to the concurrent form (is_dead_now()), which every unlinking
    // site uses, advance()'s walk included: the same rule, minus the tag
    // test. Concurrent-dead implies quiescent-dead; a reachable
    // quiescent-dead node is tagged (C2 of the sweep), so at a quiescent
    // point the two agree on every reachable node. Nothing but the sweep
    // calls this: no code path of this class unlinks on the quiescent rule.
    bool is_dead(const Node* node, word_t link, size_t j, size_t ts) const {
        if (link & MARK_BIT) return true;
        const size_t child = next_child(Hash{}(node->value), j);
        return child != j && child < ts && addr_of(buckets_[child].load(std::memory_order_relaxed)) != UNINITIALIZED;
    } // is_dead()

public:

    // DIAGNOSTICS. The get_internal_* accessors below, and the two structs
    // they return, show the arena and the counters to the tests and the
    // benchmark. They come in three classes:
    //   - RACE-FREE, approximate while operations run:
    //     get_internal_node_count(), get_internal_arena_shards(),
    //     get_internal_auto_reclaim_every(), get_internal_table_size(),
    //     get_internal_counters(), get_internal_retire_nodes_per_shard() and
    //     get_internal_slot_counts() read only atomics and values fixed at
    //     construction, and may be called concurrently with any operation,
    //     explicit call or trigger; each value they read is one its source had,
    //     but a sum over the shards is not one instant's total. At a quiescent
    //     point they are exact; get_internal_arena_shards() and
    //     get_internal_auto_reclaim_every(), values fixed at construction, are
    //     exact always.
    //   - QUIESCENT ONLY: get_internal_accounting(), the free, limbo,
    //     retired and bagged counts taken from it, and
    //     get_internal_bucket_published(). QUIESCENCE here means: no
    //     operation, no join attempt (new_handle(), refresh()'s slow path)
    //     and no advance() or reclaim() is in progress on any thread while the function
    //     runs, every earlier call happens-before it and it happens-before
    //     every later one (the caller joins or barriers the worker threads
    //     before and starts or releases them after). Live IDLE handles are
    //     allowed, and counted (`handle_refs`): an idle handle touches
    //     nothing. Under that precondition every atomic access of the sweep
    //     is relaxed: the values it reads were published to it by the
    //     caller's synchronization. Why nothing less will do: the sweep
    //     reads node values (Hash{} through is_dead()) after relaxed loads
    //     only, so a node that an insert publishes meanwhile, constructed by
    //     an append or copy-assigned after a pop, would be read with no
    //     happens-before, a data race; it reads the ring's plain fields
    //     (`state`, `gen`, the bag slots, `oldest_`), which the advancer
    //     writes under the flag (asserted clear at the sweep's start); and
    //     get_internal_bucket_published(j) indexes buckets_, which is safe
    //     concurrently only for a j below a table size the calling thread
    //     has acquired (channels 2 and 3), while the relaxed load of
    //     get_internal_table_size() does not provide one, and the sweep
    //     itself indexes buckets_ up to a relaxed-loaded table_size_, the
    //     same hazard.
    //   - RING VIEWS, no flag holder in progress -- no advance(), no
    //     reclaim() and, on a set constructed with a cadence, no OPERATION
    //     (an operation's retire push may run the automatic trigger, which
    //     holds the flag): on a cadence-off set a weaker precondition than
    //     QUIESCENCE (operations, join attempts and live handles may run),
    //     on a cadenced set operation quiescence (join attempts and live
    //     idle handles may still run):
    //     get_internal_generation(), get_internal_ring_state() and
    //     get_internal_bag_count() read the ring's plain fields and walk the
    //     bags, which only the flag holder writes (asserted: the flag
    //     clear), and the bagged nodes' `retire_link` and `link`, which no
    //     operation writes. get_internal_generation() reads `current_` with
    //     acquire, which reads the publish and so orders the holder's write
    //     of `gen` before the read; it needs nothing more. The other two
    //     also read `state`, `oldest_` and the bag slots, which a holder may
    //     write without publishing (a `nothing_retired` or `ring_full`
    //     outcome that frees bags), so they need every earlier holder to
    //     happen-before the call (the caller made it, or joined the thread
    //     that did). get_internal_handle_generation(h) and
    //     get_internal_handle_shard(h) need only a live h.
    //   THE LIVE-HANDLE COUNTS in these views are per shard underneath (THE
    //   RING): `InternalRingSlot::refs` and `InternalAccounting::handle_refs`
    //   are SUMS of relaxed loads of the S per-shard counts, and the join
    //   counters of InternalCounters are sums of the shards' join counters,
    //   so with joins or leaves running a sum is not one instant's value (as
    //   for every shard sum above); at a quiescent point they are exact.
    //   get_internal_slot_counts(i) gives the S counts of ring slot i
    //   themselves, keyed by SLOT INDEX because it reads only atomics (no
    //   `gen`, no `state`); the slot of a CURRENT or SEALED generation g is g
    //   % kGenerations (the bump makes the slot after the current one CURRENT
    //   with `gen + 1`, from slot 0 with generation 0), so a test addresses a
    //   handle's counts by get_internal_handle_generation(h) % kGenerations
    //   and the count it recorded by get_internal_handle_shard(h).
    //
    // Diagnostic accessor: total nodes ever appended to the arena (reachable,
    // free, limbo, retired and bagged alike): the sum of the shards' sizes,
    // exact at a quiescent point.
    size_t get_internal_node_count() const {
        size_t n = 0;
        for (size_t i = 0; i <= arena_mask_; ++i) n += shards_[i].nodes.size();
        return n;
    } // get_internal_node_count()
    // Diagnostic accessor: the number of arena shards (a power of two), as
    // the constructor fixed it.
    size_t get_internal_arena_shards() const { return arena_mask_ + 1; }
    // Diagnostic accessor, race-free: the effective cadence (nodes retired
    // per arena shard between two automatic reclaims) as the constructor
    // fixed it: 0 for a cadence-off set (both constructors with no cadence),
    // else the requested value clamped to 2^63 and rounded up to a power of
    // two (the constructor's rules). Decoded from the mask, which stores
    // ~(R - 1) for a cadence R and 0 for off.
    size_t get_internal_auto_reclaim_every() const noexcept { return auto_reclaim_mask_ == 0 ? 0 : ~auto_reclaim_mask_ + 1; }
    // Diagnostic accessor, race-free: every shard's `retire_nodes` (the nodes
    // retired onto that shard's retired list over the set's life: the
    // cadence's input), one relaxed load per shard, in shard order (index =
    // the thread number masked by the shard count); exact at a quiescent
    // point, a snapshot otherwise; its sum is get_internal_counters()'s
    // `retire_nodes`.
    std::vector<size_t> get_internal_retire_nodes_per_shard() const {
        std::vector<size_t> per_shard;
        per_shard.reserve(arena_mask_ + 1);
        for (size_t s = 0; s <= arena_mask_; ++s) per_shard.push_back(shards_[s].retire_nodes.load(std::memory_order_relaxed));
        return per_shard;
    } // get_internal_retire_nodes_per_shard()
    // Diagnostic accessors, quiescent only: the number of nodes on all free
    // lists / on all limbo lists, as the accounting sweep below counts them.
    // A corrupted (cyclic) list does not hang: the sweep stops at the first
    // address it has already seen -- but the count returned here is then
    // merely the nodes walked before the stop (a cycle A->B->C->A counts 3);
    // the corruption itself is visible only in get_internal_accounting()'s
    // `consistent`, which tests assert alongside these counts.
    size_t get_internal_free_count() const { return get_internal_accounting().free_nodes; }
    size_t get_internal_limbo_count() const { return get_internal_accounting().limbo; }
    // Diagnostic accessor, quiescent only: the number of nodes on all retired
    // lists, as the accounting sweep counts them (the same caveats as the two
    // above).
    size_t get_internal_retired_count() const { return get_internal_accounting().retired; }
    // Diagnostic accessor: the table size (number of buckets), for tests that
    // construct an exact geometry and must know whether a doubling happened.
    // A relaxed load: race-free, and concurrently a size the table has had,
    // which orders nothing.
    size_t get_internal_table_size() const { return table_size_.load(std::memory_order_relaxed); }
    // Diagnostic accessor, quiescent only (see DIAGNOSTICS above): whether
    // bucket j (j below the current table size) is published, i.e. its lazy
    // split has run; a test that walks the buckets at a quiescent point skips
    // the pending ones with it.
    bool get_internal_bucket_published(size_t j) const { return addr_of(buckets_[j].load(std::memory_order_relaxed)) != UNINITIALIZED; }

    // Diagnostic counters: the sums over the shards of Shard's counters (see
    // there for what each counts). Race-free (relaxed loads); exact at a
    // quiescent point, a snapshot of monotone counters otherwise. Never reset.
    struct InternalCounters {
        size_t retire_runs;              // push_retired() calls (one per unlinked run)
        size_t retire_nodes;             // nodes those runs held in total
        size_t cleanup_cas_failures;     // lost one-shot CASes of a split winner's cleanup pass, all kinds
        size_t cleanup_cas_peer_losses;  // of those, losses to a peer that bypassed the same run (nothing left behind)
        size_t walk_cas_failures;        // lost one-shot CASes of insert()'s and erase()'s walks, all kinds
        size_t walk_cas_peer_losses;     // of those, peer losses
        size_t split_attempts;           // split_bucket() calls that reached the walk of the sealed parent, recursive parent splits included
        size_t splits_published;         // of those, the ones whose publishing CAS won
        // The generation protocol's counters (AdvanceCounters, at the end of
        // the class; the two walk-site counters of Shard; and `joins` and
        // `join_retries`, the sums over the shards of the per-shard join
        // counters in the count rows, CountRow).
        size_t advance_cas_failures;     // lost one-shot CASes of advance()'s walk, all kinds
        size_t advance_cas_peer_losses;  // of those, peer losses
        size_t joins;                    // successful join attempts (new_handle(), refresh()'s slow path, the walk's join)
        size_t join_retries;             // join attempts whose re-verify failed (a bump of any cause published between the load and the re-verify)
        size_t bumps;                    // publishes of a new generation (`advanced`), by advance(), reclaim() or a proceeding automatic trigger alike
        size_t ring_full;                // `ring_full` outcomes of advance(), reclaim() and proceeding triggers
        size_t contended;                // advance() and reclaim() calls refused with `contended` (never a skipped trigger: that is `auto_skipped`)
        size_t bags_freed;               // bags the free pass freed, whoever ran it
        size_t nodes_freed;              // nodes the free pass moved to the free lists, whoever ran it
        size_t free_push_cas_failures;   // free-list pushes of the free pass that lost a CAS (to a concurrent pop; no other pusher exists) or failed spuriously
        size_t pops_raced_push;          // free-list pops whose CAS lost to a concurrent PUSH (the failure value's counter bits equal the expected ones: a pop would have advanced them) or failed spuriously
        size_t limbo_pushes;             // push_limbo() calls (one per limbo subchain)
        // The automatic cadence's counters (0 for the life of a cadence-off
        // set): one attempt per retire push that took its shard's
        // retired-node count across a multiple of the cadence, counted in
        // exactly one of the two.
        size_t auto_reclaims;            // triggers that acquired the flag, whatever their bump then decided (`advanced`, `nothing_retired` or `ring_full`)
        size_t auto_skipped;             // triggers that found the flag held and did nothing
    }; // struct InternalCounters
    InternalCounters get_internal_counters() const {
        InternalCounters c{};
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
            c.advance_cas_failures += shard.advance_cas_failures.load(std::memory_order_relaxed);
            c.advance_cas_peer_losses += shard.advance_cas_peer_losses.load(std::memory_order_relaxed);
        } // loop over the shards
        for (size_t s = 0; s <= arena_mask_; ++s) {   // the join counters, per shard in the count rows
            c.joins += rows_[s].joins.load(std::memory_order_relaxed);
            c.join_retries += rows_[s].join_retries.load(std::memory_order_relaxed);
        } // the join counters
        c.bumps = advance_counters_.bumps.load(std::memory_order_relaxed);
        c.ring_full = advance_counters_.ring_full.load(std::memory_order_relaxed);
        c.contended = advance_counters_.contended.load(std::memory_order_relaxed);
        c.bags_freed = advance_counters_.bags_freed.load(std::memory_order_relaxed);
        c.nodes_freed = advance_counters_.nodes_freed.load(std::memory_order_relaxed);
        c.free_push_cas_failures = advance_counters_.free_push_cas_failures.load(std::memory_order_relaxed);
        c.pops_raced_push = advance_counters_.pops_raced_push.load(std::memory_order_relaxed);
        c.limbo_pushes = advance_counters_.limbo_pushes.load(std::memory_order_relaxed);
        c.auto_reclaims = advance_counters_.auto_reclaims.load(std::memory_order_relaxed);
        c.auto_skipped = advance_counters_.auto_skipped.load(std::memory_order_relaxed);
        return c;
    } // get_internal_counters()

    // Diagnostic accounting sweep, quiescent only (see DIAGNOSTICS above):
    // where every arena slot is. `consistent` is true iff
    //   C1  every slot of every shard is reachable from a published bucket
    //       head, on a free list, on a limbo list, on a retired list, or in
    //       the bag of a SEALED generation, exactly one of the five and
    //       exactly once, and no list, bag or chain contains an address that
    //       is not a slot;
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
    //   C4  the generation ring is well formed: `current_` points at the one
    //       CURRENT slot; behind it, from `oldest_` up to it, a contiguous
    //       run of SEALED slots whose generation numbers decrease by one per
    //       slot going back; every other slot EMPTY; the bag slots of every
    //       slot that is not SEALED all EMPTY (a bag is filled at the bump
    //       and emptied, slot by slot, by the free pass); no per-shard count
    //       of any slot negative and every count of an EMPTY slot zero (a
    //       stale join increment is undone before the joiner retries, so at
    //       quiescence only live handles count); the advancer flag clear;
    //   C5  the resize hint is one-sided: `node_count_ >= slots - free`
    //       (the comment on node_count_ derives it);
    //   C6  under ASan, every free-list node is poisoned exactly as ASAN
    //       POISONING in the class overview says: every byte of its value
    //       extent, and neither its `link` nor its `retire_link` (a per-byte
    //       test, node_is_poisoned()); without ASan the conjunct is true.
    // The counts are what was found (a walk stops at the first address that
    // is unknown or already seen, so a cyclic list cannot hang it, and then
    // `consistent` is false; a node retired twice makes exactly such a cycle
    // in a retired list). `reachable_dead` is the number of reachable nodes
    // that is_dead() calls dead at this quiescent point: the dead nodes still
    // in their chains, which advance()'s walk would try to unlink, except
    // those behind a FROZEN node whose child is unpublished (no walk can);
    // it calls Hash{} on every reachable node that is not MARKED. Expected
    // consistent at every quiescent point, including after an exception on
    // any operation or on advance() (see EXCEPTIONS in the class overview).
    struct InternalAccounting {
        size_t slots;           // sum of the shards' sizes (get_internal_node_count())
        size_t reachable;       // nodes reachable from published heads, live and dead alike
        size_t reachable_dead;  // of those, the ones is_dead() calls dead (quiescent rule)
        size_t free_nodes;      // nodes on the free lists
        size_t limbo;           // nodes on the limbo lists
        size_t retired;         // nodes on the retired lists
        bool consistent;        // C1 (reachable + free_nodes + limbo + retired + bagged == slots, no slot twice), C2, C3, C4, C5 and C6
        // The generation protocol's view (HANDLES AND GENERATIONS in the
        // class overview). The bag sublists are walked in the same
        // exactly-once visit as the four populations above (a retired
        // sublist through `retire_link`, a limbo sublist through `link`),
        // and C3 holds for every bagged retired node as for a retired one.
        size_t bagged;                      // nodes in the bags of every SEALED generation, retired and limbo sublists alike
        long handle_refs;                   // the sum of every generation's per-shard counts: the live handles, plus any join attempt's transient +1 (none under quiescence)
        std::vector<size_t> free_per_shard; // `free_nodes` split by shard, in shard order
    }; // struct InternalAccounting
    InternalAccounting get_internal_accounting() const {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "accounting sweep during advance(), reclaim() or an automatic trigger");
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
        InternalAccounting acc{slots.size(), 0, 0, 0, 0, 0, true, 0, 0, std::vector<size_t>(arena_mask_ + 1, 0)};
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
        // Walks a limbo-shaped list (linked through `link`) from `head`,
        // visiting each node, and returns the nodes walked.
        auto walk_limbo = [&](word_t head) -> size_t {
            size_t n = 0;
            for (word_t curr = head; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++n) curr = node_of(curr)->link.load(std::memory_order_relaxed);
            return n;
        };
        // Walks a retired-shaped list (linked through `retire_link`) from
        // `head`, visiting each node and applying C3 to it; returns the
        // nodes walked.
        auto walk_retired = [&](word_t head) -> size_t {
            size_t n = 0;
            for (word_t curr = head; addr_of(curr) != EMPTY && visit(addr_of(curr)); ++n) {
                const Node* node = node_of(curr);
                const word_t link = node->link.load(std::memory_order_relaxed);
                if ((link & (MARK_BIT | FROZEN_BIT)) == 0 || !is_empty_or_slot(addr_of(link))) acc.consistent = false;   // C3
                curr = node->retire_link.load(std::memory_order_relaxed);
            } // walk the retired-shaped list
            return n;
        };
        for (size_t s = 0; s <= arena_mask_; ++s) {   // free, limbo and retired lists
            word_t curr = free_addr_of(shards_[s].free_head.load(std::memory_order_relaxed));
            for (; curr != FREE_EMPTY && visit(curr); ++acc.free_per_shard[s]) {   // walk the free list
                if (!node_is_poisoned(node_of(curr))) acc.consistent = false;   // C6
                curr = node_of(curr)->link.load(std::memory_order_relaxed);
            } // walk the free list
            acc.free_nodes += acc.free_per_shard[s];
            acc.limbo += walk_limbo(shards_[s].limbo_head.load(std::memory_order_relaxed));
            acc.retired += walk_retired(shards_[s].retired_head.load(std::memory_order_relaxed));
        } // loop over the shards
        // C4, the ring: the CURRENT slot at `current_`, the SEALED run from
        // `oldest_` up to it (generation numbers counting down from the
        // current one), EMPTY elsewhere, no bag outside a SEALED slot.
        const size_t current = slot_index(current_.load(std::memory_order_relaxed));
        const size_t sealed = (current + kGenerations - oldest_) % kGenerations;
        if (reclaiming_.load(std::memory_order_relaxed)) acc.consistent = false;
        for (size_t i = 0; i < kGenerations; ++i) {   // every slot's state and counts, and the bags of the SEALED ones
            const GenerationBlock& block = ring_[i];
            const SlotCounts counts = slot_counts(i);   // the slot's S per-shard counts: every one >= 0, and every one 0 on an EMPTY slot (min >= 0 and sum 0)
            acc.handle_refs += counts.sum;
            const size_t behind = (current + kGenerations - i) % kGenerations;   // 0 for the CURRENT slot, 1 for the slot before it, ...
            const GenerationState expected = behind == 0 ? GenerationState::CURRENT : behind <= sealed ? GenerationState::SEALED : GenerationState::EMPTY;
            if (block.state != expected || counts.min < 0 || (expected == GenerationState::EMPTY && counts.sum != 0)) acc.consistent = false;
            if (expected == GenerationState::SEALED && block.gen + behind != ring_[current].gen) acc.consistent = false;
            if (block.state != GenerationState::SEALED) {
                for (size_t s = 0; s <= arena_mask_; ++s) {
                    if (bag_retired(block, s) != EMPTY || bag_limbo(block, s) != EMPTY) acc.consistent = false;   // a bag outside a SEALED slot
                } // loop over the shards' slots
                continue;
            } // if not SEALED: no bag
            for (size_t s = 0; s <= arena_mask_; ++s) {
                acc.bagged += walk_retired(bag_retired(block, s));
                acc.bagged += walk_limbo(bag_limbo(block, s));
            } // loop over the shards' slots of this bag
        } // loop over the ring
        if (acc.reachable + acc.free_nodes + acc.limbo + acc.retired + acc.bagged != acc.slots) acc.consistent = false;   // C1's sum
        if (acc.slots < acc.free_nodes || node_count_.load(std::memory_order_relaxed) < acc.slots - acc.free_nodes) acc.consistent = false;   // C5
        assert(acc.slots < acc.free_nodes || node_count_.load(std::memory_order_relaxed) >= acc.slots - acc.free_nodes);   // C5, loudly: the resize hint under-counts the occupied slots
        return acc;
    } // get_internal_accounting()

    // ------------------------------------------------------------------------
    // HANDLES AND GENERATIONS (the class overview section of that name).
    // Every member from here to the end of the class belongs to the
    // generation protocol; they are appended after the members above so
    // that no member above moves (the data members at the very end
    // follow node_count_ in the layout): the generation block and the
    // handle, the join and the leave, the flag's guard, the poison helpers,
    // the destructor, new_handle(), advance() and reclaim() with the bump,
    // the free pass and the walk, the protocol's diagnostics, and the
    // counters, the ring, the current pointer and the flag.
    // ------------------------------------------------------------------------
public:
    // The state of a generation slot (THE RING in the class overview).
    // Public only for the diagnostic ring-state view.
    enum class GenerationState { EMPTY, CURRENT, SEALED };

private:
    // One generation: a status block in the fixed ring (never deallocated or
    // moved). 128 bytes per block (a pair of 64-byte lines on x86, so the
    // adjacent-line prefetcher does not couple neighbours): adjacent blocks
    // would otherwise share lines. The live-handle counts are not in the block
    // but in the per-shard count rows (CountRow, `rows_`); the alignment is
    // REQUIRED all the same: without it a block is four words, `ring_` a
    // quarter of its size, and every set member declared after the ring moves,
    // among them steal_hint_ and auto_reclaim_mask_, which the operations read
    // (an operation's object code would change); and `set`, which refresh()
    // loads on every call, would share a line with neighbouring blocks'
    // advancer-written fields. `state`, `gen` and the bag slots are PLAIN
    // fields written only by the holder of `reclaiming_` (and the constructor).
    // They are read by the flag holder, whose synchronization with earlier
    // holders is the flag's acquire/release; by the destructor and the
    // quiescent accounting sweep, under their quiescence; and, for `gen` alone,
    // through the publish that made the block current: by
    // get_internal_handle_generation(), safe only AFTER the handle's successful
    // re-verify (that publish then happens-before the read, and the block
    // cannot be recycled while the handle's +1 is on one of its counts), and by
    // get_internal_generation(), whose acquire load of `current_` reads that
    // publish. Joiners read only `current_` and their count (never a field of
    // the block: the count's address is computed from the block's ring index),
    // so no join path reads `gen`.
    struct alignas(128) GenerationBlock {
        GenerationState state = GenerationState::EMPTY;      // EMPTY / CURRENT / SEALED (owned by the advancer)
        uint64_t gen = 0;                                    // the generation number this slot currently represents (owned by the advancer)
        std::unique_ptr<word_t[]> bag;                       // the BAG: 2*shards words, [2s] the retired sublist head and [2s + 1] the limbo sublist head exchanged from shard s at the bump; EMPTY unless SEALED (owned by the advancer; allocated by the constructor)
        const ConcurrentResizableHashSetRCU* set = nullptr;  // the owning set, set once in the constructor: how a handle finds its set for refresh() (and, in debug builds, for the leave's membership check)
    }; // struct GenerationBlock
    // One arena shard's row of the live-handle counts (THE RING in the class
    // overview): the shard's count of EVERY ring slot, then the shard's join
    // counters. The rows (`rows_`, one per shard, allocated by the constructor,
    // never moved or freed before the set) hold the counts per shard rather
    // than one count per block, so that a join and a leave write only lines of
    // the joining thread's shard and the operations of threads on different
    // shards never write a common line to take or drop a handle. Layout: the
    // counts come first, 16 per 128-byte pair of 64-byte lines (so a multi-bag
    // free pass reads a shard's pair once per 16 consecutive bags), then the
    // two join counters; when kGenerations is a multiple of 16 (the default 64
    // included: four pairs of counts) the join counters start a pair of their
    // own, otherwise they share the last pair with the highest slots' counts.
    // Why a pair of their own is wanted: a join writes its count (the CURRENT
    // slot's line) and `joins`, while the free pass reads the OLDEST slot's
    // count, which under `ring_full` sits just after the current slot and so
    // usually on the same line; a join counter on that line would add a writer
    // the pass does not need. The row's stride is its contents rounded up to
    // 128 bytes, and no further: it is a power of two only for some
    // kGenerations (for example 128 or 256 bytes up to 30 counts, 512 from 47
    // to 62), not at the default 64 (640 bytes). Padding it to a power of two
    // would put the free pass's S loads of one slot, one per row at the same
    // offset, into a few L1 sets.
    struct alignas(128) CountRow {
        std::atomic<long> refs[kGenerations];   // refs[i]: memberships of the generation slot i currently represents joined through this shard, plus transient join attempts; RMW-only (J2): never stored after construction, never reset when the slot is recycled; value-initialized to 0
        std::atomic<size_t> joins{0};           // successful join attempts on this shard (summed into InternalCounters::joins)
        std::atomic<size_t> join_retries{0};    // join attempts on this shard whose re-verify failed (summed into InternalCounters::join_retries)
    }; // struct CountRow
    // A membership as a successful join returns it, and as a live handle
    // records it: the block joined and the count the join incremented. The
    // join returns the pair so that no caller assigns the handle's two words
    // separately (a dropped count must be impossible by construction, not by
    // care at each site).
    struct Membership {
        GenerationBlock* block;    // the generation slot joined
        std::atomic<long>* count;  // rows_[shard].refs[slot_index(block)], shard = the joining thread's
    }; // struct Membership
    // The bag slots of shard `s` in `block`: the retired sublist head
    // (nodes linked through `retire_link`) and the limbo sublist head (nodes
    // linked through `link`), a bare node address or EMPTY each. Readers (the
    // sweep, the diagnostics) get the value through a const block; the
    // writable slot is reachable only through a non-const block, which only
    // the advancer under the flag and the constructor hold: "owned by the
    // advancer" is in the signature.
    static word_t bag_retired(const GenerationBlock& block, size_t s) { return block.bag[2*s]; }
    static word_t bag_limbo(const GenerationBlock& block, size_t s) { return block.bag[2*s + 1]; }
    static word_t& bag_retired(GenerationBlock& block, size_t s) { return block.bag[2*s]; }
    static word_t& bag_limbo(GenerationBlock& block, size_t s) { return block.bag[2*s + 1]; }
    // The ring index of a block.
    size_t slot_index(const GenerationBlock* block) const noexcept { return static_cast<size_t>(block - ring_); }
    // The count of ring slot `block` for arena shard `shard`: the one place
    // that names a count by (block, shard). Never called with a slot
    // recomputed from a reloaded `current_` for an undo (THE JOIN, J1).
    std::atomic<long>& count_of(const GenerationBlock* block, size_t shard) const noexcept { return rows_[shard].refs[slot_index(block)]; }
    // The arena shard whose row holds `count`, and whether `count` is one of
    // `block`'s S counts at all (debug checks and the white-box shard
    // accessor). Address arithmetic on integers, so a foreign or null pointer
    // gives `false` rather than an unspecified pointer comparison; O(1),
    // which matters because HandleUse runs it on every operation in debug
    // builds.
    static size_t shard_of_count(const ConcurrentResizableHashSetRCU& set, const std::atomic<long>* count) noexcept {
        return static_cast<size_t>((reinterpret_cast<uintptr_t>(count) - reinterpret_cast<uintptr_t>(set.rows_.get()))/sizeof(CountRow));
    } // shard_of_count()
    static bool is_count_of(const GenerationBlock* block, const std::atomic<long>* count) noexcept {
        if (count == nullptr) return false;
        const ConcurrentResizableHashSetRCU& set = *block->set;
        if (reinterpret_cast<uintptr_t>(count) < reinterpret_cast<uintptr_t>(set.rows_.get())) return false;
        const size_t shard = shard_of_count(set, count);
        return shard <= set.arena_mask_ && count == &set.count_of(block, shard);
    } // is_count_of()
    // The sum and the minimum of slot `slot`'s S counts, each a relaxed load
    // (the diagnostics, the destructor and the accounting sweep; never the
    // free pass, which needs seq_cst loads and an early exit: free_oldest()).
    // Under concurrent joins or leaves the sum is not one instant's value.
    struct SlotCounts {
        long sum;  // the sum of the slot's counts
        long min;  // the smallest of them (negative only after a J2 violation)
    }; // struct SlotCounts
    SlotCounts slot_counts(size_t slot) const noexcept {
        SlotCounts c{0, 0};
        for (size_t s = 0; s <= arena_mask_; ++s) {   // every shard's count of the slot
            const long v = rows_[s].refs[slot].load(std::memory_order_relaxed);
            c.sum += v;
            if (s == 0 || v < c.min) c.min = v;
        } // every shard's count of the slot
        return c;
    } // slot_counts()

public:
    // A handle is a session with one generation of the set: while it lives,
    // no node it can observe is freed or reused (HANDLES in the class
    // overview). Move-only: the move constructor transfers the session --
    // the block and the per-shard count the join incremented, together, so
    // the counts are unchanged and the eventual leave lands on that count;
    // move assignment first ENDS the target's session if it has one (a
    // leave on the target's own count), then transfers; the moved-from
    // handle is empty. `block_` is the LIVENESS word: every decision to
    // leave tests it, and the count pointer is set exactly when it is. A
    // default-constructed handle is EMPTY. On an empty handle the only
    // valid operations are destruction, moving from and to it,
    // `operator bool` and `swap`; refresh() or any operation of the set
    // with an empty handle is a precondition violation (debug-asserted). An
    // empty handle may outlive its set; a live one may not. In release
    // builds a handle is two pointers (the block it joined and the per-shard
    // count it incremented); debug builds add the in-flight counter (the ODR
    // note under HANDLES). The set hands out no iterators
    // or references, so nothing obtained under a handle outlives a call:
    // a handle protects nothing between two calls. Thread rules: HANDLES
    // in the class overview (concurrent use in operations is fine;
    // refresh(), moving from the handle, move assignment over it, swap and
    // destruction are exclusive with every other use of it, debug-asserted
    // at each; on another thread they need THE HAND-OFF RULE's
    // synchronizing hand-off).
    class handle {
    public:
        // Postcondition: empty. nothrow.
        handle() noexcept = default;
        // Transfers `other`'s session -- the block AND the count its join
        // incremented, so the leave still lands on that count (J2 per count);
        // `other` is left empty. Precondition: no operation in flight on
        // `other` (debug-asserted). nothrow.
        handle(handle&& other) noexcept : block_(other.block_), count_(other.count_) {
            assert(other.not_in_flight() && "handle moved while an operation runs on it");
            other.block_ = nullptr;
            other.count_ = nullptr;
        } // handle(handle&&)
        // Ends this handle's session if it has one (a leave on THIS handle's
        // recorded count), then transfers `other`'s block and count; `other`
        // is left empty. Self-move-assignment is a no-op. Precondition: no
        // operation in flight on either handle (debug-asserted). nothrow.
        handle& operator=(handle&& other) noexcept {
            assert(not_in_flight() && "handle assigned over while an operation runs on it");
            assert(other.not_in_flight() && "handle moved while an operation runs on it");
            if (this == &other) return *this;
            if (block_) leave(block_, count_);
            block_ = other.block_;
            count_ = other.count_;
            other.block_ = nullptr;
            other.count_ = nullptr;
            return *this;
        } // handle::operator=(handle&&)
        handle(const handle&) = delete;
        handle& operator=(const handle&) = delete;
        // Ends the session (leaves the generation: a release decrement of the
        // recorded count) if live. `block_` is the liveness word: the count
        // pointer is dereferenced under that test without a test of its own,
        // so a count dropped by a defective move crashes here instead of
        // silently skipping the leave and pinning the generation forever.
        // Preconditions: if live, the set is still alive; no operation in
        // flight on this handle (debug-asserted). nothrow.
        ~handle() noexcept {
            assert(not_in_flight() && "handle destroyed while an operation runs on it");
            if (block_) leave(block_, count_);
        } // ~handle()

        // True iff this handle is live (holds a session). nothrow.
        explicit operator bool() const noexcept { return block_ != nullptr; }

        // Re-join the set's current generation. Preconditions: live; no
        // operation in flight on this handle (both debug-asserted).
        // Postcondition: the handle is a member of a generation that was
        // current at some point not happening-before the call's start (the
        // fast path may keep a generation that another thread has just
        // closed: safe, and merely conservative, it keeps that generation's
        // bag pinned until the next refresh). Returns true iff the handle
        // actually moved to a newer generation (false: it was already
        // current and nothing changed). Conversely, after a publish of any
        // cause -- an advance() or reclaim() that returned `advanced`, or a
        // trigger's bump -- that happens-before the call, a handle of an
        // older generation moves: the fast path's load then reads that
        // publish or a later one (coherence), and the handle's own block
        // cannot be CURRENT again while the handle pins it. Releases the
        // old generation for reclamation once no other handle pins it.
        // Allocates nothing. Lock-free. nothrow.
        //
        // The fast-path load is relaxed: it changes no membership, so it
        // has nothing to synchronize with. The only stale answer it can
        // give is "same" (coherence: it cannot return a value older than
        // this handle's own earlier read of `current_`, and `block_` cannot
        // be republished while this handle pins it), which keeps the handle
        // in a just-closed generation, safe and conservative (see the
        // postcondition). EXPECTED-EQUIVALENT to acquire. Slow path: join
        // the current block into a temporary handle (on the REFRESHING
        // thread's shard count), swap it with this one, and let the
        // temporary's destructor leave the old membership on the OLD
        // recorded count: join first, then leave (the order is free;
        // nothing relies on it), and no site assigns the two words
        // separately.
        bool refresh() noexcept {
            assert(block_ != nullptr && "refresh() of an empty handle");
            assert(not_in_flight() && "refresh() while an operation runs on this handle");
            const ConcurrentResizableHashSetRCU* set = block_->set;
            if (set->current_.load(std::memory_order_relaxed) == block_) return false;
            handle joined(set->join());
            swap(joined);
            return true;
        } // handle::refresh()

        // Exchanges the sessions of two handles, each block with its count;
        // either or both may be empty, and a self-swap is a no-op (the
        // handle keeps its session). Precondition: no operation in flight on
        // either (debug-asserted). nothrow.
        void swap(handle& other) noexcept {
            assert(not_in_flight() && other.not_in_flight() && "handle swapped while an operation runs on it");
            std::swap(block_, other.block_);
            std::swap(count_, other.count_);
        } // handle::swap()
        friend void swap(handle& a, handle& b) noexcept { a.swap(b); }

    private:
        friend class ConcurrentResizableHashSetRCU;
        // A live handle holding the membership `m` a successful join
        // returned (new_handle(), refresh()'s slow path, the walk).
        explicit handle(Membership m) noexcept : block_(m.block), count_(m.count) {}
        // True iff no operation of the set is running under this handle;
        // the precondition of refresh(), of the source of a move
        // construction, of either side of a move assignment or a swap, and
        // of destruction. In
        // release builds nothing is counted, and the check is vacuous.
        bool not_in_flight() const noexcept {
#ifndef NDEBUG
            return in_flight_.load(std::memory_order_relaxed) == 0;
#else
            return true;
#endif
        } // handle::not_in_flight()
        GenerationBlock* block_ = nullptr;     // the generation this handle is joined to; nullptr when empty (the LIVENESS word: every leave decision tests it). The owning set is `block_->set`
        std::atomic<long>* count_ = nullptr;   // the per-shard count the join incremented (the joining thread's shard), decremented by this handle's leave whatever thread runs it; nullptr exactly when `block_` is
#ifndef NDEBUG
        // Debug only: the operations of the set currently running under
        // this handle, counted by HandleUse at each operation's entry and
        // exit. `mutable` because operations take `const handle&`; atomic
        // because operations may share a handle concurrently (a relaxed
        // counter: it diagnoses a precondition violation, it orders
        // nothing). Not moved with the handle: a move with a non-zero count
        // is itself the violation.
        mutable std::atomic<unsigned> in_flight_{0};
#endif
    }; // class handle

private:
    // The debug check at every operation's entry, and the in-flight count
    // for the handle's duration: `h` must be live and belong to this set
    // (asserted), and the operation counts as in flight on `h` from here to
    // its exit on any path. In release builds the constructor is empty and
    // the object has no members: the operation touches nothing of its
    // handle. Not copyable.
#ifndef NDEBUG
    class HandleUse {
    public:
        HandleUse(const ConcurrentResizableHashSetRCU& set, const handle& h) noexcept : h_(h) {
            assert(h.block_ != nullptr && "operation with an empty handle");
            assert(h.block_->set == &set && "operation with a handle of another set");
            assert(is_count_of(h.block_, h.count_) && "operation with a handle whose count is not one of its block's (a move or swap lost or crossed it)");
            h_.in_flight_.fetch_add(1, std::memory_order_relaxed);
        } // HandleUse()
        ~HandleUse() { h_.in_flight_.fetch_sub(1, std::memory_order_relaxed); }
        HandleUse(const HandleUse&) = delete;
        HandleUse& operator=(const HandleUse&) = delete;
    private:
        const handle& h_;   // the handle the operation runs under
    }; // class HandleUse
#else
    struct HandleUse {
        HandleUse(const ConcurrentResizableHashSetRCU&, const handle&) noexcept {}
    }; // struct HandleUse
#endif

    // --- The join / leave protocol (THE JOIN in the class overview) ------

    // Count a new membership in the current generation, on the calling thread's
    // shard's count, and return it (block and count); `kHooked` selects whether
    // the two join hooks fire (join(): the handles' joins; join_unhooked():
    // advance()'s walk, which runs under the flag, cannot be stale, and must
    // not add to the hook call counts the test seams define). Const because a
    // const set yields handles. Lock-free: a retry happens only after an
    // advancer published a new generation. J1: between the `fetch_add` and a
    // successful re-verify this function touches NO node, head, bag slot, plain
    // block field or plain field of the count rows (`rows_` is read before the
    // increment and is constant after construction; the rows hold atomics
    // only). The calling thread's shard is computed ONCE per call (it cannot
    // change: thread_number() is fixed per thread, arena_mask_ per set), and
    // the count ONCE per attempt, before its increment; the undo decrements
    // THAT count. Recomputing the shard would be equivalent; recomputing the
    // slot from a reloaded `current_` would land the undo on another block's
    // count (a free under that block's live handle), which is why the count is
    // held, not recomputed.
    template <bool kHooked>
    Membership join_impl() const noexcept {
        const size_t shard = concurrent_hash_rcu_detail::thread_number() & arena_mask_;
        while (true) {
            // relaxed: a stale block here is caught by the re-verify
            // (EXPECTED-EQUIVALENT to acquire).
            GenerationBlock* block = current_.load(std::memory_order_relaxed);
            if constexpr (kHooked) Hooks::in_join();
            std::atomic<long>& count = count_of(block, shard);
            // seq_cst: one half of the store-buffering pair with the
            // advancer's seq_cst publish store and its seq_cst load of THIS
            // count in the free pass, which loads every count of the block
            // (DESIGNATED: an acq_rel increment lets the advancer miss this
            // handle and free the bag it is about to read).
            count.fetch_add(1, std::memory_order_seq_cst);
            if constexpr (kHooked) Hooks::in_join_after_bump();
            // seq_cst: the other half -- this load and the advancer's load
            // of this count cannot both miss (DESIGNATED: an acquire load
            // here races the same way). Pointer equality suffices (THE JOIN).
            if (current_.load(std::memory_order_seq_cst) == block) {
                rows_[shard].joins.fetch_add(1, std::memory_order_relaxed);
                return {block, &count};
            } // if the block is still current: joined
            // Undo the stale increment ON THE SAME COUNT and retry. release:
            // matches leave(); nothing was accessed since the increment (J1),
            // so there is nothing to publish and relaxed would be
            // EXPECTED-EQUIVALENT.
            rows_[shard].join_retries.fetch_add(1, std::memory_order_relaxed);
            [[maybe_unused]] const long prev = count.fetch_sub(1, std::memory_order_release);
            assert(prev > 0 && "a per-shard handle count went negative at a join's undo");   // every -1 follows its own +1 on the same count (J2)
        } // join attempts
    } // join_impl()
    Membership join() const noexcept { return join_impl<true>(); }
    Membership join_unhooked() const noexcept { return join_impl<false>(); }

    // Drop one membership of `block`, recorded on `count` (handle destruction,
    // the old side of refresh(), move assignment into a live handle), from
    // whatever thread runs it. Static: in release builds the leave is one RMW
    // on the recorded count, with no load of the block or the set. release
    // (DESIGNATED): every access made under the handle happens-before this
    // (sequenced before it, or ordered by the caller's synchronizing hand-off,
    // which THE HAND-OFF RULE under HANDLES requires of a leave on another
    // thread), and the advancer's seq_cst load of THIS count that reads the
    // resulting 0 (or a later value in the release sequence) acquires it before
    // the free pass rewrites a node's `link` and a popper assigns its value --
    // the first link of THE PER-NODE REUSE CHAIN; relaxed here lets the reuse
    // race the reader's last read. Precondition: `block` is non-null (the
    // caller tested the liveness word); debug builds check that `count` is set
    // and is one of `block`'s counts.
    static void leave([[maybe_unused]] const GenerationBlock* block, std::atomic<long>* count) noexcept {
        assert(count != nullptr && "a live handle without its count (a move or swap dropped it)");   // both null or both set
        assert(is_count_of(block, count) && "a handle's count is not one of its block's counts (a move or swap crossed it)");
        [[maybe_unused]] const long prev = count->fetch_sub(1, std::memory_order_release);
        assert(prev > 0 && "a per-shard handle count went negative at a leave");   // every -1 follows its own +1 on the same count (J2)
    } // leave()

    // Releases the advancer flag when its holder -- advance(), reclaim() or
    // the automatic trigger -- exits on any path (a throw from advance()'s
    // walk's Hash{} included: without it one throw would wedge reclamation
    // forever, every later call `contended` and every trigger skipped).
    // release: hands the plain ring fields and this holder's publish to the
    // next holder's acquire exchange.
    class AdvanceGuard {
    public:
        explicit AdvanceGuard(ConcurrentResizableHashSetRCU& set) noexcept : set_(set) {}
        ~AdvanceGuard() { set_.reclaiming_.store(false, std::memory_order_release); }
        AdvanceGuard(const AdvanceGuard&) = delete;
        AdvanceGuard& operator=(const AdvanceGuard&) = delete;
    private:
        ConcurrentResizableHashSetRCU& set_;   // the set whose flag this guard holds
    }; // class AdvanceGuard

    // Accumulates the wall time of its scope into one of the per-thread
    // timers (InternalTimers), on every exit path: a walk that throws still
    // charges its time. A steady-clock read at construction and one at
    // destruction; nothing else.
    class StepTimer {
    public:
        explicit StepTimer(size_t& accumulator) noexcept : accumulator_(accumulator), start_(std::chrono::steady_clock::now()) {}
        ~StepTimer() { accumulator_ += static_cast<size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_).count()); }
        StepTimer(const StepTimer&) = delete;
        StepTimer& operator=(const StepTimer&) = delete;
    private:
        size_t& accumulator_;                               // the timer this scope adds to
        const std::chrono::steady_clock::time_point start_; // when the scope began
    }; // class StepTimer

    // --- The three steps of advance() (the first two also reclaim()'s and
    // the trigger's): the bump, the free pass, the walk ---
    // All three run under the flag (ONE ADVANCER in the class overview):
    // `state`, `gen`, the bag slots, `oldest_`, `deal_`, `limbo_loaded_`
    // and `steal_stamp_` are plain fields of the flag holder, synchronized between holders by
    // the flag's acquire/release alone.

    // THE BUMP AND THE FINAL FREE PASS, the two steps every flag holder runs
    // (advance(), reclaim() and the automatic trigger): the bump under the
    // caller thread's bump_ns timer; the hook, once per call that publishes,
    // after the publish and before the final free pass (and, in advance(),
    // before the walk: TEST SEAMS) -- the flag is held, so a nested
    // advance() or reclaim() from the action returns `contended` and a
    // nested operation's retire push is a skipped trigger; then the final
    // free pass, for every outcome, under free_ns. Precondition: the caller
    // holds the flag. `result` accumulates the outcome and the counts.
    // noexcept: the three callees are. One copy on purpose, so that the
    // hook's position and the free pass's unconditional run are the same
    // fact for every holder.
    void bump_and_free(AdvanceResult& result) noexcept {
        {
            const StepTimer timer(internal_advance_timers().bump_ns);
            result.advance = bump(result);
        } // the bump, timed
        if (result.advance == AdvanceResult::Advance::advanced) Hooks::in_advance_after_publish();
        {
            const StepTimer timer(internal_advance_timers().free_ns);
            free_oldest(result);
        } // the final free pass, timed
    } // bump_and_free()

    // THE BUMP, steps A0-A5 of ONE ADVANCER: decide whether anything is
    // retired, close the current generation's retired and limbo lists into
    // its bag and publish the next generation. Returns the outcome, which
    // bump_and_free() stores in the result: `nothing_retired` (A0: nothing to
    // bag, no slot consumed, nothing published), `ring_full` (A2: the slot
    // after the current one is still SEALED after one free pass: no slot
    // consumed, nothing published, the free pass's counts in `result`) or
    // `advanced`. Never `contended`: the caller holds the flag. nothrow.
    AdvanceResult::Advance bump(AdvanceResult& result) noexcept {
        // A0: every shard's retired head and limbo head, relaxed: only the
        // emptiness is tested, and a push racing a load is seen by the next
        // call (EXPECTED-EQUIVALENT to acquire; the acquire that matters is
        // the exchange's, A3). ALL EMPTY -> nothing_retired, with no RMW on
        // any retired or free line: an exchange of an EMPTY head would take
        // the line for nothing. The limbo value loaded is kept per shard
        // (limbo_loaded_) and is the ONE load that decides A3's limbo
        // exchange: a limbo list that was EMPTY here is not exchanged (its
        // head shares the free line with the free head, which the shard's
        // poppers own; a push that lands after this load is bagged by the
        // next call, as a late retire push is), and no second load is made.
        bool any = false;
        for (size_t s = 0; s <= arena_mask_; ++s) {   // A0 over the shards
            const Shard& shard = shards_[s];
            limbo_loaded_[s] = shard.limbo_head.load(std::memory_order_relaxed);
            if (limbo_loaded_[s] != EMPTY || shard.retired_head.load(std::memory_order_relaxed) != EMPTY) any = true;
        } // A0 over the shards
        if (!any) return AdvanceResult::Advance::nothing_retired;
        // A1: relaxed: this holder's own store, or the previous holder's
        // through the flag's release/acquire (EXPECTED-EQUIVALENT to acquire).
        GenerationBlock* closing = current_.load(std::memory_order_relaxed);
        GenerationBlock* next = &ring_[(slot_index(closing) + 1) % kGenerations];
        // A2: the ring has wrapped: `next` is the oldest SEALED slot (the
        // SEALED run behind the CURRENT slot is contiguous, so the slot after
        // CURRENT is in it only when every other slot is). Free what can be
        // freed and look again; still occupied means the oldest generation
        // is pinned -- by a live handle, or transiently by a join attempt's
        // stale +1 -- a liveness matter either way: refuse, consume nothing.
        if (next->state != GenerationState::EMPTY) {
            free_oldest(result);
            if (next->state != GenerationState::EMPTY) {
                advance_counters_.ring_full.fetch_add(1, std::memory_order_relaxed);
                return AdvanceResult::Advance::ring_full;
            } // if the oldest generation is still pinned
        } // if the slot after the current one is in use
        assert(closing->state == GenerationState::CURRENT && "the closing block is not CURRENT");
        assert(next->state == GenerationState::EMPTY && "the next block is not EMPTY");   // structural: established by the re-check above
        // A3: every shard's retired list, and every limbo list A0 found
        // non-EMPTY, into the closing block's bag. acquire (DESIGNATED): the
        // exchange synchronizes with every release push below the value it
        // takes (the pushes are read-modify-writes on one release sequence),
        // so the sublists' `retire_link` and `link` chains are visible to
        // this holder, and through the flag to later holders, and the chain
        // of direction (I) of THE HANDLE INVARIANT runs through it to the
        // publish below; a relaxed exchange lets the free pass read a stale
        // next word, and lets a handle of the next generation read a node of
        // this bag. acq_rel would be EXPECTED-EQUIVALENT: nobody acquires the
        // EMPTY written here. EVERY exchange precedes the publish (A5): a
        // push that lands after its shard's exchange goes to the NEXT
        // generation, which is conservative by construction and is what
        // makes the bag form sound with no generation read at retire time;
        // an exchange after the publish would bag, into a generation a
        // handle may already have joined, a node that handle can still
        // reach.
        for (size_t s = 0; s <= arena_mask_; ++s) {   // A3 over the shards
            Shard& shard = shards_[s];
            bag_retired(*closing, s) = shard.retired_head.exchange(EMPTY, std::memory_order_acquire);
            if (limbo_loaded_[s] != EMPTY) bag_limbo(*closing, s) = shard.limbo_head.exchange(EMPTY, std::memory_order_acquire);
        } // A3 over the shards
        // A4: plain writes, owned by the flag holder; the bag is reachable
        // only through `closing` until the free pass empties it.
        closing->state = GenerationState::SEALED;
        next->state = GenerationState::CURRENT;
        next->gen = closing->gen + 1;
        // A5: seq_cst STORE: the advancer's half of the store-buffering pair
        // with the join's seq_cst increment and re-verify (DESIGNATED: a
        // relaxed or release publish lets a joiner use a bag a later free
        // pass frees under it; a seq_cst RMW instead of the store is
        // EXPECTED-EQUIVALENT). Also the release that direction (I)'s chain
        // ends on: sequenced after every exchange above. From here a joiner
        // lands in `next`; the hook fires in bump_and_free(), right after this.
        current_.store(next, std::memory_order_seq_cst);
        advance_counters_.bumps.fetch_add(1, std::memory_order_relaxed);
        return AdvanceResult::Advance::advanced;
    } // bump()

    // THE FREE PASS: free the SEALED bags oldest-first, cumulatively, while
    // the oldest has no live handle, accumulating into `result`. `oldest_`
    // indexes the oldest SEALED slot, or the CURRENT slot when nothing is
    // sealed, so the loop stops when ring_[oldest_] is CURRENT (nothing
    // sealed is left; the `!= SEALED` test also covers EMPTY defensively,
    // which the index never points at) or at the first of its S per-shard
    // counts that is not 0 (a handle of that generation on that count -- or
    // a join attempt's stale increment, J2 -- pins it and, by direction
    // (II) of THE HANDLE INVARIANT, every younger bag: a block is never
    // skipped). The counts are read one seq_cst load each, in ascending
    // shard order, before the bag is touched; the loop below says why each
    // load is seq_cst and why they need not be one snapshot. A freed block's
    // counts are left untouched (J2) and its bag slots are walked IN PLACE,
    // each slot holding the sublist's next head until it reads EMPTY, so a bag is
    // EMPTY the moment its last node is pushed. Per node: the next word --
    // `retire_link` on a retired sublist, `link` on a limbo sublist -- is
    // read BEFORE push_free() rewrites `link` into the free list's next
    // pointer (a limbo sublist's next IS `link`); the node is dealt
    // round-robin over the shards' free lists with the one cursor `deal_`,
    // on top of whatever those lists hold, for every bag and every sublist
    // alike: a shard whose threads lost many splits or erased many keys does
    // not hoard its own garbage, and splicing a sublist whole would save
    // nothing (finding its tail is the same walk). Then the slot goes EMPTY,
    // `oldest_` moves on, and the count is subtracted from node_count_ (the
    // resize hint: its comment says why the subtraction keeps it one-sided
    // at quiescent points and why it never goes below zero). Checks (debug
    // builds): every retired node is tagged
    // (only dead nodes are retired, and every dead node in a chain is
    // tagged: a live node here means a run walk overshot its run, or a node
    // was retired on a lost CAS), and a bag never holds more nodes than the
    // arena has slots (a longer walk is a cycle from a double retirement,
    // which would otherwise hang this loop). Under ASan push_free() poisons
    // each node's value bytes before it pushes the node (ASAN POISONING in
    // the class overview), so every node a popper can win is poisoned until
    // that popper unpoisons it; limbo nodes (never freed, or unpoisoned by
    // the popper whose assignment threw) are poisoned here like retired
    // ones, with no site of their own. nothrow.
    void free_oldest(AdvanceResult& result) noexcept {
        while (true) {   // free the oldest sealed bag while it is unpinned
            GenerationBlock& block = ring_[oldest_];
            if (block.state != GenerationState::SEALED) return;
            // The bag is unpinned iff EVERY one of its S per-shard counts
            // reads 0; the pass stops at the first that does not. Each load
            // is seq_cst: the advancer's half of the store-buffering pair
            // with the joins' seq_cst increments and re-verifies on that
            // count (DESIGNATED: an acquire load here frees under a joiner
            // whose increment it missed). Each also acquires the leaves whose
            // release decrements of that count produced the 0, so every
            // access those handles made happens-before the `link` rewrites
            // and the pushes below: the second link of THE PER-NODE REUSE
            // CHAIN. The loads need not be one snapshot: after the publish
            // that sealed this block no join of it succeeds, and no
            // membership moves between counts, so a count read 0 stays 0 for
            // memberships of this generation. Ascending order: free for the
            // protocol (any order, or a sum, would do: the counts are never
            // negative); the early exit is why this is a loop of its own and
            // not the diagnostics' relaxed sum (slot_counts()).
            for (size_t s = 0; s <= arena_mask_; ++s) {   // every shard's count of this generation
                if (rows_[s].refs[oldest_].load(std::memory_order_seq_cst) != 0) return;
            } // every shard's count of this generation
#ifndef NDEBUG
            const size_t slots = get_internal_node_count();   // the cycle bound (an assert-only shard-size sum)
#endif
            size_t count = 0;
            for (size_t s = 0; s <= arena_mask_; ++s) {   // every shard's two sublists of this bag
                word_t& retired = bag_retired(block, s);
                while (addr_of(retired) != EMPTY) {   // the retired sublist, through retire_link
                    Node* node = node_of(retired);
                    assert((node->link.load(std::memory_order_relaxed) & (MARK_BIT | FROZEN_BIT)) != 0 && "a bagged retired node is not tagged: a live node was retired");
                    retired = node->retire_link.load(std::memory_order_relaxed);   // the next word, read before push_free() rewrites `link`; retire_link stays stale
                    push_free(shards_[deal_++ & arena_mask_], node);
                    ++count;
                    assert(count <= slots && "more bagged nodes than arena slots: a sublist has a cycle (a node retired twice)");
                } // the retired sublist
                word_t& limbo = bag_limbo(block, s);
                while (addr_of(limbo) != EMPTY) {   // the limbo sublist, through link
                    Node* node = node_of(limbo);
                    limbo = node->link.load(std::memory_order_relaxed);   // the next word IS `link`: read before push_free() rewrites it
                    push_free(shards_[deal_++ & arena_mask_], node);
                    ++count;
                    assert(count <= slots && "more bagged nodes than arena slots: a sublist has a cycle");
                } // the limbo sublist
            } // every shard's two sublists of this bag
            block.state = GenerationState::EMPTY;
            oldest_ = (oldest_ + 1) % kGenerations;
            node_count_.fetch_sub(count, std::memory_order_relaxed);   // relaxed: a hint, read relaxed by insert(); never below zero (see node_count_)
            // The steal's hint: a new stamp, EMPTY clear, after this bag's
            // pushes. release: a scan that reads it with its acquire reload
            // sees every push above (THE HINT at steal_scan()). Only when
            // the bag held a node: an empty bag changed no free list.
            if (count != 0) steal_hint_.store(++steal_stamp_ << 1, std::memory_order_release);
            result.freed_nodes += count;
            ++result.freed_bags;
            advance_counters_.bags_freed.fetch_add(1, std::memory_order_relaxed);
            advance_counters_.nodes_freed.fetch_add(count, std::memory_order_relaxed);
        } // free the oldest sealed bag while it is unpinned
    } // free_oldest()

    // THE WALK: advance()'s third step, over EVERY published bucket, under a
    // handle of advance()'s own (joined through the hook-free join path:
    // under the flag it cannot be stale, and it must not add to the hook
    // call counts the test seams define; it is a `handle` so that the leave
    // is RAII, on the throwing path too), retiring what it bypasses onto the
    // retired lists of the generation that is current after the bump: two
    // clicks (ONE ADVANCER in the class overview). The shape is the split
    // winner's cleanup pass (cleanup_parent()): one writer walk per chain
    // with step_over(), one unlink attempt per dead run through the last
    // live word before it, never retried (a lost CAS is charged to
    // advance_cas_failures, and to advance_cas_peer_losses when a peer
    // bypassed the same run), dead by is_dead_now() at the `ts` acquired
    // once here, after the join, so nodes FROZEN for a doubling this `ts`
    // does not cover are left (not dead to this walk), as is every dead run
    // behind a FROZEN node whose child is unpublished (no walk can reach
    // it: a tagged node is never a pred). Hash{} is called on every FROZEN
    // node the walk passes with a usable pred (H5' through step_over(): a
    // FROZEN node right after a live predecessor, or at the head, is
    // hashed by is_dead_now() whether or not it turns out dead), and that
    // is the one source of a throw here. No split, no allocation, no lock;
    // the only values read are those of the FROZEN nodes hashed, and no
    // value is compared. C1 (ENTRY): every chain is entered through
    // `buckets_[j].load(acquire)` AFTER this handle's join and through
    // nothing else, and every further node is reached only through the
    // `link` of a node reached from that head in this pass over the chain;
    // no address is retained across buckets or from before the join.
    // UNINITIALIZED buckets are skipped and not counted (nothing is
    // published in them). The loop body is a COPY of cleanup_parent()'s,
    // plus the counting, rather than a shared helper, so that the cleanup
    // pass keeps exactly the code its correctness argument was made for;
    // the two copies are the same writer walk and change together.
    // Counts: `buckets_walked`; `walk_retired` (the nodes of
    // every run whose CAS this walk won; a won CAS is recognized after
    // step_over() by the pred word now leading to the run's successor,
    // which only this thread's own installed value can make it do, and the
    // run is then re-walked through its immutable tagged links, which this
    // thread already loaded with acquire inside unlink_run(), to count it);
    // `nodes_passed` (every node this loop loaded, plus the interior of
    // every run it won, counted by that same re-walk: a run that went
    // through unlink_run() is passed as a whole, so without the re-walk it
    // would count only its first node). A tagged node step_over() does NOT
    // attempt to bypass (an unusable pred after a lost CAS, or a FROZEN
    // node with an unpublished child, and every dead node behind such a
    // node) is exited through its own link, so this loop loads the next
    // node itself: such runs are walked node by node and counted in full.
    // Only a run whose one-shot CAS was LOST is passed as a whole and not
    // re-walked, and counts its first node alone (reasoned, not
    // demonstrated: a loss needs a concurrent writer on the same word). An
    // uncontended walk therefore passes exactly the reachable nodes.
    void walk(AdvanceResult& result) {
        const handle h(join_unhooked());   // on the advancer thread's shard count; left by its destructor on every path
        const size_t ts = table_size_.load(std::memory_order_acquire);
        for (size_t j = 0; j < ts; ++j) {   // every published bucket
            WalkState w(&buckets_[j], buckets_[j].load(std::memory_order_acquire));   // C1: the one entry into chain j
            word_t curr = addr_of(w.pred_word);
            if (curr == UNINITIALIZED) continue;
            ++result.buckets_walked;
            while (addr_of(curr) != EMPTY) {   // walk bucket j's chain
                Node* node = node_of(curr);
                const word_t link = node->link.load(std::memory_order_acquire);
                ++result.nodes_passed;
                curr = step_over(w, node, link, j, ts, UnlinkSite::advance);
                if ((link & (MARK_BIT | FROZEN_BIT)) != 0 && w.pred_ok && addr_of(w.pred_word) == curr) {   // the run from `node` was bypassed by our own CAS
                    ++result.walk_retired;   // `node` itself, already in nodes_passed
                    for (word_t r = addr_of(link); r != curr; r = addr_of(node_of(r)->link.load(std::memory_order_relaxed))) {   // the run's interior, which step_over() passed as a whole
                        ++result.walk_retired;
                        ++result.nodes_passed;
                    } // the run's interior
                } // if this walk retired the run
            } // walk bucket j's chain
        } // every published bucket
    } // walk()

    // --- ASan poisoning of free nodes (ASAN POISONING in the overview) ---
    // Three sites: push_free() poisons (the free pass), alloc_node()
    // unpoisons the node it pops, the destructor unpoisons every free node.

    // The extent poisoned while a node is free: [&value, &link), by address
    // difference and not sizeof(T) (ASan's granule is 8 bytes: sizeof(int)
    // would poison nothing) nor offsetof (`-Winvalid-offsetof` under
    // -Werror for a non-standard-layout T). `link` and `retire_link` stay
    // readable for the free list, a losing popper and the sweep.
    static size_t poison_extent(const Node* n) noexcept {
        return static_cast<size_t>(reinterpret_cast<const std::byte*>(&n->link) - reinterpret_cast<const std::byte*>(&n->value));
    } // poison_extent()
    static void poison_node(Node* n) noexcept { CHS_RCU_ASAN_POISON(&n->value, poison_extent(n)); }
    static void unpoison_node(Node* n) noexcept { CHS_RCU_ASAN_UNPOISON(&n->value, poison_extent(n)); }
    // Under ASan: every byte of the poisoned extent is poisoned (per byte:
    // `__asan_region_is_poisoned` answers "any byte", which is not the
    // test) and neither `link` nor `retire_link` is. Without ASan there is
    // no oracle: true.
    static bool node_is_poisoned([[maybe_unused]] const Node* n) noexcept {
#ifdef CHS_RCU_HAVE_ASAN
        const std::byte* bytes = reinterpret_cast<const std::byte*>(&n->value);
        for (size_t i = 0; i < poison_extent(n); ++i) {
            if (!__asan_address_is_poisoned(bytes + i)) return false;
        } // loop over the value bytes
        return __asan_region_is_poisoned(const_cast<void*>(static_cast<const void*>(&n->link)), sizeof(n->link)) == nullptr
            && __asan_region_is_poisoned(const_cast<void*>(static_cast<const void*>(&n->retire_link)), sizeof(n->retire_link)) == nullptr;
#else
        return true;
#endif
    } // node_is_poisoned()

public:
    // Preconditions (debug-asserted): no advance(), reclaim() or automatic
    // trigger in progress, no handle of this set alive, no concurrent
    // operation (the destruction is quiescent, as the constructor's note
    // says). The arena's deques then destroy every slot; bags pending at
    // destruction (SEALED generations whose holders left after the last
    // free pass, or that no free pass got to)
    // are pushed nowhere: their slots belong to the deques and are
    // destroyed with them, like every other slot. Under ASan every free
    // node's value bytes are unpoisoned first (the third poison site): the
    // deque runs ~T on every slot, and a ~T that reads its object would
    // otherwise be reported. All loads relaxed: exclusive by precondition.
    ~ConcurrentResizableHashSetRCU() {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "set destroyed during advance(), reclaim() or an automatic trigger");
#ifndef NDEBUG
        for (size_t i = 0; i < kGenerations; ++i) {   // every count of every slot is 0 (non-negative with sum 0)
            const SlotCounts counts = slot_counts(i);
            assert(counts.min == 0 && counts.sum == 0 && "set destroyed with live handles (a per-shard handle count is not 0)");
        } // loop over the ring
        // The cadence's counting invariants, exact here because destruction
        // is quiescent (THE TRIGGER in the class overview says why each
        // holds): no attempt on a cadence-off set; at most one attempt per
        // multiple of the cadence that the shards' retired-node counts
        // crossed (the division form cannot overflow at the largest
        // cadence); at a cadence of 1 exactly one attempt per retired run.
        {
            const size_t every = get_internal_auto_reclaim_every();
            const InternalCounters c = get_internal_counters();
            const size_t attempts = c.auto_reclaims + c.auto_skipped;
            assert((every != 0 || attempts == 0) && "automatic reclaim attempts on a set with the cadence off");
            assert((every == 0 || attempts <= c.retire_nodes / every) && "more automatic reclaim attempts than multiples of the cadence crossed");
            assert((every != 1 || attempts == c.retire_runs) && "at a cadence of 1 the attempts do not equal the retired runs");
        } // the cadence invariants
#endif
#ifdef CHS_RCU_HAVE_ASAN
        for (size_t s = 0; s <= arena_mask_; ++s) {   // unpoison every free node
            for (word_t curr = free_addr_of(shards_[s].free_head.load(std::memory_order_relaxed)); curr != FREE_EMPTY;) {
                Node* node = node_of(curr);
                unpoison_node(node);
                curr = node->link.load(std::memory_order_relaxed);
            } // walk the free list
        } // loop over the shards
#endif
    } // ~ConcurrentResizableHashSetRCU()
    ConcurrentResizableHashSetRCU(const ConcurrentResizableHashSetRCU&) = delete;
    ConcurrentResizableHashSetRCU& operator=(const ConcurrentResizableHashSetRCU&) = delete;

    // Join the current generation and return a live handle for it. Const:
    // a const set yields handles too. Precondition: the set is alive.
    // Postcondition: the handle is live, belongs to this set, and its
    // generation was current at some point during the call. Allocates
    // nothing and touches no node, head or bag (J1). Lock-free; never waits
    // for an advance(). nothrow.
    [[nodiscard]] handle new_handle() const noexcept {
        return handle(join());
    } // new_handle()

    // Advance the generation and reclaim (ONE ADVANCER in the class
    // overview, which states the steps A0-A5, the free pass and the walk):
    // close the current generation's retired and limbo lists into its bag,
    // publish the next generation, free every bag no live handle can
    // observe (oldest first, cumulatively), then walk every published
    // chain and retire the dead nodes it can bypass. reclaim() below is
    // this call without the walk, and the automatic trigger (THE TRIGGER)
    // is reclaim()'s work run by the set itself. Needs no handle (it
    // joins one of its own for the walk); non-const. Postcondition: the
    // result describes exactly what this call did (AdvanceResult);
    // `contended` means another holder -- an advance(), a reclaim() or, on a
    // cadenced set, an operating thread's trigger -- had the flag and this
    // call did nothing; at a quiescent point a call is never `contended`.
    // Non-blocking: never waits for another advancer or for
    // handles. Not noexcept: the walk hashes the FROZEN nodes it passes,
    // and Hash{} may throw; a throw propagates with the basic guarantee
    // (the unlinks the walk won stand, their retirees are on the current
    // generation's retired lists, the rest are stragglers for the next
    // call; the bump and the free pass completed before the walk) and the
    // result is lost; the flag and the walk's handle are released on every
    // path. A thread that holds a handle should refresh or release it
    // before calling: its own handle pins the generation this call closes.
    // Not [[nodiscard]]: calling on a schedule and ignoring the result is
    // the normal use, and every refusal is retried by calling again. Cost,
    // in kind, on the calling thread: the walk visits every published
    // bucket and every node in it, for every outcome but `ring_full` and
    // `contended` (a `nothing_retired` call walks the whole table too), and
    // the free pass deals round-robin, so a bag of at least as many nodes as
    // there are shards touches every shard's free line; the call is meant to be
    // made rarely, on a schedule the caller chooses (a caller who wants the
    // bump and the free pass often and the walk rarely calls reclaim() for
    // the former, or lets the set do so through its cadence).
    //
    // FLOW (the steps are bump(), free_oldest() and walk(), each commented
    // at its definition; steps 2 and 3 are bump_and_free(), shared with
    // reclaim() and the trigger; the per-thread timers of InternalTimers
    // bracket each of them and the whole call):
    //   1. the flag, or `contended`;
    //   2. the BUMP: A0 decides `nothing_retired` by loads; a wrapped ring
    //      gets one free pass and, still full, `ring_full`; else A3-A5
    //      exchange and publish, and the hook fires;
    //   3. the FINAL FREE PASS, for every outcome (a `nothing_retired` call
    //      frees the bags whose pins left since the last call; a
    //      `ring_full` call runs it as a second pass after the in-wrap one,
    //      which frees only what a leave between the two passes unpinned,
    //      a liveness detail that costs one state test and one seq_cst load
    //      when nothing has changed, and keeps the structure "the bump
    //      decides, then one unconditional free pass" uniform);
    //   4. the WALK, for every outcome but `ring_full` and `contended` (a
    //      cost choice: on a full ring its retirees could not be bagged
    //      until the pin goes away; ONE ADVANCER, A2, in the class overview
    //      says why it is not an invariant).
    // Against the four steps a QUIESCENT collector of this structure would
    // take (clear the pop counters; drain the limbo and retired lists onto
    // the free lists; walk every chain unlinking the dead with plain
    // stores and freeing them on the spot; reset node_count_ from the exact
    // live count), which is what this call does without quiescence: the
    // counters are never cleared (node_count_'s batching does not depend on
    // a reset); the drain is the bump's exchange into the bag plus the free
    // pass, under the handle invariant; the chain walk is the walk with
    // one-shot unlink CASes, which retires into the current generation
    // instead of freeing on the spot (two clicks); the reset is the free
    // pass's subtraction of what it freed, and no exact live count exists.
    AdvanceResult advance() {
        const StepTimer whole(internal_advance_timers().advance_ns);
        // acquire: pairs with the previous holder's release store in
        // AdvanceGuard -- this is the ONLY synchronization on the plain
        // fields `state`, `gen`, the bag slots, `oldest_`, `deal_`,
        // `limbo_loaded_` and `steal_stamp_`, and it places the previous holder's publish
        // before this holder's `refs` loads (DESIGNATED: with relaxed orders
        // here two callers race on the plain fields).
        if (reclaiming_.exchange(true, std::memory_order_acquire)) {
            advance_counters_.contended.fetch_add(1, std::memory_order_relaxed);
            return {0, 0, 0, 0, 0, AdvanceResult::Advance::contended};
        } // if another holder has the flag
        const AdvanceGuard guard(*this);
        AdvanceResult result{0, 0, 0, 0, 0, AdvanceResult::Advance::nothing_retired};
        bump_and_free(result);   // steps 2 and 3, shared with reclaim() and the trigger
        if (result.advance != AdvanceResult::Advance::ring_full) {
            const StepTimer timer(internal_advance_timers().walk_ns);
            walk(result);
        } // the walk, timed, unless the ring is full
        return result;
    } // advance()

    // Advance the generation and reclaim WITHOUT the walk: advance()'s steps
    // 1-3 (the flag or `contended`; the bump; the final free pass), nothing
    // else (advance() = reclaim() + the walk). The call for a caller who
    // wants memory reused often and the whole-table walk rarely: its cost is
    // O(shards) for the loads and exchanges plus O(nodes freed) for the free
    // pass, with no term in the table size, so a `nothing_retired` call is
    // cheap. It collects no straggler: a dead run a writer walk failed to
    // bypass stays reachable across any number of reclaim() calls, until a
    // writer walk, a split's cleanup pass or an advance() passes it. The
    // name once belonged to a quiescent collector in an earlier version of
    // this class; that collector is gone, and this reclaim() is the bump and
    // the free pass. Same flag, same ring, same
    // counters, same hook (once per publish), same result type with the walk
    // counts 0 for every outcome; the outcomes and their meaning are
    // advance()'s. The caller's own handle pins the generation this call
    // closes, exactly as for advance(): refresh or release it first.
    // noexcept: the bump, the free pass and the hook are noexcept, nothing
    // here hashes, compares, copies or allocates, and the guards' destructors
    // throw nothing. Not [[nodiscard]], as advance(). Clock reads: six per
    // call that proceeds (the whole call, the bump, the free pass), two per
    // `contended` call.
    AdvanceResult reclaim() noexcept {
        const StepTimer whole(internal_advance_timers().advance_ns);
        // acquire: as advance()'s (the ONLY synchronization between holders
        // on the plain ring fields; DESIGNATED there).
        if (reclaiming_.exchange(true, std::memory_order_acquire)) {
            advance_counters_.contended.fetch_add(1, std::memory_order_relaxed);
            return {0, 0, 0, 0, 0, AdvanceResult::Advance::contended};
        } // if another holder has the flag
        const AdvanceGuard guard(*this);
        AdvanceResult result{0, 0, 0, 0, 0, AdvanceResult::Advance::nothing_retired};
        bump_and_free(result);
        return result;
    } // reclaim()

    // --- Diagnostics of the generation protocol (see DIAGNOSTICS) --------

    // Diagnostic accessor, a RING VIEW (DIAGNOSTICS): the current
    // generation number. Precondition: no flag holder in progress -- no
    // advance(), no reclaim(), and on a cadenced set no operation (`gen` is
    // a plain field the holder writes; live idle handles are fine). Exact
    // under that precondition.
    // The load of `current_` is acquire: it reads the publish (a seq_cst
    // store, sequenced after the advancer's write of the block's `gen`), so
    // that write happens-before the read of `gen` even for a caller that
    // never synchronized with the advancer otherwise; relaxed would race
    // with that write.
    uint64_t get_internal_generation() const {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "get_internal_generation() during advance(), reclaim() or an automatic trigger");
        return current_.load(std::memory_order_acquire)->gen;
    } // get_internal_generation()
    // Diagnostic accessor: the generation a live handle is joined to.
    // Race-free: the handle's block cannot be recycled, and so its `gen`
    // cannot be rewritten, while the handle's +1 is on it, and the publish
    // that made the block current happens-before the handle's successful
    // re-verify. Precondition: `h` is live (debug-asserted).
    static uint64_t get_internal_handle_generation(const handle& h) {
        assert(h.block_ != nullptr && "generation of an empty handle");
        return h.block_->gen;
    } // get_internal_handle_generation()
    // Diagnostic accessor, a RING VIEW (DIAGNOSTICS: no flag holder in
    // progress, every earlier one happens-before the call; on a cadenced
    // set no operation either): the number of nodes in the bag of
    // generation `gen`, retired and limbo sublists over every shard, or 0 if
    // no SEALED slot represents that generation (it was freed, or never
    // closed). Walks the sublists without the sweep's cycle guard: a
    // corrupted bag is the sweep's business.
    size_t get_internal_bag_count(uint64_t gen) const {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "get_internal_bag_count() during advance(), reclaim() or an automatic trigger");
        for (const GenerationBlock& block : ring_) {
            if (block.state != GenerationState::SEALED || block.gen != gen) continue;
            size_t n = 0;
            for (size_t s = 0; s <= arena_mask_; ++s) {
                for (word_t curr = bag_retired(block, s); addr_of(curr) != EMPTY; curr = node_of(curr)->retire_link.load(std::memory_order_relaxed)) ++n;
                for (word_t curr = bag_limbo(block, s); addr_of(curr) != EMPTY; curr = node_of(curr)->link.load(std::memory_order_relaxed)) ++n;
            } // loop over the shards' slots of this bag
            return n;
        } // loop over the ring
        return 0;
    } // get_internal_bag_count()
    // Diagnostic accessor, quiescent only: the free-list length of every
    // shard, in shard order, as the accounting sweep counts them (the
    // sweep's caveats apply).
    std::vector<size_t> get_internal_free_per_shard() const { return get_internal_accounting().free_per_shard; }
    // The diagnostic ring-state view: one entry per slot, in ring order,
    // plus the indices of the CURRENT slot and of the oldest SEALED slot
    // (`oldest_`: the CURRENT slot's index when nothing is sealed).
    struct InternalRingSlot {
        GenerationState state;  // EMPTY / CURRENT / SEALED
        uint64_t gen;           // the generation number the slot represents (stale for an EMPTY slot)
        long refs;              // the slot's live-handle count: the sum of its S per-shard counts (relaxed loads: with operations running, not one instant's value)
        size_t bagged;          // nodes in the slot's bag (0 unless SEALED)
    }; // struct InternalRingSlot
    struct InternalRingState {
        size_t current_slot;                 // the index of the CURRENT slot
        size_t oldest_slot;                  // `oldest_`
        std::vector<InternalRingSlot> slots; // one per slot, index = ring index
    }; // struct InternalRingState
    // Diagnostic accessor, a RING VIEW (DIAGNOSTICS: no flag holder in
    // progress, every earlier one happens-before the call; live idle
    // handles may run, and on a cadence-off set operations too): the
    // ring-state view above.
    InternalRingState get_internal_ring_state() const {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "get_internal_ring_state() during advance(), reclaim() or an automatic trigger");
        InternalRingState view{slot_index(current_.load(std::memory_order_relaxed)), oldest_, {}};
        view.slots.reserve(kGenerations);
        for (const GenerationBlock& block : ring_) {
            view.slots.push_back({block.state, block.gen, slot_counts(slot_index(&block)).sum,
                                  block.state == GenerationState::SEALED ? get_internal_bag_count(block.gen) : size_t{0}});
        } // loop over the ring
        return view;
    } // get_internal_ring_state()
    // Diagnostic accessor, RACE-FREE (relaxed loads of atomics only; no
    // plain ring field, hence keyed by SLOT index, not by generation; no
    // flag assert, so a test may call it inside in_advance_after_publish()):
    // the S per-shard counts of ring slot `slot` (< kGenerations), in shard
    // order. With joins or leaves running, each value is one its count had,
    // not one instant's set. The slot of generation g is g % kGenerations
    // (DIAGNOSTICS).
    std::vector<long> get_internal_slot_counts(size_t slot) const {
        assert(slot < kGenerations && "get_internal_slot_counts() of a slot outside the ring");
        std::vector<long> counts;
        counts.reserve(arena_mask_ + 1);
        for (size_t s = 0; s <= arena_mask_; ++s) counts.push_back(rows_[s].refs[slot].load(std::memory_order_relaxed));
        return counts;
    } // get_internal_slot_counts()
    // Diagnostic accessor: the arena shard whose count a live handle's join
    // incremented (the shard of the thread that joined it, or that ran its last
    // refresh() that returned true), which its leave decrements whatever thread
    // runs it. Race-free: the handle's own words. Precondition: `h` is live
    // (debug-asserted).
    static size_t get_internal_handle_shard(const handle& h) {
        assert(h.block_ != nullptr && "shard of an empty handle");
        assert(h.count_ != nullptr && "shard of a live handle without its count (a move or swap dropped it)");
        return shard_of_count(*h.block_->set, h.count_);
    } // get_internal_handle_shard()

    // The per-thread timers of reclamation: the time the CALLING thread
    // spent in its own explicit calls (advance() and reclaim(): the whole
    // call and its steps) and in the automatic triggers that PROCEEDED on
    // this thread (their bump and free pass only), accumulated in
    // nanoseconds; a thread's own accumulators, reached from that thread,
    // so a benchmark thread reads its own share without attributing
    // another thread's call or trigger to itself. One set of accumulators
    // per thread per instantiation, shared by every set of that type the
    // thread reclaims on; the caller resets them through the returned
    // reference. Per thread, advance_ns - bump_ns - free_ns - walk_ns is
    // the explicit calls' own overhead for any mix of the two calls, and
    // stops being so once a trigger proceeded on that thread (its time is
    // in bump_ns and free_ns with no advance_ns to subtract from). Always
    // compiled, as the counters are: two steady-clock reads per timed
    // scope, so advance(): eight per call that walks (`advanced` or
    // `nothing_retired`), six per `ring_full` call (no walk), two per
    // `contended` call; reclaim(): six per call that proceeds, two per
    // `contended` call; a proceeding trigger four, a skipped trigger none.
    // A call that throws from the walk still charges its time (StepTimer).
    struct InternalTimers {
        size_t advance_ns;  // the explicit calls, advance() and reclaim(), from entry to return (before the flag's exchange; `contended` calls included); never a trigger
        size_t bump_ns;     // the bump, of a call or of a trigger that proceeded on this thread: the emptiness loads, the ring-wrap free pass if any, the exchanges and the publish
        size_t free_ns;     // the final free pass after the bump, of a call or of a trigger that proceeded on this thread
        size_t walk_ns;     // advance()'s walk
    }; // struct InternalTimers
    static InternalTimers& internal_advance_timers() noexcept {
        static thread_local InternalTimers timers{0, 0, 0, 0};
        return timers;
    } // internal_advance_timers()

private:
    // The generation protocol's set-wide diagnostic counters, on lines of
    // their own (never on a free or retired line); relaxed fetch_adds by the
    // thread that took the counted step, exact at a quiescent point, a
    // snapshot otherwise; summed into InternalCounters, which says what each
    // counts. Every writer is a non-const member (the join, which is
    // const, counts in the rows instead), so the member is not `mutable`.
    // The join counters are not here: they are per shard in the count rows
    // (CountRow), so a join writes no set-wide line.
    // Field ORDER is part of the layout: the two cold counters of the
    // automatic trigger (written only in the noinline auto_reclaim()) come
    // first, and the two counters an operation writes, `pops_raced_push`
    // and `limbo_pushes`, sit at offsets 64 and 72, the offsets the
    // operations' code addresses (an order that moved them would change an
    // operation's object code for no gain); 80 bytes of fields, sizeof 128.
    struct alignas(64) AdvanceCounters {
        std::atomic<size_t> auto_reclaims{0};   // automatic triggers that acquired the flag (THE TRIGGER in the class overview)
        std::atomic<size_t> auto_skipped{0};    // automatic triggers that found the flag held
        std::atomic<size_t> bumps{0};
        std::atomic<size_t> ring_full{0};
        std::atomic<size_t> contended{0};
        std::atomic<size_t> bags_freed{0};
        std::atomic<size_t> nodes_freed{0};
        std::atomic<size_t> free_push_cas_failures{0};
        std::atomic<size_t> pops_raced_push{0};
        std::atomic<size_t> limbo_pushes{0};
    }; // struct AdvanceCounters
    AdvanceCounters advance_counters_;
    // The generation ring (THE RING in the class overview): kGenerations
    // blocks, slot 0 CURRENT at construction. Declared before `current_`,
    // whose initializer names it.
    GenerationBlock ring_[kGenerations];
    // The block new handles join: written only by the flag holder (a seq_cst
    // store, the publish), read by every join attempt and by refresh()'s
    // fast path. On a line of its own, apart from node_count_ (written by
    // every allocating thread) and from the flag.
    alignas(128) std::atomic<GenerationBlock*> current_{&ring_[0]};
    // The single-advancer try-flag: set while an advance(), a reclaim() or
    // an automatic trigger is in progress (ONE ADVANCER). On a line of its
    // own; `oldest_` and `deal_`, owned by the flag holder, share it.
    alignas(128) std::atomic<bool> reclaiming_{false};
    size_t oldest_ = 0;   // index of the oldest SEALED slot, or of the CURRENT slot when nothing is sealed; owned by the flag holder
    size_t deal_ = 0;     // the free pass's round-robin cursor: the shard the next freed node goes to; owned by the flag holder
    // The bump's scratch, one word per shard (allocated by the constructor):
    // the limbo head of each shard as step A0 loaded it, which is the one
    // load that decides whether A3 exchanges that shard's limbo list. Owned
    // by the flag holder; meaningless outside a bump.
    std::unique_ptr<word_t[]> limbo_loaded_;
    // The steal's set-wide hint (THE HINT at steal_scan()): a stamp in bits
    // 63..1 and STEAL_HINT_EMPTY in bit 0, "a scan found every free list
    // empty since the last stamp". Written by the free pass (a release
    // store of a new stamp, EMPTY clear, after each bag it frees) and by a
    // scan that found nothing (one relaxed CAS setting EMPTY); read with one
    // relaxed load by every append that missed its own free list. EMPTY at
    // construction: no free list has a node. A hint only: no correctness
    // argument rests on it. A 128-byte block of its own (a pair of 64-byte
    // lines, so the adjacent-line prefetcher does not couple it to its
    // neighbours), like `current_` and `reclaiming_`: every append that
    // misses its own list reads it, while the flag's block holds `deal_`,
    // written once per freed node; the only other field in the block,
    // `steal_stamp_`, is written with the hint, once per freed bag.
    alignas(128) std::atomic<word_t> steal_hint_{STEAL_HINT_EMPTY};
    word_t steal_stamp_ = 0;   // the last stamp the free pass stored into steal_hint_ (shifted left by one there); owned by the flag holder
    // The cadence's mask from the constructor's `auto_reclaim_every`
    // (THE TRIGGER in the class overview): 0 means OFF, decided before any
    // rounding (std::bit_ceil(0) is 1, which would be a cadence of 1); a
    // value above 2^63 is clamped, silently, to 2^63 before the rounding
    // (bit_ceil() of it has no representable result; 2^63 is "never" on any
    // real run, so SIZE_MAX is a valid spelling of never); then the value is
    // rounded up to a power of two R and the mask is ~(R - 1): the test
    // `((old ^ (old + count)) & mask) != 0` is true iff (old, old + count]
    // holds a multiple of R, and identically false for the mask 0.
    static constexpr size_t auto_reclaim_mask(size_t every) noexcept {
        if (every == 0) return 0;
        if (every > (size_t{1} << 63)) every = size_t{1} << 63;
        return ~(std::bit_ceil(every) - 1);
    } // auto_reclaim_mask()
    // The cadence mask (auto_reclaim_mask() above), read by every retire push
    // and written never: `const`, set in the Options constructor's
    // mem-initializer. A 128-byte block shared only with the
    // constructor-written rows pointer below, so nothing writes it after
    // construction: the retire path reads it on every push, and the block
    // before it is written by the free pass.
    alignas(128) const size_t auto_reclaim_mask_;
    // The per-shard count rows (CountRow; THE RING in the class overview):
    // arena_shards rows, allocated by the constructor before its table_size_
    // release and never moved or freed before the set (a stale join increment
    // must land on live memory). Written once, by the constructor; read by
    // every join (the row of the joining thread's shard), by the free pass
    // (every row) and by the diagnostics. Placed last, so that no member an
    // operation reads is displaced by it (the word fits in the last block's
    // padding: the set's size does not grow), on the line nothing writes after
    // construction -- never in the flag's block, where `deal_` is written once
    // per freed node, and never in Shard, whose size is the stride of shards_[]
    // on every operation. No `mutable`: unique_ptr<T[]>::operator[] is const
    // and yields T&.
    std::unique_ptr<CountRow[]> rows_;
}; // class ConcurrentResizableHashSetRCU

#endif // CONCURRENT_HASH_SET_RCU_H
// Header-private: expanded when push_retired() above was parsed (macros expand
// at definition, not at template instantiation), so it is not exported to
// includers.
#undef CHS_RCU_UNLIKELY
