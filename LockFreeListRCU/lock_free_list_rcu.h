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
#ifndef INCLUDED_LOCK_FREE_LIST_RCU_H
#define INCLUDED_LOCK_FREE_LIST_RCU_H

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <unordered_set>
#include <utility>

// AddressSanitizer poisoning of free-list nodes (see VALUE TYPE and the
// protocol overview: a node's value storage is raw while the node is free,
// and under ASan it is POISONED so that a read through a stale iterator is
// reported). The detection is in the NESTED form on purpose: GCC defines
// __SANITIZE_ADDRESS__ and has no __has_feature, and `#if defined(X) &&
// __has_feature(Y)` on one line is a GCC syntax error. Without ASan the
// macros are no-ops and the poison check reports "poisoned".
#if defined(__SANITIZE_ADDRESS__)
#define LFL_RCU_HAVE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LFL_RCU_HAVE_ASAN 1
#endif
#endif
#ifdef LFL_RCU_HAVE_ASAN
#include <sanitizer/asan_interface.h>
#define LFL_RCU_ASAN_POISON(addr, size) ASAN_POISON_MEMORY_REGION((addr), (size))
#define LFL_RCU_ASAN_UNPOISON(addr, size) ASAN_UNPOISON_MEMORY_REGION((addr), (size))
#else
#define LFL_RCU_ASAN_POISON(addr, size) ((void)(addr), (void)(size))
#define LFL_RCU_ASAN_UNPOISON(addr, size) ((void)(addr), (void)(size))
#endif

// Lock-Free Forward List with generation-based ("RCU-style") reclamation.
//
// This is the Harris-style singly-linked list of LockFreeList/lock_free_list.h
// -- logical deletion by a mark bit in the deleted node's OWN next pointer,
// physical unlinking by swinging the predecessor past it, cooperative helping
// -- with the reference-counted pointers replaced by plain CAS on raw marked
// pointers. What used to be done by the pointer policy (keeping a node alive
// while anyone can still reach it) is done here by GENERATIONS and HANDLES:
//
// HANDLES. Every list operation takes a `handle` as its first argument
// (`list.insert_after(h, anchor, v)`, `list.begin(h)`, ...). A handle is a
// session: `list.new_handle()` joins the list's current generation; the
// handle's destructor leaves it. While a handle is alive, every node it can
// observe -- every node reachable from the head at the time it joined or
// later, every node an iterator obtained under it is parked on, and the whole
// "graveyard" chain such an iterator can still walk out through -- stays
// allocated and keeps its value: no node whose unlink the handle might not
// have seen is ever freed or reused while the handle lives. A handle protects
// for as long as it lives, or until `refresh()` moves it to the current
// generation or a move assignment ends its session, whichever comes first.
// The cost of this protection is that nothing retired at or after the
// handle's generation can be reclaimed while the handle stays put (see
// RECLAIM below).
//
// ITERATOR VALIDITY. An iterator is valid until the handle it was obtained
// under is destroyed, refreshed, or move-assigned over; a MOVED handle keeps
// its iterators valid under the move target. Iteration deliberately does not
// skip logically deleted nodes: an iterator parked on an erased node keeps
// walking the graveyard back into the live suffix (erasure never rewrites a
// node's next pointer while anyone can still read it), exactly as in the
// reference-counted list. Consequence of the validity clause: a reader
// walking a large list cannot release old generations without restarting --
// `refresh()` invalidates its iterators -- so a long traversal pins every
// node retired since it began. `iterator`/`const_iterator` are NOT standard
// iterators: no iterator traits, and range-for is impossible because `begin`
// takes a handle. This is by design: the handle is part of every operation's
// signature so that the protection a traversal relies on is visible at the
// call site.
//
// CONST LISTS. A const list yields handles too (`new_handle()` is const) and
// its `before_begin(h)`/`begin(h)` return `const_iterator`; `const` on the
// list does not stop OTHER threads from erasing and reclaiming through a
// non-const reference, so a const reader still needs a handle. One handle
// type serves both; `iterator` converts implicitly to `const_iterator`, never
// the reverse.
//
// ERASE, RETIRE, RECLAIM. `erase_after(h, anchor)` marks the successor (its
// linearization point) and then tries to unlink it physically. The node is
// RETIRED -- pushed onto the list's retired list -- at the physical unlink,
// by the unique thread whose unlink CAS succeeded (a helping thread
// included), never at the mark. A marked node whose unlink CAS failed stays
// linked ("stranded") until a later `erase_after()` over the same edge helps
// it out; a stranded node is retired then, or destroyed by the list's
// destructor, and until then it is counted as reachable. Retired nodes are
// never returned to the allocator during the list's lifetime: `reclaim()`
// groups them by generation and, once no handle can observe them, DESTROYS
// their values and moves the nodes to a free list from which
// `insert_after()`/`emplace_after()` take nodes before calling `new`. Hence
// a node's `~T` runs inside `reclaim()`, not inside `erase_after()`, and only
// once no handle that could observe it is alive. A node is in exactly one
// of four populations at any time: linked from the head (live,
// marked-but-linked, or stranded), on the retired list, in a sealed
// generation's bag, or on the free list; `get_internal_accounting()` reports
// all four.
//
// `reclaim()` is caller-driven: nothing is reclaimed unless some thread calls
// it. ONE call proceeds at a time: a `reclaim()` that finds another in
// progress returns `ReclaimResult::Advance::contended` immediately, does
// nothing, and never waits. The call that proceeds runs this exact sequence
// (steps R0-R3; the implementation's finer steps A0-A5 are in the protocol
// overview below): (R0) if nothing has been retired since the last advance,
// the result is `nothing_retired` -- checked FIRST, so it is the result even
// when every slot is also in use; (R1) if the next generation slot is in use
// (the ring has wrapped), FREE bags oldest-first as described below, once;
// if the slot is still in use the result is `ring_full` and no slot is
// consumed; (R2) ADVANCE the generation: everything retired so far becomes
// the closing generation's bag and a new generation becomes current -- new
// handles and refreshed handles join it; (R3) the final FREE pass. Whatever
// the result other than `contended`, the final free pass runs. Freeing,
// oldest-first: the oldest sealed bag is freed while its generation's count
// is zero, then the next, and so on until a generation whose count is not
// zero is reached. A generation's count is held ("pinned") by a live handle
// of that generation, or transiently by a concurrent join attempt that has
// incremented it and not yet re-checked which generation is current (such
// an increment is undone or becomes a legitimate membership moments later).
// Freeing is cumulative -- a bag is never freed while an older bag is still
// pinned -- because a handle of generation g can observe nodes that land in
// bags g, g + 1, ... and is protected only if every one of them waits for
// it. Consequences: the bag a call just sealed is freeable in the SAME call
// when no handle of its generation or an older one is alive, otherwise by a
// later call; a call that finds every slot in use but the oldest pin already
// gone frees in step R1 and then advances (with one node per bag, the
// closing generation pinned and no younger sealed generation pinned:
// `{kGenerations - 1, kGenerations - 1, advanced}`; with nothing pinned,
// `{kGenerations, kGenerations, advanced}` -- the closing bag is freed too);
// `nothing_retired` and `ring_full` do not imply that nothing was freed. A
// refusal (`nothing_retired`, `ring_full`, `contended`) costs nothing and is
// retried by calling again: `contended` after the other call returns,
// `ring_full` after the pinning handles are destroyed or refreshed -- there
// is no automatic recovery. A `ring_full` result is stale on return in
// exactly one case: the pin went away between step R1's free pass and step
// R3's (a handle released, or a joiner's transient increment undone); step
// R3 then frees it and the NEXT call advances (a pin released before the
// call never yields `ring_full`). A thread that HOLDS a handle and calls
// `reclaim()` frees nothing at or after that handle's generation, because
// its own handle pins it; the recommended sequence for a thread that both
// operates on the list and reclaims is `h.refresh(); list.reclaim();`.
//
// PROGRESS. List operations, the handle join (`new_handle()`, `refresh()`),
// the retire push, the free-list pop and the bag splice are lock-free.
// `reclaim()` is a try-lock: non-blocking by refusal -- a concurrent caller
// gets `contended` and never waits. A reclaimer thread stalled (descheduled,
// suspended) inside `reclaim()` stalls ALL reclamation until it resumes --
// every other call returns `contended` and memory grows -- but never a list
// operation; this is the same consequence class as a thread stalled while
// holding a handle, which also stops reclamation of its generation and
// younger ones.
//
// VALUE TYPE. A node's value storage is raw memory while the node is on the
// free list; `insert_after()` constructs the value by move, `emplace_after()`
// by the constructor matching its arguments, through one placement-new path
// whether the node is fresh or reused; no default construction and no
// assignment of T anywhere. T must be nothrow-destructible (`reclaim()` runs
// `~T` and is `noexcept`), nothrow-move-constructible for `insert_after()`,
// and nothrow-constructible from the arguments for `emplace_after()` --
// enforced by `static_assert`s, so that constructing into a reused node can
// never throw. Note that `std::is_nothrow_constructible_v<std::string,
// const char*>` is false: `emplace_after(h, a, "x")` does not compile; when
// the constructor may throw, construct a prvalue and use `insert_after`.
//
// THREAD SAFETY, standard-library style. Any number of threads may run any
// list operations concurrently, each under its own handle or sharing one:
// concurrent const operations on ONE handle (passing it to list operations)
// are fine; `refresh()`, move and destruction of a handle are exclusive with
// every other use of that handle. A handle may be transferred to another
// thread after a happens-before edge (e.g., created on one thread, destroyed
// on another after a join); the debug checks record no thread id. The list's
// destructor is exclusive with everything and requires that no handle is
// alive and no `reclaim()` is in progress.
//
// DIAGNOSED PRECONDITION VIOLATIONS. Debug builds (NDEBUG not defined)
// diagnose exactly these five by assertion: a handle of another list passed
// to a list operation; an iterator used with the wrong handle in a list
// operation (an iterator is bound to the handle session it was obtained
// under: a refreshed or move-assigned-over handle starts a new session, a
// moved handle keeps the session); `*` or `->` on `before_begin()`'s
// iterator; `refresh()` or a list operation on an empty handle; destruction
// of the list with live handles. The handle checks (empty, another list)
// run before the anchor is looked at, so they fire even when the anchor is
// `end()`; the iterator-session check runs for a non-`end()` anchor only (an
// `end()` iterator is bound to no session), also before the anchor node is
// touched. Other violations (`++` on `end()`, use of an invalidated
// iterator) are undefined behavior that debug builds MAY catch but the
// contract does not promise to.
//
// TEST SEAMS. The `Hooks` template parameter (default `NoHooks`, constrained
// by `ListHooks`) names four `noexcept` static functions the list calls at
// fixed points of its protocol; the `get_internal_*` members are test-only
// accessors with a quiescence precondition. Neither is part of the list's
// operational contract; both are documented at their declarations.
//
// FOOTPRINT. 8832 bytes (about 8.6 KiB) for `LockFreeListRCU<int>` at the
// default kGenerations = 64, in release and debug builds alike: 64
// generation blocks of 128 bytes each (a pair of 64-byte lines on x86, so
// adjacent reference counts share neither a line nor an adjacent-line
// prefetch) plus five private 128-byte lines. Nodes are 24 bytes for `int`
// (value, next, retire link) in release builds -- the size of the
// reference-counted list's node under its intrusive-pointer policy (the
// retire link takes the slot of that node's count word).
//
// ---------------------------------------------------------------------------
// PROTOCOL OVERVIEW (the implementation; the public contract is above). The
// memory order of every atomic operation below is justified in the comment
// at that operation, with one of three labels. DESIGNATED: a relacy model
// checker run of the protocol core kills the weaker order (the comment says
// what the weakening let happen). EXPECTED-EQUIVALENT: a weaker (or a
// different, equally strong) order would also be correct under the standard
// and the model checker cannot tell them apart; the chosen order is kept so
// that nobody "fixes" it either way. REQUIRED: the standard's happens-before
// argument needs the order; the comment gives it.
//
// THE RING. Generations are status blocks in a fixed ring of kGenerations
// `GenerationBlock`s that are never deallocated or moved. Each block has a
// live-handle count `refs` (atomic), and three PLAIN fields owned by the
// reclaimer: `bag` (the retired nodes of this generation once it is sealed),
// `state` (EMPTY, CURRENT, SEALED) and `gen` (the generation number the slot
// currently represents). `current_` points at the CURRENT block. Behind it
// sits a contiguous run of SEALED blocks whose numbers decrease by one per
// slot going back, down to the oldest at index `oldest_`; every other slot
// is EMPTY. The ring is fixed because a joiner that has loaded `current_`
// but not yet incremented its count may increment a block that was closed
// in between: the block must still exist for that stale increment to land
// somewhere harmless.
//
// THE JOIN (`join()`, from `new_handle()` and `refresh()`'s slow path): load
// `current_` -> B; `B->refs.fetch_add(1, seq_cst)`; re-verify with a seq_cst
// load that `current_` is still B (pointer equality); if not, undo with
// `fetch_sub` and retry. The join and the reclaimer's "publish, then read
// refs" form a store-buffering pair, which is why the increment, the
// re-verify load, the publish store and the reclaimer's `refs` loads are all
// seq_cst (model-checked: with any of the four weakened, or without the
// re-verify, a joiner ends up reading a node in a bag the reclaimer is
// freeing). Two invariants make the stale increment harmless: (J1) between
// its `fetch_add` and a successful re-verify a joiner touches NO node, head
// or bag, so a count it leaves on a just-closed or recycled block can delay
// that block's freeing but can never be the count that protects a
// dereference; (J2) `refs` is modified ONLY by RMWs -- +1 by joiners, -1 by
// leave and by the join's undo -- and is never stored: the reclaimer only
// reads it, and does NOT reset it when it recycles a slot. A reset would
// wipe a stale +1 and the matching -1 would then drive the count negative
// or to zero under a live handle (model-checked: a reclaimer that stores 0
// on recycle lets a later handle hold a block whose count is 0 while its bag
// is freed). Hence every -1 follows its own +1 and the count never goes
// negative (debug-asserted at every `fetch_sub`). Pointer equality suffices
// across a ring wrap: a joiner whose re-verify reads block B again after B
// was freed, recycled and republished has synchronized with that LATER
// publish and, by J1, looked at nothing before -- its join is legitimate for
// the new generation.
//
// ONE RECLAIMER. `reclaim()` runs under the global try-flag `reclaiming_`:
// `exchange(true, acquire)` at entry (a caller that finds it set returns
// `{0, 0, contended}` at once) and `store(false, release)` at exit. Inside,
// sequentially (steps A0-A5 refine the contract's R0-R2; the final free pass
// is R3): `advance()` -- (A0) nothing on the retired list: return
// `nothing_retired`; (A1) C = current block, `next` = the slot after it;
// (A2) if `next` is not EMPTY run the free pass once and re-check, still not
// EMPTY: `ring_full`; (A3) `chain = retired_head_.exchange(nullptr, acquire)`
// -- the retired list becomes C's bag, BEFORE the publish, so that a retire
// push landing after the exchange goes to the next generation (conservative
// by construction; this is what makes the bag form sound without ever
// reading a generation number at retire time); (A4) plain writes `C->bag =
// chain; C->state = SEALED; next->state = CURRENT; next->gen = C->gen + 1`;
// (A5) publish `current_.store(next, seq_cst)` -- then `free_oldest()` (R3):
// while `ring_[oldest_]` is SEALED and its `refs` (seq_cst load) is 0,
// destroy its bag's values, poison their storage (ASan), splice the bag onto
// the free list, mark the slot EMPTY (its `refs` untouched, J2) and advance
// `oldest_`. Bags are freed strictly oldest-first; a block is never skipped;
// a slot is reusable only when EMPTY. The A3-before-A5 order: an exchange
// moved to after the `in_advance_after_publish` hook is caught by the scheme
// tests; one moved to between the publish and that hook has no deterministic
// test (model checker and order review only; stress tests under
// ThreadSanitizer catch it sometimes). The flag's acquire/release pair has
// two duties: it is the ONLY synchronization on the plain fields `state`,
// `bag`, `gen` and `oldest_` between successive reclaimers, and it orders a
// holder's publish before a later holder's `refs` loads (so the
// store-buffering argument holds across flag holders, not only inside one
// call); model-checked: with relaxed flag orders, or without the flag, two
// callers race on the plain fields at once.
// Concurrent reclaimers were rejected after model checking demonstrated the
// hazards of every variant tried -- two advancers writing one closing bag,
// an advance order that wedges the ring after one wrap, a stale snapshot of
// the current block orphaning a retired chain, a free walk passing a block
// still current, and a skipped in-progress block with no happens-before to
// its generation's leave -- and their repairs bought concurrency nobody
// needs at the price of a state machine. Refusal and retry: `contended`,
// `ring_full` and `nothing_retired` consume no slot and burn nothing; the
// caller simply calls again (`ring_full` after the pinning handles leave or
// refresh -- and it is stale on return exactly when the pin was released
// between the in-advance free pass and the final one: the final pass then
// frees it and the next call advances).
//
// INVARIANT. Two directions. (I) A handle of generation g never observes a
// node in a bag of generation < g. Chain: unlink CAS, sequenced before the
// unlinker's retire push (release RMW) -> the reclaimer's exchange (acquire)
// -> the publish (seq_cst store, sequenced after the exchange in the same
// flag holder or after an earlier holder's exchange through the flag's
// release/acquire) -> the joiner's seq_cst re-verify that read it. A handle
// observes a dead node only through the word it was unlinked from or through
// a dead predecessor's frozen `next` whose own unlink happens-before the
// successor's, so a handle whose join follows the publish of generation
// g cannot reach anything bagged below g. (II) Contrapositive: a node in bag
// G is observable only by handles of generation <= G. This is what the
// cumulative rule needs -- a handle of g observes nodes whose unlink does
// not happen-before its join, and those land in bags g, g + 1, ...; freeing
// bag G requires `refs == 0` for every generation <= G, which oldest-first
// freeing from `oldest_` delivers by construction. Together: every node
// address a handle of generation g can observe -- in the list, through a
// graveyard chain, or on the free list -- is either not retired or in a bag
// of generation >= g, and that bag outlives the handle. The same invariant
// excludes ABA on the unlink CAS (the eraser's own handle keeps the expected
// node from being freed and reused) and on the free-list pop (the popper's
// handle keeps a popped-then-retired node from returning to the free list).
//
// DISPOSAL. A node popped from the free list that does not get linked
// (the anchor turned out to be marked) is DISPOSED: its value is NOT
// destroyed (the bag free does that later), a marked value is stored into
// its `next`, and it is pushed onto the retired list, from which it re-enters
// the free list only through a bag. It is never pushed back onto the free
// list -- model-checked: a stale pop CAS then succeeds by ABA and the node
// is handed out twice -- and never deleted, because a losing popper may
// still read its `retire_link`. A freshly `new`ed node that does not get
// linked is deleted after `~T`.
//
// DEBUG ASSERTS (all compiled out under NDEBUG), with their messages: the
// handle belongs to this list ("handle of another list") and is live ("empty
// handle") on every operation taking a handle; the iterator's session id
// equals the handle's ("iterator used with a different handle session") on
// every operation taking (handle, iterator), checked before the anchor node
// is touched; the node incarnation in `*`, `->`, `++` ("iterator node was
// freed and reused"); `++` on end() ("increment of end"); `*`/`->` on end()
// ("dereference of end") and on before_begin()'s iterator ("dereference of
// before_begin"); `get_internal_incarnation(end())` ("incarnation of end");
// every `fetch_sub` on `refs` returned > 0 ("generation refs went
// negative"); the retire push, the disposal and the bag free see `next`
// marked ("retired node is not marked"); inside reclaim(), the closing block
// is CURRENT before the publish ("closing block is not CURRENT"), and two
// STRUCTURAL asserts that document a precondition the preceding check has
// already established and therefore cannot fire: the next slot is EMPTY
// before the publish ("next block is not EMPTY", after A2's re-check) and
// the block being freed is SEALED ("freeing a block that is not SEALED",
// after free_oldest()'s own state test); the destructor finds no reclaim in
// progress ("list destroyed during reclaim"), every count at zero ("list
// destroyed with live handles") and every node accounted for ("node leak at
// destruction"); the accounting sweep finds no reclaim in progress
// ("accounting sweep during reclaim").
// ---------------------------------------------------------------------------

// Default hooks: every hook is an empty, inlined `noexcept` static function,
// so the list compiles to the hook-free code. Tests substitute a type with
// the same four functions to pause or re-enter the list at the points
// below. Hooks take no arguments (an argument would expose a private type);
// list operations are re-entrant from inside a hook (a hook may call
// `insert_after`, `erase_after`, `reclaim`, `new_handle` on the same list
// under the same or another handle, but must not refresh, move or destroy
// the handle the interrupted operation uses, nor call
// `get_internal_accounting()`). The list calls a hook from no function other
// than the one named at each hook; never from the destructor or the sweep.
struct NoHooks {
    // `erase_after()`: on EVERY pass of its retry loop that reaches the
    // unlink CAS -- after the marking race is decided (this call set the
    // mark, or found the successor already marked) and before the unlink
    // CAS; helping passes included. Not on a pass that returns false
    // because the anchor is end() or deleted, or has no successor.
    static void between_mark_and_unlink() noexcept {}
    // The join: on EVERY join attempt -- `new_handle()`'s first attempt and
    // each retry, and `refresh()`'s slow path (never its fast path, nor
    // moves or handle destruction) -- after the attempt read the current
    // generation and before it counts the handle in it.
    static void in_join() noexcept {}
    // The join: on EVERY join attempt (as `in_join`), after the attempt
    // counted the handle in the generation it read and before it re-checks
    // that this generation is still current (a check that fails undoes the
    // count and retries).
    static void in_join_after_bump() noexcept {}
    // `reclaim()`: once per call that returns `advanced`, after the new
    // current generation is published and before the free pass; the reclaim
    // flag is held, so a nested `reclaim()` from inside this hook returns
    // `contended`. Not on `nothing_retired`, `ring_full` or `contended`
    // calls.
    static void in_advance_after_publish() noexcept {}
}; // NoHooks

// A Hooks type provides the four static functions `NoHooks` documents, each
// callable with no arguments, returning void and `noexcept`: they are called
// from `noexcept` list members (`new_handle`, `refresh`, `erase_after`,
// `reclaim`).
template <typename H>
concept ListHooks = requires {
    { H::between_mark_and_unlink() } noexcept -> std::same_as<void>;
    { H::in_join() } noexcept -> std::same_as<void>;
    { H::in_join_after_bump() } noexcept -> std::same_as<void>;
    { H::in_advance_after_publish() } noexcept -> std::same_as<void>;
};

// What one `reclaim()` call did. Not a template: equal across every list
// instantiation, so tests typed over several lists compare against one
// literal and one `PrintTo` overload serves them all.
struct ReclaimResult {
    // Whether and how the call advanced the generation.
    enum class Advance {
        nothing_retired, // nothing was retired since the last advance; no generation slot consumed; checked first, so it wins over ring_full
        advanced,        // the retired nodes became the closing generation's bag; a new generation is current (possibly after freeing bags to make room)
        ring_full,       // every generation slot is still in use after the in-call free pass (the oldest bag is pinned by a handle, or transiently by a concurrent join attempt); no slot consumed
        contended        // another reclaim() is in progress; this call did nothing and returned at once
    };
    size_t freed_nodes;  // nodes whose value was destroyed and which moved to the free list in this call
    size_t freed_bags;   // generations whose bags were freed in this call (oldest first)
    Advance advance;     // see above; `nothing_retired` and `ring_full` do NOT imply freed_nodes == 0 -- the free pass still runs; `contended` implies {0, 0}
    bool operator==(const ReclaimResult&) const = default;
}; // ReclaimResult

// LockFreeListRCU<T, kGenerations, Hooks>: see the overview above.
//   T            the value type (nothrow-destructible; see VALUE TYPE)
//   kGenerations the number of generation slots (>= 2); one slot is always
//                the current generation, so `reclaim()` can perform
//                kGenerations - 1 advances while the oldest generation stays
//                pinned by a handle before it returns `ring_full`
//   Hooks        test seams (see NoHooks)
template <typename T, size_t kGenerations = 64, ListHooks Hooks = NoHooks>
class LockFreeListRCU {
    static_assert(kGenerations >= 2, "LockFreeListRCU needs at least two generation slots: one current, one closing");
    static_assert(std::is_nothrow_destructible_v<T>, "LockFreeListRCU<T>: reclaim() destroys values and is noexcept, so ~T must not throw");

private:
    struct Node;

    // A Node* with the deletion mark in bit 0, as a trivially copyable 8-byte
    // value type (so `std::atomic<marked_ptr>` is lock-free and the Harris
    // loops read like the reference-counted list's). The mark is part of the
    // value's identity: a marked and an unmarked pointer to the same node
    // compare unequal, which is what makes marking a node fail every CAS
    // that expected the unmarked value. Node is at least 8-byte aligned, so
    // bit 0 is free.
    class marked_ptr {
    public:
        marked_ptr() noexcept = default;                                         // null, unmarked
        explicit marked_ptr(Node* p) noexcept : bits_(reinterpret_cast<uintptr_t>(p)) {} // unmarked pointer to `p`
        bool is_marked() const noexcept { return (bits_ & kMark) != 0; }
        Node* get_unmarked() const noexcept { return reinterpret_cast<Node*>(bits_ & ~kMark); } // the pointer without the mark
        marked_ptr set_mark() const noexcept { marked_ptr r; r.bits_ = bits_ | kMark; return r; } // the same pointer, marked
        marked_ptr clear_mark() const noexcept { marked_ptr r; r.bits_ = bits_ & ~kMark; return r; } // the same pointer, unmarked (the unlink value)
        bool operator==(const marked_ptr&) const noexcept = default;             // bitwise: mark included
    private:
        static constexpr uintptr_t kMark = 1;  // the deletion mark, bit 0
        uintptr_t bits_ = 0;                   // the pointer bits with the mark in bit 0
    }; // marked_ptr
    static_assert(std::atomic<marked_ptr>::is_always_lock_free, "marked_ptr must be a lock-free atomic");

    // A list node. `value` is raw, suitably aligned storage, never a T
    // member: it holds a T from the node's (re)initialization in
    // `insert_after()`/`emplace_after()` until the node's bag is freed in
    // `reclaim()` (or the destructor runs), and is raw bytes while the node
    // is on the free list. Accessed through `std::launder` on the storage
    // (`value_of()`). `value` comes FIRST so that `&*it` is the node's
    // identity. The debug-only `incarnation` sits AFTER `retire_link`,
    // outside the extent poisoned under ASan (`[value, retire_link)`), so a
    // parked iterator can read it on a node that is on the free list and
    // report the reuse instead of tripping the sanitizer.
    struct Node {
        alignas(T) std::byte value[sizeof(T)];        // storage for the T (see above)
        std::atomic<marked_ptr> next{};                // the list link; a marked value = THIS node is logically deleted; frozen once marked until the node is reused
        std::atomic<Node*> retire_link{nullptr};       // the second link: retired-list, bag and free-list chaining -- never `next`, so parked iterators still walk the graveyard
#ifndef NDEBUG
        std::atomic<uint32_t> incarnation{0};          // debug only: bumped on every free-list pop; iterators record it to detect use across a reuse
#endif
    }; // Node

    // Lifecycle of a generation slot in the ring.
    enum class State { EMPTY, CURRENT, SEALED };

    // One generation: a status block in the fixed ring (never deallocated).
    // 128 bytes per block (a pair of 64-byte lines on x86, so the adjacent-line
    // prefetcher does not couple neighbours): adjacent `refs` would otherwise
    // share a line.
    // `bag`, `state` and `gen` are PLAIN fields written and read only by the
    // holder of `reclaiming_`, by the destructor and by the quiescent
    // accounting sweep; the flag's acquire/release is their only
    // synchronization. Joiners read only `current_` and `refs`, so no join
    // path reads `gen` -- if a `gen` read is ever added to the debug session
    // key, it must come only AFTER a successful re-verify (the publish then
    // happens-before it); earlier it races with the recycle write.
    struct alignas(128) GenerationBlock {
        std::atomic<long> refs{0};               // live handles joined to this generation; RMW-only (J2): never stored after construction
        Node* bag = nullptr;                     // the retired nodes of this generation, once SEALED (owned by the reclaimer)
        State state = State::EMPTY;              // EMPTY / CURRENT / SEALED (owned by the reclaimer)
        uint64_t gen = 0;                        // the generation number this slot currently represents (owned by the reclaimer)
        const LockFreeListRCU* list = nullptr;   // the owning list, set once in the constructor: how a handle (one pointer in release builds) finds its list for refresh() and leave
    }; // GenerationBlock

public:
    // A handle is a session with one generation of the list: it keeps every
    // node it can observe allocated (see HANDLES in the overview). Move-only,
    // shared_ptr-like: the move constructor transfers the session (the
    // generation's count is unchanged; iterators obtained under the source
    // stay valid under the target); move assignment first ENDS the target's
    // session if it has one (its iterators become invalid), then transfers;
    // the moved-from handle is empty. A default-constructed handle is EMPTY.
    // On an empty handle the only valid operations are destruction, moving
    // from and to it, `operator bool` and `swap`; `refresh()` or any list
    // operation with an empty handle is a precondition violation (debug-
    // asserted). An empty handle may outlive its list; a live one may not.
    // In release builds a handle is one pointer. Thread safety: see the
    // overview (concurrent const use is fine; refresh/move/destroy are
    // exclusive).
    class handle {
    public:
        // Postcondition: empty. nothrow.
        handle() noexcept = default;
        // Transfers `other`'s session; `other` is left empty. Iterators
        // obtained under `other` are valid under *this. nothrow.
        handle(handle&& other) noexcept : block_(other.block_) {
#ifndef NDEBUG
            session_ = other.session_;
#endif
            other.block_ = nullptr;
        } // handle(handle&&)
        // Ends this handle's session if it has one (iterators obtained under
        // *this become invalid), then transfers `other`'s; `other` is left
        // empty. Self-move-assignment is a no-op. nothrow.
        handle& operator=(handle&& other) noexcept {
            if (this == &other) return *this;
            if (block_) block_->list->leave(block_);
            block_ = other.block_;
#ifndef NDEBUG
            session_ = other.session_;
#endif
            other.block_ = nullptr;
            return *this;
        } // handle::operator=(handle&&)
        handle(const handle&) = delete;
        handle& operator=(const handle&) = delete;
        // Ends the session (leaves the generation) if live; iterators
        // obtained under *this become invalid. Precondition: if live, the
        // list is still alive. nothrow.
        ~handle() noexcept {
            if (block_) block_->list->leave(block_);
        }

        // True iff this handle is live (holds a session). nothrow.
        explicit operator bool() const noexcept { return block_ != nullptr; }

        // Re-join the list's current generation. Precondition: live (debug-
        // asserted). Postcondition: the handle is a member of a generation
        // that was current at some point not happening-before the call's
        // start (the fast path may keep a generation that another thread has
        // just closed -- safe, and merely conservative: it keeps that
        // generation's bags pinned until the next refresh); EVERY iterator
        // obtained under this handle before the call is invalid, whatever
        // the result. Returns true iff the handle actually moved to a newer
        // generation (false: it was already current, nothing changed except
        // the invalidation). Releases the old generation for reclamation
        // once no other handle pins it. Lock-free. nothrow.
        //
        // The session id is redrawn on EVERY call, fast path included, so
        // that the invalidation is diagnosable whatever the result. The
        // fast-path load is relaxed: it changes no membership, so it has
        // nothing to synchronize with. The only stale answer it can give is
        // "same" (coherence: it cannot return a value older than this
        // handle's own earlier read of `current_`, and `block_` cannot be
        // republished while this handle pins it), which keeps the handle in
        // a just-closed generation -- safe and conservative (see the
        // postcondition). Slow path: join the current block first, then
        // leave the old one (the order is free; nothing relies on it).
        bool refresh() noexcept {
            assert(block_ != nullptr && "empty handle");
#ifndef NDEBUG
            session_ = new_session_id();
#endif
            const LockFreeListRCU* list = block_->list;
            // relaxed: see above (EXPECTED-EQUIVALENT to acquire).
            if (list->current_.load(std::memory_order_relaxed) == block_) return false;
            GenerationBlock* joined = list->join();
            list->leave(block_);
            block_ = joined;
            return true;
        } // handle::refresh()

        // Exchanges the sessions (and the iterators each validates) of two
        // handles; either or both may be empty. nothrow.
        void swap(handle& other) noexcept {
            std::swap(block_, other.block_);
#ifndef NDEBUG
            std::swap(session_, other.session_);
#endif
        } // handle::swap()
        friend void swap(handle& a, handle& b) noexcept { a.swap(b); }

    private:
        friend class LockFreeListRCU;
        GenerationBlock* block_ = nullptr;  // the generation this handle is joined to; nullptr when empty. The owning list is `block_->list`
#ifndef NDEBUG
        uint64_t session_ = 0;              // debug only: globally unique id of this session; a new one on every join/refresh; moves with the handle. 0 is never a live id
#endif
    }; // handle

    // Iterator over the list, `kConst` selecting `const T&` or `T&` access.
    // Holds a bare node pointer: what keeps the node alive is the handle the
    // iterator was obtained under (see ITERATOR VALIDITY in the overview).
    // A default-constructed iterator equals `end()`. Equality compares node
    // identity only (an iterator that walked the graveyard into a live node
    // equals one that reached it through the live chain). An INVALIDATED
    // iterator may be destroyed, copied, assigned and compared, nothing
    // else. Not a standard iterator (no traits; see the overview).
    template <bool kConst>
    class basic_iterator {
    public:
        using value_type = T;
        using reference = std::conditional_t<kConst, const T&, T&>;
        using pointer = std::conditional_t<kConst, const T*, T*>;

        // Postcondition: *this == end(). nothrow.
        basic_iterator() noexcept = default;
        basic_iterator(const basic_iterator&) noexcept = default;
        basic_iterator& operator=(const basic_iterator&) noexcept = default;
        // iterator -> const_iterator, implicit; the converse does not exist.
        // The converted iterator is bound to the same handle session as the
        // original (it is the same iterator, read-only).
        template <bool kOtherConst> requires (kConst && !kOtherConst)
        basic_iterator(const basic_iterator<kOtherConst>& other) noexcept : curr_(other.curr_) {
#ifndef NDEBUG
            owner_ = other.owner_;
            session_ = other.session_;
            incarnation_ = other.incarnation_;
#endif
        } // basic_iterator(const basic_iterator<kOtherConst>&)

        // Access the value. Preconditions: valid, != end(), and not the
        // iterator `before_begin()` returned (debug-asserted: the head holds
        // no value). The node may be logically deleted (graveyard) -- its
        // value is intact while the iterator is valid. nothrow.
        reference operator*() const noexcept {
            check_deref();
            return *value_of(curr_);
        }
        pointer operator->() const noexcept {
            check_deref();
            return value_of(curr_);
        }

        // Advance to the successor, live or deleted (iteration does not skip
        // logically deleted nodes); past the last node yields end().
        // Precondition: valid and != end() (`++` on end() is a precondition
        // violation; debug builds may assert). nothrow.
        //
        // The mark is stripped: the mark is part of a pointer's identity, so
        // iterators always hold canonical unmarked values -- otherwise an
        // iterator that walked out of a deleted node would never compare
        // equal to one that reached the same node through the live chain. A
        // marked null (the last node was erased) becomes a plain end().
        // Whether the next node ITSELF is deleted is not visible here (its
        // mark lives in its own next pointer), so ++ walks into deleted nodes
        // by design.
        basic_iterator& operator++() noexcept {
            assert(curr_ != nullptr && "increment of end");
            assert(curr_->incarnation.load(std::memory_order_relaxed) == incarnation_ && "iterator node was freed and reused");
            // acquire: pairs with the release CAS that linked the successor
            // (insert) or swung this word past a dead node (unlink); we are
            // about to dereference the successor's value and next.
            curr_ = curr_->next.load(std::memory_order_acquire).get_unmarked();
#ifndef NDEBUG
            if (curr_) incarnation_ = curr_->incarnation.load(std::memory_order_relaxed);
#endif
            return *this;
        } // basic_iterator::operator++()

        // Node identity only; `end()` equals `end()` and a default-constructed
        // iterator across handles and lists, in both operand orders;
        // `iterator` and `const_iterator` compare through the conversion.
        // `!=` is rewritten from this. nothrow.
        friend bool operator==(const basic_iterator& a, const basic_iterator& b) noexcept { return a.curr_ == b.curr_; }

    private:
        friend class LockFreeListRCU;
        template <bool> friend class basic_iterator;
        // The debug preconditions of `*` and `->`: not end(), not the head,
        // and the node has not been reused since the iterator was made. The
        // incarnation lives outside the poisoned extent, so this reads
        // cleanly even on a free-list node (a contract violation we then
        // report by assertion rather than by sanitizer).
        void check_deref() const noexcept {
            assert(curr_ != nullptr && "dereference of end");
            assert(curr_ != &owner_->head_ && "dereference of before_begin");
            assert(curr_->incarnation.load(std::memory_order_relaxed) == incarnation_ && "iterator node was freed and reused");
        } // basic_iterator::check_deref()
        Node* curr_ = nullptr;  // the node this iterator is on; nullptr for end()
#ifndef NDEBUG
        const LockFreeListRCU* owner_ = nullptr;  // debug only: the list (a null iterator carries no binding); only used to recognize the head
        uint64_t session_ = 0;                    // debug only: the handle session this iterator was obtained under
        uint32_t incarnation_ = 0;                // debug only: `curr_->incarnation` when obtained; `*`, `->`, `++` compare it
#endif
    }; // basic_iterator
    using iterator = basic_iterator<false>;
    using const_iterator = basic_iterator<true>;

    // Test-only accounting sweep result: where every node is. Compiled in
    // release builds too (the benchmark reads the populations after its stop
    // barrier); the debug-only fields exist only when NDEBUG is not defined.
    // ODR NOTE: the layout of this struct, and the set of members of the
    // list, differ between NDEBUG and non-NDEBUG translation units; every TU
    // of one program that includes this header must agree on NDEBUG.
    struct InternalAccounting {
        size_t reachable;       // nodes linked from the head, excluding the head: live, marked-but-linked and stranded alike
        size_t reachable_dead;  // the logically deleted (marked) subset of `reachable`
        size_t retired;         // nodes on the retired list: unlinked, not yet assigned to a generation's bag
        size_t bagged;          // nodes in the bags of sealed generations
        size_t free;            // nodes on the free list (value storage raw)
        long handle_refs;       // the sum of every generation's live-handle count (equals the number of live handles at quiescence)
#ifndef NDEBUG
        size_t allocated;       // debug only: nodes obtained from `new` and not yet deleted (the head is not counted)
        // debug only: the conjunction of -- no node address seen twice across
        // the populations (nor within one walk); reachable + retired + bagged
        // + free == allocated; the ring's state invariant (one current
        // generation, a contiguous run of sealed generations behind it with
        // consecutive numbers, the rest empty) and no reclaim() in progress;
        // (under ASan) every free-list node is poisoned; a sealed generation
        // has a non-empty bag and a current or empty one has none; every
        // generation's handle count is >= 0 and an empty generation's is 0;
        // every retired or bagged node's `next` is marked.
        bool consistent;
#endif
    }; // InternalAccounting

    // Rule of five, decided: non-copyable and non-movable -- handles,
    // iterators and the embedded head node point into this object.

    // An empty list. Allocates nothing (the head is an embedded member).
    // nothrow. Initial ring: slot 0 CURRENT with generation 0, every other
    // slot EMPTY, `oldest_` = 0 (the index of the oldest SEALED block, or of
    // the CURRENT block when none is SEALED).
    LockFreeListRCU() noexcept : current_(&ring_[0]), reclaiming_(false), oldest_(0), retired_head_(nullptr), free_head_(nullptr) {
        for (GenerationBlock& block : ring_) block.list = this;
        ring_[0].state = State::CURRENT;
    } // LockFreeListRCU()

    // Destroys every value still constructed and frees every node in all
    // four populations. Preconditions (debug-asserted): no handle of this
    // list is alive; no `reclaim()` is in progress; no concurrent operation.
    //
    // Every node is in exactly one population, so each is deleted once: the
    // chain from the head (live, marked-but-linked, stranded), the retired
    // list, every SEALED block's bag, the free list. Values are destroyed in
    // the first three (constructed) populations, not on the free list (raw
    // storage, unpoisoned before delete). All loads are relaxed: the
    // destructor is exclusive with everything by precondition.
    ~LockFreeListRCU() {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "list destroyed during reclaim");
        for (const GenerationBlock& block : ring_) {
            assert(block.refs.load(std::memory_order_relaxed) == 0 && "list destroyed with live handles");
            (void)block;
        }
        for (Node* p = head_.next.load(std::memory_order_relaxed).get_unmarked(); p;) {
            Node* next = p->next.load(std::memory_order_relaxed).get_unmarked();
            value_of(p)->~T();
            delete_node(p);
            p = next;
        } // the chain from the head
        for (Node* p = retired_head_.load(std::memory_order_relaxed); p;) {
            Node* next = p->retire_link.load(std::memory_order_relaxed);
            value_of(p)->~T();
            delete_node(p);
            p = next;
        } // the retired list
        for (GenerationBlock& block : ring_) {
            if (block.state != State::SEALED) continue;
            for (Node* p = block.bag; p;) {
                Node* next = p->retire_link.load(std::memory_order_relaxed);
                value_of(p)->~T();
                delete_node(p);
                p = next;
            } // this block's bag
        } // the sealed bags
        for (Node* p = free_head_.load(std::memory_order_relaxed); p;) {
            Node* next = p->retire_link.load(std::memory_order_relaxed);
            LFL_RCU_ASAN_UNPOISON(p, poison_extent(p));
            delete_node(p);
            p = next;
        } // the free list
        assert(allocated_.load(std::memory_order_relaxed) == 0 && "node leak at destruction");
    } // ~LockFreeListRCU()
    LockFreeListRCU(const LockFreeListRCU&) = delete;
    LockFreeListRCU& operator=(const LockFreeListRCU&) = delete;

    // Join the current generation and return a live handle for it. Const: a
    // const list yields handles too (its operations then return
    // `const_iterator`). Postcondition: the handle is live and belongs to
    // this list. Lock-free. nothrow.
    [[nodiscard]] handle new_handle() const noexcept {
        handle h;
        h.block_ = join();
#ifndef NDEBUG
        h.session_ = new_session_id();
#endif
        return h;
    } // new_handle()

    // The anchor before the first element: the list's head, which is never
    // erased, so it is always a valid anchor for `insert_after()` and
    // `erase_after()`. Its `*`/`->` are a precondition violation (diagnosed).
    // Precondition: `h` is a live handle of this list (diagnosed). The
    // iterator is bound to `h`'s current session. nothrow.
    iterator before_begin(const handle& h) noexcept { check_handle(h); return make_iterator<false>(&head_, h); }
    const_iterator before_begin(const handle& h) const noexcept { check_handle(h); return make_iterator<true>(&head_, h); }

    // The first node (live or logically deleted -- see ITERATOR VALIDITY),
    // or `end()` if there is none. Precondition: `h` is a live handle of
    // this list (diagnosed). The iterator is bound to `h`'s session. nothrow.
    iterator begin(const handle& h) noexcept { iterator it = before_begin(h); ++it; return it; }
    const_iterator begin(const handle& h) const noexcept { const_iterator it = before_begin(h); ++it; return it; }

    // The past-the-end iterator. Static: it needs no handle and compares
    // equal to every past-the-end and default-constructed iterator of every
    // list, in both operand orders. `cend()` is the `const_iterator` form;
    // `end()` converts to it implicitly. nothrow.
    static iterator end() noexcept { return iterator(); }
    static const_iterator cend() noexcept { return const_iterator(); }

    // Insert a node holding `value` immediately after `anchor`. Returns true
    // on success; false if `anchor` is `end()` or logically deleted (linking
    // onto a deleted node would strand the new node). Preconditions: `h` is
    // a live handle of this list and `anchor` was obtained under `h`'s
    // current session (both diagnosed). `value` is consumed in every case.
    // Whenever a non-`end()` anchor is given, a node is obtained (from the
    // free list, else `new`) and a T is move-constructed into it from `value`
    // BEFORE the anchor is examined; if the anchor turns out to be deleted,
    // the call returns false and that T is destroyed -- immediately, and the
    // node deleted, for a freshly allocated node (nothing is retired); at a
    // later `reclaim()` for a node reused from the free list, which goes to
    // the retired list (it is not an erase: no `erase_after()` call returns
    // true for it) and is counted by that `reclaim()` in `freed_nodes`.
    // static_assert: `std::is_nothrow_move_constructible_v<T>`.
    // Exceptions: may throw only `std::bad_alloc`, from `new` when the free
    // list is empty -- including on a deleted anchor that would otherwise
    // return false (an `end()` anchor returns false before any allocation).
    // On a throw the list is unchanged (the caller's argument was already
    // moved from at the call site). Lock-free apart from `new`.
    [[nodiscard]] bool insert_after(const handle& h, const_iterator anchor, T value) {
        static_assert(std::is_nothrow_move_constructible_v<T>, "LockFreeListRCU::insert_after: T's move constructor must be noexcept (it runs into a reused node)");
        return init_node(h, anchor, std::move(value));
    } // insert_after()

    // As `insert_after`, constructing the value in place from `args...`.
    // static_assert: `std::is_nothrow_constructible_v<T, Args...>` -- so
    // `emplace_after(h, a, "x")` for `std::string` does not compile; when the
    // constructor may throw, construct a prvalue and use `insert_after`. The
    // arguments are forwarded, and a T constructed from them, whenever a
    // non-`end()` anchor is given (never for an `end()` anchor); if the
    // anchor turns out to be deleted that T is destroyed exactly as
    // `insert_after` describes (immediately for a fresh node, at a later
    // `reclaim()` for a reused one). Exceptions: as `insert_after`.
    template <typename... Args>
    [[nodiscard]] bool emplace_after(const handle& h, const_iterator anchor, Args&&... args) {
        static_assert(std::is_nothrow_constructible_v<T, Args...>, "LockFreeListRCU::emplace_after: T's constructor from these arguments must be noexcept (it runs into a reused node); construct a prvalue and use insert_after");
        return init_node(h, anchor, std::forward<Args>(args)...);
    } // emplace_after()

    // Erase the live node immediately after `anchor`. Returns true iff THIS
    // call logically deleted a node; false if `anchor` is `end()` or
    // logically deleted, or has no live successor. A call that loses the
    // deletion race helps unlink the dead node and retries on the new
    // successor, so `while (list.erase_after(h, list.before_begin(h))) {}`
    // drains the list under contention. Preconditions: `h` is a live handle
    // of this list and `anchor` was obtained under `h`'s current session
    // (both diagnosed). The erased node's value is NOT destroyed here (see
    // ERASE, RETIRE, RECLAIM); iterators parked on it stay usable while
    // their handles live. Lock-free. nothrow.
    //
    // Harris two-step: (1) mark the target's own next pointer to logically
    // delete it, winning the race against any concurrent eraser; (2) swing
    // anchor->next past the target to physically unlink it. Step 2 is
    // best-effort: if it fails (anchor->next changed under us), the marked
    // node stays linked until a later erase_after() over the same edge
    // removes it via the helping path. Unlike the reference-counted list,
    // the RESULT of the unlink CAS is tested on both paths: its unique
    // winner, marker or helper, is the one thread that retires the node.
    bool erase_after(const handle& h, const_iterator anchor) noexcept {
        Node* a = checked_anchor(h, anchor);
        if (!a) return false;

        // Retry loop: each pass either decides the logical-deletion race on
        // the current successor (win -> true) or helps unlink an already-
        // deleted successor and re-examines the new state. A retry happens
        // only after some CAS on anchor->next succeeded (ours or a
        // competitor's), so the list as a whole always makes progress
        // (lock-free, not wait-free).
        while (true) {
            // acquire: we dereference the successor's next below (and the
            // value it points to is republished through our unlink CAS);
            // pairs with the release CAS that linked it.
            marked_ptr target = a->next.load(std::memory_order_acquire);
            // Per the marking convention, a mark on the value loaded from
            // anchor->next is *anchor's own* deletion mark: nothing to erase,
            // or anchor itself is deleted. A target whose own next is marked
            // (already erased, not yet unlinked) passes this check and is
            // handled by the helping path below.
            if (!target.get_unmarked() || target.is_marked()) return false;
            Node* t = target.get_unmarked();

            // acquire: `target_next` is republished through the unlink CAS
            // below, and its pointee must be visible to whoever reads it
            // from anchor->next afterwards (transitively through us).
            marked_ptr target_next = t->next.load(std::memory_order_acquire);
            bool marked_by_us = false;
            // Race to set the mark on target's own next pointer. A failed CAS
            // refreshes `target_next`: if it is now marked, another eraser
            // won; if it merely changed (an insert landed after target),
            // retry the marking with the new successor. Release on success
            // (the mark is the linearization point other threads act on);
            // the acquire failure order is required because the refreshed
            // value is republished through the unlink CAS.
            while (!target_next.is_marked()) {
                marked_ptr marked_next = target_next.set_mark();
                if (t->next.compare_exchange_strong(target_next, marked_next, std::memory_order_release, std::memory_order_acquire)) {
                    marked_by_us = true;
                    target_next = marked_next; // to have the correct value for unlinking
                    break;
                }
            } // marking race

            // target is now logically deleted -- by us, by a concurrent
            // eraser, or long ago (a dead node left behind by an unlink that
            // lost its CAS). The mark freezes target->next for good:
            // insert_after() refuses a marked anchor, no second eraser can
            // win the marking race, and a retired node's next is not
            // rewritten until it is popped from the free list -- which the
            // handle invariant forbids while anyone can still reach it. So
            // target_next.clear_mark() is target's final successor and stays
            // the correct unlink value no matter how long we stall before
            // the CAS below.
            Hooks::between_mark_and_unlink();

            // Physically unlink it (helping, when we did not mark it
            // ourselves). release on success -- DESIGNATED, for PUBLICATION:
            // the value written, target's final successor, may be a node
            // first reachable from the head through this very word (a node
            // inserted after target just before target died), and a reader
            // acquiring anchor->next must see its construction, which we saw
            // through our acquire load of target->next; the release carries
            // it transitively (model-checked and seen under ThreadSanitizer:
            // with a relaxed unlink a reader's ++ races with the inserter's
            // construction). INVARIANT (I)'s chain does NOT need this
            // release (the retire push below is sequenced after the unlink
            // and carries it); the graveyard rule needs it too (reasoned,
            // not yet model-checked): with two or more erasers, a dead
            // node's successor can be unlinked through a word other than
            // the one a parked reader read, and only this release orders
            // that unlink before the reader. On failure anchor->next
            // changed under us -- an insert landed after anchor, or anchor
            // itself was erased -- and the dead node stays linked
            // ("stranded") until the next erase_after() over this edge helps
            // it out. The updated `target` is discarded (a retry reloads
            // it), hence the relaxed failure order. ABA on this CAS (anchor
            // -> t again after t was unlinked, freed, reused and re-inserted
            // here) is excluded by the invariant: our handle keeps t from
            // being freed.
            bool unlinked = a->next.compare_exchange_strong(target, target_next.clear_mark(), std::memory_order_release, std::memory_order_relaxed);
            // The unique winner of the unlink CAS -- marker or helper --
            // retires the node; a loser retires nothing (the node is still
            // linked, or someone else won and retired it).
            if (unlinked) retire(t);

            if (marked_by_us) {
                return true; // our mark is the linearization point; unlink success is not required
            }
            // We lost the marking race (or found a long-dead successor) and
            // helped unlink it (or failed to); retry on the new state of
            // anchor->next.
        } // retry / helping loop
    } // erase_after()

    // Reclaim: advance the generation and free every bag no live handle can
    // observe, oldest first (see RECLAIM in the overview for the full
    // semantics, the refusal results and the retry rule). Needs no handle;
    // non-const. Postcondition: the result describes exactly what this call
    // did; `advance == contended` means another call was in progress and
    // this one did nothing (`{0, 0, contended}`). Non-blocking: never waits
    // for another reclaimer or for handles. Runs the `~T` of every freed
    // node. nothrow.
    ReclaimResult reclaim() noexcept {
        // acquire: pairs with the previous holder's release store below --
        // this is the ONLY synchronization on the plain fields `state`,
        // `bag`, `gen`, `oldest_`, and it also places the previous holder's
        // publish before this holder's `refs` loads (DESIGNATED: with relaxed
        // orders here two callers race on the plain fields).
        if (reclaiming_.exchange(true, std::memory_order_acquire)) return {0, 0, ReclaimResult::Advance::contended};
        ReclaimResult result{0, 0, ReclaimResult::Advance::nothing_retired};
        result.advance = advance(result);
        free_oldest(result);
        // release: hands the plain fields and this call's publish to the
        // next holder's acquire exchange.
        reclaiming_.store(false, std::memory_order_release);
        return result;
    } // reclaim()

    // Test-only accessor: the accounting sweep (see InternalAccounting).
    // Precondition: quiescence -- no list operation, no `reclaim()` (debug-
    // asserted through the reclaim flag) and no join attempt (`new_handle()`
    // or `refresh()`) in progress on any thread, and never called from
    // inside a hook; handles and parked iterators may be alive (they are
    // counted, not excluded). Allocates nothing on the list; uses scratch
    // memory for the visited set (both builds: a corrupted, cyclic chain
    // then stops the walk instead of hanging it; only `consistent` is
    // debug-only).
    //
    // One visited set across all walks; a node seen twice (a cycle, or
    // membership in two populations) clears `consistent` and STOPS that
    // walk, so a doubly-listed node is attributed to the first population
    // in this fixed order: reachable, retired, bagged, free. The free walk
    // reads only `retire_link`, which lies outside the poisoned extent.
    // All loads relaxed: quiescent by precondition.
    InternalAccounting get_internal_accounting() const {
        assert(!reclaiming_.load(std::memory_order_relaxed) && "accounting sweep during reclaim");
        InternalAccounting acc{};
        std::unordered_set<const Node*> seen;
        [[maybe_unused]] bool consistent = true;
        auto visit = [&](const Node* p) {
            if (seen.insert(p).second) return true;
            consistent = false;
            return false;
        };
        for (const Node* p = head_.next.load(std::memory_order_relaxed).get_unmarked(); p && visit(p); p = p->next.load(std::memory_order_relaxed).get_unmarked()) {
            ++acc.reachable;
            if (p->next.load(std::memory_order_relaxed).is_marked()) ++acc.reachable_dead;
        } // walk 1: the chain from the head
        for (const Node* p = retired_head_.load(std::memory_order_relaxed); p && visit(p); p = p->retire_link.load(std::memory_order_relaxed)) {
            ++acc.retired;
            if (!p->next.load(std::memory_order_relaxed).is_marked()) consistent = false;  // a retired node's next must be marked
        } // walk 2: the retired list
        for (const GenerationBlock& block : ring_) {
            if (block.state != State::SEALED) continue;
            for (const Node* p = block.bag; p && visit(p); p = p->retire_link.load(std::memory_order_relaxed)) {
                ++acc.bagged;
                if (!p->next.load(std::memory_order_relaxed).is_marked()) consistent = false;  // a bagged node's next must be marked
            } // this block's bag
        } // walk 3: the sealed bags
        for (const Node* p = free_head_.load(std::memory_order_relaxed); p && visit(p); p = p->retire_link.load(std::memory_order_relaxed)) {
            ++acc.free;
            if (!node_is_poisoned(p)) consistent = false;  // every free-list node is poisoned (true without ASan)
        } // walk 4: the free list
        for (const GenerationBlock& block : ring_) acc.handle_refs += block.refs.load(std::memory_order_relaxed);
#ifndef NDEBUG
        acc.allocated = allocated_.load(std::memory_order_relaxed);
        if (acc.reachable + acc.retired + acc.bagged + acc.free != acc.allocated) consistent = false;  // population sum == allocated
        // Ring-state invariant: [CURRENT at c][SEALED run of n = (c - o) mod K
        // slots behind it, gen decreasing by 1][EMPTY elsewhere]
        const size_t c = slot_index(current_.load(std::memory_order_relaxed));
        const size_t n = (c + kGenerations - oldest_) % kGenerations;
        for (size_t i = 0; i < kGenerations; ++i) {
            const GenerationBlock& block = ring_[i];
            const size_t behind = (c + kGenerations - i) % kGenerations;  // 0 for the current slot, j for the j-th slot behind it
            State expected = behind == 0 ? State::CURRENT : behind <= n ? State::SEALED : State::EMPTY;
            if (block.state != expected) consistent = false;
            if (expected == State::SEALED && block.gen != ring_[c].gen - behind) consistent = false;
            if ((block.state == State::SEALED) != (block.bag != nullptr)) consistent = false;  // SEALED iff it has a bag
            const long refs = block.refs.load(std::memory_order_relaxed);
            if (refs < 0 || (block.state == State::EMPTY && refs != 0)) consistent = false;    // counts never negative; an EMPTY slot counts nobody
        } // ring-state invariant
        acc.consistent = consistent;
#endif
        return acc;
    } // get_internal_accounting()

#ifndef NDEBUG
    // Test-only accessor (debug builds only): the node incarnation recorded
    // on the node `it` is on -- bumped every time the node is reused from
    // the free list. Precondition: `it != end()`; quiescence as above.
    uint32_t get_internal_incarnation(const_iterator it) const {
        assert(it.curr_ != nullptr && "incarnation of end");
        return it.curr_->incarnation.load(std::memory_order_relaxed);
    } // get_internal_incarnation()
#endif

    // Test-only accessor: the number of nodes on the retired list (unlinked,
    // not yet bagged), as `get_internal_accounting().retired`. Precondition:
    // quiescence as above.
    size_t get_internal_retired_count() const { return get_internal_accounting().retired; }

private:
    // --- Precondition checks and iterator construction (the checks compile
    // --- to nothing under NDEBUG; checked_anchor() and make_iterator() do
    // --- their non-debug work in every build) ----------------------------

    // The two checks every operation taking a handle performs, before
    // anything else: the handle is live and belongs to this list.
    void check_handle([[maybe_unused]] const handle& h) const noexcept {
        assert(h.block_ != nullptr && "empty handle");
        assert(h.block_->list == this && "handle of another list");
    } // check_handle()
    // The check every operation taking (handle, non-null iterator) performs
    // before the anchor node is touched: the iterator was obtained under the
    // handle's CURRENT session (a refresh or a move-assignment over the
    // handle started a new one; a move kept it). Session ids are globally
    // unique, so this also catches an iterator of another list or handle.
    void check_iterator([[maybe_unused]] const handle& h, [[maybe_unused]] const_iterator it) const noexcept {
        assert(it.session_ == h.session_ && "iterator used with a different handle session");
    } // check_iterator()
    // The entry of every operation taking (handle, anchor): the handle checks
    // first (unconditionally, so they fire even for an end() anchor), then
    // the session check for a non-null anchor only (a null iterator carries
    // no session), all before the anchor node is touched. Returns the anchor
    // node, nullptr for end().
    Node* checked_anchor(const handle& h, const_iterator anchor) const noexcept {
        check_handle(h);
        Node* a = anchor.curr_;
        if (a) check_iterator(h, anchor);
        return a;
    } // checked_anchor()
#ifndef NDEBUG
    // A fresh, globally unique (per list type) session id; never 0.
    static uint64_t new_session_id() noexcept { return next_session_id_.fetch_add(1, std::memory_order_relaxed) + 1; }
#endif
    // An iterator on `n` bound to `h`'s session, recording `n`'s incarnation.
    template <bool kConst>
    basic_iterator<kConst> make_iterator(Node* n, [[maybe_unused]] const handle& h) const noexcept {
        basic_iterator<kConst> it;
        it.curr_ = n;
#ifndef NDEBUG
        it.owner_ = this;
        it.session_ = h.session_;
        it.incarnation_ = n->incarnation.load(std::memory_order_relaxed);
#endif
        return it;
    } // make_iterator()

    // --- Node storage helpers --------------------------------------------

    // The T living in `n`'s storage (precondition: constructed).
    static T* value_of(Node* n) noexcept { return std::launder(reinterpret_cast<T*>(n->value)); }
    static const T* value_of(const Node* n) noexcept { return std::launder(reinterpret_cast<const T*>(n->value)); }
    // The extent poisoned while a node is free: `[value, retire_link)`, by
    // address difference and not `offsetof` (`-Winvalid-offsetof` fails
    // `-Werror` for a non-standard-layout T). `retire_link` stays readable
    // for the free-list pop and the sweep; the debug incarnation after it
    // stays readable for parked iterators.
    static size_t poison_extent(const Node* n) noexcept {
        return static_cast<size_t>(reinterpret_cast<const std::byte*>(&n->retire_link) - reinterpret_cast<const std::byte*>(n));
    } // poison_extent()
    // Under ASan: every byte of the poisoned extent is poisoned and
    // `retire_link` is not. Without ASan there is no oracle: true.
    static bool node_is_poisoned([[maybe_unused]] const Node* n) noexcept {
#ifdef LFL_RCU_HAVE_ASAN
        const std::byte* bytes = reinterpret_cast<const std::byte*>(n);
        for (size_t i = 0; i < poison_extent(n); ++i) {
            if (!__asan_address_is_poisoned(bytes + i)) return false;
        }
        return __asan_region_is_poisoned(const_cast<void*>(static_cast<const void*>(&n->retire_link)), sizeof(n->retire_link)) == nullptr;
#else
        return true;
#endif
    } // node_is_poisoned()
    // `delete` plus the debug allocation count.
    void delete_node(Node* n) noexcept {
        delete n;
#ifndef NDEBUG
        allocated_.fetch_sub(1, std::memory_order_relaxed);
#endif
    } // delete_node()
    // The ring index of a block.
    size_t slot_index(const GenerationBlock* block) const noexcept { return static_cast<size_t>(block - ring_); }

    // --- The join / leave protocol (see THE JOIN in the overview) --------

    // Count a new membership in the current generation and return its
    // block. Called by new_handle() and by refresh()'s slow path; const
    // because a const list yields handles. Lock-free: a retry happens only
    // after a reclaimer published a new generation. J1: between the
    // `fetch_add` and a successful re-verify this function touches NO node,
    // head or bag.
    GenerationBlock* join() const noexcept {
        while (true) {
            // relaxed: a stale block here is caught by the re-verify
            // (EXPECTED-EQUIVALENT to acquire).
            GenerationBlock* block = current_.load(std::memory_order_relaxed);
            Hooks::in_join();
            // seq_cst: one half of the store-buffering pair with the
            // reclaimer's seq_cst publish store and seq_cst `refs` load
            // (DESIGNATED: an acq_rel increment lets the reclaimer miss this
            // handle and free the bag it is about to read).
            block->refs.fetch_add(1, std::memory_order_seq_cst);
            Hooks::in_join_after_bump();
            // seq_cst: the other half -- this load and the reclaimer's refs
            // load cannot both miss (DESIGNATED: an acquire load here races
            // the same way). Pointer equality suffices (see the overview).
            if (current_.load(std::memory_order_seq_cst) == block) return block;
            // Undo the stale increment and retry. release: matches leave();
            // nothing was accessed since the increment (J1), so there is
            // nothing to publish and relaxed would be EXPECTED-EQUIVALENT.
            [[maybe_unused]] long prev = block->refs.fetch_sub(1, std::memory_order_release);
            assert(prev > 0 && "generation refs went negative");  // every -1 follows its own +1 (J2)
        } // join attempts
    } // join()

    // Drop one membership of `block` (handle destruction, the old side of
    // refresh(), move assignment into a live handle). release (DESIGNATED):
    // every access the handle made is sequenced before this, and the
    // reclaimer's seq_cst load that reads the resulting 0 (or a later value
    // in the release sequence) acquires it before destroying the values --
    // the reader's last `*it` happens-before the `~T`; relaxed here lets the
    // `~T` race that read.
    void leave(GenerationBlock* block) const noexcept {
        [[maybe_unused]] long prev = block->refs.fetch_sub(1, std::memory_order_release);
        assert(prev > 0 && "generation refs went negative");  // every -1 follows its own +1 (J2)
    } // leave()

    // --- Retire, advance, free (see ONE RECLAIMER in the overview) -------

    // Push an unlinked node onto the retired list through `retire_link`; by
    // the unique winner of the unlink CAS only (and by dispose_unlinked()).
    // The node's `next` must be marked (frozen): parked iterators still
    // walk out through it. The push is a release RMW heading a release
    // sequence that later pushes continue, so the reclaimer's acquire
    // exchange of `retired_head_` synchronizes with EVERY pusher (it reads
    // each node's `retire_link`, written before that node's push), and
    // INVARIANT (I)'s chain runs through it (DESIGNATED: a relaxed push lets
    // a handle joined after the publish read a node of the freed bag).
    // The failure order is relaxed: the refreshed head is only stored into
    // `retire_link`, never dereferenced here.
    void retire(Node* node) noexcept {
        assert(node->next.load(std::memory_order_relaxed).is_marked() && "retired node is not marked");
        Node* head = retired_head_.load(std::memory_order_relaxed);
        do {
            node->retire_link.store(head, std::memory_order_relaxed);  // published by the release CAS below
        } while (!retired_head_.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
    } // retire()

    // Close the current generation and publish the next; the free pass in
    // reclaim() follows. Returns the Advance part of the result and
    // accumulates into `result` the frees of the one free pass it may run
    // when the next slot is not EMPTY (`ring_full` is returned only when
    // that pass could not clear it, i.e. the oldest block is pinned). Runs
    // under the flag: `current_` is published by this holder alone, so the
    // relaxed loads here are exact. Steps A0-A5 of the overview; the
    // exchange (A3) precedes the publish (A5) -- load-bearing for INVARIANT
    // (I): an exchange moved after the publish lets a handle joined to the
    // new generation read a node of the closing bag as it is freed. An
    // exchange moved to after the hook below is caught by the scheme tests;
    // one moved to between the publish and the hook has no deterministic
    // test (model checker and order review only; stress tests under
    // ThreadSanitizer catch it sometimes).
    ReclaimResult::Advance advance(ReclaimResult& result) noexcept {
        // relaxed: only the emptiness is tested; a push racing this load is
        // seen by the next call (EXPECTED-EQUIVALENT).
        if (retired_head_.load(std::memory_order_relaxed) == nullptr) return ReclaimResult::Advance::nothing_retired;
        // relaxed: this holder's own store, or the previous holder's through
        // the flag's release/acquire (EXPECTED-EQUIVALENT).
        GenerationBlock* closing = current_.load(std::memory_order_relaxed);
        GenerationBlock* next = &ring_[(slot_index(closing) + 1) % kGenerations];
        if (next->state != State::EMPTY) {
            // The ring has wrapped: `next` is the oldest SEALED block. Free
            // what can be freed and look again; still occupied means the
            // oldest generation is pinned (really, or by a transient stale
            // increment -- a liveness matter either way).
            free_oldest(result);
            if (next->state != State::EMPTY) return ReclaimResult::Advance::ring_full;
        } // if the next slot is in use
        // acquire: synchronizes with every retire push (release RMWs on one
        // release sequence), so the bag's `retire_link` chain and the frozen
        // `next` words are visible to this holder and, through the flag, to
        // later holders (DESIGNATED: a relaxed exchange lets a later handle
        // read a freed node; acq_rel EXPECTED-EQUIVALENT: nobody acquires the
        // nullptr written here).
        Node* chain = retired_head_.exchange(nullptr, std::memory_order_acquire);
        if (!chain) return ReclaimResult::Advance::nothing_retired;  // defensive: only the reclaimer removes, so unreachable after the test above
        assert(closing->state == State::CURRENT && "closing block is not CURRENT");
        assert(next->state == State::EMPTY && "next block is not EMPTY");  // structural: established by the re-check above
        // Plain writes, owned by the flag holder; the bag is reachable only
        // through this block until it is spliced to the free list.
        closing->bag = chain;
        closing->state = State::SEALED;
        next->state = State::CURRENT;
        next->gen = closing->gen + 1;
        // seq_cst STORE: the reclaimer's half of the store-buffering pair
        // with the join (DESIGNATED: a relaxed or release publish lets a
        // joiner use a bag this call frees; a seq_cst RMW instead of the
        // store is EXPECTED-EQUIVALENT). Also the release that INVARIANT
        // (I)'s chain ends on: sequenced after the exchange above.
        current_.store(next, std::memory_order_seq_cst);
        Hooks::in_advance_after_publish();
        return ReclaimResult::Advance::advanced;
    } // advance()

    // Free sealed bags oldest-first while the oldest has no live handle,
    // accumulating into `result`. Under the flag. `oldest_` indexes the
    // oldest SEALED block, or the CURRENT block when nothing is sealed, so
    // the loop stops when `ring_[oldest_]` is CURRENT (nothing sealed is
    // left; the `!= SEALED` test also covers EMPTY defensively, which the
    // index never points at) or when its `refs` is not 0 (a handle of that
    // generation -- or a stale increment, J2 -- pins it and, by the
    // cumulative rule, every younger bag). A freed block's `refs` is left
    // untouched (J2).
    void free_oldest(ReclaimResult& result) noexcept {
        while (true) {
            GenerationBlock& block = ring_[oldest_];
            if (block.state != State::SEALED) return;
            // seq_cst: the reclaimer's half of the store-buffering pair with
            // the join's seq_cst increment and re-verify (DESIGNATED: an
            // acquire load here frees under a joiner). It also acquires the
            // leaves whose release decrements produced the 0, so every
            // access those handles made happens-before the `~T`s below.
            if (block.refs.load(std::memory_order_seq_cst) != 0) return;
            assert(block.state == State::SEALED && "freeing a block that is not SEALED");  // structural: established by the state test above
            Node* bag = block.bag;
            Node* tail = nullptr;
            size_t count = 0;
            for (Node* p = bag; p; p = p->retire_link.load(std::memory_order_relaxed)) {
                assert(p->next.load(std::memory_order_relaxed).is_marked() && "retired node is not marked");
                value_of(p)->~T();
                LFL_RCU_ASAN_POISON(p, poison_extent(p));  // `retire_link` and the debug incarnation stay readable
                tail = p;
                ++count;
            } // destroy and poison every node of the bag (relaxed walk: this holder acquired the pushes through the exchange, or through the flag)
            // Splice the bag onto the free list: tail -> old head, then a
            // release CAS. release: publishes the `~T`s, the poison and the
            // `retire_link` chain to the popper's acquire load/CAS -- the
            // popper's construction of a new T into the storage must happen
            // after the destruction here. The failure order is relaxed: the
            // refreshed head is only stored into `tail->retire_link`.
            Node* head = free_head_.load(std::memory_order_relaxed);
            do {
                tail->retire_link.store(head, std::memory_order_relaxed);  // published by the release CAS below
            } while (!free_head_.compare_exchange_weak(head, bag, std::memory_order_release, std::memory_order_relaxed));
            block.bag = nullptr;
            block.state = State::EMPTY;
            oldest_ = (oldest_ + 1) % kGenerations;
            result.freed_nodes += count;
            ++result.freed_bags;
        } // free the oldest sealed bag while it is unpinned
    } // free_oldest()

    // --- Allocation, initialization, disposal ----------------------------

    struct AllocatedNode { Node* node; bool popped; };  // alloc_node()'s result: the node and whether it came from the free list (else from `new`)

    // Pop a free node or `new` one; the failure path of init_node() needs
    // to know which (dispose a popped node, delete a new one). Called under
    // the caller's handle, which is what excludes ABA on the pop CAS: a
    // node popped by another thread cannot return to the free list while
    // our handle lives (see INVARIANT in the overview). The free list is a
    // Treiber stack through `retire_link`. A popped node is unpoisoned
    // (ASan) and its incarnation bumped (debug); its `next` is still the
    // frozen marked value until init_node() overwrites it. May throw
    // std::bad_alloc from `new`.
    AllocatedNode alloc_node() {
        // acquire: synchronizes with the splice's release CAS, so the head
        // node's `retire_link` (read below), its destroyed storage and its
        // poison are visible before we touch them.
        Node* head = free_head_.load(std::memory_order_acquire);
        while (head) {
            Node* next = head->retire_link.load(std::memory_order_relaxed);  // written by the splicer before its release CAS
            // Success acq_rel (the acquire half re-synchronizes with the
            // splice that produced the value we consumed; the release half
            // publishes nothing a later popper needs -- later poppers reach
            // the splicer through the release sequence this RMW continues --
            // both halves EXPECTED-EQUIVALENT to relaxed given the acquire
            // load and failure order, kept per the order table). Failure
            // ACQUIRE is REQUIRED: a refreshed `head` may come from a LATER
            // splice, and we dereference `head->retire_link` on the next
            // pass -- with a relaxed failure order its write (and the poison,
            // and the `~T`) need not be visible yet.
            if (free_head_.compare_exchange_weak(head, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
                LFL_RCU_ASAN_UNPOISON(head, poison_extent(head));
#ifndef NDEBUG
                head->incarnation.fetch_add(1, std::memory_order_relaxed);  // relaxed: no legal iterator can observe this node now; debug only
#endif
                return {head, true};
            } // pop won: the node is ours
        } // pop loop: a failed CAS refreshed `head`; empty list -> fall through to new
        Node* fresh = new Node;
#ifndef NDEBUG
        allocated_.fetch_add(1, std::memory_order_relaxed);
#endif
        return {fresh, false};
    } // alloc_node()

    // The ONE initialization + link path behind insert_after/emplace_after:
    // a null anchor returns false before any allocation; otherwise
    // alloc_node(), placement-construct T from `args` (nothrow by the
    // callers' static_asserts), then the Harris insert loop whose first step
    // loads anchor->next; a marked anchor on any pass disposes a popped node
    // / deletes a new node after `~T` and returns false. The disposal path
    // is thereby reachable single-threaded (a dead anchor). May throw
    // std::bad_alloc from alloc_node() only, before anything changed.
    template <typename... Args>
    bool init_node(const handle& h, const_iterator anchor, Args&&... args) {
        Node* a = checked_anchor(h, anchor);
        if (!a) return false;
        AllocatedNode allocated = alloc_node();
        Node* node = allocated.node;
        ::new (static_cast<void*>(node->value)) T(std::forward<Args>(args)...);
        while (true) {
            // acquire: `expected_next` is republished through node->next
            // and its pointee must be visible to whoever reads it from there
            // (transitively through our release CAS).
            marked_ptr expected_next = a->next.load(std::memory_order_acquire);
            // Per the marking convention, a mark on the value loaded from
            // anchor->next is *anchor's own* deletion mark: linking onto it
            // would strand the node in a chain about to be unlinked.
            if (expected_next.is_marked()) break;
            // relaxed: node is not reachable by any other thread yet; the
            // release CAS below publishes it, this store and the T included.
            node->next.store(expected_next, std::memory_order_relaxed);
            if (a->next.compare_exchange_strong(expected_next, marked_ptr(node), std::memory_order_release, std::memory_order_relaxed)) return true;
            // CAS failure: a competing insert/unlink changed the successor,
            // or anchor got marked (the mark is part of the compared value,
            // so marking anchor always fails this CAS). Discard the updated
            // `expected_next` and reload; the reload re-runs the mark check.
        } // insert retry loop
        // The anchor is logically deleted: the node did not get linked.
        if (allocated.popped) {
            dispose_unlinked(node);
        } else {
            value_of(node)->~T();
            delete_node(node);
        }
        return false;
    } // init_node()

    // A popped node that did not get linked goes to the retired list (see
    // DISPOSAL in the overview), its value left constructed for the bag
    // free. Its `next` gets a marked value -- the node is private to us, so
    // relaxed; the retire push publishes it -- which keeps every "retired
    // node is marked" assert and the sweep's check true.
    void dispose_unlinked(Node* node) noexcept {
        node->next.store(marked_ptr().set_mark(), std::memory_order_relaxed);
        retire(node);
    } // dispose_unlinked()

    // Data members; the shared words each sit on their own 128-byte line.
    mutable GenerationBlock ring_[kGenerations];       // the fixed ring of generation blocks; mutable: a const list's handles join and leave
    alignas(128) std::atomic<GenerationBlock*> current_; // the block new handles join; written only by the flag holder (seq_cst store), read by joiners
    alignas(128) std::atomic<bool> reclaiming_;        // the single-reclaimer try-flag: set while a reclaim() is in progress
    size_t oldest_;                                    // index of the oldest sealed (or the current) block; owned by the reclaimer, shares its line
    alignas(128) std::atomic<Node*> retired_head_;     // nodes unlinked since the last advance, linked through `retire_link`
    alignas(128) std::atomic<Node*> free_head_;        // nodes whose values are destroyed, ready for reuse, linked through `retire_link`
    alignas(128) mutable Node head_;                   // the dummy head: never erased, retired or freed; mutable: a const list's const_iterator still holds a Node*
#ifndef NDEBUG
    std::atomic<size_t> allocated_{0};                 // debug only: nodes from `new` not yet deleted (InternalAccounting::allocated)
    static inline std::atomic<uint64_t> next_session_id_{0}; // debug only: source of the globally unique handle session ids
#endif
}; // LockFreeListRCU

#endif // INCLUDED_LOCK_FREE_LIST_RCU_H
