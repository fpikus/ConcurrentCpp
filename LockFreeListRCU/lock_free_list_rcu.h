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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

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
// for as long as it lives or until `refresh()` moves it to the current
// generation, whichever comes first. The cost of this protection is that
// nothing retired at or after the handle's generation can be reclaimed while
// the handle stays put (see RECLAIM below).
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
// once the generation that could observe it has no live handle. A node is in
// exactly one of four populations at any time: linked from the head (live,
// marked-but-linked, or stranded), on the retired list, in a sealed
// generation's bag, or on the free list; `get_internal_accounting()` reports
// all four.
//
// `reclaim()` is caller-driven: nothing is reclaimed unless some thread calls
// it. ONE call proceeds at a time: a `reclaim()` that finds another in
// progress returns `ReclaimResult::Advance::contended` immediately, does
// nothing, and never waits. The call that proceeds first ADVANCES the
// generation (everything retired so far becomes the closing generation's bag;
// a new generation becomes current -- new handles and refreshed handles join
// it) and then FREES bags oldest-first: the oldest sealed bag is freed while
// no handle of its generation is alive, then the next, and so on until a
// generation with a live handle is reached. Freeing is cumulative -- a bag is
// never freed while an older bag is still pinned -- because a handle of
// generation g can observe nodes that land in bags g, g + 1, ... and is
// protected only if every one of them waits for it. The bag a call just
// sealed is freeable in the SAME call when no handle of its generation or an
// older one is alive; otherwise by a later call. A `reclaim()` may REFUSE:
// `nothing_retired` (nothing to advance -- bags that became free may still
// have been freed), `ring_full` (every generation slot is in use because an
// old handle pins the oldest bag; no slot is consumed), or `contended`. A
// refusal costs nothing and is retried by calling again: `contended` after
// the other call returns, `ring_full` after the pinning handles are destroyed
// or refreshed -- there is no automatic recovery, and a `ring_full` result
// may already be stale on return (the same call's free pass may have cleared
// the pin). A thread that HOLDS a handle and calls `reclaim()` frees nothing
// at or after that handle's generation, because its own handle pins it; the
// recommended sequence for a thread that both operates on the list and
// reclaims is `h.refresh(); list.reclaim();`.
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
// of the list with live handles. Other violations (`++` on `end()`, use of
// an invalidated iterator) are undefined behavior that debug builds MAY
// catch but the contract does not promise to.
//
// TEST SEAMS. The `Hooks` template parameter (default `NoHooks`, constrained
// by `ListHooks`) names four `noexcept` static functions the list calls at
// fixed points of its protocol; the `get_internal_*` members are test-only
// accessors with a quiescence precondition. Neither is part of the list's
// operational contract; both are documented at their declarations.
//
// FOOTPRINT. About 8.7 KB for `LockFreeListRCU<int>` at the default
// kGenerations = 64: 64 generation blocks of 128 bytes each (one cache line
// per block so adjacent reference counts do not share a line) plus the
// list's own private lines. Nodes are 24 bytes for `int` (value, next,
// retire link) in release builds, as in the reference-counted list.
//
// [Stage 1: protocol overview -- ring, join J1/J2, single-reclaimer
// sequence, INVARIANT (I)/(II), order table reference]

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
        nothing_retired, // nothing was retired since the last advance; no generation slot consumed
        advanced,        // the retired nodes became the closing generation's bag; a new generation is current
        ring_full,       // every generation slot is in use (the oldest bag is pinned by a handle); no slot consumed
        contended        // another reclaim() is in progress; this call did nothing and returned at once
    };
    size_t freed_nodes;  // nodes whose value was destroyed and which moved to the free list in this call
    size_t freed_bags;   // generations whose bags were freed in this call (oldest first)
    Advance advance;     // see above; `nothing_retired` and `ring_full` do NOT imply freed_nodes == 0 -- the free pass still runs; `contended` implies {0, 0}
    bool operator==(const ReclaimResult&) const = default;
}; // ReclaimResult

// LockFreeListRCU<T, kGenerations, Hooks>: see the overview above.
//   T            the value type (nothrow-destructible; see VALUE TYPE)
//   kGenerations the number of generation slots (>= 2): how many advances
//                `reclaim()` can perform while the oldest generation stays
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
    // that expected the unmarked value. Stage 1 may add accessors; the bit
    // layout below is fixed (Node is at least 8-byte aligned, bit 0 is free).
    class marked_ptr {
    public:
        marked_ptr() noexcept = default;                                         // null, unmarked
        explicit marked_ptr(Node* p) noexcept : bits_(reinterpret_cast<uintptr_t>(p)) {} // unmarked pointer to `p`
        bool is_marked() const noexcept { return (bits_ & kMark) != 0; }
        Node* get_unmarked() const noexcept { return reinterpret_cast<Node*>(bits_ & ~kMark); } // the pointer without the mark
        marked_ptr set_mark() const noexcept { marked_ptr r; r.bits_ = bits_ | kMark; return r; } // the same pointer, marked
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
    // is on the free list. Accessed through `std::launder` on the storage.
    // `value` comes FIRST so that `&*it` is the node's identity.
    struct Node {
        alignas(T) std::byte value[sizeof(T)];  // storage for the T (see above)
        std::atomic<marked_ptr> next;            // the list link; a marked value = THIS node is logically deleted; frozen once marked until the node is reused
        std::atomic<Node*> retire_link;          // the second link: retired-list, bag and free-list chaining -- never `next`, so parked iterators still walk the graveyard
#ifndef NDEBUG
        std::atomic<uint32_t> incarnation;       // debug only: bumped on every free-list pop; iterators record it to detect use across a reuse
#endif
    }; // Node

    // Lifecycle of a generation slot in the ring.
    enum class State { EMPTY, CURRENT, SEALED };

    // One generation: a status block in the fixed ring (never deallocated).
    // One cache line per block: adjacent `refs` would otherwise share a line.
    struct alignas(128) GenerationBlock {
        std::atomic<long> refs;  // live handles joined to this generation (RMW-only; see Stage 1)
        Node* bag;               // the retired nodes of this generation, once SEALED (owned by the reclaimer)
        State state;             // EMPTY / CURRENT / SEALED (owned by the reclaimer)
        uint64_t gen;            // the generation number this slot currently represents (owned by the reclaimer)
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
        handle() noexcept;
        // Transfers `other`'s session; `other` is left empty. Iterators
        // obtained under `other` are valid under *this. nothrow.
        handle(handle&& other) noexcept;
        // Ends this handle's session if it has one (iterators obtained under
        // *this become invalid), then transfers `other`'s; `other` is left
        // empty. Self-move-assignment is a no-op. nothrow.
        handle& operator=(handle&& other) noexcept;
        handle(const handle&) = delete;
        handle& operator=(const handle&) = delete;
        // Ends the session (leaves the generation) if live; iterators
        // obtained under *this become invalid. Precondition: if live, the
        // list is still alive. nothrow.
        ~handle() noexcept;

        // True iff this handle is live (holds a session). nothrow.
        explicit operator bool() const noexcept;

        // Re-join the list's current generation. Precondition: live (debug-
        // asserted). Postcondition: the handle is a member of the generation
        // that was current at some point during the call; EVERY iterator
        // obtained under this handle before the call is invalid, whatever
        // the result. Returns true iff the handle actually moved to a newer
        // generation (false: it was already current, nothing changed except
        // the invalidation). Releases the old generation for reclamation
        // once no other handle pins it. Lock-free. nothrow.
        bool refresh() noexcept;

        // Exchanges the sessions (and the iterators each validates) of two
        // handles; either or both may be empty. nothrow.
        void swap(handle& other) noexcept;
        friend void swap(handle& a, handle& b) noexcept { a.swap(b); }

    private:
        friend class LockFreeListRCU;
        GenerationBlock* block_;  // the generation this handle is joined to; nullptr when empty
#ifndef NDEBUG
        const LockFreeListRCU* owner_;  // debug only: the list this handle belongs to (diagnoses a handle of another list)
        uint64_t session_;              // debug only: globally unique id of this session; a new one on every join/refresh; moves with the handle
#endif
        // Stage 1: the private constructor used by new_handle(), the join and
        // leave protocol.
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
        basic_iterator() noexcept;
        basic_iterator(const basic_iterator&) noexcept = default;
        basic_iterator& operator=(const basic_iterator&) noexcept = default;
        // iterator -> const_iterator, implicit; the converse does not exist.
        // The converted iterator is bound to the same handle session as the
        // original (it is the same iterator, read-only).
        template <bool kOtherConst> requires (kConst && !kOtherConst)
        basic_iterator(const basic_iterator<kOtherConst>& other) noexcept;

        // Access the value. Preconditions: valid, != end(), and not the
        // iterator `before_begin()` returned (debug-asserted: the head holds
        // no value). The node may be logically deleted (graveyard) -- its
        // value is intact while the iterator is valid. nothrow.
        reference operator*() const noexcept;
        pointer operator->() const noexcept;

        // Advance to the successor, live or deleted (iteration does not skip
        // logically deleted nodes); past the last node yields end().
        // Precondition: valid and != end() (`++` on end() is a precondition
        // violation; debug builds may assert). nothrow.
        basic_iterator& operator++() noexcept;

        // Node identity only; `end()` equals `end()` and a default-constructed
        // iterator across handles and lists, in both operand orders;
        // `iterator` and `const_iterator` compare through the conversion.
        // `!=` is rewritten from this. nothrow.
        friend bool operator==(const basic_iterator& a, const basic_iterator& b) noexcept { return a.curr_ == b.curr_; }

    private:
        friend class LockFreeListRCU;
        template <bool> friend class basic_iterator;
        Node* curr_;  // the node this iterator is on; nullptr for end()
#ifndef NDEBUG
        const LockFreeListRCU* owner_;  // debug only: the list (a null iterator carries no binding)
        uint64_t session_;              // debug only: the handle session this iterator was obtained under
        uint32_t incarnation_;          // debug only: `curr_->incarnation` when obtained; `*`, `->`, `++` compare it
#endif
        // Stage 1: the private constructor used by before_begin()/begin().
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
    // nothrow.
    LockFreeListRCU() noexcept;
    // Destroys every value still constructed and frees every node in all
    // four populations. Preconditions (debug-asserted): no handle of this
    // list is alive; no `reclaim()` is in progress; no concurrent operation.
    ~LockFreeListRCU();
    LockFreeListRCU(const LockFreeListRCU&) = delete;
    LockFreeListRCU& operator=(const LockFreeListRCU&) = delete;

    // Join the current generation and return a live handle for it. Const: a
    // const list yields handles too (its operations then return
    // `const_iterator`). Postcondition: the handle is live and belongs to
    // this list. Lock-free. nothrow.
    [[nodiscard]] handle new_handle() const noexcept;

    // The anchor before the first element: the list's head, which is never
    // erased, so it is always a valid anchor for `insert_after()` and
    // `erase_after()`. Its `*`/`->` are a precondition violation (diagnosed).
    // Precondition: `h` is a live handle of this list (diagnosed). The
    // iterator is bound to `h`'s current session. nothrow.
    iterator before_begin(const handle& h) noexcept;
    const_iterator before_begin(const handle& h) const noexcept;

    // The first node (live or logically deleted -- see ITERATOR VALIDITY),
    // or `end()` if there is none. Precondition: `h` is a live handle of
    // this list (diagnosed). The iterator is bound to `h`'s session. nothrow.
    iterator begin(const handle& h) noexcept;
    const_iterator begin(const handle& h) const noexcept;

    // The past-the-end iterator. Static: it needs no handle and compares
    // equal to every past-the-end and default-constructed iterator of every
    // list, in both operand orders. `cend()` is the `const_iterator` form;
    // `end()` converts to it implicitly. nothrow.
    static iterator end() noexcept;
    static const_iterator cend() noexcept;

    // Insert a node holding `value` immediately after `anchor`. Returns true
    // on success; false if `anchor` is `end()` or logically deleted (linking
    // onto a deleted node would strand the new node). Preconditions: `h` is
    // a live handle of this list and `anchor` was obtained under `h`'s
    // current session (both diagnosed). `value` is consumed in every case
    // (moved into the node); when the call returns false the node's value is
    // destroyed either before the call returns (a freshly allocated node) or
    // at a later `reclaim()` (a node reused from the free list).
    // static_assert: `std::is_nothrow_move_constructible_v<T>`.
    // Exceptions: may throw only `std::bad_alloc`, from `new` when the free
    // list is empty -- including on a deleted anchor that would otherwise
    // return false (an `end()` anchor returns false before any allocation).
    // On a throw the list is unchanged (the caller's argument was already
    // moved from at the call site). Lock-free apart from `new`.
    [[nodiscard]] bool insert_after(const handle& h, const_iterator anchor, T value);

    // As `insert_after`, constructing the value in place from `args...`.
    // static_assert: `std::is_nothrow_constructible_v<T, Args...>` -- so
    // `emplace_after(h, a, "x")` for `std::string` does not compile; when the
    // constructor may throw, construct a prvalue and use `insert_after`. The
    // arguments are forwarded only when a node is being initialized (never
    // for an `end()` anchor). Exceptions: as `insert_after`.
    template <typename... Args>
    [[nodiscard]] bool emplace_after(const handle& h, const_iterator anchor, Args&&... args);

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
    bool erase_after(const handle& h, const_iterator anchor) noexcept;

    // Reclaim: advance the generation and free every bag no live handle can
    // observe, oldest first (see RECLAIM in the overview for the full
    // semantics, the refusal results and the retry rule). Needs no handle;
    // non-const. Postcondition: the result describes exactly what this call
    // did; `advance == contended` means another call was in progress and
    // this one did nothing (`{0, 0, contended}`). Non-blocking: never waits
    // for another reclaimer or for handles. Runs the `~T` of every freed
    // node. nothrow.
    ReclaimResult reclaim() noexcept;

    // Test-only accessor: the accounting sweep (see InternalAccounting).
    // Precondition: quiescence -- no list operation, no `reclaim()` (debug-
    // asserted through the reclaim flag) and no join attempt (`new_handle()`
    // or `refresh()`) in progress on any thread, and never called from
    // inside a hook; handles and parked iterators may be alive (they are
    // counted, not excluded). Allocates nothing on the list; may use scratch
    // memory for the double-membership check in debug builds.
    InternalAccounting get_internal_accounting() const;

#ifndef NDEBUG
    // Test-only accessor (debug builds only): the node incarnation recorded
    // on the node `it` is on -- bumped every time the node is reused from
    // the free list. Precondition: `it != end()`; quiescence as above.
    uint32_t get_internal_incarnation(const_iterator it) const;
#endif

    // Test-only accessor: the number of nodes on the retired list (unlinked,
    // not yet bagged), as `get_internal_accounting().retired`. Precondition:
    // quiescence as above.
    size_t get_internal_retired_count() const;

private:
    // Private protocol -- Stage 1 (signatures provisional; names are the contract).
    GenerationBlock* join() const noexcept;            // Stage 1: count a new membership in the current generation and return its block (new_handle(), refresh()'s slow path)
    void leave(GenerationBlock* block) const noexcept; // Stage 1: drop one membership of `block`
    ReclaimResult::Advance advance() noexcept;         // Stage 1: close the current generation (the retired list becomes its bag), publish the next
    void free_oldest(ReclaimResult& result) noexcept;  // Stage 1: free sealed bags oldest-first while the oldest has no live handle
    void retire(Node* node) noexcept;                  // Stage 1: push an unlinked node onto the retired list (by the unlink CAS winner only)
    struct AllocatedNode { Node* node; bool popped; };  // alloc_node()'s result: the node and whether it came from the free list (else from `new`)
    AllocatedNode alloc_node();                        // Stage 1: pop a free node or `new` one; the failure path of init_node() disposes a popped node and deletes a new one
    template <typename... Args>
    bool init_node(const handle& h, const_iterator anchor, Args&&... args); // Stage 1: the ONE initialization + link path behind insert_after/emplace_after
    void dispose_unlinked(Node* node) noexcept;        // Stage 1: a popped node that did not get linked goes to the retired list, not back to the free list

    // Data members; the shared words each sit on their own 128-byte line.
    mutable GenerationBlock ring_[kGenerations];       // the fixed ring of generation blocks; mutable: a const list's handles join and leave
    alignas(128) std::atomic<GenerationBlock*> current_; // the block new handles join
    alignas(128) std::atomic<bool> reclaiming_;        // the single-reclaimer try-flag: set while a reclaim() is in progress
    size_t oldest_;                                    // index of the oldest sealed (or the current) block; owned by the reclaimer, shares its line
    alignas(128) std::atomic<Node*> retired_head_;     // nodes unlinked since the last advance, linked through `retire_link`
    alignas(128) std::atomic<Node*> free_head_;        // nodes whose values are destroyed, ready for reuse, linked through `retire_link`
    alignas(128) mutable Node head_;                   // the dummy head: never erased, retired or freed; mutable: a const list's const_iterator still holds a Node*
#ifndef NDEBUG
    std::atomic<size_t> allocated_;                    // debug only: nodes from `new` not yet deleted (InternalAccounting::allocated)
    static inline std::atomic<uint64_t> next_session_id_; // debug only: source of the globally unique handle session ids
#endif
}; // LockFreeListRCU

#endif // INCLUDED_LOCK_FREE_LIST_RCU_H
