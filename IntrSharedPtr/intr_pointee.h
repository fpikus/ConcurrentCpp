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
#ifndef INCLUDED_INTR_POINTEE_H_
#define INCLUDED_INTR_POINTEE_H_

#include <atomic>
#include <concepts>
#include <type_traits>

// The pointee side of the contract of the intrusive atomic shared pointers
// intr_shared_ptr (intr_shared_ptr.h; the load->AddRef gap closed by a one-bit
// spinlock) and intr_shared_ptr_hp (intr_shared_ptr_hp.h; closed by a hazard
// pointer). Both keep the strong count INSIDE the pointee and drive it through
// member functions of the pointee, the "hooks". This header is the ONE
// normative statement of what the hooks must do -- every other place that
// implements or describes them (the pointers, LockFreeList::Node, the tests)
// refers here -- and it offers a base class that implements them:
//
//   - IntrusivePointee<U>: the three hooks intr_shared_ptr requires.
//     intr_shared_ptr_hp requires a fourth, TryAddRef(), and a hazard pointer
//     base: HpIntrusivePointee<U>, in intr_shared_ptr_hp.h.
//   - intr_pointee_base<Count>: a base class providing all four hooks over an
//     atomic count of type Count, usable with BOTH pointers (the spinlock
//     pointer never calls TryAddRef; it costs the pointee nothing). The hazard
//     pointer variant, intr_pointee_base_hp<U, Count> (intr_shared_ptr_hp.h),
//     adds the hazard pointer base.
//
// The base is a convenience, not a requirement: a pointee may implement the
// hooks itself (LockFreeList::Node does, and so do the test pointees that
// instrument them), as long as every hook has exactly the semantics and the
// memory orders stated below. The pointers check only what a concept can see
// (names, signatures, noexcept); the orders are a contract enforced by review.
//
// ---------------------------------------------------------------------------
// The hooks (normative)
// ---------------------------------------------------------------------------
//
// The count is one atomic integer per object; "the count" below means its
// value. All four hooks are noexcept (a release -- a destructor, an
// assignment, store(), a CAS -- must not throw, and the pointers declare their
// own guarantees on that basis). EVERY operation that changes the count is an
// atomic read-modify-write; a plain store to the count (even "count = 1" on a
// fresh object, after construction) is forbidden, because it would end the
// release sequence that TryAddRef's acquire reads synchronize with. The count
// starts at 0 for a freshly constructed object (owned by no one) and is taken
// to 1 by its first owner (the pointers' shared_ptr_type(U*)); 0 is reached
// again exactly once, by the DelRef() that returns true, whose caller disposes
// of the object (delete for intr_shared_ptr, retire() for intr_shared_ptr_hp).
//
//   void AddRef() noexcept;
//       Increment the count. Precondition: the caller owns a reference, or
//       the object is fresh (count 0, never owned), so that a plain increment
//       never resurrects a released object; violated, the object is destroyed
//       (or retired) under a live owner, or destroyed twice -- undefined
//       behaviour. A relaxed RMW suffices (the caller's own reference pins
//       the object; nothing is published through this increment). A return
//       value, if any, is ignored.
//
//   bool DelRef() noexcept;
//       Decrement the count; return true iff THIS call made the 1 -> 0
//       transition. Precondition: the caller owns a reference (count >= 1);
//       violated, the count wraps or goes negative and the object is destroyed
//       under its live owners or never -- undefined behaviour. The RMW is
//       ACQ_REL, and both halves are load-bearing:
//       release heads the release sequence that TryAddRef's acquire reads
//       synchronize with (an object's count reaches 0 only through a DelRef);
//       acquire gives the thread that reaches 0 -- and therefore destroys or
//       retires the object -- the history of every thread that released a
//       reference before it, in particular of the thread that unpublished the
//       object from an atomic word (unpublish happens-before destroy; the
//       hazard pointer safety argument needs exactly this, and so does the
//       destructor reading fields other owners wrote).
//
//   long use_count() const noexcept;
//       The current count, as long whatever the count's type: a relaxed
//       snapshot for diagnostics and tests, stale by the time it returns.
//       The pointers never make a decision on it; LockFreeList::~Node does
//       (its exclusivity test, which rests on the pointers' own invariants,
//       not on this read's order).
//
//   bool TryAddRef() noexcept;     (required by intr_shared_ptr_hp only)
//       Increment the strong count if and only if it is nonzero. Returns true
//       iff it incremented; returns false iff it observed a count of 0, in
//       which case the count is left at 0 (an object at 0 is retired, or about
//       to be retired, and must never be revived). Required only by
//       intr_shared_ptr_hp; the other pointer policies never call it. Memory
//       orders, all load-bearing: the load that observes 0 is ACQUIRE; the CAS
//       is ACQUIRE on success and relaxed on failure; a failed CAS whose
//       refreshed value is 0 re-reads the count with an acquire load before
//       returning false, so EVERY observed 0 was read with acquire. Why:
//       intr_shared_ptr_hp::load() calls this on an object pinned only by a
//       hazard pointer. An observed 0 must synchronize with the release
//       sequence headed by the DelRef that produced it, so that the loader's
//       next acquire reload of the word is guaranteed to see the store that
//       unpublished the object; with a relaxed zero-observation the loader can
//       re-read the stale word forever (livelock). The ACQUIRE on CAS success
//       makes the loader's post-increment re-validation of the word see a swing
//       that released the word's own reference. The value a successful CAS
//       consumes need not have been read with acquire: a failed CAS refreshes
//       it with a relaxed read (only a refreshed 0 is re-read with acquire),
//       and other threads' RMWs may have rewritten it. A CAS that succeeds with
//       relaxed order on a value it read from the release sequence headed by
//       the swing's DelRef does not synchronize with that DelRef, so the
//       re-validation reload may return the stale word and
//       LockFreeList::~Node's walk can judge a node exclusive (count 1) that
//       this loader then owns with a stale next. Every operation on the count
//       is an RMW. Cost: nil on x86-64; LDAR/LDAXR on aarch64.
//
//       The loop below (intr_pointee_base::TryAddRef) is the REFERENCE FORM of
//       this contract: acquire initial load; CAS acquire/relaxed; a failed
//       CAS's relaxed-refreshed NONZERO value feeds the next CAS directly; a
//       refreshed 0 is re-read with acquire. Order by order: the acquire
//       INITIAL load (relaxed livelocks) and the acquire CAS SUCCESS (relaxed
//       lets the destructor walk judge a node exclusive, above) are
//       load-bearing. The acquire RE-READ of a refreshed 0 is a choice, not a
//       necessity: the form that returns false on the relaxed 0 is also safe
//       and lock-free -- the next protocol round's initial acquire load must,
//       by coherence, read 0 (0 is terminal) and synchronize then -- at the
//       cost of one wasted round (hazard store, fence, two reloads) instead of
//       one acquire load, and a two-step proof instead of one. The reference
//       form keeps the re-read so that "every observed 0 was read with
//       acquire" holds locally in the hook.

// IntrusivePointee<U>: what intr_shared_ptr requires of its stored type U --
// the three hooks AddRef(), DelRef(), use_count(), all noexcept, with the
// semantics and orders stated above (which no concept can check). TryAddRef()
// is not required: the spinlock pointer never takes a reference it cannot
// already account for. intr_shared_ptr checks this concept by static_assert
// inside its member function bodies, not at class scope, so that a type may
// name intr_shared_ptr<Self> while Self is still incomplete (a list node
// holding its own `next`); the diagnostic therefore appears at the first use
// of a member, not at the declaration of the atomic.
template <typename U>
concept IntrusivePointee = requires(U& u, const U& cu) {
    { u.AddRef() } noexcept;
    { u.DelRef() } noexcept -> std::same_as<bool>;
    { cu.use_count() } noexcept -> std::same_as<long>;
};

// intr_pointee_base<Count>: the four hooks over one std::atomic<Count>,
// implemented exactly as the contract above states (AddRef relaxed, DelRef
// acq_rel, use_count a relaxed load, TryAddRef the reference form). Derive from
// it publicly:
//
//   struct Node : intr_pointee_base<> {          // long count
//       intr_shared_ptr<Node> next;
//       ...
//   };
//
// Count: the count's integer type, long by default (the type use_count()
// returns). Requirements, enforced at compile time: an integral type other than
// bool, not cv-qualified (the requires-clause: a violation is "constraints not
// satisfied" at the point of use), and std::atomic<Count> lock-free (the
// class-scope static_assert below, which fires when the class is instantiated;
// the pointers' progress guarantees rest on the count's RMWs being lock-free).
// A narrower type saves space in the pointee at the price of a BOUND the
// program must respect: never more than std::numeric_limits<Count>::max()
// simultaneous owners (handles plus atomic words) of one object. Beyond it the
// count wraps (unsigned) or overflows (signed, undefined behaviour); in
// practice the wrapped count reaches 0 or 1 early, so a release destroys or
// retires the object under live owners (a use-after-free), and under
// intr_shared_ptr_hp a count that wrapped to 0 makes every load() of the
// object retry forever (TryAddRef never increments from 0). The count type is
// not checked against the number of owners at run time. For a Count whose
// range exceeds long's (unsigned long; long long where it is wider than long),
// use_count() converts to long and is exact only up to
// std::numeric_limits<long>::max(); larger values read back wrapped (modulo
// 2^N, often as a negative long) -- unreachable in practice, stated for
// completeness. The count is NOT a customization point beyond its type: the
// orders are fixed by the contract. A pointee that needs instrumented hooks
// (the test suites do) writes its own.
//
// Special members, all protected (the class is a base and never the dynamic
// type of anything; its destructor is non-virtual on purpose, like
// std::hazard_pointer_obj_base's) and all noexcept:
//   - default constructor: count 0, constexpr.
//   - copy constructor: count 0 -- a copy of a pointee is a NEW object that
//     nobody owns yet; copying the count would let the copy be destroyed or
//     kept alive by its original's owners. (boost::intrusive_ref_counter has
//     the same semantics.)
//   - copy assignment: leaves *this's count untouched -- assigning a value
//     into a pointee does not change who owns the pointee.
//   - move operations: not declared; a move falls back to the copy
//     operations above, with the same meaning (the source's count is
//     untouched as well, since its owners still own it).
// Deriving from this base therefore makes a type copyable only if its other
// members are; a pointee that must not be copied deletes its own copy
// operations as usual.
template <typename Count = long>
    requires std::integral<Count> && (!std::same_as<Count, bool>) &&
             std::same_as<Count, std::remove_cv_t<Count>>
class intr_pointee_base {
    static_assert(std::atomic<Count>::is_always_lock_free,
        "intr_pointee_base<Count>: std::atomic<Count> must be lock-free");

public:
    // AddRef(): increment, relaxed RMW. See the contract above.
    void AddRef() noexcept { count_.fetch_add(1, std::memory_order_relaxed); }

    // TryAddRef(): increment iff nonzero -- the reference form of the contract
    // above; false (count left at 0) iff it observed 0, and every observed 0
    // was read with acquire. [[nodiscard]]: a caller that ignores the result
    // has either leaked a reference or touched a dead object.
    [[nodiscard]] bool TryAddRef() noexcept {
        Count count = count_.load(std::memory_order_acquire);
        while (count != 0) {
            if (count_.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return true;
            }
            // The failed CAS refreshed `count` with a relaxed read; a 0 seen
            // that way does not synchronize. Re-read it with acquire.
            if (count == 0) count = count_.load(std::memory_order_acquire);
        } // CAS loop while the count is nonzero
        return false;
    } // intr_pointee_base::TryAddRef()

    // DelRef(): decrement, acq_rel RMW; true iff this call made the 1 -> 0
    // transition. [[nodiscard]]: ignoring the result leaks the object (nobody
    // disposes of it) -- the caller that drops a count to 0 must dispose.
    [[nodiscard]] bool DelRef() noexcept {
        return count_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    // use_count(): relaxed snapshot of the count, widened to long.
    [[nodiscard]] long use_count() const noexcept {
        return static_cast<long>(count_.load(std::memory_order_relaxed));
    }

protected:
    constexpr intr_pointee_base() noexcept = default;
    // A copy is a fresh object: count 0 (see the class comment).
    constexpr intr_pointee_base(const intr_pointee_base&) noexcept {}
    // Assignment does not touch the count: *this keeps its owners.
    constexpr intr_pointee_base& operator=(const intr_pointee_base&) noexcept { return *this; }
    ~intr_pointee_base() = default;

private:
    // The strong count: the number of owners (shared_ptr_type values and
    // atomic words) of the object this base is part of; 0 until the first
    // owner adopts the object, 0 again exactly once, when the last owner
    // releases it. Only ever changed by RMWs (the contract above).
    std::atomic<Count> count_{0};
}; // class intr_pointee_base

#endif // INCLUDED_INTR_POINTEE_H_
