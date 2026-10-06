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
#ifndef INCLUDED_INTR_SHARED_PTR_H_
#define INCLUDED_INTR_SHARED_PTR_H_

#include <atomic>
#include <cstdint>
#include <thread>
#include <cstddef>
#include <type_traits>
#include <time.h>
#include <sched.h>

#include "intr_pointee.h"           // IntrusivePointee, intr_pointee_base
#include "intr_shared_ptr_common.h" // the value type and helpers shared with intr_shared_ptr_hp

// intr_shared_ptr: an atomic, intrusively-reference-counted shared pointer with
// support for Harris-style pointer marking. It conforms to the AtomicSharedPtr
// concept (../SharedPtr/atomic_shared_ptr_concept.h) so it can parameterize
// LockFreeList. It shares its value type (shared_ptr_type) and the helpers
// around it with intr_shared_ptr_hp through intr_shared_ptr_common.h; the
// atomic operations below are its own.
//
// Template parameters:
//   T - the interface type seen through the smart pointer: operator*, operator->
//       and get() return T.
//   U - the concrete stored type (defaults to T). The atomic word holds a U*
//       (plus the mark and lock bits); U* must convert implicitly to T* (U is T
//       or is publicly derived from T). U carries the intrusive count: it must
//       satisfy IntrusivePointee<U> (intr_pointee.h) -- the three noexcept
//       hooks AddRef(), DelRef(), use_count(), with the orders stated there --
//       either hand-written or inherited from intr_pointee_base<>. U is the
//       DYNAMIC type of every pointee: the last release runs `delete p` on a
//       U*, so an object of a type derived from U is destroyed through U's
//       destructor (virtual or not). T and U differ when the pointee is
//       exposed through a base interface T while the hooks live on a derived U
//       (see the <A,B> tests). The concept is checked by static_assert inside
//       the member functions (require_pointee()), never at class scope, so
//       that a type may name intr_shared_ptr<Self> while Self is incomplete
//       (LockFreeList::Node); the diagnostic appears at the first member use.
//
// Bit layout of the stored uintptr_t (U must be at least 4-byte aligned so that
// the two low bits of a U* are free to steal -- a checked requirement, not an
// assumption: static_assert(alignof(U) >= 4) in
// intr_shared_ptr_detail::delete_disposal::require_pointee<U>(), reached from
// common_base::require_pointee() at every check point):
//   bit 0 - mark bit  (Harris logical-deletion tag; part of the value/identity)
//   bit 1 - lock bit  (an internal spinlock guarding the load->AddRef gap)
//   bits 2+ - the actual U* pointer
//
// Why a spinlock: an atomic shared pointer must atomically (load a pointer AND
// AddRef the object it points to). A lock-free design needs hazard pointers or
// DCAS to close that window (intr_shared_ptr_hp does the former); here we
// instead take a one-bit spinlock embedded in the pointer word for the brief
// critical section. The *list* built on top stays lock-free in the algorithmic
// sense even though this pointer is not.
//
// Memory-ordering contract. Every operation on the word -- load() included --
// takes the embedded lock with an acquire compare-exchange (relaxed on
// failure, relaxed spins) and leaves it with a release store, so every
// operation is at least acquire-release and synchronizes with the operation
// that last released the lock; in particular a load() synchronizes with the
// store() or CAS that published the word it returns, and a store()/CAS
// synchronizes with the publisher of the old occupant it releases. The
// `order` arguments are accepted with ANY of the six std::memory_order values
// on every operation (std::atomic would reject acquire/acq_rel/consume on a
// store; here they are harmless), and no `order` makes an operation weaker
// than the lock makes it:
//   load(order)      `order` is ignored: the lock's acquire CAS and the
//                    release store that frees it bound the read. load() is
//                    NOT a seq_cst operation even for load(seq_cst) -- it takes
//                    no place in the single total order of seq_cst operations
//                    (it is an acquire RMW followed by a release store).
//   store(d, order)  the publishing store (which doubles as the unlock) is
//                    seq_cst when `order` is seq_cst and release otherwise:
//                    relaxed, consume, acquire, acq_rel and release all mean
//                    release here. store(relaxed) is therefore safe and legal;
//                    it is promoted, not honoured.
//   compare_exchange_strong(e, d, success, failure)
//                    on success the publishing store is seq_cst when `success`
//                    is seq_cst and release otherwise (same promotion); on
//                    failure `failure` is ignored (the unlock is a release
//                    store); the replacement reference is taken under the
//                    lock, and `expected` is assigned (releasing its previous
//                    pointee) after the unlock. Strong semantics: compare and
//                    swap happen under the lock as one step, so the CAS never
//                    fails with `expected` unchanged, and there is no weak
//                    variant.
// Compared with intr_shared_ptr_hp, which follows std::atomic's rules for
// which orders each operation accepts and honours load(seq_cst): this pointer
// is simpler and stronger in effect -- everything is acquire-release through
// the lock -- but offers no usable seq_cst total order: load() takes no part in
// it (store(seq_cst) and a successful seq_cst CAS do publish with a seq_cst
// store).
//
// Reclamation is synchronous: the release that drops a pointee's count to 0
// deletes it then and there, on the releasing thread. The releases are: a
// shared_ptr_type destructor or assignment (the assignment of `expected` on a
// CAS failure included), store(), and ~intr_shared_ptr. A successful CAS also
// releases the old occupant, but that release can never be the last one:
// `expected` compared equal to the word and still owns its reference.
//
// Exceptions: nothing here allocates or throws, and every member is noexcept
// in effect; the ones declared noexcept are the two trivial constructors of
// the atomic and all of shared_ptr_type's members (intr_shared_ptr_common.h).
// Precondition on U, checked at compile time (require_pointee()): the hooks
// are noexcept (the concept) and ~U is nothrow -- destructors do not throw,
// and a pointee whose destructor is declared noexcept(false) is rejected.

namespace intr_shared_ptr_detail {

// The Disposal policy of intr_shared_ptr for common_base (see
// intr_shared_ptr_common.h): the pointee concept it asserts, and the release
// of a reference with `delete` as what happens to an object whose count
// reached 0.
struct delete_disposal {
    // The pointer's pointee concept, with its own diagnostic.
    template <typename U>
    static constexpr void require_pointee() noexcept {
        static_assert(IntrusivePointee<U>,
            "intr_shared_ptr<T, U>: U must provide noexcept AddRef(), bool DelRef() and "
            "long use_count() const (intr_pointee.h; intr_pointee_base<> provides them)");
        // The word steals bits 0 (mark) and 1 (lock) of the U*: they must be
        // zero in every object's address. The common base checks bit 0's
        // half (alignof >= 2) for both pointers; this is the lock bit's.
        static_assert(alignof(U) >= 4,
            "intr_shared_ptr<T, U>: alignof(U) must be at least 4, the mark and lock bits "
            "are the two low bits of the stored U* (an alignas(4) on U is enough)");
    } // delete_disposal::require_pointee()

    // Release one reference to the unmarked pointee p (null: no-op): DelRef,
    // and on 1->0 delete the object, here and now -- synchronous reclamation.
    // noexcept: DelRef is by contract, and ~U is required to be
    // (common_base::require_pointee() rejects a U whose destructor may throw).
    //
    // always_inline because this must compile to the in-place expression
    // `if (p && p->DelRef()) delete p;`. Left to its heuristics, GCC at -O3
    // splits this function and calls a two-instruction clone (the DelRef and
    // a tail call to operator delete) on EVERY non-null release of the
    // spinlock pointer -- a call to make a call, on the hot path of every
    // handle destructor and assignment. (retire_disposal has no such
    // attribute: there the tail is the whole retire(), which GCC outlines
    // into a clone with or without the shared helper, and that is a
    // reasonable code-size decision to leave to the compiler.)
    template <typename U>
    [[gnu::always_inline]] static void release(U* p) noexcept {
        if (p && p->DelRef()) delete p;
    }
}; // struct delete_disposal

} // namespace intr_shared_ptr_detail

template <typename T, typename U = T>
class intr_shared_ptr
    : private intr_shared_ptr_detail::common_base<T, U, intr_shared_ptr_detail::delete_disposal> {
    using base = intr_shared_ptr_detail::common_base<T, U, intr_shared_ptr_detail::delete_disposal>;
    using disposal = intr_shared_ptr_detail::delete_disposal;
    // The helpers shared with intr_shared_ptr_hp (intr_shared_ptr_common.h):
    // the pointee check and the word <-> pointer <-> handle conversions.
    using base::require_pointee;
    using base::unmarked_ptr;
    using base::adopt_word;

public:
    // The non-atomic "value" type handed out by load() and accepted by store()/CAS.
    // It owns a strong reference to its pointee (AddRef on acquire, DelRef on
    // release, delete on 1->0) and may carry the mark bit in bit 0 of its raw
    // pointer. Documented in intr_shared_ptr_common.h.
    using typename base::shared_ptr_type;

    static constexpr bool supports_marking = true;

    constexpr intr_shared_ptr() noexcept : aptr_(0) {}
    explicit(false) intr_shared_ptr(std::nullptr_t) noexcept : aptr_(0) {}

    // Take ownership of `desired`'s pointee. `desired` keeps its own reference
    // (it is taken by value), so we AddRef once more for the reference the atomic
    // itself now holds. The mark bit (if any) is preserved.
    explicit(false) intr_shared_ptr(shared_ptr_type desired) {
        require_pointee();
        uintptr_t val = reinterpret_cast<uintptr_t>(desired.get_raw());
        U* new_unmarked = unmarked_ptr(desired);
        if (new_unmarked) new_unmarked->AddRef();
        aptr_.store(val, std::memory_order_relaxed);
    }

    intr_shared_ptr(const intr_shared_ptr&) = delete;
    intr_shared_ptr& operator=(const intr_shared_ptr&) = delete;

    // ~3: the mark bit may legitimately be set (a marked pointer is still an
    // owning pointer and must be released); the lock bit cannot be -- running
    // the destructor means no operation is in flight -- so stripping it is
    // defensive. Relaxed suffices for the same reason: whatever established
    // that exclusivity already ordered all prior accesses to aptr_.
    ~intr_shared_ptr() {
        require_pointee();
        uintptr_t val = aptr_.load(std::memory_order_relaxed);
        disposal::release(reinterpret_cast<U*>(val & ~3ULL));
    }

    // Atomically snapshot the pointer and bump its refcount so the returned value
    // keeps the pointee alive. We hold the spinlock across the load+AddRef so no
    // concurrent store/CAS can DelRef the object to zero in between. `order` is
    // accepted for interface compatibility but ignored: the spinlock's acquire CAS
    // / release store already impose at-least-acquire ordering on the snapshot.
    shared_ptr_type load([[maybe_unused]] std::memory_order order = std::memory_order_seq_cst) const {
        require_pointee();
        uintptr_t val = lock<Intent::Read>();
        // Strip the lock bit, keep the mark (lock() already cleared it; defensive).
        // The handle is built BEFORE the AddRef, its reference being the one
        // taken next; nothing can see the handle in between (it is a local, and
        // the lock is held), so the order is immaterial.
        shared_ptr_type res = adopt_word(val & ~2ULL);
        U* x = unmarked_ptr(res);
        if (x) x->AddRef();
        unlock(val); // Restore exact mark state
        return res;
    }

    // Replace the held pointer with `desired` (mark and all), releasing the old one.
    // The publishing store doubles as the spinlock release, so it must carry release
    // ordering: `order` is promoted to release unless it is seq_cst (any of the six
    // orders is accepted; store(relaxed) is legal and means release here -- see the
    // memory-ordering contract above).
    void store(shared_ptr_type desired, std::memory_order order = std::memory_order_seq_cst) {
        require_pointee();
        U* new_ptr = desired.get_raw();
        U* new_unmarked = unmarked_ptr(desired);
        if (new_unmarked) new_unmarked->AddRef();

        uintptr_t val = lock<Intent::Write>();
        U* old_ptr = reinterpret_cast<U*>(val & ~3ULL); // Only the actual ptr

        // Ensure we always have release semantics to unlock the spinlock properly.
        // We override the user's requested order because:
        // 1. If `order` lacks release semantics (e.g. relaxed), the spinlock would be released
        //    without proper synchronization, potentially exposing stale memory to other threads.
        // 2. An acquire/acq_rel/consume `order` (which std::atomic::store would reject) is
        //    accepted and likewise mapped to release: the lock makes every store acquire
        //    anyway, so nothing is lost.
        std::memory_order store_order = (order == std::memory_order_seq_cst) ? std::memory_order_seq_cst : std::memory_order_release;
        aptr_.store(reinterpret_cast<uintptr_t>(new_ptr), store_order);

        disposal::release(old_ptr);
    }

    // Compare-and-swap on the *full* value including the mark bit: the swap only
    // succeeds if both the pointer and its mark match `expected`. On success
    // `expected` is UNTOUCHED (it keeps its reference to the old occupant, whose
    // release here therefore never destroys it); `desired` is unchanged either
    // way. On failure, `expected` is updated to the current value (with a fresh
    // reference) to mirror std::atomic<shared_ptr> semantics -- the one place a
    // CAS can run the last release of an object: expected's previous pointee.
    // Std semantics: the lock makes the compare and the swap one step, so this
    // never fails with `expected` equal to the value the caller passed in.
    // Orders (the contract above): `success` is promoted to release unless
    // seq_cst; `failure` is unused because the critical section is bracketed by
    // the spinlock's acquire CAS and release store.
    bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired,
                                 std::memory_order success = std::memory_order_seq_cst,
                                 [[maybe_unused]] std::memory_order failure = std::memory_order_seq_cst) {
        require_pointee();
        uintptr_t expected_val = reinterpret_cast<uintptr_t>(expected.get_raw());
        uintptr_t new_val = reinterpret_cast<uintptr_t>(desired.get_raw());

        uintptr_t val = lock<Intent::Write>();
        // Full-identity comparison: pointer AND mark bit must match. The ~2
        // is defensive only -- lock() returns the pre-lock value, whose lock
        // bit is already clear.
        if ((val & ~2ULL) == expected_val) {
            // Success
            U* new_unmarked = unmarked_ptr(desired);
            if (new_unmarked) new_unmarked->AddRef();

            U* old_unmarked = reinterpret_cast<U*>(val & ~3ULL);

            // Ensure store_order is valid for atomic store AND has release semantics for the spinlock.
            // We override the user's `success` order because:
            // 1. std::atomic::store() panics/UBs if passed acquire or acq_rel (which are valid for CAS).
            // 2. If `success` lacks release semantics (e.g. relaxed), the spinlock would be released
            //    without synchronizing memory, allowing races on the pointee object.
            std::memory_order store_order = (success == std::memory_order_seq_cst) ? std::memory_order_seq_cst : std::memory_order_release;
            // Like store(): new_val has the lock bit clear, so this single
            // store is both the CAS write and the spinlock release -- which is
            // why there is no unlock() call on the success path.
            aptr_.store(new_val, store_order);

            disposal::release(old_unmarked);
            return true;
        } else {
            // Failure: hand the caller the current value with its own
            // reference, taken under the lock like load() does.
            shared_ptr_type res = adopt_word(val & ~2ULL);
            U* x = unmarked_ptr(res);
            if (x) x->AddRef();
            unlock(val);

            expected = std::move(res);
            return false;
        }
    }

private:
    // The entire atomic state in one word: U* | mark (bit 0) | lock (bit 1).
    // mutable because load() is const from the caller's viewpoint yet must
    // take the embedded spinlock, which mutates the word.
    mutable std::atomic<uintptr_t> aptr_{0};

    // Selects the backoff policy in lock(): readers (load) burn CPU longer
    // before sleeping, writers (store/CAS) get out of the way almost
    // immediately -- see the comments in lock().
    enum class Intent { Read, Write };

    // Acquire the embedded spinlock (bit 1). Spins until it observes the lock bit
    // clear and wins the CAS that sets it. Returns the value as it was *before*
    // locking (lock bit clear, pointer + mark intact) so callers can both inspect
    // the current pointer and reconstruct the exact state to publish on unlock().
    template <Intent intent>
    uintptr_t lock() const {
        int spin_count = 0;
        uintptr_t val = aptr_.load(std::memory_order_relaxed);
        while (true) {
            if ((val & 2ULL) == 0) {
                // acquire on success pairs with the release in unlock()/store()/CAS.
                if (aptr_.compare_exchange_weak(val, val | 2ULL, std::memory_order_acquire, std::memory_order_relaxed)) {
                    return val; // Returns the old value (without lock bit)
                }
                // Spurious/contended failure: `val` was refreshed by the CAS; retry.
            } else {
                if constexpr (intent == Intent::Read) {
                    for (int i = 0; i < 8; ++i) {
                        val = aptr_.load(std::memory_order_relaxed);
                        if ((val & 2ULL) == 0) break;
#if defined(__x86_64__) || defined(__i386__)
                        __builtin_ia32_pause();
#elif defined(__aarch64__)
                        __asm__ volatile("yield" ::: "memory");
#endif
                    }
                } else {
                    // Writers get a bare recheck loop (no pause): a short,
                    // cheap last chance to see the lock freed before paying
                    // for the nanosleep syscall below.
                    for (int i = 0; i < 8; ++i) {
                        val = aptr_.load(std::memory_order_relaxed);
                        if ((val & 2ULL) == 0) break;
                    }
                }

                if ((val & 2ULL) != 0) {
                    if constexpr (intent == Intent::Read) {
                        // Readers stay out of the kernel as long as possible
                        if (spin_count < 32) {
                            std::this_thread::yield();
                            ++spin_count;
                        } else {
                            spin_count = 0;
                            struct timespec ts = { 0, 1000 }; // nominal 1us; see the note on nanosleep rounding below
                            nanosleep(&ts, nullptr);
                        }
                    } else {
                        // Writers sleep immediately to let the lock-holder
                        // finish and avoid CAS thrashing
                        if (spin_count < 8) {
                            // Nominal 1ns, but the kernel rounds nanosleep up
                            // to its timer granularity plus slack (~50us by
                            // default on Linux): the request really means "the
                            // shortest sleep available" -- the point is to
                            // deschedule, not the stated duration.
                            struct timespec ts = { 0, 1 };
                            nanosleep(&ts, nullptr);
                            ++spin_count;
                        } else {
                            // Escalation tier: insurance for a lock holder
                            // descheduled mid-critical-section. The 8 short
                            // sleeps above absorb virtually all waits, so
                            // this branch is rare even under heavy
                            // contention, and its duration does not move
                            // throughput over a wide range. 1ms bounds the
                            // latency a stale sleeper adds while still giving
                            // a preempted holder time to run.
                            spin_count = 0;
                            struct timespec ts = { 0, 1000000 }; // 1ms
                            nanosleep(&ts, nullptr);
                        }
                    }
                    val = aptr_.load(std::memory_order_relaxed);
                }
            }
        }
    }

    // Release the spinlock by republishing the pre-lock value (which clears bit 1).
    void unlock(uintptr_t old_val_without_lock) const {
        aptr_.store(old_val_without_lock, std::memory_order_release);
    }
}; // class intr_shared_ptr

#endif // INCLUDED_INTR_SHARED_PTR_H_
