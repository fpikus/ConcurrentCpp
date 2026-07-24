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
#ifndef INCLUDED_INTR_SHARED_PTR_H_
#define INCLUDED_INTR_SHARED_PTR_H_

#include <atomic>
#include <cstdint>
#include <thread>
#include <cstddef>
#include <time.h>
#include <sched.h>

// intr_shared_ptr: an atomic, intrusively-reference-counted shared pointer with
// support for Harris-style pointer marking. It conforms to the AtomicSharedPtr
// concept (see atomic_shared_ptr_concept.h) so it can parameterize LockFreeList.
//
// Template parameters:
//   T - the interface type seen through the smart pointer (operator*/-> return T).
//   U - the concrete stored type (defaults to T). U must provide the intrusive
//       reference-counting hooks:
//         void AddRef();      // increment the strong count
//         bool DelRef();      // decrement; return true iff this call dropped it to 0
//         long use_count();   // current strong count (for diagnostics/tests)
//       T and U differ when the pointee is exposed through a base interface T while
//       the refcount hooks live on a derived U (see the <A,B> tests).
//
// Bit layout of the stored uintptr_t (objects are at least 4-byte aligned, so the
// two low bits are free to steal):
//   bit 0 - mark bit  (Harris logical-deletion tag; part of the value/identity)
//   bit 1 - lock bit  (an internal spinlock guarding the load->AddRef gap)
//   bits 2+ - the actual U* pointer
//
// Why a spinlock: an atomic shared pointer must atomically (load a pointer AND
// AddRef the object it points to). A lock-free design needs hazard pointers or
// DCAS to close that window; here we instead take a one-bit spinlock embedded in
// the pointer word for the brief critical section. The *list* built on top stays
// lock-free in the algorithmic sense even though this pointer is not.
template <typename T, typename U = T>
class intr_shared_ptr {
public:
    // The non-atomic "value" type handed out by load() and accepted by store()/CAS.
    // It owns a strong reference to its pointee (AddRef on acquire, DelRef on
    // release) and may carry the mark bit in bit 0 of its raw pointer.
    class shared_ptr_type {
    public:
        shared_ptr_type() : p_(nullptr) {}
        shared_ptr_type(std::nullptr_t) : p_(nullptr) {}
        // Adopt a raw pointer (0 -> 1 for a freshly new'ed object). Unlike
        // std::shared_ptr, adoption and sharing are the same operation here:
        // the count lives inside the object, so every new owner just AddRefs
        // and there is no separate control block to allocate or find.
        explicit shared_ptr_type(U* p) : p_(p) {
            if (get_unmarked_ptr()) get_unmarked_ptr()->AddRef();
        }
        
        shared_ptr_type(const shared_ptr_type& x) : p_(x.p_) {
            if (get_unmarked_ptr()) get_unmarked_ptr()->AddRef();
        }
        
        shared_ptr_type(shared_ptr_type&& x) noexcept : p_(x.p_) {
            x.p_ = nullptr;
        }

        ~shared_ptr_type() {
            U* ptr = get_unmarked_ptr();
            if (ptr && ptr->DelRef()) {
                delete ptr;
            }
        }
        
        // AddRef the new pointee *before* DelRef-ing the old one: `this` and `x`
        // can be distinct shared_ptr_type objects holding the same pointee --
        // often as marked and unmarked variants of one pointer, which the
        // mark-as-identity design (get_unmarked(), set_mark() return copies)
        // makes routine. Releasing first could drop the last reference and
        // delete the very object we are about to AddRef; the self-assignment
        // check catches only `a = a`, not that aliasing.
        shared_ptr_type& operator=(const shared_ptr_type& x) {
            if (this == &x) return *this;
            U* new_ptr = reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(x.p_) & ~1ULL);
            if (new_ptr) new_ptr->AddRef();
            
            U* old_ptr = get_unmarked_ptr();
            if (old_ptr && old_ptr->DelRef()) {
                delete old_ptr;
            }
            p_ = x.p_;
            return *this;
        }

        // Move assignment needs no AddRef-first dance: x's reference is being
        // transferred, so even when *this and x share the pointee the count
        // includes x's reference until p_ is overwritten and stays >= 1
        // through the DelRef below.
        shared_ptr_type& operator=(shared_ptr_type&& x) noexcept {
            if (this == &x) return *this;
            U* old_ptr = get_unmarked_ptr();
            if (old_ptr && old_ptr->DelRef()) {
                delete old_ptr;
            }
            p_ = x.p_;
            x.p_ = nullptr;
            return *this;
        }

        T& operator*() const { return *get_unmarked_ptr(); }
        T* operator->() const { return get_unmarked_ptr(); }
        T* get() const { return get_unmarked_ptr(); }
        explicit operator bool() const { return get_unmarked_ptr() != nullptr; }

        bool operator==(const shared_ptr_type& rhs) const { return p_ == rhs.p_; }
        bool operator!=(const shared_ptr_type& rhs) const { return p_ != rhs.p_; }

        long use_count() const {
            U* ptr = get_unmarked_ptr();
            return ptr ? ptr->use_count() : 0;
        }

        // Harris marking API. The mark lives in bit 0 of the raw pointer and is
        // considered part of the pointer's identity (operator== compares it too),
        // so a marked and unmarked pointer to the same object are *not* equal.
        bool is_marked() const { return (reinterpret_cast<uintptr_t>(p_) & 1ULL) != 0; }

        // Two overloads: the lvalue one must copy (an AddRef/DelRef round
        // trip); the rvalue one clears the bit in place, which is what makes
        // the ubiquitous `next.load(...).get_unmarked()` refcount-churn-free.
        shared_ptr_type get_unmarked() const & {
            shared_ptr_type res(*this);
            res.p_ = reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(res.p_) & ~1ULL);
            return res;
        }

        shared_ptr_type get_unmarked() && {
            p_ = reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(p_) & ~1ULL);
            return std::move(*this);
        }
        
        shared_ptr_type set_mark() const {
            shared_ptr_type res(*this);
            res.p_ = reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(res.p_) | 1ULL);
            return res;
        }

        // Raw pointer with the mark bit still attached (used by the atomic owner).
        U* get_raw() const { return p_; }

    private:
        friend class intr_shared_ptr;
        U* p_;

        // The pointer with the mark bit cleared, safe to dereference / refcount.
        U* get_unmarked_ptr() const {
            return reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(p_) & ~1ULL);
        }
    };

    static constexpr bool supports_marking = true;

    constexpr intr_shared_ptr() noexcept : aptr_(0) {}
    explicit(false) intr_shared_ptr(std::nullptr_t) noexcept : aptr_(0) {}

    // Take ownership of `desired`'s pointee. `desired` keeps its own reference
    // (it is taken by value), so we AddRef once more for the reference the atomic
    // itself now holds. The mark bit (if any) is preserved.
    explicit(false) intr_shared_ptr(shared_ptr_type desired) {
        uintptr_t val = reinterpret_cast<uintptr_t>(desired.get_raw());
        if (desired.get_unmarked_ptr()) desired.get_unmarked_ptr()->AddRef();
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
        uintptr_t val = aptr_.load(std::memory_order_relaxed);
        U* ptr = reinterpret_cast<U*>(val & ~3ULL);
        if (ptr && ptr->DelRef()) {
            delete ptr;
        }
    }

    // Atomically snapshot the pointer and bump its refcount so the returned value
    // keeps the pointee alive. We hold the spinlock across the load+AddRef so no
    // concurrent store/CAS can DelRef the object to zero in between. `order` is
    // accepted for interface compatibility but ignored: the spinlock's acquire CAS
    // / release store already impose at-least-acquire ordering on the snapshot.
    shared_ptr_type load([[maybe_unused]] std::memory_order order = std::memory_order_seq_cst) const {
        uintptr_t val = lock<Intent::Read>();
        shared_ptr_type res;
        res.p_ = reinterpret_cast<U*>(val & ~2ULL); // Strip lock bit, keep mark (lock() already cleared it; defensive)
        if (res.get_unmarked_ptr()) res.get_unmarked_ptr()->AddRef();
        unlock(val); // Restore exact mark state
        return res;
    }

    // Replace the held pointer with `desired` (mark and all), releasing the old one.
    // The publishing store doubles as the spinlock release, so it must carry release
    // ordering (the default seq_cst does); passing a relaxed `order` would be unsafe.
    void store(shared_ptr_type desired, std::memory_order order = std::memory_order_seq_cst) {
        U* new_ptr = desired.get_raw();
        U* new_unmarked = desired.get_unmarked_ptr();
        if (new_unmarked) new_unmarked->AddRef();

        uintptr_t val = lock<Intent::Write>();
        U* old_ptr = reinterpret_cast<U*>(val & ~3ULL); // Only the actual ptr
        
        // Ensure we always have release semantics to unlock the spinlock properly.
        // We override the user's requested order because:
        // 1. If `order` lacks release semantics (e.g. relaxed), the spinlock would be released
        //    without proper synchronization, potentially exposing stale memory to other threads.
        // 2. We don't have to worry about acq_rel/acquire here as store() signatures don't allow them, 
        //    but enforcing release ensures correctness for the embedded lock.
        std::memory_order store_order = (order == std::memory_order_seq_cst) ? std::memory_order_seq_cst : std::memory_order_release;
        aptr_.store(reinterpret_cast<uintptr_t>(new_ptr), store_order);
        
        if (old_ptr && old_ptr->DelRef()) {
            delete old_ptr;
        }
    }

    // Compare-and-swap on the *full* value including the mark bit: the swap only
    // succeeds if both the pointer and its mark match `expected`. On failure,
    // `expected` is updated to the current value (with a fresh reference) to mirror
    // std::atomic<shared_ptr> semantics. The `failure` order is unused because the
    // critical section is bracketed by the spinlock's acquire/release.
    bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired,
                                 std::memory_order success = std::memory_order_seq_cst,
                                 [[maybe_unused]] std::memory_order failure = std::memory_order_seq_cst) {
        uintptr_t expected_val = reinterpret_cast<uintptr_t>(expected.get_raw());
        uintptr_t new_val = reinterpret_cast<uintptr_t>(desired.get_raw());
        
        uintptr_t val = lock<Intent::Write>();
        // Full-identity comparison: pointer AND mark bit must match. The ~2
        // is defensive only -- lock() returns the pre-lock value, whose lock
        // bit is already clear.
        if ((val & ~2ULL) == expected_val) {
            // Success
            U* new_unmarked = desired.get_unmarked_ptr();
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
            
            if (old_unmarked && old_unmarked->DelRef()) {
                delete old_unmarked;
            }
            return true;
        } else {
            // Failure
            shared_ptr_type res;
            res.p_ = reinterpret_cast<U*>(val & ~2ULL);
            if (res.get_unmarked_ptr()) res.get_unmarked_ptr()->AddRef();
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
                        // Writers sleep immediately to let the lock-holder finish and avoid CAS thrashing
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
                            // descheduled mid-critical-section. Measured
                            // 2026-07-23 (9950X, 32 logical CPUs, WriteHeavy/
                            // Graveyard/MassiveHeadInsert at 32 and 64
                            // threads): this branch fires only ~5 times per
                            // multi-second contended run -- the 8 short
                            // sleeps above absorb virtually all waits -- and
                            // durations from 10us to 10ms are throughput-
                            // neutral (all within the run-to-run noise). 1ms
                            // is chosen to bound the latency a stale sleeper
                            // adds while still giving a preempted holder time
                            // to run.
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
};

#endif // INCLUDED_INTR_SHARED_PTR_H_
