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
#ifndef ATOMIC_SHARED_PTR_CONCEPT_H
#define ATOMIC_SHARED_PTR_CONCEPT_H

#include <atomic>
#include <memory>
#include <concepts>

// Reference API for atomic shared pointers with Harris-style marking.
//
// These concepts document the contract that lock-free data structures (e.g.
// LockFreeList) are designed against; pointer implementations (the
// std::atomic<std::shared_ptr> adapter below, intr_shared_ptr, parlay's
// atomic_shared_ptr) are then plugged in through a template parameter. They
// are deliberately *not* enforced as constraints on the containers: they
// serve as the design reference, and as an opt-in conformance check for a
// new implementation (static_assert(AtomicSharedPtr<MyPtr<Node>>)).

// A shared pointer object that may hold mark bits in the least significant bit.
template <typename SP>
concept MarkedSharedPtr = requires(SP p, const SP cp) {
    { cp.is_marked() } -> std::same_as<bool>;
    { cp.get_unmarked() } -> std::same_as<SP>;
    { cp.set_mark() } -> std::same_as<SP>;
};

// An atomic shared pointer that supports operations for lock-free forward lists.
template <typename ASP>
concept AtomicSharedPtr = requires(ASP a, typename ASP::shared_ptr_type p) {
    { ASP::supports_marking } -> std::convertible_to<bool>;
    { a.load(std::memory_order_seq_cst) } -> std::same_as<typename ASP::shared_ptr_type>;
    { a.store(p, std::memory_order_seq_cst) };
    { a.compare_exchange_strong(p, p, std::memory_order_seq_cst, std::memory_order_seq_cst) } -> std::same_as<bool>;
} && MarkedSharedPtr<typename ASP::shared_ptr_type>;

// Wrapper for std::atomic<std::shared_ptr<T>> that conforms to the interface
// but supports_marking = false.
// Note: without marking, a lock-free list built on this type cannot exclude two
// anomalies (see LockFreeList and LockFreeList/lock_free_list_bugs.md, bug 2):
// - an insert after a concurrently erased node lands in a detached "black hole"
//   chain and silently vanishes once the last reference to that chain drops
//   (not a memory leak -- reference counting still reclaims the chain);
// - concurrent erases at adjacent positions can resurrect an already-erased node.
template <typename T>
class StdAtomicSharedPtrAdapter : public std::atomic<std::shared_ptr<T>> {
public:
    class shared_ptr_type : public std::shared_ptr<T> {
    public:
        using std::shared_ptr<T>::shared_ptr;
        shared_ptr_type(std::shared_ptr<T> ptr) : std::shared_ptr<T>(std::move(ptr)) {}
        
        bool is_marked() const { return false; }
        shared_ptr_type get_unmarked() const & { return *this; }
        shared_ptr_type get_unmarked() && { return std::move(*this); }
        shared_ptr_type set_mark() const { return *this; }
        
        std::shared_ptr<T>& get_std_ptr() { return *this; }
        const std::shared_ptr<T>& get_std_ptr() const { return *this; }
    };

    static constexpr bool supports_marking = false;

    using std::atomic<std::shared_ptr<T>>::atomic;
    StdAtomicSharedPtrAdapter(shared_ptr_type p) : std::atomic<std::shared_ptr<T>>(p.get_std_ptr()) {}

    shared_ptr_type load(std::memory_order order = std::memory_order_seq_cst) const {
        return shared_ptr_type(std::atomic<std::shared_ptr<T>>::load(order));
    }

    void store(const shared_ptr_type& desired, std::memory_order order = std::memory_order_seq_cst) {
        std::atomic<std::shared_ptr<T>>::store(desired.get_std_ptr(), order);
    }

    bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired, 
                                 std::memory_order success = std::memory_order_seq_cst, 
                                 std::memory_order failure = std::memory_order_seq_cst) {
        std::shared_ptr<T> exp_ptr = expected.get_std_ptr();
        bool result = std::atomic<std::shared_ptr<T>>::compare_exchange_strong(exp_ptr, desired.get_std_ptr(), success, failure);
        if (!result) {
            expected = shared_ptr_type(exp_ptr);
        }
        return result;
    }
};

#endif // ATOMIC_SHARED_PTR_CONCEPT_H
