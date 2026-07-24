#pragma once

#include <atomic>

#include "parlay/details/atomic_details.hpp"
#include "parlay/details/hazard_pointers.hpp"
#include "parlay/shared_ptr.hpp"

namespace parlay {

template<typename T>
class atomic_shared_ptr {
  
  using shared_ptr_type_internal = shared_ptr<T>;
  using control_block_type = details::control_block_base;

public:
  // Wrapper class for shared_ptr that enforces strict bit-stealing.
  // Advanced ASAN/TSAN fix: The mark bit is embedded in the lowest bit of the underlying 
  // parlay::shared_ptr control_block (size strictly remains 16 bytes). 
  // To prevent misaligned pointer SEGVs, the bit is automatically intercepted and stripped 
  // inside parlay::smart_ptr_base's internal reference counting and destruction mechanisms.
  class shared_ptr_type {
  public:
    shared_ptr_type() = default;
    explicit(false) shared_ptr_type(std::nullptr_t) : ptr_(nullptr) {}
    shared_ptr_type(shared_ptr_type_internal ptr) : ptr_(std::move(ptr)) {}

    T& operator*() const { return *get_unmarked_internal().get(); }
    T* operator->() const { return get_unmarked_internal().get(); }
    T* get() const { return get_unmarked_internal().get(); }
    explicit operator bool() const { return static_cast<bool>(get_unmarked_internal()); }

    bool operator==(const shared_ptr_type& other) const { return ptr_.control_block == other.ptr_.control_block; }
    bool operator!=(const shared_ptr_type& other) const { return ptr_.control_block != other.ptr_.control_block; }

    long use_count() const { return ptr_.use_count(); }

    bool is_marked() const {
        return (reinterpret_cast<uintptr_t>(ptr_.control_block) & 1ULL) != 0;
    }

    shared_ptr_type get_unmarked() const & {
        shared_ptr_type res(*this);
        res.ptr_.control_block = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(res.ptr_.control_block) & ~1ULL);
        return res;
    }

    shared_ptr_type get_unmarked() && {
        ptr_.control_block = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(ptr_.control_block) & ~1ULL);
        return std::move(*this);
    }

    shared_ptr_type set_mark() const {
        shared_ptr_type res(*this);
        res.ptr_.control_block = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(res.ptr_.control_block) | 1ULL);
        return res;
    }

    shared_ptr_type_internal& get_internal() { return ptr_; }
    const shared_ptr_type_internal& get_internal() const { return ptr_; }

  private:
    friend class atomic_shared_ptr;
    shared_ptr_type_internal ptr_;

    shared_ptr_type_internal get_unmarked_internal() const {
        shared_ptr_type_internal res = ptr_;
        res.control_block = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(res.control_block) & ~1ULL);
        return res;
    }
  };

  static constexpr bool supports_marking = true;

  constexpr atomic_shared_ptr() noexcept = default;
  constexpr explicit(false) atomic_shared_ptr(std::nullptr_t) noexcept
    : control_block{nullptr} { }
  
  explicit(false) atomic_shared_ptr(shared_ptr_type desired) {
    auto [ptr_, control_block_] = desired.get_internal().release_internals();
    control_block.store(control_block_, std::memory_order_relaxed);
  }
  
  atomic_shared_ptr(const atomic_shared_ptr&) = delete;
  atomic_shared_ptr& operator=(const atomic_shared_ptr&) = delete;
  
  ~atomic_shared_ptr() { store(nullptr); }
  
  [[nodiscard]] shared_ptr_type load([[maybe_unused]] std::memory_order order = std::memory_order_seq_cst) const {
    control_block_type* current_control_block = nullptr;
    
    auto& hazptr = get_hazard_list<control_block_type>();
    
    while (true) {
      current_control_block = hazptr.protect(control_block, [](auto p) {
        return reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(p) & ~1ULL);
      });
      if (current_control_block == nullptr) break;
      
      control_block_type* unmarked_cb = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(current_control_block) & ~1ULL);
      if (unmarked_cb == nullptr) break;
      if (unmarked_cb->increment_strong_count_if_nonzero()) break;
    }

    return shared_ptr_type(make_shared_from_ctrl_block(current_control_block));
  }
  
  void store(shared_ptr_type desired, std::memory_order order = std::memory_order_seq_cst) {
    auto [ptr_, control_block_] = desired.get_internal().release_internals();
    auto old_control_block = control_block.exchange(control_block_, order);
    auto old_unmarked = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(old_control_block) & ~1ULL);
    if (old_unmarked) {
      old_unmarked->decrement_strong_count();
    }
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, shared_ptr_type&& desired,
      std::memory_order success, std::memory_order failure) {

      auto expected_ctrl_block = expected.get_internal().control_block;
      auto desired_ctrl_block = desired.get_internal().control_block;

      if (control_block.compare_exchange_weak(expected_ctrl_block, desired_ctrl_block, success, failure)) {
        auto expected_unmarked = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(expected_ctrl_block) & ~1ULL);
        if (expected_unmarked) {
          expected_unmarked->decrement_strong_count();
        }
        desired.get_internal().release_internals();
        return true;
      }
      else {
        expected = load();   // It's possible that expected ABAs and stays the same on failure, hence
        return false;        // why this algorithm can not be used to implement compare_exchange_strong
      }
  }
  
  bool compare_exchange_strong(shared_ptr_type& expected, shared_ptr_type&& desired,
      std::memory_order success, std::memory_order failure) {

    auto expected_ctrl_block = expected.get_internal().control_block;

    do {
      if (compare_exchange_weak(expected, std::move(desired), success, failure)) {
        return true;
      }
    } while (expected_ctrl_block == expected.get_internal().control_block);
    
    return false;
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, const shared_ptr_type& desired,
      std::memory_order success, std::memory_order failure) {
    return compare_exchange_weak(expected, shared_ptr_type{desired}, success, failure);
  }
  
  bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired,
      std::memory_order success, std::memory_order failure) {
    return compare_exchange_strong(expected, shared_ptr_type{desired}, success, failure);
  }

  bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_strong(expected, desired, order, details::default_failure_memory_order(order));
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, const shared_ptr_type& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_weak(expected, desired, order, details::default_failure_memory_order(order));
  }
  
  bool compare_exchange_strong(shared_ptr_type& expected, shared_ptr_type&& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_strong(expected, std::move(desired), order, details::default_failure_memory_order(order));
  }
  
  bool compare_exchange_weak(shared_ptr_type& expected, shared_ptr_type&& desired, std::memory_order order = std::memory_order_seq_cst) {
    return compare_exchange_weak(expected, std::move(desired), order, details::default_failure_memory_order(order));
  }
  
 private:
  static shared_ptr_type_internal make_shared_from_ctrl_block(control_block_type* control_block_) {
    control_block_type* unmarked = reinterpret_cast<control_block_type*>(reinterpret_cast<uintptr_t>(control_block_) & ~1ULL);
    if (unmarked) {
      T* ptr = static_cast<T*>(unmarked->get_ptr());
      return shared_ptr_type_internal{ptr, control_block_}; // Keep mark in control_block_!
    }
    else {
      return shared_ptr_type_internal{nullptr, control_block_}; // Keep mark in control_block_ even for nullptr!
    }
  }
 
  mutable std::atomic<control_block_type*> control_block;
};

} // namespace parlay
