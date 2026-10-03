// Copyright (c) 2026 Maged Michael
// See LICENSES for licensing terms.

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "asymmetric_fence.hpp"

/// Forward Declarations

namespace std {
template <class T, class D = default_delete<T>> class hazard_pointer_obj_base;
class hazard_pointer;
}

namespace mm_hp_detail {
class hp_obj;
struct hp_rec;
struct hp_tc;
struct hp_tc_flush_guard;
struct hp_domain;
struct hp_impl_registrar;

[[gnu::returns_nonnull]] hp_rec* hp_domain_acquire_rec();
void hp_domain_release_rec_list(hp_rec* head, hp_rec* tail) noexcept;
void hp_domain_retire_obj(hp_obj* obj) noexcept;
void hp_domain_register_impl_flags(std::uint32_t flags) noexcept;
void hp_tc_flush() noexcept;
void hp_tc_register_flush();
hp_rec* hp_acquire_rec();
}

/// Details Required by std

namespace mm_hp_detail {

/// Concepts

template <class T, class D>
auto hazard_protectable_test(std::hazard_pointer_obj_base<T, D>* base)
    -> decltype(static_cast<T*>(base));

template <class T>
concept hazard_protectable = requires(T* ptr) { hazard_protectable_test<T>(ptr); };

template <class T>
void assert_hazard_protectable() noexcept {
  static_assert(hazard_protectable<T>, "T is not a hazard-protectable type");
}

/// Classes

class hp_obj {
  hp_obj* next_;
  void (*reclaim_)(hp_obj*);
  void* reserved_{nullptr};  // Future ABI (pointer to P3427 cohort)

  template <class T, class D> friend class std::hazard_pointer_obj_base;
  friend struct hp_domain;
}; // class hp_obj

struct alignas(128) hp_rec {
  std::atomic<const hp_obj*> hp_{nullptr};
  hp_rec* next_{nullptr};
  hp_rec* avail_next_{nullptr};
}; // struct hp_rec

static_assert(alignof(hp_rec) > 1);  // Keep low address bit free, e.g., to use as lock

/// Thread-Local Cache

struct hp_tc {
  static constexpr int k_max_capacity = 126;  // ABI max capacity
  static_assert(k_max_capacity < 256);        // capacity_ and count_ are uint8_t
  hp_rec* fast_{nullptr};
  std::uint8_t capacity_{99};
  std::uint8_t count_{0};
  bool closed_{false};
  std::array<hp_rec*, k_max_capacity> hp_recs_;

  hp_rec* pop() noexcept {
    hp_rec* rec = fast_;
    if (rec) {
      fast_ = nullptr;
      return rec;
    }
    if (count_ == 0) return nullptr;
    rec = hp_recs_[--count_];
    assert(rec);
    return rec;
  }

  bool push(hp_rec* rec) noexcept {
    assert(rec);
    if (closed_) return false;
    if (!fast_) {
      fast_ = rec;
      return true;
    }
    if (count_ == capacity_) return false;
    hp_recs_[count_++] = rec;
    return true;
  }
}; // struct hp_tc

struct hp_tc_flush_guard {
  ~hp_tc_flush_guard() noexcept { hp_tc_flush(); }
}; // struct hp_tc_flush_guard

inline constinit thread_local hp_tc t_tc{};

inline void hp_tc_flush() noexcept {
  assert(!t_tc.closed_);
  assert(t_tc.count_ <= t_tc.capacity_);
  hp_rec* head = t_tc.fast_;
  hp_rec* tail = head;
  for (std::uint8_t i = 0; i < t_tc.count_; ++i) {
    hp_rec* rec = t_tc.hp_recs_[i];
    assert(rec);
    if (!head) tail = rec;
    rec->avail_next_ = head;
    head = rec;
  }
  if (head) {
    hp_domain_release_rec_list(head, tail);
  }
  t_tc.fast_ = nullptr;
  t_tc.count_ = 0;
  t_tc.closed_ = true;
}

inline void hp_tc_register_flush() {
  static thread_local hp_tc_flush_guard t_flush_guard;
}

[[gnu::noinline]]
inline hp_rec* hp_acquire_rec() {
  if (!t_tc.closed_) hp_tc_register_flush();
  return hp_domain_acquire_rec();
}

} // namespace mm_hp_detail

/// Standard Classes and Functions

namespace std {

template <class T, class D>
class hazard_pointer_obj_base : public mm_hp_detail::hp_obj {
  [[no_unique_address]] D deleter_;

 public:
  void retire(D d = D()) noexcept {
    mm_hp_detail::assert_hazard_protectable<T>();
    deleter_ = std::move(d);
    reclaim_ = [](mm_hp_detail::hp_obj* p) {
      auto* self = static_cast<hazard_pointer_obj_base*>(p);
      self->deleter_(static_cast<T*>(self));
    };
    mm_hp_detail::hp_domain_retire_obj(this);
  }

 protected:
  hazard_pointer_obj_base() = default;
  hazard_pointer_obj_base(const hazard_pointer_obj_base&) = default;
  hazard_pointer_obj_base(hazard_pointer_obj_base&&) = default;
  hazard_pointer_obj_base& operator=(const hazard_pointer_obj_base&) = default;
  hazard_pointer_obj_base& operator=(hazard_pointer_obj_base&&) = default;
  ~hazard_pointer_obj_base() = default;
}; // class hazard_pointer_obj_base

class hazard_pointer {
  mm_hp_detail::hp_rec* hp_rec_{nullptr};

  friend hazard_pointer make_hazard_pointer();

  explicit hazard_pointer(mm_hp_detail::hp_rec* rec) noexcept : hp_rec_(rec) {}

 public:
  hazard_pointer() noexcept = default;  // Compatible with future constexpr ABI

  hazard_pointer(hazard_pointer&& hp) noexcept
      : hp_rec_(std::exchange(hp.hp_rec_, nullptr)) {}

  hazard_pointer& operator=(hazard_pointer&& hp) noexcept {
    hazard_pointer(std::move(hp)).swap(*this);
    return *this;
  }

  ~hazard_pointer() {
    if (hp_rec_) {
      reset_protection();
      if (!mm_hp_detail::t_tc.push(hp_rec_)) {
        mm_hp_detail::hp_domain_release_rec_list(hp_rec_, hp_rec_);
      }
    }
  }

  bool empty() const noexcept { return hp_rec_ == nullptr; }

  template <class T>
  T* protect(const atomic<T*>& src) noexcept {
    assert(!empty());
    T* ptr = src.load(memory_order::relaxed);
    while (!try_protect(ptr, src)) {}
    return ptr;
  }

  template <class T>
  bool try_protect(T*& ptr, const atomic<T*>& src) noexcept {
    assert(!empty());
    T* expected = ptr;
    reset_protection(expected);
    p1202::asymmetric_thread_fence_light();
    ptr = src.load(memory_order::acquire);
    if (std::memcmp(&ptr, &expected, sizeof(ptr)) != 0) {
      reset_protection();
      return false;
    }
    return true;
  }

  template <class T>
  void reset_protection(const T* ptr) noexcept {
    assert(!empty());
    mm_hp_detail::assert_hazard_protectable<T>();
    const auto* hp_base = static_cast<const mm_hp_detail::hp_obj*>(ptr);
    hp_rec_->hp_.store(hp_base, memory_order::release);
  }

  void reset_protection(nullptr_t = nullptr) noexcept {
    assert(!empty());
    hp_rec_->hp_.store(nullptr, memory_order::release);
  }

  void swap(hazard_pointer& hp) noexcept { std::swap(hp_rec_, hp.hp_rec_); }
}; // class hazard_pointer

/// Free Functions

inline void swap(hazard_pointer& a, hazard_pointer& b) noexcept { a.swap(b); }

inline hazard_pointer make_hazard_pointer() {
  mm_hp_detail::hp_rec* rec = mm_hp_detail::t_tc.pop();
  if (!rec) rec = mm_hp_detail::hp_acquire_rec();
  return hazard_pointer(rec);
}

} // namespace std

/// Implementation Flag Registration

namespace mm_hp_detail {

inline constexpr std::uint32_t k_light_fence_protection = 1u << 0;

struct hp_impl_registrar {
  hp_impl_registrar() noexcept {
    hp_domain_register_impl_flags(k_light_fence_protection);
  }
}; // struct hp_impl_registrar

namespace { const hp_impl_registrar g_impl_registrar; }

} // namespace mm_hp_detail
