// Copyright (c) 2026 Maged Michael
// See LICENSES for licensing terms.

#include "mm_hp.hpp"

#include <new>
#include <thread>
#include <type_traits>
#include <unordered_set>

namespace mm_hp_detail {

/// Domain Details

using mo = std::memory_order;

// TSAN-PATCH: TSan does not model the asymmetric fence pair. Under TSan the
// scan-side loads of hazard slots are acquire, so that a reader's release of
// its hazard happens-before the scan's free of the object; the unpatched code
// relies on the hardware not performing the free, which depends on these
// relaxed loads, before they resolve. Non-TSan builds are unchanged.
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define MM_HP_TSAN 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define MM_HP_TSAN 1
#endif
#ifdef MM_HP_TSAN
constexpr mo k_hp_scan_order = mo::acquire;
#else
constexpr mo k_hp_scan_order = mo::relaxed;
#endif

namespace { constinit thread_local bool t_reclaiming{false}; }

struct hp_domain {
  static constexpr std::size_t k_reclaim_floor = 1000;

  alignas(128) std::atomic<std::uintptr_t> avail_{0};
  alignas(128) std::atomic<std::size_t> rcount_{0};
               std::atomic<hp_obj*> retired_{nullptr};
  alignas(128) std::atomic<hp_rec*> hp_recs_{nullptr};
               std::atomic<std::size_t> hcount_{0};
               std::atomic<std::uint32_t> impl_flags_{0};

  struct hp_obj_list {
    hp_obj* head{nullptr};
    hp_obj* tail{nullptr};
    std::size_t count{0};

    void add(hp_obj* obj) noexcept {
      if (!head) tail = obj;
      obj->next_ = head;
      head = obj;
      ++count;
    }
  }; // struct hp_obj_list

  /// HP Record Operations

  hp_rec* acquire_hp_rec() {
    while (true) {
      std::uintptr_t avail = avail_.load(mo::relaxed);
      if (avail & 1) {  // locked
        std::this_thread::yield();
        continue;
      }
      if (!avail) return new_hp_rec();
      if (avail_.compare_exchange_weak(
          avail, avail | 1, mo::acquire, mo::relaxed)) {
        hp_rec* rec = reinterpret_cast<hp_rec*>(avail);
        avail_.store(reinterpret_cast<std::uintptr_t>(rec->avail_next_), mo::release);
        rec->avail_next_ = nullptr;
        return rec;
      }
    }
  }

  void release_hp_rec_list(hp_rec* head, hp_rec* tail) noexcept {
    assert(head);
    assert(tail);
    std::uintptr_t newhead = reinterpret_cast<std::uintptr_t>(head);
    while (true) {
      std::uintptr_t avail = avail_.load(mo::relaxed);
      if (avail & 1) {  // locked
        std::this_thread::yield();
        continue;
      }
      tail->avail_next_ = reinterpret_cast<hp_rec*>(avail);
      if (avail_.compare_exchange_weak(avail, newhead, mo::release, mo::relaxed)) return;
    }
  }

  hp_rec* new_hp_rec() {
    hp_rec* rec = new hp_rec();
    hp_rec* head = hp_recs_.load(mo::relaxed);
    do {
      rec->next_ = head;
    } while (!hp_recs_.compare_exchange_weak(head, rec, mo::release, mo::relaxed));
    hcount_.fetch_add(1, mo::relaxed);
    return rec;
  }

  /// Retired Object Operations

  void push_retired_obj(hp_obj* obj) noexcept {
    assert(obj);
    assert(obj->reclaim_);
    p1202::asymmetric_thread_fence_light();
    if (push_retired_list({obj, obj, 1}) >= reclaim_threshold() && !t_reclaiming) {
      do_reclamation();
    }
  }

  std::size_t push_retired_list(const hp_obj_list& list) noexcept {
    assert(list.head);
    assert(list.tail);
    assert(list.count);
    hp_obj* cur = retired_.load(mo::relaxed);
    do {
      list.tail->next_ = cur;
    } while (!retired_.compare_exchange_weak(cur, list.head, mo::release, mo::relaxed));
    return rcount_.fetch_add(list.count, mo::relaxed) + list.count;
  }

  /// Reclamation Operations

  std::size_t reclaim_threshold() const noexcept {
    std::size_t hcount = hcount_.load(mo::relaxed);
    std::size_t scaled = 2 * hcount;
    return scaled < k_reclaim_floor ? k_reclaim_floor : scaled;
  }

  [[gnu::noinline]]
  void do_reclamation() noexcept {
    hp_obj* retired = retired_.exchange(nullptr, mo::acquire);
    if (!retired) return;
    t_reclaiming = true;
    rcount_.store(0, mo::relaxed);
    p1202::asymmetric_thread_fence_heavy();
    // Must load hp_recs_ after the fence. Earlier load could miss a protecting hp_rec.
    reclaim_unprotected(retired, hp_recs_.load(mo::acquire));
    t_reclaiming = false;
  }

  void reclaim_unprotected(hp_obj* retired, hp_rec* hp_recs) noexcept {
    hp_obj_list keep;
    while (retired) {
      std::unordered_set<const hp_obj*> protected_set;
      try {
        protected_set = collect_protected(hp_recs);
      } catch (const std::bad_alloc&) {
        retired = reclaim_one_in_place(retired, hp_recs, keep);
        continue;
      }
      reclaim_with_set(retired, protected_set, keep);
      break;
    }
    if (keep.head) push_retired_list(keep);
  }

  void reclaim_with_set(hp_obj* retired,
                        const std::unordered_set<const hp_obj*>& protected_set,
                        hp_obj_list& keep) noexcept {
    while (retired) {
      hp_obj* next = retired->next_;
      if (protected_set.contains(retired)) keep.add(retired);
      else retired->reclaim_(retired);
      retired = next;
    }
  }

  hp_obj* reclaim_one_in_place(hp_obj* retired, hp_rec* hp_recs,
                               hp_obj_list& keep) noexcept {
    while (retired) {
      hp_obj* next = retired->next_;
      if (!is_protected(retired, hp_recs)) {
        retired->reclaim_(retired);
        return next;
      }
      keep.add(retired);
      retired = next;
    }
    return nullptr;
  }

  std::unordered_set<const hp_obj*> collect_protected(hp_rec* hp_recs) {
    std::unordered_set<const hp_obj*> protected_set;
    protected_set.reserve(hcount_.load(mo::relaxed));
    for (hp_rec* rec = hp_recs; rec; rec = rec->next_) {
      const hp_obj* ptr = rec->hp_.load(k_hp_scan_order);
      if (ptr) protected_set.insert(ptr);
    }
    return protected_set;
  }

  bool is_protected(hp_obj* obj, hp_rec* hp_recs) noexcept {
    for (hp_rec* rec = hp_recs; rec; rec = rec->next_) {
      if (rec->hp_.load(k_hp_scan_order) == obj) return true;
    }
    return false;
  }

}; // struct hp_domain

/// Domain Objects and Free Functions

static_assert(std::is_trivially_destructible_v<hp_domain>);

namespace { constinit hp_domain g_domain{}; }

hp_rec* hp_domain_acquire_rec() {
  return g_domain.acquire_hp_rec();
}

void hp_domain_release_rec_list(hp_rec* head, hp_rec* tail) noexcept {
  g_domain.release_hp_rec_list(head, tail);
}

void hp_domain_retire_obj(hp_obj* obj) noexcept {
  g_domain.push_retired_obj(obj);
}

void hp_domain_register_impl_flags(std::uint32_t flags) noexcept {
  g_domain.impl_flags_.fetch_or(flags, mo::relaxed);
}

/// Domain Exit Guard

struct hp_domain_exit_guard {
  ~hp_domain_exit_guard() noexcept { g_domain.do_reclamation(); }
}; // struct hp_domain_exit_guard

namespace { hp_domain_exit_guard g_domain_exit_guard; }

} // namespace mm_hp_detail
