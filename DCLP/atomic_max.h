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
#ifndef INCLUDED_ATOMIC_MAX_H
#define INCLUDED_ATOMIC_MAX_H
#include <atomic>

// ATOMIC_MAX_UNLIKELY(c) (header-private, #undef'd at the end): the condition
// `c`, marked as usually false, so the compiler lays out the code that runs
// when it is true as the cold path. With
// GCC and Clang (including clang-cl) this is __builtin_expect; elsewhere the
// hint is dropped and the code is still correct, only without the layout
// benefit. The standard [[unlikely]] attribute is not a substitute: placed on
// the loop body it does not reach the loop's block layout (see atomic_max()).
#if defined(__GNUC__) || defined(__clang__)
#define ATOMIC_MAX_UNLIKELY(c) __builtin_expect((c), 0)
#else
#define ATOMIC_MAX_UNLIKELY(c) (c)
#endif

// atomic_max(target, val, success, failure): atomically raise *target to
// max(*target, val). Returns true if the maximum was increased (val is the new
// maximum), false if val <= the current maximum (no update performed).
//
// This is the compare-and-swap idiom for a monotone maximum: load the current
// value, and while our candidate exceeds it, try to install ours -- retrying
// only if another thread moved the maximum in between (a failed exchange
// refreshes `cur` with what it found, so the loop re-tests val > cur and stops
// the moment someone else has already raised it past us).
//
// Memory ordering is the caller's, exactly as it is for compare_exchange -- two
// orders, one per outcome:
//   * success -- applied only when our exchange wins (the maximum is updated).
//     A read-modify-write order: pass release/acq_rel here to publish writes
//     the caller made happen-before this call to the next thread that reads the
//     maximum, or relaxed if the caller publishes nothing but the number.
//   * failure -- applied to every read that does NOT write: the initial load
//     AND every failed exchange (the two are the same "read-only, did not
//     update" outcome). A load order -- relaxed, consume, acquire, or seq_cst;
//     it must NOT be release or acq_rel, same constraint compare_exchange puts
//     on its failure order. Pass acquire to synchronize-with whoever last set
//     the maximum even on the no-update path; pass relaxed when only the value
//     is wanted and no happens-before is needed -- the common case for a plain
//     maximum reduction, which is why the initial load is often relaxed.
// The defaults keep the original "same as CAS" contract -- a read-write barrier
// on update, a read-only acquire when not updated -- so an unqualified call is
// fully synchronized; performance-sensitive callers pass relaxed for both.
template <typename T>
bool atomic_max(std::atomic<T>& target, T val,
                std::memory_order success = std::memory_order_acq_rel,
                std::memory_order failure = std::memory_order_acquire) {
  // The no-update read order governs the initial load too: returning without a
  // winning exchange is the same read-only outcome as a failed exchange.
  T cur = target.load(failure);
  // ATOMIC_MAX_UNLIKELY (__builtin_expect(..., 0)) biases the update as the COLD
  // path. This is a pure codegen hint -- no effect on semantics or ordering --
  // that makes the compiler sink the compare_exchange out of line, so the common
  // no-update fast path is a single taken branch instead of a forward "skip the
  // CAS" branch plus the loop back-edge. On a core that retires one taken branch
  // per cycle (e.g. Neoverse-V2) that roughly DOUBLES the read-only fast path
  // (measured: Grace 1.64 -> 3.28 G/s at t1). On x86 it depends on the core the
  // compiler tunes for: Intel (Granite Rapids) builds already have this layout,
  // so the hint is a no-op there, while Zen 5 gains 1.3-1.7x. When the maximum
  // advances on nearly every call (a monotone-increasing feed, which a warmed-up
  // maximum is not) the cost is within noise on the fleet. NOTE: the standard
  // [[unlikely]] attribute does NOT achieve this -- only the builtin on the loop
  // condition reaches the loop layout.
  while (ATOMIC_MAX_UNLIKELY(val > cur)) {
    // compare_exchange_weak (not strong) because we are already in a retry
    // loop: a spurious failure just re-tests the condition and costs one more
    // pass, whereas strong would hide an inner loop on LL/SC targets (ARM). The
    // failed exchange also refreshes `cur`, so the loop re-tests against the
    // newly observed maximum.
    if (target.compare_exchange_weak(cur, val, success, failure)) {
      return true;
    }
  } // while our candidate still exceeds the observed maximum
  return false;
} // atomic_max()

// Header-private: expanded when atomic_max() above was parsed (macros expand at
// definition, not at template instantiation), so it is not exported to
// includers.
#undef ATOMIC_MAX_UNLIKELY

#endif // INCLUDED_ATOMIC_MAX_H
