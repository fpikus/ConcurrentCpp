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
#ifndef INCLUDED_SPINLOCK_TUNE_H
#define INCLUDED_SPINLOCK_TUNE_H
#include <sched.h>
#include <time.h>
#include <atomic>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// The tunable spinlock: the TTAS lock of spinlock.h with its entire back-off
// ladder lifted into template parameters, so a benchmark can sweep the shape of
// the ladder instead of guessing at it. This header exists for tuning; the lock
// that ships is SpinLock in spinlock.h, and one point of this parameter space
// reproduces it exactly:
//
//   LadderedLock<BackOffParams{}> == SpinLock
//
// "Exactly" is checked, not asserted: the two lock() bodies compile to the same
// instruction stream. The benchmarks run SpinLock itself alongside the baseline
// so that any future drift between them shows up as a gap in the measurements.
//
// Correctness is identical to SpinLock's -- same TTAS acquire, same
// release/acquire handoff -- and is documented there. Only the waiting policy
// is parameterized here; a waiting policy cannot affect correctness, only
// throughput, which is exactly why it is worth measuring.
//
// Four pieces, in dependency order:
//   BackOffParams    every knob of the back-off ladder as one constexpr
//                    aggregate; the defaults spell the shipped ladder, so a
//                    named policy reads as its delta from SpinLock. This is
//                    what will eventually carry names like HighContention or
//                    LowContentionRead, once the campaign settles the values.
//   BackOffLadder<P> the waiting policy those parameters describe: how to
//                    acquire a lock word that is currently taken. This is the
//                    thing being tuned.
//   TunableSpinLock  the lock word alone. lock<P>() takes the policy as a
//                    template argument, so the CALLER chooses how to wait,
//                    per call site -- one flag, any number of waiting
//                    strategies (this is what replaced the old two-role
//                    RWTunableSpinLock; see spinlock_rw_tune_bm.C).
//   LockAdapter<P>   a policy-bound Lockable view of somebody else's word:
//                    RAII (std::lock_guard) for the multi-policy case, where
//                    LadderedLock cannot serve because each one owns its own
//                    flag.
//   LadderedLock<P>  word + one fixed policy behind the standard Lockable
//                    face, owning its word: the one-policy shape, used by the
//                    lock-agnostic benchmark bodies; swept by
//                    spinlock_tune_bm.C.

// Portable spin-wait hint: PAUSE on x86, YIELD on ARM, nothing elsewhere.
// PAUSE does not release the core -- it quiets the pipeline and shortens the
// memory-order-violation penalty on exiting the spin -- so the pause tier below
// is still "spinning", just politer spinning.
namespace {
  static inline void spin_pause() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#endif
  } // spin_pause()
} // anonymous namespace

// Sleep durations for ladder configurations, named so that benchmark names read
// as durations: the benchmarks pass the macro and stringify the argument, so a
// name carries `long:NS_1ms` rather than a raw nanosecond count -- which is
// also why these are macros and not constexpr variables. The odd 1 ns on top of
// each round number is inherited from spinlock.h, where it keeps the request
// from being a suspiciously exact timer tick.
#define NS_100us 100001
#define NS_1ms   1000001
#define NS_10ms  10000001
#define NS_off   0              // no long-sleep tier: the ladder just repeats

// Every knob of the back-off ladder, as one aggregate that is passed around
// as a constexpr VALUE (a C++20 structural template parameter). The defaults
// are the shipped SpinLock ladder, so a configuration written with designated
// initializers shows exactly its delta from the lock that ships:
//
//   BackOffParams{}                 == SpinLock's ladder
//   BackOffParams{.npause = 64}     == the same, with 64 PAUSE rounds first
//   BackOffParams{.nshort = 1}      == the same, parking after one short sleep
//
// Identical parameter values name the same instantiation wherever they are
// spelled, and a future knob is a new field with a default -- not a template
// signature change breaking every use site.
struct BackOffParams {
  int ntry = 8;             // acquire attempts per round, unrolled
  int npause = 0;           // rounds ending in a PAUSE-class hint
  int nyield = 0;           // rounds ending in sched_yield()
  int nshort = 8;           // rounds ending in a short_sleep_ns nanosleep
  long long_sleep_ns = NS_1ms;  // the final, real sleep; 0 == no long tier
  long short_sleep_ns = 1;  // ~1 ns rounds up to "deschedule until next tick"
}; // struct BackOffParams

// The four-tier back-off ladder described by P: the policy for acquiring a
// lock word that somebody else may be holding. It owns no state -- it is
// handed the lock word to work on -- so one lock can be acquired through any
// number of different ladders, which is the whole point of factoring the
// waiting policy out of the lock.
//
// Waiting is organized in rounds. Every round begins with a burst of P.ntry
// back-to-back acquire attempts (no back-off between them, fully unrolled),
// and ends with one back-off action, escalating cheapest-first: P.npause
// rounds of PAUSE, then P.nyield rounds of sched_yield(), then P.nshort
// rounds of a short nanosleep, then one long sleep -- after which the round
// counter resets and the whole ladder repeats. Any tier with a count of 0 is
// skipped entirely; long_sleep_ns == 0 skips the long sleep too, which turns
// the ladder into a pure cycle of the tiers above it (that is how the
// degenerate policies -- no back-off at all, pause-only, yield-only -- are
// expressed in this parameter space).
template <BackOffParams P>
struct BackOffLadder {
  static_assert(P.ntry >= 1, "ntry must be at least 1 or the lock is never taken");
  static_assert(P.npause >= 0 && P.nyield >= 0 && P.nshort >= 0,
                "back-off round counts cannot be negative");
  static_assert(P.long_sleep_ns >= 0 && P.long_sleep_ns < 1000000000,
                "long_sleep_ns must be a valid timespec nanosecond count");
  static_assert(P.short_sleep_ns >= 0 && P.short_sleep_ns < 1000000000,
                "short_sleep_ns must be a valid timespec nanosecond count");

  // Round numbers at which each tier gives way to the next. Cumulative, so a
  // tier with count 0 has last_end == this_end and its branch is dead. They are
  // unsigned (as is the round counter) so that a 0 bound makes `round < bound`
  // false by type, not by range analysis: the compiler folds a disabled tier
  // away without having to prove the counter never goes negative.
  static constexpr unsigned pause_end = P.npause;
  static constexpr unsigned yield_end = pause_end + P.nyield;
  static constexpr unsigned short_end = yield_end + P.nshort;

  // Acquire `lock_word`, blocking (spinning, then backing off) until it is held.
  // The round counter drives the ladder; the long sleep restarts it rather than
  // advancing it, so the cycle repeats from the cheapest tier. The counter is
  // unsigned and never leaves [0, short_end] so that every disabled tier
  // compiles out completely (see the tier bounds above) -- a config with
  // npause == 0 must not pay even a test for the pause tier, or the sweep would
  // be measuring the harness instead of the policy.
  static void acquire(std::atomic<int>& lock_word) {
    static constexpr timespec short_sleep = { 0, P.short_sleep_ns };
    static constexpr timespec long_sleep = { 0, P.long_sleep_ns };
    unsigned round = 0;
    while (!try_acquire_burst(lock_word)) {
      if (round < pause_end) {
        spin_pause();
      } else if (round < yield_end) {
        sched_yield();
      } else if (round < short_end) {
        nanosleep(&short_sleep, nullptr);
      } else {
        if constexpr (P.long_sleep_ns > 0) nanosleep(&long_sleep, nullptr);
        round = 0;                      // restart the ladder, do not count
        continue;                       // this round against the first tier
      }
      ++round;
    } // spin/back-off loop
  } // BackOffLadder::acquire()

  // One TTAS acquire attempt: "is the lock unavailable to me?", inverted.
  // The relaxed load is the test (waiters only read the shared line while the
  // lock is held); only a load that sees 0 pays for the exchange, whose acquire
  // is what makes a successful acquisition synchronize with the previous
  // holder's releasing unlock(). Returns true if we now hold the lock.
  static bool try_acquire_once(std::atomic<int>& lock_word) {
    return !(lock_word.load(std::memory_order_relaxed) ||
             lock_word.exchange(1, std::memory_order_acquire));
  }

  // A burst of P.ntry attempts back to back, with no back-off and no loop
  // counter between them -- so a lock released mid-burst is picked up within a
  // few instructions. The fold over the index pack expands the call P.ntry
  // times (hand-unrolled by the compiler front end rather than left to the
  // optimizer, since the burst length is the very thing being measured) and
  // short-circuits on the first success. Returns true if we now hold the lock.
  // The cast of I to void is only there to make the operand depend on the pack
  // so that it expands.
  static bool try_acquire_burst(std::atomic<int>& lock_word) {
    return [&lock_word]<std::size_t... I>(std::index_sequence<I...>) {
      return (... || (static_cast<void>(I), try_acquire_once(lock_word)));
    }(std::make_index_sequence<static_cast<std::size_t>(P.ntry)>{});
  } // BackOffLadder::try_acquire_burst()
}; // struct BackOffLadder

// The lock word alone: mutual exclusion with the waiting policy left to the
// caller, per call site. lock<P>() climbs the ladder described by P, and any
// number of call sites can climb different ladders to the same flag --
// correctness is a property of the word (one holder at a time, release/acquire
// handoff exactly as documented in spinlock.h), while the policy only decides
// how a loser waits. This is what lets a reader and a writer -- or a
// latency-critical path and a background thread -- share one lock and wait
// differently, without a dedicated two-role lock class.
class TunableSpinLock {
  public:
  TunableSpinLock() = default;
  TunableSpinLock(const TunableSpinLock&) = delete;
  TunableSpinLock& operator=(const TunableSpinLock&) = delete;

  // Acquire the lock, waiting as P prescribes, until it is held.
  template <BackOffParams P>
  void lock() { BackOffLadder<P>::acquire(lock_); }

  // Release the lock, however it was acquired; the store-release pairs with
  // the acquire on the next acquirer's winning exchange, through any ladder.
  void unlock() { lock_.store(0, std::memory_order_release); }

  // Advisory, non-synchronizing peek at the lock state; racy by construction.
  bool locked() const { return lock_.load(std::memory_order_relaxed) == 1; }

  private:
  std::atomic<int> lock_ {0};           // 0 == free, 1 == held
}; // class TunableSpinLock

// A policy-bound VIEW of a TunableSpinLock: the standard Lockable face over a
// word the adapter does not own. This is how multi-policy code gets RAII --
// LadderedLock below cannot serve there, because each LadderedLock owns its
// own word and two of them would be two different locks. The adapter binds a
// policy to somebody else's word for the duration of a scope:
//
//   TunableSpinLock lock;                 // one flag...
//   ...
//   LockAdapter<ReadPolicy> adapter(lock);
//   std::lock_guard guard(adapter);       // ...locked the reader's way, RAII
//
// so a bare lock<P>()/unlock() pair -- with its potential for early returns
// and forgotten unlocks -- never needs to be written by hand. It is a view,
// like a reference: copying it is copying the binding, not the lock.
template <BackOffParams P>
class LockAdapter {
  public:
  explicit LockAdapter(TunableSpinLock& lock) : lock_(lock) {}

  void lock() { lock_.lock<P>(); }
  void unlock() { lock_.unlock(); }
  bool locked() const { return lock_.locked(); }

  private:
  TunableSpinLock& lock_;               // the word being viewed; not owned
}; // class LockAdapter

// The word behind the standard Lockable face, with one fixed policy that OWNS
// its word: what the lock-agnostic benchmark bodies need, and the right shape
// for one-policy applications. A one-line delegate -- the policy still lives
// entirely in P. For several policies on one flag, use TunableSpinLock plus
// LockAdapter above instead.
//
//   LadderedLock<BackOffParams{}> == SpinLock
//
// "Exactly" is checked, not asserted: the two lock() bodies compile to the
// same instruction stream. The benchmarks run SpinLock itself alongside the
// baseline so that any future drift shows up as a gap in the measurements.
template <BackOffParams P>
class LadderedLock {
  public:
  LadderedLock() = default;
  LadderedLock(const LadderedLock&) = delete;
  LadderedLock& operator=(const LadderedLock&) = delete;

  void lock() { lock_.lock<P>(); }
  void unlock() { lock_.unlock(); }
  bool locked() const { return lock_.locked(); }

  private:
  TunableSpinLock lock_;
}; // class LadderedLock

#endif // INCLUDED_SPINLOCK_TUNE_H
