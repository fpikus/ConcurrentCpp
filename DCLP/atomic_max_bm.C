// Benchmark of three ways to maintain a shared running maximum under thread
// contention -- the classic double-checked-locking-pattern (DCLP) motivating
// example. Every iteration offers one candidate value to a shared maximum:
//
//   BM_cas       -- lock-free: atomic_max() (atomic_max.h), a compare-exchange
//                   loop that touches the shared word on every offer.
//   BM_spinlock  -- the "stupid" version: take a lock (the shipped SpinLock
//                   from ../Spinlock/spinlock.h) and do a plain compare-store,
//                   every single offer, whether or not it changes the maximum.
//   BM_dclp      -- double-checked locking: an unlocked atomic READ first, and
//                   only when the candidate looks like a new maximum is the
//                   lock taken and the check repeated under it. The whole point
//                   is that most offers never touch the lock.
//
// Two workloads bracket the interesting range of how often the maximum actually
// changes -- which is what decides whether DCLP's fast path pays off:
//   never -- each thread offers random 64-bit values. After a brief warm-up the
//            running maximum is enormous and essentially never advances, so DCLP
//            almost never locks and the lock/CAS is almost never contended for a
//            write. This is DCLP's best case.
//   grow  -- each thread offers a strictly increasing sequence, so EVERY offer
//            is a new global maximum: the maximum advances on every iteration,
//            DCLP's inner check always passes, and it degenerates into the
//            locked version plus the cost of the outer read. This is DCLP's
//            worst case, and the honest test of whether the double check hurts
//            when it never helps.
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <random>

#include "benchmark/benchmark.h"

#include "spinlock.h"                     // the shipped SpinLock (../Spinlock)
#include "atomic_max.h"                   // the lock-free atomic_max()

// Shared state, each on its own cache line so the three mechanisms do not
// perturb one another's coherence traffic across benchmark families. The
// lock-free and DCLP variants share the atomic word; the locked variant uses a
// plain unsigned long guarded by the lock.
alignas(64) static std::atomic<unsigned long> nmax_atomic;
alignas(64) static unsigned long nmax_plain;
alignas(64) static SpinLock lock;

// The pre-hint version of the header's atomic_max(): the same straightforward
// while loop WITHOUT the __builtin_expect update-is-cold hint. Kept only as the
// benchmark's "before": on cores that retire one taken branch per cycle its
// no-update fast path is two taken branches (a forward "skip the CAS" plus the
// loop back-edge), where the shipped atomic_max() biases the update cold and
// fuses them to one -- doubling the read-only fast path (measured on Grace).
// The experiment that established this (loop shape x branch bias, and that the
// [[likely]] attribute does NOT reach the loop layout while __builtin_expect on
// the condition does) lived here; only the winner and this baseline remain.
template <typename T>
static bool atomic_max_nohint(std::atomic<T>& target, T val,
                              std::memory_order success = std::memory_order_acq_rel,
                              std::memory_order failure = std::memory_order_acquire) {
  T cur = target.load(failure);
  while (val > cur) {
    if (target.compare_exchange_weak(cur, val, success, failure)) return true;
  }
  return false;
} // atomic_max_nohint()

// One never/grow benchmark pair per CAS flavor -- the loop body differs only in
// which max function it calls, so a macro keeps the two from being two copies.
// never: one fixed random offer per thread, so the max never advances after
// warm-up (update genuinely rare -- the hint's best case). grow: a strictly
// increasing offer, every iteration a new max (the hint's worst case).
#define CAS_BM(NAME, MAXFN)                                                     \
  void NAME##_never(benchmark::State& state) {                                 \
    if (state.thread_index() == 0)                                             \
      nmax_atomic.store(0, std::memory_order_relaxed);                         \
    std::mt19937_64 rng(state.thread_index());                                 \
    volatile unsigned long n = rng();                                          \
    for (auto _ : state) { MAXFN(nmax_atomic, n); }                            \
    benchmark::DoNotOptimize(nmax_atomic.load());                              \
    state.SetItemsProcessed(state.iterations());                              \
  } /* NAME##_never */                                                         \
  void NAME##_grow(benchmark::State& state) {                                  \
    if (state.thread_index() == 0)                                             \
      nmax_atomic.store(0, std::memory_order_relaxed);                         \
    unsigned long n = state.thread_index(), dn = state.threads();              \
    for (auto _ : state) { n += dn; MAXFN(nmax_atomic, n); }                   \
    benchmark::DoNotOptimize(nmax_atomic.load());                              \
    state.SetItemsProcessed(state.iterations());                              \
  } /* NAME##_grow */

CAS_BM(BM_cas,        atomic_max)          // header while, WITH the hint (shipped)
CAS_BM(BM_cas_nohint, atomic_max_nohint)   // same loop, WITHOUT the hint (baseline)

// --- locked: take the lock on every offer, plain compare-store -------------
void BM_spinlock_never(benchmark::State& state) {
  if (state.thread_index() == 0) nmax_plain = 0;
  std::mt19937_64 rng(state.thread_index());
  volatile unsigned long n = rng();
  for (auto _ : state) {
    std::lock_guard guard(lock);
    if (n > nmax_plain) nmax_plain = n;
  }
  benchmark::DoNotOptimize(nmax_plain);
  state.SetItemsProcessed(state.iterations());
} // BM_spinlock_never

void BM_spinlock_grow(benchmark::State& state) {
  if (state.thread_index() == 0) nmax_plain = 0;
  unsigned long n = state.thread_index(), dn = state.threads();
  for (auto _ : state) {
    n += dn;
    std::lock_guard guard(lock);
    if (n > nmax_plain) nmax_plain = n;
  }
  benchmark::DoNotOptimize(nmax_plain);
  state.SetItemsProcessed(state.iterations());
} // BM_spinlock_grow

// --- DCLP: unlocked acquire read, lock only to actually update -------------
// The outer load is acquire and the store is release, so a thread that sees a
// new maximum via the fast path also sees everything the setter published --
// the same read-only / read-write barrier split atomic_max() uses. The inner
// re-read can be relaxed: it runs while holding the lock, whose acquire already
// synchronized with the previous holder's releasing unlock().
void BM_dclp_never(benchmark::State& state) {
  if (state.thread_index() == 0) nmax_atomic.store(0, std::memory_order_relaxed);
  std::mt19937_64 rng(state.thread_index());
  volatile unsigned long n = rng();
  for (auto _ : state) {
    if (n > nmax_atomic.load(std::memory_order_acquire)) {   // fast path: read-only
      std::lock_guard guard(lock);
      if (n > nmax_atomic.load(std::memory_order_relaxed))   // re-check under lock
        nmax_atomic.store(n, std::memory_order_release);     // publish: read-write
    }
  }
  benchmark::DoNotOptimize(nmax_atomic.load());
  state.SetItemsProcessed(state.iterations());
} // BM_dclp_never

void BM_dclp_grow(benchmark::State& state) {
  if (state.thread_index() == 0) nmax_atomic.store(0, std::memory_order_relaxed);
  unsigned long n = state.thread_index(), dn = state.threads();
  for (auto _ : state) {
    n += dn;
    if (n > nmax_atomic.load(std::memory_order_acquire)) {
      std::lock_guard guard(lock);
      if (n > nmax_atomic.load(std::memory_order_relaxed))
        nmax_atomic.store(n, std::memory_order_release);
    }
  }
  benchmark::DoNotOptimize(nmax_atomic.load());
  state.SetItemsProcessed(state.iterations());
} // BM_dclp_grow

static const long numcpu = sysconf(_SC_NPROCESSORS_CONF);

#define ARGS \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// Grouped by workload so the four mechanisms sit adjacent for comparison:
// cas (shipped, hinted) vs cas_nohint (baseline) vs dclp vs the dumb lock.
#define REGISTER(SUFFIX) \
  BENCHMARK(BM_cas##SUFFIX) ARGS;        \
  BENCHMARK(BM_cas_nohint##SUFFIX) ARGS; \
  BENCHMARK(BM_dclp##SUFFIX) ARGS;       \
  BENCHMARK(BM_spinlock##SUFFIX) ARGS;
REGISTER(_never)
REGISTER(_grow)

BENCHMARK_MAIN();
