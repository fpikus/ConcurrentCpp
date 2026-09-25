// Benchmark of three ways to maintain a shared running maximum under thread
// contention -- the classic double-checked-locking-pattern (DCLP) motivating
// example. Every iteration offers one candidate value to a shared maximum:
//
//   BM_cas       -- lock-free: atomic_max() (atomic_max.h), a compare-exchange
//                   loop that touches the shared word on every offer. Whatever
//                   the shipped header does today.
//   BM_spinlock  -- the "stupid" version: take a lock (the shipped SpinLock
//                   from ../Spinlock/spinlock.h) and do a plain compare-store,
//                   every single offer, whether or not it changes the maximum.
//   BM_dclp      -- double-checked locking: an unlocked atomic READ first, and
//                   only when the candidate looks like a new maximum is the
//                   lock taken and the check repeated under it. The whole point
//                   is that most offers never touch the lock.
//
// The branch-layout experiment: for each of CAS and DCLP, three variants that
// differ only in how the update condition is hinted -- cold (BM_cas_hint,
// BM_dclp), not at all (BM_cas_nohint, BM_dclp_nohint), hot (BM_cas_uphint,
// BM_dclp_uphint); see LAYOUT_* below. The CAS variants use their own copies of
// the loop, NOT atomic_max.h, so the experiment stays fixed when the shipped
// implementation changes.
//
// Why "no update is likely" is the right prediction everywhere: single-threaded
// the question does not arise (nobody uses an atomic there). With many threads
// the maximum either changes rarely or is contended. If it changes rarely, the
// no-update path is nearly every offer, and two jumps saved on it are most of
// its cost. If it is contended, everybody loses -- a failed exchange or a
// contended lock costs far more than two jumps -- and the hint is still the
// best available layout. Measured (fleet, 2026-09-06, hinted = the then-shipped
// atomic_max()): with no updates the hint doubles CAS on Grace and M3, gives
// 1.3-1.7x on Zen 5 and nothing on Intel (whose compiler output already has
// that layout), which makes CAS equal to DCLP; with updates it has no
// consistent effect, and DCLP beats CAS by 1.5-80x at every thread count above
// one except leslie's 256 (full SMT), a cell where CAS swings 4x between runs.
//
// Two workloads bracket the interesting range of how often the maximum actually
// changes -- which is what decides whether DCLP's fast path pays off:
//   never -- each thread offers one fixed random 64-bit value on every
//            iteration. After a brief warm-up the running maximum is the largest
//            of those values and never advances, so DCLP almost never locks and
//            the lock/CAS is almost never contended for a write. DCLP's best case.
//   grow  -- each thread offers a strictly increasing sequence (thread t offers
//            t+n, t+2n, ... for n threads), so the maximum advances all the time
//            and every offer is a new maximum FOR ITS THREAD. Most are
//            nevertheless stale by the time they reach the shared word -- other
//            threads have already passed them -- so DCLP's unlocked read still
//            skips the lock for most offers (93.7% with 16 threads on a
//            Ryzen 7940HS laptop, per atomic_max_count). The test of whether the
//            double check pays when the maximum really moves.
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

// Branch-layout parameter of the experiment: how the "the maximum must be
// updated" condition is presented to the compiler.
//   LAYOUT_COLD -- __builtin_expect(c, 0): update unlikely. The compiler sinks the
//                  update out of line; the no-update fast path is ONE taken branch.
//   LAYOUT_NONE -- the bare condition: the compiler chooses. For the CAS while
//                  loop that is two taken branches on the no-update path (a forward
//                  "skip the CAS" plus the loop back-edge) on cores that retire one
//                  taken branch per cycle; for DCLP's `if` it depends on the
//                  compiler and the branch order.
//   LAYOUT_HOT  -- __builtin_expect(c, 1): update likely. Forces the layout that
//                  favors the update path, whatever the compiler would choose --
//                  the control that shows what the no-update path costs when the
//                  layout is wrong for it.
// Macros, not an inline helper: the hint must sit directly on the branch
// condition to reach the compiler's block layout.
#define LAYOUT_COLD(c) __builtin_expect((c), 0)
#define LAYOUT_NONE(c) (c)
#define LAYOUT_HOT(c)  __builtin_expect((c), 1)

// The CAS loops of the experiment, independent of atomic_max.h (which may
// change): the same while loop as the shipped function, one per layout. The
// experiment that chose the loop shape (loop shape x branch bias, and that
// [[likely]] does NOT reach the loop layout while __builtin_expect on the
// condition does) lived here earlier; only its conclusion remains.
#define CAS_LOOP(NAME, LAYOUT)                                                  \
  template <typename T>                                                        \
  static bool NAME(std::atomic<T>& target, T val,                              \
                   std::memory_order success = std::memory_order_acq_rel,      \
                   std::memory_order failure = std::memory_order_acquire) {    \
    T cur = target.load(failure);                                              \
    while (LAYOUT(val > cur)) {                                                \
      if (target.compare_exchange_weak(cur, val, success, failure)) return true; \
    }                                                                          \
    return false;                                                              \
  } /* NAME() */

CAS_LOOP(atomic_max_hint,   LAYOUT_COLD)
CAS_LOOP(atomic_max_nohint, LAYOUT_NONE)
CAS_LOOP(atomic_max_uphint, LAYOUT_HOT)

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

CAS_BM(BM_cas,        atomic_max)          // the shipped atomic_max.h, whatever it is
CAS_BM(BM_cas_hint,   atomic_max_hint)     // experiment: loop WITH the hint
CAS_BM(BM_cas_nohint, atomic_max_nohint)   // experiment: same loop WITHOUT it
CAS_BM(BM_cas_uphint, atomic_max_uphint)   // experiment: update hinted LIKELY

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
//
// BM_dclp (update cold), BM_dclp_nohint (plain `if`) and BM_dclp_uphint (update
// hot) differ only in the LAYOUT wrapped around the unlocked probe. Whether the
// plain `if` needs the hint depends on the compiler: Clang lays it out 1-1
// without a hint, while GCC always uses the 2-0 layout, which is fast or slow
// depending on the order of the branches -- lucky with this code, unlucky if
// the if/else were reversed. The hint makes the layout independent of both,
// and BM_dclp_uphint forces the wrong layout, which is how to see what the
// plain `if` would cost if the compiler did not get lucky.
// Measured with no updates (2026-09-24, 3 runs x 10 reps): on linda (Zen 5,
// GCC 16.2) the unhinted DCLP equals the cold-hinted one (1.00x) at every
// thread count -- GCC chose the fast layout for this `if` -- and forcing the
// other layout (BM_dclp_uphint) costs 12-15%. GCC's unhinted CAS loop, by
// contrast, matches the HOT layout (~0.63x of cold), which is why the shipped
// hint is worth 1.5-1.6x there. On naptime (Zen 4, clang++-22, one short run)
// the unhinted DCLP also equals the cold hint, and the forced hot layout halves
// its throughput. Whether the 12-15% vs 2x difference is the compiler or the
// core has not been separated.
// One never/grow pair per layout; LAYOUT wraps only the unlocked probe.
#define DCLP_BM(NAME, LAYOUT)                                                   \
  void NAME##_never(benchmark::State& state) {                                 \
    if (state.thread_index() == 0)                                             \
      nmax_atomic.store(0, std::memory_order_relaxed);                         \
    std::mt19937_64 rng(state.thread_index());                                 \
    volatile unsigned long n = rng();                                          \
    for (auto _ : state) {                                                     \
      if (LAYOUT(n > nmax_atomic.load(std::memory_order_acquire))) { /* read-only probe */ \
        std::lock_guard guard(lock);                                           \
        if (n > nmax_atomic.load(std::memory_order_relaxed))  /* re-check under lock */ \
          nmax_atomic.store(n, std::memory_order_release);    /* publish */    \
      }                                                                        \
    }                                                                          \
    benchmark::DoNotOptimize(nmax_atomic.load());                              \
    state.SetItemsProcessed(state.iterations());                              \
  } /* NAME##_never */                                                         \
  void NAME##_grow(benchmark::State& state) {                                  \
    if (state.thread_index() == 0)                                             \
      nmax_atomic.store(0, std::memory_order_relaxed);                         \
    unsigned long n = state.thread_index(), dn = state.threads();              \
    for (auto _ : state) {                                                     \
      n += dn;                                                                 \
      if (LAYOUT(n > nmax_atomic.load(std::memory_order_acquire))) {           \
        std::lock_guard guard(lock);                                           \
        if (n > nmax_atomic.load(std::memory_order_relaxed))                   \
          nmax_atomic.store(n, std::memory_order_release);                     \
      }                                                                        \
    }                                                                          \
    benchmark::DoNotOptimize(nmax_atomic.load());                              \
    state.SetItemsProcessed(state.iterations());                              \
  } /* NAME##_grow */

DCLP_BM(BM_dclp,        LAYOUT_COLD)   // update hinted unlikely
DCLP_BM(BM_dclp_nohint, LAYOUT_NONE)   // plain `if`: the compiler chooses
DCLP_BM(BM_dclp_uphint, LAYOUT_HOT)    // update hinted likely

static const long numcpu = sysconf(_SC_NPROCESSORS_CONF);

#define ARGS \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// Grouped by workload so the variants sit adjacent for comparison: the
// shipped cas, the three cas layouts, the three dclp layouts, the dumb lock.
#define REGISTER(SUFFIX) \
  BENCHMARK(BM_cas##SUFFIX) ARGS;         \
  BENCHMARK(BM_cas_hint##SUFFIX) ARGS;    \
  BENCHMARK(BM_cas_nohint##SUFFIX) ARGS;  \
  BENCHMARK(BM_cas_uphint##SUFFIX) ARGS;  \
  BENCHMARK(BM_dclp##SUFFIX) ARGS;        \
  BENCHMARK(BM_dclp_nohint##SUFFIX) ARGS; \
  BENCHMARK(BM_dclp_uphint##SUFFIX) ARGS; \
  BENCHMARK(BM_spinlock##SUFFIX) ARGS;
REGISTER(_never)
REGISTER(_grow)

BENCHMARK_MAIN();
