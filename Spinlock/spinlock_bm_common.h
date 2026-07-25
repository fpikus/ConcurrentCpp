// Shared infrastructure for the spinlock benchmarks (spinlock_bm.C and
// spinlock_tune_bm.C): the lock-agnostic benchmark body and the common
// argument set.
//
// Every benchmark iteration does `work` evaluations of x = sin(cos(x)) on a
// thread-local value -- work done OUTSIDE the lock -- and then briefly takes
// the lock to add the local result to a shared accumulator. The `work`
// argument is the contention dial, calibrated from measured single-thread
// times (uncontended lock + guarded add ~4 ns, one sin(cos) evaluation
// ~15 ns), as the fraction of a thread's time spent holding the lock:
//   work=0   : nothing but the guarded update -- ~100% of time under the lock;
//   work=3   : ~10% of time under the lock;
//   work=30  : ~1%;
//   work=300 : ~0.1% -- the lock is almost always free when requested.
// The estimate is order-of-magnitude only (and shrinks further once waiting
// inflates the denominator), which is all the comparisons need.
//
// Select a slice with, e.g.:
//   ./spinlock_bm --benchmark_filter='work:0/.*threads:32$'
#ifndef INCLUDED_SPINLOCK_BM_COMMON_H
#define INCLUDED_SPINLOCK_BM_COMMON_H
#include <unistd.h>
#include <cmath>
#include <mutex>

#include "benchmark/benchmark.h"

// The benchmark body, shared by every lock type: `work` local sin(cos)
// evaluations (the contention dial, outside the lock), then a guarded add of
// the local result to the shared accumulator. The lock and the accumulator
// live on separate cache lines so the coherence traffic on the lock word is
// not conflated with the traffic on the data it guards. Reports items/s ==
// guarded updates/s.
template <typename Lock>
void BM_lock(benchmark::State& state) {
  alignas(64) static Lock lock;
  alignas(64) static double shared_x;
  const long work = state.range(0);
  if (state.thread_index() == 0) shared_x = 0;
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    for (long i = 0; i < work; ++i) local_x = std::sin(std::cos(local_x));
    std::lock_guard guard(lock);
    benchmark::DoNotOptimize(shared_x += local_x);
  }
  state.SetItemsProcessed(state.iterations());
} // BM_lock

static const long numcpu = sysconf(_SC_NPROCESSORS_CONF);

// Every benchmark runs the full contention dial (work=0 hammers the lock,
// work=300 rarely touches it) and scales from 1 thread to all CPUs.
#define ARGS \
  ->ArgName("work")->Arg(0)->Arg(3)->Arg(30)->Arg(300) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

#endif // INCLUDED_SPINLOCK_BM_COMMON_H
