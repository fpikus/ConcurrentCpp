// Benchmark of the two-tier SpinLock (spinlock.h) across a range of lock
// contention, against the lock-free atomic and std::mutex reference points.
// See spinlock_bm_common.h for the benchmark body and the contention dial;
// tuning of the back-off tiers themselves lives in spinlock_tune_bm.C.
#include <atomic>

#include "spinlock.h"

#include "spinlock_bm_common.h"

// Lock-free baseline: same local work, but the guarded update is replaced by
// a single atomic fetch_add on the shared accumulator -- the lower bound on
// the cost of updating shared data at each contention level.
void BM_atomic(benchmark::State& state) {
  alignas(64) static std::atomic<double> shared_x;
  const long work = state.range(0);
  if (state.thread_index() == 0) shared_x.store(0, std::memory_order_relaxed);
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    for (long i = 0; i < work; ++i) local_x = std::sin(std::cos(local_x));
    benchmark::DoNotOptimize(shared_x.fetch_add(local_x, std::memory_order_relaxed));
  }
  state.SetItemsProcessed(state.iterations());
} // BM_atomic

BENCHMARK(BM_atomic) ARGS;
BENCHMARK_TEMPLATE(BM_lock, std::mutex)->Name("BM_mutex") ARGS;
BENCHMARK_TEMPLATE(BM_lock, SpinLock)->Name("BM_spinlock") ARGS;

BENCHMARK_MAIN();
