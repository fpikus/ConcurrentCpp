// Benchmark of the two-tier SpinLock (spinlock.h) across a range of lock
// contention, against its three practical alternatives: std::mutex, the
// wait-free fetch_add, and the CAS retry loop. The last one matters most for
// design decisions: fetch_add applies only when the update commutes and the
// target is never reclaimed -- a counter, an index into effectively infinite
// memory -- while the general shared-structure update ("swing this pointer")
// is a CAS, so lock-versus-CAS is the comparison most designs actually face.
// See spinlock_bm_common.h for the benchmark body and the contention dial;
// tuning of the back-off tiers themselves lives in spinlock_tune_bm.C.
#include <atomic>

#include "spinlock.h"

#include "spinlock_bm_common.h"

// Wait-free baseline: same local work, but the guarded update is replaced by a
// single atomic fetch_add of the work chunk's result -- one LOCK-prefixed add
// on x86, no retry, no waiting: the lower bound on the cost of updating shared
// data at each contention level. The total is an unsigned long, not a double,
// and not merely for consistency with BM_lock: there is no native FP fetch_add,
// so atomic<double>::fetch_add would quietly compile to a CAS retry loop -- a
// different mechanism, not wait-free, and measured in its own right in
// overhead_bm.C.
void BM_atomic(benchmark::State& state) {
  alignas(64) static std::atomic<unsigned long> shared_n;
  const long work = state.range(0);
  if (state.thread_index() == 0) shared_n.store(0, std::memory_order_relaxed);
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    local_x = do_work(local_x, work);
    shared_n.fetch_add(static_cast<unsigned long>(1.0 + local_x),
                       std::memory_order_relaxed);
  }
  state.SetItemsProcessed(state.iterations());
} // BM_atomic

// The lock-free retry loop: read the total, try to install the sum, repeat if
// somebody got there first. compare_exchange_weak is the right form inside a
// loop, and the loop is deliberately without back-off: this is the shape a
// lock-free update is usually written in, and how it degrades under
// contention is the reason the spinlock's ladder exists.
void BM_cas(benchmark::State& state) {
  alignas(64) static std::atomic<unsigned long> shared_n;
  const long work = state.range(0);
  if (state.thread_index() == 0) shared_n.store(0, std::memory_order_relaxed);
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    local_x = do_work(local_x, work);
    const unsigned long n = static_cast<unsigned long>(1.0 + local_x);
    unsigned long expected = shared_n.load(std::memory_order_relaxed);
    while (!shared_n.compare_exchange_weak(expected, expected + n,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
      // compare_exchange_weak refreshed `expected` with the value it found.
    }
  }
  state.SetItemsProcessed(state.iterations());
} // BM_cas

BENCHMARK(BM_atomic) ARGS;
BENCHMARK(BM_cas) ARGS;
BENCHMARK_TEMPLATE(BM_lock, std::mutex)->Name("BM_mutex") ARGS;
BENCHMARK_TEMPLATE(BM_lock, SpinLock)->Name("BM_spinlock") ARGS;

BENCHMARK_MAIN();
