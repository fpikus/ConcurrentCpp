// The back-off ladder sweep of spinlock_tune_bm.C, re-run with memory-
// streaming work in place of register-resident sin/cos: same lock candidates
// (spinlock_tune_configs.h), same contention dial, different physics between
// the lock operations.
//
// WHY THE KIND OF WORK MIGHT CHANGE THE ANSWER
//
// With sin/cos work, a thread between lock operations occupies nothing but its
// own core: registers in, registers out. The only machine-wide resources the
// benchmark contends for are the lock word and the guarded line, so the cost of
// a waiter that spins instead of sleeping is one core's worth of coherence
// probes -- and that is the regime in which the shipped ladder was tuned.
//
// Memory-streaming work contends for everything: L3 capacity, memory
// controller queues, DRAM bandwidth. That changes the terms of the back-off
// trade in both directions at once:
//   * a PARKED waiter now returns more than a core -- it stops consuming
//     bandwidth and stops evicting the other threads' (and its own future)
//     cache lines, so sleeping should be worth more than it was;
//   * a SPINNING waiter consumes no bandwidth at all -- TTAS reads and PAUSE
//     live entirely in its own cache -- so while it wastes its core, it does
//     not slow anybody else's work down, and the work it is not doing was
//     going to be bandwidth-starved anyway;
//   * the work itself slows as more threads run it, lock or no lock: this
//     workload's throughput ceiling can be the memory system rather than the
//     serialized critical section, in which case the ladder stops mattering
//     long before the contention dial says it should.
// Which effect wins is not predictable from first principles, hence this file.
//
// THE WORK
//
// do_mem_work() in spinlock_bm_common.h, shared with the memory-work variants
// in overhead_bm.C: an AXPY chunk pass streaming a thread-local 8 MiB arena,
// coupled to the shared value at both ends like every work function here.
// clang vectorizes it to full-width FMAs; one thread streams from L3, a few
// threads' combined working set exceeds it, and from there every work unit is
// a DRAM-bandwidth transaction, which is the point.
//
// RUNNING IT
//
// Same filters as spinlock_tune_bm.C, with names under BM_mem instead of
// BM_tune:
//
//   ./spinlock_mem_bm --benchmark_filter='work:0/'             # saturation
//   ./spinlock_mem_bm --benchmark_filter='sweep:(base|shape)'  # the confirmation
//
// Run with --benchmark_repetitions=10 and read the mean: memory work re-rolls
// its cache/CCD placement with every repetition's fresh threads, and a single
// repetition carries ~10-14% CV from that lottery alone (see the repetition
// guidance in overhead_bm.C, where it was measured).
//
// Note that work:0 does not touch the arena at all, so the saturated column
// must reproduce spinlock_tune_bm.C's work:0 within noise -- the two files are
// the same program there. That is the cross-file calibration; BM_spinlock is
// the in-file one.
#include <atomic>

#include "spinlock.h"
#include "spinlock_tune.h"

#include "spinlock_bm_common.h"

// The benchmark body: BM_lock of spinlock_bm_common.h with do_mem_work in
// place of do_work -- same guarded unsigned long total, same pre-lock cast,
// same separate cache lines for the lock and the total.
template <typename Lock>
void BM_memlock(benchmark::State& state) {
  alignas(64) static Lock lock;
  alignas(64) static unsigned long shared_n;
  const long work = state.range(0);
  if (state.thread_index() == 0) shared_n = 0;
  // Construct and fault in this thread's arena OUTSIDE the timed region, with
  // one full pass over it: the first touch of 8 MiB is page faults and
  // zero-fill, not the steady-state streaming being measured. This matters
  // more than it looks: the framework spawns fresh threads -- and therefore
  // fresh thread_local arenas -- for every calibration round, and a
  // high-thread-count round runs so few iterations that the construction
  // storm otherwise dominates every one of them (it measured 7 ms/iteration
  // at 32 threads, ~300x the steady state). Skipped at work:0, which never
  // touches the arena and must stay identical to spinlock_tune_bm.C's
  // saturated column.
  if (work != 0) do_mem_work(1.0, mem_size/mem_chunk);
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    local_x = do_mem_work(local_x, work);
    const unsigned long n = static_cast<unsigned long>(1.0 + local_x);
    std::lock_guard guard(lock);
    benchmark::DoNotOptimize(shared_n += n);
  }
  state.SetItemsProcessed(state.iterations());
} // BM_memlock

// The same three contention levels as spinlock_tune_bm.C, by design: the
// experiment is "same dial, different work", so a divergence between the two
// files' rankings at the same work count is attributable to the work alone.
// One caveat has no analogue in the sin/cos file: here the cost of a work
// unit -- and with it the lock occupancy fraction -- depends on the thread
// count, because the work slows down as more threads share the memory system
// (measured on the development machine: ~6 ns/unit alone, ~150 ns/unit with
// eight threads streaming). The dial still orders the contention levels; it
// no longer pins them to fixed percentages.
#define MEM_ARGS \
  ->ArgName("work")->Arg(0)->Arg(30)->Arg(100) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// The shared configuration list, under BM_mem naming:
//   BM_mem/sweep:base/try:8/pause:0/yield:0/short:8/long:NS_1ms
#define TUNE_CONFIG(sweep, ntry, npause, nyield, nshort, longns)          \
  BENCHMARK_TEMPLATE(BM_memlock,                                            \
                     LadderedLock<BackOffParams{ntry, npause, nyield,   \
                                               nshort, longns}>) \
      ->Name("BM_mem/sweep:" sweep "/try:" #ntry "/pause:" #npause       \
             "/yield:" #nyield "/short:" #nshort "/long:" #longns) MEM_ARGS

#include "spinlock_tune_configs.h"

// The header's SpinLock: must match the `sweep:base` line above. Any gap
// beyond run-to-run noise means TunableSpinLock has drifted from the real
// lock, and every number below it is being compared against the wrong
// baseline.
BENCHMARK_TEMPLATE(BM_memlock, SpinLock)->Name("BM_spinlock") MEM_ARGS;

BENCHMARK_MAIN();
