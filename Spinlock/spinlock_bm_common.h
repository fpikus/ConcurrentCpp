// Shared infrastructure for every benchmark in this directory (spinlock_bm.C,
// spinlock_tune_bm.C, spinlock_rw_tune_bm.C, overhead_bm.C): the work function
// all of them run, the lock-agnostic benchmark body, and the classic
// contention dial.
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
#include <vector>

#include "benchmark/benchmark.h"

// One chunk of thread-local work: `work` evaluations of x = sin(cos(x)),
// starting from `x` and returning the result. Every benchmark in this directory
// calls this one function, because numbers from different benchmarks are only
// comparable if the work they do outside the lock is bit-for-bit the same.
//
// The result is what couples the work to the shared data: callers fold it into
// the shared value rather than discarding it, so the update depends on the
// computation and the compiler cannot schedule the two independently. That is
// also the realistic shape -- the reason to share data is to combine
// thread-local results into global state.
//
// Note that sin(cos(x)) is a contraction: the chain converges to ~0.694 within
// a few steps, so the value folded in is numerically constant even though the
// dependency is real. Cost per evaluation is ~15 ns and stays there, which is
// the property the dial needs; an unbounded chain would drift into libm's slow
// argument-reduction path partway through a run.
static inline double do_work(double x, long work) {
  for (long i = 0; i < work; ++i) x = std::sin(std::cos(x));
  return x;
} // do_work()

// The memory-streaming counterpart of do_work(): one work unit is one AXPY
// pass (c[i] += a[i]*x) over a mem_chunk-double slice of a thread-local arena,
// the slice advancing (and wrapping) so successive units stream fresh lines
// instead of re-hitting a hot one. It computes nothing meaningful; it exists
// to keep the SIMD units fed from memory -- two loaded streams, an FMA, a
// stored stream, ~1.5 KB of traffic per unit (~6 ns from L3 on the
// development machine, the same order as one sin/cos evaluation, so the two
// work dials are roughly commensurate). The arena is 8 MiB per thread: one
// thread's working set lives in L3, a few threads' combined exceeds it, and
// from there every work unit is a memory-bandwidth transaction -- so unlike
// do_work(), the unit cost RISES with the number of running threads.
//
// Coupling, as everywhere: the seed scales what is stored, and the return
// value is loaded back from the chunk written last. The seed is folded into
// [0, 2) on entry; without that the feedback loop through the arena (the
// return value is a c element, c grows by a*x per visit, and the next seed
// tracks the return) diverges until the caller's double-to-unsigned-long cast
// is UB. The fold is skipped at work == 0, keeping the no-work point free of
// libm calls; the seed then passes through unchanged and stays bounded.
//
// Callers MUST fault the arena in outside the timed region, with
//   if (work != 0) do_mem_work(1.0, mem_size/mem_chunk);
// before the benchmark loop: the framework spawns fresh threads (and thus
// fresh thread_local arenas) for every calibration round, and the 8 MiB
// construction storm otherwise lands inside the handful of timed iterations a
// slow round runs and dominates them (measured 300x at 32 threads).
inline constexpr long mem_chunk = 64;       // doubles per unit (512 B/stream)
inline constexpr long mem_size = 1l << 19;  // doubles per stream: a + c = 8 MiB

struct MemArena {
  std::vector<double> a = std::vector<double>(mem_size, 1.0);  // read stream
  std::vector<double> c = std::vector<double>(mem_size, 0.0);  // r/w stream
  long offset = 0;                      // where the next work unit starts
}; // struct MemArena

static inline double do_mem_work(double x, long work) {
  if (work == 0) return x;
  x = std::fmod(x, 2.0);
  static thread_local MemArena arena;
  double* const c = arena.c.data();
  const double* const a = arena.a.data();
  long offset = arena.offset;
  for (long w = 0; w < work; ++w) {
    for (long i = 0; i < mem_chunk; ++i) {
      c[offset + i] += a[offset + i]*x;
    }
    offset = (offset + mem_chunk) & (mem_size - 1);
  } // loop over work units
  arena.offset = offset;
  // Read back from the chunk written last (offset has already advanced past
  // it), so the return value depends on the stores.
  return c[(offset - 1) & (mem_size - 1)];
} // do_mem_work()

// The two kinds of local work, as interchangeable policies for the benchmark
// bodies that take the work kind as a template parameter (overhead_bm.C,
// spinlock_rw_tune_bm.C). They occupy different machine resources -- registers
// versus the shared memory system -- which is the entire point of measuring
// both: a thread doing sin/cos work costs the other threads nothing, while a
// thread streaming memory competes with all of them for L3 and DRAM bandwidth.

// Register-resident work: the thread occupies nothing but its own core.
struct SinCosWork {
  static double run(double x, long work) { return do_work(x, work); }
  static void warmup(long) {}           // no per-thread state to fault in
}; // struct SinCosWork

// Memory-streaming work. warmup() faults the thread_local arena in and must be
// called outside the timed region -- see the contract on do_mem_work().
// Skipped at work:0, which never touches the arena.
struct MemWork {
  static double run(double x, long work) { return do_mem_work(x, work); }
  static void warmup(long work) {
    if (work != 0) do_mem_work(1.0, mem_size/mem_chunk);
  }
}; // struct MemWork

// The benchmark body, shared by every lock type: `work` local sin(cos)
// evaluations (the contention dial, outside the lock), then a guarded add of
// the work chunk's result to the shared total. The total is an unsigned long,
// as everywhere in this directory -- the work is double math because it should
// be compute-heavy, the shared state is an integer because that is what the
// wait-free fetch_add baseline (spinlock_bm.C) can update natively, and every
// mechanism must update the same kind of total for the comparison to mean
// anything. The cast happens before the lock is taken, so the critical section
// is the add alone.
//
// The lock and the total live on separate cache lines so the coherence traffic
// on the lock word is not conflated with the traffic on the data it guards.
// The DoNotOptimize is load-bearing: nothing ever reads shared_n, so without
// it the compiler may drop the accumulation and time an empty critical
// section. Reports items/s == guarded updates/s.
template <typename Lock>
void BM_lock(benchmark::State& state) {
  alignas(64) static Lock lock;
  alignas(64) static unsigned long shared_n;
  const long work = state.range(0);
  if (state.thread_index() == 0) shared_n = 0;
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    local_x = do_work(local_x, work);
    const unsigned long n = static_cast<unsigned long>(1.0 + local_x);
    std::lock_guard guard(lock);
    benchmark::DoNotOptimize(shared_n += n);
  }
  state.SetItemsProcessed(state.iterations());
} // BM_lock

static const long numcpu = sysconf(_SC_NPROCESSORS_CONF);

// The classic four-point contention dial (work=0 hammers the lock, work=300
// rarely touches it), from 1 thread to all CPUs. Used by spinlock_bm.C; the
// tuning and overhead benchmarks define their own argument sets.
#define ARGS \
  ->ArgName("work")->Arg(0)->Arg(3)->Arg(30)->Arg(300) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

#endif // INCLUDED_SPINLOCK_BM_COMMON_H
