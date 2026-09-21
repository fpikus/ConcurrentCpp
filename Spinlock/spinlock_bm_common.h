// Shared infrastructure for every benchmark in this directory (spinlock_bm.C,
// spinlock_tune_bm.C, spinlock_rw_tune_bm.C, spinlock_mem_bm.C,
// spinlock_layout_bm.C, overhead_bm.C, seqlock_bm.C): the work functions all of
// them run, the two lock-agnostic benchmark bodies -- BM_lock() for one guarded
// update per iteration and BM_rw() for a read:write mix, the latter with the
// plain-SpinLock comparator (SpinLockData) both of its users measure against --
// and the argument sets: the classic contention dial (ARGS) and the read:write
// grid (RW_ARGS).
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

#include "spinlock.h"                   // the lock inside SpinLockData

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

// The read/write benchmark body, shared by every mechanism measured across a
// read:write mix -- the tunable-ladder Data of spinlock_rw_tune_bm.C, and the
// seqlock, spinlock and bare atomic of seqlock_bm.C. DataT is any class with
// this file's two-call interface, `void write(unsigned long)` (add the argument
// to the shared total) and `unsigned long read()` (copy the total out); Work is
// one of the work policies above. Each iteration does `writes` guarded
// increments followed by `reads` guarded loads. The two are batched rather than
// interleaved, which is what the mixes are meant to model -- at
// reads:100/writes:1 a thread makes one update and then reads for a while, the
// shape a read-mostly structure actually sees. Every thread does both, so a
// mechanism that serializes its writers is exercised with multiple writers
// throughout.
//
// Every shared-data access carries exactly one `work` burst of thread-local
// work -- the contention dial described at the top of this file. The two loops
// put the burst on opposite sides of the access, because the coupling runs the
// other way round:
//   writes: the work runs FIRST and its result is the increment handed to
//           write(), so the shared total depends on the computation rather than
//           on a constant.
//   reads:  the read runs first and its result SEEDS the work that follows it,
//           so the load is on the dependency path of everything after it.
// Either way it is one burst per lock acquisition, and that is what keeps
// `work` meaning the same thing at every mix: occupancy depends on `work` alone
// and not on the read:write ratio. The alternative shape -- one burst per
// iteration with a batch of accesses after it, which is what overhead_bm.C
// does, where the batch size is itself the dial -- would not. At the pure
// endpoints the two placements coincide.
//
// Sweeping `work` matters here rather than being inherited out of habit: the
// single-ladder sweep (spinlock_tune_bm.C) showed that the optimal back-off at
// ~1% lock occupancy is the opposite of the optimal back-off at saturation, so
// a read-versus-write comparison run only at work:0 answers the question in one
// regime out of two.
//
// Coupling the work to the shared value at both ends is the point of sharing
// data in the first place -- a thread computes something locally and folds it
// into global state, or reads global state and computes from it. Decoupled work
// would let the compiler schedule the two halves independently and would model
// a workload nobody has. The coupling matters more for an optimistic mechanism
// than for a plain lock: a seqlock read that nothing depended on would let the
// compiler hoist or drop the validation, and the protocol's cost is exactly
// what is being measured.
//
// The read seed is masked to its low bits before it becomes a double: over a
// long mixed run the shared total grows without bound, and cos() of a huge
// argument drops into libm's slow argument-reduction path, which would make the
// work chunk quietly get more expensive as the run proceeds. The dependency is
// what this needs, not the magnitude.
//
// Reports items/s == shared-data operations/s (reads plus writes), so the
// number is comparable across mixes even though an iteration costs
// `reads + writes` accesses rather than one.
template <typename DataT, typename Work>
void BM_rw(benchmark::State& state) {
  alignas(64) static DataT data;
  const long reads = state.range(0);
  const long writes = state.range(1);
  const long work = state.range(2);
  Work::warmup(work);                   // untimed; a no-op for SinCosWork
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    for (long i = 0; i < writes; ++i) {
      local_x = Work::run(local_x, work);
      data.write(static_cast<unsigned long>(1.0 + local_x));
    }
    for (long i = 0; i < reads; ++i) {
      local_x = Work::run(static_cast<double>(data.read() & 0xFF), work);
      // Pin each chunk's result: it is otherwise dead, since the next
      // iteration reseeds from data.read(), and a compiler that proves
      // sin/cos side-effect-free could keep the guarded loads while
      // discarding the work between them. (Today libm's sin/cos may set
      // errno, which happens to prevent that; the measurement should not
      // hang on a math flag.) The write loop needs no pin -- every chunk's
      // result is consumed by data.write().
      benchmark::DoNotOptimize(local_x);
    }
  }
  state.SetItemsProcessed(state.iterations()*(reads + writes));
} // BM_rw()

// The shipped SpinLock guarding a plain unsigned long, behind BM_rw()'s
// write()/read() interface: the common comparator for every read/write
// experiment, which is why it lives here rather than in either of them. Both
// users register it as BM_spinlock, and because it is one class the two
// binaries' BM_spinlock lines measure the same thing by construction. In
// spinlock_rw_tune_bm.C it is also the cross-check on the tunable lock's
// w:base/r:base configuration -- both paths take the one lock the one way
// SpinLock knows how; in seqlock_bm.C it is the mechanism the seqlock has to
// beat on reads.
//
// The lock and the value sit on separate cache lines, as everywhere in this
// directory: the coherence traffic on the lock word must not be conflated with
// the traffic on the data it guards. Here that separation is also what makes
// the experiment readable at all -- a reader dirties only the lock word, a
// writer dirties both, and merging the two lines would hide exactly the
// difference being measured.
class SpinLockData {
  public:
  // Take the lock, add `n` to the shared total, release. The increment is
  // passed in rather than being a constant 1 because it is the result of the
  // caller's work chunk -- see BM_rw().
  void write(unsigned long n) {
    lock_.lock();
    x_ += n;
    lock_.unlock();
  } // SpinLockData::write()

  // Take the lock, copy the value out, release. Returns the value because the
  // caller feeds it into its next work chunk. A reader dirties the lock word
  // exactly as a writer does -- the point a sequence lock is measured against.
  unsigned long read() {
    lock_.lock();
    const unsigned long x = x_;
    lock_.unlock();
    return x;
  } // SpinLockData::read()

  private:
  alignas(64) SpinLock lock_;           // the one lock, alone on its line
  alignas(64) unsigned long x_ = 0;     // plain, non-atomic: the lock is the
}; // class SpinLockData                // only thing making this well-defined

static const long numcpu = sysconf(_SC_NPROCESSORS_CONF);

// The classic four-point contention dial (work=0 hammers the lock, work=300
// rarely touches it), from 1 thread to all CPUs. Used by spinlock_bm.C; the
// tuning and overhead benchmarks define their own argument sets.
#define ARGS \
  ->ArgName("work")->Arg(0)->Arg(3)->Arg(30)->Arg(300) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// The read:write counterpart of ARGS, for BM_rw(): the arguments are
// (reads, writes, work) per iteration. It is shared so that every read/write
// experiment runs the same grid and its numbers line up with the others'. The
// name reads as the ratio -- reads:100 with writes:1 is the 100:1 read-mostly
// case -- and `work` sets the contention level: saturation, then the two
// low-occupancy points where the single-ladder sweep showed the optimum moving
// (work:0 is ~100% of a thread's time inside the shared-data access, work:30
// ~1%, work:100 ~0.3%).
//
// The pure endpoints are what each experiment is actually about; a mix can only
// interpolate between them, so the default test sets (run_sets.sh) run the
// endpoints and leave the mixes for when the endpoints show a spread. For the
// ladder sweep the endpoints are also self-calibrating: at reads:0/writes:1 the
// read ladder is never used, so every `r:*` line must collapse onto the
// baseline and the whole benchmark must reproduce the write-only sweep in
// spinlock_tune_bm.C, and reads:1/writes:0 is the mirror, where only the read
// ladder can move the number. For the sequence lock reads:1/writes:0 is where
// its readers should scale and the spinlock's should not, and reads:0/writes:1
// is where it pays for the protocol with nothing to show for it.
#define RW_ENDPOINTS(work) \
  ->Args({0, 1, work})                  /* 100% writers */ \
  ->Args({1, 0, work})                  /* 100% readers */

// The read:write mixes -- the case where the shared total actually grows (see
// the read-seed masking in BM_rw(), which exists for it), and the one that
// locates the crossover between the two endpoints.
#define RW_MIXES(work) \
  ->Args({1, 100, work})->Args({1, 10, work})->Args({1, 1, work}) \
  ->Args({10, 1, work})->Args({100, 1, work})

// The whole grid: both endpoints and all five mixes at each of the three
// contention levels, from 1 thread to all CPUs.
#define RW_ARGS \
  ->ArgNames({"reads", "writes", "work"}) \
  RW_ENDPOINTS(0) RW_ENDPOINTS(30) RW_ENDPOINTS(100) \
  RW_MIXES(0) RW_MIXES(30) RW_MIXES(100) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

#endif // INCLUDED_SPINLOCK_BM_COMMON_H
