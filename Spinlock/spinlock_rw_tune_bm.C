// Benchmark to tune the back-off ladder separately for readers and writers of
// the same lock. The workload is one shared non-atomic value behind one lock
// flag, exercised through a Data-like interface -- write() increments it,
// read() copies it out -- with the read:write mix as the dial. For the
// single-ladder sweep on a pure write workload see spinlock_tune_bm.C; for the
// lock against the atomic and std::mutex reference points see spinlock_bm.C.
//
// THE HYPOTHESIS
//
// write() and read() need the same lock to be correct: the value is a plain
// unsigned long, so a reader that runs concurrently with a writer has a data
// race, and no amount of atomics on the lock word changes that. But needing the
// same lock is not the same as needing the same waiting strategy. A reader
// holds the lock for a load; a writer holds it for a read-modify-write, and
// takes the value's cache line Exclusive, so the next acquirer -- reader or
// writer -- has to pull that line back across the machine. If holding times and
// hand-off costs differ by that much between the two, the round at which a
// waiter should stop spinning and park differs too, and one ladder cannot be
// right for both. Nothing about correctness distinguishes the two cases, which
// is precisely why the question has to be settled by measurement.
//
// TunableSpinLock (spinlock_tune.h) is the instrument: one flag, one holder
// at a time, readers excluding readers exactly as before -- but each call site
// picks its own waiting policy, lock<ReadP>() against lock<WriteP>().
//
// THE EXPLORATION STRATEGY
//
// The space is now two full ladders, so a cross product is out of the question.
// The sweep is built from a handful of named ladders -- the shipped one plus
// seven neighbours that lean either toward spinning harder or toward parking
// sooner -- and runs three groups:
//
//   r:*   the write ladder pinned to the shipped ladder, the read ladder
//         varied. This is the hypothesis under test.
//   w:*   the mirror image: read ladder pinned, write ladder varied. This is
//         the control that keeps the experiment honest. If tuning the read
//         ladder helps but the same change to the write ladder helps just as
//         much, the effect is about the read:write MIX, not about reading, and
//         the hypothesis has not been demonstrated.
//   both  two configurations with the ladders deliberately opposed, to see
//         whether the interesting settings are additive or fight each other.
//
// Every group is compared against `w:base/r:base`, which is the shipped ladder
// on both paths and therefore ordinary SpinLock behaviour.
//
// RUNNING IT
//
// Benchmark names carry both ladder tags and the mix, so filters slice it
// finely (the filter is a regex):
//
//   ./spinlock_rw_tune_bm --benchmark_filter='writes:0/'   # 100% readers
//   ./spinlock_rw_tune_bm --benchmark_filter='reads:0/'    # 100% writers
//   ./spinlock_rw_tune_bm --benchmark_filter='work:0/'     # saturation only
//   ./spinlock_rw_tune_bm --benchmark_filter='w:base/r:base|BM_spinlock'
//
// At a pure endpoint only one of the two ladders is ever climbed, so half the
// registered configurations are inert -- and that is deliberate: eight
// configurations that MUST produce the same number are an eight-way control on
// run-to-run and thread-placement noise, which is how a 13% placement bias was
// caught on the development machine. If run time is tight, drop the inert half:
//
//   ...--benchmark_filter='w:.*/r:base/reads:0/'   # writers, sweep only
//   ...--benchmark_filter='w:base/r:.*/writes:0/'  # readers, sweep only
#include <atomic>

#include "spinlock.h"
#include "spinlock_tune.h"

#include "spinlock_bm_common.h"

// The shared data: one plain unsigned long behind one lock. This is the shape
// the hypothesis is about -- a value that is not itself atomic, so both the
// increment and the load must be inside the critical section, and the only
// thing that can differ between them is how a waiter waits.
//
// There is no two-role lock class here: the lock is one TunableSpinLock word,
// and each operation binds its waiting policy at the call site through a
// LockAdapter -- the writer's policy in write(), the reader's in read() --
// held by std::lock_guard, so the unlock is RAII rather than a hand-written
// pair. The policies are the template parameters of Data, so one
// instantiation is one (writer policy, reader policy) experiment.
//
// The lock and the value sit on separate cache lines, as in spinlock_bm_common.h
// and for the same reason: the traffic on the lock word must not be conflated
// with the traffic on the data it guards. Here that separation is what makes
// the experiment readable at all -- a reader dirties only the lock word, a
// writer dirties both, and merging the two lines would hide exactly the
// difference being measured.
template <BackOffParams WriteP, BackOffParams ReadP>
class Data {
  public:
  // Take the lock the writer's way, add `n` to the shared total, release. The
  // increment is passed in rather than being a constant 1 because it is the
  // result of the caller's work chunk -- see BM_rw.
  void write(unsigned long n) {
    LockAdapter<WriteP> adapter(lock_);
    std::lock_guard guard(adapter);
    x_ += n;
  } // Data::write()

  // Take the lock the reader's way, copy the value out, release. Returns the
  // value because the caller feeds it into its next work chunk.
  unsigned long read() {
    LockAdapter<ReadP> adapter(lock_);
    std::lock_guard guard(adapter);
    return x_;
  } // Data::read()

  private:
  alignas(64) TunableSpinLock lock_;
  alignas(64) unsigned long x_ = 0;     // plain, non-atomic: the lock is the
}; // class Data                        // only thing making this well-defined

// The benchmark body, shared by every lock configuration: each iteration does
// `writes` guarded increments followed by `reads` guarded loads. The two are
// batched rather than interleaved, which is what the mixes are meant to model
// -- at reads:100/writes:1 a thread makes one update and then reads for a
// while, the shape a read-mostly structure actually sees.
//
// Every guarded operation is preceded by `work` evaluations of x = sin(cos(x))
// on a thread-local value -- the contention dial of spinlock_bm_common.h, and
// the reason it matters here: the single-ladder sweep showed that the optimal
// back-off at ~1% lock occupancy is the opposite of the optimal back-off at
// saturation, so a read-versus-write comparison run only at work=0 answers the
// question in one regime out of two.
//
// The work goes before each guarded operation rather than once per iteration
// with a batch of operations after it (the shape of overhead_bm.C, where the
// batch size is itself the dial). Per-operation is what keeps `work` meaning
// the same thing at every mix: one work burst per lock acquisition, so lock
// occupancy depends on `work` alone and not on the read:write ratio. At the
// pure endpoints the two placements coincide.
//
// The work is COUPLED to the shared value at both ends, which is the point of
// sharing data in the first place -- a thread computes something locally and
// folds it into global state, or reads global state and computes from it:
//   writes: the increment handed to write() is the result of the work chunk,
//           so the shared total depends on the computation, not on a constant.
//   reads:  the value read out seeds the next work chunk, so the load is on
//           the dependency path of everything that follows it.
// Decoupled work would let the compiler schedule the two halves independently
// and would model a workload nobody has.
//
// The read seed is masked to its low bits before it becomes a double: over a
// long mixed run the shared total grows without bound, and cos() of a huge
// argument drops into libm's slow argument-reduction path, which would make the
// work chunk quietly get more expensive as the run proceeds. The dependency is
// what this needs, not the magnitude.
//
// Reports items/s == guarded operations/s (reads plus writes), so the number is
// comparable across mixes even though an iteration costs `reads + writes` lock
// acquisitions rather than one.
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
} // BM_rw

// The shipped SpinLock behind the same Data interface: both paths take the
// one lock the one way SpinLock knows how. Runs alongside as the cross-check
// on w:base/r:base.
class SpinLockData {
  public:
  void write(unsigned long n) {
    lock_.lock();
    x_ += n;
    lock_.unlock();
  } // SpinLockData::write()

  unsigned long read() {
    lock_.lock();
    const unsigned long x = x_;
    lock_.unlock();
    return x;
  } // SpinLockData::read()

  private:
  alignas(64) SpinLock lock_;
  alignas(64) unsigned long x_ = 0;
}; // class SpinLockData

// The parameter sets the sweep is built from, written as their deltas from
// the shipped ladder (BackOffParams{} == what SpinLock ships). The rest are
// one step away in the two directions the hypothesis points -- "wait harder
// before parking" (pause, yield, patient, burst, spin) and "get out of the
// way sooner" (park), plus one that parks just as eagerly but wakes to
// re-check ten times as often (quick).
inline constexpr BackOffParams ladder_base {};
inline constexpr BackOffParams ladder_pause {.npause = 64};
inline constexpr BackOffParams ladder_yield {.nyield = 8};
inline constexpr BackOffParams ladder_patient {.nshort = 32};
inline constexpr BackOffParams ladder_burst {.ntry = 32};
inline constexpr BackOffParams ladder_spin {.nshort = 0, .long_sleep_ns = NS_off};
inline constexpr BackOffParams ladder_park {.nshort = 0};
inline constexpr BackOffParams ladder_quick {.long_sleep_ns = NS_100us};

// The mix and contention arguments, as (reads, writes, work) per iteration.
// The name reads as the ratio -- reads:100 with writes:1 is the 100:1
// read-mostly case -- and `work` sets the contention level: saturation, then
// the two low-occupancy points where the single-ladder sweep showed the
// optimum moving (work:0 is ~100% of a thread's time under the lock, work:30
// ~1%, work:100 ~0.3%).
//
// The pure endpoints are what decide whether readers and writers want
// different ladders at all; a mix can only interpolate between them, so the
// default test sets (run_sets.sh) run the endpoints and leave the mixes for
// when the endpoints show a spread. The endpoints are also self-calibrating:
// at reads:0/writes:1 the read ladder is never used, so every `r:*` line must
// collapse onto the baseline and the whole benchmark must reproduce the
// write-only sweep in spinlock_tune_bm.C; reads:1/writes:0 is the mirror,
// where only the read ladder can move the number.
#define RW_ENDPOINTS(work) \
  ->Args({0, 1, work})                  /* 100% writers */ \
  ->Args({1, 0, work})                  /* 100% readers */

// The read:write mixes -- the case where the shared total actually grows (see
// the read-seed masking in BM_rw, which exists for it).
#define RW_MIXES(work) \
  ->Args({1, 100, work})->Args({1, 10, work})->Args({1, 1, work}) \
  ->Args({10, 1, work})->Args({100, 1, work})

#define RW_ARGS \
  ->ArgNames({"reads", "writes", "work"}) \
  RW_ENDPOINTS(0) RW_ENDPOINTS(30) RW_ENDPOINTS(100) \
  RW_MIXES(0) RW_MIXES(30) RW_MIXES(100) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// Register one (write ladder, read ladder) pair under both kinds of work,
// named by the two tags and the work kind:
//   BM_rw/w:base/r:patient/reads:100/writes:1/...      (sin/cos work)
//   BM_rw_mem/w:base/r:patient/reads:100/writes:1/...  (memory-streaming work)
#define RW_BM(w, r) \
  BENCHMARK_TEMPLATE(BM_rw, Data<ladder_##w, ladder_##r>, SinCosWork) \
      ->Name("BM_rw/w:" #w "/r:" #r) RW_ARGS; \
  BENCHMARK_TEMPLATE(BM_rw, Data<ladder_##w, ladder_##r>, MemWork) \
      ->Name("BM_rw_mem/w:" #w "/r:" #r) RW_ARGS

// The control: the shipped ladder on both paths, i.e. plain SpinLock behaviour.
// Every line below differs from this one in exactly one ladder.
RW_BM(base, base);

// The hypothesis: writers wait as they always have, readers wait differently.
RW_BM(base, pause);
RW_BM(base, yield);
RW_BM(base, patient);
RW_BM(base, burst);
RW_BM(base, spin);
RW_BM(base, park);
RW_BM(base, quick);

// The control for the hypothesis: the same seven changes applied to the write
// ladder instead. A change that helps here as much as it helps above is not
// telling us anything about reading.
RW_BM(pause, base);
RW_BM(yield, base);
RW_BM(patient, base);
RW_BM(burst, base);
RW_BM(spin, base);
RW_BM(park, base);
RW_BM(quick, base);

// Deliberately opposed ladders: these only say something on the mixes, where
// both roles are active at once -- at a pure endpoint one of the two ladders
// is never climbed, which the test sets exploit as a control.
RW_BM(park, spin);
RW_BM(spin, park);

// The header's SpinLock on both paths: must match w:base/r:base. Any gap beyond
// run-to-run noise means the tunable lock has drifted from the real lock.
BENCHMARK_TEMPLATE(BM_rw, SpinLockData, SinCosWork)->Name("BM_spinlock") RW_ARGS;
BENCHMARK_TEMPLATE(BM_rw, SpinLockData, MemWork)->Name("BM_spinlock_mem") RW_ARGS;

BENCHMARK_MAIN();
