// Benchmark to tune the back-off ladder separately for readers and writers of
// the same lock. The workload is one shared non-atomic value behind one lock
// flag, exercised through a Data-like interface -- write() increments it,
// read() copies it out -- with the read:write mix as the dial. For the
// single-ladder sweep on a pure write workload see spinlock_tune_bm.C; for the
// lock against the atomic and std::mutex reference points see spinlock_bm.C.
//
// The benchmark body (BM_rw()), the (reads, writes, work) argument grid
// (RW_ARGS), and the plain-SpinLock comparator (SpinLockData) are the shared
// read/write harness in spinlock_bm_common.h, which seqlock_bm.C runs as well.
// What this file adds is the instrument and the sweep: the two-policy Data
// class and the named ladders it is instantiated with.
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

// The instrument, behind the harness's write()/read() interface: one plain
// unsigned long and one TunableSpinLock, the same shape as the SpinLockData
// comparator in spinlock_bm_common.h but with the two waiting policies as
// template parameters. This is the shape the hypothesis is about -- a value
// that is not itself atomic, so both the increment and the load must be inside
// the critical section, and the only thing that can differ between them is how
// a waiter waits.
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
  // result of the caller's work chunk -- see BM_rw() in spinlock_bm_common.h.
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
  alignas(64) TunableSpinLock lock_;    // one word, climbed two different ways
  alignas(64) unsigned long x_ = 0;     // plain, non-atomic: the lock is the
}; // class Data                        // only thing making this well-defined

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

// The shipped SpinLock on both paths, via the shared comparator SpinLockData
// (spinlock_bm_common.h): must match w:base/r:base. Any gap beyond run-to-run
// noise means the tunable lock has drifted from the real lock.
BENCHMARK_TEMPLATE(BM_rw, SpinLockData, SinCosWork)->Name("BM_spinlock") RW_ARGS;
BENCHMARK_TEMPLATE(BM_rw, SpinLockData, MemWork)->Name("BM_spinlock_mem") RW_ARGS;

BENCHMARK_MAIN();
