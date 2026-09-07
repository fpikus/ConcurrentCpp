// Where does a lock stop being the right answer?
//
// Three ways to add a thread-local result into a shared total, measured against
// each other as the amount of local work per update is swept from none to a lot:
//
//   BM_spinlock -- a plain unsigned long guarded by the tunable spinlock
//   BM_atomic   -- std::atomic<unsigned long>::fetch_add, wait-free
//   BM_cas      -- a compare_exchange loop, lock-free but with no back-off
//
// This is the experiment behind the claim in README.md that at low contention
// the balance inverts: the spinlock wins the throughput of the lock operations
// themselves, but the atomic has lower overhead, and it is the surrounding
// program whose throughput actually matters. Where the lines cross is
// hardware-specific, so the point of the file is to find the crossing on the
// machine in front of you, not to know it in advance.
//
// The three differ in more than their instruction counts, and the differences
// are the finding:
//   * The spinlock serializes; both atomic forms let every thread make progress
//     on the same line at once, which is faster only until the line stops
//     being obtainable.
//   * fetch_add always completes; the CAS loop retries, and its retry rate is
//     what makes it fall apart under contention, since a failed CAS has already
//     paid for the line.
//   * The atomics are relaxed, the lock is acquire/release. On x86 that costs
//     nothing (the same LOCK-prefixed instruction either way); elsewhere it is
//     part of the honest difference between a counter and a critical section.
//
// STRUCTURE
//
// Arguments are (shared, work) per iteration, as in the chapter's
// 03_overhead.C: `work` evaluations of local computation, and `shared` updates
// to the shared total. Both counts matter. More than one update per work chunk
// is not padding -- it models global state that takes several touches to
// maintain, and it is the only way to reach the sharing-heavy ratios (10:1,
// 100:1) where synchronization is nearly all a thread does.
//
// The one departure from the chapter's version is that the two are coupled.
// There the loops run back to back and neither depends on the other, so the
// compiler may schedule them as unrelated streams and nothing in the workload
// explains why the data is shared at all. Here the work chunk runs first and
// every update that follows applies its result, which is the realistic shape --
// compute locally, fold the answer into global state -- and matches the rest of
// the benchmarks in this directory (see do_work() in spinlock_bm_common.h).
//
// Each update is synchronized on its own: one lock acquisition, or one atomic
// RMW, per update. That keeps the comparison honest, since a lock amortized
// over a compound critical section would be measuring a different design.
//
// Reports items/s == shared updates/s, so the ratios are comparable to each
// other even though an iteration costs `shared` updates rather than one.
//
// Each mechanism runs under two kinds of work (the `_mem` names): the
// compute-heavy sin/cos chain, and the memory-streaming AXPY of
// spinlock_mem_bm.C. A thread doing register-resident work costs the others
// nothing; a thread streaming memory competes with every other thread for L3
// and DRAM bandwidth, and the first measurements say the crossover moves
// accordingly -- see the note at the registrations.
//
// RUNNING IT
//
//   ./overhead_bm --benchmark_filter='shared:1/work:0/'  # saturation only
//   ./overhead_bm --benchmark_filter='threads:32$'       # one thread count
//   ./overhead_bm --benchmark_filter='shared:(10|100)/'  # sharing-heavy
//   ./overhead_bm --benchmark_filter='_mem/'             # memory work only
//   ./overhead_bm --benchmark_filter='BM_(spinlock_s[18]|atomic|cas)/'  # sin/cos only
//
// Run the _mem benchmarks with --benchmark_repetitions=10 (and
// --benchmark_report_aggregates_only=true) and read the mean: the memory work
// re-rolls its cache/CCD placement with every repetition's fresh threads, and
// a single repetition carries ~10-14% CV from that lottery alone (measured on
// the development machine via the shared:0 control, whose three identical
// programs converge to within ~4% at 10 repetitions). The sin/cos benchmarks
// sit at ~2-4% CV and can afford fewer.
//
// Repetitions only average the noise that re-rolls per repetition. A process
// is itself one draw of everything fixed at exec -- ASLR code/stack layout,
// the physical pages the allocator keeps handing back, initial placement --
// and repetitions inside it can cluster tightly around a value another
// invocation will tightly miss (observed on the development machine: two
// mechanisms 34% apart at many sigma in one process, equal in another). For
// numbers that decide anything, invoke the binary 2-3 times and compare the
// per-run means: their spread is the real error bar, and the within-run
// stddev only its floor. Add --benchmark_enable_random_interleaving on long
// runs so that slow drift (thermal, background load) decorrelates from
// benchmark order instead of biasing whichever families run last.
//
// Read it as three curves per thread count and work kind. The number to find
// is the smallest `work` at which the atomic overtakes the spinlock -- that is
// the boundary of the spinlock's domain of applicability on this machine, and
// it need not be the same boundary for the two kinds of work. The shared:0
// line is the control: with no sharing at all the three mechanisms are the
// same program, so any gap between them there is the measurement's own noise
// floor.
#include <atomic>

#include "spinlock.h"
#include "spinlock_tune.h"

#include "spinlock_bm_common.h"

// The spinlocks under test: TWO configurations, because the machines disagree
// about which one is the saturation champion and the disagreement is itself a
// finding.
//
//   s1 -- the shipped ladder with the short-sleep tier cut to one round.
//         Best at maximum contention on the development machine (16-core
//         Ryzen 9 9950X): 244 M updates/s at 8 threads and 256 at 32, against
//         the shipped ladder's 216 and 192, at identical single-thread cost.
//   s8 -- the shipped SpinLock ladder itself. On a 2-socket 128-core Granite
//         Rapids (leslie), s1 turned out bistable at saturation -- two
//         identical-by-construction instantiations measured 70 and 37 M/s,
//         both far below this ladder's 132 -- so the dev-machine choice did
//         not transfer, and both configurations are measured everywhere.
using OverheadSpinLockS1 = LadderedLock<BackOffParams{.nshort = 1}>;
using OverheadSpinLockS8 = LadderedLock<BackOffParams{}>;

// The three update mechanisms, as interchangeable policies for BM_update below.
// Each owns its shared total -- and its lock, when it has one -- with every
// piece on its own cache line, so all three present the same access pattern:
// one 64-byte line of shared state, away from any lock word. The total is
// never read or reset between runs: nothing inspects it, and a reset would
// have to race with the other threads' updates to happen at all.

// A plain unsigned long behind a lock: the update is not atomic, and the lock
// is the only thing that makes it well defined.
template <typename Lock>
class GuardedUpdate {
  public:
  // Add `n` to the total under the lock.
  void operator()(unsigned long n) {
    lock_.lock();
    // DoNotOptimize, not decoration: nothing ever reads total_, so without it
    // the compiler may drop the accumulation and leave the benchmark timing
    // lock/unlock around an empty critical section. The atomic policies need
    // no such guard -- an atomic RMW cannot be elided -- so losing this one
    // would quietly hand the lock an advantage.
    benchmark::DoNotOptimize(total_ += n);
    lock_.unlock();
  }
  private:
  alignas(64) Lock lock_;
  alignas(64) unsigned long total_ = 0;
}; // class GuardedUpdate

// The wait-free floor: one LOCK-prefixed add, no retry, no waiting. Nothing
// here can be starved, but every thread's add still drags the line to its own
// core, which is what eventually limits it.
class FetchAddUpdate {
  public:
  void operator()(unsigned long n) {
    total_.fetch_add(n, std::memory_order_relaxed);
  }
  private:
  alignas(64) std::atomic<unsigned long> total_ {0};
}; // class FetchAddUpdate

// The lock-free retry loop: read the total, try to install the sum, repeat if
// somebody got there first. compare_exchange_weak is the right form inside a
// loop -- a spurious failure costs one more pass, and forbidding it would cost
// an inner loop on the platforms that have one. Deliberately without back-off:
// this is the shape a lock-free counter is usually written in, and how it
// degrades is the reason the spinlock's ladder exists.
class CasUpdate {
  public:
  void operator()(unsigned long n) {
    unsigned long expected = total_.load(std::memory_order_relaxed);
    while (!total_.compare_exchange_weak(expected, expected + n,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
      // compare_exchange_weak refreshed `expected` with the value it found.
    }
  }
  private:
  alignas(64) std::atomic<unsigned long> total_ {0};
}; // class CasUpdate

// The benchmark body, shared by every (mechanism, work) pair -- the work kind
// is a policy from spinlock_bm_common.h (SinCosWork or MemWork): one work chunk,
// then `shared` applications of its result to the shared total. The trailing
// DoNotOptimize keeps the self-feeding work chain alive on the shared:0
// control, where no update ever consumes it.
template <typename Update, typename Work>
void BM_update(benchmark::State& state) {
  alignas(64) static Update update;
  const long shared = state.range(0);
  const long work = state.range(1);
  Work::warmup(work);
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    local_x = Work::run(local_x, work);
    const unsigned long n = static_cast<unsigned long>(1.0 + local_x);
    for (long i = 0; i < shared; ++i) update(n);
  }
  benchmark::DoNotOptimize(local_x);
  state.SetItemsProcessed(state.iterations()*shared);
} // BM_update

// The dial, as (shared, work) pairs. Spaced roughly logarithmically because the
// crossing is expected somewhere in the middle and its location is what is
// being measured, not its exact value. Calibrated from ~15 ns per sin(cos)
// evaluation and ~4 ns for an uncontended update, the fraction of a thread's
// time spent on shared state runs from all of it to almost none:
//   300:1   -- synchronization is essentially the whole program
//   100:1   --
//    10:1   --
//     1:0   -- one update, no work at all: the saturated point
//     1:1   -- ~20%
//     1:3   -- ~10%
//     1:10  -- ~3%
//     1:30  -- ~1%
//     1:100 -- ~0.3%
//     1:300 -- ~0.1%
//    1:1000 -- ~0.03%, where the choice of mechanism should stop mattering
//     0:1   -- the control: no sharing, so all three are the same program and
//              must report the same time (items/s is 0 here by construction)
#define OVERHEAD_ARGS \
  ->ArgNames({"shared", "work"}) \
  ->Args({300, 1})->Args({100, 1})->Args({10, 1}) \
  ->Args({1, 0})->Args({1, 1})->Args({1, 3})->Args({1, 10}) \
  ->Args({1, 30})->Args({1, 100})->Args({1, 300})->Args({1, 1000}) \
  ->Args({1, 3000}) \
  ->Args({0, 1}) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// Every mechanism under both kinds of work, over the same full (shared, work)
// grid -- including the shared-heavy 100:1 and 10:1 ratios, which are where
// the lock's domain lives. Both lock configurations run under both kinds of
// work: comparing BM_spinlock_s1 against BM_spinlock_s8 in the same run also
// measures the instantiation-level sensitivity that unmasked s1 on leslie.
BENCHMARK_TEMPLATE(BM_update, GuardedUpdate<OverheadSpinLockS1>, SinCosWork)
    ->Name("BM_spinlock_s1") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, GuardedUpdate<OverheadSpinLockS8>, SinCosWork)
    ->Name("BM_spinlock_s8") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, FetchAddUpdate, SinCosWork)
    ->Name("BM_atomic") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, CasUpdate, SinCosWork)
    ->Name("BM_cas") OVERHEAD_ARGS;

// The memory-work variants. Dev-machine and Granite Rapids agree: the
// crossover sits at the same ratio as with sin/cos work, but the atomic's
// mid-range lead is wider (the lock runs at DRAM speed while the atomic runs
// from cache). Past work:~100 differences at a 0.1% sync fraction are
// cache-placement lottery, not mechanism.
BENCHMARK_TEMPLATE(BM_update, GuardedUpdate<OverheadSpinLockS1>, MemWork)
    ->Name("BM_spinlock_s1_mem") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, GuardedUpdate<OverheadSpinLockS8>, MemWork)
    ->Name("BM_spinlock_s8_mem") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, FetchAddUpdate, MemWork)
    ->Name("BM_atomic_mem") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, CasUpdate, MemWork)
    ->Name("BM_cas_mem") OVERHEAD_ARGS;

// The shipped SpinLock under the same body: the chosen configuration measured
// against what it would replace, not only against the atomics. Registered but
// outside the default test sets (run_sets.sh selects lock-versus-lock-free);
// select it with --benchmark_filter='BM_shipped'.
BENCHMARK_TEMPLATE(BM_update, GuardedUpdate<SpinLock>, SinCosWork)
    ->Name("BM_shipped") OVERHEAD_ARGS;
BENCHMARK_TEMPLATE(BM_update, GuardedUpdate<SpinLock>, MemWork)
    ->Name("BM_shipped_mem") OVERHEAD_ARGS;

BENCHMARK_MAIN();
