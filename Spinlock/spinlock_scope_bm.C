// Google Benchmark harness of the lock scope benchmark: on which side of
// unlock() a thread should write the slot it has just claimed under the shipped
// SpinLock (spinlock.h). The measured operation, the variants, the shared
// layout, the handoff counter and the reasons for all of it are described in
// spinlock_scope_common.h, which both this harness and its hand-rolled twin,
// spinlock_scope_mbm.C, use. This file holds only what is specific to Google
// Benchmark: the measurement window, the throughput accounting and the
// registration.
//
// LAYOUT
//
// On top of the common blocks of spinlock_scope_common.h, this harness shares
// one more field between the threads of a run: the pointer to the per-thread
// records (time stamps and handoff counts), on a 128-byte block of its own
// after the common ones (struct BmShared below).
//
// MEASUREMENT WINDOW
//
// Google Benchmark (as of 1.9.5, src/benchmark_runner.cc) decides that a
// threaded run is long enough by comparing min_time against the time SUMMED
// over all threads. With UseRealTime(), each thread contributes the wall-clock
// time of its own benchmark loop. At t threads the default 0.5 s is therefore
// ~0.5/t s of wall clock, about 4 ms at 128 threads, measured from a cold
// start. So every thread count is registered separately with
// MinTime(kWindowSeconds*t) and MinWarmUpTime(kWarmupSeconds*t). No thread's
// loop can run longer than the run itself, so these give at least
// kWindowSeconds of wall-clock measurement after at least kWarmupSeconds of
// warm-up. The warm-up phase goes through the same code and is compared against
// the same summed time, so it needs the same scaling. The per-benchmark values
// take precedence over the seconds forms of --benchmark_min_time and
// --benchmark_min_warmup_time, which this binary therefore ignores.
// --benchmark_min_time=<N>x still works: an explicit iteration count wins over
// MinTime(), and fixes the number of iterations EACH thread runs. The run names
// carry min_time: and min_warmup_time: fields that grow with the thread count.
//
// THROUGHPUT ACCOUNTING
//
// Google Benchmark's items_per_second is the total item count divided by the
// MEAN of the threads' loop times (benchmark_runner.cc divides the summed time
// by the thread count). Every thread runs the same number of iterations, and
// under a batching lock the threads that win the streaks finish early and wait
// at the end barrier, so the mean loop time is shorter than the run and
// items_per_second overstates the throughput -- the more, the less fair the
// lock, and the variants here may differ exactly in fairness. So each run also
// reports:
//   wall_items_per_second -- total items (iterations times threads) divided by
//                            the wall-clock span from the end of the earliest
//                            first iteration of any thread to the end of the
//                            last thread's last iteration;
//   finish_spread         -- (last finish - first finish)/(that span), between
//                            0 and 1: 0 when all threads finish together, near
//                            1 when some thread finished right at the start;
//   handoffs_per_op       -- the handoffs of all threads (see HANDOFFS in
//                            spinlock_scope_common.h) divided by the total
//                            items: 0 at 1 thread. Like wall_items_per_second,
//                            an average over the whole run, the tail with
//                            fewer threads included (see below).
// Every thread stamps the end of its own first iteration, and the span starts
// at the earliest of those stamps: a thread that misses the lock's first burst
// can sleep through other threads' whole streaks, so any one thread's stamp,
// thread 0's included, could be late. Since every finish is at or after its
// own thread's first stamp, no finish precedes the span start, which is what
// keeps finish_spread within [0, 1]. The item count includes at most one
// iteration per thread from before the span starts, which is negligible. All
// three are plain values, set by thread 0 alone, and they stay next to
// items_per_second, which is kept for comparison with the other benchmarks in
// this directory. The time stamps cost one compare per iteration and the loop
// itself one more (the iteration count against the next stamp's and against
// the last), placed after the unlock, the variant's post-unlock store and the
// handoff comparison in every variant; both only read (in the builds checked,
// both compilers read the two bounds from stack slots). The rest of the loop
// bookkeeping is the compiler's to schedule, and some of it lands inside the
// critical section (register increments of the loop counter and of the
// previous index).
//
// A large finish_spread means that the tail of the run had fewer threads
// contending than its start, so even wall_items_per_second mixes contention
// levels, and so does handoffs_per_op. spinlock_scope_mbm.C runs the same loop
// body with every thread contending until a common stop, and its throughput
// (items_per_s) and handoffs_per_op do not have this problem.
//
// RUNNING IT
//
// Names are BM_spinlock_<placement>_<layout>/<arguments>, with placement
// in_scope, out_scope, in_scope_lockstore, in_scope_poststore,
// in_scope_postmiss, in_scope_inmiss, in_scope_postmiss_late or
// in_scope_postfence, so filters slice it (the filter is a regex). Note that
// 'in_scope' and '_packed/' also match the six in_scope_* variants;
// 'scope_<layout>' matches only the in/out pair:
//
//   ./spinlock_scope_bm --benchmark_filter='scope_packed/'     # 8-byte slots, in/out pair
//   ./spinlock_scope_bm --benchmark_filter='_ptrshared/'       # the pointer-placement controls
//   ./spinlock_scope_bm --benchmark_filter='scope_a(64|128)/'  # line-sized slots, in/out pairs
//   ./spinlock_scope_bm --benchmark_filter='_lockstore_'       # the lock-line store variants
//   ./spinlock_scope_bm --benchmark_filter='_poststore_'       # the post-unlock store variants
//   ./spinlock_scope_bm --benchmark_filter='_(post|in)miss_'   # the second-ring variants
//   ./spinlock_scope_bm --benchmark_filter='(_late|fence)_'    # the fenced variants
//   ./spinlock_scope_bm --benchmark_filter='threads:128$'      # one thread count
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "spinlock_scope_common.h"

#include "benchmark/benchmark.h"

// One thread's record of one run: its time stamps (see THROUGHPUT ACCOUNTING
// above) and its handoff count (see HANDOFFS in spinlock_scope_common.h). Each
// entry gets its own 128-byte block; it is written three times per run, so
// the padding matters little, but it costs nothing either.
struct alignas(128) ThreadStats {
  std::chrono::steady_clock::time_point first;  // end of this thread's first iteration
  std::chrono::steady_clock::time_point last;   // end of this thread's last iteration
  unsigned long handoffs;               // acquisitions after the first that followed another thread's
};

// The state that all threads of one benchmark share: the common blocks, then
// this harness's own block; see LAYOUT above. The offsets are checked in
// BM_scope(), for every instantiation that is registered.
template <typename Slot, PtrLine ptr_line>
struct BmShared {
  Shared<Slot, ptr_line> core;          // the lock, the slot index and the slot array pointer
  // Non-owning view of the per-thread records, one entry per thread; set by
  // thread 0 before the timed loop and cleared after it, like core.slots, and
  // touched by each thread only on its first and last iterations.
  alignas(128) ThreadStats* thread_stats {};
}; // struct BmShared

// The shared state, one object per (Slot, PtrLine) pair, not per variant: four
// objects (packed, a64, a128 and packed_ptrshared), each shared by the up to
// eight variants of its layout. Those variants run one after another, never at
// once, and every run resets what it uses (thread 0's setup in BM_scope()).
// Sharing the object gives the in/out variants of a layout, and the other
// variants of it, identical addresses for the lock, the index and the
// pointers. A namespace-scope variable template rather than a function-local
// static, so that both BM_scope() and timed_loop() address it directly.
// constinit: zero-initialized at load time, so there is no run-time guard.
template <typename Slot, PtrLine ptr_line>
static constinit BmShared<Slot, ptr_line> bm_shared {};

// The timed loop of BM_scope() for one thread: one scope_step() per iteration
// (lock, claim a slot, store, unlock, with the variant's stores on either
// side), then the handoff comparison (inside scope_step()) and one compare that
// decides whether to read the clock. Every thread reads it at the end of its
// first iteration and at the end of its last one (both at once if it runs only
// one); `mark` is the iteration count of the next reading. At the last
// iteration the thread also records its handoff count, before the end-of-loop
// barrier, so that thread 0 can read it.
//   state -- the benchmark state of this thread;
//   tid   -- this thread's index, which selects its entry of thread_stats.
//
// noinline: a function of its own, so that the only values live across the
// loop are the loop's own. lock()'s inlined back-off calls nanosleep(), so
// every value the loop carries must sit in one of the six callee-saved
// registers (on x86-64) or be spilled, and a spilled counter is a stack store
// between the unlock and the next lock (see E2 in spinlock_scope_common.h).
// Inlined into BM_scope(), the loop shares those registers with the setup and
// results code (and, with GCC, loses %rbp to a frame pointer for the stack
// realignment of that code's 512-bit vector spills), and the handoff counter is
// spilled.
template <typename Slot, Store store, PtrLine ptr_line>
[[gnu::noinline]] static void timed_loop(benchmark::State& state, int tid) {
  using Clock = std::chrono::steady_clock;
  BmShared<Slot, ptr_line>& s = bm_shared<Slot, ptr_line>;
  constexpr size_t idx_mask = kSlots - 1;
  const benchmark::IterationCount last = state.max_iterations;
  // One batch of all `last` iterations, counted by the inner loop, rather than
  // the range-for over `state`: the range-for keeps an iteration counter of
  // its own next to ours (n), one loop-carried register more (see noinline
  // above). KeepRunningBatch() calls StartKeepRunning() and FinishKeepRunning(),
  // the barriers, at the same points as the range-for, and state.iterations()
  // is the same. The loop's state is declared inside the batch, so that the
  // compiler sees v == n after every step and keeps one register for both.
  while (state.KeepRunningBatch(last)) {
    benchmark::IterationCount mark = 1; // iteration count of the next clock reading
    unsigned long v = 0;                // this thread's running count, stored into the slots
    HandoffCounter handoffs;            // this thread's handoff counter; see scope_step()
    for (benchmark::IterationCount n = 1; n <= last; ++n) {
      scope_step<store>(s.core, idx_mask, v, handoffs);
      if (n == mark) [[unlikely]] {
        const Clock::time_point now = Clock::now();
        if (n == 1) {
          s.thread_stats[tid].first = now;
          handoffs.breaks = 0;          // the first acquisition has no previous index
        }
        if (n == last) {
          s.thread_stats[tid].last = now;
          s.thread_stats[tid].handoffs = handoffs.breaks;
        }
        mark = last;
      } // if a time stamp is due
    } // timed loop
  } // the one batch
} // timed_loop()

// The benchmark body, one instantiation per registered variant.
//   Slot     -- the slot type: PackedSlot, Slot64 or Slot128;
//   store    -- the store placement (Store in spinlock_scope_common.h);
//   ptr_line -- where the pointer to the slot array lives.
// state.range(0) is the number of slots N, which must be kSlots; anything else
// skips the run with an error. The loop's index mask is the compile-time
// constant kSlots - 1, as in the twin, not a run-time value: one register less
// for the loop to carry (see timed_loop()).
//
// Thread 0 allocates the ring (and the second ring, for the variants that use
// one) and the per-thread records and resets the shared index before the timed
// loop, and computes the counters and frees the arrays after it. That is safe
// because the loop's start and end are barriers across all threads of a run in
// Google Benchmark (State::StartKeepRunning() and State::FinishKeepRunning()):
// the setup happens before any thread's first iteration, and every thread has
// written its record and stopped iterating before thread 0 gets past the end.
// Each run (every calibration round and the warm-up included) starts from
// fresh arrays and index 0.
template <typename Slot, Store store, PtrLine ptr_line>
void BM_scope(benchmark::State& state) {
  using SharedState = BmShared<Slot, ptr_line>;
  using Clock = std::chrono::steady_clock;

  // The layout claims of LAYOUT above, checked for this instantiation. The
  // common blocks are checked in spinlock_scope_common.h.
  static_assert(alignof(SharedState) == 128);
  static_assert(offsetof(SharedState, core) == 0);
  static_assert(offsetof(SharedState, thread_stats) == sizeof(Shared<Slot, ptr_line>));
  static_assert(sizeof(SharedState) == sizeof(Shared<Slot, ptr_line>) + 128);

  SharedState& s = bm_shared<Slot, ptr_line>;
  const size_t N = static_cast<size_t>(state.range(0));
  // assert() is compiled out in these -DNDEBUG builds; check it for real. Every
  // thread sees the same N, so either all threads return here or none does.
  if (N != kSlots) {
    state.SkipWithError("the number of slots must be kSlots");
    return;
  }
  const int tid = state.thread_index();

  std::unique_ptr<Slot[]> slots_owner;          // thread 0 only: owns the slot array for this run
  std::unique_ptr<Slot[]> slots2_owner;         // thread 0 only: owns the second ring, if the variant uses one
  std::unique_ptr<ThreadStats[]> stats_owner;   // thread 0 only: owns the per-thread records for this run
  if (tid == 0) {
    slots_owner = make_ring<Slot>(N);
    if constexpr (uses_second_ring<store>) slots2_owner = make_ring<Slot>(N);
    stats_owner = std::make_unique<ThreadStats[]>(static_cast<size_t>(state.threads()));
    s.core.slots = slots_owner.get();
    s.core.slots2 = slots2_owner.get();
    s.thread_stats = stats_owner.get();
    s.core.slot_idx = 0;
  } // thread 0 setup

  timed_loop<Slot, store, ptr_line>(state, tid);

  // Past the end-of-loop barrier: no thread touches the arrays any more, and
  // every thread's record is written. Thread 0 turns them into the three
  // counters (see THROUGHPUT ACCOUNTING above). They are plain
  // values, not rates, because Google Benchmark would divide a rate by the
  // mean loop time again; only thread 0 sets them, so summing the counters
  // over threads leaves them unchanged. The arrays are freed when their
  // owners go out of scope.
  if (tid == 0) {
    const int threads = state.threads();
    Clock::time_point start = s.thread_stats[0].first;          // earliest first-iteration end
    Clock::time_point first_finish = s.thread_stats[0].last;    // earliest last-iteration end
    Clock::time_point last_finish = first_finish;               // latest last-iteration end
    for (int i = 1; i != threads; ++i) {
      start = std::min(start, s.thread_stats[i].first);
      first_finish = std::min(first_finish, s.thread_stats[i].last);
      last_finish = std::max(last_finish, s.thread_stats[i].last);
    }
    // The span can be empty only in a run of one iteration per thread, and
    // then only if every stamp coincides; such runs are calibration rounds,
    // which are not reported, and they get zeros.
    const double span = std::chrono::duration<double>(last_finish - start).count();
    const double spread = std::chrono::duration<double>(last_finish - first_finish).count();
    const double items = static_cast<double>(state.iterations())*threads;
    state.counters["wall_items_per_second"] = benchmark::Counter(span > 0 ? items/span : 0.0);
    state.counters["finish_spread"] = benchmark::Counter(span > 0 ? spread/span : 0.0);
    unsigned long handoffs_total = 0;
    for (int i = 0; i != threads; ++i) handoffs_total += s.thread_stats[i].handoffs;
    state.counters["handoffs_per_op"] = benchmark::Counter(items > 0 ? static_cast<double>(handoffs_total)/items : 0.0);
    s.core.slots = nullptr;
    s.core.slots2 = nullptr;
    s.thread_stats = nullptr;
  } // thread 0 results
  state.SetItemsProcessed(state.iterations());
} // BM_scope()

// Register every variant at every thread count, then run what the command line
// selects. This replaces BENCHMARK_MAIN() because the time limits have to be
// set per thread count.
int main(int argc, char** argv) {
  // Same start-up as BENCHMARK_MAIN(), including its re-exec without ASLR.
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  const int numcpu = configured_cpu_count();
  if (numcpu == 0) return 1;

  // Thread count outside, variant inside: at each thread count all variants
  // run back to back, in the order of for_each_variant().
  for (int t : thread_counts(numcpu)) {
    for_each_variant([t]<typename Slot, Store store, PtrLine ptr_line>(const char* name) {
      benchmark::RegisterBenchmark(name, BM_scope<Slot, store, ptr_line>)
        ->ArgName("slots")->Arg(static_cast<int64_t>(kSlots))
        ->Threads(t)
        ->UseRealTime()
        ->MinTime(kWindowSeconds*t)
        ->MinWarmUpTime(kWarmupSeconds*t);
    });
  } // loop over thread counts

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
} // main()
