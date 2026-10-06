// Hand-rolled twin of spinlock_scope_bm.C (mbm = the hand-rolled
// MicroBenchMark, as in ../ConcurrentQueue): the same lock scope benchmark --
// on which side of unlock() a thread should write the slot it has just claimed
// under the shipped SpinLock (spinlock.h) -- measured with every thread
// contending for the whole measurement window.
//
// WHAT IS MEASURED
//
// The same fully inlined loop body as spinlock_scope_bm.C: both harnesses run
// scope_step() of spinlock_scope_common.h with SpinLock::lock() inlined into it
// (lock, claim a slot, store, unlock, with the store before or after the
// unlock, the extra store of the lockstore, poststore, postmiss, inmiss and
// postmiss_late variants, the fence of the postmiss_late and postfence
// variants, and the handoff comparison), on the same variants, under the same
// names, with the same rings, the same compile-time index mask and the same
// layout of the shared blocks. That header also describes the variants and why
// they exist, and why lock() is forced inline. The harness around the loop body
// differs, and so does the generated loop in one known detail: the loop ends
// with a load and compare of the stop flag, where the Google Benchmark loop
// compares its iteration count against the next time stamp's and against the
// last (both read from stack slots in the builds checked). Both are reads: in
// neither harness does the loop store anything between the unlock and the next
// lock beyond the variant's own store there (out_scope, poststore, postmiss,
// postmiss_late). They sit in the same place in every variant, so they do not
// affect comparisons between variants, but they can show when 1-thread rates
// are compared across the two harnesses.
//
// WHY A SECOND HARNESS
//
// Google Benchmark runs a fixed number of iterations in every thread. Under a
// batching lock the threads that win the streaks finish early and leave, so the
// tail of a run has fewer threads contending than its start, and the reported
// rate is a mix of contention levels -- even the wall-clock rate that
// spinlock_scope_bm.C adds for this reason (THROUGHPUT ACCOUNTING there). Its
// finish_spread counter shows how much: at higher thread counts it can be
// large, and at the worst a run is close to serial by the end. The distortion
// can differ between in_scope and out_scope, so it biases the very in/out
// comparison the benchmark is for.
// Here every thread instead runs the same loop until a common stop flag, and
// each counts its own iterations, so all threads contend for the whole window,
// and the throughput is the true total count divided by the window.
//
// HOW THE NUMBERS RELATE
//
// items_per_s here corresponds to wall_items_per_second of spinlock_scope_bm.C,
// not to its items_per_second (which divides by the mean per-thread loop time).
// At 1 thread there is no contention and no finish spread, so the two are
// expected to agree within run-to-run noise, apart from the codegen details
// under WHAT IS MEASURED. Where finish_spread is large they differ, and this
// harness gives the rate at a constant thread count. Fairness shows up here as
// unequal per-thread counts rather than as a finish spread; it is reported as
//   cv       -- the coefficient of variation of the per-thread counts
//               (population standard deviation divided by the mean): 0 when
//               every thread did the same number of operations;
//   min_rel  -- the smallest per-thread count divided by the mean: near 0 when
//               some thread was starved for most of the window;
//   max_rel  -- the largest per-thread count divided by the mean: up to the
//               thread count when one thread did nearly everything.
// All three are 0, 1, 1 at 1 thread. cv summarizes; min_rel and max_rel tell a
// starved thread apart from a dominant one. The handoff counter (HANDOFFS in
// spinlock_scope_common.h) adds
//   handoffs_per_op -- the handoffs of all threads within the window divided
//                      by items: 0 at 1 thread; the same figure as the Google
//                      Benchmark harness's handoffs_per_op counter, over the
//                      window instead of the whole run.
//
// PROTOCOL OF ONE RUN
//
// A run is one variant at one thread count. The main thread, which is not one
// of the workers:
//   1. allocates and faults in the ring (and the second ring, for the variants
//      that use one), resets the slot index and the flags, and starts the
//      workers (and, with --observer, the observer; see THE OBSERVER below),
//      which wait on a start latch;
//   2. arrives at the latch, which releases all workers at once;
//   3. sleeps for the warm-up time: the workers run the loop uncounted;
//   4. reads the clock (t0) and sets the `measure` flag: each worker, on seeing
//      it, notes its running count and handoff count and goes on running the
//      same loop;
//   5. sleeps for the window, reads the clock (t1) and sets the `stop` flag:
//      each worker, on seeing it, stores its counts since step 4 and exits;
//   6. joins the workers (and the observer) and frees the rings.
// The handoff counter runs through both phases, so a worker's first
// acquisition, which has no previous index, falls into the uncounted warm-up.
// items is the sum of the per-thread counts, window_s is t1 - t0, measured with
// steady_clock, and items_per_s is items/window_s. Each clock reading
// immediately precedes the flag store it times, so both ends of the window see
// the same delay from reading to store. The window is whatever the main thread
// actually slept, so a late wake-up lengthens the window but does not bias the
// rate.
//
// The start latch is std::latch: its waiters block (after at most a brief spin
// inside the standard library) rather than spin, which keeps them from
// competing for CPUs with the main thread while it is still creating the other
// workers. Their wake-up is staggered, but that falls into the warm-up, which
// is why the warm-up runs on the same threads and flows into the window without
// a barrier: when the window opens, every thread is already running and the
// contention is established. (A warm-up of 0 puts the latch wake-up into the
// window.)
//
// Each worker checks the flag once per operation, after the unlock and after
// the out-of-scope store: a relaxed load of the flag block, which is written
// only twice per run and so stays in the Shared state in every reader's cache.
// The flags are on their own 128-byte block, after the common ones (struct
// MbmShared below), so reading them never touches the contended lines. In and
// out variants differ only in where the store sits, as in the Google Benchmark
// harness, whose one compare per iteration sits in the same place.
//
// A worker that sees a flag in the check after an operation counts that
// operation in the phase it just left. So the operation in flight when a flag
// is set, and those completed while the flag store propagates (a cache-line
// transfer), are counted on the wrong side of it: at t0 they are left out of
// the window, at t1 added to it, at most a few operations per thread at each
// end. The two ends cancel to first order, and even without cancelling they are
// a handful per thread against millions of operations per window. They are not
// corrected for.
//
// Threads are not placed: the OS schedules them, as in the Google Benchmark
// harness.
//
// THE OBSERVER
//
// With --observer, the run also measures how long per operation the lock is
// visibly free: free as another core sees it, which is the window in which a
// waiter can take it. One more thread, the observer, not counted in the thread
// count, is started after the workers and released by the same latch. Through
// the warm-up it only waits for `measure`; from `measure` until `stop` it
// repeatedly
//   1. spins on steady_clock for a random interval, drawn from a xorshift64
//      generator, uniform within +-50% of the sampling period
//      (--observer-period, 1 us by default: 0.5 to 1.5 us), and then
//   2. loads the lock word once, relaxed (SpinLock::locked()), counting the
//      samples and those that saw the lock free.
// When it sees `stop` it writes its two counts, once, into a 128-byte block of
// its own, and exits. It writes nothing that a worker reads, and of the shared
// state it reads only the lock word and the flags, never the index or the
// rings. The wait is a spin, not a sleep: a sleeping syscall wakes up tens of
// microseconds late, by an amount that itself varies with the load.
// steady_clock rather than a cycle counter keeps it portable to the aarch64
// builds, at some tens of nanoseconds per reading, small against the default
// 1 us. The jitter makes the sampling instants independent of the lock's own
// rhythm: with a fixed period the observer could fall into step with a nearly
// periodic acquire-release cycle and keep seeing the same phase of it. Spread
// over an interval tens to hundreds of times longer than one operation, the
// samples are issued at uniformly distributed phases, so the fraction of them
// that see the lock free estimates the fraction of the time that it is free --
// with the caveat below. Each row reports
//   obs_samples    -- the samples taken in the window, about one per period;
//   obs_free_frac  -- the fraction of them that saw the lock free;
//   free_ns_per_op -- obs_free_frac*window_s*1e9/items: the visibly free time
//                     per operation, in nanoseconds.
// The free time counts every gap between one holder's unlock and the next
// acquisition by anyone: by a waiter, which is a handoff, or by the same
// holder coming back, which is not. At 1 thread all of it is of the second
// kind.
//
// The caveat: the jitter makes the times at which the loads are ISSUED uniform
// in phase, but a load reads the value the line holds when it is SERVED, and
// how long that takes depends on the line's state. A snoop can be deferred
// while an xchg holds the line, and under contention the observer's read can
// queue behind the waiters' reads and RFOs, which crowd the line precisely in
// the free window. So obs_free_frac can be biased, and the direction of the
// bias has not been established. --validate-observer (below) calibrates it for
// programmed phases: at a 10 us period it finds no bias beyond the worker's own
// timing (a roughly constant extra held time of one to two clock readings per
// cycle, over programmed cycles from about 100 ns to 1 us), while at the
// default 1 us the observer perturbs the measurement (it adds held time at long
// phases and slows a lone worker measurably). The contended windows under
// MECHANISM in spinlock_scope_common.h (W) are therefore sampled at 10 us.
//
// The observer perturbs what it measures. Each sample pulls the lock's line
// into the observer's cache, and the next write to it (an xchg or an unlock)
// has to take it back: about one extra transfer of the lock line per sampling
// period. Under contention that is small next to the waiters' own polling of
// the same line; at 1 thread, where nothing else reads it, it is not. The
// perturbation scales with the sampling rate, while a bias from the order in
// which the line serves its readers does not, so repeating a run with a longer
// --observer-period (e.g. 10 us) tells the two apart.
//
// The observer also takes a CPU of its own, so a thread count that leaves none
// for it (threads + 1 more than the CPU count) is skipped, with a note on
// stderr, rather than run oversubscribed: there, a worker descheduled while
// holding the lock would hold it for a whole time slice and distort exactly
// the free time being measured. If that skips every thread count, the harness
// exits with status 2. For the largest possible count, pass --threads with the
// CPU count minus 1. The rule counts logical CPUs, and threads are not placed,
// so with SMT the observer's spin can share a physical core with a worker,
// possibly the one holding the lock, and slow it down; runs at thread counts
// up to the number of physical cores minus 1 leave the scheduler room to
// avoid that (it does not promise to). Without --observer none of this runs:
// there is no extra thread, and the three CSV columns are empty.
//
// VALIDATING THE OBSERVER
//
// --validate-observer=<held_ns>,<free_ns> runs, instead of the variants, one
// worker (validation_worker()) that repeats lock(); a clock spin of held_ns;
// unlock(); a clock spin of free_ns, alone on a lock of its own, with the
// observer sampling as usual (at --observer-period, through the same protocol,
// for --reps runs of --window each). Nothing else touches the lock, and the
// spins touch no line that another thread uses, so the lock word is exactly
// as programmed: 1 from the xchg until the unlock, a release store, commits;
// 0 from then until the next xchg. Each run prints the measured obs_free_frac
// next to the expected free fraction, both as programmed,
// free_ns/(held_ns + free_ns), and as timed by the worker's own clock (the
// clock readings stretch short phases, and a sample delays the worker's next
// write by a line transfer). With long phases (microseconds) this calibrates
// the observer's bias where the line's state changes rarely; short ones (tens
// of nanoseconds, near the phases of a contended lock) show where the
// service-time bias starts. It does not reproduce contention: no waiters queue
// on the line. It needs 2 CPUs (exit status 2 otherwise) and ignores --threads,
// --filter and --observer.
//
// SYNCHRONIZATION
//
// The setup in step 1 (the ring pointers, the slot index, the cleared flags)
// happens before the workers exist, and the completion of the std::jthread
// constructor synchronizes-with the start of the worker. The flags are relaxed
// because they publish nothing: seeing one only moves a worker to its next
// phase, and all that is needed is that the store becomes visible, which the
// standard asks of implementations ([atomics.order]) and the hardware does
// within the coherence latency. Each worker writes its counts to its own
// elements of two vectors once, before it returns, and the completion of a
// thread synchronizes-with the return of its join(), so the main thread reads
// the counts after the joins without a data race. The observer publishes its
// counts the same way, once, into its own block. The slot index is guarded by
// the lock, and the ring stores are atomic (see scope_step()). The rings are
// freed only after the joins. This harness has not been run under TSan.
//
// ORDER
//
// For each repetition, for each thread count, every selected variant (20
// without a filter), in the order of for_each_variant(): the in/out pair of
// each layout runs back to back, next to the lockstore, poststore, postmiss,
// inmiss, postmiss_late and postfence variants of that layout where there are
// any. The variant order is reversed whenever the repetition number plus the
// index of the thread count in the list is odd, so it alternates from one
// thread count to the next and from one repetition to the next: even a single
// repetition runs half of its thread counts in-first and half out-first, and a
// slow drift in the machine (clock, temperature) biases neither side of the
// pairs. There is no random shuffle.
//
// RUNNING IT
//
//   ./spinlock_scope_mbm [--window=SECONDS] [--warmup=SECONDS] [--reps=N]
//                        [--threads=T1,T2,...] [--filter=REGEX] [--observer]
//                        [--observer-period=US]
//                        [--validate-observer=HELD_NS,FREE_NS]
//
// The defaults are the Google Benchmark harness's window and warm-up (1 s and
// 0.5 s), 1 repetition, and the thread counts of the Google Benchmark harness:
// the powers of two up to the CPU count, then the CPU count itself, and no
// observer (--observer adds it, with a mean sampling wait of --observer-period
// microseconds, 1 by default; see THE OBSERVER above). --filter
// selects variants by a regex search on the variant name (thread counts are
// chosen with --threads). Names are BM_spinlock_<placement>_<layout>, with
// placement in_scope, out_scope, in_scope_lockstore, in_scope_poststore,
// in_scope_postmiss, in_scope_inmiss, in_scope_postmiss_late or
// in_scope_postfence; 'in_scope' and '_packed$' also match the six in_scope_*
// variants, 'scope_<layout>' only the in/out pair:
//
//   ./spinlock_scope_mbm --filter='scope_packed$'     # 8-byte slots, in/out pair
//   ./spinlock_scope_mbm --filter='_ptrshared$'       # the pointer-placement controls
//   ./spinlock_scope_mbm --filter='scope_a(64|128)$'  # line-sized slots, in/out pairs
//   ./spinlock_scope_mbm --filter='_lockstore_'       # the lock-line store variants
//   ./spinlock_scope_mbm --filter='_poststore_'       # the post-unlock store variants
//   ./spinlock_scope_mbm --filter='_(post|in)miss_'   # the second-ring variants
//   ./spinlock_scope_mbm --filter='(_late|fence)_'    # the fenced variants
//   ./spinlock_scope_mbm --threads=128 --reps=5 > scope.csv
//   ./spinlock_scope_mbm --observer --filter='scope_(packed|a64)$' --threads=1,8
//   ./spinlock_scope_mbm --observer --observer-period=10 --filter='scope_packed$' --threads=8
//   ./spinlock_scope_mbm --validate-observer=200,800 --reps=3
//
// stdout is CSV only: a header line, then one line per run, written as each
// run finishes:
//   variant,threads,rep,window_s,items,items_per_s,cv,min_rel,max_rel,handoffs_per_op,
//   obs_samples,obs_free_frac,free_ns_per_op
// (one line in the output; rep counts from 0; the last three columns are empty
// without --observer). stderr gets the parameters at the start and, at the
// end, the median of each column over the repetitions of each
// (variant, threads) cell, the observer's columns only with --observer. With
// --validate-observer the CSV lines are instead
//   held_ns,free_ns,period_us,rep,window_s,cycles,cycle_ns,held_timed_ns,
//   clock_read_ns,obs_samples,obs_free_frac,expected_free_frac,timed_free_frac
// (again one line), with cycle_ns = window_s*1e9/cycles, held_timed_ns the
// held phase by the worker's clock, clock_read_ns the cost of one clock
// reading, expected_free_frac = free_ns/(held_ns + free_ns) and
// timed_free_frac = 1 - held_timed_ns/cycle_ns, and stderr gets a readable
// line per run. held_timed_ns under-counts the held phase by about one clock
// reading (the part of the first reading after the xchg, the part of the last
// before the unlock, and the unlock's commit), so obs_free_frac is expected to
// sit a steady ~clock_read_ns/cycle_ns below timed_free_frac; that offset is
// agreement, not observer bias.
//
// The exit status is 0 on success, 1 if the CPU count is unavailable or a run
// or the output failed, and 2 for a bad command line, or when --observer
// leaves no thread count to run (see THE OBSERVER above) or
// --validate-observer finds fewer than 2 CPUs.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <latch>
#include <memory>
#include <regex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "spinlock_scope_common.h"

// The flags the main thread uses to move the workers from one phase to the
// next; see PROTOCOL OF ONE RUN above. Each is written once per run, by the
// main thread, and read by every worker after every operation.
struct alignas(128) Flags {
  std::atomic<bool> measure {false};    // set at the start of the measured window
  std::atomic<bool> stop {false};       // set at its end
};

// The state that all threads of one variant share: the common blocks, then the
// flags on a block of their own. The offsets are checked in run_variant().
template <typename Slot, PtrLine ptr_line>
struct MbmShared {
  Shared<Slot, ptr_line> core;          // the lock, the slot index and the slot array pointer
  Flags flags;                          // the phase flags
}; // struct MbmShared

// The outcome of one run.
struct RunResult {
  double window_s;                      // the measured window, t1 - t0, in seconds
  std::vector<unsigned long> counts;    // operations per thread within the window
  unsigned long long handoffs;          // handoffs within the window, summed over the threads
  unsigned long long obs_samples;       // the observer's lock-word samples in the window; 0 without it
  unsigned long long obs_free;          // of those, the samples that saw the lock free
};

// The shared state, one object per (Slot, PtrLine) pair, not per variant: four
// objects (packed, a64, a128 and packed_ptrshared), each shared by the up to
// eight variants of its layout. Those variants run one after another, never at
// once, and every run resets what it uses (step 1 in run_variant()). Sharing
// the object gives the in/out variants of a layout, and the other variants of
// it, identical addresses for the lock, the index, the pointers and the flags.
// A namespace-scope variable template rather than a function-local static, so
// that both run_variant() and run_phase() address it directly. constinit:
// zero-initialized at load time, so there is no run-time guard.
template <typename Slot, PtrLine ptr_line>
static constinit MbmShared<Slot, ptr_line> mbm_shared {};

// A worker's counters, carried from the warm-up phase into the measured one.
struct WorkerCounts {
  unsigned long v = 0;                  // running count of operations, stored into the slots
  HandoffCounter handoffs;              // see scope_step()
};

// One phase of a worker: run scope_step() until the phase's flag reads true
// (`measure` ends the warm-up, `stop` the measured phase). The flag is checked
// after every operation, after the unlock, the variant's post-unlock store and
// the handoff comparison, never before the first operation, so at least one
// operation runs.
//   Slot, store, ptr_line -- the variant, as for scope_step();
//   measured              -- false for the warm-up, true for the window;
//   c                     -- the worker's counters, advanced by the phase.
//
// noinline, and the counters copied into locals for the loop and written back
// once at the end: so that the only values live across the loop are the loop's
// own. lock()'s inlined back-off calls nanosleep(), so every value the loop
// carries must sit in one of the six callee-saved registers (on x86-64) or be
// spilled, and a spilled counter is a stack store between the unlock and the
// next lock (see E2 in spinlock_scope_common.h). Inlined into the worker, GCC
// spills the running count in one phase and the handoff counter in the other.
template <typename Slot, Store store, PtrLine ptr_line, bool measured>
[[gnu::noinline]] static void run_phase(WorkerCounts& c) {
  MbmShared<Slot, ptr_line>& s = mbm_shared<Slot, ptr_line>;
  constexpr size_t idx_mask = kSlots - 1;
  const std::atomic<bool>& flag = measured ? s.flags.stop : s.flags.measure;
  unsigned long v = c.v;
  HandoffCounter handoffs = c.handoffs;
  do {
    scope_step<store>(s.core, idx_mask, v, handoffs);
  } while (!flag.load(std::memory_order_relaxed));
  c.v = v;
  c.handoffs = handoffs;
} // run_phase()

// The observer's results, on a 128-byte block of their own, so that its one
// write lands on no line that another thread uses; see THE OBSERVER above.
struct alignas(128) ObserverCounts {
  unsigned long long samples = 0;       // lock-word samples taken in the window
  unsigned long long free_samples = 0;  // of those, the samples that saw the lock free
};
static_assert(sizeof(ObserverCounts) == 128);

// The observer thread of one run; see THE OBSERVER above. Waits at the start
// latch, then, spinning, for `measure`; then takes one sample of the lock word
// after each random wait until it sees `stop`, and writes its counts to `out`.
//   lock      -- the run's lock; only read, by SpinLock::locked(), a relaxed
//                load;
//   flags     -- the run's phase flags; only read;
//   period_ns -- the mean wait between samples, in ns, at least 1 and at most
//                10^9 (--observer-period); the wait is uniform within +-50% of
//                it;
//   start     -- the start latch, shared with the other threads of the run;
//   out       -- the observer's own block, written once, before it returns.
// The counts are locals until the end, so the loop stores nothing but, at
// most, to the observer's own stack. One function for every variant: it
// depends on none of them. The flag is checked after the sample, so at least
// one sample is taken, and the last one can fall just after `stop` is set.
[[gnu::noinline]] static void observe(const SpinLock& lock, const Flags& flags, std::uint64_t period_ns,
                                      std::latch& start, ObserverCounts& out) {
  using Clock = std::chrono::steady_clock;
  std::uint64_t rng = 0x9e3779b97f4a7c15;       // xorshift64 state: any value but 0
  const std::uint64_t min_wait_ns = period_ns/2;        // the shortest wait; the longest is period_ns longer
  // Spin on the clock for a random interval, uniform in [period_ns/2,
  // period_ns/2 + period_ns] ns, so about period_ns on average: the next
  // xorshift64 (Marsaglia's 13, 7, 17) value, reduced modulo period_ns + 1
  // (the modulo bias is below period_ns*2^-64).
  const auto random_wait = [&rng, min_wait_ns, period_ns]() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    const std::uint64_t wait_ns = min_wait_ns + rng % (period_ns + 1);
    const std::chrono::nanoseconds interval(static_cast<std::chrono::nanoseconds::rep>(wait_ns));
    const Clock::time_point until = Clock::now() + std::chrono::duration_cast<Clock::duration>(interval);
    while (Clock::now() < until) {}
  }; // random_wait

  start.arrive_and_wait();
  while (!flags.measure.load(std::memory_order_relaxed)) random_wait();
  unsigned long long samples = 0;
  unsigned long long free_samples = 0;
  do {
    random_wait();
    ++samples;
    free_samples += !lock.locked();
  } while (!flags.stop.load(std::memory_order_relaxed));
  out.samples = samples;
  out.free_samples = free_samples;
} // observe()

// The main thread's part of a run, steps 1 (starting the threads) to 6 of
// PROTOCOL OF ONE RUN: start the threads, release them at the latch, sleep
// through the warm-up, open the window (t0, then `measure`), sleep through it,
// close it (t1, then `stop`), and join the threads. Returns the measured
// window, t1 - t0, in seconds. Used by every variant's run and by
// --validate-observer.
//   flags        -- the run's phase flags, cleared by the caller;
//   start        -- the start latch, which expects `participants` arrivals;
//   participants -- the threads that meet at the latch: every thread that
//                   `launch` starts, and the main thread;
//   warmup_s     -- seconds of uncounted warm-up, at least 0;
//   window_s     -- seconds of measurement requested, more than 0;
//   launch       -- called once with an empty vector, into which it starts
//                   every thread of the run (emplace_back() of a jthread).
// If `launch` throws (a thread could not be started), the threads it did start
// are let finish and joined before the exception propagates: both flags are
// set and the latch is counted down for the threads that were not started and
// for the main thread, so each started thread gets through the latch, sees
// both flags at its first check, and exits. The caller's data that those
// threads use must outlive this call, on both paths.
template <typename Launch>
static double run_window(Flags& flags, std::latch& start, std::ptrdiff_t participants, double warmup_s,
                         double window_s, Launch&& launch) {
  using Clock = std::chrono::steady_clock;
  Clock::time_point t0;                 // the start of the window: just before `measure` is set
  Clock::time_point t1;                 // its end: just before `stop` is set
  {
    // jthread joins on destruction, at the end of this block or while an
    // exception propagates out of it.
    std::vector<std::jthread> started;
    started.reserve(static_cast<size_t>(participants - 1));
    try {
      launch(started);
    } catch (...) {
      flags.measure.store(true, std::memory_order_relaxed);
      flags.stop.store(true, std::memory_order_relaxed);
      start.count_down(participants - static_cast<std::ptrdiff_t>(started.size()));
      started.clear();                  // joins the threads that did start
      throw;
    } // starting the threads

    start.arrive_and_wait();
    std::this_thread::sleep_for(std::chrono::duration<double>(warmup_s));
    t0 = Clock::now();
    flags.measure.store(true, std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::duration<double>(window_s));
    t1 = Clock::now();
    flags.stop.store(true, std::memory_order_relaxed);
  } // threads joined
  return std::chrono::duration<double>(t1 - t0).count();
} // run_window()

// One run of one variant, following PROTOCOL OF ONE RUN above.
//   Slot, store, ptr_line -- the variant, as for scope_step();
//   threads               -- the number of workers, at least 1;
//   warmup_s              -- seconds of uncounted warm-up, at least 0;
//   window_s              -- seconds of measurement requested, more than 0;
//   observer_period_ns    -- the observer's mean sampling wait in ns (see
//                            observe()), or 0 for no observer (THE OBSERVER
//                            above).
// Returns the measured window, the per-thread counts, the handoff total and
// the observer's counts (0 without it). Throws std::bad_alloc if an allocation
// fails and std::system_error if a thread cannot be started; the threads
// already started are stopped and joined first (see run_window()).
// Not reentrant, and not to be run concurrently with any other variant of the
// same layout: the shared state is one static object per (Slot, ptr_line),
// mbm_shared.
template <typename Slot, Store store, PtrLine ptr_line>
static RunResult run_variant(int threads, double warmup_s, double window_s, std::uint64_t observer_period_ns) {
  using SharedState = MbmShared<Slot, ptr_line>;

  // The flags are on their own 128-byte block after the common ones; the
  // common blocks are checked in spinlock_scope_common.h.
  static_assert(sizeof(Flags) == 128 && alignof(Flags) == 128);
  static_assert(alignof(SharedState) == 128);
  static_assert(offsetof(SharedState, core) == 0);
  static_assert(offsetof(SharedState, flags) == sizeof(Shared<Slot, ptr_line>));
  static_assert(sizeof(SharedState) == sizeof(Shared<Slot, ptr_line>) + 128);

  SharedState& s = mbm_shared<Slot, ptr_line>;

  // Step 1 of the protocol. The rings outlive the workers: they are declared
  // before them and freed after they are joined.
  const std::unique_ptr<Slot[]> ring = make_ring<Slot>(kSlots);
  std::unique_ptr<Slot[]> ring2;        // the second ring, for the variants that use one
  if constexpr (uses_second_ring<store>) ring2 = make_ring<Slot>(kSlots);
  // Per-thread operation and handoff counts within the window, one element per
  // worker, each written once by its worker before it exits.
  std::vector<unsigned long> counts(static_cast<size_t>(threads));
  std::vector<unsigned long> handoff_counts(static_cast<size_t>(threads));
  s.core.slots = ring.get();
  s.core.slots2 = ring2.get();
  s.core.slot_idx = 0;
  s.flags.measure.store(false, std::memory_order_relaxed);
  s.flags.stop.store(false, std::memory_order_relaxed);
  ObserverCounts observed;              // written by the observer, if any, as it exits
  // The threads that meet at the start latch: the workers, the observer if
  // there is one, and the main thread.
  const std::ptrdiff_t participants = static_cast<std::ptrdiff_t>(threads) + (observer_period_ns != 0 ? 1 : 0) + 1;
  std::latch start(participants);

  // A worker: wait for the start, run the warm-up phase, note the counts, run
  // the measured phase, store the counts of the measured phase. `v` is the
  // running count that scope_step() increments and stores into the slots.
  // The handoff counter runs through both phases, so the first acquisition in
  // the window compares with the last one of the warm-up; the thread's very
  // first acquisition, which has no previous index, is in the warm-up, which
  // is not counted.
  const auto worker = [&start, &counts, &handoff_counts](size_t t) {
    WorkerCounts c;
    start.arrive_and_wait();
    run_phase<Slot, store, ptr_line, false>(c);
    const WorkerCounts c0 = c;
    run_phase<Slot, store, ptr_line, true>(c);
    counts[t] = c.v - c0.v;
    handoff_counts[t] = c.handoffs.breaks - c0.handoffs.breaks;
  }; // worker

  // Steps 1 (starting the threads) to 6: the workers, then the observer, if
  // any. The ring pointers are cleared only after the threads are joined,
  // which run_window() does on both of its paths, since the workers read them
  // until they exit. On a failed start, each started worker runs one
  // operation per phase and the observer takes one sample.
  double measured_s = 0;                // the measured window, t1 - t0
  try {
    measured_s = run_window(s.flags, start, participants, warmup_s, window_s,
                            [&](std::vector<std::jthread>& started) {
      for (int t = 0; t != threads; ++t) started.emplace_back(worker, static_cast<size_t>(t));
      if (observer_period_ns != 0) {
        started.emplace_back(observe, std::cref(s.core.lock), std::cref(s.flags), observer_period_ns, std::ref(start),
                             std::ref(observed));
      }
    });
  } catch (...) {
    s.core.slots = nullptr;
    s.core.slots2 = nullptr;
    throw;
  } // running the threads
  s.core.slots = nullptr;
  s.core.slots2 = nullptr;

  unsigned long long handoffs = 0;
  for (unsigned long h : handoff_counts) handoffs += h;
  return RunResult{measured_s, std::move(counts), handoffs, observed.samples, observed.free_samples};
} // run_variant()

// The shared state of one --validate-observer run (VALIDATING THE OBSERVER
// above): the lock on a 128-byte block of its own, then the phase flags on
// theirs.
struct ValidationShared {
  alignas(128) SpinLock lock;           // the lock that the programmed worker holds and frees
  Flags flags;                          // the phase flags
}; // struct ValidationShared
static_assert(sizeof(ValidationShared) == 256);

// The programmed worker's results, on a 128-byte block of their own; written
// once, by the worker, before it returns.
struct alignas(128) ValidationCounts {
  unsigned long long cycles = 0;        // lock-hold-unlock-free cycles completed in the window
  double held_s = 0;                    // their held phases, summed, by the worker's clock
};

// One programmed cycle of validation_worker(): lock(); a clock spin of
// `held`; unlock(); a clock spin of `free_time`. Returns the held time by the
// calling thread's clock, from the first clock reading after lock() returns
// to the last reading of the held spin.
//   lock      -- the lock; the calling thread must be the only one to take it,
//                so lock() takes it with its first xchg and never backs off;
//   held      -- the programmed held time, at least 0;
//   free_time -- the programmed free time, at least 0.
// Both spins read only the clock and keep their state in registers or on the
// thread's own stack, so the lock word is 1 from the xchg until the unlock (a
// release store) commits and 0 from then until the next xchg, and nothing
// else in the cycle touches a line that another thread uses. The timed held
// time misses the xchg's completion before its first reading and the
// unlock's commit after its last, a fraction of a clock reading each, and any
// delay of the unlock's commit by the observer's reads of the line.
// always_inline and flatten, as for scope_step(): lock() and unlock() are
// inlined here as in the variants (Clang leaves lock() out of line from a
// lambda otherwise), and the binary keeps no out-of-line SpinLock::lock(),
// which is how a build is checked for the inlining that the variants need.
[[gnu::always_inline, gnu::flatten]] inline std::chrono::steady_clock::duration validation_cycle(
    SpinLock& lock, std::chrono::steady_clock::duration held, std::chrono::steady_clock::duration free_time) {
  using Clock = std::chrono::steady_clock;
  lock.lock();
  const Clock::time_point locked_at = Clock::now();
  Clock::time_point held_until = locked_at;     // the last reading of the held spin
  while (held_until - locked_at < held) held_until = Clock::now();
  lock.unlock();
  const Clock::time_point unlocked_at = Clock::now();
  while (Clock::now() - unlocked_at < free_time) {}
  return held_until - locked_at;
} // validation_cycle()

// The one worker of --validate-observer: repeat validation_cycle() through
// the warm-up and then the window, and count the cycles and the held time of
// the window by its own clock.
//   lock      -- the run's lock; this thread alone takes it;
//   flags     -- the run's phase flags; only read, once per cycle, after the
//                free spin;
//   held      -- the programmed held time, at least 0;
//   free_time -- the programmed free time, at least 0;
//   start     -- the start latch, shared with the observer and the main thread;
//   out       -- the worker's own block, written once, before it returns.
// Not one of the measured variants' loops: it shares no code with them.
[[gnu::noinline]] static void validation_worker(SpinLock& lock, const Flags& flags, std::chrono::nanoseconds held,
                                                std::chrono::nanoseconds free_time, std::latch& start,
                                                ValidationCounts& out) {
  using Clock = std::chrono::steady_clock;
  const Clock::duration held_d = std::chrono::duration_cast<Clock::duration>(held);
  const Clock::duration free_d = std::chrono::duration_cast<Clock::duration>(free_time);

  start.arrive_and_wait();
  while (!flags.measure.load(std::memory_order_relaxed)) validation_cycle(lock, held_d, free_d);
  unsigned long long cycles = 0;
  Clock::duration held_total {};        // the window's held phases, summed
  do {
    held_total += validation_cycle(lock, held_d, free_d);
    ++cycles;
  } while (!flags.stop.load(std::memory_order_relaxed));
  out.cycles = cycles;
  out.held_s = std::chrono::duration<double>(held_total).count();
} // validation_worker()

// The outcome of one --validate-observer run.
struct ValidationResult {
  double window_s;                      // the measured window, t1 - t0, in seconds
  unsigned long long cycles;            // the worker's cycles within the window
  double held_s;                        // their held phases, summed, by the worker's clock
  unsigned long long obs_samples;       // the observer's samples within the window
  unsigned long long obs_free;          // of those, the samples that saw the lock free
};

// One --validate-observer run: the programmed worker (validation_worker())
// and the observer (observe(), with mean wait `period_ns`), through the same
// protocol as a variant's run (run_window()). The arguments are as for those
// functions. Throws std::system_error if a thread cannot be started.
static ValidationResult validate_observer(std::chrono::nanoseconds held, std::chrono::nanoseconds free_time,
                                          std::uint64_t period_ns, double warmup_s, double window_s) {
  ValidationShared v;                   // lock free, flags clear
  ValidationCounts worked;              // written by the worker as it exits
  ObserverCounts observed;              // written by the observer as it exits
  constexpr std::ptrdiff_t participants = 3;    // the worker, the observer and the main thread
  std::latch start(participants);
  const double measured_s = run_window(v.flags, start, participants, warmup_s, window_s,
                                       [&](std::vector<std::jthread>& started) {
    started.emplace_back(validation_worker, std::ref(v.lock), std::cref(v.flags), held, free_time, std::ref(start),
                         std::ref(worked));
    started.emplace_back(observe, std::cref(v.lock), std::cref(v.flags), period_ns, std::ref(start),
                         std::ref(observed));
  });
  return ValidationResult{measured_s, worked.cycles, worked.held_s, observed.samples, observed.free_samples};
} // validate_observer()

// The cost of one steady_clock reading on the calling thread, in ns: the mean
// over 10^6 back-to-back readings. Reported with --validate-observer, as the
// scale of the timing error of its spins.
static double clock_read_ns() {
  using Clock = std::chrono::steady_clock;
  constexpr int kReadings = 1000000;
  const Clock::time_point first = Clock::now();
  Clock::time_point last = first;
  for (int i = 0; i != kReadings; ++i) last = Clock::now();
  return std::chrono::duration<double, std::nano>(last - first).count()/kReadings;
} // clock_read_ns()

// The fairness figures of one run; see HOW THE NUMBERS RELATE above.
struct Fairness {
  double cv;                            // standard deviation/mean of the per-thread counts
  double min_rel;                       // smallest count/mean
  double max_rel;                       // largest count/mean
};

// Compute the fairness figures of the per-thread counts `counts`, which must
// not be empty. A zero mean cannot happen (every worker runs at least one
// operation in the window) but gives all zeros rather than a division by 0.
static Fairness fairness(const std::vector<unsigned long>& counts) {
  const double n = static_cast<double>(counts.size());
  double sum = 0;
  for (unsigned long c : counts) sum += static_cast<double>(c);
  const double mean = sum/n;
  if (mean == 0) return Fairness{0, 0, 0};
  double sq = 0;                        // sum of squared deviations from the mean
  for (unsigned long c : counts) sq += (static_cast<double>(c) - mean)*(static_cast<double>(c) - mean);
  const auto [min_it, max_it] = std::minmax_element(counts.begin(), counts.end());
  return Fairness{std::sqrt(sq/n)/mean, static_cast<double>(*min_it)/mean, static_cast<double>(*max_it)/mean};
} // fairness()

// The median of `values`, which must not be empty; reorders them. For an even
// count, the mean of the two middle values.
static double median(std::vector<double>& values) {
  const size_t mid = values.size()/2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const double upper = values[mid];
  if (values.size() % 2 != 0) return upper;
  const double lower = *std::max_element(values.begin(), values.begin() + mid);
  return (lower + upper)/2;
} // median()

// A variant as the command line sees it: its name and its run function.
struct Variant {
  const char* name;                     // the run name, shared with the Google Benchmark harness
  RunResult (*run)(int, double, double, std::uint64_t); // the run_variant() instantiation
};

// The per-run figures of one (variant, threads) cell, one entry per
// repetition, collected for the summary at the end.
struct CellSamples {
  std::vector<double> window_s;         // measured window
  std::vector<double> items;            // total operations in the window
  std::vector<double> items_per_s;      // items/window_s
  std::vector<double> cv;               // Fairness::cv
  std::vector<double> min_rel;          // Fairness::min_rel
  std::vector<double> max_rel;          // Fairness::max_rel
  std::vector<double> handoffs_per_op;  // handoffs/items
  // The observer's figures (THE OBSERVER above); filled only with --observer.
  std::vector<double> obs_samples;      // RunResult::obs_samples
  std::vector<double> obs_free_frac;    // obs_free/obs_samples
  std::vector<double> free_ns_per_op;   // obs_free_frac*window_s*1e9/items
}; // struct CellSamples

// If `arg` is `name` followed by '=', return the text after the '='; otherwise
// nullptr. `arg` must be null-terminated, and so is the returned text.
static const char* option_value(const char* arg, std::string_view name) {
  const std::string_view a(arg);
  if (a.size() <= name.size() || !a.starts_with(name) || a[name.size()] != '=') return nullptr;
  return arg + name.size() + 1;
}

// Parse all of `text` as a finite double into `out`; false (and `out`
// unchanged) if it is not one. strtod() rather than from_chars(), which not
// every standard library in use implements for floating point.
static bool parse_double(const char* text, double& out) {
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(value)) return false;
  out = value;
  return true;
} // parse_double()

// Parse all of `text` as a decimal int into `out`; false (and `out`
// unchanged) if it is not one or does not fit.
static bool parse_int(std::string_view text, int& out) {
  int value = 0;
  const std::from_chars_result r = std::from_chars(text.data(), text.data() + text.size(), value);
  if (r.ec != std::errc() || r.ptr != text.data() + text.size()) return false;
  out = value;
  return true;
} // parse_int()

// Parse a comma-separated list of positive ints into `out`, replacing its
// contents; false (and `out` unspecified) if any element is not one, or the
// list is empty.
static bool parse_thread_list(std::string_view text, std::vector<int>& out) {
  out.clear();
  while (true) {
    const size_t comma = text.find(',');
    int t = 0;
    if (!parse_int(text.substr(0, comma), t) || t < 1) return false;
    out.push_back(t);
    if (comma == std::string_view::npos) return true;
    text.remove_prefix(comma + 1);
  } // loop over the list elements
} // parse_thread_list()

// Parse `text` as <held_ns>,<free_ns>, two non-negative decimal ints that are
// not both 0, into `held_ns` and `free_ns`; false (and both unchanged) if it is
// not that.
static bool parse_validation(std::string_view text, int& held_ns, int& free_ns) {
  const size_t comma = text.find(',');
  if (comma == std::string_view::npos) return false;
  int held = 0;
  int free_time = 0;
  if (!parse_int(text.substr(0, comma), held) || !parse_int(text.substr(comma + 1), free_time)) return false;
  if (held < 0 || free_time < 0 || (held == 0 && free_time == 0)) return false;
  held_ns = held;
  free_ns = free_time;
  return true;
} // parse_validation()

// Parse `text` as the observer's mean sampling wait in microseconds, between
// 0.001 and 10^6, into `period_ns`, rounded to whole nanoseconds; false (and
// `period_ns` unchanged) if it is not one.
static bool parse_period(const char* text, std::uint64_t& period_ns) {
  double period_us = 0;
  if (!parse_double(text, period_us) || period_us < 0.001 || period_us > 1e6) return false;
  period_ns = static_cast<std::uint64_t>(std::llround(period_us*1e3));
  return true;
} // parse_period()

// Run --validate-observer (VALIDATING THE OBSERVER above): `reps` runs of
// validate_observer() with the programmed phases `held_ns` and `free_ns` and
// the observer's mean wait `period_ns`, each run with the given warm-up and
// window. Writes a CSV line per run to stdout, and the parameters and a
// readable line per run to stderr. Returns main()'s exit status: 0, or 1 if a
// run or the output failed.
static int run_validation(const char* argv0, int held_ns, int free_ns, std::uint64_t period_ns, double warmup_s,
                          double window_s, int reps) {
  const double clock_ns = clock_read_ns();
  const double expected = static_cast<double>(free_ns)/(static_cast<double>(held_ns) + static_cast<double>(free_ns));
  const double period_us = static_cast<double>(period_ns)/1e3;
  std::fprintf(stderr, "# validate-observer held=%dns free=%dns period=%gus window=%gs warmup=%gs reps=%d "
               "clock_read=%.1fns\n", held_ns, free_ns, period_us, window_s, warmup_s, reps, clock_ns);
  std::printf("held_ns,free_ns,period_us,rep,window_s,cycles,cycle_ns,held_timed_ns,clock_read_ns,obs_samples,"
              "obs_free_frac,expected_free_frac,timed_free_frac\n");
  std::fflush(stdout);
  try {
    for (int rep = 0; rep != reps; ++rep) {
      const ValidationResult r = validate_observer(std::chrono::nanoseconds(held_ns), std::chrono::nanoseconds(free_ns),
                                                   period_ns, warmup_s, window_s);
      // Per-cycle figures by the worker's clock; the worker completes at
      // least one cycle and the observer takes at least one sample, but the
      // divisions are guarded anyway.
      const double cycles = static_cast<double>(r.cycles);
      const double cycle_ns = r.cycles > 0 ? r.window_s*1e9/cycles : 0.0;
      const double held_timed_ns = r.cycles > 0 ? r.held_s*1e9/cycles : 0.0;
      const double timed_free = cycle_ns > 0 ? 1 - held_timed_ns/cycle_ns : 0.0;
      const double obs_free_frac =
          r.obs_samples > 0 ? static_cast<double>(r.obs_free)/static_cast<double>(r.obs_samples) : 0.0;
      std::printf("%d,%d,%g,%d,%.6f,%llu,%.2f,%.2f,%.2f,%llu,%.6f,%.6f,%.6f\n", held_ns, free_ns, period_us, rep,
                  r.window_s, r.cycles, cycle_ns, held_timed_ns, clock_ns, r.obs_samples, obs_free_frac, expected,
                  timed_free);
      std::fflush(stdout);
      std::fprintf(stderr, "rep %d: obs_free_frac %.4f; expected %.4f as programmed, %.4f as timed (cycle %.1f ns, "
                   "held %.1f ns by the worker's clock; %llu samples)\n", rep, obs_free_frac, expected, timed_free,
                   cycle_ns, held_timed_ns, r.obs_samples);
    } // loop over repetitions
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: run failed: %s\n", argv0, e.what());
    return 1;
  }
  if (std::ferror(stdout) || std::fflush(stdout) != 0) {
    std::fprintf(stderr, "%s: writing the results failed\n", argv0);
    return 1;
  }
  return 0;
} // run_validation()

// Print the usage message to `f`.
static void usage(std::FILE* f, const char* argv0) {
  std::fprintf(f,
               "usage: %s [--window=SECONDS] [--warmup=SECONDS] [--reps=N] [--threads=T1,T2,...]\n"
               "       %*s [--filter=REGEX] [--observer] [--observer-period=US]\n"
               "       %*s [--validate-observer=HELD_NS,FREE_NS]\n"
               "  --window             measured seconds per run (default %g)\n"
               "  --warmup             uncounted seconds before each window (default %g)\n"
               "  --reps               repetitions of the whole sweep (default 1)\n"
               "  --threads            thread counts (default: powers of two up to the CPU count, and the CPU\n"
               "                       count)\n"
               "  --filter             run only the variants whose name contains a match (ECMAScript regex)\n"
               "  --observer           sample the lock word from one more thread and report how long per\n"
               "                       operation the lock is visibly free (skips thread counts that leave it no\n"
               "                       CPU; exits with 2 if that leaves none)\n"
               "  --observer-period    the observer's mean wait between samples, in microseconds, jittered\n"
               "                       by +-50%% (default 1); ignored without --observer or\n"
               "                       --validate-observer\n"
               "  --validate-observer  instead of the variants, run one worker holding the lock for HELD_NS and\n"
               "                       leaving it free for FREE_NS, with the observer, and report the observed\n"
               "                       free fraction against the programmed one (ignores --threads, --filter\n"
               "                       and --observer)\n",
               argv0, static_cast<int>(std::string_view(argv0).size()), "",
               static_cast<int>(std::string_view(argv0).size()), "", kWindowSeconds, kWarmupSeconds);
} // usage()

// Parse the command line, run the sweep (see ORDER above), write the CSV to
// stdout and the summary to stderr; or, with --validate-observer, run the
// observer's validation instead (run_validation()). Returns 0 on success, 1 if
// the CPU count is unavailable or a run or the output failed, 2 for a bad
// command line or when --observer or --validate-observer has no CPU to run on.
int main(int argc, char** argv) {
  double window_s = kWindowSeconds;     // --window
  double warmup_s = kWarmupSeconds;     // --warmup
  int reps = 1;                         // --reps
  std::vector<int> threads_list;        // --threads; empty until given or defaulted
  const char* filter_text = "";         // --filter; the empty regex matches every name
  bool observer = false;                // --observer
  std::uint64_t observer_period_ns = 1000;      // --observer-period, in ns
  bool validate = false;                // whether --validate-observer was given
  int validate_held_ns = 0;             // --validate-observer, the held phase
  int validate_free_ns = 0;             // --validate-observer, the free phase

  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    const char* value = nullptr;
    bool ok = true;
    if (std::string_view(arg) == "--help" || std::string_view(arg) == "-h") {
      usage(stdout, argv[0]);
      return 0;
    } else if ((value = option_value(arg, "--window"))) {
      ok = parse_double(value, window_s) && window_s > 0;
    } else if ((value = option_value(arg, "--warmup"))) {
      ok = parse_double(value, warmup_s) && warmup_s >= 0;
    } else if ((value = option_value(arg, "--reps"))) {
      ok = parse_int(value, reps) && reps >= 1;
    } else if ((value = option_value(arg, "--threads"))) {
      ok = parse_thread_list(value, threads_list);
    } else if ((value = option_value(arg, "--filter"))) {
      filter_text = value;
    } else if (std::string_view(arg) == "--observer") {
      observer = true;
    } else if ((value = option_value(arg, "--observer-period"))) {
      ok = parse_period(value, observer_period_ns);
    } else if ((value = option_value(arg, "--validate-observer"))) {
      ok = parse_validation(value, validate_held_ns, validate_free_ns);
      validate = ok;
    } else {
      ok = false;
    }
    if (!ok) {
      std::fprintf(stderr, "%s: bad argument '%s'\n", argv[0], arg);
      usage(stderr, argv[0]);
      return 2;
    }
  } // loop over the arguments

  const int numcpu = configured_cpu_count();
  if (numcpu == 0) return 1;
  if (validate) {
    // The programmed worker and the observer each need a CPU.
    if (numcpu < 2) {
      std::fprintf(stderr, "%s: --validate-observer needs 2 CPUs, have %d\n", argv[0], numcpu);
      return 2;
    }
    return run_validation(argv[0], validate_held_ns, validate_free_ns, observer_period_ns, warmup_s, window_s, reps);
  } // if validating the observer
  if (threads_list.empty()) threads_list = thread_counts(numcpu);
  // The observer needs a CPU of its own: drop the thread counts that leave it
  // none (see THE OBSERVER above).
  if (observer) {
    std::vector<int> runnable;
    for (int t : threads_list) {
      if (t < numcpu) {
        runnable.push_back(t);
      } else {
        std::fprintf(stderr, "# --observer: skipping threads=%d, which with the observer needs %lld CPUs of %d\n",
                     t, static_cast<long long>(t) + 1, numcpu);
      }
    } // loop over the thread counts
    if (runnable.empty()) {
      std::fprintf(stderr, "%s: --observer leaves no thread count to run on %d CPUs\n", argv[0], numcpu);
      return 2;
    }
    threads_list = std::move(runnable);
  } // if observing

  // The selected variants, in the order of for_each_variant().
  std::vector<Variant> variants;
  try {
    const std::regex filter(filter_text, std::regex::ECMAScript);
    for_each_variant([&]<typename Slot, Store store, PtrLine ptr_line>(const char* name) {
      if (std::regex_search(name, filter)) variants.push_back(Variant{name, run_variant<Slot, store, ptr_line>});
    });
  } catch (const std::regex_error& e) {
    std::fprintf(stderr, "%s: bad --filter regex '%s': %s\n", argv[0], filter_text, e.what());
    return 2;
  }
  if (variants.empty()) {
    std::fprintf(stderr, "%s: --filter '%s' matches no variant\n", argv[0], filter_text);
    return 2;
  }

  std::fprintf(stderr, "# cpus=%d window=%gs warmup=%gs reps=%d variants=%zu threads=", numcpu, window_s, warmup_s,
               reps, variants.size());
  for (size_t i = 0; i != threads_list.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", threads_list[i]);
  if (observer) std::fprintf(stderr, " observer period=%gus", static_cast<double>(observer_period_ns)/1e3);
  std::fprintf(stderr, "\n");

  // cells[vi*threads_list.size() + ti] collects the runs of variant vi at
  // thread count threads_list[ti].
  std::vector<CellSamples> cells(variants.size()*threads_list.size());
  std::printf("variant,threads,rep,window_s,items,items_per_s,cv,min_rel,max_rel,handoffs_per_op,"
              "obs_samples,obs_free_frac,free_ns_per_op\n");
  std::fflush(stdout);
  try {
    for (int rep = 0; rep != reps; ++rep) {
      for (size_t ti = 0; ti != threads_list.size(); ++ti) {
        const int threads = threads_list[ti];
        for (size_t k = 0; k != variants.size(); ++k) {
          // Reversed order when rep + ti is odd; see ORDER above.
          const size_t vi = ((static_cast<size_t>(rep) + ti) % 2 == 0) ? k : variants.size() - 1 - k;
          const RunResult r = variants[vi].run(threads, warmup_s, window_s, observer ? observer_period_ns : 0);
          unsigned long long items = 0;
          for (unsigned long c : r.counts) items += c;
          const double rate = static_cast<double>(items)/r.window_s;
          const Fairness f = fairness(r.counts);
          const double handoffs_per_op = items > 0 ? static_cast<double>(r.handoffs)/static_cast<double>(items) : 0.0;
          std::printf("%s,%d,%d,%.6f,%llu,%.0f,%.4f,%.4f,%.4f,%.6f,", variants[vi].name, threads, rep, r.window_s,
                      items, rate, f.cv, f.min_rel, f.max_rel, handoffs_per_op);
          // The observer's columns: empty without it. The observer always
          // takes at least one sample, and every worker at least one
          // operation, so neither division is by 0; they are guarded anyway.
          double free_frac = 0;
          double free_ns_per_op = 0;
          if (observer) {
            free_frac = r.obs_samples > 0 ? static_cast<double>(r.obs_free)/static_cast<double>(r.obs_samples) : 0.0;
            free_ns_per_op = items > 0 ? free_frac*r.window_s*1e9/static_cast<double>(items) : 0.0;
            std::printf("%llu,%.6f,%.4f\n", r.obs_samples, free_frac, free_ns_per_op);
          } else {
            std::printf(",,\n");
          }
          std::fflush(stdout);
          CellSamples& cell = cells[vi*threads_list.size() + ti];
          if (observer) {
            cell.obs_samples.push_back(static_cast<double>(r.obs_samples));
            cell.obs_free_frac.push_back(free_frac);
            cell.free_ns_per_op.push_back(free_ns_per_op);
          }
          cell.window_s.push_back(r.window_s);
          cell.items.push_back(static_cast<double>(items));
          cell.items_per_s.push_back(rate);
          cell.cv.push_back(f.cv);
          cell.min_rel.push_back(f.min_rel);
          cell.max_rel.push_back(f.max_rel);
          cell.handoffs_per_op.push_back(handoffs_per_op);
        } // loop over variants
      } // loop over thread counts
    } // loop over repetitions
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: run failed: %s\n", argv[0], e.what());
    return 1;
  }

  // The summary: medians over the repetitions, cell by cell, in run order.
  std::fprintf(stderr, "# median over %d repetition%s\n", reps, reps == 1 ? "" : "s");
  std::fprintf(stderr, "%-40s %7s %9s %12s %9s %7s %7s %7s %9s", "variant", "threads", "window_s", "items",
               "M items/s", "cv", "min_rel", "max_rel", "handoff/op");
  if (observer) std::fprintf(stderr, " %11s %9s %10s", "obs_samples", "free_frac", "free_ns/op");
  std::fprintf(stderr, "\n");
  for (size_t ti = 0; ti != threads_list.size(); ++ti) {
    for (size_t vi = 0; vi != variants.size(); ++vi) {
      CellSamples& cell = cells[vi*threads_list.size() + ti];
      std::fprintf(stderr, "%-40s %7d %9.3f %12.0f %9.2f %7.4f %7.4f %7.4f %9.5f", variants[vi].name,
                   threads_list[ti], median(cell.window_s), median(cell.items), median(cell.items_per_s)/1e6,
                   median(cell.cv), median(cell.min_rel), median(cell.max_rel), median(cell.handoffs_per_op));
      if (observer) {
        std::fprintf(stderr, " %11.0f %9.5f %10.3f", median(cell.obs_samples), median(cell.obs_free_frac),
                     median(cell.free_ns_per_op));
      }
      std::fprintf(stderr, "\n");
    } // loop over variants
  } // loop over thread counts

  // A failed write to stdout (a full disk, a closed pipe) would otherwise go
  // unnoticed and leave a truncated CSV behind a zero exit status.
  if (std::ferror(stdout) || std::fflush(stdout) != 0) {
    std::fprintf(stderr, "%s: writing the results failed\n", argv[0]);
    return 1;
  }
  return 0;
} // main()
