// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/ConcurrentCpp
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Wall-clock accounting for threaded Google Benchmark throughput benchmarks.
//
// WHY
//
// Google Benchmark's items_per_second is the total item count divided by the
// MEAN of the threads' loop times: every thread of a run executes the same
// number of iterations, the runner sums the threads' timers and divides the sum
// by the thread count, and every rate counter (the one SetItemsProcessed() sets
// among them) is divided by that mean. The Time column is the same mean, per
// iteration. Under a mechanism that favours some threads -- a lock that lets a
// thread run a streak, a wait that one thread loses more often than another --
// the favoured threads finish early and wait at the end-of-loop barrier, so the
// mean loop time is shorter than the run and items_per_second overstates the
// throughput: the more, the less fair the mechanism, and fairness is often
// exactly what the compared variants differ in.
//
// A clock read after the loop cannot repair this. The loop's start and end are
// barriers across all threads of a run (State::StartKeepRunning() and
// State::FinishKeepRunning(), under Google Benchmark's default thread runner),
// so a thread passes the end only after every other thread's last iteration,
// and the threads leave the barrier at times that have nothing to do with when
// each finished: a read after the loop measures nothing per thread. The stamps
// have to be taken inside the loop, by the thread itself, at the end of its
// first and of its last iteration. That is what this header does. A benchmark
// body that uses it reports, next to items_per_second (which this header does
// not alter, so that the row stays comparable with every result that has it):
//   wall_items_per_second -- total items (the body's SetItemsProcessed()
//                            argument times the thread count) divided by the
//                            wall-clock SPAN, from the end of the earliest
//                            first iteration of any thread to the end of the
//                            last thread's last iteration;
//   finish_spread         -- (last finish - first finish)/(that span), between
//                            0 and 1: 0 when all threads finish together, near
//                            1 when some thread finished right at the start;
//   start_spread          -- (latest first - earliest first)/(that span): the
//                            share of the span that passed before every thread
//                            had finished its first iteration, 0 when they all
//                            started together.
// Every thread stamps the end of its own first iteration, and the span starts
// at the earliest of those stamps: a thread can start late (one that misses a
// lock's first burst can sleep through other threads' whole streaks), so any
// one thread's stamp, thread 0's included, could be late. Since every finish
// is at or after its own thread's first stamp, no finish precedes the span
// start, which is what keeps finish_spread within [0, 1]. The item count
// includes at most one iteration per thread from before the span starts,
// which is negligible. A large finish_spread means that the tail of the run
// had fewer threads contending than its start, so even wall_items_per_second
// mixes contention levels; only a harness that runs every thread until a
// common stop avoids that, and this header does not replace one.
//
// The counters are plain values, not rates: Google Benchmark would divide a
// rate by the mean loop time again. Thread 0 alone sets them, after its loop,
// so that summing the counters over the threads, which the framework does,
// leaves them unchanged; every other thread's record is complete by then (see
// ORDERING).
//
// WHAT IS HERE
//
//   WallStamps          one thread's two stamps, `first` and `last`, the
//                       CPU it ran on at each (taken outside the iterations:
//                       after the start barrier, after the last iteration),
//                       its kernel thread id, and, with GB_WALL_PERF set, its
//                       own hardware counters over the loop; a record with
//                       either stamp unset means "this thread did not run".
//   WallRecords         the per-run store of one record per thread, shared by
//                       all threads of a run through a static instance (one
//                       instance may serve several rows that never run at
//                       the same time);
//                       thread 0 allocates it before the loop and releases it
//                       after the report.
//   WallTimed           the loop range: `for (auto _ : WallTimed(state,
//                       records))` in place of `for (auto _ : state)`; runs the
//                       same iterations between the same barriers, and stamps
//                       the end of the thread's first and last iteration.
//   WallSpan            the aggregate of a run's records: the span and the
//                       two spreads, with the stamps they are made of, the
//                       thread whose own loop time is the longest and its
//                       share of the span, and how many threads ran.
//   wall_span()         the aggregation, over a WallRecords or over an array
//                       of any record type derived from WallStamps.
//   set_wall_counters() the three counters from a WallSpan.
//   wall_thread_detail()  whether the environment variable GB_WALL_THREADS is
//                       set: the per-thread detail below is wanted.
//   wall_perf_wanted()  whether the environment variable GB_WALL_PERF is set:
//                       every thread counts its own cycles, reference cycles,
//                       instructions and scheduled time over its loop
//                       (WallPerf, Linux perf_event_open on the thread itself,
//                       user mode only) into its record. Independent of
//                       GB_WALL_THREADS: either variable alone gives its line.
//   WallPerf            the counter group: open before the start barrier, read
//                       after the last iteration, both on the cold path.
//   report_wall_threads() the per-thread detail of a run from its WallRecords:
//                       the counters slowest_thread and slowest_share, and one
//                       `# threads` console line with every thread's first and
//                       last stamp as offsets from the span's start, its CPUs
//                       and its thread id.
//   report_wall_perf()  the counter perf_threads and one `# perf` console
//                       line with every thread's counters (GB_WALL_PERF).
//   report_wall()       the one-line aggregation and report for plain bodies;
//                       with GB_WALL_THREADS set it adds the per-thread detail,
//                       with GB_WALL_PERF set the counters.
// A benchmark whose timed loop carries more per-thread state than the stamps
// keeps its own loop and its own record type, derived from WallStamps and
// aggregated by wall_span(const Rec*, int) like any other; set_wall_counters()
// gives it the three counters, and the per-thread detail is report_wall()'s
// alone (it reads a WallRecords).
//
// USE
//
// A plain benchmark function:
//   static void BM_foo(benchmark::State& state) {
//     static constinit WallRecords records;
//     records.allocate(state);
//     ... setup ...
//     for (auto _ : WallTimed(state, records)) {
//       ... one iteration ...
//     }
//     state.SetItemsProcessed(state.iterations());
//     report_wall(state, records, static_cast<double>(state.iterations())*state.threads());
//     records.release(state);
//   }
// allocate() and release() act on thread 0 and do nothing on the others, so
// every thread runs the same lines. A fixture keeps a static WallRecords
// (one may serve every row of the fixture, since the rows run one at a time):
// SetUp() calls records.allocate(state), TearDown() calls
// records.release(state) once nothing touches the records any more (after
// any post-loop drain the fixture runs), and the body is as above. `items`
// is the body's business: threads() times the argument it passes to
// SetItemsProcessed().
//
// The records add NO local with a destructor across the loop -- which is why
// the release is a call and not the end of an owner's scope: a cleanup region
// spanning the measured loop (any local with a non-trivial destructor) can
// make the compiler lay the loop out differently from the plain one, moving
// work into a guarded section or rematerializing loop invariants. The rule is
// about not adding such a local: a body's own long-lived local (a cursor that
// must survive from one iteration to the next) is not one the records add,
// and it stays as the body needs it.
//
// PLACEMENT CONTRACT (every rule is load-bearing; breaking one is a data race,
// a crash, or a row whose rates are wrong):
//   - allocate() runs before the loop, in the body or in SetUp(); release()
//     runs after report_wall() (or the body's own aggregation), in the body or
//     in TearDown().
//   - No thread other than 0 touches the records in SetUp() or TearDown():
//     nothing orders one thread's SetUp() against another's, and TearDown()
//     of a non-zero thread may run while thread 0 is still aggregating. Bodies
//     touch the records only inside the loop (through WallTimed) and, thread 0,
//     after it; a body never hoists `records.of(tid)` into a local before the
//     loop, since the record is valid only after the start barrier.
//   - The counters are set in the body, next to SetItemsProcessed(), before
//     the fixture's thread-0 TearDown() releases the records.
//   - No body returns before the loop: that thread leaves its record
//     unstamped and the framework marks the row as an error.
//   - No body leaves the loop early (break or return): the one batch call at
//     begin() has already counted every iteration, so unlike the framework's
//     own loop an early exit is not reported as an error; that thread never
//     reaches the end barrier, so its time and its `last` stamp are lost while
//     its iterations still count, and both rates come out too high.
//   - SkipWithError() is called after the loop only. Before the loop the
//     range copes with it (both barriers run inside the first batch call;
//     see THE LOOP), but the row is skipped; inside the loop the countdown
//     does not see it and keeps running to the end, and leaving the loop
//     early is excluded above.
//   - state.iterations() is read before or after the loop only: inside it, the
//     batch form of the loop makes it read max_iterations.
//
// ORDERING (Google Benchmark's default thread runner, or any ThreadRunnerBase
// whose RunThreads() returns only after every thread of the run has finished)
//
// StartKeepRunning() runs the start barrier, FinishKeepRunning() the end
// barrier; both are a mutex and a condition variable, so everything a thread
// did before entering one happens-before everything any thread does after
// leaving it. Thread 0's allocation before its loop happens-before every other
// thread's first stamp: WallTimed resolves the thread's record only after the
// batch call that ran the start barrier. Every thread's last stamp is taken
// before the batch call that runs the end barrier, so it happens-before thread
// 0's post-loop aggregation. A run's release is ordered before the next run by
// the runner's thread join and creation. The records are plain, non-atomic
// data; nothing else is needed.
//
// THE LOOP
//
// WallTimed replaces Google Benchmark's range-for with one batch call for all
// max_iterations iterations at begin() (StartKeepRunning() and the start
// barrier run inside it) and one more at exhaustion (it returns false and runs
// FinishKeepRunning() and the end barrier); state.iterations() equals
// max_iterations after the loop, as under the range-for. Per iteration the
// loop does, at source level, the same operations as the framework's own
// range-for: one decrement of a countdown and one branch on it. The countdown
// starts at 1, so that the end of the first iteration lands on its zero
// crossing, where an out-of-line slow path stamps `first` and reloads it with
// the remaining count; the second zero crossing stamps `last` and makes the
// finishing batch call. There is no second counter, no compare against a
// stored bound, and no store other than the countdown's and no call on the
// per-iteration path: the stamps live entirely on the cold path. A skipped state
// (SkipWithError() before the loop) makes the first batch call return false
// after running both barriers; the range records that and makes no second
// batch call. When max_iterations is 0 -- a run the framework can request
// with --benchmark_min_time=0x, and whose own batch call would never reach the
// barriers -- the range delegates to the framework's own range-for, which runs
// both barriers and no iteration, and takes no stamp: the row reports zero
// counters.
//
// MEMORY
//
// Each record is its own 128-byte block (two adjacent cache lines, the unit
// the prefetcher pairs), so that two threads' first and last stamps never
// share one. The over-aligned records are allocated with new[], which honours
// the alignment (aligned operator new[]), and released with delete[].
#ifndef INCLUDED_GB_WALL_CLOCK_H
#define INCLUDED_GB_WALL_CLOCK_H
#include <algorithm>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sched.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

#include "benchmark/benchmark.h"

// The branch hint of the loop's hot test: the countdown is non-zero on every
// iteration but two. __builtin_expect states the edge's weight, which the
// standard attribute leaves to the compiler; a compiler without the builtin
// drops the hint and the code is still correct.
#if defined(__GNUC__) || defined(__clang__)
#define GB_WALL_LIKELY(c) __builtin_expect(!!(c), 1)
#else
#define GB_WALL_LIKELY(c) (c)
#endif

// One thread's record of one run: the end of its first and of its last
// iteration, steady-clock, and the CPU the thread was on at each: start_cpu
// after the start barrier and before its first iteration, end_cpu after its
// last iteration and before the end barrier (sched_getcpu(), -1 when not
// taken); its kernel thread id, taken with start_cpu (the id an external tool
// such as perf names the thread by; -1 off Linux); and, when GB_WALL_PERF is
// set, the thread's own hardware counters over its loop (WallPerf below:
// counted from just before the start barrier to just after the last
// iteration, user mode only; -1 for a counter that was not counted). Unstamped (default-constructed, the clock's
// zero) means the thread did not reach that point: a record with either stamp
// unset is ignored by wall_span(). No alignment of its own: the record type
// that embeds or derives from it carries the 128-byte block (WallRecords::Rec,
// or a benchmark's own record with more fields), so that a record stays one
// block whatever it adds.
struct WallStamps {
  std::chrono::steady_clock::time_point first;  // end of this thread's first iteration
  std::chrono::steady_clock::time_point last;   // end of this thread's last iteration
  int start_cpu = -1;                   // the CPU after the start barrier, before the first iteration
  int end_cpu = -1;                     // the CPU after the last iteration, before the end barrier
  int tid = -1;                         // the kernel thread id (gettid), taken with start_cpu
  long long cycles = -1;                // core cycles the thread executed in user mode over its loop
  long long ref_cycles = -1;            // reference (TSC-rate) cycles over the same time: cycles/ref_cycles is the core's clock relative to the TSC
  long long instructions = -1;          // instructions retired in user mode
  long long task_ns = -1;               // ns the thread was scheduled on a CPU (task-clock), user and kernel
  long long enabled_ns = -1;            // ns the group was enabled while the thread was on a CPU (a per-task event's clock runs only then: about task_ns, not the wall interval)
  long long running_ns = -1;            // ns of those the group was on the PMU: below enabled_ns the group was multiplexed and the counts are partial
}; // struct WallStamps

// Whether every thread's own hardware counters over its loop are wanted: the
// environment variable GB_WALL_PERF set (to anything). Read once per process.
// Independent of GB_WALL_THREADS: report_wall() prints the `# perf` line and
// sets perf_threads whenever this is set, with or without the `# threads`
// line.
inline bool wall_perf_wanted() {
  static const bool wanted = std::getenv("GB_WALL_PERF") != nullptr;
  return wanted;
} // wall_perf_wanted()

// A thread's own counter group (Linux perf_event_open on the calling thread,
// no inheritance, every event user-only: at perf_event_paranoid 2 the kernel
// refuses an event that includes kernel mode, software events included, and
// the task clock counts the task's whole scheduled time whatever its exclusion
// bits say): core cycles as the leader, then reference cycles, instructions
// and the task clock (the thread's scheduled time, user and kernel). A member
// the PMU lacks (reference cycles on an AMD core) is left out: its counter
// stays -1 and the group counts without it; only a refused leader leaves every
// counter -1. Either case is reported once on stderr. The group is read in
// one read() with the enabled and running times, so that a multiplexed
// (partially scheduled) group is visible as running_ns < enabled_ns. Exists
// so that the counters are the thread's own over exactly its loop: an
// external `perf stat --per-thread -p <pid>` folds the counts of threads
// created after it attached into the parent thread's, and Google Benchmark
// creates a run's helper threads per repetition. Opened by WallTimed before
// the start barrier (the open and the enable are its only cost there) and
// read right after the thread's last iteration, before the end barrier; the
// read is one system call on the cold path, after the `last` stamp, so
// neither the span nor the per-thread times include it. Not available off
// Linux: open() returns false and every counter stays -1.
struct WallPerf {
  static constexpr int kEvents = 4;     // the leader and three members, in the order of WallStamps' counters
  int fd[kEvents] = {-1, -1, -1, -1};   // per event: its descriptor while it counts, -1 when absent (each stays open while it counts: closing a member removes it from the group)

  // Whether the group is open and counting on this thread.
  bool active() const noexcept { return fd[0] >= 0; }
  // Open the group on the calling thread, enabled. False, with every
  // descriptor -1, when the kernel refused the leader (reported once on
  // stderr with errno); a refused member is reported once and left absent.
  bool open() noexcept;
  // Read the group into `rec` (the counters of the open events; the absent
  // ones stay -1) and close it. False (every counter left -1) when the read
  // did not return the open events' values; closed either way.
  bool read_and_close(WallStamps& rec) noexcept;
  // Close whatever is open; no-op when nothing is.
  void close_all() noexcept {
    for (int& f : fd) {
      if (f >= 0) close(f);
      f = -1;
    } // loop over the descriptors
  } // WallPerf::close_all()
}; // struct WallPerf

#ifdef __linux__
inline bool WallPerf::open() noexcept {
  // One attribute per event; the leader is disabled until the whole group is
  // open and then enabled with its group, so that the four start together.
  const auto attr_for = [](unsigned int type, unsigned long long config, bool leader) {
    perf_event_attr a{};
    a.size = sizeof a;
    a.type = type;
    a.config = config;
    a.disabled = leader ? 1 : 0;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.inherit = 0;
    a.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return a;
  }; // attr_for()
  const auto open_one = [](perf_event_attr& a, int group) {
    return static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, group, 0UL));   // pid 0 = the calling thread, any CPU
  }; // open_one()
  perf_event_attr leader = attr_for(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, true);
  fd[0] = open_one(leader, -1);
  if (fd[0] < 0) {
    [[maybe_unused]] static const bool reported =
        (std::fprintf(stderr, "GB_WALL_PERF: perf_event_open(cycles, the group leader) failed: %s; no per-thread counters\n", std::strerror(errno)), true);
    fd[0] = -1;
    return false;
  } // if the leader failed
  struct Member { const char* name; unsigned int type; unsigned long long config; };
  const Member members[kEvents - 1] = {
    {"ref-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
  };
  for (int i = 0; i != kEvents - 1; ++i) {
    perf_event_attr a = attr_for(members[i].type, members[i].config, false);
    fd[i + 1] = open_one(a, fd[0]);
    if (fd[i + 1] < 0) {
      [[maybe_unused]] static const bool reported =
          (std::fprintf(stderr, "GB_WALL_PERF: perf_event_open(%s) failed: %s; that counter stays absent (-1)\n", members[i].name, std::strerror(errno)), true);
      fd[i + 1] = -1;
    } // if a member failed: absent, the group goes on without it
  } // loop over the group's members
  ioctl(fd[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(fd[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  return true;
} // WallPerf::open()

inline bool WallPerf::read_and_close(WallStamps& rec) noexcept {
  // PERF_FORMAT_GROUP with both times: nr, time_enabled, time_running, then
  // one value per OPEN event in the order they were opened, which is the
  // order of fd[]; an absent event has no value and keeps -1.
  unsigned long long buf[3 + kEvents] = {};
  int open_count = 0;
  for (const int f : fd) open_count += (f >= 0);
  const ssize_t want = static_cast<ssize_t>((3 + open_count)*sizeof buf[0]);
  const ssize_t got = read(fd[0], buf, sizeof buf);
  const bool whole = got == want && buf[0] == static_cast<unsigned long long>(open_count);
  if (whole) {
    rec.enabled_ns = static_cast<long long>(buf[1]);
    rec.running_ns = static_cast<long long>(buf[2]);
    long long* const counters[kEvents] = {&rec.cycles, &rec.ref_cycles, &rec.instructions, &rec.task_ns};
    int next = 3;
    for (int i = 0; i != kEvents; ++i) {
      if (fd[i] >= 0) *counters[i] = static_cast<long long>(buf[next++]);
    } // loop over the events, taking a value for each open one
  } // if the read returned the open events' values
  close_all();
  return whole;
} // WallPerf::read_and_close()
#else
inline bool WallPerf::open() noexcept { return false; }
inline bool WallPerf::read_and_close(WallStamps&) noexcept { return false; }
#endif

// The per-run store of one WallStamps per thread of a run. One instance per
// set of runs that never overlap (a function-local `static constinit
// WallRecords` of one body, or a fixture's static that serves several rows of
// that fixture, which run one at a time), constant-initialized so that it has
// no run-time guard and is addressable before any thread runs. It owns the
// array between allocate() and release(), both thread-0 operations (no-ops
// elsewhere), and holds nothing otherwise; every thread of a run addresses
// the same static instance. Not copyable: a copy would be a second owner of
// the same records.
//
// Exception guarantee: allocate() may throw std::bad_alloc, before it changes
// the instance (strong); everything else is nothrow.
class WallRecords {
  public:
  // One record, alone in its 128-byte block.
  struct alignas(128) Rec : WallStamps {};
  static_assert(sizeof(Rec) == 128);

  constexpr WallRecords() noexcept = default;
  WallRecords(const WallRecords&) = delete;
  WallRecords& operator=(const WallRecords&) = delete;

  // Allocate this run's records, on thread 0; a no-op on every other thread.
  //   state -- the calling thread's state: thread_index() selects the thread,
  //            threads() the number of records.
  // On thread 0: state.threads() value-initialized (unstamped) records, which
  // the instance holds until release(). A run that ended without release()
  // (an exception between the two) leaves its array for the next allocate()
  // to free first. Call it before the loop or in thread 0's SetUp(). Strong
  // exception guarantee: std::bad_alloc leaves the instance unchanged.
  void allocate(const benchmark::State& state);

  // Free this run's records, on thread 0; a no-op on every
  // other thread. Call it after report_wall() (or the body's own aggregation),
  // or in thread 0's TearDown() once nothing touches the records any more.
  void release(const benchmark::State& state) noexcept;

  // This thread's record (or, for thread 0 after the loop, any thread's).
  // Precondition: allocated, 0 <= tid < threads(), and the caller is past the
  // start barrier of the run that allocated (inside or after the loop).
  WallStamps& of(int tid) noexcept {
    assert(recs_ != nullptr && 0 <= tid && tid < threads_);
    return recs_[tid];
  } // WallRecords::of()
  const WallStamps& of(int tid) const noexcept {
    assert(recs_ != nullptr && 0 <= tid && tid < threads_);
    return recs_[tid];
  } // WallRecords::of() const

  // The thread count allocate() used; 0 between release() and allocate().
  int threads() const noexcept { return threads_; }

  private:
  Rec* recs_ = nullptr;                 // this run's records, one per thread, owned; null between runs
  int threads_ = 0;                     // how many; 0 between runs
}; // class WallRecords

inline void WallRecords::allocate(const benchmark::State& state) {
  if (state.thread_index() != 0) return;
  const int n = state.threads();
  // Value-initialized: every record starts unstamped. new[] of the over-aligned
  // Rec is the aligned form; release()'s delete[] matches it.
  Rec* const p = new Rec[static_cast<size_t>(n)]();
  delete[] recs_;                       // a stale array, if a run ended without release()
  recs_ = p;
  threads_ = n;
} // WallRecords::allocate()

inline void WallRecords::release(const benchmark::State& state) noexcept {
  if (state.thread_index() != 0) return;
  delete[] recs_;
  recs_ = nullptr;
  threads_ = 0;
} // WallRecords::release()

// The loop range of a wall-timed benchmark body, in place of the state itself:
//   for (auto _ : WallTimed(state, records)) { ... }
// See THE LOOP above for what it does per iteration and at the ends, and the
// PLACEMENT CONTRACT for what the body around it must and must not do. One
// object runs one loop: it is constructed in the range-for head, holds the
// state and the records by reference, and is neither copied nor moved.
class WallTimed {
  public:
  // Preconditions: `records` is the body's records (allocated by thread 0 before
  // this point); the loop that follows is the body's only loop over `state`.
  WallTimed(benchmark::State& state, WallRecords& records) noexcept
    : state_(state), records_(records) {}
  WallTimed(const WallTimed&) = delete;
  WallTimed& operator=(const WallTimed&) = delete;

  // The range-for's end marker; carries nothing.
  struct Sentinel {};

  // The range-for's iterator: the countdown and the range it belongs to. Its
  // value type is the framework's own empty one, so that the `_` of the loop
  // head is as unused there as under `for (auto _ : state)`.
  class Iterator {
    public:
    using Value = benchmark::State::StateIterator::Value;
    Value operator*() const noexcept { return Value(); }
    [[gnu::always_inline]] Iterator& operator++() noexcept {
      --left_;
      return *this;
    }
    // True while iterations remain. At a zero crossing the range's slow path
    // stamps and either reloads the countdown (after the first iteration) or
    // finishes the run (after the last) and returns 0.
    [[gnu::always_inline]] bool operator!=(Sentinel) {
      if (GB_WALL_LIKELY(left_ != 0)) return true;
      left_ = range_->next_phase();
      return left_ != 0;
    } // WallTimed::Iterator::operator!=()

    private:
    friend class WallTimed;
    Iterator(WallTimed* range, benchmark::IterationCount left) noexcept : range_(range), left_(left) {}
    WallTimed* range_;                  // the range, for the slow path
    benchmark::IterationCount left_;    // iterations until the next zero crossing
  }; // class WallTimed::Iterator

  // Start the run: the first batch call (start barrier inside), then this
  // thread's record. Returns the iterator at countdown 1, or an exhausted one
  // when the state is skipped or max_iterations is 0 (see THE LOOP).
  Iterator begin();
  Sentinel end() const noexcept { return Sentinel(); }

  private:
  // The slow path of a zero crossing. Returns the iterations still to run
  // before the next one: max_iterations - 1 after the first iteration, 0 after
  // the last (the end barrier has run by then) and 0 again when there was
  // nothing to run. noinline: the loop's only out-of-line call, so that the
  // stamps and the finishing batch call stay off the hot path. Defined here
  // (implicitly inline): with the attribute on this declaration, g++ warns
  // under -Wattributes ("inline declaration ... follows declaration with
  // attribute 'noinline'") on an out-of-class `inline` definition, an error
  // under -Werror.
  [[gnu::noinline]] benchmark::IterationCount next_phase() {
    if (finished_) return 0;
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    if (!first_stamped_) {
      mine_->first = now;                 // end of the first iteration
      first_stamped_ = true;
      const benchmark::IterationCount rest = state_.max_iterations - 1;
      if (rest != 0) return rest;
      // A one-iteration run: the first iteration is also the last; fall through.
    } // if the first zero crossing
    mine_->last = now;                    // end of the last iteration, before the end barrier
    mine_->end_cpu = sched_getcpu();      // where this thread ended its loop (after the last iteration, not in it)
    if (perf_.active()) perf_.read_and_close(*mine_);   // the thread's counters over its loop, after the `last` stamp
    finished_ = true;
    // The finishing batch call: nothing is left, so it returns false after
    // FinishKeepRunning(), which stops the timer and runs the end barrier.
    const bool more = state_.KeepRunningBatch(state_.max_iterations);
    assert(!more);
    (void)more;
    return 0;
  } // WallTimed::next_phase()

  benchmark::State& state_;             // this thread's benchmark state
  WallRecords& records_;                // the body's shared records
  WallStamps* mine_ = nullptr;          // this thread's record; resolved after the start barrier
  bool first_stamped_ = false;          // `first` is written: the next zero crossing is the last
  bool finished_ = false;               // the finishing batch call was made, or there was nothing to run
  WallPerf perf_;                       // this thread's counter group while it counts (GB_WALL_PERF); closed otherwise
}; // class WallTimed

inline WallTimed::Iterator WallTimed::begin() {
  const benchmark::IterationCount n = state_.max_iterations;
  if (n == 0) {
    // Nothing to run: the framework's own loop runs both barriers and no
    // iteration; a batch call of 0 would run neither and never return false.
    for (auto _ : state_) {}
    finished_ = true;
    return Iterator(this, 0);
  } // if max_iterations == 0
  if (wall_perf_wanted()) perf_.open();   // counting from here: before the start barrier
  if (!state_.KeepRunningBatch(n)) {
    // Skipped: both barriers have run inside this call; no second call.
    perf_.close_all();
    finished_ = true;
    return Iterator(this, 0);
  } // if skipped
  mine_ = &records_.of(state_.thread_index());  // after the start barrier
  mine_->start_cpu = sched_getcpu();            // where this thread begins its loop (a vDSO call on x86-64; before the first iteration, not in it)
#ifdef __linux__
  mine_->tid = static_cast<int>(syscall(SYS_gettid));   // the id perf and /proc name this thread by (the syscall: glibc before 2.30 has no gettid())
#endif
  return Iterator(this, 1);
} // WallTimed::begin()

// The aggregate of a run's records: the stamps the counters are made of, the
// span and the spreads, and which thread took longest. All zero (slowest -1)
// when no record is stamped.
struct WallSpan {
  std::chrono::steady_clock::time_point start {};         // earliest `first` of any thread
  std::chrono::steady_clock::time_point last_start {};    // latest `first`
  std::chrono::steady_clock::time_point first_finish {};  // earliest `last`
  std::chrono::steady_clock::time_point last_finish {};   // latest `last`
  double seconds = 0.0;                 // the span: last_finish - start
  double spread = 0.0;                  // finish_spread: (last_finish - first_finish)/seconds, 0 if seconds is 0
  double start_spread = 0.0;            // (last_start - start)/seconds: the share of the span before every thread was running, 0 if seconds is 0
  int slowest = -1;                     // the thread whose own loop time (last - first) is the longest; -1 if none stamped
  double slowest_share = 0.0;           // that thread's own loop time over the span, 0 if seconds is 0
  int stamped = 0;                      // records with both stamps (the threads that ran)
}; // struct WallSpan

namespace gb_wall_clock_detail {

// The one aggregation, over an indexing function so that both wall_span()
// overloads share it without one indexing the other's storage (a WallStamps*
// over an array of a derived record type would be undefined behaviour).
//   at      -- int -> const WallStamps&, the record of thread i;
//   threads -- how many.
// Records with `first` or `last` unset are skipped: they belong to threads
// that did not run (a zero-iteration run, an early return, a per-thread
// SkipWithError(), a rerun of the row by a memory manager with thread 0 alone
// while threads() still says N). The span and the spread are the ones every wall-timed benchmark in this
// repository reports; the arithmetic is written out exactly so.
template <typename At>
WallSpan wall_span_at(At at, int threads) {
  using TimePoint = std::chrono::steady_clock::time_point;
  using Duration = std::chrono::steady_clock::duration;
  WallSpan s;
  Duration longest {};                  // the longest own loop time seen so far (slowest's)
  for (int i = 0; i != threads; ++i) {
    const WallStamps& rec = at(i);
    if (rec.first == TimePoint() || rec.last == TimePoint()) continue;   // did not run
    if (s.stamped == 0) {
      s.start = rec.first;
      s.last_start = rec.first;
      s.first_finish = rec.last;
      s.last_finish = rec.last;
    } else {
      s.start = std::min(s.start, rec.first);
      s.last_start = std::max(s.last_start, rec.first);
      s.first_finish = std::min(s.first_finish, rec.last);
      s.last_finish = std::max(s.last_finish, rec.last);
    } // if the first stamped record
    const Duration own = rec.last - rec.first;
    if (s.stamped == 0 || own > longest) {
      longest = own;
      s.slowest = i;
    } // if this thread's loop time is the longest so far
    ++s.stamped;
  } // loop over the threads' records
  if (s.stamped == 0) return s;         // all zero
  s.seconds = std::chrono::duration<double>(s.last_finish - s.start).count();
  if (s.seconds > 0) {
    s.spread = std::chrono::duration<double>(s.last_finish - s.first_finish).count()/s.seconds;
    s.start_spread = std::chrono::duration<double>(s.last_start - s.start).count()/s.seconds;
    s.slowest_share = std::chrono::duration<double>(longest).count()/s.seconds;
  } // if the span is positive
  return s;
} // wall_span_at()

} // namespace gb_wall_clock_detail

// The aggregate of `threads` records of a benchmark's own record type, derived
// from WallStamps (WallStamps itself included). For bodies with their own
// timed loop; call it on thread 0 after the loop.
//   recs    -- the records, one per thread;
//   threads -- how many.
template <typename Rec>
  requires std::derived_from<Rec, WallStamps>
WallSpan wall_span(const Rec* recs, int threads) {
  return gb_wall_clock_detail::wall_span_at([recs](int i) -> const WallStamps& { return recs[i]; }, threads);
} // wall_span(const Rec*, int)

// The aggregate of a WallRecords' records. Precondition: allocated for this
// run, and the caller is thread 0 after its loop.
inline WallSpan wall_span(const WallRecords& records) {
  return gb_wall_clock_detail::wall_span_at([&records](int i) -> const WallStamps& { return records.of(i); },
                                            records.threads());
} // wall_span(const WallRecords&)

// Set the three counters of this run from its aggregate. Plain values, not
// rates (see WHY). The caller is thread 0 -- not checked here, so that bodies
// with their own aggregation can call it inside their own thread-0 block.
//   state -- thread 0's state;
//   s     -- the run's aggregate;
//   items -- the run's total items, over all threads.
inline void set_wall_counters(benchmark::State& state, const WallSpan& s, double items) {
  state.counters["wall_items_per_second"] = benchmark::Counter(s.seconds > 0 ? items/s.seconds : 0.0);
  state.counters["finish_spread"] = benchmark::Counter(s.spread);
  state.counters["start_spread"] = benchmark::Counter(s.start_spread);
} // set_wall_counters()

// Whether the per-thread detail of report_wall() is wanted: the environment
// variable GB_WALL_THREADS set (to anything). Read once per process.
inline bool wall_thread_detail() {
  static const bool wanted = std::getenv("GB_WALL_THREADS") != nullptr;
  return wanted;
} // wall_thread_detail()

// The per-thread detail of one run, on thread 0 after the loop, when
// wall_thread_detail() says so: the counters slowest_thread (the index of the
// thread whose own loop time is the longest) and slowest_share (its loop time
// over the span), and one console line
//   # threads <row> threads=<T> span_us=<span> start_spread=<..>
//     finish_spread=<..> slowest=<tid> slowest_share=<..>
//     first_us=<v0,v1,...> last_us=<v0,v1,...>
// with every thread's two stamps as offsets from the span's start in
// microseconds (`-` for a thread that did not run), in thread-index order,
// stamped=<n>, how many threads ran, cpu_first=<..> cpu_last=<..>, the
// CPU each thread ran on after the start barrier and after its last iteration,
// and tid=<..>, each thread's kernel thread id (WallStamps). Exists for
// reading which thread sets a run's span and when each one started: a
// finish_spread alone says that some thread straggled, not which.
//   state   -- thread 0's state;
//   records -- the run's records;
//   s       -- their aggregate (wall_span(records)).
inline void report_wall_threads(benchmark::State& state, const WallRecords& records, const WallSpan& s) {
  state.counters["slowest_thread"] = benchmark::Counter(static_cast<double>(s.slowest));
  state.counters["slowest_share"] = benchmark::Counter(s.slowest_share);
  using TimePoint = std::chrono::steady_clock::time_point;
  const auto offset_us = [&s](TimePoint t) {
    return static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(t - s.start).count());
  }; // offset_us()
  std::string first_us;
  std::string last_us;
  std::string cpu_first;
  std::string cpu_last;
  std::string tid;
  for (int i = 0; i != records.threads(); ++i) {
    const WallStamps& rec = records.of(i);
    const bool ran = rec.first != TimePoint() && rec.last != TimePoint();
    if (i != 0) {
      first_us += ',';
      last_us += ',';
      cpu_first += ',';
      cpu_last += ',';
      tid += ',';
    } // the separator before every entry but the first
    first_us += ran ? std::to_string(offset_us(rec.first)) : "-";
    last_us += ran ? std::to_string(offset_us(rec.last)) : "-";
    cpu_first += ran ? std::to_string(rec.start_cpu) : "-";
    cpu_last += ran ? std::to_string(rec.end_cpu) : "-";
    tid += ran ? std::to_string(rec.tid) : "-";
  } // loop over the threads' records
  std::printf("# threads %s threads=%d stamped=%d span_us=%lld start_spread=%.3f finish_spread=%.3f slowest=%d slowest_share=%.3f first_us=%s last_us=%s cpu_first=%s cpu_last=%s tid=%s\n",
              state.name().c_str(), records.threads(), s.stamped,
              static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(s.last_finish - s.start).count()),
              s.start_spread, s.spread, s.slowest, s.slowest_share, first_us.c_str(), last_us.c_str(), cpu_first.c_str(), cpu_last.c_str(), tid.c_str());
  std::fflush(stdout);
} // report_wall_threads()

// The per-thread counters of one run, on thread 0 after the loop, when
// wall_perf_wanted() says so: the counter perf_threads (how many threads have
// a cycles count) and one console line
//   # perf <row> counted=<n> cycles=<v0,...> ref_cycles=<..> instructions=<..>
//     task_ns=<..> enabled_ns=<..> running_ns=<..>
// with every thread's own counters over its loop (WallStamps, WallPerf), in
// thread-index order, `-` for a counter the thread does not have.
//   state   -- thread 0's state;
//   records -- the run's records.
inline void report_wall_perf(benchmark::State& state, const WallRecords& records) {
  // The six counter lists, in WallStamps order.
  std::string perf[6];
  int counted = 0;
  const auto field = [](long long v) { return v >= 0 ? std::to_string(v) : std::string("-"); };
  for (int i = 0; i != records.threads(); ++i) {
    const WallStamps& rec = records.of(i);
    if (i != 0) {
      for (std::string& p : perf) p += ',';
    } // the separator before every entry but the first
    perf[0] += field(rec.cycles);
    perf[1] += field(rec.ref_cycles);
    perf[2] += field(rec.instructions);
    perf[3] += field(rec.task_ns);
    perf[4] += field(rec.enabled_ns);
    perf[5] += field(rec.running_ns);
    if (rec.cycles >= 0) ++counted;
  } // loop over the threads' records
  state.counters["perf_threads"] = benchmark::Counter(static_cast<double>(counted));
  std::printf("# perf %s counted=%d cycles=%s ref_cycles=%s instructions=%s task_ns=%s enabled_ns=%s running_ns=%s\n",
              state.name().c_str(), counted, perf[0].c_str(), perf[1].c_str(), perf[2].c_str(), perf[3].c_str(), perf[4].c_str(), perf[5].c_str());
  std::fflush(stdout);
} // report_wall_perf()

// The one-line report for plain bodies: on thread 0, aggregate the records
// and set the counters (and the per-thread detail when GB_WALL_THREADS is
// set, the per-thread counters when GB_WALL_PERF is); on every other thread,
// nothing. Does not release the records:
// release() does. Call it after the loop, next to SetItemsProcessed(), on
// every thread.
//   state   -- the calling thread's state;
//   records -- the body's records, allocated for this run;
//   items   -- the run's total items, over all threads.
inline void report_wall(benchmark::State& state, const WallRecords& records, double items) {
  if (state.thread_index() != 0) return;
  const WallSpan s = wall_span(records);
  set_wall_counters(state, s, items);
  if (wall_thread_detail()) report_wall_threads(state, records, s);
  if (wall_perf_wanted()) report_wall_perf(state, records);
} // report_wall()

#endif // INCLUDED_GB_WALL_CLOCK_H
