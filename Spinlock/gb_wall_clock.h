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
//                            1 when some thread finished right at the start.
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
// Both counters are plain values, not rates: Google Benchmark would divide a
// rate by the mean loop time again. Thread 0 alone sets them, after its loop,
// so that summing the counters over the threads, which the framework does,
// leaves them unchanged; every other thread's record is complete by then (see
// ORDERING).
//
// WHAT IS HERE
//
//   WallStamps          one thread's two stamps, `first` and `last`; a record
//                       with either unset means "this thread did not run".
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
//                       spread, with the three stamps they are made of.
//   wall_span()         the aggregation, over a WallRecords or over an array
//                       of any record type derived from WallStamps.
//   set_wall_counters() the two counters from a WallSpan.
//   report_wall()       the one-line aggregation and report for plain bodies.
// A benchmark whose timed loop carries more per-thread state than the stamps
// keeps its own loop and its own record type, derived from WallStamps and
// aggregated by wall_span(const Rec*, int) like any other.
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
// ORDERING (Google Benchmark's default thread runner; no custom runner here)
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
#include <cstddef>

#include "benchmark/benchmark.h"

// One thread's record of one run: the end of its first and of its last
// iteration, steady-clock. Unstamped (default-constructed, the clock's zero)
// means the thread did not reach that point: a record with either stamp unset
// is ignored by wall_span(). No alignment of its own: the record type that
// embeds or derives from it carries the 128-byte block (WallRecords::Rec, or a
// benchmark's own record with more fields), so that a record stays one block
// whatever it adds.
struct WallStamps {
  std::chrono::steady_clock::time_point first;  // end of this thread's first iteration
  std::chrono::steady_clock::time_point last;   // end of this thread's last iteration
}; // struct WallStamps

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
      if (left_ != 0) [[likely]] return true;
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
  if (!state_.KeepRunningBatch(n)) {
    // Skipped: both barriers have run inside this call; no second call.
    finished_ = true;
    return Iterator(this, 0);
  } // if skipped
  mine_ = &records_.of(state_.thread_index());  // after the start barrier
  return Iterator(this, 1);
} // WallTimed::begin()

// The aggregate of a run's records: the three stamps the counters are made of,
// the span and the spread. All zero when no record is stamped.
struct WallSpan {
  std::chrono::steady_clock::time_point start {};         // earliest `first` of any thread
  std::chrono::steady_clock::time_point first_finish {};  // earliest `last`
  std::chrono::steady_clock::time_point last_finish {};   // latest `last`
  double seconds = 0.0;                 // the span: last_finish - start
  double spread = 0.0;                  // finish_spread: (last_finish - first_finish)/seconds, 0 if seconds is 0
}; // struct WallSpan

namespace gb_wall_clock_detail {

// The one aggregation, over an indexing function so that both wall_span()
// overloads share it without one indexing the other's storage (a WallStamps*
// over an array of a derived record type would be undefined behaviour).
//   at      -- int -> const WallStamps&, the record of thread i;
//   threads -- how many.
// Records with `first` or `last` unset are skipped: they belong to threads
// that did not run (an early return, a per-thread SkipWithError(), a rerun of
// the row by a memory manager with thread 0 alone while threads() still says
// N). The span and the spread are the ones every wall-timed benchmark in this
// repository reports; the arithmetic is written out exactly so.
template <typename At>
WallSpan wall_span_at(At at, int threads) {
  using TimePoint = std::chrono::steady_clock::time_point;
  WallSpan s;
  bool any = false;
  for (int i = 0; i != threads; ++i) {
    const WallStamps& rec = at(i);
    if (rec.first == TimePoint() || rec.last == TimePoint()) continue;   // did not run
    if (!any) {
      s.start = rec.first;
      s.first_finish = rec.last;
      s.last_finish = rec.last;
      any = true;
    } else {
      s.start = std::min(s.start, rec.first);
      s.first_finish = std::min(s.first_finish, rec.last);
      s.last_finish = std::max(s.last_finish, rec.last);
    } // if the first stamped record
  } // loop over the threads' records
  if (!any) return s;                   // all zero
  s.seconds = std::chrono::duration<double>(s.last_finish - s.start).count();
  s.spread = s.seconds > 0 ? std::chrono::duration<double>(s.last_finish - s.first_finish).count()/s.seconds : 0.0;
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

// Set the two counters of this run from its aggregate. Plain values, not
// rates (see WHY). The caller is thread 0 -- not checked here, so that bodies
// with their own aggregation can call it inside their own thread-0 block.
//   state -- thread 0's state;
//   s     -- the run's aggregate;
//   items -- the run's total items, over all threads.
inline void set_wall_counters(benchmark::State& state, const WallSpan& s, double items) {
  state.counters["wall_items_per_second"] = benchmark::Counter(s.seconds > 0 ? items/s.seconds : 0.0);
  state.counters["finish_spread"] = benchmark::Counter(s.spread);
} // set_wall_counters()

// The one-line report for plain bodies: on thread 0, aggregate the records
// and set the counters; on every other thread, nothing. Does not release the
// records: release() does. Call it after the loop, next to
// SetItemsProcessed(), on every thread.
//   state   -- the calling thread's state;
//   records -- the body's records, allocated for this run;
//   items   -- the run's total items, over all threads.
inline void report_wall(benchmark::State& state, const WallRecords& records, double items) {
  if (state.thread_index() != 0) return;
  set_wall_counters(state, wall_span(records), items);
} // report_wall()

#endif // INCLUDED_GB_WALL_CLOCK_H
