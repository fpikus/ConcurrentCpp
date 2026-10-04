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
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

#include "lock_free_list_rcu.h"

// Benchmarks of LockFreeListRCU, the generation-reclamation list, against the
// reference-counted LockFreeList (../LockFreeList/lock_free_list_bm.C).
//
// Why measure this: the reference-counted list pays a reference-count RMW on
// every traversal step and every anchor read, and its pointer policies free a
// node once its last reference goes (at once, or -- _IntrPtrHP -- in a later
// batch of mm_hp's hazard-pointer scan). This list pays instead once per
// handle join or refresh (a seq_cst increment of one generation's count, or a
// relaxed load when the handle is already current), one retire push per erase
// onto a global list, and caller-driven reclamation that destroys values in
// batches and recycles nodes through a global free list instead of returning
// them to malloc. Which of the two costs more depends on the operation mix
// and the thread count; that is the question every row here answers against
// its counterpart there.
//
// The five workloads are the reference-counted list's, operation for
// operation: the same mixes, the same per-thread seeds (SETUP_RNG: thread i
// of every run and repetition draws the same sequence as thread i there), the
// same read windows, the same prepopulation (1000 nodes; 100k for
// ReadDispersed, with the same evenly spaced per-thread start positions), the
// same thread range and real-time measurement. A row here and the same
// workload's rows there therefore differ only in the list and its memory
// management.
//
// Each workload runs in up to three regimes, named by the row suffix (see
// Regime below): _RCU (the intended use: a long-lived handle per thread,
// refreshed, and reclaim() called by every thread after every R = 256 of its
// own successful erases), _RCU_NoReclaim (never reclaims) and _RCU_OpHandle (a
// fresh handle per operation; head-anchored workloads only).
//
// Rates are Google Benchmark's standard items/s (SetItemsProcessed with one
// item per iteration), exactly as in the reference-counted list's benchmark,
// so the two binaries' rates are comparable as they stand. That rate divides
// the summed iterations by the MEAN per-thread time, which flatters an
// implementation whose threads finish at different times; replacing it with
// wall-clock accounting is a separate, later step for both benchmarks.
//
// Counters (user counters; Google Benchmark SUMS each over the run's threads
// and reports the total for the run, not a rate):
//   erases_ok          successful erase_after() calls
//   inserts            successful insert_after() calls
//   advances           reclaim() calls that returned `advanced`
//   freed_bags         generations whose bags were freed (sum of
//                      ReclaimResult::freed_bags); reclaimed/freed_bags is the
//                      mean bag size, advances/erases_ok the advances per erase
//   reclaimed          nodes whose values were destroyed and which moved to the
//                      free list (sum of ReclaimResult::freed_nodes)
//   free_list_hits     free-list pops, i.e. insert attempts served from the free
//                      list instead of `new` -- including a pop whose insert
//                      then failed on a deleted anchor (ReadDispersed only) and
//                      whose node was disposed to the retired list. Every node
//                      that enters the free list is counted in `reclaimed` and
//                      every node that leaves it is a pop, so pops = reclaimed
//                      - the free population left at the end (from the
//                      accounting sweep after the run)
//   retired_backlog    nodes retired but not yet freed at the end of the run
//                      (retired list + sealed bags, from the same sweep); not
//                      reported by the _RCU_NoReclaim rows (see below)
//   refresh_slow       refresh() calls that moved the handle to a newer
//                      generation (the join's seq_cst RMW instead of the fast
//                      path's relaxed load)
//   reclaim_calls      reclaim() calls (every thread's, one per kReclaimEvery of
//                      its successful erases): the denominator of the next two
//                      and of advances (the rest returned nothing_retired)
//   reclaim_contended  reclaim() calls that found another in progress and
//                      returned at once (one reclaimer at a time; with more
//                      than one thread erasing, some calls collide -- how many
//                      depends on how long a reclaim() takes against the
//                      interval between calls)
//   ring_full          reclaim() calls refused because every generation slot
//                      was still pinned: some handle stayed in one generation
//                      while kGenerations - 1 = 63 advances happened, i.e. (at
//                      most one advance per reclaim() call, one call per 256
//                      erases of a thread) while the other threads made at
//                      least 63 x 256 = ~16k successful erases. No row does
//                      that by design: the head-anchored rows refresh every
//                      iteration, and a ReadDispersed lap (list_size/50 = 2000
//                      iterations, ~100 erases per thread) spans about
//                      T x 100/256 advances, ~6 at 16 threads; and a thread
//                      that has finished its iterations releases its handle
//                      (see ThreadContext::FinishIteration()). A nonzero
//                      ring_full therefore means a holder stalled mid-run: a
//                      thread preempted while holding its handle (in the
//                      OpHandle rows, while holding a per-operation handle).
//                      [Estimate] at ~4 M successful erases/s (16-thread
//                      WriteHeavy and Graveyard _RCU rows, smoke run and
//                      2026-10-04 campaign alike: 10-14 M iterations/s x
//                      0.3-0.45 successful erases per iteration) 16k
//                      erases take ~4 ms: a stall of a few milliseconds
//                      suffices
//   minor_faults       minor page faults taken by the benchmark threads during
//                      the run (getrusage before and after the state loop):
//                      first touches of fresh heap, mostly. It depends on the
//                      process's history: the nodes an earlier run (an
//                      iteration probe, another repetition or row) freed stay
//                      in malloc's free lists, already touched, so a run that
//                      follows a bigger one may allocate millions of nodes
//                      without a fault
// The sweep runs on thread 0 after the state loop's stop barrier, outside the
// timed region. It inserts every node of every population into a hash set, so
// it is skipped where it would cost the most and tell the least:
// MassiveHeadInsert never erases (nothing retired, nothing freed: both sweep
// counters are 0 and reported as such), and the _RCU_NoReclaim rows never
// reclaim (the free list stays empty, so free_list_hits = 0 exactly), while
// their retired list holds every node erased in the run -- 0.45 per iteration
// in WriteHeavy, tens of millions of nodes at a half-second window -- so their
// retired_backlog is not reported (the counter is absent).
//
// Fixture instances: Google Benchmark creates ONE fixture object per
// registered row, and every thread of a run calls SetUp(), the body and
// TearDown() on that same object. Shared state is therefore created by thread
// 0 in SetUp() and destroyed by thread 0 in TearDown(); the state loop's start
// and stop barriers order both against the other threads, which may touch the
// shared state only inside the loop.

// The reclaim period R: every thread calls reclaim() after this many of its
// own successful erases. Large enough that a call usually finds something to
// advance, small enough that every bag stays small (about kReclaimEvery nodes
// per advance) and the free list keeps feeding inserts; the backlog left at
// the end of a run is reported (retired_backlog).
constexpr int kReclaimEvery = 256;

// How a row manages handles and reclamation; the row-name suffix.
enum class Regime {
    // _RCU: one fixture-owned handle per thread, refreshed at the start of
    // every iteration (ReadDispersed: once per lap, see there); reclaim()
    // after every kReclaimEvery own successful erases.
    Reclaim,
    // _RCU_NoReclaim: the Reclaim row's code without the reclaim() calls.
    // Erased nodes stay retired, the free list stays empty and every insert
    // calls `new`.
    NoReclaim,
    // _RCU_OpHandle: a fresh handle per iteration (an iteration is one
    // operation), destroyed at the end of it; reclaim() as in Reclaim.
    OpHandle
}; // Regime

class RcuListFixture : public benchmark::Fixture {
public:
    using List = LockFreeListRCU<int>;
    using Handle = List::handle;
    using Iterator = List::iterator;

    // One thread's fixture-owned handle on its own 128-byte block: every
    // iteration's refresh reads the handle's block pointer and a slow-path
    // refresh writes it, so handles packed into one line would false-share.
    struct alignas(128) ThreadRec {
        Handle h;  // the thread's session; joined by thread 0 in SetUp(), released by its thread on its last iteration, destroyed in TearDown()
    }; // ThreadRec

    // The list under test, shared by all threads of a run (see "Fixture
    // instances" above); created by thread 0 in SetUp(), deleted by thread 0
    // in TearDown(). Static only for the derived fixtures' convenience: a
    // member of the one fixture object would be shared just the same.
    inline static List* list = nullptr;
    // One record per thread, indexed by thread_index(); empty for the
    // OpHandle rows. Built by thread 0 in SetUp(), so a thread reads its
    // record only inside the state loop, after the start barrier.
    inline static std::vector<ThreadRec> recs;

    // A generic 1000 items, as in the reference-counted list's benchmark, for
    // the head-anchored workloads; then one handle per thread.
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            Prepopulate(1000);
            MakeThreadHandles(state.threads());
        }
    } // SetUp()

    // Every handle leaves before the list is deleted (the list's destructor
    // requires that no handle is alive); a handle its thread already released
    // on its last iteration is empty, and destroying it does nothing. Runs
    // after thread 0's stop barrier; the other threads only set counters
    // after their loops.
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            recs.clear();
            delete list;
            list = nullptr;
        }
    } // TearDown()

protected:
    // Allocates the list and inserts `count` nodes (values 0..count-1, each at
    // the head) under a short-lived handle, which leaves before the run
    // starts. Thread 0 only, before the other threads touch the list.
    static void Prepopulate(int count) {
        list = new List;
        const Handle h = list->new_handle();
        for (int i = 0; i < count; ++i) {
            (void)list->insert_after(h, list->before_begin(h), i);
        }
    } // Prepopulate()

    // Joins one handle per thread into `recs`. Thread 0 only, in SetUp():
    // each handle is transferred to its thread through the state loop's start
    // barrier (a happens-before edge, which is all the list requires of a
    // handle that changes threads). Creating the handles here rather than on
    // each thread's first pass is what lets ReadDispersed hand every thread a
    // start position obtained under that thread's own handle (see
    // RcuDispersedFixture), and it means no thread can ever reclaim while
    // another holds a position it has not yet joined to protect.
    static void MakeThreadHandles(int threads) {
        recs = std::vector<ThreadRec>(threads);
        for (ThreadRec& rec : recs) {
            rec.h = list->new_handle();
        }
    } // MakeThreadHandles()

    // One thread's view of one run: its handle discipline per the regime, its
    // reclaim period and its counts. A body local of every row: the counts
    // live in registers or on the thread's own stack, never on a shared line,
    // and are copied into state.counters after the loop.
    template <Regime kRegime>
    class ThreadContext {
    public:
        // Reads the thread's fault count before the state loop.
        explicit ThreadContext(const benchmark::State& state)
            : thread_index_(state.thread_index()), max_iterations_(state.max_iterations),
              faults_before_(MinorFaults()) {}

        // Runs `op(h)` under this iteration's handle: a fresh handle that
        // leaves when `op` returns (OpHandle), or the thread's own handle
        // after a refresh (Reclaim, NoReclaim). Head-anchored workloads only:
        // the refresh invalidates every iterator obtained before it.
        template <typename Op>
        void Run(Op&& op) {
            if constexpr (kRegime == Regime::OpHandle) {
                const Handle h = list->new_handle();
                op(h);
            } else {
                RefreshOwn();
                op(OwnHandle());
            } // if a handle per operation
        } // ThreadContext::Run()

        // The thread's fixture-owned handle (not for OpHandle rows, which
        // have none). The record is looked up on the first call, inside the
        // state loop: `recs` is built by thread 0's SetUp(), which runs
        // concurrently with this thread's SetUp().
        Handle& OwnHandle() {
            static_assert(kRegime != Regime::OpHandle, "OpHandle rows have no thread handle");
            if (rec_ == nullptr) [[unlikely]] {
                rec_ = &recs[thread_index_];
            }
            return rec_->h;
        } // ThreadContext::OwnHandle()

        // Refreshes the thread's own handle (every iteration in Run(); once
        // per lap in ReadDispersed), counting a slow-path refresh.
        void RefreshOwn() {
            if (OwnHandle().refresh()) {
                ++refresh_slow_;
            }
        } // ThreadContext::RefreshOwn()

        // Called at the end of every iteration of every row. On the thread's
        // LAST iteration (Google Benchmark runs every thread for exactly
        // state.max_iterations iterations) it releases the thread's own
        // handle. Why: threads finish at different times and the early ones
        // wait at the stop barrier; a finished thread still joined to an old
        // generation would pin every younger bag, and after kGenerations - 1
        // further advances (~16k erases by the others) every reclaim() of the
        // threads still running would return ring_full and free nothing -- the
        // tail of every _RCU row would measure the NoReclaim regime. Cost: one
        // predictable compare per iteration, the same in every row (a no-op in
        // the OpHandle rows, whose handles end with their operation). Nothing
        // may use the handle, or an iterator obtained under it, after this
        // call; TearDown() destroys the then-empty handle.
        void FinishIteration() {
            if constexpr (kRegime != Regime::OpHandle) {
                if (++iterations_ == max_iterations_) [[unlikely]] {
                    OwnHandle() = Handle{};
                }
            } // if the thread owns a handle
        } // ThreadContext::FinishIteration()

        // Records an insert_after() result.
        void Inserted(bool ok) { inserts_ += ok; }

        // Records an erase_after() result and, in the reclaiming regimes,
        // calls reclaim() after every kReclaimEvery successful erases --
        // whatever the call returns (a call that collides with another
        // thread's returns `contended` at once). The call runs while the
        // iteration's handle is alive, so it cannot free the generation that
        // handle is in; a later call frees it once the handle has moved on:
        // the next advance in the head-anchored rows, which refresh every
        // iteration, up to a lap later in ReadDispersed, which refreshes once
        // per lap (a delay, not a leak).
        void Erased(bool ok) {
            if (!ok) {
                return;
            }
            ++erases_ok_;
            if constexpr (kRegime != Regime::NoReclaim) {
                if (++since_reclaim_ == kReclaimEvery) {
                    since_reclaim_ = 0;
                    const ReclaimResult result = list->reclaim();
                    ++reclaim_calls_;
                    reclaimed_ += result.freed_nodes;
                    freed_bags_ += result.freed_bags;
                    advances_ += result.advance == ReclaimResult::Advance::advanced;
                    reclaim_contended_ += result.advance == ReclaimResult::Advance::contended;
                    ring_full_ += result.advance == ReclaimResult::Advance::ring_full;
                } // every kReclaimEvery successful erases
            } // if the regime reclaims
        } // ThreadContext::Erased()

        // After the state loop: sets this thread's counters (Google Benchmark
        // sums them over the threads). `erasing` is false for a workload that
        // never erases. Thread 0 runs the accounting sweep when the workload
        // erases and the regime reclaims (see the sweep paragraph at the top
        // of the file for the other cases; the stop barrier has passed and
        // the other threads perform no list operation after their loops: the
        // sweep's quiescence precondition) and folds the end-of-run
        // populations into free_list_hits and retired_backlog. The fault
        // count is read before the sweep, which allocates.
        void Report(benchmark::State& state, bool erasing) {
            const bool sweep = erasing && kRegime != Regime::NoReclaim;
            const int64_t faults = MinorFaults() - faults_before_;
            int64_t free_at_end = 0;     // exact without a sweep: no reclaim, no free list
            int64_t backlog_at_end = 0;  // exact without a sweep only when nothing is erased
            if (sweep && thread_index_ == 0) {
                const List::InternalAccounting acc = list->get_internal_accounting();
                free_at_end = static_cast<int64_t>(acc.free);
                backlog_at_end = static_cast<int64_t>(acc.retired + acc.bagged);
            }
            state.counters["erases_ok"] = static_cast<double>(erases_ok_);
            state.counters["inserts"] = static_cast<double>(inserts_);
            state.counters["advances"] = static_cast<double>(advances_);
            state.counters["freed_bags"] = static_cast<double>(freed_bags_);
            state.counters["reclaimed"] = static_cast<double>(reclaimed_);
            // Summed over the threads: the total reclaimed minus thread 0's
            // end-of-run free population (thread 0's own term may be negative).
            state.counters["free_list_hits"] = static_cast<double>(reclaimed_ - free_at_end);
            if (sweep || !erasing) {
                state.counters["retired_backlog"] = static_cast<double>(backlog_at_end);
            } // absent from the _RCU_NoReclaim rows of erasing workloads (not measured)
            state.counters["refresh_slow"] = static_cast<double>(refresh_slow_);
            state.counters["reclaim_calls"] = static_cast<double>(reclaim_calls_);
            state.counters["reclaim_contended"] = static_cast<double>(reclaim_contended_);
            state.counters["ring_full"] = static_cast<double>(ring_full_);
            state.counters["minor_faults"] = static_cast<double>(faults);
        } // ThreadContext::Report()

    private:
        // Minor page faults of the calling thread so far (Linux RUSAGE_THREAD).
        // Where RUSAGE_THREAD does not exist, thread 0 reports the process's
        // count and the other threads 0, so the sum is the process's faults
        // over the run (other threads' SetUp() included).
        int64_t MinorFaults() const {
            rusage usage{};
#ifdef RUSAGE_THREAD
            getrusage(RUSAGE_THREAD, &usage);
#else
            if (thread_index_ != 0) {
                return 0;
            }
            getrusage(RUSAGE_SELF, &usage);
#endif
            return usage.ru_minflt;
        } // ThreadContext::MinorFaults()

        // This thread's counts; each is reported under its name without the
        // trailing underscore (see the counter list at the top of the file).
        const int thread_index_;          // this thread's index in the run
        const benchmark::IterationCount max_iterations_;  // the iterations this thread runs (state.max_iterations)
        benchmark::IterationCount iterations_ = 0;        // iterations finished so far (FinishIteration())
        const int64_t faults_before_;     // MinorFaults() before the state loop
        ThreadRec* rec_ = nullptr;        // this thread's record; looked up on the first OwnHandle() call
        int since_reclaim_ = 0;           // successful erases since this thread's last reclaim() call
        int64_t erases_ok_ = 0;           // successful erase_after() calls
        int64_t inserts_ = 0;             // successful insert_after() calls
        int64_t advances_ = 0;            // this thread's reclaim() calls that returned `advanced`
        int64_t freed_bags_ = 0;          // sum of freed_bags over this thread's reclaim() calls
        int64_t reclaimed_ = 0;           // sum of freed_nodes over this thread's reclaim() calls
        int64_t refresh_slow_ = 0;        // refresh() calls that returned true (moved the handle)
        int64_t reclaim_calls_ = 0;       // this thread's reclaim() calls
        int64_t reclaim_contended_ = 0;   // of them, returned `contended`
        int64_t ring_full_ = 0;           // of them, returned `ring_full`
    }; // ThreadContext

    // The workload bodies, one per workload, shared by the regimes. Each is
    // the reference-counted list's row body with the handle added: the same
    // RNG draws in the same order (an operation draw, then a value draw for
    // an insert), so thread i's operation sequence is that row's.
    template <Regime kRegime>
    void ReadHeavyBody(benchmark::State& state);
    template <Regime kRegime>
    void WriteHeavyBody(benchmark::State& state);
    template <Regime kRegime>
    void GraveyardBody(benchmark::State& state);
    template <Regime kRegime>
    void MassiveHeadInsertBody(benchmark::State& state);
}; // RcuListFixture

// Deterministic per-thread seed, as in the reference-counted list's
// benchmark: every run and every repetition of a row sees the same operation
// sequence, so runs are comparable, and thread i draws what thread i draws
// there.
#define SETUP_RNG \
    std::mt19937 rng(state.thread_index() + 42); \
    std::uniform_int_distribution<int> dist(0, 99)

// ReadHeavy: 90% traversals of the first 50 nodes, 5% inserts and 5% erases
// at the head. For this list: the read side's gain -- a traversal step is a
// plain acquire load where the reference-counted list pays a count RMW on a
// node every other thread is also traversing.
template <Regime kRegime>
void RcuListFixture::ReadHeavyBody(benchmark::State& state) {
    SETUP_RNG;
    ThreadContext<kRegime> ctx(state);
    for (auto _ : state) {
        int op = dist(rng);
        ctx.Run([&](const Handle& h) {
            if (op < 90) { // 90% read
                int count = 0;
                for (Iterator it = list->begin(h); it != list->end() && count < 50; ++it) {
                    benchmark::DoNotOptimize(*it);
                    count++;
                }
            } else if (op < 95) { // 5% insert
                ctx.Inserted(list->insert_after(h, list->before_begin(h), dist(rng)));
            } else { // 5% erase
                ctx.Erased(list->erase_after(h, list->before_begin(h)));
            }
        }); // one operation under the iteration's handle
        ctx.FinishIteration();
    } // state loop
    state.SetItemsProcessed(state.iterations());
    ctx.Report(state, true);
} // RcuListFixture::ReadHeavyBody()

// WriteHeavy: 10% traversals of the first 10 nodes, 45% inserts and 45%
// erases at the head. For this list: the write side -- nearly every iteration
// hits one of the scheme's global words (a retire push per erase; a free-list
// pop per insert in _RCU, `new` in _RCU_NoReclaim) besides the head's CAS.
template <Regime kRegime>
void RcuListFixture::WriteHeavyBody(benchmark::State& state) {
    SETUP_RNG;
    ThreadContext<kRegime> ctx(state);
    for (auto _ : state) {
        int op = dist(rng);
        ctx.Run([&](const Handle& h) {
            if (op < 10) { // 10% read
                int count = 0;
                for (Iterator it = list->begin(h); it != list->end() && count < 10; ++it) {
                    benchmark::DoNotOptimize(*it);
                    count++;
                }
            } else if (op < 55) { // 45% insert
                ctx.Inserted(list->insert_after(h, list->before_begin(h), dist(rng)));
            } else { // 45% erase
                ctx.Erased(list->erase_after(h, list->before_begin(h)));
            }
        }); // one operation under the iteration's handle
        ctx.FinishIteration();
    } // state loop
    state.SetItemsProcessed(state.iterations());
    ctx.Report(state, true);
} // RcuListFixture::WriteHeavyBody()

// Graveyard: 70% erases and 30% inserts at the head, no traversals. The 1000
// prepopulated nodes are gone after ~2500 iterations (a net -0.4 nodes per
// iteration); from then on the list is nearly empty and an erase succeeds
// only after an insert, so about 3/7 of the erases succeed, and in the _RCU
// row nearly every insert pops the free list (the most retirements per
// iteration of any workload, hence the most reclamation per iteration).
template <Regime kRegime>
void RcuListFixture::GraveyardBody(benchmark::State& state) {
    SETUP_RNG;
    ThreadContext<kRegime> ctx(state);
    for (auto _ : state) {
        int op = dist(rng);
        ctx.Run([&](const Handle& h) {
            if (op < 70) { // 70% erase
                ctx.Erased(list->erase_after(h, list->before_begin(h)));
            } else { // 30% insert
                ctx.Inserted(list->insert_after(h, list->before_begin(h), dist(rng)));
            }
        }); // one operation under the iteration's handle
        ctx.FinishIteration();
    } // state loop
    state.SetItemsProcessed(state.iterations());
    ctx.Report(state, true);
} // RcuListFixture::GraveyardBody()

// MassiveHeadInsert: inserts at the head only (one value draw, no operation
// draw, as there). Nothing is erased, so nothing is retired or reclaimed and
// retired_head_ is never touched: the row isolates the per-iteration handle
// cost (refresh, or join and leave for OpHandle) on top of `new` and the head
// CAS. The list grows by one node per iteration for the whole run, so this
// row also measures fresh-heap first touches (minor_faults).
template <Regime kRegime>
void RcuListFixture::MassiveHeadInsertBody(benchmark::State& state) {
    SETUP_RNG;
    ThreadContext<kRegime> ctx(state);
    for (auto _ : state) {
        ctx.Run([&](const Handle& h) {
            ctx.Inserted(list->insert_after(h, list->before_begin(h), dist(rng)));
        }); // one insert under the iteration's handle
        ctx.FinishIteration();
    } // state loop
    state.SetItemsProcessed(state.iterations());
    ctx.Report(state, false);
} // RcuListFixture::MassiveHeadInsertBody()

// Fixture of the _RCU_OpHandle rows: the head-anchored prepopulation and NO
// thread handles. An idle fixture-owned handle would stay in the first
// generation for the whole run and, since bags are freed oldest-first, pin
// every bag after it: nothing would ever be freed and reclaim() would hit
// ring_full after kGenerations - 1 advances.
class RcuOpHandleFixture : public RcuListFixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            Prepopulate(1000);
        }
    } // SetUp()
}; // RcuOpHandleFixture

// Fixture of the ReadDispersed rows: a list two orders of magnitude larger and
// evenly spaced per-thread start positions, as in the reference-counted
// list's DispersedListFixture (same size, same spacing, same reason: threads
// begin far apart and advance at about the same rate over a list ~60x larger
// than all their read windows combined, so they rarely meet). The difference:
// thread t's start position is obtained under thread t's own handle, so it is
// valid for that handle (an iterator is bound to the handle it was obtained
// under) and protected by it from the start -- whatever the other threads
// erase and reclaim before thread t's first pass, the node stays allocated.
class RcuDispersedFixture : public RcuListFixture {
public:
    // 32 threads x 50-node windows cover 1.6% of the list; the nodes (24
    // bytes for `int`, each its own 32-byte malloc chunk: 3.2 MB in all, as
    // for the reference-counted list's intrusive policy) exceed a 1 MB L2, as
    // a dispersed workload should.
    static constexpr int list_size = 100'000;

    // One start iterator per thread, list_size/threads apart, each obtained
    // under that thread's handle. Written by thread 0 in SetUp(), read by each
    // thread on its first pass through the state loop.
    inline static std::vector<Iterator> start_positions;

    // Each position is walked from begin() under its own handle: stride*t
    // steps for thread t, list_size*(threads - 1)/2 in all (750k at 16
    // threads), untimed.
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            Prepopulate(list_size);
            MakeThreadHandles(state.threads());
            start_positions.clear();
            const int stride = list_size/state.threads();
            for (int t = 0; t < state.threads(); ++t) {
                Iterator it = list->begin(recs[t].h);
                for (int i = 0; i < stride*t; ++i) {
                    ++it;
                }
                start_positions.push_back(it);
            } // one start position per thread, under its handle
        }
    } // SetUp()

    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            start_positions.clear(); // they point into the list about to be deleted
        }
        RcuListFixture::TearDown(state);
    } // TearDown()

protected:
    template <Regime kRegime>
    void ReadDispersedBody(benchmark::State& state);
}; // RcuDispersedFixture

// ReadDispersed: every iteration reads a 50-node window from this thread's
// persistent cursor; 5% of iterations then insert and 5% erase, anchored at
// the node mid-window (the reference-counted list's row, line for line). The
// cursor survives iterations, and refresh() would invalidate it, so the
// handle is refreshed only once per lap, at the wrap back to begin(): the
// handle then pins every node retired during a lap (about list_size/50
// iterations of this thread). Reclaim and NoReclaim only: an OpHandle row
// would destroy the handle that keeps the cursor's node alive at the end of
// every iteration.
template <Regime kRegime>
void RcuDispersedFixture::ReadDispersedBody(benchmark::State& state) {
    static_assert(kRegime != Regime::OpHandle, "the cursor outlives a per-operation handle");
    SETUP_RNG;
    ThreadContext<kRegime> ctx(state);
    constexpr int window = 50;      // elements read per pass
    constexpr int mid = window/2;   // offset within the window where writes are anchored
    Iterator cursor;                // this thread's staggered start, assigned on the first pass below
    bool first_pass = true;
    for (auto _ : state) {
        if (first_pass) {
            // start_positions is built by thread 0's SetUp(), readable only
            // after the loop's start barrier.
            cursor = start_positions[state.thread_index()];
            first_pass = false;
        }
        const Handle& h = ctx.OwnHandle();
        int op = dist(rng);
        Iterator it = cursor;
        Iterator anchor = cursor; // falls back to the window start if the list is too short to reach `mid`
        for (int count = 0; it != list->end() && count < window; ++count) {
            benchmark::DoNotOptimize(*it);
            if (count == mid - 1) {
                anchor = it; // predecessor of the mutation target, mid-way through the window
            }
            ++it;
        } // read the window
        if (op >= 90) {
            if (op < 95) { // 5% insert, anchored mid-window instead of at the head
                ctx.Inserted(list->insert_after(h, anchor, dist(rng)));
            } else { // 5% erase, anchored mid-window instead of at the head
                ctx.Erased(list->erase_after(h, anchor));
            }
        } // 10% writes
        if (it == list->end()) {
            // The lap is over: release the generations this handle pinned
            // since the last wrap, then restart (the refresh invalidated
            // `cursor`, `it` and `anchor`).
            ctx.RefreshOwn();
            cursor = list->begin(h);
        } else {
            cursor = it; // resume here next pass
        } // wrap at the tail
        ctx.FinishIteration(); // last: on the final iteration it releases `h`; `cursor` is not used after it
    } // state loop
    state.SetItemsProcessed(state.iterations());
    ctx.Report(state, true);
} // RcuDispersedFixture::ReadDispersedBody()

// The rows. Why each regime is measured:
// - _RCU: the list as intended -- a long-lived handle per thread, refreshed
//   per operation (ReadDispersed: per lap), every thread sharing
//   reclamation. This is the row to set against the reference-counted
//   list's rows of the same workload.
// - _RCU_NoReclaim: the same code without reclamation, so the pair isolates
//   what reclamation costs and buys: here every insert pays glibc's malloc
//   and the footprint grows for the whole run, against the _RCU row's
//   cross-core free-list pop (a LIFO stack hands an inserter a node last
//   touched by whichever thread freed its bag), its contention on the one
//   free-list head word, and the reclaim() calls themselves. First-touch page
//   faults, measured 2026-10-04 (WSL2): the median repetition takes none or
//   a negligible number (none in 63 of 70 RCU cells, at most 0.14 per 1000
//   iterations), because earlier runs of the same process pre-touch the
//   heap; a repetition that runs on fresh heap is 19-42% slower. Read
//   minor_faults alongside the rate (see minor_faults above). On
//   MassiveHeadInsert, which never erases, _RCU and _RCU_NoReclaim execute
//   identical code: that pair is a labelled control, and its difference is
//   the noise floor of the comparison.
// - _RCU_OpHandle: a handle per operation, the worst case of the handle
//   discipline: every operation joins (a seq_cst increment of the current
//   generation's count, all threads on that one cache line, plus a seq_cst
//   re-check) and leaves (a decrement of the same line). It measures what a
//   caller pays for not keeping a handle, and the contention of that line;
//   head-anchored workloads only (see ReadDispersedBody()).
BENCHMARK_DEFINE_F(RcuListFixture, ReadHeavy_RCU)(benchmark::State& state) { ReadHeavyBody<Regime::Reclaim>(state); }
BENCHMARK_DEFINE_F(RcuListFixture, WriteHeavy_RCU)(benchmark::State& state) { WriteHeavyBody<Regime::Reclaim>(state); }
BENCHMARK_DEFINE_F(RcuListFixture, Graveyard_RCU)(benchmark::State& state) { GraveyardBody<Regime::Reclaim>(state); }
BENCHMARK_DEFINE_F(RcuListFixture, MassiveHeadInsert_RCU)(benchmark::State& state) { MassiveHeadInsertBody<Regime::Reclaim>(state); }
BENCHMARK_DEFINE_F(RcuDispersedFixture, ReadDispersed_RCU)(benchmark::State& state) { ReadDispersedBody<Regime::Reclaim>(state); }

BENCHMARK_DEFINE_F(RcuListFixture, ReadHeavy_RCU_NoReclaim)(benchmark::State& state) { ReadHeavyBody<Regime::NoReclaim>(state); }
BENCHMARK_DEFINE_F(RcuListFixture, WriteHeavy_RCU_NoReclaim)(benchmark::State& state) { WriteHeavyBody<Regime::NoReclaim>(state); }
BENCHMARK_DEFINE_F(RcuListFixture, Graveyard_RCU_NoReclaim)(benchmark::State& state) { GraveyardBody<Regime::NoReclaim>(state); }
BENCHMARK_DEFINE_F(RcuListFixture, MassiveHeadInsert_RCU_NoReclaim)(benchmark::State& state) { MassiveHeadInsertBody<Regime::NoReclaim>(state); }
BENCHMARK_DEFINE_F(RcuDispersedFixture, ReadDispersed_RCU_NoReclaim)(benchmark::State& state) { ReadDispersedBody<Regime::NoReclaim>(state); }

BENCHMARK_DEFINE_F(RcuOpHandleFixture, ReadHeavy_RCU_OpHandle)(benchmark::State& state) { ReadHeavyBody<Regime::OpHandle>(state); }
BENCHMARK_DEFINE_F(RcuOpHandleFixture, WriteHeavy_RCU_OpHandle)(benchmark::State& state) { WriteHeavyBody<Regime::OpHandle>(state); }
BENCHMARK_DEFINE_F(RcuOpHandleFixture, Graveyard_RCU_OpHandle)(benchmark::State& state) { GraveyardBody<Regime::OpHandle>(state); }
BENCHMARK_DEFINE_F(RcuOpHandleFixture, MassiveHeadInsert_RCU_OpHandle)(benchmark::State& state) { MassiveHeadInsertBody<Regime::OpHandle>(state); }

static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// As in the reference-counted list's benchmark: ThreadRange(1, num_cpu)
// doubles the thread count up to the CPU count; UseRealTime() measures wall
// time per iteration rather than CPU time.
#define REGISTER_RCU_BM(Fixture, name) \
    BENCHMARK_REGISTER_F(Fixture, name)->ThreadRange(1, num_cpu)->UseRealTime()

REGISTER_RCU_BM(RcuListFixture, ReadHeavy_RCU);
REGISTER_RCU_BM(RcuListFixture, WriteHeavy_RCU);
REGISTER_RCU_BM(RcuListFixture, Graveyard_RCU);
REGISTER_RCU_BM(RcuListFixture, MassiveHeadInsert_RCU);
REGISTER_RCU_BM(RcuDispersedFixture, ReadDispersed_RCU);

REGISTER_RCU_BM(RcuListFixture, ReadHeavy_RCU_NoReclaim);
REGISTER_RCU_BM(RcuListFixture, WriteHeavy_RCU_NoReclaim);
REGISTER_RCU_BM(RcuListFixture, Graveyard_RCU_NoReclaim);
REGISTER_RCU_BM(RcuListFixture, MassiveHeadInsert_RCU_NoReclaim);
REGISTER_RCU_BM(RcuDispersedFixture, ReadDispersed_RCU_NoReclaim);

REGISTER_RCU_BM(RcuOpHandleFixture, ReadHeavy_RCU_OpHandle);
REGISTER_RCU_BM(RcuOpHandleFixture, WriteHeavy_RCU_OpHandle);
REGISTER_RCU_BM(RcuOpHandleFixture, Graveyard_RCU_OpHandle);
REGISTER_RCU_BM(RcuOpHandleFixture, MassiveHeadInsert_RCU_OpHandle);

BENCHMARK_MAIN();
