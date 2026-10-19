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
#include <algorithm>
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: timed erase(), alone and under
 * reclamation, on two table sizes. One of the benchmark's translation units
 * (the overview of the workloads, the index of the units and why there are
 * several: concurrent_hash_set_rcu_bm.C); the code generated for these rows
 * depends on this unit alone. The erase fixture, its geometries and the
 * erase body are here; the handles, the tallies and the snapshots they use
 * are the shared header's.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  Erase_Presized_Del             timed erase() of present keys, disjoint per
 *                                 thread: what eager unlinking costs the
 *                                 eraser, and the share of erases that
 *                                 retired their own node.
 *  Erase_Presized_Del_Rcu         the same erases, and each thread releases
 *                                 its handle and calls advance() once, after
 *                                 its last erase, inside the timed region:
 *                                 against Erase_Presized_Del, one advance()
 *                                 per thread at a geometry where one walk of
 *                                 the table takes longer than the erases; at
 *                                 one thread the call frees every erased
 *                                 node, at more it bumps and walks and
 *                                 usually frees nothing (see the row).
 *  Erase_Small_Del                timed erase() on a small pre-sized table,
 *                                 the same keys erased over the run whatever
 *                                 the thread count (every prefilled key when
 *                                 the thread count divides their number): the
 *                                 control of the next row.
 *  Erase_Small_Del_Rcu            the same erases under the reclamation
 *                                 regime (one advance() per
 *                                 kSmallAdvanceEvery erases of the run, four
 *                                 per run at every thread count, every call
 *                                 but the first freeing a bag): the erase
 *                                 cost under reclamation, on a table whose
 *                                 walk is short in absolute terms.
 *  Erase_Small_Del_RcuEach        the same erases with EVERY thread an
 *                                 advancer, each after every
 *                                 kSmallAdvanceEvery/T of its own erases (the
 *                                 first calls staggered so they do not land
 *                                 together), with the walk: the attempt
 *                                 frequency of the _Auto row below, called
 *                                 explicitly (equal when the per-thread
 *                                 cadence is a power of two, which the set
 *                                 rounds its cadence up to: every power-of-two
 *                                 thread count). The pair _Auto/_RcuEach says
 *                                 what the walk's absence buys at equal
 *                                 attempt frequency; _RcuEach/_Rcu what the
 *                                 frequency alone does; neither is readable
 *                                 from _Auto/_Rcu, which differ in both.
 *  Erase_Small_Del_Auto           the same erases on a set that runs the bump
 *                                 and the free pass itself, from the retiring
 *                                 thread, every kSmallAdvanceEvery/T nodes
 *                                 retired per arena shard (the _Rcu row's
 *                                 per-thread cadence), never walking: what an
 *                                 erase costs when reclamation is automatic
 *                                 and spread over the threads, read against
 *                                 the control, with the one-advancer-with-walk
 *                                 _Rcu row and the per-thread-caller _RcuEach
 *                                 row beside it; the attempts that proceeded
 *                                 against those skipped, and which shard each
 *                                 thread retired into (the "# shards" line).
 */

// ---------------------------------------------------------------------------
// The erase rows: timed erase() of present keys.
//
// Erase_Presized_Del. Why measure: in this container an uncontended erase()
// marks its node (one CAS), unlinks it (one CAS on its predecessor's word) and
// retires it (one release CAS on the retired head of the erasing thread's
// shard): three CASes where an erase that only marks does one, plus a mark
// retried when a concurrent unlink through the node's link changes it, and a
// second walk from the bucket head when the first unlink attempt loses. The
// Erased churn cells measure what that buys the later walks; this row measures
// what it costs the eraser.
// Shape: thread 0's SetUp() prefills kErasePrefill mix()ed keys, untimed, into
// kPresizedBuckets buckets (no doubling, no split: half a node per bucket);
// thread t erases mix(t*n + i) for i < n, n = kErasePrefill/num_cpu, so the
// threads' keys are disjoint and all present, and every erase() must return
// true (checked per thread, after the loop). The prefill is single-threaded, so
// every node lives in thread 0's shard; the retire push goes to the ERASING
// thread's shard, not to the node's, so the timed erases do not all push onto one
// retired head.
// Counter, read by thread 0 after the loop (every thread has left it, so the
// container's diagnostic counters are exact there), untimed:
//   retired_per_erase -- nodes retired during the timed loop per timed erase: 1
//                        when every erase took its own node out of its chain,
//                        less by the share left in place under contention (for
//                        a later writer or advance()'s walk to collect); in the
//                        _Rcu rows it includes the nodes their walks retired
//                        (the counter walk_retired).
//
// Erase_Presized_Del_Rcu. Why measure: what reclamation adds to the erases of
// the row above, at its geometry and paired with it. The same fixture and the
// same erases; after its last erase each thread releases its handle
// (ThreadHandles::Own::finish_iteration()) and then calls advance() exactly
// once, inside the timed region. With its own handle released first, the
// call is not pinned by its caller. At one thread no other handle exists, so
// the call bags every node the erases retired and frees the bag. At more than
// one, the threads' handles were made in SetUp() and are never refreshed:
// the first thread to finish bags the retired nodes into the generation the
// still-running threads' handles hold, so its free pass frees nothing; its
// call bumps and walks. A call that finds another thread's call in progress
// returns `contended` and is not retried, so the row expects about one
// effective walk, and only a last finisher's call that comes after that walk
// could free the bag. freed_nodes reports what was freed. At this geometry one
// walk of the table takes longer than one thread's erases (the Advance_Walk
// rows give its cost), so the row is mostly walk at one thread and says so
// through walk_share. No handle is refreshed: no advance() runs before the
// last iteration, so a refresh would never move one. Counters: AdvanceTally's,
// beside retired_per_erase.
//
// Erase_Small_Del and Erase_Small_Del_Rcu. Why measure: the cost of an erase
// under reclamation, with the reclamation done at a fixed number of calls per
// population erased, on a table small enough that its walk is short in
// absolute terms (Erase_Presized_Del_Rcu's one walk is long). Thread 0's
// SetUp() prefills kSmallPrefill mix()ed keys into kSmallBuckets buckets (the
// pre-sized rows' load factor). Thread t of T erases mix(t*n + i) for
// i < n = kSmallPrefill/T (the rows are registered one thread count at a
// time, as InsertHotGrowth is): every prefilled key is erased exactly once
// when T divides kSmallPrefill (every power of two does), and the last
// kSmallPrefill mod T keys are left otherwise. The checks and
// retired_per_erase as above.
// Erase_Small_Del: the erases alone, the control.
// Erase_Small_Del_Rcu: the reclamation regime. Every thread refreshes its
// handle at the top of every iteration, and thread 0 calls advance() after
// every kSmallAdvanceEvery/T of its own erases: at the threads' equal rates,
// once per kSmallAdvanceEvery erases of the run, so every run makes
// kSmallPrefill/kSmallAdvanceEvery calls whatever the thread count
// (AdvanceTally says why one advancer). Thread 0 does every walk and finishes
// last (finish_spread); the walks' share of the threads' summed loop time
// falls as the thread count grows and the other threads' erases end sooner.
// A call bags what the erases retired since the
// previous call and frees the bags no handle pins any more: the advancer's
// own handle, refreshed before its erase, still pins the generation the call
// closes, so each bag is freed one call later. Counters: AdvanceTally's and
// ReclaimSnapshot's, beside retired_per_erase.
// Erase_Small_Del_RcuEach: every thread an advancer, each after every
// kSmallAdvanceEvery/T of its own erases (four calls per thread, 4T per run,
// the first ones staggered: AdvanceTally), with the walk. Why measure: the
// attempt frequency of Erase_Small_Del_Auto, called explicitly; the pair
// separates the walk's share (_Auto/_RcuEach) from the frequency's
// (_RcuEach/_Rcu). Every call walks the whole table while the other threads'
// loops are short (kSmallPrefill/T erases each), so at larger thread counts
// most calls are expected `contended` and the walkers to straggle
// (finish_spread); the counters say which.
// Erase_Small_Del_Auto: no call; the set is constructed with the automatic
// cadence kSmallAdvanceEvery/T nodes retired per arena shard
// (SmallAutoCadence; the set rounds it up to a power of two, so it equals
// _RcuEach's per-thread cadence at power-of-two thread counts, and the
// auto_reclaim_every counter is the witness): each thread erases kSmallPrefill/T keys into its own
// shard's count, so each crosses four times, 4T attempts per run, and the
// set runs the bump and the free pass from inside the crossing erase, on
// that thread, never walking. Every thread refreshes at the top of every
// iteration and releases on its last, as _Rcu; the trigger detection is the
// window rows' (see Churn_Window_Del, concurrent_hash_set_rcu_bm_churn.C). Why measure: what an erase costs when
// reclamation is automatic and spread over the threads, against the control,
// with the one-advancer-with-walk _Rcu row and the per-thread-caller
// _RcuEach row beside it. Two threads of a run can share an arena shard (the
// "# shards" line, ReclaimSnapshot), in which case that shard crosses twice
// as often and the attempts are the same in total.
// ---------------------------------------------------------------------------
static constexpr int    kErasePrefill = 1 << 22;
static constexpr size_t kSmallBuckets = size_t(1) << 17;
static constexpr int    kSmallPrefill = 1 << 16;
// Erase_Small_Del_Rcu's cadence: one advance() per kSmallAdvanceEvery erases
// of the run, a quarter of the prefilled keys, so four calls per run (thread 0
// calls after every kSmallAdvanceEvery/T of its own erases;
// HASH_BM_SMALL_ADVANCE_EVERY overrides it: advance_every_from_env()). Every
// call but the first frees the bag the one before it closed. A walk passes
// every live node. _RcuEach and _Auto divide it by T the same way (the
// per-thread cadence, and the per-shard one).
static constexpr int    kSmallAdvanceEvery = kSmallPrefill/4;
// The per-thread cadence of the small reclamation rows: kSmallAdvanceEvery/T
// of a thread's own erases (the _Rcu row's thread 0, every _RcuEach thread),
// and the automatic cadence of the _Auto row (nodes retired per arena shard:
// SmallAutoCadence; the set rounds it up to a power of two, so at a thread
// count or an override that is not one the _Auto row's cadence is the next
// power of two above this value), with the environment override; at least 1.
static int small_cadence(const benchmark::State& state) {
    return std::max(1, advance_every_from_env("HASH_BM_SMALL_ADVANCE_EVERY", kSmallAdvanceEvery)/state.threads());
} // small_cadence()
// The automatic cadence of Erase_Small_Del_Auto (see NoCadence).
struct SmallAutoCadence {
    static size_t auto_reclaim_every(const benchmark::State& state) { return size_t(small_cadence(state)); }
}; // struct SmallAutoCadence

// The two erase geometries, EraseFixture's second parameter: the table's
// buckets, the prefilled keys, and the keys each thread erases.
struct PresizedErase {
    static constexpr size_t buckets = kPresizedBuckets;
    static constexpr int    prefill = kErasePrefill;
    static int keys_per_thread(const benchmark::State& /*state*/) { return kErasePrefill/num_cpu; }
}; // struct PresizedErase
struct SmallErase {
    static constexpr size_t buckets = kSmallBuckets;
    static constexpr int    prefill = kSmallPrefill;
    static int keys_per_thread(const benchmark::State& state) { return kSmallPrefill/state.threads(); }
}; // struct SmallErase

// The erase fixture: the geometry `Geom` (above) and the automatic cadence
// policy `Cadence` (NoCadence of the shared header, or an _Auto row's); the cadence is a
// template parameter so that an _Auto row is a fixture of its own with its
// own set, and the kept rows' names (the fixture's written template
// arguments) do not change.
template <typename SetType, typename Geom = PresizedErase, typename Cadence = NoCadence>
class EraseFixture : public benchmark::Fixture {
public:
    using Geometry = Geom;    // the bodies read the per-thread key count through it
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles, shared header)
    using Tally = AdvanceTally<SetType>;   // one thread's advance() calls or triggers in the reclamation bodies (AdvanceTally, shared header)
    static ReclaimSnapshot<SetType> before;   // the container's counters after the prefill (ReclaimSnapshot, shared header)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() != 0) return;
        set = make_prefilled<SetType>(state, "EraseFixture", Geometry::buckets, Cadence::auto_reclaim_every(state), [](auto insert) {
            for (int k = 0; k < Geometry::prefill; ++k) insert(mix(k));
        });
        before.take(*set);
        handles.make(*set, state.threads());
    } // EraseFixture::SetUp()
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class EraseFixture
template <typename SetType, typename Geom, typename Cadence> SetType* EraseFixture<SetType, Geom, Cadence>::set = nullptr;
template <typename SetType, typename Geom, typename Cadence> ThreadHandles<SetType> EraseFixture<SetType, Geom, Cadence>::handles;
template <typename SetType, typename Geom, typename Cadence> ReclaimSnapshot<SetType> EraseFixture<SetType, Geom, Cadence>::before;

// The body of the erase rows (see above), as a macro like the shared header's bodies, so
// that every erase row runs the same code around its own regime. ITERATION is
// the loop body: it erases the thread's next key into `ok` and does what the
// row's regime adds; PRE declares what the regime needs before the loop and
// POST reports it after; both are empty in the rows without reclamation. The
// fixture's template arguments come last. Same accounting contract as the
// macro bodies: allocate() before the loop; report_wall() then release()
// after it; the per-thread SkipWithError() after the loop, never inside it.
#define DEFINE_ERASE(NAME, PRE, ITERATION, POST, ...)                         \
    BENCHMARK_TEMPLATE_DEFINE_F(EraseFixture, NAME, __VA_ARGS__)              \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        const int base = state.thread_index()*Geometry::keys_per_thread(state); \
        int next = 0;                                                         \
        int64_t ok = 0;                                                       \
        Handles::Own own(handles, state);                                     \
        PRE                                                                   \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            ITERATION                                                         \
        }                                                                     \
        if (ok != state.iterations()) {                                       \
            state.SkipWithError("erase() returned false for a present key no other thread erases"); \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        report_wall(state, records, double(state.iterations())*state.threads()); \
        if (state.thread_index() == 0) {   /* the container's diagnostic counter (see the fixture) */ \
            const double erased = double(state.iterations())*state.threads(); \
            state.counters["retired_per_erase"] = benchmark::Counter(         \
                double(set->get_internal_counters().retire_nodes - before.counters.retire_nodes)/erased); \
        }                                                                     \
        POST                                                                  \
        records.release(state);                                               \
    }

// The erase rows (see Erase_Presized_Del above): the two pre-sized rows over
// ThreadRange(1, num_cpu) with kErasePrefill/num_cpu erases per thread at
// every thread count, the two small rows with kSmallPrefill erases per run.
DEFINE_ERASE(Erase_Presized_Del,
    /* PRE */,
    /* ITERATION */
    ok += set->erase(own.get(), mix(base + next++));,
    /* POST */,
    ConcurrentSetDel)
BENCHMARK_REGISTER_F(EraseFixture, Erase_Presized_Del)
    ->ThreadRange(1, num_cpu)->Iterations(kErasePrefill/num_cpu)->UseRealTime();

DEFINE_ERASE(Erase_Presized_Del_Rcu,
    /* PRE */
    Tally tally;,
    /* ITERATION: after the last erase, the release, then the one advance() */
    ok += set->erase(own.get(), mix(base + next++));
    if (BM_UNLIKELY(own.finish_iteration())) tally.add(advance_out_of_line(*set));,
    /* POST */
    tally.report(state);
    if (state.thread_index() == 0) before.report_set(state, *set);,
    ConcurrentSetDel)
BENCHMARK_REGISTER_F(EraseFixture, Erase_Presized_Del_Rcu)
    ->ThreadRange(1, num_cpu)->Iterations(kErasePrefill/num_cpu)->UseRealTime();

DEFINE_ERASE(Erase_Small_Del,
    /* PRE */,
    /* ITERATION */
    ok += set->erase(own.get(), mix(base + next++));,
    /* POST */,
    ConcurrentSetDel, SmallErase)
[[maybe_unused]] static const bool Erase_Small_Del_registered =
    (register_shared_total<BENCHMARK_PRIVATE_CONCAT_NAME(EraseFixture, Erase_Small_Del)>(kSmallPrefill), true);

DEFINE_ERASE(Erase_Small_Del_Rcu,
    /* PRE */
    Tally tally(small_cadence(state), state);,
    /* ITERATION: refresh, erase, advance() at the cadence */
    own.refresh();
    ok += set->erase(own.get(), mix(base + next++));
    if (tally.due()) tally.add(advance_out_of_line(*set));
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    if (state.thread_index() == 0) before.report_set(state, *set);,
    ConcurrentSetDel, SmallErase)
[[maybe_unused]] static const bool Erase_Small_Del_Rcu_registered =
    (register_shared_total<BENCHMARK_PRIVATE_CONCAT_NAME(EraseFixture, Erase_Small_Del_Rcu)>(kSmallPrefill), true);

DEFINE_ERASE(Erase_Small_Del_RcuEach,
    /* PRE */
    Tally tally(small_cadence(state), state, Tally::Advancer::every_thread);,
    /* ITERATION: refresh, erase, advance() at this thread's cadence */
    own.refresh();
    ok += set->erase(own.get(), mix(base + next++));
    if (tally.due()) tally.add(advance_out_of_line(*set));
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    if (state.thread_index() == 0) before.report_set(state, *set);,
    ConcurrentSetDel, SmallErase)
[[maybe_unused]] static const bool Erase_Small_Del_RcuEach_registered =
    (register_shared_total<BENCHMARK_PRIVATE_CONCAT_NAME(EraseFixture, Erase_Small_Del_RcuEach)>(kSmallPrefill), true);

DEFINE_ERASE(Erase_Small_Del_Auto,
    /* PRE: no call; this thread's timers, read around every erase */
    Tally tally(Tally::Automatic{}, 0, state);
    Tally::Timers& timers = Tally::timers();,
    /* ITERATION: refresh, erase, the trigger detection */
    own.refresh();
    const size_t bump_before = timers.bump_ns;
    ok += set->erase(own.get(), mix(base + next++));
    if (BM_UNLIKELY(timers.bump_ns != bump_before)) tally.detected();
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    if (state.thread_index() == 0) {
        before.report_set(state, *set);
        before.report_shards(state, *set);
    },
    ConcurrentSetDel, SmallErase, SmallAutoCadence)
[[maybe_unused]] static const bool Erase_Small_Del_Auto_registered =
    (register_shared_total<BENCHMARK_PRIVATE_CONCAT_NAME(EraseFixture, Erase_Small_Del_Auto)>(kSmallPrefill), true);
