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
#include <string>
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: the index, and main().
 *
 * The benchmark is several translation units linked into one binary, one per
 * family of rows, over the shared header concurrent_hash_set_rcu_bm.h (the
 * helpers, the fixtures of the insert and lookup workloads and their macro
 * bodies). This unit registers no row: it holds the overview of the workloads
 * below, the index of which unit holds which rows, and main().
 *
 * Why several units: a compiler decides what to inline from the whole
 * translation unit (GCC's inliner, for one, grants a unit a growth budget and
 * stops inlining small functions once it is spent), so with every row in one
 * unit the code of a row's timed loop -- whether the set's insert() is inlined
 * into it, which loop counters stay in registers -- depends on how much code
 * the OTHER rows and their fixtures' prefills have already cost, and a change
 * in one row can change the code of rows that do not contain it. With a unit
 * per family, the code generated for a row depends only on its own unit: its fixtures, its
 * prefills, its bodies, and what they instantiate from the headers. The row
 * names are the same as with one unit (a fixture's name and the row's), and
 * the rows are listed and run in the order of the index below: Google
 * Benchmark runs the rows in registration order, which is the order of the
 * static initializers -- within a unit the order in its file, across units
 * the link order, which the Makefile keeps as the index has it.
 *
 * Index: the units, in link order, and their rows (each unit opens with why
 * its rows are measured; the details are at each definition):
 *
 *  concurrent_hash_set_rcu_bm_vs_baseline.C
 *      the insert and lookup workloads on the two containers:
 *      Insert_MostlyNew_Concurrent, Insert_MostlyNew_RWLocked,
 *      Insert_MostlyNew_Presized_Concurrent, Insert_MostlyNew_Presized_RWLocked,
 *      Lookup_MostlyOld_Concurrent, Lookup_MostlyOld_RWLocked,
 *      Lookup_MostlyOld_OpHandle
 *  concurrent_hash_set_rcu_bm_states_insert.C
 *      the insert workloads on the AllowDelete set empty, prefilled, and
 *      prefilled by churn (the states Control, Reclaimed, Erased):
 *      Insert_MostlyNew_Del, Insert_MostlyNew_Del_Control,
 *      Insert_MostlyNew_Del_Reclaimed, Insert_MostlyNew_Presized_Del,
 *      Insert_MostlyNew_Presized_Del_Control,
 *      Insert_MostlyNew_Presized_Del_Reclaimed,
 *      Insert_MostlyNew_Presized_Del_Erased
 *  concurrent_hash_set_rcu_bm_states_lookup.C
 *      the lookup workload on the same states:
 *      Lookup_MostlyOld_Del, Lookup_MostlyOld_Del_Control,
 *      Lookup_MostlyOld_Del_Reclaimed, Lookup_MostlyOld_Del_Erased
 *  concurrent_hash_set_rcu_bm_erase.C
 *      timed erase(), alone and under reclamation:
 *      Erase_Presized_Del, Erase_Presized_Del_Rcu, Erase_Small_Del,
 *      Erase_Small_Del_Rcu, Erase_Small_Del_RcuEach, Erase_Small_Del_Auto
 *  concurrent_hash_set_rcu_bm_churn.C
 *      steady churn, without and with reclamation:
 *      Churn_Window_Del_NoReclaim, Churn_Window_Del_Rcu,
 *      Churn_Window_Del_RcuEach, Churn_Window_Del_Auto
 *  concurrent_hash_set_rcu_bm_hot_growth.C
 *      the split's cleanup pass on hot chains:
 *      InsertHotGrowth_Concurrent, InsertHotGrowth_Del
 *  concurrent_hash_set_rcu_bm_handles.C
 *      the handle's own costs: the join and the leave alone, and the free
 *      pass's check of the handle counts under concurrent joins:
 *      Handle_JoinLeave, Reclaim_FreePassCheck
 *  concurrent_hash_set_rcu_bm_advance.C
 *      advance() alone, one thread:
 *      Advance_FreePass_*, Advance_Walk_*
 *
 * Every registration is run at the thread counts of ThreadRange(1, num_cpu)
 * (one row per thread count, all rows of a registration the same workload)
 * except the Advance_ rows (one thread, one row per geometry) and
 * Reclaim_FreePassCheck (one measured thread beside helper threads, one row
 * per argument set). A bucket count chosen so that a table does not resize
 * holds only up to some thread count: the pre-sized concurrent table doubles
 * once its arena holds about 2*kPresizedBuckets nodes (64 threads' worth of
 * timed inserts, fewer after a prefill); the lookup table once it holds about
 * 4*kPrefill, which no thread count up to 256 reaches except in the Erased
 * cell, whose arena keeps the erased victims' slots (2*kPrefill in all after
 * the prefill), so its table doubles past about 200 threads. A baseline
 * std::unordered_set rehashes once it holds more keys than buckets: never at
 * 32 threads or fewer in the pre-sized cell, 100 or fewer in the lookup cell.
 *
 * THE WORKLOADS. Fixed-composition workloads, so that every row of a
 * ThreadRange measures the *same* workload.
 *
 * Why fixed: drawing keys from a shared random pool makes the duplicate rate a
 * function of table fill, which is a function of the iteration budget, which
 * Google Benchmark chooses differently for every thread count. An insert
 * benchmark built that way sees mostly new keys at one thread and mostly
 * duplicates at many -- its mid-curve measures cheap duplicate-rejects (reads)
 * wearing an insert benchmark's name. Here, composition is pinned by
 * construction:
 *
 *  - Insert_MostlyNew: every thread inserts sequential keys from its own
 *    disjoint range. Duplicate rate is exactly 0%. Every operation is a real
 *    insertion; the table walks through many doublings live, mid-measurement.
 *    Every key is passed through mix() first (see the header): with the
 *    identity std::hash<int> and unmixed keys, thread t's n-th key
 *    t*kKeyStride + n lands in bucket n for every t while the table has fewer
 *    than kKeyStride buckets, so all threads walk and CAS the SAME bucket at
 *    the same moment and every chain is T nodes long -- an O(T) cost per
 *    insert that would be measured as the hash table's scaling.
 *    Because every key is globally unique and attempted exactly once, every
 *    insert() MUST return true -- this is verified per thread, which makes the
 *    benchmark a standing regression test for the insert return-value
 *    semantics under concurrent resize (the seal on the bucket head).
 *
 *  - Lookup_MostlyOld: the table is pre-populated with PREFILL keys (untimed)
 *    at full capacity, so that no resize occurs during the timed region in
 *    either container up to the thread counts given above (see also the
 *    caution in MostlyOldFixture::SetUp). The steady-state workload is 99%
 *    contains() with a ~50% hit rate (the lookup range is exactly twice the
 *    prefilled range, so misses exercise the negative-result revalidation
 *    path, which an all-hits workload never touches), and 1% insertion of
 *    brand-new keys on a deterministic schedule (every 100th operation),
 *    drawn from thread-disjoint ranges *above* the lookup range so the hit
 *    rate stays fixed for the whole run. The insert budget stays below either
 *    container's resize trigger up to the thread counts given above. This
 *    benchmark measures what readers pay for coexisting with writers when
 *    nothing "happens"; what readers pay when a resize DOES happen is a
 *    tail-latency question and needs a tail-latency benchmark, not a mean.
 *
 * The baseline, LockedHashSet, is std::unordered_set behind a
 * std::shared_mutex: shared_lock for readers, unique_lock for writers. This
 * is the textbook answer to "how do we let the table rehash safely" -- and it
 * illustrates the fundamental tax of that answer: to be safe against a
 * rehash, every reader must make itself VISIBLE to the writer, and visibility
 * costs an atomic RMW on a shared cache line. Every reader pays it on every
 * lookup, forever, whether or not a rehash ever happens. Industrial-strength
 * shared mutexes distribute the reader count across cores to avoid the
 * ping-pong, but that is itself a nontrivial concurrent data structure with
 * its own trade-offs; the plain shared_mutex is the honest pedagogical
 * baseline for the register-vs-validate comparison.
 *
 * Fixed iteration counts (->Iterations) are used everywhere so that the table
 * trajectory is identical across thread counts and across containers. Scale
 * kNewIters / kOldIters to taste; keep kKeyStride >= kNewIters and
 * (max_threads * kKeyStride + kNewBase) within int range.
 *
 * Every registration but the Advance_ rows and Reclaim_FreePassCheck
 * measures real time (->UseRealTime(); the row names end in
 * /real_time/threads:N), so the rates are over real time: the default CPU
 * time leaves out the time a thread spends blocked -- in the baseline's
 * std::shared_mutex, in the back-off sleeps of the arena's SpinLock -- which
 * is part of what each container costs. Those two measure manual time (their
 * row names carry /manual_time; an iteration's time is the one call it
 * measures, from the set's own timers) and report times per call instead of
 * rates. Every real-time row reports two rates and a fairness measure
 * (../Spinlock/gb_wall_clock.h defines them and does the stamping):
 *   items_per_second      -- Google Benchmark's own: the operations of all
 *                            threads over the MEAN of the threads' loop times;
 *   wall_items_per_second -- the same total over the wall-clock span of the
 *                            run, from the end of the earliest thread's first
 *                            operation to the end of the latest thread's last;
 *   finish_spread         -- (latest finish - earliest finish)/span, 0 when
 *                            all threads finish together.
 * With fixed iteration counts every thread runs the same number of operations,
 * so a thread that gets through them faster waits for the others at the end
 * of the loop, the mean loop time is shorter than the run, and
 * items_per_second overstates the throughput by more the less evenly the
 * threads progress; the wall rate is the throughput the run delivered.
 *
 * RECLAMATION. Every operation of this container runs under a handle, and
 * nodes that erases and splits retire are made reusable only by advance(),
 * reclaim() (advance() without its walk) or, on a set constructed with a
 * cadence, the set's own trigger (the header's HANDLES AND GENERATIONS).
 * advance(), which the caller runs, closes the current generation's retired
 * nodes into a bag, frees every bag no live handle can observe onto the free
 * lists, and walks every published chain, unlinking the dead nodes it passes.
 * In every row but the ones named next, each thread has one long-lived handle,
 * never refreshed, and nothing in the timed region calls advance() (see
 * ThreadHandles in the benchmark header). Lookup_MostlyOld_OpHandle takes a
 * handle of its own around every operation (its long-lived handles exist,
 * unused), Handle_JoinLeave's iterations are a join and a leave and nothing
 * else, and Reclaim_FreePassCheck's one measured thread calls reclaim() once
 * per iteration while helper threads join and leave
 * (concurrent_hash_set_rcu_bm_handles.C). The rows whose names end in _Rcu
 * run under the RECLAMATION REGIME: advance() is called from inside the timed
 * loop -- once per thread at its end, or by one thread at a cadence while
 * every thread refreshes its handle each iteration -- and the rows report
 * what the calls did and what they cost (see AdvanceTally there). The rows
 * whose names end in _RcuEach have EVERY thread call advance() at a cadence
 * of its own erases; the rows whose names end in _Auto call nothing: their
 * set is constructed with an AUTOMATIC CADENCE (the header's Options), so the
 * set itself runs the bump and the free pass -- never the walk -- from inside
 * the retiring thread's erase, once that thread's arena shard has retired
 * another R nodes; every thread still refreshes its handle each iteration.
 * Every reclamation row reports the set's own counters of what the calls and
 * the triggers did, as deltas across the timed loop (ReclaimSnapshot), and
 * the three churn reclamation rows print a per-thread time series of every
 * call or trigger return (the "# trace" lines, Trace in the churn unit). The
 * Advance_ rows measure advance() alone, one thread, nothing else running.
 */

// BENCHMARK_MAIN()'s start-up, including its re-exec without ASLR, less its
// guard against a null argv (argv is never null in a hosted program), plus
// the identification every output carries (the console header and the JSON
// "context"): the container this binary measures, the size of its Node (the
// node layout is part of what is measured), and the compiler.
int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::AddCustomContext("container", "ConcurrentResizableHashSetRCU");
    benchmark::AddCustomContext("sizeof_node", std::to_string(sizeof(ConcurrentSetDel::Node)));
    benchmark::AddCustomContext("compiler", __VERSION__);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
} // main()
