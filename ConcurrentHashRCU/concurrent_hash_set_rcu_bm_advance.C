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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: advance() alone, one thread,
 * nothing else running -- the geometry regression. One of the benchmark's
 * translation units (the overview of the workloads, the index of the units
 * and why there are several: concurrent_hash_set_rcu_bm.C); the code
 * generated for these rows depends on this unit alone. The rows are plain
 * functions registered by name, not fixtures; everything they use is here.
 *
 * Table of contents, with why each is measured (one thread, one row per
 * geometry):
 *
 *  Advance_FreePass_*             advance() alone: the bump and the free pass
 *                                 per freed node, warm and cold, the nodes in
 *                                 arena order and shuffled; the source of the
 *                                 free pass's cost per node.
 *  Advance_Walk_*                 advance() alone: the walk over pre-sized
 *                                 tables whose prefill retired nothing (one
 *                                 variant retires one key per call), per
 *                                 bucket and per node passed; the source
 *                                 of the walk's cost, from which the _Rcu
 *                                 rows' cadences are chosen.
 */

// ---------------------------------------------------------------------------
// The advance() geometry regression: advance() alone, one thread, nothing else
// running.
//
// Why measure: advance() is the one call of this container whose cost grows
// with the set rather than with the operation: its free pass pushes every node
// of the bags it frees onto the free lists, and its walk passes every
// published bucket and every node in its chain. Here it runs alone, so the
// time is its own, split by the calling thread's timers into the bump, the
// free pass and the walk; the rows under the reclamation regime are read
// against these costs per node and per bucket, and their cadences are chosen
// from them.
//  Advance_FreePass_<Warm|Cold>_<ArenaOrder|Shuffled>/nodes:N
//      A set of N buckets (so that the inserts below never double it), N keys
//      inserted and then all erased, untimed, then one timed advance(): no
//      handle is live, so the call bags the N retired nodes and frees the bag
//      in the same call, and walks the N empty buckets. The order of the
//      erases is the order in which the free pass visits the nodes (it
//      follows the retired list, latest retirement first): ArenaOrder erases
//      the keys in insertion order, which is the order of their slots in the
//      arena, so the visits run through the arena backwards, one slot after
//      the other; Shuffled erases them in a fixed pseudo-random order, so the
//      visits jump. Warm: the call runs right after the erases, with whatever
//      of the nodes the caches still hold; Cold: evict_caches() runs between
//      the erases and the call. A fresh set every iteration, so that every
//      call starts from the same arena layout.
//      Counters (per call or per node, averaged over the iterations):
//      freed_per_call, bump_ns_per_node, free_ns_per_node, and
//      walk_ns_per_bucket (the walk over the N empty buckets).
//  Advance_Walk_<NothingRetired|NothingRetired_Cold|Advanced>/buckets:B/live:L
//      A set of B buckets holding L keys (untimed prefill; a pre-sized table
//      never splits, so nothing is retired), then timed advance() calls on
//      it. NothingRetired: each call finds nothing retired, consumes no
//      generation and still walks the table (AdvanceResult's
//      `nothing_retired`): what a call costs on a set where nothing was
//      erased; NothingRetired_Cold: the same with evict_caches() before every
//      call; Advanced: one key is erased (untimed) before every call and
//      inserted again after it, so every call bumps the generation, frees one
//      node and walks under the new generation: the same walk through the
//      other branch of advance().
//      Counters (per call): walk_ns, bump_ns, free_ns, buckets_walked,
//      nodes_passed; and walk_ns_per_bucket, walk_ns over B. The walk's cost
//      per bucket and per node passed is the fit of walk_ns over the rows of
//      one B at L = 0, B/2 and B.
// Every row's iteration time is the whole advance() call, from the calling
// thread's own timers (manual time: the untimed setup of each iteration is
// left out). A call that does not do what its row describes (outcome, counts)
// marks the row as an error after the loop.
// ---------------------------------------------------------------------------
static constexpr size_t kEvictBytes = size_t(256) << 20;

// Reads one word of every 64-byte line of a buffer of kEvictBytes, so that
// afterwards no line a set used is in any cache: kEvictBytes must be several
// times the last-level cache (256 MiB is four times a 64 MiB cache). The
// buffer is allocated and written on the first call.
static void evict_caches() {
    static const std::vector<uint64_t> buffer(kEvictBytes/sizeof(uint64_t), 1);
    uint64_t sum = 0;
    for (size_t i = 0; i < buffer.size(); i += 64/sizeof(uint64_t)) sum += buffer[i];
    benchmark::DoNotOptimize(sum);
} // evict_caches()

// One timed call of an Advance_ row: zeroes the calling thread's timers,
// calls advance() on `set`, gives the whole call's time (advance_ns) to Google
// Benchmark as this iteration's time, and adds the four timers into `sum`.
// Returns the call's result.
static AdvanceResult timed_advance(benchmark::State& state, ConcurrentSetDel& set, ConcurrentSetDel::InternalTimers& sum) {
    ConcurrentSetDel::InternalTimers& timers = ConcurrentSetDel::internal_advance_timers();
    timers = {0, 0, 0, 0};
    const AdvanceResult r = set.advance();
    state.SetIterationTime(double(timers.advance_ns)*1e-9);
    sum.advance_ns += timers.advance_ns;
    sum.bump_ns += timers.bump_ns;
    sum.free_ns += timers.free_ns;
    sum.walk_ns += timers.walk_ns;
    return r;
} // timed_advance()

// One row of Advance_FreePass (see above): `nodes` keys retired and freed per
// call, the caches evicted before the call if kCold, erased in a shuffled
// order if kShuffled.
template <bool kCold, bool kShuffled>
static void advance_free_pass(benchmark::State& state, size_t nodes) {
    std::vector<int> order(nodes);   // the erase order: indices of the inserted keys
    for (size_t i = 0; i < nodes; ++i) order[i] = int(i);
    if constexpr (kShuffled) {
        XorShift32 rng(0x2545f491u);
        for (size_t i = nodes - 1; i > 0; --i) std::swap(order[i], order[rng.next() % (i + 1)]);
    } // Fisher-Yates, a fixed seed
    ConcurrentSetDel::InternalTimers sum{0, 0, 0, 0};   // the timers summed over the calls
    size_t freed = 0;                                   // nodes freed, summed over the calls
    bool as_described = true;                           // every call did what the row describes
    for (auto _ : state) {
        ConcurrentSetDel set(nodes, arena_shards_from_env());
        {   // the prep's handle, released before the call
            const ConcurrentSetDel::handle h = set.new_handle();
            for (size_t i = 0; i < nodes; ++i) as_described &= set.insert(h, mix(int(i)));
            for (size_t i = 0; i < nodes; ++i) as_described &= set.erase(h, mix(order[i]));
        } // the prep's handle
        if constexpr (kCold) evict_caches();
        const AdvanceResult r = timed_advance(state, set, sum);
        as_described &= r.advance == AdvanceResult::Advance::advanced && r.freed_nodes == nodes && r.walk_retired == 0
                        && set.get_internal_table_size() == nodes;
        freed += r.freed_nodes;
    } // loop over the calls, a fresh set each
    if (!as_described) state.SkipWithError("a prep operation failed, or advance() did not free every node of the bag");
    const double calls = double(state.iterations());
    state.counters["freed_per_call"]     = benchmark::Counter(double(freed)/calls);
    state.counters["bump_ns_per_node"]   = benchmark::Counter(double(sum.bump_ns)/double(freed));
    state.counters["free_ns_per_node"]   = benchmark::Counter(double(sum.free_ns)/double(freed));
    state.counters["walk_ns_per_bucket"] = benchmark::Counter(double(sum.walk_ns)/(calls*double(nodes)));
} // advance_free_pass()

// What the timed advance() calls of an Advance_Walk row find (see above).
enum class WalkCall { NothingRetired, NothingRetiredCold, Advanced };

// One row of Advance_Walk (see above): `buckets` buckets, `live` keys.
template <WalkCall kCall>
static void advance_walk(benchmark::State& state, size_t buckets, size_t live) {
    ConcurrentSetDel set(buckets, arena_shards_from_env());
    bool as_described = true;   // every prep operation succeeded and every call did what the row describes
    {   // the prefill's handle, released before the calls
        const ConcurrentSetDel::handle h = set.new_handle();
        for (size_t k = 0; k < live; ++k) as_described &= set.insert(h, mix(int(k)));
    } // the prefill's handle
    constexpr bool kAdvanced = kCall == WalkCall::Advanced;
    const AdvanceResult::Advance expected = kAdvanced ? AdvanceResult::Advance::advanced : AdvanceResult::Advance::nothing_retired;
    ConcurrentSetDel::InternalTimers sum{0, 0, 0, 0};   // the timers summed over the calls
    size_t walked = 0, passed = 0;                      // buckets_walked and nodes_passed, summed over the calls
    for (auto _ : state) {
        if constexpr (kAdvanced) {
            const ConcurrentSetDel::handle h = set.new_handle();
            as_described &= set.erase(h, mix(0));
        } // one key retired before the call, its handle released
        if constexpr (kCall == WalkCall::NothingRetiredCold) evict_caches();
        const AdvanceResult r = timed_advance(state, set, sum);
        if constexpr (kAdvanced) {
            const ConcurrentSetDel::handle h = set.new_handle();
            as_described &= set.insert(h, mix(0));
        } // the key back, after the call
        as_described &= r.advance == expected && r.freed_nodes == (kAdvanced ? 1u : 0u) && r.walk_retired == 0
                        && r.buckets_walked == buckets && r.nodes_passed == live - (kAdvanced ? 1 : 0);
        walked += r.buckets_walked;
        passed += r.nodes_passed;
    } // loop over the calls
    if (!as_described) state.SkipWithError("a prep operation failed, or advance() did not do what the row describes");
    const double calls = double(state.iterations());
    state.counters["walk_ns"]            = benchmark::Counter(double(sum.walk_ns)/calls);
    state.counters["bump_ns"]            = benchmark::Counter(double(sum.bump_ns)/calls);
    state.counters["free_ns"]            = benchmark::Counter(double(sum.free_ns)/calls);
    state.counters["buckets_walked"]     = benchmark::Counter(double(walked)/calls);
    state.counters["nodes_passed"]       = benchmark::Counter(double(passed)/calls);
    state.counters["walk_ns_per_bucket"] = benchmark::Counter(double(sum.walk_ns)/(calls*double(buckets)));
} // advance_walk()

// Registers the regression's rows, one registration per row, each with its
// own iteration count (Google Benchmark shares one Iterations() value per
// registration), sized so that a row takes about a second or less: the free
// pass at N = 2^10 to 2^22 nodes in steps of 4x; the walk at B = 2^14 to 2^23
// buckets, which include the bucket counts of the other units' rows (2^16: the window
// rows; 2^17: the small erase rows; 2^21: the lookup rows; 2^23: the pre-sized
// rows), at L = 0, B/2 and B live keys, and the Advanced variant at L = B/2.
static void register_advance_regression() {
    struct FreePassRow { const char* name; void (*fn)(benchmark::State&, size_t); };
    static constexpr FreePassRow free_rows[] = {
        {"Advance_FreePass_Warm_ArenaOrder", advance_free_pass<false, false>},
        {"Advance_FreePass_Cold_ArenaOrder", advance_free_pass<true, false>},
        {"Advance_FreePass_Warm_Shuffled",   advance_free_pass<false, true>},
        {"Advance_FreePass_Cold_Shuffled",   advance_free_pass<true, true>},
    };
    for (const FreePassRow& row : free_rows) {
        for (int lg = 10; lg <= 22; lg += 2) {
            const size_t nodes = size_t(1) << lg;
            const benchmark::IterationCount iters = std::clamp<benchmark::IterationCount>(benchmark::IterationCount(1 << 22)/benchmark::IterationCount(nodes), 3, 200);
            benchmark::RegisterBenchmark(std::string(row.name) + "/nodes:" + std::to_string(nodes), row.fn, nodes)
                ->Iterations(iters)->UseManualTime();
        } // loop over the node counts
    } // loop over the free-pass variants
    struct WalkRow { const char* name; void (*fn)(benchmark::State&, size_t, size_t); bool half_only; };
    static constexpr WalkRow walk_rows[] = {
        {"Advance_Walk_NothingRetired",      advance_walk<WalkCall::NothingRetired>,     false},
        {"Advance_Walk_NothingRetired_Cold", advance_walk<WalkCall::NothingRetiredCold>, false},
        {"Advance_Walk_Advanced",            advance_walk<WalkCall::Advanced>,           true},
    };
    static constexpr int bucket_logs[] = {14, 16, 17, 18, 20, 21, 22, 23};
    for (const WalkRow& row : walk_rows) {
        for (int lg : bucket_logs) {
            const size_t buckets = size_t(1) << lg;
            for (size_t live : {size_t(0), buckets/2, buckets}) {
                if (row.half_only && live != buckets/2) continue;
                const benchmark::IterationCount iters = std::clamp<benchmark::IterationCount>(benchmark::IterationCount(1 << 25)/benchmark::IterationCount(buckets + live), 3, 100);
                benchmark::RegisterBenchmark(std::string(row.name) + "/buckets:" + std::to_string(buckets) + "/live:" + std::to_string(live),
                                             row.fn, buckets, live)
                    ->Iterations(iters)->UseManualTime();
            } // loop over the populations
        } // loop over the bucket counts
    } // loop over the walk variants
} // register_advance_regression()

// The advance() geometry regression (see above): the last rows of the
// benchmark, since this is the last unit linked (the Makefile's BM_SOURCES).
[[maybe_unused]] static const bool advance_regression_registered = (register_advance_regression(), true);
