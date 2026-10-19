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
#include <cstdint>
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: growth on hot chains, a probe of
 * the split's cleanup pass. One of the benchmark's translation units (the
 * overview of the workloads, the index of the units and why there are
 * several: concurrent_hash_set_rcu_bm.C); the code generated for these rows
 * depends on this unit alone. The hot-growth keys, fixture, counters and body
 * are here.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  InsertHotGrowth_Concurrent     growth on a few hot chains, split while the
 *                                 threads still prepend to them: the cleanup
 *                                 pass's lost CASes and the stragglers they
 *                                 leave, counted; the throughput is not the
 *                                 point.
 *  InsertHotGrowth_Del            the same probe with erase() compiled in.
 */

// ---------------------------------------------------------------------------
// InsertHotGrowth: growth on hot chains, a probe of the split's cleanup pass.
//
// Why measure: the growth cells (the _vs_baseline and _states_insert units)
// mix() every key, so their chains are a few nodes long at every doubling and
// a split's cleanup pass hardly ever meets a writer on the parent chain it
// walks. This cell builds the opposite
// case: every key falls into one of 64 chains while the table has at most
// 4096 buckets, and the two doublings after that (to 8192 and to 16384
// buckets) split chains of that length while the threads are still
// prepending to them -- the one place where a cleanup pass's single unlink
// CAS per dead run loses to a concurrent insert, and where a lost CAS can
// leave a frozen node in its chain for a later writer walk or advance()'s walk
// to collect.
// The counters, all read by thread 0 after the loop (the loop's end is a
// barrier, so the container is quiescent and its diagnostic accessors are
// exact), untimed, are what a decision to retry the cleanup CAS would rest
// on; the throughput of this cell is not a headline, its chains are long by
// construction.
//   moved_nodes        -- nodes the splits copied into child buckets, counted
//                         as the frozen originals they left behind: every
//                         arena slot that is not a live key, not on a free
//                         list and not in limbo. The denominator of the
//                         per-node counters below. It is the same quantity on
//                         any container that keeps this accounting, whether
//                         the frozen originals are still in their chains or
//                         were unlinked and retired (reachable minus the key
//                         count would count only the former).
//   limbo_nodes        -- the subchains of lost publishing CASes.
//   arena_nodes_per_key, consistent -- as in the insert cells, and the
//                         accounting sweep's verdict.
// The diagnostic counters of this container (hot_growth_counters() below):
//   table_size, splits_published, split_attempts, moved_per_split -- the
//                         construction: the final bucket count, the splits
//                         and how many nodes each moved;
//   cleanup_cas_lost_per_moved    -- lost cleanup CASes per moved node, of
//                         every kind: the contention on the predecessor word;
//   cleanup_cas_nonpeer_per_moved -- of those, the losses NOT to a peer that
//                         had already bypassed the same run: a peer loss
//                         leaves nothing behind, so this is an upper bound on
//                         the stragglers the cleanup passes produced, and the
//                         number a retry would act on;
//   walk_cas_lost_per_moved, walk_cas_nonpeer_per_moved -- the same two for
//                         the unlinks the inserts' walks attempt;
//   reachable_dead     -- frozen nodes still in their chains after the join
//                         (the stragglers; the residue a later writer walk
//                         or advance()'s walk collects);
//   retired            -- frozen nodes the cleanup passes and the walks did
//                         unlink and retire.
// Keys: the key of index i is (m & 63) | (((m >> 6) & 7) << 12) | (i << 16)
// with m = mix(i): a hot pattern of 64 values in bits 0..5, bits 6..11 zero,
// three mobile bits at 12..14 (the doublings to 8192 and 16384 buckets split
// the chains by bits 12 and 13) and the index above bit 16, so every key is
// distinct and positive. Thread t of T inserts the indices t, t + T, t + 2T,
// ... (interleaved, so that every thread prepends to every chain throughout).
// The same keys at every thread count: the run inserts kHotKeys keys in all,
// kHotKeys/T per thread. ThreadRange() shares one Iterations() value among
// its thread counts, so the rows are registered one thread count at a time
// (register_shared_total()), each with its own per-thread count; the row names
// carry it as /iterations:N. The initial bucket count is the growth
// fixture's; the doublings before 4096 buckets split the 64 chains but move
// nothing. The cell is registered on both instantiations, as the insert cells
// of the other units are.
// ---------------------------------------------------------------------------
static constexpr int kHotKeys = 1 << 14;
static_assert(kHotKeys <= (1 << 14), "the index must fit above bit 16 of a positive int");

// The key of index `index` of the hot-growth construction (see above).
static inline int hot_growth_key(int index) {
    const uint32_t m = static_cast<uint32_t>(mix(index));
    return static_cast<int>((m & 63) | (((m >> 6) & 7) << 12) | (static_cast<uint32_t>(index) << 16));
} // hot_growth_key()

template <typename SetType>
class HotGrowthFixture : public benchmark::Fixture {
public:
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles, shared header)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            set = new_set<SetType>(1024);
            handles.make(*set, state.threads());
        }
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class HotGrowthFixture
template <typename SetType> SetType* HotGrowthFixture<SetType>::set = nullptr;
template <typename SetType> ThreadHandles<SetType> HotGrowthFixture<SetType>::handles;

// The counters of InsertHotGrowth that only this container's diagnostic
// accessors provide (listed above), set on `state` by thread 0 after the loop;
// `a` is the accounting sweep the body already took, `moved` its moved-node
// count. Kept apart from the body so that the body is the same for any
// container with the accounting sweep. Per-node rates are 0 when nothing
// moved (a run that moved nothing has no rate to report).
template <typename SetType>
static void hot_growth_counters(benchmark::State& state, const SetType& set,
                                const typename SetType::InternalAccounting& a, double moved) {
    const typename SetType::InternalCounters c = set.get_internal_counters();
    const double per_moved = moved > 0 ? 1/moved : 0;
    state.counters["table_size"]       = benchmark::Counter(double(set.get_internal_table_size()));
    state.counters["splits_published"] = benchmark::Counter(double(c.splits_published));
    state.counters["split_attempts"]   = benchmark::Counter(double(c.split_attempts));
    state.counters["moved_per_split"]  = benchmark::Counter(c.splits_published ? moved/double(c.splits_published) : 0);
    state.counters["cleanup_cas_lost_per_moved"]    = benchmark::Counter(double(c.cleanup_cas_failures)*per_moved);
    state.counters["cleanup_cas_nonpeer_per_moved"] = benchmark::Counter(double(c.cleanup_cas_failures - c.cleanup_cas_peer_losses)*per_moved);
    state.counters["walk_cas_lost_per_moved"]       = benchmark::Counter(double(c.walk_cas_failures)*per_moved);
    state.counters["walk_cas_nonpeer_per_moved"]    = benchmark::Counter(double(c.walk_cas_failures - c.walk_cas_peer_losses)*per_moved);
    state.counters["reachable_dead"] = benchmark::Counter(double(a.reachable_dead));
    state.counters["retired"]        = benchmark::Counter(double(a.retired));
} // hot_growth_counters()

// The body (see InsertHotGrowth above) and its registration, as a macro so
// that each instantiation gets an identical body. Same accounting contract as
// the other bodies: allocate() before the loop; report_wall() then release()
// after it; the per-thread SkipWithError() after the loop, never inside it.
#define DEFINE_HOT_GROWTH(NAME, SET_TYPE)                                     \
    BENCHMARK_TEMPLATE_DEFINE_F(HotGrowthFixture, NAME, SET_TYPE)             \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        const int threads = state.threads();                                  \
        int index = state.thread_index();                                     \
        int64_t ok = 0;                                                       \
        Handles::Own own(handles, state);                                     \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            ok += set->insert(own.get(), hot_growth_key(index));              \
            index += threads;                                                 \
        }                                                                     \
        if (ok != state.iterations()) {                                       \
            state.SkipWithError("insert() returned false for a unique key "   \
                                "(resize return-value regression)");          \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        const double keys = double(state.iterations())*threads;              \
        report_wall(state, records, keys);                                    \
        if (state.thread_index() == 0) {                                      \
            const SET_TYPE::InternalAccounting a = set->get_internal_accounting(); \
            const double moved = double(a.slots - a.free_nodes - a.limbo) - keys; \
            state.counters["moved_nodes"] = benchmark::Counter(moved);        \
            state.counters["limbo_nodes"] = benchmark::Counter(double(a.limbo)); \
            state.counters["arena_nodes_per_key"] = benchmark::Counter(double(a.slots)/keys); \
            state.counters["consistent"] = benchmark::Counter(double(a.consistent)); \
            hot_growth_counters(state, *set, a, moved);                       \
        }                                                                     \
        records.release(state);                                               \
    }                                                                         \
    [[maybe_unused]] static const bool NAME##_registered =                    \
        (register_shared_total<BENCHMARK_PRIVATE_CONCAT_NAME(HotGrowthFixture, NAME)>(kHotKeys), true);

// The hot-growth probe (see InsertHotGrowth above), on both instantiations.
DEFINE_HOT_GROWTH(InsertHotGrowth_Concurrent, ConcurrentSet)
DEFINE_HOT_GROWTH(InsertHotGrowth_Del,        ConcurrentSetDel)
