// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/LockFree
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
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include "concurrent_hash_set.h"

/*
 * ConcurrentResizableHashSet Benchmarks, v2
 *
 * This suite replaces the pool-collision benchmarks with fixed-composition
 * workloads, so that every row of a ThreadRange measures the *same* workload.
 *
 * The problem with the old suite: drawing keys from a shared random pool makes
 * the duplicate rate a function of table fill, which is a function of the
 * iteration budget, which Google Benchmark chooses differently for every
 * thread count. The rows of "Insert_MostlyNew" ranged from ~20% duplicates
 * (1 thread) to ~85% duplicates (32 threads) -- the mid-curve was mostly
 * measuring cheap duplicate-rejects (reads) wearing an insert benchmark's
 * name. Here, composition is pinned by construction:
 *
 *  - Insert_MostlyNew: every thread inserts sequential keys from its own
 *    disjoint range. Duplicate rate is exactly 0%. Every operation is a real
 *    insertion; the table walks through many doublings live, mid-measurement.
 *    Because every key is globally unique and attempted exactly once, every
 *    insert() MUST return true -- this is verified per thread, which makes the
 *    benchmark a standing regression test for the insert return-value
 *    semantics under concurrent resize (the re-insert fallback).
 *
 *  - Lookup_MostlyOld: the table is pre-populated with PREFILL keys (untimed)
 *    at full capacity, so that no resize can occur during the timed region in
 *    either container (see the caution in MostlyOldFixture::SetUp). The
 *    steady-state workload is 99% contains() with a ~50% hit rate (the
 *    lookup range is exactly twice the prefilled range, so misses exercise the
 *    negative-result revalidation path, which an all-hits workload never
 *    touches), and 1% insertion of brand-new keys on a deterministic schedule
 *    (every 100th operation), drawn from thread-disjoint ranges *above* the
 *    lookup range so the hit rate stays fixed for the whole run. The insert
 *    budget stays a factor of ~3 below the resize trigger. This benchmark
 *    measures what readers pay for coexisting with writers when nothing
 *    "happens"; what readers pay when a resize DOES happen is a tail-latency
 *    question and needs a tail-latency benchmark, not a mean.
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
 */

static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// ---------------------------------------------------------------------------
// Workload parameters
// ---------------------------------------------------------------------------
static constexpr int    kNewIters  = 1 << 18;       // per-thread insertions (MostlyNew)
static constexpr int    kOldIters  = 1 << 20;       // per-thread operations (MostlyOld)
static constexpr int    kKeyStride = 1 << 22;       // per-thread key range width
static constexpr int    kPrefill   = 1 << 20;       // keys pre-inserted for MostlyOld
static constexpr uint32_t kLookupMask = (1u << 21) - 1; // lookups in [0, 2*kPrefill): ~50% hits
static constexpr int    kNewBase   = 1 << 21;       // new keys start above the lookup range
static constexpr int    kInsertEvery = 100;         // MostlyOld: 1 insert per 100 ops

// ---------------------------------------------------------------------------
// A fast, cheap PRNG. std::mt19937 costs a nontrivial fraction of a 6 ns
// lookup; xorshift32 is a few cycles and identical for all contestants, so
// the comparison is fair and the floor is barely inflated.
// ---------------------------------------------------------------------------
struct XorShift32 {
    uint32_t s;
    explicit XorShift32(uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
};

// ---------------------------------------------------------------------------
// The baseline: unordered_set behind a reader-writer lock.
// ---------------------------------------------------------------------------
template <typename T, typename Hash = std::hash<T>>
class LockedHashSet {
    std::unordered_set<T, Hash> set_;
    mutable std::shared_mutex mtx_;
public:
    explicit LockedHashSet(size_t initial_buckets = 1024) : set_(initial_buckets) {}
    bool insert(const T& v) {
        std::unique_lock lock(mtx_);
        return set_.insert(v).second;
    }
    bool contains(const T& v) const {
        std::shared_lock lock(mtx_);              // <-- the RMW every reader pays
        return set_.find(v) != set_.end();
    }
};

using ConcurrentSet = ConcurrentResizableHashSet<int, false, std::hash<int>>;
using LockedSet     = LockedHashSet<int, std::hash<int>>;

// ---------------------------------------------------------------------------
// Fixtures.
//
// CAUTION (inherited from the v1 suite, still true): Google Benchmark runs
// SetUp() per thread with NO implicit barrier between SetUp() and the start
// of the benchmark body. The only implicit barrier is the `for (auto _ :
// state)` loop itself. Therefore: thread 0 creates (and, for MostlyOld,
// prefills) the container in its SetUp(), and no thread dereferences `set`
// anywhere except INSIDE the loop (or after it, once the loop's end barrier
// has been crossed).
// ---------------------------------------------------------------------------
template <typename SetType>
class MostlyNewFixture : public benchmark::Fixture {
public:
    static SetType* set;
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) set = new SetType(1024);
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    }
};
template <typename SetType> SetType* MostlyNewFixture<SetType>::set = nullptr;

template <typename SetType>
class MostlyOldFixture : public benchmark::Fixture {
public:
    static SetType* set;
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            /*
             * CAUTION: construct at full capacity. Steady state requires that
             * NO resize can occur, in either container, during the timed
             * region -- and for ConcurrentResizableHashSet that requirement
             * is stronger than it looks. If the table is prefilled through
             * doublings (e.g., constructed at 1024), it exits SetUp with a
             * large population of UNINITIALIZED buckets whose lazy splits
             * have not happened yet. Those splits are then performed INSIDE
             * the timed region, cooperatively, BY THE READERS -- and every
             * split allocates through the arena deque's spinlock_. The
             * nominally lock-free lookup benchmark degenerates into a
             * spinlock convoy (wall time grows with threads while CPU
             * stays flat).
             * Constructing at 2*kPrefill means the table never doubles:
             * every bucket is born EMPTY, no split ever exists, no stale
             * copies inflate the chains. The unordered_set baseline gets the
             * same courtesy (2*kPrefill buckets, so no rehash either).
             * Resize behavior under load is a tail-latency story and gets
             * its own benchmark; it has no business inside a mean.
             */
            set = new SetType(2 * kPrefill);
            for (int k = 0; k < kPrefill; ++k) set->insert(k);   // untimed
        }
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    }
};
template <typename SetType> SetType* MostlyOldFixture<SetType>::set = nullptr;

// ---------------------------------------------------------------------------
// Benchmark bodies (as macros, so each container gets an identical body).
// ---------------------------------------------------------------------------

/*
 * Insert_MostlyNew: pure insertion, zero duplicates by construction.
 * Thread t inserts kNewBase-free keys t*kKeyStride, t*kKeyStride+1, ...
 * Invariant checked per thread: every insert() returned true.
 */
#define DEFINE_MOSTLY_NEW(NAME, SET_TYPE)                                     \
    BENCHMARK_TEMPLATE_DEFINE_F(MostlyNewFixture, NAME, SET_TYPE)             \
    (benchmark::State& state) {                                               \
        const int base = state.thread_index() * kKeyStride;                   \
        int next = 0;                                                         \
        int64_t ok = 0;                                                       \
        for (auto _ : state) {                                                \
            ok += set->insert(base + next++);                                 \
        }                                                                     \
        if (ok != state.iterations()) {                                       \
            state.SkipWithError("insert() returned false for a unique key "   \
                                "(resize return-value regression)");          \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
    }                                                                         \
    BENCHMARK_REGISTER_F(MostlyNewFixture, NAME)                              \
        ->ThreadRange(1, num_cpu)->Iterations(kNewIters);

/*
 * Lookup_MostlyOld: 99% contains() at a fixed ~50% hit rate, 1% insertion of
 * brand-new keys on a deterministic schedule. New keys live above the lookup
 * range, so the workload composition is constant for the whole run.
 * Invariant checked per thread: every insert() returned true.
 */
#define DEFINE_MOSTLY_OLD(NAME, SET_TYPE)                                     \
    BENCHMARK_TEMPLATE_DEFINE_F(MostlyOldFixture, NAME, SET_TYPE)             \
    (benchmark::State& state) {                                               \
        XorShift32 rng(0x1234567u + 0x9e3779b9u * state.thread_index());      \
        const int base = kNewBase + state.thread_index() * kKeyStride;        \
        int next = 0;                                                         \
        int op = 0;                                                           \
        int64_t ok = 0, tries = 0;                                            \
        bool hit = false;                                                     \
        for (auto _ : state) {                                                \
            if (++op == kInsertEvery) {                                       \
                op = 0;                                                       \
                ++tries;                                                      \
                ok += set->insert(base + next++);                             \
            } else {                                                          \
                hit = set->contains((int)(rng.next() & kLookupMask));         \
                benchmark::DoNotOptimize(hit);                                \
            }                                                                 \
        }                                                                     \
        if (ok != tries) {                                                    \
            state.SkipWithError("insert() returned false for a unique key "   \
                                "(resize return-value regression)");          \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        state.counters["inserts"] =                                           \
            benchmark::Counter((double)tries, benchmark::Counter::kDefaults); \
    }                                                                         \
    BENCHMARK_REGISTER_F(MostlyOldFixture, NAME)                              \
        ->ThreadRange(1, num_cpu)->Iterations(kOldIters);

// ---------------------------------------------------------------------------
// Registrations: identical workloads, two containers.
// ---------------------------------------------------------------------------
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW(Insert_MostlyNew_RWLocked,   LockedSet)

DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_OLD(Lookup_MostlyOld_RWLocked,   LockedSet)

BENCHMARK_MAIN();
