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
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include "concurrent_hash_set.h"
#include "gb_wall_clock.h"      // WallRecords, WallTimed, report_wall() (../Spinlock)

/*
 * ConcurrentResizableHashSet Benchmarks
 *
 * Fixed-composition workloads, so that every row of a ThreadRange measures the
 * *same* workload.
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
 *    Every key is passed through mix() first (see there): with the identity
 *    std::hash<int> and unmixed keys, thread t's n-th key t*kKeyStride + n
 *    lands in bucket n for every t while the table has fewer than kKeyStride
 *    buckets, so all threads walk and CAS the SAME bucket at the same moment
 *    and every chain is T nodes long -- an O(T) cost per insert that would be
 *    measured as the hash table's scaling.
 *    Because every key is globally unique and attempted exactly once, every
 *    insert() MUST return true -- this is verified per thread, which makes the
 *    benchmark a standing regression test for the insert return-value
 *    semantics under concurrent resize (the seal on the bucket head).
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
 *
 * Every registration measures real time (->UseRealTime(); the row names end
 * in /real_time/threads:N), so the rates are over real time: the default CPU
 * time leaves out the time a thread spends blocked -- in the baseline's
 * std::shared_mutex, in the back-off sleeps of the arena's SpinLock -- which
 * is part of what each container costs. Every row reports two rates and a
 * fairness measure (../Spinlock/gb_wall_clock.h defines them and does the
 * stamping):
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
 */

static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// Bijective key mixer (the MurmurHash3 32-bit finalizer): every key that enters
// a set, and every key looked up, goes through it, so a workload written in
// terms of simple integers (sequential per thread, random in a range) reaches
// the identity std::hash<int> as random-looking values and different threads'
// keys spread over different buckets. A bijection keeps keys unique and hit
// rates unchanged. The result may be negative as an int; that is a valid key.
static inline int mix(int k) {
    uint32_t x = static_cast<uint32_t>(k);
    x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16;
    return static_cast<int>(x);
}

// Arena shard count for ConcurrentResizableHashSet, from the environment:
// HASH_ARENA_SHARDS=<n> (0 or unset = the header's default, the hardware
// concurrency rounded up to a power of two). Exists so one binary can measure
// how many shards a machine with many cores actually needs: the same
// workload is run with 16 shards and with one per hardware thread.
static size_t arena_shards_from_env() {
    const char* s = getenv("HASH_ARENA_SHARDS");
    long n = s ? atol(s) : 0;
    if (n < 0) n = 0;                  // nonsense means "the default"
    if (n > (1L << 16)) n = 1L << 16;  // the header's own cap
    return static_cast<size_t>(n);
}

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
// A fast, cheap PRNG. std::mt19937 costs a nontrivial fraction of a lookup of
// a few nanoseconds; xorshift32 is a few cycles and identical for all
// contestants, so the comparison is fair and the floor is barely inflated.
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
    // The second parameter mirrors ConcurrentResizableHashSet's arena_shards so
    // the fixtures can construct both containers the same way; it is ignored.
    explicit LockedHashSet(size_t initial_buckets = 1024, size_t /*arena_shards*/ = 0) : set_(initial_buckets) {}
    // Mirrors the concurrent set's test accessor for the arena_nodes_per_key
    // counter: the baseline has one node per key, so this is the key count.
    size_t get_internal_node_count() const { return set_.size(); }
    bool insert(const T& v) {
        std::unique_lock lock(mtx_);
        return set_.insert(v).second;
    }
    bool contains(const T& v) const {
        std::shared_lock lock(mtx_);              // <-- the RMW every reader pays
        return set_.find(v) != set_.end();
    }
}; // class LockedHashSet

using ConcurrentSet = ConcurrentResizableHashSet<int, false, std::hash<int>>;
using LockedSet     = LockedHashSet<int, std::hash<int>>;

// ---------------------------------------------------------------------------
// Fixtures.
//
// CAUTION: Google Benchmark runs SetUp() per thread with NO implicit barrier
// between SetUp() and the start of the benchmark body. The only implicit
// barrier is the benchmark loop itself. Therefore: thread 0
// creates (and, for MostlyOld, prefills) the container in its SetUp(), and no
// thread dereferences `set` anywhere except INSIDE the loop (or after it, once
// the loop's end barrier has been crossed).
// ---------------------------------------------------------------------------
template <typename SetType>
class MostlyNewFixture : public benchmark::Fixture {
public:
    static SetType* set;
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) set = new SetType(1024, arena_shards_from_env());
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    }
}; // class MostlyNewFixture
template <typename SetType> SetType* MostlyNewFixture<SetType>::set = nullptr;

// The same insert workload into a table constructed with kPresizedBuckets
// buckets, so that no doubling and no lazy split happens until the arena holds
// 2*kPresizedBuckets nodes (16.7M: equal to 64 threads' worth of kNewIters). It
// separates the allocator's own cost from the doubling and split path: with the
// table geometry fixed, the arena (its lock or its shards) is the only shared
// state on the insert path, and this is the fixture the README's allocator
// numbers come from. Every insert is still a random bucket-head miss.
static constexpr size_t kPresizedBuckets = size_t(1) << 23;
template <typename SetType>
class MostlyNewPresizedFixture : public benchmark::Fixture {
public:
    static SetType* set;
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) set = new SetType(kPresizedBuckets, arena_shards_from_env());
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    }
}; // class MostlyNewPresizedFixture
template <typename SetType> SetType* MostlyNewPresizedFixture<SetType>::set = nullptr;

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
             * split allocates through its arena shard's spinlock_. The
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
            set = new SetType(2*kPrefill, arena_shards_from_env());
            for (int k = 0; k < kPrefill; ++k) set->insert(mix(k));   // untimed
        }
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    }
}; // class MostlyOldFixture
template <typename SetType> SetType* MostlyOldFixture<SetType>::set = nullptr;

// ---------------------------------------------------------------------------
// Benchmark bodies (as macros, so each container gets an identical body).
//
// Each body keeps its own wall-clock records, a function-local static (one per
// instantiation), rather than one per fixture: one placement then serves all
// four fixture types (MostlyNewFixture, MostlyNewPresizedFixture,
// MostlyOldFixture, ChurnFixture). The calls follow
// gb_wall_clock.h's contract: allocate() before the loop and, after it,
// report_wall() then release(), each called on every thread and doing its work
// on thread 0 alone; state.iterations() is read only after the loop, and the
// per-thread SkipWithError() is called after the loop, never inside it. No
// object with a destructor is live across the loop: its cleanup region can
// change the loop's code.
// ---------------------------------------------------------------------------

/*
 * Insert_MostlyNew: pure insertion, zero duplicates by construction.
 * Thread t inserts mix() of the kNewBase-free keys t*kKeyStride, t*kKeyStride+1, ...
 * Invariant checked per thread: every insert() returned true.
 */
#define DEFINE_MOSTLY_NEW(NAME, SET_TYPE) DEFINE_MOSTLY_NEW_ON(MostlyNewFixture, NAME, SET_TYPE)
#define DEFINE_MOSTLY_NEW_ON(FIXTURE, NAME, SET_TYPE)                         \
    BENCHMARK_TEMPLATE_DEFINE_F(FIXTURE, NAME, SET_TYPE)                      \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        const int base = state.thread_index()*kKeyStride;                     \
        int next = 0;                                                         \
        int64_t ok = 0;                                                       \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            ok += set->insert(mix(base + next++));                            \
        }                                                                     \
        if (ok != state.iterations()) {                                       \
            state.SkipWithError("insert() returned false for a unique key "   \
                                "(resize return-value regression)");          \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        report_wall(state, records,                                           \
                    double(state.iterations())*state.threads());              \
        /* Arena nodes per inserted key, read by thread 0 once every thread */ \
        /* has left the loop (its end is a barrier): 1.0 means no waste;    */ \
        /* above it are split copies and subchains abandoned by splitters   */ \
        /* that lost the publishing CAS. Measures how redundant the         */ \
        /* cooperative splits get as the thread count grows.                */ \
        if (state.thread_index() == 0) {                                      \
            state.counters["arena_nodes_per_key"] = benchmark::Counter(       \
                double(set->get_internal_node_count())/                       \
                (double(state.iterations())*state.threads()));                \
        }                                                                     \
        records.release(state);                                               \
    }                                                                         \
    BENCHMARK_REGISTER_F(FIXTURE, NAME)                                       \
        ->ThreadRange(1, num_cpu)->Iterations(kNewIters)->UseRealTime();

/*
 * Lookup_MostlyOld: 99% contains() at a fixed ~50% hit rate, 1% insertion of
 * brand-new keys on a deterministic schedule. New keys live above the lookup
 * range, so the workload composition is constant for the whole run.
 * Invariant checked per thread: every insert() returned true.
 */
#define DEFINE_MOSTLY_OLD(NAME, SET_TYPE) DEFINE_MOSTLY_OLD_ON(MostlyOldFixture, NAME, SET_TYPE)
#define DEFINE_MOSTLY_OLD_ON(FIXTURE, NAME, SET_TYPE)                         \
    BENCHMARK_TEMPLATE_DEFINE_F(FIXTURE, NAME, SET_TYPE)                      \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        XorShift32 rng(0x1234567u + 0x9e3779b9u*state.thread_index());        \
        const int base = kNewBase + state.thread_index()*kKeyStride;          \
        int next = 0;                                                         \
        int op = 0;                                                           \
        int64_t ok = 0, tries = 0;                                            \
        bool hit = false;                                                     \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            if (++op == kInsertEvery) {                                       \
                op = 0;                                                       \
                ++tries;                                                      \
                ok += set->insert(mix(base + next++));                        \
            } else {                                                          \
                hit = set->contains(mix((int)(rng.next() & kLookupMask)));    \
                benchmark::DoNotOptimize(hit);                                \
            }                                                                 \
        }                                                                     \
        if (ok != tries) {                                                    \
            state.SkipWithError("insert() returned false for a unique key "   \
                                "(resize return-value regression)");          \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        report_wall(state, records,                                           \
                    double(state.iterations())*state.threads());              \
        state.counters["inserts"] =                                           \
            benchmark::Counter((double)tries, benchmark::Counter::kDefaults); \
        records.release(state);                                               \
    }                                                                         \
    BENCHMARK_REGISTER_F(FIXTURE, NAME)                                       \
        ->ThreadRange(1, num_cpu)->Iterations(kOldIters)->UseRealTime();

// ---------------------------------------------------------------------------
// Reclamation benchmarks.
//
// The three workloads above are re-run on the AllowDelete == true instantiation
// in three states of the set, and every cell has a purpose:
//
//   *_Del            the workload above, on an EMPTY AllowDelete == true set (the
//                    MostlyOld fixture prefills as before). Against the
//                    AllowDelete == false rows this is the price of compiling
//                    erase() in: the FROZEN handling in the split, the mark
//                    tests on the read paths. Nothing is erased.
//   *_Del_Control    the same workload on a set PREFILLED (untimed, by thread 0)
//                    with a population of live keys and nothing else: no erase,
//                    no reclaim(), every free list empty, so every allocation
//                    in the timed region APPENDS to a deque.
//   *_Del_Reclaimed  the same workload on a set that reached the same live
//                    population by CHURN: twice as many keys were inserted,
//                    interleaved (survivor, victim, survivor, victim, ... in
//                    arena order), every victim was erased, and reclaim() ran.
//                    The live keys and the timed workload are those of the
//                    Control; what differs is the state of the arena: the
//                    victims' slots (and, in the growth fixture, the split
//                    copies the prefill made) sit on the shards' free lists,
//                    dealt round-robin, and the survivors occupy every other
//                    slot of the deque blocks thread 0 filled. The timed
//                    allocations POP from the calling thread's shard's list
//                    until it is empty and then append -- the mixed regime.
//
// Control vs Del separates "a resident population" from "AllowDelete"; Reclaimed
// vs Control is the free lists in use: the pop path in alloc_node() (one CAS on
// the shard's free head instead of the deque's lock and cursor) and the locality
// of the popped slots. reclaim() deals the freed nodes round-robin over the
// shards in the order it finds them (the limbo lists first, then the dead nodes
// of its walk over the buckets), and each free list is LIFO, so a shard pops
// them in reverse dealing order. Bucket order bears no relation to slot order
// (every key goes through mix()), so consecutive pops land on scattered slots --
// about one new cache line per popped node, where an append fills a line with
// four consecutive nodes. Why measure: the free lists are the point of
// reclaim(), and these are the only cells that measure an insert that reuses a
// slot, or a lookup over chains that reclaim() relinked.
//
// The pre-sized and lookup cells share their bucket count between Control and
// Reclaimed (no doubling in either prefill), so there the free lists are the
// only difference. The GROWTH cells do not: the Reclaimed prefill inserts twice
// the keys through the same doublings, so its table is one doubling larger
// (2^22 buckets against the Control's 2^21 with 16 shards), with a
// different population of pending lazy splits, which the timed inserts then
// perform and which allocate their copies from the same free lists; and its
// node_count_ restarts from the live count, so its next doubling comes later.
// The growth Reclaimed row is therefore "churn, then growth" as a workload,
// not a controlled measurement of the allocator; read the pre-sized row for
// that. Neither prefill can be made to end at the other's geometry with the
// same live keys: the doubling threshold is a slot count and the prefills
// differ by 2x in slots.
//
// WHEN THE TIMED REGION CROSSES FROM POPS TO APPENDS. reclaim() deals the freed
// nodes over ALL shards (the shard count is the hardware concurrency rounded up
// to a power of two, or HASH_ARENA_SHARDS), and a thread pops only from its own
// shard, so every timed thread starts with free/shards nodes to pop regardless
// of the thread count, and appends after that:
//   insert fixtures: kChurnPairs = 2^21 victims (plus the growth fixture's split
//     copies, ~0.4 per prefilled key) over S shards, against kNewIters = 2^18
//     inserts per thread: S = 16 -> 2^17 pops, half the run; S = 32 -> a quarter;
//     S = 128 -> 6%; S = 256 -> 3%. On a machine whose default shard count is
//     large, run the Del benchmarks again with HASH_ARENA_SHARDS=16 to get the
//     half-and-half regime.
//   lookup fixture: kPrefill = 2^20 victims over S shards, against
//     kOldIters/kInsertEvery = 10486 inserts per thread: all pops for S <= 64,
//     8192 pops then appends at S = 128, 4096 at S = 256.
// The fixture prints one line per configuration and process, before the first
// row that uses it, with the free-list total it measured (see report_prep()).
//
// The prep is deterministic (single-threaded, fixed keys), so the arena state
// at the start of the timed region is the same for every repetition and thread
// count. The arena_nodes_per_key counter of the insert bodies divides the
// TOTAL slot count, prep included, by the timed inserts: for the churn fixtures
// read it as (prep slots + timed appends)/timed inserts -- a Reclaimed row whose
// value equals its Control's minus the free-list share is a row that popped.
// ---------------------------------------------------------------------------
using ConcurrentSetDel = ConcurrentResizableHashSet<int, true, std::hash<int>>;

// What the fixture does to the set after the prefill: nothing (Control), or
// erase every victim and reclaim() (Reclaimed). Every Reclaimed configuration
// has a Control with the same live keys and the same buckets.
enum class Churn { Control, Reclaimed };

// The insert fixtures' churn population: kChurnPairs survivors, and as many
// victims for Reclaimed. Both live in the NEGATIVE keys, disjoint from every
// timed key (the timed ranges are non-negative: see the workload parameters),
// so no timed insert meets a prefilled key. The victim of pair k is inserted
// right after its survivor so that they alternate in the arena.
static constexpr int kChurnPairs = 1 << 21;

// Churn configuration for the insert workloads: `Buckets` is the initial table
// size (1024: the growth fixture's, so the prefill doubles the table as the
// timed inserts would have, and leaves the pending lazy splits the growth
// fixture would have left; kPresizedBuckets: no doubling at all, the
// allocator alone). The prefill of 2^22 keys (Reclaimed) stays far below the
// pre-sized doubling threshold of 2^24 slots.
template <size_t Buckets, Churn C>
struct InsertChurn {
    using Set = ConcurrentSetDel;
    static constexpr size_t buckets = Buckets;
    static constexpr Churn  churn   = C;
    static constexpr int    pairs   = kChurnPairs;
    static int survivor(int k) { return -1 - 2*k; }
    static int victim(int k)   { return -2 - 2*k; }
}; // struct InsertChurn

// Churn configuration for the lookup workload: the Control's prefill is
// MostlyOldFixture's exactly (keys [0, kPrefill) at 2*kPrefill buckets), so
// Lookup_MostlyOld_Del_Control is Lookup_MostlyOld_Del reached by another
// fixture, a check on the fixture itself. The victims are the OTHER half of
// the lookup range, [kPrefill, 2*kPrefill): the Reclaimed set once held every
// key the lookups will miss, so the misses walk chains reclaim() relinked, and
// the hit rate is the 50% of the workload above. 2*kPrefill prefilled nodes
// stay below the doubling threshold of 4*kPrefill slots.
template <Churn C>
struct LookupChurn {
    using Set = ConcurrentSetDel;
    static constexpr size_t buckets = 2*kPrefill;
    static constexpr Churn  churn   = C;
    static constexpr int    pairs   = kPrefill;
    static int survivor(int k) { return k; }
    static int victim(int k)   { return k + kPrefill; }
}; // struct LookupChurn

// One name per configuration: the benchmark macros take the configuration as
// one token, and Google Benchmark prints it in the benchmark name.
using NewControl        = InsertChurn<1024, Churn::Control>;
using NewReclaimed      = InsertChurn<1024, Churn::Reclaimed>;
using PresizedControl   = InsertChurn<kPresizedBuckets, Churn::Control>;
using PresizedReclaimed = InsertChurn<kPresizedBuckets, Churn::Reclaimed>;
using OldControl        = LookupChurn<Churn::Control>;
using OldReclaimed      = LookupChurn<Churn::Reclaimed>;

// The churn fixture: builds the set described by Cfg (above) in thread 0's
// SetUp(), untimed, under the same no-barrier CAUTION as the fixtures above.
// The benchmark bodies are the unchanged macros, which see the set through
// `set` exactly as they see the other fixtures'.
template <typename Cfg>
class ChurnFixture : public benchmark::Fixture {
public:
    using SetType = typename Cfg::Set;
    static SetType* set;
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() != 0) return;
        set = new SetType(Cfg::buckets, arena_shards_from_env());
        // Every prep operation must succeed (the keys are distinct and the
        // victims are present when erased); a failure would silently change
        // the population, so it aborts the run instead.
        long failed = 0;
        for (int k = 0; k < Cfg::pairs; ++k) {
            failed += !set->insert(mix(Cfg::survivor(k)));
            if constexpr (Cfg::churn == Churn::Reclaimed) failed += !set->insert(mix(Cfg::victim(k)));
        } // prefill, survivors and victims alternating
        if constexpr (Cfg::churn == Churn::Reclaimed) {
            for (int k = 0; k < Cfg::pairs; ++k) failed += !set->erase(mix(Cfg::victim(k)));
            set->reclaim();
        } // erase every victim, then reclaim
        if (failed != 0) {
            fprintf(stderr, "ChurnFixture: %ld prep operations failed in %s\n", failed, state.name().c_str());
            abort();
        } // if the prep did not build the population it describes
        report_prep(state);
    } // ChurnFixture::SetUp()
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    }
private:
    // Prints the arena state the timed region starts from, once per
    // configuration and process (it is the same for every repetition and
    // thread count: the prep is deterministic): slots ever appended, live
    // keys, free-list and limbo totals, the sweep's consistency verdict, the
    // shard count and the free nodes a timed thread can pop before its shard's
    // list is empty. To stdout, so the line lands in the benchmark output
    // before the first row that uses the configuration (Google Benchmark
    // writes its console report to stdout too). The accounting sweep sorts
    // every slot; a few hundred milliseconds, once, untimed.
    static void report_prep(const ::benchmark::State& state) {
        static bool reported = false;
        if (reported) return;
        reported = true;
        const typename SetType::InternalAccounting a = set->get_internal_accounting();
        const size_t shards = set->get_internal_arena_shards();
        printf("# prep %s: slots=%zu reachable=%zu free=%zu limbo=%zu consistent=%d shards=%zu free_per_shard=%zu\n",
               state.name().c_str(), a.slots, a.reachable, a.free_nodes, a.limbo, int(a.consistent), shards, a.free_nodes/shards);
        fflush(stdout);
    } // ChurnFixture::report_prep()
}; // class ChurnFixture
template <typename Cfg> typename Cfg::Set* ChurnFixture<Cfg>::set = nullptr;

// ---------------------------------------------------------------------------
// Registrations: identical workloads, two containers.
// ---------------------------------------------------------------------------
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW(Insert_MostlyNew_RWLocked,   LockedSet)

DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_RWLocked,   LockedSet)

DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_OLD(Lookup_MostlyOld_RWLocked,   LockedSet)

// The reclamation cells (see the section above): the AllowDelete == true set
// empty, prefilled, and prefilled by churn, for each of the three workloads.
// Nothing here changes a registration above.
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Del, ConcurrentSetDel)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Del_Control,   NewControl)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Del_Reclaimed, NewReclaimed)

DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_Del, ConcurrentSetDel)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Control,   PresizedControl)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Reclaimed, PresizedReclaimed)

DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Del, ConcurrentSetDel)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Control,   OldControl)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Reclaimed, OldReclaimed)

BENCHMARK_MAIN();
