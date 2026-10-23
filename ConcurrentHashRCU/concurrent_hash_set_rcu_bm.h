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
#ifndef CONCURRENT_HASH_SET_RCU_BM_H
#define CONCURRENT_HASH_SET_RCU_BM_H

// The shared part of the ConcurrentResizableHashSetRCU benchmark: everything
// its translation units (concurrent_hash_set_rcu_bm_*.C; the index of the rows
// and the overview of the workloads are at the top of
// concurrent_hash_set_rcu_bm.C) have in common. The key mixer, the baseline
// container, the threads' handles, the reclamation tallies and snapshots, the
// fixtures of the insert and lookup workloads and the macro bodies of those
// workloads, the churn fixture that builds the states of the AllowDelete set
// for those workloads, the prefill helper, and the per-thread-count
// registration helper.
// Everything here is a type, a template, a macro, an inline function, a
// constant or a static_assert,
// so a unit instantiates only what its rows use and the code generated for a
// unit's rows depends on that unit alone (why the benchmark is several units:
// concurrent_hash_set_rcu_bm.C). The functions are `inline`, not `static`: one
// definition across the units, and no unused-function warning in a unit that
// does not call one of them.

#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include "concurrent_hash_set_rcu.h"
#include "gb_wall_clock.h"      // WallRecords, WallTimed, report_wall() (../Spinlock)

// The hardware thread count, the top of every ThreadRange. One object for all
// the units (an inline variable), initialized before the registrations of any
// unit that includes this header: an inline variable's dynamic initialization
// is sequenced before the dynamic initializations that follow its definition
// in every unit where it is defined, and every unit includes this header
// before any of its registrations.
inline const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// Branch hints, two strengths because GCC compiles them differently:
//
//  BM_UNLIKELY(c)  the condition `c`, usually false but taken regularly (once
//                  per cadence, at the end of a thread's iterations):
//                  __builtin_expect's default weight, under which GCC keeps
//                  both paths close and does not make the taken path several
//                  times slower. Every hint on the timed loops is this one.
//  BM_RARELY(c)    the condition `c`, true almost never (error handling):
//                  probability 0.9999 of being false, under which the
//                  compiler may move the true path out of line and let it be
//                  slow. Not on the timed loops: with GCC the stronger weight
//                  also changes which functions are inlined into rows that
//                  contain no hint.
//
// The !! makes the builtin accept exactly what the bare condition accepts.
// Why the builtins and not [[likely]]/[[unlikely]]: the standard leaves the
// attribute's strength to the compiler, and GCC weighs it as one fixed
// probability, so it cannot express the two strengths; the builtins state the
// edge's weight. With GCC and Clang (including clang-cl) these
// are the builtins; elsewhere the hint is dropped and the code is still
// correct, only without the layout benefit.
#if defined(__GNUC__) || defined(__clang__)
#define BM_UNLIKELY(c) __builtin_expect(!!(c), 0)
#define BM_RARELY(c) __builtin_expect_with_probability(!!(c), 0, 0.9999)
#else
#define BM_UNLIKELY(c) (c)
#define BM_RARELY(c) (c)
#endif

// Bijective key mixer (the MurmurHash3 32-bit finalizer): every key that enters
// a set, and every key looked up, goes through it, so a workload written in
// terms of simple integers (sequential per thread, random in a range) reaches
// the identity std::hash<int> as random-looking values and different threads'
// keys spread over different buckets. A bijection keeps keys unique and hit
// rates unchanged. The result may be negative as an int; that is a valid key.
inline int mix(int k) {
    uint32_t x = static_cast<uint32_t>(k);
    x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16;
    return static_cast<int>(x);
}

// A benchmark knob read from the environment: an integer the variable holds.
// `text` is the variable's value, nullptr when it is not set; `valid` is true
// when that value is a whole decimal integer (optional leading white space and
// sign, digits, nothing after them), and `value` is then that integer.
struct EnvCount {
    const char* text = nullptr;   // the variable's value as set; nullptr when unset
    bool valid = false;           // text is a whole decimal integer
    long long value = 0;          // that integer, when valid
}; // struct EnvCount

// Reads the knob `name` (getenv) and parses it with strtoll, not atol: an
// out-of-range value is clamped to the range of long long rather than
// undefined, and a value with no digits or with anything after them ("one",
// "0x10", "1x", "") is reported invalid rather than read as 0 or as its
// leading digits. Prints nothing: each caller says on stderr what it does
// with an invalid value.
inline EnvCount count_from_env(const char* name) {
    EnvCount c;
    c.text = getenv(name);
    if (c.text == nullptr) return c;
    char* end = nullptr;
    c.value = strtoll(c.text, &end, 10);
    c.valid = end != c.text && *end == '\0';
    return c;
} // count_from_env()

// Arena shard count for ConcurrentResizableHashSetRCU, from the environment:
// HASH_ARENA_SHARDS=<n> (0 or unset = the header's default, the hardware
// concurrency rounded up to a power of two; a negative n is the default too,
// and an n above 2^16 is 2^16, the header's own cap). A value that is not a
// whole decimal integer is the default as well, and is reported once on
// stderr. Exists so one binary can measure how many shards a machine with
// many cores actually needs: the same workload is run with 16 shards and
// with one per hardware thread.
inline size_t arena_shards_from_env() {
    const EnvCount c = count_from_env("HASH_ARENA_SHARDS");
    if (c.text == nullptr) return 0;
    if (!c.valid) {
        // Once per process: it is called at every set construction.
        [[maybe_unused]] static const bool reported =
            (fprintf(stderr, "HASH_ARENA_SHARDS=\"%s\" is not a decimal integer: the header's default shard count is used\n", c.text), true);
        return 0;
    } // if not a number
    long long n = c.value;
    if (n < 0) n = 0;                    // nonsense means "the default"
    if (n > (1LL << 16)) n = 1LL << 16;  // the header's own cap
    return static_cast<size_t>(n);
} // arena_shards_from_env()

// A new set for a fixture: `buckets` buckets, the arena shards of
// arena_shards_from_env(), and the automatic cadence `auto_reclaim_every`
// (nodes retired per arena shard between two automatic reclaims; 0 = off,
// which is every row but the _Auto rows). Every fixture constructs its set
// here, through the set's Options form, so that a row with a cadence differs
// from its twin in one argument and nothing else (the two-parameter
// constructor is the same with the cadence off).
template <typename SetType>
SetType* new_set(size_t buckets, size_t auto_reclaim_every = 0) {
    return new SetType(typename SetType::Options{buckets, arena_shards_from_env(), auto_reclaim_every});
} // new_set()

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
    // The second parameter mirrors ConcurrentResizableHashSetRCU's arena_shards so
    // the fixtures can construct both containers the same way; it is ignored.
    explicit LockedHashSet(size_t initial_buckets = 1024, size_t /*arena_shards*/ = 0) : set_(initial_buckets) {}
    // Mirrors the concurrent set's Options form, which every fixture constructs
    // its set through (new_set() below): the bucket count is used, the arena
    // shards and the automatic cadence have no counterpart here and are ignored.
    struct Options {
        size_t initial_capacity = 1024;
        size_t arena_shards = 0;
        size_t auto_reclaim_every = 0;
    }; // struct LockedHashSet::Options
    explicit LockedHashSet(const Options& options) : set_(options.initial_capacity) {}
    // Mirrors the concurrent set's handle, the first argument of each of its
    // operations, so that the fixtures and the bodies compile over both
    // containers: the baseline needs none, so its handle is an empty type,
    // new_handle() returns one, and the operations ignore it.
    struct handle {};
    handle new_handle() const noexcept { return {}; }
    // Mirrors the concurrent set's diagnostic accessor for the arena_nodes_per_key
    // counter: the baseline has one node per key, so this is the key count.
    size_t get_internal_node_count() const { return set_.size(); }
    bool insert(const handle& /*h*/, const T& v) {
        std::unique_lock lock(mtx_);
        return set_.insert(v).second;
    }
    bool contains(const handle& /*h*/, const T& v) const {
        std::shared_lock lock(mtx_);              // <-- the RMW every reader pays
        return set_.find(v) != set_.end();
    }
}; // class LockedHashSet

using ConcurrentSet = ConcurrentResizableHashSetRCU<int, false, std::hash<int>>;
using LockedSet     = LockedHashSet<int, std::hash<int>>;

// ---------------------------------------------------------------------------
// The benchmark threads' handles.
//
// Every operation of ConcurrentResizableHashSetRCU runs under a handle. Each
// fixture keeps one per thread of a run: thread 0 creates them all at the END
// of its SetUp(), after every prep step -- any handle the prep itself used is
// released before the prep's advance(), and the threads' handles are created
// after it, so that it runs with no handle alive -- and each thread uses its
// own only INSIDE the state loop, whose start is a barrier: the happens-before
// edge a handle created on one thread needs before another thread uses it (see
// the CAUTION below). Only the reclamation rows (the _Rcu, _RcuEach and _Auto
// rows: advance() called, or the set's automatic cadence running) touch
// a handle inside the timed loop beyond using it: a thread refreshes it at the
// top of every iteration in the rows with a cadence (Own::refresh()), and
// releases it on its own LAST iteration (Own::finish_iteration(); Google
// Benchmark runs every thread for exactly state.max_iterations iterations), so
// that a thread that has finished and waits at the loop's end barrier pins no
// generation of the set for the threads still advancing. In every other row no
// advance() runs while the handles are live (a *_Reclaimed prep runs its one
// before they are made), so a live idle handle pins nothing anyone measures,
// and the timed loop carries no handle bookkeeping: the handles stay live
// until thread 0's TearDown() destroys them (the leave), before it deletes the
// set -- a live handle must not outlive its set. Not released by each thread
// after its loop: the code after the state loop runs concurrently with thread
// 0's TearDown(), so a late release could reach a set already deleted. Each
// handle sits on its own 128-byte block, since a release or a refresh writes
// its thread's handle and handles packed into one line would false-share.
// ---------------------------------------------------------------------------
template <typename SetType>
class ThreadHandles {
public:
    using Handle = typename SetType::handle;

    // Thread 0 only, at the end of SetUp(): one handle of `set` per thread of
    // the run, indexed by thread_index().
    void make(const SetType& set, int threads) {
        recs_ = std::vector<Rec>(threads);
        for (Rec& rec : recs_) rec.h = set.new_handle();
    } // ThreadHandles::make()

    // Thread 0 only, in TearDown(), before the set is deleted: destroys every
    // handle; in the rows without reclamation this is the leave of each
    // thread's live handle, in the reclamation rows each was released on its
    // thread's last iteration.
    void clear() { recs_.clear(); }

    // One thread's use of its handle, a local of the body. Trivially
    // destructible (asserted below), so no destructor is live across the
    // state loop (see the note above the benchmark bodies). Constructed
    // before the loop, it reads nothing of the handles there.
    class Own {
    public:
        Own(ThreadHandles& handles, const benchmark::State& state)
            : handles_(handles), thread_(state.thread_index()), last_(state.max_iterations) {}
        // The thread's handle. Inside the state loop only, and not after
        // finish_iteration() has released it.
        const Handle& get() const { return handles_.recs_[thread_].h; }
        // The rows with an advance() cadence, at the top of every
        // iteration: moves the handle to the set's current generation if an
        // advance() has published a newer one since (handle::refresh(): a
        // relaxed load and a compare when nothing changed), so that the
        // handle pins the generation of the operation it is about to run and
        // no older one. Inside the state loop only, before
        // finish_iteration() has released the handle.
        void refresh() { (void)handles_.recs_[thread_].h.refresh(); }
        // The reclamation rows only, at the end of every iteration; on the thread's
        // last iteration it releases the handle and returns true (false on
        // every other iteration), so that a row can act after the release
        // (see Erase_Presized_Del_Rcu).
        bool finish_iteration() {
            if (BM_UNLIKELY(++done_ == last_)) {
                handles_.recs_[thread_].h = Handle{};
                return true;
            }
            return false;
        } // ThreadHandles::Own::finish_iteration()
    private:
        ThreadHandles& handles_;                  // the fixture's handles
        const int thread_;                        // this thread's index in the run
        const benchmark::IterationCount last_;    // the iterations this thread runs (state.max_iterations)
        benchmark::IterationCount done_ = 0;      // iterations finished so far
    }; // class ThreadHandles::Own
    static_assert(std::is_trivially_destructible_v<Own>, "a body local with a destructor would be live across the state loop");

private:
    // One thread's handle on its own 128-byte block (see above).
    struct alignas(128) Rec {
        Handle h;   // the thread's handle: joined by thread 0 in SetUp(); released on the thread's last iteration in the reclamation rows, by thread 0's TearDown() otherwise
    }; // struct Rec
    std::vector<Rec> recs_;   // one per thread of the run; built by make(), destroyed by clear()
}; // class ThreadHandles

// ---------------------------------------------------------------------------
// One thread's advance() calls in a timed row, or its automatic triggers.
//
// The rows under the reclamation regime call advance() from inside their
// timed loops. What a call did is in its AdvanceResult; what it cost is in
// the CALLING thread's own timers (the set's internal_advance_timers(), one
// set of accumulators per thread, so a thread is charged for its own calls
// only -- and for the automatic triggers that PROCEEDED on it: the set
// charges a trigger's bump and free pass to the thread whose erase ran it,
// in bump_ns and free_ns, and nothing to advance_ns or walk_ns). An
// AdvanceTally, a local of the body, zeroes this thread's timers when it is
// constructed (before the loop: the timers accumulate over the thread's
// life, and Google Benchmark's thread 0 is the main thread, which ran the
// fixtures' prep and other rows; the helper threads are new per repetition,
// so for them the zeroing is a formality -- either way every repetition
// reads and resets its own threads' accumulators), counts the results inside
// the loop (add()) and sets the counters after it (report()). Three regimes:
//   - a cadence with ONE advancer (the _Rcu rows): thread 0 asks due() after
//     each of its erases and calls advance() after every `every` of them,
//     while every thread refreshes its handle. One advancer, because
//     advance() runs one call at a time: with every thread calling at a
//     cadence of its own erases, the calls collide (all but one find the
//     flag held and do nothing) and the number of walks per run changes with
//     the thread count; with one advancer it does not, for the same
//     per-thread work;
//   - a cadence with EVERY thread an advancer (the _RcuEach rows): each
//     thread calls after every `every` of its own erases; the first call of
//     thread i is due after max(1, every - every*i/T) of them, so the T first
//     calls are spread over (every/T, every] and do not land together (the
//     stagger; without it the first calls of a run collide by construction).
//     Most calls at T >= 8 are expected `contended`: that is the row's point,
//     the same ATTEMPT frequency as the _Auto rows, with the walk;
//   - AUTOMATIC (the _Auto rows): nothing is called; the set triggers from
//     inside the erases. The tally counts the triggers this thread DETECTED
//     (detected(): the body sees its own bump_ns change across an erase, see
//     the _Auto rows) and reports the timers the triggers charged. due() then
//     paces thread 0's table-size samples of the "# trace" series (the churn
//     row; every = the sampling cadence, 0 = no samples, due() never called).
// Every counter is set by every thread and summed over the threads by
// Google Benchmark, so each is a total for the row:
//   advance_calls      calls made (0 on the _Auto rows);
//   advanced, nothing_retired, contended, ring_full
//                      their outcomes (AdvanceResult::Advance): `contended`
//                      did nothing (another thread's call held the flag, and
//                      the call is not retried), `ring_full` freed what it
//                      could and did not walk;
//   freed_nodes, walk_retired, buckets_walked, nodes_passed
//                      the AdvanceResult counts, summed;
//   advance_ns, bump_ns, free_ns, walk_ns
//                      the time spent in the calls, whole and by step (the
//                      bump, the free pass after it, the walk), in ns; on the
//                      _Auto rows bump_ns and free_ns are the proceeding
//                      triggers' and the other two are 0;
//   advance_share, walk_share
//                      advance_ns and walk_ns over the summed loop time of the
//                      threads (a per-thread time in seconds with
//                      kAvgThreadsRate: Google Benchmark divides the sum over
//                      the threads by the thread count and by the mean loop
//                      time). With one advancer, walk_share is that thread's
//                      walk time over every thread's loop time, and the
//                      advancer finishes last (finish_spread). Ratios: the
//                      console prints them with the "/s" it gives every rate
//                      counter;
//   advance_every      the cadence: a thread's erases between two of its
//                      calls (the explicit rows with a cadence only);
//   reclaim_ns, reclaim_share
//                      (_Auto rows only) bump_ns + free_ns, the triggers'
//                      whole cost, and its share of the summed loop time, as
//                      walk_share is formed;
//   detected_triggers  (_Auto rows only) the triggers the threads detected
//                      inside their erases; equals the set's auto_reclaims
//                      delta unless a trigger ran outside an erase (a split
//                      cleanup inside insert() or contains() on a table that
//                      doubled) or a proceeding trigger's bump took under a
//                      nanosecond (the detection reads the thread's bump_ns).
// Trivially destructible, like ThreadHandles::Own.
// ---------------------------------------------------------------------------
template <typename SetType>
class AdvanceTally {
public:
    using Timers = typename SetType::InternalTimers;
    // Which threads of an explicit row call advance() at the cadence.
    enum class Advancer { thread0, every_thread };
    // The tag of the automatic rows' constructor.
    struct Automatic {};

    // A row without a cadence (due() is never called).
    AdvanceTally() : AdvanceTally(0, int64_t{0}, false) {}
    // A row with a cadence, for the thread of `state`: the advancing threads
    // (thread 0 alone, or every thread) are due to call advance() after
    // every `every` (> 0) of their own erases, every thread's first call
    // staggered as described above; on a non-advancing thread due() is never
    // true.
    AdvanceTally(int every, const benchmark::State& state, Advancer advancer = Advancer::thread0)
        : AdvanceTally(every, first_due(every, state, advancer), false) {}
    // An automatic row, for the thread of `state`: no call is made; thread 0
    // is due (due()) after every `sample_every` of its erases to sample the
    // table for the "# trace" series, never if `sample_every` is 0.
    AdvanceTally(Automatic, int sample_every, const benchmark::State& state)
        : AdvanceTally(sample_every, first_due(sample_every, state, Advancer::thread0), true) {}

    // This thread's own timers (one set per thread, the set's
    // internal_advance_timers()): the _Auto rows hoist the reference before
    // the loop and read bump_ns through it around every erase.
    static Timers& timers() { return SetType::internal_advance_timers(); }

    // Inside the loop, after each of the thread's erases, in a row with a
    // cadence: true on every every-th call, when the thread is to call
    // advance() (or, on an automatic row, thread 0 is to sample). A
    // countdown, so the test is a decrement and a compare.
    bool due() {
        if (--countdown_ != 0) return false;
        countdown_ = every_;
        return true;
    } // AdvanceTally::due()

    // Inside the loop: one call's result.
    void add(const AdvanceResult& r) {
        ++calls_;
        switch (r.advance) {
        case AdvanceResult::Advance::advanced:        ++advanced_; break;
        case AdvanceResult::Advance::nothing_retired: ++nothing_retired_; break;
        case AdvanceResult::Advance::contended:       ++contended_; break;
        case AdvanceResult::Advance::ring_full:       ++ring_full_; break;
        } // switch on the outcome
        freed_nodes_ += r.freed_nodes;
        walk_retired_ += r.walk_retired;
        buckets_walked_ += r.buckets_walked;
        nodes_passed_ += r.nodes_passed;
    } // AdvanceTally::add()

    // Inside the loop of an automatic row: one trigger detected in this
    // thread's erase.
    void detected() { ++detected_; }

    // The nodes this thread's calls freed so far (the "# trace" entries of
    // the explicit rows carry it).
    size_t freed_nodes() const { return freed_nodes_; }

    // After the loop, on every thread: this thread's counts and timers as the
    // counters listed above.
    void report(benchmark::State& state) const {
        const Timers& t = timers();
        state.counters["advance_calls"]   = benchmark::Counter(double(calls_));
        state.counters["advanced"]        = benchmark::Counter(double(advanced_));
        state.counters["nothing_retired"] = benchmark::Counter(double(nothing_retired_));
        state.counters["contended"]       = benchmark::Counter(double(contended_));
        state.counters["ring_full"]       = benchmark::Counter(double(ring_full_));
        state.counters["freed_nodes"]     = benchmark::Counter(double(freed_nodes_));
        state.counters["walk_retired"]    = benchmark::Counter(double(walk_retired_));
        state.counters["buckets_walked"]  = benchmark::Counter(double(buckets_walked_));
        state.counters["nodes_passed"]    = benchmark::Counter(double(nodes_passed_));
        state.counters["advance_ns"]      = benchmark::Counter(double(t.advance_ns));
        state.counters["bump_ns"]         = benchmark::Counter(double(t.bump_ns));
        state.counters["free_ns"]         = benchmark::Counter(double(t.free_ns));
        state.counters["walk_ns"]         = benchmark::Counter(double(t.walk_ns));
        state.counters["advance_share"]   = benchmark::Counter(double(t.advance_ns)*1e-9, benchmark::Counter::kAvgThreadsRate);
        state.counters["walk_share"]      = benchmark::Counter(double(t.walk_ns)*1e-9, benchmark::Counter::kAvgThreadsRate);
        if (automatic_) {
            state.counters["reclaim_ns"]        = benchmark::Counter(double(t.bump_ns + t.free_ns));
            state.counters["reclaim_share"]     = benchmark::Counter(double(t.bump_ns + t.free_ns)*1e-9, benchmark::Counter::kAvgThreadsRate);
            state.counters["detected_triggers"] = benchmark::Counter(double(detected_));
        } else if (every_ != 0) {
            state.counters["advance_every"] = benchmark::Counter(double(every_), benchmark::Counter::kAvgThreads);
        } // the regime's own counters
    } // AdvanceTally::report()

private:
    // The erases until the first call (or sample) of the thread of `state`:
    // `every` on thread 0 and never on the others with one advancer; the
    // staggered max(1, every - every*i/T) on thread i of T with every thread
    // an advancer. "Never" is a countdown that no run reaches.
    static int64_t first_due(int every, const benchmark::State& state, Advancer advancer) {
        if (advancer == Advancer::every_thread) {
            return std::max<int64_t>(1, every - int64_t(every)*state.thread_index()/state.threads());
        } // every thread an advancer: the stagger
        return state.thread_index() == 0 ? int64_t{every} : std::numeric_limits<int64_t>::max();
    } // AdvanceTally::first_due()

    // The public constructors' common part: the cadence, the erases until
    // the first call, the regime, and this thread's timers zeroed.
    AdvanceTally(int every, int64_t countdown, bool automatic) : every_(every), countdown_(countdown), automatic_(automatic) {
        timers() = {0, 0, 0, 0};
    } // AdvanceTally::AdvanceTally(every, countdown, automatic)

    const int every_;              // the cadence: an advancing thread calls (an automatic row's thread 0 samples) after every every_ of its erases; 0 = no cadence
    int64_t countdown_;            // erases left until the next call or sample (rows with a cadence; never reaches 0 on a non-advancing thread)
    const bool automatic_;         // an _Auto row: report the triggers' counters instead of advance_every
    int64_t calls_ = 0;            // advance() calls
    int64_t advanced_ = 0;         // of those, by outcome
    int64_t nothing_retired_ = 0;
    int64_t contended_ = 0;
    int64_t ring_full_ = 0;
    int64_t detected_ = 0;         // _Auto rows: triggers detected in this thread's erases
    size_t freed_nodes_ = 0;       // AdvanceResult counts, summed over the calls
    size_t walk_retired_ = 0;
    size_t buckets_walked_ = 0;
    size_t nodes_passed_ = 0;
}; // class AdvanceTally

// ---------------------------------------------------------------------------
// advance() called out of line, for the timed bodies.
//
// Every timed loop that calls advance() among other operations (the _Rcu and
// _RcuEach rows of the erase and churn units, at their cadence, and
// Erase_Presized_Del_Rcu, once per thread after its last erase) calls it
// through this function, which the compiler does not inline; the Advance_
// rows, which time advance() itself, call it directly. The call is rare (once per cadence, or once per
// thread per run) and heavy (the bump, the free pass, the walk), and its
// time is part of the row's measurement whether it is inlined or not. What
// inlining it changes is the loop around it: whether GCC inlines advance()
// into a loop changes with the container header's code, including code that
// never runs on the loop's path, and with advance() inlined GCC can leave the
// loop's own erase() out of line. The per-erase code of a row would then
// depend on the header's reclamation code rather than on the row. Out of
// line, the call adds a call and a return per advance(), and whether
// advance()'s code is inlined into the loop no longer depends on the loop or
// on the header; which set operations are inlined into the loop is still
// decided across the whole unit (why the benchmark is several units:
// concurrent_hash_set_rcu_bm.C).
//
// [[gnu::noinline]]: GCC and Clang honor it; a compiler without the gnu
// attributes ignores it, and the function is then an ordinary call that may
// be inlined, as `set.advance()` would be.
//
// Returns `set.advance()`'s result; `set` is the row's set (the fixture's).
// ---------------------------------------------------------------------------
template <typename SetType>
[[gnu::noinline]] AdvanceResult advance_out_of_line(SetType& set) {
    return set.advance();
} // advance_out_of_line()

// ---------------------------------------------------------------------------
// The set's reclamation counters before a timed loop, and their deltas after
// it.
//
// Every reclamation row reports what the set's own counters say the calls
// and the triggers did, beside the threads' tallies: the two count the same
// calls on the explicit rows (set_nodes_freed == the sum of freed_nodes,
// set_bumps == the sum of advanced, set_contended == the sum of contended:
// no trigger runs on a cadence-off set), and on the _Auto rows the set's
// counters are the only readout. The fixture takes the snapshot in thread
// 0's SetUp(), after its prefill, so every reported count is a DELTA across
// the timed loop; thread 0 reads the deltas after the loop (every thread
// has left it; the set is quiescent), untimed. The counters (the set's
// InternalCounters, see the header; the `set_` prefix keeps them apart from
// AdvanceTally's per-thread ring_full and contended):
//   set_bumps, set_ring_full, set_contended, set_bags_freed, set_nodes_freed
//                      publishes, ring_full outcomes, refused explicit calls,
//                      bags and nodes freed, whoever ran them;
//   auto_reclaims, auto_skipped
//                      automatic triggers that acquired the flag (whatever
//                      their bump then decided) and that found it held (0 on
//                      every cadence-off row).
// On the _Auto rows additionally (report_shards()): which arena shard each
// thread retired into. A thread's shard is its thread number masked by the
// shard count, and Google Benchmark's helper threads are new per repetition
// while thread 0 is the main thread, so two threads of a run can share a
// shard, which then crosses the cadence twice as often (the attempts stay
// the same in total: every crossing is one push's). The per-shard
// retire_nodes deltas say whether that happened:
//   # shards <row> threads=<T> used=<n> min=<m> max=<M> prefill_splits=<s>
//   prefill_triggers=<a> deltas=<d0>,<d1>,...
// (prefill_splits and prefill_triggers: the splits the prefill published
// and the triggers it ran, before the snapshot, so the row's reader knows
// whether the untimed prefill retired or triggered anything), and as the
// counters shards_used (shards with a non-zero delta), shard_min and
// shard_max (over those), and auto_reclaim_every (the set's effective
// cadence, read from the set by thread 0 after the loop: the row's witness
// that it ran at the cadence its name says). A repetition with shards_used
// < T had a shared shard. A delta is attributed to the shard of the thread
// that PUSHED the run, which is the eraser for a self-unlink and the
// unlinking walker otherwise, so a shard's delta is about, not exactly, its
// thread's erase count.
// ---------------------------------------------------------------------------
template <typename SetType>
struct ReclaimSnapshot {
    typename SetType::InternalCounters counters{};   // the set's counters at the snapshot
    std::vector<size_t> retire_nodes_per_shard;      // every shard's retire_nodes at the snapshot, in shard order

    // Thread 0, in SetUp() after the prefill: the snapshot.
    void take(const SetType& set) {
        counters = set.get_internal_counters();
        retire_nodes_per_shard = set.get_internal_retire_nodes_per_shard();
    } // ReclaimSnapshot::take()

    // Thread 0, after the loop: the set_ and auto_ counters above as deltas.
    void report_set(benchmark::State& state, const SetType& set) const {
        const typename SetType::InternalCounters c = set.get_internal_counters();
        state.counters["set_bumps"]       = benchmark::Counter(double(c.bumps - counters.bumps));
        state.counters["set_ring_full"]   = benchmark::Counter(double(c.ring_full - counters.ring_full));
        state.counters["set_contended"]   = benchmark::Counter(double(c.contended - counters.contended));
        state.counters["set_bags_freed"]  = benchmark::Counter(double(c.bags_freed - counters.bags_freed));
        state.counters["set_nodes_freed"] = benchmark::Counter(double(c.nodes_freed - counters.nodes_freed));
        state.counters["auto_reclaims"]   = benchmark::Counter(double(c.auto_reclaims - counters.auto_reclaims));
        state.counters["auto_skipped"]    = benchmark::Counter(double(c.auto_skipped - counters.auto_skipped));
    } // ReclaimSnapshot::report_set()

    // Thread 0, after the loop, on the _Auto rows: the "# shards" line and
    // the shard counters above, and the cadence witness.
    void report_shards(benchmark::State& state, const SetType& set) const {
        const std::vector<size_t> now = set.get_internal_retire_nodes_per_shard();
        size_t used = 0, min = std::numeric_limits<size_t>::max(), max = 0;
        std::string deltas;
        for (size_t s = 0; s < now.size(); ++s) {
            const size_t delta = now[s] - retire_nodes_per_shard[s];
            if (delta != 0) {
                ++used;
                min = std::min(min, delta);
                max = std::max(max, delta);
            } // a shard some thread retired into
            deltas += (s == 0 ? "" : ",") + std::to_string(delta);
        } // loop over the shards
        if (used == 0) min = 0;
        printf("# shards %s threads=%d used=%zu min=%zu max=%zu prefill_splits=%zu prefill_triggers=%zu deltas=%s\n",
               state.name().c_str(), state.threads(), used, min, max, counters.splits_published,
               counters.auto_reclaims + counters.auto_skipped, deltas.c_str());
        fflush(stdout);
        state.counters["shards_used"]        = benchmark::Counter(double(used));
        state.counters["shard_min"]          = benchmark::Counter(double(min));
        state.counters["shard_max"]          = benchmark::Counter(double(max));
        state.counters["auto_reclaim_every"] = benchmark::Counter(double(set.get_internal_auto_reclaim_every()));
    } // ReclaimSnapshot::report_shards()
}; // struct ReclaimSnapshot

// The cadence of advance() in a row that has one (each such row says what its
// cadence counts): the row's value `registered`, unless the environment
// variable `name` sets another (a positive integer; anything else means the
// default). Exists so that one binary can be calibrated and re-run at other
// cadences; every such row reports thread 0's cadence, the one it ran with,
// as the counter advance_every.
inline int advance_every_from_env(const char* name, int registered) {
    const char* s = getenv(name);
    const long n = s ? atol(s) : 0;
    return n > 0 && n <= (1L << 30) ? int(n) : registered;
} // advance_every_from_env()

// The automatic cadence a fixture constructs its set with, as a policy type
// (the fixture's last template parameter): auto_reclaim_every(state) is the
// set's Options::auto_reclaim_every for a run of state.threads() threads.
// NoCadence, the default, is every row but the _Auto rows; the _Auto rows'
// policies are defined beside their twins' cadences (SmallAutoCadence,
// WindowAutoCadence), from the same constants and the same environment
// overrides, so that a twin and its _Auto row always run at the same
// cadence. A policy of its own per row, and not a fixture argument, because
// the fixture's SetUp() constructs the set before the body runs and the
// cadence depends on the thread count.
struct NoCadence {
    static size_t auto_reclaim_every(const benchmark::State& /*state*/) { return 0; }
}; // struct NoCadence

// ---------------------------------------------------------------------------
// Fixtures.
//
// CAUTION: Google Benchmark runs SetUp() per thread with NO implicit barrier
// between SetUp() and the start of the benchmark body. The only implicit
// barrier is the benchmark loop itself. Therefore: thread 0
// creates (and, for MostlyOld, prefills) the container in its SetUp(), and no
// thread dereferences `set` anywhere except INSIDE the loop (or after it, once
// the loop's end barrier has been crossed). The same holds for the threads'
// handles, `handles` (ThreadHandles above), which thread 0 creates last.
// ---------------------------------------------------------------------------
template <typename SetType>
class MostlyNewFixture : public benchmark::Fixture {
public:
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles above)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            set = new_set<SetType>(1024);
            handles.make(*set, state.threads());
        }
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class MostlyNewFixture
template <typename SetType> SetType* MostlyNewFixture<SetType>::set = nullptr;
template <typename SetType> ThreadHandles<SetType> MostlyNewFixture<SetType>::handles;

// The same insert workload into a table constructed with kPresizedBuckets
// buckets, so that no doubling and no lazy split happens until the arena holds
// 2*kPresizedBuckets nodes (16.7M: equal to 64 threads' worth of kNewIters). It
// separates the allocator's own cost from the doubling and split path: with the
// table geometry fixed, the arena (its lock or its shards) is the only shared
// state on the insert path; the allocator numbers ../ConcurrentHash/README.md
// quotes for the original container come from this fixture there. Every insert
// is still a random bucket-head miss.
static constexpr size_t kPresizedBuckets = size_t(1) << 23;
template <typename SetType>
class MostlyNewPresizedFixture : public benchmark::Fixture {
public:
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles above)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            set = new_set<SetType>(kPresizedBuckets);
            handles.make(*set, state.threads());
        }
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class MostlyNewPresizedFixture
template <typename SetType> SetType* MostlyNewPresizedFixture<SetType>::set = nullptr;
template <typename SetType> ThreadHandles<SetType> MostlyNewPresizedFixture<SetType>::handles;

template <typename SetType>
class MostlyOldFixture : public benchmark::Fixture {
public:
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles above)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            /*
             * CAUTION: construct at full capacity. Steady state requires that
             * NO resize can occur, in either container, during the timed
             * region -- and for ConcurrentResizableHashSetRCU that requirement
             * is stronger than it looks. If the table is prefilled through
             * doublings (e.g., constructed at 1024), it exits SetUp with a
             * large population of UNINITIALIZED buckets whose lazy splits
             * have not happened yet. Those splits are then performed INSIDE
             * the timed region, cooperatively, BY THE READERS -- and every
             * split allocates through its arena shard's spinlock_. The
             * nominally lock-free lookup benchmark degenerates into a
             * spinlock convoy (wall time grows with threads while CPU
             * stays flat).
             * Constructing at 2*kPrefill means the table does not double,
             * up to the thread counts given after the index in
             * concurrent_hash_set_rcu_bm.C: every bucket is born EMPTY, no split ever
             * exists, no stale copies inflate the chains. The unordered_set
             * baseline gets the same courtesy (2*kPrefill buckets, so no
             * rehash either, up to the thread count that index gives
             * for it).
             * Resize behavior under load is a tail-latency story and gets
             * its own benchmark; it has no business inside a mean.
             */
            set = new_set<SetType>(2*kPrefill);
            {   // the prefill's own handle, released before the threads' handles are made
                const typename SetType::handle h = set->new_handle();
                for (int k = 0; k < kPrefill; ++k) set->insert(h, mix(k));   // untimed
            }
            handles.make(*set, state.threads());
        }
    }
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class MostlyOldFixture
template <typename SetType> SetType* MostlyOldFixture<SetType>::set = nullptr;
template <typename SetType> ThreadHandles<SetType> MostlyOldFixture<SetType>::handles;

// ---------------------------------------------------------------------------
// Benchmark bodies (as macros, so each container gets an identical body).
//
// Each body keeps its own wall-clock records, a function-local static (one per
// instantiation), rather than one per fixture: one placement then serves
// every fixture type the macro is used on. The calls follow
// gb_wall_clock.h's contract: allocate() before the loop and, after it,
// report_wall() then release(), each called on every thread and doing its work
// on thread 0 alone; state.iterations() is read only after the loop, and the
// per-thread SkipWithError() is called after the loop, never inside it. No
// object with a destructor is live across the loop: its cleanup region can
// change the loop's code. Each operation runs under the thread's own handle,
// through the trivially destructible ThreadHandles::Own, and these bodies do
// nothing else with it (no row they define calls advance() while the
// handles are live: ThreadHandles says why the handles then stay live until
// TearDown()), except in Lookup_MostlyOld_OpHandle, where each operation
// takes a handle of its own (the thread's handle is still made and
// destroyed, unused).
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
        Handles::Own own(handles, state);                                     \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            ok += set->insert(own.get(), mix(base + next++));                 \
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
 * HANDLE is the expression each operation is called with as its handle: the
 * thread's own long-lived handle, own.get(), in every row but one;
 * set->new_handle() in Lookup_MostlyOld_OpHandle, a temporary handle that
 * joins the current generation before the operation and leaves it when the
 * operation's full expression ends.
 */
#define DEFINE_MOSTLY_OLD(NAME, SET_TYPE) DEFINE_MOSTLY_OLD_ON(MostlyOldFixture, NAME, SET_TYPE)
#define DEFINE_MOSTLY_OLD_ON(FIXTURE, NAME, SET_TYPE) DEFINE_MOSTLY_OLD_UNDER(FIXTURE, NAME, SET_TYPE, own.get())
#define DEFINE_MOSTLY_OLD_UNDER(FIXTURE, NAME, SET_TYPE, HANDLE)              \
    BENCHMARK_TEMPLATE_DEFINE_F(FIXTURE, NAME, SET_TYPE)                      \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        XorShift32 rng(0x1234567u + 0x9e3779b9u*state.thread_index());        \
        const int base = kNewBase + state.thread_index()*kKeyStride;          \
        int next = 0;                                                         \
        int op = 0;                                                           \
        int64_t ok = 0, tries = 0;                                            \
        bool hit = false;                                                     \
        Handles::Own own(handles, state);                                     \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            if (++op == kInsertEvery) {                                       \
                op = 0;                                                       \
                ++tries;                                                      \
                ok += set->insert(HANDLE, mix(base + next++));                \
            } else {                                                          \
                hit = set->contains(HANDLE,                                   \
                                    mix((int)(rng.next() & kLookupMask)));    \
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

// The AllowDelete == true instantiation, the set of every row that erases or
// reclaims (the "Reclamation benchmarks" below say what the *_Del rows measure
// against the AllowDelete == false rows).
using ConcurrentSetDel = ConcurrentResizableHashSetRCU<int, true, std::hash<int>>;

// The node layout these measurements were made for, checked at compile time so
// that it cannot change unnoticed: comparisons of this container with the
// original ConcurrentResizableHashSet rest on the original's layout (the key,
// then the link word) plus this container's retire_link, LAST. For int keys:
// 24 bytes, the link at offset 8, retire_link at offset 16 (the original's node
// is 16 bytes). Both instantiations share the layout.
static_assert(sizeof(ConcurrentSetDel::Node) == 24, "the node layout changed: 24 bytes for int keys expected");
static_assert(offsetof(ConcurrentSetDel::Node, link) == 8, "the node layout changed: the link word at offset 8 expected");
static_assert(offsetof(ConcurrentSetDel::Node, retire_link) == 16, "the node layout changed: retire_link last, at offset 16, expected");
static_assert(sizeof(ConcurrentSet::Node) == sizeof(ConcurrentSetDel::Node), "the two instantiations' nodes differ in size");
// AdvanceTally is a local of the bodies that use it, on this instantiation (see there).
static_assert(std::is_trivially_destructible_v<AdvanceTally<ConcurrentSetDel>>, "a body local with a destructor would be live across the state loop");

// ---------------------------------------------------------------------------
// Reclamation benchmarks.
//
// The insert and lookup workloads above are re-run on the AllowDelete == true instantiation
// in several states of the set, and every cell has a purpose:
//
//   *_Del            the workload above, on an EMPTY AllowDelete == true set (the
//                    MostlyOld fixture prefills as before). Nothing is erased.
//                    In this container the two instantiations run the same
//                    insert, lookup and split code (the split freezes the nodes
//                    it moves for both values, and with AllowDelete == false the
//                    MARK tests test a bit that is never set), so against the
//                    AllowDelete == false rows this checks that compiling
//                    erase() in costs nothing on those paths.
//   *_Del_Control    the same workload on a set PREFILLED (untimed, by thread 0)
//                    with a population of live keys and nothing else: no erase,
//                    no advance(), every free list empty, so every allocation
//                    in the timed region APPENDS to a deque.
//   *_Del_Reclaimed  the same workload on a set that reached the same live
//                    population by CHURN: twice as many keys were inserted,
//                    interleaved (survivor, victim, survivor, victim, ... in
//                    arena order), every victim was erased, and then, with no
//                    handle live, one advance() ran: it closed every retired
//                    node into a bag and, since no handle pinned the bag,
//                    freed it in the same call. The live keys and the timed
//                    workload are those of the Control; what differs is the
//                    state of the arena: the victims' slots (and, in the
//                    growth fixture, the split copies the prefill superseded)
//                    sit on the shards' free lists, dealt round-robin, and the
//                    survivors occupy every other slot of the deque blocks
//                    thread 0 filled. The timed allocations POP from the
//                    calling thread's shard's list, then from the other
//                    shards' lists, and append once every list is empty --
//                    the mixed regime (see below).
//   *_Del_Erased     (pre-sized insert and lookup only) the same churn without
//                    the advance(): the victims are erased and the timed
//                    workload starts right after. It is the only state of the
//                    set that erases built: in this container each erase()
//                    unlinked its own node at once and retired it, so the
//                    timed walks find chains of live nodes only, where a
//                    container that leaves tombstones in place until a
//                    collector runs walks past every one of them. The free
//                    lists are empty, as in the Control. Why measure: this is
//                    what eager unlinking buys a reader and a writer;
//                    Erase_Presized_Del (concurrent_hash_set_rcu_bm_erase.C)
//                    measures what it costs the eraser.
//
// Control vs Del separates "a resident population" from "AllowDelete"; Reclaimed
// vs Control is the free lists in use: the pop path in alloc_node() (one CAS on
// a shard's free head instead of the deque's lock and cursor) and the locality
// of the popped slots; Erased vs Control is the dead nodes the erases left (or,
// here, did not leave) in the chains. advance()'s free pass deals the freed
// nodes round-robin over the shards, from shard 0 on for a fresh set, in the
// order it walks the bag: shard by shard, each shard's retired nodes (that
// list is LIFO: latest retirement first) and then its limbo nodes; each free
// list is LIFO too, so a shard pops its share in reverse dealing order. In
// this container the prefill's erases unlink and retire every victim
// themselves, onto thread 0's shard's retired list, and the single-threaded
// prefill leaves no limbo: the free pass deals the victims in reverse erase
// order, so each shard pops its share in erase order, i.e. in arena order,
// every 2*shards-th slot in the pre-sized and lookup fixtures (survivors and
// victims alternate in the arena; in the growth fixture the split copies
// interleave with them, and the superseded copies were retired in split
// order, before the victims). Popped slots are therefore at least
// 2*shards*sizeof(Node) bytes apart, a new cache line per popped node at any
// shard count above one, where an append fills a line with two or three
// consecutive 24-byte nodes. The prep leaves no dead node in a chain, so the
// walk of that advance() unlinks nothing. Why measure: the free lists are what
// reclamation is for, and these are the only cells that measure an insert
// that reuses a slot, or a lookup over chains whose erased nodes were
// recycled.
//
// The pre-sized and lookup cells share their bucket count between Control and
// Reclaimed (no doubling in either prefill), so there the free lists are the
// only difference. The GROWTH cells do not: the Reclaimed prefill inserts twice
// the keys through the same doublings, so its table is one doubling larger
// (2^22 buckets against the Control's 2^21 with 16 shards), with a
// different population of pending lazy splits, which the timed inserts then
// perform and which allocate their copies from the same free lists; and the
// free pass lowered its resize hint (node_count_) by the nodes it freed, to
// about the live count, so its next doubling comes later.
// The growth Reclaimed row is therefore "churn, then growth" as a workload,
// not a controlled measurement of the allocator; read the pre-sized row for
// that. Neither prefill can be made to end at the other's geometry with the
// same live keys: the doubling threshold is a slot count and the prefills
// differ by 2x in slots.
//
// WHEN THE TIMED REGION CROSSES FROM POPS TO APPENDS. The free pass deals the
// freed nodes over ALL shards (the shard count is the hardware concurrency
// rounded up to a power of two, or HASH_ARENA_SHARDS); a thread pops from its
// own shard's list, and when that is empty, from the other shards' lists (the
// steal: alloc_node() tries each other shard's list once before it appends).
// So the timed threads together pop every freed node before any of them
// appends, whatever the shard count: at equal rates, about free/T pops per
// thread, then appends.
//   insert fixtures: kChurnPairs = 2^21 victims (plus the growth fixture's split
//     copies, ~0.4 per prefilled key) against T*kNewIters = T*2^18 inserts:
//     the pre-sized cell pops throughout at T <= 8 and for half the run at
//     T = 16; the growth cell's free lists (2^21 victims and its copies) also
//     serve the copies its timed splits allocate.
//   lookup fixture: kPrefill = 2^20 victims against T*kOldIters/kInsertEvery =
//     T*10485 inserts: all pops up to about 100 threads.
// The fixture prints one line per configuration and process, before the first
// row that uses it, with the free-list total it measured and the smallest and
// largest per-shard list (see report_prep()).
//
// The prep is deterministic (single-threaded, fixed keys), so the arena state
// at the start of the timed region is the same for every repetition and thread
// count. The arena_nodes_per_key counter of the insert bodies divides the
// TOTAL slot count, prep included, by the timed inserts: for the churn fixtures
// read it as (prep slots + timed appends)/timed inserts -- a Reclaimed row whose
// value equals its Control's minus the free-list share is a row that popped.
// ---------------------------------------------------------------------------

// What the fixture does to the set after the prefill: nothing (Control), erase
// every victim and run one advance() (Reclaimed), or erase every victim and
// nothing more (Erased). Every Reclaimed and Erased configuration has a
// Control with the same live keys and the same initial bucket count (the
// growth cells' Reclaimed table ends one doubling larger; see above).
enum class Churn { Control, Reclaimed, Erased };

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
// key the lookups will miss, so the misses walk chains those keys were
// unlinked from, and the hit rate is the 50% of the workload above. 2*kPrefill
// prefilled nodes stay below the doubling threshold of 4*kPrefill slots.
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
using PresizedErased    = InsertChurn<kPresizedBuckets, Churn::Erased>;
using OldControl        = LookupChurn<Churn::Control>;
using OldReclaimed      = LookupChurn<Churn::Reclaimed>;
using OldErased         = LookupChurn<Churn::Erased>;

// The churn fixture: builds the set described by Cfg (above) in thread 0's
// SetUp(), untimed, under the same no-barrier CAUTION as the fixtures above.
// The benchmark bodies are the unchanged macros, which see the set through
// `set` exactly as they see the other fixtures'; the rows are registered in
// concurrent_hash_set_rcu_bm_states_insert.C and _states_lookup.C.
template <typename Cfg>
class ChurnFixture : public benchmark::Fixture {
public:
    using SetType = typename Cfg::Set;
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles above)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() != 0) return;
        set = new_set<SetType>(Cfg::buckets);
        // Every prep operation must succeed (the keys are distinct and the
        // victims are present when erased); a failure would silently change
        // the population, so it aborts the run instead.
        long failed = 0;
        {   // the prep's own handle, released before the advance() below (see ThreadHandles)
            const typename SetType::handle h = set->new_handle();
            for (int k = 0; k < Cfg::pairs; ++k) {
                failed += !set->insert(h, mix(Cfg::survivor(k)));
                if constexpr (Cfg::churn != Churn::Control) failed += !set->insert(h, mix(Cfg::victim(k)));
            } // prefill, survivors and victims alternating
            if constexpr (Cfg::churn != Churn::Control) {
                for (int k = 0; k < Cfg::pairs; ++k) failed += !set->erase(h, mix(Cfg::victim(k)));
            } // erase every victim
        } // the prep's handle
        // With no handle live, one advance() bags every retired node and frees the bag in the
        // same call; the prep leaves no dead node reachable, so its walk retires nothing. A
        // call that did anything else did not build the state the rows describe: counted as
        // a failed prep operation.
        if constexpr (Cfg::churn == Churn::Reclaimed) {
            const AdvanceResult r = set->advance();
            failed += r.advance != AdvanceResult::Advance::advanced || r.freed_bags != 1 || r.walk_retired != 0;
        } // the prep's one advance()
        if (failed != 0) {
            fprintf(stderr, "ChurnFixture: %ld prep operations failed in %s\n", failed, state.name().c_str());
            abort();
        } // if the prep did not build the population it describes
        report_prep(state);
        handles.make(*set, state.threads());   // after the prep's advance() (see ThreadHandles)
    } // ChurnFixture::SetUp()
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
private:
    // Prints the arena state the timed region starts from, once per
    // configuration and process (it is the same for every repetition and
    // thread count: the prep is deterministic): slots ever appended, nodes
    // reachable from the buckets and how many of those are dead (erased or
    // superseded nodes still in their chains: what an erase or a split left
    // behind), free-list, limbo and retired-list totals (retired: dead nodes the
    // prep's operations unlinked, which only an advance() makes reusable), the
    // sweep's consistency verdict, the shard count, the mean free-list length
    // per shard (free_per_shard: the free total over the shard count), the
    // shortest and the longest shard's list, the nodes in bags (closed by an
    // advance() and not yet freed: 0 unless a handle pinned them) and the
    // table size. To stdout, so the line lands in the benchmark output
    // before the first row that uses the configuration (Google Benchmark
    // writes its console report to stdout too). The accounting sweep sorts
    // every slot; a few hundred milliseconds, once, untimed.
    static void report_prep(const ::benchmark::State& state) {
        static bool reported = false;
        if (reported) return;
        reported = true;
        const typename SetType::InternalAccounting a = set->get_internal_accounting();
        const size_t shards = set->get_internal_arena_shards();
        const auto [free_min, free_max] = std::minmax_element(a.free_per_shard.begin(), a.free_per_shard.end());
        printf("# prep %s: slots=%zu reachable=%zu reachable_dead=%zu free=%zu limbo=%zu retired=%zu consistent=%d shards=%zu free_per_shard=%zu"
               " free_shard_min=%zu free_shard_max=%zu bagged=%zu table_size=%zu\n",
               state.name().c_str(), a.slots, a.reachable, a.reachable_dead, a.free_nodes, a.limbo, a.retired, int(a.consistent),
               shards, a.free_nodes/shards, *free_min, *free_max, a.bagged, set->get_internal_table_size());
        fflush(stdout);
    } // ChurnFixture::report_prep()
}; // class ChurnFixture
template <typename Cfg> typename Cfg::Set* ChurnFixture<Cfg>::set = nullptr;
template <typename Cfg> ThreadHandles<typename Cfg::Set> ChurnFixture<Cfg>::handles;

// Thread 0's prefill in the SetUp() of EraseFixture (the _erase unit) and
// WindowFixture (the _churn unit): a new
// set of `buckets` buckets (and the arena shards of arena_shards_from_env()
// and the automatic cadence `auto_reclaim_every`, 0 = off: new_set()), into
// which `for_each_key(insert)` inserts its keys through `insert(key)`, under
// a handle of the prefill's own, released before this returns (before the
// threads' handles are made). The keys are distinct, so an insert that fails
// means the set does not hold the population the rows describe: the run
// aborts, naming `fixture` and the row. The prefill runs under the cadence
// too: a presized table splits nothing and so retires nothing, but a
// prefill that doubles its table may trigger, untimed, before the fixture's
// snapshot (ReclaimSnapshot's prefill_triggers says whether it did).
template <typename SetType, typename KeyVisitor>
SetType* make_prefilled(const ::benchmark::State& state, const char* fixture, size_t buckets, size_t auto_reclaim_every, KeyVisitor for_each_key) {
    SetType* set = new_set<SetType>(buckets, auto_reclaim_every);
    long failed = 0;
    {   // the prefill's own handle
        const typename SetType::handle h = set->new_handle();
        for_each_key([&](int key) { failed += !set->insert(h, key); });
    } // the prefill's handle
    if (failed != 0) {
        fprintf(stderr, "%s: %ld prefill inserts failed in %s\n", fixture, failed, state.name().c_str());
        abort();
    } // if the prefill did not build the population it describes
    return set;
} // make_prefilled()

// The thread counts ThreadRange(1, num_cpu) produces (1, the powers of two
// below num_cpu, num_cpu), each registered as its own row of benchmark class
// `Bench` with the per-thread count total/threads: a row whose run does the
// same `total` operations at every thread count, when the thread count
// divides it (InsertHotGrowth and the Erase_Small_Del rows). The
// registration call is the one BENCHMARK_REGISTER_F() expands to.
template <typename Bench>
void register_shared_total(int total) {
    for (int t = 1; ; t *= 2) {
        if (t >= num_cpu) t = num_cpu;
        benchmark::internal::RegisterBenchmarkInternal(std::make_unique<Bench>())
            ->Threads(t)->Iterations(total/t)->UseRealTime();
        if (t == num_cpu) break;
    } // loop over the thread counts
} // register_shared_total()

#endif // CONCURRENT_HASH_SET_RCU_BM_H
