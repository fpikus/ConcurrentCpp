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
#include <algorithm>
#include <chrono>
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

/*
 * ConcurrentResizableHashSetRCU Benchmarks
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
 *    at full capacity, so that no resize occurs during the timed region in
 *    either container up to the thread counts the table of contents gives
 *    (see also the caution in MostlyOldFixture::SetUp). The
 *    steady-state workload is 99% contains() with a ~50% hit rate (the
 *    lookup range is exactly twice the prefilled range, so misses exercise the
 *    negative-result revalidation path, which an all-hits workload never
 *    touches), and 1% insertion of brand-new keys on a deterministic schedule
 *    (every 100th operation), drawn from thread-disjoint ranges *above* the
 *    lookup range so the hit rate stays fixed for the whole run. The insert
 *    budget stays below either container's resize trigger up to the thread
 *    counts the table of contents gives. This benchmark measures what readers
 *    pay for coexisting with writers when nothing "happens"; what readers pay
 *    when a resize DOES happen is a tail-latency question and needs a
 *    tail-latency benchmark, not a mean.
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
 * ThreadHandles). The rows whose names end in _Rcu run under the RECLAMATION
 * REGIME: advance() is called from inside the timed loop -- once per thread at
 * its end, or by one thread at a cadence while every thread refreshes its
 * handle each iteration -- and the rows report what the calls did and what
 * they cost (see AdvanceTally). The rows whose names end in _RcuEach have
 * EVERY thread call advance() at a cadence of its own erases; the rows whose
 * names end in _Auto call nothing: their set is constructed with an AUTOMATIC
 * CADENCE (the header's Options), so the set itself runs the bump and the free
 * pass -- never the walk -- from inside the retiring thread's erase, once that
 * thread's arena shard has retired another R nodes; every thread still
 * refreshes its handle each iteration. Every reclamation row reports the set's
 * own counters of what the calls and the triggers did, as deltas across the
 * timed loop (ReclaimSnapshot), and the three churn reclamation rows print a
 * per-thread time series of every call or trigger return (the "# trace" lines,
 * Trace). The Advance_ rows measure advance() alone, one thread, nothing else
 * running.
 *
 * Table of contents: every registration, in the order of the file, each run
 * at the thread counts of ThreadRange(1, num_cpu) (one row per thread count,
 * all rows of a registration the same workload) except the Advance_ rows (one
 * thread, one row per geometry), with why it is measured. The
 * details are at each definition; the churn states (Control, Reclaimed,
 * Erased) are described above the ChurnFixture. A bucket count chosen so that
 * a table does not resize holds only up to some thread count: the pre-sized
 * concurrent table doubles once its arena holds about 2*kPresizedBuckets
 * nodes (64 threads' worth of timed inserts, fewer after a prefill); the
 * lookup table once it holds about 4*kPrefill, which no thread count up to
 * 256 reaches except in the Erased cell, whose arena keeps the erased
 * victims' slots (2*kPrefill in all after the prefill), so its table doubles
 * past about 200 threads. A baseline std::unordered_set rehashes once it
 * holds more keys than buckets: never at 32 threads or fewer in the
 * pre-sized cell, 100 or fewer in the lookup cell.
 *
 *  Insert_MostlyNew_Concurrent    pure insertion through live doublings and
 *                                 lazy splits: the growth path of this set.
 *  Insert_MostlyNew_RWLocked      the same inserts on the baseline: the price
 *                                 of a writer-exclusive rehash.
 *  Insert_MostlyNew_Presized_Concurrent
 *                                 the same inserts into a pre-sized table: the
 *                                 arena alone, with the doubling and split
 *                                 path subtracted out.
 *  Insert_MostlyNew_Presized_RWLocked
 *                                 the baseline pre-sized the same way: its
 *                                 lock alone.
 *  Lookup_MostlyOld_Concurrent    mostly lookups at a fixed hit rate with a
 *                                 trickle of inserts and no resize: what
 *                                 readers pay for coexisting with writers.
 *  Lookup_MostlyOld_RWLocked      the same on the baseline: the RMW every
 *                                 reader pays to be visible to a rehash.
 *  Lookup_MostlyOld_OpHandle      the lookup workload with a handle taken and
 *                                 released around every operation, where
 *                                 _Concurrent keeps one per thread: what a
 *                                 handle costs when it is not long-lived.
 *  Insert_MostlyNew_Del           the growth inserts on an EMPTY AllowDelete
 *                                 set: against _Concurrent, whether compiling
 *                                 erase() in costs the insert path anything.
 *  Insert_MostlyNew_Del_Control   the same on a set prefilled with live keys
 *                                 and empty free lists: a resident population
 *                                 with every allocation an append.
 *  Insert_MostlyNew_Del_Reclaimed the same population reached by churn and
 *                                 one advance(): inserts that pop freed slots
 *                                 in a table the larger prefill grew further --
 *                                 "churn, then growth", not an allocator cell.
 *  Insert_MostlyNew_Presized_Del  the pre-sized inserts on an EMPTY
 *                                 AllowDelete set: erase() compiled in, the
 *                                 arena alone.
 *  Insert_MostlyNew_Presized_Del_Control
 *                                 prefilled, no churn, the same buckets as the
 *                                 next two: the control for the free lists.
 *  Insert_MostlyNew_Presized_Del_Reclaimed
 *                                 churn and one advance() at fixed geometry:
 *                                 the free-list pop path and the locality of
 *                                 the popped slots, the controlled allocator
 *                                 cell.
 *  Insert_MostlyNew_Presized_Del_Erased
 *                                 churn without advance(): inserts whose scans
 *                                 walk chains the erases already unlinked
 *                                 from; what eager unlinking buys a writer.
 *  Lookup_MostlyOld_Del           the lookup workload on the AllowDelete set:
 *                                 against _Concurrent, erase() compiled in on
 *                                 the read path.
 *  Lookup_MostlyOld_Del_Control   the same prefill through the churn fixture:
 *                                 a check of the fixture against _Del.
 *  Lookup_MostlyOld_Del_Reclaimed the misses walk chains the victims were
 *                                 unlinked from, and the trickle inserts pop
 *                                 the slots advance() freed.
 *  Lookup_MostlyOld_Del_Erased    the misses walk chains the erases unlinked
 *                                 from, nothing reclaimed: what eager
 *                                 unlinking buys a reader.
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
 *  Churn_Window_Del_NoReclaim     steady churn: every thread slides a window
 *                                 of live keys (erase the oldest, insert the
 *                                 next, two lookups), nothing reclaimed, so
 *                                 every insert appends and the table grows.
 *  Churn_Window_Del_Rcu           the same under the reclamation regime, one
 *                                 advancer with the walk: the first of the
 *                                 three rows in which the free pass pushes
 *                                 freed nodes while other threads pop them.
 *  Churn_Window_Del_RcuEach       the same with every thread an advancer
 *                                 after every kWindowAdvanceEvery of its own
 *                                 erases (staggered as above): the explicit
 *                                 like-for-like pair of the next row, as the
 *                                 small pair above.
 *  Churn_Window_Del_Auto          steady churn with the automatic cadence at
 *                                 kWindowAdvanceEvery nodes retired per arena
 *                                 shard: whether a bump-and-free trigger on
 *                                 every thread reaches a steady state that
 *                                 one advancer with a walk did not (the table
 *                                 size, the slots per live key, the attempts
 *                                 proceeded against skipped, the free pass's
 *                                 share of the threads' time), and the
 *                                 per-call series of the three churn
 *                                 reclamation rows (the "# trace" lines: the
 *                                 time of every call or trigger return on
 *                                 every thread, so the concurrent phase and
 *                                 the tail are read from the data instead of
 *                                 bracketed from finish_spread).
 *  InsertHotGrowth_Concurrent     growth on a few hot chains, split while the
 *                                 threads still prepend to them: the cleanup
 *                                 pass's lost CASes and the stragglers they
 *                                 leave, counted; the throughput is not the
 *                                 point.
 *  InsertHotGrowth_Del            the same probe with erase() compiled in.
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

// Arena shard count for ConcurrentResizableHashSetRCU, from the environment:
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

// A new set for a fixture: `buckets` buckets, the arena shards of
// arena_shards_from_env(), and the automatic cadence `auto_reclaim_every`
// (nodes retired per arena shard between two automatic reclaims; 0 = off,
// which is every row but the _Auto rows). Every fixture constructs its set
// here, through the set's Options form, so that a row with a cadence differs
// from its twin in one argument and nothing else (the two-parameter
// constructor is the same with the cadence off).
template <typename SetType>
static SetType* new_set(size_t buckets, size_t auto_reclaim_every = 0) {
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
            if (++done_ == last_) [[unlikely]] {
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
// One thread's per-call time series in a churn reclamation row: the "# trace"
// lines.
//
// Why: the rates and the summed timers say how much a row's reclamation cost
// and freed, not WHEN: with one advancer the churn row has a concurrent
// phase and a tail in which the advancer walks alone, and the phases could
// only be bracketed from finish_spread; with every thread a caller or a
// trigger the question is whether the free passes run back to back. So each
// thread of the three churn reclamation rows keeps a local array of up to
// kEntries entries, one per return of its own advance() calls (every
// outcome, `contended` included: an attempt is an attempt) or of the
// triggers it detected, and thread 0 of the _Auto row also one per
// kWindowAdvanceEvery of its own erases (a sample, since it makes no calls):
// the steady_clock time of the return (ns, relative to the thread's own
// stamp taken before the loop), the kind (a letter: the outcome of the call,
// `t` a detected trigger, `s` a sample), the nodes freed so far (the thread's
// own calls' on the explicit rows; the set-wide count on the _Auto row,
// whose triggers belong to no thread's tally) and the table size. The array
// is cold: on the stack, uninitialized, touched only when an entry is
// written, once per cadence. Entries past kEntries are dropped and COUNTED
// (thread 0 of the churn _Auto row makes 32 samples per repetition beside
// its own triggers, which can exceed the array); the attempts by kind are
// counted whether or not the entry was kept, so that a dropped trigger still
// counts in the identity "detected triggers == the set's auto_reclaims".
// After the loop, untimed, every thread prints its line, prepared in one
// buffer and written with one fputs() (lines from T threads interleave in
// the console, never within a line):
//   # trace <row> threads=<T> thread=<i> start_ns=<absolute steady_clock ns
//   of the thread's stamp before the loop> bump_ns=<..> free_ns=<..>
//   walk_ns=<..> n=<entries kept> dropped=<d>
//   kinds=a:<..>,n:<..>,c:<..>,r:<..>,t:<..>,s:<..> | <t>:<kind>:<freed>:<table> ...
// (the kinds are the attempts by kind, kept and dropped; `t` of an entry is
// relative to start_ns, so start_ns + t aligns the threads of one
// repetition). start_ns is the entries' reference only: it is taken before
// the loop's start barrier, so it includes the thread's spawn skew and is
// NOT the start of the thread's loop. The thread's loop is bracketed by the
// wall-clock stamps the loop itself takes (gb_wall_clock.h: the end of the
// thread's first and of its last iteration), which thread 0 prints for
// every thread of the run, after the loop (report_stamps(); thread 0 may
// read every thread's record then, and no other thread may read its own
// after the loop, since thread 0 releases the records at the end of its
// body):
//   # stamps <row> threads=<T> first_ns=<f0>,<f1>,... last_ns=<l0>,<l1>,...
// (absolute steady_clock ns, in thread order): the per-thread phases --
// first finish, last finish, the free passes' progress against each
// thread's own loop -- are read from these two lines together. The lines
// precede the row's console line, as "# prep" lines do, so a reducer pairs
// them with the row by order (and splits them at " threads=": the _Auto
// rows' names contain a space). The record costs one steady_clock read per
// entry, once per cadence, plus two loads on the explicit rows and the
// set-wide counter sum on the _Auto row (see there): negligible, and absent
// from the controls.
// Trivially destructible, like the other body locals.
// ---------------------------------------------------------------------------
class Trace {
public:
    static constexpr int kEntries = 64;
    // What an entry records; the letter printed is the enumerator's.
    enum class Kind : uint8_t { advanced, nothing_retired, contended, ring_full, trigger, sample, kinds };
    static constexpr int kKinds = int(Kind::kinds);
    static constexpr char kLetters[kKinds + 1] = "ancrts";

    // Stamps the thread's start (before the loop; the entries' times are
    // relative to it).
    Trace() : start_(now_ns()) {}

    // The kind of an advance() return.
    static Kind kind_of(AdvanceResult::Advance a) {
        switch (a) {
        case AdvanceResult::Advance::advanced:        return Kind::advanced;
        case AdvanceResult::Advance::nothing_retired: return Kind::nothing_retired;
        case AdvanceResult::Advance::contended:       return Kind::contended;
        case AdvanceResult::Advance::ring_full:       return Kind::ring_full;
        } // switch on the outcome
        return Kind::ring_full;   // not reached: every enumerator is listed
    } // Trace::kind_of()

    // Inside the loop: one entry, at the return it records (`freed` and
    // `table` as described above; both fit 32 bits in every row of this
    // file, and are stored so to keep an entry at 24 bytes).
    void record(Kind kind, size_t freed, size_t table) {
        const int64_t t = now_ns() - start_;
        ++attempts_[int(kind)];
        if (n_ == kEntries) {
            ++dropped_;
            return;
        } // the array is full: counted, not kept
        entries_[n_] = {t, uint32_t(freed), uint32_t(table), kLetters[int(kind)]};
        ++n_;
    } // Trace::record()

    // After the loop, on every thread, untimed: this thread's "# trace" line
    // (above), with this thread's timers `t` (the tally's), written with one
    // fputs().
    template <typename Timers>
    void report(const benchmark::State& state, const Timers& t) const {
        // The buffer: a header of under 500 characters (the row's name is
        // up to about 120 of them) and at most kEntries entries of at most
        // 37 ("<t>:<k>:<freed>:<table> " with a 12-digit t and two 10-digit
        // counts); an overflow is cut, never overrun, and the cut line still
        // ends in a newline.
        char buf[4096];
        size_t len = 0;
        const auto put = [&](const char* fmt, auto... args) {
            if (len >= sizeof buf) return;
            const int n = snprintf(buf + len, sizeof buf - len, fmt, args...);
            len = n < 0 ? sizeof buf : std::min(len + size_t(n), sizeof buf);
        }; // put(): appends one formatted piece
        put("# trace %s threads=%d thread=%d start_ns=%lld bump_ns=%zu free_ns=%zu walk_ns=%zu n=%d dropped=%u kinds=",
            state.name().c_str(), state.threads(), state.thread_index(), (long long)start_,
            t.bump_ns, t.free_ns, t.walk_ns, n_, dropped_);
        for (int k = 0; k < kKinds; ++k) put("%s%c:%u", k == 0 ? "" : ",", kLetters[k], attempts_[k]);
        put("%s", " |");   /* "%s": a format with no argument trips -Wformat-security */
        for (int i = 0; i < n_; ++i) put(" %lld:%c:%u:%u", (long long)entries_[i].t, entries_[i].kind, entries_[i].freed, entries_[i].table);
        put("%s", "\n");
        if (len >= sizeof buf) buf[sizeof buf - 2] = '\n';   // cut: the line still ends in a newline
        buf[sizeof buf - 1] = '\0';
        fputs(buf, stdout);
        fflush(stdout);
    } // Trace::report()

    // After the loop, on thread 0 only, untimed, before the records are
    // released: the "# stamps" line (above) from every thread's wall-clock
    // record of this run.
    static void report_stamps(const benchmark::State& state, const WallRecords& records) {
        std::string first, last;
        for (int i = 0; i < state.threads(); ++i) {
            const WallStamps& s = records.of(i);
            first += (i == 0 ? "" : ",") + std::to_string(ns_of(s.first));
            last += (i == 0 ? "" : ",") + std::to_string(ns_of(s.last));
        } // loop over the threads' records
        printf("# stamps %s threads=%d first_ns=%s last_ns=%s\n", state.name().c_str(), state.threads(), first.c_str(), last.c_str());
        fflush(stdout);
    } // Trace::report_stamps()

private:
    // A steady_clock time point in ns since its epoch.
    static int64_t ns_of(std::chrono::steady_clock::time_point tp) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    } // Trace::ns_of()
    // steady_clock now, in ns since its epoch.
    static int64_t now_ns() { return ns_of(std::chrono::steady_clock::now()); }

    struct Entry {
        int64_t t;        // ns after start_
        uint32_t freed;   // nodes freed so far (see above)
        uint32_t table;   // the table size
        char kind;        // the kind's letter
    }; // struct Trace::Entry
    static_assert(sizeof(Entry) == 24, "an entry is meant to be 24 bytes");

    const int64_t start_;             // the thread's stamp before the loop (before the start barrier: the entries' reference, not the loop's start)
    int n_ = 0;                       // entries kept
    uint32_t dropped_ = 0;            // entries dropped (the array was full)
    uint32_t attempts_[kKinds] = {};  // entries by kind, kept and dropped alike
    Entry entries_[kEntries];         // the kept entries; uninitialized on purpose (cold)
}; // class Trace
static_assert(std::is_trivially_destructible_v<Trace>, "a body local with a destructor would be live across the state loop");

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
static int advance_every_from_env(const char* name, int registered) {
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
             * up to the thread count the table of contents (at the top of
             * this file) gives: every bucket is born EMPTY, no split ever
             * exists, no stale copies inflate the chains. The unordered_set
             * baseline gets the same courtesy (2*kPrefill buckets, so no
             * rehash either, up to the thread count the table of contents
             * gives for it).
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

// ---------------------------------------------------------------------------
// Reclamation benchmarks.
//
// The three workloads above are re-run on the AllowDelete == true instantiation
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
//                    Erase_Presized_Del below measures what it costs the
//                    eraser.
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
// `set` exactly as they see the other fixtures'.
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
// window rows' (see Churn_Window_Del). Why measure: what an erase costs when
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

// Thread 0's prefill in the SetUp() of EraseFixture and WindowFixture: a new
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
static SetType* make_prefilled(const ::benchmark::State& state, const char* fixture, size_t buckets, size_t auto_reclaim_every, KeyVisitor for_each_key) {
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

// The erase fixture: the geometry `Geom` (above) and the automatic cadence
// policy `Cadence` (NoCadence above, or an _Auto row's); the cadence is a
// template parameter so that an _Auto row is a fixture of its own with its
// own set, and the kept rows' names (the fixture's written template
// arguments) do not change.
template <typename SetType, typename Geom = PresizedErase, typename Cadence = NoCadence>
class EraseFixture : public benchmark::Fixture {
public:
    using Geometry = Geom;    // the bodies read the per-thread key count through it
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles above)
    using Tally = AdvanceTally<SetType>;   // one thread's advance() calls or triggers in the reclamation bodies (AdvanceTally above)
    static ReclaimSnapshot<SetType> before;   // the container's counters after the prefill (ReclaimSnapshot above)
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

// The body of the erase rows (see above), as a macro like the bodies above, so
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

// ---------------------------------------------------------------------------
// Churn_Window_Del: steady churn, without and with the reclamation regime.
//
// Why measure: the rows above make freed nodes available once, before the
// timed region (the Reclaimed states), or call advance() in rows that do not
// allocate (Erase_Presized_Del_Rcu, once per thread at its end;
// Erase_Small_Del_Rcu, at a cadence); none frees nodes while other threads
// allocate.
// Here every thread keeps a fixed number of live keys and replaces them
// continuously, so under the regime the free pass pushes the nodes it frees
// onto the free lists while the threads pop them for their inserts -- the free
// lists as a concurrent recycler, which is what reclamation beside the
// operations is for -- and without it every insert appends, and the arena and
// the table grow with the run. The pair says what the regime costs a churning
// workload and what it saves it.
// Shape: thread t owns the keys mix(t*kKeyStride + i), i = 0, 1, 2, ...; at
// any moment the kWindow keys with i in [lo, lo + kWindow) are live. Thread
// 0's SetUp() inserts every thread's first window (lo = 0), untimed, into a
// table of kWindowBuckets buckets (one key per bucket at 16 threads). Every
// thread then repeats a cycle of four iterations: erase the key at lo; insert
// the key at lo + kWindow (the window slides by one); look up two keys drawn
// at random from [lo - kWindow, lo + kWindow), of which half are live (those
// below lo were erased, or never inserted). kWindowIters iterations per
// thread: kWindowIters/(4*kWindow) passes over the window. Every erase and
// every insert must return true (checked per thread, after the loop).
// _NoReclaim: nothing else. Nothing is freed, so the retired lists and the
//   arena grow by one node per insert, and the table doubles as the arena
//   grows (the resize hint counts every slot that is not on a free list).
// _Rcu: the reclamation regime. Every thread refreshes its handle at the top
//   of every iteration, and thread 0 calls advance() after every
//   kWindowAdvanceEvery of its own erases (AdvanceTally says why one
//   advancer): the same number of calls per run at every thread count, each
//   walking the threads*kWindow live keys. The nodes a call frees are popped
//   by every thread's inserts: from the inserting thread's own shard's list
//   first, then from the others' (the steal).
// _RcuEach: the same with EVERY thread calling advance() after every
//   kWindowAdvanceEvery of its own erases, the first calls staggered
//   (AdvanceTally): the attempt frequency of _Auto, called explicitly and
//   walking. Most calls at T >= 8 are expected `contended`.
// _Auto: the set constructed with the automatic cadence kWindowAdvanceEvery
//   (nodes retired per arena shard: WindowAutoCadence), no call: the set runs
//   the bump and the free pass from inside the erase whose retire push
//   crossed the cadence, on that thread, never walking. Every thread still
//   refreshes at the top of every iteration and releases on its last. A
//   thread detects its own triggers by reading its own bump_ns (a
//   thread_local the set charges a proceeding trigger's bump to) just before
//   the erase and comparing after it: one thread-local load before the erase,
//   one load, a compare and a predictable not-taken branch after it, the
//   row's stated added per-erase cost beside the trigger's own (reading the
//   set's auto_reclaims per erase would be a shared-line load on every
//   thread, absent from the control), plus the sampling countdown of the
//   "# trace" series (AdvanceTally::due(): a decrement, a compare and a
//   predictable branch per erase on EVERY thread, as in the _Rcu row, though
//   only thread 0's ever reaches zero; the control carries neither). A
//   skipped trigger reads no clock and
//   is not detected; a trigger inside an insert's or a lookup's split cleanup
//   (a table that doubled) is outside the bracket: the set's auto_reclaims
//   counts all of them, detected_triggers the ones inside erases.
// Counters, read by thread 0 after the loop (every thread has left it), untimed:
//   table_size         -- the bucket count at the end;
//   slots_per_live_key -- arena slots ever appended, over the live keys at the
//                         end (threads*kWindow): the memory the run kept;
//   pops_raced_push, free_push_cas_failures
//                      -- the container's diagnostic counters over the timed
//                         loop: free-list pops whose CAS lost to a push of the
//                         free pass (or failed spuriously), and pushes whose
//                         CAS lost to a pop (or failed spuriously): the
//                         witnesses that frees and pops met;
//   retired_nodes      -- nodes retired over the timed loop (every erase of a
//                         present key retires at least its node, by the
//                         eraser or a later walker, and split copies add to
//                         it: at least threads*kWindowIters/4): the
//                         denominator of "freed against retired";
// on the reclamation rows the AdvanceTally and the ReclaimSnapshot counters,
// and their threads' "# trace" lines (Trace); on _Auto the "# shards" line.
// ---------------------------------------------------------------------------
static constexpr int    kWindow        = 1 << 12;
static constexpr size_t kWindowBuckets = size_t(1) << 16;
static constexpr int    kWindowIters   = 1 << 20;
// Churn_Window_Del_Rcu's cadence: thread 0 calls advance() after every
// kWindowAdvanceEvery of its own erases, two passes over its window
// (HASH_BM_WINDOW_ADVANCE_EVERY overrides it: advance_every_from_env()).
// _RcuEach: every thread's cadence of its own erases; _Auto: the set's
// cadence in nodes retired per arena shard (one thread per shard at the
// threads' equal rates: the same per-thread frequency), and thread 0's
// sampling cadence for the "# trace" series.
static constexpr int    kWindowAdvanceEvery = 2*kWindow;
// The automatic cadence of Churn_Window_Del_Auto (see NoCadence): the twin's
// cadence, with the same environment override.
struct WindowAutoCadence {
    static size_t auto_reclaim_every(const benchmark::State& /*state*/) {
        return size_t(advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery));
    } // WindowAutoCadence::auto_reclaim_every()
}; // struct WindowAutoCadence

// The window fixture: the automatic cadence policy `Cadence` as in
// EraseFixture (NoCadence for every row but _Auto; the kept rows' names keep
// the one written template argument).
template <typename SetType, typename Cadence = NoCadence>
class WindowFixture : public benchmark::Fixture {
public:
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles above)
    using Tally = AdvanceTally<SetType>;   // one thread's advance() calls or triggers in the reclamation bodies (AdvanceTally above)
    static ReclaimSnapshot<SetType> before;   // the container's counters after the prefill (ReclaimSnapshot above)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() != 0) return;
        const int threads = state.threads();
        set = make_prefilled<SetType>(state, "WindowFixture", kWindowBuckets, Cadence::auto_reclaim_every(state), [threads](auto insert) {
            for (int t = 0; t < threads; ++t) {
                for (int i = 0; i < kWindow; ++i) insert(mix(t*kKeyStride + i));
            } // every thread's first window
        });
        before.take(*set);
        handles.make(*set, state.threads());
    } // WindowFixture::SetUp()
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class WindowFixture
template <typename SetType, typename Cadence> SetType* WindowFixture<SetType, Cadence>::set = nullptr;
template <typename SetType, typename Cadence> ThreadHandles<SetType> WindowFixture<SetType, Cadence>::handles;
template <typename SetType, typename Cadence> ReclaimSnapshot<SetType> WindowFixture<SetType, Cadence>::before;

// The body of the window rows (see above), as a macro for the same reason as
// DEFINE_ERASE: PRE declares what the regime needs before the loop, BEFORE
// runs at the top of every iteration, BEFORE_ERASE immediately before every
// erase (the _Auto row's first read of its bump_ns; empty in every other
// row), AFTER_ERASE after every erase, END at the end of every iteration
// (the release on the last one), POST reports after the loop; all six are
// empty in _NoReclaim. The fixture's template arguments come last, as in
// DEFINE_ERASE. Same accounting contract as the other bodies.
#define DEFINE_WINDOW(NAME, PRE, BEFORE, BEFORE_ERASE, AFTER_ERASE, END, POST, ...) \
    BENCHMARK_TEMPLATE_DEFINE_F(WindowFixture, NAME, __VA_ARGS__)             \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        XorShift32 rng(0x1234567u + 0x9e3779b9u*state.thread_index());        \
        const int base = state.thread_index()*kKeyStride;                     \
        int lo = 0;                                                           \
        int phase = 0;                                                        \
        int64_t ok = 0, tries = 0;                                            \
        bool hit = false;                                                     \
        Handles::Own own(handles, state);                                     \
        PRE                                                                   \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            BEFORE                                                            \
            if (phase == 0) {                                                 \
                ++tries;                                                      \
                BEFORE_ERASE                                                  \
                ok += set->erase(own.get(), mix(base + lo));                  \
                AFTER_ERASE                                                   \
            } else if (phase == 1) {                                          \
                ++tries;                                                      \
                ok += set->insert(own.get(), mix(base + lo + kWindow));       \
                ++lo;                                                         \
            } else {                                                          \
                hit = set->contains(own.get(),                                \
                    mix(base + lo - kWindow + int(rng.next() & (2*kWindow - 1)))); \
                benchmark::DoNotOptimize(hit);                                \
            }                                                                 \
            phase = (phase + 1) & 3;                                          \
            END                                                               \
        }                                                                     \
        if (ok != tries) {                                                    \
            state.SkipWithError("erase() or insert() of a window key returned false: the keys are this thread's alone"); \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        report_wall(state, records, double(state.iterations())*state.threads()); \
        if (state.thread_index() == 0) {   /* the container's diagnostic counters (see the fixture) */ \
            const ConcurrentSetDel::InternalCounters c = set->get_internal_counters(); \
            state.counters["table_size"] = benchmark::Counter(double(set->get_internal_table_size())); \
            state.counters["slots_per_live_key"] = benchmark::Counter(        \
                double(set->get_internal_node_count())/(double(state.threads())*kWindow)); \
            state.counters["pops_raced_push"] = benchmark::Counter(double(c.pops_raced_push - before.counters.pops_raced_push)); \
            state.counters["free_push_cas_failures"] = benchmark::Counter(double(c.free_push_cas_failures - before.counters.free_push_cas_failures)); \
            state.counters["retired_nodes"] = benchmark::Counter(double(c.retire_nodes - before.counters.retire_nodes)); \
        }                                                                     \
        POST                                                                  \
        records.release(state);                                               \
    }                                                                         \
    BENCHMARK_REGISTER_F(WindowFixture, NAME)                                 \
        ->ThreadRange(1, num_cpu)->Iterations(kWindowIters)->UseRealTime();

// ---------------------------------------------------------------------------
// InsertHotGrowth: growth on hot chains, a probe of the split's cleanup pass.
//
// Why measure: the growth cells above mix() every key, so their chains are a
// few nodes long at every doubling and a split's cleanup pass hardly ever
// meets a writer on the parent chain it walks. This cell builds the opposite
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
// nothing. The cell is registered on both instantiations, as the cells above
// are.
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

// The thread counts ThreadRange(1, num_cpu) produces (1, the powers of two
// below num_cpu, num_cpu), each registered as its own row of benchmark class
// `Bench` with the per-thread count total/threads: a row whose run does the
// same `total` operations at every thread count, when the thread count
// divides it (InsertHotGrowth and the Erase_Small_Del rows). The
// registration call is the one BENCHMARK_REGISTER_F() expands to.
template <typename Bench>
static void register_shared_total(int total) {
    for (int t = 1; ; t *= 2) {
        if (t >= num_cpu) t = num_cpu;
        benchmark::internal::RegisterBenchmarkInternal(std::make_unique<Bench>())
            ->Threads(t)->Iterations(total/t)->UseRealTime();
        if (t == num_cpu) break;
    } // loop over the thread counts
} // register_shared_total()

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
// buckets, which include the bucket counts of the rows above (2^16: the window
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

// ---------------------------------------------------------------------------
// Registrations: identical workloads, two containers.
// ---------------------------------------------------------------------------
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW(Insert_MostlyNew_RWLocked,   LockedSet)

DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_RWLocked,   LockedSet)

DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_OLD(Lookup_MostlyOld_RWLocked,   LockedSet)

// Lookup_MostlyOld_OpHandle: Lookup_MostlyOld_Concurrent with every operation
// under a handle of its own, taken just before it and released when it ends
// (DEFINE_MOSTLY_OLD_UNDER's HANDLE). Why measure: the rows above keep one
// long-lived handle per thread, whose join happens once, untimed; a caller
// that cannot keep a handle pays a join and a leave per operation instead.
// The join is a seq_cst increment of the current generation's handle count
// and a seq_cst re-load of the current generation, the leave a release
// decrement of the same count, so every thread's every operation writes the
// one line that count is on; each join also increments the set's diagnostic
// join counter (relaxed), one more line that every thread writes. The threads'
// long-lived handles are still made and destroyed as in every row, unused.
// Only on the AllowDelete == false set, against _Concurrent.
DEFINE_MOSTLY_OLD_UNDER(MostlyOldFixture, Lookup_MostlyOld_OpHandle, ConcurrentSet, set->new_handle())

// The reclamation cells (see the section above): the AllowDelete == true set
// empty, prefilled, and prefilled by churn, for each of the three workloads.
// Nothing here changes a registration above.
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Del, ConcurrentSetDel)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Del_Control,   NewControl)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Del_Reclaimed, NewReclaimed)

DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_Del, ConcurrentSetDel)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Control,   PresizedControl)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Reclaimed, PresizedReclaimed)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Erased,    PresizedErased)

DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Del, ConcurrentSetDel)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Control,   OldControl)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Reclaimed, OldReclaimed)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Erased,    OldErased)

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
    if (own.finish_iteration()) [[unlikely]] tally.add(set->advance());,
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
    if (tally.due()) tally.add(set->advance());
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
    if (tally.due()) tally.add(set->advance());
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
    if (timers.bump_ns != bump_before) [[unlikely]] tally.detected();
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

// The window rows (see Churn_Window_Del above).
DEFINE_WINDOW(Churn_Window_Del_NoReclaim,
    /* PRE */, /* BEFORE */, /* BEFORE_ERASE */, /* AFTER_ERASE */, /* END */, /* POST */,
    ConcurrentSetDel)
DEFINE_WINDOW(Churn_Window_Del_Rcu,
    /* PRE */
    Tally tally(advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery), state);
    Trace trace;,
    /* BEFORE */
    own.refresh();,
    /* BEFORE_ERASE */,
    /* AFTER_ERASE: advance() at the cadence, its return recorded */
    if (tally.due()) {
        const AdvanceResult r = set->advance();
        tally.add(r);
        trace.record(Trace::kind_of(r.advance), tally.freed_nodes(), set->get_internal_table_size());
    },
    /* END */
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    trace.report(state, Tally::timers());
    if (state.thread_index() == 0) {
        Trace::report_stamps(state, records);
        before.report_set(state, *set);
    },
    ConcurrentSetDel)
DEFINE_WINDOW(Churn_Window_Del_RcuEach,
    /* PRE */
    Tally tally(advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery), state, Tally::Advancer::every_thread);
    Trace trace;,
    /* BEFORE */
    own.refresh();,
    /* BEFORE_ERASE */,
    /* AFTER_ERASE: advance() at this thread's cadence, its return recorded */
    if (tally.due()) {
        const AdvanceResult r = set->advance();
        tally.add(r);
        trace.record(Trace::kind_of(r.advance), tally.freed_nodes(), set->get_internal_table_size());
    },
    /* END */
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    trace.report(state, Tally::timers());
    if (state.thread_index() == 0) {
        Trace::report_stamps(state, records);
        before.report_set(state, *set);
    },
    ConcurrentSetDel)
// _Auto: `freed` of a trace entry is the set-wide nodes_freed (the triggers
// belong to no thread's tally), read through get_internal_counters(), the
// one accessor that reaches it: a few hundred relaxed loads per entry, once
// per detected trigger or sample, inside the loop; small beside the free
// pass the trigger just ran, and stated.
DEFINE_WINDOW(Churn_Window_Del_Auto,
    /* PRE: no call; thread 0 samples every kWindowAdvanceEvery of its erases; this thread's timers, read around every erase */
    Tally tally(Tally::Automatic{}, advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery), state);
    Trace trace;
    Tally::Timers& timers = Tally::timers();,
    /* BEFORE */
    own.refresh();,
    /* BEFORE_ERASE */
    const size_t bump_before = timers.bump_ns;,
    /* AFTER_ERASE: the trigger detection, then thread 0's sample */
    if (timers.bump_ns != bump_before) [[unlikely]] {
        tally.detected();
        trace.record(Trace::Kind::trigger, set->get_internal_counters().nodes_freed, set->get_internal_table_size());
    }
    if (tally.due()) trace.record(Trace::Kind::sample, set->get_internal_counters().nodes_freed, set->get_internal_table_size());,
    /* END */
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    trace.report(state, Tally::timers());
    if (state.thread_index() == 0) {
        Trace::report_stamps(state, records);
        before.report_set(state, *set);
        before.report_shards(state, *set);
    },
    ConcurrentSetDel, WindowAutoCadence)

// The hot-growth probe (see InsertHotGrowth above), on both instantiations.
DEFINE_HOT_GROWTH(InsertHotGrowth_Concurrent, ConcurrentSet)
DEFINE_HOT_GROWTH(InsertHotGrowth_Del,        ConcurrentSetDel)

// The advance() geometry regression (see above), last.
[[maybe_unused]] static const bool advance_regression_registered = (register_advance_regression(), true);

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
