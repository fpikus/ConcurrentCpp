/*
 * atomic_shared_ptr_bm.C
 *
 * This benchmark measures the absolute raw, unadulterated concurrency throughput of
 * different atomic shared pointer implementations by stripping away data structure scaffolding.
 *
 * It uses a `benchmark::Fixture` with Google Benchmark's native `ThreadRange` barrier
 * to ensure threads start simultaneously, avoiding OS spawn latency serialization.
 *
 * Pointer types, by the suffix of the row name:
 *   _StdAtomic     std::atomic<std::shared_ptr> (StdAtomicSharedPtrAdapter)
 *   _IntrShared    intr_shared_ptr: intrusive count; a one-bit spinlock in the pointer
 *                  word covers the gap between reading the word and the AddRef
 *   _LockFree      parlay::atomic_shared_ptr (Daniel Anderson's, lock_free_shared_ptr/).
 *                  The suffix names that pointer, not a property the other rows lack:
 *                  _IntrSharedHP is lock-free too
 *   _IntrSharedHP  intr_shared_ptr_hp: intrusive count; the gap is covered by a hazard
 *                  pointer (Maged Michael's mm_hp) and a released pointee is destroyed
 *                  later, in a batch, by whichever thread's retire crosses mm_hp's
 *                  scan threshold
 *
 * Rows ReadHeavy, WriteHeavy and HighContention report Google Benchmark's standard
 * items/s; with several threads GB divides the summed item count by the MEAN per-thread
 * loop time, which flatters a pointer that lets some threads finish early while others
 * are parked. ReadersOneWriter reports its own per-thread rates (see that row).
 *
 * Instrumentation counters (cas_ok, cas_fail, destroyed and the intr_shared_ptr_hp
 * load-retry counters) are compiled in only with -DINTR_HP_BM_COUNTERS, which the Makefile
 * passes; the destruction-burst counters (scans, max_burst_us) only with -DINTR_HP_BM_BURSTS
 * as well, a separate build that is x86-64 only and whose rates are not used. See
 * "Instrumentation counters" below for what each means and the identities they obey.
 *
 * Performance Hierarchy (16 Threads) -- SUPERSEDED: measured before the _IntrSharedHP and
 * ReadersOneWriter rows and the counters existed; kept only until a new measurement of
 * every row replaces it. Do not quote.
 * 1. IntrShared (Fastest): ~17-32 Million ops/sec. Direct intrusive ref counting bypasses control blocks and hazard records.
 * 2. Parlay Hazard Ptr (Fast): ~14-30 Million ops/sec. Thread-local hazard records avoid control blocks but add slight store/fence overhead.
 * 3. StdAtomic (Slowest): ~2-3 Million ops/sec. Suffers massive cache line bouncing on external control block locks/atomics.
 */
#include <benchmark/benchmark.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>
#include <random>
#include <unistd.h>
#ifdef INTR_HP_BM_BURSTS
#include <x86intrin.h>   // __rdtsc, the burst clock (burst builds only, x86-64)
#endif

#include "atomic_shared_ptr_concept.h"
#include "intr_shared_ptr.h"
#include "intr_shared_ptr_hp.h"
#include "hp_drain.h"
#include "lock_free_shared_ptr/atomic_shared_ptr.hpp"

static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// ---------------------------------------------------------------------------
// Instrumentation counters
// ---------------------------------------------------------------------------
//
// Two levels, each compiled in only under its macro:
// - -DINTR_HP_BM_COUNTERS: event counters (thread_local increments and the
//   CAS result; the same macro compiles intr_shared_ptr_hp's own load-retry
//   counters). The Makefile builds the benchmark with it: a paired measurement
//   found its cost within the run-to-run spread.
// - -DINTR_HP_BM_BURSTS (requires INTR_HP_BM_COUNTERS): the timing of mm_hp's
//   destruction bursts, one TSC read per DataHP destruction. Not in the
//   standard build: the same paired measurement found it costs ~10% of a
//   one-thread WriteHeavy_IntrSharedHP iteration (~21 ns). Its counters come
//   from a separate pass of a burst build, whose rates are not used. x86-64
//   only (rdtsc); the other builds are portable.
// Without a macro its hooks below are empty functions or discarded
// `if constexpr` branches, and every row compiles to its uninstrumented body.
//
// Counters, per row, as Google Benchmark user counters (GB sums them over the
// row's threads, so each is a total for the run):
// - cas_ok, cas_fail: results of compare_exchange_strong in the CAS rows. The
//   rate counts iterations, i.e. attempts; the split shows how much of it was
//   wasted, and the `destroyed` identity needs both.
// - destroyed: pointee destructions (Data or DataHP) on each thread between
//   the start and the end of its benchmark body. Every CAS attempt creates one
//   pointee and, in the end, destroys one (the one it replaced, or its own
//   unused `desired`); the pointee created in SetUp() dies inside the run and
//   the last one installed survives it. Hence, for the pointers that destroy
//   the pointee at strong-count zero (std, intr, and parlay, which defers only
//   the release of its control block): destroyed == cas_ok + cas_fail on the
//   CAS rows, destroyed == stores on ReadersOneWriter. intr_shared_ptr_hp
//   destroys in mm_hp scans, so the difference is what is still pending in
//   mm_hp's retired list when the run ends: 0 <= cas_ok + cas_fail - destroyed
//   <= about the scan threshold (max(1000, 2 x hazard records)) plus one per
//   thread (objects found protected).
// - validation_mismatch, zero_count, revalidation_mismatch (intr_shared_ptr_hp
//   rows): load()'s three retry paths (see the pointer's header). A lock-free
//   load can retry under contention; these separate retries from per-load cost.
// Burst builds only (intr_shared_ptr_hp rows):
// - scans: bursts of DataHP destructions on one thread, a burst being a run
//   of destructions less than burst_gap_us apart. A burst is the deleter phase
//   of one mm_hp scan; destroyed/scans is the mean number of objects one scan
//   destroyed. A scan preempted for longer than the gap counts twice.
// - max_burst_us: the longest burst of the run, from its first destruction to
//   its last, in microseconds. Why: deferred reclamation turns into a latency
//   tail on whichever thread crosses the threshold (it runs ~1000 destructors
//   inside one retire), which no rate shows. Covers the deleter phase only,
//   not the membarrier and the hazard collection that precede it in the same
//   scan.

#ifdef INTR_HP_BM_COUNTERS
inline constexpr bool bm_counters = true;
#else
inline constexpr bool bm_counters = false;
#endif

#ifdef INTR_HP_BM_BURSTS
#ifndef INTR_HP_BM_COUNTERS
#error "INTR_HP_BM_BURSTS requires INTR_HP_BM_COUNTERS (mean burst size = destroyed/scans)"
#endif
inline constexpr bool bm_bursts = true;
#else
inline constexpr bool bm_bursts = false;
#endif

// Event counts of one thread, monotonic over the thread's lifetime; a benchmark
// body reports the deltas over its own run (BodyCounters). thread_local plain
// integers: no atomic RMW on the measured paths.
struct BmThreadCounts {
    unsigned long cas_ok = 0;      // compare_exchange_strong returned true
    unsigned long cas_fail = 0;    // compare_exchange_strong returned false
    unsigned long destroyed = 0;   // Data/DataHP destructors run on this thread
    unsigned long scans = 0;       // DataHP destruction bursts started on this thread (burst builds)
}; // struct BmThreadCounts
thread_local BmThreadCounts bm_counts;

// Burst timing. The interface (the four functions) exists in every build; in
// a build without INTR_HP_BM_BURSTS the functions do nothing, so that the
// callers need no preprocessor conditionals and no TSC code is compiled.
#ifdef INTR_HP_BM_BURSTS

// The burst this thread is in (or was last in), in TSC ticks.
struct BurstState {
    uint64_t start = 0;            // first destruction of the burst
    uint64_t last = 0;             // latest destruction of the burst
}; // struct BurstState
thread_local BurstState burst_state;

// Longest burst of the current run on any thread, in TSC ticks. Reset by
// thread 0 in SetUp(), raised by the destructors themselves (CAS-max), read by
// thread 0 after its state loop. Raised on every destruction that extends a
// burst past the maximum, not when the burst ends: Google Benchmark has no
// barrier after the state loop that a "close the open burst" step could
// precede, so a burst still open when another thread's loop ends would race
// thread 0's read. The loop's end barrier orders every destruction made inside
// any thread's loop before thread 0's read. Own cache line: written by the
// scanning thread, read by every destruction.
alignas(64) std::atomic<uint64_t> max_burst_ticks{0};

// Destructions closer together than this belong to one burst. Consecutive
// deleter calls inside one scan are tens of ns apart: the longest scan of a
// run, 900-1000 DataHP destroyed, took 8-77 us at 1 thread and 26-56 us at 4
// and 16 threads (max_burst_us of the intr_shared_ptr_hp rows, three short
// runs, 0.05 s min_time, on a Ryzen 7940HS under WSL2). Two scans on the same
// thread are separated by at least ~1000 retirements of the whole process,
// each of which is a full iteration of some thread.
inline constexpr double burst_gap_us = 5.0;

// TSC ticks per microsecond, measured once at startup against steady_clock
// (20 ms). The TSC is the burst clock because it is read on every DataHP
// destruction and is the cheaper clock (on the machine above, reading it back
// to back in a loop: rdtsc 7.3 ns, steady_clock::now() 18.1 ns). It assumes an
// invariant TSC (constant_tsc).
static const double tsc_ticks_per_us = [] {
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const uint64_t c0 = __rdtsc();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const uint64_t c1 = __rdtsc();
    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    return static_cast<double>(c1 - c0)/std::chrono::duration<double, std::micro>(t1 - t0).count();
}();
static const uint64_t burst_gap_ticks = static_cast<uint64_t>(burst_gap_us*tsc_ticks_per_us);

// Called by DataHP's destructor (deferred reclamation): extends the current
// burst or starts a new one (++scans), then raises max_burst_ticks if this
// burst is now the longest of the run.
inline void count_burst() noexcept {
    const uint64_t now = __rdtsc();
    if (now - burst_state.last > burst_gap_ticks) {
        burst_state.start = now;
        ++bm_counts.scans;
    } // gap too long: a new burst
    burst_state.last = now;
    const uint64_t span = now - burst_state.start;
    uint64_t seen = max_burst_ticks.load(std::memory_order_relaxed);
    while (span > seen &&
           !max_burst_ticks.compare_exchange_weak(seen, span, std::memory_order_relaxed)) {
    } // CAS-max
} // count_burst()

// Thread 0, in SetUp(): a new run has no longest burst yet.
inline void reset_run_bursts() noexcept { max_burst_ticks.store(0, std::memory_order_relaxed); }

// Each thread, before its state loop: the first destruction of this run
// starts a new burst on this thread.
inline void reset_thread_burst() noexcept { burst_state = BurstState{}; }

// The longest burst of the current run, in microseconds.
inline double max_burst_us() noexcept {
    return static_cast<double>(max_burst_ticks.load(std::memory_order_relaxed))/tsc_ticks_per_us;
}

#else // !INTR_HP_BM_BURSTS: the same interface, doing nothing

inline void count_burst() noexcept {}
inline void reset_run_bursts() noexcept {}
inline void reset_thread_burst() noexcept {}
inline double max_burst_us() noexcept { return 0.0; }

#endif // INTR_HP_BM_BURSTS

// Called by every pointee destructor: counts it for `destroyed`.
inline void count_destruction() noexcept {
    if constexpr (bm_counters) {
        ++bm_counts.destroyed;
    }
}

// Records the result of one compare_exchange_strong for cas_ok/cas_fail. In an
// uninstrumented build the result is discarded, as the bodies did before.
inline void count_cas(bool succeeded) noexcept {
    if constexpr (bm_counters) {
        if (succeeded) {
            ++bm_counts.cas_ok;
        } else {
            ++bm_counts.cas_fail;
        }
    } // instrumented build
} // count_cas()

// intr_shared_ptr_hp's load-retry counters of this thread (zeros in an
// uninstrumented build, where the header does not define them).
struct HpRetryCounts {
    unsigned long validation_mismatch = 0;
    unsigned long zero_count = 0;
    unsigned long revalidation_mismatch = 0;
}; // struct HpRetryCounts
inline HpRetryCounts hp_retries_now() noexcept {
#ifdef INTR_HP_BM_COUNTERS
    return {intr_shared_ptr_hp_retries.validation_mismatch, intr_shared_ptr_hp_retries.zero_count,
            intr_shared_ptr_hp_retries.revalidation_mismatch};
#else
    return {};
#endif
} // hp_retries_now()

// True for a pointer type that declares `deferred_reclamation = true`
// (intr_shared_ptr_hp); false for the others, which do not declare it (a
// nested requirement, so the missing member is "false", not an error).
template <typename PtrType>
inline constexpr bool reclaims_deferred = requires { requires PtrType::deferred_reclamation; };
// The drains and the HP counters must not silently become no-ops (a misspelt
// member would make the detection false, not an error).
static_assert(reclaims_deferred<intr_shared_ptr_hp<int>>);
static_assert(!reclaims_deferred<intr_shared_ptr<int>>);

// One thread's counters over one benchmark body. Construct it right before the
// state loop: it snapshots this thread's monotonic counts and starts a fresh
// burst. report() writes the deltas as user counters (GB sums them over the
// threads); max_burst_us is written by thread 0 only, since it is a
// process-wide maximum and GB has no "max" aggregation. All of it is a no-op
// in an uninstrumented build.
class BodyCounters {
public:
    BodyCounters() noexcept {
        if constexpr (bm_counters) {
            start_ = bm_counts;
            retries_start_ = hp_retries_now();
            reset_thread_burst();
        }
    } // BodyCounters()

    template <typename PtrType>
    void report(benchmark::State& state) const {
        if constexpr (bm_counters) {
            state.counters["cas_ok"] = static_cast<double>(bm_counts.cas_ok - start_.cas_ok);
            state.counters["cas_fail"] = static_cast<double>(bm_counts.cas_fail - start_.cas_fail);
            state.counters["destroyed"] = static_cast<double>(bm_counts.destroyed - start_.destroyed);
            if constexpr (reclaims_deferred<PtrType>) {
                const HpRetryCounts retries = hp_retries_now();
                state.counters["validation_mismatch"] =
                    static_cast<double>(retries.validation_mismatch - retries_start_.validation_mismatch);
                state.counters["zero_count"] =
                    static_cast<double>(retries.zero_count - retries_start_.zero_count);
                state.counters["revalidation_mismatch"] =
                    static_cast<double>(retries.revalidation_mismatch - retries_start_.revalidation_mismatch);
                if constexpr (bm_bursts) {
                    state.counters["scans"] = static_cast<double>(bm_counts.scans - start_.scans);
                    if (state.thread_index() == 0) {
                        state.counters["max_burst_us"] = max_burst_us();
                    }
                } // burst build
            } // deferred reclamation: load retries, bursts
        } // instrumented build
    } // BodyCounters::report()

private:
    BmThreadCounts start_;         // this thread's counts when the body started
    HpRetryCounts retries_start_;  // this thread's load-retry counts when the body started
}; // class BodyCounters

// ---------------------------------------------------------------------------
// Pointee types
// ---------------------------------------------------------------------------

// The pointee of the std, intr and parlay rows. Its layout is the one these
// rows have always measured: intr_shared_ptr_hp's base class must not change
// it, hence the separate DataHP below.
struct Data {
    int value;
    std::atomic<int> ref_count{0};
    Data(int v) : value(v) {}
    ~Data() { count_destruction(); }   // `destroyed` (instrumented builds only)
    void AddRef() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    bool DelRef() { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const { return ref_count.load(std::memory_order_relaxed); }
};

// The pointee of the _IntrSharedHP rows: Data's payload and count plus the
// hazard pointer base intr_shared_ptr_hp requires (24 bytes at offset 0), so
// sizeof is 32 (Data: 8) and a `new DataHP` takes a 48-byte malloc chunk
// (Data: 32). The hooks are noexcept, as the pointer requires; AddRef relaxed,
// DelRef acq_rel, as Data's. TryAddRef is the reference form shared by every
// intr_shared_ptr_hp pointee in this repository.
struct DataHP : std::hazard_pointer_obj_base<DataHP> {
    int value;                       // payload
    std::atomic<int> ref_count{0};   // the intrusive strong count (0 until adopted)

    DataHP(int v) : value(v) {}

    // Runs inside an mm_hp scan, on whichever thread crossed the threshold, in
    // bursts of up to ~threshold objects; the burst counters exist only because
    // reclamation is deferred, so they are tied to the flag that says so.
    ~DataHP() {
        count_destruction();
        if constexpr (intr_shared_ptr_hp<DataHP>::deferred_reclamation) {
            count_burst();
        } // deferred reclamation: burst accounting (burst builds only)
    } // ~DataHP()

    void AddRef() noexcept { ref_count.fetch_add(1, std::memory_order_relaxed); }

    // TryAddRef(): increment the strong count if and only if it is nonzero.
    // Returns true iff it incremented; returns false iff it observed a count of
    // 0, in which case the count is left at 0 (an object at 0 is retired, or
    // about to be retired, and must never be revived). Required only by
    // intr_shared_ptr_hp; the other pointer policies never call it. Memory
    // orders, all load-bearing: the load that observes 0 is ACQUIRE; the CAS is
    // ACQUIRE on success and relaxed on failure; a failed CAS whose refreshed
    // value is 0 re-reads the count with an acquire load before returning false,
    // so EVERY observed 0 was read with acquire. Why: intr_shared_ptr_hp::load()
    // calls this on an object pinned only by a hazard pointer. An observed 0
    // must synchronize with the release sequence headed by the DelRef that
    // produced it, so that the loader's next acquire reload of the word is
    // guaranteed to see the store that unpublished the object; with a relaxed
    // zero-observation the loader can re-read the stale word forever (model
    // checked: livelock). The ACQUIRE on CAS success makes the loader's
    // post-increment re-validation of the word see a swing that released the
    // word's own reference. The value a successful CAS consumes need not have
    // been read with acquire: a failed CAS refreshes it with a relaxed read (only
    // a refreshed 0 is re-read with acquire), and other threads' RMWs may have
    // rewritten it. A CAS that succeeds with relaxed order on a value it read
    // from the release sequence headed by the swing's DelRef does not
    // synchronize with that DelRef, so the re-validation reload may return the
    // stale word and LockFreeList::~Node's walk can judge a node exclusive
    // (count 1) that this loader then owns with a stale next (model checked:
    // assertion failure). Every operation on the count is an RMW. Cost: nil on
    // x86-64; LDAR/LDAXR on aarch64.
    bool TryAddRef() noexcept {
        int count = ref_count.load(std::memory_order_acquire);
        while (count != 0) {
            if (ref_count.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return true;
            }
            // The failed CAS refreshed `count` with a relaxed read; a 0 seen
            // that way does not synchronize. Re-read it with acquire.
            if (count == 0) count = ref_count.load(std::memory_order_acquire);
        } // CAS loop while the count is nonzero
        return false;
    } // TryAddRef()

    // DelRef(): decrement, acq_rel; true iff this call made the 1 -> 0
    // transition (the pointer then retires the object).
    bool DelRef() noexcept { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const noexcept { return ref_count.load(std::memory_order_relaxed); }
}; // struct DataHP

template <typename PtrType>
typename PtrType::shared_ptr_type make_shared_data(int v) {
    if constexpr (std::is_same_v<PtrType, StdAtomicSharedPtrAdapter<Data>>) {
        return typename PtrType::shared_ptr_type(std::make_shared<Data>(v));
    } else if constexpr (std::is_same_v<PtrType, intr_shared_ptr<Data>>) {
        return typename PtrType::shared_ptr_type(new Data(v));
    } else if constexpr (std::is_same_v<PtrType, parlay::atomic_shared_ptr<Data>>) {
        return typename PtrType::shared_ptr_type(parlay::make_shared<Data>(v));
    } else if constexpr (std::is_same_v<PtrType, intr_shared_ptr_hp<DataHP>>) {
        return typename PtrType::shared_ptr_type(new DataHP(v));
    }
}

// Drains mm_hp's retired list (hp_drain.h) for a pointer type with deferred
// reclamation; a no-op for the others. Called by thread 0 only.
template <typename PtrType>
void drain_if_deferred() {
    if constexpr (reclaims_deferred<PtrType>) {
        drain_reclamation();
    }
}

template <typename PtrType>
class AtomicPtrFixture : public benchmark::Fixture {
public:
    static PtrType* ptr;

    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            // intr_shared_ptr_hp: start the run with an empty retired list, so
            // that objects an earlier run left pending neither land in this
            // run's scans nor skew its `destroyed` count. (GB calls SetUp() for
            // every run: each iteration-count probe and each thread count of a
            // row.) Within hp_drain.h's quiescence precondition: the other
            // threads of this run have not reached the state loop, and the
            // previous run's threads are joined.
            drain_if_deferred<PtrType>();
            reset_run_bursts();
            ptr = new PtrType(make_shared_data<PtrType>(0));
        }
    } // SetUp()

    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            delete ptr;
            ptr = nullptr;
            // intr_shared_ptr_hp: destroy what the run and the delete above
            // retired, AFTER the delete (the last installed pointee is retired
            // by it). OUTSIDE hp_drain.h's quiescence precondition: the other
            // threads may still be running the code after their state loop
            // (here only counter reporting, which retires nothing). If another
            // thread's scan held the drain's sentinel past the filler cap (over
            // 2 s), the drain would abort the binary. What a drain that returns
            // early leaves is drained by the next run of an intr_shared_ptr_hp
            // row (the same row's next run, else the next such row; the other
            // pointers' rows neither drain nor retire through mm_hp); after the
            // binary's last run it goes to mm_hp's scan at process exit.
            drain_if_deferred<PtrType>();
        }
    } // TearDown()
};

template <typename PtrType>
PtrType* AtomicPtrFixture<PtrType>::ptr = nullptr;

// Why the _IntrSharedHP rows (all three workloads below): they measure the HP
// tax against the spinlock's convoying. Mechanisms, not results:
// - per load, intr_shared_ptr_hp pays one locked RMW (TryAddRef's CAS) plus
//   plain stores to its hazard record and the acquire reloads of the word;
//   intr_shared_ptr pays two locked RMWs (the lock bit, AddRef) and the unlock
//   store, and under contention its waiters back off into nanosleep;
// - a released DataHP is destroyed later, in a batch of ~1000, inside the
//   retire of whichever thread crosses mm_hp's threshold (scans,
//   max_burst_us), so its cost lands on unrelated iterations;
// - Google Benchmark starts fresh threads for every run, and each fresh
//   thread's first load() takes a hazard record from mm_hp's global pool under
//   a 1-bit spinlock INSIDE the timed loop (once per thread per run).
// WriteHeavy and HighContention add allocator locality: intr frees the replaced
// Data at once, so the next `new` reuses a just-freed, cache-hot chunk; HP frees
// in batches, so the next `new DataHP` gets a chunk freed long ago.

#define DEFINE_BM_READHEAVY(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(AtomicPtrFixture, Name, PtrType)(benchmark::State& state) { \
        std::mt19937 rng(state.thread_index() + 42 + state.iterations()); \
        std::uniform_int_distribution<int> dist(0, 99); \
        BodyCounters counters; \
        for (auto _ : state) { \
            if (dist(rng) < 90) { \
                auto p = ptr->load(); \
                benchmark::DoNotOptimize(p); \
            } else { \
                auto expected = ptr->load(); \
                if (expected) { \
                    auto new_p = make_shared_data<PtrType>(expected->value + 1); \
                    count_cas(ptr->compare_exchange_strong(expected, new_p)); \
                } \
            } \
        } \
        state.SetItemsProcessed(state.iterations()); \
        counters.report<PtrType>(state); \
    } \
    BENCHMARK_REGISTER_F(AtomicPtrFixture, Name)->ThreadRange(1, num_cpu)->UseRealTime();

DEFINE_BM_READHEAVY(ReadHeavy_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_READHEAVY(ReadHeavy_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_READHEAVY(ReadHeavy_LockFree, parlay::atomic_shared_ptr<Data>)    // _LockFree = parlay
// ReadHeavy, HP: the fairness axis. 90% loads on one word is where the
// spinlock's readers convoy on the lock bit (and its occasional CAS waits
// behind them) while HP's loads never wait on each other; GB's mean-time rate
// hides who was locked out, which ReadersOneWriter below shows.
DEFINE_BM_READHEAVY(ReadHeavy_IntrSharedHP, intr_shared_ptr_hp<DataHP>)

#define DEFINE_BM_WRITEHEAVY(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(AtomicPtrFixture, Name, PtrType)(benchmark::State& state) { \
        std::mt19937 rng(state.thread_index() + 42 + state.iterations()); \
        std::uniform_int_distribution<int> dist(0, 99); \
        BodyCounters counters; \
        for (auto _ : state) { \
            if (dist(rng) < 10) { \
                auto p = ptr->load(); \
                benchmark::DoNotOptimize(p); \
            } else { \
                auto expected = ptr->load(); \
                if (expected) { \
                    auto new_p = make_shared_data<PtrType>(expected->value + 1); \
                    count_cas(ptr->compare_exchange_strong(expected, new_p)); \
                } \
            } \
        } \
        state.SetItemsProcessed(state.iterations()); \
        counters.report<PtrType>(state); \
    } \
    BENCHMARK_REGISTER_F(AtomicPtrFixture, Name)->ThreadRange(1, num_cpu)->UseRealTime();

DEFINE_BM_WRITEHEAVY(WriteHeavy_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_WRITEHEAVY(WriteHeavy_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_WRITEHEAVY(WriteHeavy_LockFree, parlay::atomic_shared_ptr<Data>)  // _LockFree = parlay
// WriteHeavy, HP: 90% load + allocate + CAS, so every iteration retires one
// DataHP (the replaced one or the unused `desired`): the batched reclamation
// and the allocator-locality effect are at their largest here (see above).
DEFINE_BM_WRITEHEAVY(WriteHeavy_IntrSharedHP, intr_shared_ptr_hp<DataHP>)

#define DEFINE_BM_HIGHCONTENTION(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(AtomicPtrFixture, Name, PtrType)(benchmark::State& state) { \
        BodyCounters counters; \
        for (auto _ : state) { \
            auto expected = ptr->load(); \
            if (expected) { \
                auto new_p = make_shared_data<PtrType>(expected->value + 1); \
                count_cas(ptr->compare_exchange_strong(expected, new_p)); \
            } \
        } \
        state.SetItemsProcessed(state.iterations()); \
        counters.report<PtrType>(state); \
    } \
    BENCHMARK_REGISTER_F(AtomicPtrFixture, Name)->Threads(16)->UseRealTime();

DEFINE_BM_HIGHCONTENTION(HighContention_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_HIGHCONTENTION(HighContention_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_HIGHCONTENTION(HighContention_LockFree, parlay::atomic_shared_ptr<Data>)  // _LockFree = parlay
// HighContention, HP: 16 threads in a load-CAS loop on one word. A failed CAS
// re-reads `expected` through the hazard-pointer load and retries; under the
// spinlock every step queues on the lock bit. Fairness again: cas_ok/cas_fail
// and the load-retry counters show where the attempts went.
DEFINE_BM_HIGHCONTENTION(HighContention_IntrSharedHP, intr_shared_ptr_hp<DataHP>)

// ---------------------------------------------------------------------------
// ReadersOneWriter: role-split row (thread 0 stores, every other thread loads)
// ---------------------------------------------------------------------------
//
// Why: the fairness axis, measured directly. A spinlock pointer can keep a high
// aggregate read rate by locking its writer out (readers hold the lock bit
// almost all the time); a lock-free pointer cannot starve the writer that way.
// The mixed rows above cannot show it: their rate is a sum over threads that
// all do the same mix.
//
// Why the stop-flag form: Google Benchmark runs every thread for the SAME
// number of iterations. A store (allocate, publish, release) is much slower
// than a load, so in the naive form the readers finish their quota in
// milliseconds and the writer then runs alone for the rest of the run; the
// readers' rate is then measured without a writer and the writer's without
// readers (a role-split probe with a 1 us writer and 10 ns readers: readers
// done in 3.9 ms, the writer alone for 2.79 s, a kIsRate loads/s 180x below
// the actual rate). Here each thread counts its own completed operations; the
// first thread to complete its quota sets `stop`, and every thread that sees
// `stop` turns its remaining iterations into no-ops. Each thread times its own
// window, from its first iteration to the moment it stopped. The windows END
// together (to within one operation: every thread stops at its first check of
// `stop` after it is set); they START when GB's start barrier releases each
// thread, which is not simultaneous, so a thread released early runs part of
// its window with fewer competitors.
//
// Counters, all PLAIN (GB sums them over threads; they are not kIsRate, which
// would divide by GB's time including the no-op tail, nor kAvgThreads, which
// divides by all threads, not by the threads of one role):
// - loads, stores: operations completed before stop;
// - loads_per_s, stores_per_s: each thread's count over its own window, summed
//   over threads, i.e. the aggregate rate of the readers and the writer's rate;
// - skipped: no-op iterations after stop; loads + stores + skipped equals GB's
//   `iterations` (the total over threads).
// SetItemsProcessed() is deliberately not called: GB's items/s would divide
// operations of two very different kinds by a time that includes the no-op
// tail. The Time column of these rows is meaningless for the same reason.
// The instrumentation counters (destroyed etc.) are reported as for the other
// rows; on these rows destroyed == stores for the std, intr and parlay pointers.

// The fixture adds the stop flag to AtomicPtrFixture.
template <typename PtrType>
class ReadersOneWriterFixture : public AtomicPtrFixture<PtrType> {
public:
    // Set by the first thread to complete its iteration quota. Reset by thread 0
    // in SetUp(), before the state loop's start barrier. Own cache line: every
    // iteration of every thread reads it.
    alignas(64) inline static std::atomic<bool> stop{false};

    void SetUp(const ::benchmark::State& state) override {
        AtomicPtrFixture<PtrType>::SetUp(state);
        if (state.thread_index() == 0) {
            stop.store(false, std::memory_order_relaxed);
        }
    } // SetUp()
}; // class ReadersOneWriterFixture

// One thread's measurement window in ReadersOneWriter. Call active() at the top
// of every iteration and complete_one() after every operation; report() after
// the state loop.
class StopFlagWindow {
public:
    // `stop`: the row's shared flag; `quota`: this thread's iteration count
    // (state.max_iterations).
    StopFlagWindow(std::atomic<bool>& stop, benchmark::IterationCount quota) noexcept
        : stop_(stop), quota_(quota) {}

    // True if this iteration should perform an operation; false once this thread
    // has stopped (its own quota done, or `stop` seen), counting the no-op.
    // The first call starts the window: it runs after GB's start barrier.
    bool active() noexcept {
        if (stopped_) {
            ++skipped_;
            return false;
        }
        if (!started_) {
            start_ = std::chrono::steady_clock::now();
            started_ = true;
        }
        if (stop_.load(std::memory_order_relaxed)) {
            end_ = std::chrono::steady_clock::now();
            stopped_ = true;
            ++skipped_;
            return false;
        } // another thread completed its quota: stop here
        return true;
    } // StopFlagWindow::active()

    // Counts one completed operation; on the last one of this thread's quota,
    // closes the window and tells the other threads to stop.
    void complete_one() noexcept {
        if (++done_ == quota_) {
            end_ = std::chrono::steady_clock::now();
            stopped_ = true;
            stop_.store(true, std::memory_order_release);
        }
    } // StopFlagWindow::complete_one()

    // Writes this thread's counters. `writer`: this thread is the row's writer
    // (thread 0), so its operations are stores, otherwise loads. The thread's
    // role gets its count and its rate over this thread's window, the other
    // role's two counters get 0 (every thread reports every counter), and
    // `skipped` the no-op iterations.
    void report(benchmark::State& state, bool writer) const {
        const double seconds = std::chrono::duration<double>(end_ - start_).count();
        const double ops = static_cast<double>(done_);
        const double rate = seconds > 0 ? ops/seconds : 0.0;
        state.counters["loads"] = writer ? 0.0 : ops;
        state.counters["loads_per_s"] = writer ? 0.0 : rate;
        state.counters["stores"] = writer ? ops : 0.0;
        state.counters["stores_per_s"] = writer ? rate : 0.0;
        state.counters["skipped"] = static_cast<double>(skipped_);
    } // StopFlagWindow::report()

private:
    std::atomic<bool>& stop_;                         // the row's stop flag
    const benchmark::IterationCount quota_;           // iterations GB runs on this thread
    benchmark::IterationCount done_ = 0;              // operations completed before stopping
    benchmark::IterationCount skipped_ = 0;           // no-op iterations after stopping
    bool started_ = false;                            // the window has started
    bool stopped_ = false;                            // the window has ended
    std::chrono::steady_clock::time_point start_{};   // first iteration
    std::chrono::steady_clock::time_point end_{};     // quota done or `stop` seen
}; // class StopFlagWindow

#define DEFINE_BM_READERSONEWRITER(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(ReadersOneWriterFixture, Name, PtrType)(benchmark::State& state) { \
        const bool writer = state.thread_index() == 0; \
        int version = 0; \
        StopFlagWindow window(stop, state.max_iterations); \
        BodyCounters counters; \
        for (auto _ : state) { \
            if (!window.active()) { \
                continue; \
            } \
            if (writer) { \
                ptr->store(make_shared_data<PtrType>(++version)); \
            } else { \
                auto p = ptr->load(); \
                benchmark::DoNotOptimize(p); \
            } \
            window.complete_one(); \
        } \
        window.report(state, writer); \
        counters.report<PtrType>(state); \
    } \
    BENCHMARK_REGISTER_F(ReadersOneWriterFixture, Name)->ThreadRange(2, num_cpu)->UseRealTime();

DEFINE_BM_READERSONEWRITER(ReadersOneWriter_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_READERSONEWRITER(ReadersOneWriter_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_READERSONEWRITER(ReadersOneWriter_LockFree, parlay::atomic_shared_ptr<Data>)  // _LockFree = parlay
DEFINE_BM_READERSONEWRITER(ReadersOneWriter_IntrSharedHP, intr_shared_ptr_hp<DataHP>)

BENCHMARK_MAIN();
