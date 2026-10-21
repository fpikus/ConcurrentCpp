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
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: the handle's own costs -- the
 * join and the leave alone, and the free pass's check of the handle counts
 * while other threads join and leave. One of the benchmark's translation
 * units (the overview of the workloads, the index of the units and why there
 * are several: concurrent_hash_set_rcu_bm.C); the code generated for these
 * rows depends on this unit alone, and these rows are in a unit of their own
 * so that they change the code of no other row (a unit's inlining depends on
 * all of its rows). Linked after every multi-threaded unit and before the
 * advance() unit, so that the advance() rows stay last.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  Handle_JoinLeave               new_handle() and the handle's destruction,
 *                                 nothing else, per iteration, at every
 *                                 thread count with the same total work:
 *                                 the per-operation handle's cost apart from
 *                                 an operation (Lookup_MostlyOld_OpHandle
 *                                 mixes it with a lookup's), and how it
 *                                 scales with the threads joining at once.
 *  Reclaim_FreePassCheck          one thread's reclaim() calls, each freeing
 *                                 what it can of a one-node bag, while W
 *                                 other threads join and leave without
 *                                 pause, at S arena shards, optionally with
 *                                 one never-refreshed handle pinning the
 *                                 oldest generation (P): what the free
 *                                 pass's test "no live handle in this
 *                                 generation" costs when the lines it reads
 *                                 are being written, per call and by the
 *                                 call's outcome.
 *
 * Every thread these rows run -- the measured threads and the helper threads
 * of Reclaim_FreePassCheck -- is a new thread (FreshThreadRunner below), and
 * each asks for its arena shard number (thread_number()) before its loop, so
 * that which threads share an arena shard is decided by the row alone, and is
 * reported by every row (the shard witnesses below).
 */

// ---------------------------------------------------------------------------
// Thread numbers and arena shards in these rows.
//
// The set stripes some per-thread state by the arena shard of the calling
// thread: its process-wide thread number (concurrent_hash_rcu_detail::
// thread_number(), assigned at the thread's first use, in increasing order)
// modulo the shard count. Two threads of a row on one shard share that
// shard's lines. Google Benchmark's default runner runs a row's thread 0 on
// the main thread, which is numbered at its first join or other use of an
// arena shard in the process (in a full run, during the first row's SetUp();
// under a filter, in whichever row ran first, or not yet), and its other
// threads on new threads, numbered consecutively from wherever the process's
// count has reached; so whether thread 0 shares a shard with another thread
// of the run depends on the rows that ran before in the same process. Here
// every thread of a run is new (FreshThreadRunner), and each takes its number
// before its loop, so a run of T threads holds T consecutive numbers and, for
// T <= S, T distinct shards, whatever ran before. thread_number() numbers a
// thread once, so the number a thread takes at the top of its body is the
// one its joins use.
// ---------------------------------------------------------------------------

// A Google Benchmark thread runner that runs every thread of a run on a new
// thread, thread 0 included, and joins them all (the default runner runs
// thread 0 on the calling thread: see above). Registered per row with
// ThreadRunner(); its RunThreads() is called every time Google Benchmark runs
// the row's threads (each repetition, and each of the warm-up runs when
// --benchmark_min_warmup_time is above 0).
class FreshThreadRunner final : public benchmark::ThreadRunnerBase {
public:
    // `threads`: the run's thread count, the row's Threads() value.
    explicit FreshThreadRunner(int threads) : threads_(threads) {}
    // Runs fn(0) .. fn(threads - 1), each on a new thread, and returns once
    // every one has returned.
    void RunThreads(const std::function<void(int)>& fn) override {
        std::vector<std::thread> pool;
        pool.reserve(threads_);
        for (int t = 0; t < threads_; ++t) pool.emplace_back(fn, t);
        for (std::thread& thread : pool) thread.join();
    } // FreshThreadRunner::RunThreads()
private:
    const int threads_;   // the threads of one run
}; // class FreshThreadRunner

// The factory ThreadRunner() takes: a FreshThreadRunner for `threads` threads.
static std::unique_ptr<benchmark::ThreadRunnerBase> fresh_thread_runner(int threads) {
    return std::make_unique<FreshThreadRunner>(threads);
} // fresh_thread_runner()

// The number of distinct arena shards, of `shards` (a power of two), that the
// thread numbers `numbers` map to: the shard witness. With n numbers, n
// distinct shards means that no two of those threads share one.
static size_t distinct_shards(const std::vector<unsigned>& numbers, size_t shards) {
    std::vector<size_t> used;
    used.reserve(numbers.size());
    for (const unsigned number : numbers) used.push_back(number & (shards - 1));
    std::sort(used.begin(), used.end());
    return size_t(std::unique(used.begin(), used.end()) - used.begin());
} // distinct_shards()

// ---------------------------------------------------------------------------
// Handle_JoinLeave: the join and the leave, nothing else.
//
// Why measure: Lookup_MostlyOld_OpHandle (concurrent_hash_set_rcu_bm_
// vs_baseline.C) measures a handle per operation as a caller pays it, a join
// and a leave around every operation (the lookups and the trickle of
// inserts), which mixes the handle's cost with the operation's and spaces the
// joins of a thread by an operation. Here every iteration is new_handle() and
// the destruction of the handle it returned (the join and the leave) and
// nothing else, so the rate is the handle's
// alone, at the highest rate of joins the threads can generate: how far the
// handle's own writes let it scale with the threads that join at once, and
// what a join and a leave cost one thread with no other writer.
//
// Workload: thread 0's SetUp() constructs an EMPTY set (1024 buckets, the
// arena shards of arena_shards_from_env(), no cadence; nothing is inserted:
// a join reads no bucket) and its TearDown() deletes it; nothing calls
// advance() or reclaim(), so every join finds the generation it loaded still
// current and none retries. The total work is join_leave_total() joins and
// leaves, join_leave_total()/T per thread, rounded down
// (register_join_leave()): the same at every thread count that divides it
// (every power of two; not a num_cpu that is not one), so a run of T threads
// that scales takes the same time as one thread's run divided by T. Measured
// over real time with the wall-clock records of every multi-threaded row
// (items_per_second, wall_items_per_second, finish_spread;
// concurrent_hash_set_rcu_bm.C says what each one is).
//
// Counters beyond the rates, set by thread 0 after the loop:
//   shards        the set's arena shard count S;
//   shards_used   the distinct arena shards of the run's T threads (the
//                 witness: T for T <= S means no two threads of the run
//                 shared a shard, by construction of the thread numbering
//                 above; fewer is a shared shard);
//   joins_per_op  the set's `joins` counter after the run over T x
//                 iterations: 1 when every iteration joined once;
//   join_retries  the set's `join_retries` counter after the run: 0, since
//                 nothing publishes a generation during the run.
// ---------------------------------------------------------------------------
// The joins and leaves of one Handle_JoinLeave run, all threads together:
// 2^27, or the environment's HASH_BM_JOIN_LEAVE_TOTAL when that is a power of
// two from 2^10 to 2^30 (any other value set there is reported on stderr and
// replaced by the default). The default is large so that a run at the full
// thread count, whose threads' joins write lines of their own and so scale,
// still lasts long enough to time; the override lets a campaign choose
// another total, e.g. a smaller one where many threads' joins share lines
// (more threads than arena shards). Read once, when the rows are registered:
// a row's iterations (and so its name) are fixed then.
static benchmark::IterationCount join_leave_total() {
    constexpr benchmark::IterationCount kDefault = benchmark::IterationCount(1) << 27;
    const char* s = getenv("HASH_BM_JOIN_LEAVE_TOTAL");
    if (s == nullptr) return kDefault;
    char* end = nullptr;
    const long long n = strtoll(s, &end, 10);   // strtoll, not atol: an out-of-range value is clamped, not undefined
    if (end != s && *end == '\0' && n >= (1LL << 10) && n <= (1LL << 30) && (n & (n - 1)) == 0) return n;
    fprintf(stderr, "HASH_BM_JOIN_LEAVE_TOTAL=\"%s\" is not a power of two from 2^10 to 2^30: Handle_JoinLeave uses 2^27\n", s);
    return kDefault;
} // join_leave_total()

// The thread numbers of a run's threads, by thread index, for the shard
// witness: written by each thread before its loop, read by thread 0 after
// its own loop (the loop's start is a barrier every thread passes after its
// write, so the writes happen-before the read). A static array rather than a
// fixture member that thread 0 allocates in SetUp(): no thread may touch
// what thread 0's SetUp() creates before the loop starts (the CAUTION above
// the fixtures in the benchmark header). A run of more threads than the
// array holds reports no witness.
static constexpr int kMaxWitnessThreads = 4096;
static unsigned join_leave_numbers[kMaxWitnessThreads];

class JoinLeaveFixture : public benchmark::Fixture {
public:
    static ConcurrentSet* set;   // the run's set: created by thread 0's SetUp(), deleted by its TearDown()
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) set = new_set<ConcurrentSet>(1024);
    } // JoinLeaveFixture::SetUp()
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { delete set; set = nullptr; }
    } // JoinLeaveFixture::TearDown()
}; // class JoinLeaveFixture
ConcurrentSet* JoinLeaveFixture::set = nullptr;

BENCHMARK_DEFINE_F(JoinLeaveFixture, Handle_JoinLeave)(benchmark::State& state) {
    static constinit WallRecords records;
    // This thread's number, before the loop (see above), recorded for the
    // witness.
    const unsigned number = concurrent_hash_rcu_detail::thread_number();
    if (state.thread_index() < kMaxWitnessThreads) join_leave_numbers[state.thread_index()] = number;
    records.allocate(state);
    for (auto _ : WallTimed(state, records)) {
        // The join; the leave at the end of the scope. The handle is
        // handed to DoNotOptimize so that it exists, joined, between the
        // two: the join and the leave are atomic read-modify-writes that no
        // compiler removes, and this keeps it so whatever a compiler does.
        // DoNotOptimize keeps the handle in memory and is a compiler barrier
        // on memory, so every iteration also stores the handle's words to
        // this thread's stack and keeps no memory value in a register across
        // it. Not const: DoNotOptimize's const-reference form is deprecated.
        ConcurrentSet::handle h = set->new_handle();
        benchmark::DoNotOptimize(h);
    } // the timed joins and leaves
    state.SetItemsProcessed(state.iterations());
    report_wall(state, records, double(state.iterations())*state.threads());
    if (state.thread_index() == 0) {
        const size_t shards = set->get_internal_arena_shards();
        state.counters["shards"] = benchmark::Counter(double(shards));
        if (state.threads() <= kMaxWitnessThreads) {
            const std::vector<unsigned> numbers(join_leave_numbers, join_leave_numbers + state.threads());
            state.counters["shards_used"] = benchmark::Counter(double(distinct_shards(numbers, shards)));
        } // the witness, when every thread's number was recorded
        const ConcurrentSet::InternalCounters c = set->get_internal_counters();
        state.counters["joins_per_op"] = benchmark::Counter(double(c.joins)/(double(state.iterations())*state.threads()));
        state.counters["join_retries"] = benchmark::Counter(double(c.join_retries));
    } // thread 0, after every thread has left the loop
    records.release(state);
} // Handle_JoinLeave

// Registers Handle_JoinLeave at the thread counts of ThreadRange(1, num_cpu)
// (1, the powers of two below num_cpu, num_cpu), each its own registration
// with join_leave_total()/T iterations per thread and every thread a new one
// (FreshThreadRunner). The registration call is the one
// BENCHMARK_REGISTER_F() expands to; register_shared_total() of the
// benchmark header is the same loop without the thread runner.
static void register_join_leave() {
    const benchmark::IterationCount total = join_leave_total();
    for (int t = 1; ; t *= 2) {
        if (t >= num_cpu) t = num_cpu;
        benchmark::internal::RegisterBenchmarkInternal(std::make_unique<BENCHMARK_PRIVATE_CONCAT_NAME(JoinLeaveFixture, Handle_JoinLeave)>())
            ->Threads(t)->Iterations(total/t)->UseRealTime()->ThreadRunner(fresh_thread_runner);
        if (t == num_cpu) break;
    } // loop over the thread counts
} // register_join_leave()
[[maybe_unused]] static const bool join_leave_registered = (register_join_leave(), true);

// ---------------------------------------------------------------------------
// Reclaim_FreePassCheck: the free pass's check of the handle counts, under
// joins and leaves on other threads.
//
// Why measure: the free pass frees a sealed generation's bag only once it
// has found no live handle in that generation, by loading that generation's
// count in every arena shard's count row, in shard order, stopping at the
// first that is not 0 (the header's free_oldest()). That test runs in every
// advance(), reclaim() and automatic trigger that finds a sealed bag (twice
// in a call that finds the ring full). A join and its leave write the current
// generation's count in the joining thread's shard's row, where the counts of
// consecutive ring slots sit side by side, eight to a 64-byte line; the
// generation the test reads is the oldest sealed one, whose slot is usually
// next to the current one (the slot just closed, or, under a full ring, the
// slot after the current one), so the lines the test reads are usually lines
// the joiners write. This row measures the test's cost as a function of S
// (the arena shards), W (the threads joining and leaving meanwhile) and P (a
// handle pinning the oldest generation, which makes every call find the ring
// full).
//
// Arguments (the row's name carries them; register_free_pass_check() lists
// the cells): W, untimed helper threads that join and leave without pause
// (new_handle() and its destruction, in a loop) from before the first call
// to after the last; S, the set's arena shard count (through Options, so one
// invocation runs every S); P, 1 for a pin: one more helper thread, placed
// on arena shard S - 1 (started threads are tried one by one until one's
// number lands there, at most S tries), holds one handle, never refreshed,
// from before the first call to after the last.
//
// Where the counts sit: SetUp() moves the ring forward one generation (an
// untimed erase, insert and reclaim()) before it starts the helpers, so the
// pin joins generation 1, in ring slot 1, and once the ring is full the
// current generation is the one in slot 0 (the ring has kCheckRingSlots
// slots): in every shard's row the count the test reads (slot 1) and the
// current count (slot 0) share a 64-byte line, as they do for 7 of the 8
// positions of the current slot in a line; the current count is written in
// the rows of the W joiners' shards (and in the measured thread's own,
// between its calls), in none at W = 0. The eighth (the current
// slot the last of its line, so the oldest the first of the next) is not
// measured at P = 1. At P = 0 the slots move on by one per call, so 1 call
// in 8 tests a count on another line than the current one. The pin_slot and
// current_slot counters witness the placement.
//
// One measured thread (Threads(1), a new thread: FreshThreadRunner), on a
// set constructed with kCheckBuckets buckets and holding the one key
// kCheckKey. Each iteration, untimed: under a handle of its own, erase the
// key (its node is retired into this thread's shard) and insert it again,
// then release the handle; timed: one reclaim() call, whose whole time
// (advance_ns, from the calling thread's own timers) is the iteration's time
// (manual time). The erase gives every call a retired node to bag, so no
// call finds nothing to do; the handle is released before the call, so it
// pins nothing the call closes. reclaim() and not advance(): advance() would
// also walk the table, whose cost has nothing to do with the check. The
// table doubles during the run (the set grows it by its count of arena
// nodes, which counts nodes in batches per shard free list and, at P = 1,
// the nodes the pin keeps from being freed: the 63 bags behind it, then the
// retired list that a full ring cannot bag); that does not enter the timed
// calls: every doubling happens in an untimed insert and reclaim() reads no
// bucket. At P = 0 the untimed operations that split the key's bucket after
// a doubling (a handful in a run) may add the nodes their splits retire to
// the next call's bag; at P = 1 those nodes join the retired list.
//
// What a call does, and where its time goes (the calling thread's timers:
// bump_ns is the bump, free_ns the final free pass, InternalTimers in the
// header):
//   `advanced`  the bump closed the current generation's retired node into
//               a bag and published the next generation (bump_ns: every
//               shard's retired and limbo heads loaded, every shard's
//               retired head exchanged, the publish); the final free pass
//               then frees the oldest sealed bags while unpinned (free_ns:
//               for each bag it frees, the test of its counts and the push of
//               its node; for the bag it stops at, the test alone). A
//               joiner caught in the generation just closed stops the pass
//               there (0 bags freed this call; the next call frees two);
//   `ring_full` the bump loads every shard's retired and limbo heads and
//               stores the limbo heads it loaded, as on an `advanced` call;
//               then its ring-wrap free pass tests the oldest generation and
//               stops, and the bump counts the outcome (one atomic
//               increment) and returns before any exchange (all in bump_ns);
//               the final free pass tests the oldest generation again and
//               stops (free_ns). So bump_ns + free_ns is both of a call's
//               tests plus those 2S loads and S stores and the increment,
//               and no other free-pass work: at S = 256 the head loads are as
//               many as the two tests' loads. With P = 1 every call from the
//               64th on is `ring_full` (the ring holds 64 generations and the
//               pinned one never frees), each test reading every shard's
//               count up to the pin's on S - 1. Without a pin, whenever a
//               handle stays in a generation for more than 63 calls (a joiner
//               descheduled between its join and its leave): the test then
//               stops wherever that handle's count is read.
// So the calls are reported by outcome, each with its own mean bump_ns and
// free_ns; the counts' test is read per outcome against bump_ns (on an
// `advanced` call, which also pays one exchange per shard; most cleanly on
// the calls that freed exactly one bag: one whole test, one node pushed,
// one test of the next sealed bag if any), or as bump_ns + free_ns (on a
// `ring_full` call).
//
// Counters (means per call of the named outcome; the outcome's call count
// beside them):
//   adv0_calls, adv0_bump_ns, adv0_free_ns
//                    `advanced` calls that freed no bag (a joiner or the pin
//                    stopped the pass at the first sealed bag);
//   adv1_calls, adv1_bump_ns, adv1_free_ns
//                    `advanced` calls that freed exactly one bag;
//   advn_calls, advn_bump_ns, advn_free_ns, advn_bags
//                    `advanced` calls that freed two bags or more, and the
//                    bags they freed per call;
//   full_calls, full_bump_ns, full_free_ns, full_check_ns
//                    `ring_full` calls, and their bump_ns + free_ns (the two
//                    tests with the bump's head loads and its increment: see
//                    `ring_full` above);
//   other_calls      calls with any other outcome (none expected: a call
//                    always has a node to bag, and nothing else holds the
//                    flag);
//   joins_per_call   the helpers' joins during the measured thread's loop
//                    over its calls: the witness that the joiners ran;
//   join_retries     the set's join attempts that found the generation
//                    replaced and joined again, across the measured thread's
//                    loop (the helpers' joins that a call's publish caught);
//   shards_used      the distinct arena shards of the measured thread and
//                    the W joiners (W + 1 means none shared one);
//   pin_shard, pin_slot, current_slot
//                    when P = 1: the pin's arena shard (S - 1), the ring slot
//                    of its generation (1) and the ring slot of the current
//                    generation after the loop (0).
// ---------------------------------------------------------------------------
static constexpr size_t kCheckBuckets = 1024;               // the buckets the row's set is constructed with: it holds one key
static constexpr int kCheckKey = 0;                         // the key erased and inserted again before every call
static constexpr benchmark::IterationCount kCheckCalls = 1 << 16;   // reclaim() calls per run
// The kGenerations template argument of a ConcurrentResizableHashSetRCU
// type, which the class does not expose: the ring's generation slots.
template <typename Set> struct RingSlotsOf;
template <typename T, bool AllowDelete, typename Hash, size_t kGenerations, concurrent_hash_rcu_detail::SetHooks Hooks>
struct RingSlotsOf<ConcurrentResizableHashSetRCU<T, AllowDelete, Hash, kGenerations, Hooks>> {
    static constexpr size_t value = kGenerations;
}; // struct RingSlotsOf
static constexpr uint64_t kCheckRingSlots = RingSlotsOf<ConcurrentSetDel>::value;   // the ring's generation slots of the row's set (64, the container's default)

// One helper thread of Reclaim_FreePassCheck: a joiner (joins and leaves
// until told to stop) or the pin (holds one handle until told to stop). On
// a 128-byte block of its own: a joiner writes its `joins` on every join.
struct alignas(128) CheckHelper {
    std::thread thread;                 // the helper; joined by the fixture's TearDown()
    std::atomic<uint64_t> joins{0};     // joins so far (a joiner's; written by the helper alone, read by the measured thread)
    std::atomic<bool> numbered{false};  // set by the helper once `number` holds its thread number (and, for the pin candidate, once it has decided)
    std::atomic<bool> pinning{false};   // pin candidates: set before `numbered` if this candidate landed on the wanted shard and holds the pin's handle
    unsigned number = 0;                // the helper's thread number, written before `numbered` is set
    uint64_t generation = 0;            // the pin's: the generation its handle joined, written before `numbered` is set
}; // struct CheckHelper

class FreePassCheckFixture : public benchmark::Fixture {
public:
    static ConcurrentSetDel* set;                     // the run's set
    static std::vector<std::unique_ptr<CheckHelper>> joiners;   // the W joiners
    static std::unique_ptr<CheckHelper> pin;          // the pin (P = 1), else null
    static std::atomic<bool> stop;                    // set by TearDown(): every helper returns
    static unsigned measured_number;                  // the measured thread's thread number

    // Thread 0 (the only measured thread): the set with its one key, the
    // ring moved one generation forward, the joiners, the pin, every helper
    // started and numbered before this returns.
    void SetUp(const ::benchmark::State& state) override {
        measured_number = concurrent_hash_rcu_detail::thread_number();   // first: the measured thread's number precedes every helper's
        const size_t shards = size_t(state.range(1));
        set = new ConcurrentSetDel(ConcurrentSetDel::Options{kCheckBuckets, shards, 0});
        {   // the key, under a handle of the prep's own
            const ConcurrentSetDel::handle h = set->new_handle();
            if (!set->insert(h, mix(kCheckKey))) abort();   // a fresh set: cannot fail
        } // the prep's handle
        {   // the ring one generation forward, so that the pin joins slot 1 (see "Where the counts sit")
            {   // one node retired, under a handle released before the call
                const ConcurrentSetDel::handle h = set->new_handle();
                if (!set->erase(h, mix(kCheckKey)) || !set->insert(h, mix(kCheckKey))) abort();   // the key is there: cannot fail
            } // the prep's second handle
            const AdvanceResult r = set->reclaim();
            if (r.advance != AdvanceResult::Advance::advanced || r.freed_bags != 1) {
                fprintf(stderr, "Reclaim_FreePassCheck: the prep's reclaim() did not advance and free generation 0\n");
                abort();
            } // if the ring did not move as described
        } // the ring forward
        stop.store(false, std::memory_order_relaxed);
        joiners.clear();
        for (int64_t w = 0; w < state.range(0); ++w) {
            joiners.push_back(std::make_unique<CheckHelper>());
            CheckHelper& helper = *joiners.back();
            helper.thread = std::thread([&helper] {
                helper.number = concurrent_hash_rcu_detail::thread_number();
                helper.numbered.store(true, std::memory_order_release);
                while (!stop.load(std::memory_order_relaxed)) {
                    ConcurrentSetDel::handle h = set->new_handle();   // not const: as in Handle_JoinLeave
                    benchmark::DoNotOptimize(h);
                    helper.joins.store(helper.joins.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);   // one writer: no RMW needed
                } // join and leave until told to stop
            });
            // Wait for this joiner's number before starting the next, so
            // that the measured thread and the joiners hold consecutive
            // numbers: W + 1 distinct shards for W < S.
            while (!helper.numbered.load(std::memory_order_acquire)) std::this_thread::yield();
        } // loop over the joiners
        // The pin last: its placement takes numbers after the joiners'
        // (which take no more), so it does not move them; on S - 1, it
        // may share that shard with a joiner or the measured thread.
        if (state.range(2) != 0) place_pin(set->get_internal_arena_shards());
    } // FreePassCheckFixture::SetUp()

    // Thread 0: stops and joins every helper (the pin leaves its handle
    // first), then deletes the set.
    void TearDown(const ::benchmark::State& /*state*/) override {
        stop.store(true, std::memory_order_relaxed);
        stop.notify_all();
        for (std::unique_ptr<CheckHelper>& helper : joiners) helper->thread.join();
        joiners.clear();
        if (pin) {
            pin->thread.join();
            pin.reset();
        } // if a pin was placed
        delete set;
        set = nullptr;
    } // FreePassCheckFixture::TearDown()

private:
    // Places the pin on arena shard `shards` - 1: starts threads one at a
    // time, each taking its thread number; the first whose number lands on
    // that shard joins the set and keeps its handle until `stop`, the others
    // return at once and are joined here. Consecutive numbers reach every
    // shard within `shards` tries (no other thread takes a number
    // meanwhile); a placement that fails aborts the run.
    static void place_pin(size_t shards) {
        const unsigned wanted = unsigned(shards - 1);
        for (size_t attempt = 0; attempt < shards; ++attempt) {
            std::unique_ptr<CheckHelper> candidate = std::make_unique<CheckHelper>();
            CheckHelper& c = *candidate;
            c.thread = std::thread([&c, wanted, shards] {
                c.number = concurrent_hash_rcu_detail::thread_number();
                if ((c.number & (shards - 1)) != wanted) {
                    c.numbered.store(true, std::memory_order_release);
                    return;
                } // not the pin's shard: done
                const ConcurrentSetDel::handle h = set->new_handle();
                c.generation = ConcurrentSetDel::get_internal_handle_generation(h);
                c.pinning.store(true, std::memory_order_relaxed);
                c.numbered.store(true, std::memory_order_release);
                stop.wait(false, std::memory_order_relaxed);   // until TearDown(); the handle leaves on return
            });
            while (!c.numbered.load(std::memory_order_acquire)) std::this_thread::yield();
            if (c.pinning.load(std::memory_order_relaxed)) {
                pin = std::move(candidate);
                return;
            } // the pin is placed
            c.thread.join();
        } // loop over the candidates
        fprintf(stderr, "Reclaim_FreePassCheck: no thread landed on arena shard %zu\n", shards - 1);
        abort();
    } // FreePassCheckFixture::place_pin()
}; // class FreePassCheckFixture
ConcurrentSetDel* FreePassCheckFixture::set = nullptr;
std::vector<std::unique_ptr<CheckHelper>> FreePassCheckFixture::joiners;
std::unique_ptr<CheckHelper> FreePassCheckFixture::pin;
std::atomic<bool> FreePassCheckFixture::stop{false};
unsigned FreePassCheckFixture::measured_number = 0;

// The per-outcome sums of one run's calls (see the row's counters above).
// Trivially destructible: a local of the body, live across the loop.
struct CheckTally {
    size_t adv0_calls = 0, adv0_bump_ns = 0, adv0_free_ns = 0;          // `advanced`, no bag freed
    size_t adv1_calls = 0, adv1_bump_ns = 0, adv1_free_ns = 0;          // `advanced`, exactly one bag freed
    size_t advn_calls = 0, advn_bump_ns = 0, advn_free_ns = 0, advn_bags = 0;   // `advanced`, two bags or more freed
    size_t full_calls = 0, full_bump_ns = 0, full_free_ns = 0;          // `ring_full`
    size_t other_calls = 0;                                             // any other outcome
}; // struct CheckTally
static_assert(std::is_trivially_destructible_v<CheckTally>, "a body local with a destructor would be live across the state loop");

BENCHMARK_DEFINE_F(FreePassCheckFixture, Reclaim_FreePassCheck)(benchmark::State& state) {
    ConcurrentSetDel::InternalTimers& timers = ConcurrentSetDel::internal_advance_timers();
    CheckTally tally;
    bool as_described = true;   // every untimed erase and insert succeeded (the key is always there to erase and absent to insert)
    const ConcurrentSetDel::InternalCounters before = set->get_internal_counters();
    uint64_t helper_joins_before = 0;
    for (const std::unique_ptr<CheckHelper>& helper : joiners) helper_joins_before += helper->joins.load(std::memory_order_relaxed);
    for (auto _ : state) {
        {   // untimed: one node retired for the call to bag, under a handle released before the call
            const ConcurrentSetDel::handle h = set->new_handle();
            as_described &= set->erase(h, mix(kCheckKey));
            as_described &= set->insert(h, mix(kCheckKey));
        } // the iteration's handle
        timers = {0, 0, 0, 0};
        const AdvanceResult r = set->reclaim();
        state.SetIterationTime(double(timers.advance_ns)*1e-9);
        if (r.advance == AdvanceResult::Advance::advanced && r.freed_bags == 0) {
            ++tally.adv0_calls;
            tally.adv0_bump_ns += timers.bump_ns;
            tally.adv0_free_ns += timers.free_ns;
        } else if (r.advance == AdvanceResult::Advance::advanced && r.freed_bags == 1) {
            ++tally.adv1_calls;
            tally.adv1_bump_ns += timers.bump_ns;
            tally.adv1_free_ns += timers.free_ns;
        } else if (r.advance == AdvanceResult::Advance::advanced) {
            ++tally.advn_calls;
            tally.advn_bump_ns += timers.bump_ns;
            tally.advn_free_ns += timers.free_ns;
            tally.advn_bags += r.freed_bags;
        } else if (r.advance == AdvanceResult::Advance::ring_full) {
            ++tally.full_calls;
            tally.full_bump_ns += timers.bump_ns;
            tally.full_free_ns += timers.free_ns;
        } else {
            ++tally.other_calls;
        } // by outcome
    } // the timed calls
    if (!as_described) state.SkipWithError("an erase or insert of the row's one key failed: the calls did not bag what the row describes");
    uint64_t helper_joins_after = 0;
    for (const std::unique_ptr<CheckHelper>& helper : joiners) helper_joins_after += helper->joins.load(std::memory_order_relaxed);
    const ConcurrentSetDel::InternalCounters after = set->get_internal_counters();
    // The means per call of an outcome; 0 when the outcome never occurred.
    const auto mean = [](size_t sum, size_t calls) { return calls == 0 ? 0.0 : double(sum)/double(calls); };
    state.counters["adv0_calls"]    = benchmark::Counter(double(tally.adv0_calls));
    state.counters["adv0_bump_ns"]  = benchmark::Counter(mean(tally.adv0_bump_ns, tally.adv0_calls));
    state.counters["adv0_free_ns"]  = benchmark::Counter(mean(tally.adv0_free_ns, tally.adv0_calls));
    state.counters["adv1_calls"]    = benchmark::Counter(double(tally.adv1_calls));
    state.counters["adv1_bump_ns"]  = benchmark::Counter(mean(tally.adv1_bump_ns, tally.adv1_calls));
    state.counters["adv1_free_ns"]  = benchmark::Counter(mean(tally.adv1_free_ns, tally.adv1_calls));
    state.counters["advn_calls"]    = benchmark::Counter(double(tally.advn_calls));
    state.counters["advn_bump_ns"]  = benchmark::Counter(mean(tally.advn_bump_ns, tally.advn_calls));
    state.counters["advn_free_ns"]  = benchmark::Counter(mean(tally.advn_free_ns, tally.advn_calls));
    state.counters["advn_bags"]     = benchmark::Counter(mean(tally.advn_bags, tally.advn_calls));
    state.counters["full_calls"]    = benchmark::Counter(double(tally.full_calls));
    state.counters["full_bump_ns"]  = benchmark::Counter(mean(tally.full_bump_ns, tally.full_calls));
    state.counters["full_free_ns"]  = benchmark::Counter(mean(tally.full_free_ns, tally.full_calls));
    state.counters["full_check_ns"] = benchmark::Counter(mean(tally.full_bump_ns + tally.full_free_ns, tally.full_calls));
    state.counters["other_calls"]   = benchmark::Counter(double(tally.other_calls));
    state.counters["joins_per_call"] = benchmark::Counter(double(helper_joins_after - helper_joins_before)/double(state.iterations()));
    state.counters["join_retries"]  = benchmark::Counter(double(after.join_retries - before.join_retries));
    // The shard witness: the measured thread and the joiners.
    std::vector<unsigned> numbers{measured_number};
    for (const std::unique_ptr<CheckHelper>& helper : joiners) numbers.push_back(helper->number);
    const size_t shards = set->get_internal_arena_shards();
    state.counters["shards_used"] = benchmark::Counter(double(distinct_shards(numbers, shards)));
    if (pin) {
        state.counters["pin_shard"]    = benchmark::Counter(double(pin->number & (shards - 1)));
        state.counters["pin_slot"]     = benchmark::Counter(double(pin->generation % kCheckRingSlots));
        state.counters["current_slot"] = benchmark::Counter(double(set->get_internal_generation() % kCheckRingSlots));
    } // the pin's witnesses
} // Reclaim_FreePassCheck

// Registers the row's cells: W in {0, 7, 15} (no joiners; 7 joiners; 15,
// which with the measured thread fill a 16-thread machine) and, on a machine
// with more than 16 hardware threads, also num_cpu - 1 (one joiner per
// hardware thread besides the measured thread's: at S = 16 more joiners than
// shards, several to a shard's row; at S = 256 on a machine of at most 256
// hardware threads, one thread to a shard's row at most); S in {16, 256} (a
// laptop's default shard count; a large server's); P in {0, 1}. Fixed calls
// per run, one measured thread, every thread a new one, manual time (each
// iteration's time is its reclaim() call alone). The registration call is
// the one BENCHMARK_REGISTER_F() expands to, with the W list built here
// because it depends on num_cpu.
static void register_free_pass_check() {
    std::vector<int64_t> joiners{0, 7, 15};
    if (num_cpu > 16) joiners.push_back(num_cpu - 1);
    benchmark::internal::RegisterBenchmarkInternal(std::make_unique<BENCHMARK_PRIVATE_CONCAT_NAME(FreePassCheckFixture, Reclaim_FreePassCheck)>())
        ->ArgNames({"W", "S", "P"})
        ->ArgsProduct({joiners, {16, 256}, {0, 1}})
        ->Threads(1)->Iterations(kCheckCalls)->UseManualTime()->ThreadRunner(fresh_thread_runner);
} // register_free_pass_check()
[[maybe_unused]] static const bool free_pass_check_registered = (register_free_pass_check(), true);
