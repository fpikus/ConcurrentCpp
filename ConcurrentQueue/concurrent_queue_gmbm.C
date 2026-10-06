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
// Google Benchmark THROUGHPUT benchmark for RingAtomicMapQueueMPMC (the "g"
// in gmbm = Google Benchmark; concurrent_queue_mbm.C is the hand-rolled twin
// of this file: the same operation, variants and names in a harness where
// every thread contends until a common stop).
//
// What it measures: aggregate enqueue+dequeue throughput (items/s) of the
// queue under MPMC contention, in both of its modes: key-only (an int* key,
// Value = void) and key/value (uint64_t key, uint64_t value). Google
// Benchmark runs BM_MP_MC simultaneously on every thread of a run, every
// thread for the same number of iterations. This is a throughput probe, NOT
// latency -- for per-op latency see concurrent_queue_lmbm.C (contended
// push->pop handoff) and concurrent_queue_ppmbm.C (1-producer/1-consumer
// round trip).
//
// Thread roles are STATIC and split by index parity: odd threads push, even
// threads pop (state.thread_index() & 1). The thread counts are the powers of
// two from 2 up to the CPU count, then the CPU count itself, so producers and
// consumers are balanced at every even thread count in the sweep. In the
// *_balanced rows a thread that misses also performs one operation of the
// other role (BALANCE in timed_loop()). (lmbm's roles, by contrast, are fully
// dynamic: every thread consumes and produces only after an empty miss.)
//
// The registered sweep fixes capacity (65536 slots unless the CQ_CAP
// environment variable says otherwise; see get_queue_capacity()) and varies
// only the per-slot ALIGN template parameter (0/16/64/128) to study how slot
// cache-line alignment trades off against contention; NTRY is pinned at 8.
// Every variant runs twice: as is, and balanced (*_balanced). The
// commented-out entries of kVariants (a8/a32/a256) are kept ready for ad-hoc
// runs.
//
// Why the balanced rows were added: with static roles and a fixed iteration
// count, producers and consumers here do the same amount of work.
// concurrent_queue_mbm.C runs every thread for the same TIME instead, and
// there the role that attempts faster pins the queue at a boundary -- full,
// say, with the producers' failing retries slowing the consumers -- a
// different and slower regime. With BALANCE, a thread that finds the queue
// full pops and one that finds it empty pushes, so neither boundary can hold
// it, under equal work or equal time. mbm runs only balanced rows; these are
// their twins.
//
// Why the key/value (kv) variants were added: the two modes share nothing
// below the locks. Key-only push/pop store and clear the key inside the
// critical sections; key/value push/pop claim the slot under the lock and
// construct/move the value outside it, with a separate per-slot protocol
// for the handoff. The key-only rows therefore say nothing about the cost
// of the key/value protocol, and changing that protocol needs its own
// throughput numbers, before and after. Key/value keys are distinct per
// element (see timed_loop()); the values are trivially copyable, so the rows measure
// the protocol, not the payload's constructors.
//
// MEASUREMENT WINDOW
//
// Google Benchmark (as of 1.9.5, src/benchmark_runner.cc) decides that a
// threaded run is long enough by comparing min_time against the real time
// SUMMED over all threads, and every thread then runs that many iterations.
// So a min_time of 1 s at 128 threads is a measurement window of about 8 ms
// per thread, from a cold start: far too short for a queue to leave its empty
// start-up state, or for the lock's 1 ms back-off tier to happen more than a
// few times. So every thread count t is registered separately (see main())
// with MinTime(kWindowSeconds*t) and MinWarmUpTime(kWarmupSeconds*t): no
// thread's loop can run longer than the run itself, so these give at least
// kWindowSeconds of wall-clock measurement after at least kWarmupSeconds of
// warm-up (the warm-up is compared against the same summed time, so it needs
// the same scaling). Every run, the warm-up included, constructs a fresh
// queue, so the measured run still starts empty (or prefilled to CQ_FILL; see
// STARTING STATE), and Google
// Benchmark starts fresh threads for every run except thread 0, which runs on
// the main thread: the warm-up brings up the clock frequency, not the queue's
// state or the workers' placement. The
// per-benchmark values take precedence over the seconds forms of
// --benchmark_min_time and --benchmark_min_warmup_time, which this binary
// therefore ignores. --benchmark_min_time=<N>x with N >= 1 still works: an
// explicit iteration count wins over MinTime(), and fixes the number of
// iterations EACH thread runs (N = 0 is not supported: timed_loop()'s batch
// loop never ends at zero iterations). The run names carry min_time: and
// min_warmup_time: fields that grow with the thread count.
//
// THROUGHPUT ACCOUNTING
//
// Google Benchmark's items_per_second is the total item count divided by the
// MEAN of the threads' loop times (benchmark_runner.cc divides the summed time
// by the thread count), and so are the push, pop, push_fail and pop_fail
// rates. Every thread runs the same number of iterations, failed pushes and
// pops included, so threads finish at different times: under the batching
// SpinLock the threads that win the streaks finish early, and a role whose
// iterations are cheaper (a consumer failing on an empty queue, a producer
// failing on a full one) can finish altogether and leave the other role to
// run alone against a queue that nobody drains or fills. The mean loop time
// is then shorter than the run, and the tail of the run measures a different
// regime from its start. So each run also reports:
//   wall_items_per_second -- total successful pushes and pops over the run's
//                            wall-clock span;
//   finish_spread         -- how unevenly the threads finished, between 0
//                            (all together) and 1;
//   push_end, pop_end     -- the end of the last producer's (consumer's) loop,
//                            measured from the start of the span, as a
//                            fraction of the span: 1 for the role that
//                            finished last; the other role's value below 1
//                            means that the rest of the run had no producers
//                            (consumers) at all. A producer here is an odd
//                            thread, also in the balanced rows, where it
//                            pops too.
// The first two are the counters of ../Spinlock/gb_wall_clock.h, which
// defines the span and finish_spread exactly and says why the span starts
// where it does; this benchmark keeps its own per-thread record (ThreadStats,
// derived from the header's WallStamps) and its own timed loop (timed_loop()),
// and aggregates the records with the header's wall_span() and
// set_wall_counters(). Every finish is at or after its own thread's first
// stamp, the earliest of which starts the span, so no finish precedes the
// span start, which is what keeps push_end and pop_end within [0, 1]. The
// item count includes at most one operation per thread from before the span
// starts, which is negligible. These counters are plain values, set by
// thread 0 alone. The time stamps cost one compare per iteration, after the
// operation. Where finish_spread is large, even
// wall_items_per_second mixes contention levels; concurrent_queue_mbm.C runs
// the same operation with every thread contending until a common stop, and its
// items_per_s does not have this problem.
//
// Two counters need care of their own: in the unbalanced rows, with equal
// iteration counts and equal numbers of producers and consumers,
// push_fail - pop_fail == fill - rem (as
// counts, fill being the prefilled elements; see STARTING STATE) by
// construction, so the fail counters do not tell you which side was waiting;
// and rem% is a snapshot of the queue at the end of a run, not a steady-state
// occupancy.
//
// STARTING STATE
//
// Every run starts from a fresh queue, empty by default. The CQ_FILL
// environment variable (a fraction of the capacity in [0, 1]; 0 by default)
// makes thread 0 push that many elements before the start barrier, so that
// every run, the measured one included, starts that full. Why: at 2^16 slots
// an unbalanced row can settle with the queue full or near empty, and the knob
// lets a run choose where it starts. Where it starts does not decide where it
// settles; which role attempts faster does (see WHY BALANCED in
// concurrent_queue_mbm.C). The prefilled elements are not counted as items.
// The run names do not show CQ_CAP or CQ_FILL; both are in the context at the
// top of the output (cq_cap, cq_fill).

#include "concurrent_queue.h"
#include "gb_wall_clock.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <timeapi.h>
  #pragma comment(lib, "winmm.lib")
  // Raise the system timer resolution to ~1 ms for the lifetime of the
  // process so Sleep() in the spin-lock fallback path is not stuck at the
  // default ~15.6 ms scheduler quantum.
  namespace {
    struct TimerResolutionGuard {
        TimerResolutionGuard()  { ::timeBeginPeriod(1); }
        ~TimerResolutionGuard() { ::timeEndPeriod(1);   }
    };
    static TimerResolutionGuard g_timer_resolution_guard;
  }
#else
  #include <unistd.h>
#endif

#include <benchmark/benchmark.h>

// Simple CHECK macros for standalone benchmark.
#ifndef CHECK
#define CHECK(x) if (!(x)) std::abort();
#endif
#ifndef CHECK_EQ
#define CHECK_EQ(a, b) if ((a) != (b)) std::abort();
#endif

// Wall-clock seconds of measurement and of warm-up per run (see MEASUREMENT
// WINDOW above). concurrent_queue_mbm.C uses the same values as its defaults;
// keep the two files in step.
constexpr double kWindowSeconds = 1.0;
constexpr double kWarmupSeconds = 0.5;

// The number of ints in each key-only thread's array of pointees, a power of
// 2: a thread pushes their addresses in turn. The queue needs its keys
// non-null, not distinct (it only stores them; nobody dereferences them), so
// a small array will do; one of N ints per thread would be 32 GB at 2^26
// slots and 128 threads. concurrent_queue_mbm.C uses the same value; keep the
// two files in step.
constexpr size_t kPointees = 1024;
static_assert(std::has_single_bit(kPointees));

using Clock = std::chrono::steady_clock;

// Smoke test: push two pointers and pop them back, asserting FIFO order and
// pointer identity. Not part of the measured loop -- called only via the
// `if (0) test(q)` guard below so it stays compiled but off by default.
template <typename Q> void test(Q* q) {
    int i = 1, j = 2;
    int* p;
    CHECK(q->push(&i));
    CHECK(q->push(&j));
    p = q->pop();
    CHECK(p);
    CHECK_EQ(*p, i);
    p = q->pop();
    CHECK(p);
    CHECK_EQ(*p, j);
}

// Push `count` elements into the empty queue `q`, from one thread, before any
// worker touches it (the CQ_FILL start; see STARTING STATE above).
// Key-only: the addresses of the kPointees ints of `storage`, in turn;
// `storage` is resized to kPointees and must outlive the elements' stay in
// the queue (nobody dereferences them). Key/value: the keys (i + 1)*key_stride,
// with values i: key_stride is the thread count plus 1, and a thread's own
// keys are p*key_stride + tid + 1 with tid + 1 < key_stride, so every key
// stays distinct. Returns false if a push fails, which cannot happen for
// count <= capacity on an empty queue.
template <typename Q, typename K, typename V>
bool prefill(Q& q, size_t count, size_t key_stride, std::vector<int>& storage) {
    constexpr bool key_only = std::is_same_v<V, void>;
    if constexpr (key_only) storage.resize(kPointees);
    for (size_t i = 0; i != count; ++i) {
        bool pushed;
        if constexpr (key_only) pushed = q.push(&storage[i & (kPointees - 1)]);
        else pushed = q.push(static_cast<K>((i + 1)*key_stride), static_cast<V>(i));
        if (!pushed) return false;
    } // loop over the prefilled elements
    return true;
} // prefill()

// The fraction of the capacity that thread 0 prefills in every run (see
// STARTING STATE above): set by main() from CQ_FILL before any benchmark
// thread starts, and only read after that.
static double g_queue_fill = 0;

// One thread's record of one run: its time stamps, `first` and `last`, from
// WallStamps, and its successful operations (see THROUGHPUT ACCOUNTING
// above). Each entry gets its own 128-byte block; it is written twice per run,
// so the padding matters little, but it costs nothing either. A type of its
// own rather than the header's WallRecords: timed_loop() writes the item
// count next to the stamps, and wall_span() takes an array of any record type
// derived from WallStamps.
struct alignas(128) ThreadStats : WallStamps {
    size_t items;                       // successful pushes and pops of this thread
};
static_assert(sizeof(ThreadStats) == 128);

// One thread's operation counts over a run.
struct OpCounts {
    size_t p = 0;                       // successful pushes
    size_t c = 0;                       // successful pops
    size_t pmiss = 0;                   // pushes that found the queue full
    size_t cmiss = 0;                   // pops that found the queue empty
};

// The timed loop of BM_MP_MC() for one thread: one push (PRODUCER) or one pop
// per iteration, then one compare that decides whether to read the clock.
// Every thread reads it at the end of its first iteration and at the end of
// its last one (both at once if it runs only one); `mark` is the iteration
// count of the next reading. At the last iteration the thread also records
// its successful operations, before the end-of-loop barrier, so that thread 0
// can read them.
//   Q, K, V   -- the queue and its key and value types (V == void: key-only);
//   PRODUCER  -- whether this thread pushes (true) or pops (false);
//   BALANCE   -- after counting a miss, additionally perform (and count) one
//                operation of the other role, so the queue cannot settle
//                permanently full or empty;
//   state     -- the benchmark state of this thread;
//   q, stats  -- the queue shared by all threads of the run and the array of
//                their records, both set by thread 0 before the start barrier
//                and so read here only after it (inside the batch): before
//                the barrier they may not exist yet. Passed as references to
//                BM_MP_MC()'s statics on purpose: a copy of the pointers
//                taken at the call would be read before the barrier;
//   tid       -- this thread's index, which selects its record;
//   v, mask   -- key-only producers (and BALANCE consumers): the thread's
//                array of mask + 1 ints (kPointees), whose addresses it pushes
//                in turn -- valid, never dereferenced, and never nullptr,
//                which is Key{}; unused otherwise;
//   key_stride, key_offset -- key/value pushes (a producer's, or a BALANCE
//                consumer's): the key of the thread's p-th pushed element is
//                p*key_stride + key_offset, with key_stride the thread count
//                plus 1 and key_offset = tid + 1, distinct across all
//                threads' elements and the prefill's (see prefill()) and
//                never Key{} (0), the reserved empty marker; the value is p;
//   out       -- receives the thread's operation counts.
// noinline: a function of its own, so that the only values live across the
// loop are the loop's own, as in concurrent_queue_mbm.C's run_phase(), whose
// loop body is the same; the two loops differ in their exit test and in this
// one's time-stamp compare (and its cold clock reading).
// flatten: push() and pop(), and SpinLock::lock() inside them, are inlined
// into the loop rather than left to the inliner -- with GCC, whose flatten
// applies to the calls that inlining exposes. Clang's does not reach lock(),
// which its inliner inlines here because it estimates the call site to be hot
// (see run_phase() in the twin for how the twin gets the same estimate). With
// lock() inlined in one loop and called out of line in the other, the two
// harnesses stop being twins: the out-of-line call can halve the 2-thread
// key-only rate. The SpinLock directory documents the same trap
// (spinlock_scope_common.h, E2): a call between the unlock and the next lock
// is a store there. Whether lock() is inlined into both loops is therefore to
// be checked in the disassembly of every build that is measured. A
// compile-time role rather than a test per iteration: the loop body is the
// operation alone.
template <typename Q, typename K, typename V, bool PRODUCER, bool BALANCE>
[[gnu::noinline, gnu::flatten]] static void timed_loop(benchmark::State& state, Q* const& q_shared, ThreadStats* const& stats,
                                         int tid, int* v, size_t mask, size_t key_stride, size_t key_offset,
                                         OpCounts& out) {
    constexpr bool key_only = std::is_same_v<V, void>;
    const benchmark::IterationCount last = state.max_iterations;
    // One batch of all `last` iterations, counted by the inner loop, rather
    // than the range-for over `state`: the range-for keeps an iteration
    // counter of its own next to ours (n). KeepRunningBatch() calls
    // StartKeepRunning() and FinishKeepRunning(), the barriers, at the same
    // points as the range-for, and state.iterations() is the same.
    while (state.KeepRunningBatch(last)) {
        Q& q = *q_shared;
        ThreadStats& rec = stats[tid];
        benchmark::IterationCount mark = 1;     // iteration count of the next clock reading
        size_t p = 0, c = 0, pmiss = 0, cmiss = 0;
        // Destination for popped values (key/value queue only; an unused int
        // for the key-only queue).
        [[maybe_unused]] std::conditional_t<key_only, int, V> value {};
        for (benchmark::IterationCount n = 1; n <= last; ++n) {
            if constexpr (PRODUCER) {
                bool pushed;
                if constexpr (key_only) pushed = q.push(&v[p & mask]);
                else pushed = q.push(static_cast<K>(p*key_stride + key_offset), static_cast<V>(p));
                if (pushed) ++p;
                else {
                    ++pmiss;
                    if constexpr (BALANCE) {
                        // Queue full: do one pop instead, and count it, so the
                        // balancing work shows up in the pop rate and items/s.
                        // The popped key and value are forced live as in the
                        // consumer's pop below.
                        K key;
                        if constexpr (key_only) key = q.pop();
                        else key = q.pop(value);
                        if (key != K{}) ++c; else ++cmiss;
                        benchmark::DoNotOptimize(key);
                        if constexpr (!key_only) benchmark::DoNotOptimize(value);
                    } // if balancing
                }
            } else {
                K key;
                if constexpr (key_only) key = q.pop();
                else key = q.pop(value);
                if (key != K{}) ++c;
                else {
                    ++cmiss;
                    if constexpr (BALANCE) {
                        // Queue empty: do one push instead (counted, as above).
                        bool pushed;
                        if constexpr (key_only) pushed = q.push(&v[p & mask]);
                        else pushed = q.push(static_cast<K>(p*key_stride + key_offset), static_cast<V>(p));
                        if (pushed) ++p; else ++pmiss;
                    }
                }
                // Force the popped key (and value) to count as "used". The
                // queue mutation in pop() is itself observable, but the key is
                // otherwise only tested against Key{}, so without this the
                // compiler could discard the loaded value and shrink the
                // measured work.
                benchmark::DoNotOptimize(key);
                if constexpr (!key_only) benchmark::DoNotOptimize(value);
            } // pop
            if (n == mark) [[unlikely]] {
                const Clock::time_point now = Clock::now();
                if (n == 1) rec.first = now;
                if (n == last) {
                    rec.last = now;
                    rec.items = p + c;
                }
                mark = last;
            } // if a time stamp is due
        } // timed loop
        out = OpCounts{p, c, pmiss, cmiss};
    } // the one batch
} // timed_loop()

// MPMC throughput benchmark. Google Benchmark runs this function
// concurrently on every thread of the run; odd thread indices produce, even
// ones consume (see timed_loop() for the operations and BALANCE).
// V == void selects the key-only queue (K must be a pointer); otherwise the
// key/value queue (K and V must be integers). state.range(0) is the capacity
// in slots, rounded down to a power of 2.
//
// Thread 0 creates the queue and the per-thread records before the timed loop
// and computes the counters and destroys them after it. That is safe because
// the loop's start and end are barriers across all threads of a run in Google
// Benchmark (State::StartKeepRunning() and State::FinishKeepRunning()): the
// setup happens before any thread's first iteration, and every thread has
// written its record and stopped iterating before thread 0 gets past the end.
// Each run (every calibration round and the warm-up included) starts from a
// fresh queue, prefilled to g_queue_fill of its capacity (see STARTING STATE).
template <typename K, typename V, size_t NTRY, size_t ALIGN, bool BALANCE = false>
void BM_MP_MC(benchmark::State& state) {
    using Q = RingAtomicMapQueueMPMC<K, V, NTRY, ALIGN>;
    constexpr bool key_only = std::is_same_v<V, void>;
    // Function-local statics share the queue and the records across all
    // benchmark threads of a run; see above for why the barriers make that
    // safe.
    static Q* q {};
    static void* memory {};
    static ThreadStats* stats {};
    constexpr size_t elem_size = Q::element_size();
    const size_t N = std::bit_floor(static_cast<size_t>(state.range(0)));
    const size_t NB = N*elem_size;
    const int tid = state.thread_index();
    const int nthreads = state.threads();
    const bool producer = tid & 1;      // Odd threads produce, even consume
    std::unique_ptr<ThreadStats[]> stats_owner;         // thread 0 only: owns the per-thread records for this run
    std::vector<int> fill_keys;         // thread 0, key-only: the pointees of the prefilled elements
    if (tid == 0) {
        memory = ::operator new(NB, std::align_val_t{Q::element_align()});
        q = new Q(memory, NB);
        CHECK_EQ(q->capacity(), N);
        if constexpr (key_only) {
            if (0) test(q);     // Smoke test, kept compiling; flip to 1 when debugging
        }
        const size_t fill = static_cast<size_t>(g_queue_fill*static_cast<double>(N));
        CHECK((prefill<Q, K, V>(*q, fill, static_cast<size_t>(nthreads) + 1, fill_keys)));
        stats_owner = std::make_unique<ThreadStats[]>(static_cast<size_t>(nthreads));
        stats = stats_owner.get();
    }
    // The key-only producers' array of pointees (see timed_loop()), and the
    // BALANCE consumers', whose balancing push uses it; empty otherwise.
    std::vector<int> v(key_only && (producer || BALANCE) ? kPointees : 0);
    for (size_t i = 0; i != v.size(); ++i) v[i] = static_cast<int>(i);
    const size_t mask = kPointees - 1;  // index mask of v

    OpCounts counts;
    const size_t key_stride = static_cast<size_t>(nthreads) + 1;
    const size_t key_offset = static_cast<size_t>(tid) + 1;
    if (producer) timed_loop<Q, K, V, true, BALANCE>(state, q, stats, tid, v.data(), mask, key_stride, key_offset, counts);
    else timed_loop<Q, K, V, false, BALANCE>(state, q, stats, tid, v.data(), mask, key_stride, key_offset, counts);

    state.SetItemsProcessed(counts.p + counts.c);
    state.counters["push"] = benchmark::Counter(static_cast<double>(counts.p), benchmark::Counter::kIsRate);
    state.counters["push_fail"] = benchmark::Counter(static_cast<double>(counts.pmiss), benchmark::Counter::kIsRate);
    state.counters["pop"] = benchmark::Counter(static_cast<double>(counts.c), benchmark::Counter::kIsRate);
    state.counters["pop_fail"] = benchmark::Counter(static_cast<double>(counts.cmiss), benchmark::Counter::kIsRate);
    // Past the end-of-loop barrier: no thread touches the queue any more, and
    // every thread's record is written. Thread 0 turns the records into the
    // counters of THROUGHPUT ACCOUNTING above: the span and the spread by
    // gb_wall_clock.h's wall_span() and set_wall_counters(), the item total
    // and the per-role ends here. They are plain values, not rates, because
    // Google Benchmark would divide a rate by the mean loop time again; only
    // thread 0 sets them, so summing the counters over threads leaves them
    // unchanged.
    if (tid == 0) {
        const WallSpan ws = wall_span(stats, nthreads);
        Clock::time_point push_finish = stats[1].last;          // latest producer last-iteration end
        Clock::time_point pop_finish = stats[0].last;           // latest consumer last-iteration end
        size_t items = 0;                                       // successful operations of all threads
        for (int i = 0; i != nthreads; ++i) {
            if (i & 1) push_finish = std::max(push_finish, stats[i].last);
            else pop_finish = std::max(pop_finish, stats[i].last);
            items += stats[i].items;
        } // loop over the threads' records
        // The span is empty when no record is stamped, which this benchmark
        // does not produce (every thread runs timed_loop() to the end), and
        // otherwise only in a run of one iteration per thread in which every
        // stamp coincides; such runs are calibration rounds, which are not
        // reported, and they get zeros, here and in set_wall_counters().
        const auto fraction = [&ws](Clock::time_point t) {
            return ws.seconds > 0 ? std::chrono::duration<double>(t - ws.start).count()/ws.seconds : 0.0;
        };
        set_wall_counters(state, ws, static_cast<double>(items));
        state.counters["push_end"] = benchmark::Counter(fraction(push_finish));
        state.counters["pop_end"] = benchmark::Counter(fraction(pop_finish));

        // Whatever is left in the queue (the prefill plus all pushes minus all
        // pops); report it as a percentage of capacity.
        size_t rem = 0;
        if constexpr (key_only) {
            while (q->pop()) ++rem;
        } else {
            [[maybe_unused]] V value {};
            while (q->pop(value) != K{}) ++rem;
        }
        state.counters["rem%"] = benchmark::Counter(100.0*rem/N);
        delete q; q = nullptr;
        ::operator delete(memory, std::align_val_t{Q::element_align()});
        memory = nullptr;
        stats = nullptr;                // the records are freed with stats_owner
    } // thread 0 results
} // BM_MP_MC()

// A registered variant: its run name and its instantiation of BM_MP_MC().
// concurrent_queue_mbm.C runs the same variants under the same names.
struct Variant {
    const char* name;                   // the run name, shared with the hand-rolled twin
    void (*fn)(benchmark::State&);      // the BM_MP_MC() instantiation
};

// The variants, in registration order at each thread count. The balanced
// rows follow the unbalanced ones (see "Why the balanced rows were added"
// above).
static constexpr Variant kVariants[] = {
    {"BM_MP_MC_void_8", BM_MP_MC<int*, void, 8, 0>},            // Packed: the slot's natural alignment
    //{"BM_MP_MC_void_8_a8", BM_MP_MC<int*, void, 8, 8>},
    {"BM_MP_MC_void_8_a16", BM_MP_MC<int*, void, 8, 16>},       // One slot per 16 bytes
    //{"BM_MP_MC_void_8_a32", BM_MP_MC<int*, void, 8, 32>},
    {"BM_MP_MC_void_8_a64", BM_MP_MC<int*, void, 8, 64>},       // One slot per 64-byte cache line
    {"BM_MP_MC_void_8_a128", BM_MP_MC<int*, void, 8, 128>},     // One slot per adjacent-line prefetch pair
    //{"BM_MP_MC_void_8_a256", BM_MP_MC<int*, void, 8, 256>},
    // Key/value queue at the same alignments as the key-only rows above.
    {"BM_MP_MC_kv_8", BM_MP_MC<uint64_t, uint64_t, 8, 0>},
    {"BM_MP_MC_kv_8_a16", BM_MP_MC<uint64_t, uint64_t, 8, 16>},
    {"BM_MP_MC_kv_8_a64", BM_MP_MC<uint64_t, uint64_t, 8, 64>},
    {"BM_MP_MC_kv_8_a128", BM_MP_MC<uint64_t, uint64_t, 8, 128>},
    // The same variants, balanced: the twins of concurrent_queue_mbm.C's rows.
    {"BM_MP_MC_void_8_balanced", BM_MP_MC<int*, void, 8, 0, true>},
    {"BM_MP_MC_void_8_a16_balanced", BM_MP_MC<int*, void, 8, 16, true>},
    {"BM_MP_MC_void_8_a64_balanced", BM_MP_MC<int*, void, 8, 64, true>},
    {"BM_MP_MC_void_8_a128_balanced", BM_MP_MC<int*, void, 8, 128, true>},
    {"BM_MP_MC_kv_8_balanced", BM_MP_MC<uint64_t, uint64_t, 8, 0, true>},
    {"BM_MP_MC_kv_8_a16_balanced", BM_MP_MC<uint64_t, uint64_t, 8, 16, true>},
    {"BM_MP_MC_kv_8_a64_balanced", BM_MP_MC<uint64_t, uint64_t, 8, 64, true>},
    {"BM_MP_MC_kv_8_a128_balanced", BM_MP_MC<uint64_t, uint64_t, 8, 128, true>},
};

// The number of configured CPUs, or 0 if it is unavailable.
static size_t get_thread_count() {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<size_t>(si.dwNumberOfProcessors);
#else
    const long n = sysconf(_SC_NPROCESSORS_CONF);
    return n > 0 ? static_cast<size_t>(n) : 0;
#endif
} // get_thread_count()

// Capacity, in slots: 65536 by default, or the value of the CQ_CAP environment
// variable (any strtoull base-0 number, e.g. CQ_CAP=0x4000000; rounded down to
// a power of 2 by BM_MP_MC, while the benchmark name shows the value as
// given). An environment variable rather than more registrations because the
// interesting capacities differ by run, and each one multiplies the sweep:
// 2^16 slots keeps the queue at its full or empty boundary, while 2^22-2^26
// slots (0.25-8 GB at a64/a128) is the regime of a buffer sized so that
// wraparound under a stalled thread is out of reach. concurrent_queue_mbm.C
// reads the same variable the same way.
// Returns the capacity to register: 65536, or CQ_CAP if it is set. A value
// that is not a whole number, or is outside [8, 2^56], ends the program with
// a message: 8 is the queue's minimum, and 2^56 slots of the largest slot (128
// bytes) is the most whose size in bytes fits a size_t; outside them the queue
// would abort in the constructor with no explanation. A value that is not a
// power of 2 is reported with the power of 2 it will be rounded down to.
// main() calls this after MaybeReenterWithoutASLR(): Google Benchmark (1.9.x)
// re-executes the program at startup with address-space randomization turned
// off, and anything that ran before the re-execution (a static initializer,
// say) runs, and prints, twice.
static size_t get_queue_capacity() {
    const char* env = std::getenv("CQ_CAP");
    if (!env) return 1UL << 16;
    char* end = nullptr;
    const unsigned long long cap = std::strtoull(env, &end, 0);
    if (end == env || *end != '\0' || cap < 8 || cap > (1ULL << 56)) {
        std::fprintf(stderr, "CQ_CAP=%s: expected a whole number of slots in [8, 2^56]\n", env);
        std::exit(1);
    }
    if (!std::has_single_bit(cap)) {
        std::fprintf(stderr, "CQ_CAP=%s: not a power of 2, using %llu slots\n", env, std::bit_floor(cap));
    }
    return static_cast<size_t>(cap);
} // get_queue_capacity()

// The fraction of the capacity to fill before the threads start: 0, or the
// value of the CQ_FILL environment variable, a number in [0, 1] (e.g.
// CQ_FILL=0.5). A value that is not one ends the program with a message.
static double get_queue_fill() {
    const char* env = std::getenv("CQ_FILL");
    if (!env) return 0;
    char* end = nullptr;
    errno = 0;
    const double fill = std::strtod(env, &end);
    if (end == env || *end != '\0' || errno == ERANGE || !(fill >= 0 && fill <= 1)) {
        std::fprintf(stderr, "CQ_FILL=%s: expected a fraction of the capacity in [0, 1]\n", env);
        std::exit(1);
    }
    return fill;
} // get_queue_fill()

// Register every variant at every thread count, then run what the command line
// selects. This replaces BENCHMARK_MAIN() because the time limits have to be
// set per thread count (see MEASUREMENT WINDOW above). Shared registration
// settings:
//   UseRealTime() -- report wall-clock, not summed CPU time. A multithreaded
//                    benchmark accrues CPU time on every thread, so CPU-time
//                    reporting would divide throughput by the thread count.
//   Arg(capacity) -- capacity is a single point, queue_capacity slots.
int main(int argc, char** argv) {
    // Same start-up as BENCHMARK_MAIN(), including its re-exec without ASLR.
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

    // At least 2: one producer and one consumer (and BM_MP_MC's records of
    // both roles), even where the CPU count is 1 or unavailable.
    const size_t thread_count = std::max<size_t>(get_thread_count(), 2);
    const size_t queue_capacity = get_queue_capacity();
    g_queue_fill = get_queue_fill();
    benchmark::AddCustomContext("cq_cap", std::to_string(std::bit_floor(queue_capacity)));
    benchmark::AddCustomContext("cq_fill", std::to_string(g_queue_fill));
    // The thread counts: the powers of two from 2 up to the CPU count, then
    // the CPU count itself if it is not a power of two (the counts of
    // ThreadRange(2, thread_count)).
    std::vector<size_t> thread_counts;
    for (size_t t = 2; t <= thread_count; t *= 2) thread_counts.push_back(t);
    if (thread_counts.empty() || thread_counts.back() != thread_count) thread_counts.push_back(thread_count);

    // Thread count outside, variant inside: at each thread count all variants
    // run back to back, in the order of kVariants.
    for (size_t t : thread_counts) {
        for (const Variant& variant : kVariants) {
            benchmark::RegisterBenchmark(variant.name, variant.fn)
                ->Arg(static_cast<int64_t>(queue_capacity))
                ->Threads(static_cast<int>(t))
                ->UseRealTime()
                ->MinTime(kWindowSeconds*static_cast<double>(t))
                ->MinWarmUpTime(kWarmupSeconds*static_cast<double>(t));
        } // loop over variants
    } // loop over thread counts

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
} // main()
