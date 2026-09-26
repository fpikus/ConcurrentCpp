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
// of this file, and produces the same measurement without the framework).
//
// What it measures: aggregate enqueue+dequeue throughput (items/s) of the
// queue under MPMC contention, in both of its modes: key-only (an int* key,
// Value = void) and key/value (uint64_t key, uint64_t value). Google
// Benchmark runs BM_MP_MC simultaneously on every thread of the ThreadRange,
// every thread for the same number of iterations, and reports items/s from
// SetItemsProcessed. This is a throughput probe, NOT latency -- for per-op
// latency see concurrent_queue_lmbm.C (contended push->pop handoff) and
// concurrent_queue_ppmbm.C (1-producer/1-consumer round trip).
//
// How long a multithreaded run really is. Google Benchmark picks the iteration
// count by comparing --benchmark_min_time against the real time SUMMED over
// all threads, and every thread runs that many iterations. So
// --benchmark_min_time=1s at 128 threads is a measurement window of about
// 1/128 s, 8 ms per thread: far too short for a queue to leave its empty
// start-up state, or for the lock's 1 ms back-off tier to happen more than a
// few times. An iteration count, by contrast, is per thread:
// --benchmark_min_time=<N>x runs exactly N iterations on every thread. For
// numbers that mean something at high thread counts, run one thread count at a
// time (--benchmark_filter='.../threads:<T>$') with an N chosen for that
// thread count -- a short time-based run tells you the rate, from which N
// follows for the window you want (a second or more per cell is a reasonable
// target). Two counters need the same care: with equal iteration counts and
// equal numbers of producers and consumers, push_fail - pop_fail == -rem (as
// counts) by construction, so the fail counters do not tell you which side was
// waiting; and rem% is a snapshot of the queue at the end of a run, not a
// steady-state occupancy.
//
// Thread roles are STATIC and split by index parity: odd threads push, even
// threads pop (state.thread_index() & 1). ThreadRange yields powers of two up
// to the hardware thread count, so producers and consumers are balanced at
// every (even) thread count in the sweep. (This is the opposite of the lmbm
// harness, whose roles are dynamic and self-balancing.)
//
// The registered sweep fixes capacity (65536 slots unless the CQ_CAP
// environment variable says otherwise; see ARGS) and varies only the
// per-slot ALIGN template parameter (0/16/64/128) to study how slot
// cache-line alignment trades off against contention; NTRY is pinned at 8.
// The commented-out registrations (a8/a32/a256 and the BALANCE variant) are
// kept ready for ad-hoc runs.
//
// Why the key/value (kv) variants were added: the two modes share nothing
// below the locks. Key-only push/pop store and clear the key inside the
// critical sections; key/value push/pop claim the slot under the lock and
// construct/move the value outside it, with a separate per-slot protocol
// for the handoff. The key-only rows therefore say nothing about the cost
// of the key/value protocol, and changing that protocol needs its own
// throughput numbers, before and after. Keys are distinct per element
// (see BM_MP_MC); the values are trivially copyable, so the rows measure
// the protocol, not the payload's constructors.

#include "concurrent_queue.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>
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

// MPMC throughput benchmark. Google Benchmark runs this function
// concurrently on every thread of the ThreadRange; odd thread indices
// produce, even ones consume. BALANCE: after counting a miss, additionally
// perform (and count) the opposite operation, so the queue cannot settle
// permanently full (producers outnumber effective consumers) or empty.
// V == void selects the key-only queue (K must be a pointer: producers push
// pointers into a per-thread array); otherwise the key/value queue (K and V
// must be integers: each element carries a distinct non-zero key and its
// producer's counter as the value).
template <typename K, typename V, size_t NTRY, size_t ALIGN, bool BALANCE = false>
void BM_MP_MC(benchmark::State& state) {
    using Q = RingAtomicMapQueueMPMC<K, V, NTRY, ALIGN>;
    constexpr bool key_only = std::is_same_v<V, void>;
    // Function-local statics share the single queue across all benchmark
    // threads. Thread 0 creates it before entering the state loop (whose
    // first iteration is a start barrier for all threads) and destroys it
    // after the loop (which ends with a stop barrier), so no other thread
    // can observe a half-built or deleted queue.
    static Q* q {};
    static void* memory {};
    constexpr size_t elem_size = Q::element_size();
    const size_t N = std::bit_floor(static_cast<size_t>(state.range(0)));
    const size_t mask = N - 1;
    const size_t NB = N*elem_size;
    if (state.thread_index() == 0) {
        memory = ::operator new(NB, std::align_val_t{Q::element_align()});
        q = new Q(memory, NB);
        CHECK_EQ(q->capacity(), N);
        if constexpr (key_only) {
            if (0) test(q);     // Smoke test, kept compiling; flip to 1 when debugging
        }
    }
    // Per-thread backing array: the key-only queue stores int*, so each
    // producer pushes pointers into its own vector -- valid, distinct, never
    // dereferenced by consumers. (Allocated but unused by the key/value
    // queue: keeping the key-only path's code unchanged matters more.)
    std::vector<int> v(N);
    for (size_t i = 0; i != N; ++i) v[i] = i;
    const bool producer = state.thread_index() & 1;     // Odd threads produce, even consume
    const size_t nthreads = state.threads();            // Key/value only: key generation
    const size_t tid = state.thread_index();            // Key/value only: key generation
    // Per-thread destination for popped values (key/value queue only; an
    // unused int for the key-only queue).
    [[maybe_unused]] std::conditional_t<key_only, int, V> value {};

    // The two modes differ only in the push/pop calls, selected in place by
    // if constexpr. (Not factored into lambdas: g++-16 then compiles the
    // key-only instantiation differently -- different frame and
    // inlining -- and the key-only rows must stay comparable with runs of
    // the version of this file that predates the key/value rows.)
    // The key/value key p*nthreads + tid + 1 is distinct across all threads'
    // elements and never Key{} (0), the reserved empty marker; the value is
    // the producer's counter.
    size_t p = 0, c = 0, pmiss = 0, cmiss = 0;
    for (auto _ : state) {
        if (producer) {
            bool pushed;
            if constexpr (key_only) pushed = q->push(&v[p & mask]);
            else pushed = q->push(static_cast<K>(p*nthreads + tid + 1), static_cast<V>(p));
            if (pushed) ++p;
            else {
                ++pmiss;
                if constexpr (BALANCE) {
                    // Queue full: do one pop instead, and count it, so the
                    // balancing work shows up in the pop rate and items/s.
                    K key;
                    if constexpr (key_only) key = q->pop();
                    else key = q->pop(value);
                    if (key != K{}) ++c; else ++cmiss;
                }
            }
        } else {
            K key;
            if constexpr (key_only) key = q->pop();
            else key = q->pop(value);
            if (key != K{}) ++c;
            else {
                ++cmiss;
                if constexpr (BALANCE) {
                    // Queue empty: do one push instead (counted, as above).
                    bool pushed;
                    if constexpr (key_only) pushed = q->push(&v[0]);
                    else pushed = q->push(static_cast<K>(tid + 1), static_cast<V>(0));
                    if (pushed) ++p; else ++pmiss;
                }
            }
            // Force the popped key (and value) to count as "used". The queue
            // mutation in pop() is itself observable, but the key is
            // otherwise only tested against Key{}, so without this the
            // compiler could discard the loaded value and shrink the
            // measured work.
            benchmark::DoNotOptimize(key);
            if constexpr (!key_only) benchmark::DoNotOptimize(value);
        }
    } // benchmark loop
    state.SetItemsProcessed(p + c);
    state.counters["push"] = benchmark::Counter(static_cast<double>(p), benchmark::Counter::kIsRate);
    state.counters["push_fail"] = benchmark::Counter(static_cast<double>(pmiss), benchmark::Counter::kIsRate);
    state.counters["pop"] = benchmark::Counter(static_cast<double>(c), benchmark::Counter::kIsRate);
    state.counters["pop_fail"] = benchmark::Counter(static_cast<double>(cmiss), benchmark::Counter::kIsRate);
    if (state.thread_index() == 0) {
        // Whatever is left in the queue is how far producers outran
        // consumers; report it as a percentage of capacity.
        size_t rem = 0;
        if constexpr (key_only) {
            while (q->pop()) ++rem;
        } else {
            while (q->pop(value) != K{}) ++rem;
        }
        state.counters["rem%"] = benchmark::Counter(100.0 * rem / N);
        delete q; q = nullptr;
        ::operator delete(memory, std::align_val_t{Q::element_align()});
        memory = nullptr;
    }
} // BM_MP_MC()

void BM_MP_MC_void_8(benchmark::State& state) { BM_MP_MC<int*, void, 8, 0>(state); }
void BM_MP_MC_void_8_balanced(benchmark::State& state) { BM_MP_MC<int*, void, 8, 0, true>(state); }
void BM_MP_MC_void_8_a8(benchmark::State& state) { BM_MP_MC<int*, void, 8, 8>(state); }
void BM_MP_MC_void_8_a16(benchmark::State& state) { BM_MP_MC<int*, void, 8, 16>(state); }
void BM_MP_MC_void_8_a32(benchmark::State& state) { BM_MP_MC<int*, void, 8, 32>(state); }
void BM_MP_MC_void_8_a64(benchmark::State& state) { BM_MP_MC<int*, void, 8, 64>(state); }
void BM_MP_MC_void_8_a128(benchmark::State& state) { BM_MP_MC<int*, void, 8, 128>(state); }
void BM_MP_MC_void_8_a256(benchmark::State& state) { BM_MP_MC<int*, void, 8, 256>(state); }
void BM_MP_MC_kv_8(benchmark::State& state) { BM_MP_MC<uint64_t, uint64_t, 8, 0>(state); }
void BM_MP_MC_kv_8_a16(benchmark::State& state) { BM_MP_MC<uint64_t, uint64_t, 8, 16>(state); }
void BM_MP_MC_kv_8_a64(benchmark::State& state) { BM_MP_MC<uint64_t, uint64_t, 8, 64>(state); }
void BM_MP_MC_kv_8_a128(benchmark::State& state) { BM_MP_MC<uint64_t, uint64_t, 8, 128>(state); }

static size_t get_thread_count() {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<size_t>(si.dwNumberOfProcessors);
#else
    return static_cast<size_t>(sysconf(_SC_NPROCESSORS_CONF));
#endif
}
static const size_t thread_count = get_thread_count();

// Shared registration args for every variant:
//   UseRealTime()  -- report wall-clock, not summed CPU time. A multithreaded
//                     benchmark accrues CPU time on every thread, so CPU-time
//                     reporting would divide throughput by the thread count.
//   ThreadRange(2, thread_count) -- run at 2,4,8,... threads up to hardware.
//   Range(cap, cap) -- capacity is a single point, queue_capacity slots;
//                     RangeMultiplier is therefore vestigial here (one value).
//
// Capacity, in slots: 65536 by default, or the value of the CQ_CAP environment
// variable (any strtoull base-0 number, e.g. CQ_CAP=0x4000000; rounded down to
// a power of 2 by BM_MP_MC, while the benchmark name shows the value as
// given). An environment variable rather than more registrations because the
// interesting capacities differ by run, and each one multiplies the sweep:
// 2^16 slots keeps the queue at its full or empty boundary, while 2^22-2^26
// slots (0.25-8 GB at a64/a128) is the regime of a buffer sized so that
// wraparound under a stalled thread is out of reach.
// Returns the capacity to register: 65536, or CQ_CAP if it is set. A value
// that is not a whole number, or is outside [8, 2^62], ends the program with
// a message: 8 is the queue's minimum (it would otherwise abort in the
// constructor with no explanation), and Google Benchmark carries the value as
// an int64_t. A value that is not a power of 2 is reported with the power of
// 2 it will be rounded down to -- twice, as a rule: Google Benchmark (1.9.x)
// re-executes the program at startup with address-space randomization turned
// off, so every static initializer, this one included, runs twice.
static size_t get_queue_capacity() {
    const char* env = std::getenv("CQ_CAP");
    if (!env) return 1UL << 16;
    char* end = nullptr;
    const unsigned long long cap = std::strtoull(env, &end, 0);
    if (end == env || *end != '\0' || cap < 8 || cap > (1ULL << 62)) {
        std::fprintf(stderr, "CQ_CAP=%s: expected a whole number of slots in [8, 2^62]\n", env);
        std::exit(1);
    }
    if (!std::has_single_bit(cap)) {
        std::fprintf(stderr, "CQ_CAP=%s: not a power of 2, using %llu slots\n", env, std::bit_floor(cap));
    }
    return static_cast<size_t>(cap);
} // get_queue_capacity()
static const size_t queue_capacity = get_queue_capacity();
#define ARGS \
  ->UseRealTime() \
  ->ThreadRange(2, thread_count) \
  ->RangeMultiplier(2)->Range(queue_capacity, queue_capacity)

BENCHMARK(BM_MP_MC_void_8) ARGS;          // Best on M3 Ultra, competitive on Grace/AMD
//BENCHMARK(BM_MP_MC_void_8_balanced) ARGS;
//BENCHMARK(BM_MP_MC_void_8_a8) ARGS;
BENCHMARK(BM_MP_MC_void_8_a16) ARGS;      // Best on M3 Ultra (tied), good all-rounder
//BENCHMARK(BM_MP_MC_void_8_a32) ARGS;
BENCHMARK(BM_MP_MC_void_8_a64) ARGS;      // Best on Grace 2t, Zen3 mid-high, Cascade Lake 32t
BENCHMARK(BM_MP_MC_void_8_a128) ARGS;     // Best on Granite Rapids, Zen5 high threads
//BENCHMARK(BM_MP_MC_void_8_a256) ARGS;
// Key/value queue at the same alignments as the key-only rows above.
BENCHMARK(BM_MP_MC_kv_8) ARGS;
BENCHMARK(BM_MP_MC_kv_8_a16) ARGS;
BENCHMARK(BM_MP_MC_kv_8_a64) ARGS;
BENCHMARK(BM_MP_MC_kv_8_a128) ARGS;

BENCHMARK_MAIN();
