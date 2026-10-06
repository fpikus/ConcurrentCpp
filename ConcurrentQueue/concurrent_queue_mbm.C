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
// Hand-rolled twin of concurrent_queue_gmbm.C (mbm = the hand-rolled
// MicroBenchMark): the same THROUGHPUT benchmark of RingAtomicMapQueueMPMC
// under MPMC contention, measured with every thread contending for the whole
// measurement window.
//
// WHAT IS MEASURED
//
// The same operation as the balanced rows of concurrent_queue_gmbm.C
// (*_balanced), on the same variants, under the same names: key-only (int*
// keys) and key/value (uint64_t keys and values) queues at slot alignments
// 0/16/64/128, NTRY 8, odd threads pushing and even threads popping, each push
// or pop counted as a success or a miss, and every miss followed by one
// operation of the other role (a push that finds the queue full is followed
// by a pop, a pop that finds it empty by a push), also counted. The loop body
// (run_phase() here, timed_loop() there with BALANCE) is the same code,
// noinline in both, with the popped key and value forced live by the same
// compiler barrier (do_not_optimize() here is benchmark::DoNotOptimize()
// for GCC and Clang). The loops differ in their exit test: a load of the stop
// flag here, a compare of the iteration count (plus the time-stamp compare)
// there. The capacity comes from the same CQ_CAP environment variable, read
// the same way (65536 slots by default).
//
// WHY A SECOND HARNESS
//
// Google Benchmark runs a fixed number of iterations in every thread, failed
// pushes and pops included. Under the batching SpinLock the threads that win
// the streaks finish early and leave, and a role whose iterations are cheaper
// can finish altogether and leave the other role to run alone against a
// queue that nobody drains or fills; the tail of the run then measures a
// different regime from its start. concurrent_queue_gmbm.C reports how much
// (finish_spread, push_end, pop_end; THROUGHPUT ACCOUNTING there) but cannot
// avoid it. Here every thread runs until a common stop flag and counts its
// own operations, so all threads contend for the whole window, and the
// throughput is the true total count divided by the window.
//
// WHY BALANCED
//
// Equal time is not equal work. With static roles and a fixed iteration count
// (gmbm), producers and consumers attempt the same number of operations. Run
// for the same time with static roles, the role that attempts faster pins the
// queue at a boundary -- full, for instance, with the consumers almost never
// failing and the producers' failing retries slowing them -- a slower regime
// than gmbm's, whose consumers keep finding the queue empty; starting
// half-full does not change where the queue settles. With every miss followed
// by an operation of the other role, neither boundary can hold the queue, and
// equal time and equal work measure the same regime. So this harness runs
// only the balanced rows, and gmbm runs them next to its unbalanced ones.
//
// HOW THE NUMBERS RELATE
//
// items_per_s here corresponds to wall_items_per_second of the *_balanced
// rows of concurrent_queue_gmbm.C, not to their items_per_second (which
// divides by the mean per-thread loop time); likewise the push and pop rates.
// Where gmbm's finish_spread is large, this harness gives the rate at a
// constant thread count. push_fail_per_s, pop_fail_per_s and rem_pct show
// where the queue sat (CQ_FILL, below, sets where a run starts). rem_pct is
// the queue's occupancy when the stop flag is set (a snapshot, as gmbm's
// rem%), but here every thread was running up to that point. Fairness shows
// up here as unequal per-thread counts rather than as a finish spread. It is
// reported per starting role, producers (odd threads) and consumers (even
// threads), over each thread's successful operations, pushes and pops:
//   cv       -- the coefficient of variation of the per-thread counts
//               (population standard deviation divided by the mean): 0 when
//               every thread of the role did the same number of operations;
//   min_rel  -- the smallest per-thread count divided by the mean: near 0 when
//               some thread was starved for most of the window;
//   max_rel  -- the largest per-thread count divided by the mean: up to the
//               role's thread count when one thread did nearly everything.
// All three are 0, 1, 1 for a role with one thread (at 2 and 3 threads),
// unless that thread did nothing at all (then 0, 0, 0).
//
// PROTOCOL OF ONE RUN
//
// A run is one variant at one thread count. The main thread, which is not one
// of the workers:
//   1. allocates the queue (whose constructor zeroes, and so faults in, the
//      ring) and the key-only threads' arrays, prefills the queue to the
//      CQ_FILL fraction of its capacity (0 by default, as in
//      concurrent_queue_gmbm.C, where every run starts that full), resets the
//      flags, and starts the workers, which wait on a start latch;
//   2. arrives at the latch, which releases all workers at once;
//   3. sleeps for the warm-up time: the workers run the loop uncounted;
//   4. reads the clock (t0) and sets the `measure` flag: each worker, on seeing
//      it, notes its running counts and goes on running the same loop;
//   5. sleeps for the window, reads the clock (t1) and sets the `stop` flag:
//      each worker, on seeing it, stores its counts since step 4 and exits;
//   6. joins the workers, drains the queue (counting what was left), and frees
//      the queue and the arrays.
// items is the sum of the successful pushes and pops, window_s is t1 - t0,
// measured with steady_clock, and items_per_s is items/window_s. Each clock
// reading immediately precedes the flag store it times, so both ends of the
// window see the same delay from reading to store. The window is whatever the
// main thread actually slept, so a late wake-up lengthens the window but does
// not bias the rate.
//
// The start latch is std::latch: its waiters block (after at most a brief spin
// inside the standard library) rather than spin, which keeps them from
// competing for CPUs with the main thread while it is still creating the other
// workers. Their wake-up is staggered, but that falls into the warm-up, which
// is why the warm-up runs on the same threads and flows into the window without
// a barrier: when the window opens, every thread is already running and the
// contention is established, and the queue has left its empty start-up state.
// (A warm-up of 0 puts the latch wake-up into the window.)
//
// Each worker checks the flag once per operation, after it: a relaxed load of
// the flag block, which is written only twice per run and so stays in the
// Shared state in every reader's cache. The flags are on a 128-byte block of
// their own, so reading them never touches the queue's lines. A worker that
// sees a flag counts the operation it just finished in the phase it just left.
// So the operation in flight when a flag is set, and those completed while the
// flag store propagates (a cache-line transfer), are counted on the wrong side
// of it: at t0 they are left out of the window, at t1 added to it, at most a
// few operations per thread at each end. The two ends cancel to first order,
// and even without cancelling they are a handful per thread against millions
// of operations per window. They are not corrected for.
//
// The flags are std::atomic, so the relaxed loads are race-free and see the
// stores eventually, which is all the protocol needs: no other data passes
// through them. Each worker writes its counts to its own element of a vector
// once, before it returns, and the completion of a thread synchronizes-with
// the return of its join(), so the main thread reads the counts after the
// joins without a data race.
//
// ORDER
//
// For each repetition, for each thread count, every selected variant, in the
// order of kVariants. The variant order is reversed whenever the repetition
// number plus the index of the thread count in the list is odd, so it
// alternates from one thread count to the next and from one repetition to the
// next, and a slow drift in the machine (clock, temperature) biases no variant
// more than another. There is no random shuffle.
//
// RUNNING IT
//
//   ./concurrent_queue_mbm [--window=SECONDS] [--warmup=SECONDS] [--reps=N]
//                          [--threads=T1,T2,...] [--filter=REGEX]
//   CQ_CAP=0x400000 ./concurrent_queue_mbm      # 2^22 slots
//   CQ_FILL=0.5 ./concurrent_queue_mbm --warmup=0  # the window starts half-full
//
// The defaults are the Google Benchmark harness's window and warm-up (1 s and
// 0.5 s), 1 repetition, and the thread counts of the Google Benchmark harness:
// the powers of two from 2 up to the CPU count, then the CPU count itself.
// Every thread count must be at least 2 (one producer, one consumer). --filter
// selects variants by a regex search on the variant name:
//
//   ./concurrent_queue_mbm --filter='_void_'          # the key-only rows
//   ./concurrent_queue_mbm --filter='_kv_'            # the key/value rows
//   ./concurrent_queue_mbm --filter='_a64_' --threads=128 --reps=5 > a64.csv
//
// stdout is CSV only: a header line, then one line per run, written as each
// run finishes:
//   variant,threads,rep,window_s,items,items_per_s,push_per_s,pop_per_s,
//   push_fail_per_s,pop_fail_per_s,rem_pct,producer_cv,producer_min_rel,
//   producer_max_rel,consumer_cv,consumer_min_rel,consumer_max_rel
// (one line in the output; rep counts from 0). stderr gets the parameters at
// the start and, at the end, the median of the main columns (all but items
// and the max_rel columns) over the repetitions of each (variant, threads)
// cell.
//
// The exit status is 0 on success, 1 if a run or the output failed, and 2 for
// a bad command line, CQ_CAP or CQ_FILL.

#include "concurrent_queue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <latch>
#include <memory>
#include <new>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <thread>
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
  #include <intrin.h>
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

namespace {

// Wall-clock seconds of measurement and of warm-up per run: the fixed values of
// concurrent_queue_gmbm.C, and the defaults here. Keep the two files in step.
constexpr double kWindowSeconds = 1.0;
constexpr double kWarmupSeconds = 0.5;

// The number of ints in each key-only thread's array of pointees, a power of
// 2: a thread pushes their addresses in turn. The queue needs its keys
// non-null, not distinct (it only stores them; nobody dereferences them), so
// a small array will do; one of N ints per thread would be 32 GB at 2^26
// slots and 128 threads. concurrent_queue_gmbm.C uses the same value; keep the
// two files in step.
constexpr size_t kPointees = 1024;
static_assert(std::has_single_bit(kPointees));

// Make the compiler treat `value` as read and written, so that the computation
// of a value nobody reads is not optimized away: the same asm as
// benchmark::DoNotOptimize(Tp&) in Google Benchmark 1.9.5, so that the two
// harnesses compile the same barrier (this one does not depend on the
// library). Google Benchmark uses that asm with GCC only for trivially
// copyable types no wider than a pointer, hence the static_assert. The MSVC
// fallback is a compiler barrier and a volatile read.
template <typename T>
[[gnu::always_inline]] inline void do_not_optimize(T& value) {
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(void*),
                  "Google Benchmark's DoNotOptimize() uses a different constraint for this type");
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#elif defined(__GNUC__)
    asm volatile("" : "+m,r"(value) : : "memory");
#else
    static_cast<void>(*static_cast<const volatile T*>(&value));
    _ReadWriteBarrier();
#endif
} // do_not_optimize()

// The flags the main thread uses to move the workers from one phase to the
// next; see PROTOCOL OF ONE RUN above. Each is written once per run, by the
// main thread, and read by every worker after every operation.
struct alignas(128) Flags {
    std::atomic<bool> measure {false};  // set at the start of the measured window
    std::atomic<bool> stop {false};     // set at its end
};
static_assert(sizeof(Flags) == 128);

// The phase flags of the current run, shared by all variants: runs happen one
// after another, and each resets them (step 1). constinit: zero-initialized at
// load time, so there is no run-time guard on the workers' path to them.
constinit Flags g_flags {};

// One thread's operation counts; carried from the warm-up phase into the
// measured one, so that the key/value producers' keys stay distinct.
struct OpCounts {
    size_t p = 0;                       // successful pushes
    size_t c = 0;                       // successful pops
    size_t pmiss = 0;                   // pushes that found the queue full
    size_t cmiss = 0;                   // pops that found the queue empty
};

// The difference of two snapshots of one thread's counts: the operations
// between them.
OpCounts operator-(const OpCounts& a, const OpCounts& b) {
    return OpCounts{a.p - b.p, a.c - b.c, a.pmiss - b.pmiss, a.cmiss - b.cmiss};
}

// One phase of a worker: push (PRODUCER) or pop, after every miss one
// operation of the other role, until the phase's flag reads true (`measure`
// ends the warm-up, `stop` the measured phase). The flag is checked after
// every operation, never before the first, so at least one operation runs.
//   Q, K, V    -- the queue and its key and value types (V == void: key-only);
//   PRODUCER   -- whether this thread pushes (true) or pops (false) first;
//   flag       -- the flag that ends the phase: g_flags.measure for the
//                 warm-up, g_flags.stop for the window;
//   q          -- the queue shared by all threads of the run;
//   v, mask    -- key-only: the thread's array of mask + 1 ints (kPointees),
//                 whose addresses it pushes in turn -- valid, never
//                 dereferenced, and never nullptr, which is Key{}; unused by
//                 the key/value queue;
//   key_stride, key_offset -- key/value: the key of the thread's p-th pushed
//                 element is p*key_stride + key_offset, with key_stride the
//                 thread count plus 1 and key_offset = tid + 1, distinct
//                 across all threads' elements and the prefill's (see
//                 prefill()) and never Key{} (0), the reserved empty marker;
//                 the value is p;
//   counts     -- the thread's counts, advanced by the phase.
// noinline, and the counts copied into locals for the loop and written back
// once at the end: so that the only values live across the loop are the loop's
// own, as in timed_loop() of concurrent_queue_gmbm.C, whose loop body is the
// same. flatten: as in timed_loop() -- with GCC. Clang's flatten inlines
// push() and pop() but not the SpinLock::lock() inside them, which its inliner
// decides on its own: lock() costs more than the default threshold but less
// than the higher one for a call site it estimates to be hot (much more
// frequent than the entry to the function). The [[unlikely]] on the flag test
// below makes that estimate what it is in timed_loop(), whose exit is just as
// unlikely; with a plain do-while on the flag, Clang calls lock() out of line
// here and inlines it there. The phase is a run-time argument, so that there
// is one instantiation per variant and role, as of timed_loop().
template <typename Q, typename K, typename V, bool PRODUCER>
[[gnu::noinline, gnu::flatten]] void run_phase(const std::atomic<bool>& flag, Q& q, int* v, size_t mask,
                                               size_t key_stride, size_t key_offset, OpCounts& counts) {
    constexpr bool key_only = std::is_same_v<V, void>;
    size_t p = counts.p, c = counts.c, pmiss = counts.pmiss, cmiss = counts.cmiss;
    // Destination for popped values (key/value queue only; an unused int for
    // the key-only queue).
    [[maybe_unused]] std::conditional_t<key_only, int, V> value {};
    while (true) {
        if constexpr (PRODUCER) {
            bool pushed;
            if constexpr (key_only) pushed = q.push(&v[p & mask]);
            else pushed = q.push(static_cast<K>(p*key_stride + key_offset), static_cast<V>(p));
            if (pushed) ++p;
            else {
                ++pmiss;
                // Queue full: do one pop instead, and count it; the popped
                // key and value are forced live as in the consumer's pop.
                K key;
                if constexpr (key_only) key = q.pop();
                else key = q.pop(value);
                if (key != K{}) ++c; else ++cmiss;
                do_not_optimize(key);
                if constexpr (!key_only) do_not_optimize(value);
            } // if the push missed
        } else {
            K key;
            if constexpr (key_only) key = q.pop();
            else key = q.pop(value);
            if (key != K{}) ++c;
            else {
                ++cmiss;
                // Queue empty: do one push instead, and count it.
                bool pushed;
                if constexpr (key_only) pushed = q.push(&v[p & mask]);
                else pushed = q.push(static_cast<K>(p*key_stride + key_offset), static_cast<V>(p));
                if (pushed) ++p; else ++pmiss;
            }
            // Force the popped key (and value) to count as "used", as gmbm
            // does: the key is otherwise only tested against Key{}, so the
            // compiler could discard the loaded value and shrink the measured
            // work.
            do_not_optimize(key);
            if constexpr (!key_only) do_not_optimize(value);
        } // pop
        if (flag.load(std::memory_order_relaxed)) [[unlikely]] break;
    } // loop until the flag is set
    counts = OpCounts{p, c, pmiss, cmiss};
} // run_phase()

// The outcome of one run.
struct RunResult {
    double window_s;                    // the measured window, t1 - t0, in seconds
    std::vector<OpCounts> counts;       // operations per thread within the window
    size_t rem;                         // elements left in the queue at the stop
    size_t capacity;                    // the queue's capacity in slots
};

// The main thread's part of a run, steps 1 (starting the threads) to 6 of
// PROTOCOL OF ONE RUN: start the threads, release them at the latch, sleep
// through the warm-up, open the window (t0, then `measure`), sleep through it,
// close it (t1, then `stop`), and join the threads. Returns the measured
// window, t1 - t0, in seconds.
//   start        -- the start latch, which expects `participants` arrivals;
//   participants -- the threads that meet at the latch: every thread that
//                   `launch` starts, and the main thread;
//   warmup_s     -- seconds of uncounted warm-up, at least 0;
//   window_s     -- seconds of measurement requested, more than 0;
//   launch       -- called once with an empty vector, into which it starts
//                   every thread of the run (emplace_back() of a jthread).
// If `launch` throws (a thread could not be started), the threads it did start
// are let finish and joined before the exception propagates: both flags are
// set and the latch is counted down for the threads that were not started and
// for the main thread, so each started thread gets through the latch, sees
// both flags at its first check, and exits. The caller's data that those
// threads use must outlive this call, on both paths.
template <typename Launch>
double run_window(std::latch& start, std::ptrdiff_t participants, double warmup_s, double window_s,
                  Launch&& launch) {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0;               // the start of the window: just before `measure` is set
    Clock::time_point t1;               // its end: just before `stop` is set
    {
        // jthread joins on destruction, at the end of this block or while an
        // exception propagates out of it.
        std::vector<std::jthread> started;
        started.reserve(static_cast<size_t>(participants - 1));
        try {
            launch(started);
        } catch (...) {
            g_flags.measure.store(true, std::memory_order_relaxed);
            g_flags.stop.store(true, std::memory_order_relaxed);
            start.count_down(participants - static_cast<std::ptrdiff_t>(started.size()));
            started.clear();            // joins the threads that did start
            throw;
        } // starting the threads

        start.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::duration<double>(warmup_s));
        t0 = Clock::now();
        g_flags.measure.store(true, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::duration<double>(window_s));
        t1 = Clock::now();
        g_flags.stop.store(true, std::memory_order_relaxed);
    } // threads joined
    return std::chrono::duration<double>(t1 - t0).count();
} // run_window()

// Push `count` elements into the empty queue `q`, from one thread, before any
// worker touches it (the CQ_FILL start; see PROTOCOL OF ONE RUN above).
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

// One run of one variant, following PROTOCOL OF ONE RUN above.
//   K, V, NTRY, ALIGN -- the variant: RingAtomicMapQueueMPMC<K, V, NTRY, ALIGN>;
//   threads           -- the number of workers, at least 2: odd indices push,
//                        even ones pop;
//   warmup_s          -- seconds of uncounted warm-up, at least 0;
//   window_s          -- seconds of measurement requested, more than 0;
//   capacity          -- the queue's capacity in slots, a power of 2, at
//                        least 8;
//   fill              -- the fraction of the capacity to prefill, in [0, 1].
// Returns the measured window, the per-thread counts and the elements left in
// the queue. Throws std::bad_alloc if an allocation fails and
// std::system_error if a thread cannot be started; the threads already started
// are stopped and joined first (see run_window()); std::logic_error if the
// prefill fails, which cannot happen.
template <typename K, typename V, size_t NTRY, size_t ALIGN>
RunResult run_variant(int threads, double warmup_s, double window_s, size_t capacity, double fill) {
    using Q = RingAtomicMapQueueMPMC<K, V, NTRY, ALIGN>;
    constexpr bool key_only = std::is_same_v<V, void>;
    constexpr std::align_val_t elem_align {Q::element_align()};

    // Step 1 of the protocol. The queue, its buffer and the arrays outlive the
    // workers: they are declared before them and freed after they are joined.
    // The queue is destroyed (it drains leftovers) before its buffer is freed.
    const size_t NB = capacity*Q::element_size();
    const auto free_memory = [](void* memory) { ::operator delete(memory, elem_align); };
    const std::unique_ptr<void, decltype(free_memory)> memory(::operator new(NB, elem_align), free_memory);
    const std::unique_ptr<Q> q = std::make_unique<Q>(memory.get(), NB);
    const size_t N = q->capacity();
    const size_t mask = kPointees - 1;  // index mask of the pointee arrays
    // The key-only threads' arrays of pointees (see run_phase()), one per
    // thread, every thread pushing; empty for the key/value queue. Allocated
    // here rather than by the workers so that a failed allocation throws
    // before any thread starts; nobody dereferences the pointers, so where the
    // arrays live does not matter to the measurement.
    std::vector<std::vector<int>> arrays(static_cast<size_t>(threads));
    if constexpr (key_only) {
        for (size_t t = 0; t != arrays.size(); ++t) {
            arrays[t].resize(kPointees);
            for (size_t i = 0; i != kPointees; ++i) arrays[t][i] = static_cast<int>(i);
        } // loop over the threads
    } // if key-only
    // The prefill. fill_keys holds its key-only pointees; declared after the
    // queue, it is destroyed first, which is harmless: nothing dereferences
    // the keys, and the queue is drained below, before either goes.
    std::vector<int> fill_keys;
    if (!prefill<Q, K, V>(*q, static_cast<size_t>(fill*static_cast<double>(N)), static_cast<size_t>(threads) + 1,
                          fill_keys)) {
        throw std::logic_error("prefilling the queue failed");
    }
    // Per-thread counts within the window, one element per worker, each
    // written once by its worker before it exits.
    std::vector<OpCounts> counts(static_cast<size_t>(threads));
    g_flags.measure.store(false, std::memory_order_relaxed);
    g_flags.stop.store(false, std::memory_order_relaxed);
    // The threads that meet at the start latch: the workers and the main thread.
    const std::ptrdiff_t participants = static_cast<std::ptrdiff_t>(threads) + 1;
    std::latch start(participants);

    // A worker: wait for the start, run the warm-up phase, note the counts, run
    // the measured phase, store the counts of the measured phase.
    const auto worker = [&](size_t t) {
        Q& queue = *q;
        int* const v = arrays[t].data();
        const size_t key_stride = static_cast<size_t>(threads) + 1;
        const size_t key_offset = t + 1;
        OpCounts c;
        start.arrive_and_wait();
        // The phase function of the thread's starting role, run once per phase.
        const auto phase = (t & 1) ? run_phase<Q, K, V, true> : run_phase<Q, K, V, false>;
        phase(g_flags.measure, queue, v, mask, key_stride, key_offset, c);
        const OpCounts c0 = c;
        phase(g_flags.stop, queue, v, mask, key_stride, key_offset, c);
        counts[t] = c - c0;
    }; // worker

    // Steps 1 (starting the threads) to 6.
    const double measured_s = run_window(start, participants, warmup_s, window_s,
                                         [&](std::vector<std::jthread>& started) {
        for (int t = 0; t != threads; ++t) started.emplace_back(worker, static_cast<size_t>(t));
    });

    // Step 6: what was left in the queue when the stop flag was set (the
    // prefill plus all pushes minus all pops).
    size_t rem = 0;
    if constexpr (key_only) {
        while (q->pop()) ++rem;
    } else {
        [[maybe_unused]] V value {};
        while (q->pop(value) != K{}) ++rem;
    }
    return RunResult{measured_s, std::move(counts), rem, N};
} // run_variant()

// A variant as the command line sees it: its name and its run function.
struct Variant {
    const char* name;                   // the run name, shared with the Google Benchmark harness
    RunResult (*run)(int, double, double, size_t, double);      // the run_variant() instantiation
};

// The variants: the balanced rows that concurrent_queue_gmbm.C registers,
// under the same names, in the same order.
constexpr Variant kVariants[] = {
    {"BM_MP_MC_void_8_balanced", run_variant<int*, void, 8, 0>},
    {"BM_MP_MC_void_8_a16_balanced", run_variant<int*, void, 8, 16>},
    {"BM_MP_MC_void_8_a64_balanced", run_variant<int*, void, 8, 64>},
    {"BM_MP_MC_void_8_a128_balanced", run_variant<int*, void, 8, 128>},
    {"BM_MP_MC_kv_8_balanced", run_variant<uint64_t, uint64_t, 8, 0>},
    {"BM_MP_MC_kv_8_a16_balanced", run_variant<uint64_t, uint64_t, 8, 16>},
    {"BM_MP_MC_kv_8_a64_balanced", run_variant<uint64_t, uint64_t, 8, 64>},
    {"BM_MP_MC_kv_8_a128_balanced", run_variant<uint64_t, uint64_t, 8, 128>},
};

// The fairness figures of one role in one run; see HOW THE NUMBERS RELATE
// above.
struct Fairness {
    double cv;                          // standard deviation/mean of the per-thread counts
    double min_rel;                     // smallest count/mean
    double max_rel;                     // largest count/mean
};

// Compute the fairness figures of the per-thread counts `counts`, which must
// not be empty. A zero mean (no thread of the role succeeded even once) gives
// all zeros rather than a division by 0.
Fairness fairness(const std::vector<size_t>& counts) {
    const double n = static_cast<double>(counts.size());
    double sum = 0;
    for (size_t c : counts) sum += static_cast<double>(c);
    const double mean = sum/n;
    if (mean == 0) return Fairness{0, 0, 0};
    double sq = 0;                      // sum of squared deviations from the mean
    for (size_t c : counts) sq += (static_cast<double>(c) - mean)*(static_cast<double>(c) - mean);
    const auto [min_it, max_it] = std::minmax_element(counts.begin(), counts.end());
    return Fairness{std::sqrt(sq/n)/mean, static_cast<double>(*min_it)/mean, static_cast<double>(*max_it)/mean};
} // fairness()

// The median of `values`, which must not be empty; reorders them. For an even
// count, the mean of the two middle values.
double median(std::vector<double>& values) {
    const size_t mid = values.size()/2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double upper = values[mid];
    if (values.size() % 2 != 0) return upper;
    const double lower = *std::max_element(values.begin(), values.begin() + mid);
    return (lower + upper)/2;
} // median()

// The CSV columns after variant, threads and rep, in output order; the
// figures of one run are a Row, and the summary takes the median of each
// column over the repetitions of a cell.
constexpr const char* kColumns[] = {
    "window_s", "items", "items_per_s", "push_per_s", "pop_per_s", "push_fail_per_s", "pop_fail_per_s", "rem_pct",
    "producer_cv", "producer_min_rel", "producer_max_rel", "consumer_cv", "consumer_min_rel", "consumer_max_rel",
};
constexpr size_t kNumColumns = std::size(kColumns);
using Row = std::array<double, kNumColumns>;                    // one run's figures
using Cell = std::array<std::vector<double>, kNumColumns>;      // a cell's figures, one entry per repetition

// The figures of one run, in the order of kColumns.
Row make_row(const RunResult& r) {
    OpCounts total;
    std::vector<size_t> producers;      // successful operations per producer (odd thread)
    std::vector<size_t> consumers;      // successful operations per consumer (even thread)
    for (size_t t = 0; t != r.counts.size(); ++t) {
        const OpCounts& c = r.counts[t];
        total.p += c.p;
        total.c += c.c;
        total.pmiss += c.pmiss;
        total.cmiss += c.cmiss;
        if (t & 1) producers.push_back(c.p + c.c);
        else consumers.push_back(c.p + c.c);
    } // loop over the threads' counts
    const Fairness fp = fairness(producers);
    const Fairness fc = fairness(consumers);
    const double w = r.window_s;
    const double items = static_cast<double>(total.p + total.c);
    return Row{w, items, items/w, static_cast<double>(total.p)/w, static_cast<double>(total.c)/w,
               static_cast<double>(total.pmiss)/w, static_cast<double>(total.cmiss)/w,
               100.0*static_cast<double>(r.rem)/static_cast<double>(r.capacity),
               fp.cv, fp.min_rel, fp.max_rel, fc.cv, fc.min_rel, fc.max_rel};
} // make_row()

// The number of configured CPUs, or 0 after printing the reason to stderr if
// it is unavailable.
int configured_cpu_count() {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
#else
    // sysconf() returns -1 both on error (errno set) and for an indeterminate
    // value (errno untouched); errno is cleared first to tell the two apart.
    errno = 0;
    const long numcpu = sysconf(_SC_NPROCESSORS_CONF);
    if (numcpu < 1) {
        if (errno != 0) {
            std::fprintf(stderr, "sysconf(_SC_NPROCESSORS_CONF) failed: %s\n", std::strerror(errno));
        } else {
            std::fprintf(stderr, "sysconf(_SC_NPROCESSORS_CONF): the number of CPUs is indeterminate\n");
        }
        return 0;
    } // if the CPU count is unavailable
    return static_cast<int>(numcpu);
#endif
} // configured_cpu_count()

// The default thread counts: the powers of two from 2 up to `numcpu`, then
// `numcpu` itself if it is not a power of two (the thread counts of
// concurrent_queue_gmbm.C); just 2 if `numcpu` is below 2.
std::vector<int> thread_counts(int numcpu) {
    std::vector<int> counts;
    for (int t = 2; t <= numcpu; t *= 2) counts.push_back(t);
    if (counts.empty() || counts.back() != numcpu) counts.push_back(std::max(numcpu, 2));
    return counts;
} // thread_counts()

// The queue capacity in slots: 65536, or the value of the CQ_CAP environment
// variable (any strtoull base-0 number) rounded down to a power of 2, the
// same as concurrent_queue_gmbm.C. Stores it in `capacity` and returns true;
// returns false after a message if CQ_CAP is not a whole number in [8, 2^56]
// (2^56 slots of the largest slot, 128 bytes, is the most whose size in bytes
// fits a size_t).
bool queue_capacity(size_t& capacity) {
    const char* env = std::getenv("CQ_CAP");
    if (!env) {
        capacity = size_t{1} << 16;
        return true;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long cap = std::strtoull(env, &end, 0);
    if (end == env || *end != '\0' || errno == ERANGE || cap < 8 || cap > (1ULL << 56)) {
        std::fprintf(stderr, "CQ_CAP=%s: expected a whole number of slots in [8, 2^56]\n", env);
        return false;
    }
    if (!std::has_single_bit(cap)) {
        std::fprintf(stderr, "CQ_CAP=%s: not a power of 2, using %llu slots\n", env, std::bit_floor(cap));
    }
    capacity = static_cast<size_t>(std::bit_floor(cap));
    return true;
} // queue_capacity()

// If `arg` is `name` followed by '=', return the text after the '='; otherwise
// nullptr. `arg` must be null-terminated, and so is the returned text.
const char* option_value(const char* arg, std::string_view name) {
    const std::string_view a(arg);
    if (a.size() <= name.size() || !a.starts_with(name) || a[name.size()] != '=') return nullptr;
    return arg + name.size() + 1;
}

// The fraction of the capacity to fill before the threads start: 0, or the
// value of the CQ_FILL environment variable, a number in [0, 1] (e.g.
// CQ_FILL=0.5). Stores it in `fill` and returns true;
// returns false after a message if CQ_FILL is not such a number.
bool queue_fill(double& fill) {
    const char* env = std::getenv("CQ_FILL");
    if (!env) {
        fill = 0;
        return true;
    }
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(env, &end);
    if (end == env || *end != '\0' || errno == ERANGE || !(value >= 0 && value <= 1)) {
        std::fprintf(stderr, "CQ_FILL=%s: expected a fraction of the capacity in [0, 1]\n", env);
        return false;
    }
    fill = value;
    return true;
} // queue_fill()

// Parse all of `text` as a finite double into `out`; false (and `out`
// unchanged) if it is not one. strtod() rather than from_chars(), which not
// every standard library in use implements for floating point.
bool parse_double(const char* text, double& out) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(value)) return false;
    out = value;
    return true;
} // parse_double()

// Parse all of `text` as a decimal int into `out`; false (and `out`
// unchanged) if it is not one or does not fit.
bool parse_int(std::string_view text, int& out) {
    int value = 0;
    const std::from_chars_result r = std::from_chars(text.data(), text.data() + text.size(), value);
    if (r.ec != std::errc() || r.ptr != text.data() + text.size()) return false;
    out = value;
    return true;
} // parse_int()

// Parse a comma-separated list of ints, each at least 2, into `out`, replacing
// its contents; false (and `out` unspecified) if any element is not one, or the
// list is empty.
bool parse_thread_list(std::string_view text, std::vector<int>& out) {
    out.clear();
    while (true) {
        const size_t comma = text.find(',');
        int t = 0;
        if (!parse_int(text.substr(0, comma), t) || t < 2) return false;
        out.push_back(t);
        if (comma == std::string_view::npos) return true;
        text.remove_prefix(comma + 1);
    } // loop over the list elements
} // parse_thread_list()

// Print the usage message to `f`.
void usage(std::FILE* f, const char* argv0) {
    std::fprintf(f,
                 "usage: %s [--window=SECONDS] [--warmup=SECONDS] [--reps=N] [--threads=T1,T2,...]\n"
                 "       %*s [--filter=REGEX]\n"
                 "  --window   measured seconds per run (default %g)\n"
                 "  --warmup   uncounted seconds before each window (default %g)\n"
                 "  --reps     repetitions of the whole sweep (default 1)\n"
                 "  --threads  thread counts, each at least 2 (default: powers of two from 2 up to the CPU\n"
                 "             count, and the CPU count)\n"
                 "  --filter   run only the variants whose name contains a match (ECMAScript regex)\n"
                 "The CQ_CAP environment variable sets the queue capacity in slots (default 65536), and\n"
                 "CQ_FILL the fraction of it filled before the threads start (default 0).\n",
                 argv0, static_cast<int>(std::string_view(argv0).size()), "", kWindowSeconds, kWarmupSeconds);
} // usage()

} // namespace

// Parse the command line, run the sweep (see ORDER above), write the CSV to
// stdout and the summary to stderr. Returns 0 on success, 1 if a run or the
// output failed, 2 for a bad command line, CQ_CAP or CQ_FILL.
int main(int argc, char** argv) {
    double window_s = kWindowSeconds;   // --window
    double warmup_s = kWarmupSeconds;   // --warmup
    int reps = 1;                       // --reps
    std::vector<int> threads_list;      // --threads; empty until given or defaulted
    const char* filter_text = "";       // --filter; the empty regex matches every name

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* value = nullptr;
        bool ok = true;
        if (std::string_view(arg) == "--help" || std::string_view(arg) == "-h") {
            usage(stdout, argv[0]);
            return 0;
        } else if ((value = option_value(arg, "--window"))) {
            ok = parse_double(value, window_s) && window_s > 0;
        } else if ((value = option_value(arg, "--warmup"))) {
            ok = parse_double(value, warmup_s) && warmup_s >= 0;
        } else if ((value = option_value(arg, "--reps"))) {
            ok = parse_int(value, reps) && reps >= 1;
        } else if ((value = option_value(arg, "--threads"))) {
            ok = parse_thread_list(value, threads_list);
        } else if ((value = option_value(arg, "--filter"))) {
            filter_text = value;
        } else {
            ok = false;
        }
        if (!ok) {
            std::fprintf(stderr, "%s: bad argument '%s'\n", argv[0], arg);
            usage(stderr, argv[0]);
            return 2;
        }
    } // loop over the arguments

    size_t capacity = 0;
    if (!queue_capacity(capacity)) return 2;
    double fill = 0;
    if (!queue_fill(fill)) return 2;
    const int numcpu = configured_cpu_count();
    if (threads_list.empty()) threads_list = thread_counts(numcpu);

    // The selected variants, in the order of kVariants.
    std::vector<const Variant*> variants;
    try {
        const std::regex filter(filter_text, std::regex::ECMAScript);
        for (const Variant& v : kVariants) {
            if (std::regex_search(v.name, filter)) variants.push_back(&v);
        }
    } catch (const std::regex_error& e) {
        std::fprintf(stderr, "%s: bad --filter regex '%s': %s\n", argv[0], filter_text, e.what());
        return 2;
    }
    if (variants.empty()) {
        std::fprintf(stderr, "%s: --filter '%s' matches no variant\n", argv[0], filter_text);
        return 2;
    }

    std::fprintf(stderr, "# cpus=%d capacity=%zu fill=%g window=%gs warmup=%gs reps=%d variants=%zu threads=", numcpu,
                 capacity, fill, window_s, warmup_s, reps, variants.size());
    for (size_t i = 0; i != threads_list.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", threads_list[i]);
    std::fprintf(stderr, "\n");

    // cells[vi*threads_list.size() + ti][k] collects column k of the runs of
    // variant vi at thread count threads_list[ti].
    std::vector<Cell> cells(variants.size()*threads_list.size());
    std::printf("variant,threads,rep");
    for (const char* column : kColumns) std::printf(",%s", column);
    std::printf("\n");
    std::fflush(stdout);
    try {
        for (int rep = 0; rep != reps; ++rep) {
            for (size_t ti = 0; ti != threads_list.size(); ++ti) {
                const int threads = threads_list[ti];
                for (size_t k = 0; k != variants.size(); ++k) {
                    // Reversed order when rep + ti is odd; see ORDER above.
                    const size_t vi = ((static_cast<size_t>(rep) + ti) % 2 == 0) ? k : variants.size() - 1 - k;
                    const RunResult r = variants[vi]->run(threads, warmup_s, window_s, capacity, fill);
                    const Row row = make_row(r);
                    // items is a whole count; the rates and fractions get fixed
                    // precision.
                    std::printf("%s,%d,%d,%.6f,%.0f", variants[vi]->name, threads, rep, row[0], row[1]);
                    for (size_t col = 2; col != kNumColumns; ++col) std::printf(",%.6g", row[col]);
                    std::printf("\n");
                    std::fflush(stdout);
                    Cell& cell = cells[vi*threads_list.size() + ti];
                    for (size_t col = 0; col != kNumColumns; ++col) cell[col].push_back(row[col]);
                } // loop over variants
            } // loop over thread counts
        } // loop over repetitions
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: run failed: %s\n", argv[0], e.what());
        return 1;
    }

    // The summary: medians over the repetitions, cell by cell, thread count
    // outside. Rates in millions per second.
    std::fprintf(stderr, "# median over %d repetition%s; rates in M/s\n", reps, reps == 1 ? "" : "s");
    std::fprintf(stderr, "%-30s %7s %8s %9s %8s %8s %8s %8s %6s %7s %7s %7s %7s\n", "variant", "threads", "window_s",
                 "items/s", "push/s", "pop/s", "pushF/s", "popF/s", "rem%", "prod_cv", "prod_mn", "cons_cv", "cons_mn");
    for (size_t ti = 0; ti != threads_list.size(); ++ti) {
        for (size_t vi = 0; vi != variants.size(); ++vi) {
            Cell& cell = cells[vi*threads_list.size() + ti];
            Row m {};
            for (size_t col = 0; col != kNumColumns; ++col) m[col] = median(cell[col]);
            std::fprintf(stderr, "%-30s %7d %8.3f %9.2f %8.2f %8.2f %8.2f %8.2f %6.1f %7.4f %7.4f %7.4f %7.4f\n",
                         variants[vi]->name, threads_list[ti], m[0], m[2]/1e6, m[3]/1e6, m[4]/1e6, m[5]/1e6,
                         m[6]/1e6, m[7], m[8], m[9], m[11], m[12]);
        } // loop over variants
    } // loop over thread counts

    // A failed write to stdout (a full disk, a closed pipe) would otherwise go
    // unnoticed and leave a truncated CSV behind a zero exit status.
    if (std::ferror(stdout) || std::fflush(stdout) != 0) {
        std::fprintf(stderr, "%s: writing the results failed\n", argv[0]);
        return 1;
    }
    return 0;
} // main()
