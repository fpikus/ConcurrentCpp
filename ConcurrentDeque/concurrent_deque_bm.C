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
#include "concurrent_deque.h"
#include "spinlock.h"
#include <deque>
#include <thread>
#include <vector>
#include <atomic>
#include <barrier>
#include <memory>
#include <cstdint>
#include <unistd.h>

// Determine the maximum number of hardware threads available on this system.
// This allows Google Benchmark to automatically scale the tests up to the
// machine's physical limits (e.g., all 256 threads on a Granite Rapids server)
// without hardcoding artificial ceilings.
static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// ============================================================================
// SpinlockDeque Baseline
// ============================================================================
// A simple thread-safe wrapper around std::deque using a SpinLock.
// This serves as the baseline to demonstrate the catastrophic performance
// collapse (thundering herd) that occurs when using locks under high contention,
// compared to the lock-free ConcurrentAppendDeque.
template <typename T>
class SpinlockDeque {
public:
    void push_back(const T& val) {
        std::lock_guard<SpinLock> lock(spinlock_);
        deque_.push_back(val);
    }
    
    void resize(size_t new_size) {
        std::lock_guard<SpinLock> lock(spinlock_);
        if (new_size > deque_.size()) {
            deque_.resize(new_size);
        }
    }

    size_t size() const {
        std::lock_guard<SpinLock> lock(spinlock_);
        return deque_.size();
    }
    
    T& operator[](size_t index) {
        std::lock_guard<SpinLock> lock(spinlock_);
        return deque_[index];
    }
    
    const T& operator[](size_t index) const {
        std::lock_guard<SpinLock> lock(spinlock_);
        return deque_[index];
    }
    
private:
    mutable SpinLock spinlock_;
    std::deque<T> deque_;
};

// ============================================================================
// Benchmark Concurrency Design Principles
// ============================================================================
// These benchmarks do NOT use Google Benchmark's native MT (state.threads())
// nor do they create std::threads inside the timing loop. Why?
// 1. Thread Creation Overhead: Spawning threads takes significant OS time. If 
//    done inside the loop, the benchmark measures OS latency, not container speed.
// 2. Staggered Starts: Without precise synchronization, Thread 1 might finish its
//    work before Thread 8 even wakes up. This destroys concurrent contention.
// 
// Solution: We spawn a persistent pool of `std::jthread`s exactly once.
// We use `std::barrier` to perfectly align all threads at a starting line.
// The main thread pauses the clock, resets the state, and then releases the 
// barrier, ensuring every thread hits the container at the exact same microsecond.
//
// Clean Shutdown: While `std::jthread` is often used simply as an RAII wrapper 
// to avoid manual `.join()` calls, we actively use its cooperative cancellation 
// mechanism (`std::stop_token`). The main thread calls `.request_stop()` on all 
// threads and then unblocks the barrier one last time, allowing the workers to 
// check `stoken.stop_requested()` and cleanly exit their infinite loops.

// ============================================================================
// Scenario 1: Element Access with No Growth
// ============================================================================
// Goal: Measure pure, wait-free scaling when the deque is already sized.
// Expectation: Perfect linear scaling up to the memory bandwidth or ALU limit,
// as threads write to strictly disjoint ranges without any atomic synchronization.
template <typename DequeType>
static void BM_AccessNoGrowth(benchmark::State& state) {
    size_t elements_per_thread = state.range(0);
    int num_threads = state.range(1);
    size_t total_elements = num_threads * elements_per_thread;
    
    // Pre-allocate the deque so no resizes occur during the benchmark.
    DequeType dq;
    dq.resize(total_elements);
    
    // num_threads workers + 1 main coordinator thread.
    std::barrier sync_start(num_threads + 1);
    std::barrier sync_end(num_threads + 1);
    
    // Using std::jthread automatically requests a stop and joins on destruction.
    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t](std::stop_token stoken) {
            size_t start_idx = t * elements_per_thread;
            size_t end_idx = start_idx + elements_per_thread;
            
            while (true) {
                // Wait at the starting line for the main thread to resume timing.
                sync_start.arrive_and_wait();
                if (stoken.stop_requested()) break;
                
                // Payload: writes to a disjoint, per-thread index range, so no
                // two threads touch the same element. This is what exposes the
                // contrast between the two DequeType instantiations: for
                // ConcurrentAppendDeque each dq[i] is a wait-free acquire load of
                // the directory plus a plain store (no cross-thread coordination
                // on disjoint indices), whereas SpinlockDeque takes and releases
                // the global lock on every single element access -- the source of
                // its contention collapse.
                for (size_t i = start_idx; i < end_idx; ++i) {
                    dq[i] = static_cast<int>(i);
                }
                
                // Wait for all threads to finish before the next iteration.
                sync_end.arrive_and_wait();
            }
        });
    }
    
    for (auto _ : state) {
        sync_start.arrive_and_wait(); // Unleash threads
        sync_end.arrive_and_wait();   // Wait for completion
    }
    
    // Cleanly shut down the thread pool
    for (auto& thread : threads) {
        thread.request_stop();
    }
    sync_start.arrive_and_wait(); // Unblock them one last time so they see the stop request
    threads.clear();
    
    // Custom counter for easy-to-read throughput (e.g., 20 G/s)
    state.counters["Elements/s"] = benchmark::Counter(
        state.iterations() * total_elements,
        benchmark::Counter::kIsRate);
}

// Registration axes (consumed as state.range(0) and state.range(1) inside the
// benchmark): arg 0 is elements-per-thread, swept geometrically 1e3..1e5 (x10);
// arg 1 is the thread count, swept 1..num_cpu (x2). ArgsProduct runs the full
// cross product.
//
// UseRealTime() is essential for these multi-threaded cases: by default Google
// Benchmark reports CPU time, which SUMS across all worker threads, so an 8-way
// run would look ~8x "slower" purely from counting eight cores' time. Wall-clock
// (real) time is what actually measures concurrent scaling here.
BENCHMARK_TEMPLATE(BM_AccessNoGrowth, SpinlockDeque<int>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

BENCHMARK_TEMPLATE(BM_AccessNoGrowth, ConcurrentAppendDeque<int, 1024>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

// ============================================================================
// Scenario 2: Element Access with Size Check
// ============================================================================
// Goal: Measure the overhead of the `memory_order_acquire` penalty.
// In reality, a reader thread must check `size()` before blindly indexing. 
// This benchmark forces every thread to bounce the atomic `size_` cache line 
// once per iteration before writing to its disjoint range.
template <typename DequeType>
static void BM_AccessWithSizeCheck(benchmark::State& state) {
    size_t elements_per_thread = state.range(0);
    int num_threads = state.range(1);
    size_t total_elements = num_threads * elements_per_thread;
    
    DequeType dq;
    dq.resize(total_elements);
    
    std::barrier sync_start(num_threads + 1);
    std::barrier sync_end(num_threads + 1);
    
    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t](std::stop_token stoken) {
            size_t start_idx = t * elements_per_thread;
            size_t end_idx = start_idx + elements_per_thread;
            
            while (true) {
                sync_start.arrive_and_wait();
                if (stoken.stop_requested()) break;
                
                // The single `acquire` load on the atomic size.
                if (dq.size() >= end_idx) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        dq[i] = static_cast<int>(i);
                    }
                }
                
                sync_end.arrive_and_wait();
            }
        });
    }
    
    for (auto _ : state) {
        sync_start.arrive_and_wait(); // Unleash threads
        sync_end.arrive_and_wait();   // Wait for completion
    }
    
    for (auto& thread : threads) {
        thread.request_stop();
    }
    sync_start.arrive_and_wait();
    threads.clear();
    
    state.counters["Elements/s"] = benchmark::Counter(
        state.iterations() * total_elements,
        benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_AccessWithSizeCheck, SpinlockDeque<int>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

BENCHMARK_TEMPLATE(BM_AccessWithSizeCheck, ConcurrentAppendDeque<int, 1024>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

// ============================================================================
// Scenario 3: Resize and Write
// ============================================================================
// Goal: Stress test the Double-Checked Locking Pattern (DCLP) and lock-free
// contention during container expansion.
// The deque starts empty. All threads simultaneously try to resize it to their
// required capacity, guaranteeing a massive "Thundering Herd" collision.
template <typename DequeType>
static void BM_ResizeAndWrite(benchmark::State& state) {
    size_t elements_per_thread = state.range(0);
    int num_threads = state.range(1);
    
    // Benchmark Balance Decision:
    // If we only write each element once, the extremely slow heap allocations
    // and spinlock contention inside `resize()` will completely dominate the time,
    // masking the wait-free read/write throughput we actually care about.
    // Real-world usage involves resizing rarely, and accessing data heavily.
    // Therefore, we amplify the access workload by 100x to simulate this reality.
    const int work_iterations = 100;
    size_t total_elements = num_threads * elements_per_thread * work_iterations;
    
    std::unique_ptr<DequeType> dq;
    
    std::barrier sync_start(num_threads + 1);
    std::barrier sync_end(num_threads + 1);
    
    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t](std::stop_token stoken) {
            size_t start_idx = t * elements_per_thread;
            size_t end_idx = start_idx + elements_per_thread;
            
            while (true) {
                sync_start.arrive_and_wait();
                if (stoken.stop_requested()) break;
                
                // The Thundering Herd hits the DCLP here. Only one thread
                // will successfully grab the lock and allocate the blocks.
                dq->resize(end_idx);
                
                // The amplified read/write workload. 
                // We add `iter` to prevent compiler loop-unrolling optimizations.
                for (int iter = 0; iter < work_iterations; ++iter) {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        (*dq)[i] = static_cast<int>(i + iter);
                    }
                }
                
                sync_end.arrive_and_wait();
            }
        });
    }
    
    for (auto _ : state) {
        // Pause the timer so we do not benchmark the teardown of the old
        // deque or the instantiation of the new one. We strictly only want
        // to measure the concurrent scaling.
        state.PauseTiming();
        dq = std::make_unique<DequeType>();
        state.ResumeTiming();
        
        sync_start.arrive_and_wait(); // Unleash threads
        sync_end.arrive_and_wait();   // Wait for completion
    }
    
    for (auto& thread : threads) {
        thread.request_stop();
    }
    sync_start.arrive_and_wait();
    threads.clear();
    
    state.counters["Elements/s"] = benchmark::Counter(
        state.iterations() * total_elements,
        benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_ResizeAndWrite, SpinlockDeque<int>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

BENCHMARK_TEMPLATE(BM_ResizeAndWrite, ConcurrentAppendDeque<int, 1024>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

// ============================================================================
// Scenario 4: Concurrent Writer vs. size()-Polling Readers (false-sharing probe)
// ============================================================================
// Goal: isolate false sharing between the reader-hammered `size_` and the
// write-path members (`spinlock_`, `directory_capacity_`). Scenarios 1-3 CANNOT
// see this: they never run a writer concurrently with polling readers inside the
// timed region (1-2 have no writer; 3 writes only during a brief resize, then
// does pure disjoint element writes with no size() access). False sharing needs
// a writer actively mutating the line WHILE readers read it.
//
// This scenario builds exactly that -- the container's single-producer /
// many-consumer design point:
//   - The benchmark thread itself is the writer: after the start barrier it does
//     a fixed number of push_back()s (arg 0). Each push takes the spinlock (TTAS
//     exchange + release-store unlock) and release-stores `size_`. Making the
//     benchmark thread the writer (rather than a spawned thread) means the write
//     burst is active the instant readers are released -- no separate-thread
//     wakeup latency during which readers would spin on an unchanging, fully
//     cached `size_` and mask the effect.
//   - `num_readers` reader threads (arg 1) spin on size() -- the wait-free
//     acquire load -- counting completed loads until the writer signals done.
//
// A/B METHODOLOGY (this is the point of the scenario): build the SAME source
// twice and compare -- once as-is (padded) and once with -DCONCURRENT_DEQUE_UNPADDED
// (packed layout, see concurrent_deque.h). The harness overhead is common-mode,
// so the difference between the two builds is the false-sharing cost. Use
// --benchmark_repetitions and compare MEDIANS -- unpinned placement makes single
// runs bimodal on multi-domain machines.
//
// What to look at (from runs on four machines; details in concurrent_deque.h):
//   - WriterPush/s carries the wall-time signal, not ReaderLoad/s: reader misses
//     are rate-limited by how often the writer invalidates the line, and polling
//     outruns pushing by orders of magnitude, so reader wall time barely moves.
//     Aggregate ReaderLoad/s instead exposes TRUE sharing: it plateaus once the
//     hot line's coherence traffic saturates (adding readers just splits it).
//   - The packed build's penalty grows with coherence-domain fragmentation:
//     zero on a single shared L3, up to 1.4-1.7x writer throughput (and far
//     noisier run-to-run) on a 16-CCX x86 server at 8-64 readers. At few
//     readers packed WINS (1.4-1.6x at 1-2 readers pinned within one NUMA
//     node, crossover near 4): probes are rare, so the lock-acquire RFO
//     carries `size_` along free (same-line-vs-separate-line rule, see
//     concurrent_deque.h). Sweeping reader count pinned vs unpinned moves the
//     crossover -- the pricier the probe, the earlier separation pays.
//   - When readers oversubscribe the pinned cores, compare writer CPU time
//     (not real time) per iteration: it is immune to the writer being
//     descheduled and to background load (measured ~480 vs ~680 ns of writer
//     CPU per push, padded vs packed, on the 16-CCX machine).
//   - The L1-miss gap (~3-4x, perf stat -e L1-dcache-load-misses) shows up on
//     every machine even when wall time does not.
//   - Rows where readers >= cores are an oversubscription artifact: the writer
//     starves, the line goes quiet, and ReaderLoad/s explodes to cache speed --
//     higher numbers there mean LESS concurrency, not faster reads.
//
// Axes differ from the other scenarios: arg 0 is the writer's push_back count
// (the measurement-window size), arg 1 is the number of polling READER threads.
// Only ConcurrentAppendDeque is registered -- SpinlockDeque::size() takes the
// lock, so it has no wait-free reader / false-sharing story (it would just
// measure lock contention).
template <typename DequeType>
static void BM_WriterVsPollingReaders(benchmark::State& state) {
    size_t writer_pushes = state.range(0);
    int num_readers = state.range(1);

    // Recreated each timed iteration (see loop) so the writer always appends from
    // empty and memory stays bounded; readers observe the fresh pointer through
    // the sync_start barrier's release/acquire.
    std::unique_ptr<DequeType> dq;

    // Set by the writer when its push burst is done; polled by readers as their
    // stop condition. Reset before each iteration's start barrier.
    std::atomic<bool> writer_done{false};

    // num_readers readers + the benchmark thread itself (acting as the writer).
    std::barrier sync_start(num_readers + 1);
    std::barrier sync_end(num_readers + 1);

    // Per-reader lifetime tally of completed size() loads, summed after join to
    // form the aggregate reader-throughput counter. Each reader writes its slot
    // once at thread exit; main reads them after readers.clear() joins.
    std::vector<uint64_t> reader_loads(num_readers, 0);

    std::vector<std::jthread> readers;
    readers.reserve(num_readers);
    for (int r = 0; r < num_readers; ++r) {
        readers.emplace_back([&, r](std::stop_token stoken) {
            uint64_t local = 0;
            volatile size_t sink = 0; // volatile: keep the acquire loads from being elided
            while (true) {
                sync_start.arrive_and_wait();
                if (stoken.stop_requested()) break;

                // Cache the raw deque pointer once so the hot loop is a bare
                // atomic acquire load of size_ (no unique_ptr indirection per
                // poll). Burst 8 loads between stop-flag checks so the size_ load
                // -- the false-sharing-sensitive op -- dominates the loop rather
                // than loop/branch overhead; a diluted loop hides the very effect
                // this scenario exists to measure.
                DequeType* d = dq.get();
                while (!writer_done.load(std::memory_order_relaxed)) {
                    sink += d->size(); sink += d->size();
                    sink += d->size(); sink += d->size();
                    sink += d->size(); sink += d->size();
                    sink += d->size(); sink += d->size();
                    local += 8;
                }

                sync_end.arrive_and_wait();
            } // per-iteration reader loop
            (void)sink;
            reader_loads[r] = local;
        });
    } // spawn readers

    for (auto _ : state) {
        // Rebuild the deque and clear the done flag outside the timed region so
        // we measure only the concurrent write-vs-poll window, not teardown.
        state.PauseTiming();
        dq = std::make_unique<DequeType>();
        writer_done.store(false, std::memory_order_relaxed);
        state.ResumeTiming();

        sync_start.arrive_and_wait(); // release readers; writer (this thread) starts now
        for (size_t i = 0; i < writer_pushes; ++i) {
            dq->push_back(static_cast<int>(i));
        }
        writer_done.store(true, std::memory_order_relaxed); // let readers exit their spin
        sync_end.arrive_and_wait();   // wait for readers to observe done
    }

    // Cooperative shutdown: request stop on every reader, then release the start
    // barrier once more so each observes it and exits (readers break before
    // sync_end, so main does not touch sync_end here).
    for (auto& reader : readers) {
        reader.request_stop();
    }
    sync_start.arrive_and_wait();
    readers.clear(); // joins readers, publishing their reader_loads slots

    uint64_t total_reader_loads = 0;
    for (uint64_t c : reader_loads) total_reader_loads += c;

    // Writer throughput: total pushes across all timed iterations, as a rate.
    state.counters["WriterPush/s"] = benchmark::Counter(
        static_cast<double>(state.iterations() * writer_pushes),
        benchmark::Counter::kIsRate);
    // Reader throughput: the false-sharing-sensitive metric to compare across the
    // padded and -DCONCURRENT_DEQUE_UNPADDED builds.
    state.counters["ReaderLoad/s"] = benchmark::Counter(
        static_cast<double>(total_reader_loads),
        benchmark::Counter::kIsRate);
} // BM_WriterVsPollingReaders()

BENCHMARK_TEMPLATE(BM_WriterVsPollingReaders, ConcurrentAppendDeque<int, 1024>)
    ->ArgsProduct({
        benchmark::CreateRange(1000, 100000, /*multi=*/10),
        benchmark::CreateRange(1, num_cpu, /*multi=*/2)
    })->UseRealTime();

BENCHMARK_MAIN();
