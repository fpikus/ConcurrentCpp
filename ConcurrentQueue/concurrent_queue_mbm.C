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
// Custom throughput benchmark for RingAtomicMapQueueMPMC (mbm = the base
// hand-rolled MicroBenchMark; gmbm is the Google Benchmark twin that reports
// the same numbers).
// Mirrors queue_gmbm.C but replaces Google Benchmark with a hand-rolled
// harness so we have tight control over start/stop synchronization and can
// later swap the throughput metric for per-op latency.
//
// Methodology differences from gmbm worth knowing: each thread count runs for
// a FIXED wall-clock duration (default 5 s) rather than Google Benchmark's
// auto-tuned iteration count, and there is no warm-up phase -- steady-state
// throughput dominates a multi-second run. Roles are the same static parity
// split as gmbm (odd index produces, even consumes). No benchmark::
// DoNotOptimize is needed: every push/pop result is consumed by an `if` and
// its count flushed to stats[t], so the work is observably live.

#include "concurrent_queue.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <memory>
#include <new>
#include <string>
#include <thread>
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

size_t hw_thread_count() {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<size_t>(si.dwNumberOfProcessors);
#else
    return static_cast<size_t>(sysconf(_SC_NPROCESSORS_CONF));
#endif
}

// Padded so adjacent threads don't false-share their counters.
struct alignas(64) ThreadStats {
    size_t p = 0;
    size_t c = 0;
    size_t pmiss = 0;
    size_t cmiss = 0;
    char   pad[64 - 4 * sizeof(size_t)];
};

struct RunResult {
    size_t n_threads;
    size_t capacity;
    double elapsed_s;
    size_t total_p = 0, total_c = 0, total_pmiss = 0, total_cmiss = 0;
};

// Single run: spin up n_threads workers (odd indices produce, even consume -
// same split as gmbm's `thread_index & 1`), sync at a start latch, run until
// the stop flag is set, join.
template <typename Q>
RunResult run_once(size_t capacity_hint, size_t n_threads,
                   std::chrono::nanoseconds duration) {
    constexpr size_t elem_size  = Q::element_size();
    constexpr size_t elem_align = Q::element_align();
    const size_t NB = capacity_hint * elem_size;

    void* memory = ::operator new(NB, std::align_val_t{elem_align});
    // Heap-allocated so the queue can be destroyed before its buffer is
    // freed below: a stack instance dies at scope exit, after the operator
    // delete, and its draining destructor would walk freed memory.
    auto q_owner = std::make_unique<Q>(memory, NB);
    Q& q = *q_owner;
    const size_t N    = q.capacity();
    const size_t mask = N - 1;

    std::latch start_latch(n_threads + 1);   // +1 for main
    std::atomic<bool> stop{false};
    std::vector<ThreadStats> stats(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            const bool producer = (t & 1) != 0;

            std::vector<int> v(N);
            for (size_t i = 0; i < N; ++i) v[i] = static_cast<int>(i);

            // Keep hot-loop counters in registers; flush to stats[t] at end.
            size_t p = 0, c = 0, pmiss = 0, cmiss = 0;

            start_latch.arrive_and_wait();

            if (producer) {
                while (!stop.load(std::memory_order_relaxed)) {
                    if (q.push(&v[p & mask])) ++p;
                    else ++pmiss;
                }
            } else {
                while (!stop.load(std::memory_order_relaxed)) {
                    int* ptr = q.pop();
                    if (ptr) ++c;
                    else ++cmiss;
                }
            }

            stats[t].p     = p;
            stats[t].c     = c;
            stats[t].pmiss = pmiss;
            stats[t].cmiss = cmiss;
        });
    }

    start_latch.arrive_and_wait();
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_relaxed);
    const auto t1 = std::chrono::steady_clock::now();

    for (auto& th : threads) th.join();

    q_owner.reset();    // Queue dtor drains leftovers; must precede the buffer free
    ::operator delete(memory, std::align_val_t{elem_align});

    RunResult r{};
    r.n_threads = n_threads;
    r.capacity  = N;
    r.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    for (auto& s : stats) {
        r.total_p     += s.p;
        r.total_c     += s.c;
        r.total_pmiss += s.pmiss;
        r.total_cmiss += s.cmiss;
    }
    return r;
}

std::string humanize(double x) {
    char buf[32];
    if      (x >= 1e9) std::snprintf(buf, sizeof(buf), "%7.3fG", x / 1e9);
    else if (x >= 1e6) std::snprintf(buf, sizeof(buf), "%7.3fM", x / 1e6);
    else if (x >= 1e3) std::snprintf(buf, sizeof(buf), "%7.3fk", x / 1e3);
    else               std::snprintf(buf, sizeof(buf), "%8.1f",  x);
    return std::string(buf);
}

void print_header() {
    std::printf("%-32s %4s %12s %8s %9s %9s %9s %9s %9s\n",
                "Benchmark", "thr", "items", "ns/op",
                "items/s", "push/s", "pop/s", "pushF/s", "popF/s");
}

void print_row(const char* name, const RunResult& r) {
    const size_t items = r.total_p + r.total_c;
    // ns/op is aggregate, not per-thread latency: total core-time
    // (elapsed * n_threads) divided by total successful ops. It answers "how
    // much CPU does one enqueue-or-dequeue cost across the machine", so it
    // rises with contention even though each thread's own loop stays busy.
    const double ns_per_op =
        r.elapsed_s * 1e9 * static_cast<double>(r.n_threads) /
        static_cast<double>(items ? items : 1);
    std::printf("%-32s %4zu %12zu %8.2f %9s %9s %9s %9s %9s\n",
                name, r.n_threads, items, ns_per_op,
                humanize(static_cast<double>(items)       / r.elapsed_s).c_str(),
                humanize(static_cast<double>(r.total_p)   / r.elapsed_s).c_str(),
                humanize(static_cast<double>(r.total_c)   / r.elapsed_s).c_str(),
                humanize(static_cast<double>(r.total_pmiss)/ r.elapsed_s).c_str(),
                humanize(static_cast<double>(r.total_cmiss)/ r.elapsed_s).c_str());
    std::fflush(stdout);
}

template <size_t NTRY, size_t ALIGN>
void sweep(const char* name, size_t capacity_hint,
           const std::vector<size_t>& thread_counts,
           std::chrono::nanoseconds duration) {
    using Q = RingAtomicMapQueueMPMC<int*, void, NTRY, ALIGN>;
    for (size_t nt : thread_counts) {
        auto r = run_once<Q>(capacity_hint, nt, duration);
        print_row(name, r);
    }
}

} // namespace

int main(int argc, char** argv) {
    size_t capacity_hint = 1UL << 16;
    double duration_s    = 5.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--duration=", 0) == 0) duration_s    = std::atof(a.c_str() + 11);
        else if (a.rfind("--capacity=", 0) == 0) capacity_hint = std::strtoull(a.c_str() + 11, nullptr, 0);
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--duration=SECONDS] [--capacity=N]\n", argv[0]);
            return 0;
        }
    }

    const size_t hw = hw_thread_count();
    std::vector<size_t> thread_counts;
    for (size_t t = 2; t <= hw; t *= 2) thread_counts.push_back(t);
    if (thread_counts.empty() || thread_counts.back() != hw)
        thread_counts.push_back(hw);

    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(duration_s));

    std::printf("hw_threads=%zu  capacity_hint=%zu  duration=%.3fs\n\n",
                hw, capacity_hint, duration_s);
    print_header();

    sweep<8, 0>  ("BM_MP_MC_void_8",      capacity_hint, thread_counts, duration);
    sweep<8, 16> ("BM_MP_MC_void_8_a16",  capacity_hint, thread_counts, duration);
    sweep<8, 64> ("BM_MP_MC_void_8_a64",  capacity_hint, thread_counts, duration);
    sweep<8, 128>("BM_MP_MC_void_8_a128", capacity_hint, thread_counts, duration);

    return 0;
}
