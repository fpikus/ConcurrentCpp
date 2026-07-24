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
// Shared helpers for queue latency benchmarks (queue_lmbm.C in each queue dir).
// Provides: rdtsc timestamps, cycle calibration, per-thread stats with a
// log-linear latency histogram, and percentile extraction.
//
// Compile-time configuration: change kLatPercentiles below and rebuild.

#ifndef LATBENCH_COMMON_H
#define LATBENCH_COMMON_H

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <limits>
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
  #include <intrin.h>
  #pragma comment(lib, "winmm.lib")
  namespace latbench {
    struct TimerResolutionGuard {
        TimerResolutionGuard()  { ::timeBeginPeriod(1); }
        ~TimerResolutionGuard() { ::timeEndPeriod(1);   }
    };
    inline TimerResolutionGuard g_timer_resolution_guard;
  }
#else
  #include <unistd.h>
  #if defined(__x86_64__) || defined(__i386__)
    #include <x86intrin.h>
  #endif
  #include <sched.h>
  #include <pthread.h>
#endif

namespace latbench {

// --- Compile-time percentile list (edit and rebuild to change) -------------
inline constexpr std::array<double, 5> kLatPercentiles{
    50.0, 90.0, 95.0, 99.0, 99.9
};

using Tick = uint64_t;

// Portable spin-wait hint: PAUSE on x86, YIELD on ARM.
static inline void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#endif
}

// Producer-side stamp: lfence bracket so the read reflects strictly prior ops.
// ARM: isb serialises the counter read against preceding stores.
static inline Tick rdtsc_start() {
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    __asm__ volatile("isb" ::: "memory");
    return t;
#else
    _mm_lfence();
    Tick t = __rdtsc();
    _mm_lfence();
    return t;
#endif
}

// Consumer-side stamp: rdtscp (waits for prior retire) + trailing lfence.
// ARM: isb serialises the counter read against preceding loads.
static inline Tick rdtsc_end() {
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    __asm__ volatile("isb" ::: "memory");
    return t;
#else
    unsigned aux;
    Tick t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#endif
}

// Bind the calling thread to `cpu`. Negative cpu leaves affinity unchanged.
// Only the bottom 64 logical CPUs are addressable on Windows without the
// extended GROUP_AFFINITY API -- sufficient for the 32-thread 9950X target.
inline void bind_current_thread_to_cpu(int cpu) {
    if (cpu < 0) return;
#ifndef _WIN32
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
#else
    ::SetThreadAffinityMask(::GetCurrentThread(),
                            static_cast<DWORD_PTR>(1) << cpu);
#endif
}

inline size_t hw_thread_count() {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<size_t>(si.dwNumberOfProcessors);
#else
    return static_cast<size_t>(sysconf(_SC_NPROCESSORS_CONF));
#endif
}

// Wall-clock vs TSC calibration. Assumes invariant TSC (Ryzen 9950X has it);
// the ratio is used to convert recorded cycle counts into nanoseconds for
// reporting. 100 ms is enough to swamp sleep_for scheduling jitter without
// adding perceptible startup delay.
inline double calibrate_cycles_per_ns() {
    auto t0 = std::chrono::steady_clock::now();
    Tick  r0 = rdtsc_start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Tick  r1 = rdtsc_end();
    auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return static_cast<double>(r1 - r0) / ns;
}

// --- Log-linear histogram ---------------------------------------------------
// 64 magnitude exponents x (1<<kHistSubBits) sub-buckets. With kHistSubBits=3
// that's 8 sub-buckets per power of 2, ~12.5% relative precision per bin.
// Layout: for v < 2^kHistSubBits (kHistSubs) we use v directly as the index.
// For larger v, exp = floor(log2(v)), sub = the kHistSubBits bits below the
// top bit; idx = (exp - kHistSubBits + 1) * kHistSubs + sub.
inline constexpr size_t kHistSubBits = 3;
inline constexpr size_t kHistSubs    = size_t{1} << kHistSubBits;   // 8
inline constexpr size_t kHistBuckets = 512;                         // max idx 495

inline size_t bucket_index(uint64_t v) {
    if (v < kHistSubs) return static_cast<size_t>(v);
    const unsigned exp =
        63u - static_cast<unsigned>(__builtin_clzll(static_cast<unsigned long long>(v)));
    const unsigned sub =
        static_cast<unsigned>((v >> (exp - kHistSubBits)) & (kHistSubs - 1));
    return (static_cast<size_t>(exp) - kHistSubBits + 1) * kHistSubs + sub;
}

// Lower bound and width of the value range covered by bucket b.
inline uint64_t bucket_lower(size_t b) {
    if (b < kHistSubs) return b;
    const size_t exp = b / kHistSubs + kHistSubBits - 1;
    const size_t sub = b % kHistSubs;
    return (kHistSubs + sub) << (exp - kHistSubBits);
}
inline uint64_t bucket_width(size_t b) {
    if (b < kHistSubs) return 1;
    const size_t exp = b / kHistSubs + kHistSubBits - 1;
    return uint64_t{1} << (exp - kHistSubBits);
}

// --- Per-thread stats ------------------------------------------------------
// alignas(64) + trailing pad: one ThreadStats owns whole cache lines so the
// hot-loop writes in record_latency() never false-share with a neighbour.
struct alignas(64) ThreadStats {
    size_t p = 0;           // successful pushes (drives items/s numerator)
    size_t c = 0;           // successful pops
    size_t pmiss = 0;       // push attempts that failed (queue full)
    size_t cmiss = 0;       // pop attempts that failed (queue empty); also
                            // triggers producer-role flip in lmbm harness
    size_t lat_count = 0;   // latencies recorded (post-warmup, non-negative)
    size_t lat_neg   = 0;   // latencies discarded as negative (TSC migration)
    double lat_sum   = 0.0; // for running mean
    double lat_sumsq = 0.0; // running sum of squares; print_row() derives the
                            // variance from it one-pass as E[x^2]-E[x]^2
                            // (naive, not Welford — see guard in print_row())
    Tick   lat_min   = std::numeric_limits<Tick>::max();
    Tick   lat_max   = 0;
    std::array<uint64_t, kHistBuckets> hist{};  // log-linear latency histogram
    char pad[64];           // isolate from the next ThreadStats cache line
};

struct RunResult {
    size_t n_threads = 0;
    size_t capacity  = 0;
    double elapsed_s = 0.0;
    size_t total_p = 0, total_c = 0, total_pmiss = 0, total_cmiss = 0;
    size_t lat_count = 0;
    size_t lat_neg   = 0;
    double lat_sum   = 0.0;
    double lat_sumsq = 0.0;
    Tick   lat_min   = std::numeric_limits<Tick>::max();
    Tick   lat_max   = 0;
    std::array<uint64_t, kHistBuckets> hist{};
};

// Hot-loop helper: record a latency cycle-count into a ThreadStats. All
// writes land in this thread's own cache line (see alignas/pad above) so
// the cost stays in the few-ns range and doesn't inflate what we measure.
static inline void record_latency(ThreadStats& s, Tick lat) {
    if (lat < s.lat_min) s.lat_min = lat;
    if (lat > s.lat_max) s.lat_max = lat;
    const double x = static_cast<double>(lat);
    s.lat_sum   += x;
    s.lat_sumsq += x * x;
    ++s.lat_count;
    ++s.hist[bucket_index(lat)];
}

inline void reduce(const std::vector<ThreadStats>& stats, RunResult& r) {
    for (const auto& s : stats) {
        r.total_p     += s.p;
        r.total_c     += s.c;
        r.total_pmiss += s.pmiss;
        r.total_cmiss += s.cmiss;
        r.lat_count   += s.lat_count;
        r.lat_neg     += s.lat_neg;
        r.lat_sum     += s.lat_sum;
        r.lat_sumsq   += s.lat_sumsq;
        if (s.lat_min < r.lat_min) r.lat_min = s.lat_min;
        if (s.lat_max > r.lat_max) r.lat_max = s.lat_max;
        for (size_t b = 0; b < kHistBuckets; ++b) r.hist[b] += s.hist[b];
    }
}

// Extract the pth percentile in cycles, linearly interpolating inside the bin.
inline double percentile_cycles(const std::array<uint64_t, kHistBuckets>& h,
                                size_t total, double p) {
    if (total == 0) return 0.0;
    const double target = p * 0.01 * static_cast<double>(total);
    uint64_t cum = 0;
    for (size_t b = 0; b < kHistBuckets; ++b) {
        const uint64_t cnt = h[b];
        if (cnt == 0) continue;
        const double new_cum = static_cast<double>(cum + cnt);
        if (new_cum >= target) {
            const double frac = (target - static_cast<double>(cum)) /
                                static_cast<double>(cnt);
            return static_cast<double>(bucket_lower(b)) +
                   frac * static_cast<double>(bucket_width(b));
        }
        cum += cnt;
    }
    // Should be unreachable; return max observed.
    return 0.0;
}

// --- Optional full-histogram dump -----------------------------------------
// The log-linear histogram (see above) gives ~12.5% relative bin width, fine
// enough to see bimodality, heavy tails, and cluster structure that p50/p99
// alone hide. Off by default because one run produces ~100 non-empty buckets
// per sweep point and a lmbm sweep has five thread counts * four alignments.
//
// Each harness's CLI sets this from --dump-hist; print_row consults it.
inline bool g_dump_hist = false;

// Print every non-empty bucket of `h` for a run with total `lat_count`, with
// bin edges converted to ns. Columns: lower_ns, upper_ns, count, pct,
// cum_pct, and an ASCII bar scaled to the modal bucket.
inline void dump_histogram(const std::array<uint64_t, kHistBuckets>& h,
                           size_t lat_count, double cyc_per_ns) {
    if (lat_count == 0) return;
    uint64_t max_cnt = 0;
    for (uint64_t c : h) if (c > max_cnt) max_cnt = c;
    if (max_cnt == 0) return;

    const double inv = 1.0 / cyc_per_ns;
    constexpr int kBarWidth = 40;

    std::printf("# histogram (%zu samples)\n", lat_count);
    std::printf("#   lower_ns    upper_ns        count    pct    cum_pct\n");
    uint64_t cum = 0;
    for (size_t b = 0; b < kHistBuckets; ++b) {
        const uint64_t cnt = h[b];
        if (cnt == 0) continue;
        cum += cnt;
        const double lower_ns = static_cast<double>(bucket_lower(b)) * inv;
        const double upper_ns = static_cast<double>(bucket_lower(b) +
                                                    bucket_width(b)) * inv;
        const double pct     = 100.0 * static_cast<double>(cnt) /
                               static_cast<double>(lat_count);
        const double cum_pct = 100.0 * static_cast<double>(cum) /
                               static_cast<double>(lat_count);
        const int bar_len = static_cast<int>(
            static_cast<double>(kBarWidth) * static_cast<double>(cnt) /
            static_cast<double>(max_cnt));
        std::printf("# %10.2f %11.2f %12llu %6.2f%% %8.2f%%  ",
                    lower_ns, upper_ns,
                    static_cast<unsigned long long>(cnt), pct, cum_pct);
        for (int i = 0; i < bar_len; ++i) std::putchar('#');
        std::putchar('\n');
    }
    std::fflush(stdout);
}

// --- Output ----------------------------------------------------------------
inline void print_header() {
    std::printf("%-16s %4s %12s %8s %10s %10s %10s %10s",
                "Benchmark", "thr", "count", "neg",
                "mean_ns", "stdev_ns", "min_ns", "max_ns");
    for (double p : kLatPercentiles) {
        char label[16];
        if (p == static_cast<int>(p)) std::snprintf(label, sizeof(label), "p%d_ns",   static_cast<int>(p));
        else                          std::snprintf(label, sizeof(label), "p%.1f_ns", p);
        std::printf(" %10s", label);
    }
    std::printf(" %10s\n", "items/s");
}

inline void print_row(const char* name, const RunResult& r, double cyc_per_ns) {
    const double n = static_cast<double>(r.lat_count);
    const double mean_c = n > 0 ? r.lat_sum / n : 0.0;
    double var_c = 0.0;
    if (n > 1) {
        // One-pass variance from the running sum/sum-of-squares accumulators
        // (E[x^2]-E[x]^2 form). This is fast but prone to catastrophic
        // cancellation when the mean is large relative to the spread, which
        // can drive the numerator slightly negative; the `num > 0` guard
        // clamps that to 0 rather than feeding sqrt() a negative value.
        const double num = r.lat_sumsq - n * mean_c * mean_c;
        var_c = (num > 0.0) ? num / (n - 1.0) : 0.0;
    }
    const double sd_c = std::sqrt(var_c);
    const double inv  = 1.0 / cyc_per_ns;

    const double mean_ns = mean_c * inv;
    const double sd_ns   = sd_c   * inv;
    const double min_ns  = static_cast<double>(r.lat_min) * inv;
    const double max_ns  = static_cast<double>(r.lat_max) * inv;
    const double items_mps = static_cast<double>(r.total_p + r.total_c) /
                             r.elapsed_s / 1e6;

    std::printf("%-16s %4zu %12zu %8zu %10.2f %10.2f %10.2f %10.2f",
                name, r.n_threads, r.lat_count, r.lat_neg,
                mean_ns, sd_ns, min_ns, max_ns);
    for (double p : kLatPercentiles) {
        const double c = percentile_cycles(r.hist, r.lat_count, p);
        std::printf(" %10.2f", c * inv);
    }
    std::printf(" %9.2fM\n", items_mps);
    std::fflush(stdout);

    if (g_dump_hist) dump_histogram(r.hist, r.lat_count, cyc_per_ns);
}

} // namespace latbench

#endif // LATBENCH_COMMON_H
