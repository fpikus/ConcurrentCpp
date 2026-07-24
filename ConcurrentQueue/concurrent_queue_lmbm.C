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
// Custom LATENCY benchmark for RingAtomicMapQueueMPMC (the "l" in lmbm =
// latency). Measures push->pop HANDOFF latency under MPMC contention: the
// time from just before a producer's push to just after a consumer's pop of
// that same element. Contrast the sibling benchmarks: gmbm/mbm report
// throughput, and ppmbm reports uncontended 1P-1C round-trip latency.
//
// Key=Value=uint64_t so the producer's timestamp travels *inside the slot* as
// the value: the consumer recovers it from the popped element and subtracts.
// "No ring race" -- there is no separate side array indexed by ring position
// that a wrapped-around producer/consumer could stomp; the stamp is bound to
// the element, so latency is attributed correctly no matter which threads
// produced and consumed it.
//
// Self-balancing roles (see run_once below): every thread prefers to consume
// and only produces after an empty miss. This holds the queue near-empty, so
// the measured interval is dominated by cross-core handoff, not by time the
// item spends buffered in a full ring. Roles are DYNAMIC here, unlike the
// static odd/even producer/consumer split in gmbm/mbm.
//
// Measured interval = rdtsc_start() taken by the producer immediately before
// push, to rdtsc_end() taken by the consumer immediately after pop. Samples
// that come out negative (TSC skew from a thread migrating across cores
// despite affinity) are counted in lat_neg and discarded; the first
// warmup_skip valid samples per thread are also dropped.
//
// See DN_queue/queue_lmbm.C for the original self-balancing harness design.

#include "concurrent_queue.h"
#include "latbench_common.h"

#include <memory>

namespace {

using namespace latbench;

template <typename Q>
RunResult run_once(size_t capacity_hint, size_t n_threads,
                   std::chrono::nanoseconds duration,
                   size_t refill_batch, size_t warmup_skip) {
    constexpr size_t elem_size  = Q::element_size();
    constexpr size_t elem_align = Q::element_align();
    const size_t NB = capacity_hint * elem_size;

    void* memory = ::operator new(NB, std::align_val_t{elem_align});
    // Heap-allocated so the queue can be destroyed before its buffer is
    // freed below: a stack instance dies at scope exit, after the operator
    // delete, and its draining destructor would walk freed memory.
    auto q_owner = std::make_unique<Q>(memory, NB);
    Q& q = *q_owner;
    const size_t N = q.capacity();

    std::latch start_latch(n_threads + 1);
    std::atomic<bool> stop{false};
    std::vector<ThreadStats> stats(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            bind_current_thread_to_cpu(static_cast<int>(t));
            ThreadStats& s = stats[t];
            size_t warmup = warmup_skip;
            constexpr uint64_t kKey = 1;  // non-zero sentinel-avoiding key
            start_latch.arrive_and_wait();

            // Self-balancing roles: every thread prefers to consume; only on
            // an empty miss does it switch to producing a batch of stamped
            // items. This keeps the queue near-empty, so the measured pop
            // latency is push-to-pop handoff time, not time spent buffered.
            while (!stop.load(std::memory_order_relaxed)) {
                Tick ts = 0;
                if (q.pop(ts) != 0) {
                    Tick now = rdtsc_end();
                    int64_t d = static_cast<int64_t>(now - ts);
                    if (d >= 0) {
                        if (warmup > 0) --warmup;
                        else            record_latency(s, static_cast<Tick>(d));
                    } else ++s.lat_neg;
                    ++s.c;
                } else {
                    ++s.cmiss;
                    for (size_t i = 0;
                         i < refill_batch && !stop.load(std::memory_order_relaxed);
                         ++i) {
                        Tick ts2 = rdtsc_start();
                        if (q.push(kKey, ts2)) ++s.p;
                        else { ++s.pmiss; break; }
                    }
                }
            }
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
    reduce(stats, r);
    return r;
}

template <size_t NTRY, size_t ALIGN>
void sweep(const char* name, size_t capacity_hint,
           const std::vector<size_t>& thread_counts,
           std::chrono::nanoseconds duration, double cyc_per_ns,
           size_t refill_batch, size_t warmup_skip) {
    using Q = RingAtomicMapQueueMPMC<uint64_t, uint64_t, NTRY, ALIGN>;
    for (size_t nt : thread_counts) {
        auto r = run_once<Q>(capacity_hint, nt, duration,
                             refill_batch, warmup_skip);
        print_row(name, r, cyc_per_ns);
    }
}

} // namespace

int main(int argc, char** argv) {
    size_t capacity_hint = 1UL << 16;
    double duration_s    = 5.0;
    size_t refill_batch  = 8;
    size_t warmup_skip   = 100;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--duration=", 0) == 0) duration_s    = std::atof(a.c_str() + 11);
        else if (a.rfind("--capacity=", 0) == 0) capacity_hint = std::strtoull(a.c_str() + 11, nullptr, 0);
        else if (a.rfind("--batch=",    0) == 0) refill_batch  = std::strtoull(a.c_str() + 8,  nullptr, 0);
        else if (a.rfind("--warmup=",   0) == 0) warmup_skip   = std::strtoull(a.c_str() + 9,  nullptr, 0);
        else if (a == "--dump-hist") latbench::g_dump_hist = true;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--duration=SECONDS] [--capacity=N] [--batch=N] [--warmup=N] [--dump-hist]\n", argv[0]);
            return 0;
        }
    }

    const size_t hw = latbench::hw_thread_count();
    const double cyc_per_ns = latbench::calibrate_cycles_per_ns();

    std::vector<size_t> thread_counts;
    for (size_t t = 2; t <= hw; t *= 2) thread_counts.push_back(t);
    if (thread_counts.empty() || thread_counts.back() != hw)
        thread_counts.push_back(hw);

    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(duration_s));

    std::printf("hw_threads=%zu capacity_hint=%zu duration=%.3fs batch=%zu warmup=%zu rdtsc=%.4f GHz\n\n",
                hw, capacity_hint, duration_s, refill_batch, warmup_skip, cyc_per_ns);
    latbench::print_header();

    sweep<8, 0>  ("Ring_a0",   capacity_hint, thread_counts, duration, cyc_per_ns, refill_batch, warmup_skip);
    sweep<8, 16> ("Ring_a16",  capacity_hint, thread_counts, duration, cyc_per_ns, refill_batch, warmup_skip);
    sweep<8, 64> ("Ring_a64",  capacity_hint, thread_counts, duration, cyc_per_ns, refill_batch, warmup_skip);
    sweep<8, 128>("Ring_a128", capacity_hint, thread_counts, duration, cyc_per_ns, refill_batch, warmup_skip);

    return 0;
}
