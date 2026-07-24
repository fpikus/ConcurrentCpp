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
// Ping-pong LATENCY benchmark for RingAtomicMapQueueMPMC. Canonical ppmbm
// harness -- other queues' queue_ppmbm.C reference this file.
//
// Goal: measure 1-producer / 1-consumer round-trip latency, a setting where
// every queue implementation has its best shot. Isolates per-op queue cost
// from the contention behaviour exercised by queue_lmbm.C.
//
// Two queues, two threads:
//   q1: sender -> receiver
//   q2: receiver -> sender (echo path)
// The sender stamps rdtsc before push, busy-waits on pop from q2, stamps
// rdtsc after. The difference is one full round-trip through two queues
// plus the receiver's trivial pop-push. Blocking on cpu_pause() keeps the
// cost close to the hardware minimum without kernel involvement.
//
// The queues are key-only (Value=void), so the rdtsc timestamp is carried as
// the KEY itself and pop() returns it directly -- no value slot, and 0 is a
// safe "empty" sentinel because a real timestamp is never 0. (lmbm instead
// carries the stamp as the value; here the key-only slot is all we need.)
//
// Capacity is deliberately small (64): we want handoff latency, not
// buffering. Large capacity lets the sender run ahead and measures queue
// throughput instead.
//
// CPU affinity: --tx-cpu / --rx-cpu pin the two threads. Pairing matters
// a lot -- SMT siblings share L1/L2 (cheapest), same-CCD different cores
// share L3, cross-CCD traverses Infinity Fabric. On WSL the topology is
// virtualised to one flat L3; run on bare Linux for cross-CCD results.
//
// Ring-specific sweep: NTRY stays at 8 (RFO round-trip budget, see the
// queue header) while ALIGN varies 0/16/64/128 to study cache-line
// alignment of the per-slot key/busy atomics.
//
// Warm-up: `warmup_skip` samples discarded to skip initial page faults,
// branch-predictor training, and rdtsc first-read skew. Items/s counts
// two per round trip: each round trip is two queue traversals (push+pop
// through q1, push+pop through q2), so the queues do twice as many item
// transfers as there are round trips. Warm-up traversals are included,
// so items/s reflects total work done, not just measured samples.

#include "concurrent_queue.h"
#include "latbench_common.h"

#include <memory>

namespace {

using namespace latbench;

constexpr size_t QCAP_HINT = 64;      // small on purpose: 1P-1C handoff, not buffering
constexpr size_t N_DEFAULT = 1UL << 25;  // ~5 s wall-clock across queues
constexpr size_t WARMUP_DEFAULT = 1000;

template <typename Q>
RunResult run_once(size_t capacity_hint, size_t n_messages, size_t warmup_skip,
                   int tx_cpu, int rx_cpu) {
    constexpr size_t elem_size  = Q::element_size();
    constexpr size_t elem_align = Q::element_align();
    const size_t NB = capacity_hint * elem_size;

    // Two independent queues for the two directions of the ping-pong.
    // Raw aligned buffers; Ring takes a caller-owned memory region.
    void* m1 = ::operator new(NB, std::align_val_t{elem_align});
    void* m2 = ::operator new(NB, std::align_val_t{elem_align});
    // Heap-allocated so the queues can be destroyed before their buffers are
    // freed below: stack instances die at scope exit, after the operator
    // deletes, and their draining destructors would walk freed memory.
    auto q1_owner = std::make_unique<Q>(m1, NB);
    auto q2_owner = std::make_unique<Q>(m2, NB);
    Q& q1 = *q1_owner;
    Q& q2 = *q2_owner;
    const size_t cap = q1.capacity();

    // Latch of 3: tx + rx + orchestrator. The orchestrator arrives last so
    // it can take the wall-clock timestamp immediately after both workers
    // start their first iteration, not before thread creation settles.
    std::latch start_latch(3);
    std::vector<ThreadStats> stats(2);

    // Receiver: trivial echo loop. pop returns the payload (0 means empty
    // -- valid because we never push a 0-valued rdtsc timestamp; the key
    // is non-zero by construction). Push may fail when q2 is full if the
    // sender hasn't drained yet; spin-wait on cpu_pause().
    std::thread rx([&] {
        bind_current_thread_to_cpu(rx_cpu);
        start_latch.arrive_and_wait();
        for (size_t i = 0; i < n_messages; ++i) {
            Tick v;
            while ((v = q1.pop()) == 0) cpu_pause();   // await sender's stamp
            while (!q2.push(v)) cpu_pause();           // forward the echo
        }
    });

    // Sender: stamp, push, wait for echo, stamp, record delta.
    std::thread tx([&] {
        bind_current_thread_to_cpu(tx_cpu);
        ThreadStats& s = stats[0];
        size_t warmup = warmup_skip;
        start_latch.arrive_and_wait();
        for (size_t i = 0; i < n_messages; ++i) {
            // Pre-push stamp: lfence-bracketed rdtsc. The queue capacity is
            // 64 so under normal conditions push does not block; the
            // cpu_pause() spin is for the rare full-queue stall.
            Tick t0 = rdtsc_start();
            while (!q1.push(t0)) cpu_pause();
            // Await echoed stamp and stop the clock. rdtscp + lfence ensures
            // t1 reflects all prior retired ops, including the pop.
            Tick echoed;
            while ((echoed = q2.pop()) == 0) cpu_pause();
            Tick t1 = rdtsc_end();
            int64_t d = static_cast<int64_t>(t1 - t0);
            if (d >= 0) {
                if (warmup > 0) --warmup;              // skip initial samples
                else            record_latency(s, static_cast<Tick>(d));
            } else {
                // Sign-flip only possible if affinity failed and the thread
                // migrated mid-sample to a core with a lagging TSC.
                ++s.lat_neg;
            }
            ++s.c;
        }
        // p + c = 2*n_messages by design: a round trip is two queue
        // traversals, and items/s counts traversals (see file header).
        s.p = n_messages;
    });

    // Hold the clock until both workers are running steady-state.
    start_latch.arrive_and_wait();
    const auto t0 = std::chrono::steady_clock::now();
    tx.join();
    rx.join();
    const auto t1 = std::chrono::steady_clock::now();

    q1_owner.reset();   // Destroy the queues before freeing their buffers
    q2_owner.reset();
    ::operator delete(m1, std::align_val_t{elem_align});
    ::operator delete(m2, std::align_val_t{elem_align});

    RunResult r{};
    r.n_threads = 2;
    r.capacity  = cap;
    r.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    reduce(stats, r);
    return r;
}

template <size_t NTRY, size_t ALIGN>
void sweep(const char* name, size_t capacity_hint,
           size_t n_messages, size_t warmup_skip, double cyc_per_ns,
           int tx_cpu, int rx_cpu) {
    using Q = RingAtomicMapQueueMPMC<uint64_t, void, NTRY, ALIGN>;
    auto r = run_once<Q>(capacity_hint, n_messages, warmup_skip, tx_cpu, rx_cpu);
    print_row(name, r, cyc_per_ns);
}

} // namespace

int main(int argc, char** argv) {
    size_t capacity_hint = QCAP_HINT;
    size_t n_messages    = N_DEFAULT;
    size_t warmup_skip   = WARMUP_DEFAULT;
    int    tx_cpu        = 0;
    int    rx_cpu        = 1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--n=",        0) == 0) n_messages    = std::strtoull(a.c_str() + 4, nullptr, 0);
        else if (a.rfind("--capacity=", 0) == 0) capacity_hint = std::strtoull(a.c_str() + 11, nullptr, 0);
        else if (a.rfind("--warmup=",   0) == 0) warmup_skip   = std::strtoull(a.c_str() + 9,  nullptr, 0);
        else if (a.rfind("--tx-cpu=",   0) == 0) tx_cpu        = std::atoi(a.c_str() + 9);
        else if (a.rfind("--rx-cpu=",   0) == 0) rx_cpu        = std::atoi(a.c_str() + 9);
        else if (a == "--dump-hist") latbench::g_dump_hist = true;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--n=MESSAGES] [--capacity=N] [--warmup=N] [--tx-cpu=N] [--rx-cpu=N] [--dump-hist]\n", argv[0]);
            return 0;
        }
    }

    const double cyc_per_ns = latbench::calibrate_cycles_per_ns();

    std::printf("ping-pong  n=%zu capacity_hint=%zu warmup=%zu tx-cpu=%d rx-cpu=%d rdtsc=%.4f GHz\n\n",
                n_messages, capacity_hint, warmup_skip, tx_cpu, rx_cpu, cyc_per_ns);
    latbench::print_header();

    sweep<8, 0>  ("Ring_a0",   capacity_hint, n_messages, warmup_skip, cyc_per_ns, tx_cpu, rx_cpu);
    sweep<8, 16> ("Ring_a16",  capacity_hint, n_messages, warmup_skip, cyc_per_ns, tx_cpu, rx_cpu);
    sweep<8, 64> ("Ring_a64",  capacity_hint, n_messages, warmup_skip, cyc_per_ns, tx_cpu, rx_cpu);
    sweep<8, 128>("Ring_a128", capacity_hint, n_messages, warmup_skip, cyc_per_ns, tx_cpu, rx_cpu);

    return 0;
}
