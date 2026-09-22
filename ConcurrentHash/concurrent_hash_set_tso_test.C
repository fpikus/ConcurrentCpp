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
// ===========================================================================
// STORE-BUFFERING REGRESSION TEST -- lost key / lost erase across a resize.
//
// WHY THIS IS A SEPARATE BINARY, AND WHY IT IS NOT IN concurrent_hash_set_test.C:
// the bug this guards against is a real-hardware store-buffering effect, and the
// conditions that make it observable are exactly the conditions the sanitizer
// test binaries do not provide.
//   - The resizing thread must reach its next bucket-head load with the
//     `table_size_` store still sitting in its store buffer. That means NO locked
//     instruction in between. ASan and TSan instrument every memory access and
//     their runtimes execute locked instructions all over the resizer's path,
//     which drain the buffer and destroy the execution.
//   - -O0 spills and reloads around every statement, stretching the path by
//     hundreds of instructions and making the drain overwhelmingly likely anyway.
// So this test must be built -O3 with no sanitizer. The Makefile builds it as its
// own target, and it is the ONE test in this directory that is not a sanitizer
// build. Everything it needs is in gtest; gtest itself adds no synchronization on
// the racing threads' path (the trial loop calls nothing but the container), so it
// is a normal gtest and `make run_tests` reports it like any other.
//
// WHAT IT DEMONSTRATES. The pre-seal header decided insert() across two
// independent atomics -- it CASed a bucket head, then re-read `table_size_` to
// find out whether the geometry had moved under it. That is the store-buffering
// (Dekker) shape: thread A writes head and reads table_size_, the resizer writes
// table_size_ and reads head. Acquire/release on two DIFFERENT atomics does not
// order it on ANY architecture, x86 included: `table_size_.store(release)` is a
// plain MOV there, and the resizer reaches split_bucket()'s parent-head load with
// nothing to drain its buffer. Measured on this x86 box: ~1000-3500 lost keys and
// ~3400 lost erases per million trials against the stock header, and 0 per six
// million against a control that differs only in making that one store seq_cst
// (i.e. an XCHG) -- which pins the cause on the store buffer and nothing else.
// The seal+freeze design removes the shape entirely: the decision and the split's
// interlock are read-modify-writes of the SAME word (the bucket head for insert,
// the node link for erase), and single-word RMWs are totally ordered everywhere.
//
// THE TARGETED EXECUTION, per trial, on a fresh 4-bucket set holding 8 nodes (all
// in bucket 0), so that the next insert doubles the table to 8:
//   R: insert(64)   -> publishes, then doubles: `table_size_ = 8` is a plain store
//                      that may sit in R's store buffer; R itself sees 8 by
//                      forwarding.
//      contains(13) -> bucket 5 is UNINITIALIZED, so R splits it, which loads
//                      bucket 1's head (the snapshot) with no locked instruction
//                      since the table_size_ store.
//   A: insert(5)    -> loaded table_size_ == 4, so bucket 1. If its CAS lands after
//                      R's snapshot and A's post-CAS table_size_ load still returns
//                      4 (R's store not yet drained), a header with the two-atomic
//                      recheck does not re-insert: key 5 sits in bucket 1 only,
//                      bucket 5 was published without it, and contains(5) is false
//                      forever. The erase-side dual pre-inserts key 5 and has A
//                      call erase(5): the split copies the node live before A's
//                      mark lands, so erase(5) returns true and the key stays.
// The hammer threads keep the `table_size_` cache line shared by several cores so
// R's request for ownership takes longer, widening the window.
// ===========================================================================
#include "concurrent_hash_set.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdio>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#if defined(__linux__)
#  include <sched.h>
#  define HS_HAVE_SCHED_AFFINITY 1
#else
// macOS and the BSDs have no sched_setaffinity (and no portable equivalent that
// pins a thread to a specific core), so the harness runs unpinned there. It is
// then a weaker probe -- the two racing threads may share a core and never race --
// but it is still sound: a loss it DOES observe is a real defect.
#  define HS_HAVE_SCHED_AFFINITY 0
#endif

// Pin the calling thread to `cpu`. Returns false if the platform has no affinity
// API or the call was rejected (containers and cpusets routinely reject it); the
// caller records that and carries on unpinned rather than failing.
static bool pin_to_cpu(int cpu) {
#if HS_HAVE_SCHED_AFFINITY
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof(s), &s) == 0;
#else
    (void)cpu;
    return false;
#endif
} // pin_to_cpu()

// Outcome of one harness run.
struct TsoResult {
    long trials = 0;      // trials actually executed
    long a_true = 0;      // times A's insert(5)/erase(5) returned true
    long lost = 0;        // trials whose final state contradicts A's `true`
    bool pinned = false;  // whether the racing threads were pinned to distinct CPUs
};

// Run `trials` targeted trials. EraseSide selects the erase-side dual.
//
// Thread roles: R is the resizer/splitter, A is the stale operation, and `hammers`
// readers keep the table_size_ line shared. All of them are persistent and are
// driven by the `go` counter -- creating threads per trial would cost a thousand
// times the window being measured.
//
// CPU assignment. The only thing that matters is that R and A sit on two DIFFERENT
// physical cores; with SMT, sibling hyperthreads share a store buffer and the race
// cannot happen at all, hence the stride of 2 when there are enough CPUs. When
// there are not, the harness falls back to stride 1 and finally to no pinning; it
// never refuses to run, because an unpinned loss is still a loss.
template <bool EraseSide>
static TsoResult run_tso_trials(long trials, int hammers) {
    using Set = ConcurrentResizableHashSet<int, EraseSide>;
    const int BATCH = 1000;

    const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
    if (ncpu > 0 && hammers > ncpu - 3) hammers = ncpu - 3;
    if (hammers < 0) hammers = 0;
    // Slot 0 is this thread, 1 is R, 2 is A, 3.. are the hammers.
    const int slots = 3 + hammers;
    const int stride = (ncpu >= 2*slots) ? 2 : 1;
    const bool want_pin = HS_HAVE_SCHED_AFFINITY && ncpu >= slots;
    std::atomic<int> pin_failures{0};
    auto pin_slot = [&](int slot) {
        if (!want_pin || pin_to_cpu(slot*stride)) return;
        pin_failures.fetch_add(1, std::memory_order_relaxed);
    };

    std::vector<std::unique_ptr<Set>> sets(BATCH);
    std::atomic<long> go{-1};      // index of the trial to run, -1 = none yet
    std::atomic<int> done{0};      // R and A each add 1 per trial
    std::atomic<bool> stop{false};
    std::atomic<long> a_true{0};

    std::thread R([&]() {
        pin_slot(1);
        long seen = -1;
        while (true) {
            long g;
            while ((g = go.load(std::memory_order_relaxed)) == seen) {
                if (stop.load(std::memory_order_relaxed)) return;
            }
            seen = g;
            Set& set = *sets[g%BATCH];
            set.insert(64);        // the 9th node: doubles 4 -> 8
            set.contains(13);      // touches bucket 5, so R itself splits it from bucket 1
            done.fetch_add(1);
        } // trial loop
    });

    std::thread A([&]() {
        pin_slot(2);
        std::minstd_rand rng(12345);
        long seen = -1;
        while (true) {
            long g;
            while ((g = go.load(std::memory_order_relaxed)) == seen) {
                if (stop.load(std::memory_order_relaxed)) return;
            }
            seen = g;
            Set& set = *sets[g%BATCH];
            // Jitter: sweep A's deciding CAS across R's store -> snapshot -> drain
            // window. Without it A would land at one fixed offset and most trials
            // would miss the window entirely.
            for (volatile unsigned d = rng()%192u; d != 0; d = d - 1) {}
            bool won;
            if constexpr (EraseSide) {
                won = set.erase(5);
            } else {
                won = set.insert(5);
            }
            if (won) a_true.fetch_add(1, std::memory_order_relaxed);
            done.fetch_add(1);
        } // trial loop
    });

    // Quiescence handshake between the hammers and this thread (seq_cst on both
    // sides -- it IS a store-buffering pattern): a hammer clears idle[h] BEFORE
    // loading cur, and this thread stores cur = -1 and THEN waits for every
    // idle[h], so no hammer is still holding a set that is about to be destroyed.
    std::atomic<long> cur{-1};
    std::vector<std::atomic<bool>> idle(hammers);
    for (std::atomic<bool>& f : idle) f.store(true);
    std::vector<std::thread> H;
    for (int h = 0; h < hammers; ++h) {
        H.emplace_back([&, h]() {
            pin_slot(3 + h);
            size_t sink = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                idle[h].store(false);
                long g = cur.load();
                // Key 8 lives in bucket 0 at every table size that occurs here, so
                // a hammer never splits bucket 5 and never touches bucket 1: it
                // only keeps the table_size_ line shared.
                if (g >= 0) sink += sets[g%BATCH]->contains(8) ? 1u : 0u;
                idle[h].store(true);
            } // hammer loop
            if (sink == ~size_t{0}) std::printf(" ");   // keep `sink` alive
        });
    } // spawn hammers

    pin_slot(0);
    TsoResult out;
    for (long base = 0; base < trials; base += BATCH) {
        cur.store(-1);
        for (int h = 0; h < hammers; ++h) { while (!idle[h].load()) {} }
        for (int i = 0; i < BATCH; ++i) {
            sets[i] = std::make_unique<Set>(4);
            if constexpr (EraseSide) {
                sets[i]->insert(5);                                    // the victim, bucket 1
                for (int k = 0; k < 56; k += 8) sets[i]->insert(k);    // 7 more nodes, all in bucket 0
            } else {
                for (int k = 0; k < 64; k += 8) sets[i]->insert(k);    // 8 nodes, all in bucket 0
            }
        } // build the batch
        for (int i = 0; i < BATCH; ++i) {
            cur.store(i);
            int d0 = done.load();
            go.store(base + i);
            while (done.load(std::memory_order_relaxed) != d0 + 2) {}
            ++out.trials;
        } // run the batch
        cur.store(-1);
        for (int h = 0; h < hammers; ++h) { while (!idle[h].load()) {} }
        for (int i = 0; i < BATCH; ++i) {
            // Insert side: A's insert(5) returned true, so 5 must be a member.
            // Erase side: A's erase(5) returned true, so 5 must NOT be a member.
            const bool present = sets[i]->contains(5);
            if (EraseSide ? present : !present) ++out.lost;
        } // score the batch
    } // batches

    stop.store(true);
    R.join();
    A.join();
    for (std::thread& h : H) h.join();
    out.a_true = a_true.load();
    out.pinned = want_pin && pin_failures.load(std::memory_order_relaxed) == 0;
    return out;
} // run_tso_trials()

// Trials per side, sized so that the whole binary runs in roughly fifteen seconds
// on a 16-thread laptop. The two sides have very different costs per trial: on the
// insert side BOTH racing threads allocate a node, so they contend for the arena's
// SpinLock (whose back-off sleeps), at ~27 us per trial; the erase side allocates
// on one side only and costs ~1.5 us. The counts are chosen to spend the time where
// it buys detection: measured against the stock header, roughly one trial in 560
// loses a key on the insert side and one in 310 loses an erase, so these counts
// expect ~700 and ~6400 detections respectively -- a run that reports zero is
// evidence, not luck.
static constexpr long TSO_TRIALS_INSERT = 400000;
static constexpr long TSO_TRIALS_ERASE = 2000000;
static constexpr int TSO_HAMMERS = 4;

static void report(const TsoResult& r, const char* what) {
    std::printf("[   INFO   ] %s: trials=%ld A_returned_true=%ld lost=%ld pinned=%s\n",
                what, r.trials, r.a_true, r.lost, r.pinned ? "yes" : "NO (weaker probe)");
    if (!r.pinned) {
        std::printf("[   INFO   ] could not pin the racing threads to distinct CPUs; the race window\n"
                    "[   INFO   ] is much less likely to open, so a PASS here is weak evidence.\n");
    }
} // report()

// A key that insert() reported as newly inserted must be a member afterwards --
// no matter what a concurrent resize did. Guards the insert-side store-buffering
// lost key.
TEST(ConcurrentHashSetTsoTest, InsertedKeyIsNotLostByAConcurrentResize) {
    const TsoResult r = run_tso_trials<false>(TSO_TRIALS_INSERT, TSO_HAMMERS);
    report(r, "insert side");
    ASSERT_GT(r.a_true, 0) << "harness broken: insert(5) never reported a new key, so nothing was tested";
    EXPECT_EQ(r.lost, 0) << r.lost << " of " << r.trials
                         << " trials: insert(5) returned true and contains(5) was false afterwards";
} // InsertedKeyIsNotLostByAConcurrentResize

// A key that erase() reported as removed must be absent afterwards -- no matter
// what a concurrent resize did. Guards the erase-side dual: the split copies the
// node live before the mark lands, so the tombstone is left on a superseded node.
TEST(ConcurrentHashSetTsoTest, ErasedKeyIsNotResurrectedByAConcurrentResize) {
    const TsoResult r = run_tso_trials<true>(TSO_TRIALS_ERASE, TSO_HAMMERS);
    report(r, "erase side");
    ASSERT_GT(r.a_true, 0) << "harness broken: erase(5) never reported a deletion, so nothing was tested";
    EXPECT_EQ(r.lost, 0) << r.lost << " of " << r.trials
                         << " trials: erase(5) returned true and contains(5) was still true afterwards";
} // ErasedKeyIsNotResurrectedByAConcurrentResize
