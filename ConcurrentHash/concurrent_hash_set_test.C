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
// THE CONTRACT THESE TESTS ENFORCE (they are written against it, never against
// observed behavior):
//   1. insert(k) returns true iff THIS call performed an absent->present
//      transition of k: exactly one insert() returns true per such transition,
//      across any number of concurrent table doublings.
//   2. erase(k) returns true iff THIS call performed a present->absent
//      transition of k: exactly one erase() returns true per such transition.
//   3. Membership is exact at all times and across all resizes: after a call
//      that made k a member returns, contains(k) is true for every thread until
//      some call removes it, and vice versa. No key is ever lost by a resize and
//      no erased key is ever resurrected by one.
// Anything weaker than this (a "best effort" boolean, an under-counting insert)
// is a defect, not a documented relaxation -- see the header's "STALE GEOMETRY"
// section, which states 1 and 2 verbatim.
// ===========================================================================
#include "concurrent_hash_set.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <barrier>
#include <memory>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// Key scrambler -- REQUIRED by every test whose point is to exercise resizes.
//
// libstdc++'s (and libc++'s) std::hash<int> is the identity, so with small
// consecutive keys 0..K a key stops changing bucket the moment table_size > K:
// bucket = hash(k) & (ts-1) == k for every larger ts. After that point a
// doubling MOVES NOTHING, no parent chain is ever split for those keys, and a
// test built on them is blind to every stale-geometry bug at every doubling
// beyond its key range. (That is not hypothetical: two earlier versions of the
// erase stress test could not fail even against a header with the fix removed.)
//
// scramble() is a bijection on 32-bit ints (multiplication by an odd constant is
// invertible mod 2^32), so distinct indices still give distinct keys and every
// disjointness argument below is preserved -- but the high bits are pseudorandom,
// so each key has an independent 1/2 chance of moving at EVERY doubling.
// ---------------------------------------------------------------------------
static int scramble(int i) {
    return static_cast<int>(static_cast<unsigned>(i)*2654435761u);
}

// Run fn(t) on T threads released together by a barrier, then join them all.
// The barrier (rather than a spin on a start flag) is what makes the racing
// window open at the same instant on every thread; it also replaces the six
// hand-rolled `std::atomic<bool> start` loops this suite used to carry.
template <typename F>
static void run_threads(int T, F fn) {
    std::barrier start(T);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&start, &fn, t]() {
            start.arrive_and_wait();
            fn(t);
        });
    } // spawn loop
    for (std::thread& th : threads) th.join();
} // run_threads()

// Custom identity hash. Several tests need to control EXACTLY which bucket a key
// lands in (bucket = hash(key) & (table_size-1)), which requires a hash whose low
// bits are the key's low bits. std::hash<int> happens to be identity in libstdc++
// and libc++, but relying on that is implementation-defined; this functor makes
// the assumption explicit and portable for the tests that depend on it.
struct CollisionHash {
    size_t operator()(int key) const {
        return static_cast<size_t>(key);
    }
};

// Test basic insertions and reads
TEST(ConcurrentHashSetTest, BasicOperations) {
    ConcurrentResizableHashSet<int> set;
    EXPECT_TRUE(set.insert(1));
    EXPECT_FALSE(set.insert(1));
    EXPECT_TRUE(set.insert(2));
    EXPECT_TRUE(set.contains(1));
    EXPECT_TRUE(set.contains(2));
    EXPECT_FALSE(set.contains(3));
}

// Single-threaded correctness across many table doublings: every inserted key must
// be found, duplicates must report false, and absent keys must not be found.
// Keys are scrambled so that roughly half of them change bucket at each of the
// ~14 doublings this test drives; with consecutive keys the later doublings would
// leave the low keys in place and never split their chains (see scramble()).
TEST(ConcurrentHashSetTest, ManyResizesSingleThread) {
    ConcurrentResizableHashSet<int> set(4);
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(set.insert(scramble(i))) << "insert(" << i << ") wrongly reported duplicate";
    }
    for (int i = 0; i < N; i += 997) {
        EXPECT_FALSE(set.insert(scramble(i))) << "re-insert(" << i << ") wrongly reported new";
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(set.contains(scramble(i))) << "missing key " << i;
    }
    for (int i = N; i < N + 5000; ++i) {
        EXPECT_FALSE(set.contains(scramble(i))) << "false positive for " << i;
    }
} // ManyResizesSingleThread

// Many threads insert disjoint key ranges concurrently (forcing many concurrent
// resizes and splits). Both halves of the contract are checked EXACTLY:
//   - every key is present afterwards (membership is exact across resizes), and
//   - insert() returned true exactly T*per times. The keys are disjoint, so each
//     one has exactly one absent->present transition and exactly one call may
//     report it.
// Note on the previous oracle: this test used to assert EXPECT_LE(inserted, T*per)
// "to catch over-count". That assertion was VACUOUS -- T*per is also the number
// of insert() calls made, so the count cannot exceed it no matter what the
// implementation does. Only equality has any content here, and equality is what
// the contract says.
TEST(ConcurrentHashSetTest, ConcurrentDisjointRanges) {
    ConcurrentResizableHashSet<int> set(4);
    const int T = 8, per = 20000;
    std::atomic<int> inserted{0};
    run_threads(T, [&set, &inserted](int t) {
        int local = 0;
        for (int k = 0; k < per; ++k) {
            if (set.insert(scramble(t*per + k))) ++local;
        }
        inserted += local;
    });

    EXPECT_EQ(inserted.load(), T*per) << "insert() did not report exactly one winner per key";
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < per; ++k) {
            EXPECT_TRUE(set.contains(scramble(t*per + k))) << "missing " << (t*per + k);
        }
    } // membership sweep
} // ConcurrentDisjointRanges

// Many threads racing to insert the SAME set of keys, with the table doubling
// underneath them. Contract, asserted exactly:
//   (1) every key is present afterwards, and
//   (2) insert() returned true exactly N times -- one absent->present transition
//       per key, one winner per transition.
// This is the test whose total count first exposed the double-winner defect
// (4001 trues for 4000 keys). It used to assert only EXPECT_LE(true_count, N) on
// the theory that the boolean was "best-effort and may occasionally under-count"
// under concurrent resizes; there is no such licence in the contract, and an
// under-count is unreachable anyway (a thread that scans and misses either wins
// its publishing CAS, or loses it and retries until it wins or finds the winner's
// node, so a winner always exists).
// ExactlyOneWinnerPerKey below strengthens this further: a total count alone
// cannot tell "one winner each" from "key A twice and key B never".
TEST(ConcurrentHashSetTest, ConcurrentSameKeyMembership) {
    for (int rep = 0; rep < 10; ++rep) {
        ConcurrentResizableHashSet<int> set(4);
        const int T = 16, N = 4000;
        std::atomic<int> true_count{0};
        run_threads(T, [&set, &true_count](int) {
            int local = 0;
            for (int k = 0; k < N; ++k) {
                if (set.insert(scramble(k))) ++local;
            }
            true_count += local;
        });

        EXPECT_EQ(true_count.load(), N) << "insert() did not report exactly one winner per key, rep " << rep;
        for (int k = 0; k < N; ++k) {
            EXPECT_TRUE(set.contains(scramble(k))) << "missing key " << k << " in rep " << rep;
        }
    } // repetitions
} // ConcurrentSameKeyMembership

// Forces heavy collisions into a handful of buckets to stress chain splitting.
struct ModHash {
    size_t operator()(int x) const { return static_cast<size_t>(x%8); }
};

TEST(ConcurrentHashSetTest, CollisionHeavyChains) {
    ConcurrentResizableHashSet<int, false, ModHash> set(4);
    const int N = 5000;
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(i));
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.contains(i));
    EXPECT_FALSE(set.contains(N + 1));
}

// Non-trivial key type to confirm the container is not hard-wired to integers.
TEST(ConcurrentHashSetTest, StringKeys) {
    ConcurrentResizableHashSet<std::string> set(4);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(set.insert("key_" + std::to_string(i)));
    }
    EXPECT_FALSE(set.insert("key_500"));
    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(set.contains("key_" + std::to_string(i)));
    }
    EXPECT_FALSE(set.contains("absent"));
} // StringKeys

// A writer continuously inserts a known range while readers poll; any key the writer
// has finished inserting must be observable, and readers must never see false positives.
// Scrambled keys: the writer drives ~12 doublings here, and with consecutive keys the
// readers would be polling keys that stopped moving long before the last of them.
TEST(ConcurrentHashSetTest, ConcurrentReadersWriters) {
    ConcurrentResizableHashSet<int> set(4);
    const int N = 20000;
    std::atomic<bool> done{false};
    std::atomic<int> progress{0};

    std::thread writer([&]() {
        for (int i = 0; i < N; ++i) {
            set.insert(scramble(i));
            progress.store(i, std::memory_order_release);
        }
        done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                int p = progress.load(std::memory_order_acquire);
                if (p > 0) {
                    // A key strictly below the writer's progress was fully inserted.
                    EXPECT_TRUE(set.contains(scramble(p - 1)));
                }
                EXPECT_FALSE(set.contains(scramble(N + 12345)));  // never inserted
            } // poll until the writer is done
        });
    } // spawn readers

    writer.join();
    for (std::thread& t : readers) t.join();
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.contains(scramble(i)));
} // ConcurrentReadersWriters

// Stresses the cooperative lazy split: many threads simultaneously touch the
// freshly-created UNINITIALIZED buckets, so they race inside split_bucket() and
// exactly one publishing CAS per bucket must win while the losers' subchains are
// abandoned harmlessly. The keys are constructed so their low 3 bits select
// buckets 4..7 once the table is size 8, which requires a hash whose low bits are
// the key's low bits -- hence the explicit CollisionHash rather than a reliance on
// std::hash<int> being the identity. The keys' HIGH bits are spread over 80000, so
// they keep moving at the later doublings too.
TEST(ConcurrentHashSetTest, SplitContention) {
    // Initial size 4
    ConcurrentResizableHashSet<int, false, CollisionHash> set(4);

    // Insert 8 elements to approach resize threshold (4 * 2 = 8).
    for (int i = 0; i < 8; ++i) {
        set.insert(100 + i);
    }

    // Insert 1 more element: arena occupancy now exceeds ts*2, so this insert
    // doubles the table to 8. Buckets 4..7 are published UNINITIALIZED (their
    // encoded value UNINITIALIZED, distinct from a real index) and will be split
    // lazily by whichever thread below touches them first.
    set.insert(200);

    // 8 threads aggressively inserting elements that hash to 4, 5, 6, 7
    run_threads(8, [&set](int i) {
        for (int k = 0; k < 500; ++k) {
            // Keys that map to bucket 4, 5, 6, 7 when the table size is 8.
            set.insert(4 + 8*k + i*10000);
            set.insert(5 + 8*k + i*10000);
            set.insert(6 + 8*k + i*10000);
            set.insert(7 + 8*k + i*10000);
        } // key loop
    });

    // Verify
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 500; ++k) {
            EXPECT_TRUE(set.contains(4 + 8*k + i*10000));
            EXPECT_TRUE(set.contains(5 + 8*k + i*10000));
            EXPECT_TRUE(set.contains(6 + 8*k + i*10000));
            EXPECT_TRUE(set.contains(7 + 8*k + i*10000));
        } // key loop
    } // membership sweep
} // SplitContention

// Exercises tombstone erase end-to-end: single-threaded correctness (erase makes
// contains() return false; a second erase of the same key returns false; erase
// does not disturb other keys; a key can be re-inserted after erase), then a
// concurrent phase where each thread erases its own disjoint key range so the
// MARK_BIT CAS and the erase/contains interleavings are stressed without two
// threads ever contending for the same key. AllowDelete=true compiles erase().
// The concurrent phase erases keys that are present and that no other thread
// touches, so EVERY erase() there must return true -- the old version discarded
// those return values, which threw away an exact oracle for free.
// NOTE ON SCOPE: the concurrent phase performs no inserts, so the table does not
// double while it runs. It is a MARK_BIT contention test, not a stale-geometry
// test; ExactlyOneEraseWinnerPerKeyDuringGrowth below covers erase under resize.
TEST(ConcurrentHashSetTest, TombstoneErase) {
    ConcurrentResizableHashSet<int, true> set(4);

    // Basic erase
    EXPECT_TRUE(set.insert(10));
    EXPECT_TRUE(set.insert(20));
    EXPECT_TRUE(set.contains(10));
    EXPECT_TRUE(set.erase(10));
    EXPECT_FALSE(set.contains(10));
    EXPECT_FALSE(set.erase(10)); // Already erased
    EXPECT_TRUE(set.contains(20));

    // Erase and insert again
    EXPECT_TRUE(set.insert(10));
    EXPECT_TRUE(set.contains(10));

    // Concurrent erase of disjoint, pre-populated ranges.
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 1000; ++k) {
            set.insert(scramble(i*10000 + k));
        }
    } // pre-population

    std::vector<size_t> not_erased(8, 0);
    run_threads(8, [&set, &not_erased](int i) {
        size_t bad = 0;
        for (int k = 0; k < 1000; ++k) {
            // The key is present and private to this thread: the contract makes
            // this call the present->absent transition, so it must return true.
            bad += !set.erase(scramble(i*10000 + k));
        }
        not_erased[i] = bad;
    });

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(not_erased[i], 0u) << "erase() of a present, uncontended key returned false, thread " << i;
        for (int k = 0; k < 1000; ++k) {
            EXPECT_FALSE(set.contains(scramble(i*10000 + k)));
        }
    } // membership sweep
} // TombstoneErase

// Meant to be run under ThreadSanitizer: it validates that concurrent
// cooperative splits are DATA-RACE free (many threads racing inside
// split_bucket() on the same UNINITIALIZED buckets, one CAS winner, losers'
// subchains abandoned). No deletes here (AllowDelete=false), so it exercises the
// split/publish paths only. The final EXPECT_EQ also checks that every attempted
// key is present -- membership must be exact across every resize.
TEST(ConcurrentHashSetTest, CooperativeSplitTSAN) {
    ConcurrentResizableHashSet<int, false, CollisionHash> set(4);

    // Pre-populate to trigger exactly one resize (N=4 -> N=8)
    for (int i = 0; i < 9; ++i) {
        set.insert(i*16);
    }

    constexpr int NUM_THREADS = 8;
    constexpr int INSERTS_PER_THREAD = 1000;

    run_threads(NUM_THREADS, [&set](int t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            int key = 5 + ((i*NUM_THREADS + t)*8);
            set.insert(key);
        } // insert loop
    });

    int success_count = 0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            int key = 5 + ((i*NUM_THREADS + t)*8);
            if (set.contains(key)) {
                success_count++;
            }
        }
    } // membership sweep
    EXPECT_EQ(success_count, NUM_THREADS*INSERTS_PER_THREAD);
} // CooperativeSplitTSAN

// TSAN stress test specifically for erase(). Unlike TombstoneErase this phase DOES
// insert, so the arena keeps growing and the table keeps doubling underneath the
// erasers. Every key is private to one thread, so every return value in the
// sequence below is fully determined by the contract and is asserted exactly; the
// old version asserted only the middle contains() and discarded four booleans.
// Keys are scrambled because the pre-population alone takes the table past the
// consecutive key range, after which unscrambled keys would stop moving.
TEST(ConcurrentHashSetTest, EraseStressTSAN) {
    ConcurrentResizableHashSet<int, true> set(4);
    constexpr int NUM_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 2000;

    // First, populate the set heavily so we force resizes and splits.
    for (int i = 0; i < NUM_THREADS*OPERATIONS_PER_THREAD; ++i) {
        set.insert(scramble(i));
    }

    // Concurrently read, insert, and erase to stress the MARK_BIT CAS, the freeze
    // CAS a concurrent split performs on the same link word, and the DCLP resize.
    std::vector<size_t> bad(NUM_THREADS, 0);
    run_threads(NUM_THREADS, [&set, &bad](int t) {
        size_t b = 0;
        for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
            int key = scramble(t*OPERATIONS_PER_THREAD + i);

            b += !set.erase(key);      // present -> absent: exactly this call
            b += set.contains(key);    // its own eraser must not still see it
            b += !set.insert(key);     // absent -> present: exactly this call
            b += !set.erase(key);      // present -> absent again
        } // operation loop
        bad[t] = b;
    });

    for (int t = 0; t < NUM_THREADS; ++t) {
        EXPECT_EQ(bad[t], 0u) << "determined erase/insert/contains sequence broke on thread " << t;
    }

    // At the end, all keys should be erased
    for (int i = 0; i < NUM_THREADS*OPERATIONS_PER_THREAD; ++i) {
        EXPECT_FALSE(set.contains(scramble(i)));
    }
} // EraseStressTSAN

// Targets the lost-key-on-resize race. Run 10000 times to make the narrow window
// likely.
//
// The race: key 4 hashes to bucket 0 at size 4 (4 & 3 == 0) but to bucket 4 at
// size 8 (4 & 7 == 4). t1 inserts 4 (publishing it into bucket 0 under size 4)
// exactly while t2 doubles the table to 8. If t2's split of the new bucket 4
// snapshots bucket 0's chain in the instant BEFORE t1's node is linked, the copy
// pass misses key 4 -- and key 4 would be stranded in a bucket no reader consults
// at size 8. What closes the window is that the split SEALS bucket 0's head
// (raising its level by CAS) before snapshotting it: seal and publication are
// read-modify-writes of the same word, so either t1's node is in the snapshot or
// t1's publishing CAS fails and it retries in the new geometry. contains(4) must
// therefore hold afterwards, and insert(4) must have reported true exactly once.
TEST(ConcurrentHashSetTest, LostUpdateOnResize) {
    for (int rep = 0; rep < 10000; ++rep) {
        ConcurrentResizableHashSet<int, false, CollisionHash> set(4);
        set.insert(0); // initialize bucket 0

        std::atomic<bool> start{false};

        std::thread t1([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            // Insert 4 which hashes to 0 initially (4 & 3 == 0)
            set.insert(4);
        });

        std::thread t2([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            // Force resize to 8: keys 16,32,... all hash to bucket 0, so nine of
            // them push arena occupancy past the doubling threshold.
            for (int i = 1; i <= 9; ++i) {
                set.insert(i*16);
            }
        });

        start.store(true, std::memory_order_release);
        t1.join();
        t2.join();

        // 4 should be in the set
        ASSERT_TRUE(set.contains(4)) << "Lost update in rep " << rep;
    } // repetitions
} // LostUpdateOnResize

// Degenerate hash: every key maps to bucket 0. This funnels all inserts through
// a single bucket head, maximizing compare_exchange contention -- precisely the
// condition under which the old "allocate a fresh node on every CAS retry" bug
// leaked a node per failed attempt. It makes the leak-detection test below
// sensitive by construction.
struct ConstantHash {
    size_t operator()(int) const { return 0; }
};

TEST(ConcurrentHashSetTest, InsertContention_NoMemoryLeak) {
    ConcurrentResizableHashSet<int, false, ConstantHash> set(4);
    const int T = 8;
    const int N = 1000;

    run_threads(T, [&set](int t) {
        for (int i = 0; i < N; ++i) {
            set.insert(t*N + i);
        }
    });

    // Total inserted is T * N = 8000.
    // If it doesn't leak, we expect around 8000 nodes (+ small bound for abandoned split subchains).
    // If the CAS failure memory leak is present, this will be significantly higher.
    size_t node_count = set.get_internal_node_count();
    EXPECT_LE(node_count, static_cast<size_t>((T*N) + 100)) << "Memory leak detected! Expected ~8000 nodes, got " << node_count;
} // InsertContention_NoMemoryLeak

// ===========================================================================
// Return-value exactness under concurrent resizes. The tests below are the
// regression guards for the four defects demonstrated against the pre-seal
// header (notes/insert_race.md): double-true insert, erase undone by a stale
// inserter, double-true erase, and a key lost through a doubling.
// ===========================================================================

// PER-KEY exactly-one-winner for insert(), across many doublings.
//
// Why the total count of ConcurrentSameKeyMembership is not enough: a repetition
// in which key A is inserted twice and key B zero times has the correct TOTAL and
// passes an equality assertion on it. Only per-key accounting distinguishes "every
// key had exactly one winner" from "the sum happened to come out right".
//
// The weak spot it targets: insert() decides by a CAS on ONE bucket head, but the
// bucket a key belongs to changes when the table doubles. If a thread working with
// a stale table size can still publish into the OLD bucket while another thread
// legitimately publishes the same key into the NEW one, the two winners CAS two
// DIFFERENT words and the single-CAS interlock never engages -- two callers both
// get true for one absent->present transition. The seal level in the bucket head
// is what forces both publications through one word; this test is what notices if
// it stops doing so.
//
// Tallies are per thread, so no thread writes a location another reads; they are
// summed after the join. The two AllowDelete flavors alternate because the freeze
// step of split_bucket() is compiled out when AllowDelete is false, which changes
// the split's timing and the code under test.
template <typename SetT>
static void one_winner_per_key_body(int T, int N, int rep, const char* flavor) {
    SetT set(4);
    std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(N, 0));
    run_threads(T, [&set, &won, N](int t) {
        for (int k = 0; k < N; ++k) {
            if (set.insert(scramble(k))) won[t][k] = 1;
        }
    });
    int over = 0, under = 0, missing = 0;
    for (int k = 0; k < N; ++k) {
        int c = 0;
        for (int t = 0; t < T; ++t) c += won[t][k];
        if (c > 1) ++over;
        if (c < 1) ++under;
        if (!set.contains(scramble(k))) ++missing;
    } // per-key check
    EXPECT_EQ(over, 0) << over << " key(s) had MORE than one successful insert(), rep " << rep << " (" << flavor << ")";
    EXPECT_EQ(under, 0) << under << " key(s) had NO successful insert(), rep " << rep << " (" << flavor << ")";
    EXPECT_EQ(missing, 0) << missing << " key(s) absent after all inserts, rep " << rep << " (" << flavor << ")";
} // one_winner_per_key_body()

TEST(ConcurrentHashSetTest, ExactlyOneWinnerPerKey) {
    const int T = 16, N = 400;
    for (int rep = 0; rep < 60; ++rep) {
        if (rep & 1) {
            one_winner_per_key_body<ConcurrentResizableHashSet<int, true>>(T, N, rep, "AllowDelete=true");
        } else {
            one_winner_per_key_body<ConcurrentResizableHashSet<int, false>>(T, N, rep, "AllowDelete=false");
        }
    } // repetitions
} // ExactlyOneWinnerPerKey

// TARGETED regression test for the double-winner interleaving: it builds the
// geometry by hand instead of waiting for it to occur by chance, so it reproduces
// in milliseconds on any machine and in any build. (The statistical tests above
// find the same defect, but their hit rate swings by two orders of magnitude with
// the build -- measured ~50% of repetitions at -O0+TSan and ~0.1% at -O3 without
// a sanitizer -- so on some machines and some builds they can look green.)
//
// Geometry, with CollisionHash so the bucket arithmetic is exact: key 4 is in
// bucket 0 at table size 4 (4 & 3 == 0) and in bucket 4 at size 8 (4 & 7 == 4),
// so the doubling MOVES it. Eight filler keys, all multiples of 8, sit in bucket 0
// and bring arena occupancy to exactly ts*2, so the next allocation doubles the
// table. Key 1 lives in bucket 1, so the resize-triggering insert never disturbs
// bucket 0's chain.
//
// Threads are persistent and meet at a barrier twice per attempt (once to start
// the attempt on a freshly built set, once to hand the result back): spawning
// three threads per attempt would cost more than the attempt itself and would put
// the thread-creation latency, not the race, in the window.
TEST(ConcurrentHashSetTest, NoDoubleWinnerAcrossResize) {
    using Set = ConcurrentResizableHashSet<int, false, CollisionHash>;
    const int KEY = 4;
    const int ATTEMPTS = 20000;
    std::unique_ptr<Set> set;
    std::atomic<int> winners{0};
    std::barrier round(4);   // three workers plus this thread

    std::vector<std::thread> workers;
    for (int role = 0; role < 3; ++role) {
        workers.emplace_back([&set, &winners, &round, KEY, role]() {
            for (int a = 0; a < ATTEMPTS; ++a) {
                round.arrive_and_wait();     // the set for this attempt is built
                if (role == 1) {
                    set->insert(1);          // the 9th node: crosses the doubling threshold
                } else if (set->insert(KEY)) {
                    winners.fetch_add(1, std::memory_order_relaxed);
                }
                round.arrive_and_wait();     // results are safe to read
            } // attempt loop
        });
    } // spawn workers

    int bad = 0, first_bad = -1, missing = 0;
    for (int a = 0; a < ATTEMPTS; ++a) {
        set = std::make_unique<Set>(4);
        for (int i = 0; i < 8; ++i) set->insert(i*8);
        winners.store(0, std::memory_order_relaxed);
        round.arrive_and_wait();
        round.arrive_and_wait();
        if (winners.load(std::memory_order_relaxed) != 1) {
            ++bad;
            if (first_bad < 0) first_bad = a;
        }
        if (!set->contains(KEY)) ++missing;
    } // attempt loop
    for (std::thread& th : workers) th.join();

    EXPECT_EQ(bad, 0) << "insert(" << KEY << ") did not report exactly one winner in " << bad
                      << " of " << ATTEMPTS << " attempts (first: attempt " << first_bad << ")";
    EXPECT_EQ(missing, 0) << "key " << KEY << " lost by the doubling in " << missing << " attempts";
} // NoDoubleWinnerAcrossResize

// PER-KEY exactly-one-winner for erase(), with the table doubling while the
// erasers run. This is the erase-side dual of ExactlyOneWinnerPerKey and the
// direct guard for the double-true erase and the erase-undone defects.
//
// The weak spot: erase() decides by a CAS on a NODE LINK, but a split COPIES a
// node into the child bucket rather than moving it, so for a while two nodes exist
// for the key. If a thread with a stale table size can mark the parent's node
// while the live copy in the child survives, that thread gets true, the key is
// still present, and the next eraser gets true as well -- two winners for one
// present->absent transition, and an erase that did not erase. The FROZEN bit is
// what forces the mark and the copy through one word; this test notices if it
// stops doing so.
//
// Shape, and why it is sized the way it is. Each round is a TINY, fresh set: 4
// buckets and 16 pre-inserted shared keys, so the table is still at 8 buckets when
// the threads are released. Every thread erases every shared key, so both
// contention modes occur -- mark versus mark (two erasers on one link) and mark
// versus freeze (an eraser and a splitter on one link) -- and the threads start at
// staggered positions in the key order so that a thread which loaded table_size_
// before a doubling is still walking when another thread has moved on. Between
// erases each thread inserts private filler keys, and those drive about six
// doublings while the erases are in flight.
// The ratio that decides whether this test can see anything is DOUBLINGS PER
// ERASE, because the defect needs a split to copy a node inside some eraser's
// load-table_size_-to-mark-CAS window. Doublings are logarithmic in arena size, so
// a big set is the worst possible shape: an earlier version of this test used 128
// shared keys in a set already grown to 64 buckets and got 0.03 doublings per
// erase and ONE violation per run against the no-freeze mutant -- a hair from
// passing on a broken header. Sixteen keys in a 4-bucket set give ~0.4 doublings
// per erase, 400 cheap rounds instead of 30 expensive ones, and dozens of
// violations per run. Prefer many tiny sets to one big one for anything that has
// to race with a resize.
struct StallingKey {
    int v;
    StallingKey(int x = 0) : v(x) {}
    // Equality that briefly stalls on every 8th MATCHING comparison made by the
    // calling thread. WHY this is here: erase() loads the node's link, THEN
    // compares the key, THEN CASes MARK onto that same link. The comparison sits
    // inside the window in which a concurrent split can FREEZE the link out from
    // under the eraser -- with plain ints the window is a few nanoseconds wide and
    // the mark-versus-freeze race is correspondingly rare. Measured against the
    // no-freeze mutant with everything else held fixed: with plain int keys, 6-12
    // violations per ASan run; with this key type, 24-36. (Under TSan both are
    // ~60, because TSan's own instrumentation already stretches the window.) The
    // extra ASan margin is worth the 0.6 s it costs, because ASan is the build in
    // which this test is closest to reporting a broken header as green. It changes
    // nothing about WHAT is tested: operator== still answers correctly, no header
    // code is touched, the oracle is unchanged, and the mutant fails the test
    // without the stall too -- just with less room to spare.
    // Only matching comparisons stall, and only every 8th, so traversals of other
    // keys run at full speed and the filler inserts still drive the doublings.
    bool operator==(const StallingKey& o) const {
        if (v != o.v) return false;
        thread_local unsigned matches = 0;
        if ((++matches & 7) == 0) {
            const std::chrono::steady_clock::time_point until =
                std::chrono::steady_clock::now() + std::chrono::microseconds(20);
            while (std::chrono::steady_clock::now() < until) {}
        } // stall on every 8th match
        return true;
    } // StallingKey::operator==()
}; // struct StallingKey

struct StallingKeyHash {
    size_t operator()(const StallingKey& k) const { return std::hash<int>{}(k.v); }
};

TEST(ConcurrentHashSetTest, ExactlyOneEraseWinnerPerKeyDuringGrowth) {
    using Set = ConcurrentResizableHashSet<StallingKey, true, StallingKeyHash>;
    const int T = 8, K = 16, ROUNDS = 400, FILL = 64;
    int over = 0, under = 0, still_present = 0, filler_dup = 0, filler_missing = 0;

    for (int r = 0; r < ROUNDS; ++r) {
        Set set(4);
        for (int k = 0; k < K; ++k) set.insert(scramble(k));

        std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(K, 0));
        std::vector<int> dup(T, 0);
        run_threads(T, [&set, &won, &dup](int t) {
            for (int i = 0; i < K; ++i) {
                const int k = (i + t*K/T)%K;     // staggered start per thread
                if (set.erase(scramble(k))) ++won[t][k];
                // Filler inserts, spread evenly through the erases: these are what
                // grow the arena and double the table mid-flight. Each key belongs
                // to one thread, so a false return is a defect in itself.
                for (int f = i*FILL/K; f < (i + 1)*FILL/K; ++f) {
                    if (!set.insert(scramble(K + t*FILL + f))) ++dup[t];
                }
            } // key loop
        });

        for (int k = 0; k < K; ++k) {
            int c = 0;
            for (int t = 0; t < T; ++t) c += won[t][k];
            if (c > 1) ++over;
            if (c < 1) ++under;
            if (set.contains(scramble(k))) ++still_present;
        } // per-key check
        for (int t = 0; t < T; ++t) {
            filler_dup += dup[t];
            for (int f = 0; f < FILL; ++f) {
                if (!set.contains(scramble(K + t*FILL + f))) ++filler_missing;
            }
        } // filler check
    } // rounds

    EXPECT_EQ(over, 0) << over << " key(s) had MORE than one successful erase()";
    EXPECT_EQ(under, 0) << under << " key(s) had NO successful erase()";
    EXPECT_EQ(still_present, 0) << still_present << " erased key(s) still present at the end of the round";
    EXPECT_EQ(filler_dup, 0) << filler_dup << " filler insert(s) of an uncontended new key returned false";
    EXPECT_EQ(filler_missing, 0) << filler_missing << " filler key(s) lost by a doubling";
} // ExactlyOneEraseWinnerPerKeyDuringGrowth

// Mixed insert/erase/contains churn on a small scrambled key range while the
// table grows, with per-key accounting that is exact after the join.
//
// The oracle, and why it is sound without knowing the interleaving: a successful
// insert is an absent->present transition and a successful erase a present->absent
// one, so in ANY linearization of the run they alternate per key, starting with an
// insert. Therefore, per key, with every thread joined:
//     0 <= #insert_true - #erase_true <= 1      and
//     final contains(k) == (#insert_true - #erase_true == 1)
// This single invariant catches a double-true insert (difference reaches 2), a
// double-true erase (difference goes negative), a lost insert, a lost erase and a
// resurrection (membership disagrees with the difference), with no dependence on
// which thread did what when. The key range is deliberately tiny (48) so that
// every key is contended by all eight threads, while the churn still allocates
// tens of thousands of nodes and drives ~10 doublings.
TEST(ConcurrentHashSetTest, InsertEraseChurnPerKeyAccounting) {
    using Set = ConcurrentResizableHashSet<int, true>;
    const int T = 8, R = 48, REPS = 60, OPS = 1600;

    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4);
        std::vector<std::vector<int>> ins(T, std::vector<int>(R, 0));
        std::vector<std::vector<int>> ers(T, std::vector<int>(R, 0));

        run_threads(T, [&set, &ins, &ers, rep](int t) {
            std::minstd_rand rng(static_cast<unsigned>(rep*977 + t + 1));
            for (int op = 0; op < OPS; ++op) {
                const unsigned x = rng();
                const int i = static_cast<int>((x >> 4)%static_cast<unsigned>(R));
                const int k = scramble(i + 1);
                switch (x & 3u) {
                    case 0:
                    case 1:  // insert twice as often as erase, so the set stays populated
                        if (set.insert(k)) ++ins[t][i];
                        break;
                    case 2:
                        if (set.erase(k)) ++ers[t][i];
                        break;
                    default:
                        set.contains(k);
                        break;
                } // operation kind
            } // operation loop
        });

        for (int i = 0; i < R; ++i) {
            long d = 0;
            for (int t = 0; t < T; ++t) {
                d += ins[t][i] - ers[t][i];
            }
            EXPECT_LE(d, 1) << "key index " << i << ": " << d << " more successful inserts than erases (rep "
                            << rep << "); the transitions must alternate, so at most one";
            EXPECT_GE(d, 0) << "key index " << i << ": more successful erases than inserts (rep " << rep << ")";
            EXPECT_EQ(set.contains(scramble(i + 1)), d == 1)
                << "key index " << i << ": final membership disagrees with the insert/erase tally " << d
                << " (rep " << rep << ")";
        } // per-key check
    } // repetitions
} // InsertEraseChurnPerKeyAccounting

// An erase must not be undone by a resize, ever.
//
// The weak spot: a split COPIES live nodes forward and never unlinks the originals,
// so a key can have a stale, still-live node sitting in an ancestor bucket. If an
// erase marks only the authoritative copy and some later doubling then copies the
// stale node into a new bucket, the erased key comes back from the dead -- with no
// insert() call anywhere. Two things must hold for that to be impossible: no stale
// live node may exist for a key that was published under a sealed head, and any
// node a split copies must be frozen first so a concurrent erase cannot leave a
// live copy behind.
//
// Shape and oracle:
//   Phase 1 -- T threads race to insert the same K keys while filler inserts double
//     the table. This is the phase that can strand a stale copy: it is exactly the
//     double-winner window. Per-key exactly-one-winner is checked here too, for free.
//   Phase 2 -- single-threaded: erase each key (must return true, it is present and
//     uncontended), erase it again (must return false), and confirm it is absent.
//     After this phase completes, NOTHING in this test inserts these keys again.
//   Phase 3 -- T threads hammer inserts of OTHER keys, enough to drive about five
//     more doublings (so every bucket that could hold a stale copy gets split), while
//     a watcher thread polls the erased keys continuously. Because phase 2 has been
//     joined and no thread inserts an erased key, contains() must be false for every
//     one of them at every instant -- during phase 3 and after it. That is an exact
//     oracle, not a probabilistic one.
//
// HONESTY ABOUT WHAT THIS TEST HAS BEEN SHOWN TO CATCH. The resurrection oracle
// (phase 3) is exact, but it has never been made to FIRE. Run against the pre-seal
// stock header and against both mutants (seal removed, freeze removed), the
// resurrection counters stayed at zero in every run; what fails on those headers is
// the phase-1 exactly-one-winner check. That is consistent with the analysis in
// notes/insert_race.md: the stale node those headers strand in an old bucket j can
// only be copied forward when bucket j's CHILD is split, and by the time the strand
// happens that child has long since been published -- so the stranded node is dead
// weight and never resurrects. In other words this test currently guards a property
// that no known defect violates. It is kept because the property is load-bearing
// (the whole copy-never-unlink design rests on it), the check is exact and costs
// half a second, and a future change to split_bucket() -- relinking, re-splitting a
// published bucket, copying an unfrozen node -- would break it first. Do not count
// it as a detector for the four known defects; NoDoubleWinnerAcrossResize and
// ExactlyOneEraseWinnerPerKeyDuringGrowth are the detectors.
TEST(ConcurrentHashSetTest, EraseNotUndoneByResize) {
    using Set = ConcurrentResizableHashSet<int, true>;
    const int T = 8, K = 64, REPS = 12, FILL = 64, HAMMER = 2000;
    int over = 0, under = 0, erase_wrong = 0, resurrected_final = 0;
    size_t resurrected_live = 0;

    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4);

        // Phase 1: contended inserts of the shared keys during growth.
        std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(K, 0));
        run_threads(T, [&set, &won](int t) {
            for (int i = 0; i < K; ++i) {
                const int k = (i + t*K/T)%K;
                if (set.insert(scramble(k))) won[t][k] = 1;
                for (int f = i*FILL/K; f < (i + 1)*FILL/K; ++f) set.insert(scramble(K + t*FILL + f));
            } // key loop
        });
        for (int k = 0; k < K; ++k) {
            int c = 0;
            for (int t = 0; t < T; ++t) c += won[t][k];
            if (c > 1) ++over;
            if (c < 1) ++under;
        } // per-key check

        // Phase 2: erase them all, single-threaded, with fully determined results.
        for (int k = 0; k < K; ++k) {
            erase_wrong += !set.erase(scramble(k));   // present -> absent
            erase_wrong += set.erase(scramble(k));    // already absent
            erase_wrong += set.contains(scramble(k));
        } // erase sweep

        // Phase 3: growth of OTHER keys, with the erased keys under observation.
        std::atomic<bool> stop{false};
        std::atomic<size_t> seen{0};
        std::thread watcher([&set, &stop, &seen]() {
            while (!stop.load(std::memory_order_acquire)) {
                for (int k = 0; k < K; ++k) {
                    if (set.contains(scramble(k))) seen.fetch_add(1, std::memory_order_relaxed);
                }
            } // poll until the hammers are done
        });
        run_threads(T, [&set](int t) {
            for (int h = 0; h < HAMMER; ++h) set.insert(scramble(K + T*FILL + t*HAMMER + h));
        });
        stop.store(true, std::memory_order_release);
        watcher.join();
        resurrected_live += seen.load();

        for (int k = 0; k < K; ++k) {
            if (set.contains(scramble(k))) ++resurrected_final;
        }
    } // repetitions

    EXPECT_EQ(over, 0) << over << " key(s) had more than one successful insert() in phase 1";
    EXPECT_EQ(under, 0) << under << " key(s) had no successful insert() in phase 1";
    EXPECT_EQ(erase_wrong, 0) << erase_wrong << " determined erase()/contains() result(s) wrong in phase 2";
    EXPECT_EQ(resurrected_live, 0u) << "an erased key became visible again " << resurrected_live
                                    << " time(s) while other keys were being inserted";
    EXPECT_EQ(resurrected_final, 0) << resurrected_final << " erased key(s) present after the growth phase";
} // EraseNotUndoneByResize

// Immediate visibility to the operation's own thread, during growth.
//
// Every key here is private to one thread, so every return value in the sequence
// is fully determined by the contract no matter what the other threads do:
//   insert 1, contains 1, insert 0, erase 1, contains 0, erase 0, insert 1, contains 1.
// The two contains() calls that follow the thread's own successful insert and its
// own successful erase are the point: the shipped suite only ever swept contains()
// after every thread had joined, by which time any self-repair the implementation
// performs has long since run. This checks what a caller actually relies on -- the
// instant insert(k) returns true, k is visible to that caller; the instant erase(k)
// returns true, it is not.
// The other threads' inserts keep the table doubling throughout, so each of these
// sequences straddles resizes; keys are scrambled so they keep moving across them.
TEST(ConcurrentHashSetTest, ThreadPrivateKeySequenceDuringGrowth) {
    using Set = ConcurrentResizableHashSet<int, true>;
    const int T = 8, PER = 1500;
    Set set(4);
    std::vector<size_t> bad(T, 0);

    run_threads(T, [&set, &bad](int t) {
        size_t b = 0;
        for (int i = 0; i < PER; ++i) {
            const int k = scramble(t*PER + i);
            b += !set.insert(k);     // absent -> present
            b += !set.contains(k);   // visible to its own inserter, immediately
            b += set.insert(k);      // already present
            b += !set.erase(k);      // present -> absent
            b += set.contains(k);    // invisible to its own eraser, immediately
            b += set.erase(k);       // already absent
            b += !set.insert(k);     // absent -> present again
            b += !set.contains(k);
        } // key loop
        bad[t] = b;
    });

    for (int t = 0; t < T; ++t) {
        EXPECT_EQ(bad[t], 0u)
            << bad[t] << " determined result(s) wrong on thread " << t
            << " (its keys are private; every return value in the sequence is fixed by the contract)";
    }
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < PER; ++i) {
            EXPECT_TRUE(set.contains(scramble(t*PER + i))) << "key lost, thread " << t << " index " << i;
        }
    } // membership sweep
} // ThreadPrivateKeySequenceDuringGrowth
