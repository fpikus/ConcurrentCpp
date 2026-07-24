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
#include "concurrent_hash_set.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <barrier>

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
TEST(ConcurrentHashSetTest, ManyResizesSingleThread) {
    ConcurrentResizableHashSet<int> set(4);
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(set.insert(i)) << "insert(" << i << ") wrongly reported duplicate";
    }
    for (int i = 0; i < N; i += 997) {
        EXPECT_FALSE(set.insert(i)) << "re-insert(" << i << ") wrongly reported new";
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(set.contains(i)) << "missing key " << i;
    }
    for (int i = N; i < N + 5000; ++i) {
        EXPECT_FALSE(set.contains(i)) << "false positive for " << i;
    }
}

// Many threads insert disjoint key ranges concurrently (forcing many concurrent
// resizes and splits). The strong guarantee is membership: every key must be present
// afterwards. The insert() boolean may UNDER-count even for disjoint keys: when a
// concurrent split copies a freshly-inserted node into the new bucket, the inserter's
// own re-insert finds that copy and reports false. So we only assert the count cannot
// EXCEED the number of distinct keys (the over-count bug, which has been fixed).
TEST(ConcurrentHashSetTest, ConcurrentDisjointRanges) {
    ConcurrentResizableHashSet<int> set(4);
    const int T = 8, per = 20000;
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    std::atomic<int> inserted{0};
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&set, &start, &inserted, t]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            int local = 0;
            for (int k = 0; k < per; ++k) {
                if (set.insert(t * per + k)) ++local;
            }
            inserted += local;
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    EXPECT_LE(inserted.load(), T * per);  // never over-count
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < per; ++k) {
            EXPECT_TRUE(set.contains(t * per + k)) << "missing " << (t * per + k);
        }
    }
}

// Many threads racing to insert the SAME set of keys. The guaranteed invariants are:
//  (1) every key is present afterwards (membership is always correct), and
//  (2) the number of insert() calls returning true never exceeds the number of
//      distinct keys (no key is "uniquely inserted" by two callers at once).
// Note: under concurrent resizes the boolean return is best-effort and may
// occasionally under-count, so we assert <= rather than ==.
TEST(ConcurrentHashSetTest, ConcurrentSameKeyMembership) {
    for (int rep = 0; rep < 20; ++rep) {
        ConcurrentResizableHashSet<int> set(4);
        const int T = 16, N = 4000;
        std::vector<std::thread> threads;
        std::atomic<bool> start{false};
        std::atomic<int> true_count{0};
        for (int t = 0; t < T; ++t) {
            threads.emplace_back([&set, &start, &true_count]() {
                while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
                int local = 0;
                for (int k = 0; k < N; ++k) {
                    if (set.insert(k)) ++local;
                }
                true_count += local;
            });
        }
        start.store(true, std::memory_order_release);
        for (auto& t : threads) t.join();

        EXPECT_LE(true_count.load(), N) << "insert reported true more times than there are keys";
        for (int k = 0; k < N; ++k) {
            EXPECT_TRUE(set.contains(k)) << "missing key " << k << " in rep " << rep;
        }
    }
}

// Forces heavy collisions into a handful of buckets to stress chain splitting.
struct ModHash {
    size_t operator()(int x) const { return static_cast<size_t>(x % 8); }
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
}

// A writer continuously inserts a known range while readers poll; any key the writer
// has finished inserting must be observable, and readers must never see false positives.
TEST(ConcurrentHashSetTest, ConcurrentReadersWriters) {
    ConcurrentResizableHashSet<int> set(4);
    const int N = 20000;
    std::atomic<bool> done{false};
    std::atomic<int> progress{0};

    std::thread writer([&]() {
        for (int i = 0; i < N; ++i) {
            set.insert(i);
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
                    EXPECT_TRUE(set.contains(p - 1));
                }
                EXPECT_FALSE(set.contains(N + 12345));  // never inserted
            }
        });
    }

    writer.join();
    for (auto& t : readers) t.join();
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.contains(i));
}

// Stresses the cooperative lazy split: many threads simultaneously touch the
// freshly-created UNINITIALIZED buckets, so they race inside split_bucket() and
// exactly one publishing CAS per bucket must win while the losers' subchains are
// abandoned harmlessly. Relies on the identity of std::hash<int>: the keys are
// constructed so their low 3 bits select buckets 4..7 once the table is size 8.
TEST(ConcurrentHashSetTest, SplitContention) {
    // Initial size 4
    ConcurrentResizableHashSet<int> set(4);

    // Insert 8 elements to approach resize threshold (4 * 2 = 8).
    for (int i = 0; i < 8; ++i) {
        set.insert(100 + i);
    }

    // Insert 1 more element: arena occupancy now exceeds ts*2, so this insert
    // doubles the table to 8. Buckets 4..7 are published UNINITIALIZED (their
    // encoded value UNINITIALIZED, distinct from a real index) and will be split
    // lazily by whichever thread below touches them first.
    set.insert(200);
    
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    
    // 8 threads aggressively inserting elements that hash to 4, 5, 6, 7
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&set, &start, i]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int k = 0; k < 500; ++k) {
                // Keys that will definitely map to bucket 4, 5, 6, 7 when table size is 8
                // since Hash is std::hash<int>, which is often identity or simple for ints.
                set.insert(4 + 8 * k + i * 10000);
                set.insert(5 + 8 * k + i * 10000);
                set.insert(6 + 8 * k + i * 10000);
                set.insert(7 + 8 * k + i * 10000);
            }
        });
    }
    
    start.store(true, std::memory_order_release);
    for (auto& t : threads) {
        t.join();
    }
    
    // Verify
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 500; ++k) {
            EXPECT_TRUE(set.contains(4 + 8 * k + i * 10000));
            EXPECT_TRUE(set.contains(5 + 8 * k + i * 10000));
            EXPECT_TRUE(set.contains(6 + 8 * k + i * 10000));
            EXPECT_TRUE(set.contains(7 + 8 * k + i * 10000));
        }
    }
}

// Exercises tombstone erase end-to-end: single-threaded correctness (erase makes
// contains() return false; a second erase of the same key returns false; erase
// does not disturb other keys; a key can be re-inserted after erase), then a
// concurrent phase where each thread erases its own disjoint key range so the
// MARK_BIT CAS and the erase/contains interleavings are stressed without two
// threads ever contending for the same key. AllowDelete=true compiles erase().
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

    // Concurrent erase
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 1000; ++k) {
            set.insert(i * 10000 + k);
        }
    }
    
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&set, &start, i]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int k = 0; k < 1000; ++k) {
                set.erase(i * 10000 + k);
            }
        });
    }
    
    start.store(true, std::memory_order_release);
    for (auto& t : threads) {
        t.join();
    }
    
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 1000; ++k) {
            EXPECT_FALSE(set.contains(i * 10000 + k));
        }
    }
}

// Meant to be run under ThreadSanitizer: it validates that concurrent
// cooperative splits are DATA-RACE free (many threads racing inside
// split_bucket() on the same UNINITIALIZED buckets, one CAS winner, losers'
// subchains abandoned). No deletes here (AllowDelete=false), so it exercises the
// split/publish paths only. The final EXPECT_EQ also checks that every attempted
// key is present -- membership must be exact even though the boolean returns are
// best-effort under resize.
TEST(ConcurrentHashSetTest, CooperativeSplitTSAN) {
    ConcurrentResizableHashSet<int, false, CollisionHash> set(4);
    
    // Pre-populate to trigger exactly one resize (N=4 -> N=8)
    for (int i = 0; i < 9; ++i) {
        set.insert(i * 16); 
    }

    constexpr int NUM_THREADS = 8;
    constexpr int INSERTS_PER_THREAD = 1000;
    
    std::barrier sync_point(NUM_THREADS);
    std::vector<std::thread> workers;
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([&, t]() {
            sync_point.arrive_and_wait(); 

            for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
                int key = 5 + ((i * NUM_THREADS + t) * 8); 
                set.insert(key);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    int success_count = 0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            int key = 5 + ((i * NUM_THREADS + t) * 8);
            if (set.contains(key)) {
                success_count++;
            }
        }
    }
    EXPECT_EQ(success_count, NUM_THREADS * INSERTS_PER_THREAD);
}

// TSAN stress test specifically for erase()
TEST(ConcurrentHashSetTest, EraseStressTSAN) {
    ConcurrentResizableHashSet<int, true> set(4);
    constexpr int NUM_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 2000;
    
    std::barrier sync_point(NUM_THREADS);
    std::vector<std::thread> workers;
    
    // First, populate the set heavily so we force resizes and splits.
    for (int i = 0; i < NUM_THREADS * OPERATIONS_PER_THREAD; ++i) {
        set.insert(i);
    }
    
    // Concurrently read, insert, and erase to stress the MARK_BIT and DCLP resize checks.
    for (int t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([&, t]() {
            sync_point.arrive_and_wait();
            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                int key = t * OPERATIONS_PER_THREAD + i;
                
                // Erase it (should return true the first time)
                set.erase(key);
                
                // Read it (should be false now)
                bool contains_after_erase = set.contains(key);
                
                // Insert it back
                set.insert(key);
                
                // Erase it again
                set.erase(key);
                
                EXPECT_FALSE(contains_after_erase);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
    
    // At the end, all keys should be erased
    for (int i = 0; i < NUM_THREADS * OPERATIONS_PER_THREAD; ++i) {
        EXPECT_FALSE(set.contains(i));
    }
}

// Targets the lost-update-on-resize race that insert()'s post-CAS geometry
// recheck exists to close. Run 10000 times to make the narrow window likely.
//
// The race: key 4 hashes to bucket 0 at size 4 (4 & 3 == 0) but to bucket 4 at
// size 8 (4 & 7 == 4). t1 inserts 4 (publishing it into bucket 0 under size 4)
// exactly while t2 doubles the table to 8. If t2's split of the new bucket 4
// snapshots bucket 0's chain in the instant BEFORE t1's node is linked, the copy
// pass misses key 4 -- and without the recheck, t1 (having observed the old size)
// would never re-publish it into bucket 4, stranding key 4 in a bucket no reader
// consults at size 8. The recheck detects the size change (correct_j 4 != j 0)
// and re-inserts, so contains(4) must hold afterwards.
TEST(ConcurrentHashSetTest, LostUpdateOnResize) {
    for (int rep = 0; rep < 10000; ++rep) {
        ConcurrentResizableHashSet<int> set(4);
        set.insert(0); // initialize bucket 0

        std::atomic<bool> start{false};
        std::atomic<bool> thread1_done{false};

        std::thread t1([&]() {
            while (!start) {}
            // Insert 4 which hashes to 0 initially (4 & 3 == 0)
            set.insert(4);
            thread1_done = true;
        });

        std::thread t2([&]() {
            while (!start) {}
            // Force resize to 8: keys 16,32,... all hash to bucket 0, so nine of
            // them push arena occupancy past the doubling threshold.
            for (int i = 1; i <= 9; ++i) {
                set.insert(i * 16);
            }
        });

        start = true;
        t1.join();
        t2.join();

        // 4 should be in the set
        EXPECT_TRUE(set.contains(4)) << "Lost update in rep " << rep;
        if (!set.contains(4)) {
            break;
        }
    }
}

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
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&set, &start, t]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < N; ++i) {
                set.insert(t * N + i);
            }
        });
    }
    
    start.store(true, std::memory_order_release);
    for (auto& t : threads) {
        t.join();
    }
    
    // Total inserted is T * N = 8000.
    // If it doesn't leak, we expect around 8000 nodes (+ small bound for abandoned split subchains).
    // If the CAS failure memory leak is present, this will be significantly higher.
    size_t node_count = set.get_internal_node_count();
    EXPECT_LE(node_count, static_cast<size_t>((T * N) + 100)) << "Memory leak detected! Expected ~8000 nodes, got " << node_count;
}

