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
//   4. reclaim(), called at a quiescent point (nothing else runs on the set, and
//      the caller's joins order it against every other call), changes no
//      membership, returns the exact number of keys, leaves exactly one
//      reachable node per key, and recycles every other arena slot: after it,
//      every slot is reachable, on a free list, or on a limbo list, exactly once
//      (get_internal_accounting()), and the limbo lists are empty. Later
//      allocations take free slots before they grow the arena, and a reused slot
//      takes its new value by copy assignment. See the header's RECLAMATION
//      section and the contract on reclaim().
// Anything weaker than this (a "best effort" boolean, an under-counting insert)
// is a defect, not a documented relaxation -- see the header's "STALE GEOMETRY"
// section, which states 1 and 2 verbatim.
// ===========================================================================
#include "concurrent_hash_set.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <bit>
#include <string>
#include <barrier>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <functional>

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

// ---------------------------------------------------------------------------
// Reclamation helpers. All of them call the test-only sweeps, so all of them
// share reclaim()'s precondition: call them only at a quiescent point (every
// worker joined).
// ---------------------------------------------------------------------------

// Accounting sweep: every arena slot must be reachable from a published bucket,
// on a free list, or on a limbo list, exactly once, and the three counts must
// sum to the slot count. The sum is also part of `consistent`; it is asserted
// separately only for its clearer failure message. No accessor is cross-checked
// here: get_internal_free_count() and get_internal_limbo_count() return fields of
// this same sweep, and `slots` is get_internal_node_count() by construction.
// `where` names the phase in failure messages. Returns the sweep for further,
// test-specific checks.
template <typename SetT>
static typename SetT::InternalAccounting expect_consistent(const SetT& set, const std::string& where) {
    const typename SetT::InternalAccounting acc = set.get_internal_accounting();
    EXPECT_TRUE(acc.consistent) << where << ": some arena slot is unaccounted for or on two lists (slots " << acc.slots
                                << ", reachable " << acc.reachable << ", free " << acc.free_nodes << ", limbo " << acc.limbo << ")";
    EXPECT_EQ(acc.reachable + acc.free_nodes + acc.limbo, acc.slots) << where;
    return acc;
} // expect_consistent()

// reclaim() with every postcondition its contract promises, against the oracle
// `expected_live` (the number of keys in the set, known to the test): the
// return value is exact; the sweep is consistent before and after; reclaim()
// neither grows nor shrinks the arena and only ever unlinks; afterwards exactly
// one node per key is reachable (so reachable == the return value) and the limbo
// lists are empty. Returns what reclaim() returned.
template <typename SetT>
static size_t reclaim_and_check(SetT& set, size_t expected_live, const std::string& where) {
    const typename SetT::InternalAccounting before = expect_consistent(set, where + ", before reclaim()");
    const size_t live = set.reclaim();
    EXPECT_EQ(live, expected_live) << where << ": reclaim() must return the exact number of keys";
    const typename SetT::InternalAccounting after = expect_consistent(set, where + ", after reclaim()");
    EXPECT_EQ(after.slots, before.slots) << where << ": reclaim() must not change the arena size";
    EXPECT_LE(after.reachable, before.reachable) << where << ": reclaim() only unlinks";
    EXPECT_EQ(after.reachable, live) << where << ": more than one reachable node for some key after reclaim()";
    EXPECT_EQ(after.limbo, 0u) << where << ": reclaim() must drain every limbo list";
    return live;
} // reclaim_and_check()

// Membership oracle sweep: the number of wrong contains() answers, where every
// key of `in` must be present and every key of `out` absent. Side effect worth
// knowing: contains() splits every UNINITIALIZED bucket it lands on, so after
// this sweep every probed key's current bucket is published.
template <typename SetT, typename K>
static int membership_errors(SetT& set, const std::vector<K>& in, const std::vector<K>& out) {
    int errors = 0;
    for (const K& k : in) errors += !set.contains(k);
    for (const K& k : out) errors += set.contains(k);
    return errors;
} // membership_errors()

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

// The arena_shards constructor parameter: rounded up to a power of two, 0 means
// the hardware concurrency. A set works with any shard count, including one
// shard shared by every thread; the node count accessor, which sums the shards,
// sees every inserted key (split copies add nodes, so it is a lower bound here).
// Keys are scrambled so the doublings inside the loop split chains (see scramble()).
TEST(ConcurrentHashSetTest, ArenaShardsParameter) {
    ConcurrentResizableHashSet<int> by_default;
    EXPECT_GE(by_default.get_internal_arena_shards(), 1u);
    EXPECT_TRUE(std::has_single_bit(by_default.get_internal_arena_shards()));
    ConcurrentResizableHashSet<int> three(4, 3);
    EXPECT_EQ(three.get_internal_arena_shards(), size_t(4));

    const int T = 4, N = 2000;
    for (size_t shards : {size_t(1), size_t(2), size_t(64)}) {
        ConcurrentResizableHashSet<int> set(4, shards);
        EXPECT_EQ(set.get_internal_arena_shards(), shards);
        std::atomic<int> inserted{0};
        run_threads(T, [&set, &inserted](int t) {
            for (int k = t; k < N; k += T) {
                if (set.insert(scramble(k))) inserted.fetch_add(1, std::memory_order_relaxed);
            }
        });
        EXPECT_EQ(inserted.load(), N);
        for (int k = 0; k < N; ++k) EXPECT_TRUE(set.contains(scramble(k)));
        EXPECT_FALSE(set.contains(scramble(N)));
        // Every inserted key has a node; split copies add more, never fewer.
        EXPECT_GE(set.get_internal_node_count(), size_t(N));
    } // for each shard count
} // ArenaShardsParameter

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
    // encoded value UNINITIALIZED, distinct from a real address) and will be split
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
// header (commit ec875ed; described in the message of c5c19e4): double-true
// insert, erase undone by a stale inserter, double-true erase, and a key lost
// through a doubling.
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
//
// The same race is also the best producer of LIMBO nodes the suite has: a thread
// that loses a publishing CAS to another inserter of the same key orphans its
// node, and threads that race to split one bucket lose their subchains. So after
// the join the body also checks the arena accounting (every orphan and every lost
// subchain must have been recorded on a limbo list at birth) and runs reclaim()
// with its full postconditions (reclaim_and_check()), immediately, with whatever
// splits are still pending, BEFORE the membership sweep, so the sweep also checks
// that reclaim() lost no key. ReclaimDrainsContendedLimbo reuses it for that.
// Interface: T threads each insert the same N scrambled keys into a fresh set with
// `shards` arena shards (0 = the default); `rep` and `flavor` label messages.
// `prefill` > 0 first inserts that many OTHER keys single-threaded, settles their
// pending splits with a contains() sweep (which leaves a stale parent copy of
// every key that moved), erases them again if the set has erase(), and reclaims,
// so the race starts with non-empty free lists: its allocations then POP while
// its orphans and lost subchains are pushed to limbo, on the same shard when
// `shards` is 1 (see ReclaimLimboPushesRaceFreeListPops). Without erase() the
// prefill keys stay in the set and the free lists hold only those stale copies,
// which is why the sweep is there, and why the prefill must cross doublings
// AFTER keys exist (see the sizes measured in ReclaimLimboPushesRaceFreeListPops).
// `min_free` is the floor the free lists must reach after the prefill reclaim().
// Returns the number of nodes that were on limbo lists before reclaim().
template <typename SetT>
static size_t one_winner_per_key_body(int T, int N, int rep, const char* flavor, size_t shards = 0, int prefill = 0, size_t min_free = 0) {
    SetT set(4, shards);
    size_t prefilled = 0;   // prefill keys still in the set when the race starts
    std::vector<int> prefill_keys;
    for (int i = 0; i < prefill; ++i) {
        prefill_keys.push_back(scramble(N + i));
        set.insert(prefill_keys.back());
    }
    if (prefill > 0) {
        EXPECT_EQ(membership_errors(set, prefill_keys, std::vector<int>{}), 0) << "prefill, rep " << rep;   // settles the splits
        if constexpr (requires { set.erase(0); }) {
            for (int i = 0; i < prefill; ++i) set.erase(scramble(N + i));
        } else {
            prefilled = static_cast<size_t>(prefill);
        }
        reclaim_and_check(set, prefilled, std::string("prefill, rep ") + std::to_string(rep));
        EXPECT_GE(set.get_internal_free_count(), min_free) << "prefill, rep " << rep << " (" << flavor
                                                           << "): test precondition: the race would start with a short free list";
    } // start with non-empty free lists
    const size_t expected = static_cast<size_t>(N) + prefilled;
    std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(N, 0));
    run_threads(T, [&set, &won, N](int t) {
        for (int k = 0; k < N; ++k) {
            if (set.insert(scramble(k))) won[t][k] = 1;
        }
    });
    const std::string where = std::string("rep ") + std::to_string(rep) + " (" + flavor + ")";
    const size_t limbo = expect_consistent(set, where + ", after the insert race").limbo;
    reclaim_and_check(set, expected, where);
    int over = 0, under = 0, missing = 0;
    for (int k = 0; k < N; ++k) {
        int c = 0;
        for (int t = 0; t < T; ++t) c += won[t][k];
        if (c > 1) ++over;
        if (c < 1) ++under;
        if (!set.contains(scramble(k))) ++missing;
    } // per-key check
    EXPECT_EQ(over, 0) << over << " key(s) had MORE than one successful insert(), " << where;
    EXPECT_EQ(under, 0) << under << " key(s) had NO successful insert(), " << where;
    EXPECT_EQ(missing, 0) << missing << " key(s) absent after all inserts and a reclaim(), " << where;
    return limbo;
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
// the phase-1 exactly-one-winner check. That is consistent with the split arithmetic:
// the stale node those headers strand in an old bucket j can
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

// ===========================================================================
// Quiescent reclamation: reclaim(), the per-shard free lists popped lock-free by
// alloc_node(), and the per-shard limbo lists. Contract clause 4 at the top of
// this file; the header's RECLAMATION section and the contract on reclaim().
//
// Every call to reclaim() and to the test-only sweeps below happens after
// run_threads() has joined its workers (or on the only thread there is), which
// is the quiescence the contract requires. Oracles are the contract's: the key
// count the test itself knows, the accounting identity (every slot in exactly one
// place), and "allocation takes a free slot before it grows the arena". Where a
// test relies on the documented node_count_ batching (256 per batch), it says so.
// ===========================================================================

// Pre-sized (no doubling), so every node the arena holds is a key or a tombstone
// and every count is exact. One arena shard: reclaim() deals freed slots
// round-robin over ALL shards, and a thread pops only its own shard's list, so
// with the default shard count the main thread could reuse only 1/shards of what
// it freed and "the arena did not grow" would be false by design, not by defect
// (ReclaimReuseAcrossShards covers the multi-shard dealing).
// Weak spots: the unlink walk over tombstones scattered through chains (first,
// middle and last nodes of a chain, runs of consecutive dead nodes); reclaim()
// run twice with nothing to do in between (it must be idempotent: same count,
// nothing new freed, no slot pushed twice -- a double push would show as a cycle
// or a duplicate in the accounting sweep); and the free-list pop path serving
// every reinsertion, including the one that empties the list.
// Sizes: N = 1000 keys in 1024 buckets. The resize hint then peaks at 1024 (four
// append batches of 256) before reclaim() and at 750 + 24 (append credit) + 256
// (first pop batch) after it, both below the doubling threshold 2*1024.
TEST(ConcurrentHashSetTest, ReclaimExactPreSized) {
    const int N = 1000, K = 250;
    {   // AllowDelete == true: tombstones are the only dead nodes
        ConcurrentResizableHashSet<int, true> set(1024, 1);
        std::vector<int> kept, erased, fresh;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(scramble(i)));
        const ConcurrentResizableHashSet<int, true>::InternalAccounting filled = expect_consistent(set, "after the fill");
        EXPECT_EQ(filled.slots, size_t(N));
        EXPECT_EQ(filled.reachable, size_t(N));
        for (int i = 0; i < N; ++i) {
            if (i%4 == 1) {   // every 4th key: tombstones land anywhere in their chains
                EXPECT_TRUE(set.erase(scramble(i)));
                erased.push_back(scramble(i));
            } else {
                kept.push_back(scramble(i));
            }
        } // erase every 4th key
        ASSERT_EQ(erased.size(), size_t(K));
        reclaim_and_check(set, N - K, "first reclaim()");
        EXPECT_EQ(set.get_internal_free_count(), size_t(K)) << "every tombstone, and nothing else, is freed";
        EXPECT_EQ(membership_errors(set, kept, erased), 0);

        // Idempotence: nothing died since, so nothing more is freed.
        reclaim_and_check(set, N - K, "second reclaim()");
        EXPECT_EQ(set.get_internal_free_count(), size_t(K)) << "a second reclaim() freed something, or pushed a slot twice";

        // Reinsert K new keys: every one must be served from the free list.
        for (int i = 0; i < K; ++i) {
            fresh.push_back(scramble(N + i));
            EXPECT_TRUE(set.insert(fresh.back()));
        }
        EXPECT_EQ(set.get_internal_node_count(), size_t(N)) << "the arena grew although the free list held a node for every insert";
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        expect_consistent(set, "after the reinsertion");
        reclaim_and_check(set, N, "third reclaim()");
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        EXPECT_EQ(membership_errors(set, kept, erased), 0);
        EXPECT_EQ(membership_errors(set, fresh, std::vector<int>{}), 0);
    } // AllowDelete == true
    {   // AllowDelete == false: pre-sized, so nothing is ever dead
        ConcurrentResizableHashSet<int, false> set(1024, 1);
        std::vector<int> keys, absent;
        for (int i = 0; i < N; ++i) {
            keys.push_back(scramble(i));
            EXPECT_TRUE(set.insert(keys.back()));
            absent.push_back(scramble(N + i));
        }
        reclaim_and_check(set, N, "AllowDelete=false, first reclaim()");
        EXPECT_EQ(set.get_internal_free_count(), 0u) << "reclaim() freed a live node";
        for (int k : keys) EXPECT_FALSE(set.insert(k));   // duplicates: no allocation, no change
        EXPECT_EQ(set.get_internal_node_count(), size_t(N));
        reclaim_and_check(set, N, "AllowDelete=false, second reclaim()");
        EXPECT_EQ(membership_errors(set, keys, absent), 0);
    } // AllowDelete == false
} // ReclaimExactPreSized

// reclaim() straight after a growing insert burst, with doublings behind it and
// splits still pending, then the reuse that follows.
//
// Weak spot: the dead-node rule, is_dead(). A node in chain j whose key maps
// elsewhere at the current size is dead only if its NEXT CHILD bucket is
// PUBLISHED; while that child is still UNINITIALIZED the node is the key's only
// reachable node, and freeing it loses the key. Right after a burst, a doubling
// has just left half the table UNINITIALIZED, which is exactly that state; a
// reclaim() after a contains() sweep would never see it (the sweep publishes
// every bucket the keys need). Hence many fresh sets, stopped at many sizes
// (1 to ~800 keys: on both sides of the doublings and of the 256-node batches),
// each reclaimed BEFORE anything reads it. Both AllowDelete values: with false,
// the dead nodes are the stale parent copies splits leave behind, with true they
// are frozen parent copies and tombstones.
// The reuse phase has a trap: a reinsertion can land in a bucket that is still
// UNINITIALIZED and split it, which allocates copies. So the node-count baseline
// is taken only after a contains() sweep over every key, which publishes every
// bucket any existing key maps to (a split of any other bucket copies nothing).
// Even then a reinsertion can double the table and split again, so the oracle is
// the design rule itself: with one shard and one thread, the arena grows only
// once the free list is empty.
template <bool AllowDelete>
static void reclaim_with_pending_splits_body() {
    using Set = ConcurrentResizableHashSet<int, AllowDelete>;
    const int sizes[] = {1, 2, 7, 9, 31, 64, 100, 129, 255, 257, 300, 500, 513, 777};
    int reuse_exercised = 0;
    for (int n : sizes) {
        const std::string where = std::string(AllowDelete ? "AllowDelete=true" : "AllowDelete=false") + ", n = " + std::to_string(n);
        Set set(4, 1);
        std::vector<int> in, out;
        for (int i = 0; i < n; ++i) EXPECT_TRUE(set.insert(scramble(i))) << where;
        for (int i = 0; i < n; ++i) {
            if constexpr (AllowDelete) {
                if (i%3 == 0) {
                    EXPECT_TRUE(set.erase(scramble(i))) << where;
                    out.push_back(scramble(i));
                    continue;
                }
            } // if erase() exists
            in.push_back(scramble(i));
        } // erase every third key
        reclaim_and_check(set, in.size(), where + ", reclaim() right after the burst");
        const size_t free1 = set.get_internal_free_count();
        reclaim_and_check(set, in.size(), where + ", second reclaim()");
        EXPECT_EQ(set.get_internal_free_count(), free1) << where << ": the second reclaim() freed something new";
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": reclaim() changed membership";

        // Reuse, with the baseline taken after the contains() sweep just done. That
        // sweep's splits popped the free list (with AllowDelete == false it is
        // usually empty afterwards) and left fresh stale copies behind; a reclaim()
        // now recycles those, so the reuse below has free slots to take.
        reclaim_and_check(set, in.size(), where + ", reclaim() after the settling sweep");
        const size_t slots0 = set.get_internal_node_count();
        const size_t free0 = set.get_internal_free_count();
        for (size_t i = 0; i < free0/2; ++i) {
            in.push_back(scramble(1000000 + n + static_cast<int>(i)));
            EXPECT_TRUE(set.insert(in.back())) << where;
        }
        const size_t free2 = set.get_internal_free_count();
        EXPECT_TRUE(free2 == 0 || set.get_internal_node_count() == slots0)
            << where << ": the arena grew from " << slots0 << " to " << set.get_internal_node_count()
            << " while the free list still held " << free2 << " node(s)";
        if (free0 > 0 && free2 > 0) ++reuse_exercised;
        reclaim_and_check(set, in.size(), where + ", reclaim() after the reuse");
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": membership after the reuse";
    } // for each burst size
    // Not an oracle on the header: a guard that the reuse check above is not vacuous
    // (measured: all 14 sizes with AllowDelete == true, 12 of 14 with false).
    EXPECT_GT(reuse_exercised, 0) << "no size exercised reuse with free nodes left over";
} // reclaim_with_pending_splits_body()

TEST(ConcurrentHashSetTest, ReclaimWithPendingSplits) {
    reclaim_with_pending_splits_body<true>();
    reclaim_with_pending_splits_body<false>();
} // ReclaimWithPendingSplits

// AllowDelete == false, concurrent growth: the dead nodes are the stale parent
// copies splits leave in place (the freeze is compiled out, so they are not even
// marked: nothing but the rule "next child published" tells them apart from live
// nodes). Four threads insert disjoint scrambled keys, one arena shard each.
// Phase 1 reclaims straight after the join (pending splits, as in
// ReclaimWithPendingSplits but with nodes of four shards). Phase 2 then settles
// the keys' splits with a contains() sweep, which leaves a stale copy behind for
// every key that moved, and reclaims again: the sweep's reachable count must have
// exceeded the key count (otherwise the phase tests nothing), and reclaim() must
// bring it back to exactly one node per key. Every insert() boolean is exact: the
// keys are disjoint and each is inserted once, so each call must return true.
TEST(ConcurrentHashSetTest, ReclaimStaleSplitCopiesNoDelete) {
    using Set = ConcurrentResizableHashSet<int, false>;
    const int T = 4, PER = 500, REPS = 4;
    for (int rep = 0; rep < REPS; ++rep) {
        const std::string where = "rep " + std::to_string(rep);
        Set set(4, T);
        std::vector<size_t> bad(T, 0);
        run_threads(T, [&set, &bad, rep](int t) {
            for (int i = 0; i < PER; ++i) bad[t] += !set.insert(scramble((rep*T + t)*PER + i));
        });
        for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ": insert() of a new, uncontended key returned false";
        std::vector<int> in, out;
        for (int i = 0; i < T*PER; ++i) in.push_back(scramble(rep*T*PER + i));
        for (int i = 0; i < 100; ++i) out.push_back(scramble(1000000 + i));
        reclaim_and_check(set, T*PER, where + ", phase 1");
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": reclaim() with pending splits changed membership";
        const size_t reachable = expect_consistent(set, where + ", after the settling sweep").reachable;
        // MEASURED precondition, not derived: with scrambled keys over many doublings
        // it is overwhelmingly likely that many keys moved, but nothing forces it.
        // (Measured: 2520-3188 reachable nodes for 2000 keys under ASan and TSan.)
        ASSERT_GT(reachable, size_t(T*PER)) << where << ": test precondition: the sweep left no stale split copy to reclaim";
        reclaim_and_check(set, T*PER, where + ", phase 2");
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": reclaim() of stale copies changed membership";
        for (int k : in) EXPECT_FALSE(set.insert(k)) << where << ": a duplicate insert() returned true after reclaim()";
    } // repetitions
} // ReclaimStaleSplitCopiesNoDelete

// Free slots dealt over the shards, and popped concurrently by threads of other
// shards. The slots all live in the MAIN thread's shard deque (it inserted
// everything); reclaim() deals them round-robin over every shard's free list; the
// worker threads then pop from their own shards' lists. So every worker that is
// not on the main thread's shard reuses slots that belong to another shard.
// Two shapes, both on a pre-sized table (no splits, so an allocation is an insert
// and nothing else):
//   - 8 threads, 2 shards (threads > shards): four threads pop each shard's list
//     at once, so the pop CAS is contended, and each list is popped 1024 times,
//     so the 9-bit pop counter packed into the free-head word wraps twice (its
//     carry must fall off the top of the word, not into the address).
//   - 2 threads, 8 shards (shards > threads): six lists are never popped.
// The oracle: K freed slots are dealt evenly (K is a multiple of both shard
// counts, so each shard gets exactly K/shards), threads started together get
// consecutive thread numbers and so spread evenly over the shards (header,
// concurrent_hash_detail), and each thread inserts exactly K/max(threads, shards)
// new keys: exactly what its shard's list can serve, shared with its shard mates.
// So the arena must not grow at all, and the free count must drop by exactly the
// number of inserts. Dealing everything to one shard, or popping another
// thread's shard, both show as arena growth.
// Pre-sizing: 16384 buckets, doubling threshold 32768; the resize hint peaks at
// 2048 live + 8 pop batches of 256, far below it.
TEST(ConcurrentHashSetTest, ReclaimReuseAcrossShards) {
    using Set = ConcurrentResizableHashSet<int, true>;
    struct Shape { int threads; size_t shards; };
    const int N = 4096, K = 2048;
    for (const Shape shape : {Shape{8, 2}, Shape{2, 8}}) {
        const int T = shape.threads;
        const int per = K/static_cast<int>(std::max<size_t>(shape.shards, T));
        const std::string where = std::to_string(T) + " threads, " + std::to_string(shape.shards) + " shards";
        Set set(16384, shape.shards);
        std::vector<int> in, out;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(scramble(i)));
        for (int i = 0; i < N; ++i) {
            if (i & 1) {
                in.push_back(scramble(i));
            } else {
                EXPECT_TRUE(set.erase(scramble(i)));
                out.push_back(scramble(i));
            }
        } // erase the even-indexed keys
        reclaim_and_check(set, N - K, where + ", after the erases");
        ASSERT_EQ(set.get_internal_free_count(), size_t(K)) << where;
        const size_t slots = set.get_internal_node_count();

        std::vector<size_t> bad(T, 0);
        run_threads(T, [&set, &bad, per](int t) {
            for (int i = 0; i < per; ++i) bad[t] += !set.insert(scramble(N + t*per + i));
        });
        for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ": insert() of a new, uncontended key returned false";
        for (int i = 0; i < T*per; ++i) in.push_back(scramble(N + i));
        EXPECT_EQ(set.get_internal_node_count(), slots) << where << ": the arena grew although every thread's shard held a free slot for each of its inserts";
        EXPECT_EQ(set.get_internal_free_count(), size_t(K - T*per)) << where;
        reclaim_and_check(set, in.size(), where + ", after the reuse");
        EXPECT_EQ(membership_errors(set, in, out), 0) << where;
    } // for each shape
} // ReclaimReuseAcrossShards

// After reclaim(), the structure is a mix of KEPT nodes (in chains whose dead
// neighbours were unlinked, links rewritten through their predecessors) and
// REUSED nodes (popped from a free list, value copy-assigned, link rewritten).
// Both must take part in the concurrent protocol exactly like fresh nodes: the
// erase MARK CAS and the split FREEZE CAS expect a live link, the split SEAL CAS
// expects the head reclaim() left behind (with its seal level kept), and a split
// copies from chains reclaim() relinked. This test runs concurrent erases and
// inserts, with the table doubling, over such a structure, round after round
// with a reclaim() between rounds so slots are recycled more than once.
// Oracle: every key is owned by one thread, so every boolean is fixed by the
// contract (as in ThreadPrivateKeySequenceDuringGrowth): per round, each thread
// inserts its fresh keys (true), erases every other one again (true, then false),
// and erases and re-inserts each of its keys that survived the previous rounds
// (true, absent, true, present). After the join, the exact key count and the
// membership of every key ever used are checked, along with the accounting.
// Shape: four threads on two shards (threads > shards, so pops race), fresh tiny
// sets, and more fresh keys per round than the set held before it, so each
// round allocates past the free lists and doubles the table while the erasers
// and re-inserters work. Keys scrambled (see scramble()).
TEST(ConcurrentHashSetTest, ReclaimThenConcurrentEraseAndGrowth) {
    using Set = ConcurrentResizableHashSet<int, true>;
    const int T = 4, REPS = 12, ROUNDS = 3, A = 96, F = 96;
    int next = 0;   // next unused key index; keys are scramble(index), all distinct
    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4, 2);
        std::vector<int> present, absent;
        for (int i = 0; i < A; ++i) {   // phase A, single-threaded: half the keys become tombstones
            const int k = scramble(next++);
            EXPECT_TRUE(set.insert(k));
            if (i & 1) {
                EXPECT_TRUE(set.erase(k));
                absent.push_back(k);
            } else {
                present.push_back(k);
            }
        } // phase A
        reclaim_and_check(set, present.size(), "rep " + std::to_string(rep) + ", phase A");

        for (int round = 0; round < ROUNDS; ++round) {
            const std::string where = "rep " + std::to_string(rep) + ", round " + std::to_string(round);
            std::vector<std::vector<int>> old(T), fresh(T);
            for (size_t i = 0; i < present.size(); ++i) old[i%T].push_back(present[i]);
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < F; ++i) fresh[t].push_back(scramble(next++));
            }
            std::vector<size_t> bad(T, 0);
            run_threads(T, [&set, &old, &fresh, &bad](int t) {
                const std::vector<int>& mine = old[t];
                const std::vector<int>& fr = fresh[t];
                size_t b = 0;
                for (size_t i = 0; i < std::max(mine.size(), fr.size()); ++i) {
                    if (i < fr.size()) {
                        b += !set.insert(fr[i]);        // absent -> present (a reused slot, early on)
                        b += !set.contains(fr[i]);
                        if (i & 1) {
                            b += !set.erase(fr[i - 1]);  // present -> absent
                            b += set.contains(fr[i - 1]);
                            b += set.erase(fr[i - 1]);   // already absent
                        }
                    } // fresh key
                    if (i < mine.size()) {
                        b += !set.erase(mine[i]);       // a kept node, or one reused in an earlier round
                        b += set.contains(mine[i]);
                        b += !set.insert(mine[i]);
                        b += !set.contains(mine[i]);
                    } // surviving key
                } // key loop
                bad[t] = b;
            });
            for (int t = 0; t < T; ++t) {
                EXPECT_EQ(bad[t], 0u) << where << ", thread " << t << ": a determined result was wrong";
                for (int i = 0; i < F; ++i) ((i & 1) ? present : absent).push_back(fresh[t][i]);
            }
            reclaim_and_check(set, present.size(), where);
            EXPECT_EQ(membership_errors(set, present, absent), 0) << where;
        } // rounds
    } // repetitions
} // ReclaimThenConcurrentEraseAndGrowth

// Instrumented key for the reuse-by-assignment tests: counts copy constructions,
// copy assignments and destructions, can be armed to throw from its copy
// assignment, and carries a MAGIC word that every operation checks, so a slot
// whose value was destroyed (or never constructed) and then used again is
// counted in `corrupt`. The destructor clears MAGIC for that purpose.
// Its throwing assignment throws BEFORE touching the left-hand side, i.e. it
// gives the strong guarantee the header asks of T ("a throwing assignment must
// leave the old object valid").
// It also carries a one-shot HOOK, run from inside a copy assignment, which is
// how ReclaimRecoversPendingInsertNodeWhenRetryThrows injects another operation
// between an insert()'s read of a bucket head and its publishing CAS.
// The counters and the hook are global, so ProbeKey is used only by
// single-threaded tests.
struct ProbeKey {
    static constexpr int MAGIC = 0x5eed;
    int v;       // the key
    int magic;   // MAGIC while the object is alive
    static inline std::atomic<int> copy_constructions{0};
    static inline std::atomic<int> copy_assignments{0};   // successful ones
    static inline std::atomic<int> destructions{0};
    static inline std::atomic<int> corrupt{0};            // operations that met an object without MAGIC
    // > 0: the copy assignment that brings it to 0 throws; <= 0: disarmed.
    static inline std::atomic<int> throw_countdown{0};
    // If set, the next copy assignment runs it once (after the countdown check,
    // before the assignment itself) and clears it first, so an assignment made
    // by the hook's own operations does not run it again.
    static inline std::function<void()> on_assign;

    ProbeKey(int x = 0) : v(x), magic(MAGIC) {}
    ProbeKey(const ProbeKey& o) : v(o.v), magic(MAGIC) {
        check(o);
        ++copy_constructions;
    }
    ProbeKey& operator=(const ProbeKey& o) {
        check(*this);   // the old value of a reused slot must still be a valid object
        check(o);
        if (throw_countdown.load() > 0 && --throw_countdown == 0) throw std::runtime_error("ProbeKey assignment armed to throw");
        if (on_assign) {
            const std::function<void()> hook = std::move(on_assign);
            on_assign = nullptr;
            hook();
        } // one-shot hook
        v = o.v;
        ++copy_assignments;
        return *this;
    } // ProbeKey::operator=()
    ~ProbeKey() {
        check(*this);
        magic = 0;
        ++destructions;
    }
    bool operator==(const ProbeKey& o) const { return v == o.v; }
    static void check(const ProbeKey& k) {
        if (k.magic != MAGIC) ++corrupt;
    }
    static void reset_counters() {
        copy_constructions = 0;
        copy_assignments = 0;
        destructions = 0;
        corrupt = 0;
        throw_countdown = 0;
        on_assign = nullptr;
    } // reset_counters()
}; // struct ProbeKey

// Identity hash of the key, like CollisionHash: the tests that use ProbeKey pick
// buckets by hand, or scramble the keys themselves.
struct ProbeKeyHash {
    size_t operator()(const ProbeKey& k) const { return static_cast<size_t>(static_cast<unsigned>(k.v)); }
};

// Reuse copy-assigns, reclaim() destroys nothing, and a throwing assignment on a
// popped slot loses nothing.
//
// Weak spot: the one exception path in alloc_node(). The slot has already left
// the free list when the assignment throws, so unless it is recorded somewhere
// it is gone for good; the design sends it to the shard's limbo list, keeps the
// old value (the assignment never completed), and propagates the exception. The
// next reclaim() must recycle the slot like any other.
// Checked, in order: the exception reaches the caller of insert(); the key is not
// in the set; the slot is on limbo (limbo 1, free list one shorter, arena
// unchanged, accounting consistent); nothing touched a destroyed object; the
// retried insert succeeds from the next free slot; reclaim() runs no destructor
// and drains limbo; and a reinsertion of exactly as many keys as there are free
// slots, the formerly-limbo one included, empties the free list without growing
// the arena, with one copy assignment and no copy construction per key.
// Pre-sized, one shard, single-threaded (ProbeKey's counters are global).
TEST(ConcurrentHashSetTest, ReclaimThrowingAssignment) {
    using Set = ConcurrentResizableHashSet<ProbeKey, true, ProbeKeyHash>;
    const int N = 64, E = 16;
    ProbeKey::reset_counters();
    {
        Set set(1024, 1);
        std::vector<int> in, out;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(scramble(i)));
        for (int i = 0; i < N; ++i) {
            if (i < E) {
                EXPECT_TRUE(set.erase(scramble(i)));
                out.push_back(scramble(i));
            } else {
                in.push_back(scramble(i));
            }
        } // erase the first E keys
        reclaim_and_check(set, N - E, "after the erases");
        ASSERT_EQ(set.get_internal_free_count(), size_t(E));
        const size_t slots = set.get_internal_node_count();

        ProbeKey::copy_assignments = 0;
        ProbeKey::throw_countdown = 1;   // the next assignment throws
        EXPECT_THROW(set.insert(scramble(N)), std::runtime_error);
        ProbeKey::throw_countdown = 0;
        EXPECT_FALSE(set.contains(scramble(N))) << "the key whose insert() threw is in the set";
        EXPECT_EQ(set.get_internal_limbo_count(), 1u) << "the popped slot must go to limbo";
        EXPECT_EQ(set.get_internal_free_count(), size_t(E - 1));
        EXPECT_EQ(set.get_internal_node_count(), slots);
        expect_consistent(set, "after the throwing insert()");
        EXPECT_EQ(ProbeKey::copy_assignments.load(), 0);

        EXPECT_TRUE(set.insert(scramble(N)));   // the retry pops the next free slot
        in.push_back(scramble(N));
        EXPECT_EQ(set.get_internal_free_count(), size_t(E - 2));
        EXPECT_EQ(set.get_internal_limbo_count(), 1u);

        ProbeKey::destructions = 0;
        reclaim_and_check(set, in.size(), "after the retry");
        EXPECT_EQ(ProbeKey::destructions.load(), 0) << "reclaim() must not destroy values (they are assigned over at reuse)";
        const size_t free_slots = set.get_internal_free_count();
        EXPECT_EQ(free_slots, size_t(E - 1)) << "the limbo slot must be back on a free list";

        ProbeKey::copy_assignments = 0;
        ProbeKey::copy_constructions = 0;
        for (size_t i = 0; i < free_slots; ++i) {
            in.push_back(scramble(N + 1 + static_cast<int>(i)));
            EXPECT_TRUE(set.insert(in.back()));
        }
        EXPECT_EQ(set.get_internal_node_count(), slots) << "the arena grew although the free list (with the limbo slot) held a slot per insert";
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        EXPECT_EQ(ProbeKey::copy_assignments.load(), static_cast<int>(free_slots)) << "a reused slot takes its value by copy assignment";
        EXPECT_EQ(ProbeKey::copy_constructions.load(), 0) << "reuse constructed a value";
        reclaim_and_check(set, in.size(), "after the reuse");
        EXPECT_EQ(membership_errors(set, in, out), 0);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0) << "an operation met a destroyed or never-constructed value";
} // ReclaimThrowingAssignment

// Geometry for the single-shard tests that must make the table double at an
// allocation of their choosing: CollisionHash (identity), initial size 256.
// fill_key(i) keeps i's low byte, so it lives in bucket i & 255 at every table
// size up to 1024 (bits 8 and 9 are clear), and a doubling to 512 therefore
// moves NO fill key and splits nothing they need. The PROBE keys below have bit
// 8 set: they move to bucket (low byte) + 256 at the doubling to 512, and a
// contains() of one then splits that bucket, which leaves a stale parent copy
// behind (reachable count = keys + 1). That is how these tests SEE a doubling.
static int fill_key(int i) { return (i & 255) | ((i >> 8) << 10); }
static constexpr int PROBE_X1 = 5 | 256;          // bucket 5 at size 256, 261 at 512
static constexpr int PROBE_X2 = 5 | 256 | 4096;   // same two buckets; a second key to move
static constexpr int PROBE_E = 5 | (7 << 10);     // bucket 5 at every size up to 1024; never a fill key

// Fills a set built as SetT(256, 1) with the keys of `first`, in order, then with
// fill_key(0), fill_key(1), ... until the arena holds exactly 512 slots; erases
// fill_key(0 .. erase-1); reclaims, checking the result. Returns the index of the
// first fill key not yet used.
// The arithmetic, from the header's documented batching of node_count_ (the
// resize hint; a doubling happens after a successful insert that sees it above
// 2*table_size, i.e. above 512 here): the appends at indices 0 and 256 add 256
// each, so the hint reaches exactly 512 and the table does not double. reclaim()
// resets the hint to the live count plus the append credit, which is 0 because
// 512 is a multiple of 256. From then on the first pop after each reclaim() adds
// 256, so it doubles the table iff live + 256 > 512, i.e. iff live > 256.
template <typename SetT>
static int fill_to_512_and_reclaim(SetT& set, const std::vector<int>& first, int erase) {
    for (int k : first) EXPECT_TRUE(set.insert(k));
    const int fills = 512 - static_cast<int>(first.size());
    for (int i = 0; i < fills; ++i) EXPECT_TRUE(set.insert(fill_key(i)));
    for (int i = 0; i < erase; ++i) EXPECT_TRUE(set.erase(fill_key(i)));
    reclaim_and_check(set, 512 - erase, "fill_to_512_and_reclaim()");
    EXPECT_EQ(set.get_internal_free_count(), size_t(erase));
    return fills;
} // fill_to_512_and_reclaim()

// The first pop after reclaim() restarts the resize hint's batching.
//
// Weak spot: node_count_, the resize hint, is reset by reclaim() to the live
// count, and pops from the free lists then count in batches of 256 through a
// counter packed into each free-head word; the header states that every 256th
// pop, THE FIRST POP AFTER A reclaim() INCLUDED, adds 256, which keeps the hint
// an over-count (the table may double early, never late). Two ways to break it:
// a pop that never adds to the hint, and a reclaim() that leaves the pop counter
// where the previous cycle's pops left it (then the first pops after the reset
// add nothing until the counter reaches its next multiple of 256).
// Neither is visible to any other test: membership, booleans and accounting are
// all unaffected; only the moment of a doubling moves. So this test builds the
// arithmetic in fill_to_512_and_reclaim() and watches for the doubling with the
// probe key:
//   1. fill, erase 300 fill keys, reclaim(): live 212 (PROBE_X1 + 211 fills);
//   2. cycle 1: 100 inserts, all pops. The first adds 256: hint 468, no doubling
//      (probe check: no split). The pop counter now stands at 100;
//   3. reclaim(): live 312, hint 312, counter cleared;
//   4. cycle 2: ONE insert, a pop. It must add 256: hint 568 > 512, so the insert
//      doubles the table to 512. contains(PROBE_X1) then splits bucket 261.
// Cycle 1 is what makes the "counter not cleared" defect visible: with the
// counter left at 100, the next batch comes only after 156 more pops.
TEST(ConcurrentHashSetTest, ReclaimRestartsPopBatching) {
    using Set = ConcurrentResizableHashSet<int, true, CollisionHash>;
    Set set(256, 1);
    int next = fill_to_512_and_reclaim(set, {PROBE_X1}, 300);
    size_t live = 212;

    for (int i = 0; i < 100; ++i) EXPECT_TRUE(set.insert(fill_key(next++)));   // cycle 1
    live += 100;
    EXPECT_TRUE(set.contains(PROBE_X1));
    ASSERT_EQ(expect_consistent(set, "cycle 1").reachable, live)
        << "test precondition: the table doubled during cycle 1, so the arithmetic above does not hold";
    ASSERT_EQ(set.get_internal_node_count(), 512u) << "test precondition: cycle 1 must be served from the free list";
    reclaim_and_check(set, live, "after cycle 1");

    EXPECT_TRUE(set.insert(fill_key(next++)));   // cycle 2: the first pop after reclaim()
    ++live;
    EXPECT_TRUE(set.contains(PROBE_X1));   // at size 512 this splits bucket 261
    EXPECT_EQ(expect_consistent(set, "cycle 2").reachable, live + 1)
        << "the first pop after reclaim() did not add its batch to the resize hint: the table did not double";
    reclaim_and_check(set, live, "after cycle 2");   // the stale parent copy of PROBE_X1 is dead now
} // ReclaimRestartsPopBatching

// A FROZEN node whose split never finished is KEPT by reclaim(), and a dead node
// after it is unlinked through its (FROZEN) link.
//
// Weak spots: (1) the dead-node rule must keep a FROZEN node whose child bucket is
// still UNINITIALIZED: it is then the key's only node (INVARIANT in the header);
// (2) the relink through a kept predecessor's link must keep its tag bits.
// A FROZEN node with an unpublished child exists at a quiescent point only when a
// split threw between the freeze and the publication, so this test makes one
// throw: the first copy the split of bucket 261 makes pops a free slot whose
// assignment throws. The split froze exactly one node when that happens (the
// first moving node of the walk), and allocated nothing yet, so nothing leaks
// (compare ReclaimAccountsForSplitCopiesWhenAssignmentThrows below, where the
// SECOND copy throws).
// Chain order is newest first; the insert order PROBE_X1, PROBE_E, PROBE_X2 makes
// bucket 5's chain end in  ... PROBE_X2 -> PROBE_E -> PROBE_X1, so the split
// freezes PROBE_X2, and after PROBE_E is erased, reclaim() unlinks PROBE_E through
// PROBE_X2's FROZEN link. Afterwards both probe keys must still be members (the
// next split of 261 copies the frozen and the live one alike), and the key count
// and accounting must be exact.
TEST(ConcurrentHashSetTest, ReclaimKeepsFrozenNodeOfUnfinishedSplit) {
    using Set = ConcurrentResizableHashSet<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = fill_to_512_and_reclaim(set, {PROBE_X1, PROBE_E, PROBE_X2}, 200);
        size_t live = 312;
        EXPECT_TRUE(set.insert(fill_key(next++)));   // the first pop: doubles the table to 512
        ++live;

        ProbeKey::throw_countdown = 1;   // the split's first copy throws
        EXPECT_THROW(set.contains(PROBE_X1), std::runtime_error);
        ProbeKey::throw_countdown = 0;
        expect_consistent(set, "after the split threw");
        EXPECT_EQ(set.get_internal_limbo_count(), 1u) << "the slot whose assignment threw must be on limbo";

        EXPECT_TRUE(set.erase(PROBE_E));   // the dead node behind the FROZEN one
        --live;
        reclaim_and_check(set, live, "after the erase");
        EXPECT_TRUE(set.contains(PROBE_X1));
        EXPECT_TRUE(set.contains(PROBE_X2));
        EXPECT_FALSE(set.contains(PROBE_E));
        EXPECT_FALSE(set.insert(PROBE_X2)) << "a key kept only by its FROZEN node was lost";
        reclaim_and_check(set, live, "after the split finished");
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // ReclaimKeepsFrozenNodeOfUnfinishedSplit

// NO ARENA SLOT IS LOST TO A THROW (header, "EXCEPTIONS: WHAT IS AND IS NOT
// HANDLED"): the partial subchain of a split that throws.
//
// Weak spot: split_bucket() builds its child subchain privately, referenced by
// nothing but the local head and tail. When an allocation in the middle of the
// walk throws (a popped slot's copy assignment, as here, or the append's copy
// construction), the copies made so far must go to limbo before the exception
// leaves, or they are lost for the life of the set. (Before the header's fix
// this test was a DISABLED_ defect reproducer: exactly one slot of 512 leaked.)
// Same setup as ReclaimKeepsFrozenNodeOfUnfinishedSplit, but the split has two
// keys to copy (PROBE_X2, then PROBE_X1) and the SECOND copy throws, so the
// subchain holds one node at the throw. Checked: the exception reaches the
// caller; limbo holds exactly two nodes (the slot whose assignment threw, pushed
// by alloc_node(), and the one-node partial subchain, pushed by split_bucket());
// the accounting is consistent; both probe keys are still members (FROZEN in
// bucket 5 with 261 unpublished, so the next split copies them again); and
// reclaim() recycles both limbo nodes with the exact key count.
TEST(ConcurrentHashSetTest, ReclaimAccountsForSplitCopiesWhenAssignmentThrows) {
    using Set = ConcurrentResizableHashSet<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = fill_to_512_and_reclaim(set, {PROBE_X1, PROBE_X2}, 200);
        size_t live = 312;
        EXPECT_TRUE(set.insert(fill_key(next++)));   // the first pop: doubles the table to 512
        ++live;

        ProbeKey::throw_countdown = 2;   // the split's second copy throws
        EXPECT_THROW(set.contains(PROBE_X1), std::runtime_error);
        ProbeKey::throw_countdown = 0;
        expect_consistent(set, "after the split threw on its second copy");
        EXPECT_EQ(set.get_internal_limbo_count(), 2u) << "the throwing slot and the one-node partial subchain must both be on limbo";
        EXPECT_TRUE(set.contains(PROBE_X1));
        EXPECT_TRUE(set.contains(PROBE_X2));
        reclaim_and_check(set, live, "after the split finished");
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // ReclaimAccountsForSplitCopiesWhenAssignmentThrows

// NO ARENA SLOT IS LOST TO A THROW: the pending node of an insert() whose
// publishing CAS lost and whose retry then throws.
//
// Weak spot: insert() allocates its node once and keeps it, private, across CAS
// retries. If a retry throws (here: from the split_bucket() it must run because
// the table doubled in between), that node must go to limbo; it is referenced by
// nothing else. Reaching the path needs a lost CAS, i.e. another insert into the
// same bucket between this insert's read of the head and its CAS, AND a doubling
// in the same window, so that the retry lands on an UNINITIALIZED bucket.
// That window is forced DETERMINISTICALLY, single-threaded: the outer insert's
// node comes from the free list, so its value is copy-assigned inside
// alloc_node(), after the head was read and before the CAS, and ProbeKey's
// one-shot hook runs a complete nested insert() right there. That is exactly an
// interleaving a second thread could produce; no lock is held on the pop path,
// so the nesting cannot deadlock. The arithmetic (fill_to_512_and_reclaim()):
//   - live 312 after the reclaim, table 256, doubling threshold 512;
//   - the outer insert(PROBE_K) reads bucket 5's head, pops a node: the first pop
//     after reclaim() adds 256 to the hint (568), then assigns: the hook runs;
//   - the nested insert(fill_key(517)), low byte 5, so also bucket 5, pops a
//     second node, publishes it (bucket 5's head changes) and, seeing 568 > 512,
//     doubles the table to 512; the hook then arms the next assignment to throw;
//   - the outer CAS fails (stale head); the retry loads size 512, where PROBE_K
//     (bit 8 set) maps to bucket 261, UNINITIALIZED; split_bucket(261) copies
//     PROBE_X1, whose popped slot's assignment throws.
// Afterwards limbo must hold exactly two nodes: the throwing slot (alloc_node())
// and the outer insert's pending node (insert()'s catch); the split's subchain
// was still empty. PROBE_K must be absent, the nested key present, and reclaim()
// must recycle both with the exact key count; PROBE_K then inserts normally.
TEST(ConcurrentHashSetTest, ReclaimRecoversPendingInsertNodeWhenRetryThrows) {
    using Set = ConcurrentResizableHashSet<ProbeKey, true, ProbeKeyHash>;
    const int PROBE_K = 5 | 256 | 8192;   // bucket 5 at size 256, 261 at 512; not a fill or probe key
    const int NESTED = fill_key(517);     // bucket 5 at every size up to 1024; beyond the fill keys used
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        fill_to_512_and_reclaim(set, {PROBE_X1}, 200);
        size_t live = 312;

        bool nested_inserted = false;
        ProbeKey::on_assign = [&set, &nested_inserted, NESTED]() {
            nested_inserted = set.insert(NESTED);
            ProbeKey::throw_countdown = 1;   // the next assignment (the retry's split copy) throws
        };
        EXPECT_THROW(set.insert(PROBE_K), std::runtime_error);
        ProbeKey::on_assign = nullptr;
        ProbeKey::throw_countdown = 0;
        ASSERT_TRUE(nested_inserted) << "test precondition: the nested insert did not run inside the outer one's window";
        ++live;
        expect_consistent(set, "after the retry threw");
        EXPECT_EQ(set.get_internal_limbo_count(), 2u) << "the throwing slot and the insert's pending node must both be on limbo";
        EXPECT_FALSE(set.contains(PROBE_K)) << "the insert that threw before publishing made its key a member";
        EXPECT_TRUE(set.contains(NESTED));
        EXPECT_TRUE(set.contains(PROBE_X1));
        reclaim_and_check(set, live, "after the retry threw");
        EXPECT_TRUE(set.insert(PROBE_K));
        ++live;
        reclaim_and_check(set, live, "after PROBE_K was inserted");
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // ReclaimRecoversPendingInsertNodeWhenRetryThrows

// Limbo gets fed under contention, and reclaim() drains it.
//
// Limbo is where the two kinds of never-published node go: the node an insert()
// prepared for a key another thread published first (an orphan), and the
// subchain of a split that lost its publishing CAS. Both need real races: 8
// threads insert the same 64 scrambled keys into a fresh 4-bucket set, so they
// collide on keys and on the splits of the doublings they drive. Two arena shards
// for eight threads, so four threads share each limbo head and its Treiber push
// is contended too. one_winner_per_key_body() checks, per repetition: exactly one
// successful insert() per key; the accounting sweep right after the race (every
// orphan and lost subchain accounted for on a limbo list); reclaim() with its
// postconditions (limbo empty afterwards, exact key count, one node per key)
// while splits are still pending; and membership after it.
// The limbo total must be positive: the test is only meaningful if the races it
// exists for happened. (Measured at -O0 over the 40 repetitions: 1070-1350 limbo
// nodes under ASan, 3120-3320 under TSan, at least 2 in every repetition in
// those runs; the assertion is on the total anyway, since nothing forces any
// single race.)
TEST(ConcurrentHashSetTest, ReclaimDrainsContendedLimbo) {
    const int T = 8, K = 64, REPS = 40;
    size_t limbo_total = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        if (rep & 1) {
            limbo_total += one_winner_per_key_body<ConcurrentResizableHashSet<int, true>>(T, K, rep, "AllowDelete=true", 2);
        } else {
            limbo_total += one_winner_per_key_body<ConcurrentResizableHashSet<int, false>>(T, K, rep, "AllowDelete=false", 2);
        }
    } // repetitions
    EXPECT_GT(limbo_total, 0u) << "no node ever reached a limbo list: the races this test exists for did not happen";
} // ReclaimDrainsContendedLimbo

// Limbo pushes racing free-list POPS on ONE shard.
//
// Weak spot: the two lists of a shard are written concurrently by different
// protocols. Pops take the free head by CAS and assume nothing else writes it
// between two quiescent points (the header's whole ABA argument); limbo pushes
// go to a separate head, by CAS. A limbo push that went to the free list instead
// (with the free list's quiescent-only plain push, or with any push at all) would
// race the pops: lost pushes (a slot leaks) or a slot popped twice (two keys in
// one node). ReclaimDrainsContendedLimbo cannot see that: its sets are fresh, so
// their free lists are empty for the whole race and nothing is popped while
// anything is pushed, and it spreads the threads over two shards.
// Shape: ONE shard for 8 threads, and a prefill (see one_winner_per_key_body())
// so the free list holds a few hundred slots when the race starts; the 8 threads
// then insert the same 64 scrambled keys, so their allocations pop that one list
// while their orphans and lost split subchains are pushed to that one shard's
// limbo. Oracle, per repetition, from one_winner_per_key_body(): exactly one
// winning insert() per key, a consistent accounting straight after the race (a
// lost push or a double pop breaks it), reclaim() exact with limbo drained,
// membership exact. The limbo total must be positive, or the pushes this test
// exists for never happened.
// Prefill size, and the free-list floor asserted before each race. With
// AllowDelete == true every prefill key is erased, so the free list holds at
// least PREFILL slots by construction (measured: 941, the rest being stale split
// copies). With AllowDelete == false the free list holds ONLY stale split copies
// of the prefill, which exist only for keys that moved at a doubling after they
// were inserted: a one-shard set jumps to 128 buckets within its first five
// inserts (the first append counts 256), so a 256-key prefill left just 6 free
// slots (measured), too few to be popped during the race. 600 keys cross the
// doublings at 256 and 512 slots with the keys already in place, and the settling
// sweep turns those splits into stale copies: measured 341 free slots. The floor
// for that flavor, PREFILL/4, is therefore a MEASURED precondition, not a derived
// one (the prefill is single-threaded, so the count is the same every run).
// Measured limbo totals over the 40 repetitions: ASan 38-63, TSan 1500-2900
// (TSan's instrumentation widens the races); a mutant header whose limbo pushes
// go to the free list instead failed this test in 10 of 10 runs under both.
TEST(ConcurrentHashSetTest, ReclaimLimboPushesRaceFreeListPops) {
    const int T = 8, K = 64, PREFILL = 600, REPS = 40;
    size_t limbo_total = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        if (rep & 1) {
            limbo_total += one_winner_per_key_body<ConcurrentResizableHashSet<int, true>>(T, K, rep, "AllowDelete=true, one shard", 1, PREFILL, PREFILL);
        } else {
            limbo_total += one_winner_per_key_body<ConcurrentResizableHashSet<int, false>>(T, K, rep, "AllowDelete=false, one shard", 1, PREFILL, PREFILL/4);
        }
    } // repetitions
    EXPECT_GT(limbo_total, 0u) << "no node ever reached a limbo list: the races this test exists for did not happen";
} // ReclaimLimboPushesRaceFreeListPops

// ACCEPTANCE: the purpose of reclaim(). A workload whose live key count is
// constant (every round, each thread erases all its keys and inserts as many new
// ones) must run in bounded memory when every round ends with a reclaim(), and
// the arena must grow without bound when none does.
// Oracles:
//   - booleans: each thread's keys are its own and new each round, so every
//     erase() and every insert() must return true;
//   - with reclaim(): the exact live count (T*W) at every reclaim(), and the arena
//     (slots, i.e. get_internal_node_count(), which the accounting also checks)
//     after every round at most BOUND times that live count. BOUND = 4 is derived,
//     not fitted: a round allocates W nodes per thread plus split copies, and
//     after the first reclaim() the free lists hold at least what the previous
//     round allocated, dealt evenly; so the arena stabilizes after round 1 at
//     about (fill + round-1 allocations + split copies), i.e. 2-3 times the live
//     count (measured: 2.7 times, 5484-5593 slots for 2048 keys, under ASan and
//     TSan, flat from round 1 or 2 on). A reclaim() that recycled nothing would
//     add at least one live count per round and cross the bound by round 3;
//   - without reclaim(): every successful insert() allocates a node that is never
//     reused, so the arena holds at least (R + 1)*T*W slots, which is more than
//     BOUND*T*W: this half proves the bound above is not vacuous.
// One arena shard per thread: T threads started together get T consecutive thread
// numbers, one per shard, whatever round they run in; with fewer shards than
// threads the bound would hold as well, but with MORE shards some rounds' threads
// would find their shards' lists short and append (dealing is even over all
// shards), which would be correct behavior and a looser bound.
TEST(ConcurrentHashSetTest, ReclaimBoundsTheArena) {
    using Set = ConcurrentResizableHashSet<int, true>;
    const int T = 8, W = 256, R = 8;
    const size_t L = size_t(T)*W, BOUND = 4;
    for (const bool with_reclaim : {true, false}) {
        const std::string flavor = with_reclaim ? "with reclaim()" : "without reclaim()";
        Set set(4, T);
        // Key of thread t, round r, index i: distinct for every (t, r, i).
        auto key = [](int t, int r, int i) { return scramble((r*T + t)*W + i); };
        std::vector<size_t> bad(T, 0);
        run_threads(T, [&set, &bad, &key](int t) {
            for (int i = 0; i < W; ++i) bad[t] += !set.insert(key(t, 0, i));
        });
        for (int r = 0; r <= R; ++r) {   // round 0 is the fill above; rounds 1..R churn
            const std::string where = flavor + ", round " + std::to_string(r);
            if (r > 0) {
                run_threads(T, [&set, &bad, &key, r](int t) {
                    for (int i = 0; i < W; ++i) {
                        bad[t] += !set.erase(key(t, r - 1, i));
                        bad[t] += !set.insert(key(t, r, i));
                    }
                });
            } // churn
            for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ", thread " << t << ": a determined result was wrong";
            if (with_reclaim) {
                reclaim_and_check(set, L, where);
                EXPECT_LE(set.get_internal_node_count(), BOUND*L) << where << ": the arena outgrew " << BOUND << " times the live count";
            } else {
                expect_consistent(set, where);
            } // with or without reclaim()
        } // rounds
        if (!with_reclaim) {
            EXPECT_GE(set.get_internal_node_count(), (R + 1)*L) << "every successful insert() allocates a node that nothing reuses";
            EXPECT_GT(set.get_internal_node_count(), BOUND*L) << "the bound is vacuous: the unreclaimed arena stays within it";
        }
        std::vector<int> in, out;
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < W; ++i) {
                in.push_back(key(t, R, i));
                out.push_back(key(t, R - 1, i));
            }
        } // last two rounds' keys
        EXPECT_EQ(membership_errors(set, in, out), 0) << flavor;
    } // with and without reclaim()
} // ReclaimBoundsTheArena
