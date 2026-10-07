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
//   4. At every quiescent point (nothing runs on the set, and the caller's joins
//      order the point against every call) each arena slot is in exactly one of
//      four places (get_internal_accounting()): reachable from a published
//      bucket, on a free list, on a limbo list (a node that was never
//      published), or on a retired list (a dead node that an operation unlinked
//      from its chain). reclaim(), called at a quiescent point, changes no
//      membership, returns the exact number of keys, leaves exactly one
//      reachable node per key, and recycles every other arena slot: after it the
//      limbo and retired lists are empty and every other slot is on a free list.
//      Later allocations take free slots before they grow the arena, and a
//      reused slot takes its new value by copy assignment. See the header's
//      RECLAMATION section and the contract on reclaim().
//   5. Between reclaim() calls a dead node (a tombstone, or a parent copy that a
//      split superseded) may be unlinked from its chain by a WRITER -- the
//      erase() that made it a tombstone, the split that superseded it, or a later
//      insert() or erase() whose walk passes it -- and retired. contains()'s own
//      walk never writes; a contains() that reaches an UNINITIALIZED bucket and
//      wins its split runs that split's cleanup like any other splitter. Where a dead node ends up is asserted only for operations that
//      ran UNCONTENDED, as POSTCONDITIONS of those operations: an uncontended
//      erase() leaves its own node unreachable unless the node's predecessor is
//      tagged and not dead; an uncontended split leaves no superseded parent copy
//      and no tombstone reachable in the parent behind an untagged predecessor;
//      an uncontended insert() or erase() walk leaves no dead run reachable
//      behind the bucket head or a live predecessor it passed. They are not
//      guarantees: under contention a dead node may stay in its chain (a
//      STRAGGLER) until a later writer walks past it or reclaim() runs, and no
//      test treats a straggler as a defect.
// Anything weaker than 1-4 (a "best effort" boolean, an under-counting insert)
// is a defect, not a documented relaxation -- see the header's "STALE GEOMETRY"
// section, which states 1 and 2 verbatim. The stragglers of 5 are the one
// documented relaxation, and they concern memory, never membership.
// ===========================================================================
#include "concurrent_hash_set_rcu.h"
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
#include <optional>
#include <cstdio>
#include <cstdint>

// ---------------------------------------------------------------------------
// Build identification: one line, printed before the first test, naming the
// compiler (__VERSION__), the sanitizer this binary was built with, and whether
// the header's assert()s are live, so that a saved run log says which build
// produced it. A gtest global environment rather than a main(), since main()
// comes from gtest_main; gtest allows registering one from a namespace-scope
// initializer, before RUN_ALL_TESTS() runs. GCC defines __SANITIZE_ADDRESS__ /
// __SANITIZE_THREAD__; clang answers __has_feature() (and recent versions define
// the GCC macros too).
// ---------------------------------------------------------------------------
#if defined(__SANITIZE_ADDRESS__)
#  define HASH_TEST_SANITIZER "AddressSanitizer"
#elif defined(__SANITIZE_THREAD__)
#  define HASH_TEST_SANITIZER "ThreadSanitizer"
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define HASH_TEST_SANITIZER "AddressSanitizer"
#  elif __has_feature(thread_sanitizer)
#    define HASH_TEST_SANITIZER "ThreadSanitizer"
#  endif
#endif
#ifndef HASH_TEST_SANITIZER
#  define HASH_TEST_SANITIZER "none"
#endif
#ifdef NDEBUG
#  define HASH_TEST_ASSERTS "off (NDEBUG)"
#else
#  define HASH_TEST_ASSERTS "live"
#endif

class BuildInfoEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::printf("[   INFO   ] compiler: %s; sanitizer: %s; assert(): %s\n", __VERSION__, HASH_TEST_SANITIZER, HASH_TEST_ASSERTS);
        std::fflush(stdout);
    }
}; // class BuildInfoEnvironment
// gtest takes ownership of the environment.
static ::testing::Environment* const build_info_environment = ::testing::AddGlobalTestEnvironment(new BuildInfoEnvironment);

// ---------------------------------------------------------------------------
// Key scrambler -- REQUIRED by every test whose point is to exercise resizes.
//
// libstdc++'s (and libc++'s) std::hash<int> is the identity, so with small
// consecutive keys 0..K a key stops changing bucket the moment table_size > K:
// bucket = hash(k) & (ts-1) == k for every larger ts. After that point a
// doubling MOVES NOTHING, no parent chain is ever split for those keys, and a
// test built on them is blind to every stale-geometry bug at every doubling
// beyond its key range.
//
// scramble() is a bijection on 32-bit ints (multiplication by an odd constant is
// invertible mod 2^32), so distinct indices still give distinct keys and every
// disjointness argument below is preserved -- but the high bits are pseudorandom,
// so each key has an independent 1/2 chance of moving at EVERY doubling.
// ---------------------------------------------------------------------------
static int scramble(int i) {
    return static_cast<int>(static_cast<unsigned>(i)*2654435761u);
}

// ---------------------------------------------------------------------------
// WHEN THE TABLE DOUBLES -- every test that places a resize relies on this.
//
// insert() doubles the table right after its publishing CAS succeeds if the
// resize hint node_count_ exceeds 2*table_size, and only if no other thread has
// doubled it since this insert loaded the size. The hint is not the arena size:
// a shard adds 256 when it appends its node 0, 256, 512, ... (header, on
// node_count_), so before any reclaim() the hint is the sum over the shards of
// their sizes rounded UP to a multiple of 256 (reclaim() resets it, and pops from
// the free lists then add 256 per 256 pops, the first pop included). That has two
// consequences for a test:
//   1. From the first allocation on, the hint is at least 256, so while the
//      table has fewer than 128 buckets EVERY successful insert doubles it (one
//      doubling per insert; under contention, an insert whose loaded size is
//      already stale does not). A fresh 4-bucket set has 8 buckets after its
//      first successful insert, 16 after its second, 128 after its fifth if they
//      run one at a time. Once a successful insert has returned, the set no
//      longer has 4 buckets: a test whose race must straddle the doubling 4 -> 8
//      has to make the set's FIRST successful insert part of the race, and a
//      set-up that inserts keys first has already doubled the table once per key.
//   2. From 128 buckets on, in a set with ONE arena shard (and no reclaim() yet)
//      the hint passes 2*table_size exactly when an allocation brings the arena
//      past 2*table_size slots: 2*table_size is then a multiple of 256, so "hint
//      > 2*table_size" is "arena size > 2*table_size". The doubling itself is done
//      by the next successful insert() whose loaded size is still current (a
//      split copy made inside contains() can cross the threshold without
//      doubling anything). With more shards, every shard that a new thread
//      touches adds 256 at once, and the table jumps by several doublings as the
//      threads start.
// Allocations include split copies, orphaned nodes and lost split subchains, so
// rule 2 counts arena slots, not keys.
// ---------------------------------------------------------------------------

// Run fn(t) on T threads released together by a barrier, then join them all.
// The barrier (rather than a spin on a start flag) is what makes the racing
// window open at the same instant on every thread.
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
// worker joined). The sweep calls Hash{} on every reachable node that is not
// MARKED, so a test that arms ProbeKey's hooks disarms them before it calls any
// of these (see the hooks at ProbeKey).
// ---------------------------------------------------------------------------

// Accounting sweep: every arena slot must be reachable from a published bucket,
// on a free list, on a limbo list, or on a retired list, exactly once; the four
// counts must sum to the slot count; every reachable dead node must be tagged
// (MARKED or FROZEN); and every retired node must be tagged with its chain link
// intact. All of that is `consistent` (the header's C1, C2, C3); the sum is
// asserted separately only for its clearer failure message. No accessor is
// cross-checked here: get_internal_free_count(), get_internal_limbo_count() and
// get_internal_retired_count() return fields of this same sweep, and `slots` is
// get_internal_node_count() by construction. `where` names the phase in failure
// messages. Returns the sweep for further, test-specific checks.
template <typename SetT>
static typename SetT::InternalAccounting expect_consistent(const SetT& set, const std::string& where) {
    const typename SetT::InternalAccounting acc = set.get_internal_accounting();
    EXPECT_TRUE(acc.consistent) << where << ": some arena slot is unaccounted for or on two lists, a dead node is untagged,"
                                << " or a retired node is untagged or lost its chain link (slots " << acc.slots
                                << ", reachable " << acc.reachable << " of them dead " << acc.reachable_dead
                                << ", free " << acc.free_nodes << ", limbo " << acc.limbo << ", retired " << acc.retired << ")";
    EXPECT_EQ(acc.reachable + acc.free_nodes + acc.limbo + acc.retired, acc.slots) << where;
    return acc;
} // expect_consistent()

// reclaim() with every postcondition its contract promises, against the oracle
// `expected_live` (the number of keys in the set, known to the test):
//   - before it, the sweep must be consistent, and that is ASSERTed: a node
//     retired twice makes a cycle in a retired list, which the sweep reports and
//     stops at, but which reclaim()'s drain would walk forever in a build
//     without assert();
//   - the return value is exact, and it is the number of reachable nodes the
//     sweep before it found not dead (the membership cross-check: exactly one
//     non-dead reachable node per key at any quiescent point);
//   - reclaim() neither grows nor shrinks the arena, unlinks exactly the
//     reachable dead nodes and nothing else, and afterwards exactly one node per
//     key is reachable and none of them dead;
//   - the limbo and retired lists are empty afterwards, and the free lists grew
//     by exactly what reclaim() collected: the limbo nodes, the retired nodes and
//     the dead nodes its chain walk unlinked.
// Not size_t-returning, so that it can ASSERT: callers wrap it in
// ASSERT_NO_FATAL_FAILURE. `live_out`, if given, receives reclaim()'s return
// value.
template <typename SetT>
static void reclaim_and_check(SetT& set, size_t expected_live, const std::string& where, size_t* live_out = nullptr) {
    const typename SetT::InternalAccounting before = expect_consistent(set, where + ", before reclaim()");
    ASSERT_TRUE(before.consistent) << where << ": reclaim() not called on an inconsistent set (a retired-list cycle would hang its drain)";
    const size_t live = set.reclaim();
    if (live_out != nullptr) *live_out = live;
    EXPECT_EQ(live, expected_live) << where << ": reclaim() must return the exact number of keys";
    EXPECT_EQ(live, before.reachable - before.reachable_dead) << where << ": reclaim()'s count disagrees with the reachable nodes the sweep found not dead";
    const typename SetT::InternalAccounting after = expect_consistent(set, where + ", after reclaim()");
    EXPECT_EQ(after.slots, before.slots) << where << ": reclaim() must not change the arena size";
    EXPECT_EQ(after.reachable, before.reachable - before.reachable_dead) << where << ": reclaim() must unlink exactly the dead nodes";
    EXPECT_EQ(after.reachable, live) << where << ": more than one reachable node for some key after reclaim()";
    EXPECT_EQ(after.reachable_dead, 0u) << where << ": a dead node is still reachable after reclaim()";
    EXPECT_EQ(after.limbo, 0u) << where << ": reclaim() must drain every limbo list";
    EXPECT_EQ(after.retired, 0u) << where << ": reclaim() must drain every retired list";
    EXPECT_EQ(after.free_nodes, before.free_nodes + before.limbo + before.retired + before.reachable_dead)
        << where << ": the free lists did not grow by exactly the limbo, retired and dead nodes";
} // reclaim_and_check()

// settle_writers(set): a writer MISS walk over every published bucket, so that
// every dead node a CONCURRENT history left in a chain (a straggler: a one-shot
// unlink CAS that lost, an erase() that gave up, a run a walker with a stale
// table size could not decide) is unlinked and retired, as the next writer to
// pass it would do. Afterwards the eager oracles hold that a concurrent history
// alone does not promise: no reachable dead node, exactly one reachable node per
// key, every other slot retired, on a limbo list or free. How: for every
// published bucket j of the current table, erase() one ABSENT key that maps to
// exactly j under the identity hash -- the first of j, j + ts, j + 2*ts, ... that
// contains() reports absent --, so that the erase() walks the whole chain (a
// miss) and unlinks every dead run it passes behind the head or a live
// predecessor. Every such erase() must return false.
// It skips the pending (UNINITIALIZED) buckets, asking the header's test-only
// get_internal_bucket_published(): a contains() or erase() there would split the
// bucket, and that split's cleanup walks the parent chain and unlinks its dead
// runs itself, so the settling would succeed even if the writer walks it exists
// to exercise unlinked nothing. Skipping them also leaves the geometry as the
// history left it: settling splits nothing.
// Requirements: quiescent and single-threaded; ProbeKey hooks disarmed; an
// AllowDelete == true set of int keys whose hash is the identity (std::hash<int>
// here, so j + m*ts maps to j; not StringKeys); and no dead run behind a FROZEN
// node whose child is unpublished (a split that threw between its freeze and its
// publication): such a run is behind a tagged predecessor, which no walk may
// CAS, and only a split of the child or reclaim() collects it.
// Not an oracle in itself: on a set with a concurrent history it is not a no-op.
template <typename SetT>
static void settle_writers(SetT& set) {
    const size_t ts = set.get_internal_table_size();
    for (size_t j = 0; j < ts; ++j) {
        if (!set.get_internal_bucket_published(j)) continue;
        bool erased_in_j = false;
        for (size_t m = 0; m < 64 && !erased_in_j; ++m) {
            const size_t k = j + m*ts;
            if (k > static_cast<size_t>(INT32_MAX)) break;
            if (set.contains(static_cast<int>(k))) continue;   // present: not a miss walk
            ASSERT_FALSE(set.erase(static_cast<int>(k))) << "settle_writers(): erase() of an absent key returned true";
            erased_in_j = true;
        } // look for an absent key of bucket j
        ASSERT_TRUE(erased_in_j) << "settle_writers(): no absent key found for bucket " << j;
    } // loop over the buckets
} // settle_writers()

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
TEST(ConcurrentHashSetRcuTest, BasicOperations) {
    ConcurrentResizableHashSetRCU<int> set;
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
TEST(ConcurrentHashSetRcuTest, ArenaShardsParameter) {
    ConcurrentResizableHashSetRCU<int> by_default;
    EXPECT_GE(by_default.get_internal_arena_shards(), 1u);
    EXPECT_TRUE(std::has_single_bit(by_default.get_internal_arena_shards()));
    ConcurrentResizableHashSetRCU<int> three(4, 3);
    EXPECT_EQ(three.get_internal_arena_shards(), size_t(4));

    const int T = 4, N = 2000;
    for (size_t shards : {size_t(1), size_t(2), size_t(64)}) {
        ConcurrentResizableHashSetRCU<int> set(4, shards);
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
TEST(ConcurrentHashSetRcuTest, ManyResizesSingleThread) {
    ConcurrentResizableHashSetRCU<int> set(4);
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
// An upper bound alone, EXPECT_LE(inserted, T*per), would be VACUOUS -- T*per is
// also the number of insert() calls made, so the count cannot exceed it no matter
// what the implementation does. Only equality has any content here, and equality
// is what the contract says.
TEST(ConcurrentHashSetRcuTest, ConcurrentDisjointRanges) {
    ConcurrentResizableHashSetRCU<int> set(4);
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
// The total count catches a double winner (one true too many). It is asserted
// exactly, not as an upper bound: the contract gives no licence for a
// "best-effort" boolean that may under-count under concurrent resizes, and an
// under-count is unreachable anyway (a thread that scans and misses either wins
// its publishing CAS, or loses it and retries until it wins or finds the winner's
// node, so a winner always exists).
// ExactlyOneWinnerPerKey below strengthens this further: a total count alone
// cannot tell "one winner each" from "key A twice and key B never".
TEST(ConcurrentHashSetRcuTest, ConcurrentSameKeyMembership) {
    for (int rep = 0; rep < 10; ++rep) {
        ConcurrentResizableHashSetRCU<int> set(4);
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

TEST(ConcurrentHashSetRcuTest, CollisionHeavyChains) {
    ConcurrentResizableHashSetRCU<int, false, ModHash> set(4);
    const int N = 5000;
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(i));
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.contains(i));
    EXPECT_FALSE(set.contains(N + 1));
}

// Non-trivial key type to confirm the container is not hard-wired to integers.
TEST(ConcurrentHashSetRcuTest, StringKeys) {
    ConcurrentResizableHashSetRCU<std::string> set(4);
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
TEST(ConcurrentHashSetRcuTest, ConcurrentReadersWriters) {
    ConcurrentResizableHashSetRCU<int> set(4);
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
TEST(ConcurrentHashSetRcuTest, SplitContention) {
    // Many fresh sets, not one: the splits of buckets 4..7 race only while the
    // table is small, and it grows out of that within the threads' first few
    // inserts (rule 1 of "WHEN THE TABLE DOUBLES"), so each repetition buys one
    // more such window.
    const int REPS = 8;
    for (int rep = 0; rep < REPS; ++rep) {
        // Initial size 4, and ONE insert before the threads start: by rule 1 that
        // insert doubles the table to 8, so buckets 4..7 are published
        // UNINITIALIZED (their encoded value UNINITIALIZED, distinct from a real
        // address) and will be split lazily by whichever thread below touches them
        // first. Any further insert here would double the table again (to 16, 32,
        // ...) before the threads start. Key 100 is in bucket 0 at size 4 and in
        // bucket 4 at size 8, so the split of bucket 4 has a node to copy.
        ConcurrentResizableHashSetRCU<int, false, CollisionHash> set(4);
        set.insert(100);

        // 8 threads aggressively inserting elements that hash to 4, 5, 6, 7. Every
        // thread's first key (k = 0) maps to bucket 4 (i*10000 is a multiple of
        // 16), so all eight start in the split of the same bucket; their own
        // inserts keep doubling the table from there (rule 1, then the shards they
        // touch).
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
                EXPECT_TRUE(set.contains(4 + 8*k + i*10000)) << "rep " << rep;
                EXPECT_TRUE(set.contains(5 + 8*k + i*10000)) << "rep " << rep;
                EXPECT_TRUE(set.contains(6 + 8*k + i*10000)) << "rep " << rep;
                EXPECT_TRUE(set.contains(7 + 8*k + i*10000)) << "rep " << rep;
            } // key loop
        } // membership sweep
        EXPECT_TRUE(set.contains(100)) << "the key the first split copied was lost, rep " << rep;
    } // repetitions
} // SplitContention

// Exercises tombstone erase end-to-end: single-threaded correctness (erase makes
// contains() return false; a second erase of the same key returns false; erase
// does not disturb other keys; a key can be re-inserted after erase), then a
// concurrent phase where each thread erases its own disjoint key range so the
// MARK_BIT CAS and the erase/contains interleavings are stressed without two
// threads ever contending for the same key. AllowDelete=true compiles erase().
// The concurrent phase erases keys that are present and that no other thread
// touches, so EVERY erase() there must return true -- an exact oracle for free.
// NOTE ON SCOPE: the concurrent phase performs no inserts, so the table does not
// double while it runs. It is a MARK_BIT contention test, not a stale-geometry
// test; ExactlyOneEraseWinnerPerKeyDuringGrowth below covers erase under resize.
TEST(ConcurrentHashSetRcuTest, TombstoneErase) {
    ConcurrentResizableHashSetRCU<int, true> set(4);

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
// subchains abandoned). No deletes here (AllowDelete=false), so the only dead
// nodes are the parent copies the splits supersede; the split still freezes them
// (for both AllowDelete values), and the winner's cleanup and the inserters'
// walks unlink and retire them while other threads insert into, seal and walk
// the same parent chains. The final EXPECT_EQ also checks that every attempted
// key is present -- membership must be exact across every resize.
// Racing in the same split takes threads that work in the same bucket WHEN it
// splits, and with CollisionHash the bucket of a key is its low bits. By rule 1 of
// "WHEN THE TABLE DOUBLES" the table passes 16, 32 and 64 buckets within the
// threads' first few inserts, while the chains are still short, and it spends
// most of the run far larger; so the keys of different threads differ only at
// bit 16 and above: the i-th key of every thread is in the same bucket at every
// table size this test reaches, the threads (released together) fill the same
// chains, and each doubling has them split the same buckets with real chains to
// copy. Bits 3..12 (the index i) make the keys move at the doublings; bits 0..2
// are 5, so at size 8 every key is in bucket 5.
TEST(ConcurrentHashSetRcuTest, CooperativeSplitTSAN) {
    ConcurrentResizableHashSetRCU<int, false, CollisionHash> set(4);

    // Exactly one resize (N=4 -> N=8) before the threads start: by rule 1 of "WHEN
    // THE TABLE DOUBLES" the set's first insert doubles it, and a second one would
    // double it again. Key 0 stays in bucket 0, so buckets 4..7 are UNINITIALIZED
    // when the threads are released.
    set.insert(0);

    constexpr int NUM_THREADS = 8;
    constexpr int INSERTS_PER_THREAD = 1000;
    // The i-th key of thread t (see above); distinct for every (t, i).
    auto key_of = [](int t, int i) { return 5 + 8*i + (t << 16); };

    run_threads(NUM_THREADS, [&set, &key_of](int t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            set.insert(key_of(t, i));
        } // insert loop
    });

    int success_count = 0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            if (set.contains(key_of(t, i))) {
                success_count++;
            }
        }
    } // membership sweep
    EXPECT_EQ(success_count, NUM_THREADS*INSERTS_PER_THREAD);
} // CooperativeSplitTSAN

// TSAN stress test specifically for erase(). Unlike TombstoneErase this phase DOES
// insert, so the arena keeps growing and the table keeps doubling underneath the
// erasers. Every key is private to one thread, so every return value in the
// sequence below is fully determined by the contract and is asserted exactly.
// Keys are scrambled because the pre-population alone takes the table past the
// consecutive key range, after which unscrambled keys would stop moving.
TEST(ConcurrentHashSetRcuTest, EraseStressTSAN) {
    ConcurrentResizableHashSetRCU<int, true> set(4);
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
// exactly while t2 doubles the table to 8 and then splits the new bucket 4. If
// that split snapshots bucket 0's chain in the instant BEFORE t1's node is
// linked, the copy pass misses key 4 -- and key 4 would be stranded in a bucket
// no reader consults at size 8. What closes the window is that the split SEALS
// bucket 0's head (raising its level by CAS) before snapshotting it: seal and
// publication are read-modify-writes of the same word, so either t1's node is in
// the snapshot or t1's publishing CAS fails and it retries in the new geometry.
// contains(4) must therefore hold afterwards, and t1's insert(4), the only
// insert of that key, must have reported true.
// Set-up: the set starts EMPTY, because by rule 1 of "WHEN THE TABLE DOUBLES"
// its first successful insert is the one that doubles it to 8; a key inserted
// beforehand would have moved the doubling out of the race. t2 inserts key 1
// (bucket 1 at every size, so it touches neither head involved), which performs
// the doubling 4 -> 8 unless t1 publishes first, and then looks up key 12, which
// maps to bucket 4 at size 8 (at size 16 it maps to bucket 12, whose split splits
// its parent, bucket 4, first): that lookup is the split of bucket 4 the race
// needs. No third operation touches bucket 4 during the race (t1's own retry
// may).
TEST(ConcurrentHashSetRcuTest, LostUpdateOnResize) {
    for (int rep = 0; rep < 10000; ++rep) {
        ConcurrentResizableHashSetRCU<int, false, CollisionHash> set(4);

        std::atomic<bool> start{false};
        bool t1_inserted = false;

        std::thread t1([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            // Insert 4 which hashes to 0 initially (4 & 3 == 0)
            t1_inserted = set.insert(4);
        });

        std::thread t2([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            set.insert(1);      // the set's first insert, as a rule: doubles 4 -> 8
            set.contains(12);   // splits bucket 4 from bucket 0's chain
        });

        start.store(true, std::memory_order_release);
        t1.join();
        t2.join();

        ASSERT_TRUE(t1_inserted) << "insert(4) of an absent, uncontended key returned false in rep " << rep;
        ASSERT_TRUE(set.contains(4)) << "Lost update in rep " << rep;
    } // repetitions
} // LostUpdateOnResize

// Degenerate hash: every key maps to bucket 0. This funnels all inserts through
// a single bucket head, maximizing compare_exchange contention -- precisely the
// condition under which allocating a fresh node on every CAS retry would leak a
// node per failed attempt. It makes the leak-detection test below sensitive by
// construction.
struct ConstantHash {
    size_t operator()(int) const { return 0; }
};

TEST(ConcurrentHashSetRcuTest, InsertContention_NoMemoryLeak) {
    ConcurrentResizableHashSetRCU<int, false, ConstantHash> set(4);
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
// Return-value exactness under concurrent resizes. The tests below guard
// against four defects that a geometry check outside the deciding CAS lets
// through: double-true insert, erase undone by a stale inserter, double-true
// erase, and a key lost through a doubling.
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
// summed after the join. The two AllowDelete flavors alternate because they are
// two instantiations of the header: their insert and split code is the same
// source (the split freezes the nodes it moves for both values), compiled
// twice, and the accounting after the race checks for each that every
// superseded parent copy still reachable is tagged.
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
// pending splits with a contains() sweep (whose splits supersede the parent copy
// of every key that moves, and retire it), erases them again if the set has
// erase(), and reclaims, so the race starts with non-empty free lists: its
// allocations then POP while its orphans and lost subchains are pushed to limbo,
// on the same shard when `shards` is 1 (see ReclaimLimboPushesRaceFreeListPops).
// Without erase() the prefill keys stay in the set and the free lists hold only
// those superseded copies, which is why the sweep is there, and why the prefill
// must cross doublings AFTER keys exist (see the prefill sizes in
// ReclaimLimboPushesRaceFreeListPops). `min_free` is the floor the free lists
// must reach after the prefill reclaim(). `limbo` receives the number of nodes
// that were on limbo lists before the race's reclaim(). Callers wrap it in
// ASSERT_NO_FATAL_FAILURE (it ASSERTs through reclaim_and_check()); its template
// argument goes through an alias there, since a comma inside the macro's
// argument would split it.
template <typename SetT>
static void one_winner_per_key_body(int T, int N, int rep, const char* flavor, size_t& limbo, size_t shards = 0, int prefill = 0, size_t min_free = 0) {
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, prefilled, std::string("prefill, rep ") + std::to_string(rep)));
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
    limbo = expect_consistent(set, where + ", after the insert race").limbo;
    ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, expected, where));
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
} // one_winner_per_key_body()

// The two instantiations the reclamation tests alternate between.
using IntSetDel = ConcurrentResizableHashSetRCU<int, true>;
using IntSetNoDel = ConcurrentResizableHashSetRCU<int, false>;

TEST(ConcurrentHashSetRcuTest, ExactlyOneWinnerPerKey) {
    const int T = 16, N = 400;
    for (int rep = 0; rep < 60; ++rep) {
        size_t limbo = 0;   // not used here
        if (rep & 1) {
            ASSERT_NO_FATAL_FAILURE(one_winner_per_key_body<IntSetDel>(T, N, rep, "AllowDelete=true", limbo));
        } else {
            ASSERT_NO_FATAL_FAILURE(one_winner_per_key_body<IntSetNoDel>(T, N, rep, "AllowDelete=false", limbo));
        }
    } // repetitions
} // ExactlyOneWinnerPerKey

// TARGETED regression test for the double-winner interleaving: it builds the
// geometry by hand instead of waiting for it to occur by chance, so it reproduces
// in milliseconds on any machine and in any build. (The statistical tests above
// find the same defect, but their hit rate swings by two orders of magnitude with
// the build -- an instrumented -O0 build hits it far more often than an -O3 one
// -- so on some machines and some builds they can look green.)
//
// Geometry, with CollisionHash so the bucket arithmetic is exact: key 4 is in
// bucket 0 at table size 4 (4 & 3 == 0) and in bucket 4 at size 8 (4 & 7 == 4),
// so the doubling 4 -> 8 MOVES it. Every attempt starts from a fresh, EMPTY
// 4-bucket set, because by rule 1 of "WHEN THE TABLE DOUBLES" the set's first
// successful insert is the one that doubles it to 8: a key inserted while
// building the set would have doubled it already. Role 1 inserts key 1, which
// lives in bucket 1 at every size, so the resize-triggering insert never touches
// bucket 0's or bucket 4's head. If one of the two inserts of key 4 publishes
// first, it doubles the table itself and the other finds its node: a correct,
// race-free attempt. A second doubling (8 -> 16, by the next successful insert
// that saw size 8) does not move key 4.
//
// Threads are persistent and meet at a barrier twice per attempt (once to start
// the attempt on a freshly built set, once to hand the result back): spawning
// three threads per attempt would cost more than the attempt itself and would put
// the thread-creation latency, not the race, in the window.
TEST(ConcurrentHashSetRcuTest, NoDoubleWinnerAcrossResize) {
    using Set = ConcurrentResizableHashSetRCU<int, false, CollisionHash>;
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
                    set->insert(1);          // the set's first insert, as a rule: doubles 4 -> 8
                } else if (set->insert(KEY)) {
                    winners.fetch_add(1, std::memory_order_relaxed);
                }
                round.arrive_and_wait();     // results are safe to read
            } // attempt loop
        });
    } // spawn workers

    int bad = 0, first_bad = -1, missing = 0;
    for (int a = 0; a < ATTEMPTS; ++a) {
        set = std::make_unique<Set>(4);   // empty: see "Geometry" above
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
// Shape, and why it is sized the way it is. Each round is a TINY, fresh set with
// 16 pre-inserted shared keys. Every thread erases every shared key, so both
// contention modes occur -- mark versus mark (two erasers on one link) and mark
// versus freeze (an eraser and a splitter on one link) -- and the threads start at
// staggered positions in the key order so that a thread which loaded table_size_
// before a doubling is still walking when another thread has moved on. Between
// erases each thread inserts private filler keys, which drive the doublings.
// Where those doublings fall is set by WHEN THE TABLE DOUBLES: inserting the
// shared keys has already taken the table to 128 buckets (rule 1), and from there
// the resize hint grows by 256 whenever a thread's first allocation touches its
// own arena shard, which is what takes the table through 256, 512 and 1024
// buckets while the erases run, and to 2048 in the rounds in which no worker
// shares the main thread's shard (a worker that does adds nothing to the hint:
// thread numbers are assigned at a thread's first allocation, so each round's
// workers take the next block of numbers and, modulo the shard count, alternate
// between blocks that include the main thread's shard and blocks that do not).
// The doubling itself is done by whichever thread next completes an insert. If
// every thread allocated from its first step on, those doublings would all come
// together in the first few operations of the round; so thread t inserts its
// first filler only at step t*K/T (its fillers are spread over the remaining
// steps). The stagger is partial: a thread's first erase() often splits an
// UNINITIALIZED bucket, and the split's copy is an allocation made before its
// fillers start. It spreads the doublings over the first part of the round
// instead of its first few operations.
// The set has 2*T arena shards, so that every worker gets a shard no other worker
// uses, on any machine (the default shard count follows the hardware
// concurrency).
// What decides whether this test can see anything is how many doublings land
// inside erasers' windows, because the defect needs a split to copy a node inside
// some eraser's load-table_size_-to-mark-CAS window. Doublings are logarithmic in
// arena size, so a big set is the worst possible shape: its doublings are few and
// far between the erases. A small fresh set per round, with its doublings spread
// through the round's erases, in many cheap rounds instead of a few expensive
// ones, is what makes a header without the freeze fail many times per run. Prefer
// many tiny sets to one big one for anything that has to race with a resize.
struct StallingKey {
    int v;
    StallingKey(int x = 0) : v(x) {}
    // Equality that briefly stalls on every 8th MATCHING comparison made by the
    // calling thread. WHY this is here: erase() loads the node's link, THEN
    // compares the key, THEN CASes MARK onto that same link. The comparison sits
    // inside the window in which a concurrent split can FREEZE the link out from
    // under the eraser -- with plain ints the window is a few nanoseconds wide and
    // the mark-versus-freeze race is correspondingly rare. The stall makes a
    // header without the freeze fail several times as often under ASan (TSan's
    // own instrumentation already stretches the window, so it gains little
    // there). The extra ASan margin is worth the fraction of a second it costs,
    // because ASan is the build in which this test is closest to reporting a
    // broken header as green. It changes nothing about WHAT is tested:
    // operator== still answers correctly, no header code is touched, the oracle
    // is unchanged, and a broken header fails the test without the stall too --
    // just with less room to spare.
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

TEST(ConcurrentHashSetRcuTest, ExactlyOneEraseWinnerPerKeyDuringGrowth) {
    using Set = ConcurrentResizableHashSetRCU<StallingKey, true, StallingKeyHash>;
    const int T = 8, K = 16, ROUNDS = 400, FILL = 64;
    int over = 0, under = 0, still_present = 0, filler_dup = 0, filler_missing = 0;

    for (int r = 0; r < ROUNDS; ++r) {
        Set set(4, 2*T);
        for (int k = 0; k < K; ++k) set.insert(scramble(k));

        std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(K, 0));
        std::vector<int> dup(T, 0);
        run_threads(T, [&set, &won, &dup](int t) {
            for (int i = 0; i < K; ++i) {
                const int k = (i + t*K/T)%K;     // staggered start per thread
                if (set.erase(scramble(k))) ++won[t][k];
                // Filler inserts, spread evenly through the erases from step s0 on
                // (see "Shape" above): these are what grow the arena and double the
                // table mid-flight. Each key belongs to one thread, so a false
                // return is a defect in itself.
                const int s0 = t*K/T;      // this thread's first filler step
                const int steps = K - s0;  // steps its fillers are spread over
                if (i >= s0) {
                    for (int f = (i - s0)*FILL/steps; f < (i - s0 + 1)*FILL/steps; ++f) {
                        if (!set.insert(scramble(K + t*FILL + f))) ++dup[t];
                    }
                } // if this thread's fillers have started
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
// thousands of nodes and drives the table from 4 to 2048 buckets.
// Where those doublings fall is set by WHEN THE TABLE DOUBLES: the first few
// successful inserts take the table to 128 buckets (rule 1), and from there the
// resize hint grows by 256 whenever a thread's first allocation touches its own
// arena shard, which is what takes the table on to 1024 buckets, and the shards'
// later 256-node batches take it to 2048. If every thread inserted from its first
// operation on, those doublings would all come together in the first few dozen
// operations and the rest of the churn would run at one size; so thread t turns
// its inserts into lookups for its first t*OPS/(2*T) operations, which staggers
// the threads' first allocations (a lookup or an erase allocates too when it
// splits a bucket, so the stagger is partial) and spreads those doublings over
// the first part of the churn. The set has 2*T arena shards, so that every thread
// gets a shard of its own on any machine.
TEST(ConcurrentHashSetRcuTest, InsertEraseChurnPerKeyAccounting) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int T = 8, R = 48, REPS = 60, OPS = 1600;

    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4, 2*T);
        std::vector<std::vector<int>> ins(T, std::vector<int>(R, 0));
        std::vector<std::vector<int>> ers(T, std::vector<int>(R, 0));

        run_threads(T, [&set, &ins, &ers, rep](int t) {
            std::minstd_rand rng(static_cast<unsigned>(rep*977 + t + 1));
            for (int op = 0; op < OPS; ++op) {
                const unsigned x = rng();
                const int i = static_cast<int>((x >> 4)%static_cast<unsigned>(R));
                const int k = scramble(i + 1);
                // Before this thread's start point its inserts are lookups (see the
                // comment above the test): kind 3 is the lookup.
                const bool started = op >= t*OPS/(2*T);
                const unsigned kind = (started || (x & 3u) >= 2) ? (x & 3u) : 3u;
                switch (kind) {
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
// The weak spot: a split COPIES live nodes forward and supersedes the originals
// in place, so for a while a key has a second node in an ancestor bucket (until
// the split's cleanup, a later writer or reclaim() unlinks it). If an erase marks
// only the authoritative copy and some later doubling then copies the stale node
// into a new bucket, the erased key comes back from the dead -- with no insert()
// call anywhere. Two things must hold for that to be impossible: no stale live
// node may exist for a key that was published under a sealed head, and any node a
// split copies must be frozen first so a concurrent erase cannot leave a live
// copy behind.
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
// HONESTY ABOUT WHAT THIS TEST CATCHES. The resurrection oracle (phase 3) is exact,
// but no known defect makes it FIRE: on a header with the geometry check outside
// the deciding CAS, or without the seal, the resurrection counters stay at zero and
// what fails is the phase-1 exactly-one-winner check. The split arithmetic says
// why: the stale node such a header strands in an old bucket j can only be copied
// forward when bucket j's CHILD is split, and by the time the strand happens that
// child has long since been published -- so the stranded node is dead weight and
// never resurrects. In other words this test guards a property that no known defect
// violates. It is kept because the property is load-bearing (copying instead of
// moving rests on it), the check is exact and costs half a second, and a future
// change to split_bucket() -- re-splitting a published bucket, copying an
// unfrozen node, unlinking a FROZEN node before its child is published -- would
// break it first. Do not count
// it as a detector for the four known defects (a header without the freeze passes
// it); NoDoubleWinnerAcrossResize and ExactlyOneEraseWinnerPerKeyDuringGrowth are
// the detectors.
TEST(ConcurrentHashSetRcuTest, EraseNotUndoneByResize) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
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
// own successful erase are the point: a contains() sweep after every thread has
// joined comes too late, as any self-repair the implementation performs has long
// since run by then. This checks what a caller actually relies on -- the
// instant insert(k) returns true, k is visible to that caller; the instant erase(k)
// returns true, it is not.
// The other threads' inserts keep the table doubling throughout, so each of these
// sequences straddles resizes; keys are scrambled so they keep moving across them.
TEST(ConcurrentHashSetRcuTest, ThreadPrivateKeySequenceDuringGrowth) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
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
// alloc_node(), the per-shard limbo lists, and the per-shard retired lists that
// dead nodes reach when an operation unlinks them from their chain. Contract
// clauses 4 and 5 at the top of this file; the header's RECLAMATION section and
// the contract on reclaim().
//
// Every call to reclaim() and to the test-only sweeps below happens after
// run_threads() has joined its workers (or on the only thread there is), which
// is the quiescence the contract requires. Oracles are the contract's: the key
// count the test itself knows, the accounting identity (every slot in exactly one
// place), and "allocation takes a free slot before it grows the arena". Where a
// test relies on the documented node_count_ batching (256 per batch), it says so.
// ===========================================================================

// Pre-sized (no doubling), so every node the arena holds is a key or an erased
// node and every count is exact. One arena shard: reclaim() deals freed slots
// round-robin over ALL shards, and a thread pops only its own shard's list, so
// with the default shard count the main thread could reuse only 1/shards of what
// it freed and "the arena did not grow" would be false by design, not by defect
// (ReclaimReuseAcrossShards covers the multi-shard dealing).
// Weak spots: the drain of the retired lists. Every erase() here is uncontended,
// so it unlinks its own node at once and retires it (contract clause 5): the
// erased nodes reach reclaim() on a retired list, not in a chain, reclaim()'s
// chain walk finds nothing dead, and the K slots reach the free list only through
// the drain's two phases (take every shard's retired list, then deal it). "free
// == K" afterwards fails if the drain is skipped or follows the wrong word.
// Before the first reclaim() the erased nodes are counted where they are --
// retired, or still in a chain as dead nodes --, K in total whoever unlinked
// them. Also: reclaim() run twice with nothing to do in between (it must be
// idempotent: same count, nothing new freed, no slot pushed twice -- a double
// push would show as a cycle or a duplicate in the accounting sweep); and the
// free-list pop path serving every reinsertion, including the one that empties
// the list.
// Sizes: N = 1000 keys in 1024 buckets. The resize hint then peaks at 1024 (four
// append batches of 256) before reclaim() and at 750 + 24 (append credit) + 256
// (first pop batch) after it, both below the doubling threshold 2*1024.
TEST(ConcurrentHashSetRcuTest, ReclaimExactPreSized) {
    const int N = 1000, K = 250;
    {   // AllowDelete == true: the erased nodes are the only dead nodes
        ConcurrentResizableHashSetRCU<int, true> set(1024, 1);
        std::vector<int> kept, erased, fresh;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(scramble(i)));
        const ConcurrentResizableHashSetRCU<int, true>::InternalAccounting filled = expect_consistent(set, "after the fill");
        EXPECT_EQ(filled.slots, size_t(N));
        EXPECT_EQ(filled.reachable, size_t(N));
        for (int i = 0; i < N; ++i) {
            if (i%4 == 1) {   // every 4th key: the erased nodes sit anywhere in their chains
                EXPECT_TRUE(set.erase(scramble(i)));
                erased.push_back(scramble(i));
            } else {
                kept.push_back(scramble(i));
            }
        } // erase every 4th key
        ASSERT_EQ(erased.size(), size_t(K));
        const ConcurrentResizableHashSetRCU<int, true>::InternalAccounting before = expect_consistent(set, "after the erases");
        EXPECT_EQ(before.retired + before.reachable_dead, size_t(K)) << "an erased node is neither retired nor dead in its chain";
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N - K, "first reclaim()"));
        EXPECT_EQ(set.get_internal_free_count(), size_t(K)) << "every erased node, and nothing else, is freed";
        EXPECT_EQ(membership_errors(set, kept, erased), 0);

        // Idempotence: nothing died since, so nothing more is freed.
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N - K, "second reclaim()"));
        EXPECT_EQ(set.get_internal_free_count(), size_t(K)) << "a second reclaim() freed something, or pushed a slot twice";

        // Reinsert K new keys: every one must be served from the free list.
        for (int i = 0; i < K; ++i) {
            fresh.push_back(scramble(N + i));
            EXPECT_TRUE(set.insert(fresh.back()));
        }
        EXPECT_EQ(set.get_internal_node_count(), size_t(N)) << "the arena grew although the free list held a node for every insert";
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        expect_consistent(set, "after the reinsertion");
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N, "third reclaim()"));
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        EXPECT_EQ(membership_errors(set, kept, erased), 0);
        EXPECT_EQ(membership_errors(set, fresh, std::vector<int>{}), 0);
    } // AllowDelete == true
    {   // AllowDelete == false: pre-sized, so nothing is ever dead
        ConcurrentResizableHashSetRCU<int, false> set(1024, 1);
        std::vector<int> keys, absent;
        for (int i = 0; i < N; ++i) {
            keys.push_back(scramble(i));
            EXPECT_TRUE(set.insert(keys.back()));
            absent.push_back(scramble(N + i));
        }
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N, "AllowDelete=false, first reclaim()"));
        EXPECT_EQ(set.get_internal_free_count(), 0u) << "reclaim() freed a live node";
        for (int k : keys) EXPECT_FALSE(set.insert(k));   // duplicates: no allocation, no change
        EXPECT_EQ(set.get_internal_node_count(), size_t(N));
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N, "AllowDelete=false, second reclaim()"));
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
// the dead nodes are the parent copies splits supersede, with true also the
// erased nodes. The burst is single-threaded, so the operations themselves unlink
// and retire all of those (contract clause 5) and reclaim() collects them from the
// retired lists; what its chain walk must decide is the live nodes whose child
// is still UNINITIALIZED, which it must keep.
// The reuse phase has a trap: a reinsertion can land in a bucket that is still
// UNINITIALIZED and split it, which allocates copies. So the node-count baseline
// is taken only after a contains() sweep over every key, which publishes every
// bucket any existing key maps to (a split of any other bucket copies nothing).
// Even then a reinsertion can double the table and split again, so the oracle is
// the design rule itself: with one shard and one thread, the arena grows only
// once the free list is empty.
template <bool AllowDelete>
static void reclaim_with_pending_splits_body() {
    using Set = ConcurrentResizableHashSetRCU<int, AllowDelete>;
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), where + ", reclaim() right after the burst"));
        const size_t free1 = set.get_internal_free_count();
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), where + ", second reclaim()"));
        EXPECT_EQ(set.get_internal_free_count(), free1) << where << ": the second reclaim() freed something new";
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": reclaim() changed membership";

        // Reuse, with the baseline taken after the contains() sweep just done. That
        // sweep's splits popped the free list (with AllowDelete == false it is
        // usually empty afterwards) and retired the parent copies they superseded;
        // a reclaim() now recycles those, so the reuse below has free slots to take.
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), where + ", reclaim() after the settling sweep"));
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), where + ", reclaim() after the reuse"));
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": membership after the reuse";
    } // for each burst size
    // Not an oracle on the header: a guard that the reuse check above is not vacuous.
    EXPECT_GT(reuse_exercised, 0) << "no size exercised reuse with free nodes left over";
} // reclaim_with_pending_splits_body()

TEST(ConcurrentHashSetRcuTest, ReclaimWithPendingSplits) {
    reclaim_with_pending_splits_body<true>();
    reclaim_with_pending_splits_body<false>();
} // ReclaimWithPendingSplits

// AllowDelete == false, concurrent growth: the dead nodes are the parent copies
// splits supersede. The split FREEZES each of them (for both AllowDelete values),
// and the split that superseded it unlinks and retires it -- or, when that
// split's one cleanup CAS lost to a concurrent writer, a later writer or
// reclaim()'s rule "next child published" does. Four threads insert disjoint
// scrambled keys, one arena shard each. Phase 1 reclaims straight after the join
// (pending splits, as in ReclaimWithPendingSplits but with nodes of four shards),
// which leaves no dead node anywhere. Phase 2 then settles the keys' splits with
// a single-threaded contains() sweep (it publishes only the buckets the probed
// keys map to): each of its splits supersedes the parent copy of every key that
// moves and, uncontended, retires them all (contract clause 5). So after the
// sweep exactly one node per key is reachable, none of them dead, and the retired
// lists hold the superseded copies (more than none, or the phase tests nothing),
// which reclaim() then recycles. Every insert() boolean is exact: the keys are
// disjoint and each is inserted once, so each call must return true.
TEST(ConcurrentHashSetRcuTest, ReclaimStaleSplitCopiesNoDelete) {
    using Set = ConcurrentResizableHashSetRCU<int, false>;
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, T*PER, where + ", phase 1"));
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": reclaim() with pending splits changed membership";
        const Set::InternalAccounting swept = expect_consistent(set, where + ", after the settling sweep");
        EXPECT_EQ(swept.reachable, size_t(T*PER)) << where << ": an uncontended split left a superseded parent copy reachable";
        EXPECT_EQ(swept.reachable_dead, 0u) << where << ": a dead node is reachable after single-threaded splits";
        // EMPIRICAL precondition, not derived: with scrambled keys over many doublings
        // it is overwhelmingly likely that the sweep's splits moved many keys, but
        // nothing forces it.
        ASSERT_GT(swept.retired, 0u) << where << ": test precondition: the sweep's splits superseded no parent copy";
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, T*PER, where + ", phase 2"));
        EXPECT_EQ(membership_errors(set, in, out), 0) << where << ": reclaim() of superseded copies changed membership";
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
// concurrent_hash_rcu_detail), and each thread inserts exactly K/max(threads, shards)
// new keys: exactly what its shard's list can serve, shared with its shard mates.
// So the arena must not grow at all, and the free count must drop by exactly the
// number of inserts. Dealing everything to one shard, or popping another
// thread's shard, both show as arena growth.
// The K erases run on the main thread, uncontended, so each unlinks its own node
// and retires it onto the MAIN thread's shard's retired list (contract clause 5):
// before the first reclaim() all K are retired and none is left dead in a chain
// (both checked), reclaim()'s chain walk finds nothing, and the K slots reach the
// free lists only through the drain of the retired lists. That drain must deal
// them round-robin like every other source: handing each retired list to its own
// shard would put all K on one shard, and the threads of the other shard would
// grow the arena.
// Pre-sizing: 16384 buckets, doubling threshold 32768; the resize hint peaks at
// 2048 live + 8 pop batches of 256, far below it.
TEST(ConcurrentHashSetRcuTest, ReclaimReuseAcrossShards) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
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
        const Set::InternalAccounting erased = expect_consistent(set, where + ", after the erases");
        EXPECT_EQ(erased.retired, size_t(K)) << where << ": an uncontended erase() did not retire its node";
        EXPECT_EQ(erased.reachable_dead, 0u) << where << ": an uncontended erase() left its node in its chain";
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N - K, where + ", after the erases"));
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), where + ", after the reuse"));
        EXPECT_EQ(membership_errors(set, in, out), 0) << where;
    } // for each shape
} // ReclaimReuseAcrossShards

// After reclaim(), the structure is a mix of KEPT nodes (in chains whose dead
// neighbours were unlinked, by an operation's unlink CAS or by reclaim(), links
// rewritten through their predecessors) and REUSED nodes (popped from a free
// list, value copy-assigned, link rewritten). Both must take part in the
// concurrent protocol exactly like fresh nodes: the erase MARK CAS, the split
// FREEZE CAS and the UNLINK CAS expect a live link, the split SEAL CAS expects
// the head reclaim() left behind (with its seal level kept), and a split copies
// from chains that were relinked. This test runs concurrent erases and
// inserts, with the table doubling, over such a structure, round after round
// with a reclaim() between rounds so slots are recycled more than once.
// Oracle: every key is owned by one thread, so every boolean is fixed by the
// contract (as in ThreadPrivateKeySequenceDuringGrowth): per round, each thread
// inserts its fresh keys (true), erases every other one again (true, then false),
// and erases and re-inserts each of its keys that survived the previous rounds
// (true, absent, true, present). After the join, the exact key count and the
// membership of every key ever used are checked, along with the accounting.
// Shape: four threads on two shards (threads > shards, so pops race), fresh tiny
// sets, and four times as many fresh keys per round as in the round before, so
// that each round allocates past the free lists and doubles the table while the
// erasers and re-inserters work. That growth is what it takes: a doubling needs
// the resize hint above 2*table_size (see WHEN THE TABLE DOUBLES), reclaim()
// resets the hint to about the live count while the table keeps its size, and
// each round's doublings at least double that threshold; a round allocating no
// more than the one before it does not reach it (with a fixed count, only the
// first round doubles), and doubling the count leaves some rounds short of it.
// Keys scrambled (see scramble()).
TEST(ConcurrentHashSetRcuTest, ReclaimThenConcurrentEraseAndGrowth) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int T = 4, REPS = 12, ROUNDS = 3, A = 96, F0 = 48;
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, present.size(), "rep " + std::to_string(rep) + ", phase A"));

        for (int round = 0; round < ROUNDS; ++round) {
            const std::string where = "rep " + std::to_string(rep) + ", round " + std::to_string(round);
            const int F = F0 << (2*round);   // fresh keys per thread this round (see "Shape" above)
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
            ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, present.size(), where));
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
// The counters and the hooks are global, so ProbeKey is used only by
// single-threaded tests.
//
// CALL HOOKS (hash_hook, equal_hook), for the tests that suspend an operation at
// one of its pinned call sites and run a nested operation, or throw, there: an
// interleaving another thread could produce, made deterministic on one thread.
// The header pins the order in which every operation loads a word, hashes or
// compares a key, and performs its deciding CAS (PINNED ORDERS before contains(),
// H1-H6), so a hook that fires on a counted call opens a known window. Rules:
//   - Hooks live only in Hash{} (ProbeKeyHash) and operator==. Not in the copy
//     constructor or the copy assignment: those run inside alloc_node(), and on
//     its append path under the arena shard's SpinLock, where a nested
//     allocation would deadlock (on_assign above runs on the pop path only,
//     which holds no lock).
//   - A hook fires on the Nth QUALIFYING call after it is armed: a Hash{} call on
//     a key with the hook's value, or an operator== call whose LEFT operand has
//     it (the header compares node->value == key, so the left operand is the
//     node's value: equal_hook selects the comparison against one node). N is
//     derived at each test from the operation's sequence of Hash{} and
//     operator== calls; an action may arm the next hook (chained arming).
//   - hook_depth separates the outer operation's calls from the nested ones:
//     while an action runs, no call counts toward a hook or toward the outer
//     tally, so the nested operations' own hashing (and that of any accounting
//     sweep an action reads) neither fires a hook nor shifts a count.
//   - Tests disarm every hook right after the outer call returns or throws, and
//     before any reclaim(), settle_writers() or sweep: get_internal_accounting()
//     and reclaim() hash every reachable unmarked node and would otherwise fire
//     an armed hook.
//   - Every hook test asserts that its hooks fired and the EFFECT its window
//     exists for, so a changed call order fails loudly instead of testing
//     nothing.
// One call hook. Plain members, not atomics: only single-threaded tests use
// ProbeKey. (At namespace scope, not nested in ProbeKey: its default member
// initializers must be complete where ProbeKey's static hooks are defined.)
struct CallHook {
    int countdown = 0;              // > 0: armed; the qualifying call that brings it to 0 fires
    int value = 0;                  // the key value that makes a call qualify
    std::function<void()> action;   // run once when the hook fires; may throw out of the hooked call
}; // struct CallHook

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

    // The call hooks (see CALL HOOKS above, and CallHook).
    using Hook = CallHook;
    static inline Hook hash_hook;       // fired from ProbeKeyHash::operator()
    static inline Hook equal_hook;      // fired from operator==, on its left operand
    static inline int hook_depth = 0;   // > 0 while an action runs: calls then neither count nor fire
    // The outer tally: Hash{} calls made outside every action on a key equal to
    // *tally_value; tally_value empty means no tally is kept.
    static inline std::optional<int> tally_value;
    static inline int tally = 0;

    // Arms `hook` to fire on the `countdown`-th qualifying call (key value
    // `value`) made outside every action, running `action` there.
    static void arm(Hook& hook, int value, int countdown, std::function<void()> action) {
        hook.value = value;
        hook.countdown = countdown;
        hook.action = std::move(action);
    } // ProbeKey::arm()
    // Disarms both call hooks and stops the tally (its count stays readable).
    static void disarm_hooks() {
        hash_hook = Hook{};
        equal_hook = Hook{};
        tally_value.reset();
    } // ProbeKey::disarm_hooks()
    // Called by every hooked call with the qualifying value (the key, or the left
    // operand): counts the call toward `hook` and fires it when its countdown
    // runs out. The action is moved out before it runs, so it may re-arm the
    // same hook; hook_depth is restored even if the action throws.
    static void observe(Hook& hook, int value) {
        if (hook_depth > 0 || hook.countdown <= 0 || value != hook.value || --hook.countdown > 0) return;
        std::function<void()> action = std::move(hook.action);
        hook.action = nullptr;
        struct Depth {
            Depth() { ++hook_depth; }
            ~Depth() { --hook_depth; }
        } depth;
        action();
    } // ProbeKey::observe()
    // Hash{}'s hook entry: the outer tally, then hash_hook.
    static void observe_hash(int value) {
        if (hook_depth == 0 && tally_value == value) ++tally;
        observe(hash_hook, value);
    } // ProbeKey::observe_hash()

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
    // The left operand is the node's value (see CALL HOOKS).
    bool operator==(const ProbeKey& o) const {
        observe(equal_hook, v);
        return v == o.v;
    }
    static void check(const ProbeKey& k) {
        if (k.magic != MAGIC) ++corrupt;
    }
    // Clears the counters and every hook, the call hooks and the tally included.
    static void reset_counters() {
        copy_constructions = 0;
        copy_assignments = 0;
        destructions = 0;
        corrupt = 0;
        throw_countdown = 0;
        on_assign = nullptr;
        disarm_hooks();
        tally = 0;
        hook_depth = 0;
    } // reset_counters()
}; // struct ProbeKey

// Identity hash of the key, like CollisionHash: the tests that use ProbeKey pick
// buckets by hand, or scramble the keys themselves. Every call goes through
// ProbeKey's hash hook.
struct ProbeKeyHash {
    size_t operator()(const ProbeKey& k) const {
        ProbeKey::observe_hash(k.v);
        return static_cast<size_t>(static_cast<unsigned>(k.v));
    }
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
TEST(ConcurrentHashSetRcuTest, ReclaimThrowingAssignment) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, N - E, "after the erases"));
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), "after the retry"));
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, in.size(), "after the reuse"));
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
// contains() of one then splits that bucket: the probe keys are what such a
// split copies, which is what the exception tests below make throw. These tests
// see a doubling through get_internal_table_size().
static int fill_key(int i) { return (i & 255) | ((i >> 8) << 10); }
static constexpr int PROBE_X1 = 5 | 256;          // bucket 5 at size 256, 261 at 512
static constexpr int PROBE_X2 = 5 | 256 | 4096;   // same two buckets; a second key to move
static constexpr int PROBE_E = 5 | (7 << 10);     // bucket 5 at every size up to 1024; never a fill key

// Fills a set built as SetT(256, 1) with the keys of `first`, in order, then with
// fill_key(0), fill_key(1), ... until the arena holds exactly 512 slots; erases
// fill_key(0 .. erase-1); reclaims, checking the result. `next` receives the
// index of the first fill key not yet used. Callers wrap it in
// ASSERT_NO_FATAL_FAILURE.
// The arithmetic, from the header's documented batching of node_count_ (the
// resize hint; a doubling happens after a successful insert that sees it above
// 2*table_size, i.e. above 512 here): the appends at indices 0 and 256 add 256
// each, so the hint reaches exactly 512 and the table does not double. reclaim()
// resets the hint to the live count plus the append credit, which is 0 because
// 512 is a multiple of 256. From then on the first pop after each reclaim() adds
// 256, so it doubles the table iff live + 256 > 512, i.e. iff live > 256.
template <typename SetT>
static void fill_to_512_and_reclaim(SetT& set, const std::vector<int>& first, int erase, int& next) {
    for (int k : first) EXPECT_TRUE(set.insert(k));
    const int fills = 512 - static_cast<int>(first.size());
    for (int i = 0; i < fills; ++i) EXPECT_TRUE(set.insert(fill_key(i)));
    for (int i = 0; i < erase; ++i) EXPECT_TRUE(set.erase(fill_key(i)));
    ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, 512 - erase, "fill_to_512_and_reclaim()"));
    EXPECT_EQ(set.get_internal_free_count(), size_t(erase));
    ASSERT_EQ(set.get_internal_table_size(), 256u) << "test precondition: the fill doubled the table";
    next = fills;
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
// arithmetic in fill_to_512_and_reclaim() and watches the table size:
//   1. fill, erase 300 fill keys, reclaim(): live 212 (PROBE_X1 + 211 fills);
//   2. cycle 1: 100 inserts, all pops. The first adds 256: hint 468, no doubling
//      (the table still has 256 buckets). The pop counter now stands at 100;
//   3. reclaim(): live 312, hint 312, counter cleared;
//   4. cycle 2: ONE insert, a pop. It must add 256: hint 568 > 512, so the insert
//      doubles the table to 512. contains(PROBE_X1) then splits bucket 261.
// Cycle 1 is what makes the "counter not cleared" defect visible: with the
// counter left at 100, the next batch comes only after 156 more pops.
TEST(ConcurrentHashSetRcuTest, ReclaimRestartsPopBatching) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CollisionHash>;
    Set set(256, 1);
    int next = 0;
    ASSERT_NO_FATAL_FAILURE(fill_to_512_and_reclaim(set, {PROBE_X1}, 300, next));
    size_t live = 212;

    for (int i = 0; i < 100; ++i) EXPECT_TRUE(set.insert(fill_key(next++)));   // cycle 1
    live += 100;
    ASSERT_EQ(set.get_internal_table_size(), 256u)
        << "test precondition: the table doubled during cycle 1, so the arithmetic above does not hold";
    ASSERT_EQ(set.get_internal_node_count(), 512u) << "test precondition: cycle 1 must be served from the free list";
    ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after cycle 1"));

    EXPECT_TRUE(set.insert(fill_key(next++)));   // cycle 2: the first pop after reclaim()
    ++live;
    EXPECT_EQ(set.get_internal_table_size(), 512u)
        << "the first pop after reclaim() did not add its batch to the resize hint: the table did not double";
    EXPECT_TRUE(set.contains(PROBE_X1));   // at size 512 this splits bucket 261
    ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after cycle 2"));
} // ReclaimRestartsPopBatching

// Builds a FROZEN node whose split never finished, for the tests below: a set
// built as SetT(256, 1) over ProbeKey is filled by fill_to_512_and_reclaim() with
// `first` (which must end with PROBE_X2 and contain PROBE_X1, every other key in
// bucket 5 at sizes up to 1024 and absent from the fill keys), erasing 200 fill
// keys; one more insert, the first pop, doubles the table to 512; and
// contains(PROBE_X1) then splits bucket 261, which walks bucket 5's chain newest
// first, freezes PROBE_X2 (the first node of the chain that moves: fill_key(261)
// = 1029, the only newer node, stays) and throws on its copy, whose popped slot's
// assignment is armed to throw. Afterwards bucket 5's chain is, newest first,
//     1029 -> PROBE_X2 (FROZEN, child 261 UNINITIALIZED) -> rest of `first`
// reversed (PROBE_X1, still live, last), and limbo holds the throwing slot.
// fill_key(5), the other fill key of bucket 5, was among the erased ones.
// `next` receives the first unused fill index and `live` the key count. Callers
// wrap it in ASSERT_NO_FATAL_FAILURE; it ASSERTs its own steps.
template <typename SetT>
static void build_unfinished_split(SetT& set, const std::vector<int>& first, int& next, size_t& live) {
    ASSERT_EQ(first.back(), PROBE_X2);
    ASSERT_NO_FATAL_FAILURE(fill_to_512_and_reclaim(set, first, 200, next));
    live = 312;
    ASSERT_TRUE(set.insert(fill_key(next++)));   // the first pop: doubles the table to 512
    ++live;
    ASSERT_EQ(set.get_internal_table_size(), 512u) << "test precondition: the first pop after reclaim() did not double the table";
    ProbeKey::throw_countdown = 1;   // the split's first copy throws
    EXPECT_THROW(set.contains(PROBE_X1), std::runtime_error);
    ProbeKey::throw_countdown = 0;
    const typename SetT::InternalAccounting acc = expect_consistent(set, "after the split threw");
    ASSERT_EQ(acc.limbo, 1u) << "the slot whose assignment threw must be on limbo";
    ASSERT_EQ(acc.reachable_dead, 0u) << "test precondition: PROBE_X2 is FROZEN with its child unpublished, so not dead";
} // build_unfinished_split()

// A FROZEN node whose split never finished is KEPT by reclaim(), and a dead node
// after it is unlinked through its (FROZEN) link.
//
// Weak spots: (1) the dead-node rule must keep a FROZEN node whose child bucket is
// still UNINITIALIZED: it is then the key's only node (INVARIANT in the header);
// (2) the relink through a kept predecessor's link must keep its tag bits;
// (3) erase() must not unlink its node through such a predecessor.
// A FROZEN node with an unpublished child exists at a quiescent point only when a
// split threw between the freeze and the publication, so this test makes one
// throw (build_unfinished_split()): the first copy the split of bucket 261 makes
// pops a free slot whose assignment throws. The split froze exactly one node when
// that happens (the first moving node of the walk), and allocated nothing yet, so
// nothing leaks (compare ReclaimAccountsForSplitCopiesWhenAssignmentThrows below,
// where the SECOND copy throws).
// Chain order is newest first; the insert order PROBE_X1, PROBE_E, PROBE_X2 makes
// bucket 5's chain 1029 -> PROBE_X2 (FROZEN) -> PROBE_E -> PROBE_X1. erase(PROBE_E)
// marks PROBE_E and then cannot unlink it: its predecessor PROBE_X2 is tagged and
// not dead, and a tagged link is never a CAS target, so the erase gives up its
// self-unlink (contract clause 5's exception) and nothing is retired; PROBE_E
// stays reachable, dead. A header that unlinked through the tagged predecessor
// would retire PROBE_E; one that treated a FROZEN node as dead with its child
// unpublished would unlink PROBE_X2 too and lose the key. Then reclaim() unlinks
// PROBE_E through PROBE_X2's FROZEN link. Afterwards both probe keys must still be
// members (the next split of 261 copies the frozen and the live one alike), and
// the key count and accounting must be exact.
TEST(ConcurrentHashSetRcuTest, ReclaimKeepsFrozenNodeOfUnfinishedSplit) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, {PROBE_X1, PROBE_E, PROBE_X2}, next, live));
        const size_t reachable = set.get_internal_accounting().reachable;

        EXPECT_TRUE(set.erase(PROBE_E));   // the dead node behind the FROZEN one
        --live;
        const Set::InternalAccounting erased = expect_consistent(set, "after the erase");
        EXPECT_EQ(erased.retired, 0u) << "erase() unlinked its node through a tagged predecessor, or unlinked the FROZEN node of an unfinished split";
        EXPECT_EQ(erased.reachable, reachable) << "PROBE_E, or PROBE_X2, left the chain";
        EXPECT_EQ(erased.reachable_dead, 1u) << "PROBE_E must stay in its chain, dead";
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after the erase"));
        EXPECT_TRUE(set.contains(PROBE_X1));
        EXPECT_TRUE(set.contains(PROBE_X2));
        EXPECT_FALSE(set.contains(PROBE_E));
        EXPECT_FALSE(set.insert(PROBE_X2)) << "a key kept only by its FROZEN node was lost";
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after the split finished"));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // ReclaimKeepsFrozenNodeOfUnfinishedSplit

// erase() gives up its self-unlink after its unlink CAS loses twice, and leaves
// the retirement to whoever unlinked the node.
//
// Kills: erase() retiring its run after a lost unlink CAS (a double retirement:
// the node is then on a retired list twice, a cycle the accounting reports).
// Hooks: (1) operator== on node 1029, the first operator== of erase(1029) (1029
// is the chain's first node, so it is the walk's first comparison: H2, between
// the link load and the mark CAS); (2) armed by (1), the SECOND Hash{}(PROBE_X2)
// the outer erase(1029) makes after (1): each unlink attempt's run walk passes the
// MARKED 1029 and hashes the FROZEN PROBE_X2 to find that the run ends there (H5)
// -- the first such call is attempt 1's, the second attempt 2's, after the
// restart, whose walk from the head to 1029 hashes nothing.
// Flow, on the build_unfinished_split() chain 1029 -> PROBE_X2 (FROZEN, child
// unpublished) -> PROBE_E -> PROBE_X1:
//   - hook (1): insert(Y1), a new key of bucket 5, prepends Y1: the head moves to
//     Y1, and 1029's link is unchanged;
//   - the mark CAS on 1029's link succeeds: erase(1029) is the deletion;
//   - attempt 1, a CAS on the head word erase() loaded before (1), fails;
//   - the restart re-reads the head and finds 1029 behind Y1: attempt 2's
//     predecessor is Y1's link;
//   - hook (2), in attempt 2's window: insert(Y2), another new key of bucket 5,
//     whose walk meets the dead 1029 behind the live Y1, unlinks it through Y1's
//     link and retires it;
//   - attempt 2's CAS on Y1's link fails, and erase() gives up.
// Oracles: both hooks fired (their inserts returned true); erase(1029) returned
// true; 1029 is unreachable and retired exactly once (by Y2's walk); the
// accounting is consistent. Y1 and Y2 are fill_key(517) and fill_key(773): bucket
// 5 at sizes 512 and 1024, beyond the fill keys used. Neither insert doubles the
// table (the hint stays at 568, below 2*512).
TEST(ConcurrentHashSetRcuTest, EraseGivesUpAfterTwoLostUnlinks) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int HEAD = fill_key(261), Y1 = fill_key(517), Y2 = fill_key(773);
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, {PROBE_X1, PROBE_E, PROBE_X2}, next, live));
        const Set::InternalCounters before = set.get_internal_counters();

        bool y1_inserted = false, y2_inserted = false;
        ProbeKey::arm(ProbeKey::equal_hook, HEAD, 1, [&set, &y1_inserted, &y2_inserted, Y1, Y2]() {   // (1)
            y1_inserted = set.insert(Y1);
            ProbeKey::arm(ProbeKey::hash_hook, PROBE_X2, 2, [&set, &y2_inserted, Y2]() {   // (2)
                y2_inserted = set.insert(Y2);
            });
        });
        const bool erased = set.erase(HEAD);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(y1_inserted) << "test precondition: hook (1) did not insert Y1 inside erase()'s compare-to-mark window";
        ASSERT_TRUE(y2_inserted) << "test precondition: hook (2) did not insert Y2 inside erase()'s second unlink attempt";
        live += 2;
        EXPECT_TRUE(erased) << "erase() of a present key returned false";
        --live;

        const Set::InternalAccounting acc = expect_consistent(set, "after erase(1029)");
        EXPECT_EQ(acc.retired, 1u) << "1029 must be retired exactly once, by Y2's walk";
        EXPECT_EQ(acc.reachable_dead, 0u) << "1029 is still in its chain";
        const Set::InternalCounters after = set.get_internal_counters();
        EXPECT_EQ(after.retire_runs - before.retire_runs, 1u);
        EXPECT_FALSE(set.contains(HEAD));
        EXPECT_TRUE(set.contains(Y1));
        EXPECT_TRUE(set.contains(Y2));
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after erase(1029)"));
        EXPECT_TRUE(set.contains(PROBE_X2));
        EXPECT_TRUE(set.contains(PROBE_E));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // EraseGivesUpAfterTwoLostUnlinks

// What erase() leaves in a chain when it cannot unlink, and how reclaim() relinks
// it: a run of tombstones behind a FROZEN node whose split never finished
// (relinked through that node's tagged link), and a tombstone at the bucket head
// left by an erase() whose Hash{} threw after the mark (relinked through the head
// word, which keeps its seal level).
//
// Hook: Hash{}(PROBE_X2), the first one erase(1029) makes: the walk stops at 1029,
// the chain's first node, so it hashes nothing; after the mark, attempt 1's run
// walk passes the MARKED 1029 and hashes the FROZEN PROBE_X2 (H5). It throws: the
// key is deleted (the mark is the deletion) but the caller sees the exception,
// and 1029 stays at the head, dead (the header's EXCEPTIONS).
// Flow, on the build_unfinished_split() chain 1029 -> PROBE_X2 (FROZEN, child
// unpublished) -> PROBE_E -> E2 -> E3 -> PROBE_X1, E2 and E3 being two more keys of
// bucket 5 that stay there at 512 and 1024:
//   - erase(PROBE_E), erase(E2), erase(E3): each marks its node and gives up its
//     self-unlink, since its predecessor is PROBE_X2, tagged and not dead; the
//     three tombstones form a run behind PROBE_X2. Nothing is retired;
//   - erase(1029) with the hook armed: throws after its mark;
//   - four dead nodes are reachable, none retired; reclaim() must unlink all four,
//     the head one through the head and the run through PROBE_X2's link, with the
//     exact key count, and every remaining key must still be a member.
TEST(ConcurrentHashSetRcuTest, ReclaimRelinksWhatEraseLeftBehind) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int HEAD = fill_key(261), E2 = 5 | (6 << 10), E3 = 5 | (5 << 10);
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, {PROBE_X1, E3, E2, PROBE_E, PROBE_X2}, next, live));
        for (int k : {PROBE_E, E2, E3}) {
            EXPECT_TRUE(set.erase(k));
            --live;
        }
        const Set::InternalAccounting run = expect_consistent(set, "after the erases behind PROBE_X2");
        EXPECT_EQ(run.retired, 0u) << "an erase() unlinked its node through the tagged PROBE_X2";
        EXPECT_EQ(run.reachable_dead, 3u);

        ProbeKey::arm(ProbeKey::hash_hook, PROBE_X2, 1, []() { throw std::runtime_error("Hash{} armed to throw"); });
        EXPECT_THROW(set.erase(HEAD), std::runtime_error);
        ProbeKey::disarm_hooks();
        --live;   // the mark landed: 1029 is deleted although the caller saw an exception
        const Set::InternalAccounting left = expect_consistent(set, "after erase(1029) threw");
        EXPECT_EQ(left.retired, 0u);
        ASSERT_EQ(left.reachable_dead, 4u) << "test precondition: erase(1029) did not leave its tombstone at the head";
        EXPECT_FALSE(set.contains(HEAD));

        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after the erases"));
        for (int k : {PROBE_X1, PROBE_X2}) EXPECT_TRUE(set.contains(k));
        for (int k : {HEAD, PROBE_E, E2, E3}) EXPECT_FALSE(set.contains(k));
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after the split finished"));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // ReclaimRelinksWhatEraseLeftBehind

// NO ARENA SLOT IS LOST TO A THROW (header, "EXCEPTIONS: WHAT IS AND IS NOT
// HANDLED"): the partial subchain of a split that throws.
//
// Weak spot: split_bucket() builds its child subchain privately, referenced by
// nothing but the local head and tail. When an allocation in the middle of the
// walk throws (a popped slot's copy assignment, as here, or the append's copy
// construction), the copies made so far must go to limbo before the exception
// leaves, or they are lost for the life of the set.
// Same setup as ReclaimKeepsFrozenNodeOfUnfinishedSplit, but the split has two
// keys to copy (PROBE_X2, then PROBE_X1) and the SECOND copy throws, so the
// subchain holds one node at the throw. Checked: the exception reaches the
// caller; limbo holds exactly two nodes (the slot whose assignment threw, pushed
// by alloc_node(), and the one-node partial subchain, pushed by split_bucket());
// the accounting is consistent; both probe keys are still members (FROZEN in
// bucket 5 with 261 unpublished, so the next split copies them again); and
// reclaim() recycles both limbo nodes with the exact key count.
TEST(ConcurrentHashSetRcuTest, ReclaimAccountsForSplitCopiesWhenAssignmentThrows) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = 0;
        ASSERT_NO_FATAL_FAILURE(fill_to_512_and_reclaim(set, {PROBE_X1, PROBE_X2}, 200, next));
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after the split finished"));
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
TEST(ConcurrentHashSetRcuTest, ReclaimRecoversPendingInsertNodeWhenRetryThrows) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int PROBE_K = 5 | 256 | 8192;   // bucket 5 at size 256, 261 at 512; not a fill or probe key
    const int NESTED = fill_key(517);     // bucket 5 at every size up to 1024; beyond the fill keys used
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        int next = 0;   // not used: the keys below are chosen by hand
        ASSERT_NO_FATAL_FAILURE(fill_to_512_and_reclaim(set, {PROBE_X1}, 200, next));
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
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after the retry threw"));
        EXPECT_TRUE(set.insert(PROBE_K));
        ++live;
        ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, "after PROBE_K was inserted"));
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
// exists for happened. The assertion is on the total, since nothing forces any
// single race.
TEST(ConcurrentHashSetRcuTest, ReclaimDrainsContendedLimbo) {
    const int T = 8, K = 64, REPS = 40;
    size_t limbo_total = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        size_t limbo = 0;
        if (rep & 1) {
            ASSERT_NO_FATAL_FAILURE(one_winner_per_key_body<IntSetDel>(T, K, rep, "AllowDelete=true", limbo, 2));
        } else {
            ASSERT_NO_FATAL_FAILURE(one_winner_per_key_body<IntSetNoDel>(T, K, rep, "AllowDelete=false", limbo, 2));
        }
        limbo_total += limbo;
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
// least PREFILL slots by construction (more, with the superseded split copies).
// With AllowDelete == false the free list holds ONLY the parent copies the
// prefill's splits superseded (retired by those splits, or unlinked by reclaim()),
// which exist only for keys that moved at a doubling after they were inserted:
// a one-shard set jumps to 128 buckets within its first five inserts (the first
// append counts 256), so the first keys cross no doubling. 600 keys cross the
// doublings at 256 and 512 slots with most keys already in place, and the
// settling sweep turns those splits into a few hundred superseded copies. The
// floor for that flavor, PREFILL/4, is therefore an EMPIRICAL precondition, not
// a derived one (the prefill is single-threaded, so the count is the same every
// run).
TEST(ConcurrentHashSetRcuTest, ReclaimLimboPushesRaceFreeListPops) {
    const int T = 8, K = 64, PREFILL = 600, REPS = 40;
    size_t limbo_total = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        size_t limbo = 0;
        if (rep & 1) {
            ASSERT_NO_FATAL_FAILURE(one_winner_per_key_body<IntSetDel>(T, K, rep, "AllowDelete=true, one shard", limbo, 1, PREFILL, PREFILL));
        } else {
            ASSERT_NO_FATAL_FAILURE(one_winner_per_key_body<IntSetNoDel>(T, K, rep, "AllowDelete=false, one shard", limbo, 1, PREFILL, PREFILL/4));
        }
        limbo_total += limbo;
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
//     about (fill + round-1 allocations + split copies), i.e. about 3 times the live
//     count. A reclaim() that recycled nothing would add at least one live
//     count per round and cross the bound by round 3;
//   - without reclaim(): every successful insert() allocates a node that is never
//     reused, so the arena holds at least (R + 1)*T*W slots, which is more than
//     BOUND*T*W: this half proves the bound above is not vacuous.
// One arena shard per thread: T threads started together get T consecutive thread
// numbers, one per shard, whatever round they run in; with fewer shards than
// threads the bound would hold as well, but with MORE shards some rounds' threads
// would find their shards' lists short and append (dealing is even over all
// shards), which would be correct behavior and a looser bound.
TEST(ConcurrentHashSetRcuTest, ReclaimBoundsTheArena) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
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
                ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, L, where));
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

// ===========================================================================
// Dead nodes leave their chains: the unlinking writers (contract clause 5).
//
// A dead node -- a tombstone, or a parent copy that a split superseded -- is
// removed from its chain by a writer: erase() unlinks its own node right after
// the mark; the split that published a child unlinks the parent copies it
// superseded (its CLEANUP, one CAS per run of dead nodes, after the publication);
// and an insert() or erase() whose walk passes a dead run behind a live
// predecessor or the bucket head unlinks it (one CAS, never retried). Every
// unlink is a CAS on the predecessor's word, never on a tagged link; the winner
// pushes exactly the run it bypassed onto its own shard's retired list, where a
// node keeps its chain link, so an operation that reached it before the unlink
// can still walk out of it.
//
// The single-threaded tests below pin the postconditions of uncontended
// operations. The hook tests reproduce, on one thread, interleavings in which a
// concurrent writer changes a word between an operation's load and its CAS,
// through ProbeKey's call hooks (see CALL HOOKS at ProbeKey). Their geometries
// use the identity hash on small keys, and the doubling rule of WHEN THE TABLE
// DOUBLES: a fresh one-shard set below 128 buckets doubles on EVERY successful
// insert, so after k inserts into a Set(4, 1) the table has 4*2^k buckets
// (insert_in_order() checks it). Keys congruent to 1 modulo 32 that differ only
// in bits 5 and 6 live in bucket 1 -- one of the constructor's buckets, published
// from the start -- at every size up to 32, so they share one chain, newest
// first; at 64 those with bit 5 set move to bucket 33. Key 2 lives in bucket 2
// at every size, so inserting it doubles the table without touching that chain.
// Each test names what it kills: the protocol step it exists for, removed.
// ===========================================================================

// Inserts `keys` in order into `set` (each must be new), then checks that the
// table has `expected_ts` buckets: the geometry the calling test derived. Callers
// wrap it in ASSERT_NO_FATAL_FAILURE.
template <typename SetT>
static void insert_in_order(SetT& set, const std::vector<int>& keys, size_t expected_ts) {
    for (int k : keys) ASSERT_TRUE(set.insert(k)) << "insert(" << k << ") of a new key returned false";
    ASSERT_EQ(set.get_internal_table_size(), expected_ts) << "test precondition: the set-up did not reach the derived table size";
} // insert_in_order()

// T1: an uncontended erase() unlinks and retires its own node, every time.
//
// Kills: erase() sending its unlinked node to a LIMBO list (or anywhere but a
// retired list): retired and limbo counts then disagree with the erases.
// 8 chains of 16 colliding keys (b + 1024*i, bucket b of a 1024-bucket table,
// which never doubles: 128 nodes are one append batch, hint 256, threshold 2048)
// are erased OLDEST first, so each erase finds its node at the tail of its chain
// behind the live nodes inserted after it, and no earlier dead node lies between
// its predecessor and it (every earlier erase already took its own node out);
// the last erase of each chain finds its node at the head. Each erase unlinks
// exactly its node, so with no reclaim() at all every slot ends up retired.
// (The erased node is the tail, so a run walk that overshoots its run has
// nothing to overshoot into: GrowthLeavesNoDeadNodeReachable catches that.)
TEST(ConcurrentHashSetRcuTest, EraseAllUnlinksEveryNode) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CollisionHash>;
    const int B = 8, PER = 16, C = 1024;
    Set set(C, 1);
    for (int i = 0; i < PER; ++i) {
        for (int b = 0; b < B; ++b) EXPECT_TRUE(set.insert(b + C*i));
    }
    ASSERT_EQ(set.get_internal_table_size(), size_t(C)) << "test precondition: the table doubled";
    const Set::InternalCounters before = set.get_internal_counters();
    for (int i = 0; i < PER; ++i) {   // oldest first: each target is its chain's tail
        for (int b = 0; b < B; ++b) {
            EXPECT_TRUE(set.erase(b + C*i));
            EXPECT_FALSE(set.contains(b + C*i));
        }
    } // erase every key
    ASSERT_EQ(set.get_internal_table_size(), size_t(C)) << "test precondition: the table doubled";
    const Set::InternalAccounting acc = expect_consistent(set, "after erasing every key");
    EXPECT_EQ(acc.slots, size_t(B*PER));
    EXPECT_EQ(acc.reachable, 0u) << "an uncontended erase() left its node in its chain";
    EXPECT_EQ(acc.reachable_dead, 0u);
    EXPECT_EQ(acc.retired, size_t(B*PER)) << "every erased node must be retired, and only once";
    EXPECT_EQ(acc.limbo, 0u) << "an unlinked node went to limbo";
    EXPECT_EQ(acc.free_nodes, 0u);
    const Set::InternalCounters after = set.get_internal_counters();
    EXPECT_EQ(after.retire_runs - before.retire_runs, size_t(B*PER)) << "one retired run per erase";
    EXPECT_EQ(after.retire_nodes - before.retire_nodes, size_t(B*PER));
} // EraseAllUnlinksEveryNode

// T2: uncontended growth leaves no dead node reachable.
//
// Kills: a split that does not FREEZE the nodes it moves in one AllowDelete
// instantiation (the superseded copies are then untagged dead nodes, which no
// writer may unlink: the accounting's tag check fails and they stay reachable);
// and a run walk that overshoots its run into the first live node (that node is
// unlinked and retired: a key is lost and a retired node is untagged).
// Single-threaded inserts of scrambled keys through many doublings, then a
// contains() sweep over them, which splits every bucket a key maps to that is
// still pending. Every split is uncontended, so its cleanup unlinks every parent
// copy it superseded; with no erase() and no throw nothing else dies. Afterwards,
// with no reclaim(): exactly one node per key is reachable and none of them dead,
// nothing is on a limbo or free list, every other slot is retired, and no
// cleanup CAS lost. Both AllowDelete values: the freeze runs for both.
template <bool AllowDelete>
static void growth_leaves_no_dead_node_body() {
    using Set = ConcurrentResizableHashSetRCU<int, AllowDelete>;
    const int N = 5000;
    const std::string where = AllowDelete ? "AllowDelete=true" : "AllowDelete=false";
    Set set(4, 1);
    std::vector<int> keys;
    for (int i = 0; i < N; ++i) {
        keys.push_back(scramble(i));
        EXPECT_TRUE(set.insert(keys.back())) << where;
    }
    EXPECT_EQ(membership_errors(set, keys, std::vector<int>{}), 0) << where;
    const typename Set::InternalAccounting acc = expect_consistent(set, where + ", after the sweep");
    EXPECT_EQ(acc.reachable, size_t(N)) << where << ": a superseded parent copy is still reachable";
    EXPECT_EQ(acc.reachable_dead, 0u) << where;
    EXPECT_EQ(acc.limbo, 0u) << where;
    EXPECT_EQ(acc.free_nodes, 0u) << where;
    EXPECT_EQ(acc.retired, acc.slots - N) << where << ": every slot that is not a key's node must be retired";
    EXPECT_GT(acc.retired, 0u) << where << ": test precondition: no split moved a key";
    const typename Set::InternalCounters c = set.get_internal_counters();
    EXPECT_EQ(c.retire_nodes, acc.retired) << where;
    EXPECT_EQ(c.cleanup_cas_failures, 0u) << where << ": a cleanup CAS lost with no other thread running";
    EXPECT_EQ(c.splits_published, c.split_attempts) << where << ": a single-threaded split lost its publication";
} // growth_leaves_no_dead_node_body()

TEST(ConcurrentHashSetRcuTest, GrowthLeavesNoDeadNodeReachable) {
    growth_leaves_no_dead_node_body<false>();
    growth_leaves_no_dead_node_body<true>();
} // GrowthLeavesNoDeadNodeReachable

// T3d: erase()'s mark CAS fails on a LIVE value, and erase() must retry it.
//
// Kills: erase() falling through after a failed mark CAS whose failure value is
// live (the key is still present, and the erase reports it absent).
// Hook: operator== on node 1, the first operator== of erase(1) (1 is the chain's
// first node): H2, between the link load and the mark CAS, whose expected value
// is the link loaded before the comparison.
// Geometry: insert(9), insert(1) take a Set(4, 1) to 16 buckets; at sizes 4 and 8
// both keys are in bucket 1 (chain 1 -> 9). At 16, key 9 belongs to bucket 9,
// still UNINITIALIZED, and key 1 stays in bucket 1. The doubling happens BEFORE
// erase(1) loads the table size, on purpose: an erase whose own size is stale
// retries in the new geometry after any failed CAS, which would hide a
// fallthrough behind that retry.
// In the window, contains(9) splits bucket 9: it freezes 9's node in bucket 1 and
// publishes the copy, and the split's cleanup unlinks the now dead parent copy
// through node 1's LIVE link (1 -> 9 becomes 1 -> EMPTY) and retires it. The
// retirement is read inside the hook: the accounting sweep runs single-threaded
// in the middle of erase(1), which has loaded values but written nothing yet.
// erase(1)'s mark CAS then fails with a live value; retried on it, it succeeds.
// Afterwards: erase(1) returned true, 1 is absent, 9 present, and erase() unlinked
// its own node too (its first attempt, on the head word it loaded before the
// split sealed the head, fails; the restart succeeds): two nodes retired.
TEST(ConcurrentHashSetRcuTest, EraseRetriesMarkAfterLiveLinkChange) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int K = 1, S = 9;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {S, K}, 16));
        bool hook_ran = false, nested_found = false;
        size_t retired_in_window = 0;
        ProbeKey::arm(ProbeKey::equal_hook, K, 1, [&]() {
            hook_ran = true;
            nested_found = set.contains(S);
            retired_in_window = set.get_internal_accounting().retired;
        });
        const bool erased = set.erase(K);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on erase(1)'s comparison did not fire";
        ASSERT_TRUE(nested_found);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: the split's cleanup did not unlink 9's parent copy through 1's link";
        EXPECT_TRUE(erased) << "erase() of a present key returned false after a live change of its link";
        EXPECT_FALSE(set.contains(K));
        EXPECT_TRUE(set.contains(S));
        const Set::InternalAccounting acc = expect_consistent(set, "after erase(1)");
        EXPECT_EQ(acc.retired, 2u) << "9's parent copy and 1's node must both be retired";
        EXPECT_EQ(acc.reachable, 1u);
        EXPECT_EQ(acc.reachable_dead, 0u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // EraseRetriesMarkAfterLiveLinkChange

// T5d: a miss in a chain from which the key's FROZEN copy was unlinked must reload
// the table size and retry in the new geometry (the reload is what finds the
// key's live copy in the child).
//
// Kills: a miss returned without reloading the table size. Before dead nodes
// left their chains a stale walker still found the key's FROZEN parent copy (a
// hit); now that copy may be gone, and only the reload finds the key.
// Geometry: insert(33), insert(65), insert(1) take a Set(4, 1) to 32 buckets with
// bucket 1's chain 1 -> 65 -> 33. In the window, insert(2) doubles the table to 64
// and contains(33) splits bucket 33 from bucket 1: it freezes 33's node and
// publishes its copy, and the cleanup unlinks the parent copy through 65's LIVE
// link -- a mid-chain unlink -- and retires it (read inside the hook).
// The outer operation, which loaded size 32, then walks bucket 1 without 33,
// misses, reloads the size (64), and retries in bucket 33. It hashes 33 exactly
// twice outside the hook (one H1 per pass; the nested operations' calls are not
// counted, see CALL HOOKS), and finds the key. The variants:
//   - contains(33), hook on Hash{}(33), its first call (H1: between the size load
//     and the head load);
//   - erase(33), hook on Hash{}(33), its first call (H1): it then erases the copy
//     in bucket 33 and unlinks it too, so two nodes are retired;
//   - contains(33), hook on operator== against node 1, its first comparison (H4:
//     node 1's link is loaded before the comparison, node 65's after it), i.e.
//     two nodes before the copy: the outer walk is already inside the chain when
//     33 is unlinked from it, and reads 65's link after the unlink. (Suspended one
//     node later, at 65, it would have loaded 65's link before the unlink and
//     walked into the unlinked copy -- a legal hit on a FROZEN node.)
// `erase`: the outer operation is erase(33), else contains(33); `at_compare`:
// the third variant's hook. Callers wrap it in ASSERT_NO_FATAL_FAILURE.
static void miss_after_unlink_body(bool erase, bool at_compare) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int K = 33, W = 1;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {K, 65, W}, 32));
        bool hook_ran = false, doubled = false, nested_found = false;
        size_t retired_in_window = 0;
        std::function<void()> window = [&]() {
            hook_ran = true;
            doubled = set.insert(2) && set.get_internal_table_size() == 64;
            nested_found = set.contains(K);
            retired_in_window = set.get_internal_accounting().retired;
        };
        if (at_compare) {
            ProbeKey::arm(ProbeKey::equal_hook, W, 1, window);
        } else {
            ProbeKey::arm(ProbeKey::hash_hook, K, 1, window);
        }
        ProbeKey::tally_value = K;
        ProbeKey::tally = 0;
        const bool result = erase ? set.erase(K) : set.contains(K);
        const int outer_hashes = ProbeKey::tally;
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook did not fire";
        ASSERT_TRUE(doubled) << "test precondition: insert(2) did not double the table to 64";
        ASSERT_TRUE(nested_found);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: the split's cleanup did not unlink 33's parent copy";
        EXPECT_TRUE(result) << (erase ? "erase" : "contains") << "(33) missed a present key whose FROZEN copy was unlinked";
        EXPECT_EQ(outer_hashes, 2) << "the outer operation must hash 33 once per pass: once in bucket 1, once after the reload";
        EXPECT_EQ(set.contains(K), !erase);
        const Set::InternalAccounting acc = expect_consistent(set, "after the outer operation");
        EXPECT_EQ(acc.retired, erase ? 2u : 1u);
        EXPECT_EQ(acc.reachable_dead, 0u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // miss_after_unlink_body()

TEST(ConcurrentHashSetRcuTest, ContainsReloadsAfterUnlinkedCopy) {
    ASSERT_NO_FATAL_FAILURE(miss_after_unlink_body(false, false));
} // ContainsReloadsAfterUnlinkedCopy

TEST(ConcurrentHashSetRcuTest, EraseReloadsAfterUnlinkedCopy) {
    ASSERT_NO_FATAL_FAILURE(miss_after_unlink_body(true, false));
} // EraseReloadsAfterUnlinkedCopy

TEST(ConcurrentHashSetRcuTest, ContainsReloadsAfterMidChainUnlink) {
    ASSERT_NO_FATAL_FAILURE(miss_after_unlink_body(false, true));
} // ContainsReloadsAfterMidChainUnlink

// The split's freeze CAS fails on a LIVE value, and the split must retry it.
//
// Kills: the split falling through after a failed freeze CAS whose failure value
// is live, and copying the node without freezing it: the key then has two live
// nodes (the parent's and the child's), a stale eraser can mark the parent's and
// return true while the child keeps the key, and the untagged parent node is dead
// and can never be unlinked (the accounting's tag check fails).
// Hook: Hash{}(33), the SECOND call contains(33) makes: the first is H1; then
// bucket 33 is UNINITIALIZED, and split_bucket(33)'s walk loads each node's link
// and hashes its value before the freeze CAS (H3), and 33 is the chain's first
// node.
// Geometry: insert(1), insert(65), insert(33), insert(2) take a Set(4, 1) to 64
// buckets with bucket 1's chain 33 -> 65 -> 1; at 64 only 33 moves (to bucket
// 33). The node behind the frozen one must stay in the parent (here 1, after the
// unlink): the parent chain then keeps exactly one live node per key.
// In the window, erase(65) marks 65 and unlinks it through 33's LIVE link (33 ->
// 65 becomes 33 -> 1) and retires it. The split's freeze CAS then fails with the
// live value 33 -> 1; retried on it, it freezes 33, the split copies 33, publishes
// bucket 33, and its cleanup unlinks the FROZEN parent copy.
// Afterwards one node per key is reachable (1 in bucket 1, 2 in bucket 2, 33 in
// bucket 33), none dead, and two nodes are retired (65, and 33's parent copy).
// Under the fallthrough, 33's unfrozen parent node stays in bucket 1 as a fourth,
// untagged dead node.
TEST(ConcurrentHashSetRcuTest, SplitRetriesFreezeAfterLiveLinkChange) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int MOVER = 33, VICTIM = 65, STAYER = 1;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {STAYER, VICTIM, MOVER, 2}, 64));
        bool hook_ran = false, nested_erased = false;
        size_t retired_in_window = 0;
        ProbeKey::arm(ProbeKey::hash_hook, MOVER, 2, [&]() {
            hook_ran = true;
            nested_erased = set.erase(VICTIM);
            retired_in_window = set.get_internal_accounting().retired;
        });
        const bool found = set.contains(MOVER);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on the split's Hash{}(33) did not fire";
        ASSERT_TRUE(nested_erased);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: erase(65) did not unlink its node through 33's link";
        EXPECT_TRUE(found);
        const Set::InternalAccounting acc = expect_consistent(set, "after the split");
        EXPECT_EQ(acc.reachable, 3u) << "a key has two reachable nodes: the split copied a node it did not freeze";
        EXPECT_EQ(acc.reachable_dead, 0u);
        EXPECT_EQ(acc.retired, 2u);
        EXPECT_TRUE(set.contains(MOVER));
        EXPECT_TRUE(set.contains(STAYER));
        EXPECT_FALSE(set.contains(VICTIM));
        EXPECT_FALSE(set.erase(VICTIM));
        EXPECT_TRUE(set.erase(MOVER));
        EXPECT_FALSE(set.contains(MOVER)) << "erase(33) returned true and the key is still present";
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // SplitRetriesFreezeAfterLiveLinkChange

// A head unlink keeps the head's seal level, and a stale insert() is turned away
// by it.
//
// Kills: an unlink CAS on a bucket head that drops the seal level. A stale
// inserter decides by a CAS on the head with the level it validated against its
// table size; a split for a larger size raises the level first, so a stale insert
// fails or retries. If the split's own cleanup then wrote the head back without
// the level, the stale insert would pass the check and publish its key into the
// parent after the child was built: the key is lost.
// Hook: Hash{}(49), the first call of insert(49) (H1: between its size load and
// its head load).
// Geometry: insert(1), insert(17) take a Set(4, 1) to 16 buckets with bucket 1's
// chain 17 -> 1. Key 49 maps to bucket 1 at 16 and to bucket 17 at 32, like 17.
// In the window, insert(2) doubles the table to 32 and contains(17) splits bucket
// 17 from bucket 1: it seals bucket 1's head (level of size 32), copies 17, and
// its cleanup unlinks 17's parent copy AT THE HEAD (head 17 -> 1 becomes 1, with
// the level) and retires it.
// insert(49), which loaded size 16, then reads bucket 1's head, finds a level
// above its own, reloads the size, and inserts into bucket 17: two passes, so it
// hashes 49 twice outside the hook. Afterwards 49 must be found.
TEST(ConcurrentHashSetRcuTest, HeadUnlinkKeepsSealLevel) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int STALE = 49, MOVER = 17;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {1, MOVER}, 16));
        bool hook_ran = false, doubled = false, nested_found = false;
        size_t retired_in_window = 0;
        ProbeKey::arm(ProbeKey::hash_hook, STALE, 1, [&]() {
            hook_ran = true;
            doubled = set.insert(2) && set.get_internal_table_size() == 32;
            nested_found = set.contains(MOVER);
            retired_in_window = set.get_internal_accounting().retired;
        });
        ProbeKey::tally_value = STALE;
        ProbeKey::tally = 0;
        const bool inserted = set.insert(STALE);
        const int outer_hashes = ProbeKey::tally;
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on insert(49)'s Hash{} did not fire";
        ASSERT_TRUE(doubled) << "test precondition: insert(2) did not double the table to 32";
        ASSERT_TRUE(nested_found);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: the split's cleanup did not unlink 17's parent copy at the head";
        EXPECT_TRUE(inserted);
        EXPECT_EQ(outer_hashes, 2) << "insert(49) did not retry after meeting the raised seal level";
        EXPECT_TRUE(set.contains(STALE)) << "the stale insert published its key into the sealed parent: the key is lost";
        EXPECT_FALSE(set.insert(STALE)) << "a second insert of the key succeeded";
        const Set::InternalAccounting acc = expect_consistent(set, "after insert(49)");
        EXPECT_EQ(acc.reachable - acc.reachable_dead, 4u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // HeadUnlinkKeepsSealLevel

// A retired node can still be walked OUT of, by an operation that reached it
// before its unlink (the header's EXIT): retirement must not touch its link.
//
// Kills: a retirement that rewrites the node's chain link -- a push onto a limbo
// list (which links through `link`), or a retired-list push through `link`
// instead of the node's own retire link.
// Hook: operator== on node 2053, the first operator== of contains(5) (2053 is the
// chain's first node; H4: its link, pointing to 1029, is loaded before the
// comparison).
// Geometry: a pre-sized Set(1024, 1) (no doubling) and keys 5, 1029, 2053, all in
// bucket 5, inserted in that order: chain 2053 -> 1029 -> 5.
// In the window, erase(1029) marks 1029, unlinks it through 2053's live link and
// retires it. contains(5) resumes AT 1029 (it loaded 2053's link before the
// unlink), reads 1029's link and must reach 5. A retirement that rewrote 1029's
// link would have written the head of the list it pushed onto, which is EMPTY
// (both lists are empty beforehand, checked): the walk would end at 1029 and miss
// a present key. The window's precondition is only that 1029 left the chain
// (two reachable nodes), whichever list it went to, so that such a retirement
// fails on the walk itself; that it went to a retired list is checked after.
TEST(ConcurrentHashSetRcuTest, RetiredNodeStaysExitable) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int K = 5, X = 5 + 1024, W = 5 + 2048;
    ProbeKey::reset_counters();
    {
        Set set(1024, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {K, X, W}, 1024));
        const Set::InternalAccounting empty = expect_consistent(set, "before");
        ASSERT_EQ(empty.limbo, 0u);
        ASSERT_EQ(empty.retired, 0u);
        bool hook_ran = false, nested_erased = false;
        size_t reachable_in_window = 0;
        ProbeKey::arm(ProbeKey::equal_hook, W, 1, [&]() {
            hook_ran = true;
            nested_erased = set.erase(X);
            reachable_in_window = set.get_internal_accounting().reachable;
        });
        const bool found = set.contains(K);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on contains(5)'s first comparison did not fire";
        ASSERT_TRUE(nested_erased);
        ASSERT_EQ(reachable_in_window, 2u) << "test precondition: erase(1029) did not unlink its node";
        EXPECT_TRUE(found) << "a walk that reached a node before its unlink could not walk out of it";
        EXPECT_FALSE(set.contains(X));
        EXPECT_TRUE(set.contains(W));
        const Set::InternalAccounting acc = expect_consistent(set, "after contains(5)");
        EXPECT_EQ(acc.reachable, 2u);
        EXPECT_EQ(acc.retired, 1u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // RetiredNodeStaysExitable

// The geometry of the split-cleanup tests below: insert(1), insert(33),
// insert(65), insert(2) take a Set(4, 1) to 64 buckets with bucket 1's chain
// 65 -> 33 -> 1; at 64 only 33 moves (to bucket 33, UNINITIALIZED).
// contains(33) then splits bucket 33 and makes these Hash{}(33) calls: (1) H1;
// (2) the split's walk hashes every node of the chain before its freeze CAS (H3);
// the split publishes bucket 33; (3) the cleanup walks bucket 1 again and hashes
// the FROZEN 33 to find that its child is published, i.e. that it is dead, before
// it CASes 65's link to unlink it (H6). arm_cleanup_window() hooks (2) and, from
// it, (3), and runs `action` in (3)'s window, i.e. after the publication and
// before the cleanup's CAS. The hooks check the window: the published-split
// counter has not moved yet at (2) and has moved by one at (3).
// `key` is the hooked key (33 here): a key whose split is made by contains(key),
// whose node is the FIRST node of the parent chain the split moves (so its H3
// call is the second Hash{}(key) of contains(key)), and which the cleanup reaches
// with a usable predecessor as the first tagged node it meets (so its H6 call is
// the third).
struct CleanupWindow {
    bool before_publish = false;   // hook (2) fired, before the split published
    bool after_publish = false;    // hook (3) fired, right after the split published
}; // struct CleanupWindow

template <typename SetT>
static void arm_cleanup_window(SetT& set, CleanupWindow& w, int key, std::function<void()> action) {
    ProbeKey::arm(ProbeKey::hash_hook, key, 2, [&set, &w, key, action]() {   // (2)
        const size_t published = set.get_internal_counters().splits_published;
        w.before_publish = true;
        ProbeKey::arm(ProbeKey::hash_hook, key, 1, [&set, &w, action, published]() {   // (3)
            w.after_publish = set.get_internal_counters().splits_published == published + 1;
            action();
        });
    });
} // arm_cleanup_window()

// The split's cleanup loses its one CAS on a run, and the loser retires nothing.
//
// Kills: a retirement after a LOST unlink CAS (the run's winner retires it too:
// a node on a retired list twice, a cycle the accounting reports).
// Hooks: arm_cleanup_window(), on contains(33).
// In the cleanup's window, insert(129) -- a new key of bucket 1 -- walks bucket 1,
// meets 33's FROZEN parent copy behind the live 65 (dead: its child is
// published), unlinks it through 65's link and retires it (the opportunistic
// unlink of a writer walk), then publishes 129. The cleanup's CAS on 65's link
// then fails, counted as a lost cleanup CAS, and the cleanup retires nothing:
// 33's parent copy is retired exactly once.
TEST(ConcurrentHashSetRcuTest, CleanupLosingItsCasRetiresNothing) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {1, 33, 65, 2}, 64));
        const Set::InternalCounters before = set.get_internal_counters();
        CleanupWindow w;
        bool nested_inserted = false;
        size_t retired_in_window = 0;
        arm_cleanup_window(set, w, 33, [&set, &nested_inserted, &retired_in_window]() {
            nested_inserted = set.insert(129);
            retired_in_window = set.get_internal_accounting().retired;
        });
        const bool found = set.contains(33);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(w.before_publish) << "test precondition: the hook on the split's Hash{}(33) did not fire";
        ASSERT_TRUE(w.after_publish) << "test precondition: the hook on the cleanup's Hash{}(33) did not fire right after the publication";
        ASSERT_TRUE(nested_inserted);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: insert(129)'s walk did not unlink 33's parent copy";
        EXPECT_TRUE(found);
        const Set::InternalAccounting acc = expect_consistent(set, "after the cleanup");
        EXPECT_EQ(acc.retired, 1u) << "33's parent copy must be retired exactly once";
        EXPECT_EQ(acc.reachable_dead, 0u);
        const Set::InternalCounters after = set.get_internal_counters();
        EXPECT_EQ(after.cleanup_cas_failures - before.cleanup_cas_failures, 1u) << "the cleanup's CAS on 65's link must have lost";
        EXPECT_EQ(after.retire_runs - before.retire_runs, 1u);
        for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CleanupLosingItsCasRetiresNothing

// A Hash{} that throws in the split's cleanup leaves a superseded parent copy in
// its chain (a straggler), and the next writer walk past it collects it.
//
// Kills: a writer walk that does not unlink a dead run behind a live predecessor
// it passes (the straggler would stay until reclaim()).
// Hooks: arm_cleanup_window(), on contains(33); the action throws.
// The throw leaves the cleanup after the publication (the header's EXCEPTIONS):
// contains(33) throws, bucket 33 is published, and 33's FROZEN parent copy stays
// in bucket 1 behind the live 65, dead. insert(129), a new key of bucket 1, walks
// past it, unlinks it and retires it.
TEST(ConcurrentHashSetRcuTest, WriterWalkCollectsCleanupStraggler) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {1, 33, 65, 2}, 64));
        CleanupWindow w;
        arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
        EXPECT_THROW(set.contains(33), std::runtime_error);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(w.before_publish) << "test precondition: the hook on the split's Hash{}(33) did not fire";
        ASSERT_TRUE(w.after_publish) << "test precondition: the hook on the cleanup's Hash{}(33) did not fire right after the publication";
        const Set::InternalAccounting left = expect_consistent(set, "after the cleanup threw");
        ASSERT_EQ(left.reachable_dead, 1u) << "test precondition: the throw did not leave 33's parent copy in its chain";
        EXPECT_TRUE(set.insert(129));
        const Set::InternalAccounting acc = expect_consistent(set, "after insert(129)");
        EXPECT_EQ(acc.reachable_dead, 0u) << "insert()'s walk passed a dead node behind a live predecessor and left it";
        EXPECT_EQ(acc.retired, left.retired + 1);
        for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // WriterWalkCollectsCleanupStraggler

// erase()'s writer walk unlinks a dead run it passes behind a live predecessor.
//
// Kills: erase()'s walk making no opportunistic unlink while insert()'s walk
// still makes them (WriterWalkCollectsCleanupStraggler collects with an insert(),
// and settle_writers() collects with erase() walks). Only the MISS variant kills
// it: a HIT walk that passes the dead run and then marks its key unlinks the run
// anyway, through its own self-unlink, whose first attempt starts at the node its
// predecessor leads to and so takes the dead run along with the erased node.
// Geometry and hooks: WriterWalkCollectsCleanupStraggler's (the cleanup throws,
// so 33's FROZEN parent copy stays in bucket 1 behind the live 65, dead). Then:
//   - miss: erase(129), absent, of bucket 1: its walk passes the whole chain and
//     unlinks the dead node: one node retired;
//   - hit: erase(1), the chain's last node, behind the dead node: the walk's
//     unlink, or else the self-unlink, takes the dead node; with 1 itself, two
//     nodes retired.
TEST(ConcurrentHashSetRcuTest, EraseWalkCollectsCleanupStraggler) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    for (const bool hit : {false, true}) {
        const std::string where = hit ? "hit walk" : "miss walk";
        ProbeKey::reset_counters();
        {
            Set set(4, 1);
            ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {1, 33, 65, 2}, 64));
            CleanupWindow w;
            arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
            EXPECT_THROW(set.contains(33), std::runtime_error);
            ProbeKey::disarm_hooks();
            ASSERT_TRUE(w.before_publish && w.after_publish) << where << ": test precondition: the cleanup window did not open";
            const Set::InternalAccounting left = expect_consistent(set, where + ", after the cleanup threw");
            ASSERT_EQ(left.reachable_dead, 1u) << where << ": test precondition: no straggler behind 65";
            if (hit) {
                EXPECT_TRUE(set.erase(1));
            } else {
                EXPECT_FALSE(set.erase(129));
            }
            const Set::InternalAccounting acc = expect_consistent(set, where + ", after the erase");
            EXPECT_EQ(acc.reachable_dead, 0u) << where << ": erase()'s walk passed a dead node behind a live predecessor and left it";
            EXPECT_EQ(acc.retired, left.retired + (hit ? 2u : 1u)) << where;
            EXPECT_TRUE(set.contains(33)) << where;
            EXPECT_TRUE(set.contains(65)) << where;
            EXPECT_EQ(set.contains(1), !hit) << where;
        } // the set's lifetime
        EXPECT_EQ(ProbeKey::corrupt.load(), 0) << where;
    } // miss, then hit
} // EraseWalkCollectsCleanupStraggler

// insert()'s publish after its OWN unlink of the bucket head expects the head
// value that unlink installed (the own-CAS rule): one attempt, one Hash{}(key).
//
// Kills: a publish CAS that expects the head word as insert() first loaded it.
// That value is gone -- the insert's own unlink replaced it --, so the CAS fails
// on the thread's own write and the insert starts over: a second table-size load,
// a second Hash{}(key), a second walk. The result stays correct, so the oracle is
// the count of Hash{}(129) calls the insert makes (outside every hook: see CALL
// HOOKS), which must be 1.
// Geometry: insert(1), insert(65), insert(33), insert(2) take a Set(4, 1) to 64
// buckets with bucket 1's chain 33 -> 65 -> 1; contains(33)'s cleanup throws at
// its Hash{}(33) (arm_cleanup_window()), leaving 33's FROZEN parent copy at the
// HEAD of bucket 1, dead. insert(129), a new key of bucket 1, then unlinks it
// through the head word and publishes 129 there.
TEST(ConcurrentHashSetRcuTest, InsertPublishAfterOwnHeadUnlink) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {1, 65, 33, 2}, 64));
        CleanupWindow w;
        arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
        EXPECT_THROW(set.contains(33), std::runtime_error);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(w.before_publish && w.after_publish) << "test precondition: the cleanup window did not open";
        const Set::InternalAccounting left = expect_consistent(set, "after the cleanup threw");
        ASSERT_EQ(left.reachable_dead, 1u) << "test precondition: no dead node at the head of bucket 1";
        ProbeKey::tally_value = 129;
        ProbeKey::tally = 0;
        EXPECT_TRUE(set.insert(129));
        const int outer_hashes = ProbeKey::tally;
        ProbeKey::disarm_hooks();
        EXPECT_EQ(outer_hashes, 1) << "the publish after the insert's own head unlink did not expect the head it installed";
        const Set::InternalAccounting acc = expect_consistent(set, "after insert(129)");
        EXPECT_EQ(acc.reachable_dead, 0u);
        EXPECT_EQ(acc.retired, left.retired + 1);
        for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // InsertPublishAfterOwnHeadUnlink

// settle_writers() collects every straggler a history left, through writer walks
// alone, and splits nothing.
//
// Kills: insert()'s and erase()'s walks making no opportunistic unlink, and
// erase()'s walk alone making none (settle_writers() collects through erase()
// miss walks). It is the deterministic test behind the oracles that the
// concurrent tests check after settle_writers(), which under contention seldom
// find a straggler to settle.
// Stragglers, single-threaded: three splits whose cleanup throws at its first
// Hash{} (arm_cleanup_window()), each leaving every parent copy it superseded in
// the parent chain, dead. Geometry, identity hash, one shard:
//   - insert 0, 256, 512, 768, 1024: bucket 0 at every size up to 256; the set
//     doubles on each (rule 1 of WHEN THE TABLE DOUBLES), from 4 to 128 buckets.
//     The buckets those doublings create stay pending, among them 5, 6 and 7,
//     the first children of buckets 1, 2 and 3;
//   - at 128 buckets (no doubling until the arena passes 256 slots), the chains
//     of buckets 1, 2 and 3, newest first, keys congruent to the bucket modulo
//     128, those with bit 7 set moving at 256:
//         bucket 1: 385 -> 129 -> 257     (two movers at the head)
//         bucket 2: 514 -> 130 -> 258     (a mover behind a live node)
//         bucket 3: 515 -> 387 -> 259 -> 131   (two movers, each behind a live node);
//   - fillers 256*64, 256*65, ... (bucket 0, outside the absent keys settling
//     looks for there) until the table doubles to 256;
//   - contains(385), contains(130), contains(387): each splits the child 128
//     above its bucket and throws in that split's cleanup at the first tagged
//     node it meets, which is the hooked key.
// Five dead nodes are then reachable: a two-node run at the head of bucket 1, a
// one-node run behind a live node in bucket 2, two one-node runs in bucket 3; and
// buckets 5, 6 and 7, pending children of the same three parents, are
// UNINITIALIZED. settle_writers() must leave no dead node reachable, retire
// exactly those five in four runs, and leave buckets 5, 6 and 7 pending: had it
// split them, their cleanups would have walked the parents and collected the
// stragglers whatever the writer walks did.
TEST(ConcurrentHashSetRcuTest, SettleWritersCollectsStragglers) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    struct Split { int bucket; int key; };   // a parent bucket, and the hooked mover whose contains() splits its child
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, {0, 256, 512, 768, 1024,   // to 128 buckets
                                                      257, 129, 385, 258, 130, 514, 131, 259, 387, 515}, 128));
        for (int i = 0; set.get_internal_table_size() == 128; ++i) {   // fillers, until the doubling to 256
            ASSERT_LT(i, 300) << "test precondition: the fillers did not double the table";
            ASSERT_TRUE(set.insert(256*(64 + i)));
        }
        ASSERT_EQ(set.get_internal_table_size(), 256u);
        for (const Split split : {Split{1, 385}, Split{2, 130}, Split{3, 387}}) {
            CleanupWindow w;
            arm_cleanup_window(set, w, split.key, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
            EXPECT_THROW(set.contains(split.key), std::runtime_error) << "bucket " << split.bucket;
            ProbeKey::disarm_hooks();
            ASSERT_TRUE(w.before_publish && w.after_publish) << "test precondition: the cleanup window of bucket " << split.bucket << "'s split did not open";
        } // leave the stragglers
        const Set::InternalAccounting left = expect_consistent(set, "after the cleanups threw");
        ASSERT_EQ(left.reachable_dead, 5u) << "test precondition: the throwing cleanups did not leave five stragglers";
        for (size_t j : {5, 6, 7}) ASSERT_FALSE(set.get_internal_bucket_published(j)) << "test precondition: bucket " << j << " is published";
        const Set::InternalCounters before = set.get_internal_counters();

        ASSERT_NO_FATAL_FAILURE(settle_writers(set));
        const Set::InternalAccounting settled = expect_consistent(set, "after settle_writers()");
        EXPECT_EQ(settled.reachable_dead, 0u) << "a straggler survived a writer miss walk over its chain";
        EXPECT_EQ(settled.reachable, left.reachable - 5);
        EXPECT_EQ(settled.retired, left.retired + 5);
        const Set::InternalCounters after = set.get_internal_counters();
        EXPECT_EQ(after.retire_runs - before.retire_runs, 4u) << "one run at the head of bucket 1, one in bucket 2, two in bucket 3";
        EXPECT_EQ(after.split_attempts, before.split_attempts) << "settle_writers() split a bucket";
        for (size_t j : {5, 6, 7}) EXPECT_FALSE(set.get_internal_bucket_published(j)) << "settle_writers() published bucket " << j;
        for (int k : {257, 129, 385, 258, 130, 514, 131, 259, 387, 515}) EXPECT_TRUE(set.contains(k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // SettleWritersCollectsStragglers

// The hot-growth construction, shared with the benchmark's insert probe: the key
// of index i (0 <= i < kHotKeys) keeps a "hot" pattern of 64 values in bits 0..5,
// bits 6..11 zero, three mobile bits at 12..14, and the index above bit 16
// (distinct, positive). All keys then fall into 64 chains while the table has at
// most 4096 buckets; the doublings to 8192 and 16384 split those ~100-node chains
// (by bits 12 and 13) while the threads are still prepending to them, which is
// where a split's one cleanup CAS per run loses to a concurrent writer. mix() is
// the benchmark's key mixer (the MurmurHash3 32-bit finalizer), so that both pick
// the same bits.
static constexpr int kHotKeys = 1 << 14;
static_assert(kHotKeys <= (1 << 14), "the index must fit above bit 16 of a positive int");
static int mix(int k) {
    uint32_t x = static_cast<uint32_t>(k);
    x ^= x >> 16; x *= 0x85ebca6bu; x ^= x >> 13; x *= 0xc2b2ae35u; x ^= x >> 16;
    return static_cast<int>(x);
} // mix()
static int hot_growth_key(int index) {
    const uint32_t m = static_cast<uint32_t>(mix(index));
    return static_cast<int>((m & 63) | (((m >> 6) & 7) << 12) | (static_cast<uint32_t>(index) << 16));
} // hot_growth_key()

// What concurrent growth on hot chains leaves behind.
//
// Not a mutant test: a GUARD on the accounting under the contention that makes
// the splits' one-shot cleanup CASes lose, and a print of what it left.
// T threads insert the kHotKeys keys of the hot-growth construction (interleaved
// indices), AllowDelete == true, one arena shard per thread. After the join: the
// accounting is consistent and has exactly one non-dead reachable node per key.
// It prints the lost cleanup CASes, the splits and the dead nodes still
// reachable (stragglers). A lost cleanup CAS is not a straggler by itself: the
// CAS loses because another writer changed the predecessor word first, and on
// these chains that writer is another walker bypassing the same run, which then
// leaves the chain anyway. So the stragglers after the join are usually none,
// and the oracles after settle_writers() (one reachable node per key, none dead,
// nothing free, every other slot retired or on a limbo list: the lost split
// subchains) are then guards rather than tests of the settling;
// SettleWritersCollectsStragglers is the test that makes them bite.
TEST(ConcurrentHashSetRcuTest, HotGrowthResidue) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int T = 8;
    Set set(4, T);
    std::vector<size_t> bad(T, 0);
    run_threads(T, [&set, &bad](int t) {
        for (int i = t; i < kHotKeys; i += T) bad[t] += !set.insert(hot_growth_key(i));
    });
    for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << "insert() of a new, uncontended key returned false, thread " << t;
    const Set::InternalAccounting joined = expect_consistent(set, "after the join");
    EXPECT_EQ(joined.reachable - joined.reachable_dead, size_t(kHotKeys)) << "not exactly one non-dead reachable node per key";
    const Set::InternalCounters c = set.get_internal_counters();
    std::printf("[   INFO   ] HotGrowthResidue: table %zu, splits published %zu of %zu, cleanup CAS lost %zu, "
                "retired %zu nodes in %zu runs, reachable dead before settling %zu, limbo %zu, slots %zu\n",
                set.get_internal_table_size(), c.splits_published, c.split_attempts, c.cleanup_cas_failures,
                joined.retired, c.retire_runs, joined.reachable_dead, joined.limbo, joined.slots);
    ASSERT_NO_FATAL_FAILURE(settle_writers(set));
    const Set::InternalAccounting settled = expect_consistent(set, "after settle_writers()");
    EXPECT_EQ(settled.reachable, size_t(kHotKeys));
    EXPECT_EQ(settled.reachable_dead, 0u) << "a dead node survived a writer walk over its chain";
    EXPECT_EQ(settled.free_nodes, 0u);
    EXPECT_EQ(settled.retired + settled.limbo, settled.slots - kHotKeys);
    int missing = 0;
    for (int i = 0; i < kHotKeys; ++i) missing += !set.contains(hot_growth_key(i));
    EXPECT_EQ(missing, 0) << "keys lost";
} // HotGrowthResidue

// T4: churn stress with the full accounting after every join.
//
// Not a mutant test: a stress of the unlinking protocol under contention --
// erases unlinking their own nodes, walks unlinking what they pass, splits'
// cleanups, all on the same chains -- whose oracles are exact at every quiescent
// point: the accounting is consistent (every slot in one of the four places once,
// every reachable dead node and every retired node tagged), there is exactly one
// non-dead reachable node per key, and the per-key insert/erase tally fixes
// membership (InsertEraseChurnPerKeyAccounting's oracle, cumulative over rounds).
// Each repetition is a fresh set churned for ROUNDS rounds, joined after each:
//   - kWithReclaim: then reclaim(), with its full postconditions;
//   - kSettle (no reclaim() at all; identity hash): then settle_writers(), after
//     which no dead node is reachable, one node per key is, nothing is free, and
//     every other slot is retired or on a limbo list. The dead nodes reachable
//     BEFORE settling (the stragglers the churn left) are printed per round,
//     summed and maximized over the repetitions. Under this churn the writers'
//     own walks collect nearly every straggler before the join, so the oracles
//     after settling are GUARDS here; SettleWritersCollectsStragglers is the
//     deterministic test of the settling;
//   - kLongChains: ConstantHash (every key in bucket 0, so splits copy nothing and
//     the one chain holds every key and every dead node), with reclaim(): erases
//     unlinking through mid-chain predecessors that other erases are marking and
//     unlinking.
// Key k is scramble(k + 1), for k < R; the set has 2*T shards.
enum class ChurnCheck { kWithReclaim, kSettle, kLongChains };

template <typename Hash>
static void churn_accounting_body(ChurnCheck check, int T, int R, int REPS, int ROUNDS, int OPS) {
    using Set = ConcurrentResizableHashSetRCU<int, true, Hash>;
    std::vector<size_t> dead_sum(ROUNDS, 0), dead_max(ROUNDS, 0);   // kSettle: reachable dead before settling
    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4, 2*T);
        std::vector<std::vector<int>> ins(T, std::vector<int>(R, 0));
        std::vector<std::vector<int>> ers(T, std::vector<int>(R, 0));
        for (int round = 0; round < ROUNDS; ++round) {
            const std::string where = "rep " + std::to_string(rep) + ", round " + std::to_string(round);
            run_threads(T, [&set, &ins, &ers, rep, round, R, OPS](int t) {
                std::minstd_rand rng(static_cast<unsigned>((rep*31 + round)*977 + t + 1));
                for (int op = 0; op < OPS; ++op) {
                    const unsigned x = rng();
                    const int i = static_cast<int>((x >> 4)%static_cast<unsigned>(R));
                    switch (x & 3u) {
                        case 0:
                        case 1:   // insert twice as often as erase, so the set stays populated
                            if (set.insert(scramble(i + 1))) ++ins[t][i];
                            break;
                        case 2:
                            if (set.erase(scramble(i + 1))) ++ers[t][i];
                            break;
                        default:
                            set.contains(scramble(i + 1));
                            break;
                    } // operation kind
                } // operation loop
            });
            // The state the churn left, before any membership probe splits a bucket.
            const typename Set::InternalAccounting acc = expect_consistent(set, where + ", after the join");
            size_t live = 0;
            for (int i = 0; i < R; ++i) {
                long d = 0;
                for (int t = 0; t < T; ++t) d += ins[t][i] - ers[t][i];
                EXPECT_TRUE(d == 0 || d == 1) << where << ", key index " << i << ": insert/erase tally " << d;
                EXPECT_EQ(set.contains(scramble(i + 1)), d == 1) << where << ", key index " << i << ": membership disagrees with the tally " << d;
                live += d == 1;
            } // per-key check
            EXPECT_EQ(acc.reachable - acc.reachable_dead, live) << where << ": not exactly one non-dead reachable node per key";
            if (check == ChurnCheck::kSettle) {
                dead_sum[round] += acc.reachable_dead;
                dead_max[round] = std::max(dead_max[round], acc.reachable_dead);
                if constexpr (std::is_same_v<Hash, std::hash<int>>) {
                    ASSERT_NO_FATAL_FAILURE(settle_writers(set));
                }
                const typename Set::InternalAccounting settled = expect_consistent(set, where + ", after settle_writers()");
                EXPECT_EQ(settled.reachable_dead, 0u) << where << ": a dead node survived a writer walk over its chain";
                EXPECT_EQ(settled.reachable, live) << where;
                EXPECT_EQ(settled.free_nodes, 0u) << where;
                EXPECT_EQ(settled.retired + settled.limbo, settled.slots - live) << where;
            } else {
                ASSERT_NO_FATAL_FAILURE(reclaim_and_check(set, live, where));
            } // settle or reclaim
        } // rounds
    } // repetitions
    if (check == ChurnCheck::kSettle) {
        for (int round = 0; round < ROUNDS; ++round) {
            std::printf("[   INFO   ] churn without reclaim(), round %d: reachable dead before settling: %zu over %d sets, at most %zu in one\n",
                        round, dead_sum[round], REPS, dead_max[round]);
        }
    } // print the stragglers
} // churn_accounting_body()

TEST(ConcurrentHashSetRcuTest, ChurnAccountingWithReclaim) {
    ASSERT_NO_FATAL_FAILURE(churn_accounting_body<std::hash<int>>(ChurnCheck::kWithReclaim, 8, 48, 20, 4, 800));
} // ChurnAccountingWithReclaim

TEST(ConcurrentHashSetRcuTest, ChurnAccountingSettled) {
    ASSERT_NO_FATAL_FAILURE(churn_accounting_body<std::hash<int>>(ChurnCheck::kSettle, 8, 48, 20, 4, 800));
} // ChurnAccountingSettled

TEST(ConcurrentHashSetRcuTest, ChurnAccountingLongChains) {
    ASSERT_NO_FATAL_FAILURE(churn_accounting_body<ConstantHash>(ChurnCheck::kLongChains, 4, 32, 100, 3, 200));
} // ChurnAccountingLongChains
