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
//   4. At every quiescent point (no operation, no join -- new_handle() or a
//      refresh() -- and no advance() or reclaim() runs on the set, and the
//      caller's joins order the point against every call; live idle handles
//      may exist) each
//      arena slot is in exactly one of five places (get_internal_accounting()):
//      reachable from a published bucket, on a free list, on a limbo list (a
//      node that was never published), on a retired list (a dead node that an
//      operation or advance()'s walk unlinked from its chain), or in the BAG of
//      a closed generation (a retired or limbo node that a bump -- of an
//      advance(), a reclaim() or an automatic trigger -- sequestered and no
//      free pass has freed yet). Neither call changes any membership.
//      Called with every handle released, one advance() frees every retired,
//      limbo and bagged node, and its walk unlinks and retires every dead node
//      it can bypass -- every one except a dead run behind a FROZEN node whose
//      split never published its child -- and those it retires are freed by
//      the next call: after the two, the limbo and retired lists and every bag
//      are empty, exactly one non-dead node per key is reachable, and every
//      other arena slot is on a free list. A live handle of generation g keeps
//      every bag of generation g or later (the cumulative rule). Later
//      allocations take free slots before they grow the arena, and a reused
//      slot takes its new value by copy assignment. See the header's HANDLES
//      AND GENERATIONS section and the contract on advance().
//   5. A node is freed, and so reused, while other operations run, but only
//      under the handle invariant: never while a live handle of its bag's
//      generation or of an older one exists, and only by a free pass: of an
//      advance() or a reclaim() (advance() without the walk) that some thread
//      calls, or, on a set constructed with a cadence
//      (Options::auto_reclaim_every > 0), of an automatic trigger -- a
//      reclaim() that a retiring operation runs itself, on its own thread
//      and under its own handle (which pins the generation the trigger
//      closes), when its retire push takes its arena shard's retired-node
//      count across a multiple of the cadence; never the walk. With the
//      cadence off (the default) nothing is freed unless advance() or
//      reclaim() is called. With every handle released before each call, a
//      node retired between two calls is freed by the next call; with a
//      long-lived handle refreshed before each call, by the second (under the
//      cadence: the trigger after the one that bagged it, the handle
//      refreshed between). A straggler the walk of advance() call k bypasses
//      is freed by call k + 1 with every handle released, and by call k + 2
//      under a handle refreshed before each call; reclaim() and the trigger
//      collect no straggler. Between advance()
//      calls a dead node (a tombstone, or a parent copy that a split
//      superseded) may be unlinked from its chain by a WRITER -- the erase()
//      that made it a tombstone, the split that superseded it, or a later
//      insert() or erase() whose walk passes it -- and retired, and advance()'s
//      walk unlinks the dead runs it passes in the same way. contains()'s own
//      walk never writes; a contains() that reaches an UNINITIALIZED bucket and
//      wins its split runs
//      that split's cleanup like any other splitter (and, on a cadenced set,
//      the trigger that the cleanup's retire push may run: so a lookup may
//      bump and free, but only on a table that has doubled since
//      construction, the one way to have an UNINITIALIZED bucket). Where a dead node ends up
//      is asserted only for operations that ran UNCONTENDED, as POSTCONDITIONS
//      of those operations: an uncontended erase() leaves its own node
//      unreachable unless a tagged, not-dead node lies between it and the last
//      live word before it; an uncontended split leaves no dead node reachable
//      in the parent behind an untagged word; an uncontended insert(),
//      erase() or advance() walk leaves no dead run reachable behind the
//      bucket head or a live predecessor it passed. They are not guarantees:
//      under contention a dead node may stay in its chain (a STRAGGLER) until
//      a later writer or advance() walks past it, and no test treats a
//      straggler as a defect.
// Anything weaker than 1-4 (a "best effort" boolean, an under-counting insert)
// is a defect, not a documented relaxation -- see the header's "STALE GEOMETRY"
// section, which states 1 and 2 with one caveat: a call that exits by an
// exception after the CAS that made its transition does not return true
// (see EXCEPTIONS there). Besides that caveat, which concerns a call that does
// not return, the stragglers of 5 are the one documented relaxation, and they
// concern memory, never membership.
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
#include <csignal>
#include <type_traits>
#include <utility>
#include <limits>

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
// node_count_), so before any advance() has freed a node the hint is the sum
// over the shards of their sizes rounded UP to a multiple of 256. advance()'s
// free pass subtracts every node it frees, and a pop from a free list adds 256
// whenever the shard's pop counter is at a multiple of 256: on the shard's
// first pop ever, and every 256 pops after it (the counter is never cleared).
// That has two consequences for a test:
//   1. From the first allocation on, the hint is at least 256, so while the
//      table has fewer than 128 buckets EVERY successful insert doubles it (one
//      doubling per insert; under contention, an insert whose loaded size is
//      already stale does not). A fresh 4-bucket set has 8 buckets after its
//      first successful insert, 16 after its second, 128 after its fifth if they
//      run one at a time. Once a successful insert has returned, the set no
//      longer has 4 buckets: a test whose race must straddle the doubling 4 -> 8
//      has to make the set's FIRST successful insert part of the race, and a
//      set-up that inserts keys first has already doubled the table once per key.
//   2. From 128 buckets on, in a set with ONE arena shard (and nothing freed yet)
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

// Run fn(h, t) on T threads released together by a barrier, then join them all.
// The barrier (rather than a spin on a start flag) is what makes the racing
// window open at the same instant on every thread. `h` is thread t's own handle
// of `set`: the thread joins it BEFORE the barrier, so that no join (an RMW on
// the generation's shared count) runs inside the racing window, and destroys it
// when fn returns, before the thread is joined, so it never outlives `set`.
template <typename SetT, typename F>
static void run_threads(int T, SetT& set, F fn) {
    std::barrier start(T);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&start, &set, &fn, t]() {
            const typename SetT::handle h = set.new_handle();
            start.arrive_and_wait();
            fn(h, t);
        });
    } // spawn loop
    for (std::thread& th : threads) th.join();
} // run_threads()

// ---------------------------------------------------------------------------
// Reclamation helpers. expect_consistent() and advance_and_check() call the
// header's accounting sweep, get_internal_accounting(), and settle_writers()
// calls get_internal_bucket_published(); both are diagnostics valid only at a
// quiescent point (no operation, no join, no advance() in progress; live idle
// handles are allowed), so call these helpers only there (every worker
// joined). The sweep calls Hash{} on every reachable node that is not MARKED,
// so a test that arms ProbeKey's hooks disarms them before it calls any of
// these (see the hooks at ProbeKey).
// ---------------------------------------------------------------------------

// Accounting sweep: every arena slot must be reachable from a published bucket,
// on a free list, on a limbo list, on a retired list, or in a closed
// generation's bag, exactly once; the five counts must sum to the slot count;
// every reachable dead node must be tagged (MARKED or FROZEN); and every
// retired node, bagged ones included, must be tagged with its chain link
// intact. All of that is `consistent` (the header's C1, C2, C3, with the
// generation protocol's conjuncts: ring state, bag slots, the generation
// counts, the resize hint's one-sidedness, and under ASan the poisoning of
// every free node's value bytes); the
// sum is asserted separately only for its clearer failure message, and so is
// the per-shard split of the free count. No accessor is cross-checked here:
// get_internal_free_count(), get_internal_limbo_count() and
// get_internal_retired_count() return fields of this same sweep, and `slots`
// is get_internal_node_count() by construction. `where` names the phase in
// failure messages. Returns the sweep for further, test-specific checks.
template <typename SetT>
static typename SetT::InternalAccounting expect_consistent(const SetT& set, const std::string& where) {
    const typename SetT::InternalAccounting acc = set.get_internal_accounting();
    EXPECT_TRUE(acc.consistent) << where << ": some arena slot is unaccounted for or on two lists, a dead node is untagged,"
                                << " a retired node is untagged or lost its chain link, or the ring, a bag, a generation count"
                                << " or a free node's poisoning is wrong (slots " << acc.slots
                                << ", reachable " << acc.reachable << " of them dead " << acc.reachable_dead
                                << ", free " << acc.free_nodes << ", limbo " << acc.limbo << ", retired " << acc.retired
                                << ", bagged " << acc.bagged << ", handle refs " << acc.handle_refs << ")";
    EXPECT_EQ(acc.reachable + acc.free_nodes + acc.limbo + acc.retired + acc.bagged, acc.slots) << where;
    size_t free_by_shard = 0;
    for (size_t n : acc.free_per_shard) free_by_shard += n;
    EXPECT_EQ(free_by_shard, acc.free_nodes) << where << ": the per-shard free counts do not add up to the free count";
    return acc;
} // expect_consistent()

// The number of published buckets of the current table: what one walk of
// advance() visits (it skips the UNINITIALIZED ones). Quiescent only.
template <typename SetT>
static size_t published_buckets(const SetT& set) {
    size_t n = 0;
    for (size_t j = 0, ts = set.get_internal_table_size(); j < ts; ++j) n += set.get_internal_bucket_published(j);
    return n;
} // published_buckets()

// Full reclamation at a quiescent point, with every postcondition the contract
// promises for it, against the oracle `expected_live` (the number of keys in
// the set, known to the test). `h` is the calling thread's handle of `set`:
// it is RELEASED first (any live handle pins the generation the call closes,
// and the oracles below would turn "freed" into "kept" silently), and joined
// again at the end, so the caller goes on with a live handle of the current
// generation. The caller must hold no other live handle of `set`: that is
// ASSERTed (`handle_refs == 0`). `stuck` is the number of reachable dead nodes
// the test has placed behind a FROZEN node whose child bucket is unpublished
// (a split that threw between its freeze and its publication): no walk may
// CAS the tagged link in front of them, so they stay (the walk still passes
// them, node by node). Checked:
//   - before, the sweep must be consistent, and that is ASSERTed: a node
//     retired twice makes a cycle in a retired list, which the sweep reports
//     and stops at, but which the free pass would walk forever;
//   - the exact-count property: exactly one non-dead reachable node per key,
//     before and after (the membership cross-check at a quiescent point);
//   - the first advance(): `advanced` iff a retired or limbo list was
//     non-empty (otherwise `nothing_retired`, which still runs the free pass
//     and the walk); it frees every retired, limbo and bagged node (with no
//     handle alive, the bag it closes is freeable in the same call), and the
//     bags it frees are every SEALED one plus the one it closes; its walk
//     visits every published bucket and every reachable node, and unlinks and
//     retires every reachable dead node except the `stuck` ones, uncontended
//     (it runs alone) -- and those retirees are not freed by this call: they
//     are on the current generation's retired list;
//   - a second advance() only when that walk retired something, because a
//     walk retiree is reusable only at a LATER call (two clicks): `advanced`,
//     exactly the walk's retirees freed in one bag, nothing new retired;
//   - afterwards the arena is unchanged, the limbo and retired lists and
//     every bag are empty, exactly the `stuck` dead nodes are reachable, every
//     other non-key slot is on a free list, and the free lists grew by exactly
//     what the calls freed; the generation moved by the number of `advanced`
//     results.
// Not value-returning, so that it can ASSERT: callers wrap it in
// ASSERT_NO_FATAL_FAILURE.
template <typename SetT>
static void advance_and_check(SetT& set, typename SetT::handle& h, size_t expected_live, const std::string& where, size_t stuck = 0) {
    using Advance = AdvanceResult::Advance;
    h = {};
    const typename SetT::InternalAccounting before = expect_consistent(set, where + ", before advance()");
    ASSERT_TRUE(before.consistent) << where << ": advance() not called on an inconsistent set (a retired-list cycle would hang its free pass)";
    ASSERT_EQ(before.handle_refs, 0) << where << ": test precondition: a handle is still alive and would pin the generation advance() closes";
    ASSERT_GE(before.reachable_dead, stuck) << where << ": test precondition: fewer reachable dead nodes than the test placed behind FROZEN nodes";
    EXPECT_EQ(before.reachable - before.reachable_dead, expected_live) << where << ": not exactly one non-dead reachable node per key before advance()";
    size_t sealed = 0;
    for (const typename SetT::InternalRingSlot& slot : set.get_internal_ring_state().slots) sealed += slot.state == SetT::GenerationState::SEALED;
    const uint64_t gen0 = set.get_internal_generation();
    const size_t buckets = published_buckets(set);
    const bool retiring = before.limbo + before.retired > 0;

    const AdvanceResult r1 = set.advance();
    EXPECT_EQ(r1.advance, retiring ? Advance::advanced : Advance::nothing_retired) << where << ", first advance(): " << ::testing::PrintToString(r1);
    EXPECT_EQ(r1.freed_nodes, before.limbo + before.retired + before.bagged) << where << ": the first advance() must free every limbo, retired and bagged node";
    EXPECT_EQ(r1.freed_bags, sealed + (retiring ? 1u : 0u)) << where << ": the first advance() must free every sealed bag and the one it closes";
    EXPECT_EQ(r1.walk_retired, before.reachable_dead - stuck) << where << ": the walk must retire every reachable dead node it can bypass";
    EXPECT_EQ(r1.buckets_walked, buckets) << where << ": the walk must visit every published bucket and no other";
    EXPECT_EQ(r1.nodes_passed, before.reachable) << where << ": the walk must pass every reachable node once";
    const typename SetT::InternalAccounting mid = expect_consistent(set, where + ", after the first advance()");
    EXPECT_EQ(mid.retired, r1.walk_retired) << where << ": the walk's retirees must be on the current generation's retired lists";
    EXPECT_EQ(mid.bagged, 0u) << where;
    EXPECT_EQ(mid.limbo, 0u) << where;
    EXPECT_EQ(mid.free_nodes, before.free_nodes + r1.freed_nodes) << where << ": the free lists did not grow by exactly what the first advance() freed";

    size_t walk_freed = 0;
    if (r1.walk_retired > 0) {   // the walk's retirees: reusable only at the next call
        const AdvanceResult r2 = set.advance();
        EXPECT_EQ(r2, (AdvanceResult{r1.walk_retired, 1, 0, buckets, before.reachable - r1.walk_retired, Advance::advanced}))
            << where << ": the second advance() must free exactly the walk's retirees, in one bag, and retire nothing";
        walk_freed = r2.freed_nodes;
    } // if the walk retired something

    const typename SetT::InternalAccounting after = expect_consistent(set, where + ", after advance()");
    EXPECT_EQ(after.slots, before.slots) << where << ": advance() must not change the arena size";
    EXPECT_EQ(after.reachable, before.reachable - before.reachable_dead + stuck) << where << ": advance() must unlink exactly the dead nodes it can bypass";
    EXPECT_EQ(after.reachable - after.reachable_dead, expected_live) << where << ": not exactly one non-dead reachable node per key after advance()";
    EXPECT_EQ(after.reachable_dead, stuck) << where << ": a dead node the walk could bypass is still reachable";
    EXPECT_EQ(after.limbo, 0u) << where << ": a limbo node survived advance()";
    EXPECT_EQ(after.retired, 0u) << where << ": a retired node survived advance()";
    EXPECT_EQ(after.bagged, 0u) << where << ": a bag survived advance() with no handle alive";
    EXPECT_EQ(after.free_nodes, before.free_nodes + r1.freed_nodes + walk_freed) << where << ": the free lists did not grow by exactly what advance() freed";
    EXPECT_EQ(after.free_nodes, after.slots - after.reachable) << where << ": a slot that is not reachable is not free";
    EXPECT_EQ(set.get_internal_generation(), gen0 + (retiring ? 1u : 0u) + (r1.walk_retired > 0 ? 1u : 0u)) << where;
    h = set.new_handle();
} // advance_and_check()

// settle_writers(set, h): a writer MISS walk over every published bucket, so that
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
// It skips the pending (UNINITIALIZED) buckets, asking the header's
// get_internal_bucket_published() (a diagnostic valid only at a quiescent
// point, as here): a contains() or erase() there would split the bucket, and
// that split's cleanup walks the parent chain and unlinks its dead runs itself,
// so the settling would succeed even if the writer walks it exists to exercise
// unlinked nothing. Skipping them also leaves the geometry as the history left
// it: settling splits nothing.
// Requirements: quiescent and single-threaded; ProbeKey hooks disarmed; an
// AllowDelete == true set of int keys whose hash is the identity (std::hash<int>
// here, so j + m*ts maps to j; not StringKeys); and no dead run behind a FROZEN
// node whose child is unpublished (a split that threw between its freeze and its
// publication): such a run is behind a tagged predecessor, which no walk may
// CAS -- advance()'s walk included --, and only a split of the child collects it.
// Not an oracle in itself: on a set with a concurrent history it is not a no-op.
// `h`: the calling thread's handle of `set`, under which every probe runs.
template <typename SetT>
static void settle_writers(SetT& set, const typename SetT::handle& h) {
    const size_t ts = set.get_internal_table_size();
    for (size_t j = 0; j < ts; ++j) {
        if (!set.get_internal_bucket_published(j)) continue;
        bool erased_in_j = false;
        for (size_t m = 0; m < 64 && !erased_in_j; ++m) {
            const size_t k = j + m*ts;
            if (k > static_cast<size_t>(INT32_MAX)) break;
            if (set.contains(h, static_cast<int>(k))) continue;   // present: not a miss walk
            ASSERT_FALSE(set.erase(h, static_cast<int>(k))) << "settle_writers(): erase() of an absent key returned true";
            erased_in_j = true;
        } // look for an absent key of bucket j
        ASSERT_TRUE(erased_in_j) << "settle_writers(): no absent key found for bucket " << j;
    } // loop over the buckets
} // settle_writers()

// Membership oracle sweep: the number of wrong contains() answers, where every
// key of `in` must be present and every key of `out` absent. Side effect worth
// knowing: contains() splits every UNINITIALIZED bucket it lands on, so after
// this sweep every probed key's current bucket is published. `h`: the calling
// thread's handle of `set`, under which every probe runs.
template <typename SetT, typename K>
static int membership_errors(SetT& set, const typename SetT::handle& h, const std::vector<K>& in, const std::vector<K>& out) {
    int errors = 0;
    for (const K& k : in) errors += !set.contains(h, k);
    for (const K& k : out) errors += set.contains(h, k);
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
    const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
    EXPECT_TRUE(set.insert(h, 1));
    EXPECT_FALSE(set.insert(h, 1));
    EXPECT_TRUE(set.insert(h, 2));
    EXPECT_TRUE(set.contains(h, 1));
    EXPECT_TRUE(set.contains(h, 2));
    EXPECT_FALSE(set.contains(h, 3));
} // BasicOperations

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
        const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
        EXPECT_EQ(set.get_internal_arena_shards(), shards);
        std::atomic<int> inserted{0};
        run_threads(T, set, [&set, &inserted](const ConcurrentResizableHashSetRCU<int>::handle& h, int t) {
            for (int k = t; k < N; k += T) {
                if (set.insert(h, scramble(k))) inserted.fetch_add(1, std::memory_order_relaxed);
            }
        });
        EXPECT_EQ(inserted.load(), N);
        for (int k = 0; k < N; ++k) EXPECT_TRUE(set.contains(h, scramble(k)));
        EXPECT_FALSE(set.contains(h, scramble(N)));
        // Every inserted key has a node; split copies add more, never fewer.
        EXPECT_GE(set.get_internal_node_count(), size_t(N));
    } // for each shard count
} // ArenaShardsParameter

// Both constructors are `explicit`: a bare size is not a set, nor is
// an Options aggregate, so neither converts implicitly (copy-initialization
// `Set s = 4096;`, a size or an Options passed where a Set is expected). A
// type trait, not a compile of `Set s = {};`, which the compilers disagree on
// for an explicit default constructor (CWG 1518).
static_assert(!std::is_convertible_v<size_t, ConcurrentResizableHashSetRCU<int>>,
              "the two-parameter constructor must be explicit");
static_assert(!std::is_convertible_v<ConcurrentResizableHashSetRCU<int>::Options, ConcurrentResizableHashSetRCU<int>>,
              "the Options constructor must be explicit");

// A capacity above 2^63 aborts the constructor, in every build: std::bit_ceil()
// has no representable result for it. The death is SIGABRT from std::abort()
// itself, with nothing on stderr; an exception, a normal return or any other
// signal fails the test. The empty stderr is what tells the constructor's
// check from libstdc++'s own: libstdc++ 15 and later enable
// _GLIBCXX_ASSERTIONS in an unoptimized build (these tests are -O0), and
// std::bit_ceil() then prints an assertion message before it aborts.
// Both arguments exceed 2^63; 2^63 itself is valid and would ask for 2^63
// buckets, so the boundary is not tested from below. The suite name ends in
// DeathTest, as GoogleTest asks, so the suite runs before every other; the
// threadsafe style re-executes the binary for the child instead of cloning a
// process that may run other threads (a sanitizer runtime's among them).
// GoogleTest restores the flag after the test.
TEST(ConcurrentHashSetRcuDeathTest, ConstructorAbortsOnUnrepresentableCapacity) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    for (size_t capacity : {static_cast<size_t>(-1), (size_t{1} << 63) + 1}) {
        EXPECT_EXIT({ ConcurrentResizableHashSetRCU<int> set(capacity, 1); }, testing::KilledBySignal(SIGABRT), "^$")
            << "capacity " << capacity;
    }
} // ConstructorAbortsOnUnrepresentableCapacity

// Single-threaded correctness across many table doublings: every inserted key must
// be found, duplicates must report false, and absent keys must not be found.
// Keys are scrambled so that roughly half of them change bucket at each of the
// ~14 doublings this test drives; with consecutive keys the later doublings would
// leave the low keys in place and never split their chains (see scramble()).
TEST(ConcurrentHashSetRcuTest, ManyResizesSingleThread) {
    ConcurrentResizableHashSetRCU<int> set(4);
    const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(set.insert(h, scramble(i))) << "insert(" << i << ") wrongly reported duplicate";
    }
    for (int i = 0; i < N; i += 997) {
        EXPECT_FALSE(set.insert(h, scramble(i))) << "re-insert(" << i << ") wrongly reported new";
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(set.contains(h, scramble(i))) << "missing key " << i;
    }
    for (int i = N; i < N + 5000; ++i) {
        EXPECT_FALSE(set.contains(h, scramble(i))) << "false positive for " << i;
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
    const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
    const int T = 8, per = 20000;
    std::atomic<int> inserted{0};
    run_threads(T, set, [&set, &inserted](const ConcurrentResizableHashSetRCU<int>::handle& h, int t) {
        int local = 0;
        for (int k = 0; k < per; ++k) {
            if (set.insert(h, scramble(t*per + k))) ++local;
        }
        inserted += local;
    });

    EXPECT_EQ(inserted.load(), T*per) << "insert() did not report exactly one winner per key";
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < per; ++k) {
            EXPECT_TRUE(set.contains(h, scramble(t*per + k))) << "missing " << (t*per + k);
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
        const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
        const int T = 16, N = 4000;
        std::atomic<int> true_count{0};
        run_threads(T, set, [&set, &true_count](const ConcurrentResizableHashSetRCU<int>::handle& h, int) {
            int local = 0;
            for (int k = 0; k < N; ++k) {
                if (set.insert(h, scramble(k))) ++local;
            }
            true_count += local;
        });

        EXPECT_EQ(true_count.load(), N) << "insert() did not report exactly one winner per key, rep " << rep;
        for (int k = 0; k < N; ++k) {
            EXPECT_TRUE(set.contains(h, scramble(k))) << "missing key " << k << " in rep " << rep;
        }
    } // repetitions
} // ConcurrentSameKeyMembership

// Forces heavy collisions into a handful of buckets to stress chain splitting.
struct ModHash {
    size_t operator()(int x) const { return static_cast<size_t>(x%8); }
};

TEST(ConcurrentHashSetRcuTest, CollisionHeavyChains) {
    ConcurrentResizableHashSetRCU<int, false, ModHash> set(4);
    const ConcurrentResizableHashSetRCU<int, false, ModHash>::handle h = set.new_handle();
    const int N = 5000;
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(h, i));
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.contains(h, i));
    EXPECT_FALSE(set.contains(h, N + 1));
} // CollisionHeavyChains

// Non-trivial key type to confirm the container is not hard-wired to integers.
TEST(ConcurrentHashSetRcuTest, StringKeys) {
    ConcurrentResizableHashSetRCU<std::string> set(4);
    const ConcurrentResizableHashSetRCU<std::string>::handle h = set.new_handle();
    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(set.insert(h, "key_" + std::to_string(i)));
    }
    EXPECT_FALSE(set.insert(h, "key_500"));
    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(set.contains(h, "key_" + std::to_string(i)));
    }
    EXPECT_FALSE(set.contains(h, "absent"));
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
        const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
        for (int i = 0; i < N; ++i) {
            set.insert(h, scramble(i));
            progress.store(i, std::memory_order_release);
        }
        done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&]() {
            const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
            while (!done.load(std::memory_order_acquire)) {
                int p = progress.load(std::memory_order_acquire);
                if (p > 0) {
                    // A key strictly below the writer's progress was fully inserted.
                    EXPECT_TRUE(set.contains(h, scramble(p - 1)));
                }
                EXPECT_FALSE(set.contains(h, scramble(N + 12345)));  // never inserted
            } // poll until the writer is done
        });
    } // spawn readers

    writer.join();
    for (std::thread& t : readers) t.join();
    const ConcurrentResizableHashSetRCU<int>::handle h = set.new_handle();
    for (int i = 0; i < N; ++i) EXPECT_TRUE(set.contains(h, scramble(i)));
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
        const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle h = set.new_handle();
        set.insert(h, 100);

        // 8 threads aggressively inserting elements that hash to 4, 5, 6, 7. Every
        // thread's first key (k = 0) maps to bucket 4 (i*10000 is a multiple of
        // 16), so all eight start in the split of the same bucket; their own
        // inserts keep doubling the table from there (rule 1, then the shards they
        // touch).
        run_threads(8, set, [&set](const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle& h, int i) {
            for (int k = 0; k < 500; ++k) {
                // Keys that map to bucket 4, 5, 6, 7 when the table size is 8.
                set.insert(h, 4 + 8*k + i*10000);
                set.insert(h, 5 + 8*k + i*10000);
                set.insert(h, 6 + 8*k + i*10000);
                set.insert(h, 7 + 8*k + i*10000);
            } // key loop
        });

        // Verify
        for (int i = 0; i < 8; ++i) {
            for (int k = 0; k < 500; ++k) {
                EXPECT_TRUE(set.contains(h, 4 + 8*k + i*10000)) << "rep " << rep;
                EXPECT_TRUE(set.contains(h, 5 + 8*k + i*10000)) << "rep " << rep;
                EXPECT_TRUE(set.contains(h, 6 + 8*k + i*10000)) << "rep " << rep;
                EXPECT_TRUE(set.contains(h, 7 + 8*k + i*10000)) << "rep " << rep;
            } // key loop
        } // membership sweep
        EXPECT_TRUE(set.contains(h, 100)) << "the key the first split copied was lost, rep " << rep;
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
    const ConcurrentResizableHashSetRCU<int, true>::handle h = set.new_handle();

    // Basic erase
    EXPECT_TRUE(set.insert(h, 10));
    EXPECT_TRUE(set.insert(h, 20));
    EXPECT_TRUE(set.contains(h, 10));
    EXPECT_TRUE(set.erase(h, 10));
    EXPECT_FALSE(set.contains(h, 10));
    EXPECT_FALSE(set.erase(h, 10)); // Already erased
    EXPECT_TRUE(set.contains(h, 20));

    // Erase and insert again
    EXPECT_TRUE(set.insert(h, 10));
    EXPECT_TRUE(set.contains(h, 10));

    // Concurrent erase of disjoint, pre-populated ranges.
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < 1000; ++k) {
            set.insert(h, scramble(i*10000 + k));
        }
    } // pre-population

    std::vector<size_t> not_erased(8, 0);
    run_threads(8, set, [&set, &not_erased](const ConcurrentResizableHashSetRCU<int, true>::handle& h, int i) {
        size_t bad = 0;
        for (int k = 0; k < 1000; ++k) {
            // The key is present and private to this thread: the contract makes
            // this call the present->absent transition, so it must return true.
            bad += !set.erase(h, scramble(i*10000 + k));
        }
        not_erased[i] = bad;
    });

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(not_erased[i], 0u) << "erase() of a present, uncontended key returned false, thread " << i;
        for (int k = 0; k < 1000; ++k) {
            EXPECT_FALSE(set.contains(h, scramble(i*10000 + k)));
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
    const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle h = set.new_handle();

    // Exactly one resize (N=4 -> N=8) before the threads start: by rule 1 of "WHEN
    // THE TABLE DOUBLES" the set's first insert doubles it, and a second one would
    // double it again. Key 0 stays in bucket 0, so buckets 4..7 are UNINITIALIZED
    // when the threads are released.
    set.insert(h, 0);

    constexpr int NUM_THREADS = 8;
    constexpr int INSERTS_PER_THREAD = 1000;
    // The i-th key of thread t (see above); distinct for every (t, i).
    auto key_of = [](int t, int i) { return 5 + 8*i + (t << 16); };

    run_threads(NUM_THREADS, set, [&set, &key_of](const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle& h, int t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            set.insert(h, key_of(t, i));
        } // insert loop
    });

    int success_count = 0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
            if (set.contains(h, key_of(t, i))) {
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
    const ConcurrentResizableHashSetRCU<int, true>::handle h = set.new_handle();
    constexpr int NUM_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 2000;

    // First, populate the set heavily so we force resizes and splits.
    for (int i = 0; i < NUM_THREADS*OPERATIONS_PER_THREAD; ++i) {
        set.insert(h, scramble(i));
    }

    // Concurrently read, insert, and erase to stress the MARK_BIT CAS, the freeze
    // CAS a concurrent split performs on the same link word, and the DCLP resize.
    std::vector<size_t> bad(NUM_THREADS, 0);
    run_threads(NUM_THREADS, set, [&set, &bad](const ConcurrentResizableHashSetRCU<int, true>::handle& h, int t) {
        size_t b = 0;
        for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
            int key = scramble(t*OPERATIONS_PER_THREAD + i);

            b += !set.erase(h, key);      // present -> absent: exactly this call
            b += set.contains(h, key);    // its own eraser must not still see it
            b += !set.insert(h, key);     // absent -> present: exactly this call
            b += !set.erase(h, key);      // present -> absent again
        } // operation loop
        bad[t] = b;
    });

    for (int t = 0; t < NUM_THREADS; ++t) {
        EXPECT_EQ(bad[t], 0u) << "determined erase/insert/contains sequence broke on thread " << t;
    }

    // At the end, all keys should be erased
    for (int i = 0; i < NUM_THREADS*OPERATIONS_PER_THREAD; ++i) {
        EXPECT_FALSE(set.contains(h, scramble(i)));
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
            const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle h = set.new_handle();
            while (!start.load(std::memory_order_acquire)) {}
            // Insert 4 which hashes to 0 initially (4 & 3 == 0)
            t1_inserted = set.insert(h, 4);
        });

        std::thread t2([&]() {
            const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle h = set.new_handle();
            while (!start.load(std::memory_order_acquire)) {}
            set.insert(h, 1);      // the set's first insert, as a rule: doubles 4 -> 8
            set.contains(h, 12);   // splits bucket 4 from bucket 0's chain
        });

        start.store(true, std::memory_order_release);
        t1.join();
        t2.join();
        const ConcurrentResizableHashSetRCU<int, false, CollisionHash>::handle h = set.new_handle();

        ASSERT_TRUE(t1_inserted) << "insert(4) of an absent, uncontended key returned false in rep " << rep;
        ASSERT_TRUE(set.contains(h, 4)) << "Lost update in rep " << rep;
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

    run_threads(T, set, [&set](const ConcurrentResizableHashSetRCU<int, false, ConstantHash>::handle& h, int t) {
        for (int i = 0; i < N; ++i) {
            set.insert(h, t*N + i);
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
// subchain must have been recorded on a limbo list at birth) and reclaims with
// every handle released (advance_and_check(): advance(), and a second call if
// its walk retired something) with its full postconditions, immediately, with
// whatever splits are still pending, BEFORE the membership sweep, so the sweep
// also checks that advance() lost no key. AdvanceDrainsContendedLimbo reuses it
// for that.
// Interface: T threads each insert the same N scrambled keys into a fresh set with
// `shards` arena shards (0 = the default); `rep` and `flavor` label messages.
// `prefill` > 0 first inserts that many OTHER keys single-threaded, settles their
// pending splits with a contains() sweep (whose splits supersede the parent copy
// of every key that moves, and retire it), erases them again if the set has
// erase(), and reclaims, so the race starts with non-empty free lists: its
// allocations then POP while its orphans and lost subchains are pushed to limbo,
// on the same shard when `shards` is 1 (see AdvanceLimboPushesRaceFreeListPops).
// Without erase() the prefill keys stay in the set and the free lists hold only
// those superseded copies, which is why the sweep is there, and why the prefill
// must cross doublings AFTER keys exist (see the prefill sizes in
// AdvanceLimboPushesRaceFreeListPops). `min_free` is the floor the free lists
// must reach after the prefill's reclamation. `limbo` receives the number of
// nodes that were on limbo lists before the race's reclamation. Callers wrap it in
// ASSERT_NO_FATAL_FAILURE (it ASSERTs through advance_and_check()); its template
// argument goes through an alias there, since a comma inside the macro's
// argument would split it.
template <typename SetT>
static void one_winner_per_key_body(int T, int N, int rep, const char* flavor, size_t& limbo, size_t shards = 0, int prefill = 0, size_t min_free = 0) {
    SetT set(4, shards);
    typename SetT::handle h = set.new_handle();
    size_t prefilled = 0;   // prefill keys still in the set when the race starts
    std::vector<int> prefill_keys;
    for (int i = 0; i < prefill; ++i) {
        prefill_keys.push_back(scramble(N + i));
        set.insert(h, prefill_keys.back());
    }
    if (prefill > 0) {
        EXPECT_EQ(membership_errors(set, h, prefill_keys, std::vector<int>{}), 0) << "prefill, rep " << rep;   // settles the splits
        if constexpr (requires { set.erase(h, 0); }) {
            for (int i = 0; i < prefill; ++i) set.erase(h, scramble(N + i));
        } else {
            prefilled = static_cast<size_t>(prefill);
        }
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, prefilled, std::string("prefill, rep ") + std::to_string(rep)));
        EXPECT_GE(set.get_internal_free_count(), min_free) << "prefill, rep " << rep << " (" << flavor
                                                           << "): test precondition: the race would start with a short free list";
    } // start with non-empty free lists
    const size_t expected = static_cast<size_t>(N) + prefilled;
    std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(N, 0));
    run_threads(T, set, [&set, &won, N](const typename SetT::handle& h, int t) {
        for (int k = 0; k < N; ++k) {
            if (set.insert(h, scramble(k))) won[t][k] = 1;
        }
    });
    const std::string where = std::string("rep ") + std::to_string(rep) + " (" + flavor + ")";
    limbo = expect_consistent(set, where + ", after the insert race").limbo;
    ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, expected, where));
    int over = 0, under = 0, missing = 0;
    for (int k = 0; k < N; ++k) {
        int c = 0;
        for (int t = 0; t < T; ++t) c += won[t][k];
        if (c > 1) ++over;
        if (c < 1) ++under;
        if (!set.contains(h, scramble(k))) ++missing;
    } // per-key check
    EXPECT_EQ(over, 0) << over << " key(s) had MORE than one successful insert(), " << where;
    EXPECT_EQ(under, 0) << under << " key(s) had NO successful insert(), " << where;
    EXPECT_EQ(missing, 0) << missing << " key(s) absent after all inserts and an advance(), " << where;
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
    // The handles of each attempt's set: handles[role] is worker `role`'s and
    // handles[3] this thread's. This thread joins them all to the new set before
    // the attempt's first barrier, so no join runs inside the race; a worker uses
    // its own only after that barrier (the happens-before edge a handle needs to
    // be used on another thread); and this thread releases them all after the
    // second barrier, before the next attempt's set replaces this one, since a
    // handle must not outlive its set. Declared after `set`, so destroyed first.
    // Each on its own 128-byte block: with assert() live every operation counts
    // itself in its handle (an RMW at entry and at exit), and handles packed
    // into one line would put that line between the racing workers.
    struct alignas(128) PaddedHandle {
        Set::handle h;   // one thread's handle of the attempt's set
    }; // struct PaddedHandle
    std::vector<PaddedHandle> handles(4);
    std::atomic<int> winners{0};
    std::barrier round(4);   // three workers plus this thread

    std::vector<std::thread> workers;
    for (int role = 0; role < 3; ++role) {
        workers.emplace_back([&set, &handles, &winners, &round, KEY, role]() {
            for (int a = 0; a < ATTEMPTS; ++a) {
                round.arrive_and_wait();     // the set for this attempt is built
                const Set::handle& h = handles[role].h;
                if (role == 1) {
                    set->insert(h, 1);       // the set's first insert, as a rule: doubles 4 -> 8
                } else if (set->insert(h, KEY)) {
                    winners.fetch_add(1, std::memory_order_relaxed);
                }
                round.arrive_and_wait();     // results are safe to read
            } // attempt loop
        });
    } // spawn workers

    int bad = 0, first_bad = -1, missing = 0;
    for (int a = 0; a < ATTEMPTS; ++a) {
        set = std::make_unique<Set>(4);   // empty: see "Geometry" above
        for (PaddedHandle& p : handles) p.h = set->new_handle();
        winners.store(0, std::memory_order_relaxed);
        round.arrive_and_wait();
        round.arrive_and_wait();
        if (winners.load(std::memory_order_relaxed) != 1) {
            ++bad;
            if (first_bad < 0) first_bad = a;
        }
        if (!set->contains(handles[3].h, KEY)) ++missing;
        for (PaddedHandle& p : handles) p.h = {};   // before the next attempt's set replaces this one
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
        const Set::handle h = set.new_handle();
        for (int k = 0; k < K; ++k) set.insert(h, scramble(k));

        std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(K, 0));
        std::vector<int> dup(T, 0);
        run_threads(T, set, [&set, &won, &dup](const Set::handle& h, int t) {
            for (int i = 0; i < K; ++i) {
                const int k = (i + t*K/T)%K;     // staggered start per thread
                if (set.erase(h, scramble(k))) ++won[t][k];
                // Filler inserts, spread evenly through the erases from step s0 on
                // (see "Shape" above): these are what grow the arena and double the
                // table mid-flight. Each key belongs to one thread, so a false
                // return is a defect in itself.
                const int s0 = t*K/T;      // this thread's first filler step
                const int steps = K - s0;  // steps its fillers are spread over
                if (i >= s0) {
                    for (int f = (i - s0)*FILL/steps; f < (i - s0 + 1)*FILL/steps; ++f) {
                        if (!set.insert(h, scramble(K + t*FILL + f))) ++dup[t];
                    }
                } // if this thread's fillers have started
            } // key loop
        });

        for (int k = 0; k < K; ++k) {
            int c = 0;
            for (int t = 0; t < T; ++t) c += won[t][k];
            if (c > 1) ++over;
            if (c < 1) ++under;
            if (set.contains(h, scramble(k))) ++still_present;
        } // per-key check
        for (int t = 0; t < T; ++t) {
            filler_dup += dup[t];
            for (int f = 0; f < FILL; ++f) {
                if (!set.contains(h, scramble(K + t*FILL + f))) ++filler_missing;
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
        const Set::handle h = set.new_handle();
        std::vector<std::vector<int>> ins(T, std::vector<int>(R, 0));
        std::vector<std::vector<int>> ers(T, std::vector<int>(R, 0));

        run_threads(T, set, [&set, &ins, &ers, rep](const Set::handle& h, int t) {
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
                        if (set.insert(h, k)) ++ins[t][i];
                        break;
                    case 2:
                        if (set.erase(h, k)) ++ers[t][i];
                        break;
                    default:
                        set.contains(h, k);
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
            EXPECT_EQ(set.contains(h, scramble(i + 1)), d == 1)
                << "key index " << i << ": final membership disagrees with the insert/erase tally " << d
                << " (rep " << rep << ")";
        } // per-key check
    } // repetitions
} // InsertEraseChurnPerKeyAccounting

// An erase must not be undone by a resize, ever.
//
// The weak spot: a split COPIES live nodes forward and supersedes the originals
// in place, so for a while a key has a second node in an ancestor bucket (until
// the split's cleanup, a later writer or advance()'s walk unlinks it). If an
// erase marks only the authoritative copy and some later doubling then copies
// the stale node
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
        const Set::handle h = set.new_handle();

        // Phase 1: contended inserts of the shared keys during growth.
        std::vector<std::vector<unsigned char>> won(T, std::vector<unsigned char>(K, 0));
        run_threads(T, set, [&set, &won](const Set::handle& h, int t) {
            for (int i = 0; i < K; ++i) {
                const int k = (i + t*K/T)%K;
                if (set.insert(h, scramble(k))) won[t][k] = 1;
                for (int f = i*FILL/K; f < (i + 1)*FILL/K; ++f) set.insert(h, scramble(K + t*FILL + f));
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
            erase_wrong += !set.erase(h, scramble(k));   // present -> absent
            erase_wrong += set.erase(h, scramble(k));    // already absent
            erase_wrong += set.contains(h, scramble(k));
        } // erase sweep

        // Phase 3: growth of OTHER keys, with the erased keys under observation.
        std::atomic<bool> stop{false};
        std::atomic<size_t> seen{0};
        std::thread watcher([&set, &stop, &seen]() {
            const Set::handle h = set.new_handle();
            while (!stop.load(std::memory_order_acquire)) {
                for (int k = 0; k < K; ++k) {
                    if (set.contains(h, scramble(k))) seen.fetch_add(1, std::memory_order_relaxed);
                }
            } // poll until the hammers are done
        });
        run_threads(T, set, [&set](const Set::handle& own, int t) {   // `own`: `h` is the loop index here
            for (int h = 0; h < HAMMER; ++h) set.insert(own, scramble(K + T*FILL + t*HAMMER + h));
        });
        stop.store(true, std::memory_order_release);
        watcher.join();
        resurrected_live += seen.load();

        for (int k = 0; k < K; ++k) {
            if (set.contains(h, scramble(k))) ++resurrected_final;
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
    const Set::handle h = set.new_handle();
    std::vector<size_t> bad(T, 0);

    run_threads(T, set, [&set, &bad](const Set::handle& h, int t) {
        size_t b = 0;
        for (int i = 0; i < PER; ++i) {
            const int k = scramble(t*PER + i);
            b += !set.insert(h, k);     // absent -> present
            b += !set.contains(h, k);   // visible to its own inserter, immediately
            b += set.insert(h, k);      // already present
            b += !set.erase(h, k);      // present -> absent
            b += set.contains(h, k);    // invisible to its own eraser, immediately
            b += set.erase(h, k);       // already absent
            b += !set.insert(h, k);     // absent -> present again
            b += !set.contains(h, k);
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
            EXPECT_TRUE(set.contains(h, scramble(t*PER + i))) << "key lost, thread " << t << " index " << i;
        }
    } // membership sweep
} // ThreadPrivateKeySequenceDuringGrowth

// ===========================================================================
// Reclamation at a quiescent point: advance() with every handle released
// (advance_and_check()), the per-shard free lists popped lock-free by
// alloc_node(), the per-shard limbo lists, the per-shard retired lists that
// dead nodes reach when an operation unlinks them from their chain, and the
// bags advance() closes them into. Contract clauses 4 and 5 at the top of this
// file; the header's HANDLES AND GENERATIONS section and the contract on
// advance(). The concurrent side of reclamation -- advance() beside running
// operations, pinned generations, the hooks -- is the section "Handles,
// generations and advance()" further down.
//
// Every call to the quiescent-only diagnostics below (the accounting sweep and
// the free, limbo and retired counts derived from it), and every advance()
// whose result a test predicts exactly, happens after run_threads() has joined
// its workers (or on the only thread there is). Oracles are the contract's: the
// key count the test itself knows, the accounting identity (every slot in
// exactly one place), and "allocation takes a free slot before it grows the
// arena". Where a test relies on the documented node_count_ batching (256 per
// batch) or on the free pass's subtraction from it, it says so.
// ===========================================================================

// Pre-sized (no doubling), so every node the arena holds is a key or an erased
// node and every count is exact. One arena shard: the free pass deals freed
// slots round-robin over ALL shards, so with several shards the reinsertions
// would pop several free lists (the own one, then the others through the
// steal), and the first pop of each list adds 256 to the resize hint (see
// Sizes below): with enough shards that crosses the doubling threshold and the
// table doubles, which the premise "pre-sized, no doubling" excludes. With one
// shard every reinsertion is a pop from the one list (AdvanceReuseAcrossShards
// covers the multi-shard dealing).
// Weak spots: the bag and the free pass's walk of it. Every erase() here is
// uncontended, so it unlinks its own node at once and retires it (contract
// clause 5): the erased nodes reach advance() on a retired list, not in a
// chain, the walk finds nothing dead, and the K slots reach the free list only
// through the bump (every shard's retired list exchanged into the closing
// generation's bag) and the free pass (the bag's retired sublist walked through
// retire_link and dealt). "free == K" afterwards fails if the bag is skipped or
// walked through the wrong word. Before the first advance() the erased nodes
// are counted where they are -- retired, or still in a chain as dead nodes --,
// K in total whoever unlinked them. Also: advance() run twice with nothing to
// do in between (the second finds every retired and limbo list empty, returns
// `nothing_retired` and frees nothing new; a slot pushed twice would show as a
// cycle or a duplicate in the accounting sweep); and the free-list pop path
// serving every reinsertion, including the one that empties the list.
// Sizes: N = 1000 keys in 1024 buckets. The resize hint then peaks at 1024
// (four append batches of 256) before advance() and at 1024 - 250 (the free
// pass's subtraction) + 256 (the shard's first pop) after it, both below the
// doubling threshold 2*1024.
TEST(ConcurrentHashSetRcuTest, AdvanceExactPreSized) {
    const int N = 1000, K = 250;
    {   // AllowDelete == true: the erased nodes are the only dead nodes
        ConcurrentResizableHashSetRCU<int, true> set(1024, 1);
        ConcurrentResizableHashSetRCU<int, true>::handle h = set.new_handle();
        std::vector<int> kept, erased, fresh;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(h, scramble(i)));
        const ConcurrentResizableHashSetRCU<int, true>::InternalAccounting filled = expect_consistent(set, "after the fill");
        EXPECT_EQ(filled.slots, size_t(N));
        EXPECT_EQ(filled.reachable, size_t(N));
        for (int i = 0; i < N; ++i) {
            if (i%4 == 1) {   // every 4th key: the erased nodes sit anywhere in their chains
                EXPECT_TRUE(set.erase(h, scramble(i)));
                erased.push_back(scramble(i));
            } else {
                kept.push_back(scramble(i));
            }
        } // erase every 4th key
        ASSERT_EQ(erased.size(), size_t(K));
        const ConcurrentResizableHashSetRCU<int, true>::InternalAccounting before = expect_consistent(set, "after the erases");
        EXPECT_EQ(before.retired + before.reachable_dead, size_t(K)) << "an erased node is neither retired nor dead in its chain";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N - K, "first advance()"));
        EXPECT_EQ(set.get_internal_free_count(), size_t(K)) << "every erased node, and nothing else, is freed";
        EXPECT_EQ(membership_errors(set, h, kept, erased), 0);

        // Idempotence: nothing died since, so nothing more is freed.
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N - K, "second advance()"));
        EXPECT_EQ(set.get_internal_free_count(), size_t(K)) << "a second advance() freed something, or pushed a slot twice";

        // Reinsert K new keys: every one must be served from the free list.
        for (int i = 0; i < K; ++i) {
            fresh.push_back(scramble(N + i));
            EXPECT_TRUE(set.insert(h, fresh.back()));
        }
        EXPECT_EQ(set.get_internal_node_count(), size_t(N)) << "the arena grew although the free list held a node for every insert";
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        expect_consistent(set, "after the reinsertion");
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N, "third advance()"));
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        EXPECT_EQ(membership_errors(set, h, kept, erased), 0);
        EXPECT_EQ(membership_errors(set, h, fresh, std::vector<int>{}), 0);
    } // AllowDelete == true
    {   // AllowDelete == false: pre-sized, so nothing is ever dead
        ConcurrentResizableHashSetRCU<int, false> set(1024, 1);
        ConcurrentResizableHashSetRCU<int, false>::handle h = set.new_handle();
        std::vector<int> keys, absent;
        for (int i = 0; i < N; ++i) {
            keys.push_back(scramble(i));
            EXPECT_TRUE(set.insert(h, keys.back()));
            absent.push_back(scramble(N + i));
        }
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N, "AllowDelete=false, first advance()"));
        EXPECT_EQ(set.get_internal_free_count(), 0u) << "advance() freed a live node";
        for (int k : keys) EXPECT_FALSE(set.insert(h, k));   // duplicates: no allocation, no change
        EXPECT_EQ(set.get_internal_node_count(), size_t(N));
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N, "AllowDelete=false, second advance()"));
        EXPECT_EQ(membership_errors(set, h, keys, absent), 0);
    } // AllowDelete == false
} // AdvanceExactPreSized

// advance() straight after a growing insert burst, with doublings behind it and
// splits still pending, then the reuse that follows.
//
// Weak spot: the dead-node rule of advance()'s walk. A node in chain j whose key
// maps elsewhere at the current size is dead only if its NEXT CHILD bucket is
// PUBLISHED; while that child is still UNINITIALIZED the node is the key's only
// reachable node, and unlinking it loses the key. Right after a burst, a
// doubling has just left half the table UNINITIALIZED, which is exactly that
// state; an advance() after a contains() sweep would never see it (the sweep
// publishes every bucket the keys need). Hence many fresh sets, stopped at many
// sizes (1 to ~800 keys: on both sides of the doublings and of the 256-node
// batches), each reclaimed BEFORE anything reads it. Both AllowDelete values:
// with false, the dead nodes are the parent copies splits supersede, with true
// also the erased nodes. The burst is single-threaded, so the operations
// themselves unlink and retire all of those (contract clause 5) and advance()
// frees them from its bag; what its walk must decide is the live nodes whose
// child is still UNINITIALIZED, which it must keep (advance_and_check() checks
// that it retires exactly the reachable dead nodes, and the key count).
// The reuse phase has a trap: a reinsertion can land in a bucket that is still
// UNINITIALIZED and split it, which allocates copies. So the node-count baseline
// is taken only after a contains() sweep over every key, which publishes every
// bucket any existing key maps to (a split of any other bucket copies nothing).
// Even then a reinsertion can double the table and split again, so the oracle is
// the design rule itself: with one shard and one thread, the arena grows only
// once the free list is empty.
template <bool AllowDelete>
static void advance_with_pending_splits_body() {
    using Set = ConcurrentResizableHashSetRCU<int, AllowDelete>;
    const int sizes[] = {1, 2, 7, 9, 31, 64, 100, 129, 255, 257, 300, 500, 513, 777};
    int reuse_exercised = 0;
    for (int n : sizes) {
        const std::string where = std::string(AllowDelete ? "AllowDelete=true" : "AllowDelete=false") + ", n = " + std::to_string(n);
        Set set(4, 1);
        typename Set::handle h = set.new_handle();
        std::vector<int> in, out;
        for (int i = 0; i < n; ++i) EXPECT_TRUE(set.insert(h, scramble(i))) << where;
        for (int i = 0; i < n; ++i) {
            if constexpr (AllowDelete) {
                if (i%3 == 0) {
                    EXPECT_TRUE(set.erase(h, scramble(i))) << where;
                    out.push_back(scramble(i));
                    continue;
                }
            } // if erase() exists
            in.push_back(scramble(i));
        } // erase every third key
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), where + ", advance() right after the burst"));
        const size_t free1 = set.get_internal_free_count();
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), where + ", second advance()"));
        EXPECT_EQ(set.get_internal_free_count(), free1) << where << ": the second advance() freed something new";
        EXPECT_EQ(membership_errors(set, h, in, out), 0) << where << ": advance() changed membership";

        // Reuse, with the baseline taken after the contains() sweep just done. That
        // sweep's splits popped the free list (with AllowDelete == false it is
        // usually empty afterwards) and retired the parent copies they superseded;
        // reclaiming now recycles those, so the reuse below has free slots to take.
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), where + ", advance() after the settling sweep"));
        const size_t slots0 = set.get_internal_node_count();
        const size_t free0 = set.get_internal_free_count();
        for (size_t i = 0; i < free0/2; ++i) {
            in.push_back(scramble(1000000 + n + static_cast<int>(i)));
            EXPECT_TRUE(set.insert(h, in.back())) << where;
        }
        const size_t free2 = set.get_internal_free_count();
        EXPECT_TRUE(free2 == 0 || set.get_internal_node_count() == slots0)
            << where << ": the arena grew from " << slots0 << " to " << set.get_internal_node_count()
            << " while the free list still held " << free2 << " node(s)";
        if (free0 > 0 && free2 > 0) ++reuse_exercised;
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), where + ", advance() after the reuse"));
        EXPECT_EQ(membership_errors(set, h, in, out), 0) << where << ": membership after the reuse";
    } // for each burst size
    // Not an oracle on the header: a guard that the reuse check above is not vacuous.
    EXPECT_GT(reuse_exercised, 0) << "no size exercised reuse with free nodes left over";
} // advance_with_pending_splits_body()

TEST(ConcurrentHashSetRcuTest, AdvanceWithPendingSplits) {
    advance_with_pending_splits_body<true>();
    advance_with_pending_splits_body<false>();
} // AdvanceWithPendingSplits

// AllowDelete == false, concurrent growth: the dead nodes are the parent copies
// splits supersede. The split FREEZES each of them (for both AllowDelete values),
// and the split that superseded it unlinks and retires it -- or, when that
// split's one cleanup CAS lost to a concurrent writer, a later writer or
// advance()'s walk does (the rule "next child published"). Four threads insert
// disjoint scrambled keys, one arena shard each. Phase 1 reclaims straight after
// the join (pending splits, as in AdvanceWithPendingSplits but with nodes of
// four shards), which leaves no dead node anywhere. Phase 2 then settles the
// keys' splits with
// a single-threaded contains() sweep (it publishes only the buckets the probed
// keys map to): each of its splits supersedes the parent copy of every key that
// moves and, uncontended, retires them all (contract clause 5). So after the
// sweep exactly one node per key is reachable, none of them dead, and the retired
// lists hold the superseded copies (more than none, or the phase tests nothing),
// which advance() then recycles. Every insert() boolean is exact: the keys are
// disjoint and each is inserted once, so each call must return true.
TEST(ConcurrentHashSetRcuTest, AdvanceStaleSplitCopiesNoDelete) {
    using Set = ConcurrentResizableHashSetRCU<int, false>;
    const int T = 4, PER = 500, REPS = 4;
    for (int rep = 0; rep < REPS; ++rep) {
        const std::string where = "rep " + std::to_string(rep);
        Set set(4, T);
        Set::handle h = set.new_handle();
        std::vector<size_t> bad(T, 0);
        run_threads(T, set, [&set, &bad, rep](const Set::handle& h, int t) {
            for (int i = 0; i < PER; ++i) bad[t] += !set.insert(h, scramble((rep*T + t)*PER + i));
        });
        for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ": insert() of a new, uncontended key returned false";
        std::vector<int> in, out;
        for (int i = 0; i < T*PER; ++i) in.push_back(scramble(rep*T*PER + i));
        for (int i = 0; i < 100; ++i) out.push_back(scramble(1000000 + i));
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, T*PER, where + ", phase 1"));
        EXPECT_EQ(membership_errors(set, h, in, out), 0) << where << ": advance() with pending splits changed membership";
        const Set::InternalAccounting swept = expect_consistent(set, where + ", after the settling sweep");
        EXPECT_EQ(swept.reachable, size_t(T*PER)) << where << ": an uncontended split left a superseded parent copy reachable";
        EXPECT_EQ(swept.reachable_dead, 0u) << where << ": a dead node is reachable after single-threaded splits";
        // EMPIRICAL precondition, not derived: with scrambled keys over many doublings
        // it is overwhelmingly likely that the sweep's splits moved many keys, but
        // nothing forces it.
        ASSERT_GT(swept.retired, 0u) << where << ": test precondition: the sweep's splits superseded no parent copy";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, T*PER, where + ", phase 2"));
        EXPECT_EQ(membership_errors(set, h, in, out), 0) << where << ": advance() of superseded copies changed membership";
        for (int k : in) EXPECT_FALSE(set.insert(h, k)) << where << ": a duplicate insert() returned true after advance()";
    } // repetitions
} // AdvanceStaleSplitCopiesNoDelete

// Free slots dealt over the shards, and popped concurrently by threads of other
// shards. The slots all live in the MAIN thread's shard deque (it inserted
// everything); advance()'s free pass deals them round-robin over every shard's
// free list; the worker threads then pop from their own shards' lists. So every
// worker that is not on the main thread's shard reuses slots that belong to
// another shard.
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
// new keys: exactly what its shard's list can serve, shared with its shard mates,
// so no thread's own list runs dry and no allocation needs another shard's.
// So the arena must not grow at all, and the free count must drop by exactly the
// number of inserts. (An allocation whose own list is empty takes a node from
// another shard's list before it appends, so an uneven deal shows here as no
// arena growth at all; the deal's evenness is checked per shard by
// FreedNodesDoNotStrandOnUnusedShards.)
// The K erases run on the main thread, uncontended, so each unlinks its own node
// and retires it onto the MAIN thread's shard's retired list (contract clause 5):
// before the first advance() all K are retired and none is left dead in a chain
// (both checked), the walk finds nothing, and the K slots reach the free lists
// only through the bag (one shard's retired sublist), which the free pass deals
// round-robin like every other source.
// Pre-sizing: 16384 buckets, doubling threshold 32768; the resize hint peaks at
// 4096 appended - 2048 freed + 8 first pops of 256, far below it.
TEST(ConcurrentHashSetRcuTest, AdvanceReuseAcrossShards) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    struct Shape { int threads; size_t shards; };
    const int N = 4096, K = 2048;
    for (const Shape shape : {Shape{8, 2}, Shape{2, 8}}) {
        const int T = shape.threads;
        const int per = K/static_cast<int>(std::max<size_t>(shape.shards, T));
        const std::string where = std::to_string(T) + " threads, " + std::to_string(shape.shards) + " shards";
        Set set(16384, shape.shards);
        Set::handle h = set.new_handle();
        std::vector<int> in, out;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(h, scramble(i)));
        for (int i = 0; i < N; ++i) {
            if (i & 1) {
                in.push_back(scramble(i));
            } else {
                EXPECT_TRUE(set.erase(h, scramble(i)));
                out.push_back(scramble(i));
            }
        } // erase the even-indexed keys
        const Set::InternalAccounting erased = expect_consistent(set, where + ", after the erases");
        EXPECT_EQ(erased.retired, size_t(K)) << where << ": an uncontended erase() did not retire its node";
        EXPECT_EQ(erased.reachable_dead, 0u) << where << ": an uncontended erase() left its node in its chain";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N - K, where + ", after the erases"));
        ASSERT_EQ(set.get_internal_free_count(), size_t(K)) << where;
        const size_t slots = set.get_internal_node_count();

        std::vector<size_t> bad(T, 0);
        run_threads(T, set, [&set, &bad, per](const Set::handle& h, int t) {
            for (int i = 0; i < per; ++i) bad[t] += !set.insert(h, scramble(N + t*per + i));
        });
        for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ": insert() of a new, uncontended key returned false";
        for (int i = 0; i < T*per; ++i) in.push_back(scramble(N + i));
        EXPECT_EQ(set.get_internal_node_count(), slots) << where << ": the arena grew although every thread's shard held a free slot for each of its inserts";
        EXPECT_EQ(set.get_internal_free_count(), size_t(K - T*per)) << where;
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), where + ", after the reuse"));
        EXPECT_EQ(membership_errors(set, h, in, out), 0) << where;
    } // for each shape
} // AdvanceReuseAcrossShards

// After advance(), the structure is a mix of KEPT nodes (in chains whose dead
// neighbours were unlinked by an operation's or advance()'s walk's unlink CAS)
// and REUSED nodes (popped from a free list, value copy-assigned, link
// rewritten). Both must take part in the concurrent protocol exactly like fresh
// nodes: the erase MARK CAS, the split FREEZE CAS and the UNLINK CAS expect a
// live link, the split SEAL CAS expects the head the unlinks left behind (with
// its seal level kept), and a split copies from chains that were unlinked
// through. This test runs concurrent erases and inserts, with the table
// doubling, over such a structure, round after round with an advance() between
// rounds so slots are recycled more than once.
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
// the resize hint above 2*table_size (see WHEN THE TABLE DOUBLES), the free
// pass lowers the hint by every node it frees while the table keeps its size,
// and each round's doublings at least double that threshold; a round
// allocating no more than the one before it does not reach it (with a fixed
// count, only the first round doubles), and doubling the count leaves some
// rounds short of it.
// Keys scrambled (see scramble()).
TEST(ConcurrentHashSetRcuTest, AdvanceThenConcurrentEraseAndGrowth) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int T = 4, REPS = 12, ROUNDS = 3, A = 96, F0 = 48;
    int next = 0;   // next unused key index; keys are scramble(index), all distinct
    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4, 2);
        Set::handle h = set.new_handle();
        std::vector<int> present, absent;
        for (int i = 0; i < A; ++i) {   // phase A, single-threaded: half the keys become tombstones
            const int k = scramble(next++);
            EXPECT_TRUE(set.insert(h, k));
            if (i & 1) {
                EXPECT_TRUE(set.erase(h, k));
                absent.push_back(k);
            } else {
                present.push_back(k);
            }
        } // phase A
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, present.size(), "rep " + std::to_string(rep) + ", phase A"));

        for (int round = 0; round < ROUNDS; ++round) {
            const std::string where = "rep " + std::to_string(rep) + ", round " + std::to_string(round);
            const int F = F0 << (2*round);   // fresh keys per thread this round (see "Shape" above)
            std::vector<std::vector<int>> old(T), fresh(T);
            for (size_t i = 0; i < present.size(); ++i) old[i%T].push_back(present[i]);
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < F; ++i) fresh[t].push_back(scramble(next++));
            }
            std::vector<size_t> bad(T, 0);
            run_threads(T, set, [&set, &old, &fresh, &bad](const Set::handle& h, int t) {
                const std::vector<int>& mine = old[t];
                const std::vector<int>& fr = fresh[t];
                size_t b = 0;
                for (size_t i = 0; i < std::max(mine.size(), fr.size()); ++i) {
                    if (i < fr.size()) {
                        b += !set.insert(h, fr[i]);        // absent -> present (a reused slot, early on)
                        b += !set.contains(h, fr[i]);
                        if (i & 1) {
                            b += !set.erase(h, fr[i - 1]);  // present -> absent
                            b += set.contains(h, fr[i - 1]);
                            b += set.erase(h, fr[i - 1]);   // already absent
                        }
                    } // fresh key
                    if (i < mine.size()) {
                        b += !set.erase(h, mine[i]);       // a kept node, or one reused in an earlier round
                        b += set.contains(h, mine[i]);
                        b += !set.insert(h, mine[i]);
                        b += !set.contains(h, mine[i]);
                    } // surviving key
                } // key loop
                bad[t] = b;
            });
            for (int t = 0; t < T; ++t) {
                EXPECT_EQ(bad[t], 0u) << where << ", thread " << t << ": a determined result was wrong";
                for (int i = 0; i < F; ++i) ((i & 1) ? present : absent).push_back(fresh[t][i]);
            }
            ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, present.size(), where));
            EXPECT_EQ(membership_errors(set, h, present, absent), 0) << where;
        } // rounds
    } // repetitions
} // AdvanceThenConcurrentEraseAndGrowth

// Instrumented key for the reuse-by-assignment tests: counts copy constructions,
// copy assignments and destructions, can be armed to throw from its copy
// assignment, and carries a MAGIC word that every operation checks, so a slot
// whose value was destroyed (or never constructed) and then used again is
// counted in `corrupt`. The destructor clears MAGIC for that purpose.
// Its throwing assignment throws BEFORE touching the left-hand side, i.e. it
// gives the strong guarantee the header asks of T ("a throwing assignment must
// leave the old object valid").
// It also carries a one-shot HOOK, run from inside a copy assignment, which is
// how AdvanceRecoversPendingInsertNodeWhenRetryThrows injects another operation
// between an insert()'s read of a bucket head and its publishing CAS.
// The counters and the hooks are global, so ProbeKey is used only by
// single-threaded tests.
//
// CALL HOOKS (hash_hook, equal_hook), for the tests that suspend an operation at
// one of its pinned call sites and run a nested operation, or throw, there: an
// interleaving another thread could produce, made deterministic on one thread.
// The header pins the order in which every operation loads a word, hashes or
// compares a key, and performs its deciding CAS (its PINNED ORDERS block before
// contains() names them H1-H6; every H-number in this file's comments is one of
// those), so a hook that fires on a counted call opens a known window. Rules:
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
//   - A nested operation in an action runs under the OUTER thread's handle, the
//     one the outer operation is in flight under: that is the thread's handle
//     at that instant, operations may share a handle, and a handle of its own
//     would add a join inside the window the hook opens. (The same holds for
//     on_assign's action, which runs inside an outer insert() too.)
//   - hook_depth separates the outer operation's calls from the nested ones:
//     while an action runs, no call counts toward a hook or toward the outer
//     tally, so the nested operations' own hashing (and that of any accounting
//     sweep an action reads) neither fires a hook nor shifts a count.
//   - Tests disarm every hook right after the outer call returns or throws, and
//     before any advance(), settle_writers() or sweep: get_internal_accounting()
//     hashes every reachable unmarked node, and advance()'s walk every FROZEN
//     node it passes, and would otherwise fire an armed hook. The exception is
//     a test whose hook is MEANT to fire in advance()'s walk (the in-walk seam
//     of the "Handles, generations and advance()" section).
//   - A nested advance() in an action is a different matter: the outer
//     thread's handle, in flight, pins its generation, so an advance() there
//     frees nothing at or after it, and a nested operation run under the outer
//     handle belongs to that generation too. A test whose action advances runs
//     the action's operations under handles created and released inside the
//     action, before each advance(), and the outer handle is then the pin
//     under test (SuspendedReaderPinsItsGeneration).
//   - A throwing action, or an armed throw_countdown, must never be reachable
//     from operations or advance() that a TestHooks action runs (see
//     TestHooks): the set calls its hooks from noexcept functions, so the
//     throw would terminate the process. hook_depth does not help there: a
//     TestHooks action does not raise it.
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

// Reuse copy-assigns, advance() destroys nothing, and a throwing assignment on a
// popped slot loses nothing.
//
// Weak spot: the one exception path in alloc_node(). The slot has already left
// the free list when the assignment throws, so unless it is recorded somewhere
// it is gone for good; the design sends it to the shard's limbo list, keeps the
// old value (the assignment never completed), and propagates the exception. The
// next advance() must bag the limbo node and free it like any other.
// Checked, in order: the exception reaches the caller of insert(); the key is not
// in the set; the slot is on limbo (limbo 1, free list one shorter, arena
// unchanged, accounting consistent); nothing touched a destroyed object; the
// retried insert succeeds from the next free slot; advance() runs no destructor
// and frees the limbo node; and a reinsertion of exactly as many keys as there
// are free slots, the formerly-limbo one included, empties the free list
// without growing
// the arena, with one copy assignment and no copy construction per key.
// Pre-sized, one shard, single-threaded (ProbeKey's counters are global).
TEST(ConcurrentHashSetRcuTest, AdvanceThrowingAssignment) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int N = 64, E = 16;
    ProbeKey::reset_counters();
    {
        Set set(1024, 1);
        Set::handle h = set.new_handle();
        std::vector<int> in, out;
        for (int i = 0; i < N; ++i) EXPECT_TRUE(set.insert(h, scramble(i)));
        for (int i = 0; i < N; ++i) {
            if (i < E) {
                EXPECT_TRUE(set.erase(h, scramble(i)));
                out.push_back(scramble(i));
            } else {
                in.push_back(scramble(i));
            }
        } // erase the first E keys
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, N - E, "after the erases"));
        ASSERT_EQ(set.get_internal_free_count(), size_t(E));
        const size_t slots = set.get_internal_node_count();

        ProbeKey::copy_assignments = 0;
        ProbeKey::throw_countdown = 1;   // the next assignment throws
        EXPECT_THROW(set.insert(h, scramble(N)), std::runtime_error);
        ProbeKey::throw_countdown = 0;
        EXPECT_FALSE(set.contains(h, scramble(N))) << "the key whose insert() threw is in the set";
        EXPECT_EQ(set.get_internal_limbo_count(), 1u) << "the popped slot must go to limbo";
        EXPECT_EQ(set.get_internal_free_count(), size_t(E - 1));
        EXPECT_EQ(set.get_internal_node_count(), slots);
        expect_consistent(set, "after the throwing insert()");
        EXPECT_EQ(ProbeKey::copy_assignments.load(), 0);

        EXPECT_TRUE(set.insert(h, scramble(N)));   // the retry pops the next free slot
        in.push_back(scramble(N));
        EXPECT_EQ(set.get_internal_free_count(), size_t(E - 2));
        EXPECT_EQ(set.get_internal_limbo_count(), 1u);

        ProbeKey::destructions = 0;
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), "after the retry"));
        EXPECT_EQ(ProbeKey::destructions.load(), 0) << "advance() must not destroy values (they are assigned over at reuse)";
        const size_t free_slots = set.get_internal_free_count();
        EXPECT_EQ(free_slots, size_t(E - 1)) << "the limbo slot must be back on a free list";

        ProbeKey::copy_assignments = 0;
        ProbeKey::copy_constructions = 0;
        for (size_t i = 0; i < free_slots; ++i) {
            in.push_back(scramble(N + 1 + static_cast<int>(i)));
            EXPECT_TRUE(set.insert(h, in.back()));
        }
        EXPECT_EQ(set.get_internal_node_count(), slots) << "the arena grew although the free list (with the limbo slot) held a slot per insert";
        EXPECT_EQ(set.get_internal_free_count(), 0u);
        EXPECT_EQ(ProbeKey::copy_assignments.load(), static_cast<int>(free_slots)) << "a reused slot takes its value by copy assignment";
        EXPECT_EQ(ProbeKey::copy_constructions.load(), 0) << "reuse constructed a value";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, in.size(), "after the reuse"));
        EXPECT_EQ(membership_errors(set, h, in, out), 0);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0) << "an operation met a destroyed or never-constructed value";
} // AdvanceThrowingAssignment

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
// fill_key(0 .. erase-1); reclaims (advance_and_check(): `h` released, advance(),
// `h` joined again), checking the result. `next` receives the index of the first
// fill key not yet used. `h`: the calling thread's handle of `set`, its only live
// handle. Callers wrap it in ASSERT_NO_FATAL_FAILURE.
// The arithmetic, from the header's documented node_count_ (the resize hint; a
// doubling happens after a successful insert that sees it above 2*table_size,
// i.e. above 512 here): the appends at indices 0 and 256 add 256 each, so the
// hint reaches exactly 512 and the table does not double. The free pass then
// subtracts the `erase` nodes it frees: 512 - erase. The shard's free list has
// never been popped, so its pop counter is 0 and its first pop adds 256: that pop
// doubles the table iff 512 - erase + 256 > 512, i.e. iff erase < 256.
template <typename SetT>
static void fill_to_512_and_advance(SetT& set, typename SetT::handle& h, const std::vector<int>& first, int erase, int& next) {
    for (int k : first) EXPECT_TRUE(set.insert(h, k));
    const int fills = 512 - static_cast<int>(first.size());
    for (int i = 0; i < fills; ++i) EXPECT_TRUE(set.insert(h, fill_key(i)));
    for (int i = 0; i < erase; ++i) EXPECT_TRUE(set.erase(h, fill_key(i)));
    ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, 512 - erase, "fill_to_512_and_advance()"));
    EXPECT_EQ(set.get_internal_free_count(), size_t(erase));
    ASSERT_EQ(set.get_internal_node_count(), 512u) << "test precondition: the fill did not end at 512 slots";
    ASSERT_EQ(set.get_internal_table_size(), 256u) << "test precondition: the fill doubled the table";
    next = fills;
} // fill_to_512_and_advance()

// advance()'s free pass lowers the resize hint by every node it frees, and the
// pops' batching of the hint runs on across advance() calls.
//
// Weak spot: node_count_, the resize hint. Nothing resets it: the free
// pass subtracts each node it frees, and a pop adds 256 whenever the shard's pop
// counter (packed into its free-head word, never cleared) is at a multiple of
// 256, which keeps the hint an over-count at every quiescent point (the table may
// double early, never late). Two ways to break it, neither visible to any oracle
// of membership, booleans or accounting -- only the moment of a doubling moves:
//   - a free pass that subtracts nothing: the hint then only grows under churn,
//     and the table doubles over and over while the live count stays flat
//     (ChurnWithAdvanceKeepsTableBounded is the churn form of the same kill);
//   - a free pass, or a bump, that clears the pop counters: a plain store racing
//     the pops' CASes, and the first pop after the call then adds 256 again.
// So this test builds the arithmetic of fill_to_512_and_advance() and watches the
// table size, on a fill of fill keys only (nothing to split):
//   1. fill, erase 300 fill keys, advance(): hint 512 - 300 = 212 (512 without
//      the subtraction); 300 nodes free; the pop counter 0;
//   2. 100 inserts, all pops. Pop 1 adds 256, pops 2..100 nothing: hint 468, no
//      doubling (768 without the subtraction: the first insert doubles). The
//      counter stands at 100;
//   3. erase those 100 keys, advance(): hint 368, 300 nodes free again;
//   4. 156 inserts, pops 101..256: none adds (a counter cleared at step 3 would
//      make pop 101 add 256: hint 624, a doubling), so the table keeps 256
//      buckets; the 157th, pop 257, adds 256: hint 624 > 512, and that insert
//      doubles the table.
TEST(ConcurrentHashSetRcuTest, AdvanceLowersResizeHintAndKeepsPopBatching) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CollisionHash>;
    Set set(256, 1);
    Set::handle h = set.new_handle();
    int next = 0;
    ASSERT_NO_FATAL_FAILURE(fill_to_512_and_advance(set, h, {}, 300, next));
    size_t live = 212;

    std::vector<int> cycle1;
    for (int i = 0; i < 100; ++i) {   // step 2
        cycle1.push_back(fill_key(next++));
        EXPECT_TRUE(set.insert(h, cycle1.back()));
        ASSERT_EQ(set.get_internal_table_size(), 256u) << "pop " << i + 1 << " doubled the table: the free pass did not lower the hint by the 300 nodes it freed";
    } // cycle 1
    live += 100;
    ASSERT_EQ(set.get_internal_node_count(), 512u) << "test precondition: step 2 must be served from the free list";
    for (int k : cycle1) EXPECT_TRUE(set.erase(h, k));   // step 3
    live -= 100;
    ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "step 3"));
    ASSERT_EQ(set.get_internal_free_count(), 300u) << "test precondition";

    for (int i = 0; i < 156; ++i) {   // step 4: pops 101..256
        EXPECT_TRUE(set.insert(h, fill_key(next++)));
        ASSERT_EQ(set.get_internal_table_size(), 256u) << "pop " << 101 + i << " doubled the table: the pop counter was cleared, or the hint batched wrongly";
    } // pops 101..256
    EXPECT_TRUE(set.insert(h, fill_key(next++)));   // pop 257
    live += 157;
    EXPECT_EQ(set.get_internal_table_size(), 512u) << "pop 257 did not add its batch to the resize hint: the table did not double";
    ASSERT_EQ(set.get_internal_node_count(), 512u) << "test precondition: step 4 must be served from the free list";
    ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after step 4"));
} // AdvanceLowersResizeHintAndKeepsPopBatching

// Builds a FROZEN node whose split never finished, for the tests below: a set
// built as SetT(256, 1) over ProbeKey is filled by fill_to_512_and_advance() with
// `first` (which must end with PROBE_X2 and contain PROBE_X1, every other key in
// bucket 5 at sizes up to 1024 and absent from the fill keys), erasing 200 fill
// keys; one more insert, the shard's first pop (hint 512 - 200 + 256 = 568, see
// fill_to_512_and_advance()), doubles the table to 512; and
// contains(PROBE_X1) then splits bucket 261, which walks bucket 5's chain newest
// first, freezes PROBE_X2 (the first node of the chain that moves: fill_key(261)
// = 1029, the only newer node, stays) and throws on its copy, whose popped slot's
// assignment is armed to throw. Afterwards bucket 5's chain is, newest first,
//     1029 -> PROBE_X2 (FROZEN, child 261 UNINITIALIZED) -> rest of `first`
// reversed (PROBE_X1, still live, last), limbo holds the throwing slot, and
// nothing is retired or bagged: an advance() with every handle released then
// frees exactly that one limbo node (and is `advanced`: limbo counts), and its
// walk hashes PROBE_X2 (the FROZEN node right after the live 1029), which is
// the in-walk seam of AdvanceWalkHoldsTheFlag.
// fill_key(5), the other fill key of bucket 5, was among the erased ones.
// `next` receives the first unused fill index and `live` the key count; `h` is
// the calling thread's handle of `set`, its only live handle (released and
// joined again by the fill's reclamation). Callers wrap it in
// ASSERT_NO_FATAL_FAILURE; it ASSERTs its own steps.
template <typename SetT>
static void build_unfinished_split(SetT& set, typename SetT::handle& h, const std::vector<int>& first, int& next, size_t& live) {
    ASSERT_EQ(first.back(), PROBE_X2);
    ASSERT_NO_FATAL_FAILURE(fill_to_512_and_advance(set, h, first, 200, next));
    live = 312;
    ASSERT_TRUE(set.insert(h, fill_key(next++)));   // the shard's first pop: doubles the table to 512
    ++live;
    ASSERT_EQ(set.get_internal_table_size(), 512u) << "test precondition: the shard's first pop did not double the table";
    ProbeKey::throw_countdown = 1;   // the split's first copy throws
    EXPECT_THROW(set.contains(h, PROBE_X1), std::runtime_error);
    ProbeKey::throw_countdown = 0;
    const typename SetT::InternalAccounting acc = expect_consistent(set, "after the split threw");
    ASSERT_EQ(acc.limbo, 1u) << "the slot whose assignment threw must be on limbo";
    ASSERT_EQ(acc.reachable_dead, 0u) << "test precondition: PROBE_X2 is FROZEN with its child unpublished, so not dead";
    ASSERT_EQ(acc.retired, 0u) << "test precondition: a node was retired";
    ASSERT_EQ(acc.bagged, 0u) << "test precondition: a node was bagged";
} // build_unfinished_split()

// A FROZEN node whose split never finished is KEPT by advance()'s walk, and the
// dead node after it STAYS in its chain until the split of the child collects it.
//
// Weak spots: (1) the dead-node rule must keep a FROZEN node whose child bucket is
// still UNINITIALIZED: it is then the key's only node (INVARIANT in the header);
// (2) no walk may unlink a node through such a predecessor's tagged link: not
// erase()'s self-unlink, and not advance()'s walk; (3) the run behind it is
// collected by the split of the child, once that split publishes.
// A FROZEN node with an unpublished child exists at a quiescent point only when a
// split threw between the freeze and the publication, so this test makes one
// throw (build_unfinished_split()): the first copy the split of bucket 261 makes
// pops a free slot whose assignment throws. The split froze exactly one node when
// that happens (the first moving node of the walk), and allocated nothing yet, so
// nothing leaks (compare AdvanceAccountsForSplitCopiesWhenAssignmentThrows below,
// where the SECOND copy throws).
// Chain order is newest first; the insert order PROBE_X1, PROBE_E, PROBE_X2 makes
// bucket 5's chain 1029 -> PROBE_X2 (FROZEN) -> PROBE_E -> PROBE_X1. erase(PROBE_E)
// marks PROBE_E and then cannot unlink it: its predecessor PROBE_X2 is tagged and
// not dead, and a tagged link is never a CAS target, so the erase gives up its
// self-unlink (contract clause 5's exception) and nothing is retired; PROBE_E
// stays reachable, dead. A header that unlinked through the tagged predecessor
// would retire PROBE_E; one that treated a FROZEN node as dead with its child
// unpublished would unlink PROBE_X2 too and lose the key. Then advance(), every
// handle released: it frees the limbo slot, and its walk passes the live 1029,
// hashes the FROZEN PROBE_X2 (not dead) and must retire NOTHING: PROBE_E is a
// dead node no walk can bypass (advance_and_check()'s `stuck` = 1), so the
// same two defects in the walk show as a walk retiree, or a lost key. Afterwards
// both probe keys must still be members, and the contains() that finds PROBE_X1
// splits bucket 261 and publishes it, so its cleanup collects the whole parent
// run, PROBE_X2, PROBE_E and PROBE_X1, in one CAS through 1029's live link;
// the next reclamation then has nothing stuck.
TEST(ConcurrentHashSetRcuTest, AdvanceKeepsFrozenNodeOfUnfinishedSplit) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        Set::handle h = set.new_handle();
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, h, {PROBE_X1, PROBE_E, PROBE_X2}, next, live));
        const size_t reachable = set.get_internal_accounting().reachable;

        EXPECT_TRUE(set.erase(h, PROBE_E));   // the dead node behind the FROZEN one
        --live;
        const Set::InternalAccounting erased = expect_consistent(set, "after the erase");
        EXPECT_EQ(erased.retired, 0u) << "erase() unlinked its node through a tagged predecessor, or unlinked the FROZEN node of an unfinished split";
        EXPECT_EQ(erased.reachable, reachable) << "PROBE_E, or PROBE_X2, left the chain";
        EXPECT_EQ(erased.reachable_dead, 1u) << "PROBE_E must stay in its chain, dead";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after the erase", 1));
        EXPECT_TRUE(set.contains(h, PROBE_X1));   // splits bucket 261: the run behind 1029 is collected
        const Set::InternalAccounting split = expect_consistent(set, "after the split of 261");
        EXPECT_EQ(split.reachable_dead, 0u) << "the split of 261 did not collect the run behind 1029";
        EXPECT_EQ(split.retired, 3u) << "PROBE_X2, PROBE_E and PROBE_X1 must be retired by the split's cleanup";
        EXPECT_TRUE(set.contains(h, PROBE_X2));
        EXPECT_FALSE(set.contains(h, PROBE_E));
        EXPECT_FALSE(set.insert(h, PROBE_X2)) << "a key kept only by its FROZEN node was lost";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after the split finished"));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceKeepsFrozenNodeOfUnfinishedSplit

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
        Set::handle h = set.new_handle();
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, h, {PROBE_X1, PROBE_E, PROBE_X2}, next, live));
        const Set::InternalCounters before = set.get_internal_counters();

        bool y1_inserted = false, y2_inserted = false;
        ProbeKey::arm(ProbeKey::equal_hook, HEAD, 1, [&set, &h, &y1_inserted, &y2_inserted, Y1, Y2]() {   // (1)
            y1_inserted = set.insert(h, Y1);
            ProbeKey::arm(ProbeKey::hash_hook, PROBE_X2, 2, [&set, &h, &y2_inserted, Y2]() {   // (2)
                y2_inserted = set.insert(h, Y2);
            });
        });
        const bool erased = set.erase(h, HEAD);
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
        EXPECT_FALSE(set.contains(h, HEAD));
        EXPECT_TRUE(set.contains(h, Y1));
        EXPECT_TRUE(set.contains(h, Y2));
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after erase(1029)"));
        EXPECT_TRUE(set.contains(h, PROBE_X2));
        EXPECT_TRUE(set.contains(h, PROBE_E));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // EraseGivesUpAfterTwoLostUnlinks

// What erase() leaves in a chain when it cannot unlink, and what advance()'s walk
// does with it: a tombstone at the bucket head left by an erase() whose Hash{}
// threw after the mark is unlinked by the walk through the head word (which keeps
// its seal level), retired into the new generation and freed by the NEXT call;
// a run of tombstones behind a FROZEN node whose split never finished stays (no
// walk CASes a tagged link) until the split of the child collects it.
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
//   - four dead nodes are reachable, none retired; advance(), every handle
//     released, frees the builder's limbo slot, and its walk must unlink and
//     retire exactly the head tombstone (one walk retiree, freed by the second
//     call advance_and_check() makes) and leave the three behind PROBE_X2
//     (`stuck` = 3: one run, which the walk passes node by node behind the
//     FROZEN predecessor), with the exact key count; every remaining key must
//     still be a member. The contains(PROBE_X1) that checks it splits bucket
//     261, whose
//     cleanup collects the run behind the head, PROBE_X2 to PROBE_X1, and the
//     next reclamation has nothing stuck.
TEST(ConcurrentHashSetRcuTest, AdvanceRelinksWhatEraseLeftBehind) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int HEAD = fill_key(261), E2 = 5 | (6 << 10), E3 = 5 | (5 << 10);
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        Set::handle h = set.new_handle();
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, h, {PROBE_X1, E3, E2, PROBE_E, PROBE_X2}, next, live));
        for (int k : {PROBE_E, E2, E3}) {
            EXPECT_TRUE(set.erase(h, k));
            --live;
        }
        const Set::InternalAccounting run = expect_consistent(set, "after the erases behind PROBE_X2");
        EXPECT_EQ(run.retired, 0u) << "an erase() unlinked its node through the tagged PROBE_X2";
        EXPECT_EQ(run.reachable_dead, 3u);

        ProbeKey::arm(ProbeKey::hash_hook, PROBE_X2, 1, []() { throw std::runtime_error("Hash{} armed to throw"); });
        EXPECT_THROW(set.erase(h, HEAD), std::runtime_error);
        ProbeKey::disarm_hooks();
        --live;   // the mark landed: 1029 is deleted although the caller saw an exception
        const Set::InternalAccounting left = expect_consistent(set, "after erase(1029) threw");
        EXPECT_EQ(left.retired, 0u);
        ASSERT_EQ(left.reachable_dead, 4u) << "test precondition: erase(1029) did not leave its tombstone at the head";
        EXPECT_FALSE(set.contains(h, HEAD));

        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after the erases", 3));
        for (int k : {PROBE_X1, PROBE_X2}) EXPECT_TRUE(set.contains(h, k));
        for (int k : {HEAD, PROBE_E, E2, E3}) EXPECT_FALSE(set.contains(h, k));
        EXPECT_EQ(expect_consistent(set, "after the split of 261").reachable_dead, 0u) << "the split of 261 did not collect the run behind the head";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after the split finished"));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceRelinksWhatEraseLeftBehind

// NO ARENA SLOT IS LOST TO A THROW (header, "EXCEPTIONS: WHAT IS AND IS NOT
// HANDLED"): the partial subchain of a split that throws; and a LIMBO-ONLY state
// through advance().
//
// Weak spots: (1) split_bucket() builds its child subchain privately, referenced
// by nothing but the local head and tail. When an allocation in the middle of the
// walk throws (a popped slot's copy assignment, as here, or the append's copy
// construction), the copies made so far must go to limbo before the exception
// leaves, or they are lost for the life of the set. (2) The bag's LIMBO sublist:
// the bump exchanges a shard's limbo list into the bag, and the free pass walks
// it through `link`, which is also the word it rewrites into the free list's
// next pointer: a free pass that reads a node's next word AFTER that rewrite
// frees the first node and loses the rest. That needs at least two limbo nodes
// in one shard's sublist, which this state has. (3) A state with limbo nodes and
// nothing retired is not `nothing_retired`: limbo counts.
// Same setup as AdvanceKeepsFrozenNodeOfUnfinishedSplit, but the split has two
// keys to copy (PROBE_X2, then PROBE_X1) and the SECOND copy throws, so the
// subchain holds one node at the throw. Checked: the exception reaches the
// caller; limbo holds exactly two nodes, both on the one shard's list (the slot
// whose assignment threw, pushed by alloc_node(), and the one-node partial
// subchain, pushed by split_bucket()); nothing is retired or bagged; the
// accounting is consistent. Then advance(), every handle released, BEFORE the
// split is finished: `advanced`, both limbo nodes freed, nothing for the walk
// (both probe keys are FROZEN in bucket 5 with 261 unpublished: not dead, and
// the walk must keep them, as the key count checks). Then both probe keys are
// still members (the contains() that checks the first one splits 261 again, and
// this time it finishes, retiring both FROZEN parent copies), and a second
// reclamation recycles those with the exact key count.
TEST(ConcurrentHashSetRcuTest, AdvanceAccountsForSplitCopiesWhenAssignmentThrows) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        Set::handle h = set.new_handle();
        int next = 0;
        ASSERT_NO_FATAL_FAILURE(fill_to_512_and_advance(set, h, {PROBE_X1, PROBE_X2}, 200, next));
        size_t live = 312;
        EXPECT_TRUE(set.insert(h, fill_key(next++)));   // the shard's first pop: doubles the table to 512
        ++live;

        ProbeKey::throw_countdown = 2;   // the split's second copy throws
        EXPECT_THROW(set.contains(h, PROBE_X1), std::runtime_error);
        ProbeKey::throw_countdown = 0;
        const Set::InternalAccounting threw = expect_consistent(set, "after the split threw on its second copy");
        EXPECT_EQ(threw.limbo, 2u) << "the throwing slot and the one-node partial subchain must both be on limbo";
        ASSERT_EQ(threw.retired + threw.bagged + threw.reachable_dead, 0u) << "test precondition: not a limbo-only state";
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "limbo only, before the split finished"));
        EXPECT_TRUE(set.contains(h, PROBE_X1));
        EXPECT_TRUE(set.contains(h, PROBE_X2));
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after the split finished"));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceAccountsForSplitCopiesWhenAssignmentThrows

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
// so the nesting cannot deadlock. The arithmetic (fill_to_512_and_advance()):
//   - hint 512 - 200 = 312 after the fill's reclamation, table 256, doubling
//     threshold 512;
//   - the outer insert(PROBE_K) reads bucket 5's head, pops a node: the shard's
//     first pop adds 256 to the hint (568), then assigns: the hook runs;
//   - the nested insert(fill_key(517)), low byte 5, so also bucket 5, pops a
//     second node, publishes it (bucket 5's head changes) and, seeing 568 > 512,
//     doubles the table to 512; the hook then arms the next assignment to throw;
//   - the outer CAS fails (stale head); the retry loads size 512, where PROBE_K
//     (bit 8 set) maps to bucket 261, UNINITIALIZED; split_bucket(261) copies
//     PROBE_X1, whose popped slot's assignment throws.
// Afterwards limbo must hold exactly two nodes: the throwing slot (alloc_node())
// and the outer insert's pending node (insert()'s catch); the split's subchain
// was still empty. PROBE_K must be absent, the nested key present, and advance()
// must recycle both with the exact key count (a limbo-only state again, with
// PROBE_X1 FROZEN in bucket 5 behind the unpublished 261, kept by the walk);
// PROBE_K then inserts normally.
TEST(ConcurrentHashSetRcuTest, AdvanceRecoversPendingInsertNodeWhenRetryThrows) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    const int PROBE_K = 5 | 256 | 8192;   // bucket 5 at size 256, 261 at 512; not a fill or probe key
    const int NESTED = fill_key(517);     // bucket 5 at every size up to 1024; beyond the fill keys used
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        Set::handle h = set.new_handle();
        int next = 0;   // not used: the keys below are chosen by hand
        ASSERT_NO_FATAL_FAILURE(fill_to_512_and_advance(set, h, {PROBE_X1}, 200, next));
        size_t live = 312;

        bool nested_inserted = false;
        ProbeKey::on_assign = [&set, &h, &nested_inserted, NESTED]() {
            nested_inserted = set.insert(h, NESTED);
            ProbeKey::throw_countdown = 1;   // the next assignment (the retry's split copy) throws
        };
        EXPECT_THROW(set.insert(h, PROBE_K), std::runtime_error);
        ProbeKey::on_assign = nullptr;
        ProbeKey::throw_countdown = 0;
        ASSERT_TRUE(nested_inserted) << "test precondition: the nested insert did not run inside the outer one's window";
        ++live;
        expect_consistent(set, "after the retry threw");
        EXPECT_EQ(set.get_internal_limbo_count(), 2u) << "the throwing slot and the insert's pending node must both be on limbo";
        EXPECT_FALSE(set.contains(h, PROBE_K)) << "the insert that threw before publishing made its key a member";
        EXPECT_TRUE(set.contains(h, NESTED));
        EXPECT_TRUE(set.contains(h, PROBE_X1));
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after the retry threw"));
        EXPECT_TRUE(set.insert(h, PROBE_K));
        ++live;
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, "after PROBE_K was inserted"));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceRecoversPendingInsertNodeWhenRetryThrows

// Limbo gets fed under contention, and advance() bags and frees it.
//
// Limbo is where the two kinds of never-published node go: the node an insert()
// prepared for a key another thread published first (an orphan), and the
// subchain of a split that lost its publishing CAS. Both need real races: 8
// threads insert the same 64 scrambled keys into a fresh 4-bucket set, so they
// collide on keys and on the splits of the doublings they drive. Two arena shards
// for eight threads, so four threads share each limbo head and its Treiber push
// is contended too. one_winner_per_key_body() checks, per repetition: exactly one
// successful insert() per key; the accounting sweep right after the race (every
// orphan and lost subchain accounted for on a limbo list); advance() with its
// postconditions (limbo empty afterwards, exact key count, one node per key)
// while splits are still pending; and membership after it.
// The limbo total must be positive: the test is only meaningful if the races it
// exists for happened. The assertion is on the total, since nothing forces any
// single race.
TEST(ConcurrentHashSetRcuTest, AdvanceDrainsContendedLimbo) {
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
} // AdvanceDrainsContendedLimbo

// Limbo pushes racing free-list POPS on ONE shard.
//
// Weak spot: the two lists of a shard are written concurrently by different
// protocols. Pops take the free head by CAS, and the only pushes onto it are the
// free pass's, which the handle invariant keeps from restoring an address a
// live popper has read (the header's ABA argument); limbo pushes go to a
// separate head, by CAS, and reach a free list only through a bag. A limbo push
// that went to the free list instead would race the pops: lost pushes (a slot
// leaks) or a slot popped twice (two keys in one node).
// AdvanceDrainsContendedLimbo cannot see that: its sets are fresh, so
// their free lists are empty for the whole race and nothing is popped while
// anything is pushed, and it spreads the threads over two shards.
// Shape: ONE shard for 8 threads, and a prefill (see one_winner_per_key_body())
// so the free list holds a few hundred slots when the race starts; the 8 threads
// then insert the same 64 scrambled keys, so their allocations pop that one list
// while their orphans and lost split subchains are pushed to that one shard's
// limbo. Oracle, per repetition, from one_winner_per_key_body(): exactly one
// winning insert() per key, a consistent accounting straight after the race (a
// lost push or a double pop breaks it), advance() exact with limbo freed,
// membership exact. The limbo total must be positive, or the pushes this test
// exists for never happened. (No advance() runs during the race here; the
// races of free-pass pushes against pops are the stress tests' further down.)
// Prefill size, and the free-list floor asserted before each race. With
// AllowDelete == true every prefill key is erased, so the free list holds at
// least PREFILL slots by construction (more, with the superseded split copies).
// With AllowDelete == false the free list holds ONLY the parent copies the
// prefill's splits superseded (retired by those splits, or by advance()'s walk),
// which exist only for keys that moved at a doubling after they were inserted:
// a one-shard set jumps to 128 buckets within its first five inserts (the first
// append counts 256), so those early doublings find at most five keys in place
// and move few. 600 keys cross the doublings at 256 and 512 slots with hundreds
// of keys already in place, and the settling sweep turns those splits into a
// few hundred superseded copies. The floor for that flavor, PREFILL/4, is
// therefore an EMPIRICAL precondition, not a derived one (the prefill is
// single-threaded, so the count is the same every run).
TEST(ConcurrentHashSetRcuTest, AdvanceLimboPushesRaceFreeListPops) {
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
} // AdvanceLimboPushesRaceFreeListPops

// ACCEPTANCE: the purpose of reclamation. A workload whose live key count is
// constant (every round, each thread erases all its keys and inserts as many new
// ones) must run in bounded memory when every round ends with every handle
// released and an advance(), and the arena must grow without bound when none
// does.
// Oracles:
//   - booleans: each thread's keys are its own and new each round, so every
//     erase() and every insert() must return true;
//   - with advance(): the exact live count (T*W) at every reclamation, and the
//     arena (slots, i.e. get_internal_node_count(), which the accounting also
//     checks) after every round at most BOUND times that live count. BOUND = 4 is
//     derived, not fitted: a round allocates W nodes per thread plus split copies,
//     and after the first reclamation the free lists hold at least what the
//     previous round allocated, dealt evenly; so the arena stabilizes after round 1
//     at about (fill + round-1 allocations + split copies), i.e. about 3 times the
//     live count. An advance() that recycled nothing would add at least one live
//     count per round and cross the bound by round 3;
//   - without advance(): every successful insert() allocates a node that is never
//     reused, so the arena holds at least (R + 1)*T*W slots, which is more than
//     BOUND*T*W: this half proves the bound above is not vacuous.
// One arena shard per thread: T threads started together get T consecutive thread
// numbers, one per shard, whatever round they run in; with fewer or more shards
// than threads the bound would hold as well (a thread whose own list runs dry
// takes nodes from other shards' lists before it appends), but each thread's
// allocations would then mix its own and other shards' nodes.
TEST(ConcurrentHashSetRcuTest, AdvanceBoundsTheArena) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int T = 8, W = 256, R = 8;
    const size_t L = size_t(T)*W, BOUND = 4;
    for (const bool with_advance : {true, false}) {
        const std::string flavor = with_advance ? "with advance()" : "without advance()";
        Set set(4, T);
        Set::handle h = set.new_handle();
        // Key of thread t, round r, index i: distinct for every (t, r, i).
        auto key = [](int t, int r, int i) { return scramble((r*T + t)*W + i); };
        std::vector<size_t> bad(T, 0);
        run_threads(T, set, [&set, &bad, &key](const Set::handle& h, int t) {
            for (int i = 0; i < W; ++i) bad[t] += !set.insert(h, key(t, 0, i));
        });
        for (int r = 0; r <= R; ++r) {   // round 0 is the fill above; rounds 1..R churn
            const std::string where = flavor + ", round " + std::to_string(r);
            if (r > 0) {
                run_threads(T, set, [&set, &bad, &key, r](const Set::handle& h, int t) {
                    for (int i = 0; i < W; ++i) {
                        bad[t] += !set.erase(h, key(t, r - 1, i));
                        bad[t] += !set.insert(h, key(t, r, i));
                    }
                });
            } // churn
            for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ", thread " << t << ": a determined result was wrong";
            if (with_advance) {
                ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, L, where));
                EXPECT_LE(set.get_internal_node_count(), BOUND*L) << where << ": the arena outgrew " << BOUND << " times the live count";
            } else {
                expect_consistent(set, where);
            } // with or without advance()
        } // rounds
        if (!with_advance) {
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
        EXPECT_EQ(membership_errors(set, h, in, out), 0) << flavor;
    } // with and without advance()
} // AdvanceBoundsTheArena

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

// Inserts `keys` in order into `set` (each must be new), under the calling
// thread's handle `h`, then checks that the table has `expected_ts` buckets: the
// geometry the calling test derived. Callers wrap it in ASSERT_NO_FATAL_FAILURE.
template <typename SetT>
static void insert_in_order(SetT& set, const typename SetT::handle& h, const std::vector<int>& keys, size_t expected_ts) {
    for (int k : keys) ASSERT_TRUE(set.insert(h, k)) << "insert(" << k << ") of a new key returned false";
    ASSERT_EQ(set.get_internal_table_size(), expected_ts) << "test precondition: the set-up did not reach the derived table size";
} // insert_in_order()

// An uncontended erase() unlinks and retires its own node, every time.
//
// Kills: erase() sending its unlinked node to a LIMBO list (or anywhere but a
// retired list): retired and limbo counts then disagree with the erases.
// 8 chains of 16 colliding keys (b + 1024*i, bucket b of a 1024-bucket table,
// which never doubles: 128 nodes are one append batch, hint 256, threshold 2048)
// are erased OLDEST first, so each erase finds its node at the tail of its chain
// behind the live nodes inserted after it, and no earlier dead node lies between
// its predecessor and it (every earlier erase already took its own node out);
// the last erase of each chain finds its node at the head. Each erase unlinks
// exactly its node, so with no advance() at all every slot ends up retired.
// (The erased node is the tail, so a run walk that overshoots its run has
// nothing to overshoot into: GrowthLeavesNoDeadNodeReachable catches that.)
TEST(ConcurrentHashSetRcuTest, EraseAllUnlinksEveryNode) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CollisionHash>;
    const int B = 8, PER = 16, C = 1024;
    Set set(C, 1);
    const Set::handle h = set.new_handle();
    for (int i = 0; i < PER; ++i) {
        for (int b = 0; b < B; ++b) EXPECT_TRUE(set.insert(h, b + C*i));
    }
    ASSERT_EQ(set.get_internal_table_size(), size_t(C)) << "test precondition: the table doubled";
    const Set::InternalCounters before = set.get_internal_counters();
    for (int i = 0; i < PER; ++i) {   // oldest first: each target is its chain's tail
        for (int b = 0; b < B; ++b) {
            EXPECT_TRUE(set.erase(h, b + C*i));
            EXPECT_FALSE(set.contains(h, b + C*i));
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

// Uncontended growth leaves no dead node reachable.
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
// with no advance(): exactly one node per key is reachable and none of them dead,
// nothing is on a limbo or free list, every other slot is retired, and no
// cleanup CAS lost. Both AllowDelete values: the freeze runs for both.
template <bool AllowDelete>
static void growth_leaves_no_dead_node_body() {
    using Set = ConcurrentResizableHashSetRCU<int, AllowDelete>;
    const int N = 5000;
    const std::string where = AllowDelete ? "AllowDelete=true" : "AllowDelete=false";
    Set set(4, 1);
    const typename Set::handle h = set.new_handle();
    std::vector<int> keys;
    for (int i = 0; i < N; ++i) {
        keys.push_back(scramble(i));
        EXPECT_TRUE(set.insert(h, keys.back())) << where;
    }
    EXPECT_EQ(membership_errors(set, h, keys, std::vector<int>{}), 0) << where;
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

// erase()'s mark CAS fails on a LIVE value, and erase() must retry it.
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {S, K}, 16));
        bool hook_ran = false, nested_found = false;
        size_t retired_in_window = 0;
        ProbeKey::arm(ProbeKey::equal_hook, K, 1, [&]() {
            hook_ran = true;
            nested_found = set.contains(h, S);
            retired_in_window = set.get_internal_accounting().retired;
        });
        const bool erased = set.erase(h, K);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on erase(1)'s comparison did not fire";
        ASSERT_TRUE(nested_found);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: the split's cleanup did not unlink 9's parent copy through 1's link";
        EXPECT_TRUE(erased) << "erase() of a present key returned false after a live change of its link";
        EXPECT_FALSE(set.contains(h, K));
        EXPECT_TRUE(set.contains(h, S));
        const Set::InternalAccounting acc = expect_consistent(set, "after erase(1)");
        EXPECT_EQ(acc.retired, 2u) << "9's parent copy and 1's node must both be retired";
        EXPECT_EQ(acc.reachable, 1u);
        EXPECT_EQ(acc.reachable_dead, 0u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // EraseRetriesMarkAfterLiveLinkChange

// A miss in a chain from which the key's FROZEN copy was unlinked must reload
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {K, 65, W}, 32));
        bool hook_ran = false, doubled = false, nested_found = false;
        size_t retired_in_window = 0;
        std::function<void()> window = [&]() {
            hook_ran = true;
            doubled = set.insert(h, 2) && set.get_internal_table_size() == 64;
            nested_found = set.contains(h, K);
            retired_in_window = set.get_internal_accounting().retired;
        };
        if (at_compare) {
            ProbeKey::arm(ProbeKey::equal_hook, W, 1, window);
        } else {
            ProbeKey::arm(ProbeKey::hash_hook, K, 1, window);
        }
        ProbeKey::tally_value = K;
        ProbeKey::tally = 0;
        const bool result = erase ? set.erase(h, K) : set.contains(h, K);
        const int outer_hashes = ProbeKey::tally;
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook did not fire";
        ASSERT_TRUE(doubled) << "test precondition: insert(2) did not double the table to 64";
        ASSERT_TRUE(nested_found);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: the split's cleanup did not unlink 33's parent copy";
        EXPECT_TRUE(result) << (erase ? "erase" : "contains") << "(33) missed a present key whose FROZEN copy was unlinked";
        EXPECT_EQ(outer_hashes, 2) << "the outer operation must hash 33 once per pass: once in bucket 1, once after the reload";
        EXPECT_EQ(set.contains(h, K), !erase);
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {STAYER, VICTIM, MOVER, 2}, 64));
        bool hook_ran = false, nested_erased = false;
        size_t retired_in_window = 0;
        ProbeKey::arm(ProbeKey::hash_hook, MOVER, 2, [&]() {
            hook_ran = true;
            nested_erased = set.erase(h, VICTIM);
            retired_in_window = set.get_internal_accounting().retired;
        });
        const bool found = set.contains(h, MOVER);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on the split's Hash{}(33) did not fire";
        ASSERT_TRUE(nested_erased);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: erase(65) did not unlink its node through 33's link";
        EXPECT_TRUE(found);
        const Set::InternalAccounting acc = expect_consistent(set, "after the split");
        EXPECT_EQ(acc.reachable, 3u) << "a key has two reachable nodes: the split copied a node it did not freeze";
        EXPECT_EQ(acc.reachable_dead, 0u);
        EXPECT_EQ(acc.retired, 2u);
        EXPECT_TRUE(set.contains(h, MOVER));
        EXPECT_TRUE(set.contains(h, STAYER));
        EXPECT_FALSE(set.contains(h, VICTIM));
        EXPECT_FALSE(set.erase(h, VICTIM));
        EXPECT_TRUE(set.erase(h, MOVER));
        EXPECT_FALSE(set.contains(h, MOVER)) << "erase(33) returned true and the key is still present";
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {1, MOVER}, 16));
        bool hook_ran = false, doubled = false, nested_found = false;
        size_t retired_in_window = 0;
        ProbeKey::arm(ProbeKey::hash_hook, STALE, 1, [&]() {
            hook_ran = true;
            doubled = set.insert(h, 2) && set.get_internal_table_size() == 32;
            nested_found = set.contains(h, MOVER);
            retired_in_window = set.get_internal_accounting().retired;
        });
        ProbeKey::tally_value = STALE;
        ProbeKey::tally = 0;
        const bool inserted = set.insert(h, STALE);
        const int outer_hashes = ProbeKey::tally;
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on insert(49)'s Hash{} did not fire";
        ASSERT_TRUE(doubled) << "test precondition: insert(2) did not double the table to 32";
        ASSERT_TRUE(nested_found);
        ASSERT_EQ(retired_in_window, 1u) << "test precondition: the split's cleanup did not unlink 17's parent copy at the head";
        EXPECT_TRUE(inserted);
        EXPECT_EQ(outer_hashes, 2) << "insert(49) did not retry after meeting the raised seal level";
        EXPECT_TRUE(set.contains(h, STALE)) << "the stale insert published its key into the sealed parent: the key is lost";
        EXPECT_FALSE(set.insert(h, STALE)) << "a second insert of the key succeeded";
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {K, X, W}, 1024));
        const Set::InternalAccounting empty = expect_consistent(set, "before");
        ASSERT_EQ(empty.limbo, 0u);
        ASSERT_EQ(empty.retired, 0u);
        bool hook_ran = false, nested_erased = false;
        size_t reachable_in_window = 0;
        ProbeKey::arm(ProbeKey::equal_hook, W, 1, [&]() {
            hook_ran = true;
            nested_erased = set.erase(h, X);
            reachable_in_window = set.get_internal_accounting().reachable;
        });
        const bool found = set.contains(h, K);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(hook_ran) << "test precondition: the hook on contains(5)'s first comparison did not fire";
        ASSERT_TRUE(nested_erased);
        ASSERT_EQ(reachable_in_window, 2u) << "test precondition: erase(1029) did not unlink its node";
        EXPECT_TRUE(found) << "a walk that reached a node before its unlink could not walk out of it";
        EXPECT_FALSE(set.contains(h, X));
        EXPECT_TRUE(set.contains(h, W));
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
// `key` is the hooked key (33 here). The hooks count Hash{}(key) calls only, so
// the counts do not depend on where key's node sits in the chain: contains(key)'s
// first call is H1; its second is the split walk's hash of key's node (H3), which
// needs that node to be one the split moves; its third is the cleanup's hash of
// the FROZEN parent copy (H6), wherever the cleanup makes it -- in the deadness
// test of the node it reaches, or in the run walk of a run that starts earlier --,
// always after the publication and before that run's CAS, which needs the
// cleanup to reach the copy with a usable predecessor (it hashes nothing it
// passes otherwise). The callers below also have key's node as the first node
// the split moves and the first tagged node the cleanup meets, which is
// sufficient for both conditions, and puts the action before any of the
// cleanup's CASes.
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {1, 33, 65, 2}, 64));
        const Set::InternalCounters before = set.get_internal_counters();
        CleanupWindow w;
        bool nested_inserted = false;
        size_t retired_in_window = 0;
        arm_cleanup_window(set, w, 33, [&set, &h, &nested_inserted, &retired_in_window]() {
            nested_inserted = set.insert(h, 129);
            retired_in_window = set.get_internal_accounting().retired;
        });
        const bool found = set.contains(h, 33);
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
        for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(h, k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CleanupLosingItsCasRetiresNothing

// A Hash{} that throws in the split's cleanup leaves a superseded parent copy in
// its chain (a straggler), and the next writer walk past it collects it.
//
// Kills: a writer walk that does not unlink a dead run behind a live predecessor
// it passes (the straggler would stay until an advance() walked past it).
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {1, 33, 65, 2}, 64));
        CleanupWindow w;
        arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
        EXPECT_THROW(set.contains(h, 33), std::runtime_error);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(w.before_publish) << "test precondition: the hook on the split's Hash{}(33) did not fire";
        ASSERT_TRUE(w.after_publish) << "test precondition: the hook on the cleanup's Hash{}(33) did not fire right after the publication";
        const Set::InternalAccounting left = expect_consistent(set, "after the cleanup threw");
        ASSERT_EQ(left.reachable_dead, 1u) << "test precondition: the throw did not leave 33's parent copy in its chain";
        EXPECT_TRUE(set.insert(h, 129));
        const Set::InternalAccounting acc = expect_consistent(set, "after insert(129)");
        EXPECT_EQ(acc.reachable_dead, 0u) << "insert()'s walk passed a dead node behind a live predecessor and left it";
        EXPECT_EQ(acc.retired, left.retired + 1);
        for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(h, k)) << k;
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
            const Set::handle h = set.new_handle();
            ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {1, 33, 65, 2}, 64));
            CleanupWindow w;
            arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
            EXPECT_THROW(set.contains(h, 33), std::runtime_error);
            ProbeKey::disarm_hooks();
            ASSERT_TRUE(w.before_publish && w.after_publish) << where << ": test precondition: the cleanup window did not open";
            const Set::InternalAccounting left = expect_consistent(set, where + ", after the cleanup threw");
            ASSERT_EQ(left.reachable_dead, 1u) << where << ": test precondition: no straggler behind 65";
            if (hit) {
                EXPECT_TRUE(set.erase(h, 1));
            } else {
                EXPECT_FALSE(set.erase(h, 129));
            }
            const Set::InternalAccounting acc = expect_consistent(set, where + ", after the erase");
            EXPECT_EQ(acc.reachable_dead, 0u) << where << ": erase()'s walk passed a dead node behind a live predecessor and left it";
            EXPECT_EQ(acc.retired, left.retired + (hit ? 2u : 1u)) << where;
            EXPECT_TRUE(set.contains(h, 33)) << where;
            EXPECT_TRUE(set.contains(h, 65)) << where;
            EXPECT_EQ(set.contains(h, 1), !hit) << where;
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
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {1, 65, 33, 2}, 64));
        CleanupWindow w;
        arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
        EXPECT_THROW(set.contains(h, 33), std::runtime_error);
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(w.before_publish && w.after_publish) << "test precondition: the cleanup window did not open";
        const Set::InternalAccounting left = expect_consistent(set, "after the cleanup threw");
        ASSERT_EQ(left.reachable_dead, 1u) << "test precondition: no dead node at the head of bucket 1";
        ProbeKey::tally_value = 129;
        ProbeKey::tally = 0;
        EXPECT_TRUE(set.insert(h, 129));
        const int outer_hashes = ProbeKey::tally;
        ProbeKey::disarm_hooks();
        EXPECT_EQ(outer_hashes, 1) << "the publish after the insert's own head unlink did not expect the head it installed";
        const Set::InternalAccounting acc = expect_consistent(set, "after insert(129)");
        EXPECT_EQ(acc.reachable_dead, 0u);
        EXPECT_EQ(acc.retired, left.retired + 1);
        for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(h, k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // InsertPublishAfterOwnHeadUnlink

// A lost unlink CAS is classified by the word's value when it lost: a PEER LOSS
// when the predecessor word already holds the successor the loser computed (a
// peer bypassed the same run, so nothing is left behind), any other loss
// otherwise (a prepend, a seal, a tag of the predecessor, or a bypass of a run of
// another extent: possibly a straggler). The diagnostic counters
// (get_internal_counters()) keep both, for the split winner's cleanup and for
// insert()'s and erase()'s walks.
//
// Kills: the classification inverted (a peer loss counted as a possible
// straggler and the other way round), at either site.
// Four deterministic cases, each on the 64-bucket geometry of the split-cleanup
// tests (insert(1), insert(33), insert(65), insert(2) or the same with 33 and 65
// swapped; 33 moves to bucket 33, 129 and 193 are new keys of bucket 1), and
// each with a nested insert() whose walk unlinks 33's dead parent copy in the
// window between the outer unlink's load of its predecessor word and its CAS:
//   - cleanup, peer: chain 65 -> 33 -> 1; contains(33)'s cleanup, hooked at its
//     Hash{}(33) (arm_cleanup_window()), is about to CAS 65's link from 33 to 1;
//     the nested insert(129) unlinks 33 through that link, installing 1, then
//     prepends at the head: the cleanup's CAS loses to a word holding 1;
//   - cleanup, prepend: chain 33 -> 65 -> 1; the cleanup is about to CAS the
//     head from 33 to 65; the nested insert(129) unlinks 33 through the head and
//     then prepends 129 there: the cleanup's CAS loses to a word holding 129;
//   - walk, peer: chain 65 -> 33 -> 1, with 33's parent copy left behind 65 by a
//     cleanup that threw; insert(129)'s walk, hooked at its first Hash{}(33) (the
//     deadness test of the tagged node it reaches with a usable predecessor:
//     insert(129) hashes 129 before its walk, and 65 is live), is about to CAS
//     65's link; the nested insert(193) unlinks 33 through it, installing 1;
//   - walk, prepend: chain 33 -> 65 -> 1, the dead copy at the head; the nested
//     insert(193) unlinks it and prepends 193.
// No case loses to a pure prepend, and none can: any writer that prepends to
// bucket 1 in the window walks past the dead copy first and unlinks it. In the
// prepend cases the run is therefore gone all the same, but the word holds the
// prepended node, not the loser's successor, so the loss counts as a possible
// straggler: the classification errs on that side.
// Each case: the hook ran (the nested insert returned true), exactly one loss at
// the hooked site and none at the other, a peer loss exactly in the peer cases,
// one node retired (by the nested insert, never by the loser), and nothing dead
// left reachable.
TEST(ConcurrentHashSetRcuTest, UnlinkLossClassification) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    struct Case {
        const char* name;
        bool cleanup;   // the hooked unlink: the split's cleanup (else insert()'s walk)
        bool peer;      // 33's predecessor is 65's link (a peer loss; else the head: a prepend)
    }; // struct Case
    for (const Case c : {Case{"cleanup, peer", true, true}, Case{"cleanup, prepend", true, false},
                         Case{"walk, peer", false, true}, Case{"walk, prepend", false, false}}) {
        const std::string where = c.name;
        ProbeKey::reset_counters();
        {
            Set set(4, 1);
            const Set::handle h = set.new_handle();
            ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, c.peer ? std::vector<int>{1, 33, 65, 2} : std::vector<int>{1, 65, 33, 2}, 64));
            bool nested_inserted = false;
            Set::InternalCounters before = set.get_internal_counters();
            size_t retired_before = set.get_internal_accounting().retired;
            // contains(33) splits bucket 33. In the cleanup cases its cleanup is the
            // hooked unlink and the nested insert runs in its window; in the walk
            // cases the cleanup throws there instead, leaving 33's parent copy in
            // bucket 1, dead, for insert(129)'s walk below.
            const std::function<void()> in_cleanup = c.cleanup
                ? std::function<void()>([&set, &h, &nested_inserted]() { nested_inserted = set.insert(h, 129); })
                : std::function<void()>([]() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
            CleanupWindow w;
            arm_cleanup_window(set, w, 33, in_cleanup);
            if (c.cleanup) {
                EXPECT_TRUE(set.contains(h, 33)) << where;
            } else {
                EXPECT_THROW(set.contains(h, 33), std::runtime_error) << where;
            }
            ProbeKey::disarm_hooks();
            ASSERT_TRUE(w.before_publish && w.after_publish) << where << ": test precondition: the cleanup window did not open";
            if (!c.cleanup) {   // the hooked unlink is insert(129)'s walk
                ASSERT_EQ(set.get_internal_accounting().reachable_dead, 1u) << where << ": test precondition: no straggler in bucket 1";
                before = set.get_internal_counters();
                retired_before = set.get_internal_accounting().retired;
                ProbeKey::arm(ProbeKey::hash_hook, 33, 1, [&set, &h, &nested_inserted]() { nested_inserted = set.insert(h, 193); });
                EXPECT_TRUE(set.insert(h, 129)) << where;
                ProbeKey::disarm_hooks();
            } // if the walk is the hooked unlink
            ASSERT_TRUE(nested_inserted) << where << ": test precondition: the nested insert did not run in the unlink's window";
            const Set::InternalCounters after = set.get_internal_counters();
            EXPECT_EQ(after.cleanup_cas_failures - before.cleanup_cas_failures, c.cleanup ? 1u : 0u) << where;
            EXPECT_EQ(after.cleanup_cas_peer_losses - before.cleanup_cas_peer_losses, (c.cleanup && c.peer) ? 1u : 0u) << where;
            EXPECT_EQ(after.walk_cas_failures - before.walk_cas_failures, c.cleanup ? 0u : 1u) << where;
            EXPECT_EQ(after.walk_cas_peer_losses - before.walk_cas_peer_losses, (!c.cleanup && c.peer) ? 1u : 0u) << where;
            const Set::InternalAccounting acc = expect_consistent(set, where + ", after the lost CAS");
            EXPECT_EQ(acc.retired, retired_before + 1) << where << ": 33's parent copy must be retired once, by the nested insert";
            EXPECT_EQ(acc.reachable_dead, 0u) << where;
            for (int k : {1, 2, 33, 65, 129}) EXPECT_TRUE(set.contains(h, k)) << where << ", key " << k;
            if (!c.cleanup) {
                EXPECT_TRUE(set.contains(h, 193)) << where;
            }
        } // the set's lifetime
        EXPECT_EQ(ProbeKey::corrupt.load(), 0) << where;
    } // loop over the cases
} // UnlinkLossClassification

// Stragglers, single-threaded, for the tests below: three splits whose cleanup
// throws at its first Hash{} (arm_cleanup_window()), each leaving every parent
// copy it superseded in the parent chain, dead. Geometry, identity hash, one
// shard, on a fresh Set(4, 1) over ProbeKey:
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
// Five dead nodes are then reachable, in FOUR runs: a two-node run at the head
// of bucket 1, a one-node run behind a live node in bucket 2, two one-node runs
// in bucket 3; and buckets 5, 6 and 7, pending children of the same three
// parents, are UNINITIALIZED (both ASSERTed). `h`: the calling thread's handle of
// `set`. Returns the sweep of that state through `left`. Callers wrap it in
// ASSERT_NO_FATAL_FAILURE.
template <typename SetT>
static void leave_five_stragglers(SetT& set, const typename SetT::handle& h, typename SetT::InternalAccounting& left) {
    struct Split { int bucket; int key; };   // a parent bucket, and the hooked mover whose contains() splits its child
    ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {0, 256, 512, 768, 1024,   // to 128 buckets
                                                  257, 129, 385, 258, 130, 514, 131, 259, 387, 515}, 128));
    for (int i = 0; set.get_internal_table_size() == 128; ++i) {   // fillers, until the doubling to 256
        ASSERT_LT(i, 300) << "test precondition: the fillers did not double the table";
        ASSERT_TRUE(set.insert(h, 256*(64 + i)));
    }
    ASSERT_EQ(set.get_internal_table_size(), 256u);
    for (const Split split : {Split{1, 385}, Split{2, 130}, Split{3, 387}}) {
        CleanupWindow w;
        arm_cleanup_window(set, w, split.key, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
        EXPECT_THROW(set.contains(h, split.key), std::runtime_error) << "bucket " << split.bucket;
        ProbeKey::disarm_hooks();
        ASSERT_TRUE(w.before_publish && w.after_publish) << "test precondition: the cleanup window of bucket " << split.bucket << "'s split did not open";
    } // leave the stragglers
    left = expect_consistent(set, "after the cleanups threw");
    ASSERT_EQ(left.reachable_dead, 5u) << "test precondition: the throwing cleanups did not leave five stragglers";
    for (size_t j : {5, 6, 7}) ASSERT_FALSE(set.get_internal_bucket_published(j)) << "test precondition: bucket " << j << " is published";
} // leave_five_stragglers()

// settle_writers() collects every straggler a history left, through writer walks
// alone, and splits nothing.
//
// Kills: insert()'s and erase()'s walks making no opportunistic unlink, and
// erase()'s walk alone making none (settle_writers() collects through erase()
// miss walks). It is the deterministic test behind the oracles that the
// concurrent tests check after settle_writers(), which under contention seldom
// find a straggler to settle.
// On leave_five_stragglers()'s state, settle_writers() must leave no dead node
// reachable, retire exactly those five in four runs, and leave buckets 5, 6 and
// 7 pending: had it split them, their cleanups would have walked the parents and
// collected the stragglers whatever the writer walks did.
TEST(ConcurrentHashSetRcuTest, SettleWritersCollectsStragglers) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        const Set::handle h = set.new_handle();
        Set::InternalAccounting left{};
        ASSERT_NO_FATAL_FAILURE(leave_five_stragglers(set, h, left));
        const Set::InternalCounters before = set.get_internal_counters();

        ASSERT_NO_FATAL_FAILURE(settle_writers(set, h));
        const Set::InternalAccounting settled = expect_consistent(set, "after settle_writers()");
        EXPECT_EQ(settled.reachable_dead, 0u) << "a straggler survived a writer miss walk over its chain";
        EXPECT_EQ(settled.reachable, left.reachable - 5);
        EXPECT_EQ(settled.retired, left.retired + 5);
        const Set::InternalCounters after = set.get_internal_counters();
        EXPECT_EQ(after.retire_runs - before.retire_runs, 4u) << "one run at the head of bucket 1, one in bucket 2, two in bucket 3";
        EXPECT_EQ(after.split_attempts, before.split_attempts) << "settle_writers() split a bucket";
        for (size_t j : {5, 6, 7}) EXPECT_FALSE(set.get_internal_bucket_published(j)) << "settle_writers() published bucket " << j;
        for (int k : {257, 129, 385, 258, 130, 514, 131, 259, 387, 515}) EXPECT_TRUE(set.contains(h, k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // SettleWritersCollectsStragglers

// advance()'s walk collects every straggler run it passes behind a bucket head
// or a live node -- a two-node run at a head, runs behind live nodes, two runs in
// one chain -- with one CAS per run, splits nothing, and counts what it did.
//
// Kills: a walk that unlinks only the first node of a run, or only the first
// run of a chain, or none behind a head (walk_retired, retire_runs and the dead
// count after it); a walk that splits a pending bucket (5, 6, 7 published); a
// walk whose counts skip the nodes inside a bypassed run (nodes_passed: every
// reachable node is visited, live and dead alike, a run's inner nodes
// included). On leave_five_stragglers()'s state, every handle released: call k
// retires the five in four runs (nothing_retired: no list held anything) and
// call k + 1 frees them (advance_and_check()), with no lost CAS.
TEST(ConcurrentHashSetRcuTest, AdvanceWalkCollectsEveryStragglerRun) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(4, 1);
        Set::handle h = set.new_handle();
        Set::InternalAccounting left{};
        ASSERT_NO_FATAL_FAILURE(leave_five_stragglers(set, h, left));
        const Set::InternalCounters before = set.get_internal_counters();
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, left.reachable - left.reachable_dead, "the five stragglers"));
        const Set::InternalCounters after = set.get_internal_counters();
        EXPECT_EQ(after.retire_runs - before.retire_runs, 4u) << "one run at the head of bucket 1, one in bucket 2, two in bucket 3";
        EXPECT_EQ(after.advance_cas_failures, before.advance_cas_failures) << "a walk CAS lost with no other thread running";
        EXPECT_EQ(after.split_attempts, before.split_attempts) << "advance()'s walk split a bucket";
        for (size_t j : {5, 6, 7}) EXPECT_FALSE(set.get_internal_bucket_published(j)) << "advance()'s walk published bucket " << j;
        for (int k : {257, 129, 385, 258, 130, 514, 131, 259, 387, 515}) EXPECT_TRUE(set.contains(h, k)) << k;
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceWalkCollectsEveryStragglerRun

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
// Kills: no specific protocol step. A GUARD on the accounting under the
// contention that makes the splits' one-shot cleanup CASes lose, and a print of
// what it left.
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
    const Set::handle h = set.new_handle();
    std::vector<size_t> bad(T, 0);
    run_threads(T, set, [&set, &bad](const Set::handle& h, int t) {
        for (int i = t; i < kHotKeys; i += T) bad[t] += !set.insert(h, hot_growth_key(i));
    });
    for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << "insert() of a new, uncontended key returned false, thread " << t;
    const Set::InternalAccounting joined = expect_consistent(set, "after the join");
    EXPECT_EQ(joined.reachable - joined.reachable_dead, size_t(kHotKeys)) << "not exactly one non-dead reachable node per key";
    const Set::InternalCounters c = set.get_internal_counters();
    std::printf("[   INFO   ] HotGrowthResidue: table %zu, splits published %zu of %zu, cleanup CAS lost %zu, "
                "retired %zu nodes in %zu runs, reachable dead before settling %zu, limbo %zu, slots %zu\n",
                set.get_internal_table_size(), c.splits_published, c.split_attempts, c.cleanup_cas_failures,
                joined.retired, c.retire_runs, joined.reachable_dead, joined.limbo, joined.slots);
    ASSERT_NO_FATAL_FAILURE(settle_writers(set, h));
    const Set::InternalAccounting settled = expect_consistent(set, "after settle_writers()");
    EXPECT_EQ(settled.reachable, size_t(kHotKeys));
    EXPECT_EQ(settled.reachable_dead, 0u) << "a dead node survived a writer walk over its chain";
    EXPECT_EQ(settled.free_nodes, 0u);
    EXPECT_EQ(settled.retired + settled.limbo, settled.slots - kHotKeys);
    int missing = 0;
    for (int i = 0; i < kHotKeys; ++i) missing += !set.contains(h, hot_growth_key(i));
    EXPECT_EQ(missing, 0) << "keys lost";
} // HotGrowthResidue

// Churn stress with the full accounting after every join.
//
// Kills: no specific protocol step. A stress of the unlinking protocol under
// contention -- erases unlinking their own nodes, walks unlinking what they pass,
// splits' cleanups, all on the same chains -- whose oracles are exact at every
// quiescent point: the accounting is consistent (every slot in one of the five
// places of contract clause 4 once, every reachable dead node and every retired
// node tagged), there is
// exactly one non-dead reachable node per key, and the per-key insert/erase tally
// fixes membership (InsertEraseChurnPerKeyAccounting's oracle, cumulative over
// rounds).
// Each repetition is a fresh set churned for ROUNDS rounds, joined after each:
//   - kWithAdvance: then every handle released and advance() (and a second
//     call if its walk retired something), with the full postconditions of
//     advance_and_check();
//   - kSettle (no advance() at all; identity hash): then settle_writers(), after
//     which no dead node is reachable, one node per key is, nothing is free, and
//     every other slot is retired or on a limbo list. The dead nodes reachable
//     BEFORE settling (the stragglers the churn left) are printed per round,
//     summed and maximized over the repetitions. Under this churn the writers'
//     own walks collect nearly every straggler before the join, so the oracles
//     after settling are GUARDS here; SettleWritersCollectsStragglers is the
//     deterministic test of the settling;
//   - kLongChains: ConstantHash (every key in bucket 0, so splits copy nothing and
//     the one chain holds every key and every dead node), with advance(): erases
//     unlinking through mid-chain predecessors that other erases are marking and
//     unlinking.
// Key k is scramble(k + 1), for k < R; the set has 2*T shards.
enum class ChurnCheck { kWithAdvance, kSettle, kLongChains };

template <typename Hash>
static void churn_accounting_body(ChurnCheck check, int T, int R, int REPS, int ROUNDS, int OPS) {
    using Set = ConcurrentResizableHashSetRCU<int, true, Hash>;
    std::vector<size_t> dead_sum(ROUNDS, 0), dead_max(ROUNDS, 0);   // kSettle: reachable dead before settling
    for (int rep = 0; rep < REPS; ++rep) {
        Set set(4, 2*T);
        typename Set::handle h = set.new_handle();
        std::vector<std::vector<int>> ins(T, std::vector<int>(R, 0));
        std::vector<std::vector<int>> ers(T, std::vector<int>(R, 0));
        for (int round = 0; round < ROUNDS; ++round) {
            const std::string where = "rep " + std::to_string(rep) + ", round " + std::to_string(round);
            run_threads(T, set, [&set, &ins, &ers, rep, round, R, OPS](const typename Set::handle& h, int t) {
                std::minstd_rand rng(static_cast<unsigned>((rep*31 + round)*977 + t + 1));
                for (int op = 0; op < OPS; ++op) {
                    const unsigned x = rng();
                    const int i = static_cast<int>((x >> 4)%static_cast<unsigned>(R));
                    switch (x & 3u) {
                        case 0:
                        case 1:   // insert twice as often as erase, so the set stays populated
                            if (set.insert(h, scramble(i + 1))) ++ins[t][i];
                            break;
                        case 2:
                            if (set.erase(h, scramble(i + 1))) ++ers[t][i];
                            break;
                        default:
                            set.contains(h, scramble(i + 1));
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
                EXPECT_EQ(set.contains(h, scramble(i + 1)), d == 1) << where << ", key index " << i << ": membership disagrees with the tally " << d;
                live += d == 1;
            } // per-key check
            EXPECT_EQ(acc.reachable - acc.reachable_dead, live) << where << ": not exactly one non-dead reachable node per key";
            if (check == ChurnCheck::kSettle) {
                dead_sum[round] += acc.reachable_dead;
                dead_max[round] = std::max(dead_max[round], acc.reachable_dead);
                if constexpr (std::is_same_v<Hash, std::hash<int>>) {
                    ASSERT_NO_FATAL_FAILURE(settle_writers(set, h));
                }
                const typename Set::InternalAccounting settled = expect_consistent(set, where + ", after settle_writers()");
                EXPECT_EQ(settled.reachable_dead, 0u) << where << ": a dead node survived a writer walk over its chain";
                EXPECT_EQ(settled.reachable, live) << where;
                EXPECT_EQ(settled.free_nodes, 0u) << where;
                EXPECT_EQ(settled.retired + settled.limbo, settled.slots - live) << where;
            } else {
                ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, live, where));
            } // settle or advance
        } // rounds
    } // repetitions
    if (check == ChurnCheck::kSettle) {
        for (int round = 0; round < ROUNDS; ++round) {
            std::printf("[   INFO   ] churn without advance(), round %d: reachable dead before settling: %zu over %d sets, at most %zu in one\n",
                        round, dead_sum[round], REPS, dead_max[round]);
        }
    } // print the stragglers
} // churn_accounting_body()

TEST(ConcurrentHashSetRcuTest, ChurnAccountingWithAdvance) {
    ASSERT_NO_FATAL_FAILURE(churn_accounting_body<std::hash<int>>(ChurnCheck::kWithAdvance, 8, 48, 20, 4, 800));
} // ChurnAccountingWithAdvance

TEST(ConcurrentHashSetRcuTest, ChurnAccountingSettled) {
    ASSERT_NO_FATAL_FAILURE(churn_accounting_body<std::hash<int>>(ChurnCheck::kSettle, 8, 48, 20, 4, 800));
} // ChurnAccountingSettled

TEST(ConcurrentHashSetRcuTest, ChurnAccountingLongChains) {
    ASSERT_NO_FATAL_FAILURE(churn_accounting_body<ConstantHash>(ChurnCheck::kLongChains, 4, 32, 100, 3, 200));
} // ChurnAccountingLongChains

// ===========================================================================
// Handles, generations and advance(): reclamation beside running operations.
//
// Every operation runs under a handle, a session with one generation; advance()
// closes the current generation's retired and limbo lists into its BAG,
// publishes the next generation, frees the bags no live handle can observe
// (oldest first, cumulatively), and walks every published chain, retiring the
// dead runs it can bypass into the new generation. Contract clauses 4 and 5 at
// the top of this file; the header's HANDLES AND GENERATIONS section and the
// contract on advance().
//
// The tests below are of three kinds:
//   - deterministic, single-threaded tests of advance()'s results and of what a
//     held handle keeps, with the exact counts the header's A0-A5, free pass and
//     walk give;
//   - hook scripts: the set's own test seams (the Hooks template parameter,
//     instantiated with TestHooks below) suspend a join before its increment,
//     a join between its increment and its re-verify, or an advance() between
//     its publish and its free pass, and run other steps right there; and
//     ProbeKey's call hooks suspend a reader inside contains(), or advance()'s
//     walk at its Hash{} of a FROZEN node. Each script states which protocol
//     defect it catches and what keeps the check from passing vacuously;
//   - stress tests, one second each, with witness counters that prove the
//     races they exist for happened, and an exact end state.
// Handle discipline: before a hooked new_handle() or advance(), a script
// releases every handle of its own thread (a live one pins its generation and
// turns "freed" into "kept"); operations nested in a TestHooks action run under
// handles created inside the action. A TestHooks action never throws, and no
// throwing ProbeKey hook or throw_countdown is ever armed where an action's
// operations could reach it: the hooks are noexcept, and a throw there ends the
// process.
// ===========================================================================

using Advance = AdvanceResult::Advance;

// GoogleTest's printer for an outcome compared on its own (found by
// argument-dependent lookup: the enum's associated namespace is the global one).
static void PrintTo(Advance a, std::ostream* os) {
    static constexpr const char* names[] = {"nothing_retired", "advanced", "ring_full", "contended"};
    *os << names[static_cast<size_t>(a)];
} // PrintTo(Advance)

// The whole result of a refused call: nothing done, every count 0.
static constexpr AdvanceResult kContended{0, 0, 0, 0, 0, Advance::contended};

// Expects advance()'s result `r` to have freed `fn` nodes in `fb` bags, its walk
// to have retired `wr` nodes, and the outcome `adv`. buckets_walked and
// nodes_passed are compared only where the contract fixes them for every set:
// a `ring_full` call does not walk, so both are 0 (a test that needs them in
// other cases compares them itself). `where` names the step in failure
// messages.
static void expect_advance(const AdvanceResult& r, size_t fn, size_t fb, size_t wr, Advance adv, const std::string& where) {
    EXPECT_EQ(r.freed_nodes, fn) << where << ": freed_nodes of " << ::testing::PrintToString(r);
    EXPECT_EQ(r.freed_bags, fb) << where << ": freed_bags of " << ::testing::PrintToString(r);
    EXPECT_EQ(r.walk_retired, wr) << where << ": walk_retired of " << ::testing::PrintToString(r);
    EXPECT_EQ(r.advance, adv) << where << ": outcome of " << ::testing::PrintToString(r);
    if (adv == Advance::ring_full) {
        EXPECT_EQ(r.buckets_walked + r.nodes_passed, 0u) << where << ": a ring_full call walked: " << ::testing::PrintToString(r);
    }
} // expect_advance()

// Expects the populations of the accounting sweep (consistent, with the
// five-way sum: expect_consistent()) to be exactly reachable `R`, of them dead
// `D`, retired `Rt`, bagged `B`, free `F`, limbo `L`, and the live handle count
// `H`. Quiescent only.
template <typename SetT>
static void expect_tuple(const SetT& set, const std::string& where, size_t R, size_t D, size_t Rt, size_t B, size_t F, size_t L, long H) {
    const typename SetT::InternalAccounting acc = expect_consistent(set, where);
    EXPECT_EQ(acc.reachable, R) << where << ": reachable";
    EXPECT_EQ(acc.reachable_dead, D) << where << ": reachable dead";
    EXPECT_EQ(acc.retired, Rt) << where << ": retired";
    EXPECT_EQ(acc.bagged, B) << where << ": bagged";
    EXPECT_EQ(acc.free_nodes, F) << where << ": free";
    EXPECT_EQ(acc.limbo, L) << where << ": limbo";
    EXPECT_EQ(acc.handle_refs, H) << where << ": handle refs";
} // expect_tuple()

// The ring-state view as one string: a token per slot in ring order -- E<refs>
// for an EMPTY slot, C<refs> for the CURRENT one, S<refs>/<bagged> for a SEALED
// one -- then "o=<oldest slot> c=<current slot>". Quiescent only (it asserts the
// advance flag clear).
template <typename SetT>
static std::string ring_string(const SetT& set) {
    const typename SetT::InternalRingState ring = set.get_internal_ring_state();
    std::string s;
    for (const typename SetT::InternalRingSlot& slot : ring.slots) {
        if (slot.state == SetT::GenerationState::SEALED) {
            s += "S" + std::to_string(slot.refs) + "/" + std::to_string(slot.bagged);
        } else {
            s += (slot.state == SetT::GenerationState::CURRENT ? "C" : "E") + std::to_string(slot.refs);
        }
        s += ' ';
    } // loop over the slots
    return s + "o=" + std::to_string(ring.oldest_slot) + " c=" + std::to_string(ring.current_slot);
} // ring_string()

// The ring_string() expected of a ring of `K` slots whose first slots are the
// space-separated tokens of `head` and every further slot EMPTY with no count
// ("E0"), with the oldest and current indices `o` and `c`.
static std::string ring_expect(size_t K, const std::string& head, size_t o, size_t c) {
    const size_t tokens = static_cast<size_t>(std::count(head.begin(), head.end(), ' ')) + 1;
    std::string s = head + ' ';
    for (size_t i = tokens; i < K; ++i) s += "E0 ";
    return s + "o=" + std::to_string(o) + " c=" + std::to_string(c);
} // ring_expect()

// One erase of `k` under a handle created for it and released before the
// advance() that follows: a node retired in the current generation, then the
// call, with no handle of the calling thread alive. Returns the call's result.
template <typename SetT>
static AdvanceResult cycle(SetT& set, int k) {
    {
        const typename SetT::handle h = set.new_handle();
        EXPECT_TRUE(set.erase(h, k)) << "cycle(): erase(" << k << ") of a present key";
    }
    return set.advance();
} // cycle()

// ---------------------------------------------------------------------------
// TestHooks: the set's three test seams (concurrent_hash_rcu_detail::SetHooks),
// each a one-shot slot a script arms with an action. The set calls
//   in_join()                  on every join attempt of new_handle() and of
//                              refresh()'s slow path, after the attempt's load
//                              of the current generation and before its
//                              increment of that generation's count;
//   in_join_after_bump()       on the same attempts, after the increment and
//                              before the re-verify;
//   in_advance_after_publish() once per publish of a generation, whichever
//                              call publishes -- an advance(), a reclaim() or,
//                              on a cadenced set, an automatic trigger inside
//                              an operation of the thread --, after the publish
//                              and before the (final) free pass and, in
//                              advance(), the walk, with the advance flag held.
// Never from refresh()'s fast path, a move, a leave or advance()'s own walk
// handle, whose join is hook-free; never from a skipped trigger.
// Rules:
//   - One-shot: firing disarms the slot and moves the action out BEFORE running
//     it, so a nested call reaching the same hook only counts, and the action
//     may re-arm any slot, its own included. Arm a slot only when its NEXT call
//     is the one wanted (a nested join reaches both join hooks).
//   - `calls` counts every call, armed or not; scripts assert its deltas.
//   - An action captures locals by reference and never owns a handle (a static
//     handle would outlive the set). It must not throw (the hooks are
//     noexcept), refresh, move or destroy the handle the interrupted join is
//     creating, or call the accounting sweep; inside in_advance_after_publish()
//     it must not call get_internal_generation(), get_internal_ring_state() or
//     get_internal_bag_count() either (they assert the advance flag clear).
//   - yield_in_windows is for the stress tests, which arm no slot: every hook
//     call then yields, which widens the windows between a join's load and its
//     increment, between its increment and its re-verify, and between a publish
//     and the free pass.
//   - HashRcuHookTest resets everything in SetUp() and TearDown(): a slot armed
//     but never fired cannot fire in a later test, and its captures are
//     dropped, never invoked.
// Single-threaded scripts arm and fire on one thread; in the stress tests no
// slot is armed, so `action` is never touched concurrently.
// ---------------------------------------------------------------------------
struct HookSlot {
    std::atomic<bool> armed{false};   // one-shot gate, set by arm(), cleared by the call that fires
    std::function<void()> action;     // written only by the arming thread while disarmed
    std::atomic<long> calls{0};       // every call of this hook since the last reset, armed or not
}; // struct HookSlot

struct TestHooks {
    static inline HookSlot join;                               // in_join()
    static inline HookSlot join_after_bump;                    // in_join_after_bump()
    static inline HookSlot after_publish;                      // in_advance_after_publish()
    static inline std::atomic<bool> yield_in_windows{false};   // stress mode: yield in every hook call

    static void in_join() noexcept { fire(join); }
    static void in_join_after_bump() noexcept { fire(join_after_bump); }
    static void in_advance_after_publish() noexcept { fire(after_publish); }

    // Arms `slot` to run `action` at its next call.
    static void arm(HookSlot& slot, std::function<void()> action) {
        slot.action = std::move(action);
        slot.armed.store(true, std::memory_order_release);
    } // TestHooks::arm()
    // Disarms every slot, drops the actions uninvoked, zeroes the call counts and
    // clears yield_in_windows.
    static void reset() noexcept {
        for (HookSlot* slot : {&join, &join_after_bump, &after_publish}) {
            slot->armed.store(false, std::memory_order_relaxed);
            slot->action = nullptr;
            slot->calls.store(0, std::memory_order_relaxed);
        } // loop over the slots
        yield_in_windows.store(false, std::memory_order_relaxed);
    } // TestHooks::reset()

private:
    // Counts the call; in stress mode yields; if armed, disarms, moves the action
    // out and runs it. The acquire exchange pairs with arm()'s release store.
    static void fire(HookSlot& slot) noexcept {
        slot.calls.fetch_add(1, std::memory_order_relaxed);
        if (yield_in_windows.load(std::memory_order_relaxed)) std::this_thread::yield();
        if (slot.armed.load(std::memory_order_relaxed) && slot.armed.exchange(false, std::memory_order_acquire)) {
            std::function<void()> action = std::move(slot.action);
            slot.action = nullptr;
            action();
        } // if armed: fire once
    } // TestHooks::fire()
}; // struct TestHooks
static_assert(concurrent_hash_rcu_detail::SetHooks<TestHooks>);

// The hooked instantiation of the scripts: int keys, erase(), the identity hash,
// a ring of K generations.
template <size_t K>
using HookSet = ConcurrentResizableHashSetRCU<int, true, std::hash<int>, K, TestHooks>;

// The fixture of every test that arms a TestHooks slot or ProbeKey's call hooks
// next to an advance(): both are global, so both are reset before and after.
class HashRcuHookTest : public ::testing::Test {
protected:
    void SetUp() override {
        TestHooks::reset();
        ProbeKey::reset_counters();
    }
    void TearDown() override {
        TestHooks::reset();
        ProbeKey::reset_counters();
    }
}; // class HashRcuHookTest

// A snapshot of the hook call counts and of the two set counters the scripts
// assert deltas of. join_retries counts the hooked joins' failed re-verifies
// only (advance()'s walk handle never retries: under the flag only the flag
// holder publishes).
struct HookCounts {
    long join;              // TestHooks::join.calls
    long join_after_bump;   // TestHooks::join_after_bump.calls
    long after_publish;     // TestHooks::after_publish.calls
    size_t join_retries;    // InternalCounters::join_retries
    size_t contended;       // InternalCounters::contended
}; // struct HookCounts
template <typename SetT>
static HookCounts hook_counts(const SetT& set) {
    const typename SetT::InternalCounters c = set.get_internal_counters();
    return {TestHooks::join.calls.load(), TestHooks::join_after_bump.calls.load(), TestHooks::after_publish.calls.load(),
            c.join_retries, c.contended};
} // hook_counts()

// A join on a fresh set lands in generation 0, and the count of that generation
// is the number of live handles.
//
// Kills: a join that does not count itself in the generation it joins (refs 0
// under a live handle: the free pass could then free under it), or that counts
// itself twice; a fresh ring that does not start with slot 0 CURRENT.
TEST(ConcurrentHashSetRcuTest, JoinOnFreshSetLandsInGenerationZero) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    Set set(256, 1);
    EXPECT_EQ(ring_string(set), ring_expect(64, "C0", 0, 0)) << "a fresh ring must have slot 0 CURRENT and nothing else";
    Set::handle h = set.new_handle();
    EXPECT_TRUE(static_cast<bool>(h));
    EXPECT_EQ(Set::get_internal_handle_generation(h), 0u);
    EXPECT_EQ(set.get_internal_generation(), 0u);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    EXPECT_EQ(ring_string(set), ring_expect(64, "C1", 0, 0));
    {
        const Set::handle h2 = set.new_handle();
        EXPECT_EQ(set.get_internal_accounting().handle_refs, 2);
    }
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1) << "a handle's destruction must leave its generation";
    h = {};
    EXPECT_FALSE(static_cast<bool>(h));
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 0) << "assigning an empty handle over a live one must leave its generation";
} // JoinOnFreshSetLandsInGenerationZero

// advance() with nothing retired and nothing in limbo returns nothing_retired,
// publishes nothing, and still runs its free pass and its walk.
//
// Kills: an empty call that bumps the generation anyway (it would consume a
// ring slot per call: the generation moves); an empty call that skips the walk
// (buckets_walked 0), or the free pass (the bag a released handle no longer pins
// stays: second half). The walk visits exactly the published buckets and passes
// every reachable node once.
// Pre-sized Set(256, 1): the constructor publishes its 256 buckets, 40 keys of
// identity hash in them are 40 reachable nodes, and nothing doubles (one append
// batch, hint 256, threshold 512).
TEST(ConcurrentHashSetRcuTest, AdvanceWithNothingRetiredStillWalks) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    Set set(256, 1);
    {
        const Set::handle h = set.new_handle();
        for (int k = 0; k < 40; ++k) ASSERT_TRUE(set.insert(h, k));
    }
    ASSERT_EQ(set.get_internal_table_size(), 256u) << "test precondition: the table doubled";
    const Set::InternalCounters c0 = set.get_internal_counters();
    EXPECT_EQ(set.advance(), (AdvanceResult{0, 0, 0, 256, 40, Advance::nothing_retired})) << "a fresh set's first advance()";
    EXPECT_EQ(set.get_internal_generation(), 0u) << "an advance() with nothing retired published a generation";
    EXPECT_EQ(set.get_internal_counters().bumps, c0.bumps);
    EXPECT_EQ(ring_string(set), ring_expect(64, "C0", 0, 0));

    // A bag that a held handle pinned at its call is freed by a later EMPTY call
    // once the handle is gone: the free pass runs on nothing_retired too.
    Set::handle h = set.new_handle();
    ASSERT_TRUE(set.erase(h, 7));
    expect_advance(set.advance(), 0, 0, 0, Advance::advanced, "the bag closed under a held handle");
    expect_tuple(set, "pinned", 39, 0, 0, 1, 0, 0, 1);
    h = {};
    EXPECT_EQ(set.advance(), (AdvanceResult{1, 1, 0, 256, 39, Advance::nothing_retired})) << "the empty call after the pin left";
    expect_tuple(set, "freed", 39, 0, 0, 0, 1, 0, 0);
} // AdvanceWithNothingRetiredStillWalks

// The walk skips the UNINITIALIZED buckets, and only those.
//
// Kills: a walk that visits pending buckets (buckets_walked would count them)
// or splits them (they would become published: the walk must split nothing).
// Five inserts take a Set(4, 1) to 128 buckets (WHEN THE TABLE DOUBLES, rule 1),
// most of them still pending.
TEST(ConcurrentHashSetRcuTest, AdvanceWalkSkipsUninitializedBuckets) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CollisionHash>;
    Set set(4, 1);
    Set::handle h = set.new_handle();
    ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {1, 2, 3, 4, 5}, 128));
    const size_t published = published_buckets(set);
    ASSERT_LT(published, 128u) << "test precondition: no bucket is pending";
    const Set::InternalCounters c0 = set.get_internal_counters();
    h = {};
    const AdvanceResult r = set.advance();
    EXPECT_EQ(r, (AdvanceResult{0, 0, 0, published, 5, Advance::nothing_retired}));
    EXPECT_EQ(published_buckets(set), published) << "advance()'s walk published a bucket";
    EXPECT_EQ(set.get_internal_counters().split_attempts, c0.split_attempts) << "advance()'s walk split a bucket";
} // AdvanceWalkSkipsUninitializedBuckets

// With every handle released, the bag advance() closes is freed by the SAME
// call; with a handle held across the erase and refreshed before each call
// (the fast path, before the first: the handle is already current), the node is
// freed by the SECOND call after its retirement.
//
// Kills: a free pass that runs before the bump (the closing bag would wait for
// the next call: freed_nodes 0 in the first half); a refresh() whose fast path
// moves the handle or whose slow path leaves the old generation pinned (the
// second half's counts). The second call of the second half finds nothing
// retired, so it is `nothing_retired`, and its free pass frees the bag.
TEST(ConcurrentHashSetRcuTest, AdvanceLatencyReleasedAndRefreshedHandle) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    {   // every handle released before the call: freed by the next call
        Set set(256, 1);
        {
            const Set::handle h = set.new_handle();
            ASSERT_TRUE(set.insert(h, 1));
            ASSERT_TRUE(set.erase(h, 1));
        } // h's scope
        EXPECT_EQ(set.advance(), (AdvanceResult{1, 1, 0, 256, 0, Advance::advanced}));
        expect_tuple(set, "released", 0, 0, 0, 0, 1, 0, 0);
        EXPECT_EQ(set.get_internal_generation(), 1u);
        EXPECT_EQ(ring_string(set), ring_expect(64, "E0 C0", 1, 1));
    } // released
    {   // a long-lived handle refreshed before each call: freed by the second call
        Set set(256, 1);
        Set::handle h = set.new_handle();
        ASSERT_TRUE(set.insert(h, 1));
        ASSERT_TRUE(set.erase(h, 1));
        EXPECT_FALSE(h.refresh()) << "refresh() of a current handle must not move it";
        expect_advance(set.advance(), 0, 0, 0, Advance::advanced, "call 1: the refreshed handle pins the bag it closes");
        expect_tuple(set, "after call 1", 0, 0, 0, 1, 0, 0, 1);
        EXPECT_EQ(ring_string(set), ring_expect(64, "S1/1 C0", 0, 1));
        EXPECT_TRUE(h.refresh()) << "refresh() after a publish must move the handle";
        EXPECT_EQ(Set::get_internal_handle_generation(h), 1u);
        EXPECT_EQ(ring_string(set), ring_expect(64, "S0/1 C1", 0, 1)) << "refresh() must leave the old generation";
        expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "call 2");
        expect_tuple(set, "after call 2", 0, 0, 0, 0, 1, 0, 1);
    } // refreshed
} // AdvanceLatencyReleasedAndRefreshedHandle

// The cumulative rule: a live handle of generation g keeps the bags of g AND of
// every later generation, although no handle of a later generation exists.
//
// Kills: a non-cumulative free pass (one that skips a pinned bag and frees the
// younger, unpinned ones behind it) and a free pass that ignores the counts, by
// the counts; the poisoned form of the same kills is
// SuspendedReaderPinsItsGeneration.
// Handle h pins generation 0; a, b and c are retired in generations 0, 1 and 2
// under handles released before each call, so bags 1 and 2 are pinned by
// nothing but the rule.
TEST(ConcurrentHashSetRcuTest, AdvanceFreesCumulativelyOldestFirst) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    Set set(256, 1);
    {
        const Set::handle hs = set.new_handle();
        for (int k : {1, 2, 3}) ASSERT_TRUE(set.insert(hs, k));
    }
    Set::handle h = set.new_handle();   // the pin, generation 0
    for (int k : {1, 2, 3}) {
        expect_advance(cycle(set, k), 0, 0, 0, Advance::advanced, "cycle(" + std::to_string(k) + ")");
    }
    expect_tuple(set, "three bags under the pin", 0, 0, 0, 3, 0, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(64, "S1/1 S0/1 S0/1 C0", 0, 3));
    h = {};
    expect_advance(set.advance(), 3, 3, 0, Advance::nothing_retired, "the pin left");
    expect_tuple(set, "all freed", 0, 0, 0, 0, 3, 0, 0);
    EXPECT_EQ(ring_string(set), ring_expect(64, "E0 E0 E0 C0", 3, 3));
    EXPECT_EQ(set.get_internal_bag_count(0), 0u) << "a freed generation has no bag";
} // AdvanceFreesCumulativelyOldestFirst

// A limbo node goes through a bag like a retired one: a held handle keeps it,
// and it is freed only once the handle is gone.
//
// Kills: a bump that pushes the limbo list straight onto a free list (bagged 0
// and free up by one under the pin: the popper ABA the header's LIMBO section
// excludes), and a bump that does not take the limbo list at all (limbo stays 1,
// and the call is nothing_retired: limbo must count in A0).
// A popped slot whose copy assignment throws goes to limbo (ProbeKey).
TEST(ConcurrentHashSetRcuTest, LimboNodesTravelThroughTheBag) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    ProbeKey::reset_counters();
    {
        Set set(256, 1);
        {
            const Set::handle hs = set.new_handle();
            for (int k : {1, 2}) ASSERT_TRUE(set.insert(hs, k));
            for (int k : {1, 2}) ASSERT_TRUE(set.erase(hs, k));
        } // hs's scope
        expect_advance(set.advance(), 2, 1, 0, Advance::advanced, "two free nodes");
        Set::handle h = set.new_handle();
        ProbeKey::throw_countdown = 1;   // the popped slot's assignment throws
        EXPECT_THROW(set.insert(h, 3), std::runtime_error);
        ProbeKey::throw_countdown = 0;
        expect_tuple(set, "after the throw", 0, 0, 0, 0, 1, 1, 1);
        expect_advance(set.advance(), 0, 0, 0, Advance::advanced, "limbo only, under the pin");
        expect_tuple(set, "the limbo node in the pinned bag", 0, 0, 0, 1, 1, 0, 1);
        EXPECT_EQ(set.get_internal_bag_count(1), 1u) << "generation 1's bag must hold the limbo node";
        h = {};
        expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "the pin left");
        expect_tuple(set, "freed", 0, 0, 0, 0, 2, 0, 0);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // LimboNodesTravelThroughTheBag

// A straggler -- a dead node left in its chain because the one-shot unlink that
// should have taken it never ran -- is bypassed by the walk of advance() call k
// and freed by call k + 1, with every handle released before each call.
//
// Kills: a walk that unlinks nothing (reachable dead stays 1); a walk that frees
// what it bypasses in the same call (free up at call k: the straggler is visible
// to handles of every live generation, so it may only be retired, into the
// generation the call left current); a walk that retires into the closed bag.
// Construction: EraseWalkCollectsCleanupStraggler's (contains(33)'s split
// cleanup throws, leaving 33's FROZEN parent copy in bucket 1, dead), in both
// positions: behind the live 65 (insert order 1, 33, 65, 2) and at the bucket
// head (1, 65, 33, 2). Nothing else is retired or in limbo, so call k finds
// every list empty: nothing_retired, with one walk retiree.
TEST(ConcurrentHashSetRcuTest, AdvanceWalkCollectsStragglerInTwoClicks) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    for (const bool at_head : {false, true}) {
        const std::string where = at_head ? "at the head" : "behind a live node";
        ProbeKey::reset_counters();
        {
            Set set(4, 1);
            Set::handle h = set.new_handle();
            ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, at_head ? std::vector<int>{1, 65, 33, 2} : std::vector<int>{1, 33, 65, 2}, 64));
            CleanupWindow w;
            arm_cleanup_window(set, w, 33, []() { throw std::runtime_error("Hash{} armed to throw in the cleanup"); });
            EXPECT_THROW(set.contains(h, 33), std::runtime_error) << where;
            ProbeKey::disarm_hooks();
            ASSERT_TRUE(w.before_publish && w.after_publish) << where << ": test precondition: the cleanup window did not open";
            h = {};
            const Set::InternalAccounting left = expect_consistent(set, where + ", the straggler");
            ASSERT_EQ(left.reachable_dead, 1u) << where << ": test precondition: no straggler";
            ASSERT_EQ(left.retired + left.limbo + left.bagged, 0u) << where << ": test precondition: something else to free";
            const uint64_t gen = set.get_internal_generation();

            expect_advance(set.advance(), 0, 0, 1, Advance::nothing_retired, where + ", call k");
            expect_tuple(set, where + ", after call k", left.reachable - 1, 0, 1, 0, left.free_nodes, 0, 0);
            EXPECT_EQ(set.get_internal_generation(), gen) << where;
            expect_advance(set.advance(), 1, 1, 0, Advance::advanced, where + ", call k + 1");
            expect_tuple(set, where + ", after call k + 1", left.reachable - 1, 0, 0, 0, left.free_nodes + 1, 0, 0);
            h = set.new_handle();
            for (int k : {1, 2, 33, 65}) EXPECT_TRUE(set.contains(h, k)) << where << ", key " << k;
        } // the set's lifetime
        EXPECT_EQ(ProbeKey::corrupt.load(), 0) << where;
    } // behind a live node, then at the head
} // AdvanceWalkCollectsStragglerInTwoClicks

// A churn loop with a fixed live count, reclaimed every few operations, keeps
// the table at its size: the resize hint follows the occupancy.
//
// Kills: a free pass that does not subtract what it frees from node_count_ (the
// hint then grows by one per pop and the table doubles over and over: past
// 2048 buckets within the loop), and a hint that wraps below zero (every insert
// would double the table). The bound, 512, is derived: the live count is W = 64
// and at most 16 erased nodes wait between two calls, so at most about 80 slots
// are occupied, and the hint over-counts them by at most 255 for the appends and
// 255 for the pops: below 2*512, so at most one doubling, from 256 to 512.
// Pre-sized Set(256, 1), identity hash, one thread.
TEST(ConcurrentHashSetRcuTest, ChurnWithAdvanceKeepsTableBounded) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int W = 64, OPS = 4000, EVERY = 16;
    Set set(256, 1);
    Set::handle h = set.new_handle();
    for (int k = 0; k < W; ++k) ASSERT_TRUE(set.insert(h, k));
    for (int i = 0; i < OPS; ++i) {
        ASSERT_TRUE(set.erase(h, i));
        ASSERT_TRUE(set.insert(h, i + W));
        if (i%EVERY == EVERY - 1) {
            h = {};
            set.advance();
            h = set.new_handle();
        } // advance every EVERY operations
    } // churn
    EXPECT_LE(set.get_internal_table_size(), 512u) << "the table doubled with a constant live count: the resize hint does not follow the frees";
    EXPECT_LE(set.get_internal_node_count(), 256u) << "the arena grew: the churn was not served from the free list";
    ASSERT_NO_FATAL_FAILURE(advance_and_check(set, h, W, "after the churn"));
} // ChurnWithAdvanceKeepsTableBounded

// Freed nodes do not strand on the free lists of shards no allocating thread
// uses: an allocation whose own shard's free list is empty takes a free node
// from another shard's list before it grows the arena.
//
// Weak spot: the free pass deals freed nodes round-robin over EVERY shard's
// free list, and a thread pops first from its own shard's list. Without the
// steal, with more shards than allocating threads, the nodes dealt to the
// other shards would never be reused, and under churn with a bounded live set
// the arena would grow without bound although every freed node is free.
// Kills: an append path that grows the arena while another shard's free list
// still holds a node.
// Shape: one allocating thread (this one) on a set of 16 shards, pre-sized to
// 4096 buckets so that nothing doubles or splits (identity hash; every key
// below 4096 has a bucket of its own; the resize hint stays far below 8192):
// L = 256 live keys, then ROUNDS rounds of R = 512 steps "erase the oldest
// live key, insert a new one" (a sliding window of keys 0, 1, 2, ...), each
// round ended by releasing the handle and one advance(). Every erase is
// uncontended, so it retires its own node and nothing is left in a chain:
// each advance() frees exactly that round's R nodes, dealt R/16 = 32 to each
// shard (R is a multiple of 16, so wherever the dealing cursor starts).
// The bound, derived: round 0 allocates its R nodes by appending (no free node
// exists yet), so the arena holds L + R slots after it. From round 1 on, the
// free lists hold exactly R nodes when the round starts and the round needs R
// allocations, so if every allocation takes a free node before appending, the
// arena stays at exactly L + R slots and every free list is empty when the
// round ends (each other shard's count falls from 32 to 0). Stranded, only
// this thread's shard would serve: 32 pops, 480 appends in round 1, and more
// every round after.
TEST(ConcurrentHashSetRcuTest, FreedNodesDoNotStrandOnUnusedShards) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const int L = 256, R = 512, ROUNDS = 6;
    const size_t SHARDS = 16;
    Set set(4096, SHARDS);
    int oldest = 0;   // the live keys are oldest .. oldest + L - 1
    {
        const Set::handle h = set.new_handle();
        for (int k = 0; k < L; ++k) ASSERT_TRUE(set.insert(h, k));
    }
    for (int round = 0; round < ROUNDS; ++round) {
        const std::string where = "round " + std::to_string(round);
        {
            const Set::handle h = set.new_handle();
            for (int i = 0; i < R; ++i, ++oldest) {
                ASSERT_TRUE(set.erase(h, oldest)) << where;
                ASSERT_TRUE(set.insert(h, oldest + L)) << where;
            } // the round's churn
        } // h's scope: the round's handle
        const Set::InternalAccounting churned = expect_consistent(set, where + ", after the churn");
        EXPECT_EQ(churned.slots, size_t(L + R)) << where << ": the arena grew while freed nodes waited on other shards' free lists";
        EXPECT_EQ(churned.free_nodes, 0u) << where << ": freed nodes were left unused on a free list while the round allocated";
        EXPECT_EQ(churned.retired, size_t(R)) << where << ": test precondition: every erase retires its own node";
        expect_advance(set.advance(), R, 1, 0, Advance::advanced, where + ": the round's nodes, in one bag");
        const Set::InternalAccounting freed = expect_consistent(set, where + ", after advance()");
        ASSERT_EQ(freed.free_per_shard.size(), SHARDS);
        for (size_t s = 0; s < SHARDS; ++s) EXPECT_EQ(freed.free_per_shard[s], size_t(R)/SHARDS) << where << ", shard " << s << ": a free list holds more than this round's even share: earlier rounds' nodes stranded there, or the deal is uneven";
    } // rounds
    EXPECT_EQ(set.get_internal_table_size(), 4096u) << "test precondition: the table doubled";
} // FreedNodesDoNotStrandOnUnusedShards

// ---------------------------------------------------------------------------
// The hook scripts. Each runs on HookSet<K> for K = 2, 3 and 64 generations: K =
// 2 is the minimum ring (every bump wraps), 3 the smallest ring with a slot
// between the oldest and the current one, 64 the default. One test per K, each
// calling the script's body template, so that every instance has a plain test
// name of its own.
// Every set is pre-sized and one-shard, Set(256, 1): no doubling, split or
// limbo node in any script (fewer than 256 slots are appended, and a doubling
// needs the hint above 512), and every freed node lands on the one free list the
// next pop reads. Notation: the tuple (R, D, Rt, B, F, L, H) of expect_tuple(),
// the ring of ring_string(), and the hook and counter deltas of hook_counts().
// Each script names the defects it kills -- the protocol step it exists for,
// removed or misplaced -- and what each one changes in its expected values.
// ---------------------------------------------------------------------------

// JOIN-BEFORE-INCREMENT (in_join): an advance() completes between a join's load
// of the current generation and its increment, so the increment lands on a slot
// that is EMPTY by then; the re-verify must fail, undo the increment and retry.
// Steps (every handle of the test thread released before each hooked call):
//   0. insert X;                                              (1,0,0,0,0,0,0)
//   1. arm in_join with { erase X under h2; rh = advance() }, then J =
//      new_handle(): rh frees X in the same call; J's first attempt is stale
//      (slot 0 EMPTY), so it retries and joins slot 1. join calls +3 (J's
//      attempt, h2's join, J's retry), join_retries +1;      (0,0,0,0,1,0,1)
//      ring [E0 C1 ...] o=1 c=1; J's generation 1;
//   2. insert Y under J: pops X's node;                       (1,0,0,0,0,0,1)
//   3. erase Y under h3;                                      (0,0,1,0,0,0,1)
//   4. advance(): J pins slot 1, nothing freed;               (0,0,0,1,0,0,1)
//   5. J released, advance(): Y's bag freed, nothing_retired. (0,0,0,0,1,0,0)
// Kills: a join without the re-verify (J keeps the EMPTY slot 0, join calls +2,
// no retry, generation 0, and step 4 frees Y under J); a failed re-verify that
// skips the undo (H = 2 at step 1, and the destructor's live-handle assert); an
// undo on the block the re-verify loaded (slot 1's count goes negative: the
// header's assert). Not vacuous: rh must be `advanced` (a call that bumps
// nothing leaves J's load current, and all three defects then change nothing),
// and no handle of the test
// thread is alive across the hooked new_handle() (a live one in slot 0 would
// keep it SEALED, and the EMPTY-slot oracle would be gone).
template <size_t K>
static void hook_join_race_retries_body() {
    using Set = HookSet<K>;
    const int X = 1, Y = 2;
    Set set(256, 1);
    {
        const typename Set::handle hs = set.new_handle();
        ASSERT_TRUE(set.insert(hs, X));
    }
    expect_tuple(set, "step 0", 1, 0, 0, 0, 0, 0, 0);
    const HookCounts c0 = hook_counts(set);
    int ran = 0;
    AdvanceResult rh{};
    TestHooks::arm(TestHooks::join, [&]() {
        ++ran;
        {
            const typename Set::handle h2 = set.new_handle();
            EXPECT_TRUE(set.erase(h2, X));
        }
        rh = set.advance();
    }); // the in_join action
    typename Set::handle J = set.new_handle();
    const HookCounts c1 = hook_counts(set);
    ASSERT_EQ(ran, 1) << "test precondition: the in_join action did not run";
    expect_advance(rh, 1, 1, 0, Advance::advanced, "step 1, the action's advance()");
    EXPECT_EQ(c1.join - c0.join, 3) << "J's attempt, h2's join and J's retry";
    EXPECT_EQ(c1.join_after_bump - c0.join_after_bump, 3);
    EXPECT_EQ(c1.join_retries - c0.join_retries, 1u) << "J's first attempt must fail its re-verify";
    EXPECT_EQ(c1.after_publish - c0.after_publish, 1) << "the action's advance() must have published";
    expect_tuple(set, "step 1", 0, 0, 0, 0, 1, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "E0 C1", 1, 1)) << "step 1";
    EXPECT_EQ(Set::get_internal_handle_generation(J), 1u) << "step 1";

    EXPECT_TRUE(set.insert(J, Y));   // step 2
    expect_tuple(set, "step 2", 1, 0, 0, 0, 0, 0, 1);
    {   // step 3
        const typename Set::handle h3 = set.new_handle();
        EXPECT_TRUE(set.erase(h3, Y));
    }
    expect_tuple(set, "step 3", 0, 0, 1, 0, 0, 0, 1);
    expect_advance(set.advance(), 0, 0, 0, Advance::advanced, "step 4: J pins slot 1");   // step 4
    expect_tuple(set, "step 4", 0, 0, 0, 1, 0, 0, 1);
    EXPECT_EQ(ring_string(set), K == 2 ? ring_expect(K, "C0 S1/1", 1, 0) : ring_expect(K, "E0 S1/1 C0", 1, 2)) << "step 4";
    J = {};   // step 5
    expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "step 5");
    expect_tuple(set, "step 5", 0, 0, 0, 0, 1, 0, 0);
    EXPECT_EQ(set.get_internal_table_size(), 256u) << "test precondition: the table doubled";
} // hook_join_race_retries_body()

TEST_F(HashRcuHookTest, HookJoinRaceRetriesK2) { ASSERT_NO_FATAL_FAILURE(hook_join_race_retries_body<2>()); }
TEST_F(HashRcuHookTest, HookJoinRaceRetriesK3) { ASSERT_NO_FATAL_FAILURE(hook_join_race_retries_body<3>()); }
TEST_F(HashRcuHookTest, HookJoinRaceRetriesK64) { ASSERT_NO_FATAL_FAILURE(hook_join_race_retries_body<64>()); }

// JOIN-BEFORE-INCREMENT, the ring wrapped back: K cycles inside in_join bring
// the current generation back to slot 0, so J's increment lands on the recycled
// CURRENT slot and the re-verify SUCCEEDS (pointer equality across a wrap is
// enough: J looked at nothing before it). The positive control of that sentence
// of THE JOIN: a re-verify by generation number would fail here, and it kills
// none of the defects above (the missing re-verify included: J's slot is
// current anyway).
//   0. insert k_0 .. k_K;                                      (K+1,0,0,0,0,0,0)
//   1. arm in_join with K cycles (k_0 .. k_{K-1}), J = new_handle(): every cycle
//      frees its own node; join calls +(K+1), no retry;      (1,0,0,0,K,0,1)
//      ring [C1 ...] o=0 c=0; J's generation K;
//   2. cycle(k_K): J pins slot 0;                             (0,0,0,1,K,0,1)
//   3. J released, advance(): freed, nothing_retired.         (0,0,0,0,K+1,0,0)
template <size_t K>
static void hook_stale_bump_reverify_succeeds_on_recycled_slot_body() {
    using Set = HookSet<K>;
    Set set(256, 1);
    {
        const typename Set::handle hs = set.new_handle();
        for (int i = 0; i <= static_cast<int>(K); ++i) ASSERT_TRUE(set.insert(hs, i + 1));
    }
    expect_tuple(set, "step 0", K + 1, 0, 0, 0, 0, 0, 0);
    const HookCounts c0 = hook_counts(set);
    int ran = 0;
    std::vector<AdvanceResult> rr;
    TestHooks::arm(TestHooks::join, [&]() {
        ++ran;
        for (int i = 0; i < static_cast<int>(K); ++i) rr.push_back(cycle(set, i + 1));
    });
    typename Set::handle J = set.new_handle();
    const HookCounts c1 = hook_counts(set);
    ASSERT_EQ(ran, 1) << "test precondition: the in_join action did not run";
    ASSERT_EQ(rr.size(), K);
    for (size_t i = 0; i < K; ++i) expect_advance(rr[i], 1, 1, 0, Advance::advanced, "step 1, cycle " + std::to_string(i));
    EXPECT_EQ(c1.join - c0.join, static_cast<long>(K) + 1);
    EXPECT_EQ(c1.join_retries - c0.join_retries, 0u) << "J's re-verify on the recycled current slot must succeed";
    expect_tuple(set, "step 1", 1, 0, 0, 0, K, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "C1", 0, 0)) << "step 1";
    EXPECT_EQ(Set::get_internal_handle_generation(J), K) << "step 1";

    expect_advance(cycle(set, static_cast<int>(K) + 1), 0, 0, 0, Advance::advanced, "step 2: J pins slot 0");
    expect_tuple(set, "step 2", 0, 0, 0, 1, K, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "S1/1 C0", 0, 1)) << "step 2";
    J = {};
    expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "step 3");
    expect_tuple(set, "step 3", 0, 0, 0, 0, K + 1, 0, 0);
} // hook_stale_bump_reverify_succeeds_on_recycled_slot_body()

TEST_F(HashRcuHookTest, HookStaleBumpReverifySucceedsOnRecycledSlotK2) { ASSERT_NO_FATAL_FAILURE(hook_stale_bump_reverify_succeeds_on_recycled_slot_body<2>()); }
TEST_F(HashRcuHookTest, HookStaleBumpReverifySucceedsOnRecycledSlotK3) { ASSERT_NO_FATAL_FAILURE(hook_stale_bump_reverify_succeeds_on_recycled_slot_body<3>()); }
TEST_F(HashRcuHookTest, HookStaleBumpReverifySucceedsOnRecycledSlotK64) { ASSERT_NO_FATAL_FAILURE(hook_stale_bump_reverify_succeeds_on_recycled_slot_body<64>()); }

// JOIN-BEFORE-INCREMENT, one cycle further: K+1 cycles inside in_join recycle
// slot 0 to CURRENT and then seal and free it again before J's increment, so
// the re-verify fails, J undoes and retries onto slot 1.
//   0. insert k_0 .. k_{K+1};                                  (K+2,0,0,0,0,0,0)
//   1. arm in_join with K+1 cycles, J = new_handle(): join calls +(K+3), one
//      retry;                                                (1,0,0,0,K+1,0,1)
//      ring [E0 C1 ...] o=1 c=1; J's generation K+1;
//   2. cycle(k_{K+1}): J pins slot 1;                         (0,0,0,1,K+1,0,1)
//   3. J released, advance(): freed, nothing_retired.         (0,0,0,0,K+2,0,0)
// Kills the join without the re-verify (step 1: join calls +(K+2), generation K,
// slot 0 E1; step 2 frees under J), the skipped undo (H = 2 at step 1), the undo
// on the re-loaded block (the header's negative-count assert at step 1). Not
// vacuous: every cycle must be `advanced` (one cycle short turns
// this script into the one above), and no test-thread handle is alive.
template <size_t K>
static void hook_stale_bump_undo_after_recycle_body() {
    using Set = HookSet<K>;
    Set set(256, 1);
    {
        const typename Set::handle hs = set.new_handle();
        for (int i = 0; i <= static_cast<int>(K) + 1; ++i) ASSERT_TRUE(set.insert(hs, i + 1));
    }
    expect_tuple(set, "step 0", K + 2, 0, 0, 0, 0, 0, 0);
    const HookCounts c0 = hook_counts(set);
    int ran = 0;
    std::vector<AdvanceResult> rr;
    TestHooks::arm(TestHooks::join, [&]() {
        ++ran;
        for (int i = 0; i <= static_cast<int>(K); ++i) rr.push_back(cycle(set, i + 1));
    });
    typename Set::handle J = set.new_handle();
    const HookCounts c1 = hook_counts(set);
    ASSERT_EQ(ran, 1) << "test precondition: the in_join action did not run";
    ASSERT_EQ(rr.size(), K + 1);
    for (size_t i = 0; i <= K; ++i) expect_advance(rr[i], 1, 1, 0, Advance::advanced, "step 1, cycle " + std::to_string(i));
    EXPECT_EQ(c1.join - c0.join, static_cast<long>(K) + 3);
    EXPECT_EQ(c1.join_after_bump - c0.join_after_bump, static_cast<long>(K) + 3);
    EXPECT_EQ(c1.join_retries - c0.join_retries, 1u) << "J's increment on the re-freed slot 0 must fail its re-verify";
    expect_tuple(set, "step 1", 1, 0, 0, 0, K + 1, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "E0 C1", 1, 1)) << "step 1";
    EXPECT_EQ(Set::get_internal_handle_generation(J), K + 1) << "step 1";

    expect_advance(cycle(set, static_cast<int>(K) + 2), 0, 0, 0, Advance::advanced, "step 2: J pins slot 1");
    expect_tuple(set, "step 2", 0, 0, 0, 1, K + 1, 0, 1);
    J = {};
    expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "step 3");
    expect_tuple(set, "step 3", 0, 0, 0, 0, K + 2, 0, 0);
} // hook_stale_bump_undo_after_recycle_body()

TEST_F(HashRcuHookTest, HookStaleBumpUndoAfterRecycleK2) { ASSERT_NO_FATAL_FAILURE(hook_stale_bump_undo_after_recycle_body<2>()); }
TEST_F(HashRcuHookTest, HookStaleBumpUndoAfterRecycleK3) { ASSERT_NO_FATAL_FAILURE(hook_stale_bump_undo_after_recycle_body<3>()); }
TEST_F(HashRcuHookTest, HookStaleBumpUndoAfterRecycleK64) { ASSERT_NO_FATAL_FAILURE(hook_stale_bump_undo_after_recycle_body<64>()); }

// INCREMENT-BEFORE-RE-VERIFY (in_join_after_bump), the two seams in one script:
// slot 0 is freed before J's increment (in_join: one cycle), then recycled to
// CURRENT with J's stale +1 on it before J's re-verify (in_join_after_bump: K-1
// cycles); the re-verify succeeds and the stale +1 IS J's count. The +1 sat on
// the slot while it went from EMPTY to CURRENT: whoever recycles a slot must not
// reset its count (J2).
//   0. insert k_0 .. k_K;                                      (K+1,0,0,0,0,0,0)
//   1. arm in_join with { cycle(k_0); arm in_join_after_bump with K-1 cycles },
//      J = new_handle(): both join hooks +(K+1), no retry;   (1,0,0,0,K,0,1)
//      ring [C1 ...] o=0 c=0; J's generation K;
//   2. cycle(k_K): J pins slot 0;                             (0,0,0,1,K,0,1)
//   3. J released, advance(): freed, nothing_retired.         (0,0,0,0,K+1,0,0)
// Kills a count reset on recycling (the advancer STORES 0 into the count of the
// slot it makes CURRENT, or right after its publish): the store runs inside the
// second action, after
// J's increment and before its re-verify, and wipes J's +1: H = 0 at step 1,
// step 2 frees the bag under J, and J's release at step 3 drives the count
// negative (the header's assert). Every placement of that store between the
// choice of the next slot and the walk dies here, since the whole call runs
// inside the action. Not vacuous: in_join_after_bump is armed from INSIDE the
// in_join action, after its cycle (armed up front, that cycle's own nested
// join would consume it); exactly K-1 cycles in the second action, each
// `advanced` (one short leaves slot 0 EMPTY at J's re-verify, and the store hits
// a slot without J's +1; asserted by J's generation and no retry); no
// test-thread handle (a pinned slot makes the cycles end in ring_full).
template <size_t K>
static void hook_stale_increment_survives_recycle_body() {
    using Set = HookSet<K>;
    Set set(256, 1);
    {
        const typename Set::handle hs = set.new_handle();
        for (int i = 0; i <= static_cast<int>(K); ++i) ASSERT_TRUE(set.insert(hs, i + 1));
    }
    expect_tuple(set, "step 0", K + 1, 0, 0, 0, 0, 0, 0);
    const HookCounts c0 = hook_counts(set);
    int ran1 = 0, ran2 = 0;
    AdvanceResult ra{};
    std::vector<AdvanceResult> rb;
    TestHooks::arm(TestHooks::join, [&]() {
        ++ran1;
        ra = cycle(set, 1);
        TestHooks::arm(TestHooks::join_after_bump, [&]() {
            ++ran2;
            for (int j = 0; j < static_cast<int>(K) - 1; ++j) rb.push_back(cycle(set, j + 2));
        });
    }); // the in_join action
    typename Set::handle J = set.new_handle();
    const HookCounts c1 = hook_counts(set);
    ASSERT_EQ(ran1, 1) << "test precondition: the in_join action did not run";
    ASSERT_EQ(ran2, 1) << "test precondition: the in_join_after_bump action did not run";
    expect_advance(ra, 1, 1, 0, Advance::advanced, "step 1, the in_join cycle");
    ASSERT_EQ(rb.size(), K - 1);
    for (size_t j = 0; j + 1 < K; ++j) expect_advance(rb[j], 1, 1, 0, Advance::advanced, "step 1, in_join_after_bump cycle " + std::to_string(j));
    EXPECT_EQ(c1.join - c0.join, static_cast<long>(K) + 1);
    EXPECT_EQ(c1.join_after_bump - c0.join_after_bump, static_cast<long>(K) + 1);
    EXPECT_EQ(c1.join_retries - c0.join_retries, 0u) << "J's re-verify on the recycled slot must succeed";
    expect_tuple(set, "step 1", 1, 0, 0, 0, K, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "C1", 0, 0)) << "step 1: J's stale increment must be J's count";
    EXPECT_EQ(Set::get_internal_handle_generation(J), K) << "step 1";

    expect_advance(cycle(set, static_cast<int>(K) + 1), 0, 0, 0, Advance::advanced, "step 2: J pins slot 0");
    expect_tuple(set, "step 2", 0, 0, 0, 1, K, 0, 1);
    J = {};
    expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "step 3");
    expect_tuple(set, "step 3", 0, 0, 0, 0, K + 1, 0, 0);
} // hook_stale_increment_survives_recycle_body()

TEST_F(HashRcuHookTest, HookStaleIncrementSurvivesRecycleK2) { ASSERT_NO_FATAL_FAILURE(hook_stale_increment_survives_recycle_body<2>()); }
TEST_F(HashRcuHookTest, HookStaleIncrementSurvivesRecycleK3) { ASSERT_NO_FATAL_FAILURE(hook_stale_increment_survives_recycle_body<3>()); }
TEST_F(HashRcuHookTest, HookStaleIncrementSurvivesRecycleK64) { ASSERT_NO_FATAL_FAILURE(hook_stale_increment_survives_recycle_body<64>()); }

// INCREMENT-BEFORE-RE-VERIFY, one cycle further: K cycles in the second action
// recycle slot 0 (carrying J's stale +1) and SEAL it again; the free pass reads
// the stale +1 and stops there (a conservative delay, harmless by J1), J's
// re-verify fails, its undo takes slot 0 to 0, and J retries onto slot 1.
//   0. insert k_0 .. k_{K+1};                                  (K+2,0,0,0,0,0,0)
//   1. as above with K cycles in the second action: the last one frees nothing
//      (slot 0, the oldest, holds J's stale +1); both join hooks +(K+3), one
//      retry;                                                (1,0,0,1,K,0,1)
//      ring [S0/1 C1 ...] o=0 c=1; J's generation K+1;
//   2. advance(): the delayed bag freed, nothing_retired;     (1,0,0,0,K+1,0,1)
//      ring [E0 C1 ...] o=1 c=1;
//   3. cycle(k_{K+1}): J pins slot 1, nothing freed;
//   4. J released, advance(): freed, nothing_retired.         (0,0,0,0,K+2,0,0)
// Kills the count reset on recycling (where the slot becomes CURRENT, or right
// after the publish) at step 1: the wiped +1 lets the last cycle free slot 0
// (`{1,1,0,advanced}`), and J's undo then drives the count negative (the
// header's assert inside new_handle()). Not vacuous: as above.
template <size_t K>
static void hook_stale_increment_on_resealed_slot_body() {
    using Set = HookSet<K>;
    Set set(256, 1);
    {
        const typename Set::handle hs = set.new_handle();
        for (int i = 0; i <= static_cast<int>(K) + 1; ++i) ASSERT_TRUE(set.insert(hs, i + 1));
    }
    expect_tuple(set, "step 0", K + 2, 0, 0, 0, 0, 0, 0);
    const HookCounts c0 = hook_counts(set);
    int ran1 = 0, ran2 = 0;
    AdvanceResult ra{};
    std::vector<AdvanceResult> rb;
    TestHooks::arm(TestHooks::join, [&]() {
        ++ran1;
        ra = cycle(set, 1);
        TestHooks::arm(TestHooks::join_after_bump, [&]() {
            ++ran2;
            for (int j = 0; j < static_cast<int>(K); ++j) rb.push_back(cycle(set, j + 2));
        });
    }); // the in_join action
    typename Set::handle J = set.new_handle();
    const HookCounts c1 = hook_counts(set);
    ASSERT_EQ(ran1, 1) << "test precondition: the in_join action did not run";
    ASSERT_EQ(ran2, 1) << "test precondition: the in_join_after_bump action did not run";
    expect_advance(ra, 1, 1, 0, Advance::advanced, "step 1, the in_join cycle");
    ASSERT_EQ(rb.size(), K);
    for (size_t j = 0; j + 1 < K; ++j) expect_advance(rb[j], 1, 1, 0, Advance::advanced, "step 1, in_join_after_bump cycle " + std::to_string(j));
    expect_advance(rb[K - 1], 0, 0, 0, Advance::advanced, "step 1, the last cycle: slot 0 carries J's stale increment");
    EXPECT_EQ(c1.join - c0.join, static_cast<long>(K) + 3);
    EXPECT_EQ(c1.join_after_bump - c0.join_after_bump, static_cast<long>(K) + 3);
    EXPECT_EQ(c1.join_retries - c0.join_retries, 1u) << "J's re-verify on the resealed slot must fail";
    expect_tuple(set, "step 1", 1, 0, 0, 1, K, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "S0/1 C1", 0, 1)) << "step 1";
    EXPECT_EQ(Set::get_internal_handle_generation(J), K + 1) << "step 1";

    expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "step 2: the delayed bag");
    expect_tuple(set, "step 2", 1, 0, 0, 0, K + 1, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "E0 C1", 1, 1)) << "step 2";
    expect_advance(cycle(set, static_cast<int>(K) + 2), 0, 0, 0, Advance::advanced, "step 3: J pins slot 1");
    J = {};
    expect_advance(set.advance(), 1, 1, 0, Advance::nothing_retired, "step 4");
    expect_tuple(set, "step 4", 0, 0, 0, 0, K + 2, 0, 0);
} // hook_stale_increment_on_resealed_slot_body()

TEST_F(HashRcuHookTest, HookStaleIncrementOnResealedSlotK2) { ASSERT_NO_FATAL_FAILURE(hook_stale_increment_on_resealed_slot_body<2>()); }
TEST_F(HashRcuHookTest, HookStaleIncrementOnResealedSlotK3) { ASSERT_NO_FATAL_FAILURE(hook_stale_increment_on_resealed_slot_body<3>()); }
TEST_F(HashRcuHookTest, HookStaleIncrementOnResealedSlotK64) { ASSERT_NO_FATAL_FAILURE(hook_stale_increment_on_resealed_slot_body<64>()); }

// OPERATIONS-BETWEEN-PUBLISH-AND-FREE-PASS (in_advance_after_publish): inside the
// hook, a handle of the NEW generation erases L. Every exchange preceded the
// publish, so L's retire push lands on the fresh retired list, and this call
// frees only A, the node its bump bagged; a nested advance() is `contended`.
//   0. insert A, L; erase A;                                   (1,0,1,0,0,0,0)
//   1. arm in_advance_after_publish with { h2 = new_handle() (its generation is
//      1: the hook runs after the publish); erase L under h2; nested =
//      advance() }, then r1 = advance(): r1 frees A only, nested is contended;
//      join calls +1 (h2), contended +1;                    (0,0,1,0,1,0,0)
//      ring [E0 C0 ...] o=1 c=1;
//   2. r2 = advance(): L's bag freed, `advanced` (the flag was released).
//                                                            (0,0,0,0,2,0,0)
// Kills: the exchanges moved after the hook (L's push lands before the late
// exchange, A and L share bag 0, and r1 frees 2); the flag not taken, or
// released right after taking it (nested runs fully, bags L and frees both
// bags, and r1 frees nothing); the flag never released (r2 is contended, and
// the sweep asserts). Not vacuous: the hook fires only on a call that
// publishes, so A is retired first (step 0's tuple); no handle of the test
// thread is alive (with one, slot 0 is pinned, and the header and the moved
// exchanges both free nothing); L's erase retires L (step 1's Rt == 1); h2's
// generation 1 is what shows that the hook is AFTER the publish.
template <size_t K>
static void hook_late_push_after_publish_body() {
    using Set = HookSet<K>;
    const int A = 1, L = 2;
    Set set(256, 1);
    {
        const typename Set::handle hs = set.new_handle();
        ASSERT_TRUE(set.insert(hs, A));
        ASSERT_TRUE(set.insert(hs, L));
        ASSERT_TRUE(set.erase(hs, A));
    } // hs's scope
    expect_tuple(set, "step 0", 1, 0, 1, 0, 0, 0, 0);
    const HookCounts c0 = hook_counts(set);
    int ran = 0;
    uint64_t gen_in_hook = 0;
    AdvanceResult nested{};
    TestHooks::arm(TestHooks::after_publish, [&]() {
        ++ran;
        {
            const typename Set::handle h2 = set.new_handle();
            gen_in_hook = Set::get_internal_handle_generation(h2);
            EXPECT_TRUE(set.erase(h2, L));
        } // h2's scope
        nested = set.advance();
    }); // the in_advance_after_publish action
    const AdvanceResult r1 = set.advance();
    const HookCounts c1 = hook_counts(set);
    ASSERT_EQ(ran, 1) << "test precondition: the in_advance_after_publish action did not run";
    expect_advance(r1, 1, 1, 0, Advance::advanced, "step 1: only A is in the bag the call closed");
    EXPECT_EQ(nested, kContended) << "an advance() inside the hook must find the flag held";
    EXPECT_EQ(gen_in_hook, 1u) << "the hook must run after the publish";
    EXPECT_EQ(c1.after_publish - c0.after_publish, 1);
    EXPECT_EQ(c1.join - c0.join, 1) << "h2's join";
    EXPECT_EQ(c1.contended - c0.contended, 1u);
    expect_tuple(set, "step 1: L on the fresh retired list", 0, 0, 1, 0, 1, 0, 0);
    EXPECT_EQ(ring_string(set), ring_expect(K, "E0 C0", 1, 1)) << "step 1";
    expect_advance(set.advance(), 1, 1, 0, Advance::advanced, "step 2");
    expect_tuple(set, "step 2", 0, 0, 0, 0, 2, 0, 0);
} // hook_late_push_after_publish_body()

TEST_F(HashRcuHookTest, HookLatePushAfterPublishK2) { ASSERT_NO_FATAL_FAILURE(hook_late_push_after_publish_body<2>()); }
TEST_F(HashRcuHookTest, HookLatePushAfterPublishK3) { ASSERT_NO_FATAL_FAILURE(hook_late_push_after_publish_body<3>()); }
TEST_F(HashRcuHookTest, HookLatePushAfterPublishK64) { ASSERT_NO_FATAL_FAILURE(hook_late_push_after_publish_body<64>()); }

// The same, on every shard: a set of S = 4 arena shards, and inside the hook S
// helper threads started and joined ONE AT A TIME, each with a handle of its
// own, each erasing one key L_s. Threads started one after another get
// consecutive process-wide thread numbers, so their retire pushes land on all
// four shards' retired lists (each helper records its shard, and the test
// asserts the four are distinct: the witness that every shard was covered).
// Kills the exchanges of only SOME shards moved past the hook: the late shards'
// L_s would be bagged with A and freed by the same call.
// HookLatePushAfterPublish cannot see that: on one shard, the one exchange is
// either before the hook or after it. (A std::thread whose construction throws
// inside the noexcept hook terminates the process, as a bad_alloc would.)
template <size_t K>
static void hook_late_push_after_publish_every_shard_body() {
    using Set = HookSet<K>;
    const int S = 4, A = 1000;
    Set set(256, S);
    {
        const typename Set::handle hs = set.new_handle();
        ASSERT_TRUE(set.insert(hs, A));
        for (int s = 0; s < S; ++s) ASSERT_TRUE(set.insert(hs, s + 1));
        ASSERT_TRUE(set.erase(hs, A));
    } // hs's scope
    expect_tuple(set, "step 0", S, 0, 1, 0, 0, 0, 0);
    int ran = 0;
    std::vector<unsigned> shard_of(S, 0);
    TestHooks::arm(TestHooks::after_publish, [&]() {
        ++ran;
        for (int s = 0; s < S; ++s) {
            std::thread helper([&set, &shard_of, s]() {
                shard_of[s] = concurrent_hash_rcu_detail::thread_number() & (S - 1);
                const typename Set::handle h = set.new_handle();
                EXPECT_TRUE(set.erase(h, s + 1));
            }); // thread helper
            helper.join();
        } // one helper at a time
    }); // the in_advance_after_publish action
    const AdvanceResult r1 = set.advance();
    ASSERT_EQ(ran, 1) << "test precondition: the in_advance_after_publish action did not run";
    std::vector<unsigned> sorted = shard_of;
    std::sort(sorted.begin(), sorted.end());
    ASSERT_EQ(sorted, (std::vector<unsigned>{0, 1, 2, 3})) << "test precondition: the helpers did not cover every shard";
    expect_advance(r1, 1, 1, 0, Advance::advanced, "step 1: only A is in the bag the call closed");
    expect_tuple(set, "step 1: every L_s on a fresh retired list", 0, 0, S, 0, 1, 0, 0);
    expect_advance(set.advance(), S, 1, 0, Advance::advanced, "step 2");
    expect_tuple(set, "step 2", 0, 0, 0, 0, S + 1, 0, 0);
} // hook_late_push_after_publish_every_shard_body()

TEST_F(HashRcuHookTest, HookLatePushAfterPublishEveryShardK2) { ASSERT_NO_FATAL_FAILURE(hook_late_push_after_publish_every_shard_body<2>()); }
TEST_F(HashRcuHookTest, HookLatePushAfterPublishEveryShardK3) { ASSERT_NO_FATAL_FAILURE(hook_late_push_after_publish_every_shard_body<3>()); }
TEST_F(HashRcuHookTest, HookLatePushAfterPublishEveryShardK64) { ASSERT_NO_FATAL_FAILURE(hook_late_push_after_publish_every_shard_body<64>()); }

// A reader suspended inside contains() pins its generation: nodes retired while
// it is suspended -- in its generation and in later ones -- are bagged but not
// freed until it leaves, and when it resumes it steps onto them intact.
//
// The seam is ProbeKey's equal_hook on the reader's comparison against node X:
// contains() loads a node's link BEFORE it compares the node's value (H4), so
// when the hook fires the reader already holds the address of X's successor S
// in a local. Keys at table size 256, identity hash: KEY = 3, S = 259, X = 515,
// all in bucket 3, inserted KEY, S, X, so the chain is X -> S -> KEY; filler Y
// = 4 in bucket 4.
//   0. insert KEY, S, X, Y;                                    (4,0,0,0,0,0,0)
//   1. h = new_handle() (generation 0: the pin); contains(h, KEY) with the hook
//      on X's comparison running: { erase Y under h1; r1 = advance(); erase S
//      under h2; r2 = advance(); arm equal_hook on S }, each handle released
//      before the call after it. Y is bagged in generation 0 and S in
//      generation 1, and neither is freed: bag 0 is pinned by h, and bag 1 only
//      by the cumulative rule. The reader resumes onto S (its link marked,
//      leading on to KEY), compares S (stepped_on_s) and finds KEY;
//                                                            (2,0,0,2,0,0,1)
//      ring [S1/1 S0/1 C0 ...] o=0 c=2 for K >= 3;
//   2. h released; r3 = advance() frees both bags.            (2,0,0,0,2,0,0)
// K = 2: the second call finds the only other slot SEALED and pinned, so it is
// ring_full (no walk, S stays on its retired list), and the third call frees bag
// 0 in its in-wrap pass, bags S, publishes and frees S's bag too.
// Kills, at K >= 3 (K = 2 cannot kill the second: its ring has at most one
// SEALED slot, so no younger bag lies past a pinned one):
//   - a free pass that ignores the counts: r1 and r2 each free their bag, and
//     the reader then reads S's value after S was freed. Under ASan that is a
//     use-after-poison in ProbeKey::operator== (the poisoning of freed value
//     bytes is what turns it into a report: this is the functional half of the
//     poison control, and the defect's ASan log must show it); without ASan,
//     S's link now leads into the free list, which ends at EMPTY: a silent
//     wrong miss, `found` false;
//   - a non-cumulative free pass (one that skips the pinned bag 0 and frees the
//     younger bag 1): r2 frees S; then the same use-after-poison, or wrong miss.
// The counts kill both in every build. Not vacuous: h is the ONLY pin, and h1
// and h2 are released before their calls (h2 alive across r2 would pin bag 1,
// and the non-cumulative pass would free nothing); the hook fires on the
// reader's first comparison (X is the chain head: ran == 1); the
// link-before-compare order is witnessed by stepped_on_s == 1; S is unlinked and
// bagged, not left as a straggler (the ring view's /1 on slot 1); TWO calls and
// the filler Y are needed, because one call after the erase of S would bag S in
// generation 0, the pinned one, where the non-cumulative pass has nothing
// younger to free (and a call with nothing retired does not bump, which is why
// Y exists). The checks inside the action are EXPECTs, never ASSERTs: an ASSERT
// returning early would leave S unerased, the reader would read a live S, and
// the poison half would vanish silently while the counts still killed the
// defect.
template <size_t K>
static void suspended_reader_pins_its_generation_body() {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash, K>;
    const int KEY = 3, S = 259, X = 515, Y = 4;
    {
        Set set(256, 1);
        {
            const typename Set::handle hs = set.new_handle();
            for (int k : {KEY, S, X, Y}) ASSERT_TRUE(set.insert(hs, k));
        }
        expect_tuple(set, "step 0", 4, 0, 0, 0, 0, 0, 0);
        int ran = 0, stepped_on_s = 0;
        AdvanceResult r1{}, r2{};
        typename Set::handle h = set.new_handle();
        ProbeKey::arm(ProbeKey::equal_hook, X, 1, [&]() {
            ++ran;
            {
                const typename Set::handle h1 = set.new_handle();
                EXPECT_TRUE(set.erase(h1, Y));
            }
            r1 = set.advance();
            {
                const typename Set::handle h2 = set.new_handle();
                EXPECT_TRUE(set.erase(h2, S));
            }
            r2 = set.advance();
            ProbeKey::arm(ProbeKey::equal_hook, S, 1, [&]() { ++stepped_on_s; });
        }); // the equal_hook action on X
        const bool found = set.contains(h, KEY);
        ProbeKey::disarm_hooks();
        ASSERT_EQ(ran, 1) << "test precondition: the hook on the reader's comparison against X did not fire";
        EXPECT_EQ(stepped_on_s, 1) << "the reader did not step onto S after the hook";
        EXPECT_TRUE(found) << "the suspended reader missed KEY: it walked through a freed node";
        expect_advance(r1, 0, 0, 0, Advance::advanced, "r1: bag 0 = {Y}, pinned by the reader");
        if constexpr (K == 2) {
            expect_advance(r2, 0, 0, 0, Advance::ring_full, "r2: the only other slot is pinned");
            expect_tuple(set, "step 1", 2, 0, 1, 1, 0, 0, 1);
            EXPECT_EQ(ring_string(set), ring_expect(K, "S1/1 C0", 0, 1)) << "step 1";
        } else {
            expect_advance(r2, 0, 0, 0, Advance::advanced, "r2: bag 1 = {S}, behind the pinned bag 0");
            expect_tuple(set, "step 1", 2, 0, 0, 2, 0, 0, 1);
            EXPECT_EQ(ring_string(set), ring_expect(K, "S1/1 S0/1 C0", 0, 2)) << "step 1";
        } // K
        h = {};
        if constexpr (K == 2) {
            expect_advance(set.advance(), 2, 2, 0, Advance::advanced, "step 2: the in-wrap pass frees bag 0, the bump bags S");
        } else {
            expect_advance(set.advance(), 2, 2, 0, Advance::nothing_retired, "step 2");
        } // K
        expect_tuple(set, "step 2", 2, 0, 0, 0, 2, 0, 0);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // suspended_reader_pins_its_generation_body()

TEST_F(HashRcuHookTest, SuspendedReaderPinsItsGenerationK2) { ASSERT_NO_FATAL_FAILURE(suspended_reader_pins_its_generation_body<2>()); }
TEST_F(HashRcuHookTest, SuspendedReaderPinsItsGenerationK3) { ASSERT_NO_FATAL_FAILURE(suspended_reader_pins_its_generation_body<3>()); }
TEST_F(HashRcuHookTest, SuspendedReaderPinsItsGenerationK64) { ASSERT_NO_FATAL_FAILURE(suspended_reader_pins_its_generation_body<64>()); }

// The in-walk seam: advance()'s walk hashes a FROZEN node it passes right after
// a usable predecessor, to decide whether it is dead. build_unfinished_split()
// leaves exactly such a node (PROBE_X2, FROZEN behind the live 1029, its child
// unpublished, so not dead) and one limbo node; nothing in the bump or the free
// pass hashes, so a ProbeKey hash hook on PROBE_X2 fires in the walk, after the
// bump and the free pass, with the advance flag held and the walk's handle
// live.
// AdvanceWalkHoldsTheFlag: a nested advance() from the hook must be `contended`,
// and the outer call `advanced` (the limbo node counts) with that node freed.
// Kills: the flag released at any point before the walk (after the bump,
// between the bump and the free pass, between the free pass and the walk), the
// flag not taken, or released right after taking it: the nested call would run.
// A following advance() must not be contended (the flag was released at the
// end). Not vacuous: `ran == 1` fails loudly if the walk hashes no FROZEN node;
// every handle is released before the hooked call (the builder's own handle
// would pin the limbo node's bag: freed_nodes 0).
TEST_F(HashRcuHookTest, AdvanceWalkHoldsTheFlag) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    {
        Set set(256, 1);
        Set::handle h = set.new_handle();
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, h, {PROBE_X1, PROBE_X2}, next, live));
        h = {};
        int ran = 0;
        AdvanceResult nested{};
        ProbeKey::arm(ProbeKey::hash_hook, PROBE_X2, 1, [&]() {
            ++ran;
            nested = set.advance();
        });
        const AdvanceResult r = set.advance();
        ProbeKey::disarm_hooks();
        ASSERT_EQ(ran, 1) << "test precondition: advance()'s walk did not hash the FROZEN PROBE_X2";
        EXPECT_EQ(nested, kContended) << "an advance() from inside the walk must find the flag held";
        expect_advance(r, 1, 1, 0, Advance::advanced, "the limbo node, bagged and freed");
        expect_advance(set.advance(), 0, 0, 0, Advance::nothing_retired, "the call after: the flag was released");
        h = set.new_handle();
        EXPECT_TRUE(set.contains(h, PROBE_X2));
        EXPECT_EQ(expect_consistent(set, "after the split finished").reachable_dead, 0u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceWalkHoldsTheFlag

// AdvanceWalkThrowReleasesFlagAndHandle: the same state, and the hash hook in
// the walk throws. advance() must propagate the exception with the basic
// guarantee: the bump and the free pass are complete (the limbo node was bagged
// and freed before the walk), the advance flag and the walk's own handle are
// released, the accounting is consistent, and the next advance() proceeds
// (nothing_retired, not contended).
// Kills: no guard releasing the flag on the throwing path (the next call is
// contended, and the sweep asserts the flag clear); the walk's handle not
// released on a throw (handle refs 1, and the destructor's live-handle assert).
// The hook is armed outside any TestHooks action (this set has NoHooks): a
// throw from a TestHooks action would terminate. `ran == 1` proves the throw
// came from INSIDE the walk, which a test without the hook cannot.
TEST_F(HashRcuHookTest, AdvanceWalkThrowReleasesFlagAndHandle) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    {
        Set set(256, 1);
        Set::handle h = set.new_handle();
        int next = 0;
        size_t live = 0;
        ASSERT_NO_FATAL_FAILURE(build_unfinished_split(set, h, {PROBE_X1, PROBE_X2}, next, live));
        h = {};
        const size_t free0 = set.get_internal_free_count();
        int ran = 0;
        ProbeKey::arm(ProbeKey::hash_hook, PROBE_X2, 1, [&ran]() {
            ++ran;
            throw std::runtime_error("Hash{} armed to throw in advance()'s walk");
        });
        EXPECT_THROW((void)set.advance(), std::runtime_error);
        ProbeKey::disarm_hooks();
        ASSERT_EQ(ran, 1) << "test precondition: the throw did not come from advance()'s walk";
        expect_tuple(set, "after the throw", live, 0, 0, 0, free0 + 1, 0, 0);
        expect_advance(set.advance(), 0, 0, 0, Advance::nothing_retired, "the call after the throw");
        h = set.new_handle();
        EXPECT_TRUE(set.contains(h, PROBE_X1));
        EXPECT_TRUE(set.contains(h, PROBE_X2));
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // AdvanceWalkThrowReleasesFlagAndHandle

// The destructor unpoisons the free nodes before the arena destroys them.
//
// Kills: a destructor that does not unpoison: under ASan the arena's destruction
// of a free slot runs ~ProbeKey, which reads its own bytes (check(*this)) and
// so reads poisoned memory: a use-after-poison report. Only a T whose
// destructor reads itself can show it (~int reads nothing, and freeing poisoned
// memory is silent), and only with a non-empty free list at destruction
// (asserted). Without ASan nothing is poisoned and the test passes either way.
TEST_F(HashRcuHookTest, DestructorUnpoisonsFreeNodes) {
    using Set = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash>;
    {
        Set set(256, 1);
        {
            const Set::handle h = set.new_handle();
            ASSERT_TRUE(set.insert(h, 1));
            ASSERT_TRUE(set.erase(h, 1));
        } // h's scope
        expect_advance(set.advance(), 1, 1, 0, Advance::advanced, "one node freed");
        ASSERT_EQ(set.get_internal_free_count(), 1u) << "test precondition: the free list is empty at destruction";
    } // the set's destruction
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // DestructorUnpoisonsFreeNodes

// The free pass poisons the value bytes of every node it frees (under ASan):
// the accounting sweep's per-byte poison conjunct over the two free nodes.
//
// Kills: a free pass that does not poison (under ASan `consistent` is false;
// without ASan the conjunct is true by construction and the test passes either
// way). The conjunct can fail only once a free pass has populated a free list,
// which this test makes sure of (two free nodes, asserted first). The other
// two poison sites have their own killers: the pop's unpoison by every test
// that inserts after a freeing advance() (the reused node's assignment would
// write into poisoned bytes: HookJoinRaceRetries's step 2 is the first), and
// the destructor's by DestructorUnpoisonsFreeNodes.
TEST(ConcurrentHashSetRcuTest, FreeNodesArePoisonedAfterAdvance) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    Set set(256, 1);
    {
        const Set::handle h = set.new_handle();
        for (int k : {1, 2}) ASSERT_TRUE(set.insert(h, k));
        for (int k : {1, 2}) ASSERT_TRUE(set.erase(h, k));
    } // h's scope
    expect_advance(set.advance(), 2, 1, 0, Advance::advanced, "both nodes in one bag");
    ASSERT_EQ(set.get_internal_free_count(), 2u) << "test precondition: the free list does not hold both nodes";
    EXPECT_TRUE(expect_consistent(set, "two free nodes").consistent) << "a free node's value bytes are not poisoned (under ASan)";
} // FreeNodesArePoisonedAfterAdvance

// ---------------------------------------------------------------------------
// Stress: advance() beside running operations. Each test runs for one second
// (per sanitizer build), then checks witness counters -- proof that the races it
// exists for happened during the run -- and an exact end state: with every
// handle released, two advance() calls (the second frees what the first one's
// walk retired) leave no handle count, nothing retired, bagged or in limbo, no
// dead node reachable (nothing here throws, so no run sits behind a FROZEN
// node), every unreachable slot free, and exactly one reachable node per key,
// the keys being known from the threads' own exact booleans.
// Roles:
//   - writer t owns its keys: it inserts key i (true) and checks it present,
//     then erases key i - W (true) and checks it absent, a sliding window of W
//     live keys; its retirements fill the bags, and its inserts pop what the
//     free passes push;
//   - reader: probes keys inserted before the run and never erased (each must be
//     present) and keys never inserted (each must be absent), plus the writers'
//     keys without an oracle, to walk chains that hold dead nodes;
//   - advancer: advance() in a loop, tallying the outcomes and the bags freed.
// Every reader and writer refreshes its long-lived handle every iteration (a
// handle refreshed rarely pins the generation the advancers close, and nothing
// would be freed), or, in the per-operation variant, takes a new handle for
// every operation.
// The two-advancer run is also the probabilistic killer of a count reset when a
// slot is freed (a free pass that stores 0 into the count of a slot it makes
// EMPTY): a joiner's stale increment landing in that window is wiped, and its
// undo or leave drives the count
// negative, which the header asserts. Readers refreshing every iteration join
// whenever the generation moved, writers erase between calls so bags hold many
// nodes, and the yield variant widens every join window.
// ---------------------------------------------------------------------------

// The shape of one stress run.
struct StressShape {
    int writers;            // writer threads
    int readers;            // reader threads
    int advancers;          // advancer threads
    bool per_op_handles;    // true: a new handle per operation; false: long-lived handles refreshed every iteration
    size_t shards;          // arena shards (few, so that pops race the free pass's pushes on each free list)
    size_t buckets;         // initial table size
    int window;             // W: live keys per writer
    int stable;             // keys inserted before the run and never erased
    std::chrono::milliseconds run_for;   // how long the threads run
    bool report;            // print the run's witnesses on an INFO line
    // The automatic cadence. Both fields default to off, so the cadence-off
    // shapes' initializers (which end at `report`) need neither.
    size_t cadence = 0;     // Options::auto_reclaim_every of the set: 0 = off; > 0 = the automatic cadence, and the run's end state goes through advance_and_check()
    size_t slot_limit = 0;  // runaway guard: the run STOPS early (it does not fail) when the arena exceeds this many slots; 0 = no guard
    // The broad cadenced stress's variations (CadenceStressMixedHandlesAndCallers). Each
    // defaults to what every shape above has, so no other shape changes.
    int reclaimers = 0;           // threads calling reclaim() in a loop, beside the `advancers` (which call advance())
    bool mixed_handles = false;   // true: even-numbered writers and readers take a handle per operation, odd-numbered ones a long-lived handle (per_op_handles is then ignored)
    int refresh_every = 1;        // a long-lived handle is refreshed on every refresh_every-th iteration of its thread (1: every iteration; must be > 0)
    std::chrono::microseconds call_pause{0};   // advancers and reclaimers sleep this long after each call (0: back to back, the flag held most of the time)
}; // struct StressShape

// What happened during the run (before the end-state calls), for the witnesses.
struct StressWitness {
    size_t bags_freed = 0;                  // freed_bags summed over the advancers' calls
    size_t outcomes[4] = {0, 0, 0, 0};      // the advancers' results by AdvanceResult::Advance
    size_t join_retries = 0;                // InternalCounters deltas over the run
    size_t pops_raced_push = 0;
    size_t free_push_cas_failures = 0;
    size_t contended = 0;
    // The automatic cadence's, filled for a cadenced run only (StressShape::cadence > 0), all
    // from counters read at the quiescent point after the join: no ring view
    // is read during the run.
    size_t auto_reclaims = 0;        // InternalCounters deltas over the run: triggers that proceeded ...
    size_t auto_skipped = 0;         // ... and triggers that found the flag held
    size_t counted_bags_freed = 0;   // InternalCounters::bags_freed delta: every free pass's, the triggers' included (bags_freed above sums only the explicit advancers' results)
    size_t nodes_freed = 0;          // InternalCounters::nodes_freed delta
    size_t pops = 0;                 // free-list pops since construction: nodes_freed (cumulative) minus the free nodes at the quiescent point; exact, since only the free pass pushes and the lists start empty
    size_t reader_triggers = 0;      // reader iterations across which the reader's own bump_ns grew: a trigger proceeded inside one of its contains() (readers make no explicit call)
    bool stopped_by_guard = false;   // the runaway guard (StressShape::slot_limit) ended the run early
    size_t reclaim_outcomes[4] = {0, 0, 0, 0};   // the reclaimers' reclaim() results by AdvanceResult::Advance (StressShape::reclaimers)
}; // struct StressWitness

static int stress_writer_key(int t, int i) { return ((t + 1) << 22) | i; }   // writer t's key i, i < 2^22
static int stress_stable_key(int i) { return 0x30000000 | i; }               // present for the whole run
static int stress_never_key(int i) { return 0x38000000 | i; }                // never inserted

// Runs the stress of `shape` on a fresh SetT (int keys, AllowDelete), checks
// every thread's own oracles and the end state, and returns the witnesses for
// the caller's assertions. `where` labels messages. Callers wrap it in
// ASSERT_NO_FATAL_FAILURE.
// The set is constructed through Options with `shape.cadence`, which for a
// cadence-off shape (cadence 0) is the two-parameter construction (it
// delegates). A cadenced run differs only where the cadence makes the
// cadence-off code wrong or blind: the end state goes through
// advance_and_check() (valid on a cadenced set at a quiescent point; the
// cadence-off runs keep their two bare calls and their oracle), the trigger witnesses are filled from the
// counters after the join, readers record whether a trigger proceeded inside
// their own contains(), and a runaway guard may end the run early.
template <typename SetT>
static void advance_stress_body(const StressShape& shape, const std::string& where, StressWitness& w) {
    SetT set(typename SetT::Options{.initial_capacity = shape.buckets, .arena_shards = shape.shards, .auto_reclaim_every = shape.cadence});
    {
        const typename SetT::handle h = set.new_handle();
        for (int i = 0; i < shape.stable; ++i) ASSERT_TRUE(set.insert(h, stress_stable_key(i)));
    }
    const typename SetT::InternalCounters c0 = set.get_internal_counters();
    const int T = shape.writers + shape.readers + shape.advancers + shape.reclaimers;
    std::atomic<bool> stop{false};
    std::barrier start(T + 1);
    std::vector<size_t> bad(T, 0);
    std::vector<int> written(shape.writers, 0);   // writer t inserted its keys 0 .. written[t]-1
    std::vector<StressWitness> tallies(shape.advancers + shape.reclaimers);   // advancers first, then reclaimers
    std::vector<size_t> reader_triggers(shape.readers, 0);   // reader r's iterations with a trigger inside (cadenced runs only)
    std::vector<std::thread> threads;
    threads.reserve(T);
    // Whether writer or reader thread t (its index among all threads) takes a
    // handle per operation: every one under per_op_handles; under
    // mixed_handles the even-numbered ones, the odd-numbered keeping a
    // long-lived handle.
    auto per_op_thread = [&shape](int t) -> bool { return shape.mixed_handles ? t % 2 == 0 : shape.per_op_handles; };
    // One operation `op(handle)` of a writer or reader under its handle
    // discipline: a handle of its own for the operation (`per_op`), or the
    // thread's long-lived `h` (refreshed by the loops below on every
    // refresh_every-th iteration).
    auto under = [&set](bool per_op, const typename SetT::handle& h, auto op) -> bool {
        if (per_op) {
            const typename SetT::handle own = set.new_handle();
            return op(own);
        }
        return op(h);
    }; // under()
    for (int t = 0; t < shape.writers; ++t) {
        threads.emplace_back([&, t]() {
            const bool per_op = per_op_thread(t);
            typename SetT::handle h = per_op ? typename SetT::handle{} : set.new_handle();
            start.arrive_and_wait();
            const int W = shape.window;
            int i = 0;
            for (; !stop.load(std::memory_order_relaxed) && i < (1 << 22); ++i) {
                if (h && i % shape.refresh_every == 0) h.refresh();
                const int in = stress_writer_key(t, i), out = stress_writer_key(t, i - W);
                bad[t] += !under(per_op, h, [&](const typename SetT::handle& x) { return set.insert(x, in); });
                bad[t] += !under(per_op, h, [&](const typename SetT::handle& x) { return set.contains(x, in); });
                if (i >= W) {
                    bad[t] += !under(per_op, h, [&](const typename SetT::handle& x) { return set.erase(x, out); });
                    bad[t] += under(per_op, h, [&](const typename SetT::handle& x) { return set.contains(x, out); });
                } // the window's oldest key leaves
            } // writer loop
            written[t] = i;
        }); // writer thread t
    } // writers
    for (int r = 0; r < shape.readers; ++r) {
        const int t = shape.writers + r;
        threads.emplace_back([&, t, r]() {
            const bool per_op = per_op_thread(t);
            typename SetT::handle h = per_op ? typename SetT::handle{} : set.new_handle();
            std::minstd_rand rng(static_cast<unsigned>(r + 1));
            start.arrive_and_wait();
            for (int i = 0; !stop.load(std::memory_order_relaxed); ++i) {
                if (h && i % shape.refresh_every == 0) h.refresh();
                // A reader makes no explicit call, so its own thread_local
                // bump_ns grows only when a trigger PROCEEDS inside one of its
                // contains() (a split's cleanup it won retired a run that
                // crossed the cadence). One thread_local read per
                // iteration, on cadenced runs only.
                const size_t bump_before = shape.cadence != 0 ? SetT::internal_advance_timers().bump_ns : 0;
                const unsigned x = rng();
                const int stable = stress_stable_key(static_cast<int>(x%static_cast<unsigned>(shape.stable)));
                const int never = stress_never_key(static_cast<int>((x >> 8)%1024u));
                const int any = stress_writer_key(static_cast<int>((x >> 16)%static_cast<unsigned>(shape.writers)), static_cast<int>((x >> 4)%4096u));
                bad[t] += !under(per_op, h, [&](const typename SetT::handle& y) { return set.contains(y, stable); });
                bad[t] += under(per_op, h, [&](const typename SetT::handle& y) { return set.contains(y, never); });
                (void)under(per_op, h, [&](const typename SetT::handle& y) { return set.contains(y, any); });
                if (shape.cadence != 0 && SetT::internal_advance_timers().bump_ns != bump_before) ++reader_triggers[r];
            } // reader loop
        }); // reader thread r
    } // readers
    for (int a = 0; a < shape.advancers; ++a) {
        threads.emplace_back([&, a]() {
            StressWitness& tally = tallies[a];
            start.arrive_and_wait();
            while (!stop.load(std::memory_order_relaxed)) {
                const AdvanceResult r = set.advance();
                tally.bags_freed += r.freed_bags;
                ++tally.outcomes[static_cast<size_t>(r.advance)];
                if (shape.call_pause.count() != 0) std::this_thread::sleep_for(shape.call_pause);
            } // advancer loop
        }); // advancer thread a
    } // advancers
    for (int c = 0; c < shape.reclaimers; ++c) {
        threads.emplace_back([&, c]() {
            StressWitness& tally = tallies[shape.advancers + c];
            start.arrive_and_wait();
            while (!stop.load(std::memory_order_relaxed)) {
                const AdvanceResult r = set.reclaim();
                ++tally.reclaim_outcomes[static_cast<size_t>(r.advance)];
                if (shape.call_pause.count() != 0) std::this_thread::sleep_for(shape.call_pause);
            } // reclaimer loop
        }); // reclaimer thread c
    } // reclaimers
    start.arrive_and_wait();
    bool stopped_by_guard = false;
    if (shape.slot_limit == 0) {
        std::this_thread::sleep_for(shape.run_for);
    } else {
        // The runaway guard: poll the arena size (a race-free diagnostic)
        // every millisecond and end the run early once it passes the limit.
        // It stops rather than fails: the end state below is still checked,
        // and the caller sees `stopped_by_guard` (a run that cannot reuse
        // memory grows the arena by every insert, and the memory cap of the
        // run would otherwise end the process instead of the test).
        const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + shape.run_for;
        while (std::chrono::steady_clock::now() < deadline) {
            if (set.get_internal_node_count() > shape.slot_limit) {
                stopped_by_guard = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } // poll until the deadline or the limit
    } // if guarded
    stop.store(true, std::memory_order_relaxed);
    for (std::thread& th : threads) th.join();

    // The witnesses: what happened during the run.
    const typename SetT::InternalCounters c1 = set.get_internal_counters();
    w = StressWitness{};
    w.stopped_by_guard = stopped_by_guard;
    for (const StressWitness& tally : tallies) {
        w.bags_freed += tally.bags_freed;
        for (size_t o = 0; o < 4; ++o) {
            w.outcomes[o] += tally.outcomes[o];
            w.reclaim_outcomes[o] += tally.reclaim_outcomes[o];
        } // loop over the four outcomes
    } // sum the advancers' and the reclaimers' tallies
    w.join_retries = c1.join_retries - c0.join_retries;
    w.pops_raced_push = c1.pops_raced_push - c0.pops_raced_push;
    w.free_push_cas_failures = c1.free_push_cas_failures - c0.free_push_cas_failures;
    w.contended = c1.contended - c0.contended;
    if (shape.cadence != 0) {
        // Every thread is joined: a quiescent point, so the counters are
        // exact and the accounting sweep is valid (pops: the free lists
        // started empty and only the free pass pushes, so what left them is
        // what was pushed minus what is still there).
        w.auto_reclaims = c1.auto_reclaims - c0.auto_reclaims;
        w.auto_skipped = c1.auto_skipped - c0.auto_skipped;
        w.counted_bags_freed = c1.bags_freed - c0.bags_freed;
        w.nodes_freed = c1.nodes_freed - c0.nodes_freed;
        w.pops = c1.nodes_freed - expect_consistent(set, where + ", after the run").free_nodes;
        for (size_t n : reader_triggers) w.reader_triggers += n;
    } // if cadenced
    for (int t = 0; t < T; ++t) EXPECT_EQ(bad[t], 0u) << where << ", thread " << t << ": a determined result was wrong";
    if (shape.report) {
        std::printf("[   INFO   ] %s: advanced %zu, nothing_retired %zu, ring_full %zu, contended %zu, bags freed %zu; "
                    "join retries %zu, pops raced a push %zu, free pushes lost a CAS %zu; table %zu, slots %zu\n",
                    where.c_str(), w.outcomes[1], w.outcomes[0], w.outcomes[2], w.outcomes[3], w.bags_freed,
                    w.join_retries, w.pops_raced_push, w.free_push_cas_failures, set.get_internal_table_size(), set.get_internal_node_count());
        if (shape.cadence != 0) {
            std::printf("[   INFO   ] %s: cadence %zu: auto_reclaims %zu, auto_skipped %zu, bags freed (all) %zu, nodes freed %zu, "
                        "pops %zu, reader iterations with a trigger %zu%s\n",
                        where.c_str(), set.get_internal_auto_reclaim_every(), w.auto_reclaims, w.auto_skipped, w.counted_bags_freed,
                        w.nodes_freed, w.pops, w.reader_triggers, w.stopped_by_guard ? "; STOPPED by the runaway guard" : "");
        } // if cadenced
        if (shape.reclaimers != 0) {
            std::printf("[   INFO   ] %s: reclaim(): advanced %zu, nothing_retired %zu, ring_full %zu, contended %zu\n", where.c_str(),
                        w.reclaim_outcomes[1], w.reclaim_outcomes[0], w.reclaim_outcomes[2], w.reclaim_outcomes[3]);
        } // if reclaimers ran
    } // report

    // The end state: every handle is gone (each thread released its own).
    size_t live = static_cast<size_t>(shape.stable);
    for (int t = 0; t < shape.writers; ++t) live += static_cast<size_t>(std::min(written[t], shape.window));
    if (shape.cadence == 0) {
        (void)set.advance();
        (void)set.advance();
    } else {
        // On a cadenced set the full oracle of a quiescent advance(): valid
        // here because no operation runs, so no trigger can hold the flag
        // (beside operations an explicit call on a cadenced set may be
        // `contended`, and the helper's exact oracles would not hold). The
        // walk's own retirements cross
        // the cadence and are skipped (the flag is the walk's): no trigger
        // changes what the two calls do. The handle is released by the
        // helper before its calls and dropped here before the checks below,
        // which count none.
        typename SetT::handle settle = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(advance_and_check(set, settle, live, where + ", end state"));
    } // if cadenced
    const typename SetT::InternalAccounting acc = expect_consistent(set, where + ", end state");
    EXPECT_EQ(acc.handle_refs, 0) << where;
    EXPECT_EQ(acc.retired, 0u) << where;
    EXPECT_EQ(acc.bagged, 0u) << where;
    EXPECT_EQ(acc.limbo, 0u) << where;
    EXPECT_EQ(acc.reachable_dead, 0u) << where;
    EXPECT_EQ(acc.free_nodes, acc.slots - acc.reachable) << where;
    EXPECT_EQ(acc.reachable - acc.reachable_dead, live) << where << ": not exactly one reachable node per key";
    EXPECT_LE(set.get_internal_table_size(), std::max(shape.buckets, 2*acc.slots + 512*shape.shards))
        << where << ": the table grew past what the resize hint can reach with the arena it has";
    const typename SetT::handle h = set.new_handle();
    int errors = 0;
    for (int i = 0; i < shape.stable; ++i) errors += !set.contains(h, stress_stable_key(i));
    for (int t = 0; t < shape.writers; ++t) {
        for (int i = std::max(0, written[t] - shape.window); i < written[t]; ++i) errors += !set.contains(h, stress_writer_key(t, i));
        for (int i = std::max(0, written[t] - 2*shape.window); i < written[t] - shape.window; ++i) errors += set.contains(h, stress_writer_key(t, i));
    } // writers' last keys
    EXPECT_EQ(errors, 0) << where << ": membership after the run";
} // advance_stress_body()

// Writers, readers and TWO advancers, long-lived handles refreshed every
// iteration, under the default NoHooks.
// Witnesses: bags were freed during the run (the free pass ran beside the
// operations); a pop raced a free-pass push or a push lost its CAS (the free
// lists were written and read concurrently, the case the pop's and the push's
// orders exist for); a call was `contended` (the two advancers met at the flag).
// Kills, probabilistically: the count reset at the free pass (see above); any
// premature free that a sanitizer sees (a freed node's value read by a reader:
// ASan's poisoning; a reuse racing a reader: TSan), or that the readers' and
// writers' exact booleans or the end state see.
TEST(ConcurrentHashSetRcuTest, StressAdvanceBesideOperations) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const StressShape shape{4, 2, 2, false, 2, 1024, 64, 256, std::chrono::milliseconds(1000), true};
    StressWitness w;
    ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "two advancers", w));
    EXPECT_GT(w.bags_freed, 0u) << "witness: no bag was freed during the run";
    EXPECT_GT(w.pops_raced_push + w.free_push_cas_failures, 0u) << "witness: no pop raced a free-pass push, and no push lost its CAS";
    EXPECT_GT(w.contended, 0u) << "witness: the two advancers never met at the flag";
} // StressAdvanceBesideOperations

// The same under the TestHooks instantiation with yield_in_windows: every join
// yields between its load and its increment and between its increment and its
// re-verify, and the advancer yields between its publish and its free pass,
// with the flag held. The windows the hook scripts force one at a time are
// widened here for every thread at once.
// Witnesses: as above, plus calls of the join hook and of the after-publish
// hook (the windows were entered). No slot is armed.
TEST_F(HashRcuHookTest, StressAdvanceYieldInWindows) {
    using Set = ConcurrentResizableHashSetRCU<int, true, std::hash<int>, 64, TestHooks>;
    const StressShape shape{4, 2, 2, false, 2, 1024, 64, 256, std::chrono::milliseconds(1000), true};
    TestHooks::yield_in_windows.store(true);
    StressWitness w;
    ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "two advancers, yield in windows", w));
    EXPECT_GT(TestHooks::join.calls.load(), 0) << "witness: no join entered its windows";
    EXPECT_GT(TestHooks::after_publish.calls.load(), 0) << "witness: no advance() entered its publish-to-free-pass window";
    EXPECT_GT(w.bags_freed, 0u) << "witness: no bag was freed during the run";
    EXPECT_GT(w.contended, 0u) << "witness: the two advancers never met at the flag";
} // StressAdvanceYieldInWindows

// Per-operation handles UNDER the TestHooks instantiation with
// yield_in_windows: every operation joins and leaves, and every join yields
// between its load and its increment and between its increment and its
// re-verify, so a joiner's stale increment meets the free pass's check of a
// generation's count, and the slot's recycling, far more often than in either
// of the two tests it combines. It is the stress that catches a free pass
// storing 0 into the count of a slot it makes EMPTY (the wiped stale
// increment's undo drives the count negative, which the header asserts).
// Witnesses: join retries, join-hook calls, after-publish calls, bags freed,
// the advancers met at the flag.
TEST_F(HashRcuHookTest, StressAdvancePerOperationHandlesYieldInWindows) {
    using Set = ConcurrentResizableHashSetRCU<int, true, std::hash<int>, 64, TestHooks>;
    const StressShape shape{4, 2, 2, true, 2, 1024, 64, 256, std::chrono::milliseconds(1000), true};
    TestHooks::yield_in_windows.store(true);
    StressWitness w;
    ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "per-operation handles, yield in windows", w));
    EXPECT_GT(w.join_retries, 0u) << "witness: no join ever retried";
    EXPECT_GT(TestHooks::join.calls.load(), 0) << "witness: no join entered its windows";
    EXPECT_GT(TestHooks::after_publish.calls.load(), 0) << "witness: no advance() entered its publish-to-free-pass window";
    EXPECT_GT(w.bags_freed, 0u) << "witness: no bag was freed during the run";
    EXPECT_GT(w.contended, 0u) << "witness: the two advancers never met at the flag";
} // StressAdvancePerOperationHandlesYieldInWindows

// Per-operation handles: every operation joins and leaves, so joins run all the
// time against the advancers' publishes.
// Witness: a join's re-verify failed (an advance() published between a joiner's
// load and its re-verify: the retry path ran), and bags were freed.
TEST(ConcurrentHashSetRcuTest, StressAdvancePerOperationHandles) {
    using Set = ConcurrentResizableHashSetRCU<int, true>;
    const StressShape shape{4, 2, 2, true, 2, 1024, 64, 256, std::chrono::milliseconds(1000), true};
    StressWitness w;
    ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "per-operation handles", w));
    EXPECT_GT(w.join_retries, 0u) << "witness: no join ever retried";
    EXPECT_GT(w.bags_freed, 0u) << "witness: no bag was freed during the run";
} // StressAdvancePerOperationHandles

// Long chains, many tiny sets: ConstantHash puts every key and every dead node
// in one chain, so every erase unlinks through mid-chain predecessors that other
// erases are marking and unlinking, every insert walks it, and advance()'s walk
// passes all of it beside them. A fresh 4-bucket set per round, with a few
// hundred operations per writer, so that sets are created, churned, advanced
// and destroyed many times in the second (the end state checked each time).
// Witness: bags were freed during the rounds.
TEST(ConcurrentHashSetRcuTest, StressAdvanceLongChainsManyTinySets) {
    using Set = ConcurrentResizableHashSetRCU<int, true, ConstantHash>;
    const StressShape shape{4, 1, 1, false, 2, 4, 8, 16, std::chrono::milliseconds(20), false};
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    size_t rounds = 0, bags_freed = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        StressWitness w;
        ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "tiny set " + std::to_string(rounds), w));
        bags_freed += w.bags_freed;
        ++rounds;
    } // rounds
    std::printf("[   INFO   ] StressAdvanceLongChainsManyTinySets: %zu sets, %zu bags freed\n", rounds, bags_freed);
    EXPECT_GT(bags_freed, 0u) << "witness: no bag was freed during any round";
} // StressAdvanceLongChainsManyTinySets

// ===========================================================================
// reclaim() AND THE AUTOMATIC CADENCE, white-box.
//
// reclaim() is advance() without the walk: the flag or `contended`, the bump,
// the hook if it published, the final free pass. A set constructed with
// Options::auto_reclaim_every = R > 0 (rounded up to a power of two; above
// 2^63 clamped to 2^63; 0 = off) runs that same work ITSELF -- the automatic
// TRIGGER -- at the end of a retire push whose node count took the retiring
// arena shard's `retire_nodes` across a multiple of R: on the retiring
// thread, inside the operation that retired, under that operation's handle,
// after the push (so a proceeding trigger bags the crossing run too), under
// the try-flag (held: counted in `auto_skipped`, nothing else; acquired:
// `auto_reclaims` +1 whatever the bump then decides), never the walk. The
// header's THE TRIGGER (class overview), push_retired(), auto_reclaim() and
// bump_and_free() say how and why; each test below gives its derivation in
// its comment.
//
// What every script here shares:
//   - A cadenced set is the same type as its cadence-off twin, constructed
//     through Options (cadenced_set() below: the cadenced fixture); ONE arena
//     shard, except where a script's point is two, so the shard's count is the
//     set-wide `retire_nodes` the scripts read and no oracle depends on the
//     test thread's thread number.
//   - Tuples are expect_tuple()'s (reachable, of them dead, retired, bagged,
//     free, limbo, handle refs); results {freed_nodes, freed_bags,
//     walk_retired, buckets_walked, nodes_passed, advance}.
//   - THE CADENCE SETTING: a presized set (no doubling, no split, no limbo),
//     one shard, erases of PRESENT keys on one thread with no explicit call in
//     progress: each erase() that returns true retires exactly one node
//     (`retire_nodes` +1, `retire_runs` +1), so the R-th erase from a known
//     phase is the crossing one. Scripts that leave the setting on purpose
//     (a split's cleanup, the stragglers' multi-node runs, limbo) read the
//     counts they need from the counters.
//   - "Attempts" = auto_reclaims + auto_skipped: one per crossing push.
//     "Fires" = attempts +1 across the step, "proceeds" = auto_reclaims +1,
//     "is skipped" = auto_skipped +1.
//   - The trigger runs under the RETIRING operation's own handle, which pins
//     the generation the trigger closes: the R-th erase's trigger bags its
//     run and frees none of it; the next crossing frees it, after a refresh
//     between operations. Every "freed" expectation below is derived with
//     that pin.
//   - On a cadenced set the hook in_advance_after_publish() fires inside the
//     test thread's OWN operations (any crossing erase publishes), so a hook
//     slot is armed only right before the one call or crossing erase the
//     script derived, never across erases whose crossings it has not; inside a
//     trigger's hook the interrupted operation's handle is in flight and the
//     action touches neither it nor the interrupted operation's bucket; nested
//     operations use handles created in the action; a nested advance() or
//     reclaim() is `contended`; a nested crossing is skipped. Ring views and
//     the accounting sweep are read only between operations, on the test
//     thread, with no other thread running (operation quiescence: on a
//     cadenced set an operation can hold the flag), and never inside an
//     action (the death test below).
//   - A throwing ProbeKey hook is armed only on NoHooks sets (CadProbeSet), so
//     no TestHooks action can run inside a trigger while it is armed (a throw
//     there would unwind into a noexcept hook).
//   - Each test's comment says which defects it kills, and says ONLY where
//     no other test here kills one.
// The cadence's !NDEBUG destructor invariants (no attempt with the cadence
// off; at most one attempt per multiple crossed; at R = 1 exactly one per
// retired run) run at the end of every test here: every cadenced set
// destroyed is a check of the counting.
// ===========================================================================

// The hooked cadenced instantiation: HookSet<K>'s type, constructed through
// Options. A separate alias only so that a cadenced script says so.
template <size_t K>
using CadHookSet = HookSet<K>;
// The ProbeKey cadenced instantiation, NoHooks (a throwing ProbeKey hook may
// be armed on it: no TestHooks action exists).
template <size_t K>
using CadProbeSet = ConcurrentResizableHashSetRCU<ProbeKey, true, ProbeKeyHash, K>;

// The cadenced fixture: a SetT constructed through Options with `capacity`
// buckets, `shards` arena shards and the cadence `every` (designated
// initializers, the form the header's comment shows). Returned by value: a
// prvalue initializes the caller's object directly (the set is neither
// copyable nor movable).
template <typename SetT>
static SetT cadenced_set(size_t capacity, size_t shards, size_t every) {
    return SetT(typename SetT::Options{.initial_capacity = capacity, .arena_shards = shards, .auto_reclaim_every = every});
} // cadenced_set()

// The trigger attempts of `set` so far: auto_reclaims + auto_skipped, exact
// at a quiescent point (race-free counters).
template <typename SetT>
static size_t cadence_attempts(const SetT& set) {
    const typename SetT::InternalCounters c = set.get_internal_counters();
    return c.auto_reclaims + c.auto_skipped;
} // cadence_attempts()

// The R-th erase from a fresh shard fires, bags the R nodes (the crossing
// one included: the trigger runs after the push) and frees nothing (its own
// handle pins them); after a refresh between operations the 2R-th erase frees
// that first bag; a final reclaim() with no handle frees the second, without
// walking.
//   step 0: insert 0 .. 2R-1 under a released handle; phase 0 asserted.
//   step 1: under h, erase 0 .. R-2: no attempt; R-1 retired, no bump.
//   step 2: erase R-1: bumps +1, auto_reclaims +1, auto_skipped +0, contended
//           +0, joins +0 (no walk: the walk always joins), bags_freed +0, the
//           hook +1; this thread's advance_ns and walk_ns unchanged (the
//           trigger charges only bump_ns and free_ns, which do not shrink,
//           and grow over all the test's triggers); tuple
//           (R, 0, 0, R, 0, 0, 1); ring "S1/R C0" (h pins slot 0).
//   step 3: refresh h; erase R .. 2R-2 (no attempt); erase 2R-1: bumps +1,
//           bags_freed +1, nodes_freed +R; tuple (0, 0, 0, R, R, 0, 1). At
//           K = 2 the bump wraps onto slot 0, freed by the in-wrap pass.
//   step 4: release h; reclaim() == {R, 1, 0, 0, 0, nothing_retired}, compared
//           WHOLE (its walk counts are 0).
// R in {1, 2, 4, 64}: R = 1 makes step 1 empty, R >= 2 puts the crossing
// strictly after the first push.
// Kills: the trigger placed BEFORE the retire push (step 2: the crossing node
// stays retired, `retired` 1 and `bagged` R-1; at R = 1 the trigger finds
// every list empty, bumps +0); the crossing decided one push late, when the
// count before the push is a multiple (step 1 at R >= 2: the first erase
// fires); a trigger that also walks (step 2: joins +1); a trigger's publish
// that does not fire the hook (step 2: the hook +0); a trigger whose free
// pass takes its own operation's pin for none, and a free pass that ignores
// the pins altogether (step 2: bag 0's only pin is h, the retiring
// operation's own handle); a reclaim() that walks (step 4: buckets_walked and
// nodes_passed under the whole-literal comparison).
// `charged` accumulates the bump_ns + free_ns the step-2 trigger charged
// this thread, for the caller's one strict check over all its calls (one
// clock delta alone may be zero on a coarse clock).
template <size_t K>
static void cadence_fires_at_the_rth_erase_body(size_t R, size_t& charged) {
    using Set = CadHookSet<K>;
    const std::string where = "R = " + std::to_string(R);
    const int r = static_cast<int>(R);
    Set set = cadenced_set<Set>(256, 1, R);
    ASSERT_EQ(set.get_internal_auto_reclaim_every(), R) << where << ": test precondition: R is a power of two, kept as is";
    {
        const typename Set::handle hs = set.new_handle();
        for (int k = 0; k < 2*r; ++k) ASSERT_TRUE(set.insert(hs, k));
    }
    ASSERT_EQ(set.get_internal_counters().retire_nodes, 0u) << where << ": test precondition: the shard's phase is 0";
    expect_tuple(set, where + ", step 0", 2*R, 0, 0, 0, 0, 0, 0);

    typename Set::handle h = set.new_handle();
    for (int k = 0; k + 1 < r; ++k) {   // step 1
        const size_t a = cadence_attempts(set);
        EXPECT_TRUE(set.erase(h, k));
        EXPECT_EQ(cadence_attempts(set), a) << where << ": erase " << k << " fired before the R-th";
    } // erase 0 .. R-2
    EXPECT_EQ(set.get_internal_counters().bumps, 0u) << where;
    expect_tuple(set, where + ", step 1", R + 1, 0, R - 1, 0, 0, 0, 1);

    const typename Set::InternalCounters c0 = set.get_internal_counters();   // step 2
    const long p0 = TestHooks::after_publish.calls.load();
    const typename Set::InternalTimers t0 = Set::internal_advance_timers();
    EXPECT_TRUE(set.erase(h, r - 1));
    const typename Set::InternalCounters c1 = set.get_internal_counters();
    const typename Set::InternalTimers t1 = Set::internal_advance_timers();
    EXPECT_EQ(c1.bumps - c0.bumps, 1u) << where << ": the R-th erase did not publish";
    EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 1u) << where;
    EXPECT_EQ(c1.auto_skipped - c0.auto_skipped, 0u) << where;
    EXPECT_EQ(c1.contended - c0.contended, 0u) << where;
    EXPECT_EQ(c1.joins - c0.joins, 0u) << where << ": the trigger walked (the walk always joins)";
    EXPECT_EQ(c1.bags_freed - c0.bags_freed, 0u) << where << ": the trigger freed a bag its own operation's handle pins";
    EXPECT_EQ(TestHooks::after_publish.calls.load() - p0, 1) << where << ": the trigger's publish did not fire the hook once";
    EXPECT_EQ(t1.advance_ns, t0.advance_ns) << where << ": a trigger charged advance_ns (the explicit calls' timer)";
    EXPECT_EQ(t1.walk_ns, t0.walk_ns) << where << ": a trigger charged walk_ns";
    EXPECT_GE(t1.bump_ns + t1.free_ns, t0.bump_ns + t0.free_ns) << where;
    charged += (t1.bump_ns + t1.free_ns) - (t0.bump_ns + t0.free_ns);
    expect_tuple(set, where + ", step 2", R, 0, 0, R, 0, 0, 1);
    EXPECT_EQ(ring_string(set), ring_expect(K, "S1/" + std::to_string(R) + " C0", 0, 1)) << where;

    h.refresh();   // step 3: between operations, so the next trigger may free bag 0
    for (int k = r; k + 1 < 2*r; ++k) {
        const size_t a = cadence_attempts(set);
        EXPECT_TRUE(set.erase(h, k));
        EXPECT_EQ(cadence_attempts(set), a) << where << ": erase " << k << " fired before the 2R-th";
    } // erase R .. 2R-2
    const typename Set::InternalCounters c2 = set.get_internal_counters();
    EXPECT_TRUE(set.erase(h, 2*r - 1));
    const typename Set::InternalCounters c3 = set.get_internal_counters();
    EXPECT_EQ(c3.bumps - c2.bumps, 1u) << where;
    EXPECT_EQ(c3.bags_freed - c2.bags_freed, 1u) << where << ": the second trigger did not free the first bag";
    EXPECT_EQ(c3.nodes_freed - c2.nodes_freed, R) << where;
    expect_tuple(set, where + ", step 3", 0, 0, 0, R, R, 0, 1);

    h = {};   // step 4
    EXPECT_EQ(set.reclaim(), (AdvanceResult{R, 1, 0, 0, 0, Advance::nothing_retired})) << where;
} // cadence_fires_at_the_rth_erase_body()
TEST_F(HashRcuHookTest, CadenceFiresAtTheRthEraseK2) {
    size_t charged = 0;
    for (size_t R : {1, 2, 4, 64}) ASSERT_NO_FATAL_FAILURE(cadence_fires_at_the_rth_erase_body<2>(R, charged));
    EXPECT_GT(charged, 0u) << "the triggers charged neither bump_ns nor free_ns of their thread";
} // CadenceFiresAtTheRthEraseK2
TEST_F(HashRcuHookTest, CadenceFiresAtTheRthEraseK64) {
    size_t charged = 0;
    for (size_t R : {1, 2, 4, 64}) ASSERT_NO_FATAL_FAILURE(cadence_fires_at_the_rth_erase_body<64>(R, charged));
    EXPECT_GT(charged, 0u) << "the triggers charged neither bump_ns nor free_ns of their thread";
} // CadenceFiresAtTheRthEraseK64

// A trigger inside a split's cleanup that a contains() started (a contains()
// that wins a split runs its cleanup like any splitter, and the cleanup's
// retire push may trigger). R = 1, int keys, NoHooks.
// A presized set has no UNINITIALIZED bucket, so ONE controlled doubling is
// built as leave_five_stragglers() builds it, without its throwing cleanups:
// under h, keys whose chains at 128 buckets put 385 -> 129 -> 257 in bucket 1,
// then fillers 256*(64 + i) until the table doubles to 256. Bucket 129 is then
// pending (asserted). h is released before the measured call: it pinned every
// bag the construction's own triggers closed (so the ring may be full of
// them), and with it released the measured trigger's bump proceeds. Then
// contains(h2, 129): it splits 129 out of bucket 1, whose two movers 385 and
// 129 sit at the head, frozen; the cleanup unlinks both as ONE run (retire_runs
// +1, retire_nodes +2), which crosses R = 1 and triggers: bumps +1 ==
// retire_runs +1, auto_reclaims +1, auto_skipped +0.
// Kills: the trigger placed before the retire push (it finds the lists empty:
// bumps +0 against retire_runs +1). There is no contains()-specific trigger
// code to break: push_retired() is the one site.
TEST(ConcurrentHashSetRcuTest, CadenceTriggerInsideSplitCleanup) {
    using Set = IntSetDel;
    Set set = cadenced_set<Set>(4, 1, 1);
    {
        const Set::handle h = set.new_handle();
        ASSERT_NO_FATAL_FAILURE(insert_in_order(set, h, {0, 256, 512, 768, 1024, 257, 129, 385, 258, 130, 514, 131, 259, 387, 515}, 128));
        for (int i = 0; set.get_internal_table_size() == 128; ++i) {   // fillers, until the doubling to 256
            ASSERT_LT(i, 300) << "test precondition: the fillers did not double the table";
            ASSERT_TRUE(set.insert(h, 256*(64 + i)));
        }
    } // h pinned the construction's bags; released here
    ASSERT_EQ(set.get_internal_table_size(), 256u);
    ASSERT_FALSE(set.get_internal_bucket_published(129)) << "test precondition: bucket 129 is published, no split to win";
    const Set::handle h2 = set.new_handle();
    const Set::InternalCounters c0 = set.get_internal_counters();
    EXPECT_TRUE(set.contains(h2, 129));
    const Set::InternalCounters c1 = set.get_internal_counters();
    EXPECT_EQ(c1.splits_published - c0.splits_published, 1u) << "the contains() did not win the split of 129";
    EXPECT_EQ(c1.retire_runs - c0.retire_runs, 1u) << "the cleanup retired the two frozen movers in other than one run";
    EXPECT_EQ(c1.retire_nodes - c0.retire_nodes, 2u);
    EXPECT_EQ(c1.bumps - c0.bumps, 1u) << "the cleanup's crossing push did not publish (bumps must equal the runs retired)";
    EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 1u);
    EXPECT_EQ(c1.auto_skipped - c0.auto_skipped, 0u);
} // CadenceTriggerInsideSplitCleanup

// A crossing inside an explicit advance()'s hook is SKIPPED (the flag
// is the advance()'s); nested advance() and reclaim() are `contended`. R = 4.
//   step 0: insert 0..9; erase 0, 1, 2 under a released handle: phase 3 = R-1
//           (asserted: one erase fewer and the nested one would not cross,
//           and every flag defect below would pass), no attempt;
//           (7,0,3,0,0,0,0).
//   step 1: arm after_publish with: under h2 created in the action, erase 3
//           (crosses 4: skipped); a nested advance() and a nested reclaim().
//           Then advance() with no handle alive: it publishes (bag 0 = {0,1,2}
//           is non-empty, so the hook fires), frees bag 0 in the same call
//           (no pin), and its walk finds nothing dead.
//           Expect: the action ran once; both nested calls kContended; the
//           outer {3, 1, 0, advanced}; bumps +1, auto_skipped +1,
//           auto_reclaims +0, contended +2, the hook +1, retire_nodes +1;
//           (6,0,1,0,3,0,0): node 3 on generation 1's retired list.
// The nested erase is in a bucket the interrupted call does not walk (it
// interrupts advance(), not an operation).
// Kills: a trigger that ignores the flag (the nested trigger proceeds: bumps
// +2, auto_reclaims +1, and its guard clears the flag, so the nested calls
// proceed too); a skipped trigger counted as `contended` (contended +3,
// auto_skipped +0); a skipped trigger that fires the hook (the hook +2); a
// reclaim() that ignores the flag (the nested reclaim() proceeds); an
// advance() that ignores the flag (the nested advance() proceeds).
TEST_F(HashRcuHookTest, CadenceTriggerSkippedUnderExplicitAdvance) {
    using Set = CadHookSet<64>;
    Set set = cadenced_set<Set>(256, 1, 4);
    {
        const Set::handle hs = set.new_handle();
        for (int k = 0; k < 10; ++k) ASSERT_TRUE(set.insert(hs, k));
    }
    {
        const Set::handle h1 = set.new_handle();
        for (int k = 0; k < 3; ++k) ASSERT_TRUE(set.erase(h1, k));
    }
    ASSERT_EQ(set.get_internal_counters().retire_nodes, 3u) << "test precondition: the phase is R - 1";
    ASSERT_EQ(cadence_attempts(set), 0u);
    expect_tuple(set, "step 0", 7, 0, 3, 0, 0, 0, 0);
    int ran = 0;
    AdvanceResult na{}, nr{};
    TestHooks::arm(TestHooks::after_publish, [&]() {
        ++ran;
        {
            const Set::handle h2 = set.new_handle();
            EXPECT_TRUE(set.erase(h2, 3)) << "the nested erase";
        }
        na = set.advance();
        nr = set.reclaim();
    });
    const Set::InternalCounters c0 = set.get_internal_counters();
    const long p0 = TestHooks::after_publish.calls.load();
    const AdvanceResult r = set.advance();
    const Set::InternalCounters c1 = set.get_internal_counters();
    EXPECT_EQ(ran, 1);
    EXPECT_EQ(na, kContended) << "a nested advance() inside the hook";
    EXPECT_EQ(nr, kContended) << "a nested reclaim() inside the hook";
    expect_advance(r, 3, 1, 0, Advance::advanced, "the explicit advance()");
    EXPECT_EQ(c1.bumps - c0.bumps, 1u) << "the skipped trigger published";
    EXPECT_EQ(c1.auto_skipped - c0.auto_skipped, 1u);
    EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 0u);
    EXPECT_EQ(c1.contended - c0.contended, 2u) << "contended counts the two nested explicit calls, never a skipped trigger";
    EXPECT_EQ(c1.retire_nodes - c0.retire_nodes, 1u);
    EXPECT_EQ(TestHooks::after_publish.calls.load() - p0, 1) << "a skipped trigger fired the hook, or a nested call published";
    expect_tuple(set, "after the call", 6, 0, 1, 0, 3, 0, 0);
} // CadenceTriggerSkippedUnderExplicitAdvance

// advance()'s walk retires through the same push as the operations,
// with the flag held by its own thread: every crossing of the walk's runs is
// SKIPPED (R = 1: every push crosses). leave_five_stragglers() under h (five
// dead nodes in four runs, ProbeKey's throwing cleanups armed and fired with
// no TestHooks action: CadProbeSet is NoHooks), h released, then advance():
// walk_retired 5; retire_runs +4; auto_skipped +4 (= the walk's runs);
// auto_reclaims +0; contended +0; no dead node reachable afterwards. The call's
// own bump may be `advanced` or `nothing_retired` (it depends on how many of
// the construction's triggers found the ring full under h): no oracle reads it.
// Why apart from the test above: at R = 1 every retirement before an explicit call
// triggers and empties the lists, so that call's hook would never fire; the
// walk's crossings need no hook.
// Kills: a trigger that ignores the flag (auto_reclaims +4); a skipped
// trigger counted as `contended` (contended +4; the destructor's R = 1
// invariant too).
TEST(ConcurrentHashSetRcuTest, CadenceWalkCrossingsAreSkipped) {
    using Set = CadProbeSet<64>;
    ProbeKey::reset_counters();
    {
        Set set = cadenced_set<Set>(4, 1, 1);
        Set::InternalAccounting left{};
        {
            const Set::handle h = set.new_handle();
            ASSERT_NO_FATAL_FAILURE(leave_five_stragglers(set, h, left));
        }
        const Set::InternalCounters c0 = set.get_internal_counters();
        const AdvanceResult r = set.advance();
        const Set::InternalCounters c1 = set.get_internal_counters();
        EXPECT_EQ(r.walk_retired, 5u) << ::testing::PrintToString(r);
        EXPECT_EQ(c1.retire_runs - c0.retire_runs, 4u) << "the walk's four runs";
        EXPECT_EQ(c1.auto_skipped - c0.auto_skipped, 4u) << "every crossing of the walk's own pushes must be skipped";
        EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 0u) << "a trigger proceeded inside advance()'s walk";
        EXPECT_EQ(c1.contended - c0.contended, 0u) << "a skipped trigger counted as contended";
        EXPECT_EQ(expect_consistent(set, "after the walk").reachable_dead, 0u);
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CadenceWalkCrossingsAreSkipped

// Inside a TRIGGER's hook: a nested advance() and reclaim() are
// `contended`, a nested crossing is skipped, and a handle created in the
// action joins the NEW generation (the hook follows the publish). R = 1.
//   step 0: insert 0..3; (4,0,0,0,0,0,0).
//   step 1: arm after_publish with: under h2 created in the action (its
//           generation recorded: 1), erase 2 (crosses: skipped); nested
//           advance() and reclaim(). Then { h; erase(h, 1); }: the erase's
//           trigger publishes (generation 1) and fires the hook inside the
//           test thread's own erase; the action touches neither h nor
//           bucket 1. Expect: the action ran once; gen_in_hook 1; both nested
//           calls kContended; bumps +1, auto_reclaims +1, auto_skipped +1,
//           contended +2, the hook +1; (2,0,1,1,0,0,0): node 1 in bag 0 (h
//           pinned it during the trigger, so its final free pass kept it),
//           node 2 on generation 1's retired list.
// Kills: a trigger's publish that does not fire the hook (the action never
// runs); a trigger that ignores the flag (the nested trigger proceeds inside
// the outer one: auto_reclaims +2, bumps +2, then the nested calls proceed);
// a skipped trigger counted as `contended`; a skipped trigger that fires the
// hook (the hook +2); a reclaim() or an advance() that ignores the flag; the
// trigger before the retire push (R = 1: the outer trigger finds the lists
// empty, no publish, the action never runs); the bump's exchanges moved after
// the hook, through a trigger (node 2 is bagged with node 1:
// (2,0,0,2,0,0,0)).
TEST_F(HashRcuHookTest, CadenceTriggerHookNestedCallsContended) {
    using Set = CadHookSet<64>;
    Set set = cadenced_set<Set>(256, 1, 1);
    {
        const Set::handle hs = set.new_handle();
        for (int k = 0; k < 4; ++k) ASSERT_TRUE(set.insert(hs, k));
    }
    expect_tuple(set, "step 0", 4, 0, 0, 0, 0, 0, 0);
    int ran = 0;
    uint64_t gen_in_hook = ~uint64_t{0};
    AdvanceResult na{}, nr{};
    TestHooks::arm(TestHooks::after_publish, [&]() {
        ++ran;
        {
            const Set::handle h2 = set.new_handle();
            gen_in_hook = Set::get_internal_handle_generation(h2);
            EXPECT_TRUE(set.erase(h2, 2)) << "the nested erase";
        }
        na = set.advance();
        nr = set.reclaim();
    });
    const Set::InternalCounters c0 = set.get_internal_counters();
    const long p0 = TestHooks::after_publish.calls.load();
    {
        const Set::handle h = set.new_handle();
        EXPECT_TRUE(set.erase(h, 1));
    }
    const Set::InternalCounters c1 = set.get_internal_counters();
    EXPECT_EQ(ran, 1) << "the trigger's publish did not fire the hook";
    EXPECT_EQ(gen_in_hook, 1u) << "a handle created in the hook did not join the published generation";
    EXPECT_EQ(na, kContended) << "a nested advance() inside a trigger's hook";
    EXPECT_EQ(nr, kContended) << "a nested reclaim() inside a trigger's hook";
    EXPECT_EQ(c1.bumps - c0.bumps, 1u);
    EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 1u);
    EXPECT_EQ(c1.auto_skipped - c0.auto_skipped, 1u) << "the nested crossing was not skipped";
    EXPECT_EQ(c1.contended - c0.contended, 2u);
    EXPECT_EQ(TestHooks::after_publish.calls.load() - p0, 1);
    expect_tuple(set, "after the erase", 2, 0, 1, 1, 0, 0, 0);
} // CadenceTriggerHookNestedCallsContended

// Under the eraser's own never-released handle a trigger frees only the
// bags OLDER than it. R = 1.
//   step 0: insert 0, 1, 2.
//   step 1: { h0; erase(h0, 0); }: the trigger bags {0} and frees nothing (h0
//           pins it); bags_freed +0.
//   step 2: h1 = new_handle() (generation 1, asserted); erase(h1, 1): the
//           trigger frees bag 0 (h0 is gone): bags_freed +1, nodes_freed +1
//           (K = 2: through the ring-wrap pass).
//   step 3: erase(h1, 2): bags_freed +0 (bag 1 is h1's); bumps +1 at K >= 3,
//           +0 at K = 2 (`ring_full`: slot 1 pinned by h1).
//   step 4: release h1; reclaim(): K >= 3 {2, 2, 0, 0, 0, nothing_retired};
//           K = 2 {2, 2, 0, 0, 0, advanced} (the in-wrap pass frees bag 1, the
//           bump bags node 2, the final pass frees it). Compared WHOLE.
// h0 is released between the erases, outside any operation.
// Kills: a trigger whose free pass takes its own operation's pin for none,
// and a free pass that ignores the pins (step 1: bags_freed +1); a free pass
// that skips a pinned bag and frees younger ones, at K >= 3 (step 3: bag 1 is
// pinned by h1, the defect frees the younger bag 2: nodes_freed +1);
// auto_reclaims counting only the triggers that published (K = 2: step 3's
// `ring_full` trigger uncounted; the destructor's R = 1 invariant too).
template <size_t K>
static void cadence_trigger_frees_only_older_bags_body() {
    using Set = CadHookSet<K>;
    Set set = cadenced_set<Set>(256, 1, 1);
    {
        const typename Set::handle hs = set.new_handle();
        for (int k = 0; k < 3; ++k) ASSERT_TRUE(set.insert(hs, k));
    }
    typename Set::InternalCounters c = set.get_internal_counters();
    {
        const typename Set::handle h0 = set.new_handle();
        EXPECT_TRUE(set.erase(h0, 0));
    }
    EXPECT_EQ(set.get_internal_counters().bags_freed - c.bags_freed, 0u) << "step 1: the trigger freed the bag its own handle pins";
    {
        const typename Set::handle h1 = set.new_handle();
        ASSERT_EQ(Set::get_internal_handle_generation(h1), 1u) << "test precondition: the first trigger did not publish";
        c = set.get_internal_counters();
        EXPECT_TRUE(set.erase(h1, 1));
        EXPECT_EQ(set.get_internal_counters().bags_freed - c.bags_freed, 1u) << "step 2: bag 0, unpinned, not freed";
        EXPECT_EQ(set.get_internal_counters().nodes_freed - c.nodes_freed, 1u) << "step 2";
        c = set.get_internal_counters();
        EXPECT_TRUE(set.erase(h1, 2));
        EXPECT_EQ(set.get_internal_counters().bags_freed - c.bags_freed, 0u) << "step 3: a bag at or after h1's generation was freed";
        EXPECT_EQ(set.get_internal_counters().nodes_freed - c.nodes_freed, 0u) << "step 3";
        EXPECT_EQ(set.get_internal_counters().bumps - c.bumps, K >= 3 ? 1u : 0u) << "step 3";
        EXPECT_EQ(set.get_internal_counters().ring_full - c.ring_full, K >= 3 ? 0u : 1u) << "step 3";
        EXPECT_EQ(set.get_internal_counters().auto_reclaims - c.auto_reclaims, 1u) << "step 3: the trigger acquired the flag and counts, whatever its bump decided";
    } // h1's scope
    if constexpr (K >= 3) {
        EXPECT_EQ(set.reclaim(), (AdvanceResult{2, 2, 0, 0, 0, Advance::nothing_retired})) << "step 4";
    } else {
        EXPECT_EQ(set.reclaim(), (AdvanceResult{2, 2, 0, 0, 0, Advance::advanced})) << "step 4, K = 2";
    }
} // cadence_trigger_frees_only_older_bags_body()
TEST_F(HashRcuHookTest, CadenceTriggerFreesOnlyOlderBagsK2) { ASSERT_NO_FATAL_FAILURE(cadence_trigger_frees_only_older_bags_body<2>()); }
TEST_F(HashRcuHookTest, CadenceTriggerFreesOnlyOlderBagsK3) { ASSERT_NO_FATAL_FAILURE(cadence_trigger_frees_only_older_bags_body<3>()); }
TEST_F(HashRcuHookTest, CadenceTriggerFreesOnlyOlderBagsK64) { ASSERT_NO_FATAL_FAILURE(cadence_trigger_frees_only_older_bags_body<64>()); }

// Triggers never walk, and neither does reclaim(): the stragglers survive
// both; advance() collects all. R = 1. leave_five_stragglers() under h (five
// dead nodes reachable), h released; under h2, erase the fillers 256*64 and
// 256*65 (bucket 0: no straggler there, and no split cleans bucket 0 at size
// 256): auto_reclaims +2, bumps +2 (h released first, so both proceed and
// publish: with h still live the construction's triggers could have left the
// ring full, and a `ring_full` trigger would make "the stragglers survived a
// trigger" vacuous), joins +0; h2 released; reachable_dead still 5. Then
// reclaim(): walk counts 0, joins +0, reachable_dead still 5. Then advance():
// walk_retired 5. No writer walk passes buckets 1, 2, 3 between the
// construction and the advance() (a step_over() would collect a straggler).
// Kills: a trigger that also walks (the first trigger collects the
// stragglers: reachable_dead 0, joins +1); a reclaim() that walks (it collects
// them: reachable_dead 0, joins +1, walk counts > 0).
TEST(ConcurrentHashSetRcuTest, CadenceTriggerLeavesStragglers) {
    using Set = CadProbeSet<64>;
    ProbeKey::reset_counters();
    {
        Set set = cadenced_set<Set>(4, 1, 1);
        Set::InternalAccounting left{};
        {
            const Set::handle h = set.new_handle();
            ASSERT_NO_FATAL_FAILURE(leave_five_stragglers(set, h, left));
        }
        {
            const Set::handle h2 = set.new_handle();
            const Set::InternalCounters c0 = set.get_internal_counters();
            EXPECT_TRUE(set.erase(h2, 256*64));
            EXPECT_TRUE(set.erase(h2, 256*65));
            const Set::InternalCounters c1 = set.get_internal_counters();
            EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 2u);
            EXPECT_EQ(c1.bumps - c0.bumps, 2u) << "test precondition: both triggers must publish, or surviving them proves nothing";
            EXPECT_EQ(c1.joins - c0.joins, 0u) << "a trigger walked";
        } // h2's scope
        EXPECT_EQ(expect_consistent(set, "after the triggers").reachable_dead, 5u) << "a trigger collected a straggler";
        const Set::InternalCounters c2 = set.get_internal_counters();
        const AdvanceResult rr = set.reclaim();
        EXPECT_EQ(rr.walk_retired + rr.buckets_walked + rr.nodes_passed, 0u) << "reclaim() walked: " << ::testing::PrintToString(rr);
        EXPECT_EQ(set.get_internal_counters().joins - c2.joins, 0u) << "reclaim() walked";
        EXPECT_EQ(expect_consistent(set, "after reclaim()").reachable_dead, 5u) << "reclaim() collected a straggler";
        EXPECT_EQ(set.advance().walk_retired, 5u) << "advance()'s walk must collect the five";
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CadenceTriggerLeavesStragglers

// An operation that throws AFTER its own trigger ran leaves the trigger's
// effects standing (the trigger's result is not "lost": nothing reports it).
// R = 1, CadProbeSet (NoHooks: a throwing ProbeKey hook may be armed).
// leave_five_stragglers() under h, h released (with h live, the construction's
// triggers could have filled the ring, and the measured trigger would be
// `ring_full`). Bucket 1 then holds the dead two-node run 385 -> 129 at its
// head, then the live 257. Under h2, insert(513) (513 & 255 == 1, absent):
// its walk unlinks the head run (ONE retire push of two nodes, which crosses
// R = 1: the trigger proceeds INSIDE the insert and publishes), then compares
// 257, where equal_hook throws. The insert allocates its node only after the
// walk, so nothing goes to limbo.
// Expect: the exception reaches the caller; retire_runs +1, retire_nodes +2,
// auto_reclaims +1, bumps +1, limbo_pushes +0; the sweep consistent, with the
// other three stragglers still reachable (3) and none of the run; 513 absent and
// 257 present; the retried insert(513) succeeds; the destructor's cadence
// invariants hold.
// Why: the trigger is the only reclamation that runs inside an operation whose
// later steps (Hash{}, operator==) may throw; this checks that unwinding
// through the operation neither undoes nor repeats the trigger's bump and
// leaves no node unaccounted for.
// Kills (measured; each also killed elsewhere): the trigger placed before the
// retire push (bumps +0); a free pass that ignores the pins, a trigger whose free
// pass takes refs <= 1 for unpinned, and a retire push that writes the list's
// next pointer into the chain link (the sweep after the throw is inconsistent);
// a flag never released (the sweep's flag assert).
TEST(ConcurrentHashSetRcuTest, CadenceThrowAfterTriggerKeepsItsEffects) {
    using Set = CadProbeSet<64>;
    ProbeKey::reset_counters();
    {
        Set set = cadenced_set<Set>(4, 1, 1);
        Set::InternalAccounting left{};
        {
            const Set::handle h = set.new_handle();
            ASSERT_NO_FATAL_FAILURE(leave_five_stragglers(set, h, left));
        } // the construction's handle, released so that the measured trigger can publish
        const Set::handle h2 = set.new_handle();
        const Set::InternalCounters c0 = set.get_internal_counters();
        int fired = 0;
        ProbeKey::arm(ProbeKey::equal_hook, 257, 1, [&fired]() {
            ++fired;
            throw std::runtime_error("operator== armed to throw after the trigger");
        });
        EXPECT_THROW(set.insert(h2, 513), std::runtime_error);
        ProbeKey::disarm_hooks();
        const Set::InternalCounters c1 = set.get_internal_counters();
        ASSERT_EQ(fired, 1) << "test precondition: the insert did not reach 257";
        ASSERT_EQ(c1.retire_runs - c0.retire_runs, 1u) << "test precondition: the walk did not unlink the head run as one push";
        EXPECT_EQ(c1.retire_nodes - c0.retire_nodes, 2u);
        EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 1u) << "the trigger did not proceed inside the insert";
        EXPECT_EQ(c1.bumps - c0.bumps, 1u) << "the trigger's publish does not stand";
        EXPECT_EQ(c1.limbo_pushes - c0.limbo_pushes, 0u);
        EXPECT_EQ(expect_consistent(set, "after the throw").reachable_dead, 3u) << "the head run left bucket 1; the other three stragglers stay";
        EXPECT_FALSE(set.contains(h2, 513));
        EXPECT_TRUE(set.contains(h2, 257));
        EXPECT_TRUE(set.insert(h2, 513)) << "the retried insert";
        EXPECT_TRUE(set.contains(h2, 513));
    } // the set's lifetime: the destructor's cadence invariants run here
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CadenceThrowAfterTriggerKeepsItsEffects

// The cadence's decoding, read back through
// get_internal_auto_reclaim_every(), the effective R. Both cadence-off forms
// read 0; 3 reads 4 and 5 reads 8 (rounded UP to a power of two); 2^63,
// 2^62 + 1, 2^63 + 1 and SIZE_MAX all read 2^63 (clamped to 2^63 before the
// rounding) and fire nothing across 3 inserts and 3 erases. Then R = 3 (= 4)
// from a fresh shard: after erase n (1-based) the attempts are n/4, so the
// 4th and the 8th fire and the 3rd does not.
// Kills: the cadence 0 rounded like any other value instead of decided first
// (it reads 1: bit_ceil(0) is 1); the cadence rounded DOWN (3 reads 2 and
// fires at the 2nd erase); a cadence above 2^63 taken as off (2^63 + 1 and
// SIZE_MAX read 0) and a clamp below 2^63 (every large value reads 2^62); no
// clamp at all, only where std::bit_ceil() checks its precondition (libstdc++'s assertions, on in these -O0 sanitizer
// builds: bit_ceil(2^63 + 1) aborts; an optimized build returns 1, a cadence
// of 1, and has no killer); the crossing decided one push late (R = 3 fires
// at the 1st erase; at 2^63 a fresh shard's first push fires).
TEST(ConcurrentHashSetRcuTest, CadenceRoundingAndClamp) {
    using Set = IntSetDel;
    EXPECT_EQ(Set(256, 1).get_internal_auto_reclaim_every(), 0u) << "the two-parameter form";
    EXPECT_EQ(cadenced_set<Set>(256, 1, 0).get_internal_auto_reclaim_every(), 0u) << "the cadence 0";
    EXPECT_EQ(cadenced_set<Set>(256, 1, 3).get_internal_auto_reclaim_every(), 4u) << "3 rounds up";
    EXPECT_EQ(cadenced_set<Set>(256, 1, 5).get_internal_auto_reclaim_every(), 8u) << "5 rounds up";
    for (size_t R : {size_t{1} << 63, (size_t{1} << 62) + 1, (size_t{1} << 63) + 1, SIZE_MAX}) {
        Set set = cadenced_set<Set>(256, 1, R);
        EXPECT_EQ(set.get_internal_auto_reclaim_every(), size_t{1} << 63) << "cadence " << R;
        const Set::handle h = set.new_handle();
        for (int k = 0; k < 3; ++k) ASSERT_TRUE(set.insert(h, k));
        for (int k = 0; k < 3; ++k) ASSERT_TRUE(set.erase(h, k));
        EXPECT_EQ(cadence_attempts(set), 0u) << "cadence " << R << " fired within three erases";
    } // the clamped cadences
    Set set = cadenced_set<Set>(256, 1, 3);
    const Set::handle h = set.new_handle();
    for (int k = 0; k < 8; ++k) ASSERT_TRUE(set.insert(h, k));
    for (int k = 0; k < 8; ++k) {
        ASSERT_TRUE(set.erase(h, k));
        EXPECT_EQ(cadence_attempts(set), static_cast<size_t>((k + 1)/4)) << "after erase " << k + 1 << " at the cadence 3 (= 4)";
    } // erase 8
} // CadenceRoundingAndClamp

// A multi-node run that jumps a multiple fires once; the count is in
// NODES. R = 2. Under h, leave_five_stragglers() (its construction retires
// runs of unknown sizes); if retire_nodes is even, one filler erase (a
// one-node run) makes it ODD (asserted: the parity step is what makes both
// kills independent of the construction's counts). Then erase(h, 1), an
// ABSENT key of bucket 1, whose miss walk unlinks the two-node run at bucket
// 1's head (385 -> 129) as ONE push (retire_runs +1, retire_nodes +2,
// asserted): (odd, odd + 2] holds a multiple of 2, so it fires once. Then one
// more filler erase: from odd + 2 (odd) it crosses again and fires.
// Kills, ONLY here: the crossing decided by "the new count is a multiple"
// (the two-node push lands on an odd count: no fire); the cadence counted in
// runs, not nodes (the run count crosses at most one of the two pushes: if it
// is even before the two-node push, that push does not fire; if it is odd, it
// does, and then the next one-node push does not).
TEST(ConcurrentHashSetRcuTest, CadenceMultiNodeRunCrossesOnce) {
    using Set = CadProbeSet<64>;
    ProbeKey::reset_counters();
    {
        Set set = cadenced_set<Set>(4, 1, 2);
        const Set::handle h = set.new_handle();
        Set::InternalAccounting left{};
        ASSERT_NO_FATAL_FAILURE(leave_five_stragglers(set, h, left));
        if (set.get_internal_counters().retire_nodes % 2 == 0) {   // the parity step: one one-node run
            ASSERT_TRUE(set.erase(h, 256*64));
        }
        const Set::InternalCounters c0 = set.get_internal_counters();
        ASSERT_EQ(c0.retire_nodes % 2, 1u) << "test precondition: the count before the two-node run is odd";
        const size_t a0 = cadence_attempts(set);
        EXPECT_FALSE(set.erase(h, 1)) << "key 1 is absent: a miss walk of bucket 1";
        const Set::InternalCounters c1 = set.get_internal_counters();
        ASSERT_EQ(c1.retire_runs - c0.retire_runs, 1u) << "test precondition: the miss walk retired other than one run";
        ASSERT_EQ(c1.retire_nodes - c0.retire_nodes, 2u) << "test precondition: the run at bucket 1's head is not two nodes";
        EXPECT_EQ(cadence_attempts(set) - a0, 1u) << "a two-node run from an odd count crosses a multiple of 2";
        const size_t a1 = cadence_attempts(set);
        EXPECT_TRUE(set.erase(h, 256*65));
        EXPECT_EQ(cadence_attempts(set) - a1, 1u) << "the next one-node push from an odd count crosses";
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CadenceMultiNodeRunCrossesOnce

// The ring wraps under a never-refreshed eraser: R = 1, K + 2 keys,
// under one handle h. Erases 0 .. K-2: each publishes (bumps +1) and frees
// nothing (h pins every bag). Erases K-1 and K: the ring is full (the next
// slot is h's pinned generation 0): `ring_full` +1, auto_reclaims +1 (it
// acquired the flag), bumps +0 each. Then (1, 0, 2, K-1, 0, 0, 1): key K+1
// reachable, the two last nodes retired, K-1 bagged. h.refresh() between
// operations, then erase K+1: bumps +1, and the free pass frees the K-1 bags
// of one node each (generations 0 .. K-2: h, refreshed into generation K-1,
// no longer pins them); (0, 0, 0, 3, K-1, 0, 1): the three newest nodes in
// the bag of generation K-1, which h now pins.
// Kills: auto_reclaims counting only the triggers that published (+0 at
// `ring_full`; the destructor's R = 1 invariant too).
template <size_t K>
static void cadence_ring_wrap_body() {
    using Set = CadHookSet<K>;
    Set set = cadenced_set<Set>(256, 1, 1);
    const int k_max = static_cast<int>(K);
    {
        const typename Set::handle hs = set.new_handle();
        for (int k = 0; k < k_max + 2; ++k) ASSERT_TRUE(set.insert(hs, k));
    }
    typename Set::handle h = set.new_handle();
    for (int k = 0; k + 1 < k_max; ++k) {
        const typename Set::InternalCounters c = set.get_internal_counters();
        ASSERT_TRUE(set.erase(h, k));
        const typename Set::InternalCounters d = set.get_internal_counters();
        EXPECT_EQ(d.bumps - c.bumps, 1u) << "erase " << k;
        EXPECT_EQ(d.bags_freed - c.bags_freed, 0u) << "erase " << k << ": a bag the never-refreshed handle pins was freed";
    } // erases 0 .. K-2
    for (int k = k_max - 1; k <= k_max; ++k) {
        const typename Set::InternalCounters c = set.get_internal_counters();
        ASSERT_TRUE(set.erase(h, k));
        const typename Set::InternalCounters d = set.get_internal_counters();
        EXPECT_EQ(d.ring_full - c.ring_full, 1u) << "erase " << k;
        EXPECT_EQ(d.auto_reclaims - c.auto_reclaims, 1u) << "erase " << k << ": a ring_full trigger acquired the flag and must count";
        EXPECT_EQ(d.bumps - c.bumps, 0u) << "erase " << k;
    } // erases K-1 and K: the ring is full
    expect_tuple(set, "the ring full", 1, 0, 2, K - 1, 0, 0, 1);
    h.refresh();
    const typename Set::InternalCounters c = set.get_internal_counters();
    ASSERT_TRUE(set.erase(h, k_max + 1));
    const typename Set::InternalCounters d = set.get_internal_counters();
    EXPECT_EQ(d.bumps - c.bumps, 1u);
    EXPECT_EQ(d.bags_freed - c.bags_freed, K - 1);
    EXPECT_EQ(d.nodes_freed - c.nodes_freed, K - 1);
    expect_tuple(set, "after the refresh", 0, 0, 0, 3, K - 1, 0, 1);
} // cadence_ring_wrap_body()
TEST_F(HashRcuHookTest, CadenceRingWrapUnderNeverRefreshedHandleK2) { ASSERT_NO_FATAL_FAILURE(cadence_ring_wrap_body<2>()); }
TEST_F(HashRcuHookTest, CadenceRingWrapUnderNeverRefreshedHandleK3) { ASSERT_NO_FATAL_FAILURE(cadence_ring_wrap_body<3>()); }
TEST_F(HashRcuHookTest, CadenceRingWrapUnderNeverRefreshedHandleK64) { ASSERT_NO_FATAL_FAILURE(cadence_ring_wrap_body<64>()); }

// Limbo pushes do not count toward the cadence (a limbo-only state is bagged
// by the next bump of any cause). R = 1.
// { h; insert 1; erase 1; } (bag 0, pinned by h during its own trigger);
// reclaim() == {1, 1, 0, 0, 0, nothing_retired} (one free node now); under
// h2, the next insert pops that node and its copy assignment throws (ProbeKey's
// throw_countdown, on a NoHooks set: no TestHooks action can be reached), so
// the node goes to limbo: limbo_pushes +1, retire_nodes +0, attempts +0.
// Kills: limbo pushes counted toward the cadence (attempts +1 and
// retire_nodes +1; the destructor's R = 1 invariant too).
TEST(ConcurrentHashSetRcuTest, CadenceLimboDoesNotCount) {
    using Set = CadProbeSet<64>;
    ProbeKey::reset_counters();
    {
        Set set = cadenced_set<Set>(1024, 1, 1);
        {
            const Set::handle h = set.new_handle();
            ASSERT_TRUE(set.insert(h, 1));
            ASSERT_TRUE(set.erase(h, 1));
        }
        EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::nothing_retired})) << "free the one node";
        const Set::handle h2 = set.new_handle();
        const Set::InternalCounters c0 = set.get_internal_counters();
        ProbeKey::throw_countdown = 1;
        EXPECT_THROW(set.insert(h2, 2), std::runtime_error) << "the popped node's assignment must throw";
        ProbeKey::throw_countdown = 0;
        const Set::InternalCounters c1 = set.get_internal_counters();
        EXPECT_EQ(c1.limbo_pushes - c0.limbo_pushes, 1u) << "test precondition: the thrown-on node did not go to limbo";
        EXPECT_EQ(c1.retire_nodes - c0.retire_nodes, 0u) << "a limbo push counted as retired nodes";
        EXPECT_EQ(c1.auto_reclaims + c1.auto_skipped, c0.auto_reclaims + c0.auto_skipped) << "a limbo push triggered";
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // CadenceLimboDoesNotCount

// The count is PER SHARD, not set-wide. R = 2, TWO arena shards. Inserts
// 0, 1, 2 on the test thread (inserts retire nothing). Two helper threads,
// each started and JOINED before the next starts, each erasing one key under
// its own handle: the header numbers threads sequentially on first use, so
// the two get consecutive numbers and therefore distinct shards (asserted
// through the per-shard counts {1, 1}: the script's precondition, not an
// oracle); each shard is at 1 = R-1: no attempt. A third helper erases 2: its
// number is the first helper's plus 2, the same shard, which reaches 2 = R:
// one attempt (the positive control).
// Kills, ONLY here: the crossing decided on the set-wide count (the second
// helper's push takes the set-wide sum to 2: one attempt after two helpers).
// Not a contract test: it relies on the sequential thread numbering, which
// the contract does not promise.
TEST(ConcurrentHashSetRcuTest, CadenceCountsPerShard) {
    using Set = IntSetDel;
    Set set = cadenced_set<Set>(256, 2, 2);
    {
        const Set::handle h = set.new_handle();
        for (int k = 0; k < 3; ++k) ASSERT_TRUE(set.insert(h, k));
    }
    // One erase of `k` on a fresh thread, under a handle of its own, joined
    // before this returns.
    auto erase_on_new_thread = [&set](int k) {
        std::thread([&set, k]() {
            const Set::handle h = set.new_handle();
            EXPECT_TRUE(set.erase(h, k)) << "helper's erase of " << k;
        }).join();
    }; // erase_on_new_thread()
    erase_on_new_thread(0);
    erase_on_new_thread(1);
    ASSERT_EQ(set.get_internal_retire_nodes_per_shard(), (std::vector<size_t>{1, 1}))
        << "test precondition: the two helpers did not retire on distinct shards";
    EXPECT_EQ(cadence_attempts(set), 0u) << "the cadence counted across shards: each shard is at R - 1";
    erase_on_new_thread(2);
    EXPECT_EQ(cadence_attempts(set), 1u) << "positive control: a shard that reaches R fires";
} // CadenceCountsPerShard

// The suspended old-generation reader under TRIGGERS:
// SuspendedReaderPinsItsGeneration with every explicit call replaced by a
// nested erase whose trigger does the work. R = 1, ProbeKey, NoHooks. Keys KEY = 3,
// S = 259, X = 515 in bucket 3 (chain X -> S -> KEY, newest first), fillers
// Y = 4, Z = 5.
//   step 0: insert KEY, S, X, Y, Z: (5, 0, 0, 0, 0, 0, 0).
//   step 1: h = new_handle() (generation 0: the PIN under test); arm
//           equal_hook on X with A = { under h1: erase Y } { under h2: erase S
//           } { under h3: erase Z }, then arm equal_hook on S to count the
//           step onto S; contains(h, KEY). The correct timeline: h1 (gen 0)
//           erases Y, its trigger bags Y in gen 0 and frees nothing (h, h1);
//           h2 joins gen 1, unlinks S through X's link (X live), its trigger
//           bags S in gen 1 and frees nothing (bag 0 pinned by h; bag 1 by h2
//           and cumulatively); h3 joins gen 2, erases Z, its trigger bags Z in
//           gen 2 (K >= 4) or finds the ring full (K = 3; at K = 2 also h2's),
//           frees nothing. The reader resumes onto S (dead, bagged, alive),
//           compares it (the chained hook) and finds KEY.
//           Expect: the action ran once, the step onto S once, found;
//           auto_reclaims +3, bags_freed +0; K >= 4: bumps +3,
//           (2, 0, 0, 3, 0, 0, 1); K = 3: bumps +2, ring_full +1,
//           (2, 0, 1, 2, 0, 0, 1); K = 2: bumps +1, ring_full +2,
//           (2, 0, 2, 1, 0, 0, 1).
//   step 2: release h; reclaim(): K >= 4 {3, 3, 0, 0, 0, nothing_retired};
//           K = 3 {3, 3, 0, 0, 0, advanced}; K = 2 {3, 2, 0, 0, 0, advanced}.
// Three nested erases and two fillers: with only Y and S, a free pass that
// skips a pinned bag and frees younger ones would find
// bag 1 pinned by h2 at the only trigger after S's erase and free nothing.
// h1, h2, h3 are each released before the next is created, so each joins the
// generation its predecessor's trigger published. EXPECT_* only inside A.
// Kills: a trigger whose free pass takes one pin for its own operation's (at
// h2's trigger bag 0's one pin is the suspended reader's, ANOTHER operation of
// the same thread: the defect frees bag 0 and bag 1 (refs 1, h2's own), and
// the reader steps onto a freed S: ASan's use-after-poison in
// ProbeKey::operator==, the required symptom; under TSan S's link leads into
// the free list, a wrong miss); a free pass that ignores the pins (the same
// use-after-poison); a free pass that skips a pinned bag and frees younger
// ones, at K >= 3 (h3's trigger, or its in-wrap pass at K = 3, skips the
// pinned bag 0 and frees bag 1 {S}). K = 2 cannot kill the last (no younger
// bag past a pinned one), as in SuspendedReaderPinsItsGeneration.
template <size_t K>
static void cadence_suspended_reader_body() {
    using Set = CadProbeSet<K>;
    const int KEY = 3, S = 259, X = 515, Y = 4, Z = 5;
    {
        Set set = cadenced_set<Set>(256, 1, 1);
        {
            const typename Set::handle hs = set.new_handle();
            for (int k : {KEY, S, X, Y, Z}) ASSERT_TRUE(set.insert(hs, k));
        }
        expect_tuple(set, "step 0", 5, 0, 0, 0, 0, 0, 0);
        int ran = 0, stepped_on_s = 0;
        typename Set::handle h = set.new_handle();
        const typename Set::InternalCounters c0 = set.get_internal_counters();
        ProbeKey::arm(ProbeKey::equal_hook, X, 1, [&]() {
            ++ran;
            {
                const typename Set::handle h1 = set.new_handle();
                EXPECT_TRUE(set.erase(h1, Y));
            }
            {
                const typename Set::handle h2 = set.new_handle();
                EXPECT_TRUE(set.erase(h2, S));
            }
            {
                const typename Set::handle h3 = set.new_handle();
                EXPECT_TRUE(set.erase(h3, Z));
            }
            ProbeKey::arm(ProbeKey::equal_hook, S, 1, [&]() { ++stepped_on_s; });
        });
        const bool found = set.contains(h, KEY);
        ProbeKey::disarm_hooks();
        const typename Set::InternalCounters c1 = set.get_internal_counters();
        ASSERT_EQ(ran, 1) << "test precondition: the reader was not suspended at X";
        EXPECT_EQ(stepped_on_s, 1) << "the reader did not step onto S";
        EXPECT_TRUE(found) << "the suspended reader lost KEY";
        EXPECT_EQ(c1.auto_reclaims - c0.auto_reclaims, 3u);
        EXPECT_EQ(c1.bags_freed - c0.bags_freed, 0u) << "a trigger freed a bag the suspended reader pins";
        if constexpr (K >= 4) {
            EXPECT_EQ(c1.bumps - c0.bumps, 3u);
            expect_tuple(set, "step 1", 2, 0, 0, 3, 0, 0, 1);
        } else if constexpr (K == 3) {
            EXPECT_EQ(c1.bumps - c0.bumps, 2u);
            EXPECT_EQ(c1.ring_full - c0.ring_full, 1u);
            expect_tuple(set, "step 1", 2, 0, 1, 2, 0, 0, 1);
        } else {
            EXPECT_EQ(c1.bumps - c0.bumps, 1u);
            EXPECT_EQ(c1.ring_full - c0.ring_full, 2u);
            expect_tuple(set, "step 1", 2, 0, 2, 1, 0, 0, 1);
        }
        h = {};
        if constexpr (K >= 4) {
            EXPECT_EQ(set.reclaim(), (AdvanceResult{3, 3, 0, 0, 0, Advance::nothing_retired})) << "step 2";
        } else if constexpr (K == 3) {
            EXPECT_EQ(set.reclaim(), (AdvanceResult{3, 3, 0, 0, 0, Advance::advanced})) << "step 2";
        } else {
            EXPECT_EQ(set.reclaim(), (AdvanceResult{3, 2, 0, 0, 0, Advance::advanced})) << "step 2";
        }
    } // the set's lifetime
    EXPECT_EQ(ProbeKey::corrupt.load(), 0);
} // cadence_suspended_reader_body()
TEST_F(HashRcuHookTest, CadenceSuspendedReaderPinsItsGenerationK2) { ASSERT_NO_FATAL_FAILURE(cadence_suspended_reader_body<2>()); }
TEST_F(HashRcuHookTest, CadenceSuspendedReaderPinsItsGenerationK3) { ASSERT_NO_FATAL_FAILURE(cadence_suspended_reader_body<3>()); }
TEST_F(HashRcuHookTest, CadenceSuspendedReaderPinsItsGenerationK4) { ASSERT_NO_FATAL_FAILURE(cadence_suspended_reader_body<4>()); }
TEST_F(HashRcuHookTest, CadenceSuspendedReaderPinsItsGenerationK64) { ASSERT_NO_FATAL_FAILURE(cadence_suspended_reader_body<64>()); }

// The cadence 0 IS the two-parameter form, with erases and literal
// results: on each, the effective cadence reads 0; { h; insert 1; erase 1; },
// then the first reclaim() bags and frees node 1 in the same call (no handle
// is alive) -- {1, 1, 0, 0, 0, advanced} -- the second finds nothing --
// {0, 0, 0, 0, 0, nothing_retired} --, and no trigger attempt was made.
// Kills: the cadence 0 rounded to 1 (the erase's own trigger bags node 1
// under h, so the first reclaim() is {1, 1, 0, 0, 0, nothing_retired}, the
// accessor reads 1, and attempts is 1).
template <typename SetT>
static void cadence_zero_body(SetT& set, const std::string& where) {
    EXPECT_EQ(set.get_internal_auto_reclaim_every(), 0u) << where;
    {
        const typename SetT::handle h = set.new_handle();
        ASSERT_TRUE(set.insert(h, 1)) << where;
        ASSERT_TRUE(set.erase(h, 1)) << where;
    }
    EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::advanced})) << where << ": bagged and freed by the call";
    EXPECT_EQ(set.reclaim(), (AdvanceResult{0, 0, 0, 0, 0, Advance::nothing_retired})) << where << ": nothing left";
    EXPECT_EQ(cadence_attempts(set), 0u) << where;
} // cadence_zero_body()
TEST(ConcurrentHashSetRcuTest, CadenceZeroIsTheTwoParameterForm) {
    using Set = IntSetDel;
    Set by_parameters(256, 1);
    ASSERT_NO_FATAL_FAILURE(cadence_zero_body(by_parameters, "Set(256, 1)"));
    Set by_options = cadenced_set<Set>(256, 1, 0);
    ASSERT_NO_FATAL_FAILURE(cadence_zero_body(by_options, "Options{256, 1, 0}"));
} // CadenceZeroIsTheTwoParameterForm

// A cadenced set destroyed with pending bags: the destructor's !NDEBUG
// cadence invariants (no attempt with the cadence off; attempts at most
// retire_nodes / R; at R = 1 attempts == retire_runs) hold, and destruction
// frees the bagged and retired slots with the arena. R = 1 and R = 2, one
// shard, six keys erased under ONE never-refreshed handle, released before
// the destruction and after the last trigger: R = 1 makes six attempts that
// bag one node each (all pinned by h: six bags, nothing freed); R = 2 makes
// three, at the 2nd, 4th and 6th erase (three bags of two). Before the
// destruction: (0, 0, 0, 6, 0, 0, 0) -- every node bagged, none free -- and
// the counters' relation asserted here too (the destructor's form, so a
// failure names the test, not only the abort).
// Kills: the crossing decided one push late (R = 2: the attempts land on the
// 1st, 3rd and 5th erase, so node 6 stays retired: `retired` 1, `bagged` 5).
TEST(ConcurrentHashSetRcuTest, CadenceDestroyWithPendingBags) {
    using Set = IntSetDel;
    for (size_t R : {1, 2}) {
        const std::string where = "R = " + std::to_string(R);
        std::unique_ptr<Set> set = std::make_unique<Set>(Set::Options{.initial_capacity = 256, .arena_shards = 1, .auto_reclaim_every = R});
        {
            const Set::handle h = set->new_handle();
            for (int k = 0; k < 6; ++k) ASSERT_TRUE(set->insert(h, k)) << where;
            for (int k = 0; k < 6; ++k) ASSERT_TRUE(set->erase(h, k)) << where;
        } // h pinned every bag its own triggers closed
        const Set::InternalCounters c = set->get_internal_counters();
        EXPECT_EQ(c.auto_reclaims, 6/R) << where;
        EXPECT_EQ(c.auto_skipped, 0u) << where;
        EXPECT_EQ(c.bags_freed, 0u) << where;
        EXPECT_LE(c.auto_reclaims + c.auto_skipped, c.retire_nodes/R) << where;
        if (R == 1) {
            EXPECT_EQ(c.auto_reclaims + c.auto_skipped, c.retire_runs) << where;
        }
        expect_tuple(*set, where + ", before the destruction", 0, 0, 0, 6, 0, 0, 0);
        set.reset();   // the destructor's invariants run here
    } // loop over the cadences
} // CadenceDestroyWithPendingBags

#ifndef NDEBUG
// A ring view inside a TRIGGER's hook aborts (on a cadenced set the ring
// views need operation quiescence, since an operation's trigger holds the
// flag; asserted, with a message that names the trigger). R = 1, CadHookSet<64>: insert 1;
// arm after_publish with get_internal_generation(); { h; erase(h, 1); }: the
// erase's trigger publishes, the hook fires inside the erase, and the view
// asserts the flag clear. Required: SIGABRT with the assertion's message
// naming the trigger (the regex), so a crash or a sanitizer report cannot
// pass for the diagnosis. The legal twin in the parent: the same erase with
// the view called after it, between operations.
// Kills: a trigger's publish that does not fire the hook (no call, no death);
// the trigger before the retire push (no publish at R = 1, no hook, no
// death).
static void cadence_ring_view_in_trigger_hook(bool inside) {
    using Set = CadHookSet<64>;
    TestHooks::reset();
    {
        Set set = cadenced_set<Set>(256, 1, 1);
        {
            const Set::handle h = set.new_handle();
            (void)set.insert(h, 1);
        }
        if (inside) TestHooks::arm(TestHooks::after_publish, [&set]() { (void)set.get_internal_generation(); });
        {
            const Set::handle h = set.new_handle();
            (void)set.erase(h, 1);
        }
        (void)set.get_internal_generation();   // between operations: legal
    } // the set's lifetime
    TestHooks::reset();
} // cadence_ring_view_in_trigger_hook()

TEST(ConcurrentHashSetRcuDeathTest, CadenceRingViewInsideTriggerHookAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    cadence_ring_view_in_trigger_hook(false);
    EXPECT_EXIT(cadence_ring_view_in_trigger_hook(true), testing::KilledBySignal(SIGABRT),
                "during advance\\(\\), reclaim\\(\\) or an automatic trigger");
} // CadenceRingViewInsideTriggerHookAborts
#endif

// ---------------------------------------------------------------------------
// The cadence stresses: advance_stress_body() with StressShape::cadence > 0.
// Writers and readers refresh their long-lived handles every iteration (with a
// handle refreshed rarely, every trigger would find its own generation pinned
// and free nothing). Witnesses come from the counters at the quiescent point
// after the join, never from a ring view during the run (on a cadenced set an
// operation can hold the flag). The end state: every thread's handle
// released, then advance_and_check() (an oracle on a cadenced set only at a
// quiescent point) and the end-state oracle of advance_stress_body().
// One second each per sanitizer build.
// ---------------------------------------------------------------------------

// The cadence alone reclaims: NO explicit advancer, R = 1 (every retire push
// crosses), writers and readers refreshing per iteration, a runaway guard on
// the arena (it stops the run early rather than failing it; the INFO line
// says so). Witnesses: triggers proceeded and were skipped (two retiring
// threads met at the flag), bags and nodes were freed, and freed nodes were
// popped again (reuse: nodes_freed > 0 and a pop, counted exactly as the
// pushes minus what the free lists hold at the end; not "slots < inserts",
// which a large arena could satisfy without any reuse).
// Kills, probabilistically: a trigger that releases the flag before its final
// free pass (another trigger or call then bumps and frees concurrently with
// it: a race on the free pass's plain fields that TSan reports, or a double
// push the end state's accounting sees); any premature free the sanitizers or
// the exact booleans see.
TEST(ConcurrentHashSetRcuTest, CadenceStressNoAdvancer) {
    using Set = IntSetDel;
    StressShape shape{4, 2, 0, false, 2, 1024, 64, 256, std::chrono::milliseconds(1000), true};
    shape.cadence = 1;
    shape.slot_limit = size_t{1} << 20;
    StressWitness w;
    ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "cadence 1, no advancer", w));
    EXPECT_GT(w.auto_reclaims, 0u) << "witness: no trigger proceeded";
    EXPECT_GT(w.auto_skipped, 0u) << "witness: no two triggers met at the flag";
    EXPECT_GT(w.counted_bags_freed, 0u) << "witness: no bag was freed during the run";
    EXPECT_GT(w.nodes_freed, 0u) << "witness: no node was freed during the run";
    EXPECT_GT(w.pops, 0u) << "witness: no freed node was reused";
} // CadenceStressNoAdvancer

// The cadence beside ONE explicit advancer: its advance() calls meet the
// triggers at the flag and may return `contended`, which on a cadenced set is
// legal beside operations. R = 1. With a single explicit
// caller, every `contended` is a trigger's hold. Witnesses: the advancer was
// refused by a trigger (contended), and also proceeded (its walk ran beside
// the triggers); triggers proceeded and were skipped; bags were freed.
// Kills, probabilistically: a trigger that releases the flag before its final
// free pass (as above, with the advancer's walk and free pass as the second
// party); a trigger that ignores the flag (two flag holders, a race TSan
// reports).
TEST(ConcurrentHashSetRcuTest, CadenceStressBesideExplicitAdvancer) {
    using Set = IntSetDel;
    StressShape shape{4, 2, 1, false, 2, 1024, 64, 256, std::chrono::milliseconds(1000), true};
    shape.cadence = 1;
    shape.slot_limit = size_t{1} << 20;
    StressWitness w;
    ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "cadence 1, one explicit advancer", w));
    EXPECT_GT(w.contended, 0u) << "witness: the explicit advancer never met a trigger at the flag";
    EXPECT_GT(w.outcomes[static_cast<size_t>(Advance::advanced)] + w.outcomes[static_cast<size_t>(Advance::nothing_retired)], 0u)
        << "witness: the explicit advancer never proceeded";
    EXPECT_GT(w.auto_reclaims, 0u) << "witness: no trigger proceeded";
    EXPECT_GT(w.auto_skipped, 0u) << "witness: no trigger was skipped";
    EXPECT_GT(w.counted_bags_freed, 0u) << "witness: no bag was freed during the run";
} // CadenceStressBesideExplicitAdvancer

// A mixing hash for the tiny cadenced sets below: mix() (the MurmurHash3
// finalizer above) of the key. With the identity hash the stress's stable
// keys (low bits 0 .. stable-1) stop moving once the table outgrows them, so a
// reader's probe of one never lands on a pending child whose parent holds a
// mover; mixed, every doubling moves about half of every chain, the stable
// keys included, and the readers' probes of stable keys split children whose
// cleanups retire frozen movers.
struct CadenceMixHash {
    size_t operator()(int k) const { return static_cast<uint32_t>(mix(k)); }
}; // struct CadenceMixHash

// Many tiny cadenced sets: a fresh 4-bucket set per round (mixing hash, two
// shards, R = 1, no explicit advancer), so the table doubles inside every
// round while the threads run, and the readers' contains() land on
// UNINITIALIZED buckets, win their splits and run the splits' cleanups, whose
// retire pushes trigger INSIDE the contains() while writers insert,
// erase and split beside them. The end state is checked each round. Witness,
// over all rounds: a reader iteration across which the reader's own bump_ns
// grew (readers make no explicit call, so only a trigger inside their own
// contains() charges it), and triggers proceeded.
// Kills, probabilistically: a trigger that releases the flag before its final
// free pass, and a trigger that ignores the flag, here with triggers inside
// readers' contains() among the parties.
TEST(ConcurrentHashSetRcuTest, CadenceStressManyTinySets) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CadenceMixHash>;
    StressShape shape{4, 2, 0, false, 2, 4, 8, 64, std::chrono::milliseconds(20), false};
    shape.cadence = 1;
    shape.slot_limit = size_t{1} << 18;
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    size_t rounds = 0, reader_triggers = 0, auto_reclaims = 0, auto_skipped = 0, stopped = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        StressWitness w;
        ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "tiny cadenced set " + std::to_string(rounds), w));
        reader_triggers += w.reader_triggers;
        auto_reclaims += w.auto_reclaims;
        auto_skipped += w.auto_skipped;
        stopped += w.stopped_by_guard;
        ++rounds;
    } // rounds
    std::printf("[   INFO   ] CadenceStressManyTinySets: %zu sets, auto_reclaims %zu, auto_skipped %zu, "
                "reader iterations with a trigger %zu, rounds stopped by the guard %zu\n",
                rounds, auto_reclaims, auto_skipped, reader_triggers, stopped);
    EXPECT_GT(auto_reclaims, 0u) << "witness: no trigger proceeded in any round";
    EXPECT_GT(reader_triggers, 0u) << "witness: no trigger proceeded inside a reader's contains()";
} // CadenceStressManyTinySets

// The broad cadenced stress: what the three cadence stresses above hold fixed
// (R = 1, long-lived handles refreshed every iteration, advance() the only
// explicit call, no yields), varied in one run. A fresh 4-bucket set per round
// (mixing hash, so the table doubles inside every round and readers' lookups
// win splits whose cleanups trigger), FOUR arena shards, R = 4 (a crossing
// every fourth node per shard, so most retire pushes do not trigger), a ring
// of 8 generations (a long-lived handle pins a bag for up to 64 of its
// iterations, so the ring fills and triggers and calls meet `ring_full`
// beside operations), and:
//   - mixed handles: the even-numbered writers and readers take a handle per
//     operation (every trigger then runs under a temporary handle that pins
//     the generation it closes), the odd-numbered keep a long-lived handle
//     refreshed only every 64th iteration;
//   - one advance() caller AND one reclaim() caller beside the triggers, so
//     all three holders of the flag meet (each refused call is `contended`,
//     each refused trigger skipped);
//   - TestHooks' yield_in_windows: every join yields between its load and its
//     increment and between its increment and its re-verify, and every
//     publish -- an explicit call's or a trigger's, inside an operation --
//     yields before its final free pass, with the flag held.
// Oracles: advance_stress_body()'s (every writer's and reader's determined
// booleans, the end state after advance_and_check() at the quiescent point,
// exact membership), and the destructor's cadence invariants per set.
// Witnesses, over all rounds: triggers proceeded and were skipped; bags and
// nodes freed and freed nodes popped again; a trigger inside a reader's
// contains(); both explicit callers proceeded and some explicit call was
// refused; joins retried; the after-publish hook ran.
// Kills, probabilistically: reclaim() ignoring the flag (the only stress with
// a reclaim() caller); a trigger releasing the flag before its final free
// pass; a trigger whose free pass takes refs <= 1 for unpinned.
// The bump's exchanges moved into the publish window are left to the cadence
// stresses above: here most explicit calls and triggers are refused or find
// the ring full.
TEST_F(HashRcuHookTest, CadenceStressMixedHandlesAndCallers) {
    using Set = ConcurrentResizableHashSetRCU<int, true, CadenceMixHash, 8, TestHooks>;
    StressShape shape{4, 2, 1, false, 4, 4, 64, 64, std::chrono::milliseconds(100), false};
    shape.cadence = 4;
    shape.slot_limit = size_t{1} << 18;
    shape.reclaimers = 1;
    shape.mixed_handles = true;
    shape.refresh_every = 64;
    shape.call_pause = std::chrono::microseconds(50);
    TestHooks::yield_in_windows.store(true);
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    StressWitness all;
    size_t rounds = 0, stopped = 0;
    // A round starts only if it can end by the deadline, so the test stays within the
    // other cadence stresses' second.
    while (std::chrono::steady_clock::now() + shape.run_for <= deadline) {
        StressWitness w;
        ASSERT_NO_FATAL_FAILURE(advance_stress_body<Set>(shape, "mixed handles and callers, set " + std::to_string(rounds), w));
        all.auto_reclaims += w.auto_reclaims;
        all.auto_skipped += w.auto_skipped;
        all.counted_bags_freed += w.counted_bags_freed;
        all.nodes_freed += w.nodes_freed;
        all.pops += w.pops;
        all.reader_triggers += w.reader_triggers;
        all.contended += w.contended;
        all.join_retries += w.join_retries;
        for (size_t o = 0; o < 4; ++o) {
            all.outcomes[o] += w.outcomes[o];
            all.reclaim_outcomes[o] += w.reclaim_outcomes[o];
        } // loop over the four outcomes
        stopped += w.stopped_by_guard;
        ++rounds;
    } // rounds
    const size_t advanced = static_cast<size_t>(Advance::advanced), nothing = static_cast<size_t>(Advance::nothing_retired);
    const size_t full = static_cast<size_t>(Advance::ring_full);
    std::printf("[   INFO   ] CadenceStressMixedHandlesAndCallers: %zu sets, auto_reclaims %zu, auto_skipped %zu, bags freed %zu, "
                "nodes freed %zu, pops %zu, reader iterations with a trigger %zu, contended %zu, join retries %zu; advance() "
                "advanced/nothing_retired/ring_full %zu/%zu/%zu, reclaim() %zu/%zu/%zu; after-publish calls %ld; rounds stopped by the guard %zu\n",
                rounds, all.auto_reclaims, all.auto_skipped, all.counted_bags_freed, all.nodes_freed, all.pops, all.reader_triggers,
                all.contended, all.join_retries, all.outcomes[advanced], all.outcomes[nothing], all.outcomes[full],
                all.reclaim_outcomes[advanced], all.reclaim_outcomes[nothing], all.reclaim_outcomes[full],
                TestHooks::after_publish.calls.load(), stopped);
    EXPECT_GT(all.auto_reclaims, 0u) << "witness: no trigger proceeded";
    EXPECT_GT(all.auto_skipped, 0u) << "witness: no trigger was skipped";
    EXPECT_GT(all.counted_bags_freed, 0u) << "witness: no bag was freed";
    EXPECT_GT(all.pops, 0u) << "witness: no freed node was reused";
    EXPECT_GT(all.reader_triggers, 0u) << "witness: no trigger proceeded inside a reader's contains()";
    EXPECT_GT(all.outcomes[advanced] + all.outcomes[nothing], 0u) << "witness: the advance() caller never proceeded";
    EXPECT_GT(all.reclaim_outcomes[advanced] + all.reclaim_outcomes[nothing], 0u) << "witness: the reclaim() caller never proceeded";
    EXPECT_GT(all.contended, 0u) << "witness: no explicit call met another holder at the flag";
    EXPECT_GT(all.join_retries, 0u) << "witness: no join ever retried";
    EXPECT_GT(TestHooks::after_publish.calls.load(), 0) << "witness: no publish entered its publish-to-free-pass window";
} // CadenceStressMixedHandlesAndCallers

// ===========================================================================
// CONTRACT TESTS: handles, generations and advance() through the public
// interface.
//
// Every test names, in words, the promises it checks: the handle's special
// members and thread rules, new_handle() and refresh(), the operations under a
// handle, advance()'s outcomes and counts, the pin and latency guarantees, the
// template parameters, value lifetime, the destructor, and the diagnostics.
// The get_internal_* accessors are used only as diagnostic instruments, at
// quiescent points where they are valid.
//
// THE MODEL the exact AdvanceResult oracles follow (advance()'s outcomes and
// counts, and the pin, latency and ring guarantees):
//   - The set has a ring of kGenerations slots; one is CURRENT. A handle is a
//     member of the generation current when it joined (new_handle()) or last
//     moved (refresh()). A fresh set is at generation 0 and every `advanced`
//     call publishes the current number plus one; no other outcome changes it.
//   - advance() (single-threaded here) first decides `nothing_retired` by
//     looking at the retired and limbo lists (step (i)); otherwise, when the
//     next slot is occupied (the ring has wrapped) it runs a free pass and
//     refuses with `ring_full` if the slot is still occupied (ii); otherwise it
//     seals the current generation with every retired and limbo node as its
//     bag and publishes the next generation (iii); then a free pass frees bags
//     oldest-first while the oldest sealed generation has no member (iv); then
//     the walk visits every published bucket (v), except after `ring_full` or
//     `contended`.
//   - THE ONE-NODE SETTING: the exact-count tests construct
//     their set with kContractBuckets = 4096 buckets and ONE arena shard, at least
//     128 times the shards their threads touch, and check the setting with
//     ContractOneNode (splits_published and the table size unchanged, nothing
//     in limbo, every one of the kContractBuckets buckets published). In it,
//     contract_retire() (an uncontended insert and erase of one key) retires
//     exactly one node and leaves nothing for the walk (the uncontended
//     operations' unlinking POSTCONDITIONS), so a bag holds exactly the nodes
//     retired into it.
//   - buckets_walked is B = the published buckets: kContractBuckets in the
//     setting (nothing was ever published beyond the constructor's 4096, which
//     ContractOneNode checks). nodes_passed is n = `reachable` before the call
//     when no dead node is reachable; with no key present it is 0.
// The AdvanceResult literals read {freed_nodes, freed_bags, walk_retired,
// buckets_walked, nodes_passed, advance}, the struct's field order.
//
// Tests that need advance()'s walk to call Hash{} (parking, throwing or
// nesting inside the walk: the single advancer, the walk's exception
// guarantee, the destructor's preconditions, joins that never wait, and what
// advance() reads of stored values) need the UNFINISHED-SPLIT construction: a
// FROZEN node at a usable predecessor, which only an unfinished split leaves.
// The contract names "a Hash that throws inside a split" as one way to leave
// one, and an operation's exception clauses make that a legal Hash (an insert
// that throws in a split it met). So the tests build it through the public
// interface (contract_leave_unfinished_split()): after the set has doubled, inserts of
// fresh keys whose Hash{} throws at its 2nd..6th call, until an advance()
// afterwards calls Hash{} at all, which the contract says happens only in
// that construction; a construction that is never reached fails the test with
// a message that says so (never a vacuous pass). The negative ("a
// merely-doubled quiescent set: no Hash{} call from advance()") is a test of
// its own.
// ===========================================================================

// The capacity of every one-node-setting set (a power of two, at least 128 x
// the one arena shard those sets use), and that shard count.
static constexpr size_t kContractBuckets = 4096;
static constexpr size_t kContractShards = 1;

// The instantiations: the default ring (64), a ring of K slots, and the
// no-delete flavor (advance() works for both AllowDelete values).
using ContractSet = ConcurrentResizableHashSetRCU<int, true>;
template <size_t K>
using ContractSetK = ConcurrentResizableHashSetRCU<int, true, std::hash<int>, K>;
using ContractSetNoDel = ConcurrentResizableHashSetRCU<int, false>;

// A 64-bit mixer (splitmix64's finalizer), so that keys spread over the
// buckets and every doubling splits many of them.
static size_t contract_mix(int k) {
    uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(k)) + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27))*0x94d049bb133111ebull;
    return static_cast<size_t>(x ^ (x >> 31));
} // contract_mix()

// ---------------------------------------------------------------------------
// ContractHashControl: the per-thread switches of the hooked Hash{} types
// below (a legal Hash: stable, default-constructible; it may throw, as the
// operations' and advance()'s exception clauses allow). Every switch is
// thread_local and armed only around one call on one thread, so a Hash{} call
// seen in a non-plain state comes from that call alone (for advance(): from
// its walk, the one place advance() hashes a stored key).
// ---------------------------------------------------------------------------
enum class ContractHashMode { plain, count, park, throw_once };
struct ContractHashThrow {};    // the exception a hooked Hash{} throws

struct ContractHashControl {
    static inline thread_local ContractHashMode mode = ContractHashMode::plain;  // this thread's mode
    static inline thread_local size_t calls = 0;          // Hash{} calls on this thread in a non-plain mode
    static inline thread_local int throw_countdown = 0;   // > 0: this thread's countdown-th Hash{} call throws
    static inline thread_local std::function<void()> nested;  // non-empty: run once by this thread's next Hash{} call
    static inline std::atomic<bool> parked{false};        // set by the thread that parked (park mode)
    static inline std::atomic<bool> resume{false};        // set by the test to let the parked thread go on

    // Clears the cross-thread flags before a new parking round.
    static void reset() {
        parked.store(false, std::memory_order_relaxed);
        resume.store(false, std::memory_order_relaxed);
    } // ContractHashControl::reset()
    // Waits (spinning) until `resume` is set; publishes `parked` first.
    static void park_here() {
        parked.store(true, std::memory_order_release);
        while (!resume.load(std::memory_order_acquire)) std::this_thread::yield();
    } // ContractHashControl::park_here()
    // Called at the top of every hooked Hash{}: runs the nested action once,
    // then the countdown throw, then the mode (count, park once, or throw
    // once, after which it falls back to counting).
    static void on_hash() {
        if (nested) {
            std::function<void()> action = std::move(nested);
            nested = nullptr;
            action();
        } // if a nested action is set
        if (throw_countdown > 0 && --throw_countdown == 0) throw ContractHashThrow{};
        switch (mode) {
            case ContractHashMode::plain:
                return;
            case ContractHashMode::count:
                ++calls;
                return;
            case ContractHashMode::park:
                ++calls;
                mode = ContractHashMode::count;
                park_here();
                return;
            case ContractHashMode::throw_once:
                ++calls;
                mode = ContractHashMode::count;
                throw ContractHashThrow{};
        } // switch on the mode
    } // on_hash()
}; // struct ContractHashControl

// Hash{} of int keys with the hook.
struct ContractHookHash {
    size_t operator()(int k) const {
        ContractHashControl::on_hash();
        return contract_mix(k);
    }
}; // struct ContractHookHash

// ---------------------------------------------------------------------------
// ContractCountedKey: an instrumented T (copy-constructible, copy-assignable,
// ==: the requirements on T; deliberately NOT default-constructible, since the
// set does not require it). Every object gets a serial at construction (copy
// included) and its destruction is counted per serial, so "every T destroyed exactly
// once" is checked per object (a double destruction cannot cancel a leak).
// Global counters of destructor, ==, assignment and copy calls give the
// value-lifetime promise "advance() touches no value but through Hash{}".
// Thread-local one-shot switches make the next copy assignment throw (a
// throwing reuse sends its node to limbo) or the next == park (to hold an
// operation in flight under its handle).
// ---------------------------------------------------------------------------
struct ContractAssignThrow {};  // the exception an armed copy assignment throws

struct ContractCountedKey {
    int value;          // the key
    unsigned serial;    // identity of this object (its storage), fixed at construction

    static constexpr unsigned kMaxSerials = 1u << 20;
    static inline std::atomic<unsigned> next_serial{0};                     // the next serial to hand out
    static inline std::atomic<unsigned char> destroyed[kMaxSerials];       // per-serial destruction count
    static inline std::atomic<size_t> destructor_calls{0};                 // all destructions
    static inline std::atomic<size_t> equality_calls{0};                   // all operator== calls
    static inline std::atomic<size_t> assign_calls{0};                     // all copy assignments
    static inline std::atomic<size_t> copy_calls{0};                       // all copy constructions
    static inline thread_local bool throw_on_assign = false;   // one-shot: this thread's next copy assignment throws
    static inline thread_local bool park_on_equality = false;  // one-shot: this thread's next == parks (ContractHashControl flags)

    static unsigned take_serial() { return next_serial.fetch_add(1, std::memory_order_relaxed); }

    explicit ContractCountedKey(int v) : value(v), serial(take_serial()) {}
    ContractCountedKey(const ContractCountedKey& other) : value(other.value), serial(take_serial()) {
        copy_calls.fetch_add(1, std::memory_order_relaxed);
    }
    ContractCountedKey& operator=(const ContractCountedKey& other) {
        assign_calls.fetch_add(1, std::memory_order_relaxed);
        if (throw_on_assign) {
            throw_on_assign = false;
            throw ContractAssignThrow{};
        }
        value = other.value;
        return *this;
    } // ContractCountedKey::operator=()
    // Reads its own bytes (serial): under ASan a destruction of a poisoned
    // value would be reported (the destructor unpoisons free nodes first).
    ~ContractCountedKey() {
        destructor_calls.fetch_add(1, std::memory_order_relaxed);
        if (serial < kMaxSerials) destroyed[serial].fetch_add(1, std::memory_order_relaxed);
    }
    bool operator==(const ContractCountedKey& other) const {
        equality_calls.fetch_add(1, std::memory_order_relaxed);
        if (park_on_equality) {
            park_on_equality = false;
            ContractHashControl::park_here();
        }
        return value == other.value;
    } // ContractCountedKey::operator==()
}; // struct ContractCountedKey

// Hash{} of ContractCountedKey with the same hook as ContractHookHash.
struct ContractCountedKeyHash {
    size_t operator()(const ContractCountedKey& k) const {
        ContractHashControl::on_hash();
        return contract_mix(k.value);
    }
}; // struct ContractCountedKeyHash

// Number of serials in [first, last) not destroyed exactly once.
static int contract_destruction_errors(unsigned first, unsigned last) {
    int errors = 0;
    for (unsigned s = first; s < last && s < ContractCountedKey::kMaxSerials; ++s) {
        if (ContractCountedKey::destroyed[s].load(std::memory_order_relaxed) != 1) ++errors;
    }
    return errors;
} // contract_destruction_errors()

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

// Retires `key`: inserts it (absent) and erases it under a handle released
// before return, so a later advance() sees the retirement (its first step's
// loads see it: the operation's return and the handle's release happen-before
// the call) and no
// handle of this helper pins anything. In the one-node setting: exactly one
// node.
template <typename SetT, typename K>
static void contract_retire(SetT& set, const K& key) {
    typename SetT::handle h = set.new_handle();
    EXPECT_TRUE(set.insert(h, key));
    EXPECT_TRUE(set.erase(h, key));
} // contract_retire()

// ContractOneNode: the testable form of the ONE-NODE SETTING. Taken at
// the start of a test on a set constructed with (kContractBuckets, kContractShards);
// holds() checks that no split published, the table did not grow, all
// kContractBuckets buckets are published (the walk's B) and, unless told otherwise,
// nothing is in limbo. Quiescent use only.
template <typename SetT>
class ContractOneNode {
public:
    explicit ContractOneNode(const SetT& set)
        : set_(set), splits_(set.get_internal_counters().splits_published), table_size_(set.get_internal_table_size()) {}
    testing::AssertionResult holds(bool check_limbo = true) const {
        const size_t splits = set_.get_internal_counters().splits_published;
        const size_t table_size = set_.get_internal_table_size();
        const size_t published = published_buckets(set_);
        const size_t limbo = check_limbo ? set_.get_internal_limbo_count() : 0;
        if (splits == splits_ && table_size == table_size_ && published == kContractBuckets && limbo == 0) return testing::AssertionSuccess();
        return testing::AssertionFailure() << "one-node setting broken: splits_published " << splits_ << " -> " << splits << ", table size "
                                           << table_size_ << " -> " << table_size << ", published " << published << ", limbo " << limbo;
    } // ContractOneNode::holds()

private:
    const SetT& set_;       // the set under test
    size_t splits_;         // splits_published at the start
    size_t table_size_;     // the table size at the start
}; // class ContractOneNode

// Calls set.advance() (single-threaded, nothing else running, one-node
// setting, no key present) and checks it against {freed_nodes, freed_bags, 0,
// B, 0, outcome}, B = kContractBuckets for the outcomes that walk and 0 for
// `ring_full` (a ring_full call does not walk). Returns an AssertionResult so that
// EXPECT_TRUE() reports the caller's line.
template <typename SetT>
static testing::AssertionResult contract_advance_is(SetT& set, size_t freed_nodes, size_t freed_bags, AdvanceResult::Advance outcome) {
    const bool walks = outcome == Advance::advanced || outcome == Advance::nothing_retired;
    const AdvanceResult want{freed_nodes, freed_bags, 0, walks ? kContractBuckets : 0, 0, outcome};
    const AdvanceResult got = set.advance();
    if (got == want) return testing::AssertionSuccess();
    return testing::AssertionFailure() << "advance() returned " << testing::PrintToString(got)
                                       << ", expected " << testing::PrintToString(want);
} // contract_advance_is()

// The ring invariant (generation numbers), through the ring-state view: K
// slots in slot order; the CURRENT slot carries the current number; the
// SEALED slots are one run from oldest_slot up to the slot before
// current_slot, their numbers consecutive and ending at the current number
// minus one; every other slot is EMPTY with nothing bagged. No advance() in
// progress.
template <typename SetT>
static testing::AssertionResult contract_ring_invariant(const SetT& set, size_t K) {
    const typename SetT::InternalRingState ring = set.get_internal_ring_state();
    const uint64_t current = set.get_internal_generation();
    const size_t n = ring.slots.size();
    if (n != K || ring.current_slot >= n || ring.oldest_slot >= n) {
        return testing::AssertionFailure() << "ring of " << n << " slots, current_slot " << ring.current_slot << ", oldest_slot " << ring.oldest_slot;
    }
    if (ring.slots[ring.current_slot].state != SetT::GenerationState::CURRENT || ring.slots[ring.current_slot].gen != current) {
        return testing::AssertionFailure() << "current slot " << ring.current_slot << " is not CURRENT with generation " << current;
    }
    const size_t sealed = (ring.current_slot + n - ring.oldest_slot)%n;
    for (size_t i = 0; i < n; ++i) {
        const size_t index = (ring.oldest_slot + i)%n;
        const typename SetT::InternalRingSlot& slot = ring.slots[index];
        if (i < sealed) {
            if (slot.state != SetT::GenerationState::SEALED || slot.gen != current - sealed + i) {
                return testing::AssertionFailure() << "slot " << index << " should be SEALED with generation " << current - sealed + i;
            }
        } else if (index != ring.current_slot) {
            if (slot.state != SetT::GenerationState::EMPTY || slot.bagged != 0) {
                return testing::AssertionFailure() << "slot " << index << " should be EMPTY with nothing bagged";
            }
        } // the SEALED run, then the EMPTY slots
    } // loop over the slots, oldest first
    return testing::AssertionSuccess();
} // contract_ring_invariant()

// The number of SEALED slots (bags pending), from the ring-state view.
template <typename SetT>
static size_t contract_pending_bags(const SetT& set) {
    const typename SetT::InternalRingState ring = set.get_internal_ring_state();
    size_t sealed = 0;
    for (const typename SetT::InternalRingSlot& slot : ring.slots) {
        if (slot.state == SetT::GenerationState::SEALED) ++sealed;
    }
    return sealed;
} // contract_pending_bags()

// The live-handle count of generation `gen` in the ring (the ring-state
// view), or -1 if no non-EMPTY slot has that generation.
template <typename SetT>
static long contract_refs_of(const SetT& set, uint64_t gen) {
    const typename SetT::InternalRingState ring = set.get_internal_ring_state();
    for (const typename SetT::InternalRingSlot& slot : ring.slots) {
        if (slot.state != SetT::GenerationState::EMPTY && slot.gen == gen) return slot.refs;
    }
    return -1;
} // contract_refs_of()

// Inserts keys K(next_key), K(next_key + 1), ... under one handle until the
// table doubles once (get_internal_table_size(), exact single-threaded), or
// 64 times the old size keys were inserted. Single-threaded use only.
template <typename K, typename SetT>
static void contract_grow_once(SetT& set, int& next_key) {
    const size_t size0 = set.get_internal_table_size();
    typename SetT::handle h = set.new_handle();
    for (size_t i = 0; i < 64*size0 && set.get_internal_table_size() == size0; ++i) {
        EXPECT_TRUE(set.insert(h, K(next_key++)));
    }
} // contract_grow_once()

// The unfinished-split construction, through the public interface: fills the
// set with 300 keys
// K(next_key++) (so that it has doubled and has UNINITIALIZED buckets), then
// inserts fresh keys with this thread's Hash{} armed to throw at its nth call
// (n = 2..6: after the key's own hash, i.e. inside a split the insert met,
// a legal throw), 200 keys per n, and after every insert that threw runs a
// probing advance() with a counting Hash{}. Returns true at the first probe
// whose walk called Hash{}: by the contract an unfinished split (a FROZEN node
// at a usable pred with an unpublished child) is then in the table, and it
// stays there until an operation reaches that child (the walk splits
// nothing). The keys whose insert threw are appended to `thrown` (present or
// absent
// depending on where the throw fell); every other key below next_key was
// inserted. Single-threaded use only.
template <typename K, typename SetT>
static bool contract_leave_unfinished_split(SetT& set, int& next_key, std::vector<int>& thrown) {
    {
        typename SetT::handle h = set.new_handle();
        for (int i = 0; i < 300; ++i) EXPECT_TRUE(set.insert(h, K(next_key++)));
    }
    for (int nth = 2; nth <= 6; ++nth) {
        for (int attempt = 0; attempt < 200; ++attempt) {
            const int key = next_key++;
            bool threw = false;
            {
                typename SetT::handle h = set.new_handle();
                ContractHashControl::throw_countdown = nth;
                try {
                    set.insert(h, K(key));
                } catch (const ContractHashThrow&) {
                    threw = true;
                }
                ContractHashControl::throw_countdown = 0;
            } // the insert's handle
            if (!threw) continue;
            thrown.push_back(key);
            ContractHashControl::mode = ContractHashMode::count;
            ContractHashControl::calls = 0;
            set.advance();
            ContractHashControl::mode = ContractHashMode::plain;
            if (ContractHashControl::calls > 0) return true;
        } // loop over attempts
    } // loop over the throwing call's index
    return false;
} // contract_leave_unfinished_split()

// The throwing-walk construction: the unfinished-split construction, then
// one advance() with this thread's Hash{} armed to throw once; returns true
// iff that call threw (the throw can only come from the walk's Hash{}, the one
// source of a throw in advance()). The discarded advance() result also checks
// at compile time that advance() is not [[nodiscard]], or -Werror rejects
// this line.
template <typename K, typename SetT>
static bool contract_throw_in_walk(SetT& set, int& next_key, std::vector<int>& thrown) {
    if (!contract_leave_unfinished_split<K>(set, next_key, thrown)) return false;
    bool threw = false;
    ContractHashControl::mode = ContractHashMode::throw_once;
    try {
        set.advance();
    } catch (const ContractHashThrow&) {
        threw = true;
    }
    ContractHashControl::mode = ContractHashMode::plain;
    return threw;
} // contract_throw_in_walk()

// ContractParkedAdvance: an advance() parked inside its walk's Hash{} on a
// thread of its own (the construction behind the single-advancer, never-wait
// and destructor tests). The constructor builds the unfinished-split
// construction, then runs advance() on a new thread
// whose first Hash{} call parks; parked() is true when it did (that call is
// then in progress until resume()). resume() lets the parked call finish,
// joins and returns its result; detach() abandons the parked thread
// (death-test children only). The destructor resumes. `thrown` as in
// contract_leave_unfinished_split().
template <typename SetT, typename K>
class ContractParkedAdvance {
public:
    ContractParkedAdvance(SetT& set, int& next_key, std::vector<int>& thrown) : set_(set) {
        if (!contract_leave_unfinished_split<K>(set, next_key, thrown)) return;
        ContractHashControl::reset();
        thread_ = std::thread([this] {
            ContractHashControl::mode = ContractHashMode::park;
            result_ = set_.advance();
            ContractHashControl::mode = ContractHashMode::plain;
            done_.store(true, std::memory_order_release);
        }); // the parked advancer thread
        while (!ContractHashControl::parked.load(std::memory_order_acquire) && !done_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        parked_ = ContractHashControl::parked.load(std::memory_order_acquire);
    } // ContractParkedAdvance()
    ~ContractParkedAdvance() { resume(); }
    ContractParkedAdvance(const ContractParkedAdvance&) = delete;
    ContractParkedAdvance& operator=(const ContractParkedAdvance&) = delete;

    bool parked() const { return parked_; }
    AdvanceResult resume() {
        if (thread_.joinable()) {
            ContractHashControl::resume.store(true, std::memory_order_release);
            thread_.join();
        }
        return result_;
    } // ContractParkedAdvance::resume()
    void detach() {
        if (thread_.joinable()) thread_.detach();
    }

private:
    SetT& set_;                         // the set the parked call runs on
    std::thread thread_;                // the advancing thread
    std::atomic<bool> done_{false};     // its advance() returned without parking
    AdvanceResult result_{};            // its result, read after the join
    bool parked_ = false;               // it parked: its call is in progress
}; // class ContractParkedAdvance

// Checks membership after an unfinished-split construction: every key below
// next_key that is not in `thrown` was inserted and must be present; keys from
// next_key on were never inserted and must be absent (membership is exact and
// unchanged by advance(); the thrown keys' fate
// depends on where the throw fell and is not checked). Returns the number of
// errors.
template <typename K, typename SetT>
static int contract_membership_errors(SetT& set, int next_key, const std::vector<int>& thrown) {
    typename SetT::handle h = set.new_handle();
    int errors = 0;
    for (int k = 0; k < next_key; ++k) {
        if (std::find(thrown.begin(), thrown.end(), k) != thrown.end()) continue;
        if (!set.contains(h, K(k))) ++errors;
    }
    for (int k = next_key; k < next_key + 1000; ++k) {
        if (set.contains(h, K(k))) ++errors;
    }
    return errors;
} // contract_membership_errors()

// ===========================================================================
// The handle type and its special members, new_handle(), and the set's
// declared surface.
// ===========================================================================

// The handle's special members (nothrow default construction, move
// construction and assignment, no copy, nothrow destruction, explicit bool),
// refresh() and swap, new_handle(), advance() not noexcept, the operations'
// handle-first signatures, the set neither copyable nor movable, and a handle
// of another instantiation being another type. Everything here is a declared
// property, so it is checked at compile time; the test body exists so that
// the checks are part of a named test.
TEST(ConcurrentHashSetRcuTest, ContractHandleSpecialMembersDeclared) {
    using H = ContractSet::handle;
    static_assert(std::is_nothrow_default_constructible_v<H>);                          // empty by default
    static_assert(std::is_nothrow_move_constructible_v<H>);                             // move construction
    static_assert(std::is_nothrow_move_assignable_v<H>);                                // move assignment
    static_assert(!std::is_copy_constructible_v<H> && !std::is_copy_assignable_v<H>);  // no copy
    static_assert(std::is_nothrow_destructible_v<H>);                                   // destruction
    static_assert(std::is_constructible_v<bool, const H&>);                             // operator bool
    static_assert(!std::is_convertible_v<const H&, bool>);                              // operator bool: explicit
    static_assert(noexcept(static_cast<bool>(std::declval<const H&>())));               // operator bool
    static_assert(noexcept(std::declval<H&>().refresh()));                              // refresh()
    static_assert(std::is_same_v<decltype(std::declval<H&>().refresh()), bool>);        // refresh()
    static_assert(noexcept(std::declval<H&>().swap(std::declval<H&>())));               // swap, member
    static_assert(std::is_nothrow_swappable_v<H>);                                      // swap, ADL
    static_assert(noexcept(std::declval<const ContractSet&>().new_handle()));          // new_handle(), callable on a const set
    static_assert(std::is_same_v<decltype(std::declval<const ContractSet&>().new_handle()), H>);
    static_assert(!noexcept(std::declval<ContractSet&>().advance()));                  // advance(): NOT noexcept
    static_assert(std::is_same_v<decltype(std::declval<ContractSet&>().advance()), AdvanceResult>);
    static_assert(std::is_same_v<decltype(&ContractSet::contains), bool (ContractSet::*)(const H&, const int&)>);  // handle-first signature
    static_assert(std::is_same_v<decltype(&ContractSet::insert), bool (ContractSet::*)(const H&, const int&)>);    // handle-first signature
    static_assert(std::is_same_v<decltype(&ContractSet::erase), bool (ContractSet::*)(const H&, const int&)>);     // handle-first signature
    static_assert(!std::is_copy_constructible_v<ContractSet> && !std::is_copy_assignable_v<ContractSet>);  // neither copyable nor movable
    static_assert(!std::is_move_constructible_v<ContractSet> && !std::is_move_assignable_v<ContractSet>);  // neither copyable nor movable
    static_assert(!std::is_same_v<H, ContractSetK<3>::handle>);                         // a type per instantiation
    static_assert(!std::is_constructible_v<H, ContractSetK<3>::handle&&>);
#ifdef NDEBUG
    static_assert(sizeof(H) == sizeof(void*));                                          // one pointer, release builds only
#endif
    SUCCEED();
} // ContractHandleSpecialMembersDeclared

// A default-constructed handle is empty, and an empty handle
// (default-constructed, or emptied by a move) may outlive its set: `outer` is
// declared before the set, joined, emptied by moving from it, and destroyed
// after the set. `sink`, live, is destroyed before the set (a live handle
// must not outlive its set). Under ASan a destructor of the empty `outer`
// that touched the dead set would be reported.
TEST(ConcurrentHashSetRcuTest, ContractEmptyHandleMayOutliveSet) {
    ContractSet::handle outer;
    EXPECT_FALSE(outer);
    {
        ContractSet set(kContractBuckets, kContractShards);
        outer = set.new_handle();
        EXPECT_TRUE(outer);
        ContractSet::handle sink(std::move(outer));
        EXPECT_FALSE(outer);
        EXPECT_TRUE(sink);
    } // the set's scope
    EXPECT_FALSE(outer);
} // ContractEmptyHandleMayOutliveSet

// new_handle() (on a const set too)
// returns a live handle whose generation was current during the call;
// single-threaded that is the current generation: 0 on a fresh set, 1 after
// one `advanced` call. handle_refs (the sweep allows and counts live idle
// handles) equals the number of live handles.
TEST(ConcurrentHashSetRcuTest, ContractNewHandleJoinsCurrentGeneration) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    const ContractSet& cset = set;
    EXPECT_EQ(set.get_internal_generation(), 0u);
    ContractSet::handle h = cset.new_handle();
    EXPECT_TRUE(h);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h), 0u);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    EXPECT_EQ(set.get_internal_generation(), 1u);
    ContractSet::handle h2 = cset.new_handle();
    EXPECT_TRUE(h2);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h2), 1u);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h), 0u);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 2);
    EXPECT_TRUE(contract_ring_invariant(set, 64));
    EXPECT_TRUE(setting.holds());
} // ContractNewHandleJoinsCurrentGeneration

// The operations under a handle keep their return values (contract items
// 1-3), a temporary handle per operation is legal, and the no-delete flavor
// works too: insert is
// true once per absent->present transition, erase once per present->absent
// one, contains is exact; a prvalue handle leaves at the end of its full
// expression (handle_refs counts only `h` afterwards).
TEST(ConcurrentHashSetRcuTest, ContractOperationsUnderHandle) {
    ContractSet set(kContractBuckets, kContractShards);
    ContractSet::handle h = set.new_handle();
    EXPECT_FALSE(set.contains(h, 7));
    EXPECT_TRUE(set.insert(h, 7));
    EXPECT_FALSE(set.insert(h, 7));
    EXPECT_TRUE(set.contains(h, 7));
    EXPECT_TRUE(set.erase(h, 7));
    EXPECT_FALSE(set.erase(h, 7));
    EXPECT_FALSE(set.contains(h, 7));
    EXPECT_TRUE(set.insert(set.new_handle(), 8));
    EXPECT_TRUE(set.contains(set.new_handle(), 8));
    EXPECT_TRUE(set.erase(set.new_handle(), 8));
    EXPECT_FALSE(set.contains(set.new_handle(), 8));
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);

    ContractSetNoDel nodel(kContractBuckets, kContractShards);
    ContractSetNoDel::handle hn = nodel.new_handle();
    EXPECT_TRUE(nodel.insert(hn, 3));
    EXPECT_FALSE(nodel.insert(nodel.new_handle(), 3));
    EXPECT_TRUE(nodel.contains(hn, 3));
    EXPECT_FALSE(nodel.contains(hn, 4));
} // ContractOperationsUnderHandle

// Move construction transfers the membership (the target is
// live in the source's generation, the source is empty, the generation's
// count is unchanged), and the moved-to handle still pins: a node retired
// before each of two calls is freed by neither while it lives (the first
// call's bag is its own generation's, the second's is younger: the pin is
// cumulative), and the first call after its release frees both bags, by the
// free pass of a `nothing_retired` call (which still runs its free pass).
TEST(ConcurrentHashSetRcuTest, ContractMoveConstructionKeepsMembershipAndPin) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle a = set.new_handle();
    const uint64_t g0 = ContractSet::get_internal_handle_generation(a);
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    std::unique_ptr<ContractSet::handle> b = std::make_unique<ContractSet::handle>(std::move(a));
    EXPECT_FALSE(a);
    EXPECT_TRUE(*b);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(*b), g0);
    EXPECT_EQ(contract_refs_of(set, g0), 1);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    contract_retire(set, 2);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    b.reset();
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 0);
    EXPECT_TRUE(contract_advance_is(set, 2, 2, Advance::nothing_retired));
    EXPECT_TRUE(setting.holds());
} // ContractMoveConstructionKeepsMembershipAndPin

// Move assignment over a LIVE handle makes the target leave its
// generation first. `a` pins g0 (bag {1}); `b` joins g1; `a = std::move(b)`:
// g0 has no member left, so the next call (one node retired in g1) frees g0's
// bag and not g1's, which `a` now pins: {1, 1, ...advanced}. Releasing `a`
// frees g1's bag in the following call. Then the other two shapes of move
// assignment:
// assigning an empty handle over a live one leaves (the target becomes empty,
// its pin ends), and assigning a live handle into an empty one only takes.
TEST(ConcurrentHashSetRcuTest, ContractMoveAssignOverLiveHandleReleasesTarget) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    std::unique_ptr<ContractSet::handle> a = std::make_unique<ContractSet::handle>(set.new_handle());
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    ContractSet::handle b = set.new_handle();
    const uint64_t g1 = ContractSet::get_internal_handle_generation(b);
    contract_retire(set, 2);
    *a = std::move(b);
    EXPECT_TRUE(*a);
    EXPECT_FALSE(b);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(*a), g1);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::advanced));
    a.reset();
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired));

    // An empty handle assigned over a live one: the target leaves and is empty.
    ContractSet::handle c = set.new_handle();
    contract_retire(set, 3);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    c = ContractSet::handle();
    EXPECT_FALSE(c);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 0);
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired));

    // A live handle assigned into an empty one: it takes the membership.
    ContractSet::handle d;
    d = set.new_handle();
    EXPECT_TRUE(d);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(d), set.get_internal_generation());
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    EXPECT_TRUE(setting.holds());
} // ContractMoveAssignOverLiveHandleReleasesTarget

// Self-move-assignment is a no-op: a live handle stays live, same
// generation, still pinning; an empty one stays empty. g++ rejects a literal
// `h = std::move(h)` under -Werror=self-move, so the source is reached
// through a reference, which is exactly the aliasing case the promise covers.
TEST(ConcurrentHashSetRcuTest, ContractSelfMoveAssignIsNoOp) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    std::unique_ptr<ContractSet::handle> h = std::make_unique<ContractSet::handle>(set.new_handle());
    const uint64_t g0 = ContractSet::get_internal_handle_generation(*h);
    ContractSet::handle& alias = *h;
    *h = std::move(alias);
    EXPECT_TRUE(*h);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(*h), g0);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    *h = std::move(alias);
    contract_retire(set, 2);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    h.reset();
    EXPECT_TRUE(contract_advance_is(set, 2, 2, Advance::nothing_retired));

    ContractSet::handle empty;
    ContractSet::handle& empty_alias = empty;
    empty = std::move(empty_alias);
    EXPECT_FALSE(empty);
    EXPECT_TRUE(setting.holds());
} // ContractSelfMoveAssignIsNoOp

// swap (member, and ADL through the two-step idiom) exchanges the
// memberships, either side possibly empty, without changing any count;
// self-swap (member and ADL) is a no-op: same generation, same count. The pin
// moves with the membership: after the swaps the g0 membership is in `e`;
// releasing the g1 holder frees nothing (g0 is pinned and older), releasing
// `e` frees both bags.
TEST(ConcurrentHashSetRcuTest, ContractSwapExchangesMemberships) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle a = set.new_handle();
    const uint64_t g0 = ContractSet::get_internal_handle_generation(a);
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    std::unique_ptr<ContractSet::handle> b = std::make_unique<ContractSet::handle>(set.new_handle());
    const uint64_t g1 = ContractSet::get_internal_handle_generation(*b);
    ASSERT_EQ(g1, g0 + 1);
    a.swap(*b);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(a), g1);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(*b), g0);
    {
        using std::swap;
        swap(a, *b);
    }
    EXPECT_EQ(ContractSet::get_internal_handle_generation(a), g0);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(*b), g1);
    a.swap(a);
    swap(a, a);
    EXPECT_TRUE(a);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(a), g0);
    EXPECT_EQ(contract_refs_of(set, g0), 1);
    std::unique_ptr<ContractSet::handle> e = std::make_unique<ContractSet::handle>();
    swap(a, *e);
    EXPECT_FALSE(a);
    EXPECT_TRUE(*e);
    EXPECT_EQ(ContractSet::get_internal_handle_generation(*e), g0);
    EXPECT_EQ(contract_refs_of(set, g0), 1);
    EXPECT_EQ(contract_refs_of(set, g1), 1);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 2);
    ContractSet::handle empty2;
    a.swap(empty2);  // both empty
    EXPECT_FALSE(a);
    EXPECT_FALSE(empty2);
    a.swap(a);       // self-swap of an empty handle
    EXPECT_FALSE(a);

    b.reset();
    contract_retire(set, 2);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    e.reset();
    EXPECT_TRUE(contract_advance_is(set, 2, 2, Advance::nothing_retired));
    EXPECT_TRUE(setting.holds());
} // ContractSwapExchangesMemberships

// ===========================================================================
// refresh(); the thread rules; the operations.
// ===========================================================================

// refresh() returns
// false (fast path, nothing changed) on a fresh set and after a
// `nothing_retired` call (no generation was published); after an `advanced`
// call that happens-before it (same thread) it returns true and joins the
// current generation (1), and a second refresh() is false again. The refresh
// released the old generation: the next call (nothing retired since) frees
// the bag by its free pass: {1, 1, ..., nothing_retired}. Then the
// happens-before edge through a join: an `advanced` call on another thread,
// joined, makes refresh() true (generation 2).
TEST(ConcurrentHashSetRcuTest, ContractRefreshFastAndSlowPath) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle h = set.new_handle();
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h), 0u);
    EXPECT_FALSE(h.refresh());
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h), 0u);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::nothing_retired));
    EXPECT_FALSE(h.refresh());
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    EXPECT_TRUE(h.refresh());
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h), 1u);
    EXPECT_FALSE(h.refresh());
    EXPECT_TRUE(h);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired));

    contract_retire(set, 2);
    AdvanceResult remote{};
    std::thread advancer([&] { remote = set.advance(); });
    advancer.join();
    EXPECT_EQ(remote.advance, Advance::advanced);
    EXPECT_TRUE(h.refresh());
    EXPECT_EQ(ContractSet::get_internal_handle_generation(h), 2u);
    EXPECT_FALSE(h.refresh());
    EXPECT_TRUE(setting.holds());
} // ContractRefreshFastAndSlowPath

// refresh() releases the old generation for reclamation once no other handle
// pins it: two handles in g0; after an `advanced` call, refreshing one
// frees nothing (the other still pins g0); refreshing the second frees g0's
// bag at the next call.
TEST(ConcurrentHashSetRcuTest, ContractRefreshReleasesWhenNoOtherHandlePins) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle a = set.new_handle();
    ContractSet::handle b = set.new_handle();
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    EXPECT_TRUE(a.refresh());
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::nothing_retired));
    EXPECT_TRUE(b.refresh());
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired));
    EXPECT_TRUE(setting.holds());
} // ContractRefreshReleasesWhenNoOtherHandlePins

// A handle has no thread affinity: one created on the main thread is used by
// another thread (100 insert/contains/erase triples: 100 retirements, nothing
// left live), then destroyed by a third thread, each after a join (the
// happens-before edge the thread rules ask for). While it lives it pins the
// bag of the
// call that closes its generation; after its destruction on the other thread
// the next call frees that bag (100 nodes, one bag). One arena shard, so the
// threads touch one shard and 4096 buckets keep the one-node setting.
TEST(ConcurrentHashSetRcuTest, ContractHandleUsedAndDestroyedOnOtherThreads) {
    constexpr int kKeys = 100;
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    std::unique_ptr<ContractSet::handle> h = std::make_unique<ContractSet::handle>(set.new_handle());
    int errors = 0;
    std::thread user([&] {
        for (int k = 0; k < kKeys; ++k) {
            if (!set.insert(*h, k)) ++errors;
            if (!set.contains(*h, k)) ++errors;
            if (!set.erase(*h, k)) ++errors;
        } // loop over the keys
    }); // thread user
    user.join();
    EXPECT_EQ(errors, 0);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    std::thread destroyer([&] { h.reset(); });
    destroyer.join();
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 0);
    EXPECT_TRUE(contract_advance_is(set, kKeys, 1, Advance::nothing_retired));
    EXPECT_TRUE(setting.holds());
} // ContractHandleUsedAndDestroyedOnOtherThreads

// Concurrent operations SHARING one handle (the same const handle&).
// Four threads insert disjoint ranges, then (after a barrier) each checks every
// thread's keys and a range never inserted: every insert true once, membership
// exact. Under TSan the sharing itself must be race-free.
TEST(ConcurrentHashSetRcuTest, ContractSharedHandleConcurrentOperations) {
    constexpr int kThreads = 4;
    constexpr int kPerThread = 500;
    ContractSet set(4);
    const ContractSet::handle h = set.new_handle();
    std::atomic<int> errors{0};
    std::barrier sync(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                if (!set.insert(h, t*kPerThread + i)) errors.fetch_add(1, std::memory_order_relaxed);
            }
            sync.arrive_and_wait();
            for (int k = 0; k < kThreads*kPerThread; ++k) {
                if (!set.contains(h, k)) errors.fetch_add(1, std::memory_order_relaxed);
                if (set.contains(h, -1 - k)) errors.fetch_add(1, std::memory_order_relaxed);
            }
        }); // thread t
    } // loop over threads
    for (std::thread& t : threads) t.join();
    EXPECT_EQ(errors.load(), 0);
    EXPECT_TRUE(h);
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 1);
} // ContractSharedHandleConcurrentOperations

// An operation that throws leaves its handle live and usable (MEMBERSHIP
// AFTER A THROW: an insert that throws from Hash{} before its publishing CAS
// leaves the key absent; a throwing contains() changes no membership; an erase
// that throws from Hash{} before its mark leaves the key present; the next
// advance() is not `contended`; the accounting is consistent), and the debug
// in-flight counter is restored on a throw (in a debug build refresh(), the
// move and the destruction below assert it zero, so a leaked increment would
// abort here). Each throw comes from the operation's FIRST Hash{} call, the
// key's own hash, which precedes any publishing CAS or mark.
TEST(ConcurrentHashSetRcuTest, ContractThrowingOperationLeavesHandleUsable) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    SetT set(kContractBuckets, kContractShards);
    SetT::handle h = set.new_handle();
    ContractHashControl::throw_countdown = 1;
    EXPECT_THROW(set.insert(h, 5), ContractHashThrow);
    ContractHashControl::throw_countdown = 0;
    EXPECT_TRUE(h);
    EXPECT_FALSE(h.refresh());
    EXPECT_FALSE(set.contains(h, 5));
    EXPECT_TRUE(set.insert(h, 6));
    ContractHashControl::throw_countdown = 1;
    EXPECT_THROW(set.contains(h, 6), ContractHashThrow);
    ContractHashControl::throw_countdown = 1;
    EXPECT_THROW(set.erase(h, 6), ContractHashThrow);
    ContractHashControl::throw_countdown = 0;
    EXPECT_TRUE(set.contains(h, 6));
    EXPECT_FALSE(set.contains(h, 5));
    SetT::handle moved(std::move(h));
    EXPECT_TRUE(moved);
    EXPECT_TRUE(set.erase(moved, 6));
    EXPECT_NE(set.advance().advance, Advance::contended);
    EXPECT_TRUE(set.get_internal_accounting().consistent);
} // ContractThrowingOperationLeavesHandleUsable

// ===========================================================================
// advance()'s results, case by case, and the pin, latency and ring
// guarantees.
// ===========================================================================

// A fresh set's call (on a set into which nothing was ever inserted, B = the
// constructor's capacity rounded up to a power of two, minimum 4: 64 -> 64,
// 5 -> 8, 1 -> 4, 4096 -> 4096; `{0, 0, 0, B, 0, nothing_retired}`), the
// generation (0, unchanged by `nothing_retired`), with or without a held
// handle, for both AllowDelete values.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceFreshSetIsNothingRetired) {
    for (const std::pair<size_t, size_t>& capacity_b : {std::pair<size_t, size_t>{64, 64}, {5, 8}, {1, 4}, {kContractBuckets, kContractBuckets}}) {
        ContractSet set(capacity_b.first, kContractShards);
        const size_t b = capacity_b.second;
        EXPECT_EQ(set.get_internal_generation(), 0u);
        EXPECT_EQ(set.advance(), (AdvanceResult{0, 0, 0, b, 0, Advance::nothing_retired})) << "capacity " << capacity_b.first;
        EXPECT_EQ(set.advance(), (AdvanceResult{0, 0, 0, b, 0, Advance::nothing_retired})) << "capacity " << capacity_b.first;
        {
            ContractSet::handle h = set.new_handle();
            EXPECT_EQ(set.advance(), (AdvanceResult{0, 0, 0, b, 0, Advance::nothing_retired})) << "capacity " << capacity_b.first;
        }
        EXPECT_EQ(set.get_internal_generation(), 0u);
        EXPECT_TRUE(contract_ring_invariant(set, 64));

        ContractSetNoDel nodel(capacity_b.first, kContractShards);
        EXPECT_EQ(nodel.advance(), (AdvanceResult{0, 0, 0, b, 0, Advance::nothing_retired})) << "capacity " << capacity_b.first;
        EXPECT_EQ(nodel.get_internal_generation(), 0u);
    } // loop over capacities
} // ContractAdvanceFreshSetIsNothingRetired

// With every handle released, the bag a call closes is freed in the SAME
// call; the generation goes up by one per `advanced`; the walk passes the
// reachable nodes: one node
// retired, no handle: {1, 1, 0, B, 0, advanced}; the next call has nothing:
// {0, 0, 0, B, 0, nothing_retired}. Three nodes retired between two calls form
// ONE bag: {3, 1, ...}. With five live keys present (no dead node reachable)
// the call passes exactly the `reachable` nodes: {1, 1, 0, B, 5, advanced}.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceReleasedHandlesFreeInSameCall) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::advanced));
    EXPECT_EQ(set.get_internal_generation(), 1u);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::nothing_retired));
    EXPECT_EQ(set.get_internal_generation(), 1u);
    contract_retire(set, 2);
    contract_retire(set, 3);
    contract_retire(set, 4);
    EXPECT_TRUE(contract_advance_is(set, 3, 1, Advance::advanced));
    EXPECT_EQ(set.get_internal_generation(), 2u);

    constexpr size_t kLive = 5;
    {
        ContractSet::handle h = set.new_handle();
        for (int k = 100; k < 100 + static_cast<int>(kLive); ++k) EXPECT_TRUE(set.insert(h, k));
    }
    contract_retire(set, 5);
    const ContractSet::InternalAccounting before = set.get_internal_accounting();
    ASSERT_EQ(before.reachable_dead, 0u);
    EXPECT_EQ(before.reachable, kLive);
    EXPECT_EQ(set.advance(), (AdvanceResult{1, 1, 0, kContractBuckets, kLive, Advance::advanced}));
    ContractSet::handle h = set.new_handle();
    for (int k = 100; k < 100 + static_cast<int>(kLive); ++k) EXPECT_TRUE(set.contains(h, k));
    EXPECT_TRUE(setting.holds());
} // ContractAdvanceReleasedHandlesFreeInSameCall

// The latency with a long-lived handle refreshed before
// EACH call. Per round: one node retired; refresh() is the fast path (false:
// the handle is current); the call closes the bag the handle pins
// ({0, 0, 0, B, 0, advanced}); refresh() moves (true); the next call frees the
// bag ({1, 1, 0, B, 0, nothing_retired}: nothing retired between the calls).
// Five rounds: the steady state, not just the first.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceRefreshBeforeEachCallTwoCallLatency) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle h = set.new_handle();
    for (int round = 0; round < 5; ++round) {
        contract_retire(set, round);
        EXPECT_FALSE(h.refresh()) << "round " << round;
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced)) << "round " << round;
        EXPECT_TRUE(h.refresh()) << "round " << round;
        EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired)) << "round " << round;
    } // loop over rounds
    EXPECT_TRUE(setting.holds());
} // ContractAdvanceRefreshBeforeEachCallTwoCallLatency

// The pin is cumulative, the caller's own handle included: a handle of g0
// held by the calling thread; nodes
// retired before each of three calls land in the bags of g0, g1 and g2; g1 and
// g2 have no member, yet nothing is freed (no bag of a generation >= g0) until
// the handle leaves; then one call frees all three bags.
TEST(ConcurrentHashSetRcuTest, ContractAdvancePinIsCumulative) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    std::unique_ptr<ContractSet::handle> h = std::make_unique<ContractSet::handle>(set.new_handle());
    for (int call = 0; call < 3; ++call) {
        contract_retire(set, call);
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced)) << "call " << call;
    }
    EXPECT_EQ(set.get_internal_accounting().bagged, 3u);
    for (uint64_t g = 0; g < 3; ++g) EXPECT_EQ(set.get_internal_bag_count(g), 1u) << "generation " << g;
    EXPECT_EQ(contract_pending_bags(set), 3u);
    EXPECT_TRUE(contract_ring_invariant(set, 64));
    h.reset();
    EXPECT_TRUE(contract_advance_is(set, 3, 3, Advance::nothing_retired));
    EXPECT_EQ(set.get_internal_accounting().bagged, 0u);
    EXPECT_TRUE(contract_ring_invariant(set, 64));
    EXPECT_TRUE(setting.holds());
} // ContractAdvancePinIsCumulative

// The ring_full-and-recovery sequence for a ring of K slots: one handle pinned
// at the first generation, one node retired before each call; calls 1..K-1 return
// `advanced` and free nothing (K-1 slots consumed). One more node
// retired; then either
// - through_ring_full: call K returns `ring_full` ({0, 0, 0, 0, 0}: the free
//   passes freed nothing, no walk), the generation is unchanged, a second call
//   is `ring_full` again (no automatic recovery; the node retired before call
//   K waits on the retired list: retired == 1, K-1 bags pending); or
// - not: no call (the release counts without the refusal).
// Nothing is retired after that. Recovery, by_refresh: the handle refreshes
// (true) and the next call frees the K-1 waiting bags and advances, the
// refreshed handle pinning the bag it closes (the one node):
// {K-1, K-1, 0, B, 0, advanced}; releasing the handle then frees that bag:
// {1, 1, ..., nothing_retired}. Recovery by release: the next call frees the
// K-1 waiting bags AND the one it closes: {K, K, 0, B, 0, advanced}. The ring
// invariant holds after every call.
template <size_t K>
static void contract_ring_full_then_recover(bool through_ring_full, bool by_refresh) {
    using SetT = ContractSetK<K>;
    SCOPED_TRACE(testing::Message() << "kGenerations " << K << (through_ring_full ? ", through ring_full" : ", no ring_full call")
                                    << (by_refresh ? ", recovery by refresh" : ", recovery by release"));
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    std::unique_ptr<typename SetT::handle> pin = std::make_unique<typename SetT::handle>(set.new_handle());
    int key = 0;
    for (size_t call = 1; call < K; ++call) {
        contract_retire(set, key++);
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced)) << "call " << call;
        EXPECT_TRUE(contract_ring_invariant(set, K)) << "call " << call;
    } // calls 1 .. K-1
    contract_retire(set, key++);
    if (through_ring_full) {
        const uint64_t g = set.get_internal_generation();
        EXPECT_EQ(g, K - 1);
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::ring_full)) << "call " << K;
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::ring_full)) << "call " << K + 1;
        EXPECT_EQ(set.get_internal_generation(), g);
        EXPECT_TRUE(contract_ring_invariant(set, K));
        const typename SetT::InternalAccounting acc = set.get_internal_accounting();
        EXPECT_EQ(acc.retired, 1u);
        EXPECT_EQ(acc.bagged, K - 1);
        EXPECT_EQ(contract_pending_bags(set), K - 1);
    } // if through ring_full
    if (by_refresh) {
        EXPECT_TRUE(pin->refresh());
        EXPECT_TRUE(contract_advance_is(set, K - 1, K - 1, Advance::advanced));
        EXPECT_TRUE(contract_ring_invariant(set, K));
        pin.reset();
        EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired));
    } else {
        pin.reset();
        EXPECT_TRUE(contract_advance_is(set, K, K, Advance::advanced));
    }
    EXPECT_TRUE(contract_ring_invariant(set, K));
    EXPECT_EQ(set.get_internal_generation(), K);
    const typename SetT::InternalAccounting acc = set.get_internal_accounting();
    EXPECT_TRUE(acc.consistent);
    EXPECT_EQ(acc.bagged, 0u);
    EXPECT_EQ(acc.retired, 0u);
    EXPECT_TRUE(setting.holds());
} // contract_ring_full_then_recover()

// ring_full after kGenerations - 1 advances
// under one pinned handle and recovery after its refresh, at the minimum ring
// (2), a small one (4) and the default (64).
TEST(ConcurrentHashSetRcuTest, ContractAdvanceRingFullAndRecoveryByRefresh) {
    contract_ring_full_then_recover<2>(true, true);
    contract_ring_full_then_recover<4>(true, true);
    contract_ring_full_then_recover<64>(true, true);
} // ContractAdvanceRingFullAndRecoveryByRefresh

// Recovery after that handle leaves (the release frees kGenerations nodes in
// kGenerations bags), same rings.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceRingFullAndRecoveryByRelease) {
    contract_ring_full_then_recover<2>(true, false);
    contract_ring_full_then_recover<4>(true, false);
    contract_ring_full_then_recover<64>(true, false);
} // ContractAdvanceRingFullAndRecoveryByRelease

// The two arithmetic facts of the counts, without the `ring_full` call: after
// kGenerations - 1 single-node advances under one pinned handle and one more
// node retired, the handle refreshed, the next call is {K-1, K-1, ...,
// advanced}; with the handle released instead, {K, K, ..., advanced}. A
// full ring with a freeable oldest slot is cleared in the call and the call
// advances.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceCountsAfterFullRing) {
    contract_ring_full_then_recover<3>(false, true);
    contract_ring_full_then_recover<3>(false, false);
    contract_ring_full_then_recover<64>(false, true);
    contract_ring_full_then_recover<64>(false, false);
} // ContractAdvanceCountsAfterFullRing

// `nothing_retired` is decided before `ring_full`: with every slot in use
// (kGenerations = 3, two advances under a pinned handle) and nothing retired,
// the call is
// `nothing_retired`, not `ring_full`: no slot consumed (generation unchanged),
// and the walk still runs (buckets_walked == B). With a node retired, the same
// state gives `ring_full`.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceNothingRetiredWinsOverFullRing) {
    using SetT = ContractSetK<3>;
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    SetT::handle pin = set.new_handle();
    for (int call = 0; call < 2; ++call) {
        contract_retire(set, call);
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced)) << "call " << call;
    }
    EXPECT_EQ(set.get_internal_generation(), 2u);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::nothing_retired));
    EXPECT_EQ(set.get_internal_generation(), 2u);
    contract_retire(set, 10);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::ring_full));
    EXPECT_EQ(set.get_internal_generation(), 2u);
    EXPECT_TRUE(contract_ring_invariant(set, 3));
    EXPECT_TRUE(setting.holds());
} // ContractAdvanceNothingRetiredWinsOverFullRing

// A pin released before the call never yields ring_full: a full ring whose
// oldest bag is freeable in the call returns `advanced`, and the oldest-first
// free pass stops at a pinned generation:
// kGenerations = 3; `pin0` in g0, `pin1` in g1, one node retired before each
// of two calls, so every slot is in use (g0, g1 sealed, g2 current). `pin0`
// leaves; with a node retired, the call's in-wrap free pass frees g0's bag
// (g1 is pinned: the pass stops there), which empties the next slot, so the
// call advances: {1, 1, 0, B, 0, advanced}, generation 3. Releasing `pin1`
// frees g1's and g2's bags at the next call.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceRingFullWithFreeableOldestAdvances) {
    using SetT = ContractSetK<3>;
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    std::unique_ptr<SetT::handle> pin0 = std::make_unique<SetT::handle>(set.new_handle());
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    std::unique_ptr<SetT::handle> pin1 = std::make_unique<SetT::handle>(set.new_handle());
    contract_retire(set, 2);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    pin0.reset();
    contract_retire(set, 3);
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::advanced));
    EXPECT_EQ(set.get_internal_generation(), 3u);
    EXPECT_TRUE(contract_ring_invariant(set, 3));
    pin1.reset();
    EXPECT_TRUE(contract_advance_is(set, 2, 2, Advance::nothing_retired));
    EXPECT_TRUE(contract_ring_invariant(set, 3));
    EXPECT_TRUE(setting.holds());
} // ContractAdvanceRingFullWithFreeableOldestAdvances

// The wrap with every handle released (ten or more consecutive `advanced`
// calls, each freeing its own bag; here 12, so the ring wraps several times;
// the generation goes up by exactly one per call), and the refresh-before-each-
// call pattern, whose bags occupy two slots at most;
// the ring invariant after every call.
template <size_t K>
static void contract_ring_wrap() {
    using SetT = ContractSetK<K>;
    SCOPED_TRACE(testing::Message() << "kGenerations " << K);
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    for (int call = 0; call < 12; ++call) {
        contract_retire(set, call);
        EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::advanced)) << "call " << call;
        EXPECT_EQ(set.get_internal_generation(), static_cast<uint64_t>(call + 1)) << "call " << call;
        EXPECT_TRUE(contract_ring_invariant(set, K)) << "call " << call;
    } // twelve calls
    typename SetT::handle h = set.new_handle();
    for (int round = 0; round < 10; ++round) {
        contract_retire(set, 100 + round);
        EXPECT_FALSE(h.refresh()) << "round " << round;
        EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced)) << "round " << round;
        EXPECT_TRUE(contract_ring_invariant(set, K)) << "round " << round;
        EXPECT_TRUE(h.refresh()) << "round " << round;
        EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::nothing_retired)) << "round " << round;
        EXPECT_TRUE(contract_ring_invariant(set, K)) << "round " << round;
    } // loop over rounds
    EXPECT_TRUE(setting.holds());
} // contract_ring_wrap()

// kGenerations = 2 compiles and works: the minimum ring (under one pinned
// handle the SECOND call with something retired is already `ring_full`; after
// the handle leaves, nothing retired since, the next call frees both bags: the
// release counts at K = 2), and the wrap at 2.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceMinimumRing) {
    using SetT = ContractSetK<2>;
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    std::unique_ptr<SetT::handle> pin = std::make_unique<SetT::handle>(set.new_handle());
    contract_retire(set, 1);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced));
    contract_retire(set, 2);
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::ring_full));
    pin.reset();
    EXPECT_TRUE(contract_advance_is(set, 2, 2, Advance::advanced));
    EXPECT_TRUE(setting.holds());
    contract_ring_wrap<2>();
} // ContractAdvanceMinimumRing

// The ring wrap at kGenerations = 3.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceRingWrapThreeGenerations) {
    contract_ring_wrap<3>();
} // ContractAdvanceRingWrapThreeGenerations

// Nothing is freed unless advance() is called: no operation frees, no
// handle's leave frees (diagnostic form: free_nodes never increases and
// retired + bagged never decreases without a call); reuse before growth; the
// diagnostic counters. After a call freed 3 nodes: inserts of 10 keys (the
// first 3 take the free nodes: free_nodes 0), erases of 5, contains, a
// refresh, a move-assignment and
// handle releases leave the free pass's counters (bags_freed, nodes_freed) and
// `bumps` unchanged, free_nodes at 0 and the 5 erased nodes on the retired
// lists, nothing bagged; then a call under a held handle bags them (pinned),
// and that handle's release frees nothing: bagged stays 5; the next call frees
// them: {5, 1, 0, B, n = 5 live keys, nothing_retired}.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceFreesNothingWithoutACall) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    contract_retire(set, 1);
    contract_retire(set, 2);
    contract_retire(set, 3);
    EXPECT_TRUE(contract_advance_is(set, 3, 1, Advance::advanced));
    const ContractSet::InternalCounters c0 = set.get_internal_counters();
    const ContractSet::InternalAccounting a0 = set.get_internal_accounting();
    EXPECT_EQ(a0.free_nodes, 3u);
    {
        ContractSet::handle h = set.new_handle();
        for (int k = 10; k < 20; ++k) EXPECT_TRUE(set.insert(h, k));
        for (int k = 10; k < 15; ++k) EXPECT_TRUE(set.erase(h, k));
        for (int k = 10; k < 20; ++k) EXPECT_EQ(set.contains(h, k), k >= 15);
        EXPECT_FALSE(h.refresh());
        ContractSet::handle h2 = set.new_handle();
        h2 = std::move(h);
    } // the handles' scope
    const ContractSet::InternalCounters c1 = set.get_internal_counters();
    const ContractSet::InternalAccounting a1 = set.get_internal_accounting();
    EXPECT_EQ(c1.nodes_freed, c0.nodes_freed);
    EXPECT_EQ(c1.bags_freed, c0.bags_freed);
    EXPECT_EQ(c1.bumps, c0.bumps);
    EXPECT_EQ(a1.free_nodes, 0u);
    EXPECT_EQ(a1.retired, 5u);
    EXPECT_EQ(a1.bagged, 0u);
    EXPECT_GE(a1.retired + a1.bagged, a0.retired + a0.bagged);
    EXPECT_EQ(a1.handle_refs, 0);
    EXPECT_TRUE(a1.consistent);
    {
        ContractSet::handle h = set.new_handle();
        ASSERT_EQ(a1.reachable_dead, 0u);
        EXPECT_EQ(set.advance(), (AdvanceResult{0, 0, 0, kContractBuckets, a1.reachable, Advance::advanced}));
    } // h's scope
    const ContractSet::InternalAccounting a2 = set.get_internal_accounting();
    EXPECT_EQ(a2.bagged, 5u);
    EXPECT_EQ(a2.free_nodes, 0u);
    EXPECT_EQ(set.get_internal_counters().nodes_freed, c0.nodes_freed);
    EXPECT_EQ(set.advance(), (AdvanceResult{5, 1, 0, kContractBuckets, 5, Advance::nothing_retired}));
    EXPECT_TRUE(setting.holds());
} // ContractAdvanceFreesNothingWithoutACall

// Reuse before growth (testable form: with arena_shards = 1, after an
// advance() that freed f nodes the next f allocations take free nodes:
// get_internal_node_count() unchanged, free_nodes down by f).
TEST(ConcurrentHashSetRcuTest, ContractReuseBeforeGrowth) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    for (int k = 1; k <= 3; ++k) contract_retire(set, k);
    EXPECT_TRUE(contract_advance_is(set, 3, 1, Advance::advanced));
    const size_t nodes0 = set.get_internal_node_count();
    const size_t free0 = set.get_internal_accounting().free_nodes;
    EXPECT_EQ(free0, 3u);
    {
        ContractSet::handle h = set.new_handle();
        for (int k = 10; k < 13; ++k) {
            EXPECT_TRUE(set.insert(h, k));
            EXPECT_EQ(set.get_internal_node_count(), nodes0) << "key " << k;
        }
    } // h's scope
    EXPECT_EQ(set.get_internal_accounting().free_nodes, free0 - 3);
    EXPECT_TRUE(setting.holds());
} // ContractReuseBeforeGrowth

// Limbo COUNTS: a limbo-only state is not nothing_retired; a throwing reuse
// sends its node to limbo, reused only after a later advance(). One shard, so
// a freed node is on the only free list and the next allocation takes it
// (reuse before growth). An insert armed to throw on its next copy
// assignment reuses that node, the assignment throws, the node goes to limbo
// and the exception propagates; the key is absent (a throw before the
// publishing CAS, from the copy assignment on a reused node); the handle stays
// live and usable. An insert after that must not take the limbo node (reused
// only after a later advance()): it appends (node count + 1) and limbo
// stays 1. A call then bags the limbo node with the one retired node and,
// with no handle, frees both: {2, 1, ...}. The limbo-only state is rebuilt
// the same way; with the retired lists empty the call is `advanced`, not
// `nothing_retired`: {1, 1, 0, B, 0, advanced}.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceLimboOnlyIsNotNothingRetired) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    contract_retire(set, ContractCountedKey(1));
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::advanced));
    ASSERT_EQ(set.get_internal_accounting().free_nodes, 1u);
    {
        SetT::handle h = set.new_handle();
        const ContractCountedKey y(2);
        ContractCountedKey::throw_on_assign = true;
        EXPECT_THROW(set.insert(h, y), ContractAssignThrow);
        EXPECT_FALSE(ContractCountedKey::throw_on_assign) << "the insert did not reuse the free node (reuse before growth)";
        ContractCountedKey::throw_on_assign = false;
        EXPECT_TRUE(h);
        EXPECT_FALSE(h.refresh());
        EXPECT_FALSE(set.contains(h, y));
        const SetT::InternalAccounting a = set.get_internal_accounting();
        EXPECT_EQ(a.limbo, 1u);
        EXPECT_EQ(a.retired, 0u);
        EXPECT_EQ(a.free_nodes, 0u);
        EXPECT_TRUE(a.consistent);
        const size_t nodes0 = set.get_internal_node_count();
        EXPECT_TRUE(set.insert(h, ContractCountedKey(4)));
        EXPECT_EQ(set.get_internal_node_count(), nodes0 + 1);
        EXPECT_EQ(set.get_internal_limbo_count(), 1u);
        EXPECT_TRUE(set.erase(h, ContractCountedKey(4)));
    } // the handle's scope
    EXPECT_TRUE(contract_advance_is(set, 2, 1, Advance::advanced));
    EXPECT_EQ(set.get_internal_limbo_count(), 0u);
    EXPECT_EQ(set.get_internal_accounting().free_nodes, 2u);

    ContractCountedKey::throw_on_assign = true;
    {
        SetT::handle h = set.new_handle();
        EXPECT_THROW(set.insert(h, ContractCountedKey(5)), ContractAssignThrow);
    }
    ContractCountedKey::throw_on_assign = false;
    const SetT::InternalAccounting before = set.get_internal_accounting();
    ASSERT_EQ(before.limbo, 1u);
    ASSERT_EQ(before.retired, 0u);
    EXPECT_TRUE(contract_advance_is(set, 1, 1, Advance::advanced));
    const SetT::InternalAccounting after = set.get_internal_accounting();
    EXPECT_EQ(after.limbo, 0u);
    EXPECT_EQ(after.free_nodes, before.free_nodes + 1);
    EXPECT_TRUE(after.consistent);
    EXPECT_TRUE(setting.holds());
} // ContractAdvanceLimboOnlyIsNotNothingRetired

// A handle's generation never becomes invalid: one of the first generation,
// never refreshed, keeps the
// operations' full contract across 100 advance() calls (63 `advanced`, then
// `ring_full` under its own pin): every operation under it returns what items
// 1-3 say, before and after each call.
TEST(ConcurrentHashSetRcuTest, ContractOldHandleKeepsFullContractAcrossAdvances) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractSet::handle old = set.new_handle();
    for (int k = 1000; k < 1010; ++k) EXPECT_TRUE(set.insert(old, k));
    for (int round = 0; round < 100; ++round) {
        EXPECT_TRUE(set.insert(old, round)) << "round " << round;
        EXPECT_TRUE(set.erase(old, round)) << "round " << round;
        const AdvanceResult r = set.advance();
        EXPECT_EQ(r.advance, round < 63 ? Advance::advanced : Advance::ring_full) << "round " << round;
        EXPECT_FALSE(set.contains(old, round)) << "round " << round;
        for (int k = 1000; k < 1010; ++k) EXPECT_TRUE(set.contains(old, k)) << "round " << round;
        EXPECT_FALSE(set.insert(old, 1000 + round%10)) << "round " << round;
    } // loop over rounds
    EXPECT_TRUE(old);
} // ContractOldHandleKeepsFullContractAcrossAdvances

// Membership is unchanged by any number of concurrent advance() calls, under
// handles of any generation, shared, refreshed or moved, beside joins that
// never wait and a single advancer at a time.
// Writers churn insert/erase on disjoint key sets (each knows its own keys'
// membership exactly and checks every return against it, items 1-3) while the
// table doubles; writers 0 and 1 SHARE one handle, refreshed only in the
// barrier's completion step, where neither uses it (refresh is exclusive
// with other uses of its handle);
// writer 2 refreshes its own handle, writer 3 move-assigns a new handle over
// its live one; a reader checks stable keys present and never-inserted keys
// absent under its own refreshed handle. Two advancers call advance() all the
// time (one holding a handle refreshed before each call, one holding none) on
// a ring of 8, and check each result's own invariants (`contended` is all
// zero; `ring_full` walks nothing; every other outcome walks at least one
// bucket). At the end, with every handle released: membership exact; then the
// drain (an uncontended call leaves no bypassable straggler): after one
// uncontended call, the next call's walk
// retires nothing, the retired, limbo and bag populations are empty and the
// accounting is consistent; a third call is `nothing_retired` freeing nothing.
TEST(ConcurrentHashSetRcuTest, ContractMembershipExactAcrossConcurrentAdvances) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash, 8>;
    constexpr int kWriters = 4;
    constexpr int kKeysPerWriter = 512;
    constexpr int kOpsPerWriter = 20000;
    constexpr int kStable = 256;
    constexpr int kAbsentBase = 1 << 24;
    SetT set(4);
    {
        SetT::handle h = set.new_handle();
        for (int i = 0; i < kStable; ++i) ASSERT_TRUE(set.insert(h, -1 - i));
    }
    std::unique_ptr<SetT::handle> shared = std::make_unique<SetT::handle>(set.new_handle());
    std::barrier phase(2, [&]() noexcept { (void)shared->refresh(); });
    std::atomic<int> errors{0};
    std::atomic<int> bad_results{0};
    std::atomic<int> writers_left{kWriters};
    std::atomic<size_t> outcomes[4] = {};
    std::vector<std::vector<char>> final_present(kWriters);

    auto writer = [&](int t) {
        std::unique_ptr<SetT::handle> own;
        if (t >= 2) own = std::make_unique<SetT::handle>(set.new_handle());
        std::mt19937 rng(t + 1);
        std::vector<char> present(kKeysPerWriter, 0);
        for (int op = 0; op < kOpsPerWriter; ++op) {
            const SetT::handle& h = t < 2 ? *shared : *own;
            const int i = static_cast<int>(rng()%kKeysPerWriter);
            const int key = t + kWriters*i;
            switch (rng()%3) {
                case 0:
                    if (set.insert(h, key) != !present[i]) errors.fetch_add(1, std::memory_order_relaxed);
                    present[i] = 1;
                    break;
                case 1:
                    if (set.erase(h, key) != static_cast<bool>(present[i])) errors.fetch_add(1, std::memory_order_relaxed);
                    present[i] = 0;
                    break;
                default:
                    if (set.contains(h, key) != static_cast<bool>(present[i])) errors.fetch_add(1, std::memory_order_relaxed);
                    break;
            } // switch on the operation
            if ((op + 1)%256 == 0) {
                if (t < 2) {
                    phase.arrive_and_wait();
                } else if (t == 2) {
                    (void)own->refresh();
                } else {
                    *own = set.new_handle();
                }
            } // every 256 operations
        } // loop over operations
        final_present[t] = std::move(present);
        writers_left.fetch_sub(1, std::memory_order_release);
    }; // writer

    auto reader = [&] {
        SetT::handle h = set.new_handle();
        while (writers_left.load(std::memory_order_acquire) > 0) {
            for (int i = 0; i < kStable; ++i) {
                if (!set.contains(h, -1 - i)) errors.fetch_add(1, std::memory_order_relaxed);
                if (set.contains(h, kAbsentBase + i)) errors.fetch_add(1, std::memory_order_relaxed);
            }
            (void)h.refresh();
        } // while the writers run
    }; // reader

    auto advancer = [&](int a) {
        std::unique_ptr<SetT::handle> mine;
        if (a == 1) mine = std::make_unique<SetT::handle>(set.new_handle());
        size_t local[4] = {};
        while (writers_left.load(std::memory_order_acquire) > 0) {
            if (mine) (void)mine->refresh();
            const AdvanceResult r = set.advance();
            ++local[static_cast<int>(r.advance)];
            if (r.advance == Advance::contended && r != kContended) {
                bad_results.fetch_add(1, std::memory_order_relaxed);
            }
            if (r.advance == Advance::ring_full && (r.walk_retired != 0 || r.buckets_walked != 0 || r.nodes_passed != 0)) {
                bad_results.fetch_add(1, std::memory_order_relaxed);
            }
            if ((r.advance == Advance::advanced || r.advance == Advance::nothing_retired) && r.buckets_walked == 0) {
                bad_results.fetch_add(1, std::memory_order_relaxed);
            }
        } // loop until the writers finish
        for (int o = 0; o < 4; ++o) outcomes[o].fetch_add(local[o], std::memory_order_relaxed);
    }; // advancer

    std::vector<std::thread> threads;
    for (int t = 0; t < kWriters; ++t) threads.emplace_back(writer, t);
    threads.emplace_back(reader);
    threads.emplace_back(advancer, 0);
    threads.emplace_back(advancer, 1);
    for (std::thread& t : threads) t.join();
    shared.reset();

    std::printf("[   INFO   ] advance() outcomes: nothing_retired %zu, advanced %zu, ring_full %zu, contended %zu\n",
                outcomes[static_cast<int>(Advance::nothing_retired)].load(), outcomes[static_cast<int>(Advance::advanced)].load(),
                outcomes[static_cast<int>(Advance::ring_full)].load(), outcomes[static_cast<int>(Advance::contended)].load());
    EXPECT_EQ(errors.load(), 0);
    EXPECT_EQ(bad_results.load(), 0);
    EXPECT_GT(outcomes[static_cast<int>(Advance::advanced)].load(), 0u) << "test sanity: no call ever advanced";
    {
        SetT::handle h = set.new_handle();
        for (int t = 0; t < kWriters; ++t) {
            for (int i = 0; i < kKeysPerWriter; ++i) {
                EXPECT_EQ(set.contains(h, t + kWriters*i), static_cast<bool>(final_present[t][i])) << "writer " << t << " key index " << i;
            }
        } // loop over the writers
        for (int i = 0; i < kStable; ++i) {
            EXPECT_TRUE(set.contains(h, -1 - i));
            EXPECT_FALSE(set.contains(h, kAbsentBase + i));
        }
    } // the checking handle's scope
    EXPECT_EQ(set.get_internal_accounting().handle_refs, 0);
    const AdvanceResult c1 = set.advance();
    EXPECT_NE(c1.advance, Advance::contended);
    EXPECT_NE(c1.advance, Advance::ring_full);
    const AdvanceResult c2 = set.advance();
    EXPECT_EQ(c2.walk_retired, 0u) << "an uncontended call left a dead run its successor's walk could retire";
    const SetT::InternalAccounting acc = set.get_internal_accounting();
    EXPECT_TRUE(acc.consistent);
    EXPECT_EQ(acc.retired, 0u);
    EXPECT_EQ(acc.limbo, 0u);
    EXPECT_EQ(acc.bagged, 0u);
    EXPECT_EQ(acc.handle_refs, 0);
    const AdvanceResult c3 = set.advance();
    EXPECT_EQ(c3.advance, Advance::nothing_retired);
    EXPECT_EQ(c3.freed_nodes, 0u);
    EXPECT_EQ(c3.freed_bags, 0u);
    EXPECT_EQ(c3.walk_retired, 0u);
} // ContractMembershipExactAcrossConcurrentAdvances


// ===========================================================================
// advance()'s reads of stored values and its walk's Hash{}, and the tests
// that need the walk inside Hash{} (the single advancer, the walk's
// exception guarantee, joins that never wait).
// ===========================================================================

// The negative: on a quiescent set that merely doubled, advance() calls
// Hash{} NOT AT ALL (uncontended doublings and operations leave no FROZEN
// node); advance() destroys and assigns nothing, and makes no ==, copy or
// assignment. Five doublings, each followed by uncontended erases and
// re-inserts of every 7th key (so that bags and free nodes exist), then an
// advance() measured with a counting Hash{} and the value counters.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceOnMerelyDoubledSetCallsNoHash) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    SetT set(64, 1);
    const size_t size0 = set.get_internal_table_size();
    int next_key = 0;
    for (int round = 0; round < 5; ++round) {
        contract_grow_once<ContractCountedKey>(set, next_key);
        {
            SetT::handle h = set.new_handle();
            for (int k = round; k < next_key; k += 7) EXPECT_TRUE(set.erase(h, ContractCountedKey(k)));
            for (int k = round; k < next_key; k += 7) EXPECT_TRUE(set.insert(h, ContractCountedKey(k)));
        } // h's scope
        const size_t d0 = ContractCountedKey::destructor_calls.load();
        const size_t e0 = ContractCountedKey::equality_calls.load();
        const size_t a0 = ContractCountedKey::assign_calls.load();
        const size_t c0 = ContractCountedKey::copy_calls.load();
        ContractHashControl::mode = ContractHashMode::count;
        ContractHashControl::calls = 0;
        const AdvanceResult r = set.advance();
        ContractHashControl::mode = ContractHashMode::plain;
        EXPECT_EQ(ContractHashControl::calls, 0u) << "round " << round;
        EXPECT_EQ(ContractCountedKey::destructor_calls.load(), d0) << "round " << round;
        EXPECT_EQ(ContractCountedKey::equality_calls.load(), e0) << "round " << round;
        EXPECT_EQ(ContractCountedKey::assign_calls.load(), a0) << "round " << round;
        EXPECT_EQ(ContractCountedKey::copy_calls.load(), c0) << "round " << round;
        EXPECT_NE(r.advance, Advance::contended);
    } // loop over doublings
    EXPECT_GE(set.get_internal_table_size(), 32*size0) << "test sanity: the set did not double five times";
} // ContractAdvanceOnMerelyDoubledSetCallsNoHash

// The one read of a stored value by advance() is Hash{} on a FROZEN node at
// a usable pred (no ==, copy, assignment or destruction), and the walk does
// hash in the unfinished-split construction, built through the public
// interface: an advance() on a
// set holding an unfinished split calls Hash{} at least once and touches no
// value otherwise.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceWalkReadsValuesOnlyThroughHash) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    SetT set(64, 1);
    int next_key = 0;
    std::vector<int> thrown;
    ASSERT_TRUE(contract_leave_unfinished_split<ContractCountedKey>(set, next_key, thrown))
        << "construction: no insert throwing inside a split left a state whose walk calls Hash{} (an unfinished split)";
    const size_t d0 = ContractCountedKey::destructor_calls.load();
    const size_t e0 = ContractCountedKey::equality_calls.load();
    const size_t a0 = ContractCountedKey::assign_calls.load();
    const size_t c0 = ContractCountedKey::copy_calls.load();
    ContractHashControl::mode = ContractHashMode::count;
    ContractHashControl::calls = 0;
    const AdvanceResult r = set.advance();
    ContractHashControl::mode = ContractHashMode::plain;
    EXPECT_GE(ContractHashControl::calls, 1u);
    EXPECT_EQ(ContractCountedKey::destructor_calls.load(), d0);
    EXPECT_EQ(ContractCountedKey::equality_calls.load(), e0);
    EXPECT_EQ(ContractCountedKey::assign_calls.load(), a0);
    EXPECT_EQ(ContractCountedKey::copy_calls.load(), c0);
    EXPECT_NE(r.advance, Advance::contended);
    EXPECT_EQ(contract_membership_errors<ContractCountedKey>(set, next_key, thrown), 0);
} // ContractAdvanceWalkReadsValuesOnlyThroughHash

// One advancer at a time (`contended`: {0, 0, 0, 0, 0, contended}), and
// joins never wait: while one advance() is in progress (parked in its walk's
// Hash{}, the unfinished-split construction), another call returns
// `contended` at once having done nothing; new_handle() and refresh() return
// (they never wait for an
// advance(); the outcome of the refresh is not asserted, the parked call may
// or may not have published), and operations run with their full contract.
// The parked call, released, is not `contended`, the next call is not
// `contended` either (the flag was released), and membership is exact.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceContendedWhileAnotherIsParked) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    SetT set(64, 1);
    int next_key = 0;
    std::vector<int> thrown;
    SetT::handle old = set.new_handle();
    ContractParkedAdvance<SetT, int> parked(set, next_key, thrown);
    ASSERT_TRUE(parked.parked()) << "construction: no parked walk (the unfinished split was not reached or the walk did not hash)";
    EXPECT_EQ(set.advance(), kContended);
    EXPECT_EQ(set.advance(), kContended);
    SetT::handle h = set.new_handle();
    EXPECT_TRUE(h);
    (void)old.refresh();
    EXPECT_TRUE(old);
    EXPECT_TRUE(set.insert(h, -5));
    EXPECT_TRUE(set.contains(old, -5));
    EXPECT_TRUE(set.erase(old, -5));
    EXPECT_FALSE(set.contains(h, -5));
    const AdvanceResult r = parked.resume();
    EXPECT_NE(r.advance, Advance::contended);
    EXPECT_NE(set.advance().advance, Advance::contended);
    EXPECT_EQ(contract_membership_errors<int>(set, next_key, thrown), 0);
} // ContractAdvanceContendedWhileAnotherIsParked

// Nested calls: from inside Hash{} while advance()'s walk is calling it
// (the unfinished-split construction), a nested advance() returns `contended`
// and a nested
// operation under a handle of its own runs (it is allowed); from inside
// Hash{} called by an OPERATION, a nested advance() proceeds like any other
// call (not `contended`).
TEST(ConcurrentHashSetRcuTest, ContractAdvanceNestedCalls) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    SetT set(64, 1);
    int next_key = 0;
    std::vector<int> thrown;
    ASSERT_TRUE(contract_leave_unfinished_split<int>(set, next_key, thrown))
        << "construction: no insert throwing inside a split left a state whose walk calls Hash{} (an unfinished split)";
    AdvanceResult nested{};
    bool ran = false;
    bool nested_contains = true;
    ContractHashControl::nested = [&] {
        nested = set.advance();
        nested_contains = set.contains(set.new_handle(), -77);
        ran = true;
    }; // the nested action
    const AdvanceResult outer = set.advance();
    ContractHashControl::nested = nullptr;
    ASSERT_TRUE(ran) << "the walk did not call Hash{}";
    EXPECT_EQ(nested, kContended);
    EXPECT_FALSE(nested_contains);
    EXPECT_NE(outer.advance, Advance::contended);

    AdvanceResult from_operation{};
    ran = false;
    ContractHashControl::nested = [&] {
        from_operation = set.advance();
        ran = true;
    };
    {
        SetT::handle h = set.new_handle();
        EXPECT_TRUE(set.insert(h, -78));
    }
    ContractHashControl::nested = nullptr;
    ASSERT_TRUE(ran);
    EXPECT_NE(from_operation.advance, Advance::contended);
    EXPECT_TRUE(set.contains(set.new_handle(), -78));
} // ContractAdvanceNestedCalls

// A throw from Hash{} inside the walk (the unfinished-split construction: BASIC
// guarantee): the call throws (the result is lost); afterwards the try-flag
// is free (the next call is `advanced` or `nothing_retired`, never
// `contended`; get_internal_generation() is callable, its precondition being
// no advance() in progress), the walk's internal handle has left
// (handle_refs == 0 with no caller handle), the accounting is consistent, and
// membership is unchanged.
TEST(ConcurrentHashSetRcuTest, ContractAdvanceWalkThrowLeavesSetUsable) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    SetT set(64, 1);
    int next_key = 0;
    std::vector<int> thrown;
    ASSERT_TRUE(contract_throw_in_walk<int>(set, next_key, thrown)) << "construction: the walk did not call the throwing Hash{} (no unfinished split)";
    (void)set.get_internal_generation();
    const SetT::InternalAccounting acc = set.get_internal_accounting();
    EXPECT_EQ(acc.handle_refs, 0);
    EXPECT_TRUE(acc.consistent);
    const AdvanceResult next = set.advance();
    EXPECT_TRUE(next.advance == Advance::advanced || next.advance == Advance::nothing_retired) << testing::PrintToString(next);
    EXPECT_NE(set.advance().advance, Advance::contended);
    EXPECT_EQ(contract_membership_errors<int>(set, next_key, thrown), 0);
    EXPECT_TRUE(set.get_internal_accounting().consistent);
} // ContractAdvanceWalkThrowLeavesSetUsable

// ===========================================================================
// The destructor.
// ===========================================================================

// One constructed T per arena slot: the destructor destroys exactly
// get_internal_node_count() values, each once, in whatever population:
// reachable, free, retired, or in a bag pending at destruction; nothing
// leaked. Under ASan the free nodes are destroyed without a report
// (~ContractCountedKey reads its own bytes). advance() destroys nothing.
// One-node setting, one shard: 10 inserts append 10 slots; 3 erased and freed
// by a call with no handle; 5 erased and bagged by a call under a handle
// (pinned); 1 erased after it (retired); key 9 reachable. Populations checked
// exactly through the accounting sweep; then the destruction count, and every
// object constructed in the test destroyed exactly once (per serial).
TEST(ConcurrentHashSetRcuTest, ContractDestroySetWithPendingBags) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    const unsigned first = ContractCountedKey::next_serial.load();
    {
        std::unique_ptr<SetT> set = std::make_unique<SetT>(kContractBuckets, kContractShards);
        const ContractOneNode<SetT> setting(*set);
        {
            SetT::handle h = set->new_handle();
            for (int k = 0; k < 10; ++k) EXPECT_TRUE(set->insert(h, ContractCountedKey(k)));
            for (int k = 0; k < 3; ++k) EXPECT_TRUE(set->erase(h, ContractCountedKey(k)));
        } // h's scope
        size_t d0 = ContractCountedKey::destructor_calls.load();
        const AdvanceResult r1 = set->advance();
        EXPECT_EQ(ContractCountedKey::destructor_calls.load(), d0);
        EXPECT_EQ(r1, (AdvanceResult{3, 1, 0, kContractBuckets, 7, Advance::advanced}));
        {
            SetT::handle h = set->new_handle();
            for (int k = 3; k < 8; ++k) EXPECT_TRUE(set->erase(h, ContractCountedKey(k)));
            d0 = ContractCountedKey::destructor_calls.load();
            const AdvanceResult r2 = set->advance();
            EXPECT_EQ(ContractCountedKey::destructor_calls.load(), d0);
            EXPECT_EQ(r2, (AdvanceResult{0, 0, 0, kContractBuckets, 2, Advance::advanced}));
            EXPECT_EQ(set->get_internal_bag_count(SetT::get_internal_handle_generation(h)), 5u)
                << "ring view: the pending bag is h's generation's";
            EXPECT_TRUE(set->erase(h, ContractCountedKey(8)));
        } // the pinning handle's scope
        const SetT::InternalAccounting acc = set->get_internal_accounting();
        EXPECT_TRUE(acc.consistent);
        EXPECT_EQ(acc.reachable, 1u);
        EXPECT_EQ(acc.free_nodes, 3u);
        EXPECT_EQ(acc.bagged, 5u);
        EXPECT_EQ(acc.retired, 1u);
        EXPECT_EQ(acc.limbo, 0u);
        EXPECT_EQ(acc.handle_refs, 0);
        EXPECT_TRUE(setting.holds());
        const size_t nodes = set->get_internal_node_count();
        EXPECT_EQ(nodes, 10u);
        d0 = ContractCountedKey::destructor_calls.load();
        set.reset();
        EXPECT_EQ(ContractCountedKey::destructor_calls.load() - d0, nodes);
    } // the set's scope
    const unsigned last = ContractCountedKey::next_serial.load();
    ASSERT_LT(last, ContractCountedKey::kMaxSerials);
    EXPECT_EQ(contract_destruction_errors(first, last), 0);
} // ContractDestroySetWithPendingBags

// ===========================================================================
// The debug-diagnosed precondition violations of the handle and the
// destructor. Each death statement is paired with the legal use of the same
// objects in the parent; SIGABRT with glibc's "Assertion" message is required,
// so a crash or a sanitizer report cannot pass for the diagnosis.
// ===========================================================================

#ifndef NDEBUG
// Destroying the set while one of its handles is live.
TEST(ConcurrentHashSetRcuDeathTest, ContractDestroySetWithLiveHandleAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    {
        ContractSet* set = new ContractSet(kContractBuckets, kContractShards);
        ContractSet::handle* h = new ContractSet::handle(set->new_handle());
        delete h;
        delete set;
    } // the set's scope
    EXPECT_EXIT({
        ContractSet* set = new ContractSet(kContractBuckets, kContractShards);
        [[maybe_unused]] ContractSet::handle* leaked = new ContractSet::handle(set->new_handle());
        delete set;
    }, testing::KilledBySignal(SIGABRT), "Assertion");
} // ContractDestroySetWithLiveHandleAborts

// Destroying the set while an advance() is in progress (parked in its walk's
// Hash{}, the unfinished-split construction). legal == true: the same
// construction, the parked call released and joined before the destruction.
// Returns whether the call parked (the construction's non-vacuity, checked in
// the parent).
static bool contract_destroy_during_parked_advance(bool legal) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    SetT* set = new SetT(64, 1);
    int next_key = 0;
    std::vector<int> thrown;
    ContractParkedAdvance<SetT, int>* parked = new ContractParkedAdvance<SetT, int>(*set, next_key, thrown);
    const bool was_parked = parked->parked();
    if (legal || !was_parked) {
        parked->resume();
        delete parked;
        delete set;
        return was_parked;
    } // if legal or not parked
    parked->detach();
    delete set;
    return was_parked;
} // contract_destroy_during_parked_advance()

TEST(ConcurrentHashSetRcuDeathTest, ContractDestroySetDuringAdvanceAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ASSERT_TRUE(contract_destroy_during_parked_advance(true)) << "construction: no parked walk (no unfinished split)";
    EXPECT_EXIT(contract_destroy_during_parked_advance(false), testing::KilledBySignal(SIGABRT), "Assertion");
} // ContractDestroySetDuringAdvanceAborts

// An operation with a live handle of ANOTHER set of the same
// instantiation.
TEST(ConcurrentHashSetRcuDeathTest, ContractForeignHandleAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    {
        ContractSet a(kContractBuckets, kContractShards);
        ContractSet b(kContractBuckets, kContractShards);
        const ContractSet::handle ha = a.new_handle();
        const ContractSet::handle hb = b.new_handle();
        EXPECT_TRUE(b.insert(hb, 1));
        EXPECT_TRUE(b.contains(hb, 1));
        EXPECT_TRUE(b.erase(hb, 1));
        EXPECT_FALSE(a.contains(ha, 1));
    } // the sets' scope
    EXPECT_EXIT({
        ContractSet a(kContractBuckets, kContractShards);
        ContractSet b(kContractBuckets, kContractShards);
        const ContractSet::handle ha = a.new_handle();
        (void)b.contains(ha, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT({
        ContractSet a(kContractBuckets, kContractShards);
        ContractSet b(kContractBuckets, kContractShards);
        const ContractSet::handle ha = a.new_handle();
        (void)b.insert(ha, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT({
        ContractSet a(kContractBuckets, kContractShards);
        ContractSet b(kContractBuckets, kContractShards);
        const ContractSet::handle ha = a.new_handle();
        (void)b.erase(ha, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
} // ContractForeignHandleAborts

// An operation or refresh() with an EMPTY handle (default
// constructed, or moved from).
TEST(ConcurrentHashSetRcuDeathTest, ContractEmptyHandleUseAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    {
        ContractSet set(kContractBuckets, kContractShards);
        ContractSet::handle h = set.new_handle();
        EXPECT_TRUE(set.insert(h, 1));
        EXPECT_FALSE(h.refresh());
    } // the set's scope
    EXPECT_EXIT({
        ContractSet set(kContractBuckets, kContractShards);
        const ContractSet::handle empty;
        (void)set.contains(empty, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT({
        ContractSet set(kContractBuckets, kContractShards);
        const ContractSet::handle empty;
        (void)set.insert(empty, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT({
        ContractSet set(kContractBuckets, kContractShards);
        const ContractSet::handle empty;
        (void)set.erase(empty, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT({
        ContractSet::handle empty;
        (void)empty.refresh();
    }, testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT({
        ContractSet set(kContractBuckets, kContractShards);
        ContractSet::handle h = set.new_handle();
        ContractSet::handle sink(std::move(h));
        (void)set.contains(h, 1);
    }, testing::KilledBySignal(SIGABRT), "Assertion");
} // ContractEmptyHandleUseAborts

// refresh() or a move from a handle while an operation runs
// under it (a contains() parked in the key's operator== on another thread).
// `what` selects the violating use: 0 refresh(), 1 move construction from it;
// legal == true: the parked operation is released and joined first. Returns
// whether the operation parked (checked in the parent).
static bool contract_use_handle_with_operation_in_flight(bool legal, int what) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    SetT set(kContractBuckets, kContractShards);
    SetT::handle h = set.new_handle();
    EXPECT_TRUE(set.insert(h, ContractCountedKey(1)));
    const ContractCountedKey probe(1);
    ContractHashControl::reset();
    std::atomic<bool> done{false};
    std::thread op([&] {
        ContractCountedKey::park_on_equality = true;
        (void)set.contains(h, probe);
        ContractCountedKey::park_on_equality = false;
        done.store(true, std::memory_order_release);
    }); // thread op
    while (!ContractHashControl::parked.load(std::memory_order_acquire) && !done.load(std::memory_order_acquire)) std::this_thread::yield();
    const bool was_parked = ContractHashControl::parked.load(std::memory_order_acquire);
    if (legal || !was_parked) {
        ContractHashControl::resume.store(true, std::memory_order_release);
        op.join();
    }
    if (what == 0) {
        (void)h.refresh();
    } else {
        SetT::handle moved(std::move(h));
    }
    if (op.joinable()) {
        ContractHashControl::resume.store(true, std::memory_order_release);
        op.join();
    }
    return was_parked;
} // contract_use_handle_with_operation_in_flight()

TEST(ConcurrentHashSetRcuDeathTest, ContractUseOfHandleWithOperationInFlightAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ASSERT_TRUE(contract_use_handle_with_operation_in_flight(true, 0)) << "construction: contains() never called operator==";
    ASSERT_TRUE(contract_use_handle_with_operation_in_flight(true, 1));
    EXPECT_EXIT(contract_use_handle_with_operation_in_flight(false, 0), testing::KilledBySignal(SIGABRT), "Assertion");
    EXPECT_EXIT(contract_use_handle_with_operation_in_flight(false, 1), testing::KilledBySignal(SIGABRT), "Assertion");
} // ContractUseOfHandleWithOperationInFlightAborts
#endif // NDEBUG

// ===========================================================================
// CONTRACT TESTS: reclaim() (the bump and the free pass without the walk),
// the Options constructor, and the opt-in automatic cadence, through the
// public interface.
//
// Every test names, in words, the promises it checks. They use the fixtures
// of the contract tests above (ContractSet, ContractSetK, ContractSetNoDel,
// kContractBuckets, kContractShards, ContractOneNode, contract_retire(),
// contract_advance_is(), contract_ring_invariant(), the hooked Hash{} and key
// types, the unfinished-split and parked-advance constructions, kContended,
// Advance and its PrintTo).
//
// THE MODEL the exact oracles below follow, in addition to the model stated
// at the top of the contract tests above:
//   - reclaim() is advance() WITHOUT THE WALK: the same try-flag, the same
//     `nothing_retired` decision first, the same ring-wrap free pass and
//     `ring_full`, the same bump and final free pass; so its result is
//     advance()'s with walk_retired == buckets_walked == nodes_passed == 0 for
//     EVERY outcome. The generation goes up by one per `advanced` of either
//     call.
//   - A CADENCED set (Options::auto_reclaim_every != 0) has an effective
//     cadence R = bit_ceil(min(value, 2^63)). Each retire push that takes its
//     arena shard's retired-node count across a multiple of R TRIGGERS once,
//     right after the push, on the retiring thread, under the retiring
//     operation's handle: the trigger does reclaim()'s work. Single-threaded,
//     with no explicit call in progress, every trigger PROCEEDS (auto_reclaims
//     + 1; it is never skipped) and does what reclaim() would do at that
//     point.
//   - THE CADENCE SETTING (the condition of every exact per-erase count): the
//     ONE-NODE SETTING's construction (kContractBuckets buckets, checked by
//     ContractOneNode), ONE arena shard, erases of present keys on one thread.
//     Then every successful erase retires exactly one node (retire_nodes +1,
//     retire_runs +1), and the one shard's count is the set-wide count. Every
//     cadence script checks the setting with ContractOneNode and the
//     retire_nodes/retire_runs deltas of each step.
//   - The trigger's own handle is live while it runs: the bag a
//     trigger closes is pinned by the retiring operation's handle, so a
//     trigger frees only bags of generations older than that handle's. With a
//     long-lived handle h refreshed between rounds, each round's trigger frees
//     the PREVIOUS round's bag; never refreshed, h pins every bag its
//     triggers close, and the ring fills.
//   - Free-list counts follow reuse before growth: with one arena shard,
//     every allocation (an insert that misses) pops a freed node before it
//     appends, so an insert after a free pass takes the free count down by
//     one while it is non-zero. The expected free counts below say where.
//   - Diagnostics used: the RACE-FREE counters (get_internal_counters()), the
//     per-shard retired-node counts and the effective cadence, and, at
//     quiescent points only (no operation in flight, no explicit call in
//     progress), the ring views and the accounting sweep.
// ===========================================================================

// The clamp of the cadence: values above it are clamped to it.
static constexpr size_t kContractCadenceClamp = size_t{1} << 63;

// Options for a set in the CADENCE SETTING with the requested (not yet
// rounded) cadence `every`: kContractBuckets buckets and one arena shard.
template <typename SetT>
static typename SetT::Options contract_cadence_options(size_t every) {
    return typename SetT::Options{.initial_capacity = kContractBuckets, .arena_shards = kContractShards, .auto_reclaim_every = every};
} // contract_cadence_options()

// The counter deltas one step of a cadence script expects; every field is
// compared, so a step that expects nothing of a counter expects it unchanged.
struct ContractCounterDelta {
    size_t retire_nodes = 0;    // nodes retired (pushed onto a shard's retired list)
    size_t retire_runs = 0;     // runs retired (retire pushes)
    size_t bumps = 0;           // publishes, of any cause (explicit call or trigger)
    size_t auto_reclaims = 0;   // triggers that acquired the flag
    size_t auto_skipped = 0;    // triggers that found the flag held
    size_t ring_full = 0;       // `ring_full` outcomes, of any cause
    size_t bags_freed = 0;      // bags freed, by any cause
    size_t nodes_freed = 0;     // nodes freed, by any cause
    bool operator==(const ContractCounterDelta&) const = default;
}; // struct ContractCounterDelta

// Prints a ContractCounterDelta in failure messages.
static std::ostream& operator<<(std::ostream& os, const ContractCounterDelta& d) {
    return os << "{retire_nodes " << d.retire_nodes << ", retire_runs " << d.retire_runs << ", bumps " << d.bumps << ", auto_reclaims "
              << d.auto_reclaims << ", auto_skipped " << d.auto_skipped << ", ring_full " << d.ring_full << ", bags_freed " << d.bags_freed
              << ", nodes_freed " << d.nodes_freed << "}";
} // operator<<(ContractCounterDelta)

// Compares the deltas between two counter snapshots with `want`. Also expects
// `contended` and `joins` unchanged: the scripts bracket only operations and
// reclaim() calls at quiescent points -- every handle they use is created
// BEFORE the first snapshot -- and none of them is refused (an explicit call
// at quiescence is never `contended`, and `contended` never counts a
// trigger) or joins (`joins` counts no trigger and no reclaim(), and an
// operation joins nothing). The counters are never reset, so after - before
// is the step's count. Returns an AssertionResult so
// that the caller's line is reported.
template <typename Counters>
static testing::AssertionResult contract_deltas_are(const Counters& before, const Counters& after, const ContractCounterDelta& want) {
    const ContractCounterDelta got{static_cast<size_t>(after.retire_nodes - before.retire_nodes),
                                   static_cast<size_t>(after.retire_runs - before.retire_runs),
                                   static_cast<size_t>(after.bumps - before.bumps),
                                   static_cast<size_t>(after.auto_reclaims - before.auto_reclaims),
                                   static_cast<size_t>(after.auto_skipped - before.auto_skipped),
                                   static_cast<size_t>(after.ring_full - before.ring_full),
                                   static_cast<size_t>(after.bags_freed - before.bags_freed),
                                   static_cast<size_t>(after.nodes_freed - before.nodes_freed)};
    const size_t contended = static_cast<size_t>(after.contended - before.contended);
    const size_t joins = static_cast<size_t>(after.joins - before.joins);
    if (got == want && contended == 0 && joins == 0) return testing::AssertionSuccess();
    return testing::AssertionFailure() << "counter deltas " << got << ", expected " << want << "; contended +" << contended << ", joins +"
                                       << joins << " (expected +0 each)";
} // contract_deltas_are()

// Expects the accounting sweep (quiescent only) to be consistent and to hold
// exactly `retired` nodes on the retired lists, `bagged` in the bags and
// `free_nodes` on the free lists.
template <typename SetT>
static testing::AssertionResult contract_sweep_is(const SetT& set, size_t retired, size_t bagged, size_t free_nodes) {
    const typename SetT::InternalAccounting acc = set.get_internal_accounting();
    if (acc.consistent && acc.retired == retired && acc.bagged == bagged && acc.free_nodes == free_nodes) return testing::AssertionSuccess();
    return testing::AssertionFailure() << "sweep: consistent " << acc.consistent << ", retired " << acc.retired << " (expected " << retired
                                       << "), bagged " << acc.bagged << " (expected " << bagged << "), free " << acc.free_nodes
                                       << " (expected " << free_nodes << ")";
} // contract_sweep_is()

// The cadence's set-wide invariants at a quiescent point, and the definition
// of the per-shard view: one entry per arena shard, summing to retire_nodes.
// With R > 0 the triggers (proceeding + skipped) are the crossing pushes, at
// most floor(retire_nodes_s / R) on each shard s; with `exact`, which a test
// passes when every push retired one node (the CADENCE SETTING), exactly
// that many. At R = 1 every push crosses, so the triggers equal retire_runs.
// On a cadence-off set (R = 0) both counters are 0.
template <typename SetT>
static testing::AssertionResult contract_cadence_invariants(const SetT& set, bool exact) {
    const typename SetT::InternalCounters c = set.get_internal_counters();
    const size_t r = set.get_internal_auto_reclaim_every();
    const std::vector<size_t> per_shard = set.get_internal_retire_nodes_per_shard();
    size_t sum = 0;
    size_t crossings_bound = 0;
    for (size_t n : per_shard) {
        sum += n;
        if (r != 0) crossings_bound += n/r;
    } // loop over the shards
    const size_t triggers = static_cast<size_t>(c.auto_reclaims + c.auto_skipped);
    if (per_shard.size() != set.get_internal_arena_shards()) {
        return testing::AssertionFailure() << "per-shard view has " << per_shard.size() << " entries for " << set.get_internal_arena_shards() << " shards";
    }
    if (sum != c.retire_nodes) {
        return testing::AssertionFailure() << "per-shard retired nodes sum to " << sum << ", retire_nodes is " << c.retire_nodes;
    }
    if (r == 0) {
        if (triggers == 0) return testing::AssertionSuccess();
        return testing::AssertionFailure() << "cadence-off set counted " << triggers << " triggers";
    }
    if (triggers > crossings_bound || (exact && triggers != crossings_bound)) {
        return testing::AssertionFailure() << "triggers " << triggers << " (auto_reclaims " << c.auto_reclaims << " + auto_skipped "
                                           << c.auto_skipped << ") vs sum over shards of floor(retire_nodes/R) " << crossings_bound
                                           << (exact ? " (expected equal)" : " (expected at most)") << ", R " << r;
    }
    if (r == 1 && triggers != c.retire_runs) {
        return testing::AssertionFailure() << "R = 1: triggers " << triggers << " != retire_runs " << c.retire_runs;
    }
    return testing::AssertionSuccess();
} // contract_cadence_invariants()

// ===========================================================================
// reclaim(): its declaration, outcomes, literals and the no-walk witnesses.
// ===========================================================================

// reclaim() returns AdvanceResult, is noexcept, and is NOT [[nodiscard]]
// (the discarded calls below would fail -Werror otherwise);
// it exists for both AllowDelete values and any ring size.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeDeclaredSurface) {
    static_assert(std::is_same_v<decltype(std::declval<ContractSet&>().reclaim()), AdvanceResult>);
    static_assert(noexcept(std::declval<ContractSet&>().reclaim()));
    static_assert(noexcept(std::declval<ContractSetNoDel&>().reclaim()));
    static_assert(noexcept(std::declval<ContractSetK<2>&>().reclaim()));
    ContractSet set(kContractBuckets, kContractShards);
    set.reclaim();
    ContractSetNoDel nodel(kContractBuckets, kContractShards);
    nodel.reclaim();
} // ContractBumpAndFreeDeclaredSurface

// A fresh set into which nothing was ever inserted returns {0, 0, 0, 0, 0,
// nothing_retired} -- whatever its capacity, since reclaim() never walks
// (contrast advance()'s {0, 0, 0, B, 0, ...}) -- with or without a live
// handle, again and again, for both AllowDelete values; no slot consumed
// (generation 0).
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeFreshSetIsNothingRetired) {
    constexpr AdvanceResult kNothing{0, 0, 0, 0, 0, Advance::nothing_retired};
    for (size_t capacity : {size_t{1}, size_t{5}, size_t{64}, kContractBuckets}) {
        ContractSet set(capacity, kContractShards);
        EXPECT_EQ(set.reclaim(), kNothing) << "capacity " << capacity;
        EXPECT_EQ(set.reclaim(), kNothing) << "capacity " << capacity;
        {
            ContractSet::handle h = set.new_handle();
            EXPECT_EQ(set.reclaim(), kNothing) << "capacity " << capacity;
        }
        EXPECT_EQ(set.get_internal_generation(), 0u);
        EXPECT_TRUE(contract_ring_invariant(set, 64));

        ContractSetNoDel nodel(capacity, kContractShards);
        EXPECT_EQ(nodel.reclaim(), kNothing) << "capacity " << capacity;
        EXPECT_EQ(nodel.get_internal_generation(), 0u);
    } // loop over capacities
} // ContractBumpAndFreeFreshSetIsNothingRetired

// One node retired, every handle released: {1, 1, 0, 0, 0, advanced} (as
// advance()'s, without the walk) -- the bag the
// call closes is freed in the same call -- and the generation goes 0 -> 1;
// then nothing retired: {0, 0, 0, 0, 0, nothing_retired}, generation
// unchanged. Three nodes retired between two calls form ONE bag:
// {3, 1, 0, 0, 0, advanced}. The diagnostic counters: bumps/bags_freed/
// nodes_freed count reclaim()'s outcomes as they count advance()'s, joins
// does not move across a reclaim() (no walk joins), and the cadence counters
// stay 0 on a cadence-off set. One-node setting (every erase retires exactly
// one node; checked on the first: retire_nodes +1). Free count at the end:
// 4 nodes freed, of which the insert of key 2 reused one (reuse before
// growth): 3.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeReleasedHandleFreesInSameCall) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::InternalCounters c0{};
    ContractSet::InternalCounters c1{};
    {
        ContractSet::handle h = set.new_handle();
        EXPECT_TRUE(set.insert(h, 1));
        c0 = set.get_internal_counters();
        EXPECT_TRUE(set.erase(h, 1));
        c1 = set.get_internal_counters();
    } // the retiring handle's scope, released before the call
    EXPECT_TRUE(contract_deltas_are(c0, c1, {.retire_nodes = 1, .retire_runs = 1}));
    EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::advanced}));
    ContractSet::InternalCounters c2 = set.get_internal_counters();
    EXPECT_TRUE(contract_deltas_are(c1, c2, {.bumps = 1, .bags_freed = 1, .nodes_freed = 1}));
    EXPECT_EQ(set.get_internal_generation(), 1u);
    EXPECT_EQ(set.reclaim(), (AdvanceResult{0, 0, 0, 0, 0, Advance::nothing_retired}));
    EXPECT_TRUE(contract_deltas_are(c2, set.get_internal_counters(), {}));
    EXPECT_EQ(set.get_internal_generation(), 1u);

    contract_retire(set, 2);
    contract_retire(set, 3);
    contract_retire(set, 4);
    c0 = set.get_internal_counters();
    EXPECT_EQ(set.reclaim(), (AdvanceResult{3, 1, 0, 0, 0, Advance::advanced}));
    EXPECT_TRUE(contract_deltas_are(c0, set.get_internal_counters(), {.bumps = 1, .bags_freed = 1, .nodes_freed = 3}));
    EXPECT_EQ(set.get_internal_generation(), 2u);
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, 3));
    EXPECT_TRUE(contract_ring_invariant(set, 64));
    EXPECT_TRUE(contract_cadence_invariants(set, true));
    EXPECT_TRUE(setting.holds());
} // ContractBumpAndFreeReleasedHandleFreesInSameCall

// The two-call latency, as advance()'s without the walk: a long-lived handle
// refreshed before EACH call. Per round: one node retired; refresh()
// is the fast path (false: the handle is current); the call closes the bag
// the handle pins: {0, 0, 0, 0, 0, advanced}; refresh() moves (true: an
// `advanced` call happened-before it); the next call frees the bag:
// {1, 1, 0, 0, 0, nothing_retired} (nothing retired between the calls).
// Five rounds: the steady state, not just the first.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeHeldHandleTwoCallLatency) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle h = set.new_handle();
    for (int round = 0; round < 5; ++round) {
        contract_retire(set, round);
        EXPECT_FALSE(h.refresh()) << "round " << round;
        EXPECT_EQ(set.reclaim(), (AdvanceResult{0, 0, 0, 0, 0, Advance::advanced})) << "round " << round;
        EXPECT_EQ(set.get_internal_bag_count(ContractSet::get_internal_handle_generation(h)), 1u) << "round " << round;
        EXPECT_TRUE(h.refresh()) << "round " << round;
        EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::nothing_retired})) << "round " << round;
    } // loop over rounds
    EXPECT_EQ(set.get_internal_generation(), 5u);
    EXPECT_TRUE(setting.holds());
} // ContractBumpAndFreeHeldHandleTwoCallLatency

// `ring_full`, with advance()'s ring arithmetic holding for reclaim() and for
// any mix of the two calls, and its counts. A ring of K slots, one handle pinned at
// generation 0, one node retired before each call: calls 1..K-1 return
// `advanced` and free nothing; with `alternate`, the odd calls are advance()
// ({0, 0, 0, B, 0, advanced}) and the even ones reclaim() ({0, 0, 0, 0, 0,
// advanced}). One more node retired: reclaim() returns {0, 0, 0, 0, 0,
// ring_full} (the free passes free nothing while the pin holds; the window
// between them is not a target and the pin never leaves here), and so does
// advance() right after it (one protocol; a ring_full advance() does not walk
// either); no slot consumed (generation K-1), the node waits on the retired
// list, K-1 bags pending. Recovery by refresh: {K-1, K-1, 0, 0, 0, advanced}
// (the refreshed handle pins the bag the call closes, the one node), then the
// handle released: {1, 1, 0, 0, 0, nothing_retired}. Recovery by release:
// {K, K, 0, 0, 0, advanced}. The ring invariant holds after every call.
template <size_t K>
static void contract_bump_ring_full_then_recover(bool alternate, bool by_refresh) {
    using SetT = ContractSetK<K>;
    SCOPED_TRACE(testing::Message() << "kGenerations " << K << (alternate ? ", alternating advance()/reclaim()" : ", reclaim() only")
                                    << (by_refresh ? ", recovery by refresh" : ", recovery by release"));
    SetT set(kContractBuckets, kContractShards);
    const ContractOneNode<SetT> setting(set);
    std::unique_ptr<typename SetT::handle> pin = std::make_unique<typename SetT::handle>(set.new_handle());
    int key = 0;
    for (size_t call = 1; call < K; ++call) {
        contract_retire(set, key++);
        if (alternate && call%2 == 1) {
            EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::advanced)) << "call " << call;
        } else {
            EXPECT_EQ(set.reclaim(), (AdvanceResult{0, 0, 0, 0, 0, Advance::advanced})) << "call " << call;
        }
        EXPECT_TRUE(contract_ring_invariant(set, K)) << "call " << call;
    } // calls 1 .. K-1
    contract_retire(set, key++);
    const typename SetT::InternalCounters c0 = set.get_internal_counters();
    EXPECT_EQ(set.reclaim(), (AdvanceResult{0, 0, 0, 0, 0, Advance::ring_full})) << "call " << K;
    EXPECT_TRUE(contract_advance_is(set, 0, 0, Advance::ring_full)) << "call " << K + 1;
    const typename SetT::InternalCounters c1 = set.get_internal_counters();
    EXPECT_EQ(c1.ring_full - c0.ring_full, 2u);
    EXPECT_EQ(c1.bumps - c0.bumps, 0u);
    EXPECT_EQ(set.get_internal_generation(), K - 1);
    EXPECT_TRUE(contract_ring_invariant(set, K));
    EXPECT_TRUE(contract_sweep_is(set, 1, K - 1, 0));
    if (by_refresh) {
        EXPECT_TRUE(pin->refresh());
        EXPECT_EQ(set.reclaim(), (AdvanceResult{K - 1, K - 1, 0, 0, 0, Advance::advanced}));
        EXPECT_TRUE(contract_ring_invariant(set, K));
        pin.reset();
        EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::nothing_retired}));
    } else {
        pin.reset();
        EXPECT_EQ(set.reclaim(), (AdvanceResult{K, K, 0, 0, 0, Advance::advanced}));
    }
    EXPECT_TRUE(contract_ring_invariant(set, K));
    EXPECT_EQ(set.get_internal_generation(), K);
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, K));
    EXPECT_TRUE(setting.holds());
} // contract_bump_ring_full_then_recover()

// reclaim() alone: `ring_full` under a held pin and both recoveries, at the
// minimum ring (2), a small one (4) and the default (64).
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeRingFullWithPinHeld) {
    contract_bump_ring_full_then_recover<2>(false, true);
    contract_bump_ring_full_then_recover<2>(false, false);
    contract_bump_ring_full_then_recover<4>(false, true);
    contract_bump_ring_full_then_recover<4>(false, false);
    contract_bump_ring_full_then_recover<64>(false, true);
    contract_bump_ring_full_then_recover<64>(false, false);
} // ContractBumpAndFreeRingFullWithPinHeld

// The same with advance() and reclaim() alternating (one protocol).
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeRingFullMixedWithAdvance) {
    contract_bump_ring_full_then_recover<3>(true, true);
    contract_bump_ring_full_then_recover<4>(true, false);
    contract_bump_ring_full_then_recover<64>(true, true);
    contract_bump_ring_full_then_recover<64>(true, false);
} // ContractBumpAndFreeRingFullMixedWithAdvance

// A fixed script of inserts, erases, handle moves and explicit calls of both
// kinds on a set in the one-node setting (kContractBuckets buckets, one
// shard), each call checked against its literal and appended to `results`.
// Expected values, from the model above, reclaim() being advance() without
// the walk (B = kContractBuckets; n = the reachable nodes before an
// advance(), checked by the sweep before each advance() that has keys
// present):
//    1. fresh: reclaim() {0,0,0,0,0,NR}; advance() {0,0,0,B,0,NR}
//    2. one node retired, no handle: reclaim() {1,1,0,0,0,ADV}        gen 1
//    3. one node retired, no handle: advance() {1,1,0,B,0,ADV}         gen 2
//    4. h joins (gen 2); keys 100..104 inserted, 100 erased under h:
//       reclaim() {0,0,0,0,0,ADV}: h pins the bag it closes            gen 3
//    5. 101 erased under h: advance() {0,0,0,B,3,ADV}: the pin is
//       cumulative, 3 keys reachable                                   gen 4
//    6. h refreshed (moves: true): reclaim() {2,2,0,0,0,NR}: the two
//       bags h pinned are freed although nothing was retired
//    7. 102 erased under h; refresh() is the fast path (false):
//       advance() {0,0,0,B,2,ADV}                                       gen 5
//    8. h released: reclaim() {1,1,0,0,0,NR}
//    9. 103, 104 retired, no handle: reclaim() {2,1,0,0,0,ADV}         gen 6
//   10. advance() {0,0,0,B,0,NR}
// A call is never `contended` here (single-threaded, quiescent). Returns
// with no handle of the set live.
template <typename SetT>
static void contract_bump_and_advance_script(SetT& set, std::vector<AdvanceResult>& results) {
    constexpr size_t B = kContractBuckets;
    // Runs one call, records it, and compares it with `want`.
    auto call = [&](bool walk, const AdvanceResult& want, const char* step) {
        const AdvanceResult got = walk ? set.advance() : set.reclaim();
        results.push_back(got);
        EXPECT_EQ(got, want) << step << (walk ? ": advance()" : ": reclaim()");
    }; // call
    // The precondition of an exact nodes_passed: `live` keys reachable, none dead.
    auto expect_reachable = [&](size_t live, const char* step) {
        const typename SetT::InternalAccounting acc = set.get_internal_accounting();
        EXPECT_EQ(acc.reachable, live) << step;
        EXPECT_EQ(acc.reachable_dead, 0u) << step;
    }; // expect_reachable
    call(false, {0, 0, 0, 0, 0, Advance::nothing_retired}, "1");
    call(true, {0, 0, 0, B, 0, Advance::nothing_retired}, "1");
    contract_retire(set, 1);
    call(false, {1, 1, 0, 0, 0, Advance::advanced}, "2");
    contract_retire(set, 2);
    call(true, {1, 1, 0, B, 0, Advance::advanced}, "3");
    {
        std::unique_ptr<typename SetT::handle> h = std::make_unique<typename SetT::handle>(set.new_handle());
        for (int k = 100; k < 105; ++k) EXPECT_TRUE(set.insert(*h, k));
        EXPECT_TRUE(set.erase(*h, 100));
        call(false, {0, 0, 0, 0, 0, Advance::advanced}, "4");
        EXPECT_TRUE(set.erase(*h, 101));
        expect_reachable(3, "5");
        call(true, {0, 0, 0, B, 3, Advance::advanced}, "5");
        EXPECT_TRUE(h->refresh());
        call(false, {2, 2, 0, 0, 0, Advance::nothing_retired}, "6");
        EXPECT_TRUE(set.erase(*h, 102));
        EXPECT_FALSE(h->refresh());
        expect_reachable(2, "7");
        call(true, {0, 0, 0, B, 2, Advance::advanced}, "7");
        h.reset();
        call(false, {1, 1, 0, 0, 0, Advance::nothing_retired}, "8");
    } // h's scope
    {
        typename SetT::handle h = set.new_handle();
        EXPECT_TRUE(set.erase(h, 103));
        EXPECT_TRUE(set.erase(h, 104));
    } // the temporary handle's scope
    call(false, {2, 1, 0, 0, 0, Advance::advanced}, "9");
    call(true, {0, 0, 0, B, 0, Advance::nothing_retired}, "10");
    EXPECT_EQ(set.get_internal_generation(), 6u);
} // contract_bump_and_advance_script()

// advance() keeps its contract exactly beside reclaim(); the two are one
// protocol (the generation +1 per `advanced` of either call, the same bags,
// pins and cumulative free pass); reclaim()'s literals and advance()'s, walk
// counts included, on one set mixing both calls: the script above. Then the
// accounting is consistent, nothing retired or bagged, and the free count is
// the 7 nodes freed minus the 2 that later inserts reused (reuse before
// growth: key 2's insert after step 2, key 100's after step 3): 5.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeMixesWithAdvanceAsOneProtocol) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    std::vector<AdvanceResult> results;
    contract_bump_and_advance_script(set, results);
    EXPECT_EQ(results.size(), 11u);
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, 5));
    EXPECT_TRUE(contract_ring_invariant(set, 64));
    EXPECT_TRUE(setting.holds());
} // ContractBumpAndFreeMixesWithAdvanceAsOneProtocol

// The per-thread timers: across a reclaim(), the calling
// thread's advance_ns, bump_ns and free_ns increase and walk_ns does not;
// `joins` does not move (reclaim() joins nothing; the walk is the one that
// would). advance_ns is entry to return, so it is at least the bump and free
// time it contains. One clock tick may be coarser than one call's bump, so
// the "increase" is asserted over 20 calls, each of which bumps and frees one
// node ({1, 1, 0, 0, 0, advanced}); walk_ns and joins are asserted unchanged
// per call.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeChargesNoWalkTime) {
    ContractSet set(kContractBuckets, kContractShards);
    const ContractOneNode<ContractSet> setting(set);
    using Timers = ContractSet::InternalTimers;
    uint64_t advance_ns = 0;
    uint64_t bump_ns = 0;
    uint64_t free_ns = 0;
    for (int round = 0; round < 20; ++round) {
        contract_retire(set, round);
        const ContractSet::InternalCounters c0 = set.get_internal_counters();
        const Timers t0 = ContractSet::internal_advance_timers();
        EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::advanced})) << "round " << round;
        const Timers t1 = ContractSet::internal_advance_timers();
        const ContractSet::InternalCounters c1 = set.get_internal_counters();
        EXPECT_EQ(t1.walk_ns, t0.walk_ns) << "round " << round << ": reclaim() charged walk time";
        EXPECT_EQ(c1.joins, c0.joins) << "round " << round << ": reclaim() joined";
        EXPECT_GE(t1.advance_ns - t0.advance_ns, (t1.bump_ns - t0.bump_ns) + (t1.free_ns - t0.free_ns)) << "round " << round;
        advance_ns += t1.advance_ns - t0.advance_ns;
        bump_ns += t1.bump_ns - t0.bump_ns;
        free_ns += t1.free_ns - t0.free_ns;
    } // loop over rounds
    EXPECT_GT(advance_ns, 0u);
    EXPECT_GT(bump_ns, 0u);
    EXPECT_GT(free_ns, 0u);
    EXPECT_TRUE(setting.holds());
} // ContractBumpAndFreeChargesNoWalkTime

// `contended`: the flag is shared, so a reclaim() that finds it held by an
// advance() returns {0, 0, 0, 0, 0, contended} at once, having done nothing;
// `contended` counts explicit refusals (+1 per refused call); a `contended`
// reclaim() charges the calling thread's advance_ns and nothing else. The
// advance() is parked inside its walk's Hash{} on another thread (the
// unfinished-split construction of the contract tests above). After it
// resumes, the next reclaim() is not `contended` (the flag was released).
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeContendedWhileAdvanceParked) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    using Timers = SetT::InternalTimers;
    constexpr int kCalls = 10;
    SetT set(64, 1);
    int next_key = 0;
    std::vector<int> thrown;
    ContractParkedAdvance<SetT, int> parked(set, next_key, thrown);
    ASSERT_TRUE(parked.parked()) << "construction: no parked walk (the unfinished split was not reached or the walk did not hash)";
    const SetT::InternalCounters c0 = set.get_internal_counters();
    const Timers t0 = SetT::internal_advance_timers();
    for (int i = 0; i < kCalls; ++i) EXPECT_EQ(set.reclaim(), kContended) << "call " << i;
    const Timers t1 = SetT::internal_advance_timers();
    const SetT::InternalCounters c1 = set.get_internal_counters();
    EXPECT_EQ(c1.contended - c0.contended, static_cast<size_t>(kCalls));
    EXPECT_EQ(c1.bumps, c0.bumps);
    EXPECT_EQ(c1.nodes_freed, c0.nodes_freed);
    EXPECT_GT(t1.advance_ns, t0.advance_ns);
    EXPECT_EQ(t1.bump_ns, t0.bump_ns);
    EXPECT_EQ(t1.free_ns, t0.free_ns);
    EXPECT_EQ(t1.walk_ns, t0.walk_ns);
    const AdvanceResult r = parked.resume();
    EXPECT_NE(r.advance, Advance::contended);
    EXPECT_NE(set.reclaim().advance, Advance::contended);
    EXPECT_EQ(contract_membership_errors<int>(set, next_key, thrown), 0);
} // ContractBumpAndFreeContendedWhileAdvanceParked

// Nested calls, with reclaim() in advance()'s place: from inside
// Hash{} while advance()'s walk is calling it (the unfinished-split
// construction), a nested reclaim() returns `contended`; from inside Hash{}
// called by an OPERATION, a nested reclaim() proceeds like any other call,
// and the operation's own handle pins the generation it closes.
// The second half is exact on a one-node-setting set: one node retired with
// no handle, then h joins (the current generation 0) and an insert under h
// runs reclaim() from its first Hash{} call: {0, 0, 0, 0, 0, advanced} (h
// pins the closed bag); h released, the next call frees it:
// {1, 1, 0, 0, 0, nothing_retired}.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeNestedCalls) {
    using SetT = ConcurrentResizableHashSetRCU<int, true, ContractHookHash>;
    {
        SetT set(64, 1);
        int next_key = 0;
        std::vector<int> thrown;
        ASSERT_TRUE(contract_leave_unfinished_split<int>(set, next_key, thrown))
            << "construction: no insert throwing inside a split left a state whose walk calls Hash{} (an unfinished split)";
        AdvanceResult nested{};
        bool ran = false;
        ContractHashControl::nested = [&] {
            nested = set.reclaim();
            ran = true;
        }; // the nested action
        const AdvanceResult outer = set.advance();
        ContractHashControl::nested = nullptr;
        ASSERT_TRUE(ran) << "the walk did not call Hash{}";
        EXPECT_EQ(nested, kContended);
        EXPECT_NE(outer.advance, Advance::contended);
    } // the walk case
    {
        SetT set(kContractBuckets, kContractShards);
        const ContractOneNode<SetT> setting(set);
        contract_retire(set, 1);
        AdvanceResult from_operation{};
        bool ran = false;
        {
            SetT::handle h = set.new_handle();
            ContractHashControl::nested = [&] {
                from_operation = set.reclaim();
                ran = true;
            }; // the nested action
            EXPECT_TRUE(set.insert(h, 2));
            ContractHashControl::nested = nullptr;
        } // h's scope
        ASSERT_TRUE(ran) << "the insert did not call Hash{}";
        EXPECT_EQ(from_operation, (AdvanceResult{0, 0, 0, 0, 0, Advance::advanced}));
        EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::nothing_retired}));
        EXPECT_TRUE(set.contains(set.new_handle(), 2));
        EXPECT_TRUE(setting.holds());
    } // the operation case
} // ContractBumpAndFreeNestedCalls

// Returns the result of one reclaim() on `set` with the hooked Hash{} counting
// and the counted key's call counters checked unchanged across it.
template <typename SetT>
static AdvanceResult contract_reclaim_reading_nothing(SetT& set, const char* where) {
    const size_t d0 = ContractCountedKey::destructor_calls.load();
    const size_t e0 = ContractCountedKey::equality_calls.load();
    const size_t a0 = ContractCountedKey::assign_calls.load();
    const size_t c0 = ContractCountedKey::copy_calls.load();
    ContractHashControl::mode = ContractHashMode::count;
    ContractHashControl::calls = 0;
    const AdvanceResult r = set.reclaim();
    ContractHashControl::mode = ContractHashMode::plain;
    EXPECT_EQ(ContractHashControl::calls, 0u) << where << ": Hash{} calls";
    EXPECT_EQ(ContractCountedKey::destructor_calls.load(), d0) << where << ": destructions";
    EXPECT_EQ(ContractCountedKey::equality_calls.load(), e0) << where << ": == calls";
    EXPECT_EQ(ContractCountedKey::assign_calls.load(), a0) << where << ": copy assignments";
    EXPECT_EQ(ContractCountedKey::copy_calls.load(), c0) << where << ": copy constructions";
    return r;
} // contract_reclaim_reading_nothing()

// The twin of advance()'s no-value-read test: reclaim() reads NO stored value -- no
// Hash{}, no ==, no copy, no assignment, no destruction. Two states: (1) the
// unfinished-split construction (the one state in which advance()'s walk
// hashes), where the outcome of the call is whatever the construction left
// (not asserted beyond "not contended, no walk counts"); (2) a one-node-setting
// set with one node retired and no handle, where the call bumps and frees:
// {1, 1, 0, 0, 0, advanced}, and still touches no value.
TEST(ConcurrentHashSetRcuTest, ContractBumpAndFreeReadsNoStoredValue) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    {
        SetT set(64, 1);
        int next_key = 0;
        std::vector<int> thrown;
        ASSERT_TRUE(contract_leave_unfinished_split<ContractCountedKey>(set, next_key, thrown))
            << "construction: no insert throwing inside a split left a state whose walk calls Hash{} (an unfinished split)";
        const AdvanceResult r = contract_reclaim_reading_nothing(set, "unfinished split");
        EXPECT_NE(r.advance, Advance::contended);
        EXPECT_EQ(r.walk_retired + r.buckets_walked + r.nodes_passed, 0u) << testing::PrintToString(r);
        EXPECT_EQ(contract_membership_errors<ContractCountedKey>(set, next_key, thrown), 0);
    } // the unfinished-split state
    {
        SetT set(kContractBuckets, kContractShards);
        const ContractOneNode<SetT> setting(set);
        contract_retire(set, ContractCountedKey(1));
        EXPECT_EQ(contract_reclaim_reading_nothing(set, "one node retired"), (AdvanceResult{1, 1, 0, 0, 0, Advance::advanced}));
        EXPECT_TRUE(setting.holds());
    } // the one-node state
} // ContractBumpAndFreeReadsNoStoredValue

// ===========================================================================
// The Options constructor and the effective cadence.
// ===========================================================================

// Options is a nested aggregate of three size_t fields in the order
// initial_capacity, arena_shards, auto_reclaim_every, defaults {4, 0, 0},
// usable with designated initializers and positionally; a plain copyable,
// movable, trivially copyable and destructible type.
TEST(ConcurrentHashSetRcuTest, ContractCadenceOptionsIsAPlainAggregate) {
    using Options = ContractSet::Options;
    static_assert(std::is_aggregate_v<Options>);
    static_assert(std::is_trivially_copyable_v<Options>);
    static_assert(std::is_trivially_destructible_v<Options>);
    static_assert(std::is_nothrow_copy_constructible_v<Options> && std::is_nothrow_move_constructible_v<Options>);
    static_assert(std::is_same_v<decltype(Options::initial_capacity), size_t>);
    static_assert(std::is_same_v<decltype(Options::arena_shards), size_t>);
    static_assert(std::is_same_v<decltype(Options::auto_reclaim_every), size_t>);
    const Options defaults{};
    EXPECT_EQ(defaults.initial_capacity, 4u);
    EXPECT_EQ(defaults.arena_shards, 0u);
    EXPECT_EQ(defaults.auto_reclaim_every, 0u);
    const Options designated{.auto_reclaim_every = 8192};
    EXPECT_EQ(designated.initial_capacity, 4u);
    EXPECT_EQ(designated.arena_shards, 0u);
    EXPECT_EQ(designated.auto_reclaim_every, 8192u);
    const Options positional{4096, 1, 8};
    EXPECT_EQ(positional.initial_capacity, 4096u);
    EXPECT_EQ(positional.arena_shards, 1u);
    EXPECT_EQ(positional.auto_reclaim_every, 8u);
    const Options copy = positional;
    EXPECT_EQ(copy.auto_reclaim_every, 8u);
} // ContractCadenceOptionsIsAPlainAggregate

// The effective cadence, through
// get_internal_auto_reclaim_every(): 0 means OFF (decided before any
// rounding: bit_ceil(0) would be 1); otherwise min(value, 2^63) rounded up to
// a power of two. Partitions: 0; powers of two (unchanged: 1, 2, 4, 2^62,
// 2^63); non-powers below the clamp (rounded up: 3 -> 4, 5 -> 8, 1000 ->
// 1024, 2^62 + 1 -> 2^63, 2^63 - 1 -> 2^63); above the clamp (clamped:
// 2^63 + 1, SIZE_MAX - 1, SIZE_MAX -> 2^63). The two-parameter constructor in
// every spelling reads 0 (the cadence off). For three instantiations
// (the value does not depend on T, AllowDelete or the ring). Each set's
// cadence counters start at 0.
TEST(ConcurrentHashSetRcuTest, ContractCadenceEffectiveValueIsRoundedAndClamped) {
    constexpr size_t kTop = kContractCadenceClamp;
    const std::pair<size_t, size_t> cases[] = {
        {0, 0}, {1, 1}, {2, 2}, {3, 4}, {4, 4}, {5, 8}, {1000, 1024},
        {kTop >> 1, kTop >> 1}, {(kTop >> 1) + 1, kTop}, {kTop - 1, kTop}, {kTop, kTop},
        {kTop + 1, kTop}, {std::numeric_limits<size_t>::max() - 1, kTop}, {std::numeric_limits<size_t>::max(), kTop},
    }; // requested -> effective
    for (const std::pair<size_t, size_t>& c : cases) {
        ContractSet set(contract_cadence_options<ContractSet>(c.first));
        EXPECT_EQ(set.get_internal_auto_reclaim_every(), c.second) << "requested " << c.first;
        const ContractSet::InternalCounters counters = set.get_internal_counters();
        EXPECT_EQ(counters.auto_reclaims, 0u) << "requested " << c.first;
        EXPECT_EQ(counters.auto_skipped, 0u) << "requested " << c.first;
        ContractSetNoDel nodel(contract_cadence_options<ContractSetNoDel>(c.first));
        EXPECT_EQ(nodel.get_internal_auto_reclaim_every(), c.second) << "requested " << c.first << ", AllowDelete false";
        ContractSetK<2> small_ring(contract_cadence_options<ContractSetK<2>>(c.first));
        EXPECT_EQ(small_ring.get_internal_auto_reclaim_every(), c.second) << "requested " << c.first << ", kGenerations 2";
    } // loop over requested cadences
    ContractSet two_parameters(kContractBuckets, kContractShards);
    EXPECT_EQ(two_parameters.get_internal_auto_reclaim_every(), 0u);
    ContractSet one_parameter(kContractBuckets);
    EXPECT_EQ(one_parameter.get_internal_auto_reclaim_every(), 0u);
    ContractSet defaulted;
    EXPECT_EQ(defaulted.get_internal_auto_reclaim_every(), 0u);
    ContractSet default_options(ContractSet::Options{});
    EXPECT_EQ(default_options.get_internal_auto_reclaim_every(), 0u);
} // ContractCadenceEffectiveValueIsRoundedAndClamped

// The capacity and shard rules, the same for both constructors: Set(c, s)
// and Set(Options{c, s, 0}) give the same get_internal_table_size() and
// get_internal_arena_shards() on a fresh set, and both follow them: capacity
// below 4 becomes 4, else rounded up to a power of two; shards 0 means
// hardware_concurrency() (1 if that is 0), capped at 2^16, rounded up to a
// power of two. A non-zero cadence changes neither (it "changes no
// allocation").
TEST(ConcurrentHashSetRcuTest, ContractCadenceOptionsConstructorSizesAsTwoParameterOne) {
    const unsigned hc = std::thread::hardware_concurrency();
    const size_t default_shards = std::bit_ceil(std::min<size_t>(hc == 0 ? 1 : hc, size_t{1} << 16));
    const std::pair<size_t, size_t> capacities[] = {{0, 4}, {1, 4}, {3, 4}, {4, 4}, {5, 8}, {100, 128}, {4096, 4096}};
    const std::pair<size_t, size_t> shards[] = {{0, default_shards}, {1, 1}, {3, 4}, {5, 8}};
    for (const std::pair<size_t, size_t>& c : capacities) {
        for (const std::pair<size_t, size_t>& s : shards) {
            SCOPED_TRACE(testing::Message() << "capacity " << c.first << ", shards " << s.first);
            ContractSet two(c.first, s.first);
            ContractSet off(ContractSet::Options{c.first, s.first, 0});
            ContractSet cadenced(ContractSet::Options{c.first, s.first, 3});
            EXPECT_EQ(two.get_internal_table_size(), c.second);
            EXPECT_EQ(off.get_internal_table_size(), c.second);
            EXPECT_EQ(cadenced.get_internal_table_size(), c.second);
            EXPECT_EQ(two.get_internal_arena_shards(), s.second);
            EXPECT_EQ(off.get_internal_arena_shards(), s.second);
            EXPECT_EQ(cadenced.get_internal_arena_shards(), s.second);
            EXPECT_EQ(cadenced.get_internal_auto_reclaim_every(), 4u);
        } // loop over shard counts
    } // loop over capacities
} // ContractCadenceOptionsConstructorSizesAsTwoParameterOne

// A set built through Options with cadence 0 is indistinguishable from one
// built with the
// two parameters, WITH erases and explicit calls of both kinds: the same
// script (contract_bump_and_advance_script(), whose every call is also checked
// against its literal) gives the same whole AdvanceResult literals on both,
// the same table size and shard count, and auto_reclaims == auto_skipped == 0
// at the end on both. A cadence of 0 rounded to 1 instead would trigger at
// every erase and change the explicit calls' results.
TEST(ConcurrentHashSetRcuTest, ContractCadenceZeroEqualsTwoParameterConstructor) {
    ContractSet two(kContractBuckets, kContractShards);
    ContractSet off(contract_cadence_options<ContractSet>(0));
    const ContractOneNode<ContractSet> two_setting(two);
    const ContractOneNode<ContractSet> off_setting(off);
    std::vector<AdvanceResult> two_results;
    std::vector<AdvanceResult> off_results;
    {
        SCOPED_TRACE("two-parameter constructor");
        contract_bump_and_advance_script(two, two_results);
    }
    {
        SCOPED_TRACE("Options constructor, cadence 0");
        contract_bump_and_advance_script(off, off_results);
    }
    ASSERT_EQ(off_results.size(), two_results.size());
    for (size_t i = 0; i < two_results.size(); ++i) EXPECT_EQ(off_results[i], two_results[i]) << "call " << i;
    EXPECT_EQ(off.get_internal_table_size(), two.get_internal_table_size());
    EXPECT_EQ(off.get_internal_arena_shards(), two.get_internal_arena_shards());
    for (const ContractSet* set : {&two, &off}) {
        const ContractSet::InternalCounters c = set->get_internal_counters();
        EXPECT_EQ(c.auto_reclaims, 0u);
        EXPECT_EQ(c.auto_skipped, 0u);
        EXPECT_EQ(set->get_internal_auto_reclaim_every(), 0u);
        EXPECT_TRUE(contract_cadence_invariants(*set, true));
    } // loop over the two sets
    EXPECT_TRUE(two_setting.holds());
    EXPECT_TRUE(off_setting.holds());
} // ContractCadenceZeroEqualsTwoParameterConstructor

// ===========================================================================
// The cadence scripts (CADENCE SETTING, one thread, no explicit call in
// progress).
// ===========================================================================

// Requested cadence -> effective R, the partitions the scripts run over:
// R = 1 (every erase triggers), small powers of two, and non-powers that the
// rounding moves (3 -> 4, 5 -> 8).
static constexpr std::pair<size_t, size_t> kContractCadences[] = {{1, 1}, {2, 2}, {3, 4}, {4, 4}, {5, 8}, {8, 8}, {16, 16}};

// The cadence's first round, its exact counting invariant and the per-shard
// view: with a handle h joined before the first erase, erases 1..R-1
// trigger nothing (bumps +0, auto_reclaims + auto_skipped +0, nothing freed;
// each retires exactly one node, the shard's entry +1 per erase) and leave
// R-1 nodes on the retired list; the R-th erase triggers and the trigger
// PROCEEDS with an `advanced` bump: bumps +1, auto_reclaims +1, auto_skipped
// +0, the retired list empty, the bag of the generation g it closed holding
// all R nodes -- the crossing run included (a trigger placed before the push
// would leave it on the retired list) -- the generation g + 1, and nothing
// freed (h pins the closed bag). Inserts retire nothing in the setting.
// A requested 3 fires at the 4th erase and a requested 5 at the 8th (the
// rounding up to a power of two).
TEST(ConcurrentHashSetRcuTest, ContractCadenceRMinusOneErasesFreeNothingRthBagsThem) {
    for (const std::pair<size_t, size_t>& cadence : kContractCadences) {
        const size_t r = cadence.second;
        SCOPED_TRACE(testing::Message() << "requested cadence " << cadence.first << ", R " << r);
        ContractSet set(contract_cadence_options<ContractSet>(cadence.first));
        ASSERT_EQ(set.get_internal_auto_reclaim_every(), r);
        const ContractOneNode<ContractSet> setting(set);
        std::unique_ptr<ContractSet::handle> h = std::make_unique<ContractSet::handle>(set.new_handle());
        ContractSet::InternalCounters before = set.get_internal_counters();
        for (int k = 0; k < static_cast<int>(r); ++k) EXPECT_TRUE(set.insert(*h, k));
        EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {})) << "the inserts";
        const uint64_t g = set.get_internal_generation();
        for (size_t i = 1; i < r; ++i) {
            before = set.get_internal_counters();
            EXPECT_TRUE(set.erase(*h, static_cast<int>(i - 1)));
            const ContractSet::InternalCounters after = set.get_internal_counters();
            EXPECT_TRUE(contract_deltas_are(before, after, {.retire_nodes = 1, .retire_runs = 1})) << "erase " << i;
            const std::vector<size_t> per_shard = set.get_internal_retire_nodes_per_shard();
            ASSERT_EQ(per_shard.size(), 1u);
            EXPECT_EQ(per_shard[0], after.retire_nodes) << "erase " << i;
        } // erases 1 .. R-1
        EXPECT_TRUE(contract_sweep_is(set, r - 1, 0, 0)) << "after R-1 erases";
        EXPECT_EQ(set.get_internal_generation(), g);
        before = set.get_internal_counters();
        EXPECT_TRUE(set.erase(*h, static_cast<int>(r - 1)));
        EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.retire_nodes = 1, .retire_runs = 1, .bumps = 1, .auto_reclaims = 1}))
            << "the R-th erase";
        EXPECT_TRUE(contract_sweep_is(set, 0, r, 0)) << "after the R-th erase";
        EXPECT_EQ(set.get_internal_bag_count(g), r);
        EXPECT_EQ(set.get_internal_generation(), g + 1);
        EXPECT_TRUE(contract_ring_invariant(set, 64));
        EXPECT_TRUE(contract_cadence_invariants(set, true));
        EXPECT_TRUE(setting.holds());
    } // loop over cadences
} // ContractCadenceRMinusOneErasesFreeNothingRthBagsThem

// Each round's trigger frees the PREVIOUS round's bag,
// given a refresh of h between rounds. Round 0: R erases, the R-th trigger
// closes the first bag (h pins it, nothing freed). Rounds 1..4: h.refresh()
// (moves: the trigger's publish happened-before it), R
// erases; the R-th trigger frees the previous round's bag (bags_freed +1,
// nodes_freed + R, free nodes + R) and closes this round's, which h now pins
// (bagged == R, the bag of the generation it closed == R). Round 5 without
// the refresh: its trigger frees nothing (h pins its own previous bag)
// and bagged == 2R. Then h released: nothing is freed by the release (no
// handle's leave frees anything), and one explicit reclaim() frees both
// bags: {2R, 2, 0, 0, 0, nothing_retired}.
TEST(ConcurrentHashSetRcuTest, ContractCadenceEachTriggerFreesThePreviousBagAfterRefresh) {
    for (const std::pair<size_t, size_t>& cadence : {std::pair<size_t, size_t>{1, 1}, {3, 4}, {8, 8}}) {
        const size_t r = cadence.second;
        SCOPED_TRACE(testing::Message() << "requested cadence " << cadence.first << ", R " << r);
        ContractSet set(contract_cadence_options<ContractSet>(cadence.first));
        const ContractOneNode<ContractSet> setting(set);
        std::unique_ptr<ContractSet::handle> h = std::make_unique<ContractSet::handle>(set.new_handle());
        constexpr int kRounds = 6;
        for (int k = 0; k < kRounds*static_cast<int>(r); ++k) EXPECT_TRUE(set.insert(*h, k));
        int next = 0;
        for (int round = 0; round < kRounds; ++round) {
            const bool refreshed = round > 0 && round < kRounds - 1;
            if (refreshed) {
                EXPECT_TRUE(h->refresh()) << "round " << round;
                EXPECT_EQ(ContractSet::get_internal_handle_generation(*h), set.get_internal_generation()) << "round " << round;
            }
            for (size_t i = 1; i < r; ++i) {
                const ContractSet::InternalCounters before = set.get_internal_counters();
                EXPECT_TRUE(set.erase(*h, next++));
                EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.retire_nodes = 1, .retire_runs = 1}))
                    << "round " << round << ", erase " << i;
            } // erases 1 .. R-1 of the round
            const uint64_t g = set.get_internal_generation();
            const ContractSet::InternalCounters before = set.get_internal_counters();
            EXPECT_TRUE(set.erase(*h, next++));
            const size_t freed_bags = refreshed ? 1 : 0;
            EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(),
                                            {.retire_nodes = 1, .retire_runs = 1, .bumps = 1, .auto_reclaims = 1, .bags_freed = freed_bags,
                                             .nodes_freed = freed_bags*r}))
                << "round " << round << ", the R-th erase";
            const size_t freed_so_far = round < kRounds - 1 ? static_cast<size_t>(round)*r : static_cast<size_t>(kRounds - 2)*r;
            const size_t bagged = round < kRounds - 1 ? r : 2*r;
            EXPECT_TRUE(contract_sweep_is(set, 0, bagged, freed_so_far)) << "round " << round;
            EXPECT_EQ(set.get_internal_bag_count(g), r) << "round " << round;
            EXPECT_EQ(set.get_internal_generation(), g + 1) << "round " << round;
            EXPECT_TRUE(contract_ring_invariant(set, 64)) << "round " << round;
        } // loop over rounds
        const size_t freed_before_release = static_cast<size_t>(kRounds - 2)*r;
        h.reset();
        EXPECT_TRUE(contract_sweep_is(set, 0, 2*r, freed_before_release)) << "after the release";
        EXPECT_EQ(set.reclaim(), (AdvanceResult{2*r, 2, 0, 0, 0, Advance::nothing_retired}));
        EXPECT_TRUE(contract_sweep_is(set, 0, 0, freed_before_release + 2*r)) << "after the explicit call";
        EXPECT_TRUE(contract_cadence_invariants(set, true));
        EXPECT_TRUE(setting.holds());
    } // loop over cadences
} // ContractCadenceEachTriggerFreesThePreviousBagAfterRefresh

// After one round (R erases under h, the R-th trigger
// closed a bag of R nodes that h pins), an explicit reclaim() at the
// quiescent point -- never `contended` -- with h refreshed first frees that
// bag: {R, 1, 0, 0, 0, nothing_retired}; with h still at the old generation
// it frees nothing: {0, 0, 0, 0, 0, nothing_retired}. An explicit call is
// not a trigger: auto_reclaims and auto_skipped do not move, and with
// nothing retired it does not bump.
TEST(ConcurrentHashSetRcuTest, ContractCadenceExplicitCallAfterFirstRound) {
    for (const std::pair<size_t, size_t>& cadence : {std::pair<size_t, size_t>{1, 1}, {3, 4}, {8, 8}}) {
        for (bool refresh : {true, false}) {
            const size_t r = cadence.second;
            SCOPED_TRACE(testing::Message() << "R " << r << (refresh ? ", h refreshed" : ", h at the old generation"));
            ContractSet set(contract_cadence_options<ContractSet>(cadence.first));
            const ContractOneNode<ContractSet> setting(set);
            ContractSet::handle h = set.new_handle();
            for (int k = 0; k < static_cast<int>(r); ++k) EXPECT_TRUE(set.insert(h, k));
            for (int k = 0; k < static_cast<int>(r); ++k) EXPECT_TRUE(set.erase(h, k));
            EXPECT_TRUE(contract_sweep_is(set, 0, r, 0)) << "after the first round";
            if (refresh) {
                EXPECT_TRUE(h.refresh());
            }
            const ContractSet::InternalCounters before = set.get_internal_counters();
            const size_t freed = refresh ? r : 0;
            EXPECT_EQ(set.reclaim(), (AdvanceResult{freed, refresh ? 1u : 0u, 0, 0, 0, Advance::nothing_retired}));
            EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.bags_freed = refresh ? 1u : 0u, .nodes_freed = freed}));
            EXPECT_TRUE(contract_sweep_is(set, 0, r - freed, freed));
            EXPECT_TRUE(setting.holds());
        } // loop over refreshed / not
    } // loop over cadences
} // ContractCadenceExplicitCallAfterFirstRound

// On a cadenced set (R = 4) at quiescent points, advance() and reclaim()
// keep their literals -- never `contended` --
// and share one protocol with the triggers. Handles are released after each
// operation (contract_retire()); the shard's retired-node count is absolute,
// so the triggers fire at the 4th and 8th node retired, whatever
// explicit calls happened in between:
//   nodes 1, 2 retired (no trigger);       advance() {2,1,0,B,0,ADV}  gen 1
//   advance() {0,0,0,B,0,NR}; reclaim() {0,0,0,0,0,NR}
//   nodes 3, 4 retired: node 4 triggers, closing gen 1 (bag 2, pinned by the
//     retiring operation's handle, freed by nobody yet)          gen 2
//   advance() {2,1,0,B,0,NR}: that bag freed, nothing retired since
//   nodes 5, 6, 7 retired;                  reclaim() {3,1,0,0,0,ADV} gen 3
//   node 8 retired: triggers, closing gen 3 (bag 1, pinned)        gen 4
//   five keys inserted for good;            advance() {1,1,0,B,5,NR}
// auto_reclaims == 2 and auto_skipped == 0 at the end (the walks retired
// nothing: no dead node is ever reachable here). Free counts (inserts
// reuse freed nodes first): 0 right after node 4's trigger (nodes 3 and 4's
// inserts reused the 2 freed by the first advance()), 1 at the end (8 nodes
// freed; the inserts of keys 3, 4, 5, 6, 8, 100 and 101 each reused one).
TEST(ConcurrentHashSetRcuTest, ContractCadenceAdvanceUnchangedAtQuiescence) {
    constexpr size_t B = kContractBuckets;
    ContractSet set(contract_cadence_options<ContractSet>(4));
    const ContractOneNode<ContractSet> setting(set);
    contract_retire(set, 1);
    contract_retire(set, 2);
    EXPECT_EQ(set.get_internal_counters().auto_reclaims, 0u);
    EXPECT_EQ(set.advance(), (AdvanceResult{2, 1, 0, B, 0, Advance::advanced}));
    EXPECT_EQ(set.advance(), (AdvanceResult{0, 0, 0, B, 0, Advance::nothing_retired}));
    EXPECT_EQ(set.reclaim(), (AdvanceResult{0, 0, 0, 0, 0, Advance::nothing_retired}));
    EXPECT_EQ(set.get_internal_generation(), 1u);
    contract_retire(set, 3);
    // Retires `key` like contract_retire(), checking that its erase alone
    // retires one node and triggers one proceeding, publishing bump.
    auto retire_crossing = [&](int key) {
        ContractSet::handle h = set.new_handle();
        EXPECT_TRUE(set.insert(h, key));
        const ContractSet::InternalCounters before = set.get_internal_counters();
        EXPECT_TRUE(set.erase(h, key));
        EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.retire_nodes = 1, .retire_runs = 1, .bumps = 1, .auto_reclaims = 1}))
            << "node " << key;
    }; // retire_crossing
    retire_crossing(4);
    EXPECT_EQ(set.get_internal_generation(), 2u);
    EXPECT_TRUE(contract_sweep_is(set, 0, 2, 0));
    EXPECT_EQ(set.advance(), (AdvanceResult{2, 1, 0, B, 0, Advance::nothing_retired}));
    contract_retire(set, 5);
    contract_retire(set, 6);
    contract_retire(set, 7);
    EXPECT_EQ(set.reclaim(), (AdvanceResult{3, 1, 0, 0, 0, Advance::advanced}));
    EXPECT_EQ(set.get_internal_generation(), 3u);
    retire_crossing(8);
    EXPECT_EQ(set.get_internal_generation(), 4u);
    {
        ContractSet::handle h = set.new_handle();
        for (int k = 100; k < 105; ++k) EXPECT_TRUE(set.insert(h, k));
    }
    EXPECT_EQ(set.advance(), (AdvanceResult{1, 1, 0, B, 5, Advance::nothing_retired}));
    const ContractSet::InternalCounters end = set.get_internal_counters();
    EXPECT_EQ(end.auto_reclaims, 2u);
    EXPECT_EQ(end.auto_skipped, 0u);
    EXPECT_EQ(end.contended, 0u);
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, 1));
    EXPECT_TRUE(contract_cadence_invariants(set, true));
    EXPECT_TRUE(setting.holds());
} // ContractCadenceAdvanceUnchangedAtQuiescence

// A thread that erases under a NEVER
// refreshed handle h pins its own garbage. R = 1 on a ring of K slots: erases
// 1..K-1 each trigger `advanced` with nothing freed (bumps +1, bags_freed
// +0); erase K's trigger is `ring_full` (ring_full +1, auto_reclaims +1,
// bumps +0, the node stays on the retired list), and so is erase K+1's; after
// h.refresh() (true: the triggers' publishes happened-before it) the next
// erase's trigger returns `advanced`, freeing every waiting bag (K-1 bags of
// one node each: advance()'s ring-wrap shape with the trigger as the caller) and closing a
// bag of the 3 nodes retired since the last `advanced` bump, which the
// refreshed h pins. h released, an explicit reclaim() frees it:
// {3, 1, 0, 0, 0, nothing_retired}.
template <size_t K>
static void contract_cadence_never_refreshed_ring_full() {
    using SetT = ContractSetK<K>;
    SCOPED_TRACE(testing::Message() << "kGenerations " << K);
    SetT set(contract_cadence_options<SetT>(1));
    const ContractOneNode<SetT> setting(set);
    std::unique_ptr<typename SetT::handle> h = std::make_unique<typename SetT::handle>(set.new_handle());
    const int keys = static_cast<int>(K) + 2;
    for (int k = 0; k < keys; ++k) EXPECT_TRUE(set.insert(*h, k));
    int next = 0;
    for (size_t erase = 1; erase < K; ++erase) {
        const typename SetT::InternalCounters before = set.get_internal_counters();
        EXPECT_TRUE(set.erase(*h, next++));
        EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.retire_nodes = 1, .retire_runs = 1, .bumps = 1, .auto_reclaims = 1}))
            << "erase " << erase;
    } // erases 1 .. K-1
    EXPECT_EQ(set.get_internal_generation(), K - 1);
    for (size_t erase = K; erase <= K + 1; ++erase) {
        const typename SetT::InternalCounters before = set.get_internal_counters();
        EXPECT_TRUE(set.erase(*h, next++));
        EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.retire_nodes = 1, .retire_runs = 1, .auto_reclaims = 1, .ring_full = 1}))
            << "erase " << erase;
        EXPECT_TRUE(contract_sweep_is(set, erase - K + 1, K - 1, 0)) << "erase " << erase;
        EXPECT_EQ(set.get_internal_generation(), K - 1) << "erase " << erase;
        EXPECT_TRUE(contract_ring_invariant(set, K)) << "erase " << erase;
    } // erases K, K+1: ring_full
    EXPECT_TRUE(h->refresh());
    const typename SetT::InternalCounters before = set.get_internal_counters();
    EXPECT_TRUE(set.erase(*h, next++));
    EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(),
                                    {.retire_nodes = 1, .retire_runs = 1, .bumps = 1, .auto_reclaims = 1, .bags_freed = K - 1, .nodes_freed = K - 1}))
        << "the erase after the refresh";
    EXPECT_EQ(set.get_internal_generation(), K);
    EXPECT_TRUE(contract_sweep_is(set, 0, 3, K - 1)) << "after the recovery";
    EXPECT_EQ(set.get_internal_bag_count(K - 1), 3u);
    EXPECT_TRUE(contract_ring_invariant(set, K));
    h.reset();
    EXPECT_EQ(set.reclaim(), (AdvanceResult{3, 1, 0, 0, 0, Advance::nothing_retired}));
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, K + 2));
    EXPECT_TRUE(contract_cadence_invariants(set, true));
    EXPECT_TRUE(setting.holds());
} // contract_cadence_never_refreshed_ring_full()

// The never-refreshed holder at the minimum ring (2), a small one (4) and
// the default (64).
TEST(ConcurrentHashSetRcuTest, ContractCadenceNeverRefreshedHandleFillsTheRing) {
    contract_cadence_never_refreshed_ring_full<2>();
    contract_cadence_never_refreshed_ring_full<4>();
    contract_cadence_never_refreshed_ring_full<64>();
} // ContractCadenceNeverRefreshedHandleFillsTheRing

// A cadence of 2^63 "fires at no erase a test can afford":
// cadences 2^63, 2^63 + 1 and SIZE_MAX (all R = 2^63) and 0 (off): 200
// erases under h trigger nothing (auto_reclaims + auto_skipped == 0, bumps
// +0, nothing freed); every node waits on the retired list until an explicit
// call; with h released, one reclaim() bags and frees all 200 in one bag:
// {200, 1, 0, 0, 0, advanced}.
TEST(ConcurrentHashSetRcuTest, ContractCadenceHugeOrZeroCadenceNeverFires) {
    constexpr int kKeys = 200;
    for (size_t every : {kContractCadenceClamp, kContractCadenceClamp + 1, std::numeric_limits<size_t>::max(), size_t{0}}) {
        SCOPED_TRACE(testing::Message() << "requested cadence " << every);
        ContractSet set(contract_cadence_options<ContractSet>(every));
        const ContractOneNode<ContractSet> setting(set);
        {
            ContractSet::handle h = set.new_handle();
            for (int k = 0; k < kKeys; ++k) EXPECT_TRUE(set.insert(h, k));
            const ContractSet::InternalCounters before = set.get_internal_counters();
            for (int k = 0; k < kKeys; ++k) EXPECT_TRUE(set.erase(h, k));
            EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {.retire_nodes = kKeys, .retire_runs = kKeys}));
            EXPECT_TRUE(contract_sweep_is(set, kKeys, 0, 0));
        } // h's scope
        EXPECT_EQ(set.reclaim(), (AdvanceResult{kKeys, 1, 0, 0, 0, Advance::advanced}));
        EXPECT_TRUE(contract_cadence_invariants(set, true));
        EXPECT_TRUE(setting.holds());
    } // loop over cadences
} // ContractCadenceHugeOrZeroCadenceNeverFires

// A proceeding trigger charges the retiring thread's bump_ns and
// free_ns, never advance_ns (no explicit call) nor walk_ns; an erase that
// does not trigger charges nothing. R = 2, h refreshed before each round so
// that each trigger also frees the previous round's bag; the "increase" is
// asserted over 20 triggers (one clock tick may be coarser than one bump).
TEST(ConcurrentHashSetRcuTest, ContractCadenceTriggerChargesBumpAndFreeOnly) {
    using Timers = ContractSet::InternalTimers;
    constexpr int kRounds = 20;
    ContractSet set(contract_cadence_options<ContractSet>(2));
    const ContractOneNode<ContractSet> setting(set);
    ContractSet::handle h = set.new_handle();
    for (int k = 0; k < 2*kRounds; ++k) EXPECT_TRUE(set.insert(h, k));
    uint64_t bump_and_free_ns = 0;
    for (int round = 0; round < kRounds; ++round) {
        if (round > 0) {
            EXPECT_TRUE(h.refresh()) << "round " << round;
        }
        const Timers t0 = ContractSet::internal_advance_timers();
        EXPECT_TRUE(set.erase(h, 2*round));
        const Timers t1 = ContractSet::internal_advance_timers();
        EXPECT_EQ(t1.advance_ns, t0.advance_ns) << "round " << round << ": an erase without a trigger";
        EXPECT_EQ(t1.bump_ns, t0.bump_ns) << "round " << round << ": an erase without a trigger";
        EXPECT_EQ(t1.free_ns, t0.free_ns) << "round " << round << ": an erase without a trigger";
        EXPECT_EQ(t1.walk_ns, t0.walk_ns) << "round " << round << ": an erase without a trigger";
        const ContractSet::InternalCounters c1 = set.get_internal_counters();
        EXPECT_TRUE(set.erase(h, 2*round + 1));
        const Timers t2 = ContractSet::internal_advance_timers();
        EXPECT_EQ(set.get_internal_counters().auto_reclaims - c1.auto_reclaims, 1u) << "round " << round;
        EXPECT_EQ(t2.advance_ns, t1.advance_ns) << "round " << round << ": the trigger charged advance_ns";
        EXPECT_EQ(t2.walk_ns, t1.walk_ns) << "round " << round << ": the trigger charged walk_ns";
        bump_and_free_ns += (t2.bump_ns - t1.bump_ns) + (t2.free_ns - t1.free_ns);
    } // loop over rounds
    EXPECT_GT(bump_and_free_ns, 0u);
    EXPECT_TRUE(setting.holds());
} // ContractCadenceTriggerChargesBumpAndFreeOnly

// Limbo does not count toward the cadence, with the reuse construction and
// a throwing reuse assignment: R = 1, one shard. One node retired (its trigger
// bags it under the erase's handle) and freed by an explicit reclaim() with
// no handle ({1, 1, 0, 0, 0, nothing_retired}), so the one free list holds
// one node. An insert whose reuse copy assignment throws routes that node to
// limbo: limbo_pushes +1 while retire_nodes, auto_reclaims + auto_skipped
// and bumps stay +0 across the throwing insert; the key is absent. The
// limbo node is then bagged and freed by the next explicit call: limbo
// counts for the bump's `nothing_retired` decision: {1, 1, 0, 0, 0, advanced}.
TEST(ConcurrentHashSetRcuTest, ContractCadenceLimboDoesNotCount) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    SetT set(contract_cadence_options<SetT>(1));
    const ContractOneNode<SetT> setting(set);
    contract_retire(set, ContractCountedKey(1));
    EXPECT_EQ(set.get_internal_counters().auto_reclaims, 1u);
    EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::nothing_retired}));
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, 1));
    const size_t slots = set.get_internal_node_count();
    bool threw = false;
    {
        SetT::handle h = set.new_handle();
        const SetT::InternalCounters before = set.get_internal_counters();
        ContractCountedKey::throw_on_assign = true;
        try {
            set.insert(h, ContractCountedKey(2));
        } catch (const ContractAssignThrow&) {
            threw = true;
        }
        ContractCountedKey::throw_on_assign = false;
        const SetT::InternalCounters after = set.get_internal_counters();
        EXPECT_EQ(after.limbo_pushes - before.limbo_pushes, 1u);
        EXPECT_TRUE(contract_deltas_are(before, after, {}));
        EXPECT_FALSE(set.contains(h, ContractCountedKey(2)));
    } // h's scope
    ASSERT_TRUE(threw) << "construction: the insert did not reuse the free node (its copy assignment did not throw)";
    EXPECT_EQ(set.get_internal_node_count(), slots) << "the insert appended instead of reusing";
    EXPECT_EQ(set.get_internal_limbo_count(), 1u);
    EXPECT_EQ(set.reclaim(), (AdvanceResult{1, 1, 0, 0, 0, Advance::advanced}));
    EXPECT_TRUE(contract_sweep_is(set, 0, 0, 1));
    EXPECT_TRUE(contract_cadence_invariants(set, true));
    EXPECT_TRUE(setting.holds());
} // ContractCadenceLimboDoesNotCount

// On a cadenced set whose table never doubled (every
// bucket published since construction), NO lookup retires, bumps or frees.
// R = 1, so any retirement would trigger. After 50 erases (each with its own
// handle, so their triggers bump and free freely), 300 contains() of present
// and absent keys leave retire_nodes, retire_runs, bumps, auto_reclaims,
// auto_skipped and nodes_freed unchanged, and the free count of the sweep
// too; every lookup returns the membership.
TEST(ConcurrentHashSetRcuTest, ContractCadenceContainsOnPresizedSetNeverTriggers) {
    ContractSet set(contract_cadence_options<ContractSet>(1));
    const ContractOneNode<ContractSet> setting(set);
    {
        ContractSet::handle h = set.new_handle();
        for (int k = 0; k < 100; ++k) EXPECT_TRUE(set.insert(h, k));
    }
    for (int k = 0; k < 50; ++k) EXPECT_TRUE(set.erase(set.new_handle(), k));
    EXPECT_EQ(set.get_internal_counters().auto_reclaims, 50u);
    const size_t free0 = set.get_internal_accounting().free_nodes;
    {
        ContractSet::handle h = set.new_handle();
        const ContractSet::InternalCounters before = set.get_internal_counters();
        for (int k = 0; k < 100; ++k) EXPECT_EQ(set.contains(h, k), k >= 50) << "key " << k;
        for (int k = 1000; k < 1200; ++k) EXPECT_FALSE(set.contains(h, k)) << "key " << k;
        EXPECT_TRUE(contract_deltas_are(before, set.get_internal_counters(), {}));
    } // the lookups' handle
    EXPECT_EQ(set.get_internal_accounting().free_nodes, free0);
    EXPECT_TRUE(contract_cadence_invariants(set, true));
    EXPECT_TRUE(setting.holds());
} // ContractCadenceContainsOnPresizedSetNeverTriggers

// A contains() retires -- and so triggers -- ONLY by
// entering an UNINITIALIZED bucket and winning its split, whose cleanup
// retires the superseded copies. A tiny cadenced set (capacity 4, one shard,
// R = 1) is grown by 2000 inserts of spread keys (each with its own handle);
// at the quiescent point some buckets of the doubled table are unpublished
// (checked: the construction exists). Then each present key is looked up
// once, each with its own fresh handle: for every lookup that retired k runs
// (retire_runs + k), auto_reclaims and bumps are + k (single-threaded, every
// trigger proceeds and publishes), auto_skipped + 0, and a split published
// during that lookup (splits_published + >= 1: it retired only through a
// split). At least one lookup must have retired (non-vacuity: otherwise the
// construction never reached an unpublished bucket holding a moved key).
TEST(ConcurrentHashSetRcuTest, ContractCadenceContainsTriggersOnlyThroughASplit) {
    ContractSet set(ContractSet::Options{.initial_capacity = 4, .arena_shards = 1, .auto_reclaim_every = 1});
    std::vector<int> keys;
    for (int i = 0; i < 2000; ++i) keys.push_back(static_cast<int>(contract_mix(i) & 0x3fffffff));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::shuffle(keys.begin(), keys.end(), std::mt19937(12345));
    for (int k : keys) EXPECT_TRUE(set.insert(set.new_handle(), k));
    ASSERT_GT(set.get_internal_table_size(), 4u) << "construction: the table never doubled";
    ASSERT_LT(published_buckets(set), set.get_internal_table_size()) << "construction: every bucket of the doubled table is published";
    size_t retiring_lookups = 0;
    for (int k : keys) {
        ContractSet::handle h = set.new_handle();
        const ContractSet::InternalCounters before = set.get_internal_counters();
        EXPECT_TRUE(set.contains(h, k)) << "key " << k;
        const ContractSet::InternalCounters after = set.get_internal_counters();
        const size_t runs = after.retire_runs - before.retire_runs;
        EXPECT_EQ(after.auto_reclaims - before.auto_reclaims, runs) << "key " << k;
        EXPECT_EQ(after.bumps - before.bumps, runs) << "key " << k;
        EXPECT_EQ(after.auto_skipped, before.auto_skipped) << "key " << k;
        EXPECT_EQ(after.contended, before.contended) << "key " << k;
        if (runs > 0) {
            ++retiring_lookups;
            EXPECT_GT(after.splits_published, before.splits_published) << "key " << k << ": a lookup retired without publishing a split";
        }
    } // loop over present keys
    std::printf("[   INFO   ] ContractCadenceContainsTriggersOnlyThroughASplit: %zu of %zu lookups retired\n", retiring_lookups, keys.size());
    EXPECT_GT(retiring_lookups, 0u) << "construction: no lookup entered an unpublished bucket holding a moved key";
    EXPECT_TRUE(contract_cadence_invariants(set, false));
    EXPECT_TRUE(set.get_internal_accounting().consistent);
} // ContractCadenceContainsTriggersOnlyThroughASplit

// The AllowDelete = false twin, with the R = 1 equality: an AllowDelete = false set with
// a cadence triggers too, because a split's cleanup retires the superseded
// copies. A tiny cadenced set (capacity 4, one shard, R = 1) grown by 2000
// inserts of keys 1..2000 (each with its own handle) has retire_runs > 0 and auto_reclaims >
// 0, every crossing counted (auto_reclaims + auto_skipped == retire_runs),
// none skipped (single-threaded, no explicit call), and membership
// exact.
TEST(ConcurrentHashSetRcuTest, ContractCadenceNoDeleteSetTriggersThroughSplits) {
    ContractSetNoDel set(ContractSetNoDel::Options{.initial_capacity = 4, .arena_shards = 1, .auto_reclaim_every = 1});
    constexpr int kKeys = 2000;
    for (int k = 1; k <= kKeys; ++k) EXPECT_TRUE(set.insert(set.new_handle(), k));
    const ContractSetNoDel::InternalCounters c = set.get_internal_counters();
    EXPECT_GT(set.get_internal_table_size(), 4u);
    EXPECT_GT(c.retire_runs, 0u);
    EXPECT_GT(c.auto_reclaims, 0u);
    EXPECT_EQ(c.auto_skipped, 0u);
    EXPECT_TRUE(contract_cadence_invariants(set, false));
    ContractSetNoDel::handle h = set.new_handle();
    for (int k = 1; k <= kKeys; ++k) EXPECT_TRUE(set.contains(h, k)) << "key " << k;
    EXPECT_FALSE(set.contains(h, 0));
    EXPECT_TRUE(contract_cadence_invariants(set, false)) << "after the lookups, whose split cleanups trigger under h";
} // ContractCadenceNoDeleteSetTriggersThroughSplits

// The destructor on a cadenced set: destroyed with bags pending (closed by
// triggers under a never-refreshed handle, then the handle released), the set
// destroys exactly get_internal_node_count() values and every object
// constructed during the test is destroyed exactly once; under ASan/LSan
// nothing leaks. The destructor's debug asserts on the cadence's counting
// invariants run here too.
TEST(ConcurrentHashSetRcuTest, ContractCadenceDestroySetWithPendingBags) {
    using SetT = ConcurrentResizableHashSetRCU<ContractCountedKey, true, ContractCountedKeyHash>;
    const unsigned first = ContractCountedKey::next_serial.load();
    {
        std::unique_ptr<SetT> set = std::make_unique<SetT>(contract_cadence_options<SetT>(1));
        {
            // The setting refers to the set, so it lives in a scope that ends before the set does.
            const ContractOneNode<SetT> setting(*set);
            {
                SetT::handle h = set->new_handle();
                for (int k = 0; k < 10; ++k) EXPECT_TRUE(set->insert(h, ContractCountedKey(k)));
                for (int k = 0; k < 6; ++k) EXPECT_TRUE(set->erase(h, ContractCountedKey(k)));
            } // h's scope
            EXPECT_TRUE(contract_sweep_is(*set, 0, 6, 0)) << "six pending bags, nothing freed";
            EXPECT_TRUE(setting.holds());
        } // the setting's scope
        const size_t slots = set->get_internal_node_count();
        const size_t d0 = ContractCountedKey::destructor_calls.load();
        set.reset();
        EXPECT_EQ(ContractCountedKey::destructor_calls.load() - d0, slots);
    } // the set's scope
    EXPECT_EQ(contract_destruction_errors(first, ContractCountedKey::next_serial.load()), 0);
} // ContractCadenceDestroySetWithPendingBags
