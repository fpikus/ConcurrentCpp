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
// Basic and stress tests of LockFreeListRCU: the test shapes of the
// reference-counted list's suite (LockFreeList/lock_free_list_test.C) ported
// to handles and generation reclamation, plus the tests of the node-reuse
// paths that list does not have (EmplaceAfter, the dead-anchor insert with an
// empty free list). The white-box scheme tests and the black-box contract
// tests are in the two other test sources linked into the same binary.
//
// Every test ends at a quiescent point with no live handle and checks there:
// the unconditional drain (`while (erase_after(h, before_begin(h))) {}` must
// leave `begin(h) == end()`), the accounting sweep (`expect_consistent()`) and
// the exact end-state oracle of `expect_quiescent_reclaim()`. Tests with
// Tracked payloads also check that every payload is destroyed once the drained
// list has been reclaimed, before the list itself is destroyed. The one
// exception is NoNodeLeakUnderStress: its point is the destructor, so it
// deliberately drops the list UNdrained, with all four node populations
// populated, and checks the payloads after the destruction (its own comment
// lists the checks it makes at the quiescent points before that).
//
// The stress tests share one shape: worker threads, each with its OWN handle
// refreshed on a fixed cadence (a handle that never refreshes pins its
// generation and nothing younger is ever freed); at least two reclaimer
// threads calling reclaim() in a loop, so most calls return `contended` (the
// counts are printed); readers that DEREFERENCE every node they visit and
// validate the payload (a node freed under a reader is an ASan
// use-after-poison report, a TSan data race between the reclaimer's ~Tracked
// and the read, or a poisoned value); and walkers that park an iterator, let
// the erasers and reclaimers run, then walk on through the graveyard to
// end(). Under the TestHooks configurations every hook call yields, widening
// the protocol windows. The end-state oracle is exact: join the threads (their
// handles die with them), drain, and the first reclaim() frees everything
// retired; conservation: successful inserts == successful erases (drain
// included) == nodes freed by all reclaim() calls.
//
// No test here relies on `++` past end(), which is a precondition violation
// for this list (the reference-counted list made it a silent no-op): every
// loop tests `!= end()` before incrementing.

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "lock_free_list_rcu.h"
#include "rcu_test_common.h"

namespace {

// The values of a list of int from begin(h) to end(), dead (graveyard) nodes
// that are still linked included. Precondition: no concurrent operation.
template <typename ListT>
std::vector<int> walk(const ListT& list, const typename ListT::handle& h) {
    std::vector<int> seen;
    for (typename ListT::const_iterator it = list.begin(h); it != ListT::cend(); ++it) seen.push_back(*it);
    return seen;
} // walk()

// Exclusive upper bound of the payload values the stress tests insert; a read
// value outside [0, kValueLimit) is a destroyed (Tracked::kPoison) or garbage
// payload.
constexpr int kValueLimit = 1 << 30;

// Dereferences the node `it` is on and reports whether its payload is a value
// a test inserted (see kValueLimit).
template <typename Iterator>
bool valid_payload(const Iterator& it) {
    const int v = it->v;
    return v >= 0 && v < kValueLimit;
} // valid_payload()

// Walks from `it` to `end`, dereferencing and validating every node on the
// way (graveyard nodes included). Returns the number of invalid payloads.
template <typename Iterator>
long validate_to_end(Iterator it, const Iterator& end) {
    long bad = 0;
    for (; it != end; ++it) {
        if (!valid_payload(it)) ++bad;
    } // loop to end()
    return bad;
} // validate_to_end()

// Outcomes of the reclaim() calls one thread made. Each reclaimer fills its
// own instance; they are summed after the threads are joined.
struct ReclaimTally {
    long advanced = 0;         // calls that advanced the generation
    long nothing_retired = 0;  // calls that found nothing to advance
    long ring_full = 0;        // calls refused because a handle pinned the oldest generation
    long contended = 0;        // calls that found another reclaim() in progress
    size_t freed_nodes = 0;    // sum of freed_nodes over all calls
    size_t freed_bags = 0;     // sum of freed_bags over all calls

    // Counts one call's result.
    void add(const ReclaimResult& r) noexcept {
        switch (r.advance) {
            case ReclaimResult::Advance::advanced: ++advanced; break;
            case ReclaimResult::Advance::nothing_retired: ++nothing_retired; break;
            case ReclaimResult::Advance::ring_full: ++ring_full; break;
            case ReclaimResult::Advance::contended: ++contended; break;
        } // switch over the outcome
        freed_nodes += r.freed_nodes;
        freed_bags += r.freed_bags;
    } // ReclaimTally::add()

    // Adds another thread's tally into this one.
    void merge(const ReclaimTally& o) noexcept {
        advanced += o.advanced;
        nothing_retired += o.nothing_retired;
        ring_full += o.ring_full;
        contended += o.contended;
        freed_nodes += o.freed_nodes;
        freed_bags += o.freed_bags;
    } // ReclaimTally::merge()

    // Prints the tally on the gtest output, tagged with the test name.
    void print(const char* test) const {
        std::printf("[          ] %s: reclaim() %ld advanced, %ld nothing_retired, %ld ring_full, %ld contended; "
                    "%zu nodes in %zu bags freed\n",
                    test, advanced, nothing_retired, ring_full, contended, freed_nodes, freed_bags);
    } // ReclaimTally::print()
}; // ReclaimTally

// Body of a reclaimer thread: calls reclaim() in a loop until `stop` is set,
// yielding between calls, and tallies the results.
template <typename ListT>
void reclaim_until(ListT& list, const std::atomic<bool>& stop, ReclaimTally& tally) {
    while (!stop.load(std::memory_order_acquire)) {
        tally.add(list.reclaim());
        std::this_thread::yield();
    } // loop until stopped
} // reclaim_until()

// Per-worker counters of the stress tests, read after the threads are joined.
struct WorkerTally {
    long inserted = 0;   // insert_after() calls that returned true
    long erased = 0;     // erase_after() calls that returned true
    long bad_reads = 0;  // dereferenced payloads that were not a value any test inserted
}; // WorkerTally

// End of every test in this file, at a quiescent point after the test has
// destroyed all its handles: the unconditional drain, then the exact
// end-state oracle. Returns {erased by the drain, nodes freed by the final
// reclaim()}.
template <typename ListT>
std::pair<size_t, size_t> drain_and_check(ListT& list, const std::string& where) {
    const size_t drained = drain(list);
    const ReclaimResult final_reclaim = expect_quiescent_reclaim(list, where);
    return {drained, final_reclaim.freed_nodes};
} // drain_and_check()

} // namespace

template <typename Config>
class LockFreeListRcuTest : public RcuListTestBase<Config> {};

TYPED_TEST_SUITE(LockFreeListRcuTest, RcuListConfigs, RcuListConfigNames);

// Two front inserts, a walk, two front erases; the accounting follows every
// step (an erased node is retired at its unlink, not destroyed).
TYPED_TEST(LockFreeListRcuTest, BasicInsertErase) {
    using List = typename TestFixture::template ListOf<int>;
    List list;
    {
        typename List::handle h = list.new_handle();
        EXPECT_TRUE(list.begin(h) == List::end());

        EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 10));
        EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 20));

        typename List::iterator it = list.begin(h);
        ASSERT_TRUE(it != List::end());
        EXPECT_EQ(*it, 20);
        ++it;
        ASSERT_TRUE(it != List::end());
        EXPECT_EQ(*it, 10);
        ++it;
        EXPECT_TRUE(it == List::end());
        typename List::InternalAccounting acc = expect_consistent(list, 1, "after two inserts");
        EXPECT_EQ(acc.reachable, 2u);
        EXPECT_EQ(acc.retired, 0u);

        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // erase 20
        it = list.begin(h);
        ASSERT_TRUE(it != List::end());
        EXPECT_EQ(*it, 10);

        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // erase 10
        EXPECT_TRUE(list.begin(h) == List::end());
        acc = expect_consistent(list, 1, "after two erases");
        EXPECT_EQ(acc.reachable, 0u);
        EXPECT_EQ(acc.retired, 2u);  // retired at the unlink, by this thread
    } // handle scope
    drain_and_check(list, "BasicInsertErase");
} // BasicInsertErase

// Erasing after the head of an empty list, after end(), and twice after the
// only node all fail cleanly; inserting after end() fails before allocating.
TYPED_TEST(LockFreeListRcuTest, EraseEdgeCases) {
    using List = typename TestFixture::template ListOf<Tracked>;
    List list;
    {
        typename List::handle h = list.new_handle();
        EXPECT_FALSE(list.erase_after(h, list.before_begin(h)));  // nothing after the head
        EXPECT_FALSE(list.erase_after(h, List::end()));           // a null (end) anchor

        // An end() anchor returns false before any allocation: no node, and
        // the by-value argument is consumed (destroyed) by the call.
        EXPECT_FALSE(list.insert_after(h, List::end(), Tracked(7)));
        EXPECT_FALSE(list.emplace_after(h, List::end(), 8));
        typename List::InternalAccounting acc = expect_consistent(list, 1, "after the end() anchors");
        EXPECT_EQ(acc.reachable + acc.retired + acc.bagged + acc.free, 0u);
#ifndef NDEBUG
        EXPECT_EQ(acc.allocated, 0u);
#endif
        EXPECT_EQ(Tracked::alive.load(), this->tracked_base_);

        EXPECT_TRUE(list.insert_after(h, list.before_begin(h), Tracked(5)));
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
        // Erasing again on the now-empty list must report failure, not retire twice.
        EXPECT_FALSE(list.erase_after(h, list.before_begin(h)));
        EXPECT_TRUE(list.begin(h) == List::end());
        acc = expect_consistent(list, 1, "after the erase");
        EXPECT_EQ(acc.reachable, 0u);
        EXPECT_EQ(acc.retired, 1u);
    } // handle scope
    drain_and_check(list, "EraseEdgeCases");
    EXPECT_EQ(Tracked::alive.load(), this->tracked_base_);
} // EraseEdgeCases

// Front inserts reverse the order; an insert after the first node lands
// second.
TYPED_TEST(LockFreeListRcuTest, OrderAndMiddleInsert) {
    using List = typename TestFixture::template ListOf<int>;
    List list;
    {
        typename List::handle h = list.new_handle();
        // Each insert_after(before_begin) pushes to the front: ends up 5,4,3,2,1.
        for (int i = 1; i <= 5; ++i) EXPECT_TRUE(list.insert_after(h, list.before_begin(h), i));
        EXPECT_EQ(walk(list, h), (std::vector<int>{5, 4, 3, 2, 1}));

        // Insert 99 after the first node -> 5,99,4,3,2,1.
        EXPECT_TRUE(list.insert_after(h, list.begin(h), 99));
        EXPECT_EQ(walk(list, h), (std::vector<int>{5, 99, 4, 3, 2, 1}));
        EXPECT_EQ(expect_consistent(list, 1, "after the inserts").reachable, 6u);
    } // handle scope
    EXPECT_EQ(drain_and_check(list, "OrderAndMiddleInsert").first, 6u);
} // OrderAndMiddleInsert

// Inserting after a logically deleted node must fail, otherwise the new node
// would be stranded in a detached chain. With an EMPTY free list the node is
// freshly allocated, so the failure path must construct, destroy and delete
// it: no node leaks (`allocated` unchanged), none is retired, and no payload
// survives. (The failure path for a node popped from the free list --
// disposal to the retired list -- is EmplaceAfter's third case.)
TYPED_TEST(LockFreeListRcuTest, InsertAfterDeletedAnchorFails) {
    using List = typename TestFixture::template ListOf<Tracked>;
    List list;
    {
        typename List::handle h = list.new_handle();
        EXPECT_TRUE(list.insert_after(h, list.before_begin(h), Tracked(10)));
        typename List::iterator on_ten = list.begin(h);
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // logically delete node 10

        const typename List::InternalAccounting before = expect_consistent(list, 1, "after the erase");
        ASSERT_EQ(before.free, 0u) << "the free list must be empty for this test";
        EXPECT_EQ(before.retired, 1u);
        const long alive_before = Tracked::alive.load();

        // Anchor is logically deleted -> insertion is refused, by both paths.
        EXPECT_FALSE(list.insert_after(h, on_ten, Tracked(99)));
        EXPECT_FALSE(list.emplace_after(h, on_ten, 98));

        const typename List::InternalAccounting after = expect_consistent(list, 1, "after the refused inserts");
        EXPECT_EQ(after.retired, before.retired) << "a freshly allocated node must not be retired";
        EXPECT_EQ(after.free, 0u);
        EXPECT_EQ(after.reachable, 0u);
#ifndef NDEBUG
        EXPECT_EQ(after.allocated, before.allocated) << "the refused node leaked";
#endif
        EXPECT_EQ(Tracked::alive.load(), alive_before) << "the refused node's payload was not destroyed";
        EXPECT_EQ(on_ten->v, 10);  // the dead anchor still holds its value
    } // handle scope
    drain_and_check(list, "InsertAfterDeletedAnchorFails");
    EXPECT_EQ(Tracked::alive.load(), this->tracked_base_);
} // InsertAfterDeletedAnchorFails

// An iterator parked on a node keeps walking after that node and its
// successor are erased: through the graveyard (dead nodes whose `next` is
// frozen) back into the live suffix. A reclaim() in between turns the erased
// nodes into a sealed bag (the parked handle pins it, so nothing is freed):
// neither retirement nor bagging may rewrite a dead node's `next`, because
// they chain nodes through the separate retire link.
TYPED_TEST(LockFreeListRcuTest, GraveyardTraversal) {
    using List = typename TestFixture::template ListOf<Tracked>;
    List list;
    {
        typename List::handle h = list.new_handle();
        // Insert 1, 2, 3
        for (int v = 3; v >= 1; --v) EXPECT_TRUE(list.insert_after(h, list.before_begin(h), Tracked(v)));
        typename List::iterator it1 = list.begin(h);  // points to 1

        {
            typename List::handle eraser = list.new_handle();
            EXPECT_TRUE(list.erase_after(eraser, list.before_begin(eraser)));  // erases 1
            EXPECT_TRUE(list.erase_after(eraser, list.before_begin(eraser)));  // erases 2
        } // eraser handle scope
        EXPECT_EQ(expect_consistent(list, 1, "after the erases").retired, 2u);

        // h's generation is the one 1 and 2 were retired in: the bag is
        // sealed and kept.
        EXPECT_EQ(list.reclaim(), (ReclaimResult{0, 0, ReclaimResult::Advance::advanced}));
        const typename List::InternalAccounting acc = expect_consistent(list, 1, "after the pinned reclaim()");
        EXPECT_EQ(acc.bagged, 2u);
        EXPECT_EQ(acc.reachable, 1u);

        // it1 is still on 1, which is logically deleted and bagged. We should
        // be able to traverse to 2 and 3, with intact payloads.
        EXPECT_EQ(it1->v, 1);
        ++it1;
        ASSERT_TRUE(it1 != List::end());
        EXPECT_EQ(it1->v, 2);
        ++it1;
        ASSERT_TRUE(it1 != List::end());
        EXPECT_EQ(it1->v, 3);
        ++it1;
        EXPECT_TRUE(it1 == List::end());
    } // handle scope: h leaves its generation
    // The bag is free now: nothing new was retired, the free pass frees it.
    EXPECT_EQ(list.reclaim(), (ReclaimResult{2, 1, ReclaimResult::Advance::nothing_retired}));
    EXPECT_EQ(Tracked::alive.load(), this->tracked_base_ + 1);  // only 3 is left
    drain_and_check(list, "GraveyardTraversal");
    EXPECT_EQ(Tracked::alive.load(), this->tracked_base_);
} // GraveyardTraversal

// emplace_after() through all three node sources of the one initialization
// path, with a payload that can ONLY be constructed from (int, int) (it
// compiles only because the list never copies, moves, default-constructs or
// assigns T):
// 1. new: the free list is empty -- `allocated` +1 per insert, free unchanged;
// 2. popped: a node freed by reclaim() is reused -- free -1, `allocated`
//    unchanged, the same address, the incarnation bumped (debug);
// 3. disposed: a dead anchor with a NON-EMPTY free list -- the popped node is
//    constructed, the link fails, and the node goes to the RETIRED list (not
//    back to the free list, not deleted) with its value alive: retired +1,
//    free -1, `allocated` unchanged; its ~T runs at the next reclaim() that
//    frees it.
// Script (front inserts; values are (a, b)):
//   C(3,30) B(2,20) A(1,10) new; erase C, reclaim -> C free; D(4,40) pops C;
//   erase D, reclaim -> D free; under h: erase B (dead anchor), emplace after B
//   pops D and is refused -> disposed; drop h; reclaim frees B and D.
TYPED_TEST(LockFreeListRcuTest, EmplaceAfter) {
    using List = typename TestFixture::template ListOf<TwoArgPayload>;
    using Advance = ReclaimResult::Advance;
    const long base = this->two_arg_base_;
    List list;

    // 1. new
    {
        typename List::handle hs = list.new_handle();
        for (int i = 1; i <= 3; ++i) {
            const typename List::InternalAccounting before = expect_consistent(list, 1, "before a new-node emplace");
            EXPECT_TRUE(list.emplace_after(hs, list.before_begin(hs), i, 10*i));
            const typename List::InternalAccounting after = expect_consistent(list, 1, "after a new-node emplace");
            EXPECT_EQ(after.reachable, before.reachable + 1);
            EXPECT_EQ(after.free, 0u);
#ifndef NDEBUG
            EXPECT_EQ(after.allocated, before.allocated + 1);
#endif
        } // loop over the three new-node emplaces
        EXPECT_EQ(list.begin(hs)->a, 3);
        EXPECT_EQ(list.begin(hs)->b, 30);
    } // hs scope
    EXPECT_EQ(TwoArgPayload::alive.load(), base + 3);

    // Free C: erase it under a scoped handle, then reclaim with no handle.
    const TwoArgPayload* addr_c = nullptr;
#ifndef NDEBUG
    uint32_t incarnation_c = 0;
#endif
    {
        typename List::handle hs = list.new_handle();
        addr_c = &*list.begin(hs);
#ifndef NDEBUG
        incarnation_c = list.get_internal_incarnation(list.begin(hs));
#endif
        EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));
    } // hs scope
    EXPECT_EQ(list.reclaim(), (ReclaimResult{1, 1, Advance::advanced}));
    EXPECT_EQ(expect_consistent(list, 0, "after freeing C").free, 1u);
    EXPECT_EQ(TwoArgPayload::alive.load(), base + 2);

    // 2. popped
    {
        typename List::handle hs = list.new_handle();
        const typename List::InternalAccounting before = expect_consistent(list, 1, "before the popped-node emplace");
        EXPECT_TRUE(list.emplace_after(hs, list.before_begin(hs), 4, 40));
        const typename List::InternalAccounting after = expect_consistent(list, 1, "after the popped-node emplace");
        EXPECT_EQ(after.free, before.free - 1);
        EXPECT_EQ(after.reachable, before.reachable + 1);
#ifndef NDEBUG
        EXPECT_EQ(after.allocated, before.allocated) << "emplace_after() allocated although the free list had a node";
        EXPECT_EQ(list.get_internal_incarnation(list.begin(hs)), incarnation_c + 1);
#endif
        EXPECT_EQ(&*list.begin(hs), addr_c) << "the freed node was not reused";
        EXPECT_EQ(list.begin(hs)->a, 4);
        EXPECT_EQ(list.begin(hs)->b, 40);
    } // hs scope
    EXPECT_EQ(TwoArgPayload::alive.load(), base + 3);

    // Free D the same way, so that the free list is non-empty for case 3.
    {
        typename List::handle hs = list.new_handle();
        EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));
    } // hs scope
    EXPECT_EQ(list.reclaim(), (ReclaimResult{1, 1, Advance::advanced}));
    EXPECT_EQ(TwoArgPayload::alive.load(), base + 2);  // B, A

    // 3. disposed
    {
        typename List::handle h = list.new_handle();
        typename List::iterator on_b = list.begin(h);
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // B: now the dead anchor
        const typename List::InternalAccounting before = expect_consistent(list, 1, "before the refused emplace");
        ASSERT_EQ(before.free, 1u) << "case 3 needs a node on the free list";
        EXPECT_EQ(before.retired, 1u);
        const long alive_before = TwoArgPayload::alive.load();

        EXPECT_FALSE(list.emplace_after(h, on_b, 5, 50));

        const typename List::InternalAccounting after = expect_consistent(list, 1, "after the refused emplace");
        EXPECT_EQ(after.free, 0u) << "the popped node went back to the free list";
        EXPECT_EQ(after.retired, 2u) << "the popped node was not disposed to the retired list";
        EXPECT_EQ(after.reachable, before.reachable);
#ifndef NDEBUG
        EXPECT_EQ(after.allocated, before.allocated) << "the popped node was deleted or a new one allocated";
#endif
        EXPECT_EQ(TwoArgPayload::alive.load(), alive_before + 1) << "the disposed node's ~T must be deferred to reclaim()";
    } // h scope
    // B and the disposed node, retired in the same generation, freed together.
    EXPECT_EQ(list.reclaim(), (ReclaimResult{2, 1, Advance::advanced}));
    EXPECT_EQ(TwoArgPayload::alive.load(), base + 1);  // A

    drain_and_check(list, "EmplaceAfter");
    EXPECT_EQ(TwoArgPayload::alive.load(), base);
} // EmplaceAfter

// Mixed-operation stress: workers insert at the front, erase at the front,
// read up to ten nodes from the front, or park on the first node and later
// walk on from it to end(); two reclaimer threads run throughout. See the
// file comment for the shared stress shape and the end-state oracle. The
// drain at the end is also the regression check for the helping path in
// erase_after(): without helping, a marked node whose unlink CAS lost to a
// concurrent insert stays linked forever and the drain stops at it
// (`../LockFreeList/lock_free_list_bugs.md`, bug 1). It is unconditional --
// the reference-counted list's test guarded it with `#ifndef NDEBUG`, which
// the test flags never define.
TYPED_TEST(LockFreeListRcuTest, StressTest) {
    using List = typename TestFixture::template ListOf<Tracked>;
    constexpr int num_workers = 6;
    constexpr int num_reclaimers = 2;
    constexpr int num_iters = 2000;
    constexpr int refresh_every = 32;  // worker operations between refreshes of its handle

    List list;
    std::vector<WorkerTally> workers(num_workers);
    std::vector<ReclaimTally> reclaimers(num_reclaimers);
    std::atomic<bool> stop_reclaimers{false};
    std::barrier start(num_workers + num_reclaimers);
    this->enable_yield_in_windows();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_workers; ++i) {
        threads.emplace_back([&, i]() {
            std::mt19937 rng(i);
            std::uniform_int_distribution<int> dist(0, 99);
            WorkerTally& tally = workers[i];
            start.arrive_and_wait();
            typename List::handle h = list.new_handle();
            for (int j = 0; j < num_iters; ++j) {
                // No iterator survives an operation here, so refreshing at
                // any operation boundary invalidates nothing in use.
                if (j % refresh_every == 0) h.refresh();
                const int op = dist(rng);
                if (op < 45) {
                    if (list.insert_after(h, list.before_begin(h), Tracked(i*num_iters + j))) ++tally.inserted;
                } else if (op < 80) {
                    if (list.erase_after(h, list.before_begin(h))) ++tally.erased;
                } else if (op < 90) {
                    // Read up to 10 nodes from the front, dereferencing each.
                    int count = 0;
                    for (typename List::iterator it = list.begin(h); it != List::end() && count < 10; ++it, ++count) {
                        if (!valid_payload(it)) ++tally.bad_reads;
                    } // loop over at most 10 nodes
                } else {
                    // Park on the first node, let the others erase and reclaim
                    // around it, then walk on to end() through the graveyard.
                    typename List::iterator parked = list.begin(h);
                    for (int k = 0; k < 3; ++k) std::this_thread::yield();
                    tally.bad_reads += validate_to_end(parked, List::end());
                } // choose an operation
            } // loop over operations
        }); // worker thread body
    } // loop over worker threads
    for (int i = 0; i < num_reclaimers; ++i) {
        threads.emplace_back([&, i]() {
            start.arrive_and_wait();
            reclaim_until(list, stop_reclaimers, reclaimers[i]);
        });
    } // loop over reclaimer threads
    for (int i = 0; i < num_workers; ++i) threads[i].join();
    stop_reclaimers.store(true, std::memory_order_release);
    for (int i = num_workers; i < num_workers + num_reclaimers; ++i) threads[i].join();

    WorkerTally total;
    for (const WorkerTally& w : workers) {
        total.inserted += w.inserted;
        total.erased += w.erased;
        total.bad_reads += w.bad_reads;
    } // loop over worker tallies
    ReclaimTally reclaimed;
    for (const ReclaimTally& r : reclaimers) reclaimed.merge(r);
    reclaimed.print("StressTest");
    EXPECT_EQ(total.bad_reads, 0) << "readers saw destroyed or garbage payloads";

    // Every worker handle died with its thread: quiescent, no handles.
    const auto [drained, final_freed] = drain_and_check(list, "StressTest");
    EXPECT_EQ(total.inserted, total.erased + static_cast<long>(drained)) << "every inserted node must be erased exactly once";
    // No refused inserts here (the head is never a dead anchor), so the only
    // retirements are the erasures, and each retired node is freed once.
    EXPECT_EQ(reclaimed.freed_nodes + final_freed, static_cast<size_t>(total.inserted)) << "nodes freed by all reclaim() calls";
    EXPECT_EQ(Tracked::alive.load(), this->tracked_base_) << "payloads alive after the final reclaim()";
} // StressTest

// Regression hammer for the helping path in erase_after(): one pure inserter
// and one pure eraser collide on the same anchor, maximizing the chance that
// an insert lands inside the eraser's window between its mark CAS and its
// unlink CAS. The unlink CAS then fails, and without helping the marked node
// stays linked for good, wedging the drain at the end. The mixed-op
// StressTest is too diffuse to hit that interleaving reliably; this shape
// stranded marked nodes within a few thousand operations in the
// reference-counted list before helping (`../LockFreeList/lock_free_list_bugs.md`,
// bug 1). Here the eraser's handle REFRESHES on a cadence -- otherwise its
// generation pins everything it retires and the reclaimers free nothing --
// two reclaimer threads run throughout, so the inserter mostly reuses nodes
// from the free list, and one walker parks on the front node and walks on
// through the graveyard, dereferencing payloads that may be freed and reused
// under it if the scheme is wrong.
TYPED_TEST(LockFreeListRcuTest, InsertEraseHammer) {
    using List = typename TestFixture::template ListOf<Tracked>;
    constexpr int num_inserts = 50000;
    constexpr int num_reclaimers = 2;
    constexpr int refresh_every = 64;  // operations between refreshes of the inserter's and the eraser's handles

    List list;
    WorkerTally inserter_tally;
    WorkerTally eraser_tally;
    WorkerTally walker_tally;
    std::vector<ReclaimTally> reclaimers(num_reclaimers);
    std::atomic<bool> stop{false};             // the inserter is done: stops the eraser and the walker
    std::atomic<bool> stop_reclaimers{false};  // the eraser and the walker are done
    std::barrier start(3 + num_reclaimers);
    this->enable_yield_in_windows();

    std::thread eraser([&]() {
        start.arrive_and_wait();
        typename List::handle h = list.new_handle();
        for (long n = 1; !stop.load(std::memory_order_acquire); ++n) {
            if (list.erase_after(h, list.before_begin(h))) ++eraser_tally.erased;
            if (n % refresh_every == 0) h.refresh();
        } // loop until the inserter is done
    }); // eraser thread body
    std::thread inserter([&]() {
        start.arrive_and_wait();
        typename List::handle h = list.new_handle();
        for (int i = 0; i < num_inserts; ++i) {
            if (list.insert_after(h, list.before_begin(h), Tracked(i))) ++inserter_tally.inserted;
            if ((i + 1) % refresh_every == 0) h.refresh();
        } // loop over inserts
    }); // inserter thread body
    std::thread walker([&]() {
        start.arrive_and_wait();
        typename List::handle h = list.new_handle();
        while (!stop.load(std::memory_order_acquire)) {
            typename List::iterator parked = list.begin(h);
            for (int k = 0; k < 3; ++k) std::this_thread::yield();
            walker_tally.bad_reads += validate_to_end(parked, List::end());
            h.refresh();  // between traversals only: it invalidates `parked`
        } // loop until the inserter is done
    }); // walker thread body
    std::vector<std::thread> reclaimer_threads;
    for (int i = 0; i < num_reclaimers; ++i) {
        reclaimer_threads.emplace_back([&, i]() {
            start.arrive_and_wait();
            reclaim_until(list, stop_reclaimers, reclaimers[i]);
        });
    } // loop over reclaimer threads
    inserter.join();
    stop.store(true, std::memory_order_release);
    eraser.join();
    walker.join();
    stop_reclaimers.store(true, std::memory_order_release);
    for (std::thread& t : reclaimer_threads) t.join();

    ReclaimTally reclaimed;
    for (const ReclaimTally& r : reclaimers) reclaimed.merge(r);
    reclaimed.print("InsertEraseHammer");
    EXPECT_EQ(inserter_tally.inserted, num_inserts);  // the head is never a dead anchor
    EXPECT_EQ(walker_tally.bad_reads, 0) << "the walker saw destroyed or garbage payloads";

    // Same end-state oracle as StressTest: the drain must empty the list,
    // dead nodes included; a wedged erase_after() leaves reachable nodes.
    const auto [drained, final_freed] = drain_and_check(list, "InsertEraseHammer");
    EXPECT_EQ(inserter_tally.inserted, eraser_tally.erased + static_cast<long>(drained));
    EXPECT_EQ(reclaimed.freed_nodes + final_freed, static_cast<size_t>(inserter_tally.inserted));
    EXPECT_EQ(Tracked::alive.load(), this->tracked_base_);
} // InsertEraseHammer

// Leak check under concurrency: workers insert, erase, park iterators on
// nodes that others erase (graveyard pins), and walk on from them; each
// worker refreshes its handle when it parks anew (between traversals, since
// a refresh invalidates the parked iterator); two reclaimer threads run
// throughout. Then, with every handle gone, the test builds all four node
// populations deterministically -- free nodes (erase + reclaim with no
// handle), a sealed bag (erase under a handle, reclaim, drop the handle
// WITHOUT reclaiming), a retired-only node (an erase after the last reclaim)
// and the reachable rest -- and drops the list. The destructor must destroy
// every payload in every population: Tracked::alive returns to its base.
// Before that, at each quiescent point, the payloads alive are exactly the
// nodes not on the free list, and the live nodes (reachable minus marked)
// are exactly the inserts minus the erasures.
TYPED_TEST(LockFreeListRcuTest, NoNodeLeakUnderStress) {
    using List = typename TestFixture::template ListOf<Tracked>;
    constexpr int num_workers = 6;
    constexpr int num_reclaimers = 2;
    constexpr int num_iters = 3000;

    const long base = this->tracked_base_;
    {
        List list;
        std::vector<WorkerTally> workers(num_workers);
        std::vector<ReclaimTally> reclaimers(num_reclaimers);
        std::atomic<bool> stop_reclaimers{false};
        std::barrier start(num_workers + num_reclaimers);
        this->enable_yield_in_windows();

        std::vector<std::thread> threads;
        for (int i = 0; i < num_workers; ++i) {
            threads.emplace_back([&, i]() {
                std::mt19937 rng(i);
                std::uniform_int_distribution<int> dist(0, 99);
                WorkerTally& tally = workers[i];
                start.arrive_and_wait();
                typename List::handle h = list.new_handle();
                typename List::iterator parked;  // pins a node across operations; end() until the first park
                for (int j = 0; j < num_iters; ++j) {
                    const int op = dist(rng);
                    if (op < 45) {
                        if (list.insert_after(h, list.before_begin(h), Tracked(i*num_iters + j))) ++tally.inserted;
                    } else if (op < 80) {
                        if (list.erase_after(h, list.before_begin(h))) ++tally.erased;
                    } else if (op < 90) {
                        h.refresh();               // releases the old generation; invalidates the old `parked`
                        parked = list.begin(h);    // later erasures turn it into a graveyard pin
                    } else {
                        int count = 0;
                        for (typename List::iterator it = parked != List::end() ? parked : list.begin(h);
                             it != List::end() && count < 10; ++it, ++count) {
                            if (!valid_payload(it)) ++tally.bad_reads;
                        } // walk at most 10 nodes, possibly through the graveyard
                    } // choose an operation
                } // loop over operations
            }); // worker thread body
        } // loop over worker threads
        for (int i = 0; i < num_reclaimers; ++i) {
            threads.emplace_back([&, i]() {
                start.arrive_and_wait();
                reclaim_until(list, stop_reclaimers, reclaimers[i]);
            });
        } // loop over reclaimer threads
        for (int i = 0; i < num_workers; ++i) threads[i].join();
        stop_reclaimers.store(true, std::memory_order_release);
        for (int i = num_workers; i < num_workers + num_reclaimers; ++i) threads[i].join();

        WorkerTally total;
        for (const WorkerTally& w : workers) {
            total.inserted += w.inserted;
            total.erased += w.erased;
            total.bad_reads += w.bad_reads;
        } // loop over worker tallies
        ReclaimTally reclaimed;
        for (const ReclaimTally& r : reclaimers) reclaimed.merge(r);
        reclaimed.print("NoNodeLeakUnderStress");
        EXPECT_EQ(total.bad_reads, 0) << "readers saw destroyed or garbage payloads";

        typename List::InternalAccounting acc = expect_consistent(list, 0, "after the threads");
        EXPECT_EQ(static_cast<long>(acc.reachable - acc.reachable_dead), total.inserted - total.erased)
            << "live nodes != inserts - erasures; accounting " << accounting_string(acc);
        EXPECT_EQ(Tracked::alive.load() - base, static_cast<long>(acc.reachable + acc.retired + acc.bagged))
            << "payloads alive != nodes off the free list; accounting " << accounting_string(acc);

        // Build every population. Three fresh front nodes guarantee the
        // erasures below find something.
        {
            typename List::handle hs = list.new_handle();
            for (int v = 0; v < 3; ++v) EXPECT_TRUE(list.insert_after(hs, list.before_begin(hs), Tracked(v)));
            EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));
        } // hs scope
        EXPECT_GE(list.reclaim().freed_nodes, 1u);  // free: no handle, everything retired is freed
        {
            typename List::handle pin = list.new_handle();
            EXPECT_TRUE(list.erase_after(pin, list.before_begin(pin)));
            EXPECT_EQ(list.reclaim(), (ReclaimResult{0, 0, ReclaimResult::Advance::advanced}));  // bagged, pinned
        } // pin scope: dropped WITHOUT reclaiming, the bag stays sealed
        {
            typename List::handle hs = list.new_handle();
            EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));  // retired only
        } // hs scope
        acc = expect_consistent(list, 0, "with every population built");
        EXPECT_GE(acc.reachable, 1u);
        EXPECT_GE(acc.retired, 1u);
        EXPECT_GE(acc.bagged, 1u);
        EXPECT_GE(acc.free, 1u);
        EXPECT_EQ(Tracked::alive.load() - base, static_cast<long>(acc.reachable + acc.retired + acc.bagged))
            << "payloads alive != nodes off the free list; accounting " << accounting_string(acc);
    } // list dropped: the destructor frees all four populations
    EXPECT_EQ(Tracked::alive.load(), base) << "node payloads still alive after the list was destroyed";
} // NoNodeLeakUnderStress
