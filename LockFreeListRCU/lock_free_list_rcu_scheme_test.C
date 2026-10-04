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

// White-box scheme tests for LockFreeListRCU: step scripts that drive the
// generation-reclamation protocol into the states where a plausible bug would
// show, with an EXACT expectation after every step.
//
// HOW THE TESTS INTERLEAVE WITHOUT THREADS. The list calls four test hooks
// (`TestHooks`, see rcu_test_common.h) at fixed points inside its protocol:
//   between_mark_and_unlink  erase_after(): after the marking race is decided,
//                            before the unlink CAS (on helping passes too);
//   in_join                  a join attempt: after it loaded the current
//                            generation, before it counts itself in it;
//   in_join_after_bump       a join attempt: after the count, before the
//                            re-check that the generation is still current;
//   in_advance_after_publish reclaim(): after the new generation is published,
//                            before the free pass (the reclaim flag is held).
// A test arms a one-shot action on a hook; the action runs complete list
// operations on the same list, standing in for a second thread that acts inside
// that window. The interleaving is then reproducible on every run and needs no
// threads. Every hook test asserts that its action ran exactly once (`ran`), so
// a hook that never fires cannot make a test pass vacuously, and asserts the
// hook call counts the script derives. Hook tests skip under NoHooks.
//
// THE ORACLES. After every step the test compares the test-only accounting
// sweep with the derived tuple (reachable, reachable_dead, retired, bagged,
// free, handle_refs, allocated) and requires `consistent` (a debug-only
// conjunction: no node in two populations, the four populations add up to the
// allocated nodes, the ring-state invariant, poisoned free nodes under ASan,
// bag/state agreement, refs >= 0 and 0 on an empty slot, every retired or
// bagged node marked). `consistent` is load-bearing, not decoration: the sweep
// stops a walk at an address it has already seen, so a node that is BOTH
// linked and retired is counted once and the `retired` count alone reads
// "unchanged" under a retire-at-mark bug. Every `ReclaimResult` is compared
// exactly. `Tracked::alive` (net of the per-test base) counts the T objects
// alive; a node's value read after reclamation dies of ASan use-after-poison
// (the free list is poisoned) or, under TSan, reads the poison value
// `~Tracked` writes, so `it->v == expected` is a premature-free oracle in both
// builds.
//
// The generation ring is not observable through the accessors; the comments
// give its expected state after each step as [slot: state refs/bag size ...]
// (C current, S sealed, E empty; `o` the oldest index, `c` the current one),
// which is what the next step's expectations are derived from. `consistent`
// checks the ring-state invariant itself.
//
// Each test is self-contained: its own list, handles and hook arming, with
// TestHooks reset before and after every test by the fixture base. Lists are
// declared FIRST in each test body so that every handle (a later local) is
// destroyed before its list, even when a fatal assertion returns early.
// Death tests use the "threadsafe" style (the child re-runs the test body up to
// the death statement, so every prefix is deterministic and single-threaded)
// and match the assert's string literal, so an unrelated abort (a sanitizer
// report, another assert) cannot make them pass.

#include <gtest/gtest.h>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "rcu_test_common.h"

// ASan builds can ask whether an address is poisoned: the free list poisons a
// node's value and `next`, so a test can check directly that a freed node's
// value storage is unreadable (independently of the sweep's own poison check).
#if defined(__SANITIZE_ADDRESS__)
#define SCHEME_TEST_HAVE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SCHEME_TEST_HAVE_ASAN 1
#endif
#endif
#ifdef SCHEME_TEST_HAVE_ASAN
#include <sanitizer/asan_interface.h>
#endif

namespace {

using Advance = ReclaimResult::Advance;

// Shorthand for an exact ReclaimResult expectation.
constexpr ReclaimResult rr(size_t freed_nodes, size_t freed_bags, Advance advance) {
    return ReclaimResult{freed_nodes, freed_bags, advance};
}

// One expected accounting tuple; `allocated` is checked only in debug builds,
// where the sweep reports it.
struct ExpectedAccounting {
    size_t reachable;       // nodes linked from the head, dead ones included
    size_t reachable_dead;  // marked subset of `reachable`
    size_t retired;         // on the retired list
    size_t bagged;          // in sealed generations' bags
    size_t free;            // on the free list
    long handle_refs;       // sum of every generation's refs == live handles
    size_t allocated;       // nodes from `new` not yet deleted (debug only)
}; // ExpectedAccounting

// Compares the quiescent accounting sweep with `e`. `consistent` is a FATAL
// check: once it is false the list's chains may be cyclic or shared (for
// example a node pushed twice onto the retired list links to itself), and
// continuing would walk them. Use through ACCOUNTING() so that the fatal
// failure also returns from the test body.
template <typename List>
void check_accounting(const List& list, const ExpectedAccounting& e, const char* where) {
    SCOPED_TRACE(where);
    const typename List::InternalAccounting a = list.get_internal_accounting();
#ifndef NDEBUG
    ASSERT_TRUE(a.consistent) << "accounting sweep not consistent";
    EXPECT_EQ(a.allocated, e.allocated) << "allocated";
#endif
    EXPECT_EQ(a.reachable, e.reachable) << "reachable";
    EXPECT_EQ(a.reachable_dead, e.reachable_dead) << "reachable_dead";
    EXPECT_EQ(a.retired, e.retired) << "retired";
    EXPECT_EQ(a.bagged, e.bagged) << "bagged";
    EXPECT_EQ(a.free, e.free) << "free";
    EXPECT_EQ(a.handle_refs, e.handle_refs) << "handle_refs";
    EXPECT_EQ(list.get_internal_retired_count(), e.retired) << "get_internal_retired_count";
} // check_accounting()

// ACCOUNTING(list, "where", R, D, Rt, B, F, H, A): the tuple check of the step
// scripts; returns from the test on an inconsistent sweep.
#define ACCOUNTING(list, where, ...) \
    ASSERT_NO_FATAL_FAILURE(check_accounting(list, ExpectedAccounting{__VA_ARGS__}, where))

// One "cycle": a front erase under a scoped handle that is dropped BEFORE the
// reclaim, then the reclaim. With no other handle pinning, the cycle's own
// generation is sealed and freed in the same call ({1, 1, advanced}).
template <typename List>
ReclaimResult cycle(List& list) {
    {
        typename List::handle h = list.new_handle();
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    }
    return list.reclaim();
} // cycle()

// The iterator on the first node holding `v`, walking from before_begin(h)
// through dead nodes too (iteration does not skip them); end() if none.
template <typename List>
typename List::iterator find(List& list, const typename List::handle& h, int v) {
    typename List::iterator it = list.before_begin(h);
    ++it;
    while (it != List::end() && it->v != v) ++it;
    return it;
} // find()

// Front-inserts the values in `vals` in order, so the list reads them
// reversed: insert_values(list, h, {3, 2, 1}) gives 1, 2, 3.
template <typename List>
void insert_values(List& list, const typename List::handle& h, std::initializer_list<int> vals) {
    for (int v : vals) EXPECT_TRUE(list.insert_after(h, list.before_begin(h), Tracked(v)));
}

// Front-inserts n nodes holding n, n-1, ..., 1 under a scoped handle.
template <typename List>
void insert_n(List& list, int n) {
    typename List::handle hs = list.new_handle();
    for (int v = n; v >= 1; --v) EXPECT_TRUE(list.insert_after(hs, list.before_begin(hs), Tracked(v)));
}

// The number of calls a hook slot received since its count was `base`.
long calls_since(const HookSlot& slot, long base) { return slot.calls.load() - base; }

} // namespace

// The fixture: the Tracked list of one (kGenerations, Hooks) configuration.
// The base (rcu_test_common.h) resets TestHooks before and after every test,
// records Tracked::alive at the start, and checks in TearDown() that every
// Tracked the test created is gone.
template <typename Config>
class LockFreeListRCUScheme : public RcuListTestBase<Config> {
protected:
    using Base = RcuListTestBase<Config>;
    using List = typename Base::template ListOf<Tracked>;  // the list under test

    // T objects alive now, net of the test's starting count.
    long alive() const { return Tracked::alive.load() - this->tracked_base_; }
}; // LockFreeListRCUScheme

TYPED_TEST_SUITE(LockFreeListRCUScheme, RcuListConfigs, RcuListConfigNames);

// Every test opens with this: the fixture's types under short names.
#define SCHEME_TYPES                                           \
    using List = typename TestFixture::List;                   \
    using handle [[maybe_unused]] = typename List::handle;     \
    using iterator [[maybe_unused]] = typename List::iterator; \
    constexpr size_t K = TestFixture::K;                       \
    (void)K

// Hook scripts open with this: under NoHooks the hooks never fire, so the
// script's interleaving cannot happen and the test is skipped (its code still
// compiles: TestHooks is a concrete type, and arming it is harmless).
#define SKIP_WITHOUT_HOOKS()                                     \
    if constexpr (!TestFixture::kHooked) {                       \
        GTEST_SKIP() << "hook script: runs only with TestHooks"; \
    }

// ---------------------------------------------------------------------------
// Retire at the physical unlink, never at the mark.
// WEAK SPOT: erase_after() marks the target, then tries to unlink it; the two
// steps are separate CASes. A node whose mark succeeded but whose unlink CAS
// lost is STILL LINKED (stranded). Retiring it at the mark would put a linked
// node on the retired list: a reader would walk into it after its bag is
// freed. The hook inserts after the anchor between the mark and the unlink, so
// the unlink CAS fails deterministically. Then the drain must help the
// stranded node out and retire it exactly once (the helping pass is the only
// place it can be retired).
TYPED_TEST(LockFreeListRCUScheme, RetireHappensAtUnlinkNotMark) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    handle h = list.new_handle();
    insert_values(list, h, {3, 2, 1});                       // 1, 2 (X), 3
    ACCOUNTING(list, "step 1: 1, 2, 3 linked", 3, 0, 0, 0, 0, 1, 3);
    EXPECT_EQ(this->alive(), 3);

    // Strand X: mark it, then (inside the window) link 9 after the anchor, so
    // the unlink CAS that expected anchor->next == X fails.
    iterator itA = list.begin(h);
    int ran = 0;
    const long mu0 = TestHooks::mark_unlink.calls.load();
    TestHooks::arm(TestHooks::mark_unlink, [&] {
        ++ran;
        EXPECT_TRUE(list.insert_after(h, itA, Tracked(9)));
    });
    EXPECT_TRUE(list.erase_after(h, itA));                   // marked by us -> true
    EXPECT_EQ(ran, 1);
    EXPECT_EQ(calls_since(TestHooks::mark_unlink, mu0), 1);
    // 1, 9, 2 (dead, stranded), 3; nothing retired: X is still linked.
    ACCOUNTING(list, "step 2: 2 stranded behind 9, nothing retired", 4, 1, 0, 0, 0, 1, 4);
    {
        iterator it = list.begin(h);
        EXPECT_EQ(it->v, 1);
        ++it;
        EXPECT_EQ(it->v, 9);
        ++it;
        EXPECT_EQ(it->v, 2);
        ++it;
        EXPECT_EQ(it->v, 3);
        ++it;
        EXPECT_TRUE(it == List::end());
    } // walk the list through the stranded node

    // Drain: erase 1 (1 pass), erase 9 (1 pass), then X is the head's
    // successor and already marked: a helping pass unlinks and retires it,
    // the retry erases 3 (2 passes), and the final call finds no successor
    // (0 passes): 3 true returns, 4 hook calls, 4 nodes retired (X by the helper).
    const long mu1 = TestHooks::mark_unlink.calls.load();
    int erased = 0;
    while (list.erase_after(h, list.before_begin(h))) ++erased;
    EXPECT_EQ(erased, 3);
    EXPECT_EQ(calls_since(TestHooks::mark_unlink, mu1), 4);
    ACCOUNTING(list, "step 3: drained, 2 retired by the helping pass", 0, 0, 4, 0, 0, 1, 4);

    // Ring [S1/4 C0 ...E]: h pins the sealed slot 0.
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    ACCOUNTING(list, "step 4: sealed, pinned by h", 0, 0, 0, 4, 0, 1, 4);

    h = handle{};                                            // leave slot 0
    EXPECT_EQ(list.reclaim(), rr(4, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 5: freed after h left", 0, 0, 0, 0, 4, 0, 4);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RetireHappensAtUnlinkNotMark

// The marker that LOST the unlink must not retire.
// WEAK SPOT: the eraser that set the mark returns true whether or not its
// unlink CAS won. If it retired on "I marked it" while a helper had already
// unlinked and retired the node, the node is pushed twice: its retire_link
// points to itself. The sweep's retired walk then stops at the repeat with the
// EXPECTED count 1, so only `consistent` sees the bug (hence the fatal check).
TYPED_TEST(LockFreeListRCUScheme, RetireHappensAtUnlinkNotMarkNestedErase) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    handle h = list.new_handle();
    insert_values(list, h, {2, 1});                          // 1, 2 (X)
    ACCOUNTING(list, "step 1: 1, 2 linked", 2, 0, 0, 0, 0, 1, 2);

    // Inside the outer call's window, a nested erase over the same anchor
    // finds X marked, helps it out (unlinks and retires it), retries, and
    // finds no successor: false. The outer unlink CAS then fails.
    iterator itA = list.begin(h);
    int ran = 0;
    bool nested = true;
    const long mu0 = TestHooks::mark_unlink.calls.load();
    TestHooks::arm(TestHooks::mark_unlink, [&] {
        ++ran;
        nested = list.erase_after(h, itA);
    });
    EXPECT_TRUE(list.erase_after(h, itA));
    EXPECT_EQ(ran, 1);
    EXPECT_FALSE(nested);
    EXPECT_EQ(calls_since(TestHooks::mark_unlink, mu0), 2);  // outer pass + nested helping pass
    ACCOUNTING(list, "step 2: 2 retired once, by the nested helping pass", 1, 0, 1, 0, 0, 1, 2);  // X retired exactly once

    int erased = 0;
    while (list.erase_after(h, list.before_begin(h))) ++erased;
    EXPECT_EQ(erased, 1);
    ACCOUNTING(list, "step 3: drained", 0, 0, 2, 0, 0, 1, 2);

    h = handle{};
    EXPECT_EQ(list.reclaim(), rr(2, 1, Advance::advanced));
    ACCOUNTING(list, "step 4: freed", 0, 0, 0, 0, 2, 0, 2);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RetireHappensAtUnlinkNotMarkNestedErase

// A HELPER whose unlink CAS failed must not retire.
// WEAK SPOT: a helping pass (the target was already marked by someone else)
// unlinks on behalf of the marker. If its CAS fails the node is still linked;
// "I am helping, so I retire" would retire a linked node. This needs the hook
// on HELPING passes: it inserts after the anchor inside the helping pass.
TYPED_TEST(LockFreeListRCUScheme, RetireHappensAtUnlinkNotMarkHelperUnlinkFailed) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    handle h = list.new_handle();
    insert_values(list, h, {3, 2, 1});                       // 1, 2 (X), 3
    ACCOUNTING(list, "step 1: 1, 2, 3 linked", 3, 0, 0, 0, 0, 1, 3);

    // Strand X behind a new node 8 (as in RetireHappensAtUnlinkNotMark).
    iterator itA = list.begin(h);
    int ran = 0;
    TestHooks::arm(TestHooks::mark_unlink, [&] {
        ++ran;
        EXPECT_TRUE(list.insert_after(h, itA, Tracked(8)));
    });
    EXPECT_TRUE(list.erase_after(h, itA));
    EXPECT_EQ(ran, 1);
    ACCOUNTING(list, "step 2: 2 stranded behind 8", 4, 1, 0, 0, 0, 1, 4);  // 1, 8, 2 (dead), 3

    // erase_after(8): pass 1 finds X marked and HELPS; inside its window the
    // hook links 9 after 8, so the helper's unlink CAS fails and it must
    // retire nothing. Pass 2 marks and unlinks 9 (true).
    iterator itY1 = find(list, h, 8);
    ASSERT_TRUE(itY1 != List::end());
    const long mu0 = TestHooks::mark_unlink.calls.load();
    TestHooks::arm(TestHooks::mark_unlink, [&] {
        ++ran;
        EXPECT_TRUE(list.insert_after(h, itY1, Tracked(9)));
    });
    EXPECT_TRUE(list.erase_after(h, itY1));
    EXPECT_EQ(ran, 2);
    EXPECT_EQ(calls_since(TestHooks::mark_unlink, mu0), 2);
    // 1, 8, 2 (dead), 3; retired = {9}; X still linked and dead.
    ACCOUNTING(list, "step 3: the helper whose unlink failed retired nothing", 4, 1, 1, 0, 0, 1, 5);

    int erased = 0;
    while (list.erase_after(h, list.before_begin(h))) ++erased;
    EXPECT_EQ(erased, 3);                                    // 1; 8; help 2, then 3
    ACCOUNTING(list, "step 4: drained", 0, 0, 5, 0, 0, 1, 5);

    h = handle{};
    EXPECT_EQ(list.reclaim(), rr(5, 1, Advance::advanced));
    ACCOUNTING(list, "step 5: freed", 0, 0, 0, 0, 5, 0, 5);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RetireHappensAtUnlinkNotMarkHelperUnlinkFailed

// A popped node that fails to link is DISPOSED onto the retired list.
// WEAK SPOT: insert_after() on a dead anchor takes a node from the free list
// before it discovers the anchor is marked. A reader that saw the node on the
// free list may still be about to read its retire_link, so the node cannot go
// back to the free list (ABA on the pop) nor be deleted (use after free): it
// must pass through a generation like any retired node, with its `next`
// marked and its T alive until that generation is freed.
TYPED_TEST(LockFreeListRCUScheme, FailedInsertDisposesToRetired) {
    SCHEME_TYPES;
    List list;
    // All three nodes come from `new`, before any node is free.
    handle hs = list.new_handle();
    insert_values(list, hs, {2, 1, 7});                      // F(7), A(1), B(2)
    ACCOUNTING(list, "step 1: F, A, B linked", 3, 0, 0, 0, 0, 1, 3);
    EXPECT_EQ(this->alive(), 3);

    EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));  // F
    hs = handle{};
    ACCOUNTING(list, "step 2: F retired", 2, 0, 1, 0, 0, 0, 3);

    // [E C0 ...E] o=1 c=1: F's node is the one free node.
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));
    ACCOUNTING(list, "step 3: F's node freed", 2, 0, 0, 0, 1, 0, 3);
    EXPECT_EQ(this->alive(), 2);

    handle h = list.new_handle();                            // joins slot 1
    iterator itA = list.begin(h);
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // A: now a dead anchor
    ACCOUNTING(list, "step 5: A erased, a dead anchor", 1, 0, 1, 0, 1, 1, 3);

    // The insert pops F's node, constructs 99 in it, finds A marked, and
    // disposes the node: free -1, retired +1, allocated unchanged, and the 99
    // stays alive until its generation is freed.
    EXPECT_FALSE(list.insert_after(h, itA, Tracked(99)));
    ACCOUNTING(list, "step 6: the popped node disposed to the retired list", 1, 0, 2, 0, 0, 1, 3);
    EXPECT_EQ(this->alive(), 3);

    // K >= 3: [E S1/2 C0 ...E] o=1 c=2; K = 2: [C0 S1/2] o=1 c=0. h pins slot 1.
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    ACCOUNTING(list, "step 7: sealed, pinned by h", 1, 0, 0, 2, 0, 1, 3);
    EXPECT_EQ(this->alive(), 3);

    h = handle{};
    EXPECT_EQ(list.reclaim(), rr(2, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 8: freed after h left", 1, 0, 0, 0, 2, 0, 3);
    EXPECT_EQ(this->alive(), 1);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.FailedInsertDisposesToRetired

// Freeing is cumulative: a younger bag waits for an older pinned one.
// WEAK SPOT: a handle of generation g can reach nodes retired in g, g + 1, ...
// (it walks the graveyard from a node it parked on). A free pass that skipped
// a pinned oldest bag and freed a younger unpinned one would free B under h0.
// At K = 2 the second reclaim cannot advance at all (the only other slot is the
// pinned sealed one): ring_full, a different but equally exact script.
TYPED_TEST(LockFreeListRCUScheme, CumulativeRule) {
    SCHEME_TYPES;
    List list;
    handle h0 = list.new_handle();
    insert_values(list, h0, {3, 2, 1});                      // A(1), B(2), C(3)
    const iterator itA = list.begin(h0);
    const iterator itB = find(list, h0, 2);
    ASSERT_TRUE(itB != List::end());
    ACCOUNTING(list, "step 1: A, B, C linked", 3, 0, 0, 0, 0, 1, 3);

    EXPECT_TRUE(list.erase_after(h0, list.before_begin(h0)));  // A
    ACCOUNTING(list, "step 2: A retired", 2, 0, 1, 0, 0, 1, 3);

    // [S1/1 C0 ...E] o=0 c=1: h0 pins A's bag.
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    ACCOUNTING(list, "step 3: A's bag pinned by h0", 2, 0, 0, 1, 0, 1, 3);

    {
        handle h1 = list.new_handle();                       // joins slot 1
        EXPECT_TRUE(list.erase_after(h1, list.before_begin(h1)));  // B
    }
    ACCOUNTING(list, "step 4: B retired under h1", 1, 0, 1, 1, 0, 1, 3);

    if constexpr (K >= 3) {
        // [S1/1 S0/1 C0 ...E] o=0 c=2: B's bag is unpinned but younger than
        // the pinned A's, so it must stay.
        EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
        ACCOUNTING(list, "step 5: B's bag waits for the older pinned one", 1, 0, 0, 2, 0, 1, 3);
    } else {
        // The next slot is slot 0, sealed and pinned: refused, B stays retired.
        EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::ring_full));
        ACCOUNTING(list, "step 5: refused, B stays retired", 1, 0, 1, 1, 0, 1, 3);
    }

    // h0's graveyard walk from A reaches B with its value intact.
    iterator it = itA;
    ++it;
    EXPECT_TRUE(it == itB);
    EXPECT_EQ(it->v, 2);
    EXPECT_EQ(itB->v, 2);

    h0 = handle{};                                           // itA, itB now invalid
    if constexpr (K >= 3) {
        // Both bags freed oldest first: [E E C0 ...E] o=2 c=2.
        EXPECT_EQ(list.reclaim(), rr(2, 2, Advance::nothing_retired));
    } else {
        // The free pass advance() runs when the next slot is in use frees A's
        // bag, the advance seals B's, the final pass frees it: [C0 E] o=0 c=0.
        EXPECT_EQ(list.reclaim(), rr(2, 2, Advance::advanced));
    }
    ACCOUNTING(list, "step 7: both bags freed, oldest first", 1, 0, 0, 0, 2, 0, 3);
    EXPECT_EQ(this->alive(), 1);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.CumulativeRule

// Every refresh() starts a new session; only a moved generation re-joins.
// WEAK SPOT: refresh() has a fast path (already current: no join). It must
// still invalidate the handle's iterators (the contract says EVERY refresh
// does), so the session id must be redrawn on the fast path too; and it must
// not re-join (or the fast path is not fast and in_join fires).
TYPED_TEST(LockFreeListRCUScheme, RefreshInvalidates) {
    SCHEME_TYPES;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    List list;
    handle h = list.new_handle();
    insert_values(list, h, {1});                             // A
    iterator it = list.begin(h);
    ACCOUNTING(list, "step 1: A linked", 1, 0, 0, 0, 0, 1, 1);

    const long j0 = TestHooks::join.calls.load();
    EXPECT_FALSE(h.refresh());                               // fast path
    if constexpr (TestFixture::kHooked) EXPECT_EQ(calls_since(TestHooks::join, j0), 0);
    ACCOUNTING(list, "step 2: the fast-path refresh changed nothing", 1, 0, 0, 0, 0, 1, 1);
#ifndef NDEBUG
    EXPECT_DEATH((void)list.erase_after(h, it), "iterator used with a different handle session");
#endif
    (void)it;

    iterator it2 = list.begin(h);
    {
        handle hs = list.new_handle();
        EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));  // A
    }
    // [S1/1 C0 ...E]: h pins slot 0.
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    ACCOUNTING(list, "step 4: A's bag pinned by h", 0, 0, 0, 1, 0, 1, 1);

    const long j1 = TestHooks::join.calls.load();
    EXPECT_TRUE(h.refresh());                                // slow path: [S0/1 C1 ...E]
    if constexpr (TestFixture::kHooked) EXPECT_EQ(calls_since(TestHooks::join, j1), 1);
#ifndef NDEBUG
    EXPECT_DEATH((void)list.erase_after(h, it2), "iterator used with a different handle session");
#endif
    (void)it2;

    // The refresh left slot 0: its bag is free now.
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 7: freed after the slow-path refresh", 0, 0, 0, 0, 1, 1, 1);

    // New iterators work after the refresh (the insert pops A's node).
    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), Tracked(2)));
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    ACCOUNTING(list, "step 8: the reused node retired", 0, 0, 1, 0, 0, 1, 1);

    h = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));
    ACCOUNTING(list, "step 9: freed", 0, 0, 0, 0, 1, 0, 1);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RefreshInvalidates

// A moved handle keeps the session; the empty source leaves nothing.
// WEAK SPOT: the move constructor must transfer the generation membership
// without touching the count. If the source kept its block pointer, its
// destructor would leave the generation the target still relies on, and the
// target's parked iterator would be freed under it.
TYPED_TEST(LockFreeListRCUScheme, MoveKeepsIterators) {
    SCHEME_TYPES;
    List list;
    std::unique_ptr<handle> h1 = std::make_unique<handle>(list.new_handle());
    insert_values(list, *h1, {1});                           // A
    iterator it = list.begin(*h1);
    ACCOUNTING(list, "step 1: A linked under h1", 1, 0, 0, 0, 0, 1, 1);

    handle h2(std::move(*h1));
    EXPECT_FALSE(bool(*h1));
    h1.reset();                                              // destroy the empty source
    ACCOUNTING(list, "step 2: moved to h2, the empty source destroyed", 1, 0, 0, 0, 0, 1, 1);

    // No assert: the session moved with the handle. A has no successor.
    EXPECT_FALSE(list.erase_after(h2, it));
    EXPECT_TRUE(list.erase_after(h2, list.before_begin(h2)));  // A
    ACCOUNTING(list, "step 4: A retired under h2", 0, 0, 1, 0, 0, 1, 1);

    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    EXPECT_EQ(it->v, 1);                                     // A's value intact under h2
    ACCOUNTING(list, "step 5: A's bag pinned by h2", 0, 0, 0, 1, 0, 1, 1);

    h2 = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 6: freed after h2 left", 0, 0, 0, 0, 1, 0, 1);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.MoveKeepsIterators

// Move assignment ENDS the target's session first.
// WEAK SPOT: the target of a move assignment holds a membership of its own; if
// the assignment overwrote the block pointer without leaving, that generation's
// count would never drop and its bag would never be freed. Self-move (through a
// reference: `h = std::move(h)` is a -Wself-move error) must be a no-op that
// keeps the session.
TYPED_TEST(LockFreeListRCUScheme, MoveAssignEndsTarget) {
    SCHEME_TYPES;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    List list;
    handle h2 = list.new_handle();
    insert_values(list, h2, {2, 1});                         // 1, 2
    iterator it2 = list.begin(h2);                           // on 1
    ACCOUNTING(list, "step 1: 1, 2 linked", 2, 0, 0, 0, 0, 1, 2);

    EXPECT_TRUE(list.erase_after(h2, list.before_begin(h2)));  // 1
    ACCOUNTING(list, "step 2: 1 retired", 1, 0, 1, 0, 0, 1, 2);

    // [S1/1 C0 ...E]: h2 pins bag G (node 1).
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    ACCOUNTING(list, "step 3: 1's bag pinned by h2", 1, 0, 0, 1, 0, 1, 2);

    handle h1 = list.new_handle();                           // slot 1
    h2 = std::move(h1);                                      // h2 leaves slot 0: [S0/1 C1 ...E]
    EXPECT_FALSE(bool(h1));
    EXPECT_TRUE(bool(h2));
    ACCOUNTING(list, "step 4: h2 move-assigned from h1, its old generation left", 1, 0, 0, 1, 0, 1, 2);
#ifndef NDEBUG
    EXPECT_DEATH((void)list.erase_after(h2, it2), "iterator used with a different handle session");
#endif
    (void)it2;

    // G is free because the target left its generation.
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 6: 1's bag freed", 1, 0, 0, 0, 1, 1, 2);

    iterator it3 = list.begin(h2);                           // on 2
    handle& alias = h2;
    h2 = std::move(alias);                                   // self-move: no-op
    EXPECT_TRUE(bool(h2));
    ACCOUNTING(list, "step 7: the self-move kept the session", 1, 0, 0, 0, 1, 1, 2);
    EXPECT_FALSE(list.erase_after(h2, it3));                 // session kept: no death

    h2 = handle{};
    ACCOUNTING(list, "step 8: h2 emptied", 1, 0, 0, 0, 1, 0, 2);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.MoveAssignEndsTarget

// The const-list surface.
// WEAK SPOT: one handle type and one iterator template serve const and
// non-const lists; an overload slip (a const reclaim(), a const_iterator that
// converts back to iterator, a const begin() returning iterator) compiles
// silently. Each negative requirement has a positive control on the non-const
// list, so a requirement that fails for an unrelated reason (a misspelt
// expression) cannot pass vacuously.
namespace {
template <typename L, typename T>
concept CanInsert = requires(L l, const typename std::remove_cvref_t<L>::handle& h,
                             typename std::remove_cvref_t<L>::const_iterator a, T v) {
    l.insert_after(h, a, std::move(v));
};
template <typename L>
concept CanEmplaceInt = requires(L l, const typename std::remove_cvref_t<L>::handle& h,
                                 typename std::remove_cvref_t<L>::const_iterator a) {
    l.emplace_after(h, a, 0);
};
template <typename L>
concept CanErase = requires(L l, const typename std::remove_cvref_t<L>::handle& h,
                            typename std::remove_cvref_t<L>::const_iterator a) {
    l.erase_after(h, a);
};
template <typename L>
concept CanReclaim = requires(L l) { l.reclaim(); };
} // namespace

TYPED_TEST(LockFreeListRCUScheme, ConstListChecks) {
    SCHEME_TYPES;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using const_iterator = typename List::const_iterator;

    static_assert(CanInsert<List&, Tracked> && !CanInsert<const List&, Tracked>);
    static_assert(CanEmplaceInt<List&> && !CanEmplaceInt<const List&>);
    static_assert(CanErase<List&> && !CanErase<const List&>);
    static_assert(CanReclaim<List&> && !CanReclaim<const List&>);
    static_assert(requires(const List& l) { { l.new_handle() } -> std::same_as<handle>; });
    static_assert(std::same_as<decltype(std::declval<const List&>().begin(std::declval<const handle&>())), const_iterator>);
    static_assert(std::same_as<decltype(std::declval<const List&>().before_begin(std::declval<const handle&>())), const_iterator>);
    static_assert(std::same_as<decltype(std::declval<List&>().begin(std::declval<const handle&>())), iterator>);
    static_assert(std::same_as<decltype(std::declval<List&>().before_begin(std::declval<const handle&>())), iterator>);
    static_assert(std::is_convertible_v<iterator, const_iterator>);
    static_assert(!std::is_convertible_v<const_iterator, iterator>);
    static_assert(!std::is_constructible_v<iterator, const_iterator>);
    static_assert(std::same_as<decltype(*std::declval<const const_iterator&>()), const Tracked&>);
    static_assert(std::same_as<decltype(*std::declval<const iterator&>()), Tracked&>);
    static_assert(requires(const_iterator c) { c == List::end(); List::end() == c; });

    List list;
    const List& cl = list;
    handle h = cl.new_handle();
    const_iterator c0 = cl.begin(h);
    EXPECT_TRUE(c0 == List::end());
    EXPECT_TRUE(List::end() == c0);
    EXPECT_TRUE(List::cend() == List::end());
    EXPECT_TRUE(const_iterator{} == List::end());
    ACCOUNTING(list, "step 1: empty list, one handle", 0, 0, 0, 0, 0, 1, 0);

    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), Tracked(1)));
    const_iterator c1 = cl.begin(h);
    EXPECT_EQ(c1->v, 1);
    EXPECT_EQ((*c1).v, 1);
    EXPECT_FALSE(c1 == List::end());
    EXPECT_FALSE(List::end() == c1);

    // Positive control: the converted iterator carries the live session.
    iterator it = list.begin(h);
    const_iterator cit = it;
    EXPECT_TRUE(cit == it);
    EXPECT_FALSE(list.erase_after(h, cit));                  // no successor, no death
    ACCOUNTING(list, "step 3: one node linked", 1, 0, 0, 0, 0, 1, 1);

    EXPECT_FALSE(h.refresh());                               // fast path: new session
#ifndef NDEBUG
    EXPECT_DEATH((void)list.erase_after(h, cit), "iterator used with a different handle session");
#endif
    (void)cit;
    h = handle{};
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.ConstListChecks

// Ring refusal, advance axis: the K-th advance with the oldest pinned.
// WEAK SPOT: the ring has exactly K slots and the oldest pinned generation
// holds one. With h0 pinning slot 0, exactly K-1 advances succeed; the K-th
// would overwrite a sealed, pinned slot. A refusal must not consume a slot nor
// lose the retired nodes (they wait on the retired list), and the release
// must free all K bags in one call: through the free pass advance() runs when
// the next slot is in use, the re-check of that slot and the final pass.
TYPED_TEST(LockFreeListRCUScheme, RingRefusalAdvanceAxis) {
    SCHEME_TYPES;
    List list;
    const size_t N = K + 1;
    insert_n(list, static_cast<int>(N));
    ACCOUNTING(list, "step 0: N nodes linked", N, 0, 0, 0, 0, 0, N);

    handle h0 = list.new_handle();                           // [C1 ...E]
    for (size_t i = 1; i <= K - 1; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(cycle(list), rr(0, 0, Advance::advanced));
        // Slots 0..i-1 sealed with one node each (slot 0 pinned), slot i current.
        ACCOUNTING(list, "advance i, the oldest generation pinned", N - i, 0, 0, i, 0, 1, N);
    } // advances 1..K-1, the oldest generation pinned
    EXPECT_EQ(cycle(list), rr(0, 0, Advance::ring_full));    // i = K
    ACCOUNTING(list, "advance K: ring_full", 1, 0, 1, K - 1, 0, 1, N);
    EXPECT_EQ(cycle(list), rr(0, 0, Advance::ring_full));    // i = K + 1: retired grows
    ACCOUNTING(list, "advance K + 1: ring_full, the retired list grows", 0, 0, 2, K - 1, 0, 1, N);

    h0 = handle{};
    // The in-advance free pass frees slots 0..K-2 (K-1 bags of 1), the advance
    // seals slot K-1 with the 2 retired nodes, the final pass frees it: K bags,
    // K + 1 nodes.
    EXPECT_EQ(list.reclaim(), rr(K + 1, K, Advance::advanced));
    ACCOUNTING(list, "recovery: every bag freed", 0, 0, 0, 0, N, 0, N);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RingRefusalAdvanceAxis

// Ring refusal, refs axis: a generation is pinned until its LAST handle
// leaves. WEAK SPOT: the count, not a flag: dropping n-1 of n handles must
// keep the bag; and `nothing_retired` must still report the bag freed by the
// last drop (the free pass runs on every non-contended call).
TYPED_TEST(LockFreeListRCUScheme, RingRefusalRefsAxis) {
    SCHEME_TYPES;
    List list;
    insert_n(list, 1);
    handle ha = list.new_handle();
    handle hb = list.new_handle();
    handle hc = list.new_handle();
    ACCOUNTING(list, "step 0: three handles", 1, 0, 0, 0, 0, 3, 1);

    EXPECT_EQ(cycle(list), rr(0, 0, Advance::advanced));     // [S3/1 C0 ...E]
    ACCOUNTING(list, "step 1: the bag pinned by three handles", 0, 0, 0, 1, 0, 3, 1);
    ha = handle{};
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::nothing_retired));
    ACCOUNTING(list, "step 2: pinned by two", 0, 0, 0, 1, 0, 2, 1);
    hb = handle{};
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::nothing_retired));
    ACCOUNTING(list, "step 3: pinned by one", 0, 0, 0, 1, 0, 1, 1);
    hc = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 4: freed after the last handle left", 0, 0, 0, 0, 1, 0, 1);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RingRefusalRefsAxis

// The ring wraps around, twice, one cycle at a time.
// WEAK SPOT: the modular index arithmetic of `current_`'s successor and of
// `oldest_`, and the SEALED -> EMPTY transition that makes a slot reusable.
// Each cycle seals the current slot and frees it in the same call; after cycle
// n the only non-empty slot is n % K. 2K + 1 cycles cross the wrap twice.
TYPED_TEST(LockFreeListRCUScheme, RingWrap) {
    SCHEME_TYPES;
    List list;
    const size_t N = 2*K + 1;
    insert_n(list, static_cast<int>(N));
    for (size_t n = 1; n <= N; ++n) {
        SCOPED_TRACE(n);
        EXPECT_EQ(cycle(list), rr(1, 1, Advance::advanced));
        ACCOUNTING(list, "cycle n", N - n, 0, 0, 0, n, 0, N);
        EXPECT_EQ(this->alive(), static_cast<long>(N - n));
    } // cycles 1..2K+1
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.RingWrap

// Only one reclaim() runs at a time.
// WEAK SPOT: the try-flag. A nested reclaim() inside the window after publish
// must return `contended` and touch nothing; the outer call must still free
// exactly its own sealed bag (not the node the nested erase retired after the
// publish), and must release the flag (the next call is not `contended`).
TYPED_TEST(LockFreeListRCUScheme, ReclaimIsExclusive) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    {
        handle hs = list.new_handle();
        insert_values(list, hs, {2, 1});                     // X(1), Y(2)
        EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));  // X
    }
    ACCOUNTING(list, "step 0: X retired", 1, 0, 1, 0, 0, 0, 2);

    int ran = 0;
    ReclaimResult nested = rr(99, 99, Advance::advanced);
    const long ap0 = TestHooks::after_publish.calls.load();
    TestHooks::arm(TestHooks::after_publish, [&] {
        ++ran;
        {
            handle h = list.new_handle();                    // joins the NEW current slot
            EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // Y: onto the fresh retired list
        }
        nested = list.reclaim();
    }); // the after-publish action
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));
    EXPECT_EQ(ran, 1);
    EXPECT_EQ(nested, rr(0, 0, Advance::contended));
    EXPECT_EQ(calls_since(TestHooks::after_publish, ap0), 1);  // the nested call made none
    ACCOUNTING(list, "step 1: the outer call freed X, the nested call was contended", 0, 0, 1, 0, 1, 0, 2);  // [E C0 ...E] o=1 c=1

    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));  // the flag was released
    ACCOUNTING(list, "step 2: the flag was released, Y freed", 0, 0, 0, 0, 2, 0, 2);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.ReclaimIsExclusive

// A node retired after the publish belongs to the NEW generation.
// WEAK SPOT: advance() must take the retired list BEFORE publishing the new
// generation. A handle that joins the new generation can park on a node that is
// then erased; if the closing generation's bag took that node (an exchange
// after the publish), the same call's free pass would free it under the new
// handle. Here the handle h1 joins inside the window, parks on Y, and Y is
// erased there; Y must stay alive until h1 leaves.
TYPED_TEST(LockFreeListRCUScheme, LatePushAfterPublish) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    {
        handle hs = list.new_handle();
        insert_values(list, hs, {3, 2, 1});                  // X(1), Y(2), Z(3)
        EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));  // X
    }
    ACCOUNTING(list, "step 0: X retired", 2, 0, 1, 0, 0, 0, 3);
    handle h1;
    iterator itY;

    int ran = 0;
    TestHooks::arm(TestHooks::after_publish, [&] {
        ++ran;
        h1 = list.new_handle();                              // the NEW generation (slot 1)
        itY = list.begin(h1);
        handle h2 = list.new_handle();
        EXPECT_TRUE(list.erase_after(h2, list.before_begin(h2)));  // Y: the fresh retired list
    }); // the after-publish action
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));  // X only
    EXPECT_EQ(ran, 1);
    ASSERT_TRUE(itY != List::end());
    EXPECT_EQ(itY->v, 2);
    ACCOUNTING(list, "step 1: X freed, Y retired after the publish", 1, 0, 1, 0, 1, 1, 3);  // [E C1 ...E] o=1 c=1
    EXPECT_EQ(this->alive(), 2);

    // K >= 3: [E S1/1 C0 ...E] o=1 c=2; K = 2: [C0 S1/1] o=1 c=0. h1 pins Y's bag.
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    EXPECT_EQ(itY->v, 2);
    ACCOUNTING(list, "step 2: Y's bag pinned by h1", 1, 0, 0, 1, 1, 1, 3);

    h1 = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 3: freed after h1 left", 1, 0, 0, 0, 2, 0, 3);
    EXPECT_EQ(this->alive(), 1);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.LatePushAfterPublish

// A stale join increment survives the slot's recycle.
// WEAK SPOT: a joiner loads the current block B, then increments B's count,
// then re-checks that B is still current. Between the load and the increment
// B can be sealed, freed and even recycled as CURRENT again (the ring has only
// K slots); between the increment and the re-check the stale +1 sits on B.
// The four parts drive every outcome deterministically, K cycles at a time:
//   S1  (in_join only) B recycled as CURRENT before the increment: the
//       re-check succeeds on the recycled B.
//   S2  (in_join only) B recycled and freed again: the re-check fails, the
//       join undoes its +1 and retries onto the new current block.
//   A1  (+ in_join_after_bump) B freed before the increment, recycled as
//       CURRENT while the stale +1 sits on it: the stale +1 becomes a
//       legitimate membership. A reclaimer that RESETS refs when a slot
//       becomes current wipes it here.
//   A2  (+ in_join_after_bump) as A1, then B is sealed again with the stale
//       +1 on it: the free pass must treat it as a pin (a conservative delay),
//       and the undo must bring B's count back to 0.
// In A1/A2 the join action runs its cycle and only THEN arms join_after_bump:
// armed up front, the cycle's own nested join would consume it.
TYPED_TEST(LockFreeListRCUScheme, StaleBumpSurvivesRecycleS1) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    const size_t N = K + 1;
    insert_n(list, static_cast<int>(N));
    ACCOUNTING(list, "step 0: N nodes linked", N, 0, 0, 0, 0, 0, N);

    int ran = 0;
    std::vector<ReclaimResult> rrs;
    const long j0 = TestHooks::join.calls.load();
    TestHooks::arm(TestHooks::join, [&] {
        ++ran;
        for (size_t i = 0; i < K; ++i) rrs.push_back(cycle(list));
    });
    handle J = list.new_handle();
    EXPECT_EQ(ran, 1);
    ASSERT_EQ(rrs.size(), K);
    for (size_t i = 0; i < K; ++i) EXPECT_EQ(rrs[i], rr(1, 1, Advance::advanced)) << "cycle " << i;
    EXPECT_EQ(calls_since(TestHooks::join, j0), static_cast<long>(K + 1));
    ACCOUNTING(list, "step 1: J joined the recycled slot", 1, 0, 0, 0, K, 1, N);  // [C1 ...E] o=0 c=0 (gen K)

    iterator itW = list.begin(J);
    EXPECT_EQ(cycle(list), rr(0, 0, Advance::advanced));     // [S1/1 C0 ...E]
    EXPECT_EQ(itW->v, static_cast<int>(N));
    ACCOUNTING(list, "step 2: W's bag pinned by J", 0, 0, 0, 1, K, 1, N);

    J = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 3: freed after J left", 0, 0, 0, 0, N, 0, N);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.StaleBumpSurvivesRecycleS1

TYPED_TEST(LockFreeListRCUScheme, StaleBumpSurvivesRecycleS2) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    const size_t N = K + 2;
    insert_n(list, static_cast<int>(N));
    ACCOUNTING(list, "step 0: N nodes linked", N, 0, 0, 0, 0, 0, N);

    // K + 1 cycles: slot 0 becomes CURRENT again at cycle K and EMPTY at K + 1.
    int ran = 0;
    std::vector<ReclaimResult> rrs;
    const long j0 = TestHooks::join.calls.load();
    TestHooks::arm(TestHooks::join, [&] {
        ++ran;
        for (size_t i = 0; i < K + 1; ++i) rrs.push_back(cycle(list));
    });
    handle J = list.new_handle();
    EXPECT_EQ(ran, 1);
    ASSERT_EQ(rrs.size(), K + 1);
    for (size_t i = 0; i < K + 1; ++i) EXPECT_EQ(rrs[i], rr(1, 1, Advance::advanced)) << "cycle " << i;
    // J's two attempts (the retry is visible) + K + 1 nested joins.
    EXPECT_EQ(calls_since(TestHooks::join, j0), static_cast<long>(K + 3));
    ACCOUNTING(list, "step 1: J retried onto the new current slot", 1, 0, 0, 0, K + 1, 1, N);  // [E C1 ...E] o=1 c=1

    iterator itW = list.begin(J);
    EXPECT_EQ(cycle(list), rr(0, 0, Advance::advanced));
    EXPECT_EQ(itW->v, static_cast<int>(N));
    ACCOUNTING(list, "step 2: W's bag pinned by J", 0, 0, 0, 1, K + 1, 1, N);

    J = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 3: freed after J left", 0, 0, 0, 0, N, 0, N);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.StaleBumpSurvivesRecycleS2

TYPED_TEST(LockFreeListRCUScheme, StaleBumpSurvivesRecycleA1) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    const size_t N = K + 1;
    insert_n(list, static_cast<int>(N));
    ACCOUNTING(list, "step 0: N nodes linked", N, 0, 0, 0, 0, 0, N);

    // J's join loads the current block, slot 0. Inside in_join (after that
    // load, before the increment) one cycle seals and frees slot 0 (current =
    // slot 1). J's +1 then lands on the EMPTY slot 0. Inside in_join_after_bump
    // (after the increment, before the re-check) K - 1 cycles run, the last of
    // which recycles slot 0 as CURRENT with the +1 on it. The re-check that
    // slot 0 is current succeeds.
    int ran1 = 0;
    int ran2 = 0;
    ReclaimResult ra = rr(99, 99, Advance::contended);
    std::vector<ReclaimResult> rb;
    const long j0 = TestHooks::join.calls.load();
    const long jb0 = TestHooks::join_after_bump.calls.load();
    TestHooks::arm(TestHooks::join, [&] {
        ++ran1;
        ra = cycle(list);
        TestHooks::arm(TestHooks::join_after_bump, [&] {
            ++ran2;
            for (size_t j = 0; j < K - 1; ++j) rb.push_back(cycle(list));
        });
    }); // the join action (arms the after-bump action)
    handle J = list.new_handle();
    EXPECT_EQ(ran1, 1);
    EXPECT_EQ(ran2, 1);
    EXPECT_EQ(ra, rr(1, 1, Advance::advanced));
    ASSERT_EQ(rb.size(), K - 1);
    for (size_t j = 0; j < K - 1; ++j) EXPECT_EQ(rb[j], rr(1, 1, Advance::advanced)) << "cycle " << j;
    EXPECT_EQ(calls_since(TestHooks::join, j0), static_cast<long>(K + 1));
    EXPECT_EQ(calls_since(TestHooks::join_after_bump, jb0), static_cast<long>(K + 1));
    // [C1 ...E] o=0 c=0. handle_refs == 1 is the release-build witness that the
    // stale +1 survived the recycle as J's membership: a reclaimer that reset
    // the count when slot 0 became current reads 0 here, with or without the
    // debug "refs went negative" assert, and frees W's bag under J below.
    ACCOUNTING(list, "step 1: the stale increment became J's membership", 1, 0, 0, 0, K, 1, N);

    // The +1 is J's real membership: W's bag must wait for J.
    iterator itW = list.begin(J);
    EXPECT_EQ(cycle(list), rr(0, 0, Advance::advanced));     // [S1/1 C0 ...E]
    EXPECT_EQ(itW->v, static_cast<int>(N));
    ACCOUNTING(list, "step 2: W's bag pinned by J", 0, 0, 0, 1, K, 1, N);

    J = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 3: freed after J left", 0, 0, 0, 0, N, 0, N);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.StaleBumpSurvivesRecycleA1

TYPED_TEST(LockFreeListRCUScheme, StaleBumpSurvivesRecycleA2) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    const size_t N = K + 2;
    insert_n(list, static_cast<int>(N));
    ACCOUNTING(list, "step 0: N nodes linked", N, 0, 0, 0, 0, 0, N);

    // As A1, but K cycles run between the increment and the re-check: the
    // first K - 1 recycle slot 0 as CURRENT with J's stale +1 on it, the K-th
    // seals it; its free pass reads the +1 and stops ({0, 0, advanced}). The
    // re-check fails, the undo takes slot 0 to 0, and the retry joins slot 1.
    int ran1 = 0;
    int ran2 = 0;
    ReclaimResult ra = rr(99, 99, Advance::contended);
    std::vector<ReclaimResult> rb;
    const long j0 = TestHooks::join.calls.load();
    const long jb0 = TestHooks::join_after_bump.calls.load();
    TestHooks::arm(TestHooks::join, [&] {
        ++ran1;
        ra = cycle(list);
        TestHooks::arm(TestHooks::join_after_bump, [&] {
            ++ran2;
            for (size_t j = 0; j < K; ++j) rb.push_back(cycle(list));
        });
    }); // the join action (arms the after-bump action)
    handle J = list.new_handle();
    EXPECT_EQ(ran1, 1);
    EXPECT_EQ(ran2, 1);
    EXPECT_EQ(ra, rr(1, 1, Advance::advanced));
    ASSERT_EQ(rb.size(), K);
    for (size_t j = 0; j + 1 < K; ++j) EXPECT_EQ(rb[j], rr(1, 1, Advance::advanced)) << "cycle " << j;
    EXPECT_EQ(rb[K - 1], rr(0, 0, Advance::advanced));
    EXPECT_EQ(calls_since(TestHooks::join, j0), static_cast<long>(K + 3));
    EXPECT_EQ(calls_since(TestHooks::join_after_bump, jb0), static_cast<long>(K + 3));
    // [S0/1 C1 ...E] o=0 c=1. Release-build witnesses (no assert needed) of the
    // stale +1 kept through the recycle: the K-th cycle freed nothing, one bag
    // is still sealed, and handle_refs is 1 (J's retry on slot 1).
    ACCOUNTING(list, "step 1: the stale increment delays the sealed bag", 1, 0, 0, 1, K, 1, N);

    // The delayed bag: free now that the undo brought slot 0's count to 0.
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 2: the delayed bag freed after the undo", 1, 0, 0, 0, K + 1, 1, N);  // [E C1 ...E] o=1 c=1

    iterator itW = list.begin(J);
    EXPECT_EQ(cycle(list), rr(0, 0, Advance::advanced));
    EXPECT_EQ(itW->v, static_cast<int>(N));
    ACCOUNTING(list, "step 3: W's bag pinned by J", 0, 0, 0, 1, K + 1, 1, N);

    J = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 4: freed after J left", 0, 0, 0, 0, N, 0, N);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.StaleBumpSurvivesRecycleA2

// A join whose block went stale before its increment must retry.
// WEAK SPOT: the re-verify after the increment. Inside in_join (after the
// joiner loaded the current block, before its increment), a nested handle
// erases the only node and a reclaim seals and frees that block. Without the
// re-check, J would sit on an EMPTY slot that the next free pass ignores, and
// the node J later parks on would be freed under it.
TYPED_TEST(LockFreeListRCUScheme, JoinRace) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    List list;
    insert_n(list, 1);                                       // X
    ACCOUNTING(list, "step 0: X linked", 1, 0, 0, 0, 0, 0, 1);

    int ran = 0;
    ReclaimResult rh = rr(99, 99, Advance::contended);
    const long j0 = TestHooks::join.calls.load();
    TestHooks::arm(TestHooks::join, [&] {
        ++ran;
        {
            handle h2 = list.new_handle();
            EXPECT_TRUE(list.erase_after(h2, list.before_begin(h2)));
        }
        rh = list.reclaim();
    }); // the join action
    handle J = list.new_handle();
    EXPECT_EQ(ran, 1);
    EXPECT_EQ(rh, rr(1, 1, Advance::advanced));
    EXPECT_EQ(calls_since(TestHooks::join, j0), 3);          // attempt 1, h2's join, retry
    ACCOUNTING(list, "step 1: J retried, X freed", 0, 0, 0, 0, 1, 1, 1);  // [E C1 ...E]; slot 0 refs 0

    EXPECT_TRUE(list.insert_after(J, list.before_begin(J), Tracked(2)));  // Y: pops X's node
    iterator itY = list.begin(J);
    ACCOUNTING(list, "step 2: Y reuses X's node", 1, 0, 0, 0, 0, 1, 1);
    EXPECT_EQ(this->alive(), 1);

    {
        handle h3 = list.new_handle();
        EXPECT_TRUE(list.erase_after(h3, list.before_begin(h3)));  // Y
    }
    ACCOUNTING(list, "step 3: Y retired", 0, 0, 1, 0, 0, 1, 1);

    // K >= 3: [E S1/1 C0 ...E]; K = 2: [C0 S1/1]. J pins Y's bag.
    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
    EXPECT_EQ(itY->v, 2);
    ACCOUNTING(list, "step 4: Y's bag pinned by J", 0, 0, 0, 1, 0, 1, 1);

    J = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    ACCOUNTING(list, "step 5: freed after J left", 0, 0, 0, 0, 1, 0, 1);
    EXPECT_EQ(this->alive(), 0);
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.JoinRace

// Freed nodes are reused, and a stale iterator on a reused node is caught.
// WEAK SPOT: the free-list pop. The popped node must be the one freed (LIFO,
// the only free node), must not come from `new` (allocated unchanged), must be
// unpoisoned before construction, and must bump its debug incarnation so that
// an iterator parked on its previous life dies in `*` instead of reading the
// new value silently.
TYPED_TEST(LockFreeListRCUScheme, FreeListReuse) {
    SCHEME_TYPES;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    List list;
    handle hs = list.new_handle();
    insert_values(list, hs, {1});                            // A
    const Tracked* const addrA = &*list.begin(hs);
#ifndef NDEBUG
    const uint32_t incA = list.get_internal_incarnation(list.begin(hs));
#endif
    const iterator stale = list.begin(hs);
    ACCOUNTING(list, "step 1: A linked", 1, 0, 0, 0, 0, 1, 1);

    EXPECT_TRUE(list.erase_after(hs, list.before_begin(hs)));
    hs = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));
    ACCOUNTING(list, "step 2: A's node free and poisoned", 0, 0, 0, 0, 1, 0, 1);  // `consistent` includes the poison check of free nodes
#ifdef SCHEME_TEST_HAVE_ASAN
    // The same fact checked from outside: A's value bytes (the node's first
    // bytes) are poisoned while the node is free. Reading them is not done
    // here: it would abort the test binary.
    EXPECT_TRUE(__asan_address_is_poisoned(addrA));
    EXPECT_TRUE(__asan_address_is_poisoned(reinterpret_cast<const char*>(addrA) + sizeof(Tracked) - 1));
#endif

    handle h2 = list.new_handle();
    EXPECT_TRUE(list.insert_after(h2, list.before_begin(h2), Tracked(2)));
    EXPECT_EQ(&*list.begin(h2), addrA);
#ifndef NDEBUG
    EXPECT_EQ(list.get_internal_incarnation(list.begin(h2)), static_cast<uint32_t>(incA + 1));
#endif
    ACCOUNTING(list, "step 3: the node reused for 2", 1, 0, 0, 0, 0, 1, 1);
#ifndef NDEBUG
    EXPECT_DEATH((void)*stale, "iterator node was freed and reused");
#endif
    (void)stale;

    EXPECT_TRUE(list.erase_after(h2, list.before_begin(h2)));
    h2 = handle{};
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));
    handle h3 = list.new_handle();
    EXPECT_TRUE(list.insert_after(h3, list.before_begin(h3), Tracked(3)));
    EXPECT_EQ(&*list.begin(h3), addrA);
#ifndef NDEBUG
    EXPECT_EQ(list.get_internal_incarnation(list.begin(h3)), static_cast<uint32_t>(incA + 2));
#endif
    ACCOUNTING(list, "step 5: the node reused again for 3", 1, 0, 0, 0, 0, 1, 1);
    h3 = handle{};
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.FreeListReuse

// The T dies at reclaim, once no handle of its generation lives.
// WEAK SPOT: ~T runs in the free pass, not in erase_after(); the free pass must
// respect the eraser's own handle (it pins the bag), and refresh() must leave
// the old generation (or the bag would be pinned forever).
TYPED_TEST(LockFreeListRCUScheme, TrackedDestroyedAtReclaim) {
    SCHEME_TYPES;
    List list;
    handle h = list.new_handle();
    insert_values(list, h, {1});
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    ACCOUNTING(list, "step 1: erased, not destroyed", 0, 0, 1, 0, 0, 1, 1);
    EXPECT_EQ(this->alive(), 1);

    EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));  // [S1/1 C0 ...E]
    EXPECT_EQ(this->alive(), 1);
    ACCOUNTING(list, "step 2: sealed, pinned by h", 0, 0, 0, 1, 0, 1, 1);

    EXPECT_TRUE(h.refresh());                                // [S0/1 C1 ...E]
    EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::nothing_retired));
    EXPECT_EQ(this->alive(), 0);
    ACCOUNTING(list, "step 4: destroyed after the refresh", 0, 0, 0, 0, 1, 1, 1);

    EXPECT_FALSE(h.refresh());
    h = handle{};
    expect_consistent(list, 0, "end of test");
} // LockFreeListRCUScheme.TrackedDestroyedAtReclaim

// The destructor frees all four populations.
// WEAK SPOT: at destruction nodes sit in four disjoint places: linked from the
// head (live, dead-but-linked, stranded), the retired list, sealed bags, and
// the free list (values already destroyed). Each needs its own walk; a
// skipped one leaks its nodes and their T objects. The strand sits on an
// interior edge no later erase passes over, so it survives to the destructor.
TYPED_TEST(LockFreeListRCUScheme, DestructorFreesFourPopulations) {
    SCHEME_TYPES;
    SKIP_WITHOUT_HOOKS();
    {
        List list;
        handle hs = list.new_handle();
        insert_values(list, hs, {8, 7, 6, 5, 4, 3, 2, 1});   // 1, 2, ..., 8
        ACCOUNTING(list, "step 1: 1..8 linked", 8, 0, 0, 0, 0, 1, 8);
        EXPECT_EQ(this->alive(), 8);

        // Strand 3 behind a new 9 on the edge 2 -> 3.
        iterator it2 = find(list, hs, 2);
        ASSERT_TRUE(it2 != List::end());
        int ran = 0;
        TestHooks::arm(TestHooks::mark_unlink, [&] {
            ++ran;
            EXPECT_TRUE(list.insert_after(hs, it2, Tracked(9)));
        });
        EXPECT_TRUE(list.erase_after(hs, it2));
        EXPECT_EQ(ran, 1);
        ACCOUNTING(list, "step 2: 3 stranded behind 9", 9, 1, 0, 0, 0, 1, 9);  // 1, 2, 9, 3 (dead), 4..8
        EXPECT_EQ(this->alive(), 9);

        // Free population: 6, through its own generation.
        EXPECT_TRUE(list.erase_after(hs, find(list, hs, 5)));
        hs = handle{};
        EXPECT_EQ(list.reclaim(), rr(1, 1, Advance::advanced));
        ACCOUNTING(list, "step 3: 6 on the free list", 8, 1, 0, 0, 1, 0, 9);
        EXPECT_EQ(this->alive(), 8);

        // Bagged population: 8, sealed under h0, which is dropped WITHOUT a
        // reclaim. K >= 3: [E S0/1 C0 ...E] o=1 c=2; K = 2: [C0 S0/1] o=1 c=0.
        {
            handle h0 = list.new_handle();
            EXPECT_TRUE(list.erase_after(h0, find(list, h0, 7)));
            EXPECT_EQ(list.reclaim(), rr(0, 0, Advance::advanced));
        }
        ACCOUNTING(list, "step 4: 8 in a sealed bag", 7, 1, 0, 1, 1, 0, 9);

        // Retired-only population: 5, erased after the last reclaim.
        {
            handle h = list.new_handle();
            EXPECT_TRUE(list.erase_after(h, find(list, h, 4)));
        }
        // Reachable: 1, 2, 9, 3 (dead), 4, 7.
        ACCOUNTING(list, "step 5: 5 on the retired list", 6, 1, 1, 1, 1, 0, 9);
        EXPECT_EQ(this->alive(), 8);
        expect_consistent(list, 0, "before destruction");
    } // the list's scope: its destructor frees the four populations
    // The dead-but-linked chain, the retired list and the sealed bag held the
    // 8 live values; the free node's value died at step 3.
    EXPECT_EQ(this->alive(), 0);
} // LockFreeListRCUScheme.DestructorFreesFourPopulations
