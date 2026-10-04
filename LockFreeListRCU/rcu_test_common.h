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
#ifndef INCLUDED_RCU_TEST_COMMON_H
#define INCLUDED_RCU_TEST_COMMON_H

// Shared test infrastructure for the three LockFreeListRCU test sources
// (lock_free_list_rcu_test.C, lock_free_list_rcu_scheme_test.C,
// lock_free_list_rcu_contract_test.C), which link into ONE gtest binary per
// sanitizer. Everything here is therefore `inline` (functions, static data
// members) so that three translation units may include it.
//
// Contents, in order:
// - payloads: `Tracked` (live-instance counter + poisoning destructor) and
//   `TwoArgPayload` (constructible only from (int, int));
// - `HookSlot` / `TestHooks`: the test seams plugged into the list's `Hooks`
//   parameter (one-shot armed actions, call counters, stress-mode yields);
// - gtest printers for `ReclaimResult` and `ReclaimResult::Advance`;
// - accounting helpers: `accounting_string()`, `expect_consistent()`,
//   `drain()`, `expect_quiescent_reclaim()`;
// - the typed-test configuration list `RcuListConfigs` (K in {2, 3, 64} x
//   {NoHooks, TestHooks}) with its name generator, and the fixture base
//   `RcuListTestBase` every typed suite derives from.
//
// Every accounting helper calls the list's test-only sweep and so inherits its
// quiescence precondition: call it only when no list operation, `reclaim()` or
// join is in progress on any thread (every worker joined) and never from
// inside a hook action.

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "lock_free_list_rcu.h"

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

// Element type that counts its live instances and poisons itself on
// destruction. One type serves both oracles the tests need:
// - LEAK / LIFETIME oracle: `alive` is incremented by every constructor and
//   decremented by the destructor, so a test compares it against a base taken
//   before the list existed: after the list is destroyed it must be back to
//   the base (nothing leaked, nothing destroyed twice), and while the list
//   lives `alive - base == reachable + retired + bagged` (a node's T is
//   destroyed exactly when its bag moves to the free list).
// - REUSE / USE-AFTER-FREE oracle: the destructor overwrites `v` with
//   `kPoison` through a volatile store (`~int` writes nothing, and a plain
//   store into an object whose lifetime is ending may be removed as a dead
//   store). A reader that dereferences a node whose value was already
//   destroyed sees `kPoison` in a single-threaded test; under TSan the
//   destructor's write and the reader's read of `v` are a reported data race
//   (unless the reclamation scheme ordered them, which a premature free does
//   not); under ASan a read of a node already on the free list is a
//   use-after-poison report before the value is even looked at.
// Tests insert only non-negative values; `kPoison` is negative, so `intact()`
// (and `v >= 0`) distinguish a live value from a destroyed one.
// All constructors are noexcept: `insert_after()` requires a nothrow move and
// `emplace_after()` a nothrow constructor for its arguments (static_asserts in
// the list). No default constructor: the list never default-constructs T.
struct Tracked {
    static constexpr int kPoison = -0x5a5a5a5b;  // written by the destructor; never a value a test inserts
    static inline std::atomic<long> alive{0};    // constructed minus destroyed Tracked objects, process-wide

    int v;  // the payload value (>= 0 while alive; kPoison once destroyed)

    Tracked(int x) noexcept : v(x) { alive.fetch_add(1, std::memory_order_relaxed); }  // implicit: insert_after(h, a, 5) works
    Tracked(const Tracked& o) noexcept : v(o.v) { alive.fetch_add(1, std::memory_order_relaxed); }
    Tracked(Tracked&& o) noexcept : v(o.v) { alive.fetch_add(1, std::memory_order_relaxed); }  // the source keeps its value
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) = default;
    ~Tracked() {
        *static_cast<volatile int*>(&v) = kPoison;
        alive.fetch_sub(1, std::memory_order_relaxed);
    }

    // True unless this object's destructor has run (read through a pointer to
    // storage that a premature free destroyed).
    bool intact() const noexcept { return v != kPoison; }
}; // Tracked

// Element type constructible ONLY from (int, int): no default, copy or move
// construction and no assignment. A list of it compiles only if the list
// never default-constructs, copies, moves or assigns T, so it is the
// compile-time check that the list has one placement-construction path, no
// default construction and no assignment; it can be used only through
// `emplace_after()` (`insert_after()` takes T by value and needs a move). It
// counts live instances and poisons its fields on destruction exactly as
// `Tracked`.
struct TwoArgPayload {
    static constexpr int kPoison = Tracked::kPoison;  // written into both fields by the destructor
    static inline std::atomic<long> alive{0};         // constructed minus destroyed objects, process-wide

    int a;  // first constructor argument
    int b;  // second constructor argument

    TwoArgPayload(int x, int y) noexcept : a(x), b(y) { alive.fetch_add(1, std::memory_order_relaxed); }
    TwoArgPayload() = delete;
    TwoArgPayload(const TwoArgPayload&) = delete;
    TwoArgPayload(TwoArgPayload&&) = delete;
    TwoArgPayload& operator=(const TwoArgPayload&) = delete;
    TwoArgPayload& operator=(TwoArgPayload&&) = delete;
    ~TwoArgPayload() {
        *static_cast<volatile int*>(&a) = kPoison;
        *static_cast<volatile int*>(&b) = kPoison;
        alive.fetch_sub(1, std::memory_order_relaxed);
    }
}; // TwoArgPayload

static_assert(std::is_nothrow_constructible_v<Tracked, int> && std::is_nothrow_move_constructible_v<Tracked> &&
              std::is_nothrow_copy_constructible_v<Tracked> && !std::is_default_constructible_v<Tracked>);
static_assert(std::is_nothrow_constructible_v<TwoArgPayload, int, int> && !std::is_default_constructible_v<TwoArgPayload> &&
              !std::is_copy_constructible_v<TwoArgPayload> && !std::is_move_constructible_v<TwoArgPayload> &&
              !std::is_copy_assignable_v<TwoArgPayload> && !std::is_move_assignable_v<TwoArgPayload>);

// ---------------------------------------------------------------------------
// Test hooks
// ---------------------------------------------------------------------------

// One hook position's state: a one-shot armed action and a call counter.
struct HookSlot {
    std::atomic<bool> armed{false};  // one-shot gate: true while an action waits for the next call
    std::function<void()> action;    // the armed action; written only by the arming thread while the slot is disarmed
    std::atomic<long> calls{0};      // every call of this hook, armed or not (the seam's call-frequency contract)
}; // HookSlot

// The `Hooks` type the TestHooks instantiations of the list use (it models
// `ListHooks`). Each of the four hook functions counts the call in its slot,
// yields in stress mode, and runs the slot's armed action, if any, ONCE.
//
// Rules for test authors:
// - One-shot: the slot is disarmed and its action MOVED OUT before the action
//   runs, so nested list operations inside the action that reach the same
//   hook only count, and the action may re-arm any slot (its own included)
//   without destroying the running std::function.
// - Arm a slot only when its NEXT call is the one wanted: nested operations
//   reach every hook too. (Example: to act between the join's bump and its
//   re-verify after a nested cycle inside `in_join`, arm `join_after_bump`
//   from inside the `join` action, after the nested operations.)
// - Actions capture locals by reference; handles are reached through
//   references, never owned here (a static handle would outlive the list and
//   trip the destructor's live-handle assert). std::function needs a copyable
//   callable, so a lambda capturing a handle by value does not compile.
// - `reset()` runs in the fixture's SetUp() AND TearDown() (RcuListTestBase):
//   an action armed but never fired cannot fire in a later test, and its
//   dangling captures are destroyed, never invoked.
// - `yield_in_windows` is for stress tests only, where no action is armed.
// All state is static: every TestHooks list instantiation shares one set of
// slots, which is fine because tests run one at a time.
struct TestHooks {
    static inline HookSlot mark_unlink;      // between_mark_and_unlink(): erase_after(), every pass that reaches the unlink CAS
    static inline HookSlot join;             // in_join(): every join attempt, after reading the current generation, before counting in it
    static inline HookSlot join_after_bump;  // in_join_after_bump(): every join attempt, after counting in, before the re-verify
    static inline HookSlot after_publish;    // in_advance_after_publish(): reclaim() that advances, after the publish, before the free pass
    static inline std::atomic<bool> yield_in_windows{false};  // stress mode: every hook call yields the CPU, widening the windows

    static void between_mark_and_unlink() noexcept { fire(mark_unlink); }
    static void in_join() noexcept { fire(join); }
    static void in_join_after_bump() noexcept { fire(join_after_bump); }
    static void in_advance_after_publish() noexcept { fire(after_publish); }

    // Arms `slot` with `f`: the next call of that hook runs `f` once.
    // Precondition: the slot is disarmed (a fresh test, or its previous
    // action already fired) and no other thread can call the hook while this
    // runs -- single-threaded scripts, or from inside an action of the
    // calling thread. The release store publishes `action` to the firing
    // thread's acquire exchange.
    static void arm(HookSlot& slot, std::function<void()> f) {
        slot.action = std::move(f);
        slot.armed.store(true, std::memory_order_release);
    } // TestHooks::arm()

    // Disarms every slot, destroys every pending action without invoking it,
    // zeroes every call counter and turns stress mode off. Precondition: no
    // other thread is inside a hook (between tests).
    static void reset() noexcept {
        for (HookSlot* slot : {&mark_unlink, &join, &join_after_bump, &after_publish}) {
            slot->armed.store(false, std::memory_order_relaxed);
            slot->action = nullptr;
            slot->calls.store(0, std::memory_order_relaxed);
        } // loop over all hook slots
        yield_in_windows.store(false, std::memory_order_relaxed);
    } // TestHooks::reset()

private:
    // Counts the call; in stress mode yields; then, if the slot is armed,
    // claims it (the exchange makes exactly one caller win even if several
    // threads arrive), moves the action out -- disarming BEFORE invoking --
    // and runs it.
    static void fire(HookSlot& slot) noexcept {
        slot.calls.fetch_add(1, std::memory_order_relaxed);
        if (yield_in_windows.load(std::memory_order_relaxed)) std::this_thread::yield();
        if (slot.armed.load(std::memory_order_relaxed) && slot.armed.exchange(false, std::memory_order_acquire)) {
            std::function<void()> f = std::move(slot.action);
            slot.action = nullptr;  // a moved-from std::function is valid but unspecified: make it empty
            f();                    // may re-arm any slot, this one included
        } // if this call claimed the armed action
    } // TestHooks::fire()
}; // TestHooks

static_assert(ListHooks<TestHooks>);

// ---------------------------------------------------------------------------
// gtest printers (found by ADL: ReclaimResult is in the global namespace)
// ---------------------------------------------------------------------------

// Prints an Advance as its enumerator name.
inline void PrintTo(ReclaimResult::Advance advance, std::ostream* os) {
    switch (advance) {
        case ReclaimResult::Advance::nothing_retired: *os << "nothing_retired"; return;
        case ReclaimResult::Advance::advanced: *os << "advanced"; return;
        case ReclaimResult::Advance::ring_full: *os << "ring_full"; return;
        case ReclaimResult::Advance::contended: *os << "contended"; return;
    } // switch over the enumerators
    *os << "Advance(" << static_cast<int>(advance) << ")";  // not an enumerator: print the raw value
} // PrintTo(ReclaimResult::Advance)

// Prints a ReclaimResult as `{freed_nodes, freed_bags, advance}`, the order of
// the brace-initializer the tests compare against.
inline void PrintTo(const ReclaimResult& result, std::ostream* os) {
    *os << "{" << result.freed_nodes << ", " << result.freed_bags << ", ";
    PrintTo(result.advance, os);
    *os << "}";
} // PrintTo(ReclaimResult)

// ---------------------------------------------------------------------------
// Accounting helpers (quiescence precondition: see the top of the file)
// ---------------------------------------------------------------------------

// Formats an InternalAccounting (of any list instantiation) for failure
// messages: every field, the debug-only ones when they exist.
template <typename Accounting>
std::string accounting_string(const Accounting& acc) {
    std::ostringstream os;
    os << "{reachable " << acc.reachable << ", reachable_dead " << acc.reachable_dead << ", retired " << acc.retired
       << ", bagged " << acc.bagged << ", free " << acc.free << ", handle_refs " << acc.handle_refs;
#ifndef NDEBUG
    os << ", allocated " << acc.allocated << ", consistent " << (acc.consistent ? "true" : "false");
#endif
    os << "}";
    return os.str();
} // accounting_string()

// Accounting sweep with the two checks every test makes at every quiescent
// point: the sum of the generations' handle counts equals `expected_handles`
// (the number of live handles of this list the test holds), and, in debug
// builds, the sweep's `consistent` conjunction holds (no node in two
// populations or seen twice, the populations sum to `allocated`, the ring
// state is well formed, free nodes are poisoned under ASan, ...). Failures
// name `where` and print the whole accounting. Returns the sweep so the
// caller can check population counts; a caller that must stop on an
// inconsistency (e.g. before walking a corrupted list again) writes
// `ASSERT_TRUE(expect_consistent(...).consistent)`.
template <typename ListT>
typename ListT::InternalAccounting expect_consistent(const ListT& list, long expected_handles, const std::string& where) {
    const typename ListT::InternalAccounting acc = list.get_internal_accounting();
    EXPECT_EQ(acc.handle_refs, expected_handles) << where << ": live-handle count mismatch; accounting " << accounting_string(acc);
#ifndef NDEBUG
    EXPECT_TRUE(acc.consistent) << where << ": accounting sweep inconsistent; accounting " << accounting_string(acc);
#endif
    return acc;
} // expect_consistent()

// Drains the list with the idiom the contract guarantees to empty it, dead
// nodes included: `while (erase_after(h, before_begin(h))) {}` under a fresh
// handle, then checks that `begin(h) == end()`. A wedged helping path (a
// marked node that no erase ever unlinks) leaves the list non-empty here.
// The handle is destroyed before the function returns. Returns the number of
// successful erasures (true returns), for the caller's conservation count.
// Precondition: no concurrent operation on the list.
template <typename ListT>
size_t drain(ListT& list) {
    typename ListT::handle h = list.new_handle();
    size_t erased = 0;
    while (list.erase_after(h, list.before_begin(h))) ++erased;
    EXPECT_TRUE(list.begin(h) == ListT::end()) << "the drain left nodes in the list";
    return erased;
} // drain()

// The exact end-state oracle at quiescence with NO live handle of this list.
// With no handle (hence no pin and no stale join bump), the FIRST reclaim()
// frees every node on the retired list and in every sealed bag: it returns
// `freed_nodes == retired + bagged` (both taken just before), `advance ==
// advanced` iff the retired list was non-empty (else `nothing_retired`), at
// least one bag per non-empty population, and leaves `retired == bagged == 0`
// with the freed nodes on the free list and the reachable chain untouched. A
// SECOND call has nothing to do: `{0, 0, nothing_retired}`. In debug builds
// every node is then reachable or free (`reachable + free == allocated`).
// `where` names the phase in failure messages. Returns the FIRST call's
// result (its `freed_nodes` completes a caller's count of freed nodes).
template <typename ListT>
ReclaimResult expect_quiescent_reclaim(ListT& list, const std::string& where) {
    const typename ListT::InternalAccounting before = expect_consistent(list, 0, where + ", before the final reclaim()");
    const ReclaimResult first = list.reclaim();
    EXPECT_EQ(first.freed_nodes, before.retired + before.bagged) << where << ": the first reclaim() with no handle must free everything retired";
    EXPECT_EQ(first.advance, before.retired != 0 ? ReclaimResult::Advance::advanced : ReclaimResult::Advance::nothing_retired) << where;
    EXPECT_GE(first.freed_bags, size_t{before.retired != 0} + size_t{before.bagged != 0}) << where;
    const typename ListT::InternalAccounting after = expect_consistent(list, 0, where + ", after the final reclaim()");
    EXPECT_EQ(after.retired, 0u) << where;
    EXPECT_EQ(after.bagged, 0u) << where;
    EXPECT_EQ(after.reachable, before.reachable) << where << ": reclaim() must not touch the reachable chain";
    EXPECT_EQ(after.free, before.free + first.freed_nodes) << where << ": every freed node must land on the free list";
    const ReclaimResult second = list.reclaim();
    EXPECT_EQ(second, (ReclaimResult{0, 0, ReclaimResult::Advance::nothing_retired})) << where << ": the second reclaim() must find nothing";
#ifndef NDEBUG
    EXPECT_EQ(after.reachable + after.free, after.allocated) << where << ": a node is neither reachable nor free";
#endif
    return first;
} // expect_quiescent_reclaim()

// ---------------------------------------------------------------------------
// Typed-test configurations and the fixture base
// ---------------------------------------------------------------------------

// One typed-test configuration: the ring size K and the Hooks type. A
// configuration is not a list type because the tests need lists of several
// element types (int, Tracked, TwoArgPayload) under the same K and hooks;
// `List<T>` builds them.
template <size_t K, ListHooks H>
struct RcuListConfig {
    static constexpr size_t kGenerations = K;                           // the list's ring size
    using Hooks = H;                                                    // NoHooks or TestHooks
    static constexpr bool kHooked = !std::is_same_v<H, NoHooks>;        // hook tests GTEST_SKIP() when false
    template <typename T> using List = LockFreeListRCU<T, K, H>;        // the list of T under this configuration
}; // RcuListConfig

// The one configuration list every typed suite of the three test sources
// uses: K = 2 (the minimum: one current and one closing slot, so `ring_full`
// is reached after a single pinned advance), K = 3 (the smallest ring with a
// sealed bag strictly between the oldest and the current generation, which
// the cumulative-free rule needs), K = 64 (the default), each with and
// without the test hooks (NoHooks compiles to the production code).
using RcuListConfigs = ::testing::Types<
    RcuListConfig<2, NoHooks>,
    RcuListConfig<3, NoHooks>,
    RcuListConfig<64, NoHooks>,
    RcuListConfig<2, TestHooks>,
    RcuListConfig<3, TestHooks>,
    RcuListConfig<64, TestHooks>
>;

// Readable typed-test names: "K2_NoHooks", ..., "K64_TestHooks" instead of
// gtest's "0" .. "5". Pass as the third argument of TYPED_TEST_SUITE.
struct RcuListConfigNames {
    template <typename Config>
    static std::string GetName(int) {
        return "K" + std::to_string(Config::kGenerations) + (Config::kHooked ? "_TestHooks" : "_NoHooks");
    }
}; // RcuListConfigNames

// Fixture base of every typed suite: `template <typename C> class MySuite :
// public RcuListTestBase<C> {};` then `TYPED_TEST_SUITE(MySuite,
// RcuListConfigs, RcuListConfigNames);`. Provides the configuration's types
// and brackets every test:
// - SetUp() resets TestHooks and records the payloads' live counts;
// - TearDown() resets TestHooks again (no armed action outlives its test)
//   and checks that every Tracked and TwoArgPayload the test created is gone.
//   GoogleTest runs TearDown() before it destroys the fixture's members, so
//   that check holds only because lists live in the test BODIES, never in a
//   fixture member: keep it that way.
// Death-test children exit inside the death statement and never reach
// TearDown().
template <typename Config>
class RcuListTestBase : public ::testing::Test {
protected:
    static constexpr size_t K = Config::kGenerations;    // ring size of this configuration
    using Hooks = typename Config::Hooks;                // NoHooks or TestHooks
    static constexpr bool kHooked = Config::kHooked;     // true under TestHooks
    template <typename T> using ListOf = typename Config::template List<T>;  // the list of T under this configuration

    void SetUp() override {
        TestHooks::reset();
        tracked_base_ = Tracked::alive.load();
        two_arg_base_ = TwoArgPayload::alive.load();
    } // RcuListTestBase::SetUp()

    void TearDown() override {
        TestHooks::reset();
        EXPECT_EQ(Tracked::alive.load(), tracked_base_) << "Tracked payloads leaked (or destroyed twice) by this test";
        EXPECT_EQ(TwoArgPayload::alive.load(), two_arg_base_) << "TwoArgPayload payloads leaked (or destroyed twice) by this test";
    } // RcuListTestBase::TearDown()

    // Stress mode: under TestHooks every hook call yields the CPU, widening
    // the protocol windows the hooks sit in (mark -> unlink, join -> bump ->
    // re-verify, publish -> free pass); a no-op under NoHooks. Call before
    // starting the threads; TearDown()'s reset() turns it off.
    static void enable_yield_in_windows() noexcept {
        if constexpr (kHooked) TestHooks::yield_in_windows.store(true, std::memory_order_relaxed);
    }

    long tracked_base_ = 0;  // Tracked::alive when the test started
    long two_arg_base_ = 0;  // TwoArgPayload::alive when the test started
}; // RcuListTestBase

#endif // INCLUDED_RCU_TEST_COMMON_H
