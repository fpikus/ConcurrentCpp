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
// Black-box contract tests of LockFreeListRCU<T, kGenerations, Hooks>.
//
// Every test here is derived from the PUBLIC contract only: the overview and
// the declaration comments of lock_free_list_rcu.h. None of them depends on
// how the list is implemented; an expected value is either stated by the
// contract or derived from its statements, and every derivation is spelled
// out next to the assertion that uses it. Where the contract does not decide
// an outcome, the test accepts every outcome the contract allows and says so.
// The test-only accounting sweep is used only through expect_consistent(),
// and only where a contract clause is observable no other way (a handle
// count unchanged by a move, the four node populations).
//
// Suites (all typed over RcuListConfigs: K in {2, 3, 64} x {NoHooks,
// TestHooks}, except the death tests, see there). No hook is ever armed here;
// under TestHooks the concurrent tests only turn on the hooks' yields, which
// widen the protocol windows without changing any outcome.
// - ContractHandle:       empty / live handles, moves, swap, refresh().
// - ContractReclaim:      the ReclaimResult of every documented case and the
//                         single-proceeder rule under a two-thread race.
// - ContractConservation: inserted = erased + remaining; ~T runs in reclaim()
//                         (never in erase_after()) and in the destructor.
// - ContractAccess:       const overloads, iterator conversions, equality,
//                         iteration and the graveyard walk.
// - ContractInsertErase:  insert_after / emplace_after / erase_after results
//                         on live, end() and dead anchors.
// - ContractDeathTest:    the five violations debug builds diagnose.
//
// THE GENERATION MODEL behind the exact ReclaimResult oracles, every piece
// taken from the overview (RECLAIM) and the ReclaimResult comments:
// - new_handle() and refresh() join the CURRENT generation; only a reclaim()
//   that returns `advanced` makes a new generation current.
// - A generation slot is in use while its generation is current or sealed
//   with a bag not yet freed; the ring has kGenerations slots. One slot is
//   always current, so while the oldest generation stays pinned reclaim()
//   can perform kGenerations - 1 advances before it returns `ring_full`.
// - A reclaim() that is not `contended` runs, in order: (0) nothing retired
//   since the last advance -> `nothing_retired`, checked FIRST (it wins over
//   a full ring); (1) if the next generation slot is in use, one free pass
//   oldest-first, and if the slot is still in use -> `ring_full`, no slot
//   consumed; (2) advance; (3) a final free pass, for every result other
//   than `contended`. A free pass frees bags oldest-first and stops at the
//   first bag whose generation's count is not zero: a live handle, or a
//   concurrent join attempt's transient increment. Single-threaded, the
//   count is exactly the generation's live handles.
// - Hence a pin released BEFORE a call never yields `ring_full`: step (1)
//   frees the unpinned bags and the same call advances. `ring_full` is stale
//   only when another thread releases the pin between steps (1) and (3),
//   which a single-threaded test cannot arrange (not tested here).

#include <gtest/gtest.h>

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "lock_free_list_rcu.h"
#include "rcu_test_common.h"

// Everything local to this file is in an anonymous namespace: the file links
// into one binary with two other test sources, and a same-named type or
// template with a different definition in another TU would be an ODR
// violation.
namespace {

using Advance = ReclaimResult::Advance;

// The expected value of one reclaim() call, in the order the printer uses:
// `{freed_nodes, freed_bags, advance}`.
constexpr ReclaimResult reclaimed(size_t freed_nodes, size_t freed_bags, Advance advance) {
    return ReclaimResult{freed_nodes, freed_bags, advance};
}

// Drains the reclamation of a list with NO live handle and returns the
// freed_nodes of the calls. Precondition: single-threaded, and no handle of
// `list` is alive. Contract basis (the reclaim() sequence): with no handle,
// no bag is pinned, so the FIRST call frees everything -- if nothing was
// retired it reports `nothing_retired` and its final free pass frees every
// sealed bag; otherwise step (1) frees every sealed bag if the next slot is
// in use, it advances, and step (3) frees the bag it just sealed. A second
// call must then find nothing at all: {0, 0, nothing_retired}.
template <typename List>
size_t reclaim_until_idle(List& list) {
    const ReclaimResult first = list.reclaim();
    EXPECT_NE(first.advance, Advance::ring_full) << "no handle is alive: nothing can pin the ring";
    EXPECT_NE(first.advance, Advance::contended);
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired)) << "the first call left work behind";
    return first.freed_nodes;
} // reclaim_until_idle()

// Inserts `values` so that the list reads them in order from begin(h): each
// value goes after before_begin(h), so the last one is inserted first. The
// head is never erased, hence always a live anchor: every insert must succeed.
template <typename List>
void fill(List& list, const typename List::handle& h, std::initializer_list<int> values) {
    for (const int* v = values.end(); v != values.begin();) {
        --v;
        EXPECT_TRUE(list.insert_after(h, list.before_begin(h), *v));
    } // loop over the values, last first
} // fill()

// As fill(), with the values first, first + 1, ..., first + count - 1.
template <typename List>
void fill_sequence(List& list, const typename List::handle& h, int first, int count) {
    for (int v = first + count - 1; v >= first; --v) EXPECT_TRUE(list.insert_after(h, list.before_begin(h), v));
} // fill_sequence()

// The values from begin(h) to end(), read through the const overloads.
// Meaningful only where nothing was erased: iteration does not skip
// logically deleted nodes, and nothing public tells a dead node from a live
// one, so after an erasure tests count with drain() instead.
template <typename List>
std::vector<int> values_of(const List& list, const typename List::handle& h) {
    std::vector<int> values;
    for (auto it = list.begin(h); it != List::cend(); ++it) values.push_back(it->v);
    return values;
} // values_of()

// Payload for the reclaim() race. Its destructor detects two destructors
// running at once: the race only emplaces (no by-value temporary) after the
// head (never a dead anchor), so between the start and the final drain every
// destructor runs inside a reclaim(), and two at once mean that two reclaim()
// calls were freeing at once -- which "ONE call proceeds at a time" forbids.
// The destructor dwells briefly so that an overlap, if the exclusion were
// broken, would be likely to be seen.
struct OverlapProbe {
    static inline std::atomic<int> in_destructor{0};      // destructors running right now
    static inline std::atomic<int> max_in_destructor{0};  // the most ever seen running at once
    static inline std::atomic<long> alive{0};             // constructed minus destroyed probes

    int v;  // the payload (unused beyond keeping the type non-empty)

    explicit OverlapProbe(int x) noexcept : v(x) { alive.fetch_add(1, std::memory_order_relaxed); }
    OverlapProbe(const OverlapProbe&) = delete;
    OverlapProbe& operator=(const OverlapProbe&) = delete;
    ~OverlapProbe() {
        record(in_destructor.fetch_add(1, std::memory_order_acq_rel) + 1);
        for (int dwell = 0; dwell < 64; ++dwell) record(in_destructor.load(std::memory_order_acquire));
        in_destructor.fetch_sub(1, std::memory_order_acq_rel);
        alive.fetch_sub(1, std::memory_order_relaxed);
    }

    // Raises max_in_destructor to `now` if it is larger.
    static void record(int now) noexcept {
        int seen = max_in_destructor.load(std::memory_order_relaxed);
        while (now > seen && !max_in_destructor.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {}
    }

    // Zeroes the counters; called at the start of the test that uses them.
    static void reset() noexcept {
        in_destructor.store(0);
        max_in_destructor.store(0);
        alive.store(0);
    }
}; // OverlapProbe

// Payload constructible only from a std::unique_ptr<int> rvalue, which it
// takes over: after emplace_after() the caller's pointer is null iff the
// argument was forwarded into a constructed value.
struct BoxHolder {
    std::unique_ptr<int> box;  // the taken-over pointer
    explicit BoxHolder(std::unique_ptr<int>&& b) noexcept : box(std::move(b)) {}
}; // BoxHolder

// Dependent requires-expressions for the negative compile-time checks. They
// must be variable templates: a requires-expression that names no template
// parameter is checked at its definition, and an ill-formed one there is a
// hard error, not `false`. Each is used with a positive control on the
// mutable type, so a typo cannot make a negative check pass vacuously.
template <typename L, typename V>
constexpr bool can_insert_v = requires (L& list, const typename std::remove_const_t<L>::handle& h,
                                        typename std::remove_const_t<L>::const_iterator anchor, V v) {
    list.insert_after(h, anchor, std::move(v));
};
template <typename L, typename V>
constexpr bool can_emplace_v = requires (L& list, const typename std::remove_const_t<L>::handle& h,
                                         typename std::remove_const_t<L>::const_iterator anchor, V v) {
    list.emplace_after(h, anchor, v);
};
template <typename L>
constexpr bool can_erase_v = requires (L& list, const typename std::remove_const_t<L>::handle& h,
                                       typename std::remove_const_t<L>::const_iterator anchor) {
    list.erase_after(h, anchor);
};
template <typename L>
constexpr bool can_reclaim_v = requires (L& list) { list.reclaim(); };
template <typename L>
constexpr bool has_handleless_begin_v = requires (L& list) { list.begin(); };
template <typename It, typename V>
constexpr bool can_assign_through_v = requires (const It& it, const V& v) { *it = v; };

} // namespace

// ===========================================================================
// The fixture base of every suite in this file
// ===========================================================================

// RcuListTestBase (hook reset, leak check at TearDown()) plus the live-value
// oracles in the form the tests compare against: the payloads THIS test
// created and has not destroyed (the base counts taken in SetUp()
// subtracted). Outside the anonymous namespace like the suites themselves;
// the Contract prefix keeps the name distinct from the other test sources'.
template <typename Config>
class ContractTestBase : public RcuListTestBase<Config> {
protected:
    // Live Tracked payloads created by this test.
    long alive() const { return Tracked::alive.load() - this->tracked_base_; }
    // Live TwoArgPayload payloads created by this test.
    long two_arg_alive() const { return TwoArgPayload::alive.load() - this->two_arg_base_; }
}; // ContractTestBase

// ===========================================================================
// ContractHandle: the `handle` class contract
// ===========================================================================

template <typename Config>
class ContractHandle : public ContractTestBase<Config> {};
TYPED_TEST_SUITE(ContractHandle, RcuListConfigs, RcuListConfigNames);

// Contract: handle is "Move-only", every special member nothrow, `explicit
// operator bool`; the list is "non-copyable and non-movable", its default
// constructor nothrow; new_handle() is const, nothrow and returns the one
// handle type; refresh() and reclaim() are nothrow.
TYPED_TEST(ContractHandle, SpecialMembersMatchTheDeclaredContract) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    static_assert(std::is_nothrow_default_constructible_v<Handle>);
    static_assert(std::is_nothrow_move_constructible_v<Handle> && std::is_nothrow_move_assignable_v<Handle>);
    static_assert(!std::is_copy_constructible_v<Handle> && !std::is_copy_assignable_v<Handle>);
    static_assert(std::is_nothrow_destructible_v<Handle> && std::is_nothrow_swappable_v<Handle>);
    static_assert(std::is_constructible_v<bool, const Handle&> && !std::is_convertible_v<const Handle&, bool>);
    static_assert(!std::is_copy_constructible_v<List> && !std::is_move_constructible_v<List>);
    static_assert(!std::is_copy_assignable_v<List> && !std::is_move_assignable_v<List>);
    static_assert(std::is_nothrow_default_constructible_v<List>);
    static_assert(std::is_same_v<decltype(std::declval<const List&>().new_handle()), Handle>);
    static_assert(noexcept(std::declval<const List&>().new_handle()));
    static_assert(noexcept(std::declval<Handle&>().refresh()) && noexcept(std::declval<List&>().reclaim()));
    // Contract: "In release builds a handle is one pointer". Compiled only in
    // a release (NDEBUG) build of these tests; the sanitizer builds are debug
    // builds, where the handle also carries its debug fields.
#ifdef NDEBUG
    static_assert(sizeof(Handle) == sizeof(void*), "contract: in release builds a handle is one pointer");
#endif
    SUCCEED();
} // ContractHandle.SpecialMembersMatchTheDeclaredContract

// Contract: "A default-constructed handle is EMPTY. On an empty handle the
// only valid operations are destruction, moving from and to it, `operator
// bool` and `swap`." Exercises exactly those, needing no list at all.
TYPED_TEST(ContractHandle, DefaultConstructedHandleIsEmptyAndSupportsItsPermittedOperations) {
    using Handle = typename TestFixture::template ListOf<Tracked>::handle;
    Handle e;
    EXPECT_FALSE(e);
    Handle moved_to(std::move(e));  // move construction from an empty handle
    EXPECT_FALSE(moved_to);
    EXPECT_FALSE(e);
    Handle assigned;
    assigned = std::move(moved_to);  // move assignment empty -> empty
    EXPECT_FALSE(assigned);
    EXPECT_FALSE(moved_to);
    assigned.swap(e);  // member and free swap of two empty handles
    swap(e, assigned);
    EXPECT_FALSE(e);
    EXPECT_FALSE(assigned);
} // ContractHandle.DefaultConstructedHandleIsEmptyAndSupportsItsPermittedOperations

// Contract: "An empty handle may outlive its list; a live one may not." Two
// empty handles declared before the list (so destroyed after it): one never
// live, one emptied by a move whose live target dies before the list.
TYPED_TEST(ContractHandle, EmptyHandleMayOutliveItsList) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    Handle never_live;
    Handle moved_from;
    {
        List list;
        moved_from = list.new_handle();
        EXPECT_TRUE(moved_from);
        Handle sink(std::move(moved_from));  // declared after the list: destroyed before it
        EXPECT_FALSE(moved_from);
        EXPECT_TRUE(sink);
    } // the list's scope
    EXPECT_FALSE(never_live);
    EXPECT_FALSE(moved_from);
} // ContractHandle.EmptyHandleMayOutliveItsList

// Contract: new_handle() "Postcondition: the handle is live and belongs to
// this list"; "A const list yields handles too"; "One handle type serves
// both". A handle from the const view drives the mutable list's operations,
// and the live-handle count is the number of handles created.
TYPED_TEST(ContractHandle, NewHandleIsLiveOnMutableAndConstList) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    const List& view = list;
    Handle h = list.new_handle();
    Handle hc = view.new_handle();
    EXPECT_TRUE(h);
    EXPECT_TRUE(hc);
    EXPECT_TRUE(list.begin(h) == List::end());  // a new list is empty
    EXPECT_TRUE(view.begin(hc) == List::cend());
    EXPECT_TRUE(list.insert_after(hc, list.before_begin(hc), 4));
    EXPECT_EQ(view.begin(h)->v, 4);
    expect_consistent(list, 2, "two handles created");
} // ContractHandle.NewHandleIsLiveOnMutableAndConstList

// Contract (move constructor): "Transfers `other`'s session; `other` is left
// empty. Iterators obtained under `other` are valid under *this"; "the
// generation's count is unchanged". The protection moves with the session:
// a node erased after the move stays allocated, with its value, until the
// TARGET dies.
TYPED_TEST(ContractHandle, MoveConstructionTransfersSessionIteratorsAndProtection) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    {
        Handle src = list.new_handle();  // joins generation g0
        fill(list, src, {0, 1, 2});
        auto it0 = list.begin(src);  // on 0, to be erased
        auto it1 = it0;
        ++it1;                       // on 1, stays live
        Handle dst(std::move(src));
        EXPECT_FALSE(src);
        EXPECT_TRUE(dst);
        expect_consistent(list, 1, "after the move construction");
        EXPECT_TRUE(list.erase_after(dst, list.before_begin(dst)));  // erases 0
        // g0 is sealed with 0 in its bag; dst holds g0's session: pinned.
        EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));
        EXPECT_EQ(it0->v, 0);  // valid under dst: the value is intact
        EXPECT_TRUE(it0->intact());
        EXPECT_TRUE(list.insert_after(dst, it1, 5));  // it1 is bound to the transferred session
        EXPECT_EQ(this->alive(), 4);
    } // dst (holding g0) ends here
    // Nothing retired since: nothing_retired, whatever K (it wins over a
    // full ring).
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "after the move target died";
    EXPECT_EQ(this->alive(), 3);
} // ContractHandle.MoveConstructionTransfersSessionIteratorsAndProtection

// Contract (move assignment): "Ends this handle's session if it has one ...,
// then transfers `other`'s; `other` is left empty." The target's old
// generation loses its only handle (a bag it pinned becomes freeable), and
// the source's iterators are valid under the target.
TYPED_TEST(ContractHandle, MoveAssignmentEndsTargetSessionThenTransfers) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle target = list.new_handle();  // g0
    fill(list, target, {0, 1});
    EXPECT_TRUE(list.erase_after(target, list.before_begin(target)));      // erases 0
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));         // bag g0 = {0}, pinned by target
    Handle source = list.new_handle();  // g1
    auto anchor = list.before_begin(source);
    target = std::move(source);
    EXPECT_FALSE(source);
    EXPECT_TRUE(target);
    expect_consistent(list, 1, "after the move assignment");
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "the target's old generation was released";
    EXPECT_EQ(this->alive(), 1);
    EXPECT_TRUE(list.insert_after(target, anchor, 7));  // the source's iterator, now under the target
} // ContractHandle.MoveAssignmentEndsTargetSessionThenTransfers

// Contract (move assignment) with an EMPTY source: the target's session ends
// and "then transfers `other`'s" -- none -- so the target is empty and its
// generation released.
TYPED_TEST(ContractHandle, MoveAssignmentFromAnEmptyHandleEndsTheTargetSession) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle target = list.new_handle();  // g0
    fill(list, target, {0, 1});
    EXPECT_TRUE(list.erase_after(target, list.before_begin(target)));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // pinned by target
    target = Handle{};
    EXPECT_FALSE(target);
    expect_consistent(list, 0, "after assigning an empty handle");
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "the target's generation was released";
    EXPECT_EQ(this->alive(), 1);
} // ContractHandle.MoveAssignmentFromAnEmptyHandleEndsTheTargetSession

// Contract: "Self-move-assignment is a no-op." Written through a reference
// alias (`h = std::move(h)` is a -Wself-move error). A no-op keeps the
// session: the generation it pins stays pinned (a session ended and
// re-joined would land in the newer current generation and release the
// bag), and iterators obtained before stay bound to the handle.
TYPED_TEST(ContractHandle, SelfMoveAssignmentThroughAnAliasIsANoOp) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();  // g0
    fill(list, h, {0, 1});
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // bag g0 pinned by h
    auto anchor = list.before_begin(h);
    Handle& alias = h;
    h = std::move(alias);
    EXPECT_TRUE(h);
    expect_consistent(list, 1, "after the self-move");
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired)) << "g0 must still be pinned";
    EXPECT_EQ(this->alive(), 2);
    EXPECT_TRUE(list.insert_after(h, anchor, 7));  // still bound to h's session

    Handle e;
    Handle& empty_alias = e;
    e = std::move(empty_alias);
    EXPECT_FALSE(e);
} // ContractHandle.SelfMoveAssignmentThroughAnAliasIsANoOp

// Contract: swap "Exchanges the sessions (and the iterators each validates)
// of two handles". Both live, in different generations: each handle's
// iterators work under the other afterwards, and ending the handle that
// RECEIVED the old session releases the old generation (were the swap a
// no-op, that handle would hold the new generation and nothing would be
// freed).
TYPED_TEST(ContractHandle, SwapExchangesSessionsIteratorsAndProtection) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h1 = list.new_handle();  // g0
    fill(list, h1, {0, 1});
    EXPECT_TRUE(list.erase_after(h1, list.before_begin(h1)));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // bag g0 pinned by h1
    Handle h2 = list.new_handle();  // g1
    auto it1 = list.before_begin(h1);
    auto it2 = list.before_begin(h2);
    swap(h1, h2);  // the hidden friend, found by ADL
    EXPECT_TRUE(h1);
    EXPECT_TRUE(h2);
    expect_consistent(list, 2, "after the swap");
    EXPECT_TRUE(list.insert_after(h2, it1, 5));
    EXPECT_TRUE(list.insert_after(h1, it2, 6));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired)) << "g0 pinned, now by h2";
    h2 = Handle{};  // ends g0's session
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "g0 released with h2";
    EXPECT_EQ(this->alive(), 3);
} // ContractHandle.SwapExchangesSessionsIteratorsAndProtection

// Contract: swap: "either or both may be empty". Member swap of a live and
// an empty handle moves the session and its iterators to the empty one.
TYPED_TEST(ContractHandle, SwapWithAnEmptyHandleMovesTheSession) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    Handle e;
    auto anchor = list.before_begin(h);
    h.swap(e);
    EXPECT_FALSE(h);
    EXPECT_TRUE(e);
    EXPECT_TRUE(list.insert_after(e, anchor, 3));
    expect_consistent(list, 1, "after swapping into the empty handle");
} // ContractHandle.SwapWithAnEmptyHandleMovesTheSession

// Contract (refresh): "Returns true iff the handle actually moved to a newer
// generation (false: it was already current ...)"; only an `advanced`
// reclaim() makes a newer generation current, and "new handles and refreshed
// handles join it". "Releases the old generation for reclamation once no
// other handle pins it." Iterators obtained after a refresh are bound to it.
TYPED_TEST(ContractHandle, RefreshReturnsTrueIffTheHandleMovedToANewerGeneration) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();  // g0
    fill(list, h, {0, 1, 2});
    EXPECT_FALSE(h.refresh());  // nothing advanced since the join
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
    EXPECT_FALSE(h.refresh());  // a refusal advances nothing
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // g1 current; bag g0 pinned by h
    Handle late = list.new_handle();  // joins g1
    EXPECT_FALSE(late.refresh());
    EXPECT_TRUE(h.refresh());   // g0 -> g1
    EXPECT_FALSE(h.refresh());  // already current
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "g0 released by the refresh";
    EXPECT_EQ(this->alive(), 2);
    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 9));  // a fresh iterator under the refreshed session
} // ContractHandle.RefreshReturnsTrueIffTheHandleMovedToANewerGeneration

// Contract (THREAD SAFETY): "A handle may be transferred to another thread
// after a happens-before edge (e.g., created on one thread, destroyed on
// another after a join); the debug checks record no thread id." The handle
// and an iterator obtained under it on this thread are used, and the handle
// destroyed, on a worker; the thread start and join are the edges.
TYPED_TEST(ContractHandle, HandleMayBeTransferredToAnotherThread) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    fill(list, h, {0, 1});
    auto anchor = list.begin(h);  // on 0
    std::thread worker([&list, &h, anchor] {
        Handle mine(std::move(h));
        EXPECT_TRUE(list.insert_after(mine, anchor, 5));                // anchor's session moved with the handle
        EXPECT_TRUE(list.erase_after(mine, list.before_begin(mine)));  // erases 0
    }); // mine is destroyed on the worker
    worker.join();
    EXPECT_FALSE(h);
    // No handle is alive: the bag sealed by this call is freed by it.
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::advanced));
    EXPECT_EQ(this->alive(), 2);
} // ContractHandle.HandleMayBeTransferredToAnotherThread

// ===========================================================================
// ContractReclaim: ReclaimResult for every documented case
// ===========================================================================

template <typename Config>
class ContractReclaim : public ContractTestBase<Config> {};
TYPED_TEST_SUITE(ContractReclaim, RcuListConfigs, RcuListConfigNames);

// Contract: `nothing_retired` -- "nothing was retired since the last advance;
// no generation slot consumed"; reclaim() "Needs no handle". On a list where
// nothing was ever erased every call refuses with nothing freed, with or
// without live handles, with or without values.
TYPED_TEST(ContractReclaim, NothingRetiredWhenNothingWasErased) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
    Handle h = list.new_handle();
    fill(list, h, {0, 1, 2});
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
    h = Handle{};
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
    EXPECT_EQ(this->alive(), 3);
} // ContractReclaim.NothingRetiredWhenNothingWasErased

// Contract: "The bag a call just sealed is freeable in the SAME call when no
// handle of its generation or an older one is alive"; ~T runs inside
// reclaim(). Two nodes erased under a handle that dies before the call: one
// bag, two nodes, both values destroyed by this call.
TYPED_TEST(ContractReclaim, JustSealedBagIsFreedInTheSameCallWhenUnpinned) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    {
        Handle h = list.new_handle();
        fill(list, h, {0, 1, 2});
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    } // the only handle ends
    EXPECT_EQ(this->alive(), 3);
    EXPECT_EQ(list.reclaim(), reclaimed(2, 1, Advance::advanced));
    EXPECT_EQ(this->alive(), 1);
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
} // ContractReclaim.JustSealedBagIsFreedInTheSameCallWhenUnpinned

// Contract: "A thread that HOLDS a handle and calls `reclaim()` frees nothing
// at or after that handle's generation, because its own handle pins it."
TYPED_TEST(ContractReclaim, OwnHandlePinsWhatItsHolderErases) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    fill(list, h, {0, 1});
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));
    EXPECT_EQ(this->alive(), 2);
} // ContractReclaim.OwnHandlePinsWhatItsHolderErases

// Contract: the oldest sealed bag is freed while its generation's count is
// zero. A handle counts only in the generation it joined, so a handle of a
// YOUNGER generation does not pin an older bag. (The count also includes a
// concurrent join attempt's transient increment; there is none in this
// single-threaded test.)
TYPED_TEST(ContractReclaim, YoungerHandleDoesNotPinAnOlderBag) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle old_handle = list.new_handle();  // g0
    fill(list, old_handle, {0, 1});
    EXPECT_TRUE(list.erase_after(old_handle, list.before_begin(old_handle)));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // bag g0 pinned by old_handle
    Handle young = list.new_handle();  // g1
    old_handle = Handle{};
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "only a g1 handle is alive";
    EXPECT_EQ(this->alive(), 1);
    EXPECT_TRUE(young);
} // ContractReclaim.YoungerHandleDoesNotPinAnOlderBag

// Contract: "Freeing is cumulative -- a bag is never freed while an older bag
// is still pinned". Bag g1 has no handle of its own but is younger than the
// pinned bag g0: it waits; refreshing the pinning handle frees both, oldest
// first, in one call. Needs two sealed bags plus the current generation, so
// K >= 3 (with K = 2 the second advance is refused as ring_full).
TYPED_TEST(ContractReclaim, FreeingIsCumulativeOldestFirst) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    constexpr size_t K = TestFixture::K;
    if (K < 3) GTEST_SKIP() << "needs two sealed generations plus the current one: K >= 3";
    List list;
    Handle old_handle = list.new_handle();  // g0
    fill(list, old_handle, {0, 1, 2});
    EXPECT_TRUE(list.erase_after(old_handle, list.before_begin(old_handle)));  // 0 -> g0
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));           // bag g0 pinned
    {
        Handle young = list.new_handle();  // g1
        EXPECT_TRUE(list.erase_after(young, list.before_begin(young)));      // 1 -> g1
    } // g1 has no handle any more
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // bag g1 unpinned, but g0 is older and pinned
    EXPECT_EQ(this->alive(), 3);
    EXPECT_TRUE(old_handle.refresh());  // g0 -> g2
    EXPECT_EQ(list.reclaim(), reclaimed(2, 2, Advance::nothing_retired)) << "both bags, oldest first";
    EXPECT_EQ(this->alive(), 1);
} // ContractReclaim.FreeingIsCumulativeOldestFirst

// Fills the ring of `list` under the pinning handle `pin` (joined to g0 and
// never refreshed): K - 1 erase + reclaim() rounds, each `advanced` with
// nothing freed (the contract allows kGenerations - 1 advances while the
// oldest generation stays pinned), leaving g0 .. g(K-2) sealed and g(K-1)
// current, every slot in use. Precondition: the list holds at least K - 1
// values.
namespace {
template <typename List>
void fill_ring_under_pin(List& list, const typename List::handle& pin, size_t K) {
    for (size_t advance = 0; advance + 1 < K; ++advance) {
        EXPECT_TRUE(list.erase_after(pin, list.before_begin(pin)));
        EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced)) << "advance " << advance;
    } // loop over the K - 1 advances that fill the ring
} // fill_ring_under_pin()
} // namespace

// Contract: kGenerations - 1 advances while the oldest generation stays
// pinned, then `ring_full` with no slot consumed; when nothing is retired
// and the ring is full as well, `nothing_retired` is reported (step (0) is
// checked first); there is no automatic recovery; and a pin released before
// the call never yields `ring_full`: the first call after the pin's refresh
// frees the K - 1 unpinned bags in step (1), then advances, sealing the
// pending node into g(K-1), which both handles (now in g(K-1)) pin:
// {K - 1, K - 1, advanced}.
TYPED_TEST(ContractReclaim, RingFullWhileTheOldestGenerationIsPinnedThenRecovers) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    constexpr size_t K = TestFixture::K;
    List list;
    Handle pin = list.new_handle();  // g0
    fill_sequence(list, pin, 0, static_cast<int>(K) + 1);
    fill_ring_under_pin(list, pin, K);
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired)) << "full ring, nothing retired";
    EXPECT_TRUE(list.erase_after(pin, list.before_begin(pin)));  // something to advance
    Handle current = list.new_handle();  // g(K-1), the current generation
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::ring_full));
    EXPECT_FALSE(current.refresh());  // no slot consumed: still the current generation
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::ring_full));  // no automatic recovery
    EXPECT_EQ(this->alive(), static_cast<long>(K) + 1);

    EXPECT_TRUE(pin.refresh());  // g0 -> g(K-1): the pin is gone before the call
    EXPECT_EQ(list.reclaim(), reclaimed(K - 1, K - 1, Advance::advanced));
    EXPECT_EQ(this->alive(), 2);
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired)) << "g(K-1) pinned by both handles";
    EXPECT_TRUE(pin.refresh());
    EXPECT_TRUE(current.refresh());
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::nothing_retired)) << "after both handles left g(K-1)";
    EXPECT_EQ(this->alive(), 1);
} // ContractReclaim.RingFullWhileTheOldestGenerationIsPinnedThenRecovers

// Contract: the recovery call's step (3) final free pass also frees the bag
// it just sealed when no handle of its generation or an older one remains,
// giving {kGenerations, kGenerations, advanced}. The ring is filled under
// `pin`, one more node is retired (refused as ring_full), and the pin is
// DESTROYED: no handle at all remains. One call then frees g0 .. g(K-2) in
// step (1), seals g(K-1) holding the pending node, and frees it in step
// (3): K bags, K nodes.
TYPED_TEST(ContractReclaim, RingFullRecoveryFreesTheClosingBagTooWhenNoHandleRemains) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    constexpr size_t K = TestFixture::K;
    List list;
    {
        Handle pin = list.new_handle();  // g0
        fill_sequence(list, pin, 0, static_cast<int>(K) + 1);
        fill_ring_under_pin(list, pin, K);
        EXPECT_TRUE(list.erase_after(pin, list.before_begin(pin)));
        EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::ring_full));
    } // pin ends: no handle remains
    EXPECT_EQ(list.reclaim(), reclaimed(K, K, Advance::advanced));
    EXPECT_EQ(this->alive(), 1);
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));
} // ContractReclaim.RingFullRecoveryFreesTheClosingBagTooWhenNoHandleRemains

// Contract: "the recommended sequence for a thread that both operates on the
// list and reclaims is `h.refresh(); list.reclaim();`". One node erased per
// round under the one handle h; for every K:
// - round 1: refresh() finds no newer generation (false); the call seals g0,
//   which h pins: {0, 0, advanced};
// - every later round: refresh() moves h to the generation the previous call
//   made current (true); the call frees the previous round's bag (in step (1)
//   when the ring is full, as at K = 2, else in step (3)), seals h's
//   generation, which h pins: {1, 1, advanced}. Each node is freed one round
//   after its erasure.
// In every round, refresh() returns true iff the previous call advanced:
// only an `advanced` call makes a newer generation current.
TYPED_TEST(ContractReclaim, RecommendedRefreshThenReclaimSequenceFreesThePreviousRound) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    constexpr int kRounds = 7;
    List list;
    Handle h = list.new_handle();
    fill_sequence(list, h, 0, kRounds + 1);
    size_t freed = 0;
    bool previous_advanced = false;  // whether the previous round's call advanced
    for (int round = 1; round <= kRounds; ++round) {
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
        const bool moved = h.refresh();
        const ReclaimResult r = list.reclaim();
        freed += r.freed_nodes;
        EXPECT_EQ(moved, previous_advanced) << "round " << round;
        EXPECT_EQ(r, reclaimed(round == 1 ? 0 : 1, round == 1 ? 0 : 1, Advance::advanced)) << "round " << round;
        EXPECT_EQ(this->alive(), kRounds + 1 - static_cast<long>(freed)) << "round " << round;
        previous_advanced = r.advance == Advance::advanced;
    } // loop over the rounds
} // ContractReclaim.RecommendedRefreshThenReclaimSequenceFreesThePreviousRound

// Contract: "ONE call proceeds at a time: a `reclaim()` that finds another in
// progress returns `ReclaimResult::Advance::contended` immediately, does
// nothing"; "`contended` implies {0, 0}"; the result "describes exactly what
// this call did". Two threads each run rounds of {new handle, emplace, erase,
// end the handle} then reclaim(). Checked:
// - every contended result is {0, 0, contended};
// - no two OverlapProbe destructors ever ran at once (destructors run only
//   inside a proceeding reclaim(): two at once = two calls proceeding);
// - the calls' freed_nodes plus a final drain add up to exactly the nodes
//   inserted (a double free, or a node freed by two overlapping calls, would
//   overcount), and every probe is destroyed.
// The race need not produce a single `contended` (the contract promises
// exclusion, not contention); the count is printed so a run shows whether
// the window was hit. Under TestHooks the hooks yield inside reclaim(),
// which makes overlaps much more likely.
TYPED_TEST(ContractReclaim, ConcurrentCallsProceedOneAtATime) {
    using List = typename TestFixture::template ListOf<OverlapProbe>;
    using Handle = typename List::handle;
    constexpr int kRounds = 3000;

    // Per-thread results, read after the join.
    struct Tally {
        size_t emplaced = 0;            // emplace_after() returned true
        size_t erased = 0;              // erase_after() returned true
        size_t calls = 0;               // reclaim() calls
        size_t contended = 0;           // ... that returned contended
        size_t contended_not_empty = 0; // ... contended with a non-zero count (a violation)
        size_t freed_nodes = 0;         // sum of freed_nodes
    }; // Tally

    OverlapProbe::reset();
    this->enable_yield_in_windows();
    {
        List list;
        std::atomic<int> ready{0};
        // One racing thread: `first_value` keeps the two threads' values apart.
        auto race = [&list, &ready](Tally& tally, int first_value) {
            ready.fetch_add(1);
            while (ready.load() < 2) {}  // start together
            for (int round = 0; round < kRounds; ++round) {
                {
                    Handle h = list.new_handle();
                    if (list.emplace_after(h, list.before_begin(h), first_value + round)) ++tally.emplaced;
                    if (list.erase_after(h, list.before_begin(h))) ++tally.erased;
                } // this thread's handle ends before its reclaim()
                const ReclaimResult r = list.reclaim();
                ++tally.calls;
                tally.freed_nodes += r.freed_nodes;
                if (r.advance == Advance::contended) {
                    ++tally.contended;
                    if (r.freed_nodes != 0 || r.freed_bags != 0) ++tally.contended_not_empty;
                } // if contended
            } // loop over the rounds
        }; // race
        Tally t1;
        Tally t2;
        std::thread a(race, std::ref(t1), 0);
        std::thread b(race, std::ref(t2), kRounds);
        a.join();
        b.join();

        EXPECT_EQ(t1.contended_not_empty + t2.contended_not_empty, 0u);
        EXPECT_EQ(t1.emplaced + t2.emplaced, 2u*kRounds);  // the head is always a live anchor
        const size_t remaining = drain(list);
        EXPECT_EQ(t1.emplaced + t2.emplaced, t1.erased + t2.erased + remaining);
        const size_t final_freed = reclaim_until_idle(list);
        EXPECT_EQ(t1.freed_nodes + t2.freed_nodes + final_freed, t1.emplaced + t2.emplaced);
        EXPECT_EQ(OverlapProbe::alive.load(), 0);
        EXPECT_LE(OverlapProbe::max_in_destructor.load(), 1) << "two reclaim() calls freed at the same time";
        std::printf("      [contended reclaim() calls: %zu of %zu]\n", t1.contended + t2.contended, t1.calls + t2.calls);
    } // the list's scope
} // ContractReclaim.ConcurrentCallsProceedOneAtATime

// ===========================================================================
// ContractConservation: values inserted = erased + remaining; when ~T runs
// ===========================================================================

template <typename Config>
class ContractConservation : public ContractTestBase<Config> {};
TYPED_TEST_SUITE(ContractConservation, RcuListConfigs, RcuListConfigNames);

// Contract: "a node's `~T` runs inside `reclaim()`, not inside
// `erase_after()`"; erase_after() "Returns true iff THIS call logically
// deleted a node". Single-threaded counts: every true insert and erase is
// one value; neither erasing nor ending the handle destroys a value; the
// reclaim() destroys exactly the erased ones; the drain finds exactly the
// rest.
TYPED_TEST(ContractConservation, ValuesAreDestroyedInReclaimNotInErase) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    size_t inserted = 0;
    size_t erased = 0;
    {
        Handle h = list.new_handle();
        for (int v = 0; v < 8; ++v) inserted += list.insert_after(h, list.before_begin(h), v);
        for (int e = 0; e < 3; ++e) erased += list.erase_after(h, list.before_begin(h));
        EXPECT_EQ(this->alive(), 8) << "erase_after() must not destroy values";
    } // the only handle ends
    EXPECT_EQ(inserted, 8u);
    EXPECT_EQ(erased, 3u);
    EXPECT_EQ(this->alive(), 8) << "ending a handle must not destroy values";
    EXPECT_EQ(list.reclaim(), reclaimed(3, 1, Advance::advanced));
    EXPECT_EQ(this->alive(), 5);
    const size_t remaining = drain(list);
    EXPECT_EQ(inserted, erased + remaining);
    EXPECT_EQ(this->alive(), 5) << "the drain's erasures destroy nothing either";
    EXPECT_EQ(list.reclaim(), reclaimed(5, 1, Advance::advanced));
    EXPECT_EQ(this->alive(), 0);
} // ContractConservation.ValuesAreDestroyedInReclaimNotInErase

// Contract (~LockFreeListRCU): "Destroys every value still constructed and
// frees every node in all four populations"; "A node is in exactly one of
// four populations ...; get_internal_accounting() reports all four". Builds
// a list with nodes in every population -- reachable (3, 4), retired (2),
// bagged (1, sealed while pinned, never freed) and free (0, already
// destroyed) -- checks the four counts once, then destroys the list. The
// oracle is the Tracked live count: every value still constructed must be
// destroyed exactly once. That every NODE is freed is left to the debug
// build's own leak check at destruction (the destructor aborts on a node it
// did not free), which this test exercises with all four populations
// non-empty.
TYPED_TEST(ContractConservation, DestructorDestroysValuesInEveryPopulation) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    {
        List list;
        {
            Handle t = list.new_handle();  // g0
            fill(list, t, {0, 1, 2, 3, 4});
            EXPECT_TRUE(list.erase_after(t, list.before_begin(t)));  // 0
        } // t ends
        EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::advanced));  // 0 -> free list; g1 current
        {
            Handle pin = list.new_handle();  // g1
            EXPECT_TRUE(list.erase_after(pin, list.before_begin(pin)));  // 1
            EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // bag g1 = {1}, pinned
            EXPECT_TRUE(list.erase_after(pin, list.before_begin(pin)));  // 2 -> retired, never reclaimed
        } // pin ends; no reclaim() runs after it
        EXPECT_EQ(this->alive(), 4);  // 1, 2, 3, 4
        const typename List::InternalAccounting acc = expect_consistent(list, 0, "every population populated");
        EXPECT_EQ(acc.reachable, 2u);
        EXPECT_EQ(acc.retired, 1u);
        EXPECT_EQ(acc.bagged, 1u);
        EXPECT_EQ(acc.free, 1u);
    } // the list's scope
    EXPECT_EQ(this->alive(), 0);
} // ContractConservation.DestructorDestroysValuesInEveryPopulation

// Contract (THREAD SAFETY): "Any number of threads may run any list
// operations concurrently, each under its own handle"; erase_after() true iff
// THIS call deleted; the drain idiom "drains the list under contention";
// reclaim() runs ~T only for nodes no handle can observe. Workers with their
// own handles insert (after the head, and after the first node, which may be
// dead), erase, walk and refresh; a reclaimer refreshes its own handle and
// reclaims (the recommended sequence) until they finish. Checked:
// - inserted == erased + remaining (the drain after the join);
// - every value a walk reads is intact (a premature free poisons it; under
//   ASan a free node is poisoned memory, under TSan the destructor's write
//   races the read);
// - after the drain and a final reclaim with no handle every value is
//   destroyed BEFORE the list's destructor: ~T ran in reclaim();
// - every reclaim() freed node is accounted: at least every inserted node,
//   at most that plus the failed inserts (a failed insert into a reused node
//   is freed by a later reclaim(); one into a fresh node is not).
TYPED_TEST(ContractConservation, ConcurrentInsertedEqualsErasedPlusRemaining) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    constexpr int kWorkers = 3;
    constexpr int kOps = 3000;
    constexpr int kWalk = 16;  // nodes a walk visits at most
    this->enable_yield_in_windows();
    std::atomic<size_t> inserted{0};
    std::atomic<size_t> failed_inserts{0};
    std::atomic<size_t> head_insert_failures{0};
    std::atomic<size_t> erased{0};
    std::atomic<size_t> damaged{0};
    std::atomic<size_t> freed{0};
    std::atomic<size_t> contended_not_empty{0};
    std::atomic<int> workers_done{0};
    {
        List list;
        // One worker: a random mix of the list operations under its own handle.
        auto worker = [&](unsigned seed) {
            Handle h = list.new_handle();
            std::minstd_rand rng(seed);
            for (int op = 0; op < kOps; ++op) {
                if (op % 32 == 31) h.refresh();  // no iterator is held across it
                const int value = static_cast<int>(rng() % 1000);
                switch (rng() % 5) {
                case 0:
                    if (list.insert_after(h, list.before_begin(h), value)) {
                        ++inserted;
                    } else {
                        ++head_insert_failures;
                    }
                    break;
                case 1: {
                    auto first = list.begin(h);
                    if (first == List::end()) break;
                    if (list.insert_after(h, first, value)) {
                        ++inserted;
                    } else {
                        ++failed_inserts;  // the first node was erased concurrently
                    }
                    break;
                } // case 1
                case 2:
                    if (list.erase_after(h, list.before_begin(h))) ++erased;
                    break;
                case 3: {
                    auto first = list.begin(h);
                    if (first != List::end() && list.erase_after(h, first)) ++erased;
                    break;
                } // case 3
                default: {
                    int steps = 0;
                    for (auto it = list.begin(h); it != List::end() && steps < kWalk; ++it, ++steps) {
                        if (!it->intact() || it->v < 0) ++damaged;
                    }
                    break;
                } // walk
                } // switch over the operations
            } // loop over the operations
            workers_done.fetch_add(1);
        }; // worker
        // The reclaimer: the recommended sequence until every worker is done.
        auto reclaimer = [&] {
            Handle h = list.new_handle();
            while (workers_done.load() < kWorkers) {
                h.refresh();
                const ReclaimResult r = list.reclaim();
                freed += r.freed_nodes;
                if (r.advance == Advance::contended && (r.freed_nodes != 0 || r.freed_bags != 0)) ++contended_not_empty;
            } // loop until the workers finish
        }; // reclaimer
        std::vector<std::thread> threads;
        for (int w = 0; w < kWorkers; ++w) threads.emplace_back(worker, 17u + w);
        threads.emplace_back(reclaimer);
        for (std::thread& t : threads) t.join();

        EXPECT_EQ(head_insert_failures.load(), 0u) << "the head is never erased: inserting after it must succeed";
        EXPECT_EQ(damaged.load(), 0u) << "a walk read a destroyed value";
        EXPECT_EQ(contended_not_empty.load(), 0u);
        const size_t remaining = drain(list);
        EXPECT_EQ(inserted.load(), erased.load() + remaining);
        freed += reclaim_until_idle(list);
        EXPECT_EQ(this->alive(), 0) << "values outlived the final reclaim()";
        EXPECT_GE(freed.load(), inserted.load());
        EXPECT_LE(freed.load(), inserted.load() + failed_inserts.load());
        expect_consistent(list, 0, "after the final reclaim");
    } // the list's scope
} // ContractConservation.ConcurrentInsertedEqualsErasedPlusRemaining

// Contract (THREAD SAFETY): "concurrent const operations on ONE handle
// (passing it to list operations) are fine". Several threads share one
// handle created on this thread, running inserts, erasures and walks, while
// another thread reclaims without a handle (the shared handle pins its
// generation the whole time, so reclaim() may stop at ring_full -- a refusal,
// not an error). Conservation and value integrity as above.
TYPED_TEST(ContractConservation, ConcurrentOperationsSharingOneHandle) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    constexpr int kWorkers = 3;
    constexpr int kOps = 2000;
    this->enable_yield_in_windows();
    std::atomic<size_t> inserted{0};
    std::atomic<size_t> erased{0};
    std::atomic<size_t> damaged{0};
    std::atomic<int> workers_done{0};
    {
        List list;
        Handle shared = list.new_handle();
        // One worker: inserts, erasures and walks under the shared handle.
        auto worker = [&](unsigned seed) {
            std::minstd_rand rng(seed);
            for (int op = 0; op < kOps; ++op) {
                switch (rng() % 3) {
                case 0:
                    if (list.insert_after(shared, list.before_begin(shared), static_cast<int>(rng() % 1000))) ++inserted;
                    break;
                case 1:
                    if (list.erase_after(shared, list.before_begin(shared))) ++erased;
                    break;
                default:
                    for (auto it = list.begin(shared); it != List::end(); ++it) {
                        if (!it->intact() || it->v < 0) ++damaged;
                    }
                    break;
                } // switch over the operations
            } // loop over the operations
            workers_done.fetch_add(1);
        }; // worker
        std::vector<std::thread> threads;
        for (int w = 0; w < kWorkers; ++w) threads.emplace_back(worker, 31u + w);
        threads.emplace_back([&] {
            while (workers_done.load() < kWorkers) (void)list.reclaim();
        });
        for (std::thread& t : threads) t.join();
        shared = Handle{};

        EXPECT_EQ(damaged.load(), 0u) << "a walk read a destroyed value";
        const size_t remaining = drain(list);
        EXPECT_EQ(inserted.load(), erased.load() + remaining);
        reclaim_until_idle(list);
        EXPECT_EQ(this->alive(), 0) << "values outlived the final reclaim()";
    } // the list's scope
} // ContractConservation.ConcurrentOperationsSharingOneHandle

// ===========================================================================
// ContractAccess: const overloads, conversions, equality, iteration
// ===========================================================================

template <typename Config>
class ContractAccess : public ContractTestBase<Config> {};
TYPED_TEST_SUITE(ContractAccess, RcuListConfigs, RcuListConfigNames);

// Contract (CONST LISTS): a const list's "before_begin(h)/begin(h) return
// const_iterator"; `reference` is `const T&` for const_iterator, `T&` for
// iterator (and `pointer` likewise); end() is `iterator`, cend()
// `const_iterator`. At run time, a const view walks the values, and its
// before_begin() is an anchor for the mutable list's insert.
TYPED_TEST(ContractAccess, ConstListYieldsConstIteratorsAndConstReferences) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    using iterator = typename List::iterator;
    using const_iterator = typename List::const_iterator;
    static_assert(std::is_same_v<decltype(std::declval<List&>().begin(std::declval<const Handle&>())), iterator>);
    static_assert(std::is_same_v<decltype(std::declval<List&>().before_begin(std::declval<const Handle&>())), iterator>);
    static_assert(std::is_same_v<decltype(std::declval<const List&>().begin(std::declval<const Handle&>())), const_iterator>);
    static_assert(std::is_same_v<decltype(std::declval<const List&>().before_begin(std::declval<const Handle&>())), const_iterator>);
    static_assert(std::is_same_v<decltype(*std::declval<const const_iterator&>()), const Tracked&>);
    static_assert(std::is_same_v<decltype(std::declval<const const_iterator&>().operator->()), const Tracked*>);
    static_assert(std::is_same_v<decltype(*std::declval<const iterator&>()), Tracked&>);
    static_assert(std::is_same_v<decltype(std::declval<const iterator&>().operator->()), Tracked*>);
    static_assert(std::is_same_v<decltype(List::end()), iterator> && std::is_same_v<decltype(List::cend()), const_iterator>);

    List list;
    const List& view = list;
    Handle h = list.new_handle();
    fill(list, h, {3, 4});
    const_iterator cit = view.begin(h);
    EXPECT_EQ((*cit).v, 3);
    EXPECT_EQ(cit->v, 3);
    ++cit;
    EXPECT_EQ(cit->v, 4);
    ++cit;
    EXPECT_TRUE(cit == List::cend());
    EXPECT_TRUE(list.insert_after(h, view.before_begin(h), 2));
    EXPECT_EQ(values_of(list, h), (std::vector<int>{2, 3, 4}));
} // ContractAccess.ConstListYieldsConstIteratorsAndConstReferences

// Contract: "iterator -> const_iterator, implicit; the converse does not
// exist"; "The converted iterator is bound to the same handle session ...
// (it is the same iterator, read-only)".
TYPED_TEST(ContractAccess, IteratorConvertsToConstIteratorNeverTheReverse) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    using iterator = typename List::iterator;
    using const_iterator = typename List::const_iterator;
    static_assert(std::is_nothrow_convertible_v<iterator, const_iterator>);
    static_assert(!std::is_convertible_v<const_iterator, iterator> && !std::is_constructible_v<iterator, const_iterator>);
    static_assert(!std::is_assignable_v<iterator&, const_iterator>);

    List list;
    Handle h = list.new_handle();
    fill(list, h, {1, 2});
    iterator it = list.begin(h);
    const_iterator cit = it;
    EXPECT_TRUE(cit == it);
    EXPECT_EQ(&*cit, &*it);  // the same node
    ++it;
    EXPECT_TRUE(list.insert_after(h, cit, 7));  // the converted iterator works with the original's handle
    ++cit;
    EXPECT_EQ(cit->v, 7);
} // ContractAccess.IteratorConvertsToConstIteratorNeverTheReverse

// Contract: insert_after/emplace_after/erase_after/reclaim are non-const
// members (a const list "yields handles too" but only const_iterators), and
// a const_iterator gives `const T&`; begin() always takes a handle ("range-for
// is impossible because `begin` takes a handle"). Negative checks are
// dependent requires-expressions with positive controls; int keeps the
// controls simple (T& assignable).
TYPED_TEST(ContractAccess, ConstListAndConstIteratorRejectMutation) {
    using IntList = typename TestFixture::template ListOf<int>;
    static_assert(can_insert_v<IntList, int> && !can_insert_v<const IntList, int>);
    static_assert(can_emplace_v<IntList, int> && !can_emplace_v<const IntList, int>);
    static_assert(can_erase_v<IntList> && !can_erase_v<const IntList>);
    static_assert(can_reclaim_v<IntList> && !can_reclaim_v<const IntList>);
    static_assert(can_assign_through_v<typename IntList::iterator, int>);
    static_assert(!can_assign_through_v<typename IntList::const_iterator, int>);
    static_assert(!has_handleless_begin_v<IntList> && !has_handleless_begin_v<const IntList>);

    IntList list;
    typename IntList::handle h = list.new_handle();
    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 1));
    *list.begin(h) = 5;  // `reference` of iterator is T&
    EXPECT_EQ(*static_cast<const IntList&>(list).begin(h), 5);
} // ContractAccess.ConstListAndConstIteratorRejectMutation

// Contract: end() "compares equal to every past-the-end and
// default-constructed iterator of every list, in both operand orders.
// `cend()` is the const_iterator form; `end()` converts to it"; a
// default-constructed iterator "Postcondition: *this == end()"; begin() of
// an empty list is end(); `!=` is rewritten from `==`.
TYPED_TEST(ContractAccess, EndCendAndDefaultIteratorsCompareEqualInBothOrders) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    using iterator = typename List::iterator;
    using const_iterator = typename List::const_iterator;
    const iterator def{};
    const const_iterator cdef{};
    EXPECT_TRUE(def == List::end());
    EXPECT_TRUE(List::end() == def);
    EXPECT_TRUE(cdef == List::cend());
    EXPECT_TRUE(List::cend() == cdef);
    EXPECT_TRUE(List::end() == List::cend());
    EXPECT_TRUE(List::cend() == List::end());
    EXPECT_TRUE(def == cdef);
    EXPECT_TRUE(cdef == def);
    EXPECT_FALSE(List::end() != List::cend());
    EXPECT_FALSE(List::cend() != List::end());

    List a;
    List b;
    const List& b_view = b;
    Handle ha = a.new_handle();
    Handle hb = b.new_handle();
    EXPECT_TRUE(a.begin(ha) == b_view.begin(hb));  // two empty lists' begin() are both end()
    EXPECT_TRUE(b_view.begin(hb) == a.begin(ha));
    EXPECT_TRUE(List::cend() == a.begin(ha));
    EXPECT_TRUE(a.begin(ha) == List::cend());
    fill(a, ha, {1});
    EXPECT_TRUE(a.begin(ha) != List::end());
    EXPECT_TRUE(List::end() != a.begin(ha));
    EXPECT_TRUE(cdef != a.begin(ha));
    EXPECT_TRUE(a.begin(ha) != cdef);
} // ContractAccess.EndCendAndDefaultIteratorsCompareEqualInBothOrders

// Contract: equality is "Node identity only"; "`iterator` and
// `const_iterator` compare through the conversion" -- in both orders, and
// across handles (two handles' begin() on one node are equal).
TYPED_TEST(ContractAccess, IteratorsCompareByNodeIdentityInBothOrders) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    using iterator = typename List::iterator;
    using const_iterator = typename List::const_iterator;
    List list;
    const List& view = list;
    Handle h1 = list.new_handle();
    Handle h2 = list.new_handle();
    fill(list, h1, {0, 1});
    const iterator it = list.begin(h1);
    const_iterator cit = view.begin(h2);
    EXPECT_TRUE(it == cit);
    EXPECT_TRUE(cit == it);
    EXPECT_TRUE(list.begin(h1) == list.begin(h2));
    ++cit;
    EXPECT_TRUE(it != cit);
    EXPECT_TRUE(cit != it);
} // ContractAccess.IteratorsCompareByNodeIdentityInBothOrders

// Contract: begin() is "The first node ... or end() if there is none"; `++`
// advances "to the successor ...; past the last node yields end()";
// before_begin()'s iterator is valid and not end(), so `++` on it is allowed
// (only its `*`/`->` are violations) and yields begin().
TYPED_TEST(ContractAccess, IterationVisitsEveryNodeInOrderThenEnd) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    EXPECT_TRUE(list.begin(h) == List::end());
    fill(list, h, {0, 1, 2});
    auto it = list.before_begin(h);
    ++it;
    EXPECT_TRUE(it == list.begin(h));
    for (int expected = 0; expected < 3; ++expected, ++it) {
        ASSERT_TRUE(it != List::end());
        EXPECT_EQ(it->v, expected);
    } // loop over the expected values
    EXPECT_TRUE(it == List::end());
} // ContractAccess.IterationVisitsEveryNodeInOrderThenEnd

// Contract (ITERATOR VALIDITY): "an iterator parked on an erased node keeps
// walking the graveyard back into the live suffix (erasure never rewrites a
// node's next pointer while anyone can still read it)"; the value of a
// logically deleted node "is intact while the iterator is valid"; an
// iterator that walked the graveyard into a live node equals one that
// reached it through the live chain. List 0 1 2 3; park on 1; erase 1, then
// 2; reclaim while the handle lives. 1's next still leads to 2 (erased after
// it) and 2's to 3, the live suffix.
TYPED_TEST(ContractAccess, ParkedIteratorWalksTheGraveyardIntoTheLiveSuffix) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    fill(list, h, {0, 1, 2, 3});
    auto parked = list.begin(h);
    ++parked;                       // on 1
    auto three = parked;
    ++three;
    ++three;                        // on 3, reached through the live chain
    EXPECT_TRUE(list.erase_after(h, list.begin(h)));  // erases 1 (0's successor)
    EXPECT_TRUE(list.erase_after(h, list.begin(h)));  // erases 2 (0's new successor)
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::advanced));  // h pins both
    EXPECT_EQ(parked->v, 1);
    EXPECT_TRUE(parked->intact());
    ++parked;
    ASSERT_TRUE(parked != List::end());
    EXPECT_EQ(parked->v, 2);
    EXPECT_TRUE(parked->intact());
    ++parked;
    EXPECT_TRUE(parked == three);
    EXPECT_EQ(parked->v, 3);
} // ContractAccess.ParkedIteratorWalksTheGraveyardIntoTheLiveSuffix

// ===========================================================================
// ContractInsertErase: results on live, end() and dead anchors
// ===========================================================================

template <typename Config>
class ContractInsertErase : public ContractTestBase<Config> {};
TYPED_TEST_SUITE(ContractInsertErase, RcuListConfigs, RcuListConfigNames);

// Contract: insert_after() inserts "immediately after `anchor`" and "Returns
// true on success"; any live node, the head included, is an anchor; the
// anchor may come from the const view (it is a const_iterator parameter).
TYPED_TEST(ContractInsertErase, InsertAfterLiveAnchorLinksImmediatelyAfterIt) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    const List& view = list;
    Handle h = list.new_handle();
    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 1));  // 1
    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 0));  // 0 1
    EXPECT_TRUE(list.insert_after(h, list.begin(h), 5));         // 0 5 1
    auto last = view.begin(h);
    ++last;
    ++last;
    EXPECT_TRUE(list.insert_after(h, last, 9));                  // 0 5 1 9
    EXPECT_EQ(values_of(list, h), (std::vector<int>{0, 5, 1, 9}));
    EXPECT_EQ(this->alive(), 4);
} // ContractInsertErase.InsertAfterLiveAnchorLinksImmediatelyAfterIt

// Contract: insert_after() returns "false if `anchor` is `end()`"; "`value`
// is consumed in every case"; "an `end()` anchor returns false before any
// allocation". end(), cend() and a default-constructed iterator (equal to
// end()) are all rejected, the list is unchanged, and no value survives the
// call (the by-value parameter dies with it).
TYPED_TEST(ContractInsertErase, InsertAfterEndReturnsFalseAndKeepsNoValue) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    fill(list, h, {0});
    EXPECT_FALSE(list.insert_after(h, List::end(), 4));
    EXPECT_FALSE(list.insert_after(h, List::cend(), 5));
    EXPECT_FALSE(list.insert_after(h, typename List::const_iterator{}, 6));
    EXPECT_EQ(this->alive(), 1);
    EXPECT_EQ(values_of(list, h), (std::vector<int>{0}));
    EXPECT_EQ(list.reclaim(), reclaimed(0, 0, Advance::nothing_retired));  // nothing was retired either
} // ContractInsertErase.InsertAfterEndReturnsFalseAndKeepsNoValue

// Contract: insert_after() returns false if `anchor` is logically deleted.
// For any non-end() anchor a node is obtained (from the free list, else from
// `new`) and the T constructed into it before the anchor is examined; on a
// deleted anchor a freshly allocated node has its T destroyed at once and
// is deleted, and nothing is retired. The free list is filled only by
// reclaim(), so on a list that never reclaimed the node is fresh: the value
// is gone when the call returns, and the later reclaim() frees only the two
// ERASED nodes.
TYPED_TEST(ContractInsertErase, InsertAfterDeadAnchorWithAFreshNodeDestroysTheValueBeforeReturning) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    {
        Handle h = list.new_handle();
        fill(list, h, {10, 20});
        auto dead = list.begin(h);  // on 10
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
        const long before = this->alive();
        EXPECT_FALSE(list.insert_after(h, dead, 7));
        EXPECT_EQ(this->alive(), before);
        EXPECT_EQ(drain(list), 1u);  // only 20 is live: 7 was not linked
    } // h ends
    EXPECT_EQ(list.reclaim(), reclaimed(2, 1, Advance::advanced)) << "10 and 20 only: the fresh node was not retired";
    EXPECT_EQ(this->alive(), 0);
} // ContractInsertErase.InsertAfterDeadAnchorWithAFreshNodeDestroysTheValueBeforeReturning

// Contract: as above, the reused-node branch: with a node on the free list
// the insert takes it, and on a dead anchor that node goes to the retired
// list -- which is not an erasure, so no erase_after() count changes -- and
// a later reclaim() destroys its value and counts it in `freed_nodes`. The
// free list is filled by a reclaim() with no handle alive; the dead anchor
// is a node erased under a handle joined afterwards.
// (The handle's begin() is 20 for certain: 10 was freed, so it was unlinked,
// since a node is retired only at its physical unlink.)
TYPED_TEST(ContractInsertErase, InsertAfterDeadAnchorWithAReusedNodeDestroysTheValueAtALaterReclaim) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    {
        Handle t = list.new_handle();
        fill(list, t, {10, 20});
        EXPECT_TRUE(list.erase_after(t, list.before_begin(t)));  // 10
    } // t ends
    EXPECT_EQ(list.reclaim(), reclaimed(1, 1, Advance::advanced));  // 10's node -> free list
    {
        Handle h = list.new_handle();
        auto dead = list.begin(h);
        ASSERT_EQ(dead->v, 20);
        EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // 20
        const long before = this->alive();
        EXPECT_FALSE(list.insert_after(h, dead, 7));
        EXPECT_EQ(this->alive(), before + 1) << "a reused node's value is destroyed at a later reclaim(), not in the call";
        EXPECT_EQ(drain(list), 0u) << "the disposed node is neither linked nor an erasure";
    } // h ends
    // Both retired since the last advance, no handle left: one bag, 20 and 7.
    EXPECT_EQ(list.reclaim(), reclaimed(2, 1, Advance::advanced));
    EXPECT_EQ(this->alive(), 0);
} // ContractInsertErase.InsertAfterDeadAnchorWithAReusedNodeDestroysTheValueAtALaterReclaim

// Contract: emplace_after() is "As `insert_after`, constructing the value in
// place from `args...`" with "one placement-new path" and "no default
// construction and no assignment of T anywhere" -- TwoArgPayload is
// constructible only from (int, int). Live anchor: true, value built from the
// two arguments; end(): false, nothing constructed; dead anchor on a list
// that never reclaimed (fresh node): false, value destroyed before return.
TYPED_TEST(ContractInsertErase, EmplaceAfterWithTheTwoArgumentPayload) {
    using List = typename TestFixture::template ListOf<TwoArgPayload>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    EXPECT_TRUE(list.emplace_after(h, list.before_begin(h), 3, 4));
    EXPECT_TRUE(list.emplace_after(h, list.before_begin(h), 1, 2));  // 1,2 then 3,4
    auto it = list.begin(h);
    EXPECT_EQ(it->a, 1);
    EXPECT_EQ(it->b, 2);
    ++it;
    EXPECT_EQ((*it).a, 3);
    EXPECT_EQ((*it).b, 4);
    EXPECT_EQ(this->two_arg_alive(), 2);
    EXPECT_FALSE(list.emplace_after(h, List::end(), 5, 6));
    EXPECT_EQ(this->two_arg_alive(), 2);
    auto dead = list.begin(h);
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // 1,2
    EXPECT_FALSE(list.emplace_after(h, dead, 7, 8));
    EXPECT_EQ(this->two_arg_alive(), 2);  // 1,2 (erased, pinned) and 3,4
    EXPECT_EQ(drain(list), 1u);
} // ContractInsertErase.EmplaceAfterWithTheTwoArgumentPayload

// Contract: for any non-end() anchor, emplace_after() obtains a node and
// constructs the T into it from the forwarded arguments before it examines
// the anchor; for an end() anchor nothing is constructed, so the arguments
// are not forwarded. A rvalue unique_ptr is
// left untouched by an end() anchor and taken over by a live anchor and by
// a dead one alike.
TYPED_TEST(ContractInsertErase, EmplaceAfterForwardsArgumentsOnlyWhenInitializingANode) {
    using List = typename TestFixture::template ListOf<BoxHolder>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    std::unique_ptr<int> box = std::make_unique<int>(7);
    EXPECT_FALSE(list.emplace_after(h, List::end(), std::move(box)));
    ASSERT_NE(box, nullptr);
    EXPECT_EQ(*box, 7);
    EXPECT_TRUE(list.emplace_after(h, list.before_begin(h), std::move(box)));
    EXPECT_EQ(box, nullptr);
    EXPECT_EQ(*list.begin(h)->box, 7);
    auto dead = list.begin(h);
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));
    box = std::make_unique<int>(8);
    EXPECT_FALSE(list.emplace_after(h, dead, std::move(box)));
    EXPECT_EQ(box, nullptr);
} // ContractInsertErase.EmplaceAfterForwardsArgumentsOnlyWhenInitializingANode

// Contract: erase_after() "Returns true iff THIS call logically deleted a
// node; false if `anchor` is `end()` or logically deleted, or has no live
// successor". Every false case on a quiet list, each followed by a check
// that nothing was erased; then `while (erase_after(h, before_begin(h)))`
// empties the list.
TYPED_TEST(ContractInsertErase, EraseAfterReturnsTrueIffThisCallDeletedANode) {
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    EXPECT_FALSE(list.erase_after(h, list.before_begin(h)));  // empty list: no successor
    fill(list, h, {0, 1, 2});
    EXPECT_FALSE(list.erase_after(h, List::end()));
    auto last = list.begin(h);
    ++last;
    ++last;
    EXPECT_FALSE(list.erase_after(h, last));  // no successor
    auto first = list.begin(h);
    EXPECT_TRUE(list.erase_after(h, list.before_begin(h)));  // 0
    EXPECT_FALSE(list.erase_after(h, first));  // dead anchor, though its successor 1 is live
    size_t drained = 0;
    while (list.erase_after(h, list.before_begin(h))) ++drained;
    EXPECT_EQ(drained, 2u);  // 1 and 2: the false calls erased nothing
    EXPECT_EQ(this->alive(), 3) << "erase_after() destroys no value";
} // ContractInsertErase.EraseAfterReturnsTrueIffThisCallDeletedANode

// ===========================================================================
// ContractDeathTest: the five diagnosed precondition violations
// ===========================================================================
//
// Contract (DIAGNOSED PRECONDITION VIOLATIONS): "Debug builds (NDEBUG not
// defined) diagnose exactly these five by assertion": (1) a handle of
// another list passed to a list operation; (2) an iterator used with the
// wrong handle in a list operation (a refreshed or move-assigned-over handle
// starts a new session, a moved handle keeps it); (3) `*` or `->` on
// before_begin()'s iterator; (4) refresh() or a list operation on an empty
// handle; (5) destruction of the list with live handles.
//
// "By assertion" is checked as death by SIGABRT (what a failed assert()
// raises): stricter than "any death", so a crash (SIGSEGV) or a sanitizer
// report (an exit status) does not pass. The message is not matched: its
// text is not part of the contract. Each test pairs its violations with the
// legal use of the same objects (in the parent process), so a death cannot
// come from the objects being unusable. Every prefix is single-threaded and
// deterministic; the "threadsafe" style re-executes the binary for each
// death statement (TSan makes every process multi-threaded, and the fast
// style warns then). Typed over a subset: the diagnoses depend on neither
// the ring size nor the hooks, and each death statement costs a process; K =
// 2 without hooks and K = 64 (the default) with them cover both axes.
#ifndef NDEBUG

#define EXPECT_DIAGNOSED(statement) EXPECT_EXIT(statement, ::testing::KilledBySignal(SIGABRT), "")

template <typename Config>
class ContractDeathTest : public ContractTestBase<Config> {};
using ContractDeathConfigs = ::testing::Types<RcuListConfig<2, NoHooks>, RcuListConfig<64, TestHooks>>;
TYPED_TEST_SUITE(ContractDeathTest, ContractDeathConfigs, RcuListConfigNames);

// Violation (1): every list operation that takes a handle, given a live
// handle of ANOTHER list (with an anchor obtained under that handle, so the
// handle is the only fault).
TYPED_TEST(ContractDeathTest, HandleOfAnotherListIsDiagnosed) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List a;
    List b;
    const List& a_view = a;
    Handle hb = b.new_handle();
    EXPECT_DIAGNOSED((void)a.before_begin(hb));
    EXPECT_DIAGNOSED((void)a.begin(hb));
    EXPECT_DIAGNOSED((void)a_view.begin(hb));
    EXPECT_DIAGNOSED((void)a.insert_after(hb, b.before_begin(hb), 1));
    EXPECT_DIAGNOSED((void)a.emplace_after(hb, b.before_begin(hb), 1));
    EXPECT_DIAGNOSED((void)a.erase_after(hb, b.before_begin(hb)));
    // The handle checks (empty, another list) run before the anchor is looked
    // at, so they fire even when the anchor is end().
    EXPECT_DIAGNOSED((void)a.insert_after(hb, List::end(), 1));
    EXPECT_DIAGNOSED((void)a.emplace_after(hb, List::end(), 1));
    EXPECT_DIAGNOSED((void)a.erase_after(hb, List::end()));
    EXPECT_TRUE(b.insert_after(hb, b.before_begin(hb), 1));  // legal on its own list
    EXPECT_FALSE(b.insert_after(hb, List::end(), 1));
} // ContractDeathTest.HandleOfAnotherListIsDiagnosed

// Violation (2): an iterator obtained under h1 used with another live handle
// h2 of the same list -- as iterator and as converted const_iterator ("bound
// to the same handle session as the original") -- and an iterator of another
// list used with this list's handle.
TYPED_TEST(ContractDeathTest, IteratorUsedWithTheWrongHandleIsDiagnosed) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    List other;
    Handle h1 = list.new_handle();
    Handle h2 = list.new_handle();
    Handle ho = other.new_handle();
    fill(list, h1, {0});
    auto it1 = list.begin(h1);
    const typename List::const_iterator cit1 = it1;
    EXPECT_DIAGNOSED((void)list.insert_after(h2, it1, 1));
    EXPECT_DIAGNOSED((void)list.emplace_after(h2, it1, 1));
    EXPECT_DIAGNOSED((void)list.erase_after(h2, list.before_begin(h1)));
    EXPECT_DIAGNOSED((void)list.insert_after(h2, cit1, 1));
    EXPECT_DIAGNOSED((void)list.insert_after(h1, other.before_begin(ho), 1));
    // The iterator-session check runs only for a non-end() anchor (an end()
    // iterator is bound to no session): end() with either live handle is legal.
    EXPECT_FALSE(list.insert_after(h2, List::end(), 1));
    EXPECT_FALSE(list.erase_after(h2, List::end()));
    EXPECT_TRUE(list.insert_after(h1, cit1, 1));  // legal with its own handle
    EXPECT_TRUE(list.erase_after(h1, list.before_begin(h1)));
} // ContractDeathTest.IteratorUsedWithTheWrongHandleIsDiagnosed

// Violation (2) through refresh(): "EVERY iterator obtained under this handle
// before the call is invalid, whatever the result" and "a refreshed ...
// handle starts a new session". refresh() here returns false (nothing
// advanced), and the old iterator is still diagnosed.
TYPED_TEST(ContractDeathTest, RefreshStartsANewSessionEvenWhenItReturnsFalse) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    fill(list, h, {0});
    auto old_it = list.before_begin(h);
    EXPECT_FALSE(h.refresh());
    EXPECT_DIAGNOSED((void)list.insert_after(h, old_it, 1));
    EXPECT_DIAGNOSED((void)list.erase_after(h, old_it));
    EXPECT_TRUE(list.insert_after(h, list.before_begin(h), 1));  // a new iterator works
} // ContractDeathTest.RefreshStartsANewSessionEvenWhenItReturnsFalse

// Violation (2) through move assignment: "a ... move-assigned-over handle
// starts a new session"; its old iterators are diagnosed, while an iterator
// of the moved-in session is legal under it.
TYPED_TEST(ContractDeathTest, MoveAssignedOverHandleStartsANewSession) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h = list.new_handle();
    fill(list, h, {0});
    auto old_it = list.before_begin(h);
    Handle source = list.new_handle();
    auto moved_in = list.before_begin(source);
    h = std::move(source);
    EXPECT_DIAGNOSED((void)list.erase_after(h, old_it));
    EXPECT_TRUE(list.insert_after(h, moved_in, 1));
} // ContractDeathTest.MoveAssignedOverHandleStartsANewSession

// Violation (2) through swap: the iterators each session validates move with
// it, so after swap(h1, h2) an iterator obtained under h1 is wrong for h1
// and right for h2.
TYPED_TEST(ContractDeathTest, SwapMovesTheIteratorBindingWithTheSession) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    Handle h1 = list.new_handle();
    Handle h2 = list.new_handle();
    auto it1 = list.before_begin(h1);
    swap(h1, h2);
    EXPECT_DIAGNOSED((void)list.insert_after(h1, it1, 1));
    EXPECT_TRUE(list.insert_after(h2, it1, 1));
} // ContractDeathTest.SwapMovesTheIteratorBindingWithTheSession

// Violation (3): `*` and `->` on before_begin()'s iterator, mutable and const
// overloads (the head holds no value); `++` on it is legal.
TYPED_TEST(ContractDeathTest, DereferencingBeforeBeginIsDiagnosed) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    const List& view = list;
    Handle h = list.new_handle();
    fill(list, h, {0});
    EXPECT_DIAGNOSED((void)*list.before_begin(h));
    EXPECT_DIAGNOSED((void)list.before_begin(h)->v);
    EXPECT_DIAGNOSED((void)*view.before_begin(h));
    EXPECT_DIAGNOSED((void)view.before_begin(h)->v);
    auto it = list.before_begin(h);
    ++it;
    EXPECT_EQ(it->v, 0);
} // ContractDeathTest.DereferencingBeforeBeginIsDiagnosed

// Violation (4): refresh() and every list operation on an empty handle --
// default-constructed or moved-from. The insert/erase calls take an end()
// anchor, so the empty handle is their only fault: "the handle checks
// (empty, another list) run before the anchor is looked at, so they fire
// even when the anchor is `end()`".
TYPED_TEST(ContractDeathTest, EmptyHandleInRefreshOrAListOperationIsDiagnosed) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    List list;
    const List& view = list;
    Handle h = list.new_handle();
    Handle e;
    Handle source = list.new_handle();
    Handle target(std::move(source));
    EXPECT_DIAGNOSED(e.refresh());
    EXPECT_DIAGNOSED(source.refresh());
    EXPECT_DIAGNOSED((void)list.before_begin(e));
    EXPECT_DIAGNOSED((void)list.begin(e));
    EXPECT_DIAGNOSED((void)view.begin(e));
    EXPECT_DIAGNOSED((void)list.begin(source));
    EXPECT_DIAGNOSED((void)list.insert_after(e, List::end(), 1));
    EXPECT_DIAGNOSED((void)list.emplace_after(e, List::end(), 1));
    EXPECT_DIAGNOSED((void)list.erase_after(e, List::end()));
    EXPECT_DIAGNOSED((void)list.erase_after(source, List::end()));
    EXPECT_FALSE(target.refresh());  // the moved session is live and current
    EXPECT_FALSE(list.insert_after(h, List::end(), 1));  // the same calls with a live handle are legal
} // ContractDeathTest.EmptyHandleInRefreshOrAListOperationIsDiagnosed

// Violation (5): the list destroyed while a handle is alive -- one from
// new_handle() on the list, one from its const view, one moved elsewhere.
// The list is destroyed through std::optional::reset() inside the death
// statement (the list is neither copyable nor movable).
TYPED_TEST(ContractDeathTest, DestroyingTheListWithALiveHandleIsDiagnosed) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    using List = typename TestFixture::template ListOf<Tracked>;
    using Handle = typename List::handle;
    EXPECT_DIAGNOSED({
        std::optional<List> list;
        list.emplace();
        Handle h = list->new_handle();
        list.reset();
    }); // h is still alive at the reset
    EXPECT_DIAGNOSED({
        std::optional<List> list;
        list.emplace();
        Handle h = std::as_const(*list).new_handle();
        list.reset();
    }); // h, from the const view, is still alive at the reset
    EXPECT_DIAGNOSED({
        Handle outer;
        std::optional<List> list;
        list.emplace();
        Handle h = list->new_handle();
        outer = std::move(h);
        list.reset();
    }); // the handle moved to `outer` is still alive at the reset
    // The legal order: the handles end first.
    std::optional<List> list;
    list.emplace();
    {
        Handle h = list->new_handle();
    } // h ends
    list.reset();
} // ContractDeathTest.DestroyingTheListWithALiveHandleIsDiagnosed

#undef EXPECT_DIAGNOSED

#endif // NDEBUG
