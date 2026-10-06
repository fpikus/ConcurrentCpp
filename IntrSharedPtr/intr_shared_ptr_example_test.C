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
// An example of building your own concurrent data structure on IntrSharedPtr,
// written the way README.md ("Building your own data structure") tells a user
// to: a Treiber stack (push and pop by CAS on the head word), built once on each
// pointer, with the two pointee routes the README describes, and GoogleTest
// tests of the stack and of the lifetime promises the README makes.
//
// Contents
//   1. Node census       -- per-value destruction counters: the oracle for
//                           "every node destroyed exactly once"
//   2. The two nodes     -- SpinNode: intr_shared_ptr, the base class route
//                           (intr_pointee_base<>); HpNode: intr_shared_ptr_hp,
//                           the hand-written hooks route
//   3. TreiberStack<Node> -- the structure, one template for both nodes
//   4. Fixture           -- census reset, and the drain that deferred
//                           reclamation needs before "destroyed" is asserted
//   5. Tests, both nodes -- empty pop, LIFO order, every node destroyed exactly
//                           once (popped, and left in the stack), concurrent
//                           push/pop stress
//   6. Tests, spinlock   -- reclamation is synchronous
//   7. Tests, HP         -- reclamation waits for a scan; a linked chain dies
//                           one link per drain (why pop() unlinks)
//
// Build (README, "Include and link"): -I<repository>/IntrSharedPtr; because one
// of the nodes uses intr_shared_ptr_hp, also compile mm_hp/mm_hp.cpp on the same
// command line, with the same flags, and link GoogleTest.
//
// Not used here: the Harris mark bit. A stack never deletes from the middle, so
// there is nothing to mark; the README presents marking for lists
// ("Using the atomic"), and ../LockFreeList/lock_free_list.h shows it.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <latch>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "intr_shared_ptr.h"     // intr_shared_ptr, intr_pointee_base
#include "intr_shared_ptr_hp.h"  // intr_shared_ptr_hp; std::hazard_pointer_obj_base (mm_hp)
#include "hp_drain.h"            // drain_reclamation(): README, "Living with deferred reclamation"
#include "hp_drain_gtest.h"      // the early scan before the first test (same section)

namespace {

// ---------------------------------------------------------------------------
// 1. Node census
// ---------------------------------------------------------------------------
//
// Every node carries a distinct int value, and its destructor bumps the counter
// of that value. "Destroyed exactly once" is then a per-value check (a total
// alone would let a double destruction cancel a leak). Relaxed: the counters
// are read only after the threads that wrote them are joined, or on the thread
// that wrote them; they must not add synchronization the stack itself lacks.
// Under intr_shared_ptr_hp the destructor runs on whichever thread runs a scan
// (README, "Living with deferred reclamation": T must not be thread-affine), so
// the counters are global, never thread-local.
//
// Test independence: each test opens a new census generation, and a node
// counts only in the generation it was created in. A node an earlier test left
// pending (under the HP pointer a failed test can leave a whole destructor
// cascade behind, which one drain does not flush) may be destroyed during a
// later test, by any scan; it then leaves that test's census alone.

constexpr int kMaxValue = 1 << 16;                  // every value a test uses is below this
std::array<std::atomic<int>, kMaxValue> g_destroyed{}; // destructions per value, this generation
std::atomic<long> g_live{0};                        // constructed minus destroyed, this generation
std::atomic<unsigned> g_generation{0};              // the current test's census generation

// The payload every node carries: the value, and the census bookkeeping in its
// constructor and destructor.
struct Payload {
    // `value`: the stack element, and the node's census id (0 <= value < kMaxValue).
    explicit Payload(int value) noexcept
        : value(value), generation_(g_generation.load(std::memory_order_relaxed)) {
        g_live.fetch_add(1, std::memory_order_relaxed);
    }
    ~Payload() {
        // A node left over from an earlier test does not count against this one.
        if (generation_ != g_generation.load(std::memory_order_relaxed)) return;
        g_destroyed[value].fetch_add(1, std::memory_order_relaxed);
        g_live.fetch_sub(1, std::memory_order_relaxed);
    }
    Payload(const Payload&) = delete;
    Payload& operator=(const Payload&) = delete;

    const int value;    // written once, before the node is published

private:
    const unsigned generation_;     // the census generation this node counts in
}; // struct Payload

// Opens a new census generation with every counter at zero. Call with no test
// thread running (a fixture's SetUp()): the threads a test starts later see the
// new generation through their creation.
void reset_census() {
    g_generation.fetch_add(1, std::memory_order_relaxed);
    for (std::atomic<int>& count : g_destroyed) count.store(0, std::memory_order_relaxed);
    g_live.store(0, std::memory_order_relaxed);
} // reset_census()

// Success iff every value in [0, n) was destroyed exactly once and no node is
// alive; otherwise names the first few offenders.
::testing::AssertionResult each_destroyed_once(int n) {
    std::string bad;
    int bad_count = 0;
    for (int value = 0; value < n; ++value) {
        const int count = g_destroyed[value].load(std::memory_order_relaxed);
        if (count == 1) continue;
        if (++bad_count <= 8) {
            bad += " value " + std::to_string(value) + " x" + std::to_string(count);
        }
    } // loop over the values the test created
    const long live = g_live.load(std::memory_order_relaxed);
    if (bad_count == 0 && live == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << bad_count << " value(s) not destroyed exactly once:"
                                         << bad << "; live nodes: " << live;
} // each_destroyed_once()

// ---------------------------------------------------------------------------
// 2. The two nodes
// ---------------------------------------------------------------------------
//
// A node is the pointee: it carries the reference count, and it holds the link
// to the next node as an atomic pointer to its own type. README, "The
// pointee": a node may hold a pointer to its own type while still incomplete;
// the pointers check their requirements at the first use of a member.

// Route 1, the base class (README, "The pointee: two routes"; the default
// route): intr_pointee_base<> supplies AddRef, DelRef, use_count (and
// TryAddRef, unused by this pointer) with the orders intr_pointee.h requires.
// The count is a long, so the owner bound (README, "The pointee") is academic.
// Alignment: intr_shared_ptr needs alignof(Node) >= 4 (README, "The domain of
// applicability"); the base's atomic long gives 8.
struct SpinNode : intr_pointee_base<>, Payload {
    explicit SpinNode(int value) noexcept : Payload(value) {}
    intr_shared_ptr<SpinNode> next;     // the link; null for the bottom node
}; // struct SpinNode

// Route 2, hand-written hooks (README, "The pointee: two routes"): for
// intr_shared_ptr_hp the node derives publicly from
// std::hazard_pointer_obj_base<HpNode> (defined by mm_hp, reached through
// intr_shared_ptr_hp.h; never combine with a real <hazard_pointer>) and writes
// the four hooks with exactly the semantics and orders of intr_pointee.h ("The
// hooks (normative)"). This node could have derived from
// intr_pointee_base_hp<HpNode> instead; it writes the hooks to show the route.
// The count starts at 0 (a new node is owned by nobody) and is changed only by
// read-modify-writes -- a plain store, even to initialize it, is forbidden.
// U (here HpNode) is the dynamic type: mm_hp deletes an HpNode*.
struct HpNode : std::hazard_pointer_obj_base<HpNode>, Payload {
    explicit HpNode(int value) noexcept : Payload(value) {}

    // Increment; relaxed suffices: the caller already owns a reference (or
    // the node is fresh), so nothing is published through it.
    void AddRef() noexcept { count_.fetch_add(1, std::memory_order_relaxed); }

    // Decrement; true iff this call made 1 -> 0 (the caller then retires the
    // node). acq_rel, both halves load-bearing: release heads the sequence
    // TryAddRef's acquire reads synchronize with; acquire gives the retiring
    // thread the history of every earlier releaser.
    bool DelRef() noexcept { return count_.fetch_sub(1, std::memory_order_acq_rel) == 1; }

    // A relaxed snapshot, for diagnostics; the stack never decides on it.
    long use_count() const noexcept { return count_.load(std::memory_order_relaxed); }

    // Increment iff nonzero; false iff it observed 0 (the node is retired or
    // about to be, and must never be revived). Copied from the reference form,
    // intr_pointee_base::TryAddRef, as the README says to: acquire initial
    // load, CAS acquire on success and relaxed on failure, and a 0 refreshed
    // by a failed CAS re-read with acquire, so that every observed 0 is an
    // acquire read.
    bool TryAddRef() noexcept {
        long count = count_.load(std::memory_order_acquire);
        while (count != 0) {
            if (count_.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return true;
            }
            if (count == 0) count = count_.load(std::memory_order_acquire);
        } // CAS loop while the count is nonzero
        return false;
    } // HpNode::TryAddRef()

    intr_shared_ptr_hp<HpNode> next;    // the link; null for the bottom node

private:
    std::atomic<long> count_{0};        // the strong count: owners of this node
}; // struct HpNode

// ---------------------------------------------------------------------------
// 3. TreiberStack<Node>
// ---------------------------------------------------------------------------
//
// A lock-free (on intr_shared_ptr_hp; on intr_shared_ptr, lock-free in the
// algorithmic sense over a pointer that takes a short lock: README,
// "Progress") stack of ints. The head is an atomic pointer to the top node;
// each node's `next` points to the node below. The stack works with whichever
// pointer the node links with: the atomic type is taken from Node::next.
//
// ABA: the classic Treiber hazard -- the top is popped, freed, and a new node
// reuses its address before a stalled popper's CAS -- cannot happen, because
// `top` in pop() is an owning handle: the node it names cannot be destroyed
// (or, under the HP pointer, even retired) while the handle lives, so its
// address cannot be reused.
template <typename Node>
class TreiberStack {
    using atomic_ptr = decltype(Node::next);                    // intr_shared_ptr<Node> or _hp
    using node_ptr = typename atomic_ptr::shared_ptr_type;      // the owning handle

public:
    TreiberStack() = default;
    TreiberStack(const TreiberStack&) = delete;
    TreiberStack& operator=(const TreiberStack&) = delete;

    // Pops every remaining node, one at a time, instead of letting head_'s
    // destructor release the top node: that release would destroy the chain
    // through the nodes' own `next` members -- recursively, as deep as the
    // stack, under intr_shared_ptr, and one link per reclamation scan under
    // intr_shared_ptr_hp (README, "The domain of applicability"). pop()
    // unlinks each node it removes, so neither happens. Precondition: no
    // other thread uses the stack (as for any object being destroyed).
    ~TreiberStack() {
        while (pop()) {}
    }

    // Push `value`. Allocates a node (may throw std::bad_alloc; so may the
    // HP pointer's first load() on a thread -- README, "Living with deferred
    // reclamation" -- with nothing changed either way).
    void push(int value) {
        // Adopting the raw pointer takes the fresh node's count from 0 to 1
        // (README, "The value type"): `node` owns it.
        node_ptr node(new Node(value));
        // acquire: on success the new node points at `expected`, and a popper
        // that acquires the new node through head_ must also see `expected`'s
        // construction; this load (or the CAS refresh below) is the link in
        // that chain of synchronization.
        node_ptr expected = head_.load(std::memory_order_acquire);
        do {
            // Link the node to the current top before publishing it. Nobody
            // else can see `node` yet, so relaxed would do on both pointers
            // (each accepts it and promotes it: see the headers' ordering
            // contracts); release states the intent.
            node->next.store(expected, std::memory_order_release);
            // Publish: swing head_ from `expected` to `node`. release: a
            // popper that acquires `node` sees its value and its link. On
            // failure the strong CAS refreshes `expected` with the current top
            // and a live reference, and never fails with `expected` unchanged
            // (README, "The atomic"), so every retry relinks to a newer top.
            // acquire on failure for the same reason as the load above.
        } while (!head_.compare_exchange_strong(expected, node,
                                                std::memory_order_release,
                                                std::memory_order_acquire));
    } // TreiberStack::push()

    // Pop the top value; nullopt iff the stack was empty when observed.
    std::optional<int> pop() {
        node_ptr top = head_.load(std::memory_order_acquire);
        while (top) {
            // `top` owns a reference, so the node and its `next` stay alive
            // here even if another thread pops it meanwhile.
            node_ptr below = top->next.load(std::memory_order_acquire);
            // Swing head_ past `top`. release: the next popper that acquires
            // `below` through head_ synchronizes with this CAS, which happens
            // after this thread acquired `below` from top->next. On failure
            // `top` is refreshed to the current top (live reference), and the
            // loop retries -- or ends if the stack became empty.
            if (head_.compare_exchange_strong(top, below, std::memory_order_release,
                                              std::memory_order_acquire)) break;
        } // CAS loop until a node is removed or the stack is empty
        if (!top) return std::nullopt;
        const int value = top->value;
        // Unlink the removed node: drop its reference to the node below, so
        // that the removed node's destruction releases nothing. Without this,
        // a popped node keeps the one below alive, a chain of popped nodes
        // releases each other from their destructors, and under
        // intr_shared_ptr_hp such a chain dies one link per scan (README,
        // "Living with deferred reclamation"; test 7b shows it). Safe while
        // other poppers still hold `top` as a stale expected value: they may
        // read null from top->next, but their CAS on head_ then fails, because
        // head_ no longer names `top` and never will again (nodes are never
        // pushed twice, and `top`'s reference keeps the address from reuse).
        top->next.store(node_ptr(), std::memory_order_release);
        return value;
    } // TreiberStack::pop()

private:
    atomic_ptr head_;   // the top node, null when empty; the stack's one shared word
}; // class TreiberStack

// ---------------------------------------------------------------------------
// 4. Fixture
// ---------------------------------------------------------------------------

// True iff the pointer destroys a released object later rather than at its
// last release: the optional member `deferred_reclamation` (README, "The
// atomic"; ../SharedPtr/atomic_shared_ptr_concept.h, "Optional members").
// intr_shared_ptr does not declare it, so it is detected with a requires
// expression, the way the concept header prescribes (a plain
// `P::deferred_reclamation` would not compile for intr_shared_ptr).
template <typename AtomicPtr>
constexpr bool reclamation_is_deferred = requires { requires AtomicPtr::deferred_reclamation; };

template <typename Node>
class StackTest : public ::testing::Test {
protected:
    static constexpr bool kDeferred = reclamation_is_deferred<decltype(Node::next)>;

    // Starts every test from an empty census (a new generation, see the
    // census). Under the HP pointer it also drains first, so that the test
    // starts with mm_hp's pending list empty -- or, after a failed earlier test,
    // holding at most the next layer of the cascade it left: far below the
    // scan threshold either way, which test 7a relies on (hp_drain.h,
    // "CONTRACT"). The process is quiescent here (no test thread exists), as
    // the drain requires.
    void SetUp() override {
        if constexpr (kDeferred) drain_reclamation();
        reset_census();
    }

    // Makes every released node actually destroyed, so that the census can be
    // checked: nothing to do for intr_shared_ptr (synchronous reclamation,
    // README "Reclamation"); one drain for intr_shared_ptr_hp (README, "Living
    // with deferred reclamation": a test that asserts "destroyed" must drain
    // first). One drain is enough because pop() unlinks every node it removes:
    // no node's destructor releases another node, so the destructor cascade has
    // depth 1. Precondition: quiescence -- call only after joining every
    // thread that used the stack.
    static void reclaim_released() {
        if constexpr (kDeferred) drain_reclamation();
    }
}; // class StackTest

// Readable typed-test names: the pointer, then the pointee route.
struct NodeName {
    template <typename Node>
    static std::string GetName(int) {
        if constexpr (std::is_same_v<Node, SpinNode>) {
            return "Spinlock_BaseClass";
        } else {
            return "HazardPointer_HandWrittenHooks";
        }
    }
}; // struct NodeName

using NodeTypes = ::testing::Types<SpinNode, HpNode>;
TYPED_TEST_SUITE(StackTest, NodeTypes, NodeName);

// ---------------------------------------------------------------------------
// 5. Tests, both nodes
// ---------------------------------------------------------------------------

// A fresh stack is empty: pop() reports nothing, and keeps doing so.
TYPED_TEST(StackTest, PopOnEmptyStackReturnsNothing) {
    TreiberStack<TypeParam> stack;
    EXPECT_EQ(stack.pop(), std::nullopt);
    EXPECT_EQ(stack.pop(), std::nullopt);
}

// LIFO: values come out in the reverse order of their pushes, then the stack
// is empty again.
TYPED_TEST(StackTest, PopsInLifoOrder) {
    constexpr int kCount = 16;
    TreiberStack<TypeParam> stack;
    for (int value = 0; value < kCount; ++value) stack.push(value);
    for (int value = kCount - 1; value >= 0; --value) EXPECT_EQ(stack.pop(), value);
    EXPECT_EQ(stack.pop(), std::nullopt);
}

// LIFO holds across interleaved pushes and pops, through empty and back.
TYPED_TEST(StackTest, InterleavedPushPopKeepsLifoOrder) {
    TreiberStack<TypeParam> stack;
    stack.push(1);
    stack.push(2);
    EXPECT_EQ(stack.pop(), 2);
    stack.push(3);
    EXPECT_EQ(stack.pop(), 3);
    EXPECT_EQ(stack.pop(), 1);
    EXPECT_EQ(stack.pop(), std::nullopt);
    stack.push(4);
    EXPECT_EQ(stack.pop(), 4);
} // InterleavedPushPopKeepsLifoOrder

// Every popped node is destroyed exactly once, once the pointer's reclamation
// has run (immediately for the spinlock pointer, after a drain for the HP one).
TYPED_TEST(StackTest, EveryPoppedNodeDestroyedExactlyOnce) {
    constexpr int kCount = 100;
    {
        TreiberStack<TypeParam> stack;
        for (int value = 0; value < kCount; ++value) stack.push(value);
        for (int value = 0; value < kCount; ++value) ASSERT_TRUE(stack.pop().has_value());
    } // the (empty) stack is gone
    this->reclaim_released();
    EXPECT_TRUE(each_destroyed_once(kCount));
} // EveryPoppedNodeDestroyedExactlyOnce

// Nodes still in the stack when it is destroyed are destroyed exactly once
// too (the destructor pops them).
TYPED_TEST(StackTest, NodesLeftInStackDestroyedWithIt) {
    constexpr int kCount = 100;
    {
        TreiberStack<TypeParam> stack;
        for (int value = 0; value < kCount; ++value) stack.push(value);
        ASSERT_EQ(stack.pop(), kCount - 1);
    } // the stack is destroyed with kCount - 1 nodes in it
    this->reclaim_released();
    EXPECT_TRUE(each_destroyed_once(kCount));
} // NodesLeftInStackDestroyedWithIt

// Concurrent stress: several threads push distinct values and pop, all at
// once. Oracles: (1) the values popped -- by the threads, then the rest by the
// main thread -- are exactly the values pushed, each once (no value lost,
// duplicated or invented); (2) once the stack and every handle are gone and,
// for the HP pointer, after the drain, every node was destroyed exactly once.
// Sized to finish in seconds under ThreadSanitizer.
TYPED_TEST(StackTest, ConcurrentPushPopPreservesValuesAndNodes) {
    constexpr int kThreads = 4;
    constexpr int kPerThread = 4000;
    constexpr int kTotal = kThreads*kPerThread;
    static_assert(kTotal <= kMaxValue);
    std::vector<std::vector<int>> popped(kThreads);     // per thread: what it popped
    std::vector<int> all;                               // everything popped, in the end
    {
        TreiberStack<TypeParam> stack;
        std::latch start(kThreads);                     // release the threads together
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&stack, &start, &out = popped[t], t] {
                start.arrive_and_wait();
                // Push this thread's values; pop after two of every three
                // pushes, so the stack grows slowly and pops contend with
                // pushes and with each other, including on a near-empty stack.
                for (int i = 0; i < kPerThread; ++i) {
                    stack.push(t*kPerThread + i);
                    if (i % 3 == 0) continue;
                    const std::optional<int> value = stack.pop();
                    if (value) out.push_back(*value);
                } // loop over this thread's values
            }); // thread body
        } // loop starting the threads
        for (std::thread& thread : threads) thread.join();
        for (const std::vector<int>& out : popped) all.insert(all.end(), out.begin(), out.end());
        while (const std::optional<int> value = stack.pop()) all.push_back(*value);
    } // the stack is gone; every handle was local to it or to a joined thread
    std::sort(all.begin(), all.end());
    std::vector<int> pushed(kTotal);
    std::iota(pushed.begin(), pushed.end(), 0);
    EXPECT_EQ(all, pushed);
    this->reclaim_released();      // quiescent: every thread is joined
    EXPECT_TRUE(each_destroyed_once(kTotal));
} // ConcurrentPushPopPreservesValuesAndNodes

// ---------------------------------------------------------------------------
// 6. Tests, spinlock pointer: synchronous reclamation
// ---------------------------------------------------------------------------
//
// README, "Reclamation": intr_shared_ptr deletes an object at its last
// release, synchronously, on the releasing thread. No drain anywhere below.

class SpinStackTest : public StackTest<SpinNode> {};

// The popped node's last owner is pop()'s local handle: the node is gone by
// the time pop() returns.
TEST_F(SpinStackTest, PoppedNodeDiesAtLastRelease) {
    TreiberStack<SpinNode> stack;
    stack.push(0);
    stack.push(1);
    ASSERT_EQ(stack.pop(), 1);
    EXPECT_EQ(g_destroyed[1].load(std::memory_order_relaxed), 1);
    EXPECT_EQ(g_destroyed[0].load(std::memory_order_relaxed), 0);
    EXPECT_EQ(g_live.load(std::memory_order_relaxed), 1);
} // PoppedNodeDiesAtLastRelease

// The stack's destructor destroys every remaining node before it returns.
TEST_F(SpinStackTest, StackDestructorDestroysNodesSynchronously) {
    constexpr int kCount = 50;
    {
        TreiberStack<SpinNode> stack;
        for (int value = 0; value < kCount; ++value) stack.push(value);
        EXPECT_EQ(g_live.load(std::memory_order_relaxed), kCount);
    } // the stack is destroyed here
    EXPECT_TRUE(each_destroyed_once(kCount));
} // StackDestructorDestroysNodesSynchronously

// ---------------------------------------------------------------------------
// 7. Tests, HP pointer: deferred reclamation
// ---------------------------------------------------------------------------
//
// README, "Living with deferred reclamation": "the count reached zero" and
// "the object is gone" are different moments; the object is destroyed at a
// scan, which runs inside a retire() that brings the pending retirements to
// the threshold (at least 1000), and a drain makes that scan happen. The
// fixture drained in SetUp(), so the pending list starts empty (hp_drain.h,
// "CONTRACT": after a quiescent drain nothing retired before it is pending,
// barring hazard-protected objects, and none is protected here).

class HpStackTest : public StackTest<HpNode> {};

// 7a. A popped node is retired at pop()'s last release but not destroyed: one
// retirement is far below the threshold, so no scan runs. The drain destroys
// it. (The census counter is read, never the retired node: README, nobody may
// read a retired object.)
TEST_F(HpStackTest, PoppedNodeWaitsForScan) {
    TreiberStack<HpNode> stack;
    stack.push(0);
    stack.push(1);
    ASSERT_EQ(stack.pop(), 1);
    EXPECT_EQ(g_destroyed[1].load(std::memory_order_relaxed), 0);  // retired, pending
    EXPECT_EQ(g_live.load(std::memory_order_relaxed), 2);
    drain_reclamation();
    EXPECT_EQ(g_destroyed[1].load(std::memory_order_relaxed), 1);  // gone at the scan
    EXPECT_EQ(g_destroyed[0].load(std::memory_order_relaxed), 0);  // still in the stack
} // PoppedNodeWaitsForScan

// 7b. Why pop() unlinks the node it removes. Three nodes linked through their
// own `next` (as a stack's nodes are, if pop() did not unlink them) and
// released as one chain die one link per scan: the scan that destroys the
// first node runs its destructor, whose `next` releases the second node --
// retired inside the scan, so reclaimed only at the next one (README, "Living
// with deferred reclamation"; hp_drain.h, "one drain reclaims one layer of a
// destructor cascade").
TEST_F(HpStackTest, LinkedChainDiesOneLinkPerDrain) {
    using node_ptr = intr_shared_ptr_hp<HpNode>::shared_ptr_type;
    {
        node_ptr first(new HpNode(0));
        node_ptr second(new HpNode(1));
        node_ptr third(new HpNode(2));
        first->next.store(second);
        second->next.store(third);
    } // all three handles die: first retired (count 0); second and third still owned by links
    EXPECT_EQ(g_live.load(std::memory_order_relaxed), 3);
    for (int link = 0; link < 3; ++link) {
        drain_reclamation();
        for (int value = 0; value < 3; ++value) {
            EXPECT_EQ(g_destroyed[value].load(std::memory_order_relaxed), value <= link ? 1 : 0)
                << "after drain " << link + 1 << ", node " << value;
        } // loop over the chain's nodes
    } // loop over drains, one link each
    EXPECT_TRUE(each_destroyed_once(3));
} // LinkedChainDiesOneLinkPerDrain

} // namespace
