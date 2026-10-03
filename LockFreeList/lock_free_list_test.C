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
#include <gtest/gtest.h>
#include <cstdio>
#include <thread>
#include <vector>
#include <random>

#include "atomic_shared_ptr_concept.h"
#include "hp_drain.h"
#include "hp_drain_gtest.h"
#include "intr_shared_ptr.h"
#include "intr_shared_ptr_hp.h"
#include "lock_free_shared_ptr/atomic_shared_ptr.hpp"
#include "lock_free_list.h"

// Element type that counts live instances, so reclamation tests can confirm every
// node's payload is destroyed (i.e. the whole chain is freed) when the list and all
// iterators are gone.
struct Tracked {
    static std::atomic<int> alive;
    int v;
    Tracked(int x = 0) : v(x) { alive.fetch_add(1, std::memory_order_relaxed); }
    Tracked(const Tracked& o) : v(o.v) { alive.fetch_add(1, std::memory_order_relaxed); }
    Tracked(Tracked&& o) noexcept : v(o.v) { alive.fetch_add(1, std::memory_order_relaxed); }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) = default;
    ~Tracked() { alive.fetch_sub(1, std::memory_order_relaxed); }
};
std::atomic<int> Tracked::alive{0};

// Wrappers for TYPED_TEST_SUITE: gtest typed tests take a list of *types*,
// but LockFreeList is parameterized by a template-template argument, so each
// wrapper smuggles its pointer template through as a member alias
// (Wrapper::template ptr_type) that the fixture unpacks.
struct StdAtomicWrapper {
    template <typename U> using ptr_type = StdAtomicSharedPtrAdapter<U>;
};
struct IntrPtrWrapper {
    template <typename U> using ptr_type = intr_shared_ptr<U, U>;
};
struct ParlayWrapper {
    template <typename U> using ptr_type = parlay::atomic_shared_ptr<U>;
};
struct IntrPtrHPWrapper {
    template <typename U> using ptr_type = intr_shared_ptr_hp<U, U>;
};

// Uniform node factory: the pointer families construct their shared_ptr_type
// differently (std adapter: from make_shared; the two intrusive pointers:
// adopting a raw new, since the count lives in the node; parlay: from
// parlay::make_shared). Tests funnel all node creation through this one
// callable, which dispatches on the concrete shared_ptr_type at compile time.
template <typename Wrapper>
struct Factory {
    template <typename Node>
    typename Wrapper::template ptr_type<Node>::shared_ptr_type operator()(int v = 0) const {
        using SharedPtr = typename Wrapper::template ptr_type<Node>::shared_ptr_type;
        if constexpr (std::is_same_v<SharedPtr, typename StdAtomicSharedPtrAdapter<Node>::shared_ptr_type>) {
            return SharedPtr(std::make_shared<Node>(v));
        } else if constexpr (std::is_same_v<SharedPtr, typename intr_shared_ptr<Node, Node>::shared_ptr_type> ||
                             std::is_same_v<SharedPtr, typename intr_shared_ptr_hp<Node, Node>::shared_ptr_type>) {
            return SharedPtr(new Node(v));
        } else {
            return SharedPtr(parlay::make_shared<Node>(v));
        }
    }
};

// Fixture: brackets every test with mm_hp drains, for every pointer policy (a
// drain is ~1000 filler retirements, milliseconds; the policies without
// deferred reclamation are unaffected, so there is no per-policy branch).
// Under intr_shared_ptr_hp a released node is only retired, and a dropped list
// is a destructor CASCADE: the dummy node's ~Node() walk retires the chain it
// owns, which a second scan destroys (ReclaimsEntireChain pins exactly two
// rounds), and a graveyard chain held elsewhere can add a layer. So:
// - SetUp() drains `setup_drain_rounds` whole rounds and only then captures the
//   Tracked base. Most tests here use int lists, which have no live counter to
//   drain against; fixed rounds also clear their pending layers, so that a
//   test starts with (almost) nothing pending and no scan fires in the middle
//   of a test that counts drains.
// - TearDown() drains until every Tracked node a test created is gone
//   (drain_until() with the live-count predicate, which returns at once when
//   nothing is pending), so a leak fails as that one test.
// GoogleTest runs TearDown() BEFORE it destroys the fixture's data members, so
// a member owning a list would still hold its nodes at the drain. The only
// member here is the stateless factory; lists live in the test bodies, which
// have ended by then. Keep it that way, or reset such a member in TearDown()
// before the drain.
template <typename Wrapper>
class LockFreeListTest : public ::testing::Test {
protected:
    using List = LockFreeList<int, Wrapper::template ptr_type>;
    using Node = typename List::Node;

    // Whole drain rounds in SetUp(): a dropped list needs two, plus one for a
    // graveyard layer; extra rounds cost milliseconds each.
    static constexpr int setup_drain_rounds = 3;
    // Round cap of TearDown()'s drain_until(); above any cascade these tests
    // leave, so reaching it means a leak.
    static constexpr int teardown_drain_rounds = 8;

    // Tracked::alive after SetUp()'s drains.
    int tracked_base_ = 0;

    void SetUp() override {
        for (int round = 0; round < setup_drain_rounds; ++round) drain_reclamation();
        tracked_base_ = Tracked::alive.load();
    }

    void TearDown() override {
        ASSERT_TRUE(drain_until([this] { return Tracked::alive.load() == tracked_base_; }, teardown_drain_rounds))
            << Tracked::alive.load() - tracked_base_ << " Tracked node payloads still alive after "
            << teardown_drain_rounds << " drain rounds";
    }

    Factory<Wrapper> factory;   // stateless node factory
}; // class LockFreeListTest

using PtrWrappers = ::testing::Types<
    StdAtomicWrapper,
    IntrPtrWrapper,
    ParlayWrapper,
    IntrPtrHPWrapper
>;

TYPED_TEST_SUITE(LockFreeListTest, PtrWrappers);

TYPED_TEST(LockFreeListTest, BasicInsertErase) {
    typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());
    
    EXPECT_EQ(list.begin(), list.end());
    
    EXPECT_TRUE(list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(10)));
    EXPECT_TRUE(list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(20)));
    
    auto it = list.begin();
    ASSERT_NE(it, list.end());
    EXPECT_EQ(*it, 20);
    
    ++it;
    ASSERT_NE(it, list.end());
    EXPECT_EQ(*it, 10);
    
    ++it;
    EXPECT_EQ(it, list.end());
    
    // Erase 20
    EXPECT_TRUE(list.erase_after(list.before_begin()));
    
    it = list.begin();
    ASSERT_NE(it, list.end());
    EXPECT_EQ(*it, 10);
    
    // Erase 10
    EXPECT_TRUE(list.erase_after(list.before_begin()));
    EXPECT_EQ(list.begin(), list.end());
}

TYPED_TEST(LockFreeListTest, GraveyardTraversal) {
    typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());
    
    // Insert 1, 2, 3
    list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(3));
    list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(2));
    list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(1));
    
    auto it1 = list.begin(); // points to 1
    
    // Erase 1 and 2
    list.erase_after(list.before_begin()); // erases 1
    list.erase_after(list.before_begin()); // erases 2
    
    // it1 is still on 1, which is logically deleted. 
    // We should be able to traverse to 2 and 3.
    EXPECT_EQ(*it1, 1);
    ++it1;
    ASSERT_NE(it1, list.end());
    EXPECT_EQ(*it1, 2);
    ++it1;
    ASSERT_NE(it1, list.end());
    EXPECT_EQ(*it1, 3);
}

TYPED_TEST(LockFreeListTest, EraseEdgeCases) {
    typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());

    // Nothing after the dummy head, and a null (end) anchor: both must fail cleanly.
    EXPECT_FALSE(list.erase_after(list.before_begin()));
    EXPECT_FALSE(list.erase_after(list.end()));

    list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(5));
    EXPECT_TRUE(list.erase_after(list.before_begin()));
    // Erasing again on the now-empty list must report failure, not double-free.
    EXPECT_FALSE(list.erase_after(list.before_begin()));
    EXPECT_EQ(list.begin(), list.end());
}

TYPED_TEST(LockFreeListTest, OrderAndMiddleInsert) {
    typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());

    // Each insert_after(before_begin) pushes to the front: ends up 5,4,3,2,1.
    for (int i = 1; i <= 5; ++i)
        list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(i));

    std::vector<int> seen;
    for (auto it = list.begin(); it != list.end(); ++it) seen.push_back(*it);
    EXPECT_EQ(seen, (std::vector<int>{5, 4, 3, 2, 1}));

    // Insert 99 after the first node -> 5,99,4,3,2,1.
    EXPECT_TRUE(list.insert_after(list.begin(), this->factory.template operator()<typename TestFixture::Node>(99)));
    seen.clear();
    for (auto it = list.begin(); it != list.end(); ++it) seen.push_back(*it);
    EXPECT_EQ(seen, (std::vector<int>{5, 99, 4, 3, 2, 1}));
}

// Marking-only invariant: inserting after a node that has been logically deleted
// must fail, otherwise the new node would be stranded in a detached chain.
TYPED_TEST(LockFreeListTest, InsertAfterDeletedAnchorFails) {
    using PtrForNode = typename TypeParam::template ptr_type<typename TestFixture::Node>;
    if constexpr (PtrForNode::supports_marking) {
        typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());

        list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(10));
        auto on_ten = list.begin(); // strong ref to node 10

        EXPECT_TRUE(list.erase_after(list.before_begin())); // logically delete node 10
        EXPECT_TRUE(on_ten.curr_); // iterator still pins the deleted node

        // Anchor is logically deleted -> insertion is refused.
        EXPECT_FALSE(list.insert_after(on_ten, this->factory.template operator()<typename TestFixture::Node>(99)));
    }
}

// Node::TryAddRef(), the production hook intr_shared_ptr_hp::load() relies on
// to never revive a node whose count reached 0: from 0 it returns false and
// leaves 0; from n >= 1 it returns true and leaves n + 1. The hooks are the same
// code for every policy's Node (which is why the test runs for all of them),
// and are noexcept (the hazard-pointer pointer requires it). Also prints
// sizeof(Node) for int, which differs only by the policy's pointee base: the
// hazard pointer base adds 24 bytes to the intr_shared_ptr_hp Node and the
// empty base adds nothing to the others.
TYPED_TEST(LockFreeListTest, NodeTryAddRef) {
    using Node = typename TestFixture::Node;
    std::printf("[          ] sizeof(Node) = %zu\n", sizeof(Node));

    Node node(7);
    static_assert(noexcept(node.AddRef()) && noexcept(node.DelRef()) &&
                  noexcept(node.TryAddRef()) && noexcept(node.use_count()));
    EXPECT_EQ(node.use_count(), 0);
    EXPECT_FALSE(node.TryAddRef()) << "TryAddRef() incremented from 0";
    EXPECT_EQ(node.use_count(), 0) << "a failed TryAddRef() changed the count";

    node.AddRef();
    for (long n = 1; n <= 3; ++n) {
        EXPECT_EQ(node.use_count(), n);
        EXPECT_TRUE(node.TryAddRef()) << "TryAddRef() failed at count " << n;
        EXPECT_EQ(node.use_count(), n + 1);
    } // loop over starting counts 1..3

    // Back to 0 (DelRef() true only on the 1 -> 0), so the node is destroyed
    // as a node that no pointer ever adopted.
    for (long n = 4; n > 1; --n) EXPECT_FALSE(node.DelRef());
    EXPECT_TRUE(node.DelRef());
    EXPECT_EQ(node.use_count(), 0);
} // NodeTryAddRef

// Whole-chain reclamation: build a list with a tracked payload, churn it, then drop
// the list and every iterator. The iterative ~Node() must free the entire chain
// (including any logically-deleted graveyard nodes) with no payload left alive.
TYPED_TEST(LockFreeListTest, ReclaimsEntireChain) {
    using TList = LockFreeList<Tracked, TypeParam::template ptr_type>;
    using TNode = typename TList::Node;
    using TPtr = typename TypeParam::template ptr_type<TNode>;
    // True for a policy that declares `deferred_reclamation = true`
    // (intr_shared_ptr_hp). A nested requirement, so that a policy without the
    // member yields false instead of a hard error.
    constexpr bool deferred = requires { requires TPtr::deferred_reclamation; };

    const int base = Tracked::alive.load();
    {
        TList list(this->factory.template operator()<TNode>());

        for (int i = 0; i < 50; ++i)
            list.insert_after(list.before_begin(), this->factory.template operator()<TNode>(i));

        // Hold an iterator on a middle node, then erase the front several times so
        // that node lands in a graveyard chain reachable only through the iterator.
        auto mid = list.begin();
        for (int i = 0; i < 5; ++i) ++mid;
        for (int i = 0; i < 20; ++i) list.erase_after(list.before_begin());

        EXPECT_GT(Tracked::alive.load(), base); // nodes still alive while in scope
        // `mid`, `list` go out of scope here.
    }
    if constexpr (!deferred) {
        EXPECT_EQ(Tracked::alive.load(), base); // every node payload reclaimed
    } else {
        // Deferred reclamation: EXACTLY two drains, a white-box oracle that the
        // ~Node() walk still turns the chain into one batch under hazard
        // pointers. Traced: the erased front node 49 was retired when it was
        // unlinked and, still pending, holds the graveyard 48..30 and through
        // 30->next the live 29..0; dropping the list retires the dummy last.
        // The retired list is LIFO, so drain 1 destroys the dummy first: its
        // walk stops at 29, which 30->next still holds; then node 49, whose
        // walk carries 48..0 down the chain, retiring each node (51 payloads
        // alive -> 49). Drain 2 destroys those 49 (-> 0). A third drain would
        // find nothing, so do not add one: needing it would mean the walk
        // stopped working and the chain is dying one node per scan.
        drain_reclamation();
        EXPECT_GT(Tracked::alive.load(), base) << "one drain reclaimed the whole chain";
        drain_reclamation();
        EXPECT_EQ(Tracked::alive.load(), base) << "two drains did not reclaim the whole chain";
    } // deferred reclamation: exactly two drains
} // ReclaimsEntireChain

TYPED_TEST(LockFreeListTest, StressTest) {
    typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());
    const int num_threads = 8;
    const int num_iters = 1000;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            std::mt19937 rng(i);
            std::uniform_int_distribution<int> dist(0, 99);
            
            for (int j = 0; j < num_iters; ++j) {
                int op = dist(rng);
                if (op < 50) {
                    // insert
                    list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(j));
                } else if (op < 80) {
                    // erase
                    list.erase_after(list.before_begin());
                } else {
                    // traverse
                    int count = 0;
                    for (auto it = list.begin(); it != list.end() && count < 10; ++it) {
                        count++;
                        int val = *it;
                        // use val to prevent optimization out
                        volatile int dummy = val;
                        (void)dummy;
                    }
                }
            }
        });
    }
    
    for (std::thread& t : threads) {
        t.join();
    }

#ifndef NDEBUG
    // Debug-only invariant check: with all threads joined, a drain loop must
    // empty the list completely, dead nodes included. This is the regression
    // check for the helping path in erase_after(): without helping, a marked
    // node whose unlink CAS lost to a concurrent insert stays linked forever,
    // and erase_after() over that edge keeps returning false while live nodes
    // remain reachable behind it -- a state this test cannot see otherwise,
    // because a false return also legitimately means "list empty" (details:
    // `lock_free_list_bugs.md`, bug 1).
    while (list.erase_after(list.before_begin())) {}
    EXPECT_EQ(list.begin(), list.end());
#endif // NDEBUG
}

// Regression hammer for the helping path in erase_after(): one pure inserter and
// one pure eraser collide on the same anchor, maximizing the chance that an
// insert lands inside an eraser's window between its mark CAS and its unlink
// CAS. The unlink CAS then fails, and without helping the marked node stays
// linked for good, wedging the drain below. The mixed-op StressTest above is
// too diffuse to hit that interleaving reliably; this shape strands marked
// nodes within a few thousand operations pre-helping (details:
// `lock_free_list_bugs.md`, bug 1).
TYPED_TEST(LockFreeListTest, InsertEraseHammer) {
    typename TestFixture::List list(this->factory.template operator()<typename TestFixture::Node>());
    constexpr int num_inserts = 50000;

    std::atomic<bool> stop{false};
    std::thread eraser([&]() {
        while (!stop.load()) { list.erase_after(list.before_begin()); }
    });
    std::thread inserter([&]() {
        for (int i = 0; i < num_inserts; ++i) {
            list.insert_after(list.before_begin(), this->factory.template operator()<typename TestFixture::Node>(i));
        }
    });
    inserter.join();
    stop.store(true);
    eraser.join();

#ifndef NDEBUG
    // Same invariant as in StressTest: the drain must empty the list, dead
    // nodes included; a wedged erase_after() leaves reachable nodes behind.
    while (list.erase_after(list.before_begin())) {}
    EXPECT_EQ(list.begin(), list.end());
#endif // NDEBUG
} // InsertEraseHammer

// Leak check under concurrency for every policy: threads insert, erase and
// traverse a list of Tracked payloads, traversers parking iterators on nodes
// that others erase (graveyard pins); then the list is dropped and every node
// payload must be gone. Under intr_shared_ptr_hp nodes are destroyed by mm_hp
// scans, on whichever thread crosses the threshold (including the workers,
// while they run), so after the joins and the list's destruction the test
// drains until the count is back to its base: the live count after drains is
// the leak oracle for retired nodes (LeakSanitizer does not see retired but
// unreclaimed objects). For the synchronous policies the predicate already
// holds and no drain runs.
TYPED_TEST(LockFreeListTest, NoNodeLeakUnderStress) {
    using TList = LockFreeList<Tracked, TypeParam::template ptr_type>;
    using TNode = typename TList::Node;
    constexpr int num_threads = 6;
    constexpr int num_iters = 3000;
    // Round cap: a dropped list needs two rounds and graveyard layers a few
    // more; reaching the cap means a leak.
    constexpr int max_drain_rounds = 8;

    const int base = Tracked::alive.load();
    {
        TList list(this->factory.template operator()<TNode>());
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i]() {
                std::mt19937 rng(i);
                std::uniform_int_distribution<int> dist(0, 99);
                typename TList::iterator parked;   // pins a node across iterations
                for (int j = 0; j < num_iters; ++j) {
                    int op = dist(rng);
                    if (op < 45) {
                        list.insert_after(list.before_begin(), this->factory.template operator()<TNode>(j));
                    } else if (op < 80) {
                        list.erase_after(list.before_begin());
                    } else if (op < 90) {
                        parked = list.begin();     // later erasures turn it into a graveyard pin
                    } else {
                        int count = 0;
                        for (auto it = parked.curr_ ? parked : list.begin(); it != list.end() && count < 10; ++it) {
                            ++count;
                            EXPECT_GE((*it).v, 0);
                        } // walk at most 10 nodes, possibly through the graveyard
                    }
                } // loop over operations
            });
        } // loop over threads
        for (std::thread& t : threads) t.join();
    } // list dropped
    ASSERT_TRUE(drain_until([base] { return Tracked::alive.load() == base; }, max_drain_rounds))
        << Tracked::alive.load() - base << " node payloads still alive after " << max_drain_rounds << " drain rounds";
} // NoNodeLeakUnderStress
