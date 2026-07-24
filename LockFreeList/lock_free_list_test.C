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
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <random>

#include "atomic_shared_ptr_concept.h"
#include "intr_shared_ptr.h"
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

// Uniform node factory: the three pointer families construct their
// shared_ptr_type differently (std adapter: from make_shared; intrusive:
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
        } else if constexpr (std::is_same_v<SharedPtr, typename intr_shared_ptr<Node, Node>::shared_ptr_type>) {
            return SharedPtr(new Node(v));
        } else {
            return SharedPtr(parlay::make_shared<Node>(v));
        }
    }
};

template <typename Wrapper>
class LockFreeListTest : public ::testing::Test {
protected:
    using List = LockFreeList<int, Wrapper::template ptr_type>;
    using Node = typename List::Node;
    
    Factory<Wrapper> factory;
};

using PtrWrappers = ::testing::Types<
    StdAtomicWrapper,
    IntrPtrWrapper,
    ParlayWrapper
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

// Whole-chain reclamation: build a list with a tracked payload, churn it, then drop
// the list and every iterator. The iterative ~Node() must free the entire chain
// (including any logically-deleted graveyard nodes) with no payload left alive.
TYPED_TEST(LockFreeListTest, ReclaimsEntireChain) {
    using TList = LockFreeList<Tracked, TypeParam::template ptr_type>;
    using TNode = typename TList::Node;

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
    EXPECT_EQ(Tracked::alive.load(), base); // every node payload reclaimed
}

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
    
    for (auto& t : threads) {
        t.join();
    }

#ifndef NDEBUG
    // Debug-only invariant check: with all threads joined, a drain loop must
    // empty the list completely, dead nodes included. This is the regression
    // check for the helping path in erase_after(): without helping, a stuck
    // logically-deleted node wedges erase_after() into returning false while
    // live nodes remain reachable (lock_free_list_bugs.md, bug 1) -- a state
    // this test cannot see otherwise, because a false return also
    // legitimately means "list empty".
    while (list.erase_after(list.before_begin())) {}
    EXPECT_EQ(list.begin(), list.end());
#endif // NDEBUG
}

// Regression hammer for lock_free_list_bugs.md bug 1: one pure inserter and one
// pure eraser collide on the same anchor, maximizing the chance that an insert
// lands inside an eraser's window between its mark CAS and its unlink CAS. The
// mixed-op StressTest above is too diffuse to hit that interleaving reliably;
// this shape strands marked nodes within a few thousand operations pre-helping.
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
