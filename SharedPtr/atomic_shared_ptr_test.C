#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <memory>
#include <random>

#include "atomic_shared_ptr_concept.h"
#include "intr_shared_ptr.h"
#include "lock_free_shared_ptr/atomic_shared_ptr.hpp"

struct Data {
    // Counts currently-alive Data objects so tests can assert that no instance is
    // leaked and that the managed object is reclaimed promptly when its last strong
    // reference is dropped (all three pointer types dispose the object synchronously
    // at strong-count zero, even parlay, which only *defers* control-block memory).
    static std::atomic<int> live_count;

    int value;
    std::atomic<int> ref_count{0};
    Data(int v) : value(v) { live_count.fetch_add(1, std::memory_order_relaxed); }
    ~Data() { live_count.fetch_sub(1, std::memory_order_relaxed); }
    void AddRef() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    bool DelRef() { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const { return ref_count.load(std::memory_order_relaxed); }
};

std::atomic<int> Data::live_count{0};

template <typename PtrType>
typename PtrType::shared_ptr_type make_shared_data(int v) {
    if constexpr (std::is_same_v<PtrType, StdAtomicSharedPtrAdapter<Data>>) {
        return typename PtrType::shared_ptr_type(std::make_shared<Data>(v));
    } else if constexpr (std::is_same_v<PtrType, intr_shared_ptr<Data>>) {
        return typename PtrType::shared_ptr_type(new Data(v));
    } else if constexpr (std::is_same_v<PtrType, parlay::atomic_shared_ptr<Data>>) {
        return typename PtrType::shared_ptr_type(parlay::make_shared<Data>(v));
    }
}

template <typename PtrType>
class AtomicSharedPtrTest : public ::testing::Test {};

using PtrTypes = ::testing::Types<
    StdAtomicSharedPtrAdapter<Data>,
    intr_shared_ptr<Data>,
    parlay::atomic_shared_ptr<Data>
>;

TYPED_TEST_SUITE(AtomicSharedPtrTest, PtrTypes);

TYPED_TEST(AtomicSharedPtrTest, BasicFunctionality) {
    TypeParam ptr;
    EXPECT_FALSE(ptr.load());
    
    auto sp = make_shared_data<TypeParam>(42);
    ptr.store(sp);
    
    auto loaded = ptr.load();
    EXPECT_TRUE(loaded);
    EXPECT_EQ(loaded->value, 42);
}

TYPED_TEST(AtomicSharedPtrTest, StressTest) {
    TypeParam ptr(make_shared_data<TypeParam>(0));
    
    constexpr int num_threads = 8;
    constexpr int num_iters = 10000;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            std::mt19937 rng(std::random_device{}());
            std::uniform_int_distribution<int> dist(0, 10);
            
            for (int j = 0; j < num_iters; ++j) {
                int op = dist(rng);
                if (op < 6) { // 60% load
                    auto p = ptr.load();
                    if (p) {
                        EXPECT_GE(p->value, 0);
                    }
                } else if (op < 8) { // 20% CAS same
                    auto expected = ptr.load();
                    if (expected) {
                        ptr.compare_exchange_strong(expected, expected);
                    }
                } else { // 20% CAS new
                    auto expected = ptr.load();
                    if (expected) {
                        auto new_p = make_shared_data<TypeParam>(expected->value + 1);
                        ptr.compare_exchange_strong(expected, new_p);
                    }
                }
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    auto final_p = ptr.load();
    EXPECT_TRUE(final_p);
}

TYPED_TEST(AtomicSharedPtrTest, MarkingCorrectness) {
    if constexpr (TypeParam::supports_marking) {
        TypeParam ptr(make_shared_data<TypeParam>(42));
        
        auto p = ptr.load();
        EXPECT_FALSE(p.is_marked());
        
        auto marked_p = p.set_mark();
        EXPECT_TRUE(marked_p.is_marked());
        
        bool res = ptr.compare_exchange_strong(p, marked_p);
        EXPECT_TRUE(res);
        
        auto loaded = ptr.load();
        EXPECT_TRUE(loaded.is_marked());
        EXPECT_EQ(loaded->value, 42);
        
        // CAS expecting unmarked should fail
        auto unmarked_p = loaded.get_unmarked();
        auto new_p = make_shared_data<TypeParam>(100);
        res = ptr.compare_exchange_strong(unmarked_p, new_p);
        EXPECT_FALSE(res);
        EXPECT_TRUE(unmarked_p.is_marked()); // expected gets updated to actual value (marked)
        
        // CAS expecting marked should succeed
        res = ptr.compare_exchange_strong(marked_p, new_p);
        EXPECT_TRUE(res);
        
        loaded = ptr.load();
        EXPECT_FALSE(loaded.is_marked());
        EXPECT_EQ(loaded->value, 100);
    }
}

TYPED_TEST(AtomicSharedPtrTest, MarkedPointerLoadCrash) {
    if constexpr (TypeParam::supports_marking) {
        TypeParam ptr; // initially nullptr
        
        // 1. Store a marked nullptr
        auto null_ptr = ptr.load();
        auto marked_null = null_ptr.set_mark();
        ptr.store(marked_null);
        
        // 2. Load the marked nullptr.
        // In the buggy implementation, this crashed because unmarked_cb became nullptr 
        // after stripping the mark bit, and increment_strong_count_if_nonzero() was 
        // called on nullptr, causing a SEGV.
        auto loaded_null = ptr.load();
        EXPECT_TRUE(loaded_null.is_marked());
        EXPECT_FALSE(loaded_null); // should still evaluate to false (nullptr)

        // 3. Store a marked valid pointer
        auto valid_ptr = make_shared_data<TypeParam>(100);
        auto marked_valid = valid_ptr.set_mark();
        ptr.store(marked_valid);

        // 4. Load the marked valid pointer.
        // In the buggy implementation, this crashed because the copy constructor of
        // parlay::shared_ptr attempted to increment the reference count of an unmasked
        // control block (with the mark bit still attached), causing pointer corruption.
        auto loaded_valid = ptr.load();
        EXPECT_TRUE(loaded_valid.is_marked());
        EXPECT_TRUE(loaded_valid);
        EXPECT_EQ(loaded_valid->value, 100);
        
        // 5. Test copy constructor directly on a marked pointer
        auto copy_of_marked = loaded_valid;
        EXPECT_TRUE(copy_of_marked.is_marked());
        EXPECT_EQ(copy_of_marked->value, 100);
    }
}

TYPED_TEST(AtomicSharedPtrTest, CompareExchangeMemoryOrderUB) {
    TypeParam ptr;
    auto expected = make_shared_data<TypeParam>(1);
    auto desired = make_shared_data<TypeParam>(2);
    
    ptr.store(expected);
    
    // CAS with acq_rel. std::atomic::store cannot take acq_rel!
    // This used to cause an assertion failure/UB in intr_shared_ptr.
    bool res = ptr.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
    EXPECT_TRUE(res);
    EXPECT_EQ(ptr.load()->value, 2);
}

// store() over a non-null value must release (and reclaim) the previous object,
// and destroying the atomic must release the last one. Verifies there is no leak
// and no premature free across a replace.
TYPED_TEST(AtomicSharedPtrTest, StoreReplacesAndReclaims) {
    const int base = Data::live_count.load();
    {
        TypeParam ptr;
        ptr.store(make_shared_data<TypeParam>(1));
        EXPECT_EQ(Data::live_count.load(), base + 1);
        EXPECT_EQ(ptr.load()->value, 1);

        // Replace; with no outstanding loads the old object is reclaimed immediately.
        ptr.store(make_shared_data<TypeParam>(2));
        EXPECT_EQ(Data::live_count.load(), base + 1);
        EXPECT_EQ(ptr.load()->value, 2);
    }
    EXPECT_EQ(Data::live_count.load(), base); // atomic destroyed -> last object gone
}

// A value obtained from load() owns an independent strong reference: it must remain
// valid (and keep the object alive) after the atomic is overwritten or destroyed.
TYPED_TEST(AtomicSharedPtrTest, LoadOutlivesAtomic) {
    const int base = Data::live_count.load();
    typename TypeParam::shared_ptr_type held;
    {
        TypeParam ptr(make_shared_data<TypeParam>(7));
        held = ptr.load();

        const long shared = held.use_count();
        {
            auto second = ptr.load();           // another owner of the same object
            EXPECT_GT(held.use_count(), shared); // refcount grew while `second` lives
        }
        EXPECT_EQ(held.use_count(), shared);     // and shrank back

        ptr.store(make_shared_data<TypeParam>(99));
        EXPECT_EQ(held->value, 7);               // `held` still pins the old object
    }
    EXPECT_EQ(held->value, 7);                   // survives the atomic's destruction
    EXPECT_GE(Data::live_count.load(), base + 1);
    held = typename TypeParam::shared_ptr_type{};
    EXPECT_EQ(Data::live_count.load(), base);
}

// Generic CAS value semantics (independent of marking support): a matching expected
// swaps the value; a stale expected fails, leaves the atomic untouched, and is
// rewritten to the current value with a live reference.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeValueSemantics) {
    TypeParam ptr(make_shared_data<TypeParam>(1));

    auto snapshot = ptr.load();                   // observes value 1
    auto expected = snapshot;
    ASSERT_TRUE(ptr.compare_exchange_strong(expected, make_shared_data<TypeParam>(2)));
    EXPECT_EQ(ptr.load()->value, 2);

    // `snapshot` is now stale (atomic holds 2): CAS must fail.
    auto stale = snapshot;
    EXPECT_FALSE(ptr.compare_exchange_strong(stale, make_shared_data<TypeParam>(3)));
    EXPECT_EQ(ptr.load()->value, 2);              // unchanged by the failed CAS
    ASSERT_TRUE(stale);
    EXPECT_EQ(stale->value, 2);                   // expected updated to current value
}

// The concurrent StressTest exercises races; this adds the reclamation invariant:
// after all threads quiesce, exactly the one object still held by the atomic is
// alive (no reference-count drift, no leaked intermediates).
TYPED_TEST(AtomicSharedPtrTest, NoRefDriftUnderStress) {
    const int base = Data::live_count.load();
    {
        TypeParam ptr(make_shared_data<TypeParam>(0));

        constexpr int num_threads = 8;
        constexpr int num_iters = 5000;

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                std::mt19937 rng(std::random_device{}());
                std::uniform_int_distribution<int> dist(0, 1);
                for (int j = 0; j < num_iters; ++j) {
                    if (dist(rng) == 0) {
                        auto p = ptr.load();
                        if (p) EXPECT_GE(p->value, 0);
                    } else {
                        auto expected = ptr.load();
                        if (expected) {
                            ptr.compare_exchange_strong(expected,
                                make_shared_data<TypeParam>(expected->value + 1));
                        }
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        // Only the single value currently held by `ptr` should survive.
        EXPECT_EQ(Data::live_count.load(), base + 1);
    }
    EXPECT_EQ(Data::live_count.load(), base);
}
