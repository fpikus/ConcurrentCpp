#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <memory>
#include <random>

#include "atomic_shared_ptr_concept.h"
#include "hp_drain.h"
#include "hp_drain_gtest.h"
#include "intr_shared_ptr.h"
#include "intr_shared_ptr_hp.h"
#include "lock_free_shared_ptr/atomic_shared_ptr.hpp"

// The one pointee type shared by every TypeParam. Its hazard pointer base is
// UNCONDITIONAL: intr_shared_ptr_hp requires it, and a policy-conditional base
// is not expressible with one non-template Data. It adds 24 bytes to every Data
// and changes nothing else for the other pointer types (they never use it).
struct Data : std::hazard_pointer_obj_base<Data> {
    // Counts currently-alive Data objects so tests can assert that no instance is
    // leaked and that the managed object is reclaimed when its last strong
    // reference is dropped. The std adapter, intr_shared_ptr and parlay (which
    // only *defers* control-block memory) destroy the object synchronously at
    // strong-count zero; intr_shared_ptr_hp only retires it, and it is destroyed
    // at a later mm_hp scan. So tests compare live_count against a base captured
    // after a drain, and drain again before every "object gone" assertion
    // (hp_drain.h).
    static std::atomic<int> live_count;

    // Resurrection checker. A 0 -> 1 increment is legal exactly once, when the
    // first owner adopts a fresh object; every later 0 -> 1 is a resurrection of
    // a retired object (what a plain fetch_add in place of TryAddRef() in a
    // hazard-pointer load would do). `adopted` flips on the first 0 -> 1; a
    // 0 -> 1 that finds it already set bumps `resurrections`, which the
    // fixture's TearDown() asserts is 0. TryAddRef() never increments from 0
    // and needs no check. Only the intrusive pointers call the hooks.
    static std::atomic<int> resurrections;
    std::atomic<bool> adopted{false};    // set by the first (adopting) 0 -> 1

    int value;                           // payload; poisoned to -1 by the destructor
    std::atomic<int> ref_count{0};       // the intrusive strong count (0 until adopted)

    Data(int v) : value(v) { live_count.fetch_add(1, std::memory_order_relaxed); }

    // Poisons the payload: StressTest and NoRefDriftUnderStress assert
    // `EXPECT_GE(p->value, 0)` on loaded handles (every value a test stores is
    // >= 0), so a load() that returned a reclaimed object whose memory was not
    // reused yet reads -1. The plain store survives only at -O0/-O1 (compilers
    // remove it as a dead store at -O2 and above); the tests are built at -O0.
    // Under intr_shared_ptr_hp this destructor runs on whichever thread crosses
    // mm_hp's scan threshold.
    ~Data() {
        value = -1;
        live_count.fetch_sub(1, std::memory_order_relaxed);
    } // ~Data()

    // AddRef(): increment; records a resurrection if it incremented from 0 on
    // an already-adopted object.
    void AddRef() noexcept {
        if (ref_count.fetch_add(1, std::memory_order_relaxed) == 0 &&
            adopted.exchange(true, std::memory_order_relaxed)) {
            resurrections.fetch_add(1, std::memory_order_relaxed);
        } // 0 -> 1 on an object that was adopted before: resurrection
    } // AddRef()

    // TryAddRef(): the reference form of the hook contract in
    // IntrSharedPtr/intr_pointee.h (increment iff nonzero; every observed 0 read
    // with acquire; the CAS acquire on success). No resurrection check needed: it
    // never increments from 0.
    bool TryAddRef() noexcept {
        int count = ref_count.load(std::memory_order_acquire);
        while (count != 0) {
            if (ref_count.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return true;
            }
            // The failed CAS refreshed `count` with a relaxed read; a 0 seen
            // that way does not synchronize. Re-read it with acquire.
            if (count == 0) count = ref_count.load(std::memory_order_acquire);
        } // CAS loop while the count is nonzero
        return false;
    } // TryAddRef()

    // DelRef(): decrement, acq_rel; true iff this call made the 1 -> 0
    // transition (the caller then deletes or retires, per policy).
    bool DelRef() noexcept { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const noexcept { return ref_count.load(std::memory_order_relaxed); }
}; // struct Data

std::atomic<int> Data::live_count{0};
std::atomic<int> Data::resurrections{0};

template <typename PtrType>
typename PtrType::shared_ptr_type make_shared_data(int v) {
    if constexpr (std::is_same_v<PtrType, StdAtomicSharedPtrAdapter<Data>>) {
        return typename PtrType::shared_ptr_type(std::make_shared<Data>(v));
    } else if constexpr (std::is_same_v<PtrType, intr_shared_ptr<Data>> ||
                         std::is_same_v<PtrType, intr_shared_ptr_hp<Data>>) {
        return typename PtrType::shared_ptr_type(new Data(v));   // intrusive: adopt a raw new
    } else if constexpr (std::is_same_v<PtrType, parlay::atomic_shared_ptr<Data>>) {
        return typename PtrType::shared_ptr_type(parlay::make_shared<Data>(v));
    }
}

// Fixture: brackets every test with mm_hp drains, for every TypeParam (a drain
// is ~1000 filler retirements, milliseconds; the policies without deferred
// reclamation are unaffected, so there is no per-policy branch).
// - SetUp() drains FIRST and then captures the bases, so nothing retired by an
//   earlier test is pending (Data's destructor retires nothing, so one round
//   suffices) and a test that forgets its own drain fails deterministically.
// - TearDown() drains and checks that the test leaked nothing and resurrected
//   nothing. GoogleTest runs TearDown() BEFORE it destroys the fixture's data
//   members, so a member holding a pointer would still be alive at the drain;
//   this fixture has no such member, and must not get one (or must reset it in
//   TearDown() before the drain). The test body's locals are gone by then.
template <typename PtrType>
class AtomicSharedPtrTest : public ::testing::Test {
protected:
    // Live Data count after SetUp()'s drain.
    int base_ = 0;

    void SetUp() override {
        drain_reclamation();
        base_ = Data::live_count.load();
        Data::resurrections.store(0);
    }

    void TearDown() override {
        drain_reclamation();
        EXPECT_EQ(Data::resurrections.load(), 0) << "an adopted Data went 0 -> 1 again";
        EXPECT_EQ(Data::live_count.load(), base_) << "Data leaked (or destroyed twice) by this test";
    }
}; // class AtomicSharedPtrTest

using PtrTypes = ::testing::Types<
    StdAtomicSharedPtrAdapter<Data>,
    intr_shared_ptr<Data>,
    parlay::atomic_shared_ptr<Data>,
    intr_shared_ptr_hp<Data>
>;
static_assert(AtomicSharedPtr<intr_shared_ptr_hp<Data>>);

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
    
    for (std::thread& t : threads) t.join();
    
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
        // Stripping the mark bit leaves a null control block, which the load
        // must not pass to increment_strong_count_if_nonzero(): that is a
        // SEGV.
        auto loaded_null = ptr.load();
        EXPECT_TRUE(loaded_null.is_marked());
        EXPECT_FALSE(loaded_null); // should still evaluate to false (nullptr)

        // 3. Store a marked valid pointer
        auto valid_ptr = make_shared_data<TypeParam>(100);
        auto marked_valid = valid_ptr.set_mark();
        ptr.store(marked_valid);

        // 4. Load the marked valid pointer.
        // The copy constructor of parlay::shared_ptr must increment the
        // reference count of the control block with the mark bit stripped;
        // incrementing through the address with the mark bit still attached
        // corrupts memory.
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
    
    // CAS with acq_rel. std::atomic::store cannot take acq_rel, so a pointer
    // whose successful CAS publishes with a store (intr_shared_ptr) must not
    // pass the success order to that store unchanged.
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

        // Replace; with no outstanding loads the old object is released at once:
        // destroyed synchronously by three of the policies, retired by
        // intr_shared_ptr_hp and destroyed by the drain.
        ptr.store(make_shared_data<TypeParam>(2));
        drain_reclamation();
        EXPECT_EQ(Data::live_count.load(), base + 1);
        EXPECT_EQ(ptr.load()->value, 2);
    }
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base); // atomic destroyed -> last object gone
} // StoreReplacesAndReclaims

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
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base);
} // LoadOutlivesAtomic

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
                        // Braces: EXPECT_GE expands to an if/else, which g++
                        // flags as a dangling else under -Werror.
                        if (p) {
                            EXPECT_GE(p->value, 0);
                        }
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
        for (std::thread& t : threads) t.join();

        // Only the single value currently held by `ptr` should survive, once
        // the objects released by the threads are reclaimed (quiescent: every
        // thread is joined).
        drain_reclamation();
        EXPECT_EQ(Data::live_count.load(), base + 1);
    }
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base);
} // NoRefDriftUnderStress

// ===========================================================================
// Contract tests
// ===========================================================================
//
// Black-box tests: every expectation below is derived from the documented
// contract of the pointer (the comments of intr_shared_ptr_hp.h and the
// AtomicSharedPtr concept in atomic_shared_ptr_concept.h) and from the
// documented contract of the drain helper (hp_drain.h), never from how the
// pointer happens to be implemented. Each test names the contract clause it
// checks and says why a violation of that clause makes it fail. They use the
// Data, make_shared_data<>() and AtomicSharedPtrTest defined above.
//   - Helpers used by both groups below.
//   - Typed tests: properties every pointer of the AtomicSharedPtr concept
//     promises (std compare_exchange_strong semantics, ownership, value
//     reclamation), written against the concept only and run over all four
//     pointers (std adapter, intr_shared_ptr, parlay, intr_shared_ptr_hp);
//     mark-dependent parts are guarded by `supports_marking`, like the marking
//     tests above.
//   - intr_shared_ptr_hp-only tests (fixture IntrSharedPtrHpTest): the value
//     type's documented member contract, compare_exchange_weak (which only this
//     pointer has), deferred reclamation, destruction on another thread,
//     T != U, a pointee that names the pointer while incomplete, and the
//     asserted memory-order preconditions.
// Drain discipline (hp_drain.h): the fixture drains in SetUp() and only then
// captures the live-count base; every "object is gone" assertion is preceded by
// a drain; drains run only when no other thread retires (after joins) and never
// from a destructor. A "still alive" assertion needs no drain.

// ===========================================================================
// Contract tests: helpers
// ===========================================================================

// Number of compare-exchange calls one changed-back run makes.
inline constexpr int changed_back_cas_count = 10'000;

// Value identity for any pointer of the concept: the same pointee and the same
// mark. Spelled through get() and is_marked() rather than operator== so that it
// means the same for every pointer type (the std adapter has no mark).
template <typename SP>
bool same_value(const SP& x, const SP& y) {
    return x.get() == y.get() && x.is_marked() == y.is_marked();
}

// Callables selecting the compare-exchange flavor a shared test body uses:
// cas(atomic, expected, desired) -> bool, with default memory orders.
struct StrongCas {
    template <typename PtrType>
    bool operator()(PtrType& atomic, typename PtrType::shared_ptr_type& expected,
                    const typename PtrType::shared_ptr_type& desired) const {
        return atomic.compare_exchange_strong(expected, desired);
    }
}; // struct StrongCas

struct WeakCas {
    template <typename PtrType>
    bool operator()(PtrType& atomic, typename PtrType::shared_ptr_type& expected,
                    const typename PtrType::shared_ptr_type& desired) const {
        return atomic.compare_exchange_weak(expected, desired);
    }
}; // struct WeakCas

// How the compare-exchange calls of one run_changed_back() ended.
struct ChangedBackOutcome {
    long successes = 0;          // calls that returned true
    long failures = 0;           // calls that returned false
    long failures_unchanged = 0; // false returns that left `expected` equal to the original value
    long failures_foreign = 0;   // false returns that left `expected` equal to neither value
};

// Changed-back race: an atomic starts at `original`; a flipper thread stores
// `other` then `original`, over and over, while the calling thread makes
// `num_cas` calls cas(atomic, expected, original), each with a fresh `expected`
// equal to `original`. The word therefore only ever holds `original` or
// `other`, and a call whose `expected` matches can lose only to a flip that is
// undone before (or while) it completes. Returns how the calls ended. The atomic
// is destroyed before returning, so the caller's handles are the only remaining
// owners of both pointees.
template <typename PtrType, typename Cas>
ChangedBackOutcome run_changed_back(const typename PtrType::shared_ptr_type& original,
                                    const typename PtrType::shared_ptr_type& other,
                                    Cas cas, int num_cas) {
    using SP = typename PtrType::shared_ptr_type;
    ChangedBackOutcome outcome;
    PtrType atomic(original);
    std::atomic<bool> flipping{false};   // set once the flipper runs
    std::atomic<bool> done{false};       // set when the CAS loop is over
    std::thread flipper([&] {
        flipping.store(true);
        while (!done.load(std::memory_order_relaxed)) {
            atomic.store(other);
            atomic.store(original);
        } // flip until the CAS loop is over
    });
    while (!flipping.load()) std::this_thread::yield();
    for (int i = 0; i < num_cas; ++i) {
        SP expected = original;
        if (cas(atomic, expected, original)) {
            ++outcome.successes;
            continue;
        }
        ++outcome.failures;
        if (same_value(expected, original)) {
            ++outcome.failures_unchanged;
        } else if (!same_value(expected, other)) {
            ++outcome.failures_foreign;
        }
    } // loop over compare-exchange calls
    done.store(true);
    flipper.join();
    return outcome;
} // run_changed_back()

// How a run_cas_counter() ended.
struct CounterOutcome {
    int final_value = 0;       // the value held by the atomic after all threads joined
    long poisoned_reads = 0;   // reads of `value` < 0 through a handle that should own a live object
};

// Lock-free counter: the atomic starts at Data(0); each of `num_threads`
// threads makes `increments_per_thread` increments with the documented loop
//   expected = load(); do { desired = Data(expected->value + 1) } while (!cas(expected, desired));
// re-deriving `desired` from the refreshed `expected` after every failure.
// With a linearizable CAS that refreshes `expected` to the current value on
// failure, every increment lands exactly once: the final value is the total
// number of increments. Every handle `expected` is read through must own a
// live object; a negative read (the poisoned payload of a reclaimed Data) is
// counted.
template <typename PtrType, typename Cas>
CounterOutcome run_cas_counter(Cas cas, int num_threads, int increments_per_thread) {
    using SP = typename PtrType::shared_ptr_type;
    PtrType atomic(make_shared_data<PtrType>(0));
    std::atomic<long> poisoned{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&] {
            for (int k = 0; k < increments_per_thread; ++k) {
                SP expected = atomic.load();
                if (expected->value < 0) poisoned.fetch_add(1);
                SP desired = make_shared_data<PtrType>(expected->value + 1);
                while (!cas(atomic, expected, desired)) {
                    if (expected->value < 0) poisoned.fetch_add(1);
                    desired = make_shared_data<PtrType>(expected->value + 1);
                } // retry with the refreshed expected
            } // loop over increments
        });
    } // loop over threads
    for (std::thread& t : threads) t.join();
    CounterOutcome outcome;
    outcome.final_value = atomic.load()->value;
    outcome.poisoned_reads = poisoned.load();
    return outcome;
} // run_cas_counter()

// Aliased compare-exchange under contention: `num_threads` threads each make
// `iterations` steps on one atomic; one step in four stores a fresh Data, the
// others load a value `e` and call cas(atomic, e, e) -- `expected` and
// `desired` are the same object, so on failure `desired` is reassigned in the
// middle of the call. Returns the number of poisoned reads through `e` (before
// and after the call), which must be 0: `e` owns a live reference throughout.
// The atomic is destroyed before returning; the caller checks the live count
// after a drain (an unbalanced pre-count of `desired` leaks an object, an extra
// release frees one early).
template <typename PtrType, typename Cas>
long run_alias_contention(Cas cas, int num_threads, int iterations) {
    using SP = typename PtrType::shared_ptr_type;
    PtrType atomic(make_shared_data<PtrType>(0));
    std::atomic<long> poisoned{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < iterations; ++i) {
                if ((i + t) % 4 == 3) {
                    atomic.store(make_shared_data<PtrType>(i));
                    continue;
                }
                SP e = atomic.load();
                if (e->value < 0) poisoned.fetch_add(1);
                (void)cas(atomic, e, e);
                if (e->value < 0) poisoned.fetch_add(1);
            } // loop over steps
        });
    } // loop over threads
    for (std::thread& t : threads) t.join();
    return poisoned.load();
} // run_alias_contention()

// ===========================================================================
// Contract tests: typed, valid for every pointer of the AtomicSharedPtr concept
// ===========================================================================

// Clause (compare_exchange_strong, success): returns true; the atomic holds
// `desired`'s value; the word's new reference to `desired`'s pointee is counted
// and `desired` keeps its own; the old pointee loses the word's reference;
// `expected` is untouched and still owns its reference.
// Fails if: the swap does not happen, `expected` is overwritten on success, or
// a count is not transferred (the use_count deltas are exact in one thread).
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongSuccessPostconditions) {
    using SP = typename TypeParam::shared_ptr_type;
    SP old_value = make_shared_data<TypeParam>(1);
    SP new_value = make_shared_data<TypeParam>(2);
    {
        TypeParam atomic(old_value);
        SP expected = old_value;
        const long old_before = old_value.use_count();   // handle, word, expected
        const long new_before = new_value.use_count();   // handle
        ASSERT_TRUE(atomic.compare_exchange_strong(expected, new_value));
        EXPECT_TRUE(same_value(atomic.load(), new_value));
        EXPECT_TRUE(same_value(expected, old_value));    // untouched
        EXPECT_EQ(expected->value, 1);
        EXPECT_EQ(old_value.use_count(), old_before - 1); // the word's reference was released
        EXPECT_EQ(new_value.use_count(), new_before + 1); // the word owns one; desired keeps its own
    }
    // The old value's last owner is `old_value`; dropping it reclaims the object.
    old_value = SP{};
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), this->base_ + 1);
} // CompareExchangeStrongSuccessPostconditions

// Clause (compare_exchange_strong, failure): returns false; the atomic is
// unchanged; `expected` is assigned the current value with a live reference
// (null if the word is null); `desired` is unchanged (its pre-count, if any,
// is undone). Partitions of `expected` vs the word: stale non-null vs
// non-null, null vs non-null, non-null vs null; plus the null-vs-null success.
// Fails if: the atomic changes, `expected` is not refreshed or does not own a
// reference (its pointee's count would not grow), or `desired`'s count drifts.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongFailurePostconditions) {
    using SP = typename TypeParam::shared_ptr_type;
    SP current = make_shared_data<TypeParam>(1);
    SP stale = make_shared_data<TypeParam>(2);
    SP desired = make_shared_data<TypeParam>(3);
    TypeParam atomic(current);

    // Stale non-null expected against a non-null word.
    SP expected = stale;
    const long current_before = current.use_count();   // handle, word
    const long stale_before = stale.use_count();       // handle, expected
    const long desired_before = desired.use_count();   // handle
    EXPECT_FALSE(atomic.compare_exchange_strong(expected, desired));
    EXPECT_TRUE(same_value(atomic.load(), current));   // unchanged
    EXPECT_TRUE(same_value(expected, current));        // refreshed
    ASSERT_TRUE(expected);
    EXPECT_EQ(expected->value, 1);
    EXPECT_EQ(current.use_count(), current_before + 1); // expected owns a live reference
    EXPECT_EQ(stale.use_count(), stale_before - 1);     // expected's old reference released
    EXPECT_EQ(desired.use_count(), desired_before);     // desired unchanged

    // Null expected against a non-null word.
    SP null_expected;
    EXPECT_FALSE(atomic.compare_exchange_strong(null_expected, desired));
    EXPECT_TRUE(same_value(null_expected, current));
    EXPECT_EQ(desired.use_count(), desired_before);

    // Non-null expected against a null word: refreshed to null.
    TypeParam empty;
    SP nonnull_expected = current;
    const long current_with_expected = current.use_count();
    EXPECT_FALSE(empty.compare_exchange_strong(nonnull_expected, desired));
    EXPECT_FALSE(nonnull_expected);
    EXPECT_FALSE(empty.load());
    EXPECT_EQ(current.use_count(), current_with_expected - 1);
    EXPECT_EQ(desired.use_count(), desired_before);

    // Null expected against a null word: succeeds.
    SP matching_null;
    EXPECT_TRUE(empty.compare_exchange_strong(matching_null, desired));
    EXPECT_TRUE(same_value(empty.load(), desired));
    EXPECT_EQ(desired.use_count(), desired_before + 1);
} // CompareExchangeStrongFailurePostconditions

// Clause (value = pointer AND mark; compare_exchange_strong compares the full
// word): an `expected` that differs from the word only in the mark fails, and
// is refreshed to the word's mark state; the refreshed value then succeeds.
// Both directions: marked word / unmarked expected and the reverse.
// Fails if the compare ignores the mark (the first CAS would succeed) or the
// refresh drops it.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongComparesMarkBit) {
    if constexpr (TypeParam::supports_marking) {
        using SP = typename TypeParam::shared_ptr_type;
        SP plain = make_shared_data<TypeParam>(5);
        SP marked = plain.set_mark();
        SP other = make_shared_data<TypeParam>(6);
        TypeParam atomic(marked);

        SP expected = plain;                       // unmarked twin of the word
        EXPECT_FALSE(atomic.compare_exchange_strong(expected, other));
        EXPECT_TRUE(same_value(atomic.load(), marked));
        EXPECT_TRUE(expected.is_marked());
        EXPECT_TRUE(same_value(expected, marked));
        EXPECT_TRUE(atomic.compare_exchange_strong(expected, other));
        EXPECT_TRUE(same_value(atomic.load(), other));

        SP marked_expected = other.set_mark();     // marked twin of the (unmarked) word
        EXPECT_FALSE(atomic.compare_exchange_strong(marked_expected, plain));
        EXPECT_FALSE(marked_expected.is_marked());
        EXPECT_TRUE(same_value(marked_expected, other));
        EXPECT_TRUE(same_value(atomic.load(), other));
    }
} // CompareExchangeStrongComparesMarkBit

// Clause (aliasing: `expected` and `desired` may be the same object; the counts
// balance). Three partitions: success (the word is replaced by the same value);
// failure where the caller holds another reference to the aliased pointee; and
// failure where the aliased object is the ONLY owner of its pointee, so that
// reassigning `expected` drops the pointee's last reference in the middle of the
// call. Exact use_count deltas check the first two; the live count after a
// drain checks the third (a pre-count undone through the reassigned object
// leaks or double-releases).
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongAliasedExpectedDesired) {
    using SP = typename TypeParam::shared_ptr_type;
    SP x = make_shared_data<TypeParam>(1);
    SP y = make_shared_data<TypeParam>(2);
    TypeParam atomic(x);
    {
        SP e = x;                                  // x: handle, word, e
        EXPECT_TRUE(atomic.compare_exchange_strong(e, e));
        EXPECT_TRUE(same_value(atomic.load(), x));
        EXPECT_TRUE(same_value(e, x));
        EXPECT_EQ(x.use_count(), 3);
    }
    EXPECT_EQ(x.use_count(), 2);

    atomic.store(y);                               // x: handle; y: handle, word
    {
        SP e = x;                                  // x: handle, e
        EXPECT_FALSE(atomic.compare_exchange_strong(e, e));
        EXPECT_TRUE(same_value(e, y));
        EXPECT_TRUE(same_value(atomic.load(), y));
        EXPECT_EQ(x.use_count(), 1);               // e's reference released, pre-count undone
        EXPECT_EQ(y.use_count(), 3);               // handle, word, e
    }
    EXPECT_EQ(y.use_count(), 2);

    {
        SP e = make_shared_data<TypeParam>(3);     // e is the only owner of Data(3)
        EXPECT_FALSE(atomic.compare_exchange_strong(e, e));
        EXPECT_TRUE(same_value(e, y));
        EXPECT_EQ(e->value, 2);
    }
    EXPECT_EQ(y.use_count(), 2);
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), this->base_ + 2);   // x and y; Data(3) reclaimed once
} // CompareExchangeStrongAliasedExpectedDesired

// Clause (orders): `success` may be any order; `failure` may be relaxed,
// acquire or seq_cst (consume is also permitted but is deprecated in C++26 and
// not exercised). For every combination, a matching CAS succeeds and a stale
// one fails with a refreshed `expected`.
// Fails if some permitted combination is rejected (an assert fires) or
// changes the outcome.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongAcceptsPermittedOrders) {
    using SP = typename TypeParam::shared_ptr_type;
    constexpr std::memory_order success_orders[] = {
        std::memory_order_relaxed, std::memory_order_acquire, std::memory_order_release,
        std::memory_order_acq_rel, std::memory_order_seq_cst};
    constexpr std::memory_order failure_orders[] = {
        std::memory_order_relaxed, std::memory_order_acquire, std::memory_order_seq_cst};
    SP first = make_shared_data<TypeParam>(1);
    SP second = make_shared_data<TypeParam>(2);
    for (std::memory_order success : success_orders) {
        for (std::memory_order failure : failure_orders) {
            TypeParam atomic(first);
            SP expected = first;
            EXPECT_TRUE(atomic.compare_exchange_strong(expected, second, success, failure));
            EXPECT_TRUE(same_value(atomic.load(), second));
            SP stale = first;
            EXPECT_FALSE(atomic.compare_exchange_strong(stale, first, success, failure));
            EXPECT_TRUE(same_value(stale, second));
        } // loop over failure orders
    } // loop over success orders
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(second.use_count(), 1);
} // CompareExchangeStrongAcceptsPermittedOrders

// Clause (store/load orders): store accepts relaxed, release and seq_cst; load
// accepts relaxed, acquire and seq_cst (consume: see above). Every combination
// stores and reads back the value.
TYPED_TEST(AtomicSharedPtrTest, StoreAndLoadAcceptPermittedOrders) {
    constexpr std::memory_order store_orders[] = {
        std::memory_order_relaxed, std::memory_order_release, std::memory_order_seq_cst};
    constexpr std::memory_order load_orders[] = {
        std::memory_order_relaxed, std::memory_order_acquire, std::memory_order_seq_cst};
    TypeParam atomic;
    int value = 0;
    for (std::memory_order store_order : store_orders) {
        for (std::memory_order load_order : load_orders) {
            ++value;
            atomic.store(make_shared_data<TypeParam>(value), store_order);
            EXPECT_EQ(atomic.load(load_order)->value, value);
        } // loop over load orders
    } // loop over store orders
} // StoreAndLoadAcceptPermittedOrders

// Clause (compare_exchange_strong, std semantics): it NEVER returns false with
// `expected` equal (pointer and mark) to the value the caller passed in; if the
// word changed and changed back it retries until it succeeds or observes a
// genuinely different value. With the word flipping A -> B -> A, every false
// return must leave `expected == B`.
// Fails if a changed-back word is reported as a failure with the original
// `expected` (failures_unchanged > 0) -- the oracle is 0, never a rate -- or if
// `expected` is refreshed to a value the word never held. The use counts after
// the run check that no reference leaked or was over-released on any path.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongChangedBack) {
    using SP = typename TypeParam::shared_ptr_type;
    SP a = make_shared_data<TypeParam>(1);
    SP b = make_shared_data<TypeParam>(2);
    const ChangedBackOutcome outcome =
        run_changed_back<TypeParam>(a, b, StrongCas{}, changed_back_cas_count);
    EXPECT_EQ(outcome.failures_unchanged, 0)
        << outcome.failures << " failures, " << outcome.successes << " successes";
    EXPECT_EQ(outcome.failures_foreign, 0);
    EXPECT_EQ(outcome.successes + outcome.failures, changed_back_cas_count);
    EXPECT_EQ(a.use_count(), 1);
    EXPECT_EQ(b.use_count(), 1);
} // CompareExchangeStrongChangedBack

// Clause (as above, "including a marked original"): the word flips between a
// value and its mark twin -- same pointee, different mark -- in both
// directions (original marked, other unmarked; and the reverse). A compare that
// ignores the mark, or a changed-back check that compares the unmarked pointer,
// shows up as a false return with the original `expected`.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongChangedBackMarkedTwin) {
    if constexpr (TypeParam::supports_marking) {
        using SP = typename TypeParam::shared_ptr_type;
        SP plain = make_shared_data<TypeParam>(1);
        SP marked = plain.set_mark();
        const ChangedBackOutcome marked_original =
            run_changed_back<TypeParam>(marked, plain, StrongCas{}, changed_back_cas_count);
        EXPECT_EQ(marked_original.failures_unchanged, 0) << marked_original.failures << " failures";
        EXPECT_EQ(marked_original.failures_foreign, 0);
        const ChangedBackOutcome plain_original =
            run_changed_back<TypeParam>(plain, marked, StrongCas{}, changed_back_cas_count);
        EXPECT_EQ(plain_original.failures_unchanged, 0) << plain_original.failures << " failures";
        EXPECT_EQ(plain_original.failures_foreign, 0);
        EXPECT_EQ(plain.use_count(), 2);           // plain and marked
    }
} // CompareExchangeStrongChangedBackMarkedTwin

// Clauses (load and compare_exchange_strong are linearizable; on failure
// `expected` holds the current value with a live reference; the documented
// loop "do { ... } while (!cas(expected, desired))" makes progress): four
// threads each make 2500 increments of a counter held as Data(value); the final
// value must be exactly 10000. A lost increment (a success reported for a
// stale `expected`), a duplicated one, or a refresh that does not carry the
// current value changes the total; a refresh that hands out a reclaimed object
// reads the poisoned payload. Leaked intermediates are caught by the fixture's
// live-count check after the TearDown drain.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongLoopCountsEveryIncrement) {
    constexpr int num_threads = 4;
    constexpr int increments_per_thread = 2500;
    const CounterOutcome outcome =
        run_cas_counter<TypeParam>(StrongCas{}, num_threads, increments_per_thread);
    EXPECT_EQ(outcome.final_value, num_threads*increments_per_thread);
    EXPECT_EQ(outcome.poisoned_reads, 0);
} // CompareExchangeStrongLoopCountsEveryIncrement

// Clause (aliasing under contention: cas(e, e) balances its counts whatever
// the outcome). See run_alias_contention(). After the run and a drain, every
// Data the run created is gone (the fixture checks the live count again after
// its own drain); no read through `e` sees a reclaimed object.
TYPED_TEST(AtomicSharedPtrTest, CompareExchangeStrongAliasBalanceUnderContention) {
    EXPECT_EQ(run_alias_contention<TypeParam>(StrongCas{}, 4, 5000), 0);
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), this->base_);
}

// Clauses (ownership): store() releases the previously held pointee; the
// atomic's destructor releases the held pointee; a value obtained elsewhere
// keeps its pointee alive across both; an object is reclaimed once its last
// owner is gone. Each "gone" check follows a drain; the "still alive" checks
// hold with or without one.
// Fails if a replaced or destroyed word leaks its pointee (live count too
// high after the drain) or releases one it does not own (too low, or the
// poisoned payload read through `held`).
TYPED_TEST(AtomicSharedPtrTest, OwnershipAcrossReplaceAndDestroy) {
    using SP = typename TypeParam::shared_ptr_type;
    const int base = this->base_;
    SP held = make_shared_data<TypeParam>(1);
    {
        TypeParam atomic(held);
        atomic.store(make_shared_data<TypeParam>(2));    // object 1 is still held
        EXPECT_GE(Data::live_count.load(), base + 2);
        drain_reclamation();
        EXPECT_EQ(Data::live_count.load(), base + 2);
        EXPECT_EQ(held->value, 1);

        atomic.store(make_shared_data<TypeParam>(3));    // object 2 had only the word
        drain_reclamation();
        EXPECT_EQ(Data::live_count.load(), base + 2);    // objects 1 and 3
        EXPECT_EQ(atomic.load()->value, 3);

        atomic.store(held);                              // object 3 had only the word
        drain_reclamation();
        EXPECT_EQ(Data::live_count.load(), base + 1);
        EXPECT_EQ(held.use_count(), 2);
    } // the atomic's destructor releases its reference to object 1
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base + 1);
    EXPECT_EQ(held->value, 1);
    EXPECT_EQ(held.use_count(), 1);
    {
        TypeParam atomic(make_shared_data<TypeParam>(4)); // the word is the only owner
    } // the atomic's destructor releases the last reference
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base + 1);
    held = SP{};
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base);
} // OwnershipAcrossReplaceAndDestroy

// Clause (load() is linearizable: it returns a value the atomic held at some
// instant during the call): one writer stores Data(1), Data(2), ...,
// Data(num_stores) in order; each reader's successive loads must therefore
// never go backward, and a load made after the reader saw the writer's
// completion flag must return the last value (the flag is set after the last
// store; seq_cst on both sides). Every loaded handle must own a live object.
// Fails if a load returns a value older than one already returned to the same
// reader, a stale value after the writer finished, or a reclaimed object.
TYPED_TEST(AtomicSharedPtrTest, LoadNeverGoesBackward) {
    using SP = typename TypeParam::shared_ptr_type;
    constexpr int num_readers = 3;
    constexpr int num_stores = 10'000;
    TypeParam atomic(make_shared_data<TypeParam>(0));
    std::atomic<bool> writer_done{false};
    std::atomic<long> backward{0};       // loads older than the reader's previous load
    std::atomic<long> poisoned{0};       // loads of a reclaimed object
    std::atomic<long> stale_at_end{0};   // final loads that missed the last store
    std::vector<std::thread> readers;
    for (int r = 0; r < num_readers; ++r) {
        readers.emplace_back([&] {
            int last = 0;
            bool done = false;
            while (!done) {
                done = writer_done.load();
                const SP p = atomic.load();
                const int v = p->value;
                if (v < 0) {
                    poisoned.fetch_add(1);
                    continue;
                }
                if (v < last) backward.fetch_add(1);
                if (done && v != num_stores) stale_at_end.fetch_add(1);
                last = v;
            } // loop until a load made after the writer finished
        });
    } // loop over readers
    for (int i = 1; i <= num_stores; ++i) atomic.store(make_shared_data<TypeParam>(i));
    writer_done.store(true);
    for (std::thread& t : readers) t.join();
    EXPECT_EQ(backward.load(), 0);
    EXPECT_EQ(poisoned.load(), 0);
    EXPECT_EQ(stale_at_end.load(), 0);
} // LoadNeverGoesBackward

// ===========================================================================
// Contract tests: intr_shared_ptr_hp only
// ===========================================================================

using HpPtr = intr_shared_ptr_hp<Data>;
using HpValue = HpPtr::shared_ptr_type;

// The same drain-and-count fixture as the typed suite, for tests of members
// and behaviour only this pointer has.
class IntrSharedPtrHpTest : public AtomicSharedPtrTest<HpPtr> {};

// Bound on the retry loop around a compare_exchange_weak whose `expected`
// matches: std code loops without a bound; a test bounds it so that a weak CAS
// that never succeeds fails the test instead of hanging it.
inline constexpr int weak_cas_retry_bound = 1000;

// A weak CAS used the way std code uses one: called until it succeeds, here at
// most weak_cas_retry_bound times. For single-threaded use with an `expected`
// that matches the word: nothing else changes the word, so a (spurious)
// failure must refresh `expected` to the value it already had; any other
// refresh is reported as a test failure. Returns true iff a call succeeded
// within the bound.
[[nodiscard]] bool weak_cas_with_retries(HpPtr& atomic, HpValue& expected, const HpValue& desired,
                                         std::memory_order success = std::memory_order_seq_cst,
                                         std::memory_order failure = std::memory_order_seq_cst) {
    const HpValue original = expected;
    for (int attempt = 0; attempt < weak_cas_retry_bound; ++attempt) {
        if (atomic.compare_exchange_weak(expected, desired, success, failure)) return true;
        EXPECT_TRUE(expected == original) << "a spurious failure refreshed `expected` to another value";
    } // bounded retry loop
    ADD_FAILURE() << "no success in " << weak_cas_retry_bound << " weak CAS attempts";
    return false;
} // weak_cas_with_retries()

// The four pointee hooks with the documented semantics, for the test-only
// pointee types below (Data has its own, with the resurrection check).
struct PlainIntrusiveCount {
    std::atomic<int> ref_count{0};   // the intrusive strong count (0 until adopted)

    void AddRef() noexcept { ref_count.fetch_add(1, std::memory_order_relaxed); }

    // TryAddRef(): increment iff nonzero; false (count left at 0) iff it
    // observed 0, and every observed 0 is read with acquire.
    bool TryAddRef() noexcept {
        int count = ref_count.load(std::memory_order_acquire);
        while (count != 0) {
            if (ref_count.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return true;
            }
            if (count == 0) count = ref_count.load(std::memory_order_acquire);
        } // CAS loop while the count is nonzero
        return false;
    } // TryAddRef()

    // DelRef(): decrement, acq_rel; true iff this call made the 1 -> 0 transition.
    bool DelRef() noexcept { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const noexcept { return ref_count.load(std::memory_order_relaxed); }
}; // struct PlainIntrusiveCount

// T != U: the interface type T is a plain, non-polymorphic struct with a
// non-virtual destructor; the stored type U puts its hazard pointer base FIRST,
// so the T subobject sits at a nonzero offset and U* -> T* is a real
// adjustment.
struct Payload {
    int value;   // the payload read through T
    explicit Payload(int v) : value(v) {}
}; // struct Payload

struct Carrier : std::hazard_pointer_obj_base<Carrier>, Payload, PlainIntrusiveCount {
    // Counts currently-alive Carrier objects; only ~Carrier decrements it, so
    // it shows that reclamation ran U's destructor.
    static std::atomic<int> live;
    explicit Carrier(int v) : Payload(v) { live.fetch_add(1, std::memory_order_relaxed); }
    ~Carrier() { live.fetch_sub(1, std::memory_order_relaxed); }
}; // struct Carrier

inline std::atomic<int> Carrier::live{0};

// A pointee that names intr_shared_ptr_hp<Link> while it is still incomplete:
// in its own base clause (pointee_base) and in a data member, the shape of a
// lock-free list node. Each Link owns the next one, so releasing the head
// starts a destructor cascade.
struct Link : intr_shared_ptr_hp<Link>::pointee_base, PlainIntrusiveCount {
    // Counts currently-alive Link objects.
    static std::atomic<int> live;
    intr_shared_ptr_hp<Link> next;   // the owning link to the next node
    Link() { live.fetch_add(1, std::memory_order_relaxed); }
    ~Link() { live.fetch_sub(1, std::memory_order_relaxed); }
}; // struct Link

inline std::atomic<int> Link::live{0};

// Pointee-concept partitions: each violates exactly one requirement of
// HpIntrusivePointee. Declarations only (used in unevaluated contexts).
struct MissingTryAddRef : std::hazard_pointer_obj_base<MissingTryAddRef> {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct MissingTryAddRef

struct MissingHpBase {
    void AddRef() noexcept;
    bool TryAddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct MissingHpBase

struct ThrowingAddRef : std::hazard_pointer_obj_base<ThrowingAddRef> {
    void AddRef();
    bool TryAddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct ThrowingAddRef

// Clauses (declared interface): conformance to AtomicSharedPtr and
// MarkedSharedPtr; the pointee concept accepts Data and rejects each single
// violation; the optional policy members (pointee_base, supports_marking,
// deferred_reclamation, compare_exchange_weak); the atomic is non-copyable and
// non-movable, constexpr/noexcept default-constructible and implicitly
// constructible from nullptr and from a value; the value type is a regular
// type with noexcept moves and the documented noexcept accessors; get() yields
// T*, get_raw() U*.
TEST_F(IntrSharedPtrHpTest, InterfaceShape) {
    static_assert(AtomicSharedPtr<HpPtr>);
    static_assert(MarkedSharedPtr<HpValue>);

    static_assert(HpIntrusivePointee<Data>);
    static_assert(HpIntrusivePointee<Carrier>);
    static_assert(!HpIntrusivePointee<MissingTryAddRef>);
    static_assert(!HpIntrusivePointee<MissingHpBase>);
    static_assert(!HpIntrusivePointee<ThrowingAddRef>);

    static_assert(std::is_same_v<HpPtr::pointee_base, std::hazard_pointer_obj_base<Data>>);
    static_assert(HpPtr::supports_marking);
    static_assert(HpPtr::deferred_reclamation);
    static_assert(requires(HpPtr a, HpValue e, const HpValue d) {
        { a.compare_exchange_weak(e, d, std::memory_order_seq_cst, std::memory_order_seq_cst) }
            -> std::same_as<bool>;
    });

    static_assert(!std::is_copy_constructible_v<HpPtr>);
    static_assert(!std::is_copy_assignable_v<HpPtr>);
    static_assert(!std::is_move_constructible_v<HpPtr>);
    static_assert(!std::is_move_assignable_v<HpPtr>);
    static_assert(std::is_nothrow_default_constructible_v<HpPtr>);
    static_assert(std::is_nothrow_constructible_v<HpPtr, std::nullptr_t>);
    static_assert(std::is_convertible_v<std::nullptr_t, HpPtr>);
    static_assert(std::is_convertible_v<HpValue, HpPtr>);
    static_assert(std::is_nothrow_destructible_v<HpPtr>);

    static_assert(std::is_copy_constructible_v<HpValue> && std::is_copy_assignable_v<HpValue>);
    static_assert(std::is_nothrow_move_constructible_v<HpValue>);
    static_assert(std::is_nothrow_move_assignable_v<HpValue>);
    static_assert(std::is_nothrow_destructible_v<HpValue>);
    static_assert(std::is_constructible_v<HpValue, Data*> && !std::is_convertible_v<Data*, HpValue>);
    static_assert(noexcept(std::declval<const HpValue&>().is_marked()));
    static_assert(noexcept(std::declval<const HpValue&>().get_raw()));
    static_assert(noexcept(std::declval<const HpValue&>().get()));
    static_assert(noexcept(static_cast<bool>(std::declval<const HpValue&>())));
    static_assert(!std::is_convertible_v<HpValue, bool>);   // explicit operator bool
    static_assert(std::is_same_v<decltype(std::declval<const HpValue&>().get()), Data*>);

    using CarrierValue = intr_shared_ptr_hp<Payload, Carrier>::shared_ptr_type;
    static_assert(std::is_same_v<decltype(std::declval<const CarrierValue&>().get()), Payload*>);
    static_assert(std::is_same_v<decltype(std::declval<const CarrierValue&>().operator->()), Payload*>);
    static_assert(std::is_same_v<decltype(*std::declval<const CarrierValue&>()), Payload&>);
    static_assert(std::is_same_v<decltype(std::declval<const CarrierValue&>().get_raw()), Carrier*>);

    // The default constructor is constexpr: constant initialization is possible.
    static constinit HpPtr constant_initialized;
    EXPECT_FALSE(constant_initialized.load());
} // InterfaceShape

// Clauses (null values): a default- or nullptr-constructed value is null,
// unmarked, with use_count() 0; set_mark() on null yields the marked null
// (word 1): is_marked(), operator bool false, get() null, use_count() 0, and
// it differs from null; get_unmarked() of the marked null is null.
TEST_F(IntrSharedPtrHpTest, NullAndMarkedNullValues) {
    const HpValue nulls[] = {HpValue(), HpValue(nullptr), HpValue(static_cast<Data*>(nullptr))};
    for (const HpValue& n : nulls) {
        EXPECT_FALSE(n);
        EXPECT_FALSE(n.is_marked());
        EXPECT_EQ(n.get(), nullptr);
        EXPECT_EQ(n.get_raw(), nullptr);
        EXPECT_EQ(n.use_count(), 0);
        EXPECT_TRUE(n == HpValue());
        EXPECT_FALSE(n != HpValue());
    } // loop over the null partitions

    const HpValue marked_null = HpValue().set_mark();
    EXPECT_TRUE(marked_null.is_marked());
    EXPECT_FALSE(marked_null);
    EXPECT_EQ(marked_null.get(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(marked_null.get_raw()), std::uintptr_t{1});
    EXPECT_EQ(marked_null.use_count(), 0);
    EXPECT_FALSE(marked_null == HpValue());
    EXPECT_TRUE(marked_null != HpValue());

    const HpValue unmarked = marked_null.get_unmarked();
    EXPECT_FALSE(unmarked.is_marked());
    EXPECT_TRUE(unmarked == HpValue());
} // NullAndMarkedNullValues

// Clause (shared_ptr_type(U* p)): adopting a fresh object takes its count
// 0 -> 1; adopting an object that already has owners adds one (adoption and
// sharing are the same operation); bit 0 of `p` is kept as the mark, and get()
// is the unmarked pointer. The object is reclaimed once all owners are gone.
TEST_F(IntrSharedPtrHpTest, AdoptionAddsOneOwner) {
    {
        HpValue first(new Data(5));
        EXPECT_EQ(first.use_count(), 1);
        EXPECT_FALSE(first.is_marked());
        EXPECT_EQ(first->value, 5);
        EXPECT_EQ(Data::live_count.load(), base_ + 1);

        HpValue second(first.get());
        EXPECT_EQ(first.use_count(), 2);
        EXPECT_TRUE(second == first);

        HpValue marked(first.set_mark().get_raw());   // the temporary owner keeps the count nonzero
        EXPECT_TRUE(marked.is_marked());
        EXPECT_EQ(marked.get(), first.get());
        EXPECT_EQ(marked.get_raw(), first.set_mark().get_raw());
        EXPECT_EQ(first.use_count(), 3);
    }
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_);
} // AdoptionAddsOneOwner

// Clauses (copy and move construction): a copy equals its source and adds one
// owner; a move steals the source's reference (no count change) and leaves the
// source null and unmarked -- also when the source was the marked null; marks
// travel with the value.
TEST_F(IntrSharedPtrHpTest, CopyAndMoveConstruction) {
    HpValue original(new Data(7));
    HpValue copy(original);
    EXPECT_TRUE(copy == original);
    EXPECT_EQ(original.use_count(), 2);

    HpValue moved(std::move(copy));
    EXPECT_TRUE(moved == original);
    EXPECT_FALSE(copy);
    EXPECT_FALSE(copy.is_marked());
    EXPECT_EQ(original.use_count(), 2);

    HpValue marked = original.set_mark();
    HpValue moved_marked(std::move(marked));
    EXPECT_TRUE(moved_marked.is_marked());
    EXPECT_EQ(moved_marked.get(), original.get());
    EXPECT_FALSE(marked);
    EXPECT_FALSE(marked.is_marked());
    EXPECT_EQ(original.use_count(), 3);

    HpValue marked_null = HpValue().set_mark();
    HpValue moved_marked_null(std::move(marked_null));
    EXPECT_TRUE(moved_marked_null.is_marked());
    EXPECT_FALSE(moved_marked_null);
    EXPECT_FALSE(marked_null.is_marked());
    EXPECT_TRUE(marked_null == HpValue());
} // CopyAndMoveConstruction

// Clauses (copy and move assignment): copy assignment makes *this == x and
// moves one count from the old pointee to the new one; assigning a value that
// holds the same pointee with the other mark keeps the count; self-assignment
// (copy or move) is a no-op; move assignment releases the old pointee, takes
// x's word and nulls x. Exact counts in one thread; the released object is
// gone after a drain.
TEST_F(IntrSharedPtrHpTest, CopyAndMoveAssignment) {
    HpValue a(new Data(1));
    HpValue b(new Data(2));
    HpValue target;
    target = a;
    EXPECT_TRUE(target == a);
    EXPECT_EQ(a.use_count(), 2);

    target = b;
    EXPECT_TRUE(target == b);
    EXPECT_EQ(a.use_count(), 1);
    EXPECT_EQ(b.use_count(), 2);

    HpValue& self = target;
    target = self;                                 // self copy-assignment
    EXPECT_TRUE(target == b);
    EXPECT_EQ(b.use_count(), 2);

    const HpValue marked_b = b.set_mark();         // b: 3
    target = marked_b;                             // same pointee, other mark
    EXPECT_TRUE(target == marked_b);
    EXPECT_TRUE(target.is_marked());
    EXPECT_EQ(b.use_count(), 3);
    target = b;
    EXPECT_TRUE(target == b);
    EXPECT_FALSE(target.is_marked());
    EXPECT_EQ(b.use_count(), 3);

    HpValue source(new Data(3));
    target = std::move(source);                    // releases target's reference to b
    EXPECT_FALSE(source);
    EXPECT_EQ(target->value, 3);
    EXPECT_EQ(target.use_count(), 1);
    EXPECT_EQ(b.use_count(), 2);

    target = std::move(self);                      // self move-assignment
    EXPECT_EQ(target->value, 3);
    EXPECT_EQ(target.use_count(), 1);

    a = HpValue();                                 // Data(1)'s last reference
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_ + 2);
} // CopyAndMoveAssignment

// Clause (the mark is part of the value): a value and its mark twin compare
// unequal but share get(); get_raw() is the word's bit pattern (get() | 1 when
// marked, get() when not); a marked handle dereferences like its unmarked twin;
// set_mark() of a marked value stays marked; get_unmarked() of the twin equals
// the original.
TEST_F(IntrSharedPtrHpTest, MarkIsPartOfIdentity) {
    HpValue p(new Data(9));
    const HpValue m = p.set_mark();
    EXPECT_FALSE(p == m);
    EXPECT_TRUE(p != m);
    EXPECT_EQ(m.get(), p.get());
    EXPECT_EQ(p.get_raw(), p.get());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(m.get_raw()),
              reinterpret_cast<std::uintptr_t>(p.get()) | std::uintptr_t{1});
    EXPECT_TRUE(m);
    EXPECT_EQ(m->value, 9);
    EXPECT_EQ((*m).value, 9);
    EXPECT_EQ(&*m, p.get());
    EXPECT_TRUE(m.set_mark() == m);
    EXPECT_TRUE(m.get_unmarked() == p);

    HpValue q(new Data(9));                        // equal payload, different object
    EXPECT_FALSE(p == q);
    EXPECT_FALSE(m == q.set_mark());
} // MarkIsPartOfIdentity

// Clause (get_unmarked / set_mark): the lvalue get_unmarked() and set_mark()
// return a new owner (count + 1) and leave the source as it was; the rvalue
// get_unmarked() hands the source's reference to the result with no count
// change -- once the source is destroyed, the result and the other owners are
// still exactly counted (a result that did not own a reference, or a source
// that kept and later released one, would leave the count one short).
TEST_F(IntrSharedPtrHpTest, GetUnmarkedAndSetMarkCounts) {
    HpValue p(new Data(4));
    HpValue m = p.set_mark();
    EXPECT_EQ(p.use_count(), 2);
    EXPECT_FALSE(p.is_marked());                   // source unchanged
    {
        HpValue u = m.get_unmarked();
        EXPECT_FALSE(u.is_marked());
        EXPECT_EQ(u.get(), p.get());
        EXPECT_EQ(p.use_count(), 3);
        EXPECT_TRUE(m.is_marked());                // source unchanged
    }
    EXPECT_EQ(p.use_count(), 2);

    HpValue result;
    {
        HpValue source = p.set_mark();             // p: 3
        result = std::move(source).get_unmarked();
        EXPECT_FALSE(result.is_marked());
        EXPECT_EQ(result.get(), p.get());
        EXPECT_EQ(p.use_count(), 3);               // p, m, result: no count traffic
    } // `source` is destroyed here, whatever state the call left it in
    EXPECT_EQ(p.use_count(), 3);
    EXPECT_FALSE(HpValue().get_unmarked());
    EXPECT_FALSE(HpValue().get_unmarked().is_marked());
} // GetUnmarkedAndSetMarkCounts

// Clauses (atomic construction and destruction): constructing from a value
// counts one reference for the word while the argument keeps its own; a marked
// value is kept as such; a moved-in value transfers its reference; the
// destructor releases the word's reference, and an object whose last owner was
// the word is reclaimed; default and nullptr construction hold null; both
// conversions are implicit.
TEST_F(IntrSharedPtrHpTest, AtomicConstructionAndDestructionCountTheWord) {
    HpValue v(new Data(11));
    {
        HpPtr atomic(v);
        EXPECT_EQ(v.use_count(), 2);
        const HpValue loaded = atomic.load();
        EXPECT_TRUE(loaded == v);
        EXPECT_EQ(v.use_count(), 3);
    }
    EXPECT_EQ(v.use_count(), 1);
    {
        HpPtr atomic(v.set_mark());
        const HpValue loaded = atomic.load();
        EXPECT_TRUE(loaded.is_marked());
        EXPECT_EQ(loaded.get(), v.get());
        EXPECT_EQ(v.use_count(), 3);               // v, word, loaded
    }
    EXPECT_EQ(v.use_count(), 1);
    {
        HpPtr atomic(std::move(v));
        EXPECT_FALSE(v);
        EXPECT_EQ(atomic.load().use_count(), 2);   // word and the loaded temporary
    } // the word held the last reference
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_);

    HpPtr empty_default;
    HpPtr empty_nullptr(nullptr);
    EXPECT_FALSE(empty_default.load());
    EXPECT_FALSE(empty_default.load().is_marked());
    EXPECT_FALSE(empty_nullptr.load());
    EXPECT_FALSE(empty_nullptr.load().is_marked());

    HpPtr implicit_null = nullptr;
    HpPtr implicit_value = HpValue(new Data(12));
    EXPECT_FALSE(implicit_null.load());
    EXPECT_EQ(implicit_value.load()->value, 12);
} // AtomicConstructionAndDestructionCountTheWord

// Clause (load()): returns a value equal (pointer and mark) to the one the
// atomic holds, owning one reference of its own while non-null; null and the
// marked null are returned as such.
TEST_F(IntrSharedPtrHpTest, LoadReturnsHeldWordWithItsOwnReference) {
    HpValue v(new Data(21));
    HpPtr atomic(v);
    {
        const HpValue loaded = atomic.load();
        EXPECT_TRUE(loaded == v);
        EXPECT_EQ(loaded->value, 21);
        EXPECT_EQ(v.use_count(), 3);
    }
    EXPECT_EQ(v.use_count(), 2);

    atomic.store(v.set_mark());
    {
        const HpValue loaded = atomic.load();
        EXPECT_TRUE(loaded.is_marked());
        EXPECT_TRUE(loaded == v.set_mark());
        EXPECT_EQ(loaded->value, 21);
        EXPECT_EQ(v.use_count(), 3);
    }

    atomic.store(HpValue());
    EXPECT_EQ(v.use_count(), 1);
    EXPECT_FALSE(atomic.load());
    EXPECT_FALSE(atomic.load().is_marked());

    atomic.store(HpValue().set_mark());
    const HpValue loaded_marked_null = atomic.load();
    EXPECT_TRUE(loaded_marked_null.is_marked());
    EXPECT_FALSE(loaded_marked_null);
    EXPECT_EQ(loaded_marked_null.use_count(), 0);
} // LoadReturnsHeldWordWithItsOwnReference

// Clause (store()): replaces the held word by `desired`'s (pointer and mark);
// the word's reference to the new pointee is counted while `desired` keeps its
// own (or hands it over when moved in); the old pointee's count drops by one.
// Storing the value already held keeps the count; storing over the last
// reference reclaims the old object (after a drain).
TEST_F(IntrSharedPtrHpTest, StoreReplacesWordAndReleasesOld) {
    HpValue first(new Data(1));
    HpValue second(new Data(2));
    HpPtr atomic(first);
    atomic.store(second);
    EXPECT_TRUE(atomic.load() == second);
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(second.use_count(), 2);

    atomic.store(second);                          // the value already held
    EXPECT_EQ(second.use_count(), 2);

    atomic.store(second.set_mark());               // its mark twin
    EXPECT_TRUE(atomic.load() == second.set_mark());
    EXPECT_EQ(second.use_count(), 2);

    atomic.store(std::move(first));                // the word becomes Data(1)'s only owner
    EXPECT_FALSE(first);
    EXPECT_EQ(atomic.load()->value, 1);
    EXPECT_EQ(atomic.load().use_count(), 2);       // word and the loaded temporary
    EXPECT_EQ(second.use_count(), 1);

    atomic.store(HpValue().set_mark());            // releases Data(1)'s last reference
    EXPECT_TRUE(atomic.load().is_marked());
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_ + 1);
} // StoreReplacesWordAndReleasesOld

// Clause (both CASes compare the full word; the marked null is a value
// distinct from null): null vs marked null fails in both directions with
// `expected` refreshed to the word's mark state; the refreshed value then
// succeeds.
TEST_F(IntrSharedPtrHpTest, CompareExchangeDistinguishesNullFromMarkedNull) {
    HpValue desired(new Data(1));
    for (bool weak : {false, true}) {
        // One call of the CAS under test (a single weak attempt may fail spuriously).
        auto cas = [weak](HpPtr& atomic, HpValue& expected, const HpValue& d) {
            if (weak) return atomic.compare_exchange_weak(expected, d);
            return atomic.compare_exchange_strong(expected, d);
        };
        HpPtr marked_word(HpValue().set_mark());
        HpValue expected;                          // unmarked null
        EXPECT_FALSE(cas(marked_word, expected, desired)) << "weak = " << weak;
        EXPECT_TRUE(expected.is_marked());
        EXPECT_FALSE(expected);
        EXPECT_TRUE(marked_word.load().is_marked());
        if (weak) {
            EXPECT_TRUE(weak_cas_with_retries(marked_word, expected, desired));
        } else {
            EXPECT_TRUE(marked_word.compare_exchange_strong(expected, desired));
        }
        EXPECT_TRUE(marked_word.load() == desired);

        HpPtr null_word;
        HpValue marked_expected = HpValue().set_mark();
        EXPECT_FALSE(cas(null_word, marked_expected, desired)) << "weak = " << weak;
        EXPECT_FALSE(marked_expected.is_marked());
        EXPECT_TRUE(marked_expected == HpValue());
        EXPECT_FALSE(null_word.load());
    } // loop over strong and weak
    EXPECT_EQ(desired.use_count(), 1);             // the words of both rounds are gone
} // CompareExchangeDistinguishesNullFromMarkedNull

// Clauses (compare_exchange_weak): success exactly as the strong CAS's (the
// atomic holds `desired`; `expected` untouched; counts transferred); a
// matching `expected` succeeds within a bounded retry loop, the way std code
// uses a weak CAS, and every spurious failure along the way refreshes
// `expected` to the (unchanged) current value; a stale `expected` fails, the
// atomic is unchanged, `expected` is refreshed with a live reference and
// `desired` keeps its count; a successful swap releases the old value (gone
// after a drain).
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakValueSemantics) {
    HpValue current(new Data(1));
    HpValue next(new Data(2));
    HpPtr atomic(current);

    HpValue expected = current;
    ASSERT_TRUE(weak_cas_with_retries(atomic, expected, next));
    EXPECT_TRUE(atomic.load() == next);
    EXPECT_TRUE(expected == current);              // untouched on success
    EXPECT_EQ(current.use_count(), 2);             // current, expected; the word's reference released
    EXPECT_EQ(next.use_count(), 2);                // next, word

    HpValue stale = current;
    HpValue desired(new Data(3));
    EXPECT_FALSE(atomic.compare_exchange_weak(stale, desired));
    EXPECT_TRUE(atomic.load() == next);
    EXPECT_TRUE(stale == next);
    EXPECT_EQ(stale->value, 2);
    EXPECT_EQ(next.use_count(), 3);                // next, word, stale
    EXPECT_EQ(current.use_count(), 2);             // current, expected
    EXPECT_EQ(desired.use_count(), 1);

    // The word is the only owner of `next`'s object once the handles go; a
    // successful weak swap must release it.
    stale = HpValue();
    HpValue swap_expected = std::move(next);
    ASSERT_TRUE(weak_cas_with_retries(atomic, swap_expected, desired));
    EXPECT_TRUE(atomic.load() == desired);
    swap_expected = HpValue();                     // Data(2)'s last reference
    EXPECT_GE(Data::live_count.load(), base_ + 2); // still alive or pending: no assertion of either
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_ + 2); // Data(1) and Data(3)
} // CompareExchangeWeakValueSemantics

// Clause (compare_exchange_weak compares the full word, marks included): as
// the strong test, both directions.
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakComparesMarkBit) {
    HpValue plain(new Data(5));
    const HpValue marked = plain.set_mark();
    HpValue other(new Data(6));
    HpPtr atomic(marked);

    HpValue expected = plain;
    EXPECT_FALSE(atomic.compare_exchange_weak(expected, other));
    EXPECT_TRUE(atomic.load() == marked);
    EXPECT_TRUE(expected == marked);
    ASSERT_TRUE(weak_cas_with_retries(atomic, expected, other));
    EXPECT_TRUE(atomic.load() == other);

    HpValue marked_expected = other.set_mark();
    EXPECT_FALSE(atomic.compare_exchange_weak(marked_expected, plain));
    EXPECT_TRUE(marked_expected == other);
    EXPECT_TRUE(atomic.load() == other);
} // CompareExchangeWeakComparesMarkBit

// Clause (compare_exchange_weak aliasing: as the strong CAS, counts balance):
// the success, shared-failure and only-owner-failure partitions of the strong
// aliasing test.
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakAliasedExpectedDesired) {
    HpValue x(new Data(1));
    HpValue y(new Data(2));
    HpPtr atomic(x);
    {
        HpValue e = x;
        ASSERT_TRUE(weak_cas_with_retries(atomic, e, e));
        EXPECT_TRUE(atomic.load() == x);
        EXPECT_TRUE(e == x);
        EXPECT_EQ(x.use_count(), 3);
    }
    EXPECT_EQ(x.use_count(), 2);

    atomic.store(y);
    {
        HpValue e = x;
        EXPECT_FALSE(atomic.compare_exchange_weak(e, e));
        EXPECT_TRUE(e == y);
        EXPECT_EQ(x.use_count(), 1);
        EXPECT_EQ(y.use_count(), 3);
    }
    EXPECT_EQ(y.use_count(), 2);

    {
        HpValue e(new Data(3));
        EXPECT_FALSE(atomic.compare_exchange_weak(e, e));
        EXPECT_TRUE(e == y);
    }
    EXPECT_EQ(y.use_count(), 2);
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_ + 2);
} // CompareExchangeWeakAliasedExpectedDesired

// Clause (compare_exchange_weak orders: as the strong CAS). With nothing else
// touching the word, a matching CAS succeeds within the retry bound and a stale
// one fails with a refreshed `expected`, for every permitted combination.
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakAcceptsPermittedOrders) {
    constexpr std::memory_order success_orders[] = {
        std::memory_order_relaxed, std::memory_order_acquire, std::memory_order_release,
        std::memory_order_acq_rel, std::memory_order_seq_cst};
    constexpr std::memory_order failure_orders[] = {
        std::memory_order_relaxed, std::memory_order_acquire, std::memory_order_seq_cst};
    HpValue first(new Data(1));
    HpValue second(new Data(2));
    for (std::memory_order success : success_orders) {
        for (std::memory_order failure : failure_orders) {
            HpPtr atomic(first);
            HpValue expected = first;
            EXPECT_TRUE(weak_cas_with_retries(atomic, expected, second, success, failure));
            EXPECT_TRUE(atomic.load() == second);
            HpValue stale = first;
            EXPECT_FALSE(atomic.compare_exchange_weak(stale, first, success, failure));
            EXPECT_TRUE(stale == second);
        } // loop over failure orders
    } // loop over success orders
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(second.use_count(), 1);
} // CompareExchangeWeakAcceptsPermittedOrders

// Clause (compare_exchange_weak MAY fail spuriously, with the refreshed
// `expected` equal to the original; otherwise as the strong CAS): in the
// changed-back race, false returns with `expected == original` are allowed and
// not asserted on. What must hold: every refresh is one of the two values the
// word ever held (a live reference to a current value), and all counts balance
// afterwards -- exactly one owner each, and nothing leaked after a drain.
// Both the distinct-object and the mark-twin variants.
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakChangedBackKeepsCountsBalanced) {
    HpValue a(new Data(1));
    HpValue b(new Data(2));
    const ChangedBackOutcome distinct =
        run_changed_back<HpPtr>(a, b, WeakCas{}, changed_back_cas_count);
    EXPECT_EQ(distinct.failures_foreign, 0);
    EXPECT_EQ(distinct.successes + distinct.failures, changed_back_cas_count);
    EXPECT_EQ(a.use_count(), 1);
    EXPECT_EQ(b.use_count(), 1);

    const HpValue marked_a = a.set_mark();
    const ChangedBackOutcome twin =
        run_changed_back<HpPtr>(marked_a, a, WeakCas{}, changed_back_cas_count);
    EXPECT_EQ(twin.failures_foreign, 0);
    EXPECT_EQ(a.use_count(), 2);                   // a and marked_a

    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_ + 2);
} // CompareExchangeWeakChangedBackKeepsCountsBalanced

// Clauses (compare_exchange_weak in the documented loop
// "while (!cas_weak(expected, desired)) { re-derive desired }" -- failure
// refreshes `expected` with the current value, success as strong): the
// four-thread counter of the strong test must end at exactly the number of
// increments, with no read of a reclaimed object.
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakLoopCountsEveryIncrement) {
    constexpr int num_threads = 4;
    constexpr int increments_per_thread = 2500;
    const CounterOutcome outcome =
        run_cas_counter<HpPtr>(WeakCas{}, num_threads, increments_per_thread);
    EXPECT_EQ(outcome.final_value, num_threads*increments_per_thread);
    EXPECT_EQ(outcome.poisoned_reads, 0);
} // CompareExchangeWeakLoopCountsEveryIncrement

// Clause (compare_exchange_weak aliasing under contention: counts balance):
// as the strong contention test.
TEST_F(IntrSharedPtrHpTest, CompareExchangeWeakAliasBalanceUnderContention) {
    EXPECT_EQ(run_alias_contention<HpPtr>(WeakCas{}, 4, 5000), 0);
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_);
}

// Clause (deferred reclamation): a pointee whose count reaches 0 is NOT
// deleted; it is retired and destroyed at a later scan, and a scan starts only
// inside a retire() that crosses the threshold (at least 1000 pending). After
// the fixture's drain nothing is pending, so one release -- of a value, or of
// an atomic's word -- leaves the object alive until the next drain.
// Fails if the last release destroys the object synchronously (live count
// already back at base) or never makes it reclaimable (still there after the
// drain).
TEST_F(IntrSharedPtrHpTest, ReleaseDefersDestructionUntilScan) {
    {
        HpValue v(new Data(1));
    } // last owner released
    EXPECT_EQ(Data::live_count.load(), base_ + 1);
    {
        HpPtr atomic(HpValue(new Data(2)));
    } // the word was the last owner
    EXPECT_EQ(Data::live_count.load(), base_ + 2);
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_);
} // ReleaseDefersDestructionUntilScan

// Clause (deferred reclamation runs "on WHICHEVER thread made that retire --
// possibly a thread that never touched this pointer"; ~T must not assume the
// releasing thread): an object may be destroyed on a thread other than the one
// that released it, and that must stay correct.
//   Direction 1: a thread takes a reference with load(), the word lets go, and
//   the thread's handle -- now the last owner -- dies with the thread. After
//   the join the object is still alive (no scan ran: the contract places scans
//   only inside a threshold-crossing retire), and the main thread's drain
//   destroys it, once.
//   Direction 2: the main thread releases the last reference; a thread that
//   never touched any pointer runs the drain and destroys it, once.
// Fails if the object is destroyed twice or never (live count off by one
// after the drain), if destruction needs the releasing thread (the exited
// thread cannot run it), or under ASan if it touches freed memory.
TEST_F(IntrSharedPtrHpTest, PointeeReleasedOnOneThreadReclaimedOnAnother) {
    HpPtr atomic(HpValue(new Data(1)));
    std::thread releaser([&atomic] {
        HpValue mine = atomic.load();
        atomic.store(HpValue());                   // `mine` is now the only owner
        EXPECT_EQ(mine->value, 1);
    }); // `mine` dies here: the last release happens on this thread
    releaser.join();
    EXPECT_EQ(Data::live_count.load(), base_ + 1);
    drain_reclamation();
    EXPECT_EQ(Data::live_count.load(), base_);

    {
        HpValue v(new Data(2));
    } // last release on the main thread
    EXPECT_EQ(Data::live_count.load(), base_ + 1);
    std::thread reclaimer([] { drain_reclamation(); });
    reclaimer.join();
    EXPECT_EQ(Data::live_count.load(), base_);
} // PointeeReleasedOnOneThreadReclaimedOnAnother

// Clauses (T != U): operator*, operator-> and get() return T, converted from
// the stored U* (a real pointer adjustment here); get_raw() is the U*; the word
// stores U* with the mark; reclamation runs U's destructor -- `delete
// static_cast<U*>(p)` -- even though T's destructor is not virtual. Exercised
// through construction, load, store of a marked value and a CAS.
// Fails if get() returns the U* unadjusted (wrong payload), or if a reclaimed
// object is not destroyed as a U (Carrier::live stays up after the drain).
TEST_F(IntrSharedPtrHpTest, InterfaceTypeDiffersFromStoredType) {
    using CarrierPtr = intr_shared_ptr_hp<Payload, Carrier>;
    using CarrierValue = CarrierPtr::shared_ptr_type;
    const int carriers_before = Carrier::live.load();
    {
        CarrierValue v(new Carrier(31));
        Payload* const as_t = v.get();
        ASSERT_NE(static_cast<void*>(as_t), static_cast<void*>(v.get_raw()))
            << "test premise: T sits at a nonzero offset in U";
        EXPECT_EQ(as_t, static_cast<Payload*>(v.get_raw()));
        EXPECT_EQ(as_t->value, 31);
        EXPECT_EQ(v->value, 31);
        EXPECT_EQ((*v).value, 31);

        CarrierPtr atomic(v);
        EXPECT_TRUE(atomic.load() == v);
        EXPECT_EQ(atomic.load()->value, 31);

        atomic.store(v.set_mark());
        const CarrierValue loaded = atomic.load();
        EXPECT_TRUE(loaded.is_marked());
        EXPECT_EQ(loaded.get(), as_t);
        EXPECT_EQ(loaded->value, 31);

        CarrierValue expected = v.set_mark();
        const CarrierValue desired(new Carrier(32));
        EXPECT_TRUE(atomic.compare_exchange_strong(expected, desired));
        EXPECT_EQ(atomic.load()->value, 32);
        EXPECT_EQ(Carrier::live.load(), carriers_before + 2);
    }
    drain_reclamation();
    EXPECT_EQ(Carrier::live.load(), carriers_before);
} // InterfaceTypeDiffersFromStoredType

// Clauses (no class-level constraint on U: intr_shared_ptr_hp<U> and its
// pointee_base can be named while U is incomplete; deferred reclamation (c): a
// release inside a destructor run by a scan is reclaimed at the NEXT scan, and
// one drain reclaims one layer of a destructor cascade). A chain head -> L1 ->
// L2 -> L3 -> L4 in which each Link owns the next: releasing the head retires
// L1, and each drain destroys exactly one more layer.
// Fails if the self-referential pointee does not compile, if a destructor
// chain releases a link twice or not at all (live count after the drains), or
// if a layer is not reclaimed within one drain of its release.
TEST_F(IntrSharedPtrHpTest, SelfReferentialPointeeDestructorChain) {
    using LinkPtr = intr_shared_ptr_hp<Link>;
    using LinkValue = LinkPtr::shared_ptr_type;
    constexpr int depth = 4;
    const int links_before = Link::live.load();
    {
        LinkPtr head;
        {
            LinkValue tail;
            for (int i = 0; i < depth; ++i) {
                LinkValue node(new Link);
                node->next.store(tail);
                tail = node;
            } // loop building the chain back to front
            head.store(tail);
        } // only the links own the chain below the head now
        EXPECT_EQ(Link::live.load(), links_before + depth);
    } // head released: L1 retired
    EXPECT_EQ(Link::live.load(), links_before + depth);
    for (int layer = 1; layer <= depth; ++layer) {
        drain_reclamation();
        EXPECT_EQ(Link::live.load(), links_before + depth - layer) << "after drain " << layer;
    } // loop over cascade layers
} // SelfReferentialPointeeDestructorChain

#ifndef NDEBUG
// Clauses (order preconditions, asserted): load() requires relaxed, consume,
// acquire or seq_cst; store() requires relaxed, release or seq_cst; the
// failure order of both CASes requires relaxed, consume, acquire or seq_cst.
// A violation is an assert, so these are death tests and exist only in a
// build without NDEBUG. Each CAS is made to take its failure path (stale
// `expected`), so the assertion is required to fire wherever it is checked.
// The matcher requires the C library's assertion message: a death by any
// other cause (a crash, an abort elsewhere) does not pass.
using IntrSharedPtrHpDeathTest = IntrSharedPtrHpTest;

TEST_F(IntrSharedPtrHpDeathTest, LoadRejectsReleaseAndAcqRel) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    HpPtr atomic(HpValue(new Data(1)));
    EXPECT_DEATH((void)atomic.load(std::memory_order_release), "Assertion");
    EXPECT_DEATH((void)atomic.load(std::memory_order_acq_rel), "Assertion");
}

TEST_F(IntrSharedPtrHpDeathTest, StoreRejectsAcquireAndAcqRel) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    HpPtr atomic;
    const HpValue v(new Data(1));
    EXPECT_DEATH(atomic.store(v, std::memory_order_acquire), "Assertion");
    EXPECT_DEATH(atomic.store(v, std::memory_order_acq_rel), "Assertion");
}

TEST_F(IntrSharedPtrHpDeathTest, CompareExchangeRejectsReleaseFailureOrders) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const HpValue current(new Data(1));
    const HpValue stale(new Data(2));
    HpPtr atomic(current);
    for (std::memory_order failure : {std::memory_order_release, std::memory_order_acq_rel}) {
        EXPECT_DEATH({
            HpValue expected = stale;
            atomic.compare_exchange_strong(expected, current, std::memory_order_seq_cst, failure);
        }, "Assertion");
        EXPECT_DEATH({
            HpValue expected = stale;
            (void)atomic.compare_exchange_weak(expected, current, std::memory_order_seq_cst, failure);
        }, "Assertion");
    } // loop over forbidden failure orders
} // CompareExchangeRejectsReleaseFailureOrders
#endif // NDEBUG
