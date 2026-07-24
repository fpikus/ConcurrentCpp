/*
 * atomic_shared_ptr_bm.C
 *
 * This benchmark measures the absolute raw, unadulterated concurrency throughput of
 * different atomic shared pointer implementations by stripping away data structure scaffolding.
 *
 * It uses a `benchmark::Fixture` with Google Benchmark's native `ThreadRange` barrier
 * to ensure threads start simultaneously, avoiding OS spawn latency serialization.
 *
 * Performance Hierarchy (16 Threads):
 * 1. IntrShared (Fastest): ~17-32 Million ops/sec. Direct intrusive ref counting bypasses control blocks and hazard records.
 * 2. Parlay Hazard Ptr (Fast): ~14-30 Million ops/sec. Thread-local hazard records avoid control blocks but add slight store/fence overhead.
 * 3. StdAtomic (Slowest): ~2-3 Million ops/sec. Suffers massive cache line bouncing on external control block locks/atomics.
 */
#include <benchmark/benchmark.h>
#include <thread>
#include <vector>
#include <random>
#include <unistd.h>

#include "atomic_shared_ptr_concept.h"
#include "intr_shared_ptr.h"
#include "lock_free_shared_ptr/atomic_shared_ptr.hpp"

static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

struct Data {
    int value;
    std::atomic<int> ref_count{0};
    Data(int v) : value(v) {}
    void AddRef() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    bool DelRef() { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const { return ref_count.load(std::memory_order_relaxed); }
};

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
class AtomicPtrFixture : public benchmark::Fixture {
public:
    static PtrType* ptr;

    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            ptr = new PtrType(make_shared_data<PtrType>(0));
        }
    }

    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            delete ptr;
            ptr = nullptr;
        }
    }
};

template <typename PtrType>
PtrType* AtomicPtrFixture<PtrType>::ptr = nullptr;

#define DEFINE_BM_READHEAVY(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(AtomicPtrFixture, Name, PtrType)(benchmark::State& state) { \
        std::mt19937 rng(state.thread_index() + 42 + state.iterations()); \
        std::uniform_int_distribution<int> dist(0, 99); \
        for (auto _ : state) { \
            if (dist(rng) < 90) { \
                auto p = ptr->load(); \
                benchmark::DoNotOptimize(p); \
            } else { \
                auto expected = ptr->load(); \
                if (expected) { \
                    auto new_p = make_shared_data<PtrType>(expected->value + 1); \
                    ptr->compare_exchange_strong(expected, new_p); \
                } \
            } \
        } \
        state.SetItemsProcessed(state.iterations()); \
    } \
    BENCHMARK_REGISTER_F(AtomicPtrFixture, Name)->ThreadRange(1, num_cpu)->UseRealTime();

DEFINE_BM_READHEAVY(ReadHeavy_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_READHEAVY(ReadHeavy_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_READHEAVY(ReadHeavy_LockFree, parlay::atomic_shared_ptr<Data>)

#define DEFINE_BM_WRITEHEAVY(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(AtomicPtrFixture, Name, PtrType)(benchmark::State& state) { \
        std::mt19937 rng(state.thread_index() + 42 + state.iterations()); \
        std::uniform_int_distribution<int> dist(0, 99); \
        for (auto _ : state) { \
            if (dist(rng) < 10) { \
                auto p = ptr->load(); \
                benchmark::DoNotOptimize(p); \
            } else { \
                auto expected = ptr->load(); \
                if (expected) { \
                    auto new_p = make_shared_data<PtrType>(expected->value + 1); \
                    ptr->compare_exchange_strong(expected, new_p); \
                } \
            } \
        } \
        state.SetItemsProcessed(state.iterations()); \
    } \
    BENCHMARK_REGISTER_F(AtomicPtrFixture, Name)->ThreadRange(1, num_cpu)->UseRealTime();

DEFINE_BM_WRITEHEAVY(WriteHeavy_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_WRITEHEAVY(WriteHeavy_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_WRITEHEAVY(WriteHeavy_LockFree, parlay::atomic_shared_ptr<Data>)

#define DEFINE_BM_HIGHCONTENTION(Name, PtrType) \
    BENCHMARK_TEMPLATE_DEFINE_F(AtomicPtrFixture, Name, PtrType)(benchmark::State& state) { \
        for (auto _ : state) { \
            auto expected = ptr->load(); \
            if (expected) { \
                auto new_p = make_shared_data<PtrType>(expected->value + 1); \
                ptr->compare_exchange_strong(expected, new_p); \
            } \
        } \
        state.SetItemsProcessed(state.iterations()); \
    } \
    BENCHMARK_REGISTER_F(AtomicPtrFixture, Name)->Threads(16)->UseRealTime();

DEFINE_BM_HIGHCONTENTION(HighContention_StdAtomic, StdAtomicSharedPtrAdapter<Data>)
DEFINE_BM_HIGHCONTENTION(HighContention_IntrShared, intr_shared_ptr<Data>)
DEFINE_BM_HIGHCONTENTION(HighContention_LockFree, parlay::atomic_shared_ptr<Data>)

BENCHMARK_MAIN();
