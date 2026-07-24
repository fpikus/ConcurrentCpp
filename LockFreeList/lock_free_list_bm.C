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
#include <benchmark/benchmark.h>
#include <thread>
#include <vector>
#include <random>
#include <unistd.h>

// Benchmark Performance Summary:
// - IntrPtr: The fastest and lowest-overhead for insertions, deletions, and 
//   graveyard traversal (~25 ns).
// - StdAtomic (std::atomic<std::shared_ptr>): Provided strong baseline speeds 
//   (~53 ns for writes, ~28 ns for graveyard) thanks to libstdc++ improvements.
// - HazardPtr (parlay): Slower than the other two due to the overhead of 
//   Hazard Pointer allocations and control block traversals (~100 ns writes, ~50 ns graveyard).

#include "atomic_shared_ptr_concept.h"
#include "intr_shared_ptr.h"
#include "lock_free_shared_ptr/atomic_shared_ptr.hpp"
#include "lock_free_list.h"

// Wrappers for the benchmark fixtures (same trick as in lock_free_list_test.C):
// BENCHMARK_TEMPLATE_DEFINE_F takes a *type* argument, but LockFreeList is
// parameterized by a template-template argument, so each wrapper carries its
// pointer template as a member alias (Wrapper::template ptr_type).
struct StdAtomicWrapper {
    template <typename U> using ptr_type = StdAtomicSharedPtrAdapter<U>;
};
struct IntrPtrWrapper {
    template <typename U> using ptr_type = intr_shared_ptr<U, U>;
};
struct ParlayWrapper {
    template <typename U> using ptr_type = parlay::atomic_shared_ptr<U>;
};

// Uniform node factory (see lock_free_list_test.C): hides the three different
// shared_ptr_type construction idioms behind one compile-time-dispatched callable.
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
class ListFixture : public benchmark::Fixture {
public:
    using List = LockFreeList<int, Wrapper::template ptr_type>;
    using Node = typename List::Node;
    
    // The list under test is shared by all benchmark threads, but Google
    // Benchmark constructs a *separate fixture instance per thread* -- hence
    // static members: allocated by thread 0 in SetUp(), freed in TearDown(),
    // with the state-loop barrier ordering both against the other threads.
    static List* list;
    static Factory<Wrapper> factory;

    void SetUp(const ::benchmark::State& state) override {
        // A generic 1000 items is good enough for the head-anchored benchmarks
        // (ReadHeavy, WriteHeavy, Graveyard, MassiveHeadInsert); fixtures that
        // need a different prepopulation override SetUp and call Prepopulate()
        // with their own count (see DispersedListFixture).
        if (state.thread_index() == 0) {
            Prepopulate(1000);
        }
    } // SetUp()

    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            delete list;
            list = nullptr;
        }
    } // TearDown()

protected:
    // Allocates the shared list (with its dummy head) and fills it with `count`
    // nodes. Must be called by exactly one thread, before the others touch the
    // list; the state-loop barrier provides that ordering.
    static void Prepopulate(int count) {
        list = new List(factory.template operator()<Node>());
        for (int i = 0; i < count; ++i) {
            list->insert_after(list->before_begin(), factory.template operator()<Node>(i));
        }
    } // Prepopulate()
}; // ListFixture

template <typename Wrapper>
typename ListFixture<Wrapper>::List* ListFixture<Wrapper>::list = nullptr;

template <typename Wrapper>
Factory<Wrapper> ListFixture<Wrapper>::factory;

// Fixture for the dispersed-read benchmarks. Reuses ListFixture's list and
// factory (benchmarks run sequentially, so sharing the statics is safe) but
// prepopulates a list two orders of magnitude larger and precomputes evenly
// spaced starting positions, one per thread. Both together are what makes the
// workload genuinely dispersed: threads begin far apart and advance at roughly
// equal rates over a list ~60x larger than all their read windows combined, so
// they rarely traverse the same nodes at the same time -- rarely, not never,
// which is the point of benchmarking a thread-safe list this way.
template <typename Wrapper>
class DispersedListFixture : public ListFixture<Wrapper> {
public:
    using Base = ListFixture<Wrapper>;
    using Iterator = typename Base::List::iterator;

    // 32 threads x 50-node windows cover only 1.6% of 100k nodes, so window
    // collisions are rare; the footprint (several MB of nodes) also pushes
    // reads out of L1/L2, as a dispersed workload should.
    static constexpr int list_size = 100'000;

    // One starting iterator per thread, list_size/threads apart. Written by
    // thread 0 in SetUp, read by each thread on its first pass through the
    // state loop; the loop's start barrier orders the two. The iterators hold
    // strong references, so TearDown clears them before deleting the list.
    static std::vector<Iterator> start_positions;

    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            this->Prepopulate(list_size);
            start_positions.clear();
            Iterator it = Base::list->begin();
            const int stride = list_size/state.threads();
            for (int t = 0; t < state.threads(); ++t) {
                start_positions.push_back(it);
                if (t + 1 < state.threads()) {
                    for (int i = 0; i < stride; ++i) {
                        ++it;
                    }
                } // skip the walk after the last position
            } // one start position per thread
        }
    } // SetUp()

    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) {
            start_positions.clear(); // drop the strong node refs before the list is destroyed
        }
        Base::TearDown(state);
    } // TearDown()
}; // DispersedListFixture

template <typename Wrapper>
std::vector<typename DispersedListFixture<Wrapper>::Iterator> DispersedListFixture<Wrapper>::start_positions;

// Deterministic per-thread seed: every run and every repetition of a
// benchmark sees the same operation sequence, so runs are comparable.
#define SETUP_RNG \
    std::mt19937 rng(state.thread_index() + 42); \
    std::uniform_int_distribution<int> dist(0, 99)

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, ReadHeavy_StdAtomic, StdAtomicWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, StdAtomicWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 90) { // 90% read
            int count = 0;
            for (auto it = list->begin(); it != list->end() && count < 50; ++it) {
                benchmark::DoNotOptimize(*it);
                count++;
            }
        } else if (op < 95) { // 5% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        } else { // 5% erase
            list->erase_after(list->before_begin());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, WriteHeavy_StdAtomic, StdAtomicWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, StdAtomicWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 10) { // 10% read
            int count = 0;
            for (auto it = list->begin(); it != list->end() && count < 10; ++it) {
                benchmark::DoNotOptimize(*it);
                count++;
            }
        } else if (op < 55) { // 45% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        } else { // 45% erase
            list->erase_after(list->before_begin());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, Graveyard_StdAtomic, StdAtomicWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, StdAtomicWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 70) { // 70% erase
            list->erase_after(list->before_begin());
        } else { // 30% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, ReadHeavy_IntrPtr, IntrPtrWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, IntrPtrWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 90) { // 90% read
            int count = 0;
            for (auto it = list->begin(); it != list->end() && count < 50; ++it) {
                benchmark::DoNotOptimize(*it);
                count++;
            }
        } else if (op < 95) { // 5% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        } else { // 5% erase
            list->erase_after(list->before_begin());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, WriteHeavy_IntrPtr, IntrPtrWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, IntrPtrWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 10) { // 10% read
            int count = 0;
            for (auto it = list->begin(); it != list->end() && count < 10; ++it) {
                benchmark::DoNotOptimize(*it);
                count++;
            }
        } else if (op < 55) { // 45% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        } else { // 45% erase
            list->erase_after(list->before_begin());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, Graveyard_IntrPtr, IntrPtrWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, IntrPtrWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 70) { // 70% erase
            list->erase_after(list->before_begin());
        } else { // 30% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, ReadHeavy_HazardPtr, ParlayWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, ParlayWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 90) { // 90% read
            int count = 0;
            for (auto it = list->begin(); it != list->end() && count < 50; ++it) {
                benchmark::DoNotOptimize(*it);
                count++;
            }
        } else if (op < 95) { // 5% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        } else { // 5% erase
            list->erase_after(list->before_begin());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, WriteHeavy_HazardPtr, ParlayWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, ParlayWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 10) { // 10% read
            int count = 0;
            for (auto it = list->begin(); it != list->end() && count < 10; ++it) {
                benchmark::DoNotOptimize(*it);
                count++;
            }
        } else if (op < 55) { // 45% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        } else { // 45% erase
            list->erase_after(list->before_begin());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, Graveyard_HazardPtr, ParlayWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, ParlayWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        int op = dist(rng);
        if (op < 70) { // 70% erase
            list->erase_after(list->before_begin());
        } else { // 30% insert
            list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, MassiveHeadInsert_StdAtomic, StdAtomicWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, StdAtomicWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, MassiveHeadInsert_IntrPtr, IntrPtrWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, IntrPtrWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(ListFixture, MassiveHeadInsert_HazardPtr, ParlayWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, ParlayWrapper::template ptr_type>::Node;
    for (auto _ : state) {
        list->insert_after(list->before_begin(), factory.template operator()<Node>(dist(rng)));
    }
    state.SetItemsProcessed(state.iterations());
}

// ReadDispersed: read-heavy workload with contention dispersed across the list
// instead of pinned to the head. Every iteration reads a 50-element window; on
// top of that, 5% of iterations insert and 5% erase (the mutation anchor is a
// node discovered during the read pass, so a read always precedes a write --
// unlike ReadHeavy, where each iteration is either a read or a write). Each
// thread keeps a persistent cursor (declared outside the `for (auto _ : state)`
// loop, so it survives across iterations) that resumes where the previous read
// window ended, wrapping back to begin() at the tail. The cursor starts at this
// thread's slot in start_positions, so threads begin evenly spread around the
// large list (see DispersedListFixture) instead of convoying from begin().
// Inserts/erases anchor on a node captured mid-window, so mutations track
// wherever this thread's reads currently are rather than always hitting the
// head node.
//
// Note for the StdAtomic variant: this is the only benchmark whose mutation
// anchors are erasable nodes, and StdAtomicSharedPtrAdapter has
// supports_marking = false. It therefore exercises the documented non-marking
// limitations (black-hole inserts, and the erase/erase resurrect race of
// lock_free_list_bugs.md bug 2) that the head-anchored benchmarks never could.
// Both are memory-safe here -- every involved node is pinned by strong
// references -- but an occasional insert can vanish into a detached chain and
// an occasional erase can fail to shrink the list, so the StdAtomic numbers
// carry that semantic slack.
BENCHMARK_TEMPLATE_DEFINE_F(DispersedListFixture, ReadDispersed_StdAtomic, StdAtomicWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, StdAtomicWrapper::template ptr_type>::Node;
    using Iterator = LockFreeList<int, StdAtomicWrapper::template ptr_type>::iterator;
    constexpr int window = 50; // elements read per pass
    constexpr int mid = window / 2; // offset within the window where writes are anchored
    Iterator cursor; // this thread's staggered start, assigned on the first pass below
    bool first_pass = true;
    for (auto _ : state) {
        if (first_pass) {
            // The cursor cannot be initialized before the state loop: fixture
            // threads other than 0 can get there before thread 0's SetUp() has
            // built `list` and start_positions -- the barrier that makes them
            // safe to read only exists at the `for (auto _ : state)` above.
            cursor = start_positions[state.thread_index()];
            first_pass = false;
        }
        int op = dist(rng);
        Iterator it = cursor;
        Iterator anchor = cursor; // falls back to the window start if the list is too short to reach `mid`
        for (int count = 0; it != list->end() && count < window; ++count) {
            benchmark::DoNotOptimize(*it);
            if (count == mid - 1) {
                anchor = it; // predecessor of the mutation target, mid-way through the window
            }
            ++it;
        }
        if (op >= 90) {
            if (op < 95) { // 5% insert, anchored mid-window instead of at the head
                list->insert_after(anchor, factory.template operator()<Node>(dist(rng)));
            } else { // 5% erase, anchored mid-window instead of at the head
                list->erase_after(anchor);
            }
        }
        cursor = (it == list->end()) ? list->begin() : it; // wrap at the tail, otherwise resume here next pass
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(DispersedListFixture, ReadDispersed_IntrPtr, IntrPtrWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, IntrPtrWrapper::template ptr_type>::Node;
    using Iterator = LockFreeList<int, IntrPtrWrapper::template ptr_type>::iterator;
    constexpr int window = 50; // elements read per pass
    constexpr int mid = window / 2; // offset within the window where writes are anchored
    Iterator cursor; // this thread's staggered start, assigned on the first pass (see ReadDispersed_StdAtomic)
    bool first_pass = true;
    for (auto _ : state) {
        if (first_pass) {
            cursor = start_positions[state.thread_index()];
            first_pass = false;
        }
        int op = dist(rng);
        Iterator it = cursor;
        Iterator anchor = cursor; // falls back to the window start if the list is too short to reach `mid`
        for (int count = 0; it != list->end() && count < window; ++count) {
            benchmark::DoNotOptimize(*it);
            if (count == mid - 1) {
                anchor = it; // predecessor of the mutation target, mid-way through the window
            }
            ++it;
        }
        if (op >= 90) {
            if (op < 95) { // 5% insert, anchored mid-window instead of at the head
                list->insert_after(anchor, factory.template operator()<Node>(dist(rng)));
            } else { // 5% erase, anchored mid-window instead of at the head
                list->erase_after(anchor);
            }
        }
        cursor = (it == list->end()) ? list->begin() : it; // wrap at the tail, otherwise resume here next pass
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_DEFINE_F(DispersedListFixture, ReadDispersed_HazardPtr, ParlayWrapper)(benchmark::State& state) {
    SETUP_RNG;
    using Node = LockFreeList<int, ParlayWrapper::template ptr_type>::Node;
    using Iterator = LockFreeList<int, ParlayWrapper::template ptr_type>::iterator;
    constexpr int window = 50; // elements read per pass
    constexpr int mid = window / 2; // offset within the window where writes are anchored
    Iterator cursor; // this thread's staggered start, assigned on the first pass (see ReadDispersed_StdAtomic)
    bool first_pass = true;
    for (auto _ : state) {
        if (first_pass) {
            cursor = start_positions[state.thread_index()];
            first_pass = false;
        }
        int op = dist(rng);
        Iterator it = cursor;
        Iterator anchor = cursor; // falls back to the window start if the list is too short to reach `mid`
        for (int count = 0; it != list->end() && count < window; ++count) {
            benchmark::DoNotOptimize(*it);
            if (count == mid - 1) {
                anchor = it; // predecessor of the mutation target, mid-way through the window
            }
            ++it;
        }
        if (op >= 90) {
            if (op < 95) { // 5% insert, anchored mid-window instead of at the head
                list->insert_after(anchor, factory.template operator()<Node>(dist(rng)));
            } else { // 5% erase, anchored mid-window instead of at the head
                list->erase_after(anchor);
            }
        }
        cursor = (it == list->end()) ? list->begin() : it; // wrap at the tail, otherwise resume here next pass
    }
    state.SetItemsProcessed(state.iterations());
}

static const int num_cpu = sysconf(_SC_NPROCESSORS_CONF);

// ThreadRange(1, num_cpu) doubles the thread count at each step up to the
// core count. UseRealTime() reports wall-clock time per iteration instead of
// accumulated per-thread CPU time; CPU time would flatter implementations
// that block instead of spinning (the intr spinlock naps in nanosleep, the
// libstdc++ std::atomic<shared_ptr> waits on a mutex), while wall time is the
// throughput actually observed.
#define REGISTER_BMS(name) \
    BENCHMARK_REGISTER_F(ListFixture, ReadHeavy_##name)->ThreadRange(1, num_cpu)->UseRealTime(); \
    BENCHMARK_REGISTER_F(ListFixture, WriteHeavy_##name)->ThreadRange(1, num_cpu)->UseRealTime(); \
    BENCHMARK_REGISTER_F(ListFixture, Graveyard_##name)->ThreadRange(1, num_cpu)->UseRealTime(); \
    BENCHMARK_REGISTER_F(ListFixture, MassiveHeadInsert_##name)->ThreadRange(1, num_cpu)->UseRealTime(); \
    BENCHMARK_REGISTER_F(DispersedListFixture, ReadDispersed_##name)->ThreadRange(1, num_cpu)->UseRealTime();

REGISTER_BMS(StdAtomic)
REGISTER_BMS(IntrPtr)
REGISTER_BMS(HazardPtr)

BENCHMARK_MAIN();
