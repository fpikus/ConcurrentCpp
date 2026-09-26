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
// Test-only interleaving hooks (see RING_QUEUE_HOOK in concurrent_queue.h).
// Defining the macro BEFORE the include routes every hook point of the
// key-value push()/pop() to ring_queue_hook(), which parks the calling thread
// if the running test asked for that point (the Gate machinery in the
// "Forced interleavings" section below). Every test in this binary runs with
// the hooks compiled in; for a thread that owns no Gate a hook is, in an
// optimized build, one thread-local load and a branch (the sanitizer builds
// here are -O0, and pay a call). Production builds never define the macro
// and get no code at all for the hook points.
enum class RingQueueHook {
    NONE,               // no point: a Gate with stop_at == NONE parks nowhere
    PUSH_LOADED,        // push: slot state loaded under the tail lock, before deciding
    PUSH_CLAIMED,       // push: key written and slot claimed (tail advanced), lock released, value not yet constructed
    PUSH_COMMITTING,    // push: value constructed, before the commit store
    POP_LOADED,         // pop: slot state loaded under the head lock, before deciding
    POP_CLAIMED,        // pop: key read and cleared, slot claimed (head advanced), lock released, value not yet moved
    POP_RELEASING       // pop: value moved out and destroyed, before the release store
};
static void ring_queue_hook(RingQueueHook point);
#define RING_QUEUE_HOOK(point) ring_queue_hook(RingQueueHook::point)
#include "concurrent_queue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits.h>
#include <memory>
#include <numeric>
#include <set>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#include <gtest/gtest.h>

//================================================================================
// How this suite is organized (a template for testing concurrent containers):
//
//   1. Single-threaded correctness -- push/pop/empty/full/FIFO/wrap-around on
//      one thread. These pin down the sequential contract (what the queue must
//      do) before any threading is involved; a data structure that is wrong
//      single-threaded cannot be salvaged by careful synchronization.
//   2. Type-behavior tests -- copy-vs-move on push, destructor accounting,
//      move-only value types. These verify the queue manages element lifetimes
//      correctly, independent of concurrency.
//   3. Concurrency tests (SPSC/MPSC/SPMC/MPMC) -- multiple producer/consumer
//      threads. These check the *concurrent* contract: every pushed item is
//      delivered exactly once. NOTE: they deliberately do NOT check FIFO order
//      -- with several consumers interleaving under the head lock, global order
//      is not observable and not promised; only exactly-once delivery is. FIFO
//      is verified by the single-threaded FifoOrder/WrapAround tests instead.
//   4. TSAN/ASAN stress tests -- tiny queues (capacity 8) at high thread counts
//      to force constant wrap-around and slot reuse, which is where the
//      lock-free slot handoff (the key store in the key-only queue, the
//      per-slot sequence number protocol in the key-value queue -- see the
//      header) is most likely to race. Run under the sanitizers, these turn
//      rare interleavings into reproducible failures.
//   5. Forced interleavings -- the two schedules that broke the previous
//      key-value protocol, replayed deterministically through the header's
//      test-only hook points, plus a repeated-key stress test with a per-slot
//      live-count oracle. A stress test reaches a specific interleaving by
//      luck; these reach it by construction.
//
// Two integrity-checking strategies appear below, and the difference matters:
//   - std::set of consumed values: detects LOSS (a missing value shrinks the
//     set) but is BLIND to pure duplication (re-inserting a value is a no-op).
//   - atomic count + atomic sum vs. the closed-form expected sum: detects both
//     loss AND duplication. The stress tests use this stronger check; the
//     comment on RunMPMCStress spells out why.
//================================================================================

//================================================================================
// Test Fixture for Key-Only Queue (int)
//================================================================================

// Fixture: allocates a raw byte buffer and constructs a queue over it. This
// mirrors real use -- the queue never allocates; the caller owns the storage
// and passes a span. One fixture template serves every element-type/capacity
// combination via the type aliases below.
//   K, V  - queue key and value types (V = void selects the key-only queue).
//   Cap   - requested slot count. bytes is sized so the queue's power-of-2
//           floor (see RingAtomicMapQueueMPMC's capacity_ math) lands exactly
//           on Cap; using powers of 2 for Cap keeps capacity() == Cap.
//   memory_ - backing storage. Declared BEFORE queue_ so it is destroyed AFTER
//             queue_ (members destruct in reverse order): the queue's
//             destructor, which drains slots, must run while the buffer is
//             still alive.
//   queue_  - the queue under test, constructed in place over memory_.
template <typename K, typename V = void, size_t Cap = 16>
class QueueTester : public ::testing::Test {
    public:
    using queue_t = RingAtomicMapQueueMPMC<K, V>;
    static constexpr size_t capacity = Cap;
    static constexpr size_t bytes = capacity*queue_t::element_size();
    std::unique_ptr<char[]> memory_ { new char[bytes] };
    queue_t queue_ { memory_.get(), bytes };
};

using QueueTesterInt = QueueTester<int>;

// Property: a freshly constructed queue is empty and reports the requested
// capacity (the power-of-2 floor of the buffer size).
TEST_F(QueueTesterInt, Construct) {
    EXPECT_TRUE(queue_.empty());
    EXPECT_EQ(capacity, queue_.capacity());
}

// Property: a single push makes the queue non-empty.
TEST_F(QueueTesterInt, PushOne) {
    EXPECT_TRUE(queue_.push(1));
    EXPECT_FALSE(queue_.empty());
}

// Property: push then pop returns the same key and leaves the queue empty.
TEST_F(QueueTesterInt, PopOne) {
    EXPECT_TRUE(queue_.push(1));
    EXPECT_EQ(1, queue_.pop());
    EXPECT_TRUE(queue_.empty());
}

// Property: popping an empty key-only queue returns Key{} (0 for int) -- the
// reserved empty-slot sentinel doubles as the "queue empty" signal.
TEST_F(QueueTesterInt, PopEmpty) {
    EXPECT_EQ(0, queue_.pop());
    EXPECT_TRUE(queue_.empty());
}

// Property: exactly `capacity` slots are usable (no reserved gap between head
// and tail, unlike a classic ring buffer), and the (capacity+1)-th push fails
// as full. Confirms every slot is claimable and full is detected per-slot.
TEST_F(QueueTesterInt, FillToCapacity) {
    for (size_t i = 1; i <= capacity; ++i) {
        EXPECT_TRUE(queue_.push(i)) << "Failed to push item " << i;
    }

    EXPECT_FALSE(queue_.push(capacity + 1));
    EXPECT_FALSE(queue_.empty());
}

// Property: interleaved push/pop preserve FIFO order and empty/non-empty
// transitions on a single thread.
TEST_F(QueueTesterInt, PushPop) {
    EXPECT_TRUE(queue_.push(1));
    EXPECT_TRUE(queue_.push(2));
    EXPECT_TRUE(queue_.push(3));
    EXPECT_EQ(1, queue_.pop());
    EXPECT_EQ(2, queue_.pop());
    EXPECT_FALSE(queue_.empty());
    EXPECT_EQ(3, queue_.pop());
    EXPECT_TRUE(queue_.empty());
    EXPECT_EQ(0, queue_.pop());
    EXPECT_TRUE(queue_.empty());
}

// Property: FIFO order and slot reuse hold across two fill/drain rounds, so
// that slots freed by pop become available to later pushes (indices advance
// monotonically; freed slots are reclaimed only via wrap-around).
TEST_F(QueueTesterInt, PushPopTwice) {
    EXPECT_TRUE(queue_.push(1));
    EXPECT_TRUE(queue_.push(2));
    EXPECT_TRUE(queue_.push(3));
    EXPECT_TRUE(queue_.push(4));
    EXPECT_EQ(1, queue_.pop());
    EXPECT_EQ(2, queue_.pop());
    EXPECT_TRUE(queue_.push(5));
    EXPECT_TRUE(queue_.push(6));
    EXPECT_EQ(3, queue_.pop());
    EXPECT_EQ(4, queue_.pop());
    EXPECT_EQ(5, queue_.pop());
    EXPECT_FALSE(queue_.empty());
    EXPECT_EQ(6, queue_.pop());
    EXPECT_TRUE(queue_.empty());
    EXPECT_EQ(0, queue_.pop());
    EXPECT_TRUE(queue_.empty());
}

// Property: push keeps succeeding until exactly `capacity` items are in, then
// fails. The pre-increment ++i also runs for the failing push, so i overshoots
// by one -- capacity+1 is the expected terminal value, not an off-by-one bug.
TEST_F(QueueTesterInt, PushFull) {
    size_t i = 0;
    while (queue_.push(++i)) {};
    // i was also incremented for the push that failed, hence capacity + 1.
    EXPECT_EQ(i, capacity + 1);
}

// Property: after the queue is full, a single pop frees exactly one slot, and
// the next push (into that freed slot, via wrap-around) succeeds. Verifies
// full->not-full recovery.
TEST_F(QueueTesterInt, PushFullPop) {
    size_t i = 0;
    while (queue_.push(++i)) {};
    EXPECT_EQ(1, queue_.pop());
    EXPECT_TRUE(queue_.push(++i));
}

// Property: strict FIFO -- items come out in push order. This is the
// single-threaded oracle for ordering; the concurrent tests below cannot make
// this claim and only check exactly-once delivery.
TEST_F(QueueTesterInt, FifoOrder) {
    // Push a sequence of numbers.
    for (size_t i = 1; i <= capacity; ++i) {
        queue_.push(i);
    }

    // Pop them and verify they come out in the same order.
    for (size_t i = 1; i <= capacity; ++i) {
        EXPECT_EQ(int(i), queue_.pop()) << "Popped wrong item at position " << i;
    }
    EXPECT_TRUE(queue_.empty());
}

// Property: FIFO and fullness are preserved when head/tail indices wrap past
// the physical array end. Strategy: fill, drain half (advancing head into the
// array), refill so tail wraps around and reuses the freed low slots, confirm
// full again, then drain and check the surviving originals precede the new
// items in order. This exercises the capacity_mask_ index arithmetic and
// per-slot full/empty detection across the wrap boundary.
TEST_F(QueueTesterInt, WrapAround) {
    const size_t half_cap = capacity / 2;

    // 1. Fill the queue.
    for (size_t i = 1; i <= capacity; ++i) {
        queue_.push(i);
    }

    // 2. Pop the first half.
    for (size_t i = 1; i <= half_cap; ++i) {
        EXPECT_EQ(int(i), queue_.pop());
    }
    EXPECT_FALSE(queue_.empty());

    // 3. Push more items to force the internal index to wrap around.
    for (size_t i = capacity + 1; i <= capacity + half_cap; ++i) {
        EXPECT_TRUE(queue_.push(i));
    }
    
    // 4. The queue should now be full again.
    EXPECT_FALSE(queue_.push(999));

    // 5. Pop the remaining items and verify the order.
    // The second half of the original items.
    for (size_t i = half_cap + 1; i <= capacity; ++i) {
        EXPECT_EQ(int(i), queue_.pop());
    }
    // The new items that were pushed.
    for (size_t i = capacity + 1; i <= capacity + half_cap; ++i) {
        EXPECT_EQ(int(i), queue_.pop());
    }

    EXPECT_TRUE(queue_.empty());
}

// Property: capacity is the largest power of 2 of slots that fits the buffer
// (std::bit_floor), NOT the raw slot count -- the power-of-2 size is what makes
// capacity_mask_ (capacity_-1) a valid index mask. 31 requested slots -> 16.
TEST_F(QueueTesterInt, CapacityCalculation) {
    // Test that the queue correctly calculates capacity as the previous power of 2.
    // Request memory for 31 elements, which should result in a capacity of 16.
    const size_t requested_elements = 31;
    using queue_t = RingAtomicMapQueueMPMC<int>;
    const size_t bytes = requested_elements*queue_t::element_size();
    auto mem = std::make_unique<char[]>(bytes);
    queue_t q(mem.get(), bytes);

    EXPECT_EQ(16u, q.capacity());
}

//================================================================================
// Test Fixture for Key-Only Queue (Pointers)
//================================================================================

// Using raw pointers as keys to test the nullptr-as-invalid-value use case.
// For a pointer key, Key{} == nullptr is the reserved empty-slot marker, so a
// null pointer must never be pushed and pop() returns nullptr when empty. This
// is the natural use case: a queue of non-null resource pointers.
using QueueTesterPtr = QueueTester<int*>;

// Property: pointer keys round-trip in FIFO order (the key is copied bit-wise
// atomically, so any trivially-copyable lock-free key type works).
TEST_F(QueueTesterPtr, PushAndPopPointers) {
    int x = 1, y = 2, z = 3;
    
    EXPECT_TRUE(queue_.push(&x));
    EXPECT_TRUE(queue_.push(&y));
    EXPECT_TRUE(queue_.push(&z));
    EXPECT_FALSE(queue_.empty());
    
    EXPECT_EQ(&x, queue_.pop());
    EXPECT_EQ(&y, queue_.pop());
    EXPECT_EQ(&z, queue_.pop());
    EXPECT_TRUE(queue_.empty());
}

// Property: empty pop returns Key{} == nullptr for a pointer-key queue.
TEST_F(QueueTesterPtr, PopFromEmpty) {
    // Popping from an empty queue should return nullptr.
    EXPECT_EQ(nullptr, queue_.pop());
}

//================================================================================
// Test Fixture for Key-Value Queue (int, int)
//================================================================================

// KV queue: the slot carries a sequence number and a value in addition to the
// key (the key-only specialization has just the atomic key). These tests
// exercise the value-returning pop(val) overload, which writes the value by
// move-assignment into the caller's object.
using QueueTesterIntInt = QueueTester<int, int>;

TEST_F(QueueTesterIntInt, Construct) {
    EXPECT_TRUE(queue_.empty());
    EXPECT_EQ(capacity, queue_.capacity());
}

// Property: push(key, value) enqueues a pair; queue becomes non-empty.
TEST_F(QueueTesterIntInt, PushOne) {
    EXPECT_TRUE(queue_.push(1, 2));
    EXPECT_FALSE(queue_.empty());
}

// Property: pop(value) returns the key and delivers the paired value into the
// caller's out-parameter.
TEST_F(QueueTesterIntInt, PopOne) {
    EXPECT_TRUE(queue_.push(1, 2));
    int value = 0;
    EXPECT_EQ(1, queue_.pop(value));
    EXPECT_EQ(2, value);
    EXPECT_TRUE(queue_.empty());
}

// Property: on empty, pop(value) returns Key{} and leaves the caller's value
// object UNTOUCHED (7 stays 7) -- the value is only assigned on a real pop.
TEST_F(QueueTesterIntInt, PopEmpty) {
    int value = 7;
    EXPECT_EQ(0, queue_.pop(value));
    EXPECT_EQ(7, value);
    EXPECT_TRUE(queue_.empty());
}

//================================================================================
// Test Fixture for Key-Value Queue (int, string)
//================================================================================

// KV queue with a non-trivial (heap-owning, throwing-move-if-not-noexcept)
// value type. These tests verify the perfect-forwarding push -- an lvalue is
// copied into the slot, an rvalue is moved -- and that the value survives the
// placement-new-in-push / move-out-in-pop round trip intact.
using QueueTesterIntString = QueueTester<int, std::string>;

// Property: a value pushed as a COPY leaves the caller's source string
// unchanged (still "hello"). The copy is made explicitly by the caller,
// std::string(v_in): push() accepts only arguments from which the value is
// nothrow-constructible, and a std::string lvalue is not one (its copy
// constructor can throw), so push(1, v_in) does not compile -- see the
// compile-time checks in NothrowRequirements below. The explicit copy is a
// prvalue, moved into the slot without throwing. This distinguishes the
// caller-side copy from the move path tested next.
TEST_F(QueueTesterIntString, PushAndPop) {
    std::string v_in = "hello";
    bool pushed = queue_.push(1, std::string(v_in));
    EXPECT_TRUE(pushed);
    EXPECT_FALSE(queue_.empty());

    // Check that the original string was copied, not moved.
    EXPECT_EQ("hello", v_in);

    std::string val_out;
    int key_out = queue_.pop(val_out);
    
    EXPECT_EQ(1, key_out);
    EXPECT_EQ("hello", val_out);
    EXPECT_TRUE(queue_.empty());
}

// Property: push(key, rvalue) MOVE-constructs the value; the moved-from source
// is left valid-but-empty. The move actually happening (not a hidden copy) is
// what makes the queue usable for move-only and expensive-to-copy types.
TEST_F(QueueTesterIntString, PushByMove) {
    std::string v_in = "world";
    // Using std::move to test move-construction.
    EXPECT_TRUE(queue_.push(1, std::move(v_in)));
    EXPECT_FALSE(queue_.empty());
    
    // The original string should be in a valid but moved-from state (likely empty).
    EXPECT_TRUE(v_in.empty());

    std::string val_out;
    EXPECT_EQ(1, queue_.pop(val_out));
    EXPECT_EQ("world", val_out);
    EXPECT_TRUE(queue_.empty());
}


// Property: empty pop returns Key{} and does NOT touch the caller's value
// object (same contract as the int-value case, but here it matters more --
// no spurious move-assignment that could clobber a live string).
TEST_F(QueueTesterIntString, PopFromEmpty) {
    std::string val_out = "initial";
    int key_out = queue_.pop(val_out);

    // Popping from an empty queue should return a default key and not modify the value.
    EXPECT_EQ(0, key_out);
    EXPECT_EQ("initial", val_out);
}

// Property: KV queue also caps at exactly `capacity` slots and reports full.
TEST_F(QueueTesterIntString, FillToCapacity) {
    for (size_t i = 1; i <= capacity; ++i) {
        EXPECT_TRUE(queue_.push(i, "value_" + std::to_string(i)));
    }
    // Queue should now be full.
    EXPECT_FALSE(queue_.push(99, std::string("should_fail")));
}

// Property: FIFO order holds for KV, and each key arrives paired with its
// original value (checks the key/value are stored and retrieved together, not
// mismatched across slots).
TEST_F(QueueTesterIntString, FifoOrder) {
    for (size_t i = 1; i <= capacity; ++i) {
        queue_.push(i, "value_" + std::to_string(i));
    }

    for (size_t i = 1; i <= capacity; ++i) {
        std::string val;
        EXPECT_EQ(int(i), queue_.pop(val));
        EXPECT_EQ("value_" + std::to_string(i), val);
    }
    EXPECT_TRUE(queue_.empty());
}

// Property: KV wrap-around preserves both FIFO order and key/value pairing
// when slots are reused (the value's placement-new/destruct cycle must survive
// wrap-around, not just the key store). Same fill/drain-half/refill strategy as
// the key-only WrapAround.
TEST_F(QueueTesterIntString, WrapAround) {
    const size_t half_cap = capacity / 2;

    // 1. Fill queue.
    for (size_t i = 1; i <= capacity; ++i) {
        queue_.push(i, "v" + std::to_string(i));
    }

    // 2. Pop first half.
    for (size_t i = 1; i <= half_cap; ++i) {
        std::string val;
        EXPECT_EQ(int(i), queue_.pop(val));
        EXPECT_EQ("v" + std::to_string(i), val);
    }

    // 3. Push more to wrap around.
    for (size_t i = capacity + 1; i <= capacity + half_cap; ++i) {
        EXPECT_TRUE(queue_.push(i, "v" + std::to_string(i)));
    }
    
    // 4. Pop remaining items and verify order.
    for (size_t i = half_cap + 1; i <= capacity + half_cap; ++i) {
        std::string val;
        EXPECT_EQ(int(i), queue_.pop(val));
        EXPECT_EQ("v" + std::to_string(i), val);
    }
    EXPECT_TRUE(queue_.empty());
}

//================================================================================
// Concurrency Tests (MPMC) - Key-only queue (int)
//================================================================================
// running_under_TAP: true when the tests run inside the CI harness (env var
// CALJENKINS_UNIT_TESTS set). CI machines are loaded, so per-thread scheduling
// can be very uneven; checks that assume every consumer gets to run (the
// per-consumer "popped at least one" assertions below) are disabled there to
// avoid false failures from a starved-but-correct consumer.
static const bool running_under_TAP = ::getenv("CALJENKINS_UNIT_TESTS");

using QueueTesterIntConcurrent = QueueTester<int, void, 1024>; // Larger capacity for concurrency tests

// Synchronization pattern shared by all concurrency tests below:
//
// - barrier is a countdown start-gate: every worker decrements it, then spins
//   until it reaches zero, so all threads leave the gate together and spend
//   the maximum time overlapping. This is a stress technique, not a
//   correctness mechanism: the loads/stores are relaxed, so the barrier
//   establishes NO happens-before edge. Correctness rests entirely on the
//   queue's own acquire/release protocol and on remaining_producers below;
//   the barrier only makes races more likely to be hit.
//
// - remaining_producers uses release (decrement, after a producer's last push)
//   / acquire (consumer's load) so that when a consumer observes 0, every push
//   is visible to it. It is the termination signal: while it is > 0 a consumer
//   that sees an empty queue must keep trying (items may still be coming);
//   once it is 0, no more pushes will ever happen.
//
// - Integrity oracle (these set-based tests): each producer i emits the
//   disjoint value range [i*items_per_producer+1 .. (i+1)*items_per_producer],
//   so every value in the run is globally unique. Consumers union their
//   consumed values into all_consumed_items; size == total_items proves no
//   item was LOST. (This oracle cannot see pure duplication -- re-inserting a
//   value is idempotent; the stress tests use a sum-based oracle that can.)
//   These tests assert exactly-once delivery, NOT FIFO order.
//
// Termination: while remaining_producers > 0, a consumer that sees pop()==0
// keeps polling. Once it observes remaining_producers == 0 it DRAINS the
// queue (pops until empty) before exiting. The drain is required: pop()==0
// and the counter load are two separate observations, so a producer can push
// its final items and drop the counter to 0 between them; without the drain
// the consumer would break with those items stranded and fail the size and
// empty() checks. Draining after the 0 observation is complete because no
// further push can ever occur.
TEST_F(QueueTesterIntConcurrent, SingleProducerSingleConsumer) {
    const size_t items_per_producer = 1000;
    const size_t producer_count = 1;
    const size_t consumer_count = 1;
    const size_t total_items = producer_count*items_per_producer;
    std::atomic<int> barrier {producer_count + consumer_count};
    std::atomic<int> remaining_producers {producer_count};

    std::thread producer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
        for (size_t j = 0; j < items_per_producer; ++j) {
            int value = j + 1;
            while (!queue_.push(value)) { std::this_thread::yield(); }
        }
        remaining_producers.fetch_sub(1, std::memory_order_release);
    });

    std::set<int> all_consumed_items;
    std::mutex m;       // Guards all_consumed_items
    std::thread consumer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
        std::set<int> consumed_items;
        while (true) {
            int value = queue_.pop();
            if (value != 0) {
                consumed_items.insert(value);
            } else if (remaining_producers.load(std::memory_order_acquire)) { // Producers still running, keep trying
                std::this_thread::yield();
            } else {    // Producers done -- drain items pushed after our empty
                        // pop() but before the counter hit 0, then exit (see
                        // the termination note above).
                while ((value = queue_.pop()) != 0) { consumed_items.insert(value); }
                break;
            }
        }
        std::lock_guard g(m);
        for (int value : consumed_items) all_consumed_items.insert(value);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntConcurrent.SingleProducerSingleConsumer

// Topology: 4 producers, 1 consumer (MPSC). Stresses concurrent tail claims by
// multiple producers against a single draining consumer; each producer's
// disjoint value range lets the single set detect any lost item.
TEST_F(QueueTesterIntConcurrent, MultiProducerSingleConsumer) {
    const size_t items_per_producer = 1000;
    const size_t producer_count = 4;
    const size_t consumer_count = 1;
    const size_t total_items = producer_count*items_per_producer;
    std::atomic<int> barrier {producer_count + consumer_count};
    std::atomic<int> remaining_producers {producer_count};

    std::vector<std::thread> producers(producer_count);
    for (size_t i = 0; i != producer_count; ++i) {
        producers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
            for (size_t j = 0; j < items_per_producer; ++j) {
                int value = items_per_producer*i + j + 1;
                while (!queue_.push(value)) { std::this_thread::yield(); }
            }
            remaining_producers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::set<int> all_consumed_items;
    std::mutex m;       // Guards all_consumed_items
    std::thread consumer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
        std::set<int> consumed_items;
        while (true) {
            int value = queue_.pop();
            if (value != 0) {
                consumed_items.insert(value);
            } else if (remaining_producers.load(std::memory_order_acquire)) { // Producers still running, keep trying
                std::this_thread::yield();
            } else {    // Producers done -- drain items pushed after our empty
                        // pop() but before the counter hit 0, then exit (see
                        // the termination note above).
                while ((value = queue_.pop()) != 0) { consumed_items.insert(value); }
                break;
            }
        }
        std::lock_guard g(m);
        for (int value : consumed_items) all_consumed_items.insert(value);
    });

    for (auto& producer : producers) producer.join();
    consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntConcurrent.MultiProducerSingleConsumer

// Topology: 4 producers, 4 consumers (full MPMC). Both the tail (producers)
// and head (consumers) locks are contended simultaneously -- the hardest case
// for the slot-handoff protocol. Consumers merge their per-thread sets under a
// mutex (the queue does not track who consumed what). The extra per-consumer
// EXPECT_NE(0) checks that no consumer was completely starved (skipped under
// CI, where uneven scheduling can legitimately starve one).
TEST_F(QueueTesterIntConcurrent, MultiProducerMultiConsumer) {
    const size_t items_per_producer = 1000;
    const size_t producer_count = 4;
    const size_t consumer_count = 4;
    const size_t total_items = producer_count*items_per_producer;
    std::atomic<int> barrier {producer_count + consumer_count};
    std::atomic<int> remaining_producers {producer_count};

    std::vector<std::thread> producers(producer_count);
    for (size_t i = 0; i != producer_count; ++i) {
        producers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
            for (size_t j = 0; j < items_per_producer; ++j) {
                int value = items_per_producer*i + j + 1;
                while (!queue_.push(value)) { std::this_thread::yield(); }
            }
            remaining_producers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::set<int> all_consumed_items;
    std::mutex m;       // Guards all_consumed_items
    std::vector<std::thread> consumers(consumer_count);
    for (size_t i = 0; i != consumer_count; ++i) {
        consumers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
            std::set<int> consumed_items;
            while (true) {
                int value = queue_.pop();
                if (value != 0) {
                    consumed_items.insert(value);
                } else if (remaining_producers.load(std::memory_order_acquire)) { // Producers still running, keep trying
                    std::this_thread::yield();
                } else {    // Producers done -- drain items pushed after our
                            // empty pop() but before the counter hit 0, then
                            // exit (see the termination note above).
                    while ((value = queue_.pop()) != 0) { consumed_items.insert(value); }
                    break;
                }
            }
            std::lock_guard g(m);
            for (int value : consumed_items) all_consumed_items.insert(value);
            if (!running_under_TAP) { EXPECT_NE(0u, consumed_items.size()) << "Consumer " << i << " did not pop any elements"; }
        });
    }

    for (auto& producer : producers) producer.join();
    for (auto& consumer : consumers) consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntConcurrent.MultiProducerMultiConsumer

//================================================================================
// Concurrency Tests (MPMC) - Key-value queue (int, string)
//================================================================================
// Same topologies as above, but with a std::string value. Beyond exactly-once
// delivery these add a per-item integrity check: the popped value string must
// equal std::to_string(key). A key that arrives paired with the wrong value
// (a torn key/value handoff, or a value read from a half-committed slot) is
// caught by EXPECT_EQ(value, std::stoi(s)). This is what the per-slot sequence
// number protocol in the header exists to prevent: the key and the value are
// published together by the one release store that commits the slot.

using QueueTesterIntStringConcurrent = QueueTester<int, std::string, 1024>; // Larger capacity for concurrency tests

TEST_F(QueueTesterIntStringConcurrent, SingleProducerSingleConsumer) {
    const size_t items_per_producer = 1000;
    const size_t producer_count = 1;
    const size_t consumer_count = 1;
    const size_t total_items = producer_count*items_per_producer;
    std::atomic<int> barrier {producer_count + consumer_count};
    std::atomic<int> remaining_producers {producer_count};

    std::thread producer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
        for (size_t j = 0; j < items_per_producer; ++j) {
            int value = j + 1;
            while (!queue_.push(value, std::to_string(value))) { std::this_thread::yield(); }
        }
        remaining_producers.fetch_sub(1, std::memory_order_release);
    });

    std::set<int> all_consumed_items;
    std::mutex m;       // Guards all_consumed_items
    std::thread consumer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
        std::set<int> consumed_items;
        std::string s;
        while (true) {
            int value = queue_.pop(s);
            if (value != 0) {
                EXPECT_EQ(value, std::stoi(s));
                consumed_items.insert(value);
            } else if (remaining_producers.load(std::memory_order_acquire)) { // Producers still running, keep trying
                std::this_thread::yield();
            } else {    // Producers done -- drain items pushed after our empty
                        // pop() but before the counter hit 0, then exit (see
                        // the termination note above).
                while ((value = queue_.pop(s)) != 0) {
                    EXPECT_EQ(value, std::stoi(s));
                    consumed_items.insert(value);
                }
                break;
            }
        }
        std::lock_guard g(m);
        for (int value : consumed_items) all_consumed_items.insert(value);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntStringConcurrent.SingleProducerSingleConsumer

TEST_F(QueueTesterIntStringConcurrent, MultiProducerSingleConsumer) {
    const size_t items_per_producer = 1000;
    const size_t producer_count = 4;
    const size_t consumer_count = 1;
    const size_t total_items = producer_count*items_per_producer;
    std::atomic<int> barrier {producer_count + consumer_count};
    std::atomic<int> remaining_producers {producer_count};

    std::vector<std::thread> producers(producer_count);
    for (size_t i = 0; i != producer_count; ++i) {
        producers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
            for (size_t j = 0; j < items_per_producer; ++j) {
                int value = items_per_producer*i + j + 1;
                while (!queue_.push(value, std::to_string(value))) { std::this_thread::yield(); }
            }
            remaining_producers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::set<int> all_consumed_items;
    std::mutex m;       // Guards all_consumed_items
    std::thread consumer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
        std::set<int> consumed_items;
        std::string s;
        while (true) {
            int value = queue_.pop(s);
            if (value != 0) {
                EXPECT_EQ(value, std::stoi(s));
                consumed_items.insert(value);
            } else if (remaining_producers.load(std::memory_order_acquire)) { // Producers still running, keep trying
                std::this_thread::yield();
            } else {    // Producers done -- drain items pushed after our empty
                        // pop() but before the counter hit 0, then exit (see
                        // the termination note above).
                while ((value = queue_.pop(s)) != 0) {
                    EXPECT_EQ(value, std::stoi(s));
                    consumed_items.insert(value);
                }
                break;
            }
        }
        std::lock_guard g(m);
        for (int value : consumed_items) all_consumed_items.insert(value);
    });

    for (auto& producer : producers) producer.join();
    consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntStringConcurrent.MultiProducerSingleConsumer

TEST_F(QueueTesterIntStringConcurrent, MultiProducerMultiConsumer) {
    const size_t items_per_producer = 1000;
    const size_t producer_count = 4;
    const size_t consumer_count = 4;
    const size_t total_items = producer_count*items_per_producer;
    std::atomic<int> barrier {producer_count + consumer_count};
    std::atomic<int> remaining_producers {producer_count};

    std::vector<std::thread> producers(producer_count);
    for (size_t i = 0; i != producer_count; ++i) {
        producers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
            for (size_t j = 0; j < items_per_producer; ++j) {
                int value = items_per_producer*i + j + 1;
                while (!queue_.push(value, std::to_string(value))) { std::this_thread::yield(); }
            }
            remaining_producers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::set<int> all_consumed_items;
    std::mutex m;       // Guards all_consumed_items
    std::vector<std::thread> consumers(consumer_count);
    for (size_t i = 0; i != consumer_count; ++i) {
        consumers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {} // Wait for all threads to be ready
            std::set<int> consumed_items;
            std::string s;
            while (true) {
                int value = queue_.pop(s);
                if (value != 0) {
                    EXPECT_EQ(value, std::stoi(s));
                    consumed_items.insert(value);
                } else if (remaining_producers.load(std::memory_order_acquire)) { // Producers still running, keep trying
                    std::this_thread::yield();
                } else {    // Producers done -- drain items pushed after our
                            // empty pop() but before the counter hit 0, then
                            // exit (see the termination note above).
                    while ((value = queue_.pop(s)) != 0) {
                        EXPECT_EQ(value, std::stoi(s));
                        consumed_items.insert(value);
                    }
                    break;
                }
            }
            std::lock_guard g(m);
            for (int value : consumed_items) all_consumed_items.insert(value);
            if (!running_under_TAP) { EXPECT_NE(0u, consumed_items.size()) << "Consumer " << i << " did not pop any elements"; }
        });
    }

    for (auto& producer : producers) producer.join();
    for (auto& consumer : consumers) consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntStringConcurrent.MultiProducerMultiConsumer

//================================================================================
// Value Destructor Test
//================================================================================

// Instrumented value type: counts constructions and destructions (atomically,
// so it is safe to reuse in threaded tests) to verify the queue balances
// placement-new against explicit destructor calls and never leaks or
// double-destroys a slot value. The move constructor and move assignment are
// noexcept because push()/pop() with a CountedValue argument require it; the
// destructor, because the class requires it.
struct CountedValue {
    static inline std::atomic<int> destructor_count {0};
    static inline std::atomic<int> constructor_count {0};
    int x;
    CountedValue() : x(0) { constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue(int v) : x(v) { constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue(const CountedValue& o) : x(o.x) { constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue(CountedValue&& o) noexcept : x(o.x) { o.x = 0; constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue& operator=(CountedValue&& o) noexcept { x = o.x; o.x = 0; return *this; }
    ~CountedValue() noexcept { destructor_count.fetch_add(1, std::memory_order_relaxed); }
    static void reset() { destructor_count = 0; constructor_count = 0; }
};

using QueueTesterCounted = QueueTester<int, CountedValue>;

// Property: pop() runs exactly one value destructor per popped slot -- the
// in-place ~Value() that matches push()'s placement-new -- with no leak and no
// double-destroy.
// Strategy: measure the destructor count as a DELTA across the three pops
// only. Taking dtors_before AFTER the pushes deliberately excludes the
// destructors of the three push-argument temporaries (each push builds a
// temporary CountedValue and move-constructs it into the slot), isolating the
// count to pop()'s own destructor calls. Hence the expected delta is 3, not 6.
TEST_F(QueueTesterCounted, DestructorCalledOnPop) {
    CountedValue::reset();
    // Push 3 elements (each push creates a temporary + in-place move construction).
    EXPECT_TRUE(queue_.push(1, CountedValue(10)));
    EXPECT_TRUE(queue_.push(2, CountedValue(20)));
    EXPECT_TRUE(queue_.push(3, CountedValue(30)));

    int dtors_before = CountedValue::destructor_count.load();

    // Pop all three. Each pop explicitly destructs the in-queue value.
    CountedValue v;
    EXPECT_EQ(1, queue_.pop(v));
    EXPECT_EQ(10, v.x);
    EXPECT_EQ(2, queue_.pop(v));
    EXPECT_EQ(20, v.x);
    EXPECT_EQ(3, queue_.pop(v));
    EXPECT_EQ(30, v.x);

    int dtors_after = CountedValue::destructor_count.load();
    // 3 explicit destructor calls in pop (one per slot).
    EXPECT_EQ(3, dtors_after - dtors_before);
    EXPECT_TRUE(queue_.empty());
}

//================================================================================
// Move-Only Value Type Test
//================================================================================

// Move-only value type (deleted copy). Its mere use as the queue's Value proves
// at compile time that push/pop never require a copy -- push forwards an rvalue
// into a move-construction and pop uses move-assignment. Both are noexcept, as
// push()/pop() require.
struct MoveOnly {
    int x;
    MoveOnly() : x(0) {}
    MoveOnly(int v) : x(v) {}
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&& o) noexcept : x(o.x) { o.x = 0; }
    MoveOnly& operator=(MoveOnly&& o) noexcept { x = o.x; o.x = 0; return *this; }
};

using QueueTesterMoveOnly = QueueTester<int, MoveOnly>;

// Property: a move-only value round-trips through push (by move) and pop (by
// move-assign) with the payload preserved.
TEST_F(QueueTesterMoveOnly, PushAndPopMoveOnly) {
    EXPECT_TRUE(queue_.push(1, MoveOnly(42)));
    EXPECT_TRUE(queue_.push(2, MoveOnly(84)));

    MoveOnly v;
    EXPECT_EQ(1, queue_.pop(v));
    EXPECT_EQ(42, v.x);
    EXPECT_EQ(2, queue_.pop(v));
    EXPECT_EQ(84, v.x);
    EXPECT_TRUE(queue_.empty());
}

//================================================================================
// Nothrow requirements (compile-time)
//================================================================================

// The queue has no exception path (see the class comment in the header):
// push() accepts only arguments from which Value is nothrow-constructible,
// pop() only a destination that Value can be nothrow-assigned into, and the
// class a Value whose destructor cannot throw. These checks pin that down at compile time. Each is a
// concept whose requires-expression asks whether the call is well-formed for
// the given argument type (a concept, not a bare requires-expression, so that
// an ill-formed call is a false value rather than a compile error). A
// std::string rvalue is accepted, a std::string lvalue is rejected (its copy
// constructor can throw), and pop() into an object whose assignment from the
// value is not noexcept is rejected. A Value type whose own move operations
// can throw is refused by the class itself (see below).
namespace nothrow_requirements {
    // Two pop() destinations that can be assigned from a std::string: one
    // noexcept (accepted), one not (refused). The pair pins that the refusal
    // is about noexcept, not about assignability.
    struct NothrowSink {
        std::string s;
        NothrowSink& operator=(std::string&& v) noexcept { s = std::move(v); return *this; }
    };
    struct ThrowingSink {
        std::string s;
        ThrowingSink& operator=(std::string&& v) { s = std::move(v); return *this; }
    };
    // pushable<Q, Arg>: q.push(key, arg) compiles for a queue Q and an
    // argument of type Arg (a reference type makes it an lvalue argument).
    template <typename Q, typename Arg>
    concept pushable = requires(Q& q, Arg&& arg) { q.push(1, std::forward<Arg>(arg)); };
    // poppable<Q, V>: q.pop(v) compiles for a queue Q and an lvalue V.
    template <typename Q, typename V>
    concept poppable = requires(Q& q, V& v) { q.pop(v); };
    using string_queue_t = RingAtomicMapQueueMPMC<int, std::string>;

    static_assert(pushable<string_queue_t, std::string>);           // rvalue string: accepted
    static_assert(!pushable<string_queue_t, std::string&>);         // lvalue string (throwing copy): rejected
    static_assert(!pushable<string_queue_t, const char*>);          // std::string's converting constructor can throw: rejected
    static_assert(poppable<string_queue_t, std::string>);           // nothrow move assignment: accepted
    static_assert(poppable<string_queue_t, NothrowSink>);           // nothrow assignment from the value: accepted
    static_assert(!poppable<string_queue_t, ThrowingSink>);         // assignment from the value can throw: rejected
    // Value need not be movable: std::atomic<int> is neither move-constructible
    // nor move-assignable, yet is nothrow-constructible from int and
    // nothrow-convertible to it (see the NonMovableValue test).
    using atomic_queue_t = RingAtomicMapQueueMPMC<int, std::atomic<int>>;
    static_assert(pushable<atomic_queue_t, int>);
    static_assert(poppable<atomic_queue_t, int>);
    // A queue of a Value whose destructor may throw is refused by the class's
    // static_assert. That is a hard error at instantiation, which no
    // requires-expression can observe, so it is not tested here; it was
    // checked by compiling it once (the build fails at the static_assert).
    // push()/pop() are noexcept in both modes.
    static_assert(noexcept(std::declval<string_queue_t&>().push(1, std::declval<std::string&&>())));
    static_assert(noexcept(std::declval<string_queue_t&>().pop(std::declval<std::string&>())));
    static_assert(noexcept(std::declval<RingAtomicMapQueueMPMC<int>&>().push(1)));
    static_assert(noexcept(std::declval<RingAtomicMapQueueMPMC<int>&>().pop()));
} // namespace nothrow_requirements

// A Value that cannot be moved at all: pushed from an int (constructed in
// place), popped into an int (converted out), destroyed in its slot. The class
// requires only the nothrow destructor; the rest is per call.
TEST(NothrowRequirements, NonMovableValue) {
    using queue_t = RingAtomicMapQueueMPMC<int, std::atomic<int>>;
    constexpr size_t N = 8;
    alignas(queue_t::element_align()) static char buffer[N * queue_t::element_size()];
    queue_t q(buffer, sizeof(buffer));
    EXPECT_TRUE(q.push(1, 42));
    EXPECT_TRUE(q.push(2, 43));
    int v = 0;
    EXPECT_EQ(1, q.pop(v));
    EXPECT_EQ(42, v);
    EXPECT_EQ(2, q.pop(v));
    EXPECT_EQ(43, v);
    EXPECT_EQ(0, q.pop(v));
    EXPECT_EQ(43, v);           // Unchanged on empty
}

//================================================================================
// Element Size and Alignment Tests
//================================================================================

// Property: the caller can size a buffer from element_size() before knowing
// the queue's internal slot layout. A key-only slot is at least the size of
// its atomic key (the key-only specialization stores only the key).
TEST(ElementSizeTest, KeyOnly) {
    using queue_t = RingAtomicMapQueueMPMC<int>;
    EXPECT_GE(queue_t::element_size(), sizeof(std::atomic<int>));
}

// Property: a KV slot holds at least key + value (plus the sequence number and
// any padding, hence >= rather than ==).
TEST(ElementSizeTest, KeyValue) {
    using queue_t = RingAtomicMapQueueMPMC<int, std::string>;
    EXPECT_GE(queue_t::element_size(), sizeof(std::atomic<int>) + sizeof(std::string));
}

// Property: the ALIGN template parameter (here 64) pads each slot to a whole
// cache line so adjacent slots never share a line -- element_size() becomes a
// multiple of 64 and at least 64. This is the knob that trades memory for the
// elimination of false sharing between neighboring slots.
TEST(ElementSizeTest, CacheLineAligned) {
    using queue_t = RingAtomicMapQueueMPMC<int, void, 8, 64>;
    EXPECT_EQ(0u, queue_t::element_size() % 64u);
    EXPECT_GE(queue_t::element_size(), 64u);
}

//================================================================================
// Stress Test (many wrap-arounds)
//================================================================================

// Property: single-threaded slot reuse is correct over ~100 full laps of the
// ring. Strategy: push capacity*100 items, popping one whenever the queue is
// full to make room, then drain; every pop must return a real item (never 0),
// and the total popped must equal the total pushed. Forcing the index far past
// SIZE_MAX-free wrap exercises the capacity_mask_ arithmetic under sustained
// churn without the noise of threading.
TEST_F(QueueTesterInt, StressWrapAround) {
    // Push and pop many more items than the capacity to stress wrap-around.
    const size_t total = capacity * 100;
    size_t popped = 0;
    for (size_t i = 1; i <= total; ++i) {
        while (!queue_.push(i)) {
            // Queue full, pop one to make room.
            int v = queue_.pop();
            EXPECT_NE(0, v);
            ++popped;
        }
    }
    // Drain remaining items.
    while (!queue_.empty()) {
        int v = queue_.pop();
        EXPECT_NE(0, v);
        ++popped;
    }
    EXPECT_EQ(total, popped);
    EXPECT_TRUE(queue_.empty());
}

//================================================================================
// Concurrency Tests (SPMC) - Key-only queue (int)
//================================================================================
// Single producer, many consumers. The single producer emits one contiguous
// range 1..total_items (no per-producer offset needed since there is only one),
// and the consumers race to divide the work. Exercises head-lock contention
// among consumers against a lone producer.

TEST_F(QueueTesterIntConcurrent, SingleProducerMultiConsumer) {
    const size_t total_items = 4000;
    const size_t consumer_count = 4;
    std::atomic<int> barrier {1 + static_cast<int>(consumer_count)};
    std::atomic<int> remaining_producers {1};

    std::thread producer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {}
        for (size_t j = 0; j < total_items; ++j) {
            int value = j + 1;
            while (!queue_.push(value)) { std::this_thread::yield(); }
        }
        remaining_producers.fetch_sub(1, std::memory_order_release);
    });

    std::set<int> all_consumed_items;
    std::mutex m;
    std::vector<std::thread> consumers(consumer_count);
    for (size_t i = 0; i != consumer_count; ++i) {
        consumers[i] = std::thread([&]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {}
            std::set<int> consumed_items;
            while (true) {
                int value = queue_.pop();
                if (value != 0) {
                    consumed_items.insert(value);
                } else if (remaining_producers.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                } else {    // Producers done -- drain items pushed after our
                            // empty pop() but before the counter hit 0, then
                            // exit (see the termination note above).
                    while ((value = queue_.pop()) != 0) { consumed_items.insert(value); }
                    break;
                }
            }
            std::lock_guard g(m);
            for (int value : consumed_items) all_consumed_items.insert(value);
        });
    }

    producer.join();
    for (auto& consumer : consumers) consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntConcurrent.SingleProducerMultiConsumer

//================================================================================
// Concurrency Tests (SPMC) - Key-value queue (int, string)
//================================================================================
// SPMC as above, with the added per-item key==stoi(value) integrity check.

TEST_F(QueueTesterIntStringConcurrent, SingleProducerMultiConsumer) {
    const size_t total_items = 4000;
    const size_t consumer_count = 4;
    std::atomic<int> barrier {1 + static_cast<int>(consumer_count)};
    std::atomic<int> remaining_producers {1};

    std::thread producer([&]() {
        barrier.fetch_sub(1, std::memory_order_relaxed);
        while (barrier.load(std::memory_order_relaxed) != 0) {}
        for (size_t j = 0; j < total_items; ++j) {
            int value = j + 1;
            while (!queue_.push(value, std::to_string(value))) { std::this_thread::yield(); }
        }
        remaining_producers.fetch_sub(1, std::memory_order_release);
    });

    std::set<int> all_consumed_items;
    std::mutex m;
    std::vector<std::thread> consumers(consumer_count);
    for (size_t i = 0; i != consumer_count; ++i) {
        consumers[i] = std::thread([&]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {}
            std::set<int> consumed_items;
            std::string s;
            while (true) {
                int value = queue_.pop(s);
                if (value != 0) {
                    EXPECT_EQ(value, std::stoi(s));
                    consumed_items.insert(value);
                } else if (remaining_producers.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                } else {    // Producers done -- drain items pushed after our
                            // empty pop() but before the counter hit 0, then
                            // exit (see the termination note above).
                    while ((value = queue_.pop(s)) != 0) {
                        EXPECT_EQ(value, std::stoi(s));
                        consumed_items.insert(value);
                    }
                    break;
                }
            }
            std::lock_guard g(m);
            for (int value : consumed_items) all_consumed_items.insert(value);
        });
    }

    producer.join();
    for (auto& consumer : consumers) consumer.join();

    EXPECT_EQ(total_items, all_consumed_items.size());
    EXPECT_TRUE(queue_.empty());
} // QueueTesterIntStringConcurrent.SingleProducerMultiConsumer

//================================================================================
// TSAN Stress Tests — high contention, long duration, sum-based integrity
//================================================================================

// Helper: run an MPMC key-only stress test with configurable topology.
//   producer_count / consumer_count - thread counts.
//   items_per_producer              - items each producer pushes.
// Integrity oracle (stronger than the set-based tests above): each item is
// counted AND summed. total_popped must equal total_items (catches loss/extra
// pops) and total_sum must equal the closed-form sum 1..total_items (catches
// duplication AND loss). A std::set cannot see pure duplication because
// re-inserting a value is idempotent; the count+sum pair can, which is why the
// stress tests -- where slot reuse is most aggressive -- use it.
//
// Termination: a consumer that sees pop()==0 while producers still run keeps
// polling. Once remaining_producers hits 0 it performs an explicit DRAIN loop
// (pop until 0) before exiting. The drain is load-bearing, not decoration: a
// consumer can observe an empty queue (pop()==0) in the instant BEFORE a
// still-running producer pushes its final items and then decrements
// remaining_producers to 0; without the drain that consumer would break and
// strand those items. Draining after remaining_producers==0 is safe because no
// further push can occur, so pop()==0 then means the queue is globally empty.
template <typename K, typename V, size_t Cap>
void RunMPMCStress(size_t producer_count, size_t consumer_count,
                   size_t items_per_producer) {
    using queue_t = RingAtomicMapQueueMPMC<K, V>;
    constexpr size_t bytes = Cap * queue_t::element_size();
    auto mem = std::make_unique<char[]>(bytes);
    queue_t queue(mem.get(), bytes);

    const size_t total_items = producer_count * items_per_producer;
    // Expected sum: each producer pushes values base+1 .. base+items_per_producer
    // where base = i * items_per_producer.  Total sum = sum(1..total_items).
    const int64_t expected_sum =
        static_cast<int64_t>(total_items) * (total_items + 1) / 2;

    std::atomic<int> barrier{
        static_cast<int>(producer_count + consumer_count)};
    std::atomic<int> remaining_producers{static_cast<int>(producer_count)};

    std::vector<std::thread> producers(producer_count);
    for (size_t i = 0; i != producer_count; ++i) {
        producers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {}
            for (size_t j = 0; j < items_per_producer; ++j) {
                int value = static_cast<int>(i * items_per_producer + j + 1);
                while (!queue.push(value)) {
                    std::this_thread::yield();
                }
            }
            remaining_producers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::atomic<size_t> total_popped{0};
    std::atomic<int64_t> total_sum{0};

    std::vector<std::thread> consumers(consumer_count);
    for (size_t i = 0; i != consumer_count; ++i) {
        consumers[i] = std::thread([&]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {}
            size_t local_count = 0;
            int64_t local_sum = 0;
            while (true) {
                int value = queue.pop();
                if (value != 0) {
                    local_sum += value;
                    ++local_count;
                } else if (remaining_producers.load(
                               std::memory_order_acquire)) {
                    std::this_thread::yield();
                } else {
                    // Producers are all done (remaining_producers == 0), but
                    // items pushed just before that observation may still sit
                    // in the queue -- drain them before exiting (see helper
                    // header comment for why this pass is required).
                    while ((value = queue.pop()) != 0) {
                        local_sum += value;
                        ++local_count;
                    }
                    break;
                }
            } // consumer poll/drain loop
            total_popped.fetch_add(local_count, std::memory_order_relaxed);
            total_sum.fetch_add(local_sum, std::memory_order_relaxed);
        }); // consumer lambda
    } // spawn consumers

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    EXPECT_EQ(total_items, total_popped.load());
    EXPECT_EQ(expected_sum, total_sum.load())
        << "Sum mismatch: items were duplicated or lost";
    EXPECT_TRUE(queue.empty());
} // RunMPMCStress

// The cases below sweep queue size against thread count. Tiny queues (8 slots)
// with many threads are the important stressor: the ring wraps constantly, so
// every slot is reused thousands of times and the producer/consumer handoff
// protocol is hammered -- exactly the conditions under which a memory-ordering
// or TOCTOU bug in the slot claim shows up. Larger queues raise raw throughput
// and lock contention instead. Run under TSAN/ASAN, these convert rare races
// into deterministic failures.

// --- Tiny queue (8 slots), extreme contention ---

TEST(TSANStress, TinyQueue_MPMC_4x4) {
    RunMPMCStress<int, void, 8>(4, 4, 10000);
}

TEST(TSANStress, TinyQueue_MPMC_8x8) {
    RunMPMCStress<int, void, 8>(8, 8, 5000);
}

TEST(TSANStress, TinyQueue_MPSC_4x1) {
    RunMPMCStress<int, void, 8>(4, 1, 10000);
}

TEST(TSANStress, TinyQueue_SPMC_1x4) {
    RunMPMCStress<int, void, 8>(1, 4, 40000);
}

// --- Medium queue (64 slots) ---

TEST(TSANStress, MediumQueue_MPMC_4x4) {
    RunMPMCStress<int, void, 64>(4, 4, 50000);
}

TEST(TSANStress, MediumQueue_MPMC_8x2) {
    RunMPMCStress<int, void, 64>(8, 2, 25000);
}

TEST(TSANStress, MediumQueue_MPMC_2x8) {
    RunMPMCStress<int, void, 64>(2, 8, 100000);
}

// --- Large queue (1024 slots), many threads ---

TEST(TSANStress, LargeQueue_MPMC_16x16) {
    RunMPMCStress<int, void, 1024>(16, 16, 10000);
}

TEST(TSANStress, LargeQueue_MPMC_32x32) {
    RunMPMCStress<int, void, 1024>(32, 32, 5000);
}

TEST(TSANStress, LargeQueue_MPSC_16x1) {
    RunMPMCStress<int, void, 1024>(16, 1, 10000);
}

TEST(TSANStress, LargeQueue_SPMC_1x16) {
    RunMPMCStress<int, void, 1024>(1, 16, 160000);
}

// --- Key-Value stress (int, std::string) on tiny queue ---

// KV counterpart of RunMPMCStress. Same count+sum integrity oracle, plus a
// per-item key/value consistency check: every popped value string must equal
// std::to_string(key). value_mismatches counts any pair that came apart -- the
// signature of a torn or half-committed KV handoff -- and must end at 0. The KV
// path is the one that constructs and destroys values outside the locks, with
// only the slot's sequence number ordering the two sides, so this is where
// value corruption (as opposed to mere loss) would surface.
template <size_t Cap>
void RunMPMCStressKV(size_t producer_count, size_t consumer_count,
                     size_t items_per_producer) {
    using queue_t = RingAtomicMapQueueMPMC<int, std::string>;
    constexpr size_t bytes = Cap * queue_t::element_size();
    auto mem = std::make_unique<char[]>(bytes);
    queue_t queue(mem.get(), bytes);

    const size_t total_items = producer_count * items_per_producer;
    const int64_t expected_sum =
        static_cast<int64_t>(total_items) * (total_items + 1) / 2;

    std::atomic<int> barrier{
        static_cast<int>(producer_count + consumer_count)};
    std::atomic<int> remaining_producers{static_cast<int>(producer_count)};

    std::vector<std::thread> producers(producer_count);
    for (size_t i = 0; i != producer_count; ++i) {
        producers[i] = std::thread([&, i]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {}
            for (size_t j = 0; j < items_per_producer; ++j) {
                int value = static_cast<int>(i * items_per_producer + j + 1);
                while (!queue.push(value, std::to_string(value))) {
                    std::this_thread::yield();
                }
            }
            remaining_producers.fetch_sub(1, std::memory_order_release);
        });
    }

    std::atomic<size_t> total_popped{0};
    std::atomic<int64_t> total_sum{0};
    std::atomic<size_t> value_mismatches{0};

    std::vector<std::thread> consumers(consumer_count);
    for (size_t i = 0; i != consumer_count; ++i) {
        consumers[i] = std::thread([&]() {
            barrier.fetch_sub(1, std::memory_order_relaxed);
            while (barrier.load(std::memory_order_relaxed) != 0) {}
            size_t local_count = 0;
            int64_t local_sum = 0;
            size_t local_mismatches = 0;
            std::string s;
            while (true) {
                int value = queue.pop(s);
                if (value != 0) {
                    if (std::to_string(value) != s) {
                        ++local_mismatches;
                    }
                    local_sum += value;
                    ++local_count;
                } else if (remaining_producers.load(
                               std::memory_order_acquire)) {
                    std::this_thread::yield();
                } else {
                    // Final drain after producers finished (see RunMPMCStress).
                    while ((value = queue.pop(s)) != 0) {
                        if (std::to_string(value) != s) {
                            ++local_mismatches;
                        }
                        local_sum += value;
                        ++local_count;
                    }
                    break;
                }
            } // consumer poll/drain loop
            total_popped.fetch_add(local_count, std::memory_order_relaxed);
            total_sum.fetch_add(local_sum, std::memory_order_relaxed);
            value_mismatches.fetch_add(local_mismatches, std::memory_order_relaxed);
        }); // consumer lambda
    } // spawn consumers

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    EXPECT_EQ(total_items, total_popped.load());
    EXPECT_EQ(expected_sum, total_sum.load())
        << "Sum mismatch: items were duplicated or lost";
    EXPECT_EQ(0u, value_mismatches.load())
        << "Key-value mismatch: value corruption detected";
    EXPECT_TRUE(queue.empty());
} // RunMPMCStressKV

TEST(TSANStress, KV_TinyQueue_MPMC_4x4) {
    RunMPMCStressKV<8>(4, 4, 5000);
}

TEST(TSANStress, KV_MediumQueue_MPMC_8x8) {
    RunMPMCStressKV<64>(8, 8, 5000);
}

TEST(TSANStress, KV_LargeQueue_MPMC_16x16) {
    RunMPMCStressKV<1024>(16, 16, 5000);
}

//================================================================================
// Forced interleavings (key-value queue)
//================================================================================
// The key-value push()/pop() claim a slot under a lock and construct (resp.
// move out and destroy) the value after releasing it, so a producer and a
// consumer, or two producers one lap of the ring apart, can be inside the
// same slot's protocol at the same time. The previous protocol (key as the
// state word, a busy flag written by both sides, a key->busy->key
// validation) had a logic race in exactly that situation, reachable by two
// schedules that a stress test practically never produces: each needs a
// thread preempted between two adjacent loads for the duration of another
// thread's whole operation. The two tests below replay those schedules
// deterministically. Both fail on the previous protocol (one element lost
// and a value constructed over a live one; one element popped twice, the
// second time from a destroyed value) and pass on the sequence-number
// protocol, whose per-index state values cannot be confused across laps.
//
// Mechanism. The header's key-value push()/pop() call RING_QUEUE_HOOK(point)
// at the boundaries between protocol steps (the enum at the top of this
// file lists the points). A worker thread that owns a Gate parks inside the
// hook whenever it reaches the point the Gate names; the test (the driver)
// advances it from point to point. A hook is a pure delay -- it changes no
// load, store or memory order of the algorithm, it only decides where the
// "scheduler" preempts the thread -- so every schedule below is a legal
// execution of the unmodified protocol. Two rules for writing one: a worker
// parked at PUSH_LOADED or POP_LOADED holds the tail or head SpinLock, so no
// push (resp. pop) from the driver may run while it is parked there; and the
// steps use EXPECT, never ASSERT -- an early return would leave parked
// worker threads joinable and terminate the process; the schedule runs to
// its end and the accounting at the end is the oracle. Every wait is bounded
// (see wait_timeout): a worker that never reaches its point fails the test
// and every gate is opened, so the process exits with a failure rather than
// hanging.

// One worker thread's parking control.
//   stop_at  - the point at which the owning thread parks next (NONE: never)
//   arrived  - the owning thread is parked at stop_at
//   go       - the driver released the parked thread
//   done     - the owning thread's operation returned (set by hooked_thread)
// All fields are protected by m; cv is signaled on every change. Every Gate
// registers itself in live_gates for the duration of its lifetime, so that a
// timed-out wait can open all of them (see wait_timeout and the release-all
// loop in wait_parked_or_done()).
struct Gate {
    std::mutex m;
    std::condition_variable cv;
    RingQueueHook stop_at = RingQueueHook::NONE;
    bool arrived = false;
    bool go = false;
    bool done = false;
    Gate();
    ~Gate();
    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;
};

// The gates of the running test, in construction order (tests run one at a
// time on the main thread, which also constructs and destroys every Gate).
static std::vector<Gate*> live_gates;
Gate::Gate() { live_gates.push_back(this); }
Gate::~Gate() { live_gates.erase(std::find(live_gates.begin(), live_gates.end(), this)); }

// Bound on every driver-side wait. A schedule step normally completes in
// microseconds; a worker that has not reached its point after this long
// never will (the schedule is wrong for the protocol under test, or the
// protocol deadlocked), and the test fails instead of hanging.
static constexpr std::chrono::seconds wait_timeout {10};

// Name of a hook point, for failure messages.
static const char* hook_name(RingQueueHook p) {
    switch (p) {
        case RingQueueHook::NONE: return "NONE";
        case RingQueueHook::PUSH_LOADED: return "PUSH_LOADED";
        case RingQueueHook::PUSH_CLAIMED: return "PUSH_CLAIMED";
        case RingQueueHook::PUSH_COMMITTING: return "PUSH_COMMITTING";
        case RingQueueHook::POP_LOADED: return "POP_LOADED";
        case RingQueueHook::POP_CLAIMED: return "POP_CLAIMED";
        case RingQueueHook::POP_RELEASING: return "POP_RELEASING";
    } // switch on the point
    return "?";
} // hook_name()

// The Gate of the current thread, if it is a hooked worker; null for the
// driver and for every thread of every other test.
static thread_local Gate* my_gate = nullptr;

// Called by the queue at every hook point (through RING_QUEUE_HOOK). Parks
// only if this thread owns a Gate that names this point; otherwise returns
// at once.
static void ring_queue_hook(RingQueueHook point) {
    Gate* g = my_gate;
    if (!g) { return; }
    std::unique_lock<std::mutex> l(g->m);
    if (g->stop_at != point) { return; }
    g->arrived = true;
    g->cv.notify_all();
    g->cv.wait(l, [&] { return g->go; });
    g->go = false;
} // ring_queue_hook()

// Start op() on a thread owned by Gate g, which will park at point `first`.
// The thread marks g done when op() returns, so a driver waiting for a point
// the thread never reaches (its operation took another path) is released too.
template <typename F>
std::thread hooked_thread(Gate& g, RingQueueHook first, F op) {
    g.stop_at = first;      // before the thread exists: no race
    return std::thread([&g, op = std::move(op)]() {
        my_gate = &g;
        op();
        std::lock_guard<std::mutex> l(g.m);
        g.done = true;
        g.cv.notify_all();
    });
} // hooked_thread()

// Driver: release g's parked thread and let it run to completion (join it
// afterwards).
static void release(Gate& g) {
    std::unique_lock<std::mutex> l(g.m);
    g.arrived = false;
    g.stop_at = RingQueueHook::NONE;
    g.go = true;
    g.cv.notify_all();
} // release()

// Driver: the wait behind wait_arrived() and run_to(), with g.m held by l.
// Waits until g's thread is parked (arrived) or finished (done), for at most
// wait_timeout. On timeout: records a test failure naming the point, opens
// every live gate (so all workers run to completion and the joins return),
// and returns false; the schedule then runs to its end with nothing parked
// and the accounting at the end reports whatever happened.
static bool wait_parked_or_done(Gate& g, std::unique_lock<std::mutex>& l, RingQueueHook p) {
    if (g.cv.wait_for(l, wait_timeout, [&] { return g.arrived || g.done; })) { return true; }
    ADD_FAILURE() << "worker never reached " << hook_name(p) << " within " << wait_timeout.count() << " s";
    l.unlock();
    for (Gate* other : live_gates) { release(*other); }
    return false;
} // wait_parked_or_done()

// Driver: wait until g's thread is parked at its current stop_at, or done.
// Returns false (after failing the test and opening every gate) on timeout.
static bool wait_arrived(Gate& g) {
    std::unique_lock<std::mutex> l(g.m);
    return wait_parked_or_done(g, l, g.stop_at);
} // wait_arrived()

// Driver: release g's parked thread, let it run to point p, and wait until
// it parks there (or finishes without reaching p). Returns false (after
// failing the test and opening every gate) on timeout.
// Precondition: g's thread is parked (arrived) or finished (done) -- i.e. the
// previous wait on g returned. Releasing a thread that is not parked would
// consume its next park instead, and the schedule would silently drift; the
// precondition is checked and reported.
static bool run_to(Gate& g, RingQueueHook p) {
    std::unique_lock<std::mutex> l(g.m);
    EXPECT_TRUE(g.arrived || g.done) << "run_to(" << hook_name(p) << ") on a thread that is not parked";
    g.arrived = false;
    g.stop_at = p;
    g.go = true;
    g.cv.notify_all();
    return wait_parked_or_done(g, l, p);
} // run_to()

// Per-slot live-count oracle. A correct protocol constructs exactly one value
// in a slot, then destroys it, then constructs the next: the slot's count of
// live values goes 0 -> 1 -> 0 -> 1 ... A double claim by two producers makes
// it go 1 -> 2, a double pop makes it go 0 -> -1; the value type below
// reports either as a violation the moment it happens, from inside the
// constructor or destructor that broke the sequence, which is the earliest
// and most precise point at which a protocol defect can be observed. The
// oracle maps a value's address to its slot, so it watches one queue buffer
// at a time (arm() before constructing the queue, disarm() after destroying
// it); values that live elsewhere (push arguments, pop out-parameters) are
// outside the buffer and ignored.
//   base, stride, slots - the watched buffer: start, element_size(), capacity
//   live                - live[p] is the number of constructed values in slot p
//   violations          - number of 0->1 / 1->0 transitions that did not hold
struct SlotOracle {
    static inline std::uintptr_t base = 0;
    static inline size_t stride = 0;
    static inline size_t slots = 0;
    static inline std::atomic<int>* live = nullptr;
    static inline std::atomic<size_t> violations {0};

    // Watch the buffer [buffer, buffer + stride*slots), with live[] (at least
    // slots entries, all zero) as the counters.
    static void arm(const void* buffer, size_t stride_, size_t slots_, std::atomic<int>* live_) {
        base = reinterpret_cast<std::uintptr_t>(buffer);
        stride = stride_;
        slots = slots_;
        live = live_;
        violations.store(0, std::memory_order_relaxed);
    } // SlotOracle::arm()
    // Stop watching: every value constructed or destroyed from now on is
    // ignored (until the next arm()).
    static void disarm() { base = 0; stride = 0; slots = 0; live = nullptr; }

    // Slot index of the object at address a, or slots if it is outside the
    // watched buffer.
    static size_t slot_of(const void* a) {
        std::uintptr_t const x = reinterpret_cast<std::uintptr_t>(a);
        if (base == 0 || x < base || x >= base + stride*slots) { return slots; }
        return (x - base)/stride;
    } // SlotOracle::slot_of()
    // Record a value constructed at address a: its slot's count must go 0 -> 1.
    static void constructed(const void* a) {
        size_t const p = slot_of(a);
        if (p == slots) { return; }
        if (live[p].fetch_add(1, std::memory_order_relaxed) != 0) { violations.fetch_add(1, std::memory_order_relaxed); }
    } // SlotOracle::constructed()
    // Record a value destroyed at address a: its slot's count must go 1 -> 0.
    static void destroyed(const void* a) {
        size_t const p = slot_of(a);
        if (p == slots) { return; }
        if (live[p].fetch_sub(1, std::memory_order_relaxed) != 1) { violations.fetch_add(1, std::memory_order_relaxed); }
    } // SlotOracle::destroyed()
}; // SlotOracle

// Value type for the oracle: reports its construction and destruction to
// SlotOracle, carries a unique payload, and owns a heap block (text is longer
// than the small-string buffer) so that a value constructed over a live one
// also leaks, and a value moved out of a destroyed one double-frees --
// ASan/LSan then flag the same defects the oracle counts.
//   payload - the item's identity (unique per push in the tests below)
//   text    - std::to_string(payload) padded past the SSO limit; checked
//             against payload on every pop to detect a torn value
// The move constructor, move assignment and destructor are noexcept, as
// push()/pop() require.
struct TrackedValue {
    int64_t payload;
    std::string text;
    // The text a value with payload p carries: to_string(p) plus 40 'x'.
    static std::string text_of(int64_t p) { return std::to_string(p) + std::string(40, 'x'); }
    explicit TrackedValue(int64_t p) : payload(p), text(text_of(p)) { SlotOracle::constructed(this); }
    TrackedValue(TrackedValue&& o) noexcept : payload(o.payload), text(std::move(o.text)) { o.payload = 0; SlotOracle::constructed(this); }
    TrackedValue& operator=(TrackedValue&& o) noexcept { payload = o.payload; text = std::move(o.text); o.payload = 0; return *this; }
    ~TrackedValue() noexcept { SlotOracle::destroyed(this); }
    // True if text is what text_of(payload) produces: the value was neither
    // torn nor moved out of a destroyed slot.
    bool consistent() const { return text == text_of(payload); }
}; // TrackedValue

// A capacity-8 key-value queue of TrackedValue over an oracle-watched buffer,
// plus the push/pop accounting the forced-interleaving tests check at the end.
// The queue is owned through a unique_ptr so that a test can deliberately
// LEAK it when the oracle reports a violation: a protocol that double-claimed
// a slot has also broken the head <= tail invariant the destructor's drain
// loop relies on (on the previous protocol that loop then runs ~2^64 times).
//   pushed / popped - number of push() calls that returned true / pop() calls
//                     that returned a key
//   pushed_sum / popped_sum - sums of the payloads of those elements
struct ForcedQueue {
    // Default NTRY (8). PushSideLapAhead needs NTRY >= 3: its lap-ahead
    // producer loads the slot state three times (free-for-the-previous-lap,
    // committed, free-for-this-lap) before it is allowed to claim.
    using queue_t = RingAtomicMapQueueMPMC<int, TrackedValue>;
    static constexpr size_t capacity = 8;
    static constexpr size_t bytes = capacity*queue_t::element_size();
    std::unique_ptr<char[]> memory { new char[bytes] };
    std::atomic<int> live[capacity] {};
    std::unique_ptr<queue_t> q;
    std::atomic<size_t> pushed {0}, popped {0};
    std::atomic<int64_t> pushed_sum {0}, popped_sum {0};

    ForcedQueue() {
        SlotOracle::arm(memory.get(), queue_t::element_size(), capacity, live);
        q.reset(new queue_t(memory.get(), bytes));
    } // ForcedQueue()
    ~ForcedQueue() {
        if (SlotOracle::violations.load() == 0 && pushed.load() == popped.load()) {
            q.reset();      // the queue is consistent: destroy it normally
        } else {
            (void)q.release();  // broken invariants: leak rather than hang in the drain loop
        }
        SlotOracle::disarm();
    } // ~ForcedQueue()

    // push(key, TrackedValue(payload)), recording the result.
    bool push(int key, int64_t payload) {
        bool const ok = q->push(key, TrackedValue(payload));
        if (ok) { pushed.fetch_add(1); pushed_sum.fetch_add(payload); }
        return ok;
    } // ForcedQueue::push()
    // pop(), recording the result and checking the value's integrity; returns
    // the key. `expected_key` (if not 0) must be the key that comes back.
    int pop(int expected_key = 0) {
        TrackedValue v(0);
        int const key = q->pop(v);
        if (key != 0) {
            popped.fetch_add(1); popped_sum.fetch_add(v.payload);
            EXPECT_TRUE(v.consistent()) << "torn value: payload " << v.payload << " text " << v.text;
            if (expected_key != 0) { EXPECT_EQ(expected_key, key); }
        }
        return key;
    } // ForcedQueue::pop()
    // The end-of-test oracle: every push that returned true was popped exactly
    // once, and no slot ever held two live values or destroyed a dead one.
    void check() const {
        EXPECT_EQ(pushed.load(), popped.load()) << "elements lost or duplicated";
        EXPECT_EQ(pushed_sum.load(), popped_sum.load()) << "elements lost or duplicated";
        EXPECT_EQ(0u, SlotOracle::violations.load()) << "a slot held two live values or destroyed a dead one";
    } // ForcedQueue::check()
}; // ForcedQueue

// Schedule 1 (push side): a producer one lap ahead reads a slot while the
// previous lap's producer commits it and a consumer pops it. Capacity 8, so
// index 8 is slot 0 again. Keys equal payloads here; the keys are distinct,
// this schedule needs no repeated key.
//
// On the previous protocol P1's three loads (key, busy, key) straddled P0's
// commit (key 0 -> 1) and C's pop (key 1 -> 0): the validation passed on a
// slot that was never stably empty, P1 claimed it, C's trailing busy=0 store
// clobbered P1's claim, and a third producer wrapping around claimed the same
// slot again -- two producers owned one slot, one element was lost and a
// value was constructed over a live one. On the sequence-number protocol P1
// compares against seq == 8, a value the slot reaches only as C's LAST act;
// P1 sees "full" until then, and claims index 8 only once C is entirely
// out of the slot -- after which the ring is full again and the extra push
// of step 10 is the one that is refused.
TEST(ForcedInterleaving, PushSideLapAhead) {
    ForcedQueue f;
    Gate gP0, gP1, gC;
    bool p0_ok = false, p1_ok = false;
    // 1. P0 claims index 0 (slot 0) -- key written, tail advanced, lock
    //    released -- and stalls before constructing its value.
    std::thread tP0 = hooked_thread(gP0, RingQueueHook::PUSH_CLAIMED, [&] { p0_ok = f.q->push(1, TrackedValue(1)); });
    wait_arrived(gP0);
    // 2. Other producers fill indices 1..7; the tail is now at index 8.
    //    Consumers cannot get past slot 0 (nothing committed there yet).
    for (int k = 2; k <= 8; ++k) { EXPECT_TRUE(f.push(k, k)); }
    // 3. P1 takes the tail lock at index 8 (slot 0), reads the slot's state --
    //    P0 has not committed -- and stalls before deciding.
    std::thread tP1 = hooked_thread(gP1, RingQueueHook::PUSH_LOADED, [&] { p1_ok = f.q->push(9, TrackedValue(9)); });
    wait_arrived(gP1);
    // 4. P0 commits: slot 0 holds element 1.
    release(gP0); tP0.join();
    // 5. P1 reads the slot's state again and stalls (previous protocol: this is
    //    its busy load, which sees 0 now that P0 is done; here: a retry of the
    //    seq load, which sees 1 == "committed for index 0", not "free for 8").
    run_to(gP1, RingQueueHook::PUSH_LOADED);
    // 6. C pops index 0: claims, moves the value out, destroys it, and runs
    //    to the last step of its release (previous protocol: the key is
    //    cleared, the busy flag not yet; here: the release is one store and C
    //    has made it, so C simply finishes).
    std::thread tC = hooked_thread(gC, RingQueueHook::POP_RELEASING, [&] { f.pop(1); });
    wait_arrived(gC);
    run_to(gC, RingQueueHook::POP_RELEASING);
    // 7. P1 decides. Previous protocol: its second key load sees Key{} (C has
    //    cleared it), the validation passes although the slot was never
    //    stable, P1 claims slot 0 for index 8 and stalls before constructing.
    //    Here: P1's retry sees seq == 8, which C stored as its LAST act, and
    //    claims index 8 legitimately -- at no earlier point did the slot look
    //    free for index 8.
    run_to(gP1, RingQueueHook::PUSH_CLAIMED);
    // 8. C finishes (previous protocol: its busy=0 clobbers P1's busy=1 and
    //    slot 0 looks free although P1 owns it).
    release(gC); tC.join();
    // 9. The ring wraps once more while P1 is still parked mid-commit: pop the
    //    seven elements of indices 1..7, push seven (indices 9..15).
    for (int k = 2; k <= 8; ++k) { EXPECT_EQ(k, f.pop(k)); }
    for (int k = 10; k <= 16; ++k) { EXPECT_TRUE(f.push(k, k)); }
    // 10. One more push, at index 16 = slot 0. Previous protocol: the slot
    //     looks free, the push constructs a value in a slot P1 owns and
    //     returns true. Here: the slot holds P1's uncommitted element of
    //     index 8 (seq == 8, not 16), the ring is full, the push returns
    //     false. The result is recorded, not asserted: it is the divergence.
    f.push(17, 17);
    // 11. P1 completes (previous protocol: constructs over the live value of
    //     step 10 -- the oracle sees 1 -> 2 -- and commits key 9 over key 17).
    release(gP1); tP1.join();
    if (p0_ok) { f.pushed.fetch_add(1); f.pushed_sum.fetch_add(1); }
    if (p1_ok) { f.pushed.fetch_add(1); f.pushed_sum.fetch_add(9); }
    EXPECT_TRUE(p0_ok);
    EXPECT_TRUE(p1_ok);     // step 7: P1 claims index 8 once C has released the slot
    // 12. Drain and account.
    while (f.pop() != 0) {}
    EXPECT_TRUE(f.q->empty());
    f.check();
} // ForcedInterleaving.PushSideLapAhead

// Schedule 2 (pop side): a consumer one lap ahead reads a slot while the
// previous lap's consumer finishes and a producer refills it with the SAME
// key. Every element here has key 5 (keys need not be unique; the payloads
// are unique). Capacity 8, so index 8 is slot 0 again.
//
// On the previous protocol C1 read C0's stale key (5), then busy == 0 after
// C0 finished, then key == 5 again after P refilled the slot with key 5: the
// key->busy->key validation passed although the slot had gone through empty
// in between. C1 claimed the slot, P's trailing busy=0 store clobbered C1's
// claim, and after another lap a consumer found the slot stable with key 5
// and popped it again -- from a value C1 had already destroyed. On the
// sequence-number protocol C1 compares against seq == 9, which the slot
// reaches only when P commits; C1 saw 1 (C0's lap) and returns empty.
TEST(ForcedInterleaving, PopSideRepeatedKey) {
    ForcedQueue f;
    Gate gC0, gC1, gP;
    bool p_ok = false;
    int c1_key = -1;        // what C1's pop returned; must be Key{} (0), see step 6
    constexpr int key = 5;
    // 1. One element (payload 1) at index 0, slot 0.
    EXPECT_TRUE(f.push(key, 1));
    // 2. C0 pops it: reads and clears the key, claims, unlocks, moves the
    //    value out, destroys it, and stalls before releasing the slot
    //    (previous protocol: key 5 still visible).
    std::thread tC0 = hooked_thread(gC0, RingQueueHook::POP_RELEASING, [&] { f.pop(key); });
    wait_arrived(gC0);
    // 3. The ring wraps: push seven (indices 1..7), pop seven; the head is at
    //    index 8, slot 0 again.
    for (int p = 2; p <= 8; ++p) { EXPECT_TRUE(f.push(key, p)); }
    for (int p = 2; p <= 8; ++p) { EXPECT_EQ(key, f.pop(key)); }
    // 4. C1 takes the head lock at index 8, reads slot 0's state -- C0 is still
    //    inside it -- and stalls before deciding.
    std::thread tC1 = hooked_thread(gC1, RingQueueHook::POP_LOADED, [&] { c1_key = f.pop(key); });
    wait_arrived(gC1);
    // 5. C0 finishes: slot 0 is free for index 8.
    release(gC0); tC0.join();
    // 6. C1 reads the slot's state again and stalls (previous protocol: its
    //    busy load, 0 now that C0 is done; here: C1 has already decided on the
    //    state it loaded in step 4 -- seq 1, not 9 -- and returns empty, so it
    //    finishes and this call returns at once).
    run_to(gC1, RingQueueHook::POP_LOADED);
    // 7. P pushes key 5 (payload 9) at index 8 into slot 0: writes the key and
    //    claims under the lock, constructs, and stalls before its commit store.
    std::thread tP = hooked_thread(gP, RingQueueHook::PUSH_COMMITTING, [&] { p_ok = f.q->push(key, TrackedValue(9)); });
    wait_arrived(gP);
    // 8. C1 decides. Previous protocol: its second key load sees 5 == the key
    //    it read in step 4, the validation passes, C1 claims index 8, moves
    //    P's value out, destroys it and stalls before clearing the key. Here:
    //    C1 is done (step 6), nothing happens.
    run_to(gC1, RingQueueHook::POP_RELEASING);
    // 9. P completes its commit (previous protocol: busy=0 clobbers C1's
    //    busy=1; slot 0 now reads as a stable element with key 5 whose value
    //    C1 has destroyed).
    release(gP); tP.join();
    if (p_ok) { f.pushed.fetch_add(1); f.pushed_sum.fetch_add(9); }
    EXPECT_TRUE(p_ok);
    // 10. Another lap: push seven, pop seven.
    for (int p = 10; p <= 16; ++p) { EXPECT_TRUE(f.push(key, p)); }
    for (int p = 10; p <= 16; ++p) { EXPECT_EQ(key, f.pop(key)); }
    // 11. One more pop. Previous protocol: index 16 is slot 0, which looks
    //     like a stable element with key 5: popped a second time, from the
    //     destroyed value (the oracle sees 0 -> -1; ASan sees the double
    //     free). Here: index 15, the last element of step 10.
    f.pop(key);
    // 12. C1 completes. It saw seq == 1 at index 8 -- the element of index 0
    //     being drained, nothing committed for index 8 -- so its answer is
    //     "empty". (On the previous protocol it returned key 5: P's element,
    //     legitimately, but by a path that also let it be popped twice.)
    release(gC1); tC1.join();
    EXPECT_EQ(0, c1_key);
    // 13. Drain and account.
    while (f.pop(key) != 0) {}
    EXPECT_TRUE(f.q->empty());
    f.check();
} // ForcedInterleaving.PopSideRepeatedKey

//================================================================================
// Repeated-key stress test with the per-slot live-count oracle
//================================================================================
// KV stress in the conditions the forced schedules above need: a tiny ring
// that wraps constantly, keys that repeat (three keys for the whole run, so
// every slot sees the same key again and again -- the previous protocol's
// validation compared keys), and consumers that back off when they see empty
// (which holds the head back while producers wrap, widening the window in
// which a lap-ahead producer meets the previous lap's commit and pop). The
// oracle is the same as above: every slot must alternate one construction
// with one destruction, and every push that returned true must be popped
// exactly once, with its key and payload intact. Unlike the forced schedules
// this test reaches a bad interleaving only by luck; it is here so that luck
// gets a chance on every run, under both sanitizers.
//   producer_count / consumer_count / items_per_producer - as in RunMPMCStress
template <size_t Cap>
void RunMPMCStressKVRepeatedKeys(size_t producer_count, size_t consumer_count,
                                 size_t items_per_producer) {
    using queue_t = RingAtomicMapQueueMPMC<int, TrackedValue>;
    constexpr size_t bytes = Cap*queue_t::element_size();
    std::unique_ptr<char[]> mem(new char[bytes]);
    std::atomic<int> live[Cap] {};
    SlotOracle::arm(mem.get(), queue_t::element_size(), Cap, live);
    {
        queue_t queue(mem.get(), bytes);

        const size_t total_items = producer_count*items_per_producer;
        const int64_t expected_sum = static_cast<int64_t>(total_items)*(total_items + 1)/2;
        // Three keys for the whole run: key_of(payload) is what pop must pair
        // with the payload.
        auto key_of = [](int64_t payload) { return static_cast<int>(1 + payload % 3); };

        std::atomic<int> barrier {static_cast<int>(producer_count + consumer_count)};
        std::atomic<int> remaining_producers {static_cast<int>(producer_count)};

        std::vector<std::thread> producers(producer_count);
        for (size_t i = 0; i != producer_count; ++i) {
            producers[i] = std::thread([&, i]() {
                barrier.fetch_sub(1, std::memory_order_relaxed);
                while (barrier.load(std::memory_order_relaxed) != 0) {}
                for (size_t j = 0; j < items_per_producer; ++j) {
                    int64_t const payload = static_cast<int64_t>(i*items_per_producer + j + 1);
                    while (!queue.push(key_of(payload), TrackedValue(payload))) {
                        std::this_thread::yield();
                    }
                }
                remaining_producers.fetch_sub(1, std::memory_order_release);
            });
        } // spawn producers

        std::atomic<size_t> total_popped {0};
        std::atomic<int64_t> total_sum {0};
        std::atomic<size_t> mismatches {0};

        std::vector<std::thread> consumers(consumer_count);
        for (size_t i = 0; i != consumer_count; ++i) {
            consumers[i] = std::thread([&]() {
                barrier.fetch_sub(1, std::memory_order_relaxed);
                while (barrier.load(std::memory_order_relaxed) != 0) {}
                size_t local_count = 0;
                int64_t local_sum = 0;
                size_t local_mismatches = 0;
                TrackedValue v(0);
                // Account for one popped element: key/payload pairing and
                // value integrity.
                auto consume = [&](int key) {
                    if (key != key_of(v.payload) || !v.consistent()) { ++local_mismatches; }
                    local_sum += v.payload;
                    ++local_count;
                };
                while (true) {
                    int const key = queue.pop(v);
                    if (key != 0) {
                        consume(key);
                    } else if (remaining_producers.load(std::memory_order_acquire)) {
                        // Back off: a real sleep, not a yield, so the ring
                        // fills and wraps behind a slow head.
                        std::this_thread::sleep_for(std::chrono::microseconds(20));
                    } else {
                        // Final drain after producers finished (see RunMPMCStress).
                        int k;
                        while ((k = queue.pop(v)) != 0) { consume(k); }
                        break;
                    }
                } // consumer poll/drain loop
                total_popped.fetch_add(local_count, std::memory_order_relaxed);
                total_sum.fetch_add(local_sum, std::memory_order_relaxed);
                mismatches.fetch_add(local_mismatches, std::memory_order_relaxed);
            }); // consumer lambda
        } // spawn consumers

        for (auto& t : producers) t.join();
        for (auto& t : consumers) t.join();

        EXPECT_EQ(total_items, total_popped.load());
        EXPECT_EQ(expected_sum, total_sum.load()) << "Sum mismatch: items were duplicated or lost";
        EXPECT_EQ(0u, mismatches.load()) << "Key/payload mismatch or torn value";
        EXPECT_EQ(0u, SlotOracle::violations.load()) << "a slot held two live values or destroyed a dead one";
        EXPECT_TRUE(queue.empty());
    } // queue lifetime: destroyed (drained) while the oracle is still armed
    SlotOracle::disarm();
} // RunMPMCStressKVRepeatedKeys

TEST(TSANStress, KV_RepeatedKeys_TinyQueue_MPMC_4x4) {
    RunMPMCStressKVRepeatedKeys<8>(4, 4, 5000);
}

TEST(TSANStress, KV_RepeatedKeys_TinyQueue_MPMC_8x2) {
    RunMPMCStressKVRepeatedKeys<8>(8, 2, 2500);
}

TEST(TSANStress, KV_RepeatedKeys_MediumQueue_MPMC_8x8) {
    RunMPMCStressKVRepeatedKeys<64>(8, 8, 5000);
}
