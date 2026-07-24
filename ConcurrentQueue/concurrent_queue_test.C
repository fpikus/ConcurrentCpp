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
#include "concurrent_queue.h"

#include <atomic>
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
//      lock-free slot handoff (key/busy protocol in the header) is most likely
//      to race. Run under the sanitizers, these turn rare interleavings into
//      reproducible failures.
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

// KV queue: the slot now carries a value plus the busy flag (the key-only
// specialization drops busy). These tests exercise the value-returning pop(val)
// overload, which writes the value by move-assignment into the caller's object.
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

// Property: push(key, lvalue) COPY-constructs the value in the slot; the
// caller's source string is left unchanged (still "hello"). This distinguishes
// the copy path from the move path tested next -- push takes V&& and forwards,
// so an lvalue binds to the copy constructor.
TEST_F(QueueTesterIntString, PushAndPop) {
    std::string v_in = "hello";
    bool pushed = queue_.push(1, v_in);
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
    EXPECT_FALSE(queue_.push(99, "should_fail"));
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
            if (!running_under_TAP) EXPECT_NE(0u, consumed_items.size()) << "Consumer " << i << " did not pop any elements";
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
// caught by EXPECT_EQ(value, std::stoi(s)). This is what the busy flag and the
// key->busy->key validation in the header exist to prevent.

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
            if (!running_under_TAP) EXPECT_NE(0u, consumed_items.size()) << "Consumer " << i << " did not pop any elements";
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
// double-destroys a slot value.
struct CountedValue {
    static inline std::atomic<int> destructor_count {0};
    static inline std::atomic<int> constructor_count {0};
    int x;
    CountedValue() : x(0) { constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue(int v) : x(v) { constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue(const CountedValue& o) : x(o.x) { constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue(CountedValue&& o) : x(o.x) { o.x = 0; constructor_count.fetch_add(1, std::memory_order_relaxed); }
    CountedValue& operator=(CountedValue&& o) { x = o.x; o.x = 0; return *this; }
    ~CountedValue() { destructor_count.fetch_add(1, std::memory_order_relaxed); }
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
// into a move-construction and pop uses move-assignment.
struct MoveOnly {
    int x;
    MoveOnly() : x(0) {}
    MoveOnly(int v) : x(v) {}
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&& o) : x(o.x) { o.x = 0; }
    MoveOnly& operator=(MoveOnly&& o) { x = o.x; o.x = 0; return *this; }
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
// Element Size and Alignment Tests
//================================================================================

// Property: the caller can size a buffer from element_size() before knowing
// the queue's internal slot layout. A key-only slot is at least the size of
// its atomic key (the key-only specialization stores only the key).
TEST(ElementSizeTest, KeyOnly) {
    using queue_t = RingAtomicMapQueueMPMC<int>;
    EXPECT_GE(queue_t::element_size(), sizeof(std::atomic<int>));
}

// Property: a KV slot holds at least key + value (plus the busy flag and any
// padding, hence >= rather than ==).
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
// path is the one that uses the busy flag and key->busy->key validation, so
// this is where value corruption (as opposed to mere loss) would surface.
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
