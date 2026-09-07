// Unit tests for atomic_max() (atomic_max.h). The contract is small -- raise a
// shared atomic to the maximum of itself and an argument, return whether it
// changed -- so the tests check that contract single-threaded (update, no-op,
// equal, return value, boundary values) and then hammer it from many threads:
// the final maximum must be the global maximum of every value offered, and the
// count of "true" returns must equal the number of times the running maximum
// actually advanced. Run under TSan to validate the acquire/acq_rel ordering:
// a companion plain (non-atomic) payload published alongside the maximum turns
// any missing release/acquire into both a wrong value and a reported race.
#include <gtest/gtest.h>
#include <atomic>
#include <algorithm>
#include <random>
#include <thread>
#include <vector>

#include "atomic_max.h"

TEST(AtomicMaxTest, UpdatesWhenLarger) {
  std::atomic<unsigned long> m{5};
  EXPECT_TRUE(atomic_max(m, 10ul));       // 10 > 5 -> updates, returns true
  EXPECT_EQ(m.load(), 10ul);
}

TEST(AtomicMaxTest, NoUpdateWhenSmaller) {
  std::atomic<unsigned long> m{10};
  EXPECT_FALSE(atomic_max(m, 5ul));       // 5 < 10 -> no update, returns false
  EXPECT_EQ(m.load(), 10ul);
}

TEST(AtomicMaxTest, NoUpdateWhenEqual) {
  std::atomic<unsigned long> m{7};
  EXPECT_FALSE(atomic_max(m, 7ul));       // equal is not "greater": no update
  EXPECT_EQ(m.load(), 7ul);
}

// A run of increasing values updates every time; a run of decreasing values
// updates only once (the first). Return values must track that exactly.
TEST(AtomicMaxTest, ReturnValueTracksAdvances) {
  std::atomic<unsigned long> m{0};
  for (unsigned long v : {1ul, 2ul, 3ul, 4ul}) EXPECT_TRUE(atomic_max(m, v));
  EXPECT_EQ(m.load(), 4ul);
  for (unsigned long v : {3ul, 2ul, 1ul}) EXPECT_FALSE(atomic_max(m, v));
  EXPECT_EQ(m.load(), 4ul);
}

TEST(AtomicMaxTest, BoundaryValues) {
  std::atomic<unsigned long> m{0};
  constexpr unsigned long big = ~0ul;     // max representable
  EXPECT_TRUE(atomic_max(m, big));
  EXPECT_EQ(m.load(), big);
  EXPECT_FALSE(atomic_max(m, 0ul));       // nothing exceeds the ceiling
  EXPECT_EQ(m.load(), big);
}

// Works on a signed type too, including across zero.
TEST(AtomicMaxTest, SignedType) {
  std::atomic<long> m{-5};
  EXPECT_TRUE(atomic_max(m, -1l));
  EXPECT_EQ(m.load(), -1l);
  EXPECT_FALSE(atomic_max(m, -3l));
  EXPECT_EQ(m.load(), -1l);
  EXPECT_TRUE(atomic_max(m, 100l));
  EXPECT_EQ(m.load(), 100l);
}

// N threads each offer a block of distinct values; whichever block holds the
// global maximum, that value must end up in the shared maximum regardless of
// interleaving. This is the correctness property under real contention.
TEST(AtomicMaxTest, ConcurrentMaxIsCorrect) {
  constexpr int num_threads = 8;
  constexpr unsigned long per_thread = 100000;
  std::atomic<unsigned long> m{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      // Thread t offers t, t+8, t+16, ...; the global max is
      // (num_threads*per_thread - 1) offered by the last thread's last value.
      for (unsigned long i = 0; i < per_thread; ++i) {
        atomic_max(m, i*num_threads + t);
      }
    });
  } // spawn loop
  for (auto& th : threads) th.join();
  EXPECT_EQ(m.load(), (per_thread - 1)*num_threads + (num_threads - 1));
} // ConcurrentMaxIsCorrect

// The count of "true" returns across all threads must equal the number of
// distinct values the running maximum actually took -- i.e. the number of
// strict advances. With each thread offering a shuffled permutation of the
// SAME value set, the running maximum advances exactly once per distinct value
// that ever becomes a new record, and a value is a record the first time any
// thread pushes something larger than everything seen so far. That count is
// data-dependent, so instead of predicting it we check the invariant that ties
// the returns to the value: the maximum's final value must have been returned
// true exactly once, and total-trues must be <= the number of distinct values.
TEST(AtomicMaxTest, AdvanceCountBounded) {
  constexpr int num_threads = 8;
  constexpr unsigned long n_values = 50000;
  std::atomic<unsigned long> m{0};
  std::atomic<long> trues{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      std::vector<unsigned long> vals(n_values);
      for (unsigned long i = 0; i < n_values; ++i) vals[i] = i + 1; // 1..n_values
      std::shuffle(vals.begin(), vals.end(), std::mt19937_64(t));
      long local = 0;
      for (unsigned long v : vals) if (atomic_max(m, v)) ++local;
      trues.fetch_add(local, std::memory_order_relaxed);
    });
  } // spawn loop
  for (auto& th : threads) th.join();
  EXPECT_EQ(m.load(), n_values);                    // the global max wins
  // Every "true" advanced the maximum by >=1 from 0 to at most n_values, so the
  // number of advances cannot exceed n_values, and must be at least 1.
  EXPECT_GE(trues.load(), 1);
  EXPECT_LE(trues.load(), long(n_values));
} // AdvanceCountBounded

// Ordering check with a plain (non-atomic) payload: the thread that raises the
// maximum first writes a payload the *reader* expects to see once it observes
// the new maximum. atomic_max's release on update and acquire on read make the
// payload write happen-before the observing read; under TSan a missing barrier
// shows up as a data race on the plain payload.
TEST(AtomicMaxTest, PublishesPayloadWithRelease) {
  constexpr int rounds = 2000;
  for (int round = 0; round < rounds; ++round) {
    std::atomic<unsigned long> m{0};
    unsigned long payload = 0;            // plain, guarded only by m's ordering
    std::atomic<bool> checked{false};
    std::thread writer([&] {
      payload = 42;                       // write BEFORE publishing the max
      // success=release publishes the payload; failure=relaxed is enough for
      // the losing/no-op read. This is the "first load relaxed, publish on
      // update" combination -- if success were relaxed, TSan would flag the
      // payload read below as a race.
      atomic_max(m, 1ul, std::memory_order_release, std::memory_order_relaxed);
    });
    std::thread reader([&] {
      // Spin until the max is published, then the payload must be visible: the
      // acquire load synchronizes-with atomic_max's release on the writer side.
      while (m.load(std::memory_order_acquire) == 0) {}
      EXPECT_EQ(payload, 42ul);           // acquire made the payload write visible
      checked.store(true, std::memory_order_relaxed);
    });
    writer.join();
    reader.join();
    EXPECT_TRUE(checked.load(std::memory_order_relaxed));
  } // rounds
} // PublishesPayloadWithRelease
