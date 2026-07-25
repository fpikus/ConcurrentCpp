// Unit tests for the TTAS SpinLock (spinlock.h). Mostly pro-forma: the
// interface is tiny, so the tests check the interface contract (lock/unlock/
// try_lock/locked, std::lock_guard and std::unique_lock compatibility) plus
// mutual exclusion under real thread contention. Run under TSan to validate
// the release/acquire handoff: the counters below are deliberately plain
// (non-atomic), so any mutual-exclusion or ordering bug is both a wrong count
// and a reported data race.
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "spinlock.h"

TEST(SpinLockTest, InitiallyUnlocked) {
  SpinLock lock;
  EXPECT_FALSE(lock.locked());
}

TEST(SpinLockTest, LockUnlock) {
  SpinLock lock;
  lock.lock();
  EXPECT_TRUE(lock.locked());
  lock.unlock();
  EXPECT_FALSE(lock.locked());
}

TEST(SpinLockTest, TryLockWhenFree) {
  SpinLock lock;
  EXPECT_TRUE(lock.try_lock());
  EXPECT_TRUE(lock.locked());
  lock.unlock();
  EXPECT_FALSE(lock.locked());
}

// try_lock() against a lock held by another thread must give up and return
// false. It is a bounded spin, not a single-shot try (see spinlock.h), so the
// worker takes a few short sleeps before failing; the lock is held for the
// whole test, so false is guaranteed.
TEST(SpinLockTest, TryLockWhenHeld) {
  SpinLock lock;
  lock.lock();
  bool acquired = true;
  std::thread worker([&] { acquired = lock.try_lock(); });
  worker.join();
  EXPECT_FALSE(acquired);
  EXPECT_TRUE(lock.locked());           // still held by this thread
  lock.unlock();
}

// SpinLock satisfies the standard Lockable requirements: std::lock_guard and
// std::unique_lock (including try_to_lock) must work with it.
TEST(SpinLockTest, StdGuardCompatibility) {
  SpinLock lock;
  {
    std::lock_guard g(lock);
    EXPECT_TRUE(lock.locked());
  }
  EXPECT_FALSE(lock.locked());
  {
    std::unique_lock ul(lock, std::try_to_lock);
    EXPECT_TRUE(ul.owns_lock());
    EXPECT_TRUE(lock.locked());
  }
  EXPECT_FALSE(lock.locked());
}

// A waiter blocked in lock() must acquire the lock once the holder releases
// it. The holder keeps the lock for ~50 ms -- long enough for the waiter to
// exhaust its short-yield budget and reach the long (1 ms) sleep tier -- so
// this also exercises the back-off escalation path.
TEST(SpinLockTest, WaiterAcquiresAfterUnlock) {
  SpinLock lock;
  lock.lock();
  std::atomic<bool> acquired{false};
  std::thread waiter([&] {
    lock.lock();
    acquired.store(true, std::memory_order_relaxed);
    lock.unlock();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(acquired.load(std::memory_order_relaxed)); // we still hold it
  lock.unlock();
  waiter.join();                        // returns only after the waiter got the lock
  EXPECT_TRUE(acquired.load(std::memory_order_relaxed));
  EXPECT_FALSE(lock.locked());
} // WaiterAcquiresAfterUnlock

// N threads increment a plain (non-atomic) counter under the lock. The final
// count is exact only if the lock provides mutual exclusion and the release/
// acquire handoff publishes each increment to the next holder.
TEST(SpinLockTest, MutualExclusion) {
  constexpr int num_threads = 4;
  constexpr int num_iters = 50000;
  SpinLock lock;
  long counter = 0;                     // deliberately not atomic, guarded by lock
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < num_iters; ++i) {
        std::lock_guard g(lock);
        ++counter;
      }
    });
  } // spawn loop
  for (auto& t : threads) t.join();
  EXPECT_EQ(counter, long(num_threads)*num_iters);
} // MutualExclusion

// Same through try_lock(): each successful try_lock() guards exactly one
// increment, so the counter must equal the total number of successes (failed
// attempts, and there will be some under contention, must not touch it).
TEST(SpinLockTest, TryLockMutualExclusion) {
  constexpr int num_threads = 4;
  constexpr int num_iters = 20000;
  SpinLock lock;
  long counter = 0;                     // deliberately not atomic, guarded by lock
  std::atomic<long> successes{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&] {
      long local_successes = 0;
      for (int i = 0; i < num_iters; ++i) {
        if (lock.try_lock()) {
          ++counter;
          ++local_successes;
          lock.unlock();
        } // if acquired
      }
      successes.fetch_add(local_successes, std::memory_order_relaxed);
    });
  } // spawn loop
  for (auto& t : threads) t.join();
  EXPECT_EQ(counter, successes.load());
} // TryLockMutualExclusion
