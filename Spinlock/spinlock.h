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
#ifndef INCLUDED_SPINLOCK_H
#define INCLUDED_SPINLOCK_H
#include <time.h>
#include <atomic>

// A test-and-test-and-set (TTAS) spinlock with a two-tier backoff.
//
// The lock is a single int: 0 == free, 1 == held. Acquisition uses the TTAS
// idiom (see lock()) so that spinning threads mostly read a shared cache line
// instead of repeatedly issuing read-for-ownership atomic RMWs, which is what
// makes a naive test-and-set spinlock collapse under contention.
//
// Back-off sleeps for the spin loops below. Anonymous namespace gives these
// internal linkage, so every translation unit that includes this header gets
// its own copy with no ODR clash (the extra `static` is redundant belt-and-
// suspenders). The timespec format is {seconds, nanoseconds}; two escalating
// tiers:
//   short (~1 ns request): nanosleep rounds this up to the next scheduling
//     opportunity, so in practice it just relinquishes the CPU -- a cheap yield
//     used while the lock looks briefly contended.
//   long  (~1 ms):         a real sleep, used once short yields have failed
//     enough times, so a thread waiting on a long-held lock stops burning a
//     core and lets the OS schedule useful work instead.
namespace {
  static const struct timespec spin_wait_short = { 0, 1 };
  static const struct timespec spin_wait_long  = { 0, 1000001 };
  static inline void spin_wait_short_sleep() { nanosleep(&spin_wait_short, nullptr); }
  static inline void spin_wait_long_sleep()  { nanosleep(&spin_wait_long,  nullptr); }
} // anonymous namespace

// A test-and-test-and-set (TTAS) spinlock with escalating back-off: cheap and
// fast under low contention, and it degrades gracefully (yielding, then
// sleeping) rather than melting a core when the lock is held for a long time.
// The lock word is a single int: 0 == free, 1 == held.
//
// Memory-ordering contract:
//   - A successful acquire uses exchange(..., acquire); unlock() uses
//     store(0, release). The release/acquire pair makes everything the previous
//     holder did in its critical section happen-before the next holder's -- the
//     standard lock handoff.
//   - locked() is a relaxed, advisory peek and synchronizes nothing.
class SpinLock {
  public:
  // Acquire the lock, blocking (spinning, then sleeping) until it is held.
  //
  // The key expression, reused throughout, is the TTAS idiom:
  //     lock_.load(relaxed) || lock_.exchange(1, acquire)
  // Read it as "is the lock unavailable to me?":
  //   * The relaxed load is the "test": while the lock is held it reads 1 and
  //     short-circuits, so waiters only *read* the shared cache line instead of
  //     each hammering it with an atomic RMW -- avoiding the cache-line storm a
  //     plain test-and-set spinlock suffers under contention.
  //   * Only when the load sees 0 ("looks free") do we pay for the exchange
  //     ("test-and-set"): it atomically writes 1 and returns the prior value,
  //     so 0 means WE won the lock (expression false -> stop spinning) and 1
  //     means another thread beat us between the load and the exchange (keep
  //     spinning). The relaxed load may be stale, but that only ever costs an
  //     extra spin or a failed exchange, never incorrect acquisition --
  //     correctness rides entirely on the winning exchange's acquire, which
  //     synchronizes-with the previous holder's releasing unlock().
  //
  // Loop shape: the for-condition is one acquire attempt; false means we
  // acquired and fall straight out. The body is that same attempt hand-unrolled
  // seven more times back-to-back -- a burst of eight cheap retries before
  // paying for a syscall -- so a just-released lock is grabbed within a few
  // instructions of the holder's unlock, without branching on the back-off
  // counter between attempts. Only if all eight attempts fail do we sleep, and
  // the back-off escalates: eight rounds of near-zero yields (spin_count 0..7),
  // then one ~1 ms sleep, resetting the counter and repeating -- so a
  // contended-but-brief lock stays busy-ish while a long-held lock lets the
  // waiter get out of the scheduler's way.
  void lock() {
    for (int spin_count = 0;
        lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire);
        ++spin_count)
    {
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return;
      if (spin_count < 8) {
        spin_wait_short_sleep();        // first 8 rounds: yield, stay hot
      } else {
        spin_count = 0;                 // then escalate to a real sleep and
        spin_wait_long_sleep();         // start the yield budget over
      }
    } // spin/back-off loop
  } // SpinLock::lock()

  // Release the lock. The store-release pairs with the acquire on the next
  // acquirer's winning exchange (see the class memory-ordering contract),
  // handing off this holder's critical-section writes to the next holder.
  void unlock() {
    lock_.store(0, std::memory_order_release);
  }

  // Bounded acquire: returns true holding the lock, false if it could not be
  // taken within the short-spin budget.
  //
  // This is deliberately NOT a single-shot try. Even when the lock is free,
  // its cache line is likely in the Shared state (other cores have probed it
  // recently); the exchange must first get the line Exclusive, and that
  // ownership transfer costs a cache-coherence round trip. A single-shot
  // attempt would report "busy" for what is really just coherence latency,
  // not actual lock contention. Instead, this runs the same TTAS spin as
  // lock() -- the unrolled burst of eight attempts rides out the ownership
  // transfer, and up to eight rounds of near-zero yields ride out a brief
  // holder -- and gives up only where lock() would escalate to the ~1 ms
  // long sleep. So false means the lock was genuinely held the whole time,
  // but a "failed" try_lock() has already spun and yielded for a short while;
  // callers that need a truly non-blocking (and correspondingly racy) probe
  // should use locked() instead.
  bool try_lock() {
    for (int spin_count = 0;
        lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire);
        ++spin_count)
    {
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (!(lock_.load(std::memory_order_relaxed) || lock_.exchange(1, std::memory_order_acquire))) return true;
      if (spin_count < 8) {
        spin_wait_short_sleep();        // spin/yield through the short tier...
      } else {
        return false;                   // ...then give up instead of long-sleeping
      }
    } // bounded spin loop
    return true;                        // for-condition saw the lock free and the exchange won it
  } // SpinLock::try_lock()

  // Advisory, non-synchronizing peek at the lock state. Relaxed and racy: the
  // answer may be stale the instant it returns, so it is only useful as a hint
  // (e.g. an assert or a diagnostic) -- never build mutual exclusion on it.
  bool locked() const {
    return lock_.load(std::memory_order_relaxed) == 1;
  }

  private:
  std::atomic<int> lock_ {0};           // 0 == free, 1 == held
}; // class SpinLock

#endif // INCLUDED_SPINLOCK_H
