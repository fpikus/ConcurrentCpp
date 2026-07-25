// Benchmark to tune the second (long-sleep) tier of the two-tier SpinLock
// (spinlock.h): the same lock algorithm with a second tier of 0.1, 1, and
// 10 ms, run across the contention dial (see spinlock_bm_common.h). The
// header's SpinLock runs alongside as a cross-check. For the lock against
// the atomic and std::mutex reference points, see spinlock_bm.C.
//
// Empirical results so far (256-CPU, 2 L3-domain machine): 1 ms is best or
// tied everywhere; 100 us ties it up to 64 threads but loses at 128; 10 ms
// wins at work=0 with 8-32 threads (parked waiters disturb the holder least)
// but collapses at >=128 threads, and at 256 threads for every work level --
// hence the 1 ms second tier in spinlock.h.
#include <time.h>
#include <atomic>

#include "spinlock.h"

#include "spinlock_bm_common.h"

// Parameterized copy of SpinLock from spinlock.h: the same TTAS acquire with
// the same unrolled burst of eight attempts and the same two-tier back-off
// (eight rounds of ~1 ns yield-sleeps, then a long sleep, repeat), except the
// long-sleep duration is the template parameter -- the knob this benchmark
// exists to tune. TwoTierSpinlock<1000001> is exactly the header's SpinLock;
// BM_spinlock below runs the header class itself as a cross-check.
template <long LongSleepNs>
class TwoTierSpinlock {
  public:
  TwoTierSpinlock() = default;
  TwoTierSpinlock(const TwoTierSpinlock&) = delete;
  TwoTierSpinlock& operator=(const TwoTierSpinlock&) = delete;
  void lock() {
    static constexpr timespec short_sleep = { 0, 1 };
    static constexpr timespec long_sleep = { 0, LongSleepNs };
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
        nanosleep(&short_sleep, nullptr);
      } else {
        spin_count = 0;
        nanosleep(&long_sleep, nullptr);
      }
    } // spin/back-off loop
  } // TwoTierSpinlock::lock()
  void unlock() { lock_.store(0, std::memory_order_release); }
  private:
  std::atomic<int> lock_ {0};           // 0 == free, 1 == held
}; // class TwoTierSpinlock

// The second-tier sweep: 0.1, 1, and 10 ms long sleeps.
BENCHMARK_TEMPLATE(BM_lock, TwoTierSpinlock<100001>)->Name("BM_spin2t_100us") ARGS;
BENCHMARK_TEMPLATE(BM_lock, TwoTierSpinlock<1000001>)->Name("BM_spin2t_1ms") ARGS;
BENCHMARK_TEMPLATE(BM_lock, TwoTierSpinlock<10000001>)->Name("BM_spin2t_10ms") ARGS;
// The header's SpinLock (second tier 1 ms): must match BM_spin2t_1ms; any gap
// means TwoTierSpinlock has drifted from the real lock.
BENCHMARK_TEMPLATE(BM_lock, SpinLock)->Name("BM_spinlock") ARGS;

BENCHMARK_MAIN();
