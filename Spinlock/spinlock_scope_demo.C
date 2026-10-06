// The lock scope demo: two Google Benchmark benchmarks that show, on a laptop,
// that writing the slot you have just claimed AFTER unlock() is slower than
// writing it before, under the shipped SpinLock (spinlock.h). It is the
// smallest piece of the lock scope benchmark (spinlock_scope_bm.C and its twin
// spinlock_scope_mbm.C, both built on spinlock_scope_common.h) that still shows
// the effect on a client CPU; those two, and the MECHANISM section of
// spinlock_scope_common.h, separate the mechanisms behind it. This file depends
// on nothing but spinlock.h and Google Benchmark, so that it can be read on its
// own.
//
// WHAT IS MEASURED
//
// Every iteration takes the lock, claims the next slot of a shared ring by
// advancing a shared index under the lock, and stores a per-thread counter into
// the claimed slot. Once claimed, the slot belongs to the claiming thread
// alone, so the store does not need the lock, and the two benchmarks differ
// only in where it is:
//   BM_store_before_unlock -- the store is the last thing in the critical
//                             section;
//   BM_store_after_unlock  -- the same store is the first thing after
//                             unlock().
// The ring has 2^16 slots, one per 64-byte cache line (struct Slot), so every
// slot store touches a new line (THE INGREDIENTS below says why that matters).
// The loop body is lock, claim, [store], unlock, [store], then a register-only
// comparison that counts lock handoffs (HANDOFFS below). The whole of it,
// SpinLock::lock() included, is inlined into the loop (step() below).
//
// WHY STORING AFTER THE UNLOCK LOSES
//
// Under contention this lock's throughput comes from batching: a waiter that
// does not get the lock within its burst of eight attempts parks in nanosleep
// (eight rounds of a short sleep that the kernel stretches to tens of
// microseconds, then a 1 ms sleep, and again), so most of the time one thread
// holds the lock, finds its line already in its own cache, and runs a long
// streak of operations. The lock only changes hands when a waiter's attempt
// lands while the lock is visibly free. Storing after the unlock makes that
// happen more often, and the handoffs_per_op counter shows how much more. Why,
// in the terms of spinlock_scope_common.h (W under MECHANISM there), carried
// over to the laptop as the working hypothesis, not measured there: a slot
// store before the unlock keeps the visibly-free window short, because the
// unlock store cannot become visible before the older slot store has committed
// (x86 commits stores in order), so while the holder waits for the slot's cache
// line the lock still looks held; the same store after the unlock leaves the
// lock visibly free until the holder's next lock() takes it again, and on x86
// servers that wait is the length of the slot store's miss. Whether a Zen 4
// core holds its next locked exchange back for the full miss the same way is
// not known. The cost of a handoff is reasoned, not measured here: cache-line
// transfers of the lock, the index and the slot lines the new holder then walks
// cold, and the previous holder, still in its burst of attempts, taking the
// lock straight back.
//
// THE INGREDIENTS, AND WHY A LAPTOP NEEDS THE THREAD COUNTS IT GETS HERE
//
// The scope benchmark shows the loss on the two-socket x86 servers it was built
// for. On a Zen 4 laptop (8 cores / 16 threads, one L3) the same loop at 1 to
// 16 threads shows nothing: storing after the unlock is no slower. Two things
// are different on the laptop, and the thread counts registered here are the
// answer to the first:
//   - Too few waiters. A handoff needs a waiter awake in its burst while the
//     window is open, and a parked waiter is awake for a fraction of a percent
//     of the time (reasoned from the lock's sleeps: a burst of eight attempts
//     is well under a microsecond, each park tens of microseconds to a
//     millisecond); with 15 waiters the window, open or not, is almost never
//     probed. The handoff counter shows the window is there (at 16 threads,
//     out-of-scope hands the lock over several times as often as in-scope) but
//     at a rate too low to cost anything. The servers have over a hundred
//     waiters; the laptop gets them by running more threads than it has CPUs.
//     Waiters spend their time asleep, not on a CPU, so oversubscription
//     changes the waiter count, not the holder's streaks. The loss appears at
//     128 threads, peaks around 256 and fades past 512, where the kernel's work
//     of parking and waking hundreds of threads slows both variants alike.
//     Hence the registered counts, 128, 256 and 512.
//   - Cheap handoffs. With one L3, a cache-line transfer between two of the
//     laptop's cores should cost tens of nanoseconds, against a hundred or more
//     across the sockets and dies of the servers (reasoned from the topology,
//     not measured). On a desktop part with two core dies (a Ryzen 9 9950X,
//     say) the same handoff rate should cost more, so the loss there is
//     expected to be at least as large, at the same thread counts; that is a
//     prediction, not a measurement.
// Variations of the loop add nothing the plain loop does not show -- rings of
// 2^10 or 2^20 slots instead of 2^16, 128-byte slots, slots chosen by hashing
// the index over a 64 MiB ring so that every store misses to memory, eight
// 8-byte slots per line instead of one, a second group of threads writing the
// same ring under a lock of its own as the consumers of ConcurrentQueue do --
// so the demo keeps the plain loop.
//
// HANDOFFS
//
// Each thread remembers the unmasked index it claimed last; if this
// acquisition's index is not that one plus 1, another thread held the lock in
// between, and the thread counts one handoff. The comparison is register-only,
// placed after the unlock and the post-unlock store like the loop's exit check,
// so it adds no store to the loop. A thread's first acquisition has no previous
// index and is not counted. Summed over the threads and divided by the
// operations, it is the handoffs_per_op counter: the reciprocal of the mean
// streak length.
//
// MEASUREMENT AND HOW TO READ THE OUTPUT
//
// Google Benchmark's items_per_second divides the total item count by the
// MEAN of the threads' loop times. With a fixed iteration count per thread
// (the usual Google Benchmark loop) that overstates the throughput of an
// unfair lock: the threads that win the streaks finish early, the tail of the
// run has fewer threads contending, and the mean loop time is shorter than the
// run. This demo instead runs every thread until a common stop flag, raised
// by a timekeeper thread after kWindow of wall-clock time, and each thread
// counts its own operations. Then every thread's loop time is the window, up
// to the drain at the end: a thread waiting in lock() when the flag goes up
// cannot see it until it has won the lock once more, so the remaining threads
// leave one after another, each after one more operation, interleaved with
// their sleeps of up to a millisecond; how long that takes can differ between
// the two benchmarks, and the wall-clock cross-check below bounds it. With
// that, items_per_second is the true count over the window: the column to
// read. Each run is registered as one iteration
// (Iterations(1)), so --benchmark_min_time does not apply; sample with
// --benchmark_repetitions. The counters:
//   items_per_second      -- all threads' operations over the mean loop time,
//                            i.e. over the window; the honest column here;
//   wall_items_per_second -- the same count over the span from the earliest
//                            thread's loop start to the latest thread's loop
//                            end, the cross-check: it can only be lower, and
//                            close agreement means the drain is negligible;
//   handoffs_per_op       -- see HANDOFFS.
// A representative run:
//   build/$(hostname)/spinlock_scope_demo --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
//
// LAYOUT
//
// Everything the threads share is laid out in 128-byte blocks (the prefetch
// sector on x86, the line size on Apple M-series), so that nothing shares a
// line with the lock by accident: the lock, the slot index, the pointer to the
// ring, the stop flag, and the pointer to the per-thread records. The ring
// pointer and the records pointer are written only before the threads start
// and read in the loop, so they stay Shared in every core's cache; the stop
// flag is read on every iteration and written once.
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>

#include "spinlock.h"

#include "benchmark/benchmark.h"

// A slot of the ring: one per 64-byte line (alignas pads it to 64 bytes).
struct alignas(64) Slot {
  unsigned long v;                      // the payload; stored once per claim, never read
};
static_assert(sizeof(Slot) == 64);

// The number of slots in the ring, a power of two: the index wraps by masking
// with kSlots - 1. 2^16 slots of 64 bytes is 4 MiB, the same ring as the scope
// benchmark's a64 variants.
constexpr size_t kSlots = size_t{1} << 16;
constexpr size_t kIdxMask = kSlots - 1;
static_assert(std::has_single_bit(kSlots));

// Wall-clock length of one run: every thread loops until the timekeeper raises
// the stop flag this long after the run starts (MEASUREMENT above).
constexpr std::chrono::milliseconds kWindow {1000};

// One thread's record of one run: when its loop started and ended (for
// wall_items_per_second) and its handoff count. On a 128-byte block of its own,
// written twice per run.
struct alignas(128) ThreadStats {
  std::chrono::steady_clock::time_point first;  // just before this thread's first operation
  std::chrono::steady_clock::time_point last;   // just after its last one
  unsigned long handoffs;               // acquisitions after the first that followed another thread's
};

// The state that all threads of a run share; see LAYOUT above. One object,
// shared by both benchmarks, which never run at once: every run resets what it
// uses (thread 0's setup in BM_scope_demo()). constinit: zero-initialized at
// load time, no run-time guard.
struct Shared {
  alignas(128) SpinLock lock;           // the lock every operation takes
  alignas(128) size_t slot_idx {};      // next slot to claim (before masking); guarded by lock
  alignas(128) Slot* slots {};          // non-owning view of the ring, set before the threads start
  alignas(128) std::atomic<bool> stop {false};  // raised by the timekeeper when the window ends
  alignas(128) ThreadStats* thread_stats {};    // non-owning view of the per-thread records
}; // struct Shared
static_assert(offsetof(Shared, slot_idx) == 128);
static_assert(offsetof(Shared, slots) == 256);
static_assert(offsetof(Shared, stop) == 384);
static_assert(offsetof(Shared, thread_stats) == 512);
static constinit Shared shared {};

// A thread's handoff counter (HANDOFFS above), kept in registers by the loop.
struct HandoffCounter {
  size_t prev = 0;                      // the unmasked index this thread claimed last
  unsigned long breaks = 0;             // acquisitions whose index was not prev + 1
};

// Store `value` into a slot's payload. A relaxed atomic store compiles to a
// plain mov, but is not a data race if a preempted thread's post-unlock store
// ever overlaps the next owner's store to the same slot (a lap of 2^16
// claims later), and the optimizer cannot drop it although nothing reads it.
inline void store_relaxed(unsigned long& word, unsigned long value) {
  std::atomic_ref<unsigned long>(word).store(value, std::memory_order_relaxed);
}

// The measured operation: lock, claim a slot, store the incremented count
// into it on the side of unlock() that `before_unlock` selects, then the
// handoff comparison.
//   s        -- the shared state; s.slots must point to kSlots slots;
//   count    -- the calling thread's running count of operations, incremented
//               once per call; the incremented value is what is stored, so
//               every store writes a new value;
//   handoffs -- the calling thread's handoff counter (HANDOFFS above).
// always_inline and flatten: what is measured is the order of the stores, the
// unlock and the next lock, so the whole operation, SpinLock::lock() included,
// must be inlined into the loop on every compiler (left to itself, GCC keeps
// lock() out of line in some harnesses and Clang in all; a call's pushes are
// stores between the unlock and the next lock, which on recent Intel cores cost
// extra; see E2 in spinlock_scope_common.h). flatten reaches lock()'s back-off
// path too, down to the nanosleep() calls, which stay calls.
// The empty asm statement takes the claimed index as an in/out operand and
// clobbers memory, so the comparison after it cannot be scheduled above the
// stores before it; it emits no instruction.
template <bool before_unlock>
[[gnu::always_inline, gnu::flatten]] inline void step(Shared& s, unsigned long& count, HandoffCounter& handoffs) {
  s.lock.lock();
  size_t claimed = s.slot_idx++;        // unmasked: the handoff comparison needs the full index
  unsigned long& slot = s.slots[claimed & kIdxMask].v;
  if constexpr (before_unlock) store_relaxed(slot, ++count);
  s.lock.unlock();
  if constexpr (!before_unlock) store_relaxed(slot, ++count);
  asm volatile("" : "+r"(claimed) : : "memory");
  handoffs.breaks += (claimed != handoffs.prev + 1);
  handoffs.prev = claimed;
} // step()

// One thread's loop of a run: step() until the stop flag is up. The first
// operation is done before the loop and its handoff comparison discarded (a
// thread's first acquisition has no previous index). Returns the number of
// operations; the thread's handoff count goes to `handoffs_out`.
//
// noinline: a function of its own, so that the only values live across the
// loop are the loop's own. lock()'s inlined back-off calls nanosleep(), so
// every value the loop carries must sit in a callee-saved register or be
// spilled, and a spilled counter is a stack store between the unlock and the
// next lock (see step()). What matters is that the steady-state loop stores
// nothing but the index, the slot and the unlock: the count and the handoff
// fields stay in registers. The stop flag's load each iteration reads a line
// that is Shared in every core until the timekeeper writes it.
template <bool before_unlock>
[[gnu::noinline]] static unsigned long run_until_stop(Shared& s, unsigned long& handoffs_out) {
  unsigned long count = 0;
  HandoffCounter handoffs;
  step<before_unlock>(s, count, handoffs);
  handoffs.breaks = 0;
  while (!s.stop.load(std::memory_order_relaxed)) step<before_unlock>(s, count, handoffs);
  handoffs_out = handoffs.breaks;
  return count;
} // run_until_stop()

// The benchmark body, one instantiation per benchmark: before_unlock selects
// the store's side. Thread 0 allocates the ring and the per-thread records and
// resets the index and the stop flag before the loop; the loop's start and end
// are barriers across all threads of a run in Google Benchmark
// (State::StartKeepRunning() and FinishKeepRunning()), so the setup happens
// before any thread's first operation, and every thread has written its record
// before thread 0 computes the counters after the loop. Inside its one
// iteration, thread 0 also starts the timekeeper, a std::jthread that sleeps
// for kWindow and raises the stop flag; the jthread joins when the iteration
// ends, after every thread has left run_until_stop(). The ring is written once
// at allocation, so its pages are present before the measurement.
template <bool before_unlock>
void BM_scope_demo(benchmark::State& state) {
  using Clock = std::chrono::steady_clock;
  Shared& s = shared;
  const int tid = state.thread_index();
  const int threads = state.threads();

  std::unique_ptr<Slot[]> slots_owner;          // thread 0 only: owns the ring for this run
  std::unique_ptr<ThreadStats[]> stats_owner;   // thread 0 only: owns the per-thread records
  if (tid == 0) {
    slots_owner = std::make_unique_for_overwrite<Slot[]>(kSlots);
    for (size_t i = 0; i != kSlots; ++i) slots_owner[i].v = i;
    stats_owner = std::make_unique<ThreadStats[]>(static_cast<size_t>(threads));
    s.slots = slots_owner.get();
    s.thread_stats = stats_owner.get();
    s.slot_idx = 0;
    s.stop.store(false, std::memory_order_relaxed);
  } // thread 0 setup

  unsigned long count = 0;              // this thread's operations in the run
  for (auto _ : state) {
    std::jthread timekeeper;
    if (tid == 0) {
      timekeeper = std::jthread([] {
        std::this_thread::sleep_for(kWindow);
        shared.stop.store(true, std::memory_order_relaxed);
      });
    } // thread 0 starts the timekeeper
    ThreadStats& mine = s.thread_stats[tid];
    mine.first = Clock::now();
    count = run_until_stop<before_unlock>(s, mine.handoffs);
    mine.last = Clock::now();
  } // the one iteration of the run

  // Past the end-of-loop barrier: every record is written. Each thread's count
  // goes into items_processed, which Google Benchmark sums over the threads and
  // turns into items_per_second. The two counters are plain values set by
  // thread 0 alone (a rate would be divided by the mean loop time again; other
  // threads do not set them, so the sum over threads leaves them unchanged).
  // The total count is the final shared index: every operation advanced it
  // once, so it equals the sum of the threads' counts.
  state.SetItemsProcessed(static_cast<int64_t>(count));
  if (tid == 0) {
    Clock::time_point start = s.thread_stats[0].first;      // earliest loop start
    Clock::time_point finish = s.thread_stats[0].last;      // latest loop end
    unsigned long handoffs = 0;
    for (int i = 0; i != threads; ++i) {
      start = std::min(start, s.thread_stats[i].first);
      finish = std::max(finish, s.thread_stats[i].last);
      handoffs += s.thread_stats[i].handoffs;
    } // loop over the per-thread records
    const double span = std::chrono::duration<double>(finish - start).count();
    const double total = static_cast<double>(s.slot_idx);
    state.counters["wall_items_per_second"] = benchmark::Counter(span > 0 ? total/span : 0.0);
    state.counters["handoffs_per_op"] = benchmark::Counter(total > 0 ? static_cast<double>(handoffs)/total : 0.0);
    s.slots = nullptr;
    s.thread_stats = nullptr;
  } // thread 0 results
} // BM_scope_demo()

// Register both benchmarks at the thread counts where the laptop shows the
// loss (THE INGREDIENTS above): more threads than CPUs, on purpose. This
// replaces BENCHMARK_MAIN() because the benchmarks are registered in a loop.
int main(int argc, char** argv) {
  // Same start-up as BENCHMARK_MAIN(), including its re-exec without ASLR.
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  for (int t : {128, 256, 512}) {
    benchmark::RegisterBenchmark("BM_store_before_unlock", BM_scope_demo<true>)
      ->Threads(t)->UseRealTime()->Iterations(1);
    benchmark::RegisterBenchmark("BM_store_after_unlock", BM_scope_demo<false>)
      ->Threads(t)->UseRealTime()->Iterations(1);
  } // loop over thread counts

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
} // main()
