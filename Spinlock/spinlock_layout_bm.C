// Step-6 aside: does the payload belong on the lock's cache line?
//
// The folklore says yes: whoever acquired the lock has just fetched its line,
// so payload sharing the line is free. The hypothesis under test says that at
// high contention the folklore is exactly wrong: waiters hammer the lock line
// continuously -- TTAS reads keep it Shared, failed exchanges keep yanking it
// Exclusive -- so a payload on that line has its accesses caught in the storm,
// while a payload on its own quiet line is touched only by the holder.
//
// The struct places the same payload at four distances from the lock:
//   x0 -- the lock's own cache line (the folklore layout)
//   x1 -- the next line: different 64B line, same 128B sector, so the L2
//         spatial prefetcher still pairs it with the lock line
//   x2 -- the next 128B sector over (its partner line held by an untouched
//         spacer, so the sector is otherwise quiet)
//   x3 -- two sectors away, fully isolated (the "far" control; this is also
//         what every other benchmark in this directory does)
// Measured: guarded increment of each x_i at work:0 (saturation) and work:100
// (low contention), across the thread range. If the folklore is right, x0
// wins everywhere; if the hypothesis is right, x0 loses at saturation and the
// x1/x2 columns show how far the disturbance reaches past the line itself.
#include <cstddef>
#include <atomic>

#include "spinlock.h"
#include "spinlock_tune.h"

#include "spinlock_bm_common.h"

struct S {
  alignas(64) TunableSpinLock lock;     // the lock word, head of its line
  unsigned long x0;                     // payload on the SAME line as the lock
  alignas(64) unsigned long x1;         // next line, same 128B prefetch sector
  alignas(64) unsigned long x2;         // next sector over
  alignas(64) unsigned long dummy;      // spacer: keeps x2's sector partner
                                        // line untouched
  alignas(64) unsigned long x3;         // two sectors away, fully isolated
}; // struct S

static_assert(offsetof(S, x0) < 64);
static_assert(offsetof(S, x1) == 64);
static_assert(offsetof(S, x2) == 128);
static_assert(offsetof(S, x3) == 256);

// The benchmark body: the standard coupled-work guarded update of
// spinlock_bm_common.h, with the guarded field selected by member pointer --
// the ONLY thing that differs between the four registered variants is where
// the payload lives relative to the lock. The ladder is the shipped one
// (BackOffParams{} == SpinLock).
template <unsigned long S::*Payload>
void BM_layout(benchmark::State& state) {
  alignas(64) static S s;
  const long work = state.range(0);
  if (state.thread_index() == 0) s.*Payload = 0;
  double local_x = 1.0 + state.thread_index();
  for (auto _ : state) {
    local_x = do_work(local_x, work);
    const unsigned long n = static_cast<unsigned long>(1.0 + local_x);
    s.lock.lock<BackOffParams{}>();
    // DoNotOptimize is load-bearing, as in BM_lock: nothing reads the field.
    benchmark::DoNotOptimize(s.*Payload += n);
    s.lock.unlock();
  }
  state.SetItemsProcessed(state.iterations());
} // BM_layout

#define LAYOUT_ARGS \
  ->ArgName("work")->Arg(0)->Arg(100) \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

BENCHMARK_TEMPLATE(BM_layout, &S::x0)->Name("BM_layout/field:x0_same_line") LAYOUT_ARGS;
BENCHMARK_TEMPLATE(BM_layout, &S::x1)->Name("BM_layout/field:x1_same_sector") LAYOUT_ARGS;
BENCHMARK_TEMPLATE(BM_layout, &S::x2)->Name("BM_layout/field:x2_next_sector") LAYOUT_ARGS;
BENCHMARK_TEMPLATE(BM_layout, &S::x3)->Name("BM_layout/field:x3_far") LAYOUT_ARGS;

BENCHMARK_MAIN();
