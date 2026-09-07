// Benchmark to tune the back-off ladder of the TTAS spinlock: BackOffParams
// (spinlock_tune.h) holds every waiting parameter as one constexpr aggregate,
// and this file explores that parameter space against the guarded-increment
// workload of spinlock_bm_common.h, across the full contention dial and thread
// range. For the lock against the atomic and std::mutex reference points, see
// spinlock_bm.C.
//
// THE EXPLORATION STRATEGY
//
// The space is NTry x NPause x NYield x NShortSleep x LongSleepNs, and a full
// cross product of even three values per axis is 243 configurations -- times 4
// contention levels times ~9 thread counts, which is a multi-day run. So this
// is not a grid search. It is a one-factor-at-a-time sweep around the shipped
// baseline (the `sweep:base` line below, which is exactly SpinLock), plus a
// `sweep:shape` group of the degenerate policies that a coordinate sweep can
// never reach because they are far from the baseline in several axes at once
// (no back-off at all, pause-only, yield-only, single-tier sleeping).
//
// The coordinate sweep answers "is the shipped value the best value of this
// knob, all else equal?", which is the question that matters as long as the
// axes do not interact strongly. Where a sweep says otherwise -- some knob has
// a better setting -- move the baseline there and re-run; that is a coordinate
// descent, and two or three passes of it get much closer to the optimum than a
// coarse grid would, for a small fraction of the machine time. Where two knobs
// are suspected of interacting (pause and short sleep are the likely pair: both
// are "wait a little before parking"), add the 2-D block explicitly rather than
// widening the whole product.
//
// RUNNING IT
//
// Every configuration in spinlock_tune_configs.h (shared with
// spinlock_mem_bm.C) is registered; which subset runs is decided at run time.
// The named test sets live in run_sets.sh -- the saturation proof runs
// `sweep:(base|shape)` at work:0, where the spinning shapes must lose and the
// margin is the finding; the low-contention probe runs the candidate groups at
// work:30 and work:100, where the development machine reversed the saturation
// ranking. For ad-hoc slices, names carry the sweep tag and every parameter
// (the filter is a regex):
//
//   ./spinlock_tune_bm --benchmark_filter='work:0/'             # saturation
//   ./spinlock_tune_bm --benchmark_filter='sweep:(base|shape)'  # the confirmation
//   ./spinlock_tune_bm --benchmark_filter='threads:32$'         # one thread count
//
// A sweep is only comparable against the baseline, so include `base` in the
// filter whenever you run one group -- it is the point every other line in the
// group differs from by exactly one knob. The threads:1 column is worth keeping
// in any run: the ladder is never climbed there, so it measures each
// configuration's uncontended cost and exposes fast-path codegen differences
// that would otherwise be mistaken for back-off effects.
//
// EMPIRICAL RESULTS SO FAR (256-CPU, 2 L3-domain machine)
//
// From the long-sleep sweep, which is what this benchmark originally existed
// for: 1 ms is best or tied everywhere; 100 us ties it up to 64 threads but
// loses at 128; 10 ms wins at work=0 with 8-32 threads (parked waiters disturb
// the holder least) but collapses at >=128 threads, and at 256 threads for
// every work level -- hence the 1 ms second tier in spinlock.h.
#include <atomic>

#include "spinlock.h"
#include "spinlock_tune.h"

#include "spinlock_bm_common.h"

// Contention levels for this sweep, overriding the four-point dial of
// spinlock_bm_common.h: saturation, plus the two low-occupancy points where the
// optimum was seen to move. work:3 is dropped because it sits between two
// levels that already disagree, and work:300 because work:100 reaches the same
// regime for a third of the machine time.
//   work:0   -- ~100% of a thread's time under the lock
//   work:30  -- ~1%
//   work:100 -- ~0.3%
#define TUNE_ARGS \
  ->ArgName("work")->Arg(0)->Arg(30)->Arg(100) \
  ->Arg(300)->Arg(1000)->Arg(3000)   /* knee sweep: genuine low contention */ \
  ->ThreadRange(1, numcpu) \
  ->UseRealTime()

// Register one parameter set under a name that spells out the whole ladder,
// e.g.
//   BM_tune/sweep:short/try:8/pause:0/yield:0/short:1/long:NS_1ms
// The configuration list itself lives in spinlock_tune_configs.h, shared with
// the memory-work sweep (spinlock_mem_bm.C) so that both work types always
// measure the same candidates.
#define TUNE_CONFIG(sweep, ntry, npause, nyield, nshort, longns)          \
  BENCHMARK_TEMPLATE(BM_lock,                                            \
                     LadderedLock<BackOffParams{ntry, npause, nyield,   \
                                               nshort, longns}>) \
      ->Name("BM_tune/sweep:" sweep "/try:" #ntry "/pause:" #npause      \
             "/yield:" #nyield "/short:" #nshort "/long:" #longns) TUNE_ARGS

#include "spinlock_tune_configs.h"

// The header's SpinLock: must match the `sweep:base` line above, which is the
// same ladder written with template parameters. Any gap beyond run-to-run
// noise means TunableSpinLock has drifted from the real lock, and every number
// below it is being compared against the wrong baseline.
BENCHMARK_TEMPLATE(BM_lock, SpinLock)->Name("BM_spinlock") TUNE_ARGS;

BENCHMARK_MAIN();
