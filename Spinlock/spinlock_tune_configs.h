// The back-off ladder configurations under test, shared by every benchmark
// that sweeps them -- currently spinlock_tune_bm.C (register-resident sin/cos
// work) and spinlock_mem_bm.C (memory-streaming work). One list, so the two
// work types always measure the same candidates.
//
// This is an X-macro list: the including file defines
//
//   TUNE_CONFIG(sweep, ntry, npause, nyield, nshort, longns)
//
// to register one TunableSpinLock<ntry, npause, nyield, nshort, longns>
// configuration under its own benchmark body and naming, then includes this
// header. `sweep` is the group tag used for filtering.
//
// EVERY configuration is registered; registration is cheap, and which subset
// actually runs is decided at run time -- by the named test sets in
// run_sets.sh, or by an ad-hoc --benchmark_filter. The comments below record
// what the development machine (16-core Ryzen 9 9950X, uncontrolled thread
// placement, sin/cos work) measured for each group, and the caveats that
// should temper how the numbers are read; they are why the default test sets
// probe some configurations and skip others, not reasons a configuration
// cannot be run.
#ifndef INCLUDED_SPINLOCK_TUNE_CONFIGS_H
#define INCLUDED_SPINLOCK_TUNE_CONFIGS_H

// The baseline: identical to SpinLock in spinlock.h. Every coordinate sweep
// below differs from this line in exactly one parameter, and every test set
// includes it -- a sweep is only comparable against its baseline.
TUNE_CONFIG("base", 8, 0, 0, 8, NS_1ms);

// NTry -- attempts per round. Too few and a lock released mid-round is missed
// until after a back-off; too many and every waiter is issuing exchanges into
// a line the holder needs back in order to unlock.
//
// Dev-machine results, with a trap in them: NTry of 1, 2 and 4 measured BEST
// (270 M/s at one thread) -- but the ladder is provably never climbed at one
// thread, so ~15% of their advantage is inlining and code layout on the
// uncontended fast path, not waiting policy. NTry=16 is the honest candidate:
// ~245 M/s flat from 1 to 32 threads at work:0 where the baseline decayed to
// 192, with the SAME single-thread cost as the baseline (236 M/s). NTry=32
// was worse than the baseline everywhere. Always read this group against the
// threads:1 column -- it is the codegen control.
TUNE_CONFIG("ntry", 1, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 2, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 4, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 16, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 32, 0, 0, 8, NS_1ms);

// NPause -- rounds of PAUSE-hinted spinning inserted ahead of the sleeps. This
// is the tier that keeps the waiter on its core: worthless at saturation and
// the single most valuable knob at ~1% occupancy on the dev machine, with the
// effect monotone through the intermediate points (1 and 4 sit between the
// endpoints; the default sets probe 16 and 64).
//
// Read PAUSE results against the machine's SMT state. PAUSE exists to hand
// pipeline resources to the sibling thread, so a run where every core carries
// two threads measures something different than one where each thread has a
// core to itself.
// 128 and 256 chase a finding from the Granite Rapids runs (leslie, Sept
// 2026): with memory-streaming work at ~1% occupancy the gain was still
// rising at 64 rounds -- the axis had not plateaued at the edge of the
// original grid. (With sin/cos work it plateaus by 16.)
TUNE_CONFIG("pause", 8, 1, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 4, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 16, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 64, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 128, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 256, 0, 8, NS_1ms);

// NYield -- rounds of sched_yield() ahead of the sleeps. Unlike a nanosleep,
// a yield only gives up the core if some other thread wants it, so at low
// thread counts it is nearly free and at high counts it is nearly a sleep.
// Same dev-machine story as NPause: dead weight at saturation, a large win at
// 1% occupancy; 1 was too small to separate from the baseline.
// 32 and 64 extend the axis for the same reason as pause 128/256 above; the
// sleep tiers stay as the backstop -- the yield-only shape still collapses at
// scale, so the question is pre-park budget, not replacing the sleeps.
TUNE_CONFIG("yield", 8, 0, 1, 8, NS_1ms);
TUNE_CONFIG("yield", 8, 0, 4, 8, NS_1ms);
TUNE_CONFIG("yield", 8, 0, 16, 8, NS_1ms);
TUNE_CONFIG("yield", 8, 0, 32, 8, NS_1ms);
TUNE_CONFIG("yield", 8, 0, 64, 8, NS_1ms);

// NShortSleep -- how many ~1 ns sleeps (in practice, deschedule-until-next-
// scheduling-opportunity) before escalating to the long sleep. 0 parks a
// waiter on the first failed round.
//
// Dev-machine results: 0 and 1 both beat the baseline's 8 at saturation (S0
// by 30% up to 16 threads before collapsing at 32; S1 by 33% at 32 threads
// and flat throughout), and they bracket whether the short tier earns its
// place at all; 4, 16 and 32 all landed on the baseline within noise, so the
// knob is flat in that range. Park-eager configurations (S0 especially) also
// showed the largest per-process placement sensitivity on the dev machine --
// prefer means over multiple invocations when reading them.
TUNE_CONFIG("short", 8, 0, 0, 0, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 1, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 4, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 16, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 32, NS_1ms);

// LongSleepNs -- the original tier sweep: 0.1 ms and 10 ms against the
// baseline's 1 ms. This is the claim the README makes, so it gets re-run on
// every machine.
TUNE_CONFIG("long", 8, 0, 0, 8, NS_100us);
TUNE_CONFIG("long", 8, 0, 0, 8, NS_10ms);

// Shape references: policies several axes away from the baseline, which is
// where the interesting failures live. Each is a whole class of spinlock in
// its own right, expressed as a point in this parameter space.
//
// The first three are the confirmation that sleeping is what makes this lock
// work: at saturation they must lose, and lose badly (5 to 40 fold on the dev
// machine). They are cheap to run and they are the evidence behind the
// README's central claim. shape.short tracked the baseline exactly on the dev
// machine; shape.full is the all-four-tiers ladder, the best probe of the
// low-occupancy regime.
TUNE_CONFIG("shape.spin", 8, 0, 0, 0, NS_off);   // no back-off at all: pure TTAS
TUNE_CONFIG("shape.pause", 8, 64, 0, 0, NS_off); // pause-only, never deschedules
TUNE_CONFIG("shape.yield", 8, 0, 64, 0, NS_off); // sched_yield-only
TUNE_CONFIG("shape.short", 8, 0, 0, 8, NS_off);  // single tier of short sleeps
TUNE_CONFIG("shape.full", 8, 16, 4, 8, NS_1ms);  // all four tiers, cheapest first

#endif // INCLUDED_SPINLOCK_TUNE_CONFIGS_H
