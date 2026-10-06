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
// actually runs is decided at run time with --benchmark_filter. The comments
// below say what each group probes and why its values are there, and the
// caveats that should temper how the numbers are read; they are reasons to
// probe some configurations first, not reasons a configuration cannot be run.
#ifndef INCLUDED_SPINLOCK_TUNE_CONFIGS_H
#define INCLUDED_SPINLOCK_TUNE_CONFIGS_H

// The baseline: identical to SpinLock in spinlock.h. Every coordinate sweep
// below differs from this line in exactly one parameter, and every run should
// include it -- a sweep is only comparable against its baseline.
TUNE_CONFIG("base", 8, 0, 0, 8, NS_1ms);

// NTry -- attempts per round. Too few and a lock released mid-round is missed
// until after a back-off; too many and every waiter is issuing exchanges into
// a line the holder needs back in order to unlock.
//
// The trap in this group: the ladder is provably never climbed at one thread,
// so whatever NTry gains or loses there is inlining and code layout on the
// uncontended fast path, not waiting policy. Always read this group against
// the threads:1 column -- it is the codegen control.
TUNE_CONFIG("ntry", 1, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 2, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 4, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 16, 0, 0, 8, NS_1ms);
TUNE_CONFIG("ntry", 32, 0, 0, 8, NS_1ms);

// NPause -- rounds of PAUSE-hinted spinning inserted ahead of the sleeps. This
// is the tier that keeps the waiter on its core.
//
// Read PAUSE results against the machine's SMT state. PAUSE exists to hand
// pipeline resources to the sibling thread, so a run where every core carries
// two threads measures something different than one where each thread has a
// core to itself.
// 128 and 256 are there because on a 2-socket Granite Rapids server, with
// memory-streaming work at ~1% occupancy, the gain is still rising at 64
// rounds. (With sin/cos work it plateaus by 16.)
TUNE_CONFIG("pause", 8, 1, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 4, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 16, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 64, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 128, 0, 8, NS_1ms);
TUNE_CONFIG("pause", 8, 256, 0, 8, NS_1ms);

// NYield -- rounds of sched_yield() ahead of the sleeps. Unlike a nanosleep,
// a yield only gives up the core if some other thread wants it, so at low
// thread counts it is nearly free and at high counts it is nearly a sleep.
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
// 0 and 1 bracket whether the short tier earns its place at all; 4, 16 and 32
// probe the knob around the baseline's 8. Park-eager configurations (S0
// especially) are the most sensitive to per-process placement -- prefer means
// over multiple invocations when reading them.
TUNE_CONFIG("short", 8, 0, 0, 0, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 1, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 4, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 16, NS_1ms);
TUNE_CONFIG("short", 8, 0, 0, 32, NS_1ms);

// LongSleepNs -- the tier sweep behind the 1 ms long sleep of spinlock.h:
// 0.1 ms and 10 ms against the baseline's 1 ms. This is the claim the README
// makes, so it gets re-run on every machine.
TUNE_CONFIG("long", 8, 0, 0, 8, NS_100us);
TUNE_CONFIG("long", 8, 0, 0, 8, NS_10ms);

// Shape references: policies several axes away from the baseline, which is
// where the interesting failures live. Each is a whole class of spinlock in
// its own right, expressed as a point in this parameter space.
//
// The first three are the confirmation that sleeping is what makes this lock
// work: at saturation they must lose, and lose badly. They are cheap to run
// and they are the evidence behind the README's central claim. shape.full is
// the all-four-tiers ladder, the best probe of the low-occupancy regime.
TUNE_CONFIG("shape.spin", 8, 0, 0, 0, NS_off);   // no back-off at all: pure TTAS
TUNE_CONFIG("shape.pause", 8, 64, 0, 0, NS_off); // pause-only, never deschedules
TUNE_CONFIG("shape.yield", 8, 0, 64, 0, NS_off); // sched_yield-only
TUNE_CONFIG("shape.short", 8, 0, 0, 8, NS_off);  // single tier of short sleeps
TUNE_CONFIG("shape.full", 8, 16, 4, 8, NS_1ms);  // all four tiers, cheapest first

#endif // INCLUDED_SPINLOCK_TUNE_CONFIGS_H
