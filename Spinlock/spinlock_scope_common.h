// The lock scope benchmark: on which side of unlock() a thread should write
// the slot it has just claimed under the shipped SpinLock (spinlock.h), and why
// the answer differs from machine to machine and with the number of threads.
// This header is the primary explanation of what the benchmark has found
// (MECHANISM below; the APPENDIX at the end of this comment keeps the
// hypotheses rejected on the way, the open questions and a side finding), and
// it holds everything its two harnesses share -- the slot types, the shared
// state and its layout, the measured operation, the list of variants and the
// thread counts -- so that both run the same fully inlined loop body
// (scope_step(), with SpinLock::lock() inlined into it):
//   spinlock_scope_bm.C  -- the Google Benchmark harness;
//   spinlock_scope_mbm.C -- its hand-rolled twin, which runs every thread for
//                           the whole measurement window (see its top comment
//                           for why the twin exists).
//
// WHY
//
// ConcurrentQueue stores each slot inside its critical section, although once
// claimed the slot belongs to the claiming thread and the store could follow
// the unlock; moving the store after the unlock made the queue slower. This
// benchmark takes the observation out of the queue. The first numbers came from
// the prototype of the Google Benchmark harness on a two-socket Xeon 6767P
// (Granite Rapids) server at 128 threads: in scope 120.5 M items/s against out
// of scope 90.0 M items/s (means of 10 repetitions; Google Benchmark's
// items_per_second, see THROUGHPUT ACCOUNTING in spinlock_scope_bm.C). In the
// terms of MECHANISM below, that gap is E1 plus W: out of scope pays E1 on
// every operation, and at 128 threads it also hands the lock over several times
// as often. The prototype's placement of the slot-array pointer, which the
// ptrshared controls reproduce, did not contribute (THE VARIANTS).
//
// WHAT IS MEASURED
//
// Every iteration takes the lock, claims the next slot of a shared ring by
// advancing a shared index under the lock, and stores a per-thread counter into
// the claimed slot. Once claimed, the slot belongs to the claiming thread
// alone, so the store does not need the lock; the variants differ in where it
// is:
//   in_scope  -- the store is the last thing in the critical section, before
//                unlock();
//   out_scope -- the store is the first thing after unlock().
// Both placements are measured on three slot layouts:
//   packed    -- 8-byte slots, eight to a 64-byte line;
//   a64       -- one slot per 64 bytes (alignas(64));
//   a128      -- one slot per 128 bytes (alignas(128)).
// plus a control pair on packed slots, ptrshared, and, on packed and a64 slots
// only, six more placements, all described under THE VARIANTS below:
//   in_scope_lockstore     -- in_scope, plus one store to a scratch word on
//                             the lock's cache line right after lock();
//   in_scope_poststore     -- in_scope, plus one store to a thread-private
//                             dummy word right after unlock();
//   in_scope_postmiss      -- in_scope, plus one store to a second ring right
//                             after unlock();
//   in_scope_inmiss        -- in_scope, plus one store to a second ring right
//                             after the first ring's store, before unlock();
//   in_scope_postmiss_late -- in_scope, plus a full fence right after
//                             unlock(), then one store to a second ring;
//   in_scope_postfence     -- in_scope, plus the same fence right after
//                             unlock(), and no other store.
// That makes 20 variants. The ring has 2^16 slots (512 KiB packed, 4 MiB a64,
// 8 MiB a128); the second ring, allocated only for postmiss, inmiss and
// postmiss_late, has the same size and layout and is indexed with the same
// masked index. The loop body is otherwise identical in all variants: lock,
// [lock-line store], claim, [slot store], [second-ring store], unlock, [slot
// store], [post-unlock store], [second-ring store], [fence], [second-ring
// store], handoff comparison (scope_step() below).
//
// HANDOFFS
//
// Each operation also counts lock handoffs, without an extra store. Every
// thread remembers the unmasked index it claimed last; if this acquisition's
// index is not that one plus 1, another thread held the lock in between, and
// the thread counts one handoff into itself. A thread's first acquisition of a
// run has no previous index and is not counted. Each handoff is counted once,
// by the thread that receives the lock, so the sum over the threads is the
// number of handoffs (less at most one per thread), and handoffs per operation
// is the reciprocal of the mean streak length: 0 when one thread runs alone, 1
// when the lock changes hands on every acquisition. The comparison is
// register-only and sits after the unlock and any post-unlock store, like the
// loop's exit check (see scope_step()). Both harnesses report handoffs per
// operation as handoffs_per_op: a Google Benchmark counter, a CSV column of the
// twin.
//
// MECHANISM
//
// Three mechanisms are confirmed. The measurements behind them were taken with
// GCC 16 at -O3 -march=native and SpinLock::lock() inline. At 1 thread they
// are core cycles per loop iteration, most of them measured with a separate
// harness of controlled loop variants, not included in this repository
// (pinned, 10^8 iterations). Under contention they are medians of the twin
// from four sweeps, named where they are quoted: the first sweep (in_scope,
// out_scope, lockstore, poststore), the handoff sweep (with the handoff
// counter and the second-ring variants), the fence sweep, and the observer
// sweep (--observer at a 10 us period). The lock serializes the operations, so
// a fixed cost per operation shows as a fixed difference in time per
// operation, 1/rate of the whole system, at any thread count.
//
//   E1 -- the early-store penalty (Intel: Broadwell, Emerald Rapids, Granite
//         Rapids). After a locked read-modify-write on a line, a store to that
//         line pays about 6-7 cycles (11-13 on Broadwell) if it is the first
//         or second store after the read-modify-write, counting stores to any
//         line; from the third store on it is free. The unlock is such a
//         store: out_scope's is the second store after the lock's xchg and
//         pays, in_scope's is the third.
//   E2 -- the same-line fast path (Granite Rapids, Emerald Rapids). When the
//         unlock itself was free (the in-scope shape), the next xchg is fast
//         if nothing is stored to another line between them; a store to the
//         lock's own line keeps it fast, and stores to other lines cost 7-12
//         cycles. out_scope, which pays E1, pays no E2 on top. So the 1-thread
//         in-scope advantage needs nothing stored between unlock() and the
//         next lock().
//   W  -- the visibly-free window (contention, x86). How often the lock
//         changes hands, and with it the contended throughput, follows how
//         long per operation the lock is visibly free while its holder is not
//         yet ready to take it again. A slot store before the unlock keeps
//         that window short; the same store after the unlock opens it for the
//         length of the store's miss.
//
// E1 and E2 cost a roughly fixed time per operation and add few handoffs; W is
// what changes with contention. The APPENDIX at the end of this comment keeps
// the hypotheses that were tested and rejected on the way, the questions still
// open, and a side finding.
//
// E1, THE EARLY-STORE PENALTY
//
// After a locked read-modify-write on a line, a plain store to that line --
// the same word or another word on it -- costs extra if it is the first or
// second store after the read-modify-write in program order. The
// read-modify-write arms the line, and E1's window, in which a store to the
// armed line pays, is the next two stores, to whatever lines they go. On the
// Xeon 6767P (Granite Rapids) and the Xeon Gold 6548Y+ (Emerald Rapids), which
// agree cycle for cycle, the first and second stores pay about 7 and 6.5-7
// cycles, the same for the armed word and for another word on its line. On the
// Xeon E7-4850 v4 (Broadwell) the armed word pays more, +11 and +12.5 (the
// unlock's case; the lock loop's in/out gap there is 13 cycles), against +5.5
// and +7 for another word on the line. From the third store on E1's window is
// closed, and it stays closed (tested to the sixth on the Granite Rapids,
// Emerald Rapids and Cascade Lake parts; on Broadwell the armed word still pays
// about +1.5 at the third store). The count of stores
// closes it, not the time: two stores to other lines close it whatever their
// addresses, while a 26-cycle dependency delay in their place leaves +5
// (Granite Rapids). xchg, lock cmpxchg and a seq_cst store compiled to xchg all
// arm the line; a seq_cst store compiled to mov + mfence does not (its mov
// pays E1 like any plain store when an earlier read-modify-write has armed the
// line).
//
// In the benchmark the lock's xchg arms the lock's line, and the unlock is the
// store that pays. In scope, the stores after the xchg are the index, the slot
// and the unlock: the unlock is the third, and free. Out of scope it is the
// second, and pays. Two more loops on Granite Rapids show the count at work:
// without a slot store at all the unlock is only the second store and pays
// (26.0 cycles, as out of scope); in in_scope_lockstore the unlock is the
// fourth store and free, but the lock-line store right after the xchg is the
// first and pays in its place (26.2). Cycles per iteration at 1 thread, in
// scope / out of scope:
//   Intel Xeon E7-4850 v4 (Broadwell)           24.7 / 38.0
//   Intel Xeon Gold 6226R (Cascade Lake)        22.1 / 21.1
//   Intel Xeon Gold 6548Y+ (Emerald Rapids)     21.1 / 28.0
//   Intel Xeon 6767P (Granite Rapids)           20.2 / 26.1
//   AMD EPYC 7F72 and 7742 (Zen 2)              19.1 / 19.1
//   AMD EPYC 9555 (Zen 5)                       21.0 / 21.0
// (The Google Benchmark harness measures 177.3 against 137.5 M items/s at 1
// thread on the Xeon 6767P with packed slots: the same ratio, 1.29.)
// Broadwell, Emerald Rapids and Granite Rapids follow the rule. Cascade Lake
// has a penalty of about 7 cycles but follows a different rule, under which
// out of scope is not slower (APPENDIX, A4). Zen 2 and Zen 5 have none. On an
// NVIDIA Grace (Arm Neoverse V2), out of scope, the shorter critical section,
// is faster at 1 thread (in/out throughput 0.88 in the Google Benchmark
// harness; in the twin 0.88 with packed slots and 0.90 with a64).
//
// Demonstrated with performance counters and the controlled variants on the
// Xeon 6767P, where the extra time is spent at the next xchg (85-88% of the
// samples), not in cache misses, the front end, machine clears,
// store-buffer-full stalls or mispredictions, and with the timing of the same
// variants on the other parts. Under contention E1 stays a roughly fixed cost
// per operation, and in_scope_lockstore hands the lock over at in_scope's rate
// (the handoff sweep). In the first sweep, on the Xeon 6767P, lockstore -
// in_scope is about 1.6 ns (6 cycles) from 1 to 32 threads and 1.8 ns at 64;
// with packed slots it grows to 2.2 ns at 128 threads and 3.5 ns at 256
// (compare the caveat on lockstore under THE VARIANTS), with a64 slots it stays
// at 1.3-1.7 ns.
//
// For lock-free code the same rule reads, from Broadwell to Granite Rapids
// (Cascade Lake included): a CAS or xchg followed within two stores by a plain
// store to another word of the same line pays about 5.5-7 cycles (to the same
// word, on Broadwell, 11-12.5), and two stores to other lines in between remove
// it. Making that store seq_cst (an xchg) costs about 16 cycles more than a
// plain store to an unrelated line, in any position, so it never avoids E1
// cheaply; to the armed word itself it takes another shape (APPENDIX, A3).
//
// The hardware cause is not known (APPENDIX, A4).
//
// E2, THE SAME-LINE FAST PATH
//
// E2 applies when the unlock itself was free, in the in-scope shape (xchg, two
// stores, unlock). The next xchg on the lock is then fast as long as nothing
// is stored to another line between the unlock and it. With m stores to other
// lines there, the loop takes, on Granite Rapids / Emerald Rapids, 20 / 20
// cycles for m = 0, a spike of 32 / 30 (+12 / +10) for m = 1, and a floor of
// 27 / 27 (+7) for m = 2 to 7; the same stores before the unlock cost a cycle
// each (20 + m). Stores to the lock's own line after the unlock (another word,
// or the lock word again) keep the fast path (20 cycles), and the order
// counts: unlock, lock-line store, other-line store gives the spike; unlock,
// other-line store, lock-line store the floor. E2 exists only when the next
// locked operation is on the unlock's line: with the next xchg or lock cmpxchg
// on another line there is none, and lock cmpxchg on the same word shows the
// same shape as xchg (spike / floor 32 / 27 on Granite Rapids, 34 / 29 on
// Emerald Rapids). Loads between the unlock and the next xchg are free. On
// Emerald Rapids the floor is time: two dependent multiplies after the unlock,
// with no store, already give 27, and longer delays cost the same after the
// unlock as before it; no delay reproduces the spike, which needs a store to
// another line to be the last store before the xchg. Granite Rapids cannot be
// tested that way, since ALU delays there do not lengthen the loop at all.
// Broadwell, Cascade Lake, Zen 2 and Zen 5 show no E2.
//
// out_scope, whose unlock pays E1, pays no E2 on top, although its slot store
// sits between the unlock and the next xchg: at 1 thread on Granite Rapids it
// costs 26.1 cycles, as the loop without a slot store (26.0), in which nothing
// is stored after the unlock.
//
// Under contention E2 stays a roughly fixed cost and adds few handoffs. In the
// first sweep, on the Xeon 6767P with packed slots, poststore - in_scope is
// about 3.2 ns up to 32 threads, 3.6 ns at 64, 4.1 ns at 128 and 5.0 ns at 256
// (with a64 slots 2.9-3.7 ns throughout); in the handoff sweep poststore hands
// the lock over 1.15-1.33 times as often as in_scope on the Xeon from 64
// threads up.
//
// It is why, on these two parts, the 1-thread in-scope advantage needs, besides
// the two stores between the xchg and the unlock (E1), no store to another
// line between the unlock and the next lock: a call to a lock() that is not
// inlined (its return address and saved registers), a register spill or a
// result store there removes it, and both orders then cost about 26-28 cycles.
// That is why lock() is forced inline here (scope_step() below).
//
// The hardware cause is not known (APPENDIX, A4). The same shape -- adjacent
// fast, one store in between a spike, more stores a floor -- appears for a
// seq_cst xchg to the word of a preceding locked read-modify-write (APPENDIX,
// A3).
//
// W, THE VISIBLY-FREE WINDOW
//
// Under contention the lock serializes the operations; what the placements
// change is how often it changes hands. In the handoff sweep, at 128 threads,
// out_scope handed the lock over 3-10 times as often as in_scope (streaks of
// about 100-250 operations for in_scope, 25-42 for out_scope; 32-56 for
// out_scope in the observer sweep), and beyond E1 its loss is that handoff
// effect. On x86, where stores commit in order, the handoff rate follows the
// visibly-free window: the time per operation during which the lock is free,
// as another core sees it, while its holder is not yet ready to take it again.
// In that window a waiter's xchg wins the lock. A slot store inside the
// critical section (older than the unlock) keeps the unlock from becoming
// visible until the store has committed: while the holder is stalled on the
// store's miss, the lock does not look free. The same store after the unlock
// leaves the lock visibly free for the whole miss, because the holder's next
// xchg has to drain the store buffer, the missing store included, before it
// can take the lock again. What counts is time after the unlock is visible,
// not the presence of a store: E2 costs time but adds few handoffs.
//
// Two results demonstrate it on x86. First, the fence variants insert a window
// with no store: in_scope_postfence stalls the holder after its unlock has
// become visible (the fence waits for the store buffer to drain and then costs
// its own time), and in_scope_postmiss_late adds a miss after the fence. In
// the fence sweep, at every thread count from 2 up on the Xeon 6767P and the
// EPYC 9555, the handoff rates order in_scope < postfence < postmiss_late;
// postfence is near out_scope's rate, below it on the Xeon at 4 to 16 threads
// with packed slots and at 32 with a64, at or above it elsewhere. Grace shows
// the same direction of change in handoffs only, at tiny absolute rates. (The
// fence dominates their time per operation, 14-17 ns at 1 thread on x86: read
// them from handoffs, not time.)
//
// Second, the twin's observer (THE OBSERVER in spinlock_scope_mbm.C) samples
// the lock word under contention. It measures all the time the lock is
// visibly free, the holder's own return to it included, not only the part in
// which the holder is not ready; what is demonstrated is that this measure
// ranks the variants exactly as their handoff rates do, in every cell from 16
// threads up, on both x86 parts. Visibly-free ns per operation (handoffs per
// 1000 operations) in the observer sweep, at 128 threads on the Xeon 6767P
// (Xeon) and 127 on the EPYC 9555 (EPYC; one CPU left for the observer):
//                 in_scope    out_scope    postmiss    postfence  postmiss_late
//   Xeon packed  2.7 (8.5)   6.0 (30.6)  3.6 (11.4)  17.9 (46.2)    26.2 (76.4)
//   Xeon a64     2.7 (6.8)   6.3 (31.3)   4.0 (9.8)  18.0 (34.1)    32.4 (82.5)
//   EPYC packed  3.7 (4.6)   9.3 (18.0)  7.4 (10.5)  32.1 (46.6)     88.5 (146)
//   EPYC a64     3.8 (4.1)  12.5 (24.5)   5.0 (5.9)  68.4 (84.7)      216 (366)
// Here postfence has 5-21 times in_scope's handoffs and 1.1-3.5 times
// out_scope's, and postmiss_late 1.6-4.3 times postfence's. The Xeon's
// 128-thread cells run 129 threads on its 128 physical cores; the ranking is
// the same at 16 to 64 threads, within the physical cores. At a 1 us sampling
// period the ranking is the same except in two cells on the Xeon where
// out_scope and postfence nearly tie (packed at 16 threads, a64 at 128).
// Calibrated with --validate-observer, the observer showed no bias at the
// 10 us period beyond the worker's own timing: a roughly constant 20-35 ns of
// extra held time per cycle, 1.2-1.9 clock readings, over programmed cycles of
// 97 ns to 1.1 us. At 1 us it measurably slows the workers, so the 10 us data
// are the primary ones. Under contention its loads may still be served in an
// order that depends on the line's state (THE OBSERVER); the demonstration
// rests on the ranking.
//
// The relation is monotone but not linear, on both parts. Handoffs per 1000
// operations per nanosecond of window, out_scope against postfence /
// postmiss_late: Xeon packed 5.1 against 2.6 / 2.9, a64 5.0 against 1.9 / 2.5;
// EPYC packed 1.9 against 1.45 / 1.65, a64 2.0 against 1.24 / 1.7 (why is
// open: APPENDIX, A4). in_scope_postmiss, a second store that misses after the
// unlock with the slot store kept in scope, is consistent with the rule: in
// three cells its window is near in_scope's, and so are its handoffs; on the
// EPYC with packed slots it is a partial case, with a window near out_scope's
// and handoffs halfway between (10.5 against 4.6 and 18.0). Why its window
// differs between the cells is open (A4).
//
// How the three add up, on the Xeon 6767P with packed slots (the first sweep,
// nanoseconds per operation over in_scope):
//   threads   out_scope   lockstore   poststore
//         1        1.64        1.68        3.28
//        16        2.06        1.57        3.11
//        64        3.46        1.84        3.55
//       128        6.67        2.21        4.12
//       256       15.51        3.49        5.00
// lockstore - in_scope is E1 and poststore - in_scope is E2; out_scope pays E1
// but not E2. Up to 8 threads the out-of-scope loss is essentially E1 (out -
// in 1.64-1.86 ns against lockstore - in 1.60-1.68; at 8 threads 0.26 ns is
// left over); by 16 threads W has started on top of it (2.06 against 1.57),
// and from about 32 threads it grows: at 256 threads it is 12-14 of the
// 15.5 ns. a64 slots show the same shape. On the EPYC 9555 lockstore and
// poststore stay within +-0.7 ns of in_scope (no E1, no E2), and out_scope
// costs under 0.4 ns up to 16 threads, then 0.64, 2.99 and 8.54 ns at 32, 64
// and 128 threads with packed slots (20.5 ns at 128 with a64): W alone. Grace
// is in a regime of its own (APPENDIX, A4).
//
// THE VARIANTS
//
// The a64 and a128 variants were added because slot size changes how often the
// ring store misses. With packed slots, a holder in a streak claims consecutive
// slots, which share a line, so its store misses only once per eight
// operations; with one slot per line every store touches a new line. A miss
// inside the critical section lengthens it; a miss after the unlock opens the
// visibly-free window (W under MECHANISM). With packed slots out of scope, a
// thread's store to its slot can also race the next holder's store to a
// neighbouring slot on the same line, a conflict that the lock serializes in
// scope and that one slot per line removes. The handoff counter shows which of
// these changes how the lock is handed off.
//
// a128 is there because a64 does not mean the same thing on every host: with
// 128-byte cache lines (Apple M-series) a64 still puts two slots on a line,
// and on x86 the L2 spatial prefetcher pairs 64-byte lines into 128-byte
// sectors. a128 is one slot per line and per sector everywhere.
//
// The ptrshared controls reproduce the prototype's placement of the pointer to
// the slot array: right after the slot index, on the index's 64-byte line. Only
// that placement is reproduced; the lock and the index are laid out as in the
// main variants. (In the prototype they were separate alignas(64) globals,
// which in the one build that was inspected, clang on x86, ended up on adjacent
// lines of one 128-byte sector; other builds were not checked.) The pointer is
// loaded on every operation, from the line that the next holder writes (the
// index's). The controls measured the same as the main packed layout: in the
// first sweep (MECHANISM), on both x86 parts and at every thread count, within
// 4% on the Xeon 6767P and 5% on the EPYC 9555 (on the Xeon at 1 thread,
// in/out 177.4 / 137.4 M items/s packed,
// 177.5 / 137.4 ptrshared). So the prototype's in/out gap did not come from the
// pointer's placement. (Out of scope the pointer is loaded after the unlock in
// program order, but that load can execute before the unlock commits; see
// full_fence() below.)
//
// in_scope_lockstore is in-scope order plus one relaxed store to
// Shared::lock_line_word, a word on the lock's line that is not the lock word,
// right after lock() and before the index update. It keeps in_scope's handoff
// timing -- the slot store is still before the unlock -- but pays E1 like
// out_scope. E1 is an absolute cost per operation, not a ratio: under
// contention the lock serializes, so a full carry-over adds its ~6 cycles to a
// critical section plus handoff that is much longer than at 1 thread, and shows
// up as far less than 1.3x in throughput. So compare times per operation:
// 1/rate(lockstore) - 1/rate(in_scope) is E1 in the contended regime (it
// carries over: about 1.6 ns, 6 cycles, on the Xeon 6767P; E1 under MECHANISM),
// and comparing that difference with 1/rate(out_scope) - 1/rate(in_scope) gives
// the share of the out-of-scope loss that is E1 rather than W. One caveat:
// spinning waiters' TTAS reads pull the lock's line into the Shared state
// during the critical section, so a store to it has to take the line back.
// in_scope pays such a transfer at most once per critical section, at the
// unlock; lockstore can pay one more, right after the xchg, and the stores
// behind it drain in order after it.
//
// in_scope_poststore is in-scope order plus one relaxed store to a per-thread
// dummy word (post_unlock_block below) right after unlock() and before the
// loop's exit check: E2 alone. It stands in for real code, which usually stores
// something between two lock operations, most often to thread-private data.
// Under contention it shows that E2 survives as a roughly fixed cost that adds
// few handoffs (E2 under MECHANISM); as with lockstore, read it as a
// difference in time per operation from in_scope.
//
// in_scope_postmiss and in_scope_inmiss were added to test a hypothesis about
// out_scope's handoffs that the results rejected (APPENDIX, A1); postmiss has
// since become one of W's cases. They store into a second ring
// (Shared::slots2) at the same masked index as the first, so the second store
// misses the same way the slot store does (its line was last written a lap
// earlier by whoever claimed that index). postmiss puts it right after the
// unlock, where out_scope's slot store is, while keeping the first ring's store
// in scope; inmiss, the control, puts it inside the critical section, right
// after the first ring's store.
//
// in_scope_postmiss_late and in_scope_postfence demonstrate W (MECHANISM) by
// inserting a visibly-free window after the unlock. postmiss_late is postmiss
// with full_fence() (below) between the unlock and the second-ring store: the
// same ring, the same index, the same value. The fence writes no memory, but
// whether draining the store buffer right after the unlock interacts with E2 is
// not known: postfence measures whatever the fence costs. The fence is to keep
// the second ring's pointer from being loaded, and so the store's address from
// being known, until the unlock is globally visible (full_fence() says what of
// that is guaranteed and what is assumed). postfence is in-scope order and the
// same fence right after the unlock, with no second ring: the fence's own cost
// and its own effect on the handoffs, so that postmiss_late - postfence
// isolates a miss that starts only after the unlock is visible. The fence
// dominates their time per operation, so read them from their handoff rates
// rather than from time.
//
// LAYOUT
//
// Everything that threads share is laid out in 128-byte blocks, the line size
// on Apple M-series and the prefetch sector on x86, so that no two of these
// fields share a line or a sector by accident (struct Shared below): the lock
// (with the scratch word of the lockstore variants on its line, which no other
// variant touches), the slot index, and the pointers to the two rings (except
// in the ptrshared controls, where they follow the index on purpose; those
// variants never read the second one). Each harness puts its own shared state
// on further 128-byte blocks after these: the Google Benchmark harness a
// pointer to its per-thread records, the twin its measure and stop flags (its
// start latch is a local of the main thread). The ring pointers are never
// written during the measured loop, so on their own line they stay in the
// Shared state in every reader's cache. The dummy word of the poststore
// variants is not shared at all: each thread has its own, alone on a 128-byte
// block of thread-local storage (post_unlock_block).
//
// APPENDIX: HYPOTHESES TESTED AND REJECTED, OPEN QUESTIONS AND SIDE FINDINGS
//
// Not part of the explanation above, and not needed to follow it: the
// hypotheses tested on the way to it and rejected, with the results that
// decided them, the questions still open, and a side finding. The numbers were
// measured as those under MECHANISM; the 1-thread results of A2 to A5 come
// from separate harnesses, not included in this repository.
//
// A1. Rejected: "a store that misses between the unlock and the next xchg
// causes out_scope's handoffs." The idea: out_scope's slot store must fetch its
// line from another core (whoever claimed that index a lap earlier), the
// holder's next xchg waits for it with the lock visibly free, and a waiter
// takes the lock. in_scope_postmiss and in_scope_inmiss were built to test it
// (THE VARIANTS): if it held, postmiss would match out_scope in handoffs and in
// cost, and inmiss would match in_scope's handoffs. At 128 threads (the
// handoff sweep, a separate sweep from the observer sweep under W; the twin's
// medians; handoffs per operation, then ns per operation over in_scope):
//                   in   out  postmiss      out  postmiss
//   Xeon packed   .009  .031      .011     7.05      4.32
//   Xeon a64      .007  .033      .011     7.07      4.94
//   EPYC packed   .004  .024      .012    10.64      9.19
//   EPYC a64      .004  .040      .006    21.13      2.79
// (Xeon = Xeon 6767P, EPYC = EPYC 9555.) inmiss behaved like in_scope, as
// predicted, but postmiss handed the lock over about as rarely as in_scope in
// three of the four cells, at far less cost than out_scope (on the Xeon, part
// of its cost is E2); only the EPYC with packed slots came close in cost, with
// half out_scope's handoffs. A store that misses after the unlock is not
// enough by itself; W replaced the hypothesis.
//
// A2. Rejected as a predictor: the visibly-free window measured with one
// worker. A pinned worker ran the real lock loop while a helper thread on
// another core kept the ring lines Modified ahead of it, so that every ring
// store was a cross-core miss, and an observer thread sampled the lock word
// about once a microsecond. Visibly-free ns per iteration, Xeon 6767P packed /
// a64, EPYC 9555 packed / a64:
//   in_scope    2.2 / 1.6 / 3.5 / 2.7
//   out_scope   12.4 / 14.0 / 8.9 / 9.0
//   postmiss    4.0 / 9.8 / 4.6-5.1 / 5.4-6.5
// out_scope's window was 2.5-9 times in_scope's (Xeon 5.8 and 8.9, EPYC 2.5
// and 3.4). As a predictor of the contended postmiss it failed in three of the
// four cells, one of them partly: on the Xeon with a64 slots postmiss had 70%
// of out_scope's window but in_scope-like handoffs; on the EPYC with a64 slots
// 60-72% of it and in_scope-like handoffs; on the EPYC with packed slots about
// half of it, where under contention its window was near out_scope's and its
// handoffs halfway between in_scope's and out_scope's. Measured under
// contention (W), the window ranks them as their handoffs. What the experiment
// did demonstrate: the RFO of a store after the unlock overlaps that of the
// in-scope store (by the Xeon's counters, about 3 demand RFOs outstanding with
// postmiss against about 1 with a fence between them with packed slots, 11
// against 1.9 with a64) when the two rings' line crossings (where a streak
// moves on to a new line) coincide, and shifted crossings expose the miss
// (postmiss then comes close to out_scope); and a store after the unlock to
// the same slot or to an adjacent element behaves like in_scope: line
// ownership decides, with no special handling of the same address.
//
// A3. The floor of a seq_cst xchg to the same word. After a locked
// read-modify-write, a seq_cst store compiled to xchg to the same word costs
// nothing extra as the first store after it (34 cycles, the same as to another
// line), 11 (Granite Rapids) or 9 (Emerald Rapids) cycles more as the second,
// and from the third on the loop takes a flat 41 cycles while the baseline with
// stores to other lines rises by a cycle per store: a floor, not E1's window.
// To another word on the line it is never penalized. Rejected: a fixed latency
// of dependent pairs (two adjacent dependent xchg to one address cost 34
// cycles, as two independent ones do); and, for this case, the drain model,
// that an xchg waits only for the penalized stores ahead of it (the stores
// ahead here go to other lines and are not penalized). Demonstrated: on
// Emerald Rapids the floor is time (any separation -- one multiply, two stores,
// a dependency chain gating the second xchg's data -- gives max(~41,
// baseline)); and a plain load of the word just written by a locked
// read-modify-write is blocked from store forwarding (+14-16 cycles), after
// which the second xchg costs the same as one to an unrelated line. Hypothesis,
// no counter shows it: an adjacent same-address pair keeps the line locked
// across both; anything in between releases it, and the second
// read-modify-write then waits until ~41 cycles after the first started.
//
// A4. Open questions and unconfirmed observations.
//   - Why the contended postmiss window is near out_scope's on the EPYC 9555
//     with packed slots and near in_scope's in the other three cells. No
//     factor that tells the cells apart is known.
//   - Why the handoffs per nanosecond of window differ between the variants,
//     on both parts (W): out_scope's window comes with 1.8-2.6 times the
//     handoffs per nanosecond of the fence variants' on the Xeon 6767P, and
//     1.15-1.6 times on the EPYC 9555. The hypothesis: out_scope's window
//     opens while the waiters are still in their burst of attempts.
//   - Grace's regime. The lock almost never changes hands (1e-6 to 4e-5
//     handoffs per operation, streaks of 25,000 to 1,000,000 operations) and
//     starves threads heavily (some threads get almost no operations, from 8
//     threads up in one sweep and from 32 in another), yet with packed slots
//     at 128 threads in_scope ran at 65 and out_scope at 35 M items/s. The
//     cost there is presumably the waiters' interference with the winning
//     holder, not handoffs (reasoned, not measured). The in/out order was not
//     stable across builds (out_scope faster in one sweep, in_scope faster
//     with packed slots in the next), so no conclusion is drawn from it.
//   - Cascade Lake's rule for E1: a penalty of about 7 cycles, which a slot
//     store after the unlock (its address from a load) cancels and a store to
//     an unrelated line does not. Its timings also vary with code layout (the
//     JCC erratum changes which loops the decoded-uop cache delivers), so only
//     its +7 at the first and second store is firm.
//   - E2 and the unlock's visibility. In the 1-thread harness of A2 (Xeon
//     6767P, the lock word sampled about once a microsecond), a store after
//     the unlock to the slot just written paid E2 (6.0-13.5 cycles per
//     iteration over in_scope) and left the visibly-free window at 2.3 ns per
//     iteration against 2.15 without it: E2's cycles appear to be spent before
//     the unlock becomes visible. A supporting observation, not established;
//     it fits E2 adding few handoffs under contention.
//   - The hardware causes of E1 and E2. For E1 the working hypothesis is a
//     window counted in stores, not time, after a locked read-modify-write, in
//     which younger stores to the same line are rejected and retried; for E2,
//     that the line is held hot in the store-commit path for a short time
//     after the unlock commits. No counter shows either.
//
// A5. A side finding on Granite Rapids, confirmed in the runs checked: two
// RFOs in flight to addresses congruent modulo 64 KiB (tested at 64 KiB and
// 4 MiB strides) serialize, and modulo 4 KiB they partly do. With second-ring
// stores aliased that way, postmiss with a64 slots took 577 cycles per
// iteration and left the lock visibly free for 150 ns. The two rings of this
// benchmark, in the builds checked on the Xeon 6767P, are separate mappings a
// ring plus 4 KiB apart (0x81000 bytes packed, 0x401000 a64): congruent modulo
// 4 KiB, which partly serializes at 1 thread; yet the contended postmiss
// window on the Xeon is near in_scope's (W), so the aliasing does not dominate
// there. An allocator that placed the rings congruent modulo 64 KiB would
// change the second-ring variants' results on that part.
#ifndef INCLUDED_SPINLOCK_SCOPE_COMMON_H
#define INCLUDED_SPINLOCK_SCOPE_COMMON_H
#include <unistd.h>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "spinlock.h"

// Where the slot store goes relative to unlock().
enum class Store {
  in_scope,                             // before unlock(), inside the critical section
  out_scope,                            // right after unlock()
  in_scope_lockstore,                   // as in_scope, plus a store to Shared::lock_line_word
                                        // right after lock()
  in_scope_poststore,                   // as in_scope, plus a store to the thread's own
                                        // post_unlock_block right after unlock()
  in_scope_postmiss,                    // as in_scope, plus a store to the second ring's slot
                                        // right after unlock()
  in_scope_inmiss,                      // as in_scope, plus a store to the second ring's slot
                                        // right after the first ring's, before unlock()
  in_scope_postmiss_late,               // as in_scope, plus full_fence() right after unlock(),
                                        // then a store to the second ring's slot
  in_scope_postfence,                   // as in_scope, plus full_fence() right after unlock()
};

// Whether a variant stores into the second ring (Shared::slots2); only those
// variants allocate it.
template <Store store>
inline constexpr bool uses_second_ring =
    (store == Store::in_scope_postmiss || store == Store::in_scope_inmiss || store == Store::in_scope_postmiss_late);

// Whether a variant puts full_fence() right after unlock().
template <Store store>
inline constexpr bool fences_after_unlock =
    (store == Store::in_scope_postmiss_late || store == Store::in_scope_postfence);

// Where the pointer to the slot array lives.
enum class PtrLine {
  own,                                  // its own 128-byte block (the main variants)
  index,                                // right after the slot index, on its line (the
                                        // prototype's layout, kept as the ptrshared control)
};

// A slot of the ring, aligned, and therefore padded, to Align bytes.
template <size_t Align>
struct alignas(Align) RingSlot {
  unsigned long v;                      // the payload; stored once per claim, never read
};

using PackedSlot = RingSlot<alignof(unsigned long)>;    // eight to a 64-byte line
using Slot64 = RingSlot<64>;                            // one per 64 bytes
using Slot128 = RingSlot<128>;                          // one per 128 bytes
static_assert(sizeof(PackedSlot) == 8);
static_assert(sizeof(Slot64) == 64);
static_assert(sizeof(Slot128) == 128);

// What store_relaxed() of a slot's payload needs of a slot type: the payload at
// offset 0, aligned for atomic_ref, and a lock-free atomic_ref so that the
// store is one instruction.
template <typename Slot>
consteval bool slot_ok() {
  static_assert(offsetof(Slot, v) == 0);
  static_assert(alignof(Slot) >= std::atomic_ref<unsigned long>::required_alignment);
  static_assert(std::atomic_ref<unsigned long>::is_always_lock_free);
  return true;
} // slot_ok()
static_assert(slot_ok<PackedSlot>());
static_assert(slot_ok<Slot64>());
static_assert(slot_ok<Slot128>());

// The number of slots in the ring. It must be a power of two: the index wraps
// by masking with kSlots - 1.
inline constexpr size_t kSlots = size_t{1} << 16;
static_assert(std::has_single_bit(kSlots));

// Wall-clock seconds of measurement and of warm-up per run: the fixed values of
// the Google Benchmark harness, and the defaults of the twin.
inline constexpr double kWindowSeconds = 1.0;
inline constexpr double kWarmupSeconds = 0.5;

// The state that all threads of one variant share and that the measured
// operation touches; see LAYOUT above. Every separately placed member is
// alignas(128), which also makes the struct 128-aligned and pads it to a
// multiple of 128, so a harness that embeds it and places its own members
// alignas(128) after it keeps them off these blocks. The offsets are checked
// by shared_layout_ok() below.
template <typename Slot, PtrLine ptr_line>
struct Shared {
  alignas(128) SpinLock lock;           // the lock every operation takes
  // Scratch word on the lock's cache line, written only by the
  // in_scope_lockstore variants (right after lock()) and never read.
  unsigned long lock_line_word {};
  alignas(128) size_t slot_idx {};      // next slot to claim (before masking); guarded by lock
  // Non-owning view of the slot array, set by the harness before the threads
  // start the measured loop and cleared after they stop. Either on its own
  // block or right after slot_idx.
  alignas(ptr_line == PtrLine::own ? 128 : alignof(Slot*)) Slot* slots {};
  // Non-owning view of the second ring, stored into by the postmiss, inmiss and
  // postmiss_late variants at the same masked index as the first; null in the
  // others (see uses_second_ring). Set and cleared like `slots`, and read-only
  // in between, so it shares the slots pointer's block (in the ptrshared
  // layout, which never reads it, the index's line).
  Slot* slots2 {};
}; // struct Shared

// The layout claims of LAYOUT above for one instantiation of Shared; asserted
// below for every one that the variants use.
template <typename Slot, PtrLine ptr_line>
consteval bool shared_layout_ok() {
  using SharedState = Shared<Slot, ptr_line>;
  static_assert(alignof(SharedState) == 128);
  static_assert(offsetof(SharedState, lock) == 0);
  // lock_line_word follows the lock word on the same 64-byte line.
  static_assert(offsetof(SharedState, lock_line_word) >= sizeof(SpinLock));
  static_assert(offsetof(SharedState, lock_line_word) + sizeof(unsigned long) <= 64);
  static_assert(alignof(unsigned long) >= std::atomic_ref<unsigned long>::required_alignment);
  static_assert(offsetof(SharedState, slot_idx) == 128);
  static_assert(offsetof(SharedState, slots) == (ptr_line == PtrLine::own ? 256 : 128 + sizeof(size_t)));
  static_assert(offsetof(SharedState, slots2) == offsetof(SharedState, slots) + sizeof(Slot*));
  static_assert(sizeof(SharedState) == (ptr_line == PtrLine::own ? 384 : 256));
  return true;
} // shared_layout_ok()
static_assert(shared_layout_ok<PackedSlot, PtrLine::own>());
static_assert(shared_layout_ok<Slot64, PtrLine::own>());
static_assert(shared_layout_ok<Slot128, PtrLine::own>());
static_assert(shared_layout_ok<PackedSlot, PtrLine::index>());

// The dummy word of the in_scope_poststore variants, padded to a 128-byte block
// of its own.
struct alignas(128) PostUnlockBlock {
  unsigned long word;                   // stored once per operation, never read
};
static_assert(sizeof(PostUnlockBlock) == 128);

// Each thread's own post-unlock dummy block. Per thread, not shared: a single
// word shared by all threads would, under contention, make every handoff pull
// that line from the previous holder's core before the next holder's xchg
// (the locked xchg drains the store buffer, so it waits for its own
// post-unlock store to get the line). That is a coherence cost, not the
// store-order cost the variant exists to expose, and it would act on exactly
// the handoff dynamics the contended runs try to separate; real code that
// stores between two lock operations usually writes thread-private data.
// thread_local rather than an array indexed by thread: scope_step() needs no
// thread index (neither harness passes one), and there is still a single
// store, with no load of an array base: on x86-64 one %fs-relative store; on
// aarch64 the address comes from TPIDR_EL0 (mrs, add, str; GCC does not hoist
// the mrs out of the poststore loop, so it is one register read per operation
// more, still one store). The padding keeps any other thread-local variable
// off the block. constinit: no dynamic initialization, so no TLS guard or
// wrapper call on access.
inline thread_local constinit PostUnlockBlock post_unlock_block {};

// A thread's handoff counter; see HANDOFFS above. Kept by each thread in its
// loop (a local, so the compiler keeps both fields in registers) and advanced
// by scope_step().
struct HandoffCounter {
  size_t prev = 0;                      // the unmasked index this thread claimed last
  unsigned long breaks = 0;             // acquisitions whose index was not prev + 1
};

// Store `value` into `word` (a slot's payload, Shared::lock_line_word or
// post_unlock_block.word). The store is a relaxed atomic one, which compiles
// to a plain store instruction (mov on x86, str on aarch64) but, unlike a plain
// store, is not a data race if two threads ever store to the same slot at once
// (see the wrap-around note at scope_step()). Being atomic, it is also not
// removed by the optimizer although nothing reads the word.
inline void store_relaxed(unsigned long& word, unsigned long value) {
  std::atomic_ref<unsigned long>(word).store(value, std::memory_order_relaxed);
}

// A full hardware memory fence that stores nothing, and a compiler barrier: the
// fence of the in_scope_postmiss_late and in_scope_postfence variants, right
// after unlock() (see THE VARIANTS above). Its job there is to keep the
// second-ring store of postmiss_late from starting its RFO before the unlock
// is globally visible. Architecturally it orders visibility: no load or store
// after it becomes globally visible before every load and store before it,
// the unlock included, has. The "memory" clobber keeps the compiler from moving
// memory accesses across it, so the second ring's pointer, which the store's
// address depends on, is loaded after it. That the hardware also holds that
// load back until the fence completes, rather than executing it early and
// checking it later, is assumed, not verified.
//
// A load-based address dependency without a fence would not do: a load after
// the unlock executes out of order before the unlock commits, or takes its
// value from the store buffer, so the address would still be known early.
//
// Not std::atomic_thread_fence(std::memory_order_seq_cst): GCC 16 and clang 22
// with the tunings used here (-march=native on Zen 4) emit that on x86-64 as
// `lock or $0,(%rsp)` (other tunings may emit mfence), a locked
// read-modify-write of the stack. It is a store to another line between the
// unlock and the next xchg, which on Granite Rapids and Emerald Rapids pays E2,
// and a locked read-modify-write, after which the next stores to the stack's
// line can pay E1 (MECHANISM above). mfence writes no memory.
// On aarch64, dmb ish: a full barrier over the inner shareable domain, which
// takes in every core.
[[gnu::always_inline]] inline void full_fence() {
#if defined(__x86_64__) || defined(__i386__)
  asm volatile("mfence" ::: "memory");
#elif defined(__aarch64__)
  asm volatile("dmb ish" ::: "memory");
#else
#error "full_fence(): no fence instruction for this architecture"
#endif
} // full_fence()

// The measured operation, one call per loop iteration in both harnesses: lock,
// claim a slot, store, unlock, with the store on the side of the unlock that
// `store` selects, then the handoff comparison. The other variants add one
// store of the incremented count (the value the slot store writes, so that the
// compiler needs no register for another value): in_scope_lockstore into
// s.lock_line_word right after lock(); in_scope_poststore into this thread's
// post_unlock_block right after unlock(); in_scope_postmiss into the second
// ring's slot right after unlock(); in_scope_inmiss into the second ring's
// slot right after the first ring's store, before unlock();
// in_scope_postmiss_late into the second ring's slot after unlock() and
// full_fence(). in_scope_postfence adds full_fence() right after unlock() and
// no store.
//   s        -- the shared state; s.slots, and for the variants that use the
//               second ring (uses_second_ring) s.slots2, must point to at least
//               idx_mask + 1 slots;
//   idx_mask -- the number of slots minus one (the count is a power of two);
//   count    -- the calling thread's running count of operations: incremented
//               once per call, and the incremented value is what is stored, so
//               every store writes a new value;
//   handoffs -- the calling thread's handoff counter: the unmasked index
//               claimed by this call is compared with the previous one (see
//               HANDOFFS above). The caller discards the first comparison of
//               a run, which has no previous index.
// The ring is large enough that a slot is normally not claimed again while its
// previous owner is still storing to it; if a preempted thread's out-of-scope
// store ever does overlap the next owner's, both stores are atomic, so this is
// not a data race.
//
// always_inline and flatten: the order of the stores, the unlock, the next
// lock and whatever the caller does in between (its loop-exit check) is what
// is measured, so the whole operation, SpinLock::lock() included, must be
// inlined into both harnesses' loops on every compiler, not left to the
// inliner. always_inline puts scope_step() into the caller's loop; flatten
// inlines everything scope_step() calls: lock() and unlock(), and, on lock()'s
// back-off path, the sleep wrappers down to their nanosleep() calls, which stay
// calls and are off the steady-state path. Left to itself, GCC 16 kept lock()
// out of line in the twin, called from every loop, and that function pushes
// its return address and three saved registers before its xchg. Those are
// stores between the unlock and the next xchg, which on Granite Rapids cost
// extra when the unlock is in in-scope position (E2 under MECHANISM above):
// the twin's in-scope loop cost 28 cycles against 20 in the Google Benchmark
// harness, where GCC did inline lock(). clang 22 kept lock() out of line in
// both harnesses. So the loop shape, and on Intel the 1-thread result,
// depended on the inliner; with flatten it does not. A header change
// (spinlock.h) would have done the same for every user of the lock; this keeps
// the change to the benchmark. The same stores can come back as register
// spills: with lock()'s back-off inlined, every value the caller's loop
// carries must sit in a callee-saved register, so each harness keeps its loop
// in a noinline function of its own (timed_loop(), run_phase()).
//
// The handoff comparison is register-only, and it must come after the unlock
// and any post-unlock store, like the caller's exit check, so that it does not
// lengthen the critical section. The compiler is otherwise free to schedule
// register arithmetic before the unlock store; the empty asm statement at the
// end, which takes the claimed index as an in/out operand and clobbers memory,
// emits no instruction, but the comparison depends on its output and it cannot
// move above the stores before it. (prev + 1 does not depend on it, and the
// compiler may still compute it inside the critical section: one register
// increment.)
template <Store store, typename Slot, PtrLine ptr_line>
[[gnu::always_inline, gnu::flatten]] inline void scope_step(Shared<Slot, ptr_line>& s, size_t idx_mask,
                                                           unsigned long& count, HandoffCounter& handoffs) {
  s.lock.lock();
  if constexpr (store == Store::in_scope_lockstore) store_relaxed(s.lock_line_word, count + 1);
  size_t claimed = s.slot_idx++;        // unmasked: the handoff comparison needs the full index
  const size_t my_slot_idx = (claimed & idx_mask);
  if constexpr (store != Store::out_scope) store_relaxed(s.slots[my_slot_idx].v, ++count);
  if constexpr (store == Store::in_scope_inmiss) store_relaxed(s.slots2[my_slot_idx].v, count);
  s.lock.unlock();
  if constexpr (store == Store::out_scope) store_relaxed(s.slots[my_slot_idx].v, ++count);
  if constexpr (store == Store::in_scope_poststore) store_relaxed(post_unlock_block.word, count);
  if constexpr (store == Store::in_scope_postmiss) store_relaxed(s.slots2[my_slot_idx].v, count);
  if constexpr (fences_after_unlock<store>) full_fence();
  if constexpr (store == Store::in_scope_postmiss_late) store_relaxed(s.slots2[my_slot_idx].v, count);
  asm volatile("" : "+r"(claimed) : : "memory");
  handoffs.breaks += (claimed != handoffs.prev + 1);
  handoffs.prev = claimed;
} // scope_step()

// Allocate a ring of `n` slots and write every slot once, so that its pages
// are faulted in before the measurement starts. Throws std::bad_alloc if the
// allocation fails.
template <typename Slot>
std::unique_ptr<Slot[]> make_ring(size_t n) {
  std::unique_ptr<Slot[]> ring = std::make_unique_for_overwrite<Slot[]>(n);
  for (size_t i = 0; i != n; ++i) ring[i].v = i;
  return ring;
}

// Call f.template operator()<Slot, store, ptr_line>(name) once per variant, in
// the order the variants run at each thread count: the in/out pair of each
// layout back to back, followed on packed and a64 by the lockstore, poststore,
// postmiss, inmiss, postmiss_late and postfence variants. That makes 20
// variants: 8 on packed, 8 on a64, 2 on a128 and 2 on packed_ptrshared. `name`
// is the variant's run name, BM_spinlock_<placement>_<layout> with placement
// in_scope, out_scope, in_scope_lockstore, in_scope_poststore,
// in_scope_postmiss, in_scope_inmiss, in_scope_postmiss_late or
// in_scope_postfence, shared by both harnesses (the Google Benchmark one
// appends its arguments to it). f is typically a lambda with an explicit
// template parameter list:
//   for_each_variant([&]<typename Slot, Store store, PtrLine ptr_line>(const char* name) { ... });
template <typename F>
void for_each_variant(F&& f) {
  f.template operator()<PackedSlot, Store::in_scope, PtrLine::own>("BM_spinlock_in_scope_packed");
  f.template operator()<PackedSlot, Store::out_scope, PtrLine::own>("BM_spinlock_out_scope_packed");
  f.template operator()<PackedSlot, Store::in_scope_lockstore, PtrLine::own>("BM_spinlock_in_scope_lockstore_packed");
  f.template operator()<PackedSlot, Store::in_scope_poststore, PtrLine::own>("BM_spinlock_in_scope_poststore_packed");
  f.template operator()<PackedSlot, Store::in_scope_postmiss, PtrLine::own>("BM_spinlock_in_scope_postmiss_packed");
  f.template operator()<PackedSlot, Store::in_scope_inmiss, PtrLine::own>("BM_spinlock_in_scope_inmiss_packed");
  f.template operator()<PackedSlot, Store::in_scope_postmiss_late, PtrLine::own>(
      "BM_spinlock_in_scope_postmiss_late_packed");
  f.template operator()<PackedSlot, Store::in_scope_postfence, PtrLine::own>("BM_spinlock_in_scope_postfence_packed");
  f.template operator()<Slot64, Store::in_scope, PtrLine::own>("BM_spinlock_in_scope_a64");
  f.template operator()<Slot64, Store::out_scope, PtrLine::own>("BM_spinlock_out_scope_a64");
  f.template operator()<Slot64, Store::in_scope_lockstore, PtrLine::own>("BM_spinlock_in_scope_lockstore_a64");
  f.template operator()<Slot64, Store::in_scope_poststore, PtrLine::own>("BM_spinlock_in_scope_poststore_a64");
  f.template operator()<Slot64, Store::in_scope_postmiss, PtrLine::own>("BM_spinlock_in_scope_postmiss_a64");
  f.template operator()<Slot64, Store::in_scope_inmiss, PtrLine::own>("BM_spinlock_in_scope_inmiss_a64");
  f.template operator()<Slot64, Store::in_scope_postmiss_late, PtrLine::own>("BM_spinlock_in_scope_postmiss_late_a64");
  f.template operator()<Slot64, Store::in_scope_postfence, PtrLine::own>("BM_spinlock_in_scope_postfence_a64");
  f.template operator()<Slot128, Store::in_scope, PtrLine::own>("BM_spinlock_in_scope_a128");
  f.template operator()<Slot128, Store::out_scope, PtrLine::own>("BM_spinlock_out_scope_a128");
  f.template operator()<PackedSlot, Store::in_scope, PtrLine::index>("BM_spinlock_in_scope_packed_ptrshared");
  f.template operator()<PackedSlot, Store::out_scope, PtrLine::index>("BM_spinlock_out_scope_packed_ptrshared");
} // for_each_variant()

// The default thread counts: the powers of two from 1 up to `numcpu`, then
// `numcpu` itself if it is not a power of two (e.g. 1, 2, ..., 128, 144 for
// 144 CPUs).
inline std::vector<int> thread_counts(int numcpu) {
  std::vector<int> counts;
  for (int t = 1; t <= numcpu; t *= 2) counts.push_back(t);
  if (!std::has_single_bit(static_cast<unsigned>(numcpu))) counts.push_back(numcpu);
  return counts;
} // thread_counts()

// The number of configured CPUs, or 0 after printing the reason to stderr if
// it is unavailable. sysconf() returns -1 both on error (errno set) and for an
// indeterminate value (errno untouched); errno is cleared first to tell the two
// apart.
inline int configured_cpu_count() {
  errno = 0;
  const long numcpu = sysconf(_SC_NPROCESSORS_CONF);
  if (numcpu < 1) {
    if (errno != 0) {
      std::fprintf(stderr, "sysconf(_SC_NPROCESSORS_CONF) failed: %s\n", std::strerror(errno));
    } else {
      std::fprintf(stderr, "sysconf(_SC_NPROCESSORS_CONF): the number of CPUs is indeterminate\n");
    }
    return 0;
  } // if the CPU count is unavailable
  return static_cast<int>(numcpu);
} // configured_cpu_count()

#endif // INCLUDED_SPINLOCK_SCOPE_COMMON_H
