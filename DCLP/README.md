# Atomic maximum, three ways

Raising a shared value to the maximum of itself and each thread's offer is the
smallest problem that still forces every real decision in concurrent code:
whether to lock at all, how much of the memory model you actually need, and
whether the compiler lays your fast path out the way you meant. This directory
holds three answers to that one question and the benchmarks that decide between
them on your hardware.

- **Lock-free CAS** (`atomic_max.h`): load the current maximum, and while your
  candidate exceeds it, install it with a compare-exchange. The facility this
  directory ships.
- **Double-checked locking (DCLP)**: read the maximum unlocked; only when your
  candidate looks like a new record do you take a lock, re-read under it, and
  store. Most offers never touch the lock — that is the whole point.
- **The dumb spinlock**: take the lock on every offer and compare under it —
  the version that exists to lose, and to show by how much.

## The function

`atomic_max(target, val, success, failure)` (`atomic_max.h`) raises `*target`
to `max(*target, val)` and returns whether it changed it. The memory ordering
is the caller's, exactly as `compare_exchange` takes it — two orders, one per
outcome:

- **`failure`** governs every read that does not write: the initial load and
  every losing exchange. It is a load order. A plain maximum reduction passes
  `relaxed`; a caller publishing data alongside the maximum passes `acquire`,
  and then a thread that observes the new maximum also observes everything that
  happened-before it.
- **`success`** governs the winning exchange alone: `release`/`acq_rel` to
  publish, `relaxed` for a bare number.

The defaults (`acq_rel`, `acquire`) make an unqualified call fully
synchronized, and only the thread that actually advances the maximum pays for
the barrier.

One line earns its keep:

```cpp
while (ATOMIC_MAX_UNLIKELY(val > cur)) { ... }   // update is the cold path
```

(`ATOMIC_MAX_UNLIKELY` is `__builtin_expect(c, 0)` under GCC and Clang,
clang-cl included, and the bare condition elsewhere: correct everywhere, fast
where the builtin exists.)

The hint marks the update cold, so the compiler sinks the compare-exchange out
of line and the common no-update fast path becomes a *single* taken branch
instead of a forward "skip the CAS" plus the loop back-edge. On a core that
retires one taken branch per cycle this roughly doubles the read-only fast
path; on a core that already lays it out that way it costs nothing, and it
costs a few percent only under a monotone-increasing feed that a warmed-up
maximum never sees. The standard `[[unlikely]]` attribute does *not* achieve
this — only the builtin on the loop condition reaches the compiler's loop
layout. Why that is — and why the memory barrier you first reach for is the
wrong suspect — is in the book.

## What the benchmark shows

`atomic_max_bm.C` runs all three mechanisms against two workloads that bracket
how often the maximum actually moves. Alongside the shipped `atomic_max()`, the
CAS loop (kept in the benchmark, independent of the header) and DCLP each run in
three layouts — update hinted unlikely, not hinted, and hinted likely — to
measure what the branch layout alone is worth:

- **`never`** — each thread offers one fixed random value, so after a brief
  warm-up the maximum never advances and the read-only fast path is
  everything. CAS and DCLP scale with the cores; the dumb lock, which cannot
  skip the lock, sits flat.
- **`grow`** — each thread offers a strictly increasing sequence, so the
  maximum climbs on every round. Even here most offers are stale by the time
  they reach the shared word, so the read pretest still pays; the dumb lock
  pays a full lock/unlock on all of them and loses by a wide margin, while CAS
  and DCLP separate only on the offers that genuinely contend.

Where DCLP beats the lock-free CAS, and where they converge, is
hardware-specific — the same code reads differently on Intel, AMD, and Arm —
and the benchmark scales its thread range to whatever it runs on. The numbers
are one `make run_benchmarks` away from being yours.

The hinted/unhinted pairs say to predict "no update", and the reason is not
specific to this benchmark. Single-threaded, nobody would use an atomic here.
With many threads, the maximum either changes rarely or is contended. If it
changes rarely, the no-update path is nearly every offer, and two jumps saved on
it are most of its cost. If it is contended, everybody loses — a failed exchange
or a contended lock costs far more than two jumps — and the hinted layout is
still the best available. On the fleet the hint doubles CAS with no updates on
Grace and Apple M3, gives 1.3–1.7× on Zen 5 and nothing on Intel (whose compiler
output already has that layout), which brings CAS level with DCLP. It does not
rescue CAS when the maximum moves: there DCLP wins at every thread count above
one, by 1.5× on the best case for CAS and by an order of magnitude or more on
Intel and M3. (The one exception is a 256-thread Intel cell where CAS throughput
swings fourfold from run to run.)

DCLP's plain `if` may or may not need the hint, and that depends on the
compiler rather than the hardware: Clang lays it out well without one, while
GCC's layout is fast or slow depending on the order of the branches — this code
happens to get the fast one, the reversed if/else would not. The hint takes the
luck out of it. Measured with GCC 16 on a 128-thread Zen 5 server, hinted and
plain DCLP are indistinguishable at every thread count, updates or not, while
the same hint is worth 1.2–1.6× to the CAS loop when nothing changes: GCC's
default layout for the `while` loop is the slow one. Hinting the DCLP update as
*likely* forces the other layout and shows what the luck is worth: 12–15% with
GCC on that server, half the throughput with Clang on a Zen 4 laptop.

Where the lock goes matters too. Putting DCLP's lock on the same cache line as
the maximum — the arrangement that wins for an ordinary lock and its data at low
contention — never pays off here: on the 128-thread Zen 5 server it ties at low
contention and loses up to half the throughput under contention
(`BM_dclp_sameline_work` against `BM_dclp_work`, with a work dial between
offers). DCLP's offers mostly only read the maximum, and every lock acquisition
by a writer drags the readers' copy of that line away even when it ends up not
writing. When the contention is not known in advance, keep them apart.

Timing says *that* DCLP wins on `grow`; `atomic_max_count.C` says *why*. It
runs the same feed through instrumented copies of the CAS loop and the DCLP
body and tallies what each offer did: dodged on the read, updated, or wasted —
a failed compare-exchange for CAS, a lock taken for nothing for DCLP. On a
16-thread laptop (Ryzen 7940HS), CAS wastes about three failed exchanges for every
successful one, each of which drags the cache line; DCLP wastes about one lock
acquisition in ten thousand offers, and more than nine offers in ten never write
at all. The read pretest does more than skip the lock: it filters out the stale
offers before any read-modify-write happens, which the CAS loop only does
after its first exchange has already failed.

## Building and testing

```sh
make benchmarks     # the benchmark and the outcome counter — needs only Google Benchmark
make                # benchmarks + ASan/TSan unit tests (needs GoogleTest)
make run_tests      # run both sanitizer test binaries
make run_benchmarks # the benchmark over the full thread range, then the counter
```

The unit tests (`atomic_max_test.C`) check the contract single-threaded, then
hammer it from many threads; run under TSan, a plain payload published through
the maximum turns any missing release/acquire into a reported data race.

Binaries land in `build/<hostname>/`, so several machines can build and run
concurrently in one shared tree. The machine-specific configuration —
compiler, `-march` target, C++ standard, and the Google Benchmark/GoogleTest
install paths — lives in `../config.mk`, shared by every benchmark directory
and written once per machine; the Makefile is machine-independent. The
`SpinLock` behind the DCLP and dumb-lock variants comes from
`../Spinlock/spinlock.h`, so this directory uses the same lock as the rest of
the repository rather than a private copy.

- `atomic_max.h` — the lock-free maximum (the shipped facility)
- `atomic_max_test.C` — unit tests (built with ASan and TSan)
- `atomic_max_bm.C` — the three mechanisms (CAS and DCLP each with and without the branch hint) across two workloads and the thread range
- `atomic_max_count.C` — per-offer outcomes (dodged / updated / wasted) of CAS and DCLP on the `grow` feed; `atomic_max_count [nthreads [iters]]`

## The book

This directory accompanies Chapter 6 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The chapter holds what this README
only asserts: the double-checked locking pattern done correctly, with the
acquire/release handoff traced through the memory model rather than bolted on;
the minimum ordering each outcome actually requires, and what a needlessly
strong barrier costs; the disassembly-level account of why a compare-exchange
retry loop and a plain `if` compile to different fast paths, and why the effect
that looks like a hardware branch predictor is really the compiler's basic-block
layout; and the honest reckoning of when lock-free is worth it — where a lock
that most threads dodge outruns an atomic that everyone must contend for.
