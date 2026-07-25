# Spinlock

A test-and-test-and-set (TTAS) spinlock with a two-tier escalating back-off —
the lock used throughout this repository. A spinlock exists because the
operating system's mutex is the wrong instrument for guarding an operation
that takes nanoseconds: a lock that parks threads in the kernel cannot be
cheap, and a lock that never parks them melts the machine. Between those two
failure modes lies a narrow band of designs that are actually fast, and every
decision that puts this lock inside that band was made by measurement, not
folklore. The benchmarks that made those decisions are in this directory,
ready to be re-run against your hardware.

## The contract

`SpinLock` (`spinlock.h`) is a standard Lockable — `lock()`, `unlock()`,
`try_lock()`, and `std::lock_guard`/`std::unique_lock` work as expected —
plus a `locked()` peek that is advisory, relaxed, and racy: a hint for asserts
and diagnostics, never a foundation for mutual exclusion. The release/acquire
handoff is the classic one: everything the previous holder wrote in its
critical section happens-before everything the next holder does.

One clause deserves attention: `try_lock()` is a bounded spin, not a
single-shot attempt. Even a free lock's cache line is usually owned by some
other core, and a one-shot try would report "busy" for what is merely
coherence latency. This `try_lock()` rides out the ownership transfer and a
briefly-held lock, and gives up only where `lock()` would escalate to a real
sleep — so `false` means the lock was genuinely held, not merely far away.

## Design principles

Stated here, earned in the book — and re-verified by the benchmarks in this
directory:

- Test first, then test-and-set. Waiters *read* the lock word and pay for the
  atomic exchange only when it reads free. A read leaves the cache line
  shared among all waiters; a naive exchange-in-a-loop demands exclusive
  ownership of the line on every iteration just to replace a 1 with a 1.
- When the lock stays busy, sleep — do not pause. The pause instruction
  quiets the pipeline but leaves the core answering coherency probes;
  `nanosleep` removes the thread from the scheduler entirely, and a waiter
  that is not running is not dragging the line away the instant the owner
  needs it back to unlock. At high contention, reducing the number of cores
  actively fighting over the cache line is the single optimization that
  matters above everything else.
- Between sleeps, a burst of eight unrolled acquisition attempts — enough to
  catch a just-released lock within a few instructions of the release, cheap
  enough to precede every syscall. Eight to sixteen is the measured optimum.
- Two tiers of sleep: eight rounds of ~1 ns sleeps (which the kernel rounds
  up to the next scheduling opportunity — a cheap yield), then 1 ms of real
  sleep, and the cycle repeats. A briefly contended lock stays hot; a
  long-held lock costs its waiters nothing.

The 1 ms is not folklore either. The second tier was swept at 0.1, 1, and
10 ms (`spinlock_tune_bm.C`) across the full contention range on servers up
to 256 hardware threads and two L3 domains. The verdict: 1 ms is best or
statistically tied at every contention level and every thread count; 10 ms
buys a little mid-range throughput and then collapses five- to ten-fold the
moment threads span an L3 domain; 0.1 ms sends too many probes at the highest
thread counts. Where the collapse begins is machine-dependent. Which tier
wins is not.

## The domain of applicability

This is the lock for critical sections of a few instructions: a counter, a
pointer swap, the innards of a concurrent data structure. Such locks are
naturally "captive" — the guarded code calls nothing that could take another
lock, so deadlock is impossible by construction (the residual hazard, a
holder that never returns, is not a deadlock but a hostage situation). Three
honest boundaries:

- A critical section longer than a few dozen nanoseconds belongs under
  `std::mutex`: a thread facing a long wait should be asleep, and the mutex's
  per-operation inefficiency stops mattering.
- At low contention the balance inverts. The spinlock still wins the
  throughput of the lock operations themselves, but atomic operations have
  lower overhead, and it is the surrounding program whose throughput you care
  about. Where the crossover sits is hardware-specific: some server CPUs make
  locking expensive even when it is 0.1% of the work; some client cores make
  the atomics the expensive ones.
- The back-off is deliberately unfair. A releasing thread can re-acquire
  while the beaten waiters sleep; throughput is bought with lock hogging. If
  fairness is a requirement, this is not your lock.

## Performance

Measured on servers from 64 to 256 hardware threads, and stated plainly:

- Under maximum contention — the critical section is a single shared
  update — throughput holds within ~10% of the single-thread rate from 1 to
  64 threads (~140 M guarded updates/s on our test servers). Flat is the
  ceiling for an inherently serial operation, and this lock sits at it.
- That is 5–15× the standard mutex, and beyond a handful of threads it also
  outruns the wait-free atomic increment, which has no way to back off and no
  way to stop dragging the cache line across the machine. At 256 threads the
  lock finally yields ground — and still leads the mutex by 3× and the
  atomic by 6×.
- Every alternative back-off was benchmarked and lost. No back-off at all
  collapses under contention by nearly two orders of magnitude, pause with
  it; `sched_yield` survives only to modest thread counts; single-tier sleeps
  either park too eagerly or not eagerly enough. The difference between a
  spinlock with manners and one without is the difference between flat and
  falling.

`spinlock_bm.C` measures the lock against the atomic and mutex baselines
across a contention dial from 100% of time under the lock down to 0.1%;
`spinlock_tune_bm.C` re-runs the tier sweep. Both scale their thread range to
the machine they run on — the numbers above are one `make run_benchmarks`
away from being your numbers.

## Building and testing

```sh
make                # benchmarks + ASan/TSan unit tests
make run_tests      # run both sanitizer test binaries
make run_benchmarks # the contention sweep, then the tier sweep
```

Requires clang (the Makefile uses `clang++-22`, C++23), Google Benchmark and
GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your installations if they
are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.

- `spinlock.h` — the lock
- `spinlock_test.C` — unit tests (built with ASan and TSan)
- `spinlock_bm.C` — the lock against its alternatives, across the contention range
- `spinlock_tune_bm.C` — the back-off tier sweep
- `spinlock_bm_common.h` — the shared benchmark harness

## The book

This directory accompanies Chapter 6 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The chapter holds what this
README only asserts: the cache-coherency forensics of why sleeping beats
pausing, traced through the coherence states line by line; the
performance-counter investigation that catches the spinlock red-handed
poisoning the execution unit — store-buffer stalls by the hundred billion,
and a reorder-buffer count that reads backward until you see why; the quiet
reversal of a decade of conventional wisdom about where lock-free code
actually earns its keep; and the parade every lock drags behind it —
deadlock, livelock, convoying, priority inversion, lock hogging — with an
account of which of them a spinlock escapes by construction.
