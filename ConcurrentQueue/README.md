# RingAtomicMapQueueMPMC

A fixed-capacity, multi-producer multi-consumer (MPMC) ring-buffer queue with
a transactional interface, two isolated spinlock domains, and a lock-free
handoff between producers and consumers. (In the book it appears under the
shorter name `RingQueue`; the class in `concurrent_queue.h` is the same
design.)

The starting point is an interface observation: `std::queue`'s
`empty()`/`front()`/`pop()` triple is as hostile to concurrency as an API can
possibly be. Any concurrent queue needs a single, transactional `pop()` that
either removes and returns the front element or safely reports that the queue
was empty. This queue provides exactly that — and then arranges the internals
so that the transaction is fast.

## The contract

The queue stores a `Key` — a cheap built-in type with native lock-free
atomics (a pointer or an integer), with the default-constructed value reserved
to mean "empty" — and an optional `Value` payload. `push()` returns `false` if
the queue is truly full; `pop()` returns the reserved key value if it is
empty. The queue runs in caller-provided memory and its capacity is fixed at
construction: the largest power of 2 that fits in the buffer. There is an
`empty()`, and you must not use it for control flow; whatever it returns is
stale before the next line of your code runs.

Nothing in the queue throws, and the compiler holds you to it: `push()` and
`pop()` are `noexcept`; `push()` accepts only arguments that `Value` can be
built from without throwing, `pop()` only a destination it can be assigned
into without throwing, and the class refuses outright a `Value` whose
destructor may throw — every popped value is destroyed, so such a queue could
never be popped, and it has nothing else to offer. Exception friendliness is
what split `std::queue`'s `pop()` in two in the first place. It costs nothing
in practice: a high-throughput queue lives on a fast path where you are not
allowed to allocate anyway, so a value that owns memory is built on your side,
before the push, and moved in.

## Design principles

Stated here, argued in the book:

- A fixed-size array used as a ring buffer; indexing is a bitmask, no bounds
  checks anywhere.
- Two separate contention domains: producers contend on the tail under one
  spinlock, consumers on the head under another, on separate cache lines.
  Producers and consumers meet at the same slot only when the queue runs
  nearly empty or nearly full — and then at most one of each.
- The producer–consumer handoff is lock-free, through atomics located in the
  slot itself, not in the queue object. In the key-only queue the key is the
  slot's state; in the key-value queue each slot carries a sequence number
  that says not just "full" or "empty" but "full or empty *for the index you
  are at*" — for index `i` it goes `i` (free), `i+1` (committed),
  `i+capacity` (free for the next lap).
- A lock hand-off protocol releases the domain spinlock as soon as a slot is
  claimed, so payload construction — the expensive part — runs outside every
  lock.
- Every successful push and pop ends its critical section with a store to its
  slot, and that store is there for the lock, not for the slot. Under
  contention the spinlock's throughput comes from batching — the releasing
  thread takes the lock again while the waiters are asleep — and that works
  only if the holder leaves the critical section with nothing pending. Moving
  the store after the unlock is perfectly correct and up to 8x slower; it has
  been tried, measured, and reverted twice. The header explains the mechanism;
  the numbers are under Performance.
- The producer/consumer wraparound race — real, rare, and catastrophic — is
  where the key-value queue changed. The earlier design (the one in the book)
  used the key as the state word plus a `busy` flag, validated by reading key,
  flag, key again. It is wrong: key values recur (the empty key every lap, and
  any user key may repeat), so a thread a full lap ahead can pass the check on
  a slot that is never stable, and two threads end up owning one slot. The
  sequence number fixes it, because the values a thread compares against occur
  exactly once in the slot's life. The unit tests reproduce both failures of
  the old protocol deterministically, by parking threads at chosen points
  (random stress, with or without ThreadSanitizer, essentially never hits
  them). A concurrent program that is only *highly likely* to be correct will
  slip into undefined behavior at the worst possible moment — and so will one
  that has been carefully reasoned about, if the reasoning was never made to
  fail on demand.

## Performance

In its proper domain — a high-throughput MPMC queue moving pointers or
integers between threads, the operating mode of most task queues and thread
pool schedulers — this design outpaces five leading open-source MPMC queues
on a large x86 Granite Rapids server, scaling almost perfectly linearly up to
32 threads before NUMA takes over. Along the way, the benchmarks expose a
hardware reality: on the newest Intel CPUs, aggressive prefetchers double the
effective false-sharing range, and 128-byte slot alignment measurably beats
the traditional 64.

Latency tells the other half of the story, and it is a genuine trade-off. At
128 threads on the same machine, against a highly optimized lock-free CAS
queue built for latency-sensitive work:

|                | RingAtomicMapQueueMPMC | Lock-free comparison queue |
|----------------|-----------------------:|---------------------------:|
| Median         | 320 ns                 | 82 µs                      |
| 99.9%          | 410 µs                 | 546 µs                     |
| Worst case     | ~10 ms outliers        | ~1 ms cutoff               |

The strictly lock-free design buys a hard cap on the worst case and pays for
it with a 250x worse median. Whether that trade is worth making depends
entirely on your application; the numbers above are what you are trading.

One line in the file is worth more than any other: the store to the slot that
every successful push and pop makes before releasing its lock. The key-value
protocol was once rewritten without it. On a 144-core Arm server (Grace: two
Neoverse-V2 dies) that version ran at 0.19-0.42x of the old one at 4-32
threads, and at about 0.4x on a two-socket EPYC 9555 at 128 threads (both with
64- and 128-byte slots). Instrumenting the lock showed why: with the store,
0.00% of lock acquisitions changed hands on that Grace; without it, 30-42%. It
had happened before, in the key-only queue, where unlocking before the key
store lost up to 8x on Zen 4 and on a 72-thread Grace, and won about 2x only
on Cascade Lake with packed slots. With the store back, the new key-value
queue against the old one, at 2^16, 2^22 and 2^26 slots: faster in most cells
on Arm — up to 1.85x on that Grace and 2.6x on an Apple M3 Ultra (24
performance cores, Linux in a VM), with 15 of 156 cells slightly slower
(0.84-0.99x) — and broadly even on x86 (EPYC 9555, Xeon 6767P), with losses of
0.6-0.9x in the packed layouts at some thread counts. Push-to-pop latency is
unchanged. These are throughput ratios from `concurrent_queue_gmbm` with
`<uint64_t, uint64_t>` elements, and every one of them belongs to this
spinlock's back-off: retune the back-off and measure again.

## Building and testing

```sh
make                 # all four benchmarks + ASan/TSan unit tests
make run_tests       # run both sanitizer test binaries
make run_benchmarks
```

The compiler, C++ standard and library paths come from `../config.mk`, shared
by every directory in this repository and written once per machine; binaries
go to `build/<hostname>/`. Requires Google Benchmark and GoogleTest.
`make benchmarks` and `make tests` build either group alone.

A warning for multithreaded Google Benchmark runs: `--benchmark_min_time=1s`
is compared against real time summed over all threads, so at 128 threads it
is a measurement window of about 8 ms per thread. An iteration count is per
thread, so run one thread count at a time with `--benchmark_min_time=<N>x`
and an `N` chosen for that thread count (the header of
`concurrent_queue_gmbm.C` has the details). The queue's capacity in the
throughput benchmark is 65536 slots, or the value of the `CQ_CAP` environment
variable rounded down to a power of 2.

- `concurrent_queue.h` — the queue
- `concurrent_queue_test.C` — unit tests (built with ASan and TSan)
- `concurrent_queue_gmbm.C` / `concurrent_queue_mbm.C` — throughput
  benchmarks (Google Benchmark and a hand-rolled twin producing the same
  measurement)
- `concurrent_queue_lmbm.C` — push-to-pop handoff latency under MPMC
  contention, using the hardware timestamp counter (x86 and ARM). Sweeps
  thread counts and all four slot alignments by default; `--threads=N` and
  `--align=A` pin one of each for long single-point runs
- `concurrent_queue_ppmbm.C` — 1-producer/1-consumer ping-pong round-trip
  latency

## The book

This directory accompanies Chapter 7 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The chapter walks through the
full implementation — including the wraparound race and how to actually
reproduce it, why measuring queue latency by inverting throughput is
meaningless, and the non-sequentially-consistent "queue pack" that trades
strict ordering for another order of magnitude of scaling. The book describes
the earlier key-and-flag slot protocol of the key-value queue; the code here
has since replaced it with a per-slot sequence number, for the reason given
above.
