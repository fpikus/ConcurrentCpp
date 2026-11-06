# RingAtomicMapQueueMPMC

A fixed-capacity, multi-producer multi-consumer (MPMC) ring-buffer queue with
a transactional interface, two isolated spinlock domains, and a lock-free
handoff between producers and consumers. (In the book it appears under the
shorter name `RingQueue`; the class in `concurrent_queue.h` keeps its
structure but has a different key-value protocol: see the wraparound race
below.)

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
pool schedulers — this design outpaces five open-source MPMC queues on a
two-socket Granite Rapids Xeon (256 hardware threads, measured November
2026) at every thread count from 4 to 256, by 5x to 240x. At 2 threads the
picture is mixed: with the default packed slots this queue beats four of the
five but loses 2.5x to the best single-CAS ring (Rigtorp's); with 128-byte
slots it beats all five there too, by 2.5x to 42x.

What it does NOT do is scale with the thread count, and neither does anything
else in the comparison. Every push and every pop here goes through one of
two spinlocks whose holder batches work while the losers park, so the
aggregate rate is set by what the two lock holders can do, not by the number
of threads: with packed slots it is flat at about 42 M items/s from 2 to 64
threads, 45 at 128 and 56 at 256; with 128-byte slots it starts at 264 M/s
and declines gently to 163 M/s at 128 threads. The rivals fall roughly as
1/threads — all but moodycamel, which stays within 4-8 M/s up to 32 threads
— from 10-105 M items/s at 2 threads to 3-4 M/s at 8 and under 1 M/s at 128,
where each role's one contended counter serializes every thread.
Pinning the benchmark to one socket changes nothing up to 64 threads: the
curve is the lock's, not the interconnect's.

The two layouts above are the real alignment story. Slot padding matters
enormously at low thread counts — 42 M/s packed against about 250 M/s at 64-
or 128-byte slots with 2 to 16 threads — and the reason is a cache line
travelling between two cores: put the producer and the consumer on the two
SMT threads of one core and the packed queue runs at 173 M/s, as fast as the
padded one there (165). Between 64- and 128-byte slots the machine does not
distinguish in these rows: the ratio stays between 0.92 and 1.11 at every
thread count but 128 as the benchmark's own code placement is varied, and
the 1.27x at 128 threads, the one cell that looked like a win for 128 bytes,
drops to 1.05-1.11x when nothing but the code placement changes.

The numbers (M items/s: successful pushes plus pops per second of wall-clock
time, the median of six rounds of five repetitions, key-only elements, 65536
slots (moodycamel: its initial capacity), every thread pushing or popping and
doing nothing else). A rival's cell at 128 and 256 threads is an upper bound
over the rounds, and moodycamel's from 8 threads up a range over the rounds,
because one role finishes its share long before the other there and the
wall-clock rate mixes two regimes; the 2-thread cells depend on where the
scheduler puts the two threads (here, two cores of one socket).

| threads | this queue (packed) | this queue (128-byte slots) | lockfree (DN) | Rigtorp (ER) | DCAS (ES) | atomic_queue (ME) | moodycamel (MC) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 42 | 264 | 17 | 105 | 26 | 9.7 | 6.3 |
| 8 | 42 | 244 | 2.8 | 3.5 | 4.3 | 3.0 | 5-8 |
| 32 | 42 | 220 | 1.0 | 1.2 | 1.9 | 1.1 | 4-8 |
| 128 | 45 | 163 | < 0.5 | < 0.8 | < 0.9 | < 0.7 | 2-3 |
| 256 | 56 | 75 | < 0.8 | < 1.1 | < 0.3 | < 0.3 | 1-2 |

Latency tells the other half of the story, and it is a genuine trade-off.
The latency benchmark records, at 128 threads pinned one per core across
both sockets, a thread's own push-to-pop round trip through the contended
queue at near-empty depth (most samples are the thread popping what it
pushed a moment earlier, with everyone else contending), in 60-second runs
repeated three times, against a highly optimized DCAS queue built for
latency-sensitive work:

|                  | RingAtomicMapQueueMPMC | Lock-free DCAS queue |
|------------------|-----------------------:|---------------------:|
| Median           | 0.55 µs                | 84 µs                |
| 99th percentile  | 0.30 ms                | 0.36 ms              |
| 99.9th           | 1.57 ms                | 0.59 ms              |
| Worst, 60 s run  | 12-19 ms               | 2-7 ms               |
| Worst, 900 s run | 22 ms                  | 11 ms                |

The lock-free design pays 150x at the median — and 3.6x at the 90th
percentile, 2x at the 95th, 1.2x at the 99th — and gets back a 99.9th
percentile 2.7x tighter and a worst case two to six times better over a
minute and about half as bad over a quarter-hour. What it does not get is a
hard cap: the DCAS queue's worst case grew to 19 ms in one five-minute run.
Whether the trade is worth making depends entirely on your application; the
numbers above are what you are trading.

Where this queue's tail comes from is no mystery: at 128 threads it is the
lock's back-off ladder. Every sampled operation above the 90th percentile
(57 µs) had waited through at least one of the lock's ~50 µs parks (58 µs
each, measured), the 99th percentile is about five of them, and every
operation above the 99.9th percentile (1.56 ms) had gone through the
ladder's full cycle of eight parks and its 1 ms sleep. At 16 threads the
lock parks in one operation in a thousand and the tail begins at the 99.9th
percentile. (Measured on every sample at 128 threads, the 60% a thread pops
back itself and the 40% popped by another thread alike: above the 90th
percentile every sample of either kind had waited through at least one park,
and above the 99.9th 99.3-99.7% through the 1 ms sleep.)

Uncontended, the two are the same queue: a 1-producer/1-consumer round trip
through two queues costs 0.4-0.5 µs on two cores of one socket and 1.0-1.2 µs
across sockets for this design, the DCAS queue and the single-CAS ring
alike.

One line in the file is worth more than any other: the store to the slot that
every successful push and pop makes before releasing its lock. The key-value
protocol was once rewritten without it. On a 144-core Arm server (Grace: two
Neoverse-V2 dies) that version ran at 0.19-0.42x of the old one at 4-32
threads, and at 0.39x (64-byte slots) and 0.51x (128-byte slots) on a
two-socket EPYC 9555 at 128 threads. Instrumenting the lock showed why: with
the store, 0.00% of lock acquisitions changed hands on that Grace; without it,
28-48% at 4-8 threads. It had happened before, in the key-only queue, where
unlocking before the key store lost up to 8x on Zen 4 and on a 72-thread
Grace, and won about 2x only on Cascade Lake with packed slots. With the store
back, the new key-value queue against the old one, at 2^16, 2^22 and 2^26
slots: faster in most cells on Arm — up to 1.85x on that Grace and 2.6x on an
Apple M3 Ultra (24 performance cores, Linux in a VM), with 15 of 156 cells
slightly slower (0.84-0.99x) — and mixed on x86 (EPYC 9555, Xeon 6767P):
gains of up to 2.5x, mostly with padded slots, and losses down to 0.6x,
mostly with packed slots and at the highest thread counts. At 2 threads the
result depends on where the two threads land, and the x86 ratios scatter from
0.24x to 2.6x.
Push-to-pop latency is unchanged. These are throughput ratios from
`concurrent_queue_gmbm` with `<uint64_t, uint64_t>` elements, and every one of
them belongs to this spinlock's back-off: retune the back-off and measure
again.

## Building and testing

```sh
make                 # all four benchmarks + ASan/TSan unit tests
make run_tests       # run both sanitizer test binaries
make run_benchmarks
```

The compiler, C++ standard and library paths come from `../config.mk`, shared
by every directory in this repository and written once per machine; binaries
go to `build/<hostname>/`. Requires Google Benchmark and GoogleTest. The
Google Benchmark binary (`concurrent_queue_gmbm`) also includes
`../Spinlock/gb_wall_clock.h` (through `-I../Spinlock`), the wall-clock
accounting behind its `wall_items_per_second` and `finish_spread` counters.
`make benchmarks` and `make tests` build either group alone.

Two warnings for multithreaded Google Benchmark runs. Google Benchmark
compares `min_time` against real time summed over all threads, so 1 s at 128
threads would be a measurement window of about 8 ms per thread;
`concurrent_queue_gmbm` therefore registers every thread count with its own
window (at least 1 s of wall clock, after a warm-up of at least 0.5 s) and
ignores the seconds form of `--benchmark_min_time`. And every thread runs the same number of
iterations, so threads, or a whole role, can finish early and leave the end of
a run less contended than its start; `items_per_second` hides that, and the
benchmark adds `wall_items_per_second`, `finish_spread`, `push_end` and
`pop_end` to show it (the header of `concurrent_queue_gmbm.C` has the
details). `concurrent_queue_mbm` runs the same operation with every thread
contending until a common stop; its `items_per_s` corresponds to
`wall_items_per_second` of the `*_balanced` rows. In both, the queue's capacity is 65536 slots,
or the value of the `CQ_CAP` environment variable rounded down to a power of
2, and the queue starts empty, or filled to the fraction of its capacity that
`CQ_FILL` gives (in `concurrent_queue_gmbm`, every run starts that full; in
`concurrent_queue_mbm`, the warm-up does, so use `--warmup=0` to measure from
that start).

- `concurrent_queue.h` — the queue
- `concurrent_queue_test.C` — unit tests (built with ASan and TSan)
- `concurrent_queue_gmbm.C` / `concurrent_queue_mbm.C` — throughput
  benchmarks (Google Benchmark, and a hand-rolled twin running the same
  operation with every thread contending for the whole window; CSV on stdout).
  The twin runs only the balanced rows (`*_balanced`, where a thread that
  finds the queue full pops and one that finds it empty pushes): run for the
  same time rather than the same number of operations, the role that is
  faster otherwise pins the queue full or empty
- `concurrent_queue_lmbm.C` — push-to-pop round-trip latency under MPMC
  contention (mostly a thread popping its own batch back; the cross-core
  handoff is `ppmbm`), using the hardware timestamp counter (x86 and ARM). Sweeps
  thread counts and all four slot alignments by default; `--threads=N` and
  `--align=A` pin one of each for long single-point runs
- `concurrent_queue_ppmbm.C` — 1-producer/1-consumer ping-pong round-trip
  latency

## The book

Chapter 7 of *The Art of Writing Efficient Programs, Second Edition* by
Fedor G. Pikus covers an earlier version of this code; the code here has
moved on since.
