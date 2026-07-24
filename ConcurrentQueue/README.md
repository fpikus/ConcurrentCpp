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
to mean "empty slot" — and an optional `Value` payload. `push()` returns
`false` if the queue is truly full; `pop()` returns the reserved key value if
it is empty. The queue runs in caller-provided memory and its capacity is
fixed at construction: the largest power of 2 that fits in the buffer.
There is an `empty()`, and you
must not use it for control flow; whatever it returns is stale before the next
line of your code runs.

## Design principles

Stated here, argued in the book:

- A fixed-size array used as a ring buffer; indexing is a bitmask, no bounds
  checks anywhere.
- Two separate contention domains: producers contend on the tail under one
  spinlock, consumers on the head under another, on separate cache lines.
  Producers and consumers meet at the same slot only when the queue runs
  nearly empty or nearly full — and then at most one of each.
- The producer–consumer handoff is lock-free, through atomics located in the
  slot itself, not in the queue object.
- The key doubles as the slot-state flag; a `busy` flag guards payload
  construction, giving each slot three states: free, occupied, transitional.
- A lock hand-off protocol releases the domain spinlock as soon as a slot is
  claimed, so payload construction — the expensive part — runs outside every
  lock.
- The producer/consumer wraparound race — real, rare, and catastrophic — is
  defended by the busy flag and a triple-check sequence; a tiny 4-slot queue
  under ThreadSanitizer reproduces it on demand. A concurrent program that is
  only *highly likely* to be correct will slip into undefined behavior at the
  worst possible moment.

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

## Building and testing

```sh
make                 # all four benchmarks + ASan/TSan unit tests
make run_tests       # run both sanitizer test binaries
make run_benchmarks
```

Requires clang (the Makefile uses `clang++-22`, C++23), Google Benchmark and
GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your installations if they
are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.

- `concurrent_queue.h` — the queue
- `concurrent_queue_test.C` — unit tests (built with ASan and TSan)
- `concurrent_queue_gmbm.C` / `concurrent_queue_mbm.C` — throughput
  benchmarks (Google Benchmark and a hand-rolled twin producing the same
  measurement)
- `concurrent_queue_lmbm.C` — push-to-pop handoff latency under MPMC
  contention, using the hardware timestamp counter (x86 and ARM)
- `concurrent_queue_ppmbm.C` — 1-producer/1-consumer ping-pong round-trip
  latency

## The book

This directory accompanies Chapter 7 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The chapter walks through the
full implementation — including the wraparound race and how to actually
reproduce it, why measuring queue latency by inverting throughput is
meaningless, and the non-sequentially-consistent "queue pack" that trades
strict ordering for another order of magnitude of scaling.
