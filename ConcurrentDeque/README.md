# ConcurrentAppendDeque

An array-like, random-access container that can be grown by some threads while
other threads access its elements — with strictly wait-free element access.
This is the data structure you reach for when multiple threads share a data
repository, each thread works on its own range of elements, and the container
must grow *while they work*.

Standard containers serve this pattern very poorly: the weak thread-safety
guarantee of the STL demands that every read be locked if even one thread may
be resizing, and a `std::vector`-style reallocation that moves elements is a
use-after-free factory. The structural way out is the one `std::deque`
discovered long ago — elements, once constructed, never move — taken here to
its concurrent conclusion.

## The contract

`ConcurrentAppendDeque<T, BlockSize>` (see `concurrent_deque.h`) is an
append-only deque: it grows, it never shrinks. A thread that obtains the
authoritative size — by calling `size()`, or by growing the container itself
with `resize()` — may freely use every element with an index below that size,
concurrently with any number of other readers and resizers, with no locking of
any kind. Pointers and references to existing elements are never invalidated.
The only mathematically guaranteed way to know which elements exist is to ask
the container itself; that sentence is most of the user's manual.

## Design principles

The full reasoning behind each of these decisions is worked out in Chapter 7
of *The Art of Writing Efficient Programs, Second Edition*; here we only state
them:

- All memory is allocated in fixed-size blocks; an element, once constructed,
  never moves.
- The blocks are reached through a directory (an array of block pointers),
  which is reached through an atomic root pointer.
- Growth is serialized by a spinlock; element access is wait-free and never
  takes the lock.
- Synchronization between the two paths is a publishing protocol on two atomic
  variables — the root pointer and the size — arranged so that every
  intermediate state of a resize is safe to observe.
- Outdated directories are never deleted, only retired; the hardest question
  in concurrent data structure design — *when is it safe to free this?* — is
  answered here with "never" (until the destructor).
- `resize()` requests that would shrink the container safely do nothing:
  hence *Append*Deque.
- `BlockSize` must be a power of 2, so indexing is bit math, not division.

## Performance

In its proper domain — concurrent element access, with growth happening but
rare relative to the accesses — the deque is not marginally faster than the
locked alternative; it is a different kind of fast. The baseline is a
`std::deque` guarded by a spinlock, measured on a large x86 Granite Rapids
server:

| Threads | Spinlocked `std::deque` | `ConcurrentAppendDeque` |
|--------:|------------------------:|------------------------:|
| 1       | 136 M elements/s        | 1.8 G elements/s        |
| 2       | 40 M/s                  | 3.6 G/s                 |
| 32      | 83 M/s                  | 27.4 G/s                |
| 128     | 94 M/s                  | 53.6 G/s                |

The single-threaded result is over 13x faster, before contention even enters
the picture, and the scaling is nearly linear until NUMA effects take over.
In the dynamic pattern — every thread repeatedly reserves a new range with
`resize()` and then works on it — the locked baseline degrades into serialized
execution at ~100–130 M elements/s regardless of thread count, while the
concurrent deque reaches 22.7 G elements/s at 128 threads. The same shape
repeats on ARM (NVIDIA Grace) and on a desktop Ryzen; only the absolute
numbers change.

## Building and testing

```sh
make            # benchmark + ASan/TSan unit tests
make run_tests  # run both sanitizer test binaries
```

Requires clang (the Makefile uses `clang++-22`, C++23), Google Benchmark and
GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your installations if they
are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.

- `concurrent_deque.h` — the container (used as the backing store by the
  concurrent hash set in `../ConcurrentHash`)
- `concurrent_deque_test.C` — unit tests (built with ASan and TSan)
- `concurrent_deque_bm.C` — the benchmarks quoted above

## The book

This directory accompanies Chapter 7 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The README states the design
decisions; the book explains why they are correct — including the memory
ordering argument for why every intermediate state of a concurrent resize is
safe, which is the part you actually want.
