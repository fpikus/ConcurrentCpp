# ConcurrentCpp

High-performance concurrent data structures in C++, built and benchmarked in
the open. Each subdirectory is a self-contained project with its own sources,
tests, benchmarks, and a README stating what the structure does, where it
applies, and how fast it is in its proper domain.

## The projects

The first four data structures originate from Chapter 7 of *The Art of
Writing Efficient Programs, Second Edition* by Fedor G. Pikus, where their
design was first worked out. The code here is live and has evolved past the
version printed in the book.

- **[ConcurrentDeque](ConcurrentDeque/)** — `ConcurrentAppendDeque`, an
  array-like container with strictly wait-free element access that can be
  grown concurrently; elements never move, pointers are never invalidated.
- **[ConcurrentQueue](ConcurrentQueue/)** — `RingAtomicMapQueueMPMC`, a
  fixed-capacity MPMC ring-buffer queue with two isolated spinlock domains
  and a lock-free producer–consumer handoff.
- **[ConcurrentHash](ConcurrentHash/)** — `ConcurrentResizableHashSet`, a
  chained hash set with live resizing, lock-free except where it allocates:
  a lookup's walk writes nothing and an insert publishes with one CAS, while
  a new node, whether an insert's or a copy made by a lazy split (which a
  lookup can run too), is appended under a spinlock unless `reclaim()` left
  one on a free list, and the bucket array grows under a resize lock. It is
  built on three refusals: never free, never relink, never unlink (the one
  exception, `reclaim()`, runs when the caller has quiesced the set).
- **[LockFreeList](LockFreeList/)** — a Harris-style lock-free singly-linked
  list that reclaims memory for real, with never-invalidated iterators,
  parameterized over four atomic shared pointer implementations.

Supporting components shared by the projects above:

- **[IntrSharedPtr](IntrSharedPtr/)** — atomic intrusively
  reference-counted smart pointers, a reusable component: one with an
  embedded one-bit lock, and its sibling with a hazard pointer in place of
  the lock, lock-free after a thread's first load, built on Maged Michael's
  hazard pointers; the pointee supplies the count through a base class or
  hand-written hooks checked by a concept.
- **[SharedPtr](SharedPtr/)** — the atomic shared pointer concepts
  LockFreeList is written against, adapters that fit
  `std::atomic<std::shared_ptr>` and Daniel Anderson's genuinely lock-free
  `atomic_shared_ptr` to them, and the harness that tests and benchmarks all
  four pointers side by side. Maged Michael's and Daniel Anderson's code are
  built from their upstream repositories with small patches of ours (see
  [SharedPtr](SharedPtr/), "Third-party code").
- **[Spinlock](Spinlock/)** — the TTAS spinlock with a two-tier back-off
  used throughout, with the benchmarks that tuned it.

New projects added after the book's publication will be exactly that — new,
and documented on their own terms.

- **[LockFreeListRCU](LockFreeListRCU/)** — the LockFreeList algorithm with
  the reference counts taken out: memory is reclaimed by generations, every
  operation runs under a handle, erased nodes are recycled through a free
  list by a caller-driven `reclaim()`, and an iterator stays valid until the
  handle it was obtained under is destroyed, refreshed or move-assigned over.
- **[ConcurrentHashRCU](ConcurrentHashRCU/)** — a copy of
  `ConcurrentResizableHashSet` that reclaims memory by generations, without a
  quiescent point: every operation runs under a handle; dead nodes
  (tombstones, superseded split copies) are bypassed by the concurrent
  writers themselves, best effort, one CAS per run (an erase tries its own
  tombstone at most twice), while a lookup's own walk writes nothing;
  and a caller-driven `advance()` recycles them through the free lists once
  no handle can still see them, and walks the table for the dead nodes the
  writers left behind; `reclaim()` recycles without the walk, and a set
  constructed with a cadence runs it by itself.
- **[Thread](Thread/)** — a thread used like `std::jthread`, launched by its
  constructor and joined by its destructor, that also sets what the standard
  cannot: a stack size and a guard size per thread, and one fixed start mask,
  the asynchronous signals blocked and the faults (and SIGPROF) unblocked,
  whatever the creating thread had. Each thread's stack top is staggered
  within a page, and an exception that escapes the callable unwinds the
  thread's stack before it calls `std::terminate()`.

## Building

Each project builds independently with its own Makefile:

```sh
cd ConcurrentDeque   # or any other project
make                 # benchmarks + unit tests: ASan/TSan, some also optimized
make run_tests
```

Requirements: a recent clang (the Makefiles use `clang++-22`, C++23),
[Google Benchmark](https://github.com/google/benchmark) and
[GoogleTest](https://github.com/google/googletest); set `GBENCH_DIR` and
`GTEST_DIR` if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.
Some projects (ConcurrentDeque, ConcurrentHash, ConcurrentHashRCU,
ConcurrentQueue) share headers via relative symlinks, so clone on a filesystem
that supports them (on Windows, use WSL or enable `core.symlinks`); SharedPtr
and LockFreeList include their sibling projects' headers through `-I` instead,
and so do the benchmarks of ConcurrentHash, ConcurrentHashRCU, ConcurrentQueue,
IntrSharedPtr and LockFreeListRCU, for the wall-clock accounting in
`Spinlock/gb_wall_clock.h`.
IntrSharedPtr, SharedPtr and LockFreeList build on Linux only: the hazard
pointers they link issue `membarrier(2)`. Before their first build, clone the
third-party code they use (SharedPtr's README, "Third-party code"); their
first `make` then makes the patched copies the build uses. `make imports` makes
only those copies: in IntrSharedPtr its `mm_hp/`, in SharedPtr its
`lock_free_shared_ptr/parlay/`, and in LockFreeList both (by asking those two
projects).
Thread builds on Linux with glibc and libstdc++ only (any other C or C++
library stops the build with an `#error`), and needs GoogleTest but not
Google Benchmark: it has no benchmark yet.

## License

MIT — see [LICENSE](LICENSE).
