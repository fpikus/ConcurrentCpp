# ConcurrentCpp

High-performance concurrent data structures in C++, built and benchmarked in
the open. Each subdirectory is a self-contained project with its own sources,
tests, benchmarks, and a README stating what the structure does, where it
applies, and how fast it is in its proper domain.

## The projects

The first four data structures originate from Chapter 7 of *The Art of
Writing Efficient Programs, Second Edition* by Fedor G. Pikus, where their
design is worked out in full — every principle these READMEs state without
explanation is argued there, benchmark by benchmark. The code here is live
and may continue to evolve past the version printed in the book.

- **[ConcurrentDeque](ConcurrentDeque/)** — `ConcurrentAppendDeque`, an
  array-like container with strictly wait-free element access that can be
  grown concurrently; elements never move, pointers are never invalidated.
- **[ConcurrentQueue](ConcurrentQueue/)** — `RingAtomicMapQueueMPMC`, a
  fixed-capacity MPMC ring-buffer queue with two isolated spinlock domains
  and a lock-free producer–consumer handoff.
- **[ConcurrentHash](ConcurrentHash/)** — `ConcurrentResizableHashSet`, a
  chained hash set with lock-free lookups and live resizing (inserts publish
  with one CAS, and append a new node under a spinlock unless `reclaim()` left
  one on a free list), built on three refusals: never free, never relink,
  never unlink (the one exception, `reclaim()`, runs when the caller has
  quiesced the set).
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

## Building

Each project builds independently with its own Makefile:

```sh
cd ConcurrentDeque   # or any other project
make                 # benchmarks + ASan/TSan unit tests
make run_tests
```

Requirements: a recent clang (the Makefiles use `clang++-22`, C++23),
[Google Benchmark](https://github.com/google/benchmark) and
[GoogleTest](https://github.com/google/googletest); set `GBENCH_DIR` and
`GTEST_DIR` if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.
Some projects (ConcurrentDeque, ConcurrentHash, ConcurrentQueue) share headers
via relative symlinks, so clone on a filesystem that supports them (on Windows,
use WSL or enable `core.symlinks`); SharedPtr and LockFreeList include their
sibling projects' headers through `-I` instead, and so do the benchmarks of
ConcurrentHash, ConcurrentQueue, IntrSharedPtr and LockFreeListRCU, for the
wall-clock accounting in `Spinlock/gb_wall_clock.h`.
IntrSharedPtr, SharedPtr and LockFreeList build on Linux only: the hazard
pointers they link issue `membarrier(2)`. Before their first build, clone the
third-party code they use (SharedPtr's README, "Third-party code"); their
first `make` then makes the patched copies the build uses. `make imports` makes
only those copies: in IntrSharedPtr its `mm_hp/`, in SharedPtr its
`lock_free_shared_ptr/parlay/`, and in LockFreeList both (by asking those two
projects).

## License

MIT — see [LICENSE](LICENSE).
