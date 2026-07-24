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
  chained hash set with wait-free lookups, lock-free insertion, and live
  resizing, built on three refusals: never free, never relink, never unlink.
- **[LockFreeList](LockFreeList/)** — a Harris-style lock-free singly-linked
  list that reclaims memory for real, with never-invalidated iterators,
  parameterized over three atomic shared pointer implementations.

Supporting components shared by the projects above:

- **[SharedPtr](SharedPtr/)** — atomic reference-counted smart pointers: an
  intrusive pointer with an embedded one-bit lock, and an adapter for Daniel
  Anderson's genuinely lock-free `atomic_shared_ptr` (fetched separately;
  see [SharedPtr/lock_free_shared_ptr](SharedPtr/lock_free_shared_ptr/)).
- **[Spinlock](Spinlock/)** — the spinlock used throughout.

New projects added after the book's publication will be exactly that — new,
and documented on their own terms.

## Building

Each of the four data-structure projects builds independently with its own
Makefile (the supporting components are header-only and are built as part of
the projects that use them):

```sh
cd ConcurrentDeque   # or any other project
make                 # benchmarks + ASan/TSan unit tests
make run_tests
```

Requirements: a recent clang (the Makefiles use `clang++-22`, C++23),
[Google Benchmark](https://github.com/google/benchmark) and
[GoogleTest](https://github.com/google/googletest); set `GBENCH_DIR` and
`GTEST_DIR` if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.
The projects share headers via relative symlinks, so clone on a filesystem
that supports them (on Windows, use WSL or enable `core.symlinks`).

## License

MIT — see [LICENSE](LICENSE).
