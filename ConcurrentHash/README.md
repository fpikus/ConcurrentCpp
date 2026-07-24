# ConcurrentResizableHashSet

A chained concurrent hash set with wait-free lookups, lock-free insertion,
optional deletion, and live resizing — safe for any number of threads, in any
mix of operations, with no external synchronization.

If you have ever tried to design a concurrent hash table, you know that two
problems dominate the effort: safe memory reclamation (when may a node be
freed while another thread may still be reading it?) and resizing (how do you
rehash while readers are traversing?). The textbook solutions — hazard
pointers, epochs, reader-writer locks — all tax the common path to pay for
the rare event. This data structure takes a different route: it does not solve
either problem; it arranges for neither problem to exist.

## Design principles

The entire design rests on three refusals, each stated in one line (and each
earning its own section in the book):

1. Nodes, once allocated, are never freed, moved, or reused before the set
   itself is destroyed.
2. A growing table never relinks its chains; new buckets are populated lazily,
   by copying.
3. Deletion sets a mark on the node rather than physically unlinking it.

Everything else follows: no hazard pointers, no epochs, no reference counts,
no reader registration of any kind. Readers *validate* instead of
*registering* — a trailing re-check of the table size asks "did the geometry
change under me?" and retries on the rare "yes". No use-after-free is possible
under any interleaving; no ABA problem is possible because integer indices are
never recycled. The elements live in an append-only node arena built on the
`ConcurrentAppendDeque` (from `../ConcurrentDeque`), which decouples the hash
geometry from data placement entirely: when the table grows, only small
integers move — the payload data sits still, forever.

## The domain of applicability

The limits are as much a part of the design as the speed, and they are strict:

- **Memory is reclaimed exactly once, at destruction.** Tombstoned nodes,
  stale parent-chain copies, and lost speculative splits all remain allocated
  until the set dies. This is the right trade for build-heavy, delete-light
  workloads with a bounded lifetime. For a long-lived table with continuous
  churn, it is the wrong structure — use a simple mutex-protected table and
  keep your memory.
- `erase()` exists only when the `AllowDelete` template parameter says so;
  with the default `false`, the delete-free usage pattern is enforced by the
  type system at zero runtime cost.
- There is no `size()`, no iteration, no `clear()` — deliberately. Every
  operation offered is transactional; those are not.
- The hash functor must be stateless; element copies must be equivalent to
  their originals; the destructor is not a concurrent operation.

## Performance

In its proper domain, the numbers are the argument. The baseline is the
textbook answer: `std::unordered_set` behind a `std::shared_mutex`.

Lookup-dominated workload (99% `contains()`, 1% `insert()`, half the lookups
missing): single-threaded, 19 ns per lookup against the baseline's 41 ns.
Under concurrency the baseline collapses first — down to ~6 million lookups/s
at 4 threads (750 ns each) as the reader count bounces between cores — while
this set scales to **490 million lookups per second** on a 16-core desktop
Ryzen and **1.1 billion lookups per second** on a large Granite Rapids server.

Insert-dominated workload (all keys new, table doubling live mid-benchmark):
comparable to the baseline at one thread, then sharply divergent — the
throughput amortizes allocation and lazy rehashing across threads until it
peaks between 8 and 16 threads, at which point the benchmark is no longer
measuring the hash table at all, only the physical limit of the arena
allocator's spinlock (acquired once per 1,024 elements, courtesy of the
deque underneath).

While these benchmarks run, the CPU fans rev up. Under the reader-writer-lock
baseline, the machine stays quiet: at 32 threads its threads spend over 95% of
their lifecycle asleep, waiting for the lock. If your high-performance
concurrency benchmark falls quiet, be suspicious.

## Building and testing

```sh
make            # benchmark + ASan/TSan unit tests
make run_tests  # run both sanitizer test binaries
```

Requires clang (the Makefile uses `clang++-22`, C++23), Google Benchmark and
GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your installations if they
are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.

- `concurrent_hash_set.h` — the hash set
- `concurrent_deque.h` — the backing store (see `../ConcurrentDeque`)
- `concurrent_hash_set_test.C` — unit tests (built with ASan and TSan)
- `concurrent_hash_set_bm.C` — the benchmarks quoted above

## The book

This directory accompanies Chapter 7 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The chapter contains what this
README deliberately omits: the split-bucket arithmetic that makes stale
copies mathematically incapable of resurrection, the insert-versus-resize race
and both of its resolutions, and the benchmark that lied — a lazily-evaluated
structure that made "setup" a fiction until the deferred work was forced to
quiesce.
