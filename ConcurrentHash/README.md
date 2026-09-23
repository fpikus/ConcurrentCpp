# ConcurrentResizableHashSet

A chained concurrent hash set with optional deletion and live resizing — safe
for any number of threads, in any mix of operations, with no external
synchronization. Lookups take no lock and write no shared memory, except when a
lookup is the first to touch a bucket after a resize and performs its lazy
split. Inserts publish with a single CAS but allocate their node under the
arena's spinlock (see [Performance](#performance)), so insertion is not
lock-free.

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
allocator's spinlock. There are two locks, and they are taken at very
different rates: the bucket table is doubled under a lock once per doubling,
but every new node — including every node copied by a lazy split — is
appended to the arena deque under its lock, one acquisition per element.

While these benchmarks run, the CPU fans rev up. Under the reader-writer-lock
baseline, the machine stays quiet: at 32 threads its threads spend over 95% of
their lifecycle asleep, waiting for the lock. If your high-performance
concurrency benchmark falls quiet, be suspicious.

## Building and testing

```sh
make            # benchmark + ASan/TSan unit tests + the -O3 store-buffering test
make run_tests  # run all three test binaries
```

The compiler, C++ standard and library paths come from `../config.mk`, shared
by every directory in this repository and written once per machine; binaries
go to `build/<hostname>/`. Requires Google Benchmark and GoogleTest.

- `concurrent_hash_set.h` — the hash set
- `concurrent_deque.h` — the backing store (see `../ConcurrentDeque`)
- `concurrent_hash_set_test.C` — unit tests (built with ASan and TSan)
- `concurrent_hash_set_tso_test.C` — the store-buffering regression test; it
  can only fail when built `-O3` without a sanitizer, so it is its own binary
- `concurrent_hash_set_bm.C` — the benchmarks quoted above

## The book

This directory accompanies Chapter 7 of *The Art of Writing Efficient
Programs, Second Edition* by Fedor G. Pikus. The chapter contains what this
README deliberately omits: the split-bucket arithmetic that makes stale
copies mathematically incapable of resurrection, the insert-versus-resize race
and the rule that resolves it (every decision is made on one atomic word, so
the decision and the split that could invalidate it are totally ordered), and
the benchmark that lied — a lazily-evaluated structure that made "setup" a
fiction until the deferred work was forced to quiesce.
