# ConcurrentResizableHashSet

A chained concurrent hash set with optional deletion and live resizing — safe
for any number of threads, in any mix of operations, with no external
synchronization (the one exception, `reclaim()`, is described below). Lookups take no lock and write no shared memory, except when a
lookup is the first to touch a bucket after a resize and performs its lazy
split. Inserts publish with a single CAS; their node is popped, lock-free,
from the free list of the calling thread's arena shard when `reclaim()` has
left one there, and otherwise appended under the shard's spinlock (see
[Performance](#performance)), so insertion is not lock-free.

If you have ever tried to design a concurrent hash table, you know that two
problems dominate the effort: safe memory reclamation (when may a node be
freed while another thread may still be reading it?) and resizing (how do you
rehash while readers are traversing?). The textbook solutions — hazard
pointers, epochs, reader-writer locks — all tax the common path to pay for
the rare event. This data structure takes a different route: it does not solve
either problem; it arranges for neither problem to exist.

## Design principles

The entire design rests on three refusals, each stated in one line:

1. Nodes, once allocated, are never freed, moved, or reused while any
   operation can see them.
2. A growing table never relinks its chains; new buckets are populated lazily,
   by copying.
3. Deletion sets a mark on the node rather than physically unlinking it.

Everything else follows: no hazard pointers, no epochs, no reference counts,
no reader registration of any kind. Readers *validate* instead of
*registering* — a trailing re-check of the table size asks "did the geometry
change under me?" and retries on the rare "yes". No use-after-free is possible
under any interleaving; no ABA problem is possible because a node's address is
never reused while any operation is running. The elements live in an append-only node arena built on the
`ConcurrentAppendDeque` (from `../ConcurrentDeque`), whose blocks never move,
which decouples the hash geometry from data placement entirely: when the table
grows, only pointers move — the payload data sits still, forever.

The one exception to the refusals is `reclaim()`, and it is an exception in
time, not in mechanism: the caller may run it only when nothing else is
running on the set. It takes no lock; quiescence is the caller's promise.
`reclaim()` unlinks tombstones and superseded split copies, collects the nodes
that lost races left behind, and deals them all out to per-shard free lists
that later inserts pop from. Every guarantee above holds within one period
bounded by `reclaim()` calls. The client keeps no references into the table
during or across `reclaim()` (the API hands out none), and the only thing that
carries across it is membership: a value in the set is still in it, a value
not in it still is not. No ABA problem can reach the client, and none reaches
the free lists either: nothing but `reclaim()` pushes onto a free list, and no
CAS can be held across a `reclaim()` call, because nothing else runs during it.

## The domain of applicability

The limits are as much a part of the design as the speed, and they are strict:

- **Memory is recycled only at quiescent points.** Tombstoned nodes, stale
  parent-chain copies, and lost speculative splits remain allocated until the
  next `reclaim()`, which recycles them for later inserts; without one, until
  the set dies. `reclaim()` never returns memory to the system and never
  shrinks the table. This is the right trade for build-heavy, delete-light
  workloads, and for churn that comes in rounds with a natural point where
  every thread stops. For a long-lived table with continuous churn and no such
  point, it is the wrong structure — use a simple mutex-protected table and
  keep your memory.
- `erase()` exists only when the `AllowDelete` template parameter says so;
  with the default `false`, the delete-free usage pattern is enforced by the
  type system at zero runtime cost.
- There is no `size()`, no iteration, no `clear()` — deliberately. Every
  concurrent operation offered is transactional; those are not. (`reclaim()`
  returns the exact element count: at a quiescent point, exactness is free.)
- The hash functor must be stateless; element copies must be equivalent to
  their originals; elements must be copy-assignable, because a recycled node
  is reused by assignment; an erased element's resources are released when
  its node is reused (the assignment overwrites it) and its destructor runs
  only when the set dies, never in `erase()`; the destructor and
  `reclaim()` are not concurrent operations.

## Performance

In its proper domain, the numbers are the argument. The baseline is the
textbook answer: `std::unordered_set` behind a `std::shared_mutex`.

Lookup-dominated workload (99% `contains()`, 1% `insert()`, half the lookups
missing; measured before the key mixer, with keys that were already random):
single-threaded, 19 ns per lookup against the baseline's 41 ns.
Under concurrency the baseline collapses first — down to ~6 million lookups/s
at 4 threads (750 ns each) as the reader count bounces between cores — while
this set scales to **490 million lookups per second** on a 16-core desktop
Ryzen and **1.1 billion lookups per second** on a large Granite Rapids server.

Insert-dominated workload (all keys new, table doubling live mid-benchmark):
the benchmark inserts each thread's keys through a bijective mixer, so that
inserts land on random buckets and every insert into a large table is a
cache miss on its bucket head. An earlier version of the benchmark inserted
sequential keys under the identity hash, which streams through consecutive
bucket heads and lets every thread's n-th key share bucket n with every
other thread's: that measured the prefetcher and chains as long as the
thread count, and its numbers, once quoted here, were wrong by an order of
magnitude. Under random access, every new node — including every node
copied by a lazy split — used to be appended to one arena deque under one
spinlock, and that lock was the ceiling. (A lock-free counter in its place
measured slower still, but only with the flawed keys; that comparison has
not been repeated.) The arena is now sharded: by default one append-only
deque per hardware thread, the count rounded up to a power of two, and each
thread allocates from the shard its thread number selects, popping a node from
that shard's free list when `reclaim()` has left one there and otherwise
appending under the shard's lock, so the lock is contended only when thread
numbers collide modulo the shard count. On a 16-thread laptop (Ryzen 7940HS) that took the pre-sized
8M-bucket fixture (`Insert_MostlyNew_Presized`) from about 30 to 77-95
million random inserts per second at 16 threads (two runs a day apart), against 12 to 14 million
single-threaded; on a 256-thread Granite Rapids server one shard collapses to
3 million per second at 256 threads and the default shard count reaches about
300 million. The bucket table is still doubled under one lock, once per
doubling.

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
go to `build/<hostname>/`. Requires Google Benchmark and GoogleTest. The
benchmark also includes `../Spinlock/gb_wall_clock.h` (through
`-I../Spinlock`), the wall-clock accounting behind its `wall_items_per_second`
and `finish_spread` counters.

- `concurrent_hash_set.h` — the hash set
- `concurrent_deque.h` — the backing store (see `../ConcurrentDeque`)
- `concurrent_hash_set_test.C` — unit tests (built with ASan and TSan)
- `concurrent_hash_set_tso_test.C` — the store-buffering regression test; it
  can only fail when built `-O3` without a sanitizer, so it is its own binary
- `concurrent_hash_set_bm.C` — the benchmarks quoted above

## The book

Chapter 7 of *The Art of Writing Efficient Programs, Second Edition* by
Fedor G. Pikus covers an earlier version of this code; the code here has
moved on since.
