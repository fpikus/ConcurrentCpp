# LockFreeList

A Harris-style lock-free singly-linked list that deletes for real: nodes are
physically unlinked while other threads traverse them, and their memory is
returned to the allocator at the earliest safe moment (or, with the
hazard-pointer policy, at the next reclamation scan after it). Determining "the
earliest safe moment" is the lock-free memory reclamation problem, and this
directory is what its full solution looks like — in code and in benchmarks.

The concurrent hash set next door (`../ConcurrentHash`) achieved its
performance by refusing to reclaim memory at all. That strategy has a strict
domain of applicability, and outside of it the memory bill eventually comes
due. This data structure pays it.

## The contract

`LockFreeList<T, AtomicPtr>` (see `lock_free_list.h`) supports concurrent
insertion and erasure at arbitrary positions, with lock-free progress:
`insert_after()` and `erase_after()` follow the exactly-one-winner discipline,
losers help the winner finish, and `while (erase_after(head)) {}` drains the
list no matter how many threads fight over it. The headline clause is the
iterator contract: **iterators are never invalidated** — not by insertion,
not by erasure, not even by erasure of the very node the iterator points to. A
parked iterator holds its node alive and, when advanced, walks through the
graveyard of deleted nodes back into the live list. One habit of mind must be
surrendered at the door: there is no such thing as "the current contents" of a
lock-free list; each visited node was in the list at the moment it was
visited, and no stronger statement can be made.

## Design principles

Stated here, earned in the book:

- Every link — the head, every `next`, every iterator — is a strong,
  reference-counted pointer; globally shared ones are *atomic* shared
  pointers. Both classic dragons die structurally: readers cannot dangle, and
  the ABA problem cannot occur because the witness protects the evidence.
- Deletion is two-step, Harris-style: a marking CAS on the victim's own `next`
  pointer is the linearization point; physical unlinking is best-effort
  housekeeping that any thread may finish.
- The mark is part of the pointer's identity, so a dying anchor fails a
  competitor's CAS by itself — no separate race window to close.
- The destructor is iterative, not recursive; a million-node list must not die
  of stack overflow on its way out.
- Reference-count discipline: relaxed increments, acquire-release decrements.
  Relax the decrement and you have a use-after-free that strikes once a month
  in production. The hazard-pointer policy adds a third operation, the
  conditional increment `TryAddRef()` (increment only if the count is not
  zero), whose zero-observing load and successful CAS are acquire by contract:
  relax the load and a reader can re-read a stale pointer forever; relax the
  CAS and it can end up owning a node whose `next` the list's destructor has
  already torn down. The hooks' contract, orders included, is stated once in
  `../IntrSharedPtr/intr_pointee.h`; `Node` here writes the hooks by hand
  rather than inheriting the component's `intr_pointee_base`, because one
  `Node` serves all four policies, and a count supplied by a policy base
  would change the size of the `std::atomic` and parlay policies' nodes,
  which do not use it.

The honest summary: this is not really a lock-free list at all — it is an
atomic shared pointer written four different ways, with one simple list
balanced on top. The `AtomicPtr` policy parameter selects among:

1. `std::atomic<std::shared_ptr<T>>` — the standard's answer, with a hidden
   one-bit spinlock and no room for a mark bit (it forces a weaker deletion
   algorithm, restricted to non-adjacent erasure).
2. An intrusive pointer (`intr_shared_ptr.h`) — count embedded in the node,
   an explicit one-bit lock with deliberately asymmetric reader/writer
   backoff, and the Harris mark in bit 0.
3. Daniel Anderson's `parlay::atomic_shared_ptr`
   (`lock_free_shared_ptr/`) — genuinely lock-free, hazard pointers demoted
   to an implementation detail inside the pointer.
4. A hazard-pointer intrusive pointer (`intr_shared_ptr_hp.h`) — the
   intrusive pointer with the lock taken out: Maged Michael's hazard pointers
   (`mm_hp/`) close the gap between reading a link and counting it, which
   makes it lock-free once a thread has made its first load. The price is
   deferred reclamation: a node whose count reaches zero is not destroyed
   then, but later, in a batch, by whichever thread happens to trigger the
   next reclamation scan.

## The domain of applicability

This is the structure for nodal data — lists, and by extension trees, graphs,
skip lists — where elements must be inserted and removed at arbitrary
positions, memory must actually be returned, and iterators must survive
concurrent surgery. The cost is stated plainly: because of the reference
counting, even a read-only traversal is a write at the hardware level, so
read-side throughput does not scale the way the hash set's does. If your
workload does not need real reclamation, this design is roughly a thirty-fold
overpayment — use the hash set's arena approach, or partition the data and
skip shared structures entirely. Manipulating shared data is a way to enforce
synchronization boundaries, not the program itself; the realistic goal for a
hot shared structure is flat performance under concurrency, and that is the
right lens for everything below.

## Performance

Five workloads (read-heavy, write-heavy, graveyard, insertion-at-head,
dispersed) were benchmarked across the first three pointer policies, on
hardware from a 16-core Ryzen desktop to a 72-core NVIDIA Grace server; the
trends are remarkably consistent.

- **Dispersed workloads** — threads working on mostly separate sections of
  the list, the design's true fast path — perform excellently: full
  concurrent insertion, deletion, and never-dangling traversal at a baseline
  cost that stays flat as threads are added.
- **Read-heavy workloads**: the intrusive pointer (`intr_shared_ptr`) scales
  best at moderate thread counts — not by a better algorithm (it and the
  standard pointer both hide a one-bit spinlock) but by a leaner
  implementation and a backoff policy with manners; the difference between a
  spinlock with manners and one without is two orders of magnitude. At high
  thread counts the genuinely lock-free parlay pointer keeps scaling after
  both spinlock-based pointers collapse — it is bottlenecked by a per-core
  resource (the store buffer) rather than the global coherency mesh.
- **Insertion at the head** funnels every thread through one pointer and will
  never scale, on any implementation; there the simplest, tightest pointer
  (the intrusive one, `intr_shared_ptr`) wins.

The stark ledger against the hash set: the set refused to reclaim memory and
its readers scaled without visible limit; the list reclaims perfectly — every
node freed at the earliest safe moment, every iterator forever valid — and
its readers stand still under contention. The choice of a memory reclamation
strategy is not an implementation detail of a concurrent design. On read-heavy
workloads, it *is* the design.

## Building and testing

```sh
make            # benchmark + ASan/TSan unit tests
make run_tests  # run both sanitizer test binaries
```

Requires Linux, clang (`../config.mk` selects `clang++-22` and C++23), Google
Benchmark and GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your
installations if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.
Linux because every binary links Maged Michael's hazard pointers, which issue
`membarrier(2)`. Those, and Daniel Anderson's pointer, are made from their
upstream repositories by the `make imports` targets of `../IntrSharedPtr` and
`../SharedPtr`, which the first build here runs; clone the upstream
repositories first (see SharedPtr's README, "Third-party code").

- `lock_free_list.h` — the list
- `lock_free_list_test.C` — unit tests (built with ASan and TSan)
- `lock_free_list_bm.C` — the benchmarks described above; the row suffixes
  `_StdAtomic`, `_IntrPtr`, `_HazardPtr` and `_IntrPtrHP` are the four
  policies in the order listed above — `_HazardPtr` is the parlay pointer, a
  name older than the hazard-pointer policy, whose rows are `_IntrPtrHP`
- `lock_free_list_bugs.md` — the record of a correctness review of the list
  algorithm: what broke, why, and what was done about it

The pointer policies come from two sibling projects, through
`-I../IntrSharedPtr -I../SharedPtr`:

- `../IntrSharedPtr/intr_shared_ptr.h` — the intrusive atomic shared pointer
- `../IntrSharedPtr/intr_shared_ptr_hp.h` — the hazard-pointer intrusive
  atomic shared pointer
- `../IntrSharedPtr/intr_pointee.h` — the contract of the hooks `Node`
  implements (and the base class a node not shared between policies would
  inherit instead)
- `../IntrSharedPtr/mm_hp/` — Maged Michael's hazard pointers, which it uses
  (made by `make imports` in `../IntrSharedPtr`)
- `../IntrSharedPtr/hp_drain.h`, `../IntrSharedPtr/hp_drain_gtest.h` — drain
  the hazard pointers' pending reclamations, so that a test can assert that a
  node is gone
- `../SharedPtr/lock_free_shared_ptr/` — Daniel Anderson's lock-free atomic
  shared pointer (its `parlay/` made by `make imports` in `../SharedPtr`)
- `../SharedPtr/atomic_shared_ptr_concept.h` — the concept the pointer
  policies model

The benchmark also includes `../Spinlock/gb_wall_clock.h` (through
`-I../Spinlock`), the wall-clock accounting behind its `wall_items_per_second`
and `finish_spread` counters.

The pointers are tested on their own in `../SharedPtr` (see
[../SharedPtr/README.md](../SharedPtr/README.md)) and `../IntrSharedPtr` (see
[../IntrSharedPtr/README.md](../IntrSharedPtr/README.md), which also says how
to build a structure of your own on the intrusive pointers).

## The book

Chapter 7 of *The Art of Writing Efficient Programs, Second Edition* by
Fedor G. Pikus covers an earlier version of this code; the code here has
moved on since.
