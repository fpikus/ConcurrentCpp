# LockFreeListRCU

The lock-free list of `../LockFreeList`, with the reference counts taken out.
That list keeps a node alive by counting every pointer to it, and it pays for
the count at every step of every traversal: even a read-only walk is a write
at the hardware level. This one keeps nodes alive by *generations*. Every
operation runs under a handle, every handle belongs to a generation, and an
erased node is not freed until no handle that could still see it is left. A
traversal step goes back to being a single acquire load. The bill does not
disappear; it moves — to the handle, which pins memory for as long as it
lives, and to whoever calls `reclaim()`.

The list itself is the same Harris-style algorithm, now on plain CAS over raw
marked pointers: a mark on the victim's own `next` pointer is the
linearization point of an erase, the physical unlink comes after it, and any
thread may finish the unlink.

## Handles

`list.new_handle()` joins the list's current generation; the handle's
destructor leaves it. While a handle lives, everything it can observe stays
allocated and keeps its value: every node reachable from the head at the time
it joined or later, every node an iterator obtained under it is parked on,
and the whole graveyard chain such an iterator can still walk out through. As
in the reference-counted list, iteration does not skip deleted nodes; a
parked iterator walks the graveyard back into the live list.

The handle is the first argument of every list operation (`reclaim()` needs
none), so that the protection a traversal relies on is visible at the call
site. It also means these are not standard iterators: `begin()` takes a
handle, and range-for is out.

The iterator contract carries one amendment. `LockFreeList`'s iterators are
never invalidated; here, **an iterator is valid until the handle it was
obtained under is destroyed, refreshed, or move-assigned over**. A moved
handle keeps its iterators: they stay valid under the move target. The price
of the clause is paid by long walks. `h.refresh()` moves a handle to the
current generation and releases the old one, and it invalidates every
iterator obtained under the handle — so a thread walking a large list cannot
release old generations without restarting, and a long traversal pins every
node retired since it began.

A const list yields handles too, and its `begin()` returns a
`const_iterator`: `const` on your reference does not stop other threads from
erasing and reclaiming through theirs, so a const reader still needs a handle.

## Retirement and reclamation

A node is *retired* — pushed onto the list's retired list — at its physical
unlink, by the one thread whose unlink CAS succeeded, and never at the mark:
until it is unlinked, a node is still reachable from the head, and a node
reachable from the head is not garbage yet. (A marked node whose unlink lost a
race stays linked until a later erase over the same edge helps it out.)

Nothing is freed until somebody calls `reclaim()`. A call closes the current
generation — everything retired so far becomes that generation's *bag* —
opens a new generation for new and refreshed handles to join, and then frees
bags oldest-first, stopping at the first generation that still has a live
handle. Freeing is cumulative: a bag is never freed while an older generation
is still pinned, because a handle of generation *g* can observe nodes that
land in bags *g*, *g* + 1, … and is protected only if every one of them waits
for it. A freed node is not returned to the allocator. Its value is destroyed
and the node goes to a free list, from which `insert_after()` and
`emplace_after()` take nodes before they call `new`; a node that has once been
in the list goes back to the allocator only in the list's destructor. So an
erased node's `~T` runs inside `reclaim()`, on whichever thread called it —
not inside `erase_after()`.

One `reclaim()` runs at a time, enforced by a try-flag: a call that finds
another in progress returns `contended` at once, having done nothing.
`reclaim()` refuses rather than waits — the other refusals are
`nothing_retired` and `ring_full`, the latter when the fixed ring of
generations is full and its oldest generation is still pinned — and the
remedy for every refusal is to call again later; there is no automatic
recovery. A reclaimer stalled inside `reclaim()` (preempted, suspended)
stalls all reclamation until it resumes, and memory grows; it never stalls a
list operation. That is the same consequence as a thread sitting on a handle,
which holds back the reclamation of its generation and of every younger one,
and nothing else. A thread that holds a handle frees nothing at or after its
own handle's generation when it calls `reclaim()`, so a thread that both
works on the list and reclaims should call `h.refresh(); list.reclaim();`.

List operations (an insert apart from its `new`), handle creation and
refresh are lock-free; `reclaim()` is non-blocking by refusal.

The full contract — what each `reclaim()` result means and when to retry,
the exception guarantees of `insert_after()` and `emplace_after()`, the
thread-safety rules for handles, the precondition violations a debug build
diagnoses — is the overview comment at the top of `lock_free_list_rcu.h`. That
comment, not this page, is the authority, and its protocol overview justifies
every memory order in the code.

## Footprint

`LockFreeListRCU<int>` with the default 64 generations is 8832 bytes (about
8.6 KiB) before its first node: 64 generation blocks of 128 bytes each, so
that neighboring generations' counts never share a cache line, plus five
128-byte lines for the list's shared words (the same in a debug build). In
a release build a node is 24 bytes for `int` (value, next, retire link) and a
handle is one pointer; debug builds add checking fields to both.

## Performance (2026-10-04)

Measured on a 16-thread laptop (Ryzen 9 7940HS, 8 cores) under WSL2, both
benchmarks built with clang++-22 `-O3 -march=native`. Rates are Google
Benchmark's standard items/s (summed iterations over the mean per-thread
time), in millions per second; each cell is the median of 5 repetitions,
averaged over two processes run in ABBA order against `../LockFreeList`'s
benchmark at the same thread count. The workloads are `LockFreeList`'s five,
operation for operation. The first four columns are its pointer policies —
`HazardPtr` is Daniel Anderson's `parlay::atomic_shared_ptr` (the name
predates the hazard-pointer policy, whose column is `IntrPtrHP`) — and `RCU`
is this list with a long-lived handle per thread, refreshed on every
operation (once per lap in ReadDispersed), and `reclaim()` called by every
thread after every 256 of its own successful erases. Shown: 1, 4 and 16
threads and the `_RCU` rows; cells whose ABBA halves disagreed in the first
round were re-run, and the re-run is shown. The benchmark also runs 2 and 8
threads and two more regimes, `_RCU_NoReclaim` and `_RCU_OpHandle` (see
`lock_free_list_rcu_bm.C`).

| Workload          | Threads | StdAtomic | IntrPtr | HazardPtr | IntrPtrHP |    RCU |
|-------------------|--------:|----------:|--------:|----------:|----------:|-------:|
| ReadHeavy         |       1 |      5.53 |    3.81 |      1.32 |      5.65 |  24.36 |
| ReadHeavy         |       4 |     2.24† |    2.49 |     0.60† |      1.12 |  54.49 |
| ReadHeavy         |      16 |     2.96† |    3.47 |      1.71 |      3.24 |  65.01 |
| WriteHeavy        |       1 |     20.33 |   22.15 |    11.23† |     20.92 |  72.39 |
| WriteHeavy        |       4 |     1.90† |   17.86 |     2.18† |     3.14† | 13.65† |
| WriteHeavy        |      16 |      0.70 |   12.74 |      1.07 |      1.55 |   9.90 |
| Graveyard         |       1 |    36.73† |  38.58† |     22.05 |     34.71 |  96.11 |
| Graveyard         |       4 |     3.22† |   32.03 |     3.50† |     5.50† | 20.95† |
| Graveyard         |      16 |      1.02 |   22.28 |      1.58 |      2.78 |  14.31 |
| MassiveHeadInsert |       1 |    27.41† |  32.81† |    24.52† |    33.91† | 70.50† |
| MassiveHeadInsert |       4 |     2.26† |  15.91† |     2.81† |     3.31† | 72.45† |
| MassiveHeadInsert |      16 |     0.64† |   12.85 |      1.52 |      1.70 |  25.28 |
| ReadDispersed     |       1 |     2.82† |   2.81† |     1.05† |     2.43† |   2.56 |
| ReadDispersed     |       4 |     2.87† |   4.16† |     0.66† |     3.15† |   8.33 |
| ReadDispersed     |      16 |    10.05† |   12.53 |      5.44 |    10.11† | 24.26† |

† Not a conclusion: the repetitions' coefficient of variation exceeds 5% (at
1–4 threads) or 10% (at 16 threads), or the medians of the two ABBA halves
differ by more than the larger of their two coefficients of variation, in
the re-run as well.

## Building and testing

```sh
make                 # benchmark + ASan/TSan unit tests (same as make all)
make benchmarks      # the benchmark only (Google Benchmark, no sanitizers)
make tests           # the two sanitizer test binaries
make run_tests       # build and run both sanitizer test binaries
make run_benchmarks  # build and run the benchmark
```

The binaries are written to `build/$(hostname)/`: `lock_free_list_rcu_bm`,
`lock_free_list_rcu_test_asan` and `lock_free_list_rcu_test_tsan`. The
comparison rows above come from `../LockFreeList`'s own benchmark, built there
with `make benchmarks`.

Requires clang (`../config.mk` selects `clang++-22` and C++23), Google
Benchmark and GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your
installations if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.

- `lock_free_list_rcu.h` — the list; its overview comment is the contract
- `rcu_test_common.h` — shared test payloads, test hooks and accounting
  checks
- `lock_free_list_rcu_test.C` — basic and stress tests (ported from
  `../LockFreeList`)
- `lock_free_list_rcu_scheme_test.C` — white-box tests of the reclamation
  scheme
- `lock_free_list_rcu_contract_test.C` — black-box tests of the public
  contract
- `lock_free_list_rcu_bm.C` — the benchmarks: `LockFreeList`'s five
  workloads under up to three regimes, the row suffixes `_RCU` (a long-lived
  handle per thread), `_RCU_NoReclaim` (the same, never reclaiming) and
  `_RCU_OpHandle` (a fresh handle per operation; not on ReadDispersed)
