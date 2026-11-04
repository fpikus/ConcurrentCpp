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

## Performance (2026-11-02)

Measured on a two-socket EPYC 9555 (Zen 5; 128 hardware threads, 8 NUMA
nodes), both benchmarks built with GCC 16.2 `-O3 -march=native`. Rates are
wall-clock: every thread's items summed over the span from the first thread's
start to the last thread's finish, in millions per second. (Google Benchmark's
own items/s divides by the mean per-thread time; it agrees at one thread and
runs higher wherever threads finish apart.) Each cell is the median over four
processes per binary (two rounds, each run in ABBA order against
`../LockFreeList`'s benchmark at the same thread count), 5 repetitions per
process, unpinned. The workloads
are `LockFreeList`'s five, operation for operation. The first three columns
are its pointer policies, and `RCU` is this list with a long-lived handle per
thread, refreshed on every operation (once per lap in ReadDispersed), and
`reclaim()` called by every thread after every 256 of its own successful
erases. Shown: 1, 4, 16, 64 and 128 threads and the `_RCU` rows. The benchmark
also runs 2, 8 and 32 threads and two more regimes, `_RCU_NoReclaim` and
`_RCU_OpHandle` (see `lock_free_list_rcu_bm.C`).

| Workload          | Threads | StdAtomic | IntrPtr | IntrPtrHP |    RCU |
|-------------------|--------:|----------:|--------:|----------:|-------:|
| ReadHeavy         |       1 |      1.15 |    1.49 |      2.09 | 31.14† |
| ReadHeavy         |       4 |     0.71† |   0.67† |     0.45† | 25.51† |
| ReadHeavy         |      16 |     0.62† |   1.31† |     0.40† | 15.26† |
| ReadHeavy         |      64 |         — |   1.25† |      0.86 | 10.46† |
| ReadHeavy         |     128 |         — |       — |      0.98 |   8.16 |
| WriteHeavy        |       1 |      7.89 |    9.18 |     10.04 |  69.14 |
| WriteHeavy        |       4 |     0.67† |   7.92† |     1.28† |  4.83† |
| WriteHeavy        |      16 |     0.18† |   5.13† |     0.35† |  2.25† |
| WriteHeavy        |      64 |         — |   2.06† |     0.17† |  1.21† |
| WriteHeavy        |     128 |     0.04† |   1.05† |      0.13 |  0.93† |
| Graveyard         |       1 |     14.41 |   18.34 |     21.39 |  96.54 |
| Graveyard         |       4 |     1.31† |  16.14† |     2.26† |  8.65† |
| Graveyard         |      16 |     0.44† |   9.18† |     0.73† |  3.65† |
| Graveyard         |      64 |     0.10† |       — |      0.27 |  1.70† |
| Graveyard         |     128 |     0.06† |       — |      0.20 |  1.36† |
| MassiveHeadInsert |       1 |      9.56 |   12.84 |     14.28 | 76.53† |
| MassiveHeadInsert |       4 |     0.83† |  10.07† |     1.52† | 28.36† |
| MassiveHeadInsert |      16 |     0.18† |   7.08† |     0.54† |      — |
| MassiveHeadInsert |      64 |      0.06 |   2.81† |     0.25† |      — |
| MassiveHeadInsert |     128 |         — |   1.23† |     0.21† |      — |
| ReadDispersed     |       1 |      1.05 |    1.37 |      1.81 |  2.82† |
| ReadDispersed     |       4 |     1.91† |   1.37† |     1.28† |  8.12† |
| ReadDispersed     |      16 |     4.43† |   3.42† |     4.03† | 16.89† |
| ReadDispersed     |      64 |     7.96† |   7.96† |     6.98† | 27.43† |
| ReadDispersed     |     128 |    11.48† |  12.09† |     11.55 | 18.42† |

† Not a conclusion: the repetitions' coefficient of variation exceeds 5% (at
1–4 threads) or 10% (at 16 threads and more); or the medians of the two ABBA
halves of either round differ by more than 1% and by more than the larger
of their two coefficients of variation; or the per-process medians fall into
two levels more than 10% apart (an unpinned process can land on different
chiplets or sockets from one run to the next, and so can its number).

— Left out: the cell's threads finish more than half the run's span apart,
so a run-wide rate mixes the contended phase with a tail in which few threads
are still running; neither that rate nor the mean per-thread rate is the
steady-state figure.

Code placement alone moves the one-thread cells: rebuilding the same code
with a 16-byte pad or a different loop alignment shifts the pointer policies'
cells by up to 1.1% and the RCU cells by up to 4%.

Two things the table says. At one thread, where nothing contends, the RCU
list is 1.6× (ReadDispersed, which is bound by the memory walk) to 15×
(ReadHeavy) faster than the fastest pointer policy: an operation costs it no
atomic increment and decrement per node visited. Under contention the picture
splits by workload. The read-heavy rate of the RCU list falls with the thread
count, from 31 million operations per second at one thread to 8 million at
128, but at 128 threads it is still 8× the rate of `IntrPtrHP`, the one
pointer cell quotable there. On the write-heavy workloads (WriteHeavy,
Graveyard) the spinlock-counted `IntrPtr` is ahead of the RCU list by
1.6–2.5× from 4 to 16 threads, and the gap narrows to about 1.1× on
WriteHeavy by 128 threads (Graveyard's `IntrPtr` cells are not quotable
there). MassiveHeadInsert, which inserts at the head and reads nothing, keeps
the RCU list ahead where its cell is quotable, 5.4× at one thread and 2.8× at
four; from 16 threads up its threads finish too far apart for a rate. Most of
these ratios come from daggered cells; their direction holds in every process
of the run.

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

Requires Google Benchmark and GoogleTest (`../config.mk` selects `clang++-22`
and C++23; the code also builds with `g++-16`); point `GBENCH_DIR` and
`GTEST_DIR` at your installations if they are not in `$HOME/GoogleBench` and
`$HOME/GoogleTest`.
The benchmark also includes `../Spinlock/gb_wall_clock.h` (through
`-I../Spinlock`), the wall-clock accounting behind its `wall_items_per_second`
and `finish_spread` counters.

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
