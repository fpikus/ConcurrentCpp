# ConcurrentResizableHashSetRCU

A copy of `ConcurrentResizableHashSet` (see `../ConcurrentHash`) with one
change: a node that goes dead — the tombstone `erase()` leaves, or the parent
copy a lazy split superseded — is taken out of its chain by the concurrent
operations themselves, instead of waiting in it for `reclaim()`. Everything
else is the original's: the append-only sharded arena, the lazy copying split,
lock-free lookups, inserts that publish with one CAS, and `reclaim()` at a
quiescent point. Read the original's README first; this one says only what is
different, and why.

## Why

The original keeps a dead node in its chain, where every later walk steps over
it, until `reclaim()` unlinks it. That is cheap and simple, and it is why the
original can promise that no node is ever reused while a thread may be looking
at it: at a quiescent point, nobody is. It is also why reclamation has to wait
for one. The road to reclaiming memory *without* stopping the world — by
epochs, generations, RCU, whatever you call it — begins by splitting "dead" in
two. A dead node still in its chain can be reached by any operation that starts
later, so nobody can ever say who might be looking at it. A dead node out of
its chain can be held only by an operation that started before it left, and it
can be freed once every such operation has finished. This copy manufactures
the second kind. The bookkeeping that would act on it — handles, generations,
reclamation that runs alongside the readers — is not here yet. This is step
one.

## What differs

The original rests on three refusals: never free, never relink, never unlink.
The first two stand — nothing is freed, and a doubling still copies rather
than relinks. The third is reworded:

3. Deletion sets a mark on the node; the node is then *bypassed*, never
   rewritten. A live link may be CASed to skip over a run of dead nodes; a dead
   node's own link is never written again by a concurrent operation, so a
   thread that reached the node before it was bypassed can still walk out of
   it, into the live chain.

A node is dead when it is a tombstone, or when it is a frozen parent copy
whose child bucket has been published (a frozen copy whose child is not yet
published still answers for its key, and no walk bypasses it). Dead nodes
leave their chains here:

- The split's winner, right after it publishes the child bucket, makes one
  pass over the parent chain and removes the copies it has just superseded,
  along with any other dead node it passes. The pass is lazy with respect to
  the doubling, as splitting is, and it belongs to whichever operation won
  the split.
- `erase()` removes its own tombstone at once: one attempt through the
  predecessor it tracked on the way in, one restart from the head, then it
  gives up.
- The walks of `insert()` and `erase()` bypass every dead run they pass behind
  a live word. A hit stops a walk at its key, so only a miss walks a whole
  chain.
- `contains()`'s own walk never writes; a lookup pays for none of this. The
  one way a lookup unlinks anything is by winning a split, in which case it
  runs that split's cleanup pass like any other splitter — just as, in the
  original, a lookup that meets an uninitialized bucket already does the
  split's work.

Removal is best effort, not a guarantee. Every site above spends one CAS per
dead run and never retries (the erase's restart is the one exception, bounded
at two attempts): a lost CAS means another thread completed a step on the same
word, and this one moves on. When that step was a peer bypassing the same run,
nothing is left behind. Otherwise — a node was prepended, the bucket was
sealed for the next doubling, the predecessor was itself tagged, a peer
bypassed a run of a different extent — the dead nodes stay reachable for a
while: a *straggler*. Stragglers also come from a walker whose view of the
table size is too stale to decide a frozen node, from a split stalled or
thrown between freezing a node and publishing its copy, and from a few
exception paths where `Hash{}` throws in the middle of an unlink. A straggler
is collected by the next writer walk that passes it, by the next split of any
child of its bucket (that pass walks the whole parent chain), or by
`reclaim()`, which still walks every chain and collects everything.

What no concurrent operation does, in this step, is reuse memory. An unlinked
node goes onto a per-shard *retired* list and waits there, as a tombstone waits
in the original, for `reclaim()` at a quiescent point. Retirement changes when
a dead node stops being walked, not when its memory comes back.

What the class does promise, it promises for an uncontended call — one during
which nothing else runs on the set, and which returns normally. Then: a split's
cleanup leaves no dead node reachable in the parent chain behind an untagged
word (the bucket head or a live node's link); an `insert()` or `erase()` walk
leaves no dead run reachable behind the head or any live node it passed; an
`erase()` leaves its tombstone unreachable, unless a frozen node whose child is
not yet published stands between the tombstone and the last live word before
it, which, uncontended, only an earlier split that threw can leave behind.

What it costs, in kind (what it costs in numbers, per type of workload, is
under [Performance](#performance)):

- a node grows by one word, the retired list's link — a word of its own,
  because the chain link of a retired node must stay as it was for the threads
  still walking out of it;
- an uncontended `erase()` performs three CASes where the original performs
  one: the mark, the unlink, and the push onto the retired list;
- every split copy freezes its parent node, for both values of `AllowDelete`
  (the original compiles the freeze out of the delete-free instantiation),
  because removing a run depends on every dead node in it being tagged;
- a writer walk re-hashes the frozen nodes it passes (once each; twice for
  one that ends a run), to find the child bucket that decides whether the
  node is dead; a lookup does not.

## What is not here yet

Possible next steps, none of them started and none of them promised:

- Handles and generations: every operation runs under a handle, the table
  bumps a generation now and then, and a retired node is freed once no handle
  old enough to have seen it is alive.
- Reclamation alongside the readers. The drain that returns retired nodes to
  the free lists needs less than quiescence (it must start after every
  operation that could still hold a retired node has finished, and operations
  ordered after its start may run while it drains, since nothing they can
  reach is on its lists), but the free lists would then take pushes while
  inserts pop from them, which today they do not have to.
- A `contains()` that also bypasses the dead runs it passes. It would make
  every reader a writer, which is why it was not done here; whether it pays
  is a measurement, and this version is the baseline it would be measured
  against.
- A reclaim in three calls: *retire* (a dummy writer that walks the published
  chains like a writer miss walk, unlinking what it passes and inserting
  nothing), *sequester* (set the retired lists aside, one pointer per shard),
  and *reclaim* (return what was set aside to the free lists).

## Performance

Measured against the original on the same fixtures, each pair built by the
same compiler, on four machines: a Ryzen 7940HS laptop (16 hardware threads,
clang-22), a 2-socket Xeon 6767P with 256 hardware threads, a 128-thread Zen 5
EPYC, and an Apple M3 Ultra running Linux (24 threads), the last three with
gcc 16.2; all at `-O3 -march=native`, with the header's default arena shard
count (the hardware thread count rounded up to a power of two: 16, 256, 128
and 32 shards) and threads unpinned. The figure quoted is the copy's
wall-clock throughput over the original's: the median over rounds (six on the
laptop, three on the servers), at thread counts from 1 to the machine's. Every
figure includes the cost of the 24-byte node against the original's 16.

- For read-heavy workloads (99% lookups on a stable set, 1% inserts) the
  speed is the same within 10-20%, except on the 2-socket Xeon at high thread
  counts (up to 25% slower). What loss there is comes mostly from the bigger
  node.
- For lookups over a set with many deletions (half the keys erased, no
  `reclaim()`) the copy is 1.1-1.5x faster, on every machine at every thread
  count.
- For insert-heavy growth (every key new, the table doubling through the run)
  the speed is within about 20% on x86; on the M3 Ultra the copy is 11-31%
  slower.
- For inserts into a pre-sized set the copy is 4-13% slower on x86 (up to 5%
  faster on the EPYC at high thread counts) and 1-18% slower on the M3 Ultra;
  the 2-socket Xeon is up to 28% slower at high thread counts.
- For erase-heavy workloads the original is 1.6-2.1x as fast (three CASes per
  erase against one) on the laptop, on the EPYC, and on the Xeon at low
  thread counts, and 1.4x on the M3 Ultra at low thread counts; it gets worse
  with threads on the big machines, up to 4.4x on the 2-socket Xeon and 2.6x
  on the M3 Ultra.
- For inserts into a set with many deletions (half the keys erased, no
  `reclaim()`) the copy is 5-22% faster on the laptop, on the M3 Ultra and on
  the EPYC at low thread counts, but up to 43% slower on the 2-socket Xeon at
  mid-range thread counts; elsewhere on the two servers it is within 10%
  either way where the rounds agree at all.
- A delete-free instantiation (`AllowDelete == false`), where the original
  skips the freeze and the copy does not: growth inserts are up to 40% slower
  than the original's delete-free instantiation, and the freeze itself costs
  the original up to 37% (on the EPYC). Pre-sized inserts built with gcc 16.2
  come out up to 27% slower than the copy's deletable instantiation from the
  same source; clang-22 does not show that difference.

No cell mixes inserts, lookups and erases, so there is no steady-state churn
number here: each line measures one kind of operation at a time, two of them
on a state that erases built.

## Building and testing

```sh
make            # benchmark + ASan/TSan unit tests + the -O3 store-buffering test
make tests      # the three test binaries only
make benchmarks # the benchmark only
make run_tests  # run all three test binaries
```

The compiler, C++ standard and library paths come from `../config.mk`, shared
by every directory in this repository and written once per machine; binaries
go to `build/<hostname>/`. Requires Google Benchmark and GoogleTest. The
benchmark also includes `../Spinlock/gb_wall_clock.h` (through
`-I../Spinlock`), the wall-clock accounting behind its `wall_items_per_second`
and `finish_spread` counters.

- `concurrent_hash_set_rcu.h` — the hash set; its design overview says where
  dead nodes leave their chains, what that guarantees, and what it does not
- `concurrent_deque.h` — the backing store (see `../ConcurrentDeque`)
- `spinlock.h` — the lock the arena's deques append under, and the resize
  lock (see `../Spinlock`)
- `concurrent_hash_set_rcu_test.C` — unit tests (built with ASan and TSan)
- `concurrent_hash_set_rcu_tso_test.C` — the store-buffering regression test;
  it can only fail when built `-O3` without a sanitizer, so it is its own
  binary
- `concurrent_hash_set_rcu_bm.C` — the benchmarks: the original's fixtures,
  so that the two sets' rows pair up, plus the cells the original lacks: two
  whose state erases built, one that times `erase()` itself, and a probe
  (`InsertHotGrowth`) that splits long, hot chains while threads insert into
  them, and counts the unlink CASes the splits' cleanup loses and the dead
  nodes left in their chains
