# ConcurrentResizableHashSetRCU

A copy of `ConcurrentResizableHashSet` (see `../ConcurrentHash`) with two
changes. A node that goes dead — the tombstone `erase()` leaves, or the parent
copy a lazy split superseded — is taken out of its chain by the concurrent
operations themselves, best effort, instead of waiting in it for a quiescent
point. And its memory comes back while the operations run: every operation runs
under a handle, and a caller-driven `advance()` frees what no handle can still
see; `reclaim()` is that call without its walk of the table, and a set
constructed with a cadence does that work by itself; there is no quiescent
point. Everything else is the original's: the append-only sharded arena, the
lazy copying split, inserts that publish with one CAS, and operations that are
lock-free except where they allocate: the hash's own protocol (walks, splits,
unlinks, the publishing CAS) takes no lock; a node comes from a free list
without a lock, or from an append under the arena shard's spinlock (a lookup
can get there too, by splitting a bucket it meets uninitialized); and the
insert that crosses the doubling threshold grows the bucket array under the
table's resize lock, after its own key is published, while another insert that
has also crossed a doubling threshold waits for it. Read the original's README
first; this one says only what is different, and why.

The requirements on `T` and `Hash` are the original's, spelled out here in one
place: `Hash` must be default-constructible and callable on a `const T&`,
returning the same value for a key every call, and since it is
default-constructed wherever a hash is needed and never stored, a `Hash` with
state only ever has its default state; `T` must be copy-constructible and
copy-assignable (every node holds a copy of its key, and a node reused from a
free list takes its new value by assignment) and comparable as
`value == key`, a stored `T` against a `const T&`, consistently with `Hash`:
keys that compare equal must hash equal. `Hash` is applied to stored values
as well as to the keys of calls, and one of the places is the walk of
`advance()`, which hashes the frozen nodes it passes as every writer walk
does: a `Hash` that throws can make `advance()` throw. `reclaim()` hashes
nothing and never throws.

## Why

The original keeps a dead node in its chain, where every later walk steps over
it, until the original's `reclaim()` unlinks it. That is cheap and simple,
and it is why the original can promise that no node is ever reused while a
thread may be looking at it: at a quiescent point, nobody is. It is also why
reclamation has to wait for one. The road to reclaiming memory *without*
stopping the world — by epochs, generations, RCU, whatever you call it —
begins by splitting "dead" in two. A dead node still in its chain can be
reached by any operation that starts later, so nobody can ever say who might
be looking at it. A dead node out of its chain can be held only by an
operation that did not start after it left, in happens-before order, and it
can be freed once every such operation has finished, in the same order. This
copy manufactures the second kind, and keeps the books that act on it:
handles and generations tell when every operation that could still hold such
a node has finished, and its memory comes back while the others keep running.

## What differs

The original rests on three refusals: never free, never relink, never unlink.
The first two stand — a node is never freed or reused while an operation can
see it, and a doubling still copies rather than relinks — but who can see a
node is decided by handles and generations (the section after this one), not
by a quiescent point. The third is reworded:

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
  pass over the parent chain and tries to remove the copies it has just
  superseded, along with any other dead node it passes — except those behind
  a frozen node the pass cannot yet decide (its own child unpublished, or
  published for a doubling the pass has not seen), which it cannot touch
  (stragglers, below). The pass is lazy with respect to the doubling, as
  splitting is, and it belongs to whichever operation won the split.
- `erase()` tries to remove its own tombstone at once: one attempt through the
  predecessor it tracked on the way in, one restart from the head, then it
  gives up.
- The walks of `insert()` and `erase()` try to bypass every dead run they pass
  behind a live word. A hit stops a walk at its key, so only a miss walks a
  whole chain.
- `contains()`'s own walk never writes; a lookup pays for none of this. The one
  way a lookup unlinks anything is by winning a split, in which case it runs
  that split's cleanup pass like any other splitter — just as, in the original,
  a lookup that meets an uninitialized bucket already does the split's work. On
  a set constructed with a cadence, the retirements of that cleanup pass can
  trigger a reclamation like any other's (the cadence, below).

Removal is best effort, not a guarantee. Every site above spends one CAS per
dead run and never retries (the erase's restart is the one exception, bounded
at two attempts): a lost CAS means another thread completed a step on the same
word, and this one moves on. When that step was a peer bypassing the same run,
nothing is left behind. Otherwise — a node was prepended, the bucket was
sealed for the next doubling, the predecessor was itself tagged, a peer
bypassed a run of a different extent — the dead nodes stay reachable for a
while: a *straggler*. Stragglers also come from a walker or a cleanup pass
whose view of the table size is too stale to decide a frozen node, from a
split stalled or thrown between freezing a node and publishing its copy, and
from a few exception paths where `Hash{}` throws in the middle of an unlink. A
straggler gets another attempt from the next writer walk that passes it, from
the next split of any child of its bucket (that pass walks the whole parent
chain), and from the walk of `advance()`, which passes every published chain
with the same one attempt per run: a dead run that no writer walks past again
waits for the next `advance()`, so its lifetime is bounded by how often the
caller calls it (and by that one CAS winning), never by time. The exception
is a straggler behind a frozen node whose child is not yet published: that
node is not dead, so no walk bypasses it, and its link, the word in front of
the straggler, is written by no concurrent operation; the straggler stays
through any number of walks, splits, doublings and `advance()` calls, until
the node's child is published (from then on the cleanup pass of the split
that published it, or any walk that passes the node, `advance()`'s included,
can take the node and the straggler out together).

What no concurrent operation does of its own accord is make memory reusable:
inserts and splits do take nodes off the free lists, but only a *bump* and the
free pass after it — `advance()`'s, `reclaim()`'s, or, on a set constructed
with a cadence, the one an operation's own retirement triggers — put nodes on
them. An unlinked node goes onto a per-shard *retired* list and waits there
for the next bump, which bags it with the generation it closes; a node an
operation allocated and never published — a split's subchain that lost its
publishing CAS or was abandoned by a throw, an insert's node that a retry
found redundant, a reused node whose value assignment threw — goes onto the
shard's *limbo* list and takes the same road. Retirement changes when a dead
node stops being walked, not when its memory comes back; the bag decides
that, and the next section says how.

What the class does promise about removal, it promises for an uncontended
call — one that every other call on the set happens-before or happens-after,
and which returns normally. Then: a split's cleanup leaves no dead node
reachable in the parent chain behind an untagged word (the bucket head or a
live node's link); an `insert()` or `erase()` walk leaves no dead run
reachable behind the head or any live node it passed; an `erase()` leaves its
tombstone unreachable, unless a frozen node whose child is not yet published
stands between the tombstone and the last live word before it, which,
uncontended, only an earlier split that threw can leave behind.

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
- a writer walk re-hashes the frozen nodes it passes (at most once each; at
  most twice for one that ends a run), to find the child bucket that decides
  whether the node is dead; a lookup's own walk does not;
- every operation takes a handle, and in a release build reads nothing of it:
  what a handle costs is paid when it is taken, refreshed or destroyed — a
  join writes lines that every joining thread writes too, the generation's
  handle count among them — and never on an operation's path;
- the free-list pop and the limbo push carry orders the original's do not
  (acquire on the pop, release on the push), because freed nodes come back
  onto the free lists while the operations run; on x86 they compile to the
  same instructions;
- an allocation that finds its own shard's free list empty tries the other
  shards' before it appends, unless a set-wide hint (one load) says that no
  free list holds a node: the free pass deals what it frees over every shard,
  not back to the shard that retired it;
- every push onto a retired list tests the cadence, on a set constructed
  without one too: the push adds its node count with an instruction that
  returns the old count, the old and new counts are tested against a constant
  mask, and on a set without a cadence the branch to the trigger is never
  taken;
- and `advance()` walks the whole table, on the thread that calls it (the next
  section).

## Handles, generations and reclamation

Reclamation beside the readers is done the way `../LockFreeListRCU` does it,
and this section keeps that README's words wherever the mechanism is the same:
handles that join and leave generations, bags, a free pass that is cumulative
and oldest-first, and a reclamation call that refuses rather than waits — the
list's `reclaim()`. The hash has two such calls: `reclaim()`, the same bump and
free pass under the same name, and `advance()`, which is `reclaim()` plus a
walk of the table; and a set constructed with a cadence does `reclaim()`'s work
by itself. Where the hash differs — it hands out no iterators, its retired
lists are per arena shard, and its collector walks the table — the text says
so.

**Handles.** Every operation runs under a handle, its first argument:
`set.contains(h, key)`, `set.insert(h, key)`, `set.erase(h, key)`.
`set.new_handle()` joins the set's current generation and returns a live
handle; the handle's destructor leaves it. `h.refresh()` moves the handle to
the current generation and releases the old one, and returns whether it moved
at all (false: the handle was current already, and nothing changed). A handle
is move-only: a copy would be a second membership hiding behind an innocent
assignment, so a second handle is a second `new_handle()`. The handle is where
the hash parts company with the list most visibly. The list hands out
iterators, a handle keeps alive the nodes an iterator is parked on, and a
refresh therefore invalidates every iterator obtained under the handle. The
hash hands out nothing: no iterators, no pointers, no references; every
operation returns a `bool`. So a handle protects nothing the caller holds
between two calls — there is nothing to hold — and its only effect while idle
is to pin its generation, and with it every node retired since, against
reuse. Refreshing costs the caller nothing in validity, and between two calls
a handle can be refreshed as often as wanted. The advice is to *refresh
regularly*, not only before advancing: a handle that is never refreshed keeps
memory from being reused for as long as it lives, and one refreshed only just
before an `advance()` or a `reclaim()` pins the very generation that call
closes (the latency paragraph below).

Thread rules: any number of threads may run operations concurrently, each
under its own handle or sharing one — concurrent operations under one handle
are fine. What is exclusive with every other use of a handle is changing its
membership: `refresh()`, moving from it or over it, swapping it, destroying
it. These are exclusive with uses of *that* handle only — not with other
handles, not with `new_handle()`, not with `advance()` or `reclaim()`. A
handle has no
thread affinity: created on one thread, it may be used or destroyed on
another after a happens-before edge (a `std::thread::join()`, a hand-off
under a mutex); no thread id is recorded. Taking a handle and refreshing one
are lock-free and never wait for a reclamation. Debug builds count the
operations in flight under a handle and diagnose a refresh, move, swap or
destruction while that count is not zero, an operation under an empty handle
or under a handle of another set, and a set destroyed with a live handle;
release builds diagnose none of these, and an operation reads nothing of its
handle.

**Generations, `advance()` and `reclaim()`.** Nothing is ever freed unless
some thread calls `advance()` or `reclaim()`, or, on a set constructed with a
cadence (below), an operation's own retirement triggers what `reclaim()`
does: no handle's leaving frees anything, and on a set without a cadence no
operation does either. One reclamation proceeds at a time, under a try-flag
shared by the two calls and the cadence's triggers; a call that finds another
in progress returns `contended` at once, having done nothing, and never waits.
Any thread may call either, with or without a handle. An `advance()` that
proceeds does three things, in order; a `reclaim()` does the first two and
stops. It *bumps*: looks at every shard's
retired list and limbo list, and if all are empty reports `nothing_retired`
and consumes no generation; otherwise it moves what is on them into the
closing generation's *bag* and publishes the next generation — unless the
ring is full, when it first tries to free the oldest bag and, if that bag is
still pinned, reports `ring_full`, having consumed nothing. Then it *frees*:
bags oldest-first, cumulatively, for as long as the oldest sealed generation
has no live handle — a bag is never freed while an older generation is still
pinned, because a handle of generation *g* can observe nodes that land in
bags *g*, *g* + 1, … and is protected only if every one of them waits for it.
Every call that is not `contended` runs this pass, so a `nothing_retired` or
`ring_full` call can still free something: whatever a handle unpinned since
the last call. Then it *walks*: every published bucket, like a writer's miss
walk that inserts nothing, bypassing the dead runs it passes with one CAS
each, never retried, and retiring what it bypasses into the generation that
is current after the bump. Every `advance()` but `ring_full` and `contended`
walks, a `nothing_retired` one included. The result reports which of the four
outcomes it was and what the call freed, walked and retired; a `reclaim()`
reports the same four outcomes, decided the same way, with the walk's counts
zero. The refusals are the list's three and so is the remedy: call again
later; there is no automatic recovery. The walk is the hash's own. The list's
`reclaim()` walks nothing — a marked node whose unlink lost a race stays
linked until a later erase over the same edge helps it out — and neither does
the hash's: a dead run that a writer walk failed to bypass stays reachable
across any number of `reclaim()` calls, until a writer walk, a split's
cleanup pass or an `advance()` passes it. Of the two calls, `advance()` alone
collects stragglers.

A thread that holds a live handle and calls `advance()` or `reclaim()` frees
nothing at or after that handle's generation: its own handle pins it. A
handle refreshed right before the call is at the current generation, which is
the one the call closes, so it pins the closing bag, and what that bag holds
waits for the next call. To free in one call, release the handle before it.

What a call costs, in kind: the walk is the price, paid by the calling thread
on every `advance()` that is not `ring_full` or `contended` — a
`nothing_retired` one pays for a whole-table walk — and it grows with the
published buckets and the nodes in them; the free pass pushes every node it
frees, dealing them round-robin over the shards' free lists, so a large bag
touches every shard's; the bump is at most two exchanges per arena shard, one
store and a counter increment. A `reclaim()` pays the bump and the free pass
alone: nothing in it grows with the table, so a `nothing_retired` `reclaim()`
is cheap. `advance()` is meant to be called rarely; a caller who wants memory
back often and the walk rarely calls `reclaim()` often and `advance()` rarely,
or lets the set do `reclaim()`'s work for it (the cadence, below). What they
cost in numbers is under [Performance](#performance).

**Two guarantees, kept apart.** *Made unreachable* — a dead node leaving
its chain — is best effort: one CAS per dead run at every site, `advance()`'s
walk included, and nothing promises that a given dead node ever leaves its
chain. *Made reusable* — its memory coming back — is a guarantee: a node that
has left its chain (retired) or never entered one (limbo) is freed when no
handle of its bag's generation or an older one is live, at an `advance()` or
`reclaim()` call or at a trigger of the cadence, and never otherwise; a node
that is freed is never one a live handle can observe.

Latency: a node retired between two calls (`advance()` or `reclaim()`, in any
mix) is freed by the next call if every handle was released before it, and by
the *second* call after its retirement under a long-lived handle refreshed
before each call (the first call closes the bag the refreshed handle still
pins; the second frees it). A straggler that the walk of one `advance()`
bypasses is retired into the generation that is current after that call's
bump, so it waits for the following call either way: with every handle
released, that call frees it (two clicks); under a handle refreshed before
each call, the one after that. Nothing is freed earlier than the next bump
after its retirement, whichever call or trigger makes it.

No eventual bound, for either guarantee: a dead run may stay reachable through
any number of calls — a lost CAS every time, or a run behind a frozen node
whose child nobody ever touches; a handle holder that stalls (preempted,
suspended, or simply never refreshing) pins its generation and every node
retired since for as long as it stalls, the memory under it grows without
bound, and after `kGenerations - 1` bumps every call or trigger with nodes to
bag finds `ring_full` while retired nodes pile up on the retired lists. The set
cannot tell a stalled holder from a slow one. That is the trade of epoch-based
reclamation, accepted here, and the mitigations are inside the design: the
refresh discipline, `ring_full` as a visible signal, and the diagnostics — the
handle count and the bagged count per generation. A thread stalled inside
`advance()`, `reclaim()` or a trigger stalls all reclamation too, every other
call `contended` and every other trigger skipped until it resumes; it never
stalls another thread's operation.

**The cadence.** A set constructed with a cadence reclaims by itself. The
cadence is opt-in, through the constructor's `Options` aggregate —
`Set::Options{.auto_reclaim_every = R}`, with `initial_capacity` and
`arena_shards` beside it, the two parameters of the other constructor — and
is off by default, so a set constructed any other way behaves exactly as if
the cadence did not exist, and nothing on it is ever freed unless a thread
calls `advance()` or `reclaim()`. R counts nodes retired per arena shard: not
runs, not set-wide, not per thread (two threads retiring into one shard share
its count). It is rounded up to a power of two, except 0, the default, which
is off; a value above 2⁶³ is clamped to 2⁶³, silently, which is "never" on
any real run, so `SIZE_MAX` is a valid spelling of "never". On a cadenced
set every push onto a retired list — the winner of an unlink CAS retiring a
run — moves its shard's retired-node count from *old* to *old* + *n*, *n* the
run's length, and when (*old*, *old* + *n*] contains a multiple of R the push
*triggers*, once, however many multiples the run jumps: right after the push,
on the retiring thread, inside the operation that retired, under that
operation's handle, the set attempts the try-flag and, holding it, does what
`reclaim()` does — the bump and the free pass — and never the walk. The walk
is never automatic; it runs only when a caller calls `advance()`. A trigger
that finds the flag held — by an `advance()` or `reclaim()` on any thread, by
another thread's trigger, or by this thread's own `advance()`, whose walk is
retiring — is *skipped*: it does nothing, and nothing re-fires for it; the
next attempt is at the next multiple, R nodes on, because the multiples are
absolute positions of the shard's counter, never reset.

What the cadence guarantees is therefore a bound on spacing, not a count: at
most one attempt per R nodes retired on a shard, and no promise about how many
attempts proceed. Under contention the skipped attempts can far outnumber the
ones that proceed, and the set promises no ratio between them; under a flag
held nearly continuously, by callers or by triggers, every attempt can be
skipped. A caller who expects a count reads the counters (`auto_reclaims`, the
triggers that acquired the flag, and `auto_skipped`, the ones that did not).
The run whose push crossed the multiple is in the bag the trigger closes when
the trigger's bump reports `advanced` (the push and the exchange are on one
thread, in that order); when it reports `nothing_retired`, a concurrent bump of
any cause took the lists between this thread's push and its flag acquisition,
and the run is in that bump's bag; when it reports `ring_full`, or the trigger
was skipped, the run stays on the retired list for the next bump of any cause.
With the cadence on, an `advance()` or `reclaim()` called while operations run
may return `contended` where it could not before — an operating thread's
trigger may hold the flag — and at a quiescent point it never does, since a
trigger lives only inside an operation. A lookup can reach a trigger only one
way: by entering an uninitialized bucket, winning its split and retiring the
dead runs that split's cleanup pass bypasses; such a bucket exists only after a
doubling and only until a split publishes it, so on a cadenced set whose table
has not doubled no lookup ever bumps or frees, and one that does is a lookup on
a table that has doubled, of a key whose bucket had not been split since.

Latency under the cadence is one reclamation longer than under explicit calls
with every handle released, because the trigger runs under a live handle, the
retiring thread's own: a node retired by a thread is freed only by a free pass
*after* the bump that bagged it, never by that bump's own, and only once every
handle of that bag's generation, the retiring thread's included, has refreshed
or left. A thread that erases under a never-refreshed handle pins its own
garbage: every trigger it runs closes a bag it then pins, and after
`kGenerations - 1` such bumps every further trigger that proceeds finds
`ring_full` and frees nothing, while the retired lists, and the memory under
them, grow without bound — until the handle refreshes or leaves, when the next
trigger frees every waiting bag. The same happens, more slowly, when every
handle merely lags: the ring fills behind the oldest of them, the triggers that
proceed turn into `ring_full` refusals, and memory grows until the handles
refresh. The cadence makes the refresh discipline more important, not less.
What a trigger adds to the operation that retired is work, never a wait: the
bump's loads and exchanges, and the free pass of every unpinned bag, bounded by
the bagged nodes — unbounded in time only in that the first trigger after a
long stall ends frees everything the stall pinned — with no lock, no
allocation, no call of `Hash` or `operator==`, and no exception; the operation
then returns what it would have returned.

On a cadenced set the diagnostics that read the generation ring — the current
generation, a generation's bagged count, the ring's state — need no operation
in flight on any thread, in addition to no reclamation in progress and every
earlier reclamation ordered before the call: an operation may be holding the
try-flag. On a set without a cadence they need only the latter; the counters
need neither.

**Values.** A freed node keeps its old, constructed `T`; no reclamation,
`advance()`'s, `reclaim()`'s or a trigger's, destroys anything. Reuse
copy-assigns the new value over the old one, while other
operations run — safe because the node is freed only after every handle that
could observe it has left or refreshed, and it is popped by one thread while
on no chain. An erased value's resources are therefore released at reuse,
the assignment's left-hand side, or when the value is destroyed with the set;
never at `erase()`, never at a reclamation. Callers must not rely on
destructor timing. This is the other way round from the list, which destroys
an erased value inside its `reclaim()` and constructs the next one in its
place at reuse;
here `T` must be copy-assignable, no `T` is destroyed before the set is, and a
copy assignment that throws at reuse sends the node to limbo, to be bagged
and freed by a later call, while the exception leaves the operation as any
other throw does.

**Template parameters.** Two more, after the original's `T`, `AllowDelete`
and `Hash`, so that every existing instantiation means what it meant:
`kGenerations`, the number of generations in the ring, 64 by default and at
least 2, since one slot is always the current generation; and `Hooks`, last,
the test seams — a type with three static `noexcept` functions that the set
calls at fixed points of the join and of the bump, whichever call or trigger
makes it, whose default,
`concurrent_hash_rcu_detail::NoHooks`, compiles to nothing. The hooks are
not part of the contract. The ring is part of the set object: `kGenerations`
blocks, each with its handle count on a cache line of its own, and each with
a bag of two words per arena shard, which the constructor allocates, so the
constructor can throw before any operation exists.

## What is not here yet

Possible next steps, none of them started and none of them promised:

- A cheaper walk. Every `advance()` that walks at all walks the whole table;
  it could instead walk a bounded slice per call, follow a hint of the buckets
  worth walking, or split the uninitialized buckets it skips, so that a dead
  run waiting behind a frozen node for that node's child to be published
  could go too. The walk does not split because a split allocates and may
  take the arena shard's lock, and the walk does neither.
- A `contains()` that also bypasses the dead runs it passes. It would make
  every reader a writer, which is why it was not done here; whether it pays
  is a measurement, and this version is the baseline it would be measured
  against.

## Performance

Two sets of measurements. The first compares the unlinking against the
original, on four machines; it was taken on the copy without handles, with a
quiescent `reclaim()`, since removed, in the place of `advance()` (not this
version's `reclaim()`, which is `advance()` without the walk). The second, on
the laptop alone, measures what handles and `advance()` add against that same
build, and finds the operations under a long-lived handle within noise of it
on all but a few rows, so on the laptop the first set describes this code as
well; the servers have not been measured with handles.

### Unlinking, against the original

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
- For lookups over a set with many deletions (half the keys erased, nothing
  reclaimed) the copy is 1.1-1.5x faster, on every machine at every thread
  count.
- For insert-heavy growth (every key new, the table doubling through the run)
  the speed is within about 20% on x86; on the M3 Ultra the copy is 11-31%
  slower.
- For inserts into a pre-sized set the copy is 3-13% slower on x86 (up to 5%
  faster on the EPYC at high thread counts) and 1-18% slower on the M3 Ultra;
  the 2-socket Xeon is up to 28% slower at high thread counts.
- For erase-heavy workloads the original is 1.6-2.3x as fast (three CASes per
  erase against one) on the laptop, on the EPYC, and on the Xeon at low
  thread counts, and 1.4x on the M3 Ultra at low thread counts; it gets worse
  with threads on the big machines, up to 4.4x on the 2-socket Xeon and 2.6x
  on the M3 Ultra.
- For inserts into a set with many deletions (half the keys erased, nothing
  reclaimed) the copy is 5-22% faster on the laptop, on the M3 Ultra and on
  the EPYC at low thread counts, but 16-43% slower on the 2-socket Xeon at 16
  and 32 threads; elsewhere on the two servers it is within 10%
  either way where the rounds agree at all.
- A delete-free instantiation (`AllowDelete == false`), where the original
  skips the freeze and the copy does not: growth inserts are up to 40% slower
  than the original's delete-free instantiation, and the freeze itself costs
  the original up to 37% (on the EPYC). Pre-sized inserts built with gcc 16.2
  come out up to 27% slower than the copy's deletable instantiation from the
  same source; clang-22 does not show that difference.

No cell of this set mixes inserts, lookups and erases, so there is no
steady-state churn number in it: each line measures one kind of operation at
a time, two of them on a state that erases built.

### Handles and `advance()`

Measured on the laptop alone (clang-22, `-O3 -march=native`, 16 arena shards,
threads unpinned), with the same fixtures as above plus rows of their own.

- **The handle.** Under a long-lived handle, and with the steal (next) set
  aside, the operations run within noise of the build without handles on 78
  of the 85 rows the two share, and every lookup row is at parity.
- **The steal** is free when nothing is free: where no free list holds a
  node, the set-wide hint skips it. Where it does reuse other shards' nodes,
  it trades throughput for memory: under steady churn (every thread erasing
  its oldest key, inserting the next and looking two up, one thread calling
  `advance()` on a cadence) at one to four threads, the arena holds 5-9 slots
  per live key against 52-61 without the steal, at 0.59-0.87 of the rate.
- **The walk**, timed alone on one thread: about 1 ns per bucket plus 2-10 ns
  per node while the table is in cache, and about 40 ns per node beyond it.
  One walk of 2²³ buckets holding 2²² keys takes 178-194 ms, and the same
  table empty 7.4-8.3 ms, which a `nothing_retired` call pays too (each range
  spans two runs a day apart).
- **Erase under reclamation**, on a small table (2¹⁶ keys in 2¹⁷ buckets),
  every thread refreshing its handle before each erase and one thread
  calling `advance()` four times in the run, after each quarter of its own
  erases: 53 M erases/s at 8 threads and 61 M/s at 16, against 126 and 156
  M/s for the same erases without reclamation. At these thread counts the
  walk takes 0.21 and 0.16 of the threads' time; at fewer threads the one
  walking thread's share is larger.
- **A handle per operation** instead of one per thread: lookups at 0.80 of
  the long-lived handle's rate at one thread and at 0.22 at sixteen, because
  every join and every leave writes lines that all joining threads share.

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
  dead nodes leave their chains, what that guarantees, and what it does not,
  and how handles and generations decide when their memory comes back
- `concurrent_deque.h` — the backing store (see `../ConcurrentDeque`)
- `spinlock.h` — the lock the arena's deques append under, and the resize
  lock (see `../Spinlock`)
- `concurrent_hash_set_rcu_test.C` — unit tests (built with ASan and TSan)
- `concurrent_hash_set_rcu_tso_test.C` — the store-buffering regression test;
  it can only fail when built `-O3` without a sanitizer, so it is its own
  binary
- `concurrent_hash_set_rcu_bm.C` — the benchmarks: the original's fixtures,
  so that the two sets' rows pair up, plus the cells the original lacks: two
  whose state erases built, one that times `erase()` itself, a probe
  (`InsertHotGrowth`) that splits long, hot chains while threads insert into
  them, and counts the unlink CASes the splits' cleanup loses and the dead
  nodes left in their chains; the rows that call `advance()` while they run
  (the timed erases on two table sizes, and steady churn, each beside its
  control without reclamation), the small-table erases and the churn on a
  set that reclaims by itself through its cadence (`_Auto`), each beside a row
  where every thread calls `advance()` itself, at the cadence the `_Auto`
  row's set is constructed with (`_RcuEach`), lookups under a handle taken
  per operation,
  and the `Advance_` rows, which time `advance()` alone, one thread, nothing
  else running: the walk per bucket and per node, the free pass per freed
  node
