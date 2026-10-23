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
  join and a leave write the count row of the joining thread's arena shard
  (the leave from whatever thread runs it), and no line that the threads of
  other shards write for handles of their own — and never on an operation's
  path;
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
handle has no thread affinity, but it has a hand-off rule: created on one
thread, it may be used on another, and its membership may be changed —
destroyed, refreshed, assigned over, moved from, swapped — on another thread
than the one that used it last, only after a *synchronizing* hand-off between
the two threads (a `std::thread::join()`, a hand-off under a mutex, a release
store read by an acquire load: any happens-before edge). Every use of the
handle, its creation included, must happen-before the call that ends or moves
its membership, as its creation must happen-before its first use elsewhere.
Not overlapping in time is not enough: the leave orders only the accesses that
happen before it, so after a hand-off that does not synchronize, the free pass
can reuse a node a reader on the other thread is still reading. The handle
records the per-shard count its join incremented, never a thread id, so a
leave on any thread decrements that count. Taking a handle and refreshing one
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
published buckets and the nodes in them; the free pass reads one handle count
per arena shard of each bag it examines (a bag that stays pinned costs one
read per pass while the count that stopped the last pass is still above zero:
the pass resumes at that count and reads on only once it has dropped) and pushes
every node it frees, dealing them round-robin over the shards' free lists, so
a large bag touches every shard's; the bump is at most two exchanges per arena
shard, one store and a counter increment. A `reclaim()` pays the bump and the
free pass alone: nothing in it grows with the table, so a `nothing_retired`
`reclaim()` is cheap. `advance()` is meant to be called rarely; a caller who
wants memory back often and the walk rarely calls `reclaim()` often and
`advance()` rarely, or lets the set do `reclaim()`'s work for it (the cadence,
below). What they cost in numbers is under [Performance](#performance).

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
live-handle count per generation, which is a sum over the arena shards' counts
(exact at a quiescent point; with joins or leaves running, each shard's count
is one it had, but their sum is no one instant's), and the bagged count per
generation. A thread stalled inside `advance()`, `reclaim()` or a trigger
stalls all reclamation too, every other call `contended` and every other
trigger skipped until it resumes; it never stalls another thread's operation.

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
blocks of 128 bytes, each with a bag of two words per arena shard. The
live-handle counts are not in the blocks but in one *count row* per arena
shard — the shard's count of every generation slot, then its two join
counters, rounded up to 128 bytes — so that a join and a leave write only
the row of the joining thread's shard; the free pass reads one count per
shard of each bag it examines. The bags and the rows are allocated by the
constructor, so the constructor can throw before any operation exists.
`arena_shards` therefore sets more than the arena's sharding: it is how far
the handle counts are spread — threads whose numbers are congruent modulo the
shard count share a row — and how many counts a free pass reads per bag. One
shard does not restore the costs of a single count per generation: the join
and the leave keep the per-shard code, which finds the count through the
thread's row, and the check of a pinned oldest bag then reads a count on the
one row every joiner writes.

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

Three sets of measurements. The first compares the unlinking against the
original, on four machines; it was taken on the copy without handles, with a
quiescent `reclaim()`, since removed, in the place of `advance()` (not this
version's `reclaim()`, which is `advance()` without the walk). The second
measures what handles and `advance()` add against that same build: on the
laptop it finds the operations under a long-lived handle within noise of it on
all but a few rows, so there the first set describes this code as well; on the
servers — the two x86 servers and a fifth machine, a 2-socket Grace (144
Neoverse V2 cores, aarch64) — it finds the lookups within noise or up to 15%
either way, the inserts moving both ways, and the rows where the steal
reuses other shards' nodes far below. The third measures what `reclaim()` and
the cadence add to the build with handles, and the cadence against explicit
calls, on the laptop and on the three servers. The live-handle count per
generation is kept per arena shard (the count rows under "Template parameters"
above); the second set was measured with that layout on the laptop, on the
EPYC and on the Grace, and with one count per generation on the Xeon, whose
figures are marked so where the two layouts differ; the third set's figures
were taken with one count per generation, except where a sentence gives both
layouts' figures, and the second set says how the same rows run with the count
per shard. The M3 Ultra was not measured with handles.

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

Measured on the laptop (clang-22, `-O3 -march=native`, 16 arena shards,
threads unpinned), with the same fixtures as above plus rows of their own;
the last bullet is the servers', and the one before it names the servers
where it leaves the laptop.

- **The handle.** Under a long-lived handle, and with the steal (next) set
  aside, the operations run within noise of the build without handles on 78
  of the 85 rows the two share, and every lookup row is at parity.
- **The steal** is free when nothing is free: where no free list holds a
  node, the set-wide hint skips it. Where it does reuse other shards' nodes,
  it trades throughput for memory: under steady churn (every thread erasing
  its oldest key, inserting the next and looking two up, one thread calling
  `advance()` after every 8192 of its erases) at one to four threads, the
  arena holds 5-9 slots per live key against 52-61 without the steal, at
  0.59-0.87 of the rate.
- **The walk**, timed alone on one thread: about 1 ns per bucket plus 2-10 ns
  per node while the table is in cache, and about 40 ns per node beyond it.
  One walk of 2²³ buckets holding 2²² keys takes 178-211 ms, and the same
  table empty 7.4-8.3 ms, which a `nothing_retired` `advance()` pays too and
  a `reclaim()` never does (each range spans three runs).
- **Erase under reclamation**, on a small table (2¹⁶ keys in 2¹⁷ buckets),
  every thread refreshing its handle before each erase and one thread
  calling `advance()` four times in the run, after each quarter of its own
  erases: 53 M erases/s at 8 threads and 61 M/s at 16, against 126 and 156
  M/s for the same erases without reclamation. At these thread counts the
  walk takes 0.21 and 0.16 of the threads' time; at fewer threads the one
  walking thread's share is larger.
- **A handle per operation** instead of one per thread: lookups at 0.74 of
  the long-lived handle's rate at one thread and 0.72 at sixteen (146 against
  44 M lookups/s at sixteen for the same row with one count per generation).
  A join and a leave write the count row of the joining thread's arena shard
  and nothing another shard's threads write for handles of their own, so with
  as many shards as threads the per-operation handle scales as the long-lived
  one does; with one count per generation, which every joining thread writes,
  the same row ran at 0.85 of the long-lived handle's rate at one thread and
  0.22 at sixteen. The per-shard count has a one-thread price: a join and a
  leave execute 31 instructions against 18 (the shard's row address replaces
  one field of the generation block), and the per-operation lookup row itself,
  measured against the same row with one count, runs at 0.91 at one thread on
  the laptop, 0.79 on the EPYC and 0.72 on the Grace (the two ratios to the
  long-lived handle above would give 0.88 on the laptop: the long-lived row's
  own rate moved 2-3% between the two builds' runs, on identical code). Of the
  rows measured with both layouts — every operation row and every reclaiming
  row on the laptop, the reclaiming rows and the long-lived lookups on the EPYC
  and on the Grace — most run within noise of the same code with one count per
  generation. On the laptop two cells moved beyond it, growth inserts into a
  delete-free set at one thread (0.93) and lookups over a reclaimed set at
  sixteen (0.94); on the Grace the long-lived lookups read 0.91 at one thread
  and 0.77 at 144, with the one-count build's own runs spread as far. Under
  gcc the compiler laid out some rows' code differently in the two builds, and
  those rows moved both ways with nothing attributed to the layout: on the
  EPYC the small-table erases under reclamation 1.1-1.3x at 16-128 threads,
  the churn row where every thread calls `advance()` 1.8x at 128; on the Grace
  the pre-sized erases 0.67-1.15 at 2-32 (on the EPYC the churn row without
  reclamation read 0.91 at 128 on identical code: a cell that moved between
  two runs of the same binary). One row whose timed body the compiler did not
  change is slower, though the `advance()` it calls out of line moved: on the
  EPYC the pre-sized erases followed by one `advance()` per thread
  (`Erase_Presized_Del_Rcu`) run 3-5% slower at 1-32 threads with the count
  per shard, within noise at 64 and 128.
- **On the servers** (the 2-socket Xeon, the EPYC and the Grace, gcc 16.2, `-O3
  -march=native`, 256, 128 and 256 arena shards, threads unpinned, from one
  thread to the machine's), the handle and the generation protocol against the
  build without them, the steal set aside. On the Xeon, with one handle count
  per generation, the lookups are within noise of the build without handles,
  or up to 15% faster, on every row; the inserts into a deletable set are
  2-24% slower wherever they leave the noise (8-10% on the plain pre-sized row
  at 1-8 threads, up to 24% over a resident population), except the growth
  row, 17-18% faster at 16-32 threads; into a delete-free set the pre-sized
  inserts are 1.5-1.7x faster at 16-32 and the growth inserts, where they
  leave the noise, 4-21% faster at 16-64 and 6% slower at 256. On the EPYC,
  with the count per shard, the lookups are within noise but on the deletable
  set, 10-15% slower at 1-2 threads and 10-15% faster at 16-64; the growth
  inserts into a deletable set are 6-12% slower where they leave the noise,
  the pre-sized ones 5-25% slower at 4-16 and 128 threads, the delete-free
  rows within noise but for 2-3% slower at 1 and 4 threads on the growth row;
  the `InsertHotGrowth` probe is 8-18% slower at 4 and 32 threads on the
  delete-free set and 8-24% faster at 4-16 on the deletable one. On the Grace,
  with the count per shard, every lookup row is within noise or 1-4% either
  way; the inserts into a deletable set are 9-29% slower at 1-16 threads,
  1-11% faster at 32-64 and 10-18% slower at 128-144, the `InsertHotGrowth`
  probe within noise on every cell; into a delete-free set the growth inserts
  are 3% faster at 16-32 and the pre-sized ones 4-15% faster at 1-64 where
  they leave the noise and 6.5% slower at 144. The fixtures' control rows (the
  same populations built without churn) and the inserts over a set erased but
  not reclaimed, which the sentences above leave out, lie at 0.86-1.23 of the
  build without handles on the EPYC where they leave the noise (the control
  lookups 10-23% faster at 1-8 threads) and at 0.64-0.94 on the Grace. Where
  the steal reuses other shards' nodes (inserts into a set half of whose keys
  were erased and reclaimed) the build with handles runs at 0.01-0.2 of the
  build without them from 4 threads up to, not including, each x86 machine's
  full count, at which it is at 0.74-0.94 (and at 0.25-0.71 at 1-2 threads on
  the EPYC), and at 0.025-0.33 from 4 threads to 144 on the Grace (0.67-0.76
  at 1-2); lookups over that set run at 0.04-0.67 of the other build's at
  16-128 threads on the Xeon, 1.18 at 8, 0.84 at 32 and 0.56 at 64 on the EPYC
  and 0.16-0.85 at 16-144 on the Grace. A handle per operation, with the count
  per shard: on the EPYC 0.52-0.80 of the long-lived handle's lookup rate from
  16 threads to 128, on the Grace 0.39-0.70 from 16 to 144, where one count
  per generation gave 0.025 and 0.007 at the two machines' full counts; on the
  Xeon, with one count per generation, 0.76 at one thread (24.7 against 32.7
  M/s) and 0.002 at 256 (4.8 against 2035 M/s).

### `reclaim()` and the cadence

Measured on the laptop (clang-22, `-O3 -march=native`, 16 arena shards, threads
unpinned) and on the three servers (gcc 16.2, `-O3 -march=native`; 256, 128 and
256 shards on the Xeon, the EPYC and the Grace; threads unpinned, from one
thread to the machine's). The cadence's test is measured on the laptop against
the build with handles and `advance()` but neither `reclaim()` nor the cadence,
and on the servers against this build with the test removed; the cadence is
measured against this build's own rows that reclaim by explicit calls or not at
all. Unless a sentence names a machine, it is the laptop's. The servers'
figures were taken with one handle count per generation; the EPYC's and the
Grace's rows run again with the count per shard are within noise of them, or
moved as "A handle per operation" above says.

- **The cadence off.** Every push onto a retired list tests the cadence whether
  the set has one or not (under clang four instructions per retired run on x86;
  the list under "What differs" says which). On the laptop the operations run
  within noise of the build without the test on 75 of the 85 rows the two
  share; of the ten others nine straddle parity, and one, the `InsertHotGrowth`
  probe at 8 threads, runs 12% slower by the wall-clock rate and within noise
  by the threads' mean loop time. Under gcc 16.2 the test costs the pre-sized
  erase row 7-11% at one thread on all three servers, 9-10% at 4-8 threads on
  the Xeon and 6-11% at 2-64 threads on the Grace; on the EPYC at 2-16 threads
  that row runs 8-13% slower with it, inside the machine's run-to-run noise.
  The small-table erase row pays 2% at one thread on the EPYC and runs 5-14%
  slower there at 4-128 threads and 6% slower at 2-8 on the Xeon, inside the
  noise; on the Grace it is within noise. The insert rows on a set half of
  whose keys were erased before the run, their nodes still retired or already
  freed for reuse, measured at 1-8 threads, run within noise of the build
  without the test on the EPYC, within noise or 2% faster on the Grace, and
  4-8% slower at one and two threads on the Xeon.
- **A `reclaim()` that finds nothing retired** pays the bump and an empty free
  pass, timed on one thread from the `advance()` rows (the bump and the free
  pass are the same code in both), with one handle count per generation and
  with the count per shard alike. On an empty table: about 30-650 ns for the
  bump with the generation ring in cache (the larger figures on the rows with
  bigger tables, where an `advance()` walks the whole table between two timed
  bumps; the bump's own work does not grow with the table) and 20-40 ns for the
  pass while the walk between two bumps leaves the ring in cache (up to 2²²
  buckets). After a walk over a populated table of 2²⁰ buckets and more the
  pass costs 40-95 ns with one count and 55-110 ns with the count per shard;
  in the per-shard build that is one miss on the ring block's line the walk
  evicted, in the one-count build the cause was not traced. With the ring
  cold, 0.8-1.3 µs and 60-160 ns. It never walks, so the 8 ms an empty
  2²³-bucket walk costs `advance()` is not in it.
- **The automatic cadence, erasing a small table.** On the small table above
  (2¹⁶ keys in 2¹⁷ buckets, every thread refreshing its handle before each
  erase), a set that reclaims by itself every 16384/T nodes retired per shard,
  the number at which the one advancing thread above calls, runs at 97, 126 and
  129 M erases/s at 4, 8 and 16 threads. In the same run (the figures of "Erase
  under reclamation" above are from an earlier build, measured in another run)
  the same erases ran at 112, 147 and 167 M/s without reclamation and at 46, 56
  and 64 with the one advancer: the set reclaiming by itself runs at 0.87, 0.86
  and 0.77 of the first and 2.0-2.3x the second, and at 2.3-2.6x a row where
  every thread calls `advance()` after every 16384/T of its own erases (39, 49
  and 56 M/s). Of 16, 32 and 63 triggers per run (medians; at most 4T), 11, 16
  and 28 proceeded and the others found the flag held; the bump and the free
  passes took 7-8% of the threads' time.
- **The automatic cadence under churn.** Under the steady churn above (every
  thread erasing its oldest key, inserting the next and looking two up; 2¹⁸
  erases per thread), a set that reclaims by itself every 8192 nodes retired
  per shard runs at 76, 91 and 100 M operations/s at 4, 8 and 16 threads,
  against 144, 181 and 141 with no reclamation, 68, 47 and 46 with one thread
  calling `advance()` after every 8192 of its erases, and 76, 83 and 78 with
  every thread calling it that often. The arena held 4.6, 11 and 39 slots per
  live key against 67-68 with no reclamation and 8.8, 29 and 46 with the one
  advancer. Of 127, 256 and 528 triggers per run (medians; about 32T, more
  where the table grows), 68, 27 and 19 proceeded and the others found the flag
  held.
- **What limits it.** The cadence changes who reclaims and when, not how fast
  a node is freed. At 16 threads under churn the free pass ran for 0.9 of the
  run's span and freed about 28 M nodes per second of its own time while the
  sixteen threads retired about 53 M per second between them: the bags grew
  to 0.1-0.4 M nodes, 509 of 528 triggers found the pass still running, and
  the table doubled to 2²¹ as it does with one advancer. At 8 threads the
  pass kept pace (29 M/s freed against 26 M/s retired) and the table stopped
  at 2¹⁹ against the one-advancer row's 2²⁰. A pass lands whole on the thread
  whose trigger won the flag, so at 16 threads one or two threads carry
  most of the passes and finish last.
- **Freeing a node** costs 2-4 ns on the small-table rows, whose bags hold
  2-22 K nodes in arena order, 10-26 ns at 2-4 churning threads and 22-36 ns at
  8-16, whatever the bag size there (8 K to 0.3 M nodes); on one thread a
  shuffled bag of 2²⁰-2²² nodes costs 90-125 ns per node.
- **The servers, at few threads.** On both x86 servers the small-table row
  reads as on the laptop at 1-8 threads: the set reclaiming by itself runs at
  0.72-0.84 of the rate without reclamation and 1.4-2.6x the row where every
  thread calls `advance()` at the cadence's number; on the Grace 0.63-0.81 and
  1.1-2.4x. From 16 threads up on the Grace every reclaiming variant of that
  row, automatic or by calls, runs at 19-26 M erases/s while the row without
  reclamation reaches 378 M/s at 64 threads.
- **The servers, under churn.** At the machine's full thread count (256 on the
  Xeon, 128 on the EPYC) the set reclaiming by itself runs at 3.1-4.5x the
  one-advancer row and 0.33-0.39 of the row without reclamation, with the free
  pass busy for 0.94-0.98 of the span, 87-444 triggers skipped per one that
  proceeds, bags of 0.5-1.0 M nodes, and the table growing as with one
  advancer; the pass freed nodes at 0.3-0.5 of the rate the threads retired
  them (half on the laptop at 16). The Grace at 144 runs at 0.13 of the row
  without reclamation with the pass busy for 0.97 of the span. At the full
  thread count on all three, as on the laptop at 16, one or two threads carry
  0.94-0.95 of the free passes' time and finish last. Between those counts and
  16 threads every churn row with reclamation, automatic or by calls, runs at
  0.01-0.13 of the row without it at the same thread count (the reclaiming rows
  at 4.6-29 M operations/s, the rows without reclamation at 125-460), the
  automatic cadence within noise of the per-thread callers on the x86 servers,
  the free pass busy for only 0.15-0.6 of the span below 128 threads, and the
  table at 2¹⁷-2²¹ against the one-advancer row's 2¹⁹-2²³ at 16-64 threads. The
  collapse ends at the thread count equal to the arena shard count on the two
  x86 servers; the Grace, whose 144 threads are below its 256 shards, runs at
  0.07-0.13 at 128-144.
- **Freeing a node on the servers** costs 5-8 ns at one thread under churn,
  30-290 ns at 2 to 128 threads on the Xeon (37-260 ns at 2-64 on the EPYC,
  30-130 ns at 2-144 on the Grace), and 25-35 ns at the machine's full
  thread count on the two x86 servers.

## Building and testing

```sh
make            # benchmark + ASan/TSan unit tests + the -O3 store-buffering test
make tests      # the three test binaries only
make benchmarks # the benchmark only
make run_tests  # run all three test binaries
```

The compiler, C++ standard and library paths come from `../config.mk`, shared
by every directory in this repository and written once per machine; binaries
go to `build/<hostname>/`, the benchmark's per-unit objects beside it.
Requires Google Benchmark and GoogleTest. The
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
- `concurrent_hash_set_rcu_bm.C`, `concurrent_hash_set_rcu_bm_*.C` and
  `concurrent_hash_set_rcu_bm.h` — the benchmarks, one translation unit per
  family of rows over a shared header, linked into one binary (the index at
  the top of `concurrent_hash_set_rcu_bm.C` names which unit holds which rows,
  and why there are several: the code a compiler generates for a row's loop
  then depends on that row's unit alone, not on every other row). The rows:
  the original's fixtures, so that the two sets' rows pair up, plus the cells
  the original lacks: two
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
