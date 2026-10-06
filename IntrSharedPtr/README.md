# IntrSharedPtr

Two atomic, intrusively reference-counted shared pointers with Harris-style
marking, for building lock-free linked structures: `intr_shared_ptr`, which
closes the load/increment gap with a one-bit spinlock inside the pointer word,
and `intr_shared_ptr_hp`, which closes it with a hazard pointer and is
lock-free after a thread's first load. The count lives in the object they
point to — supplied by a base class from this directory or by four hooks you
write yourself — so there is no control block to allocate, and a pointer is
one machine word.

An atomic shared pointer has one problem that a plain shared pointer does
not. To copy the pointer, a thread must read it and then increment the count
of the object it points to — and between the two, another thread may swing the
pointer away and drop the last reference, so that the object dies in the gap
and the increment lands on a corpse. The two pointers here differ, above all,
in how they close that gap; everything else about them — the value type they
hand out, the pointee they require, the mark bit — is the same.

## The contract

The headers' comments are the contract; this section is the map. Every
clause below is stated, with its reasons, in `intr_shared_ptr.h`,
`intr_shared_ptr_hp.h`, `intr_shared_ptr_common.h` (the value type the two
pointers share) and `intr_pointee.h` (what the pointee must provide).

**The atomic.** `intr_shared_ptr<T, U = T>` and `intr_shared_ptr_hp<T, U = T>`
model the `AtomicSharedPtr` concept of
`../SharedPtr/atomic_shared_ptr_concept.h`: `load(order)`, `store(value,
order)`, `compare_exchange_strong(expected, desired, success, failure)`, and
`supports_marking == true`. `T` is the interface type seen through the
pointer, `U` the stored type (`U*` must convert to `T*`); `U` is the dynamic
type of every pointee — the last release runs `delete` on a `U*`, so a type
derived from `U` is destroyed through `U`'s destructor, virtual or not. The
atomics are non-copyable and non-movable, like `std::atomic`. The strong CAS
has `std::atomic` semantics: it compares the full word (pointer and mark),
never fails with `expected` unchanged, leaves `expected` untouched on success
and refreshes it with a live reference on failure. `intr_shared_ptr_hp` also
offers `compare_exchange_weak` (one attempt, may fail spuriously) and
declares `deferred_reclamation = true` and `pointee_base`
(`std::hazard_pointer_obj_base<U>`), the optional policy members the concept
header describes; the spinlock pointer declares none of them.

**The value type.** `shared_ptr_type` (the same class template for both
pointers, in `intr_shared_ptr_common.h`) owns one strong reference and may
carry the mark in bit 0 of its raw pointer. The mark is part of the value: a
marked and an unmarked handle to one object compare unequal, and a marked null
is a value distinct from null. `get()`, `operator->`, `operator*` and
`use_count()` see through the mark; `get_raw()` does not; `is_marked()`,
`get_unmarked()` and `set_mark()` are the marking API, and the rvalue
`get_unmarked()` clears the bit in place so that
`next.load(...).get_unmarked()` costs no count traffic. A handle is
default-constructed null, and `nullptr` converts to it, so `a.store(nullptr)`
clears a link and `h = nullptr` releases a handle's reference; there is no
`reset()` and no `swap()` (move-assign a null, or `std::swap` the handles).
Adopting a raw pointer (`shared_ptr_type(U*)`) takes the object from count 0
to 1 — adoption and sharing are one operation, since the count lives in the
object. Every member is `noexcept`. The assignments are safe when the source
lives inside the pointee being released: `h = h->child` and
`h = std::move(h->child)` are well-defined when `h` is the parent's only
owner, as for `std::shared_ptr` — and likewise when the source lives inside
any other object the release destroys (under `intr_shared_ptr_hp`, an
unrelated pending object reclaimed by a scan the release happens to run): the
source is read before the release.
One handle shared by several threads follows `std::shared_ptr`'s rule:
concurrent const operations on it (copying from it, `get()`, `->`, `*`,
`use_count()`, the marking queries) are race-free; any write to it
(assignment, moving from it, the rvalue `get_unmarked()`, destruction) needs
exclusion from every other access. A pointer that several threads read and
write is what the atomic is for.

**The pointee.** `intr_shared_ptr` requires `IntrusivePointee<U>`: three
`noexcept` hooks, `void AddRef()`, `bool DelRef()` (true iff this call made
the 1 → 0 transition) and `long use_count() const`. `intr_shared_ptr_hp`
requires `HpIntrusivePointee<U>`: those three, a fourth,
`bool TryAddRef()` (increment iff the count is not zero), and public
derivation from `std::hazard_pointer_obj_base<U>`. The hooks' semantics and
memory orders — `AddRef` relaxed, `DelRef` acquire-release, `TryAddRef`'s
acquire loads and acquire CAS — are stated once, normatively, in
`intr_pointee.h`; every operation on the count must be a read-modify-write.
The concepts check what a concept can (names, signatures, `noexcept`); the
orders are a contract you keep. Two bases implement the hooks exactly:
`intr_pointee_base<Count = long>` (all four hooks, for either pointer) and
`intr_pointee_base_hp<U, Count = long>` (the same plus the hazard pointer
base, CRTP on the final type `U`; it must be `U`'s only hazard pointer base).
A copy of a pointee is a new object nobody owns yet: the bases give the copy
count 0 (and, for the HP base, a fresh hazard pointer subobject) and leave the
count alone on assignment. `Count` may be any integral type other than
`bool`, not cv-qualified, with a lock-free `std::atomic<Count>`; a narrower
type saves space at the price of a bound — never more than
`std::numeric_limits<Count>::max()` simultaneous owners of one object, not
checked at run time (`intr_pointee.h` says what happens beyond it). The
pointers check their concept, `U* → T*`, the alignment, and that `~U` does
not throw, by `static_assert` inside their member functions rather than at
class scope, so that a node type can hold a pointer to itself
(`struct Node { intr_shared_ptr<Node> next; }`) while it is still incomplete;
the diagnostic comes at the first use of a member. A pointee whose destructor
is declared `noexcept(false)` is rejected: destructors do not throw, and the
pointers assert it instead of reasoning around it.

**Reclamation.** `intr_shared_ptr` deletes an object at its last release,
synchronously, on the releasing thread — inside a handle's destructor or
assignment, `store()`, a CAS failure's refresh of `expected`, or the atomic's
destructor. `intr_shared_ptr_hp` only *retires* it, to Maged Michael's hazard
pointer domain, and the object is destroyed at a later reclamation scan, in a
batch, by whichever thread's `retire()` crosses the scan threshold — a thread
that may never have touched this pointer. Its header's "Deferred
reclamation" section is the authority; the consequences for your code are in
"Building your own data structure" below.

**Progress.** `intr_shared_ptr` is not lock-free and does not pretend to be:
every operation takes the lock bit, and waiters back off — readers spin,
yield, then sleep; writers sleep almost at once, with an escalation tier for a
lock holder descheduled mid-critical-section. A structure built on it can
still be lock-free in the algorithmic sense, because the lock is held only
across a read and a count increment. `intr_shared_ptr_hp` is lock-free
exactly: `load()`, `store()`, both CASes and every operation on the value type
are lock-free algorithms, with one exception inside `mm_hp` — a thread's first
`load()` takes a hazard record from a global pool under a one-bit spinlock
(and a thread's exit returns its records under it). After a thread's first
`load()`, everything it does with this pointer is lock-free — until the
thread is running its thread-local destructors, when `mm_hp` has closed its
record cache and loads go to the pool again (the header's overview,
"Lock-freedom, exactly").

## Design principles

- The count is in the object. There is no control block, no allocation per
  pointer, and a pointer is one word, which is what makes a CAS on it a CAS on
  the whole value — pointer and mark together.
- The mark is part of the pointer's identity. A Harris-style structure
  marks a node's own `next` to delete it; because the mark travels with the
  value and `operator==` compares it, a marked anchor fails a competitor's
  CAS by itself, with no separate race window to close.
- Counted before published, released after unpublished. Every path that
  puts a reference into an atomic word increments the count first and
  decrements the old occupant's only after the word no longer names it. A
  structure's destructor that judges a node exclusive by `use_count() == 1`
  rests on this (see `~Node()` in `../LockFreeList/lock_free_list.h`).
- The spinlock pointer buys simplicity with a lock bit and asymmetric
  back-off: readers burn a little CPU before they sleep, writers get out of
  the way at once, so that a lock holder is rarely kept from finishing.
- The hazard pointer one never writes the word on a load. It publishes the
  object's address in a hazard record, re-reads the word to validate the
  hazard, increments the count only if it is not zero (`TryAddRef`: an
  object at zero is dead and must never be revived), and re-validates the
  word after the increment, giving the reference back if the word moved.
  Each memory order in that protocol is load-bearing, and the header's
  overview says which failure each one prevents.
- Mark-aware protection, by hand. `mm_hp`'s own `protect()`/`try_protect()`
  compare the whole word and so never accept a legitimately marked one; the
  pointer builds its validation from the public pieces instead. That
  sequence is sound for `mm_hp`'s implementation, and tied to it.
- Reference-count discipline: relaxed increments, acquire-release
  decrements, and for the hazard pointer protocol an acquire zero-observing
  load and an acquire CAS success. The `intr_pointee.h` contract says, for
  each order, what relaxing it breaks, which orders are load-bearing and
  which is a choice.

## The domain of applicability

These are the pointers for linked structures whose links must be swung
atomically and whose nodes must be reclaimed for real: lists, and by
extension trees, skip lists, graphs. `../LockFreeList` is the worked example.
The boundaries, stated plainly:

- The pointee is yours to shape. There are no weak pointers, no custom
  deleters, no allocators, no `make_shared`: you `new` the object and hand it
  to a handle, and the last release deletes it (or retires it). An object
  must be owned through one and the same `U` everywhere.
- `intr_shared_ptr` is a lock. Under contention, readers and writers queue on
  one bit of one word, and a writer waits for every reader that beat it; a
  structure in which one word is hammered from many threads will show that.
- `intr_shared_ptr_hp` defers destruction, and the deferral is not yours to
  control: an object whose count reached zero is destroyed at some later
  scan, on some thread, among roughly a thousand others. A destructor that
  must run promptly, on a particular thread, or that touches thread-local
  state, does not belong under it. Memory stays allocated until the scan,
  and a chain of objects that release each other dies one link per scan.
- `intr_shared_ptr_hp` is Linux-only and `mm_hp`-specific: the hazard
  pointers issue `membarrier(2)`, define `std::hazard_pointer` and
  `std::hazard_pointer_obj_base` themselves (never combine them with a real
  `<hazard_pointer>`), and the pointer's mark-aware validation relies on
  `mm_hp`'s implementation, so a standard `<hazard_pointer>` is not a
  drop-in replacement. Built and tested on x86-64 with clang-22 and gcc-16;
  untested on aarch64.
- Alignment: the pointers steal the low bits of the object's address — bit 0
  for the mark, and bit 1 for the spinlock pointer's lock — so `U` must be at
  least 4-byte aligned for `intr_shared_ptr` and 2-byte aligned for
  `intr_shared_ptr_hp` (its hazard pointer base guarantees 8). Enforced at
  compile time, with the other pointee checks.
- A GCC quirk, not a defect: `g++-16 -O3` without a sanitizer can report
  `-Wstringop-overflow` ("writing 8 bytes into a region of size 0 ...
  destination object is likely at address zero") from
  `intr_shared_ptr_hp::store()` inlined into a caller's CAS loop that stores
  into a freshly adopted node's link (the example test's `push()`). It is a
  false positive — the handle is non-null there, the warning does not appear
  at `-O2`, under a sanitizer, with clang, or in a smaller translation unit,
  and a non-null hint in the handle's `operator->` does not remove it. Build
  such a translation unit with `-Wno-stringop-overflow` under `-Werror`, or
  at `-O2`. The component's own `-O3` builds are not affected.
- The count's width is a bound (above). The default `long` makes it
  academic.

## Building your own data structure

What a structure needs from this directory, in the order you will need it.

**Include and link.** A user of `intr_shared_ptr` alone includes
`intr_shared_ptr.h` with `-I<repository>/IntrSharedPtr` and links nothing:
the pointer, its pointee concept and base (`intr_pointee.h`) and the shared
value type (`intr_shared_ptr_common.h`) are headers. A user of
`intr_shared_ptr_hp` includes `intr_shared_ptr_hp.h` the same way and, in
addition, compiles `mm_hp/mm_hp.cpp` into every program that uses it, with
that program's own flags — optimization and sanitizer included, which is why
the Makefiles here put it on the same command line as the translation unit it
links with — and links with `-pthread`, because `mm_hp` uses threads. (The
Makefiles here pass `-lpthread -lrt -lm`; `mm_hp` needs only the first.)
`hp_drain.h` and `hp_drain_gtest.h` include `mm_hp/mm_hp.hpp` too, so a test
that drains needs `mm_hp/mm_hp.cpp` even if it never names
`intr_shared_ptr_hp`. Linux only.
`mm_hp/` is not in the repository: it is made from Maged Michael's upstream
repository by `make imports` in this directory, which needs the clone
described under "Building and testing".

**The pointee: two routes.** Either derive from the base —

```cpp
#include "intr_shared_ptr.h"

struct Node : intr_pointee_base<> {        // AddRef, DelRef, use_count (and TryAddRef)
    int value;
    intr_shared_ptr<Node> next;             // a node may hold a pointer to its own type
};

using NodePtr = intr_shared_ptr<Node>::shared_ptr_type;
NodePtr n(new Node);                        // count 0 -> 1: n owns the node
```

and, for the hazard pointer variant, from `intr_pointee_base_hp<Node>`
(CRTP: the base's parameter is the final type, and it must be the type's only
hazard pointer base):

```cpp
#include "intr_shared_ptr_hp.h"

struct Node : intr_pointee_base_hp<Node> {  // the four hooks + std::hazard_pointer_obj_base<Node>
    int value;
    intr_shared_ptr_hp<Node> next;
};
```

— or write the hooks yourself, as `Node` in `../LockFreeList/lock_free_list.h`
does: `void AddRef() noexcept`, `bool DelRef() noexcept`,
`long use_count() const noexcept`, and for `intr_shared_ptr_hp` also
`bool TryAddRef() noexcept` plus public derivation from
`std::hazard_pointer_obj_base<Node>`. Copy the orders from `intr_pointee.h`
exactly; `intr_pointee_base::TryAddRef` is the reference form of that hook.
The hook route is for a type that cannot take a base (a node shared by
several pointer policies, as the list's is) or that needs to instrument its
count (the tests do); the base is the default.

**Using the atomic.** `load()` returns a handle that owns a reference;
`store(h)` publishes `h`'s pointee (and mark) and releases the old occupant;
`compare_exchange_strong(expected, desired)` swings the word iff it still
holds `expected`, pointer and mark alike. The marking API is on the handle:
`h.set_mark()` is a copy with the mark set, `h.is_marked()` reads it,
`h.get_unmarked()` clears it. A Harris-style erase marks the victim's own
`next` with a CAS from `next` to `next.set_mark()`, then unlinks it with a CAS
on the predecessor's `next`; `../LockFreeList/lock_free_list.h` shows the
whole dance, including what a reader does when it finds a marked word.

Memory orders differ between the two pointers, and the headers' contracts are
exact; in short: `intr_shared_ptr` accepts any `std::memory_order` on every
operation and is at least acquire-release on every operation regardless (its
lock is taken with an acquire CAS and freed with a release store; a relaxed
`store()` is legal and promoted to release), but `load(seq_cst)` is not a
seq_cst operation — it takes no place in the single total order.
`intr_shared_ptr_hp` follows `std::atomic`'s rules for which orders each
operation accepts (asserted), promotes its publishing read-modify-writes to
at least acquire-release, and honours `load(seq_cst)`; a non-null handle it
returns synchronizes with its word's publisher whatever order was passed
(the word is re-read with acquire inside the protocol); only a null from a
relaxed first read is a plain relaxed read. A template that serves both
pointers may pass the orders `std::atomic` would need; each pointer makes of
them what it must.

CAS loops over these pointers are free of the memory-reuse ABA: `expected` is
a handle obtained from the word (by `load()` or a failed CAS), it owns a
reference, and an object's address cannot be reused while any reference to
it exists — the object is destroyed (or retired, and later reclaimed) only
at count zero — so a CAS with that `expected` can only succeed against the
very object it was taken from, and the comparison is of the whole word,
pointer and mark. What the pointers do not and cannot prevent is the logical
ABA of the same object legitimately re-published, exactly as for
`std::atomic<std::shared_ptr>`; a structure that re-inserts a removed node
while other threads may still hold handles to it must reason about that
itself.

**Removed nodes and their links.** A node removed from a structure still
holds its link to the next node, and that link holds a reference: removed
nodes that are not unlinked form a chain whose nodes release each other from
their destructors — recursively, as deep as the chain, under
`intr_shared_ptr`, and one link per reclamation scan under
`intr_shared_ptr_hp`. Two ways out. The simple one, when no one traverses
removed nodes: after a successful removal, clear the removed node's link
(`node->next.store(nullptr)`). It is safe under both pointers: a slower
remover that still holds a handle to the node reads either the old link or
null, and its CAS on the anchor fails either way because the anchor no longer
holds that node — and the node's address cannot have been reused, since that
remover's handle owns a reference. (Re-inserting the same removed node while
others may hold handles to it brings back the logical ABA of the paragraph
above; that is the structure's problem, as with any shared pointer.) The
other way is for structures whose readers traverse removed nodes — the list's
never-invalidated iterators walk the graveyard of deleted nodes — which must
keep the links and dismantle chains in the destructor:

**A destructor that walks.** A structure's node destructor that dismantles a
chain iteratively (to avoid recursion as deep as the chain) typically judges a
node exclusive when `use_count() == 1` and then tears its `next` down. That
test is sound only because of two guarantees these pointers give: a word's
reference is counted before the word is published and released after it is
unpublished, so count 1 means no word points at the node now or later; and a
loader that takes a reference optimistically (the hazard pointer one)
re-validates the word after its increment and gives the reference back if the
word moved. The comment above `~Node()` in `../LockFreeList/lock_free_list.h`
states this precondition in full, with what goes wrong without it; read it
before writing such a destructor.

**Living with deferred reclamation (`intr_shared_ptr_hp` only).**

- `~U` (hence `~T`, when `U` derives from `T`) runs on whichever thread
  crosses `mm_hp`'s scan threshold (max(1000, 2 × the number of hazard
  records ever allocated)), inside that thread's `retire()` — which may be
  inside a handle's destructor, an assignment, `store()`, a CAS or the
  atomic's destructor, anywhere in the program — or once more at process
  exit. The pointee must not be thread-affine: no thread-local state, no
  assumption about the releasing thread.
- "The count reached zero" and "the object is gone" are different moments.
  `use_count()` of a live handle never reads 0, but memory is reclaimed, and
  destructors run, only at a scan. A scan takes the whole pending list and
  resets the pending count to zero, so right after a scan (a drain, in
  particular) a single retirement does not run another; the next scan comes
  after another threshold's worth of retirements. A thread's exit reclaims
  nothing. Objects retired by destructors running inside a scan wait for the
  next one, so a chain of objects that release each other dies one link per
  scan. A retired object has no owner and nobody may read it: the next scan,
  which any thread's `retire()` may run at any moment, destroys it. The
  exception is a program in which no scan can run — a single-threaded test
  between a release and its drain — where the header says exactly what may be
  done.
- A test that asserts "destroyed" must drain first: `hp_drain.h` provides
  `drain_reclamation()` (retire a sentinel, then fillers until the sentinel's
  destructor runs; returns the number of fillers), `drain_until(pred,
  max_rounds)` for a cascade of depth greater than one, and
  `force_early_scan()`; `hp_drain_gtest.h`, included in a GoogleTest binary,
  runs the early scan once before the first test so that `mm_hp`'s first scan
  (and its `membarrier` registration) happens at the same point in every run.
  The drain's precondition is quiescence — no other thread retiring or
  scanning while it runs — and `hp_drain.h`'s contract comment says what
  happens without it. A drain is for tests and benchmark fixtures, not for
  production code.
- `load()` — and a CAS's failure path, which refreshes `expected` through
  `load()` — can throw `std::bad_alloc` from `mm_hp` on a thread's first call
  (its hazard record), with the counts balanced and nothing of the caller's
  changed; every other operation is nothrow. The header's "Exception safety"
  section states the guarantee exactly, with its one corner case (a thread
  already running its thread-local destructors).

**What TSan can and cannot tell you.** A clean ThreadSanitizer run of a
structure built on `intr_shared_ptr_hp` says nothing about hazard pointer
lifetimes: every protecting `load()` ends with a release store to the
reader's hazard record, which the scan acquires, so an access the protocol
failed to protect is, in TSan's eyes, ordered before the free, and goes
unreported. TSan still checks your code around the pointer, and the count's
orders. The pointer's own protocol is checked by the deterministic seam tests
in this directory, which open each race window on purpose through the
pointee's hooks.

## Maged Michael's hazard pointers (`mm_hp/`)

The hazard pointers under `intr_shared_ptr_hp` and the drain helper are Maged
Michael's `mm_hp`, an implementation of the C++26 hazard pointers
(`[saferecl.hp]`). His code is not in this repository: `mm_hp/` is made from

https://github.com/magedm/mm_hp

at commit `b26e5ed`, together with its README and license files, by
`make imports` here. The code may be used under the MIT license or the Apache
License 2.0 with LLVM exception, at the user's option (see `mm_hp/LICENSES`);
we use it under the MIT license. The README in `mm_hp/` is upstream's and is
left exactly as upstream wrote it; our notes are here.

- **One local change.** `mm_hp.cpp` gets a ThreadSanitizer-only patch
  (`mm_hp.patch`, marked `TSAN-PATCH` in the code): under TSan, the scan
  loads the hazard slots with acquire instead of relaxed. TSan does not model
  `mm_hp`'s asymmetric fence pair (`membarrier(2)` on the scanning side), and
  without the patch it reports false races between a reader and the deleter.
  Builds without TSan are unchanged. The patch's header says what the change
  is for; `mm_hp/SOURCE.txt`, written when the directory is made, records
  what it was made from.
- **Linux only**, with no guards: `asymmetric_fence.hpp` includes
  `<linux/membarrier.h>`, and `mm_hp` aborts at its first reclamation where
  `membarrier(2)` is missing.
- **Never combine with a real `<hazard_pointer>`.** `mm_hp` defines
  `std::hazard_pointer`, `std::hazard_pointer_obj_base` and
  `std::make_hazard_pointer` itself. Nor is a standard `<hazard_pointer>` a
  drop-in replacement once the name clash is resolved: `intr_shared_ptr_hp`
  must protect a pointer whose mark bit may be set, which the standard's
  `protect()` and `try_protect()` cannot do, so it hand-rolls its validation
  from the public pieces; that sequence is sound for `mm_hp`'s implementation
  but not covered by the standard's contract.
- **When deleters run** — on which thread, at which threshold, what a scan
  does to the pending count, and what a thread's exit does not do — is in
  "Living with deferred reclamation" above; it is the user-facing side of
  `mm_hp`'s design, not a change of ours.

## Building and testing

```sh
make imports         # mm_hp/ from the HazardPtr clone (the first build does it too)
make benchmarks      # the microbenchmark of both pointers — needs only Google Benchmark
make                 # benchmarks + the sanitizer test binaries (needs GoogleTest)
make run_tests       # run every test binary: the TSan stress program, the drain
                     # self-test, the seam tests, the pointee contract tests and
                     # the example structure's tests, the GoogleTest ones in ASan
                     # and TSan builds
make run_benchmarks  # run the microbenchmark
```

Binaries land in `build/<hostname>/`, so any number of machines can build and
run concurrently in one shared tree; `make clean` removes this host's
directory only and leaves `mm_hp/` in place. The machine-specific
configuration — compiler, `-march` target, C++ standard, and the Google
Benchmark/GoogleTest install paths — lives in `../config.mk`, shared by every
project directory and written once per machine; the Makefile itself is
machine-independent. The reference `config.mk` uses `clang++-22`,
`-march=native` and C++23; the code also builds with `g++-16`
(`make CXX=g++-16 SAN_CXX=g++-16`). Requires Linux, Google Benchmark and
GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your installations if they
are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`. The microbenchmark also
includes `../Spinlock/gb_wall_clock.h` (through `-I../Spinlock`), the
wall-clock accounting behind its `wall_items_per_second` and `finish_spread`
counters.

Before the first build, clone Maged Michael's repository where the build
looks for it, at the commit our patch was made against:

```sh
mkdir -p ../ThirdParty && cd ../ThirdParty
git clone https://github.com/magedm/mm_hp HazardPtr
git -C HazardPtr checkout b26e5ed
cd ../IntrSharedPtr && make imports
```

The rules are in `../third_party.mk`, shared with `../SharedPtr`, whose
README ("Third-party code") describes them and lists the other clones that
project needs: the content is taken from the pinned commit, never from the
clone's working tree, the patch is applied without fuzz, a clone at another
commit is refused, and on a machine without the clone a `mm_hp/` made
elsewhere and copied in is kept with a warning. `../SharedPtr` and
`../LockFreeList`, which use `intr_shared_ptr_hp`, get `mm_hp/` by asking this
directory's `make imports`; `mm_hp.mk` is the build's one description of what
linking `mm_hp` takes, included by all three Makefiles.

- `intr_shared_ptr.h` — the spinlock intrusive pointer
- `intr_shared_ptr_hp.h` — the hazard pointer intrusive pointer; also
  `HpIntrusivePointee` and `intr_pointee_base_hp`
- `intr_pointee.h` — the pointee side of the contract: the hooks (normative),
  `IntrusivePointee`, `intr_pointee_base`
- `intr_shared_ptr_common.h` — the value type and the helpers the two
  pointers share (an implementation header; include the pointers, not it)
- `mm_hp/` — Maged Michael's hazard pointers, made by `make imports` with
  `mm_hp.patch` (above); `mm_hp.mk` — what linking them takes, for the
  Makefiles here and in the two projects that use them
- `hp_drain.h` — drains `mm_hp`'s pending reclamations on demand, so that a
  test can assert that an object is gone: `mm_hp` has no public flush, and
  without one "destroyed" means "at some later scan"
- `hp_drain_gtest.h` — included by a GoogleTest binary, runs one drain before
  the first test, so that `mm_hp`'s first scan happens at the same point in
  every run
- `hp_drain_selftest.C` — tests of the drain itself (built with ASan and TSan)
- `intr_pointee_test.C` — black-box contract tests of the pointee side: the
  hooks, the two concepts, the bases, and the value type's contract over both
  pointers (built with ASan and TSan)
- `intr_shared_ptr_hp_seam_test.C` — deterministic white-box tests of the
  hazard pointer protocol: hooks in the pointee open each race window on
  purpose, in every run (built with ASan and TSan)
- `intr_shared_ptr_example_test.C` — a user's structure built from this README
  and the headers' contracts alone: a Treiber stack on each pointer, one with
  the base-class route and one with hand-written hooks, with tests of the
  stack and of the lifetime promises made above (every node destroyed exactly
  once; synchronous reclamation under the spinlock pointer; under the hazard
  pointer one, destruction waits for a scan, and a chain of removed nodes
  that keep their links dies one link per drain — which is why its `pop()`
  unlinks). Built with ASan and TSan; the place to start when writing your
  own.
- `intr_shared_ptr_tsan.C` — a standalone ThreadSanitizer stress program,
  typed over both pointers: concurrent readers of one word, and cross-assigning
  writers
- `intr_shared_ptr_mbm.C` — the microbenchmark of both pointers on their own
  operations (load and dereference, copy, assign, cross-assign), with no data
  structure around them; the head-to-head with the other pointers is
  `../SharedPtr/atomic_shared_ptr_bm.C`

## The book

Chapter 7 of *The Art of Writing Efficient Programs, Second Edition* by
Fedor G. Pikus covers an earlier version of `intr_shared_ptr`, as a part of
the lock-free list; the hazard pointer one, the pointee bases and the
concepts are not in the book. The code here has moved on since.
