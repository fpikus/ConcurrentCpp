# SharedPtr

Atomic reference-counted smart pointers brought to one interface and compared
on it: the concepts that `../LockFreeList` is written against, the adapters
that fit other people's pointers to those concepts, and the harness — one
typed test suite and one benchmark — that runs every pointer through the same
operations with no data structure in the way.

Everything in this directory is an adapter or a harness. The pointers
themselves live elsewhere: the two intrusive pointers in `../IntrSharedPtr`
(the released component, with its own README), `std::atomic<std::shared_ptr>`
in the standard library, and Daniel Anderson's `parlay::atomic_shared_ptr` in
his repository, from which the build makes a patched copy.

An atomic shared pointer has one problem that a plain shared pointer does
not. To copy the pointer, a thread must read it and then increment the count
of the object it points to — and between the two, another thread may swing the
pointer away and drop the last reference, so that the object dies in the gap
and the increment lands on a corpse. The four pointers compared here differ,
above all, in how they close that gap.

## The concepts

`atomic_shared_ptr_concept.h` states the contract that the lock-free
containers are written against. `MarkedSharedPtr` is a shared pointer that can
carry a Harris deletion mark in its low bit (`is_marked()`, `get_unmarked()`,
`set_mark()`); `AtomicSharedPtr` is an atomic of such pointers — `load()`,
`store()` and `compare_exchange_strong()` with memory orders, plus a
`supports_marking` flag, `false` for the one pointer that cannot mark. The
concepts are a design reference and an opt-in conformance check
(`static_assert(AtomicSharedPtr<MyPtr<Node>>)`), deliberately not constraints
on the containers. A pointer may also declare a base class its pointee must
derive from (`pointee_base`), that its pointees die late
(`deferred_reclamation`), and a single-attempt `compare_exchange_weak`; the
first two are declared by `intr_shared_ptr_hp` alone, the weak CAS by it and
by the parlay adapter.

## The four pointers: two adapted, two native

Two pointers do not speak the concepts' language natively and are adapted:

1. `StdAtomicSharedPtrAdapter<T>` (in the concept header) adapts
   `std::atomic<std::shared_ptr<T>>`: its `shared_ptr_type` is a
   `std::shared_ptr<T>` with the marking API bolted on, and the atomic's
   `load()`, `store()` and `compare_exchange_strong()` convert between the
   two. It has no room for a mark (`supports_marking` is `false`), which
   forces a weaker deletion algorithm on the list built on it. Not lock-free:
   libstdc++ closes the gap with a lock bit hidden in its control-block
   pointer, and says so (`is_always_lock_free` is `false`).
2. `parlay::atomic_shared_ptr<T>` (`lock_free_shared_ptr/atomic_shared_ptr.hpp`)
   adapts Daniel Anderson's `parlay::shared_ptr` and hazard pointers: our
   class of that name keeps the Harris mark in bit 0 of the control-block
   pointer, and a patch of ours (`lock_free_shared_ptr/parlay.patch`) makes his
   reference counting ignore it and replaces his folly dependency with
   standard equivalents. Genuinely lock-free, with hazard pointers demoted to
   an implementation detail inside the pointer. His code is not in this
   repository: it is built from his repository (see "Third-party code" below
   and [lock_free_shared_ptr/README.md](lock_free_shared_ptr/README.md)).

The other two pointers model the concepts as written and need no adapter;
they are the component in `../IntrSharedPtr` and are included from there
(`-I../IntrSharedPtr`):

3. `intr_shared_ptr` — intrusive: the count lives in the pointee, the Harris
   mark in bit 0 of the pointer word, and a one-bit spinlock in bit 1 guards
   the gap. Not lock-free, and it does not pretend to be; a list built on it
   is still lock-free in the algorithmic sense.
4. `intr_shared_ptr_hp` — the intrusive pointer with the lock taken out: the
   gap is closed by a hazard pointer (Maged Michael's `mm_hp`), and the
   increment is conditional, so a dying object is never revived. Lock-free
   after a thread's first `load()`. The price is deferred reclamation: an
   object whose count reaches zero is retired, not deleted, and destroyed at
   a later `mm_hp` scan by whichever thread triggers it.

Their contract, their pointee requirements and `mm_hp` are documented in
[../IntrSharedPtr/README.md](../IntrSharedPtr/README.md) and in their headers.

## The harness

`atomic_shared_ptr_test.C` is one GoogleTest suite typed over the four
pointers: the same contract tests — loads, stores, compare-exchanges, marks,
counts, reclamation, stress — run on each, with the pointer-specific pointee
each needs and a drain of `mm_hp` (`../IntrSharedPtr/hp_drain.h`) before every
"object gone" assertion, since under the hazard pointer policy "destroyed"
means "at some later scan"; the hazard pointer policy's own clauses have their
tests in the same file. Built with ASan and with TSan.

`atomic_shared_ptr_bm.C` runs the four head to head on one shared pointer
word with a payload behind it: read-heavy and write-heavy mixes, CAS rows,
and a readers-plus-one-writer row driven by a stop flag (so that a slow writer
is measured rather than averaged into its readers' time). The benchmark build
carries event counters — every compare-exchange's outcome, every
destruction, the hazard pointer policy's load retries — whose identities
(destructions against compare-exchange outcomes, for one) cross-check each
row. Rows are suffixed `_StdAtomic`, `_IntrShared`, `_LockFree` (parlay) and
`_IntrSharedHP`. A second build of the same source,
`atomic_shared_ptr_bursts_bm`, adds timing of the hazard pointer policy's
destruction bursts (made on demand, below). No numbers are quoted here: the
harness is one `make run_benchmarks` away from being yours.

## Building and testing

```sh
make                 # benchmark and both sanitizer builds of the unit tests
make run_tests       # run both test binaries
make run_benchmarks  # run the benchmark
make build/$(hostname)/atomic_shared_ptr_bursts_bm  # on demand: + destruction-burst counters
```

Binaries go to `build/<hostname>/`. Requires Linux (every binary links
`mm_hp`, which issues `membarrier(2)`), Google Benchmark and GoogleTest; point
`GBENCH_DIR` and `GTEST_DIR` at your installations if they are not in
`$HOME/GoogleBench` and `$HOME/GoogleTest`. The benchmark also includes
`../Spinlock/gb_wall_clock.h` (through `-I../Spinlock`), the wall-clock
accounting behind its `wall_items_per_second` and `finish_spread` counters.
The compiler, `-march` target and C++ standard come from `../config.mk` (the reference one selects `clang++-22`
and C++23; the code also builds with `g++-16`).

### Third-party code

Other people's code is used from their repositories, never copied into this
one. Clone each into `ThirdParty/` at the top of this repository, at the
commit our patches were made against; the build then makes the directories it
uses. From `SharedPtr/`:

```sh
mkdir -p ../ThirdParty && cd ../ThirdParty
git clone https://github.com/magedm/mm_hp HazardPtr
git -C HazardPtr checkout b26e5ed
git clone https://github.com/DanielLiamAnderson/atomic_shared_ptr AtomicSharedPtr
git -C AtomicSharedPtr checkout 3c213ef
git clone https://github.com/cmuparlay/parlaylib ParlayLib
git -C ParlayLib checkout 5101769
cd ../SharedPtr && make
```

`make imports` here makes `lock_free_shared_ptr/parlay/` (Daniel Anderson's
`parlay::shared_ptr` and hazard pointers plus `lock_free_shared_ptr/parlay.patch`,
with ParlayLib's headers for its pool allocator); `make imports` in
`../IntrSharedPtr` makes `../IntrSharedPtr/mm_hp/` (Maged Michael's hazard
pointers plus `../IntrSharedPtr/mm_hp.patch`). Each patch begins with what each
change is for. Every binary here needs both, so every build here asks both
projects' `imports` first (this directory's own, and
`make -C ../IntrSharedPtr imports`), and a plain `make` is enough; LockFreeList
does the same. A user of `../IntrSharedPtr` alone needs only the first clone.
Either directory is made again at the next build after its patch changes, or
after `../third_party.mk` does (the rules, the pinned commits and what is
taken from each clone). The rules take the content from the pinned commit,
never from the clone's working tree, apply the patch without fuzz, and refuse
a clone at any commit other than the pinned one, since a patch may not fit
other code; set `THIRD_PARTY` if the clones live elsewhere. On a machine
without the clones, a build keeps the directories already there (made
elsewhere and copied in) with a warning; their `SOURCE.txt` records what they
were made from.

- `atomic_shared_ptr_concept.h` — the concepts, and the `std::atomic` adapter
- `lock_free_shared_ptr/` — our adapter over Daniel Anderson's pointer, and
  `parlay.patch`; its `parlay/` is made by `make imports` (above)
- `atomic_shared_ptr_test.C` — unit tests, run on all four pointers (built
  with ASan and TSan)
- `atomic_shared_ptr_bm.C` — the four pointers head to head; rows are suffixed
  `_StdAtomic`, `_IntrShared`, `_LockFree` (parlay) and `_IntrSharedHP`

The intrusive pointers, `mm_hp/`, the drain helper and their own tests and
microbenchmark are in `../IntrSharedPtr`
([README](../IntrSharedPtr/README.md)), included through `-I../IntrSharedPtr`.
