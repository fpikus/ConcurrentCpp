# SharedPtr

Atomic reference-counted smart pointers: the machinery under
`../LockFreeList`, tested and benchmarked here on their own, with no data
structure in the way.

An atomic shared pointer has one problem that a plain shared pointer does
not. To copy the pointer, a thread must read it and then increment the count
of the object it points to — and between the two, another thread may swing the
pointer away and drop the last reference, so that the object dies in the gap
and the increment lands on a corpse. The four pointers in this directory
differ, above all, in how they close that gap.

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
derive from (`pointee_base`) and that its pointees die late
(`deferred_reclamation`); only `intr_shared_ptr_hp`, below, declares either.

## The four pointers

1. `StdAtomicSharedPtrAdapter` (in the concept header) —
   `std::atomic<std::shared_ptr<T>>` dressed in the concept's interface. It
   has no room for a mark (`supports_marking` is `false`), which forces a
   weaker deletion algorithm on the list built on it. Not lock-free:
   libstdc++ closes the gap with a lock bit hidden in its control-block
   pointer, and says so (`is_always_lock_free` is `false`).
2. `intr_shared_ptr` (`intr_shared_ptr/`) — intrusive: the count lives in the
   pointee, the Harris mark in bit 0 of the pointer word, and a one-bit
   spinlock in bit 1 guards the gap. Not lock-free, and it does not pretend to
   be; a list built on it is still lock-free in the algorithmic sense.
3. Daniel Anderson's `parlay::atomic_shared_ptr` (`lock_free_shared_ptr/`) —
   genuinely lock-free, with hazard pointers demoted to an implementation
   detail inside the pointer. His headers are fetched separately; see
   [lock_free_shared_ptr/README.md](lock_free_shared_ptr/README.md).
4. `intr_shared_ptr_hp` (`intr_shared_ptr_hp/`) — the intrusive pointer with
   the lock taken out: the gap is closed by a hazard pointer (Maged Michael's
   `mm_hp`, below), and the increment is conditional — `TryAddRef()` adds a
   reference only to an object whose count is not already zero, so a dying
   object is never revived. Lock-free, exactly: `load()`, `store()`, both
   compare-exchanges and every operation on the shared pointer it hands out
   are lock-free algorithms, with one exception inside `mm_hp` — a thread's
   first `load()` takes a hazard record from a global pool guarded by a
   one-bit spinlock (and the thread's exit returns its records under the same
   lock). After a thread's first `load()`, everything it does with this
   pointer is lock-free. The price is deferred reclamation: an object whose
   count reaches zero is not deleted but retired, and it is destroyed later,
   at an `mm_hp` scan, by whichever thread happens to trigger that scan —
   possibly one that never touched this pointer. The pointee derives publicly
   from `std::hazard_pointer_obj_base<U>`, and its destructor must not care
   which thread runs it. The overview comment at the top of
   `intr_shared_ptr_hp.h` is the authority on the protocol and on every claim
   in this paragraph.

## Maged Michael's hazard pointers (`mm_hp/`)

The hazard pointers under `intr_shared_ptr_hp` are Maged Michael's `mm_hp`, an
implementation of the C++26 hazard pointers (`[saferecl.hp]`). Unlike Daniel
Anderson's code, this one *is* in the repository: `mm_hp/` is a copy of

https://github.com/magedm/mm_hp

at commit `b26e5ed`, together with its README and license files. The code is
dual-licensed, MIT or Apache 2.0 with LLVM exception at the user's option (see
`mm_hp/LICENSES`); we use it under the MIT license. The README in `mm_hp/` is
upstream's and is left exactly as upstream wrote it; our notes are here.

- **One local change.** `mm_hp.cpp` carries a ThreadSanitizer-only patch
  (marked `TSAN-PATCH`): under TSan, the scan loads the hazard slots with
  acquire instead of relaxed. TSan does not model `mm_hp`'s asymmetric fence
  pair (`membarrier(2)` on the scanning side), and without the patch it
  reports false races between a reader and the deleter. Builds without TSan
  are unchanged.
- **TSan is still not a lifetime oracle for hazard pointers**, patch or no
  patch. Every `load()` that protects an object ends with a release store to
  its hazard record, and the scan acquires that record — so the reader's
  entire earlier history, including an access it made with no protection at
  all, is ordered before the free. A clean TSan run says nothing about the
  protocol; model checking and the deterministic seam tests (below) are the
  oracles. TSan still checks everything around the protocol.
- **Linux only**, with no guards: `asymmetric_fence.hpp` includes
  `<linux/membarrier.h>`, and `mm_hp` aborts at its first reclamation where
  `membarrier(2)` is missing. Built and tested on x86-64 with clang-22 and
  gcc-16; untested on aarch64.
- **Never combine with a real `<hazard_pointer>`.** `mm_hp` defines
  `std::hazard_pointer`, `std::hazard_pointer_obj_base` and
  `std::make_hazard_pointer` itself. Nor is a standard `<hazard_pointer>` a
  drop-in replacement once the name clash is resolved: `intr_shared_ptr_hp`
  must protect a pointer whose mark bit may be set, which the standard's
  `protect()` and `try_protect()` cannot do, so it hand-rolls its validation
  from the public pieces; that sequence is sound for `mm_hp`'s implementation
  but not covered by the standard's contract.
- **Deleters run on whichever thread crosses `mm_hp`'s reclamation
  threshold** — inside that thread's `retire()`, once the pending
  retirements reach max(1000, 2 × the number of hazard records) — and once
  more at process exit. A thread's exit reclaims nothing. Objects retired by
  destructors running inside a scan wait for the next one, so a chain of
  objects that release each other dies one link per scan.

## Building and testing

```sh
make                              # benchmark, microbenchmarks, every test
make run_tests                    # run every test binary
make run_benchmarks               # run the benchmark and both microbenchmarks
make atomic_shared_ptr_bursts_bm  # on demand: + destruction-burst counters
```

Requires Linux, clang (the Makefile uses `clang++-22`, C++23), Google
Benchmark and GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your
installations if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.
The pointer benchmark and the unit tests also need Daniel Anderson's `parlay/`
headers (see above).

- `atomic_shared_ptr_concept.h` — the concepts, and the `std::atomic` adapter
- `intr_shared_ptr/` — the spinlock intrusive pointer, with its
  microbenchmark (`intr_shared_ptr_mbm.C`) and standalone TSan stress test
  (`intr_shared_ptr_tsan.C`); `intr_shared_ptr.h` is a symlink to its header
- `intr_shared_ptr_hp/` — the hazard-pointer intrusive pointer, with the same
  two companions (`intr_shared_ptr_hp_mbm.C`, `intr_shared_ptr_hp_tsan.C`);
  `intr_shared_ptr_hp.h` is a symlink to its header
- `lock_free_shared_ptr/` — our adapter over Daniel Anderson's pointer (his
  headers fetched separately)
- `mm_hp/` — Maged Michael's hazard pointers, vendored (above)
- `hp_drain.h` — drains `mm_hp`'s pending reclamations on demand, so that a
  test can assert that an object is gone: `mm_hp` has no public flush, and
  without one "destroyed" means "at some later scan"
- `hp_drain_gtest.h` — included by a GoogleTest binary, runs one drain before
  the first test, so that `mm_hp`'s first scan happens at the same point in
  every run
- `hp_drain_selftest.C` — tests of the drain itself (built with ASan and TSan)
- `intr_shared_ptr_hp_seam_test.C` — deterministic white-box tests of the
  hazard-pointer protocol: hooks in the pointee open each race window on
  purpose, in every run (built with ASan and TSan)
- `atomic_shared_ptr_test.C` — unit tests, run on all four pointers (built
  with ASan and TSan)
- `atomic_shared_ptr_bm.C` — the four pointers head to head; rows are suffixed
  `_StdAtomic`, `_IntrShared`, `_LockFree` (parlay) and `_IntrSharedHP`
