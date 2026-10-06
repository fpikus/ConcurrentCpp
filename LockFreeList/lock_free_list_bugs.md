# LockFreeList — Bug Record (Review of 2026-07-16)

Findings of a correctness review of `lock_free_list.h`, kept for future
reference. Scope: the list algorithm itself, treating the three atomic shared
pointers (`StdAtomicSharedPtrAdapter`, `intr_shared_ptr`,
`parlay::atomic_shared_ptr`) as black-box atomics per the design contract.

| # | Severity | Mode | Status |
|---|----------|------|--------|
| 1 | Critical — permanent loss of erase functionality | marking | **Fixed** (helping) |
| 2 | Semantic anomaly — lost erase / node resurrection | non-marking | Documented limitation (unfixable without a mark bit) |
| 3 | Latent data race — under-synchronized CAS failure orders | both | **Fixed** (`acquire` failure orders) |

---

## Bug 1: Stuck logically-deleted nodes permanently wedge `erase_after`

### Symptom

After a concurrent insert/erase workload, logically deleted (marked) nodes
remained physically linked in the list *forever*. Consequences:

- Traversal yields the erased values indefinitely.
- The node's memory stays pinned by its predecessor for the list's lifetime.
- Worst: `erase_after` over the edge leading to a stuck node returns `false`
  forever, because it finds the successor's next pointer already marked and
  concludes there is nothing for it to erase. A drain loop
  (`while (erase_after(h)) {}`) terminates with the list still full.

Observed empirically (2 threads doing `insert_after`/`erase_after` at
`before_begin()`, 50k inserts): first round produced **277 stuck nodes**, and
the subsequent drain erased **zero** of the ~39,600 still-reachable nodes.

### Root cause

The original code performed the physical unlink as a *single* best-effort CAS
and only for the thread that won the marking race:

```cpp
if (!marked_by_us) {
    return false; // Someone else marked it (erased it) concurrently
}
// Physically unlink it. We ignore failure because if it fails, someone else unlinked it.
anchor.curr_->next.compare_exchange_strong(target, target_next.get_unmarked(), ...);
return true;
```

The comment's justification was wrong: no other code path ever unlinks a
marked node (traversal deliberately walks *through* marked nodes for graveyard
semantics, and `erase_after` bailed out on them). The unlink CAS fails when
`anchor->next` changed between the mark and the unlink, which happens in two
reachable interleavings:

1. **Insert wins over unlink.** Eraser marks X (the successor of anchor A);
   a concurrent `insert_after(A, N)` swings `A->next: X -> N` first; the
   eraser's unlink CAS (expects X) fails. X stays linked after N.
2. **Anchor erased concurrently.** While the eraser of X stalls between mark
   and unlink, another thread erases anchor A itself; A's eraser splices A's
   predecessor directly to X — putting the marked X back into the live chain.

Either way the marked node has no owner left that will ever remove it. The
classic Harris algorithm avoids this by *helping*: any operation that
encounters a marked node unlinks it as a side effect.

### Fix

Helping in `erase_after` (marking branch). The unlink CAS now runs regardless
of who won the marking race, and a thread that finds the successor already
marked — whether it just lost the race or the node is long dead — unlinks it
and **retries on the new successor** instead of returning `false`:

```cpp
// target is logically deleted (by us or not). Unlink it physically (helping).
anchor.curr_->next.compare_exchange_strong(target, target_next.get_unmarked(), ...);
if (marked_by_us) {
    return true; // our mark is the linearization point
}
// Helped unlink someone else's dead node; retry on the new state.
```

Progress guarantee: a retry happens only after some CAS on `anchor->next`
succeeded (ours or a competitor's), so the algorithm remains lock-free.

**Semantic change:** losing the marking race no longer returns `false`; the
call moves on and tries to erase the *new* current successor. `true` still
means "this call logically deleted a node"; `false` now reliably means
"anchor is null/deleted or has no live successor", which is what makes a
drain loop correct under contention. Residual garbage is bounded and
transient: a dead node whose unlink lost its CAS stays linked only until the
next `erase_after` over the same edge.

### Overhead

None on the normal path — the uncontended success path (two loads, mark CAS,
unlink CAS) is instruction-for-instruction the same as before; only the branch
moved after the unlink. Helping adds one CAS + reload per dead node
encountered, on paths that previously returned a wrong answer; system-wide it
amortizes to zero extra successful CASes (each dead node is unlinked exactly
once either way — helping only changes which thread pays).

Measured (clang -O3, medians of 3 reps, 1000-node prepopulated list):

- **Single-threaded** (no races, helping never triggers — the clean measure of
  normal-path overhead): identical within the ~2% noise floor. IntrPtr:
  WriteHeavy 89.1 → 88.8 ns, Graveyard 44.8 → 45.0 ns, MassiveHeadInsert
  71.0 → 69.4 ns.
- **8 threads**: structurally *not* apples-to-apples, because the pre-fix list
  stops doing real erase work once an edge wedges (`erase_after` degenerates
  into a cheap `false`-returning no-op) while the fixed list performs every
  erase it reports. IntrPtr: WriteHeavy 795 → 827 ns/op (~+4%: the price of
  actually erasing, plus occasional helping), Graveyard 603 → 418 ns/op (the
  wedged baseline degenerates outright), MassiveHeadInsert — insert-only,
  whose code is unchanged — 607 → 600 ns/op, which establishes the ~2% noise
  floor. StdAtomic's erase code is effectively unchanged (non-marking branch;
  the failure-order change is immaterial to the mutex-based implementation)
  and its 8-thread numbers swing ±20% run-to-run — pure noise.

Measurement pitfall, recorded for future baseline comparisons: `#include "..."`
searches the *includer's* directory before any `-I` path, so a test/benchmark
`.C` living in the project directory always picks up the project's
`lock_free_list.h` no matter what `-I` says. To build against an older header,
copy the `.C` file next to that header (or use `-iquote`). The first attempt at
this comparison silently benchmarked the fixed header twice.

### Verification

- Repro/verifier program below: pre-fix, wedged on round 0 (reproduced at -O0
  and -O1, with and without ASan); post-fix, 200/200 rounds drain to empty,
  with the drain count exactly matching the live-node count (helped-out dead
  nodes are not reported as erasures).
- `InsertEraseHammer` regression test in `lock_free_list_test.C`: the same
  two-thread collision shape as the repro, plus a debug-only (`#ifndef
  NDEBUG`) post-join drain assertion. Against the pre-fix header it fails for
  both marking pointer types (IntrPtr, Parlay) and passes for the non-marking
  StdAtomic — exactly the expected selectivity, since bug 1 lives in the
  marking branch. The mixed-op StressTest carries the same drain assertion,
  but its diffuse workload (8 threads, RNG-mixed ops with traversals) does not
  reliably produce the mark-vs-insert interleaving on its own.
- Full gtest suite (all three pointer types) passes under ASan and TSan.

---

## Bug 2: Non-marking `erase_after` can resurrect a concurrently erased node

### Status: documented limitation, not fixed

Applies only to the non-marking fallback (`supports_marking == false`, i.e.
`StdAtomicSharedPtrAdapter`). With `H -> X -> Y -> Z`:

1. Thread 1, `erase_after(H)`: loads `target = X`, then `target_next = Y`.
2. Thread 2, `erase_after(it_X)`: CAS `X->next: Y -> Z` succeeds, returns
   `true` — Y is erased.
3. Thread 1: CAS `H->next: X -> Y` still succeeds (`H->next` never changed).

Result: the list is `H -> Y -> Z`. Y is back in the list even though its
erasure reported success (a lost update). This is precisely the anomaly
Harris marking exists to prevent: the mark *freezes* a deleted node's next
pointer, so step 2 and step 3 cannot both succeed. `std::shared_ptr` offers
no bit to steal, so the anomaly is unfixable in the adapter; it is documented
in `lock_free_list.h` (non-marking erase branch) and in
`atomic_shared_ptr_concept.h`. The related insert-side anomaly (an insert
after a concurrently erased anchor vanishes into a detached "black hole"
chain) was already documented; note that it is a lost insertion, not a memory
leak — reference counting still reclaims the chain.

If these semantics are unacceptable for a workload, use a marking pointer, or
restrict the non-marking list to insert-only / single-eraser usage.

---

## Bug 3: Under-synchronized CAS failure memory orders

### Status: fixed

Three CAS sites passed `std::memory_order_relaxed` as the failure order and
then *reused* the refreshed `expected` value:

1. `erase_after` marking race: the refreshed `target_next` is republished via
   the unlink CAS into `anchor->next`.
2. `insert_after` blind (non-marking) branch: the refreshed `expected_next`
   is stored into `new_node->next` and published with the insert CAS.
3. `erase_after` non-marking branch: the refreshed `target` is dereferenced
   and its successor republished into `anchor->next`.

A reader that acquires the republished pointer synchronizes only with the
republishing thread, not with the thread that constructed the pointee — a
formal data race (stale reads of the node's contents) under a minimally
conforming `AtomicPtr`. It was latent in practice because all three current
pointer implementations over-synchronize internally (embedded spinlock,
libstdc++ mutex, hazard-pointer fences), and on x86 every `lock cmpxchg` is a
full barrier anyway.

Fix: `acquire` failure order at those three sites. The two sites whose
refreshed `expected` is discarded (marking-insert CAS, which reloads with
`acquire`; the unlink/help CAS, whose retry reloads) keep `relaxed`, with
comments explaining why. Cost: zero on x86; on ARM64 at most a few cycles on
the failure path (`casal` vs `casl`), unmeasurable in the benchmarks above.

---

## Appendix: repro / verifier program

Originally used to demonstrate bug 1; now serves as a regression check
(exit code 0 = healthy). Build standalone against the list headers, e.g.
`clang++ -std=c++23 -O1 -fsanitize=address -I. -I../IntrSharedPtr -I../SharedPtr stuck_node_repro.C -lpthread`.

```cpp
// Interleaving exercised (2 threads hammering the same anchor H):
//   Eraser:   target = H.next (= X); marks X.next  ... [preempted]
//   Inserter: CAS(H.next: X -> N) succeeds, N.next = X
//   Eraser:   unlink CAS(H.next: X -> X.next) fails
// Pre-fix: the failure was ignored and X stayed linked-but-dead forever.
// Post-fix: the next erase_after over that edge helps X out; a drain loop
// must always empty the list.
#include <atomic>
#include <cstdio>
#include <thread>

#include "intr_shared_ptr.h"
#include "lock_free_list.h"

template <typename U> using Ptr = intr_shared_ptr<U, U>;
using List = LockFreeList<int, Ptr>;
using Node = List::Node;
using SP = List::SharedNodePtr;

int main() {
    for (int round = 0; round < 200; ++round) {
        List list{SP(new Node(0))};
        std::atomic<bool> go{false};
        std::atomic<bool> stop{false};

        std::thread eraser([&] {
            while (!go.load()) {}
            while (!stop.load()) { list.erase_after(list.before_begin()); }
        });
        std::thread inserter([&] {
            while (!go.load()) {}
            for (int i = 1; i <= 50000; ++i) {
                list.insert_after(list.before_begin(), SP(new Node(i)));
            }
        });
        go.store(true);
        inserter.join();
        stop.store(true);
        eraser.join();

        // Quiescent now: any node whose own `next` is marked is logically
        // deleted yet still physically reachable.
        int stuck = 0, live = 0;
        for (auto it = list.begin(); it != list.end(); ++it) {
            if (it.curr_->next.load().is_marked()) ++stuck; else ++live;
        } // scan for stuck nodes

        // Drain: on a correct list this removes every node -- live ones and
        // any transient dead ones left by an unlink that lost its CAS just
        // before the threads stopped. On the buggy list it wedges at the
        // first stuck node.
        int drained = 0;
        while (list.erase_after(list.before_begin())) ++drained;
        int remaining = 0;
        for (auto it = list.begin(); it != list.end(); ++it) ++remaining;

        if (remaining > 0) {
            std::printf("round %d: BUG -- %d stuck / %d live at quiescence; "
                        "drained %d, %d node(s) remain unreachable to erase\n",
                        round, stuck, live, drained, remaining);
            return 1;
        }
        if (stuck > 0) {
            std::printf("round %d: %d transient dead node(s) at quiescence, "
                        "%d live; drain removed all %d -- OK\n",
                        round, stuck, live, drained);
        }
    } // repro rounds
    std::printf("all 200 rounds drained to empty -- no wedged nodes\n");
    return 0;
} // main()
```
