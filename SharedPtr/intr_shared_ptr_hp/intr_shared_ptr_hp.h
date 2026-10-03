// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/ConcurrentCpp
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#ifndef INCLUDED_INTR_SHARED_PTR_HP_H_
#define INCLUDED_INTR_SHARED_PTR_HP_H_

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

// Maged Michael's hazard pointers (vendored in mm_hp/ beside this header's
// directory). How the quoted include resolves: a TU that opens the header
// through the symlink SharedPtr/intr_shared_ptr_hp.h or
// LockFreeList/intr_shared_ptr_hp.h looks next to the symlink AS SPELLED,
// where each directory keeps mm_hp/ (a directory symlink in LockFreeList/);
// the TUs inside intr_shared_ptr_hp/ itself (the microbenchmark and the TSan
// program) include the header by its real path, and for them mm_hp/ resolves
// through the Makefile's -I. (SharedPtr/). It defines std::hazard_pointer,
// std::hazard_pointer_obj_base and std::make_hazard_pointer ITSELF -- see
// "mm_hp, the standard and the platform" below.
#include "mm_hp/mm_hp.hpp"

// intr_shared_ptr_hp: an atomic, intrusively-reference-counted shared pointer
// with Harris-style pointer marking, whose load->AddRef gap is closed by a
// hazard pointer instead of intr_shared_ptr's one-bit spinlock. It conforms to
// the AtomicSharedPtr concept (atomic_shared_ptr_concept.h) and is the fourth
// AtomicPtr policy of LockFreeList. It started as a complete copy of
// intr_shared_ptr and was modified as needed; the two headers are deliberately
// independent for now (complete copy first, reuse later).
//
// Template parameters:
//   T - the interface type seen through the smart pointer: operator*, operator->
//       and get() return T. T must not be thread-affine (see "Deferred
//       reclamation" below: ~T runs on whichever thread triggers a scan).
//   U - the concrete stored type (defaults to T). The atomic word holds a U*
//       (plus the mark bit); U* must convert implicitly to T* (U is T or is
//       publicly derived from T). U carries the intrusive count and the hazard
//       pointer base: it must satisfy HpIntrusivePointee<U> (below). U is the
//       DYNAMIC type of every pointee: reclamation runs std::default_delete<U>,
//       i.e. `delete static_cast<U*>(p)`, so an object of a type derived from U
//       is destroyed through U's destructor (virtual or not -- the hazard
//       pointer base fixes the deleter, nothing here can change it). T and U
//       differ when the pointee is exposed through a base interface T while
//       the hooks live on a derived U.
//
// Bit layout of the stored uintptr_t:
//   bit 0   - mark bit (Harris logical-deletion tag; part of the value/identity)
//   bits 1+ - the U* pointer. The hazard pointer base forces alignof(U) >= 8,
//             so bit 0 is always free. Unlike intr_shared_ptr there is no lock
//             bit: bit 1 is an ordinary address bit here.
//
// The value of the atomic is the pair (pointer, mark) = the raw word. Two
// shared_ptr_type values are equal iff their raw words are equal: a marked and
// an unmarked pointer to the same object are DIFFERENT values, and a marked
// null (word == 1) is a legitimate value distinct from null (is_marked() true,
// operator bool false, get() == nullptr, use_count() == 0). Every operation
// below compares and publishes the full word.
//
// ---------------------------------------------------------------------------
// The load protocol (orders in brackets; every one of them is load-bearing)
// ---------------------------------------------------------------------------
//
// load() is the only operation that acquires a reference it does not already
// own, so it is the only one that needs a hazard pointer. The problem it
// solves: between reading the word (X) and incrementing X's count, another
// thread may swing the word away and release the word's reference, dropping
// X's count to 0 and retiring X. A hazard pointer on X keeps X's MEMORY alive
// across that gap; the count protocol keeps the OBJECT from being revived.
//
//   1. Read the word with the caller's `order`. A null or marked-null word is
//      returned at once: nothing to protect, nothing to count.
//   2. Publish the UNMARKED address X in a hazard record [release store, by
//      hazard_pointer::reset_protection<U>(X), which forms the address of the
//      hazard-pointer base subobject with a static_cast from U*; it is never
//      formed from the raw word, because with another base class first the
//      hazard-pointer base sits at a nonzero offset inside U].
//   3. Light fence [p1202::asymmetric_thread_fence_light(): a compiler fence;
//      the ordering against the reclaimer's hazard scan comes from the
//      membarrier(2) heavy fence the scan executes -- mm_hp's design].
//   4. Reload the word [acquire] and compare it with X with the mark masked.
//      On mismatch go back to 1 with the reloaded word (the hazard record is
//      kept and re-aimed). On match X is protected: a retire of X must follow
//      the store that unpublished X, and since our reload still saw X the scan
//      that follows that retire sees our hazard. The reload must be ACQUIRE,
//      not relaxed, because it is this read that synchronizes with the
//      publisher of the object the hook is about to touch: the first read may
//      be relaxed and stale, the object it named may have been unpublished,
//      retired and freed, and a NEW object may occupy the same address and be
//      published (ABA on the address). The masked compare then matches the new
//      occupant, and with a relaxed reload step 5 reads its count with no
//      happens-before to its construction (model-checked: dies by address
//      reuse; clean with acquire).
//   5. TryAddRef() X: increment the count iff it is nonzero. A count of 0 means
//      X is retired or about to be retired and must never be revived (a plain
//      fetch_add would resurrect it and its later release would retire it a
//      second time; "fetch_add then undo on zero" is not a substitute either --
//      the undo races a re-validation-failure release into a double retire).
//      If the count was 0, reload the word [acquire] and go back to 1. The 0
//      was read with ACQUIRE (TryAddRef's contract), so it synchronizes with
//      the release sequence headed by the acq_rel DelRef that produced it, and
//      that DelRef follows (in its thread, or through another acq_rel DelRef)
//      the store that unpublished X: the reload is guaranteed to see the word
//      changed. This is what makes the retry lock-free rather than a spin: a
//      model-checked interleaving with a RELAXED zero-observing load re-reads
//      the stale word forever (livelock). The reload itself must be ACQUIRE
//      too, for a different reason: the word it returns (a null, say) may have
//      been published by a LATER store than the one whose release produced
//      the 0, and only an acquire reload synchronizes with that publisher as
//      load(acquire) promises (a deterministic TSan test kills the relaxed
//      form).
//   6. Re-validate: reload the word [acquire]; if its unmarked address is no
//      longer X, leave the hazard scope (the record is cleared with a release
//      store and returned to the thread cache), release the reference just
//      taken (retire X on 1->0) and go back to 1 with the reloaded word. Why
//      this step exists: LockFreeList::~Node's iterative walk judges a node
//      exclusive when its count is exactly 1. A loader that validated X and
//      then stalled can increment X AFTER the word was swung away and the
//      word's reference released, i.e. after the walk saw count 1 and froze
//      X's `next`; the loader would then own a dead node with a destroyed
//      `next`. With this step that loader releases X again instead: a word's
//      reference is counted before the word is published and released after
//      it is unpublished (store(), both CASes, the adopting constructor), so
//      count == 1 at the walk's test means no word points at X then or later,
//      and the re-validation fails. TryAddRef's CAS must be ACQUIRE on
//      success for this to hold. The value a successful CAS consumes is not
//      necessarily one this thread read with acquire: TryAddRef's failed CAS
//      refreshes it with a relaxed read and feeds that into the next attempt
//      (only a refreshed 0 is re-read with acquire), and other threads' RMWs
//      may have rewritten the count meanwhile. A relaxed success on a value
//      read from the release sequence headed by the swing's DelRef does not
//      synchronize with it, nothing orders this loader after the swing's
//      release, and the acquire re-validation reload may still return the
//      stale X (model-checked: assertion failure). The reload itself must be
//      ACQUIRE for a second reason,
//      independent of the walk: the word it returns IS the loaded value, and a
//      publisher may have changed only the MARK (plain writes, then a release
//      CAS X -> X|1); the pointee's construction was already acquired by the
//      step-4 reload, but the mark-only publication is acquired by nobody else,
//      so a relaxed re-validation reload breaks load(acquire)'s promise for
//      X|1 (killed by a deterministic TSan test; model checkers that do not
//      model mark-only publication let it survive).
//   7. Return the re-validated word's mark state with the one reference taken.
//      The hazard record is cleared [release store] and returned on scope exit,
//      after the increment -- never before it.
//
// Why not mm_hp's protect()/try_protect(): both compare the full word with the
// expected pointer, so a legitimately marked word (a deleted node's `next`,
// which this pointer must be able to load) never matches, and protect() would
// publish X|1, which protects nothing. The validated protect above is
// hand-rolled from the public pieces (reset_protection, the light fence, an
// acquire load, a masked compare) -- the sequence of mm_hp's own try_protect()
// with a masked compare. It is sound for mm_hp's IMPLEMENTATION (same fences,
// same scan-side heavy fence) but it is NOT covered by the P2530R3 contract:
// [saferecl.hp.general]/6.3.2 ties the "load observed a value published before
// the retire" guarantee to an epoch begun by try_protect; reset_protection plus
// a load has only 6.3.1 (epoch began before the retire). This header therefore
// relies on mm_hp, and a future standard <hazard_pointer> is not a drop-in
// replacement even once the name clash below is resolved.
//
// Lock-freedom, exactly: load(), store(), both CASes and every shared_ptr_type
// operation are lock-free algorithms -- a retry happens only after another
// thread's COMPLETED store/CAS on the word (or after a count hit 0, which, with
// the acquire zero-load, implies one). The one exception is mm_hp's
// hazard-record acquisition: a thread's first load() (empty thread cache) takes
// a record from the global pool under a 1-bit spinlock (acquire_hp_rec, which
// yields while the bit is set) or allocates one, and the thread's exit returns
// its records under the same lock. With one record per load() and a thread
// cache of 100 records (fast slot + 99), the miss happens once per thread, so
// load/store/CAS are lock-free after a thread's first load. Retire is a
// lock-free push plus a fetch_add; the scan (membarrier, a hash set of the
// hazards, the deleters) runs on the thread that crosses the threshold and
// waits on no other thread.
//
// ---------------------------------------------------------------------------
// Deferred reclamation
// ---------------------------------------------------------------------------
//
// A pointee whose count reaches 0 is NOT deleted; the releaser that made the
// 1->0 transition calls retire() on it (release() below is the one place).
// mm_hp pushes retired objects onto a global list and scans it when a retire
// brings the pending count to the threshold (max(1000, 2 x the number of
// hazard records)): the scan runs INSIDE that retire(), on WHICHEVER thread
// made it -- possibly a thread that never touched this pointer or its list --
// executes membarrier(2), collects the published hazards and runs the deleter
// (std::default_delete<U>: ~U, hence ~T) of every retired object that is not
// protected. Objects retired by destructors running inside a scan are pushed
// but do not start a nested scan; they are reclaimed in the NEXT scan, so a
// chain of objects whose destructors release each other dies one layer per
// scan (which is why LockFreeList::~Node walks its chain iteratively instead
// of relying on the cascade). Precisely, a scan runs at exactly two kinds of
// point: inside a retire() whose push brings the pending count to the
// threshold (and only if the retiring thread is not itself inside a scan),
// and once at process exit (a static destructor of mm_hp). A THREAD's exit is
// NOT a scan point: the thread's cached hazard records go back to the global
// pool, nothing is reclaimed. Objects retired by destructors inside the exit
// pass stay allocated, reachable from mm_hp's retired list, and are not
// reported by LSan. Tests that need "it is destroyed now" retire enough
// fillers to cross the threshold (hp_drain.h).
//
// Consequences, all part of the contract: (a) use_count() of a live handle
// never reads 0, but "the object is gone" happens only at a later scan --
// tests drain mm_hp (hp_drain.h) before asserting live counts; (b) ~T must not
// assume the releasing thread, thread_local state, or any thread affinity;
// (c) a destructor chain runs inside the retiring call, which may be a
// shared_ptr_type destructor, an assignment, store(), a CAS or ~intr_shared_ptr_hp
// anywhere in the program; (d) retire() and the scan are noexcept -- mm_hp
// handles an allocation failure of its hazard set internally (it falls back
// to scanning one object at a time), but a std::bad_alloc from
// make_hazard_pointer() inside a deleter's load() (a thread-cache miss in a
// destructor chain) propagates into a destructor and terminates. load() returns
// its record before the one release it can perform (step 6), so a nested load
// inside the resulting reclamation reuses that record and the miss cannot
// happen on that path -- with one exception: once a thread's hazard cache has
// been CLOSED (mm_hp flushes it from a thread_local destructor, so this is a
// thread already running its thread-local destructors), every
// make_hazard_pointer() goes to the global pool and may allocate (and throw)
// at any load().
//
// ---------------------------------------------------------------------------
// mm_hp, the standard and the platform
// ---------------------------------------------------------------------------
//
// mm_hp defines std::hazard_pointer and std::hazard_pointer_obj_base itself
// (adding names to namespace std -- accepted here as a vendored
// implementation of the proposal). A program using this header must not also
// use a real <hazard_pointer>, and because of the try_protect note above a
// standard <hazard_pointer> is not a drop-in replacement anyway.
//
// Linux only, no guards: asymmetric_fence.hpp includes <linux/membarrier.h>
// and the first reclamation calls std::abort() where membarrier(2) is missing.
// x86-64 built and tested with clang-22 and gcc-16; aarch64 untested (the
// acquire orders above cost LDAR/LDAXR there; nil on x86-64).
//
// TSan cannot see hazard-pointer lifetime violations, even with mm_hp's
// TSan-only patch (scan-side hazard loads acquire): every load() ends with a
// release store to its hazard record, the scan acquires that record, and the
// reader's whole earlier history -- including an unprotected access -- is
// thereby ordered before the free. A clean TSan run of this pointer says
// nothing about the protocol; model checking and deterministic seam tests
// through the pointee's hooks are the oracles. TSan still checks everything
// outside the protocol (callers, the count orders' visible effects).
//
// ---------------------------------------------------------------------------
// Exception safety
// ---------------------------------------------------------------------------
//
// No exception escapes any release (~shared_ptr_type, the assignments,
// ~intr_shared_ptr_hp, store(), CAS success): the hooks are required noexcept,
// retire() is noexcept. load() can throw only std::bad_alloc from mm_hp's
// make_hazard_pointer() on a thread-cache miss -- normally a thread's first
// load, before any side effect; on a thread whose hazard cache is closed (see
// above) every load() may miss, and a miss on the retry after a step-6
// re-validation failure comes after that step's release, which may have
// retired an object. Either way the counts are balanced and nothing of the
// caller's is changed: strong guarantee. The CAS failure paths refresh
// `expected` through load() and inherit that guarantee (see each CAS).
// Public members declared noexcept: the two trivial constructors of the
// atomic, shared_ptr_type's move constructor and move assignment,
// is_marked(), get_raw(), get() and operator bool; the destructors are
// implicitly noexcept (the private helpers are noexcept as well). The other
// public members cannot throw either once this thread has a cached hazard
// record, but are not declared so.
//
// ---------------------------------------------------------------------------
// Design notes: where this differs from the obvious or from the earlier
// prototype of the same protocol
// ---------------------------------------------------------------------------
//
// - The hazard record is acquired once per load() call and kept across the
//   validation-mismatch and zero-count retries (re-aimed with
//   reset_protection); it is returned to the thread cache BEFORE the one
//   release load() can perform (step 6) and re-acquired for the retry. The
//   prototype acquired a record per retry round and released the reference
//   while still holding it; the order here is what keeps a nested load inside
//   the resulting reclamation from allocating.
// - The publishing RMWs (store()'s exchange, the CASes' success) use the
//   caller's order promoted to at least ACQ_REL, not release: release so that
//   a loader's acquire validation reload sees the pointee's construction,
//   acquire because the same RMW hands the caller the OLD occupant, whose
//   count it then decrements and whose destructor it may run -- without the
//   acquire half those accesses have no happens-before with the old
//   occupant's publisher (a data race TSan reports, see publish_order()). The
//   earlier prototype used seq_cst for both CAS orders and did not have the
//   problem; a design that promotes only to release does. The CAS failure
//   order is acquire and is not load-bearing: the word a failed CAS reports is
//   discarded, because a reference to it can only be taken through the load
//   protocol, which re-reads the word.
// - A failed CAS refreshes `expected` through load(failure) -- the caller's
//   failure order applied to the first read of the word -- rather than the
//   prototype's fixed load(acquire). The protocol's internal reloads are
//   acquire regardless, so the two differ only in what the first read costs
//   and promises (a relaxed failure order on a null word does not
//   synchronize, exactly as std::atomic specifies).
// - store() asserts its order to the set std::atomic::store accepts (relaxed,
//   release, seq_cst); the exchange it uses would silently accept any order
//   (and runs with the promoted order above). load() and the CAS failure
//   order are asserted to the set std::atomic::load accepts.
// - TryAddRef() (a pointee hook, see HpIntrusivePointee) re-reads the count
//   with acquire when a failed CAS reports 0 before returning false, so that
//   EVERY observed 0 is an acquire read. A form that returns false on the
//   relaxed value the failed CAS reports is also safe and lock-free -- the
//   next round's initial acquire load must, by coherence, read 0 (0 is
//   terminal) and synchronize then -- but it costs one wasted protocol round
//   (hazard store, fence, two reloads) instead of one acquire load, and it
//   makes the proof two steps instead of one.
// - `aptr_` is not `mutable`: load() never writes the word (the spinlock
//   pointer's load() had to).

// HpIntrusivePointee<U>: what intr_shared_ptr_hp requires of its stored type U.
// Checked by static_assert INSIDE member function bodies (require_pointee()
// below), NEVER at class scope: LockFreeList::Node is incomplete at its own
// base clause `struct Node : pointee_base_of<AtomicPtr<Node>>::type`, where
// intr_shared_ptr_hp<Node> is first named, and a class-level constraint makes
// Node ill-formed.
//
// Requirements:
//   - U derives PUBLICLY and unambiguously from std::hazard_pointer_obj_base<U>
//     (std::derived_from checks exactly that). The deleter is the default
//     std::default_delete<U>, so U is the dynamic type of every pointee. The
//     base may sit at any offset inside U: the pointer forms the hazard address
//     through the typed reset_protection<U>(U*), never from the raw word.
//   - The four hooks, all noexcept, all RMWs on one atomic count (a plain store
//     to the count would break the release sequence that TryAddRef's acquire
//     zero-load synchronizes with -- protocol step 5):
//       void AddRef() noexcept;          increment (relaxed suffices); the return
//                                        value, if any, is ignored
//       bool TryAddRef() noexcept;       increment iff nonzero -- contract below
//       bool DelRef() noexcept;          decrement, acq_rel; true iff this call
//                                        made the 1 -> 0 transition. Both halves
//                                        are load-bearing: release heads the
//                                        sequence TryAddRef's acquire reads
//                                        synchronize with; acquire gives the
//                                        thread that reaches 0 (and retires) the
//                                        history of the thread that unpublished
//                                        the object, which the hazard-pointer
//                                        safety argument needs (unpublish
//                                        happens-before retire)
//       long use_count() const noexcept; current count (diagnostics/tests only;
//                                        a snapshot, stale by the time it returns)
//     The count starts at 0 for a freshly constructed object and is taken to 1
//     by its first owner (shared_ptr_type(U*)); 0 is reached again exactly once,
//     by the DelRef() that returns true, whose caller must retire() the object.
//
// TryAddRef() contract (carried into every implementation's comment):
//
//   TryAddRef(): increment the strong count if and only if it is nonzero.
//   Returns true iff it incremented; returns false iff it observed a count of
//   0, in which case the count is left at 0 (an object at 0 is retired, or
//   about to be retired, and must never be revived). Required only by
//   intr_shared_ptr_hp; the other pointer policies never call it. Memory
//   orders, all load-bearing: the load that observes 0 is ACQUIRE; the CAS is
//   ACQUIRE on success and relaxed on failure; a failed CAS whose refreshed
//   value is 0 re-reads the count with an acquire load before returning false,
//   so EVERY observed 0 was read with acquire. Why: intr_shared_ptr_hp::load()
//   calls this on an object pinned only by a hazard pointer. An observed 0
//   must synchronize with the release sequence headed by the DelRef that
//   produced it, so that the loader's next acquire reload of the word is
//   guaranteed to see the store that unpublished the object; with a relaxed
//   zero-observation the loader can re-read the stale word forever (model
//   checked: livelock). The ACQUIRE on CAS success makes the loader's
//   post-increment re-validation of the word see a swing that released the
//   word's own reference. The value a successful CAS consumes need not have
//   been read with acquire: a failed CAS refreshes it with a relaxed read (only
//   a refreshed 0 is re-read with acquire), and other threads' RMWs may have
//   rewritten it. A CAS that succeeds with relaxed order on a value it read
//   from the release sequence headed by the swing's DelRef does not
//   synchronize with that DelRef, so the re-validation reload may return the
//   stale word and LockFreeList::~Node's walk can judge a node exclusive
//   (count 1) that this loader then owns with a stale next (model checked:
//   assertion failure). Every operation on the count is an RMW. Cost: nil on
//   x86-64; LDAR/LDAXR on aarch64.
//
// Reference form:
//
//   bool TryAddRef() noexcept {
//       long count = ref_count.load(std::memory_order_acquire);
//       while (count != 0) {
//           if (ref_count.compare_exchange_weak(count, count + 1,
//                   std::memory_order_acquire, std::memory_order_relaxed)) {
//               return true;
//           }
//           // The failed CAS refreshed `count` with a relaxed read; a 0 seen
//           // that way does not synchronize. Re-read it with acquire.
//           if (count == 0) count = ref_count.load(std::memory_order_acquire);
//       } // CAS loop while the count is nonzero
//       return false;
//   } // TryAddRef()
template <typename U>
concept HpIntrusivePointee =
    std::derived_from<U, std::hazard_pointer_obj_base<U>> &&
    requires(U& u, const U& cu) {
        { u.AddRef() } noexcept;
        { u.TryAddRef() } noexcept -> std::same_as<bool>;
        { u.DelRef() } noexcept -> std::same_as<bool>;
        { cu.use_count() } noexcept -> std::same_as<long>;
    };

// Retry counters for benchmarks, compiled only under -DINTR_HP_BM_COUNTERS.
// One thread_local instance per thread, shared by every instantiation of the
// pointer; a benchmark reads the deltas per thread. Each field counts one of
// load()'s three retry paths and is incremented ONLY when that path retries,
// so an uncontended load() touches none of them:
//   validation_mismatch   - step 4: the word changed between the first read
//                           and the protected reload
//   zero_count            - step 5: the protected object's count was 0
//   revalidation_mismatch - step 6: the word changed after the increment
#ifdef INTR_HP_BM_COUNTERS
struct intr_shared_ptr_hp_retry_counters {
    unsigned long validation_mismatch = 0;
    unsigned long zero_count = 0;
    unsigned long revalidation_mismatch = 0;
}; // struct intr_shared_ptr_hp_retry_counters
inline thread_local intr_shared_ptr_hp_retry_counters intr_shared_ptr_hp_retries;
#define INTR_HP_COUNT_RETRY_(field) (++intr_shared_ptr_hp_retries.field)
#else
#define INTR_HP_COUNT_RETRY_(field) ((void)0)
#endif

template <typename T, typename U = T>
class intr_shared_ptr_hp {
public:
    // Optional policy members (atomic_shared_ptr_concept.h, "Optional members"):
    //
    // pointee_base: the base class a pointee must derive from. LockFreeList's
    // pointee_base_of trait detects this alias and makes Node derive from it;
    // the other policies do not declare it and Node gets an empty base.
    using pointee_base = std::hazard_pointer_obj_base<U>;

    // Marks are supported: bit 0 of the word (required by the concept).
    static constexpr bool supports_marking = true;

    // A released pointee is destroyed at a later mm_hp scan, not at count 0
    // (see "Deferred reclamation" above). Exists for documentation and for
    // `if constexpr` in pointee types; tests do not detect it -- they drain
    // unconditionally.
    static constexpr bool deferred_reclamation = true;

    // The non-atomic "value" type handed out by load() and accepted by store()
    // and the CASes. It owns one strong reference to its pointee (AddRef on
    // acquire, DelRef on release; retire() on 1->0) and may carry the mark bit
    // in bit 0 of its raw pointer. Identical in interface to
    // intr_shared_ptr::shared_ptr_type.
    //
    // Special members: rule of five, by hand, because the object owns a counted
    // reference. Copy = AddRef; move = steal (source becomes null); destroy =
    // release. Value semantics with mark-as-identity (operator== compares the
    // raw word including the mark).
    class shared_ptr_type {
    public:
        // Null. Postcondition: !*this, !is_marked(), use_count() == 0.
        shared_ptr_type() : p_(nullptr) {}
        shared_ptr_type(std::nullptr_t) : p_(nullptr) {}

        // Adopt a raw pointer and become one of its owners: AddRef once (0 -> 1
        // for a freshly new'ed object; n -> n+1 for an object that already has
        // owners -- adoption and sharing are the same operation because the
        // count lives in the object, so there is no control block to allocate
        // or find). Bit 0 of `p` may carry the mark; it is kept. Precondition:
        // `p` is null, or points to a live U whose count is nonzero, or to a
        // fresh U whose count is 0 and that has never been owned (an owned
        // object whose count reached 0 has been retired and must not be
        // revived). A raw marked null (the bit pattern 1) is allowed and yields
        // marked null, with no count traffic. Postcondition: get() == unmarked
        // p, is_marked() == (p & 1).
        explicit shared_ptr_type(U* p) : p_(p) {
            require_pointee();
            if (get_unmarked_ptr()) get_unmarked_ptr()->AddRef();
        }

        // Copy: AddRef the shared pointee (if non-null). Postcondition:
        // *this == x; x.use_count() grew by one if non-null.
        shared_ptr_type(const shared_ptr_type& x) : p_(x.p_) {
            require_pointee();
            if (get_unmarked_ptr()) get_unmarked_ptr()->AddRef();
        }

        // Move: steal x's reference. Postcondition: *this holds x's old word,
        // x is null (unmarked null, even if x was marked null).
        shared_ptr_type(shared_ptr_type&& x) noexcept : p_(x.p_) {
            x.p_ = nullptr;
        }

        // Release: DelRef the pointee (if non-null); on 1->0 retire() it (it
        // is destroyed at a later scan, on some thread -- see the overview).
        ~shared_ptr_type() {
            release(get_unmarked_ptr());
        }

        // AddRef the new pointee *before* releasing the old one: `this` and `x`
        // can be distinct shared_ptr_type objects holding the same pointee --
        // often as marked and unmarked variants of one pointer, which the
        // mark-as-identity design (get_unmarked(), set_mark() return copies)
        // makes routine. Releasing first could drop the last reference and
        // retire the very object we are about to AddRef; the self-assignment
        // check catches only `a = a`, not that aliasing.
        shared_ptr_type& operator=(const shared_ptr_type& x) {
            if (this == &x) return *this;
            U* new_ptr = unmarked_ptr(reinterpret_cast<uintptr_t>(x.p_));
            if (new_ptr) new_ptr->AddRef();
            release(get_unmarked_ptr());
            p_ = x.p_;
            return *this;
        }

        // Move assignment needs no AddRef-first dance: x's reference is being
        // transferred, so even when *this and x share the pointee the count
        // includes x's reference until p_ is overwritten and stays >= 1
        // through the release below.
        shared_ptr_type& operator=(shared_ptr_type&& x) noexcept {
            if (this == &x) return *this;
            release(get_unmarked_ptr());
            p_ = x.p_;
            x.p_ = nullptr;
            return *this;
        }

        // Access through the UNMARKED pointer, as T. Precondition for * and ->:
        // non-null (get() != nullptr); undefined otherwise. A marked handle
        // dereferences like its unmarked twin.
        T& operator*() const { return *get_unmarked_ptr(); }
        T* operator->() const { return get_unmarked_ptr(); }
        T* get() const noexcept { return get_unmarked_ptr(); }
        explicit operator bool() const noexcept { return get_unmarked_ptr() != nullptr; }

        // Identity comparison of the raw word: pointer AND mark. Marked and
        // unmarked handles to one object compare unequal; marked null != null.
        bool operator==(const shared_ptr_type& rhs) const { return p_ == rhs.p_; }
        bool operator!=(const shared_ptr_type& rhs) const { return p_ != rhs.p_; }

        // The pointee's current count (0 for null and marked null): a snapshot
        // for diagnostics and tests; never 0 for a non-null handle (this handle
        // is one owner).
        long use_count() const {
            U* ptr = get_unmarked_ptr();
            return ptr ? ptr->use_count() : 0;
        }

        // Harris marking API. The mark lives in bit 0 of the raw pointer and is
        // considered part of the pointer's identity (operator== compares it too),
        // so a marked and unmarked pointer to the same object are *not* equal.
        bool is_marked() const noexcept { return (reinterpret_cast<uintptr_t>(p_) & 1ULL) != 0; }

        // Two overloads: the lvalue one must copy (an AddRef/DelRef round
        // trip; the source is untouched); the rvalue one clears the bit in
        // place and MOVES the reference into the result, so the source is
        // left null (unmarked null) -- which is what makes the ubiquitous
        // `next.load(...).get_unmarked()` refcount-churn-free. Postcondition
        // of both: result.get() == get(), !result.is_marked().
        shared_ptr_type get_unmarked() const & {
            shared_ptr_type res(*this);
            res.p_ = res.get_unmarked_ptr();
            return res;
        }

        shared_ptr_type get_unmarked() && {
            p_ = get_unmarked_ptr();
            return std::move(*this);
        }

        // Copy with the mark set. Legal on null (yields marked null).
        shared_ptr_type set_mark() const {
            shared_ptr_type res(*this);
            res.p_ = reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(res.p_) | 1ULL);
            return res;
        }

        // Raw pointer with the mark bit still attached (the word's bit pattern).
        // Do not dereference. Used by the atomic owner and by tests comparing
        // identities.
        U* get_raw() const noexcept { return p_; }

    private:
        friend class intr_shared_ptr_hp;

        // The raw word: U* with the mark in bit 0. Null when no reference is owned.
        U* p_;

        // The pointer with the mark bit cleared: safe to dereference and to
        // pass to the hooks.
        U* get_unmarked_ptr() const noexcept {
            return unmarked_ptr(reinterpret_cast<uintptr_t>(p_));
        }
    }; // class shared_ptr_type

    // Null atomic.
    constexpr intr_shared_ptr_hp() noexcept : aptr_(0) {}
    explicit(false) intr_shared_ptr_hp(std::nullptr_t) noexcept : aptr_(0) {}

    // Take ownership of `desired`'s pointee, mark and all. `desired` keeps its
    // own reference (taken by value, released when the parameter dies), so the
    // atomic AddRefs once more for the reference the word now owns -- counted
    // BEFORE the word is written, like every publication in this class. The
    // store is relaxed: a constructor cannot be racing with operations on
    // *this; publishing *this to other threads is the caller's job.
    explicit(false) intr_shared_ptr_hp(shared_ptr_type desired) {
        require_pointee();
        U* new_unmarked = desired.get_unmarked_ptr();
        if (new_unmarked) new_unmarked->AddRef();
        aptr_.store(reinterpret_cast<uintptr_t>(desired.get_raw()), std::memory_order_relaxed);
    }

    // Non-copyable and non-movable, like std::atomic: other threads may hold
    // the address of *this. (The deleted copy suppresses the implicit moves.)
    intr_shared_ptr_hp(const intr_shared_ptr_hp&) = delete;
    intr_shared_ptr_hp& operator=(const intr_shared_ptr_hp&) = delete;

    // Release the held pointee's reference (mask bit 0 only -- the mark bit may
    // legitimately be set, a marked pointer is still an owning pointer; there
    // is no lock bit); retire() on 1->0. Precondition: no operation on *this is
    // in flight on any thread (as for any std::atomic), which is why a relaxed
    // read of the word suffices: whatever established that exclusivity already
    // ordered all prior accesses to aptr_. That retire() may cross mm_hp's
    // threshold and run a scan on this thread: the destructors of up to
    // ~threshold unrelated retired objects run inside this destructor.
    ~intr_shared_ptr_hp() {
        require_pointee();
        release(unmarked_ptr(aptr_.load(std::memory_order_relaxed)));
    }

    // Atomically snapshot the word and take a strong reference to its pointee
    // (the protocol of the overview, steps 1-7).
    //
    // Returns a shared_ptr_type whose raw word (pointer AND mark) is a value
    // the atomic held at some instant during the call -- the instant of the
    // protocol's final validating read -- owning one strong reference to the
    // pointee when non-null. Null and marked null are returned as such without
    // touching any count. Linearizable with respect to every other operation
    // of this object, with the one qualification the next paragraph states:
    // under a relaxed (or consume) `order` a null returned from the FIRST read
    // is a relaxed read, which may be older than what an acquire reader of the
    // same word already saw.
    //
    // order: MUST be relaxed, consume, acquire or seq_cst (the rule for
    // std::atomic::load; release and acq_rel are undefined for a load and
    // libstdc++ asserts on them) -- asserted here, on entry, whatever the word
    // holds. (consume is accepted because std::atomic::load accepts it; it is
    // deprecated in C++26 and treated as acquire by the compilers in use.)
    // `order` is applied to the FIRST read of the word; the protocol's reloads
    // (steps 4, 5, 6) are acquire regardless. A null or marked null is returned
    // from whichever read produced it -- the first read, or an acquire reload
    // when the word became null during the protocol; a non-null word is always
    // returned from an acquire reload. With acquire or seq_cst the call
    // synchronizes with the store()/CAS that published the returned word, null
    // words included. seq_cst buys no more than that: the returned non-null
    // word comes from an acquire reload, so the call does not take part in the
    // single total order of seq_cst operations the way one std::atomic load
    // would; code needing that order across different atomics must not rely
    // on load(seq_cst) here. relaxed and consume promise only what they say
    // for a null returned from the first read.
    //
    // Guarantee: strong. The only throw is std::bad_alloc from
    // make_hazard_pointer() on a thread-cache miss -- normally a thread's first
    // load, before any side effect; on a thread whose hazard cache is closed
    // (overview, "Deferred reclamation") also on a retry after a step-6
    // release. In every case the counts balance and nothing the caller holds
    // has changed. Lock-free after a thread's first call (overview). Count
    // traffic: exactly one TryAddRef on the returned pointee; on a
    // re-validation retry also one release of the reference just taken (which
    // may retire the object and run a scan on this thread) -- the hazard
    // record is returned to the thread cache before that release.
    [[nodiscard]] shared_ptr_type load(std::memory_order order = std::memory_order_seq_cst) const {
        require_pointee();
        assert(is_load_order(order));
        uintptr_t word = aptr_.load(order);                     // step 1
        while (true) {
            U* x = unmarked_ptr(word);
            if (!x) return adopt_word(word);    // null or marked null: no count, no hazard
            // The reference to give back if the re-validation fails. Set
            // inside the hazard scope, used after it: the record must be back
            // in the thread cache before the release, so that a scan triggered
            // by it (and the load() calls the deleters make) finds the record
            // and allocates nothing.
            U* revalidation_victim = nullptr;
            {
                // May throw (bad_alloc on a cache miss): on the first round nothing
                // has been done yet; on a retry after a step-6 failure the released
                // reference stays released -- counts balanced either way (see the doc).
                std::hazard_pointer hp = std::make_hazard_pointer();
                while (true) {
                    hp.reset_protection(x);                     // step 2: release store, typed U*
                    p1202::asymmetric_thread_fence_light();                // step 3
                    uintptr_t reloaded = aptr_.load(std::memory_order_acquire); // step 4
                    if (unmarked_ptr(reloaded) != x) {
                        INTR_HP_COUNT_RETRY_(validation_mismatch);
                        word = reloaded;
                        x = unmarked_ptr(word);
                        if (!x) return adopt_word(word);                   // ~hp clears the record
                        continue;                               // re-aim the record at the new x
                    } // the word moved before the hazard was visible
                    if (x->TryAddRef()) break;                  // step 5: the count pins X now
                    INTR_HP_COUNT_RETRY_(zero_count);
                    // The acquire read of 0 synchronizes with the DelRef that
                    // produced it, which follows the store that unpublished X:
                    // this reload sees a different word. Acquire here because
                    // the word it returns (possibly null, returned below) may
                    // come from a later store than that one, and load(acquire)
                    // must synchronize with whoever published what it returns.
                    word = aptr_.load(std::memory_order_acquire);
                    x = unmarked_ptr(word);
                    if (!x) return adopt_word(word);
                } // protect-and-validate loop, one hazard record re-aimed on every retry
                word = aptr_.load(std::memory_order_acquire);              // step 6
                if (unmarked_ptr(word) == x) return adopt_word(word);   // step 7: re-validated mark
                INTR_HP_COUNT_RETRY_(revalidation_mismatch);
                revalidation_victim = x;
            } // ~hp: the record is cleared (release) and returned to the thread cache
            release(revalidation_victim);       // may retire X and run a scan here
        } // retry with the re-validated word
    } // load()

    // Atomically replace the held value by `desired`'s raw word (pointer and
    // mark), then release the previously held pointee (retire on 1->0).
    // `desired`'s pointee is AddRef'd for the word BEFORE the word is written
    // (the word's reference must be counted before any loader can find it: an
    // uncounted published reference can be released by a concurrent store()
    // and the object retired while the caller still holds it -- model checked,
    // dies); `desired` itself keeps its reference until the parameter dies at
    // return. Postcondition: a subsequent load() returns a value equal to the
    // `desired` passed in (same pointer, same mark) until the next store/CAS;
    // the old pointee's count dropped by one.
    //
    // order: MUST be relaxed, release or seq_cst (the rule for
    // std::atomic::store; asserted because the exchange used internally would
    // silently accept more). The publishing exchange runs with at least
    // acq_rel regardless (publish_order()): release so that a loader that
    // validates the word sees the pointee's construction, acquire so that the
    // release of the old occupant below synchronizes with whoever published
    // it.
    //
    // Nothrow in effect (no allocation; the hooks are noexcept). A retire here
    // may run a scan on this thread.
    void store(shared_ptr_type desired, std::memory_order order = std::memory_order_seq_cst) {
        require_pointee();
        assert(order == std::memory_order_relaxed || order == std::memory_order_release ||
               order == std::memory_order_seq_cst);
        U* new_unmarked = desired.get_unmarked_ptr();
        if (new_unmarked) new_unmarked->AddRef();   // the word's reference, counted first
        uintptr_t old_word = aptr_.exchange(reinterpret_cast<uintptr_t>(desired.get_raw()),
                                            publish_order(order));
        release(unmarked_ptr(old_word));                 // released only after it is unpublished
    } // store()

    // Compare-and-swap on the FULL value (pointer and mark): succeeds iff the
    // word equals expected.get_raw() exactly.
    //
    // Success (returns true): the atomic holds `desired`'s raw word; the word's
    // new reference was counted before publication; the old pointee's
    // reference is released after it (retire on 1->0, possibly a scan on this
    // thread); `expected` is UNTOUCHED (it still owns its reference to the old
    // pointee, which therefore stays alive).
    //
    // Failure (returns false): the atomic is unchanged; `expected` is assigned
    // the current value with a LIVE reference (null if the word is null, mark
    // as currently stored); `desired` is unchanged. Std semantics: this
    // function NEVER returns false with `expected` equal (pointer and mark) to
    // the value the caller passed in -- if the word changed and changed back
    // (including a marked original), it retries internally until it either
    // succeeds or observes a genuinely different value. Hence the idiom
    //   do { ... } while (!a.compare_exchange_strong(expected, desired));
    // makes progress on every failed iteration.
    //
    // Aliasing: `expected` and `desired` may be the same object
    // (`a.compare_exchange_strong(e, e)`): the undo of `desired`'s pre-count
    // goes through a pointer captured before `expected` is reassigned; the
    // counts balance. `desired` is const&: the caller's reference keeps its
    // pointee alive across the call, so the failure-path release of the
    // pre-count never reaches 0.
    //
    // Orders: `success` may be any of the six; `failure` MUST be relaxed,
    // consume, acquire or seq_cst (std's rule; asserted ON ENTRY, so the
    // assert fires on the success path too -- the same set load() accepts,
    // because the failure refresh IS a load; consume is deprecated in C++26,
    // see load()). Internally the word CAS uses `success` promoted to at least
    // acq_rel (publish_order(): release for the loaders of the new word,
    // acquire for this thread's release of the old occupant) and acquire on
    // failure; on failure `expected` is refreshed through load(failure).
    //
    // Guarantee: strong. The only throw is the one load() can throw, on the
    // failure path, after the pre-count of `desired` has been undone and
    // before `expected` is assigned: on a throw the atomic is unchanged, all
    // counts balance, `expected` still equals the caller's original value
    // (possibly a refreshed copy of it from an earlier changed-back
    // iteration), `desired` is unchanged. Never throws once this thread has a
    // cached hazard record.
    //
    // Not [[nodiscard]]: LockFreeList::erase_after() ignores the result of its
    // best-effort unlink CAS, and -Werror would reject it.
    bool compare_exchange_strong(shared_ptr_type& expected, const shared_ptr_type& desired,
                                 std::memory_order success = std::memory_order_seq_cst,
                                 std::memory_order failure = std::memory_order_seq_cst) {
        return compare_exchange(expected, desired, success, failure, true);
    }

    // Weak compare-and-swap: ONE attempt at the word, then return. Not part of
    // the AtomicSharedPtr concept (an optional policy member); LockFreeList
    // does not use it.
    //
    // Success (returns true): exactly as compare_exchange_strong's success.
    // Failure (returns false): the atomic is unchanged; `desired`'s pre-count
    // is undone; `expected` is refreshed ONCE through load(failure) with a
    // live reference. MAY FAIL SPURIOUSLY: a false return with the refreshed
    // `expected` equal (pointer and mark) to the caller's original is allowed
    // (std semantics for a weak CAS). Exactly when it happens here: the one
    // attempt is a compare_exchange_strong on the word, so it fails ONLY if
    // the word differed from `expected` at that instant; the refresh then
    // reads the word again, and if another thread changed it back in between,
    // the refreshed `expected` equals the original and the call still returns
    // false (where compare_exchange_strong would retry and succeed) -- a
    // deliberate single-attempt semantics. Consequently, without concurrent
    // modification of this word (single-threaded, or the word quiescent) a
    // weak CAS with a matching `expected` succeeds, and a mismatching one
    // fails with a genuinely different refreshed `expected`: no spurious
    // failure is possible then. Callers must still loop:
    //   while (!a.compare_exchange_weak(expected, desired)) { /* re-derive desired */ }
    // Aliasing, orders and the exception guarantee: as compare_exchange_strong.
    //
    // [[nodiscard]]: ignoring a weak CAS's result is always a bug.
    [[nodiscard]] bool compare_exchange_weak(
            shared_ptr_type& expected, const shared_ptr_type& desired,
            std::memory_order success = std::memory_order_seq_cst,
            std::memory_order failure = std::memory_order_seq_cst) {
        return compare_exchange(expected, desired, success, failure, false);
    }

private:
    // The entire atomic state in one word: U* | mark (bit 0). Not `mutable`:
    // unlike the spinlock pointer, load() never writes the word, so a const
    // load() needs no mutable member.
    std::atomic<uintptr_t> aptr_;

    // The pointee check, called at the top of every member that can first
    // produce a non-null handle or first touch a hook on a word it reads --
    // load, store, both CASes, the destructor, the adopting constructor, and
    // shared_ptr_type's adopting and copy constructors. (The other hook users,
    // shared_ptr_type's assignments, destructor and use_count() and release(),
    // only operate on handles one of those produced, so the check has already
    // fired by the time they are instantiated with a bad U.) A static_assert in a
    // member BODY is evaluated when that body is instantiated, i.e. only once
    // U is complete; the class definition itself (instantiated by
    // `AtomicPtr<Node> next;` while Node is incomplete) is unconstrained.
    static constexpr void require_pointee() noexcept {
        static_assert(HpIntrusivePointee<U>,
            "intr_shared_ptr_hp<T, U>: U must derive publicly from std::hazard_pointer_obj_base<U> "
            "and provide noexcept AddRef(), bool TryAddRef(), bool DelRef(), "
            "long use_count() const");
        static_assert(std::is_convertible_v<U*, T*>,
            "intr_shared_ptr_hp<T, U>: U* must convert implicitly to T*");
    } // require_pointee()

    // The word with the mark bit cleared, as the pointer it is: safe to
    // dereference and to pass to the hooks. Null for null and marked null.
    static U* unmarked_ptr(uintptr_t word) noexcept {
        return reinterpret_cast<U*>(word & ~1ULL);
    }

    // Wrap a raw word (pointer and mark) in a shared_ptr_type WITHOUT touching
    // the count: the reference it will own was already taken (TryAddRef in
    // load()) or does not exist (null, marked null).
    static shared_ptr_type adopt_word(uintptr_t word) noexcept {
        shared_ptr_type res;
        res.p_ = reinterpret_cast<U*>(word);
        return res;
    }

    // Release one reference: DelRef; on 1->0 retire() (never delete -- another
    // thread may hold a hazard on the object and be about to read its count).
    // Null is a no-op. The one place reclamation is triggered from; the retire
    // may cross mm_hp's threshold and run a scan on the calling thread.
    // noexcept: DelRef and retire() are.
    static void release(U* unmarked) noexcept {
        if (unmarked && unmarked->DelRef()) unmarked->retire();
    }

    // The set std::atomic::load accepts; load()'s `order` and the CASes'
    // `failure` are asserted against it.
    static bool is_load_order(std::memory_order order) noexcept {
        return order == std::memory_order_relaxed || order == std::memory_order_consume ||
               order == std::memory_order_acquire || order == std::memory_order_seq_cst;
    }

    // The order of a publishing RMW (store()'s exchange, the CASes' success):
    // the caller's order promoted to at least acq_rel. Both halves are needed.
    // Release: a loader whose acquire validation reload returns the new word
    // must see the pointee's construction and whatever the publisher wrote
    // before. Acquire: the same RMW READS the old word and the caller then
    // releases the old occupant -- a DelRef on its count and, on 1->0, a
    // retire that may run its destructor -- so it must synchronize with the
    // publisher of the OLD occupant, or it touches an object whose construction
    // it has no happens-before with (TSan demonstrates it: with a release-only
    // exchange, a storer's DelRef races with the allocation and construction of
    // the object another thread had published through a CAS -- a data race on
    // the count's initialization, and through the retire chain on every plain
    // field the destructor writes). intr_shared_ptr had this acquire for free
    // from its lock CAS. Cost: nil on x86-64 (every locked RMW is a full
    // barrier); LDAXR instead of LDXR on aarch64. Only seq_cst is stronger.
    static std::memory_order publish_order(std::memory_order order) noexcept {
        return order == std::memory_order_seq_cst ? std::memory_order_seq_cst
                                                  : std::memory_order_acq_rel;
    }

    // The shared body of the two CASes; `strong` selects the changed-back retry.
    //
    // Flow: capture the caller's original word (`original`) and `desired`'s
    // unmarked pointer `d` BEFORE anything touches `expected` (they may alias:
    // cas(e, e)); AddRef d for the word (the word's reference is counted
    // before it can be published); one CAS on the word with the promoted
    // success order and acquire on failure. Success: release the old occupant
    // -- after it is unpublished -- and return true. Failure: undo d's
    // pre-count through the captured pointer (it cannot reach 0: the caller's
    // `desired` is one owner), refresh `expected` through load(failure) (the
    // hazard record of that load is back in the cache before `expected` is
    // assigned, so the release of expected's old pointee inside the assignment
    // can run a scan without allocating a record), then: weak returns false;
    // strong returns false only if the refreshed raw word differs from the
    // original (pointer or mark) and otherwise retries, because a strong CAS
    // must never report failure with an unchanged `expected`.
    //
    // The failed CAS's reported word is discarded: a reference to it can only
    // be taken through the load protocol, so the failure order of the word CAS
    // (acquire) is not load-bearing.
    bool compare_exchange(shared_ptr_type& expected, const shared_ptr_type& desired,
                          std::memory_order success, std::memory_order failure, bool strong) {
        require_pointee();
        assert(is_load_order(failure));
        const uintptr_t original = reinterpret_cast<uintptr_t>(expected.get_raw());
        const uintptr_t new_word = reinterpret_cast<uintptr_t>(desired.get_raw());
        U* const d = desired.get_unmarked_ptr();    // captured before `expected` can change
        const std::memory_order cas_order = publish_order(success);
        while (true) {
            if (d) d->AddRef();                              // the word's reference, counted first
            uintptr_t observed = original;
            if (aptr_.compare_exchange_strong(observed, new_word, cas_order,
                                              std::memory_order_acquire)) {
                release(unmarked_ptr(observed));    // == original; unpublished, now released
                return true;
            } // the word was swapped
            release(d);                                      // undo the pre-count; never 0 here
            shared_ptr_type current = load(failure);    // may throw: counts balanced, no change
            const uintptr_t refreshed = reinterpret_cast<uintptr_t>(current.get_raw());
            expected = std::move(current);              // `d`, `new_word` stay valid if aliased
            if (!strong || refreshed != original) return false;
            // Changed and changed back (same pointer AND mark): retry, so the
            // caller never sees a failure with an unchanged `expected`.
        } // CAS attempt loop (one iteration for weak)
    } // compare_exchange()
}; // class intr_shared_ptr_hp

#undef INTR_HP_COUNT_RETRY_

#endif // INCLUDED_INTR_SHARED_PTR_HP_H_
