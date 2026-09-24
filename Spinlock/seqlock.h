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
#ifndef INCLUDED_SEQLOCK_H
#define INCLUDED_SEQLOCK_H
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <functional>
#include <mutex>
#include <type_traits>
#include <utility>

#include "spinlock.h"
#include "spinlock_tune.h"              // spin_pause()

// A sequence lock: one payload of type T that any number of readers copy out
// WITHOUT writing to shared memory, and any number of writers replace under
// mutual exclusion.
//
// Overall flow. A sequence counter names the payload's generation: the counter
// is 2k while generation k is in place and 2k+1 while generation k+1 is being
// written. A writer takes the embedded SpinLock (writers exclude each other;
// readers never touch the lock), makes the counter odd, stores the payload,
// makes the counter even again, and unlocks. A reader loads the counter, loads
// the payload, loads the counter again, and accepts the payload only if the
// two counter values are the same EVEN number; otherwise it pauses and
// retries. The invariant everything below serves is
//
//     a reader that validates counter 2k returns exactly generation k:
//     not an older payload, and not the payload of a write still in flight.
//
// The reader's cost is three loads and no stores, so readers do not contend
// with each other at all: the counter's cache line stays Shared among them.
//
// Progress. Writers are serialized by SpinLock and inherit its properties.
// Readers are NOT lock-free: a reader retries whenever a write overlaps its
// read, so a steady stream of writers can starve readers indefinitely (each
// retry can find yet another write in flight), and a writer descheduled with
// the counter odd stalls every reader until it runs again. A sequence lock is
// for read-mostly data with short writes. Readers wait with spin_pause() only
// -- no yield, no sleep ladder -- because the expected wait is the few
// instructions of a writer's odd window.
//
// The payload is one machine word held in a std::atomic<T>, so that loading it
// while a writer stores it is not a data race. With a single word that leaves
// the counter with nothing to protect against tearing; the protocol is
// nevertheless implemented, and its orderings are chosen, exactly as the
// invariant above requires, since it is the protocol that is being
// demonstrated and measured. A production multi-word version would copy the
// payload in and out memcpy-style for speed; see the note at the end of the
// memory-ordering contract for what that does to the orderings.
//
// Memory-ordering contract (a release store happens-before an acquire load of
// the SAME atomic that observes it or a later value; relaxed operations create
// no edge, but must respect the edges that exist: a load can never return a
// value overwritten by a store that happens-before it):
//   - Writer to writer: SpinLock. unlock() is a release store and the winning
//     exchange an acquire, so everything one writer did to seq_ and payload_
//     happens-before the next writer's critical section. Writers therefore
//     read both with relaxed loads, and -- being the only thread that can
//     modify the counter -- compute its next values locally: no atomic
//     read-modify-write on seq_, ever.
//   - "Not older than generation k": the closing store of 2k is a release, and
//     the reader's FIRST counter load an acquire. Observing 2k puts all of
//     write k, its payload store included, before the reader's payload load,
//     which then cannot return generation k-1 or earlier.
//   - "Not newer than generation k": this edge has to be carried by the
//     PAYLOAD -- payload store release, payload load acquire -- and cannot be
//     carried by the counter. What validation relies on is: "if my payload
//     load observed write k+1, my second counter load observes 2k+1 or later".
//     The writer sequences the odd store before the payload store, the reader
//     sequences the payload load before the second counter load; the missing
//     link is payload store -> payload load, and only a release/acquire pair
//     on the payload itself supplies it. With that link in place the odd store
//     and the second counter load are both plain relaxed operations: the chain
//     odd store -> payload store -> payload load -> second load already
//     forbids the second load from returning the overwritten 2k.
//     The tempting alternative -- relaxed payload, release on the odd store,
//     acquire on the second load -- has the same number of barriers and is NOT
//     sufficient: an acquire load synchronizes only with the store it
//     observes, so a second load that observes the stale 2k gains no edge from
//     the odd store, and nothing prevents it from observing the stale 2k. The
//     claim "saw the new payload, hence sees the odd counter" would be
//     circular. It holds on x86 (TSO delivers one thread's stores in order,
//     and a load never returns a value older than one an earlier load of the
//     same thread returned) and fails on weakly ordered machines, in two
//     independent ways:
//       * writer side: the payload store may become visible to the reader
//         before the odd store that precedes it in the program does;
//       * reader side, even if the writer's stores do become visible in
//         program order: nothing orders the reader's relaxed payload load
//         against its own second counter load, and on aarch64 a plain load
//         and the LDAR that follows it may execute out of order. The second
//         counter load runs early and returns 2k; the writer then makes the
//         counter odd and stores generation k+1; the payload load runs last
//         and returns generation k+1. The reader validates 2k and returns
//         generation k+1.
//   - What needs what. The PROTOCOL invariant above needs both pairs: "not
//     older" stands on the counter -> payload edge (closing store release,
//     first load acquire), "not newer" on the payload -> counter edge (payload
//     store release, payload load acquire). The user-visible contract of
//     load() is weaker -- some complete write's value, with that write
//     happening-before the return -- and with a one-word payload the payload
//     pair alone delivers it: a single word cannot be torn, and the edge comes
//     from the payload load itself. Without the counter pair a reader could
//     validate 2k and return generation k-1, and no caller could tell. This is
//     the same one-word degeneracy as above: the counter pair is there for
//     the protocol, not for anything a user of SeqLock<one word> can observe.
//   - Net cost: two release stores per write, two acquire loads per read; all
//     other operations are relaxed. On x86 every one of them is a plain MOV.
//   - The alternative with the same barrier count, recorded here and NOT
//     implemented: leave every access to the payload relaxed and order it
//     against the counter with standalone fences, as the Linux seqcount does
//     with smp_wmb/smp_rmb.
//         writer: odd store (relaxed), std::atomic_thread_fence(release),
//                 payload store (relaxed), closing store (release);
//         reader: first load (acquire), payload load (relaxed),
//                 std::atomic_thread_fence(acquire), second load (relaxed).
//     Two barriers per write and two per read, as here. It is the form a
//     multi-word memcpy-style payload needs: such a payload cannot use the
//     payload edge (that would take a release/acquire on every word, and a
//     plain memcpy has none), so the ordering between the odd store and the
//     payload stores, and between the payload loads and the second counter
//     load, has to come from fences. On aarch64 a fence lowers to a DMB where
//     the present code has an STLR or an LDAR; which of the two is cheaper on
//     the machines measured here (Grace, M3) has not been measured.
template <typename T>
class SeqLock {
  // std::atomic<T> requires trivially copyable T; stated here for a readable
  // diagnostic. Always-lock-free is the real constraint: a std::atomic that
  // falls back to an internal lock would make readers block on writers (and
  // write to shared memory), which is everything a sequence lock exists to
  // avoid. In practice this limits T to one machine word (or two, where the
  // target has a double-word atomic).
  static_assert(std::is_trivially_copyable_v<T>,
                "SeqLock payload must be trivially copyable");
  static_assert(std::atomic<T>::is_always_lock_free,
                "SeqLock payload must fit a lock-free std::atomic");
  // Same constraint on the counter; holds on every 64-bit target.
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                "SeqLock counter must fit a lock-free std::atomic");

  public:
  // Construct holding a value-initialized payload (generation 0). T{} is
  // evaluated inside this constructor, so it is what decides noexcept; the
  // rest cannot throw (see below).
  SeqLock() noexcept(noexcept(T{})) : SeqLock(T{}) {}

  // Construct holding `initial` as generation 0. Making the SeqLock object
  // itself visible to other threads is the caller's responsibility, as for any
  // object. Unconditionally noexcept: `initial` is constructed by the caller,
  // outside this function, and copying a trivially copyable T into the
  // std::atomic cannot throw.
  //
  // The layout is pinned here rather than at class scope because offsetof
  // needs a complete type, and a class is complete inside its member function
  // bodies but not inside its own member-specification; every SeqLock that is
  // ever constructed passes through this body. offsetof is only
  // conditionally-supported (and warns under -pedantic) for types that are not
  // standard-layout, so that is asserted first. All data members are private
  // (one access level), which is SeqLock's own part of the requirement; the
  // members must be standard-layout too, and that the standard guarantees
  // only for the integral, pointer and floating-point specializations of
  // std::atomic. For std::atomic<a user's struct> it is up to the library,
  // and this assert is what catches a library that says no.
  explicit SeqLock(T initial) noexcept : payload_(initial) {
    static_assert(std::is_standard_layout_v<SeqLock>);
    static_assert(alignof(SeqLock) == 64);
    static_assert(offsetof(SeqLock, seq_) == 0);
    static_assert(offsetof(SeqLock, lock_) < 64);      // same line as the counter
    static_assert(offsetof(SeqLock, payload_) == 64);  // payload heads the next line
    static_assert(sizeof(SeqLock) == 128);
  } // SeqLock::SeqLock()

  SeqLock(const SeqLock&) = delete;
  SeqLock& operator=(const SeqLock&) = delete;

  // Return a consistent snapshot of the payload: the value of one complete
  // write (or the initial value), never that of a write in flight. The write
  // that produced the value, and everything its thread did before it
  // (including the side effects of update()'s callable), happens-before the
  // return. Never writes to shared memory. Blocks -- spinning with
  // spin_pause() -- while a write is in flight, and retries whenever a write
  // overlapped the attempt, so it is not lock-free and writers can starve it
  // (see the class comment). Returns by value: there is no stable object to
  // refer to. The protocol itself, with its ordering comments, is in
  // load_validated() below; this drops the counter value it also returns.
  T load() const {
    return load_validated().value;
  } // SeqLock::load()

  // Replace the payload with `value`, excluding other writers. By value, not
  // by const reference: T is constrained to a machine word, which travels in a
  // register; a reference would add an indirection and an aliasing question
  // (a reference into storage another thread is writing) for nothing. Blocks
  // in SpinLock::lock() while another writer holds the lock.
  void store(T value) {
    std::lock_guard guard(lock_);
    write(value);
  } // SeqLock::store()

  // Read-modify-write of the payload under writer exclusion: replaces the
  // payload with f(current payload), atomically with respect to all other
  // store()/update() calls, so no update is lost.
  //
  // `f` is called exactly once, as std::invoke(std::forward<F>(f), current),
  // where `current` is a prvalue T: a COPY of the payload (there is no T
  // object to hand out a reference to -- the payload lives in a std::atomic),
  // and it is the latest value, since only lock holders change it. f must
  // return the new payload, as a T or something implicitly convertible to T.
  // The static_assert below states exactly this call: std::is_invocable_r
  // is defined in terms of std::invoke, with F's value category preserved
  // and a T rvalue as the argument.
  //
  // f runs with the writer lock held but BEFORE the counter goes odd:
  // concurrent readers keep succeeding (returning the old value) however long
  // f takes; only other writers wait. Keep f short anyway -- it is inside a
  // spinlock. f must not call store() or update() on this SeqLock (SpinLock is
  // not recursive: self-deadlock). If f throws, the lock is released and the
  // payload and the counter are untouched. Whatever f writes elsewhere
  // happens-before any load() that returns the value f computed.
  template <typename F>
  void update(F&& f) {
    static_assert(std::is_invocable_r_v<T, F, T>,
                  "SeqLock::update(f): f must be callable as T f(T)");
    std::lock_guard guard(lock_);
    // relaxed: the previous writer's payload store happens-before this load
    // through the SpinLock release/acquire handoff, so a relaxed load cannot
    // return anything but the latest payload. The loaded prvalue goes straight
    // to f, and f has returned (or thrown) before write() touches the counter.
    write(std::invoke(std::forward<F>(f), payload_.load(std::memory_order_relaxed)));
  } // SeqLock::update()

  private:
  // Test-only access (seqlock_test.C defines it). The peer can play a writer
  // that holds the counter odd, which no sequence of public calls can do, and
  // can see which counter value a load validated. It does both by calling the
  // private steps below -- the very ones load() and write() are made of -- so
  // that what the tests exercise is this header's protocol and not a copy.
  friend struct SeqLockTestPeer;

  // What one successful read attempt produced: the payload and the (even)
  // counter value that validated it. The protocol invariant says `value` is
  // exactly generation seq/2.
  struct Snapshot {
    T value;                            // the payload returned to the caller
    std::uint64_t seq;                  // the counter value, 2k, both loads agreed on
  }; // struct Snapshot

  // The whole reader protocol; load() is this, minus the counter. Spins until
  // an attempt validates, and returns that attempt's payload together with
  // the counter value it validated. Never writes to shared memory.
  Snapshot load_validated() const {
    for (;;) {
      // acquire: pairs with the writer's closing release store. Observing 2k
      // makes write k -- payload store included -- happen-before the payload
      // load below, so that load cannot return anything older than generation
      // k. (An odd value may be observed too; it creates no obligation, since
      // the attempt is abandoned.) Free on TSO.
      const std::uint64_t seq_before = seq_.load(std::memory_order_acquire);
      if ((seq_before & 1) == 0) {
        // acquire: pairs with the writer's release store of the payload. If
        // this load observes a write newer than seq_before, that write's odd
        // counter store -- sequenced before its payload store -- now
        // happens-before the validating load below. This is the edge the
        // validation stands on; see the class contract for why it cannot live
        // on the counter. Free on TSO.
        const T value = payload_.load(std::memory_order_acquire);
        // relaxed: needs no edge of its own. It is sequenced after the
        // payload load, so if that load saw a newer write, the chain odd store
        // -> payload store -> payload load -> here already forbids this load
        // from returning the overwritten seq_before. If the payload load saw
        // generation seq_before/2, any value this load returns is a correct
        // verdict: equal accepts a correct snapshot, different costs a retry.
        if (seq_.load(std::memory_order_relaxed) == seq_before) return {value, seq_before};
      } // if no write was in flight at the first counter load
      spin_pause();
    } // retry until consistent snapshot
  } // SeqLock::load_validated()

  // The writer protocol, in its three steps. Common precondition: the caller
  // holds lock_, which is what makes the plain (non-RMW) counter arithmetic
  // correct. All three are noexcept and straight-line: once the counter is
  // odd nothing can prevent it from becoming even again. write() is the only
  // caller in this header; the test peer calls the steps one at a time, to
  // stop between them.

  // Step 1: open the odd window. Returns the odd value now in the counter,
  // which the caller hands to end_write().
  std::uint64_t begin_write() noexcept {
    // relaxed: same argument as the payload load in update() -- the previous
    // writer's closing store reaches us through the SpinLock handoff. This
    // thread is the counter's only writer until unlock, so both values stored
    // during this write are computed from this one load; no fetch_add.
    const std::uint64_t seq = seq_.load(std::memory_order_relaxed);
    // relaxed: opens the odd window. It needs no edge of its own: it is
    // sequenced before the payload store in step 2, whose release carries it
    // to any reader that observes the new payload. A reader that does not
    // observe the new payload does not need to see the odd value.
    seq_.store(seq + 1, std::memory_order_relaxed);
    return seq + 1;
  } // SeqLock::begin_write()

  // Step 2: put `value` in place. Precondition: the counter is odd (between
  // begin_write() and end_write()). One call per write, except for the test
  // peer, which stores an intermediate value first.
  void store_payload(T value) noexcept {
    // release: pairs with the acquire payload load in load_validated().
    // Publishes the odd store of step 1 (and, for update(), the side effects
    // of f) to a reader that observes this value, forcing that reader's
    // validation to fail until the closing store is visible as well. Free on
    // TSO.
    payload_.store(value, std::memory_order_release);
  } // SeqLock::store_payload()

  // Step 3: close the odd window. `odd_seq` is what begin_write() returned.
  void end_write(std::uint64_t odd_seq) noexcept {
    // release: pairs with the first (acquire) counter load in
    // load_validated(). A reader that observes this even value has the whole
    // write, payload included, happen-before its payload load. Free on TSO.
    seq_.store(odd_seq + 1, std::memory_order_release);
  } // SeqLock::end_write()

  // One write: counter odd, payload, counter even. Precondition: the caller
  // holds lock_. The order of the three steps IS the protocol.
  void write(T value) noexcept {
    const std::uint64_t odd_seq = begin_write();
    store_payload(value);
    end_write(odd_seq);
  } // SeqLock::write()

  // Layout. Two cache lines: {seq_, lock_} and {payload_}. What this buys is
  // that the payload's line is dirtied exactly once per write and is left
  // alone by the lock traffic. What it does NOT buy is readers isolated from
  // that traffic: every load() polls seq_, twice, and seq_ shares its line
  // with lock_, so each write invalidates the line readers depend on up to
  // four times (lock exchange, odd store, closing store, unlock), and a
  // reader touches two lines per load. The alternative -- {seq_, payload_} on
  // one line, so that a reader touches one line and sees two invalidations
  // per write, with lock_ alone on the next -- has not been measured.
  // The alignment is hard-coded to 64. Where cache lines are 128 bytes (Apple
  // M3) counter, lock and payload can all share one line -- and a SeqLock can
  // share a line with its neighbours -- so none of the above applies.

  // Generation counter: 2k while generation k is in place, 2k+1 while
  // generation k+1 is being written. Modified only under lock_; read by every
  // reader. 64 bits on every target, so it cannot wrap in any realistic run
  // (2^63 writes).
  alignas(64) std::atomic<std::uint64_t> seq_ {0};
  // Writer-to-writer mutual exclusion, and the happens-before edge from one
  // writer's critical section to the next. Readers never touch it -- but
  // they do share its cache line; see the layout note above.
  SpinLock lock_;
  // The payload, at the head of the next cache line. Atomic so that a reader's
  // load concurrent with a writer's store is not a data race.
  alignas(64) std::atomic<T> payload_;
}; // class SeqLock

#endif // INCLUDED_SEQLOCK_H
