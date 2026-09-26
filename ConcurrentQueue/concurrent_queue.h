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

#ifndef RING_ATOMIC_QUEUE_H_
#define RING_ATOMIC_QUEUE_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <type_traits>
#include <utility>
#include <time.h>
#include <bit>
#include <mutex>
#include "spinlock.h"

// ---------------------------------------------------------------------------
// Simple CHECK macros (abort on failure, no messages).
// ---------------------------------------------------------------------------
#define CHECK(c) if (!(c)) std::abort();
#define CHECK_GE(a, b) if ((a) < (b)) std::abort();
#define CHECK_EQ(a, b) if ((a) != (b)) std::abort();

// ---------------------------------------------------------------------------
// RING_QUEUE_HOOK(point): TEST-ONLY interleaving hook. Expands to nothing
// unless the includer defined it before including this header.
//
// The key-value push()/pop() below invoke it at the boundaries between the
// steps of the slot protocol. The point names are the arguments at the call
// sites: PUSH_LOADED, PUSH_CLAIMED (key written, tail advanced, unlocked),
// PUSH_COMMITTING, POP_LOADED, POP_CLAIMED (key read and cleared, head
// advanced, unlocked) and POP_RELEASING. A test defines the macro to call a
// function that parks the calling thread at a chosen point until the test
// says go; that turns one specific interleaving of several threads -- the
// kind that a stress test reaches once in a billion runs, if ever -- into a
// deterministic test. concurrent_queue_test.C does exactly this for the
// interleavings that broke the previous version of the protocol.
//
// With the macro undefined (every build but that test) each call site
// expands to a null statement -- a bare `;` -- which has no effect on the
// program and nothing for the compiler to generate, so the hook points are
// free by construction, not by optimization. The default definition is
// header-private: it is #undef'd at the end of the header so it does not
// leak into includers.
// ---------------------------------------------------------------------------
#ifndef RING_QUEUE_HOOK
#define RING_QUEUE_HOOK(point)
#define RING_QUEUE_HOOK_DEFAULT_
#endif


// ---------------------------------------------------------------------------
// RingAtomicMapQueueMPMC
// ---------------------------------------------------------------------------

// Key and Value together form the element stored in the queue. The Key type
// has to support lock-free atomic operations; this implies that it is
// trivially copyable and destructible. The value type can be any type; the
// default (void) means that the queue contains only keys.
//
// Key{} (the default/zero key) is reserved: pop() returns it to report an
// empty queue, so an element with key Key{} would be indistinguishable from
// "nothing there" to pop()'s caller, and it must never be pushed. In the
// key-only queue the reservation is also structural: the key is the slot's
// state word and Key{} is its "empty slot" value, so pushing Key{} advances
// the tail but leaves a slot no consumer can ever pop past, stranding every
// element behind it. Both push() overloads enforce the rule with a debug-only
// assert: it is always a bug in the caller, but correct callers should not
// pay for the check.
//
// The queue has no exceptions in it, by requirement rather than by luck: every
// operation it performs on the element types must be nothrow, and the compiler
// enforces it. Whatever makes every push() or every pop() impossible is
// refused at the class, not at the first call: a Key that is not
// nothrow-copyable, and a Value that is not nothrow-destructible (every popped
// value is destroyed in its slot). What depends on the call stays on the call:
// push() accepts only arguments from which Value is nothrow-constructible (a
// std::string rvalue is fine; a std::string lvalue, whose copy can throw, is
// rejected), and pop() only an object that Value can be nothrow-assigned into.
// Value itself need not be movable: a std::atomic<int> value is pushed from an
// int and popped into one. push() and pop() are noexcept to say so in the
// interface. This is deliberate: exception friendliness is what forces
// std::queue into a front() that returns the element and a pop() that does not
// -- the split that makes it unusable from more than one thread -- and a
// concurrent queue cannot pay that price. There is no throwing path left to
// recover from, so there is no recovery code either. Nor does the requirement
// cost real users anything: a high-throughput queue sits on a fast path, where
// memory allocation is not allowed anyway. A value that owns memory is built
// (and allocates) on the caller's side, before the push, and is moved in --
// which is exactly what push() accepts.
//
// Unlike a classic ring buffer, the head and tail indices are never compared
// to each other: full/empty is decided per-slot, from the key in the key-only
// queue and from the slot's sequence number in the key-value queue (see the
// slot protocol below). This is why all capacity_ slots are usable, with no
// reserved gap between head and tail.
//
// The key-value slot protocol
// ---------------------------
// Producers are serialized by the tail spinlock and consumers by the head
// spinlock, and the indices are free-running: tail_.i is the index the next
// push takes, head_.i the index the next pop takes, and index i lives in slot
// i & capacity_mask_. So slot p serves the indices p, p + capacity, p +
// 2*capacity, ... -- one per lap of the ring -- and the same slot is visited,
// once per lap, by a producer and then by a consumer, each of which claims
// the slot under its lock but does the expensive part (constructing, or
// moving out and destroying, the value) after releasing the lock. What the
// slot has to tell a thread that reaches it is therefore not just "full or
// empty" but "full or empty FOR THE INDEX YOU ARE AT": a slot that is free
// for index i is not free for index i + capacity until the element of index
// i has been produced and consumed.
//
// Each key-value slot carries a sequence number, seq, for exactly that
// purpose (the scheme is that of Dmitry Vyukov's bounded MPMC queue; here
// the locks take the place of its compare-and-swap on the indices). For the
// index i that a slot currently serves, seq takes three values in order, and
// they are the slot's states:
//
//   seq            state                            who acts, and how
//   ------------   ------------------------------   ---------------------------
//   i              free for index i; or claimed by  the producer at tail index
//                  the producer at index i, value   i: writes the key and
//                  under construction               claims (++tail_.i) under
//                                                   the tail lock, unlocks,
//                                                   constructs the value, then
//                                                   stores seq = i + 1
//   i + 1          committed: key and value valid;  the consumer at head index
//                  or claimed by the consumer at    i: reads and clears the key
//                  index i, value being moved out   and claims (++head_.i)
//                                                   under the head lock,
//                                                   unlocks, moves the value
//                                                   out and destroys it, then
//                                                   stores seq = i + capacity
//   i + capacity   free for index i + capacity:     the first row again, one
//                  the first row of the next lap    lap later
//
// The "or claimed by" halves of the first two rows are not visible in seq at
// all: the claim is the index increment under the lock, and it is enough,
// because claims are handed out in index order and only the thread that
// claimed index i ever acts on the slot for index i. From the claim to the
// final seq store the slot is owned by that thread: no producer touches it
// until seq == i + capacity, no consumer until seq == i + 1, and both are
// stored by the owner as its last act. That is why the value can be
// constructed, or moved and destroyed, outside the lock with no flag guarding
// it -- and why the key could be, too: it is written and read under the lock
// for a reason that has nothing to do with the protocol (see "Why every
// critical section ends with a store to the slot" below).
//
// What a thread that finds the slot in some other state knows, given the
// index invariant tail_.i - head_.i in [0, capacity] (the difference taken
// modulo 2^64; it holds because a consumer claims only committed indices,
// and a producer claims only a slot whose previous element has been
// consumed) and that claims are in index order on each side:
//   * A producer at index i sees seq == i - capacity (the previous lap's
//     producer is still constructing) or i - capacity + 1 (its element is
//     committed and not yet fully consumed). Either way the element of index
//     i - capacity is still in the ring, so all capacity slots are taken: the
//     queue is full, push() returns false. It retries NTRY times first,
//     because the common transient is a consumer that has claimed the slot
//     and is a few instructions from releasing it.
//   * A consumer at index i sees seq == i (nothing committed at the head: the
//     producer at index i has not claimed, or has claimed and is still
//     constructing) or i - capacity + 1 (the consumer of the previous lap is
//     still moving its value out; then the producer at index i cannot have
//     claimed, so tail_.i == head_.i == i). Either way no element of index
//     >= i is committed: the queue is empty, pop() returns Key{}.
//
// Why a sequence number and not a flag: the values of seq that a thread at
// index i compares against -- i, i + 1, i + capacity -- occur exactly once in
// the slot's life, so the transient states of one lap can never be mistaken
// for the states of another. The previous protocol used the key as the state
// word plus a busy flag written by both sides, and validated by re-reading
// the key around the flag; but the key's values recur (Key{} every lap, and
// any user key may repeat), so three loads that straddled a commit and a pop
// could pass the validation on a slot that was never stable, two threads
// would own one slot, and the loser's flag store would clobber the winner's
// claim. Here the state word is monotone per slot and its comparison is
// exact, and the key is plain data published by the seq store like the value.
//
// Wrap-around of the indices: indices and seq are size_t, so all of this is
// arithmetic modulo 2^64. Because capacity is a power of 2 it divides 2^64,
// so the indices that land in slot p -- p, p + capacity, ... -- continue
// seamlessly through the wrap (2^64 - capacity + p is followed by p), and the
// three values compared for index i are distinct from each other (i + 1 and
// i + capacity coincide only for capacity == 1). Across laps, the only value
// shared is the one the table shares on purpose -- i + capacity, lap L's
// exit and lap L + 1's entry -- because the values of different laps differ
// by a nonzero multiple of capacity, or by that plus or minus one, and
// neither is a multiple of 2^64 unless the indices themselves coincide.
// Every comparison is an exact == on size_t; no signed differences are
// computed, so wrap needs no special case.
//
// Why every critical section ends with a store to the slot
// --------------------------------------------------------
// Each push() and pop() that succeeds writes to its slot while it still holds
// the lock. (The failing paths -- full, empty -- store nothing; at the full or
// empty boundary, where they are frequent, the argument below applies only to
// the operations that get through.) The key-only queue stores or clears the
// key there; the key-value queue writes the key (push) or clears it (pop) just
// before the index increment, although its protocol would be exactly as
// correct with every slot write after the unlock -- the key is published and
// ordered by seq either way, and the cleared key is read by nobody. The store
// is there for the lock, not for the slot. Moving it after the unlock is the
// obvious "optimization"; it has been made, measured and reverted twice --
// first in the key-only queue, then again when the key-value protocol was
// rewritten around seq and the store was lost with the busy flag -- and this
// section exists so that nobody steps on that rake a third time.
//
// The mechanism, established with instrumented locks and hybrid variants on
// large Arm and x86 servers: under contention this SpinLock's throughput comes
// from batching. The thread that just released the lock takes it again while
// the waiters sit in the back-off, and nearly every acquisition is such a
// re-acquisition. A store to the slot inside the critical section usually
// misses (the slot's line was last touched on the other side of the ring), and
// the release store of unlock() cannot become visible before it -- stores are
// ordered under x86-TSO, and an Arm release store orders the stores before it
// -- so a waiter sees the lock free only once the holder's slot write is done,
// and by then the holder is ready to take the lock again. Without the store
// the lock shows free while the holder's slot writes are still pending; its
// next acquisition waits for them, and the waiters that notice the free lock
// take it, or at least pull its cache line away. The lock ping-pongs, every
// operation pays cross-core transfers of the lock line and of a cold slot
// line, and the walk along the ring stops being prefetchable. What makes the
// batching regime so productive, and its loss so expensive, is the back-off: a
// waiter that misses the eight-attempt burst is parked for ~50 us (the "short"
// nanosleep is rounded up by the kernel's default 50 us timer slack on
// Linux). The discriminating experiments: a pure delay inside the critical
// section restores the regime but stalls the holder and is slower still; a
// prefetch-for-write of the slot inside the critical section does not help;
// only a store works.
//
// The rule that follows: the store is for the lock. Moving it after the unlock
// is correct and slower. The effect belongs to this lock's back-off -- with a
// fair lock (an MCS lock, say) the batching disappears and every variant
// collapses together -- so a retuned back-off means measuring this again
// before believing any of it. The measurements are in README.md, under
// "Performance".
//
// Note on pop() and "empty": the empty answer is a snapshot of the head slot.
// A producer may have claimed index i and be constructing while another
// producer's push() at index i + 1 has already committed and returned true,
// so "pop() returned Key{}" does not mean "every push() that returned before
// it has been consumed". This is inherent in claiming under the lock and
// committing outside it (it was true of the previous protocol too); callers
// that need a stronger guarantee must synchronize outside the queue.
template <typename Key, typename Value = void,
          size_t NTRY = 8,              // Number of attempts to acquire a slot before giving up
          size_t ALIGN = 0>             // Additional alignment; 64 to align each queue slot on a cache line
class RingAtomicMapQueueMPMC {
  // In the key-only queue the key is the slot's atomic state word (empty/full),
  // so it must be lock-free -- a mutex-backed "atomic" key would defeat the
  // whole per-slot lock-free coordination. (The key-value queue publishes the
  // key through the slot's sequence number and would not need this; the
  // requirement is kept uniform so a Key type works in both modes.)
  static_assert(std::atomic<Key>::is_always_lock_free);
  // In the key-value queue the key is plain data, copied into the slot by
  // push() and out of it by pop(); those copies must not throw (see the class
  // comment: nothing in push()/pop() may throw). For the pointers and
  // integers the queue is meant for this holds already; it is stated so the
  // requirement is visible where the others (on Value) are.
  static_assert(std::is_nothrow_copy_constructible_v<Key> && std::is_nothrow_copy_assignable_v<Key>);
  // In the key-value queue every popped value is destroyed in its slot, and
  // so is every value the destructor drains; a Value whose destructor may
  // throw could never be popped, so it is refused here, for the whole class.
  // Only that: constructing Value (from whatever push() is given) and moving
  // it out (into whatever pop() is given) depend on the argument, and are
  // push()'s and pop()'s own requirements.
  static_assert(std::is_same_v<Value, void> || std::is_nothrow_destructible_v<Value>);
  // Key-value slot type. seq is the slot's state word (see the slot protocol
  // in the class comment); key and value are plain data, written by the
  // producer that owns the slot and read by the consumer that owns it next,
  // with the release/acquire pair on seq ordering the two. alignas is the max
  // of: the caller-requested ALIGN (e.g. 64 to give each slot its own cache
  // line and kill false sharing between adjacent slots), the sequence
  // number's, the key's and the value's own requirements -- so the slot is
  // always at least as aligned as any member needs.
  template <typename K, typename V> struct alignas(std::max({ALIGN, alignof(std::atomic<size_t>), alignof(K), alignof(V)})) queue_slot_t {
    std::atomic<size_t> seq;    // Slot state: i (free for index i), i + 1 (committed), i + capacity (free for the next lap)
    K key;                      // Element key; valid while seq == i + 1
    V value;                    // Element value; constructed by push, destroyed by pop, valid while seq == i + 1
  };
  // Key-only specialization: the key atomic alone carries the full slot
  // state (zero = empty, non-zero = full). Since there is no value to
  // construct/destruct, push/pop do the key store inside the lock and need
  // no sequence number -- the critical section stays tiny (~3-4 cycles of
  // actual work) and each slot shrinks to the size of the key.
  template <typename K> struct alignas(std::max({ALIGN, alignof(std::atomic<K>)})) queue_slot_t<K, void> {
    std::atomic<K> key;
  };
  using slot_t = queue_slot_t<Key, Value>;

  public:
  // Constructor initializes the queue with the span of memory to be used as
  // a ring buffer.  While any size is allowed, the queue will use the
  // largest array of elements whose size is a power of 2 that can fit into
  // the provided buffer. The buffer must have space for at least 8 elements.
  RingAtomicMapQueueMPMC(void* memory, size_t bytes) :
    queue_(static_cast<slot_t*>(memory)),
    capacity_(std::bit_floor(bytes/sizeof(slot_t))),
    capacity_mask_(capacity_ - 1)
  {
    CHECK_GE(capacity_, 8);
    // Zero-filling puts every key-only slot into the "empty" state (key ==
    // Key{}). Slot objects are never formally constructed in the caller's
    // buffer (pedantically UB, accepted by every real compiler; slot_t is not
    // an implicit-lifetime type because std::atomic's default constructor is
    // non-trivial, so std::start_lifetime_as_array would not fix it either).
    // Values in particular exist only between push's placement-new and the
    // matching explicit destructor call in pop() or ~RingAtomicMapQueueMPMC().
    ::memset(static_cast<void*>(queue_), 0, capacity_ * sizeof(slot_t));
    if constexpr (!std::is_same_v<Value, void>) {
      // A key-value slot is free for the first index it serves, which is its
      // own position p, so seq must start at p, not 0 (see the slot protocol
      // in the class comment: seq == i means "free for index i").
      // relaxed: nothing is published here; the constructor is ordered before
      // any push/pop by whatever hands the queue to the threads that use it
      // (thread creation, a lock, ...), as for any object.
      for (size_t p = 0; p != capacity_; ++p) {
        queue_[p].seq.store(p, std::memory_order_relaxed);
      }
    } // if key-value queue
  } // RingAtomicMapQueueMPMC()

  // Destructor drains the queue if there are any elements remaining.
  ~RingAtomicMapQueueMPMC() {
    // Drain remaining elements. For KV queues this runs value destructors
    // that would otherwise leak. No-op if the queue is already empty.
    // No locks: destruction requires exclusive access like any destructor.
    // head_.i <= tail_.i always holds once threads quiesce (a pop can only
    // claim a slot some push has committed), and every index in [head_.i,
    // tail_.i) holds a constructed value: tail_.i advances only when a push
    // claims the slot, and a push that has claimed always goes on to
    // construct the value -- there is no exit between the two, since the
    // constructor cannot throw (enforced on push()).
    for (size_t i = head_.i; i != tail_.i; ++i) {
      slot_t& slot = queue_[i & capacity_mask_];
      if constexpr (!std::is_same_v<Value, void>) {
        slot.value.~Value();
      } else {
        slot.key.store(Key{}, std::memory_order_relaxed);
      }
    } // drain [head_.i, tail_.i)
  } // ~RingAtomicMapQueueMPMC()

  // Push an element (key-value pair) onto the queue.
  // This method copies or moves the value by the corresponding constructor;
  // the key is copied.
  // The method returns true if the operation was successful, and false if
  // the queue is full. In the latter case, the value is not moved-from and
  // is otherwise unchanged.
  //
  // Flow (see the slot protocol in the class comment): under the tail lock,
  // read the sequence number of the slot at tail index i. If it is i, the
  // slot is free for this index: write the key, claim the slot by advancing
  // tail_.i, release the lock, construct the value in the slot, and commit by
  // storing seq = i + 1. Any other value means the element of index i -
  // capacity is still in the slot: retry NTRY times (a consumer may be a few
  // instructions from releasing it), then report full.
  template <typename V>
    requires std::is_nothrow_constructible_v<Value, V>
  bool push(Key key, V&& value) noexcept {
    // Key{} is what pop() returns for "empty" (see the class comment); an
    // element with that key could never be told apart from an empty queue by
    // the caller of pop(). Debug-only: always a caller bug.
    assert(!(key == Key{}));
    std::unique_lock<SpinLock> l(tail_.l);
    const size_t i = tail_.i;                   // Index this push takes if the slot is free
    slot_t& slot = queue_[i & capacity_mask_];
    for (size_t n = 0; n != NTRY; ++n) {
      // acquire: synchronizes-with the release store of seq == i by the
      // consumer of index i - capacity, so its ~Value() and its read and clear
      // of the key happen-before our key write and placement-new (on the first
      // lap the slot has never been used and there is nothing to order
      // against).
      const size_t seq = slot.seq.load(std::memory_order_acquire);
      RING_QUEUE_HOOK(PUSH_LOADED);
      if (seq == i) {
        // The slot is ours from this moment (seq == i, seen under the tail
        // lock): no consumer touches it before seq == i + 1, which only this
        // thread stores, and no producer before seq == i + capacity, which
        // only the consumer that claims index i stores, after this thread's
        // commit. The key is written here, inside the critical section, on
        // purpose: this store is what lets the lock batch (see "Why every
        // critical section ends with a store to the slot" in the class
        // comment). The protocol would be correct with it after the unlock;
        // the throughput would not.
        slot.key = key;
        // Claim: advancing tail_.i under the lock hands the *next* index to
        // the next producer, and the expensive value construction stays out
        // of the critical section.
        ++tail_.i;
        l.unlock();     // The slot's sequence number is enough from now on
        RING_QUEUE_HOOK(PUSH_CLAIMED);
        ::new(&slot.value) Value(std::forward<V>(value));       // Move/copy-constructed in place
        RING_QUEUE_HOOK(PUSH_COMMITTING);
        // release: commits the element -- the consumer whose acquire load of
        // seq sees i + 1 also sees the constructed value and the key (written
        // under the lock above; the lock orders it against other producers,
        // this store publishes it to the consumer). This is the single store
        // that makes the element visible; there is no state between "key
        // visible" and "committed".
        slot.seq.store(i + 1, std::memory_order_release);
        return true;
      } // if slot is free for index i
    } // NTRY loop
    // The slot never became free for this index: the queue is full (or the
    // slot is still being drained by a wrapped-around consumer, which to the
    // caller is the same thing -- the ring has no room at the tail right now).
    return false;
  } // push(key, value)

  // Push for key-only queue.
  // Observing key==0 under the tail lock claims the slot: no other producer
  // can land on it until the ring wraps, and consumers cannot advance head
  // past a slot whose key is still 0 (pop treats key==0 as empty and bails).
  // That means the store-release of the key -- the cross-lock handoff to
  // the consumer -- would be correct outside the critical section too. It
  // stays inside for the lock's sake: see "Why every critical section ends
  // with a store to the slot" in the class comment (this queue is where the
  // effect was measured first; the numbers are in README.md).
  bool push(Key key) noexcept requires std::is_same_v<Value, void> {
    assert(!(key == Key{}));    // Key{} is the reserved empty-slot marker; see class comment
    std::unique_lock<SpinLock> l(tail_.l);
    // Next slot to store the data into.
    slot_t& slot = queue_[tail_.i & capacity_mask_];
    // If the slot is empty, we can enqueue and increment producer index.
    // Otherwise, the queue is full, bail out.
    // NTRY retries tolerate a consumer that's currently draining this
    // slot (tail wrapped into a slot whose clear-to-0 hasn't landed yet).
    for (size_t i = 0; i != NTRY; ++i) {
      if (slot.key.load(std::memory_order_acquire) == Key{}) {
        slot.key.store(key, std::memory_order_release);
        ++tail_.i;
        return true;
      }
    }
    return false;
  } // push()

  // Pop the first element from the queue. The key is returned; if the queue
  // is empty, the default-constructed value is returned.  Value is returned
  // by move assignment only if the queue is not empty, otherwise caller's
  // value is unchanged.
  //
  // Flow (see the slot protocol in the class comment): under the head lock,
  // read the sequence number of the slot at head index i. If it is not i + 1
  // nothing is committed at the head: report empty. Otherwise read and clear
  // the key, claim the slot by advancing head_.i, release the lock, move the
  // value out and destroy it, and release the slot to the next lap's producer
  // by storing seq = i + capacity.
  //
  // pop() never waits. The previous protocol had two slot states it had to
  // wait out under the head lock: a producer between its key store and its
  // busy-flag clear (the element visible but not yet committed), and a
  // previous-lap consumer still clearing the slot. Neither exists here: the
  // commit is the single seq store, so a visible element is a committed one,
  // and a previous-lap consumer still draining the slot means tail_.i ==
  // head_.i -- the queue is genuinely empty (see the class comment). The head
  // lock is held for one load, one compare, the key read and clear, and one
  // increment.
  template <typename V = Value>
    requires (!std::is_same_v<void, V>) && std::is_nothrow_assignable_v<V&, Value&&>
  Key pop(V& value) noexcept {
    std::unique_lock<SpinLock> l(head_.l);
    const size_t i = head_.i;                   // Index this pop takes if the slot is committed
    // Next slot to read the data from.
    slot_t& slot = queue_[i & capacity_mask_];
    // acquire: synchronizes-with the producer's release store of seq == i + 1,
    // so its key write and its placement-new of the value happen-before our
    // key read (and clear) and our move of the value.
    const size_t seq = slot.seq.load(std::memory_order_acquire);
    RING_QUEUE_HOOK(POP_LOADED);
    // Not committed for this index: seq == i (the producer at index i has not
    // claimed, or is still constructing) or seq == i - capacity + 1 (the
    // previous lap's consumer is still moving its value out, in which case the
    // producer at index i cannot have claimed either). Either way there is no
    // element of index >= i in the ring: the queue is empty.
    if (seq != i + 1) return Key{};
    // The slot is ours from this moment (seq == i + 1, seen under the head
    // lock) until the release store below. The key is read here, and then
    // cleared. The clear exists ONLY for its store: nothing ever reads a
    // cleared key (emptiness is seq's job, and the next producer overwrites
    // the key before anyone looks at the slot). It sits inside the critical
    // section so that the lock can batch -- see "Why every critical section
    // ends with a store to the slot" in the class comment. Do not remove it
    // as dead code: it is the difference between this queue's throughput and
    // a fraction of it.
    const Key ret = slot.key;
    slot.key = Key{};
    // Claim: mirror of push. head_.i advances so the next consumer gets the
    // next index; the value move/destruction runs outside the critical
    // section.
    ++head_.i;
    l.unlock();     // The slot's sequence number is enough from now on
    RING_QUEUE_HOOK(POP_CLAIMED);
    value = std::move(slot.value);  // Move-assigned to the caller
    slot.value.~Value();            // Destructed in place, push will construct again
    RING_QUEUE_HOOK(POP_RELEASING);
    // release: hands the slot to the producer of index i + capacity -- its
    // acquire load of seq == i + capacity orders the destruction above before
    // its placement-new, and our read and clear of the key before its key
    // write.
    slot.seq.store(i + capacity_, std::memory_order_release);
    return ret;
  } // pop(value)

  // Pop for key-only queue. Mirror of key-only push; see that comment for
  // why the clear-to-0 store sits inside the critical section.
  Key pop() noexcept requires std::is_same_v<Value, void> {
    std::unique_lock<SpinLock> l(head_.l);
    slot_t& slot = queue_[head_.i & capacity_mask_];
    Key ret = slot.key.load(std::memory_order_acquire);
    if (ret == Key{}) return Key{};
    slot.key.store(Key{}, std::memory_order_release);
    ++head_.i;
    return ret;
  } // pop()

  // Element size, with the requested alignment.
  static constexpr size_t element_size() { return sizeof(slot_t); }

  // Element alignment.
  static constexpr size_t element_align() { return alignof(slot_t); }

  // Current queue capacity, in number of elements.
  size_t capacity() const { return capacity_; }

  // Check if the queue is empty. If the queue is not empty, pop() would have
  // succeeded were it done instead of empty(). No guarantee that the next
  // pop() call succeeds unless there are no consumers active at the same
  // time.
  bool empty() const {
    std::unique_lock<SpinLock> l(head_.l);
    // Next slot to read the data from.
    slot_t& slot = queue_[head_.i & capacity_mask_];
    // The queue is empty if the head slot is not committed: key == Key{} in
    // the key-only queue, seq != head index + 1 in the key-value queue.
    // Relaxed is enough: empty() is advisory (see the contract above), no
    // data is read based on the answer, and the head lock already orders
    // this call against slot claims by concurrent consumers.
    if constexpr (std::is_same_v<Value, void>) {
      return slot.key.load(std::memory_order_relaxed) == Key{};
    } else {
      return slot.seq.load(std::memory_order_relaxed) != head_.i + 1;
    }
  } // empty()

  private:
  slot_t* const queue_;                 // Queue memory (slot_t[capacity_] array)
  const size_t capacity_;               // Current queue capacity (number of elements, always a power of 2)
  const size_t capacity_mask_;          // Cached index bit mask (capacity_ - 1)

  // Tail - producer data. alignas(256) rather than 64: producers and
  // consumers hammer their respective lock+index, so the two blocks are
  // padded past not just the cache line but also the 128-byte granularity
  // of adjacent-line prefetchers (and 128/256-byte lines on some CPUs),
  // which would otherwise re-couple them.
  alignas(256) struct tail_t {
    mutable SpinLock l;
    size_t i {};
  } tail_;

  // Head - consumer data.
  alignas(256) struct head_t {
    mutable SpinLock l;
    size_t i {};
  } head_;
}; // RingAtomicMapQueueMPMC

// Header-private default of the test hook: expanded when the templates above
// were parsed (macros expand at definition, not at template instantiation),
// so it is not exported to includers. A test's own definition is left alone.
#ifdef RING_QUEUE_HOOK_DEFAULT_
#undef RING_QUEUE_HOOK
#undef RING_QUEUE_HOOK_DEFAULT_
#endif

#endif // RING_ATOMIC_QUEUE_H_
