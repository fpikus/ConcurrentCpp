// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/LockFree
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
#include <thread>
#include "spinlock.h"

// ---------------------------------------------------------------------------
// Simple CHECK macros (abort on failure, no messages).
// ---------------------------------------------------------------------------
#define CHECK(c) if (!(c)) std::abort();
#define CHECK_GE(a, b) if ((a) < (b)) std::abort();
#define CHECK_EQ(a, b) if ((a) != (b)) std::abort();


// ---------------------------------------------------------------------------
// RingAtomicMapQueueMPMC
// ---------------------------------------------------------------------------

// Key and Value together form the element stored in the queue. The Key type
// has to support lock-free atomic operations; this implies that it is
// trivially copyable and destructible. The value type can be any type; the
// default (void) means that the queue contains only keys.
//
// Key{} (the default/zero key) is reserved as the "empty slot" marker and
// must never be pushed: a slot holding Key{} is indistinguishable from an
// empty one, so pushing Key{} advances the tail but leaves a slot no
// consumer can ever pop past, stranding every element behind it. This is
// enforced by a debug-only assert in push(): it is always a bug in the
// caller, but correct callers should not pay for the check.
//
// The queue is not exception-safe: if the Value constructor or move
// assignment throws inside push()/pop(), the slot is abandoned mid-commit
// and the queue is permanently wedged. This is deliberate: the whole API is
// not exception-friendly -- exception friendliness is how std::queue ends up
// with a pop() that does not return the value, which is exactly what a
// concurrent queue cannot afford. Use noexcept value types.
//
// Unlike a classic ring buffer, the head and tail indices are never compared
// to each other: full/empty is decided per-slot from the key (and, for KV
// queues, the busy flag). This is why all capacity_ slots are usable, with
// no reserved gap between head and tail.
template <typename Key, typename Value = void,
          size_t NTRY = 8,              // Number of attempts to acquire a slot before giving up
          size_t ALIGN = 0>             // Additional alignment; 64 to align each queue slot on a cache line
class RingAtomicMapQueueMPMC {
  // The key doubles as the slot's atomic state word (empty/full, and the target
  // of the key->busy->key validation), so it must be lock-free -- a mutex-backed
  // "atomic" key would defeat the whole per-slot lock-free coordination.
  static_assert(std::atomic<Key>::is_always_lock_free);
  // Queue slot type (with and without value). alignas is the max of: the caller-
  // requested ALIGN (e.g. 64 to give each slot its own cache line and kill false
  // sharing between adjacent slots), the key atomic's own requirement, and the
  // value's -- so the slot is always at least as aligned as any member needs.
  template <typename K, typename V> struct alignas(std::max({ALIGN, alignof(std::atomic<K>), alignof(V)})) queue_slot_t {
    std::atomic<K> key;
    std::atomic<int> busy;      // In-flight commit: set under lock before unlock, cleared after key store/clear
    V value;
  };
  // Key-only specialization: the key atomic alone carries the full slot
  // state (zero = empty, non-zero = full). Since there is no value to
  // construct/destruct, push/pop do the key store inside the lock and
  // drop the busy flag entirely -- the critical section stays tiny
  // (~3-4 cycles of actual work) and each slot shrinks by a cache line.
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
    // Zero-filling puts every slot into the "empty" state: key == Key{} and
    // busy == 0. Slot objects are never formally constructed in the caller's
    // buffer (pedantically UB, accepted by every real compiler; slot_t is not
    // an implicit-lifetime type because std::atomic's default constructor is
    // non-trivial, so std::start_lifetime_as_array would not fix it either).
    // Values in particular exist only between push's placement-new and the
    // matching explicit destructor call in pop() or ~RingAtomicMapQueueMPMC().
    ::memset(static_cast<void*>(queue_), 0, capacity_ * sizeof(slot_t));
  } // RingAtomicMapQueueMPMC()

  // Destructor drains the queue if there are any elements remaining.
  ~RingAtomicMapQueueMPMC() {
    // Drain remaining elements. For KV queues this runs value destructors
    // that would otherwise leak. No-op if the queue is already empty.
    // No locks: destruction requires exclusive access like any destructor.
    // head_.i <= tail_.i always holds once threads quiesce (a pop can only
    // claim a slot some push has committed), and tail_.i only advances on a
    // successful push, so every slot in [head_.i, tail_.i) holds a
    // constructed value.
    for (size_t i = head_.i; i != tail_.i; ++i) {
      slot_t& slot = queue_[i & capacity_mask_];
      if constexpr (!std::is_same_v<Value, void>) {
        slot.value.~Value();
      }
      slot.key.store(Key{}, std::memory_order_relaxed);
    } // drain [head_.i, tail_.i)
  } // ~RingAtomicMapQueueMPMC()

  // Push an element (key-value pair) onto the queue.
  // This method copies or moves the value by the corresponding constructor.
  // The key is copied bit-wise, atomically.
  // The method returns true if the operation was successful, and false if
  // the queue is full. In the latter case, the value is not moved-from and
  // is otherwise unchanged.
  template <typename V>
    requires std::is_constructible_v<Value, V>
  bool push(Key key, V&& value) {
    // Key{} is the reserved empty-slot marker (see the class comment);
    // pushing it wedges the queue. Debug-only: always a caller bug.
    assert(!(key == Key{}));
    std::unique_lock<SpinLock> l(tail_.l);
    // Next slot to store the data into.
    slot_t& slot = queue_[tail_.i & capacity_mask_];
    // If the slot is empty, we can enqueue and increment producer index.
    // Otherwise, the queue is full, bail out.
    // We must check key -> busy -> key to prevent the ABA problem on the busy flag.
    // A delayed Producer could read key=0, get preempted, and by the time it reads
    // busy=0, a Consumer could have finished emptying it. Or a Consumer could read key=10,
    // get preempted, and by the time it reads busy=0, a Producer could have finished
    // filling it. The key->busy->key sequence ensures the slot didn't transition
    // while we were reading.
    for (size_t i = 0; i != NTRY; ++i) {
      if (slot.key.load(std::memory_order_acquire) == Key{} &&
          !slot.busy.load(std::memory_order_acquire) &&
          slot.key.load(std::memory_order_acquire) == Key{}) {
        // Claim the slot: busy=1 marks an in-flight commit, so a thread that
        // wraps around to this slot (whose key is still Key{}) can tell it is
        // not stably empty. Advancing tail_.i before unlocking hands the
        // *next* slot to the next producer; from here on this slot is owned
        // by this thread, guarded by busy/key alone, and the expensive value
        // construction stays out of the critical section.
        slot.busy.store(1, std::memory_order_release);
        ++tail_.i;
        l.unlock();     // Atomic operations on queue slots are enough from now on
        // Commit order matters: the key store-release publishes the
        // constructed value (pop loads the key with acquire), and busy is
        // cleared last so no other thread can pass its key->busy->key check
        // (busy==0 means "stable") while the slot is half-committed.
        ::new(&slot.value) Value(std::forward<V>(value));       // Move/copy-constructed in place
        slot.key.store(key, std::memory_order_release);         // Slot is released
        slot.busy.store(0, std::memory_order_release);
        return true;
      } // if slot is stably empty
    } // NTRY loop
    // The slot never looked stably empty: the queue is full (or the slot is
    // still being drained by a wrapped-around consumer, which to the caller
    // is the same thing -- the ring has no room at the tail right now).
    return false;
  } // push(key, value)

  // Push for key-only queue.
  // Observing key==0 under the tail lock claims the slot: no other producer
  // can land on it until the ring wraps, and consumers cannot advance head
  // past a slot whose key is still 0 (pop treats key==0 as empty and bails).
  // That means the store-release of the key -- the cross-lock handoff to
  // the consumer -- is correct both inside and outside the critical section.
  // We keep it inside the lock because the l.unlock()-before-store variant
  // loses badly on Zen4 and Grace ARM (up to 8x throughput collapse at
  // a64/a128 with 4+ threads), even though it wins ~2x on Cascade Lake at
  // a0/a16.  Tested on gaudi (Cascade Lake 64t), flex7589 (Zen4 64t),
  // cal-arm-21 (Grace ARM 72t) -- store-inside-lock is the only variant
  // that delivers consistent throughput across all three architectures.
  bool push(Key key) requires std::is_same_v<Value, void> {
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
  template <typename V = Value>
    requires (!std::is_same_v<void, V>)
  Key pop(V& value) {
    std::unique_lock<SpinLock> l(head_.l);
    // Next slot to read the data from.
    slot_t& slot = queue_[head_.i & capacity_mask_];
    while (true) {
      Key ret = slot.key.load(std::memory_order_acquire);
      // key==Key{} means no committed element at head: the queue is empty,
      // or a producer is still mid-commit (busy set, key not stored yet).
      // Either way that push() has not returned to its caller, so reporting
      // "empty" is a correct, linearizable answer -- no need to wait on busy.
      if (ret == Key{}) return Key{};
      // key -> busy -> key validation
      if (!slot.busy.load(std::memory_order_acquire) &&
          slot.key.load(std::memory_order_acquire) == ret) {

        // Claim: mirror of push. busy=1 marks the in-flight removal,
        // head_.i advances so the next consumer gets the next slot, and the
        // value move/destruction runs outside the critical section.
        slot.busy.store(1, std::memory_order_release);
        ++head_.i;
        l.unlock();  // Atomic operations on queue slots are enough from now on
        value = std::move(slot.value);  // Move-assigned to the caller
        slot.value.~Value();            // Destructed in place, push will construct again
        // Key is cleared before busy: a producer accepts this slot only after
        // seeing busy==0, and its acquire load of busy then guarantees the
        // destruction above completed before its placement-new.
        slot.key.store(Key{}, std::memory_order_release);
        slot.busy.store(0, std::memory_order_release);
        return ret;
      } // if slot is stably full
      // If it was busy, a producer is finishing its commit or a consumer
      // from a previous lap of the ring is still clearing this slot. We must
      // NOT return Key{} here: the caller would take that as "queue empty"
      // and could stop polling while committed elements sit in later slots,
      // stranded forever (head can advance past this slot only through this
      // loop). Spin until the slot settles into either key==Key{} (return
      // empty above) or a stable poppable element.
      if (slot.busy.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    } // retry loop
  } // pop(value)

  // Pop for key-only queue. Mirror of key-only push; see that comment for
  // why the clear-to-0 store sits inside the critical section.
  Key pop() requires std::is_same_v<Value, void> {
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
    // If the slot is not empty, the queue is not empty either.
    // Relaxed is enough: empty() is advisory (see the contract above), no
    // data is read based on the answer, and the head lock already orders
    // this call against slot claims by concurrent consumers.
    Key key = slot.key.load(std::memory_order_relaxed);
    return key == Key{};
  } // empty()

  private:
  slot_t* const queue_;                 // Queue memory (slot_t[capacity_] array)
  size_t const capacity_;               // Current queue capacity (number of elements, always a power of 2)
  size_t const capacity_mask_;          // Cached index bit mask (capacity_ - 1)

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

#endif // RING_ATOMIC_QUEUE_H_
