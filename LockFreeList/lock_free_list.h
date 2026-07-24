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
#ifndef INCLUDED_CONCURRENT_LIST_H
#define INCLUDED_CONCURRENT_LIST_H

#include <atomic>
#include <memory>
#include <type_traits>

// Generic Lock-Free Forward List parameterized by a custom AtomicPtr type.
//
// This is a Harris-style singly-linked list. Each node's `next` is an AtomicPtr,
// and logical deletion is recorded by setting the mark bit on the *deleted node's
// own* next pointer; physical unlinking then swings the predecessor past it.
//
// Architectural Choices:
// 1. Fully isolated template design: Nodes must be externally allocated using the
//    underlying atomic pointer structure. The list is decoupled from specific ptr logic.
// 2. Compile-time branching: insert_after and erase_after leverage
//    `if constexpr (AtomicPtr<Node>::supports_marking)` to enable Harris-style marking
//    or fallback to simpler implementations if marking is unavailable.
// 3. Safe memory reclamation is delegated entirely to AtomicPtr: every reachable
//    node (and every node an iterator sits on) is kept alive by a strong reference,
//    which also gives ABA-freedom for the CAS loops since a pointer value cannot be
//    recycled while any party still references it.
// 4. Cooperative physical unlinking ("helping", marking mode only): the mark is
//    the linearization point of erase_after; the unlink CAS is best-effort and
//    can lose to a concurrent insert or to the erasure of the anchor itself. An
//    erase_after that finds an already-marked successor unlinks it and retries
//    on the new successor, so a dead node left behind by a failed unlink is
//    removed by the next erasure over the same edge instead of staying linked
//    forever (see lock_free_list_bugs.md, bug 1).
//
// Marking convention: "X is logically deleted" is recorded in X's *own* next
// pointer, so the mark bit travels with the value loaded *from* X->next. In
// other words, a marked value read out of anchor->next means *anchor* is
// deleted; whether anchor's successor is deleted can only be seen by loading
// the successor's own next pointer. Both insert_after and erase_after rely on
// this convention when they interpret is_marked() on loaded values.
//
// Traversal semantics: iteration deliberately does not skip logically deleted
// nodes. An iterator parked on an erased node can keep walking through the
// "graveyard" chain back into the live suffix (erasure never clears a node's
// next pointer), and a traversal racing with an eraser may observe the victim
// before it is physically unlinked.
//
// Concurrency contract: the head is a caller-supplied dummy node that is never
// erased, so before_begin() always yields a valid anchor.
template <typename T, template<typename> class AtomicPtr>
class LockFreeList {
public:
    struct Node {
        T value;
        AtomicPtr<Node> next;
        
        // Intrusive refcount hooks, used only when AtomicPtr is the intrusive
        // shared pointer (which requires them on the pointee); for the other
        // pointer types they are 8 bytes of dead weight per node, accepted so
        // that Node does not have to know which pointer it is instantiated
        // with. The count starts at 0: a freshly new'ed node is owned by no
        // one until the first shared_ptr_type adopts it (0 -> 1); every owner
        // thereafter counts itself via AddRef. DelRef() returns true on the
        // 1 -> 0 transition; its acq_rel order makes all prior writes to the
        // node visible to the deleter.
        std::atomic<long> ref_count{0};
        void AddRef() { ref_count.fetch_add(1, std::memory_order_relaxed); }
        bool DelRef() { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }
        long use_count() const { return ref_count.load(std::memory_order_relaxed); }
        
        Node(T val) : value(std::move(val)) {}
        Node() : value() {}
        
        // Iterative destruction via use_count():
        // We manually walk the linked list and unlink nodes to prevent unbounded
        // recursion and stack overflows when a large list goes out of scope.
        //
        // Why the use_count() == 1 test is race-free: this destructor only runs
        // once this node is unreachable, and `curr` counts as one reference to
        // the node it holds. A count of exactly 1 therefore means no atomic
        // next pointer and no iterator refer to that node, so no other thread
        // can ever acquire it and we may dismantle it. If the count is higher
        // we stop: whichever holder releases last re-enters this destructor
        // and resumes the walk from there.
        //
        // Nulling curr->next *before* advancing is what keeps this iterative:
        // the `curr = ...` assignment deletes the old node, whose own ~Node()
        // then finds next == nullptr and returns without recursing.
        //
        // Relaxed ordering suffices because exclusivity was already established
        // by the synchronization that dropped each refcount to its final value.
        // Note: `curr` may carry the mark bit of a logically deleted node; the
        // shared_ptr accessors (bool, ->, use_count) strip it internally.
        ~Node() {
            auto curr = next.load(std::memory_order_relaxed);
            next.store(nullptr, std::memory_order_relaxed);
            
            while (curr && curr.use_count() == 1) {
                auto next_curr = curr->next.load(std::memory_order_relaxed);
                curr->next.store(nullptr, std::memory_order_relaxed);
                curr = std::move(next_curr);
            } // chain walk while sole owner
        } // ~Node()
    }; // Node

    using SharedNodePtr = typename AtomicPtr<Node>::shared_ptr_type;

    // The iterator holds a *strong* reference (SharedNodePtr) to its current node.
    // This is what makes graveyard traversal safe: even after the node it points to
    // is logically deleted and physically unlinked, the node (and its still-intact
    // next pointer) stays alive, so ++ can transparently walk into the surviving
    // downstream nodes. It is also what gives the list ABA-freedom.
    class iterator {
    public:
        SharedNodePtr curr_;

        iterator() = default;
        explicit iterator(SharedNodePtr node) : curr_(std::move(node)) {}

        T& operator*() const { return curr_->value; }
        T* operator->() const { return &curr_->value; }

        // Advance to the successor, stripping the mark bit: the mark is part of
        // a pointer's identity (marked and unmarked pointers compare unequal),
        // so iterators always hold canonical unmarked values -- otherwise an
        // iterator that walked out of a deleted node would never compare equal
        // to one that reached the same node through the live chain. A marked
        // null (the last node was erased) becomes a plain end(). Whether the
        // next node *itself* is deleted is not visible here (its mark lives in
        // its own next pointer), so ++ walks into deleted nodes by design.
        iterator& operator++() {
            if (curr_) {
                curr_ = curr_->next.load(std::memory_order_acquire).get_unmarked();
            }
            return *this;
        }

        bool operator==(const iterator& other) const { return curr_ == other.curr_; }
        bool operator!=(const iterator& other) const { return curr_ != other.curr_; }
    }; // iterator

    LockFreeList(SharedNodePtr dummy_head) {
        head_.store(std::move(dummy_head));
    }

    // get_unmarked() here is defensive only: the dummy head is never erased
    // (class contract), so the value in head_ is never actually marked.
    iterator before_begin() const {
        return iterator(head_.load(std::memory_order_acquire).get_unmarked());
    }
    
    iterator begin() const {
        auto it = before_begin();
        ++it;
        return it;
    }
    
    iterator end() const {
        return iterator(nullptr);
    }

    // Insert `new_node` immediately after `anchor`. Returns true on success.
    // With marking: fails (returns false) if `anchor` has been logically deleted,
    // because linking onto a deleted node would strand the new node in a detached
    // "black hole" chain that is about to be unlinked.
    bool insert_after(iterator anchor, SharedNodePtr new_node) {
        if (!anchor.curr_) return false;

        // Compile-time branch: Uses Harris marking loop if supported, else simple CAS
        if constexpr (AtomicPtr<Node>::supports_marking) {
            while (true) {
                auto expected_next = anchor.curr_->next.load(std::memory_order_acquire);
                // Per the marking convention, a mark on the value loaded from
                // anchor->next is *anchor's own* deletion mark.
                if (expected_next.is_marked()) {
                    return false; // anchor is logically deleted
                }
                // Relaxed is enough: new_node is not reachable by any other
                // thread yet; the release CAS below publishes it, store included.
                new_node->next.store(expected_next, std::memory_order_relaxed);
                if (anchor.curr_->next.compare_exchange_strong(expected_next, new_node, std::memory_order_release, std::memory_order_relaxed)) {
                    return true;
                }
                // CAS failure: a competing insert/unlink changed the successor,
                // or anchor got marked (the mark is part of the compared value,
                // so marking anchor always fails this CAS). Discard the updated
                // `expected_next` and reload; the reload re-runs the mark check.
            } // marking insert retry loop
        } else {
            // Blind CAS
            // Note: This risks inserting into a detached "black hole" if the anchor was concurrently erased.
            auto expected_next = anchor.curr_->next.load(std::memory_order_acquire);
            new_node->next.store(expected_next, std::memory_order_relaxed);
            // A failed CAS refreshes `expected_next` with the current successor,
            // so re-point new_node at it before retrying. The acquire failure
            // order is required: the refreshed value is republished through
            // new_node->next, and its consumers must observe the pointee fully
            // constructed.
            while (!anchor.curr_->next.compare_exchange_strong(expected_next, new_node, std::memory_order_release, std::memory_order_acquire)) {
                new_node->next.store(expected_next, std::memory_order_relaxed);
            }
            return true;
        } // else: no marking support
    } // insert_after()

    // Erase the live node immediately after `anchor`. Returns true iff this call
    // logically deleted a node; returns false only if anchor is null or logically
    // deleted, or anchor has no live successor. Harris two-step: (1) mark the
    // target's own next pointer to logically delete it, winning the race against
    // any concurrent eraser; (2) swing anchor->next past the target to physically
    // unlink it. Step 2 is best-effort: if it fails (anchor->next changed under
    // us), the marked node stays linked until a later erase_after() over the same
    // edge removes it via the helping path. A call that loses the marking race
    // helps unlink the dead node and retries on the new successor rather than
    // report failure, so `while (erase_after(h)) {}` drains the list even under
    // contention.
    bool erase_after(iterator anchor) {
        if (!anchor.curr_) return false;

        if constexpr (AtomicPtr<Node>::supports_marking) {
            // Retry loop: each pass either decides the logical-deletion race on
            // the current successor (win -> true) or helps unlink an already-
            // deleted successor and re-examines the new state. A retry happens
            // only after some CAS on anchor->next succeeded (ours or a
            // competitor's), so the list as a whole always makes progress
            // (lock-free, not wait-free).
            while (true) {
                auto target = anchor.curr_->next.load(std::memory_order_acquire);
                // Per the marking convention, a mark on the value loaded from
                // anchor->next is *anchor's own* deletion mark: nothing to erase,
                // or anchor itself is deleted. A target whose own next is marked
                // (already erased, not yet unlinked) passes this check and is
                // handled by the helping path below.
                if (!target || target.is_marked()) return false;

                auto target_next = target->next.load(std::memory_order_acquire);
                bool marked_by_us = false;
                // Race to set the mark on target's own next pointer. A failed CAS
                // refreshes `target_next`: if it is now marked, another eraser
                // won; if it merely changed (an insert landed after target),
                // retry the marking with the new successor. The acquire failure
                // order is required: the refreshed value is republished through
                // the unlink CAS below, and its consumers must observe the
                // pointee fully constructed.
                while (!target_next.is_marked()) {
                    auto marked_next = target_next.set_mark();
                    if (target->next.compare_exchange_strong(target_next, marked_next, std::memory_order_release, std::memory_order_acquire)) {
                        marked_by_us = true;
                        target_next = marked_next; // to have the correct value for unlinking
                        break;
                    }
                } // marking race
                
                // target is now logically deleted -- by us, by a concurrent
                // eraser, or long ago (a dead node left behind by an unlink that
                // lost its CAS). The mark freezes target->next for good:
                // insert_after() refuses a marked anchor and no second eraser can
                // win the marking race, so target_next.get_unmarked() is target's
                // final successor and stays the correct unlink value no matter
                // how long we stall before the CAS below.
                //
                // Physically unlink it (helping, when we did not mark it
                // ourselves). On failure anchor->next changed under us -- an
                // insert landed after anchor, or anchor itself was erased -- and
                // the dead node stays linked until the next erase_after() over
                // this edge helps it out. The updated `target` is discarded (a
                // retry reloads it), hence the relaxed failure order.
                anchor.curr_->next.compare_exchange_strong(target, target_next.get_unmarked(), std::memory_order_release, std::memory_order_relaxed);

                if (marked_by_us) {
                    return true; // our mark is the linearization point; unlink success is not required
                }
                // We lost the marking race (or found a long-dead successor) and
                // helped unlink it; retry on the new state of anchor->next.
            } // retry / helping loop
        } else {
            // Direct physical unlinking without marking. A failed CAS refreshes
            // `target` with the current successor, so the loop retries on the
            // new front until it unlinks one node or the list runs empty. The
            // acquire failure order is required: the refreshed `target` is both
            // dereferenced and republished (its successor goes back into
            // anchor->next), so we must synchronize with whoever produced it.
            //
            // Known limitation (the reason marking exists): without logical
            // deletion, concurrent erases at adjacent positions can resurrect a
            // node. With H -> X -> Y -> Z, an eraser at H that already read X's
            // successor Y can still swing H -> Y after another thread erased Y
            // (its CAS of X->next: Y -> Z succeeded and returned true). Y is
            // then back in the list even though its erase reported success.
            // Unfixable without a mark bit; see lock_free_list_bugs.md, bug 2.
            auto target = anchor.curr_->next.load(std::memory_order_acquire);
            while (target) {
                auto target_next = target->next.load(std::memory_order_acquire);
                if (anchor.curr_->next.compare_exchange_strong(target, target_next, std::memory_order_release, std::memory_order_acquire)) {
                    return true;
                }
            } // retry until unlinked or list empty
            return false;
        } // else: no marking support
    } // erase_after()

private:
    AtomicPtr<Node> head_; // holds the dummy head node: set once in the constructor, never marked or erased
}; // LockFreeList

#endif // INCLUDED_CONCURRENT_LIST_H
