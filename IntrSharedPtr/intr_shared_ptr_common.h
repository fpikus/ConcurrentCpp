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
#ifndef INCLUDED_INTR_SHARED_PTR_COMMON_H_
#define INCLUDED_INTR_SHARED_PTR_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

// Implementation header of intr_shared_ptr (intr_shared_ptr.h) and
// intr_shared_ptr_hp (intr_shared_ptr_hp.h): the code the two intrusive atomic
// shared pointers share. Not meant to be included directly; it declares no
// public name outside namespace intr_shared_ptr_detail. The public pointee
// contract is in intr_pointee.h.
//
// What the two pointers share and what they do not. Both store one word
// (U* with the Harris mark in bit 0; the spinlock pointer also keeps a lock
// bit in bit 1) and hand out the same non-atomic value type, shared_ptr_type:
// an owning handle that AddRefs its pointee when it is created from another
// owner and releases it when it dies. The one thing the value type does
// differently in the two pointers is what happens when a release drops the
// count to 0: intr_shared_ptr deletes the object; intr_shared_ptr_hp retires
// it to the hazard pointer domain, which deletes it at a later scan. The
// atomic operations themselves (load, store, the CASes), their orders and the
// word layout differ entirely and stay in the pointer headers.
//
// So the shared part is common_base<T, U, Disposal> below: the value type and
// the helpers that convert between the raw word and the value (unmarked_ptr,
// adopt_word). Disposal is the variation point -- a policy class, one per
// pointer (intr_shared_ptr_detail::delete_disposal in intr_shared_ptr.h,
// retire_disposal in intr_shared_ptr_hp.h), with two static members:
//
//   template <typename U> static constexpr void require_pointee() noexcept;
//       static_asserts the pointer's pointee concept on U with the pointer's
//       own diagnostic (IntrusivePointee for the spinlock pointer,
//       HpIntrusivePointee for the hazard pointer one).
//   template <typename U> static void release(U* p) noexcept;
//       release one reference to the UNMARKED pointee p: DelRef(), and on the
//       1->0 transition dispose of the object -- `delete p` or `p->retire()`.
//       Null is a no-op. The one place every release of that pointer goes
//       through, from the value type here and from the pointer's own members.
//       noexcept: the hooks are, retire() is, and ~U is required to be
//       (require_pointee()).
//       (The whole release is the policy's, not just the disposal, so that
//       each pointer controls the exact shape of its release's object code:
//       see the policies.)
//
// Why a (private) BASE CLASS rather than a shared value-type template that the
// pointers alias as `using shared_ptr_type = ...;`: two reasons. First, the
// name intr_shared_ptr<T, U>::shared_ptr_type must keep denoting a class whose
// injected-class-name is shared_ptr_type, because this spelling of an in-place
// destruction must keep compiling:
//
//   void destroy(intr_shared_ptr<P>::shared_ptr_type* p) { p->~shared_ptr_type(); }
//
// With `using shared_ptr_type = detail::value<P, P, ...>;` in intr_shared_ptr
// it does not: in `p->~shared_ptr_type()` the name after `~` is looked up in
// the class of *p (and in the context of the expression), and the alias is a
// member of intr_shared_ptr<P>, visible in neither, so the compiler finds no
// class-name after `~`. (A typedef-name IS allowed after `~` when the
// lookup finds it, e.g. a namespace-scope alias; this one is not found.) A
// nested class inherited from a base keeps the spelling valid. Second, the
// base gives the pointers protected access to the helpers without any friend
// declaration crossing header boundaries. Each pointer derives PRIVATELY
// (implemented-in-terms-of; the base is not part of the pointer's interface)
// and re-exports shared_ptr_type with a using-declaration.
//
// Object code: every member here is a one-line inline function, and the
// pointers compile to the same instructions as with this code written in
// place (a compiler may lay out an inlined helper's blocks and registers
// differently).
namespace intr_shared_ptr_detail {

template <typename T, typename U, typename Disposal>
class common_base {
public:
    // The non-atomic "value" type handed out by load() and accepted by store()
    // and the CASes. It owns one strong reference to its pointee (AddRef on
    // acquire, DelRef on release; on the 1->0 transition the pointer's
    // Disposal: delete or retire) and may carry the mark bit in bit 0 of its
    // raw pointer. Both pointers hand out this same type, instantiated with
    // their own Disposal; it is reached as intr_shared_ptr<T, U>::shared_ptr_type
    // and intr_shared_ptr_hp<T, U>::shared_ptr_type.
    //
    // The value of a handle is the pair (pointer, mark) = the raw word. Two
    // handles are equal iff their raw words are equal: a marked and an
    // unmarked pointer to the same object are DIFFERENT values, and a marked
    // null (word == 1) is a legitimate value distinct from null (is_marked()
    // true, operator bool false, get() == nullptr, use_count() == 0).
    //
    // Special members: rule of five, by hand, because the object owns a counted
    // reference. Copy = AddRef; move = steal (source becomes null); destroy =
    // release. Value semantics with mark-as-identity (operator== compares the
    // raw word including the mark).
    //
    // Thread safety of ONE handle, std::shared_ptr's rule: concurrent const
    // operations on the same handle are race-free -- copying from it, get(),
    // operator->, operator*, use_count(), is_marked(), the lvalue
    // get_unmarked(), set_mark(), the comparisons: each reads the handle's one
    // word (p_) and makes at most an atomic RMW on the count. Any mutating
    // operation -- assignment into it, moving from it, the rvalue
    // get_unmarked(), destruction -- needs exclusion against every other
    // access to that handle. A pointer that several threads read and write is
    // what the atomic is for.
    //
    // Aliasing guarantee of the assignments: the source may be a handle stored
    // INSIDE the pointee that the assignment releases -- `h = h->child` and
    // `h = std::move(h->child)` are well-defined when h is the parent's only
    // owner, exactly as for std::shared_ptr. Releasing the old pointee may
    // destroy the source handle (synchronously for intr_shared_ptr; for
    // intr_shared_ptr_hp when the retire runs a scan in place), so both
    // assignments read everything they need from the source BEFORE the
    // release, and the move assignment nulls the source before it too. The
    // same order covers a source owned by ANY object the release destroys,
    // not only the old pointee itself: under intr_shared_ptr_hp the retire
    // may run a scan that reclaims unrelated pending objects, and a handle
    // inside one of those (an older, still pending parent's member) is just
    // as dead after the release -- and just as fully read before it. After
    // `h = h->child` on the spinlock pointer h is the child's only owner; on
    // the hazard pointer one the retired parent's member still owns the child
    // until the parent is reclaimed (use_count() 2, then 1).
    //
    // Exceptions: nothing here can throw, and every member is declared
    // noexcept. The hooks are noexcept by contract (intr_pointee.h, checked
    // by the concept), the disposal is noexcept (retire(); delete of a U whose
    // destructor is nothrow), and a U whose destructor is not nothrow is
    // rejected at compile time (require_pointee(): destructors do not throw
    // -- the general C++ assumption, asserted here rather than reasoned
    // around). operator* and operator-> are noexcept as well; their
    // precondition (non-null) is the caller's, as for a raw pointer.
    class shared_ptr_type {
    public:
        // Null. Postcondition: !*this, !is_marked(), use_count() == 0.
        constexpr shared_ptr_type() noexcept : p_(nullptr) {}
        constexpr shared_ptr_type(std::nullptr_t) noexcept : p_(nullptr) {}

        // Adopt a raw pointer and become one of its owners: AddRef once (0 -> 1
        // for a freshly new'ed object; n -> n+1 for an object that already has
        // owners -- adoption and sharing are the same operation because the
        // count lives in the object, so there is no control block to allocate
        // or find). Bit 0 of `p` may carry the mark; it is kept. Precondition:
        // `p` is null, or points to a live U whose count is nonzero, or to a
        // fresh U whose count is 0 and that has never been owned (an owned
        // object whose count reached 0 has been deleted or retired and must
        // not be revived). A raw marked null (the bit pattern 1) is allowed
        // and yields marked null, with no count traffic. Postcondition:
        // get() == unmarked p, is_marked() == (p & 1).
        explicit shared_ptr_type(U* p) noexcept : p_(p) {
            require_pointee();
            if (get_unmarked_ptr()) get_unmarked_ptr()->AddRef();
        }

        // Copy: AddRef the shared pointee (if non-null). Postcondition:
        // *this == x; x.use_count() grew by one if non-null.
        shared_ptr_type(const shared_ptr_type& x) noexcept : p_(x.p_) {
            require_pointee();
            if (get_unmarked_ptr()) get_unmarked_ptr()->AddRef();
        }

        // Move: steal x's reference. Postcondition: *this holds x's old word,
        // x is null (unmarked null, even if x was marked null).
        shared_ptr_type(shared_ptr_type&& x) noexcept : p_(x.p_) {
            x.p_ = nullptr;
        }

        // Release: DelRef the pointee (if non-null); on 1->0 dispose of it
        // (intr_shared_ptr: delete, here and now; intr_shared_ptr_hp: retire,
        // destroyed at a later scan, on some thread -- see that header).
        ~shared_ptr_type() {
            require_pointee();
            Disposal::release(get_unmarked_ptr());
        }

        // Copy assignment. Order: read x's word, AddRef the new pointee, release
        // the old one, store the word -- everything read from x comes BEFORE
        // the release, because x may live inside the pointee the release
        // destroys (`h = h->child`, h the parent's only owner: the release of
        // the parent runs ~shared_ptr_type on x itself). Reading x.p_ after the
        // release would be a use-after-free (with the spinlock pointer always;
        // with the hazard pointer one whenever the retire runs a scan in
        // place). The AddRef precedes the release for the same aliasing: x's
        // own reference dies with x, and the new pointee must be counted for
        // *this before that. (Not, as one might think, because releasing first
        // could free the object about to be AddRef'd: while x is alive its
        // reference keeps that object alive.) `this` and `x` may also be
        // distinct handles to one pointee, often marked and unmarked variants,
        // which the mark-as-identity design makes routine; the self-assignment
        // check catches only `a = a`.
        shared_ptr_type& operator=(const shared_ptr_type& x) noexcept {
            require_pointee();
            if (this == &x) return *this;
            U* const new_word = x.p_;   // read before the release: x may die in it
            U* new_ptr = unmarked_ptr(reinterpret_cast<uintptr_t>(new_word));
            if (new_ptr) new_ptr->AddRef();
            Disposal::release(get_unmarked_ptr());
            p_ = new_word;
            return *this;
        }

        // Move assignment: take x's word and null x, THEN release the old
        // pointee. No AddRef is needed (x's reference is transferred), but the
        // order matters for the same aliasing as in the copy assignment: with
        // `h = std::move(h->child)` the release of the old pointee destroys x,
        // so x must be read and nulled before it (a destroyed x that still
        // held its word would also release the reference being transferred).
        shared_ptr_type& operator=(shared_ptr_type&& x) noexcept {
            require_pointee();
            if (this == &x) return *this;
            U* const old_ptr = get_unmarked_ptr();   // released last, below
            p_ = x.p_;
            x.p_ = nullptr;
            Disposal::release(old_ptr);
            return *this;
        }

        // Access through the UNMARKED pointer, as T. Precondition for * and ->:
        // non-null (get() != nullptr); undefined otherwise. A marked handle
        // dereferences like its unmarked twin.
        T& operator*() const noexcept { return *get_unmarked_ptr(); }
        T* operator->() const noexcept { return get_unmarked_ptr(); }
        T* get() const noexcept { return get_unmarked_ptr(); }
        explicit operator bool() const noexcept { return get_unmarked_ptr() != nullptr; }

        // Identity comparison of the raw word: pointer AND mark. Marked and
        // unmarked handles to one object compare unequal; marked null != null.
        bool operator==(const shared_ptr_type& rhs) const noexcept { return p_ == rhs.p_; }
        bool operator!=(const shared_ptr_type& rhs) const noexcept { return p_ != rhs.p_; }

        // The pointee's current count (0 for null and marked null): a snapshot
        // for diagnostics and tests; never 0 for a non-null handle (this handle
        // is one owner).
        long use_count() const noexcept {
            require_pointee();
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
        shared_ptr_type get_unmarked() const & noexcept {
            shared_ptr_type res(*this);
            res.p_ = res.get_unmarked_ptr();
            return res;
        }

        shared_ptr_type get_unmarked() && noexcept {
            p_ = get_unmarked_ptr();
            return std::move(*this);
        }

        // Copy with the mark set. Legal on null (yields marked null).
        shared_ptr_type set_mark() const noexcept {
            shared_ptr_type res(*this);
            res.p_ = reinterpret_cast<U*>(reinterpret_cast<uintptr_t>(res.p_) | 1ULL);
            return res;
        }

        // Raw pointer with the mark bit still attached (the word's bit pattern).
        // Do not dereference. Used by the atomic owner and by tests comparing
        // identities.
        U* get_raw() const noexcept { return p_; }

    private:
        // The enclosing base reads and writes p_ directly: adopt_word() builds
        // a handle around a word whose reference is already counted, and the
        // pointers reach p_ only through the base's helpers.
        friend common_base;

        // The raw word: U* with the mark in bit 0. Null when no reference is owned.
        U* p_;

        // The pointer with the mark bit cleared: safe to dereference and to
        // pass to the hooks.
        U* get_unmarked_ptr() const noexcept {
            return unmarked_ptr(reinterpret_cast<uintptr_t>(p_));
        }
    }; // class shared_ptr_type

protected:
    // Only the pointers construct and destroy this base, as a subobject.
    common_base() = default;
    ~common_base() = default;

    // The pointee check, called at the top of EVERY member that calls a hook
    // or disposes of an object -- in shared_ptr_type: the adopting and copy
    // constructors, the destructor, both assignments and use_count(); in the
    // pointers: the adopting constructor, the destructor, load, store and the
    // CASes -- so that a non-conforming U is always reported by these
    // static_asserts' messages (a compiler may print the raw "no member named
    // 'DelRef'" of the hook call as well, but never instead). (The members
    // that call no hook -- the null and move constructors, the accessors, the
    // comparisons, get_unmarked() and set_mark(), which copy and so check
    // through the copy constructor -- do not check.) A static_assert in a
    // member BODY is evaluated when that body is instantiated, i.e. only once
    // U is complete; the class definitions themselves (instantiated by
    // `AtomicPtr<Node> next;` while Node is incomplete) are unconstrained, so
    // a type may hold a pointer to itself; the diagnostic appears at the first
    // instantiation of one of the members above, which is where it first
    // matters. Three separate static_asserts, so that each failure names its
    // own cause.
    //
    // The destructor assert: destructors do not throw -- the general C++
    // assumption (the standard makes them noexcept by default; one that
    // throws during a release would propagate out of a handle's destructor
    // and terminate), asserted here instead of reasoned around in every
    // releasing member, which can therefore all be noexcept. Kept separate
    // from the IntrusivePointee concept because it is not about the hooks:
    // the concept describes what the pointee must PROVIDE, this describes
    // what its destructor must not do, and a type that fails only this gets
    // this message.
    static constexpr void require_pointee() noexcept {
        Disposal::template require_pointee<U>();
        static_assert(std::is_convertible_v<U*, T*>,
            "intrusive atomic shared pointer <T, U>: U* must convert implicitly to T*");
        static_assert(std::is_nothrow_destructible_v<U>,
            "intrusive atomic shared pointer <T, U>: U's destructor must not throw "
            "(declare it noexcept, or do not declare it noexcept(false)): the pointer "
            "releases references inside destructors and noexcept members");
        // Bit 0 of the stored U* is the mark: it must be zero in every object's
        // address. (intr_shared_ptr additionally requires alignof(U) >= 4 for
        // its lock bit, asserted in its own Disposal::require_pointee.)
        static_assert(alignof(U) >= 2,
            "intrusive atomic shared pointer <T, U>: alignof(U) must be at least 2, "
            "bit 0 of the stored U* is the mark bit");
    } // require_pointee()

    // The word with the mark bit cleared, as the pointer it is: safe to
    // dereference and to pass to the hooks. Null for null and marked null.
    // The caller strips any other stolen bit (the spinlock pointer's lock
    // bit) before calling: this helper knows only the mark.
    static U* unmarked_ptr(uintptr_t word) noexcept {
        return reinterpret_cast<U*>(word & ~1ULL);
    }

    // The same for a handle: its pointee, unmarked (null for null and marked
    // null). What the pointers AddRef before publishing a handle's word.
    static U* unmarked_ptr(const shared_ptr_type& p) noexcept {
        return p.get_unmarked_ptr();
    }

    // Wrap a raw word (pointer and mark) in a shared_ptr_type WITHOUT touching
    // the count: the reference it will own was already taken (AddRef or
    // TryAddRef in the pointer's load()) or does not exist (null, marked null).
    // The word must carry no bit other than the pointer and the mark.
    static shared_ptr_type adopt_word(uintptr_t word) noexcept {
        shared_ptr_type res;
        res.p_ = reinterpret_cast<U*>(word);
        return res;
    }
}; // class common_base

} // namespace intr_shared_ptr_detail

#endif // INCLUDED_INTR_SHARED_PTR_COMMON_H_
