// Black-box contract tests of the POINTEE side of the intrusive atomic shared
// pointers: the hook contract, the pointee concepts and the count bases.
// Built twice, with ASan and with TSan (both at -O0), linked with
// mm_hp/mm_hp.cpp.
//
// What is tested and where its contract lives. Every expected value below is
// derived from the comments of the public interface, never from the function
// bodies:
// - intr_pointee.h, "The hooks (normative)": what AddRef(), DelRef(),
//   use_count() and TryAddRef() must do; the concept IntrusivePointee<U>; the
//   count base intr_pointee_base<Count> (its class comment: Count constraints,
//   special members, copy/assignment semantics).
// - intr_shared_ptr_hp.h: the concept HpIntrusivePointee<U> and the hazard
//   pointer base intr_pointee_base_hp<U, Count> (the comments above each:
//   CRTP on the final type, ONE hazard pointer base, any offset, layout), and
//   "Deferred reclamation" (a pointee whose count reaches 0 is retired and
//   destroyed at a later mm_hp scan).
// - intr_shared_ptr.h, the comment at the top: U must satisfy IntrusivePointee,
//   checked in member bodies, never at class scope; the T/U rules (U is the
//   dynamic type, deleted through U*); synchronous reclamation on the
//   releasing thread and the list of releasing members; the CAS's treatment of
//   `expected`.
// - intr_shared_ptr_common.h: the value type shared_ptr_type -- its null
//   constructors (constexpr noexcept), every member declared noexcept, and the
//   "Aliasing guarantee of the assignments" (`h = h->child`).
// - hp_drain.h: how a test observes "destroyed" under intr_shared_ptr_hp
//   (drain first; the drain must be quiescent; one drain per layer of a
//   destructor cascade).
// Why a separate file: the pointee contract is a surface of its own (the count
// base, the concepts, the hand-written-hook route kept first class); the
// pointers' own operations are covered by ../SharedPtr/atomic_shared_ptr_test.C
// and the HP seam tests. Here the pointers appear only as far as the pointee
// contract reaches into them (lifetime, reclamation, the concept check).
//
// Oracles used throughout:
// - Single-threaded use_count() values: exact (the hooks are the count).
// - Tracked: every test pointee that the pointers reclaim carries a unique id
//   and a per-id destruction counter. "Each object created in this test was
//   destroyed exactly once" is checked per object, not as balanced totals (a
//   double destruction of one object plus a leak of another would balance).
// - Under intr_shared_ptr_hp, "destroyed" is asserted only after a drain
//   (hp_drain.h); fixtures drain at SetUp, so mm_hp's retired list starts each
//   test empty and a single retirement cannot reach mm_hp's scan threshold
//   (at least 1000).
// - Memory orders of the hooks (relaxed / acq_rel / acquire) are not visible
//   to a single thread. The tests that pin them run two or more threads and
//   read plain (non-atomic) fields across threads, ordered ONLY by the hooks;
//   they fail only under TSan (as a data race report). In the ASan build on
//   x86-64 they pass regardless; there they still check values and counts.
//
// Table of contents (test suites, in file order):
//   1. IntrusivePointeeConcept, HpIntrusivePointeeConcept, PointeeCheckPlacement
//        compile-time partitions: one requirement violated per type.
//   2. IntrPointeeBaseStatic, IntrPointeeBaseTest (typed over Count),
//      IntrPointeeBaseNarrowTest
//        intr_pointee_base<Count>: Count constraints, protected special members,
//        noexcept, constexpr default constructor, count semantics of every
//        hook, copy/assignment/move semantics, the full Count range.
//   3. IntrPointeeBaseConcurrent
//        the hooks under concurrency: RMW atomicity, DelRef's 1->0 exactly
//        once, the orders' visible effects (TSan oracles), TryAddRef never
//        revives.
//   4. IntrPointeeBaseHpStatic, IntrPointeeBaseHpLayout, IntrPointeeBaseHp,
//      HpPointeeEndToEnd (typed over pointee), HpChainTest
//        intr_pointee_base_hp<U, Count>: the concept, bases, layout, offset,
//        copy semantics; end-to-end with intr_shared_ptr_hp, deferred
//        reclamation, exactly-once destruction after a drain.
//   5. SpinLifetime (typed over pointee), SpinTUTest, SpinChainTest
//        intr_shared_ptr over a base-derived pointee and a hand-written one:
//        synchronous destruction on the releasing thread, CAS and `expected`,
//        marks, T != U.
//   6. ValueTypeContract (typed over pointer)
//        shared_ptr_type's constexpr noexcept null constructors and every
//        member noexcept; the atomics' trivial constructors.
//   7. SpinAliasingTest, HpAliasingTest
//        `h = h->child` and `h = std::move(h->child)` with h the parent's only
//        owner: well-defined, with the documented use_count() afterwards.
//   8. SpinStressTest, HpStressTest
//        several threads load/store/CAS shared words; every pointee destroyed
//        exactly once when all handles are gone (after a drain for HP).
//
// Threads: single-threaded unless a test says otherwise; every multi-threaded
// test joins its threads before any drain, so the drain's quiescence
// precondition holds at every drain.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <latch>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "intr_pointee.h"
#include "intr_shared_ptr.h"
#include "intr_shared_ptr_hp.h"
#include "hp_drain_gtest.h"   // also registers the early mm_hp scan (see that header)

namespace {

// ===========================================================================
// Shared instrumentation
// ===========================================================================

// The value every live Tracked object carries in `value`; the destructor
// overwrites it. A reader that finds anything else read a destroyed object.
inline constexpr int k_magic = 0x5eed;

// Base of every test pointee that a pointer reclaims. Gives each object a
// unique id and counts destructions per id, so that a test can assert "every
// object created in this test was destroyed exactly once".
struct Tracked {
    // Capacity of the per-id table. One run of the binary creates far fewer
    // objects than this (the two stress tests dominate, with about 4 * 100000 *
    // 2.5/8 = 125000 objects each), even with --gtest_repeat=5; ids past it are
    // not tracked and all_destroyed_once() reports the overflow.
    static constexpr std::size_t max_objects = std::size_t{1} << 22;

    // The id the next constructed object gets; never reset (a test checks the
    // ids it created, [id at SetUp, id at check)).
    inline static std::atomic<std::size_t> next_id{0};

    // destructions[id]: how many times the object with that id was destroyed.
    // Relaxed counters: they must not create happens-before edges between
    // threads, or they could hide a race TSan is meant to find.
    inline static std::array<std::atomic<int>, max_objects> destructions{};

    // This object's id.
    const std::size_t id;

    // If not null, the destructor stores the destroying thread's id here (the
    // test owns the variable and reads it after the release).
    std::thread::id* const destroyed_on;

    // k_magic while alive; overwritten by the destructor.
    int value = k_magic;

    Tracked() noexcept : Tracked(nullptr) {}
    explicit Tracked(std::thread::id* where) noexcept
        : id(next_id.fetch_add(1, std::memory_order_relaxed)), destroyed_on(where) {}
    // A copy is a new object with a new id.
    Tracked(const Tracked&) noexcept : Tracked(nullptr) {}
    // Assignment keeps this object's identity.
    Tracked& operator=(const Tracked&) noexcept { return *this; }
    ~Tracked() {
        if (id < max_objects) destructions[id].fetch_add(1, std::memory_order_relaxed);
        if (destroyed_on != nullptr) *destroyed_on = std::this_thread::get_id();
        value = 0;
    }
}; // struct Tracked

// Number of destructions recorded for the object with this id.
int destructions_of(std::size_t id) {
    return Tracked::destructions[id].load(std::memory_order_relaxed);
}

// Number of objects with an id in [first, last) not destroyed yet.
std::size_t count_alive(std::size_t first, std::size_t last) {
    std::size_t alive = 0;
    for (std::size_t id = first; id != last && id != Tracked::max_objects; ++id) {
        if (destructions_of(id) == 0) ++alive;
    }
    return alive;
}

// Success iff every object with an id in [first, last) was destroyed exactly
// once. The message says how many were never destroyed (leaked, or not yet
// reclaimed) and how many more than once (double destruction).
::testing::AssertionResult all_destroyed_once(std::size_t first, std::size_t last) {
    if (last > Tracked::max_objects) {
        return ::testing::AssertionFailure() << "more than Tracked::max_objects objects created: raise it";
    }
    std::size_t never = 0;
    std::size_t repeated = 0;
    std::size_t first_bad = last;
    for (std::size_t id = first; id != last; ++id) {
        const int d = destructions_of(id);
        if (d == 0) ++never;
        if (d > 1) ++repeated;
        if (d != 1 && first_bad == last) first_bad = id;
    } // loop over the ids created by this test
    if (never == 0 && repeated == 0) {
        return ::testing::AssertionSuccess() << last - first << " objects, each destroyed once";
    }
    return ::testing::AssertionFailure() << never << " of " << last - first << " objects never destroyed, "
                                         << repeated << " destroyed more than once (first bad id "
                                         << first_bad << ")";
} // all_destroyed_once()

// Fixture of the tests whose pointees are reclaimed synchronously
// (intr_shared_ptr): remembers the first id this test creates.
class TrackedTest : public ::testing::Test {
protected:
    void SetUp() override { first_id_ = Tracked::next_id.load(std::memory_order_relaxed); }

    // Every object created since SetUp destroyed exactly once.
    ::testing::AssertionResult created_objects_destroyed_once() const {
        return all_destroyed_once(first_id_, Tracked::next_id.load(std::memory_order_relaxed));
    }

    // The first id created by this test.
    std::size_t first_id_ = 0;
}; // class TrackedTest

// Fixture of the tests whose pointees are reclaimed by mm_hp
// (intr_shared_ptr_hp). SetUp drains first: nothing retired by an earlier test
// is pending, and mm_hp's retired list starts empty, so (with the scan
// threshold at least 1000) a test's few retirements cannot trigger a scan by
// themselves. Quiescent: no test thread exists at SetUp.
class HpTrackedTest : public TrackedTest {
protected:
    void SetUp() override {
        drain_reclamation();
        TrackedTest::SetUp();
    }

    // Drains until every object created since SetUp is destroyed, at most
    // `max_rounds` rounds, then checks exactly-once destruction. One round
    // reclaims one layer of a destructor cascade (hp_drain.h), so max_rounds is
    // the depth of the deepest cascade the test leaves: 1 for pointees that own
    // nothing. A larger value would let a "needs one more drain" defect pass.
    ::testing::AssertionResult drain_then_check(int max_rounds) {
        const std::size_t last = Tracked::next_id.load(std::memory_order_relaxed);
        if (!drain_until([&] { return count_alive(first_id_, last) == 0; }, max_rounds)) {
            return all_destroyed_once(first_id_, last) << " after " << max_rounds << " drain round(s)";
        }
        return all_destroyed_once(first_id_, last);
    }
}; // class HpTrackedTest

// Runs body(t) on n threads (t = 0 .. n-1), released together by a latch so
// that they overlap, and joins them all.
template <typename Body>
void run_on_threads(int n, Body body) {
    std::latch start(n);
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (int t = 0; t != n; ++t) {
        threads.emplace_back([&start, &body, t] {
            start.arrive_and_wait();
            body(t);
        });
    } // loop starting the threads
    for (std::thread& thread : threads) thread.join();
} // run_on_threads()

// n AddRef() calls on p.
template <typename Pointee>
void add_refs(Pointee& p, long n) {
    for (long i = 0; i != n; ++i) p.AddRef();
}

// Drops `refs` references from p with DelRef(), one at a time. Returns the
// 1-based position of the single call that returned true, 0 if none did, -1 if
// more than one did. For p at count `refs`, the contract ("true iff THIS call
// made the 1 -> 0 transition") requires exactly `refs`.
template <typename Pointee>
long position_of_zero(Pointee& p, long refs) {
    long position = 0;
    for (long i = 1; i <= refs; ++i) {
        if (p.DelRef()) position = (position == 0) ? i : -1;
    }
    return position;
} // position_of_zero()

// ===========================================================================
// Test pointees
// ===========================================================================

// --- Count-only pointees (never owned by a pointer) ------------------------

// The smallest pointee: the count base and nothing else. Copyable.
template <typename Count>
struct CountOnly : intr_pointee_base<Count> {};

// A pointee whose other member is move-only: copy operations deleted
// implicitly, moves real (the base's copy constructor/assignment serve them).
template <typename Count>
struct MoveOnly : intr_pointee_base<Count> {
    std::unique_ptr<int> payload;
}; // struct MoveOnly

// Same small member after a narrow and a wide count (size comparison only).
struct NarrowTagged : intr_pointee_base<signed char> {
    char tag;
};
struct WideTagged : intr_pointee_base<long> {
    char tag;
};

// A plain field next to the count, written and read across threads in the
// concurrency tests, ordered only by the hooks.
struct PayloadPointee : intr_pointee_base<> {
    int payload = 0;
};

// Number of threads of the concurrency tests: enough for real overlap, few
// enough for TSan (the machine has more cores).
inline constexpr int k_threads = 4;

// One plain slot per thread (ZeroReacherSeesEveryReleasersWrites).
struct SlotsPointee : intr_pointee_base<> {
    std::array<int, k_threads> slots{};
};

// --- Pointees for intr_shared_ptr ------------------------------------------

// Hooks from the count base (the convenience route).
template <typename Count = long>
struct BaseNode : intr_pointee_base<Count>, Tracked {
    using Tracked::Tracked;
};

// Hand-written hooks (the first-class route): its own count, of a type other
// than long, with the orders of the normative text (AddRef relaxed, DelRef
// acq_rel, use_count a relaxed load) and no TryAddRef (the spinlock pointer
// never calls it). AddRef returns the new count: "a return value, if any, is
// ignored", so a hook with a result must be accepted and work.
struct HandNode : Tracked {
    using Tracked::Tracked;
    HandNode() noexcept = default;
    HandNode(const HandNode&) = delete;
    HandNode& operator=(const HandNode&) = delete;

    long AddRef() noexcept { return count_.fetch_add(1, std::memory_order_relaxed) + 1; }
    bool DelRef() noexcept { return count_.fetch_sub(1, std::memory_order_acq_rel) == 1; }
    long use_count() const noexcept { return count_.load(std::memory_order_relaxed); }

private:
    // The strong count; starts at 0, changed only by RMWs.
    std::atomic<int> count_{0};
}; // struct HandNode

// Names intr_shared_ptr<Self> while Self is incomplete (allowed because the
// concept is checked in member bodies, not at class scope).
struct ChainNode : intr_pointee_base<>, Tracked {
    intr_shared_ptr<ChainNode> next;
};

// T != U: the hooks live on Impl; the pointer exposes Iface. Iface has a
// NON-virtual destructor and sits at a nonzero offset inside Impl (after the
// count base), so deleting through Iface* would free the wrong address (ASan
// reports it) and skip ~Impl (Tracked sees no destruction).
struct Iface {
    int iface_value = 7;
};
struct Impl : intr_pointee_base<>, Iface, Tracked {
    using Tracked::Tracked;
};

// --- Pointees for intr_shared_ptr_hp -----------------------------------------

// The documented shape: intr_pointee_base_hp<Self> first.
struct HpNode : intr_pointee_base_hp<HpNode>, Tracked {
    using Tracked::Tracked;
};

// Another base precedes the hazard pointer base ("other bases may precede this
// one"; "the hazard pointer subobject may sit at any offset inside U"): the
// hazard pointer subobject is at least 32 bytes into the object.
struct Prefix {
    std::array<long, 4> words{1, 2, 3, 4};
};
struct HpNodeAfter : Prefix, intr_pointee_base_hp<HpNodeAfter>, Tracked {
    using Tracked::Tracked;
};

// A narrow count under the hazard pointer base.
struct HpNodeShort : intr_pointee_base_hp<HpNodeShort, short>, Tracked {
    using Tracked::Tracked;
};

// The narrowest count (alignment check only: alignof(U) >= 8 must still hold).
struct HpNodeChar : intr_pointee_base_hp<HpNodeChar, signed char> {};

// The count base used with the hazard pointer pointer WITHOUT
// intr_pointee_base_hp: intr_pointee_base is "usable with BOTH pointers", and a
// hand-assembled U derives from std::hazard_pointer_obj_base<U> itself.
struct ManualHpNode : intr_pointee_base<>, std::hazard_pointer_obj_base<ManualHpNode>, Tracked {
    using Tracked::Tracked;
};

// Names intr_shared_ptr_hp<Self> while Self is incomplete (the documented
// example of intr_pointee_base_hp).
struct HpChainNode : intr_pointee_base_hp<HpChainNode>, Tracked {
    intr_shared_ptr_hp<HpChainNode> next;
};

// --- Pointees owning a handle to another pointee (the aliasing tests) -----

// `child` is a shared_ptr_type stored INSIDE the pointee: the source of
// `h = h->child`, which the release of the parent destroys.
struct SpinParent : intr_pointee_base<>, Tracked {
    intr_shared_ptr<SpinParent>::shared_ptr_type child;
};
struct HpParent : intr_pointee_base_hp<HpParent>, Tracked {
    intr_shared_ptr_hp<HpParent>::shared_ptr_type child;
};

// --- Concept partitions: declarations only (unevaluated contexts) ---------
// Each type violates exactly one requirement, named by the type.

struct HandHooks {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HandHooks

struct AddRefReturnsCount {
    long AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct AddRefReturnsCount

struct NoAddRef {
    bool DelRef() noexcept;
    long use_count() const noexcept;
};

struct NoDelRef {
    void AddRef() noexcept;
    long use_count() const noexcept;
};

struct NoUseCount {
    void AddRef() noexcept;
    bool DelRef() noexcept;
};

struct AddRefMayThrow {
    void AddRef();
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct AddRefMayThrow

struct DelRefMayThrow {
    void AddRef() noexcept;
    bool DelRef();
    long use_count() const noexcept;
}; // struct DelRefMayThrow

struct UseCountMayThrow {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const;
}; // struct UseCountMayThrow

struct DelRefReturnsInt {
    void AddRef() noexcept;
    int DelRef() noexcept;
    long use_count() const noexcept;
}; // struct DelRefReturnsInt

struct DelRefReturnsVoid {
    void AddRef() noexcept;
    void DelRef() noexcept;
    long use_count() const noexcept;
}; // struct DelRefReturnsVoid

struct UseCountReturnsInt {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    int use_count() const noexcept;
}; // struct UseCountReturnsInt

struct UseCountReturnsUnsignedLong {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    unsigned long use_count() const noexcept;
}; // struct UseCountReturnsUnsignedLong

struct UseCountNotConst {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() noexcept;
}; // struct UseCountNotConst

// The base must be derived from publicly.
struct PrivateCountBase : private intr_pointee_base<> {};

// Conforming hooks, throwing destructor: the concept describes what the
// pointee provides, not its destructor, so it accepts this type; the pointers
// reject it separately (require_pointee(), not testable in-suite).
struct ThrowingDestructor {
    ~ThrowingDestructor() noexcept(false);
    void AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct ThrowingDestructor

struct HandHpHooks : std::hazard_pointer_obj_base<HandHpHooks> {
    void AddRef() noexcept;
    bool TryAddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HandHpHooks

struct HpNoTryAddRef : std::hazard_pointer_obj_base<HpNoTryAddRef> {
    void AddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HpNoTryAddRef

struct HpTryAddRefMayThrow : std::hazard_pointer_obj_base<HpTryAddRefMayThrow> {
    void AddRef() noexcept;
    bool TryAddRef();
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HpTryAddRefMayThrow

struct HpTryAddRefReturnsInt : std::hazard_pointer_obj_base<HpTryAddRefReturnsInt> {
    void AddRef() noexcept;
    int TryAddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HpTryAddRefReturnsInt

// Fails HpIntrusivePointee through its IntrusivePointee part.
struct HpNoAddRef : std::hazard_pointer_obj_base<HpNoAddRef> {
    bool TryAddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HpNoAddRef

struct HpNoHpBase {
    void AddRef() noexcept;
    bool TryAddRef() noexcept;
    bool DelRef() noexcept;
    long use_count() const noexcept;
}; // struct HpNoHpBase

// The count base alone: an IntrusivePointee, not an HpIntrusivePointee.
struct CountBaseOnly : intr_pointee_base<> {};

// Hazard pointer base not public.
struct HpPrivateHpBase : intr_pointee_base<>, private std::hazard_pointer_obj_base<HpPrivateHpBase> {};

// Hazard pointer base for a type other than U itself.
struct HpWrongHpBase : intr_pointee_base<>, std::hazard_pointer_obj_base<HpNode> {};

// Two hazard pointer bases of U, the second through another base: "makes the
// derived_from check of HpIntrusivePointee ambiguous and the type unusable".
struct TwoHpBases;
struct ExtraHpBase : std::hazard_pointer_obj_base<TwoHpBases> {};
struct TwoHpBases : intr_pointee_base_hp<TwoHpBases>, ExtraHpBase {};

// A scoped enumeration (a non-integral Count candidate).
enum class Color { red };

// True iff intr_pointee_base<C> names a valid specialization: a constraint
// failure is a substitution failure here, not a hard error.
template <typename C>
concept CountAccepted = requires { typename intr_pointee_base<C>; };

// ===========================================================================
// Type names for the typed suites (readable test names)
// ===========================================================================

template <typename T> struct TypeName;
template <> struct TypeName<long> { static constexpr const char* value = "long"; };
template <> struct TypeName<int> { static constexpr const char* value = "int"; };
template <> struct TypeName<short> { static constexpr const char* value = "short"; };
template <> struct TypeName<signed char> { static constexpr const char* value = "schar"; };
template <> struct TypeName<unsigned char> { static constexpr const char* value = "uchar"; };
template <> struct TypeName<unsigned short> { static constexpr const char* value = "ushort"; };
template <> struct TypeName<unsigned int> { static constexpr const char* value = "uint"; };
template <> struct TypeName<unsigned long> { static constexpr const char* value = "ulong"; };
template <> struct TypeName<long long> { static constexpr const char* value = "llong"; };
template <> struct TypeName<BaseNode<>> { static constexpr const char* value = "BaseNode"; };
template <> struct TypeName<HandNode> { static constexpr const char* value = "HandNode"; };
template <> struct TypeName<HpNode> { static constexpr const char* value = "HpNode"; };
template <> struct TypeName<HpNodeAfter> { static constexpr const char* value = "HpNodeAfter"; };
template <> struct TypeName<HpNodeShort> { static constexpr const char* value = "HpNodeShort"; };
template <> struct TypeName<ManualHpNode> { static constexpr const char* value = "ManualHpNode"; };
template <> struct TypeName<intr_shared_ptr<BaseNode<>>> { static constexpr const char* value = "Spin_BaseNode"; };
template <> struct TypeName<intr_shared_ptr<HandNode>> { static constexpr const char* value = "Spin_HandNode"; };
template <> struct TypeName<intr_shared_ptr_hp<HpNode>> { static constexpr const char* value = "Hp_HpNode"; };
template <> struct TypeName<intr_shared_ptr_hp<HpNodeAfter>> { static constexpr const char* value = "Hp_HpNodeAfter"; };

// GoogleTest name generator over TypeName.
struct TypeNames {
    template <typename T>
    static std::string GetName(int) { return TypeName<T>::value; }
};

// ===========================================================================
// 1. Concept partitions
// ===========================================================================

// Clauses: IntrusivePointee<U> = the three hooks AddRef(), DelRef(),
// use_count(), all noexcept, DelRef returning bool, use_count() const returning
// long; AddRef's return value, if any, is ignored (so any return type is
// accepted); TryAddRef is not required. Both routes are accepted: hand-written
// hooks and the count base. Each rejected type violates exactly one clause.
TEST(IntrusivePointeeConcept, Partitions) {
    static_assert(IntrusivePointee<HandHooks>);
    static_assert(IntrusivePointee<AddRefReturnsCount>);
    static_assert(IntrusivePointee<CountOnly<long>>);
    static_assert(IntrusivePointee<CountOnly<signed char>>);
    static_assert(IntrusivePointee<BaseNode<>>);
    static_assert(IntrusivePointee<HandNode>);
    static_assert(IntrusivePointee<HpNode>);       // the HP base carries the same hooks
    static_assert(IntrusivePointee<TwoHpBases>);   // the hooks themselves are unambiguous
    static_assert(IntrusivePointee<ThrowingDestructor>);   // the destructor is checked apart

    static_assert(!IntrusivePointee<NoAddRef>);
    static_assert(!IntrusivePointee<NoDelRef>);
    static_assert(!IntrusivePointee<NoUseCount>);
    static_assert(!IntrusivePointee<AddRefMayThrow>);
    static_assert(!IntrusivePointee<DelRefMayThrow>);
    static_assert(!IntrusivePointee<UseCountMayThrow>);
    static_assert(!IntrusivePointee<DelRefReturnsInt>);
    static_assert(!IntrusivePointee<DelRefReturnsVoid>);
    static_assert(!IntrusivePointee<UseCountReturnsInt>);
    static_assert(!IntrusivePointee<UseCountReturnsUnsignedLong>);
    static_assert(!IntrusivePointee<UseCountNotConst>);
    static_assert(!IntrusivePointee<PrivateCountBase>);
} // Partitions

// Clauses: HpIntrusivePointee<U> = IntrusivePointee<U> + a noexcept bool
// TryAddRef() + U derived publicly and unambiguously from
// std::hazard_pointer_obj_base<U>. Accepted: intr_pointee_base_hp<U> at the
// front or after another base, with a narrow Count, the count base plus a
// hand-added hazard pointer base, fully hand-written hooks. Rejected, one
// violation each: no TryAddRef, throwing TryAddRef, TryAddRef not bool, the
// IntrusivePointee part violated, no hazard pointer base, the count base alone,
// a private hazard pointer base, a hazard pointer base of another type, two
// hazard pointer bases of U.
TEST(HpIntrusivePointeeConcept, Partitions) {
    static_assert(HpIntrusivePointee<HpNode>);
    static_assert(HpIntrusivePointee<HpNodeAfter>);
    static_assert(HpIntrusivePointee<HpNodeShort>);
    static_assert(HpIntrusivePointee<HpNodeChar>);
    static_assert(HpIntrusivePointee<ManualHpNode>);
    static_assert(HpIntrusivePointee<HpChainNode>);
    static_assert(HpIntrusivePointee<HandHpHooks>);

    static_assert(!HpIntrusivePointee<HpNoTryAddRef>);
    static_assert(!HpIntrusivePointee<HpTryAddRefMayThrow>);
    static_assert(!HpIntrusivePointee<HpTryAddRefReturnsInt>);
    static_assert(!HpIntrusivePointee<HpNoAddRef>);
    static_assert(!HpIntrusivePointee<HpNoHpBase>);
    static_assert(!HpIntrusivePointee<CountBaseOnly>);
    static_assert(!HpIntrusivePointee<BaseNode<>>);
    static_assert(!HpIntrusivePointee<HpPrivateHpBase>);
    static_assert(!HpIntrusivePointee<HpWrongHpBase>);
    static_assert(!HpIntrusivePointee<TwoHpBases>);
} // Partitions

// Clause (intr_shared_ptr.h top, intr_pointee.h, intr_shared_ptr_hp.h): the
// pointee concept is checked by static_assert inside member function bodies,
// NEVER at class scope. Hence the class templates instantiate with a U that
// violates the concept (sizeof instantiates the class definition, not the
// member bodies); the positive use of the same rule -- naming the pointer of
// an incomplete Self -- is ChainNode and HpChainNode, exercised in
// SpinChainTest and HpChainTest. The failing static_asserts themselves (the
// concept, U* -> T*, and the nothrow destructor) fire only in a program that
// does not compile, so they are not testable here: that needs a compile-fail
// test outside GoogleTest.
TEST(PointeeCheckPlacement, PointerClassesInstantiateWithANonConformingPointee) {
    static_assert(sizeof(intr_shared_ptr<NoDelRef>) > 0);
    static_assert(sizeof(intr_shared_ptr<NoDelRef>::shared_ptr_type) > 0);
    static_assert(sizeof(intr_shared_ptr<ThrowingDestructor>) > 0);
    static_assert(sizeof(intr_shared_ptr<ThrowingDestructor>::shared_ptr_type) > 0);
    static_assert(sizeof(intr_shared_ptr_hp<HpNoHpBase>) > 0);
    static_assert(sizeof(intr_shared_ptr_hp<HpNoHpBase>::shared_ptr_type) > 0);
}

// ===========================================================================
// 2. intr_pointee_base<Count>
// ===========================================================================

// Clause: "Count: ... Requirements, enforced at compile time: an integral type
// other than bool, not cv-qualified (the requires-clause: a violation is
// 'constraints not satisfied' at the point of use)". Accepted: signed and
// unsigned integers of every width, the character types; rejected: bool,
// floating point, pointers, enumerations, class types, and cv-qualified
// integers. The third requirement, std::atomic<Count> lock-free, is a
// class-scope static_assert (a hard error at instantiation, not a constraint),
// so a requires-expression cannot see it; it needs a compile-fail test, and no
// standard integral type is a counterexample on x86-64 anyway.
TEST(IntrPointeeBaseStatic, CountConstraints) {
    static_assert(std::is_same_v<intr_pointee_base<>, intr_pointee_base<long>>);

    static_assert(CountAccepted<long>);
    static_assert(CountAccepted<int>);
    static_assert(CountAccepted<short>);
    static_assert(CountAccepted<signed char>);
    static_assert(CountAccepted<long long>);
    static_assert(CountAccepted<unsigned char>);
    static_assert(CountAccepted<unsigned short>);
    static_assert(CountAccepted<unsigned int>);
    static_assert(CountAccepted<unsigned long>);
    static_assert(CountAccepted<char>);
    static_assert(CountAccepted<char8_t>);
    static_assert(CountAccepted<char32_t>);

    static_assert(!CountAccepted<bool>);
    static_assert(!CountAccepted<float>);
    static_assert(!CountAccepted<double>);
    static_assert(!CountAccepted<long*>);
    static_assert(!CountAccepted<Color>);
    static_assert(!CountAccepted<std::atomic<long>>);
    static_assert(!CountAccepted<const long>);
    static_assert(!CountAccepted<volatile int>);
    static_assert(!CountAccepted<const volatile short>);
    static_assert(!CountAccepted<const bool>);
} // CountConstraints

// Clause: "Special members, all protected (the class is a base and never the
// dynamic type of anything)": the base alone can be neither constructed,
// copied, assigned nor destroyed from outside.
TEST(IntrPointeeBaseStatic, SpecialMembersAreProtected) {
    using B = intr_pointee_base<>;
    static_assert(!std::is_default_constructible_v<B>);
    static_assert(!std::is_copy_constructible_v<B>);
    static_assert(!std::is_move_constructible_v<B>);
    static_assert(!std::is_copy_assignable_v<B>);
    static_assert(!std::is_move_assignable_v<B>);
    static_assert(!std::is_destructible_v<B>);
    static_assert(!std::is_default_constructible_v<intr_pointee_base<short>>);
    static_assert(!std::is_destructible_v<intr_pointee_base<short>>);
} // SpecialMembersAreProtected

// Clause: "its destructor is non-virtual on purpose": no virtual destructor,
// and the base adds no polymorphism to a pointee.
TEST(IntrPointeeBaseStatic, NonVirtualDestructorNotPolymorphic) {
    static_assert(!std::has_virtual_destructor_v<intr_pointee_base<>>);
    static_assert(!std::is_polymorphic_v<intr_pointee_base<>>);
    static_assert(!std::is_polymorphic_v<CountOnly<long>>);
}

// Clauses: "Deriving from this base therefore makes a type copyable only if its
// other members are"; "move operations: not declared; a move falls back to the
// copy operations" -- so a type with a move-only member is move-only (copy
// deleted, moves available through the base's copy operations).
TEST(IntrPointeeBaseStatic, CopyabilityFollowsOtherMembers) {
    static_assert(std::is_copy_constructible_v<CountOnly<long>>);
    static_assert(std::is_copy_assignable_v<CountOnly<long>>);
    static_assert(!std::is_copy_constructible_v<MoveOnly<long>>);
    static_assert(!std::is_copy_assignable_v<MoveOnly<long>>);
    static_assert(std::is_move_constructible_v<MoveOnly<long>>);
    static_assert(std::is_move_assignable_v<MoveOnly<long>>);
}

// Clause: "A narrower type saves space in the pointee." The only size promise
// of the count base; checked on the smallest pointee where it matters (one
// char beside the count).
TEST(IntrPointeeBaseStatic, NarrowCountSavesSpace) {
    static_assert(sizeof(NarrowTagged) < sizeof(WideTagged));
}

// Typed over the count type: every clause of the hooks must hold whatever Count
// is (signed, unsigned, narrow, wide).
template <typename Count>
class IntrPointeeBaseTest : public ::testing::Test {};
using CountTypes = ::testing::Types<long, int, short, signed char, unsigned char, unsigned int, unsigned long,
                                    long long>;
TYPED_TEST_SUITE(IntrPointeeBaseTest, CountTypes, TypeNames);

// Clauses: the base provides all four hooks; all four are noexcept; DelRef()
// and TryAddRef() return bool; use_count() is const and returns long "whatever
// the count's type".
TYPED_TEST(IntrPointeeBaseTest, HooksAreNoexceptWithDocumentedTypes) {
    using P = CountOnly<TypeParam>;
    static_assert(IntrusivePointee<P>);
    static_assert(noexcept(std::declval<P&>().AddRef()));
    static_assert(noexcept(std::declval<P&>().DelRef()));
    static_assert(noexcept(std::declval<P&>().TryAddRef()));
    static_assert(noexcept(std::declval<const P&>().use_count()));
    static_assert(std::is_same_v<decltype(std::declval<P&>().DelRef()), bool>);
    static_assert(std::is_same_v<decltype(std::declval<P&>().TryAddRef()), bool>);
    static_assert(std::is_same_v<decltype(std::declval<const P&>().use_count()), long>);
} // HooksAreNoexceptWithDocumentedTypes

// Clauses: special members "all noexcept"; "default constructor: count 0,
// constexpr". Checked on a derived type (the base's own are protected): its
// implicit members are noexcept/constexpr only if the base's are. constinit
// proves constant initialization, i.e. a constexpr default constructor.
TYPED_TEST(IntrPointeeBaseTest, SpecialMembersNoexceptDefaultConstexpr) {
    using P = CountOnly<TypeParam>;
    static_assert(std::is_nothrow_default_constructible_v<P>);
    static_assert(std::is_nothrow_copy_constructible_v<P>);
    static_assert(std::is_nothrow_copy_assignable_v<P>);
    static_assert(std::is_nothrow_move_constructible_v<P>);
    static_assert(std::is_nothrow_move_assignable_v<P>);
    static_assert(std::is_nothrow_destructible_v<P>);

    static constinit P constant_initialized;
    EXPECT_EQ(constant_initialized.use_count(), 0);
} // SpecialMembersNoexceptDefaultConstexpr

// Clause: "The count starts at 0 for a freshly constructed object (owned by no
// one)."
TYPED_TEST(IntrPointeeBaseTest, FreshObjectCountIsZero) {
    const CountOnly<TypeParam> fresh;
    EXPECT_EQ(fresh.use_count(), 0);
}

// Clause: AddRef "Increment the count. Precondition: the caller owns a
// reference, or the object is fresh (count 0, never owned)": the fresh 0 -> 1
// (adoption) and an owned n -> n+1.
TYPED_TEST(IntrPointeeBaseTest, AddRefIncrementsFromFreshAndFromOwned) {
    CountOnly<TypeParam> p;
    p.AddRef();
    EXPECT_EQ(p.use_count(), 1);
    p.AddRef();
    EXPECT_EQ(p.use_count(), 2);
    EXPECT_EQ(position_of_zero(p, 2), 2);
}

// Clause: DelRef "return true iff THIS call made the 1 -> 0 transition".
// Boundaries: a single reference (the first DelRef is the 1 -> 0), and 100
// references (only the 100th is; 100 fits every Count in the list).
TYPED_TEST(IntrPointeeBaseTest, DelRefTrueExactlyOnOneToZero) {
    CountOnly<TypeParam> single;
    single.AddRef();
    EXPECT_EQ(position_of_zero(single, 1), 1);
    EXPECT_EQ(single.use_count(), 0);

    CountOnly<TypeParam> many;
    add_refs(many, 100);
    EXPECT_EQ(position_of_zero(many, 100), 100) << "0 = no call reported 1 -> 0, -1 = several did";
    EXPECT_EQ(many.use_count(), 0);
} // DelRefTrueExactlyOnOneToZero

// Clause: TryAddRef "returns false iff it observed a count of 0, in which case
// the count is left at 0 (... must never be revived)". Both zeros: a fresh
// object, and one whose count went 1 -> 0. Repeated calls: still 0.
TYPED_TEST(IntrPointeeBaseTest, TryAddRefOnZeroFailsAndLeavesZero) {
    CountOnly<TypeParam> fresh;
    for (int i = 0; i != 3; ++i) {
        EXPECT_FALSE(fresh.TryAddRef());
        EXPECT_EQ(fresh.use_count(), 0);
    }

    CountOnly<TypeParam> released;
    released.AddRef();
    ASSERT_TRUE(released.DelRef());
    for (int i = 0; i != 3; ++i) {
        EXPECT_FALSE(released.TryAddRef());
        EXPECT_EQ(released.use_count(), 0);
    }
} // TryAddRefOnZeroFailsAndLeavesZero

// Clause: TryAddRef "Increment the strong count if and only if it is nonzero.
// Returns true iff it incremented."
TYPED_TEST(IntrPointeeBaseTest, TryAddRefOnNonzeroIncrements) {
    CountOnly<TypeParam> p;
    p.AddRef();
    EXPECT_TRUE(p.TryAddRef());
    EXPECT_EQ(p.use_count(), 2);
    EXPECT_TRUE(p.TryAddRef());
    EXPECT_EQ(p.use_count(), 3);
    EXPECT_EQ(position_of_zero(p, 3), 3);
} // TryAddRefOnNonzeroIncrements

// Clause: "copy constructor: count 0 -- a copy of a pointee is a NEW object
// that nobody owns yet"; the original keeps its count.
TYPED_TEST(IntrPointeeBaseTest, CopyConstructionIsAFreshObject) {
    CountOnly<TypeParam> original;
    add_refs(original, 3);
    const CountOnly<TypeParam> copy(original);
    EXPECT_EQ(copy.use_count(), 0);
    EXPECT_EQ(original.use_count(), 3);
}

// Clause: "copy assignment: leaves *this's count untouched"; the source keeps
// its count too. Self-assignment (through an alias, to keep the compiler from
// flagging it) is the boundary where "untouched" and "copied" coincide in value
// only if the count is left alone.
TYPED_TEST(IntrPointeeBaseTest, CopyAssignmentKeepsBothCounts) {
    CountOnly<TypeParam> source;
    add_refs(source, 5);
    CountOnly<TypeParam> target;
    add_refs(target, 2);
    target = source;
    EXPECT_EQ(target.use_count(), 2);
    EXPECT_EQ(source.use_count(), 5);

    CountOnly<TypeParam>& alias = target;
    target = alias;
    EXPECT_EQ(target.use_count(), 2);
} // CopyAssignmentKeepsBothCounts

// Clause: "move operations: not declared; a move falls back to the copy
// operations above, with the same meaning (the source's count is untouched as
// well, since its owners still own it)". On MoveOnly the derived move is real
// (the unique_ptr moves), so this pins the count semantics of a genuine move.
TYPED_TEST(IntrPointeeBaseTest, MoveFallsBackToCopySemantics) {
    MoveOnly<TypeParam> source;
    source.payload = std::make_unique<int>(9);
    add_refs(source, 4);
    MoveOnly<TypeParam> moved(std::move(source));
    EXPECT_EQ(moved.use_count(), 0) << "a move-constructed pointee must be a new, unowned object";
    EXPECT_EQ(source.use_count(), 4) << "the moved-from pointee's owners still own it";
    ASSERT_NE(moved.payload, nullptr);
    EXPECT_EQ(*moved.payload, 9);

    MoveOnly<TypeParam> target;
    target.AddRef();
    target = std::move(moved);
    EXPECT_EQ(target.use_count(), 1);
    EXPECT_EQ(moved.use_count(), 0);
} // MoveFallsBackToCopySemantics

// The narrow counts, driven to their maximum.
template <typename Count>
class IntrPointeeBaseNarrowTest : public ::testing::Test {};
using NarrowCountTypes = ::testing::Types<signed char, unsigned char, short, unsigned short>;
TYPED_TEST_SUITE(IntrPointeeBaseNarrowTest, NarrowCountTypes, TypeNames);

// Clauses: the BOUND -- "never more than std::numeric_limits<Count>::max()
// simultaneous owners ... of one object" -- so exactly max owners is within
// the contract; use_count() is "the current count, as long whatever the
// count's type"; DelRef's 1 -> 0 is reported once, at the end. Boundary: the
// bound itself (127, 255, 32767, 65535), where a narrowing or sign error in the
// widening would show (unsigned char 255 must read 255, not -1). The wide types
// cannot be driven to their maximum in a test.
TYPED_TEST(IntrPointeeBaseNarrowTest, UseCountReportsTheFullRangeAsLong) {
    // Nothing beyond the bound is attempted: one more owner is undefined
    // behaviour for a signed Count and a wrapped count for an unsigned one.
    constexpr long max = std::numeric_limits<TypeParam>::max();
    CountOnly<TypeParam> p;
    add_refs(p, max);
    EXPECT_EQ(p.use_count(), max);
    EXPECT_EQ(position_of_zero(p, max), max);
    EXPECT_EQ(p.use_count(), 0);
} // UseCountReportsTheFullRangeAsLong

// ===========================================================================
// 3. The hooks under concurrency (intr_pointee_base<long>)
// ===========================================================================

// Clause: "EVERY operation that changes the count is an atomic
// read-modify-write". Oracle: no lost update -- k_threads threads each do
// per_thread AddRef() and per_thread TryAddRef() on one object that starts at 1;
// the count must end at exactly 1 + 2*k_threads*per_thread, and no TryAddRef
// may fail (the count is never 0 during the run).
TEST(IntrPointeeBaseConcurrent, IncrementsLoseNothing) {
    constexpr long per_thread = 20'000;
    CountOnly<long> object;
    object.AddRef();
    std::atomic<long> failed_try{0};
    run_on_threads(k_threads, [&](int) {
        for (long i = 0; i != per_thread; ++i) {
            object.AddRef();
            if (!object.TryAddRef()) failed_try.fetch_add(1, std::memory_order_relaxed);
        }
    });
    EXPECT_EQ(failed_try.load(), 0) << "TryAddRef failed on a nonzero count";
    EXPECT_EQ(object.use_count(), 1 + 2*k_threads*per_thread) << "an increment was lost";
} // IncrementsLoseNothing

// Clauses: DelRef is an atomic RMW and returns true iff THIS call made the
// 1 -> 0 transition; "0 is reached again exactly once, by the DelRef() that
// returns true". Oracle: an object at count k_threads*per_thread, every thread
// drops per_thread references concurrently; exactly one call in all threads
// returns true and the count ends at 0. Repeated on fresh objects so that the
// winning thread varies.
TEST(IntrPointeeBaseConcurrent, DecrementsReportZeroExactlyOnce) {
    constexpr int rounds = 50;
    constexpr long per_thread = 1000;
    int bad_rounds = 0;
    for (int r = 0; r != rounds; ++r) {
        CountOnly<long> object;
        add_refs(object, k_threads*per_thread);
        std::atomic<int> zero_transitions{0};
        run_on_threads(k_threads, [&](int) {
            for (long i = 0; i != per_thread; ++i) {
                if (object.DelRef()) zero_transitions.fetch_add(1, std::memory_order_relaxed);
            }
        });
        if (zero_transitions.load() != 1 || object.use_count() != 0) ++bad_rounds;
    } // loop over rounds
    EXPECT_EQ(bad_rounds, 0) << "rounds where the 1 -> 0 transition was reported other than once";
} // DecrementsReportZeroExactlyOnce

// Clause: DelRef's acq_rel -- "acquire gives the thread that reaches 0 -- and
// therefore destroys or retires the object -- the history of every thread that
// released a reference before it ... the destructor reading fields other owners
// wrote". Oracle (TSan): each of k_threads owners writes its own PLAIN slot of
// the object, then drops its reference; the thread whose DelRef returns true
// reads every slot. Only DelRef orders those writes before that read; a DelRef
// without release, or without acquire, is reported by TSan as a data race.
// Values and "exactly one zero per object" are checked in both builds.
// A template over the pointee (SlotsPointee here), like the two helpers below,
// so that the detectors' sensitivity can be checked by instantiating them over
// hand-written pointees with one clause weakened each (DelRef relaxed; DelRef
// release-only; every count read in TryAddRef relaxed; TryAddRef a plain
// increment that revives): TSan (or a value check) fails each of those, while
// intr_pointee_base stays clean.
template <typename Slots>
void run_zero_reacher() {
    constexpr int rounds = 500;
    std::vector<Slots> objects(rounds);
    for (Slots& object : objects) add_refs(object, k_threads);
    std::vector<std::atomic<int>> zero_transitions(rounds);
    std::atomic<long> stale_slots{0};
    run_on_threads(k_threads, [&](int t) {
        for (int r = 0; r != rounds; ++r) {
            Slots& object = objects[r];
            object.slots[t] = r*k_threads + t + 1;
            if (object.DelRef()) {
                zero_transitions[r].fetch_add(1, std::memory_order_relaxed);
                for (int u = 0; u != k_threads; ++u) {
                    if (object.slots[u] != r*k_threads + u + 1) stale_slots.fetch_add(1, std::memory_order_relaxed);
                }
            } // this thread made the 1 -> 0 transition: read every owner's slot
        } // loop over objects
    }); // thread body
    EXPECT_EQ(stale_slots.load(), 0) << "the zero-reaching thread did not see an owner's write";
    int bad_objects = 0;
    for (const std::atomic<int>& z : zero_transitions) {
        if (z.load() != 1) ++bad_objects;
    }
    EXPECT_EQ(bad_objects, 0) << "objects whose 1 -> 0 was reported other than once";
} // run_zero_reacher()

TEST(IntrPointeeBaseConcurrent, ZeroReacherSeesEveryReleasersWrites) {
    run_zero_reacher<SlotsPointee>();
}

// Owner/observer handshake behind the two tests below. The owner writes a PLAIN
// payload, drops its reference with DelRef, then raises a RELAXED flag; the
// observer waits for the flag, calls TryAddRef, and reads the payload. The
// relaxed flag carries no happens-before edge, so the observer's read is
// ordered after the owner's write ONLY through the count: the owner's DelRef
// (release) and the observer's TryAddRef (acquire). TSan reports a data race if
// TryAddRef's read of the count does not acquire. The flag still makes the
// outcome deterministic: when the observer runs TryAddRef, the owner's DelRef
// is already in the count's modification order.
//   observer_shares == false: the owner holds the only reference, so its DelRef
//     is the 1 -> 0, and TryAddRef must observe 0 and return false.
//   observer_shares == true: the test holds a second reference, the owner's
//     DelRef is 2 -> 1, and TryAddRef must succeed (1 -> 2).
template <typename Payload>
void run_release_then_try_add_ref(bool observer_shares) {
    constexpr int written = 42;
    Payload object;
    object.AddRef();                        // the owner's reference
    if (observer_shares) object.AddRef();   // the test's reference: keeps the count above 0
    std::atomic<bool> released{false};
    bool owner_reached_zero = false;
    bool observer_incremented = false;
    int observed_payload = 0;
    std::thread owner([&] {
        object.payload = written;
        owner_reached_zero = object.DelRef();
        released.store(true, std::memory_order_relaxed);
    });
    std::thread observer([&] {
        while (!released.load(std::memory_order_relaxed)) std::this_thread::yield();
        observer_incremented = object.TryAddRef();
        observed_payload = object.payload;
    });
    owner.join();
    observer.join();
    EXPECT_EQ(observed_payload, written);
    if (observer_shares) {
        EXPECT_FALSE(owner_reached_zero);
        EXPECT_TRUE(observer_incremented) << "TryAddRef failed on a count of 1";
        EXPECT_EQ(object.use_count(), 2);
        EXPECT_EQ(position_of_zero(object, 2), 2);
    } else {
        EXPECT_TRUE(owner_reached_zero);
        EXPECT_FALSE(observer_incremented) << "TryAddRef revived an object whose count reached 0";
        EXPECT_EQ(object.use_count(), 0) << "TryAddRef changed a count of 0";
    } // which outcome the count dictates
} // run_release_then_try_add_ref()

// Clause: TryAddRef -- "EVERY observed 0 was read with acquire"; "An observed 0
// must synchronize with the release sequence headed by the DelRef that produced
// it". Oracle: TSan (see run_release_then_try_add_ref()); deterministic outcome
// in both builds (false, count left at 0).
TEST(IntrPointeeBaseConcurrent, ObservedZeroSynchronizesWithTheZeroingDelRef) {
    run_release_then_try_add_ref<PayloadPointee>(false);
}

// Clause: TryAddRef -- "the load that observes 0 is ACQUIRE; the CAS is ACQUIRE
// on success": a successful TryAddRef reads the count with acquire whether the
// value comes from its first load or from its CAS, so it synchronizes with the
// owner's releasing DelRef that wrote that value. Oracle: TSan; deterministic
// outcome (true, count 2) in both builds.
TEST(IntrPointeeBaseConcurrent, SuccessfulTryAddRefSynchronizesWithTheRelease) {
    run_release_then_try_add_ref<PayloadPointee>(true);
}

// Clauses: TryAddRef never revives ("an object at 0 ... must never be
// revived"), and the 1 -> 0 happens exactly once, under a real race. Per round,
// one object at count 1: the owner writes a plain payload and drops the
// reference; the contender loops { TryAddRef; if it got a reference, DelRef }
// until it either observes 0 or itself makes the 1 -> 0 transition (when its
// DelRef followed the owner's). Oracle per round: exactly one DelRef (owner's or
// contender's) returned true and the count ends at 0. The contender reads the
// payload only where the contract orders it after the owner's write: after
// observing 0 (acquire) or after its own 1 -> 0 DelRef (acq_rel) -- TSan
// reports otherwise. Which path a round takes depends on timing; the counts of
// both are recorded as test properties (--gtest_output=xml) for inspection, not
// asserted (the scheduler, not the contract, decides them).
template <typename Payload>
void run_try_add_ref_race() {
    constexpr int rounds = 2000;
    std::vector<Payload> objects(rounds);
    for (Payload& object : objects) object.AddRef();   // the owner's reference
    std::vector<char> owner_zero(rounds, 0);
    std::vector<char> contender_zero(rounds, 0);
    std::vector<char> observed_zero(rounds, 0);
    std::atomic<long> wrong_payload{0};
    std::barrier<> round_start(2);
    std::thread owner([&] {
        for (int r = 0; r != rounds; ++r) {
            round_start.arrive_and_wait();
            objects[r].payload = r + 1;
            owner_zero[r] = objects[r].DelRef();
        }
    });
    std::thread contender([&] {
        for (int r = 0; r != rounds; ++r) {
            round_start.arrive_and_wait();
            Payload& object = objects[r];
            while (true) {
                if (!object.TryAddRef()) {
                    observed_zero[r] = 1;
                    break;
                }
                if (object.DelRef()) {
                    contender_zero[r] = 1;
                    break;
                }
            } // loop until 0 is observed or this thread made the 1 -> 0
            if (object.payload != r + 1) wrong_payload.fetch_add(1, std::memory_order_relaxed);
        } // loop over rounds
    }); // contender thread
    owner.join();
    contender.join();
    int bad_rounds = 0;
    int observed_zero_rounds = 0;
    int contender_zero_rounds = 0;
    for (int r = 0; r != rounds; ++r) {
        if (owner_zero[r] + contender_zero[r] != 1 || objects[r].use_count() != 0) ++bad_rounds;
        observed_zero_rounds += observed_zero[r];
        contender_zero_rounds += contender_zero[r];
    } // loop over rounds
    EXPECT_EQ(bad_rounds, 0) << "rounds with the 1 -> 0 reported other than once, or a revived count";
    EXPECT_EQ(wrong_payload.load(), 0);
    ::testing::Test::RecordProperty("observed_zero_rounds", observed_zero_rounds);
    ::testing::Test::RecordProperty("contender_zero_rounds", contender_zero_rounds);
} // run_try_add_ref_race()

TEST(IntrPointeeBaseConcurrent, TryAddRefRacingTheLastReleaseNeverRevives) {
    run_try_add_ref_race<PayloadPointee>();
}

// ===========================================================================
// 4. intr_pointee_base_hp<U, Count> and intr_shared_ptr_hp
// ===========================================================================

// Clauses: "the four hooks of intr_pointee_base<Count> ... plus the hazard
// pointer base std::hazard_pointer_obj_base<U>" (the class's base list: both
// public); Count defaults to long and is passed through; "its destructor is
// non-virtual"; "Normative: alignof(U) >= 2, so that bit 0 of a U* is free for
// the mark" -- checked for the narrowest Count as well, where the count base
// alone would not guarantee it. (mm_hp's "24 bytes, 8-aligned" is marked
// informative and is not pinned.)
TEST(IntrPointeeBaseHpStatic, BasesAndAlignment) {
    static_assert(std::derived_from<intr_pointee_base_hp<HpNode>, intr_pointee_base<long>>);
    static_assert(std::derived_from<intr_pointee_base_hp<HpNodeShort, short>, intr_pointee_base<short>>);
    static_assert(std::derived_from<intr_pointee_base_hp<HpNode>, std::hazard_pointer_obj_base<HpNode>>);
    static_assert(std::derived_from<HpNode, std::hazard_pointer_obj_base<HpNode>>);
    static_assert(!std::has_virtual_destructor_v<intr_pointee_base_hp<HpNode>>);

    static_assert(alignof(HpNode) >= 2);
    static_assert(alignof(HpNodeShort) >= 2);
    static_assert(alignof(HpNodeChar) >= 2);
    static_assert(alignof(HpNodeAfter) >= 2);
} // BasesAndAlignment

// Clause: "Special members: protected, as for both bases (this class is only
// ever a base ...), and all noexcept"; "Moves fall back to these copy
// operations". Protected: the base alone is not constructible, copyable or
// destructible from outside. Noexcept: checked on derived pointees whose other
// members are nothrow (their implicit members are noexcept only if the base's
// are).
TEST(IntrPointeeBaseHpStatic, SpecialMembersProtectedAndNoexcept) {
    using B = intr_pointee_base_hp<HpNode>;
    static_assert(!std::is_default_constructible_v<B>);
    static_assert(!std::is_copy_constructible_v<B>);
    static_assert(!std::is_copy_assignable_v<B>);
    static_assert(!std::is_destructible_v<B>);

    static_assert(std::is_nothrow_default_constructible_v<HpNodeChar>);
    static_assert(std::is_nothrow_copy_constructible_v<HpNodeChar>);
    static_assert(std::is_nothrow_copy_assignable_v<HpNodeChar>);
    static_assert(std::is_nothrow_move_constructible_v<HpNodeChar>);
    static_assert(std::is_nothrow_move_assignable_v<HpNodeChar>);
    static_assert(std::is_nothrow_destructible_v<HpNodeChar>);
    static_assert(std::is_nothrow_copy_constructible_v<HpNode>);
    static_assert(std::is_nothrow_copy_assignable_v<HpNode>);
} // SpecialMembersProtectedAndNoexcept

// Offset in bytes of the Base subobject of d from the start of d.
template <typename Base, typename Derived>
std::ptrdiff_t base_offset(const Derived& d) {
    return reinterpret_cast<const char*>(static_cast<const Base*>(&d)) - reinterpret_cast<const char*>(&d);
}

// Clauses: "The count base comes first in the layout, then the hazard pointer
// base"; "The hazard pointer subobject may sit at any offset inside U ... so
// other bases may precede this one". Pins the order of the two subobjects for
// each test pointee, and that HpNodeAfter really places the hazard pointer base
// behind its 32-byte prefix (so the end-to-end tests on HpNodeAfter exercise a
// hazard pointer base at an offset other than the pointee's address).
TEST(IntrPointeeBaseHpLayout, CountBaseFirstHazardBaseAtAnyOffset) {
    const HpNode front{};
    EXPECT_LT((base_offset<intr_pointee_base<long>>(front)),
              (base_offset<std::hazard_pointer_obj_base<HpNode>>(front)));

    const HpNodeShort narrow{};
    EXPECT_LT((base_offset<intr_pointee_base<short>>(narrow)),
              (base_offset<std::hazard_pointer_obj_base<HpNodeShort>>(narrow)));

    const HpNodeAfter after{};
    EXPECT_LT((base_offset<intr_pointee_base<long>>(after)),
              (base_offset<std::hazard_pointer_obj_base<HpNodeAfter>>(after)));
    EXPECT_GE((base_offset<std::hazard_pointer_obj_base<HpNodeAfter>>(after)),
              static_cast<std::ptrdiff_t>(sizeof(Prefix)));
} // CountBaseFirstHazardBaseAtAnyOffset

// Clause: "The count semantics are intr_pointee_base's: a copy is a NEW object
// with count 0, assignment leaves *this's count alone ... Moves fall back to
// these copy operations." The objects are never owned by a pointer, so they
// are destroyed as plain locals.
TEST(IntrPointeeBaseHp, CopyIsAFreshObjectAssignmentKeepsCount) {
    HpNode original;
    add_refs(original, 2);
    const HpNode copy(original);
    EXPECT_EQ(copy.use_count(), 0);
    EXPECT_EQ(original.use_count(), 2);

    HpNode target;
    target.AddRef();
    target = original;
    EXPECT_EQ(target.use_count(), 1);
    EXPECT_EQ(original.use_count(), 2);

    HpNode moved(std::move(original));
    EXPECT_EQ(moved.use_count(), 0);
    EXPECT_EQ(original.use_count(), 2);
    target = std::move(moved);
    EXPECT_EQ(target.use_count(), 1);
} // CopyIsAFreshObjectAssignmentKeepsCount

// Clause: "The hazard pointer subobject follows the same rule, by hand: the
// copy constructor VALUE-initializes it ... instead of copying the source's: a
// fresh, never retired object ... a copy made of a retired object ... must not
// inherit them". Oracle: a copy -- of an owned object, and of a RETIRED, not
// yet reclaimed one -- is an independent pointee: adopted, released and
// drained, every object (copies and sources) is destroyed exactly once. The
// retired source is read after its last release on purpose: the clause names
// that case, and the object exists until a scan, which cannot run here (SetUp
// drained; four retirements stay far below the threshold). The "fresh hazard
// pointer subobject" part itself is NOT observable black-box: what the
// subobject holds is mm_hp-internal and retire() overwrites it, so a copy
// constructor that copied the source's subobject would pass this test and the
// whole file. What is pinned is the observable consequence: the copy is an
// independent pointee.
class HpCopyTest : public HpTrackedTest {};
TEST_F(HpCopyTest, CopyIsAnIndependentPointee) {
    using SP = intr_shared_ptr_hp<HpNode>::shared_ptr_type;
    {
        const SP owned(new HpNode);
        const SP copy_of_owned(new HpNode(*owned));
        EXPECT_EQ(copy_of_owned.use_count(), 1) << "the copy must start unowned (count 0) and be adopted 0 -> 1";
        EXPECT_EQ(owned.use_count(), 1);
    } // both released: two retirements
    HpNode* retired = nullptr;
    {
        const SP source(new HpNode);
        retired = source.get();
    } // the source's last release: retired, not yet reclaimed
    {
        const SP copy_of_retired(new HpNode(*retired));
        EXPECT_EQ(copy_of_retired.use_count(), 1);
    } // the copy retired as well
    EXPECT_TRUE(drain_then_check(1));
} // CopyIsAnIndependentPointee

// End-to-end use of the HP pointees with intr_shared_ptr_hp, typed over the
// pointee: base first (HpNode), base after another base (HpNodeAfter), narrow
// count (HpNodeShort), and the count base with a hand-added hazard pointer base
// (ManualHpNode).
template <typename Node>
class HpPointeeEndToEnd : public HpTrackedTest {
protected:
    using Atomic = intr_shared_ptr_hp<Node>;
    using SP = typename Atomic::shared_ptr_type;
};
using HpNodes = ::testing::Types<HpNode, HpNodeAfter, HpNodeShort, ManualHpNode>;
TYPED_TEST_SUITE(HpPointeeEndToEnd, HpNodes, TypeNames);

// Clauses (intr_shared_ptr_hp.h, store/load/CAS; the hooks drive the counts):
// shared_ptr_type(U*) takes a fresh object 0 -> 1; store() leaves the word
// owning one reference; load() returns the stored word with one more reference;
// a successful strong CAS publishes `desired` (counted), releases the word's
// reference to the old occupant and leaves `expected` untouched; a failed one
// leaves the word and `desired` unchanged and assigns `expected` the current
// value with a live reference; a weak CAS on a quiescent word with a matching
// `expected` succeeds. Every use_count() below is exact (single thread).
// Finally every object is destroyed exactly once after one drain.
TYPED_TEST(HpPointeeEndToEnd, StoreLoadCasCounts) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    {
        SP x(new TypeParam);
        ASSERT_EQ(x.use_count(), 1);
        Atomic a;
        a.store(x);
        EXPECT_EQ(x.use_count(), 2) << "store(): the word must own exactly one reference";
        SP loaded = a.load();
        EXPECT_TRUE(loaded == x);
        EXPECT_EQ(x.use_count(), 3);

        SP y(new TypeParam);
        SP expected = x;                                 // x: 4
        EXPECT_TRUE(a.compare_exchange_strong(expected, y));
        EXPECT_TRUE(expected == x) << "a successful CAS must leave expected untouched";
        EXPECT_EQ(x.use_count(), 3) << "the word's reference to the old occupant was not released";
        EXPECT_EQ(y.use_count(), 2);
        EXPECT_TRUE(a.load() == y);

        SP z(new TypeParam);
        EXPECT_FALSE(a.compare_exchange_strong(expected, z));   // the word holds y, expected x
        EXPECT_TRUE(expected == y) << "a failed CAS must refresh expected to the current value";
        EXPECT_EQ(y.use_count(), 3) << "the refreshed expected must own a live reference";
        EXPECT_EQ(x.use_count(), 2) << "expected's old reference was not released";
        EXPECT_EQ(z.use_count(), 1) << "a failed CAS changed desired's count";
        EXPECT_TRUE(a.load() == y);

        EXPECT_TRUE(a.compare_exchange_weak(expected, z)) << "a quiescent weak CAS with matching expected failed";
        EXPECT_TRUE(a.load() == z);
        EXPECT_EQ(y.use_count(), 2);
        EXPECT_EQ(z.use_count(), 2);
    } // every handle and the atomic released
    EXPECT_TRUE(this->drain_then_check(1));
} // StoreLoadCasCounts

// Clauses: the mark is part of the value (pointer, mark) and a marked word is
// an owning word (intr_shared_ptr_hp.h, "The value of the atomic"; the value
// type's identity comparison). A marked store owns a reference; load() returns
// the marked word; a CAS whose expected differs only in the mark fails and
// refreshes expected to the marked value; with that expected it succeeds. The
// last owner is a marked word, released by the atomic's destructor.
TYPED_TEST(HpPointeeEndToEnd, MarkedWordIsOwningAndDistinct) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    {
        Atomic a;                                        // declared first: destroyed last
        SP x(new TypeParam);
        a.store(x.set_mark());
        EXPECT_EQ(x.use_count(), 2) << "a marked word must own a reference";
        SP marked = a.load();
        EXPECT_TRUE(marked.is_marked());
        EXPECT_EQ(marked.get(), x.get());
        EXPECT_FALSE(marked == x);

        SP expected = x;
        EXPECT_FALSE(a.compare_exchange_strong(expected, SP()));
        EXPECT_TRUE(expected == marked) << "refreshed expected must carry the stored mark";
        EXPECT_TRUE(a.compare_exchange_strong(expected, x));
        EXPECT_TRUE(a.load() == x);
        a.store(x.set_mark());
    } // handles die first, then the atomic holding the marked word
    EXPECT_TRUE(this->drain_then_check(1));
} // MarkedWordIsOwningAndDistinct

// Clauses ("Deferred reclamation"): a pointee whose count reaches 0 is NOT
// deleted but retired, and destroyed at a later scan; a scan runs only inside a
// retire() that brings the pending count to the threshold (at least 1000) or at
// process exit. SetUp's drain emptied the retired list, so this one retirement
// cannot start a scan: the object must still exist after its last release. One
// drain destroys it (one layer: it owns nothing); a second drain must not
// destroy it again.
TYPED_TEST(HpPointeeEndToEnd, LastReleaseRetiresAndOneDrainDestroysOnce) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    std::size_t id = 0;
    {
        SP x(new TypeParam);
        id = x->id;
        const Atomic a(x);
        const SP loaded = a.load();
        EXPECT_EQ(loaded.use_count(), 3);
    } // the last release retires the object
    EXPECT_EQ(destructions_of(id), 0) << "destroyed at count 0: intr_shared_ptr_hp must retire, not delete";
    drain_reclamation();
    EXPECT_EQ(destructions_of(id), 1) << "one drain did not reclaim the retired pointee";
    drain_reclamation();
    EXPECT_EQ(destructions_of(id), 1) << "a later scan destroyed the pointee again";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // LastReleaseRetiresAndOneDrainDestroysOnce

// Clauses: the documented example "struct Node : intr_pointee_base_hp<Node> {
// intr_shared_ptr_hp<Node> next; }" compiles (the concept is checked in member
// bodies, so Node may be incomplete where the pointer is named) and works; a
// destructor chain dies "one layer per scan" (Deferred reclamation; hp_drain.h:
// one drain per layer). Three nodes in a chain need exactly three drain rounds.
class HpChainTest : public HpTrackedTest {};
TEST_F(HpChainTest, SelfReferentialChainReclaimedLayerByLayer) {
    using SP = intr_shared_ptr_hp<HpChainNode>::shared_ptr_type;
    {
        SP head(new HpChainNode);
        SP second(new HpChainNode);
        SP third(new HpChainNode);
        second->next.store(third);
        head->next.store(second);
        EXPECT_TRUE(head->next.load() == second);
        EXPECT_EQ(second.use_count(), 2);
    } // head 1 -> 0 (retired); second and third still owned by the chain
    EXPECT_TRUE(drain_then_check(3));
} // SelfReferentialChainReclaimedLayerByLayer

// ===========================================================================
// 5. intr_shared_ptr over the pointees
// ===========================================================================

// Typed over the pointee: hooks from intr_pointee_base<> and hand-written hooks.
template <typename Node>
class SpinLifetime : public TrackedTest {
protected:
    using Atomic = intr_shared_ptr<Node>;
    using SP = typename Atomic::shared_ptr_type;
};
using SpinNodes = ::testing::Types<BaseNode<>, HandNode>;
TYPED_TEST_SUITE(SpinLifetime, SpinNodes, TypeNames);

// The tests below pin intr_shared_ptr.h: "Reclamation is synchronous: the
// release that drops a pointee's count to 0 deletes it then and there, on the
// releasing thread. The releases are: a shared_ptr_type destructor or
// assignment (the assignment of `expected` on a CAS failure included),
// store(), and ~intr_shared_ptr. A successful CAS also releases the old
// occupant, but that release can never be the last one". One test per
// releasing member; each asserts "destroyed" immediately after the releasing
// call returns, and "not destroyed" just before it.

// Release by the handle destructor.
TYPED_TEST(SpinLifetime, HandleDestructorDestroysAtOnce) {
    using SP = typename TestFixture::SP;
    std::size_t id = 0;
    {
        const SP x(new TypeParam);
        id = x->id;
        EXPECT_EQ(destructions_of(id), 0);
    }
    EXPECT_EQ(destructions_of(id), 1) << "not destroyed by the last handle's destructor";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // HandleDestructorDestroysAtOnce

// Release by copy assignment.
TYPED_TEST(SpinLifetime, HandleCopyAssignmentDestroysAtOnce) {
    using SP = typename TestFixture::SP;
    SP x(new TypeParam);
    const std::size_t id = x->id;
    const SP null_handle;
    EXPECT_EQ(destructions_of(id), 0);
    x = null_handle;
    EXPECT_EQ(destructions_of(id), 1) << "not destroyed by the releasing copy assignment";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // HandleCopyAssignmentDestroysAtOnce

// Release by move assignment.
TYPED_TEST(SpinLifetime, HandleMoveAssignmentDestroysAtOnce) {
    using SP = typename TestFixture::SP;
    SP x(new TypeParam);
    const std::size_t id = x->id;
    EXPECT_EQ(destructions_of(id), 0);
    x = SP();
    EXPECT_EQ(destructions_of(id), 1) << "not destroyed by the releasing move assignment";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // HandleMoveAssignmentDestroysAtOnce

// Release by store().
TYPED_TEST(SpinLifetime, StoreDestroysAtOnce) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    Atomic a(SP(new TypeParam));                         // the word is the only owner
    const std::size_t id = a.load()->id;
    EXPECT_EQ(destructions_of(id), 0);
    a.store(nullptr);
    EXPECT_EQ(destructions_of(id), 1) << "not destroyed by the releasing store()";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // StoreDestroysAtOnce

// CAS success (intr_shared_ptr.h, compare_exchange_strong): "On success
// `expected` is UNTOUCHED (it keeps its reference to the old occupant, whose
// release here therefore never destroys it); `desired` is unchanged either
// way". Oracle: exact counts -- the word's reference to the old occupant is
// released (old: word + expected = 2 -> 1), expected still holds the old word,
// the published pointee gains the word's reference (desired: 1 -> 2) -- and
// the old occupant is NOT destroyed by the CAS but synchronously when its last
// handle, `expected`, is dropped.
TYPED_TEST(SpinLifetime, CasSuccessLeavesExpectedAndReleasesTheWord) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    std::size_t desired_id = 0;
    {
        Atomic a(SP(new TypeParam));
        SP expected = a.load();
        TypeParam* const old_word = expected.get_raw();
        const std::size_t old_id = expected->id;
        SP desired(new TypeParam);
        desired_id = desired->id;
        TypeParam* const desired_word = desired.get_raw();
        EXPECT_EQ(expected.use_count(), 2);
        EXPECT_TRUE(a.compare_exchange_strong(expected, desired));
        EXPECT_EQ(expected.get_raw(), old_word) << "a successful CAS must leave expected untouched";
        EXPECT_EQ(expected.use_count(), 1) << "the word's reference to the old occupant was not released";
        EXPECT_EQ(desired.get_raw(), desired_word) << "desired must be unchanged";
        EXPECT_EQ(desired.use_count(), 2);
        EXPECT_TRUE(a.load() == desired);
        EXPECT_EQ(destructions_of(old_id), 0) << "a successful CAS destroyed the old occupant";
        expected = SP();
        EXPECT_EQ(destructions_of(old_id), 1) << "the old occupant's last release did not destroy it at once";
        desired = SP();
        EXPECT_EQ(destructions_of(desired_id), 0) << "the word does not own the published pointee";
    } // ~intr_shared_ptr releases the published pointee
    EXPECT_EQ(destructions_of(desired_id), 1);
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // CasSuccessLeavesExpectedAndReleasesTheWord

// CAS failure: "`expected` is updated to the current value (with a fresh
// reference) ... -- the one place a CAS can run the last release of an object:
// expected's previous pointee". `expected` holds the only reference to E (not
// in the word); the failing CAS must destroy E before it returns, refresh
// expected to the word's value with a fresh reference, and leave desired as it
// was.
TYPED_TEST(SpinLifetime, CasFailureReleasesExpectedsPreviousPointeeAtOnce) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    {
        const SP occupant(new TypeParam);
        Atomic a(occupant);
        SP expected(new TypeParam);
        const std::size_t previous_id = expected->id;
        const SP desired(new TypeParam);
        EXPECT_EQ(occupant.use_count(), 2);
        EXPECT_FALSE(a.compare_exchange_strong(expected, desired));
        EXPECT_EQ(destructions_of(previous_id), 1) << "expected's previous pointee was not destroyed by the failing CAS";
        EXPECT_TRUE(expected == occupant) << "expected must be refreshed to the current value";
        EXPECT_EQ(occupant.use_count(), 3) << "the refreshed expected must own a fresh reference";
        EXPECT_EQ(desired.use_count(), 1) << "a failed CAS changed desired's count";
    } // every handle and the atomic released
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // CasFailureReleasesExpectedsPreviousPointeeAtOnce

// Release by ~intr_shared_ptr.
TYPED_TEST(SpinLifetime, AtomicDestructorDestroysAtOnce) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    std::size_t id = 0;
    {
        const Atomic a(SP(new TypeParam));
        id = a.load()->id;
        EXPECT_EQ(destructions_of(id), 0);
    }
    EXPECT_EQ(destructions_of(id), 1) << "not destroyed by the releasing ~intr_shared_ptr";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // AtomicDestructorDestroysAtOnce

// The converse: while any owner remains (a word, a handle), the object lives;
// the count reaches 0 only at the last release.
TYPED_TEST(SpinLifetime, AliveWhileAnyOwnerRemains) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    Atomic a;
    SP x(new TypeParam);
    const std::size_t id = x->id;
    a.store(x);
    x = SP();
    EXPECT_EQ(destructions_of(id), 0) << "destroyed while the word still owned it";
    SP loaded = a.load();
    a.store(nullptr);
    EXPECT_EQ(destructions_of(id), 0) << "destroyed while a loaded handle still owned it";
    EXPECT_EQ(loaded.use_count(), 1);
    loaded = SP();
    EXPECT_EQ(destructions_of(id), 1);
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // AliveWhileAnyOwnerRemains

// "on the releasing thread": the last release happens on a second thread (a
// store() there); the pointee's destructor must have run on that thread, and
// before that store() returned (the thread itself sees the destruction).
TYPED_TEST(SpinLifetime, DestroyedOnTheReleasingThread) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    std::thread::id destroyed_on;
    Atomic a(SP(new TypeParam(&destroyed_on)));
    const std::size_t id = a.load()->id;
    std::thread::id releaser;
    int destructions_seen_by_releaser = -1;
    std::thread thread([&] {
        releaser = std::this_thread::get_id();
        a.store(nullptr);
        destructions_seen_by_releaser = destructions_of(id);
    });
    thread.join();
    EXPECT_NE(releaser, std::this_thread::get_id());
    EXPECT_EQ(destructions_seen_by_releaser, 1) << "not destroyed by the time the releasing store() returned";
    EXPECT_EQ(destroyed_on, releaser) << "destroyed on a thread other than the releasing one";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // DestroyedOnTheReleasingThread

// Mark semantics documented for intr_shared_ptr (bit 0 of the word, "part of
// the value/identity"; the CAS compares "the *full* value including the mark
// bit"; on failure "expected is updated to the current value (with a fresh
// reference)"): a marked word owns a reference, loads as marked, differs from
// the unmarked handle; a CAS with an unmarked expected fails and refreshes
// expected to the marked value; with that expected it succeeds. The last owner
// is a marked word, released by ~intr_shared_ptr: synchronous destruction.
TYPED_TEST(SpinLifetime, MarkedWordIsOwningAndDistinct) {
    using Atomic = typename TestFixture::Atomic;
    using SP = typename TestFixture::SP;
    std::size_t id = 0;
    {
        Atomic a;                                        // declared first: destroyed last
        SP x(new TypeParam);
        id = x->id;
        a.store(x.set_mark());
        EXPECT_EQ(x.use_count(), 2) << "a marked word must own a reference";
        SP marked = a.load();
        EXPECT_TRUE(marked.is_marked());
        EXPECT_EQ(marked.get(), x.get());
        EXPECT_FALSE(marked == x);
        EXPECT_TRUE(marked == x.set_mark());

        SP expected = x;
        EXPECT_EQ(x.use_count(), 4);
        EXPECT_FALSE(a.compare_exchange_strong(expected, SP()));
        EXPECT_TRUE(expected == marked) << "refreshed expected must carry the stored mark";
        EXPECT_EQ(x.use_count(), 4) << "the refresh must trade expected's reference for a fresh one";
        EXPECT_TRUE(a.compare_exchange_strong(expected, x));
        EXPECT_TRUE(a.load() == x);
        a.store(x.set_mark());
    } // handles die first, then the atomic holding the marked word
    EXPECT_EQ(destructions_of(id), 1) << "the marked word's reference was not released by ~intr_shared_ptr";
    EXPECT_TRUE(this->created_objects_destroyed_once());
} // MarkedWordIsOwningAndDistinct

// Clauses (intr_shared_ptr.h, template parameters): T is the interface seen
// through get() and operator->, U the stored type carrying the hooks, U*
// converting to T*; "U is the DYNAMIC type of every pointee: the last release
// runs `delete p` on a U*". Impl puts Iface at a nonzero offset with a
// non-virtual destructor: deleting through Iface* would free the wrong address
// (ASan) and skip ~Impl (no destruction recorded).
class SpinTUTest : public TrackedTest {};
TEST_F(SpinTUTest, InterfaceTypeDistinctFromStoredType) {
    using Atomic = intr_shared_ptr<Iface, Impl>;
    using SP = Atomic::shared_ptr_type;
    static_assert(std::is_same_v<decltype(std::declval<const SP&>().get()), Iface*>);
    static_assert(std::is_same_v<decltype(std::declval<const SP&>().get_raw()), Impl*>);
    std::size_t id = 0;
    {
        Impl* const raw = new Impl;
        id = raw->id;
        const Atomic a(SP{raw});
        const SP loaded = a.load();
        EXPECT_EQ(loaded->iface_value, 7);
        EXPECT_EQ(loaded.get(), static_cast<Iface*>(raw));
        EXPECT_EQ(loaded.get_raw(), raw);
    } // the handle and the atomic released: the last release deletes
    EXPECT_EQ(destructions_of(id), 1) << "the last release did not delete through U*";
    EXPECT_TRUE(created_objects_destroyed_once());
} // InterfaceTypeDistinctFromStoredType

// Clauses: a node may hold intr_shared_ptr<Self> while Self is incomplete (the
// concept is checked in member bodies); reclamation is synchronous, so the last
// release of the head destroys the whole chain within that call.
class SpinChainTest : public TrackedTest {};
TEST_F(SpinChainTest, SelfReferentialChainCascadesSynchronously) {
    using SP = intr_shared_ptr<ChainNode>::shared_ptr_type;
    SP head(new ChainNode);
    {
        SP second(new ChainNode);
        const SP third(new ChainNode);
        second->next.store(third);
        head->next.store(std::move(second));
    }
    EXPECT_EQ(head.use_count(), 1);
    head = SP();
    EXPECT_TRUE(created_objects_destroyed_once()) << "the chain was not destroyed within the head's release";
} // SelfReferentialChainCascadesSynchronously

// ===========================================================================
// 6. The value type's and the atomics' noexcept/constexpr promises
// ===========================================================================

// Typed over the pointer: both pointers, each over a base-derived pointee and
// over a second pointee (hand-written hooks; the hazard base after a prefix).
template <typename Atomic>
class ValueTypeContract : public ::testing::Test {};
using PointerTypes = ::testing::Types<intr_shared_ptr<BaseNode<>>, intr_shared_ptr<HandNode>,
                                      intr_shared_ptr_hp<HpNode>, intr_shared_ptr_hp<HpNodeAfter>>;
TYPED_TEST_SUITE(ValueTypeContract, PointerTypes, TypeNames);

// Clauses (intr_shared_ptr_common.h): "constexpr shared_ptr_type() noexcept",
// "constexpr shared_ptr_type(std::nullptr_t) noexcept"; "Null. Postcondition:
// !*this, !is_marked(), use_count() == 0". constinit proves both constructors
// usable in constant initialization (constexpr).
TYPED_TEST(ValueTypeContract, NullConstructorsAreConstexprAndNoexcept) {
    using SP = typename TypeParam::shared_ptr_type;
    static_assert(std::is_nothrow_default_constructible_v<SP>);
    static_assert(std::is_nothrow_constructible_v<SP, std::nullptr_t>);
    static_assert(std::is_convertible_v<std::nullptr_t, SP>);

    static constinit SP default_null;
    static constinit SP nullptr_null{nullptr};
    for (const SP* p : {&default_null, &nullptr_null}) {
        EXPECT_FALSE(static_cast<bool>(*p));
        EXPECT_FALSE(p->is_marked());
        EXPECT_EQ(p->use_count(), 0);
        EXPECT_EQ(p->get(), nullptr);
    }
} // NullConstructorsAreConstexprAndNoexcept

// Clause (intr_shared_ptr_common.h, "Exceptions"): "nothing here can throw,
// and every member is declared noexcept ... operator* and operator-> are
// noexcept as well". Every public member of shared_ptr_type, one by one.
//
// What makes this possible is a compile-time rejection that this file cannot
// exercise: a U whose destructor is not nothrow is rejected by a static_assert
// in require_pointee() (intr_shared_ptr_common.h, "The destructor assert";
// intr_shared_ptr.h, "Exceptions"; intr_shared_ptr_hp.h, "Exception safety").
// A static_assert in a member body is a hard error, not a substitution
// failure, so it cannot be probed in-suite; it needs a compile-fail test. What
// can be checked here: the concept itself does not look at the destructor
// (IntrusivePointeeConcept.Partitions accepts ThrowingDestructor), and the
// pointer classes still instantiate with such a U (PointeeCheckPlacement).
TYPED_TEST(ValueTypeContract, EveryMemberIsNoexcept) {
    using SP = typename TypeParam::shared_ptr_type;
    using U = std::remove_pointer_t<decltype(std::declval<const SP&>().get_raw())>;
    static_assert(std::is_nothrow_constructible_v<SP, U*>);              // adopting constructor
    static_assert(std::is_nothrow_copy_constructible_v<SP>);
    static_assert(std::is_nothrow_move_constructible_v<SP>);
    static_assert(std::is_nothrow_copy_assignable_v<SP>);
    static_assert(std::is_nothrow_move_assignable_v<SP>);
    static_assert(std::is_nothrow_destructible_v<SP>);
    static_assert(noexcept(*std::declval<const SP&>()));
    static_assert(noexcept(std::declval<const SP&>().operator->()));
    static_assert(noexcept(std::declval<const SP&>().get()));
    static_assert(noexcept(static_cast<bool>(std::declval<const SP&>())));
    static_assert(noexcept(std::declval<const SP&>() == std::declval<const SP&>()));
    static_assert(noexcept(std::declval<const SP&>() != std::declval<const SP&>()));
    static_assert(noexcept(std::declval<const SP&>().use_count()));
    static_assert(noexcept(std::declval<const SP&>().is_marked()));
    static_assert(noexcept(std::declval<const SP&>().get_unmarked()));   // const & overload
    static_assert(noexcept(std::declval<SP&&>().get_unmarked()));        // && overload
    static_assert(noexcept(std::declval<const SP&>().set_mark()));
    static_assert(noexcept(std::declval<const SP&>().get_raw()));
} // EveryMemberIsNoexcept

// Clause (both pointer headers): "Declared noexcept: the two trivial
// constructors of the atomic"; the default one is declared constexpr
// (constinit), and the nullptr one is implicit (explicit(false)).
TYPED_TEST(ValueTypeContract, AtomicNullConstructorsAreNoexcept) {
    static_assert(std::is_nothrow_default_constructible_v<TypeParam>);
    static_assert(std::is_nothrow_constructible_v<TypeParam, std::nullptr_t>);
    static_assert(std::is_convertible_v<std::nullptr_t, TypeParam>);
    static constinit TypeParam constant_initialized;
    EXPECT_FALSE(static_cast<bool>(constant_initialized.load()));
}

// ===========================================================================
// 7. Aliasing guarantee of the assignments
// ===========================================================================
//
// Clause (intr_shared_ptr_common.h, "Aliasing guarantee of the assignments"):
// "the source may be a handle stored INSIDE the pointee that the assignment
// releases -- `h = h->child` and `h = std::move(h->child)` are well-defined
// when h is the parent's only owner ... Releasing the old pointee may destroy
// the source handle (synchronously for intr_shared_ptr; for intr_shared_ptr_hp
// when the retire runs a scan in place) ... After `h = h->child` on the
// spinlock pointer h is the child's only owner; on the hazard pointer one the
// retired parent's member still owns the child until the parent is reclaimed
// (use_count() 2, then 1)." After the move, the member was nulled ("the move
// assignment nulls the source"), so h is the child's only owner on both
// pointers. Oracles: no sanitizer report (an assignment that reads its source
// after the release reads a destroyed handle: ASan reports a
// heap-use-after-free), h holds the child afterwards, the use_count() just
// quoted, and exactly-once destruction of every object.

// The assignment under test: copy (`h = h->child`) or move
// (`h = std::move(h->child)`).
template <typename SP>
void assign_from_child(SP& h, bool move) {
    if (move) {
        h = std::move(h->child);
    } else {
        h = h->child;
    }
} // assign_from_child()

// Spinlock pointer: the release of the parent deletes it synchronously inside
// the assignment, every time -- one round is deterministic.
class SpinAliasingTest : public TrackedTest {
protected:
    void run(bool move) {
        using SP = intr_shared_ptr<SpinParent>::shared_ptr_type;
        SP h(new SpinParent);
        const std::size_t parent_id = h->id;
        std::size_t child_id = 0;
        {
            const SP child(new SpinParent);
            child_id = child->id;
            h->child = child;
        } // the child is owned only by the parent's member
        ASSERT_EQ(h.use_count(), 1) << "precondition of the clause: h is the parent's only owner";
        assign_from_child(h, move);
        EXPECT_EQ(destructions_of(parent_id), 1) << "the parent's last release must delete it inside the assignment";
        ASSERT_TRUE(static_cast<bool>(h));
        EXPECT_EQ(h->id, child_id) << "h does not hold the child";
        EXPECT_EQ(h.use_count(), 1) << "h must be the child's only owner";
        EXPECT_EQ(destructions_of(child_id), 0);
        h = SP();
        EXPECT_TRUE(created_objects_destroyed_once());
    } // SpinAliasingTest::run()
}; // class SpinAliasingTest

TEST_F(SpinAliasingTest, CopyAssignFromMemberOfTheReleasedPointee) {
    run(false);
}

TEST_F(SpinAliasingTest, MoveAssignFromMemberOfTheReleasedPointee) {
    run(true);
}

// Hazard pointer pointer: the source dies inside the assignment only "when the
// retire runs a scan in place", i.e. when the parent's retire brings mm_hp's
// pending count to its threshold (intr_shared_ptr_hp.h, "Deferred
// reclamation"). The threshold is not observable, so the test repeats the
// assignment on fresh parent/child pairs, each round retiring exactly ONE
// object (the parent, inside the assignment: every resulting h is kept until
// the end, so no other release reaches 0), until the parent was observed
// destroyed right after its own assignment -- the in-place scan -- at least
// twice, or max_rounds rounds. Asserting that the in-place case occurred makes
// the test non-vacuous. Per round: h holds the child; after the move its
// use_count() is 1; after the copy it is 2 while the parent is pending and 1
// if the parent was reclaimed in place (the clause's "2, then 1").
// Final drain depth: copy 2 (a pending parent's destructor releases its child's
// last reference, which retires the child during that scan: next round); move
// 1 (the members were nulled).
class HpAliasingTest : public HpTrackedTest {
protected:
    void run(bool move) {
        using SP = intr_shared_ptr_hp<HpParent>::shared_ptr_type;
        constexpr int max_rounds = 20'000;     // far above mm_hp's threshold (1000 with few hazard records)
        constexpr int wanted_in_place = 2;
        std::vector<SP> kept;
        kept.reserve(max_rounds);
        int in_place = 0;
        int wrong_holder = 0;
        int wrong_count = 0;
        int child_destroyed = 0;
        for (int round = 0; round != max_rounds && in_place != wanted_in_place; ++round) {
            SP h(new HpParent);
            const std::size_t parent_id = h->id;
            std::size_t child_id = 0;
            {
                const SP child(new HpParent);
                child_id = child->id;
                h->child = child;
            } // the child is owned only by the parent's member
            assign_from_child(h, move);
            const bool reclaimed_in_place = destructions_of(parent_id) == 1;
            if (reclaimed_in_place) ++in_place;
            const long expected_count = (move || reclaimed_in_place) ? 1 : 2;
            if (!h || h->id != child_id) {
                ++wrong_holder;
            } else if (h.use_count() != expected_count) {
                ++wrong_count;
            }
            if (destructions_of(child_id) != 0) ++child_destroyed;
            kept.push_back(std::move(h));
        } // loop over rounds until the in-place scan was hit
        EXPECT_EQ(in_place, wanted_in_place) << "no assignment ran a scan in place within " << max_rounds
                                             << " rounds: the source-destroying case was not reached";
        EXPECT_EQ(wrong_holder, 0) << "rounds where h did not hold the child";
        EXPECT_EQ(wrong_count, 0) << "rounds where the child's use_count() was not the documented one";
        EXPECT_EQ(child_destroyed, 0) << "rounds where the child was destroyed";
        kept.clear();
        EXPECT_TRUE(drain_then_check(move ? 1 : 2));
    } // HpAliasingTest::run()
}; // class HpAliasingTest

TEST_F(HpAliasingTest, CopyAssignFromMemberOfTheReleasedPointee) {
    run(false);
}

TEST_F(HpAliasingTest, MoveAssignFromMemberOfTheReleasedPointee) {
    run(true);
}

// ===========================================================================
// 8. Stress
// ===========================================================================

// Sizes of the stress runs: 4 threads on 4 shared words, so that operations
// on one word collide often; 100000 operations per thread keep a TSan run of
// each test well under a second at -O0. Seeds are fixed (thread index + 1), so
// a failure replays the same operation sequence per thread.
inline constexpr int k_stress_threads = 4;
inline constexpr int k_stress_words = 4;
inline constexpr int k_stress_iterations = 100'000;

// Violations observed by the stress threads, asserted zero after they join.
struct StressViolations {
    // A loaded pointee's `value` was not k_magic: a destroyed object was read.
    std::atomic<long> corrupt_pointee{0};
    // A non-null handle reported use_count() 0 ("never 0 for a non-null
    // handle (this handle is one owner)").
    std::atomic<long> zero_use_count{0};
    // A strong CAS returned false with `expected` unchanged (a strong CAS fails
    // only on a genuinely different current value, which it hands back).
    std::atomic<long> unchanged_cas_failure{0};
}; // struct StressViolations

// k_stress_threads threads apply random operations to k_stress_words shared
// words of type Atomic: load (keeping some results in two per-thread handle
// slots, so that releases happen on every thread and at varied times), store of
// a new pointee, of another word's value, of a marked new pointee, of null, a
// strong CAS with a freshly loaded expected and a new or shared desired, and
// dropping a held handle. Returns after joining the threads and destroying the
// words, so that every handle and every word is gone.
template <typename Atomic, typename Node>
void run_stress(StressViolations& violations) {
    using SP = typename Atomic::shared_ptr_type;
    std::array<Atomic, k_stress_words> words;
    // Checks a non-null value read from a word.
    const auto check = [&violations](const SP& p) {
        if (!p) return;
        if (p->value != k_magic) violations.corrupt_pointee.fetch_add(1, std::memory_order_relaxed);
        if (p.use_count() < 1) violations.zero_use_count.fetch_add(1, std::memory_order_relaxed);
    };
    run_on_threads(k_stress_threads, [&](int t) {
        std::minstd_rand rng(static_cast<std::minstd_rand::result_type>(t) + 1);
        std::array<SP, 2> held;
        for (int i = 0; i != k_stress_iterations; ++i) {
            Atomic& word = words[rng() % k_stress_words];
            Atomic& other = words[rng() % k_stress_words];
            switch (rng() % 8) {
            case 0:
            case 1: {
                SP loaded = word.load();
                check(loaded);
                held[rng() % 2] = std::move(loaded);
                break;
            }
            case 2:
                word.store(SP(new Node));
                break;
            case 3:
                word.store(other.load());
                break;
            case 4:
                word.store(SP(new Node).set_mark());
                break;
            case 5:
                word.store(nullptr);
                break;
            case 6: {
                SP expected = word.load();
                const SP original = expected;
                const SP desired = (rng() % 2 == 0) ? SP(new Node) : other.load();
                if (!word.compare_exchange_strong(expected, desired)) {
                    if (expected == original) violations.unchanged_cas_failure.fetch_add(1, std::memory_order_relaxed);
                    check(expected);
                }
                break;
            } // case 6: strong CAS
            default:
                held[rng() % 2] = SP();
                break;
            } // switch on the operation
        } // loop over operations
    }); // thread body
} // run_stress()

// Expects every violation counter to be zero.
void expect_no_violations(const StressViolations& violations) {
    EXPECT_EQ(violations.corrupt_pointee.load(), 0) << "a destroyed pointee was read through a loaded handle";
    EXPECT_EQ(violations.zero_use_count.load(), 0) << "a non-null handle reported use_count() 0";
    EXPECT_EQ(violations.unchanged_cas_failure.load(), 0) << "a strong CAS failed with expected unchanged";
}

// Clauses: synchronous reclamation (intr_shared_ptr.h) and the hook contract
// under contention, over intr_pointee_base<>. Oracle: once every handle and
// every word is gone, every pointee created by the run was destroyed exactly
// once -- no drain, since reclamation is synchronous -- and no thread read a
// destroyed pointee. TSan checks the count orders' visible effects (each
// destructor runs after the plain accesses of every former owner).
class SpinStressTest : public TrackedTest {};
TEST_F(SpinStressTest, EveryPointeeDestroyedExactlyOnce) {
    StressViolations violations;
    run_stress<intr_shared_ptr<BaseNode<>>, BaseNode<>>(violations);
    expect_no_violations(violations);
    EXPECT_TRUE(created_objects_destroyed_once());
}

// Clauses: deferred reclamation (intr_shared_ptr_hp.h) and the hook contract
// under contention, over intr_pointee_base_hp placed AFTER another base (the
// hazard pointer subobject at a nonzero offset, the case where a hazard formed
// from the raw word would protect nothing). Oracle: after the threads are
// joined (quiescence) and every word and handle is gone, one drain destroys
// every pointee created by the run exactly once (they own nothing: one layer).
// ASan reports a premature reclamation as a use after free; TSan cannot see
// hazard pointer lifetimes (intr_shared_ptr_hp.h), it checks the rest.
class HpStressTest : public HpTrackedTest {};
TEST_F(HpStressTest, EveryPointeeDestroyedExactlyOnceAfterADrain) {
    StressViolations violations;
    run_stress<intr_shared_ptr_hp<HpNodeAfter>, HpNodeAfter>(violations);
    expect_no_violations(violations);
    EXPECT_TRUE(drain_then_check(1));
}

} // namespace
