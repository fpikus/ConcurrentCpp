// White-box seam tests of intr_shared_ptr_hp (intr_shared_ptr_hp/intr_shared_ptr_hp.h).
// Built twice, with ASan and with TSan (both at -O0), linked with mm_hp/mm_hp.cpp.
//
// What a "seam" is here. Every interleaving that can break the pointer's
// hazard-pointer protocol needs a second thread to act inside a short window of
// one operation: between load()'s validation of the word and its increment of
// the count, between the increment and the re-validation, between a store's
// increment of the new pointee and its publishing exchange, and so on. Stress
// tests reach those windows by luck, and the sanitizers judge hazard-pointer
// lifetimes badly anyway (a premature retire is freed a whole reclamation batch
// later, so ASan rarely sees the access; every load() ends with a release store
// to its hazard record, which hides nearly every premature-free race from TSan).
// But the pointer calls user code inside every one of those windows: the
// pointee's own reference-count hooks AddRef() and TryAddRef(). The pointees
// below are test-owned, so their hooks can run a COMPLETE nested operation on
// the same atomic (a store, a release, a drain of mm_hp's retired list) at
// exactly the point a second thread could have run it. That interleaving is
// legal for a real second thread, so the correct header must survive it; and it
// happens on every run, on one thread, so a defect in the window fails every
// run, not one in a thousand. The nesting stands in for a second thread.
//
// Rules the hooks follow (each one is there because the obvious alternative
// lets a defect through):
// - A hook is KEYED to the object it must fire on. "Fire at the first hook call
//   after arming" fires in the wrong place: a compare-exchange increments its
//   `desired` before the refresh load increments the current value.
// - A hook fires at the ENTRY of AddRef()/TryAddRef(), before the count is read.
//   A hook that ran after the increment could not see a hazard cleared before
//   the increment (the object would already be pinned by its count).
// - A second, separate hook fires right AFTER a successful TryAddRef()
//   increment; that is the only way to make the word change between load()'s
//   increment and its re-validation, i.e. to reach the path where load() itself
//   must release the reference it just took (and retire the object if that
//   reference was the last one).
// - A third hook fires at the ENTRY of DelRef(). Only StorePublishesPointee
//   uses it, to hold the main thread inside store() (see that test).
// - Hooks are one-shot: firing disarms them (target pointer included, so that a
//   leaked object is not kept reachable by a stale static pointer, which would
//   hide it from LeakSanitizer). Every test asserts that its hook ran, so a
//   test cannot pass vacuously because the window it targets was never reached.
// - A store under test is passed `std::move(handle)`: passing a copy would run
//   AddRef() in the copy constructor and fire the hook there, outside the store.
//
// Oracles. "Destroyed" is observed through a per-object flag that the object's
// destructor sets (the flag is shared-owned, so it outlives the object however
// late mm_hp reclaims it). mm_hp destroys a retired object only at a later scan,
// so every "was reclaimed" assertion follows a drain (hp_drain.h), and every
// "was NOT reclaimed" assertion is paired with a positive control that the SAME
// drain did reclaim; otherwise "not reclaimed" could mean "the drain did
// nothing". The fixture also checks, for every test, that the number of live
// pointees returns to its starting value after a final drain (no leak, no
// double destruction) and that no reference count went 0 -> 1 on an object that
// had already been adopted once (no resurrection of a retired object). Defects
// that free an object while it is still used are also reported by the
// sanitizers (a use after free), usually first.
//
// Every test runs for two pointer types:
// - HpBaseAt0:  intr_shared_ptr_hp<Data>, T == U, the hazard-pointer base first,
//   at offset 0 of the pointee;
// - HpBaseAt32: intr_shared_ptr_hp<Iface, Impl>, T != U (a polymorphic interface
//   T and a concrete U), the hazard-pointer base SECOND, at offset 32. A hazard
//   formed from the raw word instead of through the typed U* points 32 bytes
//   before the hazard-pointer base that mm_hp compares against, so it protects
//   nothing; at offset 0 the two addresses coincide and that defect is
//   invisible. This is the only T != U coverage of the pointer.
//
// Defects pinned (deterministically unless noted; each test's comment says
// which, and on what oracle):
// - Window, WindowMarked: hazard cleared before the increment; `delete` instead
//   of retire on the last release; a plain increment instead of TryAddRef
//   (revives a retired object, which is then retired twice); hazard published on
//   the marked word (WindowMarked); hazard address formed from the word (at
//   offset 32 only).
// - Reval: no re-validation of the word after the increment.
// - RevalRetire: on a failed re-validation, the reference just taken is not
//   released, or released without retiring the object when that was the last.
// - StoreCounted, CasCounted: the published word's reference counted AFTER the
//   publishing exchange/compare-exchange instead of before.
// - ChangedBack{Strong,Weak,MarkedStrong,MarkedWeak}: a strong compare-exchange
//   that does not retry when the word changed and changed back; one that
//   compares the refreshed value with the mark masked off; no re-validation
//   (again); and the weak compare-exchange's single-attempt semantics.
// - Alias{Strong,Weak}: on failure, the pre-count of `desired` undone through
//   `desired` after `expected` (the same object) was already refreshed.
// - ReleaseRetiresAndOneDrainReclaims: the positive control of the whole file;
//   also `delete` instead of retire.
// - StorePublishesPointee, CasPublishesPointee, StoreAcquiresReplacedPointee,
//   AcquireLoadOfNullSynchronizes (two threads; detected by TSan only, as a data
//   race report, on every run; in the ASan build they cannot fail on x86): the
//   publishing exchange or compare-exchange not at least `release`; the
//   publishing RMW not also `acquire` (it reads the old occupant, which the
//   caller then releases); load()'s first read of the word not performed with
//   the caller's `order` on the path that returns null without a hazard.
// - AcquireLoadSeesMarkOnlyPublication, AcquireLoadOfNullAfterZeroCountSynchronizes
//   (two threads, TSan only, every run; a hook stops the reader inside load()):
//   the post-increment re-validation reload, or the reload after TryAddRef saw
//   0, not acquire -- each is the only read of the word that observed the
//   publishing store.
//
// Not reachable here: windows with no pointee call in them (between the first
// read of the word and the publication of the hazard: publishing without
// validation), memory orders of the count operations themselves, and the
// production LockFreeList::Node, which has no test hooks.
//
// Single-threaded except the six publication tests, which join their thread
// before the fixture drains: the drain's quiescence precondition holds at every
// drain, and every scan runs synchronously inside one of this thread's retire()
// calls. Hooks never run inside a scan (no pointee destructor calls a hook), so
// the drains inside hooks are legal.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "intr_shared_ptr_hp.h"
#include "hp_drain_gtest.h"   // also registers the early mm_hp scan (see that header)

namespace {

// A one-shot hook keyed to one object. arm() sets the target and the action;
// fire_if(obj) runs the action exactly once, the first time it is called with
// obj == target, after disarming itself (so the action may call back into the
// hooked functions, on the same object too, without re-firing). Armed and
// disarmed on the test's main thread while no other thread exists (SetUp,
// TearDown, or before a test creates its thread). Normally fired on the main
// thread too; the two re-validation/zero-count publication tests arm it so that
// it fires on their reader thread, and read `ran` only after joining it. Any
// thread may call fire_if() with a non-target object at any time (the
// publication tests' threads do, while the other thread is inside a fired
// hook): that call reads only `target`.
struct KeyedHook {
    // The object the hook fires on (a HookedCount*, compared as an address);
    // nullptr when disarmed. Reset when the hook fires, so a leaked target is not
    // kept reachable by this static. Atomic, accessed relaxed, only because a
    // second thread may read it while the main thread disarms the hook: relaxed
    // accesses carry no happens-before edge, so the hook machinery cannot mask a
    // missing edge that TSan is meant to find in the publication tests.
    std::atomic<const void*> target{nullptr};

    // The nested operation to run; empty when disarmed.
    std::function<void()> action;

    // Set when the hook fires; reset by arm(). Tests assert it to prove that the
    // window they target was reached.
    bool ran = false;

    // Arms the hook: `action` runs at the next fire_if(t).
    void arm(const void* t, std::function<void()> a) {
        action = std::move(a);
        ran = false;
        target.store(t, std::memory_order_relaxed);
    }

    // Disarms the hook without running it (fixture SetUp/TearDown: a hook that a
    // defective header never reached must not fire in a later test).
    void disarm() {
        target.store(nullptr, std::memory_order_relaxed);
        action = nullptr;
    }

    // Runs the action if `obj` is the target. The action is moved out and the
    // hook disarmed BEFORE it runs.
    void fire_if(const void* obj) {
        const void* const t = target.load(std::memory_order_relaxed);
        if (t == nullptr || t != obj) return;
        std::function<void()> a = std::move(action);
        disarm();
        ran = true;
        a();
    } // KeyedHook::fire_if()
}; // struct KeyedHook

// The intrusive count and the test instrumentation shared by both pointee types
// (a base class, so that the hazard-pointer base can be placed before it at
// whatever offset the pointee type needs). Provides the four hooks
// intr_shared_ptr_hp requires of its U (AddRef, TryAddRef, DelRef, use_count),
// with the memory orders the pointer requires of them: every operation on the
// count is an RMW; the load that observes 0 and the successful CAS of TryAddRef
// are acquire; DelRef is acq_rel.
struct HookedCount {
    // Number of HookedCount objects alive (both pointee types). The fixture
    // compares it with its value at SetUp after a final drain.
    inline static std::atomic<long> live{0};

    // Number of 0 -> 1 increments through AddRef() on an object that had already
    // been adopted once, i.e. resurrections of an object whose count reached 0
    // (retired, or about to be). Must stay 0; the fixture checks it.
    inline static std::atomic<long> resurrections{0};

    // Fires at the ENTRY of AddRef() and TryAddRef() on its target, before the
    // count is read.
    inline static KeyedHook entry_hook;

    // Fires right AFTER a successful TryAddRef() increment on its target (before
    // TryAddRef returns to load(), i.e. before load()'s re-validation).
    inline static KeyedHook after_increment_hook;

    // Fires at the ENTRY of DelRef() on its target, before the count is changed.
    // Used to stop store() between its publishing exchange and its return (the
    // release of the replaced value is the one call in between).
    inline static KeyedHook release_hook;

    // The intrusive strong count; 0 until the first owner adopts the object.
    std::atomic<long> ref_count{0};

    // Set by the first 0 -> 1 (the adoption); a later 0 -> 1 is a resurrection.
    std::atomic<bool> adopted{false};

    // Set by the destructor. Shared with the test (gone_flag()), so the test can
    // read it after mm_hp has reclaimed the object, at whatever time it does.
    const std::shared_ptr<std::atomic<bool>> gone;

    HookedCount() : gone(std::make_shared<std::atomic<bool>>(false)) {
        live.fetch_add(1, std::memory_order_relaxed);
    }
    ~HookedCount() {
        gone->store(true, std::memory_order_relaxed);
        live.fetch_sub(1, std::memory_order_relaxed);
    }
    HookedCount(const HookedCount&) = delete;
    HookedCount& operator=(const HookedCount&) = delete;

    // Increment (relaxed: the caller already owns a reference, or is adopting a
    // fresh object). Records a resurrection on a 0 -> 1 after the adoption.
    void AddRef() noexcept {
        entry_hook.fire_if(this);
        if (ref_count.fetch_add(1, std::memory_order_relaxed) == 0 &&
            adopted.exchange(true, std::memory_order_relaxed)) {
            resurrections.fetch_add(1, std::memory_order_relaxed);
        } // 0 -> 1 on an object that was adopted before: resurrection
    } // HookedCount::AddRef()

    // TryAddRef(): increment the strong count if and only if it is nonzero.
    // Returns true iff it incremented; returns false iff it observed a count of
    // 0, in which case the count is left at 0 (an object at 0 is retired, or
    // about to be retired, and must never be revived). Every observed 0 is read
    // with acquire (the zero-load, or the re-read after a failed CAS), and the
    // CAS is acquire on success, relaxed on failure.
    bool TryAddRef() noexcept {
        entry_hook.fire_if(this);
        long count = ref_count.load(std::memory_order_acquire);
        while (count != 0) {
            if (ref_count.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                after_increment_hook.fire_if(this);
                return true;
            }
            // The failed CAS refreshed `count` with a relaxed read; a 0 seen that
            // way does not synchronize. Re-read it with acquire.
            if (count == 0) count = ref_count.load(std::memory_order_acquire);
        } // CAS loop while the count is nonzero
        return false;
    } // HookedCount::TryAddRef()

    // Decrement; true iff this call made the 1 -> 0 transition.
    bool DelRef() noexcept {
        release_hook.fire_if(this);
        return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    // Current count (a snapshot; single-threaded tests read it exactly).
    long use_count() const noexcept { return ref_count.load(std::memory_order_relaxed); }
}; // struct HookedCount

// Pointee with T == U and the hazard-pointer base at offset 0.
struct Data : std::hazard_pointer_obj_base<Data>, HookedCount {
    int value;   // payload, read through T by the tests (get())
    explicit Data(int v) : value(v) {}
    int get() const { return value; }
};

// The interface T of the T != U pointee. Polymorphic (vptr at offset 0), so the
// tests read the payload through a virtual call on T*, and exactly 32 bytes, so
// that Impl's hazard-pointer base lands at offset 32.
struct Iface {
    virtual ~Iface() = default;
    virtual int get() const = 0;
    int value;      // payload, set by Impl's constructor
    long pad[2];    // vptr 8 + value 4 (+4) + 16 = 32 bytes
    explicit Iface(int v) : value(v), pad{} {}
}; // struct Iface
static_assert(sizeof(Iface) == 32, "Iface must be 32 bytes so Impl's hp base lands at offset 32");

// The concrete U of the T != U pointee: the hazard-pointer base SECOND, at
// offset 32 (checked by the fixture). Reclaimed through std::default_delete<Impl>.
struct Impl : Iface, std::hazard_pointer_obj_base<Impl>, HookedCount {
    explicit Impl(int v) : Iface(v) {}
    int get() const override { return value; }
};

// Byte offset of U's hazard-pointer base inside a U object. Constructs a probe
// (a base-subobject offset is not a constant expression); the probe is never
// owned or retired.
template <typename U>
std::ptrdiff_t hp_base_offset() {
    const U probe(0);
    const std::hazard_pointer_obj_base<U>* base = &probe;
    return reinterpret_cast<const char*>(base) - reinterpret_cast<const char*>(&probe);
} // hp_base_offset()

// The offset each pointee type must have; the fixture asserts it, so that a
// reordering of Impl's bases cannot silently turn the offset-32 runs into a
// second copy of the offset-0 runs.
template <typename U> constexpr std::ptrdiff_t expected_hp_base_offset = 0;
template <> constexpr std::ptrdiff_t expected_hp_base_offset<Impl> = 32;

// A test's handle on "has this object been destroyed yet".
using GoneFlag = std::shared_ptr<const std::atomic<bool>>;

// The plain datum a two-thread test publishes through the pointer: an int alone
// in its own 64-byte line, so alone in TSan's 8-byte shadow cell. TSan keeps at
// most 4 accesses per 8 bytes and evicts one at random when a fifth arrives; an
// int that shares its 8 bytes with locals the other thread writes (the hand-over
// results) can lose the record of the racing write before the racing read, and
// the report then appears in only some runs (measured: a stack `int payload`
// next to such locals was reported in 0-20 of 20 runs depending on the stack
// layout; isolated, 20 of 20).
struct alignas(64) LonePayload {
    int value = 0;   // written by one thread, read by the other
};

// Hand-over wait of the two-thread tests: spins (yielding) until `flag` is set,
// reading it RELAXED, so the wait creates no happens-before edge between the
// threads (those tests are about edges the pointer must create). Returns false
// if the flag is still clear after 30 s (the other thread never reached its
// side of the hand-over, e.g. a defective header skipped the hooked call); the
// caller records that and lets the test fail instead of hanging.
bool wait_for(const std::atomic<bool>& flag) {
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!flag.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    } // spin until the other thread sets the flag
    return true;
} // wait_for()

// Fixture for every seam test. SetUp: disarms all hooks, checks the
// hazard-pointer base offset, drains mm_hp (so nothing retired by an earlier
// test is pending and the retired list is empty) and THEN captures the live
// count. TearDown (after the test's locals are destroyed): disarms the hooks,
// drains once (the seam pointees own no pointers, so there is no destructor
// cascade) and checks that the live count is back at its SetUp value and that
// no resurrection happened.
template <typename AtomicPtr>
class SeamTest : public ::testing::Test {
protected:
    using SP = typename AtomicPtr::shared_ptr_type;
    using U = std::remove_pointer_t<decltype(std::declval<const SP&>().get_raw())>;

    // HookedCount::live after SetUp's drain.
    long base_live_ = 0;

    void SetUp() override {
        HookedCount::entry_hook.disarm();
        HookedCount::after_increment_hook.disarm();
        HookedCount::release_hook.disarm();
        ASSERT_EQ(hp_base_offset<U>(), expected_hp_base_offset<U>);
        drain_reclamation();
        base_live_ = HookedCount::live.load();
        HookedCount::resurrections.store(0);
    } // SeamTest::SetUp()

    void TearDown() override {
        HookedCount::entry_hook.disarm();
        HookedCount::after_increment_hook.disarm();
        HookedCount::release_hook.disarm();
        drain_reclamation();
        EXPECT_EQ(HookedCount::live.load(), base_live_)
            << "seam pointees leaked (count never reached 0, or reached 0 without a retire) "
               "or were destroyed twice";
        EXPECT_EQ(HookedCount::resurrections.load(), 0)
            << "a reference count went 0 -> 1 on an object that was already retired";
    } // SeamTest::TearDown()

    // A fresh pointee with payload `v`, adopted by the returned handle (count 1).
    static SP make(int v) { return SP(new U(v)); }

    // The hook key of a handle's pointee: the HookedCount subobject of its U
    // (the mark, if any, is ignored). No count traffic.
    static const HookedCount* key(const SP& p) {
        return static_cast<const HookedCount*>(static_cast<const U*>(p.get()));
    }

    // The destroyed-flag of a handle's pointee (non-null handle).
    static GoneFlag gone_flag(const SP& p) { return key(p)->gone; }

    // The raw word of a handle with the mark bit set (to compare with get_raw()).
    static U* marked_raw(const SP& p) {
        return reinterpret_cast<U*>(reinterpret_cast<std::uintptr_t>(p.get_raw()) | 1);
    }

    // Window / WindowMarked. Oracle: the hazard-pointer protection guarantee --
    // an object protected by load()'s hazard from validation to increment is not
    // reclaimed while protected, and an object at count 0 is never revived.
    //
    // The word holds X's only reference (marked or not). The entry hook on X
    // fires inside load(), after the word was validated and before TryAddRef
    // reads X's count. The nested operations swing the word to Y (X: 1 -> 0, X
    // retired while the in-flight load's hazard protects it), retire an
    // unprotected control Z, and drain. Then: Z is gone (positive control: this
    // drain reclaims), X is not (protected); TryAddRef sees 0, load() retries and
    // returns Y; nothing was revived. Finally, with the load done, X is reclaimed
    // by one drain (its hazard was cleared).
    // Defects pinned: a hazard cleared before the increment, or a `delete`
    // instead of retire (X destroyed inside the hook; then a use after free); a
    // plain increment instead of TryAddRef (X revived from 0 -- a resurrection --
    // and retired a second time: double free); a hazard on the marked word
    // (WindowMarked: X|1 protects nothing); a hazard address formed from the word
    // rather than the typed U* (HpBaseAt32 only: protects the wrong address).
    void run_window(bool marked) {
        AtomicPtr ptr;
        AtomicPtr control;
        SP x = make(1);
        const HookedCount* const x_key = key(x);
        const GoneFlag x_gone = gone_flag(x);
        ptr.store(marked ? x.set_mark() : x);
        x = nullptr;                        // the word now owns X's only reference
        SP z = make(2);
        const GoneFlag z_gone = gone_flag(z);
        control.store(std::move(z));        // Z's only reference is `control`'s word
        SP y = make(3);

        HookedCount::entry_hook.arm(x_key, [&] {
            ptr.store(y);                   // X: 1 -> 0, retired; the in-flight load's hazard pins it
            control.store(nullptr);         // Z: 1 -> 0, retired; nothing protects it
            drain_reclamation();
            EXPECT_TRUE(z_gone->load()) << "positive control: the drain reclaimed nothing";
            EXPECT_FALSE(x_gone->load()) << "X reclaimed while load() was protecting it";
        });
        SP r = ptr.load();
        ASSERT_TRUE(HookedCount::entry_hook.ran) << "the hook on X never fired: window not reached";
        EXPECT_EQ(r.get_raw(), y.get_raw());   // the only value left; Y is unmarked
        ASSERT_TRUE(r);
        EXPECT_EQ(r->get(), 3);
        EXPECT_EQ(HookedCount::resurrections.load(), 0) << "X was revived from count 0";

        r = nullptr;
        drain_reclamation();
        EXPECT_TRUE(x_gone->load()) << "X not reclaimed after load() returned (hazard never cleared?)";
    } // SeamTest::run_window()

    // ChangedBack{Strong,Weak,MarkedStrong,MarkedWeak}. The word holds B; the
    // caller's `expected` is A; desired is C. The CAS attempt fails (B != A).
    // The entry hook on B fires inside the refresh load (its TryAddRef on B) and
    // stores A back into the word -- unmarked, or marked when `to_marked` -- so
    // the value the refresh must report is A (or A|1).
    //
    // Oracles (the documented CAS contracts):
    // - strong, A: the word changed and changed back to the caller's value; a
    //   strong CAS never returns false with `expected` equal to the caller's
    //   original value, so it retries and succeeds: true, word == C.
    // - weak, A: ONE attempt and one refresh; it returns false with `expected ==
    //   A`, a spurious failure that std allows and that this pointer documents as
    //   its deterministic single-attempt semantics (a design decision; this test
    //   pins it, where the strong CAS above returns true).
    // - strong or weak, A|1: A|1 is a different value from A (the mark is part of
    //   the value), so both return false with `expected == A|1`.
    // Defects pinned: a strong CAS without the changed-back retry (strong, A); a
    // changed-back comparison with the mark masked off, which retries on A|1 and
    // succeeds (strong, A|1); no re-validation after the increment, which makes
    // the refresh return the stale B (all four).
    void run_changed_back(bool weak, bool to_marked) {
        AtomicPtr ptr;
        SP a = make(12);
        SP b = make(13);
        SP c = make(14);
        ptr.store(b);
        SP e = a;

        HookedCount::entry_hook.arm(key(b), [&] { ptr.store(to_marked ? a.set_mark() : a); });
        const bool ok = weak ? ptr.compare_exchange_weak(e, c) : ptr.compare_exchange_strong(e, c);
        ASSERT_TRUE(HookedCount::entry_hook.ran) << "the hook on B never fired: refresh load not reached";

        if (to_marked) {
            EXPECT_FALSE(ok);
            EXPECT_EQ(e.get_raw(), marked_raw(a));       // refreshed to A|1, a different value
            EXPECT_EQ(ptr.load().get_raw(), marked_raw(a));
        } else if (weak) {
            EXPECT_FALSE(ok) << "weak CAS makes exactly one attempt";
            EXPECT_EQ(e.get_raw(), a.get_raw());          // false with expected == original
            EXPECT_EQ(ptr.load().get_raw(), a.get_raw()); // and the word is untouched
        } else {
            EXPECT_TRUE(ok) << "strong CAS returned false with expected equal to the original";
            EXPECT_EQ(e.get_raw(), a.get_raw());          // success leaves expected as passed
            EXPECT_EQ(ptr.load().get_raw(), c.get_raw());
        } // expectations per CAS flavour and changed-back value
    } // SeamTest::run_changed_back()

    // Alias{Strong,Weak}: `expected` and `desired` are the same object, as in
    // `a.compare_exchange_strong(e, e)` (the typed suite's StressTest does this).
    // `e` holds A's only reference; the word holds B's only reference. The CAS
    // pre-counts desired (A: 1 -> 2), fails (word B != A), must undo that
    // pre-count on A, and refreshes `e` to B (releasing e's A: A -> 0, retired).
    // Oracle: reference-count balance -- afterwards B has exactly two owners (the
    // word and e), and A, which lost its last reference, is reclaimed by a drain.
    // Defect pinned: the undo performed through `desired` AFTER `expected` was
    // reassigned -- it then decrements B (one owner short: a later premature
    // retire) and leaves A at count 1 forever (a leak).
    void run_alias(bool weak) {
        AtomicPtr ptr;
        SP e = make(15);
        const GoneFlag a_gone = gone_flag(e);
        SP b = make(16);
        U* const b_raw = b.get_raw();
        ptr.store(std::move(b));            // the word holds B's only reference

        const bool ok = weak ? ptr.compare_exchange_weak(e, e) : ptr.compare_exchange_strong(e, e);
        EXPECT_FALSE(ok);
        EXPECT_EQ(e.get_raw(), b_raw);
        EXPECT_EQ(e.use_count(), 2) << "B must be owned by exactly the word and e";
        drain_reclamation();
        EXPECT_TRUE(a_gone->load()) << "A lost both references but was not retired";
    } // SeamTest::run_alias()
}; // class SeamTest

using SeamPointers = ::testing::Types<intr_shared_ptr_hp<Data>, intr_shared_ptr_hp<Iface, Impl>>;

// Names the two instantiations by where the hazard-pointer base sits.
struct SeamPointerNames {
    template <typename AtomicPtr>
    static std::string GetName(int) {
        if constexpr (std::is_same_v<AtomicPtr, intr_shared_ptr_hp<Data>>) return "HpBaseAt0";
        else return "HpBaseAt32";
    }
};

TYPED_TEST_SUITE(SeamTest, SeamPointers, SeamPointerNames);

} // namespace

// The positive control behind every "reclaimed" / "not reclaimed" assertion in
// this file, and the deferred-reclamation contract: the last release retires the
// object instead of deleting it. SetUp's drain left mm_hp's retired list empty,
// so one retirement cannot reach mm_hp's scan threshold (at least 1000): the
// object is still alive right after its last release, and one drain destroys it.
// Defect pinned: `delete` instead of retire (destroyed immediately).
TYPED_TEST(SeamTest, ReleaseRetiresAndOneDrainReclaims) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    SP x = TestFixture::make(40);
    const GoneFlag x_gone = TestFixture::gone_flag(x);
    ptr.store(std::move(x));
    ptr.store(nullptr);                     // X: 1 -> 0, retired
    EXPECT_FALSE(x_gone->load()) << "destroyed at count 0: deleted, not retired";
    drain_reclamation();
    EXPECT_TRUE(x_gone->load()) << "one drain did not reclaim an unprotected retired object";
}

// Unmarked word.
// Oracle and pinned defects: SeamTest::run_window().
TYPED_TEST(SeamTest, Window) {
    this->run_window(false);
}

// Marked word; also pins a hazard published on the marked word.
// Oracle and pinned defects: SeamTest::run_window().
TYPED_TEST(SeamTest, WindowMarked) {
    this->run_window(true);
}

// Oracle: walk exclusivity, NOT linearizability. Returning X would be
// linearizable (X was in the word when load() started), but the pointer
// guarantees more: a word's reference is counted before the word is published
// and released after it is unpublished, and load() re-validates the word after
// its increment and gives the reference back if the word moved. Together these
// make "count == 1 and the one owner is me" mean "nobody can acquire this object
// any more", which LockFreeList's node destructor relies on when it judges a
// node exclusive. Here the test holds X; the entry hook on X, inside load()'s
// validated window, swings the word to Y, leaving X at count 1 (the test's
// handle). load() must not return X, X's count must be back at 1, and the only
// other value the word held during the call is Y.
// Defect pinned: no re-validation after the increment (load() returns X).
TYPED_TEST(SeamTest, Reval) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    SP hx = TestFixture::make(4);
    ptr.store(hx);                          // X: the word + hx
    SP y = TestFixture::make(5);

    HookedCount::entry_hook.arm(TestFixture::key(hx), [&] {
        ptr.store(y);                       // X: 2 -> 1, only hx owns it now
        EXPECT_EQ(hx.use_count(), 1);
    });
    SP r = ptr.load();
    ASSERT_TRUE(HookedCount::entry_hook.ran) << "the hook on X never fired: window not reached";
    EXPECT_NE(r.get_raw(), hx.get_raw()) << "load() acquired X after its count reached 1 = exclusive owner";
    EXPECT_EQ(r.get_raw(), y.get_raw());
    EXPECT_EQ(hx.use_count(), 1) << "load() kept a reference to X";
}

// The word holds X's only reference. The after-increment hook on X fires once
// load()'s TryAddRef has taken X from 1 to 2 and swings the word to Y (X: 2 ->
// 1). load()'s re-validation then fails and the reference it just took is X's
// LAST one: load() itself must make the 1 -> 0 transition and retire X.
// Oracle: reference-count balance and retire-on-zero -- load() returns Y, and X
// is reclaimed by one drain (TearDown's live-count check would also see it).
// With the hook's target reset on firing, a leaked X is unreachable, so the ASan
// build's LeakSanitizer reports it at exit too.
// Defects pinned: on a failed re-validation, no release of the new reference (X
// leaks at count 1), or a release that does not retire X when it reaches 0 (X
// leaks at count 0). Also no re-validation at all (load() returns X).
TYPED_TEST(SeamTest, RevalRetire) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    SP x = TestFixture::make(30);
    const HookedCount* const x_key = TestFixture::key(x);
    const GoneFlag x_gone = TestFixture::gone_flag(x);
    ptr.store(std::move(x));                // the word holds X's only reference
    SP y = TestFixture::make(31);

    HookedCount::after_increment_hook.arm(x_key, [&] { ptr.store(y); });
    SP r = ptr.load();
    ASSERT_TRUE(HookedCount::after_increment_hook.ran) << "the after-increment hook on X never fired";
    EXPECT_EQ(r.get_raw(), y.get_raw());
    drain_reclamation();
    EXPECT_TRUE(x_gone->load()) << "load() dropped X's last reference without releasing and retiring it";
}

// store() must count the reference the word will own BEFORE the publishing
// exchange. The caller passes A's only reference (std::move: a copy would fire
// the hook in the copy constructor). The entry hook on A fires at store()'s
// increment of A and, standing in for another thread, replaces the word with Z
// and drains.
// Oracle: reference ownership -- A is owned by the caller's handle throughout,
// so it is never retired; afterwards the word holds A with one reference.
// Correct order: the hook runs before A is published (the nested store replaces
// the old value W, not A), then A is counted and published.
// Defect pinned: counting after the exchange. A is then already in the word with
// an uncounted reference when the hook runs; the nested store unpublishes it and
// releases what is really the caller's reference, A reaches 0, is retired and
// freed by the drain while the caller still owns it.
TYPED_TEST(SeamTest, StoreCounted) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    ptr.store(TestFixture::make(6));        // W
    SP a = TestFixture::make(7);
    const HookedCount* const a_key = TestFixture::key(a);
    const GoneFlag a_gone = TestFixture::gone_flag(a);
    SP z = TestFixture::make(8);

    HookedCount::entry_hook.arm(a_key, [&] {
        ptr.store(std::move(z));
        drain_reclamation();
        EXPECT_FALSE(a_gone->load()) << "A retired while the caller owned it";
    });
    ptr.store(std::move(a));
    ASSERT_TRUE(HookedCount::entry_hook.ran) << "the hook on A never fired";
    drain_reclamation();
    EXPECT_FALSE(a_gone->load()) << "A retired while the word owned it";
    SP now = ptr.load();
    EXPECT_EQ(TestFixture::key(now), a_key);
    EXPECT_EQ(now.use_count(), 2) << "A must be owned by exactly the word and `now`";
    EXPECT_EQ(HookedCount::resurrections.load(), 0);
}

// The compare-exchange counterpart of StoreCounted: `desired` (D) must be counted
// before the publishing CAS. The caller holds D's only reference; `expected`
// equals the word (W). The entry hook on D fires at the CAS's increment of D and
// replaces the word with Z, then drains.
// Oracle: reference ownership (D is never retired while the caller holds it) and
// the CAS contract: with the correct order the hook runs before the CAS attempt,
// so the CAS sees Z != W, fails, and refreshes `expected` to Z.
// Defect pinned: counting after the CAS. The CAS then publishes D with an
// uncounted reference; the hook unpublishes it, the release takes D to 0, and
// the drain frees D while the caller still owns it.
TYPED_TEST(SeamTest, CasCounted) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    ptr.store(TestFixture::make(9));        // W
    SP d = TestFixture::make(10);           // the caller's only reference to D
    const GoneFlag d_gone = TestFixture::gone_flag(d);
    SP e = ptr.load();                      // W
    SP z = TestFixture::make(11);
    typename TestFixture::U* const z_raw = z.get_raw();

    HookedCount::entry_hook.arm(TestFixture::key(d), [&] {
        ptr.store(std::move(z));
        drain_reclamation();
        EXPECT_FALSE(d_gone->load()) << "D retired while the caller owned it";
    });
    const bool ok = ptr.compare_exchange_strong(e, d);
    ASSERT_TRUE(HookedCount::entry_hook.ran) << "the hook on D never fired";
    EXPECT_FALSE(ok) << "the word was replaced before the CAS attempt; it must fail";
    EXPECT_EQ(e.get_raw(), z_raw);
    EXPECT_FALSE(d_gone->load());
    EXPECT_EQ(d.use_count(), 1) << "the failed CAS must undo its pre-count of D";
    EXPECT_EQ(HookedCount::resurrections.load(), 0);
}

// Strong CAS, changed back to A: retries and succeeds.
// Oracle and pinned defects: SeamTest::run_changed_back().
TYPED_TEST(SeamTest, ChangedBackStrong) {
    this->run_changed_back(false, false);
}

// Weak CAS, changed back to A: false with expected == A (single attempt).
// Oracle and pinned defects: SeamTest::run_changed_back().
TYPED_TEST(SeamTest, ChangedBackWeak) {
    this->run_changed_back(true, false);
}

// Strong CAS, changed to A|1: false with expected == A|1.
// Oracle and pinned defects: SeamTest::run_changed_back().
TYPED_TEST(SeamTest, ChangedBackMarkedStrong) {
    this->run_changed_back(false, true);
}

// Weak CAS, changed to A|1: false with expected == A|1.
// Oracle and pinned defects: SeamTest::run_changed_back().
TYPED_TEST(SeamTest, ChangedBackMarkedWeak) {
    this->run_changed_back(true, true);
}

// Strong CAS with expected and desired the same object.
// Oracle and pinned defects: SeamTest::run_alias().
TYPED_TEST(SeamTest, AliasStrong) {
    this->run_alias(false);
}

// Weak CAS with expected and desired the same object.
// Oracle and pinned defects: SeamTest::run_alias().
TYPED_TEST(SeamTest, AliasWeak) {
    this->run_alias(true);
}

// Publication tests (two threads; the oracle is TSan's happens-before analysis,
// so these can fail only in the TSan build -- in the ASan build on x86 they are
// plain smoke tests). The main thread creates the reader thread FIRST and only
// then constructs and publishes the pointee, so thread creation orders nothing
// the reader later reads: the reader's plain reads of the pointee's
// constructor-written fields (its payload, and for Iface its vptr) are ordered
// after the constructor only through the pointer. The contract: the publishing
// RMW is at least `release` whatever `order` the caller passed (a relaxed order
// is promoted), and a load(acquire) synchronizes with the store/CAS that
// published the word it returns. So the tests publish with
// memory_order_relaxed and read with load(acquire). If the publishing RMW is
// relaxed, the reader's accesses race with the constructor and TSan reports a
// data race (the binary exits 66 although the test body passes). TSan judges
// the missing edge, not the timing, so this fails on every run PROVIDED that
// nothing else carries an edge from the main thread to the reader:
// - the pointee's AddRef() is a relaxed increment, so the reader's acquire
//   reads of the count (TryAddRef) synchronize with nothing the main thread did
//   BEFORE publishing;
// - but store() takes `desired` by value and releases that parameter's
//   reference when it returns: an acq_rel decrement of the published pointee's
//   count. A reader whose TryAddRef reads the count after that decrement
//   synchronizes with it and hides the defect (measured: such a test caught a
//   relaxed exchange in 1 of 21 runs). So StorePublishesPointee stops the main
//   thread INSIDE store(), between the exchange and the return, in the release
//   hook of the replaced value W, and lets the reader do all its work there;
// - the two threads hand over with relaxed atomic flags, which carry no edge;
// - the compare-exchange takes `desired` by const reference and keeps no
//   reference of its own past the call, so CasPublishesPointee needs no stop.
// Defect pinned: store()'s publishing exchange relaxed (always, or when the
// caller passes relaxed).
TYPED_TEST(SeamTest, StorePublishesPointee) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    SP w = TestFixture::make(49);
    const HookedCount* const w_key = TestFixture::key(w);
    ptr.store(std::move(w));                // W, published before the reader exists

    std::atomic<bool> published{false};     // main -> reader: the exchange is done (relaxed)
    std::atomic<bool> reader_done{false};   // reader -> main: the reader has read the pointee
    int seen = -1;                          // written by the reader, read after join()
    bool reader_timed_out = false;          // set by the hook if the reader never finishes
    // Armed before the reader exists, so the reader's DelRef calls (other
    // objects) only ever read a stable target.
    HookedCount::release_hook.arm(w_key, [&] {
        published.store(true, std::memory_order_relaxed);
        reader_timed_out = !wait_for(reader_done);   // wait for the reader, inside store()
    });
    std::thread reader([&] {
        while (!published.load(std::memory_order_relaxed)) std::this_thread::yield();
        SP r = ptr.load(std::memory_order_acquire);
        if (r) seen = r->get();
        reader_done.store(true, std::memory_order_relaxed);
    });
    ptr.store(TestFixture::make(50), std::memory_order_relaxed);
    reader.join();
    ASSERT_TRUE(HookedCount::release_hook.ran) << "store() did not release the replaced value";
    EXPECT_FALSE(reader_timed_out);
    EXPECT_EQ(seen, 50);
}

// As StorePublishesPointee, publishing with compare_exchange_strong(null ->
// D, relaxed, relaxed). Defect pinned: the publishing compare-exchange relaxed
// (always, or when the caller passes relaxed).
TYPED_TEST(SeamTest, CasPublishesPointee) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    int seen = -1;                          // written by the reader, read after join()
    std::thread reader([&] {
        SP r;
        while (!(r = ptr.load(std::memory_order_acquire))) std::this_thread::yield();
        seen = r->get();
    });
    SP expected;
    const SP d = TestFixture::make(51);
    EXPECT_TRUE(ptr.compare_exchange_strong(expected, d, std::memory_order_relaxed,
                                            std::memory_order_relaxed));
    reader.join();
    EXPECT_EQ(seen, 51);
}

// The other half of a publishing RMW's order: store()'s exchange READS the old
// word and store() then releases the old occupant (an RMW on its count, and on
// 1 -> 0 a retire whose scan runs its destructor). Those accesses must be
// ordered after the old occupant's construction on the thread that published
// it, so the exchange must also ACQUIRE, whatever `order` the caller passed (a
// release is promoted to acq_rel). Oracle: the pointer's own operations are
// free of data races for every valid `order`.
// A publisher thread, created first, constructs O and publishes it with a
// compare-exchange (const& desired: the publisher touches O's count only with
// relaxed increments), keeps its own handle until the main thread is done, and
// hands over with relaxed flags. The main thread then store()s null with
// memory_order_release: its exchange reads O, and its release of O is an RMW on
// a count whose initialization (a plain write in O's constructor) it is ordered
// after only through the exchange's acquire half. Missing that, TSan reports a
// data race on every run. Same structure as the publication tests above, the
// other way round.
// Defect pinned: publishing RMWs promoted only to `release` (or not promoted).
TYPED_TEST(SeamTest, StoreAcquiresReplacedPointee) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    std::atomic<bool> published{false};     // publisher -> main: O is in the word (relaxed)
    std::atomic<bool> replaced{false};      // main -> publisher: O was replaced and released
    bool publish_ok = false;                // written by the publisher, read after join()
    std::thread publisher([&] {
        const SP o = TestFixture::make(60);
        SP expected;
        publish_ok = ptr.compare_exchange_strong(expected, o, std::memory_order_release,
                                                 std::memory_order_relaxed);
        published.store(true, std::memory_order_relaxed);
        // Keep `o` (one of O's two owners) until the main thread's release of O is
        // done, so that the main thread's RMW on O's count cannot read from this
        // handle's acq_rel decrement and synchronize through it.
        while (!replaced.load(std::memory_order_relaxed)) std::this_thread::yield();
    });
    while (!published.load(std::memory_order_relaxed)) std::this_thread::yield();
    ptr.store(nullptr, std::memory_order_release);   // reads O, releases the word's reference to O
    replaced.store(true, std::memory_order_relaxed);
    publisher.join();
    EXPECT_TRUE(publish_ok);
    EXPECT_FALSE(ptr.load());
}

// load(acquire) that returns null must synchronize with the store that wrote the
// null, like any acquire load: the null path returns before any hazard or count
// operation, so the first read of the word is its only read and must use the
// caller's order. The word starts as MARKED null (no pointee at all, so no count
// operation can carry an edge either); the main thread writes a plain int, then
// stores null (seq_cst, i.e. a release publish); the reader spins on
// load(acquire) until it sees unmarked null, then reads the int. Same oracle as
// above (TSan; created before the write).
// Defect pinned: load()'s first read of the word performed relaxed (ignoring the
// caller's order), which leaves the null/marked-null return unsynchronized. The
// non-null path does not show this defect: its acquire validation reload
// synchronizes anyway.
TYPED_TEST(SeamTest, AcquireLoadOfNullSynchronizes) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    ptr.store(SP().set_mark());             // marked null: a value with no pointee
    LonePayload payload;                    // the plain data published by the null store
    int seen = -1;                          // written by the reader, read after join()
    std::thread reader([&] {
        while (ptr.load(std::memory_order_acquire).is_marked()) std::this_thread::yield();
        seen = payload.value;
    });
    payload.value = 42;
    ptr.store(nullptr);
    reader.join();
    EXPECT_EQ(seen, 42);
}

// load(acquire) must synchronize with a publisher that changed ONLY THE MARK:
// the word X -> X|1 by a compare-exchange with release, after the publisher
// wrote plain data. A reader whose first read and validation reload both saw the
// unmarked X takes the returned word (X|1) from its post-increment
// re-validation reload, which must therefore be ACQUIRE: neither earlier read
// observed the mark CAS, and nothing else orders the data before the reader's
// read of it.
// The after-increment hook on X fires on the READER thread, inside its load(),
// after TryAddRef took X and before the re-validation. It tells the main thread
// (relaxed flag) and waits (relaxed flag); the main thread writes a plain int,
// marks the word with compare_exchange_strong(X -> X|1, release), and releases
// the reader. The reader's load() then returns X|1 and the reader reads the int.
// No other edge exists: the reader was created before the int was written, the
// hand-over flags are relaxed, and the main thread's count operations on X
// (the CAS's pre-count, its release of the old occupant) come after the
// reader's last count operation before the read (its TryAddRef). Same oracle as
// the publication tests above (TSan; every run).
// Defect pinned: the re-validation reload relaxed.
TYPED_TEST(SeamTest, AcquireLoadSeesMarkOnlyPublication) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    SP x = TestFixture::make(70);
    ptr.store(x);                           // the word holds X, unmarked
    const SP x_marked = x.set_mark();       // the mark CAS's desired, made before the reader exists
    std::atomic<bool> in_window{false};     // reader -> main: inside load(), X incremented
    std::atomic<bool> marked{false};        // main -> reader: data written, word marked
    LonePayload payload;                    // the plain data published by the mark CAS
    bool reader_timed_out = false;          // written by the hook (reader thread), read after join()
    bool saw_mark = false;                  // written by the reader, read after join()
    int seen = -1;                          // written by the reader, read after join()
    HookedCount::after_increment_hook.arm(TestFixture::key(x), [&] {
        in_window.store(true, std::memory_order_relaxed);
        reader_timed_out = !wait_for(marked);
    });
    std::thread reader([&] {
        const SP r = ptr.load(std::memory_order_acquire);
        saw_mark = r.is_marked();
        if (saw_mark) seen = payload.value;
    });
    const bool main_timed_out = !wait_for(in_window);
    payload.value = 42;
    SP expected = x;
    const bool ok = ptr.compare_exchange_strong(expected, x_marked, std::memory_order_release,
                                                std::memory_order_relaxed);
    marked.store(true, std::memory_order_relaxed);
    reader.join();
    ASSERT_TRUE(HookedCount::after_increment_hook.ran) << "the reader's load() never incremented X";
    EXPECT_FALSE(main_timed_out);
    EXPECT_FALSE(reader_timed_out);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(saw_mark) << "load() returned a value older than its re-validation";
    EXPECT_EQ(seen, 42);
}

// load(acquire) returning null from the ZERO-COUNT path must synchronize with
// the store that wrote the null. The reader validated X; at the entry of its
// TryAddRef on X (entry hook, reader thread) the main thread, standing by,
// stores Y (X: 1 -> 0, retired, protected by the reader's hazard), THEN writes
// a plain int, THEN stores null, and releases the reader. TryAddRef reads 0 --
// with acquire, so it synchronizes with the main thread's decrement of X, but
// that decrement came BEFORE the int was written -- and load() reloads the word
// to retry, gets null and returns it at once. That reload is the only read of
// the null-publishing store, so it must be ACQUIRE. Splitting the two stores is
// what makes this observable: if the store that unpublished X also wrote the
// null, the acquire zero-count read would carry the edge by itself. No other
// edge: reader created first, relaxed hand-over, the reader never touches Y,
// and the main thread runs no scan (one retirement after the SetUp drain).
// Defect pinned: the zero-count reload relaxed.
TYPED_TEST(SeamTest, AcquireLoadOfNullAfterZeroCountSynchronizes) {
    using SP = typename TestFixture::SP;
    TypeParam ptr;
    SP x = TestFixture::make(80);
    const HookedCount* const x_key = TestFixture::key(x);
    ptr.store(std::move(x));                // the word holds X's only reference
    const SP y = TestFixture::make(81);
    std::atomic<bool> in_window{false};     // reader -> main: inside load(), X validated
    std::atomic<bool> swung{false};         // main -> reader: X unpublished, data written, null stored
    LonePayload payload;                    // the plain data published by the null store
    bool reader_timed_out = false;          // written by the hook (reader thread), read after join()
    bool saw_null = false;                  // written by the reader, read after join()
    int seen = -1;                          // written by the reader, read after join()
    HookedCount::entry_hook.arm(x_key, [&] {
        in_window.store(true, std::memory_order_relaxed);
        reader_timed_out = !wait_for(swung);
    });
    std::thread reader([&] {
        const SP r = ptr.load(std::memory_order_acquire);
        saw_null = !r && !r.is_marked();
        if (saw_null) seen = payload.value;
    });
    const bool main_timed_out = !wait_for(in_window);
    ptr.store(y);                           // X: 1 -> 0 (retired; the reader's hazard pins it)
    payload.value = 42;
    ptr.store(nullptr);                     // the store whose null the reader returns
    swung.store(true, std::memory_order_relaxed);
    reader.join();
    ASSERT_TRUE(HookedCount::entry_hook.ran) << "the reader's load() never reached TryAddRef on X";
    EXPECT_FALSE(main_timed_out);
    EXPECT_FALSE(reader_timed_out);
    EXPECT_TRUE(saw_null);
    EXPECT_EQ(seen, 42);
}
