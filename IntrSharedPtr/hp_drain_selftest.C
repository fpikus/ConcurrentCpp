// Self-test of hp_drain.h: the drain helper that tests and benchmarks of
// intr_shared_ptr_hp rely on before asserting live counts, and of the oracles
// those tests use. Built twice, with ASan and with TSan (both at -O0).
//
// What is pinned here, and why each matters to the tests built on the drain:
// - A destructor cascade of depth d needs exactly d drain rounds, one layer per
//   round (a test that releases a chain must drain once per layer, or use
//   drain_until() with enough rounds).
// - drain_until() stops at its round cap with every round completed and returns
//   false, and returns true without draining when the predicate already holds.
// - A drain does not reclaim an object protected by a hazard pointer, while the
//   same drain does reclaim an unprotected one (the positive control: without it,
//   "not reclaimed" could mean "the drain did nothing").
// - A drain still reclaims everything when the scan threshold is raised above
//   the default 1000 by many hazard records: the reason the drain waits for a
//   sentinel instead of retiring a fixed number of fillers.
// - A drain called inside a scan (from a deleter) hits the filler cap and aborts
//   (SIGABRT) with its message instead of spinning or returning with its
//   sentinel pending.
// - LeakSanitizer reports an object that was allocated and never retired, and
//   does NOT report objects that were retired but are still pending in mm_hp at
//   exit: they stay reachable from mm_hp's retired list, a global (ASan build
//   only). So LSan can catch a reference-count drift that never reaches 0, and
//   the live count after a drain is the oracle for everything that was retired.
//
// Everything runs on the main thread (or in a single-threaded death-test child):
// the drain's quiescence precondition holds throughout, and every scan runs
// synchronously inside one of this thread's retire() calls.

#include <gtest/gtest.h>

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "hp_drain.h"
#include "hp_drain_gtest.h"

// Sanitizer detection, the same two spellings mm_hp.cpp uses for TSan: clang
// has __has_feature(); GCC defines __SANITIZE_ADDRESS__ / __SANITIZE_THREAD__.
#if defined(__SANITIZE_ADDRESS__)
#define HP_DRAIN_SELFTEST_ASAN 1
#endif
#if defined(__SANITIZE_THREAD__)
#define HP_DRAIN_SELFTEST_TSAN 1
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HP_DRAIN_SELFTEST_ASAN 1
#endif
#if __has_feature(thread_sanitizer)
#define HP_DRAIN_SELFTEST_TSAN 1
#endif
#endif

namespace {

// A retirable test object: counts live instances, optionally reports its own
// destruction through a flag, and optionally owns one child that its destructor
// retires (one layer of a destructor cascade, as a list node's destructor does
// when it releases its successor).
struct Tracked : std::hazard_pointer_obj_base<Tracked> {
    // Number of Tracked objects constructed and not yet destroyed. Relaxed
    // everywhere: every test here is single-threaded.
    static std::atomic<int> live;

    // `child`: retired by this object's destructor, or nullptr.
    // `destroyed`: set by this object's destructor, or nullptr.
    explicit Tracked(Tracked* child = nullptr, std::atomic<bool>* destroyed = nullptr) noexcept
        : child_(child), destroyed_(destroyed) {
        live.fetch_add(1, std::memory_order_relaxed);
    }

    // Poisons the payload. ASan reports a read of a reclaimed object by itself.
    // TSan's use-after-free detection is not something to rely on for the
    // single-threaded reads in this file; but a read of a reclaimed object whose
    // memory was not reused yet sees -1 and fails the payload check. (The store
    // is kept because the tests are built at -O0.)
    ~Tracked() {
        live.fetch_sub(1, std::memory_order_relaxed);
        payload_ = -1;
        if (child_) child_->retire();
        if (destroyed_) destroyed_->store(true, std::memory_order_relaxed);
    }

    Tracked(const Tracked&) = delete;
    Tracked& operator=(const Tracked&) = delete;

    Tracked* const child_;                  // next layer of the cascade, or nullptr
    std::atomic<bool>* const destroyed_;    // destruction report, or nullptr
    int payload_ = 42;                      // readable only while the object is alive
}; // struct Tracked

std::atomic<int> Tracked::live{0};

// Builds a chain of `depth` Tracked objects, each owning the next, and returns
// its head. Retiring the head starts a destructor cascade of depth `depth`.
Tracked* make_chain(int depth) {
    Tracked* head = nullptr;
    for (int i = 0; i < depth; ++i) head = new Tracked(head);
    return head;
}

// The current number of live Tracked objects (Tracked::live, read relaxed: the
// tests are single-threaded).
int live_tracked() { return Tracked::live.load(std::memory_order_relaxed); }

// A retired object whose destructor calls a drain: the drain then runs inside a
// scan, where it can never finish (see the death test).
struct DrainsInDestructor : std::hazard_pointer_obj_base<DrainsInDestructor> {
    ~DrainsInDestructor() { drain_reclamation(); }
};

} // namespace

// A drain is one round: with nothing protected and no cascade, a drain leaves no
// pending retirement, and the next retirement does not reclaim anything until a
// further drain (the scan threshold is not reached by one retirement).
TEST(HpDrain, OneRoundReclaimsUnprotectedRetirements) {
    drain_reclamation();
    const int base = live_tracked();
    for (int i = 0; i < 10; ++i) (new Tracked)->retire();
    EXPECT_EQ(live_tracked(), base + 10) << "retire() reclaimed below the scan threshold";
    drain_reclamation();
    EXPECT_EQ(live_tracked(), base);
} // OneRoundReclaimsUnprotectedRetirements

// One layer of a destructor cascade per round: objects retired by destructors
// during a scan are pushed after that scan took the retired list, and the
// scanning thread starts no nested scan. Checked after every round, so the test
// pins "exactly one layer per round", not only the total.
TEST(HpDrain, CascadeOfDepthDNeedsDRounds) {
    for (int depth : {1, 2, 3, 5}) {
        SCOPED_TRACE(::testing::Message() << "depth " << depth);
        drain_reclamation();
        const int base = live_tracked();
        make_chain(depth)->retire();
        ASSERT_EQ(live_tracked(), base + depth);
        for (int round = 1; round <= depth; ++round) {
            drain_reclamation();
            EXPECT_EQ(live_tracked(), base + depth - round) << "after round " << round;
        } // loop over drain rounds
        drain_reclamation();
        EXPECT_EQ(live_tracked(), base) << "an extra round changed the count";
    } // loop over cascade depths
} // CascadeOfDepthDNeedsDRounds

// drain_until(): false at the round cap, every round completed (the binary keeps
// running, and a later drain_until() finishes the job); true without draining
// when the predicate holds on entry (an unprotected retired object survives it).
TEST(HpDrain, DrainUntilStopsAtRoundCap) {
    drain_reclamation();
    const int base = live_tracked();
    make_chain(3)->retire();
    auto all_reclaimed = [base] { return live_tracked() == base; };
    EXPECT_FALSE(drain_until(all_reclaimed, 2));
    EXPECT_EQ(live_tracked(), base + 1) << "two rounds must reclaim exactly two layers";
    EXPECT_TRUE(drain_until(all_reclaimed, 1));
    EXPECT_EQ(live_tracked(), base);

    (new Tracked)->retire();
    EXPECT_TRUE(drain_until([] { return true; }, 5));
    EXPECT_EQ(live_tracked(), base + 1) << "drain_until() drained although the predicate held on entry";
    EXPECT_TRUE(drain_until(all_reclaimed, 1));
} // DrainUntilStopsAtRoundCap

// A hazard pointer keeps a retired object alive across a drain, while the
// unprotected control retired just before the drain is reclaimed by it. The
// protected object's payload is read after the drain: under ASan a premature
// free is a reported use-after-free, and the destructor's poison catches reuse.
// The hazard is published with reset_protection(const T*), the call the HP
// pointer's load protocol is specified to use (protect()/try_protect() cannot
// validate a marked word).
TEST(HpDrain, ProtectedObjectSurvivesWhileControlIsReclaimed) {
    drain_reclamation();
    std::atomic<bool> protected_destroyed{false};
    std::atomic<bool> control_destroyed{false};
    Tracked* guarded = new Tracked(nullptr, &protected_destroyed);
    Tracked* control = new Tracked(nullptr, &control_destroyed);
    {
        std::hazard_pointer hp = std::make_hazard_pointer();
        hp.reset_protection(guarded);
        guarded->retire();
        control->retire();
        drain_reclamation();
        EXPECT_TRUE(control_destroyed.load(std::memory_order_relaxed)) << "positive control not reclaimed";
        EXPECT_FALSE(protected_destroyed.load(std::memory_order_relaxed)) << "protected object reclaimed";
        EXPECT_EQ(guarded->payload_, 42);
        hp.reset_protection();
    } // hazard pointer scope
    drain_reclamation();
    EXPECT_TRUE(protected_destroyed.load(std::memory_order_relaxed)) << "not reclaimed after the hazard was cleared";
} // ProtectedObjectSurvivesWhileControlIsReclaimed

// A drain inside a scan (here: from the destructor of a retired object, run by
// the outer drain's scan) can never see its sentinel reclaimed, because mm_hp
// starts no scan on a thread that is already scanning. It must abort at the
// filler cap with its message. The "threadsafe" style re-executes the binary for
// the child instead of forking a process that may have sanitizer threads.
// The oracle is SIGABRT plus the message, and the statement contains exactly one
// outer drain: a cap that RETURNED would let the outer drain's scan continue,
// and with a second drain in the statement the dangling stack flag would still
// kill the child later, so a looser oracle (any death, or the message alone)
// passes with a broken cap. The test takes over 2 s: the drain's pauses.
TEST(HpDrainDeathTest, DrainInsideScanAbortsAtFillerCap) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            (new DrainsInDestructor)->retire();
            drain_reclamation();
        },
        ::testing::KilledBySignal(SIGABRT),
        "drain_reclamation\\(\\): the sentinel was not reclaimed after 1000000 fillers");
} // DrainInsideScanAbortsAtFillerCap

namespace {

// Hazard records the threshold-independence test holds at once. mm_hp's scan
// threshold is max(1000, 2 * records), so 600 records raise it to 1200, above
// the 1000 retirements a fixed-count "drain" would make.
constexpr int raised_records = 600;

} // namespace

// Threshold independence: with the scan threshold raised to 1200 by 600 hazard
// records, one drain must still reclaim an unprotected control retired before
// it, and must have needed more than 1000 fillers (proof the raised threshold
// was in force). Runs in an exit-test child because mm_hp never frees hazard
// records: the raised threshold would otherwise stay for every later test.
// The child exits 0 on success and prints what it saw for the regex.
TEST(HpDrainDeathTest, DrainAdaptsToRaisedThreshold) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            std::vector<std::hazard_pointer> records;
            records.reserve(raised_records);
            for (int i = 0; i < raised_records; ++i) records.push_back(std::make_hazard_pointer());
            std::atomic<bool> control_destroyed{false};
            (new Tracked(nullptr, &control_destroyed))->retire();
            const long fillers = drain_reclamation();
            const bool destroyed = control_destroyed.load(std::memory_order_relaxed);
            std::fprintf(stderr, "fillers=%ld control_destroyed=%d\n", fillers, destroyed ? 1 : 0);
            std::exit(destroyed && fillers > 1000 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "fillers=[0-9]+ control_destroyed=1");
} // DrainAdaptsToRaisedThreshold

#ifdef HP_DRAIN_SELFTEST_ASAN

namespace {

// Number of objects the LSan positive control leaks. An unusual count, so that
// the "in 7 object(s)" of the expected report line identifies this leak and no
// other: LSan groups leaks by allocation stack, and the seven allocations share
// one.
constexpr int leaked_objects = 7;

// Allocates `leaked_objects` Tracked objects and drops every pointer to them
// without retiring them: a never-retired leak, the kind a reference-count drift
// that never reaches 0 produces. noinline keeps the pointers in this frame,
// which is dead after the return (and then scrubbed).
[[gnu::noinline]] void leak_never_retired() {
    for (int i = 0; i < leaked_objects; ++i) {
        Tracked* volatile leaked = new Tracked;
        (void)leaked;
    } // loop over leaked objects
} // leak_never_retired()

// Overwrites the dead stack region below the caller's frame with zeros. LSan
// scans stacks conservatively, including uninitialized slots of live frames, and
// a stale copy of a leaked pointer left by a dead frame can hide the leak. The
// scrub makes the control independent of the stack layout of the child.
[[gnu::noinline]] void scrub_stack() {
    volatile unsigned char buffer[64*1024];
    for (std::size_t i = 0; i < sizeof(buffer); ++i) buffer[i] = 0;
}

// Exit-status predicate: the child exited normally (not by a signal) with a
// non-zero status. LSan turns the child's exit(0) into its own failure status
// (1 under ASan, unless ASAN_OPTIONS sets exitcode).
bool exited_nonzero(int status) {
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

} // namespace

// LSan positive control: a child that leaks never-retired objects and calls
// exit(0) must be turned into a failing exit by LeakSanitizer's at-exit check,
// with the report naming exactly our objects. The negative twin (same child,
// no leak) must exit 0, so the report is caused by the leak and not by anything
// else the child process holds at exit (gtest, mm_hp's records and domain).
// The third child retires a depth-5 cascade and exits without draining: mm_hp's
// exit-time scan reclaims one layer and four stay pending, reachable from mm_hp's
// retired list (a global), so LSan reports nothing and the child exits 0. That
// is the blind spot: retired-but-pending objects are invisible to LSan.
TEST(HpDrainDeathTest, LeakSanitizerReportsNeverRetiredObject) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            scrub_stack();
            std::exit(0);
        },
        ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(
        {
            leak_never_retired();
            scrub_stack();
            std::exit(0);
        },
        exited_nonzero,
        "LeakSanitizer: detected memory leaks(.|\n)*in " + std::to_string(leaked_objects) + " object\\(s\\)");
    EXPECT_EXIT(
        {
            make_chain(5)->retire();
            scrub_stack();
            std::exit(0);
        },
        ::testing::ExitedWithCode(0), "");
} // LeakSanitizerReportsNeverRetiredObject

#else // HP_DRAIN_SELFTEST_ASAN

TEST(HpDrainDeathTest, LeakSanitizerReportsNeverRetiredObject) {
#ifdef HP_DRAIN_SELFTEST_TSAN
    GTEST_SKIP() << "not applicable under TSan: ThreadSanitizer has no leak checker "
                    "(LeakSanitizer runs with ASan or standalone); the ASan build of this "
                    "test is the LSan positive control";
#else
    GTEST_SKIP() << "not applicable without ASan: LeakSanitizer is not linked in; the "
                    "ASan build of this test is the LSan positive control";
#endif
} // LeakSanitizerReportsNeverRetiredObject

#endif // HP_DRAIN_SELFTEST_ASAN
