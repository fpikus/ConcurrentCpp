// Contract tests for Thread (thread.h): black-box tests of what the header's
// declarations and contract comments promise, one group per promise:
//   - the type: move-only, nothrow members, explicit and constrained
//     constructors, the Attributes aggregate;
//   - invocation: the callable runs std::invoke on decay-copies made in the
//     creating thread; the copies are destroyed exactly once, by the new
//     thread, after the callable returns and before join() returns;
//   - lifetime: the destructor and move assignment join, join()/detach()
//     transitions and their std::system_error codes, get_id() and
//     native_handle() in every state, swap, move construction;
//   - failures: the class's EINVAL before anything is copied (f and args
//     untouched), pthread_create's error after the copies (copies destroyed
//     in the creating thread before the exception leaves, rvalue arguments
//     moved from), an exception while copying; no leak on any of them;
//   - the start mask: what the callable's first action sees for creators with
//     different masks, the creator's mask after the constructor returns or
//     throws, signals sent to a worker;
//   - stack and guard: lower bounds below the callable's first frame, the
//     guard mapping, the process default stack size and its late binding, the
//     room for the static TLS, the per-thread stagger;
//   - termination: the destructor or move assignment on the thread it would
//     join, an exception escaping the callable, pthread_exit in the callable.
//
// The binary is built three ways: under AddressSanitizer and under
// ThreadSanitizer at -O0, and optimized without a sanitizer. The sanitizer
// runtimes intercept pthread_create and change what the new thread begins
// with (AddressSanitizer sets the new thread's signal mask back to its
// creator's, ThreadSanitizer enlarges small stack requests), and -O0 keeps
// code that an optimizer may remove or inline; only the optimized build runs
// the header as it ships. Under ThreadSanitizer several tests are also
// happens-before oracles: a plain (non-atomic) field written on one side of a
// promised ordering and read on the other is reported as a race if the
// ordering is missing.
//
// Conventions of the file:
//   - process-wide state a test changes (the default stack size, signal
//     dispositions, an environment variable) is restored by RAII objects;
//   - tests that set a signal mask do it in a helper std::thread (the
//     "creator"), never in the GoogleTest main thread;
//   - masks are compared signal by signal with sigismember over
//     1..SIGRTMAX: the kernel fills only part of a sigset_t and reports
//     SIGKILL, SIGSTOP and glibc's internal signals as unblocked, so a byte
//     comparison or a comparison with a full set is meaningless;
//   - stack and guard sizes are checked as lower bounds: glibc reuses cached
//     stacks larger than the request, and ThreadSanitizer raises small
//     requests;
//   - every object whose lifetime is checked is a Tracked (below), counted
//     per object, so a double destruction cannot cancel a leak in a total.
#include <gtest/gtest.h>

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "thread.h"

// Which sanitizer this binary is built with: some oracles do not apply under
// one of them (each use says why).
#if defined(__SANITIZE_THREAD__)
#define THREAD_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define THREAD_TEST_TSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define THREAD_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define THREAD_TEST_ASAN 1
#endif
#endif

// The sanitizers' heap counter. Declared weak by hand: it is defined by the
// AddressSanitizer and ThreadSanitizer runtimes and absent (null) in the
// optimized build, where the leak checks are skipped.
extern "C" __attribute__((weak)) std::size_t __sanitizer_get_current_allocated_bytes();

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

// 128 KiB of static TLS in every thread of this binary. glibc carves the
// static TLS out of each thread's stack block, so a Thread whose stack
// request is smaller than this fails or gets less than its request unless
// the class adds room for the static TLS, as Attributes::stack_size promises.
// Every stack bound in this file is therefore checked with this TLS present.
// The array is zero-initialized (no dynamic initialization); the TLS test
// takes its address so that it is kept.
[[gnu::tls_model("initial-exec")]] thread_local std::array<unsigned char, 128 * 1024> big_tls;

// ---------------------------------------------------------------------------
// Census: per-object construction and destruction records for Tracked.
//
// Every Tracked object gets a fresh id. Its slot counts constructions and
// destructions (relaxed atomics: they add no happens-before edge, so they do
// not hide a missing one from ThreadSanitizer) and records, in plain fields,
// the thread that made it and the thread that destroyed it. The plain fields
// are read only after an edge the contract promises (join() returning, the
// copies being complete before the invocation), so a missing edge is a
// reported race. Ids are never reused within a process; tests record the
// next id before they begin (census::mark()) and check the ids from there.
namespace census {

constexpr int kSlots = 1 << 16;

struct Slot {
    std::atomic<int> constructions{0};  // must end at exactly 1
    std::atomic<int> destructions{0};   // must end at exactly 1
    std::thread::id made_on;            // plain: the thread whose constructor made it
    std::thread::id destroyed_on;       // plain: the thread whose destructor ended it
}; // struct Slot

std::array<Slot, kSlots> slots;
std::atomic<int> next_id{0};

// Allocates an id for a new object and records its construction.
int enroll() {
    const int id = next_id.fetch_add(1, std::memory_order_relaxed);
    if (id >= kSlots) {
        std::fputs("census: out of slots\n", stderr);
        std::abort();
    }
    slots[id].constructions.fetch_add(1, std::memory_order_relaxed);
    slots[id].made_on = std::this_thread::get_id();
    return id;
} // enroll()

// Records the destruction of the object with this id.
void retire(int id) {
    slots[id].destructions.fetch_add(1, std::memory_order_relaxed);
    slots[id].destroyed_on = std::this_thread::get_id();
} // retire()

// The id the next object will get: a test's first id.
int mark() { return next_id.load(std::memory_order_relaxed); }

int destructions(int id) { return slots[id].destructions.load(std::memory_order_relaxed); }

// Success if every object enrolled in [first, mark()) was constructed once
// and destroyed exactly once.
testing::AssertionResult each_destroyed_once(int first) {
    const int last = mark();
    for (int id = first; id < last; ++id) {
        const int c = slots[id].constructions.load(std::memory_order_relaxed);
        const int d = slots[id].destructions.load(std::memory_order_relaxed);
        if (c != 1 || d != 1) {
            return testing::AssertionFailure() << "object " << id << ": " << c << " constructions, " << d
                                               << " destructions";
        }
    } // loop over the test's objects
    return testing::AssertionSuccess();
} // each_destroyed_once()

// Success if every object enrolled in [first, mark()) has been destroyed
// exactly once, and on the thread `who`.
testing::AssertionResult each_destroyed_once_on(int first, std::thread::id who) {
    testing::AssertionResult once = each_destroyed_once(first);
    if (!once) return once;
    for (int id = first; id < mark(); ++id) {
        if (slots[id].destroyed_on != who) {
            return testing::AssertionFailure() << "object " << id << " destroyed on " << slots[id].destroyed_on
                                               << ", expected " << who;
        }
    } // loop over the test's objects
    return testing::AssertionSuccess();
} // each_destroyed_once_on()

} // namespace census

// A copyable and movable value whose every instance is a census object. A
// move leaves the source with value() == kMovedFrom, so a test can tell a
// moved-from argument from an untouched one.
class Tracked {
public:
    static constexpr int kMovedFrom = -1;

    explicit Tracked(int value = 0) : id_(census::enroll()), value_(value) {}
    Tracked(const Tracked& other) : id_(census::enroll()), value_(other.value_) {}
    Tracked(Tracked&& other) noexcept : id_(census::enroll()), value_(std::exchange(other.value_, kMovedFrom)) {}
    Tracked& operator=(const Tracked&) = delete;
    Tracked& operator=(Tracked&&) = delete;
    ~Tracked() { census::retire(id_); }

    int id() const { return id_; }
    int value() const { return value_; }
    bool moved_from() const { return value_ == kMovedFrom; }

private:
    int id_;     // census id of this instance
    int value_;  // the value carried; kMovedFrom after a move out of it
}; // class Tracked

// What a callable observed, written by the new thread and read by the test
// after join() (plain fields: the read is ordered by join()).
struct Report {
    std::thread::id worker;            // std::this_thread::get_id() inside the callable
    pthread_t self = {};               // pthread_self() inside the callable
    int fn_id = -1;                    // census id of the callable object invoked
    int arg_id = -1;                   // census id of the argument object it received
    int arg_value = 0;                 // value of that argument
    std::thread::id fn_made_on;        // the thread that constructed the callable object
    std::thread::id arg_made_on;       // the thread that constructed the argument object
    bool returned = false;             // set by the callable as its last action
    bool arg_saw_return = false;       // set by ReturnWitness's destructor: `returned` was set then
}; // struct Report

// A move-only argument whose destructor records whether the callable had
// already returned when the copy was destroyed. Only the instance that still
// points at a Report records (a moved-from one has none).
class ReturnWitness {
public:
    explicit ReturnWitness(Report* report) : report_(report) {}
    ReturnWitness(ReturnWitness&& other) noexcept : report_(std::exchange(other.report_, nullptr)) {}
    ReturnWitness(const ReturnWitness&) = delete;
    ReturnWitness& operator=(const ReturnWitness&) = delete;
    ReturnWitness& operator=(ReturnWitness&&) = delete;
    ~ReturnWitness() {
        if (report_ != nullptr) report_->arg_saw_return = report_->returned;
    } // ReturnWitness::~ReturnWitness()

private:
    Report* report_;  // where to record; null once moved from
}; // class ReturnWitness

// A move-only callable that is invocable only as an rvalue (as the contract's
// std::invoke(std::move(f'), ...) requires) and records into a Report what it
// was invoked on: its own census id (so the object invoked is f'), the
// argument's id (the parameter is an rvalue reference, so it binds to args'
// itself), and who constructed both.
struct RecordingFn {
    std::unique_ptr<Report> unused_box;  // makes the callable move-only
    Tracked tag;                         // census identity of this callable object
    Report* report;                      // where to record

    void operator()(Tracked&& arg, ReturnWitness&&) && {
        report->worker = std::this_thread::get_id();
        report->self = pthread_self();
        report->fn_id = tag.id();
        report->arg_id = arg.id();
        report->arg_value = arg.value();
        report->fn_made_on = census::slots[tag.id()].made_on;
        report->arg_made_on = census::slots[arg.id()].made_on;
        report->returned = true;
    } // RecordingFn::operator()()
}; // struct RecordingFn

// A callable for the failure tests: invocable with any arguments, it only
// sets a flag. It carries a census object and a move-only member, so a test
// can see whether it was copied, moved from, or left untouched.
struct FlagFn {
    std::unique_ptr<int> box = std::make_unique<int>(7);  // null once moved from
    Tracked tag{1};                                         // census identity
    std::atomic<bool>* invoked = nullptr;                   // set when the callable runs

    template <class... A>
    void operator()(A&&...) && {
        invoked->store(true);
    } // FlagFn::operator()()
}; // struct FlagFn

// Thrown by Thrower's copy or move constructor.
struct CopyError : std::exception {};

// An argument whose copy constructor (kOnMove false) or move constructor
// (kOnMove true) throws CopyError; the other one succeeds.
template <bool kOnMove>
class Thrower {
public:
    Thrower() = default;
    Thrower(const Thrower&) {
        if (!kOnMove) throw CopyError{};
    } // Thrower::Thrower(const Thrower&)
    Thrower(Thrower&&) {
        if (kOnMove) throw CopyError{};
    } // Thrower::Thrower(Thrower&&)
    Thrower& operator=(const Thrower&) = delete;
    Thrower& operator=(Thrower&&) = delete;
}; // class Thrower

// The result of a call that may throw std::system_error.
struct Caught {
    bool threw = false;     // a std::system_error was thrown
    std::error_code code;   // its code
    std::string what;       // its what()
}; // struct Caught

// Calls fn() and reports the std::system_error it threw, if any.
template <class Fn>
Caught catch_system_error(Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::system_error& e) {
        return Caught{true, e.code(), e.what()};
    }
    return Caught{};
} // catch_system_error()

// Success if `c` is a std::system_error with the generic category and code e.
testing::AssertionResult is_error(const Caught& c, std::errc e) {
    if (!c.threw) return testing::AssertionFailure() << "no std::system_error thrown";
    if (c.code.category() != std::generic_category()) {
        return testing::AssertionFailure() << "category " << c.code.category().name() << ", expected generic";
    }
    if (c.code != std::make_error_code(e)) {
        return testing::AssertionFailure() << "code " << c.code.value() << " (" << c.code.message() << "), expected "
                                           << std::make_error_code(e).value();
    }
    return testing::AssertionSuccess();
} // is_error()

// Writes `s` to stderr without buffering or a newline: the death tests'
// markers, written from a terminate handler or a destructor that ends with
// _Exit, where stdio buffers would be lost.
void marker(const char* s) {
    const ssize_t n = ::write(2, s, std::strlen(s));
    static_cast<void>(n);
} // marker()

// A one-shot gate: open() releases every wait(). Used instead of std::latch,
// whose wait() can complete inside the (uninstrumented) standard library,
// where ThreadSanitizer does not see the acquire, so the data handed over
// through it would be reported as a race. Spinning on an inline atomic load
// keeps the happens-before edge visible.
class Gate {
public:
    void open() { open_.store(true, std::memory_order_release); }
    void wait() const {
        while (!open_.load(std::memory_order_acquire)) std::this_thread::yield();
    } // Gate::wait()

private:
    std::atomic<bool> open_{false};  // set once by open()
}; // class Gate

// ---------------------------------------------------------------------------
// Signal masks.

// The seven signals the start mask leaves unblocked.
constexpr std::array<int, 7> kSeven = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGPROF, SIGTRAP, SIGSYS};

bool is_one_of_seven(int s) {
    for (int u : kSeven) {
        if (u == s) return true;
    }
    return false;
} // is_one_of_seven()

// The contract's start mask, signal by signal: every blockable signal is
// blocked except the seven; SIGKILL, SIGSTOP and glibc's two internal signals
// are never blocked. glibc's internal signals are the ones between the
// kernel's first real-time signal (__SIGRTMIN) and the first one it gives to
// applications (SIGRTMIN).
bool start_mask_blocks(int s) {
    if (is_one_of_seven(s)) return false;
    if (s == SIGKILL || s == SIGSTOP) return false;
    if (s >= __SIGRTMIN && s < SIGRTMIN) return false;
    return true;
} // start_mask_blocks()

// The calling thread's current signal mask.
sigset_t current_mask() {
    sigset_t m;
    sigemptyset(&m);
    pthread_sigmask(SIG_BLOCK, nullptr, &m);
    return m;
} // current_mask()

// Empty if `m` blocks exactly the signals start_mask_blocks() names (checked
// over 1..SIGRTMAX); otherwise the signals that differ.
std::string start_mask_mismatch(const sigset_t& m) {
    std::string out;
    for (int s = 1; s <= SIGRTMAX; ++s) {
        const bool blocked = sigismember(&m, s) == 1;
        if (blocked != start_mask_blocks(s)) {
            out += std::to_string(s) + (blocked ? " blocked; " : " unblocked; ");
        }
    } // loop over all signals
    return out;
} // start_mask_mismatch()

// Empty if `a` and `b` block the same signals over 1..SIGRTMAX; otherwise the
// signals that differ.
std::string mask_difference(const sigset_t& a, const sigset_t& b) {
    std::string out;
    for (int s = 1; s <= SIGRTMAX; ++s) {
        if ((sigismember(&a, s) == 1) != (sigismember(&b, s) == 1)) out += std::to_string(s) + " ";
    } // loop over all signals
    return out;
} // mask_difference()

// The creator masks the mask tests use: none blocked, everything blocked,
// exactly the seven blocked (the case where the new thread must unblock what
// its creator blocked), and a mix of some of the seven and other signals.
enum class CreatorMask { kEmpty, kFull, kSeven, kMixed };

constexpr std::array<CreatorMask, 4> kCreatorMasks = {CreatorMask::kEmpty, CreatorMask::kFull, CreatorMask::kSeven,
                                                      CreatorMask::kMixed};

const char* name_of(CreatorMask k) {
    switch (k) {
    case CreatorMask::kEmpty: return "empty";
    case CreatorMask::kFull: return "full";
    case CreatorMask::kSeven: return "the seven";
    case CreatorMask::kMixed: return "mixed";
    }
    return "?";
} // name_of()

sigset_t make_mask(CreatorMask k) {
    sigset_t m;
    sigemptyset(&m);
    switch (k) {
    case CreatorMask::kEmpty: break;
    case CreatorMask::kFull:
        // Everything blockable: SIGKILL and SIGSTOP cannot be blocked and
        // are reported unblocked, so they are left out of the request too.
        sigfillset(&m);
        sigdelset(&m, SIGKILL);
        sigdelset(&m, SIGSTOP);
        break;
    case CreatorMask::kSeven:
        for (int s : kSeven) sigaddset(&m, s);
        break;
    case CreatorMask::kMixed:
        sigaddset(&m, SIGSEGV);
        sigaddset(&m, SIGPROF);
        sigaddset(&m, SIGUSR1);
        sigaddset(&m, SIGRTMIN + 3);
        break;
    } // switch over the creator masks
    return m;
} // make_mask()

// Calls fn() in a new std::thread whose mask is first set to make_mask(k),
// and waits for it. The GoogleTest main thread's mask is never changed.
template <class Fn>
void on_creator(CreatorMask k, Fn&& fn) {
    std::thread creator([k, &fn] {
        const sigset_t m = make_mask(k);
        pthread_sigmask(SIG_SETMASK, &m, nullptr);
        fn();
    });
    creator.join();
} // on_creator()

// The callable of the mask tests: its first action reads its own mask.
[[gnu::noinline]] void read_own_mask(sigset_t* out) {
    pthread_sigmask(SIG_BLOCK, nullptr, out);
} // read_own_mask()

// Installs a counting handler for one signal and restores the previous
// disposition on destruction. The handler counts deliveries and records the
// receiving thread.
class CountingHandler {
public:
    explicit CountingHandler(int sig) : sig_(sig) {
        counts()[sig].store(0);
        receivers()[sig].store(pthread_t{});
        struct sigaction sa {};
        sa.sa_handler = &CountingHandler::handle;
        sigemptyset(&sa.sa_mask);
        sigaction(sig, &sa, &previous_);
    } // CountingHandler::CountingHandler()
    CountingHandler(const CountingHandler&) = delete;
    CountingHandler& operator=(const CountingHandler&) = delete;
    ~CountingHandler() { sigaction(sig_, &previous_, nullptr); }

    int count() const { return counts()[sig_].load(); }
    pthread_t receiver() const { return receivers()[sig_].load(); }

private:
    static constexpr int kSignals = 65;  // signal numbers 1..64

    static std::array<std::atomic<int>, kSignals>& counts() {
        static std::array<std::atomic<int>, kSignals> c{};
        return c;
    } // CountingHandler::counts()
    static std::array<std::atomic<pthread_t>, kSignals>& receivers() {
        static std::array<std::atomic<pthread_t>, kSignals> r{};
        return r;
    } // CountingHandler::receivers()
    static void handle(int sig) {
        counts()[sig].fetch_add(1);
        receivers()[sig].store(pthread_self());
    } // CountingHandler::handle()

    int sig_;                       // the signal handled
    struct sigaction previous_ {};  // the disposition to restore
}; // class CountingHandler

// ---------------------------------------------------------------------------
// Stacks and guards.

std::size_t page_size() { return static_cast<std::size_t>(sysconf(_SC_PAGESIZE)); }

std::size_t page_round(std::size_t n) { return (n + page_size() - 1)/page_size()*page_size(); }

// The system minimum stack size the contract raises smaller requests to.
std::size_t minimum_stack() { return static_cast<std::size_t>(sysconf(_SC_THREAD_STACK_MIN)); }

// Bytes between the frame address and the canonical frame address (where the
// caller's frame ends and this function's frame begins). On x86-64 with a
// frame pointer (which __builtin_frame_address(0) forces) the frame address
// points at the saved frame pointer, with the return address above it. On
// other targets 0: the bound is then measured from a point inside the frame,
// which is stricter.
#if defined(__x86_64__)
constexpr std::uintptr_t kFrameRecord = 2*sizeof(void*);
#else
constexpr std::uintptr_t kFrameRecord = 0;
#endif

// The stack as seen from the callable's first frame.
struct StackView {
    std::uintptr_t lo = 0;       // lowest address of the stack (pthread_getattr_np)
    std::size_t size = 0;        // stack size reported by pthread_getattr_np
    std::size_t guard = 0;       // guard size reported by pthread_getattr_np
    std::uintptr_t frame = 0;    // where the callable's first frame begins
    std::uintptr_t tls = 0;      // address of this thread's big_tls
}; // struct StackView

// The callable of the stack tests, called through a function pointer and
// never inlined, so its frame is the callable's first frame. Its frame address
// is a real stack address in every build (AddressSanitizer moves only
// address-taken locals to its fake stack).
[[gnu::noinline]] void view_stack(StackView* out) {
    out->frame = reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0)) + kFrameRecord;
    pthread_attr_t attr;
    pthread_getattr_np(pthread_self(), &attr);
    void* addr = nullptr;
    pthread_attr_getstack(&attr, &addr, &out->size);
    pthread_attr_getguardsize(&attr, &out->guard);
    pthread_attr_destroy(&attr);
    out->lo = reinterpret_cast<std::uintptr_t>(addr);
    out->tls = reinterpret_cast<std::uintptr_t>(big_tls.data());
} // view_stack()

// Bytes of stack below the callable's first frame.
std::size_t below_frame(const StackView& v) { return v.frame - v.lo; }

// The inaccessible region directly below `lo`: the bytes of the run of
// contiguous "---p" mappings of this process that ends at `lo` (0 if the
// mapping below `lo` is not inaccessible). A run, not one mapping: when glibc
// enlarges the guard of a stack it reuses from its cache, the old and the new
// inaccessible parts can remain separate adjacent mappings. An unrelated
// inaccessible mapping adjacent to the guard would be counted too, which can
// only make the check more lenient, never fail it.
std::size_t inaccessible_below(std::uintptr_t lo) {
    struct Range {
        std::uintptr_t begin;  // first address
        std::uintptr_t end;    // one past the last address
        bool inaccessible;     // permissions are "---p"
    }; // struct Range
    std::vector<Range> ranges;
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (f == nullptr) return 0;
    char* line = nullptr;
    std::size_t capacity = 0;
    while (getline(&line, &capacity, f) != -1) {
        unsigned long b = 0;
        unsigned long e = 0;
        char perms[5] = {};
        if (std::sscanf(line, "%lx-%lx %4s", &b, &e, perms) == 3) {
            ranges.push_back(Range{b, e, std::strcmp(perms, "---p") == 0});
        }
    } // loop over the mappings
    std::free(line);
    std::fclose(f);
    std::uintptr_t edge = lo;
    for (bool extended = true; extended;) {
        extended = false;
        for (const Range& r : ranges) {
            if (r.end == edge && r.inaccessible && r.begin < edge) {
                edge = r.begin;
                extended = true;
                break;
            }
        } // loop over the mappings
    } // loop extending the run downwards
    return lo - edge;
} // inaccessible_below()

// The guard as seen from the callable, while its stack is live.
struct GuardView {
    StackView stack;              // the stack's bounds and reported sizes
    std::size_t inaccessible = 0; // bytes of inaccessible memory directly below the stack
}; // struct GuardView

// The callable of the guard tests.
[[gnu::noinline]] void view_guard(GuardView* out) {
    view_stack(&out->stack);
    out->inaccessible = inaccessible_below(out->stack.lo);
} // view_guard()

// Success if the inaccessible region directly below the stack seen in `v`
// is at least `requested` rounded up to a page, and so is the reported guard
// size.
testing::AssertionResult guard_at_least(const GuardView& v, std::size_t requested) {
    const std::size_t want = page_round(requested);
    if (v.inaccessible < want) {
        return testing::AssertionFailure() << "inaccessible below the stack " << v.inaccessible << " < " << want;
    }
    if (v.stack.guard < want) return testing::AssertionFailure() << "reported guard " << v.stack.guard << " < " << want;
    return testing::AssertionSuccess();
} // guard_at_least()

// Sets the process default stack size and restores the previous one.
class DefaultStackScope {
public:
    explicit DefaultStackScope(std::size_t bytes) : previous_(Thread::set_default_stack_size(bytes)) {}
    DefaultStackScope(const DefaultStackScope&) = delete;
    DefaultStackScope& operator=(const DefaultStackScope&) = delete;
    ~DefaultStackScope() { Thread::set_default_stack_size(previous_); }

private:
    std::size_t previous_;  // the default to restore
}; // class DefaultStackScope

// Sets an environment variable and restores its previous state.
class EnvScope {
public:
    EnvScope(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        if (old != nullptr) previous_ = old;
        setenv(name, value, 1);
    } // EnvScope::EnvScope()
    EnvScope(const EnvScope&) = delete;
    EnvScope& operator=(const EnvScope&) = delete;
    ~EnvScope() {
        if (previous_) {
            setenv(name_, previous_->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    } // EnvScope::~EnvScope()

private:
    const char* name_;                     // the variable
    std::optional<std::string> previous_;  // its previous value, if it had one
}; // class EnvScope

// The heap counter of the sanitizer runtime, or null in the optimized build.
using AllocatedBytesFn = std::size_t (*)();
AllocatedBytesFn allocated_bytes() { return &__sanitizer_get_current_allocated_bytes; }

// Heap bytes still allocated after one more call of fn(), measured after a
// first call has warmed up any lazily allocated state. 0 means no leak.
template <class Fn>
long long heap_delta(Fn&& fn) {
    const AllocatedBytesFn bytes = allocated_bytes();
    fn();
    const std::size_t before = bytes();
    fn();
    const std::size_t after = bytes();
    return static_cast<long long>(after) - static_cast<long long>(before);
} // heap_delta()

// ===========================================================================
// The type: special members, noexcept, constraints, Attributes.

// Move-only (copying deleted), and the members the contract calls nothrow
// are noexcept.
TEST(ThreadTypeTest, MoveOnlyAndNothrowAsDeclared) {
    static_assert(!std::is_copy_constructible_v<Thread>);
    static_assert(!std::is_copy_assignable_v<Thread>);
    static_assert(std::is_nothrow_default_constructible_v<Thread>);
    static_assert(std::is_nothrow_move_constructible_v<Thread>);
    static_assert(std::is_nothrow_move_assignable_v<Thread>);
    static_assert(std::is_nothrow_destructible_v<Thread>);
    static_assert(std::is_nothrow_swappable_v<Thread>);
    static_assert(std::is_same_v<Thread::native_handle_type, pthread_t>);
    Thread t;
    static_assert(noexcept(t.joinable()));
    static_assert(noexcept(t.native_handle()));
    static_assert(noexcept(t.get_id()));
    static_assert(noexcept(t.swap(t)));
    static_assert(noexcept(Thread::hardware_concurrency()));
    static_assert(noexcept(Thread::set_default_stack_size(0)));
    static_assert(noexcept(Thread::default_stack_size()));
    SUCCEED();
} // TEST(ThreadTypeTest, MoveOnlyAndNothrowAsDeclared)

// The starting constructors are explicit, and the constraint keeps them from
// taking a Thread or an Attributes as the callable: without it, a Thread
// lvalue or an Attributes alone would bind F&& and the traits below would
// report the class constructible from them (traits look only at the
// declaration).
TEST(ThreadTypeTest, StartingConstructorsExplicitAndConstrained) {
    using Fp = void (*)();
    static_assert(std::is_constructible_v<Thread, Fp>);
    static_assert(!std::is_convertible_v<Fp, Thread>);
    static_assert(std::is_constructible_v<Thread, const Thread::Attributes&, Fp>);
    static_assert(!std::is_constructible_v<Thread, Thread&>);
    static_assert(!std::is_constructible_v<Thread, const Thread&>);
    static_assert(!std::is_constructible_v<Thread, Thread::Attributes&>);
    static_assert(!std::is_constructible_v<Thread, const Thread::Attributes&>);
    static_assert(!std::is_constructible_v<Thread, Thread::Attributes>);
    SUCCEED();
} // TEST(ThreadTypeTest, StartingConstructorsExplicitAndConstrained)

// Attributes is an aggregate of two optional sizes, empty by default (empty
// means "take the default").
TEST(ThreadTypeTest, AttributesAggregateEmptyByDefault) {
    static_assert(std::is_aggregate_v<Thread::Attributes>);
    static_assert(std::is_same_v<decltype(Thread::Attributes::stack_size), std::optional<std::size_t>>);
    static_assert(std::is_same_v<decltype(Thread::Attributes::guard_size), std::optional<std::size_t>>);
    const Thread::Attributes a{};
    EXPECT_FALSE(a.stack_size.has_value());
    EXPECT_FALSE(a.guard_size.has_value());
    const Thread::Attributes b{.guard_size = 0};
    EXPECT_FALSE(b.stack_size.has_value());
    ASSERT_TRUE(b.guard_size.has_value());
    EXPECT_EQ(*b.guard_size, 0u);
} // TEST(ThreadTypeTest, AttributesAggregateEmptyByDefault)

// A non-const lvalue Attributes selects the attributes constructor (if it
// bound F&&, the Mandates static_assert would reject an Attributes as the
// callable and this would not compile), and its sizes are honoured: the
// stack request is larger than anything the default could give.
TEST(ThreadTypeTest, NonConstAttributesLvalueSelectsAttributesConstructor) {
    Thread::Attributes a{.stack_size = 5*kMiB + 1};
    StackView v;
    Thread t(a, &view_stack, &v);
    t.join();
    EXPECT_GE(below_frame(v), 5*kMiB + 1);
} // TEST(ThreadTypeTest, NonConstAttributesLvalueSelectsAttributesConstructor)

// hardware_concurrency() is std::thread::hardware_concurrency().
TEST(ThreadTypeTest, HardwareConcurrencyIsStdThreads) {
    EXPECT_EQ(Thread::hardware_concurrency(), std::thread::hardware_concurrency());
} // TEST(ThreadTypeTest, HardwareConcurrencyIsStdThreads)

// ===========================================================================
// A Thread that owns no thread.

// Default construction: not joinable, id{}, a zero handle.
TEST(ThreadEmptyTest, DefaultConstructedOwnsNothing) {
    const Thread t;
    EXPECT_FALSE(t.joinable());
    EXPECT_EQ(t.get_id(), std::thread::id{});
    EXPECT_EQ(t.native_handle(), Thread::native_handle_type{});
} // TEST(ThreadEmptyTest, DefaultConstructedOwnsNothing)

// join() and detach() on a Thread that is not joinable throw invalid_argument
// (generic category) and, by the strong guarantee, leave it unchanged.
TEST(ThreadEmptyTest, JoiningOrDetachingNothingThrowsInvalidArgument) {
    Thread t;
    EXPECT_TRUE(is_error(catch_system_error([&] { t.join(); }), std::errc::invalid_argument));
    EXPECT_FALSE(t.joinable());
    EXPECT_EQ(t.get_id(), std::thread::id{});
    EXPECT_TRUE(is_error(catch_system_error([&] { t.detach(); }), std::errc::invalid_argument));
    EXPECT_FALSE(t.joinable());
    EXPECT_EQ(t.native_handle(), Thread::native_handle_type{});
} // TEST(ThreadEmptyTest, JoiningOrDetachingNothingThrowsInvalidArgument)

// ===========================================================================
// Invocation: decay-copies, std::invoke, and the copies' lifetime.

// The central promise, with a move-only callable invocable only as an rvalue
// and move-only arguments: the callable runs on the decay-copies f' and args'
// (the object invoked and the argument it receives are copies, not the
// originals), both made in the creating thread and complete before the
// invocation (a plain field read in the callable; a race under
// ThreadSanitizer otherwise); the copies are destroyed exactly once, by the
// new thread, after the callable returns (ReturnWitness) and before join()
// returns (plain fields read after join()). Every other object the
// construction made (temporaries, the moved-from originals) is destroyed
// exactly once as well.
TEST(ThreadInvokeTest, CallableRunsOnDecayCopiesDestroyedByTheNewThread) {
    const int first = census::mark();
    Report report;
    int original_fn_id = -1;
    int original_arg_id = -1;
    {
        RecordingFn fn{std::make_unique<Report>(), Tracked{}, &report};
        Tracked arg{42};
        original_fn_id = fn.tag.id();
        original_arg_id = arg.id();
        Thread t(std::move(fn), std::move(arg), ReturnWitness(&report));
        t.join();
        EXPECT_EQ(report.arg_value, 42);
        EXPECT_NE(report.fn_id, original_fn_id);
        EXPECT_NE(report.arg_id, original_arg_id);
        EXPECT_NE(report.worker, std::this_thread::get_id());
        EXPECT_EQ(report.fn_made_on, std::this_thread::get_id());
        EXPECT_EQ(report.arg_made_on, std::this_thread::get_id());
        EXPECT_TRUE(report.returned);
        EXPECT_TRUE(report.arg_saw_return);
        // The copies: destroyed once, on the new thread, before join() returned.
        EXPECT_EQ(census::destructions(report.fn_id), 1);
        EXPECT_EQ(census::destructions(report.arg_id), 1);
        EXPECT_EQ(census::slots[report.fn_id].destroyed_on, report.worker);
        EXPECT_EQ(census::slots[report.arg_id].destroyed_on, report.worker);
    } // scope of the originals
    EXPECT_TRUE(census::each_destroyed_once(first));
} // TEST(ThreadInvokeTest, CallableRunsOnDecayCopiesDestroyedByTheNewThread)

// Lvalue arguments are copied, not referenced: the callable changes its copy,
// the original keeps its value; the copy is destroyed exactly once.
TEST(ThreadInvokeTest, LvalueArgumentsAreCopied) {
    const int first = census::mark();
    {
        Tracked original{5};
        int seen = 0;
        int copy_id = -1;
        Thread t([&seen, &copy_id](Tracked&& copy) {
            seen = copy.value();
            copy_id = copy.id();
        }, original);
        t.join();
        EXPECT_EQ(seen, 5);
        EXPECT_NE(copy_id, original.id());
        EXPECT_FALSE(original.moved_from());
        EXPECT_EQ(original.value(), 5);
    } // scope of the original
    EXPECT_TRUE(census::each_destroyed_once(first));
} // TEST(ThreadInvokeTest, LvalueArgumentsAreCopied)

// std::invoke semantics: a member function pointer with an object pointer, a
// std::reference_wrapper argument (its decay-copy still refers to the
// original), an array argument (decays to a pointer), and a callable whose
// return value is discarded.
TEST(ThreadInvokeTest, InvokeSemantics) {
    struct Widget {
        int value = 0;
        void set(int v) { value = v; }
    }; // struct Widget
    Widget w;
    Thread t1(&Widget::set, &w, 7);
    t1.join();
    EXPECT_EQ(w.value, 7);

    int counter = 0;
    Thread t2([](int& c) { ++c; }, std::ref(counter));
    t2.join();
    EXPECT_EQ(counter, 1);

    static const char kText[] = "text";
    const char* seen = nullptr;
    Thread t3([&seen](const char* s) { seen = s; }, kText);
    t3.join();
    EXPECT_EQ(seen, kText);

    Thread t4([] { return 17; });
    t4.join();
    SUCCEED();
} // TEST(ThreadInvokeTest, InvokeSemantics)

// A plain function pointer as the callable, with no arguments.
TEST(ThreadInvokeTest, FunctionPointerWithoutArguments) {
    static std::atomic<int> calls{0};
    calls.store(0);
    Thread t(+[] { calls.fetch_add(1); });
    t.join();
    EXPECT_EQ(calls.load(), 1);
} // TEST(ThreadInvokeTest, FunctionPointerWithoutArguments)

// Every copy is destroyed exactly once over many threads, alive at the same
// time and joined in reverse order.
TEST(ThreadInvokeTest, ManyThreadsEachCopyDestroyedOnce) {
    const int first = census::mark();
    {
        std::vector<Thread> threads;
        for (int i = 0; i < 16; ++i) {
            threads.emplace_back([](Tracked&&, Tracked&&) {}, Tracked{i}, Tracked{-i});
        } // loop creating the threads
        while (!threads.empty()) threads.pop_back();
    } // scope of the threads
    EXPECT_TRUE(census::each_destroyed_once(first));
} // TEST(ThreadInvokeTest, ManyThreadsEachCopyDestroyedOnce)

// ===========================================================================
// Lifetime: join, detach, destructor, moves, ids, handles.

// Postconditions of a starting constructor: joinable, get_id() is the new
// thread's id (equal to std::this_thread::get_id() inside it), and
// native_handle() is its pthread_t. The thread stays joinable after its
// callable has finished, until join().
TEST(ThreadLifetimeTest, ConstructorPostconditionsAndIdentity) {
    Report report;
    std::atomic<bool> done{false};
    Thread t(
        [&report, &done] {
            report.worker = std::this_thread::get_id();
            report.self = pthread_self();
            done.store(true);
        });
    EXPECT_TRUE(t.joinable());
    const std::thread::id id = t.get_id();
    const pthread_t handle = t.native_handle();
    EXPECT_NE(id, std::thread::id{});
    EXPECT_NE(id, std::this_thread::get_id());
    while (!done.load()) std::this_thread::yield();
    EXPECT_TRUE(t.joinable());
    t.join();
    EXPECT_FALSE(t.joinable());
    EXPECT_EQ(id, report.worker);
    EXPECT_NE(pthread_equal(handle, report.self), 0);
    EXPECT_EQ(t.get_id(), std::thread::id{});
    EXPECT_EQ(t.native_handle(), Thread::native_handle_type{});
} // TEST(ThreadLifetimeTest, ConstructorPostconditionsAndIdentity)

// join() waits for the callable: the callable writes a plain field after a
// delay; after join() it is set (and ThreadSanitizer would report the read as
// a race without the happens-before join() promises). A second join() throws
// invalid_argument.
TEST(ThreadLifetimeTest, JoiningWaitsAndHappensBefore) {
    int result = 0;
    Thread t([&result] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        result = 1;
    });
    t.join();
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(is_error(catch_system_error([&] { t.join(); }), std::errc::invalid_argument));
} // TEST(ThreadLifetimeTest, JoiningWaitsAndHappensBefore)

// The destructor joins: the same oracle as above, with the Thread going out
// of scope instead of join(); its copies are destroyed by then too.
TEST(ThreadLifetimeTest, DestructorJoins) {
    const int first = census::mark();
    int result = 0;
    {
        Thread t(
            [&result](Tracked&&) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                result = 1;
            },
            Tracked{});
    } // scope of the Thread
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(census::each_destroyed_once(first));
} // TEST(ThreadLifetimeTest, DestructorJoins)

// join() on the thread *this owns throws resource_deadlock_would_occur and
// leaves *this unchanged (still joinable, same id). The callable gets its own
// Thread* through a Gate opened after the constructor returns, as the
// contract requires; a second Gate keeps the creator's join() from running
// while the callable still uses the object (calls on one object from two
// threads are a data race unless all are const). Inside the thread, get_id()
// and native_handle() name the thread itself.
TEST(ThreadLifetimeTest, SelfJoiningThrowsDeadlockAndLeavesThreadUnchanged) {
    struct Shared {
        Gate handed_over;
        Gate used;
        Thread* self = nullptr;
        Caught caught;
        bool still_joinable = false;
        bool id_is_own = false;
        bool handle_is_own = false;
    }; // struct Shared
    Shared s;
    Thread t([&s] {
        s.handed_over.wait();
        s.caught = catch_system_error([&s] { s.self->join(); });
        s.still_joinable = s.self->joinable();
        s.id_is_own = s.self->get_id() == std::this_thread::get_id();
        s.handle_is_own = pthread_equal(s.self->native_handle(), pthread_self()) != 0;
        s.used.open();
    });
    s.self = &t;
    s.handed_over.open();
    s.used.wait();
    t.join();
    EXPECT_TRUE(is_error(s.caught, std::errc::resource_deadlock_would_occur));
    EXPECT_TRUE(s.still_joinable);
    EXPECT_TRUE(s.id_is_own);
    EXPECT_TRUE(s.handle_is_own);
} // TEST(ThreadLifetimeTest, SelfJoiningThrowsDeadlockAndLeavesThreadUnchanged)

// detach(): not joinable afterwards (id{}, zero handle), the thread goes on
// and finishes on its own, destroying its copies; a second detach() and a
// join() throw invalid_argument. The test waits until the copy's destructor
// has run, so the detached thread has finished its work before the test
// returns.
TEST(ThreadLifetimeTest, DetachReleasesOwnership) {
    struct Done {
        std::atomic<bool>* flag;
        explicit Done(std::atomic<bool>* f) : flag(f) {}
        Done(Done&& other) noexcept : flag(std::exchange(other.flag, nullptr)) {}
        Done(const Done&) = delete;
        Done& operator=(const Done&) = delete;
        Done& operator=(Done&&) = delete;
        ~Done() {
            if (flag != nullptr) {
                flag->store(true, std::memory_order_release);
            }
        } // Done::~Done()
    }; // struct Done
    std::atomic<bool> copies_gone{false};
    Gate go;
    Thread t([&go](Done&&) { go.wait(); }, Done(&copies_gone));
    t.detach();
    EXPECT_FALSE(t.joinable());
    EXPECT_EQ(t.get_id(), std::thread::id{});
    EXPECT_EQ(t.native_handle(), Thread::native_handle_type{});
    EXPECT_TRUE(is_error(catch_system_error([&] { t.detach(); }), std::errc::invalid_argument));
    EXPECT_TRUE(is_error(catch_system_error([&] { t.join(); }), std::errc::invalid_argument));
    go.open();
    // Bounded, so that a thread that never destroys its copies fails the test
    // instead of hanging it.
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!copies_gone.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(copies_gone.load()) << "the detached thread did not destroy its copies";
} // TEST(ThreadLifetimeTest, DetachReleasesOwnership)

// Move construction takes the thread: the target has the source's id, the
// source owns nothing; the target joins it.
TEST(ThreadLifetimeTest, MoveConstructionTransfersOwnership) {
    Gate go;
    Thread a([&go] { go.wait(); });
    const std::thread::id id = a.get_id();
    Thread b(std::move(a));
    EXPECT_FALSE(a.joinable());
    EXPECT_EQ(a.get_id(), std::thread::id{});
    EXPECT_TRUE(b.joinable());
    EXPECT_EQ(b.get_id(), id);
    go.open();
    b.join();
    EXPECT_FALSE(b.joinable());
} // TEST(ThreadLifetimeTest, MoveConstructionTransfersOwnership)

// Move assignment onto a joinable Thread joins its thread first: the target's
// thread writes a plain field after a delay, and it is set when the
// assignment returns (and its copies are destroyed). The target then owns the
// source's thread; the source owns nothing.
TEST(ThreadLifetimeTest, MoveAssignmentJoinsTheTargetFirst) {
    const int first = census::mark();
    int target_done = 0;
    Gate go;
    Thread target(
        [&target_done](Tracked&&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            target_done = 1;
        },
        Tracked{});
    Thread source([&go] { go.wait(); });
    const std::thread::id source_id = source.get_id();
    target = std::move(source);
    EXPECT_EQ(target_done, 1);
    EXPECT_TRUE(census::each_destroyed_once(first));
    EXPECT_FALSE(source.joinable());
    EXPECT_EQ(target.get_id(), source_id);
    go.open();
    target.join();
} // TEST(ThreadLifetimeTest, MoveAssignmentJoinsTheTargetFirst)

// Move assignment onto an empty Thread, and from an empty Thread onto an
// empty one.
TEST(ThreadLifetimeTest, MoveAssignmentOntoEmpty) {
    Thread target;
    Thread source([] {});
    const std::thread::id id = source.get_id();
    target = std::move(source);
    EXPECT_EQ(target.get_id(), id);
    EXPECT_FALSE(source.joinable());
    Thread empty;
    source = std::move(empty);
    EXPECT_FALSE(source.joinable());
    EXPECT_FALSE(empty.joinable());
    target.join();
} // TEST(ThreadLifetimeTest, MoveAssignmentOntoEmpty)

// Self-move-assignment does nothing: the Thread still owns its thread. The
// self-move goes through a reference so that the compiler sees no self-move.
TEST(ThreadLifetimeTest, SelfMoveAssignmentDoesNothing) {
    Gate go;
    Thread t([&go] { go.wait(); });
    const std::thread::id id = t.get_id();
    Thread& alias = t;
    t = std::move(alias);
    EXPECT_TRUE(t.joinable());
    EXPECT_EQ(t.get_id(), id);
    go.open();
    t.join();
} // TEST(ThreadLifetimeTest, SelfMoveAssignmentDoesNothing)

// swap (member and the non-member found by argument-dependent lookup)
// exchanges the owned threads, including with an empty Thread.
TEST(ThreadLifetimeTest, SwapExchangesThreads) {
    Gate go;
    Thread a([&go] { go.wait(); });
    Thread b([&go] { go.wait(); });
    const std::thread::id ida = a.get_id();
    const std::thread::id idb = b.get_id();
    a.swap(b);
    EXPECT_EQ(a.get_id(), idb);
    EXPECT_EQ(b.get_id(), ida);
    using std::swap;
    swap(a, b);
    EXPECT_EQ(a.get_id(), ida);
    EXPECT_EQ(b.get_id(), idb);
    Thread empty;
    swap(a, empty);
    EXPECT_FALSE(a.joinable());
    EXPECT_EQ(empty.get_id(), ida);
    go.open();
} // TEST(ThreadLifetimeTest, SwapExchangesThreads)

// ===========================================================================
// Failures of the constructor.

// The class's EINVAL (step 1) is thrown before anything is allocated or
// copied: no census object was made, the rvalue callable and arguments are
// not moved from, the callable never runs. Each request below overflows by
// the contract's own description: stack SIZE_MAX (its example); a
// page-aligned stack just below SIZE_MAX (the class adds room for the TLS
// and its frames and rounds up to a page, which must overflow for any
// positive addition); guard SIZE_MAX (its page rounding does not fit); a
// page-aligned guard just below SIZE_MAX (fits alone, not together with any
// stack).
TEST(ThreadFailureTest, OverflowThrowsInvalidArgumentBeforeCopying) {
    const std::size_t page_aligned_max = kSizeMax & ~(page_size() - 1);
    const std::array<Thread::Attributes, 4> requests = {
        Thread::Attributes{.stack_size = kSizeMax},
        Thread::Attributes{.stack_size = page_aligned_max},
        Thread::Attributes{.guard_size = kSizeMax},
        Thread::Attributes{.guard_size = page_aligned_max},
    };
    for (const Thread::Attributes& attrs : requests) {
        SCOPED_TRACE(testing::Message() << "stack " << attrs.stack_size.value_or(0) << " guard "
                                        << attrs.guard_size.value_or(0));
        std::atomic<bool> invoked{false};
        FlagFn fn;
        fn.invoked = &invoked;
        Tracked arg{3};
        std::unique_ptr<int> box = std::make_unique<int>(4);
        const int first = census::mark();
        const Caught c = catch_system_error([&] { Thread t(attrs, std::move(fn), std::move(arg), std::move(box)); });
        EXPECT_TRUE(is_error(c, std::errc::invalid_argument));
        EXPECT_EQ(census::mark(), first) << "an object was copied before the EINVAL";
        EXPECT_NE(fn.box, nullptr);
        EXPECT_FALSE(fn.tag.moved_from());
        EXPECT_FALSE(arg.moved_from());
        EXPECT_NE(box, nullptr);
        EXPECT_FALSE(invoked.load());
    } // loop over the overflowing requests
} // TEST(ThreadFailureTest, OverflowThrowsInvalidArgumentBeforeCopying)

// pthread_create's failure (step 4), forced with the contract's example
// stack_size = 2^62 (EAGAIN): the copies were made (census objects exist) and
// are all destroyed, exactly once and on the calling thread, before the
// exception leaves the constructor (checked inside the handler); the rvalue
// callable and arguments are left moved from; the callable never ran; the
// message names the failing call.
TEST(ThreadFailureTest, PthreadCreateFailureDestroysCopiesBeforeThrowing) {
    std::atomic<bool> invoked{false};
    FlagFn fn;
    fn.invoked = &invoked;
    Tracked arg{3};
    std::unique_ptr<int> box = std::make_unique<int>(4);
    const int first = census::mark();
    bool threw = false;
    try {
        Thread t(Thread::Attributes{.stack_size = std::size_t{1} << 62}, std::move(fn), std::move(arg),
                 std::move(box));
    } catch (const std::system_error& e) {
        threw = true;
        EXPECT_EQ(e.code().category(), std::generic_category());
        EXPECT_EQ(e.code(), std::make_error_code(std::errc::resource_unavailable_try_again));
        EXPECT_NE(std::string(e.what()).find("pthread_create"), std::string::npos) << e.what();
        EXPECT_GT(census::mark(), first) << "no copy was made before pthread_create";
        EXPECT_TRUE(census::each_destroyed_once_on(first, std::this_thread::get_id()));
    }
    EXPECT_TRUE(threw);
    EXPECT_EQ(fn.box, nullptr);
    EXPECT_TRUE(fn.tag.moved_from());
    EXPECT_TRUE(arg.moved_from());
    EXPECT_EQ(box, nullptr);
    EXPECT_FALSE(invoked.load());
} // TEST(ThreadFailureTest, PthreadCreateFailureDestroysCopiesBeforeThrowing)

// A guard that fits in size_t with the stack but is too large for glibc
// fails "as glibc reports it": EAGAIN or EINVAL from pthread_create, so after
// the copies (they were made, and are destroyed on the calling thread).
TEST(ThreadFailureTest, GuardTooLargeForGlibcFailsInPthreadCreate) {
    const int first = census::mark();
    bool threw = false;
    try {
        Thread t(Thread::Attributes{.guard_size = std::size_t{1} << 62}, [](Tracked&&) {}, Tracked{});
    } catch (const std::system_error& e) {
        threw = true;
        EXPECT_EQ(e.code().category(), std::generic_category());
        EXPECT_TRUE(e.code() == std::errc::resource_unavailable_try_again || e.code() == std::errc::invalid_argument)
            << e.code().value();
        EXPECT_NE(std::string(e.what()).find("pthread_create"), std::string::npos) << e.what();
        EXPECT_GT(census::mark(), first + 1) << "no copy was made before pthread_create";
    }
    EXPECT_TRUE(threw);
    EXPECT_TRUE(census::each_destroyed_once(first));
} // TEST(ThreadFailureTest, GuardTooLargeForGlibcFailsInPthreadCreate)

// An exception from a copy constructor (lvalue arguments) or a move
// constructor (rvalue arguments) during step 3 propagates unchanged; no
// thread was created (the callable never runs) and the copies already made
// are destroyed exactly once. The arguments around the throwing one are
// census objects, so whichever of them the class copied before the throw is
// accounted for.
TEST(ThreadFailureTest, ExceptionWhileCopyingPropagatesAndCleansUp) {
    {
        std::atomic<bool> invoked{false};
        const int first = census::mark();
        {
            FlagFn fn;
            fn.invoked = &invoked;
            Tracked a{1};
            Tracked b{2};
            Thrower<false> thrower;
            EXPECT_THROW(Thread(std::move(fn), a, thrower, b), CopyError);
        } // scope of the originals
        EXPECT_FALSE(invoked.load());
        EXPECT_TRUE(census::each_destroyed_once(first));
    } // copy constructor throws
    {
        std::atomic<bool> invoked{false};
        const int first = census::mark();
        {
            FlagFn fn;
            fn.invoked = &invoked;
            Tracked a{1};
            Tracked b{2};
            Thrower<true> thrower;
            EXPECT_THROW(Thread(std::move(fn), std::move(a), std::move(thrower), std::move(b)), CopyError);
        } // scope of the originals
        EXPECT_FALSE(invoked.load());
        EXPECT_TRUE(census::each_destroyed_once(first));
    } // move constructor throws
} // TEST(ThreadFailureTest, ExceptionWhileCopyingPropagatesAndCleansUp)

// No failure path leaks heap (sanitizer builds only: the optimized build has
// no heap counter). After one warm-up, one more failing construction of each
// kind, and one successful construction and join, must leave the heap where
// it was: the copies, the attributes and the exception are all freed.
TEST(ThreadFailureTest, NoPathLeaksHeap) {
    if (allocated_bytes() == nullptr) GTEST_SKIP() << "no sanitizer heap counter in this build";
    const auto overflow = [] {
        try {
            Thread t(Thread::Attributes{.stack_size = kSizeMax}, [](Tracked&&) {}, Tracked{});
        } catch (const std::system_error&) {
        }
    };
    const auto eagain = [] {
        try {
            Thread t(Thread::Attributes{.stack_size = std::size_t{1} << 62}, [](Tracked&&) {}, Tracked{});
        } catch (const std::system_error&) {
        }
    };
    const auto copy_throws = [] {
        try {
            Thrower<false> thrower;
            Thread t([](Tracked&&, Thrower<false>&&) {}, Tracked{}, thrower);
        } catch (const CopyError&) {
        }
    };
    const auto succeeds = [] {
        Thread t([](Tracked&&, std::unique_ptr<int>&&) {}, Tracked{}, std::make_unique<int>(1));
        t.join();
    };
    EXPECT_EQ(heap_delta(overflow), 0);
    EXPECT_EQ(heap_delta(eagain), 0);
    EXPECT_EQ(heap_delta(copy_throws), 0);
    EXPECT_EQ(heap_delta(succeeds), 0);
} // TEST(ThreadFailureTest, NoPathLeaksHeap)

// ===========================================================================
// The start mask.

// Whatever the creator's mask, the callable's first action sees the start
// mask; the creator's mask is unchanged right after the constructor returns
// and after the join. A first-action check shows the mask the callable
// begins with; it cannot show that no signal was delivered earlier in the
// new thread with another mask (that is the header's construction, not
// observable here).
TEST(ThreadMaskTest, CallableSeesStartMaskForEveryCreatorMask) {
    for (CreatorMask k : kCreatorMasks) {
        SCOPED_TRACE(name_of(k));
        on_creator(k, [k] {
            const sigset_t before = current_mask();
            EXPECT_EQ(mask_difference(before, make_mask(k)), "");
            sigset_t seen;
            sigemptyset(&seen);
            Thread t(&read_own_mask, &seen);
            const sigset_t after_return = current_mask();
            t.join();
            EXPECT_EQ(start_mask_mismatch(seen), "");
            EXPECT_EQ(mask_difference(before, after_return), "");
            EXPECT_EQ(mask_difference(before, current_mask()), "");
        });
    } // loop over the creator masks
} // TEST(ThreadMaskTest, CallableSeesStartMaskForEveryCreatorMask)

// When the constructor throws, the creator's mask is what it was before, on
// each failure path: the class's EINVAL, pthread_create's EAGAIN, and an
// exception while copying.
TEST(ThreadMaskTest, CreatorMaskUnchangedAfterEveryThrow) {
    for (CreatorMask k : kCreatorMasks) {
        SCOPED_TRACE(name_of(k));
        on_creator(k, [] {
            const sigset_t before = current_mask();
            const Caught einval = catch_system_error([] { Thread t({.stack_size = kSizeMax}, [] {}); });
            EXPECT_TRUE(is_error(einval, std::errc::invalid_argument));
            EXPECT_EQ(mask_difference(before, current_mask()), "");
            const Caught eagain = catch_system_error([] { Thread t({.stack_size = std::size_t{1} << 62}, [] {}); });
            EXPECT_TRUE(is_error(eagain, std::errc::resource_unavailable_try_again));
            EXPECT_EQ(mask_difference(before, current_mask()), "");
            Thrower<false> thrower;
            EXPECT_THROW(Thread([](Thrower<false>&&) {}, thrower), CopyError);
            EXPECT_EQ(mask_difference(before, current_mask()), "");
        });
    } // loop over the creator masks
} // TEST(ThreadMaskTest, CreatorMaskUnchangedAfterEveryThrow)

// A Thread created by a Thread's callable (whose own mask is the start mask)
// begins with the start mask as well, and the outer callable's mask is
// unchanged by the construction.
TEST(ThreadMaskTest, NestedThreadSeesStartMask) {
    sigset_t outer_before;
    sigset_t outer_after;
    sigset_t inner;
    sigemptyset(&outer_before);
    sigemptyset(&outer_after);
    sigemptyset(&inner);
    Thread outer([&] {
        outer_before = current_mask();
        Thread t(&read_own_mask, &inner);
        outer_after = current_mask();
        t.join();
    });
    outer.join();
    EXPECT_EQ(start_mask_mismatch(outer_before), "");
    EXPECT_EQ(start_mask_mismatch(inner), "");
    EXPECT_EQ(mask_difference(outer_before, outer_after), "");
} // TEST(ThreadMaskTest, NestedThreadSeesStartMask)

// A signal aimed at a worker with pthread_kill stays pending while the
// worker blocks it: the worker sees it in sigpending(), the handler has not
// run, and sigtimedwait() then consumes it. Signals outside the seven: a
// standard one, a terminating one, a real-time one.
TEST(ThreadMaskTest, SignalAimedAtWorkerStaysPending) {
    for (int sig : {SIGUSR1, SIGTERM, SIGRTMIN}) {
        SCOPED_TRACE(testing::Message() << "signal " << sig);
        const CountingHandler handler(sig);
        std::atomic<bool> sent{false};
        bool pending = false;
        int count_while_pending = -1;
        int consumed = 0;
        Thread t([&, sig] {
            while (!sent.load(std::memory_order_acquire)) std::this_thread::yield();
            const std::chrono::steady_clock::time_point deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!pending && std::chrono::steady_clock::now() < deadline) {
                sigset_t p;
                sigemptyset(&p);
                sigpending(&p);
                pending = sigismember(&p, sig) == 1;
            } // loop until the signal is pending
            count_while_pending = handler.count();
            sigset_t wanted;
            sigemptyset(&wanted);
            sigaddset(&wanted, sig);
            const timespec timeout{1, 0};
            consumed = sigtimedwait(&wanted, nullptr, &timeout);
        });
        ASSERT_EQ(pthread_kill(t.native_handle(), sig), 0);
        sent.store(true, std::memory_order_release);
        t.join();
        EXPECT_TRUE(pending);
        EXPECT_EQ(count_while_pending, 0);
        EXPECT_EQ(consumed, sig);
        EXPECT_EQ(handler.count(), 0);
    } // loop over the signals
} // TEST(ThreadMaskTest, SignalAimedAtWorkerStaysPending)

// Some of the seven, aimed at a worker, are delivered to it (its handler
// runs on the worker), whatever the creator's mask: the kernel-side view of
// "unblocked". Counts are asserted as at least one: deliveries may be
// coalesced. Only signals no sanitizer runtime handles itself are used.
TEST(ThreadMaskTest, UnblockedSignalsReachTheWorker) {
    for (int sig : {SIGPROF, SIGSYS, SIGTRAP}) {
        SCOPED_TRACE(testing::Message() << "signal " << sig);
        const CountingHandler handler(sig);
        on_creator(CreatorMask::kFull, [&handler, sig] {
            pthread_t worker_self{};
            Thread t([&handler, &worker_self] {
                worker_self = pthread_self();
                const std::chrono::steady_clock::time_point deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (handler.count() == 0 && std::chrono::steady_clock::now() < deadline) usleep(1000);
            });
            EXPECT_EQ(pthread_kill(t.native_handle(), sig), 0);
            t.join();
            EXPECT_GE(handler.count(), 1);
            EXPECT_NE(pthread_equal(handler.receiver(), worker_self), 0);
        });
    } // loop over the signals
} // TEST(ThreadMaskTest, UnblockedSignalsReachTheWorker)

// ===========================================================================
// Stack size.

// The stack cases: a request (empty = default) and the bytes the contract
// promises below the callable's first frame. Requests below the system
// minimum are raised to it; sizes that are not page multiples are included.
struct StackCase {
    std::optional<std::size_t> request;  // Attributes::stack_size
    std::size_t bound;                   // promised bytes below the first frame
}; // struct StackCase

std::vector<StackCase> stack_cases() {
    const std::size_t min = minimum_stack();
    return {
        {std::nullopt, Thread::default_stack_size()},
        {0, min},
        {1, min},
        {min - 1, min},
        {min, min},
        {64*1024 + 1, 64*1024 + 1},
        {300000, 300000},
        {3*kMiB + 1, 3*kMiB + 1},
    };
} // stack_cases()

// Empty if every stack case gets at least its bound below the callable's
// first frame; otherwise a description of each shortfall. Each thread's
// callable is view_stack with an ordinary (one-argument) list.
std::string stack_bound_failures() {
    std::string out;
    for (const StackCase& c : stack_cases()) {
        StackView v;
        try {
            Thread t(Thread::Attributes{.stack_size = c.request}, &view_stack, &v);
            t.join();
        } catch (const std::exception& e) {
            out += "request " + std::to_string(c.request.value_or(0)) + " threw " + e.what() + "; ";
            continue;
        }
        if (below_frame(v) < c.bound) {
            out += "request " + (c.request ? std::to_string(*c.request) : std::string("default")) + ": " +
                   std::to_string(below_frame(v)) + " bytes below the first frame < " + std::to_string(c.bound) +
                   "; ";
        }
    } // loop over the stack cases
    return out;
} // stack_bound_failures()

// At least stack_size bytes (the default, or the minimum for smaller
// requests) below the callable's first frame. In this process glibc may hand
// a thread a larger cached stack, which can hide a shortfall; the next test
// repeats the cases without the cache.
TEST(ThreadStackTest, LowerBoundBelowFirstFrame) {
    EXPECT_EQ(stack_bound_failures(), "");
} // TEST(ThreadStackTest, LowerBoundBelowFirstFrame)

// The same cases in a child process with glibc's stack cache disabled, so
// every thread gets a block of the size the class requested: a request that
// leaves less than the promised room fails here even when a cached larger
// block hides it in the test above. The child is a fresh exec of this binary
// (threadsafe death-test style), which reads the environment variable at
// load time.
TEST(ThreadStackTest, LowerBoundOnFreshStacks) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const EnvScope env("GLIBC_TUNABLES", "glibc.pthread.stack_cache_size=0");
    EXPECT_EXIT(
        {
            const std::string failures = stack_bound_failures();
            if (!failures.empty()) {
                marker(failures.c_str());
                std::_Exit(1);
            }
            std::_Exit(0);
        },
        testing::ExitedWithCode(0), "");
} // TEST(ThreadStackTest, LowerBoundOnFreshStacks)

// The default stack size is 2 MiB until changed; set_default_stack_size()
// stores its argument as given (0 and SIZE_MAX included) and returns the
// previous value; default_stack_size() returns the last value set.
TEST(ThreadStackTest, DefaultStackSizeSetAndQuery) {
    EXPECT_EQ(Thread::default_stack_size(), 2*kMiB);
    const DefaultStackScope restore(Thread::default_stack_size());
    for (std::size_t bytes : {std::size_t{12345}, std::size_t{0}, kSizeMax, 3*kMiB}) {
        const std::size_t previous = Thread::default_stack_size();
        EXPECT_EQ(Thread::set_default_stack_size(bytes), previous);
        EXPECT_EQ(Thread::default_stack_size(), bytes);
    } // loop over the values set
} // TEST(ThreadStackTest, DefaultStackSizeSetAndQuery)

// Late binding: an Attributes made before set_default_stack_size() and
// passed to a constructor after it gets the new default; so does the
// constructor without Attributes. The new default (9 MiB + 1) is larger than
// any block the old 2 MiB default could get from glibc's cache (at most four
// times the request), so a default read too early fails the bound.
TEST(ThreadStackTest, DefaultReadAtConstruction) {
    const std::size_t bytes = 9*kMiB + 1;
    const Thread::Attributes made_before{};
    const Thread::Attributes guard_only{.guard_size = 4096};
    const DefaultStackScope scope(bytes);
    StackView v1;
    StackView v2;
    StackView v3;
    Thread t1(made_before, &view_stack, &v1);
    Thread t2(guard_only, &view_stack, &v2);
    Thread t3(&view_stack, &v3);
    t1.join();
    t2.join();
    t3.join();
    EXPECT_GE(below_frame(v1), bytes);
    EXPECT_GE(below_frame(v2), bytes);
    EXPECT_GE(below_frame(v3), bytes);
} // TEST(ThreadStackTest, DefaultReadAtConstruction)

// A default of 0 gives threads the minimum stack; a default of SIZE_MAX makes
// the constructors that use the default throw EINVAL before copying, while a
// constructor with an explicit stack_size does not use it and succeeds.
TEST(ThreadStackTest, DefaultZeroAndSizeMax) {
    {
        const DefaultStackScope scope(0);
        StackView v;
        Thread t(&view_stack, &v);
        t.join();
        EXPECT_GE(below_frame(v), minimum_stack());
    } // default 0
    {
        const DefaultStackScope scope(kSizeMax);
        Tracked arg{1};
        const int first = census::mark();
        EXPECT_TRUE(is_error(catch_system_error([&] { Thread t([](Tracked&&) {}, std::move(arg)); }),
                             std::errc::invalid_argument));
        EXPECT_TRUE(is_error(
            catch_system_error([&] { Thread t(Thread::Attributes{}, [](Tracked&&) {}, std::move(arg)); }),
            std::errc::invalid_argument));
        EXPECT_EQ(census::mark(), first);
        EXPECT_FALSE(arg.moved_from());
        StackView v;
        Thread t(Thread::Attributes{.stack_size = kMiB}, &view_stack, &v);
        t.join();
        EXPECT_GE(below_frame(v), kMiB);
    } // default SIZE_MAX
} // TEST(ThreadStackTest, DefaultZeroAndSizeMax)

// set_default_stack_size() may be called concurrently with constructions and
// with other calls: setters alternate between two valid sizes while
// creators construct Threads with the default; every construction succeeds
// (and ThreadSanitizer reports no race).
TEST(ThreadStackTest, DefaultSetConcurrentlyWithConstructions) {
    const DefaultStackScope restore(Thread::default_stack_size());
    std::atomic<bool> stop{false};
    std::atomic<int> created{0};
    std::vector<std::thread> setters;
    for (int s = 0; s < 2; ++s) {
        setters.emplace_back([&stop, s] {
            std::size_t i = 0;
            while (!stop.load()) Thread::set_default_stack_size(((i++ + s) % 2 == 0) ? kMiB : 2*kMiB + 4096);
        });
    } // loop creating the setters
    std::vector<std::thread> creators;
    for (int c = 0; c < 2; ++c) {
        creators.emplace_back([&created] {
            for (int i = 0; i < 50; ++i) {
                Thread t([] {});
                t.join();
                created.fetch_add(1);
            } // loop over the constructions
        });
    } // loop creating the creators
    for (std::thread& c : creators) c.join();
    stop.store(true);
    for (std::thread& s : setters) s.join();
    EXPECT_EQ(created.load(), 100);
} // TEST(ThreadStackTest, DefaultSetConcurrentlyWithConstructions)

// The class adds room for the static TLS: with 128 KiB of static TLS in this
// binary (big_tls), a 64 KiB request still gets 64 KiB below the callable's
// first frame, and the thread's TLS lies in the same stack block above the
// frame (so the room had to come from the request plus an addition). Without
// the addition glibc would refuse the request (EINVAL) or leave less than
// the request. Skipped under ThreadSanitizer: its pthread_create raises small
// requests by more than the TLS, so a missing addition would not show.
TEST(ThreadStackTest, RoomForStaticTls) {
#if defined(THREAD_TEST_TSAN)
    GTEST_SKIP() << "ThreadSanitizer enlarges small stack requests, hiding a missing TLS addition";
#endif
    const std::size_t request = 64*1024;
    StackView v;
    Thread t(Thread::Attributes{.stack_size = request}, &view_stack, &v);
    t.join();
    EXPECT_GE(below_frame(v), request);
    EXPECT_GT(v.tls, v.frame);
    EXPECT_LT(v.tls, v.lo + v.size);
} // TEST(ThreadStackTest, RoomForStaticTls)

// The stagger: thread n reserves (n % 16) * 256 bytes at the top of its stack
// before the callable, n being a process-wide creation count. Black-box, n is
// unknown but consecutive constructions (nothing else constructs a Thread
// meanwhile) have consecutive n, so the depth of the callable's first frame
// below the stack top, minus (n % 16) * 256, must be the same for 32
// consecutive threads for one value of n mod 16 at the first. The depth is
// measured from the top pthread_getattr_np reports, so it does not depend on
// where the stack block lies.
TEST(ThreadStackTest, StaggerReservesConsecutive256ByteSteps) {
    constexpr int kThreads = 32;
    std::array<long long, kThreads> depth{};
    for (int i = 0; i < kThreads; ++i) {
        StackView v;
        Thread t(&view_stack, &v);
        t.join();
        depth[i] = static_cast<long long>(v.lo + v.size) - static_cast<long long>(v.frame);
    } // loop over consecutive threads
    int fitting = -1;
    for (int r = 0; r < 16; ++r) {
        bool fits = true;
        for (int i = 0; i < kThreads; ++i) {
            const long long expected = 256LL*((r + i) % 16 - r);
            if (depth[i] - depth[0] != expected) fits = false;
        } // loop over the threads
        if (fits) fitting = r;
    } // loop over the candidate first counts
    std::string depths;
    for (long long d : depth) depths += std::to_string(d - depth[0]) + " ";
    EXPECT_GE(fitting, 0) << "depths relative to the first: " << depths;
} // TEST(ThreadStackTest, StaggerReservesConsecutive256ByteSteps)

// ===========================================================================
// Guard size.

// The guard is an inaccessible mapping directly below the stack, at least
// the requested size rounded up to a page: the default (1 MiB), one page, a
// nonzero value smaller than the default and not a page multiple, and a
// value larger than the default. Checked from /proc/self/maps while the
// thread's stack is live, and from pthread_getattr_np.
TEST(ThreadGuardTest, GuardMappingAtLeastRequested) {
    struct GuardCase {
        std::optional<std::size_t> request;  // Attributes::guard_size
        std::size_t bound;                   // promised guard before page rounding
    }; // struct GuardCase
    const std::array<GuardCase, 4> cases = {
        GuardCase{std::nullopt, kMiB},
        GuardCase{4096, 4096},
        GuardCase{5000, 5000},
        GuardCase{4*kMiB + 1, 4*kMiB + 1},
    };
    for (const GuardCase& c : cases) {
        SCOPED_TRACE(testing::Message() << "guard " << c.request.value_or(0));
        GuardView v;
        Thread t(Thread::Attributes{.guard_size = c.request}, &view_guard, &v);
        t.join();
        EXPECT_TRUE(guard_at_least(v, c.bound));
    } // loop over the guard cases
} // TEST(ThreadGuardTest, GuardMappingAtLeastRequested)

// Guard 0 is accepted and passed to glibc, which reports 0 for the thread.
// Nothing is checked about the absence of a guard mapping: a stack glibc
// reuses from its cache may keep an earlier thread's guard, and the contract
// promises nothing about it.
TEST(ThreadGuardTest, GuardZeroAccepted) {
    StackView v;
    Thread t(Thread::Attributes{.guard_size = 0}, &view_stack, &v);
    t.join();
    EXPECT_EQ(v.guard, 0u);
} // TEST(ThreadGuardTest, GuardZeroAccepted)

// ===========================================================================
// Termination: deadlocking destructor and assignment, escaping exceptions,
// pthread_exit.

// The terminate handler of the deadlock death tests: proves std::terminate()
// was called (rather than some other abort) by its marker and exit code.
[[noreturn]] void deadlock_terminate_handler() {
    marker("terminate-called");
    std::_Exit(41);
} // deadlock_terminate_handler()

// The destructor run on the thread it would join (the callable destroying
// its own Thread object) calls std::terminate(). The object is handed over
// through a Gate opened after the constructor returns.
TEST(ThreadDeathTest, DestructorOnOwnThreadTerminates) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            std::set_terminate(&deadlock_terminate_handler);
            std::optional<Thread> slot;
            Gate handed_over;
            slot.emplace([&slot, &handed_over] {
                handed_over.wait();
                slot.reset();
            });
            handed_over.open();
            std::this_thread::sleep_for(std::chrono::seconds(30));
            std::_Exit(40);
        },
        testing::ExitedWithCode(41), "terminate-called");
} // TEST(ThreadDeathTest, DestructorOnOwnThreadTerminates)

// Move assignment run on the thread *this owns calls std::terminate().
TEST(ThreadDeathTest, MoveAssignmentOnOwnThreadTerminates) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            std::set_terminate(&deadlock_terminate_handler);
            Thread* self = nullptr;
            Gate handed_over;
            Thread t([&self, &handed_over] {
                handed_over.wait();
                *self = Thread();
            });
            self = &t;
            handed_over.open();
            std::this_thread::sleep_for(std::chrono::seconds(30));
            std::_Exit(40);
        },
        testing::ExitedWithCode(41), "terminate-called");
} // TEST(ThreadDeathTest, MoveAssignmentOnOwnThreadTerminates)

// The exception of the escape death test, and what its callable records for
// the terminate handler (written and read on the same thread).
struct EscapeError : std::exception {};
int escape_arg_id = -1;
bool escape_local_unwound = false;

// The terminate handler of the escape death test. Exit codes: 42 all as
// promised; 43 no current exception; 44 a current exception of another
// type; 45 the callable's local was not destroyed (frames not unwound); 46
// the argument's copy was already destroyed (the contract keeps the copies
// alive on this path).
[[noreturn]] void escape_terminate_handler() {
    const std::exception_ptr current = std::current_exception();
    if (!current) std::_Exit(43);
    try {
        std::rethrow_exception(current);
    } catch (const EscapeError&) {
        if (!escape_local_unwound) std::_Exit(45);
        if (escape_arg_id < 0 || census::destructions(escape_arg_id) != 0) std::_Exit(46);
        marker("terminate-saw-EscapeError");
        std::_Exit(42);
    } catch (...) {
        std::_Exit(44);
    }
} // escape_terminate_handler()

// An exception escaping the callable: the callable's frames are unwound (a
// local's destructor runs, marker first), then std::terminate() is called
// with the exception current (the handler sees it through
// std::current_exception()), and the copies are still alive.
TEST(ThreadDeathTest, EscapingExceptionUnwindsThenTerminates) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            std::set_terminate(&escape_terminate_handler);
            Thread t(
                [](Tracked&& arg) {
                    struct Local {
                        ~Local() {
                            marker("local-unwound;");
                            escape_local_unwound = true;
                        } // Local::~Local()
                    }; // struct Local
                    escape_arg_id = arg.id();
                    Local local;
                    throw EscapeError{};
                },
                Tracked{});
            t.join();
            std::_Exit(40);
        },
        testing::ExitedWithCode(42), "local-unwound;terminate-saw-EscapeError");
} // TEST(ThreadDeathTest, EscapingExceptionUnwindsThenTerminates)

// pthread_exit in the callable: the thread's stack is unwound (a local is
// destroyed, on the thread), the copies are destroyed exactly once, and
// join() returns (with the destructions complete: plain fields read after
// join()).
TEST(ThreadExitTest, PthreadExitUnwindsAndDestroysCopies) {
    const int first = census::mark();
    int local_id = -1;
    int arg_id = -1;
    std::thread::id worker;
    {
        Thread t(
            [&local_id, &arg_id, &worker](Tracked&& arg) {
                const Tracked local{9};
                local_id = local.id();
                arg_id = arg.id();
                worker = std::this_thread::get_id();
                pthread_exit(nullptr);
            },
            Tracked{8});
        t.join();
        EXPECT_FALSE(t.joinable());
        ASSERT_GE(local_id, 0);
        EXPECT_EQ(census::destructions(local_id), 1);
        EXPECT_EQ(census::slots[local_id].destroyed_on, worker);
        EXPECT_EQ(census::destructions(arg_id), 1);
    } // scope of the Thread
    EXPECT_TRUE(census::each_destroyed_once(first));
} // TEST(ThreadExitTest, PthreadExitUnwindsAndDestroysCopies)

} // namespace
