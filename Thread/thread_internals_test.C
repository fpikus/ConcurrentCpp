// White-box tests of thread.h: the internals of Thread (namespace
// thread_detail), each tested through the seam the header exposes for it.
// thread_test.C tests the documented contract; this file tests the mechanism
// behind it, where the public interface cannot see a defect:
//   - the size arithmetic (stagger_for, page_round, compute_stack_size,
//     guard_fits, tls_carve_bound, tls_info_from, tls_size_from), at its real
//     boundaries: the clamp, page boundaries +-1, the overflow points near
//     SIZE_MAX, 4 KiB, 16 KiB and 64 KiB pages; as static_asserts and as
//     runtime checks;
//   - the machine parameters (page_size, min_stack_size, stack_granule,
//     static_tls_size) against the system's own answers;
//   - AttrOwner: the stack size it passes to glibc equals the test's own
//     computation exactly, and every overflow throws EINVAL;
//   - the start mask: start_mask() membership without threads, the creating
//     thread's side (CreationMaskScope, create_thread) and the new thread's
//     side (start_routine), from creators with different masks;
//   - the stack stagger: the counter, the depth of the callable's frame below
//     the stack top (one constant plus the thread's stagger amount, for
//     threads alive at once, for one reused block, for mixed sizes and guards,
//     across the counter's wrap), the page offsets it produces, and the stack
//     left below the callable's first frame at the deepest stagger slot;
//   - the construction's failure paths: an overflow copies nothing and takes
//     no creation number; a failing pthread_create destroys the copies in the
//     creating thread and leaves the handle reset; the copies are destroyed
//     exactly once, by the new thread, on the normal and the pthread_exit
//     paths, and never on the terminate path.
//
// The oracles are the documented rule, not the header's own helpers: the
// stagger amount is (n % 16) * 256 and the stack extra 3840 + 2048 written
// out here, the static TLS term and the rounding unit are computed here from
// glibc's query and sysconf. A test that used stagger_for(), stack_extra or
// static_tls_size() as its expected value would pass with a mutated header.
//
// Builds. Every source is built three ways (see the Makefile): ASan and TSan
// at -O0, and -O3 without a sanitizer. Two defects are visible only in the
// optimized build: an unused stack reservation (an alloca whose pointer
// nothing reads) is deleted by the optimizer, and a callable invoker that is
// inlined into the start routine puts the callable's locals above the
// reservation; at -O0 neither happens. And the sanitizer runtimes intercept
// pthread_create: ASan sets the new thread's mask back to its creator's,
// TSan raises small stack requests, so only the optimized build runs the
// header's code as it ships.
//
// This binary has a 64 KiB thread_local (big_tls below), so its static TLS
// block is large: the TLS term of the stack size matters (a minimum stack
// without it fails with EINVAL), and every stack-size expectation here
// includes it.
//
// Process-wide state the tests change (the stagger counter, the default stack
// size, RLIMIT_NPROC) is restored by RAII. Signal masks are changed only on
// helper std::thread creators, never on the gtest main thread. Every thread a
// test creates is joined before the test returns.
#include <gtest/gtest.h>

#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <latch>
#include <limits>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "thread.h"

// THREAD_TEST_TSAN: defined in ThreadSanitizer builds (GCC defines
// __SANITIZE_THREAD__, Clang answers __has_feature).
#if defined(__SANITIZE_THREAD__)
#define THREAD_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define THREAD_TEST_TSAN 1
#endif
#endif

// The sanitizer allocator's count of live heap bytes, declared by hand (GCC
// ships no <sanitizer/allocator_interface.h>) and weak: null in the build
// without a sanitizer, where the heap checks are skipped.
extern "C" std::size_t __sanitizer_get_current_allocated_bytes() __attribute__((weak));

namespace {

// A 64 KiB thread_local: part of the static TLS block of every thread of this
// binary (initial-exec model; in an executable the compiler uses the
// equivalent local-exec model). Large enough that a minimum stack without the
// TLS term cannot hold it.
// Marked used: the tests only write it, and an optimizer would otherwise
// delete the variable, and its TLS with it.
[[gnu::used]] __attribute__((tls_model("initial-exec"))) thread_local char big_tls[std::size_t{64} << 10];

constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();

// ---------------------------------------------------------------------------
// The test's own statement of the rules (the oracles).
// ---------------------------------------------------------------------------

// The stagger amount of creation number n, as documented: (n % 16) * 256.
constexpr std::size_t own_stagger(unsigned n) {
    return (n % 16)*256;
} // own_stagger()

// What the stack size adds to the request, as documented: the deepest
// stagger reservation (15 * 256) and the start routine's frames (2048).
constexpr std::size_t own_extra = 3840 + 2048;

// `n` rounded up to a multiple of `unit` (no overflow: callers pass small
// values).
constexpr std::size_t round_up(std::size_t n, std::size_t unit) {
    return (n + unit - 1)/unit*unit;
} // round_up()

// The static TLS size and alignment glibc reports for this binary; {0, 1} if
// the query is absent.
std::pair<std::size_t, std::size_t> own_tls_info() {
    std::size_t size = 0;
    std::size_t align = 1;
    if (thread_detail::_dl_get_tls_static_info != nullptr) thread_detail::_dl_get_tls_static_info(&size, &align);
    if (align == 0) align = 1;
    return {size, align};
} // own_tls_info()

// The documented TLS term: the static TLS size rounded up to its alignment,
// plus alignment - 1.
std::size_t own_tls_term() {
    const auto [size, align] = own_tls_info();
    return round_up(size, align) + (align - 1);
} // own_tls_term()

// The system's page size.
std::size_t own_page() {
    return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
} // own_page()

// The documented rounding unit: the page size, or the static TLS alignment
// if that is larger.
std::size_t own_granule() {
    const std::size_t align = own_tls_info().second;
    return align > own_page() ? align : own_page();
} // own_granule()

// The smallest stack glibc accepts.
std::size_t own_minimum() {
    return static_cast<std::size_t>(sysconf(_SC_THREAD_STACK_MIN));
} // own_minimum()

// The documented stack size for a request of `requested` bytes:
// round_up(max(requested, minimum) + extra + TLS term, granule).
std::size_t own_stack_size(std::size_t requested) {
    const std::size_t clamped = requested < own_minimum() ? own_minimum() : requested;
    return round_up(clamped + own_extra + own_tls_term(), own_granule());
} // own_stack_size()

// ---------------------------------------------------------------------------
// Fixtures for process-wide state.
// ---------------------------------------------------------------------------

// Saves the stagger counter and restores it at scope end.
class CounterSaver {
public:
    CounterSaver() : saved_(thread_detail::stagger_counter.load(std::memory_order_relaxed)) {}
    ~CounterSaver() { thread_detail::stagger_counter.store(saved_, std::memory_order_relaxed); }
    CounterSaver(const CounterSaver&) = delete;
    CounterSaver& operator=(const CounterSaver&) = delete;

private:
    unsigned saved_;            // the counter's value when the scope began
}; // class CounterSaver

// Sets the process default stack size and restores the previous one at scope
// end.
class DefaultStackScope {
public:
    explicit DefaultStackScope(std::size_t bytes) : saved_(Thread::set_default_stack_size(bytes)) {}
    ~DefaultStackScope() { Thread::set_default_stack_size(saved_); }
    DefaultStackScope(const DefaultStackScope&) = delete;
    DefaultStackScope& operator=(const DefaultStackScope&) = delete;

private:
    std::size_t saved_;         // the default before the scope, restored by the destructor
}; // class DefaultStackScope

// Lowers the soft RLIMIT_NPROC to 0, which makes every clone of this process
// fail (pthread_create returns EAGAIN after glibc has set up the new thread's
// stack and stored its handle), and restores the limit at scope end. A
// privileged process is not limited; lowered() then says whether the call
// succeeded, not whether creation fails.
class NoNewThreadsScope {
public:
    NoNewThreadsScope() {
        lowered_ = getrlimit(RLIMIT_NPROC, &saved_) == 0;
        if (!lowered_) return;
        rlimit none = saved_;
        none.rlim_cur = 0;
        lowered_ = setrlimit(RLIMIT_NPROC, &none) == 0;
    }
    ~NoNewThreadsScope() {
        if (lowered_) setrlimit(RLIMIT_NPROC, &saved_);
    }
    NoNewThreadsScope(const NoNewThreadsScope&) = delete;
    NoNewThreadsScope& operator=(const NoNewThreadsScope&) = delete;

    bool lowered() const { return lowered_; }

private:
    rlimit saved_{};            // the limit before the scope
    bool lowered_ = false;      // whether the soft limit was lowered (and must be restored)
}; // class NoNewThreadsScope

// Net heap bytes allocated by fn() in a sanitizer build, measured over a
// second call after one warm-up call (the first call may initialize caches
// that stay allocated); empty in the build without a sanitizer.
template <class Fn>
std::optional<long long> heap_delta(Fn fn) {
    if (__sanitizer_get_current_allocated_bytes == nullptr) return {};
    fn();
    const std::size_t before = __sanitizer_get_current_allocated_bytes();
    fn();
    const std::size_t after = __sanitizer_get_current_allocated_bytes();
    return static_cast<long long>(after) - static_cast<long long>(before);
} // heap_delta()

// The generic-category error code of the std::system_error fn() throws; -1 if
// it throws nothing, -2 for any other exception, -3 for another category.
template <class Fn>
int system_error_code(Fn fn) {
    try {
        fn();
    } catch (const std::system_error& e) {
        if (e.code().category() != std::generic_category()) return -3;
        return e.code().value();
    } catch (...) {
        return -2;
    }
    return -1;
} // system_error_code()

// ---------------------------------------------------------------------------
// Signal masks. Compared by sigismember over 1..SIGRTMAX, never as bytes: the
// kernel writes only part of a sigset_t, and reports SIGKILL and SIGSTOP as
// never blocked.
// ---------------------------------------------------------------------------

// The seven signals a new Thread always has unblocked, the test's own list.
constexpr std::array<int, 7> own_seven = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGPROF, SIGTRAP, SIGSYS};

bool in_seven(int signo) {
    for (const int s : own_seven) {
        if (s == signo) return true;
    }
    return false;
} // in_seven()

// Whether the documented start mask blocks `signo`: every signal except the
// seven, SIGKILL, SIGSTOP and glibc's internal signals (from 32 up to
// SIGRTMIN, exclusive), which no thread can block.
bool start_blocked(int signo) {
    if (in_seven(signo) || signo == SIGKILL || signo == SIGSTOP) return false;
    return !(signo >= 32 && signo < SIGRTMIN);
} // start_blocked()

bool is_blocked(const sigset_t& mask, int signo) {
    return sigismember(&mask, signo) == 1;
} // is_blocked()

// The calling thread's mask.
sigset_t current_mask() {
    sigset_t mask;
    pthread_sigmask(SIG_BLOCK, nullptr, &mask);
    return mask;
} // current_mask()

// Success if `mask` blocks exactly the signals s of 1..SIGRTMAX for which
// expected(s) is true; the failure lists the signals in the wrong state.
template <class Pred>
testing::AssertionResult mask_is(const sigset_t& mask, Pred expected) {
    std::string wrong;
    for (int signo = 1; signo <= SIGRTMAX; ++signo) {
        if (is_blocked(mask, signo) != expected(signo)) {
            wrong += " " + std::to_string(signo) + (is_blocked(mask, signo) ? "(blocked)" : "(unblocked)");
        }
    }
    if (wrong.empty()) return testing::AssertionSuccess();
    return testing::AssertionFailure() << "signals in the wrong state:" << wrong;
} // mask_is()

// A creating thread's mask, named for the failure messages.
struct CreatorMask {
    const char* name;
    sigset_t mask;
}; // struct CreatorMask

// The creator masks the mask tests run from: nothing blocked; everything
// blocked; the seven blocked (so the new thread must unblock them); and an
// unrelated user mask that includes one of the seven (SIGUSR1, SIGPROF).
std::vector<CreatorMask> creator_masks() {
    std::vector<CreatorMask> masks(4);
    masks[0].name = "empty";
    sigemptyset(&masks[0].mask);
    masks[1].name = "full";
    sigfillset(&masks[1].mask);
    masks[2].name = "the seven";
    sigemptyset(&masks[2].mask);
    for (const int signo : own_seven) sigaddset(&masks[2].mask, signo);
    masks[3].name = "SIGUSR1 and SIGPROF";
    sigemptyset(&masks[3].mask);
    sigaddset(&masks[3].mask, SIGUSR1);
    sigaddset(&masks[3].mask, SIGPROF);
    return masks;
} // creator_masks()

// Runs fn() on a helper std::thread whose mask is `mask`, and joins it.
template <class Fn>
void on_creator(const sigset_t& mask, Fn fn) {
    std::thread helper([&mask, &fn] {
        pthread_sigmask(SIG_SETMASK, &mask, nullptr);
        fn();
    });
    helper.join();
} // on_creator()

// ---------------------------------------------------------------------------
// Stack placement.
// ---------------------------------------------------------------------------

// The calling thread's stack as glibc reports it: the low end (above the
// guard) and the size, which includes the static TLS at the top.
struct StackSpan {
    std::uintptr_t low = 0;
    std::size_t size = 0;

    std::uintptr_t top() const { return low + size; }
}; // struct StackSpan

StackSpan own_stack() {
    pthread_attr_t attr;
    pthread_getattr_np(pthread_self(), &attr);
    void* low = nullptr;
    std::size_t size = 0;
    pthread_attr_getstack(&attr, &low, &size);
    pthread_attr_destroy(&attr);
    return {reinterpret_cast<std::uintptr_t>(low), size};
} // own_stack()

// Where one thread's callable found itself.
struct FrameRecord {
    std::uintptr_t frame = 0;   // __builtin_frame_address(0) in the callable's body
    StackSpan stack;            // the thread's stack

    // Bytes from the stack top down to the callable's frame.
    std::size_t depth() const { return stack.top() - frame; }
}; // struct FrameRecord

// The callable of the placement tests. It records its own frame address,
// taken in its own body, and the stack; then, if `hold` is set, waits until
// every thread of the test has done the same (so that all are alive at once
// and no stack block is reused among them). Always inlined into its caller,
// so its frame is the frame of whatever function invokes it: with the header
// as written that is invoke_state(), below the stagger reservation; an
// invoker inlined into the start routine would put it above. (A separate
// non-inlined probe function would be called below the reservation in both
// cases and could not tell them apart.) One type for every placement test,
// so every thread runs the same start routine instantiation and the frames
// between the stack top and the callable do not vary between threads.
struct RecordFrame {
    [[gnu::always_inline]] void operator()(FrameRecord* record, std::latch* hold) const {
        record->frame = reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0));
        record->stack = own_stack();
        if (hold != nullptr) hold->arrive_and_wait();
    }
}; // struct RecordFrame

// The `hold` argument of a RecordFrame that does not wait, typed as the
// parameter, so that every placement test instantiates one State type.
std::latch* const no_hold = nullptr;

// Calls start_routine<S>(raw) through a volatile function pointer: the call
// cannot be inlined (as glibc's call through a pointer cannot), so the start
// routine keeps a frame of its own below the caller's.
template <class S>
void call_start_routine(S* raw) {
    void* (*volatile routine)(void*) = &thread_detail::start_routine<S>;
    routine(raw);
} // call_start_routine()

// Creates one Thread per entry of `attributes`, all alive at once (each
// callable waits at a latch until all have recorded), and joins them. The
// creation numbers are first, first + 1, ...: the counter is set to `first`
// here. Returns the records in creation order.
std::vector<FrameRecord> place_alive(const std::vector<Thread::Attributes>& attributes, unsigned first) {
    std::vector<FrameRecord> records(attributes.size());
    std::latch hold(static_cast<std::ptrdiff_t>(attributes.size()));
    std::vector<Thread> threads;
    threads.reserve(attributes.size());
    thread_detail::stagger_counter.store(first, std::memory_order_relaxed);
    for (std::size_t i = 0; i != attributes.size(); ++i) {
        try {
            threads.emplace_back(attributes[i], RecordFrame{}, &records[i], &hold);
        } catch (...) {
            // Releases the threads already waiting, then reports the failure.
            hold.count_down(static_cast<std::ptrdiff_t>(attributes.size() - i));
            throw;
        }
    } // for every thread to create
    return records;             // `threads` joins every thread here
} // place_alive()

// Success if depth(i) - own_stagger(first + i) is the same for every record:
// each thread's callable frame lies exactly its stagger amount deeper than
// an unstaggered one. Independent of where the stacks are and of their sizes.
testing::AssertionResult depths_follow_stagger(const std::vector<FrameRecord>& records, unsigned first) {
    const long long base = static_cast<long long>(records[0].depth()) - static_cast<long long>(own_stagger(first));
    for (std::size_t i = 0; i != records.size(); ++i) {
        const unsigned n = first + static_cast<unsigned>(i);
        const long long unstaggered =
            static_cast<long long>(records[i].depth()) - static_cast<long long>(own_stagger(n));
        if (unstaggered != base) {
            return testing::AssertionFailure() << "thread " << i << " (number " << n << "): depth "
                                               << records[i].depth() << " minus stagger " << own_stagger(n) << " is "
                                               << unstaggered << ", thread 0 gives " << base;
        }
    } // for every record
    return testing::AssertionSuccess();
} // depths_follow_stagger()

// The number of distinct values of the callable frames' page offsets.
std::size_t distinct_page_offsets(const std::vector<FrameRecord>& records) {
    std::set<std::uintptr_t> offsets;
    for (const FrameRecord& record : records) offsets.insert(record.frame % own_page());
    return offsets.size();
} // distinct_page_offsets()

// The number of distinct values of (frame + stagger) mod page: 1 when every
// thread's stack top has the same page offset.
std::size_t distinct_unstaggered_offsets(const std::vector<FrameRecord>& records, unsigned first) {
    std::set<std::uintptr_t> offsets;
    for (std::size_t i = 0; i != records.size(); ++i) {
        offsets.insert((records[i].frame + own_stagger(first + static_cast<unsigned>(i))) % own_page());
    }
    return offsets.size();
} // distinct_unstaggered_offsets()

// ---------------------------------------------------------------------------
// Copy accounting.
// ---------------------------------------------------------------------------

// Counts of the copies of Tracked objects made by a construction.
struct CopyCounts {
    std::atomic<int> copies{0};                 // copies and moves constructed
    std::atomic<int> copies_destroyed{0};       // of those, destroyed
    std::atomic<int> destroyed_by_creator{0};   // of those, destroyed on the thread that made these counts
    const std::thread::id creator = std::this_thread::get_id();
}; // struct CopyCounts

// A value whose copies and moves are counted in a CopyCounts. The originals a
// test makes are not counted; every object made from another one is (the
// decay-copies of a construction). A moved-from object knows it.
struct Tracked {
    explicit Tracked(CopyCounts* c) : counts(c) {}
    Tracked(const Tracked& other) : counts(other.counts), is_copy(true) { ++counts->copies; }
    Tracked(Tracked&& other) noexcept : counts(other.counts), is_copy(true) {
        other.moved_from = true;
        ++counts->copies;
    }
    Tracked& operator=(const Tracked&) = delete;
    ~Tracked() {
        if (!is_copy) return;
        ++counts->copies_destroyed;
        if (std::this_thread::get_id() == counts->creator) ++counts->destroyed_by_creator;
    }

    CopyCounts* counts;         // where the copies are counted
    bool is_copy = false;       // made from another Tracked (counted)
    bool moved_from = false;    // left moved-from by a move
}; // struct Tracked

// How a TrackedCallable ends.
enum class Ending { normal, thread_exit, exception };

// Set by a TrackedCallable's local object when it is destroyed (unwound).
std::atomic<int> locals_destroyed{0};

// A local object of TrackedCallable, counting its destruction.
struct CallableLocal {
    ~CallableLocal() { ++locals_destroyed; }
}; // struct CallableLocal

// A callable that carries a Tracked (so its own decay-copy is counted), takes
// a Tracked argument by rvalue reference (so the invocation makes no further
// copy) and ends as told.
struct TrackedCallable {
    Tracked marker;
    Ending ending = Ending::normal;

    void operator()(Tracked&& argument) {
        const CallableLocal local;
        static_cast<void>(argument);
        if (ending == Ending::thread_exit) pthread_exit(nullptr);
        if (ending == Ending::exception) throw std::runtime_error("from the callable");
    }
}; // struct TrackedCallable

// A callable that stores the mask it runs with in *out.
struct MaskRecorder {
    void operator()(sigset_t* out) const { *out = current_mask(); }
}; // struct MaskRecorder

// A start routine for create_thread(): stores the mask the new thread has on
// entry, before any other code runs in it, in *arg.
void* record_entry_mask(void* arg) {
    pthread_sigmask(SIG_BLOCK, nullptr, static_cast<sigset_t*>(arg));
    return nullptr;
} // record_entry_mask()

// A start routine that does nothing.
void* do_nothing(void*) {
    return nullptr;
} // do_nothing()

// A pthread_attr_t with an optional stack size, destroyed at scope end.
class PlainAttr {
public:
    explicit PlainAttr(std::optional<std::size_t> stack = {}) {
        pthread_attr_init(&attr_);
        if (stack) pthread_attr_setstacksize(&attr_, *stack);
    }
    ~PlainAttr() { pthread_attr_destroy(&attr_); }
    PlainAttr(const PlainAttr&) = delete;
    PlainAttr& operator=(const PlainAttr&) = delete;

    const pthread_attr_t& get() const { return attr_; }

private:
    pthread_attr_t attr_;
}; // class PlainAttr

} // namespace

// ===========================================================================
// Size arithmetic, at compile time. The mutation runs define
// THREAD_INTERNALS_NO_STATIC_CHECKS to compile these out, so that a mutated
// header builds and the runtime tests below show which of them kills it.
// ===========================================================================

#if !defined(THREAD_INTERNALS_NO_STATIC_CHECKS)
// stagger_for: (n % 16) * 256, continuing across the counter's wrap.
static_assert(thread_detail::stagger_for(0) == 0);
static_assert(thread_detail::stagger_for(1) == 256);
static_assert(thread_detail::stagger_for(15) == 3840);
static_assert(thread_detail::stagger_for(16) == 0);
static_assert(thread_detail::stagger_for(UINT_MAX) == 3840);
static_assert(thread_detail::stagger_for(UINT_MAX - 15) == 0);

// page_round at the edges: 0, the unit +-1, the largest multiple that fits.
static_assert(thread_detail::page_round(0, 4096) == std::size_t{0});
static_assert(thread_detail::page_round(1, 4096) == std::size_t{4096});
static_assert(thread_detail::page_round(4095, 4096) == std::size_t{4096});
static_assert(thread_detail::page_round(4096, 4096) == std::size_t{4096});
static_assert(thread_detail::page_round(4097, 4096) == std::size_t{8192});
static_assert(thread_detail::page_round(size_max - 4095, 4096) == size_max - 4095);
static_assert(!thread_detail::page_round(size_max - 4094, 4096));
static_assert(thread_detail::page_round(16385, 16384) == std::size_t{32768});
static_assert(thread_detail::page_round(size_max - 65535, 65536) == size_max - 65535);
static_assert(!thread_detail::page_round(size_max - 65534, 65536));
static_assert(thread_detail::page_round(size_max, 1) == size_max);

// compute_stack_size with extra 5888 and the TLS term 4287 (a 4224-byte block
// aligned to 64): the clamp applies to the request, the additions on top.
static_assert(thread_detail::compute_stack_size(0, 5888, 4287, 4096, 16384) == std::size_t{28672});
static_assert(thread_detail::compute_stack_size(1, 5888, 0, 4096, 16384) == std::size_t{24576});
static_assert(thread_detail::compute_stack_size(16383, 0, 0, 4096, 16384) == std::size_t{16384});
static_assert(thread_detail::compute_stack_size(16385, 0, 0, 4096, 16384) == std::size_t{20480});
static_assert(thread_detail::compute_stack_size(22593, 5888, 4287, 4096, 16384) == std::size_t{32768});
static_assert(thread_detail::compute_stack_size(22594, 5888, 4287, 4096, 16384) == std::size_t{36864});
static_assert(thread_detail::compute_stack_size(0, 5888, 4287, 16384, 16384) == std::size_t{32768});
static_assert(thread_detail::compute_stack_size(0, 5888, 4287, 65536, 131072) == std::size_t{196608});
static_assert(!thread_detail::compute_stack_size(size_max - 5887, 5888, 0, 4096, 16384));
static_assert(!thread_detail::compute_stack_size(size_max - 5888 - 4286, 5888, 4287, 4096, 16384));
static_assert(thread_detail::compute_stack_size(size_max - 4095 - 5888 - 4287, 5888, 4287, 4096, 16384) ==
              size_max - 4095);
static_assert(!thread_detail::compute_stack_size(size_max - 4094 - 5888 - 4287, 5888, 4287, 4096, 16384));

// guard_fits: the guard is rounded to a page before the sum is checked.
static_assert(thread_detail::guard_fits(0, 0, 4096));
static_assert(thread_detail::guard_fits(0, size_max - 4095, 4096));
static_assert(!thread_detail::guard_fits(0, size_max, 4096));
static_assert(thread_detail::guard_fits(size_max - 4096, 1, 4096));
static_assert(!thread_detail::guard_fits(size_max - 4095, 1, 4096));

// tls_carve_bound: rounded size plus alignment - 1, saturating.
static_assert(thread_detail::tls_carve_bound(0, 1) == 0);
static_assert(thread_detail::tls_carve_bound(5, 0) == 5);
static_assert(thread_detail::tls_carve_bound(4224, 64) == 4287);
static_assert(thread_detail::tls_carve_bound(1001, 256) == 1279);
static_assert(thread_detail::tls_carve_bound(1, 4096) == 8191);
static_assert(thread_detail::tls_carve_bound(size_max, 64) == size_max);
#endif // !THREAD_INTERNALS_NO_STATIC_CHECKS

// The constructor constraint: never a Thread or an Attributes as the callable,
// whatever their cv and reference qualification.
static_assert(!thread_detail::thread_callable<Thread, Thread>);
static_assert(!thread_detail::thread_callable<Thread&, Thread>);
static_assert(!thread_detail::thread_callable<const Thread&, Thread>);
static_assert(!thread_detail::thread_callable<Thread::Attributes&, Thread>);
static_assert(!thread_detail::thread_callable<const Thread::Attributes&&, Thread>);
static_assert(thread_detail::thread_callable<void (*)(), Thread>);
static_assert(thread_detail::thread_callable<RecordFrame&, Thread>);

// The storage test below reads the handle from the bytes of a Thread: the
// handle must be its only member.
static_assert(sizeof(Thread) == sizeof(pthread_t));

// ===========================================================================
// Size arithmetic, at run time (the same boundaries, with failure messages).
// ===========================================================================

// The stagger rotation: 16 positions 256 bytes apart, continuing across the
// counter's wrap from UINT_MAX to 0 (2^32 is a multiple of 16).
TEST(StaggerFor, RotationAndWrap) {
    for (unsigned n = 0; n != 40; ++n) EXPECT_EQ(thread_detail::stagger_for(n), own_stagger(n)) << "n = " << n;
    for (unsigned n = UINT_MAX - 20; n != 0; ++n) EXPECT_EQ(thread_detail::stagger_for(n), own_stagger(n));
    EXPECT_EQ(thread_detail::stagger_for(UINT_MAX), 3840u);
}

// next_stagger takes one creation number and returns its stagger amount;
// from UINT_MAX it wraps to 0.
TEST(StaggerCounter, NextTakesOneNumber) {
    const CounterSaver saver;
    thread_detail::stagger_counter.store(37, std::memory_order_relaxed);
    EXPECT_EQ(thread_detail::next_stagger(), own_stagger(37));
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 38u);
    thread_detail::stagger_counter.store(UINT_MAX, std::memory_order_relaxed);
    EXPECT_EQ(thread_detail::next_stagger(), std::size_t{3840});
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(thread_detail::next_stagger(), std::size_t{0});
}

// page_round: rounding at the unit's edges, and the overflow point: the
// largest multiple of the unit that fits is SIZE_MAX + 1 - unit; one more
// byte cannot be rounded up.
TEST(PageRound, EdgesAndOverflow) {
    for (const std::size_t unit : {std::size_t{4096}, std::size_t{16384}, std::size_t{65536}}) {
        SCOPED_TRACE("unit " + std::to_string(unit));
        EXPECT_EQ(thread_detail::page_round(0, unit), std::size_t{0});
        EXPECT_EQ(thread_detail::page_round(1, unit), unit);
        EXPECT_EQ(thread_detail::page_round(unit - 1, unit), unit);
        EXPECT_EQ(thread_detail::page_round(unit, unit), unit);
        EXPECT_EQ(thread_detail::page_round(unit + 1, unit), 2*unit);
        EXPECT_EQ(thread_detail::page_round(size_max - (unit - 1), unit), size_max - (unit - 1));
        EXPECT_FALSE(thread_detail::page_round(size_max - (unit - 2), unit));
        EXPECT_FALSE(thread_detail::page_round(size_max, unit));
    }
    EXPECT_EQ(thread_detail::page_round(size_max, 1), size_max);
}

// The clamp applies to the request BEFORE the additions: a request below the
// minimum gets the minimum plus the extra and the TLS term, not the larger of
// (request + additions) and the minimum.
TEST(ComputeStackSize, ClampBeforeAdditions) {
    EXPECT_EQ(thread_detail::compute_stack_size(0, 5888, 4287, 4096, 16384), std::size_t{28672});
    EXPECT_EQ(thread_detail::compute_stack_size(1, 5888, 0, 4096, 16384), std::size_t{24576});
    EXPECT_EQ(thread_detail::compute_stack_size(16383, 5888, 4287, 4096, 16384), std::size_t{28672});
    EXPECT_EQ(thread_detail::compute_stack_size(16383, 0, 0, 4096, 16384), std::size_t{16384});
    EXPECT_EQ(thread_detail::compute_stack_size(16384, 0, 0, 4096, 16384), std::size_t{16384});
    EXPECT_EQ(thread_detail::compute_stack_size(16385, 0, 0, 4096, 16384), std::size_t{20480});
}

// Page boundaries of the sum: request + extra + TLS = 32767, 32768, 32769.
// And the TLS term alone carrying the sum over a page boundary (20480 + 4000
// stays in the page, + 200 more does not).
TEST(ComputeStackSize, PageBoundaries) {
    EXPECT_EQ(thread_detail::compute_stack_size(22592, 5888, 4287, 4096, 16384), std::size_t{32768});
    EXPECT_EQ(thread_detail::compute_stack_size(22593, 5888, 4287, 4096, 16384), std::size_t{32768});
    EXPECT_EQ(thread_detail::compute_stack_size(22594, 5888, 4287, 4096, 16384), std::size_t{36864});
    EXPECT_EQ(thread_detail::compute_stack_size(20480, 4000, 0, 4096, 16384), std::size_t{24576});
    EXPECT_EQ(thread_detail::compute_stack_size(20480, 4000, 200, 4096, 16384), std::size_t{28672});
}

// Other page sizes and minimums: 16 KiB pages, and 64 KiB pages with the
// 128 KiB minimum of such systems.
TEST(ComputeStackSize, LargePages) {
    EXPECT_EQ(thread_detail::compute_stack_size(0, 5888, 4287, 16384, 16384), std::size_t{32768});
    EXPECT_EQ(thread_detail::compute_stack_size(16384*3 - 5888 - 4287, 5888, 4287, 16384, 16384),
              std::size_t{16384*3});
    EXPECT_EQ(thread_detail::compute_stack_size(16384*3 - 5888 - 4286, 5888, 4287, 16384, 16384),
              std::size_t{16384*4});
    EXPECT_EQ(thread_detail::compute_stack_size(0, 5888, 4287, 65536, 131072), std::size_t{196608});
    EXPECT_EQ(thread_detail::compute_stack_size(131071, 0, 0, 65536, 131072), std::size_t{131072});
    EXPECT_EQ(thread_detail::compute_stack_size(131073, 0, 0, 65536, 131072), std::size_t{196608});
}

// Each overflow point of the computation: request + extra, + TLS, the
// round-up; just below each, a value. A request so large that request + extra
// wraps must not come back as a small size.
TEST(ComputeStackSize, OverflowPoints) {
    EXPECT_FALSE(thread_detail::compute_stack_size(size_max, 5888, 4287, 4096, 16384));
    EXPECT_FALSE(thread_detail::compute_stack_size(size_max - 100, 5888, 4287, 4096, 16384));
    EXPECT_FALSE(thread_detail::compute_stack_size(size_max - 5887, 5888, 0, 4096, 16384));
    EXPECT_FALSE(thread_detail::compute_stack_size(size_max - 5888 - 4286, 5888, 4287, 4096, 16384));
    EXPECT_FALSE(thread_detail::compute_stack_size(16384, 5888, size_max, 4096, 16384));
    EXPECT_FALSE(thread_detail::compute_stack_size(16384, size_max, 0, 4096, 16384));
    EXPECT_EQ(thread_detail::compute_stack_size(size_max - 4095 - 5888 - 4287, 5888, 4287, 4096, 16384),
              size_max - 4095);
    EXPECT_FALSE(thread_detail::compute_stack_size(size_max - 4094 - 5888 - 4287, 5888, 4287, 4096, 16384));
}

// guard_fits: the guard is rounded up to a page (as glibc does) before it is
// added to the stack. With a 1 MiB stack the largest guard that fits is
// SIZE_MAX + 1 - 1 MiB - page; a guard of 1 byte does not fit above a stack
// of SIZE_MAX - 4095, although 1 <= 4095.
TEST(GuardFits, RoundedGuardPlusStack) {
    const std::size_t stack = std::size_t{1} << 20;
    EXPECT_TRUE(thread_detail::guard_fits(stack, 0, 4096));
    EXPECT_TRUE(thread_detail::guard_fits(stack, size_max - stack - 4095, 4096));
    EXPECT_FALSE(thread_detail::guard_fits(stack, size_max - stack - 4094, 4096));
    EXPECT_FALSE(thread_detail::guard_fits(0, size_max, 4096));
    EXPECT_TRUE(thread_detail::guard_fits(0, size_max - 4095, 4096));
    EXPECT_TRUE(thread_detail::guard_fits(size_max - 4096, 1, 4096));
    EXPECT_FALSE(thread_detail::guard_fits(size_max - 4095, 1, 4096));
    EXPECT_FALSE(thread_detail::guard_fits(std::size_t{1} << 63, std::size_t{1} << 63, 4096));
}

// tls_carve_bound: the size rounded up to the alignment plus alignment - 1
// (the descriptor is aligned down from the top of the block, leaving up to
// alignment - 1 bytes unused above it). Covers alignment 0 (treated as 1),
// an over-aligned block (4096, 8192), and saturation instead of a wrap.
TEST(TlsCarveBound, RoundedPlusSlack) {
    EXPECT_EQ(thread_detail::tls_carve_bound(0, 1), std::size_t{0});
    EXPECT_EQ(thread_detail::tls_carve_bound(0, 0), std::size_t{0});
    EXPECT_EQ(thread_detail::tls_carve_bound(5, 0), std::size_t{5});
    EXPECT_EQ(thread_detail::tls_carve_bound(4224, 64), std::size_t{4287});
    EXPECT_EQ(thread_detail::tls_carve_bound(4225, 64), std::size_t{4351});
    EXPECT_EQ(thread_detail::tls_carve_bound(1001, 256), std::size_t{1279});
    EXPECT_EQ(thread_detail::tls_carve_bound(1, 4096), std::size_t{8191});
    EXPECT_EQ(thread_detail::tls_carve_bound(8192, 8192), std::size_t{16383});
    EXPECT_EQ(thread_detail::tls_carve_bound(size_max - 127, 64), size_max - 64);
    EXPECT_EQ(thread_detail::tls_carve_bound(size_max, 64), size_max);
    EXPECT_EQ(thread_detail::tls_carve_bound(size_max - 62, 64), size_max);
    // With a power-of-two alignment the rounded size plus alignment - 1
    // always fits once the rounding fits; alignment 3 (not one glibc
    // reports) reaches the saturation of the addition itself: SIZE_MAX - 2
    // rounds to SIZE_MAX, and + 2 would wrap to 1.
    EXPECT_EQ(thread_detail::tls_carve_bound(size_max - 2, 3), size_max);
}

// tls_info_from and tls_size_from with a null query (the symbol absent from
// glibc) and with fake queries: the alignment 0 a query may report is read
// as 1; the size term is tls_carve_bound of what the query reports.
TEST(TlsQuery, NullAndFakeQueries) {
    const thread_detail::StaticTlsInfo none = thread_detail::tls_info_from(nullptr);
    EXPECT_EQ(none.size, std::size_t{0});
    EXPECT_EQ(none.align, std::size_t{1});
    EXPECT_EQ(thread_detail::tls_size_from(nullptr), std::size_t{0});

    const thread_detail::tls_query_fn odd = [](std::size_t* size, std::size_t* align) {
        *size = 1001;
        *align = 256;
    };
    EXPECT_EQ(thread_detail::tls_size_from(odd), std::size_t{1279});
    EXPECT_EQ(thread_detail::tls_info_from(odd).align, std::size_t{256});

    const thread_detail::tls_query_fn unaligned = [](std::size_t* size, std::size_t* align) {
        *size = 4224;
        *align = 0;
    };
    EXPECT_EQ(thread_detail::tls_info_from(unaligned).align, std::size_t{1});
    EXPECT_EQ(thread_detail::tls_size_from(unaligned), std::size_t{4224});

    const thread_detail::tls_query_fn over_aligned = [](std::size_t* size, std::size_t* align) {
        *size = 1;
        *align = 4096;
    };
    EXPECT_EQ(thread_detail::tls_size_from(over_aligned), std::size_t{8191});
}

// The cached machine parameters equal the system's own answers, and the TLS
// term includes this binary's 64 KiB thread_local (else the TLS test below
// tests nothing).
TEST(MachineParameters, MatchTheSystem) {
    EXPECT_EQ(thread_detail::page_size(), own_page());
    EXPECT_EQ(thread_detail::min_stack_size(), own_minimum());
    EXPECT_EQ(thread_detail::stack_granule(), own_granule());
    EXPECT_EQ(thread_detail::static_tls_size(), own_tls_term());
    const thread_detail::tls_query_fn query = &thread_detail::_dl_get_tls_static_info;
    ASSERT_NE(query, nullptr) << "glibc without the static TLS query";
    EXPECT_GT(own_tls_term(), sizeof big_tls);
}

// ===========================================================================
// AttrOwner: the sizes it passes to glibc.
// ===========================================================================

// The stack size in the attributes is exactly the documented computation of
// the request (or of the process default), with the test's own extra, TLS
// term, unit and minimum; the guard is passed as requested (1 MiB if unset),
// unrounded. Requests: unset, 0, 1, the minimum -1/0/+1, page multiples and
// odd sizes. An exact comparison: the read-back of a running thread's stack
// is only a lower bound, so a stack that is too large (the guard added in) is
// visible only here.
TEST(AttrOwnerSizes, ExactStackAndGuard) {
    const std::vector<std::optional<std::size_t>> requests = {
        std::nullopt,      std::size_t{0},     std::size_t{1},          own_minimum() - 1,
        own_minimum(),     own_minimum() + 1,  std::size_t{1} << 20,    (std::size_t{2} << 20) + 1,
        std::size_t{3} << 20, std::size_t{1234567}, std::size_t{1} << 62};
    const std::vector<std::optional<std::size_t>> guards = {std::nullopt, std::size_t{0}, std::size_t{1},
                                                            std::size_t{5000}, std::size_t{1} << 20};
    for (const std::optional<std::size_t>& request : requests) {
        for (const std::optional<std::size_t>& guard : guards) {
            SCOPED_TRACE("request " + (request ? std::to_string(*request) : std::string("unset")) + ", guard " +
                         (guard ? std::to_string(*guard) : std::string("unset")));
            const thread_detail::AttrOwner owner(request, guard);
            const std::size_t expected_stack = own_stack_size(request.value_or(Thread::default_stack_size()));
            const std::size_t expected_guard = guard.value_or(std::size_t{1} << 20);
            std::size_t stack = 0;
            std::size_t guard_read = 0;
            ASSERT_EQ(pthread_attr_getstacksize(&owner.get(), &stack), 0);
            ASSERT_EQ(pthread_attr_getguardsize(&owner.get(), &guard_read), 0);
            EXPECT_EQ(stack, expected_stack);
            EXPECT_EQ(owner.stack_size(), expected_stack);
            EXPECT_EQ(guard_read, expected_guard);
            EXPECT_EQ(owner.guard_size(), expected_guard);
            // Cross-check of the seam itself with the header's own parameters.
            EXPECT_EQ(stack, thread_detail::compute_stack_size(request.value_or(Thread::default_stack_size()),
                                                               thread_detail::stack_extra,
                                                               thread_detail::static_tls_size(),
                                                               thread_detail::stack_granule(),
                                                               thread_detail::min_stack_size()));
        } // for every guard
    } // for every request
}

// An unset request takes the process default as it is when the AttrOwner is
// made (an odd value, so the computation shows).
TEST(AttrOwnerSizes, UnsetUsesCurrentDefault) {
    const std::size_t odd_default = (std::size_t{3} << 20) + 12345;
    const DefaultStackScope scope(odd_default);
    const thread_detail::AttrOwner owner(std::nullopt, std::nullopt);
    std::size_t stack = 0;
    ASSERT_EQ(pthread_attr_getstacksize(&owner.get(), &stack), 0);
    EXPECT_EQ(stack, own_stack_size(odd_default));
}

// Every overflow throws EINVAL (generic category): the stack request
// (SIZE_MAX; SIZE_MAX - extra, where only the additions overflow), the guard
// (SIZE_MAX, and the smallest guard whose page round-up overflows), their sum
// (2^63 each: each fits alone), and a default of SIZE_MAX. The largest guard
// that fits above a 1 MiB request's stack does not throw; one byte more does.
TEST(AttrOwnerSizes, OverflowThrowsEinval) {
    using std::nullopt;
    const std::size_t page = own_page();
    EXPECT_EQ(system_error_code([] { thread_detail::AttrOwner owner(size_max, nullopt); }), EINVAL);
    EXPECT_EQ(system_error_code([] { thread_detail::AttrOwner owner(size_max - own_extra, nullopt); }), EINVAL);
    EXPECT_EQ(system_error_code([] { thread_detail::AttrOwner owner(nullopt, size_max); }), EINVAL);
    EXPECT_EQ(system_error_code([page] { thread_detail::AttrOwner owner(nullopt, size_max - page + 2); }), EINVAL);
    EXPECT_EQ(system_error_code([] {
                  thread_detail::AttrOwner owner(std::size_t{1} << 63, std::size_t{1} << 63);
              }),
              EINVAL);
    EXPECT_EQ(system_error_code([] { thread_detail::AttrOwner owner(std::size_t{1} << 63, nullopt); }), -1);
    EXPECT_EQ(system_error_code([] { thread_detail::AttrOwner owner(nullopt, std::size_t{1} << 63); }), -1);

    const std::size_t stack = own_stack_size(std::size_t{1} << 20);
    const std::size_t largest_guard = size_max - stack - page + 1;
    EXPECT_EQ(system_error_code([largest_guard] {
                  thread_detail::AttrOwner owner(std::size_t{1} << 20, largest_guard);
              }),
              -1);
    EXPECT_EQ(system_error_code([largest_guard] {
                  thread_detail::AttrOwner owner(std::size_t{1} << 20, largest_guard + 1);
              }),
              EINVAL);

    const DefaultStackScope scope(size_max);
    EXPECT_EQ(system_error_code([] { thread_detail::AttrOwner owner(nullopt, nullopt); }), EINVAL);
}

// ===========================================================================
// The start mask.
// ===========================================================================

// start_mask() blocks every signal of 1..SIGRTMAX except the seven, SIGKILL,
// SIGSTOP and glibc's internal signals. No thread involved.
TEST(StartMaskSet, Membership) {
    EXPECT_TRUE(mask_is(thread_detail::start_mask(), start_blocked));
}

// CreationMaskScope, from each creator mask: inside the scope the calling
// thread's mask is its previous mask plus the start mask's blocked signals,
// so the seven are as the creator had them (not all blocked, not all
// unblocked); saved() is the previous mask; after the scope the previous mask
// is back.
TEST(CreationMaskScopeTest, AddsBlockedPartKeepsSevenRestores) {
    for (const CreatorMask& creator : creator_masks()) {
        SCOPED_TRACE(std::string("creator mask: ") + creator.name);
        on_creator(creator.mask, [] {
            const sigset_t before = current_mask();
            const auto previous = [&before](int signo) { return is_blocked(before, signo); };
            {
                const thread_detail::CreationMaskScope scope;
                EXPECT_TRUE(mask_is(current_mask(), [&before](int signo) {
                    return is_blocked(before, signo) || start_blocked(signo);
                }));
                EXPECT_TRUE(mask_is(scope.saved(), previous));
            }
            EXPECT_TRUE(mask_is(current_mask(), previous));
        });
    } // for every creator mask
}

// create_thread: the new thread's mask on entry (its first instruction, before
// any of the class's code runs in it) is its creator's mask plus the start
// mask's blocked signals; the creator's mask is restored after a successful
// and after a failing pthread_create (stack 2^62: EAGAIN).
TEST(CreationMaskScopeTest, CreateThreadWindowAndRestore) {
    for (const CreatorMask& creator : creator_masks()) {
        SCOPED_TRACE(std::string("creator mask: ") + creator.name);
        on_creator(creator.mask, [] {
            const sigset_t before = current_mask();
            const auto previous = [&before](int signo) { return is_blocked(before, signo); };
            sigset_t entry;
            sigemptyset(&entry);
            const PlainAttr attr;
            pthread_t handle{};
            ASSERT_EQ(thread_detail::create_thread(handle, attr.get(), &record_entry_mask, &entry), 0);
            EXPECT_TRUE(mask_is(current_mask(), previous));
            pthread_join(handle, nullptr);
            EXPECT_TRUE(mask_is(entry, [&before](int signo) {
                return is_blocked(before, signo) || start_blocked(signo);
            }));

            const PlainAttr huge(std::size_t{1} << 62);
            pthread_t failed{};
            EXPECT_EQ(thread_detail::create_thread(failed, huge.get(), &do_nothing, nullptr), EAGAIN);
            EXPECT_TRUE(mask_is(current_mask(), previous));
        });
    } // for every creator mask
}

// start_routine, called directly on a helper thread with each creator mask
// (no pthread_create, so nothing is inherited from a creation window): the
// callable sees exactly the start mask, so the routine SETS it (an added
// block would leave a creator-blocked signal of the seven blocked).
TEST(StartRoutineDirect, SetsStartMask) {
    for (const CreatorMask& creator : creator_masks()) {
        SCOPED_TRACE(std::string("creator mask: ") + creator.name);
        sigset_t seen;
        sigemptyset(&seen);
        on_creator(creator.mask, [&seen] {
            using S = thread_detail::State<MaskRecorder, sigset_t*>;
            call_start_routine(new S(0, MaskRecorder{}, &seen));
        });
        EXPECT_TRUE(mask_is(seen, start_blocked));
    } // for every creator mask
}

// Through Thread, from each creator mask: the callable sees the start mask,
// and the creator's mask is unchanged after a successful construction and
// after a failing one (EAGAIN).
TEST(ThreadMask, CallableSeesStartMaskCreatorUnchanged) {
    for (const CreatorMask& creator : creator_masks()) {
        SCOPED_TRACE(std::string("creator mask: ") + creator.name);
        sigset_t seen;
        sigemptyset(&seen);
        on_creator(creator.mask, [&seen] {
            const sigset_t before = current_mask();
            const auto previous = [&before](int signo) { return is_blocked(before, signo); };
            {
                const Thread worker([&seen] { seen = current_mask(); });
                EXPECT_TRUE(mask_is(current_mask(), previous));
            }
            EXPECT_EQ(system_error_code([] { const Thread worker({.stack_size = std::size_t{1} << 62}, [] {}); }),
                      EAGAIN);
            EXPECT_TRUE(mask_is(current_mask(), previous));
        });
        EXPECT_TRUE(mask_is(seen, start_blocked));
    } // for every creator mask
}

// ===========================================================================
// The stack stagger.
// ===========================================================================

// Calls start_routine directly with a RecordFrame state of `stagger` bytes
// and returns the record, its stack top replaced by this function's own frame
// address: a reference point at a fixed distance above the start routine's
// frame, wherever this function is called from (the caller's stack pointer at
// each call may differ, e.g. in an unrolled loop).
[[gnu::noinline]] FrameRecord record_direct(std::size_t stagger) {
    using S = thread_detail::State<RecordFrame, FrameRecord*, std::latch*>;
    FrameRecord record;
    call_start_routine(new S(stagger, RecordFrame{}, &record, no_hold));
    record.stack = {.low = reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0)), .size = 0};
    return record;
} // record_direct()

// start_routine, called directly on one helper thread with each of the 16
// stagger amounts: the callable's frame lies exactly the stagger amount
// deeper each time. The reservation is kept and the callable runs below it.
TEST(StartRoutineDirect, ReservationLowersCallableFrame) {
    std::vector<FrameRecord> records(16);
    std::thread helper([&records] {
        for (unsigned n = 0; n != 16; ++n) records[n] = record_direct(own_stagger(n));
    });
    helper.join();
    EXPECT_TRUE(depths_follow_stagger(records, 0));
}

// 16 Threads alive at once (distinct stack blocks): the counter advances by
// 16, and each callable frame lies its stagger amount deeper than thread 0's
// position predicts.
TEST(StaggerDepth, SixteenAliveAtOnce) {
    const CounterSaver saver;
    const std::vector<Thread::Attributes> attributes(16, Thread::Attributes{.stack_size = std::size_t{1} << 18});
    const std::vector<FrameRecord> records = place_alive(attributes, 100);
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 116u);
    EXPECT_TRUE(depths_follow_stagger(records, 100));
}

// 16 Threads created and joined one at a time: glibc hands the same cached
// block back each time, and the stagger alone moves the frame.
TEST(StaggerDepth, CreateAndJoinReusesBlock) {
    const CounterSaver saver;
    thread_detail::stagger_counter.store(5, std::memory_order_relaxed);
    std::vector<FrameRecord> records(16);
    for (FrameRecord& record : records) {
        const Thread worker({.stack_size = std::size_t{1} << 16, .guard_size = 4096}, RecordFrame{}, &record,
                            no_hold);
    }
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 21u);
    EXPECT_TRUE(depths_follow_stagger(records, 5));
}

// Mixed stack sizes (16 KiB to 3 MiB: fresh and cached blocks) and guards
// (0, a page, 1 MiB), alive at once. The depth is measured from each
// thread's own stack top, so it does not depend on the sizes.
TEST(StaggerDepth, MixedSizesAndGuards) {
    const CounterSaver saver;
    const std::array<std::size_t, 4> sizes = {std::size_t{16} << 10, std::size_t{64} << 10, std::size_t{2} << 20,
                                              std::size_t{3} << 20};
    const std::array<std::size_t, 3> guards = {0, 4096, std::size_t{1} << 20};
    std::vector<Thread::Attributes> attributes;
    for (std::size_t i = 0; i != 16; ++i) {
        attributes.push_back({.stack_size = sizes[i % sizes.size()], .guard_size = guards[i % guards.size()]});
    }
    const std::vector<FrameRecord> records = place_alive(attributes, 0);
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 16u);
    EXPECT_TRUE(depths_follow_stagger(records, 0));
}

// The counter's wrap: numbers UINT_MAX - 7 .. UINT_MAX, then 0 .. 7. The
// rotation continues (slots 8..15, then 0..7).
TEST(StaggerDepth, CounterWrap) {
    const CounterSaver saver;
    const std::vector<Thread::Attributes> attributes(16, Thread::Attributes{.stack_size = std::size_t{1} << 16});
    const std::vector<FrameRecord> records = place_alive(attributes, UINT_MAX - 7);
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 8u);
    EXPECT_TRUE(depths_follow_stagger(records, UINT_MAX - 7));
}

// What the stagger is for: at the default stack size, 16 consecutive
// Threads alive at once have their callable frames at 16 distinct page
// offsets, and frame + stagger has one page offset (every stack top has the
// same page offset, since every size is a multiple of the page).
TEST(StaggerPlacement, DefaultSizeSixteenPageOffsets) {
    const CounterSaver saver;
    const std::vector<Thread::Attributes> attributes(16, Thread::Attributes{});
    const std::vector<FrameRecord> records = place_alive(attributes, 0);
    EXPECT_EQ(distinct_page_offsets(records), std::size_t{16});
    EXPECT_EQ(distinct_unstaggered_offsets(records, 0), std::size_t{1});
}

// Requests of 2 MiB and more that differ modulo the page (2 MiB + i * 100):
// the stack sizes are rounded to the page, so the stack tops still share one
// page offset and the stagger alone decides each frame's offset. (Sizes of
// 2 MiB and more, so that no sanitizer raises them and no smaller block from
// another test is reused.)
TEST(StaggerPlacement, RequestsDifferingModuloPage) {
    const CounterSaver saver;
    std::vector<Thread::Attributes> attributes;
    for (std::size_t i = 0; i != 16; ++i) attributes.push_back({.stack_size = (std::size_t{2} << 20) + i*100});
    const std::vector<FrameRecord> records = place_alive(attributes, 0);
    EXPECT_EQ(distinct_page_offsets(records), std::size_t{16});
    EXPECT_EQ(distinct_unstaggered_offsets(records, 0), std::size_t{1});
}

// The stack below the callable's first frame at the deepest stagger slot
// (15), for every rounding slack granule - 64, ..., 64, 0: the request is
// chosen so that request + extra + TLS term + slack is one block size, so at
// slack 0 nothing of the rounding is left over. glibc reuses a cached block
// that is larger than the request, which would hide a shortfall, so the test
// checks that each thread's stack is exactly the block and fails otherwise,
// and avoids such reuse: the block is larger than any other stack of this
// binary, and the slack decreases, so the requests increase and no earlier
// block of the sweep is larger than a later request's (whatever sizes a
// defective computation gives them). The callable is the
// short-argument-list RecordFrame.
TEST(StaggerHeadroom, DeepestSlotEverySlack) {
    const CounterSaver saver;
    pthread_attr_t defaults;
    ASSERT_EQ(pthread_getattr_default_np(&defaults), 0);
    std::size_t default_size = 0;
    pthread_attr_getstacksize(&defaults, &default_size);
    pthread_attr_destroy(&defaults);
    const std::size_t granule = own_granule();
    std::size_t block = std::size_t{16} << 20;
    while (block <= 2*default_size) block *= 2;
    for (std::size_t slack = granule - 64; slack < granule; slack -= 64) {
        SCOPED_TRACE("slack " + std::to_string(slack));
        const std::size_t request = block - own_extra - own_tls_term() - slack;
        ASSERT_EQ(own_stack_size(request), block);
        thread_detail::stagger_counter.store(15, std::memory_order_relaxed);
        FrameRecord record;
        {
            const Thread worker({.stack_size = request}, RecordFrame{}, &record, no_hold);
        }
        EXPECT_EQ(record.stack.size, block) << "the thread did not get a block of exactly the computed size";
        EXPECT_GE(record.frame - record.stack.low, request);
    } // for every slack, down to 0 (the unsigned wrap below 0 ends the loop)
}

// The TLS term matters: a minimum stack (request 0) in this binary, whose
// static TLS block holds a 64 KiB thread_local, is created and has at least
// the minimum below the callable's first frame. Without the TLS term the
// stack would be smaller than the TLS block and glibc would refuse it.
TEST(TlsTerm, MinimumStackWithLargeThreadLocal) {
#if defined(THREAD_TEST_TSAN)
    GTEST_SKIP() << "ThreadSanitizer raises small stack requests above the static TLS size, so a request "
                    "without the TLS term would be created too: the test cannot fail in this build";
#endif
    std::uintptr_t frame = 0;
    StackSpan stack;
    EXPECT_EQ(system_error_code([&frame, &stack] {
                  const Thread worker({.stack_size = 0}, [&frame, &stack] {
                      big_tls[1] = 1;
                      frame = reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0));
                      stack = own_stack();
                  });
              }),
              -1);
    EXPECT_GE(frame - stack.low, own_minimum());
}

// ===========================================================================
// Construction failure paths and the copies.
// ===========================================================================

// An overflow (stack, guard, their sum, a SIZE_MAX default) throws EINVAL
// before step 3: no copy of the callable or the argument is made, the rvalues
// are not moved from, no creation number is taken, and no heap is left
// allocated (sanitizer builds).
TEST(ThreadOverflow, NothingCopiedNoNumberTaken) {
    const CounterSaver saver;
    const std::vector<Thread::Attributes> overflowing = {
        {.stack_size = size_max},
        {.guard_size = size_max},
        {.stack_size = std::size_t{1} << 63, .guard_size = std::size_t{1} << 63}};
    for (const Thread::Attributes& attributes : overflowing) {
        CopyCounts counts;
        TrackedCallable callable{Tracked(&counts)};
        Tracked argument(&counts);
        thread_detail::stagger_counter.store(9, std::memory_order_relaxed);
        EXPECT_EQ(system_error_code([&] { const Thread worker(attributes, std::move(callable), std::move(argument)); }),
                  EINVAL);
        EXPECT_EQ(counts.copies.load(), 0);
        EXPECT_FALSE(callable.marker.moved_from);
        EXPECT_FALSE(argument.moved_from);
        EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 9u);
        const std::optional<long long> delta = heap_delta([&attributes] {
            static_cast<void>(system_error_code([&attributes] { const Thread worker(attributes, [] {}); }));
        });
        if (delta) {
            EXPECT_EQ(*delta, 0);
        }
    } // for every overflowing request

    const DefaultStackScope scope(size_max);
    CopyCounts counts;
    Tracked argument(&counts);
    EXPECT_EQ(system_error_code([&] { const Thread worker(TrackedCallable{Tracked(&counts)}, std::move(argument)); }),
              EINVAL);
    EXPECT_EQ(counts.copies.load(), 0);
    EXPECT_FALSE(argument.moved_from);
}

// A failing pthread_create (stack 2^62: EAGAIN): the copies were made (one of
// the callable, one of the argument), and are destroyed by the creating
// thread before the exception reaches the caller; the rvalue argument is
// moved-from; one creation number was taken; no heap is left allocated.
TEST(ThreadCreateFailure, CopiesDestroyedByCreator) {
    const CounterSaver saver;
    CopyCounts counts;
    Tracked argument(&counts);
    thread_detail::stagger_counter.store(9, std::memory_order_relaxed);
    try {
        const Thread worker({.stack_size = std::size_t{1} << 62}, TrackedCallable{Tracked(&counts)},
                            std::move(argument));
        ADD_FAILURE() << "pthread_create with a 2^62-byte stack succeeded";
    } catch (const std::system_error& e) {
        EXPECT_EQ(e.code().value(), EAGAIN);
        EXPECT_EQ(counts.copies.load(), 2);
        EXPECT_EQ(counts.copies_destroyed.load(), 2);
        EXPECT_EQ(counts.destroyed_by_creator.load(), 2);
    }
    EXPECT_TRUE(argument.moved_from);
    EXPECT_EQ(thread_detail::stagger_counter.load(std::memory_order_relaxed), 10u);
    const std::optional<long long> delta = heap_delta([] {
        static_cast<void>(system_error_code([] { const Thread worker({.stack_size = std::size_t{1} << 62}, [] {}); }));
    });
    if (delta) {
        EXPECT_EQ(*delta, 0);
    }
}

// A pthread_create that fails after glibc stored the new thread's handle (the
// clone fails: RLIMIT_NPROC 0) leaves the handle member reset: read from the
// bytes of the storage after the throw. Both starting constructors. (A
// constructor that delegated to another one would run ~Thread() on that
// handle.) Skipped when the process limit does not stop creation (a
// privileged process) or glibc leaves the handle untouched (then there is
// nothing to reset).
TEST(ThreadCreateFailure, HandleResetAfterFailedClone) {
    std::thread([] {}).join();  // the sanitizer runtimes' own threads exist before the limit
    const NoNewThreadsScope limit;
    ASSERT_TRUE(limit.lowered());
    pthread_t raw{};
    const int raw_rc = pthread_create(&raw, nullptr, &do_nothing, nullptr);
    if (raw_rc == 0) {
        pthread_join(raw, nullptr);
        GTEST_SKIP() << "RLIMIT_NPROC does not stop thread creation in this process";
    }
    ASSERT_EQ(raw_rc, EAGAIN);
    if (raw == pthread_t{}) GTEST_SKIP() << "glibc does not store the handle before a failing clone";

    for (const bool with_attributes : {false, true}) {
        SCOPED_TRACE(with_attributes ? "Thread(attributes, f)" : "Thread(f)");
        alignas(Thread) unsigned char storage[sizeof(Thread)];
        std::memset(storage, 0xA5, sizeof storage);
        int code = -1;
        try {
            if (with_attributes) {
                ::new (static_cast<void*>(storage)) Thread(Thread::Attributes{}, [] {});
            } else {
                ::new (static_cast<void*>(storage)) Thread([] {});
            }
            ADD_FAILURE() << "the construction succeeded";
            std::launder(reinterpret_cast<Thread*>(storage))->~Thread();
        } catch (const std::system_error& e) {
            code = e.code().value();
        }
        EXPECT_EQ(code, EAGAIN);
        pthread_t left;
        std::memcpy(&left, storage, sizeof left);
        EXPECT_EQ(left, pthread_t{}) << "the handle was not reset";
    } // for both constructors
}

// The normal path: the callable and the argument are each copied once, and
// both copies are destroyed once, by the new thread, before join() returns.
TEST(ThreadCopies, DestroyedOnceByNewThread) {
    CopyCounts counts;
    {
        const Thread worker(TrackedCallable{Tracked(&counts)}, Tracked(&counts));
    }
    EXPECT_EQ(counts.copies.load(), 2);
    EXPECT_EQ(counts.copies_destroyed.load(), 2);
    EXPECT_EQ(counts.destroyed_by_creator.load(), 0);
}

// pthread_exit in the callable: the stack unwinds through the start routine
// (which rethrows the forced unwind), the callable's local and both copies
// are destroyed by the new thread, and the thread ends normally (join
// returns). In a death-test child, because a start routine that swallowed
// the forced unwind would end the process.
TEST(ThreadCopiesDeathTest, ThreadExitDestroysCopies) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            CopyCounts counts;
            locals_destroyed = 0;
            {
                const Thread worker(TrackedCallable{Tracked(&counts), Ending::thread_exit}, Tracked(&counts));
            }
            int code = 0;
            if (counts.copies.load() != 2) code = 3;
            else if (counts.copies_destroyed.load() != 2) code = 4;
            else if (counts.destroyed_by_creator.load() != 0) code = 5;
            else if (locals_destroyed.load() != 1) code = 6;
            std::_Exit(code);
        },
        testing::ExitedWithCode(0), "");
}

namespace {

// The counts the terminate handler of the next test checks.
CopyCounts* terminate_counts = nullptr;

// The terminate handler of the next test: exit code 42 if the exception is
// current, the callable's local was unwound and the two copies are alive;
// other codes say which check failed.
[[noreturn]] void check_at_terminate() {
    if (std::current_exception() == nullptr) std::_Exit(3);
    if (locals_destroyed.load() != 1) std::_Exit(4);
    if (terminate_counts->copies.load() != 2) std::_Exit(5);
    if (terminate_counts->copies_destroyed.load() != 0) std::_Exit(6);
    std::_Exit(42);
} // check_at_terminate()

} // namespace

// An exception escaping the callable: the callable's frame is unwound, then
// std::terminate runs with the exception current, and the copies are not
// destroyed (the state is never released on this path).
TEST(ThreadCopiesDeathTest, ExceptionTerminatesWithCopiesAlive) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            static CopyCounts counts;
            terminate_counts = &counts;
            locals_destroyed = 0;
            std::set_terminate(&check_at_terminate);
            const Thread worker(TrackedCallable{Tracked(&counts), Ending::exception}, Tracked(&counts));
            std::this_thread::sleep_for(std::chrono::seconds(10));
            std::_Exit(7);
        },
        testing::ExitedWithCode(42), "");
}

// ===========================================================================
// The invoker.
// ===========================================================================

// A callable whose overloads say how it was called: 1 for an rvalue callable
// with an rvalue argument, the call std::thread makes.
struct ValueCategoryProbe {
    int* result;

    void operator()(int&&) && { *result = 1; }
    void operator()(int&) && { *result = 2; }
    void operator()(int&&) & { *result = 3; }
    void operator()(int&) & { *result = 4; }
}; // struct ValueCategoryProbe

// invoke_state calls the callable as an rvalue with the arguments as rvalues.
TEST(InvokeState, RvalueCallableRvalueArguments) {
    int result = 0;
    using S = thread_detail::State<ValueCategoryProbe, int>;
    static_assert(S::arity == 1);
    S state(256, ValueCategoryProbe{&result}, 7);
    EXPECT_EQ(state.stagger, std::size_t{256});
    thread_detail::invoke_state(state, std::make_index_sequence<S::arity>{});
    EXPECT_EQ(result, 1);
}

// A class with a member function, for the member-pointer invocations.
struct Accumulator {
    int value = 0;

    void add(int amount) { value += amount; }
}; // struct Accumulator

// invoke_state dispatches a member-function pointer on an object pointer and
// on a reference_wrapper (std::invoke's rules).
TEST(InvokeState, MemberPointers) {
    Accumulator target;
    using ByPointer = thread_detail::State<void (Accumulator::*)(int), Accumulator*, int>;
    ByPointer by_pointer(0, &Accumulator::add, &target, 5);
    thread_detail::invoke_state(by_pointer, std::make_index_sequence<ByPointer::arity>{});
    EXPECT_EQ(target.value, 5);
    using ByReference =
        thread_detail::State<void (Accumulator::*)(int), std::reference_wrapper<Accumulator>, int>;
    ByReference by_reference(0, &Accumulator::add, std::ref(target), 6);
    thread_detail::invoke_state(by_reference, std::make_index_sequence<ByReference::arity>{});
    EXPECT_EQ(target.value, 11);
}
