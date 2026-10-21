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
#ifndef INCLUDED_THREAD_H
#define INCLUDED_THREAD_H

// Thread: a thread used like std::jthread (constructed from a callable and
// its arguments, launched by the constructor, joined by the destructor) that
// also sets what std::thread cannot: the stack size and the guard size,
// chosen by the caller, and the signal mask the new thread starts with,
// which is fixed (THE START MASK below). The destructor joins (there is no
// stop request: a callable that loops needs its own flag). An exception that
// escapes the callable unwinds the thread's stack and then calls
// std::terminate() with the exception current.
//
// Requirements. Linux with glibc, and libstdc++: the class creates threads
// with pthread_create and glibc's thread attributes, adds glibc's static TLS
// size to the stack it requests, and builds get_id() from libstdc++'s
// std::thread::id(native_handle_type) constructor. Another C library or
// standard library fails the build with an #error naming the requirement.
//
// Overall flow of a construction.
//   1. The stack and guard sizes are computed from the attributes and the
//      process defaults, and checked for overflow (size_t arithmetic); an
//      overflow throws EINVAL before anything is allocated or copied.
//   2. A pthread_attr_t is set up with those sizes (thread_detail::AttrOwner).
//   3. The callable and the arguments are decay-copied into one heap
//      allocation (thread_detail::State), together with this thread's stagger
//      amount (below).
//   4. The creating thread blocks the start mask's signals (adding to its own
//      mask) for the duration of pthread_create (thread_detail::
//      CreationMaskScope), and restores its own mask afterwards on every path.
//   5. The new thread's first action sets its mask to the start mask; it then
//      takes ownership of the heap state, reserves its stagger amount at the
//      top of its stack, and calls the callable through a function that is
//      never inlined, so the callable's frames lie below the reservation.
//
// THE START MASK. Every Thread starts with one fixed mask, whatever the
// creating thread's mask: every blockable signal is blocked except SIGSEGV,
// SIGBUS, SIGFPE, SIGILL, SIGPROF, SIGTRAP and SIGSYS, which are unblocked.
// Why: a process-directed signal (Ctrl-C, kill, a timer) is delivered to some
// thread that does not block it, so worker threads that block such signals
// leave their delivery to the threads the application keeps unblocked
// (normally the main thread). The seven are synchronous (raised by the
// faulting thread itself) or profiling signals: a fault in the callable must
// reach its handler (a blocked synchronous signal raised by the kernel is
// delivered with the default action, bypassing any handler), and SIGPROF
// must reach the thread so that a profiler sampling the process samples the
// thread; so the seven are unblocked even if the creating thread blocked
// them. There is no API to change the mask a Thread starts with; a callable
// that needs a signal unblocks it itself.
// Consequences for the user: a process-directed signal reaches only a thread
// that leaves it unblocked; a signal aimed at a worker (raise, pthread_kill)
// stays pending until the worker unblocks it; threads and processes a worker
// creates inherit its mask unless they reset it.
//
// Why both sides set the mask (creator blocks, new thread sets): the new
// thread inherits its creator's mask at pthread_create, so blocking the start
// mask's signals in the creator first means every signal outside the seven is
// blocked from the new thread's first instruction (no window in which a
// signal can be delivered to it with its creator's mask). The creator only
// ADDS blocks, and leaves the seven as it had them, so a fault in the creator
// during pthread_create still reaches the creator's handler and a profiler
// keeps sampling it. The new thread's first action then sets the exact start
// mask, which unblocks any of the seven its creator had blocked. During
// pthread_create the creating thread still receives at once any of the
// seven it has unblocked. Any other signal aimed at the creating thread in
// that window stays pending and is delivered to it when its mask is
// restored; any other process-directed signal goes to another thread that
// leaves it unblocked, or stays pending for the process until one does.
// pthread_attr_setsigmask_np is not used: AddressSanitizer's pthread_create
// interceptor discards it.
//
// THE STACK STAGGER. With page-multiple stack sizes every thread's top frame
// has the same address modulo the page size, wherever its stack lies, so the
// hot top-of-stack cache lines of many threads compete for the same cache
// sets. Each Thread therefore reserves (n % 16) * 256 bytes at the top of
// its stack before the callable runs, n being a process-wide creation count:
// consecutive threads' top frames start at 16 different page offsets, 256
// bytes apart. The effect on caching is reasoned from the cache geometry,
// not promised.

// Included first: any libstdc++ header defines __GLIBCXX__, and <cstdlib>
// (glibc's <stdlib.h>, through <features.h>) defines __GLIBC__, so the
// requirement checks run before any header that only glibc provides.
#include <cstddef>
#include <cstdlib>

// The rest of the header is compiled only when both requirements hold, so
// that an unsupported library reports the requirement and nothing else
// (compilers continue after #error).
#if !defined(__GLIBC__)
#error "thread.h requires Linux with glibc (pthread attributes, static TLS query)"
#elif !defined(__GLIBCXX__)
#error "thread.h requires libstdc++ (std::thread::id(native_handle_type))"
#else // the requirements hold

#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <cxxabi.h>                     // abi::__forced_unwind

#include <array>
#include <atomic>
#include <cerrno>
#include <concepts>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

// THREAD_RARELY(c) (header-private, #undef'd at the end): the condition `c`,
// marked as almost never true, for the error paths: a failed pthread call or
// a size overflow. With GCC and Clang this is
//   __builtin_expect_with_probability(!!(c), 0, 0.9999);
// elsewhere the hint is dropped and the code is still correct, only without
// the layout benefit. The !! makes the builtin accept exactly what the bare
// condition accepts. The builtin rather than the standard [[unlikely]]
// attribute: the standard leaves the attribute's strength to the compiler,
// while the builtin states the edge's weight.
#if defined(__GNUC__) || defined(__clang__)
#define THREAD_RARELY(c) __builtin_expect_with_probability(!!(c), 0, 0.9999)
#else
#define THREAD_RARELY(c) (c)
#endif

class Thread;

// Internals of Thread. Not part of the public interface, but every name here
// is readable by tests (the white-box tests use them as seams); none of them
// is meant for use by other code.
namespace thread_detail {

// glibc's query of the static TLS layout (size and alignment of the block
// that every thread's TLS and thread descriptor occupy at the top of its
// stack). A GLIBC_PRIVATE symbol of the dynamic linker, with no stability
// promise: declared weak, so that a glibc without it links and runs, and the
// address is then null (tls_size_from() treats that as "no static TLS").
extern "C" void _dl_get_tls_static_info(std::size_t* size, std::size_t* align) __attribute__((weak));

// The type of the query above; tls_size_from() takes the function as an
// argument so that a test can pass null or a fake.
using tls_query_fn = void (*)(std::size_t*, std::size_t*);

// ---------------------------------------------------------------------------
// Size arithmetic. Pure, constexpr and noexcept, with the machine parameters
// (page size, minimum stack) as arguments, so that other page sizes and
// minimums are testable at compile time.
// ---------------------------------------------------------------------------

// The stagger: thread number n reserves stagger_for(n) bytes at the top of
// its stack. stagger_step bytes (four 64-byte cache lines) per position,
// stagger_slots positions: 16 positions of 256 bytes span 4096 bytes, so
// with 4 KiB pages consecutive threads' top frames start at 16 different
// page offsets.
inline constexpr std::size_t stagger_step = 256;
inline constexpr unsigned stagger_slots = 16;
inline constexpr std::size_t max_stagger = (stagger_slots - 1)*stagger_step;  // 3840

// Room for the frames a thread's start-up puts on the stack before the
// callable's first frame, besides the stagger reservation and the static
// TLS; without this allowance they would come out of the requested size.
// Two parts:
//   - ABOVE the reservation: glibc's own thread-start frame, in sanitizer
//     builds the sanitizer runtime's thread-start frame, start_routine()'s
//     fixed frame, and under AddressSanitizer the redzone it puts above the
//     alloca;
//   - BELOW the reservation: under AddressSanitizer the redzone it puts
//     below the alloca, invoke_state()'s frame, which in unoptimized builds
//     also holds the forwarded arguments, and for a member-pointer callable
//     std::invoke's forwarding frames.
// With ordinary argument lists the part above is the larger one.
// Part of `stack_extra`, so that the callable gets at least the requested
// size below its first frame even at the deepest stagger slot. Sized for the
// unoptimized sanitizer builds, whose frames are the largest; an optimized
// build needs a fraction of it. Measuring only the part below the
// reservation understates what the allowance must cover.
inline constexpr std::size_t start_frames_allowance = 2048;

// What compute_stack_size() adds to the requested size: the deepest stagger
// reservation and the start-up frames (start_frames_allowance).
inline constexpr std::size_t stack_extra = max_stagger + start_frames_allowance;

// The rule's two invariants: the positions span one 4 KiB page exactly, and
// the creation counter's wrap (from UINT_MAX to 0) continues the rotation.
static_assert(stagger_slots*stagger_step == 4096, "the 16 stagger positions span one 4 KiB page");
static_assert(std::numeric_limits<unsigned>::max() % stagger_slots == stagger_slots - 1,
              "the creation counter's wrap must continue the stagger rotation");

// Returns the stagger amount of thread number n: (n % 16) * 256. The counter
// that hands out n is unsigned and wraps at 2^32, a multiple of 16, so the
// rotation continues across the wrap. nothrow.
constexpr std::size_t stagger_for(unsigned n) noexcept {
    return (n % stagger_slots)*stagger_step;
} // stagger_for()

// Returns `n` rounded up to a multiple of `granule`, or an empty optional if
// that multiple does not fit in size_t. Precondition: granule > 0. Exact for
// a granule that is a power of two (every granule the class uses is: page
// sizes and TLS alignments); for another granule the overflow check is
// conservative and can report an overflow for an `n` whose multiple would
// still fit. nothrow.
constexpr std::optional<std::size_t> page_round(std::size_t n, std::size_t granule) noexcept {
    if (n > std::numeric_limits<std::size_t>::max() - (granule - 1)) return {};
    return (n + (granule - 1))/granule*granule;
} // page_round()

// Returns the stack size the class passes to pthread_attr_setstacksize:
//   page_round(max(requested, minimum) + extra + tls, granule),
// or an empty optional if any step overflows size_t. Precondition:
// granule > 0; exact for a power of two (see page_round()). nothrow. The
// terms:
//   - requested: the stack size the caller asked for (or the default);
//   - minimum: the smallest stack glibc accepts (min_stack_size()); the clamp
//     applies to the request BEFORE the additions, so the additions are never
//     absorbed by the clamp: a request below the minimum still gets the extra
//     and the TLS on top of the minimum;
//   - extra: stack_extra, the deepest stagger reservation plus the start-up
//     frames above and below it (start_frames_allowance);
//   - tls: what glibc carves out of the requested stack for the static TLS
//     and the thread descriptor (static_tls_size()); glibc places them at the
//     top of the stack it maps, inside the requested size;
//   - granule: the page size (or the static TLS alignment when that is
//     larger, stack_granule()). The rounding keeps every thread's stack top
//     at the same offset modulo the page size, so the stagger alone decides
//     where a thread's top frame lies within its page.
// The guard is not part of the sum: glibc maps the guard in addition to the
// requested stack.
constexpr std::optional<std::size_t> compute_stack_size(std::size_t requested, std::size_t extra, std::size_t tls,
                                                        std::size_t granule, std::size_t minimum) noexcept {
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    const std::size_t clamped = requested < minimum ? minimum : requested;
    if (extra > max - clamped) return {};
    const std::size_t with_extra = clamped + extra;
    if (tls > max - with_extra) return {};
    return page_round(with_extra + tls, granule);
} // compute_stack_size()

// Returns true if the guard of `guard` bytes, rounded up to `page` the way
// glibc rounds it, fits in size_t together with a stack of `stack` bytes
// (glibc maps the two as one block). Precondition: page > 0; exact for a
// power of two (see page_round()). nothrow.
constexpr bool guard_fits(std::size_t stack, std::size_t guard, std::size_t page) noexcept {
    const std::optional<std::size_t> rounded = page_round(guard, page);
    return rounded && *rounded <= std::numeric_limits<std::size_t>::max() - stack;
} // guard_fits()

// Returns an upper bound of the bytes glibc carves out of the top of a
// thread's stack for the static TLS block and the thread descriptor, given
// the static TLS size and alignment glibc reports: the size rounded up to the
// alignment (glibc subtracts the rounded size), plus alignment - 1 (glibc
// aligns the thread descriptor down to the alignment from the top of the
// block, which leaves less than one alignment unit unused above it). The
// rounded size alone is not enough: whenever the static TLS alignment exceeds
// the thread descriptor's own alignment (an over-aligned thread_local), the
// unused part above the descriptor comes out of the requested stack too. An
// alignment of 0 is treated as 1. Exact for an alignment that is a power of
// two (glibc's always are); for another alignment the rounding's overflow
// check is conservative (see page_round()). Saturates at SIZE_MAX instead of
// wrapping, so that compute_stack_size() reports the overflow. nothrow.
constexpr std::size_t tls_carve_bound(std::size_t size, std::size_t align) noexcept {
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    if (align == 0) align = 1;
    const std::optional<std::size_t> rounded = page_round(size, align);
    if (!rounded || *rounded > max - (align - 1)) return max;
    return *rounded + (align - 1);
} // tls_carve_bound()

// ---------------------------------------------------------------------------
// Machine parameters, each read once and cached in a function-local static
// (initialized on first use, so safe whatever the order of static
// initialization; thread-safe by the language's rules for such statics).
// ---------------------------------------------------------------------------

// The static TLS size and alignment glibc reports.
struct StaticTlsInfo {
    std::size_t size;           // bytes of the static TLS block, the thread descriptor included
    std::size_t align;          // its alignment; 1 when unknown
}; // struct StaticTlsInfo

// Returns what `query` reports, or {0, 1} if `query` is null (the symbol is
// absent from this glibc). nothrow.
inline StaticTlsInfo tls_info_from(tls_query_fn query) noexcept {
    StaticTlsInfo info{0, 1};
    if (query == nullptr) return info;
    query(&info.size, &info.align);
    if (info.align == 0) info.align = 1;
    return info;
} // tls_info_from()

// Returns the bytes the class adds to a stack request for the static TLS:
// tls_carve_bound() of what `query` reports; 0 if `query` is null (then
// tls_info_from() gives {0, 1}, and tls_carve_bound(0, 1) is 0). nothrow.
inline std::size_t tls_size_from(tls_query_fn query) noexcept {
    const StaticTlsInfo info = tls_info_from(query);
    return tls_carve_bound(info.size, info.align);
} // tls_size_from()

// tls_size_from(&_dl_get_tls_static_info), cached. The reported size is
// fixed before main(): it covers the static TLS of the executable and of the
// libraries loaded at startup, plus glibc's surplus. A library loaded later
// (dlopen) with initial-exec TLS is placed in that surplus, inside the
// reported block, and the reported size does not change; a library that
// does not fit in the surplus fails to load. So the cached value never goes
// stale.
// nothrow.
inline std::size_t static_tls_size() noexcept {
    static const std::size_t size = tls_size_from(&_dl_get_tls_static_info);
    return size;
} // static_tls_size()

// The page size, cached. nothrow.
inline std::size_t page_size() noexcept {
    static const std::size_t size = [] {
        const long page = sysconf(_SC_PAGESIZE);
        return page > 0 ? static_cast<std::size_t>(page) : std::size_t{4096};
    }();
    return size;
} // page_size()

// The smallest stack glibc accepts, sysconf(_SC_THREAD_STACK_MIN), cached.
// No fallback value: PTHREAD_STACK_MIN, the obvious one, expands to this
// same sysconf call on glibc with _GNU_SOURCE, so it cannot answer when the
// call fails. Were the call to fail, its -1 converts to SIZE_MAX, and every
// construction then fails with EINVAL (the stack computation overflows)
// instead of creating a thread with a stack that is too small. nothrow.
inline std::size_t min_stack_size() noexcept {
    static const std::size_t size = static_cast<std::size_t>(sysconf(_SC_THREAD_STACK_MIN));
    return size;
} // min_stack_size()

// The unit compute_stack_size() rounds the stack size to: the page size, or
// the static TLS alignment if that is larger (glibc truncates the requested
// stack size down to a multiple of the static TLS alignment, so a size that
// is not such a multiple would lose up to one alignment unit). Both are
// powers of two, so the result is a multiple of the page size. Cached.
// nothrow.
inline std::size_t stack_granule() noexcept {
    static const std::size_t granule = [] {
        const std::size_t align = tls_info_from(&_dl_get_tls_static_info).align;
        const std::size_t page = page_size();
        return align > page ? align : page;
    }();
    return granule;
} // stack_granule()

// ---------------------------------------------------------------------------
// Process-wide state.
// ---------------------------------------------------------------------------

// The process default stack size, in bytes, used when Thread::Attributes
// leaves stack_size unset. Constant-initialized (no store happens at
// startup). Written by Thread::set_default_stack_size() with
// exchange(release) and read with load(acquire), at each construction and by
// Thread::default_stack_size(). The value is the only thing it carries:
// nothing else is published through it, and any order would deliver the
// value itself (an atomic load reads some value in its modification order).
// Release/acquire is the convention for a process-wide setting written by
// one thread and read by others; concurrent writers are allowed, the last
// writer in the modification order wins. Stored as given: the clamp, the
// rounding and the overflow check happen at construction.
inline std::atomic<std::size_t> default_stack_bytes{std::size_t{2} << 20};     // 2 MiB

// The guard size, in bytes, used when Thread::Attributes leaves guard_size
// unset: 1 MiB, a constant.
inline constexpr std::size_t default_guard_bytes = std::size_t{1} << 20;

// The process-wide creation count that numbers threads for the stagger: each
// construction that reaches the copying of the callable and the arguments
// (step 3 of the constructor's contract) takes one value, also one that then
// fails copying or in pthread_create, with fetch_add(1, relaxed). Relaxed
// suffices: a read-modify-write always reads the value immediately before
// its own write in the counter's modification order, so concurrent
// constructions get distinct values, and nothing is published through the
// counter. This is a shared counter, not a member of an object under
// construction, so the rule against read-modify-writes on an object's own
// atomics during its construction does not apply. Tests read and write it
// (relaxed) to place a thread in a chosen slot.
inline std::atomic<unsigned> stagger_counter{0};

// Takes the next creation number and returns its stagger amount. nothrow.
inline std::size_t next_stagger() noexcept {
    return stagger_for(stagger_counter.fetch_add(1, std::memory_order_relaxed));
} // next_stagger()

// ---------------------------------------------------------------------------
// The start mask.
// ---------------------------------------------------------------------------

// The signals a new Thread always has unblocked.
inline constexpr std::array<int, 7> unblocked_signals = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGPROF, SIGTRAP, SIGSYS};

// Returns the mask every Thread starts with: sigfillset minus the seven of
// unblocked_signals and minus SIGKILL and SIGSTOP (which no thread can block,
// so the set equals what the kernel reports for the thread). glibc's
// sigfillset already leaves out its two internal signals (32 and 33).
// Returned by reference to a function-local static, so the start routine
// holds no copy of the 128-byte set in its frame; the creating thread calls
// this before pthread_create, so the static is initialized before any new
// thread reads it. nothrow.
inline const sigset_t& start_mask() noexcept {
    static const sigset_t mask = [] {
        sigset_t set;
        sigfillset(&set);
        for (const int signo : unblocked_signals) sigdelset(&set, signo);
        sigdelset(&set, SIGKILL);
        sigdelset(&set, SIGSTOP);
        return set;
    }(); // the start mask's initializer
    return mask;
} // start_mask()

// The creating thread's side of the start mask (RAII). The constructor adds
// every signal of start_mask() to the calling thread's mask
// (pthread_sigmask(SIG_BLOCK)) and keeps the previous mask in `saved_`; the
// destructor restores `saved_` (SIG_SETMASK). While the scope is alive the
// calling thread's mask is its previous mask plus start_mask()'s signals;
// the seven of unblocked_signals stay as the calling thread had them. A new
// thread created inside the scope inherits that mask, so every signal outside
// the seven is blocked in it from its first instruction. Inside the scope
// the calling thread still receives at once any of the seven it has
// unblocked. Any other signal aimed at the calling thread stays pending and
// is delivered to it when the destructor restores the mask; any other
// process-directed signal goes to another thread that leaves it unblocked,
// or stays pending for the process until one does.
// The pthread_sigmask calls cannot fail: `how` is valid and the pointers are
// valid, which are pthread_sigmask's only error conditions.
// Not copyable or movable: the scope belongs to one thread's stack frame.
class CreationMaskScope {
public:
    CreationMaskScope() noexcept {
        pthread_sigmask(SIG_BLOCK, &start_mask(), &saved_);
    }
    ~CreationMaskScope() {
        pthread_sigmask(SIG_SETMASK, &saved_, nullptr);
    }
    CreationMaskScope(const CreationMaskScope&) = delete;
    CreationMaskScope& operator=(const CreationMaskScope&) = delete;

    // The calling thread's mask before the scope began. nothrow.
    [[nodiscard]] const sigset_t& saved() const noexcept { return saved_; }

private:
    sigset_t saved_;            // the calling thread's mask before the scope, restored by the destructor
}; // class CreationMaskScope

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

// Throws std::system_error(code, std::generic_category()) with a message that
// names the failed call, the computed stack size and the guard size passed to
// glibc. Out of line from the callers' hot text.
[[noreturn, gnu::cold, gnu::noinline]] inline void throw_error(int code, const char* call, std::size_t stack,
                                                              std::size_t guard) {
    std::string message = std::string("Thread: ") + call + " failed (stack " + std::to_string(stack) +
                          " bytes, guard " + std::to_string(guard) + " bytes)";
    if (code == EAGAIN) message += ": stack or guard too large, or too many threads";
    throw std::system_error(code, std::generic_category(), message);
} // throw_error()

// Throws std::system_error(EINVAL, std::generic_category()) when the stack
// size computation or the guard check overflows size_t: the class's own
// failure, before any pthread call, so the message names the size
// computation instead of a call. It reports the REQUESTED sizes (`stack`
// before the clamp and the additions, the guard as requested): those are
// what the caller passed and can act on, and when the stack computation is
// the one that overflows no computed stack size exists.
[[noreturn, gnu::cold, gnu::noinline]] inline void throw_overflow(std::size_t stack, std::size_t guard) {
    throw std::system_error(EINVAL, std::generic_category(),
                            "Thread: stack and guard size computation overflows (requested stack " +
                                std::to_string(stack) + " bytes, guard " + std::to_string(guard) + " bytes)");
} // throw_overflow()

// ---------------------------------------------------------------------------
// The thread attributes.
// ---------------------------------------------------------------------------

// RAII owner of the pthread_attr_t a construction passes to pthread_create.
// It composes the request with the defaults itself (a test can therefore
// check that pthread_attr_getstacksize() of get() equals compute_stack_size()
// of the same inputs):
//   stack: compute_stack_size(stack_request or the process default,
//          stack_extra, static_tls_size(), stack_granule(), min_stack_size());
//   guard: guard_request or default_guard_bytes, passed to glibc as is
//          (glibc rounds it up to a page; 0 requests no guard).
// Throws: std::system_error EINVAL if the stack computation overflows or the
// page-rounded guard does not fit in size_t together with the stack; this
// check runs before pthread_attr_init, so nothing is allocated. A failure of
// pthread_attr_init or of a setter (which glibc never reports for these
// values) throws std::system_error with its code, after destroying the attr.
// Strong guarantee: on a throw nothing is left to release.
// The destructor calls pthread_attr_destroy on every path (POSIX requires it).
// Not copyable or movable: POSIX does not allow using a copy of a
// pthread_attr_t.
class AttrOwner {
public:
    AttrOwner(const std::optional<std::size_t>& stack_request, const std::optional<std::size_t>& guard_request) {
        // Acquire: see default_stack_bytes.
        const std::size_t requested =
            stack_request ? *stack_request : default_stack_bytes.load(std::memory_order_acquire);
        guard_bytes_ = guard_request.value_or(default_guard_bytes);
        const std::optional<std::size_t> stack =
            compute_stack_size(requested, stack_extra, static_tls_size(), stack_granule(), min_stack_size());
        if (THREAD_RARELY(!stack || !guard_fits(*stack, guard_bytes_, page_size()))) {
            throw_overflow(requested, guard_bytes_);
        }
        stack_bytes_ = *stack;
        int rc = pthread_attr_init(&attr_);
        if (THREAD_RARELY(rc != 0)) throw_error(rc, "pthread_attr_init", stack_bytes_, guard_bytes_);
        rc = pthread_attr_setstacksize(&attr_, stack_bytes_);
        if (THREAD_RARELY(rc != 0)) {
            pthread_attr_destroy(&attr_);
            throw_error(rc, "pthread_attr_setstacksize", stack_bytes_, guard_bytes_);
        }
        rc = pthread_attr_setguardsize(&attr_, guard_bytes_);
        if (THREAD_RARELY(rc != 0)) {
            pthread_attr_destroy(&attr_);
            throw_error(rc, "pthread_attr_setguardsize", stack_bytes_, guard_bytes_);
        }
    } // AttrOwner::AttrOwner()
    ~AttrOwner() { pthread_attr_destroy(&attr_); }
    AttrOwner(const AttrOwner&) = delete;
    AttrOwner& operator=(const AttrOwner&) = delete;

    // The attributes, for pthread_create. nothrow.
    [[nodiscard]] const pthread_attr_t& get() const noexcept { return attr_; }
    // The stack size passed to pthread_attr_setstacksize. nothrow.
    [[nodiscard]] std::size_t stack_size() const noexcept { return stack_bytes_; }
    // The guard size passed to pthread_attr_setguardsize. nothrow.
    [[nodiscard]] std::size_t guard_size() const noexcept { return guard_bytes_; }

private:
    std::size_t stack_bytes_ = 0;   // the computed stack size, as passed to glibc
    std::size_t guard_bytes_ = 0;   // the guard size, as passed to glibc
    pthread_attr_t attr_;           // initialized by the constructor; destroyed by the destructor
}; // class AttrOwner

// ---------------------------------------------------------------------------
// The new thread's side.
// ---------------------------------------------------------------------------

// The heap state of one thread: the decay-copies of the callable and of the
// arguments, and the thread's stagger amount. Created by the creating thread;
// owned by the new thread once pthread_create succeeds, which destroys it
// before it ends (or, on the terminate path, never). Holds no pointer back to
// the Thread object, which may move while the thread runs, and no mask (the
// start mask is a constant).
template <class F, class... Args>
struct State {
    // Stores the stagger amount and decay-copies `f` into `callable` and each
    // of `args` into `arguments` (G and As are F and Args before the decay:
    // the constructor's own argument types, forwarded). Propagates whatever a
    // copy or move constructor throws; the members already constructed are
    // then destroyed.
    template <class G, class... As>
    explicit State(std::size_t stagger_bytes, G&& f, As&&... args)
        : stagger(stagger_bytes), callable(std::forward<G>(f)), arguments(std::forward<As>(args)...) {}

    // The number of arguments, for invoke_state()'s index sequence.
    static constexpr std::size_t arity = sizeof...(Args);

    const std::size_t stagger;          // bytes reserved at the top of the stack before the callable runs
    F callable;                         // the decay-copy of the callable
    std::tuple<Args...> arguments;      // the decay-copies of the arguments
}; // struct State

// Invokes the callable with the arguments, both as rvalues, as std::thread
// does: INVOKE(std::move(callable), std::move(arguments)...); `I` is
// 0..S::arity-1. Never inlined, so that the callable's frames lie below the
// stagger reservation of start_routine(): inlined, the callable's locals
// would join start_routine()'s fixed frame, which lies ABOVE the reservation,
// and the reservation would move nothing. A callable that is not a member
// pointer is called directly rather than through std::apply or std::invoke:
// in an unoptimized build each of their forwarding layers is a frame of its
// own, between this frame and the callable's, that grows with the number of
// arguments and would come out of the stack the caller asked for; called
// directly, the callable's first frame is the next one below this. Member
// pointers need std::invoke's dispatch (object, pointer or
// reference_wrapper). Propagates whatever the callable throws.
template <class S, std::size_t... I>
[[gnu::noinline]] void invoke_state(S& state, std::index_sequence<I...>) {
    if constexpr (std::is_member_pointer_v<decltype(state.callable)>) {
        static_cast<void>(std::invoke(std::move(state.callable), std::get<I>(std::move(state.arguments))...));
    } else {
        static_cast<void>(std::move(state.callable)(std::get<I>(std::move(state.arguments))...));
    }
} // invoke_state()

// The start routine of every Thread (the function pthread_create runs), for
// the state type S. Deliberately not noexcept (see the catch clauses).
//   1. Sets the thread's mask to start_mask(): its first action, before any
//      other code of the class runs in the thread.
//   2. Takes ownership of the state (a unique_ptr, so every path that leaves
//      this function other than std::terminate destroys the copies here, in
//      the new thread).
//   3. Reserves the stagger amount on the stack with alloca. The empty asm
//      that takes the pointer keeps the reservation: without a use, the
//      optimizer deletes an alloca whose result is unused.
//   4. Calls invoke_state() inside the try. The call cannot become a tail
//      call (which would release this frame, the reservation with it, before
//      the callable runs): the unique_ptr is destroyed after it returns and
//      the try needs the frame for its handlers.
// Exceptions: abi::__forced_unwind (pthread_exit, cancellation) is rethrown,
// so the unwinding continues through this frame (destroying the state) into
// glibc, which ends the thread normally. Any other exception is caught by
// catch (...), which by then has unwound every frame between the throw and
// this one (their destructors have run); std::terminate() is called INSIDE
// the handler, so the std::set_terminate handler runs with the exception
// still current (std::current_exception() returns it). The state is not
// destroyed on that path: the terminate handler must end the process.
template <class S>
void* start_routine(void* raw) {
    pthread_sigmask(SIG_SETMASK, &start_mask(), nullptr);
    const std::unique_ptr<S> state(static_cast<S*>(raw));
    void* const reservation = __builtin_alloca(state->stagger);
    __asm__ volatile("" : : "r"(reservation) : "memory");
    try {
        invoke_state(*state, std::make_index_sequence<S::arity>{});
    } catch (abi::__forced_unwind&) {
        throw;
    } catch (...) {
        std::terminate();
    }
    return nullptr;
} // start_routine()

// Runs pthread_create(&handle, &attr, routine, arg) inside a
// CreationMaskScope, so the new thread inherits the calling thread's mask
// plus start_mask()'s signals, and the calling thread's mask is restored
// before this returns, whatever pthread_create returned. Returns
// pthread_create's result. glibc writes `handle` before the new thread
// starts; on failure `handle` may hold a stale value. nothrow.
inline int create_thread(pthread_t& handle, const pthread_attr_t& attr, void* (*routine)(void*), void* arg) noexcept {
    const CreationMaskScope scope;
    return pthread_create(&handle, &attr, routine, arg);
} // create_thread()

// Constrains Thread's attribute-less constructor: F, without cv and
// reference, is neither Self (so the template never hijacks copying or
// moving a Self) nor Self::Attributes (so a Thread(attributes, f, args...)
// call, even with a non-const lvalue `attributes`, selects the constructor
// that takes the attributes). Self is a parameter, not Thread itself, because
// the constraint is written inside Thread's definition, where Thread is
// incomplete and Thread::Attributes cannot yet be named; the dependent
// `typename Self::Attributes` is looked up only when the constraint is
// checked. Invocability and copyability of F and the arguments are not part
// of the constraint: they are static_asserts in the constructor (a misuse is
// reported as such, not as "no matching constructor").
template <class F, class Self>
concept thread_callable =
    !std::same_as<std::remove_cvref_t<F>, Self> && !std::same_as<std::remove_cvref_t<F>, typename Self::Attributes>;

} // namespace thread_detail

// A thread of execution, launched by the constructor and joined by the
// destructor (like std::jthread, without the stop token), with a stack size,
// a guard size and a fixed start mask (see the top of this header).
//
// Value semantics: move-only. A Thread owns at most one thread of execution
// ("is joinable"); moving transfers the ownership, copying is not allowed.
// Rule of five, explicitly: the destructor joins, so the compiler-generated
// moves would be wrong (they would copy the handle and leave the source
// joinable).
//
// Thread safety: concurrent calls on one Thread object are a data race unless
// all of them are calls of const members, as for std::thread. The callable
// must not use its own Thread object (for example through a captured pointer
// or reference) unless that use is synchronized with the end of the
// constructor, for example through an atomic flag that the creating thread
// stores with release order after the constructor returns and the callable
// loads with acquire order before the use: the constructor writes the
// object, and the object may be moved after it returns.
class Thread {
public:
    using native_handle_type = pthread_t;

    // Per-thread attributes, an aggregate for designated initializers:
    //   Thread t({.stack_size = std::size_t{1} << 20, .guard_size = 0}, f);
    // A member left empty takes the default, read when the Thread is
    // constructed (not when the Attributes object is): an Attributes built
    // before a set_default_stack_size() call and passed to a constructor
    // after it gets the new default.
    struct Attributes {
        // Stack size in bytes. Empty: default_stack_size() at construction
        // (2 MiB unless changed). The thread gets AT LEAST this many bytes of
        // stack below the callable's first frame, that is, between the
        // address __builtin_frame_address(0) returns in the callable's body
        // and the lowest usable address of the stack (the top of the guard,
        // if there is one);
        // the callable's locals and everything it calls use this space. A
        // lower bound, never an exact size: the class adds room for the
        // static TLS, the stagger and its own frames, and rounds the sum up
        // to a page or to the static TLS alignment; glibc may reuse a larger
        // cached stack; ThreadSanitizer raises small requests. The room for
        // the class's own frames covers ordinary argument lists; in an
        // unoptimized build each argument enlarges those frames, so a
        // callable with an unusually long argument list (for a
        // member-function pointer a shorter list, since its call goes
        // through std::invoke's frames) can get less than the full amount.
        // A value below the system minimum, sysconf(_SC_THREAD_STACK_MIN),
        // is raised to that minimum (0 gets the minimum). A value for which
        // the size computation overflows size_t makes the constructor throw
        // EINVAL (step 1 of the constructor's contract states the exact
        // condition; SIZE_MAX is one such value).
        std::optional<std::size_t> stack_size = {};
        // Guard size in bytes: an inaccessible region below the stack, which
        // turns a stack overflow into a fault. Empty: 1 MiB. Any other value
        // is passed to glibc, which rounds it up to a page: the guard is then
        // AT LEAST the value rounded up to a page. 0 requests no guard; a
        // stack glibc reuses from its cache may keep the guard of an earlier
        // thread, so 0 promises nothing about the absence of a guard mapping.
        // The guard is not part of stack_size: glibc maps it in addition.
        // A value whose page-rounded size does not fit in size_t together
        // with the computed stack size makes the constructor throw EINVAL
        // (step 1 of the constructor's contract); values that fit but are too
        // large for glibc fail as glibc reports them (EAGAIN or EINVAL from
        // pthread_create, step 4).
        std::optional<std::size_t> guard_size = {};
    }; // struct Attributes

    // A Thread that owns no thread. Postconditions: joinable() is false,
    // get_id() == std::thread::id{}. nothrow.
    Thread() noexcept = default;

    // Starts a thread that runs
    //   std::invoke(std::move(f'), std::move(args')...),
    // f' and args' being decay-copies of f and args made in the calling
    // thread, with default attributes. Identical to
    // Thread(Attributes{}, f, args...); see that constructor for the
    // contract. Constrained (thread_callable) so that it never takes a Thread
    // or an Attributes as the callable.
    template <class F, class... Args>
        requires thread_detail::thread_callable<F, Thread>
    explicit Thread(F&& f, Args&&... args) {
        launch(Attributes{}, std::forward<F>(f), std::forward<Args>(args)...);
    } // Thread::Thread()

    // Starts a thread with the given attributes that runs
    //   std::invoke(std::move(f'), std::move(args')...),
    // f' and args' being decay-copies of f and args made in the calling thread.
    //
    // Mandates (static_assert): std::decay_t<F> is constructible from F, each
    // std::decay_t<Args> from its Args, and std::decay_t<F> is invocable with
    // rvalues of the std::decay_t<Args>.
    //
    // Stack and guard: see Attributes. The stack below the callable's first
    // frame (as defined there) is at least the requested (or default) stack
    // size, except as stated there for unusually long argument lists in
    // unoptimized builds.
    //
    // The stagger: a construction that reaches step 3 below takes the next
    // number n of a process-wide creation count (unsigned, wrapping), also
    // when it then fails in step 3 or 4; one that fails in step 1 or 2 takes
    // none.
    // The new thread reserves (n % 16) * 256 bytes at the top of its stack
    // before the callable runs, so the callable's first frame lies that many
    // bytes deeper than with no reservation; the reservation comes out of the
    // room the class adds, not out of the requested size. Its purpose, to
    // spread the top frames of consecutive threads over different cache
    // sets, is not a promise.
    //
    // The start mask: the callable starts with every blockable signal blocked
    // except SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGPROF, SIGTRAP and SIGSYS,
    // which are unblocked, whatever the creating thread's mask. SIGKILL and
    // SIGSTOP cannot be blocked, and glibc's two internal signals (32 and 33,
    // the realtime signals below SIGRTMIN) are left unblocked: glibc's sigset
    // functions and pthread_sigmask never block them. When the constructor
    // returns or throws, the creating thread's mask is what it was before.
    //
    // Order of the steps, and what each failure leaves:
    //   1. Sizes computed and checked. With R the requested (or default)
    //      stack size and G the requested guard (or 1 MiB), the check fails
    //      when max(R, sysconf(_SC_THREAD_STACK_MIN)) plus the class's
    //      additions (thread_detail::stack_extra and
    //      thread_detail::static_tls_size()), rounded up to
    //      thread_detail::stack_granule(), does not fit in size_t, or when G
    //      rounded up to the page size, plus that stack size, does not
    //      (thread_detail::compute_stack_size() returns empty, or
    //      thread_detail::guard_fits() returns false). stack_size = SIZE_MAX
    //      and guard_size = SIZE_MAX both fail it. A failed check throws
    //      std::system_error with EINVAL before anything is allocated or
    //      copied: f and args are untouched (an rvalue argument is not moved
    //      from).
    //   2. Attributes set up (pthread_attr_init and setters; they do not fail
    //      for the computed values).
    //   3. f and args decay-copied into a heap allocation. An exception from
    //      a copy or move constructor, or std::bad_alloc, propagates; no
    //      thread was created, and the copies already made are destroyed.
    //   4. pthread_create. On failure the constructor throws std::system_error
    //      with pthread_create's return value (EAGAIN: stack or guard too
    //      large for the system, or too many threads; for example stack_size
    //      = std::size_t{1} << 62 fails with EAGAIN); the copies were made and
    //      are destroyed in the calling thread before the exception leaves
    //      the constructor; rvalue arguments are left moved-from.
    // Every std::system_error thrown here uses std::generic_category().
    // Step 1's carries EINVAL, the class's own value, and a message naming
    // the stack and guard size computation and the REQUESTED stack and guard
    // sizes (R and G). Steps 2 and 4 carry the failing pthread call's return
    // value (never errno) and a message naming the call, the COMPUTED stack
    // size and the guard size passed to glibc (G). No other exceptions than
    // those of steps 1-4 (and std::bad_alloc from building the message).
    // Exception safety: if the constructor throws, no thread was created, no
    // copy of f or args remains, and the creating thread's mask is unchanged.
    //
    // Postconditions (on return): joinable() is true; get_id() is the new
    // thread's id. The completion of the constructor's copies happens before
    // the start of the callable's invocation.
    //
    // While the thread runs:
    //   - The copies are destroyed by the new thread, after the callable
    //     returns and before the thread ends (so before join() returns).
    //   - If the callable calls pthread_exit, the thread's stack is unwound
    //     (destructors of the callable's locals run), the copies are
    //     destroyed by the new thread during that unwinding, and the thread
    //     ends normally: join() returns. Thread cancellation through
    //     native_handle() is not supported or promised.
    //   - If an exception escapes the callable, the frames between the throw
    //     and the class's start routine are unwound (destructors run), and
    //     then std::terminate() is called with the exception still current:
    //     the std::set_terminate handler sees it through
    //     std::current_exception(). The copies are not destroyed on this
    //     path: the terminate handler runs with them alive, and the process
    //     ends without destroying them.
    template <class F, class... Args>
    explicit Thread(const Attributes& attributes, F&& f, Args&&... args) {
        launch(attributes, std::forward<F>(f), std::forward<Args>(args)...);
    } // Thread::Thread()

    // Joins the owned thread, if any (waits for the callable to return; there
    // is no stop request). noexcept: if the join fails, std::terminate() is
    // called; in particular when the destructor runs on the thread it would
    // join (the callable destroying its own Thread object), which is EDEADLK.
    ~Thread() noexcept {
        if (joinable()) join();
    } // Thread::~Thread()

    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

    // Takes the thread `other` owns. Postconditions: *this owns what `other`
    // owned; other.joinable() is false. nothrow.
    Thread(Thread&& other) noexcept : handle_(std::exchange(other.handle_, native_handle_type{})) {}

    // If *this is joinable, joins it first (as the destructor does), then
    // takes the thread `other` owns. Postconditions: *this owns what `other`
    // owned; other.joinable() is false. Self-move-assignment does nothing.
    // noexcept: a failing join calls std::terminate(), in particular when run
    // on the thread *this owns (EDEADLK).
    Thread& operator=(Thread&& other) noexcept {
        if (this == &other) return *this;
        if (joinable()) join();
        handle_ = std::exchange(other.handle_, native_handle_type{});
        return *this;
    } // Thread::operator=()

    // True if *this owns a thread of execution: constructed by a starting
    // constructor (or moved into) and not yet joined, detached or moved from.
    // nothrow.
    [[nodiscard]] bool joinable() const noexcept { return handle_ != native_handle_type{}; }

    // Waits for the owned thread to end. The end of the thread (the
    // callable's return or pthread_exit, and the destruction of the copies)
    // happens before join() returns. Postcondition: joinable() is false.
    // Throws std::system_error (std::generic_category()):
    //   - std::errc::invalid_argument if !joinable();
    //   - std::errc::resource_deadlock_would_occur if called on the thread
    //     *this owns (get_id() == std::this_thread::get_id());
    //   - pthread_join's error otherwise.
    // Strong guarantee: on a throw *this is unchanged.
    void join() {
        if (THREAD_RARELY(!joinable())) {
            throw std::system_error(EINVAL, std::generic_category(), "Thread::join: not joinable");
        }
        if (THREAD_RARELY(pthread_equal(handle_, pthread_self()) != 0)) {
            throw std::system_error(EDEADLK, std::generic_category(), "Thread::join: called on the thread itself");
        }
        const int rc = pthread_join(handle_, nullptr);
        if (THREAD_RARELY(rc != 0)) {
            throw std::system_error(rc, std::generic_category(), "Thread::join: pthread_join failed");
        }
        handle_ = native_handle_type{};
    } // Thread::join()

    // Lets the owned thread run on independently: it releases its resources
    // when it ends. Postcondition: joinable() is false.
    // Throws std::system_error (std::generic_category()):
    //   - std::errc::invalid_argument if !joinable();
    //   - pthread_detach's error otherwise.
    // Strong guarantee: on a throw *this is unchanged.
    void detach() {
        if (THREAD_RARELY(!joinable())) {
            throw std::system_error(EINVAL, std::generic_category(), "Thread::detach: not joinable");
        }
        const int rc = pthread_detach(handle_);
        if (THREAD_RARELY(rc != 0)) {
            throw std::system_error(rc, std::generic_category(), "Thread::detach: pthread_detach failed");
        }
        handle_ = native_handle_type{};
    } // Thread::detach()

    // The pthread_t of the owned thread; native_handle_type{} (zero) if
    // !joinable(). Calling pthread_cancel on it is not supported. nothrow.
    [[nodiscard]] native_handle_type native_handle() const noexcept { return handle_; }

    // The id of the owned thread, equal to std::this_thread::get_id() inside
    // it; std::thread::id{} if !joinable(). nothrow.
    [[nodiscard]] std::thread::id get_id() const noexcept {
        // libstdc++'s id(native_handle_type); a zero handle gives id{}.
        return std::thread::id(handle_);
    } // Thread::get_id()

    // Exchanges the threads owned by *this and `other`. nothrow.
    void swap(Thread& other) noexcept { std::swap(handle_, other.handle_); }
    friend void swap(Thread& a, Thread& b) noexcept { a.swap(b); }

    // std::thread::hardware_concurrency(). nothrow.
    [[nodiscard]] static unsigned int hardware_concurrency() noexcept { return std::thread::hardware_concurrency(); }

    // Sets the process default stack size (used when Attributes::stack_size
    // is empty, read at each construction) to `bytes`, stored as given, and
    // returns the previous value. No clamping or rounding here: that happens
    // at construction, so 0 gives threads the minimum stack and SIZE_MAX makes
    // the constructors that use the default throw EINVAL. Safe to call
    // concurrently with constructions and with other calls (the last writer
    // wins). nothrow.
    static std::size_t set_default_stack_size(std::size_t bytes) noexcept {
        // Release: see thread_detail::default_stack_bytes.
        return thread_detail::default_stack_bytes.exchange(bytes, std::memory_order_release);
    } // Thread::set_default_stack_size()

    // The process default stack size, as last set (2 MiB initially). nothrow.
    [[nodiscard]] static std::size_t default_stack_size() noexcept {
        // Acquire: see thread_detail::default_stack_bytes.
        return thread_detail::default_stack_bytes.load(std::memory_order_acquire);
    } // Thread::default_stack_size()

private:
    // The body of both starting constructors (which do not delegate to each
    // other or to another constructor: a throw after a completed delegating
    // constructor would run ~Thread() on the object). Steps as documented at
    // Thread(const Attributes&, F&&, Args&&...). pthread_create writes the
    // handle member directly (glibc stores it before the new thread starts;
    // copying it from a local afterwards would race with a callable that
    // reads its own Thread object); on a failure the member is reset to the
    // not-joinable value before the throw, since glibc may have stored a
    // stale value.
    template <class F, class... Args>
    void launch(const Attributes& attributes, F&& f, Args&&... args) {
        static_assert(std::is_constructible_v<std::decay_t<F>, F>,
                      "Thread: the callable must be constructible (copied or moved) from the argument passed");
        static_assert((std::is_constructible_v<std::decay_t<Args>, Args> && ...),
                      "Thread: each argument must be constructible (copied or moved) from the value passed");
        static_assert(std::is_invocable_v<std::decay_t<F>, std::decay_t<Args>...>,
                      "Thread: the callable must be invocable with the decay-copied arguments as rvalues");
        // The rest only for well-formed requests, so a misuse reports the
        // static_assert above and not the errors it would cause further down.
        if constexpr (std::is_constructible_v<std::decay_t<F>, F> &&
                      (std::is_constructible_v<std::decay_t<Args>, Args> && ...) &&
                      std::is_invocable_v<std::decay_t<F>, std::decay_t<Args>...>) {
            using S = thread_detail::State<std::decay_t<F>, std::decay_t<Args>...>;
            // Not const: ThreadSanitizer's pthread_create interceptor writes
            // the raised stack size back into the attributes it is passed.
            thread_detail::AttrOwner attr(attributes.stack_size, attributes.guard_size);            // steps 1, 2
            std::unique_ptr<S> state = std::make_unique<S>(thread_detail::next_stagger(), std::forward<F>(f),
                                                           std::forward<Args>(args)...);           // step 3
            const int rc = thread_detail::create_thread(handle_, attr.get(), &thread_detail::start_routine<S>,
                                                        state.get());                              // step 4
            if (THREAD_RARELY(rc != 0)) {
                handle_ = native_handle_type{};
                // `state` destroys the copies while this exception unwinds the
                // frame, before it leaves the constructor.
                thread_detail::throw_error(rc, "pthread_create", attr.stack_size(), attr.guard_size());
            } // if pthread_create failed
            // The new thread owns the state now and may already have destroyed
            // it: this thread must not touch it again.
            static_cast<void>(state.release());
        } // if the request is well-formed
    } // Thread::launch()

    // The owned thread's pthread_t; native_handle_type{} (zero, never a valid
    // thread on glibc) when *this owns no thread. Written by pthread_create.
    native_handle_type handle_ = {};
}; // class Thread

// Header-private: expanded where the functions above were parsed (macros
// expand at definition, not at template instantiation), so it is not exported
// to includers.
#undef THREAD_RARELY

#endif // the requirements hold
#endif // INCLUDED_THREAD_H
