# Thread

`std::jthread` fixed the worst thing about `std::thread`: forget to join it,
and its destructor joins for you instead of terminating the program. What
neither of them will do is let you choose anything about the thread itself.
There is no attribute object, so no stack size and no guard size, and no say
over the signal mask — a new thread inherits whatever mask its creator
happened to have at that moment, a perfectly reasonable policy right up until
the creator is a thread that blocked SIGSEGV.

`Thread` (`thread.h`, header-only) is used the way `std::jthread` is:
construct it from a callable and its arguments and the thread is running;
destroy it and the thread is joined. On top of that it gives every thread

- **a stack size and a guard size**, per thread, through an attributes
  object, with a process-wide default for the stack;
- **a fixed start mask**: the asynchronous signals blocked, the faults (and
  SIGPROF) unblocked, whatever the creating thread had;
- **a staggered stack top**: consecutive threads' top frames land at
  different offsets within a page, meant to keep them from all fighting over
  the same cache sets;
- **unwinding before `std::terminate()`**: an exception that escapes the
  callable runs the destructors on its way out, and the terminate handler
  still sees it.

What it leaves out is just as deliberate. There is no stop token or any other
cooperative stop request, no API to change the start mask or to handle
signals, and no cancellation.

## Using it

```cpp
#include "thread.h"

std::atomic<bool> done{false};
{
    Thread poller({.stack_size = std::size_t{256} << 10}, [&done](int fd) {
        while (!done.load(std::memory_order_acquire)) poll_once(fd);
    }, fd);
    // ... the main thread goes about its business ...
    done.store(true, std::memory_order_release);
}   // ~Thread() joins here
```

The callable and its arguments are decay-copied in the creating thread, and
the new thread calls the copy with the copied arguments as rvalues, exactly as
`std::thread` does; the copies are destroyed by the new thread, before
`join()` returns. The attributes come first and are optional; without them,
`Thread poller(f, fd)` takes the defaults. The rest of the interface is
`std::thread`'s: `join()`, `detach()`, `joinable()`, `get_id()`,
`native_handle()`, `swap()` and `hardware_concurrency()`. A `Thread` is
move-only; move assignment joins the thread it is about to replace before it
takes the new one. The class reports its own failures, and those of the
pthread calls, as `std::system_error`.

The destructor joins, and that is all it does. There is no stop request: a
thread whose loop never ends keeps its destructor waiting forever. If the
callable loops, give it a flag of your own, as above. A destructor or move
assignment that would join the very thread it runs on — a callable destroying
its own `Thread` — calls `std::terminate()`, since the alternative is waiting
for itself. More generally, the callable should leave its own `Thread` object
alone unless that use is synchronized with the end of the constructor: the
constructor writes the object, and the object may be moved once it returns.

An exception that escapes the callable is a bug, and `Thread` treats it as
one, politely. The frames between the throw and the thread's start routine are
unwound first, so their destructors run (a destructor that writes a debug dump
gets to write it), and only then is `std::terminate()` called, from inside the
`catch` handler, so a `std::set_terminate` handler sees the exception through
`std::current_exception()`. The standard leaves it to the implementation
whether anything is unwound on the way to `std::terminate()`; `Thread` makes
it a promise. The copies of the callable and its arguments are not destroyed
on that path: the process ends with them alive.

`pthread_exit()` from the callable is not an error in this sense: it unwinds
the thread's stack, destroys the copies, and ends the thread normally, and
`join()` returns. Cancelling the thread through `native_handle()` is not
supported.

## Stack and guard

`Thread::Attributes` is an aggregate of two `std::optional<std::size_t>`
members, made for designated initializers:

```cpp
Thread t({.stack_size = std::size_t{1} << 20, .guard_size = 0}, f, args...);
```

A member left empty takes the default. For the stack that is the process
default, 2 MiB unless `Thread::set_default_stack_size()` changed it
(`Thread::default_stack_size()` reads it), and it is read when the `Thread` is
constructed, not when the `Attributes` object is. For the guard it is 1 MiB.

The stack promise is a lower bound, and it is stated where it matters: the
callable gets **at least `stack_size` bytes of stack below its own first
frame**. That is not the same as asking `pthread_create` for `stack_size`
bytes, because a good deal of the stack is gone before the callable's first
instruction: glibc carves the static TLS block (every `thread_local` of the
program and of the libraries loaded at startup) and the thread descriptor out
of the stack it is asked for, so in a program with a 1 MiB `thread_local` a
plain 2 MiB request leaves the thread half the stack it asked for. The class
therefore asks glibc for more:

- the requested size, raised to the system minimum
  (`sysconf(_SC_THREAD_STACK_MIN)`) if it is below it, so 0 asks for the
  minimum;
- plus 3840 bytes, the deepest stagger reservation (below);
- plus a 2 KiB allowance for the frames the thread's start-up puts on the
  stack before the callable's first one — glibc's, a sanitizer runtime's,
  the class's own;
- plus room for the static TLS block and the thread descriptor, computed from
  glibc's own report of the TLS size and alignment;
- all rounded up to a page, or to the static TLS alignment if an over-aligned
  `thread_local` makes that larger. The rounding also keeps every thread's
  stack top at the same offset within its page, which is what lets the
  stagger alone decide where a thread's top frame lands.

The stack you get may be larger still: glibc reuses the cached stack of a
thread that has ended, when it is big enough, and ThreadSanitizer raises small
requests. The floor is promised, the exact size never. The fine print: the
start-up allowance covers ordinary argument lists, but in an unoptimized
build every argument enlarges the start-up frames, so a callable with an
unusually long argument list (a member-function pointer gets there sooner)
can come up a little short of the full amount.

glibc reports the static TLS size through a `GLIBC_PRIVATE` function of its
dynamic linker, which promises nothing about its future. The class references
it weakly: a glibc without it still links and runs, and the class then adds
nothing for the TLS.

The guard is an inaccessible region below the stack, which turns a stack
overflow into a fault instead of a quiet scribble over whatever lies beneath.
glibc's default guard is one page, which a single large stack frame can step
right over; hence the 1 MiB default. Any other size can be set, larger or
smaller, and glibc rounds it up to a page. The guard is not part of
`stack_size`: glibc maps it in addition. A guard of 0 requests no guard at
all, with one caveat the class cannot remove without allocating stacks itself:
a stack that glibc reuses from its cache may keep the guard of an earlier
thread, so 0 promises nothing about the absence of a guard mapping.

A stack or guard size so large that the computation overflows `size_t`
(`SIZE_MAX`, say) throws `std::system_error` with `EINVAL` before anything is
allocated or copied; the arguments are untouched. A failure in
`pthread_create` itself — a size that fits in `size_t` but not in the
machine, or one thread too many — throws with `pthread_create`'s return
value, usually `EAGAIN`. By then the copies were made: they are destroyed
before the exception leaves the constructor, and rvalue arguments are left
moved-from.

## The stagger

glibc puts each thread's stack top at the top of its mapping, and with
page-multiple stack sizes that top, and with it the thread's first frame, lies
at the same offset within its page in every thread. The same page offset means
the same L1 cache set on the common designs, whose L1 index bits lie inside
the page offset, so the hottest lines of every thread's stack — the top frame,
where a thread that runs one long loop spends its life — compete for the same
few sets.

So every `Thread` reserves `(n % 16) * 256` bytes at the top of its stack
before the callable runs, `n` being its number in a process-wide count of
constructions: sixteen consecutive threads' top frames start at sixteen
different page offsets, 256 bytes (four cache lines) apart, meant to keep them
off the same cache sets. (A construction that fails its size check takes no
number; one that fails later has used its number up.) The reservation comes
out of the room the class adds, not out of your `stack_size`. The benefit is
reasoned from the cache geometry, not measured; whether it changes anything
you can time is an experiment not yet run. There is no knob for it.

## The start mask

Every `Thread` starts with the same signal mask, whatever its creator's: every
blockable signal blocked except SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGPROF,
SIGTRAP and SIGSYS, which are unblocked. (SIGKILL and SIGSTOP cannot be
blocked by anyone, and glibc keeps its two internal signals unblocked.)

The blocked ones are the signals that come from outside: SIGINT from Ctrl-C,
SIGTERM from `kill`, SIGALRM from a timer, SIGCHLD from a child. A signal sent
to the process is delivered to some thread that does not block it, and the
kernel does not ask the application which one; a worker interrupted in the
middle of its loop is rarely the thread that ought to be handling SIGTERM.
With every worker blocking them, these signals go where the application
expects them, to the threads it keeps unblocked — normally the main thread.

The seven are the other kind. When SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP
or SIGSYS comes from the thread's own instruction — a bad pointer, a
breakpoint, a forbidden system call — it goes to that thread alone, and a
synchronous signal that is blocked when the kernel raises it is not deferred:
it is delivered with its default action, which kills the process without
consulting any handler. So a fault in the callable reaches its handler only if
the signal is unblocked in the thread, and `Thread` unblocks the seven even
when the creating thread had them blocked. SIGPROF is unblocked
so that a profiler sampling the process samples the workers too. A creator
that blocks one of the seven — SIGPROF, say, to keep a profiler out of a
stretch of its own code — does not pass that block on to the `Thread`s it
creates.

The mask is fixed: no attribute, no process-wide setting, no handler facility.
One policy, until a real need for another one shows up. A callable that needs
a signal unblocks it itself with `pthread_sigmask`, one line, written where the
reason for it lives.

Every signal outside the seven is blocked from the new thread's very first
instruction, and the exact start mask is in place before the callable runs.
For the duration of `pthread_create` the creating thread adds the blocked
signals to its own mask — it only adds, so its own handlers for the seven keep
working and a profiler keeps sampling it — and it restores its mask before the
constructor returns or throws. The new thread inherits the creator's mask with
those additions, and the class's start routine, before anything else, sets the
exact start mask, which unblocks any of the seven its creator had blocked
(until then, glibc's thread start, and a sanitizer runtime's, run with them
as the creator had them). A signal aimed at the creator while `pthread_create`
runs waits for the restore, unless it is one of the seven: those the creator's
own mask governs throughout. The obvious tool,
`pthread_attr_setsigmask_np`, is not used: AddressSanitizer's `pthread_create`
interceptor throws its mask away.

## Living with the mask

A mask that blocks nearly everything has consequences, and they are the
user's to know:

- **Keep one thread unblocked.** A process-directed signal is delivered only
  to a thread that leaves it unblocked. No `Thread` does unless its callable
  unblocks it, so the application keeps one — normally the main thread, whose
  mask a `Thread` construction leaves as it found it — for SIGINT, SIGTERM
  and the like. If none does, Ctrl-C does nothing: the signal waits, pending,
  for a thread that will take it. (A worker that collects signals with
  `sigwait()` is the exception that proves the rule: `sigwait()` wants them
  blocked, so the worker needs nothing unblocked, but then the main thread
  should block them too, or it may take them instead.)
- **A worker that needs a signal unblocks it itself.**
- **Signals aimed at a worker wait.** `raise()` in a worker, `pthread_kill()`,
  a descriptor whose `F_SETOWN_EX` owner is a worker: the signal stays pending
  until the worker unblocks it. That includes SIGPIPE: a worker's write to a
  closed pipe or socket fails with `EPIPE` as usual, and the SIGPIPE that comes
  with it is delivered later, whenever the worker unblocks it, if it ever
  does. The usual cures — ignoring SIGPIPE, or `MSG_NOSIGNAL` on sockets —
  work here as anywhere.
- **The mask is inherited.** Threads a worker creates, a `std::thread` or an
  OpenMP pool it brings up, inherit its mask.
- **So are child processes.** A process forked or spawned from a worker
  inherits the mask across `exec`, and with SIGTERM, SIGINT and SIGHUP blocked
  a polite `kill` does nothing: the signal stays pending until the child
  resets its mask. Fork and spawn from the main thread, or reset the child's
  mask (`posix_spawnattr_setsigmask` with `POSIX_SPAWN_SETSIGMASK`).
- **Libraries that signal threads directly need that signal unblocked.** A
  garbage collector that stops the world with a signal is the classic case: on
  a thread that blocks the signal, the collector waits, and waits.

SIGABRT is blocked too, and it does not matter: `abort()` unblocks it itself,
so `std::terminate()`, `assert` and `abort()` work in a worker as they do
everywhere else.

## What is not here

- **A stop request.** No `std::stop_token`; the destructor and move
  assignment only join. An atomic flag does the job.
- **A signal API.** No mask attribute, no process-wide default mask, no
  public signal sets, no handler facility.
- **Cancellation.** `pthread_cancel` on `native_handle()` is not supported.
- **Other attributes.** No thread name, CPU affinity or scheduling priority
  yet; `Attributes` is an aggregate, and members can be added to it later.

## Requirements

Linux with glibc, and libstdc++ (with GCC or with Clang). The class creates
threads with `pthread_create` and glibc's thread attributes, asks glibc for
the static TLS size, and builds `get_id()` from libstdc++'s
`std::thread::id(native_handle_type)` constructor; any other C library or
standard library stops the build with an `#error` naming what is missing.

## Building and testing

```sh
make                # the six test binaries (there is no benchmark yet)
make tests          # the same
make run_tests      # build them, then run all six
make clean          # remove this host's build directory (and *.bak, *~ here)
```

The tests need GoogleTest (`GTEST_DIR` in `../config.mk`). Two test sources,
each built three ways:

- `thread_test.C` — the contract, black-box: what the header promises, and
  nothing it does not. The type, invocation and the lifetime of the copies,
  joining, detaching and moving, every failure path and what it leaves
  behind, the start mask seen from creators with empty, full and mixed masks
  and with the seven blocked, the creator's own mask afterwards, signals sent
  to a worker, the stack and guard lower bounds, the default stack size, the
  room for the static TLS, the stagger, and the unusual endings (an escaping
  exception, `pthread_exit`, a `Thread` destroyed or move-assigned on its own
  thread).
- `thread_internals_test.C` — the mechanism, white-box, through the seams the
  header's `thread_detail` namespace exposes: the size arithmetic at its
  boundaries (the clamp, page boundaries, the overflow points, 4, 16 and
  64 KiB pages), the machine parameters against the system's own answers, the
  exact size passed to glibc, both sides of the start mask, the stagger
  counter, the placement of every thread's top frame and the stack left below
  it at the deepest stagger slot, and the failure paths.

The three builds are AddressSanitizer and ThreadSanitizer at `-O0`, and
`-O3 -DNDEBUG` without a sanitizer: `thread_test_asan`, `thread_test_tsan`,
`thread_test_opt`, and the same three of `thread_internals_test`. The
optimized build is not there for speed. The sanitizer runtimes intercept
`pthread_create` and change what the new thread starts with — AddressSanitizer
sets the child's mask back to its creator's, ThreadSanitizer enlarges small
stack requests — and `-O0` keeps what an optimizer would delete or inline: a
stack reservation that nothing reads, the call that puts the callable's frames
below it. Only the optimized build runs the header as it ships. A few tests
skip themselves in one build or another and say why (the TLS tests under
ThreadSanitizer, the heap-accounting checks where there is no sanitizer heap
to ask), and one skips depending on the environment: the test of the handle
reset after a failed thread creation needs an `RLIMIT_NPROC` that actually
stops thread creation (it does not for a privileged process) and a glibc that
stores the handle before the failing clone.

`make run_tests` passes your environment to the binaries unchanged, so the
sanitizer options are yours to set, for example `symbolize=0` where the
symbolizer is slow or hangs:

```sh
TSAN_OPTIONS=symbolize=0 ASAN_OPTIONS=symbolize=0 make run_tests
```

Binaries land in `build/<hostname>/`, so several machines can build and run
concurrently in one shared tree. The compiler, `-march` target, C++ standard
and the GoogleTest path come from `../config.mk`, shared by every project
directory and written once per machine. A compiler given on the command line
is not a prerequisite of the binaries, so switching compilers needs a
`make clean` first, or a build directory of its own:

```sh
make CXX=g++-16 BUILD_DIR=build/gcc run_tests
```

- `thread.h` — the class (header-only)
- `thread_test.C` — the contract tests
- `thread_internals_test.C` — the white-box tests of the mechanism
