// Microbenchmark of the two intrusive atomic shared pointers of this directory,
// intr_shared_ptr (intr_shared_ptr.h: the gap between reading the word and the
// AddRef closed by a one-bit spinlock in the word) and intr_shared_ptr_hp
// (intr_shared_ptr_hp.h: the gap closed by a hazard pointer, released pointees
// destroyed later by an mm_hp scan), side by side on the same four operations.
// It measures the pointers' own operations on one or two program-wide atomics,
// with no data structure around them; the SharedPtr comparison harness
// (../SharedPtr/atomic_shared_ptr_bm.C) measures the same pointers among the
// other policies, with a payload and read/write mixes.
//
// What it measures (every row runs for both pointers, as BM_<op><IntrShared>
// and BM_<op><IntrSharedHP>, at 1, 2, 4, 8, 16, 32, 64, 80, 120 and 128
// threads, real time):
//
//   BM_deref    *p1.load() copied into a local: load(), one read through the
//               returned handle, and the handle's release. The read path alone,
//               the operation a reader of shared state pays for every access.
//               Why: it is where the two designs differ most -- the spinlock
//               pointer WRITES the word twice per load (lock CAS, unlock store)
//               and so serializes readers on the word's cache line, while the
//               hazard pointer one never writes the word but publishes a hazard
//               and runs TryAddRef's CAS loop; every thread still RMWs the one
//               pointee's count in both.
//   BM_copy     an atomic constructed from p1.load() and destroyed: load(), the
//               adopting constructor's AddRef, and two releases (the
//               temporary handle's and the atomic's). Why: the cost of making
//               a new owner from shared state, i.e. BM_deref's read path plus
//               two more count RMWs on the same contended count; the
//               difference from BM_deref is what an extra owner costs.
//   BM_assign   q1.store(p1.load()): a read of one word and a store into
//               another. In the steady state both words hold the same pointee,
//               so the release inside store() never reaches 0 and nothing is
//               reclaimed. Why: the write path (spinlock: lock and publish q1;
//               hazard pointer: an acq_rel exchange) without reclamation,
//               with every thread writing q1 and reading p1.
//   BM_xassign  odd-numbered threads q1.store(p1.load()), even-numbered ones
//               p1.store(q1.load()); with one thread only the second. Why:
//               two-way traffic, every word both read and written by
//               different threads at once -- the case where the spinlock's
//               readers and writers queue on the same lock bit and where the
//               hazard pointer's validation reloads see the word move. Thread
//               0 first replaces both pointees (outside the timed loop), so
//               every run starts from fresh objects.
//
// Every row reports, next to the Time column, two rates of one item per
// iteration and a fairness measure (../Spinlock/gb_wall_clock.h defines them
// and does the stamping):
//   items_per_second       Google Benchmark's rate: the iterations summed over
//                          the threads, divided by the MEAN per-thread loop
//                          time;
//   wall_items_per_second  the same total over the wall-clock span of the run,
//                          from the end of the earliest thread's first
//                          iteration to the end of the latest thread's last;
//   finish_spread          (latest finish - earliest finish)/span, from 0 when
//                          all threads finish together to near 1 when some
//                          thread finishes right at the start.
// Time is the mean per-thread time per iteration (the threads' summed real
// loop times over their summed iterations), not the wall-clock time per
// operation of the run. Every thread runs the same number of iterations, so
// under a pointer that lets some threads finish early, those threads wait at
// the end barrier, the mean loop time is shorter than the run, and both Time
// and items_per_second flatter that pointer; wall_items_per_second is what
// the run delivered, and finish_spread says how much of the run had fewer
// threads left.
//
// No row measures reclamation: in the steady state of every row no count
// reaches 0. intr_shared_ptr_hp's deferred destruction (an mm_hp scan of
// ~1000 objects inside some thread's retire) is measured by the SharedPtr
// harness's write rows, not here.
//
// Rows above the machine's CPU count are oversubscribed (on a 16-CPU machine:
// 32, 64, 80, 120 and 128) and measure scheduling as much as the pointer;
// compare rows with at most as many threads as CPUs. Under oversubscription
// the spinlock pointer's lock holder can be descheduled mid-critical-section,
// which its nanosleep back-off is designed for; the hazard pointer one has no
// such state.
//
// The pointee, B<HasHazardBase> below, has its own hand-written hooks and
// layout, held fixed here independently of the bases intr_pointee_base /
// intr_pointee_base_hp, so that results of this program stay comparable as
// those bases evolve. B differs from them in two ways that reach the measured
// code: AddRef is acq_rel (stronger than the relaxed the contract in
// intr_pointee.h needs), and the hazard pointer base precedes the count (a
// different offset in reset_protection's address arithmetic).
//
// Both pointers run in one process. intr_shared_ptr never touches mm_hp, and
// mm_hp runs only inside a retire() (a scan, when one is due), which only the
// hazard pointer rows call; what they leave behind is a few retired B objects
// pending in mm_hp's list, which nothing reads during the spinlock rows.
#include "intr_shared_ptr.h"
#include "intr_shared_ptr_hp.h"     // brings mm_hp/ (std::hazard_pointer_obj_base)

#include <atomic>
#include <type_traits>

#include "benchmark/benchmark.h"

#include "gb_wall_clock.h"          // WallRecords, WallTimed, report_wall() (../Spinlock)

#define ARGS(N) \
  ->Threads(N) \
  ->UseRealTime()

using namespace std;

// The interface type T of both pointers: what operator* returns. The volatile
// assignment is what BM_deref copies the pointee into, so that the read is not
// optimized away.
struct A {
  int i;
  A(int i = 0) : i(i) {}
  A& operator=(const A& rhs) { i = rhs.i; return *this; }
  volatile A& operator=(const A& rhs) volatile { i = rhs.i; return *this; }
};

// Live B objects (never read here). Atomic because B objects are destroyed on
// benchmark threads: under intr_shared_ptr by whichever thread releases the
// last reference, under intr_shared_ptr_hp inside an mm_hp scan on whichever
// thread crosses the threshold; two threads may do either at once.
static atomic<unsigned long> B_count{0};

// B's base when it is NOT a hazard pointer pointee: empty, so B<false> is 16
// bytes, A's int at offset 0 and the count at offset 8 (an empty base after A
// shares A's offset).
struct no_hazard_base {};

// The pointee U of both pointers, with T = A. HasHazardBase selects the
// pointer: false for intr_shared_ptr (IntrusivePointee: AddRef, DelRef,
// use_count), true for intr_shared_ptr_hp, which also needs the hazard pointer
// base (CRTP on the final type) and TryAddRef (HpIntrusivePointee). TryAddRef
// exists in B<false> too but is never instantiated there: the spinlock pointer
// does not call it.
// The hooks are hand-written, not inherited from intr_pointee_base (see the
// file comment).
template <bool HasHazardBase>
struct B : public A,
           conditional_t<HasHazardBase, std::hazard_pointer_obj_base<B<HasHazardBase>>, no_hazard_base> {
  B(int i = 0) : A(i), ref_cnt_(0) {
    ++B_count;
  }
  ~B() { --B_count; }
  B(const B& x) = delete;
  B& operator=(const B& x) = delete;
  atomic<unsigned long> ref_cnt_;   // the intrusive strong count
  // The hooks (intr_pointee.h states their contract). AddRef is acq_rel,
  // stronger than the relaxed the contract needs; on x86-64 both orders
  // compile to the same `lock inc` when the result is unused.
  void AddRef() noexcept { ref_cnt_.fetch_add(1, std::memory_order_acq_rel); }
  // TryAddRef(): the reference form of the contract in intr_pointee.h
  // (increment iff nonzero; every observed 0 read with acquire; the CAS acquire
  // on success). Called only by intr_shared_ptr_hp::load().
  bool TryAddRef() noexcept {
    unsigned long count = ref_cnt_.load(std::memory_order_acquire);
    while (count != 0) {
      if (ref_cnt_.compare_exchange_weak(count, count + 1,
              std::memory_order_acquire, std::memory_order_relaxed)) {
        return true;
      }
      // The failed CAS refreshed `count` with a relaxed read; a 0 seen that
      // way does not synchronize. Re-read it with acquire.
      if (count == 0) count = ref_cnt_.load(std::memory_order_acquire);
    } // CAS loop while the count is nonzero
    return false;
  } // B::TryAddRef()
  bool DelRef() noexcept { return ref_cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1; }
  // Required by both pointers' concepts; this program never calls it.
  long use_count() const noexcept { return static_cast<long>(ref_cnt_.load(std::memory_order_relaxed)); }
}; // struct B

// The two pointers the benchmarks are typed over. Each policy names the
// pointee and the atomic; the policy's name is the template argument that
// appears in the row names (BM_deref<IntrShared>, BM_deref<IntrSharedHP>, ...),
// matching the _IntrShared / _IntrSharedHP suffixes of the SharedPtr harness.
struct IntrShared {
  using pointee = B<false>;
  using atomic = intr_shared_ptr<A, pointee>;
};

struct IntrSharedHP {
  using pointee = B<true>;
  using atomic = intr_shared_ptr_hp<A, pointee>;
};

// The two shared atomics of pointer P, p1 (initially B(42)) and q1 (initially
// B(7)). Function-local statics rather than namespace-scope objects because of
// the exit order, which matters for intr_shared_ptr_hp: a static is destroyed
// in the reverse order of the completion of its construction. mm_hp's exit
// guard, a namespace-scope object in mm_hp.cpp, is constructed before main();
// these are constructed on first use, inside a benchmark, so they are
// destroyed BEFORE the guard, and the B objects they release at exit are
// retired before the guard's final scan reclaims them. (Namespace-scope
// atomics in this TU have an unspecified order relative to the guard in
// another TU; destroyed after it, their B objects would stay retired but
// unreclaimed -- harmless, still reachable from mm_hp's retired list, but
// untidy.) intr_shared_ptr deletes at release and does not care; it uses the
// same form so that the bodies are one.
// Each benchmark binds the references once, before its timed loop, so the
// statics' initialization guard is not part of the measured work.
template <typename P>
typename P::atomic& shared_p1() {
  static typename P::atomic p1(typename P::atomic::shared_ptr_type(new typename P::pointee(42)));
  return p1;
}

template <typename P>
typename P::atomic& shared_q1() {
  static typename P::atomic q1(typename P::atomic::shared_ptr_type(new typename P::pointee(7)));
  return q1;
}

// The wall-clock records of every body follow gb_wall_clock.h's contract: one
// function-local static per instantiation (per operation and pointer),
// allocate() before the loop and, after it, report_wall() then release(), each
// called on every thread and doing its work on thread 0 alone. The calls are
// explicit, not a scope-bound owner, because no local with a destructor may be
// live across the timed loop: its cleanup region can change the loop's code.

// BM_deref: load(), read through the handle, release (see the file comment).
template <typename P>
void BM_deref(benchmark::State& state) {
  static constinit WallRecords records;
  typename P::atomic& p1 = shared_p1<P>();
  volatile A x;
  records.allocate(state);
  for (auto _ : WallTimed(state, records)) {
    benchmark::DoNotOptimize(x = *p1.load());
  }
  state.SetItemsProcessed(state.iterations());
  report_wall(state, records, static_cast<double>(state.iterations())*state.threads());
  records.release(state);
} // BM_deref()

// BM_copy: an atomic constructed from load() and destroyed (see the file
// comment). volatile keeps the otherwise unused atomic from being elided.
template <typename P>
void BM_copy(benchmark::State& state) {
  static constinit WallRecords records;
  typename P::atomic& p1 = shared_p1<P>();
  records.allocate(state);
  for (auto _ : WallTimed(state, records)) {
    volatile typename P::atomic q(p1.load());
  }
  state.SetItemsProcessed(state.iterations());
  report_wall(state, records, static_cast<double>(state.iterations())*state.threads());
  records.release(state);
} // BM_copy()

// BM_assign: q1.store(p1.load()) (see the file comment).
template <typename P>
void BM_assign(benchmark::State& state) {
  static constinit WallRecords records;
  typename P::atomic& p1 = shared_p1<P>();
  typename P::atomic& q1 = shared_q1<P>();
  records.allocate(state);
  for (auto _ : WallTimed(state, records)) {
    q1.store(p1.load());
  }
  state.SetItemsProcessed(state.iterations());
  report_wall(state, records, static_cast<double>(state.iterations())*state.threads());
  records.release(state);
} // BM_assign()

// BM_xassign: cross-assignment between p1 and q1, by the parity of the thread
// index (see the file comment). Thread 0 replaces both pointees before its
// loop: every thread's timer starts at the loop's start barrier, which thread
// 0 reaches only after the replacement, so the replacement is not timed. Both
// loops are wall-timed over the same records: each thread runs exactly one of
// them, and that one stamps the thread's record.
template <typename P>
void BM_xassign(benchmark::State& state) {
  static constinit WallRecords records;
  typename P::atomic& p1 = shared_p1<P>();
  typename P::atomic& q1 = shared_q1<P>();
  using shared_ptr_type = typename P::atomic::shared_ptr_type;
  if (state.thread_index() == 0) {
      p1.store(shared_ptr_type(new typename P::pointee(42)));
      q1.store(shared_ptr_type(new typename P::pointee(7)));
  }
  records.allocate(state);
  if (state.thread_index() & 1) {
    for (auto _ : WallTimed(state, records)) {
      q1.store(p1.load());
    }
  } else {
    for (auto _ : WallTimed(state, records)) {
      p1.store(q1.load());
    }
  }
  state.SetItemsProcessed(state.iterations());
  report_wall(state, records, static_cast<double>(state.iterations())*state.threads());
  records.release(state);
} // BM_xassign()

// Registration: for each thread count, each operation for both pointers in
// adjacent rows, so the two can be read side by side.
#define ALL_BENCHMARKS(N) \
BENCHMARK_TEMPLATE(BM_deref, IntrShared) ARGS(N);       \
BENCHMARK_TEMPLATE(BM_deref, IntrSharedHP) ARGS(N);     \
BENCHMARK_TEMPLATE(BM_copy, IntrShared) ARGS(N);        \
BENCHMARK_TEMPLATE(BM_copy, IntrSharedHP) ARGS(N);      \
BENCHMARK_TEMPLATE(BM_assign, IntrShared) ARGS(N);      \
BENCHMARK_TEMPLATE(BM_assign, IntrSharedHP) ARGS(N);    \
BENCHMARK_TEMPLATE(BM_xassign, IntrShared) ARGS(N);     \
BENCHMARK_TEMPLATE(BM_xassign, IntrSharedHP) ARGS(N);   \
struct dummy##N {}

// Thread counts (see the file comment on the oversubscribed rows).
ALL_BENCHMARKS(1);
ALL_BENCHMARKS(2);
ALL_BENCHMARKS(4);
ALL_BENCHMARKS(8);
ALL_BENCHMARKS(16);
ALL_BENCHMARKS(32);
ALL_BENCHMARKS(64);
ALL_BENCHMARKS(80);
ALL_BENCHMARKS(120);
ALL_BENCHMARKS(128);

BENCHMARK_MAIN();
