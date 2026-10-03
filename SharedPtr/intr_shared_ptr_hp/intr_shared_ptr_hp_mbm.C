// Microbenchmark of intr_shared_ptr_hp, the counterpart of
// intr_shared_ptr/intr_shared_ptr_mbm.C: the same four operations on
// program-wide shared atomics (dereference of a load, copy of a load, store
// of a load, and two-way cross-assignment), with the rows named
// BM_intr_shared_ptr_hp_* so that they do not collide with the original
// program's rows when both run (make run_benchmarks).
#include "intr_shared_ptr_hp.h"
// B derives from std::hazard_pointer_obj_base. Found through -I. as
// SharedPtr/mm_hp/ (there is no intr_shared_ptr_hp/mm_hp/).
#include "mm_hp/mm_hp.hpp"

#include <string.h>
#include <atomic>
#include <memory>

#include "benchmark/benchmark.h"

#define REPEAT2(x) {x} {x}
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT64(x) REPEAT32(x) REPEAT32(x)
#define REPEAT(x) REPEAT64(x)

#define ARGS(N) \
  ->Threads(N) \
  ->UseRealTime()

using namespace std;

struct A {
  int i;
  A(int i = 0) : i(i) {}
  A& operator=(const A& rhs) { i = rhs.i; return *this; }
  volatile A& operator=(const A& rhs) volatile { i = rhs.i; return *this; }
};

// Live B objects (never read here; kept as in the original program). Atomic
// because under intr_shared_ptr_hp ~B runs inside an mm_hp scan, on whichever
// benchmark thread crosses the threshold, and two threads may scan at once.
static atomic<unsigned long> B_count{0};
// The pointee: derives from the hazard pointer base, as intr_shared_ptr_hp
// requires, and provides the four noexcept hooks.
struct B : public A, std::hazard_pointer_obj_base<B> {
  B(int i = 0) : A(i), ref_cnt_(0) {
    ++B_count;
  }
  ~B() { --B_count; }
  B(const B& x) = delete;
  B& operator=(const B& x) = delete;
  atomic<unsigned long> ref_cnt_;   // the intrusive strong count
  void AddRef() noexcept { ref_cnt_.fetch_add(1, std::memory_order_acq_rel); }
  // TryAddRef(): increment the strong count if and only if it is nonzero.
  // Returns true iff it incremented; returns false iff it observed a count of
  // 0, in which case the count is left at 0 (an object at 0 is retired, or
  // about to be retired, and must never be revived). Required only by
  // intr_shared_ptr_hp; the other pointer policies never call it. Memory
  // orders, all load-bearing: the load that observes 0 is ACQUIRE; the CAS is
  // ACQUIRE on success and relaxed on failure; a failed CAS whose refreshed
  // value is 0 re-reads the count with an acquire load before returning false,
  // so EVERY observed 0 was read with acquire. Why: intr_shared_ptr_hp::load()
  // calls this on an object pinned only by a hazard pointer. An observed 0
  // must synchronize with the release sequence headed by the DelRef that
  // produced it, so that the loader's next acquire reload of the word is
  // guaranteed to see the store that unpublished the object; with a relaxed
  // zero-observation the loader can re-read the stale word forever (model
  // checked: livelock). The ACQUIRE on CAS success makes the loader's
  // post-increment re-validation of the word see a swing that released the
  // word's own reference. The value a successful CAS consumes need not have
  // been read with acquire: a failed CAS refreshes it with a relaxed read (only
  // a refreshed 0 is re-read with acquire), and other threads' RMWs may have
  // rewritten it. A CAS that succeeds with relaxed order on a value it read
  // from the release sequence headed by the swing's DelRef does not
  // synchronize with that DelRef, so the re-validation reload may return the
  // stale word and LockFreeList::~Node's walk can judge a node exclusive
  // (count 1) that this loader then owns with a stale next (model checked:
  // assertion failure). Every operation on the count is an RMW. Cost: nil on
  // x86-64; LDAR/LDAXR on aarch64.
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
  } // TryAddRef()
  bool DelRef() noexcept { return ref_cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1; }
  long use_count() const noexcept { return static_cast<long>(ref_cnt_.load(std::memory_order_relaxed)); }
}; // struct B

// The two shared atomics. Function-local statics rather than namespace-scope
// objects (as in the original program) because of the exit order: a static is
// destroyed in the reverse order of the completion of its construction.
// mm_hp's exit guard, a namespace-scope object in mm_hp.cpp, is constructed
// before main(); these are constructed on first use, inside a benchmark, so
// they are destroyed BEFORE the guard, and the B objects they release at exit
// are retired before the guard's final scan reclaims them. (Namespace-scope
// atomics in another TU have an unspecified order relative to the guard; if
// destroyed after it, their B objects would stay retired but unreclaimed --
// harmless, still reachable from mm_hp's retired list, but untidy.)
// Each benchmark binds the references once, before its timed loop, so the
// statics' initialization guard is not part of the measured work.
intr_shared_ptr_hp<A, B>& shared_p1() {
  static intr_shared_ptr_hp<A, B> p1(intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(42)));
  return p1;
}

intr_shared_ptr_hp<A, B>& shared_q1() {
  static intr_shared_ptr_hp<A, B> q1(intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(7)));
  return q1;
}

void BM_intr_shared_ptr_hp_deref(benchmark::State& state) {
  intr_shared_ptr_hp<A, B>& p1 = shared_p1();
  volatile A x;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(x = *p1.load());
  }
}

void BM_intr_shared_ptr_hp_copy(benchmark::State& state) {
  intr_shared_ptr_hp<A, B>& p1 = shared_p1();
  while (state.KeepRunning()) {
    volatile intr_shared_ptr_hp<A, B> q(p1.load());
  }
}

void BM_intr_shared_ptr_hp_assign(benchmark::State& state) {
  intr_shared_ptr_hp<A, B>& p1 = shared_p1();
  intr_shared_ptr_hp<A, B>& q1 = shared_q1();
  while (state.KeepRunning()) {
    q1.store(p1.load());
  }
}

void BM_intr_shared_ptr_hp_xassign(benchmark::State& state) {
  intr_shared_ptr_hp<A, B>& p1 = shared_p1();
  intr_shared_ptr_hp<A, B>& q1 = shared_q1();
  if (state.thread_index() == 0) {
      p1.store(intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(42)));
      q1.store(intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(7)));
  }
  if (state.thread_index() & 1) {
    while (state.KeepRunning()) {
      q1.store(p1.load());
    }
  } else {
    while (state.KeepRunning()) {
      p1.store(q1.load());
    }
  }
} // BM_intr_shared_ptr_hp_xassign()

#define ALL_BENCHMARKS(N) \
BENCHMARK(BM_intr_shared_ptr_hp_deref) ARGS(N);     \
BENCHMARK(BM_intr_shared_ptr_hp_copy) ARGS(N);      \
BENCHMARK(BM_intr_shared_ptr_hp_assign) ARGS(N);    \
BENCHMARK(BM_intr_shared_ptr_hp_xassign) ARGS(N);   \
struct dummy##N {}

// Thread counts as in the original program. The rows above the machine's CPU
// count are oversubscribed (on a 16-CPU machine: 32, 64, 80, 120 and 128) and
// measure scheduling as much as the pointer; compare rows with at most as many
// threads as CPUs.
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
