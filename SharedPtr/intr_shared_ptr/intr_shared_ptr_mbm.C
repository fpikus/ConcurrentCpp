#include "intr_shared_ptr.h"

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

static unsigned long B_count = 0;
struct B : public A {
  B(int i = 0) : A(i), ref_cnt_(0) {
    ++B_count;
  }
  ~B() { --B_count; }
  B(const B& x) = delete;
  B& operator=(const B& x) = delete;
  atomic<unsigned long> ref_cnt_;
  void AddRef() { ref_cnt_.fetch_add(1, std::memory_order_acq_rel); }
  bool DelRef() { return ref_cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1; }
};

intr_shared_ptr<A, B> p1(intr_shared_ptr<A, B>::shared_ptr_type(new B(42)));

void BM_intr_shared_ptr_deref(benchmark::State& state) {
  volatile A x;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(x = *p1.load());
  }
}

void BM_intr_shared_ptr_copy(benchmark::State& state) {
  while (state.KeepRunning()) {
    volatile intr_shared_ptr<A, B> q(p1.load());
  }
}

intr_shared_ptr<A, B> q1(intr_shared_ptr<A, B>::shared_ptr_type(new B(7)));

void BM_intr_shared_ptr_assign(benchmark::State& state) {
  while (state.KeepRunning()) {
    q1.store(p1.load());
  }
}

void BM_intr_shared_ptr_xassign(benchmark::State& state) {
  if (state.thread_index() == 0) {
      p1.store(intr_shared_ptr<A, B>::shared_ptr_type(new B(42)));
      q1.store(intr_shared_ptr<A, B>::shared_ptr_type(new B(7)));
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
}

#define ALL_BENCHMARKS(N) \
BENCHMARK(BM_intr_shared_ptr_deref) ARGS(N);        \
BENCHMARK(BM_intr_shared_ptr_copy) ARGS(N);         \
BENCHMARK(BM_intr_shared_ptr_assign) ARGS(N);       \
BENCHMARK(BM_intr_shared_ptr_xassign) ARGS(N);      \
struct dummy##N {}

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
