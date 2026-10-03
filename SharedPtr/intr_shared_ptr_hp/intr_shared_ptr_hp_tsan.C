// Protocol smoke test of intr_shared_ptr_hp under TSan (no GoogleTest), the
// counterpart of intr_shared_ptr/intr_shared_ptr_tsan.C. It runs concurrent
// load()/store() traffic and checks the reference counts, but it is NOT a
// lifetime test: TSan cannot see hazard-pointer lifetime violations (every
// load() ends with a release store to its hazard record, which hides the race),
// and with only three B objects mm_hp's scan threshold (1000 retirements) is
// never crossed while the threads run, so no scan ever runs concurrently with
// them. Retired B objects are destroyed by explicit drains (hp_drain.h) on the
// main thread after the joins, before every B_count check that follows a
// release (Test1's check inside its scope follows none and needs no drain).
#include "intr_shared_ptr_hp.h"
// B derives from std::hazard_pointer_obj_base. Found through -I. as
// SharedPtr/mm_hp/ (there is no intr_shared_ptr_hp/mm_hp/).
#include "mm_hp/mm_hp.hpp"
#include "hp_drain.h"

#include <cassert>
#include <atomic>
#include <thread>
#include <vector>

// Defined after <cassert> was included, so it does not disable the assert()s
// below; an include after this line that re-included <cassert> would.
#define NDEBUG 

using namespace std;

struct A {
  int i;
  A(int i = 0) : i(i) {}
};

// Live B objects. Atomic because under intr_shared_ptr_hp ~B runs inside an
// mm_hp scan, on whichever thread crosses the threshold (here: the main
// thread's drains, but nothing in B may rely on that).
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
  // TSAN bug demo (also try TSAN_OPTIONS="force_seq_cst_atomics=1" ./intr_shared_ptr_hp_tsan)
  //void AddRef() noexcept { ref_cnt_.fetch_add(1, std::memory_order_relaxed); }
  //bool DelRef() noexcept { return ref_cnt_.fetch_sub(1, std::memory_order_relaxed) == 1; }
}; // struct B

void copy_ptr(const intr_shared_ptr_hp<A, B>& p, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    vector<intr_shared_ptr_hp<A, B>::shared_ptr_type> v;
    for (size_t j = 0; j < M; ++j) {
      v.push_back(p.load());
    }
  }
}

void Test1() {
  {
    intr_shared_ptr_hp<A, B> p(typename intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(42)));
    {
      thread t1(copy_ptr, std::ref(p), 100, 10000);
      thread t2(copy_ptr, std::ref(p), 100, 10000);
      t1.join();
      t2.join();
    }
    assert(p.load());
    assert(2 == p.load().use_count());
    assert(1 == B_count);   // no release reached 0: nothing retired, no drain needed
    assert(42 == p.load()->i);
  }
  drain_reclamation();      // p's destructor retired the B; the threads are joined
  assert(0 == B_count);
} // Test1()

void cross_assign(intr_shared_ptr_hp<A, B>* p, intr_shared_ptr_hp<A, B>* q, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    p->store(q->load());
  }
}

void Test2() {
  {
    intr_shared_ptr_hp<A, B> p(typename intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(42)));
    intr_shared_ptr_hp<A, B> q(typename intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(7)));
    intr_shared_ptr_hp<A, B> r(typename intr_shared_ptr_hp<A, B>::shared_ptr_type(new B(314)));
    {
      thread t1(cross_assign, &p, &q, 10000);
      thread t2(cross_assign, &q, &r, 10000);
      thread t3(cross_assign, &r, &p, 10000);
      t1.join();
      t2.join();
      t3.join();
    }
    const intr_shared_ptr_hp<A, B>::shared_ptr_type pv = p.load();
    const intr_shared_ptr_hp<A, B>::shared_ptr_type qv = q.load();
    const intr_shared_ptr_hp<A, B>::shared_ptr_type rv = r.load();
    assert(pv);
    assert(qv);
    assert(rv);
    // The three words need NOT converge to one B. Each thread's last store
    // copies whatever its source word held at its last load, e.g. t1's last
    // p.store(q.load()), then t2's last q.store(r.load()), then t3's last
    // r.store(p.load()) legally ends with p == r != q (two live B objects);
    // all three can even stay distinct. What the program guarantees: after the
    // drain, exactly the B objects some word still holds are alive (each held
    // one at least once, each other one retired and destroyed).
    const unsigned long distinct = 1 + (qv.get_raw() != pv.get_raw()) +
                                   (rv.get_raw() != pv.get_raw() && rv.get_raw() != qv.get_raw());
    drain_reclamation();    // the B objects no atomic holds any more were retired
    assert(distinct == B_count);
  }
  drain_reclamation();      // the remaining B objects, retired by the atomics' destructors
  assert(0 == B_count);
} // Test2()

int main() {
  force_early_scan();       // registers membarrier before any thread starts (hp_drain.h)
  Test1();
  Test2();
}
