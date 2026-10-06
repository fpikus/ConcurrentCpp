// Concurrency smoke test of the two intrusive atomic shared pointers of this
// directory, intr_shared_ptr (intr_shared_ptr.h) and intr_shared_ptr_hp
// (intr_shared_ptr_hp.h), under ThreadSanitizer. Standalone (no GoogleTest),
// built -O1 -fsanitize=thread WITHOUT -DNDEBUG: its checks are assert()s, and
// TSan's report of a data race makes the process exit nonzero (66) at the end.
// Every test runs for both pointers, typed over the pointer:
//
//   Test1  two threads each load() the same atomic 1,000,000 times into
//          short-lived vectors of handles; then the count and the payload must
//          be intact (exactly the atomic and one handle own the object).
//          Exercises concurrent readers on one word: the spinlock pointer's
//          readers all take the lock bit, the hazard pointer one's publish
//          hazards and run TryAddRef's CAS loop on one count.
//   Test2  three threads cross-assign p <- q, q <- r, r <- p 10,000 times each
//          (load from one word, store into another, each word read by one
//          thread and written by another); then exactly the B objects some
//          word still holds are alive, and none after the atomics die.
//          Exercises concurrent store/load on the same words and the release
//          of replaced pointees (delete, or retire) on the worker threads.
//
// What it is NOT: a lifetime test of intr_shared_ptr_hp. TSan cannot see
// hazard-pointer lifetime violations (every load() that protects an object
// ends with a release store to its hazard record, which hides the race; see
// intr_shared_ptr_hp.h), and with only three B objects mm_hp's scan threshold
// (1000 retirements) is never crossed while the threads run, so no scan runs
// concurrently with them. The hazard pointer protocol's oracles are the seam
// tests (intr_shared_ptr_hp_seam_test.C). What TSan does check here: the
// count's orders and every access the pointers make outside the protocol (a
// relaxed DelRef, say: see the demo in B).
//
// Reclamation differs between the two and is the one thing the bodies must
// know: intr_shared_ptr deletes a pointee at its last release, on the
// releasing thread; intr_shared_ptr_hp retires it, and an mm_hp scan destroys
// it later. Each pointer's policy below has a reclaim() that makes every
// pending destruction happen (nothing for the spinlock pointer, a drain of
// mm_hp, hp_drain.h, for the hazard pointer one); the tests call it on the
// main thread, after the joins, before every B_count check that follows a
// release.
//
// Usage: intr_shared_ptr_tsan [pointer [test]], pointer one of IntrShared,
// IntrSharedHP or all (the default), test 1, 2 or all (the default); e.g.
// `intr_shared_ptr_tsan IntrShared 2` runs Test2 of the spinlock pointer only
// (for repeated runs of one test).
#include "intr_shared_ptr.h"
#include "intr_shared_ptr_hp.h"     // brings mm_hp/ (std::hazard_pointer_obj_base)
#include "hp_drain.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef NDEBUG
#error "intr_shared_ptr_tsan.C: its checks are assert()s; build it without NDEBUG"
#endif

using namespace std;

// The interface type T of both pointers.
struct A {
  int i;
  A(int i = 0) : i(i) {}
};

// Live B objects, of both pointee types. Atomic because B objects are
// destroyed on worker threads (intr_shared_ptr: by whichever thread releases
// the last reference, in Test2 any of the three) or inside an mm_hp scan
// (intr_shared_ptr_hp: on whichever thread crosses the threshold -- here the
// main thread's drains, but nothing in B may rely on that).
static atomic<unsigned long> B_count{0};

// B's base when it is NOT a hazard pointer pointee: empty.
struct no_hazard_base {};

// The pointee U of both pointers, with T = A. HasHazardBase selects the
// pointer: false for intr_shared_ptr (IntrusivePointee: AddRef, DelRef,
// use_count), true for intr_shared_ptr_hp, which also needs the hazard pointer
// base (CRTP on the final type) and TryAddRef (HpIntrusivePointee). TryAddRef
// exists in B<false> too but is never instantiated there. The hooks are
// hand-written rather than inherited from intr_pointee_base so that the TSan
// demo below can weaken them.
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
  // stronger than the relaxed the contract needs.
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
  long use_count() const noexcept { return static_cast<long>(ref_cnt_.load(std::memory_order_relaxed)); }
  // TSAN bug demo: replace the two hooks above by these and TSan reports the
  // destruction of B racing with the other threads' last accesses to it (also
  // try TSAN_OPTIONS="force_seq_cst_atomics=1" ./intr_shared_ptr_tsan):
  //void AddRef() noexcept { ref_cnt_.fetch_add(1, std::memory_order_relaxed); }
  //bool DelRef() noexcept { return ref_cnt_.fetch_sub(1, std::memory_order_relaxed) == 1; }
}; // struct B

// The two pointers the tests are typed over. Each policy names the pointee and
// the atomic, says how to make every pending destruction happen (reclaim(),
// see the file comment), and gives the name main() selects it by.
struct IntrShared {
  using pointee = B<false>;
  using atomic = intr_shared_ptr<A, pointee>;
  static constexpr const char* name = "IntrShared";
  // Synchronous reclamation: the last release already deleted the object.
  static void reclaim() {}
};

struct IntrSharedHP {
  using pointee = B<true>;
  using atomic = intr_shared_ptr_hp<A, pointee>;
  static constexpr const char* name = "IntrSharedHP";
  // One drain destroys everything retired before it (the pointees own no
  // pointers, so there is no cascade); requires quiescence: the callers have
  // joined their threads.
  static void reclaim() { drain_reclamation(); }
};

// Test1's reader: N rounds of M load()s of `p` into a vector, which drops all
// M handles at the end of the round.
template <typename P>
void copy_ptr(const typename P::atomic& p, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    vector<typename P::atomic::shared_ptr_type> v;
    for (size_t j = 0; j < M; ++j) {
      v.push_back(p.load());
    }
  }
}

// Test1 (see the file comment). No release reaches 0 while the threads run,
// so nothing needs reclaiming before the first B_count check.
template <typename P>
void Test1() {
  using atomic_type = typename P::atomic;
  {
    atomic_type p(typename atomic_type::shared_ptr_type(new typename P::pointee(42)));
    {
      thread t1(copy_ptr<P>, std::ref(p), 100, 10000);
      thread t2(copy_ptr<P>, std::ref(p), 100, 10000);
      t1.join();
      t2.join();
    }
    assert(p.load());
    assert(2 == p.load().use_count());  // the atomic and the temporary handle
    assert(1 == B_count);
    assert(42 == p.load()->i);
  }
  P::reclaim();             // p's destructor released the B (deleted, or retired)
  assert(0 == B_count);
} // Test1()

// Test2's worker: N times p <- q.
template <typename P>
void cross_assign(typename P::atomic* p, typename P::atomic* q, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    p->store(q->load());
  }
}

// Test2 (see the file comment). The three words need NOT converge to one B.
// Each thread's last store copies whatever its source word held at its last
// load, e.g. t1's last p.store(q.load()), then t2's last q.store(r.load()),
// then t3's last r.store(p.load()) legally ends with p == r != q (two live B
// objects); all three can even stay distinct (each thread's last load
// preceded the other threads' last stores). What the program guarantees: once
// pending destructions have run, exactly the B objects some word still holds
// are alive -- each B any word stopped holding lost its last reference when
// that happened (the threads' handles are gone after the joins) and was
// destroyed, every B a word holds is alive. Under TSan's timing the words
// usually do converge to one B, so in this build the check rarely sees more
// than one survivor.
template <typename P>
void Test2() {
  using atomic_type = typename P::atomic;
  using shared_ptr_type = typename atomic_type::shared_ptr_type;
  {
    atomic_type p(shared_ptr_type(new typename P::pointee(42)));
    atomic_type q(shared_ptr_type(new typename P::pointee(7)));
    atomic_type r(shared_ptr_type(new typename P::pointee(314)));
    {
      thread t1(cross_assign<P>, &p, &q, 10000);
      thread t2(cross_assign<P>, &q, &r, 10000);
      thread t3(cross_assign<P>, &r, &p, 10000);
      t1.join();
      t2.join();
      t3.join();
    }
    // The handles keep the held pointees alive; they are the same objects the
    // words hold, so they do not change what B_count must be.
    const shared_ptr_type pv = p.load();
    const shared_ptr_type qv = q.load();
    const shared_ptr_type rv = r.load();
    assert(pv);
    assert(qv);
    assert(rv);
    const unsigned long distinct = 1 + (qv.get_raw() != pv.get_raw()) +
                                   (rv.get_raw() != pv.get_raw() && rv.get_raw() != qv.get_raw());
    P::reclaim();           // the B objects no word holds any more (retired ones)
    assert(distinct == B_count);
  }
  P::reclaim();             // the remaining B objects, released by the atomics' destructors
  assert(0 == B_count);
} // Test2()

// Runs the selected tests of pointer P: test is "1", "2" or "all".
template <typename P>
void run_tests(const char* test) {
  if (strcmp(test, "all") == 0 || strcmp(test, "1") == 0) Test1<P>();
  if (strcmp(test, "all") == 0 || strcmp(test, "2") == 0) Test2<P>();
}

// Usage in the file comment. An unknown pointer or test name is an error
// (exit 2), so a typo in a repeat loop cannot pass by running nothing.
int main(int argc, char** argv) {
  const char* const pointer = argc > 1 ? argv[1] : "all";
  const char* const test = argc > 2 ? argv[2] : "all";
  const bool all_pointers = strcmp(pointer, "all") == 0;
  if ((!all_pointers && strcmp(pointer, IntrShared::name) != 0 &&
       strcmp(pointer, IntrSharedHP::name) != 0) ||
      (strcmp(test, "all") != 0 && strcmp(test, "1") != 0 && strcmp(test, "2") != 0)) {
    fprintf(stderr, "usage: %s [IntrShared|IntrSharedHP|all [1|2|all]]\n", argv[0]);
    return 2;
  }
  // Registers membarrier before any thread starts (hp_drain.h says why TSan's
  // sensitivity depends on it), whichever pointer runs: unconditional, so
  // that every selection starts from the same mm_hp state.
  force_early_scan();
  if (all_pointers || strcmp(pointer, IntrShared::name) == 0) run_tests<IntrShared>(test);
  if (all_pointers || strcmp(pointer, IntrSharedHP::name) == 0) run_tests<IntrSharedHP>(test);
} // main()
