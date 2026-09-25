// Count the per-offer OUTCOMES of CAS-max vs DCLP-max under the grow_always
// feed, to see why DCLP beats CAS without inferring. Not a timing test -- the
// counters would perturb timing; here we only want the outcome distribution.
//
// Same feed as atomic_max_bm.C grow: thread t offers t+dn, t+2dn, ... with
// dn = nthreads, so every offer is strictly increasing (a new max FOR THAT
// THREAD), but most are stale by the time they reach the shared word.
//
// Outcomes tallied. `fast` and `updated` are one per offer; CAS `wasted` is one
// per FAILED EXCHANGE, so a CAS offer can add several (DCLP: at most one):
//   CAS : fast    -- val <= cur at the acquire load; no compare_exchange at all
//         updated -- a compare_exchange SUCCEEDED (the max advanced)
//         wasted  -- a compare_exchange FAILED (RMW that bounced the line, lost)
//   DCLP: fast    -- val <= cur at the acquire load; lock never taken
//         updated -- took the lock and stored a new max
//         wasted  -- took the lock but the max had already moved past val
//                    (lock acquired, no store)
//
// cas_offer() and dclp_offer() are instrumented COPIES of atomic_max()
// (atomic_max.h) and of BM_dclp_grow's loop body (atomic_max_bm.C): same memory
// orders, same weak-CAS retry that gives up once the maximum has passed val,
// same acquire probe / relaxed re-check / release store. The counts describe
// the benchmarked code only while the copies stay in step with it; change one,
// change both. (The branch-layout hint in the originals affects timing, not
// outcomes, so the copies omit it.)
//
// Usage: ./atomic_max_count [nthreads [iters_per_thread]]
//   nthreads defaults to all configured CPUs, iters_per_thread to 1,000,000.
//   Built by `make benchmarks` (build/<hostname>/atomic_max_count).
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "spinlock.h"                    // shipped SpinLock (../Spinlock)

alignas(64) static std::atomic<unsigned long> nmax;
alignas(64) static SpinLock lock;

struct Counts { unsigned long fast = 0, updated = 0, wasted = 0; };

// CAS-max, instrumented: the while form, counting each outcome.
static void cas_offer(unsigned long val, Counts& c) {
  unsigned long cur = nmax.load(std::memory_order_acquire);
  if (val <= cur) { ++c.fast; return; }               // dodge: pure read
  while (true) {
    if (nmax.compare_exchange_weak(cur, val,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire)) {
      ++c.updated; return;                            // won the exchange
    }
    ++c.wasted;                                       // failed exchange (RMW bounce)
    if (val <= cur) return;                           // someone passed us: give up
  } // retry loop
} // cas_offer()

// DCLP-max, instrumented: acquire read, lock only to update.
static void dclp_offer(unsigned long val, Counts& c) {
  if (val <= nmax.load(std::memory_order_acquire)) { ++c.fast; return; } // dodge
  std::lock_guard<SpinLock> guard(lock);
  if (val > nmax.load(std::memory_order_relaxed)) {
    nmax.store(val, std::memory_order_release);
    ++c.updated;                                      // locked and stored
  } else {
    ++c.wasted;                                       // locked, but max already moved
  }
} // dclp_offer()

// Run one mechanism across nthreads for iters offers each, summing outcomes.
template <typename Offer>
static Counts run(Offer offer, int nthreads, unsigned long iters) {
  nmax.store(0, std::memory_order_relaxed);
  std::vector<std::thread> threads;
  std::vector<Counts> per(nthreads);
  for (int t = 0; t < nthreads; ++t) {
    threads.emplace_back([&, t] {
      unsigned long n = t, dn = nthreads;
      Counts c;
      for (unsigned long i = 0; i < iters; ++i) { n += dn; offer(n, c); }
      per[t] = c;
    });
  } // spawn
  for (std::thread& th : threads) th.join();
  Counts tot;
  for (const Counts& c : per) {
    tot.fast += c.fast; tot.updated += c.updated; tot.wasted += c.wasted;
  } // sum
  return tot;
} // run()

// Print one mechanism's tallies. `offers` is the number of offers made
// (threads x iterations): the fast-path share is a share of OFFERS, not of
// tallies, because a CAS offer can be tallied as several failed exchanges.
static void report(const char* name, const Counts& c, unsigned long offers) {
  double pct = offers ? 100.0/offers : 0.0;
  printf("  %-5s fast=%-12lu updated=%-10lu wasted=%-12lu | "
         "fast=%.2f%% of offers, wasted/updated=%.1f\n",
         name, c.fast, c.updated, c.wasted,
         c.fast*pct, c.updated ? double(c.wasted)/c.updated : 0.0);
} // report()

int main(int argc, char** argv) {
  int nthreads = argc > 1 ? atoi(argv[1]) : int(sysconf(_SC_NPROCESSORS_CONF));
  unsigned long iters = argc > 2 ? strtoul(argv[2], nullptr, 10) : 1000000;
  if (nthreads < 1 || iters < 1) {
    fprintf(stderr, "usage: %s [nthreads >= 1 [iters_per_thread >= 1]]\n", argv[0]);
    return 1;
  }
  printf("grow_always, threads=%d, iters/thread=%lu\n", nthreads, iters);
  const unsigned long offers = static_cast<unsigned long>(nthreads)*iters;
  report("CAS",  run(cas_offer,  nthreads, iters), offers);
  report("DCLP", run(dclp_offer, nthreads, iters), offers);
  return 0;
} // main()
