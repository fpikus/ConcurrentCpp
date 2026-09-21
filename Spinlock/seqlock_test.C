// Unit tests for SeqLock (seqlock.h): the single-thread interface contract,
// writer exclusion under contention, and the reader protocol under concurrent
// writes. Run under TSan: as in spinlock_test.C, the shared state the tests
// keep NEXT TO the lock is deliberately plain (non-atomic), so that a missing
// happens-before edge is a reported data race and not just a (rarely) wrong
// value.
//
// What can and cannot fail here. The payload is one atomic word, so a reader
// can never see a torn value, whatever the protocol does: through the public
// interface alone, a SeqLock whose reader skipped the odd check or the
// validation is indistinguishable from a correct one (it would return the
// value of a write in flight, which is a legitimate value a moment later).
// Three devices give the tests something to catch; all the privileged access
// they need goes through SeqLockTestPeer, which calls the header's own
// private protocol steps and copies none of them:
//   - The ledger (ReadersAndWriters): writers record each generation in a
//     plain array from inside update()'s callable; a reader looks up the
//     generation it loaded. That lookup is race-free only if load()
//     synchronized with the write that produced the value.
//   - The validated counter (ReadersAndWriters): the payload carries its own
//     generation number, and the peer reports which counter value a read
//     validated. The header's invariant -- a reader that validates 2k returns
//     exactly generation k -- is then checked literally, on every read. This
//     is what catches a writer that does not open the odd window, or opens it
//     after the payload store. The final counter value (2 per write, in
//     MultiWriterUpdate and MultiWriterStore) catches a lost counter update,
//     i.e. missing writer exclusion, where the payload cannot show it.
//   - The two-step writer (ReaderWaitsForWriteInFlight,
//     ReadersNeverSeeWriteInFlight): a test-side writer that, like a
//     multi-word write, passes through an intermediate payload value while the
//     counter is odd. A reader that returns the intermediate value has a
//     broken odd check or a broken validation.
// Mutation check, on x86 under TSan. Caught: load() without the validation,
// or without the odd check; update(), or store(), without the lock; write()
// without the odd store, or with the odd store after the payload store; all
// four release/acquire orderings of seqlock.h relaxed together; SpinLock's
// orderings relaxed. NOT caught, and not catchable here: any SINGLE ordering
// weakened from acquire/release to relaxed. On x86 the generated code is the
// same, and in any one execution TSan observes, the remaining edges cover for
// the missing one. See the header for the argument that each is needed.
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "seqlock.h"

// The test payload: generation g in the low half, ~g in the high half. Only a
// value written whole satisfies the invariant; the two-step writer's
// intermediate value, and any garbage, do not.
static constexpr unsigned long pack(unsigned long g) {
  return ((~g & 0xFFFFFFFFul) << 32) | (g & 0xFFFFFFFFul);
}
// The generation number a payload claims to be: its low half. Meaningful only
// for a well-formed payload, but defined for any value.
static constexpr unsigned long generation(unsigned long v) { return v & 0xFFFFFFFFul; }
// True if `v` is pack(g) for some g, i.e. its two halves agree.
static constexpr bool well_formed(unsigned long v) { return pack(generation(v)) == v; }
// Deliberately ill-formed: what a reader must never return.
static constexpr unsigned long in_flight = 0xDEADul;
static_assert(!well_formed(in_flight));

// What a writer records in the ledger for generation g; anything but 0, the
// ledger's initial content.
static constexpr unsigned long ledger_entry(unsigned long g) { return g*2654435761ul + 1; }

// Test-side access to SeqLock's internals (SeqLock befriends this name). It
// adds no protocol of its own: every counter and payload access below is a
// call to one of the private steps SeqLock::write() and SeqLock::load() are
// themselves built from, so a defect in those steps is a defect the tests see.
struct SeqLockTestPeer {
  // First half of a two-step write: take the writer lock, make the counter
  // odd, store `value` as the intermediate payload. Returns the token
  // end_write() needs (the odd counter value). Until end_write() is called the
  // counter stays odd and `value` stays in place, for as long as the test
  // likes.
  template <typename T>
  [[nodiscard]] static std::uint64_t begin_write(SeqLock<T>& s, T value) {
    s.lock_.lock();
    const std::uint64_t odd_seq = s.begin_write();
    s.store_payload(value);
    return odd_seq;
  } // SeqLockTestPeer::begin_write()

  // Second half: store the final `value`, make the counter even, unlock.
  // `odd_seq` is what the matching begin_write() returned.
  template <typename T>
  static void end_write(SeqLock<T>& s, std::uint64_t odd_seq, T value) {
    s.store_payload(value);
    s.end_write(odd_seq);
    s.lock_.unlock();
  } // SeqLockTestPeer::end_write()

  // s.load(), also reporting through `seq` the counter value the successful
  // attempt validated. This IS load(): both are SeqLock::load_validated().
  template <typename T>
  static T load_validated(const SeqLock<T>& s, std::uint64_t& seq) {
    const typename SeqLock<T>::Snapshot snapshot = s.load_validated();
    seq = snapshot.seq;
    return snapshot.value;
  } // SeqLockTestPeer::load_validated()

  // The raw counter. Meant for a quiescent SeqLock (all writers joined), where
  // it must be exactly twice the number of writes ever made; acquire, so that
  // the last write's closing release store is what it observes.
  template <typename T>
  static std::uint64_t seq(const SeqLock<T>& s) {
    return s.seq_.load(std::memory_order_acquire);
  }
}; // struct SeqLockTestPeer

TEST(SeqLockTest, InitialValue) {
  SeqLock<unsigned long> zero;
  EXPECT_EQ(zero.load(), 0ul);
  SeqLock<unsigned long> s(42);
  EXPECT_EQ(s.load(), 42ul);
  const SeqLock<long> c(-7);            // load() is const
  EXPECT_EQ(c.load(), -7);
}

TEST(SeqLockTest, StoreLoad) {
  SeqLock<unsigned long> s;
  s.store(1);
  EXPECT_EQ(s.load(), 1ul);
  s.store(pack(5));
  EXPECT_EQ(s.load(), pack(5));
  EXPECT_EQ(s.load(), pack(5));         // loading does not disturb the value
}

// update() hands the callable the current value, exactly once, and stores what
// it returns; it accepts lvalue and rvalue callables, and any one-word T.
TEST(SeqLockTest, Update) {
  SeqLock<unsigned long> s(10);
  int calls = 0;
  s.update([&](unsigned long v) { ++calls; EXPECT_EQ(v, 10ul); return v + 5; });
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(s.load(), 15ul);
  auto twice = [](unsigned long v) { return 2*v; };
  s.update(twice);
  EXPECT_EQ(s.load(), 30ul);

  SeqLock<double> d(1.5);
  d.update([](double v) { return v*2; });
  EXPECT_EQ(d.load(), 3.0);

  int x = 0, y = 0;
  SeqLock<int*> p(&x);
  p.update([&](int* v) { EXPECT_EQ(v, &x); return &y; });
  EXPECT_EQ(p.load(), &y);
} // Update

// A callable that throws leaves the payload unchanged and the lock free: the
// following store() would hang on the lock, and the load() on an odd counter,
// if either were left behind.
TEST(SeqLockTest, UpdateThrows) {
  SeqLock<unsigned long> s(7);
  EXPECT_THROW(s.update([](unsigned long) -> unsigned long { throw 1; }), int);
  EXPECT_EQ(s.load(), 7ul);
  s.store(8);
  EXPECT_EQ(s.load(), 8ul);
}

// A reader must not return while a write is in flight. The peer holds the
// counter odd with an ill-formed payload in place; the reader announces that
// it is running and calls load(); ~50 ms later load() must still be spinning,
// and once the write completes it must return the final value.
// A load() that lost its odd check returns the ill-formed payload as soon as
// it runs, and fails both expectations. The only way for it to pass is for
// the reader thread, having announced itself, to get no CPU time at all for
// the whole 50 ms. The same goes for a writer that never makes the counter
// odd.
TEST(SeqLockTest, ReaderWaitsForWriteInFlight) {
  SeqLock<unsigned long> s(pack(0));
  const std::uint64_t odd_seq = SeqLockTestPeer::begin_write(s, in_flight);
  EXPECT_EQ(odd_seq, 1u);
  std::atomic<bool> started{false};     // reader: "I am running, load() is next"
  std::atomic<bool> returned{false};    // reader: "load() has returned"
  unsigned long value = 0;              // plain: published by join()
  std::thread reader([&] {
    started.store(true, std::memory_order_release);
    value = s.load();
    returned.store(true, std::memory_order_release);
  });
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(returned.load(std::memory_order_acquire));
  SeqLockTestPeer::end_write(s, odd_seq, pack(1));
  reader.join();
  EXPECT_EQ(value, pack(1));
  EXPECT_EQ(SeqLockTestPeer::seq(s), 2u);
} // ReaderWaitsForWriteInFlight

// N writers increment through update(). The final value is exact only if
// update() is a read-modify-write under writer exclusion. `shadow` is a plain
// counter touched from inside the callable: guarded by nothing but SeqLock's
// writer lock, so broken exclusion is a TSan report as well as a wrong count.
// The counter must have advanced by exactly 2 per write.
TEST(SeqLockTest, MultiWriterUpdate) {
  constexpr int num_threads = 4;
  constexpr int num_iters = 20000;
  SeqLock<unsigned long> s;
  long shadow = 0;                      // deliberately not atomic
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < num_iters; ++i) {
        s.update([&](unsigned long v) { ++shadow; return v + 1; });
      }
    });
  } // spawn loop
  for (std::thread& t : threads) t.join();
  EXPECT_EQ(s.load(), static_cast<unsigned long>(num_threads)*num_iters);
  EXPECT_EQ(shadow, long(num_threads)*num_iters);
  EXPECT_EQ(SeqLockTestPeer::seq(s), std::uint64_t(2)*num_threads*num_iters);
} // MultiWriterUpdate

// N writers call store(). Nothing a store() leaves in the payload can show
// whether writers excluded each other -- any of the stored values is a
// legitimate final value -- but the counter can: its arithmetic is a plain
// load and two plain stores, correct only under the lock, so two writers
// inside write() together lose a count (or leave the counter odd, and the
// final load() spinning).
TEST(SeqLockTest, MultiWriterStore) {
  constexpr int num_threads = 4;
  constexpr unsigned long num_iters = 20000;
  SeqLock<unsigned long> s;
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&] {
      for (unsigned long i = 1; i <= num_iters; ++i) s.store(i);
    });
  } // spawn loop
  for (std::thread& t : threads) t.join();
  // ASSERT, not EXPECT: a wrong counter may well be an odd one, on which the
  // load() below would spin forever.
  ASSERT_EQ(SeqLockTestPeer::seq(s), std::uint64_t(2)*num_threads*num_iters);
  EXPECT_EQ(s.load(), num_iters);       // every thread's last store
} // MultiWriterStore

// Writers advance the generation through update(), recording each generation
// in a plain ledger from inside the callable (so: under writer exclusion,
// before the payload store). The payload is its own generation number: the
// k-th write stores pack(k). Readers check every value they load:
//   - well-formed, and never going backwards (per reader);
//   - ledger[g] holds generation g's entry. The ledger is NOT atomic: the
//     lookup is race-free only because load() makes the write that produced g
//     happen-before its return. A load() that synchronizes with nothing is a
//     TSan report here;
//   - the counter value the read validated is exactly 2g: the header's
//     invariant, word for word. Readers load through the peer to learn that
//     counter value; it is the same code as load().
// Finally the generation count and the counter must be exact (no lost update).
//
// Readers and writers are made to interleave, not just hoped to: a load that
// returns a generation strictly between the initial and the final one
// completed while the writers were at work. Each reader counts those, and
// announces its first; each writer stops halfway through its writes until
// every reader has announced. So however the threads are scheduled, every
// reader is known to be polling before the second half of the writes, and the
// closing per-reader check cannot fail short of a broken handshake. (That is
// evidence that reads and writes interleaved, not that any one read
// overlapped any one write's odd window: a reader cannot observe its own
// retries, and on a single core nothing can force an overlap.) The handshake
// adds reader -> writer edges only; the ledger lookup needs a writer -> reader
// edge, and still gets one from nowhere but load().
TEST(SeqLockTest, ReadersAndWriters) {
  constexpr int num_writers = 2;
  constexpr int num_readers = 3;
  constexpr unsigned long num_iters = 20000;
  constexpr unsigned long final_generation = num_writers*num_iters;
  SeqLock<unsigned long> s(pack(0));
  std::vector<unsigned long> ledger(final_generation + 1, 0);
  ledger[0] = ledger_entry(0);
  std::atomic<int> mid_run{0};          // readers that have loaded a mid-run generation
  std::atomic<bool> done{false};        // all writers have been joined
  std::vector<long> errors(num_readers, 0);  // one slot per reader, read after join
  std::vector<long> during(num_readers, 0);  // loads that completed mid-run, ditto
  std::vector<std::thread> writers, readers;
  for (int r = 0; r < num_readers; ++r) {
    readers.emplace_back([&, r] {
      unsigned long last = 0;
      // The flag is loaded BEFORE the pass it controls, so every reader makes
      // at least one pass after the last write.
      for (bool last_pass = false; !last_pass;) {
        last_pass = done.load(std::memory_order_acquire);
        std::uint64_t seq = 0;
        const unsigned long v = SeqLockTestPeer::load_validated(s, seq);
        const unsigned long g = generation(v);
        if (!well_formed(v) || g < last || g >= ledger.size() ||
            ledger[g] != ledger_entry(g) || seq != 2*g) {
          ++errors[r];
        }
        if (g > 0 && g < final_generation && ++during[r] == 1) {
          mid_run.fetch_add(1, std::memory_order_acq_rel);
        }
        last = g;
      } // read until the writers are done
    }); // reader thread
  } // reader spawn loop
  for (int w = 0; w < num_writers; ++w) {
    writers.emplace_back([&] {
      for (unsigned long i = 0; i < num_iters; ++i) {
        // Halfway: the generation is mid-run and stays so until this writer
        // moves on, so every reader that runs at all gets to announce.
        while (i == num_iters/2 && mid_run.load(std::memory_order_acquire) < num_readers) {
          std::this_thread::yield();
        }
        s.update([&](unsigned long v) {
          const unsigned long g = generation(v) + 1;
          ledger[g] = ledger_entry(g);
          return pack(g);
        });
      } // num_iters updates
    }); // writer thread
  } // writer spawn loop
  for (std::thread& t : writers) t.join();
  done.store(true, std::memory_order_release);
  for (std::thread& t : readers) t.join();
  for (int r = 0; r < num_readers; ++r) {
    EXPECT_EQ(errors[r], 0) << "reader " << r;
    EXPECT_GT(during[r], 0) << "reader " << r << " never ran while the writers did";
  }
  EXPECT_EQ(s.load(), pack(final_generation));
  EXPECT_EQ(SeqLockTestPeer::seq(s), 2*final_generation);
} // ReadersAndWriters

// The reader protocol against a write that is NOT a single store: the peer
// stores an ill-formed value first and the real one second, all while the
// counter is odd -- a two-word write in miniature. A reader that returns the
// ill-formed value read the payload during a write and failed to notice: with
// the odd check gone it notices nothing when it starts during a write, with
// the validation gone it notices nothing when a write starts during the read.
// Readers go through the public load() here. Same handshake as in
// ReadersAndWriters: the writer stops halfway until every reader has loaded a
// mid-run generation.
TEST(SeqLockTest, ReadersNeverSeeWriteInFlight) {
  constexpr int num_readers = 3;
  constexpr unsigned long num_iters = 50000;
  SeqLock<unsigned long> s(pack(0));
  std::atomic<int> mid_run{0};          // readers that have loaded a mid-run generation
  std::atomic<bool> done{false};        // the last write is complete
  std::vector<long> errors(num_readers, 0);  // one slot per reader, read after join
  std::vector<long> during(num_readers, 0);  // loads that completed mid-run, ditto
  std::vector<std::thread> readers;
  for (int r = 0; r < num_readers; ++r) {
    readers.emplace_back([&, r] {
      unsigned long last = 0;
      for (bool last_pass = false; !last_pass;) {
        last_pass = done.load(std::memory_order_acquire);
        const unsigned long v = s.load();
        const unsigned long g = generation(v);
        if (!well_formed(v) || g < last) ++errors[r];
        if (g > 0 && g < num_iters && ++during[r] == 1) {
          mid_run.fetch_add(1, std::memory_order_acq_rel);
        }
        last = g;
      } // read until the writer is done
    }); // reader thread
  } // reader spawn loop
  for (unsigned long g = 1; g <= num_iters; ++g) {
    // Halfway: generation g-1 is in place, mid-run, and stays until every
    // reader has announced a mid-run load.
    while (g == num_iters/2 && mid_run.load(std::memory_order_acquire) < num_readers) {
      std::this_thread::yield();
    }
    const std::uint64_t odd_seq = SeqLockTestPeer::begin_write(s, in_flight);
    SeqLockTestPeer::end_write(s, odd_seq, pack(g));
  } // num_iters two-step writes
  done.store(true, std::memory_order_release);
  for (std::thread& t : readers) t.join();
  for (int r = 0; r < num_readers; ++r) {
    EXPECT_EQ(errors[r], 0) << "reader " << r;
    EXPECT_GT(during[r], 0) << "reader " << r << " never ran while the writer did";
  }
  EXPECT_EQ(s.load(), pack(num_iters));
  EXPECT_EQ(SeqLockTestPeer::seq(s), 2*num_iters);
} // ReadersNeverSeeWriteInFlight
