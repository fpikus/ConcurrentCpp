// Benchmark of the sequence lock (seqlock.h) against the shipped SpinLock
// (spinlock.h) for read/write access to one shared word, across the same
// read:write mixes and the same contention dial as the reader/writer ladder
// sweep in spinlock_rw_tune_bm.C. For the spinlock against std::mutex and the
// CAS retry loop on a pure write workload see spinlock_bm.C; for the back-off
// ladders themselves see spinlock_tune_bm.C.
//
// THE PREMISE
//
// A spinlock makes readers exclude readers. Two threads that only want to copy
// the shared word out cannot do it at the same time, and -- worse for the
// hardware -- each of them has to WRITE the lock word to be allowed to read the
// data, so every reader takes the lock's cache line Exclusive and rips it away
// from every other core that was using it. The reader traffic on a spinlock is
// therefore not read traffic at all; it is a stream of ownership transfers on
// one line, and it costs the same coherence round trip whether the caller
// intends to modify anything or not.
//
// A sequence lock is the standard answer to that. Readers write NOTHING: a
// reader loads the generation counter, loads the payload, loads the counter
// again, and accepts the value if the two counter reads match and are even.
// The counter's line stays Shared among all readers, so N readers can be
// serviced out of N private L1 copies with no coherence traffic at all. The
// price is that the protocol is optimistic: a read that a write overlaps is
// thrown away and retried, writers are still serialized against each other by
// an embedded spinlock, and a descheduled writer stalls every reader.
//
// THE HYPOTHESIS
//
// Reader scalability is the whole claim, so it should show up as a difference
// in SHAPE, not merely in level: as the thread count rises on the pure-read
// endpoint, the spinlock's throughput should be flat or falling (one line, one
// owner at a time, more cores fighting for it) while the seqlock's should rise
// roughly with the number of threads, since nothing is being transferred.
// Going the other way, on the pure-write endpoint the seqlock must LOSE: a
// write is a spinlock acquire plus two extra stores to the generation counter,
// so it is strictly the spinlock's work plus a constant. The constant is
// smaller than it first looks, though, and it is worth being precise about why.
// SeqLock puts the counter and the writer lock on ONE cache line and the
// payload at the head of the next (seqlock.h static_asserts the layout), so a
// seqlock writer dirties two lines -- {seq_, lock_} and {payload_} -- which is
// exactly as many as the spinlock comparator dirties, {lock_} and {x_}. The
// counter costs no extra ownership transfer; it costs two more stores to a line
// the writer has already pulled Exclusive to take the lock. The mixes
// interpolate, and the interesting question is where the crossover sits -- at
// what read:write ratio the readers' saved coherence traffic pays for the
// writers' extra stores.
//
// THE CONTROL
//
// The payload here is a single unsigned long, which means a third mechanism is
// available that neither lock needs: leave the word in a std::atomic and use
// fetch_add and load with the weakest orderings the workload admits (relaxed
// on both -- nothing else is published through this word, so no reader needs
// an edge to anything). On x86 that is one LOCK XADD per write and one plain
// MOV per read: no mutual exclusion, no retry, no protocol. It is the honest
// lower bound for this payload, and it is in the binary precisely so that the
// seqlock's numbers are not read as a win against the wrong baseline. Any
// seqlock result should be judged by how much of the gap between the spinlock
// and the bare atomic it closes -- and by remembering that the bare atomic is
// not available at all once the payload is bigger than a word or the update is
// not a single commutative RMW.
//
// THE KNOWN LIMITATION
//
// With a one-word payload a torn read is impossible: the payload lives in a
// std::atomic<unsigned long>, and a reader's load of it is indivisible whether
// or not the counter protocol is there. So this benchmark does NOT measure the
// thing a sequence lock is normally bought for -- consistent snapshots of
// multi-word state. It measures the OVERHEAD of the protocol (two extra stores
// per write, three loads and a validation branch per read, plus retries) and
// the reader scalability that comes with writing nothing, on the one payload
// size where an apples-to-apples comparison against both a spinlock and a bare
// atomic is possible at all. The multi-word advantage is real and is not
// visible here; it would need a payload the atomic control cannot hold, at
// which point the control disappears and the comparison changes character.
//
// RESULTS IN BRIEF
//
// Indicative numbers from one machine (WSL2 on a Ryzen 7940HS, 16 hardware
// threads); the shapes are the point, not the digits. The reader-scalability
// claim holds, and holds big: on the pure-read endpoint at 16 threads the
// seqlock runs at ~23x the spinlock's throughput and ~0.75x the bare atomic's,
// so it closes most of the distance to the lower bound. Read-mostly mixes keep
// a good deal of that -- 100:1 is ~3x the spinlock at saturation and ~2.2x at
// work:30 -- and by 1:1 the advantage is down to ~15%. Write-heavy is where the
// protocol gets paid for: within a few percent of the spinlock at saturation up
// to 8 threads, but ~20-25% slower at work:30 with 8 to 16 threads.
//
// The author's conclusion is less flattering to the mechanism than those
// multiples make it sound. For a payload of one word the atomic wins outright,
// and a read-heavy workload over a value that rarely changes is usually
// double-checked-locking territory, which runs at atomic speed even behind a
// spinlock when it is coded right. What is genuinely left for the sequence lock
// is payloads an atomic cannot hold -- and that is precisely what this
// benchmark does not measure; see THE KNOWN LIMITATION above.
//
// THE HARNESS
//
// The benchmark body (BM_rw()), the (reads, writes, work) argument grid
// (RW_ARGS), and the plain-SpinLock comparator (SpinLockData) come from
// spinlock_bm_common.h, shared with the reader/writer ladder sweep in
// spinlock_rw_tune_bm.C: same per-iteration shape, same work placement, same
// coupling of the work to the shared value, same masking of the read seed, same
// items/s accounting, so the numbers here are directly comparable with the
// existing read/write results by construction rather than by inspection. What
// this file adds is the three mechanisms plugged into that harness's
// write()/read() interface; the ladder-tuning machinery (BackOffParams,
// LockAdapter, the named ladders) stays in spinlock_rw_tune_bm.C.
//
// RUNNING IT
//
// Names carry the mechanism and the mix, so filters slice it (the filter is a
// regex):
//
//   ./seqlock_bm --benchmark_filter='writes:0/'        # 100% readers
//   ./seqlock_bm --benchmark_filter='reads:0/'         # 100% writers
//   ./seqlock_bm --benchmark_filter='work:0/'          # saturation only
//   ./seqlock_bm --benchmark_filter='BM_seqlock/'      # one mechanism, sin/cos
#include <atomic>

#include "spinlock.h"
#include "seqlock.h"

#include "spinlock_bm_common.h"

// The mechanism under test, behind the harness's write()/read() interface: one
// SeqLock<unsigned long>, write() through update() and read() through load().
//
// write() goes through update() rather than load()-then-store() because the
// increment must not be lost: update() runs the callable under the writer lock,
// so the read-modify-write is atomic with respect to every other writer, which
// is exactly what the spinlock comparator's `x_ += n` gets from holding the
// lock across both halves. A load()/store() pair would be a different (and
// wrong) workload -- two writer-lock acquisitions and a lost-update race.
//
// No alignas here: SeqLock pins its own layout (counter and writer lock on one
// 64-byte line, payload at the head of the next), which is the same separation
// the spinlock comparator SpinLockData (spinlock_bm_common.h) spells out by
// hand with two alignas(64) members -- and the reason both mechanisms dirty the
// same number of lines per write; see THE HYPOTHESIS above.
class SeqLockData {
  public:
  // Add `n` to the shared total under the writer lock. The callable takes the
  // current value by value and returns the new one; it is called before the
  // generation counter goes odd, so concurrent readers keep succeeding on the
  // old value while it runs.
  void write(unsigned long n) {
    seqlock_.update([n](unsigned long x) { return x + n; });
  } // SeqLockData::write()

  // Copy the value out through the reader protocol: counter, payload, counter,
  // retry on disagreement. Writes nothing to shared memory. Returns the value
  // because the caller feeds it into its next work chunk.
  unsigned long read() {
    return seqlock_.load();
  } // SeqLockData::read()

  private:
  SeqLock<unsigned long> seqlock_;      // counter + writer lock + payload
}; // class SeqLockData

// The lower bound: no lock and no protocol, just the word itself in a
// std::atomic, with the weakest orderings that are sufficient here.
//
// Relaxed on both sides is not a shortcut: nothing else is published through
// this word -- the readers use the value only as a seed for their own local
// arithmetic, and no other shared state has to become visible along with it --
// so neither operation needs to carry a happens-before edge, and the counter
// is correct under relaxed because fetch_add is a single indivisible RMW on
// one location. On x86 the write is one LOCK XADD and the read one plain MOV.
// This is what the two locks are spending their instructions to emulate for a
// payload that does not actually need them; see THE CONTROL above.
class AtomicData {
  public:
  // Add `n` to the shared total with one atomic read-modify-write: wait-free,
  // no waiting and no retry, but it exists only because the update is a
  // commutative operation on a single word.
  void write(unsigned long n) {
    x_.fetch_add(n, std::memory_order_relaxed);
  } // AtomicData::write()

  // Load the shared total. Readers never write, so -- like the seqlock's
  // readers -- they keep the line Shared and do not contend with each other.
  unsigned long read() {
    return x_.load(std::memory_order_relaxed);
  } // AtomicData::read()

  private:
  // The shared total, and the whole of this mechanism's state: no lock word and
  // no generation counter to go with it. alignas(64) gives it a line to itself,
  // as the other two mechanisms give their payloads, so the comparison is
  // between protocols and not between accidental neighbours on a line.
  alignas(64) std::atomic<unsigned long> x_ {0};
}; // class AtomicData

// Register one mechanism under both kinds of local work, named by the
// mechanism and the work kind:
//   BM_seqlock/reads:100/writes:1/work:0/...       (sin/cos work)
//   BM_seqlock_mem/reads:100/writes:1/work:0/...   (memory-streaming work)
// Both kinds are registered because they load different machine resources: a
// thread doing sin/cos work costs the other threads nothing, while a thread
// streaming memory competes with all of them for L3 and DRAM bandwidth -- and
// the seqlock's claim is precisely about how much of the memory system the
// readers leave alone.
#define RW_BM(name, data) \
  BENCHMARK_TEMPLATE(BM_rw, data, SinCosWork)->Name("BM_" #name) RW_ARGS; \
  BENCHMARK_TEMPLATE(BM_rw, data, MemWork)->Name("BM_" #name "_mem") RW_ARGS

RW_BM(seqlock, SeqLockData);            // the mechanism under test
RW_BM(spinlock, SpinLockData);          // the comparator it must beat on reads
RW_BM(atomic, AtomicData);              // the lower bound for a one-word payload

BENCHMARK_MAIN();
