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
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: steady churn, without and with
 * reclamation. One of the benchmark's translation units (the overview of the
 * workloads, the index of the units and why there are several:
 * concurrent_hash_set_rcu_bm.C); the code generated for these rows depends
 * on this unit alone. The per-call trace, the window fixture and the window
 * body are here; the handles, the tallies and the snapshots they use are the
 * shared header's.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  Churn_Window_Del_NoReclaim     steady churn: every thread slides a window
 *                                 of live keys (erase the oldest, insert the
 *                                 next, two lookups), nothing reclaimed, so
 *                                 every insert appends and the table grows.
 *  Churn_Window_Del_Rcu           the same under the reclamation regime, one
 *                                 advancer with the walk: the first of the
 *                                 three rows in which the free pass pushes
 *                                 freed nodes while other threads pop them.
 *  Churn_Window_Del_RcuEach       the same with every thread an advancer
 *                                 after every kWindowAdvanceEvery of its own
 *                                 erases (staggered as above): the explicit
 *                                 like-for-like pair of the next row, as the
 *                                 small erase pair (Erase_Small_Del_RcuEach
 *                                 and _Auto, concurrent_hash_set_rcu_bm_erase.C).
 *  Churn_Window_Del_Auto          steady churn with the automatic cadence at
 *                                 kWindowAdvanceEvery nodes retired per arena
 *                                 shard: whether a bump-and-free trigger on
 *                                 every thread reaches a steady state that
 *                                 one advancer with a walk did not (the table
 *                                 size, the slots per live key, the attempts
 *                                 proceeded against skipped, the free pass's
 *                                 share of the threads' time), and the
 *                                 per-call series of the three churn
 *                                 reclamation rows (the "# trace" lines: the
 *                                 time of every call or trigger return on
 *                                 every thread, so the concurrent phase and
 *                                 the tail are read from the data instead of
 *                                 bracketed from finish_spread).
 */

// ---------------------------------------------------------------------------
// One thread's per-call time series in a churn reclamation row: the "# trace"
// lines.
//
// Why: the rates and the summed timers say how much a row's reclamation cost
// and freed, not WHEN: with one advancer the churn row has a concurrent
// phase and a tail in which the advancer walks alone, and the phases could
// only be bracketed from finish_spread; with every thread a caller or a
// trigger the question is whether the free passes run back to back. So each
// thread of the three churn reclamation rows keeps a local array of up to
// kEntries entries, one per return of its own advance() calls (every
// outcome, `contended` included: an attempt is an attempt) or of the
// triggers it detected, and thread 0 of the _Auto row also one per
// kWindowAdvanceEvery of its own erases (a sample, since it makes no calls):
// the steady_clock time of the return (ns, relative to the thread's own
// stamp taken before the loop), the kind (a letter: the outcome of the call,
// `t` a detected trigger, `s` a sample), the nodes freed so far (the thread's
// own calls' on the explicit rows; the set-wide count on the _Auto row,
// whose triggers belong to no thread's tally) and the table size. The array
// is cold: on the stack, uninitialized, touched only when an entry is
// written, once per cadence. Entries past kEntries are dropped and COUNTED
// (thread 0 of the churn _Auto row makes 32 samples per repetition beside
// its own triggers, which can exceed the array); the attempts by kind are
// counted whether or not the entry was kept, so that a dropped trigger still
// counts in the identity "detected triggers == the set's auto_reclaims".
// After the loop, untimed, every thread prints its line, prepared in one
// buffer and written with one fputs() (lines from T threads interleave in
// the console, never within a line):
//   # trace <row> threads=<T> thread=<i> start_ns=<absolute steady_clock ns
//   of the thread's stamp before the loop> bump_ns=<..> free_ns=<..>
//   walk_ns=<..> n=<entries kept> dropped=<d>
//   kinds=a:<..>,n:<..>,c:<..>,r:<..>,t:<..>,s:<..> | <t>:<kind>:<freed>:<table> ...
// (the kinds are the attempts by kind, kept and dropped; `t` of an entry is
// relative to start_ns, so start_ns + t aligns the threads of one
// repetition). start_ns is the entries' reference only: it is taken before
// the loop's start barrier, so it includes the thread's spawn skew and is
// NOT the start of the thread's loop. The thread's loop is bracketed by the
// wall-clock stamps the loop itself takes (gb_wall_clock.h: the end of the
// thread's first and of its last iteration), which thread 0 prints for
// every thread of the run, after the loop (report_stamps(); thread 0 may
// read every thread's record then, and no other thread may read its own
// after the loop, since thread 0 releases the records at the end of its
// body):
//   # stamps <row> threads=<T> first_ns=<f0>,<f1>,... last_ns=<l0>,<l1>,...
// (absolute steady_clock ns, in thread order): the per-thread phases --
// first finish, last finish, the free passes' progress against each
// thread's own loop -- are read from these two lines together. The lines
// precede the row's console line, as "# prep" lines do, so a reducer pairs
// them with the row by order (and splits them at " threads=": the _Auto
// rows' names contain a space). The record costs one steady_clock read per
// entry, once per cadence, plus two loads on the explicit rows and the
// set-wide counter sum on the _Auto row (see there): negligible, and absent
// from the controls.
// Trivially destructible, like the other body locals.
// ---------------------------------------------------------------------------
class Trace {
public:
    static constexpr int kEntries = 64;
    // What an entry records; the letter printed is the enumerator's.
    enum class Kind : uint8_t { advanced, nothing_retired, contended, ring_full, trigger, sample, kinds };
    static constexpr int kKinds = int(Kind::kinds);
    static constexpr char kLetters[kKinds + 1] = "ancrts";

    // Stamps the thread's start (before the loop; the entries' times are
    // relative to it).
    Trace() : start_(now_ns()) {}

    // The kind of an advance() return.
    static Kind kind_of(AdvanceResult::Advance a) {
        switch (a) {
        case AdvanceResult::Advance::advanced:        return Kind::advanced;
        case AdvanceResult::Advance::nothing_retired: return Kind::nothing_retired;
        case AdvanceResult::Advance::contended:       return Kind::contended;
        case AdvanceResult::Advance::ring_full:       return Kind::ring_full;
        } // switch on the outcome
        return Kind::ring_full;   // not reached: every enumerator is listed
    } // Trace::kind_of()

    // Inside the loop: one entry, at the return it records (`freed` and
    // `table` as described above; both fit 32 bits in every row that keeps
    // a trace, and are stored so to keep an entry at 24 bytes).
    void record(Kind kind, size_t freed, size_t table) {
        const int64_t t = now_ns() - start_;
        ++attempts_[int(kind)];
        if (n_ == kEntries) {
            ++dropped_;
            return;
        } // the array is full: counted, not kept
        entries_[n_] = {t, uint32_t(freed), uint32_t(table), kLetters[int(kind)]};
        ++n_;
    } // Trace::record()

    // After the loop, on every thread, untimed: this thread's "# trace" line
    // (above), with this thread's timers `t` (the tally's), written with one
    // fputs().
    template <typename Timers>
    void report(const benchmark::State& state, const Timers& t) const {
        // The buffer: a header of under 500 characters (the row's name is
        // up to about 120 of them) and at most kEntries entries of at most
        // 37 ("<t>:<k>:<freed>:<table> " with a 12-digit t and two 10-digit
        // counts); an overflow is cut, never overrun, and the cut line still
        // ends in a newline.
        char buf[4096];
        size_t len = 0;
        const auto put = [&](const char* fmt, auto... args) {
            if (len >= sizeof buf) return;
            const int n = snprintf(buf + len, sizeof buf - len, fmt, args...);
            len = n < 0 ? sizeof buf : std::min(len + size_t(n), sizeof buf);
        }; // put(): appends one formatted piece
        put("# trace %s threads=%d thread=%d start_ns=%lld bump_ns=%zu free_ns=%zu walk_ns=%zu n=%d dropped=%u kinds=",
            state.name().c_str(), state.threads(), state.thread_index(), (long long)start_,
            t.bump_ns, t.free_ns, t.walk_ns, n_, dropped_);
        for (int k = 0; k < kKinds; ++k) put("%s%c:%u", k == 0 ? "" : ",", kLetters[k], attempts_[k]);
        put("%s", " |");   /* "%s": a format with no argument trips -Wformat-security */
        for (int i = 0; i < n_; ++i) put(" %lld:%c:%u:%u", (long long)entries_[i].t, entries_[i].kind, entries_[i].freed, entries_[i].table);
        put("%s", "\n");
        if (len >= sizeof buf) buf[sizeof buf - 2] = '\n';   // cut: the line still ends in a newline
        buf[sizeof buf - 1] = '\0';
        fputs(buf, stdout);
        fflush(stdout);
    } // Trace::report()

    // After the loop, on thread 0 only, untimed, before the records are
    // released: the "# stamps" line (above) from every thread's wall-clock
    // record of this run.
    static void report_stamps(const benchmark::State& state, const WallRecords& records) {
        std::string first, last;
        for (int i = 0; i < state.threads(); ++i) {
            const WallStamps& s = records.of(i);
            first += (i == 0 ? "" : ",") + std::to_string(ns_of(s.first));
            last += (i == 0 ? "" : ",") + std::to_string(ns_of(s.last));
        } // loop over the threads' records
        printf("# stamps %s threads=%d first_ns=%s last_ns=%s\n", state.name().c_str(), state.threads(), first.c_str(), last.c_str());
        fflush(stdout);
    } // Trace::report_stamps()

private:
    // A steady_clock time point in ns since its epoch.
    static int64_t ns_of(std::chrono::steady_clock::time_point tp) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    } // Trace::ns_of()
    // steady_clock now, in ns since its epoch.
    static int64_t now_ns() { return ns_of(std::chrono::steady_clock::now()); }

    struct Entry {
        int64_t t;        // ns after start_
        uint32_t freed;   // nodes freed so far (see above)
        uint32_t table;   // the table size
        char kind;        // the kind's letter
    }; // struct Trace::Entry
    static_assert(sizeof(Entry) == 24, "an entry is meant to be 24 bytes");

    const int64_t start_;             // the thread's stamp before the loop (before the start barrier: the entries' reference, not the loop's start)
    int n_ = 0;                       // entries kept
    uint32_t dropped_ = 0;            // entries dropped (the array was full)
    uint32_t attempts_[kKinds] = {};  // entries by kind, kept and dropped alike
    Entry entries_[kEntries];         // the kept entries; uninitialized on purpose (cold)
}; // class Trace
static_assert(std::is_trivially_destructible_v<Trace>, "a body local with a destructor would be live across the state loop");

// ---------------------------------------------------------------------------
// Churn_Window_Del: steady churn, without and with the reclamation regime.
//
// Why measure: the other units' rows make freed nodes available once, before the
// timed region (the Reclaimed states), or call advance() in rows that do not
// allocate (Erase_Presized_Del_Rcu, once per thread at its end;
// Erase_Small_Del_Rcu, at a cadence); none frees nodes while other threads
// allocate.
// Here every thread keeps a fixed number of live keys and replaces them
// continuously, so under the regime the free pass pushes the nodes it frees
// onto the free lists while the threads pop them for their inserts -- the free
// lists as a concurrent recycler, which is what reclamation beside the
// operations is for -- and without it every insert appends, and the arena and
// the table grow with the run. The pair says what the regime costs a churning
// workload and what it saves it.
// Shape: thread t owns the keys mix(t*kKeyStride + i), i = 0, 1, 2, ...; at
// any moment the kWindow keys with i in [lo, lo + kWindow) are live. Thread
// 0's SetUp() inserts every thread's first window (lo = 0), untimed, into a
// table of kWindowBuckets buckets (one key per bucket at 16 threads). Every
// thread then repeats a cycle of four iterations: erase the key at lo; insert
// the key at lo + kWindow (the window slides by one); look up two keys drawn
// at random from [lo - kWindow, lo + kWindow), of which half are live (those
// below lo were erased, or never inserted). kWindowIters iterations per
// thread: kWindowIters/(4*kWindow) passes over the window. Every erase and
// every insert must return true (checked per thread, after the loop).
// _NoReclaim: nothing else. Nothing is freed, so the retired lists and the
//   arena grow by one node per insert, and the table doubles as the arena
//   grows (the resize hint counts every slot that is not on a free list).
// _Rcu: the reclamation regime. Every thread refreshes its handle at the top
//   of every iteration, and thread 0 calls advance() after every
//   kWindowAdvanceEvery of its own erases (AdvanceTally says why one
//   advancer): the same number of calls per run at every thread count, each
//   walking the threads*kWindow live keys. The nodes a call frees are popped
//   by every thread's inserts: from the inserting thread's own shard's list
//   first, then from the others' (the steal).
// _RcuEach: the same with EVERY thread calling advance() after every
//   kWindowAdvanceEvery of its own erases, the first calls staggered
//   (AdvanceTally): the attempt frequency of _Auto, called explicitly and
//   walking. Most calls at T >= 8 are expected `contended`.
// _Auto: the set constructed with the automatic cadence kWindowAdvanceEvery
//   (nodes retired per arena shard: WindowAutoCadence), no call: the set runs
//   the bump and the free pass from inside the erase whose retire push
//   crossed the cadence, on that thread, never walking. Every thread still
//   refreshes at the top of every iteration and releases on its last. A
//   thread detects its own triggers by reading its own bump_ns (a
//   thread_local the set charges a proceeding trigger's bump to) just before
//   the erase and comparing after it: one thread-local load before the erase,
//   one load, a compare and a predictable not-taken branch after it, the
//   row's stated added per-erase cost beside the trigger's own (reading the
//   set's auto_reclaims per erase would be a shared-line load on every
//   thread, absent from the control), plus the sampling countdown of the
//   "# trace" series (AdvanceTally::due(): a decrement, a compare and a
//   predictable branch per erase on EVERY thread, as in the _Rcu row, though
//   only thread 0's ever reaches zero; the control carries neither). A
//   skipped trigger reads no clock and
//   is not detected; a trigger inside an insert's or a lookup's split cleanup
//   (a table that doubled) is outside the bracket: the set's auto_reclaims
//   counts all of them, detected_triggers the ones inside erases.
// Counters, read by thread 0 after the loop (every thread has left it), untimed:
//   table_size         -- the bucket count at the end;
//   slots_per_live_key -- arena slots ever appended, over the live keys at the
//                         end (threads*kWindow): the memory the run kept;
//   pops_raced_push, free_push_cas_failures
//                      -- the container's diagnostic counters over the timed
//                         loop: free-list pops whose CAS lost to a push of the
//                         free pass (or failed spuriously), and pushes whose
//                         CAS lost to a pop (or failed spuriously): the
//                         witnesses that frees and pops met;
//   retired_nodes      -- nodes retired over the timed loop (every erase of a
//                         present key retires at least its node, by the
//                         eraser or a later walker, and split copies add to
//                         it: at least threads*kWindowIters/4): the
//                         denominator of "freed against retired";
// on the reclamation rows the AdvanceTally and the ReclaimSnapshot counters,
// and their threads' "# trace" lines (Trace); on _Auto the "# shards" line.
// ---------------------------------------------------------------------------
static constexpr int    kWindow        = 1 << 12;
static constexpr size_t kWindowBuckets = size_t(1) << 16;
static constexpr int    kWindowIters   = 1 << 20;
// Churn_Window_Del_Rcu's cadence: thread 0 calls advance() after every
// kWindowAdvanceEvery of its own erases, two passes over its window
// (HASH_BM_WINDOW_ADVANCE_EVERY overrides it: advance_every_from_env()).
// _RcuEach: every thread's cadence of its own erases; _Auto: the set's
// cadence in nodes retired per arena shard (one thread per shard at the
// threads' equal rates: the same per-thread frequency), and thread 0's
// sampling cadence for the "# trace" series.
static constexpr int    kWindowAdvanceEvery = 2*kWindow;
// The automatic cadence of Churn_Window_Del_Auto (see NoCadence): the twin's
// cadence, with the same environment override.
struct WindowAutoCadence {
    static size_t auto_reclaim_every(const benchmark::State& /*state*/) {
        return size_t(advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery));
    } // WindowAutoCadence::auto_reclaim_every()
}; // struct WindowAutoCadence

// The window fixture: the automatic cadence policy `Cadence` as in
// EraseFixture (concurrent_hash_set_rcu_bm_erase.C; NoCadence for every row but _Auto; the kept rows' names keep
// the one written template argument).
template <typename SetType, typename Cadence = NoCadence>
class WindowFixture : public benchmark::Fixture {
public:
    static SetType* set;
    using Handles = ThreadHandles<SetType>;
    static Handles handles;   // the threads' handles (ThreadHandles, shared header)
    using Tally = AdvanceTally<SetType>;   // one thread's advance() calls or triggers in the reclamation bodies (AdvanceTally, shared header)
    static ReclaimSnapshot<SetType> before;   // the container's counters after the prefill (ReclaimSnapshot, shared header)
    void SetUp(const ::benchmark::State& state) override {
        if (state.thread_index() != 0) return;
        const int threads = state.threads();
        set = make_prefilled<SetType>(state, "WindowFixture", kWindowBuckets, Cadence::auto_reclaim_every(state), [threads](auto insert) {
            for (int t = 0; t < threads; ++t) {
                for (int i = 0; i < kWindow; ++i) insert(mix(t*kKeyStride + i));
            } // every thread's first window
        });
        before.take(*set);
        handles.make(*set, state.threads());
    } // WindowFixture::SetUp()
    void TearDown(const ::benchmark::State& state) override {
        if (state.thread_index() == 0) { handles.clear(); delete set; set = nullptr; }
    }
}; // class WindowFixture
template <typename SetType, typename Cadence> SetType* WindowFixture<SetType, Cadence>::set = nullptr;
template <typename SetType, typename Cadence> ThreadHandles<SetType> WindowFixture<SetType, Cadence>::handles;
template <typename SetType, typename Cadence> ReclaimSnapshot<SetType> WindowFixture<SetType, Cadence>::before;

// The body of the window rows (see above), as a macro for the same reason as
// DEFINE_ERASE (concurrent_hash_set_rcu_bm_erase.C): PRE declares what the regime needs before the loop, BEFORE
// runs at the top of every iteration, BEFORE_ERASE immediately before every
// erase (the _Auto row's first read of its bump_ns; empty in every other
// row), AFTER_ERASE after every erase, END at the end of every iteration
// (the release on the last one), POST reports after the loop; all six are
// empty in _NoReclaim. The fixture's template arguments come last, as in
// DEFINE_ERASE. Same accounting contract as the other bodies.
#define DEFINE_WINDOW(NAME, PRE, BEFORE, BEFORE_ERASE, AFTER_ERASE, END, POST, ...) \
    BENCHMARK_TEMPLATE_DEFINE_F(WindowFixture, NAME, __VA_ARGS__)             \
    (benchmark::State& state) {                                               \
        static constinit WallRecords records;                                 \
        XorShift32 rng(0x1234567u + 0x9e3779b9u*state.thread_index());        \
        const int base = state.thread_index()*kKeyStride;                     \
        int lo = 0;                                                           \
        int phase = 0;                                                        \
        int64_t ok = 0, tries = 0;                                            \
        bool hit = false;                                                     \
        Handles::Own own(handles, state);                                     \
        PRE                                                                   \
        records.allocate(state);                                              \
        for (auto _ : WallTimed(state, records)) {                            \
            BEFORE                                                            \
            if (phase == 0) {                                                 \
                ++tries;                                                      \
                BEFORE_ERASE                                                  \
                ok += set->erase(own.get(), mix(base + lo));                  \
                AFTER_ERASE                                                   \
            } else if (phase == 1) {                                          \
                ++tries;                                                      \
                ok += set->insert(own.get(), mix(base + lo + kWindow));       \
                ++lo;                                                         \
            } else {                                                          \
                hit = set->contains(own.get(),                                \
                    mix(base + lo - kWindow + int(rng.next() & (2*kWindow - 1)))); \
                benchmark::DoNotOptimize(hit);                                \
            }                                                                 \
            phase = (phase + 1) & 3;                                          \
            END                                                               \
        }                                                                     \
        if (ok != tries) {                                                    \
            state.SkipWithError("erase() or insert() of a window key returned false: the keys are this thread's alone"); \
        }                                                                     \
        state.SetItemsProcessed(state.iterations());                          \
        report_wall(state, records, double(state.iterations())*state.threads()); \
        if (state.thread_index() == 0) {   /* the container's diagnostic counters (see the fixture) */ \
            const ConcurrentSetDel::InternalCounters c = set->get_internal_counters(); \
            state.counters["table_size"] = benchmark::Counter(double(set->get_internal_table_size())); \
            state.counters["slots_per_live_key"] = benchmark::Counter(        \
                double(set->get_internal_node_count())/(double(state.threads())*kWindow)); \
            state.counters["pops_raced_push"] = benchmark::Counter(double(c.pops_raced_push - before.counters.pops_raced_push)); \
            state.counters["free_push_cas_failures"] = benchmark::Counter(double(c.free_push_cas_failures - before.counters.free_push_cas_failures)); \
            state.counters["retired_nodes"] = benchmark::Counter(double(c.retire_nodes - before.counters.retire_nodes)); \
        }                                                                     \
        POST                                                                  \
        records.release(state);                                               \
    }                                                                         \
    BENCHMARK_REGISTER_F(WindowFixture, NAME)                                 \
        ->ThreadRange(1, num_cpu)->Iterations(kWindowIters)->UseRealTime();

// The window rows (see Churn_Window_Del above).
DEFINE_WINDOW(Churn_Window_Del_NoReclaim,
    /* PRE */, /* BEFORE */, /* BEFORE_ERASE */, /* AFTER_ERASE */, /* END */, /* POST */,
    ConcurrentSetDel)
DEFINE_WINDOW(Churn_Window_Del_Rcu,
    /* PRE */
    Tally tally(advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery), state);
    Trace trace;,
    /* BEFORE */
    own.refresh();,
    /* BEFORE_ERASE */,
    /* AFTER_ERASE: advance() at the cadence, its return recorded */
    if (tally.due()) {
        const AdvanceResult r = advance_out_of_line(*set);
        tally.add(r);
        trace.record(Trace::kind_of(r.advance), tally.freed_nodes(), set->get_internal_table_size());
    },
    /* END */
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    trace.report(state, Tally::timers());
    if (state.thread_index() == 0) {
        Trace::report_stamps(state, records);
        before.report_set(state, *set);
    },
    ConcurrentSetDel)
DEFINE_WINDOW(Churn_Window_Del_RcuEach,
    /* PRE */
    Tally tally(advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery), state, Tally::Advancer::every_thread);
    Trace trace;,
    /* BEFORE */
    own.refresh();,
    /* BEFORE_ERASE */,
    /* AFTER_ERASE: advance() at this thread's cadence, its return recorded */
    if (tally.due()) {
        const AdvanceResult r = advance_out_of_line(*set);
        tally.add(r);
        trace.record(Trace::kind_of(r.advance), tally.freed_nodes(), set->get_internal_table_size());
    },
    /* END */
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    trace.report(state, Tally::timers());
    if (state.thread_index() == 0) {
        Trace::report_stamps(state, records);
        before.report_set(state, *set);
    },
    ConcurrentSetDel)
// _Auto: `freed` of a trace entry is the set-wide nodes_freed (the triggers
// belong to no thread's tally), read through get_internal_counters(), the
// one accessor that reaches it: a few hundred relaxed loads per entry, once
// per detected trigger or sample, inside the loop; small beside the free
// pass the trigger just ran, and stated.
DEFINE_WINDOW(Churn_Window_Del_Auto,
    /* PRE: no call; thread 0 samples every kWindowAdvanceEvery of its erases; this thread's timers, read around every erase */
    Tally tally(Tally::Automatic{}, advance_every_from_env("HASH_BM_WINDOW_ADVANCE_EVERY", kWindowAdvanceEvery), state);
    Trace trace;
    Tally::Timers& timers = Tally::timers();,
    /* BEFORE */
    own.refresh();,
    /* BEFORE_ERASE */
    const size_t bump_before = timers.bump_ns;,
    /* AFTER_ERASE: the trigger detection, then thread 0's sample */
    if (BM_UNLIKELY(timers.bump_ns != bump_before)) {
        tally.detected();
        trace.record(Trace::Kind::trigger, set->get_internal_counters().nodes_freed, set->get_internal_table_size());
    }
    if (tally.due()) trace.record(Trace::Kind::sample, set->get_internal_counters().nodes_freed, set->get_internal_table_size());,
    /* END */
    own.finish_iteration();,
    /* POST */
    tally.report(state);
    trace.report(state, Tally::timers());
    if (state.thread_index() == 0) {
        Trace::report_stamps(state, records);
        before.report_set(state, *set);
        before.report_shards(state, *set);
    },
    ConcurrentSetDel, WindowAutoCadence)
