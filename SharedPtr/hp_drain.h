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
#ifndef INCLUDED_HP_DRAIN_H_
#define INCLUDED_HP_DRAIN_H_

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

// Resolved relative to the directory of this header AS SPELLED by the includer:
// SharedPtr/mm_hp/ directly, or LockFreeList/mm_hp/ (a directory symlink to it)
// when a LockFreeList TU includes LockFreeList/hp_drain.h (a symlink to this file).
#include "mm_hp/mm_hp.hpp"

// Deterministic draining of mm_hp's global hazard-pointer domain, for tests and
// benchmarks of code that retires objects through mm_hp (intr_shared_ptr_hp).
// GoogleTest-free on purpose: benchmarks use it too. The gtest registration of
// the early scan is in hp_drain_gtest.h.
//
// Why this exists: with hazard pointers, "the object is gone when its count
// reaches 0" is false by design. A retired object is destroyed at the next scan,
// and mm_hp scans only when the number of pending retirements reaches a threshold
// (max(1000, 2 * number of hazard records)) that a test cannot observe and that
// grows with the number of hazard records ever allocated (records are never
// freed; measured: with 3001 hazard records allocated, threshold 6002, a drain
// needed 6001 fillers). A test that asserts a live count after a release must
// first make the pending retirements run; mm_hp has no public flush, so the
// drain builds one from its public API:
//
//   1. retire a SENTINEL object whose destructor sets a flag local to this call;
//   2. retire empty FILLER objects, one at a time, until the flag is set.
//
// The retired list is a stack that a scan takes whole (an exchange with null).
// The scan that destroys the sentinel therefore took, in the same list, every
// object retired before the sentinel; each of them was either destroyed in that
// scan or found protected by a hazard pointer and pushed back for a later scan.
// The fillers retired after the sentinel are destroyed by the same scan. So when
// the flag is set, nothing retired before the drain is still pending except
// objects protected when that scan collected the hazard pointers and objects
// retired DURING that scan (by the destructors it ran). The number of fillers
// adapts to whatever the threshold is, which is why the drain uses a sentinel
// rather than a fixed number of retirements. Hence:
//
// CONTRACT (all three functions below):
// - Precondition, quiescence: no other thread retires objects or runs a scan for
//   the duration of the call. Then every scan runs on the calling thread, inside
//   one of its own retire() calls, so the scan that destroys the sentinel has
//   completed when the flag is observed. Without quiescence, another thread's
//   scan can destroy the sentinel and set the flag while the older objects
//   behind it in the same list are still being destroyed; the drain then returns
//   early. Worse, if another thread's scan holds the list that contains the
//   sentinel (slow destructors, or that thread is preempted mid-scan) for longer
//   than the drain takes to reach its filler cap (over 2 s, see the pauses in
//   hp_drain_detail), the drain ABORTS THE PROCESS. Quiescence holds after
//   joining every thread that used the retiring code. It is NOT guaranteed in a
//   Google Benchmark fixture's TearDown(): there, other benchmark threads may
//   still be destroying their body's locals (and retiring). A drain in
//   TearDown() is outside this contract: it may return before everything is
//   reclaimed, and in the worst case it may abort the benchmark binary.
// - One drain reclaims one layer of a destructor cascade. Objects retired by
//   destructors during a scan are pushed onto the list after that scan took it,
//   and mm_hp does not start a nested scan on a thread that is already scanning,
//   so a cascade of depth d (each destroyed object retires the next) needs d
//   drains (measured: depths 1/2/3/5 need 1/2/3/5 drains; hp_drain_selftest.C
//   pins this). The same holds for an object that was protected when a scan
//   collected the hazard pointers and whose protection a destructor run by that
//   scan released: it is pushed back and reclaimed one round later.
// - Never call a drain from inside a scan, i.e. from the destructor of an object
//   mm_hp is reclaiming (directly or through anything that destructor calls).
//   No retire() on that thread can start a scan while the scan is running, so the
//   sentinel is never destroyed: the drain reaches its filler cap and aborts.
// - A drain allocates (one sentinel and up to the filler cap of fillers) and may
//   run arbitrary pending destructors on the calling thread. drain_reclamation()
//   and force_early_scan() are noexcept: an allocation failure terminates the
//   program. Unwinding is not an option once the sentinel is retired, for the
//   same reason the cap aborts (the pending sentinel points at the drain's stack
//   frame); terminate is also what mm_hp's own noexcept retire() and scan do on
//   an allocation failure.

namespace hp_drain_detail {

// The one retirable type for both roles. The sentinel carries a pointer to the
// draining call's flag; a filler carries nullptr. Retired through mm_hp's default
// deleter (std::default_delete<DrainObject>).
struct DrainObject : std::hazard_pointer_obj_base<DrainObject> {
    // `flag`: the flag to set on destruction (the sentinel), or nullptr (a filler).
    explicit DrainObject(std::atomic<bool>* flag) noexcept : flag_(flag) {}

    // Release store: pairs with the drain's acquire load of the flag, so a drain
    // that sees the flag set also sees everything this scan did before
    // destroying the sentinel (relevant only when the scan runs on another
    // thread, i.e. outside the quiescence precondition).
    ~DrainObject() {
        if (flag_) flag_->store(true, std::memory_order_release);
    }

    DrainObject(const DrainObject&) = delete;
    DrainObject& operator=(const DrainObject&) = delete;

    // The sentinel's flag, owned by the drain_reclamation() call that retired
    // this object; nullptr for fillers.
    std::atomic<bool>* const flag_;
}; // struct DrainObject

// Retirements a single drain may make before it gives up. A drain normally needs
// at most the scan threshold, max(1000, 2 * hazard records), so the cap is far
// above it; reaching it means the sentinel cannot be reclaimed (called inside a
// scan, half a million hazard records exist, or another thread's scan held the
// sentinel for the whole of the drain's pauses).
inline constexpr long filler_cap = 1'000'000;

// A drain that has retired another `fillers_per_pause` fillers without seeing
// its sentinel destroyed sleeps for `pause_length` before continuing. The first
// pause comes after 1000 fillers, one default threshold's worth, so a quiescent
// drain at the default threshold never pauses (it needs at most 999 fillers)
// and one at a raised threshold pauses once per extra 1000 (e.g. 6 pauses,
// ~12 ms, at threshold 6002). The pauses exist for drains outside the quiescence
// precondition: retiring 1,000,000 fillers takes only 13-30 ms at -O1 (0.14-0.21
// s under TSan; measured on a Ryzen 7940HS under WSL2), so without them the cap
// fires as soon as another thread's scan holds the sentinel's list for that
// long, which preemption alone can do (demonstrated). With 999 pauses the cap is
// at least 2 s away (measured: 2.4 s at -O1, 2.6 s under TSan). A sleep,
// not std::this_thread::yield(): a yield returns at once when nothing else is
// runnable on this CPU and would not lengthen the window.
inline constexpr long fillers_per_pause = 1000;
inline constexpr std::chrono::milliseconds pause_length{2};

} // namespace hp_drain_detail

// One drain round: retires a sentinel, then fillers until a scan has destroyed
// the sentinel (see the CONTRACT above). Returns the number of fillers retired
// (about 1000 minus the retirements already pending, for a default threshold).
//
// The flag is local to this call, not a shared static: a static would cross-talk
// between nested or concurrent drains (a drain inside a deleter, a straggler's
// scan running the sentinel's destructor on another thread), letting one drain
// return on another's sentinel.
//
// At the filler cap the drain prints a message and calls abort() instead of
// returning: the sentinel is still pending and holds the address of this call's
// stack flag, so returning would let a later scan write to a dead stack frame.
// Aborting also stops a test that misuses the drain at a clear message instead of
// letting it continue with a wrong premise. noexcept for the same reason (see
// the CONTRACT): an exception must not unwind past the pending sentinel.
inline long drain_reclamation() noexcept {
    std::atomic<bool> sentinel_destroyed{false};
    (new hp_drain_detail::DrainObject(&sentinel_destroyed))->retire();
    // Check before each filler: with a scan already due, retiring the sentinel
    // itself may have destroyed it.
    for (long fillers = 0; fillers < hp_drain_detail::filler_cap; ++fillers) {
        if (sentinel_destroyed.load(std::memory_order_acquire)) return fillers;
        if (fillers != 0 && fillers % hp_drain_detail::fillers_per_pause == 0) {
            std::this_thread::sleep_for(hp_drain_detail::pause_length);
        }
        (new hp_drain_detail::DrainObject(nullptr))->retire();
    } // loop retiring fillers until the sentinel is destroyed
    if (sentinel_destroyed.load(std::memory_order_acquire)) return hp_drain_detail::filler_cap;
    std::fprintf(stderr,
                 "drain_reclamation(): the sentinel was not reclaimed after %ld fillers; "
                 "the drain was probably called inside an mm_hp scan (from the destructor "
                 "of a retired object), where no scan can start, or while another thread's "
                 "scan held the sentinel (a drain without quiescence). Aborting: the pending "
                 "sentinel points at this call's stack frame.\n",
                 hp_drain_detail::filler_cap);
    std::abort();
} // drain_reclamation()

// Repeats whole drain rounds until `pred()` holds, at most `max_rounds` rounds.
// Returns true as soon as `pred()` holds (checked before each round and after the
// last one; true without draining if it holds on entry), false when `pred()`
// still fails after `max_rounds` rounds. Every round runs to completion, so on
// either return no sentinel is pending and the caller may continue: a test that
// leaks fails as that one test, e.g.
//   ASSERT_TRUE(drain_until([&] { return live() == base; }, 5)) << live() - base;
// instead of aborting the whole binary. Pick max_rounds >= the deepest destructor
// cascade the test can leave pending (one round per layer).
// Requirements: `pred` is callable with no arguments and returns something
// convertible to bool; max_rounds >= 0. Same CONTRACT as drain_reclamation().
// Not noexcept: an exception thrown by `pred()` propagates. That is safe, since
// `pred()` is only called between complete rounds, when no sentinel is pending;
// the rounds themselves cannot throw (drain_reclamation() is noexcept).
template <typename Pred>
[[nodiscard]] bool drain_until(Pred pred, int max_rounds) {
    for (int round = 0; round < max_rounds; ++round) {
        if (pred()) return true;
        drain_reclamation();
    } // loop over drain rounds
    return static_cast<bool>(pred());
} // drain_until()

// One drain at a known early point, before the code under test runs (test
// binaries do it from a gtest Environment, hp_drain_gtest.h). Two effects:
// - mm_hp's first scan registers the process for membarrier (a function-local
//   static in p1202::asymmetric_thread_fence_heavy()); doing it here puts the
//   registration at the same place in every run instead of inside whichever test
//   first crosses the threshold. TSan's sensitivity to hazard-pointer misuse
//   depends on when that registration happens (measured with deliberately broken
//   protocols: with "no hazard at all", 0 of 30 runs reported with registration
//   at the first scan during the test, 10 of 10 with one scan done before the
//   reader thread started; the mechanism is not understood).
// - At process start (nothing protected, no destructor that retires), leaves
//   mm_hp's retired list empty at a known point. Later, protected objects and a
//   cascade's next layer can remain pending, as after any drain.
// Same CONTRACT as drain_reclamation().
inline void force_early_scan() noexcept {
    drain_reclamation();
}

#endif // INCLUDED_HP_DRAIN_H_
