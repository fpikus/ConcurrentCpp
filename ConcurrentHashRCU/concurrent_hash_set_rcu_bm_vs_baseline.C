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
#include "concurrent_hash_set_rcu_bm.h"

/*
 * ConcurrentResizableHashSetRCU Benchmarks: the insert and lookup workloads on
 * the two containers, the concurrent set against the baseline. One of the
 * benchmark's translation units (the overview of the workloads, the index of
 * the units and why there are several: concurrent_hash_set_rcu_bm.C); the
 * code generated for these rows depends on this unit alone. The fixtures and
 * the bodies are the shared header's; this unit only registers them.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  Insert_MostlyNew_Concurrent    pure insertion through live doublings and
 *                                 lazy splits: the growth path of this set.
 *  Insert_MostlyNew_RWLocked      the same inserts on the baseline: the price
 *                                 of a writer-exclusive rehash.
 *  Insert_MostlyNew_Presized_Concurrent
 *                                 the same inserts into a pre-sized table: the
 *                                 arena alone, with the doubling and split
 *                                 path subtracted out.
 *  Insert_MostlyNew_Presized_RWLocked
 *                                 the baseline pre-sized the same way: its
 *                                 lock alone.
 *  Lookup_MostlyOld_Concurrent    mostly lookups at a fixed hit rate with a
 *                                 trickle of inserts and no resize: what
 *                                 readers pay for coexisting with writers.
 *  Lookup_MostlyOld_RWLocked      the same on the baseline: the RMW every
 *                                 reader pays to be visible to a rehash.
 *  Lookup_MostlyOld_OpHandle      the lookup workload with a handle taken and
 *                                 released around every operation, where
 *                                 _Concurrent keeps one per thread: what a
 *                                 handle costs when it is not long-lived.
 */

// ---------------------------------------------------------------------------
// Registrations: identical workloads, two containers.
// ---------------------------------------------------------------------------
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW(Insert_MostlyNew_RWLocked,   LockedSet)

DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_RWLocked,   LockedSet)

DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Concurrent, ConcurrentSet)
DEFINE_MOSTLY_OLD(Lookup_MostlyOld_RWLocked,   LockedSet)

// Lookup_MostlyOld_OpHandle: Lookup_MostlyOld_Concurrent with every operation
// under a handle of its own, taken just before it and released when it ends
// (DEFINE_MOSTLY_OLD_UNDER's HANDLE). Why measure: the rows above keep one
// long-lived handle per thread, whose join happens once, untimed; a caller
// that cannot keep a handle pays a join and a leave per operation instead.
// The join is a seq_cst increment of the current generation's count in the
// joining thread's arena shard and a seq_cst re-load of the current
// generation, plus a relaxed increment of that shard's diagnostic join
// counter; the leave is a release decrement of the count the join
// incremented. So a thread's joins and leaves write lines of its own shard's
// count row only, and threads share those lines only when they share a shard
// (more threads than shards, or two thread numbers equal modulo the shard
// count: Google Benchmark's thread 0 is the main thread, numbered before the
// others). The threads' long-lived handles are still made and destroyed as
// in every row, unused.
// Only on the AllowDelete == false set, against _Concurrent.
DEFINE_MOSTLY_OLD_UNDER(MostlyOldFixture, Lookup_MostlyOld_OpHandle, ConcurrentSet, set->new_handle())
