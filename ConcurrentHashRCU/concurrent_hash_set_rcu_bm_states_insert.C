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
 * ConcurrentResizableHashSetRCU Benchmarks: the insert workloads on the
 * AllowDelete set in its states -- empty, prefilled with live keys, and
 * prefilled by churn with its victims reclaimed or merely erased. One of the
 * benchmark's translation units (the overview of the workloads, the index of
 * the units and why there are several: concurrent_hash_set_rcu_bm.C); the
 * code generated for these rows depends on this unit alone. The fixtures,
 * the churn fixture that builds the states (its comment describes them) and
 * the bodies are the shared header's; this unit only registers them. The
 * lookup workload on the same states is concurrent_hash_set_rcu_bm_states_lookup.C.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  Insert_MostlyNew_Del           the growth inserts on an EMPTY AllowDelete
 *                                 set: against _Concurrent, whether compiling
 *                                 erase() in costs the insert path anything.
 *  Insert_MostlyNew_Del_Control   the same on a set prefilled with live keys
 *                                 and empty free lists: a resident population
 *                                 with every allocation an append.
 *  Insert_MostlyNew_Del_Reclaimed the same population reached by churn and
 *                                 one advance(): inserts that pop freed slots
 *                                 in a table the larger prefill grew further --
 *                                 "churn, then growth", not an allocator cell.
 *  Insert_MostlyNew_Presized_Del  the pre-sized inserts on an EMPTY
 *                                 AllowDelete set: erase() compiled in, the
 *                                 arena alone.
 *  Insert_MostlyNew_Presized_Del_Control
 *                                 prefilled, no churn, the same buckets as the
 *                                 next two: the control for the free lists.
 *  Insert_MostlyNew_Presized_Del_Reclaimed
 *                                 churn and one advance() at fixed geometry:
 *                                 the free-list pop path and the locality of
 *                                 the popped slots, the controlled allocator
 *                                 cell.
 *  Insert_MostlyNew_Presized_Del_Erased
 *                                 churn without advance(): inserts whose scans
 *                                 walk chains the erases already unlinked
 *                                 from; what eager unlinking buys a writer.
 */

// The reclamation cells of the insert workloads (the header's "Reclamation
// benchmarks": the AllowDelete == true set empty, prefilled, and prefilled
// by churn). Nothing here changes a registration of another unit.
DEFINE_MOSTLY_NEW(Insert_MostlyNew_Del, ConcurrentSetDel)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Del_Control,   NewControl)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Del_Reclaimed, NewReclaimed)

DEFINE_MOSTLY_NEW_ON(MostlyNewPresizedFixture, Insert_MostlyNew_Presized_Del, ConcurrentSetDel)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Control,   PresizedControl)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Reclaimed, PresizedReclaimed)
DEFINE_MOSTLY_NEW_ON(ChurnFixture, Insert_MostlyNew_Presized_Del_Erased,    PresizedErased)
