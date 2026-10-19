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
 * ConcurrentResizableHashSetRCU Benchmarks: the lookup workload on the
 * AllowDelete set in its states -- empty (prefilled as the lookup fixture
 * always is), prefilled through the churn fixture, and prefilled by churn
 * with its victims reclaimed or merely erased. One of the benchmark's
 * translation units (the overview of the workloads, the index of the units
 * and why there are several: concurrent_hash_set_rcu_bm.C); the code
 * generated for these rows depends on this unit alone. The fixtures, the
 * churn fixture that builds the states (its comment describes them) and the
 * body are the shared header's; this unit only registers them. The insert
 * workloads on the same states are concurrent_hash_set_rcu_bm_states_insert.C.
 *
 * Table of contents, in the order of the file, with why each is measured:
 *
 *  Lookup_MostlyOld_Del           the lookup workload on the AllowDelete set:
 *                                 against _Concurrent, erase() compiled in on
 *                                 the read path.
 *  Lookup_MostlyOld_Del_Control   the same prefill through the churn fixture:
 *                                 a check of the fixture against _Del.
 *  Lookup_MostlyOld_Del_Reclaimed the misses walk chains the victims were
 *                                 unlinked from, and the trickle inserts pop
 *                                 the slots advance() freed.
 *  Lookup_MostlyOld_Del_Erased    the misses walk chains the erases unlinked
 *                                 from, nothing reclaimed: what eager
 *                                 unlinking buys a reader.
 */

// The reclamation cells of the lookup workload (the header's "Reclamation
// benchmarks"). Nothing here changes a registration of another unit.
DEFINE_MOSTLY_OLD(Lookup_MostlyOld_Del, ConcurrentSetDel)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Control,   OldControl)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Reclaimed, OldReclaimed)
DEFINE_MOSTLY_OLD_ON(ChurnFixture, Lookup_MostlyOld_Del_Erased,    OldErased)
