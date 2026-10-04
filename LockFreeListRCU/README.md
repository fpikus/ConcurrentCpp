# LockFreeListRCU

A Harris-style lock-free singly-linked list that reclaims memory by
generations instead of reference counts: every operation runs under a
handle, erased nodes are retired at their physical unlink, and `reclaim()`
frees a generation's nodes for reuse once no handle can still observe them.
The reference-counted list it is compared against lives in `../LockFreeList`.

## Building and testing

```sh
make tests      # the two sanitizer test binaries
make run_tests  # build and run both sanitizer test binaries
```

The test binaries are written to `build/$(hostname)/`:
`lock_free_list_rcu_test_asan` and `lock_free_list_rcu_test_tsan`. The
benchmark (`lock_free_list_rcu_bm.C`, binary `lock_free_list_rcu_bm`) does
not exist yet; until it does, `make`, `make all`, `make benchmarks` and
`make run_benchmarks` fail, so use `make tests` and `make run_tests`.

Requires clang (`../config.mk` selects `clang++-22` and C++23), Google
Benchmark and GoogleTest; point `GBENCH_DIR` and `GTEST_DIR` at your
installations if they are not in `$HOME/GoogleBench` and `$HOME/GoogleTest`.

- `lock_free_list_rcu.h` — the list
- `rcu_test_common.h` — shared test payloads, test hooks and accounting
  checks
- `lock_free_list_rcu_test.C` — basic and stress tests (ported from
  `../LockFreeList`)
- `lock_free_list_rcu_scheme_test.C` — white-box tests of the reclamation
  scheme
- `lock_free_list_rcu_contract_test.C` — black-box tests of the public
  contract
- `lock_free_list_rcu_bm.C` — the benchmarks (not yet present)
