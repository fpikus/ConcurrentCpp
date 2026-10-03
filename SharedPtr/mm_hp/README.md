# mm_hp

- A performant, complete implementation of C++26 hazard pointers ([saferecl.hp]) for Linux x86_64 and gcc.
- Intended as guidance for standard library implementers on ABI robustness and as a performance reference point.
- For readability, internal names deliberately avoid using the standard library `__x` reserved style.
- Licensed so that standard library implementers can take all or part of this code — see [LICENSES](LICENSES).
- If an implementation needs different terms, additional licenses may be considered.
- Bug reports and written suggestions are welcome; patches and code cannot be accepted in order to keep the copyright in one place to facilitate adding licenses.
- No guarantee of support of any kind.

## Performance results

```
Operation                                    1-thread ns/op   8-thread scalability
load and dereference, unprotected            0.233            7.93x
+ hazard pointer protection                  0.410 (Δ 0.177)  7.92x
+ hazard pointer ctor/dtor per iteration     1.732 (Δ 1.499)  7.94x
```

- 10,000 pre-existing hazard pointers make no measurable difference.
- Measured with [hp_bench](https://github.com/magedm/hp_bench).
- Extracted from [mm_hp-2026-09-28-105818.txt](https://github.com/magedm/hp_bench/blob/main/results/mm_hp-2026-09-28-105818.txt).
