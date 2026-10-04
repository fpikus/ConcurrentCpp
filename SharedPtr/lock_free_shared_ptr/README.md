# Lock-free atomic shared pointer

The genuinely lock-free atomic shared pointer benchmarked in this repository
is built on Daniel Anderson's `parlay::shared_ptr` and hazard pointers,
presented at CppCon 2023 ("Lock-free Atomic Shared Pointers Without a Split
Reference Count? It Can Be Done!"):

https://github.com/DanielLiamAnderson/atomic_shared_ptr

```
SharedPtr/lock_free_shared_ptr/
├── README.md                 (this file)
├── atomic_shared_ptr.hpp     (our adapter: parlay::atomic_shared_ptr with a
│                              Harris mark in bit 0 of the control-block pointer)
├── parlay.patch              (our changes to his internals)
└── parlay/                   (made by ../make_third_party.sh; not in the repository)
```

His code is not copied into this repository. `../make_third_party.sh` makes
`parlay/` from a clone of his repository plus `parlay.patch`, and adds the
headers of ParlayLib (https://github.com/cmuparlay/parlaylib), whose pool
allocator his `shared_ptr` uses; SharedPtr's README, "Third-party code", has
the commands. The patch's header says what each change is for:

- his code also needs Facebook's folly, for a hash set and a pair of
  asymmetric fences; the patch replaces them with `std::unordered_set` and a
  sequentially consistent fence on both sides, so folly is not needed;
- `atomic_shared_ptr.hpp` hands out pointers that may carry the list's
  deletion mark, so the patch strips bit 0 of the control-block pointer in
  the reference-count helpers every `parlay::shared_ptr` operation goes
  through (`parlay::weak_ptr`'s converting constructors excepted; nothing
  here uses `weak_ptr`).
