# Lock-free atomic shared pointer (external dependency)

The genuinely lock-free atomic shared pointer benchmarked in this repository
is Daniel Anderson's `parlay::atomic_shared_ptr`, presented at CppCon 2023
("Lock-free Atomic Shared Pointers Without a Split Reference Count? It Can Be
Done!"). His code is not redistributed here; get it from the source:

https://github.com/DanielLiamAnderson/atomic_shared_ptr

To build the lock-free pointer variant of the list benchmarks, place his
`parlay/` header directory here, next to this README:

```
SharedPtr/lock_free_shared_ptr/
├── README.md                 (this file)
├── atomic_shared_ptr.hpp     (our adapter over the parlay internals)
└── parlay/                   (from Daniel Anderson's repository)
```

`atomic_shared_ptr.hpp` is our own adapter and is included in this
repository.
