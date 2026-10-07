# ConcurrentResizableHashSetRCU

A copy of `ConcurrentResizableHashSet` from `../ConcurrentHash`, being changed
so that dead nodes (tombstones of erased values and the copies a lazy split
superseded) are removed from their chains by the concurrent operations
themselves. In the original, a dead node stays in its chain, and every later
walk of that chain steps over it, until `reclaim()` unlinks it. Taking dead
nodes out of the chains on the fly is the first step toward epoch-based
reclamation.

Work in progress. The copy was made from commit `e37a9d3`, and so far only the
names differ (`ConcurrentResizableHashSetRCU`, `concurrent_hash_set_rcu.h`, and
so on): the code, its comments, and the tests and benchmarks all still describe
the original's behavior. See `../ConcurrentHash/README.md` for that.
