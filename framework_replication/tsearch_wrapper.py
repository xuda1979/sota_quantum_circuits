#!/usr/bin/env python3
"""Reproducible wrapper for the T>=5 leg (and other tsearch depths).

Runs tsearch.solve(max_ops) for a given op count and prints a
machine-readable result line with node count and wall time.

Usage: python3 tsearch_wrapper.py <max_ops>
"""
import sys
import time

sys.path.insert(0, "/home/david/software/sota_quantum_circuits/framework_replication")
import tsearch  # noqa: E402


def count_nodes(max_ops: int) -> tuple[bool, int, float]:
    """Re-run the exact tsearch.solve DFS with node counting.

    tsearch.solve's internal DFS is replicated 1:1 here with a node
    counter (verified to produce identical FOUND/NOT_FOUND and identical
    solution paths at max_ops=4 and 5).
    """
    memo = set()
    nodes = [0]
    t0 = time.time()
    rref = tsearch.rref
    contains = tsearch.contains
    prods = tsearch.prods
    TGT_D2 = tsearch.TGT_D2

    def dfs(t, u, bounds, depth):
        nodes[0] += 1
        if depth == max_ops:
            vals = [v for pair in bounds for v in pair]
            basis = rref([tsearch.d2(v) for v in vals])
            return all(contains(basis, td) for td in TGT_D2)
        key = (t, u, depth)
        if key in memo:
            return False
        memo.add(key)
        vals0 = [v for pair in bounds for v in pair]
        basis = rref([tsearch.d2(v) for v in vals0])
        nb = list(basis)
        miss = 0
        for td in TGT_D2:
            if not contains(tsearch.rref(nb), td):
                nb.append(td)
                nb = list(tsearch.rref(nb))
                miss += 1
        if miss > max_ops - depth:
            return False
        for p in prods(t, u):
            for tgt in ("t", "u"):
                if tgt == "t":
                    nt, nu = t ^ p, u
                else:
                    nt, nu = t, u ^ p
                if dfs(nt, nu, bounds + ((nt, nu),), depth + 1):
                    return True
        return False

    found = dfs(0, 0, ((0, 0),), 0)
    dt = time.time() - t0
    return found, nodes[0], dt


if __name__ == "__main__":
    max_ops = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    found, n, dt = count_nodes(max_ops)
    print(f"max_ops={max_ops} {'FOUND' if found else 'NOT_FOUND'} "
          f"nodes={n} seconds={dt:.3f}")
    sys.exit(0 if not found else 1)
