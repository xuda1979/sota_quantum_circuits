#!/usr/bin/env python3
"""Verified factorized search for the paper's 5-Toffoli F24 structure.

Model (Ext. Data Fig. 1c, generalized to two ancillas):
  - ops: ccx(shaped a, shaped b, target) — operands from {x0..x3, t, u},
    targets in {t, u}; every op's product joins the accumulator.
  - snapshots: after any op, free cx(acc -> h[b]) for any subset.
  - final: free affine assembly into each h[b].
  - accept: for each bit, D2(TGT[b]) in span{D2(boundary accumulator
    values)} — equivalent to subset-XOR realizability because a residual
    with zero D2 part is affine (free).

The search factorizes: op sequence determines the boundary values
(t_i, u_i); acceptance is a D2-span check on those values. Snapshot
schedules are then solved post-hoc (linear algebra), and the final
affine assembly is free.
"""
import sys
from itertools import combinations

XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF
INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]
MONOS = [m for m in range(16) if bin(m).count("1") >= 2]
NM = len(MONOS)
assert all(t <= FULL for t in TGT)


def d2(fn):
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    v = 0
    for i, mm in enumerate(MONOS):
        if f[mm]:
            v |= 1 << i
    return v


TGT_D2 = [d2(t) for t in TGT]


def rref(vecs):
    basis = []
    for v in vecs:
        if not v:
            continue
        for bb in basis:
            v = min(v, v ^ bb)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return tuple(basis)


def contains(basis, v):
    for bb in basis:
        v = min(v, v ^ bb)
    return v == 0


_scache = {}


def shapes(base):
    hit = _scache.get(base)
    if hit is not None:
        return hit
    out = []
    for r in range(5):
        for supp in combinations(range(4), r):
            for c in range(2):
                v = base
                for j in supp:
                    v ^= XF[j]
                v ^= FULL if c else 0
                out.append(v)
    out = list(dict.fromkeys(out))
    _scache[base] = out
    return out


def products_from(t, u):
    """All realizable products: shaped-x & shaped-x, shaped-x & shaped-t,
    shaped-x & shaped-u, shaped-t & shaped-u."""
    out = set()
    for i in range(4):
        for j in range(i + 1, 4):
            for a in shapes(XF[i]):
                for b in shapes(XF[j]):
                    p = a & b
                    if p:
                        out.add(p)
    for w in (t, u):
        if not w:
            continue
        for i in range(4):
            for a in shapes(XF[i]):
                for b in shapes(w):
                    p = a & b
                    if p:
                        out.add(p)
    if t and u:
        for a in shapes(t):
            for b in shapes(u):
                p = a & b
                if p:
                    out.add(p)
    return out


_pcache = {}


def prods(t, u):
    key = (t, u)
    hit = _pcache.get(key)
    if hit is None:
        hit = products_from(t, u)
        _pcache[key] = hit
    return hit


def solve(max_ops=5):
    memo = set()
    path = []

    def dfs(t, u, bounds, depth):
        if depth == max_ops:
            vals = [v for pair in bounds for v in pair]
            basis = rref([d2(v) for v in vals])
            return all(contains(basis, td) for td in TGT_D2)
        key = (t, u, depth)
        if key in memo:
            return False
        memo.add(key)
        # prune: missing D2-rank vs remaining ops
        vals0 = [v for pair in bounds for v in pair]
        basis = rref([d2(v) for v in vals0])
        nb = list(basis)
        miss = 0
        for td in TGT_D2:
            if not contains(rref(nb), td):
                nb.append(td)
                nb = list(rref(nb))
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
                    path.append((t, u, p, tgt))
                    return True
        return False

    ok = dfs(0, 0, ((0, 0),), 0)
    if ok:
        path.reverse()
        return path
    return None


def verify(path):
    """Independently replay and verify the found path."""
    t, u = 0, 0
    bounds = [(0, 0)]
    for k, (pt, pu, p, tgt) in enumerate(path):
        assert (pt, pu) == (t, u), f"op{k+1}: state mismatch"
        assert p in prods(t, u), f"op{k+1}: product not realizable"
        assert p <= FULL
        if tgt == "t":
            t = t ^ p
        else:
            u = u ^ p
        bounds.append((t, u))
    # acceptance: D2 span
    vals = [v for pair in bounds for v in pair]
    basis = rref([d2(v) for v in vals])
    for b, td in enumerate(TGT_D2):
        assert contains(basis, td), f"bit {b} not in span"
    return bounds


if __name__ == "__main__":
    path = solve()
    if path is None:
        print("NOT FOUND")
        sys.exit(1)
    print(f"FOUND: {len(path)} ops")
    for k, (t, u, p, tgt) in enumerate(path):
        print(f"  op{k+1}: (t={t:04x}, u={u:04x}) ^ p={p:04x} -> {tgt}")
    bounds = verify(path)
    print("verified. boundary values:")
    for i, (t, u) in enumerate(bounds):
        print(f"  boundary {i}: t={t:04x} u={u:04x}")
