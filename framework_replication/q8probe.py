#!/usr/bin/env python3
"""Probe: Q=8 (no ancilla) and Q=9 (one ancilla) feasibility for T=5 F24.

Model facts used:
  - Every wire value = affine(x,h_in) ^ XOR of ccx outputs landing in it
    (directly or via t); hence deg>=2 part of every wire lies in the span of
    the deg>=2 parts of the <=T ccx outputs. So T >= dim span{d2(INV_b)}.
  - No-ancilla (Q=8): products of affine-x pairs land in bits directly;
    bit b needs d2(INV_b) = XOR of its own products' d2 vecs (exact).
  - One-ancilla (Q=9): t-products (x-only operands, else unfixable with a
    single ancilla) are available to every bit via prefix snapshots; direct
    products may also use t's current prefix value as an operand.
"""
import sys
from itertools import combinations

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
MONOS = [m for m in range(1, 16) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)
XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF


def d2(fn):
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    v = 0
    for m in range(1, 16):
        if f[m] and m in IDX:
            v |= 1 << IDX[m]
    return v


TGT = [d2(sum(((INV[x] >> b) & 1) << x for x in range(16))) for b in range(4)]


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


def dim(vecs):
    return len(rref(vecs))


# affine forms over x: supp mask + const
AFF = [(s, c) for s in range(16) for c in (0, 1)]


def aff_tt(s, c):
    t = FULL if c else 0
    for j in range(4):
        if s >> j & 1:
            t ^= XF[j]
    return t


# menu: distinct d2 vecs of products of two affine forms (nonzero d2 part),
# keeping one realization (A, B) per vec.
menu = {}
for i, (sa, ca) in enumerate(AFF):
    ta = aff_tt(sa, ca)
    for sb, cb in AFF:
        if (sb, cb) == (sa, ca):
            continue
        tb = aff_tt(sb, cb)
        v = d2(ta & tb)
        if v and v not in menu:
            menu[v] = (ta, tb)
print(f"dim W = {dim(TGT)}   (lower bound: T >= {dim(TGT)})")
print(f"target vecs: {[hex(t) for t in TGT]}")
print(f"menu size: {len(menu)}")

# --- Q=8: per-bit min products (BFS over XOR of menu vecs) ------------------
def min_products(target, max_k=4):
    if target == 0:
        return 0, []
    frontier = {v: [v] for v in menu}
    if target in frontier:
        return 1, [target]
    for k in range(2, max_k + 1):
        nxt = {}
        for v, path in frontier.items():
            for m in menu:
                w = v ^ m
                if w not in nxt and w not in frontier:
                    nxt[w] = path + [m]
        frontier.update(nxt)
        if target in frontier:
            return k, frontier[target]
    return None, None


total = 0
ok8 = True
per_bit = []
for b in range(4):
    k, path = min_products(TGT[b])
    if k is None:
        ok8 = False
        per_bit.append(None)
        print(f"bit {b}: needs > 4 products — Q=8 infeasible for T=5")
        break
    per_bit.append(path)
    total += k
    print(f"bit {b}: min {k} products (within 4): {[hex(p) for p in path]}")
if ok8:
    print(f"sum of per-bit minima = {total}")
    if total <= 5:
        print("Q=8 SOLUTION EXISTS with T=%d (TQ=%d)" % (total, 8 * total))
    else:
        print("Q=8 infeasible at T=5 (needs >= %d ops)" % total)

# --- Q=9, k=5 all-into-t: 5 menu vecs spanning W ------------------------------
W = TGT
basis = rref(W)
cand = [v for v in menu if contains(basis, v)]
print(f"\nmenu vecs inside W: {len(cand)}")


def span_cover(vs, need, k):
    """choose k vecs from vs whose span contains all of `need`."""
    out = []

    def dfs(chosen, depth):
        cur = rref(chosen)
        if all(contains(cur, t) for t in need):
            return chosen
        if depth == k:
            return None
        # prune: missing rank <= remaining slots
        nb = list(cur)
        miss = 0
        for t in need:
            if not contains(rref(nb), t):
                nb.append(t)
                nb = list(rref(nb))
                miss += 1
        if miss > k - depth:
            return None
        for v in vs:
            if v in chosen:
                continue
            r = dfs(chosen + [v], depth + 1)
            if r:
                return r
        return None

    return dfs([], 0)


import time
t0 = time.time()
res = span_cover(sorted(cand), W, 5)
if res:
    print(f"Q=9 ALL-INTO-T SOLUTION (k=5): {[hex(v) for v in res]} "
          f"({time.time()-t0:.1f}s)")
    print(f"  -> T=5, Q=9, TQ=45 (paper: TQ=50)")
else:
    print(f"k=5 all-into-t NOT coverable from menu-in-W ({time.time()-t0:.1f}s)")
    # try menu vecs outside W too (their excursions must cancel in XORs —
    # but for pure span-cover of W with 5 vecs, chosen vecs must be
    # independent and span exactly W... only if dim W == 5)
    if dim(W) == 5:
        print("dim W == 5: any 5-vec cover is a basis of W — no other options")
