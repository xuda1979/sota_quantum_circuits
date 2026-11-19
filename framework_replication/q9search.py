#!/usr/bin/env python3
"""Definitive Q=9 (one-ancilla) search for T=5 F24 inversion.

Model (complete for one ancilla, operands from {x-wires, t} only):
  - ops: ccx(A, B, tgt), A/B = transient affine shapings of two DISTINCT
    wires from {x0..x3, t}; tgt in {t, h0..h3}.
  - t-op operands: x-affine pairs only (no aliasing ccx(t,..,t)).
  - h-op operands: x-affine pairs or (t-prefix shape & x-affine shape).
  - snapshots: cx(t, h_b) at ANY time (free) -> any subset of t-prefix
    d2 content reaches any bit; prefix set is triangular in the t-op
    products P_i, so acceptance uses span{d2(P_i)}.
  - final affine assembly into each h_b: free.
  - MBU free of t: per-product cz(A_i, B_i, cond=m) reproduces t's value
    exactly (each product's operand shapes are re-creatable for free).

Acceptance: for every bit b: d2(TGT_b ^ acc_b) in span{d2(P_i)}.
Admissible prune: rank of span{P_i so far} + {TGT_b ^ acc_b} must grow by
at most the remaining op count.

Run: python3 q9search.py [max_ops]
"""
import sys
import time
from itertools import combinations

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF
MONOS = [m for m in range(1, 16) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}

TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]


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


def dim(vecs):
    return len(rref(vecs))


# all non-const affine forms over x, grouped by a wire that can hold them
# (a wire j can hold any affine form whose ANF contains x_j)
AFFS = []
for supp in range(1, 16):
    for c in (0, 1):
        tt = FULL if c else 0
        for j in range(4):
            if supp >> j & 1:
                tt ^= XF[j]
        AFFS.append((supp, c, tt))
# per-wire holdable forms
BY_WIRE = [[f for f in AFFS if f[0] >> w & 1] for w in range(4)]


def xpair_products():
    """Distinct product TTs of two affine forms on distinct wires."""
    seen = set()
    out = []
    for w1 in range(4):
        for w2 in range(w1 + 1, 4):
            for (s1, c1, t1) in BY_WIRE[w1]:
                for (s2, c2, t2) in BY_WIRE[w2]:
                    p = t1 & t2
                    if p and p not in seen:
                        seen.add(p)
                        out.append(((w1, s1, c1), (w2, s2, c2), p))
    return out


XPRODS = xpair_products()
print(f"x-affine pair products: {len(XPRODS)}", file=sys.stderr)


def tpair_products(t_tt):
    """Distinct product TTs of (t-shape & x-affine), t's current value t_tt."""
    if not t_tt:
        return []
    seen = set()
    out = []
    # t-shape: t_tt ^ l(x) ^ c for any affine l, c
    shapes = {t_tt}
    for supp in range(16):
        base = t_tt
        for j in range(4):
            if supp >> j & 1:
                base ^= XF[j]
        shapes.add(base)
        shapes.add(base ^ FULL)
    for sh in shapes:
        for w in range(4):
            for (s, c, tt) in BY_WIRE[w]:
                p = sh & tt
                if p and p not in seen:
                    seen.add(p)
                    out.append((("t", sh), (w, s, c), p))
    return out


def search(max_ops=5, want_first=True):
    memo = set()
    best = [None]
    nodes = [0]
    t0 = time.time()

    def prune(t_d2_basis, acc, depth):
        need = [TGT_D2[b] ^ d2(acc[b]) for b in range(4)]
        allv = list(t_d2_basis) + need
        extra = dim(allv) - dim(t_d2_basis)
        return extra > max_ops - depth

    def dfs(t_tt, t_d2_basis, acc, depth, ops):
        nodes[0] += 1
        if nodes[0] % 200000 == 0:
            print(f"  nodes={nodes[0]} depth={depth} "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)
        if depth == max_ops:
            for b in range(4):
                need = TGT_D2[b] ^ d2(acc[b])
                if need and not contains(t_d2_basis, need):
                    return False
            best[0] = list(ops)
            return True
        if prune(t_d2_basis, acc, depth):
            return False
        key = (t_tt, tuple(acc), depth)
        if key in memo:
            return False
        memo.add(key)

        # -- ops into t: x-affine pair products --------------------------
        for (a1, a2, p) in XPRODS:
            nt = t_tt ^ p
            npd2 = d2(p)
            nb = t_d2_basis if contains(t_d2_basis, npd2) \
                else rref(list(t_d2_basis) + [npd2])
            if dfs(nt, nb, acc, depth + 1,
                   ops + [("t", a1, a2, p)]):
                return True

        # -- ops into h_b -------------------------------------------------
        hmenu = [((w1, s1, c1), (w2, s2, c2), p)
                 for (w1, s1, c1, w2, s2, c2) in
                 [(a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2])
                  for a in XPRODS]] + tpair_products(t_tt)
        # dedup on TT
        seen = set()
        hmenu = [h for h in hmenu if not (h[2] in seen or seen.add(h[2]))]
        for b in range(4):
            for (a1, a2, p) in hmenu:
                nacc = list(acc)
                nacc[b] ^= p
                if dfs(t_tt, t_d2_basis, tuple(nacc), depth + 1,
                       ops + [(b, a1, a2, p)]):
                    return True
        return False

    ok = dfs(0, tuple(), (0, 0, 0, 0), 0, [])
    return best[0], nodes[0], ok


if __name__ == "__main__":
    max_ops = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res, n, ok = search(max_ops)
    if ok:
        print(f"FOUND T={max_ops} Q=9 (or Q=8 if no t-op) after {n} nodes")
        for k, op in enumerate(res):
            tgt, a1, a2, p = op
            print(f"  op{k+1}: {a1} & {a2} -> {tgt}: "
                  f"prod d2={d2(p):03x} tt={p:04x}")
        if all(op[0] == "t" for op in [] ) or not any(
                isinstance(op[0], int) and False for op in res):
            pass
        if not any(op[0] == "t" for op in res):
            print("  (no t-ops: circuit uses Q=8!)")
    else:
        print(f"NOT FOUND after {n} nodes")
