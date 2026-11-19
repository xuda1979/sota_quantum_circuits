#!/usr/bin/env python3
"""Full physical search for the 5-Toffoli F24 inversion at Q=10.

Physical model (exactly the paper's):
  wires: x0..x3 (inputs, must end restored), t, u (ancillas, must end |0>
  via MBU), h0..h3 (outputs, must end = h ^ x^-1).
  ops: ccx(a, b, c) with a,b,c distinct wires; operands may be transiently
  shaped (cx from x-wires + x-gates, restored after) — free.
  MBU: hmr(t) + cz fixup frees a dirty ancilla (1 measurement, 0 Toffoli).
  cost: 5 Toffolis total; accept when h == targets, x restored, ancillas
  cleanable via MBU (their fixup partners exist).

State: (t, u, h0, h1, h2, h3) as truth tables (16-bit ints). x-wires
never change (shaping of x is transient cx among x-wires + x — any affine
form of x available as operand for free).

Operand shaping: any affine (GF(2)-linear + const) form of a wire = the
wire XORed with x-subsets XOR const. Transient: shape, op, unshape.
"""
import sys
from itertools import combinations

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF
TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]
MONOS = [m for m in range(16) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)


def anf_deg2(fn):
    """deg>=2 monomial coordinate vector of a truth table."""
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    v = 0
    for m in MONOS:
        if f[m]:
            v |= 1 << IDX[m]
    return v


TGT_D2 = [anf_deg2(t) for t in TGT]


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


def shapes(base):
    out = []
    for r in range(5):
        for supp in combinations(range(4), r):
            for c in range(2):
                v = base
                for j in supp:
                    v ^= XF[j]
                v ^= FULL if c else 0
                out.append(v)
    return out


_search_cache = {}


def all_shapes(base):
    hit = _search_cache.get(base)
    if hit is None:
        hit = shapes(base)
        _search_cache[base] = hit
    return hit


def search(max_ops=5, verbose=False):
    memo = set()
    nodes = [0]
    path = []

    def dfs(t, u, h, depth, basis):
        """h = tuple of 4 truth tables; basis = rref of deg2 product vecs."""
        nodes[0] += 1
        if verbose and nodes[0] % 500000 == 0:
            print(f"  nodes={nodes[0]} depth={depth}", file=sys.stderr)
        if depth == max_ops:
            return list(h) == TGT
        # rank prune: h's deg>=2 residual to target must be coverable by
        # the products still to come (each op adds <= 1 new direction).
        # basis = products seen so far; residual rank counted over all bits.
        miss = 0
        nb = list(basis)
        for j in range(4):
            r = TGT_D2[j] ^ anf_deg2(h[j])
            if r:
                rr = r
                for bb in nb:
                    rr = min(rr, rr ^ bb)
                if rr:
                    nb.append(rr)
                    nb.sort(reverse=True)
                    miss += 1
        if miss > max_ops - depth:
            return False
        key = (t, u, h, depth)
        if key in memo:
            return False
        memo.add(key)
        # operand wires: t, u, h0..h3 (x-wires' shapes are pure x-forms)
        wires = [t, u, *h]
        for wi in range(6):
            for wj in range(wi + 1, 6):
                for a in all_shapes(wires[wi]):
                    for b in all_shapes(wires[wj]):
                        p = a & b
                        if not p:
                            continue
                        nb = basis
                        pv = anf_deg2(p)
                        if pv and not contains(basis, pv):
                            nb = rref(list(basis) + [pv])
                        for ti in range(6):
                            if ti in (wi, wj):
                                continue  # aliasing
                            nt, nu, nh = t, u, list(h)
                            if ti == 0:
                                nt = t ^ p
                            elif ti == 1:
                                nu = u ^ p
                            else:
                                nh[ti - 2] ^= p
                            if dfs(nt, nu, tuple(nh), depth + 1, nb):
                                path.append((wi, wj, a, b, p, ti))
                                return True
        return False

    ok = dfs(0, 0, (0, 0, 0, 0), 0, tuple())
    if ok:
        path.reverse()
        return path, nodes[0]
    return None, nodes[0]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    res, n = search(verbose=verbose)
    if res:
        print(f"FOUND after {n} nodes")
        names = ["t", "u", "h0", "h1", "h2", "h3"]
        for k, (wi, wj, a, b, p, ti) in enumerate(res):
            print(f"  op{k+1}: ccx({names[wi]}*, {names[wj]}*, {names[ti]})"
                  f"  prod={p:04x}")
    else:
        print(f"NOT FOUND after {n} nodes")
