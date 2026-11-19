#!/usr/bin/env python3
"""Guided physical wiring search for the 5-product F24 inversion.

The span solution is known (fastspan.py, op1 #3):
    p1 = {6} (x0*x3... see MONOS), p2 = {9,10,12,14}, p3 = {3,5,6,10,11},
    p4 = {5,7,14}, p5 = {3,7,10,12,13}
    h[0] ^= p1^p4^x0^x1^x2^x3; h[1] ^= p3^x3; h[2] ^= p2^p4^p5^x2^x3;
    h[3] ^= p2^x1^x2^x3

Question: is there a PHYSICAL 2-ancilla schedule (operands only shaped
{x0..x3, t, u}, targets t/u/h_i) realizing exactly 5 Toffolis with all
targets in span afterwards?

This search fixes NOTHING about the products: it enumerates all physical
5-op schedules but with the accept condition = the four TGT functions
being in span(ever-held wires + injections + x + 1), and a STRONGER
prune: after each op, the degree>=2 part of every wire ever held joins a
basis; accept only if all four targets are reachable. The speedup vs
physsearch: state = (t, u) + basis (rref over 11 monomials, NOT truth
tables) — wires are functions of x only, so the basis is small and the
memo is over (rref-basis, t_deg2, u_deg2, injections).
"""
import sys
from itertools import combinations

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
MONOS = [3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = 11
FULL = 0xFFFF

TGT = []
for bit in range(4):
    fn = sum(((INV[x] >> bit) & 1) << x for x in range(16))
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    v = 0
    for m in range(1, 16):
        if f[m] and m in IDX:
            v |= 1 << IDX[m]
    TGT.append(v)


def tt_fn(fn):
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


def anf_masks(fn):
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    return {m for m in range(16) if f[m]}


XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]


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


def mul_lin(v, supp):
    """v * l where l = XOR of x_j (j in supp): idempotent monomial growth."""
    r = 0
    for i in range(NM):
        if v >> i & 1:
            m = MONOS[i]
            for j in supp:
                m2 = m if (m >> j) & 1 else m | (1 << j)
                r ^= 1 << IDX[m2]
    return r


def prod_vec(a, b):
    """AND of two functions given as (deg2 vec, linear supp, const)."""
    return a, b  # placeholder


def fn_of(spec):
    """spec = (vec, supp, const) -> truth table."""
    vec, supp, const = spec
    t = 0
    for i in range(NM):
        if vec >> i & 1:
            m = MONOS[i]
            t ^= sum(1 << x for x in range(16) if (x & m) == m)
    for j in supp:
        t ^= XF[j]
    if const:
        t ^= FULL
    return t


def spec_of(fn):
    """truth table -> (deg2 vec, supp tuple, const)."""
    masks = anf_masks(fn)
    vec = 0
    for m in masks:
        if bin(m).count("1") >= 2 and m in IDX:
            vec |= 1 << IDX[m]
    supp = tuple(j for j in range(4) if (1 << j) in masks)
    const = 0 in masks
    return (vec, supp, const)


def and_spec(a, b):
    """AND of two function specs -> spec of product (exhaustive TT)."""
    fa, fb = fn_of(a), fn_of(b)
    fp = fa & fb
    return spec_of(fp)


_shape_cache = {}


def shape_specs(spec):
    """All affine/const shapings of a wire spec (same vec, any supp/const)."""
    key = spec[0]
    hit = _shape_cache.get(key)
    if hit is not None:
        return hit
    vec = spec[0]
    out = []
    for r in range(5):
        for supp in combinations(range(4), r):
            for c in range(2):
                out.append((vec, supp, c))
    _shape_cache[key] = out
    return out


def search(max_ops=5, verbose=False):
    memo = set()
    nodes = [0]
    path = []

    def missing_rank(basis, inj):
        """Number of independent deg>=2 directions missing to cover TGT
        (injections pre-cancel their bits)."""
        nb = list(basis)
        n0 = len(nb)
        for j in range(4):
            need = TGT[j]
            for (jj, v) in inj:
                if jj == j:
                    need ^= v
            if need and not contains(rref(nb), need):
                nb = list(rref(nb + [need]))
        return len(nb) - n0

    def dfs(tspec, uspec, basis, inj, depth):
        # tspec/uspec: (vec, supp, const) of the two ancilla wires
        # basis: rref tuple over NM coords of deg>=2 parts ever produced
        # inj: tuple of (bit, vec) injections into h (deg>=2 vecs only;
        #      deg<=1 residuals handled by the final linear assembly)
        nodes[0] += 1
        if verbose and nodes[0] % 1000000 == 0:
            print(f"  nodes={nodes[0]} depth={depth}", file=sys.stderr)
        if depth == max_ops:
            for j in range(4):
                need = TGT[j]
                for (jj, v) in inj:
                    if jj == j:
                        need ^= v
                if need and not contains(basis, need):
                    return False
            return True
        # rank prune: each remaining op adds <= 1 new deg>=2 direction
        if missing_rank(basis, inj) > max_ops - depth:
            return False
        key = (tspec, uspec, basis, inj, depth)
        if key in memo:
            return False
        memo.add(key)
        bases = [(0, (i,), 0) for i in range(4)] + [tspec, uspec]
        seen_pairs = set()
        for bi in range(6):
            for bj in range(bi + 1, 6):
                a0, b0 = bases[bi], bases[bj]
                pk = (a0, b0)
                if pk in seen_pairs:
                    continue
                seen_pairs.add(pk)
                for a in shape_specs(a0):
                    for b in shape_specs(b0):
                        pspec = and_spec(a, b)
                        pvec = pspec[0]
                        if not pvec:
                            continue
                        for tgt in ("t", "u", 0, 1, 2, 3):
                            if tgt == "t" and 4 in (bi, bj):
                                continue  # ccx(_, t, t) aliasing
                            if tgt == "u" and 5 in (bi, bj):
                                continue  # ccx(_, u, u) aliasing
                            nt, nu, ninj = tspec, uspec, inj
                            if tgt == "t":
                                nt = spec_of(fn_of(tspec) ^ fn_of(pspec))
                            elif tgt == "u":
                                nu = spec_of(fn_of(uspec) ^ fn_of(pspec))
                            else:
                                ninj = tuple(sorted(inj + ((tgt, pvec),)))
                            nb = basis if contains(basis, pvec) \
                                else rref(list(basis) + [pvec])
                            if dfs(nt, nu, nb, ninj, depth + 1):
                                path.append((a0, b0, a, b, pspec, tgt))
                                return True
        return False

    zero = (0, (), 0)
    ok = dfs(zero, zero, tuple(), tuple(), 0)
    if ok:
        path.reverse()
        return path, nodes[0]
    return None, nodes[0]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    res, n = search(verbose=verbose)
    if res:
        print(f"FOUND after {n} nodes")
        for k, (a0, b0, a, b, pspec, tgt) in enumerate(res):
            ms = [MONOS[i] for i in range(NM) if pspec[0] >> i & 1]
            print(f"  op{k+1}: shape({a0[0]:04x}|{a0[1]}|{a0[2]}, "
                  f"{b0[0]:04x}|{b0[1]}|{b0[2]}) -> {tgt}: prod {ms}")
    else:
        print(f"NOT FOUND after {n} nodes")
