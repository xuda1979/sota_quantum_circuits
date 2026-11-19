#!/usr/bin/env python3
"""Definitive search for the 5-Toffoli F24 inversion circuit (Q=10, 2 ancillas).

Model: 5 paid ops (ccx), each = AND of two wire values, target in {t, u, h0..h3}.
Wire values: x0..x3 (immutable), t, u (accumulating: value = XOR of products
landed + free affine shaping). Free ops: cx/x (affine shaping, snapshots into h).
Goal: for each bit j, n_j ^ inj_j in span(all raw product pis, x, 1).

Child pi generator (complete): a = sum(alpha_i * p_i for live i) ^ l_a ^ c_a,
b = sum(beta_i * p_i) ^ l_b ^ c_b; child pi =
  sum_{i,i'} alpha_i beta_i' pq(pi_i, pi_i')   [i=i' term = pi_i, idempotent]
^ sum_i alpha_i ext(pi_i, supp(l_b)) ^ sum_i' beta_i' ext(pi_i', supp(l_a))
^ bip(supp(l_a), supp(l_b))
^ (c_b * sum_i alpha_i pi_i) ^ (c_a * sum_i' beta_i' pi_i')   [const crosses]

Run: python3 search5.py [worker_id n_workers]
"""
import sys
from itertools import product as iproduct

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
MONOS = [3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]  # degree>=2 masks
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)
FULL = (1 << NM) - 1


def anf_vec(fn):
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    r = 0
    for m in range(1, 16):
        if f[m] and m in IDX:
            r |= 1 << IDX[m]
    return r


TGT = []
for b in range(4):
    fn = sum(((INV[x] >> b) & 1) << x for x in range(16))
    TGT.append(anf_vec(fn))


def ext(pi, supp):
    """pi of p * l where l = XOR of x_j, j in supp (per monomial, idempotent)."""
    r = 0
    for i in range(NM):
        if pi >> i & 1:
            m = MONOS[i]
            acc = 1 << i  # empty product: the monomial itself contributes once...
            # p*l = XOR_{j in supp} (p * x_j): each monomial m -> m|{j} (collapse if j in m)
            for j in supp:
                m2 = m if (m >> j) & 1 else m | (1 << j)
                acc ^= 1 << IDX[m2]
            r ^= acc
    return r


def pq(pi1, pi2):
    """pi of p*q: pairwise monomial unions (with collapse); i=j term NOT here."""
    r = 0
    for i in range(NM):
        if not (pi1 >> i & 1):
            continue
        m1 = MONOS[i]
        for k in range(NM):
            if not (pi2 >> k & 1):
                continue
            mu = m1 | MONOS[k]
            if mu in IDX:
                r ^= 1 << IDX[mu]
    return r


def bip(s1, s2):
    r = 0
    for i in s1:
        for j in s2:
            if i == j:
                continue
            r ^= 1 << IDX[(1 << i) | (1 << j)]
    return r


SUPS = [tuple(i for i in range(4) if s >> i & 1) for s in range(16)]
BITS = [(a, b, ca, cb) for a in range(2) for b in range(2) for ca in range(2) for cb in range(2)]


def children(pt, pu):
    """All raw-product pis from one ccx with live products pt, pu."""
    out = set()
    P = [pt, pu]
    # operand product-parts: (alpha_t, alpha_u) in {0,1}^2 (not both 0 handled via affines)
    for at, au, bt, bu in iproduct(range(2), repeat=4):
        pa = (at * pt) ^ (au * pu)
        pb = (bt * pt) ^ (bu * pu)
        # cross terms p*q (only pairs where both parts nonzero in that operand)
        for sa, sb, ca, cb in iproduct(SUPS, SUPS, range(2), range(2)):
            v = 0
            # p_a * p_b terms: (at*t+au*u)*(bt*t+bu*u) = at*bt*t + au*bu*u + (at*bu^au*bt)*t*u
            if at and bt:
                v ^= pt
            if au and bu:
                v ^= pu
            if (at and bu) ^ (au and bt):
                v ^= pq(pt, pu)
            # p_a * l_b and l_a * p_b
            if pa:
                v ^= ext(pa, sb)
            if pb:
                v ^= ext(pb, sa)
            # l_a * l_b
            if sa and sb:
                v ^= bip(sa, sb)
            # const crosses: a*c_b adds p_a; c_a*b adds p_b
            if cb and pa:
                v ^= pa
            if ca and pb:
                v ^= pb
            if v:
                out.add(v)
    return out


def rref(vecs):
    basis = []
    for v in vecs:
        if not v:
            continue
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return basis


def contains(basis, v):
    for b in basis:
        v = min(v, v ^ b)
    return v == 0


def solve():
    # DFS over 5 ops. State: (pt, pu, all-pis frozenset, inj tuple, depth).
    # Snapshots: every t/u value ever = accumulation; we conservatively
    # require span over raw product pis only + injections per bit.
    memo = set()
    order = []

    def dfs(pt, pu, prods, inj, depth, path):
        if depth == 5:
            basis = rref(list(prods))
            for j in range(4):
                need = TGT[j]
                for (jj, v) in inj:
                    if jj == j:
                        need ^= v
                # affine part of targets is free (x, 1 wires): the vec
                # representation already excludes degree<2, so need must be
                # fully in span of prods.
                if need and not contains(basis, need):
                    return False
            order.append(list(path))
            return True
        key = (pt, pu, prods, inj, depth)
        if key in memo:
            return False
        memo.add(key)
        for v in sorted(children(pt, pu), reverse=True):
            for tgt in ("t", "u", 0, 1, 2, 3):
                npt, npu, ninj = pt, pu, inj
                if tgt == "t":
                    npt = pt ^ v
                elif tgt == "u":
                    npu = pu ^ v
                else:
                    ninj = tuple(sorted(inj + ((tgt, v),)))
                nprods = frozenset(prods | {v})
                if dfs(npt, npu, nprods, ninj, depth + 1, path + [(v, tgt)]):
                    return True
        return False

    ok = dfs(0, 0, frozenset(), tuple(), 0, [])
    if ok:
        print("FOUND")
        for step in order[0]:
            v, tgt = step
            ms = [MONOS[i] for i in range(NM) if v >> i & 1]
            print(f"  op -> {tgt}: pi {sorted(ms)}")
    else:
        print("NOT FOUND", "memo:", len(memo))


if __name__ == "__main__":
    solve()
