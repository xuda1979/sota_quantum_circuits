#!/usr/bin/env python3
"""Parallel exact search for a realizable 5-op F24 inversion circuit.

Worker w: enumerates op1 (first raw product) choices w, w+J, w+2J, ... from
the 35 bipartite quads, then DFS over ops 2..5 with:
  - state: accum_t, accum_u (spans of raw products landed on each ancilla,
    as pi vectors), set of raw-product pis (P), per-bit injections.
  - transitions: op k: raw product v in cosets(a, b) for a in span(P)+{x},
    b in span(P)+{x} (x = affine 0); target t/u (accum ^= v, P |= {v}) or
    h_j (injection).
  - accept at depth 5: for each j: n_j ^ inj_j in span(P ∪ {x, 1}).

Uses the reduction: snapshots make span-of-accumulations = span(P), and
targets' n_j include affine parts (handled by free x-wires).

Usage: python3 seqsearch.py <worker> <n_workers>
"""
import sys

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
MONOS = [3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)
FULLMASK = (1 << NM) - 1
TGT = []
for _b in range(4):
    fn = sum(((INV[x] >> _b) & 1) << x for x in range(16))
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
SUPPS = [tuple(i for i in range(4) if s >> i & 1) for s in range(16)]


def vec(masks):
    r = 0
    for m in masks:
        r |= 1 << IDX[m]
    return r


def ext(a, supp):
    r = 0
    for i in range(NM):
        if a >> i & 1:
            m = MONOS[i]
            for j in supp:
                m2 = m if (m >> j) & 1 else m | (1 << j)
                r ^= 1 << IDX[m2]
    return r


def pq(a, b):
    r = 0
    for i in range(NM):
        if not (a >> i & 1):
            continue
        m1 = MONOS[i]
        for k in range(NM):
            if not (b >> k & 1):
                continue
            mu = m1 | MONOS[k]
            if mu in IDX:
                r ^= 1 << IDX[mu]
    return r


# 35 bipartite quads
BIP = set()
for abits in range(1, 16):
    for bbits in range(1, 16):
        A = tuple(i for i in range(4) if abits >> i & 1)
        B = tuple(i for i in range(4) if bbits >> i & 1)
        masks = frozenset((1 << i) | (1 << j) for i in A for j in B if i != j)
        if masks:
            BIP.add(vec(masks))
BIP = sorted(BIP)


def cosets(a, b):
    """All raw-product pis from operands with product-parts a, b."""
    out = set()
    base = pq(a, b)
    ea = {ext(a, s) for s in SUPPS}
    eb = {ext(b, s) for s in SUPPS}
    for va in ea:
        for vb in eb:
            x = base ^ va ^ vb
            out.update((x, x ^ a, x ^ b, x ^ a ^ b))
    return out


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
    return basis


def contains(basis, v):
    for bb in basis:
        v = min(v, v ^ bb)
    return v == 0


def spansub(P, extra):
    """span of P (list) + extra vecs; returns basis."""
    return rref(list(P) + list(extra))


def solve(op1_list, tag=""):
    memo = set()
    found = []

    def dfs(at, au, P, inj, depth):
        if depth == 5:
            basis = rref(list(P))
            for j in range(4):
                need = TGT[j]
                for (jj, v) in inj:
                    if jj == j:
                        need ^= v
                if need and not contains(basis, need):
                    return False
            found.append((list(P), inj))
            return True
        key = (at, au, tuple(sorted(P)), tuple(sorted(inj)), depth)
        if key in memo:
            return False
        memo.add(key)
        # operand product-parts: 0, at, au, at^au
        parts = sorted({0, at, au, at ^ au})
        cand = set()
        for a in parts:
            for b in parts:
                if a == 0 and b == 0:
                    continue
                cand |= cosets(a, b)
        # op1 (if depth == 0 restricted outside); all candidates:
        for v in sorted(cand):
            for tgt in ("t", "u", 0, 1, 2, 3):
                nat, nau, ninj = at, au, inj
                if tgt == "t":
                    nat = at ^ v
                elif tgt == "u":
                    nau = au ^ v
                else:
                    ninj = tuple(sorted(inj + ((tgt, v),)))
                nP = tuple(sorted(set(P) | {v}))
                if dfs(nat, nau, nP, ninj, depth + 1):
                    return True
        return False

    for i, op1 in enumerate(op1_list):
        for tgt in ("t", "u"):
            at = op1 if tgt == "t" else 0
            au = op1 if tgt == "u" else 0
            P = (op1,)
            if dfs(at, au, P, tuple(), 1):
                P, inj = found[0]
                print(f"FOUND {tag} op1=#{i}")
                print("P:", [hex(p) for p in P])
                print("inj:", [(j, hex(v)) for (j, v) in inj])
                return True
    return False


def main():
    w = int(sys.argv[1])
    nw = int(sys.argv[2])
    mine = BIP[w::nw]
    ok = solve(mine, tag=f"w{w}")
    if not ok:
        print(f"w{w}: no solution among {len(mine)} op1 choices")


if __name__ == "__main__":
    main()
