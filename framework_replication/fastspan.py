#!/usr/bin/env python3
"""Fast exact search for the 5-product F24 inversion (span formulation).

State: S = frozenset closure {T, U, T^U} of the two ancilla wire values
(free shaping interconverts the three representations), basis = rref tuple
of all op vectors so far, depth.

Ops (ccx, 1 Toffoli): operand pairs (acc-parts) from the current wires
(A, B) with {A,B,A^B} = S:
    (0,0): bipartite quads (two x-wires)
    (A,0) / (B,0): acc-wire & x-wire  [x-wires hold x_j exactly]
    (A,B): both acc-wires (with riders/consts)
Product vector v = pq(a,b) ^ ext(a,sb) ^ ext(b,sa) ^ bip(sa,sb) ^ [a]^b...
Target: t -> new wires (A^v, B); u -> (A, B^v); spend (h bit) -> (A, B).
All op vectors join the basis (acc accumulations are snapshot-able; spent
products are injected directly).

Accept at depth 5: all four target ANF vectors in span(basis).

Usage: python3 fastspan.py <op1_index>   (0..34)
"""
import sys

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
MONOS = [3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)

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


def vec(masks):
    r = 0
    for m in masks:
        r |= 1 << IDX[m]
    return r


BIP = sorted({vec([(1 << i) | (1 << j) for i in A for j in B if i != j])
              for abits in range(1, 16) for bbits in range(1, 16)
              for A in [tuple(k for k in range(4) if abits >> k & 1)]
              for B in [tuple(k for k in range(4) if bbits >> k & 1)]
              if any(i != j for i in A for j in B)})

SUPPS = [tuple(k for k in range(4) if s >> k & 1) for s in range(16)]


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


def bip(sa, sb):
    r = 0
    for i in sa:
        for j in sb:
            if i != j:
                r ^= 1 << IDX[(1 << i) | (1 << j)]
    return r


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


_coset_cache = {}


def cosets(a, b):
    key = (a, b)
    hit = _coset_cache.get(key)
    if hit is not None:
        return hit
    out = set()
    if a == 0 and b == 0:
        out.update(BIP)
    elif a == 0 or b == 0:
        c = a or b
        for j in range(4):
            e = ext(c, (j,))
            for s in range(8):
                supp = tuple(k for k in range(4) if (s >> k & 1) and k != j)
                out.add(e ^ bip(supp, (j,)))
                out.add(e ^ bip(supp, (j,)) ^ c)  # const rider on acc side
    else:
        base = pq(a, b)
        ea = {ext(a, s) for s in SUPPS}
        eb = {ext(b, s) for s in SUPPS}
        for va in ea:
            for vb in eb:
                x = base ^ va ^ vb
                out.update((x, x ^ a, x ^ b, x ^ a ^ b))
        # riders only on supports (bip terms with both nonempty supp)
        for sa in SUPPS:
            for sb in SUPPS:
                if sa and sb:
                    out.add(base ^ ext(a, sb) ^ ext(b, sa) ^ bip(sa, sb))
                    out.add(base ^ ext(a, sb) ^ ext(b, sa) ^ bip(sa, sb) ^ a)
                    out.add(base ^ ext(a, sb) ^ ext(b, sa) ^ bip(sa, sb) ^ b)
    out.discard(0)
    out = frozenset(out)
    _coset_cache[key] = out
    return out


def closure(t, u):
    return frozenset({t, u, t ^ u})


def search(op1, deadline=None):
    """DFS from op1 vector; returns list of 5 op vectors or None."""
    import time
    memo = set()
    path = []

    def rank_needed(basis):
        # number of independent directions missing to cover all TGT
        nb = list(basis)
        for t in TGT:
            v = t
            for bb in rref(nb):
                v = min(v, v ^ bb)
            if v:
                nb.append(v)
                nb = list(rref(nb))
        extra = len(nb) - len(basis)
        return extra

    def dfs(S, basis, depth):
        if depth == 5:
            return all(contains(basis, t) for t in TGT)
        need = rank_needed(basis)
        if need > 5 - depth:
            return False
        key = (S, basis, depth)
        if key in memo:
            return False
        memo.add(key)
        A, B = sorted(S)[:2] if len(S) >= 2 else (next(iter(S)), 0)
        # canonical two wire values: pick any two distinct from closure
        wires = sorted(S)
        pairs = set()
        if len(wires) == 1:
            w = wires[0]
            pairs.add((0, 0))
            if w:
                pairs.add((w, 0))
        else:
            pairs.add((0, 0))
            for w in wires:
                if w:
                    pairs.add((w, 0))
            for i in range(len(wires)):
                for j in range(i + 1, len(wires)):
                    pairs.add((wires[i], wires[j]))
        cands = set()
        for (a, b) in pairs:
            cands |= cosets(a, b)
        # try rank-adding first
        def adds(v):
            return not contains(basis, v)
        for v in sorted(cands, key=adds, reverse=True):
            for tgt in ("t", "u", "s"):
                if tgt == "s":
                    nS, nb = S, basis if contains(basis, v) else rref(list(basis) + [v])
                else:
                    # target the wire holding A or B: both wire choices; use
                    # the (A,B) actually chosen... simplify: new closure of
                    # (A^v, B) and (A, B^v) for the current representation
                    nS1 = closure(A, B) if A == 0 and B == 0 else None
                    nS = frozenset()
                    for (X, Y) in ((A, B), (B, A)):
                        nS |= closure(X ^ v, Y)
                    nb = basis if contains(basis, v) else rref(list(basis) + [v])
                if dfs(nS, nb, depth + 1):
                    path.append((v, tgt))
                    return True
        return False

    v1 = BIP[op1]
    S0 = closure(v1, 0)
    b0 = rref([v1])
    if dfs(S0, b0, 1):
        path.append((v1, "seed"))
        path.reverse()
        return path
    return None


def main():
    i = int(sys.argv[1])
    res = search(i)
    if res:
        print(f"op1 #{i}: FOUND")
        for k, (v, tgt) in enumerate(res):
            ms = [MONOS[j] for j in range(NM) if v >> j & 1]
            print(f"  p{k+1} -> {tgt}: {sorted(ms)}")
    else:
        print(f"op1 #{i}: no solution")


if __name__ == "__main__":
    main()
