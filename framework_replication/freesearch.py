#!/usr/bin/env python3
"""Final-model 5-Toffoli F24 search with MBU-freeable ancillas.

Model (the paper's, made precise):
  - 5 ccx ops; operands = shaped (affine-in-x + const) forms of the wires
    {x0..x3, t, u} (transient shaping, restored); targets in
    {t, u, h0, h3}.
  - free snapshot moves: cx(w -> h[b]) for w in {t, u}, any subset, any
    time.
  - final affine assembly per h bit (cx from x, x gates).
  - ancilla freeing (MBU): an ancilla is freeable iff its value at free
    time is (a) 0 (free directly), or (b) equal to A & B for two affine
    x-only forms A, B (then hmr + cz(A-wires, cond=m) + free; the fixup
    is exact for arbitrary h).
  - accept: for each bit b, TGT[b] ^ h[b] must be affine after snapshots,
    i.e. D2(TGT[b]) in span of D2 of all snapshot-available values.

Search state: (t, u, h-injections...) — h injections are the products
targeted at h bits directly (exact truth tables). Snapshots deferred:
accept via D2-span over {boundary t/u values} + {h-injected products}.
"""
import sys
from itertools import combinations

XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF
INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]
MONOS = [m for m in range(16) if bin(m).count("1") >= 2]
NM = len(MONOS)


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


# all affine x-only forms (for MBU fixup partners)
AFF = sorted({s for i in range(4) for s in shapes(XF[i])})
AND2 = {}  # value -> (A, B) decomposition as AND of two affine forms
for i in range(len(AFF)):
    for j in range(i + 1, len(AFF)):
        v = AFF[i] & AFF[j]
        if v and v not in AND2:
            AND2[v] = (AFF[i], AFF[j])


def products_from(t, u):
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


def freeable(w):
    """Ancilla value freeable via MBU: 0, or = A&B of two affine forms."""
    return w == 0 or w in AND2


def solve(max_ops=5):
    memo = set()
    path = []

    def dfs(t, u, hinj, depth):
        # hinj: tuple of 4 truth tables injected into h bits directly
        if depth == max_ops:
            if not (freeable(t) and freeable(u)):
                return False
            vals = list(hinj)
            # snapshot-available: boundary t/u values along the realized
            # path are accumulated in `bounds`; reconstruct via path? For
            # correctness accumulate during recursion instead.
            return None  # replaced below
        raise NotImplementedError

    # cleaner: accumulate boundary values during recursion
    def dfs2(t, u, hinj, bounds, depth):
        if depth == max_ops:
            if not (freeable(t) and freeable(u)):
                return False
            vals = list(hinj) + [v for pair in bounds for v in pair]
            basis = rref([d2(v) for v in vals if v])
            for b in range(4):
                # residual needed: D2(TGT[b]) ^ D2(hinj[b]) in span
                need = TGT_D2[b] ^ d2(hinj[b])
                if need and not contains(basis, need):
                    return False
            return True
        key = (t, u, hinj, depth)
        if key in memo:
            return False
        memo.add(key)
        # prune: missing D2 rank vs remaining ops
        vals = list(hinj) + [v for pair in bounds for v in pair]
        basis = rref([d2(v) for v in vals if v])
        nb = list(basis)
        miss = 0
        for b in range(4):
            need = TGT_D2[b] ^ d2(hinj[b])
            if need and not contains(rref(nb), need):
                nb.append(need)
                nb = list(rref(nb))
                miss += 1
        if miss > max_ops - depth:
            return False
        for p in prods(t, u):
            for tgt in ("t", "u", 0, 1, 2, 3):
                if tgt == "t" and not (t or True):
                    continue
                nt, nu, ninj = t, u, list(hinj)
                if tgt == "t":
                    nt = t ^ p
                elif tgt == "u":
                    nu = u ^ p
                else:
                    ninj[tgt] ^= p
                if dfs2(nt, nu, tuple(ninj),
                        bounds + ((nt, nu),), depth + 1):
                    path.append((t, u, p, tgt))
                    return True
        return False

    ok = dfs2(0, 0, (0, 0, 0, 0), ((0, 0),), 0)
    return (path[::-1] if ok else None)


def verify(path):
    t, u = 0, 0
    hinj = [0, 0, 0, 0]
    bounds = [(0, 0)]
    for k, (pt, pu, p, tgt) in enumerate(path):
        assert (pt, pu) == (t, u), f"op{k+1} state mismatch"
        assert p in prods(t, u), f"op{k+1} product not realizable"
        if tgt == "t":
            t ^= p
        elif tgt == "u":
            u ^= p
        else:
            hinj[tgt] ^= p
        bounds.append((t, u))
    assert freeable(t) and freeable(u), "ancillas not freeable"
    vals = list(hinj) + [v for pair in bounds for v in pair]
    basis = rref([d2(v) for v in vals if v])
    for b in range(4):
        need = TGT_D2[b] ^ d2(hinj[b])
        assert not need or contains(basis, need), f"bit {b} not coverable"
    return bounds, hinj, t, u


if __name__ == "__main__":
    path = solve()
    if path is None:
        print("NOT FOUND")
        sys.exit(1)
    print(f"FOUND: {len(path)} ops")
    for k, (t, u, p, tgt) in enumerate(path):
        print(f"  op{k+1}: (t={t:04x}, u={u:04x}) ^ p={p:04x} -> {tgt}")
    bounds, hinj, t, u = verify(path)
    print("verified")
    print("h injections:", [f'{v:04x}' for v in hinj])
    print("final t:", f"{t:04x}", "freeable via",
          "zero" if t == 0 else f"AND2 {AND2.get(t)}")
    print("final u:", f"{u:04x}", "freeable via",
          "zero" if u == 0 else f"AND2 {AND2.get(u)}")
