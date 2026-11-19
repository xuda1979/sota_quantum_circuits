#!/usr/bin/env python3
"""Physical 5-Toffoli F24 search, restricted operand model (paper-faithful).

Operand wires: x0..x3, t, u (each with transient affine/const shaping).
Target wires: t, u, h0..h3. h tracked as exact truth tables (products
targeting h XOR in directly). Accept: h == targets exactly at depth 5.

This is the paper's Extended Data Fig. 1 model: five Toffoli products
computed from x-shaped operands and ancilla wires, results XORed into h
(the transcript's b.cx(t, h[0]) snapshot is one such XOR; here modeled
exactly by letting ccx target h directly or t/u then snapshot).

Also tracks a snapshot AFTER each op (cx(t->h[b]) or cx(u->h[b]) free
moves) to capture the paper's cx(t, h[0]) idiom.
"""
import sys
from itertools import combinations

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF
TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]

_shape_cache = {}


def shapes(base):
    hit = _shape_cache.get(base)
    if hit is not None:
        return hit
    out = []
    seen = set()
    for r in range(5):
        for supp in combinations(range(4), r):
            for c in range(2):
                v = base
                for j in supp:
                    v ^= XF[j]
                v ^= FULL if c else 0
                if v not in seen:
                    seen.add(v)
                    out.append(v)
    _shape_cache[base] = out
    return out


def search(max_ops=5, verbose=False):
    memo = set()
    nodes = [0]
    path = []

    def dfs(t, u, h, depth):
        nodes[0] += 1
        if verbose and nodes[0] % 1000000 == 0:
            print(f"  nodes={nodes[0]} depth={depth}", file=sys.stderr)
        if depth == max_ops:
            return list(h) == TGT
        key = (t, u, h, depth)
        if key in memo:
            return False
        memo.add(key)
        bases = [XF[0], XF[1], XF[2], XF[3], t, u]
        for bi in range(6):
            for bj in range(bi + 1, 6):
                a0, b0 = bases[bi], bases[bj]
                for a in shapes(a0):
                    for b in shapes(b0):
                        p = a & b
                        if not p:
                            continue
                        # targets: t, u (no aliasing), h0..h3
                        for tgt in ("t", "u", 0, 1, 2, 3):
                            if tgt == "t" and bi >= 4:
                                continue
                            if tgt == "u" and bj >= 4:
                                # careful: bi<bj; t is index 4, u index 5
                                pass
                            if tgt == "t" and 4 in (bi, bj):
                                continue
                            if tgt == "u" and 5 in (bi, bj):
                                continue
                            nt, nu, nh = t, u, list(h)
                            if tgt == "t":
                                nt = t ^ p
                            elif tgt == "u":
                                nu = u ^ p
                            else:
                                nh[tgt] ^= p
                            # free snapshot move: cx(t->h[b]) or cx(u->h[b])
                            # for any b (captures the paper's cx(t, h[0]))
                            snaps = []
                            for b in range(4):
                                snaps.append(("t", nt, b))
                                snaps.append(("u", nu, b))
                            if dfs(nt, nu, tuple(nh), depth + 1):
                                path.append((a0, b0, a, b, p, tgt, None))
                                return True
                            for (w, val, b) in snaps:
                                nh2 = list(nh)
                                nh2[b] ^= val
                                if dfs(nt, nu, tuple(nh2), depth + 1):
                                    path.append((a0, b0, a, b, p, tgt, (w, b)))
                                    return True
        return False

    ok = dfs(0, 0, (0, 0, 0, 0), 0)
    if ok:
        path.reverse()
        return path, nodes[0]
    return None, nodes[0]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    res, n = search(verbose=verbose)
    if res:
        print(f"FOUND after {n} nodes")
        for k, (a0, b0, a, b, p, tgt, snap) in enumerate(res):
            s = f"snapshot {snap[0]}->h[{snap[1]}]" if snap else "no snapshot"
            print(f"  op{k+1}: shape({a0:04x},{b0:04x}) shaped({a:04x},"
                  f"{b:04x}) prod={p:04x} -> {tgt}; {s}")
    else:
        print(f"NOT FOUND after {n} nodes")
