#!/usr/bin/env python3
"""Exact search over the PAPER'S OWN STRUCTURE (Ext. Data Fig. 1c):
five ccx products, ALL targeting t (operands = shaped {x0..x3, t, u});
free snapshots cx(t, h[b]) and cx(u, h[b]) at every op boundary; final
h[b] must equal the target bits exactly (deg<=1 residual free via final
cx from x).

State: (t, u, h0..h3) truth tables. Snapshots: any subset of
{t->h[b], u->h[b]} after each op. Free linear assembly at the end:
h[b] ^= XOR of x_i + const.

This is a restricted, paper-faithful subset of xeqsearch.
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


def lin_span_ok(h, tgt):
    """tgt ^ h must be affine in x (XOR of x_i + const)."""
    r = tgt ^ h
    return r == 0 or r == FULL or any(r == xf for xf in XF) or \
        any(r == xf ^ FULL for xf in XF) or \
        any(r == xf1 ^ xf2 for i, xf1 in enumerate(XF)
            for xf2 in XF[i + 1:]) or \
        any(r == xf1 ^ xf2 ^ FULL for i, xf1 in enumerate(XF)
            for xf2 in XF[i + 1:]) or \
        any(r == xf1 ^ xf2 ^ xf3
            for i, xf1 in enumerate(XF) for j, xf2 in enumerate(XF[i + 1:])
            for xf3 in XF[i + j + 2:]) or \
        any(r == xf1 ^ xf2 ^ xf3 ^ FULL
            for i, xf1 in enumerate(XF) for j, xf2 in enumerate(XF[i + 1:])
            for xf3 in XF[i + j + 2:])


ALL_AFFINE = set()
for r in range(5):
    for supp in combinations(range(4), r):
        for c in range(2):
            ALL_AFFINE.add(sum(XF[j] for j in supp) ^ (FULL if c else 0))


def search(max_ops=5, verbose=False):
    memo = set()
    nodes = [0]
    path = []

    def snap_moves(t, u, h):
        """All (nh-tuple, move-desc) from free snapshots."""
        out = [(h, ())]
        for b in range(4):
            if t:
                nh = list(h)
                nh[b] ^= t
                out.append((tuple(nh), (("t", b),)))
                if u:
                    nh2 = list(nh)
                    nh2[b] ^= u
                    out.append((tuple(nh2), (("t", b), ("u", b))))
            if u:
                nh = list(h)
                nh[b] ^= u
                out.append((tuple(nh), (("u", b),)))
        return out

    def dfs(t, u, h, depth):
        nodes[0] += 1
        if verbose and nodes[0] % 1000000 == 0:
            print(f"  nodes={nodes[0]} depth={depth}", file=sys.stderr)
        if depth == max_ops:
            # final free linear assembly: each h[b] needs affine residual
            for b in range(4):
                if (TGT[b] ^ h[b]) not in ALL_AFFINE:
                    return False
            return True
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
                        # paper structure: target t only (u kept 0)
                        if 4 in (bi, bj):
                            continue  # ccx(_, t, t) aliasing
                        nt = t ^ p
                        for nh, snaps in snap_moves(nt, u, h):
                            if dfs(nt, u, nh, depth + 1):
                                path.append((a0, b0, a, b, p, "t", snaps))
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
        for k, (a0, b0, a, b, p, tgt, snaps) in enumerate(res):
            print(f"  op{k+1}: shape({a0:04x},{b0:04x}) shaped({a:04x},"
                  f"{b:04x}) prod={p:04x} -> {tgt}; snaps={snaps}")
    else:
        print(f"NOT FOUND after {n} nodes")
