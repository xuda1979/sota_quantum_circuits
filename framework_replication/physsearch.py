#!/usr/bin/env python3
"""Exact 5-Toffoli F24 search with PHYSICAL wire tracking (Q=10: 2 ancillas).

Fixes the closure over-approximation in fastspan.py: operands may only be
the physical wires {x0..x3, t, u} with per-wire affine/const shaping —
never the virtual t^u (which would need a third ancilla).

State: (t, u) as concrete truth-table values (16-bit ints).
Ops: ccx(shape(a), shape(b), target), a,b in {x_i, t, u} distinct wires,
target in {t, u, h0..h3}; shaping = XOR of x-subset + const on each
operand wire (free cx/x gates), applied transiently (restored after).
Accept: all four target functions in span(all wire values ever held +
injections + x + 1) after 5 ops.

The shaping is transient: physically, shape t (free), ccx, unshape t
(free). This is exactly the paper's model.
"""
import sys
from itertools import combinations, product as iproduct

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
        for aff in combinations(range(4), r):
            for c in range(2):
                v = base
                for j in aff:
                    v ^= XF[j]
                if c:
                    v ^= FULL
                if v not in seen and v:
                    seen.add(v)
                    out.append(v)
    _shape_cache[base] = out
    return out


def span_contains(vals, t):
    rows = {}
    for v in vals:
        if not v:
            continue
        for b in range(15, -1, -1):
            if (v >> b) & 1:
                r = rows.get(b)
                if r is None:
                    rows[b] = v
                    break
                v ^= r
    for b in range(15, -1, -1):
        if (t >> b) & 1:
            r = rows.get(b)
            if r is None:
                return False
            t ^= r
    return t == 0


def search(max_ops=5, verbose=False):
    memo = set()
    nodes = [0]
    path = []

    def dfs(t, u, S, inj, depth):
        nodes[0] += 1
        if verbose and nodes[0] % 500000 == 0:
            print(f"  nodes={nodes[0]} depth={depth}", file=sys.stderr)
        if depth == max_ops:
            for j in range(4):
                need = TGT[j]
                for (jj, v) in inj:
                    if jj == j:
                        need ^= v
                if need and not span_contains(list(S) + XF + [FULL], need):
                    return False
            return True
        key = (t, u, S, inj, depth)
        if key in memo:
            return False
        memo.add(key)
        # operand wires: x0..x3 (fixed), t, u (current)
        bases = [XF[0], XF[1], XF[2], XF[3], t, u]
        for bi in range(6):
            for bj in range(bi + 1, 6):
                a0, b0 = bases[bi], bases[bj]
                # skip degenerate: same wire twice (distinct wires enforced
                # by indices), and skip t-vs-u when equal values
                for a in shapes(a0):
                    for b in shapes(b0):
                        p = a & b
                        if not p:
                            continue
                        for tgt in ("t", "u", 0, 1, 2, 3):
                            nt, nu, ninj = t, u, inj
                            if tgt == "t":
                                nt = t ^ p
                            elif tgt == "u":
                                nu = u ^ p
                            else:
                                ninj = tuple(sorted(inj + ((tgt, p),)))
                            nS = frozenset(set(S) | {nt, nu, p} - {0})
                            if dfs(nt, nu, nS, ninj, depth + 1):
                                path.append((a0, b0, a, b, p, tgt))
                                return True
        return False

    ok = dfs(0, 0, frozenset(), tuple(), 0)
    if ok:
        path.reverse()
        return path, nodes[0]
    return None, nodes[0]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    res, n = search(verbose=verbose)
    if res:
        print(f"FOUND after {n} nodes")
        for k, (a0, b0, a, b, p, tgt) in enumerate(res):
            print(f"  op{k+1}: base({a0:04x},{b0:04x}) shaped({a:04x},{b:04x})"
                  f" -> {tgt}: prod {p:04x}")
    else:
        print(f"NOT FOUND after {n} nodes")
