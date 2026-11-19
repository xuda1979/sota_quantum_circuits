#!/usr/bin/env python3
"""Generalized final-model search: 5 products, t = exactly one product
(MBU-freeable with its own operands' cz), remaining 4 products assigned
freely to h bits (multi-product bits allowed), snapshots cx(t -> h[b])
free, final affine assembly per bit.

Per bit b: XOR of products assigned to b, ^ (Pt if snap_b), must leave
affine residual: d2(XOR) = TGT_D2[b] ^ (snap_b ? d2(Pt) : 0).

Search space: Pt (menu) x assignment of 4 products to bits (5 options
each incl. t? NO — t takes exactly one: the other 4 go to h bits 0..3:
4^4 = 256 assignments) x snap pattern (16) x product choices. Factorize:
for each (Pt, snap, assignment), each bit b needs d2(XOR of its assigned
products) = need_b (exact). Bits with k_b products: XOR of k_b menu d2s
must equal need_b. Build per-k d2-sum reachability iteratively (DP over
products). Total products to place: 4.
"""
import sys
from itertools import product as iproduct

sys.path.insert(0, ".")
from finalsearch import MENU, TGT_D2, d2, TGT

d2_of = [d2(p) for p in MENU]
D2S = sorted(set(d2_of))

# reach[k] = set of d2 values = XOR of k menu-product d2s
reach = {0: {0}}
for k in range(1, 5):
    prev = reach[k - 1]
    cur = set()
    for v in prev:
        for d in D2S:
            cur.add(v ^ d)
    reach[k] = cur

import time
t0 = time.time()
found = None
L = len(MENU)
for i in range(L):
    Pt = MENU[i]
    PtD2 = d2_of[i]
    for snap in iproduct((0, 1), repeat=4):
        needs = [TGT_D2[b] ^ (PtD2 if snap[b] else 0) for b in range(4)]
        # assign 4 products to bits: counts (k0, k1, k2, k3) sum 4
        for k0 in range(5):
            for k1 in range(5 - k0):
                for k2 in range(5 - k0 - k1):
                    k3 = 4 - k0 - k1 - k2
                    ks = (k0, k1, k2, k3)
                    if any(needs[b] not in reach[ks[b]] for b in range(4)):
                        continue
                    # feasible at d2 level; now find concrete products
                    # (DP reconstruct per bit)
                    prods = []
                    ok = True
                    for b in range(4):
                        need = needs[b]
                        k = ks[b]
                        if k == 0:
                            if need != 0:
                                ok = False
                            continue
                        # find k menu indices XORing to need
                        sol = _find_sum = None
                        # simple DFS over D2S values
                        def dfs_sum(kk, target, acc):
                            if kk == 0:
                                return acc if target == 0 else None
                            for d in D2S:
                                r = dfs_sum(kk - 1, target ^ d, acc + [d])
                                if r is not None:
                                    return r
                            return None
                        sol = dfs_sum(k, need, [])
                        if sol is None:
                            ok = False
                            break
                        # map each d2 to a concrete menu product
                        idx_by_d2 = {}
                        for idx, dv in enumerate(d2_of):
                            idx_by_d2.setdefault(dv, idx)
                        prods.append([MENU[idx_by_d2[dv]] for dv in sol])
                    if ok:
                        found = (Pt, snap, ks, prods)
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    if found:
        break
    if i % 200 == 0:
        print(f"  i={i} ({time.time()-t0:.0f}s)", flush=True)

if found:
    Pt, snap, ks, prods = found
    print(f"FOUND ({time.time()-t0:.0f}s)")
    print(f"  t product: {Pt:04x}; snapshots into bits {[b for b in range(4) if snap[b]]}")
    for b in range(4):
        ps = prods[b] if b < len(prods) else []
        print(f"  h[{b}]: {ks[b]} product(s) {[f'{p:04x}' for p in ps]}")
    # verify d2 exactness
    for b in range(4):
        v = 0
        for p in (prods[b] if b < len(prods) else []):
            v ^= p
        if snap[b]:
            v ^= Pt
        r = TGT[b] ^ v
        assert d2(r) == 0, f"bit {b} residual not affine"
    print("verified: all residuals affine")
else:
    print(f"NOT FOUND ({time.time()-t0:.0f}s)")
