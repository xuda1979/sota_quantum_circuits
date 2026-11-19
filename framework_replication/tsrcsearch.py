#!/usr/bin/env python3
"""t-as-source model: the paper's literal circuit.

op1: ccx(X0*, X1*, t) -> t holds ONE product p_t = X0* & X1* (any shaped
     x-pair, any of the 6 x-wire pairs).
ops2-5: ccx(A*, B*, target) with A, B shaped wires from {x0..x3, t}
     (t as SOURCE keeps t intact), target in {h0..h3}.
snapshots: cx(t, h[b]) any subset (adds p_t to those bits).
final: affine assembly per bit.
free t: hmr(t) + cz(op1 operands' wires, cond=m) — exact since
     t = p_t = A1 & B1 (the op1 operand values).

Accept per bit: XOR(h-products into b) ^ (p_t if snap_b) leaves affine
residual vs TGT[b].

RESULT (run 2026-11-17): NOT FOUND (8s). The paper's literal structure
(one t-op + four h-ops + snapshots + affine assembly) admits no T=5,
Q=9 solution; see q9search.py for the definitive exhaustive one-ancilla
search (same negative result, complete for its model).
"""
import sys
from itertools import combinations, product as iproduct

sys.path.insert(0, ".")
from finalsearch import MENU, TGT_D2, d2, TGT, XF, FULL
from tsearch import shapes, rref, contains

# op1 product options: shaped x-pair products (= MENU)
# h-product options: shaped pairs from {x0..x3, t} where t = p_t:
#   x&x (MENU), x&t-shaped, (t-shaped)&(t-shaped) = t-shaped pair (same
#   wire twice? no — distinct wires only, so t&t impossible; x&t only).
def hmenu(pt):
    out = set()
    for p in MENU:
        out.add(p)
    if pt:
        for i in range(4):
            for a in shapes(XF[i]):
                for b in shapes(pt):
                    v = a & b
                    if v:
                        out.add(v)
    return sorted(out)


def solve():
    # enumerate pt (MENU), snap patterns, and 4 h-products with free bit
    # assignment (each op targets any h bit; multiple ops per bit allowed)
    # Factorize per bit: need_b = TGT_D2[b] ^ (snap_b ? d2(pt) : 0);
    # XOR of products assigned to bit b must have d2 == need_b... plus the
    # AFFINE residual check on full truth tables.
    # DP per (pt, snap): assignment of 4 products to 4 bits: per-bit XOR.
    # Search: choose counts (k0..k3) sum 4; per bit choose k_b products
    # from hmenu(pt) with XOR d2 = need_b; verify affine residual.
    for pt in MENU:
        hm = hmenu(pt)
        hd2 = sorted({d2(v) for v in hm})
        for snap in iproduct((0, 1), repeat=4):
            needs = [TGT_D2[b] ^ (d2(pt) if snap[b] else 0)
                     for b in range(4)]
            # quick reachability: XOR of up to k menu d2s
            reach = {0: {0}}
            for k in range(1, 5):
                reach[k] = {v ^ d for v in reach[k - 1] for d in hd2}
            for k0 in range(5):
                for k1 in range(5 - k0):
                    for k2 in range(5 - k0 - k1):
                        k3 = 4 - k0 - k1 - k2
                        ks = (k0, k1, k2, k3)
                        if any(needs[b] not in reach[ks[b]]
                               for b in range(4)):
                            continue
                        # concrete: DFS per bit for products XORing to a
                        # value with d2 == need_b AND full residual affine
                        def find_sum(k, target_d2, acc):
                            if k == 0:
                                if target_d2 == 0:
                                    yield acc
                                return
                            for v in hm:
                                yield from find_sum(k - 1,
                                                    target_d2 ^ d2(v),
                                                    acc + [v])
                        prods = []
                        ok = True
                        for b in range(4):
                            found_b = None
                            for cand in find_sum(ks[b], needs[b], []):
                                v = 0
                                for c in cand:
                                    v ^= c
                                if snap[b]:
                                    v ^= pt
                                if d2(TGT[b] ^ v) == 0:
                                    found_b = cand
                                    break
                            if found_b is None:
                                ok = False
                                break
                            prods.append(found_b)
                        if ok:
                            return pt, snap, prods
    return None


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = solve()
    if res is None:
        print(f"NOT FOUND ({time.time()-t0:.0f}s)")
        sys.exit(1)
    pt, snap, prods = res
    print(f"FOUND ({time.time()-t0:.0f}s): T=5, Q=9, TQ=45")
    print(f"  op1: ccx(*, *, t): p_t = {pt:04x}")
    print(f"  snapshots: bits {[b for b in range(4) if snap[b]]}")
    for b in range(4):
        for p in prods[b]:
            print(f"  ccx -> h[{b}]: product {p:04x}")
    # verify
    for b in range(4):
        v = 0
        for p in prods[b]:
            v ^= p
        if snap[b]:
            v ^= pt
        assert d2(TGT[b] ^ v) == 0
    print("verified: residuals affine")
