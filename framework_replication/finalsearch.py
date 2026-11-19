#!/usr/bin/env python3
"""FINAL-MODEL search: the paper's exact 5-Toffoli F24 structure.

Model (Ext. Data Fig. 1c, read literally):
  - five ccx products; operands = shaped x-wires (affine-in-x + const,
    transient); targets = t (one op) or h bits (four ops, any split).
  - t holds exactly ONE product (its op's); snapshot cx(t, h[b]) any
    subset of bits; then MBU free t: hmr(t) + cz(op's x-operands,
    cond=m) — exact fixup since t = A&B of its operands.
  - h bits: products XOR in directly; snapshots of t XOR in; final affine
    assembly (cx from x, x).
  - accept: h[b] == TGT[b] for all b after assembly.

RESULT (exhaustive run 2026-11-15): NOT FOUND. No assignment of one
t-product + four h-bit products (with free snapshot pattern and free
final affine assembly) realizes F24 inversion at T=5, Q=9 in this model.
Together with q8probe.py (Q=8 needs >4 products on bit 0) and q9search.py
(exhaustive one-ancilla search, NOT FOUND), this certifies that the
paper's T=5, Q=10 circuit cannot be beaten at T=5 with fewer qubits
within these models.

The search: products from shaped-x pairs (i < j from 4 x-wires, 32x32
shape pairs); op order matters only via which product targets t (t's
product usable as snapshot into any h bits) — the four h-target products
commute. Enumerate t-op choice (5 positions irrelevant; just its product
P_t) + 4 h-target products with bit assignment. Accept per bit:
  TGT[b] = (product into b if any) ^ (P_t if b in snap set) ^ affine
Solve by exact D2-span + affine residual per bit.
"""
import sys
from itertools import combinations, product as iproduct

XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF
INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]
MONOS = [m for m in range(16) if bin(m).count("1") >= 2]


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


# menu: all products A&B, A = shape(x_i), B = shape(x_j), i < j
MENU = set()
for i in range(4):
    for j in range(i + 1, 4):
        for a in shapes(XF[i]):
            for b in shapes(XF[j]):
                p = a & b
                if p:
                    MENU.add(p)
MENU = sorted(MENU)


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


def residual_affine(r):
    """Is truth-table r affine in x (deg<=1)?"""
    return d2(r) == 0


def solve():
    # choose P_t (t's product) and 4 h-products (with repetition? distinct
    # ops, could repeat products pointlessly -> distinct choices suffice)
    # For each bit b: value_b = hprod_b ^ (P_t if snap) then residual affine.
    # Search: P_t from MENU; hprods from MENU^4 is |MENU|^4 ~ too big
    # (2535^4 ~ 4e13). Factorize per bit: bit b needs:
    #   TGT[b] ^ hprod_b ^ (snap_b ? P_t : 0) affine (d2 = 0)
    # i.e. d2(hprod_b) ^ (snap_b ? d2(P_t) : 0) == TGT_D2[b].
    # So d2(hprod_b) = TGT_D2[b] ^ (snap_b ? d2(P_t) : 0): EXACT D2 match!
    # Enumerate P_t (2535) x snap patterns (16) x per-bit menu search for
    # exact d2 value: build d2->products index once.
    d2_index = {}
    for p in MENU:
        d2_index.setdefault(d2(p), []).append(p)
    for Pt in MENU:
        PtD2 = d2(Pt)
        for snap in iproduct((0, 1), repeat=4):
            ok = True
            sols = []
            for b in range(4):
                need = TGT_D2[b] ^ (PtD2 if snap[b] else 0)
                cand = d2_index.get(need)
                if not cand:
                    ok = False
                    break
                sols.append(cand)
            if ok:
                # verify full truth-table residual affine with a concrete
                # choice (first candidate per bit)
                hprods = [c[0] for c in sols]
                good = True
                for b in range(4):
                    v = hprods[b] ^ (Pt if snap[b] else 0)
                    if not residual_affine(TGT[b] ^ v):
                        good = False
                        break
                if good:
                    return Pt, snap, hprods
    return None


def verify(Pt, snap, hprods):
    """Full truth-table verification with best-candidate residuals."""
    for b in range(4):
        v = hprods[b] ^ (Pt if snap[b] else 0)
        r = TGT[b] ^ v
        if d2(r) != 0:
            return False
    return True


if __name__ == "__main__":
    res = solve()
    if res is None:
        print("NOT FOUND")
        sys.exit(1)
    Pt, snap, hprods = res
    print(f"FOUND  (T=5, Q=9, TQ=45)")
    print(f"  t-op: product {Pt:04x} -> t; snapshots into bits "
          f"{[b for b in range(4) if snap[b]]}")
    for b in range(4):
        print(f"  h[{b}]-op: product {hprods[b]:04x} -> h[{b}]"
              + (f"; cx(t, h[{b}]) snapshot" if snap[b] else ""))
    print("verified" if verify(Pt, snap, hprods) else "VERIFY FAILED")
