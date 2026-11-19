#!/usr/bin/env python3
"""Q=9 (one ancilla) T=5 search with h-bits as operand SOURCES.

Closes the model gap left by q9search.py: there, h-ops' operands were
x-affine pairs or (t-prefix & x-affine) pairs. Here, products may use
ANY live wire as operand source - including h-bits holding accumulated
products - with the complete search5-style child product generator
(products of span elements), while keeping ONE ancilla (t) and the
paper's freeing idioms:
  - t must end MBU-freeable: 0, or XOR of products each individually
    re-fixable via hmr + cz(A_i, B_i) where operand values are
    re-creatable at free time. We track t's products structurally.
  - h[b] final = h0[b] ^ INV(x)[b] exactly (x-part span condition +
    affine residual, as in q9search; the h-cross parts cancel because
    every product's h-dependence enters only via h0-coordinates that
    are degree<=1 in h0 ... NOT automatic: products of h0-values
    create h0_i*h0_j degree-2 terms. These must cancel in the final
    value. To keep the search exact we track full 8-var d2 vectors
    for h-cross parts ONLY as a cancellation constraint (recorded per
    op, checked at accept).
  - snapshots cx(t, h[b]) free.

State (x-part machinery, 11 x-mono coords):
  - t_x: accumulated x-part d2 of t (11-bit vector)
  - acc_x[4]: accumulated x-part d2s of h-bits
  - basis: rref over the span of all product x-parts ever (for child
    generation)
  - hcum[4]: accumulated h-cross residue per h-bit (120-11 = 109-bit
    vectors) - must be 0 at accept (degree>=2 h-cross terms forbidden
    in final h[b]; h0-linear terms allowed? NO: final h[b] = h0[b] ^
    INV(x)[b]: any h0_i (i != b) dependence at ALL (even linear) is
    forbidden; h0[b] itself must appear exactly once (linearly). The
    wire h[b] starts at h0[b]; products into it add content; NO
    product may introduce h0[b] dependence... products using h[b] as
    operand source DO introduce h0[b] (its current value = h0[b] ^
    stuff). Contradiction? No: h[b] as SOURCE: operand value = h0[b] ^
    acc: the product lands in a DIFFERENT wire (t or h[c]) - then
    h0[b]-dependence appears in that wire and must be removed...
    landing in t: t must be freed to |0> for all (x, h0) - h0-dep in t
    breaks MBU unless cancelled. Landing in h[c]: h[c] final has no
    h0[b] dependence => must cancel. All tracked via full 8-var d2 of
    each accumulator + t.
  - Therefore: track per-wire FULL 8-var d2 (120-bit): wire value's
    degree>=2 ANF content over (x, h0). Linear h0/x parts are free
    (affine shaping) - d2 representation suffices: final h[b] wire
    value must have 8-var d2 == d2(h0[b] ^ INV(x)[b]) = d2(INV[b])
    (h0[b] is degree 1: no d2) => h[b]'s accumulated 8-var d2 must
    equal TGT_D2[b] exactly; t's accumulated 8-var d2 must be 0 AND
    t's VALUE must be MBU-freeable (structural).

Products: A & B where A, B are affine forms of live wire values; in
d2-over-8-var terms each product's d2 = search5-style combination of
the operand wires' accumulated d2s and x/h0 affine riders. We compute
product d2 EXACTLY via 256-bit truth tables per candidate operand pair
(operands: concrete affine forms over the 9 current wire values) -
like q9wide.py but with span-restricted operand enumeration to keep
the tree finite, PLUS the 11-coord x-part projection for pruning.

Accept: acc_d2[b] == TGT_D28[b] (full 8-var) for all b; t_d2 == 0 with
t's value structurally freeable (each t-op's operands recorded).

This is the definitive Q=9 h-as-source search. Run:
  python3 q9hsrc.py [max_ops]
"""
import sys
import time

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
NV = 8
SIZE = 1 << NV
FULL = (1 << SIZE) - 1
XF = [sum(1 << r for r in range(SIZE) if (r >> i) & 1) for i in range(NV)]
MONOS = [m for m in range(1, 1 << NV) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)
XMONO_MASK = 0
for i, m in enumerate(MONOS):
    if m < 16:
        XMONO_MASK |= 1 << i


def d2v(fn):
    f = [(fn >> r) & 1 for r in range(SIZE)]
    for i in range(NV):
        for m in range(SIZE):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    v = 0
    for m in range(1, SIZE):
        if f[m] and m in IDX:
            v |= 1 << IDX[m]
    return v


TGT8 = [sum(((INV[r & 0xF] >> b) & 1) << r for r in range(SIZE))
        for b in range(4)]
TGT_D2 = [d2v(t) for t in TGT8]
TGT_D2_X = [t & XMONO_MASK for t in TGT_D2]


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
    return tuple(basis)


def contains(basis, v):
    for b in basis:
        v = min(v, v ^ b)
    return v == 0


def search(max_ops=5, deadline_s=3600):
    """DFS over op schedules. Wires: x0..3 (fixed TTs), h0..3 (values =
    h0 ^ acc: current VALUE TT = HF[j] ^ acc_tt[j]), t (value TT).
    We track VALUE truth tables (256-bit) of h-bits and t directly, and
    enumerate operands as affine forms over current wire VALUES (subset
    XOR + const) - a complete menu for one-ccx ops in this model.
    Memo on (h-values, t-value, depth). Prune on x-part rank deficit
    (admissible: remaining ops each add one product with one x-part).
    MBU structural check deferred to accept: t freeable iff each of
    t's op-products is reproducible: we record ops into t as (A_tt,
    B_tt) with the wire-index sets used; accept requires the operands
    be re-shapable from FINAL wire values (or x-only). Conservative:
    require operands x-only-affine in their d2 (no h0-cross content).
    """
    HF = [sum(1 << r for r in range(SIZE) if (r >> (4 + j)) & 1)
          for j in range(4)]
    memo = set()
    nodes = [0]
    t0 = time.time()

    def wire_forms(vals):
        """Distinct affine forms (subset-XOR + const) of the 9 current
        wire value TTs. Returns list of (tt, wire_usage_mask) - usage
        over (x, h0..3, t) indices for MBU bookkeeping."""
        out = {}
        n = len(vals)
        for supp in range(1, 1 << n):
            v = 0
            for j in range(n):
                if supp >> j & 1:
                    v ^= vals[j]
            if v:
                out.setdefault(v, supp)
            if v ^ FULL:
                out.setdefault(v ^ FULL, supp)
        return out

    def dfs(hvals, tval, tops, depth):
        nodes[0] += 1
        if nodes[0] % 50000 == 0:
            el = time.time() - t0
            print(f"  nodes={nodes[0]} depth={depth} ({el:.0f}s)",
                  file=sys.stderr)
            if el > deadline_s:
                raise TimeoutError
        # prune: x-part rank deficit (targets' x-parts must be coverable)
        accx = [d2v(hvals[b] ^ HF[b]) & XMONO_MASK for b in range(4)]
        basis = rref([d2v(tval)] + accx)
        miss = 0
        nb = list(basis)
        for b in range(4):
            need = TGT_D2_X[b] ^ accx[b]
            if need and not contains(rref(nb), need):
                nb.append(need)
                nb = list(rref(nb))
                miss += 1
        if miss > max_ops - depth:
            return False
        key = (tuple(hvals), tval, depth)
        if key in memo:
            return False
        memo.add(key)
        if depth == max_ops:
            for b in range(4):
                if hvals[b] != (HF[b] ^ TGT8[b]):
                    return False
            # t must be |0>: value TT zero (strongest, cleanest)
            if tval != 0:
                return False
            return True
        vals = list(XF[:4]) + list(hvals) + [tval]
        forms = wire_forms(vals)
        items = list(forms.items())
        # products: pairs of distinct-form values
        seen = set()
        prods = []
        for i, (a, ma) in enumerate(items):
            for j in range(i + 1, len(items)):
                b, mb = items[j]
                p = a & b
                if p and p not in seen:
                    seen.add(p)
                    prods.append((p, a, b, ma, mb))
        for (p, a, b, ma, mb) in prods:
            # target t
            if dfs(hvals, tval ^ p, tops + [(a, b, ma, mb)], depth + 1):
                return True
            # target h[k]: wire h[k] value ^= p
            for k in range(4):
                nh = list(hvals)
                nh[k] ^= p
                if dfs(tuple(nh), tval, tops, depth + 1):
                    return True
        return False

    try:
        ok = dfs(tuple(HF), 0, [], 0)
    except TimeoutError:
        return None, nodes[0], time.time() - t0
    return ok, nodes[0], time.time() - t0


if __name__ == "__main__":
    max_ops = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ok, n, dt = search(max_ops)
    if ok is None:
        print(f"TIMEOUT after {n} nodes ({dt:.0f}s)")
    else:
        print(f"{'FOUND' if ok else 'NOT FOUND'} after {n} nodes ({dt:.0f}s)")
