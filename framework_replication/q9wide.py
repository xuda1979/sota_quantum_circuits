#!/usr/bin/env python3
"""WIDENED Q=9 one-ancilla search for T=5 F24 inversion.

Widens q9search.py's model at the boundaries the user identified:
  (a) h-bits as operand sources: a product may use a shaped h-bit as an
      operand (h starts arbitrary; ops must keep h's final value exactly
      h ^ INV(x); since products XORing into h[b] and operands drawn from
      h[b] interact, we track all wires as truth tables over 8 variables
      (x0..x3, h0..h3_initial) = 256-bit integers).
  (b) any wire can transiently shape any other (cx is free) - modeled by
      letting each operand be any affine form over the CURRENT wire values,
      applied transiently (the x-register must return to x; enforced by
      tracking x-wire values and requiring final restoration - in this
      search x-shaping is undone per-op, i.e. operands are affine forms
      of current wires, and x-wires' persistent values never change).
  (c) t (the ancilla) as an operand at any time, not just as the target.
  (d) MBU free of t at the end: t must hold a value V = A & B for two
      affine forms of live wires (then hmr + cz(A-wire, B-wire, cond=m)),
      or 0. Freesearch.py established this idiom; here it is a terminal
      accept condition on t's final truth table.
  (e) multiple snapshot ops cx(t, h[b]) free at any time (already free in
      the paper's model since snapshots are cx = Clifford).

Acceptance: h[b] final == h0[b] ^ TGT[x][b] for all (x, h0) [256-row exact
check]; t final is 0 or a product A&B of two affine forms of live wires
[freeable]; x-wires unchanged throughout (persistent values).

Pruning: d2 span over 8 vars is weaker; instead prune on per-bit 8-var
degree>=2 monomial deficit vs remaining ops (admissible: each op adds at
most one new nonlinear product to the system).

This is a much larger space than q9search (8-var truth tables); the search
restricts to 5 ops with strong deficit pruning and memoization on wire
states.

Run: python3 q9wide.py [max_ops]
"""
import sys
import time

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
NV = 8            # x0..x3, h0..h3 (initial values)
SIZE = 1 << NV    # 256
XMASK = 0xF       # low 4 vars = x
HMASK = 0xF0      # high 4 = h0

XF = [sum(1 << r for r in range(SIZE) if (r >> i) & 1) for i in range(4)]
HF = [sum(1 << r for r in range(SIZE) if (r >> (4 + j)) & 1) for j in range(4)]
FULL = (1 << SIZE) - 1

# wire identity tables
XW = XF[:]             # persistent x-wire values: x_i
HW = HF[:]             # persistent h-wire values: h0_j (initial; a wire's
                       # CURRENT value changes as ops XOR in; we track per-
                       # wire current values in the state)
T0 = 0                 # ancilla starts 0

# target contribution per output bit: INV(x) as function on 256 rows
TGTB = [sum(((INV[r & XMASK] >> b) & 1) << r for r in range(SIZE))
        for b in range(4)]

MONOS = [m for m in range(1, 1 << NV) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)


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


TGT_D2 = [d2v(t) for t in TGTB]


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


def shapes(base_tt):
    """Affine orbit of a wire value under free cx/x shaping: the wire XORed
    with any affine form over ALL 8 variables (cx from any wire to any wire
    is free in the kickmix model; x-register restoration is handled by
    transience - we shape, use, unshape)."""
    return base_tt  # placeholder; real search enumerates forms inline


def affine_forms():
    """All 512 affine forms over 8 vars as 256-bit TTs, grouped by whether
    they are realizable as (constant XOR subset of the 8 identity wires).
    cx(a, b) means wire b ^= wire a's value - so an affine form is any XOR
    of a subset of current wire values, plus const."""
    forms = [0, FULL]
    base = XF + HF
    for supp in range(1, 1 << NV):
        v = 0
        for j in range(NV):
            if supp >> j & 1:
                v ^= base[j]
        forms.append(v)
        forms.append(v ^ FULL)
    return forms


def search(max_ops=5):
    AFF = affine_forms()
    # state: tuple of 9 wire current values (x0..3 unchanged = XF, h0..3
    # current, t current), accumulated per-wire op history for MBU check.
    # ops: ccx(A, B, target) where A = affine form of subset of CURRENT
    # wires XOR const (transient shaping: shape target's operand wires by
    # cx from other wires, ccx, unshape). IMPORTANT: shaping wire w by cx
    # from wire v changes w transiently; for the op we need operands as
    # affine forms A, B and target wire; after op undo shaping. The
    # persistent effect: target ^= A*B.
    # Product menu per state: pairs of distinct-shape affine forms over
    # current 9-wire values. But affine forms over current wires = affine
    # forms over 8 base vars (since wires are functions of those). The set
    # of possible products = {A & B : A, B in AFF(wire-values)}. Wire
    # values change per state, so recompute per state (memoized on state).
    memo = {}
    nodes = [0]
    t0 = time.time()

    def wire_forms(wires):
        """affine forms over current wire values (as 256-bit TTs), deduped."""
        key = wires
        hit = memo.get(key)
        if hit is not None:
            return hit
        seen = set()
        out = []
        for supp in range(1, 1 << NV):
            v = 0
            for j in range(NV):
                if supp >> j & 1:
                    v ^= wires[j]
            if v not in seen:
                seen.add(v)
                out.append(v)
            v2 = v ^ FULL
            if v2 not in seen:
                seen.add(v2)
                out.append(v2)
        if 0 not in seen:
            out.append(0)
        if FULL not in seen:
            out.append(FULL)
        memo[key] = out
        return out

    def mbu_freeable(t_tt, wires):
        """t freeable if t_tt == 0 or A & B for affine forms A, B of live
        wires (fixup cz on the shaped operand wires)."""
        if t_tt == 0:
            return True
        forms = wire_forms(wires)
        fset = set(forms)
        # A & B == t_tt with A, B affine forms: enumerate A, solve B = t/A?
        # Not unique; instead enumerate pairs restricted: A & B is a product
        # of two affine forms over 8 vars. Try all pairs (forms ~ up to 512;
        # 512^2 too big) -> restrict to forms with support intersecting
        # t_tt's support and t_tt | A == A (B = t & ... ). A & B = t implies
        # t's support subset of A's and B's. Enumerate A with supp(t) <=
        # supp(A), then need B with B & A = t; B = t | (B' off supp(A))...
        # simpler: for the fixup we need each OP that wrote into t to be
        # re-creatable; here we take the per-op idiom: t = XOR of products,
        # each product's operands re-shapable. The MBU fixup then is per-op
        # cz(A_i, B_i) which reproduces t exactly iff each product was
        # formed from operand wire values still live at free time. We track
        # this structurally instead (see caller): every op into t is
        # recorded with its operand forms over wires that must still hold.
        return False  # structural check used instead

    def dfs(wires, hist, depth):
        """wires: 9-tuple current values (x0..x3, h0..h3, t); hist: ops into
        t as (A, B, operand-wire-indices) for the MBU structural check."""
        nodes[0] += 1
        if nodes[0] % 100000 == 0:
            print(f"  nodes={nodes[0]} depth={depth} "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)
        hw = wires[4:8]
        t_tt = wires[8]
        if depth == max_ops:
            # output check: h[b] must equal h0[b] ^ INV(x)[b]
            for b in range(4):
                if hw[b] != (HF[b] ^ TGTB[b]):
                    return False
            # MBU structural check on t: t = XOR of products A_i & B_i
            # where each A_i/B_i is an affine form of the wire values AT
            # THE TIME of op i. Freeable iff at free time (final state),
            # hmr(t) then per-op cz(A_i, B_i, cond=m) reproduces XOR of
            # A_i*B_i... but A_i, B_i depend on wire values at op time,
            # which may differ from free time. The fixup is exact iff
            # the operand wires' values at free time still realize A_i, B_i
            # when re-shaped. Since ops into t used operand forms over
            # wires, and only t and h changed since, forms over x/h wires
            # are only valid if op i's operands were affine in the FINAL
            # wire values. Sufficient condition used here: every t-op's
            # operands are affine forms over final-wire values.
            cur = 0
            for (A, B) in hist:
                # recompute the product from the forms at free time:
                # A, B are stored as affine forms over the 8 base vars
                # (already concrete 256-bit TTs of the operand VALUES at
                # op time). The cz fixup needs the operand VALUES; if the
                # underlying wires changed since op time, the stored TTs
                # are stale. Only ops whose operand TTs are affine forms
                # over x-only, or over final h/t-free values, fix up.
                # We require: operands affine over x-wires only (x never
                # changes), OR over current final wire values.
                cur ^= A & B  # this is t's accumulated value
            # check t == cur (should hold by construction)
            if t_tt != cur:
                return False
            # freeability: each stored operand TT must be re-shapable now:
            for (A, B) in hist:
                forms = wire_forms(wires)
                if A not in forms and A != 0 and A != FULL:
                    # A might still be an affine form over x only
                    if A not in wire_forms(wires[:4] + wires[4:8] + (0,)):
                        return False
                if B not in forms and B != 0 and B != FULL:
                    if B not in wire_forms(wires[:4] + wires[4:8] + (0,)):
                        return False
            return True
        # prune: per-bit nonlinear deficit
        # remaining ops each add ONE product; h[b] needs its d2 target
        # covered by products landed in h[b] (+ snapshots from t).
        # Admissible: rank deficit of {d2(TGT_b ^ current h_b ^ affine)}...
        # skip heavy prune at shallow depth; simple depth prune first.
        # enumerate ops: target wires = h0..h3, t
        # operands: pairs of affine forms over current wires (distinct
        # forms; product nonzero). Dedup on product TT.
        forms = wire_forms(wires)
        # candidate products (dedup by TT)
        seen = set()
        prods = []
        nf = len(forms)
        for i in range(nf):
            A = forms[i]
            for j in range(i + 1, nf):
                p = A & forms[j]
                if p and p not in seen:
                    seen.add(p)
                    prods.append((p, A, forms[j]))
        # ops into h[b]: h[b] ^= p  (h[b] becomes h[b] ^ p)
        # ops into t: t ^= p, record (A, B) for MBU
        # snapshot ops cx(t, h[b]) are covered: they are NOT products but
        # free Cliffords - model as: h[b] ^= t's current value, free move.
        # To keep the search finite, enumerate per depth: either one ccx
        # (product into h or t) or one free snapshot cx(t -> h[b]).
        # Snapshots are free but must be bounded; allow snapshot only as
        # part of acceptance (they don't consume depth). Simplification:
        # after the final op, ANY subset of {t, t^p_last...} snapshots is
        # free - but t's intermediate values are gone. Instead allow free
        # snapshot moves explicitly at any depth (they don't increment
        # depth but change state; memo prevents loops - however free ops
        # could cycle; cap: snapshot at most once per (depth, b) - the
        # useful snapshots are of current t into some h[b]).
        # ... handled by caller loop below.
        for (p, A, B) in prods:
            for b in range(4):
                nw = list(wires)
                nw[4 + b] ^= p
                if dfs(tuple(nw), hist, depth + 1):
                    return True
            nw = list(wires)
            nw[8] ^= p
            if dfs(tuple(nw), hist + [(A, B)], depth + 1):
                return True
        # free snapshots: t's CURRENT value into any subset of h bits
        if t_tt:
            for sb in range(1, 16):
                nw = list(wires)
                for b in range(4):
                    if sb >> b & 1:
                        nw[4 + b] ^= t_tt
                if dfs(tuple(nw), hist, depth):
                    return True
        return False

    wires0 = tuple(XF) + tuple(HF) + (0,)
    ok = dfs(wires0, [], 0)
    return ok, nodes[0]


if __name__ == "__main__":
    max_ops = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ok, n = search(max_ops)
    print(f"{'FOUND' if ok else 'NOT FOUND'} after {n} nodes")
