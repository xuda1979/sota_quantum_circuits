#!/usr/bin/env python3
"""
existsearch.py -- sound existence search for an F24 (GF(16) inversion)
circuit at T=5, Q=10, aiming to close the certificate's existence leg with
our OWN emitted circuit.

Model (two ancillas t, u; h = 4-bit output register):
  * State: 16-bit truth tables t_tt, u_tt of the ancilla wires;
    per-h-bit accumulators acc[0..3] (bitmask of h-targeted products so far);
    shared quadratic basis B = rref of {d2(P) : P ancilla-targeted product}.
  * Ops (each consumes one Toffoli):
      - ancilla-targeted (tgt in {t, u}): ccx(A, B, tgt) where A, B are
        PROPER affine shapings of two distinct x-wires (XPRODS menu only).
        Sound for MBU: freeing t/u later needs x-affine-only operands.
        No operand/target aliasing, no t/u operands on ancilla targets.
      - h-targeted (tgt = h[b]): product of
          (x-affine, x-affine)  [XPRODS]
          (t-shape, x-affine)   [32 t-shapes: t_tt ^ any x-subset ^ {0,const}]
          (u-shape, x-affine)   [same, for u]
          (t-shape, u-shape)    [cross pairs]
          (h-acc, x-affine | t-shape | u-shape | h-acc)  [h-source operands,
           v3: widened operand-source model; emission uses transient shaping]
        deduped on product TT per (bit, depth) menu.
        Aliasing rule: an h-source op may not target the same h-bit it reads
        as an operand control (ccx with an aliased control is not the modeled
        toggle), enforced in dfs.
  * Acceptance (prefix triangularity, q9search-style, two accumulators):
      at depth == max_ops, for every b: d2(TGT_b ^ acc[b]) in span(B).
      Free affine assembly on h afterwards (cx from x-wires, x const).
      Mid-circuit snapshots cx(t, h_b) / cx(u, h_b) are free, so h-targeted
      products never enter the shared basis (h is never freed).
  * Prune: missing_rank > remaining ops  ->  cut (rank must close in time).
  * Memo: key includes the basis (q9search's memo omitted it -> lossy).
  * Deadline + node counter with periodic progress on stderr.

Output: results/existsearch.txt (FOUND schedule or honest NOT FOUND /
TIMEOUT record).
"""

import sys
import time
from functools import lru_cache

# ---------------------------------------------------------------- core

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
XF = [sum(1 << x for x in range(16) if (x >> i) & 1) for i in range(4)]
FULL = 0xFFFF

# deg>=2 monomials over x0..x3 (ANF bitmask convention, bit i = monomial i)
MONOS = [m for m in range(1, 16) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}

TGT = [sum(((INV[x] >> b) & 1) << x for x in range(16)) for b in range(4)]


def d2(fn):
    """Mobius transform; project TT onto its deg>=2 ANF coefficients."""
    f = [(fn >> x) & 1 for x in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    v = 0
    for m in range(1, 16):
        if f[m] and m in IDX:
            v |= 1 << IDX[m]
    return v


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


def dim(vecs):
    return len(rref(vecs))


TGT_D2 = [d2(TGT[b]) for b in range(4)]

# ---------------------------------------------------------------- shapes

# all non-const affine forms over x, grouped by a wire that can hold them
AFFS = []
for supp in range(1, 16):
    for c in (0, 1):
        tt = FULL if c else 0
        for j in range(4):
            if supp >> j & 1:
                tt ^= XF[j]
        AFFS.append((supp, c, tt))
BY_WIRE = [[f for f in AFFS if f[0] >> w & 1] for w in range(4)]


def xpair_products():
    seen = set()
    out = []
    for w1 in range(4):
        for w2 in range(w1 + 1, 4):
            for (s1, c1, t1) in BY_WIRE[w1]:
                for (s2, c2, t2) in BY_WIRE[w2]:
                    p = t1 & t2
                    if p and p not in seen:
                        seen.add(p)
                        out.append(((w1, s1, c1), (w2, s2, c2), p))
    return out


XPRODS = xpair_products()

# 32 shapings of a base TT: base ^ any x-subset, with/without const
def shapes_of(base):
    out = set()
    for supp in range(16):
        v = base
        for j in range(4):
            if supp >> j & 1:
                v ^= XF[j]
        out.add(v)
        out.add(v ^ FULL)
    return out


@lru_cache(maxsize=None)
def shaped_x_products(base_tt):
    """Products of a shaping of base_tt (t or u) with any x-affine form.
    Returns list of (operand_tag, x_wire, x_supp, x_const, product_tt)."""
    seen = set()
    out = []
    for sh in shapes_of(base_tt):
        for w in range(4):
            for (s, c, tt) in BY_WIRE[w]:
                p = sh & tt
                if p and p not in seen:
                    seen.add(p)
                    out.append((sh, w, s, c, p))
    return out


def menus(t_tt, u_tt, h_tts):
    """h-target product menu for a given state: XPRODS + t-shapes +
    u-shapes + t&u cross-shapes + h-source products (h_b & x-aff, h_b &
    t-shape, h_b & u-shape, h_a & h_b), deduped on product TT.
    Returns list of (kind, a, b, p)."""
    seen = set()
    out = []
    for (a1, a2, p) in XPRODS:
        if p not in seen:
            seen.add(p)
            out.append(("x", a1, a2, p))
    for (sh, w, s, c, p) in shaped_x_products(t_tt):
        if p not in seen:
            seen.add(p)
            out.append(("t", sh, (w, s, c), p))
    for (sh, w, s, c, p) in shaped_x_products(u_tt):
        if p not in seen:
            seen.add(p)
            out.append(("u", sh, (w, s, c), p))
    for sh_t in shapes_of(t_tt):
        for sh_u in shapes_of(u_tt):
            p = sh_t & sh_u
            if p and p not in seen:
                seen.add(p)
                out.append(("tu", sh_t, sh_u, p))
    # h-source products (emission: transient shaping of h wires, undone)
    for b in range(4):
        hb = h_tts[b]
        if not hb:
            continue
        for (s, c, tt) in AFFS:
            p = hb & tt
            if p and p not in seen:
                seen.add(p)
                out.append(("hx", b, (s, c), p))
        for sh in shapes_of(hb):
            for w in range(4):
                for (s, c, tt) in BY_WIRE[w]:
                    p = sh & tt
                    if p and p not in seen:
                        seen.add(p)
                        out.append(("hx", b, (w, s, c), p))
        for sh_t in shapes_of(t_tt):
            p = sh_t & hb
            if p and p not in seen:
                seen.add(p)
                out.append(("ht", b, sh_t, p))
        for sh_u in shapes_of(u_tt):
            p = sh_u & hb
            if p and p not in seen:
                seen.add(p)
                out.append(("hu", b, sh_u, p))
        for a in range(4):
            if a == b:
                continue
            p = h_tts[a] & hb
            if p and p not in seen:
                seen.add(p)
                out.append(("hh", b, a, p))
    return out


def missing_rank(basis, acc):
    """# of TGT d2 vectors not yet reachable from span(basis) ^ acc offsets."""
    need = [TGT_D2[b] ^ d2(acc[b]) for b in range(4) if TGT_D2[b] ^ d2(acc[b])]
    if not need:
        return 0
    m = 0
    for v in need:
        if not contains(basis, v):
            m += 1
    return m


def prune(basis, acc, depth, max_ops):
    need = [TGT_D2[b] ^ d2(acc[b]) for b in range(4)]
    allv = list(basis) + need
    return dim(allv) - dim(basis) > max_ops - depth


# ---------------------------------------------------------------- search

class DeadlineHit(Exception):
    pass


def dfs(t_tt, u_tt, basis, acc, depth, ops, max_ops, deadline, stats):
    """Two-ancilla DFS.  Returns True (solution in stats["solution"]) or
    False (no solution from this state).  Raises DeadlineHit on timeout."""
    stats["nodes"] += 1
    if time.monotonic() > deadline:
        stats["deadline_hit"] = True
        raise DeadlineHit
    if stats["nodes"] % 200000 == 0:
        stats["rate"] = stats["nodes"] / max(time.monotonic() - stats["t0"], 1e-9)
        print("[existsearch] nodes=%d  rate=%.0f/s  depth=%d  memo=%d"
              % (stats["nodes"], stats["rate"], depth, len(memo)),
              file=sys.stderr, flush=True)
    if depth == max_ops:
        for b in range(4):
            need = TGT_D2[b] ^ d2(acc[b])
            if need and not contains(basis, need):
                return False
        stats["solution"] = [op for op in ops]
        return True
    if prune(basis, acc, depth, max_ops):
        return False
    # memo key MUST include the basis (q9search's lossy-memo lesson)
    key = (t_tt, u_tt, basis, acc, depth)
    if key in memo:
        return False
    memo.add(key)
    # -- ancilla-targeted ops: x-affine operand pairs only (XPRODS).
    #    MBU fixups always re-evaluable; no operand/target aliasing.
    for (a1, a2, p) in XPRODS:
        for tgt in ("t", "u"):
            if tgt == "t":
                nt, nu = t_tt ^ p, u_tt
            else:
                nt, nu = t_tt, u_tt ^ p
            npd2 = d2(p)
            nb = basis if contains(basis, npd2) else rref(list(basis) + [npd2])
            if dfs(nt, nu, nb, acc, depth + 1,
                   ops + [("acc", tgt, a1, a2, p)], max_ops, deadline, stats):
                return True
    # -- h-targeted ops: full menu incl. h-source products (v3).
    #    Aliasing rule: an h-based entry reads h-bit a1 (and a2 for "hh");
    #    it cannot toggle the same bit it reads (a ccx with an aliased
    #    control is not the modeled snapshot toggle).
    menu = menus(t_tt, u_tt, acc)
    for g in range(4):
        for (kind, a1, a2, p) in menu:
            if kind in ("hx", "ht", "hu") and g == a1:
                continue
            if kind == "hh" and (g == a1 or g == a2):
                continue
            nacc = acc[:g] + (acc[g] ^ p,) + acc[g + 1:]
            if dfs(t_tt, u_tt, basis, nacc, depth + 1,
                   ops + [("h", g, kind, a1, a2, p)], max_ops, deadline, stats):
                return True
    return False


def run(max_ops, deadline_minutes):
    deadline = time.monotonic() + deadline_minutes * 60
    stats = {"nodes": 0, "t0": time.monotonic(), "deadline_hit": False,
             "solution": None, "rate": 0.0}
    try:
        found = dfs(0, 0, (), (0, 0, 0, 0), 0, [], max_ops, deadline, stats)
    except DeadlineHit:
        found = False
    elapsed = time.monotonic() - stats["t0"]
    outcome = "FOUND" if stats["solution"] is not None else \
              ("TIMEOUT" if stats["deadline_hit"] else "NOT FOUND")
    print("[existsearch] %s  max_ops=%d  nodes=%d  elapsed=%.1fs  memo=%d"
          % (outcome, max_ops, stats["nodes"], elapsed, len(memo)),
          file=sys.stderr, flush=True)
    return outcome, stats, elapsed


def main():
    max_ops = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    deadline_min = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
    out = sys.argv[3] if len(sys.argv) > 3 else \
        "results/existsearch.txt"
    print("[existsearch] model sizes: |XPRODS|= %d  |AFFS|= %d"
          % (len(XPRODS), len(AFFS)), file=sys.stderr, flush=True)
    outcome, stats, elapsed = run(max_ops, deadline_min)
    lines = []
    lines.append("existsearch: F24 inversion T=%d Q=10 (2 ancilla) sound search"
                 % max_ops)
    lines.append("outcome: %s" % outcome)
    lines.append("nodes: %d" % stats["nodes"])
    lines.append("memo_states: %d" % len(memo))
    lines.append("elapsed_seconds: %.1f" % elapsed)
    lines.append("rate_nodes_per_sec: %.0f" % (stats["nodes"] / max(elapsed, 1e-9)))
    if stats["solution"] is not None:
        lines.append("solution (%d ops):" % len(stats["solution"]))
        for op in stats["solution"]:
            lines.append("  " + repr(op))
    else:
        lines.append("model_scope: "
                     "T-q-anc spatterns under the finalized sound model; "
                     "NOT FOUND = no circuit in model, TIMEOUT = inconclusive")
    txt = "\n".join(lines) + "\n"
    print(txt, end="")
    with open(out, "w") as f:
        f.write(txt)
    print("[existsearch] wrote %s" % out, file=sys.stderr, flush=True)


memo = set()

if __name__ == "__main__":
    main()
