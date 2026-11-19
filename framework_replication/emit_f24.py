#!/usr/bin/env python3
"""Emit a physical Builder circuit from the tsearch 5-op F24 solution.

Physical realizability gate: every ccx product must decompose as
  (x_i ^ rider_i ^ c_i) & (x_j ^ rider_j ^ c_j),  i != j,
with riders over OTHER input wires only (supp must NOT contain the wire's
own index — cx(x_i, x_i) is destructive). tsearch's `shapes` allowed
supp containing the wire (unrealizable), so membership in this PROPER
menu must be checked before emission.

Emission: transient shaping around each ccx (cx riders on/off), boundary
snapshots cx(t|h, h[b]), MBU frees (hmr + per-op re-shaped cz fixups,
recycle), then the free affine assembly per output bit.
"""
import subprocess
import sys
from itertools import combinations

sys.path.insert(0, "/home/david/software/sota_quantum_circuits/framework_replication")
from tsearch import XF, FULL, TGT, d2, verify  # noqa: E402

# tsearch solution (2026-11-17 run), state-verified below
PATH = [
    (0x0000, 0x0000, 0xAA00, "t"),
    (0xAA00, 0x0000, 0x6024, "t"),
    (0xCA24, 0x0000, 0x02C8, "u"),
    (0xCA24, 0x02C8, 0xA018, "t"),
    (0x6A3C, 0x02C8, 0x8230, "t"),
]

OUT = "/home/david/software/sota_quantum_circuits/framework_replication/candidates/f24_tsearch5.py"


def proper_shapes(i):
    others = [j for j in range(4) if j != i]
    out = []
    for r in range(4):
        for supp in combinations(others, r):
            for c in (0, 1):
                v = XF[i] ^ (FULL if c else 0)
                for j in supp:
                    v ^= XF[j]
                out.append((supp, c, v))
    return out


MENU = {}  # product tt -> (i, suppA, cA, j, suppB, cB)
for i in range(4):
    for j in range(i + 1, 4):
        for (sA, cA, vA) in proper_shapes(i):
            for (sB, cB, vB) in proper_shapes(j):
                p = vA & vB
                if p and p not in MENU:
                    MENU[p] = (i, sA, cA, j, sB, cB)


def anf_coefs(tt):
    f = [(tt >> v) & 1 for v in range(16)]
    for i in range(4):
        for m in range(16):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    return f


def subset_solve(values, target_d2):
    """Mask over `values` whose d2-XOR equals target_d2, or None."""
    basis = []  # (vec, mask)
    for i, v in enumerate(values):
        cur, m = d2(v), 1 << i
        changed = True
        while changed:
            changed = False
            for (bv, bm) in basis:
                cand = cur ^ bv
                if cand < cur:
                    cur, m = cand, m ^ bm
                    changed = True
        if cur:
            basis.append((cur, m))
            basis.sort(key=lambda e: -e[0])
    cur, m = target_d2, 0
    changed = True
    while changed:
        changed = False
        for (bv, bm) in basis:
            cand = cur ^ bv
            if cand < cur:
                cur, m = cand, m ^ bm
                changed = True
    return None if cur else m


def main():
    # 1. state-verify the path (tsearch's own replay)
    bounds = verify(PATH)
    print("path state-verified; boundaries:",
          [f"({t:04x},{u:04x})" for t, u in bounds])

    # 2. physical-menu membership
    missing = [f"{p:04x}" for (_, _, p, _) in PATH if p not in MENU]
    if missing:
        print("NOT physically realizable as pure input-shaped products:",
              missing)
        sys.exit(2)
    print("all 5 products in the proper menu")

    # 3. snapshot solve: per bit, subset of distinct boundary values
    values = {}  # tt -> (wire, boundary)
    for bi in range(1, len(bounds)):
        tv, uv = bounds[bi]
        if tv and tv not in values:
            values[tv] = ("t", bi)
        if uv and uv not in values:
            values[uv] = ("u", bi)
    vlist = list(values.keys())
    snaps = {}  # bit -> [(wire, boundary, tt)]
    for b in range(4):
        m = subset_solve(vlist, d2(TGT[b]))
        assert m is not None, f"bit {b}: subset solve failed"
        snaps[b] = [(*values[vlist[i]], vlist[i]) for i in range(len(vlist))
                    if m >> i & 1]
    for b in range(4):
        print(f"bit {b} snapshots:",
              [(w, bi, f"{tt:04x}") for (w, bi, tt) in snaps[b]])

    # 4. emit circuit
    L = [
        '"""F24 inversion, 5 Toffolis, from the deterministic tsearch span',
        'search (2026-11-17). Emitted by emit_f24.py; all products use',
        'transient input-wire shaping only."""',
        'import sys',
        'sys.path.insert(0, "/home/david/software/sota_quantum_circuits/uncompute")',
        'from uncompute import Builder',
        '',
        '',
        'def build(b):',
        '    x = b.qubits(4); b.qubit_register(x)',
        '    h = b.qubits(4); b.qubit_register(h)',
        '    t = b.qubit(); u = b.qubit()',
    ]

    def shape_lines(i, supp, c, wire):
        for j in supp:
            L.append(f"    b.cx(x[{j}], {wire})")
        if c:
            L.append(f"    b.x({wire})")

    def unshape_lines(i, supp, c, wire):
        if c:
            L.append(f"    b.x({wire})")
        for j in reversed(supp):
            L.append(f"    b.cx(x[{j}], {wire})")

    ops = [(p, tgt, MENU[p]) for (_, _, p, tgt) in PATH]
    for k, (p, tgt, (i, sA, cA, j, sB, cB)) in enumerate(ops):
        acc = "t" if tgt == "t" else "u"
        L.append(f"    # op{k + 1}: p={p:04x} -> {acc}")
        shape_lines(i, sA, cA, f"x[{i}]")
        shape_lines(j, sB, cB, f"x[{j}]")
        L.append(f"    b.ccx(x[{i}], x[{j}], {acc})")
        unshape_lines(j, sB, cB, f"x[{j}]")
        unshape_lines(i, sA, cA, f"x[{i}]")
        # snapshots at boundary k+1
        for b in range(4):
            for (w, bi, tt) in snaps[b]:
                if bi == k + 1:
                    L.append(f"    b.cx({w}, h[{b}])"
                             f"  # snap {tt:04x}")

    # frees: MBU per ancilla, fixups re-create the same shaped operands
    for acc in ("t", "u"):
        L.append(f"    m_{acc} = b.hmr({acc})")
        for (p, tgt, (i, sA, cA, j, sB, cB)) in ops:
            if tgt != acc:
                continue
            shape_lines(i, sA, cA, f"x[{i}]")
            shape_lines(j, sB, cB, f"x[{j}]")
            L.append(f"    b.cz(x[{i}], x[{j}], cond=m_{acc})")
            unshape_lines(j, sB, cB, f"x[{j}]")
            unshape_lines(i, sA, cA, f"x[{i}]")
        L.append(f"    b.recycle({acc})")

    # free affine assembly per bit
    for b in range(4):
        resid = TGT[b]
        for (w, bi, tt) in snaps[b]:
            resid ^= tt
        coef = anf_coefs(resid)
        assert all(coef[m] == 0 for m in range(16)
                   if bin(m).count("1") >= 2), f"bit {b}: residual not affine"
        if coef[0]:
            L.append(f"    b.x(h[{b}])")
        for j in range(4):
            if coef[1 << j]:
                L.append(f"    b.cx(x[{j}], h[{b}])")
    L.append("    return x, h")
    L.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {OUT}")

    # 5. evaluate
    r = subprocess.run(
        [sys.executable,
         "/home/david/software/sota_quantum_circuits/framework_replication/f24_evaluator.py",
         OUT], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip():
        print("STDERR:", r.stderr.strip()[:2000])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
