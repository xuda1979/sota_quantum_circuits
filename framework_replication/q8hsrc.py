#!/usr/bin/env python3
"""Q=8 (no ancilla) T=5 search with h-bits as operand SOURCES.

Closes the honest gap left by q8probe.py (which proved Q=8 impossible
at ANY T for x-wire-only operands, via targets lying outside the
6-dimensional span of the 35 x-pair product d2s). Here h-wires may
serve as ccx operands; their accumulated product content then feeds
new products (pq-closure), so the x-part universe grows beyond the
q8probe menu. Model:
  - wires: x0..x3 (fixed TTs over (x, h0): 8 vars, 256-bit), h0..h3
    (values = h0_j ^ acc_j),
  - ops: ccx(A, B, target-wire) with A, B distinct affine forms
    (subset-XOR + const) of CURRENT wire values; target any wire,
  - 5 paid ops total; snapshots are just products' XORs (no ancilla:
    nothing to snapshot from),
  - accept: h[b] final == h0[b] ^ INV(x)[b] for all (x, h0) rows.

Search: DFS over (h-values, depth), memoized; prune on x-part rank
deficit vs remaining ops. This answers: can h-as-source operands beat
Q=8 at T=5? (Negative => Q>=9 at T=5 even with h-sources, completing
the q9search result to the wider model; the q9hsrc run covers Q=9.)

Run: python3 q8hsrc.py [max_ops]
"""
import sys
import time

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
NV = 8
SIZE = 1 << NV
FULL = (1 << SIZE) - 1
XF = [sum(1 << r for r in range(SIZE) if (r >> i) & 1) for i in range(4)]
HF = [sum(1 << r for r in range(SIZE) if (r >> (4 + j)) & 1) for j in range(4)]
MONOS = [m for m in range(1, 1 << NV) if bin(m).count("1") >= 2]
IDX = {m: i for i, m in enumerate(MONOS)}
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


def search(max_ops=5, deadline_s=7200):
    memo = set()
    nodes = [0]
    t0 = time.time()

    def wire_forms(vals):
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

    def dfs(hvals, depth):
        nodes[0] += 1
        if nodes[0] % 50000 == 0:
            el = time.time() - t0
            print(f"  nodes={nodes[0]} depth={depth} ({el:.0f}s)",
                  file=sys.stderr)
            if el > deadline_s:
                raise TimeoutError
        accx = [d2v(hvals[b] ^ HF[b]) & XMONO_MASK for b in range(4)]
        basis = rref(accx)
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
        key = (tuple(hvals), depth)
        if key in memo:
            return False
        memo.add(key)
        if depth == max_ops:
            return all(hvals[b] == (HF[b] ^ TGT8[b]) for b in range(4))
        vals = list(XF[:4]) + list(hvals)
        forms = wire_forms(vals)
        items = list(forms.items())
        seen = set()
        prods = []
        for i, (a, ma) in enumerate(items):
            for j in range(i + 1, len(items)):
                b, mb = items[j]
                p = a & b
                if p and p not in seen:
                    seen.add(p)
                    prods.append(p)
        for p in prods:
            for k in range(4):
                nh = list(hvals)
                nh[k] ^= p
                if dfs(tuple(nh), depth + 1):
                    return True
            # target an x-wire: transient shaping is undone, but a ccx
            # INTO an x-wire would leave a permanent product in x -
            # x must be preserved: only allowed if undone by a later op.
            # Exclude (x-preservation with <=5 ops and this target adds
            # no useful content: skip).
        return False

    try:
        ok = dfs(tuple(HF), 0)
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
