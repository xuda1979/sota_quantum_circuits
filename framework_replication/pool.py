#!/usr/bin/env python3
"""Compute the reachable raw-product pi pool for the F24 5-Toffoli search.

A raw product of one ccx with operands (accum_t, accum_u) (values = XOR of
earlier raw products plus free affine riders) has pi
    v = pq(a, b) ^ ext(a, Sb) ^ ext(b, Sa) ^ bip(Sa, Sb) ^ [a] ^ [b]
for some a, b in span(pool) (accumulated operands), supports Sa, Sb, and
optional const-rider terms [a], [b].

Pool fixed point: S <- {v : a, b in S, v in coset(a, b)}. S0 = quad space
(span of single-pair bipartite products; op1's products live in BIP35 but
the span closure makes the fixed point cleaner; BIP-membership for op1 is
re-checked later).

Run: python3 pool.py   (writes pool to stdout as hex masks; ~1-2 min)
"""
import sys

MONOS = [3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
IDX = {m: i for i, m in enumerate(MONOS)}
NM = len(MONOS)
FULLMASK = (1 << NM) - 1
QUAD_IDX = [IDX[m] for m in (3, 5, 6, 9, 10, 12)]
QUADS = (1 << NM) - 1


def to_vec(masks):
    r = 0
    for m in masks:
        r |= 1 << IDX[m]
    return r


def mono_mask_vec(m):
    return 1 << IDX[m]


# quad space: all 64 vectors over the 6 quad coordinates
QUADSPACE = set()
for v in range(1 << 6):
    q = 0
    for i in range(6):
        if v >> i & 1:
            q |= 1 << QUAD_IDX[i]
    QUADSPACE.add(q)

# E(a): {ext(a, S) : S subset of {0..3}} where ext(a, S) = XOR_{j in S} mono-extend(a, j)
# mono-extend(m, j) = m if j in m else m | (1<<j)  (idempotent x_j)


def ext(a, supp):
    r = 0
    for i in range(NM):
        if a >> i & 1:
            m = MONOS[i]
            for j in supp:
                m2 = m if (m >> j) & 1 else m | (1 << j)
                r ^= 1 << IDX[m2]
    return r


SUPPS = [tuple(i for i in range(4) if s >> i & 1) for s in range(16)]


def e_set(a):
    return {ext(a, s) for s in SUPPS}


def pq(a, b):
    r = 0
    for i in range(NM):
        if not (a >> i & 1):
            continue
        m1 = MONOS[i]
        for k in range(NM):
            if not (b >> k & 1):
                continue
            mu = m1 | MONOS[k]
            if mu in IDX:
                r ^= 1 << IDX[mu]
    return r


def bip(sa, sb):
    r = 0
    for i in sa:
        for j in sb:
            if i == j:
                continue
            r ^= mono_mask_vec((1 << i) | (1 << j))
    return r


def grow(span_set):
    """One fixed-point step: all v reachable from operand pairs in span_set."""
    lst = sorted(span_set)
    out = set(span_set)
    n = len(lst)
    # cache E-sets and pq
    ecache = [e_set(a) for a in lst]
    for ia in range(n):
        a = lst[ia]
        ea = ecache[ia]
        for ib in range(n):
            b = lst[ib]
            base = pq(a, b)
            eb = ecache[ib]
            # v = base ^ e_a ^ e_b ^ bip ^ [a] ^ [b]
            for va in ea:
                for vb in eb:
                    x = base ^ va ^ vb
                    out.add(x)
                    out.add(x ^ a)
                    out.add(x ^ b)
                    out.add(x ^ a ^ b)
                    if len(out) > 60000:
                        return out
    # add bipartite sets with all supports
    for sa in SUPPS:
        for sb in SUPPS:
            if sa and sb:
                out.add(bip(sa, sb))
    return out


def main():
    S = set(QUADSPACE)
    for it in range(8):
        S2 = grow(S)
        span_size = len(S2)
        print(f"iter {it}: |S| = {span_size}", file=sys.stderr)
        if S2 == S:
            break
        S = S2
        if span_size >= (1 << NM):
            break
    vecs = sorted(S)
    print(f"# pool size {len(vecs)} of {1<<NM}")
    for v in vecs:
        if v:
            print(hex(v))


if __name__ == "__main__":
    main()
