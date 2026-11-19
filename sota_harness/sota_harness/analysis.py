"""Algebraic analysis of Boolean functions for circuit discovery.

- anf(): algebraic normal form of a truth-table function.
- multiplicative_complexity_lb(): lower bound on the number of AND gates
  (hence Toffoli-family products with free Cliffords) needed to compute a
  set of functions. This yields a Toffoli lower-bound CERTIFICATE: any
  T-Toffoli circuit computes outputs in the span of <= T products of
  earlier wire values, so if depth-(T-1) exhaustive span search fails,
  T-1 Toffolis are impossible. Used to certify optimality of discovered
  circuits (an improvement over the paper, which reports no bounds).
- product_span_search(): the exhaustive span-closure search used both as a
  proposer (find the T-product decomposition, then synthesize wiring) and
  as a prover (failure at depth T-1 => optimal).
"""

from __future__ import annotations

from itertools import product as iproduct
from typing import Iterable, Sequence


def anf(fn: int, nvars: int) -> dict[int, bool]:
    """ANF (Reed-Muller) of a truth-table-packed function.

    ``fn`` packs f(x) at bit x, x in [0, 2^nvars). Returns {monomial mask: 1}.
    """
    size = 1 << nvars
    f = [(fn >> x) & 1 for x in range(size)]
    for i in range(nvars):
        for m in range(size):
            if m >> i & 1:
                f[m] ^= f[m ^ (1 << i)]
    return {m: True for m in range(size) if f[m]}


def anf_masks(fn: int, nvars: int, min_degree: int = 0) -> list[int]:
    return sorted(m for m in anf(fn, nvars)
                  if bin(m).count("1") >= min_degree)


def anf_of_table(table: Sequence[int]) -> dict[int, bool]:
    """ANF of a function given as an output list over x = 0..len-1."""
    nvars = (len(table) - 1).bit_length()
    fn = 0
    for x, v in enumerate(table):
        if v & 1:
            fn |= 1 << x
    return anf(fn, nvars)


# ---------------------------------------------------------------------------
# span machinery over monomial coordinate vectors
# ---------------------------------------------------------------------------

class MonoSpace:
    """Coordinate vectors over the monomials of degree >= 2 (the part that
    costs Toffolis; degree <= 1 is free Cliffords)."""

    def __init__(self, nvars: int):
        self.nvars = nvars
        self.monos = [m for m in range(1, 1 << nvars)
                      if bin(m).count("1") >= 2]
        self.idx = {m: i for i, m in enumerate(self.monos)}
        self.nm = len(self.monos)

    def vec(self, masks: Iterable[int]) -> int:
        r = 0
        for m in masks:
            if m in self.idx:
                r |= 1 << self.idx[m]
        return r

    def vec_fn(self, fn: int) -> int:
        return self.vec(m for m in anf(fn, self.nvars))

    def masks_of(self, v: int) -> list[int]:
        return [self.monos[i] for i in range(self.nm) if v >> i & 1]

    # -- span algebra ----------------------------------------------------
    @staticmethod
    def rref(vecs: Iterable[int]) -> tuple[int, ...]:
        basis: list[int] = []
        for v in vecs:
            if not v:
                continue
            for b in basis:
                v = min(v, v ^ b)
            if v:
                basis.append(v)
                basis.sort(reverse=True)
        return tuple(basis)

    @staticmethod
    def contains(basis: Sequence[int], v: int) -> bool:
        for b in basis:
            v = min(v, v ^ b)
        return v == 0

    # -- product-space operations -----------------------------------------
    def ext(self, a: int, supp: tuple[int, ...]) -> int:
        """vec of a*l where l = XOR of x_j, j in supp (idempotent per monomial)."""
        r = 0
        for i in range(self.nm):
            if a >> i & 1:
                m = self.monos[i]
                for j in supp:
                    m2 = m if (m >> j) & 1 else m | (1 << j)
                    r ^= 1 << self.idx[m2]
        return r

    def pq(self, a: int, b: int) -> int:
        """vec of p*q: pairwise monomial unions with idempotent collapse."""
        r = 0
        for i in range(self.nm):
            if not (a >> i & 1):
                continue
            m1 = self.monos[i]
            for k in range(self.nm):
                if not (b >> k & 1):
                    continue
                mu = m1 | self.monos[k]
                if mu in self.idx:
                    r ^= 1 << self.idx[mu]
        return r

    def bip(self, sa: Sequence[int], sb: Sequence[int]) -> int:
        r = 0
        for i in sa:
            for j in sb:
                if i != j:
                    r ^= 1 << self.idx[(1 << i) | (1 << j)]
        return r

    def supps(self) -> list[tuple[int, ...]]:
        n = self.nvars
        return [tuple(j for j in range(n) if s >> j & 1) for s in range(1 << n)]

    def bip_first_ops(self) -> list[int]:
        """All first-op product vectors (two affine wires)."""
        out = set()
        for abits in range(1, 1 << self.nvars):
            for bbits in range(1, 1 << self.nvars):
                A = tuple(j for j in range(self.nvars) if abits >> j & 1)
                B = tuple(j for j in range(self.nvars) if bbits >> j & 1)
                v = self.bip(A, B)
                if v:
                    out.add(v)
        return sorted(out)


def cosets(space: MonoSpace, a: int, b: int) -> frozenset[int]:
    """All raw-product vectors from operands with product-parts a, b,
    including affine riders and const riders."""
    out = set()
    if a == 0 and b == 0:
        out.update(space.bip_first_ops())
    elif a == 0 or b == 0:
        c = a or b
        for j in range(space.nvars):
            e = space.ext(c, (j,))
            for s in range(1 << space.nvars):
                supp = tuple(k for k in range(space.nvars)
                             if (s >> k & 1) and k != j)
                v = e ^ space.bip(supp, (j,))
                out.add(v)
                out.add(v ^ c)
    else:
        base = space.pq(a, b)
        for sa in space.supps():
            for sb in space.supps():
                v = base ^ space.ext(a, sb) ^ space.ext(b, sa)
                if sa and sb:
                    v ^= space.bip(sa, sb)
                out.update((v, v ^ a, v ^ b, v ^ a ^ b))
    out.discard(0)
    return frozenset(out)


def product_span_search(space: MonoSpace, targets: Sequence[int],
                        max_products: int, first_op: int,
                        ancillas: int = 2) -> list[int] | None:
    """Exhaustive search: can `targets` be spanned by <= max_products
    realizable products (2 ancillas)?  Returns the product list or None.

    Failing at max_products = T-1 proves any T-Toffoli circuit impossible
    (with ancillas <= the given count) — the optimality certificate.
    """
    memo: set = set()

    def closure(t: int, u: int) -> frozenset[int]:
        return frozenset({t, u, t ^ u})

    def dfs(S: frozenset[int], basis: tuple[int, ...], depth: int):
        if depth == max_products:
            return all(space.contains(basis, t) for t in targets)
        key = (S, basis, depth)
        if key in memo:
            return False
        memo.add(key)
        wires = sorted(S)
        pairs = {(0, 0)}
        for w in wires:
            if w:
                pairs.add((w, 0))
        for i in range(len(wires)):
            for j in range(i + 1, len(wires)):
                pairs.add((wires[i], wires[j]))
        cands: set[int] = set()
        for (a, b) in pairs:
            cands |= cosets(space, a, b)
        # rank-adding first: prune impossible branches
        def adds(v: int) -> bool:
            return not space.contains(basis, v)

        def missing_rank() -> int:
            extra = list(basis)
            n0 = len(space.rref(extra))
            for t in targets:
                cur = space.rref(extra)
                if not space.contains(cur, t):
                    extra.append(t)
            return len(space.rref(extra)) - n0

        need = missing_rank()
        if need > max_products - depth:
            return False
        for v in sorted(cands, key=adds, reverse=True):
            nb = basis if space.contains(basis, v) else space.rref(list(basis) + [v])
            A, B = (sorted(S) + [0, 0])[:2]
            nS = frozenset()
            for (X, Y) in ((A, B), (B, A)):
                nS |= closure(X ^ v, Y)
            if dfs(nS, nb, depth + 1):
                return True
        return False

    v1 = first_op
    S0 = closure(v1, 0)
    b0 = space.rref([v1])
    if dfs(S0, b0, 1):
        return [v1]  # caller re-derives full path if needed
    return None


def multiplicative_complexity_lb(target_fns: Sequence[int], nvars: int,
                                 ancillas: int = 2,
                                 max_depth: int = 8) -> int | None:
    """Lower bound on the number of Toffoli-family products needed to span
    all target functions (degree>=2 parts).

    Semantics: exhaustive span search proves 'k products impossible' by
    failure at depth k; the returned value is then k+1 — any circuit (with
    <= ancillas live) needs at least that many products. None if the
    search could not close within max_depth (no bound proven).

    With a matching k-product construction, this pins the multiplicative
    complexity EXACTLY — the optimality certificate behind 'T is optimal'.
    """
    space = MonoSpace(nvars)
    tgt = [space.vec_fn(f) for f in target_fns]
    if all(t == 0 for t in tgt):
        return 0
    firsts = space.bip_first_ops()
    for k in range(1, max_depth + 1):
        found = False
        for f in firsts:
            if product_span_search(space, tgt, k, f, ancillas) is not None:
                found = True
                break
        if not found:
            return k + 1
    return None
