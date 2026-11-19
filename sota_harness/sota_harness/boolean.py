"""Exact Boolean-function algebra for synthesis and lower bounds.

Truth tables are ints (bit x = f(x) of the input assignment x). ANF vectors
pack the degree>=2 monomials into a coordinate vector (int) — the parts that
cost Toffolis; degree<=1 parts are free Clifford gates.

Core results used across the harness:

- ``anf_vec(tt)``: the deg>=2 ANF coordinate vector of a truth table.
- ``rank_lb``: any circuit's wires have deg>=2 parts in the span of the
  deg>=2 parts of its <=T non-Clifford outputs, so
  T >= dim span{deg2(outputs)} — a rigorous Toffoli lower bound.
- ``menu``: every realizable first-op product (AND of two affine forms on
  distinct wires), used by the exact DFS in synth.py.
"""

from __future__ import annotations

from itertools import combinations


class BoolSpace:
    """Truth-table algebra over n input bits (and m output bits)."""

    def __init__(self, nvars: int):
        self.nvars = nvars
        self.size = 1 << nvars
        self.full = (1 << self.size) - 1
        self.x = [sum(1 << v for v in range(self.size) if (v >> i) & 1)
                  for i in range(nvars)]
        self.monos = [m for m in range(1, self.size)
                      if bin(m).count("1") >= 2]
        self.idx = {m: i for i, m in enumerate(self.monos)}

    # -- ANF ---------------------------------------------------------------
    def anf(self, tt: int) -> dict[int, bool]:
        f = [(tt >> v) & 1 for v in range(self.size)]
        for i in range(self.nvars):
            for m in range(self.size):
                if m >> i & 1:
                    f[m] ^= f[m ^ (1 << i)]
        return {m: True for m in range(self.size) if f[m]}

    def anf_vec(self, tt: int) -> int:
        """Coordinate vector of the deg>=2 ANF monomials."""
        v = 0
        for m in self.anf(tt):
            if m in self.idx:
                v |= 1 << self.idx[m]
        return v

    def tt_of_vec(self, vec: int) -> int:
        """Truth table of a deg>=2 ANF vector (no affine/const part)."""
        t = 0
        for i, m in enumerate(self.monos):
            if vec >> i & 1:
                t ^= sum(1 << v for v in range(self.size) if (v & m) == m)
        return t

    def residual_affine(self, r: int) -> bool:
        return self.anf_vec(r) == 0

    def affine_tt(self, supp: int, const: int) -> int:
        t = self.full if const else 0
        for j in range(self.nvars):
            if supp >> j & 1:
                t ^= self.x[j]
        return t

    # -- GF(2) linear algebra on coordinate vectors -------------------------
    @staticmethod
    def rref(vecs) -> tuple[int, ...]:
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
    def contains(basis, v: int) -> bool:
        for b in basis:
            v = min(v, v ^ b)
        return v == 0

    @classmethod
    def dim(cls, vecs) -> int:
        return len(cls.rref(vecs))

    # -- products -------------------------------------------------------------
    def product_vecs(self, a_tt: int, b_tt: int) -> int:
        """deg>=2 vector of a&b."""
        return self.anf_vec(a_tt & b_tt)

    def menu(self) -> list[tuple[tuple, tuple, int]]:
        """All AND-of-two-affine-forms products on distinct input wires.

        Returns (formA, formB, tt) with form = (supp, const). Distinct
        truth tables only; these are exactly the realizable first-op
        products (both operands shaped from distinct input wires).
        """
        seen: set[int] = set()
        out = []
        wires = [(1 << w,) for w in range(self.nvars)]
        forms_per_wire = []
        for w in range(self.nvars):
            forms = []
            for r in range(self.nvars + 1):
                for supp in combinations(range(self.nvars), r):
                    s = 0
                    for j in supp:
                        s |= 1 << j
                    if not (s >> w) & 1:
                        continue
                    for c in (0, 1):
                        forms.append((s, c))
            forms_per_wire.append(forms)
        for w1 in range(self.nvars):
            for w2 in range(w1 + 1, self.nvars):
                for f1 in forms_per_wire[w1]:
                    for f2 in forms_per_wire[w2]:
                        t1 = self.affine_tt(*f1)
                        t2 = self.affine_tt(*f2)
                        p = t1 & t2
                        if p and p not in seen:
                            seen.add(p)
                            out.append(((w1,) + f1, (w2,) + f2, p))
        return out


def rank_lb(space: BoolSpace, output_tts) -> int:
    """Rigorous Toffoli lower bound: every wire's deg>=2 part lies in the
    span of the deg>=2 parts of the non-Clifford outputs (each op XORs an
    AND of earlier wire values into its target), so
    T >= dim span{deg2(output truth tables)}."""
    return space.dim([space.anf_vec(t) for t in output_tts])
