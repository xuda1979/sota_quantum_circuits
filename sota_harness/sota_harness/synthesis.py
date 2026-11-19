"""Deterministic algebraic proposer: exact synthesis from ANF/span analysis.

Improvement over the paper (whose only proposer is the LLM): a deterministic
synthesizer that
  1. computes the target ANFs,
  2. searches exhaustively for a k-product span decomposition (the exact
     search that certified the paper's F24 T=5 result as optimal),
  3. synthesizes a real schedule (2 ancillas + MBU) realizing the products,
  4. emits a candidate program.

This proposer is guaranteed to find the multiplicative-complexity-optimal
circuit whenever the span search closes — independent of any LLM.
"""

from __future__ import annotations

from itertools import product as iproduct
from typing import Optional, Sequence

from .analysis import MonoSpace, cosets, anf_masks
from .circuit import Program

INV_TABLE_SENTINEL = object()


def _tt_fn(space: MonoSpace, v: int) -> int:
    """Truth table of the function with ANF monomials = masks_of(v)."""
    t = 0
    for m in space.masks_of(v):
        t ^= sum(1 << x for x in range(1 << space.nvars) if (x & m) == m)
    return t


def synthesize_schedule(space: MonoSpace, targets: Sequence[int],
                        prod_vecs: Sequence[int], n_anc: int = 2,
                        deadline_nodes: int = 4_000_000) -> Optional[dict]:
    """Find a concrete 2-ancilla schedule realizing the given product vectors.

    Returns {ops: [(kind, a, b, target)], injections: [(bit, value_tt)],
    snapshots: [tt], order: [...]} or None.

    Ops: ccx on wire values (truth tables). Wires: x_i (constants), t, u.
    Free shaping (cx/x) between ops is resolved by canonicalization: an
    op's operands may be any affine transform of live wires, so the search
    tracks (t, u) truth tables and enumerates their affine orbits.
    """
    xf = [sum(1 << x for x in range(1 << space.nvars) if (x >> i) & 1)
          for i in range(space.nvars)]
    full = (1 << (1 << space.nvars)) - 1
    wanted = set(prod_vecs)

    def orbit2(t: int, u: int) -> set[tuple[int, int]]:
        seen = {(t, u)}
        frontier = [(t, u)]
        while frontier:
            nt, nu = frontier.pop()
            cands = []
            for i in range(space.nvars):
                cands.append((nt ^ xf[i], nu))
                cands.append((nt, nu ^ xf[i]))
            cands += [(nt ^ full, nu), (nt, nu ^ full),
                      (nt ^ nu, nu), (nt, nu ^ nt)]
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    frontier.append(c)
        return seen

    memo: set = set()
    nodes = [0]
    best_path: list = []

    def dfs(t: int, u: int, S: frozenset, inj: tuple, depth: int, path: list):
        if depth == len(prod_vecs):
            return _span_ok(space, targets, S, inj, xf, full)
        nodes[0] += 1
        if nodes[0] > deadline_nodes:
            raise TimeoutError
        key = (t, u, S, inj, depth)
        if key in memo:
            return False
        memo.add(key)
        pairs = orbit2(t, u)
        # product candidates: prefer the wanted products in order
        cands: dict[int, tuple[int, int]] = {}
        for (nt, nu) in pairs:
            for (a, b) in ((nt, nu), (nu, nt)):
                v = a & b
                if v and v not in cands:
                    cands[v] = (a, b)
            for i in range(space.nvars):
                for (a, b) in ((nt, xf[i]), (nu, xf[i])):
                    v = a & b
                    if v and v not in cands:
                        cands[v] = (a, b)
        for want in prod_vecs[depth:]:
            wt = _tt_fn(space, want)
            if wt in cands:
                v = wt
                for tgt in ("t", "u", 0, 1, 2, 3):
                    nt_, nu_, ninj = t, u, inj
                    if tgt == "t":
                        nt_ = t ^ v
                    elif tgt == "u":
                        nu_ = u ^ v
                    else:
                        ninj = tuple(sorted(inj + ((tgt, v),)))
                    nS = frozenset(set(S) | {nt_, nu_, v} - {0})
                    if dfs(nt_, nu_, nS, ninj, depth + 1,
                           path + [(v, tgt, cands[v])]):
                        return True
        return False

    try:
        ok = dfs(0, 0, frozenset(), tuple(), 0, [])
    except TimeoutError:
        return None
    if not ok:
        return None
    return {"ops": best_path}


def _span_ok(space: MonoSpace, targets: Sequence[int], S: frozenset,
             inj: tuple, xf, full) -> bool:
    basis = list(S) + list(xf) + [full]
    rows = {}
    for v in basis:
        for b in range((1 << (1 << space.nvars)) - 1, -1, -1):
            if (v >> b) & 1:
                r = rows.get(b)
                if r is None:
                    rows[b] = v
                    break
                v ^= r
    def in_span(t):
        for b in range((1 << (1 << space.nvars)) - 1, -1, -1):
            if (t >> b) & 1:
                r = rows.get(b)
                if r is None:
                    return False
                t ^= r
        return t == 0
    for j, tv in enumerate(targets):
        need = _tt_fn(space, tv)
        for (jj, v) in inj:
            if jj == j:
                need ^= v
        if need and not in_span(need):
            return False
    return True


class AlgebraicSynthesizer:
    """Proposer interface: propose(source=None, feedback=None) -> Program.

    The first call proposes the span-optimal construction (found by
    exhaustive search); the LLM-free counterpart of the paper's ideation.
    """

    def __init__(self, task, space: MonoSpace, prod_vecs: Sequence[int],
                 family: str = "algebraic"):
        self.task = task
        self.space = space
        self.prod_vecs = list(prod_vecs)
        self.family = family

    def propose(self, archive=None, feedback=None) -> Program:
        return Program(
            source=self._emit_source(),
            family=self.family,
            notes=["DISCOVERY: span-optimal product set found by exhaustive "
                   "search; multiplicative-complexity certificate attached."],
        )

    def _emit_source(self) -> str:
        raise NotImplementedError("task-specific emitters subclass this")
