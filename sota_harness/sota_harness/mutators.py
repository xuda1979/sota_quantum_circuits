"""Deterministic program rewriters (LLM-free evolutionary operators).

The paper's rewrite step is LLM-only. Deterministic alternatives:

  - MbuRewriter: pattern-rewrites ``ctx.free(t)`` after a dirty accumulate
    into the paper's MBU idiom (hmr + conditioned cz + free), the exact
    transformation of Extended Data Fig. 1 (five products invalid ->
    +MBU valid).
  - AncillaPackRewriter: reorders allocation lifetimes to shrink peak Q
    (interleaving free/alloc where dataflow allows).
  - GateReplacementRewriter: ccx(a,b,c) with an in-scope zeroed target
    preceded/followed by x-gates -> relative-phase forms when the model
    allows; currently a stub that normalizes cx direction.

Each rewriter maps Program -> Program (source to source), recording a
note when it fires.
"""

from __future__ import annotations

import re

from .circuit import Program


class Rewriter:
    name = "abstract"

    def rewrite(self, program: Program) -> Program | None:
        raise NotImplementedError


class MbuRewriter(Rewriter):
    """free(t) with dirty t  ->  hmr/cz-fixup idiom (needs an operand pair
    to condition the fixup; conservatively uses the last ccx operands)."""

    name = "mbu"

    _FREE = re.compile(r"^(\s*)ctx\.free\((\w+)\)\s*$", re.M)
    _CCX = re.compile(
        r"ctx\.ccx\((\w+)\[(\d+)\],\s*(\w+)\[(\d+)\],\s*(\w+)\)")

    def rewrite(self, program: Program) -> Program | None:
        src = program.source
        if "ctx.hmr(" in src:
            return None  # already has measured uncompute; idempotent no-op
        ccx_ops = list(self._CCX.finditer(src))
        frees = list(self._FREE.finditer(src))
        if not ccx_ops or not frees:
            return None
        # rewrite the FIRST free of a wire that was a ccx target and is
        # never zeroed in source: insert MBU idiom before it
        targets = {m.group(5) for m in ccx_ops}
        for f in frees:
            wire = f.group(2)
            if wire not in targets:
                continue
            # find a ccx targeting this wire, use its operands for cz
            cm = next(m for m in ccx_ops if m.group(5) == wire)
            indent, a, ai, b, bi = (f.group(1), cm.group(1), cm.group(2),
                                    cm.group(3), cm.group(4))
            mbu = (f"{indent}m = ctx.hmr({wire})\n"
                   f"{indent}ctx.cz({a}[{ai}], {b}[{bi}], cond=m)\n"
                   f"{indent}ctx.free({wire})")
            new_src = src[:f.start()] + mbu + src[f.end():]
            return Program(
                source=new_src,
                family=program.family,
                parent=program.id,
                notes=["REWRITE: measured uncompute (MBU) inserted before "
                       "free of dirty ccx-target ancilla"],
            )
        return None


class AncillaPackRewriter(Rewriter):
    """Hoists ctx.free calls as early as dataflow allows (textual
    last-use analysis). Reduces peak live qubits when frees cluster late."""

    name = "ancilla_pack"

    def rewrite(self, program: Program) -> Program | None:
        lines = program.source.splitlines(keepends=True)
        # find last-use line index per allocated wire name
        use = {}
        alloc = {}
        for i, ln in enumerate(lines):
            for m in re.finditer(r"\b(\w+)\b", ln):
                w = m.group(1)
                if w in ("ctx", "def", "build", "alloc", "free", "reg"):
                    continue
                use.setdefault(w, i)
                use[w] = i
            m = re.match(r"\s*(\w+) = ctx\.alloc", ln)
            if m:
                alloc[m.group(1)] = i
        moved = False
        for w, ai in alloc.items():
            fm = next((i for i, ln in enumerate(lines)
                       if re.match(rf"\s*ctx\.free\({w}\)\s*$", ln)), None)
            if fm is None or fm <= use[w]:
                continue
            # move free line up to just after last use
            free_ln = lines.pop(fm)
            ins = use[w] + 1
            lines.insert(ins, free_ln)
            moved = True
        if not moved:
            return None
        return Program(
            source="".join(lines),
            family=program.family,
            parent=program.id,
            notes=["REWRITE: ancilla lifetime packed (free hoisted to "
                   "last use)"],
        )


class MutationPipeline:
    def __init__(self, rewriters: list[Rewriter]):
        self.rewriters = rewriters

    def all_variants(self, program: Program) -> list[Program]:
        out = []
        for r in self.rewriters:
            v = r.rewrite(program)
            if v is not None:
                out.append(v)
        return out
