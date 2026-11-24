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


class RedundantPairElimRewriter(Rewriter):
    """Ripple-carry structural mutation: drop the redundant top MAJ/UMA pair.

    A full CDKM-style adder runs n MAJ stages then n UMA stages on a single
    carry ancilla, but the carry-out c_n produced by the top MAJ stage is
    immediately discarded by the top UMA stage (mod-2^n addition), and the
    top sum bit is fully determined by c_{n-1}: s_{n-1} = c_{n-1} ^ a_{n-1}
    ^ b_{n-1}.  Rewriting ``range(n)`` -> ``range(n-1)`` on both loops and
    XORing the chain carry straight into the top sum bit eliminates one
    MAJ + one UMA (2 Toffolis) at identical qubit count, keeping the carry
    ancilla clean (carries c_1..c_{n-1} are still computed and uncomputed
    pairwise).  This is the 2n-2 Toffoli order of Takahashi-Kunihiro (2005),
    expressed here as a *general pattern rewrite* over the family of
    ripple-carry sources the harness produces.
    """

    name = "pair_elim"

    def rewrite(self, program: Program) -> Program | None:
        lines = program.source.splitlines(keepends=True)

        # Forward loop: "for i in range(N):" + body call "fn(i)".  Trailing
        # comments after ")" are allowed (the CDKM template carries them).
        fwd = None  # (line_idx, indent, ivar, n, fn_f)
        for i, ln in enumerate(lines):
            m = re.match(r"^(\s*)for\s+(\w+)\s+in\s+range\((\d+)\):", ln)
            if not m or i + 1 >= len(lines):
                continue
            cm = re.match(r"^\s+(\w+)\(" + re.escape(m.group(2)) +
                          r"\)(\s*#.*)?\s*$", lines[i + 1])
            if not cm:
                continue
            fwd = (i, m.group(1), m.group(2), int(m.group(3)), cm.group(1))
            break
        if fwd is None:
            return None
        f_i, f_ind, ivar, n, fn_f = fwd
        if n < 2:
            return None

        # Backward loop over the same induction var:
        # "for i in range(N - 1, -1, -1):" + body call "fn2(i)".
        bwd = None  # (line_idx, fn_b)
        bpat = re.compile(r"^\s*for\s+" + re.escape(ivar) +
                          r"\s+in\s+range\(" + str(n) +
                          r"\s*-\s*1,\s*-1,\s*-1\):")
        for j in range(f_i + 2, len(lines)):
            if not bpat.match(lines[j]) or j + 1 >= len(lines):
                continue
            cm = re.match(r"^\s+(\w+)\(" + re.escape(ivar) +
                          r"\)(\s*#.*)?\s*$", lines[j + 1])
            if not cm:
                continue
            bwd = (j, cm.group(1))
            break
        if bwd is None:
            return None
        b_j, fn_b = bwd
        if fn_f == fn_b:
            return None

        # Forward fn def body: lines strictly deeper than the "def" line.
        fdef = None
        for k, ln in enumerate(lines):
            m = re.match(r"^(\s*)def\s+" + re.escape(fn_f) + r"\(\w+\):", ln)
            if m:
                fdef = (k, len(m.group(1)))
                break
        if fdef is None:
            return None
        d_i, d_ind = fdef
        body = []
        for ln in lines[d_i + 1:]:
            if ln.strip() and not ln.startswith(" " * (d_ind + 1)):
                break
            body.append(ln)
        body_txt = "".join(body)

        # Carry wire: the only bare (unindexed) name in the fn body's
        # ctx calls, e.g. c in "ctx.cx(c, b[i])" / "ctx.ccx(a[i], b[i], c)".
        bare = set()
        for arglist in re.findall(
                r"ctx\.(?:cx|ccx|swap|x|cz|free)\(([^)]*)\)", body_txt):
            for tok in re.finditer(r"[(),\s](\w+)(?![\[\w])", arglist):
                bare.add(tok.group(1))
        bare -= {"ctx", "cond"}
        if len(bare) != 1:
            return None
        carry = bare.pop()

        # b/a registers from the first two-indexed-arg cx in the fn body
        # (CDKM maj starts with ctx.cx(b[i], a[i])).
        m0 = re.search(r"ctx\.cx\((\w+)\[\w+\],\s*(\w+)\[\w+\]\)", body_txt)
        if not m0:
            return None
        breg, areg = m0.group(1), m0.group(2)

        top = n - 1
        # 1) forward loop: range(n) -> range(n-1)
        lines[f_i] = re.sub(r"range\(" + str(n) + r"\):",
                            f"range({n - 1}):", lines[f_i], count=1)
        # 2) XOR the chain carry into the top sum bit, between the loops
        #    (at the for-line indent, so it runs once, outside both loops)
        ins = (f"{f_ind}ctx.cx({carry}, {areg}[{top}])"
               f"              # a{top} ^= c{top}\n"
               f"{f_ind}ctx.cx({breg}[{top}], {areg}[{top}])"
               f"              # a{top} = s{top}\n")
        lines.insert(b_j, ins)
        # 3) backward loop: range(n - 1, -1, -1) -> range(n - 2, -1, -1)
        lines[b_j + 1] = re.sub(
            r"range\(" + str(n) + r"\s*-\s*1,",
            f"range({n - 2},", lines[b_j + 1], count=1)

        return Program(
            source="".join(lines),
            family=program.family,
            parent=program.id,
            notes=["REWRITE: redundant top MAJ/UMA pair eliminated "
                   "(2n-2 ripple-carry; skip discarded carry-out)"],
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
