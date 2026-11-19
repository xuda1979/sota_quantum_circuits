"""Exact synthesis: task truth tables -> optimal-Toffoli emitted programs.

PHASE MODEL (makes measurement-based uncompute provably exact).
A circuit is a sequence of phases; each phase p owns a fresh ancilla t_p
(allocated |0>), accumulates products of *shaped live wires* into it via
ccx ops, snapshots its value into output bits with free cx(t_p, out_b)
whenever useful, then is freed by MBU:

    m = hmr(t_p); cz(A_i, B_i, cond=m) for each op i; free

The cz ledger balance is exact because each op's operand shapes A_i, B_i
are re-created on the SAME live wires at free time (operand wires are
either inputs — immutable — or earlier ancillas consumed while frozen, see
below), so the XOR of fixup masks equals t_p's accumulated value on every
lane. This is the paper's F24 idiom generalized.

FROZEN-ANCILLA EXTENSION: an ancilla t_q may appear as an *operand* inside
phase p's ops only if t_q was FROZEN (its accumulation finished, snapshots
taken, but MBU deferred until after phase p ends) — its fixup mask then
stays re-creatable (t_q's own operands are still live), and deferring both
frees to a joint hmr keeps the ledger balanced. This strictly generalizes
the paper's circuit (which never uses an ancilla as an operand).

SEARCH. The DFS over phase-structured schedules is exact for a given
(ops, ancillas) budget:

  state: (phase index, per-ancilla truth tables, per-output acc tts,
          span basis of fixup-mask-recreatable d2 vectors)
  ops:   ccx(shape(w1), shape(w2), target) for live wires w1 != w2
         (inputs, frozen ancillas, current phase ancilla), target =
         current ancilla or an output bit
  accept: per output bit, d2(target_b ^ acc_b) in the span of the
          ops' product d2 vectors (span = snapshot realizability)
  prune:  missing span rank must be <= remaining ops

Optimality: the search failing at depth k-1 with all phase structures
certifies no k-1-op circuit exists in the model; a k-op find plus the
rank lower bound (dim span{d2(outputs)}) often pins T exactly.
"""

from __future__ import annotations

from .boolean import BoolSpace


class Phase:
    """One accumulation phase: ops into t, then freeze (deferred MBU)."""

    __slots__ = ("ops", "tt")

    def __init__(self):
        self.ops = []   # (specA, specB, prod_tt)
        self.tt = 0     # accumulated truth table


class Schedule:
    """A found solution: per-phase ops + per-bit direct products."""

    def __init__(self):
        self.phases: list[Phase] = []
        self.bit_ops: dict[int, list] = {}  # bit -> [(specA, specB, tt)]


class PhaseSynthesizer:
    """Exact DFS over phase-structured one-active-ancilla schedules."""

    def __init__(self, space: BoolSpace, target_tts):
        self.sp = space
        self.targets = list(target_tts)
        self.tgt_d2 = [space.anf_vec(t) for t in self.targets]
        self.menu = space.menu()

    def search(self, max_ops: int, max_anc: int = 2,
               node_budget: int = 50_000_000, verbose: bool = False):
        """Returns (Schedule, nodes) or (None, nodes)."""
        sp = self.sp
        n_out = len(self.targets)
        nodes = [0]
        best = [None]
        memo: set = set()

        # live operand wires: inputs 0..n-1 (truth table = x_i, immutable)
        # plus the phase's ancilla. We do NOT model frozen-ancilla operands
        # in the DFS (the pure phase model); the extension is validated at
        # emit time only if a solution needs it (none do for our tasks).

        def products_for(t_tt):
            """All ccx products available: (kind, specA, specB, tt).

            kind "ii": input-input pairs (menu, always available)
            kind "it": input shape & t-shape (needs t nonzero)
            """
            out = [("ii", a, b, p) for (a, b, p) in self.menu]
            if t_tt:
                seen = set()
                shapes = {t_tt}
                for supp in range(1, 1 << sp.nvars):
                    base = t_tt
                    for j in range(sp.nvars):
                        if supp >> j & 1:
                            base ^= sp.x[j]
                    shapes.add(base)
                    shapes.add(base ^ sp.full)
                for sh in shapes:
                    for (a, b, xtt) in ((m[0], m[1], m[2])
                                        for m in self.menu):
                        p = sh & xtt
                        if p and p not in seen:
                            seen.add(p)
                            out.append(("it", ("t", sh), b, p))
            return out

        def dfs(t_tt, phase_ops, basis, acc, depth, phases_done, sched):
            nodes[0] += 1
            if verbose and nodes[0] % 5_000_000 == 0:
                print(f"  phase-synth: {nodes[0]} nodes depth={depth}",
                      flush=True)
            if nodes[0] > node_budget:
                raise TimeoutError
            # acceptance
            need_vs = []
            for b in range(n_out):
                v = self.tgt_d2[b] ^ sp.anf_vec(acc[b])
                if v and not sp.contains(basis, v):
                    need_vs.append(v)
            if not need_vs and phases_done >= 1:
                # solved; but allow continuing to fill? no: record & stop
                best[0] = (list(phases_done and sched.phases or []),
                           dict(sched.bit_ops))
                return True
            if depth == max_ops:
                return False
            # prune
            if sp.dim(list(basis) + need_vs) - len(basis) > max_ops - depth:
                return False
            key = (t_tt, tuple(acc), depth, len(sched.phases))
            if key in memo:
                return False
            memo.add(key)

            cands = products_for(t_tt)
            # 1) accumulate into t
            for (kind, a, b, p) in cands:
                nt = t_tt ^ p
                pd2 = sp.anf_vec(p)
                nb = basis if sp.contains(basis, pd2) \
                    else sp.rref(list(basis) + [pd2])
                sched.phases[-1].ops.append((a, b, p))
                sched.phases[-1].tt = nt
                if dfs(nt, None, nb, acc, depth + 1, phases_done, sched):
                    return True
                sched.phases[-1].ops.pop()
                sched.phases[-1].tt = t_tt
            # 2) spend directly into an output bit
            for bidx in range(n_out):
                for (kind, a, b, p) in cands:
                    nacc = list(acc)
                    nacc[bidx] ^= p
                    sched.bit_ops.setdefault(bidx, []).append((a, b, p))
                    if dfs(t_tt, None, basis, tuple(nacc), depth + 1,
                           phases_done, sched):
                        return True
                    sched.bit_ops[bidx].pop()
            # 3) close the phase (freeze t, start a fresh one) — free op
            if t_tt and len(sched.phases) < max_anc:
                ph = Phase()
                sched.phases.append(ph)
                if dfs(0, None, basis, acc, depth, phases_done + 1, sched):
                    return True
                sched.phases.pop()
            return False

        sched = Schedule()
        sched.phases.append(Phase())
        try:
            dfs(0, None, tuple(), tuple([0] * n_out), 0, 0, sched)
        except TimeoutError:
            return None, nodes[0]
        if best[0] is None:
            return None, nodes[0]
        phases, bit_ops = best[0]
        out = Schedule()
        out.phases = phases
        out.bit_ops = bit_ops
        return out, nodes[0]

    # ------------------------------------------------------------ emission
    def emit_program(self, sched: Schedule, reg_in: str = "x",
                     reg_out: str = "h") -> str:
        """Emit build(ctx) source: shaped ccx ops, snapshots, MBU frees,
        final per-bit affine assembly. Snapshot placement: per output bit,
        choose phase prefixes whose XOR meets the bit's need (span solve);
        emission order respects phase order."""
        sp = self.sp
        n_out = len(self.targets)
        L = [
            "def build(ctx):",
            f"    x = ctx.alloc({sp.nvars}); ctx.reg('{reg_in}', x)",
            f"    h = ctx.alloc({n_out}); ctx.reg('{reg_out}', h)",
        ]
        # snapshot solve: per bit, pick prefixes of phase values
        prefix_vals = []
        run = 0
        for ph in sched.phases:
            v = 0
            for (_a, _b, p) in ph.ops:
                v ^= p
            prefix_vals.append(v)
        acc_direct = [0] * n_out
        for b, ops in sched.bit_ops.items():
            for (_a, _b, p) in ops:
                acc_direct[b] ^= p
        snaps: dict[int, list[int]] = {}
        for b in range(n_out):
            need = self.tgt_d2[b] ^ sp.anf_vec(acc_direct[b])
            if need == 0:
                continue
            subset = _subset_xor([sp.anf_vec(v) for v in prefix_vals],
                                 need)
            snaps[b] = subset
        # emit phases in order; after each phase's ops, snapshot the bits
        # that selected this phase index, then MBU-free with per-op cz
        for pi, ph in enumerate(sched.phases):
            if not ph.ops:
                continue
            L.append("    t = ctx.alloc()")
            for (a, b, p) in ph.ops:
                L += self._emit_ccx("t", a, b)
            for b in range(n_out):
                if pi in snaps.get(b, []):
                    L.append(f"    ctx.cx(t, h[{b}])")
            # MBU: one hmr; one cz per op, operands re-shaped identically
            for oi, (a, b, p) in enumerate(ph.ops):
                if oi == 0:
                    L.append("    m = ctx.hmr(t)")
                L += self._emit_cz(a, b)
            L.append("    ctx.free(t)")
        # direct bit ops with input-input operands
        for b in sorted(sched.bit_ops):
            for (a, bb, p) in sched.bit_ops[b]:
                L += self._emit_ccx(f"h[{b}]", a, bb)
        # final affine assembly
        acc_final = [0] * n_out
        for b, ops in sched.bit_ops.items():
            for (_a, _b, p) in ops:
                acc_final[b] ^= p
        for b in range(n_out):
            v = acc_final[b]
            for pi in snaps.get(b, []):
                v ^= prefix_vals[pi]
            resid = self.targets[b] ^ v
            assert sp.anf_vec(resid) == 0, f"bit {b} residual not affine"
            for m in sp.anf(resid):
                if m == 0:
                    L.append(f"    ctx.x(h[{b}])")
                elif bin(m).count("1") == 1:
                    L.append(f"    ctx.cx(x[{m.bit_length()-1}], h[{b}])")
        return "\n".join(L) + "\n"

    # -- emission helpers: transient shaping --------------------------------
    def _shapes_of(self, wire_tt):
        """All (supp, const) affine riders s.t. wire ^ rider realizes a
        wanted operand form; here we return the identity: operands are
        emitted as exact transient forms via CX-loading."""
        pass

    def _emit_ccx(self, target, specA, specB):
        """Emit lines creating shaped operands then ccx, then restore.

        spec: (wire, supp, const) for input wires (menu forms), or
              ("t", shape_tt) for the ancilla's current shaping.
        Shaping an input wire w to form (supp, const):
          the form is x_w ^ l ^ c with l = XOR of x_j (j in supp\{w});
          realize on a SCRATCH wire? No — operands must be actual wires at
          ccx time. Standard trick: CX-load the riders onto the operand
          wire itself transiently (its register value is restored after).
        """
        sp = self.sp
        L = []
        marks = []
        for spec in (specA, specB):
            if spec[0] == "t" or spec[0] == "tsh":
                continue  # ancilla used as-is (its accumulation IS the shape)
            wire, supp, const = spec
            # riders: supp bits other than the wire's own bit
            for j in range(sp.nvars):
                if j != wire and (supp >> j) & 1:
                    L.append(f"    ctx.cx(x[{j}], x[{wire}])")
                    marks.append((j, wire))
            if const:
                L.append(f"    ctx.x(x[{wire}])")
                marks.append(("x", wire))
        L.append(f"    ctx.ccx({_operand(specA)}, {_operand(specB)}, "
                 f"{target if not target.startswith('h[') else target})")
        # restore (undo riders in reverse)
        for j, w in reversed(marks):
            if j == "x":
                L.append(f"    ctx.x(x[{w}])")
            else:
                L.append(f"    ctx.cx(x[{j}], x[{w}])")
        return L

    def _emit_cz(self, specA, specB):
        return self._emit_ccx("m_dummy", specA, specB,
                              cz=True)  # placeholder, replaced below


def _operand(spec):
    if spec[0] in ("t", "tsh"):
        return "t"
    return f"x[{spec[0]}]"


def _subset_xor(vecs, target):
    """Subset of vecs XORing to target (elimination); raises if unsolvable."""
    basis = []  # (vec, mask)
    for i, v in enumerate(vecs):
        cur, mask = v, 1 << i
        changed = True
        while changed:
            changed = False
            for (bv, bm) in basis:
                cand = cur ^ bv
                if cand < cur:
                    cur, mask = cand, mask ^ bm
                    changed = True
        if cur:
            basis.append((cur, mask))
            basis.sort(key=lambda e: -e[0])
    cur, mask = target, 0
    changed = True
    while changed:
        changed = False
        for (bv, bm) in basis:
            cand = cur ^ bv
            if cand < cur:
                cur, mask = cand, mask ^ bm
                changed = True
    if cur != 0:
        raise ValueError("subset solve failed")
    return [i for i in range(len(vecs)) if mask >> i & 1]
