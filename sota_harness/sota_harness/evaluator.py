"""The independent evaluator: ground truth for validity and cost.

Mirrors the paper's evaluator semantics exactly (per its Methods):
  - outputs must match the reference on every test shot,
  - global phase must be zero (or the task's g),
  - every non-register qubit must return to |0>,
  - cost = executed Toffoli count x peak live qubits.

Improvements over the paper's loop feedback (which prints only
``valid: executed_toffoli=...; qubits=...`` or ``invalid: ancilla t not
returned to |0> at free``):
  - categorized diagnostics (WRONG_OUTPUT / DIRTY_ANCILLA / PHASE /
    CRASH / MALFORMED) with the first failing shot named,
  - per-register output tables for quick diffing,
  - exact T, Q, and TQ on valid circuits plus a Toffoli lower bound
    certificate when the task provides one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .circuit import CircuitContext, run_program, CleanlinessError
from .sim import batch_simulate


@dataclass
class Verdict:
    valid: bool
    category: str = "OK"            # OK | WRONG_OUTPUT | DIRTY_ANCILLA |
                                    # PHASE | CRASH | MALFORMED
    message: str = ""
    toffoli: float = 0.0            # executed T per shot (exact avg)
    qubits: int = 0                 # peak live qubits
    tq: float = float("inf")
    shot: int = -1                  # first failing shot index (-1 if none)
    meta: dict = field(default_factory=dict)

    def one_line(self) -> str:
        if self.valid:
            return (f"valid: executed_toffoli={self.toffoli:g}; "
                    f"qubits={self.qubits}; TQ={self.tq:g}")
        return f"invalid[{self.category}]: {self.message}"


class Evaluator:
    """Evaluates candidate programs against a Task."""

    def __init__(self, task, seed: int = 0):
        self.task = task
        self.seed = seed

    def evaluate(self, source: str) -> Verdict:
        try:
            ctx = run_program(source, list(self.task.registers_by_name))
        except Exception as e:  # noqa: BLE001 - diagnostics are the point
            return Verdict(False, "CRASH", f"build failed: {type(e).__name__}: {e}")

        missing = [r for r in self.task.registers_by_name
                   if r not in ctx.registers]
        if missing:
            return Verdict(False, "MALFORMED",
                           f"missing interface register(s): {missing}")

        shots = self.task.example_shots()
        try:
            res = batch_simulate(ctx, shots)
        except CleanlinessError as e:
            # simulator strict mode surfaces dirty ancillas / phase garbage
            msg = str(e).splitlines()[0]
            cat = "PHASE" if "phase" in msg else "DIRTY_ANCILLA"
            return Verdict(False, cat, msg)
        except Exception as e:  # noqa: BLE001
            return Verdict(False, "CRASH", f"simulation failed: {e}")

        # output check
        for i, shot in enumerate(shots):
            expected = self.task.reference(**shot)
            for name, want in expected.items():
                got = res.outputs[name][i]
                if got != want:
                    return Verdict(
                        False, "WRONG_OUTPUT",
                        f"register {name}: got {got}, want {want} "
                        f"on shot {i} {shot}",
                        shot=i,
                    )

        T = res.toffoli_per_shot
        Q = ctx.b.peak_qubits
        v = Verdict(True, toffoli=T, qubits=Q, tq=T * Q)
        lb = getattr(self.task, "toffoli_lower_bound", None)
        if lb is not None:
            v.meta["toffoli_lower_bound"] = lb
            v.meta["certified_optimal"] = (T <= lb)
        return v
