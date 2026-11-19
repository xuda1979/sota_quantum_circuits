"""Built-in tasks: F24 inversion (the paper's toy demonstration) and the
Table I tasks share spec text with the replication scripts."""

from __future__ import annotations

from .task import Register, Task

INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]

F24_SPEC = """Task: |x>|h> -> |x>|h XOR x^-1> in F24, with 0^-1 = 0 and
p(X) = X^4 + X + 1.
Interface: preserve four-qubit x and XOR the result into arbitrary
four-qubit h. Be exact for every x, h, clean every ancilla, and leave
zero global phase.
Model: X, CNOT, Toffoli, SWAP, phase gates, and corrected X-basis
measurement; emission-time classical computation is allowed.
Objective: minimize executed Toffoli x peak live qubits (TQ-cost)."""


def f24_reference(x: int, h: int) -> dict[str, int]:
    return {"x": x, "h": h ^ INV[x]}


def f24_task() -> Task:
    return Task(
        name="f24_inversion",
        spec_text=F24_SPEC,
        registers=(Register("x", 4), Register("h", 4)),
        reference=f24_reference,
        input_domains={"x": range(16), "h": range(16)},
        failure_budget=0.0,
        toffoli_lower_bound=5,  # MC certificate: 4 products impossible (2 anc)
    )


def f24_task_no_lb() -> Task:
    """Same task without the lower bound (for replicating the paper's
    run, which had no certificate)."""
    t = f24_task()
    t.toffoli_lower_bound = None
    return t
