"""Built-in tasks: in-place binary subtractor a -= b over n qubits.

The subtractor is the inverse map of the in-place adder
(|a>|b> -> |a+b mod 2^n>|b>), i.e. |a>|b> -> |a-b mod 2^n>|b>, and is
implemented by the exact gate-reversal of the CDKM adder circuit (every
gate is self-inverse). It is the same Shor-kernel arithmetic family,
held out from the operator design: the structural rewriter was derived
on the MAJ-then-UMA adder ordering, so the mirrored UMA-then-MAJ
ordering tests operator transfer, not operator recall.
"""

from __future__ import annotations

from .task import Register, Task


def subtractor_spec(n: int) -> str:
    return f"""Task: in-place reversible subtraction |a>|b> -> |a-b mod 2^n>|b> over n={n}
qubits.
Interface: preserve n-qubit b and replace n-qubit a by (a-b) mod 2^n.
Be exact for every a, b (0 <= a,b < 2^n), clean every ancilla, and leave
zero global phase.
Model: X, CNOT, Toffoli, SWAP, phase gates, and corrected X-basis
measurement; emission-time classical computation is allowed.
Objective: minimize executed Toffoli x peak live qubits (TQ-cost)."""


def subtractor_reference(a: int, b: int, n: int) -> dict[str, int]:
    return {"a": (a - b) % (2**n), "b": b}


def subtractor_task(n: int) -> Task:
    return Task(
        name=f"subtractor_inplace_{n}",
        spec_text=subtractor_spec(n),
        registers=(Register("a", n), Register("b", n)),
        reference=lambda a, b: subtractor_reference(a, b, n),
        input_domains={"a": range(2**n), "b": range(2**n)},
        failure_budget=0.0,
        toffoli_lower_bound=None,
    )


def subtractor_tasks() -> dict[int, Task]:
    return {n: subtractor_task(n) for n in (3, 4)}
