"""Built-in tasks: in-place binary adder a += b over n qubits.

The adder family is the canonical "classical algorithm implemented as a
quantum circuit" (the integer-arithmetic kernel inside Shor's algorithm).
In-place addition is reversible: |a>|b> -> |a+b mod 2^n>|b>.
"""

from __future__ import annotations

from .task import Register, Task


def adder_spec(n: int) -> str:
    return f"""Task: in-place reversible addition |a>|b> -> |a+b mod 2^n>|b> over n={n}
qubits.
Interface: preserve n-qubit b and replace n-qubit a by (a+b) mod 2^n.
Be exact for every a, b (0 <= a,b < 2^n), clean every ancilla, and leave
zero global phase.
Model: X, CNOT, Toffoli, SWAP, phase gates, and corrected X-basis
measurement; emission-time classical computation is allowed.
Objective: minimize executed Toffoli x peak live qubits (TQ-cost)."""


def adder_reference(a: int, b: int, n: int) -> dict[str, int]:
    return {"a": (a + b) % (2**n), "b": b}


def adder_task(n: int) -> Task:
    return Task(
        name=f"adder_inplace_{n}",
        spec_text=adder_spec(n),
        registers=(Register("a", n), Register("b", n)),
        reference=lambda a, b: adder_reference(a, b, n),
        input_domains={"a": range(2**n), "b": range(2**n)},
        failure_budget=0.0,
        toffoli_lower_bound=None,  # set by M2 certificate work
    )


def adder_tasks() -> dict[int, Task]:
    return {n: adder_task(n) for n in (3, 4)}
