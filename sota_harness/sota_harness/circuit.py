"""Candidate programs and the circuit-building API they code against.

A candidate is a Python source string (the paper's representation: circuit
GENERATING programs, so evolutionary rewrites act on code). The program
defines ``build(ctx)`` using the tiny API below, mirroring the paper's
alloc/free/ccx/free-Clifford/MBU model:

    x = ctx.alloc(4); h = ctx.alloc(4); t = ctx.alloc()
    ctx.ccx(x[0], x[1], t)
    ctx.cx(t, h[0])
    m = ctx.hmr(t)                  # X-basis measurement, demolishes t to 0
    ctx.cz(x[0], x[1], cond=m)      # conditioned phase fixup
    ctx.free(t)

Backend: the official repo's Builder/Simulator (exact semantics).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

_UNCOMPUTE_PATH = "/home/david/software/sota_quantum_circuits/uncompute"
if _UNCOMPUTE_PATH not in sys.path:
    sys.path.insert(0, _UNCOMPUTE_PATH)

from uncompute import Builder, Simulator  # noqa: E402
from uncompute.sim import CleanlinessError  # noqa: E402


@dataclass
class Program:
    """A candidate circuit-generating program."""
    source: str
    family: str
    parent: Optional[str] = None          # program id of the parent
    notes: list[str] = field(default_factory=list)  # DISCOVERY notes
    id: str = ""

    _counter = 0

    def __post_init__(self):
        if not self.id:
            Program._counter += 1
            self.id = f"p{Program._counter:04d}"


class CircuitContext:
    """The API surface a candidate program builds against (wraps Builder)."""

    def __init__(self, registers: list[str]):
        self.b = Builder()
        self.registers: dict[str, list[int]] = {}
        self._reg_names = registers  # names that count as interface
        self.wires: dict[str, list[int]] = {}

    # -- allocation ----------------------------------------------------
    def alloc(self, n: int | None = None):
        """alloc() -> single qubit (int); alloc(n) -> list of n qubits."""
        if n is None:
            return self.b.qubit()
        return list(self.b.qubits(n))

    def reg(self, name: str, wires) -> None:
        """Declare an interface register (exempt from ancilla-cleanliness)."""
        if isinstance(wires, int):
            wires = [wires]
        self.b.qubit_register(wires)
        self.registers[name] = list(wires)
        self.wires[name] = list(wires)

    def free(self, q) -> None:
        if isinstance(q, int):
            self.b.recycle(q)
        else:
            for w in q:
                self.b.recycle(w)

    # -- gates ----------------------------------------------------------
    def x(self, q: int) -> None:
        self.b.x(q)

    def cx(self, a: int, b: int) -> None:
        self.b.cx(a, b)

    def swap(self, a: int, b: int) -> None:
        self.b.swap(a, b)

    def ccx(self, a: int, b: int, c: int) -> None:
        self.b.ccx(a, b, c)

    def cz(self, a: int, b: int, cond=None) -> None:
        self.b.cz(a, b, cond=cond)

    def hmr(self, q: int):
        return self.b.hmr(q)


def run_program(source: str, register_names: list[str]) -> CircuitContext:
    """Execute a candidate program's build(ctx) and return the context."""
    ctx = CircuitContext(register_names)
    env: dict = {}
    code = compile(source, "<candidate>", "exec")
    exec(code, env)  # noqa: S102 - candidates are our own generated code
    if "build" not in env:
        raise ValueError("candidate program does not define build(ctx)")
    env["build"](ctx)
    return ctx


__all__ = ["Program", "CircuitContext", "run_program", "CleanlinessError",
           "Builder", "Simulator"]
