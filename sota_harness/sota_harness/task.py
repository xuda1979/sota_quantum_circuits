"""Task model: reversible-map specs in the kickmix circuit model.

A Task fixes:
  - input registers (name, n qubits) and output/interface registers,
  - the reference map pi on basis states (exhaustive table or oracle),
  - the phase function g (default 0),
  - validity gates: interface preservation, phase, ancilla cleanliness,
  - the cost: TQ = (executed Toffoli count) x (peak live qubits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class Register:
    name: str
    n: int


@dataclass
class Task:
    name: str
    spec_text: str  # full natural-language spec handed to proposers
    registers: tuple[Register, ...]
    reference: Callable[..., dict[str, int]]  # inputs dict -> outputs dict
    input_domains: dict[str, Sequence[int]]  # exhaustive or sampled test set
    failure_budget: float = 0.0  # max fraction of wrong shots (0 = exact)
    toffoli_lower_bound: int | None = None  # optional MC certificate

    @property
    def registers_by_name(self) -> list[str]:
        return [r.name for r in self.registers]

    def example_shots(self, max_shots: int = 4096) -> list[dict[str, int]]:
        """Cartesian product of input domains (capped)."""
        import itertools

        names = sorted(self.input_domains)
        shots = []
        for combo in itertools.product(*(self.input_domains[n] for n in names)):
            shots.append(dict(zip(names, combo)))
            if len(shots) >= max_shots:
                break
        return shots
