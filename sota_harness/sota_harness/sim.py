"""Batch simulation of a built circuit context over a list of shots.

Uses the official repo's wide-lane Simulator (bit-parallel, exact), so the
billing (stats.toffoli per shot) matches the paper's evaluator. Strict mode
is ON: MBU fixups are proven algebraically, not just sampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .circuit import Simulator


@dataclass
class SimResult:
    toffoli_per_shot: float
    qubits: int
    outputs: dict[str, list[int]] = field(default_factory=dict)
    register_wires: dict[str, list[int]] = field(default_factory=dict)


def batch_simulate(ctx, shots: Iterable[dict], seed: int = 0) -> SimResult:
    shots = list(shots)
    n = len(shots)
    if n == 0:
        raise ValueError("no shots to simulate")
    sim = Simulator(ctx.b, batch=n, seed=seed)

    # write inputs: each register's domain values, little-endian
    for name, wires in ctx.registers.items():
        vals = [s[name] for s in shots]
        sim.write(wires, vals)

    sim.run()

    # register_qubits bookkeeping for assert_clean exemption
    sim.register_qubits = {q for wires in ctx.registers.values() for q in wires}
    sim.assert_clean()

    outputs = {name: sim.read(wires) for name, wires in ctx.registers.items()}
    return SimResult(
        toffoli_per_shot=sim.stats.toffoli / n,
        qubits=ctx.b.peak_qubits,
        outputs=outputs,
        register_wires={k: list(v) for k, v in ctx.registers.items()},
    )
