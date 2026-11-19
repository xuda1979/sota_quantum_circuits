#!/usr/bin/env python3
"""Toy-task evaluator: reversible inversion in F_2^4, p(X) = X^4 + X + 1.

Task (paper Extended Data Figure 1):
    |x>|h> -> |x>|h XOR x^-1>,  0^-1 = 0
    preserve 4-qubit x, XOR the inverse into arbitrary 4-qubit h,
    be exact for every x, h, clean every ancilla, leave zero global phase.

Cost = (executed Toffoli count T) x (peak live qubits Q), both measured
from the circuit exactly (Simulator billing + Builder.peak_qubits).

Usage:
    python f24_evaluator.py <candidate.py> [candidate.py ...]

Each candidate module defines build(b) (b a fresh Builder) or runs at
import with a Builder it creates itself and exposes it as `b`.
Exhaustive test: all 16 x-values x 8 h-values per candidate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "/home/david/software/sota_quantum_circuits/uncompute")

from uncompute import Builder, Simulator  # noqa: E402
from uncompute.sim import CleanlinessError  # noqa: E402

# GF(16) inversion table, p(X) = X^4 + X + 1 (little-endian bit packing).
INV = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]

H_VALS = [0b0000, 0b1010, 0b0101, 0b1111, 0b0001, 0b0110, 0b1011, 0b1101]


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(path: Path) -> dict:
    mod = load(path)
    if hasattr(mod, "build"):
        b = Builder()
        x, h = mod.build(b)
    else:
        b = mod.b
        x, h = mod.x, mod.h
    if not b._registers:  # candidate forgot to declare registers
        return {"valid": False, "fail": "no qubit registers declared"}

    xs = list(range(16)) * len(H_VALS)
    hs = [hv for hv in H_VALS for _ in range(16)]
    sim = Simulator(b, batch=len(xs), seed=0)
    sim.write(x, xs)
    sim.write(h, hs)
    sim.run()
    ok = all(
        xo == xi and ho == (hi ^ INV[xi])
        for xo, ho, xi, hi in zip(sim.read(x), sim.read(h), xs, hs)
    )
    if not ok:
        return {"valid": False, "fail": "wrong output (h != h XOR inv(x))"}
    try:
        sim.assert_clean()
    except CleanlinessError as e:
        return {"valid": False, "fail": str(e).splitlines()[0]}
    T = sim.stats.toffoli / sim.batch
    Q = b.peak_qubits
    return {"valid": True, "toffoli": T, "qubits": Q, "tq": T * Q}


def main(argv):
    for name in argv[1:]:
        res = evaluate(Path(name))
        if res["valid"]:
            print(f"{name}: valid: executed_toffoli={res['toffoli']:g}; "
                  f"qubits={res['qubits']}; TQ={res['tq']:g}")
        else:
            print(f"{name}: invalid: {res['fail']}")


if __name__ == "__main__":
    main(sys.argv)
