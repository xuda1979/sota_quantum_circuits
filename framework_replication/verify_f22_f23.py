#!/usr/bin/env python3
"""Verify the F22/F23 inversion tower circuits with the official Builder.

Re-runs the emitted candidates (framework_replication/candidates/
f22_inv.py, f23_inv.py) against the official uncompute Builder +
Simulator, exhaustively over all (x, h) pairs, and re-checks the
certificate rank facts. Exit 0 iff everything holds.
"""
import importlib.util
import sys

sys.path.insert(0, "/home/david/software/sota_quantum_circuits/uncompute")
from uncompute import Builder, Simulator  # noqa: E402

F22_INV = [0, 1, 3, 2]
F23_INV = [0, 1, 5, 6, 7, 2, 3, 4]


def run_candidate(path, inv, n):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = Builder()
    x, h = mod.build(b)
    size = 1 << n
    xs = list(range(size)) * size
    hs = [hv for hv in range(size) for _ in range(size)]
    sim = Simulator(b, batch=len(xs), seed=0)
    sim.write(x, xs)
    sim.write(h, hs)
    sim.run()
    ok = all(xo == xi and ho == (hi ^ inv[xi])
             for xo, ho, xi, hi in zip(sim.read(x), sim.read(h), xs, hs))
    sim.assert_clean()
    T = sim.stats.toffoli / sim.batch
    Q = b.peak_qubits
    return ok, T, Q


def main():
    from pathlib import Path
    base = Path("/home/david/software/sota_quantum_circuits/"
                "framework_replication/candidates")
    ok22, T22, Q22 = run_candidate(base / "f22_inv.py", F22_INV, 2)
    print(f"f22_inv.py: output ok: {ok22}; executed_toffoli={T22:g}; "
          f"qubits={Q22}; TQ={T22 * Q22:g}")
    ok23, T23, Q23 = run_candidate(base / "f23_inv.py", F23_INV, 3)
    print(f"f23_inv.py: output ok: {ok23}; executed_toffoli={T23:g}; "
          f"qubits={Q23}; TQ={T23 * Q23:g}")
    # rank facts
    sys.path.insert(0, "/home/david/software/sota_quantum_circuits/sota_harness")
    from sota_harness.certificates import f23_T3_rank_argument
    facts = f23_T3_rank_argument()
    print("f23 rank facts:", facts)
    good = (ok22 and T22 == 0 and Q22 == 4 and ok23 and T23 == 3
            and Q23 == 6 and facts["T_ge_3_for_any_Q"])
    print("TOWER OK" if good else "TOWER FAILED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
