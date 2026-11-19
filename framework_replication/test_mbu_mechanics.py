"""API smoke test: temporary AND + MBU erase, checked with the strict Simulator.

Verifies the exact mechanics the paper's toy example uses (Extended Data
Figure 1c): ccx into an ancilla, cx into h, hmr (X-basis measurement),
cz(..., cond=outcome) fixup, recycle. Confirms phase stays zero and the
ancilla returns to |0> on every lane.
"""
import sys

sys.path.insert(0, "/home/david/software/sota_quantum_circuits/uncompute")

from uncompute import Builder, Simulator

b = Builder()
x = b.qubits(2)
b.qubit_register(x)
h = b.qubits(1)
b.qubit_register(h)
t = b.qubit()
b.ccx(x[0], x[1], t)          # t = x0*x1           (1 Toffoli)
b.cx(t, h[0])                 # h0 ^= x0*x1          (free)
m = b.hmr(t)                  # erase t -> |0>, outcome m
b.cz(x[0], x[1], cond=m)      # phase fixup          (free, Clifford)
b.recycle(t)

sim = Simulator(b, batch=4, seed=0)
sim.write(x, [0b00, 0b01, 0b10, 0b11])
sim.write(h, [0, 1, 0, 1])
sim.run()
got = sim.read(h)
print("h out:", got, "expected:", [hh ^ ((xx) & (xx >> 1) & 1) for hh, xx in zip([0, 1, 0, 1], [0, 1, 2, 3])])
sim.assert_clean()
print("assert_clean passed: phase zero, ancillas zero on all lanes")
print("stats: toffoli =", sim.stats.toffoli / sim.batch, "peak_qubits =", b.peak_qubits)
