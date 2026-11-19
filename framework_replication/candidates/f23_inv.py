"""F23 inversion (GF(8), p=X^3+X+1): 3 Toffolis, 0 ancillas, Q=6.
Found by deterministic per-bit span decomposition (harness
proposer, 2026-11-17). Products: x1&x2, x0&x1, x0&x2 + affine."""
import sys
sys.path.insert(0, "/home/david/software/sota_quantum_circuits/uncompute")
from uncompute import Builder


def build(b):
    x = b.qubits(3); b.qubit_register(x)
    h = b.qubits(3); b.qubit_register(h)
    b.ccx(x[1], x[2], h[0])
    b.ccx(x[0], x[1], h[1])
    b.ccx(x[0], x[2], h[2])
    # affine residuals: TGT[b] ^ product
    b.cx(x[0], h[0])
    b.cx(x[1], h[0])
    b.cx(x[2], h[0])
    b.cx(x[2], h[1])
    b.cx(x[1], h[2])
    b.cx(x[2], h[2])
    return x, h
