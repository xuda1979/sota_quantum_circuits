"""F22 inversion (GF(4), p=X^2+X+1): z^-1 = z^2 (Frobenius) for
z != 0, 0^-1 = 0: (b0,b1) -> (b0^b1, b1); 0 Toffolis, Q=4."""
import sys
sys.path.insert(0, "/home/david/software/sota_quantum_circuits/uncompute")
from uncompute import Builder

def build(b):
    x = b.qubits(2); b.qubit_register(x)
    h = b.qubits(2); b.qubit_register(h)
    b.cx(x[0], h[0])
    b.cx(x[1], h[0])
    b.cx(x[1], h[1])
    return x, h
