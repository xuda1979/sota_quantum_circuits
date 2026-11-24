"""CDKM in-place ripple-carry subtractor baseline for a -= b over n
qubits: the exact gate-reversal of the CDKM a += b adder (Cuccaro,
Draper, Kutin, Moulton 2004). Every gate is self-inverse (X/CNOT/Toffoli),
so reversing the adder's gate sequence computes the inverse map
|a>|b> -> |a-b mod 2^n>|b>. 2n Toffolis, 1 carry ancilla.

Adder stage order:  maj(0..n-1) then uma(n-1..0).
Mirror stage order (this template):
  forward:  uma_rev(i) for i = 0..n-1     (gate-reversed UMA, LSB->MSB)
  backward: maj_rev(i) for i = n-1..0     (gate-reversed MAJ, MSB->LSB)

uma_rev(i) = reverse of uma(i)'s gates:
    cx(c -> a[i]); cx(c -> b[i]); cx(b[i] -> c); ccx(a[i], b[i], c)
maj_rev(i) = reverse of maj(i)'s gates:
    cx(b[i] -> c); ccx(a[i], b[i], c); cx(c -> b[i]); cx(b[i] -> a[i])

The single carry ancilla returns to |0> exactly and global phase is zero
(inverse of a circuit with those properties).
"""

from __future__ import annotations

CDKM_SUB_SOURCE = """\
def build(ctx):
    a = ctx.alloc({n}); ctx.reg("a", a)
    b = ctx.alloc({n}); ctx.reg("b", b)
    c = ctx.alloc()                 # single carry ancilla, |0>

    def uma_rev(i):
        # gate-reversed UMA: adder uncompute stage, gate order flipped
        ctx.cx(c, a[i])
        ctx.cx(c, b[i])
        ctx.cx(b[i], c)
        ctx.ccx(a[i], b[i], c)

    def maj_rev(i):
        # gate-reversed MAJ: adder carry stage, gate order flipped
        ctx.cx(b[i], c)
        ctx.ccx(a[i], b[i], c)
        ctx.cx(c, b[i])
        ctx.cx(b[i], a[i])

    for i in range({n}):           # forward: reverse-UMA stages LSB->MSB
        uma_rev(i)
    for i in range({n} - 1, -1, -1):   # backward: reverse-MAJ MSB->LSB
        maj_rev(i)

    ctx.free(c)
"""


def cdkm_sub_source(n: int) -> str:
    assert n >= 1
    return CDKM_SUB_SOURCE.format(n=n)


def cdkm_sub_templates(n: int) -> dict[str, str]:
    return {f"cdkm_sub_n{n}": cdkm_sub_source(n)}
