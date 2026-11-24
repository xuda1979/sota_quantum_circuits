"""CDKM in-place ripple-carry adder baseline for a += b over n qubits
(Cuccaro, Draper, Kutin, Moulton 2004). 2n Toffolis, 1 carry ancilla.

MAJ(a, b, c) [c = carry-in]:
    cx(b -> a); cx(c -> b); ccx(a, b, c); cx(b -> c)
  maps (a, b, c) -> (a^b, b^c, majority(a,b,c))            (verified by algebra)
UMA(a, b, c):
    ccx(a, b, c); cx(b -> c); cx(c -> b); cx(c -> a)
  maps (a^b, b^c, carry_out) -> (a+b mod 2… sum, b, carry_in)  (verified by algebra)

Forward: n MAJ stages LSB->MSB (carry threads the single ancilla).
Backward: n UMA stages MSB->LSB restores b, writes sum into a, and returns
the carry ancilla to |0> exactly (c_0 = 0). Global phase is zero (only
X/CNOT/Toffoli; no controlled-phases, no measurement).
"""

from __future__ import annotations

CDKM_MAJ_SOURCE = """\
def build(ctx):
    a = ctx.alloc({n}); ctx.reg("a", a)
    b = ctx.alloc({n}); ctx.reg("b", b)
    c = ctx.alloc()                 # single carry ancilla, |0>

    def maj(i):
        # (a[i], b[i], c) -> (a^b, b^c, carry_out)
        ctx.cx(b[i], a[i])
        ctx.cx(c, b[i])
        ctx.ccx(a[i], b[i], c)
        ctx.cx(b[i], c)

    def uma(i):
        # (a^b, b^c, carry_out) -> (sum, b, carry_in)
        ctx.ccx(a[i], b[i], c)
        ctx.cx(b[i], c)
        ctx.cx(c, b[i])
        ctx.cx(c, a[i])

    for i in range({n}):           # forward: propagate carries LSB->MSB
        maj(i)
    for i in range({n} - 1, -1, -1):   # backward: uncompute carries MSB->LSB
        uma(i)

    ctx.free(c)
"""


def cdkm_source(n: int) -> str:
    assert n >= 1
    return CDKM_MAJ_SOURCE.format(n=n)


def cdkm_templates(n: int) -> dict[str, str]:
    return {f"cdkm_n{n}": cdkm_source(n)}
