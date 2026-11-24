def build(ctx):
    a = ctx.alloc(3); ctx.reg("a", a)
    b = ctx.alloc(3); ctx.reg("b", b)
    t = ctx.alloc()                  # single carry ancilla, |0>

    # 4-Toffoli in-place 3-bit adder (2n-2; CDKM uses 2n = 6).
    # Carries: c1 = a0b0;  c2 = a1b1 ^ c1(a1^b1).
    # c1 computed+uncomputed (2 Toffolis); c2 computed directly INTO the
    # sum wire a2 and absorbed by s2 (2 Toffolis, no uncompute needed).
    ctx.ccx(a[0], b[0], t)           # t = c1
    ctx.ccx(a[1], b[1], a[2])        # a2 ^= a1b0... (a1b1)
    ctx.cx(b[1], a[1])               # a1 = a1^b1
    ctx.ccx(t, a[1], a[2])           # a2 ^= c1(a1^b1)  -> a2 = a2^c2
    ctx.cx(b[2], a[2])               # a2 = s2
    ctx.cx(t, a[1])                  # a1 = s1
    ctx.ccx(a[0], b[0], t)           # t = 0 (uncompute c1)
    ctx.cx(b[0], a[0])               # a0 = s0

    ctx.free(t)
