def build(ctx):
    a = ctx.alloc(6); ctx.reg("a", a)
    b = ctx.alloc(6); ctx.reg("b", b)
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

    for i in range(5):           # forward: propagate carries LSB->MSB
        maj(i)
    ctx.cx(c, a[5])              # a5 ^= c5
    ctx.cx(b[5], a[5])              # a5 = s5
    for i in range(4, -1, -1):   # backward: uncompute carries MSB->LSB
        uma(i)

    ctx.free(c)
