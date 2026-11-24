def build(ctx):
    a = ctx.alloc(4); ctx.reg("a", a)
    b = ctx.alloc(4); ctx.reg("b", b)
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

    for i in range(3):           # forward: propagate carries LSB->MSB
        maj(i)
    ctx.cx(c, a[3])              # a3 ^= c3
    ctx.cx(b[3], a[3])              # a3 = s3
    for i in range(2, -1, -1):   # backward: uncompute carries MSB->LSB
        uma(i)

    ctx.free(c)
