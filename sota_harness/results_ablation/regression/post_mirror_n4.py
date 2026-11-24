def build(ctx):
    a = ctx.alloc(4); ctx.reg("a", a)
    b = ctx.alloc(4); ctx.reg("b", b)
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

    for i in range(3):           # forward: reverse-UMA stages LSB->MSB
        uma_rev(i)
    ctx.cx(c, a[3])              # a3 ^= c3
    ctx.cx(b[3], a[3])              # a3 = s3
    for i in range(2, -1, -1):   # backward: reverse-MAJ MSB->LSB
        maj_rev(i)

    ctx.free(c)
