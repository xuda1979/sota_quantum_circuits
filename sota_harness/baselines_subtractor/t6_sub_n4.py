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

    # 2n-2 Toffoli in-place subtractor: exact gate-reversal of the 2n-2
    # adder (skip the top uma_rev/maj_rev pair; the top difference bit is
    # read straight off the carry chain).
    for i in range(3):              # uma_rev(0), uma_rev(1), uma_rev(2)
        uma_rev(i)
    ctx.cx(b[3], a[3])              # a3 ^= b3 (splice, order flipped)
    ctx.cx(c, a[3])                 # a3 ^= c3  -> a3 = d3
    for i in range(2, -1, -1):      # maj_rev(2), maj_rev(1), maj_rev(0)
        maj_rev(i)

    ctx.free(c)
