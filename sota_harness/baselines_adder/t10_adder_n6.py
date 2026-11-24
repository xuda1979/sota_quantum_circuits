def build(ctx):
    a = ctx.alloc(6); ctx.reg("a", a)
    b = ctx.alloc(6); ctx.reg("b", b)
    c = ctx.alloc()                 # single carry ancilla, |0>

    def maj(i):
        ctx.cx(b[i], a[i])
        ctx.cx(c, b[i])
        ctx.ccx(a[i], b[i], c)
        ctx.cx(b[i], c)

    def uma(i):
        ctx.ccx(a[i], b[i], c)
        ctx.cx(b[i], c)
        ctx.cx(c, b[i])
        ctx.cx(c, a[i])

    # 2n-2 Toffoli in-place adder: skip the top MAJ/UMA pair; the top sum
    # bit is read straight off the carry chain (c holds c5 after maj(0..4)),
    # and the dropped carry-out c6 makes the skipped pair redundant.
    for i in range(5):              # maj(0)..maj(4): c = c5
        maj(i)
    ctx.cx(c, a[5])                 # a5 ^= c5
    ctx.cx(b[5], a[5])              # a5 = s5 (a5 was untouched by maj chain)
    for i in range(4, -1, -1):      # uma(4), uma(3), uma(2), uma(1), uma(0)
        uma(i)

    ctx.free(c)
