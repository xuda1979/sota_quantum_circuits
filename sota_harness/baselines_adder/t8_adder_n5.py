def build(ctx):
    a = ctx.alloc(5); ctx.reg("a", a)
    b = ctx.alloc(5); ctx.reg("b", b)
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
    # bit is read straight off the carry chain (c holds c4 after maj(0..3)),
    # and the dropped carry-out c5 makes the skipped pair redundant.
    for i in range(4):              # maj(0), maj(1), maj(2), maj(3): c = c4
        maj(i)
    ctx.cx(c, a[4])                 # a4 ^= c4
    ctx.cx(b[4], a[4])              # a4 = s4 (a4 was untouched by maj chain)
    for i in range(3, -1, -1):      # uma(3), uma(2), uma(1), uma(0)
        uma(i)

    ctx.free(c)
