def build(ctx):
    a = ctx.alloc(3); ctx.reg("a", a)
    b = ctx.alloc(3); ctx.reg("b", b)
    t1 = ctx.alloc()                 # ancilla 1: a0*b0
    t2 = ctx.alloc()                 # ancilla 2: a1*b1, then c2

    # Three Toffoli products (multiplicative-complexity-optimal, LB=3):
    #   p1 = a0*b0                       (carry c1)
    #   p2 = a1*b1
    #   p3 = a0*b0*(a1^b1) = a0a1b0 ^ a0b0b1
    ctx.ccx(a[0], b[0], t1)          # t1 = a0*b0
    ctx.ccx(a[1], b[1], t2)          # t2 = a1*b1
    ctx.cx(b[1], a[1])               # a1 ^= b1          (a1 = a1^b1)
    ctx.ccx(t1, a[1], t2)            # t2 = a1b1 ^ a0b0(a1^b1) = c2
    ctx.cx(t2, a[2])                 # a2 ^= c2
    ctx.cx(b[2], a[2])               # a2 ^= b2          (a2 = s2)
    ctx.cx(t1, a[1])                 # a1 ^= c1          (a1 = s1)
    ctx.cx(b[0], a[0])               # a0 ^= b0          (a0 = s0)

    ctx.free(t1)
    ctx.free(t2)
