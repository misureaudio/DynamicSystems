import sympy as sp
import numpy as np
# ---- Lorenz fixed points & eigenvalues at origin ----
sig, be, ro = 10, sp.Rational(8,3), 28
x,y,z = sp.symbols('x y z')
f = [sig*(y-x), x*(ro-z)-y, x*y - be*z]
sols = sp.solve([sp.Eq(a,0) for a in f],[x,y,z],dict=True)
print("Lorenz fixed points:")
for s in sols:
    print("  ", {k: sp.nsimplify(v) for k,v in s.items()}, " -> ", {k: float(v) for k,v in s.items()})
vars_ = (x,y,z)
J = sp.Matrix([[sp.diff(f[i], v) for v in vars_] for i in range(3)])
J0 = J.subs({x:0,y:0,z:0})
print("\nJ at origin:\n", J0)
print("\neigenvalues at origin:")
for e in J0.eigenvals():
    print("  ", sp.simplify(e), "=", float(e))
div = sp.simplify(sum(sp.diff(f[i],v) for i,v in enumerate((x,y,z))))
print("\ndivergence =", div, "=", float(div), "  ( = -(sigma+1+beta) )")
print("-(sigma+1+beta) =", float(-(sig+1+be)))
# ---- Henon fixed point & multipliers ----
a,b = sp.Rational(14,10), sp.Rational(3,10)
X,Y = sp.symbols('X Y')
h = [1 - a*X**2 + Y, b*X]
fs = sp.solve([sp.Eq(h[0]-X,0), sp.Eq(h[1]-Y,0)],[X,Y],dict=True)
print("\nHenon fixed points:")
for s in fs:
    xs, ys = sp.simplify(s[X]), sp.simplify(s[Y])
    print("  x*=", xs, "=", float(xs), "  y*=", ys, "=", float(ys))
DH = sp.Matrix([[-2*a*X, 1],[b, 0]])
for s in fs:
    D = DH.subs(s)
    ev = D.eigenvals()
    print("  multipliers:", [float(e) for e in ev])
    print("  det DH =", float(D.det()), "  ( = -b = -0.3 )")
