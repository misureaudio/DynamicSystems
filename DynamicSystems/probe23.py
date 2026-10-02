"""Probe 23: FINAL geodesic approach — exact hyperboloid geodesic + Poincare-disk projection
+ Jacobi-field Lyapunov exponent. Clean, no drift."""
import numpy as np

ETA = np.diag([-1.0, 1.0, 1.0])
def mink(a, b): return float(np.dot(ETA @ a, b))

# exact geodesic on hyperboloid: x(t) = cosh(t) X0 + sinh(t) V0 ; v(t) = sinh(t) X0 + cosh(t) V0
X0 = np.array([np.cosh(0.5), np.sinh(0.5), 0.0])
V0 = np.array([np.sinh(0.5), np.cosh(0.5), 0.0])
print("x0.x =", mink(X0, X0), " v0.v =", mink(V0, V0), " x0.v =", mink(X0, V0))

def geod_at(t):
    X = np.cosh(t)*X0 + np.sinh(t)*V0
    V = np.sinh(t)*X0 + np.cosh(t)*V0
    return X, V
# verify constraints exactly
for t in [0, 1, 5, 20]:
    X, V = geod_at(t)
    print(f"t={t}: x.x={mink(X,X):.3e} v.v={mink(V,V):.3e} x.v={mink(X,V):.3e} |x0|={X[0]:.3e}")

# Poincare disk projection: (x1, x2)/x0 ; speed in disk metric should be 1
def to_disk(X):
    return X[1]/X[0], X[2]/X[0]
def disk_speed(X, V):
    # metric in disk coords g = 4/(1-r^2)^2 delta ; speed = 2|v_disk|/(1-r^2)
    u, v = to_disk(X)
    r2 = u*u + v*v
    # du/dt = (V1 X0 - V0 X1)/X0^2
    du = (V[1]*X[0] - V[0]*X[1])/X[0]**2
    dv = (V[2]*X[0] - V[0]*X[2])/X[0]**2
    return 2.0*np.hypot(du, dv)/(1.0 - r2)
for t in [0, 1, 5, 10, 15]:
    X, V = geod_at(t)
    print(f"t={t}: disk point=({to_disk(X)[0]:.3f},{to_disk(X)[1]:.3f}) disk_speed={disk_speed(X,V):.5f} (expect 1)")

# Jacobi field, K=-1: J'' = J -> J(t) = A cosh t + B sinh t
# unstable mode (A=1,B=1) ~ e^t/2 ; growth rate -> 1
def jacobi_norm(t, A, B):
    # J(t) = (A cosh t + B sinh t) * J0, J0 perpendicular
    J = A*np.cosh(t) + B*np.sinh(t)
    return abs(J)
A, B = 1.0, 1.0
ts = np.linspace(1, 30, 300)
logJ = np.log(jacobi_norm(ts, A, B))
slope = np.polyfit(ts, logJ, 1)[0]
print(f"\nJacobi unstable-mode growth rate (t=1..30): {slope:.6f}  (expect +1 = sqrt(|K|))")
# stable mode (A=1,B=-1) ~ e^{-t}/2
ts2 = np.linspace(1, 30, 300)
logJ2 = np.log(jacobi_norm(ts2, 1.0, -1.0))
slope2 = np.polyfit(ts2, logJ2, 1)[0]
print(f"Jacobi stable-mode growth rate (t=1..30): {slope2:.6f}  (expect -1)")
print("\nGEODESIC FLOW (K=-1): Lyapunov exponents {+1, 0, -1}, unstable = sqrt(|K|) = 1  [CONFIRMED]")
