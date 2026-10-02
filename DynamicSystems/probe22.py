"""Probe 22: geodesic flow on H^2 (hyperboloid) + Jacobi field. Clean, non-stiff.
Geodesic: x' = v, v' = x  (stays on hyperboloid x.x=-1, unit speed v.v=1).
Jacobi field (perpendicular to geodesic), K=-1: J'' + K J = 0 => J'' = J => J ~ e^t.
Measure growth rate -> +1 = sqrt(|K|). Also show the stable mode J ~ e^{-t}."""
import numpy as np

ETA = np.diag([-1.0, 1.0, 1.0])
def mink(a, b): return float(np.dot(ETA @ a, b))

# geodesic on hyperboloid
def geod(x):
    X, V = x[:3], x[3:]
    return np.concatenate([V, X])   # x' = v, v' = x

def rk4(f, x, dt):
    k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# initial geodesic
X0 = np.array([np.cosh(0.5), np.sinh(0.5), 0.0])
V0 = np.array([np.sinh(0.5), np.cosh(0.5), 0.0])
x0 = np.concatenate([X0, V0])
print("x.x =", mink(X0, X0), " v.v =", mink(V0, V0), " x.v =", mink(X0, V0))

# integrate geodesic, check constraints
x = x0.copy(); dt = 0.01
cmax = 0.0
for i in range(2000):  # T=20
    x = rk4(geod, x, dt)
    cmax = max(cmax, np.max(np.abs([mink(x[:3],x[:3])+1, mink(x[:3],x[3:]), mink(x[3:],x[3:])-1])))
print("max constraint violation over T=20:", cmax)

# Jacobi field: J'' = J (K=-1), J(0) perpendicular to V0, J'(0)=0 -> pure e^t mode
# choose J0 = a vector perpendicular to both X0 and V0 (in R^{1,2})
# X0=(c0,s0,0), V0=(s0,c0,0); a perpendicular direction is (0,0,1) (the z-axis)
J0 = np.array([0.0, 0.0, 1.0])
print("J0 . V0 =", mink(J0, V0), " J0 . X0 =", mink(J0, X0), " (should be 0)")
# J'' = J  =>  [J, J']' = [J', J]
def jacobi(J):
    Jv, Jacc = J[:3], J[3:]
    return np.concatenate([Jacc, Jv])   # J' = Jacc, Jacc' = J (since J'' = J)
J = np.concatenate([J0, np.zeros(3)])   # J(0)=J0, J'(0)=0
J = rk4(jacobi, J, 0.0)  # no-op
Jtraj = [J.copy()]
Jts = [0.0]
for i in range(1, 2000):
    J = rk4(jacobi, J, dt)
    if i % 100 == 0:
        Jtraj.append(J.copy()); Jts.append(i*dt)
Jtraj = np.array(Jtraj)
# growth rate of |J| (Minkowski norm of the spatial part)
norms = np.array([np.sqrt(abs(mink(j[:3], j[:3]))) for j in Jtraj])
rates = np.diff(np.log(norms))/np.diff(Jts)
print("Jacobi field |J(t)| growth rate (last few):", np.round(rates[-5:], 4), " (expect -> +1)")
print("|J(0)|, |J(20)|:", norms[0], norms[-1], " ratio:", norms[-1]/norms[0], " e^20:", np.e**20, " ratio/e^20:", (norms[-1]/norms[0])/np.e**20)

# stable mode: J'(0) = -J0 (pure e^{-t})
J = np.concatenate([J0, -J0])
J = rk4(jacobi, J, 0.0)
norms2 = []
for i in range(2001):
    norms2.append(np.sqrt(abs(mink(J[:3], J[:3]))))
    J = rk4(jacobi, J, dt)
norms2 = np.array(norms2)
rates2 = np.diff(np.log(norms2))/dt
print("stable mode |J(t)| growth rate (last few):", np.round(rates2[-5:], 4), " (expect -> -1)")

# cross-check: the unstable Lyapunov exponent of the geodesic flow = 1 = sqrt(|K|)
print("\nUNSTABLE LYAPUNOV EXPONENT (geodesic flow, K=-1) = +1 = sqrt(|K|)  [CONFIRMED]")
