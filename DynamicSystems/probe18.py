"""Probe 18: geodesic flow on the HYPERBOLOID model (Minkowski) — clean, non-stiff.
H^2 = {x in R^{1,2}: x.x = -1, x0>0}, eta = diag(-1,1,1).
Geodesic flow: x' = v, v' = (v.v) x = x  (unit speed v.v=1). LINEAR system.
Constraints: x.x=-1, x.v=0, v.v=1 (3 constraints, 6 vars -> 3-dim ST M).
Expect Lyapunov exponents {+1, 0, -1} = {sqrt(|K|), 0, -sqrt(|K|)} for K=-1."""
import numpy as np

ETA = np.diag([-1.0, 1.0, 1.0])
def mink(a, b):
    return np.dot(ETA @ a, b)

def geod_flow(x):
    # x = [x0,x1,x2, v0,v1,v2]
    X = x[:3]; V = x[3:]
    dX = V
    dV = X  # (v.v) X = 1 * X
    return np.concatenate([dX, dV])

def Df(x):
    # constant Jacobian: d/dx [V, X] = [[0, I],[I, 0]]
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:,:3] = np.eye(3)
    return A

def constraints(x):
    X = x[:3]; V = x[3:]
    return np.array([mink(X,X)+1.0, mink(X,V), mink(V,V)-1.0])

def grad_constraints(x):
    X = x[:3]; V = x[3:]
    G = np.zeros((3,6))
    G[0,:3] = 2*(ETA@X)
    G[1,:3] = ETA@V; G[1,3:] = ETA@X
    G[2,3:] = 2*(ETA@V)
    return G

def tangent_frame(x, seed):
    rng = np.random.default_rng(seed)
    G = grad_constraints(x)          # 3x6
    W = rng.standard_normal((6,8))
    # project onto ker(G): null space of G
    # use SVD of G
    U,S,Vt = np.linalg.svd(G)
    N = Vt[3:,:]                       # 3x6 basis for null space
    # orthonormalize
    Q,_ = np.linalg.qr(N.T)
    return Q.T[:, :3]                  # 6x3

def rk4(x, V, dt):
    A = Df(x)
    k1x = geod_flow(x); k1v = A @ V
    k2x = geod_flow(x + 0.5*dt*k1x); k2v = A @ (V + 0.5*dt*k1v)
    k3x = geod_flow(x + 0.5*dt*k2x); k3v = A @ (V + 0.5*dt*k2v)
    k4x = geod_flow(x + dt*k3x); k4v = A @ (V + dt*k3v)
    return (x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x),
            V + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v))

# initial condition on the hyperboloid, unit speed, in the (x0,x1) plane
X0 = np.array([np.cosh(0.5), np.sinh(0.5), 0.0])
V0 = np.array([np.sinh(0.5), np.cosh(0.5), 0.0])  # dX/dt; check v.v = 1
print("x.x =", mink(X0,X0), " v.v =", mink(V0,V0), " x.v =", mink(X0,V0))
x0 = np.concatenate([X0, V0])
print("constraints:", constraints(x0))

# constraint conservation
x = x0.copy(); dt = 0.01
cmax = 0.0
for i in range(2000):
    x, _ = rk4(x, None, dt) if False else (x, None)
    # integrate x only
    k1 = geod_flow(x); k2 = geod_flow(x+0.5*dt*k1); k3 = geod_flow(x+0.5*dt*k2); k4 = geod_flow(x+dt*k3)
    x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
    cmax = max(cmax, np.max(np.abs(constraints(x))))
print("max |constraint violation| over 20s:", cmax)

# Benettin
V = tangent_frame(x0, 0)
tot = np.zeros(3)
T = 100.0
for i in range(int(T/dt)):
    x, V = rk4(x, V, dt)
    Q, R = np.linalg.qr(V)
    sgn = np.sign(np.diag(R)); sgn[sgn==0]=1
    tot += np.log(np.abs(np.diag(R)))
    V = Q*sgn
lam = tot/T
print("Lyapunov exponents (K=-1):", np.round(lam, 4), " (expect {+1, 0, -1})")

# eigenvalues of constant Jacobian
eigs = np.linalg.eigvals(Df(x0))
print("eigenvalues of constant Jacobian:", np.round(np.sort(eigs.real), 4),
      "(real parts +-1 -> Lyapunov exponents +-1, plus 0 for flow direction)")
