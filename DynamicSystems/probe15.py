"""Probe 15: robust Benettin on Poincare-disk geodesic flow.
Decoupled: Radau base trajectory (dense) + segmented variational equation with analytic Df.
Expect Lyapunov exponent -> +1 (and -1 for the contraction)."""
import numpy as np
from scipy.integrate import solve_ivp

def f_base(t, x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    return np.array([vx, vy, 2.0*(v2*px - 2.0*xv*vx)/one, 2.0*(v2*py - 2.0*xv*vy)/one])

# analytic Jacobian for the FACTOR-2 ODE (verified via sympy)
def Df(x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    t = r2 - 1.0  # = -(1-r2)
    t2 = t*t
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    J20 = 2*(2*px*(px*v2 - 2*vx*xv) + (vx*vx - vy*vy)*t)/t2
    J21 = 4*(py*(px*v2 - 2*vx*xv) + vx*vy*t)/t2
    J22 = 4*(px*vx + py*vy)/t
    J23 = 4*(-px*vy + py*vx)/t
    J30 = 4*(px*(py*v2 - 2*vy*xv) + vx*vy*t)/t2
    J31 = 2*(2*py*(py*v2 - 2*vy*xv) + (-vx*vx + vy*vy)*t)/t2
    J32 = 4*(px*vy - py*vx)/t
    J33 = 4*(px*vx + py*vy)/t
    J = np.zeros((4,4))
    J[0,2]=1; J[1,3]=1
    J[2,0]=J20; J[2,1]=J21; J[2,2]=J22; J[2,3]=J23
    J[3,0]=J30; J[3,1]=J31; J[3,2]=J32; J[3,3]=J33
    return J

# verify Df against FD at interior points
rng = np.random.default_rng(0)
maxerr = 0.0
for _ in range(20):
    x = np.array([0.3*rng.standard_normal(), 0.3*rng.standard_normal(), 0.3, 0.0])
    x[3] = 0.3*rng.standard_normal()
    h=1e-6; Jn=np.zeros((4,4)); f0=f_base(0,x)
    for j in range(4):
        xp=x.copy(); xp[j]+=h; Jn[:,j]=(f_base(0,xp)-f0)/h
    maxerr = max(maxerr, np.abs(Df(x)-Jn).max())
print("max |Df_an - Df_fd| (interior):", maxerr)

# base trajectory with Radau, dense output
px, py = 0.0, 0.2
vflat = (1.0 - (px*px+py*py))/2.0
x0 = np.array([px, py, vflat, 0.0])
T = 40.0
sol = solve_ivp(f_base, [0, T], x0, method="Radau", rtol=1e-10, atol=1e-13, dense_output=True, max_step=0.2)
print("base ok:", sol.success, " r_end:", np.hypot(*sol.y[:2,-1]))

# tangent frame (projected)
def grad_F(x):
    px, py, vx, vy = x
    v = np.hypot(vx, vy)
    return np.array([2*px, 2*py, 2*vx/v, 2*vy/v])
def tangent_frame(x, seed):
    rng2 = np.random.default_rng(seed)
    g = grad_F(x)
    W = rng2.standard_normal((4,6))
    proj = W - np.outer(g, (g@W)/np.dot(g,g))
    Q,_ = np.linalg.qr(proj)
    return Q[:,:3]

def A_of_t(t):
    return Df(sol.sol(t))

# segmented variational integration
def benettin_geodesic(T, seg=2.0):
    V = tangent_frame(x0, 0)
    tot = np.zeros(3)
    t = 0.0
    while t < T - 1e-9:
        t1 = min(t + seg, T)
        Vflat = V.ravel()
        s = solve_ivp(lambda tt, vv: (A_of_t(tt) @ vv.reshape(4,3)).ravel(),
                      [t, t1], Vflat, method="Radau", rtol=1e-9, atol=1e-12)
        V = s.y[:, -1].reshape(4,3)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.diag(R)); sgn[sgn==0]=1
        tot += np.log(np.abs(np.diag(R)))
        V = Q*sgn
        t = t1
    return tot/T
lam = benettin_geodesic(40.0, seg=2.0)
print("Lyapunov exponents (K=-1):", np.round(lam, 4), " (expect {+1, 0, -1})")
