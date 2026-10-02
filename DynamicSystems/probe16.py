"""Probe 16: Benettin on Poincare-disk geodesic flow, CORRECT ODE + FD Jacobian,
segmented Radau. Expect Lyapunov exponents {+1, 0, -1}."""
import numpy as np
from scipy.integrate import solve_ivp

def f_base(t, x):
    px, py, vx, vy = x
    one = 1.0 - (px*px + py*py)
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    return np.array([vx, vy, (2*v2*px - 4*xv*vx)/one, (2*v2*py - 4*xv*vy)/one])

def Df(x):
    h = 1e-6
    J = np.zeros((4,4)); f0 = f_base(0, x)
    for j in range(4):
        xp = x.copy(); xp[j] += h
        J[:, j] = (f_base(0, xp) - f0)/h
    return J

def speed(x):
    px, py, vx, vy = x
    return 2.0*np.hypot(vx, vy)/(1.0 - (px*px + py*py))

def grad_F(x):
    px, py, vx, vy = x
    v = np.hypot(vx, vy)
    return np.array([2*px, 2*py, 2*vx/v, 2*vy/v])

def tangent_frame(x, seed):
    rng = np.random.default_rng(seed)
    g = grad_F(x)
    W = rng.standard_normal((4, 6))
    proj = W - np.outer(g, (g @ W)/np.dot(g, g))
    Q, _ = np.linalg.qr(proj)
    return Q[:, :3]

px, py = 0.0, 0.2
vflat = (1.0 - (px*px + py*py))/2.0
x0 = np.array([px, py, vflat, 0.0])
T = 50.0
sol = solve_ivp(f_base, [0, T], x0, method="Radau", rtol=1e-10, atol=1e-13,
                dense_output=True, max_step=0.2)
print("base ok:", sol.success, " r_end:", np.hypot(*sol.y[:2,-1]))

def A_of_t(t):
    return Df(sol.sol(t))

def benettin(T, seg=1.0):
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
        sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R)))
        V = Q * sgn
        t = t1
    return tot/T

lam = benettin(50.0, seg=1.0)
print("Lyapunov exponents (K=-1):", np.round(lam, 4), " (expect {+1, 0, -1})")
