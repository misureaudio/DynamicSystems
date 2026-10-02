"""Probe 17: geodesic Lyapunov exponent, single Radau variational call over dense base."""
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
T = 40.0
sol = solve_ivp(f_base, [0, T], x0, method="Radau", rtol=1e-10, atol=1e-13,
                dense_output=True, max_step=0.2)
print("base ok:", sol.success, " r_end:", np.hypot(*sol.y[:2,-1]))

def f_var(t, Vflat):
    V = Vflat.reshape(4, 3)
    return (Df(sol.sol(t)) @ V).ravel()

V0 = tangent_frame(x0, 0)
sv = solve_ivp(f_var, [0, T], V0.ravel(), method="Radau", rtol=1e-9, atol=1e-12,
               max_step=0.2)
Vend = sv.y[:, -1].reshape(4, 3)
# accumulate exponents via QR at the end (single QR is wrong for exponents; need periodic).
# Instead, redo with periodic QR using dense base but explicit RK4 for V at small dt is stiff.
# Use the QR-decomposition accumulation but integrate V with Radau in chunks of the dense base.
def benettin(T, nchunks=20):
    V = tangent_frame(x0, 0)
    tot = np.zeros(3)
    tb = np.linspace(0, T, nchunks + 1)
    for ci in range(nchunks):
        t0, t1 = tb[ci], tb[ci+1]
        s = solve_ivp(f_var, [t0, t1], V.ravel(), method="Radau", rtol=1e-9, atol=1e-12, max_step=0.2)
        V = s.y[:, -1].reshape(4, 3)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R)))
        V = Q * sgn
    return tot/T
lam = benettin(40.0, nchunks=20)
print("Lyapunov exponents (K=-1):", np.round(lam, 4), " (expect {+1, 0, -1})")
