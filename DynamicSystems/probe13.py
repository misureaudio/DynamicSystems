"""Probe 13: Benettin on Poincare-disk geodesic flow, projected tangent frame. Expect {+1,0,-1}."""
import numpy as np

def geodesic_rhs(x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    return np.array([vx, vy, 2.0*(v2*px - 2.0*xv*vx)/one, 2.0*(v2*py - 2.0*xv*vy)/one])

def jac_fd(x):
    h = 1e-6
    J = np.zeros((4,4)); f0 = geodesic_rhs(x)
    for j in range(4):
        xp = x.copy(); xp[j] += h
        J[:, j] = (geodesic_rhs(xp) - f0)/h
    return J

def metric_speed(x):
    px, py, vx, vy = x
    return 2.0*np.hypot(vx, vy)/(1.0 - (px*px + py*py))

def grad_F(x):
    px, py, vx, vy = x
    v = np.hypot(vx, vy)
    return np.array([2*px, 2*py, 2*vx/v, 2*vy/v])

def tangent_frame(x, seed):
    rng = np.random.default_rng(seed)
    g = grad_F(x)
    W = rng.standard_normal((4, 5))
    proj = W - np.outer(g, (g @ W)/np.dot(g, g))
    Q, R = np.linalg.qr(proj)
    return Q[:, :3]

def rk4(x, V, dt):
    k1x = geodesic_rhs(x); k1v = jac_fd(x) @ V
    k2x = geodesic_rhs(x + 0.5*dt*k1x); k2v = jac_fd(x + 0.5*dt*k1x) @ (V + 0.5*dt*k1v)
    k3x = geodesic_rhs(x + 0.5*dt*k2x); k3v = jac_fd(x + 0.5*dt*k2x) @ (V + 0.5*dt*k2v)
    k4x = geodesic_rhs(x + dt*k3x); k4v = jac_fd(x + dt*k3x) @ (V + dt*k3v)
    return (x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x),
            V + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v))

px, py = 0.0, 0.2
vflat = (1.0 - (px*px + py*py))/2.0
x0 = np.array([px, py, vflat, 0.0])
print("initial speed:", metric_speed(x0))
x = x0.copy()
V = tangent_frame(x0, seed=0)
tot = np.zeros(3)
T, dt = 300.0, 0.01
for s in range(int(T/dt)):
    x, V = rk4(x, V, dt)
    Q, R = np.linalg.qr(V)
    sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
    tot += np.log(np.abs(np.diag(R)))
    V = Q * sgn
print("final speed:", metric_speed(x), " r_final:", np.hypot(x[0], x[1]))
print("exponents:", np.round(tot/T, 4), " (expect {+1, 0, -1})")
