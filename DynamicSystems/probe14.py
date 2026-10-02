"""Probe 14: geodesic flow Lyapunov exponent via two-geodesic Riemannian separation,
stiff solver (Radau), perpendicular Jacobi perturbation. Expect exponent -> +1."""
import numpy as np
from scipy.integrate import solve_ivp

def geodesic_rhs(t, x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    return [vx, vy, 2.0*(v2*px - 2.0*xv*vx)/one, 2.0*(v2*py - 2.0*xv*vy)/one]

def metric_speed(x):
    px, py, vx, vy = x
    return 2.0*np.hypot(vx, vy)/(1.0 - (px*px + py*py))

def hyp_dist(a, b):
    d = np.linalg.norm(a - b)/2.0
    d = min(d, 1.0 - 1e-12)
    return 2.0*np.arctanh(d)

px, py = 0.0, 0.2
r2 = px*px + py*py
vflat = (1.0 - r2)/2.0
vx0, vy0 = vflat, 0.0
x0 = np.array([px, py, vx0, vy0])
# perpendicular base-point perturbation (Jacobi direction), keep same velocity
ex, ey = -vy0, vx0
ex, ey = ex/np.hypot(ex, ey), ey/np.hypot(ex, ey)
d0 = 1e-3
y0 = np.array([px + d0*ex, py + d0*ey, vx0, vy0])
print("speeds:", metric_speed(x0), metric_speed(y0))
# initial Riemannian separation
d_init = hyp_dist(x0[:2], y0[:2])
print("initial Riemannian separation:", d_init)
solx = solve_ivp(geodesic_rhs, [0, 40], x0, method="Radau", rtol=1e-9, atol=1e-12, max_step=0.05)
soly = solve_ivp(geodesic_rhs, [0, 40], y0, method="Radau", rtol=1e-9, atol=1e-12, max_step=0.05)
for Tt in [5, 10, 20, 40]:
    i = int(Tt/0.05)
    # find index
    ix = np.searchsorted(solx.t, Tt); iy = np.searchsorted(soly.t, Tt)
    dx = solx.y[:, ix]; dy = soly.y[:, iy]
    dR = hyp_dist(dx[:2], dy[:2])
    print(f"T={Tt}: speed_x={metric_speed(dx):.5f} dR={dR:.6f} exponent={np.log(dR/d_init)/Tt:.4f}")
