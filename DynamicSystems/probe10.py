"""Probe 10: geodesic flow Lyapunov exponent via perpendicular base-point perturbation."""
import numpy as np

def geodesic_rhs(x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    ax = (v2*px - 2.0*xv*vx)/one
    ay = (v2*py - 2.0*xv*vy)/one
    return np.array([vx, vy, ax, ay])

def rk4(f, x, dt):
    k1 = f(x); k2 = f(x+0.5*dt*k1); k3 = f(x+0.5*dt*k2); k4 = f(x+dt*k3)
    return x + (dt/6.0)*(k1+2*k2+2*k3+k4)

def metric_speed(x):
    px, py, vx, vy = x
    return 2.0*np.hypot(vx, vy)/(1.0 - (px*px + py*py))

def hyp_dist(a, b):
    d = np.linalg.norm(a - b)/2.0
    d = min(d, 0.999999)
    return 2.0*np.arctanh(d)

px, py = 0.0, 0.2
r2 = px*px + py*py
vflat = (1.0 - r2)/2.0
vx0, vy0 = vflat, 0.0
# perpendicular unit direction in the flat metric (good enough for tiny delta)
ex, ey = -vy0, vx0
ex, ey = ex/np.hypot(ex, ey), ey/np.hypot(ex, ey)
x0 = np.array([px, py, vx0, vy0])
d0 = 1e-4
y0 = np.array([px + d0*ex, py + d0*ey, vx0, vy0])
print("speeds:", metric_speed(x0), metric_speed(y0))
dt = 0.01
for Tt in [5, 10, 20, 40]:
    x = x0.copy(); y = y0.copy()
    for i in range(int(Tt/dt)):
        x = rk4(geodesic_rhs, x, dt); y = rk4(geodesic_rhs, y, dt)
    dR = hyp_dist(x[:2], y[:2])
    print(f"T={Tt}: dR={dR:.5f} ratio={dR/d0:.4f} exponent={np.log(dR/d0)/Tt:.4f}  (expect -> 1)")
