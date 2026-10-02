"""Probe 9: correct geodesic ODE (no factor 2) + direction perturbation; Greene I_center=4.18."""
import numpy as np
import time

# ---------- 1. Geodesic flow, Poincare disk K=-1 ----------
# Christoffel: Gamma^i_jk = (x_j delta^ik + x_k delta^ij - x^i delta_jk)/(1-r^2)
# => acc = [(v.v)x - 2(x.v)v]/(1-r^2)
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
x0 = np.array([px, py, vflat, 0.0])
print("initial metric speed:", metric_speed(x0))
# speed conservation
x = x0.copy(); dt = 0.01
sp = [metric_speed(rk4(geodesic_rhs, x, dt)) for _ in range(1000)]
print("speed after 10s:", min(sp), max(sp))
# direction perturbation: rotate velocity by small angle, keep metric speed
eps = 1e-4
c, s = np.cos(eps), np.sin(eps)
v_rot = np.array([c*x0[2] - s*x0[3], s*x0[2] + c*x0[3]])
y0 = np.array([x0[0], x0[1], v_rot[0], v_rot[1]])
print("perturbed metric speed:", metric_speed(y0))
delta0_R = eps  # angle on unit tangent sphere ~ initial base separation scale; use hyp_dist directly
x = x0.copy(); y = y0.copy()
for Tt in [5, 10, 20, 40]:
    x = x0.copy(); y = y0.copy()
    for i in range(int(Tt/dt)):
        x = rk4(geodesic_rhs, x, dt); y = rk4(geodesic_rhs, y, dt)
    dR = hyp_dist(x[:2], y[:2])
    print(f"T={Tt}: dR={dR:.5f} exponent={np.log(dR)/Tt:.4f}  (expect -> 1)")

# ---------- 2. Greene with I_center at golden torus ----------
def std_period_q_multi(K, p, q, I_center, dI_range=0.5, n_starts=4, maxit=200, tol=1e-13):
    target = np.array([2*np.pi*p, 0.0])
    def Fq(x):
        y = x.copy(); M = np.eye(2)
        for _ in range(q):
            th, I = y
            s, c = np.sin(th), np.cos(th)
            M = np.array([[1 + K*c, 1.0], [K*c, 1.0]]) @ M
            y = np.array([th + I + K*s, I + K*s])
        return y, M
    best_R = None
    for th0 in np.linspace(0, 2*np.pi, n_starts, endpoint=False):
        for dI in np.linspace(-dI_range, dI_range, 5):
            x = np.array([th0, I_center + dI])
            conv = False
            for it in range(maxit):
                y, M = Fq(x)
                resid = y - x - target
                if np.linalg.norm(resid) < tol:
                    conv = True; break
                try:
                    dx = np.linalg.solve(M, resid)
                except np.linalg.LinAlgError:
                    break
                r0 = np.linalg.norm(resid)
                step = 1.0; x_new = x - dx
                for _ in range(25):
                    y2, _ = Fq(x_new)
                    r2 = np.linalg.norm(y2 - x_new - target)
                    if r2 < r0: break
                    step *= 0.5; x_new = x - step*dx
                x = x_new
            if not conv:
                continue
            y, M = Fq(x)
            tr = np.trace(M)
            R = (1 - tr/2)**2/4
            if best_R is None or R > best_R:
                best_R = R
    return best_R

t0 = time.time()
Ks = np.linspace(0.9, 1.1, 9)
I_gold = 4.1757
for (p, q) in [(8, 13), (13, 21), (21, 34)]:
    row = f"q={q}: "
    for Kt in Ks:
        R = std_period_q_multi(Kt, p, q, I_gold)
        row += f"{R if R is not None else float('nan'):.4f} "
    print(row, flush=True)
print(f"K grid: {np.round(Ks,3)}", flush=True)
print(f"greene time: {time.time()-t0:.1f}s", flush=True)
