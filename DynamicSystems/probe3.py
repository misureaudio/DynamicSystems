"""Probe 3: fixed Greene target, geodesic ODE sign + 2-geodesic separation, RT cascade tuning."""
import numpy as np
import time

# ---------- 1. Greene residue, target (2*pi*p, 0) ----------
def std_period_q(K, p, q, I0_guess, th0=0.0, maxit=300, tol=1e-13):
    x = np.array([th0, I0_guess])
    target = np.array([2*np.pi*p, 0.0])
    def Fq(x):
        y = x.copy(); M = np.eye(2)
        for _ in range(q):
            th, I = y
            s, c = np.sin(th), np.cos(th)
            M = np.array([[1 + K*c, 1.0], [K*c, 1.0]]) @ M
            y = np.array([th + I + K*s, I + K*s])
        return y, M
    conv = False; best = None
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
        step = 1.0
        x_new = x - dx
        for _ in range(30):
            y2, _ = Fq(x_new)
            r2 = np.linalg.norm(y2 - x_new - target)
            if r2 < r0:
                break
            step *= 0.5
            x_new = x - step*dx
        x = x_new
    y, M = Fq(x)
    tr = np.trace(M)
    R = (1 - tr/2)**2/4
    return x, R, conv, np.linalg.norm(y - x - target)

# locate golden I0 at K=0.9716
def std_rot_num_vec(K, I0s, N=5000):
    th = np.zeros_like(I0s); I = I0s.copy(); th0 = np.zeros_like(I0s)
    for _ in range(N):
        In = I + K*np.sin(th); th = th + In; I = In
    return (th - th0)/(2*np.pi*N), I

K = 0.9716
I0s = np.linspace(0, 2*np.pi, 800)
rho, Ifin = std_rot_num_vec(K, I0s, N=5000)
spread = np.abs(Ifin - I0s)
i_g = int(np.argmin(np.where(spread < 1.5, np.abs(rho - 0.618), 1e9)))
print(f"golden I0 ~ {I0s[i_g]:.4f} (rho={rho[i_g]:.4f}, spread={spread[i_g]:.3f})")
t0 = time.time()
for Kt in [0.95, 0.9716, 1.0, 1.05]:
    line = f"K={Kt}: "
    for (p, q) in [(3,5),(5,8),(8,13),(13,21),(21,34)]:
        x, R, conv, err = std_period_q(Kt, p, q, I0s[i_g])
        line += f"q={q}: R={R:.3f} err={err:.1e} "
    print(line)
print(f"greene time: {time.time()-t0:.1f}s")

# ---------- 2. Geodesic flow: fixed ODE, speed conservation, 2-geodesic separation ----------
def geodesic_rhs(x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    # acc = 2[(v.v)x - 2(x.v)v]/(1-r2)
    ax = 2.0*(v2*px - 2.0*xv*vx)/one
    ay = 2.0*(v2*py - 2.0*xv*vy)/one
    return np.array([vx, vy, ax, ay])

def metric_speed(x):
    px, py, vx, vy = x
    return 2.0*np.hypot(vx, vy)/(1.0 - (px*px + py*py))

# pick initial condition with unit metric speed: |v|_g = 2|v|/(1-r2) = 1
px, py = 0.0, 0.2
r2 = px*px + py*py
vflat = (1.0 - r2)/2.0
x0 = np.array([px, py, vflat, 0.0])
print("initial metric speed:", metric_speed(x0))
# integrate and check speed conservation
def rk4(f, x, dt):
    k1 = f(x); k2 = f(x+0.5*dt*k1); k3 = f(x+0.5*dt*k2); k4 = f(x+dt*k3)
    return x + (dt/6.0)*(k1+2*k2+2*k3+k4)
x = x0.copy(); dt = 0.01
speeds = []
for i in range(2000):
    x = rk4(geodesic_rhs, x, dt)
    if i % 200 == 0:
        speeds.append(metric_speed(x))
print("speed over time:", np.round(speeds, 5))
print("position r_end:", np.hypot(x[0], x[1]))

# 2-geodesic separation for largest exponent
def sep_exponent(x0, d0, T, dt=0.01):
    y0 = x0.copy(); y0[2] += d0  # perturb velocity
    x = x0.copy(); y = y0.copy()
    lam = 0.0
    n = int(T/dt)
    for i in range(n):
        x = rk4(geodesic_rhs, x, dt); y = rk4(geodesic_rhs, y, dt)
    # distance between two points on unit tangent bundle ~ Euclidean in (x,v)
    d = np.linalg.norm(x - y)
    return np.log(d/d0)/T
for d0 in [1e-3, 1e-4]:
    print(f"sep exponent d0={d0}:", sep_exponent(x0, d0, 40.0))

# ---------- 3. RT cascade tuning ----------
OMEGA_RT = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
def rt_rhs(z, mu, b):
    a, s2, s3 = 1.0, 0.5, 1.0
    z1, z2, z3 = z
    return np.array([
        (mu + 1j*OMEGA_RT[0])*z1 - a*np.abs(z1)**2*z1 - b*z2*np.conj(z3),
        (mu - s2 + 1j*OMEGA_RT[1])*z2 - a*np.abs(z2)**2*z2 - b*z1*np.conj(z3),
        (mu - s3 + 1j*OMEGA_RT[2])*z3 - a*np.abs(z3)**2*z3 - b*z1*np.conj(z2)])
def rt_state(mu, b, T=600.0, dt=0.02):
    z = np.array([0.1+0.05j, 0.1+0.05j, 0.1+0.05j], complex)
    n = int(T/dt)
    for i in range(n):
        k1 = rt_rhs(z, mu, b); k2 = rt_rhs(z+0.5*dt*k1, mu, b)
        k3 = rt_rhs(z+0.5*dt*k2, mu, b); k4 = rt_rhs(z+dt*k3, mu, b)
        z = z + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return z
def rt_lam1(mu, b, T=1000.0, dt=0.02):
    z = np.array([0.1+0.05j, 0.1+0.05j, 0.1+0.05j], complex)
    def F(x):
        zz = x[:3] + 1j*x[3:]
        out = rt_rhs(zz, mu, b)
        return np.concatenate([out.real, out.imag])
    x = np.concatenate([z.real, z.imag])
    V = np.eye(6); tot = 0.0; n = int(T/dt)
    for i in range(n):
        k1 = F(x); k2 = F(x+0.5*dt*k1); k3 = F(x+0.5*dt*k2); k4 = F(x+dt*k3)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        h = 1e-6; J = np.zeros((6,6)); f0 = F(x)
        for j in range(6):
            xp = x.copy(); xp[j] += h
            J[:, j] = (F(xp) - f0)/h
        V = V + dt*(J@V)
        Q, Rr = np.linalg.qr(V); sgn = np.sign(np.diag(Rr)); sgn[sgn==0]=1
        tot += np.log(abs(np.diag(Rr)[0])); V = Q*sgn
    return tot/(n*dt)
# mode amplitudes across the cascade
for mu in [0.0, 0.3, 0.6, 0.9, 1.3, 1.6]:
    z = rt_state(mu, 0.5, T=400)
    print(f"mu={mu}: |z1|={np.abs(z[0]):.3f} |z2|={np.abs(z[1]):.3f} |z3|={np.abs(z[2]):.3f}")
for b in [0.5, 1.0, 1.5, 2.0]:
    print(f"RT lam1 (b={b}, mu=1.6):", rt_lam1(1.6, b))
