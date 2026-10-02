"""Probe 4: Greene multi-start, geodesic Riemannian separation, RT 4-mode cascade."""
import numpy as np
import time

# ---------- 1. Greene: multi-start, max residue among converged ----------
def std_period_q_multi(K, p, q, I_center, n_starts=9, maxit=200, tol=1e-13):
    target = np.array([2*np.pi*p, 0.0])
    def Fq(x):
        y = x.copy(); M = np.eye(2)
        for _ in range(q):
            th, I = y
            s, c = np.sin(th), np.cos(th)
            M = np.array([[1 + K*c, 1.0], [K*c, 1.0]]) @ M
            y = np.array([th + I + K*s, I + K*s])
        return y, M
    best_R = None; best_x = None; best_err = None
    for th0 in np.linspace(0, 2*np.pi, n_starts, endpoint=False):
        for dI in [-0.15, 0.0, 0.15]:
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
            # dedup: keep the max residue (most unstable = golden-chain orbit)
            if best_R is None or R > best_R:
                best_R, best_x, best_err = R, x, np.linalg.norm(y - x - target)
    return best_R, best_x, best_err

t0 = time.time()
Ks = np.linspace(0.9, 1.1, 9)
for (p, q) in [(8, 13), (13, 21), (21, 34)]:
    I_center = 2*np.pi*p/q
    row = f"q={q} (I_c={I_center:.3f}): "
    for Kt in Ks:
        R, x, err = std_period_q_multi(Kt, p, q, I_center, n_starts=4)
        row += f"{R if R is not None else float('nan'):.3f} "
    print(row, flush=True)
print(f"greene multi time: {time.time()-t0:.1f}s")

# ---------- 2. Geodesic: Riemannian base separation ----------
def geodesic_rhs(x):
    px, py, vx, vy = x
    r2 = px*px + py*py; one = 1.0 - r2
    v2 = vx*vx + vy*vy; xv = px*vx + py*vy
    ax = 2.0*(v2*px - 2.0*xv*vx)/one
    ay = 2.0*(v2*py - 2.0*xv*vy)/one
    return np.array([vx, vy, ax, ay])
def rk4(f, x, dt):
    k1 = f(x); k2 = f(x+0.5*dt*k1); k3 = f(x+0.5*dt*k2); k4 = f(x+dt*k3)
    return x + (dt/6.0)*(k1+2*k2+2*k3+k4)
def hyp_dist(a, b):
    d = np.linalg.norm(a - b)/2.0
    d = min(d, 0.999999)
    return 2.0*np.arctanh(d)
px, py = 0.0, 0.2
r2 = px*px + py*py
vflat = (1.0 - r2)/2.0
x0 = np.array([px, py, vflat, 0.0])
d0 = 1e-4
y0 = x0.copy(); y0[2] += d0
delta0_R = 2.0*d0/(1.0 - r2)  # Riemannian initial velocity separation
T = 30.0; dt = 0.01
x = x0.copy(); y = y0.copy()
for i in range(int(T/dt)):
    x = rk4(geodesic_rhs, x, dt); y = rk4(geodesic_rhs, y, dt)
dR = hyp_dist(x[:2], y[:2])
exponent = np.log(dR/delta0_R)/T
print(f"geodesic Riemannian separation: dR(T)={dR:.4f}, delta0_R={delta0_R:.2e}, exponent={exponent:.4f} (expect ~1)")
# also at shorter T to see convergence
for Tt in [5, 10, 20]:
    x = x0.copy(); y = y0.copy()
    for i in range(int(Tt/dt)):
        x = rk4(geodesic_rhs, x, dt); y = rk4(geodesic_rhs, y, dt)
    dR = hyp_dist(x[:2], y[:2])
    print(f"  T={Tt}: exponent={np.log(dR/delta0_R)/Tt:.4f}")

# ---------- 3. RT 4-mode cascade ----------
OM = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0)])
def rt4_rhs(z, mu, b, a=1.0):
    s = np.array([0.0, 0.5, 1.0, 1.6])
    out = np.empty(4, complex)
    for i in range(4):
        out[i] = (mu - s[i] + 1j*OM[i])*z[i] - a*np.abs(z[i])**2*z[i]
    out[0] -= b*z1_c(z)
    return out
def z1_c(z): return 0
# write explicitly to avoid confusion
def rt4(z, mu, b, a=1.0):
    s = np.array([0.0, 0.5, 1.0, 1.6])
    z1, z2, z3, z4 = z
    return np.array([
        (mu + 1j*OM[0])*z1 - a*np.abs(z1)**2*z1 - b*(z2*np.conj(z3) + z4*np.conj(z3)),
        (mu - s[1] + 1j*OM[1])*z2 - a*np.abs(z2)**2*z2 - b*(z1*np.conj(z3) + z4*np.conj(z1)),
        (mu - s[2] + 1j*OM[2])*z3 - a*np.abs(z3)**2*z3 - b*(z1*np.conj(z2) + z1*np.conj(z4)),
        (mu - s[3] + 1j*OM[3])*z4 - a*np.abs(z4)**2*z4 - b*(z3*np.conj(z1) + z2*np.conj(z1))])
def rt4_state(mu, b, T=500.0, dt=0.02):
    z = np.array([0.1+0.05j, 0.1+0.05j, 0.1+0.05j, 0.1+0.05j], complex)
    n = int(T/dt)
    for i in range(n):
        k1 = rt4(z, mu, b); k2 = rt4(z+0.5*dt*k1, mu, b)
        k3 = rt4(z+0.5*dt*k2, mu, b); k4 = rt4(z+dt*k3, mu, b)
        z = z + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return z
def rt4_lam1(mu, b, T=1200.0, dt=0.02):
    z = np.array([0.1+0.05j, 0.1+0.05j, 0.1+0.05j, 0.1+0.05j], complex)
    def F(x):
        zz = x[:4] + 1j*x[4:]
        out = rt4(zz, mu, b)
        return np.concatenate([out.real, out.imag])
    x = np.concatenate([z.real, z.imag])
    V = np.eye(8); tot = 0.0; n = int(T/dt)
    for i in range(n):
        k1 = F(x); k2 = F(x+0.5*dt*k1); k3 = F(x+0.5*dt*k2); k4 = F(x+dt*k3)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        h = 1e-6; J = np.zeros((8,8)); f0 = F(x)
        for j in range(8):
            xp = x.copy(); xp[j] += h
            J[:, j] = (F(xp) - f0)/h
        V = V + dt*(J@V)
        Q, Rr = np.linalg.qr(V); sgn = np.sign(np.diag(Rr)); sgn[sgn==0]=1
        tot += np.log(abs(np.diag(Rr)[0])); V = Q*sgn
    return tot/(n*dt)
for mu in [0.0, 0.3, 0.6, 1.0, 1.4, 1.8, 2.2]:
    z = rt4_state(mu, 0.5, T=400)
    print(f"mu={mu}: " + " ".join(f"|z{i+1}|={np.abs(z[i]):.3f}" for i in range(4)))
for mu in [1.8, 2.2, 2.6, 3.0]:
    print(f"RT4 lam1 (b=0.5, mu={mu}):", rt4_lam1(mu, 0.5))
