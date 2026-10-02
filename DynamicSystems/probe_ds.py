"""Probe 1 (trimmed): Greene residue pipeline, RT cascade, Henon period-2, geodesic flow."""
import numpy as np
import time
from scipy.optimize import fsolve

# ---------- Greene residue pipeline ----------
def std_rot_num_vec(K, I0s, N=5000):
    th = np.zeros_like(I0s); I = I0s.copy()
    th0 = np.zeros_like(I0s)
    for _ in range(N):
        In = I + K*np.sin(th)
        th = th + In
        I = In
    return (th - th0)/(2*np.pi*N), I

def std_period_q(K, p, q, I0_guess, th0=0.0, maxit=200, tol=1e-12):
    x = np.array([th0, I0_guess])
    target = np.array([2*np.pi*p, 2*np.pi*q])
    def Fq(x):
        y = x.copy(); M = np.eye(2)
        for _ in range(q):
            th, I = y
            s, c = np.sin(th), np.cos(th)
            M = np.array([[1 + K*c, 1.0], [K*c, 1.0]]) @ M
            y = np.array([th + I + K*s, I + K*s])
        return y, M
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
        step = 1.0
        x_new = x - dx
        for _ in range(25):
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
    return x, R, conv

t0 = time.time()
K = 0.9716
I0s = np.linspace(0, 2*np.pi, 800)
rho, Ifin = std_rot_num_vec(K, I0s, N=5000)
spread = np.abs(Ifin - I0s)
cand = spread < 1.5
i_g = int(np.argmin(np.where(cand, np.abs(rho - 0.618), 1e9)))
print(f"golden I0 ~ {I0s[i_g]:.4f} (rho={rho[i_g]:.4f}, spread={spread[i_g]:.3f})")
for Kt in [0.9, 0.9716, 1.0, 1.1]:
    Rvals = {}
    for (p, q) in [(3,5),(5,8),(8,13),(13,21),(21,34)]:
        x, R, conv = std_period_q(Kt, p, q, I0s[i_g])
        Rvals[q] = (R, conv)
    print(f"K={Kt}: " + " ".join(f"q={q}: R={Rv[0]:.3f} conv={Rv[1]}" for q, Rv in Rvals.items()))
print(f"greene probe time: {time.time()-t0:.1f}s")

# ---------- RT cascade ----------
OMEGA_RT = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
def rt_rhs(z, mu, b):
    a, s2, s3 = 1.0, 0.5, 1.0
    z1, z2, z3 = z
    return np.array([
        (mu + 1j*OMEGA_RT[0])*z1 - a*np.abs(z1)**2*z1 - b*z2*np.conj(z3),
        (mu - s2 + 1j*OMEGA_RT[1])*z2 - a*np.abs(z2)**2*z2 - b*z1*np.conj(z3),
        (mu - s3 + 1j*OMEGA_RT[2])*z3 - a*np.abs(z3)**2*z3 - b*z1*np.conj(z2)])

def rt_lam1(mu, b, T=800.0, dt=0.02):
    z = np.array([0.1+0.05j, 0.1+0.05j, 0.1+0.05j], complex)
    def F(x):
        zz = x[:3] + 1j*x[3:]
        out = rt_rhs(zz, mu, b)
        return np.concatenate([out.real, out.imag])
    x = np.concatenate([z.real, z.imag])
    V = np.eye(6)
    tot = 0.0
    nsteps = int(T/dt)
    for i in range(nsteps):
        k1 = F(x); k2 = F(x+0.5*dt*k1); k3 = F(x+0.5*dt*k2); k4 = F(x+dt*k3)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        h = 1e-6
        J = np.zeros((6,6)); f0 = F(x)
        for j in range(6):
            xp = x.copy(); xp[j] += h
            J[:, j] = (F(xp) - f0)/h
        V = V + dt*(J@V)
        Q, Rr = np.linalg.qr(V)
        sgn = np.sign(np.diag(Rr)); sgn[sgn==0] = 1
        tot += np.log(abs(np.diag(Rr)[0]))
        V = Q*sgn
    return tot/(nsteps*dt)

print("RT lam1 (b=0.1, mu=1.5):", rt_lam1(1.5, 0.1))
print("RT lam1 (b=0.8, mu=1.5):", rt_lam1(1.5, 0.8))

# ---------- Henon period-2 ----------
a, b = 1.4, 0.3
def H(x): return np.array([1 - a*x[0]**2 + x[1], b*x[0]])
def Hinv(x): return np.array([x[1]/b, x[0] - 1 + a*(x[1]/b)**2])
p = np.array([0.0, 0.0])
for _ in range(100): p = H(p)
print("henon H(Hinv(p)) - p:", H(Hinv(p)) - p)
sols = []
rng = np.random.default_rng(3)
for _ in range(60):
    x0 = rng.uniform(-1.5, 1.5, 4)
    def eqs(v):
        x1, y1, x2, y2 = v
        x2c = 1 - a*x1*x1 + y1
        y2c = b*x1
        x1c = 1 - a*x2c*x2c + y2c
        y1c = b*x2c
        return [x2c - x2, y2c - y2, x1c - x1, y1c - y1]
    v, info, ier, msg = fsolve(eqs, x0, full_output=True)
    if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-10:
        v = np.round(v, 10)
        if not any(np.max(np.abs(v - s)) < 1e-6 for s in sols):
            sols.append(v)
print(f"period-2 candidates: {len(sols)}")
for s in sols:
    p1 = s[:2]
    print("  ", np.round(s, 6), " H(p1)=p2?", np.max(np.abs(H(p1) - s[2:])), " fixed?", np.max(np.abs(H(p1)-p1)) < 1e-8)

# ---------- Geodesic flow Benettin ----------
def geodesic_rhs(x):
    px, py, vx, vy = x
    one = 1.0 - (px*px + py*py)
    acc = (4.0*(px*vx + py*vy)*np.array([vx, vy]) - 2.0*np.array([px, py])*(vx*vx + vy*vy))/one
    return np.array([vx, vy, acc[0], acc[1]])
x0 = np.array([0.0, 0.2, (1-0.04)/2, 0.0])
V = np.eye(4); tot = np.zeros(4); nsteps = int(60.0/0.02); dt = 0.02
x = x0.copy()
for i in range(nsteps):
    k1 = geodesic_rhs(x); k2 = geodesic_rhs(x+0.5*dt*k1); k3 = geodesic_rhs(x+0.5*dt*k2); k4 = geodesic_rhs(x+dt*k3)
    x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
    h = 1e-6; J = np.zeros((4,4)); f0 = geodesic_rhs(x)
    for j in range(4):
        xp = x.copy(); xp[j] += h
        J[:, j] = (geodesic_rhs(xp) - f0)/h
    V = V + dt*(J@V)
    Q, Rr = np.linalg.qr(V); sgn = np.sign(np.diag(Rr)); sgn[sgn==0]=1
    tot += np.log(np.abs(np.diag(Rr)))
    V = Q*sgn
print("geodesic exponents:", np.round(tot/(nsteps*dt), 4), " r_end:", np.hypot(x[0], x[1]))
