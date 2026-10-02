"""Probe 8: scan richer forcing for a low-D_LY chaotic NS regime."""
import numpy as np
import time
import scipy.sparse as sp

def make_ns2d(N, nu, fdict):
    ks = [(i, j) for i in range(-N, N + 1) for j in range(-N, N + 1)
          if 0 < i * i + j * j <= N * N]
    n = len(ks)
    karr = np.array(ks, float)
    k2 = np.hypot(karr[:, 0], karr[:, 1]) ** 2
    idx = {k: m for m, k in enumerate(ks)}
    rows, cols, ls, cs = [], [], [], []
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            diff = (int(ki[0] - kj[0]), int(ki[1] - kj[1]))
            if diff in idx:
                cr = float(ki[0] * kj[1] - ki[1] * kj[0])
                if cr != 0.0:
                    rows.append(i); cols.append(j); ls.append(idx[diff]); cs.append(cr / k2[j])
    rows = np.array(rows); cols = np.array(cols); ls = np.array(ls); cs = np.array(cs, float)
    fhat = np.zeros(n, complex)
    for (k, c) in fdict.items():
        if k in idx:
            fhat[idx[k]] = c
    dnu = nu * k2
    def M_assemble(om):
        return sp.coo_matrix((cs * om[ls], (rows, cols)), shape=(n, n)).tocsr()
    def rhs(om):
        return M_assemble(om) @ om - dnu * om + fhat
    def jac(om):
        M = M_assemble(om)
        E = sp.coo_matrix((cs * om[cols], (rows, ls)), shape=(n, n)).tocsr()
        return (M + E).tocsr() - sp.diags(dnu)
    return rhs, jac, n, ks, idx

def even_init(ks, idx, scale, seed):
    rng = np.random.default_rng(seed)
    om = np.zeros(len(ks), complex)
    for m, k in enumerate(ks):
        if (k[0], k[1]) >= (0, 0):
            om[m] = scale * rng.random()
    for m, k in enumerate(ks):
        km = (-k[0], -k[1])
        if km in idx and m > idx[km]:
            om[m] = np.conj(om[idx[km]])
    return om

def rk4(f, x, dt):
    k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def benettin(rhs, jac, om0, T, dt, m):
    om = om0.copy(); n = om0.shape[0]
    V = np.eye(n, dtype=complex)[:, :m]
    tot = np.zeros(m); nsteps = int(T/dt)
    for s in range(nsteps):
        k1 = rhs(om); J1 = jac(om)
        k2 = rhs(om + 0.5*dt*k1); J2 = jac(om + 0.5*dt*k1)
        k3 = rhs(om + 0.5*dt*k2); J3 = jac(om + 0.5*dt*k2)
        k4 = rhs(om + dt*k3); J4 = jac(om + dt*k3)
        om = om + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V1 = J1 @ V; V2 = J2 @ (V + 0.5*dt*V1); V3 = J3 @ (V + 0.5*dt*V2); V4 = J4 @ (V + dt*V3)
        V = V + (dt/6.0)*(V1 + 2*V2 + 2*V3 + V4)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.abs(np.diag(R))); sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R))); V = Q * sgn
    return tot/(nsteps*dt), om

def real_spectrum(lam_c):
    return np.sort(np.concatenate([np.real(lam_c), np.real(lam_c)]))[::-1]

def kaplan_yorke(lam_real):
    s = 0.0; j = 0
    for i, l in enumerate(lam_real):
        if s + l >= 0:
            s += l; j = i + 1
        else:
            break
    if j == 0: return 0.0
    if j < len(lam_real): return j + s/abs(lam_real[j])
    return float(j)

# richer forcing: (1,1) pair + (2,1) pair
def fdict2(A):
    return {
        (1, 1): A/4, (1, -1): -A/4, (-1, 1): A/4, (-1, -1): -A/4,
        (2, 1): A/6, (2, -1): -A/6, (-2, 1): A/6, (-2, -1): -A/6,
    }
print("=== 2-mode forcing scan (N=6) ===", flush=True)
for (nu, A) in [(0.05, 0.5), (0.05, 0.8), (0.05, 1.2), (0.08, 0.8), (0.08, 1.2),
                (0.1, 0.8), (0.1, 1.2), (0.12, 1.2), (0.15, 1.2), (0.15, 1.5)]:
    t0 = time.time()
    try:
        rhs, jac, n, ks, idx = make_ns2d(6, nu, fdict2(A))
        om = even_init(ks, idx, 0.01, 1)
        for s in range(int(300/0.02)):
            om = rk4(rhs, om, 0.02)
        lam_c, om = benettin(rhs, jac, om, 150.0, 0.02, m=min(30, n))
        lam_r = real_spectrum(lam_c)
        D = kaplan_yorke(lam_r)
        pos = int(np.sum(lam_r > 1e-3))
        print(f"nu={nu}, A={A}: t={time.time()-t0:.0f}s pos={pos} top4={np.round(lam_r[:4],3)} D_LY={D:.3f}", flush=True)
    except Exception as e:
        print(f"nu={nu}, A={A}: ERR {e}", flush=True)
print("=== done ===", flush=True)
