"""Probe 2: NS Galerkin (sparse analytic jac, complex Benettin) + Greene + RT + misc."""
import numpy as np
import time
import scipy.sparse as sp

# ---------- 1. 2D NS Galerkin, vorticity form, torus [0,2pi)^2 ----------
# om_t = -nu|k|^2 om_k + f_k + sum_{k1+k2=k} (km x kp) om_{kp} om_{km-kp}/|kp|^2   (km x kp = km1 kp2 - km2 kp1)
def make_ns2d(N, nu, A):
    ks = [(i, j) for i in range(-N, N + 1) for j in range(-N, N + 1)
          if 0 < i * i + j * j <= N * N]
    n = len(ks)
    karr = np.array(ks, float)
    k2 = np.hypot(karr[:, 0], karr[:, 1]) ** 2
    idx = {k: m for m, k in enumerate(ks)}
    # triples (i, j, l, c): M[i,j] = sum_l c * om_l ; c = (ki x kj)/|kj|^2, ki+? target:
    # M(om) @ om: (M om)_i = sum_j M[i,j] om_j, M[i,j] = sum_{l: k_j + k_l = k_i} (k_i x k_j)/|k_j|^2
    rows, cols, ls, cs = [], [], [], []
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            diff = (int(ki[0] - kj[0]), int(ki[1] - kj[1]))
            if diff in idx:
                cr = float(ki[0] * kj[1] - ki[1] * kj[0])
                if cr != 0.0:
                    l = idx[diff]
                    rows.append(i); cols.append(j); ls.append(l); cs.append(cr / k2[j])
    rows = np.array(rows); cols = np.array(cols); ls = np.array(ls); cs = np.array(cs, float)
    nnz = len(cs)
    fhat = np.zeros(n, complex)
    for (k, c) in [((1, 1), A / 4), ((1, -1), -A / 4), ((-1, 1), A / 4), ((-1, -1), -A / 4)]:
        if k in idx:
            fhat[idx[k]] = c
    dnu = nu * k2
    def M_assemble(om):
        # M[i,j] = sum_l cs[t] om_{ls[t]}
        data = cs * om[ls]
        return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    def rhs(om):
        return M_assemble(om) @ om - dnu * om + fhat
    def jac(om):
        # d(M(om) om)/dom : J[i,j] = M[i,j] + sum_{j'} cs[i,j',j] om[j']
        # extra term E[i,j] = sum_{t: rows=t.i, ls=t.l=j} cs[t] om[cols[t]]
        M = M_assemble(om)
        dataE = cs * om[cols]
        E = sp.coo_matrix((dataE, (rows, ls)), shape=(n, n)).tocsr()
        J = (M + E).tocsr()
        J = J - sp.diags(dnu)
        return J
    return rhs, jac, n, ks, idx

def rk4(f, x, dt):
    k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

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

t0 = time.time()
N, nu, A = 8, 0.025, 1.0
rhs, jac, n, ks, idx = make_ns2d(N, nu, A)
om = even_init(ks, idx, 0.01, 1)
T, dt = 100.0, 0.02
for s in range(int(T/dt)):
    om = rk4(rhs, om, dt)
E = np.sum(np.abs(om)**2) / 2
print(f"NS N={N} (n={n}): transient t={time.time()-t0:.1f}s, E={E:.4f}, "
      f"om(1,1)={om[idx[(1,1)]]:.4f} (linear ~ {A/4/(nu*2):.3f}), max|om|={np.abs(om).max():.3f}")

# energy-conservation check of nonlinear term (sanity: wrong sign would pump energy)
def energy(om): return 0.5*np.sum(np.abs(om)**2)
print("energy(om) =", energy(om))

# Benettin timing: complex tangent on n x n
def benettin_ns(rhs, jac, om0, T, dt, m=None):
    om = om0.copy()
    V = np.eye(n := om0.shape[0], dtype=complex)
    if m: V = V[:, :m]
    tot = np.zeros(V.shape[1])
    nsteps = int(T/dt)
    for s in range(nsteps):
        k1 = rhs(om); J1 = jac(om)
        k2 = rhs(om + 0.5*dt*k1); J2 = jac(om + 0.5*dt*k1)
        k3 = rhs(om + 0.5*dt*k2); J3 = jac(om + 0.5*dt*k2)
        k4 = rhs(om + dt*k3); J4 = jac(om + dt*k3)
        om = om + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V1 = J1 @ V
        V2 = J2 @ (V + 0.5*dt*V1)
        V3 = J3 @ (V + 0.5*dt*V2)
        V4 = J4 @ (V + dt*V3)
        V = V + (dt/6.0)*(V1 + 2*V2 + 2*V3 + V4)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.abs(np.diag(R)))
        sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R)))
        V = Q * sgn
    return tot/(nsteps*dt), om
t1 = time.time()
lam, om = benettin_ns(rhs, jac, om, 60.0, 0.05, m=12)
t2 = time.time()
print(f"Benettin NS N=8, T=60, dt=0.05, m=12: {t2-t1:.1f}s")
print("top complex exponents:", np.round(lam, 4))
# real spectrum = each complex exponent with multiplicity 2
lam_real = np.sort(np.concatenate([lam, lam]))[::-1]
# Kaplan-Yorke dimension on real spectrum
j = 0
s = 0.0
for i, l in enumerate(lam_real):
    if s + l >= 0:
        s += l; j = i+1
    else:
        break
if j < len(lam_real):
    D = j + s/abs(lam_real[j])
else:
    D = float(j)
print(f"D_LY (real) = {D:.3f}, j={j}")
