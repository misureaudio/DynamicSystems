"""Probe 7: nu=0.1 A=0.2 NS — D_LY vs N, attractor state, mode amplitudes."""
import numpy as np
import time
import scipy.sparse as sp

def make_ns2d(N, nu, A):
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
    for (k, c) in [((1, 1), A / 4), ((1, -1), -A / 4), ((-1, 1), A / 4), ((-1, -1), -A / 4)]:
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

nu, A = 0.1, 0.2
for N in [4, 6, 8]:
    t0 = time.time()
    rhs, jac, n, ks, idx = make_ns2d(N, nu, A)
    om = even_init(ks, idx, 0.01, 1)
    for s in range(int(300/0.02)):
        om = rk4(rhs, om, 0.02)
    lam_c, om = benettin(rhs, jac, om, 200.0, 0.02, m=min(24, n))
    lam_r = real_spectrum(lam_c)
    D = kaplan_yorke(lam_r)
    pos = int(np.sum(lam_r > 1e-4))
    print(f"N={N}: t={time.time()-t0:.1f}s n={n} pos={pos} top8={np.round(lam_r[:8],4)} D_LY={D:.3f}")

# attractor state at N=6: top mode amplitudes + phase portrait data
rhs, jac, n, ks, idx = make_ns2d(6, nu, A)
om = even_init(ks, idx, 0.01, 1)
traj = []
for s in range(int(600/0.02)):
    om = rk4(rhs, om, 0.02)
    if s % 25 == 0:
        traj.append(np.array([om[idx[(1,1)]].real, om[idx[(1,1)]].imag,
                              om[idx[(2,0)]].real, om[idx[(1,0)]].real]))
traj = np.array(traj)
print("phase portrait (11Re,11Im,20Re,10Re): range per coord:",
      [f"{traj[:,i].min():.3f}..{traj[:,i].max():.3f}" for i in range(4)])
# top 8 mode amplitudes
amps = np.abs(om)
order = np.argsort(amps)[::-1][:8]
print("top modes:", [(ks[m], round(float(amps[m]), 3)) for m in order])

# two independent initial conditions -> same attractor?
om2 = even_init(ks, idx, 0.01, 99)
for s in range(int(600/0.02)):
    om2 = rk4(rhs, om2, 0.02)
d = np.linalg.norm(om - om2)
print(f"distance between two evolved states: {d:.3e} (small => attractor)")
