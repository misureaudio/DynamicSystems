"""Probe 28: Greene K_c via (a) closest-to-parabolic orbit selection, (b) rotation number."""
import numpy as np
import time

def period_q_orbit(K, p, q, I0_guess, th0_guess=0.0, maxit=100, tol=1e-13):
    x = np.array([th0_guess, I0_guess], float)
    target = np.array([2*np.pi*p, 0.0])
    def Fq(x):
        th, I = x; M = np.eye(2)
        for _ in range(q):
            s, c = np.sin(th), np.cos(th)
            M = np.array([[1 + K*c, 1.0], [K*c, 1.0]]) @ M
            In = I + K*s; th, I = th + In, In
        return np.array([th, I]), M
    conv = False
    for it in range(maxit):
        y, M = Fq(x); resid = y - x - target
        if np.linalg.norm(resid) < tol: conv = True; break
        try: dx = np.linalg.solve(M - np.eye(2), resid)
        except np.linalg.LinAlgError: break
        r0 = np.linalg.norm(resid); step = 1.0; x_new = x - dx
        for _ in range(30):
            y2, _ = Fq(x_new); r2 = np.linalg.norm(y2 - x_new - target)
            if r2 < r0: break
            step *= 0.5; x_new = x - step*dx
        x = x_new
    y, M = Fq(x)
    return x, M, conv

def find_all(K, p, q, n_starts=16):
    I_center = 2*np.pi*p/q
    out = []
    for i in range(n_starts):
        for th0 in np.linspace(0, 2*np.pi, 6, endpoint=False):
            for dI in np.linspace(-0.6, 0.6, 5):
                x, M, conv = period_q_orbit(K, p, q, I_center + dI, th0)
                if not conv: continue
                tr2 = np.trace(M)/2
                # dedup
                if any(np.linalg.norm(x - o[0]) < 1e-5 for o in out): continue
                out.append((x, tr2))
    return out

def residue(tr2):
    return (1 - tr2)**2/4

# (a) closest-to-parabolic selection: among convergent orbits, the one with tr2 nearest 1
#     (the golden torus orbit is parabolic at K_c).  Track its elliptic/hyperbolic status.
t0 = time.time()
Ks = np.linspace(0.9, 1.05, 8)
print("(a) closest-to-parabolic orbit (p,q)=(8,13):", flush=True)
for K in Ks:
    orbits = find_all(K, 8, 13, n_starts=10)
    if not orbits:
        print(f"  K={K:.3f}: none", flush=True); continue
    # pick orbit with tr2 closest to 1
    x, tr2 = min(orbits, key=lambda o: abs(o[1] - 1.0))
    status = "elliptic" if abs(tr2) < 1 else "hyperbolic"
    print(f"  K={K:.3f}: tr/2={tr2:.5f} R={residue(tr2):.5f} {status} (n_orbits={len(orbits)})", flush=True)
print(f"time: {time.time()-t0:.1f}s", flush=True)

# (b) rotation number: start near golden torus action, iterate, measure rotation number.
# The golden torus action I_gold: find by locating where a long orbit's rotation number
# is closest to the golden mean and stays bounded.
GOLDEN = 0.618034
def rot_number(K, th0, I0, N=20000):
    th, I = th0, I0
    th_start = th
    for _ in range(N):
        In = I + K*np.sin(th); th, I = th + In, In
    return (th - th_start)/(2*np.pi*N)
print("\n(b) rotation number scan at I=4.18 (near golden torus), th0=0:", flush=True)
for K in [0.9, 0.95, 0.97, 0.9716, 0.98, 1.0, 1.05]:
    for I0 in [4.0, 4.1, 4.18, 4.3]:
        rho = rot_number(K, 0.0, I0, N=20000)
        print(f"  K={K:.4f} I0={I0}: rho={rho:.5f}  (golden={GOLDEN:.5f}, |diff|={abs(rho-GOLDEN):.4f})", flush=True)
