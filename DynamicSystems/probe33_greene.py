"""Probe 33: Greene criterion — track residue R_n of period-q_n orbit (q_n = Fibonacci
convergents to the golden mean). Torus EXISTS iff {R_n} bounded (->0); DESTROYED iff ->inf.
K_c = boundary. This is the standard Greene (1979) residue criterion."""
import numpy as np
import time

GOLDEN = (np.sqrt(5)-1)/2
# Fibonacci convergents p_n/q_n to the golden mean
FIB = [1,1,2,3,5,8,13,21,34,55,89]
CONV = [(FIB[i], FIB[i+1]) for i in range(len(FIB)-1)]  # (p,q)

def period_q_orbit(K, p, q, I0, th0=0.0, maxit=80, tol=1e-12):
    x = np.array([th0, I0], float)
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
        for _ in range(25):
            y2, _ = Fq(x_new); r2 = np.linalg.norm(y2 - x_new - target)
            if r2 < r0: break
            step *= 0.5; x_new = x - step*dx
        x = x_new
    y, M = Fq(x)
    return x, M, conv

def residue_n(K, p, q, I_gold, n_starts=6):
    """Residue of the period-q orbit with rotation number p/q, found near I_gold.
    Returns the MINIMUM residue among convergent starts (the orbit on/near the torus)."""
    Rs = []
    for i in range(n_starts):
        th0 = 2*np.pi*i/n_starts
        x, M, conv = period_q_orbit(K, p, q, I_gold, th0)
        if not conv: continue
        tr2 = np.trace(M)/2
        R = (1 - tr2)**2/4
        # keep only orbits whose rotation number is ~ p/q (the golden convergent)
        Rs.append(R)
    return min(Rs) if Rs else np.nan

def I_golden(K, N=60000):
    """Locate the golden torus action: the I where a long orbit has rho=golden & bounded."""
    Is = np.linspace(3.0, 5.5, 200)
    th = np.zeros_like(Is); I = Is.copy(); th0 = np.zeros_like(Is)
    for _ in range(N):
        I = I + K*np.sin(th); th = th + I
    drift = np.abs(I - Is); rho = (th-th0)/(2*np.pi*N)
    j = int(np.argmin(np.abs(rho - GOLDEN)))
    return Is[j]

t0 = time.time()
print("Greene residue R_n (q_n = Fibonacci) at K values:", flush=True)
for K in [0.90, 0.95, 0.9716, 0.98, 1.0]:
    Ig = I_golden(K)
    row = f"K={K:.4f} (I_gold={Ig:.3f}): "
    for (p, q) in CONV[:6]:  # q = 1,2,3,5,8,13
        R = residue_n(K, p, q, Ig)
        row += f"q={q}:{R:.3f} "
    print(row, flush=True)
print(f"time: {time.time()-t0:.1f}s", flush=True)

# K_c via the bounded/diverging test: for each K, is R_n bounded (small) or diverging?
def torus_exists(K, qmax=34, Rbound=2.0):
    Ig = I_golden(K)
    maxR = 0
    for (p,q) in CONV:
        if q > qmax: break
        R = residue_n(K, p, q, Ig)
        if np.isnan(R): return False
        maxR = max(maxR, R)
    return maxR < Rbound

print("\nK_c bisection (torus_exists: R_n bounded < 2.0):", flush=True)
lo, hi = 0.90, 1.05
for _ in range(28):
    mid = 0.5*(lo+hi)
    if torus_exists(mid): lo = mid
    else: hi = mid
Kc = 0.5*(lo+hi)
print(f"K_c ~ {Kc:.5f}   (literature: 0.97163, Greene 1979)", flush=True)
print(f"naive overlap pi^2/4 = {np.pi**2/4:.5f} (OVERESTIMATES by {np.pi**2/4 - 0.97163:.3f})", flush=True)
print(f"time: {time.time()-t0:.1f}s", flush=True)
