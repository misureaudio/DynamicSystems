"""Probe 32: Greene residue criterion, correct orbit selection.
For Fibonacci (p,q): find all period-q orbits (multi-start Newton), keep ELLIPTIC ones
(|tr/2|<1). The golden torus orbit is the one closest to parabolic = smallest residue
R=(1-tr/2)^2/4 (it is the LAST to be destroyed, parabolic at K_c). K_c = K where even this
orbit becomes hyperbolic (no elliptic period-q orbit near the golden action)."""
import numpy as np
import time

def period_q_orbit(K, p, q, I0_guess, th0_guess=0.0, maxit=80, tol=1e-12):
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
        for _ in range(25):
            y2, _ = Fq(x_new); r2 = np.linalg.norm(y2 - x_new - target)
            if r2 < r0: break
            step *= 0.5; x_new = x - step*dx
        x = x_new
    y, M = Fq(x)
    return x, M, conv

def golden_residue(K, p, q, n_starts=24):
    """Return (R_min, tr2_min, n_elliptic) for the golden (most-parabolic) elliptic
    period-q orbit near the golden action. None if no elliptic orbit found."""
    I_center = 2*np.pi*p/q
    best = None
    n_ell = 0
    seen = []
    for i in range(n_starts):
        th0 = 2*np.pi*i/n_starts
        for dI in np.linspace(-0.5, 0.5, 5):
            x, M, conv = period_q_orbit(K, p, q, I_center + dI, th0)
            if not conv: continue
            tr2 = np.trace(M)/2
            if abs(tr2) >= 1.0: continue   # hyperbolic, skip
            n_ell += 1
            # dedup
            if any(np.linalg.norm(x - s[0]) < 1e-5 for s in seen): continue
            seen.append((x, tr2))
            R = (1 - tr2)**2/4
            if best is None or R < best[0]:
                best = (R, tr2)
    if best is None: return None
    return best[0], best[1], n_ell

t0 = time.time()
print("Greene residue (p,q)=(8,13), golden (most-parabolic elliptic) orbit:", flush=True)
Ks = np.linspace(0.90, 1.05, 16)
prev_elliptic = True
Kc_est = None
for K in Ks:
    res = golden_residue(K, 8, 13)
    if res is None:
        print(f"  K={K:.4f}: NO elliptic period-13 orbit near golden action => destroyed", flush=True)
        if prev_elliptic and Kc_est is None:
            Kc_est = K
    else:
        R, tr2, n_ell = res
        print(f"  K={K:.4f}: R={R:.5f} tr/2={tr2:.5f} (elliptic, n_ell={n_ell})", flush=True)
    prev_elliptic = res is not None
# bisect to refine
if Kc_est is not None:
    lo, hi = Kc_est - (Ks[1]-Ks[0]), Kc_est
    for _ in range(30):
        mid = 0.5*(lo+hi)
        if golden_residue(mid, 8, 13) is None: hi = mid
        else: lo = mid
    Kc_est = 0.5*(lo+hi)
print(f"\nK_c (Greene residue, golden orbit becomes hyperbolic) ~ {Kc_est:.5f}  (lit: 0.97163)", flush=True)
print(f"naive overlap pi^2/4 = {np.pi**2/4:.5f} (OVERESTIMATES)", flush=True)
print(f"time: {time.time()-t0:.1f}s", flush=True)
