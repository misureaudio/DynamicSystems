"""Probe 27: robust Greene residue criterion for the Chirikov standard map.
Find period-q orbits (p,q) Fibonacci near I = 2*pi*p/q via Newton; residue R = (1-tr/2)^2/4.
Elliptic iff |tr/2| < 1. The golden-mean torus is destroyed when its approximating orbits
flip elliptic->hyperbolic, at K_c ~ 0.9716."""
import numpy as np
import time

def std_map(th, I, K):
    In = I + K*np.sin(th)
    return th + In, In

def period_q_orbit(K, p, q, I0_guess, th0_guess=0.0, maxit=100, tol=1e-13):
    """Newton for the period-q orbit with rotation number p/q.
    Condition: M^q(th,I) = (th + 2*pi*p, I).  Returns (th0, I0, M^q, converged)."""
    x = np.array([th0_guess, I0_guess], float)
    target = np.array([2*np.pi*p, 0.0])
    def Fq(x):
        th, I = x
        M = np.eye(2)
        for _ in range(q):
            s, c = np.sin(th), np.cos(th)
            M = np.array([[1 + K*c, 1.0], [K*c, 1.0]]) @ M
            In = I + K*s
            th, I = th + In, In
        return np.array([th, I]), M
    conv = False
    for it in range(maxit):
        y, M = Fq(x)
        resid = y - x - target
        if np.linalg.norm(resid) < tol:
            conv = True; break
        try:
            dx = np.linalg.solve(M - np.eye(2), resid)
        except np.linalg.LinAlgError:
            break
        r0 = np.linalg.norm(resid)
        step = 1.0; x_new = x - dx
        for _ in range(30):
            y2, _ = Fq(x_new)
            r2 = np.linalg.norm(y2 - x_new - target)
            if r2 < r0: break
            step *= 0.5; x_new = x - step*dx
        x = x_new
    y, M = Fq(x)
    return x, M, conv

def residue(M):
    tr = np.trace(M)
    return (1 - tr/2)**2/4, tr/2

def find_golden(K, p, q):
    """Multi-start Newton for the period-q orbit near I = 2*pi*p/q."""
    I_center = 2*np.pi*p/q
    best = None
    for th0 in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for dI in [0.0, 0.05, -0.05, 0.1, -0.1]:
            x, M, conv = period_q_orbit(K, p, q, I_center + dI, th0)
            if not conv: continue
            R, tr2 = residue(M)
            # verify genuinely periodic & near the expected action
            if abs(x[1] - I_center) > 1.0: continue
            if best is None or R > best[0]:
                best = (R, tr2, x, conv)
    return best

# test convergence at a few K
t0 = time.time()
for K in [0.5, 0.9, 0.9716, 1.0, 1.1]:
    row = f"K={K}: "
    for (p, q) in [(5, 8), (8, 13), (13, 21), (21, 34)]:
        res = find_golden(K, p, q)
        if res is None:
            row += f"q={q}: --- "
        else:
            R, tr2, x, conv = res
            stable = "elliptic" if abs(tr2) < 1 else "HYPERBOLIC"
            row += f"q={q}: R={R:.4f} tr/2={tr2:.4f} {stable} "
    print(row, flush=True)
print(f"time: {time.time()-t0:.1f}s", flush=True)
