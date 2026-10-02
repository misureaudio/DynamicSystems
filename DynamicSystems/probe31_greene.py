"""Probe 31: golden-torus K_c via fine-grid rotation-number + boundedness detector."""
import numpy as np
import time
GOLDEN = (np.sqrt(5)-1)/2   # 0.6180339887

def golden_torus_exists(K, Is, N=40000, drift_tol=1.0):
    """Return (exists, I*, rho*, drift*) for the orbit whose rotation number is closest
    to the golden mean; exists = that orbit is bounded (KAM) not drifting (chaotic)."""
    th = np.zeros_like(Is); I = Is.copy(); th0 = np.zeros_like(Is)
    for _ in range(N):
        I = I + K*np.sin(th); th = th + I
    drift = np.abs(I - Is)
    rho = (th - th0)/(2*np.pi*N)
    j = int(np.argmin(np.abs(rho - GOLDEN)))
    exists = drift[j] < drift_tol
    return exists, Is[j], rho[j], drift[j]

t0 = time.time()
Is = np.linspace(3.5, 5.0, 400)   # fine grid
# bisection on K
lo, hi = 0.90, 1.05
# first confirm bracket
e_lo = golden_torus_exists(lo, Is)[0]
e_hi = golden_torus_exists(hi, Is)[0]
print(f"bracket: K={lo} exists={e_lo}, K={hi} exists={e_hi}", flush=True)
for it in range(40):
    mid = 0.5*(lo+hi)
    e, Istar, rho, drift = golden_torus_exists(mid, Is)
    if e:
        lo = mid
    else:
        hi = mid
Kc = 0.5*(lo+hi)
print(f"\nK_c (golden torus destroyed) ~ {Kc:.5f}   (literature: 0.97163, Greene 1979)", flush=True)
print(f"naive resonance-overlap K_c = pi^2/4 = {np.pi**2/4:.5f}  (OVERESTIMATES)", flush=True)
print(f"time: {time.time()-t0:.1f}s", flush=True)
# show the transition
print("\ntransition detail:", flush=True)
for K in [0.968, 0.970, 0.971, 0.9716, 0.972, 0.974, 0.976]:
    e, Istar, rho, drift = golden_torus_exists(K, Is)
    print(f"  K={K:.4f}: golden orbit I*={Istar:.4f} rho={rho:.6f} drift={drift:.3f} -> {'KAM (bounded)' if e else 'CHAOTIC (drifting)'}", flush=True)
