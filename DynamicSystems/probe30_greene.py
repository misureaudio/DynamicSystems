"""Probe 30: vectorized golden-torus K_c search."""
import numpy as np
import time
GOLDEN = (np.sqrt(5)-1)/2
def K_c_search():
    Ks = np.linspace(0.94, 1.00, 13)
    Is = np.linspace(3.0, 5.5, 51)
    N = 30000
    results = []
    for K in Ks:
        th = np.zeros_like(Is); I = Is.copy(); th0 = np.zeros_like(Is)
        for _ in range(N):
            I = I + K*np.sin(th); th = th + I
        drift = np.abs(I - Is)
        rho = (th - th0)/(2*np.pi*N)
        bounded = drift < 1.5
        idx = np.where(bounded)[0]
        if len(idx) == 0:
            results.append((K, None)); print(f"  K={K:.4f}: NO bounded golden torus => destroyed", flush=True)
            continue
        j = idx[np.argmin(np.abs(rho[idx]-GOLDEN))]
        results.append((K, Is[j]))
        print(f"  K={K:.4f}: golden torus I0={Is[j]:.3f} rho={rho[j]:.5f} drift={drift[j]:.3f}", flush=True)
    for K, I0 in results:
        if I0 is None:
            return K
    return None
t0 = time.time()
Kc = K_c_search()
print(f"\nK_c (golden torus destroyed) ~ {Kc:.4f}   (literature: 0.97163)")
print(f"naive overlap K_c = pi^2/4 = {np.pi**2/4:.4f} (OVERESTIMATES)")
print(f"time: {time.time()-t0:.1f}s", flush=True)
