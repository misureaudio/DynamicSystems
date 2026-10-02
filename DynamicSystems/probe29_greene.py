"""Probe 29: locate the golden-mean torus (bounded, rho=golden) and find its destruction K_c.
For each K, scan initial action I; classify each orbit as bounded (KAM torus) or drifting
(chaotic). The golden torus is the bounded orbit with rotation number nearest the golden mean.
K_c = the K where the golden torus (rho~0.618, bounded) first becomes drifting."""
import numpy as np
import time

GOLDEN = (np.sqrt(5)-1)/2  # 0.618034

def orbit_classify(K, th0, I0, N=50000):
    th, I = th0, I0
    I_start = I
    for _ in range(N):
        In = I + K*np.sin(th); th, I = th + In, In
    drift = abs(I - I_start)
    rho = (th - th0)/(2*np.pi*N)
    bounded = drift < 1.5
    return rho, drift, bounded

t0 = time.time()
print("K_c search: for each K, find the bounded orbit with rho nearest golden.", flush=True)
Ks = np.linspace(0.94, 1.00, 13)
Is = np.linspace(3.0, 5.5, 51)
results = []
for K in Ks:
    best = None
    for I0 in Is:
        rho, drift, bounded = orbit_classify(K, 0.0, I0, N=30000)
        if bounded and (best is None or abs(rho-GOLDEN) < best[1]):
            best = (I0, abs(rho-GOLDEN), rho, drift)
    if best is None:
        print(f"  K={K:.4f}: NO bounded golden torus  => destroyed", flush=True)
        results.append((K, None))
    else:
        I0, d, rho, drift = best
        print(f"  K={K:.4f}: golden torus at I0={I0:.3f} rho={rho:.5f} drift={drift:.3f}", flush=True)
        results.append((K, I0))
# K_c = first K with no bounded golden torus
for K, I0 in results:
    if I0 is None:
        Kc = K; break
print(f"\nK_c (golden torus destroyed) ~ {Kc:.4f}   (literature: 0.97163, Greene 1979)")
print(f"time: {time.time()-t0:.1f}s", flush=True)
# naive overlap estimate for comparison
print(f"naive resonance-overlap K_c = pi^2/4 = {np.pi**2/4:.4f}  (OVERESTIMATES the true value)")
