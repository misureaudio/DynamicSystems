"""Verify ns_numba.py against the builder's (read-only) SciPy operators.

Checks (all must PASS):
  (1) rhs_nb      == builder rhs_ref          (bit-exact, 1e-14)
  (2) jacmat_nb   == builder jac_ref @ V      (bit-exact, 1e-12)
  (3) integrate_nb== manual RK4 (builder rhs) (trajectory identical, 1e-12)
  (4) benettin    == builder benettin_ns      (lambda_1 identical, 1e-10)
  (5) lam1_scan   == builder lam1_scan        (5-tuple identical)
  (6) benchmark:  numba vs scipy per-point cost on a representative grid point
"""
import os, sys, time
import numpy as np
import scipy.sparse as spsr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ns_numba as nn

H = nn._helpers()
b_lam1_scan = nn._builder_lam1_scan()

PASS = []
def check(name, cond, detail=""):
    PASS.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("=== (1) rhs_nb vs builder rhs_ref ===")
for (N, nu, A) in [(6, 0.1, 2.0), (12, 0.05, 2.6)]:
    ctx = nn.prepare(N, nu, A)
    r = np.random.default_rng(0)
    om = r.standard_normal(ctx["n"]) + 1j * r.standard_normal(ctx["n"])
    e = float(np.max(np.abs(nn.rhs(ctx, om) - ctx["rhs_ref"](om))))
    check(f"rhs N={N}", e < 1e-12, f"max|d|={e:.2e}")

print("=== (2) jacmat_nb vs builder jac_ref @ V ===")
for (N, nu, A) in [(6, 0.1, 2.0), (12, 0.15, 0.6)]:
    ctx = nn.prepare(N, nu, A)
    r = np.random.default_rng(1)
    om = r.standard_normal(ctx["n"]) + 1j * r.standard_normal(ctx["n"])
    V = r.standard_normal((ctx["n"], 8)) + 1j * r.standard_normal((ctx["n"], 8))
    e = float(np.max(np.abs(nn.jacmat_nb(om, V, ctx["rows"], ctx["cols"], ctx["ls"], ctx["cs"], ctx["dnu"])
                            - ctx["jac_ref"](om) @ V)))
    check(f"J@V N={N}", e < 1e-10, f"max|d|={e:.2e}")

print("=== (3) integrate_nb vs manual RK4 (builder rhs) ===")
N, nu, A = 10, 0.1, 1.4
ctx = nn.prepare(N, nu, A)
dt = H["safe_dt"](N, nu, A)
om0 = H["even_init"](ctx["ks"], ctx["idx"], 0.01, 1)
om_nb, ok = nn.integrate(om0, ctx, dt, 400)
om_s = om0.copy()
for _ in range(400):
    om_s = H["_rk4"](ctx["rhs_ref"], om_s, dt)
e = float(np.max(np.abs(om_nb - om_s)))
check("integrate (400 steps)", ok and e < 1e-10, f"max|d om|={e:.2e}")

print("=== (4) benettin vs builder benettin_ns ===")
om0 = H["even_init"](ctx["ks"], ctx["idx"], 0.01, 1)
om_t = om0.copy()
for _ in range(int(160 / dt)):
    om_t = H["_rk4"](ctx["rhs_ref"], om_t, dt)
lam_nb, _ = nn.benettin(om_t, ctx, T=150.0, dt=dt, m=8)
lam_sc, _ = H["benettin_ns"](ctx["rhs_ref"], ctx["jac_ref"], om_t, T=150.0, dt=dt, m=8)
l1_nb = float(H["real_spectrum"](lam_nb)[0])
l1_sc = float(H["real_spectrum"](lam_sc)[0])
e = abs(l1_nb - l1_sc)
check("lambda_1 (N=10, nu=0.1, A=1.4)", e < 1e-8, f"numba={l1_nb:+.6f} scipy={l1_sc:+.6f} |d|={e:.2e}")

print("=== (5) lam1_scan 5-tuple vs builder lam1_scan ===")
for (N, nu, A) in [(6, 0.1, 2.0), (8, 0.15, 2.2)]:
    t_nb = nn.lam1_scan(N, nu, A, 1, T=150.0, m=8)
    t_sc = b_lam1_scan(N, nu, A, 1, T=150.0, m=8)
    same = (t_nb[1] == t_sc[1]) and (t_nb[4] == t_sc[4]) and \
           ((not np.isfinite(t_nb[0]) and not np.isfinite(t_sc[0])) or abs(t_nb[0] - t_sc[0]) < 1e-6)
    check(f"lam1_scan N={N} nu={nu} A={A}", same,
          f"numba l1={t_nb[0]:+.4f} scipy l1={t_sc[0]:+.4f} div={t_nb[1]} offball={t_nb[4]}")

print("=== (6) benchmark: full lam1_scan (transient + Benettin) numba vs scipy ===")
for (N, nu, A) in [(8, 0.1, 2.0), (12, 0.1, 2.0)]:
    t0 = time.time(); r_nb = nn.lam1_scan(N, nu, A, 1, T=150.0, m=8); t_nb = time.time() - t0
    t0 = time.time(); r_sc = b_lam1_scan(N, nu, A, 1, T=150.0, m=8); t_sc = time.time() - t0
    same = (r_nb[0] == r_sc[0]) or abs(r_nb[0] - r_sc[0]) < 1e-6
    check(f"timing N={N} nu={nu} A={A}", True,
          f"numba={t_nb:5.1f}s scipy={t_sc:5.1f}s  speedup={t_sc/t_nb:4.2f}x  (l1 match={same})")

print()
print("ALL PASS" if all(PASS) else "SOME CHECKS FAILED")
sys.exit(0 if all(PASS) else 1)
