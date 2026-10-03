"""Numba implementation of the 2D-NS Galerkin RHS / Jacobian for NS_N_Dependence.

STANDALONE MODULE -- does NOT modify build_ns_n_dependence_nb.py or any notebook.
It reuses the builder's EXACT sparse structure (read-only, via exec of the helper
block) so every operator is bit-identical to the SciPy fast path by construction.

Why Numba (measured, N=12, n=440):
  * The convective operator has 111,120 nonzeros but ~80% of the SciPy per-eval
    cost is Python overhead (CSR rebuild, cs*om[ls] dense array, dtype dispatch).
  * A flat @njit scatter loop over the triads removes that overhead:
      rhs: 2903 us/eval (scipy) -> 677 us/eval (numba)   [4.3x, bit-exact]
  * The transient (pure RK4) can be fused into ONE Numba call -> no per-step
    Python overhead at all.
  * Benettin keeps the Python loop (complex QR is not in Numba) but each stage
    calls the Numba rhs / J@V kernels, so the per-step cost drops to the QR +
    the two Numba matvecs.

API:
    ctx   = prepare(N, nu, A)          # raw arrays + reference closures
    om, ok = integrate(om, ctx, h, n)  # fused RK4 transient (divergence-aware)
    lam, om = benettin(om, ctx, T, dt, m)
    l1, div, ommax, l2, offball = lam1_scan(N, nu, A, seed, T, m)
"""
import os
import numpy as np
import scipy.sparse as spsr
from numba import njit

_BUILDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "build_ns_n_dependence_nb.py")

_H = None
def _helpers():
    """Exec the builder's helper block (defs only, no top-level side effects)."""
    global _H
    if _H is None:
        src = open(_BUILDER, encoding="utf-8").read()
        # defs from _rk4 through safe_dt (includes R_diss, safe_dt, even_init,
        # benettin_ns, real_spectrum); stops before lam1_scan so nothing runs.
        block = src[src.index("def _rk4"):src.index("def lam1_scan")]
        ns = {"np": np, "spsr": spsr}
        exec(block, ns)
        _H = ns
    return _H

def _builder_lam1_scan():
    """The builder's own lam1_scan (for verification), defs only."""
    src = open(_BUILDER, encoding="utf-8").read()
    # from the "the scan" section (NU_LIST/A_GRID/TRANS/DT/T_SCAN/M_SCAN +
    # guard defs + lam1_scan + scan_one) up to the SELF-CHECK (no side effects)
    block = src[src.index("# ---------------- the scan ----------------"):
                src.index("# ---------------- SELF-CHECK")]
    ns = dict(_helpers())
    exec(block, ns)
    return ns["lam1_scan"]

# ---------------------------------------------------------------------------
# raw structure (bit-identical to the builder's fast path)
# ---------------------------------------------------------------------------
def prepare(N, nu, A):
    """Build the raw triad arrays for the Numba kernels.

    Returns ctx = dict with:
      rows, cols, ls : int64 triad indices of the convective pattern (M structure)
      cs             : float64 coefficients
      fhat           : complex128 forcing
      dnu            : float64 viscous diagonal (nu*k^2)
      n, ks, idx     : mode bookkeeping
      rhs_ref, jac_ref : the builder's SciPy closures (for verification only)
    """
    H = _helpers()
    rhs_ref, jac_ref, _, _, dnu, n, ks, idx = H["make_ns2d_fast"](N, nu, A)
    ks_, n_, k2_, idx_, cs, cols, ls, indptr_M, ls_E, colj_E, cs_E, indptr_E = H["_NSC"][N]
    # explicit row index of each triad, recovered exactly from the CSR indptr
    rows = np.repeat(np.arange(n), np.diff(indptr_M)).astype(np.int64)
    fhat = np.zeros(n, complex)
    for k in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        if k in idx_:
            fhat[idx_[k]] = A / 4.0
    return dict(rows=rows, cols=cols.astype(np.int64), ls=ls.astype(np.int64),
                cs=cs.astype(np.float64), fhat=fhat, dnu=dnu.astype(np.float64),
                n=n, ks=ks, idx=idx, rhs_ref=rhs_ref, jac_ref=jac_ref)

# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
@njit
def rhs_nb(om, rows, cols, ls, cs, fhat, dnu):
    """rhs_full(om) = convective + forcing - nu*k^2*om  (scatter form)."""
    n = om.shape[0]
    out = np.empty(n, dtype=np.complex128)
    for i in range(n):
        out[i] = fhat[i]
    for k in range(rows.shape[0]):
        out[rows[k]] += cs[k] * om[ls[k]] * om[cols[k]]
    for i in range(n):
        out[i] -= dnu[i] * om[i]
    return out

@njit
def _rhs_into(om, out, rows, cols, ls, cs, fhat, dnu):
    """rhs into a preallocated buffer (no allocation; for fused RK4)."""
    n = om.shape[0]
    for i in range(n):
        out[i] = fhat[i]
    for k in range(rows.shape[0]):
        out[rows[k]] += cs[k] * om[ls[k]] * om[cols[k]]
    for i in range(n):
        out[i] -= dnu[i] * om[i]
    return out

@njit
def integrate_nb(om, rows, cols, ls, cs, fhat, dnu, h, nsteps):
    """Fused RK4 transient: nsteps of h. Returns (om, ok); ok=False if non-finite.
    One Numba call -> zero per-step Python overhead."""
    n = om.shape[0]
    k1 = np.empty(n, dtype=np.complex128)
    k2 = np.empty(n, dtype=np.complex128)
    k3 = np.empty(n, dtype=np.complex128)
    k4 = np.empty(n, dtype=np.complex128)
    tmp = np.empty(n, dtype=np.complex128)
    for s in range(nsteps):
        _rhs_into(om, k1, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            tmp[i] = om[i] + 0.5 * h * k1[i]
        _rhs_into(tmp, k2, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            tmp[i] = om[i] + 0.5 * h * k2[i]
        _rhs_into(tmp, k3, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            tmp[i] = om[i] + h * k3[i]
        _rhs_into(tmp, k4, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            om[i] = om[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            if not (np.isfinite(om[i].real) and np.isfinite(om[i].imag)):
                return om, False
    return om, True

@njit
def integrate_energy_nb(om, rows, cols, ls, cs, fhat, dnu, h, nsteps):
    """nsteps of RK4 while accumulating eacc = sum |om|^2 after each step
    (matches the builder's active-mode loop: om=_rk4(...); eacc += |om|^2).
    Returns (om, eacc)."""
    n = om.shape[0]
    k1 = np.empty(n, dtype=np.complex128)
    k2 = np.empty(n, dtype=np.complex128)
    k3 = np.empty(n, dtype=np.complex128)
    k4 = np.empty(n, dtype=np.complex128)
    tmp = np.empty(n, dtype=np.complex128)
    eacc = np.zeros(n)
    for s in range(nsteps):
        _rhs_into(om, k1, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            tmp[i] = om[i] + 0.5 * h * k1[i]
        _rhs_into(tmp, k2, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            tmp[i] = om[i] + 0.5 * h * k2[i]
        _rhs_into(tmp, k3, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            tmp[i] = om[i] + h * k3[i]
        _rhs_into(tmp, k4, rows, cols, ls, cs, fhat, dnu)
        for i in range(n):
            om[i] = om[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            eacc[i] += om[i].real * om[i].real + om[i].imag * om[i].imag
    return om, eacc

@njit
def jacmat_nb(om, V, rows, cols, ls, cs, dnu):
    """J(om) @ V  WITHOUT forming J (V: n x m complex).

    (M@om)[i] = sum_{k: row=i} cs[k]*om[ls[k]]*om[cols[k]]  so
      d/dom[cols[k]] -> cs[k]*om[ls[k]]   (M term)
      d/dom[ls[k]]   -> cs[k]*om[cols[k]] (E term)
    J@V[i,t] = sum_k ( a*V[cols[k],t] + b*V[ls[k],t] ) - dnu[i]*V[i,t]
    """
    n = om.shape[0]
    m = V.shape[1]
    out = np.empty((n, m), dtype=np.complex128)
    for i in range(n):
        for t in range(m):
            out[i, t] = 0.0 + 0.0j
    for k in range(rows.shape[0]):
        a = cs[k] * om[ls[k]]
        b = cs[k] * om[cols[k]]
        r = rows[k]
        for t in range(m):
            out[r, t] += a * V[cols[k], t] + b * V[ls[k], t]
    for i in range(n):
        for t in range(m):
            out[i, t] -= dnu[i] * V[i, t]
    return out

# ---------------------------------------------------------------------------
# Python-level wrappers (mirror the builder's interface exactly)
# ---------------------------------------------------------------------------
def _ctx_args(ctx):
    return ctx["rows"], ctx["cols"], ctx["ls"], ctx["cs"], ctx["fhat"], ctx["dnu"]

def rhs(ctx, om):
    return rhs_nb(om, *_ctx_args(ctx))

def integrate(om0, ctx, h, nsteps):
    """Fused RK4 transient. Returns (om, ok)."""
    return integrate_nb(om0.copy(), ctx["rows"], ctx["cols"], ctx["ls"], ctx["cs"],
                        ctx["fhat"], ctx["dnu"], h, nsteps)

def integrate_energy(om0, ctx, h, nsteps):
    """Fused RK4 while accumulating eacc = sum |om|^2 (active-mode loop).
    Returns (om, eacc)."""
    return integrate_energy_nb(om0.copy(), ctx["rows"], ctx["cols"], ctx["ls"],
                               ctx["cs"], ctx["fhat"], ctx["dnu"], h, nsteps)

def benettin(om0, ctx, T, dt, m):
    """Benettin with Numba rhs / J@V kernels; complex QR via numpy (mirrors the
    builder's benettin_ns step-for-step). Returns (lam, om)."""
    a = _ctx_args(ctx)
    om = om0.copy()
    n = om.shape[0]
    V = np.eye(n, dtype=np.complex128)[:, :m]
    tot = np.zeros(m)
    nsteps = int(round(T / dt))
    for _ in range(nsteps):
        k1 = rhs_nb(om, *a);              V1 = jacmat_nb(om, V, *a[:4], a[5])
        om2 = om + 0.5 * dt * k1
        k2 = rhs_nb(om2, *a);             V2 = jacmat_nb(om2, V + 0.5 * dt * V1, *a[:4], a[5])
        om3 = om + 0.5 * dt * k2
        k3 = rhs_nb(om3, *a);             V3 = jacmat_nb(om3, V + 0.5 * dt * V2, *a[:4], a[5])
        om4 = om + dt * k3
        k4 = rhs_nb(om4, *a);             V4 = jacmat_nb(om4, V + dt * V3, *a[:4], a[5])
        om = om + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        V = V + (dt / 6.0) * (V1 + 2 * V2 + 2 * V3 + V4)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.abs(np.diag(R)))
        sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R)))
        V = Q * sgn
        if not np.isfinite(om).all():
            raise FloatingPointError("diverged")
    return tot / (nsteps * dt), om

def lam1_scan(N, nu, A, seed=1, T=150.0, m=8):
    """Mirror of the builder's lam1_scan, using the Numba kernels.
    Returns (l1, div, ommax, l2, offball) -- same 5-tuple."""
    H = _helpers()
    ctx = prepare(N, nu, A)
    dt = H["safe_dt"](N, nu, A)
    om = H["even_init"](ctx["ks"], ctx["idx"], 0.01, seed)
    om, ok = integrate(om, ctx, dt, int(160.0 / dt))
    if not ok:
        return np.nan, True, float("inf"), float("inf"), True
    try:
        lam_c, om = benettin(om, ctx, T=T, dt=dt, m=m)
    except FloatingPointError:
        return np.nan, True, float("inf"), float("inf"), True
    l1 = H["real_spectrum"](lam_c)[0]
    ommax = float(np.max(np.abs(om)))
    l2 = float(np.sqrt(np.sum(np.abs(om) ** 2)))
    offball = l2 > H["R_diss"](A, nu) * 1.05
    div = (not np.isfinite(l1)) or ommax > 1e8
    return (l1 if np.isfinite(l1) else np.nan), div, ommax, l2, offball

def scan_one_nb(N, seed=1, T_SCAN=150.0, M_SCAN=8, A_GRID=(0.6, 1.0, 1.4, 1.8, 2.0, 2.2, 2.6),
                NU_LIST=(0.05, 0.10, 0.15)):
    """Numba-backed mirror of the builder's scan_one (same output dict + prints,
    same L3/L4 logic).  Uses the Numba kernels for every integration."""
    H = _helpers()
    import time as _time
    t0 = _time.time()
    out = {"grid": {}, "Astar": {}, "divbound": {}, "active": {}, "best": None,
           "best_spread": np.nan, "best_offball": False, "best_seeds": [], "n": None}
    rf0, _, _, _, dnu0, n, ks0, idx0 = H["make_ns2d_fast"](N, 0.1, 2.0)
    out["n"] = n
    print(f"===== N = {N}  (n = {n} modes)  [Numba]")
    print(f"   nu   " + "".join(f"A={a:<6.2f}" for a in A_GRID))
    for nu in NU_LIST:
        row = f"{nu:<6.2f}"
        lams = []
        for Aq in A_GRID:
            l1, div, ommax, l2, offball = lam1_scan(N, nu, Aq, seed, T=T_SCAN, m=M_SCAN)
            out["grid"][(nu, Aq)] = (l1, div, ommax, l2, offball)
            lams.append(l1)
            if div or (not np.isfinite(l1)):
                row += "   DIV "
            elif offball:
                row += f"  {l1:+6.3f}*"
            else:
                row += f"  {l1:+6.3f}"
        print(row)
        bounded = [Aq for Aq in A_GRID if not out["grid"][(nu, Aq)][1]]
        out["divbound"][nu] = max(bounded) if bounded else None
        Astar = None
        for i in range(len(A_GRID) - 1):
            la, lb = lams[i], lams[i + 1]
            if np.isfinite(la) and np.isfinite(lb) and la <= 0 < lb:
                ob_a = out["grid"][(nu, A_GRID[i])][4]
                ob_b = out["grid"][(nu, A_GRID[i + 1])][4]
                if ob_a or ob_b:
                    continue
                lo, hi = A_GRID[i], A_GRID[i + 1]
                for _ in range(6):
                    mid = 0.5 * (lo + hi)
                    lm, dm, _, _, ob_m = lam1_scan(N, nu, mid, seed, T=T_SCAN, m=M_SCAN)
                    if dm or (not np.isfinite(lm)) or ob_m or lm <= 0:
                        lo = mid
                    else:
                        hi = mid
                Astar = 0.5 * (lo + hi)
                break
        out["Astar"][nu] = Astar
        # active modes at A = 1.4 (Numba fused energy loop)
        ctx = prepare(N, nu, 1.4)
        dt2 = H["safe_dt"](N, nu, 1.4)
        om = H["even_init"](ctx["ks"], ctx["idx"], 0.01, seed)
        om, _ = integrate(om, ctx, dt2, int(160 / dt2))          # transient
        om, eacc = integrate_energy(om, ctx, dt2, 800)           # 800 steps, eacc
        e = eacc / 800.0
        c = np.cumsum(np.sort(e)[::-1]) / e.sum()
        out["active"][nu] = int(np.searchsorted(c, 0.99)) + 1
    # best point (nu=0.1, A=2.0), m=16, T=300 -- L4 multi-seed (Numba)
    ctx3 = prepare(N, 0.1, 2.0)
    dt3 = H["safe_dt"](N, 0.1, 2.0)
    seed_lams, seed_l2s, seed_divs = [], [], []
    for s in [1, 2, 3]:
        om = H["even_init"](ctx3["ks"], ctx3["idx"], 0.01, s)
        om, ok = integrate(om, ctx3, dt3, int(300 / dt3))
        if not ok:
            seed_lams.append(np.nan); seed_l2s.append(float("inf")); seed_divs.append(True); continue
        try:
            lam_c, om = benettin(om, ctx3, T=300.0, dt=dt3, m=16)
            l1 = float(H["real_spectrum"](lam_c)[0])
            seed_lams.append(l1)
            seed_l2s.append(float(np.sqrt(np.sum(np.abs(om) ** 2))))
            seed_divs.append(not np.isfinite(l1))
        except FloatingPointError:
            seed_lams.append(np.nan); seed_l2s.append(float("inf")); seed_divs.append(True)
    finite = [l for l in seed_lams if np.isfinite(l)]
    out["best_seeds"] = seed_lams
    out["best"] = float(np.mean(finite)) if finite else np.nan
    out["best_spread"] = (max(finite) - min(finite)) if len(finite) >= 2 else 0.0
    out["best_offball"] = any(l2 > H["R_diss"](2.0, 0.1) * 1.05 for l2 in seed_l2s)
    out["best_diverged"] = any(seed_divs)
    margs = [l1 * T_SCAN for (l1, div, _, _, ob) in out["grid"].values()
             if (not div) and np.isfinite(l1) and (not ob)]
    out["max_margin"] = max(margs) if margs else 0.0
    print(f"   A*(nu): " + ", ".join(f"nu={nu}: " + (f"{out['Astar'][nu]:.3f}" if out["Astar"][nu] is not None else "n/a") for nu in NU_LIST))
    print(f"   div bound: " + ", ".join(f"nu={nu}: " + (f"A<={out['divbound'][nu]:.1f}" if out["divbound"][nu] is not None else "all diverged") for nu in NU_LIST))
    print(f"   active modes (A=1.4): " + ", ".join(f"nu={nu}: {out['active'][nu]}/{n}" for nu in NU_LIST))
    best_txt = (f"lambda_1 = {out['best']:+.4f} (3 seeds: "
                + ", ".join(f"{l:+.3f}" for l in seed_lams)
                + f", spread={out['best_spread']:.3f}"
                + (", OFF-BALL" if out["best_offball"] else "")
                + (", DIVERGED" if out["best_diverged"] else "") + ")")
    print(f"   best (nu=0.1, A=2.0): {best_txt}   max grid margin = {out['max_margin']:.2f}   [took {_time.time()-t0:.0f}s]")
    return out
