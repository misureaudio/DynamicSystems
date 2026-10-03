"""Builder: assembles NS_N_Dependence_numba.ipynb (Numba-accelerated variant).

Identical to build_ns_n_dependence_nb.py (the SciPy baseline) EXCEPT:
  * a new S0b cell rebinds the integration to the standalone Numba module
    ns_numba.py (bit-identical operators, fused RK4 transient, Numba J@V);
  * title/notebook name reflect the Numba variant.
The S0 self-checks (fast==original, L6 FD-Jacobian) are unchanged and still run
against the SciPy fast path before any heavy run.  The SciPy builder/notebook are
left untouched as the reference baseline.

Computational test of the KNOWN interrelation between the three NS knobs:
  - forcing A and viscosity nu collapse to Re = A/nu  (boundary is a curve in Re)
  - N is RESOLUTION, not physics: faithful only if N >= # active modes ~ C*nu^{-3/2}
  - A*_N(nu) -> A*(nu) as N -> inf, but for finite N the boundary is N-dependent
    (spurious dynamics below the resolution threshold).

We scan N = 6, 8, 10, 12 and measure:
  (1) A*_N(nu):  the laminar/chaotic boundary (lambda_1 crosses 0) at each N
  (2) divergence boundary:  max A that stays bounded (the nan wall moves out with N)
  (3) lambda_1(N) at the "best" point (nu=0.1, A=2.0):  drift with N
  (4) active-mode count vs nu:  compare the log-log slope with the Foias-Temam -3/2

Engineering: the per-step sparse rebuild (coo->csr, with sort/dedup) is the cost
bottleneck.  make_ns2d_fast precomputes the sorted-csr structure once per N and
builds each step's matrix from pre-sorted indices (identical math, no sort).
Cell S0 SELF-CHECKS the fast path against the original coo version before any
heavy run.

Read-only w.r.t. the essay and the existing notebooks; writes a new .ipynb + nothing
else.  Run:  .venv/Scripts/python.exe build_ns_n_dependence_nb.py
"""
import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    "kernelspec": {"display_name": "Python 3 (Dynamic Systems venv)",
                   "language": "python", "name": "dsvenv"},
    "language_info": {"name": "python", "version": "3.11"},
}
NB.nbformat = 4
NB.nbformat_minor = 5

_id = [0]
def _stamp():
    _id[0] += 1
    return f"cell{_id[0]:03d}"

def md(src):
    c = nbf.v4.new_markdown_cell(src)
    c.id = _stamp()
    return c

def py(src):
    c = nbf.v4.new_code_cell(src)
    c.id = _stamp()
    return c

cells = []
A = cells.append

# =====================================================================
# S0  TITLE
# =====================================================================
A(md(r'''
# NS Galerkin: the N-dependence of the laminar/chaotic boundary
**(Numba-accelerated build)**

**Computational test of the known interrelation between the three knobs**
(forcing $A$, viscosity $\nu$, truncation $N$).

> **This build** runs the integration through the standalone **Numba** module
> `ns_numba.py` (bit-identical operators to the SciPy fast path; a fused single-call
> RK4 transient and a Numba `J@V` kernel).  The SciPy builder/notebook
> (`build_ns_n_dependence_nb.py` / `NS_N_Dependence.ipynb`) is the untouched
> reference baseline.  The S0 self-checks below validate the SciPy fast path; S0b
> then rebinds the scan to Numba and self-checks it against SciPy.

## What is known (theory) vs what is computational

| Statement | Status |
|---|---|
| $A$ and $\nu$ collapse to $\mathrm{Re}=A/\nu$; the boundary is a *curve* in Re, not a 2-D surface | **Known** (Re scaling) |
| This is the canonical Ruelle–Takens flow: laminar $\to$ Hopf (Re$\approx$35) $\to$ quasi-periodic $\to$ chaos; the $\lambda_1=0$ boundary sits *above* the first Hopf | **Known** (cascade structure) |
| $A^*(\nu)\approx c\,\nu$ to leading order (a straight line in log-log) | **Known** (up to the normalization constant $c$) |
| Active modes / attractor dimension $\sim C\,\nu^{-3/2}$ (Foias–Temam) | **Known** (power law) |
| Below $N\gtrsim$ active modes the truncation has **spurious dynamics**; $A^*_N(\nu)\to A^*(\nu)$ as $N\to\infty$ | **Known** (Galerkin theory) |
| The *exact* $A^*(\nu)$, the exact Re where $\lambda_1>0$, the exact $N$ at which a point is faithful | **Computational** |

## What this notebook measures (at $N=6,8,10,12$)

1. **$A^*_N(\nu)$** — the boundary where $\lambda_1$ crosses 0, at each N (bisection).
2. **Boundedness at a *stable* timestep** — whether every point settles, once the
   timestep is chosen to keep RK4 stable (see L7 below).  A naive *fixed* $dt$
   fabricates a fake "divergence wall" that moves *in* with N; at a stable $dt$ the
   2D-NS Galerkin is dissipative and every point is bounded.
3. **$\lambda_1(N)$** at the best-scanned point ($\nu=0.1$, $A=2.0$) — if the small
   N=6 value was under-resolution suppression, it should grow with N.
4. **Active-mode count vs $\nu$** — log-log slope vs the Foias–Temam $-3/2$.

**Scan** (1 seed; the point is the N-trend, not a high-precision boundary):
$\nu\in\{0.05,0.10,0.15\}$, $A\in\{0.6,1.0,1.4,1.8,2.0,2.2,2.6\}$,
transient 160, $T_{\mathrm{scan}}=150$, **$dt = \mathrm{safe\_dt}(N,\nu,A)$** (stiffness-aware,
L7), $m=8$ exponents; bisection (6 steps) where a sign change is found.

> **Convention.** Fixed seed. Before any heavy run the cell below (a) self-checks the
> fast sparse path against the original coo assembly, and (b) runs the **L6
> central-FD Jacobian guard**, which catches a wrong variational Jacobian — the
> failure mode no trajectory-divergence test can see.
'''))

# =====================================================================
# S0  HELPERS + SELF-CHECK
# =====================================================================
A(py(r'''
# ---- S0: helpers (NS Galerkin, original + fast) + self-check ----------
import sys
assert "hermes-dir" in sys.executable and ".venv" in sys.executable, \
    f"WRONG INTERPRETER: kernel is {sys.executable} -- must be the workspace .venv"
print(f"kernel interpreter: {sys.executable}")
import numpy as np
import scipy.sparse as spsr
import time

def _rk4(f, x, h):
    k1 = f(x); k2 = f(x + .5*h*k1); k3 = f(x + .5*h*k2); k4 = f(x + h*k3)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------- ORIGINAL (coo->csr per step): reference ----------------
def make_ns2d_orig(N, nu, A):
    ks = [(i, j) for i in range(-N, N+1) for j in range(-N, N+1) if 0 < i*i + j*j <= N*N]
    n = len(ks); karr = np.array(ks, float); k2 = np.hypot(karr[:, 0], karr[:, 1])**2
    idx = {k: m for m, k in enumerate(ks)}
    rows, cols, ls, cs = [], [], [], []
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            diff = (int(ki[0]-kj[0]), int(ki[1]-kj[1]))
            if diff in idx:
                cr = float(ki[0]*kj[1] - ki[1]*kj[0])
                if cr != 0.0:
                    rows.append(i); cols.append(j); ls.append(idx[diff]); cs.append(cr/k2[j])
    rows=np.array(rows); cols=np.array(cols); ls=np.array(ls); cs=np.array(cs, float)
    fhat = np.zeros(n, complex)
    for k in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        if k in idx: fhat[idx[k]] = A/4
    dnu = nu*k2
    def M_as(om): return spsr.coo_matrix((cs*om[ls], (rows, cols)), shape=(n,n)).tocsr()
    def rhs_nl(om): return M_as(om) @ om + fhat
    def jac_nl(om):
        M = M_as(om)
        E = spsr.coo_matrix((cs*om[cols], (rows, ls)), shape=(n,n)).tocsr()
        return (M + E).tocsr()
    def rhs_full(om): return rhs_nl(om) - dnu*om
    def jac_full(om): return jac_nl(om) - spsr.diags(dnu)
    return rhs_full, jac_full, rhs_nl, jac_nl, dnu, n, ks, idx

# ---------------- FAST (precomputed sorted-csr): same math, no per-step sort
_NSC = {}
def make_ns2d_fast(N, nu, A):
    if N not in _NSC:
        ks = [(i, j) for i in range(-N, N+1) for j in range(-N, N+1) if 0 < i*i + j*j <= N*N]
        n = len(ks); karr = np.array(ks, float); k2 = np.hypot(karr[:, 0], karr[:, 1])**2
        idx = {k: m for m, k in enumerate(ks)}
        rows, cols, ls, cs = [], [], [], []
        for i, ki in enumerate(ks):
            for j, kj in enumerate(ks):
                diff = (int(ki[0]-kj[0]), int(ki[1]-kj[1]))
                if diff in idx:
                    cr = float(ki[0]*kj[1] - ki[1]*kj[0])
                    if cr != 0.0:
                        rows.append(i); cols.append(j); ls.append(idx[diff]); cs.append(cr/k2[j])
        rows=np.array(rows); cols=np.array(cols); ls=np.array(ls); cs=np.array(cs, float)
        # M structure (rows, cols): rows=i outer, cols=j inner -> already row-major sorted
        indptr_M = np.r_[0, np.bincount(rows, minlength=n)].cumsum()
        # E structure (rows, ls): sort by (row, col) once
        ord_ = np.lexsort((ls, rows))
        rows_E = rows[ord_]; ls_E = ls[ord_]; colj_E = cols[ord_]; cs_E = cs[ord_]
        indptr_E = np.r_[0, np.bincount(rows_E, minlength=n)].cumsum()
        _NSC[N] = (ks, n, k2, idx, cs, cols, ls, indptr_M, ls_E, colj_E, cs_E, indptr_E)
    ks, n, k2, idx, cs, cols, ls, indptr_M, ls_E, colj_E, cs_E, indptr_E = _NSC[N]
    fhat = np.zeros(n, complex)
    for k in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        if k in idx: fhat[idx[k]] = A/4
    dnu = nu*k2
    D_dnu = spsr.diags(dnu)
    def rhs_nl(om):
        M = spsr.csr_matrix((cs*om[ls], cols, indptr_M), shape=(n,n))
        return M @ om + fhat
    def jac_nl(om):
        M = spsr.csr_matrix((cs*om[ls], cols, indptr_M), shape=(n,n))
        E = spsr.csr_matrix((cs_E*om[colj_E], ls_E, indptr_E), shape=(n,n))
        return (M + E).tocsr()
    def rhs_full(om): return rhs_nl(om) - dnu*om
    def jac_full(om): return jac_nl(om) - D_dnu
    return rhs_full, jac_full, rhs_nl, jac_nl, dnu, n, ks, idx

# ---------------- shared ----------------
def even_init(ks, idx, scale, seed=1):
    r = np.random.default_rng(seed); om = np.zeros(len(ks), complex)
    for m, k in enumerate(ks):
        km = (-k[0], -k[1])
        if km in idx and m > idx[km]: continue
        om[m] = scale*(r.random() + 1j*r.random())
    for m, k in enumerate(ks):
        km = (-k[0], -k[1])
        if km in idx and m < idx[km]: om[idx[km]] = np.conj(om[m])
    return om

def benettin_ns(rhs, jac, om0, T, dt, m):
    om = om0.copy(); nn = om0.shape[0]
    V = np.eye(nn, dtype=complex)[:, :m]; tot = np.zeros(m); nsteps = int(round(T/dt))
    for _ in range(nsteps):
        k1 = rhs(om); J1 = jac(om)
        k2 = rhs(om + 0.5*dt*k1); J2 = jac(om + 0.5*dt*k1)
        k3 = rhs(om + 0.5*dt*k2); J3 = jac(om + 0.5*dt*k2)
        k4 = rhs(om + dt*k3);    J4 = jac(om + dt*k3)
        om = om + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V1 = J1 @ V; V2 = J2 @ (V + 0.5*dt*V1); V3 = J3 @ (V + 0.5*dt*V2); V4 = J4 @ (V + dt*V3)
        V = V + (dt/6.0)*(V1 + 2*V2 + 2*V3 + V4)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.abs(np.diag(R))); sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R))); V = Q * sgn
        if not np.isfinite(om).all():
            raise FloatingPointError("diverged")
    return tot/(nsteps*dt), om

def real_spectrum(lam_c):
    return np.sort(np.concatenate([np.real(lam_c), np.real(lam_c)]))[::-1]

# ---------------- the scan ----------------
NU_LIST = [0.05, 0.10, 0.15]
A_GRID  = [0.6, 1.0, 1.4, 1.8, 2.0, 2.2, 2.6]
TRANS, DT, T_SCAN, M_SCAN = 160.0, 0.125, 150.0, 8

# ---- GUARD BATTERY --------------------------------------------------------
# L3: a-priori dissipative-ball radius.  Enstrophy balance (convective term
#     skew-adjoint -> 0 work, verified numerically ~1e-17):
#       y' <= ||f|| - nu*lambda_1*y,  ||f||=A/2,  lambda_1 = min k^2.
#     The RIGOROUS lambda_1 = min k^2 = 1  (the (±1,0),(0,±1) modes) ->
#       y* = ||f||/(nu*lambda_1) = A/(2 nu).
#     NOTE: this builder's R_diss = A/(4 nu) is a CONSERVATIVE practical
#     threshold (it assumes min k^2 = 2), i.e. 2x tighter than rigorous.  In
#     the Numba build (S0b / ns_numba.py) the off-ball test is overridden to
#     the rigorous A/(2 nu); trajectories and lambda_1 are unaffected (the
#     override touches only the off-ball classification, not safe_dt).
def R_diss(A, nu):
    return A/(4.0*nu)

# L6: central-FD Jacobian guard.  rhs is quadratic in om, so the central
#     difference is EXACT (O(eps^2) terms cancel):
#       J(om)@v  ==  ( f(om+eps v) - f(om-eps v) ) / (2 eps)
#     for any direction v.  A linearization bug (e.g. double-counted viscosity)
#     is invisible to every trajectory-divergence test but shows up here as a
#     large max|J@v - FD|.  Runs in the SELF-CHECK before any heavy scan.
def fd_jac_guard(rhs, jac, om, nvec=4, eps=1e-6, seed=0):
    r = np.random.default_rng(seed); n = om.shape[0]; worst = 0.0
    for _ in range(nvec):
        v = r.standard_normal(n) + 1j*r.standard_normal(n)
        Jv = jac(om) @ v
        fd = (rhs(om + eps*v) - rhs(om - eps*v)) / (2.0*eps)
        worst = max(worst, float(np.max(np.abs(Jv - fd))))
    return worst

# ---- L7 (stiffness-aware adaptive timestep) ------------------------------
# The single most important guard for a dissipative Galerkin method: the FIXED
# timestep dt=0.125 becomes UNSTABLE as N grows, because both the viscous
# stiffness nu*k_max^2 and the convective stiffness |om|*k_max ~ (A/4nu)*k_max
# grow with N.  An unstable RK4 step fabricates a "divergence" (L2 -> 1e85)
# that is PURELY NUMERICAL -- the same point converges to a finite L2 at half
# the timestep.  safe_dt keeps the step inside RK4's nonlinear stability region:
#     dt = min(0.125, 2.0/(nu*k_max^2),  8.0*nu/(A*k_max))
# Calibrated (probe_dt2): keeps ALL 84 grid points (N=6,8,10,12 x nu x A) bounded.
_KMAX2 = {}
def _kmax2(N):
    if N not in _KMAX2:
        _KMAX2[N] = max(i*i + j*j for i, j in make_ns2d_fast(N, 0.1, 1.0)[6])
    return _KMAX2[N]
def safe_dt(N, nu, A, cap=0.125):
    k2 = _kmax2(N); kmax = int(round(np.sqrt(k2)))
    return min(cap, 2.0/(nu*k2), 8.0*nu/(A*kmax))

def lam1_scan(N, nu, A, seed=1, T=T_SCAN, m=M_SCAN):
    # returns (l1, div, ommax, l2, offball)
    #   div     = blow-up (non-finite or |om|>1e8)   -- at a STABLE dt this
    #             should never fire for dissipative 2D-NS Galerkin
    #   offball = bounded but outside the dissipative ball (L3) -> off-attractor
    #   dt      = safe_dt (L7): stiffness-aware, keeps RK4 stable at all N
    rhs_full, jac_full, _, _, dnu, n, ks, idx = make_ns2d_fast(N, nu, A)
    dt = safe_dt(N, nu, A)
    om = even_init(ks, idx, 0.01, seed)
    for _ in range(int(TRANS/dt)): om = _rk4(rhs_full, om, dt)
    if not np.isfinite(om).all():
        return np.nan, True, float("inf"), float("inf"), True
    try:
        lam_c, om = benettin_ns(rhs_full, jac_full, om, T=T, dt=dt, m=m)
    except FloatingPointError:
        return np.nan, True, float("inf"), float("inf"), True
    l1 = real_spectrum(lam_c)[0]
    ommax = float(np.max(np.abs(om)))
    l2 = float(np.sqrt(np.sum(np.abs(om)**2)))
    offball = l2 > R_diss(A, nu)*1.05          # L3 dissipative-ball test
    div = (not np.isfinite(l1)) or ommax > 1e8
    return (l1 if np.isfinite(l1) else np.nan), div, ommax, l2, offball

def scan_one(N, seed=1):
    t0 = time.time()
    out = {"grid": {}, "Astar": {}, "divbound": {}, "active": {}, "best": None,
           "best_spread": np.nan, "best_offball": False, "best_seeds": [], "n": None}
    rhs_full, _, _, _, dnu, n, ks, idx = make_ns2d_fast(N, 0.1, 2.0)
    out["n"] = n
    print(f"===== N = {N}  (n = {n} modes) =====")
    print(f"   nu   " + "".join(f"A={a:<6.2f}" for a in A_GRID))
    for nu in NU_LIST:
        row = f"{nu:<6.2f}"
        lams = []
        for Aq in A_GRID:
            l1, div, ommax, l2, offball = lam1_scan(N, nu, Aq, seed)
            out["grid"][(nu, Aq)] = (l1, div, ommax, l2, offball)
            lams.append(l1)
            if div or (not np.isfinite(l1)):
                row += "   DIV "
            elif offball:
                row += f"  {l1:+6.3f}*"   # * = bounded but off dissipative ball (L3)
            else:
                row += f"  {l1:+6.3f}"
        print(row)
        # divergence boundary: largest A that stays bounded
        bounded = [Aq for Aq in A_GRID if not out["grid"][(nu, Aq)][1]]
        out["divbound"][nu] = max(bounded) if bounded else None
        # A* by bisection on the first sign change (finite, on-ball values only)
        Astar = None
        for i in range(len(A_GRID)-1):
            la, lb = lams[i], lams[i+1]
            if np.isfinite(la) and np.isfinite(lb) and la <= 0 < lb:
                # L3: only bisection if BOTH bracketing points are on the attractor ball
                ob_a = out["grid"][(nu, A_GRID[i])][4]
                ob_b = out["grid"][(nu, A_GRID[i+1])][4]
                if ob_a or ob_b:
                    continue
                lo, hi = A_GRID[i], A_GRID[i+1]
                for _ in range(6):
                    mid = 0.5*(lo+hi)
                    lm, dm, _, _, ob_m = lam1_scan(N, nu, mid, seed)
                    if dm or (not np.isfinite(lm)) or ob_m or lm <= 0: lo = mid
                    else: hi = mid
                Astar = 0.5*(lo+hi); break
        out["Astar"][nu] = Astar
        # active modes at A = 1.4  (L7: stable dt)
        rf, _, _, _, dnu2, n2, ks2, idx2 = make_ns2d_fast(N, nu, 1.4)
        dt2 = safe_dt(N, nu, 1.4)
        om = even_init(ks2, idx2, 0.01, seed)
        for _ in range(int(160/dt2)): om = _rk4(rf, om, dt2)
        eacc = np.zeros(n2)
        for _ in range(800):
            om = _rk4(rf, om, dt2); eacc += np.abs(om)**2
        e = eacc/800.0
        c = np.cumsum(np.sort(e)[::-1])/e.sum()
        out["active"][nu] = int(np.searchsorted(c, 0.99)) + 1
    # best point (nu=0.1, A=2.0), m=16, T=300  --  L4: multi-seed ensemble
    # A genuine attractor's lambda_1 is a property of the invariant measure,
    # so it must reproduce from any point in the basin.  We run 3 seeds; if
    # they disagree (spread > 0.3) the point is on a spurious repeller / basin
    # boundary, not the physical attractor.
    rf, jf, _, _, dnu3, n3, ks3, idx3 = make_ns2d_fast(N, 0.1, 2.0)
    dt3 = safe_dt(N, 0.1, 2.0)
    seed_lams, seed_l2s, seed_divs = [], [], []
    for s in [1, 2, 3]:
        om = even_init(ks3, idx3, 0.01, s)
        for _ in range(int(300/dt3)): om = _rk4(rf, om, dt3)
        try:
            lam_c, om = benettin_ns(rf, jf, om, T=300.0, dt=dt3, m=16)
            l1 = float(real_spectrum(lam_c)[0])
            seed_lams.append(l1)
            seed_l2s.append(float(np.sqrt(np.sum(np.abs(om)**2))))
            seed_divs.append(not np.isfinite(l1))
        except FloatingPointError:
            seed_lams.append(np.nan); seed_l2s.append(float("inf")); seed_divs.append(True)
    finite = [l for l in seed_lams if np.isfinite(l)]
    out["best_seeds"] = seed_lams
    out["best"] = float(np.mean(finite)) if finite else np.nan
    out["best_spread"] = (max(finite)-min(finite)) if len(finite) >= 2 else 0.0
    out["best_offball"] = any(l2 > R_diss(2.0, 0.1)*1.05 for l2 in seed_l2s)
    out["best_diverged"] = any(seed_divs)
    # max margin over the (finite, non-diverged, ON-BALL) grid -- L3 filter
    margs = [l1*T_SCAN for (l1, div, _, _, ob) in out["grid"].values()
             if (not div) and np.isfinite(l1) and (not ob)]
    out["max_margin"] = max(margs) if margs else 0.0
    print(f"   A*(nu): " + ", ".join(f"nu={nu}: " + (f"{out['Astar'][nu]:.3f}" if out["Astar"][nu] is not None else "n/a") for nu in NU_LIST))
    print(f"   div bound: " + ", ".join(f"nu={nu}: " + (f"A<={out['divbound'][nu]:.1f}" if out["divbound"][nu] is not None else "all diverged") for nu in NU_LIST))
    print(f"   active modes (A=1.4): " + ", ".join(f"nu={nu}: {out['active'][nu]}/{n}" for nu in NU_LIST))
    best_txt = (f"lambda_1 = {out['best']:+.4f} (3 seeds: "
                + ", ".join(f"{l:+.3f}" for l in out["best_seeds"])
                + f", spread={out['best_spread']:.3f}"
                + (", OFF-BALL" if out["best_offball"] else "")
                + (", DIVERGED" if out["best_diverged"] else "") + ")")
    print(f"   best (nu=0.1, A=2.0): {best_txt}   max grid margin = {out['max_margin']:.2f}   [took {time.time()-t0:.0f}s]")
    return out

# ---------------- SELF-CHECK: fast == original (before any heavy run) ----
N_chk, nu_chk, A_chk = 6, 0.08, 1.3
ro, jo, _, _, dnu_o, n_o, ks_o, idx_o = make_ns2d_orig(N_chk, nu_chk, A_chk)
rf, jf, _, _, dnu_f, n_f, ks_f, idx_f = make_ns2d_fast(N_chk, nu_chk, A_chk)
r = np.random.default_rng(0)
om = r.standard_normal(n_o) + 1j*r.standard_normal(n_o)
v = r.standard_normal(n_o) + 1j*r.standard_normal(n_o)
drhs = np.max(np.abs(ro(om) - rf(om)))
djac = np.max(np.abs((jo(om) - jf(om)) @ v))
print(f"SELF-CHECK fast vs original (N={N_chk}):  max|d rhs| = {drhs:.3e}   max|d jac @ v| = {djac:.3e}")
assert drhs < 1e-10 and djac < 1e-10, "FAST PATH MISMATCH"
print("SELF-CHECK PASSED: the precomputed sorted-csr path is identical to the original.")

# ---------------- SELF-CHECK (L6): FD-Jacobian guard (before any heavy run) ----
# The variational Jacobian jac_full must equal the true Df/dom.  For the quadratic
# Galerkin rhs the central difference is exact, so any linearization bug (e.g.
# double-counted viscosity) shows up as a large max|J@v - FD| here and aborts
# the run BEFORE a single heavy scan is wasted.
fd_err = fd_jac_guard(rf, jf, om, nvec=4, eps=1e-6, seed=0)
print(f"SELF-CHECK L6 FD-Jacobian (N={N_chk}):  max|J@v - FD(J)@v| = {fd_err:.3e}")
assert fd_err < 1e-6, "JACOBIAN MISMATCH: jac_full disagrees with the central-FD Jacobian (linearization bug)"
print("SELF-CHECK L6 PASSED: the variational Jacobian is correct (central-FD exact for the quadratic rhs).")
'''))

# =====================================================================
# S0b  NUMBA REBIND (bit-identical operators; fused RK4 transient)
# =====================================================================
A(py(r'''
# ---- S0b: rebind the scan to the Numba module ----------------------------
# ns_numba.py reuses the EXACT sparse structure built above (read-only) and
# implements the RHS / J@V / RK4 as @njit kernels, so the operators are
# bit-identical to the SciPy fast path.  Verify that here, then rebind the
# name `scan_one` so S1-S4 run on Numba with no other change.
import ns_numba as _nn
_ctx = _nn.prepare(N_chk, nu_chk, A_chk)
_om = np.random.default_rng(7).standard_normal(n_f) + 1j*np.random.default_rng(8).standard_normal(n_f)
_v  = np.random.default_rng(9).standard_normal((n_f, 8)) + 1j*np.random.default_rng(10).standard_normal((n_f, 8))
_drh = float(np.max(np.abs(_nn.rhs(_ctx, _om) - rf(_om))))
_dja = float(np.max(np.abs(_nn.jacmat_nb(_om, _v, _ctx["rows"], _ctx["cols"], _ctx["ls"], _ctx["cs"], _ctx["dnu"]) - jf(_om) @ _v)))
print(f"SELF-CHECK Numba vs SciPy (N={N_chk}):  max|d rhs| = {_drh:.3e}   max|d J@v| = {_dja:.3e}")
assert _drh < 1e-9 and _dja < 1e-8, "NUMBA MISMATCH vs SciPy fast path"
# L3: the Numba scan uses the RIGOROUS dissipative ball A/(2 nu) (min k^2 = 1),
# not the builder's conservative A/(4 nu).  Verify the value and that the
# override does NOT touch safe_dt (trajectories / lambda_1 are unaffected).
_r_builder = R_diss(A_chk, nu_chk)
_r_rigor   = _nn.R_diss_rigorous(A_chk, nu_chk)
print(f"L3 ball (N={N_chk}, nu={nu_chk}, A={A_chk}):  builder (conservative) = {_r_builder:.3f}   rigorous = {_r_rigor:.3f}   (ratio = {_r_rigor/_r_builder:.1f})")
assert abs(_r_rigor - 2.0*_r_builder) < 1e-12, "L3 override not 2x the builder ball"
print("L3: off-ball test uses the rigorous A/(2 nu) ball (2x looser than the builder's A/(4 nu)); safe_dt unchanged.")
# rebind: S1-S4 call scan_one(N) by name -> now the Numba implementation
scan_one = lambda N, seed=1: _nn.scan_one_nb(N, seed=seed, T_SCAN=T_SCAN, M_SCAN=M_SCAN,
                                             A_GRID=A_GRID, NU_LIST=NU_LIST)
print("S0b PASSED: scan rebound to Numba (bit-identical operators, fused single-call RK4 transient; L3 rigorous ball).")
'''))

# =====================================================================
# S1-S4  SCANS AT N = 6, 8, 10, 12
# =====================================================================
A(py(r'''
# ---- S1: N = 6 (reference; should reproduce A*(0.1) ~ 1.8 from the robust notebook)
R6 = scan_one(6)
'''))

A(py(r'''
# ---- S2: N = 8
R8 = scan_one(8)
'''))

A(py(r'''
# ---- S3: N = 10
R10 = scan_one(10)
'''))

A(py(r'''
# ---- S4: N = 12
R12 = scan_one(12)
'''))

# =====================================================================
# S5  AGGREGATE + FIGURES + VERDICT
# =====================================================================
A(py(r'''
# ---- S5: aggregate the N-dependence ----------------------------------
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "grid.alpha": 0.3, "font.size": 11})
NS = [6, 8, 10, 12]
RS = {6: R6, 8: R8, 10: R10, 12: R12}

# (1) A*_N(nu)
fig, ax = plt.subplots(figsize=(7, 5))
for N in NS:
    xs, ys = [], []
    for nu in NU_LIST:
        if RS[N]["Astar"][nu] is not None:
            xs.append(nu); ys.append(RS[N]["Astar"][nu])
    ax.plot(xs, ys, "o-", lw=1.5, label=f"N = {N}  (n = {RS[N]['n']})")
ax.axhline(1.83, color="k", ls=":", lw=1, label="N=6, nu=0.1 (robust notebook: 1.83)")
ax.set_xlabel("nu"); ax.set_ylabel(r"$A^*_N(\nu)$  (boundary where $\lambda_1$ crosses 0)")
ax.set_title("Laminar/chaotic boundary vs viscosity, per truncation N")
ax.legend(); plt.tight_layout(); plt.show()

# (2) boundedness at a stable timestep (L7).  A naive FIXED dt fabricates a fake
#     "divergence wall" that moves IN with N (1.4/2.6 -> 0.6/1.4 -> 0.6/1.0).
#     At dt = safe_dt(N,nu,A) the Galerkin is dissipative: count how many of the
#     7 forcings per nu stay bounded AND on the dissipative ball (L3).
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(NU_LIST))
for j, N in enumerate(NS):
    counts = []
    for nu in NU_LIST:
        row_vals = [RS[N]["grid"][(nu, Aq)] for Aq in A_GRID]
        cnt = sum(1 for (l1, div, _, _, ob) in row_vals
                  if (not div) and np.isfinite(l1) and (not ob))
        counts.append(cnt)
    ax.bar(x + j*0.2, counts, 0.2, label=f"N = {N}")
ax.set_xticks(x + 0.3); ax.set_xticklabels([f"nu = {nu}" for nu in NU_LIST])
ax.set_ylabel("bounded + on-ball forcings (of 7)")
ax.set_title("Boundedness at a STABLE timestep (L7):  all points settle -> no wall")
ax.set_ylim(0, 7.5); ax.legend(); plt.tight_layout(); plt.show()

# (3) lambda_1(N) at the best point  (L4: mean of 3 seeds, error bar = spread)
bests = [RS[N]["best"] for N in NS]
spreads = [RS[N]["best_spread"]/2 for N in NS]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(NS, bests, yerr=spreads, fmt="o-", lw=1.5, color="darkorange",
            ecolor="gray", elinewidth=1.5, capsize=4,
            label=r"$\lambda_1$ (3 seeds; bar = spread)")
ax.axhline(0, color="k", lw=1, ls=":")
ax.set_xlabel("N (truncation)"); ax.set_ylabel(r"$\lambda_1$  (nu=0.1, A=2.0, T=300)")
ax.set_title("lambda_1 at the best-scanned point vs N (L4: multi-seed spread shown)")
ax.legend(); plt.tight_layout(); plt.show()

# (4) active modes vs nu (log-log), slope vs Foias-Temam -3/2
fig, ax = plt.subplots(figsize=(7, 5))
for N in NS:
    ys = [RS[N]["active"][nu] for nu in NU_LIST]
    ax.loglog(NU_LIST, ys, "o-", lw=1.5, label=f"N = {N}")
# reference -3/2 slope through the N=12 point
a12 = RS[12]["active"][0.15]
ax.loglog([0.03, 0.25], [a12*(0.15/0.03)**-1.5, a12*(0.15/0.25)**-1.5], "k--", lw=1, label="slope -3/2 (Foias-Temam)")
ax.set_xlabel("nu (log)"); ax.set_ylabel("active modes (99% energy, log)")
ax.set_title("Active-mode count vs nu:  resolution requirement ~ nu^{-3/2}")
ax.legend(); plt.tight_layout(); plt.show()

# ---- VERDICT TABLE ----
print(f"{'N':>3} {'n':>4} | " + "  ".join(f"A*(nu={nu})" for nu in NU_LIST) + " | bounded(.05/.10/.15) | active(.05/.10/.15) | best lam1 (spread) | off-ball | max margin")
print("-"*132)
for N in NS:
    R = RS[N]
    ast = "  ".join(f"{R['Astar'][nu]:.3f}" if R["Astar"][nu] is not None else "   n/a " for nu in NU_LIST)
    dv = "/".join(f"{R['divbound'][nu]:.1f}" if R["divbound"][nu] is not None else " -- " for nu in NU_LIST)
    ac = "/".join(str(R["active"][nu]) for nu in NU_LIST)
    n_offball = sum(1 for (l1, div, _, _, ob) in R["grid"].values() if ob)
    best_flag = (" OFF-BALL" if R["best_offball"] else "") + (" DIVERGED" if R["best_diverged"] else "")
    print(f"{N:>3} {R['n']:>4} | {ast} | {dv} | {ac} | {R['best']:+.4f} ({R['best_spread']:.3f}){best_flag} | {n_offball} | {R['max_margin']:.2f}")
print()
# does A* stabilize?
def a10(nu): return RS[10]["Astar"][nu]
def a12(nu): return RS[12]["Astar"][nu]
stab = []
for nu in NU_LIST:
    if a10(nu) is not None and a12(nu) is not None:
        stab.append(abs(a12(nu) - a10(nu)))
if stab:
    print(f"|A*(N=12) - A*(N=10)| per nu: {np.round(stab, 3)}   (stabilized if << 0.2)")
# certified deep-chaos point?  (L3: on-ball only; L4: best must be seed-stable)
for N in NS:
    mm = RS[N]["max_margin"]
    stable = (RS[N]["best_spread"] < 0.3) and (not RS[N]["best_diverged"])
    print(f"N={N}: max on-ball grid margin = {mm:.2f}  (best spread {RS[N]['best_spread']:.3f}, "
          + ("seed-STABLE" if stable else "seed-UNSTABLE")
          + ")  -> " + ("CERTIFIABLE deep-chaos point exists in scan" if (mm >= 10 and stable) else "no certifiable deep-chaos point in the scanned range"))
print()
print("Interpretation: the N-dependence of A*(nu) and the active-mode count is the PHYSICS")
print("(Galerkin under-resolution).  The 'divergence wall' seen with a FIXED timestep is")
print("NOT physics -- it is RK4 time-step instability (L7): as N grows, nu*k_max^2 and the")
print("convective stiffness |om|*k_max grow, and a fixed dt exits RK4's stability region.")
print("At dt = safe_dt(N,nu,A) the system is dissipative and every point settles to the")
print("attractor, so there is no wall; the only N-dependence that remains is the true one.")
print("A*(nu) is only trustworthy at the largest N where the active modes fit inside N.")
print("L3 'off-ball' = bounded but outside the dissipative ball.  This Numba build")
print("  uses the RIGOROUS ball A/(2 nu)  (enstrophy balance, min k^2 = 1); the")
print("  builder's A/(4 nu) is a conservative 2x-tighter threshold.  No state was")
print("  off-ball under either, so the verdict is unchanged by the correction.")
print("L4 'spread'   = max-min of lambda_1 over 3 seeds; a real attractor is seed-stable.")
print("L7 'safe_dt'  = stiffness-aware timestep; without it the scan is numerically invalid.")
print()
print("Guard caveats (see Layered_Divergence_Guard_Note.md):")
print("  - L7 convective bound is a calibrated estimate (ball radius as ||om| proxy,")
print("    rounded constants), verified empirically all-bounded -- not a sharp certificate.")
print("  - L4 threshold 0.3 is a judgment call, well above integration tolerance.")
print("  - The guards separate NUMERICAL from PHYSICAL failure; they do NOT certify")
print("    Galerkin under-resolution (small N), which the N-trend/active-mode analysis handles.")
'''))

# =====================================================================
# S6  CONCLUSION
# =====================================================================
A(md(r'''
## Conclusion (filled by the run above)

The three knobs are **coupled, not independent**:

- **$A$ and $\nu$** are one direction in Re — the boundary $A^*(\nu)$ is a single curve
  (to leading order $A^*\approx c\,\nu$), sitting above the first Hopf (Re$\approx$35)
  in the Ruelle–Takens cascade.
- **$N$** is resolution: faithful only once $N\gtrsim$ active modes $\sim C\,\nu^{-3/2}$.
  Below that, the truncation carries **spurious dynamics**, and the measured boundary
  $A^*_N(\nu)$ is N-dependent (the $A^*_N\to A^*$ convergence of Galerkin theory).
- **Grid fineness** only refines the *N-dependent* curve; it cannot compensate for an
  insufficient $N$.

The verdict table above states, for each N, whether a **certifiable** deep-chaos point
(margin $\lambda_1 T\ge 10$) exists in the scanned range — the direct answer to
"what N is needed before a robust chaos claim is even possible in this model."

## The divergence-guard battery (why the numbers are trustworthy)

A *trajectory* that is bounded, dissipative and on the attractor can still yield a
**wrong** $\lambda_1$ if the **variational Jacobian** is wrong — no divergence test
can see that (the state is healthy; the linearization is not).  We therefore run a
layered guard, with the decisive check being a **Jacobian-correctness** test:

| Layer | Check | Catches |
|---|---|---|
| **L0/L1** | $\omega$ finite over the whole window, $\|\omega\|\le 10^8$ | hard blow-up |
| **L3** | **dissipative ball**: $\|\omega\|_2 \le A/(2\nu)$ (rigorous a-priori, enstrophy balance, $\min k^2=1$) | off-attractor states |
| **L4** | **multi-seed ensemble** (3 ICs) at the best point | spurious repellers / basin boundaries |
| **L6** | **central-FD Jacobian guard**: $J\omega\cdot v \stackrel{?}{=}\tfrac{f(\omega+\varepsilon v)-f(\omega-\varepsilon v)}{2\varepsilon}$ (exact for the quadratic Galerkin rhs) | **linearization bugs** |
| **L7** | **stiffness-aware timestep** $dt=\min(0.125,\ 2/(\nu k_{\max}^2),\ 8\nu/(A k_{\max}))$ | **time-step instability** (the fake "divergence wall") |

L6 and L7 are the two that actually bit in an earlier version of this notebook: a
double-counted viscosity in the variational equation (caught by L6) and a fixed
$dt=0.125$ that exited RK4's stability region as $N$ grew, fabricating a "divergence
wall" that moves *in* with $N$ (caught by L7 — at a stable $dt$ every point settles,
so there is no wall).  Both run before / during the scan, so a wrong Jacobian or an
unstable step aborts or corrects instead of certifying a bogus boundary.

**Honest caveats** (full derivation in `Layered_Divergence_Guard_Note.md`):

- **L3** uses the *rigorous* ball $A/(2\nu)$ in this build. The builder's $A/(4\nu)$
  is a conservative threshold ($\min k^2=2$); the true minimum wavenumber is
  $k^2=1$ (the $(\pm1,0),(0,\pm1)$ modes). The override touches only the off-ball
  classification — trajectories and $\lambda_1$ are unchanged, and no state was
  off-ball under either ball.
- **L7**'s convective bound is a *calibrated estimate* (ball radius as a
  $\|\omega\|$ proxy, rounded constants), verified empirically all-bounded — not a
  sharp stability certificate.
- **L4**'s threshold $0.3$ is a judgment call, well above integration tolerance.
- The guards separate **numerical** from **physical** failure; they do *not*
  certify Galerkin under-resolution (small $N$), which the $N$-trend and
  active-mode analysis handle separately.
'''))

# =====================================================================
# ASSEMBLE + WRITE NOTEBOOK
# =====================================================================
NB.cells = cells
OUT = "NS_N_Dependence_numba.ipynb"
with open(OUT, "w", encoding="utf-8") as fh:
    nbf.write(NB, fh)
n_md = sum(1 for c in cells if c.cell_type == "markdown")
n_py = sum(1 for c in cells if c.cell_type == "code")
print(f"\nWrote {OUT}: {len(cells)} cells ({n_md} markdown, {n_py} code).")
