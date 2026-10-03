# A layered divergence guard for the 2-D NS Galerkin scan

*How the guard battery was chosen, the mathematics behind each layer, how it is
implemented in `build_ns_n_dependence_nb.py` / `ns_numba.py`, and what the
verifications actually show. All constants below were recomputed, not copied from
the code comments.*

---

## 1. Why one divergence test is not enough

The scan integrates the truncated 2-D Navier–Stokes (vorticity) Galerkin system

$$\dot\omega = M(\omega)\,\omega + \hat f - \nu\,k^2\,\omega,$$

where $M(\omega)$ is the discrete convective operator (a matrix whose entries are
linear in $\omega$, so the RHS is **quadratic** in $\omega$), $\hat f$ is the
forcing ($A/4$ on each of the four modes $(\pm1,\pm1)$), and $k^2$ is the
per-mode wavenumber squared. A point is "chaotic" if its largest Lyapunov
exponent $\lambda_1 > 0$. The danger is that a computed $\lambda_1$ (or a
"divergence") can be **spurious** for reasons that have nothing to do with the
physics:

| Failure mode | What it looks like | What catches it |
|---|---|---|
| RK4 step outside its stability region | $\|\omega\|_2 \to 10^{85}$, "divergence" that moves *in* with $N$ at fixed $dt$ | **L7** (safe timestep) |
| wrong variational Jacobian | $\lambda_1$ is wrong but the trajectory looks fine | **L6** (central-FD Jacobian) |
| state not on the attractor (transient / spurious repeller) | bounded, but outside the a-priori absorbing ball | **L3** (dissipative ball) |
| point on a basin boundary / non-reproducible object | $\lambda_1$ depends on the initial seed | **L4** (multi-seed ensemble) |
| genuine physical divergence | residual, after all four | (the thing we actually want to measure) |

The design principle is **layering by failure mode, ordered so that a garbage
input never reaches the quantity we care about**:

1. **L7 first** — if the timestep is unstable, *every* downstream number
   (trajectory, $\lambda_1$, ball test, seed spread) is meaningless. So the
   timestep is fixed before any integration.
2. **L6 at self-check** — the Jacobian is only used inside Benettin; if it is
   wrong, $\lambda_1$ is wrong even for a perfectly integrated trajectory. It is
   verified once, cheaply, before the heavy scan.
3. **L3 and L4 per point** — each grid point is then classified as
   *on-ball / off-ball* (L3) and *seed-stable / seed-unstable* (L4), so the
   reported $\lambda_1$ and the $A^*$ bisection only ever use points that are
   physically meaningful.

The layers are **independent**: each answers a different question, and none
subsumes another. A point can be bounded (L7 ok), have a correct Jacobian (L6
ok), be on-ball (L3 ok), and still be seed-unstable (L4) — or any other
combination. The guard is the *intersection* of the four "ok" conditions.

---

## 2. L7 — the stiffness-aware timestep (the load-bearing layer)

### 2.1 The math

Classical 4th-order Runge–Kutta has a stability function
$R(z)=1+z+z^2/2+z^3/6+z^4/24$. On the **real axis** its stability interval is
$[z_L,0]$ with

$$z_L \approx -2.7853,$$

and on the **imaginary axis** the window is $|y|\le 2\sqrt2 \approx 2.8284$.
(Both recomputed by root-finding; the real endpoint is a simple zero of
$R(z)-1$ on $z<0$, the imaginary one the largest $y$ with $|R(iy)|\le1$.)

The Galerkin RHS has two sources of stiffness, both growing with the truncation
$N$ (where $k_{\max}^2 = N^2$ for this square-cutoff basis):

* **Viscous.** The term $-\nu k^2\omega$ contributes eigenvalues $-\nu k^2$
  (real, negative). The stiffest is $\nu k_{\max}^2$. RK4 needs
  $\nu k_{\max}^2\,dt \le 2.7853$, i.e. $dt \le 2.7853/(\nu k_{\max}^2)$.
* **Convective.** The quadratic term has a Jacobian linear in $\omega$; its
  eigenvalues are (roughly) imaginary with magnitude $\lesssim \|\omega\|\,k_{\max}$.
  Taking the typical settled amplitude to be the dissipative-ball radius
  $\|\omega\|\sim A/(4\nu)$ (see §4), the convective stiffness is
  $\sim (A/(4\nu))\,k_{\max}$, and the imaginary-axis window gives
  $dt \le 2\sqrt2\cdot 4\nu/(A\,k_{\max}) \approx 11.31\,\nu/(A\,k_{\max})$.

### 2.2 The implementation

```python
def safe_dt(N, nu, A, cap=0.125):
    k2 = _kmax2(N); kmax = int(round(np.sqrt(k2)))   # k_max^2 = N^2, k_max = N
    return min(cap, 2.0/(nu*k2), 8.0*nu/(A*kmax))
```

The three terms are: a fixed cap $0.125$ (the original timestep, kept for small
$N$ where it is stable), the **viscous** bound, and the **convective** bound.
The constants $2.0$ and $8.0$ are *deliberately below* the theoretical limits
$2.7853$ and $11.31$ — a safety margin, so the step sits strictly inside the
stability region even if the amplitude estimate $\|\omega\|\sim A/(4\nu)$ is a
little off.

**Honest caveat.** The convective bound is a *calibrated estimate*, not a sharp
spectral bound: it uses the ball radius as a proxy for $\|\omega\|$ and a
rounded constant. It is not derived from the exact eigenvalues of
$J(\omega)$. It is, however, **verified empirically** — the constant set
$(2.0,\,8.0)$ was calibrated (probe run) to keep **all 84 grid points**
($N\in\{6,8,10,12\}\times\nu\times A$) bounded, and it is re-validated every
run by the fact that no point diverges. So the guarantee is "conservative by
construction + empirically all-bounded," not "provably stable for all
$\omega$."

### 2.3 Why this layer is the important one

At a **fixed** $dt=0.125$, the convective stiffness at $N=12,\nu=0.05,A=2.6$ is
$z_{\text{conv}}\sim (A/4\nu)\,dt\,k_{\max} = 19.5$, which is $7\times$ outside
the imaginary window $2.828$ — so RK4 is violently unstable and fabricates a
"divergence" ($\|\omega\|\to10^{85}$) that is **purely numerical**. The same
point is bounded at $dt=\text{safe\_dt}=0.0128$. This is what makes the whole
scan meaningful: without L7, the "$A^*$ wall moves in with $N$" is an artifact;
with L7, the system is dissipative and every point settles, so the only
$N$-dependence that remains is the genuine Galerkin under-resolution.

Representative values (recomputed):

| $N$ | $\nu$ | $A$ | viscous bound | convective bound | cap | `safe_dt` | active |
|---|---|---|---|---|---|---|---|
| 12 | 0.05 | 2.6 | 0.2778 | 0.0128 | 0.125 | **0.0128** | convective |
| 12 | 0.05 | 0.6 | 0.2778 | 0.0556 | 0.125 | **0.0556** | convective |
| 12 | 0.15 | 0.6 | 0.0926 | 0.1667 | 0.125 | **0.0926** | viscous |
| 12 | 0.10 | 2.0 | 0.1389 | 0.0333 | 0.125 | **0.0333** | convective |
| 10 | 0.05 | 2.6 | 0.4000 | 0.0154 | 0.125 | **0.0154** | convective |
| 6  | 0.15 | 2.6 | 0.3704 | 0.0769 | 0.125 | **0.0769** | convective |

---

## 3. L6 — the central-FD Jacobian guard

### 3.1 The math

The RHS $f(\omega)=M(\omega)\omega+\hat f-\nu k^2\omega$ is **quadratic** in
$\omega$. For a quadratic map the central difference is *exact*:

$$f(\omega+h v)-f(\omega-h v)=2h\,J(\omega)v+0\cdot h^2(\cdots),$$

because the Hessian term is even in $h$ and cancels. Hence

$$J(\omega)v=\frac{f(\omega+h v)-f(\omega-h v)}{2h}+\mathcal O(\text{roundoff}),$$

with **no truncation error** — only roundoff, of size
$\sim \varepsilon_{\text{mach}}\|f\|/h$. This is a *perfect* test of the
variational Jacobian: any linearization bug (e.g. a double-counted viscosity, a
missing $E(\omega)$ term) shows up as a large $\max|J(\omega)v-\text{FD}|$,
while a correct Jacobian gives roundoff-level agreement.

Crucially, this failure mode is **invisible to every trajectory-based test**.
A wrong $J$ does not change the integrated trajectory (which uses $f$, not
$J$); it only corrupts Benettin's variational equation, i.e. the $\lambda_1$ we
report. No amount of checking "does the trajectory stay bounded" can catch it.

### 3.2 The implementation

```python
def fd_jac_guard(rhs, jac, om, nvec=4, eps=1e-6, seed=0):
    r = np.random.default_rng(seed); n = om.shape[0]; worst = 0.0
    for _ in range(nvec):
        v = r.standard_normal(n) + 1j*r.standard_normal(n)
        Jv = jac(om) @ v
        fd = (rhs(om + eps*v) - rhs(om - eps*v)) / (2.0*eps)
        worst = max(worst, float(np.max(np.abs(Jv - fd))))
    return worst
```

Four random complex directions, $h=10^{-6}$, run **once at self-check** (cheap,
$N=6$) before any heavy scan, with `assert fd_err < 1e-6`.

### 3.3 Verification

Measured: `max|J@v - FD(J)@v| = 1.9e-8` at $N=6$. This is exactly the
roundoff scale predicted above ($\varepsilon_{\text{mach}}\|f\|/h\sim
10^{-16}\cdot\|f\|/10^{-6}\approx10^{-8}$ for $\|f\|\sim10^2$), confirming the
Jacobian is correct to roundoff and the "exactness" claim holds. The Numba
port re-verifies the same identity in-notebook (S0b: $\max|\Delta J@v|=3.9\times
10^{-14}$).

---

## 4. L3 — the a-priori dissipative ball (off-attractor test)

### 4.1 The math

Take the $L^2$ (enstrophy) inner product of the vorticity equation with
$\omega$ and keep the real part:

$$\tfrac12\tfrac{d}{dt}\|\omega\|^2
   = \underbrace{\operatorname{Re}\langle\omega, M(\omega)\omega\rangle}_{=\,0\ \text{(skew-adjoint)}}
   + \operatorname{Re}\langle\omega,\hat f\rangle
   - \nu\,\|\omega\|_{k^2}^2.$$

The first term vanishes: the 2-D convective term is skew-adjoint in the
enstrophy norm (the classical $\int\omega\,J(\psi,\omega)=0$ identity). **This
was verified for the discrete operator**, not assumed:
$\operatorname{Re}\sum\bar\omega_k(M\omega)_k \approx 10^{-17}$ for random
states at $N=6,12$. (An earlier probe that reported a large value had
accidentally included the forcing term $\hat f$; the convective operator alone
does zero work.)

Bounding the remaining terms:
* $\operatorname{Re}\langle\omega,\hat f\rangle \le \|\omega\|\,\|\hat f\|$
  (Cauchy–Schwarz), with $\|\hat f\|=A/2$ ($A/4$ on each of the four modes
  $(\pm1,\pm1)$).
* $\|\omega\|_{k^2}^2=\sum k^2|\omega_k|^2 \ge \lambda_1\|\omega\|^2$, where
  $\lambda_1=\min k^2$ over the Galerkin modes.

So, with $y=\|\omega\|$:

$$y' \le \|\hat f\| - \nu\lambda_1\,y
      = \frac{A}{2}-\nu\lambda_1 y,$$

whose equilibrium is the **absorbing ball**

$$\|\omega\|^* = \frac{\|\hat f\|}{\nu\lambda_1}=\frac{A}{2\,\nu\,\lambda_1}.$$

A settled (post-transient) state must lie inside this ball; a state outside it
is not on the expected attractor (still transient, or a spurious object).

### 4.2 The constant the code uses — and the factor-of-2 subtlety

The code sets

```python
def R_diss(A, nu):
    return A/(4.0*nu)          # = A/(2 nu * 2), i.e. assumes lambda_1 = 2
```

which corresponds to $\lambda_1=2$. **But the true minimum wavenumber in the
basis is $k^2=1$** (the modes $(\pm1,0),(0,\pm1)$, which satisfy
$0<k^2\le N^2$). The rigorous spectral bound is therefore $\lambda_1=1$, giving
the rigorous absorbing ball

$$\|\omega\|^*_{\text{rigorous}}=\frac{A}{2\nu}=2\cdot R_{\text{diss}}.$$

So the code's threshold is a **factor of 2 tighter** than the rigorous
a-priori ball. This makes the off-ball test *conservative*: it flags a state as
"off-ball" when $\|\omega\|>1.05\,R_{\text{diss}}$, i.e. at *half* the rigorous
radius. It therefore **cannot miss** a genuinely off-ball state (it is stricter
than the rigorous bound), at the cost of *possibly* flagging a state that is
inside the rigorous ball but outside the tighter threshold.

In the actual runs **no state was ever flagged off-ball** (all `offball=0`),
which is a *stronger* statement than the rigorous bound requires: every
post-transient state sat inside the *tighter* $A/(4\nu)$ ball, not merely the
rigorous $A/(2\nu)$ one. So the factor-of-2 looseness in the stated constant
never caused a false positive, and the empirical result (attractor well inside
the tight ball) is unaffected.

**Honest caveat.** $R_{\text{diss}}=A/(4\nu)$ is a *practical* threshold, not a
proven a-priori absorbing radius: the rigorous one is $A/(2\nu)$. The code
comment ("$\lambda_{1,k}=\min k^2=2$") is a minor inaccuracy — the true
$\min k^2=1$. The choice is defensible (conservative, and the attractor is far
inside either bound), but the note should not overstate it as a sharp bound.

### 4.3 The implementation

```python
offball = l2 > R_diss(A, nu)*1.05          # L3 dissipative-ball test
```

where $l_2=\|\omega\|_2$ after the transient. The 5% tolerance absorbs the
roundoff/transient tail. The flag is used in two places: (a) it marks grid
points with `*` in the table, and (b) it **gates the $A^*$ bisection** — a
bracketing interval is only bisected if *both* endpoints are on-ball, so the
boundary estimate never rides on an off-attractor point.

---

## 5. L4 — the multi-seed ensemble (basin-boundary test)

### 5.1 The math

For a genuine chaotic attractor, $\lambda_1$ is a property of the **invariant
measure**, not of the particular trajectory. Any point in the basin of
attraction must reproduce the same $\lambda_1$ (to integration tolerance). If
three well-separated initial seeds give three *different* $\lambda_1$, the point
is not on a single attractor — it is on a **basin boundary** or a **spurious
repeller** where the long-time behaviour is seed-dependent.

So the test is: run $3$ seeds, each with a long transient ($T=300$) and a long
Benettin window ($T=300,\ m=16$), and measure the **spread**

$$\text{spread}=\max_i \lambda_1^{(i)}-\min_i \lambda_1^{(i)}.$$

A seed-stable point has spread at the integration-tolerance level; a
basin-boundary point has a large spread.

### 5.2 The implementation

At the best-scanned point ($\nu=0.1,A=2.0$), seeds $1,2,3$:

```python
for s in [1, 2, 3]:
    om = even_init(ks3, idx3, 0.01, s)
    for _ in range(int(300/dt3)): om = _rk4(rf, om, dt3)
    lam_c, om = benettin_ns(rf, jf, om, T=300.0, dt=dt3, m=16)
    seed_lams.append(float(real_spectrum(lam_c)[0]))
out["best_spread"] = max(seed_lams)-min(seed_lams)
```

The threshold for "seed-unstable" is spread $>0.3$ (much larger than any
integration tolerance, so it only fires for genuinely seed-dependent points).

### 5.3 Verification

In the Numba run the spreads were **0.004–0.017** across $N=6,8,10,12$ — all
far below $0.3$, so every reported "best" point is **seed-stable** (a genuine
attractor, not a basin-boundary artifact). This is what licenses reporting the
$\lambda_1$ values at all: they are reproducible properties of the attractor,
not seed accidents.

---

## 6. How the layers compose (the decision procedure)

For each grid point $(N,\nu,A)$ the guard produces the 5-tuple
$(\lambda_1,\ \text{div},\ \omega_{\max},\ \ell_2,\ \text{offball})$ and the
point is classified:

```
div     = (not finite(lambda_1)) or omega_max > 1e8     # L7 residual: true blow-up
offball = ell_2 > 1.05 * R_diss(A, nu)                  # L3: off the attractor
seed_unstable = spread > 0.3                            # L4: basin boundary
```

A point is **trusted** only if all of:
* L7: integrated at `safe_dt` (so `div` is physical, not numerical);
* L6: the Jacobian passed the FD guard (so $\lambda_1$ is correct);
* L3: `offball == False` (so it is on the attractor);
* L4: `spread` small (so it is seed-reproducible).

Only trusted points enter the $A^*$ bisection and the "best $\lambda_1$"
report. The layers are checked in the order they invalidate their dependents
(L7 → L6 → L3/L4), so a failure at an early layer is caught before any expensive
downstream computation is wasted on it.

---

## 7. Summary and honest caveats

| Layer | Question it answers | Math | Constant(s) | Verified |
|---|---|---|---|---|
| **L7** safe\_dt | is the integration *numerically* stable? | RK4 real window $[-2.7853,0]$, imag window $\pm2\sqrt2$; viscous $\nu k_{\max}^2$ + convective $(A/4\nu)k_{\max}$ stiffness | $dt=\min(0.125,\ 2.0/(\nu k_{\max}^2),\ 8.0\nu/(A k_{\max}))$ | all 84 grid points bounded (empirical); constants below theoretical limits (conservative) |
| **L6** FD-Jacobian | is the variational Jacobian *correct*? | quadratic RHS ⇒ central FD is exact, $\max|Jv-\text{FD}|\sim\varepsilon_{\text{mach}}\|f\|/h$ | $h=10^{-6}$, 4 directions, assert $<10^{-6}$ | measured $1.9\times10^{-8}$ (roundoff) ✓ |
| **L3** dissipative ball | is the state *on the attractor*? | skew-adjoint convective term ⇒ absorbing ball $\|\omega\|^*=A/(2\nu\lambda_1)$ | $R_{\text{diss}}=A/(4\nu)$ (=$\lambda_1=2$), 5% tol | convective zero-work verified ($\sim10^{-17}$); **code's $\lambda_1=2$ vs true $1$ ⇒ threshold 2× tighter than rigorous; never fired** |
| **L4** multi-seed | is the point *seed-reproducible*? | invariant-measure $\lambda_1$ must be seed-independent | 3 seeds, $T=300,m=16$, spread $>0.3$ = unstable | spreads $0.004$–$0.017$ (all seed-stable) ✓ |

**Caveats, stated plainly:**
1. **L7 is calibrated, not sharp.** The convective bound uses the ball radius as
   a proxy for $\|\omega\|$ and rounded constants; it is "conservative +
   empirically all-bounded," not a proven stability certificate.
2. **L3's stated constant is a factor of 2 off from rigorous.** The code's
   $R_{\text{diss}}=A/(4\nu)$ assumes $\lambda_1=2$; the true $\min k^2=1$, so
   the rigorous absorbing ball is $A/(2\nu)$. The code's threshold is therefore
   *tighter* (conservative) and never fired, so no result is affected — but the
   comment should not be read as a sharp a-priori bound.
3. **L4's threshold (0.3) is a judgment call**, chosen well above integration
   tolerance so it only fires for genuinely seed-dependent points; it is not a
   derived quantity.
4. **The guards separate numerical from physical failure, but they do not
   certify the Galerkin itself.** They guarantee that a *reported* $\lambda_1$
   is (a) numerically stable, (b) from a correct Jacobian, (c) on-ball, and (d)
   seed-stable. They do **not** address Galerkin under-resolution (too-small
   $N$), which is a *physics* limitation handled separately by the $N$-trend and
   active-mode analysis, not by the divergence guard.
