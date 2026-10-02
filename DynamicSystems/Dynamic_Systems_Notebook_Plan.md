# Implementation Plan — JupyterLab Notebook for *Dynamic Systems Theory* (v2)

**Source essay (read-only):** `D:\Source\hermes-dir\Dynamic_Systems_Theory_v2.md`
**Hard constraint:** the v2 essay — and every other saved version of it, at the workspace root and in `DynamicSystems/` — is **not modified in any way**. All new artifacts go into new files (Section 4).

---

## 1. Goal

A single, self-contained JupyterLab notebook (`.ipynb`) that treats **every aspect of the v2 essay that is symbolically or numerically treatable**:

- **Symbolically** with `sympy`: exact fixed points, multipliers, characteristic polynomials, divergences, symplectic/Poisson identities, fiber invariance, Jacobi fields, flow/conjugacy verification.
- **Numerically** with `numpy`/`scipy`/`matplotlib`: Lyapunov spectra (Benettin), invariant manifolds, shadowing, Poincaré sections (KAM/Greene), pressure & Gibbs measures, Kaplan–Yorke / box / correlation / information dimensions, a 2D Navier–Stokes Galerkin attractor, structure-preserving vs. non-symplectic integration, predictability horizons.

The notebook **mirrors the essay's section numbering (1–8)** so every cell can be cross-referenced to its source text. Pure-theory content (theorems, conjectures, open problems) appears as **markdown cells with the essay's precise statement and citation**, immediately followed by whatever computational evidence exists.

## 2. Environment (verified on 2026-10-01)

| Item | Value |
|---|---|
| venv | `D:\Source\hermes-dir\.venv` (Python 3.11) |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| sympy | 1.14.0 |
| matplotlib | 3.11.2 |
| nbformat / nbclient / ipykernel | 5.11.1 / 0.11.0 / 7.3.0 |

All required modules are present — **no installation step needed**. All Python execution (builder, headless execution, verification) uses `D:/Source/hermes-dir/.venv/Scripts/python.exe` (native forward-slash paths; never MSYS paths).

## 3. Scope decisions

**In scope (computed):**

| Essay § | Aspect |
|---|---|
| 1 | Flow property, conjugacy, Liouville/symplectic preservation (symbolic) |
| 2.1 | Fixed/periodic points of Hénon; numerical non-wandering set |
| 2.2 | Hyperbolicity constants for a toral automorphism; grown $W^s, W^u$ of the Hénon saddle |
| 2.3 | Shadowing lemma, verified numerically on an Anosov map |
| 2.4 | Hopf fibration: fiber invariance (symbolic), isometric Hopf flow (numeric), 3-D picture |
| 2.5 | Geodesic flow on the hyperbolic plane: Lyapunov exponent $= \sqrt{|K|} = 1$; Jacobi fields (symbolic) |
| 2.6 | Stable/unstable foliation of a toral automorphism; holonomy |
| 3.1 | Birkhoff averages converging to space averages |
| 3.2 | Oseledets: numeric exponents of the cat map vs. exact $\log|\lambda_i|$; Jacobian identity |
| 3.3 | Pesin formula for an Anosov map; Ruelle's inequality for a non-SRB (delta) measure |
| 3.4 | KS entropy by partition refinement (Hénon); entropy–Jacobian relation (cat map) |
| 3.5 | Pressure and Gibbs property for a subshift of finite type (Ruelle–Perron–Frobenius operator) |
| 3.6 | Kaplan–Yorke dimension for Lorenz and Hénon |
| 4.1 | Action–angle variables; Poisson brackets of first integrals (symbolic) |
| 4.2 | Diophantine condition: measure of the Diophantine set as a function of $\gamma$ |
| 4.3 | Chirikov standard map: Poincaré sections across $K$; **Greene's residue criterion** estimating $K_c \approx 0.9716$ (vs. naive $\pi^2/4$) |
| 4.4 | Arnold diffusion: order-one action drift in the standard map, compared qualitatively with Nekhoroshev-type times |
| 4.5 | Symplectic duality $E^u = (E^s)^{\omega^\perp}$ for a symplectic Anosov map (symbolic + numeric check) |
| 5.1 | Lorenz: attractor, exact linearization at the origin, divergence, Benettin spectrum, $D_{LY}\approx 2.06$, homoclinic tangle, failure of uniform hyperbolicity at the saddle |
| 5.2 | Hénon: exact fixed point & multipliers, attractor, spectrum, dimensions, "interval × Cantor" structure |
| 5.3 | Ruelle–Takens route: three successive Hopf bifurcations (fixed point → $\mathbb T^1$ → $\mathbb T^2$ → $\mathbb T^3$/chaos) via a 4-D normal-form ODE |
| 5.4 | 2D Navier–Stokes on the torus: Galerkin truncation, attractor, Kaplan–Yorke dimension of the truncated system, convergence in $N$, Foias–Temam-type dimension bound |
| 5.5 | Predictability horizon $t_{pred} \sim \lambda_1^{-1}\log(1/\delta_0)$, fitted from two-particle experiments |
| 6.1 | Full Benettin implementation; demonstration of the need for re-orthogonalization |
| 6.2 | RK4 vs. symplectic integrator: long-time energy drift; symplecticity of the standard map |
| 6.3 | Shadowing: success on the Anosov cat map vs. drift on the Hénon attractor (scope boundary of the lemma) |
| 6.4 | Box-counting, correlation (Grassberger–Procaccia), information dimensions of Hénon (and Lorenz); comparison table with $D_{LY}$ and essay-quoted values |

**Out of scope (markdown discussion only, no computation):** the Pugh–Shub stable ergodicity conjecture (§7.1), local/global rigidity, generic Arnold diffusion rates, 3D Navier–Stokes regularity (Millennium problem), blenders and the classification of partially hyperbolic systems, the quantum–classical interface, JSJ decomposition and foliation regularity theory, the exact value of the Hopf invariant (stated as a theorem; the *related* properties — fiber invariance, isometry of the Hopf flow, nontriviality of $S^3\to S^2$ — are verified numerically/symbolically where possible).

## 4. Deliverables and file layout

All new files go into `D:\Source\hermes-dir\DynamicSystems\` (which currently holds only essay copies — those remain untouched):

| Artifact | Path |
|---|---|
| This plan | `DynamicSystems\Dynamic_Systems_Notebook_Plan.md` |
| Builder script (assembles cells → `.ipynb`) | `DynamicSystems\build_dynamic_systems_nb.py` |
| **Notebook (the deliverable)** | `DynamicSystems\Dynamic_Systems_v2_notebook.ipynb` |
| Verification script | `DynamicSystems\verify_dynamic_systems_nb.py` |

The notebook is a standard `.ipynb` (JupyterLab-compatible). For local use in JupyterLab the kernel should be the venv Python; optionally (only if the user confirms) register it once with
`.venv\Scripts\python.exe -m ipykernel install --user --name dynamic-systems --display-name "Dynamic Systems (venv)"`.

## 5. Notebook architecture

**~75–85 cells** (markdown + code), **~23 embedded figures**, one reusable helper block. Section order mirrors the essay:

```
S0  Setup: %matplotlib inline, imports, seeds, reusable components
S1  The unifying viewpoint (essay §1)
S2  Geometric & topological foundations (essay §2.1–2.6)
S3  Ergodic & measure-theoretic theory (essay §3.1–3.6)
S4  Hamiltonian dynamics & KAM (essay §4.1–4.5)
S5  Physical applications (essay §5.1–5.5)
S6  Numerical analysis (essay §6.1–6.4)
S7  Open problems (markdown summary, cross-referencing essay §7)
S8  Summary: computed constants vs. essay-quoted values (verification table)
```

**Reusable components (defined once in S0, used throughout):**

- `benettin_lyapunov(vecfield_or_map, x0, T, dT, m)` — Benettin algorithm with Gram–Schmidt re-orthogonalization; returns the top `m` exponents and the per-window growth factors (also exposes the *non-re-orthogonalized* variant for the S6.1 contrast figure).
- `box_dimension(pts, epsilons)`, `correlation_dimension(pts, radii)`, `information_dimension(pts, epsilons)` — Grassberger–Procaccia-style estimators with log-log slope fits.
- System definitions: `cat_map`, `hene_map`, `lorenz_vecfield`, `standard_map` (vectorized for Poincaré sections), `hyperbolic_geodesic_flow`, `ns2d_galerkin(N)`.
- `shadow_orbit(map, pseudo_orbit, x0_guess)` — least-squares search for the true orbit shadowing a noisy pseudo-orbit (`scipy.optimize`).
- `poincare_section(K, n_orbits, n_iter)` — vectorized numpy Poincaré sections of the standard map.

**Determinism:** all stochastic ingredients (pseudo-orbit noise, random initial conditions, point sampling for dimension estimators) use a fixed `np.random.default_rng(seed)` so re-execution reproduces identical figures and numbers.

## 6. Section-by-section plan

For each section: **MD** = markdown (theorem statements, citations, scope remarks), **SY** = sympy cell(s), **NU** = numerical cell(s), **FIG** = figure(s).

### S1 — The unifying viewpoint (essay §1)
- **MD:** group-action definition, flows vs. diffeomorphisms, conjugacy; Riemannian/symplectic/measure structure; the local-vs-global tension.
- **SY-1:** linear vector field $X = Ax$ on $\mathbb R^2$: solve $\dot x = Ax$ symbolically, get $\varphi_t = e^{At}$, and verify the flow property $\varphi_{t+s} = \varphi_t \circ \varphi_s$ and $\varphi_0 = \mathrm{id}$ (symbolic simplification to 0).
- **SY-2:** conjugacy: for two similar matrices $A, B$, compute $h = P$ with $B = PAP^{-1}$ and verify $g \circ h = h \circ f$ symbolically.
- **SY-3 (Liouville):** for a general 2-D Hamiltonian $H(q,p)$, construct $X_H$ from $\iota_{X_H}\omega = dH$ and verify $\nabla \cdot X_H = 0$ and $\mathcal L_{X_H}\omega = 0$ symbolically → conservation of phase volume.

### S2 — Geometric & topological foundations (essay §2)
- **S2.1 (orbits, invariant sets):**
  - **SY-4:** Hénon fixed points: solve $x = 1 - ax^2 + y,\ y = bx$ exactly → $x = \frac{-(1-b) \pm \sqrt{(1-b)^2 + 4a}}{2a}$; evaluate at $a=1.4, b=0.3$: $x_* \approx 0.6314,\ y_* \approx 0.1894$ (matches essay's $(0.63, 0.19)$).
  - **NU:** period-2 points of Hénon (solve the 4-D polynomial system numerically with `scipy.optimize`), mark fixed/periodic points on the attractor; long-orbit return test as a numerical non-wandering-set indicator for a dissipative map.
  - **FIG-1:** Hénon attractor with fixed and period-2 points marked.
- **S2.2 (hyperbolicity, invariant manifolds):**
  - **MD:** hyperbolicity definition, Anosov map/flow, Hirsch–Pugh–Shub theorem, inclination lemma.
  - **NU:** cat map $A = \begin{pmatrix}1&1\\1&2\end{pmatrix}$: iterate $Tf^n$ on random unit vectors and verify $\|Tf^n v\| \le C\lambda^n$ with $\lambda \to \log\varphi^2$ (plot $\tfrac1n\log\|Tf^n v\|$ vs. $n$, converging to $\log|\lambda_{\max}| \approx 0.9624$); extract $C, \lambda$ numerically.
  - **NU:** grow local $W^s, W^u$ of the Hénon saddle by iterating small arcs forward/backward (standard manifold-growing algorithm), verify tangency to the eigenvectors of $DH(x_*)$ (computed exactly in **SY-5**: multipliers $\approx -1.924, +0.156$, matching the essay).
  - **FIG-2:** $\frac1n \log\|Tf^n v\|$ convergence (cat map). **FIG-3:** Hénon stable/unstable manifold tangle with eigenvector directions.
- **S2.3 (structural stability, Axiom A, shadowing):**
  - **MD:** Axiom A, spectral decomposition, Smale's $Q$-stability, shadowing lemma statement (with the essay's scope caveat).
  - **NU:** shadowing experiment on the cat map: generate an $\varepsilon$-pseudo-orbit (Gaussian noise, fixed seed), find the true orbit via least squares on the initial condition, plot pseudo- vs. true-orbit and the shadowing distance (bounded, $\delta$ small) — the quantitative content of the lemma.
  - **FIG-4:** pseudo-orbit vs. true orbit (cat map).
- **S2.4 (Hopf fibration):**
  - **SY-6:** on $S^3 \subset \mathbb C^2$, verify symbolically that the Hopf map $h(z_1,z_2) = [z_1:z_2]$ is invariant under the fiber action $(z_1,z_2)\mapsto(e^{i\theta}z_1, e^{i\theta}z_2)$, and that the fiber generator has unit speed (fibers are great circles).
  - **NU:** Hopf flow isometry check: evolve two points on $S^3$ along the flow and verify the round distance is preserved (to machine precision); **MD:** Hopf invariant $= 1$ stated as a theorem (not numerically computable); the essay's point that the Hopf flow is *not* hyperbolic.
  - **FIG-5:** fibers over 3 base points, embedded in $\mathbb R^4$ and projected to $\mathbb R^3$ (plus the base map $S^3 \to S^2$).
- **S2.5 (geodesic flows, negative curvature):**
  - **MD:** Anosov's theorem for geodesic flows on $K \le -1$; Jacobi-equation proof sketch.
  - **SY-7:** Jacobi field equation on a constant-curvature surface, $v'' + K v = 0$: solve symbolically → for $K = -1$, $v(t) = c_1 e^t + c_2 e^{-t}$ — the uniform exponential growth/decay behind the Anosov property.
  - **NU:** integrate geodesics in the Poincaré disk ($K=-1$); run Benettin on the geodesic flow and verify the Lyapunov exponent converges to $\sqrt{|K|} = 1$.
  - **FIG-6:** geodesics in the Poincaré disk + Lyapunov exponent convergence to 1.
- **S2.6 (foliations, holonomy):**
  - **NU:** cat map stable/unstable foliation = families of parallel lines with eigenvector slopes (plot); holonomy map computed explicitly (linear, constant along the foliation) — the model case of the essay's regularity discussion.
  - **FIG-7:** foliation plot on the torus (fundamental domain).

### S3 — Ergodic & measure-theoretic theory (essay §3)
- **S3.1 (invariant measures, ergodicity):**
  - **NU:** Birkhoff averages of an observable (e.g. $g(x) = x_1^2$) under the cat map with Lebesgue measure: convergence to $\int g\,d\mathrm{Leb}$ (exact value known); same for Hénon with the empirical (SRB-type) measure.
  - **FIG-8:** Birkhoff average vs. $n$ (two panels).
- **S3.2 (Oseledets):**
  - **NU:** Benettin exponents of the cat map vs. exact $\log\varphi^2 \approx 0.9624$, $\log\varphi^{-2} \approx -0.9624$ (agreement to ~3 digits); Jacobian identity $\sum_i \lambda_i = \log|\det A| = 0$ checked numerically.
- **S3.3 (Pesin):**
  - **NU:** Pesin entropy formula for the cat map: $h_{\mathrm{Leb}}(A) = \sum_{\lambda_i>0}\lambda_i = \log\varphi^2$ (verified against the partition-entropy estimate of S3.4); Ruelle's inequality for the delta measure at the origin: $h_\delta(A) = 0 \le \int \sum_{\lambda_i>0}\lambda_i\,d\delta$ (both sides computed, strict inequality displayed).
- **S3.4 (KS entropy, entropy–Jacobian):**
  - **NU:** KS entropy of the Hénon map by refinement of a 2-D grid partition: $h \approx \lambda_1 \approx 0.49$ (consistent with the essay's SRB discussion); cat map: $h = \sum_{|\lambda_i|>1}\log|\lambda_i| = \log\varphi^2$ — the entropy–Jacobian relation of the essay, with the explicit note that the full Jacobian gives 0 for volume-preserving maps.
- **S3.5 (thermodynamic formalism, Gibbs measures):**
  - **MD:** pressure, equilibrium states, Gibbs property, Bowen's Markov partition theorem.
  - **NU:** full 2-shift (SFT): compute the pressure $P(\varphi)$ for a Hölder potential (e.g. $\varphi = c\,\mathbf 1_{\{0\}}$) via the leading eigenvalue of the Ruelle–Perron–Frobenius matrix (finite-section, convergence in section size); $P(0) = \log 2$; verify the Gibbs inequality $C^{-1} \le \mu(R)/e^{-nP(\varphi) + S_n\varphi} \le C$ on rectangles (plot the ratio, bounded); MME = Bernoulli$(\tfrac12,\tfrac12)$, $h = \log 2$ via partition entropy.
  - **FIG-9:** Gibbs ratio on rectangles (bounded band).
- **S3.6 (Kaplan–Yorke dimension):**
  - **NU:** $D_{LY}$ for Lorenz (from S5.1 spectrum: $\approx 2.06$) and Hénon (from S5.2 spectrum: $\approx 1.29$); MD notes the Kaplan–Yorke conjecture's status and the essay's quoted $1.26$ (see Risk R2).

### S4 — Hamiltonian dynamics & KAM (essay §4)
- **S4.1 (integrable systems):**
  - **SY-8:** harmonic oscillator → action–angle variables $(\theta, I)$, $H = \omega I$, motion $\theta(t) = \theta_0 + \omega t$.
  - **SY-9:** separable 2-DOF Hamiltonian $H = \tfrac{p_1^2}{2} + V_1(q_1) + \tfrac{p_2^2}{2} + V_2(q_2)$: first integrals $E_1 = H_1(q_1,p_1)$, $E_2 = H - E_1$; verify $\{E_1, E_2\} = 0$ symbolically (involution).
- **S4.2 (small divisors, Diophantine condition):**
  - **NU:** for $n=2$ and fixed $\tau$: sweep $\gamma$, measure the Lebesgue measure of the complement of the Diophantine set $\{|k\cdot\omega| \ge \gamma/|k|^\tau\ \forall |k|\le K_{max}$} on the frequency square (Monte Carlo + grid refinement, fixed seed); plot the measure gap closing as $\gamma \to 0$ — the "most tori survive" content.
  - **FIG-10:** measure of non-Diophantine frequencies vs. $\gamma$.
- **S4.3 (KAM theorem; standard map):**
  - **MD:** KAM theorem statement (twist + Diophantine), Cantor structure of surviving tori, Moser's twist-map case.
  - **NU (flagship):** vectorized Poincaré sections of the Chirikov standard map for $K \in \{0.5,\ 0.9716,\ 1.5,\ 3,\ 5,\ 10\}$: KAM tori → last (golden-mean) torus → chaotic sea with island chains.
  - **NU (Greene):** residue criterion — for a range of $K$, compute the residue of the period-$q$ orbit near the golden mean ($q$ = Fibonacci numbers, renormalized), locate the $K$ where the residue diverges; compare the estimate with $K_c \approx 0.9716$ (essay) and the naive overlap value $\pi^2/4 \approx 2.467$ (shown to *overestimate*, as the essay states).
  - **FIG-11:** 2×3 Poincaré-section panel (flagship figure). **FIG-12:** Greene residue vs. $K$ with the $K_c$ marker.
- **S4.4 (Arnold diffusion, Nekhoroshev):**
  - **NU:** standard map at $K = 5$: track the action $I_n$ of a chaotic orbit; show order-one drift across the chaotic sea over $10^4$–$10^5$ iterations; MD compares the observed drift time with the Nekhoroshev scale $\exp(c/\varepsilon^a)$ qualitatively (no rigorous bound claimed — essay's scope remark).
  - **FIG-13:** action drift plot.
- **S4.5 (symplectic constraint on hyperbolicity):**
  - **SY-10:** cat map with $\det A = 1$: verify $A^T J A = J$ symbolically ($J = \begin{pmatrix}0&1\\-1&0\end{pmatrix}$) — area-preserving 2-D = symplectic.
  - **NU:** symplectic duality: with $E^s, E^u$ the eigenspaces, verify numerically that $E^u = (E^s)^{\omega\perp}$ (the symplectic-orthogonal of $E^s$ is $E^u$) and that $\omega$ vanishes on each (1-D, automatic, stated) — the essay's Lagrangian/dual-pair structure in its simplest instance.

### S5 — Physical applications (essay §5)
- **S5.1 (Lorenz system):**
  - **SY-11:** Jacobian at the origin; characteristic polynomial $(\lambda + \beta)(\lambda^2 + (\sigma+1)\lambda + \sigma(1-\rho))$; at $\sigma=10, \beta=8/3, \rho=28$: eigenvalues $\frac{-11 \pm \sqrt{1201}}{2} \approx 11.83, -22.83$ and $-\beta \approx -2.667$ — reproduces the essay's linearization exactly.
  - **SY-12:** divergence $\nabla\cdot f = -(\sigma + 1 + \beta) = -41/3 \approx -13.667$ (dissipation, symbolic).
  - **NU:** integrate with `scipy.integrate.solve_ivp` (RK45, tight tolerances) → attractor (3-D + $x$–$y$ projection); Benettin spectrum $\approx (0.906, 0, -14.57)$; $D_{LY} \approx 2.06$; homoclinic tangle: grow $W^s, W^u$ of the origin (restricted to the attractor's Poincaré-section-like plane) and plot; non-uniform hyperbolicity: plot the local stable contraction rate vs. distance to the origin (degenerates toward $-\beta$, the weak stable direction) — the singular-hyperbolicity mechanism of the essay.
  - **FIG-14:** Lorenz attractor (3-D + projection). **FIG-15:** homoclinic tangle at the origin.
- **S5.2 (Hénon map):**
  - (fixed point & multipliers: SY-4/SY-5 from S2.1, cross-referenced)
  - **NU:** attractor (200k iterates, discard transient); Benettin spectrum $\approx (0.490, -1.694)$ with $\lambda_1 + \lambda_2 = \log 0.3$; box/correlation/information dimensions (estimators from S0); the essay's geometric structure: histogram of $x$ (fills an interval $\approx[-1.28, 1.27]$) and vertical slices at fixed $x$ (Cantor-like $y$-sets) — both plotted.
  - **FIG-16:** Hénon attractor + $x$-histogram + three vertical slices (3-panel).
- **S5.3 (strange attractors, Ruelle–Takens):**
  - **MD:** strange attractor definition; Ruelle–Takens hypothesis; Newhouse–Ruelle–Takens theorem.
  - **SY-13:** Hopf bifurcation normal form (symbolic derivation sketch: eigenvalues crossing the imaginary axis, center-manifold reduction to $\dot z = (\mu + i\omega)z - c|z|^2 z$).
  - **NU:** 4-D normal-form ODE implementing three successive Hopf bifurcations: fixed point → limit cycle → $\mathbb T^2$ → $\mathbb T^3$/strange, as a parameter $\mu$ is increased; 2×2 phase portraits.
  - **FIG-17:** Ruelle–Takens sequence (2×2 panel).
- **S5.4 (Navier–Stokes, finite-dimensional attractors):**
  - **SY-14:** 2-D stream-function identity: $u = (\partial_y \psi, -\partial_x \psi) \Rightarrow \nabla\cdot u = 0$ (symbolic).
  - **NU:** 2-D NS on the torus in vorticity form with time-periodic Taylor–Green-type forcing; Galerkin truncation to Fourier modes $|k| \le N$ (triad-interaction ODE, standard). With $N \approx 10$–$12$: (i) long-time attractor, projected to the first few Fourier coefficients (phase portrait); (ii) Benettin on the truncated ODE for the top $m \approx 20$ exponents → Kaplan–Yorke dimension; (iii) convergence: $D_{LY}$ vs. $N$ (stabilizing); (iv) a simple Foias–Temam-type upper bound in terms of the effective Reynolds number, displayed against the computed $D_{LY}$.
  - **MD:** the Hopf hypothesis in 2-D (theorem) vs. the open 3-D problem (Millennium); Galerkin convergence remark.
  - **FIG-18:** NS attractor projection + $D_{LY}(N)$ convergence panel.
  - *This is the heaviest cell (see R1); parameters are tuned in Phase 2 to keep total runtime bounded.*
- **S5.5 (Lyapunov exponents as a physical diagnostic):**
  - **NU:** two-particle experiment on Lorenz: for several initial separations $\delta_0 \in 10^{-8}\ldots10^{-2}$, measure the time $t_{pred}$ to $O(1)$ separation; log-log fit of $t_{pred}$ vs. $\log(1/\delta_0)$ → slope $\approx 1/\lambda_1$ (with $\lambda_1$ from S5.1) — the butterfly-effect content, quantified.
  - **FIG-19:** predictability-horizon log-log fit.

### S6 — Numerical analysis (essay §6)
- **S6.1 (Benettin algorithm):**
  - **MD:** the algorithm as in the essay (evolve, re-orthogonalize, average logs); the key role of Gram–Schmidt.
  - **NU:** the `benettin_lyapunov` implementation (already exercised in S2–S5) presented here with the full code + a contrast run **without** re-orthogonalization: all tangent vectors collapse onto the maximal-expansion direction (only $\lambda_1$ recoverable) — the essay's key numerical issue, demonstrated.
  - **FIG-20:** with vs. without re-orthogonalization (vector alignment + recovered exponents).
- **S6.2 (structure-preserving integration):**
  - **NU:** a nonlinear 2-DOF Hamiltonian (e.g. $H = \tfrac{p_1^2}{2} + \tfrac{p_2^2}{2} + \tfrac{q_1^4}{4} + \tfrac{q_2^4}{4} + \lambda q_1^2 q_2^2$): long-time integration ($T \approx 10^4$) with RK4 vs. the Störmer–Verlet (symplectic) integrator at equal step size; plot $H(t) - H(0)$: systematic RK4 drift vs. bounded symplectic oscillation (modified-Hamiltonian behavior, as in the essay).
  - **SY-15/NU:** the standard map is exactly symplectic: verify $J_f^T \Omega J_f = \Omega$ for the map Jacobian (symbolically in $(\theta, I)$, and numerically along an orbit).
  - **FIG-21:** energy drift RK4 vs. symplectic (log scale, long time).
- **S6.3 (shadowing and numerical reliability):**
  - **MD:** the essay's precise scope remark: the lemma is a theorem for *uniformly* hyperbolic systems; Lorenz/Hénon are not Axiom A.
  - **NU:** contrast: (i) cat map — shadowing distance stays bounded (recap of S2.3, tighter $\varepsilon$); (ii) Hénon attractor — the best-shadowing distance of a noisy pseudo-orbit grows with the orbit window (plot the window-size vs. min-distance curve) — the scope boundary made visible.
  - **FIG-22:** shadowing distance vs. window length (cat map flat, Hénon growing).
- **S6.4 (dimension estimation):**
  - **NU:** box-counting, correlation (Grassberger–Procaccia), and information dimensions of the Hénon attractor (and Lorenz, as a second example); log-log plots with the fitted slopes; **final comparison table**: computed $D_B, D_2, D_1, D_{LY}$ vs. essay-quoted $D_B \approx 1.26$, $D_2 \approx 1.21$, $D_{LY} \approx 1.26$ (Hénon) and $D_{LY} \approx 2.06$ (Lorenz), with a note on estimator sensitivity (R2).
  - **FIG-23:** $N(\varepsilon)$ vs. $\varepsilon$ log-log (Hénon) + summary table.

### S7 — Open problems (essay §7)
- **MD only:** the six frontiers (stable ergodicity & rigidity, Arnold diffusion, NS regularity, partial hyperbolicity & blenders, quantum interface, infinite-dimensional systems), each with a one-paragraph cross-reference to the essay and a pointer to which notebook cells illustrate the *computable* part of each.

### S8 — Summary
- **MD + NU:** the full verification table (Section 7 of this plan) as a printed table: every computed constant next to its essay-quoted value, with pass/flag marks.

## 7. Expected constants (verification table)

The verification script checks these (whitespace-normalized string match on printed output):

| # | Quantity | System | Expected | Essay source |
|---|---|---|---|---|
| 1 | Eigenvalues at origin (exact) | Lorenz | $\frac{-11\pm\sqrt{1201}}{2} \approx 11.83, -22.83$; $-8/3$ | §5.1 |
| 2 | Divergence (exact) | Lorenz | $-(\sigma+1+\beta) = -41/3 \approx -13.667$ | §5.1 |
| 3 | Lyapunov spectrum (numeric) | Lorenz | $\approx (0.906,\ 0,\ -14.57)$ | §5.1, §5.5 |
| 4 | $D_{LY}$ (numeric) | Lorenz | $\approx 2.06$ | §5.1 |
| 5 | Fixed point (exact) | Hénon | $x_* = -\tfrac14 + \tfrac{\sqrt{609}}{28} \approx 0.63135$, $y_* = \tfrac{3\sqrt{609}}{280} - \tfrac{3}{40} \approx 0.18941$ | §5.2 |
| 6 | Multipliers (numeric, from exact eig) | Hénon | $\approx (-1.924,\ +0.156)$ | §5.2 |
| 7 | $\det DH$ | Hénon | $-b = -0.3$ | §5.2 |
| 8 | Lyapunov spectrum (numeric) | Hénon | $\approx (0.490,\ -1.694)$; sum $= \log 0.3$ | §5.2 |
| 9 | $D_{LY}$ (numeric) | Hénon | $\approx 1.29$ — essay quotes $\approx 1.26$ (flag, see R2) | §3.6, §6.4 |
| 10 | Box / correlation dimension (numeric) | Hénon | $\approx 1.26$ / $\approx 1.21$ | §5.2, §6.4 |
| 11 | Eigenvalues (exact) | Cat map | $\varphi^2 \approx 2.618$, $\varphi^{-2} \approx 0.382$ | §2.2 |
| 12 | $h_{\mathrm{Leb}}$ (numeric) | Cat map | $\log\varphi^2 \approx 0.9624 = \sum_{\lambda_i>0}\lambda_i$ | §3.3–3.4 |
| 13 | $\sum_i \lambda_i$ (numeric) | Cat map | $0 = \log|\det A|$ (Jacobian identity) | §3.2 |
| 14 | Shadowing distance (numeric) | Cat map | bounded $\delta \ll 1$ for $\varepsilon$-pseudo-orbits | §2.3 |
| 15 | Lyapunov exponent (numeric) | Geodesic flow, $K=-1$ | $1 = \sqrt{|K|}$ | §2.5 |
| 16 | Naive resonance-overlap $K_c$ | Standard map | $\pi^2/4 \approx 2.467$ (overestimate) | §4.4 |
| 17 | Greene $K_c$ (numeric) | Standard map | $\approx 0.9716$ | §4.4 |
| 18 | $P(0)$ (numeric) | Full 2-shift | $\log 2 \approx 0.6931$ | §3.5 |
| 19 | Symplectic condition (symbolic) | Cat map | $A^T J A = J$ | §4.5 |
| 20 | Hopf fiber invariance (symbolic) | Hopf fibration | $h(e^{i\theta}z) = h(z)$ | §2.4 |
| 21 | Poisson bracket (symbolic) | Separable 2-DOF | $\{E_1, E_2\} = 0$ | §4.1 |
| 22 | Jacobi fields (symbolic) | $K=-1$ surface | $v(t) = c_1 e^t + c_2 e^{-t}$ | §2.5 |
| 23 | $t_{pred}$ slope (numeric) | Lorenz | $\approx 1/\lambda_1$ (log-log fit) | §5.5 |

## 8. Implementation phases and acceptance criteria

**Phase 0 — Environment (DONE).** All packages verified present in the venv (Section 2). No action.

**Phase 1 — Builder script.**
Write `build_dynamic_systems_nb.py`: a cell list (markdown with LaTeX + code) assembled per Section 6, serialized via `nbformat.from_dict` → `nbformat.validate` → `nbformat.write`.
*Acceptance:* valid `.ipynb` written; all cells have unique `id`s + `metadata` (nbformat 5.x); first code cell contains `%matplotlib inline` (never `matplotlib.use("Agg")` — that would suppress inline PNG capture); builder re-runnable (idempotent overwrite of the notebook).

**Phase 2 — Headless execution.**
Execute with `nbclient.NotebookClient` using a temp kernel spec whose `argv` is `[D:/Source/hermes-dir/.venv/Scripts/python.exe, "-m", "ipykernel_launcher", "-f", "{connection_file}"]` and `JUPYTER_DATA_DIR` pointed at a temp dir. Run as a **tracked background process** (expected total runtime 15–40 min, dominated by the NS Galerkin cell and the long-time integrator comparison); iterate on failures cell-by-cell.
*Acceptance:* execution completes; every code cell has no `error` output; no cell exceeds a sane wall-time cap (flag and shrink parameters if so).

**Phase 3 — Verification (independent of execution).**
`verify_dynamic_systems_nb.py` re-reads the saved `.ipynb` and checks:
1. `nbformat.validate` passes;
2. zero `output_type == "error"` across all code cells;
3. `image/png` output count equals the expected figure count (23), counting **every** display mechanism (`plt.show()`, bare `Figure` expressions);
4. every constant in the Section-7 table appears in printed output (whitespace-normalized match) within its tolerance.
A failed check is re-run against the saved file before any re-execution (a buggy check is the usual cause of false failures).
*Acceptance:* all four checks pass.

**Phase 4 — Delivery.**
Deliver `MEDIA:D:\Source\hermes-dir\DynamicSystems\Dynamic_Systems_v2_notebook.ipynb` with a short report: figure count, table of computed vs. essay-quoted constants, any flagged discrepancies. Offer (do not perform without confirmation) the optional user-level kernel-spec registration for direct JupyterLab use.

## 9. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Runtime:** the 2-D NS Galerkin cell (ODE system of a few hundred modes + Benettin on it) and the $T\approx10^4$ integrator comparison are the heaviest cells | Vectorize everything possible; limit NS to $N \le 12$ and top $m \le 20$ tangent vectors; run execution in background with a long timeout; if a cell is still too slow, reduce parameters and note the reduction in the cell's markdown |
| R2 | **$D_{LY}$ discrepancy for Hénon:** the essay quotes $D_{LY} \approx 1.26$, but $D_{LY} = 1 + \lambda_1/|\lambda_2| \approx 1.29$ with $\lambda \approx (0.490, -1.694)$ (sum constrained to $\log 0.3$) | Do **not** alter the essay; the notebook reports the computed value, shows the arithmetic, and flags the difference as estimator/literature sensitivity (the box dimension $\approx 1.26$ is the value the Kaplan–Yorke conjecture identifies with $D_{LY}$) |
| R3 | **Shadowing on Hénon is scope-bound:** the lemma does not apply; the "growing distance" plot can be noisy | Use a robust metric (max distance over the window, min over initial conditions), fixed seed, and the essay's own scope caveat in the markdown |
| R4 | **Greene residue criterion is heuristic:** threshold for "divergence" is not unique | Present $K_c$ as an estimate, compare against the literature value $0.9716$, and state the method's status (the essay already attributes it to Greene, 1979) |
| R5 | **Dimension-estimator finite-size effects:** $D_B, D_2$ depend on the $\varepsilon$-range and sample size | Log-log slope fits over a documented range, large samples (≥200k points), fixed seed; report the range used |
| R6 | **Builder pitfalls (known):** nested quotes in cell strings, missing cell ids, Agg backend | Outer cell delimiters `r'''...'''`; ids stamped by the cell helpers; `%matplotlib inline` in the first code cell only |
| R7 | **Windows paths for native tools** | Native forward-slash paths (`D:/...`) everywhere; MSYS paths only for bash builtins |

## 10. What is deliberately *not* computed (and why)

- **Hopf invariant $= 1$** — a homotopy-theoretic fact; the notebook verifies the checkable consequences (fiber invariance, isometric Hopf flow) instead.
- **Pugh–Shub conjecture, local/global rigidity, blenders, generic Arnold diffusion, 3-D NS regularity, quantum interface** (essay §7) — open problems; markdown cross-references only, with pointers to the notebook cells that cover their computable shadows (e.g. the standard-map diffusion cell for Arnold diffusion, the NS Galerkin cell for attractor structure).
- **JSJ decomposition / foliation regularity theory** (essay §2.5–2.6) — deep 3-manifold/foliation geometry; the notebook covers the computable model case (toral automorphism foliation + holonomy).
- **Exact Markov partitions** (Bowen) — construction is non-constructive in general; the notebook uses the SFT side of the bridge (2-shift pressure/Gibbs) instead.

---

*Plan status: ready for Phase 1 (builder script) upon approval. The v2 essay and all its saved versions remain untouched throughout.*
