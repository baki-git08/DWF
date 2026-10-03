# Instructions: replacing the theoretical slave battery with the physical (photodiode+MPPT) one

> **Note (code layout):** `build_2D_dwf.py` has since been split into the `dwf2d/` package (see README section 2,
> "Code layout"). Line numbers of the form `build_2D_dwf.py:NNN` below refer to the old single file; the
> functions keep their names and are still importable from `build_2D_dwf`.

**How to use this file:** paste/point Claude Code at this file and say "implement Option B" or
"implement Option C" from `Instructions.md`. Each option is self-contained: math, file-by-file
changes, pseudocode, and a verification checklist. Do not implement both at once — read the
"Decision guide" at the end first if unsure which one you want.

---

## 0. Background (read this first, both options depend on it)

### 0.1 Files involved
- `advanced_DWF/build_2D_dwf.py` — the DWF solver (`joint_dwf`), its data classes
  (`JointParams`, `JointResult`), the physical-chain post-solve layer added previously
  (`PhysicalStorageResult`, `physical_stored_power()`), and all plots.
- `advanced_DWF/physical_chain.py` — receive-chain physics: `photodiode_stage()`,
  `mppt_stage()`, `optical_to_electrical()`. **Do not need to change this file for either option**
  — its formulas are correct; the question is only how `build_2D_dwf.py` *uses* them.
- `advanced_DWF/dwf_diagnostics.py` — human-readable report (`diagnose()`); currently assumes
  `jp.beta` is a scalar in a couple of print lines.
- `advanced_DWF/build_dwf.py` — master-only harvest data/`MasterParams` (upstream, untouched by
  either option).

### 0.2 Current model (the thing both options replace)
`joint_dwf()` treats the master→slave energy tap as **linear**:

```
a_i = beta * alpha_i                      # JointParams.a property, build_2D_dwf.py:~246
xs(i) = a_i * d[i] + g[i] - g[i+1]         # slave data energy, build_2D_dwf.py:343
```

i.e. "one Joule sent (`delta_i`) always yields `a_i` Joules in the slave battery," where `beta`
(recharge efficiency) is currently a single placeholder scalar (0.8–0.9 in the demo).

### 0.3 What the real chain actually does (`physical_chain.py`)
```
P_opt   = alpha_i * delta_i / tau                  # optical power arriving at the photodiode [W]
I       = R * P_opt                                # photocurrent, R = responsivity [A/W]
Voc     = V_T * ln(1 + I / I_dark)                 # open-circuit voltage
P_elec  = FF * Voc * I                              # photodiode electrical power
P_mppt  = eta_mppt * P_elec                          # = physical_chain.mppt_output_power
```
With the DP/SB photodiode (`R=0.32 A/W`, `I_dark=10 nA`), `FF=0.7`, `eta_mppt=0.95`,
`V_T≈0.025852 V` at 300 K, the measured effective gain in the demo scenario
(`optical_to_electrical` fed `r.arrived_power`) is:

```
eff_phys = P_mppt / P_opt  ≈ 0.102 – 0.106      (vs. the assumed beta = 0.9)
```

This ratio is **not exactly constant** — it drifts slowly with `P_opt` because of the `ln(1+I/I_dark)`
term — which is precisely why the two options below exist: Option B treats it as
*locally constant per slot* (a calibrated secant gain), Option C treats it as the true nonlinear
function inside the optimizer.

### 0.4 Existing building blocks to reuse (do not duplicate)
- `physical_stored_power(jp, r, wavelength_nm, mppt_efficiency, fill_factor, temp_K) -> PhysicalStorageResult`
  (build_2D_dwf.py) — already computes `stored_power_phys`, `eff_phys`, `b_phys`, `b_peak_phys`,
  `floor_violations`, `ceiling_violations` for any solved `JointResult`. Both options call this.
- `optical_to_electrical(...)` (physical_chain.py) — the underlying chain function; Option C's
  math is this function symbolically differentiated (done for you in section 2.1).
- `joint_dwf(jp)`, `check_kkt_joint(jp, r)`, `reference_solution_joint(jp)` — the existing solver
  and its two independent correctness checks. Option B does not touch them. Option C's cvxpy path
  reuses the same constraint structure as `reference_solution_joint`.

---

## 1. OPTION B — Calibrated per-slot beta (fixed-point loop around the unmodified solver)

**Effort: ~60–80 lines across 2 files. No changes to `joint_dwf`, `step`, `apply`, the 9 moves,
`polish`, or `check_kkt_joint`. Recommended default.**

### 1.1 Idea
`beta` becomes a **per-slot array** instead of a scalar. Solve once with an initial guess, measure
the real secant gain `eff_phys_i = stored_power_phys_i / arrived_power_i` the chain delivered for
that solution's `delta_i`, feed it back in as the new `beta_i`, and repeat. Since `eff_phys_i` here
only ranges 0.102–0.106 (it barely depends on the operating point), this fixed point is reached in
a handful of iterations. At convergence, `a_i*delta_i/tau == mppt_output_power_i` exactly, so the
solver's battery (`r.b`, `r.b_peak`) **equals** the physically re-simulated one (`ph.b_phys`,
`ph.b_peak_phys`), and the existing `B_min`/`B_max` constraints (S1/S2) are now enforced against
real delivered energy.

### 1.2 Math
Fixed point equation, per slot `i` with `delta_i > 0`:
```
beta_i^{(n+1)} = g(alpha_i * delta_i^{(n)} / tau) / (alpha_i * delta_i^{(n)} / tau)
```
where `g(P) = eta_mppt * FF * V_T * ln(1 + R*P/I_dark) * R*P` is `optical_to_electrical`'s
P_opt→P_mppt map. Slots with `delta_i == 0` have no signal to calibrate on; keep their previous
`beta_i` (or the initial guess) unchanged — they don't affect the battery since no energy moves.

This converges because `g(P)/P` (the secant slope) changes slowly with `P` here (it's
`eta*FF*V_T*R*ln(1+kP)`, and `ln` is a slowly-varying function) — i.e. the fixed-point map is a
contraction in this regime. Verify this empirically in your run (see 1.5); if it ever oscillates,
under-relax: `beta_i^{(n+1)} = 0.5*beta_i^{(n)} + 0.5*(new estimate)`.

### 1.3 Implementation

**(a) `JointParams` (build_2D_dwf.py:~191-259): allow `beta` to be per-slot.**
```python
beta: np.ndarray | float = 0.8   # was float = 0.8 — no default value change needed
```
In `__post_init__` (build_2D_dwf.py:220), broadcast it exactly like `alpha` is already broadcast:
```python
self.beta = np.broadcast_to(np.asarray(self.beta, float), (n,)).copy()
```
Change the validation line (build_2D_dwf.py:235) from
`not (0 <= self.beta <= 1)` to `np.any((self.beta < 0) | (self.beta > 1))`, matching the existing
`alpha` check style just above it. **Nothing else in `joint_dwf` needs to change** — `jp.a` already
does `self.beta * self.alpha`, which is elementwise whether `beta` is scalar-broadcast or now a
real per-slot array; `step()`/`apply()` already treat `a` as `a[k]`, a per-slot array, everywhere.

**(b) New function in build_2D_dwf.py, in the "Verification helpers" section (near
`physical_stored_power`):**
```python
import dataclasses

def solve_with_physical_chain(
    jp: JointParams,
    wavelength_nm: float = 450,
    mppt_efficiency: float = 0.95,
    fill_factor: float = 0.7,
    temp_K: float = 300.0,
    tol: float = 1e-6,
    max_outer: int = 20,
):
    """Fixed-point calibration: re-solve joint_dwf() with beta_i = the chain's real secant gain
    until beta stops changing. joint_dwf() itself is never modified -- see Instructions.md Option B.

    Returns (jp_calibrated, r, ph, n_outer):
        jp_calibrated : JointParams with the converged per-slot beta
        r             : final JointResult (r.b, r.b_peak now equal the physical battery)
        ph            : final PhysicalStorageResult (ph.b_phys should match r.b to `tol`)
        n_outer       : number of outer iterations used
    """
    beta = np.broadcast_to(np.asarray(jp.beta, float), (jp.master.N,)).copy()
    for n_outer in range(1, max_outer + 1):
        jp_i = dataclasses.replace(jp, beta=beta.copy())
        r = joint_dwf(jp_i)
        ph = physical_stored_power(jp_i, r, wavelength_nm, mppt_efficiency, fill_factor, temp_K)
        on = r.delta > 1e-9 * max(jp.master.E.mean(), 1.0)          # only calibrate where energy actually moves
        beta_new = beta.copy()
        beta_new[on] = ph.eff_phys[on]
        moved = np.max(np.abs(beta_new - beta)) if np.any(on) else 0.0
        beta = beta_new
        if moved < tol:
            break
    return jp_i, r, ph, n_outer
```
Note: `ph.eff_phys` is already NaN-safe for `arrived_power == 0` (build_2D_dwf.py's
`physical_stored_power`); the `on` mask additionally protects against calibrating on numerically
tiny `delta_i`.

**(c) `physical_stored_power()` — add battery ceiling clipping.** Currently (build_2D_dwf.py, the
`for i in range(m.N): level = level + stored_energy_phys[i] - tau_pbar[i]` loop) has no `B_max`
cap, so if calibration ever overshoots mid-loop the level can run away unrealistically (this was
observed when responsivity was increased ad hoc). Add:
```python
overflow = np.zeros(m.N)
...
for i in range(m.N):
    b_prev_phys[i] = level
    level = level + stored_energy_phys[i] - tau_pbar[i]
    if level > jp.B_max:
        overflow[i] = level - jp.B_max        # energy the real battery would have to reject
        level = jp.B_max
    b_phys[i] = level
```
Add `overflow: np.ndarray` and `overflow_energy: float` (`= overflow.sum()`) to
`PhysicalStorageResult`. At the converged fixed point this should be ~0; it stays informative if
someone calls `physical_stored_power` directly with an uncalibrated `jp` (e.g. Option A style).

**(d) `dwf_diagnostics.py`** — only touch the scalar-only print lines (around line 253):
```python
print(f"  beta  (recharge eff.)  : {beta:.4g}          -> {-float(_dB(beta)):.3g} dB recharge loss")
```
becomes (mirroring the existing `_rng(...)` helper already used one line below for `a`):
```python
print(f"  beta  (recharge eff.)  : {_rng(beta)}          -> {-float(_dB(np.mean(beta))):.3g} dB recharge loss (mean)")
```
Everything else in that file already indexes `beta`/`a` as arrays or reduces with `.mean()`, so no
further changes needed there — double check by re-running `diagnose()` (verification step below)
rather than grepping further.

**(e) Demo (`__main__`, build_2D_dwf.py bottom):** replace
```python
r = joint_dwf(jp)
```
with
```python
jp, r, ph, n_outer = solve_with_physical_chain(jp, wavelength_nm=450, mppt_efficiency=0.95)
print(f"calibration outer iters = {n_outer}, beta range = [{jp.beta.min():.4f}, {jp.beta.max():.4f}]")
print(f"max |solver battery - physical battery| = {np.max(np.abs(r.b - ph.b_phys)):.3e} J")
```
keep the rest of the demo (`diagnose`, `check_kkt_joint`, all plots) unchanged — `jp` now carries
the calibrated per-slot `beta`, so every downstream plot/print (`plot_link_power`,
`plot_slave_battery`, `plot_physical_stored_power`, `plot_physical_slave_battery`, etc.) picks it
up automatically since they all read from `jp`/`r`/`ph`. `plot_physical_slave_battery` becomes a
*consistency check* — the two curves should now sit on top of each other.

### 1.4 Known limitation (document, don't try to fix under Option B)
The solver equalizes marginal *secant* gains (`m_k = a_k * s_k` using `a_k = g(P)/P`), but the true
physical marginal is `g'(P)`. Here:
```
g'(P) / (g(P)/P) = 1 + 1 / ln(1 + R*P/I_dark)  ≈ 1.06   (a ~6% gap, roughly constant in this regime)
```
So Option B gives an **exact, feasible battery** but a delta-split that is ~6% off the true
optimum. If that residual matters, that's what Option C closes.

### 1.5 Verification
- `python build_2D_dwf.py`: `n_outer` converges within a handful of iterations (expect < 10); final
  printed beta range should sit close to the previously observed `eff_phys` range (0.102–0.106).
- `max |r.b - ph.b_phys|` should be ~1e-6 J or less (floating-point noise only).
- `check_kkt_joint(jp, r)`: `slave_min_level >= B_min` and `slave_max_level <= B_max` (previously,
  before calibration, `slave_min_level` was deeply negative-going / `floor_violations = 60`).
- `ph.floor_violations == 0` and `ph.ceiling_violations == 0` (or `ph.overflow_energy ≈ 0`) after
  calibration.
- Sanity regression: call `joint_dwf(jp)` directly with a *scalar* `beta` (no calibration loop) and
  confirm output is bit-identical to before this change — proves `JointParams` broadcasting didn't
  alter solver behavior for the old call pattern.
- Expect the slave to receive/share much less than the old `beta=0.9` runs showed (~9x less, since
  0.10/0.9 ≈ 1/9) — this is physically correct, not a regression.

---

## 2. OPTION C — True nonlinear photodiode response inside the optimizer

**Effort: multi-day. Requires re-deriving math and re-validating against an independent solver.
Read the convexity result in 2.1 before starting — it changes what "correct" even means here.**

### 2.1 The math: why this is hard (not just "more code")

Define the stored-energy-per-slot as a function of `delta_i` (energy the master sends, Joules),
folding in the chain from `physical_chain.py`:
```
I(delta)  = s_i * delta,                s_i = R * alpha_i / tau            (photocurrent slope)
S_i(delta) = D_i * delta * ln(1 + k_i*delta),   with
             k_i = s_i / I_dark  = R*alpha_i / (tau * I_dark)
             D_i = tau * eta_mppt * FF * V_T * s_i = eta_mppt*FF*V_T*R*alpha_i
```
(`S_i(delta)` replaces `a_i * delta` in the energy balance `xs(i) = S_i(d[i]) + g[i] - g[i+1]`.)

Differentiate twice:
```
S_i'(delta)  = D_i [ ln(1+k_i*delta) + k_i*delta/(1+k_i*delta) ]
S_i''(delta) = D_i * k_i * [ 1/(1+k_i*delta) + 1/(1+k_i*delta)^2 ]   > 0   for all delta >= 0
```
**`S_i` is convex (accelerating), not concave**, over its entire domain. This is the crux of why
Option C is qualitatively different from just "swap the formula in":

- The battery floor constraint (S1) is `b_n = b0 + sum_{i<=n}(S_i(delta_i) - xs_i) >= B_min`. With
  `S_i` convex, `b_n` (as a function of the `delta` decision variables) is convex, so `b_n >= B_min`
  is a **superlevel set of a convex function — generically non-convex** (think `x^2 >= 1`, a
  disconnected feasible region). The overall problem is **no longer a convex program.**
- Consequence 1: `reference_solution_joint()`'s cvxpy formulation (build_2D_dwf.py:726-763) cannot
  be used as-is — cvxpy's DCP ruleset will reject (or silently mis-handle) a convex expression
  required to be `>=` a constant inside a maximization. You cannot just paste `S_i(delta)` into
  that `cons` list.
- Consequence 2: `joint_dwf()`'s block-coordinate-ascent proof of correctness ("a move can never
  reduce the objective," module docstring section 5) relied on every constraint being *linear* in
  the flow variables and the objective being *concave* — i.e. a genuinely convex program, where
  local optimality (KKT) implies global optimality. With a non-convex feasible region, the same
  coordinate-ascent moves still won't *decrease* the objective (each 1-D move along `t` can still
  be solved/clipped correctly for that subproblem), but the algorithm can now **stop at a local
  optimum that isn't global**, and there is no cheap independent certificate (no more trustworthy
  cvxpy cross-check) the way `check_kkt_joint`/`reference_solution_joint` provide today.

This doesn't mean Option C is wrong to attempt — it means "implement it" necessarily includes
"decide how to live without a global-optimality guarantee," which is a real design choice, not a
coding detail.

### 2.2 Two viable implementation paths

#### Path C1 (recommended if you pursue C): Successive Convex Programming (SCP / DC programming) via cvxpy
Exploit that `S_i` is convex and smooth: at each outer iteration, replace `S_i(delta)` by its
**tangent-line (concave) upper bound is wrong direction — use the tangent as a *linear* stand-in**,
i.e. linearize around the previous iterate `delta^{(n)}`:
```
S_i(delta) ≈ S_i(delta^{(n)}) + S_i'(delta^{(n)}) * (delta - delta^{(n)})
```
This linear approximation turns the battery constraints back into **linear** (hence convex)
constraints for that iteration, solvable exactly by `reference_solution_joint`'s existing cvxpy
machinery (or a lightly modified copy of it). Iterate: solve the linearized convex QP/log-problem,
re-linearize `S_i` around the new `delta`, repeat until `delta` stops moving. This is the standard
"convex-concave procedure" / DC-programming pattern for a convex-in-a-concave-constraint problem.

Pseudocode (new function, e.g. `solve_nonlinear_chain_scp(jp, wavelength_nm, mppt_efficiency, ..., max_outer=30, tol=1e-6)`):
```
delta0 = jp.a * initial_guess (e.g. Option B's converged delta, or zeros)
for outer in range(max_outer):
    slope_i  = S_i'(delta0_i)     for each slot i     # closed form above, or np.gradient as a check
    intercept_i = S_i(delta0_i) - slope_i * delta0_i
    # Solve the SAME cvxpy problem as reference_solution_joint(), but replace
    #   jp.b0 + cum(multiply(a, dl) - xb) >= B_min
    # with
    #   jp.b0 + cum(multiply(slope, dl) + intercept - xb) >= B_min
    # (and similarly for the S2 ceiling term), i.e. an affine-in-dl surrogate for S_i(dl).
    delta1 = result.dl
    if max(abs(delta1 - delta0)) < tol: break
    delta0 = delta1
```
Properties: each outer iterate is a small, standard convex program (cvxpy/CLARABEL), so this is
comparatively easy to get *running*; the hard part is that DC/SCP procedures converge to a
**local** optimum and are sensitive to the starting point — run it from a couple of different
initializations (e.g. `delta=0` and Option B's converged `delta`) and keep the better objective.

Add a dedicated cross-check: after convergence, verify the *exact* (non-linearized) constraint
`S_i(delta)` is still satisfied (it should be, since you always re-linearize at the current point,
but confirm with the true `physical_chain.optical_to_electrical()` call, not the tangent
approximation) — this reuses `physical_stored_power()` exactly as Option B does.

**Effort**: a new solve function (~100–150 lines: the linearization bookkeeping + a modified copy
of `reference_solution_joint`'s cvxpy problem), plus a validation harness comparing against Option
B's objective (Option C should never do *worse* than Option B, since Option B's solution — with the
real chain plugged in exactly, not linearized — is a feasible point for the true problem too; if
C1's result is worse, something is implemented incorrectly).

#### Path C2: Modify `joint_dwf()`'s moves to solve a 1-D nonlinear equation per step
Keep the existing sweep/move structure (moves (1)-(9), `polish()`), but replace the closed-form
`step()` (build_2D_dwf.py:348-379, currently linear-in-`t` hence exactly solvable) with a 1-D
Newton/bisection solve of the true marginal-balance condition for the down-tap moves (move (1) and
its swap/pass-through variants (2)-(5)) — e.g. for move (1):
```
m_k(t) = th1*h_k / (2*(1 + h_k*(xm(k)-t)/tau))
s_k(t) = th2*hb_k / (2*(1 + hb_k*(xs(k) + [S_k(d_k+t) - S_k(d_k)])/tau))
solve  m_k(t) = S_k'(d_k+t) * s_k(t)     for t      (Newton, a handful of iterations; this
                                                      replaces the current closed-form t*)
```
then clip `t` to `[lo, hi]` exactly as `apply()` already does (that clipping logic doesn't change —
only how `t*` itself is computed changes for down-tap moves; the pure right-tap moves (6)/(7)
between same-type nodes are unaffected since they don't involve `S_i` at all).

This preserves more of the existing code's structure and avoids a full outer SCP loop, but: (a) it
must be done for every down-tap move variant ((1)-(5) and the cross-paths inside `polish()`) — more
places to touch than C1; (b) because the feasible region is non-convex (2.1), coordinate ascent can
still land in a bad local optimum with no cross-check available (no cvxpy DCP formulation to
compare against, unlike C1 which still uses cvxpy as its inner solver); (c) `check_kkt_joint()`
would need a new nonlinear version of its down-tap gap check (replace `a_k` with `S_k'(delta_k)` in
the `gap` computation, build_2D_dwf.py:706).

**Effort**: larger than C1 (touches 5+ move blocks plus `polish()`'s cross-paths, ~200+ lines) and
gives up the independent cvxpy cross-check entirely. **Not recommended unless you specifically need
to preserve the exact sweep/move architecture** (e.g. for consistency with a paper's algorithm
description).

### 2.3 Recommendation between C1 and C2
Start with **C1**. It reuses proven convex-solver machinery (cvxpy/CLARABEL, already a dependency
via `reference_solution_joint`), isolates the new nonlinear logic to one function instead of nine
move blocks, and gives you a natural per-iteration feasibility check. Only consider C2 if C1's
runtime (an outer SCP loop, each iteration a full cvxpy solve) is unacceptable for your problem
size, or if you need the result to come from the DWF sweep algorithm specifically.

### 2.4 Verification (either path)
- Compare final objective to Option B's calibrated result: Option C's objective should be `>=`
  Option B's (Option B's solution is a feasible starting point for C, just not locally optimal for
  it) — if it's lower, something is wrong (bad linearization sign, wrong constraint direction, etc).
- Re-run `physical_stored_power()`-equivalent on Option C's final `delta` (i.e. push it through the
  *exact*, non-linearized `optical_to_electrical()`), confirm `B_min`/`B_max` are respected using
  the true chain, not just the tangent surrogate.
- Run from >= 2 different starting points (e.g. zero-sharing and Option B's solution); if the two
  converge to different objectives, you've confirmed the non-convexity in 2.1 is actually biting in
  this scenario, and should keep/report the better one plus flag it (don't silently pick one).
- Sanity check the closed-form derivatives in 2.1 numerically (e.g. `np.gradient` of
  `optical_to_electrical(...).mppt_output_power` against `S_i'(delta)` at a few sample `delta`
  values) before trusting the linearization in C1 or the Newton step in C2.

---

## 3. Decision guide

- Default to **Option B**. It's a small, low-risk change, keeps every existing correctness
  guarantee (`check_kkt_joint`, `reference_solution_joint`) intact, and makes the reported battery
  physically exact. The only thing it doesn't get you is the last ~6% of the delta-sharing
  schedule's optimality.
- Reach for **Option C** only if that ~6% gap is actually material to what you're using the model
  for (e.g. a paper claim about optimality, or a downstream decision sensitive to the exact sharing
  schedule) — and go in accepting you're trading a clean convex-optimization story for a
  non-convex one with no cheap global-optimality certificate.
