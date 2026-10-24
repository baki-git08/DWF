# Critical functions of the joint master + slave DWF solver

This document explains, function by function, how the directional water-filling (DWF) solver works:
what each function computes, the equation behind it, the constraints it respects, and how the
functions call one another.

It covers the three functions named as critical (`step()`, `apply()`, `polish()`), the driver that
joins them (`joint_dwf()`), and every other function the result depends on.

> **Where the code lives.** `build_2D_dwf.py` contains no algorithm code. It is a re-export shim
> (`from dwf2d.solver import joint_dwf, ...`) kept so old imports still work. The real code is in the
> `dwf2d/` package:
>
> | file | contents |
> |---|---|
> | `dwf2d/solver.py` | `joint_dwf()` with nested `step()`, `apply()`, `polish()`; start points; stage-6 post-pass |
> | `dwf2d/params.py` | `JointParams` (inputs and derived arrays), `JointResult` (outputs) |
> | `dwf2d/verification.py` | `check_kkt_joint()`, `reference_solution_joint()` (cvxpy), parity and refusal gates |
> | `dwf2d/physical.py` | photodiode + MPPT reality check, beta calibration, event refusal loop |
>
> Line numbers below refer to `dwf2d/solver.py` unless another file is named. They match the file
> as it is at the time of writing; the file has uncommitted changes, so they may drift.

---

## Contents

1. [The problem in one page](#1-the-problem-in-one-page)
2. [How the pieces fit together](#2-how-the-pieces-fit-together)
3. [`step()`: the one equation](#3-step-the-one-equation)
4. [`apply()`: clip to the feasible interval and move](#4-apply-clip-to-the-feasible-interval-and-move)
5. [`polish()`: long moves through dead slots](#5-polish-long-moves-through-dead-slots)
6. [`joint_dwf()`: the driver](#6-joint_dwf-the-driver)
7. [The ten sweep moves, one by one](#7-the-ten-sweep-moves-one-by-one)
8. [Small helpers inside `joint_dwf()`](#8-small-helpers-inside-joint_dwf)
9. [Start points: making the first point feasible](#9-start-points-making-the-first-point-feasible)
10. [`JointParams` and `JointResult`](#10-jointparams-and-jointresult)
11. [Verification: is the answer actually optimal?](#11-verification-is-the-answer-actually-optimal)
12. [Physical receive chain and beta calibration](#12-physical-receive-chain-and-beta-calibration)
13. [Stage 6: events, refusals and maximal sharing](#13-stage-6-events-refusals-and-maximal-sharing)
14. [Things worth knowing (observations from reading the code)](#14-things-worth-knowing-observations-from-reading-the-code)
15. [Computational complexity](#15-computational-complexity)
16. [Quick reference](#16-quick-reference)

---

## 1. The problem in one page

### 1.1 The physical picture

There are `N` time slots, each of length `tau` seconds. All energies are in Joules.

- **Master.** It harvests `E_i` at the start of slot `i`. It spends `tau*P_i` on its own data link
  and sends `delta_i` Joules of energy to the slave over an optical link.
- **Slave.** It harvests nothing. Of the `delta_i` sent, only `a_i * delta_i` lands in its battery,
  with `a_i = beta_i * alpha_i`:
  - `alpha_i = exp(-c d)` is the Beer-Lambert transmittance of the water path;
  - `beta_i` is the photodiode + MPPT + storage efficiency.

  The slave spends `tau*Pbar_i` on data sent back to the master. Optionally it also burns a fixed
  electronics load `c_i` every slot.

### 1.2 Objective

A weighted sum-rate, in nats:

```
max  sum_i (th1*tau/2) ln(1 + h_i P_i)  +  sum_i (th2*tau/2) ln(1 + hbar_i Pbar_i)
```

With a stage-6 rate window the slave term becomes `(th2*T_i/2) ln(1 + hbar_i x_i / T_i)`, where
`x_i = tau*Pbar_i` is the burst energy radiated over `T_i` seconds instead of `tau`.

### 1.3 The free variables (the solver's state)

The solver does **not** optimise powers directly. It optimises three arrays of energy flows:

| code | math | meaning | fixed ends |
|---|---|---|---|
| `f[k+1]` | `F_k` | master energy carried from slot `k` into slot `k+1` ("master right tap k") | `f[0] = 0`, `f[N] = fL[N]` (0 without a master floor) |
| `g[k+1]` | `Gt_k = b_k - B_min` | slave battery level above its floor at the end of slot `k` ("slave right tap k") | `g[0] = b0 - B_min`, `g[N] = 0` |
| `d[k]` | `delta_k` | energy the master sends to the slave in slot `k` ("down tap k") | none |

The data energies are **derived** from these flows by the energy balance. In the code they are two
lambdas (lines 322 and 324):

```
xm(i) = E[i] + f[i] - f[i+1] - d[i]                    # tau*P_i     harvest + carry in - carry out - shared
xs(i) = a[i]*d[i] + g[i] - g[i+1] - cS[i]              # tau*Pbar_i  arrival + carry in - carry out - drain
```

This choice of variables is the main design decision in the solver. Because `xm` and `xs` are
*defined* by the balance, the energy-balance equalities can never be violated. Every remaining
constraint becomes a **bound** on a linear function of `(f, g, d)`, and linear bounds are exactly
what `apply()` knows how to enforce.

### 1.4 Constraints and their code form

| tag | meaning | code form | enforced where |
|---|---|---|---|
| M1 | master causality `F_n >= 0` | `f >= 0` | every move touching `f`; `apply()` |
| M6 | master reserve floor (stage 6) | `f >= fL` | replaces the `0` in M1 when `master_floor_J` is set |
| M2 | master capacity | `f[k+1] <= capM[k] = E_max - E_{k+1}` | every move touching `f`; `apply()` |
| M3 | non-negativity | `xm >= 0`, `d >= 0` | every move |
| M5 | laser cap (stage 6) | `d[k] <= Dc[k]` | moves touching `d` when `has_m5` |
| S1 | slave floor `b_n >= B_min` | `g >= 0` | every move touching `g` |
| S2 | slave ceiling, checked at arrival | `g[k] + a[k]*d[k] <= W`, with `W = B_max - B_min` | moves touching `g[k]` or `d[k]` |
| S3 | slave power `Pbar >= 0` | `xs >= 0` | every move with a slave end |
| S4 | schedule | `xs = 0` in silent slots (`tx[i] = False`) | moves whose slave end is silent are skipped |
| S5 | MPPT input rating | `a[k]*d[k] <= capA[k]` | moves touching `d` |
| S6 | burst energy cap (stage 6) | `xs(i) <= X[i]` | `slave_lim()`, `sx()` |
| S7 | burst energy floor (stage 6) | `xs(i) >= Lf[i]` in transmit slots | `slave_lim()`, `sx()` |

**Timing within a slot.** The battery ceiling S2 is checked at *arrival*: the level is
`b_{k-1} + a_k delta_k` before anything is spent. The floor S1 is checked at the *end* of the slot,
after spending and drain. That is why S2 involves `g[k]` (the carry *into* slot `k`) plus
`a[k]*d[k]`, while S1 involves `g[k+1]`.

### 1.5 Optimality in water-filling language

The marginal value of one more Joule of data energy in a slot is:

```
m_k = th1*h_k    / (2*(1 + h_k*P_k))          master
s_k = th2*hbar_k / (2*(1 + hbar_k*Pbar_k))     slave
```

Define the master **water level** `nu_k = P_k + 1/h_k`. Then `m_k = th1 / (2 nu_k)`, so equal
marginal values mean equal water levels (for equal weights). The KKT conditions say:

- **Right tap open and not full** (`fL < F_k < cap`): `nu_k = nu_{k+1}`. Levels are equal across
  the tap.
- **Right tap closed** (`F_k` at its lower bound): `nu_{k+1} >= nu_k`. Energy *would* like to flow
  backward in time, but causality forbids it.
- **Right tap full** (`F_k = cap`): `nu_{k+1} <= nu_k`.
- **Down tap open and the arrival cap slack**: `m_k = a_k * s_k`. One master Joule is worth the
  `a_k` slave Joules it becomes.
- **Down tap closed** (`delta_k = 0`): `m_k >= a_k * s_k`.

The solver **never computes the multipliers**. It recognises optimality as "no available move
improves the objective". Each time a move is clipped by a bound, that bound is active and its
multiplier is the non-negative slack that explains why the condition holds as an inequality.

---

## 2. How the pieces fit together

```
JointParams  (params.py)  -- validates inputs, derives a = beta*alpha, W, capA, overhead, ...
     |
     v
joint_dwf(jp)  (solver.py:208)
  |
  |-- master_floor_profile(jp) ------------> fL   (calls _jit_start for the slave's need)
  |
  |-- choose start point
  |     share-nothing      (default)
  |     warm               (x0 accepted by _warm_start_violation)
  |     just-in-time       (_jit_start; used when a slave load / S7 / S6 makes share-nothing infeasible)
  |     lp-fallback        (_lp_start, scipy HiGHS, if just-in-time breaks a bound)
  |     raise InfeasibleSlaveBudget       (if the LP is infeasible too)
  |
  |-- main loop, repeated up to max_iter sweeps
  |     for each slot k (forward on even sweeps, backward on odd ones):
  |        moves (1)..(10):  t = step(...)            unclipped optimum along the move
  |                          t = clip(t, lo, hi)      inline, or via apply()
  |                          update f / g / d
  |     if the largest move < tol*scale:
  |        polish()   --> families D, A, B, C, each step() + apply()
  |        if polish's largest move < tol*scale:  converged, stop
  |
  |-- post-processing: flows -> P, Pbar, F, b, b_peak, rates, objective
  v
JointResult
     |
     |-- check_kkt_joint()            (verification.py) feasibility + down-tap KKT test
     |-- reference_solution_joint()   (verification.py) independent cvxpy optimum
     |-- physical_stored_power()      (physical.py) what the real chain would store
     |-- solve_with_physical_chain()  (physical.py) beta fixed point, re-calls joint_dwf
     |-- solve_with_refusals()        (physical.py) event refusal loop, re-calls the above
     |-- maximal_sharing()            (solver.py) tie-break post-pass for theta1 = 0
```

`step()`, `apply()` and `polish()` are **nested functions** inside `joint_dwf()`. They are closures:
they read and write `f`, `g`, `d` and read `E`, `a`, `W`, `tau`, `fL`, `capM`, `capA`, `tx`, ...
directly from the enclosing scope. That is why their argument lists are short and why they cannot
be imported or tested on their own.

### 2.1 Critical pieces beyond `step()`, `apply()` and `polish()`

These three functions are the core of the algorithm, but the result also depends on the pieces
below. Each is a summary; the linked section has the details.

- **`_jit_start` and `_lp_start`.** When the slave has a load, they decide whether the problem is
  feasible at all. Every move keeps a feasible point feasible, but no move can make an infeasible
  point feasible, so without a feasible start the loop cannot begin.
  - With a slave drain `c > 0`, the default "share nothing" start breaks the fixed end `g[N] = 0`.
  - `_jit_start` sends in each slot exactly what the slave is short of there.
  - If that breaks a bound, `_lp_start` solves a feasibility LP that may borrow energy through the
    master carry.
  - If the LP is infeasible too, `joint_dwf()` raises `InfeasibleSlaveBudget`.

  See [section 9.2](#92-_jit_startjp-line-24) and [section 9.4](#94-_lp_startjp-line-114).

- **`master_floor_profile`.** It computes `fL`, the per-slot M6 reserve floor of the master carry.
  `fL` is the lower bound on `f` everywhere it appears:
  - in `apply()`;
  - in the inline bounds of moves (3), (4), (6) and (8);
  - in the start point, where `f = fL`;
  - in `_lp_start`, `reference_solution_joint` and `check_kkt_joint`.

  It calls `_jit_start` itself, to learn the slave's minimum need before ramping the floor up.
  Without a master floor it returns zeros, which leaves the original `f >= 0` unchanged. See
  [section 9.3](#93-master_floor_profilejp-line-79).

- **Moves (2)–(5), (8), (9) and (10).** They look like boilerplate next to the textbook moves (1),
  (6) and (7), but several of them stop the sweep stalling at corners:
  - swaps (2) and (4) get past a tight slave arrival cap;
  - pass-throughs (3), (5), (8) and (9) cross a slot with zero power;
  - move (5) is the only sweep move that can send energy into a silent slot.

  Move (10) is missing from the list in the module docstring, which stops at (9). It is the only
  **sweep** move that re-times when the master sends without changing what the slave spends;
  `polish()` family D does the same between slots that are not adjacent. See
  [section 7](#7-the-ten-sweep-moves-one-by-one).

- **`slave_lim` and `sx`.** These helpers add the S6 burst cap and S7 burst floor to the bounds:
  `slave_lim` to the inline bounds of the moves, and `sx` to the end-node tuples that `apply()`
  receives. See [section 8](#8-small-helpers-inside-joint_dwf).

- **`check_kkt_joint`.** An optimality check that runs after the solve. The solver stops when no
  move improves the objective, but that only implies optimality if the move set is rich enough.
  `check_kkt_joint` checks the result from outside:
  - feasibility: battery window, non-negative powers, and the slave energy ledger;
  - the **down-tap** KKT condition `m_k = a_k s_k`.

  It does not test the water-level conditions across time, so passing it is necessary for
  optimality but not sufficient. See [section 11.1](#111-check_kkt_jointjp-r-verificationpy14).

- **`reference_solution_joint`.** It solves the same convex problem with cvxpy and shares no code
  with `joint_dwf()`, so agreeing with it is a genuine test. The module docstring says that on
  extreme random settings the solver was within about 0.1% of the optimum in 3 of 25 tests, but
  that is out of date. When the master gives the slave all its energy, the solver often stops 2–9%
  below the optimum, because one move is missing (README open item 17; measured in `TESTS.md`).
  When a configuration matters, this comparison is how you trust the result. `parity_matrix` runs
  the comparison over 11 problem families. See
  [section 11.2](#112-reference_solution_jointjp-verificationpy148).

- **`warm_start_from` and `_warm_start_violation`.** They matter when you re-solve repeatedly, as
  the refusal loop does:
  - `warm_start_from` turns a previous result back into `(f, g, d)` flows.
  - `_warm_start_violation` rejects that start if it breaks any bound of the *new* problem, for
    example after a schedule or beta change.

  A start that breaks a bound would never recover, so this check is essential. See
  [section 9.5](#95-warm_start_fromjp-r-line-171) and
  [section 9.6](#96-_warm_start_violationjp-f-g-d-tol-line-184).

- **`JointParams`.** It validates the inputs and derives every array the solver reads: `a = beta*alpha`,
  `W`, `capA`, `g_init`, `overhead`, `txcap`, `txfloor`, `dcap` and `Twin`. The solver trusts these
  values completely, so a wrong value here gives a wrong answer that **still converges cleanly**.
  See [section 10.1](#101-jointparams-inputs-validation-and-derived-arrays).

- **Stage-6 wrappers.** They call `joint_dwf()` from outside:
  - `solve_with_refusals` re-runs the calibrated solve after refusing events. It refuses an event
    when its burst is below `x_min`, or when the solve raises `InfeasibleSlaveBudget`. In the second
    case it finds the slot to blame with `fullest_trajectory`, and falls back to the exception's
    `.slots` only when that trajectory never drops below `B_min`.
  - `maximal_sharing` is a tie-break pass after the solve, for `theta1 = 0`. It keeps every slave
    burst and fills the slave supercap as early as possible.
  - `solve_with_physical_chain` is the beta calibration. It re-solves with each slot's beta set to
    the real photodiode + MPPT gain until beta stops changing.
  - `physical_stored_power` is the step that runs a solution through the physical chain.

  See [section 12](#12-physical-receive-chain-and-beta-calibration) and
  [section 13](#13-stage-6-events-refusals-and-maximal-sharing).

---

## 3. `step()`: the one equation

**Location:** `solver.py:329`

```python
def step(thu, hu, xu, s_, thw, hw, xw, r_, Tu=None, Tw=None) -> float
```

### 3.1 What question it answers

Every move in the solver has the same shape. Energy leaves a **source node** `u` and arrives at a
**sink node** `w`. A node is the data energy of one slot: a master slot (`xm`) or a slave slot
(`xs`).

- Moving `t` Joules lowers the source's data energy by `s*t`.
- It raises the sink's data energy by `r*t`.

`r` is the loss factor of the path: `r = 1` for a battery carry, `r = a_k` for the optical down tap,
`r = a_k/a_j` for a delta shift.

`step()` returns the `t` that maximises the objective along that line, ignoring every constraint.

### 3.2 Derivation

Along the move, only the two end nodes change, so the objective as a function of `t` is:

```
phi(t) = U_u(x_u - s t) + U_w(x_w + r t),        U(x) = (th*tau/2) ln(1 + h x / tau)
```

Its derivative is:

```
phi'(t) = - s * th_u h_u / (2 (1 + h_u (x_u - s t)/tau))
          + r * th_w h_w / (2 (1 + h_w (x_w + r t)/tau))
```

The first term is the marginal value lost at the source and the second is the marginal value gained
at the sink. Set `phi'(t) = 0` and cross-multiply:

```
s th_u h_u (1 + h_w (x_w + r t)/tau) = r th_w h_w (1 + h_u (x_u - s t)/tau)
```

Both sides are **linear in t**, because the logarithm's derivative is a reciprocal. So the optimum
has a closed form. With `s = 1`, which is what every call in the code passes:

```
t* = tau * [ r th_w h_w (1 + h_u x_u/tau) - th_u h_u (1 + h_w x_w/tau) ]
           / [ h_u h_w (th_u r + r th_w) ]
```

In the code, `num` and `den` are exactly this numerator and denominator (lines 374–378). `phi` is
strictly concave whenever `den > 0`, so `t*` is the unique maximiser along the line.

### 3.3 Interpretation of the sign

- `t* > 0`: the sink's (loss-adjusted) marginal value exceeds the source's, so move energy
  `u -> w`.
- `t* < 0`: the source is worth more, so move energy back `w -> u`. That is only possible if the
  flow being changed is currently positive, which `apply()` or the inline bounds check.
- `t* = 0`: already balanced.

### 3.4 The two special cases worth recognising

**Right tap (master k → master k+1, `s = r = 1`, same weight).** The formula reduces to:

```
t* = 0.5 * tau * (nu_k - nu_{k+1}),       nu = P + 1/h
```

This is classic water-filling: pour half the level difference across so both levels meet.

*Worked example* (`tau = 1`, `h = 1`, `th = 1`): `x_u = 3`, `x_w = 1`, so `nu_u = 4` and
`nu_w = 2`. Then `t* = 0.5 * (4 - 2) = 1`. From the general formula: `num = 1*(1+3) - 1*(1+1) = 2`,
`den = 1*(1+1) = 2`, `t* = 1`. After the move both slots hold 2 J, so the levels are equal.

**Down tap (master k → slave k, `s = 1`, `r = a_k`).** The formula solves
`th1 h/(1+hP) = a_k th2 hbar/(1+hbar Pbar)`, i.e. `m_k = a_k s_k`.

*Worked example* (`tau = 1`, `h = hbar = 1`, `th1 = 1`, `th2 = 2`, `a = 0.5`, `x_m = 4`, `x_s = 0`):

```
num = 0.5*2*1*(1 + 4) - 1*1*(1 + 0) = 4
den = 1*1*(1*0.5 + 0.5*2)           = 1.5
t*  = 4 / 1.5 = 8/3 J
```

After the move, `x_m = 4/3` and `x_s = 0.5 * 8/3 = 4/3`. The master marginal value is
`1/(1 + 4/3) = 3/7` and the scaled slave value is `0.5 * 2/(1 + 4/3) = 3/7`. They are equal, as
required.

### 3.5 Rate window (stage 6)

When the slave transmits a short burst of `T` seconds instead of the whole slot, its utility is
`(th*T/2) ln(1 + h x/T)` and its marginal value is `th h / (2(1 + h x/T))`. The balance equation is
still linear in `t`:

```
t* = [ r th_w h_w (1 + h_u x_u/T_u) - th_u h_u (1 + h_w x_w/T_w) ]
     / [ h_u h_w r (th_u/T_w + th_w s/T_u) ]
```

`Tu = None` or `Tw = None` means "this end is a master node", so `T = tau`. When **both** are
`None` the code takes the original branch (lines 374–378) with no extra arithmetic, which keeps the
legacy results bit-identical.

### 3.6 Degenerate case

If `den <= 0`, `step()` returns `0`. This happens when both end weights are zero, for example two
master nodes when `th1 = 0`. There is nothing to gain from moving energy between two worthless
nodes. (See [section 14](#14-things-worth-knowing-observations-from-reading-the-code) for what this
means at `th1 = 0`.)

### 3.7 What `step()` does **not** do

It does not look at any constraint and it does not change any variable. It is a pure function of
the current end-node energies. Feasibility is the caller's job (inline bounds or `apply()`).

---

## 4. `apply()`: clip to the feasible interval and move

**Location:** `solver.py:405`

```python
def apply(fch, gch, dch, src, snk, t_star) -> float
```

### 4.1 Inputs

| argument | form | meaning |
|---|---|---|
| `fch` | list of `(q, c)` | after the move, `f[q] += c*t` |
| `gch` | list of `(q, c)` | after the move, `g[q] += c*t` |
| `dch` | list of `(q, c)` | after the move, `d[q] += c*t` |
| `src` | `(x, coef)` or `(x, coef, L, U)` | the source node's current data energy and how it changes per unit `t` (negative coefficient) |
| `snk` | `(x, coef)` or `(x, coef, L, U)` | the sink node, same format (positive coefficient) |
| `t_star` | float | unclipped optimum from `step()` |

The three lists together are a **path**: a direction in `(f, g, d)` space. The path is built so
that every intermediate slot's data energy is unchanged ("pass-through") and only the two end nodes
change.

### 4.2 Algorithm

**Step 1: start with an unbounded interval.** `lo = -inf`, `hi = +inf`.

**Step 2: turn every affected constraint into a bound on t.** Every constraint is linear in the
flows. After the move, each affected quantity is `q(t) = v + c*t` and must satisfy `L <= q(t) <= U`.
The inner helper `lim(v, c, L, U)` converts this into a bound on `t`:

```
c > 0 :   t <= (U - v)/c    and   t >= (L - v)/c
c < 0 :   t >= (U - v)/c    and   t <= (L - v)/c      (dividing by a negative flips the inequality)
c = 0 :   no bound
```

The constraints checked, in order:

| line | quantity | bounds | constraint |
|---|---|---|---|
| 446–447 | `f[q]` | `[fL[q], capM[q-1]]` | M1/M6 (floor), M2 (capacity of master tap `q-1`) |
| 448–449 | `g[q]` | `[0, inf)` | S1 |
| 450–451 | `d[q]` | `[0, inf)` | M3 (`delta >= 0`) |
| 452–453 | `g[q] + a[q]*d[q]` | `(-inf, W]` | S2 arrival cap of slot `q` |
| 454–455 | `a[q]*d[q]` | `(-inf, capA[q]]` | S5 MPPT input rating |
| 456–458 | `d[q]` | `(-inf, Dc[q]]` | M5 laser cap (only if `has_m5`) |
| 459 | source data energy | `[0, inf)` or its own `[L, U]` | M3 / S3, plus S6/S7 for a slave end |
| 460 | sink data energy | same | same |

**The arrival-cap detail.** A move may change both `g[q]` and `d[q]` in the same slot. For example,
move (10) changes `d[k]`, and polish family D changes `g[k+1..j]` and `d[k]`. The arrival quantity
`g[q] + a[q]*d[q]` then changes by `(c_g + a[q]*c_d)*t`. The dictionary `arr` accumulates that
combined coefficient per slot before the cap is checked. Checking `g` and `d` separately would get
it wrong: it would either over-constrain or miss a violation.

**Step 3: empty-interval guard.** If `lo > hi`, return `0` and change nothing. The current point is
always feasible, so `0` is always inside `[lo, hi]` in exact arithmetic. An empty interval can only
come from round-off.

**Step 4: project and move.** `t = min(max(t_star, lo), hi)` (line 463), then apply `c*t` to every
listed variable.

**Step 5: return `|t|`.** The caller compares the largest `|t|` of a sweep with `tol*scale` to
decide convergence.

### 4.3 Why clipping is optimal, not just safe

`phi(t)` (section 3.2) is concave, and the feasible set along the line is an interval `[lo, hi]`.
For a concave function of one variable, the constrained maximiser is the unconstrained maximiser
projected onto the interval: if `t* > hi` the function is still increasing at `hi`, so `hi` is best.
So each move is the **exact** best step along its direction, and the objective can never decrease.

### 4.4 The complementary-slackness reading

When `t` lands on `lo` or `hi`, the constraint that set that bound is now **active**. That is the
moment its KKT multiplier becomes (potentially) positive. The solver never stores the multiplier,
but this is how a converged point ends up satisfying the inequality forms of the KKT conditions in
section 1.5.

### 4.5 Who uses `apply()`

Only the "long" or "multi-variable" moves call it: move (10) and all four `polish()` families. The
nine frequent moves (1)–(9) inline the same arithmetic by hand, as `lo = max(...)`,
`hi = min(...)`, `t = min(max(t, lo), hi)`. They do this to avoid building lists and dictionaries
`N` times per sweep. The logic is identical; the inline versions just list only the bounds that can
bind for that particular move.

---

## 5. `polish()`: long moves through dead slots

**Location:** `solver.py:469`

### 5.1 The problem it solves

Call a slot **dead** if its data energy is (almost) zero:

```python
dead = lambda xv: xv <= 1e-6 * scale        # line 401
```

In a dead master slot, `P = 0`, so the water level sits on the floor `nu = 1/h`. Now suppose
energy should move from slot `i` to slot `j = i + 2`, because `nu_i > nu_j`. But the slot in
between is dead and its floor `1/h_{i+1}` is **above** `nu_i`, for example because the channel is
very poor in slot `i+1`:

```
slot:          i          i+1             i+2
P:             3          0 (dead)        0.5
level nu:      4          floor = 10      2.5
```

- The single-tap move `i → i+1` sees a *higher* level at `i+1`, so `step()` says "don't move" (or
  move backward, which causality forbids).
- The move `i+1 → j` has nothing to give, because `xm(i+1) = 0`.

Each local move looks unprofitable, yet the end-to-end move `i → j` is profitable. Block coordinate
ascent with only single-tap moves gets **stuck** at such a corner. This is the classic weakness of
coordinate methods on non-smooth (bound-constrained) problems.

`polish()` fixes this with end-to-end moves. It raises every carry along the path by `t`. Each
intermediate slot then gets `+t` carried in and `-t` carried out, so its data energy is unchanged,
and only the two end nodes see the change.

### 5.2 When it runs

It runs only when a full sweep moved less than `tol*scale` (line 730). The plain moves are cheap and
handle almost everything; `polish()` is the expensive escape hatch for corners. If `polish()` moves
anything, the main loop simply continues sweeping. If it also moves nothing, the solver declares
convergence.

### 5.3 The four families

All four use the same recipe:

1. Build the path lists `fch`, `gch`, `dch`.
2. Call `step()` on the two **end nodes** only.
3. Call `apply()` with the path.

The `apply()` call guarantees the intermediate carries stay within their bounds (`f` in
`[fL, cap]`, `g >= 0`, arrival caps).

#### Family D: delta shift across a run (lines 506–516)

Runs only if `use_cross` is on. By default that means a slave load is attached
(`cross_run=None → has_load`).

```
for k, for j >= k+2 (both a > 0):
    d[k]      += t
    g[k+1..j] += a_k t           # the extra arrival rides the slave battery from k to j
    d[j]      -= (a_k/a_j) t     # the master sends that much less at j
```

| quantity | effect |
|---|---|
| `xs(k)` | `+a_k t - a_k t = 0` |
| intermediate `xs` | `+a_k t - a_k t = 0` |
| `xs(j)` | `a_j * (-(a_k/a_j) t) + a_k t = 0` |
| `xm(k)` | `-t` |
| `xm(j)` | `+(a_k/a_j) t` |

The slave's spending is unchanged everywhere. The move re-times **when the master sends**:
earlier, at the cost of storing it in the slave battery. The balance is
`m_k = (a_k/a_j) m_j`, which is `step(th1, h[k], xm(k), 1, th1, h[j], xm(j), a_k/a_j)`.

Why it exists: with a slave load, silent slots drain the battery, so energy must be routed through
runs of silent slots. Move (10) only shifts between adjacent slots, and a high master level in an
intermediate slot can block the chain of adjacent shifts even though the end-to-end shift pays.

Cost: `O(N^2)` pairs, each with a path of length up to `N`, so `O(N^3)` per `polish()` call. This
is the most expensive part of the solver on long horizons.

#### Family A: master chain (lines 517–523)

```
for i, for j = i+2, i+3, ... while slot j-1 is dead:
    f[i+1..j] += t                 # carry from i all the way to j
```

Source `xm(i)` gets `-t`, sink `xm(j)` gets `+t`, and every dead intermediate slot is unchanged.
The balance is `nu_i = nu_j` (`r = 1`). The loop **extends `j` only while the newest intermediate
slot is dead**, because energy cannot be passed through a live slot in this way. Single moves
handle live slots anyway.

#### Family B: slave chain (lines 524–531)

The same as A on `g`, using `xs` for the dead test. Both ends must be transmit slots
(`tx[i] and tx[j]`): a silent end must keep `Pbar = 0` (S4). It uses `sx()` so a slave end carries
its own S6/S7 bounds, and `Ts()` for the rate window.

#### Family C: cross paths, master → down tap → slave (lines 532–570)

This is the most general family. For each down tap `k` with `a_k > 0`:

1. **Collect master sources** `mo`: a list of `(source slot i, f-changes)`.
   - `i = k`: no carry change.
   - `i < k`: walk left. The energy is carried forward from `i` to `k`, so `f[i+1..k] += t`.
   - `i > k`: walk right. Less energy is carried forward from `k` to `i`, so `f[k+1..i] -= t`
     ("undo carry"). The master at slot `i` gives up energy it would have received from `k`, so `k`
     can send more.
   - Each walk **stops after the first live slot**. A live slot can be the source, but energy
     cannot pass *through* it.

2. **Collect slave sinks** `so`: a list of `(sink slot j, g-changes)`.
   - `j = k` if `tx[k]`.
   - `j > k`: the arrival rides the slave battery forward, so `g[k+1..j] += a_k t`.
   - `j < k`: the slave spends earlier and carries less forward, so `g[j+1..k] -= a_k t`.
   - Silent sinks are not added (S4), but the walk continues through them. Each walk stops after
     the first live slave slot.

3. **Try every (source, sink) pair** with at least two changes in total. Pairs with fewer than two
   are exactly moves (1)–(5), which the sweep already does. For each pair:

   ```
   t = step(th1, h[i], xm(i), 1, th2, hb[j], xs(j), a[k], None, Ts(j))
   apply(fch, gch, [(k, 1.0)], (xm(i), -1.0), sx(j, a[k]), t)
   ```

   The balance is `m_i = a_k s_j`: a master Joule at `i`, moved by carries to `k`, sent over the
   down tap, and carried by the slave battery to `j`.

### 5.4 Return value

The largest `|t|` applied across all families. `joint_dwf()` compares it with `tol*scale`.

### 5.5 Guarantee

Every `polish()` move is a `step()` + `apply()` pair, so by the argument of section 4.3 none of them
can lower the objective. `polish()` only ever helps.

---

## 6. `joint_dwf()`: the driver

**Location:** `solver.py:208`

```python
def joint_dwf(jp, tol=1e-10, max_iter=1000, cross_run=None, x0=None) -> JointResult
```

### 6.1 Parameters

| parameter | meaning |
|---|---|
| `jp` | `JointParams`, the whole problem |
| `tol` | relative stop tolerance. A move counts as "nothing" if it is smaller than `tol * scale`, with `scale = max(mean(E), 1)` (line 327) |
| `max_iter` | cap on the number of **sweeps**. One sweep visits every slot once. `polish()` calls are not counted |
| `cross_run` | force `polish()` family D on or off. `None` turns it on only when a slave load exists |
| `x0` | optional warm start `(f, g, d)`, normally from `warm_start_from()` |

### 6.2 Phase 1: unpack and set feature flags (lines 251–268)

It pulls everything it needs out of `jp` into local names: `E`, `h`, `hb`, `a`, `th1`, `th2`, `W`,
`capM`, `tx`, `cS`, `capA`, and the stage-6 arrays `X`, `Lf`, `Dc`, `Tw`. It then sets boolean
flags `has_s6`, `has_s7`, `has_m5` and `has_T`.

Every stage-6 bound check is guarded by one of these flags. The design goal, stated in the
comments: with the stage-6 options off, the legacy path executes **not one extra operation**, so
old results reproduce bit-for-bit. That is what `tests/baseline` relies on.

### 6.3 Phase 2: the start point (lines 270–319)

Block coordinate ascent needs a **feasible** start. Every move keeps feasibility, but no move can
restore it. The start is chosen in this order:

1. **Share-nothing (default).**
   - `f = fL`: the master holds exactly its floor; `fL` is all zeros without a floor.
   - `d = 0`.
   - `g[0] = b0 - B_min`, and if the first slots are silent, that initial excess is carried
     unchanged up to the first transmit slot (line 283). Otherwise S4 would be broken: the excess
     must be spent somewhere, and only transmit slots may spend.

   It is "spend what you harvest, share nothing".

2. **Warm start** (`x0` given). `_warm_start_violation()` checks every bound to
   `1e-9 * scale`. If the check passes, tiny negatives are clipped and the fixed ends are re-imposed
   exactly. If it fails, the cold start is used and the reason is appended to `start`, for example
   `"share-nothing (warm start rejected: S5 broken by 3e-4 J)"`.

3. **Just-in-time.** Used when there is a slave load (`c > 0`), an S7 floor, or an S6 cap with
   initial charge. Share-nothing is then infeasible, because the slave must end at `g[N] = 0` and
   must pay `c_i` every slot, so it has to be fed. `_jit_start()` sends in each slot exactly what
   the slave is short of.

4. **LP fallback.** If just-in-time breaks a bound (typically: the master's harvest in a slot is
   too small to cover that slot's need with `f = 0`), `_lp_start()` solves a feasibility LP that may
   borrow energy through the master carry.

5. **Infeasible.** If the LP is infeasible too, `InfeasibleSlaveBudget` is raised. Its `.slots`
   attribute lists the slots just-in-time blamed, and the stage-6 refusal loop uses those slots.

The string `start` records which path was taken and ends up in `JointResult.start`.

### 6.4 Phase 3: the main loop (lines 574–733)

```
it = 0
while it < max_iter:
    big = 0
    order = 0..N-1 if it is even else N-1..0           # alternate direction
    for k in order:
        try moves (1)..(10) at slot k; big = max(big, |t| of each move)
    it += 1
    if big < tol*scale:
        if polish() < tol*scale:
            converged = True; break
```

**Why alternate direction.** Energy flowing forward in time propagates one tap per slot visited.
In a forward sweep, a surplus at slot 0 can ripple all the way to slot `N-1` in one pass. A deficit
at the end, pulling energy backward, would need `N` forward sweeps but only one backward sweep.
Alternating gives both directions fast propagation.

**Why moves are applied immediately (Gauss-Seidel style).** Each move updates `f`, `g`, `d` at
once, and the next move sees the new values through the `xm`/`xs` lambdas. This is standard block
coordinate ascent and converges faster than computing all moves from the same old point.

**Stopping rule.** Convergence needs **two** quiet events in the same iteration: a quiet sweep,
then a quiet `polish()`. If `max_iter` is reached first, `converged = False` is returned. The result
is still feasible, but it may be below the optimum.

**Line 576** prints `joint_dwf sweep {it} (start = ...)` on every sweep. See section 14.

### 6.5 Phase 4: post-processing (lines 735–753)

| result field | computed as | meaning |
|---|---|---|
| `x`, `xb` | `max(xm(i), 0)`, `max(xs(i), 0)` | data energies; the clamp removes only round-off |
| `P`, `Pbar` | `x/tau`, `xb/tau` | powers [W] |
| `F` | `cumsum(E - x - delta)` | master battery after each slot (rebuilt from the balance, not copied from `f`) |
| `b` | `B_min + g[1:]` | slave battery at the end of each slot |
| `b_peak` | `B_min + g[:-1] + a*delta` | slave battery right after arrival (the left side of S2) |
| `sent_power` | `delta/tau` | power the master sends |
| `arrived_power` | `alpha*delta/tau` | power reaching the slave's photodiode |
| `stored_power` | `a*delta/tau` | power ending up in the battery |
| `rx_master`, `rx_slave` | `h*P`, `hb*Pbar` | SNRs (noise-normalised received power) |
| `rate_master`, `rate_slave` | `sum 0.5 tau ln(1 + h P)` and the slave version (burst form if `has_T`) | nats |
| `objective` | `th1*rate_master + th2*rate_slave` | the maximised value |
| `iters`, `converged`, `start` | as above | diagnostics |

---

## 7. The ten sweep moves, one by one

Notation: `k` is the current slot and `t` is the Joules moved. Each move lists:

- the **variables changed**;
- the **effect** on data energies (only the ends change; everything else is pass-through);
- the **balance** that `step()` solves;
- the **bounds** used to clip.

Each bound is labelled with the multiplier that becomes active when it binds.

### (1) Down tap, master k → slave k (lines 590–597)

- **Changes:** `d[k] += t`
- **Effect:** `xm(k) -= t`, `xs(k) += a_k t`
- **Balance:** `m_k = a_k s_k`. This is the condition `Lambda_k = a_k Gamma_k` with the arrival
  cap slack.
- **Bounds:**
  - lower: `d[k] >= 0` (rho_k); `xs(k) >= 0` (eta_bar_k); S7 floor.
  - upper: `xm(k) >= 0` (eta_k); arrival cap `g[k] + a_k(d[k]+t) <= W` (mu_bar_{k-1}); S5; M5; S6.
- **Skipped if:** `a_k = 0`, or slot `k` is silent (the slave end would break S4).

This is the move that *creates* sharing; all the others redistribute it.

### (2) In-swap, master k → slave k−1 (lines 606–613)

- **Changes:** `d[k] += t`, `g[k] -= a_k t`
- **Effect:** `xm(k) -= t`; slave slot `k` unchanged (arrival `+a_k t`, carry-in `-a_k t`);
  `xs(k-1) += a_k t`, because its carry-out fell.
- **Balance:** `m_k = a_k s_{k-1}`
- **Why it exists:** if the arrival cap at `k` is tight, move (1) is blocked. Here the arrival
  quantity `g[k] + a_k d[k]` is **unchanged**, so the cap never blocks it. The slave spends
  earlier, which frees battery room for the master to send more at `k`.
- **Bounds:** `d[k] >= 0` and `xs(k-1) >= 0` below; `xm(k) >= 0`, `g[k] >= 0`, S5 and M5 above.

### (3) Master pass-through, master k−1 → slave k (lines 622–630)

- **Changes:** `f[k] += t`, `d[k] += t`
- **Effect:** `xm(k-1) -= t`; master slot `k` unchanged (carry-in `+t`, shared `+t`);
  `xs(k) += a_k t`
- **Balance:** `m_{k-1} = a_k s_k`
- **Why it exists:** if master slot `k` is dead, it cannot fund move (1). This move routes energy
  from `k-1` *through* `k` to the slave.
- **Bounds:** `f[k] in [fL[k], capM[k-1]]`, `d[k] >= 0`, `xs(k) >= 0`, `xm(k-1) >= 0`, arrival cap,
  S5, M5, S6/S7.

### (4) Master out-swap, master k+1 → slave k (lines 638–646)

- **Changes:** `f[k+1] -= t`, `d[k] += t`
- **Effect:** `xm(k)` unchanged (carry-out `-t`, shared `+t`); `xm(k+1) -= t`; `xs(k) += a_k t`
- **Balance:** `m_{k+1} = a_k s_k`
- **Meaning:** energy the master *was* carrying forward to `k+1` is sent to the slave at `k`
  instead. It "pulls" from the future, which is legal only because the carry `f[k+1]` already
  exists: hi `= f[k+1] - fL[k+1]`.
- **Bounds:** also `f[k+1] <= capM[k]` when `t < 0`, `xm(k+1) >= 0`, arrival cap, S5, M5, S6/S7.

### (5) Slave pass-through, master k → slave k+1 (lines 654–663)

- **Changes:** `d[k] += t`, `g[k+1] += a_k t`
- **Effect:** `xm(k) -= t`; slave slot `k` unchanged (arrival `+a_k t`, carry-out `+a_k t`);
  `xs(k+1) += a_k t`
- **Balance:** `m_k = a_k s_{k+1}`
- **Notable:** only `tx[k+1]` is required, not `tx[k]`. So this is how the master **sends energy
  into a silent slot** to be stored and spent later. Move (1) can't, because its slave end would be
  silent.
- **Bounds:** the arrival cap at **both** `k` and `k+1` (both levels rise), plus `g[k+1] >= 0`,
  `xm(k) >= 0`, `xs(k+1) >= 0`, S5, M5, S6/S7.

Moves (6)–(10) need a right neighbour, so they are skipped at `k = N-1` (line 664).

### (6) Master right tap, master k → master k+1 (lines 671–674)

- **Changes:** `f[k+1] += t`
- **Effect:** `xm(k) -= t`, `xm(k+1) += t`
- **Balance:** `nu_k = nu_{k+1}`. This is pure water-filling over time on the master side.
- **Bounds:** `f[k+1] >= fL[k+1]` (closed tap / floor, lambda_k); `f[k+1] <= capM[k]` (full tap,
  mu_k); `xm(k) >= 0`; `xm(k+1) >= 0`.

### (7) Slave right tap, slave k → slave k+1 (lines 681–689)

- **Changes:** `g[k+1] += t`
- **Effect:** `xs(k) -= t`, `xs(k+1) += t`
- **Balance:** `nubar_k = nubar_{k+1}`
- **Bounds:** `g[k+1] >= 0` (slave at floor, lambda_bar_k); arrival cap of the **next** slot
  `g[k+1] + a_{k+1} d[k+1] <= W` (mu_bar_k); `xs >= 0` at both ends; S6/S7 at both ends.
- **Skipped if:** either end is silent.

### (8) Master carry pass-through, master k−1 → master k+1 (lines 694–700)

- **Changes:** `f[k] += t`, `f[k+1] += t`
- **Effect:** `xm(k-1) -= t`, `xm(k+1) += t`, `xm(k)` unchanged
- **Balance:** `nu_{k-1} = nu_{k+1}`
- **Why:** this is polish family A for a run of length one, done cheaply on every sweep.

### (9) Slave carry pass-through, slave k−1 → slave k+1 (lines 706–715)

- **Changes:** `g[k] += t`, `g[k+1] += t`
- **Effect:** `xs(k-1) -= t`, `xs(k+1) += t`, `xs(k)` unchanged
- **Balance:** `nubar_{k-1} = nubar_{k+1}`
- **Bounds:** both carries `>= 0`, both arrival caps (slots `k` and `k+1`), `xs >= 0` at the ends,
  S6/S7.

### (10) Delta shift, master k → master k+1 via the slave battery (lines 722–726)

- **Changes:** `d[k] += t`, `g[k+1] += a_k t`, `d[k+1] -= (a_k/a_{k+1}) t`
- **Effect:** `xm(k) -= t`, `xm(k+1) += (a_k/a_{k+1}) t`; `xs(k)` and `xs(k+1)` both unchanged
- **Balance:** `m_k = (a_k/a_{k+1}) m_{k+1}`
- **Meaning:** it re-times *when* the master sends, without changing what the slave spends. This is
  useful when the master's own value differs between `k` and `k+1` but the slave is indifferent.
- **Implementation:** uses `apply()`, because the arrival cap at `k` changes through `d[k]` while
  the cap at `k+1` is unchanged (`+a_k t` from `g`, `-a_k t` from `d`). The `arr` dictionary
  handles that bookkeeping.

### Why so many moves?

Moves (1), (6) and (7) alone are the textbook DWF: one tap per type, balancing marginal values.
They suffice whenever no constraint binds in an awkward place. The other seven are **compound
directions**: two or three variables moving together so that a bound which blocks every single
variable no longer blocks the combination.

- Dead slots block single taps → pass-throughs (3), (5), (8), (9), and `polish()` A–C.
- A tight arrival cap blocks the down tap → swaps (2) and (4).
- The timing of sending versus spending is coupled → delta shift (10) and `polish()` D.

---

## 8. Small helpers inside `joint_dwf()`

| helper | line | what it does |
|---|---|---|
| `xm(i)`, `xs(i)` | 322, 324 | Energy balance. They are the **only** definition of the data energies; nothing caches them, so they are always consistent with the current flows. |
| `scale` | 327 | `max(mean(E), 1)`. Makes `tol` relative to the typical harvest per slot. |
| `Ts(i)` | 267 | Rate window of slave slot `i` for `step()`; `None` when no window is set (legacy arithmetic). |
| `slave_lim(lo, hi, j, xv, c)` | 380 | Adds S6 (`xs(j) <= X[j]`) and S7 (`xs(j) >= Lf[j]`, transmit slots only) to an inline move's interval. `c` is how `xs(j)` changes per unit `t`; the sign of `c` decides which end of the interval each bound tightens. |
| `sx(j, coef)` | 395 | Builds the end-node tuple for `apply()` when the end is a slave slot: `(xs(j), coef)` normally, or `(xs(j), coef, L, U)` when S6/S7 are on. |
| `dead(xv)` | 401 | `xv <= 1e-6*scale`. The "this slot has zero power" test used by `polish()` to extend paths. |

**S6/S7 in the KKT conditions** (from the `slave_lim` docstring): they add multipliers `kappa_j`
(cap) and `iota_j` (floor) to the slave stationarity, `s_j - kappa_j + iota_j = Lambda_bar_j`. A
binding S6 lowers the value of a Joule landing in slot `j` to what the battery could carry on.

---

## 9. Start points: making the first point feasible

### 9.1 `InfeasibleSlaveBudget` (line 15)

A `ValueError` subclass meaning "no schedule can keep the slave alive". The slave drain, window
`W`, MPPT cap or master harvest make S1–S5 with `g[N] = 0` unsatisfiable. Constraints are **never
relaxed**; the error is raised instead.

`.slots` holds the slots the just-in-time start blamed. `solve_with_refusals()` uses them to decide
which event to refuse.

### 9.2 `_jit_start(jp)` (line 24)

The just-in-time feasible start: the master sends in slot `k` exactly what the slave is short of
there.

```
f = 0, xs = 0 (the slave spends nothing on data yet)
short_k = max(0, c_k - g[k])            # drain not covered by the battery
d[k]    = short_k / a_k                  # send just enough
g[k+1]  = g[k] + a_k d[k] - c_k          # = 0 when short_k > 0, else g[k] - c_k
```

The resulting schedule hovers at `B_min`. It is then checked against every bound, and each failure
is recorded as `(slot, reason)`:

- `a_k = 0` while the slave is short: no energy can land at all.
- `d[k] > E[k]`: the master's own harvest can't pay, because `f = 0` here. The LP fallback may fix
  this by borrowing.
- arrival above `W`;
- `a_k d_k > capA_k` (S5);
- `d_k > Dc_k` (M5).

**S7.** With an S7 floor, the floor is added to the drain (line 38), so every transmit slot starts
exactly at its floor.

**Leftover initial charge.** If `b0` exceeds the total drain, `g[N] > 0` at the end. That leftover
is spent as data in the first transmit slot (`g[j+1:] -= left`), unless S6 forbids it or no
transmit slot exists, which is recorded as a violation.

Returns `(d, g, violations)`. An empty `violations` list means the start is feasible.

### 9.3 `master_floor_profile(jp)` (line 79)

Builds `fL`, the per-slot lower bound on the master carry for the M6 reserve floor `F_min`. If
`master_floor_J` is `None` it returns zeros, which gives the original `f >= 0` constraint exactly.

The design problem: a flat floor `f >= F_min` everywhere could be infeasible. At the start the
master holds nothing, and it may need everything it harvests to feed the slave. So the floor
**ramps up** as the master can afford it:

```
need_k  = the just-in-time delta of slot k (what the slave is short of)
S_k     = min(capF_k, max(0, S_{k-1} + E_k - need_k))        forward: fullest the master could be
D_k     = max(0, D_{k+1} + need_{k+1} - E_{k+1})             backward: future need that future harvest can't pay
fL[k+1] = min(F_min, max(0, S_k - D_k))                       floor = what can be held without starving a later need
fL[N]   = fL[N-1 slot]                                         the master ENDS holding its reserve
```

**Priority order.** The slave's minimum need first, the master's reserve second, everything else
(sharing, spill) from what is above the floor. `F = S` is itself feasible and satisfies `F >= fL`,
so the floor can never make a feasible problem infeasible.

### 9.4 `_lp_start(jp)` (line 114)

A feasibility LP solved with scipy `linprog` (`method="highs"`) and a **zero objective**: it only
looks for some feasible point.

**Variables** (length `3N`): `x` (master data energy), `d` (delta), `xb` (slave data energy).

**Constraints** are written with cumulative-sum matrices. `L` is lower-triangular ones (sum up to
`k`); `Lm` is strictly lower (sum up to `k-1`).

| row block | constraint |
|---|---|
| `L x + L d <= cumsum(E) - fL` | M1/M6: the master never spends more than it has harvested, minus its floor |
| `-L(a d) + L xb <= b0 - B_min - cumsum(c)` | S1: slave floor |
| `L(a d) - Lm xb <= B_max - b0 + Lm c` | S2: slave ceiling at arrival |
| `-L x - L d <= E_max - E_{k+1} - cumsum(E)` (finite `E_max` only) | M2 |
| `sum(x + d) = sum(E) - fL[N]` | the master uses all but its end floor |
| `sum(a d) - sum(xb) = B_min - b0 + sum(c)` | slave terminal `g[N] = 0` |
| bounds | `xb = 0` in silent slots (S4); `d <= capA/a` (S5); `d <= Dc` (M5); `xb in [txfloor, txcap]` (S6/S7) |

The LP solution is then converted back into flows:

- `f[1:] = max(cumsum(E - x - d), fL[1:])`, with `f[N] = fL[N]`;
- `g[1:] = max(g_init + cumsum(a d - xb - c), 0)`, with `g[N] = 0`.

The clamps remove LP round-off. Returns `(d, f, g)`, or `None` if the LP is infeasible or scipy is
missing.

### 9.5 `warm_start_from(jp, r)` (line 171)

Converts a previous `JointResult` back into `(f, g, d)` flows for `x0`: `f[1:N] = r.F[:-1]`,
`g[1:N] = r.b[:-1] - B_min`, `d = r.delta`, with the fixed ends re-imposed. It is used by the refusal
loop: after one event changes, the previous solution is usually nearly optimal, so warm starting
saves most of the sweeps.

### 9.6 `_warm_start_violation(jp, f, g, d, tol)` (line 184)

Returns `None` if `(f, g, d)` satisfies every bound to `tol`, otherwise a description of the first
violated bound. It checks:

- array lengths;
- the fixed ends;
- `f >= 0`, `f >= fL`, `f <= cap`, `g >= 0`, `d >= 0`;
- `P >= 0`, `Pbar >= 0`, `Pbar = 0` in silent slots;
- the arrival cap, S5, S6, S7 and M5.

A warm start from a *different* `jp` (another schedule or another beta) often fails one of these,
so this gate is essential. Without it, the solver could start infeasible and never recover.

---

## 10. `JointParams` and `JointResult`

**File:** `dwf2d/params.py`

### 10.1 `JointParams`: inputs, validation and derived arrays

**Fields:**

- `master` (`MasterParams`: `E`, `E_max`, `tau`, `h`, `N`, `cap`);
- `alpha`, `beta`, `B_min`, `B_max`, `b0`, `h_bar`, `theta1`, `theta2`, `slave_schedule`;
- `slave_overhead_J` (`c_i`), `mppt_cap_J` (S5);
- the stage-6 fields `tx_cap_J` (S6), `tx_floor_J` (S7), `delta_cap_J` (M5), `slave_rate_window`
  (`T_i`) and `master_floor_J` (M6).

**`__post_init__` is critical.** It is where the model's assumptions are enforced:

- broadcasts scalar `alpha`, `beta` and `h_bar` to length `N`;
- defaults `b0 = B_min` (the slave starts with zero usable energy);
- validates `0 <= B_min < B_max`, `B_min <= b0 <= B_max`, `alpha` and `beta` in `[0, 1]`, and
  `h_bar > 0`;
- turns `slave_schedule` into a boolean array, repeating a short pattern cyclically (`[1, 0]` means
  every other slot);
- rejects a schedule with no transmit slot when `b0 > B_min`, because that energy could never be
  spent;
- validates the per-slot stage-6 fields, and rejects `tx_floor_J > 0` in silent slots and
  `tx_floor_J > tx_cap_J`;
- checks `0 <= master_floor_J <= E_max - max(E[1:])`, so the floor fits under every M2 cap.

**Derived properties.** The solver reads these instead of the raw fields:

| property | value | used for |
|---|---|---|
| `a` | `beta * alpha` | net down-tap gain |
| `W` | `B_max - B_min` | usable slave window (upper bound of `g + a d`) |
| `g_init` | `b0 - B_min` | `g[0]` |
| `overhead` | `c_i`, or zeros | slave drain |
| `capA` | `mppt_cap_J`, or inf | S5 |
| `txcap` / `txfloor` | S6 cap (inf) / S7 floor (0) | burst bounds |
| `dcap` | `delta_cap_J`, or inf | M5 |
| `Twin` | `slave_rate_window`, or `tau` | rate window |

Because the solver reads only these properties, a wrong value here produces a wrong answer that
**still converges cleanly**. Errors in `JointParams` are the hardest kind to spot.

### 10.2 `JointResult`

The output record; see the table in section 6.5. Two notes:

- With `theta1 = 0` (the stage-6 master has no data link), `P` is **not data power**. `tau*P_i` is
  the master's unused or spilled energy, and `rate_master` is meaningless.
- `start` tells you which start path was used. `"lp-fallback"` or a rejected warm start is a hint
  that the problem is tight.

---

## 11. Verification: is the answer actually optimal?

**File:** `dwf2d/verification.py`

The solver's stopping rule is "no move I know improves". That is only equivalent to optimality if
the move set is rich enough. These functions check it from outside.

### 11.1 `check_kkt_joint(jp, r)` (verification.py:14)

Takes a finished result and returns a dictionary of **feasibility** and **down-tap optimality**
diagnostics.

**Marginal values from the result:**

```
mk = th1*h  / (2(1 + h P))
sk = th2*hb / (2(1 + hb Pbar))              (burst form with a rate window)
gap_k = (mk - a_k sk) / mk                  (absolute gap when th1 = 0)
```

**The down-tap test** comes from the stationarity condition `-Lambda_k + a_k Gamma_k + rho_k = 0`
with `rho_k >= 0` and `rho_k delta_k = 0`:

- `delta_k > 0` (`rho_k = 0`) and no cap binding: `gap_k ≈ 0`. This is
  `delta_cond_violation_active`.
- `delta_k = 0`: `gap_k >= 0`, so sending more would not pay. This is
  `delta_cond_violation_inactive`.

Slots are **excluded** from the equality test when the arrival cap, S5, S6 or M5 binds there,
because the bound's multiplier legitimately breaks the equality. Silent slots are excluded as well,
because their energy is valued at a later slot. With `theta1 = 0`, slots where the master is out of
energy are also excluded, because there the master's own budget stops `delta`.

**Feasibility and bookkeeping keys:**

- `final_master_carry`, `slave_end_level`;
- `min_P`, `min_Pbar`, `min_delta`;
- `slave_min_level`, `slave_max_level`, `min_master_carry`;
- `max_Pbar_in_silent_slots`, `max_cap_violation_J`, `min_slave_level_margin_J`;
- `slave_ledger_error_J`, an independent energy ledger of the slave
  (`b0 + sum(a delta) - sum(c) - sum(tau Pbar) - b_end`), which should be about `1e-9`;
- counts of slots where each bound binds;
- stage-6 extras: S6, S7, M5, the M6 floor margin, and M2 "master full".

**Limitation.** It tests only the **down-tap** condition. It does not test the right-tap
water-level conditions (`nu_k = nu_{k+1}` where the master tap is open, and the same for the
slave). A result can pass `check_kkt_joint` and still be suboptimal in how it distributes energy
over time. For a full optimality check, use `reference_solution_joint`.

### 11.2 `reference_solution_joint(jp)` (verification.py:148)

An **independent** solve of the same convex problem with cvxpy and the CLARABEL interior-point
solver. It shares no code with `joint_dwf()` (only `master_floor_profile`), so agreement between
the two is a genuine test.

- **Variables:** `x = tau*P`, `dl = delta` and `xb = tau*Pbar`, all non-negative.
- **Constraints:** written in cumulative-sum form, exactly as in `_lp_start` but with the real
  objective: master causality and floor, master uses all, master ceiling, slave floor, slave
  ceiling at arrival, slave terminal `= B_min`, S4, S5, S6, S7, M5.
- **Objective:** the weighted sum of `log` rates; the burst form if a window is set.
- **Returns:** `(P, Pbar, delta, objective)`. Raises `InfeasibleSlaveBudget` if cvxpy finds the
  problem infeasible.

Compare `r.objective` with the reference objective. Because the problem is strictly concave in the
powers, the powers `P` and `Pbar` are unique. The split of carried energy between the two batteries
is **not** unique, so compare objectives and powers, not `F` or `b`.

### 11.3 `random_parity_case()` and `parity_matrix()` (verification.py:230, 300)

The regression gate. `random_parity_case(rng, family)` generates a random small problem (`N` of 8 to
14) from one of 11 families:

`default`, `tight_window`, `per_slot_beta`, `finite_E_max`, `s6_binding`, `s6_s5_binding`,
`rate_window`, `theta1_zero`, `integrated_like`, `master_floor`, `harvest_gap`.

Each family is built so that a particular bound actually binds.

`parity_matrix()` runs `joint_dwf` and `reference_solution_joint` on `n_cases` problems per family
and classifies each case by the relative gap:

- **exact:** the gap is at most `rtol`.
- **below:** the DWF is suboptimal. This is not rare. On the `default` family, 23 of 25 cases are
  below, with gaps up to 7.3%, mostly because the master-carry delta shift move is missing (README
  open item 17; see `TESTS.md` section 4).
- **above:** the DWF *beats* the reference. This **must never happen**: it would mean the DWF broke
  a constraint, or cvxpy failed.

It also counts non-converged and infeasible cases, and how often each bound binds. **Run this
after any change to the moves or bounds.**

---

## 12. Physical receive chain and beta calibration

**File:** `dwf2d/physical.py`

`joint_dwf()` assumes the down tap is **linear**: `stored = beta * alpha * delta`. A real
photodiode + MPPT chain is nonlinear, and its efficiency depends on the incident power. These
functions close that gap without touching the solver.

### 12.1 `physical_stored_power(jp, r, ...)` (physical.py:42)

Pure post-processing; nothing is fed back to the solver.

1. Feed `r.arrived_power` (`alpha*delta/tau`, the optical power at the slave) into
   `physical_chain.optical_to_electrical()`. It returns the photocurrent, `Voc`, the maximum
   electrical power and the MPPT output.
2. `stored_raw = mppt_output_power * supercap_eff`
3. `eff_phys = stored_raw / arrived_power`: the real secant gain of the chain, slot by slot.
4. Clip the stored energy at the MPPT cap `capA`. The excess goes to `overflow_mppt`, because
   `physical_chain` does not saturate by itself.
5. **Re-simulate the slave battery** with the same spending (`tau*Pbar`) and drain (`c`) as the
   solver, but the *physical* arrivals. Anything above `B_max` is recorded as `overflow` and
   clipped.
6. Count floor and ceiling violations with the tolerance `battery_tol_J`.

`battery_tol_J` must be looser than the solver's own convergence residual (`tol * scale`).
Otherwise slots sitting exactly on S1 or S2 read as false violations.

### 12.2 `solve_with_physical_chain(jp, ...)` (physical.py:130)

A **fixed-point iteration on beta**:

```
beta = jp.beta
repeat up to max_outer:
    r  = joint_dwf(jp with beta)
    ph = physical_stored_power(jp, r)
    on = slots where delta > 0
    beta_new[on] = ph.eff_phys[on]          # replace the assumed gain by the measured one
    (optionally) beta_new[~on] = median(eff_phys[on])
    stop when max |beta_new - beta| < tol
```

**Why it converges.** The chain's gain varies only slightly with the operating point, so the map
`beta → eff_phys(joint_dwf(beta))` is a contraction in practice. It typically takes a handful of
iterations.

**At the fixed point**, `a_i * delta_i / tau` equals the real MPPT output, so the solver's battery
`r.b` matches the physically re-simulated `ph.b_phys`. The `B_min`/`B_max` constraints are then
enforced against **real** energy.

Slots with `delta = 0` give no information (`eff_phys` is NaN there), so their beta is kept. With
`fill_idle_beta=True` they get the median gain instead, which matters for the maximal-sharing
post-pass, which may start sending in those slots. A warm start `x0` is used only in the first outer
iteration.

Returns `(jp_calibrated, r, ph, n_outer)`.

---

## 13. Stage 6: events, refusals and maximal sharing

### 13.1 `fullest_trajectory(jp, x)` (solver.py:762)

A forward greedy pass. With the slave burst energies `x` fixed, the master sends in each slot as
much as possible: the minimum of the remaining battery headroom `(B_max - level)/a`, the MPPT cap
`capA/a`, the laser cap `Dc`, and its own available energy `E_k + carry`. With M6, it shares only
what is above the floor.

It records:

- `delta`, `spill`, `b` and `b_peak`;
- **which bound stopped the master** in each slot (`stop`: `ceiling`, `S5`, `M5`, `master`,
  `floor`, `full`);
- `fail_slot`: the first slot where even this fullest trajectory drops below `B_min`.

With `E_max = inf` this trajectory is pointwise the fullest possible, so `fail_slot is None` is an
**exact feasibility test** for a given set of bursts. `solve_with_refusals` uses it with `x = 0` to
find which slot makes an event schedule unaffordable.

### 13.2 `maximal_sharing(jp, r)` (solver.py:811)

A tie-break used only when `theta1 = 0`. When master energy is worthless, many `delta` schedules
give the same objective. This pass keeps every slave burst `x_i = tau*Pbar_i`, so the objective,
events and `Pbar` are unchanged, and picks the schedule with the **fullest slave supercap**: the
master shares as early and as much as the slave can take.

The solver's terminal condition `g[N] = 0` is dropped: the slave may end above `B_min`. The pass is
not applied if the fullest trajectory would fall below `B_min`, which happens when the optimum
deliberately kept master energy for later.

Returns `(r_new, info)`, where `info` holds `applied`, the per-slot `stop` reasons and, if not
applied, a reason.

### 13.3 `solve_with_refusals(build_jp, schedule, x_min, max_delay, ...)` (physical.py:234)

The outer **event-refusal loop**, which mirrors the hardware: wake, check the energy, refuse if
short.

```
effective = released events
repeat (at most #events*(max_delay+1)+1 passes):
    solve  (solve_with_physical_chain(build_jp(effective)), beta carried over, warm start)
    if it raised InfeasibleSlaveBudget:
        k0 = fullest_trajectory(jp, 0).fail_slot  (or e.slots if the floor is reachable)
        refuse the latest event at or before k0   (none -> stop: the base sleep load is unaffordable)
    else:
        refuse every run event whose burst x_i < x_min   ("proportional" mode)
        none refused -> converged, stop
    slave_load.reschedule: each refused event moves to the first free later slot within
        max_delay of its RELEASE, otherwise it is dropped
```

It terminates because every pass moves some event strictly later or drops it, and delays are
bounded. It is a **greedy rule, not a combinatorial optimum**. `brute_force_events()` exists to
measure how far from optimal it is.

Every released event ends as `run`, `rescheduled`, `dropped`, or `unresolved` (if the loop hit its
pass cap). A bookkeeping assertion guarantees that no event disappears.

Returns a `RefusalResult`: `jp`, `r`, `ph`, `events`, `effective`, `passes`, `converged`, `history`,
and the `counts` property.

### 13.4 Stage-6 verification

All in `verification.py`.

| function | what it checks |
|---|---|
| `brute_force_events()` | Tries every placement of every event (each one in `[release, release + max_delay]` or dropped). Scores are compared as `(events run, objective)`. This is the ground truth for the greedy loop on small problems. |
| `max_level_lp()` | LP that maximises `sum b_k` over `delta` with the bursts fixed. An independent check that `maximal_sharing`'s greedy pass really is the fullest trajectory. |
| `check_maximal_sharing()` | Checks that the post-pass left the objective and every burst unchanged; that each slot was stopped by exactly one bound; that `delta = 0` where the battery arrived full; that the level stays inside `[B_min, B_max]`; and that the result agrees with the LP. |
| `refusal_gate()` | Runs the whole stage-6 pipeline over the scenario kinds `scarce`, `zero_harvest`, `tiny_supercap`, `all_on`, `single_on` and `fixed_bias`, and lists every broken invariant. |
| `stage6_checks()` | Checks that the THz presets differ only in `h_bar`. Checks monotonicity: the objective must not fall when `x_cap`, `B_max` or `beta` rise, and must not rise when the THz distance or the slave load rise. Also checks the slave ledger and the silent-slot `Pbar`, and times scalability. |

---

## 14. Things worth knowing (observations from reading the code)

These are observations, not confirmed bugs. None of them affects results on the code paths that
are currently exercised. They are worth knowing before you modify the solver.

1. **`step()` is only correct for `s = 1`.** For general `s` the denominator should be
   `h_u h_w r s (th_u + th_w)` and the second numerator term should carry a factor `s`. The code
   has `den = hu*hw*(thu*r_ + r_*thw*s_)` and no `s` in the numerator, and the docstring's
   cross-multiplied equation also drops the `s` on the left. Every call site passes `s_ = 1`, so
   results are correct today. If you ever add a move whose source loses `s != 1` Joules per unit,
   fix the formula first.

2. **At `theta1 = 0`, master-to-master moves are inert.** `step()` returns 0 when `den <= 0`.
   With both ends being master nodes and `th1 = 0`, that applies to moves (6), (8) and (10) and to
   `polish()` families A and D, which never move. This is logically consistent, because moving
   worthless energy between master slots gains nothing. It means master energy reaches the slave
   only through moves whose sink is a slave node: (1)–(5) and family C.

   Family C cannot pass *through* a live master slot. If you ever see a `theta1_zero` or
   `master_floor` parity case come out "below", this is the first place to look.

3. **The list of moves in the docstrings is out of date.** The module docstring in
   `build_2D_dwf.py` lists moves (1)–(9) and three `polish()` families. The code also has move (10)
   and family D. The `joint_dwf` docstring also says the start is `f = 0`, while the code starts at
   `f = fL` (identical without a master floor).

4. **There is a `print` on every sweep** (line 576). The legacy demo takes 1410 sweeps, and the
   refusal loop calls `joint_dwf` many times, so this floods stdout. It may also be part of the
   `tests/baseline/*/stdout.txt` snapshots, so check the baselines before removing it.

5. **`check_kkt_joint` checks only the down-tap condition.** See section 11.1. Passing it is
   necessary for optimality but not sufficient; `reference_solution_joint` is the real test.

6. **The battery split is not unique.** When both batteries have slack, energy can be carried in
   the master battery or the slave battery with the same objective. `F` and `b` can therefore
   differ between the DWF and cvxpy, or between runs with different start points, while `P`,
   `Pbar` and the objective agree.

7. **The cost of `polish()` grows with N.** Family D is `O(N^3)` per call and family C is
   `O(N * run_length^2 * path_length)`. On long horizons (`stage6_checks` scales to `n_periods`),
   `polish()` dominates the run time whenever the sweeps go quiet often. Full analysis in
   [section 15](#15-computational-complexity).

8. **A move is missing: the master-carry delta shift** (README open item 17). The direction is
   `d[i] -= t, f[i+1..k] += t, d[k] += t`: send less at slot `i`, carry it in the master battery,
   and send it at a slot `k` where `a` is higher.

   When the master spends nothing on its own data (`P = 0` everywhere), no single existing move
   produces this direction. The nearest is move (4) run backwards followed by move (1), and the
   first of those two steps loses rate on its own, so `step()` refuses it. The solver then stops
   2–9% below the optimum and still reports `converged=True`. A large
   `check_kkt_joint(...)["delta_cond_violation_active"]` (3–6 instead of about 0) gives this away.
   Measurements are in `TESTS.md` section 4.

---

## 15. Computational complexity

**Notation:**
- N = number of slots;
- K = number of sweeps, the main-loop iterations;
- P = number of `polish()` calls;
- L = length of a run of consecutive dead slots (data energy ≈ 0, as tested by `dead()`).

All bounds are from reading `dwf2d/solver.py`. Section 15.6 checks them against measured runs; the
runs themselves are documented in `TESTS.md`.

### 15.1 Summary

| function | time | what drives it |
|---|---|---|
| `step()` | **Θ(1)** | a fixed formula |
| `apply()` | **Θ(m), at most O(N)** | m = number of variables in the path |
| one sweep | **Θ(N)** | N slots × 10 moves of constant cost |
| `polish()` | **Θ(N) at best, O(N⁴) at worst**; **at least Θ(N³) with a slave load** | dead-slot runs; family D |
| `joint_dwf()` | **T_start + K·Θ(N) + P·C_polish(N) + Θ(N)**, with 1 ≤ P ≤ K ≤ `max_iter` | K, which has no polynomial bound |
| guaranteed worst case | **O(max_iter · N⁴)** | the iteration cap |
| memory | **O(N)**, or O(N²) when the LP fallback runs | |

### 15.2 `step()`: Θ(1)

`step()` computes one numerator and one denominator, divides them, and has one branch for the rate
window. It has no loops and does not depend on N.

### 15.3 `apply()`: Θ(m), at most O(N)

Let m = `len(fch) + len(gch) + len(dch)`, the number of variables the move changes. `apply()` makes
four passes over them:

1. one bound per variable;
2. one dictionary update per `g` or `d` entry, to combine the arrival-cap coefficients;
3. one arrival-cap bound per slot in that dictionary, plus the S5 and M5 bounds per `d` entry;
4. one update per variable.

Each pass is O(m), and dictionary operations are O(1) on average. A path never touches more than
about 2N+1 variables: polish family C, with a master chain, the down tap and a slave chain. So one
call costs at most O(N).

### 15.4 One sweep: Θ(N)

The loop body visits every slot once and tries moves (1)–(10) there.
- Moves (1)–(9) are written inline: one `step()` call and a fixed list of bounds, so each is O(1).
- Move (10) calls `apply()` on a path of 3 variables, also O(1).

The work per slot is a constant that does not depend on the data. A move whose condition is false
(for example, a silent slot) just costs less. So one sweep is **Θ(N)**, which is both a lower and an
upper bound.

### 15.5 `polish()`: Θ(N) to O(N⁴)

| family | number of `apply()` calls | path length per call | total work |
|---|---|---|---|
| **D**, delta shift (only when `cross_run` is on, which by default means there is a slave load) | every pair (k, j) with j ≥ k+2 and a > 0: about N²/2 | j − k + 2 | **Θ(N³)**, about N³/6, **whatever the data** |
| **A**, master chain | for each start slot i, one call per dead slot after it: L_i | up to L_i + 1 | Θ(Σ L_i²): **Θ(N)** with no dead slots, **O(N³)** if almost all master slots are dead |
| **B**, slave chain | the same with dead slave slots, and only between transmit slots | up to L_i + 1 | Θ(N) to O(N³) |
| **C**, cross paths | at each down tap k: (number of sources) × (number of sinks) | sources + sinks | Θ(N) to **O(N⁴)** |

**Why family C can reach N⁴.** At down tap `k`:
- the list of sources grows to the left and to the right through dead master slots;
- the list of sinks grows the same way through dead slave slots;
- each list stops just after the first live slot.

When nearly every master slot and nearly every slave slot is dead, both lists have about N
entries. That gives about N² pairs per tap, each with a path of about N variables, so about N³ work
per tap and **N⁴ over the N taps**. Building the source and sink lists themselves costs another
O(N²) per tap.

Not every slot can be dead, because the harvest has to be spent somewhere. But one live slot on each
side is enough, so the N⁴ bound can be reached in principle.

**Cost of one `polish()` call, overall:**
- **without a slave load:** Θ(N) at best (no dead slots), O(N⁴) at worst;
- **with a slave load:** at least Θ(N³) because of family D, and O(N⁴) at worst.

### 15.6 Check against measured counts

In the test problems of `TESTS.md`, the master spends nothing in any slot. So **every master slot is
dead** and every slave slot is live. That gives:

- family C: up to about N sources per tap, but at most 3 sinks (the slave lists stop at the first
  live slot);
- family A: every start slot extends to the end of the horizon;
- family B: no calls.

Predicted number of `apply()` calls in one `polish()` call at N = 40:

| family | count | predicted calls |
|---|---|---|
| D | Σ_{k=0}^{37} (38 − k) | 741 |
| C | ≈ 40 taps × up to 40 sources × 3 sinks, less the pairs left to the sweep | ≈ 4,800 |
| A | Σ_i (N − 2 − i) | ≈ 740 |
| B | slave slots are live | 0 |
| **total** | | **≈ 6,280** |

**Measured (run C in `TESTS.md`, N = 40, seed 0):**

| quantity | value |
|---|---|
| `apply()` calls in total | 763,896 |
| calls from move (10): 39 per sweep × 2,645 sweeps | 103,155 |
| remaining calls, all from `polish()` | 660,741 |
| `polish()` calls | 105 |
| **`apply()` calls per `polish()` call** | **6,293** |

The prediction and the measurement agree to within 0.2%. The same check at N = 10 predicts 36 + ≈ 300
+ ≈ 36 calls per `polish()` call, and run C measured 369.

**Why the measured time looked like N² and not N³.** In this regime one `polish()` call makes Θ(N²)
`apply()` calls, each with a path of Θ(N), so the true cost is Θ(N³). At N ≤ 40 the paths are short,
so the fixed overhead of each Python call (building the lists, the dictionary, the function call
itself) is larger than the per-variable work. The measured time per `polish()` call therefore grew
about as N² (0.016 s at N = 10, 0.27 s at N = 40). The N³ term should show at larger N; the crossover
has not been measured.

### 15.7 The whole algorithm

```
T(N) = T_start + K·Θ(N) + P·C_polish(N) + Θ(N)            1 <= P <= K <= max_iter
```

| term | cost |
|---|---|
| `T_start`, share-nothing / warm / just-in-time | Θ(N): `master_floor_profile`, `_jit_start`, `_warm_start_violation` are single passes |
| `T_start`, LP fallback | O(N²) time and memory to build the dense N×N cumulative-sum blocks, plus the HiGHS LP solve (polynomial; not bounded or measured here) |
| sweeps | K · Θ(N) |
| `polish()` | P · C_polish(N), with C_polish from section 15.5. `polish()` runs at most once per sweep, and only after a sweep that moved nothing, so P ≤ K. When the solver converges, P ≥ 1 |
| post-processing | Θ(N) |

**Worst case per iteration:** one sweep plus one `polish()` call costs O(N) + O(N⁴) = **O(N⁴)**.

**The number of sweeps K has no polynomial bound.**
- Coordinate ascent converges linearly at best, and only for smooth problems with a unique optimum.
  This problem has neither: its bounds bind, which makes it non-smooth, and the split of stored
  energy between the two batteries is not unique.
- Even the easiest case, spreading energy evenly over a run of L slots with no bound active, behaves
  like Gauss-Seidel iteration on a chain. That needs about L² sweeps for each digit of accuracy, so
  for a run as long as the horizon, K grows like N² · log(1/tol) even in this best case.
- Bounds that bind make the solver zig-zag. The missing move (section 14, item 8) can make it stall
  short of the optimum, using sweeps without making progress.

So the only guaranteed bound on the whole algorithm comes from the iteration cap:

```
T(N) = O(max_iter · N⁴)        worst case
```

With `max_iter` fixed (default 1000) that is formally O(N⁴), but **within that bound the solver is not
guaranteed to have converged**. The time to actually reach the tolerance is K(N, tol, data) × the
per-iteration cost, and it is data-dependent.

**Measured** (`TESTS.md`, random problems, N = 10–40):

| quantity | growth |
|---|---|
| K | roughly N^3.7–N^3.9 |
| P | 1 at N ≤ 20, about 30–200 at N = 30–40 |
| total time | roughly N⁵ |

These exponents describe those test problems, not the algorithm in general.

**Memory:** O(N) for the arrays `f`, `g`, `d` and the inputs. O(N²) when the LP fallback runs, for
its dense matrices.

### 15.8 The wrappers around `joint_dwf()`

| function | cost |
|---|---|
| `check_kkt_joint`, `fullest_trajectory`, `maximal_sharing`, `physical_stored_power` | Θ(N) each |
| `solve_with_physical_chain` | `n_outer` × T(N), with `n_outer` usually a handful and at most `max_outer` = 30 |
| `solve_with_refusals` | up to E·(D+1)+1 calibrated solves, each `n_outer` × T(N) (E = number of events, D = `max_delay`) |
| `brute_force_events` | about (D+2)^E solves: **exponential in the number of events** |
| `reference_solution_joint` (cvxpy) | one interior-point solve, polynomial. Measured about 0.02 s at N ≤ 40, against up to two minutes for `joint_dwf()` on the same problems (`TESTS.md` section 4.1) |
| `parity_matrix` | 25 × (number of families) × (T(N) + one cvxpy solve) |

### 15.9 Best-case and worst-case time for one solve

This is the time for one `joint_dwf()` call (one simulation) to return an optimal response.

| | in theory | measured, N = 10–40 (one solve, `TESTS.md`) |
|---|---|---|
| **Best case** | **Θ(N)**: one sweep and one `polish()` call, when the start point is already optimal. With a slave load it is at least **Θ(N³)**, because `polish()` family D always runs. | about **0.02–0.03 s** (N = 10, 9–10 sweeps) |
| **Worst case, until the solver stops** | **O(max_iter · N⁴)**: each iteration is one sweep, O(N), plus one `polish()` call, up to O(N⁴), until the iteration cap stops it | **116 s** (N = 40, 14,023 sweeps, with a slave load), and that result is still 3.1% below the optimum |
| **Worst case, until the optimum is reached** | **No finite bound.** When the master gives the slave all its energy, the solver can stop below the optimum and never get there (the missing move, section 14 item 8). | never reached: all 12 problems with a slave load stopped 1.8–9.2% below the optimum, after 0.02–116 s |

The measured times come from runs under `cProfile`, which made them about 1.5× slower. Without the
profiler, the 116 s worst case would be roughly 75 s.

**What this means in practice.** With `joint_dwf()` as it is now, the honest worst-case answer is
that it is **not guaranteed to reach the optimum at all**. It only guarantees that it stops, after
at most `max_iter` iterations, with `converged=False` if it hits the cap. And `converged=True` does
not prove the result is optimal: check `check_kkt_joint(...)["delta_cond_violation_active"]`, which
was 3–6 instead of about 0 in exactly these suboptimal cases.

**When a guaranteed optimum is needed** for each simulation, `reference_solution_joint()` (cvxpy,
interior point) solved the same problems to the optimum in about 0.02 s at N = 40. It has not been
measured at larger N.

---

## 16. Quick reference

### Call map

| function | file:line | called by | role |
|---|---|---|---|
| `joint_dwf` | solver.py:208 | demos, calibration, refusals, gates | the solver |
| `step` (nested) | solver.py:329 | every move, `polish` | closed-form line optimum |
| `apply` (nested) | solver.py:405 | move (10), `polish` | clip to bounds and move |
| `polish` (nested) | solver.py:469 | main loop when a sweep is quiet | long pass-through moves |
| `slave_lim`, `sx`, `dead` (nested) | solver.py:380, 395, 401 | moves, `polish` | S6/S7 bounds, dead-slot test |
| `_jit_start` | solver.py:24 | `joint_dwf`, `master_floor_profile` | just-in-time feasible start |
| `master_floor_profile` | solver.py:79 | `joint_dwf`, LP, cvxpy, KKT | M6 floor `fL` |
| `_lp_start` | solver.py:114 | `joint_dwf` | LP feasible start |
| `warm_start_from` | solver.py:171 | refusal loop | result → `(f, g, d)` |
| `_warm_start_violation` | solver.py:184 | `joint_dwf` | warm-start gate |
| `fullest_trajectory` | solver.py:762 | `maximal_sharing`, refusal loop | greedy fullest supercap / feasibility test |
| `maximal_sharing` | solver.py:811 | stage-6 demos, gate | `theta1 = 0` tie-break |
| `JointParams.__post_init__` | params.py:92 | construction | validation and broadcasting |
| `check_kkt_joint` | verification.py:14 | demos, gates | feasibility + down-tap KKT |
| `reference_solution_joint` | verification.py:148 | parity gate | independent cvxpy optimum |
| `parity_matrix` | verification.py:300 | manual / tests | DWF vs cvxpy over 11 families |
| `physical_stored_power` | physical.py:42 | calibration | real-chain re-simulation |
| `solve_with_physical_chain` | physical.py:130 | demos, refusal loop | beta fixed point |
| `solve_with_refusals` | physical.py:234 | integrated demo, gate | event refusal / reschedule loop |

### The core equation, once more

```
t* = tau * [ r th_w h_w (1 + h_u x_u/tau) - th_u h_u (1 + h_w x_w/tau) ] / [ h_u h_w r (th_u + th_w) ]
t  = clip(t*, lo, hi)          [lo, hi] = every linear bound of section 1.4 along the move
```

### The whole algorithm in five lines

1. Pick a feasible start: share-nothing, warm, just-in-time, or LP.
2. Sweep the slots, alternating direction. At each slot, try ten moves, each `clip(step(...))`.
3. When a sweep is quiet, try `polish()`: longer paths through dead slots.
4. When `polish()` is quiet too, stop. No known direction improves, which means the KKT conditions
   hold.
5. Convert flows to powers, batteries and rates. Verify with `check_kkt_joint` and, when in doubt,
   `reference_solution_joint`.
