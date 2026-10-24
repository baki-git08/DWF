# Tests of `joint_dwf()`: run-time scaling and optimality

This file records experiments on the solver `joint_dwf()` (`dwf2d/solver.py`):

- how its run time grows with the number of slots N, without and with a slave load;
- whether the results it returns are optimal, checked against the independent cvxpy solution
  `reference_solution_joint()`.

It sets out what was run, the raw results, what they show, and the limits of the experiments.

Run on 2026-10-22.

---

## Summary of findings

1. **One sweep is cheap and grows linearly:** 1–4 ms for N = 10–40. **The number of sweeps is the
   problem.** It grew roughly as N^3.7–N^3.9 on these random problems, and total time roughly as
   N^5. That holds with and without a slave load.
2. **Attaching a slave load hardly changes the run time** at N ≤ 40. `polish()` family D (Θ(N³)),
   which runs only with a load, does not yet dominate at these sizes.
3. **One problem (N = 30, seed 2) never converged**, in both runs, even with 20,000 sweeps.
4. **With the default `max_iter=1000`, every N = 40 problem would stop early** and return
   `converged=False`.
5. **The results are 1.7–9.2% below the true optimum in every slave-load case**, including all the
   ones that converged. The no-load cases checked show the same gaps. This is not a bug in the test
   setup and not caused by uncommitted changes:
   - the repo's own `parity_matrix` gives the same picture (2/25 exact);
   - the committed code gives identical numbers;
   - the README already records the cause as open item 17, a **missing move: the master-carry delta
     shift**.

---

## 1. Why the experiments were run

Reading the code gives the cost of **one** sweep and **one** `polish()` call (see
[section 6](#6-expected-cost-from-the-code) and `CRITICAL_FUNCTIONS_DWF.md`). It cannot give the
**number** of sweeps, because block coordinate ascent has no general bound on how many sweeps it
needs. That has to be measured.

The slave-load runs were added because a load switches on `polish()` family D, which is Θ(N³). The
optimality check was added because a solver that stops is only useful if it stops at the optimum.

---

## 2. Setup

### 2.1 Environment

| item | value |
|---|---|
| OS | Windows 11 Home, 10.0.26300 |
| CPU | Intel64 Family 6 Model 154 (12th-gen Core) |
| Python | 3.12.4 |
| NumPy / SciPy / cvxpy | 2.5.2 / 1.18.1 / 1.9.3 (cvxpy solver: CLARABEL) |
| code | branch `Adv_DWF_slave_profile`, commit `4dcd561`, **plus uncommitted working-tree changes** in `dwf2d/` |

Runs A–D used the working tree. Run E used a clean copy of the commit `4dcd561` (HEAD), extracted
with `git archive` into a temporary folder so the working tree was not touched.

### 2.2 Test problems

These are random problems built like the `"default"` family of `verification.random_parity_case()`,
but with N chosen freely instead of drawn from 8–14. For a given N and seed:

| parameter | value | meaning |
|---|---|---|
| `E` | `U(2, 40)` J per slot | master harvest |
| `h` | `U(0.02, 0.3)` per slot | master link gain |
| `h_bar` | `h * U(1.5, 6)` per slot | slave link gain (always better than the master's) |
| `alpha` | `U(0.6, 1)` per slot | water transmittance |
| `beta` | 0.8 | recharging efficiency, so `a = 0.8*alpha`, in 0.48–0.8 |
| `B_min`, `B_max` | 1 J, `1 + 2*sum(E)` J | a very large slave window: the battery ceiling S2 practically never binds |
| `theta1`, `theta2` | 1, 2 | the slave rate counts double |
| `tau` | 10 s | slot length |
| `E_max` | inf | no master battery cap |
| S5–S7, M5, M6, rate window | off | |
| schedule | all slots transmit | no silent slots |
| **slave load** `c_i` (runs C and D only) | `U(0.05, 0.5)` J per slot | non-TX slave electronics drain |

The slave load is drawn **after** the other parameters from the same random generator. So the
problem "N, seed s, with load" is exactly the problem "N, seed s, no load" plus a drain. The total
drain is 2.6–12.4 J, about 1.3% of the total harvest. With a load, the solver starts from the
just-in-time point (`r.start == "just-in-time"` in every case), and `cross_run`, and with it
`polish()` family D, is switched on.

Every slot has its own random channel and transmittance, so the water levels jump around from slot
to slot. That is harsher than the demos, whose channels are smoother.

**Which regime this is.** The slave's link is 1.5–6× better and its rate counts double, so in these
problems the master spends nothing on its own data (`P = 0` in every slot) and sends everything it
harvests to the slave. This is the "master gives everything" regime. It matters for the optimality
result in [section 4](#4-optimality-results).

### 2.3 Solver settings

- `joint_dwf(jp, max_iter=20000)`. The default `tol = 1e-10` was kept. `max_iter` was raised from
  the default of 1000 so that convergence could be observed instead of cut off.
- The `print` that `joint_dwf()` makes on every sweep (line 576) was redirected to an in-memory
  buffer, so console output is not part of the timing.
- Time was measured with `time.perf_counter()` around the solver call only. Problem construction is
  excluded.

### 2.4 The runs

| run | what | N | seeds | profiler |
|---|---|---|---|---|
| A | timing only, no load | 20, 40 | 0, 1 | no |
| B | sweeps, `polish()` calls and time split, no load | 10, 20, 30, 40 | 0, 1, 2 | `cProfile` |
| C | as B, **with slave load** (also counts `apply()` calls and records start type and objective) | 10, 20, 30, 40 | 0, 1, 2 | `cProfile` |
| D | **optimality**: run C's objectives vs `reference_solution_joint()`; also 3 no-load cases with `check_kkt_joint()` | 10–40 | 0, 1, 2 | no |
| E | repo's own gate: `parity_matrix(n_cases=25, families=("default",))` on the **working tree** and on **HEAD** | 8–14 (random) | the gate's own | no |

Run A was meant to continue to N = 80 and 160, but it was stopped after N = 40 because a single
N = 40 solve already took about 30 s.

**Profiler overhead.** Run A gives about 31 s per N = 40 solve, against about 46 s for the same two
problems under the profiler in run B, so the profiler costs about 1.5×. **The absolute times in
runs B and C are inflated; the counts and ratios are not.**

### 2.5 Script

One script reproduces runs A–D. Run it from the project root (`advanced_DWF/`). Expect about
15 minutes in total; runs B and C take about 5 minutes each.

```python
import sys, io, time, contextlib, cProfile, pstats
import numpy as np
sys.path.insert(0, ".")
from build_dwf import MasterParams
from dwf2d.params import JointParams
from dwf2d.solver import joint_dwf
from dwf2d.verification import reference_solution_joint, check_kkt_joint


def case(N, seed, load):
    """Random 'default'-family problem with N slots (load=True attaches a per-slot slave drain)."""
    rng = np.random.default_rng(seed)
    E = rng.uniform(2, 40, N); h = rng.uniform(0.02, 0.3, N); hb = h * rng.uniform(1.5, 6, N)
    kw = dict(alpha=rng.uniform(0.6, 1, N), beta=0.8, B_min=1.0, B_max=1 + 2 * E.sum(), h_bar=hb,
              theta1=1.0, theta2=2.0)
    if load:
        kw["slave_overhead_J"] = rng.uniform(0.05, 0.5, N)
    return JointParams(master=MasterParams(E=E, E_max=np.inf, tau=10.0, h=h), **kw)


def solve(jp):
    with contextlib.redirect_stdout(io.StringIO()):      # silence the per-sweep print
        return joint_dwf(jp, max_iter=20000)


def run_a():
    """Run A: timing only, no load, averaged over seeds 0 and 1."""
    for N in (20, 40):
        its, ts = [], []
        for s in range(2):
            jp = case(N, s, False)
            t = time.perf_counter(); r = solve(jp)
            ts.append(time.perf_counter() - t); its.append(r.iters)
        print(f"N={N:4d}  sweeps {int(np.mean(its)):6d}  time {np.mean(ts):8.2f} s")


def run_profiled(load):
    """Runs B (load=False) and C (load=True): one line per (N, seed). Returns {(N, seed): objective}."""
    objs = {}
    for N in (10, 20, 30, 40):
        for s in range(3):
            jp = case(N, s, load)
            pr = cProfile.Profile()
            t = time.perf_counter()
            pr.enable(); r = solve(jp); pr.disable()
            T = time.perf_counter() - t
            st = pstats.Stats(pr).stats
            calls = {fn[2]: (v[1], v[3]) for fn, v in st.items() if fn[2] in ("polish", "apply")}
            pc, pt = calls.get("polish", (0, 0.0))       # (call count, cumulative time)
            ac = calls.get("apply", (0, 0.0))[0]
            objs[(N, s)] = r.objective
            print(f"N={N:3d} seed {s}: sweeps {r.iters:5d}  polish calls {pc:4d}  apply calls {ac:8d}  "
                  f"time {T:7.2f} s  in polish {pt / max(T, 1e-9):5.1%}  conv {r.converged}  "
                  f"start {r.start}  load {jp.overhead.sum():.3f} J  obj {r.objective:.6f}")
    return objs


def run_d(load_objs):
    """Run D: DWF objective vs the cvxpy optimum (load cases), then 3 no-load cases with the KKT check."""
    for (N, s), o in load_objs.items():
        ref = reference_solution_joint(case(N, s, True))[3]
        print(f"load N={N:3d} seed {s}: dwf {o:12.6f}  cvxpy {ref:12.6f}  rel gap {(ref - o) / abs(ref):+.2e}")
    for N, s in ((10, 0), (10, 1), (20, 1)):
        jp = case(N, s, False)
        r = solve(jp)
        ref = reference_solution_joint(jp)[3]
        k = check_kkt_joint(jp, r)
        print(f"no-load N={N} seed {s}: rel gap {(ref - r.objective) / ref:+.2e}  "
              f"down-tap KKT violation {k['delta_cond_violation_active']:.2e}")


if __name__ == "__main__":
    run_a()
    run_profiled(load=False)              # run B
    run_d(run_profiled(load=True))        # runs C and D
```

**Run E** is the repo's own gate. Run it in the working tree, and again in a clean copy of HEAD
(`git archive HEAD | tar -x -C <folder>`):

```python
from dwf2d.verification import parity_matrix
parity_matrix(n_cases=25, families=("default",))
```

---

## 3. Run-time results

### 3.1 Run A: timing only, no load (mean of seeds 0 and 1)

| N | mean sweeps | mean time per solve |
|---|---|---|
| 20 | 124 | 0.20 s |
| 40 | 3835 | 30.76 s |

Doubling N gave about **31× the sweeps** and about **150× the time**.

### 3.2 Run B: profiled, no load

| N | seed | sweeps | `polish()` calls | time [s] | time in `polish()` | converged |
|---|---|---|---|---|---|---|
| 10 | 0 | 9 | 1 | 0.03 | 53.7% | yes |
| 10 | 1 | 31 | 1 | 0.06 | 27.1% | yes |
| 10 | 2 | 30 | 1 | 0.05 | 32.7% | yes |
| 20 | 0 | 98 | 1 | 0.32 | 20.2% | yes |
| 20 | 1 | 151 | 1 | 0.45 | 16.9% | yes |
| 20 | 2 | 22 | 1 | 0.14 | 56.9% | yes |
| 30 | 0 | 746 | 29 | 7.62 | 61.6% | yes |
| 30 | 1 | 238 | 1 | 1.06 | 16.7% | yes |
| 30 | 2 | **20000** | 53 | 67.40 | 11.0% | **no** |
| 40 | 0 | 2505 | 104 | 33.51 | 72.7% | yes |
| 40 | 1 | 5165 | 124 | 58.94 | 64.7% | yes |
| 40 | 2 | 11011 | 175 | 91.19 | 53.8% | yes |

Runs A and B agree. For seeds 0 and 1, run B's mean sweeps are (98+151)/2 = 124.5 at N = 20 and
(2505+5165)/2 = 3835 at N = 40, as in run A. The solver is deterministic, so this is expected.

### 3.3 Run C: profiled, with slave load

| N | seed | total drain [J] | sweeps | `polish()` calls | `apply()` calls | time [s] | time in `polish()` | converged |
|---|---|---|---|---|---|---|---|---|
| 10 | 0 | 2.694 | 10 | 1 | 459 | 0.02 | 52.2% | yes |
| 10 | 1 | 3.168 | 21 | 1 | 537 | 0.03 | 35.4% | yes |
| 10 | 2 | 2.634 | 30 | 1 | 597 | 0.04 | 26.1% | yes |
| 20 | 0 | 7.189 | 98 | 1 | 3,350 | 0.23 | 22.5% | yes |
| 20 | 1 | 4.718 | 80 | 1 | 3,049 | 0.20 | 26.3% | yes |
| 20 | 2 | 4.062 | 22 | 1 | 1,947 | 0.10 | 56.0% | yes |
| 30 | 0 | 8.647 | 746 | 29 | 117,305 | 5.91 | 64.5% | yes |
| 30 | 1 | 8.284 | 238 | 1 | 10,635 | 0.79 | 19.7% | yes |
| 30 | 2 | 8.459 | **20000** | 65 | 822,796 | 64.06 | 15.0% | **no** |
| 40 | 0 | 12.438 | 2645 | 105 | 763,896 | 37.85 | 75.7% | yes |
| 40 | 1 | 11.183 | 4867 | 114 | 992,708 | 53.78 | 67.0% | yes |
| 40 | 2 | 11.455 | 14023 | 217 | 2,108,385 | 115.76 | 58.1% | yes |

Every case started from the just-in-time point.

### 3.4 Derived numbers

**Cost of one sweep** = time outside `polish()` ÷ sweeps:

| N | no load (run B) | with load (run C) |
|---|---|---|
| 10 | 1.5 ms (seed 0) | 1.0 ms (seed 0) |
| 20 | 2.5 ms (seed 1) | 1.8 ms (seed 1) |
| 30 | 3.0 ms (seed 2) | 2.7 ms (seed 2) |
| 40 | 3.7–3.8 ms (seeds 0, 2) | 3.5 ms (seeds 0, 2) |

The N = 10 figures come from times of 0.02–0.03 s and are only accurate to about ±30%.

**Cost of one `polish()` call** = time in `polish()` ÷ calls:

| N | no load (run B) | with load (run C, family D on) |
|---|---|---|
| 10 | 0.016 s | 0.010 s |
| 20 | 0.065–0.080 s | 0.052–0.056 s |
| 30 | 0.14–0.16 s | 0.13–0.16 s |
| 40 | 0.23–0.31 s | 0.27–0.32 s |

**Growth from N = 10 to N = 40** (4× N), using the median over the three seeds:

| quantity | no load: N = 10 → 40 | p (ratio = 4^p) | with load: N = 10 → 40 | p |
|---|---|---|---|---|
| sweeps | 30 → 5165 (172×) | ≈ 3.7 | 21 → 4867 (232×) | ≈ 3.9 |
| time per solve | 0.05 → 58.9 s (~1200×) | ≈ 5.1 | 0.03 → 53.8 s (~1800×) | ≈ 5.4 |
| time per sweep | 1.5 → 3.7 ms | ≈ 0.7 | 1.0 → 3.5 ms | ≈ 0.9 |
| time per `polish()` call | 0.016 → 0.27 s | ≈ 2.0 | 0.010 → 0.30 s | ≈ 2.4 |

The exponents with a load are less reliable than without, because the N = 10 times are so small
(0.02–0.04 s).

### 3.5 With load against without load, same problem

| N | seed | sweeps, no load | sweeps, with load | time, no load | time, with load |
|---|---|---|---|---|---|
| 10 | 0 / 1 / 2 | 9 / 31 / 30 | 10 / 21 / 30 | 0.03 / 0.06 / 0.05 s | 0.02 / 0.03 / 0.04 s |
| 20 | 0 / 1 / 2 | 98 / 151 / 22 | 98 / 80 / 22 | 0.32 / 0.45 / 0.14 s | 0.23 / 0.20 / 0.10 s |
| 30 | 0 / 1 / 2 | 746 / 238 / 20000 | 746 / 238 / 20000 | 7.62 / 1.06 / 67.40 s | 5.91 / 0.79 / 64.06 s |
| 40 | 0 / 1 / 2 | 2505 / 5165 / 11011 | 2645 / 4867 / 14023 | 33.5 / 58.9 / 91.2 s | 37.9 / 53.8 / 115.8 s |

In 6 of the 12 problems the sweep count is **identical** with and without the load. The drain is
small (about 1.3% of the harvest) and barely changes how the solver moves through the problem. The
same N = 30 problem fails to converge in both runs.

---

## 4. Optimality results

### 4.1 Run D: DWF objective against the cvxpy optimum, with slave load

The relative gap is `(cvxpy − DWF) / cvxpy`. A positive value means the DWF is below the optimum.

| N | seed | DWF objective | cvxpy optimum | relative gap | DWF converged |
|---|---|---|---|---|---|
| 10 | 0 | 71.679687 | 72.964309 | +1.76% | yes |
| 10 | 1 | 59.023530 | 61.774169 | +4.45% | yes |
| 10 | 2 | 54.136259 | 55.825312 | +3.03% | yes |
| 20 | 0 | 126.418310 | 133.334859 | +5.19% | yes |
| 20 | 1 | 124.099018 | 136.629415 | **+9.17%** | yes |
| 20 | 2 | 130.497238 | 137.548062 | +5.13% | yes |
| 30 | 0 | 199.312096 | 208.231758 | +4.28% | yes |
| 30 | 1 | 200.052614 | 210.007989 | +4.74% | yes |
| 30 | 2 | 183.125879 | 190.342697 | +3.79% | **no** |
| 40 | 0 | 267.754455 | 280.030176 | +4.38% | yes |
| 40 | 1 | 259.958908 | 279.758411 | +7.08% | yes |
| 40 | 2 | 260.962414 | 269.397327 | +3.13% | yes |

**All 12 are below the optimum, by 1.8% to 9.2%.** "Converged" only means "no known move improves";
it does not mean "optimal".

cvxpy took about **0.02 s** per problem; the first call took 1.6 s, which includes setup. That is
more than 1000× faster than the DWF at N = 40 on these problems.

### 4.2 Run D: the same check without load, with the KKT diagnostic

| N | seed | relative gap | down-tap KKT violation (`delta_cond_violation_active`, should be ≈ 0) |
|---|---|---|---|
| 10 | 0 | +1.73% | 5.27 |
| 10 | 1 | +4.40% | 3.01 |
| 20 | 1 | +9.02% | 6.44 |

The gap is **not caused by the load**: the no-load versions of the same problems are below the
optimum by almost the same amount. In both cases, `check_kkt_joint()` flags the result clearly: the
relative violation of `m_k = a_k s_k` is 3–6, where a correct optimum gives about 0. This is a
useful practical point: **a large `delta_cond_violation_active` is a reliable warning sign.**

### 4.3 Where the lost rate goes (N = 10, seed 1, no load)

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `a = beta*alpha` | 0.65 | 0.52 | 0.68 | 0.73 | 0.68 | **0.77** | **0.49** | 0.65 | 0.63 | 0.50 |
| harvest `E` [J] | 21.45 | 38.12 | 7.48 | 38.05 | 13.85 | 18.09 | 33.45 | 17.55 | 22.88 | 3.05 |
| `delta`, DWF [J] | 20.29 | 29.45 | 17.31 | 25.19 | 25.11 | 19.68 | 33.45 | 17.55 | 22.88 | 3.05 |
| `delta`, cvxpy [J] | 21.45 | 7.06 | 12.51 | **64.07** | 0.00 | **31.94** | **0.00** | 51.00 | 22.88 | 3.05 |

- Both solvers give the master's data link nothing (`tau*P = 0` in every slot).
- Both send the same total energy, 213.96 J, the whole harvest.
- **cvxpy uses the master battery to send in the slots where `a` is high.** For example, it sends
  nothing in slot 6 (a = 0.49) and carries that harvest forward to slot 7. The DWF sends roughly
  each slot's own harvest, including all 33.45 J of slot 6 at a = 0.49.
- So the energy stored in the slave is 146.35 J for cvxpy against 134.38 J for the DWF. That is 9%
  more for the slave to transmit with, which is where the 4.4% rate gap comes from.

### 4.4 Why the DWF cannot find this

The improving direction is: send `t` less at slot `i`, carry it in the master battery, and send it at
slot `k` instead:

```
d[i] -= t,   f[i+1..k] += t,   d[k] += t          (slave ends: xs(i) -= a_i t,  xs(k) += a_k t)
```

The master's own spending is unchanged everywhere. The gain is `(a_k − a_i)·t` Joules in the slave
battery.

**None of the solver's moves follows this direction.** It can only be built from two moves, and the
first of them loses rate on its own:
- Move (10) and `polish()` family D re-time `delta`, but they keep the slave's spending fixed and
  take energy from a master slot's data energy, which is zero here.
- Moves (3) and (4) and `polish()` family C take energy from a master slot's data energy, which is
  zero here.
- Move (4) run backwards (`t < 0`) does the first half: it hands slave energy at `i` back to the
  master at `i+1`. Move (1) at `i+1` would then send it on. But the master's own data link values
  that Joule at only `m = th1*h/2`, far less than the `a_i*s_i` the slave loses, so `step()`
  correctly refuses the first half. Each half lowers the objective; only the pair raises it.

So no single move improves, and the solver stops at a point that is feasible but not optimal. This
is the "corner" problem of coordinate ascent that the pass-through moves and `polish()` were built
to solve, for a direction they don't cover.

This is a **known open item**. The README's open item 17 says:

> **Missing move: master-carry delta shift** (`d[i] -= t, f[i+1..k] += t, d[k] += t`: re-time sharing
> between two down taps through the master carry, slave ends `xs(i)`, `xs(k)`). Its absence explains
> most below-optimum cases of the stage-6 parity matrix (the "master gives everything" regime). Not
> added: the spec forbids ad-hoc moves without a decision.

### 4.5 Run E: the repo's own parity gate, working tree and HEAD

`parity_matrix(n_cases=25, families=("default",))`:

| code | exact | below | above | infeasible | non-converged (at `max_iter=1000`) | largest gap |
|---|---|---|---|---|---|---|
| working tree | 2 | 23 | 0 | 0 | 6 | 7.31% (case #13) |
| HEAD (`4dcd561`) | 2 | 23 | 0 | 0 | 6 | 7.31% (case #13) |

The case-by-case gaps are identical between the two. `STAGE6_REPORT.md` reports the same "default:
2 exact, 23 below" in its cvxpy table. So:

- the gap is **not** caused by the uncommitted changes;
- the gap is **not** caused by the problem generator used in this file;
- "0 above" holds: the DWF never beat the optimum, so it never broke a constraint.

**Correction to an earlier statement.** The module docstring of `build_2D_dwf.py` says the solver is
"within about 0.1% of the optimum in 3 of 25 tests" on extreme random settings, and
`CRITICAL_FUNCTIONS_DWF.md` repeated that. On the parity `"default"` family, and on the problems in
this file, the gap is much larger and much more common. The README's open item 17 and
`STAGE6_REPORT.md` are the up-to-date description.

---

## 5. What the results show

1. **One sweep is cheap and scales as expected.** It costs 1–4 ms for N = 10–40, close to linear in
   N, as predicted by the O(N) per-sweep cost from the code.

2. **One `polish()` call grows about as N²–N^2.4.** With a load, family D adds its Θ(N³) term, but
   at N ≤ 40 it is not yet visible: a `polish()` call costs about the same with and without a load.
   It should start to dominate at larger N.

3. **The number of sweeps dominates the run time, and it grows very fast:** roughly N^3.7–N^3.9 on
   these problems, with large differences between problems of the same size (2,505 to 14,023
   sweeps at N = 40). Total time grows roughly as N^5. The exponents come from 3 problems per size
   over a 4× range of N, so they describe this test set, not a law of the algorithm.

4. **`polish()` keeps restarting the sweeps at larger N.** At N ≤ 20 it ran once, at the end. At
   N ≥ 30 it ran 29–217 times and took up to 76% of the time. This ties in with finding 6: the solver
   keeps finding small improvements but cannot reach the optimum, because the direction it needs is
   missing.

5. **A slave load barely changes the run time.** Half the problems took exactly the same number of
   sweeps, and the problem that failed to converge failed in both runs.

6. **The solver stops 2–9% below the optimum on every problem in this regime**, with and without a
   load, including all the "converged" ones. The cause is the missing master-carry delta shift move
   (README open item 17). The regime, where the master gives the slave everything, arises whenever
   the slave's link is much better and weighted more heavily than the master's.

7. **The default `max_iter=1000` is too small for these problems.** Every N = 40 problem needed
   2,505–14,023 sweeps, and 6 of the 25 small parity cases (N = 8–14) also did not converge within
   1000.

8. **Practical advice for now:**
   - check `r.converged`;
   - check `check_kkt_joint(jp, r)["delta_cond_violation_active"]`, which is large (3–6) exactly in
     the cases that are far from the optimum;
   - in this regime, compare with `reference_solution_joint()`, which also takes only about 0.02 s at
     these sizes.

---

## 6. Expected cost from the code

For comparison with the measurements, this is the cost per piece from reading `dwf2d/solver.py`
(full analysis in `CRITICAL_FUNCTIONS_DWF.md`, section 15).

| piece | cost |
|---|---|
| one sweep: N slots × 10 moves, each `step()` plus a fixed set of bounds | O(N) |
| `polish()` family A or B (L = longest run of zero-power slots) | O(N·L²), at most O(N³) |
| `polish()` family C | O(N·L_m·L_s·(L_m+L_s)), at most O(N⁴) |
| `polish()` family D (only with a slave load) | Θ(N³) |
| total | O(K·N + P·C_polish): K sweeps, P ≤ K `polish()` calls |

The measurements confirm the per-sweep and per-`polish()` costs. They show that **K, the number of
sweeps, is what makes the solver slow** as N grows.

---

## 7. Limits of these experiments

- **Small sample.** There were only 3 random problems per N, and only N ≤ 40.
- **One problem family.** Random per-slot channels with a huge slave window, in the "master gives
  everything" regime. The demos (`legacy_demo`, `physical_demo`, `integrated_demo`) were not timed
  or checked here. The README reports that `integrated_demo` is exact against cvxpy.
- **Small slave load.** The drain is about 1.3% of the harvest. A heavier load, silent slots, or a
  tight battery window were not tested.
- **Profiler overhead.** It inflates the absolute times in runs B and C by about 1.5×.
- **Partial no-load optimality check.** Only 3 of the 12 no-load problems were compared with cvxpy.
  Run E covers 25 more small ones.
- **One machine and an uncommitted working tree.** Run E shows that the optimality results are the
  same on HEAD. The timing was not repeated on HEAD.

---

## 8. Possible next tests

1. **Add the missing move** (master-carry delta shift, README item 17), then rerun run D and run E
   to measure the gap rate before and after, as the README asks. It may also cut the sweep count,
   since many of the extra sweeps are probably the solver circling a point it cannot leave.
2. Time and check the three demos at their real N.
3. Repeat runs C and D with a heavier load, silent slots and a tight battery window, where family D
   and the arrival-cap moves actually bind.
4. Extend runs B and C to N = 80 and 160, or at least run them once, to see whether family D
   starts to dominate `polish()`.
5. For the non-converging problem (N = 30, seed 2), count which moves keep firing, to see whether the
   solver is zig-zagging between two moves.
