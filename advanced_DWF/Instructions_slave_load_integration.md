# Integrating the slave load profile into the DWF solver

Instructions for Claude Code. Read this together with the DWF context README (`Master + Slave
Directional Water-Filling`), `slave_load_profile.py` (the example/standalone module), `build_2D_dwf.py`,
`dwf_diagnostics.py`, `physical_chain.py` and `Instructions.md`. This file was written from the README and
from the standalone example, not from a fresh read of `build_2D_dwf.py`: **check every function name,
signature and index convention against the real code before editing**, and tell the user where they differ.

The goal: the slave's electronics (camera, MCU, external memory, MPPT) become a known, per-slot energy
drain inside the DWF problem, driven by **editable duration variables that are dummies today and will be
replaced by measured values later**. Nothing about the structure of the code or its outputs may change when
real values are typed in.

---------------------------------------------------------------------------------------------

## 0. Ground rules (hard)

1. `joint_dwf()` stays the single solver. No second solver, no cvxpy in the algorithm path.
2. `build_dwf.py` and `physical_chain.py` are **not modified**. `build_2D_dwf.py` and `dwf_diagnostics.py`
   may be extended. `slave_load_profile.py` is refactored (section 3).
3. Everything new is **off by default**. With no load attached the solver must reproduce today's result
   exactly (README demo: objective 1902.8194 nats, `iters = 1410`, converged). Record the real baseline
   from the current code *before* any edit (step 0 of section 9) and compare against that, not against
   this text.
4. Keep 0-indexed slots, the `f[k+1]` / `g[k+1]` convention (`f[0]`, `f[N]`, `g[N]` fixed), plain-ASCII math
   in comments, and the rule that every move is an exact 1-D maximiser of a concave function clipped to
   feasibility, so the objective never decreases.
5. **Never invent real values.** Every duration, image size, memory rate, link rate and TX-stage number in
   the repo today is a dummy. Do not tune them to make a demo look good.
6. Do not pin `F` or `b` exactly in tests (the battery split is not unique, README section 7).
7. Report pass/fail **rates** for cross-checks, not single successes (README section 10).

---------------------------------------------------------------------------------------------

## 1. What is real, what is dummy, what to ignore

### Real (datasheet) values: keep, do not make editable "dummies"

| Item | Value |
|---|---|
| Camera active / deep sleep | 0.30 W (energy equation; V x I = 0.28 W) / 10 uW |
| MCU active / deep sleep | 0.060 W (stated; V x I = 49.5 mW) / 16 uW (3.3 V x 5 uA) |
| Memory active / retention | 0.080 W (stated; V x I = 99 mW) / 0.200 mW |
| MPPT quiescent / sleep | 320 nA x 3 V = 0.96 uW / 480 nA x 3 V = 1.44 uW |
| MPPT input rating | 3.3 V x 100 mA = 0.33 W |
| Storage voltage range | 2.2 - 5.2 V |
| MPPT efficiency | 0.95 (same value as the `physical_chain.py` demo) |

The stated-vs-V x I mismatches are reported by `datasheet_check()`; keep that report. Use the **stated**
powers until the user says otherwise. The MPPT sleep current (480 nA) being larger than its operating
quiescent current (320 nA) looks backwards; keep the numbers, keep the note.

### Dummy (user will replace): all durations and link/image assumptions

Everything in section 3's `SlotTiming` and `TX_EFFICIENCY`, plus the helper assumptions image size,
memory throughput, link bit rate. Mark each with `# DUMMY` in the code.

### Demo-only: ignore for the integration (do not import into the solver path)

* `DEMO_SLAVE_DATA_J`, `DEMO_SENT_J`, `DEMO_HARVEST_J`, `DEMO_ON_SLOTS` and the "scale check" block.
* `slave_schedule_demo()`: the real schedule is `jp.slave_schedule`.
* Hard-coded `beta = 0.311` and `alpha = exp(-0.046*0.0005)` in `__main__`: the real `a_k = beta_k*alpha_k`
  comes from `JointParams` (per-slot `beta` after calibration).
* `arrival_profile()` (look-ahead sharing policy): in the integrated flow the **solver chooses `delta`**.
  Keep the function only as a standalone illustration.
* `p_tx_W` and the placeholder TX energy: in the integrated flow TX energy is the solver's slave data
  energy `xs(i) = tau*Pbar_i`. Keep the placeholder only for `python slave_load_profile.py` standalone.
* `sensitivity()` and the `margin = 2.0` storage margin: keep as standalone/reporting helpers; the margin
  becomes a named, editable constant.

---------------------------------------------------------------------------------------------

## 2. How the load enters the math

Define `c_i >= 0` [J]: the **non-TX electronics energy** the slave burns in slot `i` (camera + MCU + memory +
MPPT). The THz transmit stage is not in `c_i`: its energy is the data energy `xs(i)`.

Slave energy balance (README section 3) becomes

```
tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i - c_i          (c_i = 0 recovers today's model)
Gt_{-1} = b0 - B_min ,  Gt_{N-1} = 0                      (unchanged)
```

Constraints S1-S4 keep their form. Two assumptions to state in code comments and README:

* Within a slot the arrival happens first and the drain after it, so the **ceiling is checked at the
  arrival** (`Gt_{k-1} + a_k*delta_k <= W`, unchanged) and the **floor at the end of the slot**
  (`Gt_k >= 0`, unchanged). This is conservative for the ceiling.
* The MPPT quiescent/sleep choice in `c_i` is by **schedule** (quiescent in ON slots, sleep in silent
  slots), not by `delta_i > 0`. The difference is about 30 uJ per slot and a solver-dependent `c_i` would
  make the problem circular.

New constraint **S5 (MPPT input rating)**: the stored energy landing in one slot cannot exceed what the MPPT
can accept,

```
a_k*delta_k <= capA_k ,   capA_k = mppt_in_V * mppt_in_I_max * mppt_eff * tau      (= 18.81 J at the defaults)
```

`a_k` already contains the MPPT efficiency (`beta`), so this is a bound on stored energy. It is one more
linear bound on `d[k]`: `apply()` already handles "affected quantity `v + c*t` must stay in `[L, U]`", so S5
is a new quantity `a[k]*d[k]` with upper bound `capA[k]`. With `capA = inf` nothing changes.

Silent slots now **drain** the slave battery: `Pbar_i = 0` gives `Gt_i = Gt_{i-1} + a_i*delta_i - c_i`. The README
statement "the initial excess is carried unchanged to the first ON slot" is no longer true; update it.

Optional TX-stage efficiency `eta_tx` (DUMMY, default 1.0): radiated data power is `eta_tx*Pbar`. Fold it
into the link gain, `hbar_eff = eta_tx*hbar`, when building `JointParams`. The solver is untouched, `Pbar` stays
the electrical power drawn by the TX stage, and radiated power is `eta_tx*Pbar`. With 1.0 nothing changes.

**Averaging convention (state it, warn on it).** The DWF rate `tau/2 ln(1 + hbar*Pbar)` treats `Pbar` as the
average power over the whole slot. A real TX that transmits for `t_tx < tau` radiates at `xs/t_tx` for a
short burst, and by concavity that yields fewer nats than the DWF credits. Do not change the solver for
this. Do add a runtime warning when the user's TX window (inside `mcu_active_s` / `memory_read_s`) is shorter
than `tau`, and a README note under known limits.

---------------------------------------------------------------------------------------------

## 3. The editable block: `slave_config.py` (data only, no logic)

Create one file the user edits by hand. Pure data, no functions except an `as_*` accessor if needed.
`slave_load_profile.py` imports it; the `__main__` demo uses it too.

```python
# slave_config.py  --  EDIT THIS FILE ONLY.   Units: seconds, watts, joules, volts.
TIMING_IS_DUMMY = True        # flip to False once every duration below is a measured value

TAU_S = 60.0                  # must equal MasterParams.tau; validated at attach time

# ---------------- REAL: datasheet powers (change only if the part changes) ----------------
POWERS = dict(
    camera_active_W=0.30,  camera_sleep_W=10e-6,
    mcu_active_W=0.060,    mcu_sleep_W=0.016e-3,
    memory_active_W=0.080, memory_retain_W=0.200e-3,
    mppt_quiescent_W=320e-9*3.0, mppt_sleep_W=480e-9*3.0,
    mppt_in_V=3.3, mppt_in_I_max_A=0.100, mppt_eff=0.95,
    v_store_min=2.2, v_store_max=5.2,
)

# ---------------- DUMMY: durations per slot type (scalar, or array of length N) ----------------
# *_sleep_s / memory_retain_s = None  ->  remainder of the slot (tau - the active times).
ON_SLOT = dict(                      # a slot in which the slave transmits (schedule = 1)
    camera_active_s=0.5,   camera_sleep_s=None,     # DUMMY: capture, per image
    mcu_active_s=4.5,      mcu_sleep_s=None,        # DUMMY: capture + write + read/TX window
    memory_write_s=2.0,    memory_read_s=2.0,       # DUMMY
    memory_retain_s=None,
    mppt_active_s=60.0,    mppt_sleep_s=None,       # DUMMY: quiescent time (whole slot)
)
SILENT_SLOT = dict(                  # a slot in which the slave does not transmit (schedule = 0)
    camera_active_s=0.0,   camera_sleep_s=None,
    mcu_active_s=0.0,      mcu_sleep_s=None,
    memory_write_s=0.0,    memory_read_s=0.0,
    memory_retain_s=None,
    mppt_active_s=0.0,     mppt_sleep_s=None,
)

# ---------------- DUMMY: other assumptions ----------------
ETA_TX = 1.0                         # THz TX-stage efficiency (radiated / electrical)
STORAGE_MARGIN = 2.0                 # window margin used by size_storage()
```

Requirements for the loader (`SlotTiming` dataclass + `load_timing()` in `slave_load_profile.py`):

* Each field accepts a scalar or a length-`N` array (broadcast), so measured per-slot values (for example a
  capture only in some silent slots) need no code change.
* `None` for a sleep/retain field means `tau - (active times)`.
* Validation, raising `ValueError` with the field name and slot index: every duration `>= 0`;
  `camera_active + camera_sleep <= tau`, same for MCU and MPPT; `memory_write + memory_read + memory_retain
  <= tau`. If an explicit sleep makes a component's times sum to less than `tau`, assume the remainder is
  powered off (0 W), allow it, and print one line saying so.
* While `TIMING_IS_DUMMY` is `True`, print a one-line banner at the start of every run and at the top of the
  `diagnose()` report: `slave timing: DUMMY values (edit slave_config.py)`. Nothing else may differ when the
  flag is flipped.
* Dummy defaults above must reproduce the standalone baseline: silent slot 13.646 mJ (non-TX),
  ON slot non-TX overhead 0.7527 J (camera 0.1506, MCU 0.2709, memory 0.3312, MPPT 5.8e-5), and
  `sum(c) = 15.601 J` over the 60-slot demo schedule. Re-run the standalone script first and use *its*
  numbers if they differ.

Optional helper (dummy generator only): `derive_dummy_timing(image_MB, memory_MBps, link_bps)` that builds
the dicts above from image/memory/link assumptions, used to regenerate the defaults. It must not run on the
solver path.

---------------------------------------------------------------------------------------------

## 4. `slave_load_profile.py` refactor

Keep the module standalone-runnable (`python slave_load_profile.py` still prints the same sections and
writes the same files). Changes:

* Read powers and durations from `slave_config.py`. Keep `SlaveHardware` as the in-memory container if it
  keeps the diff small, but its fields must come from the config, not from literals.
* `slot_energy(...)` becomes timing-aware: energy per component =
  `P_active*t_active + P_sleep*t_sleep` (memory: `P_active*(t_write + t_read) + P_retain*t_retain`).
  Same component keys and order as today: `camera, mcu, flash, mppt, tx`. (`flash` is the key used in the
  CSV and plots; keep it for output compatibility even though the config says `memory_*`.)
* New `overhead_profile(timing, schedule, n_slots=None) -> dict` returning per-component arrays (length N)
  for `camera, mcu, flash, mppt` and `c = sum of the four`. No TX here.
* `build_profile(...)` keeps its signature. With `Pbar` given it adds `tx = tau*Pbar_i` in every slot
  (ON slots only in practice, since `Pbar = 0` in silent slots) and `load = c + tx`. Without `Pbar` it
  behaves as today (placeholder TX, standalone only).
* New `attach_slave_load(jp, timing=None, *, size_storage=False, use_mppt_cap=True, eta_tx=None) -> JointParams`
  (lives in `slave_load_profile.py` or `build_2D_dwf.py`, whichever avoids a circular import; the solver
  module must not import the component model). It returns a **copy** of `jp` with: `slave_overhead_J = c`,
  `mppt_cap_J = capA` (or `None`), `h_bar` scaled by `eta_tx`, and, if `size_storage=True`,
  `B_max = B_min + W_std` and `b0 - B_min = lead-run drain` from `size_storage()`. It validates
  `TAU_S == jp.master.tau`, `len(schedule) == N`, and that `jp.slave_schedule` is the schedule used to build `c`
  (the schedule is read from `jp`, never from `slave_schedule_demo()`).
* Keep `size_storage()`, `datasheet_check()`, plots and CSV writer, but make the plots/CSV take solver
  results (section 7).

---------------------------------------------------------------------------------------------

## 5. `build_2D_dwf.py` changes (the solver)

### 5.1 `JointParams`: two optional fields, defaults preserve today's behaviour
* `slave_overhead_J: np.ndarray | None = None` (length N, `>= 0`; `None` is treated as zeros).
* `mppt_cap_J: float | np.ndarray | None = None` (stored-energy cap per slot; `None` is treated as infinity).

Validate lengths and signs in the existing validation block. Per-slot `beta` handling stays as is.

### 5.2 Solver internals
* Add arrays `cS[i]` (overhead) and `capA[k]` (cap) next to `a[k]`.
* `xs(i) = a[i]*d[i] + g[i] - g[i+1] - cS[i]` everywhere `xs` is computed: `step()` inputs, `apply()`,
  moves 1-10, `polish()` (the "dead slot" test `xs <= 1e-6*scale` now uses the net data energy, which is
  what is meant), `check_kkt_joint`, `reference_solution_joint`, post-processing (`Pbar = xs/tau`).
* `apply()`: add the S5 quantity `a[k]*d[k] <= capA[k]` for every move that changes `d[k]` (moves 1-5, 10).
  Move 10 changes `d[k]` and `d[k+1]`; bound both.
* Move **effects** on `xs` are unchanged (`c` is constant), so the ten moves and their gating by the schedule
  need no new equations. If, after implementing, a feasible case ends below the cvxpy optimum, that is the
  signal that a new move is needed (section 5.4); do not add one speculatively.
* Post-processing: `b = B_min + g[1:]` and `b_peak = B_min + g[:-1] + a*delta` are unchanged; the energy
  ledger gains one line: slave overhead `sum(c)`, split by component by the reporting layer.

### 5.3 Start point: "share nothing" is no longer feasible
Today `d = 0` is feasible. With `c > 0` and `g[N] = 0` fixed it is not, unless the initial charge already
covers every future drain. Replace the start with a **just-in-time construction** (only when `c.any()`):

```
f = 0 everywhere;  xs = 0 everywhere (no TX energy yet)
for k in 0..N-1:
    shortfall = max(0, cS[k] - g[k])
    d[k]   = shortfall / a[k]                 # a[k] == 0 and shortfall > 0  -> infeasible
    g[k+1] = g[k] + a[k]*d[k] - cS[k]         # >= 0 by construction
leftover = g[N]                               # > 0 only if b0 - B_min exceeded the total overhead
if leftover > 0: spend it as slave data energy in the first ON slot (xs >= 0 there), set g[N] = 0.
```

Check the start against every bound: `d[k] <= E[k]` (master energy with `f = 0`), `g[k] + a[k]*d[k] <= W`,
`a[k]*d[k] <= capA[k]`. If any fails, raise a dedicated `InfeasibleSlaveBudget` error naming the slot and the
binding quantity (harvest too small, window `W` smaller than the largest silent-run drain, `c_k` above the
MPPT cap, `a_k = 0`). If the JIT start fails but a feasible point may still exist (it ignores borrowing
against future harvest through `f`), fall back to a feasibility solve with `reference_solution_joint`'s
constraints (cvxpy) to get a start, and say so in the solver status. Do not silently relax a constraint.

Terminal condition: `g[N] = 0` is kept. With a trailing silent run this forces `g[N-1] + a*d = c` in the
last slots (exact balance); that is feasible whenever the schedule and window allow it and is consistent
with the cvxpy reference as long as both use the same terminal condition.

### 5.4 If optimality gaps appear
The new coupling is across **silent runs** (energy must be routed through slots where `Pbar = 0` and now also
drains). The existing `polish()` already crosses runs of dead slots. If the cvxpy comparison (section 8) shows
gaps on realistic inputs, add a cross-run move in `polish()` first, following README section 10 (effect on
`xm`/`xs`, the marginal-balance equation, every bound and its multiplier). Report the gap rate before and
after.

### 5.5 `solve_with_physical_chain()` and `physical_stored_power()`
* Pass `slave_overhead_J` and `mppt_cap_J` through unchanged in the fixed-point loop (they are not
  `beta`-dependent). Keep calibrating `beta_i` only on slots with non-negligible `delta`.
* Check whether `optical_to_electrical()` already saturates at the MPPT input rating. If it does not, do
  **not** edit `physical_chain.py`: clip in `physical_stored_power()` in `build_2D_dwf.py`, add an
  `overflow_mppt` field to `PhysicalStorageResult` (default 0), and count it in the violation summary. With S5
  active in the solver this should stay 0 at the fixed point; a nonzero value means the calibrated `beta`
  drifted past the cap.
* The re-simulated physical battery must subtract the same `c_i`. At the fixed point the solver and physical
  batteries still agree to about 1e-9 J.

---------------------------------------------------------------------------------------------

## 6. Reference solver and KKT check

* `reference_solution_joint(jp)`: add `xb[i] == a[i]*d[i] + G[i-1] - G[i] - c[i]` and
  `a[k]*d[k] <= capA[k]`; keep the S4 constraint and the same terminal condition as the solver. **Do this
  before touching the solver** (section 9, step 3), so the new behaviour has an independent arbiter.
* `check_kkt_joint(jp, r)`: use the new `xs`; additionally return `slave_overhead_total_J`,
  `max_cap_violation_J` (should be about 0), `min_slave_level_margin_J`; keep every existing key
  (`slave_silent_slots`, `max_Pbar_in_silent_slots`, ...). Test the `delta` condition at transmitting slots
  only, as today, **and** skip it at slots where S5 binds (the cap multiplier is nonzero there).
* Slave energy accounting must close to about 1e-9 J: `(b0-B_min) + sum(a*delta) - sum(c) - sum(xs) - g[N] = 0`.

---------------------------------------------------------------------------------------------

## 7. Outputs: same structure, plus the load panels

Existing outputs stay exactly as they are (the nine `plot_*.png` files, the `diagnose()` report, the KKT and
calibration summaries). Add, using the same `_new_axis()` / `_finish()` helpers and one figure per panel:

| New file | Content |
|---|---|
| `plot_slave_load_per_slot.png` | stacked per-slot energy: camera, MCU, memory, MPPT (profile) + TX data energy `tau*Pbar_i` (solver); silent slots shaded |
| `plot_slave_energy_breakdown.png` | horizon energy by component with percentages (TX from the solver) |
| `plot_slave_cumulative_energy.png` | cumulative consumed (`c + xs`) vs cumulative stored (`a*delta`) plus the initial charge |
| `plot_slave_battery_window.png` | `b - B_min` after each slot and right after each arrival, with `W = B_max - B_min` and the profile's `W_req` lines |
| `slave_energy_profile.csv` | one row per slot; **columns unchanged**: `slot,on,camera_J,mcu_J,flash_J,mppt_J,tx_J,load_J,stored_J,delta_sent_J,battery_end_J` where `tx_J = tau*Pbar`, `load_J = c + tx_J`, `stored_J = a*delta`, `delta_sent_J = delta`, `battery_end_J = b - B_min` |

`diagnose(jp, r, load=None)` keeps its current lines and, when a load is attached, adds: the timing banner
(section 3), a slave overhead line per component in the ledger, the MPPT-cap status (number of slots where S5
binds, largest `a*delta` as a fraction of `capA`), and the window check (`W` vs the solution's
`max(b_peak) - B_min` and vs the profile's `W_req`). Add a short `size_storage` summary: required capacitance
for the storage voltage range and the nearest standard value. Fix the known scalar-`beta` print lines in
`diagnose()` while there if they are still wrong (README section 9).

Plots take the solver result, not the standalone arrival policy. Keep the standalone functions callable on
the standalone profile so `python slave_load_profile.py` still writes its four PNGs and the CSV.

---------------------------------------------------------------------------------------------

## 8. Two named demo configurations

The README stage-4 demo is at a scale the real electronics cannot support (slave data energy 29,600 J vs the
MPPT's 18.8 J per slot; stored power 24.7 W vs a 0.314 W rating). Provide both, switchable by one flag in
`build_2D_dwf.py`'s `__main__`:

1. **`legacy_demo`** (default): no load attached. Reproduces the README numbers exactly.
2. **`physical_demo`**: `attach_slave_load(jp, size_storage=True)` with the dummy timing, S5 on,
   `B_max - B_min` from `size_storage()` (about 5 J), `b0 - B_min` from the lead-run drain. The master side
   (winter harvest, `tau`, `h`, `E_max = inf`) is unchanged.

Expect `physical_demo` to differ from `legacy_demo` by orders of magnitude in slave data energy. That is the
point of the exercise, **not a solver bug**. Do not retune `h0`, `theta2` or the harvest to hide it. Say in
the report that the result is at dummy durations and placeholder `h0`/`theta2`.

---------------------------------------------------------------------------------------------

## 9. Order of work (each step ends with its acceptance check)

0. **Baseline.** Run `python build_2D_dwf.py` and `python slave_load_profile.py` unmodified. Save objective,
   `iters`, KKT/calibration summaries, the printed standalone numbers and a file list to a baseline note.
1. **Config + loader.** Create `slave_config.py`, refactor `slave_load_profile.py` (section 4). Accept:
   standalone output numbers identical to the baseline; changing a duration in the config changes `c` and
   nothing else; every validation error path hit once.
2. **Plumbing, no solver change.** Add the two `JointParams` fields and `attach_slave_load()`. Accept:
   attaching nothing, or attaching zeros with `mppt_cap_J = None`, gives the identical `legacy_demo` result
   (objective and `iters` equal to the baseline).
3. **Reference first.** Extend `reference_solution_joint` (section 6). Accept: with `c = 0`, `cap = None` it
   matches the old reference to about 1e-9 on the default, no-sharing and tight-battery cases.
4. **Solver.** Sections 5.1-5.3. Accept: `c = 0` path bit-identical to the baseline; `c > 0` cases converge
   and agree with the reference to about 1e-6 on the default, no-sharing and tight-battery cases.
5. **Physical chain.** Section 5.5. Accept: calibration converges, solver/physical batteries agree to about
   1e-9 J, `overflow_mppt = 0`.
6. **Reporting.** Section 7. Accept: baseline outputs unchanged; four new PNGs and the CSV written; the
   ledger closes (section 6).
7. **Validation matrix** (below).
8. **Docs.** Update the README: file table (section 2), the energy balance and S5 (section 3), start point
   and move notes (section 4), placeholders table (section 6: durations, `eta_tx`, margin all DUMMY), results
   (section 7: add `physical_demo`), verification (section 8: new rows), limits (section 9: averaging
   convention, ceiling-then-drain assumption, silent-run coupling), open items (section 11: real durations,
   real TX stage, cross-run move if needed).

### Validation matrix (report rates)
* Equivalence: `c = 0`, `cap = None` equals the old solver on every case below.
* Cross-check vs `reference_solution_joint`, at least 25 random cases each for: random `c` (per slot and
  scalar), random schedules (all ON, one ON slot, leading/trailing silent runs, `[1,0]`), tight `W`,
  per-slot `beta`, finite `E_max`, and S5 binding. Report: exact, within 1e-6, below optimum (with gap).
* Invariants: objective non-increasing in `c` (scale `c` by 0, 1, 2, 5); non-decreasing in `W` and in the
  cap; scaling `E, B, c, cap` by `s` and `h, hbar` by `1/s` scales `P, Pbar` by `s`; both-node energy
  accounting closes to about 1e-9 J; `Pbar = 0` in silent slots; `0 <= Gt <= W - a*delta` at every slot.
* Infeasible inputs raise `InfeasibleSlaveBudget` with a useful message: `W` below the largest silent drain,
  `c_k` above `capA`, `a_k = 0` with `c_k > 0`, harvest zero when sharing is needed.
* Dummy-to-real swap: change every duration in `slave_config.py` (including a per-slot array) and re-run;
  no code edit may be needed and the only output differences are numeric.

---------------------------------------------------------------------------------------------

## 10. Things not to do

* Do not edit `build_dwf.py` or `physical_chain.py`.
* Do not import `DEMO_*`, `slave_schedule_demo`, or the standalone `arrival_profile` into the solver path.
* Do not hard-code durations, `beta`, `alpha` or the schedule anywhere outside `slave_config.py` / `JointParams`.
* Do not make `c_i` depend on the solver's `delta` (circular).
* Do not "fix" a large change between `legacy_demo` and `physical_demo` by retuning inputs.
* Do not relax `S1`-`S5` to get a start point; raise `InfeasibleSlaveBudget` instead.
* Do not remove or rename existing outputs, result fields, CSV columns or plot files.

## 11. Questions to settle with the user (proceed with the stated defaults if unattended, and record them in the README)

1. TX window convention: whole slot (consistent with the DWF) or a short burst (default dummy: burst, with a
   warning)?
2. `eta_tx`, the THz source efficiency (default 1.0).
3. Initial slave charge `b0` (README: still an assumption; `size_storage=True` sets it to the lead-run drain).
4. Whether captures should move into silent slots (then `SILENT_SLOT.camera_active_s` etc. are nonzero and the
   memory must hold the image across slots).
5. Real supercap leakage (not modelled; at a few uA it competes with the 227 uW sleep floor).
