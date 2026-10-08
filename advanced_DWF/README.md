# Master + Slave Directional Water-Filling (DWF)

Context file for Claude Code. Read this before touching the code. It explains what the
project is, what the algorithm does, how it maps to the math, what has been verified, and what
is still open. Everything here reflects the state of the code as delivered; nothing is aspirational
unless it is under "Open items".

claude --resume c97f929b-cf69-483f-9bf1-54f542f3a0b6

---------------------------------------------------------------------------------------------

## 1. Project goal

Build, step by step, a **directional water-filling (DWF)** power/energy-sharing algorithm for an
**underwater energy-harvesting pair**:

* **Master** (node 1) harvests energy `E_i` from the environment every slot, transmits data at
  power `P_i`, and **shares** energy `delta_i >= 0` with the slave.
* **Slave** (node 2) harvests **nothing**. Its only energy is what the master shares, reduced by
  two losses in series: channel attenuation `alpha_i` (Beer-Lambert: `alpha = exp(-c*d)`) and
  battery recharging efficiency `beta`. Only `a_i*delta_i` lands in its battery, `a_i = beta*alpha_i`.
* The slave draws on its **finite battery** (`B_min <= level <= B_max`) to send data back to the
  master at power `Pbar_i`.
* The master battery capacity `E_max` can be taken to infinity to recover the simple case.
* Two structural properties: **right taps** (a node's own energy flowing forward in time, i.e.
  temporal / energy causality) and **down taps** (master -> slave energy transfer, i.e. spatial).
  Terminology follows Gurakan et al. (right = temporal, down = spatial). The original brief had
  these swapped; the user did not object to using the paper's convention.

The work has been done in stages: (1) master alone, (2) master + energy sharing `delta`, (3) master +
slave with a finite battery window, (4) slave **transmission schedule** (the slave may only send data in
some slots) and a **physical receive-chain calibration** of the down tap (photodiode/GaAs cell + MPPT).
(5) the slave's **electronics load**: camera, MCU, external memory and MPPT become a known per-slot energy drain
`c_i` plus an MPPT input cap (S5), driven by one editable file, `slave_config.py` (section 4, "Slave electronics
load"). (6) **Integrated scenario** (`implementation.md`, section 4 "Stage 6"): the master only harvests and shares
energy over UOWC (no data, `theta1 = 0`); the slave transmits only THz photomixer **bursts**, one per scheduled
**event**, with a burst cap S6, a burst-aware rate, event refusal / rescheduling and a maximal-sharing tie-break;
`python build_2D_dwf.py integrated_demo`. Stage 6 is the current state. Everything it adds is off by default (the
legacy demos are bit-for-bit unchanged) and `joint_dwf()` stays the single solver.

---------------------------------------------------------------------------------------------

## 2. Files (all in `advanced_DWF/`)

| File | What it is |
|---|---|
| `build_dwf.py` | Stage 1: master-only DWF. `MasterParams`, `directional_water_filling`, `check_kkt`, `reference_solution` (scipy), `plot_result`. Holds `Harvested_energy_summer` and `Harvested_energy_winter` (60 samples, J per 1-minute slot). |
| `build_2D_dwf.py` | Stages 3-4 entry point: the theory reading guide (module docstring), the `DEMO` switch and `python build_2D_dwf.py [legacy_demo \| physical_demo]`. The code itself lives in `dwf2d/`; this file re-exports every name it has always had, so `from build_2D_dwf import ...` keeps working unchanged. |
| `dwf2d/` | Stages 3-4 implementation, one module per concern (see "Code layout" below): joint master + slave DWF, slave schedule (S4), physical-chain post-processing and calibration, verification, plots, demos. Imports `Harvested_energy_summer`/`Harvested_energy_winter` and `MasterParams` from `build_dwf.py`, `diagnose` from `dwf_diagnostics.py`, and `optical_to_electrical`, `OpticalReceiveChainResult` from `physical_chain.py`. |
| `tests/` | Golden-master regression check: `python tests/regression.py check` (or `python -m pytest tests`) re-runs the three demos and compares stdout, every result array (bit-for-bit), every PNG/CSV hash and the `build_2D_dwf` API against `tests/baseline/`. Changes that only ADD something switched off (a new dataclass field left at `None`, a new name, a trailing parameter with a default) are listed as `ADDITIVE`, not as differences. `python tests/regression.py capture [case]` re-records the baseline (one case only when named; never re-capture the legacy demos for stage-6 work). Stage-6 gates: `parity [n]` (joint_dwf vs cvxpy), `events [n]` (refusal loop vs brute force, post-pass), `stage6 [n]` (presets, invariants, scalability; slow). `test_regression.py` also holds fast checks of the stage-6 numbers. |
| `Modulation_THz_slave_profile.ipynb` | THz modulation notebook (link budget, antenna gains, 4/8/16/64-QAM and PAM, Monte-Carlo BER) and the slave energy profile (section 13). A thin front end: the link and modulation code is imported from `dwf2d/channels.py`, the slave profile from `slave_load.py`. `LEGACY_GAINS = True` (5/5 dBi at both frequencies) reproduces the earlier saved outputs bit for bit. Runs in the Anaconda environment (pandas, IPython). |
| `implementation.md` | The stage-6 spec. `STAGE6_REPORT.md` is the report of how it was built and verified. |
| `out_integrated_demo/` | Output of `python build_2D_dwf.py integrated_demo`: the joint / physical / slave-load panels, `plot_event_bursts.png` and `slave_energy_profile.csv` (legacy columns, then `photomixer_J`, the per-event link columns and `master_stop`). `thz_link/` holds the notebook's 13 THz-link figures (BER vs SNR / distance / data rate / received power, link characteristics, constellations, eye diagrams, simulation vs theory, the five antenna-gain figures) at the `slave_config.py` link values, with the DWF operating point (preset, `D_THZ_M`, burst SNR at the 20 mW cap) marked in purple. |
| `physical_chain.py` | Receive side of the optical energy link: optical power at the slave -> GaAs/Ge cell array (Spectrolab datasheet: Voc = 1.025 V, FF = 0.82, QE-derived responsivity at 450/532 nm) -> MPPT -> electrical power into the battery. `optical_to_electrical()`, `photodiode_stage()`, `mppt_stage()`, `size_array_by_intensity()`. Never modifies the solver. |
| `dwf_diagnostics.py` | `diagnose(jp, r, load=None)`: human-readable report (sharing decision, energy ledger, active constraints, solver status). With `load` (from `slave_load.make_load`) it also prints the dummy-timing banner, the electronics energy per component, the MPPT-cap status, the window check and the supercap sizing. Called by the demo. Stage 6: `diagnose_no_data()` (the report when `theta1 = 0`; `diagnose()` divides by `theta1`), `link_table()` / `link_report()` (per-event THz table). |
| `slave_config.py` | **The only file to edit** for the slave electronics: real datasheet powers, DUMMY durations per slot type (scalars or length-N arrays, `None` = rest of the slot), `TIMING_IS_DUMMY`, `ETA_TX`, `STORAGE_MARGIN`. Stage 6 adds its own block (used only by the event model): `SLAVE_SCHEDULE`, `THZ_PRESET`, link and payload values, photomixer (`ETA_TX_PM` = 0.1, a different knob from the legacy `ETA_TX`), `PM_DC_MODE`, `MPPT_EFF`, `ETA_SC`, `SUPERCAP_F`, `EVENT_DURATIONS`, the `EVENT_COMPONENTS` table (a component = a row) and `DUMMY_VALUES` (banner). Pure data. |
| `slave_load.py` | (called `slave_load_profile.py` in the task text) Component model of the slave: `load_timing()` (validated `SlotTiming`), `overhead_profile()` (per-component energies and `c = camera + mcu + memory + mppt`, no TX), `attach_slave_load(jp, ...)`, `make_load(jp)`, `size_storage()`, `derive_dummy_timing()`, and the standalone illustration (`python slave_load.py`). Imported lazily by the demo; the solver module never imports it. Stage 6: `event_profile()` (per-event load from the component table), `tx_time_s`, `tx_bounds_J` (`x_cap`, `x_min`), `thz_link()`, `supercap_window()`, `attach_event_load()` (= `attach_slave_load(..., model="event")`), `make_event_load()`, `reschedule()` (refusal queue), `SlaveState`, `dummy_banner()`. |
| `Instructions_slave_load_integration.md` | The spec this stage was built from (section 4, "Slave electronics load", records where the code differed from it). |
| `Instructions.md` | Design notes for replacing the linear slave-battery model with the physical chain: Option B (per-slot calibrated `beta`, **implemented**) and Option C (nonlinear tap inside the optimiser, **not implemented**). |
| `PD cell/` | Datasheets (photodiodes, APD, STM32L010, ODD-5W) used for the receive-chain parameters. |
| `plot_link_power.png`, `plot_transmit_power.png`, `plot_slave_battery.png`, `plot_throughput_per_slot.png`, `plot_throughput_cumulative.png`, `plot_harvested_energy.png`, `plot_sharing_energy.png`, `plot_physical_stored_power.png`, `plot_physical_slave_battery.png` | Output of the demo: one PNG per panel (see section 4's "Public API" for what each shows). Each function saves exactly one figure, so panels can be read, sized and shared independently. |
| `out_physical_demo/` | Output of `python build_2D_dwf.py physical_demo`: all the plots above plus `plot_slave_load_per_slot.png`, `plot_slave_energy_breakdown.png`, `plot_slave_cumulative_energy.png`, `plot_slave_battery_window.png` and `slave_energy_profile.csv` (columns `slot,on,camera_J,mcu_J,flash_J,mppt_J,tx_J,load_J,stored_J,delta_sent_J,battery_end_J`; `tx_J = tau*Pbar`, `load_J = c + tx_J`, `stored_J = a*delta`, `battery_end_J = b - B_min`). Kept in its own folder so the legacy plots are not overwritten. `python slave_load.py` writes the same file names into the current folder from its standalone illustration. |
| `dwf_master_inf.png`, `dwf_master_3300.png` | Stage 1 plots (E_max = inf and 3300 J). |
| Reference papers (PDF, provided by the user) | Gurakan et al., *Energy Cooperation in Energy Harvesting Communications* (2-D DWF, right/down taps); Ozel et al., *Transmission with Energy Harvesting Nodes in Fading Wireless Channels* (and the INFOCOM'11 version); Ozel et al., *Optimal Broadcast Scheduling ... Finite Capacity Battery*; *Sum-rate optimal power policies ... interference channel*. |

`build_dwf.py` and `physical_chain.py` must not be modified as a side effect of solver work; `build_2D_dwf.py` and `dwf2d/` only import from them.

### Code layout (`dwf2d/`)

| Module | Contents |
|---|---|
| `dwf2d/channels.py` | `beer_lambert`, the THz absorption table (`thz_freq`, `thz_alpha_cm`, `thz_alpha_m`), `thz_absorption`, `thz_gain`; stage 6: the photomixer link budget moved from the notebook (`link_budget`, `fspl_db`, `absorption_db`, `near_field_limit_m`, `noise_floor_w`, `preset_gains`, `thz_hbar`), `uowc_alpha`, and the modulation engine (`gray_pam`, `gray_qam`, `qam_ber_theory`, `pam_ber_theory`, `make_scheme`, `qam_scheme`, `rrc_pulse`, `min_distance_decode`, `make_link_simulator`) |
| `dwf2d/params.py` | `JointParams` (stage 6: `tx_cap_J` S6, `tx_floor_J` S7, `delta_cap_J` M5, `slave_rate_window`), `JointResult` |
| `dwf2d/solver.py` | `joint_dwf` (stage 6: S6/S7/M5 bounds, rate window in `step()`, warm start `x0`), its start points `_jit_start` / `_lp_start`, `InfeasibleSlaveBudget` (`.slots`), `warm_start_from`; stage 6: `fullest_trajectory`, `maximal_sharing` (post-pass) |
| `dwf2d/physical.py` | `PhysicalStorageResult`, `physical_stored_power` (`supercap_eff`), `solve_with_physical_chain` (`supercap_eff`, `solver_kw`); stage 6: `solve_with_refusals`, `RefusalResult` |
| `dwf2d/verification.py` | `check_kkt_joint`, `reference_solution_joint` (cvxpy; stage 6: S6/S7/M5, burst rate); stage-6 gates: `parity_matrix`, `random_parity_case`, `small_event_problem`, `brute_force_events`, `max_level_lp`, `check_maximal_sharing`, `refusal_gate`, `stage6_checks` |
| `dwf2d/plotting/style.py` | palette, `BITS_PER_NAT`, `EDGE_TOL`, shared helpers (`_new_axis`, `_new_axes`, `_finish`, `_legend`, `_mark`, `_shade_silent`, `_bound_masks`, `_rates_bits`) |
| `dwf2d/plotting/joint.py` | panels of the joint solution (`plot_link_power` ... `plot_sharing_energy`) |
| `dwf2d/plotting/physical.py` | `plot_physical_stored_power`, `plot_physical_slave_battery`, `energy_loss_breakdown`, `plot_loss_histogram` |
| `dwf2d/plotting/slave_panels.py` | slave-load panels, `slave_load_components`, `write_slave_energy_csv`; stage 6: `plot_event_bursts` |
| `dwf2d/plotting/thz_link.py` | the 13 THz-link figures of the notebook (sections 7-12; the notebook is only the reference and is not changed), drawn by `integrated_demo`: `ThzLinkSetup` (the notebook configuration), `setup_from_slave_config` (slave_config link values + the DWF operating point), one `plot_*` per figure (returns the figure), `simulate_demo` / `simulate_validation` (Monte-Carlo), `write_thz_link_plots` |
| `dwf2d/demos.py` | `_demo_problem`, `_run_demo`, `legacy_demo`, `physical_demo`; stage 6: `_integrated_problem`, `integrated_demo` |

Dependencies only point "down" this table (`params` ← `solver` ← `physical` ← plots ← `demos`), so there are no import
cycles. Where new code goes:
- **A new plot:** add it to the matching `dwf2d/plotting/` module, call it from `_run_demo`, and re-export it in `build_2D_dwf.py`.
- **A new model or channel:** add it to `channels.py`, or to a new module.
- **Before committing any refactor:** run `python tests/regression.py check`. It must print `IDENTICAL to baseline`.

Dependencies: `numpy`, `matplotlib` (plots), plus the local modules above. `cvxpy` (with the CLARABEL solver) only for
`reference_solution_joint()`, which is a cross-check, not part of the algorithm.
`scipy` is used by the stage-1 `reference_solution()` and, only when a slave load is attached and the just-in-time
start breaks a bound, by the LP start fallback (`scipy.optimize.linprog`).

Run the demo: `cd advanced_DWF && python build_2D_dwf.py` (= `legacy_demo`: prints the `diagnose()` report, the KKT and calibration summaries, and writes the nine `plot_*.png` files above). `python build_2D_dwf.py physical_demo` runs the same problem with the slave electronics attached (see section 7); `python build_2D_dwf.py integrated_demo` runs the stage-6 scenario into `out_integrated_demo/`; the choice is also the `DEMO` constant in `build_2D_dwf.py` (currently `physical_demo`).

---------------------------------------------------------------------------------------------

## 3. The problem being solved (stage 3)

Slots `i = 0..N-1`, slot length `tau` (60 s), energies in Joules, powers in W. Rates in nats.

```
max   sum_i (th1*tau/2) ln(1 + h_i * P_i)   +   sum_i (th2*tau/2) ln(1 + hbar_i * Pbar_i)
```

* `h_i`: master link gain over noise. `hbar_i`: slave return-link gain over noise.
* `th1, th2`: rate weights (1 = plain sum-rate).
* "Received power" in plots/results is in noise-normalised units (N0 = 1 W): `h*P` is the SNR
  inside the log. The energy-link power is `delta/tau` (sent), `alpha*delta/tau` (after channel),
  `beta*alpha*delta/tau` (stored).

### Variables and energy balance

Free variables (all Joules): master carry `F_k`, slave carry above the floor `Gt_k`, and `delta_k`.
The data energies are *defined* by the energy balance:

```
tau*P_i    = E_i + F_{i-1} - F_i - delta_i            (master)
tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i - c_i      (slave; c_i = 0 without a slave load)
F_{-1} = F_{N-1} = 0 ;  Gt_{-1} = b0 - B_min ,  Gt_{N-1} = 0
```

`c_i >= 0` [J] is the non-TX electronics energy the slave burns in slot i (camera + MCU + memory + MPPT), see
"Slave electronics load" in section 4. The THz transmit stage is not in `c_i`: its energy is the data energy `tau*Pbar_i`.

`Gt_k = b_k - B_min`, where `b_k` is the slave battery level at the end of slot k.

### Constraints (multipliers in brackets)

```
M1  F_n >= 0                               master causality            [lambda_n]
M2  F_n <= E_max - E_{n+1}                 master capacity at arrival  [mu_n]
M3  P_i >= 0, delta_i >= 0                                             [eta_i, rho_i]
S1  b_n >= B_min                           slave floor, end of slot    [lambda_bar_n]
S2  b_{n-1} + a_n*delta_n <= B_max         slave ceiling, at ARRIVAL   [mu_bar_{n-1}]
S3  Pbar_i >= 0                                                        [eta_bar_i]
S4  Pbar_i = 0 in every slot with schedule s_i = 0     (JointParams.slave_schedule)
S5  a_k*delta_k <= capA_k                  MPPT input rating, stored energy per slot (JointParams.mppt_cap_J; inf = off)
S6  tau*Pbar_i <= X_i                      burst cap (photomixer at 20 mW radiated)  (JointParams.tx_cap_J; stage 6)
S7  tau*Pbar_i >= Lf_i  (transmit slots)   burst floor, optional                     (JointParams.tx_floor_J; stage 6)
M5  delta_k <= Dc_k                        laser cap                                 (JointParams.delta_cap_J; stage 6)
```

Stage 6 also changes the slave rate term when `JointParams.slave_rate_window` (`T_i`) is set: the slave radiates
`x_i = tau*Pbar_i` as a burst of length `T_i` (= `t_tx`), so its term is `th2*(T_i/2) ln(1 + hbar_i x_i/T_i)` (with
`T_i = tau` it is the expression above). With `theta1 = 0` the master has no data: `tau*P_i` is its unused / spill
energy. See section 4, "Stage 6".

**S4 (slave schedule).** `s_i in {0,1}`; a shorter pattern repeats cyclically (`[1,0]` = every other
slot). A silent slot (`s_i = 0`) still *receives* `delta_i` and stores it; it is a pure pass-through for
the slave battery (carry in = carry out + arrival). Consequence: energy sent in a silent slot has no local
marginal value; it is valued at the next transmitting slot, so the interior condition `m_k = a_k s_k` is
only tested at transmitting slots. The initial excess `b0 - B_min` can only be spent as data from the first ON
slot on. **With a slave load attached, silent slots also drain `c_i`**, so the excess is no longer "carried
unchanged" to the first ON slot: it falls through the leading silent run (`Gt_i = Gt_{i-1} + a_i*delta_i - c_i`).
Validation: the schedule must be 0/1, and `b0 > B_min` with no ON slot is rejected.

**S5 (MPPT input rating).** `a_k*delta_k <= capA_k`, `capA_k = mppt_in_V*mppt_in_I_max*mppt_eff*tau` (18.81 J at the
defaults). `a_k` already contains the efficiency, so this bounds the stored energy. It is one more linear bound on
`d[k]`, added through `apply()` and to the explicit bounds of moves 1-5; with `capA = inf` nothing changes. Within a
slot the arrival happens first and the drain after it (ceiling checked at arrival, floor at the end of the slot).

S1 and S2 combine into one two-sided bound on the slave carry:
`B_min <= b_n <= B_max - a_{n+1}*delta_{n+1}`. The slave's right-tap capacity therefore depends on a
decision variable (`delta_{n+1}`); this is the only place the down tap feeds back into the slave's
temporal chain. Within a slot the level only falls after the arrival, so the floor is checked at the
end of the slot and the ceiling at the arrival.

### Stationarity (what the moves are derived from)

```
m_k = th1*h_k / (2(1 + h_k P_k))            master marginal value of a Joule   = Lambda_k      (P_k > 0)
s_k = th2*hbar_k / (2(1 + hbar_k Pbar_k))   slave  marginal value of a Joule   = Lambda_bar_k  (Pbar_k > 0)

P_k    = [ nu_k - 1/h_k ]^+ ,   nu_k = th1 / (2 Lambda_k)         (water level nu = P + 1/h)
Lambda_k     = sum_{n>=k} lambda_n - sum_{n>=k} mu_n              (master tap term)
Lambda_bar_k = sum_{n>=k} lambda_bar_n - sum_{n>=k} mu_bar_n      (slave tap term)
Gamma_k      = Lambda_bar_k - mu_bar_{k-1}                         (value of a Joule LANDING at slave slot k)

delta_k :  -Lambda_k + a_k * Gamma_k + rho_k = 0 ,  rho_k >= 0 ,  rho_k*delta_k = 0
           interior (delta_k > 0, arrival cap slack):   m_k = a_k * s_k
           water levels then satisfy:   nubar_k = (a_k*th2/th1) * nu_k
```

Right-tap behaviour (complementary slackness): a tap that is open (`lambda = mu = 0`) equalises the
two water levels; a closed tap (`F = 0`, battery empty / at floor) allows only a level *rise*
(`nu_{n+1} >= nu_n`); a full tap (`F = cap`) allows only a level *fall*. With `E_max = inf` all
`mu = 0`, so `nu` is non-decreasing in time.

Important facts that follow:

* With infinite batteries and a harvest that is high early and lower later (as in the summer array),
  no tap closes before the last slot, so `nu` (and `P`) is flat.
* `P[i] > P[i+1]` is **not** implied by `E[i] > E[i+1]`. It can occur only where the tap is full
  (battery ceiling reached). The rule is about water levels and battery state, not single harvests.
* With `S(delta) = 0` the master never shares (`rho_k = Lambda_k > 0`).
* Sharing only happens when `a_k*th2*hbar_k > th1*h_k / (1 + h_k P_k)`. With `hbar = h` the master
  shares nothing (verified).

---------------------------------------------------------------------------------------------

## 4. The algorithm (`joint_dwf`)

**Block coordinate ascent over "taps".** Not a closed form, not a solver: repeated exact 1-D
maximisations, each clipped to the feasible interval, so the objective never decreases.

Network picture (each arrow is a flow variable):

```
 master   M_0 --f--> M_1 --f--> M_2 --f--> ...      f[k+1] = F_k          (right taps, temporal)
           |          |          |
           | delta    | delta    | delta            d[k]   = delta_k      (down taps, spatial, gain a_k)
           v (x a_k)  v          v
 slave    S_0 --g--> S_1 --g--> S_2 --g--> ...      g[k+1] = Gt_k         (right taps, temporal)
```

### State in the code (0-indexed)

| Code | Math |
|---|---|
| `E[i]`, `h[i]`, `hb[i]`, `a[k]` | `E_i`, `h_i`, `hbar_i`, `a_k = beta*alpha_k` |
| `f[k+1]` | `F_k` master carry k -> k+1 (`f[0] = f[N] = 0` fixed) |
| `capM[k]` | `E_max - E_{k+1}` capacity of master right tap k |
| `g[k+1]` | `Gt_k` slave carry above floor (`g[0] = b0 - B_min`, `g[N] = 0` fixed) |
| `d[k]` | `delta_k` |
| `W` | `B_max - B_min` |
| `xm(i)` | `tau*P_i = E[i] + f[i] - f[i+1] - d[i]` |
| `cS[i]` | `c_i` slave electronics energy (`jp.overhead`; zeros without a load) |
| `capA[k]` | MPPT stored-energy cap (`jp.capA`; inf without a cap) |
| `xs(i)` | `tau*Pbar_i = a[i]*d[i] + g[i] - g[i+1] - cS[i]` |

Because the flows are the variables, the energy-balance equalities hold by construction and the
constraints become bounds: `f >= 0`, `f[k+1] <= capM[k]`, `g >= 0`, `d >= 0`,
`g[k] + a[k]*d[k] <= W` (slave arrival cap), `a[k]*d[k] <= capA[k]` (S5), `xm >= 0`, `xs >= 0`.

### Start point
`f = 0`, `g = 0` (except `g[0]`), `d = 0`: "spend what you harvest, share nothing". If the first slots are
silent (S4), the initial slave excess `g[0]` is carried unchanged up to the first ON slot so the start is feasible.
This is the path with no slave load (bit-identical to the original solver).

With a load (`c > 0`) "share nothing" is infeasible (`g[N] = 0` is fixed), so the start is **just-in-time**:
`f = 0`, `xs = 0`, `d[k] = max(0, c_k - g[k])/a[k]`, `g[k+1] = g[k] + a_k d_k - c_k`; any initial excess beyond the
total drain is spent as data in the first ON slot. If that start breaks a bound (`d[k] > E[k]` with `f = 0`,
`g[k] + a_k d_k > W`, `a_k d_k > capA_k`, or `a_k = 0` with a shortfall) a feasibility LP with the constraints of
`reference_solution_joint` is tried (it may borrow through `f`); if that is infeasible too, `InfeasibleSlaveBudget`
is raised naming the slots and the binding quantity. No constraint is ever relaxed. `JointResult.start` reports
`share-nothing`, `just-in-time` or `lp-fallback`.

### The one equation: `step()`

Moving `t` Joules from node u (loses `s*t`) to node w (gains `r*t`) has derivative zero when the
marginal values balance: `th_u h_u (1 + h_w P_w) = r th_w h_w (1 + h_u P_u)`. This is linear in
`t`:

```
t* = tau * [ r*th_w*h_w*(1 + h_u*x_u/tau) - th_u*h_u*(1 + h_w*x_w/tau) ] / [ h_u*h_w*(th_u*r + r*th_w*s) ]
```

* Right taps (`s = r = 1`, same theta): reduces to `t* = 0.5*tau*(nu_u - nu_w)` (equalise levels).
* Down tap (`s = 1`, `r = a_k`): the interior condition `m_k = a_k s_k`.
* `step` returns 0 if the denominator is not positive (e.g. both weights 0).

The applied move is `t = clip(t*, lo, hi)`, where `[lo, hi]` keeps every constraint of section 3 valid.
A clipped move means that constraint is active, i.e. its multiplier is nonzero (complementary
slackness in code form). **Multipliers are never computed explicitly**; optimality is recognised by
"no move can improve".

### The ten moves at slot k (all inside `joint_dwf`, in this order)

Let `tx[i] = slave_schedule[i]`. Any move whose **slave end** is a silent slot is skipped (that end would
get `Pbar > 0`): move 1 needs `tx[k]`, 2 needs `tx[k-1]`, 3 and 4 need `tx[k]`, 5 needs `tx[k+1]`, 7 needs
`tx[k]` and `tx[k+1]`, 9 needs `tx[k-1]` and `tx[k+1]`. Moves 6, 8 and 10 only touch master data energy and
are never gated.

| # | Path (source -> sink) | Variables that move | Equation enforced |
|---|---|---|---|
| 1 | master k -> slave k (down tap) | `d[k] += t` | `m_k = a_k s_k` |
| 2 | master k -> slave k-1 (in-swap) | `d[k] += t`, `g[k] -= a_k t` | `m_k = a_k s_{k-1}` (used when slave arrival cap is tight) |
| 3 | master k-1 -> slave k (master pass-through) | `f[k] += t`, `d[k] += t` | `m_{k-1} = a_k s_k` |
| 4 | master k+1 -> slave k (master out-swap) | `f[k+1] -= t`, `d[k] += t` | `m_{k+1} = a_k s_k` |
| 5 | master k -> slave k+1 (slave pass-through) | `d[k] += t`, `g[k+1] += a_k t` | `m_k = a_k s_{k+1}` |
| 6 | master k -> master k+1 | `f[k+1] += t` | `nu_k = nu_{k+1}` |
| 7 | slave k -> slave k+1 | `g[k+1] += t` | `nubar_k = nubar_{k+1}` |
| 8 | master k-1 -> master k+1 (carry pass-through) | `f[k] += t`, `f[k+1] += t` | `nu_{k-1} = nu_{k+1}` |
| 9 | slave k-1 -> slave k+1 (carry pass-through) | `g[k] += t`, `g[k+1] += t` | `nubar_{k-1} = nubar_{k+1}` |
| 10 | master k -> master k+1 across slave right tap k (delta shift) | `d[k] += t`, `g[k+1] += a_k t`, `d[k+1] -= (a_k/a_{k+1}) t` | `nu_k = nu_{k+1}` (slave spending `xs(k)`, `xs(k+1)` unchanged) |

Moves 1, 6, 7 are the plain equations. Move 10 is new with the schedule: it **re-times when the master
sends** (e.g. moves `delta` out of a slot and into a later one through the slave battery) without changing what
the slave spends; `step()` is called with sink gain `r = a_k/a_{k+1}` and the move is applied through `apply()`.
Moves 2-5, 8, 9 are the same equations with two variables
moving together. They exist **because single-variable moves can get stuck**: a slot with zero power
(`P = 0` or `Pbar = 0`, a "dead" slot, water level sitting on the floor `1/h`) cannot supply energy,
so no single tap looks profitable although energy should pass *through* that slot; likewise the
slave arrival cap couples `delta_k` with the incoming slave carry. Moves 6-10 need `k+1`, so the last
slot skips them.

### `apply()` and `polish()`

* `apply(fch, gch, dch, src, snk, t_star)` is the generic path mover. Every constraint is linear in
  the flows, so each affected quantity is `v + c*t` and must stay in `[L, U]`; the tightest bounds
  form `[lo, hi]`; the move is `clip(t_star, lo, hi)`.
* `polish()` extends the pass-through idea to **runs of consecutive dead slots** (paths across two or
  more dead slots): master chains, slave chains, and cross paths master -> down tap -> slave. When a slave load is
  attached it adds **family D, the cross-run delta shift**: `d[k] += t`, `g[k+1..j] += a_k t`, `d[j] -= (a_k/a_j) t`
  for any `k < j` (every `xs(q)` unchanged, equation `nu_k = (a_k/a_j)-weighted nu_j`, the same `step()` as move 10, all
  bounds via `apply()`, incl. S5). Move 10 only shifts between adjacent slots and a master slot with a high water
  level in between blocks the chain, although the end-to-end shift pays; with a load the silent runs route energy
  through exactly such slots. The joint_dwf argument `cross_run` (default: on iff a load is attached) switches it, so
  the no-load path stays bit-identical.
  A slot is "dead" if its data energy `<= 1e-6*scale`. It is only called when the fast sweeps have
  stopped moving. Under S4 the slave chain is attempted only if both end slots transmit, and silent slots are
  excluded as slave sinks of the cross paths (they may still be crossed).

### Loop and stopping rule

1. One **sweep** = one pass over all slots trying moves 1-10 (forward on even sweeps, backward on odd).
2. If the largest move in the sweep is `< tol*scale` (`tol = 1e-10`, `scale = mean(E)`), call
   `polish()`. If `polish()` also moves less than `tol*scale`, the KKT conditions hold: converged.
3. `max_iter = 100000` sweeps is only a safety cap. The result reports `iters` and `converged`.

### Post-processing
`P = xm/tau`, `Pbar = xs/tau` (with `max(., 0)` only to remove round-off), `F = cumsum(E - x - delta)`,
`b = B_min + g[1:]`, `b_peak = B_min + g[:-1] + a*delta` (battery right after each arrival, must be
`<= B_max`), rates `sum tau/2 ln(1+hP)` and `sum tau/2 ln(1+hbar*Pbar)`.

### Public API of `build_2D_dwf.py`
`beer_lambert(c, d)`, `thz_absorption(freq_thz)`, `thz_gain(freq_thz, d, h0)`, `JointParams`
(now with `slave_schedule` and per-slot `beta`), `JointResult`, `PhysicalStorageResult`,
`joint_dwf(jp, tol, max_iter, cross_run)`, `InfeasibleSlaveBudget`, `physical_stored_power(jp, r, ...)`,
`solve_with_physical_chain(jp, ...)`, `check_kkt_joint(jp, r)`, `reference_solution_joint(jp)` (cvxpy), the demos
`legacy_demo()` / `physical_demo()`, and one plotting function per panel (each saves a single
PNG to `path`; `title` is an optional figure-level caption). The four slave-load panels
(`plot_slave_load_per_slot`, `plot_slave_energy_breakdown`, `plot_slave_cumulative_energy`, `plot_slave_battery_window`,
all `(jp, r, load, path, title)`) and `write_slave_energy_csv(path, jp, r, load)` take the dict from `slave_load.make_load(jp)`:

| Function | Panel |
|---|---|
| `plot_link_power(jp, r, path, title)` | power over the energy link: sent / received after channel / stored after recharge |
| `plot_transmit_power(jp, r, path, title)` | master vs slave transmit (data) power |
| `plot_slave_battery(jp, r, path, title)` | slave battery level after each slot and right after each arrival, with `B_min`/`B_max` |
| `plot_throughput_per_slot(jp, r, path, title)` | master vs slave throughput per slot [nats] |
| `plot_throughput_cumulative(jp, r, path, title)` | running total of that throughput |
| `plot_harvested_energy(jp, r, path, title)` | energy the master harvests per slot, `E_i` |
| `plot_sharing_energy(jp, r, path, title)` | energy balance of sharing: slave gain `a*delta`, master loss `delta`, net `a*delta - delta` |
| `plot_physical_stored_power(jp, r, ph, path, title)` | arrived optical power vs solver-assumed stored power vs physical MPPT output |
| `plot_physical_slave_battery(jp, r, ph, path, title)` | slave battery per the solver vs re-simulated with the physical chain's arrivals |

`plot_transmit_power` and `plot_throughput_per_slot` shade the silent slots (`_shade_silent`).
`check_kkt_joint` now also returns `slave_silent_slots` and `max_Pbar_in_silent_slots` (S4, should be ~0), and
tests the `delta` condition only at transmitting slots, and, with a load, `slave_overhead_total_J`,
`max_cap_violation_J` (S5, ~0), `mppt_cap_active_slots` (the `delta` condition is skipped where S5 binds),
`min_slave_level_margin_J` (>= 0) and `slave_ledger_error_J` (`(b0-B_min) + sum a delta - sum c - sum tau Pbar -
(b_{N-1}-B_min)`, ~1e-12). `reference_solution_joint` adds `xb[silent] == 0`, the drain `c`, the S5 bound and the same
terminal condition as the solver (`g[N] = 0`); it raises `InfeasibleSlaveBudget` when cvxpy reports infeasible.
`plot_harvested_energy` and `plot_sharing_energy` used to be one combined plot (`plot_energy_balance`),
but `E_i` is normally one to three orders of magnitude larger than the sharing quantities, which
flattened the sharing curves to the baseline; they are now separate figures with independent scales.
Likewise the old three-panel `plot_joint` and two-panel `plot_throughput` were split so each panel is
its own figure (a shared `_new_axis()`/`_finish()` helper pair keeps the styling consistent).

### Physical receive chain (stage 4: post-processing + calibration)

`joint_dwf()` assumes a **linear** tap, `a_i = beta*alpha_i`. The real receive chain is slightly nonlinear
and its efficiency differs from the placeholder `beta`. Two functions bridge this without touching the solver
(Option B of `Instructions.md`):

* `physical_stored_power(jp, r, wavelength_nm=532, mppt_efficiency=0.95, ...)`: (with a load it also subtracts `c_i` from the
  re-simulated battery and clips the stored energy to the MPPT cap, `overflow_mppt`; `physical_chain.py` does not
  saturate at the MPPT rating and is not edited. `eff_phys` stays the unclipped chain gain so calibration is stable.)
  Feeds `r.arrived_power`
  (`alpha*delta/tau`) through `optical_to_electrical()` and returns a `PhysicalStorageResult`: MPPT output
  power/energy, `eff_phys = P_mppt / P_arrived` (NaN where nothing arrives), the slave battery re-simulated
  with the physical arrivals (`b_phys`, `b_peak_phys`), `overflow` above `B_max`, and floor/ceiling violation
  counts (tolerance `battery_tol_J = 1e-6` J, looser than the solver's own residual). Pure post-processing.
* `solve_with_physical_chain(jp, ..., tol=1e-3, max_outer=30)`: fixed-point loop. Solve, measure `eff_phys`
  on slots where `delta` is non-negligible, set per-slot `beta_i = eff_phys_i`, re-solve, until `beta` moves
  less than `tol`. Slots with `delta = 0` keep their previous `beta`. At the fixed point
  `a_i*delta_i/tau == mppt_output_power_i`, so `r.b` equals `ph.b_phys` and B_min/B_max are enforced against
  the real energy. Returns `(jp_calibrated, r, ph, n_outer)`.

Option C (nonlinear tap inside the optimiser) is described in `Instructions.md` and is **not** built.

`thz_absorption` interpolates the measured molecular-absorption table (`thz_freq` [THz] against
`thz_alpha_m` [1/m], the latter converted from the given `thz_alpha_cm`). `thz_gain` turns that into
a channel gain via `h = h0 * exp(-alpha_m*d)`, the same shape as `beer_lambert`'s transmittance but
with a free reference constant `h0` (default now 10.0) standing in for whatever the absorption coefficient
alone does not capture.

### Slave electronics load (stage 5)

Everything here is **off by default**: without a load the solver reproduces the stage-4 numbers bit for bit.

```python
import slave_load
jp_l = slave_load.attach_slave_load(jp, size_storage=True)   # copy of jp + c_i, MPPT cap, eta_tx, window
load = slave_load.make_load(jp_l)                            # components, c, sizing: for plots / CSV / diagnose
jp_l, r, ph, n = solve_with_physical_chain(jp_l)             # or joint_dwf(jp_l)
diagnose(jp_l, r, load=load)
```

* **What is `c_i`.** `slave_load.overhead_profile()` returns camera, MCU, memory (key `flash`, kept for CSV/plot
  compatibility) and MPPT energy per slot, `E = P_active*t_active + P_sleep*t_sleep` (memory:
  `P_active*(t_write + t_read) + P_retain*t_retain`), and `c` = their sum. No TX. The schedule is read from
  `jp.slave_schedule`; ON-slot durations apply where `s_i = 1`, SILENT-slot durations where `s_i = 0`.
  The MPPT quiescent/sleep choice follows the **schedule**, not `delta_i > 0` (about 30 uJ per slot, and a solver-dependent
  `c_i` would be circular).
* **Editing.** Only `slave_config.py`. Durations are scalars or length-N arrays; `None` for a sleep/retain time means
  the rest of the slot; if an explicit sleep leaves a component's times short of `tau` the remainder is treated as powered
  off (0 W) and one line says so. Validation raises `ValueError` naming the field and slot (negative duration, active +
  sleep > `tau`, memory write + read + retain > `tau`, per-slot array of the wrong length, TX window longer than
  `mcu_active_s`, `None` where a time is required). While `TIMING_IS_DUMMY = True` every run prints
  `slave timing: DUMMY values (edit slave_config.py)` (also at the top of `diagnose(load=...)`); flipping the flag removes
  only that banner. Swapping every duration for another value (including per-slot arrays) needs no code edit and changes
  only numbers (tested: same output files, same CSV header, same printed structure).
* **Dummy defaults** reproduce the standalone baseline: silent slot 13.646 mJ, ON-slot non-TX 0.7527 J (camera 0.1506,
  MCU 0.2709, memory 0.3312, MPPT 5.8e-5), `sum(c) = 15.601 J` over the 60-slot demo schedule.
* **`attach_slave_load`** validates `slave_config.TAU_S == jp.master.tau` and the schedule length, scales `h_bar` by
  `eta_tx` (radiated = `eta_tx*Pbar`; `Pbar` stays the electrical TX power), sets `mppt_cap_J` (or `None` with
  `use_mppt_cap=False`) and, with `size_storage=True`, `B_max = B_min + W_std` and `b0 - B_min` = drain of the leading
  silent run. It is not idempotent (attaching twice scales `h_bar` twice). The sizing uses the **placeholder TX energy**
  (0.5 W x 2 s, DUMMY), because the true TX energy is the solver's output; the solver then spends the whole window it is
  given (about 4.5 J of TX per ON slot in `physical_demo` against the 1 J placeholder), so `W_req` is a lower bound, not what
  the solution uses.

Decisions taken where the spec left a question open (defaults stated in the spec; change them in `slave_config.py`):

| Question | Default used |
|---|---|
| TX window convention | **Short burst**: `tx_window_s` = 2 s (DUMMY) inside the read window. The DWF credits `tau/2 ln(1 + hbar*Pbar)` with `Pbar` averaged over the whole slot, so a real burst radiating `xs/t_tx` delivers fewer nats (concavity). The solver is not changed; `attach_slave_load` warns when the TX window is shorter than `tau`. |
| `eta_tx` (THz source efficiency) | 1.0 (DUMMY); folded into `h_bar`. |
| Initial slave charge `b0` | Legacy: `B_min` (unchanged). `size_storage=True`: `b0 - B_min` = leading silent-run drain (0.1774 J at the defaults). |
| Captures in silent slots | None (`SILENT_SLOT.camera_active_s = 0`); a per-slot array puts them in. Holding an image in memory across slots is not modelled. |
| Supercap leakage | Not modelled (at a few uA it competes with the 227 uW sleep floor). |
| `B_min` | Kept from `jp` (20 J in the demo); `size_storage` only sets the window `W`. |

Where the spec / task text differed from the code: the task text calls the module `slave_load_profile.py` (file is
`slave_load.py`); `solve_with_physical_chain` and `PhysicalStorageResult.overflow` already existed (only `c_i` and the MPPT
cap were added); `physical_chain.py` is the constant-Voc GaAs/Ge model (no log/dark-current Voc as `Instructions.md` describes) and
does not saturate at the MPPT rating, so the clip lives in `physical_stored_power`; the spec's start-point fallback names
cvxpy, but the rule "no cvxpy in the algorithm path" is kept by using `scipy.optimize.linprog`; `tx_window_s` was added
to the config (needed for the TX-window warning and the standalone placeholder); the demo comment saying `h0 = 1` was stale
(`thz_gain` defaults to `h0 = 10`); "W below the largest silent drain" does **not** make the problem infeasible, because
silent slots can still receive `delta` and the master can feed the slave just in time (checked against cvxpy): only
`W < c_k` for a single slot does.

### Stage 6: integrated scenario (`integrated_demo`; spec `implementation.md`, report `STAGE6_REPORT.md`)

Off by default: with none of the new fields set the solver runs the legacy arithmetic exactly (every new term sits
behind a Python `if`), and `tests/regression.py check` stays IDENTICAL for `legacy_demo` and `physical_demo`.

* **Roles.** Master: harvest + UOWC energy sharing only, `theta1 = 0` (`tau*P` is unused / spill; `h = 1` is a
  placeholder that only meets weight 0). Slave: PV -> MPPT -> supercap, transmits only THz photomixer bursts to a third-party
  receiver. `SLAVE_SCHEDULE` releases at most one **event** per slot (camera 0.5 s, PSRAM write + read 10.5 ms each, MCU
  527.3 ms, a 16.777 ms burst); PV/MPPT charge in every slot.
* **Link.** `thz_hbar`: `coupling = FSPL + absorption - G_tx - G_rx`, `hbar = ETA_TX_PM / (10^(coupling/10) N)`,
  `N = k_B T F R_s` = 4.0039e-11 W; Friis only for `d >= d_nf = lambda/(4 pi) 10^((G_tx+G_rx)/20)` (clamped with a
  warning). `THZ_PRESET` "0.3THz" (10/10 dBi): `hbar` = 7199 1/W at 1 mm; "0.1THz" (5/5 dBi): 6479 1/W. `hbar` already
  contains `ETA_TX_PM`, so it is applied once.
* **Load.** `event_profile`: event slot `c_i = sum [P_a t_a + P_s (tau - t_a)]` (photomixer active term excluded) =
  198.41 mJ, sleep slot `sum P_s tau` = 15.962 mJ (PSRAM standby; 3.218 mJ in deep power-down).
* **Burst (S6, rate window).** `x_i = tau*Pbar_i` is the electrical burst energy, `x_i <= x_cap = P_PM_ELEC_W t_tx` =
  3.3554 mJ in event slots and 0 elsewhere (`tx_cap_J`); the rate is `th2 (t_tx/2) ln(1 + hbar x_i/t_tx)`.
  `step()` balance: `th_u h_u (1 + h_w x_w/T_w) = r th_w h_w (1 + h_u x_u/T_u)`, still linear in `t`.
  S6 / S7 / M5 enter moves 1-5, 7, 9, `apply()` (and so polish families B, C, D, move 10) as one more bound each:
  S6 `xs(j) <= X_j` (multiplier `kappa_j`), S7 `xs(j) >= Lf_j` (`iota_j`), M5 `d[k] <= Dc_k` (`rho'_k`); the slave
  stationarity becomes `s_j - kappa_j + iota_j = Lambda_bar_j`. `check_kkt_joint` skips the delta test where S6 or M5
  binds and uses an absolute gap when `theta1 = 0`.
* **beta.** `beta_i = R x MPPT_EFF x ETA_SC`: `physical_stored_power(..., supercap_eff=ETA_SC)`; the calibration loop is
  unchanged (0.379762 at `ETA_SC = 1`, x 0.95 exactly with `ETA_SC = 0.95`).
* **Refusal** (`solve_with_refusals`). An event with `x_i < x_min` (1 mW radiated, 0.16777 mJ; `x_i = 0` included) is
  refused: its slot returns to sleep load and the event moves to the first later slot without its own event, at most
  `MAX_DELAY_SLOTS` after its release, else it is dropped. An unaffordable event load (or `fixed_bias`) raises
  `InfeasibleSlaveBudget`; the loop then refuses the event at or before the first slot where even the fullest supercap
  trajectory falls below `B_min`. Pass cap `#events (MAX_DELAY_SLOTS + 1) + 1`. Greedy, not a combinatorial optimum.
* **Maximal sharing** (`maximal_sharing`, post-pass, `theta1 = 0` only): with every `x_i` fixed, the master sends in
  each slot as much as the ceiling `B_max`, S5, M5 and its own energy allow. Objective, `x` and events are unchanged;
  the supercap ends above `B_min` and every slot is stopped by exactly one of those bounds (`delta = 0` when full).
* **fixed_bias** (`PM_DC_MODE`): every event costs `P_PM_ELEC_W t_tx` inside `c_i`, `x = 0` for the solver; the link
  report credits the constant `(t_tx/2) ln(1 + hbar P_PM_ELEC_W)` per run event.

---------------------------------------------------------------------------------------------

## 5. Stage 1 (master only, `build_dwf.py`) in brief

Same tap idea with a single node: flows `f_n`, `x_i = E_i + f_{i-1} - f_i`, one move per tap
(equalise `nu_i = P_i + 1/h_i`) clipped by `0 <= f <= E_max - E_{n+1}` and `x >= 0`. No pass-through
moves. Converges slowly for `E_max = inf` (about 5700 sweeps on the summer array) but exactly.
Summer array, `tau = 60`, `h = 1/50`: `E_max = inf` gives flat `P = 53.38 W`, throughput 1307.4416 nats;
`E_max = 3300 J` gives 1307.4255 nats. It matched an independent solver in these cases and on 8 random
per-slot-`h` cases. Its scipy SLSQP reference stalls for `E_max = inf` (use trust-constr or cvxpy).

---------------------------------------------------------------------------------------------

## 6. Placeholders and assumptions (NOT from the user's data)

The demo uses these; replace them when real values exist.

| Item | Value | Status |
|---|---|---|
| `B_min`, `B_max` | 20 J, 1500 J (earlier demos: 200/5000, then 200/3000) | chosen for the current demo; the slave battery is small enough that the ceiling binds |
| `E` | `Harvested_energy_winter` (J per 1-min slot; the summer array is still available) | given by the user |
| `LINK_DISTANCE_M` | 0.0005 m, shared by the THz data link and the optical energy link (was 0.001) | given by the user |
| `alpha` | `beer_lambert(0.046, 0.0005)`: 450 nm blue, c = 0.046 1/m (was 0.0399) | given by the user |
| `beta` | initial 0.4, then **per-slot, calibrated** by `solve_with_physical_chain()` to the photodiode+MPPT secant gain (about 0.380 to 0.400 in the demo) | the 0.4 start is a PLACEHOLDER; calibrated values come from `physical_chain.py` (532 nm, MPPT 0.95) |
| `h`, `hbar` (master/slave THz data link, symmetric) | `thz_gain(0.30, 0.0005)` = h0 * exp(-12300 * 0.0005) | alpha_m given by the user (0.30 THz, 12300 1/m); **h0 = 10.0 is a PLACEHOLDER** (default changed from 1.0) |
| `theta1`, `theta2` | 1 and **3** (default changed; slave rate weighted 3x) | assumption (weighted sum-rate) |
| `slave_schedule` | 30-slot pattern (13 silent, 4 ON, 4 silent, 3 ON, 1 silent, 3 ON, 2 silent) repeated over 60 slots: 20 ON, 40 silent | assumption / test pattern (slave transmits in 1/3 of the slots) |
| `E_max` (master) | inf | assumption (user said it can go to infinity) |
| `b0` (initial slave charge) | `B_min` (legacy); lead-run drain with `size_storage=True` | ASSUMPTION, asked, not yet answered |
| Slave durations (`ON_SLOT`, `SILENT_SLOT` in `slave_config.py`): capture 0.5 s, MCU awake 4.5 s, memory write/read 2 s each, MPPT quiescent whole slot, TX window 2 s | dummies | **DUMMY** (marked in the file; `TIMING_IS_DUMMY`) |
| Image 2 MB, memory 1 MB/s, link 10 Mbit/s, TX stage 0.5 W (`STANDALONE`) | dummies | **DUMMY**, standalone illustration / `derive_dummy_timing` only, never on the solver path |
| `ETA_TX` | 1.0 | **DUMMY** |
| `STORAGE_MARGIN` | 2.0 | **DUMMY** (design window = margin x `W_req`) |
| Datasheet powers (camera 0.30/10 uW, MCU 60 mW/16 uW, memory 80 mW/0.2 mW, MPPT 0.96/1.44 uW, input 3.3 V x 100 mA, eff 0.95, 2.2-5.2 V) | stated powers (not V x I) | real (`datasheet_check()` reports the V x I mismatches) |
| Receive chain | GaAs/Ge cell (Spectrolab datasheet), FF 0.82, Voc 1.025 V constant, array sized by intensity | datasheet numbers; Voc fall-off at low illumination is NOT modelled (see `physical_chain.py` docstring) |
| Arrival timing | energy sent in slot k is usable by the slave in slot k | ASSUMPTION (as in Gurakan; acoustic/optical delay negligible vs 60 s) |
| Stage 6: `ETA_SC` (0.95), `ETA_LD` (1.0), `LASER_P_OPT_MAX_W` (None = M5 off), `MAX_DELAY_SLOTS` (5), `D_UOWC_M` (1 mm) | | **DUMMY / PLACEHOLDER** (`slave_config.DUMMY_VALUES`, banner on every event-model run) |
| Stage 6: event powers and durations, photomixer (20 mW / 200 mW / 1 mW floor), 16-QAM 8 MiB, `R_s`, noise (290 K, NF 10 dB), `ALPHA_THZ`, presets, `D_THZ_M` = 1 mm, 0.047 F supercap, `MPPT_EFF` | | given by the user / the notebook (real); the receiver noise figure and bandwidth are notebook defaults, not the UWDrone receiver's |
| Stage 6: master `h = 1` | | PLACEHOLDER, inert with `theta1 = 0` |
| Stage 6: `b0` | `B_min` (supercap starts at 2.2 V) | ASSUMPTION |
| Stage 6: beta calibration wavelength | 532 nm (`PV_CAL_WAVELENGTH_NM`, same as the legacy demos; the UOWC attenuation is the 450 nm value) | ASSUMPTION |

The sharing test is `R = a*(th2/th1)*(hbar/h)*(1+h*P0)`. With `th2/th1 = 3` and `h0 = 10` it is about 2.75
(> 1), so sharing pays in the current demo, unlike the earlier `th2 = th1`, `h0 = 1` demo where `R < 1` forced
`delta = 0` everywhere. `h0` and `theta2` are therefore the knobs that decide whether anything is shared; do
not present demo numbers as results about a real system.

---------------------------------------------------------------------------------------------

## 7. Current results (demo: winter array, schedule, calibrated beta)

> **Numbers at the current code (2026-10-08).** The baseline in `tests/baseline/` was re-captured from HEAD on
> 2026-10-08: the committed `_demo_problem` uses the 7-event schedule, `beta` 0.9 start, `B_min` 200 J, `B_max` 1500 J.
> `legacy_demo`: objective **1781.9933** (master 1303.2194, slave 159.5913), 505 iterations, calibration 4 outer
> iterations, `beta` in [0.3798, 0.9000], solver vs physical battery 5.3e-12 J. `physical_demo`: objective 1537.8144,
> 1633 iterations, W = 2.442 J, sum c = 2.735 J. The paragraphs below are the earlier (stage-4/5) runs and keep their
> numbers as history.
>
> **`integrated_demo` (stage 6).** 60 slots, 14 events, preset 0.3THz at 1 mm, 0.047 F supercap: all 14 events run in
> their release slot at `x_cap` = 3.3554 mJ (20 mW radiated, SNR 31.58 dB, 16-QAM BER 5e-65), 0.0610 nats each,
> objective 2.562422 nats (= 14 x 0.0610 x theta2 3), equal to cvxpy; 0 refusals; `beta` = 0.360774; master sends
> 11.27 J of 227,633 J (0.005%), the rest is spill; after the post-pass the supercap is refilled to `B_max` in every slot
> (stopped by the ceiling in all 60) and ends 0.506 J above the floor; solver vs physical battery 1e-16 J. Preset 0.1THz:
> `hbar` 6479 1/W, objective 2.525328; `fixed_bias`: same events, constant 0.8541 nats credited.

`iters = 1410`, converged, objective **1902.8194 nats** (master 1141.7442, slave 253.6917; objective =
th1*master + th2*slave). Master data power 41.58 W mean (min 40.91), slave data power 8.22 W mean (zero in
the 40 silent slots, about 7e-16 per `check_kkt_joint`), shared 21.65 W mean = **34.2% of the harvest**
(77945 J sent of 227633 J). Ledger: sent 77945 J -> channel loss 1.8 J, recharge loss 48344 J, stored 29600 J
= slave data 29600 J (battery change 0). All 20 transmitting slots hit the slave ceiling on arrival (`B_max`);
`delta_active_slots = 20`, slave at the floor in 57/60 slots, master taps closed in 1/59.
`check_kkt_joint`: final master carry 0, slave end level 20 J, `delta`-condition violations 0.
Calibration: 4 outer iterations, `beta` in [0.3798, 0.4000], max |solver battery - physical battery| =
3.9e-12 J, stored energy solver = physical = 29600 J, no floor/ceiling violations, overflow 0.

### `physical_demo` (slave electronics attached, DUMMY durations)

Same master (winter harvest, `tau = 60`, `h`), same schedule, same placeholder `h0` and `theta2`, but
`attach_slave_load(jp, size_storage=True)`: window `W = 5.217 J` (0.47 F supercap over 2.2-5.2 V), `b0 - B_min = 0.1774 J`,
`sum(c) = 15.6007 J`, MPPT cap 18.81 J. Result: objective **1538.8069**, 1650 iterations, converged; master 1535.9518, slave
0.9517 nats; slave data energy **89.3 J** (against 29,600 J in `legacy_demo`), master -> slave **275.7 J** (against 77,945 J),
stored 104.7 J, calibration in 4 outer iterations (`beta = 0.3798`), solver and physical batteries agree to 5e-14 J, no floor,
ceiling or MPPT overflow, slave ledger error 1e-14 J, every ON slot ends at the window. The huge difference to `legacy_demo`
is expected: the legacy stage-4 flows (29.6 kJ of slave data, 24.7 W of stored power) are far outside what the real electronics
and the MPPT can absorb. It is the point of the exercise, not a solver bug; `h0`, `theta2` and the harvest were **not** retuned,
and the numbers are at DUMMY durations and placeholder `h0`/`theta2`.

**The slave battery trajectory is not unique.** `(P, Pbar)` are unique (strict concavity), but the
split of carried energy between the master battery and the slave battery is not when both have slack
(same objective, different `F` and `b`). Do not write tests that pin `F` or `b` exactly unless a
tie-breaker is added.

---------------------------------------------------------------------------------------------

## 8. What has been verified

* **Reduction to master-only:** `theta2 = 0`, or `beta = 0`, gives the `build_dwf.py` result (max
  difference about 1e-8 W).
* **Independent solver (cvxpy, `reference_solution_joint`):** default, no-sharing (`hbar = h`) and a
  tight-battery case (window 200-900 J, `b0 = 500`, time-varying `alpha`, `hbar`, finite `E_max`) all
  match to about 1e-6 in objective.
* **Randomised comparison, 25 extreme cases** (per-slot random `h`, `hbar`, `alpha`, `E`, weights, tight
  batteries, finite `E_max`): 22 exact, 3 ended about 0.1% below optimum (gaps 0.3-1.9 of about 2000).
  Those need joint rerouting over several slots that no implemented move covers. Realistic smooth
  inputs (constant `h`, smooth `alpha`) have not shown this.
* **Invariants:** objective non-decreasing in `B_max` and saturating once `B_max` exceeds the peak battery
  level; `B_max = 5000` vs `1e6` identical; scaling `E, B` by c and `h, hbar` by 1/c scales `P` by c;
  flat harvest gives flat powers; energy accounting closes to about 1e-11 J on both nodes.
* **Published example (Gurakan Sec. V-B):** E = [0,12,0], slave harvest Ebar = [6,6,0], alpha = 1,
  theta1 = theta2, infinite batteries: optimum P = [0,4.8,4.8], Pbar = [4.8,4.8,4.8], total delta = 2.4
  (the split of delta across slots is not unique). A scratch copy of the solver (one-line change adding
  slave harvest) reaches this: P within 0.002 after 3 sweeps, converged in 10 sweeps. The paper's
  transient values from its tap orderings are *not* expected to match the sweeps (the Lagrangian gives
  the optimum, not a path).
* **Gurakan Sec. VIII-A relay example 1** as printed in the provided PDF is inconsistent (cumulative
  `P + delta` exceeds cumulative `E`); example 2 is consistent but is a relay-throughput problem with
  data causality, which this code does not model. Do not use example 1 as ground truth.
* **Slave load (stage 5)**, checked against `reference_solution_joint` (extended first, with `c`, S5 and the terminal
  condition; with `c = 0`, `cap = None` it matches the old reference to 4e-11 relative on the default, no-sharing and tight cases):
  * *Equivalence*: `c = 0`/`cap = None` and `c = zeros` vs the ORIGINAL solver on 250 random cases (25 per family): 250/250
    bit-identical (arrays, `iters`, objective); `legacy_demo` reproduces 1902.8194 / 1410 and every legacy PNG byte for byte.
  * *Random cross-check*, 25 cases per family (N = 8-14, random `h`, `hbar`, `alpha`, `E`, `c`, weights, windows, caps; no case
    infeasible, none non-converged, none above the reference): random per-slot `c` 23 exact / 2 below (max 5.5e-4 relative),
    scalar `c` 24/1 (1.2e-4), all ON 25/0, one ON slot 21/4 (2.2e-3), leading/trailing silent runs 25/0, `[1,0]` 22/3 (1.1e-2),
    tight `W` 19 exact + 1 within 1e-6 / 5 below (3.3e-4), per-slot `beta` 21/4 (9.6e-3), finite `E_max` 25/0, S5 binding 25/0 (S5
    binds in all 25). Total: **230 exact, 1 within 1e-6, 19 below the optimum (7.6%), 0 above**.
  * *Gap rate before and after the cross-run move* (family D): 81/250 below (32%, 167 exact, max 1.1%) with `cross_run=False`, 19/250 after (230 exact). The same
    generator with `c = 0` shows the old solver already stalls on it (50/250 below; 18/250 with `cross_run=True`, same worst
    case 1.1%), so the remaining gaps are mostly the pre-existing corner-stalling of section 9, not the load. They need
    random per-slot `h`/`alpha`/`beta` and tight windows; **realistic inputs** (winter and summer harvest, `h` constant,
    run-structured schedules of 60 slots, dummy-timing `c`, S5 on, with and without `size_storage`): 20/20 exact.
    A generalisation of family D that takes the master end node through dead master slots was tried and dropped: it did not
    finish the random matrix in 10 minutes (it multiplies the `apply()` calls of every polish).
  * *Invariants* (random cases with generous `W`): objective non-increasing in `c` (scale 0, 1, 2, 5), non-decreasing in `W` and in the cap: 39/39 each for cvxpy and for the solver (1 of 40 cases skipped as infeasible at 5x `c`); scaling `E, B, c, cap` by `s` and `h, hbar` by `1/s` scales `P, Pbar` by `s`: 38/39 within 1e-5 (the other 1.2e-4, a solver that stalls at a slightly different point); energy accounting on both nodes closes to 3.9e-12 J (39/39 under 1e-8); `Pbar = 0` in silent slots 39/39; `0 <= Gt` and arrival `<= W`, S5 respected 39/39.
  * *Infeasible inputs* raise `InfeasibleSlaveBudget` with slot and quantity: `W` below a single slot's drain, `c_k` above
    `capA`, `a_k = 0` with `c_k > 0`, zero harvest when sharing is needed. `W` below the longest silent-run drain is feasible
    (silent slots can be fed) and solver and cvxpy agree on it (145.0069 nats).
  * *Dummy-to-real swap*: every duration in `slave_config.py` changed (including per-slot arrays for the camera and the silent
    slots) and `TIMING_IS_DUMMY = False`, `python build_2D_dwf.py physical_demo` re-run with no code edit: same files, same CSV
    header, same printed structure, only numbers (and the banner) differ.
* **Schedule (S4):** the demo's `max_Pbar_in_silent_slots` is about 7e-16 and the `delta` condition holds at
  every transmitting slot. `reference_solution_joint` includes the S4 constraint, but a systematic cvxpy
  comparison over random schedules (and for move 10) has NOT been run yet.
* **Physical calibration:** at the fixed point the solver battery and the re-simulated physical battery agree to
  about 4e-12 J (section 7).
* Comment-only edits were verified by comparing the AST with docstrings stripped and re-running the demo (done for the stage-3 code; redo
  it against the current outputs, see section 10).
* **Stage 6** (details and numbers in `STAGE6_REPORT.md`):
  * *Legacy unchanged*: `tests/regression.py check` IDENTICAL for `legacy_demo` and `physical_demo` after every phase
    (arrays bit-for-bit, stdout, PNG/CSV hashes); the only differences are additive (new fields at `None`, new names,
    trailing parameters with defaults).
  * *Notebook*: with `LEGACY_GAINS = True` all 91 compared arrays (tables, BER theory, seeded `simulate_link`) equal the
    original notebook **bit for bit** (tolerance 0).
  * *Numbers*: event slot 198.412 mJ, sleep slot 15.962 mJ (dpd 3.218 mJ), sum c = 1.7560 J per 30 slots, `t_tx`
    16.777216 ms, `x_cap` 3.3554 mJ, `x_min` 0.16777 mJ, N 4.0039e-11 W, `hbar` 7199 / 6479 1/W, 0.0610 nats per event.
  * *cvxpy parity* (`regression.py parity`, 25 random cases x 9 families): **0 above the optimum**; `integrated_like`
    (theta1 = 0, events, c, S6, burst rate) 25/25 exact; the other families 0-20 exact with gaps 1e-6..1e-1. The gaps
    are the pre-existing stall (HEAD gives bit-identical objectives on the same cases), mostly in a regime where the
    master gives everything (`P = 0` everywhere) and the optimum re-routes `delta` between down taps through the master
    carry, a path no move covers.
  * *Refusal loop* (`regression.py events`, 20 cases x 6 kinds, N = 7-10): all converge; brute force over every
    placement matches 97/100 (gaps: 1 placement with +0.0129 nats, 2 fixed_bias cases with one more event); bookkeeping,
    delays and `x >= x_min` hold; post-pass: objective and every `x` unchanged (0), every slot stopped by one bound,
    `delta = 0` in full slots, level inside the window, an LP (maximise sum b with x fixed) agrees to 2e-15 J.
  * *Stage-6 checks* (`regression.py stage6`): the preset switch changes only `h_bar`; ledgers close to 2.4e-16 J,
    `Pbar` in sleep slots and `x - x_cap` <= 3.2e-16 J; objective per period identical for N = 60 and 600. One
    monotonicity failure: a bigger supercap (0.1 F) LOWERED the objective of a scarce case from 0.549 to 0.183 because
    joint_dwf stalled (cvxpy: 0.549 both); on 32 scarce feasible cases joint_dwf is exact in 29 and 33% below in 3 (one
    event left at x = 0), all from the LP-fallback start. See open item 18.

---------------------------------------------------------------------------------------------

## 9. Known limits and gotchas

* **Not guaranteed optimal on extreme inputs** (section 8). If an unusual configuration matters, check
  the objective against `reference_solution_joint`; there is no built-in optimality certificate other
  than `check_kkt_joint` (which only tests feasibility and the `delta` condition).
* **Speed:** pure-Python loops. About 10 s for N = 60 with `E_max = inf` (slow diffusion of energy along
  the master chain). Larger N or tighter `tol` will be slower; vectorising the sweep would be the fix.
* **Slave load limits.** (1) *Averaging convention*: the DWF credits the slave rate with `Pbar` averaged over the whole slot;
  a TX burst shorter than `tau` delivers fewer nats (warned at attach time, solver unchanged). (2) *Ceiling-then-drain*:
  the arrival is assumed to come first and the drain after it (ceiling at arrival, floor at end of slot), conservative for the
  ceiling. (3) *Silent-run coupling*: energy is routed through silent slots that now drain; the plain moves can stall there, the
  cross-run delta shift removes most of it (section 8) but not all on extreme random inputs. (4) The result depends on DUMMY
  durations and on `h0`/`theta2`; do not read `physical_demo` numbers as results about the real system. (5) `size_storage`
  uses the placeholder TX energy, so its window is a lower bound for what the solver then uses. (6) Supercap leakage and
  images held across slots are not modelled. (7) The LP start fallback uses scipy and a slightly looser feasibility tolerance (1e-9).
* **No multipliers are returned.** They can be recovered from the water levels (`Lambda = th/(2 nu)`),
  but only for slots with positive power.
* **Slave has no harvest input** (`Ebar`); the published examples all have one. Adding it is small:
  `xs(i)` gains `+ Ebar[i]` and the arrival cap becomes `g[k] + a[k]*d[k] + Ebar[k] <= W` everywhere
  (`apply`, moves 1-5, 7, 9, `check_kkt_joint`, `reference_solution_joint`, `b_peak`).
* `check_kkt_joint(..., tol)`: `tol` is unused.
* `dwf_diagnostics.diagnose()` assumed a scalar `jp.beta` in a couple of print lines (noted in
  `Instructions.md`); that line now uses the array summary (`_rng`) and the mean for the dB figure.
* `solve_with_physical_chain` calibrates only slots where `delta` moves energy; other slots keep the initial
  `beta`, which is harmless because they carry no energy. Whether the secant gain is a good tap model at other
  operating points is untested (that is Option C's job).
* `B_min` only matters when `B_max` is tight or `b0` is small; with `B_max = 5000` the objective is
  identical for `B_min = 0` and `200`. Do not read that as a bug.
* Units: `delta` is **energy** (J), not power. `E` values are J per slot; `E/tau` is watts.
* Do not assume `P[i] > P[i+1]` follows from `E[i] > E[i+1]` (see section 3).
* The moves rely on `a[k] > 0`; slots with `a[k] = 0` skip moves 1-5.

---------------------------------------------------------------------------------------------

## 10. Conventions for changing the code

* **Functionality-preserving edits (comments/docstrings/renames):** verify with the AST check
  `ast.dump` of old vs new with docstrings removed, and re-run the demo (objective must stay
  1902.8194, `iters = 1410`, with the current demo settings of section 6).
* Keep 0-indexed slots and the `f[k+1]`, `g[k+1]` index convention; `f[0]`, `f[N]`, `g[N]` are fixed.
* Every move must remain "exact 1-D maximiser of a concave function, clipped to feasibility", so the
  objective can never decrease. A new move needs: effect on `xm`/`xs`, the marginal-balance equation,
  and every bound (which multiplier each corresponds to).
* Keep math in docstrings/comments as plain ASCII (`nu = P + 1/h`, `tau*P_i = E_i + F_{i-1} - F_i - delta_i`).
* Any change to the solver should be re-checked against `reference_solution_joint` on the default
  case, the no-sharing case, the tight-battery case and a batch of random cases; report failure rates,
  not just single successes.
* Minimal cross-check pattern:

```python
import numpy as np
from build_dwf import MasterParams, Harvested_energy_summer
from build_2D_dwf import JointParams, beer_lambert, joint_dwf, reference_solution_joint

m  = MasterParams(E=Harvested_energy_summer, E_max=np.inf, tau=60.0, h=1/50)
jp = JointParams(master=m, alpha=beer_lambert(0.2, 5.0), beta=0.8, B_min=200.0, B_max=5000.0, h_bar=0.1)
r  = joint_dwf(jp)
_, _, _, ref = reference_solution_joint(jp)
assert r.converged and abs(ref - r.objective) < 1e-5 * abs(ref)
```

---------------------------------------------------------------------------------------------

## 11. Open items (suggested next steps, none started unless noted)

1. Add an optional slave-harvest input `Ebar` and turn the Gurakan Sec. V-B example into a test
   (expected: P = [0,4.8,4.8], Pbar = [4.8,4.8,4.8], sum(delta) = 2.4).
2. Write a pytest suite from section 8 (reduction tests, cvxpy cross-checks over many seeds, invariants,
   monotonicity in `B_max`, `beta`, `alpha`, `E`; edge cases N = 1, zero-harvest slots, `max_iter` reached).
3. Replace placeholders with real `alpha_i` (Beer-Lambert with a water-type `c`; e.g. one survey table
   gives c at 520 nm of 0.15 pure sea, 0.25 clear ocean, 0.50 coastal, 1.50 turbid harbor 1/m; other
   sources differ by wavelength), real `h_i`, `hbar_i`, and decide `b0`.
4. Public data for realistic tests: NREL MIDC 1-minute solar irradiance (`E_i = G_i * A * eta * tau`),
   NSRDB / PVGIS hourly, CRAWDAD columbia/enhants light traces; Watermark / ACommSet for acoustic
   channel gains (not Beer-Lambert). None has been downloaded or used yet.
5. Load the other seasons' arrays ("similar arrays" to be supplied by the user).
6. Compute the minimum useful slave capacity `B*` (peak battery of the unconstrained optimum) and the
   sensitivity `d(objective)/d(B_max) = sum(mu_bar)`.
7. Optionally add a battery tie-breaker so `F` and `b` become unique; optionally a master floor `E_min`.
8. Extensions discussed but not built: one-slot energy-transfer delay, data-causality (relay) objective,
   max-min or master-only objectives, time-varying `c` per slot.
9. Performance: vectorise the sweeps.
10. Calibrate `h0` in `thz_gain` (or replace pure absorption with a full THz path-loss model:
    spreading loss + molecular absorption + antenna/beam gains). `h0 = 10` and `theta2 = 3` currently make
    sharing profitable; until `h0` is real, the THz/optical demo numbers (section 6) show the link model
    wired in correctly, not a realistic operating point. **Stage 6: done for the integrated scenario**: `thz_hbar`
    is the full budget (FSPL + absorption - antenna gains, thermal noise, near-field clamp). The legacy demos keep
    `thz_gain` and `h0 = 10` on purpose (bit-for-bit baselines).
11. Verify S4 and move 10 against `reference_solution_joint` over many random schedules (all ON, a single ON
    slot, leading/trailing silent runs, per-slot `beta`), reporting failure rates as in section 10.
12. Implement Option C of `Instructions.md` (nonlinear photodiode/MPPT tap inside the optimiser) if the
    secant-gain calibration proves inadequate; add Voc fall-off at low illumination to `physical_chain.py`.
14. Replace the DUMMY durations in `slave_config.py` with measured values (set `TIMING_IS_DUMMY = False`), and the TX stage
    (`ETA_TX`, TX window, image size, memory/link rates) with the real hardware; decide whether captures move into silent
    slots and model the memory holding an image across slots. **Stage 6**: the event model now uses the real event
    timing (camera, PSRAM, MCU, 8 MiB 16-QAM burst); still DUMMY there: `ETA_SC`, `ETA_LD`, `LASER_P_OPT_MAX_W`,
    `MAX_DELAY_SLOTS`, `D_UOWC_M` (`slave_config.DUMMY_VALUES`). The stage-5 slot model keeps its DUMMY durations.
15. If the remaining random gaps matter, extend the cross-run move (master end node through dead master slots) in a form that
    does not blow up the polish cost; report the gap rate before and after as in section 10.
16. Slave supercap leakage, and a rerun of `physical_demo` with the real `h0`/`theta2`. **Stage 6**: leakage, an image held
    across slots and the receiver's real noise figure / bandwidth are still not modelled; the supercap storage loss is in
    `beta` (`ETA_SC`).
17. **Missing move: master-carry delta shift** (`d[i] -= t, f[i+1..k] += t, d[k] += t`: re-time sharing between two down
    taps through the master carry, slave ends `xs(i)`, `xs(k)`). Its absence explains most below-optimum cases of the
    stage-6 parity matrix (the "master gives everything" regime). Not added: the spec forbids ad-hoc moves without a decision.
18. **Stage-6 stall with scarce master energy.** When the master's harvest is the binding resource (fractions of a joule
    per slot) joint_dwf can stop with an event at `x = 0` (33% below cvxpy in 3 of 32 cases; a 3-tap re-route is needed).
    The refusal loop then refuses an event the optimum could run. The integrated demo is not affected (kJ of master
    energy, exact against cvxpy). Options to decide: a cvxpy check of every refusal, a new multi-tap move, or a better
    start than the LP fallback for `theta1 = 0`.
19. **Speed.** polish family D costs O(N^2) `apply()` calls of O(N) length, i.e. O(N^3): joint_dwf on the integrated
    problem takes 0.28 s at N = 60, 28 s at N = 240, 216 s at N = 600; N = 6000 would be about 80 h (not run). The O(N)
    parts (event load, queue, post-pass, link table) take 1-22 ms at N = 6000. Deferred: vectorised sweeps or
    rolling-horizon blocks.
20. The refusal pass cap is `#events (MAX_DELAY_SLOTS + 1) + 1` (the spec's `#events MAX_DELAY_SLOTS + 1` is one
    refusal per event short of the worst case).
13. The schedule is a fixed input. Choosing which slots the slave transmits in would be a combinatorial
    extension and is not attempted.

---------------------------------------------------------------------------------------------

## 12. Notation cheat sheet

`tau` slot length; `E_i` harvest; `E_max` master capacity; `P_i`, `Pbar_i` master/slave data power;
`delta_i` shared energy; `alpha_i = exp(-c d)`; `beta` recharge efficiency; `a_i = beta alpha_i`;
`h_i`, `hbar_i` link gains over noise; `th1`, `th2` rate weights; `F_n` master carry; `b_n` slave level;
`Gt_n = b_n - B_min`; `W = B_max - B_min`; `nu = P + 1/h` master water level; `nubar = Pbar + 1/hbar`;
`lambda_n, mu_n` master causality/capacity multipliers; `lambda_bar_n, mu_bar_n` slave floor/ceiling
multipliers; `eta, eta_bar, rho` non-negativity multipliers for `P, Pbar, delta`;
`Lambda_k` master tap term; `Lambda_bar_k` slave tap term; `Gamma_k = Lambda_bar_k - mu_bar_{k-1}`.
