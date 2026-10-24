# implementation.md: integrate the THz modulation notebook and the new slave model into the Master + Slave DWF

Audience: Claude Code, working in `advanced_DWF/` (resume with `claude --resume c97f929b-cf69-483f-9bf1-54f542f3a0b6`).
Read `README.md` of the DWF project first, then this file. Source of truth for decisions: the approved plan
(`scalable-dreaming-kahn.md`) and the conversation of 2026-10-08. Where this file and the README disagree, this file wins
for the new scenario; the README still describes the legacy behaviour, which must not change.

---------------------------------------------------------------------------------------------

## 0. Working rules (read before touching code)

1. **No new script files.** Add features to the existing files named in section 5. If one of them becomes unmanageable, stop and say which and why before splitting it.
2. **Everything new is off by default.** With no new setting touched, `python tests/regression.py check` must print `IDENTICAL to baseline` (all arrays bit-for-bit, all PNG/CSV hashes, the `build_2D_dwf` API). Run it after every phase. Re-capture the baseline only for the new `integrated_demo`, never for the legacy demos.
3. **Never modify `build_dwf.py` or `physical_chain.py`.** `dwf2d/`, `build_2D_dwf.py`, `slave_config.py`, `slave_load.py`, `dwf_diagnostics.py`, the notebook and `tests/` may change.
4. **The README names are assumptions until confirmed.** Phase 0 lists every function/file this document uses and confirms or corrects it. Do not guess a signature; read it.
5. Keep the README conventions: 0-indexed slots, `f[k+1]` / `g[k+1]` carries, `f[0]`, `f[N]`, `g[N]` fixed in the solver, ASCII math in docstrings, units J / s / W.
6. Every new solver move or bound must state: effect on `xm` / `xs`, the marginal-balance equation, every bound and its multiplier (README section 10).
7. Mark every value that is not measured with the DUMMY mechanism of `slave_config.py` (`TIMING_IS_DUMMY` banner). Real values taken from the notebook or the user are not DUMMY.
8. At the end of each phase report: files touched, what the gate printed, anything that deviated. Stop and ask if a gate fails; do not weaken a gate.
9. Do not retune `h0`, `theta2`, the harvest array or any legacy default to make a result look better.

---------------------------------------------------------------------------------------------

## 1. What is being built (one paragraph)

The master only harvests and shares energy over UOWC (no data link). The slave receives that optical energy through
PV, MPPT and a supercap, and transmits **only** over a THz photomixer link (16-QAM, 8 MB payload). The slave's
`SLAVE_SCHEDULE` marks the slots in which one transmission **event** is released (not 60 s of transmitting). An event is
about 0.53 s of camera, MCU and PSRAM activity with a 16.78 ms THz burst inside it; everything sleeps the rest of
the 60 s slot. `joint_dwf` stays the only decision-maker for how much the master shares and how much each burst
spends. An event whose affordable burst falls below the 1 mW radiated floor is refused and rescheduled to the next available
slot. The master stops sharing when the supercap reaches `B_max`.

## 2. Decisions that are fixed (do not reopen)

| Topic | Decision |
|---|---|
| Roles | Master: UOWC energy only. Slave: UOWC receive, THz transmit only, to a third-party receiver whose noise sets `hbar`. |
| Schedule | `1` = one event released in that slot, `0` = sleep slot. At most one event per slot (`EVENTS_PER_SLOT_MAX = 1`). Cyclic repetition of a short pattern stays. |
| PV | No gating: PV/MPPT charges in every slot. The schedule switches only the load and the TX. |
| Solver | Optimiser only. No separate fill policy. |
| Stop at `B_max` | The S2 arrival ceiling `b_{k-1} + a_k*delta_k <= B_max` is hard. In a slot that arrives full, `delta_k = 0`. |
| Modulation / payload | 16-QAM, 8 MiB = 67,108,864 bits, `Rs` = 1 Gsps, `t_tx = payload_bits/(Rs*4) = 16.777216 ms`. |
| PSRAM | 8 MB read = write = 10.5 ms each; the read runs inside the TX window. Notebook values (39.6 mW active, 237.6 uW standby, 25.2 uW deep power down) replace the README memory block. Default silent state: standby. |
| Camera | 0.30 W x 0.5 s = 0.15 J (README config value). |
| MPPT | Input rating 3.3 V x 100 mA stays the S5 cap (18.81 J per slot at 95%). Quiescent 0.96 uW / sleep 1.44 uW stay a **load**. The notebook's 330 mW x t_active (174 mJ) is **not** a load. Conversion efficiency 0.90-0.95 (`MPPT_EFF`, default 0.95) lives in `beta`. |
| `beta` | `beta_i = PV responsivity/array chain x MPPT_EFF x ETA_SC` and nothing else. |
| Photomixer | 20 mW radiated cap (antenna 50-100 ohm), 200 mW electrical while transmitting, `eta_tx = 0.1`. Radiated floor 1 mW. Two modes behind `PM_DC_MODE` (section 4.4). |
| THz | `THZ_PRESET`: "0.3THz" = 0.3 THz with 10 dBi Tx and 10 dBi Rx, "0.1THz" = 0.1 THz with 5 dBi each. Notebook thermal noise (290 K, NF 10 dB, bandwidth = `Rs`, N = 4.0039e-11 W), `ALPHA_THZ` = 12300 1/m, `Pt` = 20 mW, RRC 0.35, `BER_TARGET` 1e-3 are unchanged. |
| Distances | `D_THZ_M` = 1.0 mm. `D_UOWC_M` set by hand. No distance sweeps anywhere. |
| Refusal | Refused event is rescheduled to the next available slot (first later slot with no event of its own). `MAX_DELAY_SLOTS` = 5 (placeholder, configurable). Dropped events are reported. |
| Notebook | Sections 1-12 keep plot and table code, layout, colours, titles, figure sizes. |

---------------------------------------------------------------------------------------------

## 3. Settings to add (all in `slave_config.py`, pure data)

| Name | Default | Meaning / status |
|---|---|---|
| `THZ_PRESET` | `"0.3THz"` | selects `(FREQS_HZ key, Gtx dBi, Grx dBi)` from `GAIN_DBI_BY_FREQ = {"0.1 THz": (5, 5), "0.3 THz": (10, 10)}` |
| `LEGACY_GAINS` | `False` | `True`: 5/5 dBi at both frequencies (reproduces the saved notebook outputs) |
| `D_THZ_M`, `D_UOWC_M` | `1.0e-3`, set by hand | metres |
| `NOISE_MODE`, `T_NOISE_K`, `NOISE_FIGURE_DB`, `R_s`, `ALPHA_THZ`, `PT_W`, `BER_TARGET` | notebook values | keep the notebook names |
| `PAYLOAD_BYTES`, `MODULATION` | `8*1024*1024`, `"16-QAM"` | derives `t_tx` |
| `P_PM_RAD_MAX_W`, `P_PM_ELEC_W`, `P_PM_RAD_MIN_W` | 0.020, 0.200, 0.001 | `ETA_TX = P_PM_RAD_MAX_W / P_PM_ELEC_W` = 0.1 |
| `PM_DC_MODE` | `"proportional"` | or `"fixed_bias"` |
| `MPPT_EFF` | 0.95 | range 0.90-0.95 |
| `ETA_SC` | 0.95 | supercap storage efficiency, **DUMMY** |
| `ETA_LD`, `LASER_P_OPT_MAX_W` | `1.0`, `None` | UOWC laser efficiency / max optical power (M5 off when `None`), **DUMMY** |
| `PSRAM_SILENT_STATE` | `"standby"` | or `"dpd"` |
| `EVENTS_PER_SLOT_MAX`, `MAX_DELAY_SLOTS` | 1, 5 | |
| `MAXIMAL_SHARING` | `True` for `integrated_demo` | tie-break post-pass (section 4.7) |
| `SLAVE_SCHEDULE` | user's pattern | list of 0/1, repeated cyclically |

Component table (replace the hard-coded component list in `slave_load.py` by reading this; adding a component must be a row, not code):

| Component | Active W | Active time per event | Sleep W | Notes |
|---|---|---|---|---|
| camera | 0.300 | 0.5 s | 10e-6 | |
| MCU | 0.060 | `t_cam + t_write + t_tx` = 0.527277 s | 16e-6 | covers camera, write and the TX burst |
| PSRAM (key `flash` kept) | 0.0396 | `t_write + t_read` = 0.021 s | 237.6e-6 (standby) / 25.2e-6 (dpd) | read overlaps the TX window |
| MPPT | 0 (not a load) | 0 | 1.44e-6 | input rating is S5 |
| photomixer | 0.200 | `t_tx` | 1e-6 | active energy is the DWF variable (section 4.4) |

---------------------------------------------------------------------------------------------

## 4. Specifications

### 4.1 THz link (in `dwf2d/channels.py`)
```
lambda    = c / f
fspl_db   = 20*log10(4*pi*d*f/c)
abs_db    = 10*log10(e) * ALPHA_THZ * d
coupling  = fspl_db + abs_db - Gtx_dBi - Grx_dBi        [dB]
N         = k_B * T_NOISE_K * 10^(NF/10) * R_s           [W]  (= 4.0039e-11)
hbar_i    = ETA_TX / (10^(coupling/10) * N)              [1/W]  slave electrical burst power -> SNR
SNR_burst = hbar_i * P_pk,i ,   P_pk,i = x_i / t_tx      (x_i = slave TX burst energy, J)
```
* Near field: Friis is valid only for `d >= d_nf = lambda/(4*pi) * 10^((Gtx+Grx)/20)` (0.7958 mm for 0.3 THz / 10 dBi, 0.7549 mm for 0.1 THz / 5 dBi). Use `d_eff = max(d, d_nf)` and return a flag; print one warning line when it triggers.
* Keep the notebook functions' signatures and results; add the per-frequency gain lookup (`link_budget(freq, d, g_tx=None, g_rx=None, ...)` defaults to the preset gains). With `LEGACY_GAINS = True` all notebook tables must be reproduced to 1e-12.
* UOWC: `uowc_alpha(d, c, eta_ld, geom)` = `ETA_LD * exp(-c*d) * geom` (`geom` = 1 unless a beam model is added). `c` = 0.046 1/m (given). The existing `beer_lambert` stays.
* Move the modulation engine (`gray_pam`, `gray_qam`, `qam_ber_theory`, `pam_ber_theory`, `make_scheme`, `rrc_pulse`, `min_distance_decode`, `simulate_link`) out of the notebook verbatim, with `rng`, `PULSE`, `sps`, `DELAY` passed in or created by a small factory instead of module globals.

### 4.2 `beta` chain (in `dwf2d/physical.py`)
* `physical_stored_power(..., supercap_eff=ETA_SC)`: stored energy = `ETA_SC x` MPPT output (after the existing S5 clip). `eff_phys` becomes the full `R x MPPT_EFF x ETA_SC` secant gain, so `solve_with_physical_chain` calibrates `beta_i` to it with no change to the loop.
* Pass `mppt_efficiency=MPPT_EFF` through. `physical_chain.py` is not edited.
* Arriving optical power per slot is `alpha_i * delta_i / tau` with `alpha_i` from `uowc_alpha` (laser efficiency already inside).

### 4.3 Event-based slave load (in `slave_load.py`)
For slot `i` with `s_i` (the **effective** schedule, after refusals):
```
event slot : c_i = sum over components of [P_active * t_active + P_sleep * (tau - t_active)]   (photomixer active term excluded, see 4.4)
sleep slot : c_i = sum over components of P_sleep * tau
```
Worked values (standby, `tau` = 60 s): camera 150.000 mJ + MCU 31.637 mJ + PSRAM active 0.832 mJ + sleep remainders 15.944 mJ = **198.41 mJ** per event slot (non-TX); **15.962 mJ** per sleep slot (PSRAM dpd: 3.218 mJ). Any `t_active` > `tau`, negative duration, or per-slot array of the wrong length raises `ValueError` naming field and slot (existing behaviour of `load_timing()`).
* Durations are per **event**, not per slot. `t_tx` is its own field (derived from payload, `Rs`, bits per symbol). The README's "short burst" warning at `attach_slave_load` is replaced by this model (burst-aware rate, 4.5).
* Slave-state object: supercap level, pending-event queue, counters, indexed by slave id (single slave now). All arrays sized from the schedule length.
* `attach_slave_load` keeps its non-idempotence note, scales `hbar` by `ETA_TX` exactly once, and now takes `hbar` from 4.1.

### 4.4 Photomixer modes and TX bounds
Let `x_i = tau*Pbar_i` be the electrical TX burst energy of slot `i` (the existing solver variable `xs(i)`).
```
x_cap = P_PM_ELEC_W * t_tx              = 3.3554 mJ   (20 mW radiated)
x_min = (P_PM_RAD_MIN_W/ETA_TX) * t_tx  = 0.16777 mJ  (1 mW radiated)
P_pk  = x_i / t_tx ;  radiated = ETA_TX * P_pk (<= 20 mW)
```
* **`proportional` (default):** `x_i in {0} U [x_min, x_cap]` in a run-event slot, `x_i = 0` elsewhere. Add S6 `tx_cap_J[i] = x_cap` (run-event slots; `0` in sleep slots, `inf` = feature off). The lower bound is **not** put inside the moves; it is enforced by the refusal loop (4.6).
* **`fixed_bias`:** a run event costs `P_PM_ELEC_W * t_tx` = 3.3554 mJ of fixed energy, added to that slot's `c_i`; `x_i` is fixed at the cap and not a free variable (S6 cap 0 for the variable part; the rate credit per run event is the constant `T/2 ln(1 + hbar * P_PM_ELEC_W)` reported by the link report). An event the supercap cannot afford raises `InfeasibleSlaveBudget` at that slot; the refusal loop catches it (4.6). The solver then only decides `delta` timing.
* Optional S7 `tx_floor_J[i]` (e.g. a BER-margin minimum) uses the same plumbing as S6; default `None`.

### 4.5 Solver changes (`dwf2d/params.py`, `dwf2d/solver.py`, `dwf2d/verification.py`)
New `JointParams` fields, all default to "off = bit-identical":

| Field | Default | Meaning |
|---|---|---|
| `tx_cap_J` | `None` (inf) | S6: `xs(i) <= tx_cap_J[i]` |
| `tx_floor_J` | `None` | S7: `xs(i) >= tx_floor_J[i]` in run slots |
| `delta_cap_J` | `None` | M5: `d[k] <= delta_cap_J[k]` (laser) |
| `slave_rate_window` | `None` (= tau) | `T_i` for the slave rate |
| `theta1` | unchanged | set to `0` in the integrated scenario (master has no data) |

1. **S6 / S7 / M5** are added exactly as S5 was: bounds in `apply()`, in the explicit bounds of moves 1-5, in moves 7 and 9, in `polish()` (family D), in `check_kkt_joint` (report active slots; skip the `delta` stationarity test where a cap binds) and in `reference_solution_joint`.
2. **Burst-aware slave rate.** Objective term becomes `th2 * (T_i/2) * ln(1 + hbar_i * x_i / T_i)` with `T_i = slave_rate_window[i]` (default `tau`, which gives the README expression). Marginal value is `s_i = th2*hbar_i / (2*(1 + hbar_i*x_i/T_i))`, so the form is unchanged; only the `P` inside uses `T_i`. In `step()` generalise the balance condition to
   `th_u*h_u*(1 + h_w*x_w/T_w) = r*th_w*h_w*(1 + h_u*x_u/T_u)` with `x_u -> x_u - s*t`, `x_w -> x_w + r*t`: still linear in `t`. Master end: `T = tau`. Update `check_kkt_joint`, `reference_solution_joint` (cvxpy `log` term) and the objective/rate bookkeeping the same way.
3. **Master with no data (`theta1 = 0`).** `step()` returns 0 when both weights are 0 (unchanged); master-to-master moves become inert; master-to-slave moves push `delta` to the slave's marginal value. `xm` (the old "master data energy") is spill: relabel it **unused / spill** in `diagnose`, plots, CSV and `JointResult` docs; drop master throughput and the master nats from reports. Do not force `xm = 0` (with `F_{N-1} = 0` that would force the master to deliver its whole harvest). If `h` appears in a denominator with `theta1 = 0`, use a placeholder `h = 1` and say so in a comment.
4. **Terminal level.** The solver keeps `g[N] = 0` (legacy, unchanged). The maximal-sharing post-pass (4.7) is what lets the final supercap level end above `B_min`; the ledger check in `check_kkt_joint` is extended accordingly (4.7).
5. Cross-check every change against `reference_solution_joint` (cvxpy) before wiring it into the demo (phase 4 gate).

### 4.6 Refusal and rescheduling (outer loop in `dwf2d/physical.py`, queue in `slave_load.py`)
```
effective = copy(SLAVE_SCHEDULE expanded to N)         # 1 = event released at that slot
queue = []                                              # FIFO of (release_slot, age)
loop (max passes = (#events) * MAX_DELAY_SLOTS + 1):
    r = solve (joint_dwf or solve_with_physical_chain) with `effective`, S6 cap on, warm start from previous r
    refused = [i for run slots if 0 < x_i < x_min]       # fixed_bias: slots raising InfeasibleSlaveBudget
    if none: break
    for i in refused (ascending):
        effective[i] = 0                                  # nothing powers up: sleep load, Pbar = 0
        place event in the first slot j > i with effective[j] == 0 and j - release_slot <= MAX_DELAY_SLOTS
        if no such slot: mark dropped
        else: effective[j] = 1 ; record delay
```
* Slots count as available only if they carry no event of their own (so scheduled events keep their timing); at most `EVENTS_PER_SLOT_MAX` events per slot.
* Warm start: pass the previous `f`, `g`, `d` as the start point where bounds allow it (add an optional `x0`/start argument to `joint_dwf`; default behaviour unchanged). The first pass has no warm start.
* The result carries a per-event table: release slot, run slot, delay, `x`, outcome (`run`, `rescheduled`, `dropped`), and the number of passes. No event may disappear: `run + dropped = released`.
* Termination is guaranteed because each pass moves or drops at least one event and delays are bounded. If the pass cap is hit, report non-convergence; do not return a silently inconsistent schedule.
* It is a greedy rule that mirrors the hardware (wake, check the supercap, refuse). Document that it is not a combinatorial optimum; verify against brute force on small N (section 6).

### 4.7 Maximal-sharing tie-break (post-pass in `dwf2d/solver.py`)
Purpose: with master joules worth nothing, many `delta` trajectories are optimal. Pick the one where the master shares as early and as much as the supercap can take, so the supercap sits as full as possible and the master stops only at `B_max`. **Objective, `x_i`, events and `Pbar` do not change.**
```
given final x_i (all i), c_i, a_k, capA_k (S5), laser cap (M5), E_k, E_max:
for k = 0 .. N-1 (forward):
    headroom_k   = B_max - level_{k-1}                       # arrival ceiling S2
    delta_k      = max(0, min(headroom_k/a_k, capA_k/a_k, delta_cap_k, E_k + f_k))   # master energy available
    level_arr    = min(B_max, level_{k-1} + a_k*delta_k)
    level_k      = level_arr - c_k - x_k                       # must stay >= B_min (guaranteed: trajectory dominates the optimum)
    f_{k+1}      = E_k + f_k - delta_k - spill_k ;  spill_k = max(0, f_{k+1}_unclipped - (E_max - E_{k+1}))
```
* `a_k*delta_k` is then fixed by the ceiling, so the `beta` calibration fixed point is unaffected; run the post-pass after calibration converges, then recompute `physical_stored_power` once as a check.
* The ledger check changes from "slave ends at `B_min`" to "slave ends at `level_{N-1} >= B_min`", and `slave_ledger_error_J` is extended to include the final level. Report both the optimal-solution ledger and the post-pass ledger.
* Master energy: delivering early may use energy the master would otherwise have kept; the master's carry `f` and spill are recomputed in the same pass and must stay in `[0, E_max - E_{k+1}]`. If a slot cannot be filled because the master has no energy, that slot is stopped by `E_k + f_k`.
* Property to verify: every slot is stopped by exactly one of the ceiling `B_max`, S5, M5, or the master's energy; `delta_k = 0` whenever the arrival is already full.
* Switch `MAXIMAL_SHARING` (default `True` in `integrated_demo`; `False` leaves the raw optimiser output).

### 4.8 Link report (in `dwf_diagnostics.py` and `dwf2d/plotting/slave_panels.py`)
For every run event: `x`, `P_pk = x/t_tx`, radiated power `ETA_TX*P_pk`, `SNR_burst = hbar*P_pk`, theoretical BER (16-QAM `ber_func(SNR/4)`), pass/fail against `BER_TARGET`, nats credited. Also: events at the cap, events refused, rescheduled and dropped, supercap sizing (`W_req` vs `W`), DUMMY banner while any value is unmeasured, near-field clamp flag. Optionally call `simulate_link` for a constellation at the worst event's SNR. CSV `slave_energy_profile.csv` keeps its columns; add new columns at the end only.

### 4.9 Notebook (`Modulation_THz_slave_profile.ipynb`)
* Becomes a thin front end importing 4.1 (link, modulation) and 4.3 (slave load). Sections 1-12 keep code for tables and plots, layout, colours, titles and figure sizes.
* Config cell: the global `G_TX_DBI` / `G_RX_DBI` become the per-frequency lookup; both THz columns of the existing figures use their own preset gains, so the 0.3 THz numbers change versus the saved outputs. `LEGACY_GAINS = True` must reproduce the saved outputs.
* §13: replace the hard-coded profile by a call into `slave_load` with the new component table; the table and the two plots keep their look, the MPPT row shows quiescent/sleep only, the 330 mW input rating is shown as a note.
* Keep notebook outputs reproducible with the fixed seed (`SEED = 42`).

---------------------------------------------------------------------------------------------

## 5. File map (no new files)

| File | Work |
|---|---|
| `slave_config.py` | section 3 settings, component table, presets |
| `dwf2d/channels.py` | 4.1: link budget, `hbar`, near-field clamp, `uowc_alpha`, modulation engine |
| `slave_load.py` | 4.3, queue and slave-state, photomixer component |
| `dwf2d/physical.py` | 4.2 `supercap_eff`, 4.6 outer loop |
| `dwf2d/params.py`, `dwf2d/solver.py` | 4.5 fields, bounds, rate window, `theta1 = 0` handling, warm start, 4.7 post-pass |
| `dwf2d/verification.py` | extended `check_kkt_joint` and `reference_solution_joint` |
| `dwf_diagnostics.py`, `dwf2d/plotting/slave_panels.py` | 4.8 |
| `dwf2d/demos.py`, `build_2D_dwf.py` | `integrated_demo()` and re-exports; `python build_2D_dwf.py integrated_demo` writes to `out_integrated_demo/` |
| `Modulation_THz_slave_profile.ipynb` | 4.9 |
| `tests/` | new cases (section 6), baseline captured only for `integrated_demo` |

---------------------------------------------------------------------------------------------

## 6. Phases and gates

### Phase 0: preconditions
- [ ] Locate the tree; list every function/class/file this document names and confirm or correct it (write corrections as a short list in your report).
- [ ] `python tests/regression.py check` prints IDENTICAL. If it does not, stop and report (the project docs note a mismatch between README section 7 values and an earlier diagnostic: `B_min` 20 vs 200 J, `sum(c)` 15.6 vs 2.7 J).
- [ ] Time `joint_dwf` on the legacy demo for later comparison.

### Phase 1: move the link and modulation code (no behaviour change)
- [ ] Add 4.1 functions and the modulation engine to `dwf2d/channels.py` (verbatim moves).
- [ ] Notebook imports them; per-frequency gains and `LEGACY_GAINS`.
- [ ] Gate: with `LEGACY_GAINS = True`, `df_budget`, `df_energy`, `df_summary`, `df_antenna`, `df_gain_range`, BER theory and seeded `simulate_link` (SEED 42) equal the saved notebook values to 1e-12; regression IDENTICAL.

### Phase 2: slave config and load
- [ ] Section 3 settings; component table; event-based `c_i` (4.3); photomixer component; PSRAM notebook values; camera 0.30 W; MPPT sleep only.
- [ ] Gate: event slot 198.41 mJ and sleep slot 15.962 mJ (standby) to 0.01 mJ; `dpd` sleep slot 3.218 mJ; the user's 30-slot pattern gives 7 events and `sum(c)` = 1.7560 J per 30 slots (7 x 198.41 mJ + 23 x 15.962 mJ, non-TX), 1.7795 J with the 7 photomixer bursts at the cap (7 x 3.355 mJ); regression IDENTICAL; swapping any duration or adding a dummy component changes only numbers.

### Phase 3: `beta` chain and UOWC
- [ ] `supercap_eff`, `MPPT_EFF`, `uowc_alpha`, laser cap data.
- [ ] Gate: with `ETA_SC = 1`, `MPPT_EFF = 0.95` the calibrated `beta` equals the README value (0.3798 at 532 nm); with `ETA_SC = 0.95` it scales by exactly 0.95; solver and physical batteries agree to 1e-12 J; regression IDENTICAL.

### Phase 4: solver features
- [ ] S6, S7, M5; `slave_rate_window`; `theta1 = 0` handling and relabelling.
- [ ] Extend `reference_solution_joint` first, then the solver.
- [ ] Gate: cvxpy comparison, 25 random cases per family (default, tight window, per-slot `beta`, finite `E_max`, S6 binding, S6 + S5 binding, burst-aware rate with random `T_i`, `theta1 = 0`): report exact / below / above counts; **0 above optimum**; every case below the optimum is listed with its gap. Defaults off => regression IDENTICAL.

### Phase 5: refusal, retry and tie-break
- [ ] Outer loop and warm start (4.6); post-pass (4.7).
- [ ] Gate: brute force over all event placements within `MAX_DELAY_SLOTS` on N <= 10 matches the greedy loop's event table or reports the gap; stress cases pass (tiny supercap, zero-harvest slots, all-ON schedule, single ON slot, `fixed_bias` mode); `run + dropped = released`; delay <= `MAX_DELAY_SLOTS`; post-pass leaves objective and every `x_i` unchanged to 1e-9 and each slot is stopped by one of ceiling / S5 / M5 / master energy; `delta_k = 0` in slots that arrive full; supercap within `[B_min, B_max]`.

### Phase 6: link report and `integrated_demo`
- [ ] 4.8 outputs; `integrated_demo()` with the user's pattern, `THZ_PRESET = "0.3THz"`, `D_THZ_M = 1.0 mm`.
- [ ] Gate: sanity numbers below; every run event has `x` in `[0.168, 3.355]` mJ; plots/CSV written to `out_integrated_demo/`; capture a new baseline for this demo only; legacy regression IDENTICAL.

### Phase 7: verification and docs
- [ ] Scalability run (section 7); README sections 2 (file table), 3-4 (new constraints and moves), 6 (placeholders), 8 (verification results), 11 (open items 10, 14, 16 updated) and a short "stage 6" section; save the final report next to the README.

---------------------------------------------------------------------------------------------

## 7. Verification matrix (summary)

| Area | Check |
|---|---|
| Regression | `tests/regression.py check` after every phase |
| Notebook | 1e-12 equivalence under `LEGACY_GAINS` |
| Optimality | cvxpy parity as in phase 4; failure rates reported, not single successes |
| Invariants | objective non-decreasing in `x_cap`, `B_max`, `a`, `ETA_SC`; non-increasing in `d_thz`, `c`; ledgers close to ~1e-11 J; `Pbar = 0` in sleep slots; `x_i <= x_cap`; supercap in `[B_min, B_max]` |
| Stop rule | `delta_k = 0` when the arrival is full; no `delta` pushes the level above `B_max` |
| Tie-break | objective and `x` unchanged; pointwise maximal supercap trajectory (check with a cvxpy LP: maximise `sum(b_k)` with the objective fixed to the optimum) |
| Refusal / retry | event table consistent; greedy vs brute force on N <= 10 |
| Scalability | schedule repeated to double N gives matching per-period results; runtime for N = 60, 600, 6000 reported for the optimiser and for the O(N) parts (load, queue, link report, post-pass) separately; do not change the solver speed in this task, only report |
| Presets | `THZ_PRESET` switch changes only `hbar` (and printed labels); both presets run end to end |

## 8. Sanity numbers for `integrated_demo` (check, do not tune)

* `t_tx` = 16.777216 ms; `x_cap` = 3.3554 mJ; `x_min` = 0.16777 mJ; slot-average cap 55.9 uW; burst-to-average ratio 3576.
* N = 4.0039e-11 W (-103.98 dBW). `hbar` at 1.0 mm: 7.2e3 for "0.3THz" (10 dBi each), 6.48e3 for "0.1THz" (5 dBi each). SNR at 20 mW radiated: 31.6 dB and 31.1 dB. 16-QAM needs Es/N0 16.6 dB for BER 1e-3 and 20.4 dB for 1e-6.
* At the cap and 1.0 mm / "0.3THz": `SNR_burst` = 7.2e3 x 0.2 W = 1440, `ln(1+1440)` = 7.273, rate credit per event `T/2 ln(1+SNR)` = 0.0610 nats before `theta2`.
* Event slot non-TX 198.41 mJ, sleep slot 15.962 mJ, photomixer 3.355 mJ electrical (0.336 mJ radiated, 1.7% of an event). The user's 30-slot pattern has 7 events.
* Supercap window per event about 0.2 J: 0.022 F over 2.2-5.2 V gives 0.244 J, 0.047 F gives 0.522 J. If `W` < `c + x_min` no event can ever run: expect refusals until dropped.
* The README's 0.5 mm link distance is inside the Friis-invalid region for both presets and must not be used.

## 9. Known risks (report if they show up)

* The optimiser takes about 10 s for N = 60 (pure-Python loops) and the retry loop multiplies it; this task only measures and reports (section 7). Decision on speeding up (vectorised sweeps vs rolling-horizon blocks) is deferred.
* With `theta1 = 0` and the cap binding, `delta` is non-unique; the post-pass fixes the representative. If `joint_dwf` stalls near the cap (README section 9 corner-stalling), report the gap against cvxpy instead of adding ad-hoc moves.
* The photomixer is not a linear PA; `proportional` mode is a modelling convenience, `fixed_bias` is the sensitivity check. Report both when both are run.
* Greedy event rescheduling is not a global optimum.
* `ETA_SC`, `ETA_LD`, `LASER_P_OPT_MAX_W`, `MAX_DELAY_SLOTS` and `D_UOWC_M` are DUMMY/placeholder values until the user supplies real ones.
* Supercap leakage, images held across slots and the UWDrone receiver's real noise figure and bandwidth are not modelled (notebook defaults: NF 10 dB, 1 GHz).
* The excerpted README may differ from the source in small ways; trust the code.

## 10. Reporting format at the end

One page: what changed per file, gate results per phase (numbers), the cvxpy failure-rate table, the per-event table for the demo, runtimes, and the list of DUMMY values still in use.
