# Stage 6 report: THz modulation notebook + event-based slave in the Master + Slave DWF

Spec: `implementation.md` (2026-10-08). Everything new is off by default; `legacy_demo` and `physical_demo` are bit-for-bit
unchanged. No new script files; `build_dwf.py` and `physical_chain.py` untouched.

## Decisions taken with the user during the work
1. **Baseline.** At the start the golden master did not match the committed code (116 differences). It had been captured from
   uncommitted state: slot 1 silent in the schedule, `beta` 0.4, `B_min` 20, `B_max` 2000; commit `300e999` changed the
   inputs. Re-captured from HEAD once, with your approval, and checked twice for determinism.
2. **Gate.** `check` accepts additive-only changes: new fields left at `None`, new names, trailing parameters with
   defaults.
3. **Demo supercap.** 0.047 F.
4. **Refusal rule.** An event is refused when `x < x_min`, `x = 0` included.
5. **Battery gate.** The 1e-12 J battery gate applies at the integrated supercap scale. On the legacy problem the residual is
   5.3e-12 J on a 200-1500 J battery, i.e. relative round-off, and it was the same before this work.

## What changed per file
| File | Change |
|---|---|
| `dwf2d/channels.py` | link budget, `thz_hbar` (near-field clamp), `uowc_alpha`, modulation engine moved verbatim from the notebook (`make_link_simulator` factory) |
| `slave_config.py` | STAGE 6 block: schedule, presets, link, photomixer (`ETA_TX_PM`), `PM_DC_MODE`, `MPPT_EFF`, `ETA_SC`, supercap, `EVENT_DURATIONS`, `EVENT_COMPONENTS`, `DUMMY_VALUES` |
| `slave_load.py` | `event_profile`, `tx_bounds_J`, `thz_link`, `attach_event_load` (`model="event"`), `make_event_load`, `reschedule`, `SlaveState` |
| `dwf2d/params.py` | `tx_cap_J` (S6), `tx_floor_J` (S7), `delta_cap_J` (M5), `slave_rate_window` |
| `dwf2d/solver.py` | S6/S7/M5 in moves 1-5, 7, 9 and `apply()`; burst-rate `step()`; warm start `x0`; `.slots` on the infeasibility error; `fullest_trajectory`, `maximal_sharing` |
| `dwf2d/physical.py` | `supercap_eff`, `solver_kw`; `solve_with_refusals` / `RefusalResult` |
| `dwf2d/verification.py` | reference solution and KKT check with S6/S7/M5, burst rate and `theta1 = 0`; gate harnesses |
| `dwf_diagnostics.py`, `dwf2d/plotting/*` | `diagnose_no_data`, `link_table` / `link_report`, `plot_event_bursts`, CSV columns appended, `theta1 = 0` labels |
| `dwf2d/demos.py`, `build_2D_dwf.py` | `integrated_demo` (`out_integrated_demo/`), re-exports |
| `tests/` | additive allowance, single-case capture, `parity` / `events` / `stage6` modes, fast pytest checks, `integrated_demo` baseline |
| notebook | imports `dwf2d.channels`; per-frequency gains plus `LEGACY_GAINS`; section 13 driven by `slave_load` |
| `dwf2d/plotting/thz_link.py` | the notebook's 13 THz-link figures (notebook used as reference only), drawn by `integrated_demo` into `out_integrated_demo/thz_link/` with the DWF operating point marked |

## Gates per phase
| Phase | Result |
|---|---|
| 0 | IDENTICAL after the re-capture; `joint_dwf` 2.0 s (legacy) / 2.8 s (with load) for N = 60 |
| 1 | `LEGACY_GAINS = True`: 91 notebook arrays equal at tolerance **0**; regression IDENTICAL |
| 2 | 198.412 / 15.962 / 3.218 mJ; 7 events; sum c 1.7560 J (1.7795 J with bursts); validation errors name field and slot |
| 3 | `beta` 0.379762 (`ETA_SC = 1`), ratio 0.950000000 at `ETA_SC = 0.95`; integrated battery agreement 1e-16 J |
| 4 | cvxpy parity: **0 above** (table below) |
| 5 | 120/120 converge; brute force 97/100 match; post-pass invariants exact; LP agreement 2e-15 J |
| 6 | sanity numbers met (below); all `x` in [0.168, 3.355] mJ; `integrated_demo` baseline captured; legacy IDENTICAL |
| 7 | presets differ only in `h_bar`; ledgers 2.4e-16 J; **1 invariant failure** (solver stall, see open issues) |

## cvxpy failure-rate table (25 random cases per family)
| family | exact | below | above |
|---|---|---|---|
| default | 2 | 23 | 0 |
| tight_window | 20 | 5 | 0 |
| per_slot_beta | 5 | 20 | 0 |
| finite_E_max | 0 | 25 | 0 |
| s6_binding | 0 | 25 | 0 |
| s6_s5_binding | 2 | 23 | 0 |
| rate_window | 3 | 22 | 0 |
| theta1_zero | 10 | 15 | 0 |
| integrated_like | **25** | 0 | 0 |

- Below-optimum gaps range from 1e-6 to 1e-1 relative.
- The **unmodified HEAD solver gives bit-identical objectives** on these cases, so the gaps are the pre-existing stall.
- It occurs in a regime where the master gives everything and the optimum needs a master-carry delta shift, which no move
  provides.

## `integrated_demo` per-event table (0.3THz, 1 mm, 0.047 F)
| Item | Value |
|---|---|
| Events released / run | 14 / 14, all in their release slot (1, 14-16, 21-23, 31, 44-46, 51-53) |
| Refused / rescheduled / dropped | 0 / 0 / 0, in 1 pass |
| Burst energy `x` per event | 3.3554 mJ, at the cap |
| Peak power `P_pk` | 200 mW electrical, 20 mW radiated |
| SNR | 31.58 dB |
| BER (16-QAM) | 5.2e-65 (pass) |
| Rate per event | 0.0610 nats |
| Objective | 2.562422 nats (cvxpy identical) |

- After the post-pass the supercap is refilled to `B_max` in all 60 slots and ends 0.506 J above the floor.
- The window `W` = 0.522 J covers `W_req` = 0.393 J.
- No near-field clamp at 1 mm.
- Preset 0.1THz: objective 2.525328. `fixed_bias` mode: 14 events, 0.8541 nats credited.

## Runtimes
| N | `joint_dwf` (integrated) | event load | queue | post-pass | link table |
|---|---|---|---|---|---|
| 60 | 0.28 s | 0.45 ms | 0.06 ms | 0.25 ms | 0.37 ms |
| 600 | 216 s | 0.39 ms | 0.08 ms | 2.1 ms | 0.7 ms |
| 6000 | not run (about 80 h, O(N^3) from polish family D) | 1.3 ms | 0.6 ms | 22 ms | 9.6 ms |

The objective per 60-slot period is identical for N = 60 and N = 600 (2.562422054).

## DUMMY / placeholder values still in use
- `ETA_SC` 0.95, `ETA_LD` 1.0, `LASER_P_OPT_MAX_W` None (M5 off), `MAX_DELAY_SLOTS` 5, `D_UOWC_M` 1 mm.
- Master `h = 1` (inert while `theta1 = 0`), `b0 = B_min`, calibration wavelength 532 nm.
- Receiver noise figure and bandwidth are the notebook defaults.
- The stage-5 slot model's durations stay DUMMY.

## Deviations from the spec, and open issues
- **Pass cap.** The refusal loop's pass cap is `#events (MAX_DELAY_SLOTS + 1) + 1`; the spec's formula is one refusal per
  event too small.
- **Infeasible-solve refusal.** It uses the fullest-supercap forward pass to find the slot to blame (the hardware rule).
  If the base sleep load itself is unaffordable, events are reported as `unresolved`, not "run".
- **Name clash.** The legacy `slave_config.ETA_TX` (1.0) stays; the photomixer efficiency is `ETA_TX_PM` (0.1).
- **Notebook titles.** When the two frequencies have different gains, the titles and the section-8 "G_tx + G_rx" line show
  one value per frequency.
- **Scarce-energy stall.** With scarce master energy, `joint_dwf` can stall with an event at `x = 0`: 3 of 32 scarce cases
  are 33% below cvxpy. The refusal loop then refuses an affordable event. README open items 17-19 cover this, the missing
  move and the O(N^3) runtime. Each needs your decision; nothing was tuned or added.
- **README drift.** Section 7 of the README described older runs; current numbers were added on top.
