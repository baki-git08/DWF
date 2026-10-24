"""The two demo problems (legacy_demo, physical_demo) run by `python build_2D_dwf.py`.
"""
from __future__ import annotations

import numpy as np

from build_dwf import Harvested_energy_winter, Harvested_energy_winter_half, MasterParams
from dwf_diagnostics import diagnose

from .channels import beer_lambert, thz_gain, uowc_alpha
from .params import JointParams
from .physical import solve_with_physical_chain
from .verification import check_kkt_joint
from .plotting.joint import (
    plot_cumulative_energy, plot_dwf_allocation, plot_harvested_energy, plot_link_power, plot_master_battery, plot_sharing_energy,
    plot_slave_battery, plot_throughput_cumulative, plot_throughput_per_slot, plot_transmit_power,
)
from .plotting.physical import plot_loss_histogram, plot_physical_slave_battery, plot_physical_stored_power
from .plotting.slave_panels import (
    plot_event_bursts, plot_slave_battery_window, plot_slave_cumulative_energy, plot_slave_energy_breakdown,
    plot_slave_load_per_slot, write_refusal_history_csv, write_slave_energy_csv,
)
from .plotting.thz_link import plot_received_power_vs_distance, setup_from_slave_config, write_thz_link_plots


# ----------------------------------------------------------------------------
# Demos.  DEMO selects one (or: python build_2D_dwf.py physical_demo)
#   legacy_demo   : no slave load attached; reproduces the README numbers (objective 1902.8194, 1410 iters).
#   physical_demo : dummy slave timing (slave_config.py), MPPT cap S5 on, window from size_storage.
#                   The result is at DUMMY durations and placeholder h0 / theta2; it differs from legacy_demo
#                   by orders of magnitude in slave data energy (the real electronics cannot absorb the
#                   legacy stage-4 flows).  That is expected, not a solver bug: inputs are NOT retuned.
#   integrated_demo : stage 6 (slave_config.py STAGE 6 block): master shares over UOWC only (theta1 = 0), event-based
#                   slave with THz bursts, refusal / rescheduling loop and maximal-sharing post-pass; master battery
#                   capped at MASTER_E_MAX_J with a reserve floor MASTER_FLOOR_J it keeps for itself.
# ----------------------------------------------------------------------------



def _demo_problem() -> JointParams:
    """The master + slave problem shared by both demos (winter harvest, tau = 60 s, THz link at 0.30 THz)."""
    SLAVE_SCHEDULE = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]               # 1 = slave transmits, 0 = silent; repeats cyclically over the slots
    LINK_DISTANCE_M = 0.0005             # master<->slave separation [m], shared by both links
    FREQ_THZ = 0.30                      # THz operating frequency chosen for the data link
    OPTICAL_ATTEN_450NM = 0.046         # Beer-Lambert c [1/m], 450 nm (blue) light in water

    # THz data link: symmetric, so the same gain is used for master->x and slave->master
    h_thz = thz_gain(FREQ_THZ, LINK_DISTANCE_M)          # h0 = thz_gain default (10.0): PLACEHOLDER, uncalibrated

    # 1) master: winter harvest array, unlimited master battery, 60 s slots, THz data link
    master = MasterParams(E=Harvested_energy_winter, E_max=np.inf, tau=60.0, h=h_thz)
    # 2) slave: alpha = Beer-Lambert transmittance of the 450 nm optical energy link,
    #    beta = 0.8, battery window [200, 5000] J, return data link = same THz channel
    return JointParams(master=master, alpha=beer_lambert(OPTICAL_ATTEN_450NM, LINK_DISTANCE_M),
                       beta=0.9, B_min=200.0, B_max=1500.0, h_bar=h_thz,
                       slave_schedule=SLAVE_SCHEDULE)


def _run_demo(jp0: JointParams, outdir: str = ".", load: dict | None = None):
    """Calibrate beta against the physical chain, solve, print the report and write the plots."""
    import os
    if outdir != ".":
        os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)
    master = jp0.master

    # Calibrate beta_i against the real photodiode+MPPT chain (Option B, Instructions.md), then
    # solve with it. joint_dwf() itself is unmodified; only jp.beta becomes per-slot and real.
    jp, r, ph, n_outer = solve_with_physical_chain(jp0, wavelength_nm=532, mppt_efficiency=0.95)
    diagnose(jp, r) if load is None else diagnose(jp, r, load=load)
    k = check_kkt_joint(jp, r)              # feasibility + delta-condition report
    print(f"iters={r.iters} converged={r.converged}")
    print(f"objective={r.objective:.4f}  master={r.rate_master:.4f}  slave={r.rate_slave:.4f}")
    print(f"P  mean={r.P.mean():.2f} W   Pbar mean={r.Pbar.mean():.2f} W   "
          f"shared mean={r.sent_power.mean():.2f} W")
    for key, v in k.items():
        print(f"  {key}: {v}")

    # Calibration diagnostics: the solver's battery should now equal the physical one.
    print(f"calibration outer iters = {n_outer}, beta range = "
          f"[{jp.beta.min():.4f}, {jp.beta.max():.4f}]")
    print(f"max |solver battery - physical battery| = {np.max(np.abs(r.b - ph.b_phys)):.3e} J")
    print(f"stored energy  solver={r.stored_power.sum()*master.tau:.2f} J   "
          f"physical={ph.stored_energy_phys.sum():.2f} J")
    print(f"physical battery floor_violations={ph.floor_violations}  "
          f"ceiling_violations={ph.ceiling_violations}  overflow_energy={ph.overflow_energy:.3e} J  "
          f"overflow_mppt={ph.overflow_mppt_energy:.3e} J")

    caption = "Master + slave DWF (THz data link, 450 nm optical energy link)"
    plot_link_power(jp, r, out("plot_link_power.png"), caption)
    plot_transmit_power(jp, r, out("plot_transmit_power.png"), caption)
    plot_slave_battery(jp, r, out("plot_slave_battery.png"), caption)
    plot_throughput_per_slot(jp, r, out("plot_throughput_per_slot.png"), caption)
    plot_throughput_cumulative(jp, r, out("plot_throughput_cumulative.png"), caption)
    plot_harvested_energy(jp, r, out("plot_harvested_energy.png"), caption)
    plot_sharing_energy(jp, r, out("plot_sharing_energy.png"), caption)
    plot_cumulative_energy(jp, r, out("plot_cumulative_energy.png"), caption)
    plot_dwf_allocation(jp, r, out("plot_dwf_allocation.png"), caption)
    plot_physical_stored_power(jp, r, ph, out("plot_physical_stored_power.png"), caption)
    plot_physical_slave_battery(jp, r, ph, out("plot_physical_slave_battery.png"), caption)
    plot_loss_histogram(jp, r, ph, out("plot_loss_histogram.png"), caption)
    if load is not None:
        cap2 = caption + "  [slave electronics attached; DUMMY timing]" if load.get("timing_is_dummy") else caption
        plot_slave_load_per_slot(jp, r, load, out("plot_slave_load_per_slot.png"), cap2)
        plot_slave_energy_breakdown(jp, r, load, out("plot_slave_energy_breakdown.png"), cap2)
        plot_slave_cumulative_energy(jp, r, load, out("plot_slave_cumulative_energy.png"), cap2)
        plot_slave_battery_window(jp, r, load, out("plot_slave_battery_window.png"), cap2)
        write_slave_energy_csv(out("slave_energy_profile.csv"), jp, r, load)
        print(f"wrote the 4 slave-load PNGs and slave_energy_profile.csv to {outdir}")
    return jp, r, ph


def legacy_demo():
    """No slave load: reproduces the README stage-4 numbers."""
    return _run_demo(_demo_problem())


def _integrated_problem(preset: str | None = None, n_periods: int = 1):
    """Stage-6 problem: first-half winter harvest (slots 0-29; 0 afterwards), master with NO data link (theta1 = 0), battery capacity MASTER_E_MAX_J and
    reserve floor MASTER_FLOOR_J, UOWC energy link, event-based slave (slave_config.py STAGE 6 block).  Returns (jp0, build, link): build(effective) attaches the event load.
    n_periods repeats the 60-slot harvest (and the cyclic schedule) to N = 60 * n_periods (scalability runs)."""
    import dataclasses
    import slave_config as cfg
    import slave_load
    # h = 1.0 is a PLACEHOLDER: with theta1 = 0 the master has no data, so h only appears in step() with weight 0.
    # Master battery: capacity MASTER_E_MAX_J (M2) and reserve floor MASTER_FLOOR_J (M6) from slave_config.py.
    # Half-harvest scenario: Harvested_energy_winter_half (use Harvested_energy_winter for the full-day harvest).
    master = MasterParams(E=np.tile(Harvested_energy_winter_half, int(n_periods)), E_max=float(cfg.MASTER_E_MAX_J),
                          tau=60.0, h=1.0)
    jp0 = JointParams(master=master, alpha=uowc_alpha(cfg.D_UOWC_M, cfg.UOWC_C_PER_M, cfg.ETA_LD, cfg.UOWC_GEOM), beta=0.8,
                      theta1=0.0, slave_schedule=cfg.SLAVE_SCHEDULE, master_floor_J=cfg.MASTER_FLOOR_J)
    link = slave_load.thz_link(preset)

    def build(effective):
        return slave_load.attach_event_load(dataclasses.replace(jp0, slave_schedule=np.asarray(effective, int)), link=link)
    return jp0, build, link


def integrated_demo(outdir: str = "out_integrated_demo", preset: str | None = None):
    """Stage 6: the master only harvests and shares over UOWC; the slave sends one THz burst per event.

    refusal loop (physical.solve_with_refusals) around the calibrated solve (beta = R x MPPT_EFF x ETA_SC),
    then the maximal-sharing post-pass (MAXIMAL_SHARING), the THz link report and the plots / CSV in outdir.
    Inputs are NOT tuned: first-half winter harvest, THZ_PRESET, D_THZ_M = 1 mm, 0.047 F supercap, DUMMY values flagged."""
    import os
    import slave_config as cfg
    import slave_load
    from dwf_diagnostics import diagnose_no_data, link_report, link_table
    from .physical import physical_stored_power, solve_with_refusals
    from .solver import maximal_sharing
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)
    banner = slave_load.dummy_banner()
    if banner:
        print(banner)
    jp0, build, link = _integrated_problem(preset)
    tb = slave_load.tx_bounds_J()
    calib = dict(wavelength_nm=cfg.PV_CAL_WAVELENGTH_NM, mppt_efficiency=cfg.MPPT_EFF, supercap_eff=cfg.ETA_SC,
                 fill_idle_beta=True)     # the post-pass may send in slots the optimiser left idle
    #   tol: joint_dwf stops at tol * mean(E) (kJ per slot); the bursts are mJ, so the default 1e-10 would stop at
    #   ~4e-7 J.  1e-15 puts the stop near 4e-12 J (the solver itself is unchanged).
    rr = solve_with_refusals(build, jp0.slave_schedule, tb["x_min"], cfg.MAX_DELAY_SLOTS, mode=cfg.PM_DC_MODE,
                             calib_kw=calib, solver_kw=dict(tol=1e-15))
    print(f"integrated_demo: {jp0.master.N} slots, {int(np.sum(jp0.slave_schedule))} events released, "
          f"preset {link['preset']}, refusal loop {rr.passes} pass(es), converged {rr.converged}")
    write_refusal_history_csv(out("refusal_history.csv"), rr.history)   # saved even when the loop did not converge
    if not rr.converged or rr.r is None:
        print("integrated_demo: the refusal loop did not converge; no consistent schedule to report")
        for h in rr.history:
            print(f"  {h}")
        return rr.jp, rr.r, rr.ph
    jp, r, ph = rr.jp, rr.r, rr.ph
    k_opt = check_kkt_joint(jp, r)
    stop = None
    r_rep, ph_rep = r, ph
    if cfg.MAXIMAL_SHARING:
        r2, info = maximal_sharing(jp, r)
        if info["applied"]:
            stop = info["stop"]
            r_rep = r2
            ph_rep = physical_stored_power(jp, r2, cfg.PV_CAL_WAVELENGTH_NM, cfg.MPPT_EFF, supercap_eff=cfg.ETA_SC)
        else:
            print(f"maximal-sharing post-pass NOT applied: {info['reason']}")
    load = slave_load.make_event_load(jp)
    rows = link_table(jp, r_rep, rr.events, link, tb["t_tx"], tb["x_min"], tb["x_cap"], cfg.BER_TARGET, cfg.MODULATION,
                      cfg.PM_DC_MODE)
    diagnose_no_data(jp, r_rep, stop=stop, profile=load)
    link_report(jp, rr, link, rows, load, load["sizing"], cfg.BER_TARGET)
    k_rep = check_kkt_joint(jp, r_rep)
    print(f"optimiser : objective {r.objective:.9f} nats, iters {r.iters}, converged {r.converged}, start {r.start}")
    print(f"            slave ledger error {k_opt['slave_ledger_error_J']:.3e} J, end level above floor "
          f"{r.b[-1] - jp.B_min:.4g} J, bursts at cap {k_opt.get('tx_cap_active_slots', 0)}")
    if stop is not None:
        print(f"post-pass : objective {r_rep.objective:.9f} nats (change {abs(r_rep.objective - r.objective):.1e}), "
              f"max |x change| {np.max(np.abs(jp.master.tau * (r_rep.Pbar - r.Pbar))):.1e} J")
        print(f"            slave ledger error {k_rep['slave_ledger_error_J']:.3e} J, end level above floor "
              f"{r_rep.b[-1] - jp.B_min:.4g} J, supercap in [{r_rep.b.min():.6g}, {r_rep.b_peak.max():.6g}] J "
              f"(window [{jp.B_min:.6g}, {jp.B_max:.6g}])")
    print(f"calibration: {rr.n_outer} outer iteration(s), beta in [{jp.beta.min():.6f}, {jp.beta.max():.6f}] "
          f"(R x MPPT_EFF {cfg.MPPT_EFF:g} x ETA_SC {cfg.ETA_SC:g}, {cfg.PV_CAL_WAVELENGTH_NM} nm)")
    print(f"            max |solver battery - physical battery| = {np.max(np.abs(r.b - ph.b_phys)):.3e} J (optimiser)"
          + (f", {np.max(np.abs(r_rep.b - ph_rep.b_phys)):.3e} J (post-pass)" if stop is not None else ""))
    for key in ("tx_cap_active_slots", "max_tx_cap_violation_J", "unused_spill_J", "min_slave_level_margin_J",
                "final_master_carry", "master_floor_binding_slots", "min_master_floor_margin_J", "master_full_slots"):
        if key in k_rep:
            print(f"  {key}: {k_rep[key]}")

    cap = (f"Integrated DWF: UOWC energy sharing ({cfg.D_UOWC_M:g} m), "
           f"slave THz bursts ({link['preset']}, {link['d_m'] * 1e3:g} mm)")
    if banner:
        cap += "  [DUMMY values in use]"
    plot_link_power(jp, r_rep, out("plot_link_power.png"), cap)
    plot_transmit_power(jp, r_rep, out("plot_transmit_power.png"), cap)
    plot_slave_battery(jp, r_rep, out("plot_slave_battery.png"), cap)
    plot_master_battery(jp, r_rep, out("plot_master_battery.png"), cap)
    plot_throughput_per_slot(jp, r_rep, out("plot_throughput_per_slot.png"), cap)
    plot_throughput_cumulative(jp, r_rep, out("plot_throughput_cumulative.png"), cap)
    plot_harvested_energy(jp, r_rep, out("plot_harvested_energy.png"), cap)
    plot_sharing_energy(jp, r_rep, out("plot_sharing_energy.png"), cap)
    plot_cumulative_energy(jp, r_rep, out("plot_cumulative_energy.png"), cap)
    plot_dwf_allocation(jp, r_rep, out("plot_dwf_allocation.png"), cap)
    plot_physical_stored_power(jp, r_rep, ph_rep, out("plot_physical_stored_power.png"), cap)
    plot_physical_slave_battery(jp, r_rep, ph_rep, out("plot_physical_slave_battery.png"), cap)
    plot_slave_load_per_slot(jp, r_rep, load, out("plot_slave_load_per_slot.png"), cap)
    plot_slave_energy_breakdown(jp, r_rep, load, out("plot_slave_energy_breakdown.png"), cap)
    plot_slave_cumulative_energy(jp, r_rep, load, out("plot_slave_cumulative_energy.png"), cap)
    plot_slave_battery_window(jp, r_rep, load, out("plot_slave_battery_window.png"), cap)
    plot_event_bursts(jp, r_rep, rows, load, out("plot_event_bursts.png"), cap, cfg.BER_TARGET)
    plot_received_power_vs_distance(jp, r_rep, link, out("plot_received_power_vs_distance.png"), cap)
    write_slave_energy_csv(out("slave_energy_profile.csv"), jp, r_rep, load, link_rows=rows, stop=stop)
    print(f"wrote 17 PNGs, slave_energy_profile.csv and refusal_history.csv to {outdir}")
    # The THz-link figures of the notebook (sections 7-12) at the slave_config link values, with this demo's operating
    # point marked; their Monte-Carlo parts use their own seeded rng (no effect on the numbers above).
    thz = write_thz_link_plots(setup_from_slave_config(preset=preset), out("thz_link"))
    print(f"wrote {len(thz)} THz-link PNGs to {out('thz_link')}")
    return jp, r_rep, ph_rep


def integrated_demo_uowc_sweep(outdir: str = "out_uowc_sweep", c_values=None, preset: str | None = None):
    """integrated_demo once per UOWC attenuation c in c_values (default slave_config.UOWC_C_PER_M_SWEEP).

    Each run writes its full output to outdir/c_<value>/ (0.315 -> c_0p315); outdir/summary.csv has one row per c.
    slave_config.UOWC_C_PER_M is set for the run and restored afterwards."""
    import csv
    import os
    import slave_config as cfg
    c_values = cfg.UOWC_C_PER_M_SWEEP if c_values is None else c_values
    os.makedirs(outdir, exist_ok=True)
    c_orig = cfg.UOWC_C_PER_M
    rows = []
    try:
        for c in c_values:
            cfg.UOWC_C_PER_M = float(c)
            sub = os.path.join(outdir, f"c_{float(c):g}".replace(".", "p"))
            print(f"\n===== UOWC sweep: c = {float(c):g} 1/m -> {sub} =====")
            jp, r, _ = integrated_demo(sub, preset)
            ok = r is not None          # False: the refusal loop found no feasible schedule (see refusal_history.csv)
            alpha = uowc_alpha(cfg.D_UOWC_M, cfg.UOWC_C_PER_M, cfg.ETA_LD, cfg.UOWC_GEOM)
            rows.append(dict(c_per_m=float(c), alpha=float(np.ravel(alpha)[0]), converged=ok,
                             objective_nats=r.objective if ok else "", shared_J=float(r.delta.sum()) if ok else "",
                             slave_tx_J=float(jp.master.tau * r.Pbar.sum()) if ok else "",
                             events_scheduled=int(np.sum(jp.slave_schedule)) if ok else "", outdir=sub))
    finally:
        cfg.UOWC_C_PER_M = c_orig
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} sweep runs and summary.csv to {outdir}")
    return rows


def physical_demo(outdir: str = "out_physical_demo"):
    """Slave electronics at DUMMY timing (slave_config.py), S5 on, window from size_storage()."""
    import slave_load
    jp0 = _demo_problem()
    jp_l = slave_load.attach_slave_load(jp0, size_storage=True)
    load = slave_load.make_load(jp_l)
    print(f"physical_demo: window W = {jp_l.W:.4g} J (size_storage), b0 - B_min = {jp_l.g_init:.4g} J, "
          f"sum c = {jp_l.overhead.sum():.4g} J, MPPT cap = {jp_l.capA.min():.4g} J, eta_tx = {slave_load.cfg.ETA_TX:g}")
    jp, r, ph = _run_demo(jp_l, outdir, load)
    jl, rl, _, _ = solve_with_physical_chain(jp0, wavelength_nm=532, mppt_efficiency=0.95)
    dur = "DUMMY durations" if slave_load.cfg.TIMING_IS_DUMMY else "measured durations"
    print(f"physical_demo vs legacy_demo (same master; h0 and theta2 = {jp0.theta2:g} are placeholders; {dur}):")
    print(f"  slave data energy {jp.master.tau * r.Pbar.sum():.4g} J vs {jl.master.tau * rl.Pbar.sum():.4g} J ; "
          f"master -> slave sent {r.delta.sum():.4g} J vs {rl.delta.sum():.4g} J ; objective {r.objective:.1f} vs {rl.objective:.1f}")
    return jp, r, ph
