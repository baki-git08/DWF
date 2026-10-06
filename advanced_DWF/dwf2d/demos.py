"""The two demo problems (legacy_demo, physical_demo) run by `python build_2D_dwf.py`.
"""
from __future__ import annotations

import numpy as np

from build_dwf import Harvested_energy_winter, MasterParams
from dwf_diagnostics import diagnose

from .channels import beer_lambert, thz_gain
from .params import JointParams
from .physical import solve_with_physical_chain
from .verification import check_kkt_joint
from .plotting.joint import (
    plot_cumulative_energy, plot_harvested_energy, plot_link_power, plot_sharing_energy, plot_slave_battery,
    plot_throughput_cumulative, plot_throughput_per_slot, plot_transmit_power,
)
from .plotting.physical import plot_loss_histogram, plot_physical_slave_battery, plot_physical_stored_power
from .plotting.slave_panels import (
    plot_slave_battery_window, plot_slave_cumulative_energy, plot_slave_energy_breakdown,
    plot_slave_load_per_slot, write_slave_energy_csv,
)


# ----------------------------------------------------------------------------
# Demos.  DEMO selects one (or: python build_2D_dwf.py physical_demo)
#   legacy_demo   : no slave load attached; reproduces the README numbers (objective 1902.8194, 1410 iters).
#   physical_demo : dummy slave timing (slave_config.py), MPPT cap S5 on, window from size_storage.
#                   The result is at DUMMY durations and placeholder h0 / theta2; it differs from legacy_demo
#                   by orders of magnitude in slave data energy (the real electronics cannot absorb the
#                   legacy stage-4 flows).  That is expected, not a solver bug: inputs are NOT retuned.
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
