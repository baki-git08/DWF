"""Slave-load panels and CSV (need the dict from slave_load.make_load(jp)).
"""
from __future__ import annotations

import numpy as np

from ..params import JointParams, JointResult
from .style import C_LINK, C_MASTER, C_SLAVE, GRID, INK, INK2, _finish, _legend, _new_axis, _shade_silent


# ----------------------------------------------------------------------------
# Slave load panels (need the dict from slave_load.make_load(jp); the solver module itself never
# imports the component model).  TX energy is the solver's slave data energy tau*Pbar_i.
# ----------------------------------------------------------------------------
C_MCU, C_MEM, C_MPPT = "#8a5cd6", "#c9a227", "#d6457a"
LOAD_COLORS = {"camera": C_MASTER, "mcu": C_MCU, "flash": C_MEM, "mppt": C_MPPT, "tx": C_SLAVE}
LOAD_LABELS = {"camera": "camera", "mcu": "MCU", "flash": "external memory", "mppt": "MPPT", "tx": "THz TX data energy, tau*Pbar_k (solver)"}


def slave_load_components(jp: JointParams, r: JointResult, load: dict) -> dict:
    """Per-slot energies [J]: the four non-TX components from the profile, tx = tau*Pbar_i from the solver,
    c = their non-TX sum and load = c + tx."""
    comp = {k: np.asarray(load[k], float) for k in ("camera", "mcu", "flash", "mppt")}
    comp["tx"] = jp.master.tau * r.Pbar
    comp["c"] = np.asarray(load["c"], float)
    comp["load"] = comp["c"] + comp["tx"]
    return comp


def plot_slave_load_per_slot(jp: JointParams, r: JointResult, load: dict, path: str, title: str = ""):
    """Stacked per-slot slave energy: camera, MCU, memory, MPPT (profile) + TX data energy tau*Pbar (solver).
    Silent slots are shaded; their electronics drain is tiny next to an ON slot, so it is hard to see."""
    comp = slave_load_components(jp, r, load)
    t = np.arange(jp.master.N)
    fig, ax = _new_axis()
    _shade_silent(ax, jp)
    bottom = np.zeros(jp.master.N)
    for k in ("camera", "mcu", "flash", "mppt", "tx"):
        ax.bar(t, comp[k], bottom=bottom, color=LOAD_COLORS[k], width=0.8, label=LOAD_LABELS[k], zorder=2)
        bottom += comp[k]
    ax.set_ylabel("slave energy per slot [J]")
    _legend(ax)
    _finish(fig, ax, path, "Slave energy per slot by component", title, jp)


def plot_slave_energy_breakdown(jp: JointParams, r: JointResult, load: dict, path: str, title: str = ""):
    """Horizon energy of the slave by component with percentages (TX = solver data energy)."""
    comp = slave_load_components(jp, r, load)
    ks = ["camera", "mcu", "flash", "mppt", "tx"]
    tot = {k: float(comp[k].sum()) for k in ks}
    T = sum(tot.values()) or 1.0
    fig, ax = _new_axis(figsize=(10.5, 4.2))
    y = np.arange(len(ks))
    ax.barh(y, [tot[k] for k in ks], color=[LOAD_COLORS[k] for k in ks], height=0.6, zorder=2)
    for yi, k in zip(y, ks):
        ax.annotate(f"  {tot[k]:.4g} J ({100 * tot[k] / T:.1f}%)", (tot[k], yi), va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(["camera", "MCU", "external memory", "MPPT", "THz TX data (solver)"], color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3 * max(tot.values()))
    ax.set_xlabel("energy over all slots [J]")
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    _finish(fig, ax, path, "Slave energy over the horizon by component", title, None)
    # (not a per-slot plot: _finish leaves the x label set above)


def plot_slave_cumulative_energy(jp: JointParams, r: JointResult, load: dict, path: str, title: str = ""):
    """Cumulative energy consumed by the slave (electronics c + TX data) against what was stored from the
    energy link plus the initial excess.  The gap is the energy held above the floor, b_k - B_min."""
    comp = slave_load_components(jp, r, load)
    t = np.arange(jp.master.N)
    consumed = np.cumsum(comp["load"])
    stored = jp.g_init + np.cumsum(jp.a * r.delta)
    fig, ax = _new_axis()
    _shade_silent(ax, jp)
    ax.plot(t, stored, color=INK2, lw=2, label="cumulative stored from the link + initial excess, (b0 - B_min) + sum a_i delta_i")
    ax.plot(t, consumed, color=C_SLAVE, lw=2, label="cumulative consumed, sum (c_i + tau*Pbar_i)")
    ax.plot(t, np.cumsum(comp["c"]), color=C_SLAVE, lw=1.2, ls="--", label="of which electronics only, sum c_i")
    ax.fill_between(t, consumed, stored, color=C_SLAVE, alpha=0.12, lw=0, label="gap = energy held above the floor")
    ax.set_ylabel("cumulative energy [J]")
    ax.set_ylim(bottom=0)
    _legend(ax)
    _finish(fig, ax, path, "Slave cumulative energy: consumed vs stored", title, jp)


def plot_slave_battery_window(jp: JointParams, r: JointResult, load: dict, path: str, title: str = ""):
    """Usable slave energy above the floor, b - B_min, after each slot and right after each arrival, against
    the window W = B_max - B_min and the profile's sizing lines (W_req, design window)."""
    sz = load["sizing"]
    t = np.arange(jp.master.N)
    fig, ax = _new_axis()
    _shade_silent(ax, jp)
    ax.step(t, r.b - jp.B_min, where="mid", color=C_SLAVE, lw=2, label="end of slot k, b_k - B_min")
    ax.step(t, r.b_peak - jp.B_min, where="mid", color=C_SLAVE, lw=1.2, ls="--", alpha=0.8,
            label="right after arrival, b_(k-1) + a_k delta_k - B_min")
    ax.axhline(jp.W, color=INK, lw=1.2, ls=":", label=f"window W = B_max - B_min = {jp.W:.4g} J")
    ax.axhline(sz["W_req"], color=C_LINK, lw=1.2, ls="--", label=f"profile W_req = {sz['W_req']:.4g} J")
    ax.axhline(sz["W_design"], color=C_MCU, lw=1.2, ls="--", label=f"profile W_design = {sz['W_design']:.4g} J")
    ax.set_ylabel("usable slave energy above the floor [J]")
    ax.set_ylim(bottom=0)
    _legend(ax)
    _finish(fig, ax, path, "Slave storage window", title, jp)


def write_slave_energy_csv(path: str, jp: JointParams, r: JointResult, load: dict):
    """slave_energy_profile.csv, same columns as slave_load.py writes: slot,on,camera_J,mcu_J,flash_J,mppt_J,
    tx_J,load_J,stored_J,delta_sent_J,battery_end_J   (tx_J = tau*Pbar, load_J = c + tx_J, stored_J = a*delta,
    delta_sent_J = delta, battery_end_J = b - B_min)."""
    import csv
    comp = slave_load_components(jp, r, load)
    stored = jp.a * r.delta
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "on", "camera_J", "mcu_J", "flash_J", "mppt_J", "tx_J", "load_J",
                    "stored_J", "delta_sent_J", "battery_end_J"])
        for i in range(jp.master.N):
            w.writerow([i, int(jp.slave_schedule[i]), *[f"{comp[k][i]:.6g}" for k in ("camera", "mcu", "flash", "mppt", "tx")],
                        f"{comp['load'][i]:.6g}", f"{stored[i]:.6g}", f"{r.delta[i]:.6g}", f"{r.b[i] - jp.B_min:.6g}"])
