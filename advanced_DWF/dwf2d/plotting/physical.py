"""Plots comparing the solver with the physical receive chain, and the loss breakdown.
"""
from __future__ import annotations

import numpy as np

from ..params import JointParams, JointResult
from ..physical import PhysicalStorageResult
from .style import AXIS_FONTSIZE, LEGEND_FONTSIZE, C_LINK, C_MASTER, C_SLAVE, GRID, INK, INK2, _finish, _legend, _new_axis


def plot_physical_stored_power(jp: JointParams, r: JointResult, ph: PhysicalStorageResult,
                                path: str, title: str = ""):
    """Solver assumption vs physical reality for the power that ends up in the slave battery.

        arrived optical power    alpha*delta/tau                (dashed)  reaches the photodiode
        solver-assumed stored    beta*alpha*delta/tau            (dotted)  joint_dwf()'s linear a_i model
        physical MPPT output     chain.mppt_output_power         (solid)   photodiode + MPPT, physical_chain.py
    """
    t = np.arange(jp.master.N)
    fig, ax = _new_axis()
    ax.plot(t, r.arrived_power, color=C_LINK, lw=2, ls="--", label="optical in, αδ/τ")
    ax.plot(t, r.stored_power, color=C_LINK, lw=2, ls=":", label="solver model, βαδ/τ")
    ax.plot(t, ph.stored_power_phys, color=C_SLAVE, lw=2, label="physical PD + MPPT")
    ax.set_ylabel("average power over the slot [W]", fontsize=AXIS_FONTSIZE)
    ax.set_ylim(bottom=0)
    _legend(ax)
    _finish(fig, ax, path, "Stored power: solver assumption vs physical receive chain", title, jp)


def plot_physical_slave_battery(jp: JointParams, r: JointResult, ph: PhysicalStorageResult,
                                 path: str, title: str = ""):
    """Slave battery level per the solver (linear a_i model) vs re-simulated with the physical
    receive chain's actual arrivals, same B_min/B_max limits as plot_slave_battery.
    """
    t = np.arange(jp.master.N)
    fig, ax = _new_axis()
    ax.plot(t, r.b, color=C_SLAVE, lw=2, label="solver model")
    ax.plot(t, ph.b_phys, color=C_MASTER, lw=2, ls="--", label="physical PD + MPPT")
    ax.axhline(jp.B_max, color=INK2, lw=1, ls=":")
    ax.axhline(jp.B_min, color=INK2, lw=1, ls=":")
    ax.annotate(f"ceiling B_max = {jp.B_max:g} J", (t[-1], jp.B_max), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=LEGEND_FONTSIZE, color=INK2)
    ax.annotate(f"floor B_min = {jp.B_min:g} J", (t[-1], jp.B_min), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=LEGEND_FONTSIZE, color=INK2)
    ax.set_ylabel("slave battery energy at end of slot [J]", fontsize=AXIS_FONTSIZE)
    _legend(ax, "center right")
    _finish(fig, ax, path, "Slave battery: solver model vs physical receive chain", title, jp)


def energy_loss_breakdown(jp: JointParams, r: JointResult, ph: PhysicalStorageResult):
    """Total energy [J] of every stage of the master -> slave energy link, over all slots.

    Chain (each loss is the difference of two consecutive stages):
        sent by master          sum delta
        - water attenuation     sum (1 - alpha) delta           -> arrived at photodiode
        - photodiode conversion sum tau (P_arrived - P_max_el)  -> max electrical power
        - MPPT / charger        sum tau (P_max_el  - P_mppt)    -> stored in battery
        - battery overflow      ph.overflow_energy (a real battery rejects it; ~0 once calibrated)
    Returns (losses, stages): ordered dicts {label: Joules}.
    """
    tau = jp.master.tau
    sent = float(r.delta.sum())
    arrived = float(tau * r.arrived_power.sum())
    electrical = float(tau * ph.chain.max_electrical_power.sum())
    stored = float(ph.stored_energy_phys.sum())
    losses = {
        "water attenuation\n(Beer-Lambert, 1-α)": sent - arrived,
        "photodiode conversion\n(optical → electrical)": arrived - electrical,
        "MPPT / charger\n(1 - η_mppt)": electrical - stored,
        "battery overflow\n(above B_max)": float(ph.overflow_energy),
    }
    stages = {"sent by master": sent, "arrived at photodiode": arrived,
              "max electrical power": electrical, "stored in battery": stored}
    return losses, stages


def plot_loss_histogram(jp: JointParams, r: JointResult, ph: PhysicalStorageResult,
                        path: str, title: str = ""):
    """One bar per loss element of the energy link (total over all slots), plus the delivered energy.

    Bars are in Joules; the label above each bar gives its share of the energy the master sent.
    Loss bars are coloured by the physical stage; the last bar (green) is the energy that is
    actually stored, i.e. what is NOT lost.
    """
    losses, stages = energy_loss_breakdown(jp, r, ph)
    sent = stages["sent by master"]
    names = list(losses) + ["delivered to\nslave battery"]
    vals = list(losses.values()) + [stages["stored in battery"]]
    cols = [C_MASTER, C_SLAVE, INK2, GRID, C_LINK]

    fig, ax = _new_axis(figsize=(10.5, 5.2))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=cols, width=0.62, edgecolor=INK2, linewidth=0.6)
    top = max(vals) if max(vals) > 0 else 1.0
    for b, v in zip(bars, vals):
        pct = 100.0 * v / sent if sent > 0 else 0.0
        ax.annotate(f"{v:.3g} J\n{pct:.3g} % of sent", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=LEGEND_FONTSIZE, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=LEGEND_FONTSIZE, color=INK2)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("energy over all slots [J]", fontsize=AXIS_FONTSIZE)
    ax.set_xlabel(" ")                                   # not a per-slot plot: suppress the slot-index label
    ax.text(0.99, 0.97, f"energy sent by master: {sent:.4g} J", transform=ax.transAxes,
            ha="right", va="top", fontsize=LEGEND_FONTSIZE, color=INK2)
    _finish(fig, ax, path, "Energy lost at each element of the master → slave link", title, None)
