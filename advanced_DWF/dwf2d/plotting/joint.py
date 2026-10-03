"""Per-panel plots of the joint solution (one PNG per function).
"""
from __future__ import annotations

import numpy as np

from ..params import JointParams, JointResult
from .style import (
    BITS_PER_NAT, C_LINK, C_MASTER, C_SLAVE, GRID, INK, INK2, _bound_masks, _finish, _legend, _mark,
    _new_axes, _new_axis, _rates_bits, _shade_silent,
)


def plot_link_power(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Power over the underwater energy link, three stages of the same flow.

        sent by the master      delta/tau                          (solid)
        received after channel  alpha*delta/tau                    (dashed)  Beer-Lambert loss
        stored after recharge   beta*alpha*delta/tau = a*delta/tau (dotted)  recharging loss
    """
    t = np.arange(jp.master.N)
    fig, ax = _new_axis()
    ax.plot(t, r.sent_power, color=C_LINK, lw=2, ls="-", label="sent by master, δ/τ")
    ax.plot(t, r.arrived_power, color=C_LINK, lw=2, ls="--",
            label="arrived at slave photodiode, αδ/τ  (after water loss α)")
    ax.plot(t, r.stored_power, color=C_LINK, lw=2, ls=":",
            label="stored in slave battery, βαδ/τ  (after recharge loss β)")
    ax.set_ylabel("average optical / electrical power over the slot [W]")
    ax.set_ylim(bottom=0)
    _legend(ax)
    _finish(fig, ax, path, "Optical energy link: power sent, received and stored per slot", title, jp)


def plot_transmit_power(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Data transmit power against the water-level staircase, one panel per node.

    Water-filling form (module docstring section 4):  P_k = [nu_k - 1/h_k]^+ , so the plotted level
    nu_k = P_k + 1/h_k is the "water surface". Where taps are open, nu is constant over a run of
    slots (a staircase); it can only step where a constraint binds, marked on the curve:
        master battery empty (F_k=0)  -> level may only rise afterwards
        master battery full  (F_k=cap)-> level may only fall afterwards
        slave battery at floor / full on arrival (S1 / S2)  likewise
    Slots with zero power are cases where the level sits below the floor 1/h (non-negativity binds),
    and silent slave slots are forced to zero by the schedule (S4), not by water.
    """
    m, t = jp.master, np.arange(jp.master.N)
    nu_m = r.P + 1.0 / m.h
    nu_s = r.Pbar + 1.0 / jp.h_bar
    m_empty, m_full, s_empty, s_full = _bound_masks(jp, r)

    fig, (axm, axs) = _new_axes(2)

    axm.step(t, nu_m, where="mid", color=INK2, lw=1.4, ls="--",
             label="water level ν_k = P_k + 1/h_k (staircase)")
    axm.plot(t, r.P, color=C_MASTER, lw=2, label="master data transmit power P_k")
    axm.plot(t, 1.0 / m.h, color=GRID, lw=1.2, ls=":", label="floor 1/h_k (level needed before any power flows)")
    _mark(axm, t, nu_m, r.P <= 1e-9, "P_k = 0: level below floor (non-negativity binds)", "x", INK)
    _mark(axm, t, nu_m, m_empty, "master battery empty, F_k = 0 (causality binds)", "v", C_MASTER)
    _mark(axm, t, nu_m, m_full, "master battery full, F_k = cap (overflow binds)", "^", C_LINK)
    axm.set_ylabel("master power [W]")
    axm.set_title("Master: data power P_k vs water level", loc="left", color=INK2, fontsize=10)
    axm.set_ylim(bottom=0)
    _legend(axm, "upper left")

    _shade_silent(axs, jp)
    axs.step(t, nu_s, where="mid", color=INK2, lw=1.4, ls="--",
             label="water level ν̄_k = P̄_k + 1/h̄_k (staircase)")
    axs.plot(t, r.Pbar, color=C_SLAVE, lw=2, label="slave data transmit power P̄_k")
    axs.plot(t, 1.0 / jp.h_bar, color=GRID, lw=1.2, ls=":", label="floor 1/h̄_k")
    _mark(axs, t, nu_s, (r.Pbar <= 1e-9) & jp.slave_schedule,
          "P̄_k = 0 in a transmit slot: level below floor", "x", INK)
    _mark(axs, t, nu_s, s_empty & jp.slave_schedule, "slave battery at floor B_min after a transmit slot (causality binds)", "v", C_SLAVE)
    _mark(axs, t, nu_s, s_full & jp.slave_schedule, "slave battery full on arrival, B_max (overflow binds)", "^", C_LINK)
    # Slave powers can span many orders of magnitude (the floor 1/h̄ is huge next to P̄), so use a
    # log axis; symlog keeps zero-power slots on the plot, linear below the smallest positive P̄.
    pos = r.Pbar[r.Pbar > 1e-12]
    linthresh = float(pos.min()) if pos.size else 1e-12
    axs.set_yscale("symlog", linthresh=linthresh)
    axs.set_ylabel("slave power [W] (log scale)")
    axs.set_title("Slave: data power P̄_k vs water level (log scale)", loc="left", color=INK2, fontsize=10)
    axs.set_ylim(bottom=0)
    _legend(axs, "upper left")
    _finish(fig, (axm, axs), path, "Transmit power against the water-filling level", title, jp)


def plot_slave_battery(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Slave battery level after each slot b_n = B_min + Gt_n (solid) and level right after
    each arrival b_{n-1} + a_n delta_n (dashed), with the limits B_min and B_max as dotted lines.
    Slots at the floor (causality binding) and at the ceiling on arrival (overflow binding) are marked.
    """
    t = np.arange(jp.master.N)
    _, _, s_empty, s_full = _bound_masks(jp, r)
    fig, ax = _new_axis()
    ax.plot(t, r.b, color=C_SLAVE, lw=2, label="battery level at end of slot k, b_k")
    ax.plot(t, r.b_peak, color=C_SLAVE, lw=1.2, ls="--",
            label="battery level just after energy arrives in slot k, b_(k-1) + a_k δ_k")
    ax.axhline(jp.B_max, color=INK2, lw=1, ls=":")          # ceiling B_max (S2)
    ax.axhline(jp.B_min, color=INK2, lw=1, ls=":")          # floor   B_min (S1)
    ax.annotate(f"ceiling B_max = {jp.B_max:g} J", (t[-1], jp.B_max), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.annotate(f"floor B_min = {jp.B_min:g} J", (t[-1], jp.B_min), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    _mark(ax, t, r.b, s_empty, "empty: b_k = B_min (causality binding; last slot excluded)", "v", INK)
    _mark(ax, t, r.b_peak, s_full, "full: arrival reaches B_max (overflow binding)", "^", C_LINK)
    ax.set_ylim(0, jp.B_max * 1.08)
    ax.set_ylabel("slave battery energy [J]")
    _legend(ax, "center right")
    _finish(fig, ax, path, "Slave battery level against its floor and ceiling", title, jp)


def plot_cumulative_energy(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Cumulative energy that has ARRIVED vs cumulative energy CONSUMED, per node.

    Causality (M1 / S1) says consumption can never exceed arrival:
        master : sum_{i<=k}(tau P_i + delta_i)  <=  sum_{i<=k} E_i
        slave  : sum_{i<=k} tau Pbar_i          <=  (b0 - B_min) + sum_{i<=k} a_i delta_i
    The gap between the two curves is exactly the energy held in the battery (F_k, or b_k - B_min);
    the curves touch where the battery is empty, i.e. at the water-level breakpoints.
    """
    m, t = jp.master, np.arange(jp.master.N)
    tau = m.tau
    arr_m = np.cumsum(m.E)
    use_m = np.cumsum(tau * r.P + r.delta)
    arr_s = jp.g_init + np.cumsum(jp.a * r.delta)
    use_s = np.cumsum(tau * r.Pbar)
    m_empty, _, s_empty, _ = _bound_masks(jp, r)

    fig, (axm, axs) = _new_axes(2)
    axm.plot(t, arr_m, color=INK2, lw=2, label="cumulative harvested, Σ E_i")
    axm.plot(t, use_m, color=C_MASTER, lw=2, label="cumulative consumed, Σ (τ P_i + δ_i)  [own data + shared]")
    axm.plot(t, np.cumsum(tau * r.P), color=C_MASTER, lw=1.2, ls="--", label="of which own data only, Σ τ P_i")
    axm.fill_between(t, use_m, arr_m, color=C_MASTER, alpha=0.12, lw=0,
                     label="gap = energy stored in master battery, F_k")
    _mark(axm, t, use_m, m_empty, "curves touch: battery empty (causality binds)", "o", INK)
    axm.set_ylabel("cumulative energy [J]")
    axm.set_title("Master: harvested vs consumed", loc="left", color=INK2, fontsize=10)
    axm.set_ylim(bottom=0)
    _legend(axm, "upper left")

    axs.plot(t, arr_s, color=INK2, lw=2, label="cumulative arrived in battery, (b0 − B_min) + Σ a_i δ_i")
    axs.plot(t, use_s, color=C_SLAVE, lw=2, label="cumulative consumed by data, Σ τ P̄_i")
    axs.fill_between(t, use_s, arr_s, color=C_SLAVE, alpha=0.12, lw=0,
                     label="gap = energy held above the floor, b_k − B_min")
    _mark(axs, t, use_s, s_empty, "curves touch: battery at floor (causality binds)", "o", INK, size=22)
    axs.set_ylabel("cumulative energy [J]")
    axs.set_title("Slave: arrived vs consumed", loc="left", color=INK2, fontsize=10)
    axs.set_ylim(bottom=0)
    _legend(axs, "upper left")
    _finish(fig, (axm, axs), path, "Cumulative energy arrived vs consumed (causality check)", title, jp)


def plot_throughput_per_slot(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Throughput delivered in each slot [bits]:
        master  (tau/2) ln(1 + h_k    P_k)    / ln 2
        slave   (tau/2) ln(1 + hbar_k Pbar_k) / ln 2
    These are the per-slot terms that sum to r.rate_master / r.rate_slave (module docstring,
    section 1 objective, unweighted by theta1/theta2), converted from nats to bits.
    """
    t = np.arange(jp.master.N)
    rate_m, rate_s = _rates_bits(jp, r)
    fig, ax = _new_axis()
    _shade_silent(ax, jp)
    ax.plot(t, rate_m, color=C_MASTER, lw=2, label="master → receiver, (τ/2) log₂(1 + h_k P_k)")
    ax.plot(t, rate_s, color=C_SLAVE, lw=2, label="slave → master, (τ/2) log₂(1 + h̄_k P̄_k)")
    ax.set_ylabel("data delivered in the slot [bits]")
    ax.set_ylim(bottom=0)
    _legend(ax)
    _finish(fig, ax, path, "Throughput per slot", title, jp)


def plot_throughput_cumulative(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Running total (cumulative sum) of the per-slot throughput, i.e. data delivered so far [bits]."""
    t = np.arange(jp.master.N)
    rate_m, rate_s = _rates_bits(jp, r)
    fig, ax = _new_axis()
    ax.plot(t, np.cumsum(rate_m), color=C_MASTER, lw=2,
            label=f"master, total after last slot = {r.rate_master * BITS_PER_NAT:.1f} bits")
    ax.plot(t, np.cumsum(rate_s), color=C_SLAVE, lw=2,
            label=f"slave, total after last slot = {r.rate_slave * BITS_PER_NAT:.1f} bits")
    ax.set_ylabel("cumulative data delivered up to slot k [bits]")
    ax.set_ylim(bottom=0)
    _legend(ax, "upper left")
    _finish(fig, ax, path, "Cumulative throughput", title, jp)


def plot_harvested_energy(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Energy entering each node per slot, with the conversion efficiency visible.

    Top    : master harvest E_i. This is already ELECTRICAL energy out of the solar array (the model
             takes E_i as given), so there is no optical input to show on this side.
    Bottom : slave harvest through the optical energy link. OPTICAL energy that reaches the
             photodiode (alpha*delta) vs ELECTRICAL energy actually stored (beta*alpha*delta); the
             ratio beta_k is drawn on the right-hand axis.
    Kept off the sharing plot because E is normally 1-3 orders of magnitude larger than the shared energy.
    """
    m, t = jp.master, np.arange(jp.master.N)
    tau = m.tau
    opt_in = tau * r.arrived_power                  # alpha*delta   [J] optical energy at the receiver
    elec_out = tau * r.stored_power                 # beta*alpha*delta [J] electrical energy in the battery

    fig, (axm, axs) = _new_axes(2)
    axm.plot(t, m.E, color=INK2, lw=2, label="electrical energy harvested by master, E_k")
    axm.set_ylabel("energy per slot [J]")
    axm.set_title("Master: solar harvest (electrical)", loc="left", color=INK2, fontsize=10)
    axm.set_ylim(bottom=0)
    _legend(axm, "upper left")

    axs.plot(t, opt_in, color=C_LINK, lw=2, ls="--", label="optical energy received, α_k δ_k  (in)")
    axs.plot(t, elec_out, color=C_SLAVE, lw=2, label="electrical energy stored, β_k α_k δ_k  (out)")
    axs.set_ylabel("energy per slot [J]")
    axs.set_ylim(bottom=0)
    axs.set_title("Slave: optical in → electrical out through the photodiode + charger", loc="left",
                  color=INK2, fontsize=10)
    ax2 = axs.twinx()
    ax2.plot(t, jp.beta, color=INK2, lw=1.2, ls=":", label="conversion efficiency β_k (right axis)")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("efficiency β_k  [–]", color=INK2)
    ax2.tick_params(colors=INK2)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    h1, l1 = axs.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axs.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", bbox_to_anchor=(1.08, 1.0),
               fontsize=8.5, labelcolor=INK2, borderaxespad=0)
    _finish(fig, (axm, axs), path, "Energy harvested per slot: master solar array and slave optical link", title, jp)


def plot_sharing_energy(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Signed energy flows of each node per slot [J]: IN is positive, OUT is negative.

        master  in : harvest E_k
                out: own data tau*P_k, and energy sent to the slave delta_k
        slave   in : energy that lands in the battery a_k*delta_k
                out: data tau*Pbar_k
                (grey hatched = the (1-a_k) delta_k lost in the water + charger before it lands)
    Whatever is in minus out each slot is what the battery gains or loses (see the battery plots).
    """
    m, t = jp.master, np.arange(jp.master.N)
    tau = m.tau
    lost = (1.0 - jp.a) * r.delta

    fig, (axm, axs) = _new_axes(2)
    axm.bar(t, m.E, color=INK2, width=0.8, label="IN: harvested, E_k")
    axm.bar(t, -tau * r.P, color=C_MASTER, width=0.8, label="OUT: own data, τ P_k")
    axm.bar(t, -r.delta, bottom=-tau * r.P, color=C_LINK, width=0.8,
            label="OUT: sent to slave, δ_k")
    axm.axhline(0, color=INK2, lw=0.8)
    axm.set_ylabel("energy per slot [J]\n(+ in, − out)")
    axm.set_title("Master energy flows", loc="left", color=INK2, fontsize=10)
    _legend(axm, "upper left")

    axs.bar(t, jp.a * r.delta, color=C_LINK, width=0.8, label="IN: received and stored, a_k δ_k")
    axs.bar(t, lost, bottom=jp.a * r.delta, color=GRID, hatch="//", edgecolor=INK2, linewidth=0,
            width=0.8, label="lost on the way, (1 − a_k) δ_k  (not received)")
    axs.bar(t, -tau * r.Pbar, color=C_SLAVE, width=0.8, label="OUT: data sent to master, τ P̄_k")
    axs.axhline(0, color=INK2, lw=0.8)
    axs.set_ylabel("energy per slot [J]\n(+ in, − out)")
    axs.set_title("Slave energy flows", loc="left", color=INK2, fontsize=10)
    _legend(axs, "upper left")
    _finish(fig, (axm, axs), path, "Energy transferred in and out of each node per slot", title, jp)
