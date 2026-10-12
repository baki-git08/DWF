"""Shared plot styling, palette and helpers used by every dwf2d.plotting module.
"""
from __future__ import annotations

import numpy as np

from ..params import JointParams, JointResult


# ----------------------------------------------------------------------------
# Plots: one panel per figure (each function saves exactly one PNG)
# ----------------------------------------------------------------------------
# Categorical slots 1-3 of the reference palette (light surface), fixed order.
C_MASTER, C_SLAVE, C_LINK = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


BITS_PER_NAT = 1.0 / np.log(2.0)    # the objective is in nats; plots report bits
EDGE_TOL = 1e-3                     # [J] a battery/carry within this of a bound counts as "at the bound"
#   (joint_dwf converges to ~1e-6 J, so slots sitting ON a bound land well inside this tolerance)


def _style(ax):
    """Shared, low-key axis styling."""
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def _new_axis(figsize=(12.5, 5.0)):
    """Create a single-panel figure with the shared, low-key styling used by every plot here."""
    import matplotlib
    matplotlib.use("Agg")                                   # draw to a file, no window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax)
    return fig, ax


def _new_axes(nrows=2, figsize=(12.5, 3.7)):
    """Create `nrows` stacked panels sharing the slot axis (figsize = size of ONE panel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows, 1, figsize=(figsize[0], figsize[1] * nrows),
                            sharex=True, facecolor="white")
    for ax in axs:
        _style(ax)
    return fig, axs


def _legend(ax, loc=None):
    """Legend outside the axes (right) so it never hides data; `loc` kept for call-site compatibility."""
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5,
              labelcolor=INK2, borderaxespad=0)


def _finish(fig, ax, path, title, suptitle, jp: JointParams | None = None):
    """Common finishing touches: titles, x-label, layout, save.

    Single axis : `title` is the plot title, `suptitle` the scenario caption above it.
    Several axes: `title` + caption become one figure heading (panels carry their own subtitles).
    """
    import matplotlib.pyplot as plt
    axes = np.atleast_1d(ax)
    xlabel = "time slot index k"
    if jp is not None:
        xlabel += f"  (slot length τ = {jp.master.tau:g} s)"
    if not axes[-1].get_xlabel():
        axes[-1].set_xlabel(xlabel)
    if axes.size == 1:
        axes[0].set_title(title, loc="left", color=INK, fontsize=11)
        if suptitle:
            fig.suptitle(suptitle, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
    else:
        head = title + (f"\n{suptitle}" if suptitle else "")
        fig.suptitle(head, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _shade_silent(ax, jp: JointParams):
    """Shade the slots where the slave's schedule is 0 (it may not transmit data)."""
    first = True
    for i in np.flatnonzero(~jp.slave_schedule):
        ax.axvspan(i - 0.5, i + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0,
                   label="slave silent slot (schedule s_k = 0)" if first else None)
        first = False


def _mark(ax, t, y, mask, label, marker, color, size=55):
    """Scatter markers on the slots selected by `mask` (no legend entry if none are selected)."""
    mask = np.asarray(mask, bool)
    if mask.any():
        ax.scatter(t[mask], np.asarray(y)[mask], s=size, marker=marker, color=color,
                   zorder=5, label=label, edgecolors="white", linewidths=0.6)


def _rates_bits(jp: JointParams, r: JointResult):
    """Per-slot throughput of the master and slave links in bits: (tau/2) ln(1+gain*P) / ln 2.
    With jp.slave_rate_window (burst of length T_k) the slave term is (T_k/2) ln(1 + hbar*tau*Pbar/T_k) / ln 2."""
    m = jp.master
    if jp.slave_rate_window is not None:
        T = jp.Twin
        return (0.5 * m.tau * np.log1p(m.h * r.P) * BITS_PER_NAT,
                0.5 * T * np.log1p(jp.h_bar * m.tau * r.Pbar / T) * BITS_PER_NAT)
    return (0.5 * m.tau * np.log1p(m.h * r.P) * BITS_PER_NAT,
            0.5 * m.tau * np.log1p(jp.h_bar * r.Pbar) * BITS_PER_NAT)


def _bound_masks(jp: JointParams, r: JointResult):
    """Slots where a causality / capacity constraint is binding (the water level may jump there).

    master_empty : F_k = fL    master battery empty / at its reserve floor after slot k  (M1 / M6)  [not the last slot]
    master_full  : F_k = cap_k master battery full after slot k    (M2, overflow)    [finite E_max only]
    slave_empty  : b_k = B_min slave battery at its floor          (S1, causality)   [not the last slot]
    slave_full   : b_peak_k = B_max  slave battery full on arrival (S2, overflow)
    """
    from ..solver import master_floor_profile
    m, N = jp.master, jp.master.N
    master_empty = np.zeros(N, bool)
    master_full = np.zeros(N, bool)
    master_empty[:-1] = r.F[:-1] <= master_floor_profile(jp)[1:-1] + EDGE_TOL
    cap = np.asarray(m.cap, float)
    fin = np.isfinite(cap)
    master_full[:-1][fin] = r.F[:-1][fin] >= cap[fin] - EDGE_TOL
    slave_empty = np.zeros(N, bool)
    slave_empty[:-1] = r.b[:-1] <= jp.B_min + EDGE_TOL
    slave_full = r.b_peak >= jp.B_max - EDGE_TOL
    return master_empty, master_full, slave_empty, slave_full
