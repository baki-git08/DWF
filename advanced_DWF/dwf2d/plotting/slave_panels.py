"""Slave-load panels and CSV (need the dict from slave_load.make_load(jp)).
"""
from __future__ import annotations

import numpy as np

from ..params import JointParams, JointResult
from .style import C_LINK, C_MASTER, C_SLAVE, GRID, INK, INK2, _finish, _legend, _new_axes, _new_axis, _shade_silent


# ----------------------------------------------------------------------------
# Slave load panels (need the dict from slave_load.make_load(jp); the solver module itself never
# imports the component model).  TX energy is the solver's slave data energy tau*Pbar_i.
# ----------------------------------------------------------------------------
C_MCU, C_MEM, C_MPPT = "#8a5cd6", "#c9a227", "#d6457a"
LOAD_COLORS = {"camera": C_MASTER, "mcu": C_MCU, "flash": C_MEM, "mppt": C_MPPT, "tx": C_SLAVE}
LOAD_LABELS = {"camera": "camera", "mcu": "MCU", "flash": "external memory", "mppt": "MPPT", "tx": "THz TX data energy, tau*Pbar_k (solver)"}
LEGACY_KEYS = ("camera", "mcu", "flash", "mppt")
# stage-6 event components beyond the four above (colour, label); LOAD_COLORS / LOAD_LABELS stay as they were
_EXTRA = {"photomixer": ("#5a8f6b", "photomixer (sleep)")}
_color = lambda k: LOAD_COLORS.get(k, _EXTRA.get(k, (INK2, k))[0])
_label = lambda k: LOAD_LABELS.get(k, _EXTRA.get(k, (INK2, k))[1])


def _overhead_keys(load: dict) -> tuple:
    """Non-TX component keys of the load dict: the stage-6 event profile lists its own (overhead_keys)."""
    return tuple(load.get("overhead_keys", LEGACY_KEYS))


def slave_load_components(jp: JointParams, r: JointResult, load: dict) -> dict:
    """Per-slot energies [J]: the non-TX components from the profile (camera, MCU, memory, MPPT; the event profile
    may add more, e.g. the photomixer's sleep draw), tx = tau*Pbar_i from the solver, c = their non-TX sum and
    load = c + tx."""
    comp = {k: np.asarray(load[k], float) for k in _overhead_keys(load)}
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
    for k in _overhead_keys(load) + ("tx",):
        ax.bar(t, comp[k], bottom=bottom, color=_color(k), width=0.8, label=_label(k), zorder=2)
        bottom += comp[k]
    ax.set_ylabel("slave energy per slot [J]")
    _legend(ax)
    _finish(fig, ax, path, "Slave energy per slot by component", title, jp)


def plot_slave_energy_breakdown(jp: JointParams, r: JointResult, load: dict, path: str, title: str = ""):
    """Horizon energy of the slave by component with percentages (TX = solver data energy)."""
    comp = slave_load_components(jp, r, load)
    ks = list(_overhead_keys(load)) + ["tx"]
    tot = {k: float(comp[k].sum()) for k in ks}
    T = sum(tot.values()) or 1.0
    fig, ax = _new_axis(figsize=(10.5, 4.2))
    y = np.arange(len(ks))
    ax.barh(y, [tot[k] for k in ks], color=[_color(k) for k in ks], height=0.6, zorder=2)
    for yi, k in zip(y, ks):
        ax.annotate(f"  {tot[k]:.4g} J ({100 * tot[k] / T:.1f}%)", (tot[k], yi), va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    names = {"camera": "camera", "mcu": "MCU", "flash": "external memory", "mppt": "MPPT", "tx": "THz TX data (solver)",
             "photomixer": "photomixer (sleep)"}
    ax.set_yticklabels([names.get(k, k) for k in ks], color=INK2)
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


def write_slave_energy_csv(path: str, jp: JointParams, r: JointResult, load: dict, link_rows: list | None = None,
                           stop: list | None = None):
    """slave_energy_profile.csv, same columns as slave_load.py writes: slot,on,camera_J,mcu_J,flash_J,mppt_J,
    tx_J,load_J,stored_J,delta_sent_J,battery_end_J   (tx_J = tau*Pbar, load_J = c + tx_J, stored_J = a*delta,
    delta_sent_J = delta, battery_end_J = b - B_min).

    Stage 6 (only when given; columns APPENDED at the end, the legacy columns are unchanged):
      <extra component>_J      non-TX components beyond the four above (e.g. photomixer_J, its sleep draw)
      link_rows (dwf_diagnostics.link_table): event_release, event_outcome, x_cap_J, P_pk_W, radiated_W, snr_burst,
                               ber, ber_pass, nats  (per run slot; empty in slots without an event)
      stop (solver.maximal_sharing): master_stop  (the bound that stopped the master's sharing in that slot)"""
    import csv
    comp = slave_load_components(jp, r, load)
    stored = jp.a * r.delta
    extra = [k for k in _overhead_keys(load) if k not in LEGACY_KEYS]
    by_run = {x["run"]: x for x in (link_rows or []) if x.get("run") is not None and x["outcome"] in ("run", "rescheduled")}
    ev_cols = ["event_release", "event_outcome", "x_cap_J", "P_pk_W", "radiated_W", "snr_burst", "ber", "ber_pass", "nats"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "on", "camera_J", "mcu_J", "flash_J", "mppt_J", "tx_J", "load_J",
                    "stored_J", "delta_sent_J", "battery_end_J"]
                   + [f"{k}_J" for k in extra] + (ev_cols if link_rows is not None else []) + (["master_stop"] if stop is not None else []))
        for i in range(jp.master.N):
            row = [i, int(jp.slave_schedule[i]), *[f"{comp[k][i]:.6g}" for k in ("camera", "mcu", "flash", "mppt", "tx")],
                   f"{comp['load'][i]:.6g}", f"{stored[i]:.6g}", f"{r.delta[i]:.6g}", f"{r.b[i] - jp.B_min:.6g}"]
            row += [f"{comp[k][i]:.6g}" for k in extra]
            if link_rows is not None:
                x = by_run.get(i)
                row += ([x["release"], x["outcome"], f"{float(jp.txcap[i]):.6g}", f"{x['P_pk_W']:.6g}", f"{x['radiated_W']:.6g}",
                         f"{x['snr']:.6g}", f"{x['ber']:.6g}", int(x["ber_pass"]), f"{x['nats']:.6g}"] if x else [""] * len(ev_cols))
            if stop is not None:
                row.append(stop[i])
            w.writerow(row)


def write_refusal_history_csv(path: str, history: list):
    """refusal_history.csv: the stage-6 refusal loop pass by pass (physical.solve_with_refusals, RefusalResult.history).
    One row per event action, so it filters easily:

        pass, reason, start, action, from_slot, to_slot, release
          pass    1-based pass number (the "refusal loop N pass(es)" count)
          reason  "supercap short at slot k0" (the event load cannot be paid, latest event at or before k0 refused),
                  "x < x_min" (burst below the 1 mW radiated floor), "all events affordable" (last pass), or
                  "infeasible at slot k0 with no event at or before it: ..." (the loop stops, unresolved)
          start   how joint_dwf started on that pass ("infeasible" when the solve raised InfeasibleSlaveBudget)
          action  "moved" (to_slot = the slot it moves to), "dropped" (to_slot empty), "" = pass with no action
    Every refused event appears as exactly one moved or dropped row, so "refused" is not a separate column."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pass", "reason", "start", "action", "from_slot", "to_slot", "release"])
        for p, h in enumerate(history, 1):
            head = [p, h["reason"], h["start"]]
            rows = ([["moved", i, j, rel] for i, j, rel in h["moves"]]
                    + [["dropped", i, "", rel] for i, rel in h["dropped"]])
            for row in sorted(rows, key=lambda x: x[1]) or [["", "", "", ""]]:
                w.writerow(head + row)


def plot_event_bursts(jp: JointParams, r: JointResult, link_rows: list, profile: dict, path: str, title: str = "",
                      ber_target: float = 1e-3):
    """Stage 6: THz burst of every event.  Top: burst energy x per run slot against x_min (1 mW radiated floor)
    and x_cap (20 mW radiated cap), refused / dropped releases marked.  Bottom: burst SNR and theoretical BER
    against the BER target."""
    t = np.arange(jp.master.N)
    fig, (ax1, ax2) = _new_axes(2)
    _shade_silent(ax1, jp)
    run = [x for x in link_rows if x["outcome"] in ("run", "rescheduled")]
    rs = np.array([x["run"] for x in run], int)
    xs = np.array([x["x_J"] for x in run], float)
    ax1.bar(rs, xs * 1e3, color=C_SLAVE, width=0.8, zorder=2, label="burst energy x_k (event ran)")
    resched = [x for x in run if x["outcome"] == "rescheduled"]
    if resched:
        ax1.scatter([x["run"] for x in resched], [x["x_J"] * 1e3 for x in resched], marker="o", s=40, facecolor="none",
                    edgecolor=INK, zorder=3, label="rescheduled event (ran after its release slot)")
    dropped = [x["release"] for x in link_rows if x["outcome"] == "dropped"]
    if dropped:
        ax1.scatter(dropped, np.zeros(len(dropped)), marker="x", s=50, color=C_LINK, zorder=3, label="dropped event (release slot)")
    ax1.axhline(profile["x_cap"] * 1e3, color=INK2, lw=1.2, ls="--", label=f"x_cap = {profile['x_cap'] * 1e3:.4g} mJ (20 mW radiated)")
    ax1.axhline(profile["x_min"] * 1e3, color=C_LINK, lw=1.2, ls=":", label=f"x_min = {profile['x_min'] * 1e3:.4g} mJ (1 mW radiated)")
    ax1.set_ylabel("burst energy [mJ]")
    ax1.set_ylim(bottom=0)
    ax1.set_title("THz burst energy per event", loc="left", color=INK2, fontsize=10)
    _legend(ax1, "upper left")
    if run:
        snr_db = np.array([x["snr_dB"] for x in run], float)
        ax2.plot(rs, snr_db, "o", color=C_SLAVE, label="burst SNR [dB]")
        fin = snr_db[np.isfinite(snr_db)]
        if fin.size:                                  # equal SNRs would otherwise get an offset axis
            ax2.set_ylim(fin.min() - 1.0, fin.max() + 1.0)
    ax2.ticklabel_format(axis="y", useOffset=False)
    ax2.set_ylabel("burst SNR [dB]")
    ax2.set_xlim(-0.5, jp.master.N - 0.5)
    ax3 = ax2.twinx()
    if run:
        ax3.semilogy(rs, [max(x["ber"], 1e-300) for x in run], "s", color=C_MASTER, mfc="none", label="theoretical BER (right)")
    ax3.axhline(ber_target, color=C_MASTER, lw=1, ls=":", label=f"BER target {ber_target:g}")
    ax3.set_ylabel("BER", color=C_MASTER)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", frameon=False)
    ax2.set_title("Burst SNR and theoretical BER", loc="left", color=INK2, fontsize=10)
    _finish(fig, (ax1, ax2), path, "Slave THz bursts: energy, SNR and BER per event", title, jp)
