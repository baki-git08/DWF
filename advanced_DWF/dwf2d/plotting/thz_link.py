"""THz-link figures of the DWF, taken from Modulation_THz_slave_profile.ipynb (sections 7-12; the notebook is the
reference and is not changed by this module).

The plotting code follows the notebook's cells; the notebook globals it read are the fields of a ThzLinkSetup (S).
Every plot_* function draws one figure in the notebook's look (NOTEBOOK_RC) and RETURNS it; write_thz_link_plots
saves them (integrated_demo writes them to out_integrated_demo/thz_link/).

    S = setup_from_slave_config()            # DWF: slave_config.py link values + the integrated_demo operating point
    write_thz_link_plots(S, "out_integrated_demo/thz_link")

    figure                         notebook   function
    BER vs SNR / distance / rate / Pr   7     plot_ber_sweep(S, sweep)
    link characteristics                8     plot_link_characteristics(S)
    received constellations             9     plot_constellations(S, demo_results)   (demo_results = simulate_demo(S))
    eye diagrams                        9     plot_eye_diagrams(S, demo_results)
    simulation vs theory               10     plot_sim_vs_theory(S, validation)      (validation = simulate_validation(S))
    aperture gain / Friis limit      12.1     plot_gain_vs_aperture(S)
    SNR and Pr vs distance, gains    12.2     plot_snr_power_vs_distance(S)
    BER vs distance, gains           12.3     plot_ber_vs_distance_gains(S)
    maximum range vs gain            12.4     plot_max_range_vs_gain(S)
    range heat-map (G_tx, G_rx)      12.5     plot_max_range_heatmap(S)

S.operating_point (dict or None) adds the DWF operating point (the slave's burst at the photomixer cap, D_THZ_M) to
the SNR / distance / received-power sweeps, the coupling-loss panel and the gain-study distance plots.  None keeps
the notebook figures exactly as they were.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from .. import channels as ch

# The notebook's rcParams (its config cell); applied per figure so the DWF's own style is not touched.
NOTEBOOK_RC = {"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.35,
               "axes.spines.top": False, "axes.spines.right": False}
SWEEP_ORDER = ["SNR", "Distance", "Data Rate", "Received Power"]
OP_STYLE = dict(color="tab:purple", ls="-.", lw=1.6)        # DWF operating point marker


def notebook_schemes(pam_bits: int = 2):
    """The notebook's scheme registry (section 3): (SCHEMES, SCHEME_COLORS, SCHEME_LS, SCHEME_MARK, PAM_NAME)."""
    PAM_NAME = f"{2 ** pam_bits}-PAM"
    SCHEMES = {
        "4-QAM":  ch.make_scheme("4-QAM",  2, ch.gray_qam(1, 1), ch.qam_ber_theory(1, 1)),
        "8-QAM":  ch.make_scheme("8-QAM",  3, ch.gray_qam(2, 1), ch.qam_ber_theory(2, 1)),
        "16-QAM": ch.make_scheme("16-QAM", 4, ch.gray_qam(2, 2), ch.qam_ber_theory(2, 2)),
        "64-QAM": ch.make_scheme("64-QAM", 6, ch.gray_qam(3, 3), ch.qam_ber_theory(3, 3)),
        PAM_NAME: ch.make_scheme(PAM_NAME, pam_bits, ch.gray_pam(pam_bits), ch.pam_ber_theory(pam_bits)),
    }
    SCHEME_COLORS = {name: plt.cm.tab10(i) for i, name in enumerate(SCHEMES)}   # one fixed colour per scheme
    SCHEME_LS = {name: ((0, (5, 2)) if name == PAM_NAME else "-") for name in SCHEMES}   # PAM dashed
    SCHEME_MARK = {name: ("s" if name == PAM_NAME else "o") for name in SCHEMES}
    return SCHEMES, SCHEME_COLORS, SCHEME_LS, SCHEME_MARK, PAM_NAME


@dataclass
class ThzLinkSetup:
    """Everything the notebook's THz-link figures read (its section-1 configuration, by the notebook's names)."""
    freqs_hz: dict                      # FREQS_HZ   {"0.1 THz": 0.1e12, ...}
    gains_dbi: dict                     # GAINS_DBI  {fname: (G_tx, G_rx)}
    pt_w: float                         # PT_W
    noise_floor_w: float                # NOISE_FLOOR_W
    noise_psd_w_hz: float               # NOISE_PSD_W_HZ
    r_s: float                          # R_s
    schemes: dict                       # SCHEMES
    scheme_colors: dict                 # SCHEME_COLORS
    scheme_ls: dict                     # SCHEME_LS
    scheme_mark: dict                   # SCHEME_MARK
    pam_name: str                       # PAM_NAME
    simulate_link: object               # channels.make_link_simulator(...) (carries .PULSE, .DELAY, .sps)
    d_demo_m: dict = field(default_factory=dict)     # D_DEMO_M (solved for demo_target_snr_db when empty)
    alpha: float = ch.ALPHA_THZ                      # ALPHA_THZ
    dist_sweep_m: np.ndarray = field(default_factory=lambda: np.linspace(0.01, 3.0, 400) * 1e-3)   # DIST_SWEEP_M
    snr_db_range: np.ndarray = field(default_factory=lambda: np.linspace(0, 30, 300))
    p_rec_sweep_dbw: np.ndarray | None = None        # default: noise floor -5 .. +40 dB
    data_rates_sweep: np.ndarray = field(default_factory=lambda: np.logspace(8, 11.3, 200))
    ebn0_validation_db: np.ndarray = field(default_factory=lambda: np.arange(0, 21, 2))
    num_symbols_ber: int = 200_000
    demo_target_snr_db: float = 20.0
    demo_fallback_m: float = 0.4e-3
    gain_study_dbi: list = field(default_factory=lambda: [0, 5, 10, 20, 40])
    eye_schemes: list = field(default_factory=lambda: ["4-QAM", "16-QAM"])
    heatmap_scheme: str = "16-QAM"
    ber_target: float = 1e-3
    aperture_a_m: float = ch.APERTURE_A_M
    aperture_b_m: float = ch.APERTURE_B_M
    aperture_efficiency: float = ch.APERTURE_EFFICIENCY
    operating_point: dict | None = None              # dict(freq_label, d_m, snr_db, pr_dbw, label) or None

    def __post_init__(self):
        if self.p_rec_sweep_dbw is None:
            fl = self.noise_floor_dbw
            self.p_rec_sweep_dbw = np.linspace(fl - 5, fl + 40, 300)
        if not self.d_demo_m:                        # demo distance per frequency, as in the notebook (section 4)
            for fname, f in self.freqs_hz.items():
                d = self.solve_distance_for_snr(f, self.demo_target_snr_db)
                if not np.isfinite(d):
                    d = self.demo_fallback_m
                    print(f"Warning: {fname} cannot reach {self.demo_target_snr_db} dB SNR; falling back to {d*1e3} mm")
                self.d_demo_m[fname] = d

    # ---- link budget bound to this setup (the notebook's wrappers) ----
    @property
    def pt_dbw(self) -> float:
        return 10 * np.log10(self.pt_w)

    @property
    def noise_floor_dbw(self) -> float:
        return 10 * np.log10(self.noise_floor_w)

    def gains(self, fname):
        """(G_tx, G_rx) in dBi used in the main sections for carrier `fname`."""
        return self.gains_dbi[fname]

    def gains_hz(self, freq_hz):
        """(G_tx, G_rx) in dBi for a carrier given in Hz (one of freqs_hz)."""
        return next(self.gains_dbi[n] for n, f in self.freqs_hz.items() if f == freq_hz)

    def gain_label(self):
        """Gain text for figure titles: one pair if both frequencies share it, else one pair per frequency."""
        if len(set(self.gains_dbi.values())) == 1:
            g_tx, g_rx = next(iter(self.gains_dbi.values()))
            return f"G_tx = {g_tx:g} dBi, G_rx = {g_rx:g} dBi"
        return ", ".join(f"{n}: G_tx/G_rx = {gt:g}/{gr:g} dBi" for n, (gt, gr) in self.gains_dbi.items())

    def _resolve(self, freq_hz, g_tx_dbi, g_rx_dbi):
        g_tx, g_rx = self.gains_hz(freq_hz)
        return (g_tx if g_tx_dbi is None else g_tx_dbi), (g_rx if g_rx_dbi is None else g_rx_dbi)

    def link_budget(self, freq_hz, d_m, g_tx_dbi=None, g_rx_dbi=None, p_tx_w=None, noise_floor_w=None):
        """Full link budget at distance d (scalar or array); gains default to this frequency's gains_dbi."""
        g_tx, g_rx = self._resolve(freq_hz, g_tx_dbi, g_rx_dbi)
        return ch.link_budget(freq_hz, d_m, g_tx, g_rx, self.pt_w if p_tx_w is None else p_tx_w,
                              self.noise_floor_w if noise_floor_w is None else noise_floor_w, alpha=self.alpha)

    def solve_distance_for_snr(self, freq_hz, target_snr_db, g_tx_dbi=None, g_rx_dbi=None):
        """Distance (m) at which the budget gives the target SNR (SNR falls monotonically with d)."""
        g_tx, g_rx = self._resolve(freq_hz, g_tx_dbi, g_rx_dbi)
        return ch.solve_distance_for_snr(freq_hz, target_snr_db, g_tx, g_rx,
                                         p_tx_w=self.pt_w, noise_floor_w=self.noise_floor_w, alpha=self.alpha)

    def ber_at(self, freq_hz, scheme, d_m, g_tx_dbi, g_rx_dbi):
        """Theoretical BER at distance d (array ok) for given antenna gains."""
        return ch.ber_at(freq_hz, scheme, d_m, g_tx_dbi, g_rx_dbi, p_tx_w=self.pt_w,
                         noise_floor_w=self.noise_floor_w, alpha=self.alpha)

    def max_range_m(self, freq_hz, scheme, g_tx_dbi, g_rx_dbi, target=None):
        """Largest distance with theoretical BER <= target (BER rises monotonically with distance)."""
        return ch.max_range_m(freq_hz, scheme, g_tx_dbi, g_rx_dbi, self.ber_target if target is None else target,
                              p_tx_w=self.pt_w, noise_floor_w=self.noise_floor_w, alpha=self.alpha)

    def _op(self, fname):
        """The operating point if it belongs to carrier `fname`, else None."""
        op = self.operating_point
        return op if (op is not None and op["freq_label"] == fname) else None


def setup_from_slave_config(seed: int = 42, operating_point: bool = True, pam_bits: int = 2,
                            preset: str | None = None) -> ThzLinkSetup:
    """ThzLinkSetup from slave_config.py (STAGE 6 block): frequencies, per-frequency gains (LEGACY_GAINS), Pt, noise,
    R_s, alpha, BER target; the sweep grids and figure choices at the notebook defaults; its own seeded rng (seed).
    operating_point=True marks the integrated_demo burst: THZ_PRESET at D_THZ_M with the photomixer at its 20 mW
    radiated cap (= hbar x P_PM_ELEC_W, slave_load.thz_link; preset None = slave_config.THZ_PRESET)."""
    import slave_config as cfg
    import slave_load
    freqs = dict(cfg.THZ_FREQS_HZ)
    gains = {n: (ch.LEGACY_GAIN_DBI if cfg.LEGACY_GAINS else tuple(cfg.GAIN_DBI_BY_FREQ[n])) for n in freqs}
    psd = ch.noise_psd_w_hz(cfg.T_NOISE_K, cfg.NOISE_FIGURE_DB) if cfg.NOISE_MODE == "thermal" else 1e-10
    nf = ch.noise_floor_w(cfg.NOISE_MODE, cfg.T_NOISE_K, cfg.NOISE_FIGURE_DB, cfg.R_s)
    schemes, colors, ls, mark, pam_name = notebook_schemes(pam_bits)
    sim = ch.make_link_simulator(0.35, 8, 4, np.random.default_rng(seed), 10_000)   # notebook RRC 0.35, span 8, sps 4
    S = ThzLinkSetup(freqs_hz=freqs, gains_dbi=gains, pt_w=cfg.PT_W, noise_floor_w=nf, noise_psd_w_hz=psd, r_s=cfg.R_s,
                     schemes=schemes, scheme_colors=colors, scheme_ls=ls, scheme_mark=mark, pam_name=pam_name,
                     simulate_link=sim, alpha=cfg.ALPHA_THZ, ber_target=cfg.BER_TARGET)
    if operating_point:
        lk = slave_load.thz_link(preset)
        b = S.link_budget(lk["freq_hz"], lk["d_eff"], p_tx_w=lk["eta_tx"] * cfg.P_PM_ELEC_W)
        S.operating_point = dict(freq_label=lk["freq_label"], d_m=lk["d_eff"], snr_db=float(b["snr_db"]),
                                 pr_dbw=float(b["pr_dbw"]),
                                 label=f"DWF operating point ({lk['preset']}, {lk['d_eff'] * 1e3:.3g} mm, "
                                       f"{float(b['snr_db']):.1f} dB)")
    return S


# ----------------------------------------------------------------------------
# Section 7: BER sweeps
# ----------------------------------------------------------------------------
def make_sweeps(S: ThzLinkSetup, fname):
    """SWEEPS[freq][sweep] = (x, ebn0_fn(k), xlabel, xscale)."""
    f = S.freqs_hz[fname]
    snr_lin_range = 10 ** (S.snr_db_range / 10)
    p_rec_lin = 10 ** (S.p_rec_sweep_dbw / 10)
    snr_dist = S.link_budget(f, S.dist_sweep_m)["snr"]
    pr_demo = S.link_budget(f, S.d_demo_m[fname])["pr_w"]
    return {
        "SNR":            (S.snr_db_range,           lambda k: snr_lin_range / k,                       "Es/N0 (dB)",           "linear"),
        "Distance":       (S.dist_sweep_m * 1e3,     lambda k: snr_dist / k,                            "Distance (mm)",        "linear"),
        "Data Rate":      (S.data_rates_sweep / 1e6, lambda k: pr_demo / (S.noise_psd_w_hz * S.data_rates_sweep), "Data rate (Mbps)", "log"),
        "Received Power": (S.p_rec_sweep_dbw,        lambda k: p_rec_lin / S.noise_floor_w / k,         "Received power (dBW)", "linear"),
    }


def plot_ber_curves(S: ThzLinkSetup, ax, fname, sweep, sweeps=None):
    x, ebn0_fn, xlabel, xscale = (sweeps or make_sweeps(S, fname))[sweep]
    for name, cfg in S.schemes.items():
        ax.plot(x, cfg["ber_func"](ebn0_fn(cfg["bits_per_symbol"])), lw=1.8, ls=S.scheme_ls[name], color=S.scheme_colors[name], label=name)
    if sweep == "Distance":
        d_nf = ch.near_field_limit_m(S.freqs_hz[fname], *S.gains(fname)) * 1e3
        ax.axvspan(0, min(d_nf, x.max()), color="grey", alpha=0.18, label="Friis not valid")
        ax.axvline(S.d_demo_m[fname] * 1e3, color="k", ls=":", lw=1.2, label="demo distance")
    op = S._op(fname)
    if op is not None and sweep in ("SNR", "Distance", "Received Power"):
        xv = {"SNR": op["snr_db"], "Distance": op["d_m"] * 1e3, "Received Power": op["pr_dbw"]}[sweep]
        ax.axvline(xv, **OP_STYLE, label=op["label"])
    ax.set(xscale=xscale, yscale="log", ylim=(1e-6, 1), xlabel=xlabel, ylabel="BER",
           title=f"{fname}: BER vs {sweep.lower()}")
    ax.grid(True, which="both")
    ax.legend(fontsize=8)


def plot_ber_sweep(S: ThzLinkSetup, sweep: str):
    """One section-7 figure: BER vs `sweep` (one of SWEEP_ORDER), one column per carrier frequency."""
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(1, len(S.freqs_hz), figsize=(6.5 * len(S.freqs_hz), 4.6), sharey=True)
        for ax, fname in zip(axes, S.freqs_hz):
            plot_ber_curves(S, ax, fname, sweep)
        fig.suptitle(f"BER vs {sweep.lower()}   (Pt = {S.pt_w*1e3:.0f} mW, {S.gain_label()})",
                     fontweight="bold")
        plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Section 8: link characteristics
# ----------------------------------------------------------------------------
def plot_link_characteristics(S: ThzLinkSetup):
    """Coupling loss / received power vs distance, loss components, Shannon limit."""
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        freq_col = {"0.1 THz": "tab:blue", "0.3 THz": "tab:red"}
        snr_lin_range = 10 ** (S.snr_db_range / 10)

        # (1) coupling loss and received power
        ax = axes[0]
        for fname, f in S.freqs_hz.items():
            b = S.link_budget(f, S.dist_sweep_m)
            x = S.dist_sweep_m * 1e3
            ax.plot(x, b["coupling_loss_db"], color=freq_col[fname], ls="-", lw=2, label=f"{fname}")
            ax.axvline(S.d_demo_m[fname] * 1e3, color=freq_col[fname], ls=":", lw=1, label=f"{fname} demo ({S.d_demo_m[fname]*1e3:.2f} mm)")
            ax.axvspan(0, min(ch.near_field_limit_m(f, *S.gains(fname)) * 1e3, x.max()), color=freq_col[fname], alpha=0.07)
            op = S._op(fname)
            if op is not None:
                ax.axvline(op["d_m"] * 1e3, **OP_STYLE, label=op["label"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="Distance (mm)", ylabel="Coupling loss (dB)  [negative: Pr > Pt]",
               title="Coupling loss and received power\n(shaded: Friis not valid)")
        pt_dbw = S.pt_dbw
        sec = ax.secondary_yaxis("right", functions=(lambda pl: pt_dbw - pl, lambda pr: pt_dbw - pr))
        sec.set_ylabel("Received power (dBW)")
        ax.legend(fontsize=8)

        # (2) loss components
        ax = axes[1]
        for fname, f in S.freqs_hz.items():
            b = S.link_budget(f, S.dist_sweep_m)
            x = S.dist_sweep_m * 1e3
            ax.plot(x, b["fspl_db"], color=freq_col[fname], ls="-", lw=1.8, label=f"FSPL {fname}")
        ax.plot(x, S.link_budget(S.freqs_hz["0.1 THz"], S.dist_sweep_m)["absorption_db"], color="k", lw=2, label="Absorption (both)")
        g_sums = {fname: sum(S.gains(fname)) for fname in S.freqs_hz}
        if len(set(g_sums.values())) == 1:
            g_sum = next(iter(g_sums.values()))
            ax.axhline(g_sum, color="tab:green", ls="--", lw=1.8, label=f"G_tx + G_rx = {g_sum:g} dB")
        else:
            for fname, g_sum in g_sums.items():
                ax.axhline(g_sum, color=freq_col[fname], ls="--", lw=1.8, label=f"G_tx + G_rx = {g_sum:g} dB ({fname})")
        ax.set(xlabel="Distance (mm)", ylabel="Loss / gain (dB)", title="Where the dB go: spreading vs absorption vs gain")
        ax.legend(fontsize=8)

        # (3) Shannon limit
        ax = axes[2]
        capacity = S.r_s * np.log2(1 + snr_lin_range) / 1e9
        ax.plot(S.snr_db_range, capacity, color="black", lw=2, label=f"Shannon limit, B = {S.r_s/1e9:g} GHz")
        for name, cfg in S.schemes.items():
            ax.axhline(S.r_s * cfg["bits_per_symbol"] / 1e9, color=S.scheme_colors[name], ls=":", lw=1.5,
                       label=f"{name}: {S.r_s * cfg['bits_per_symbol'] / 1e9:g} Gbps")
        ax.set(xlabel="Es/N0 (dB)", ylabel="Data rate (Gbps)", title="Shannon limit vs SNR")
        ax.legend(fontsize=8)

        plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Section 9: constellations and eye diagrams (Monte-Carlo)
# ----------------------------------------------------------------------------
def simulate_demo(S: ThzLinkSetup) -> dict:
    """demo_results[(frequency, scheme)]: simulate_link at the SNR the budget gives at each demo distance."""
    demo_results = {}      # (frequency, scheme) -> simulation result at the demo operating point
    for fname, f in S.freqs_hz.items():
        snr_db = float(S.link_budget(f, S.d_demo_m[fname])["snr_db"])
        for name, cfg in S.schemes.items():
            demo_results[(fname, name)] = S.simulate_link(cfg, snr_db)
    return demo_results


def plot_constellations(S: ThzLinkSetup, demo_results: dict):
    """Received constellations: rows = frequency, columns = scheme; red crosses = ideal points."""
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(len(S.freqs_hz), len(S.schemes), figsize=(4.2 * len(S.schemes), 4.4 * len(S.freqs_hz)))
        for r, fname in enumerate(S.freqs_hz):
            for c, (name, cfg) in enumerate(S.schemes.items()):
                ax = axes[r, c]
                res = demo_results[(fname, name)]
                ax.scatter(res["rx_sym"].real, res["rx_sym"].imag, s=1.5, alpha=0.45, color=S.scheme_colors[name])
                ax.scatter(cfg["points"].real, cfg["points"].imag, color="red", marker="x", s=40, linewidths=1.6, label="ideal")
                ax.set_title(f"{fname}: {name}\n{res['snr_db']:.1f} dB @ {S.d_demo_m[fname]*1e3:.2f} mm, sim BER = {res['ber']:.2e}", fontsize=9)
                if cfg["is_one_dim"]:
                    ax.set_ylim(-0.5, 0.5)
                else:
                    ax.set_aspect("equal", "box")
                if r == 0 and c == 0:
                    ax.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
    return fig


def plot_eye_diagrams(S: ThzLinkSetup, demo_results: dict):
    """Eye diagrams (I-channel, matched-filter output): rows = scheme (eye_schemes), columns = frequency."""
    sps, DELAY = S.simulate_link.sps, S.simulate_link.DELAY
    with plt.rc_context(NOTEBOOK_RC):
        n_traces, span = 200, 2 * sps
        starts = DELAY - sps // 2 + sps * np.arange(5, 5 + n_traces)[:, None] + np.arange(span + 1)[None, :]
        t = np.arange(span + 1) / sps - (sps // 2) / sps      # symbol periods; decision instants at t = 0 and 1

        fig, axes = plt.subplots(len(S.eye_schemes), len(S.freqs_hz), figsize=(6.5 * len(S.freqs_hz), 4.2 * len(S.eye_schemes)), squeeze=False)
        for r, scheme_key in enumerate(S.eye_schemes):
            for c, fname in enumerate(S.freqs_hz):
                ax = axes[r, c]
                res = demo_results[(fname, scheme_key)]
                ax.plot(t, np.real(res["rx_mf"][starts]).T, color=S.scheme_colors[scheme_key], alpha=0.1)
                ax.set(xlabel="Time (symbol periods)", ylabel="Amplitude",
                       title=f"Eye diagram: {fname} {scheme_key} ({res['snr_db']:.1f} dB)")
        plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Section 10: simulation vs theory (Monte-Carlo)
# ----------------------------------------------------------------------------
def simulate_validation(S: ThzLinkSetup) -> dict:
    """Monte-Carlo BER over ebn0_validation_db for every scheme (num_symbols_ber symbols per point).
    Returns dict(sim={scheme: array, 0 -> NaN}, rows=[...] (the notebook's validation_rows))."""
    validation_rows, sims = [], {}
    for name, cfg in S.schemes.items():
        k = cfg["bits_per_symbol"]
        sim = []
        for eb in S.ebn0_validation_db:
            res = S.simulate_link(cfg, eb + 10 * np.log10(k), n_symbols=S.num_symbols_ber)
            sim.append(res["ber"])
            validation_rows.append({"Modulation": name, "Eb/N0 (dB)": eb,
                                    "Theory BER": float(cfg["ber_func"](10 ** (eb / 10))),
                                    "Simulated BER": res["ber"], "Bit errors": res["bit_errors"]})
        sim = np.array(sim, dtype=float)
        sim[sim == 0] = np.nan
        sims[name] = sim
    return dict(sim=sims, rows=validation_rows)


def plot_sim_vs_theory(S: ThzLinkSetup, validation: dict):
    """Theoretical BER vs Eb/N0 (lines) and the Monte-Carlo points of simulate_validation (markers)."""
    with plt.rc_context(NOTEBOOK_RC):
        fig, ax = plt.subplots(figsize=(9, 6))
        ebn0_fine_db = np.linspace(0, 20, 200)
        for name, cfg in S.schemes.items():
            ax.plot(ebn0_fine_db, cfg["ber_func"](10 ** (ebn0_fine_db / 10)), lw=1.6, ls=S.scheme_ls[name], color=S.scheme_colors[name], label=f"{name} (theory)")
            ax.plot(S.ebn0_validation_db, validation["sim"][name], S.scheme_mark[name], color=S.scheme_colors[name], ms=5,
                    mfc="none" if name == S.pam_name else None, label=f"{name} (sim)")
        ax.set(yscale="log", ylim=(1e-6, 1), xlabel="Eb/N0 (dB)", ylabel="BER", title="Simulation vs theory (all schemes)")
        ax.grid(True, which="both")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Section 12: antenna-gain study
# ----------------------------------------------------------------------------
def plot_gain_vs_aperture(S: ThzLinkSetup):
    """12.1: aperture gain vs area; distance beyond which the Friis budget is consistent vs gain."""
    with plt.rc_context(NOTEBOOK_RC):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        areas_mm2 = np.logspace(-1, 3.2, 300)                         # 0.1 mm^2 .. ~1600 mm^2
        ga = lambda f: ch.aperture_gain_dbi(f, S.aperture_a_m, S.aperture_b_m, S.aperture_efficiency)
        for fname, f in S.freqs_hz.items():
            lam = ch.wavelength(f)
            g = 10 * np.log10(4 * np.pi * S.aperture_efficiency * areas_mm2 * 1e-6 / lam ** 2)
            axes[0].plot(areas_mm2, g, lw=2, label=f"{fname} (lambda = {lam*1e3:g} mm)")
            axes[0].plot(S.aperture_a_m * S.aperture_b_m * 1e6, ga(f), "o", ms=8, color="k")
            axes[0].annotate(f"{ga(f):.1f} dBi", (S.aperture_a_m * S.aperture_b_m * 1e6, ga(f)),
                             textcoords="offset points", xytext=(-48, 6), fontsize=9)
        for g_val in S.gain_study_dbi:
            axes[0].axhline(g_val, color="grey", lw=0.7, ls=":")
        axes[0].set(xscale="log", xlabel="Aperture area (mm$^2$)", ylabel="Gain (dBi)",
                    title=f"Aperture gain (eta = {S.aperture_efficiency}); dot = 2 cm x 1.5 cm horn")
        axes[0].legend()

        for fname, f in S.freqs_hz.items():
            gs = np.arange(0, 41)
            axes[1].plot(gs, ch.near_field_limit_m(f, gs, gs) * 1e3, lw=2, label=f"{fname}")
        axes[1].axhspan(0, 2, color="tab:green", alpha=0.12, label="usable link range (about 0 to 2 mm)")
        axes[1].set(yscale="log", xlabel="Gain at each end (dBi)", ylabel="Friis-valid beyond (mm)",
                    title="Distance beyond which the Friis budget is consistent")
        axes[1].legend()
        plt.tight_layout()
    return fig


def plot_validity(ax, x_mm, y, d_nf_mm, color, label):
    """Solid where Friis is valid (d >= d_nf), dashed where it is not."""
    valid = x_mm >= d_nf_mm
    ax.plot(x_mm[~valid], y[~valid], color=color, ls="--", lw=1.4, alpha=0.7)
    ax.plot(x_mm[valid], y[valid], color=color, ls="-", lw=2, label=label)


def plot_snr_power_vs_distance(S: ThzLinkSetup):
    """12.2: SNR (top) and received power (bottom) vs distance for every gain of gain_study_dbi."""
    with plt.rc_context(NOTEBOOK_RC):
        gain_colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(S.gain_study_dbi)))
        x_mm = S.dist_sweep_m * 1e3
        fig, axes = plt.subplots(2, len(S.freqs_hz), figsize=(6.8 * len(S.freqs_hz), 8.6), sharex=True)
        for c, (fname, f) in enumerate(S.freqs_hz.items()):
            for g, col in zip(S.gain_study_dbi, gain_colors):
                b = S.link_budget(f, S.dist_sweep_m, g, g)
                d_nf = ch.near_field_limit_m(f, g, g) * 1e3
                plot_validity(axes[0, c], x_mm, b["snr_db"], d_nf, col, f"G = {g} dBi each")
                plot_validity(axes[1, c], x_mm, b["pr_dbw"], d_nf, col, f"G = {g} dBi each")
            axes[0, c].axhline(0, color="red", ls=":", lw=1)
            axes[0, c].set(ylabel="Es/N0 (dB)", title=f"{fname}: SNR vs distance  (dashed = Friis not valid)", ylim=(-40, 160))
            axes[1, c].axhline(S.pt_dbw, color="k", ls=":", lw=1, label=f"Pt = {S.pt_dbw:.0f} dBW")
            axes[1, c].axhline(S.noise_floor_dbw, color="red", ls=":", lw=1, label="noise floor")
            axes[1, c].set(xlabel="Distance (mm)", ylabel="Received power (dBW)", title=f"{fname}: received power vs distance", ylim=(-300, 60))
            op = S._op(fname)
            if op is not None:
                axes[0, c].plot(op["d_m"] * 1e3, op["snr_db"], marker="*", ms=14, color=OP_STYLE["color"], ls="none", label=op["label"])
                axes[1, c].plot(op["d_m"] * 1e3, op["pr_dbw"], marker="*", ms=14, color=OP_STYLE["color"], ls="none", label=op["label"])
            axes[0, c].legend(fontsize=8)
            axes[1, c].legend(fontsize=8)
        plt.tight_layout()
    return fig


def plot_ber_vs_distance_gains(S: ThzLinkSetup):
    """12.3: BER vs distance, rows = frequency, columns = gain (grey = Friis not valid, dotted = BER target)."""
    with plt.rc_context(NOTEBOOK_RC):
        x_mm = S.dist_sweep_m * 1e3
        fig, axes = plt.subplots(len(S.freqs_hz), len(S.gain_study_dbi), figsize=(4.0 * len(S.gain_study_dbi), 3.9 * len(S.freqs_hz)),
                                 sharey=True, sharex=True)
        for r, (fname, f) in enumerate(S.freqs_hz.items()):
            op = S._op(fname)
            for c, g in enumerate(S.gain_study_dbi):
                ax = axes[r, c]
                for name, cfg in S.schemes.items():
                    ax.plot(x_mm, S.ber_at(f, cfg, S.dist_sweep_m, g, g), color=S.scheme_colors[name], ls=S.scheme_ls[name], lw=1.6, label=name)
                d_nf = ch.near_field_limit_m(f, g, g) * 1e3
                ax.axvspan(0, min(d_nf, x_mm.max()), color="grey", alpha=0.2)
                ax.axhline(S.ber_target, color="k", ls=":", lw=0.9)
                if op is not None:
                    ax.axvline(op["d_m"] * 1e3, **OP_STYLE, label="DWF operating distance")
                ax.set(yscale="log", ylim=(1e-6, 1), title=f"{fname}, G = {g} dBi each")
                if r == len(S.freqs_hz) - 1: ax.set_xlabel("Distance (mm)")
                if c == 0: ax.set_ylabel("BER")
                ax.grid(True, which="both")
        axes[0, 0].legend(fontsize=7)
        if S.operating_point is not None:
            r_op = list(S.freqs_hz).index(S.operating_point["freq_label"])
            axes[r_op, 0].legend(fontsize=7)
        fig.suptitle("BER vs distance for different antenna gains (grey = Friis not valid, dotted = BER target)", fontweight="bold")
        plt.tight_layout()
    return fig


def plot_max_range_vs_gain(S: ThzLinkSetup):
    """12.4: maximum range at the BER target vs the gain at each end, per frequency and scheme."""
    with plt.rc_context(NOTEBOOK_RC):
        GAIN_GRID_DBI = np.arange(0, 41, 2)
        fig, axes = plt.subplots(1, len(S.freqs_hz), figsize=(7 * len(S.freqs_hz), 5), sharey=True)
        for ax, (fname, f) in zip(axes, S.freqs_hz.items()):
            for name, cfg in S.schemes.items():
                r_mm = np.array([S.max_range_m(f, cfg, g, g) for g in GAIN_GRID_DBI]) * 1e3
                ax.plot(GAIN_GRID_DBI, r_mm, S.scheme_mark[name], ls=S.scheme_ls[name], ms=3.5, lw=1.8, color=S.scheme_colors[name], label=name)
            d_nf = ch.near_field_limit_m(f, GAIN_GRID_DBI, GAIN_GRID_DBI) * 1e3
            ax.plot(GAIN_GRID_DBI, d_nf, "k:", lw=2, label="Friis-valid limit")
            ax.fill_between(GAIN_GRID_DBI, 1e-3, d_nf, color="grey", alpha=0.2, label="Friis not valid (below limit)")
            ax.set(yscale="log", ylim=(1e-1, 1e2), xlabel="Antenna gain at each end (dBi)", ylabel=f"Max range at BER {S.ber_target:g} (mm)",
                   title=f"{fname}: maximum range vs gain\n(a curve below the dotted line is not physical)")
            ax.grid(True, which="both")
            ax.legend(fontsize=8)
        plt.tight_layout()
    return fig


def plot_max_range_heatmap(S: ThzLinkSetup):
    """12.5: maximum range over (G_tx, G_rx) for heatmap_scheme (hatched = Friis not valid at that range)."""
    with plt.rc_context(NOTEBOOK_RC):
        # In the Friis budget only G_tx + G_rx matters, so contours are anti-diagonal; the map shows the trade-off between ends.
        g_axis = np.arange(0, 41, 2)
        fig, axes = plt.subplots(1, len(S.freqs_hz), figsize=(7.2 * len(S.freqs_hz), 5.6))
        cfg = S.schemes[S.heatmap_scheme]
        for ax, (fname, f) in zip(axes, S.freqs_hz.items()):
            Z = np.array([[S.max_range_m(f, cfg, gt, gr) * 1e3 for gt in g_axis] for gr in g_axis])        # rows = G_rx, cols = G_tx
            GT, GR = np.meshgrid(g_axis, g_axis)
            V = (GT + GR) - ch.fspl_db(Z * 1e-3, f)                  # > 0 : FSPL < G_tx + G_rx at the max range -> not valid
            im = ax.pcolormesh(GT, GR, Z, shading="nearest", cmap="viridis")
            ax.contourf(GT, GR, V, levels=[0, 1e9], colors="none", hatches=["///"])
            ax.contour(GT, GR, V, levels=[0], colors="white", linewidths=1.5)
            fig.colorbar(im, ax=ax, label=f"Max range at BER {S.ber_target:g} (mm)")
            ax.set(xlabel="G_tx (dBi)", ylabel="G_rx (dBi)", title=f"{fname}, {S.heatmap_scheme}\n(hatched = Friis not valid at that range)")
            ax.grid(False)
        plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# All figures
# ----------------------------------------------------------------------------
THZ_LINK_FILES = ("ber_vs_snr.png", "ber_vs_distance.png", "ber_vs_data_rate.png", "ber_vs_received_power.png",
                  "link_characteristics.png", "constellations.png", "eye_diagrams.png", "sim_vs_theory.png",
                  "gain_vs_aperture.png", "snr_power_vs_distance_gains.png", "ber_vs_distance_gains.png",
                  "max_range_vs_gain.png", "max_range_heatmap.png")


def write_thz_link_plots(S: ThzLinkSetup, outdir: str, dpi: int = 100) -> list[str]:
    """Save the 13 THz-link figures of the notebook (sections 7-12) as PNGs in outdir, in the notebook's order
    (the Monte-Carlo figures draw from S.simulate_link's rng in the same order as the notebook).  Returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []

    def save(fig, name):
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(p)

    for sweep, name in zip(SWEEP_ORDER, THZ_LINK_FILES[:4]):
        save(plot_ber_sweep(S, sweep), name)
    save(plot_link_characteristics(S), "link_characteristics.png")
    demo_results = simulate_demo(S)
    save(plot_constellations(S, demo_results), "constellations.png")
    save(plot_eye_diagrams(S, demo_results), "eye_diagrams.png")
    save(plot_sim_vs_theory(S, simulate_validation(S)), "sim_vs_theory.png")
    save(plot_gain_vs_aperture(S), "gain_vs_aperture.png")
    save(plot_snr_power_vs_distance(S), "snr_power_vs_distance_gains.png")
    save(plot_ber_vs_distance_gains(S), "ber_vs_distance_gains.png")
    save(plot_max_range_vs_gain(S), "max_range_vs_gain.png")
    save(plot_max_range_heatmap(S), "max_range_heatmap.png")
    return paths


# ----------------------------------------------------------------------------
# Both links of the integrated scenario: received power vs distance (DWF panel style)
# ----------------------------------------------------------------------------
def _dbm(p_w):
    """Power [W] -> dBm (zeros map to -inf)."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.asarray(p_w, float)) + 30


def plot_received_power_vs_distance(jp, r, link: dict, path: str, title: str = ""):
    """Received power vs distance for the two links of the integrated scenario (one PNG, two panels).

    Left, optical energy link (UOWC, master laser -> slave PV), distance 0.1 mm .. 100 m:
        P_rx(d)     = (delta/tau) * uowc_alpha(d, c, eta_ld, geom) optical power at the PV  (ETA_LD exp(-c d) UOWC_GEOM)
        P_stored(d) = beta * P_rx(d)                              electrical power into the supercap
        delta/tau is taken from the DWF solution r (peak and mean over the slots that share).
    Right, THz data link (slave photomixer -> receiver), distance 0.01 .. 3 mm:
        Pr(d) = link_budget at the photomixer cap (ETA_TX_PM x P_PM_ELEC_W = 20 mW radiated), one curve per preset
        (dashed where Friis is not valid), with the noise floor and the power the BER target needs.
    The operating distances D_UOWC_M and D_THZ_M are marked.  link = slave_load.thz_link() (preset, gains, noise).
    """
    import slave_config as cfg
    from scipy.optimize import brentq
    from .style import C_LINK, C_SLAVE, GRID, INK, INK2, _style

    def legend_below(ax):
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.16), fontsize=8.5, labelcolor=INK2,
                  borderaxespad=0)

    tau = jp.master.tau
    fig, (axo, axt) = plt.subplots(1, 2, figsize=(15.5, 7.4), facecolor="white")
    for ax in (axo, axt):
        _style(ax)

    # ---- optical energy link -------------------------------------------------------------------
    d_o = np.logspace(-4, 2, 400)                                       # 0.1 mm .. 100 m
    alpha_d = ch.uowc_alpha(d_o, cfg.UOWC_C_PER_M, cfg.ETA_LD, cfg.UOWC_GEOM)
    sent = np.asarray(r.delta, float) / tau                             # master electrical sharing power [W]
    on = sent > 1e-12 * max(sent.max(), 1e-300)
    beta = float(np.mean(jp.beta[on])) if on.any() else float(np.mean(jp.beta))
    if on.any():
        for p_tx, ls, name in ((sent[on].max(), "-", "peak"), (sent[on].mean(), "-.", "mean")):
            axo.plot(d_o, _dbm(p_tx * alpha_d), color=C_LINK, lw=2, ls=ls,
                     label=f"optical at the PV, {name} slot ({p_tx * 1e3:.3g} mW sent)")
            axo.plot(d_o, _dbm(beta * p_tx * alpha_d), color=C_LINK, lw=1.2, ls=":" if name == "peak" else (0, (1, 3)),
                     label=f"stored after PV/MPPT/supercap (x beta {beta:.3f}), {name} slot")
        d_op = cfg.D_UOWC_M
        p_op = sent[on].max() * float(ch.uowc_alpha(d_op, cfg.UOWC_C_PER_M, cfg.ETA_LD, cfg.UOWC_GEOM))
        axo.axvline(d_op, color=INK, lw=1, ls="--")
        axo.plot(d_op, _dbm(p_op), "o", color=INK, ms=6,
                 label=f"DWF operating point: D_UOWC_M = {d_op:g} m, {_dbm(p_op):.2f} dBm (peak)")
    else:
        axo.text(0.5, 0.5, "the master shares nothing in this solution", transform=axo.transAxes, ha="center", color=INK2)
    axo.set_xscale("log")
    axo.set_xlabel("master - slave distance [m]  (log scale)")
    axo.set_ylabel("received power [dBm]")
    axo.axvspan(1.0, 5.0, color=GRID, alpha=0.35, lw=0, label="design range of the energy link, 1 - 5 m")
    axo.set_title(f"Optical energy link (UOWC): c = {cfg.UOWC_C_PER_M:g} 1/m, ETA_LD = {cfg.ETA_LD:g}, "
                  f"beam capture UOWC_GEOM = {cfg.UOWC_GEOM:g}",
                  loc="left", color=INK2, fontsize=10)
    axo.text(0.01, 0.02, f"Beer-Lambert x UOWC_GEOM = {cfg.UOWC_GEOM:g} (no beam-spreading model): "
             "optimistic at metres while UOWC_GEOM = 1",
             transform=axo.transAxes, fontsize=8, color=INK2)
    legend_below(axo)

    # ---- THz data link -----------------------------------------------------------------------
    d_t = np.linspace(0.01, 3.0, 400) * 1e-3
    p_rad = link["eta_tx"] * cfg.P_PM_ELEC_W                             # radiated at the photomixer cap [W]
    n_w = link["noise_w"]
    tones = {"0.1 THz": "#7fb8e0", "0.3 THz": C_SLAVE}
    for label, f in cfg.THZ_FREQS_HZ.items():
        g_tx, g_rx = ch.LEGACY_GAIN_DBI if cfg.LEGACY_GAINS else tuple(cfg.GAIN_DBI_BY_FREQ[label])
        b = ch.link_budget(f, d_t, g_tx, g_rx, p_rad, n_w, alpha=cfg.ALPHA_THZ)
        pr = b["pr_dbw"] + 30
        d_nf = float(ch.near_field_limit_m(f, g_tx, g_rx))
        valid = d_t >= d_nf
        col = tones.get(label, C_SLAVE)
        axt.plot(d_t[~valid] * 1e3, pr[~valid], color=col, lw=1.4, ls="--", alpha=0.8)
        axt.plot(d_t[valid] * 1e3, pr[valid], color=col, lw=2,
                 label=f"{label}, {g_tx:g}/{g_rx:g} dBi (dashed: Friis not valid, d < {d_nf * 1e3:.2f} mm)")
    axt.axhline(_dbm(n_w), color=GRID, lw=1.2, ls=":", label=f"noise floor N = {_dbm(n_w):.1f} dBm")
    sch = ch.qam_scheme(cfg.MODULATION)
    k = sch["bits_per_symbol"]
    snr_req_db = brentq(lambda x: float(sch["ber_func"](10 ** (x / 10) / k)) - cfg.BER_TARGET, -10.0, 60.0)
    axt.axhline(_dbm(n_w) + snr_req_db, color=INK2, lw=1.2, ls="-.",
                label=f"needed for BER {cfg.BER_TARGET:g} ({cfg.MODULATION}): {_dbm(n_w) + snr_req_db:.1f} dBm")
    b_op = ch.link_budget(link["freq_hz"], link["d_eff"], link["g_tx_dbi"], link["g_rx_dbi"], p_rad, n_w, alpha=cfg.ALPHA_THZ)
    pr_op = float(b_op["pr_dbw"]) + 30
    axt.axvline(link["d_eff"] * 1e3, color=INK, lw=1, ls="--")
    axt.plot(link["d_eff"] * 1e3, pr_op, "o", color=INK, ms=6,
             label=f"DWF operating point: {link['preset']}, D_THZ_M = {link['d_eff'] * 1e3:g} mm, {pr_op:.1f} dBm "
                   f"(SNR {pr_op - _dbm(n_w):.1f} dB)")
    axt.set_ylim(_dbm(n_w) - 60, _dbm(p_rad) + 30)
    axt.set_xlabel("slave - receiver distance [mm]")
    axt.set_ylabel("received power [dBm]")
    axt.set_title(f"THz data link: {p_rad * 1e3:g} mW radiated (photomixer cap), {_dbm(p_rad):.1f} dBm",
                  loc="left", color=INK2, fontsize=10)
    legend_below(axt)

    head = "Received power vs distance: optical energy link and THz data link" + (f"\n{title}" if title else "")
    fig.suptitle(head, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140)
    plt.close(fig)
