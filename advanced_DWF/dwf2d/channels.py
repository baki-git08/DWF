"""Channel models: Beer-Lambert transmittance of the optical energy link and the THz data-link gain.

Moved verbatim from build_2D_dwf.py (its module docstring is the theory reference).

Below the legacy models (beer_lambert, thz_absorption, thz_gain; unchanged) this module also holds
  * the photomixing THz link budget of Modulation_THz_slave_profile.ipynb (sections 2 and 4):
    FSPL + absorption - antenna gains, thermal noise, the Friis near-field limit, and thz_hbar(), the
    slave's electrical-burst-power -> SNR gain used by the DWF;
  * uowc_alpha(), the UOWC energy-link transmittance including the laser efficiency;
  * the modulation engine of the notebook (sections 3 and 5): Gray-coded PAM/QAM constellations, their
    theoretical BER, the RRC pulse and the bit-level Monte-Carlo link (make_link_simulator).
They were moved out of the notebook verbatim; former notebook globals are parameters whose defaults are
the notebook values, so the notebook's tables are reproduced exactly (LEGACY_GAINS, see preset_gains).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
def beer_lambert(c: float | np.ndarray, d: float) -> np.ndarray:
    """Beer-Lambert transmittance of the underwater energy link.

    Equation:   alpha = exp(-c * d)

    Parameters
    ----------
    c : attenuation coefficient [1/m]  (scalar, or one value per slot for a time-varying water)
    d : distance between master and slave [m]

    Returns alpha in (0, 1]: the fraction of the sent POWER that reaches the slave's receiver
    (before the battery-recharging efficiency beta is applied).
    """
    return np.exp(-np.asarray(c, dtype=float) * d)


# ----------------------------------------------------------------------------
# THz data link (master <-> slave communication, both directions)
# ----------------------------------------------------------------------------
# Molecular-absorption coefficient of the THz path, measured at each frequency below.
# thz_alpha_cm is given in 1/cm; thz_alpha_m = thz_alpha_cm * 100 converts it to 1/m.
thz_freq = np.array([0.30, 0.45, 0.48, 0.60, 0.66, 0.90, 0.99, 1.05,
                      1.26, 1.29, 1.31, 1.40, 1.53, 1.56, 1.83, 1.98,
                      2.13, 2.52, 2.84, 3.09, 3.42, 3.72])          # [THz]

thz_alpha_cm = np.array([123, 148, 156, 167, 180, 210, 220, 237, 256,
                          269, 271, 275, 294, 295, 326, 347, 367, 433,
                          502, 549, 622, 739])                       # [1/cm]

thz_alpha_m = thz_alpha_cm * 100.0                                   # [1/m]


def thz_absorption(freq_thz: float | np.ndarray) -> np.ndarray:
    """Molecular-absorption coefficient alpha_m [1/m] of the THz path at freq_thz [THz].

    Linear interpolation over the measured (thz_freq, thz_alpha_m) table. At a table
    frequency (e.g. 0.30) this returns the measured value exactly (here 12300 1/m).
    """
    return np.interp(np.asarray(freq_thz, dtype=float), thz_freq, thz_alpha_m)


def thz_gain(freq_thz: float | np.ndarray, d: float, h0: float = 10.0) -> np.ndarray:
    """THz channel gain over noise (the h used inside ln(1 + h*P)).

    Equation:   h = h0 * exp(-alpha_m(freq_thz) * d)

    h0 folds in everything the absorption coefficient alone does not capture
    (antenna/beam gains, transmit-to-noise reference, free-space spreading, ...).
    Left at 1.0 (same convention as the MasterParams.h default) unless calibrated.
    """
    return h0 * np.exp(-thz_absorption(freq_thz) * d)


# ----------------------------------------------------------------------------
# THz photomixer link budget (moved from Modulation_THz_slave_profile.ipynb, sections 2 and 4)
# ----------------------------------------------------------------------------
#   Pr [dBW] = Pt + G_tx + G_rx - FSPL(f, d) - 10 log10(e) * alpha * d
#   FSPL     = 20 log10(4 pi d f / c)
#   coupling = FSPL + absorption - G_tx - G_rx            (total loss Tx input -> Rx output, dB)
#   SNR      = Pr / N ,  N = k_B T F R_s                  (thermal noise in the symbol bandwidth)
C_LIGHT = 3e8                     # speed of light (m/s), same value as the notebook
K_B = 1.380649e-23                # Boltzmann constant (J/K)
ALPHA_THZ = 12300.0               # molecular extinction coefficient (1/m), power law exp(alpha*d)
PT_W = 20e-3                      # photomixer radiated power (W)
APERTURE_A_M = 0.020              # horn aperture, 2 cm
APERTURE_B_M = 0.015              # horn aperture, 1.5 cm
APERTURE_EFFICIENCY = 0.51        # aperture efficiency

# Antenna gains (G_tx, G_rx) [dBi] per carrier frequency [Hz].  LEGACY_GAIN_DBI reproduces the saved
# notebook outputs (5 dBi at both ends at BOTH frequencies).
GAIN_DBI_BY_FREQ = {0.1e12: (5.0, 5.0), 0.3e12: (10.0, 10.0)}
LEGACY_GAIN_DBI = (5.0, 5.0)


def preset_gains(freq_hz: float, legacy: bool = False) -> tuple[float, float]:
    """(G_tx, G_rx) [dBi] of the preset for carrier freq_hz; legacy=True gives 5/5 dBi at every frequency."""
    if legacy:
        return LEGACY_GAIN_DBI
    for f, g in GAIN_DBI_BY_FREQ.items():
        if np.isclose(freq_hz, f, rtol=1e-9, atol=0.0):
            return g
    raise ValueError(f"no antenna-gain preset for {freq_hz:g} Hz (known: {sorted(GAIN_DBI_BY_FREQ)})")


def wavelength(freq_hz):
    """Free-space wavelength (m)."""
    return C_LIGHT / freq_hz


def aperture_gain_dbi(freq_hz, a_m=APERTURE_A_M, b_m=APERTURE_B_M, eta=APERTURE_EFFICIENCY):
    """Gain (dBi) of an a x b aperture antenna: G = 4*pi*eta*A / lambda^2."""
    lam = wavelength(freq_hz)
    return 10 * np.log10(4 * np.pi * eta * a_m * b_m / lam ** 2)


def far_field_distance_m(freq_hz, a_m=APERTURE_A_M, b_m=APERTURE_B_M):
    """Classical far-field distance 2*D^2/lambda, D = aperture diagonal."""
    D = np.hypot(a_m, b_m)
    return 2 * D ** 2 / wavelength(freq_hz)


def near_field_limit_m(freq_hz, g_tx_dbi, g_rx_dbi):
    """Distance where FSPL = G_tx + G_rx. Closer than this, the Friis budget gives Pr > Pt (unphysical).

    d_nf = lambda/(4 pi) * 10^((G_tx + G_rx)/20)   (0.7958 mm at 0.3 THz / 10+10 dBi, 0.7549 mm at 0.1 THz / 5+5 dBi)
    """
    return wavelength(freq_hz) / (4 * np.pi) * 10 ** ((np.asarray(g_tx_dbi) + np.asarray(g_rx_dbi)) / 20)


def noise_psd_w_hz(t_noise_k: float = 290.0, noise_figure_db: float = 10.0) -> float:
    """Thermal noise PSD N0 = k_B * T * F  [W/Hz]  (F = 10^(NF/10))."""
    return K_B * t_noise_k * 10 ** (noise_figure_db / 10)


def noise_floor_w(mode: str = "thermal", t_noise_k: float = 290.0, noise_figure_db: float = 10.0,
                  r_s: float = 1e9, notebook_floor_w: float = 1e-3) -> float:
    """Noise power in the symbol bandwidth [W]: "thermal" N0*R_s (= 4.0039e-11 W at 290 K, 10 dB, 1 Gsps),
    "notebook" the original fixed 1 mW floor."""
    if mode == "thermal":
        return noise_psd_w_hz(t_noise_k, noise_figure_db) * r_s
    if mode == "notebook":
        return notebook_floor_w
    raise ValueError(f"NOISE_MODE must be 'thermal' or 'notebook', got {mode!r}")


def fspl_db(d_m, freq_hz):
    """Free-space path loss (dB)."""
    d_m = np.maximum(np.asarray(d_m, dtype=float), 1e-12)
    return 20 * np.log10(4 * np.pi * d_m * freq_hz / C_LIGHT)


def absorption_db(d_m, alpha=ALPHA_THZ):
    """Molecular/absorption loss (dB) from exp(alpha*d) on power."""
    return 10 * np.log10(np.e) * alpha * np.asarray(d_m, dtype=float)


def link_budget(freq_hz, d_m, g_tx_dbi=None, g_rx_dbi=None, p_tx_w=None, noise_floor_w=None,
                *, alpha=ALPHA_THZ, legacy_gains=False):
    """Full link budget at distance d (scalar or array).

    Gains left as None come from the per-frequency preset (preset_gains; legacy_gains=True -> 5/5 dBi),
    p_tx_w None -> PT_W, noise_floor_w None -> the thermal floor of noise_floor_w().
    """
    g_tx, g_rx = (None, None) if (g_tx_dbi is not None and g_rx_dbi is not None) else preset_gains(freq_hz, legacy_gains)
    g_tx = g_tx if g_tx_dbi is None else g_tx_dbi
    g_rx = g_rx if g_rx_dbi is None else g_rx_dbi
    p_tx = PT_W if p_tx_w is None else p_tx_w
    nf   = _noise_floor_default() if noise_floor_w is None else noise_floor_w
    fs, ab = fspl_db(d_m, freq_hz), absorption_db(d_m, alpha)
    coupling = fs + ab - g_tx - g_rx
    pr_dbw = 10 * np.log10(p_tx) - coupling
    snr_db = pr_dbw - 10 * np.log10(nf)
    return {"fspl_db": fs, "absorption_db": ab, "coupling_loss_db": coupling,
            "pr_dbw": pr_dbw, "pr_w": 10 ** (pr_dbw / 10), "snr_db": snr_db, "snr": 10 ** (snr_db / 10),
            "friis_invalid": fs < (g_tx + g_rx)}


_noise_floor_default = noise_floor_w      # link_budget's parameter of the same name shadows the function


def solve_distance_for_snr(freq_hz, target_snr_db, g_tx_dbi=None, g_rx_dbi=None, **budget_kw):
    """Distance (m) at which the budget gives the target SNR (SNR falls monotonically with d).
    budget_kw (p_tx_w, noise_floor_w, alpha, legacy_gains) are passed to link_budget."""
    d = np.logspace(-7, -1.5, 20000)
    snr = link_budget(freq_hz, d, g_tx_dbi, g_rx_dbi, **budget_kw)["snr_db"]
    if snr[0] < target_snr_db or snr[-1] > target_snr_db:
        return np.nan
    return float(np.interp(target_snr_db, snr[::-1], d[::-1]))


FINE_D_M = np.logspace(-6, -2, 4000)      # 1 um .. 10 mm search grid for the maximum range


def ber_at(freq_hz, scheme, d_m, g_tx_dbi, g_rx_dbi, **budget_kw):
    """Theoretical BER at distance d (array ok) for given antenna gains.
    budget_kw (p_tx_w, noise_floor_w, alpha, legacy_gains) are passed to link_budget."""
    snr = link_budget(freq_hz, d_m, g_tx_dbi, g_rx_dbi, **budget_kw)["snr"]
    return scheme["ber_func"](snr / scheme["bits_per_symbol"])


def max_range_m(freq_hz, scheme, g_tx_dbi, g_rx_dbi, target=1e-3, d_grid=FINE_D_M, **budget_kw):
    """Largest distance with theoretical BER <= target (BER rises monotonically with distance)."""
    ok = np.where(ber_at(freq_hz, scheme, d_grid, g_tx_dbi, g_rx_dbi, **budget_kw) <= target)[0]
    return float(d_grid[ok[-1]]) if len(ok) else 0.0


def thz_hbar(freq_hz: float, d_m: float, g_tx_dbi=None, g_rx_dbi=None, eta_tx: float = 1.0,
             noise_w: float | None = None, alpha: float = ALPHA_THZ, legacy_gains: bool = False,
             warn: bool = True) -> dict:
    """Gain over noise of the slave's THz burst, per watt of ELECTRICAL photomixer power [1/W].

        coupling  = fspl_db + abs_db - G_tx - G_rx                      [dB]
        hbar      = eta_tx / (10^(coupling/10) * N)                     [1/W]
        SNR_burst = hbar * P_pk ,   P_pk = x / t_tx                     (x = burst energy, J)

    Friis is only valid for d >= d_nf = near_field_limit_m(f, G_tx, G_rx); closer than that the budget would
    give Pr > Pt, so the distance is clamped to d_eff = max(d, d_nf) (one warning line, clamped = True).
    eta_tx = radiated / electrical power of the photomixer; hbar already contains it (scale h_bar ONCE).
    Returns dict(hbar, d_eff, d_nf, clamped, coupling_db, g_tx_dbi, g_rx_dbi, noise_w).
    """
    g_tx, g_rx = preset_gains(freq_hz, legacy_gains)
    g_tx = g_tx if g_tx_dbi is None else g_tx_dbi
    g_rx = g_rx if g_rx_dbi is None else g_rx_dbi
    n_w = noise_floor_w() if noise_w is None else noise_w
    d_nf = float(near_field_limit_m(freq_hz, g_tx, g_rx))
    clamped = bool(d_m < d_nf)
    d_eff = max(float(d_m), d_nf)
    if clamped and warn:
        print(f"thz_hbar: d = {d_m * 1e3:.4g} mm is inside the Friis-invalid region (d_nf = {d_nf * 1e3:.4g} mm at "
              f"{freq_hz / 1e12:g} THz, {g_tx:g}+{g_rx:g} dBi); using d_eff = d_nf")
    coupling = float(fspl_db(d_eff, freq_hz) + absorption_db(d_eff, alpha) - g_tx - g_rx)
    hbar = eta_tx / (10 ** (coupling / 10) * n_w)
    return dict(hbar=hbar, d_eff=d_eff, d_nf=d_nf, clamped=clamped, coupling_db=coupling,
                g_tx_dbi=g_tx, g_rx_dbi=g_rx, noise_w=n_w)


# ----------------------------------------------------------------------------
# UOWC energy link (master laser -> slave PV)
# ----------------------------------------------------------------------------
def uowc_alpha(d: float, c: float | np.ndarray, eta_ld: float = 1.0, geom: float = 1.0) -> np.ndarray:
    """Fraction of the master's electrical sharing energy that arrives as optical energy at the slave PV.

    Equation:   alpha = eta_ld * exp(-c * d) * geom
        eta_ld : laser (electrical -> optical) efficiency [DUMMY until measured]
        c      : water attenuation [1/m] (0.046 at 450 nm),  d : distance [m]
        geom   : beam/aperture capture fraction (1 = no beam model)
    With eta_ld = geom = 1 this is beer_lambert(c, d).
    """
    return eta_ld * beer_lambert(c, d) * geom


# ----------------------------------------------------------------------------
# Modulation engine (moved from the notebook, sections 3 and 5)
# ----------------------------------------------------------------------------
def gray_pam(n_bits):
    """PAM levels (..., -3, -1, +1, +3, ...) indexed by their Gray-coded bit label."""
    L = 2 ** n_bits
    positions = np.arange(L)
    levels = np.arange(-(L - 1), L, 2)
    levels_by_label = np.empty(L)
    levels_by_label[positions ^ (positions >> 1)] = levels
    return levels_by_label


def gray_qam(bits_i, bits_q):
    """Rectangular QAM, Gray-coded I/Q axes, label = (I bits | Q bits), unit average power."""
    i, q = gray_pam(bits_i), gray_pam(bits_q)
    pts = (i[:, None] + 1j * q[None, :]).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def qam_ber_theory(bits_i, bits_q):
    """Gray-coded rectangular QAM: BER as a function of Eb/N0 (linear)."""
    from scipy.special import erfc
    L_i, L_q, k = 2 ** bits_i, 2 ** bits_q, bits_i + bits_q
    coeff = ((1 - 1 / L_i) + (1 - 1 / L_q)) / k
    arg = 3 * k / ((L_i ** 2 - 1) + (L_q ** 2 - 1))
    return lambda ebn0: coeff * erfc(np.sqrt(arg * np.asarray(ebn0)))


def pam_ber_theory(bits):
    """Gray-coded real (bipolar) M-PAM: BER as a function of Eb/N0 (linear)."""
    from scipy.special import erfc
    L = 2 ** bits
    coeff, arg = (1 - 1 / L) / bits, 3 * bits / (L ** 2 - 1)
    return lambda ebn0: coeff * erfc(np.sqrt(arg * np.asarray(ebn0)))


def make_scheme(name, k, points, ber_func):
    points = np.asarray(points)
    assert len(points) == 2 ** k, f"{name}: needs 2^{k} constellation points"
    return {"name": name, "bits_per_symbol": k, "points": points,
            "is_one_dim": not np.iscomplexobj(points),
            "avg_power": float(np.mean(np.abs(points) ** 2)),
            "ber_func": ber_func}


QAM_SPLIT = {"4-QAM": (1, 1), "8-QAM": (2, 1), "16-QAM": (2, 2), "64-QAM": (3, 3)}   # (I bits, Q bits)


def qam_scheme(name: str) -> dict:
    """The notebook's scheme entry for a QAM name ("16-QAM" -> 4 bits, gray_qam(2, 2), qam_ber_theory(2, 2))."""
    if name not in QAM_SPLIT:
        raise ValueError(f"unknown modulation {name!r} (known: {sorted(QAM_SPLIT)})")
    bi, bq = QAM_SPLIT[name]
    return make_scheme(name, bi + bq, gray_qam(bi, bq), qam_ber_theory(bi, bq))


def rrc_pulse(beta, span, sps):
    """Unit-energy root-raised-cosine pulse, length span*sps + 1."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps          # time in symbol periods
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif np.isclose(abs(4 * beta * ti), 1.0):
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                   / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.linalg.norm(h)


def min_distance_decode(rx_sym, pts, chunk=50_000):
    """Nearest constellation point for every received sample (chunked to bound memory)."""
    idx = np.empty(len(rx_sym), dtype=np.int64)
    for s in range(0, len(rx_sym), chunk):
        idx[s:s + chunk] = np.abs(rx_sym[s:s + chunk, None] - pts[None, :]).argmin(axis=1)
    return idx


def make_link_simulator(rolloff, span, sps, rng, n_symbols=10_000):
    """Factory for the notebook's simulate_link: the RRC pulse, its delay and the rng are bound here instead
    of being module globals.  The returned function also carries .PULSE, .DELAY and .sps (eye diagrams).

    simulate_link(scheme, snr_db, n_symbols, rng): Monte-Carlo link at symbol-domain SNR snr_db (Es/N0):
    bits -> Gray-labelled symbols -> upsampling + RRC -> AWGN -> matched filter -> sampling -> min-distance
    decoding.  Returns dict(rx_sym, rx_mf, ber, bit_errors, snr_db).  Draws from `rng` in the same order as
    the notebook, so a seeded run is reproduced exactly.
    """
    from scipy.signal import upfirdn
    PULSE = rrc_pulse(rolloff, span, sps)
    DELAY = span * sps                    # TX + RX group delay (samples)

    def simulate_link(scheme, snr_db, n_symbols=n_symbols, rng=rng):
        """Monte-Carlo link at symbol-domain SNR `snr_db` (Es/N0). Returns dict(rx_sym, rx_mf, ber, ...)."""
        k, pts = scheme["bits_per_symbol"], scheme["points"]

        bits = rng.integers(0, 2, (n_symbols, k))
        labels = bits @ (1 << np.arange(k - 1, -1, -1))
        tx = upfirdn(PULSE, pts[labels], up=sps)

        # AWGN: N0 set from the average symbol power; N0/2 per real dimension
        N0 = scheme["avg_power"] / 10 ** (snr_db / 10)
        sigma = np.sqrt(N0 / 2)
        noise = sigma * rng.standard_normal(len(tx))
        if not scheme["is_one_dim"]:
            noise = noise + 1j * sigma * rng.standard_normal(len(tx))
        rx = tx + noise

        rx_mf = upfirdn(PULSE, rx)                                     # matched filter, full rate (eye diagrams)
        rx_sym = rx_mf[DELAY: DELAY + n_symbols * sps: sps]            # symbol-rate samples

        idx = min_distance_decode(rx_sym, pts)
        rx_bits = (idx[:, None] >> np.arange(k - 1, -1, -1)) & 1
        errors = int(np.count_nonzero(rx_bits != bits))
        return {"rx_sym": rx_sym, "rx_mf": rx_mf, "ber": errors / bits.size, "bit_errors": errors, "snr_db": snr_db}

    simulate_link.PULSE, simulate_link.DELAY, simulate_link.sps = PULSE, DELAY, sps
    return simulate_link
