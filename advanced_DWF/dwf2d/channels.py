"""Channel models: Beer-Lambert transmittance of the optical energy link and the THz data-link gain.

Moved verbatim from build_2D_dwf.py (its module docstring is the theory reference).
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
