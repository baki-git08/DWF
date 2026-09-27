"""
slave_physical_chain.py
========================
Physical-layer power-conversion model for the optical energy-sharing link,
RECEIVE side only (slave's front end): optical power arriving at the slave
-> photodiode -> MPPT -> usable electrical power.

This is a physical-realizability layer that sits ALONGSIDE dwf_master_slave.py.
It does NOT modify, call, or get called by the DWF solver. It takes a
received-optical-power value/array (whatever the Beer-Lambert channel model
in dwf_master_slave.py would produce, i.e. P_transmitted * alpha) and reports
what a real photodiode + MPPT receive chain would actually deliver.

Chain implemented here (scope approved so far -- optical to electrical only):

    P_received_optical --[photodiode: DP/SB series]--> photocurrent, Voc, max_electrical_power
                        --[MPPT]--> MPPT_output_power   (usable electrical power at the slave)

Formulas (as specified)
------------------------
    photocurrent          = R(wavelength) * P_received_optical
    Voc                   = V_T * ln(1 + photocurrent / I_dark)
    max_electrical_power  = FF * Voc * photocurrent
    MPPT_output_power     = eta_mppt * max_electrical_power

NOT yet implemented in this file (pending confirmation on fill factor /
composition -- see the plan discussion): LED driver + LED electro-optic
stage (transmit side), RTD THz transmitter stage, rectenna THz receiver
stage. Add them the same way once those are settled.

Assumptions used here (change the defaults below if you have better numbers):
    fill_factor (FF) = 0.7   -- typical silicon photodiode; not on the
                                datasheet snippets supplied, so this is a
                                placeholder default, passed as an argument
                                everywhere so it's easy to override.
    temperature      = 300 K -> thermal voltage V_T = kT/q ~= 0.025852 V
"""

from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Photodiode part library
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhotodiodeSpec:
    name: str
    r_450nm: float          # responsivity at 450nm, A/W
    r_532nm: float          # responsivity at 532nm, A/W
    dark_current_A: float   # dark current, A

    def responsivity(self, wavelength_nm: float) -> float:
        """Look up responsivity R (A/W) for a supported wavelength.

        Only 450nm and 532nm are characterized from the supplied datasheet
        values, so any other wavelength raises rather than silently
        interpolating/extrapolating a responsivity we don't actually have.
        """
        if wavelength_nm == 450:
            return self.r_450nm
        if wavelength_nm == 532:
            return self.r_532nm
        raise ValueError(
            f"No responsivity data for {wavelength_nm} nm on {self.name}; "
            "only 450 nm and 532 nm are characterized."
        )


# Dark current ranges are collapsed to a single representative value
# (midpoint of the given range) since the formulas need one number.
PHOTODIODES = {
    "ODD-5WBISOLDS": PhotodiodeSpec("ODD-5WBISOLDS", r_450nm=0.20, r_532nm=0.35, dark_current_A=2e-9),    # 1-3 nA -> 2 nA
    "DP/SB":         PhotodiodeSpec("DP/SB",         r_450nm=0.32, r_532nm=0.40, dark_current_A=10e-9),   # 10 nA
    "C30954EH":      PhotodiodeSpec("C30954EH",      r_450nm=30.0, r_532nm=35.0, dark_current_A=150e-9),  # 100-200 nA -> 150 nA
}

# Chosen for the optical -> electrical chain, as instructed.
DEFAULT_PHOTODIODE = PHOTODIODES["DP/SB"]

BOLTZMANN_K = 1.380649e-23     # J/K
ELECTRON_Q = 1.602176634e-19   # C


def thermal_voltage(temp_K: float = 300.0) -> float:
    """V_T = kT/q, the thermal voltage used in the photodiode's Voc formula."""
    return BOLTZMANN_K * temp_K / ELECTRON_Q


# ---------------------------------------------------------------------------
# Stage 1: Photodiode (optical -> electrical)
# ---------------------------------------------------------------------------

@dataclass
class PhotodiodeOutput:
    photocurrent: np.ndarray          # A
    voc: np.ndarray                   # V, open-circuit voltage
    max_electrical_power: np.ndarray  # W


def photodiode_stage(
    p_received_optical,
    wavelength_nm: float,
    spec: PhotodiodeSpec = DEFAULT_PHOTODIODE,
    fill_factor: float = 0.7,
    temp_K: float = 300.0,
) -> PhotodiodeOutput:
    """Convert received optical power into electrical power at the photodiode.

        photocurrent          = R(wavelength) * P_received_optical   [A]
        Voc                    = V_T * ln(1 + photocurrent / I_dark)  [V]
        max_electrical_power   = FF * Voc * photocurrent              [W]

    Parameters
    ----------
    p_received_optical : scalar or array-like, W.
        Optical power arriving at the slave's photodiode -- e.g. the output
        of the Beer-Lambert channel model in build_2D_dwf.py
        (P_transmitted * alpha).
    wavelength_nm : 450 or 532.
        Must match a characterized responsivity on the chosen part.
    spec : PhotodiodeSpec, default DP/SB series (as chosen).
    fill_factor : FF, default 0.7. Not given on the datasheet snippets
        supplied -- this is an assumption; override if you have a real value
        for this part.
    temp_K : operating temperature for the thermal-voltage calc, default 300K.
    """
    p_opt = np.asarray(p_received_optical, dtype=float)
    r = spec.responsivity(wavelength_nm)
    v_t = thermal_voltage(temp_K)

    photocurrent = r * p_opt
    voc = v_t * np.log1p(photocurrent / spec.dark_current_A)  # ln(1 + x) via log1p for accuracy at small x
    max_electrical_power = fill_factor * voc * photocurrent

    return PhotodiodeOutput(photocurrent=photocurrent, voc=voc, max_electrical_power=max_electrical_power)


# ---------------------------------------------------------------------------
# Stage 2: MPPT (maximum power point tracker)
# ---------------------------------------------------------------------------

def mppt_stage(max_electrical_power, mppt_efficiency: float):
    """MPPT_output_power = eta_mppt * max_electrical_power.

    mppt_efficiency : fraction in (0, 1]. No value was supplied for this
        system's MPPT -- typical MPPT ICs run 0.85-0.98 -- so pass whichever
        you want to use.
    """
    if not (0.0 < mppt_efficiency <= 1.0):
        raise ValueError("mppt_efficiency must be in (0, 1]")
    return mppt_efficiency * np.asarray(max_electrical_power, dtype=float)


# ---------------------------------------------------------------------------
# Convenience: full optical -> electrical receive chain
# ---------------------------------------------------------------------------

@dataclass
class OpticalReceiveChainResult:
    p_received_optical: np.ndarray
    photocurrent: np.ndarray
    voc: np.ndarray
    max_electrical_power: np.ndarray
    mppt_output_power: np.ndarray


def optical_to_electrical(
    p_received_optical,
    wavelength_nm: float,
    mppt_efficiency: float,
    spec: PhotodiodeSpec = DEFAULT_PHOTODIODE,
    fill_factor: float = 0.7,
    temp_K: float = 300.0,
) -> OpticalReceiveChainResult:
    """Run the full received-side optical energy chain: photodiode -> MPPT.

    Returns every intermediate quantity (photocurrent, Voc, max_electrical_power)
    plus the final usable electrical power (mppt_output_power) so each stage's
    loss can be inspected, not just the end result.
    """
    pd = photodiode_stage(p_received_optical, wavelength_nm, spec, fill_factor, temp_K)
    mppt_out = mppt_stage(pd.max_electrical_power, mppt_efficiency)
    return OpticalReceiveChainResult(
        p_received_optical=np.asarray(p_received_optical, dtype=float),
        photocurrent=pd.photocurrent,
        voc=pd.voc,
        max_electrical_power=pd.max_electrical_power,
        mppt_output_power=mppt_out,
    )

