"""
slave_physical_chain.py
========================
Physical-layer power-conversion model for the optical energy-sharing link,
RECEIVE side only (slave's front end): optical power arriving at the slave
-> GaAs/Ge photodiode ARRAY -> MPPT -> usable electrical power.

Sits ALONGSIDE dwf_master_slave.py; it never modifies the DWF solver.

Chain
-----
    P_received_optical --[split over N identical GaAs/Ge cells in parallel]-->
        per cell:  photocurrent, Voc (literature value), max electrical power
        array total --[MPPT]--> MPPT_output_power   (usable electrical power at the slave)

Formulas (per cell; p_each = P_received_optical / n_cells)
------------------------------------------------------------
    photocurrent_each     = R(wavelength) * p_each
    Voc                   = 1.025 V   (measured, literature value -- see below;
                                        NOT derived from a log(I/I_dark) curve)
    max_electrical_power  = n_cells * FF * Voc * photocurrent_each
    MPPT_output_power     = eta_mppt * max_electrical_power

PART: GaAs/Ge single-junction cell (Spectrolab datasheet) -- the only part modeled
-------------------------------------------------------------------------------------
Chosen over generic Si photodiodes because it's backed by real measured numbers
instead of assumptions: Voc = 1.025 V, FF = 0.82, Jsc = 30.5 mA/cm^2, at AM0
135.3 mW/cm^2, 28 degC. r_450nm/r_532nm are derived from its QE plot via
R[A/W] = QE * wavelength_nm / 1239.8 (QE ~=82% at 450 nm, ~=90% at 532 nm) --
both wavelengths sit on the cell's QE plateau (~80-92%, roughly 420-850 nm).

IMPORTANT SIMPLIFICATION: Voc is NOT derived from a log(I/I_dark) curve -- the
datasheet doesn't publish a dark current, and fitting one from the single Voc/Jsc
data point it does give would need an assumed cell area not given either.
Inventing a dark current just to keep a log-based model would trade one assumption
for another, so Voc is instead the constant 1.025 V measured at the datasheet's
rated irradiance (135.3 mW/cm^2). This is only valid when each cell is actually
operated near that irradiance -- see size_array_by_intensity(), which sizes the
array by matching that irradiance rather than by a per-diode power cap. Real GaAs
Voc does still fall off at lower illumination (much less steeply than Si's, thanks
to its very low dark current); this simplification does not capture that falloff.

ASSUMPTIONS
-----------
    fill_factor      FF = 0.82           measured (Spectrolab datasheet)
    rated_irradiance    = 0.1353 W/cm^2  Spectrolab's AM0 test condition
    cell_area_cm2       = 49.0           7x7 cm, largest size Spectrolab lists --
                                          override if a real cell size is chosen
"""

import dataclasses
from dataclasses import dataclass
import math
import numpy as np

# ---------------------------------------------------------------------------
# Part: GaAs/Ge single-junction cell (Spectrolab datasheet)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhotodiodeSpec:
    name: str
    r_450nm: float                      # responsivity at 450nm, A/W
    r_532nm: float                      # responsivity at 532nm, A/W
    voc_V: float                        # measured open-circuit voltage, V (fixed, literature)
    fill_factor: float                  # measured fill factor
    rated_irradiance_W_per_cm2: float   # illumination voc_V/fill_factor were measured at

    def responsivity(self, wavelength_nm: float) -> float:
        """Look up responsivity R (A/W). Only 450 nm and 532 nm are characterized."""
        if wavelength_nm == 450:
            return self.r_450nm
        if wavelength_nm == 532:
            return self.r_532nm
        raise ValueError(
            f"No responsivity data for {wavelength_nm} nm on {self.name}; "
            "only 450 nm and 532 nm are characterized."
        )


GAAS_GE = PhotodiodeSpec(
    "GaAs/Ge (Spectrolab)",
    r_450nm=0.30,   # derived: QE~=82% x 450/1239.8
    r_532nm=0.39,   # derived: QE~=90% x 532/1239.8
    voc_V=1.025,    # measured Voc, AM0 135.3 mW/cm^2, 28 degC
    fill_factor=0.85,
    rated_irradiance_W_per_cm2=0.1353,
)

DEFAULT_PHOTODIODE = GAAS_GE
DEFAULT_CELL_AREA_CM2 = 49.0   # 7x7 cm, largest size Spectrolab lists -- ASSUMPTION,
                                # override if a real cell size is chosen for the design


def size_array_by_intensity(
    p_peak_optical_W: float,
    spec: PhotodiodeSpec = DEFAULT_PHOTODIODE,
    cell_area_cm2: float = DEFAULT_CELL_AREA_CM2,
) -> int:
    """Smallest number of identical cells so the array's total active area, illuminated
    at spec.rated_irradiance_W_per_cm2, absorbs p_peak_optical_W.

    spec.voc_V is only valid near the irradiance it was measured at, so the design
    goal here is matching that irradiance, not minimizing power-per-cell.

    p_peak_optical_W : largest optical power arriving at the slave in any slot.
    cell_area_cm2     : active area of one cell. ASSUMPTION unless a real size is chosen.
    """
    if cell_area_cm2 <= 0:
        raise ValueError("cell_area_cm2 must be > 0")
    per_cell_W = spec.rated_irradiance_W_per_cm2 * cell_area_cm2
    return max(1, math.ceil(float(p_peak_optical_W) / per_cell_W))


# ---------------------------------------------------------------------------
# Stage 1: Photodiode array (optical -> electrical)
# ---------------------------------------------------------------------------

@dataclass
class PhotodiodeOutput:
    photocurrent: np.ndarray          # A, TOTAL array current
    voc: np.ndarray                   # V, open-circuit voltage (same for every cell)
    max_electrical_power: np.ndarray  # W, total over the array
    p_per_diode_W: np.ndarray         # W, optical power on ONE cell
    n_diodes: int


def photodiode_stage(
    p_received_optical,
    wavelength_nm: float,
    spec: PhotodiodeSpec = DEFAULT_PHOTODIODE,
    n_diodes: int = 1,
) -> PhotodiodeOutput:
    """Convert received optical power into electrical power on an array of n_diodes in parallel."""
    if n_diodes < 1:
        raise ValueError("n_diodes must be >= 1")
    p_opt = np.asarray(p_received_optical, dtype=float)
    p_each = p_opt / n_diodes
    r = spec.responsivity(wavelength_nm)

    i_each = r * p_each
    voc = np.full_like(np.asarray(i_each, dtype=float), spec.voc_V)
    max_electrical_power = n_diodes * spec.fill_factor * voc * i_each

    return PhotodiodeOutput(photocurrent=n_diodes * i_each, voc=voc,
                            max_electrical_power=max_electrical_power,
                            p_per_diode_W=p_each, n_diodes=n_diodes)


# ---------------------------------------------------------------------------
# Stage 2: MPPT
# ---------------------------------------------------------------------------

def mppt_stage(max_electrical_power, mppt_efficiency: float):
    """MPPT_output_power = eta_mppt * max_electrical_power."""
    if not (0.0 < mppt_efficiency <= 1.0):
        raise ValueError("mppt_efficiency must be in (0, 1]")
    return mppt_efficiency * np.asarray(max_electrical_power, dtype=float)


# ---------------------------------------------------------------------------
# Full optical -> electrical receive chain
# ---------------------------------------------------------------------------

@dataclass
class OpticalReceiveChainResult:
    p_received_optical: np.ndarray
    photocurrent: np.ndarray
    voc: np.ndarray
    max_electrical_power: np.ndarray
    mppt_output_power: np.ndarray
    fill_factor: float
    p_per_diode_W: np.ndarray = None
    n_diodes: int = 1


def optical_to_electrical(
    p_received_optical,
    wavelength_nm: float,
    mppt_efficiency: float,
    fill_factor: float = None,
    spec: PhotodiodeSpec = DEFAULT_PHOTODIODE,
    n_diodes: int = 1,
    temp_K: float = 300.0,
) -> OpticalReceiveChainResult:
    """Run photodiode array -> MPPT. Returns every intermediate quantity.

    temp_K : accepted for interface compatibility with callers (build_2D_dwf.py's
        physical_stored_power()) but unused -- this model's Voc is the datasheet's fixed,
        measured literature value (see module docstring), not derived from a log(I/I_dark)
        curve, so it has no temperature dependence to apply here.
    """
    if fill_factor is not None:
        spec = dataclasses.replace(spec, fill_factor=fill_factor)
    pd = photodiode_stage(p_received_optical, wavelength_nm, spec, n_diodes)
    mppt_out = mppt_stage(pd.max_electrical_power, mppt_efficiency)
    return OpticalReceiveChainResult(
        p_received_optical=np.asarray(p_received_optical, dtype=float),
        photocurrent=pd.photocurrent,
        voc=pd.voc,
        max_electrical_power=pd.max_electrical_power,
        mppt_output_power=mppt_out,
        fill_factor=fill_factor if fill_factor is not None else spec.fill_factor,
        p_per_diode_W=pd.p_per_diode_W,
        n_diodes=pd.n_diodes,
    )


# if __name__ == "__main__":
#     # Demo: GaAs/Ge array sized to the datasheet's rated irradiance, at a 35 W peak
#     # arriving optical power, over a couple of cell-area choices and both wavelengths.
#     p_peak = 35.0
#     print(f"peak received optical power = {p_peak} W, GaAs/Ge (Spectrolab), MPPT 0.95, FF {GAAS_GE.fill_factor}")
#     print(f"{'cell_area':>10} {'n_cells':>8} {'W/cm2':>7} {'wl[nm]':>7} {'Voc [V]':>8} {'beta':>7} {'out [W]':>8}")
#     for area_cm2 in (10.0, 49.0):
#         n = size_array_by_intensity(p_peak, cell_area_cm2=area_cm2)
#         irradiance = p_peak / (n * area_cm2)
#         for wl in (450, 532):
#             c = optical_to_electrical(p_peak, wl, 0.95, n_diodes=n)
#             beta = float(c.mppt_output_power) / p_peak
#             print(f"{area_cm2:>10g} {n:>8d} {irradiance:>7.4f} {wl:>7d} "
#                   f"{float(c.voc):>8.3f} {beta:>7.4f} {float(c.mppt_output_power):>8.3f}")