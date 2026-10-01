"""
dwf_master_slave.py
===================
Directional water-filling (DWF) for a MASTER node that harvests energy and shares
part of it with a SLAVE node that harvests nothing (underwater, Beer-Lambert link).

Step 2 of the project: builds on dwf_master.py (master alone). Nothing in
dwf_master.py is modified; this module only imports its data and MasterParams.

READING ORDER (suggested)
-------------------------
    1. This docstring (model, notation, constraints, algorithm)
    2. step()            - the ONE equation everything rests on
    3. joint_dwf() moves (1), (6), (7)  - the plain equations
    4. check_kkt_joint() - how optimality is tested
    5. moves (2)-(5), (8), (9) and polish() - the same equations, used to get past
       "corners" where a single variable cannot move on its own

1. MODEL  (slots i = 0..N-1, slot length tau [s], all energies in Joules)
--------------------------------------------------------------------------
Master : harvests E_i at the start of slot i, spends
             tau*P_i   on its own data (power P_i)      and
             delta_i   on energy it sends to the slave  (delta_i >= 0)
Slave  : harvests nothing. Of the delta_i sent, only
             a_i * delta_i    lands in its battery,     a_i = beta * alpha_i
         where alpha_i = exp(-c*d) is the Beer-Lambert transmittance of the water path
         and beta is the battery-recharging efficiency (two losses in series).
         It spends tau*Pbar_i on data sent back to the master (power Pbar_i).

Objective (weighted sum-rate, nats):

    max   sum_i (th1*tau/2) ln(1 + h_i * P_i)   +   sum_i (th2*tau/2) ln(1 + hbar_i * Pbar_i)

    h_i     master link gain over noise         hbar_i  slave return-link gain over noise
    th1,th2 rate weights (1 = plain sum-rate)

"Received power" is reported in noise-normalised units (N0 = 1 W): the received data power
of a link is h*P, which is exactly the SNR appearing inside ln(1 + h*P).

2. NOTATION: math <-> code  (0-indexed slots)
---------------------------------------------
    E_i            E[i]        harvested energy in slot i
    F_k            f[k+1]      master energy carried from slot k to slot k+1   (right tap k)
    E_max - E_{k+1} capM[k]    capacity of master right tap k
    delta_k        d[k]        energy sent over the DOWN tap of slot k
    a_k = beta*alpha_k  a[k]   net gain of the down tap
    b_k            B_min+g[k+1] slave battery level at the END of slot k
    Gt_k = b_k - B_min  g[k+1] slave carry ABOVE the floor (right tap k of the slave)
    b_0 - B_min    g[0]        initial slave excess (b0 = initial charge)
    W = B_max-B_min W          usable slave window
    tau*P_i        xm(i)       master data energy in slot i
    tau*Pbar_i     xs(i)       slave  data energy in slot i

Energy balance (this is how xm and xs are DEFINED in the code):

    tau*P_i    = E_i + F_{i-1} - F_i - delta_i              (master: harvest + carry in - carry out - shared)
    tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i               (slave : arrival  + carry in - carry out)

with F_{-1} = F_{N-1} = 0 (nothing before the first / after the last slot) and
Gt_{-1} = b0 - B_min, Gt_{N-1} = 0 (all energy is used by the end).

3. CONSTRAINTS AND THEIR MULTIPLIERS
------------------------------------
Because (f, g, d) are the free variables, the energy-balance equalities hold by
construction, and the original inequality constraints become simple BOUNDS:

    (M1) master causality   F_n >= 0                   f >= 0                       lambda_n
    (M2) master capacity    F_n <= E_max - E_{n+1}     f[k+1] <= capM[k]            mu_n
    (M3) non-negativity     P_i >= 0 , delta_i >= 0    xm >= 0 , d >= 0             eta_i , rho_i
    (S1) slave floor        b_n >= B_min               g >= 0                       lambda_bar_n
    (S2) slave ceiling      b_{n-1} + a_n*delta_n <= B_max   g[k] + a[k]*d[k] <= W  mu_bar_{n-1}
         (checked at ARRIVAL: within a slot the level only falls after the arrival)
    (S3) slave power        Pbar_i >= 0                xs >= 0                      eta_bar_i
    (S4) schedule           Pbar_i = 0 if s_i = 0      xs = 0 in silent slots       (JointParams.slave_schedule)
         A silent slot still receives (delta_i) and stores energy; it is a pure pass-through for
         the slave battery. Moves whose slave end is a silent slot are skipped in joint_dwf().

Whenever a move is CLIPPED by one of these bounds, that constraint is active and its
multiplier is >= 0 (complementary slackness). The multipliers are never computed
explicitly; optimality is recognised by "no move improves the objective".

4. STATIONARITY (where the moves come from)
-------------------------------------------
Marginal value of one Joule of data energy (derivative of the rate w.r.t. energy):

    m_k = th1*h_k / (2*(1 + h_k*P_k))          master,  equals Lambda_k     when P_k > 0
    s_k = th2*hbar_k / (2*(1 + hbar_k*Pbar_k)) slave,   equals Lambda_bar_k when Pbar_k > 0

    Water-filling form:  P_k = [ nu_k - 1/h_k ]^+ ,  nu_k = th1/(2*Lambda_k)   (level = P + 1/h)
    Down tap (delta_k):  Lambda_k = a_k * Gamma_k ,  Gamma_k = Lambda_bar_k - mu_bar_{k-1}
                         => interior (arrival cap slack):  m_k = a_k * s_k

    Right taps:  lambda_n = mu_n = 0  =>  nu_n = nu_{n+1}  (equal water levels)
                 F_n = 0 (closed)     =>  nu_{n+1} >= nu_n   (level can only rise)
                 F_n = cap (full)     =>  nu_{n+1} <= nu_n   (level can only fall)

5. ALGORITHM (block coordinate ascent over "taps")
--------------------------------------------------
Start at "spend what you harvest, share nothing" (f = 0, g = 0, d = 0). Repeatedly sweep
the slots forward and backward; at each slot try a list of MOVES. A move changes one or
two flow variables by t Joules, and t is chosen as

        t = clip( t_star , lo , hi )

where t_star is the exact maximiser of the (concave) objective along that move and [lo, hi]
is the interval on which all constraints of section 3 still hold. A move can therefore never
reduce the objective. The sweep stops when no move exceeds tol*scale; then polish() tries
longer moves; when those also do nothing the point satisfies the KKT conditions.

Moves (k = current slot):
    (1) down tap          master k  -> slave k          delta_k up                 m_k = a_k s_k
    (6) master right tap  master k  -> master k+1       F_k up                     nu_k = nu_{k+1}
    (7) slave right tap   slave k   -> slave k+1        Gt_k up                    nubar_k = nubar_{k+1}
    (2)-(5),(8),(9)       same equations, but two variables move together so that energy can
                          cross a slot that has ZERO power (pass-through) or so that the slave's
                          arrival cap can be respected (swap). They exist because moving ONE
                          variable at a time can get stuck at such corners.
    polish()              the same idea over RUNS of consecutive zero-power slots.

6. KNOWN LIMITS
---------------
* The two powers (P, Pbar) are unique; the split of carried energy between the master battery
  and the slave battery is NOT unique when both have slack (same objective, different F and b).
* On extreme random settings (per-slot random h, hbar, alpha, E, weights, tight batteries) the
  solver was within about 0.1% of the optimum in 3 of 25 tests. reference_solution_joint()
  (cvxpy) can confirm any configuration.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from build_dwf import Harvested_energy_summer, MasterParams, Harvested_energy_winter
from dwf_diagnostics import diagnose
from physical_chain import optical_to_electrical, OpticalReceiveChainResult


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


@dataclass
class JointParams:
    """All inputs of the master + slave problem.  Values marked PLACEHOLDER are not from your data.

    Attributes
    ----------
    master : MasterParams
        Master-only data from dwf_master.py: harvest E_i, master capacity E_max (np.inf allowed),
        slot length tau, master link gain h_i.
    alpha : Beer-Lambert transmittance alpha_i (scalar or per-slot).   [PLACEHOLDER: exp(-0.2*5)]
    beta : battery-recharging efficiency in [0, 1] (scalar or per-slot).  [PLACEHOLDER 0.8]
        Only a_i*delta_i = beta*alpha_i*delta_i of the shared energy ends up in the slave battery.
        Per-slot beta lets solve_with_physical_chain() calibrate it against the real photodiode+
        MPPT chain (physical_chain.py) without changing joint_dwf() itself.
    B_min, B_max : slave battery floor / ceiling [J]  (given: 200 and 5000).
        Battery level always satisfies  B_min <= b_n  and  b_{n-1} + a_n*delta_n <= B_max.
    b0 : initial slave battery level [J]. Default (None) = B_min, i.e. the reserve is already
        in the battery and the slave starts with zero usable energy.  [ASSUMPTION]
    h_bar : slave -> master data-link gain over noise hbar_i (scalar or per-slot).  [PLACEHOLDER 0.1]
    theta1, theta2 : weights of the master and slave rate in the objective.
    slave_schedule : transmission schedule s_i in {0, 1} (array/list). s_i = 0 forces the
        slave's data power to zero in slot i (constraint S4); it still receives and stores energy.
        A pattern shorter than N repeats cyclically ([1, 0] = transmit every other slot).
        None (default) = always allowed. Stored as a boolean array of length N.
    """
    master: MasterParams
    alpha: np.ndarray | float = 0.368     # PLACEHOLDER  exp(-0.2 1/m * 5 m)
    beta: np.ndarray | float = 0.8        # PLACEHOLDER  recharging efficiency (scalar or per-slot)
    B_min: float = 200.0                  # given
    B_max: float = 5000.0                 # given
    b0: float | None = None               # initial slave charge; default = B_min (ASSUMPTION)
    h_bar: np.ndarray | float = 0.1       # PLACEHOLDER  slave -> master link gain (SNR per W)
    theta1: float = 1.0                   # weight of master rate
    theta2: float = 3.0                   # weight of slave rate
    slave_schedule: np.ndarray | list | None = None   # 1 = slave may transmit, 0 = silent (None = always 1)

    def __post_init__(self):
        """Broadcast scalars to one value per slot and validate the model assumptions."""
        n = self.master.N
        # alpha_i, beta_i and hbar_i may be given as scalars; the solver needs one value per slot.
        self.alpha = np.broadcast_to(np.asarray(self.alpha, float), (n,)).copy()
        self.beta = np.broadcast_to(np.asarray(self.beta, float), (n,)).copy()
        self.h_bar = np.broadcast_to(np.asarray(self.h_bar, float), (n,)).copy()
        # Default initial charge: start exactly at the floor (no usable energy yet).
        if self.b0 is None:
            self.b0 = self.B_min
        # Feasibility of the battery window:  0 <= B_min < B_max  and  B_min <= b0 <= B_max.
        if not (0 <= self.B_min < self.B_max):
            raise ValueError("need 0 <= B_min < B_max")
        if not (self.B_min <= self.b0 <= self.B_max):
            raise ValueError("need B_min <= b0 <= B_max")
        # alpha and beta are fractions (losses), hbar must give a valid log-rate.
        if np.any((self.alpha < 0) | (self.alpha > 1)) or np.any((self.beta < 0) | (self.beta > 1)):
            raise ValueError("alpha and beta must be in [0, 1]")
        if np.any(self.h_bar <= 0):
            raise ValueError("h_bar must be > 0")
        # Transmission schedule s_i: None -> always on; a shorter pattern (e.g. [1, 0]) repeats cyclically.
        if self.slave_schedule is None:
            sched = np.ones(n)
        else:
            sched = np.asarray(self.slave_schedule, dtype=float).ravel()
            if sched.size == 0 or not np.all((sched == 0) | (sched == 1)):
                raise ValueError("slave_schedule must be a non-empty array of 0s and 1s")
            sched = np.resize(sched, n)
        self.slave_schedule = sched.astype(bool)
        # With no ON slot the slave could never spend any energy it holds above the floor.
        if not self.slave_schedule.any() and self.b0 > self.B_min:
            raise ValueError("slave_schedule has no transmit slot but b0 > B_min")

    @property
    def a(self) -> np.ndarray:            # net gain of the down tap
        """Net gain of the master->slave energy tap:  a_i = beta * alpha_i  (< 1).

        One Joule sent by the master (delta_i) becomes a_i Joules in the slave battery.
        """
        return self.beta * self.alpha   # <- 

    @property
    def W(self) -> float:                 # usable battery window
        """Usable slave battery window  W = B_max - B_min  [J].

        After subtracting the floor, the slave carry Gt = b - B_min lives in [0, W - a*delta_next].
        """
        return self.B_max - self.B_min

    @property
    def g_init(self) -> float:            # initial energy above the floor
        """Initial slave energy above the floor  b0 - B_min  [J]  (this is g[0] in the solver)."""
        return self.b0 - self.B_min


@dataclass
class JointResult:
    """Everything joint_dwf() returns.  Arrays have one entry per slot.

    Powers are in W, energies in J.  For the battery arrays, index n means "end of slot n"
    unless stated otherwise.
    """
    P: np.ndarray             # master data power                    [W]     P_i = xm(i)/tau
    Pbar: np.ndarray          # slave  data power                    [W]     Pbar_i = xs(i)/tau
    delta: np.ndarray         # energy sent over the tap             [J]     delta_i
    F: np.ndarray             # master battery after slot n          [J]     F_n = sum_{i<=n}(E_i - tau*P_i - delta_i)
    b: np.ndarray             # slave  battery after slot n          [J]     b_n = B_min + Gt_n
    b_peak: np.ndarray        # slave battery right after arrival k  [J]     b_{k-1} + a_k*delta_k  (must be <= B_max)
    sent_power: np.ndarray    # delta/tau                            [W]     power the master sends
    arrived_power: np.ndarray # alpha*delta/tau  (after the channel) [W]     power that reaches the slave receiver
    stored_power: np.ndarray  # beta*alpha*delta/tau (after recharge)[W]     power that ends up in the battery
    rx_master: np.ndarray     # h*P     received data power (N0=1)          SNR of the master link
    rx_slave: np.ndarray      # hbar*Pbar                                   SNR of the slave link
    rate_master: float        # sum tau/2 ln(1+hP)        [nats]
    rate_slave: float         # sum tau/2 ln(1+hbar*Pbar) [nats]
    objective: float          # th1*rate_master + th2*rate_slave
    iters: int                # number of sweeps used
    converged: bool           # True if the stop rule (and polish) was met before max_iter


@dataclass
class PhysicalStorageResult:
    """Physical-layer reality check for the down-tap, computed AFTER joint_dwf() has solved.

    joint_dwf() assumes a constant linear tap gain a_i = beta*alpha_i (module docstring,
    section 1): "of the delta_i sent, only a_i*delta_i lands in the slave battery". This result
    reports what the actual photodiode + MPPT receive chain (physical_chain.py) would deliver
    for that SAME delta schedule, without feeding anything back into the solver.

    Arrays have one entry per slot.
    """
    chain: OpticalReceiveChainResult   # photocurrent, Voc, max_electrical_power, mppt_output_power
    stored_power_phys: np.ndarray      # [W] chain.mppt_output_power -- physically real stored power
    stored_energy_phys: np.ndarray     # [J] tau * stored_power_phys
    eff_phys: np.ndarray               # mppt_output_power / arrived_power (NaN where arrived_power = 0)
    b_phys: np.ndarray                 # [J] slave battery re-simulated with the physical arrivals
    b_peak_phys: np.ndarray            # [J] battery level right after arrival (physical)
    overflow: np.ndarray               # [J] per slot energy above B_max that a real battery would reject
    overflow_energy: float             # [J] sum(overflow); ~0 once beta is calibrated to the chain
    floor_violations: int              # slots where b_phys < B_min
    ceiling_violations: int            # slots where b_peak_phys > B_max


def physical_stored_power(
    jp: "JointParams",
    r: JointResult,
    wavelength_nm: float = 532,
    mppt_efficiency: float = 0.95,
    fill_factor: float = 1.0,
    temp_K: float = 300.0,
    battery_tol_J: float = 1e-6,
) -> PhysicalStorageResult:
    """Run the solver's delta schedule through the real photodiode+MPPT chain.

    Takes r.arrived_power (alpha*delta/tau, the optical power actually reaching the slave --
    exactly the "P_transmitted*alpha" input physical_chain.py's optical_to_electrical() expects)
    and reports the power that chain would really deliver into the battery
    (chain.mppt_output_power), alongside a re-simulation of the slave battery using that
    physical arrival instead of the solver's linear a_i*delta_i assumption.

    Pure post-processing: does not modify jp or r, and never feeds back into joint_dwf().

    battery_tol_J : numerical tolerance [J] used for the overflow clip and the floor/ceiling
        violation counts below. joint_dwf() only converges to within `tol * scale` (its own `tol`
        argument, default 1e-10 * mean harvested energy per slot) -- typically ~1e-7 to 1e-6 J at
        the energy scales this model runs at -- so slots that sit exactly ON a bound (common: S1/
        S2 are frequently active) will show a residual of about that size even at an exact beta
        calibration. This must stay looser than that residual, or every such slot reads as a false
        "violation" instead of a real infeasibility.
    """
    m = jp.master
    tau = m.tau
    chain = optical_to_electrical(
        r.arrived_power, wavelength_nm, mppt_efficiency,
        fill_factor=fill_factor, temp_K=temp_K,
    )
    stored_power_phys = chain.mppt_output_power
    stored_energy_phys = tau * stored_power_phys

    with np.errstate(divide="ignore", invalid="ignore"):
        eff_phys = np.where(r.arrived_power > 0, stored_power_phys / np.where(r.arrived_power > 0, r.arrived_power, 1.0), np.nan)

    # Re-simulate the slave battery: same spending (tau*Pbar) as the solver's solution, but the
    # PHYSICAL arrival instead of a_i*delta_i.  Starts at b0 (same initial charge as the solver).
    tau_pbar = r.Pbar * tau                                       # tau*Pbar_i, the slave's actual spend (unchanged)
    b_prev_phys = np.empty(m.N)
    b_phys = np.empty(m.N)
    overflow = np.zeros(m.N)
    level = jp.b0                                                 # running battery level, starts at b0
    for i in range(m.N):
        b_prev_phys[i] = level
        level = level + stored_energy_phys[i] - tau_pbar[i]
        if level > jp.B_max + battery_tol_J:                      # a real battery can't hold more than B_max
            overflow[i] = level - jp.B_max
            level = jp.B_max
        b_phys[i] = level
    b_peak_phys = b_prev_phys + stored_energy_phys

    return PhysicalStorageResult(
        chain=chain,
        stored_power_phys=stored_power_phys,
        stored_energy_phys=stored_energy_phys,
        eff_phys=eff_phys,
        b_phys=b_phys,
        b_peak_phys=b_peak_phys,
        overflow=overflow,
        overflow_energy=float(overflow.sum()),
        floor_violations=int(np.sum(b_phys < jp.B_min - battery_tol_J)),
        ceiling_violations=int(np.sum(b_peak_phys > jp.B_max + battery_tol_J)),
    )


def solve_with_physical_chain(
    jp: JointParams,
    wavelength_nm: float = 532,
    mppt_efficiency: float = 0.95,
    fill_factor: float = 1.0,
    temp_K: float = 300.0,
    tol: float = 1e-3,
    max_outer: int = 30,
):
    """Fixed-point calibration: re-solve joint_dwf() with a per-slot beta equal to the real
    photodiode+MPPT chain's secant gain, until beta stops changing. joint_dwf() itself is never
    modified -- see Instructions.md, Option B, for the full derivation.

    Why this works: the down tap's real gain is eff_i = mppt_output_power_i / arrived_power_i.
    It is not perfectly constant (the chain is nonlinear), but it varies only slightly with the
    operating point here, so feeding it back in as beta_i and re-solving converges in a handful
    of iterations. At the fixed point, a_i*delta_i/tau == mppt_output_power_i exactly, so the
    solver's battery (r.b, r.b_peak) matches the physically re-simulated one (ph.b_phys,
    ph.b_peak_phys), and the existing B_min/B_max constraints are enforced against real energy.

    Slots with delta_i == 0 have nothing to calibrate on (arrived_power_i == 0 -> eff_phys_i is
    NaN); their beta is left unchanged from the previous iterate.

    Returns (jp_calibrated, r, ph, n_outer):
        jp_calibrated : JointParams with the converged per-slot beta
        r             : final JointResult from joint_dwf(jp_calibrated)
        ph            : final PhysicalStorageResult (ph.b_phys should equal r.b to `tol`)
        n_outer       : number of outer iterations used
    """
    beta = np.broadcast_to(np.asarray(jp.beta, float), (jp.master.N,)).copy()
    jp_i = jp
    r = None
    ph = None
    for n_outer in range(1, max_outer + 1):
        jp_i = dataclasses.replace(jp, beta=beta.copy())
        r = joint_dwf(jp_i)
        ph = physical_stored_power(jp_i, r, wavelength_nm, mppt_efficiency, fill_factor, temp_K)
        on = r.delta > 1e-9 * max(jp.master.E.mean(), 1.0)         # only calibrate where energy actually moves
        beta_new = beta.copy()
        beta_new[on] = ph.eff_phys[on]
        moved = float(np.max(np.abs(beta_new - beta))) if np.any(on) else 0.0
        beta = beta_new
        if moved < tol:
            break
    return jp_i, r, ph, n_outer


# ----------------------------------------------------------------------------
# Core solver
# ----------------------------------------------------------------------------
def joint_dwf(jp: JointParams, tol: float = 1e-10, max_iter: int = 100_000) -> JointResult:
    """Solve the master + slave problem by directional water-filling (block coordinate ascent).

    Problem solved (see module docstring, sections 1-3):

        max  sum_i (th1*tau/2) ln(1+h_i P_i) + (th2*tau/2) ln(1+hbar_i Pbar_i)
        s.t. M1-M3 (master), S1-S3 (slave)

    Variables (all Joules): master carry f, slave carry g, shared energy d.
    Data energies follow from the energy balance:  xm = tau*P,  xs = tau*Pbar.

    Procedure
    ---------
    1. Start from f = 0, g = 0 (g[0] = initial slave excess), d = 0:
       "spend what you harvest, share nothing".
    2. Sweep the slots (forward on even sweeps, backward on odd ones).  At every slot try the
       moves (1)-(9).  Each move is  t = clip(t_star, lo, hi)  where t_star comes from step()
       (marginal rates balance) and [lo, hi] keeps every constraint of section 3 satisfied.
    3. When a whole sweep moves less than tol*scale, call polish() (longer pass-through paths).
       If polish() also moves less than tol*scale, the point satisfies the KKT conditions and
       the solver has converged.
    4. Convert the final flows to powers, batteries and rates.

    Parameters
    ----------
    jp : JointParams
    tol : relative stop tolerance (moves are compared with tol * scale, scale = mean(E) in J)
    max_iter : safety cap on the number of sweeps (one sweep = one pass over all slots)

    Returns
    -------
    JointResult
    """
    m = jp.master
    N, tau = m.N, m.tau
    E, h, hb, a = m.E, m.h, jp.h_bar, jp.a
    th1, th2, W = jp.theta1, jp.theta2, jp.W
    capM = m.cap                                  # E_max - E_{n+1}, n = 0..N-2
    tx = jp.slave_schedule                        # tx[i] False -> Pbar_i = 0 forced (S4); slave only stores

    # ---- state: the three families of flow variables (all in Joules) -------------------------
    f = np.zeros(N + 1)                           # f[k+1] = master carry over tap k
    #   f[k+1] = F_k = master energy carried from slot k to k+1;  f[0] = f[N] = 0 stay fixed
    #   (nothing arrives before slot 0 and nothing is left after slot N-1).
    g = np.zeros(N + 1)                           # g[k+1] = slave carry above floor
    #   g[k+1] = Gt_k = b_k - B_min;  g[0] = b0 - B_min (initial excess), g[N] = 0 stays fixed
    #   (the slave also uses all its energy by the end).
    g[0] = jp.g_init
    #   Feasible start under S4: if the first slots are silent, the initial excess cannot be spent
    #   there, so it is carried (unchanged) up to the first transmit slot.
    if tx.any():
        g[1:int(np.argmax(tx)) + 1] = g[0]
    d = np.zeros(N)                               # delta
    #   d[k] = delta_k = energy the master sends over the down tap in slot k.

    # ---- energy balance: these two lambdas DEFINE the data energies -----------------------------
    xm = lambda i: E[i] + f[i] - f[i + 1] - d[i]          # master data energy
    #   tau*P_i = E_i + F_{i-1} - F_i - delta_i    (harvest + carry in - carry out - shared)
    xs = lambda i: a[i] * d[i] + g[i] - g[i + 1]          # slave  data energy
    #   tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i (arrival + carry in - carry out)

    scale = max(float(E.mean()), 1.0)             # typical energy per slot [J]; makes tol relative

    def step(thu, hu, xu, s_, thw, hw, xw, r_):
        """Joules t moved from node u (loses s_*t) to node w (gains r_*t) that equalise
        the marginal rates  th_u h_u/(1+h_u P_u) = r_ th_w h_w/(1+h_w P_w).

        This is the ONE equation the solver rests on.  Derivation
        ----------------------------------------------------------
        The objective along a move is  U_u(x_u - s*t) + U_w(x_w + r*t)  with
        U(x) = (th*tau/2) ln(1 + h*x/tau),  so its derivative in t is
            -s * th_u h_u / (2(1 + h_u (x_u - s t)/tau))  +  r * th_w h_w / (2(1 + h_w (x_w + r t)/tau)).
        Setting it to zero (marginal value lost = marginal value gained) and cross-multiplying gives
            th_u h_u (1 + h_w (x_w + r t)/tau) = r th_w h_w (1 + h_u (x_u - s t)/tau),
        which is LINEAR in t, so
            t* = tau * [ r th_w h_w (1 + h_u x_u/tau) - th_u h_u (1 + h_w x_w/tau) ]
                       / [ h_u h_w ( th_u r + r th_w s ) ].
        (num and den below are exactly the numerator and denominator of this expression.)

        Special cases
        -------------
        * Right taps (s = r = 1, same theta):  t* = 0.5*tau*(nu_u - nu_w) with nu = P + 1/h,
          i.e. the water levels are equalised (lambda = mu = 0).
        * Down tap (u = master k, w = slave k, s = 1, r = a_k):  t* solves
          th1 h/(1+hP) = a_k th2 hbar/(1+hbar Pbar), the interior condition Lambda_k = a_k Gamma_k.

        Parameters (per node): th = rate weight, h = link gain, x = current data energy [J],
        s_ / r_ = how many Joules of x the node loses / gains per Joule moved.
        Returns t* (positive = from u to w), UNCLIPPED. Returns 0 if both ends are worthless.
        """
        den = hu * hw * (thu * r_ + r_ * thw * s_)
        if den <= 0:                       # both ends worthless (e.g. weight 0): nothing to gain
            return 0.0
        num = r_ * thw * hw * (1 + hu * xu / tau) - thu * hu * (1 + hw * xw / tau)
        return tau * num / den

    dead = lambda xv: xv <= 1e-6 * scale         # slot spending (almost) nothing
    #   "dead" slot = a slot with P = 0 (or Pbar = 0): its water level sits on the floor 1/h and
    #   it cannot supply energy, so single-variable moves cannot push energy THROUGH it.

    def apply(fch, gch, dch, src, snk, t_star):
        """Move t along a path: f[q]+=c*t, g[q]+=c*t, d[q]+=c*t. src/snk = (x_value, coef)
        of the two end nodes. t = t_star clipped to keep every constraint satisfied.

        What it does
        ------------
        A "path" is a list of coordinate changes.  Every constraint of the problem is LINEAR in
        the flows, so after the move each affected quantity is  q(t) = v + c*t  and must satisfy
        L <= q(t) <= U.  Each such requirement gives one bound on t; the tightest ones form the
        feasible interval [lo, hi].  The move is then  t = clip(t_star, lo, hi).

        Constraints checked (equation numbers from the module docstring)
        ---------------------------------------------------------------
        * f[q]              in [0, capM[q-1]]            M1, M2  (master right tap q-1)
        * g[q]              >= 0                         S1      (slave floor)
        * d[q]              >= 0                         M3      (delta >= 0)
        * g[q] + a[q]*d[q]  <= W                         S2      (slave arrival cap at slot q)
        * src / snk data energy >= 0                     M3, S3  (P >= 0, Pbar >= 0)

        Parameters
        ----------
        fch, gch, dch : lists of (index, coefficient c) for the f, g, d variables that move
        src, snk : (current data energy x, coefficient) of the SOURCE (loses) and SINK (gains) node
        t_star : unclipped optimal step from step()

        Returns |t| actually applied (used by the caller to detect convergence).
        """
        lo, hi = -np.inf, np.inf

        def lim(v, c, L, U):
            """Bound on t so that  L <= v + c*t <= U  (c > 0 bounds t from above by U, from below by L)."""
            nonlocal lo, hi
            if c > 0:
                hi = min(hi, (U - v) / c); lo = max(lo, (L - v) / c)
            elif c < 0:
                lo = max(lo, (U - v) / c); hi = min(hi, (L - v) / c)
        arr = {}                                # slave arrival-cap coefficient per slot q
        for q, c in fch:
            lim(f[q], c, 0.0, capM[q - 1])
        for q, c in gch:
            lim(g[q], c, 0.0, np.inf); arr[q] = arr.get(q, 0.0) + c
        for q, c in dch:
            lim(d[q], c, 0.0, np.inf); arr[q] = arr.get(q, 0.0) + a[q] * c
        for q, c in arr.items():
            lim(g[q] + a[q] * d[q], c, -np.inf, W)          # slave arrival cap
        lim(src[0], src[1], 0.0, np.inf)
        lim(snk[0], snk[1], 0.0, np.inf)
        if lo > hi:
            return 0.0                          # numerically infeasible interval: do nothing
        t = min(max(t_star, lo), hi)            # projection of t* on [lo, hi] (concave => optimal)
        for q, c in fch: f[q] += c * t
        for q, c in gch: g[q] += c * t
        for q, c in dch: d[q] += c * t
        return abs(t)

    def polish():
        """Longer paths that cross RUNS of slots with zero spending (pass-through).
        Single-tap moves cannot cross such a run; these moves can. Never worsen the objective.

        Why it is needed
        ----------------
        In a "dead" slot (P = 0 or Pbar = 0) the water level sits on the floor 1/h.  If energy
        should travel from slot i to a later slot j THROUGH one or more such slots, no single tap
        looks profitable (each neighbour sees a worse level) although the end-to-end move is.
        This routine tries those end-to-end moves; the intermediate slots keep their spending
        (pass-through), only the two END nodes change:

            source node loses t      sink node gains  s*t   (s = 1 for carries, a_k over the down tap)

        and t is chosen by step() on the two end nodes, then clipped by apply().

        Three families of paths are tried
        ---------------------------------
        A. master chain   master slot i -> master slot j   (through dead master slots)   f[i+1..j] += t
        B. slave chain    slave slot i  -> slave slot j    (through dead slave slots)    g[i+1..j] += t
        C. cross paths    master slot i -> master chain -> down tap k -> slave chain -> slave slot j
                          (delta_k += t, master carries and slave carries adjusted so that every
                          intermediate slot keeps its spending)

        Returns the largest |t| applied, so the caller knows whether anything moved.
        """
        big = 0.0
        for i in range(N):                                          # master chain i -> j
            j = i + 2                                               # needs >= 1 intermediate slot
            while j <= N - 1 and dead(xm(j - 1)):                   # extend while the intermediate is dead
                t = step(th1, h[i], xm(i), 1, th1, h[j], xm(j), 1)  # equalise the two END levels nu_i, nu_j
                big = max(big, apply([(q, 1.0) for q in range(i + 1, j + 1)], [], [],
                                     (xm(i), -1.0), (xm(j), 1.0), t))
                j += 1
        for i in range(N):                                          # slave chain i -> j
            j = i + 2
            while j <= N - 1 and dead(xs(j - 1)):
                if tx[i] and tx[j]:                                 # silent end slots keep Pbar = 0 (S4)
                    t = step(th2, hb[i], xs(i), 1, th2, hb[j], xs(j), 1)  # equalise slave levels nubar_i, nubar_j
                    big = max(big, apply([], [(q, 1.0) for q in range(i + 1, j + 1)], [],
                                         (xs(i), -1.0), (xs(j), 1.0), t))
                j += 1
        for k in range(N):                                          # master -> down tap k -> slave
            if a[k] <= 0:
                continue                                            # no energy can land: skip
            mo = [(k, [])]                                          # (source slot i, f changes)
            i = k - 1                                               # sources BEFORE k: f[i+1..k] += t
            while i >= 0:
                mo.append((i, [(q, 1.0) for q in range(i + 1, k + 1)]))
                if not dead(xm(i)):
                    break                                           # cannot pass through a live slot
                i -= 1
            i = k + 1                                               # sources AFTER k: undo carry f[k+1..i] -= t
            while i <= N - 1:
                mo.append((i, [(q, -1.0) for q in range(k + 1, i + 1)]))
                if not dead(xm(i)):
                    break
                i += 1
            so = [(k, [])] if tx[k] else []                         # (sink slot j, g changes); silent sinks excluded (S4)
            j = k + 1                                               # sinks AFTER k: g[k+1..j] += a_k*t
            while j <= N - 1:
                if tx[j]:
                    so.append((j, [(q, a[k]) for q in range(k + 1, j + 1)]))
                if not dead(xs(j)):
                    break
                j += 1
            j = k - 1                                               # sinks BEFORE k: g[j+1..k] -= a_k*t
            while j >= 0:
                if tx[j]:
                    so.append((j, [(q, -a[k]) for q in range(j + 1, k + 1)]))
                if not dead(xs(j)):
                    break
                j -= 1
            for i, fch in mo:
                for j, gch in so:
                    if len(fch) + len(gch) < 2:
                        continue                                    # plain moves are in the fast sweep
                    # balance: th1 h_i/(1+h_i P_i)  =  a_k th2 hbar_j/(1+hbar_j Pbar_j)
                    t = step(th1, h[i], xm(i), 1, th2, hb[j], xs(j), a[k])
                    big = max(big, apply(fch, gch, [(k, 1.0)],
                                         (xm(i), -1.0), (xs(j), a[k]), t))
        return big

    # ---- main loop: sweeps over the slots -------------------------------------------------------
    it, converged = 0, False
    while it < max_iter:
        big = 0.0                                                   # largest move in this sweep [J]
        order = range(N) if it % 2 == 0 else range(N - 1, -1, -1)   # alternate direction each sweep
        for k in order:
            ak = a[k]
            # (1) DOWN TAP  M_k -> S_k : delta_k += t
            #     effect  : master data energy xm(k) -= t ;  slave data energy xs(k) += a_k*t
            #     equation: m_k = a_k * s_k   (Lambda_k = a_k Gamma_k, arrival cap slack)
            #     bounds  : delta_k + t >= 0        -> lo  (rho_k)
            #               xs(k) + a_k t >= 0      -> lo  (eta_bar_k)
            #               xm(k) - t >= 0          -> hi  (eta_k)
            #               g[k] + a_k(d[k]+t) <= W -> hi  (mu_bar_{k-1}, slave arrival cap)
            if ak > 0 and tx[k]:
                x, xb = xm(k), xs(k)
                t = step(th1, h[k], x, 1, th2, hb[k], xb, ak)
                t = min(max(t, max(-d[k], -xb / ak)), min(x, (W - g[k]) / ak - d[k]))
                d[k] += t; big = max(big, abs(t))
            # (2) IN-SWAP  M_k -> S_{k-1}: delta_k += t, carry g_{k-1} -= a_k t
            #     (needed when the slave arrival cap g_{k-1}+a_k delta_k <= W is tight)
            #     effect  : xm(k) -= t ; slave slot k unchanged (arrival +a_k t, carry-in -a_k t);
            #               slave slot k-1 spends a_k*t more (its carry-out g[k] fell by a_k t)
            #     equation: m_k = a_k * s_{k-1}   (value of the joule at the slave's EARLIER slot)
            #     bounds  : delta_k >= 0 and xs(k-1) >= 0 -> lo ;  xm(k) >= 0 and g[k] >= 0 -> hi
            #     the arrival cap g[k] + a_k delta_k is unchanged, so it never blocks this move
            if ak > 0 and k >= 1 and tx[k - 1]:
                x, xb1 = xm(k), xs(k - 1)
                t = step(th1, h[k], x, 1, th2, hb[k - 1], xb1, ak)
                t = min(max(t, max(-d[k], -xb1 / ak)), min(x, g[k] / ak))
                d[k] += t; g[k] -= ak * t; big = max(big, abs(t))
            # (3) MASTER pass-through  M_{k-1} -> S_k  (f_{k-1}, delta_k up together)
            #     lets energy cross a master slot that has P_k = 0 on its way to the slave
            #     effect  : f[k] += t, d[k] += t ;  xm(k-1) -= t ; master slot k unchanged
            #               (carry-in +t, shared +t) ;  xs(k) += a_k*t
            #     equation: m_{k-1} = a_k * s_k
            #     bounds  : lo: f[k] >= 0, delta_k >= 0, xs(k) >= 0
            #               hi: f[k] <= capM[k-1] (M2), xm(k-1) >= 0, slave arrival cap
            if ak > 0 and k >= 1 and tx[k]:
                xa, xb = xm(k - 1), xs(k)
                t = step(th1, h[k - 1], xa, 1, th2, hb[k], xb, ak)
                lo = max(-f[k], -d[k], -xb / ak)
                hi = min(capM[k - 1] - f[k], xa, (W - g[k]) / ak - d[k])
                t = min(max(t, lo), hi)
                f[k] += t; d[k] += t; big = max(big, abs(t))
            # (4) MASTER out-swap  M_{k+1} -> S_k  (carry f_k down, delta_k up)
            #     effect  : f[k+1] -= t (undo carry), d[k] += t ; xm(k) unchanged ;
            #               xm(k+1) -= t ; xs(k) += a_k*t
            #     equation: m_{k+1} = a_k * s_k
            #     bounds  : lo: delta_k >= 0, xs(k) >= 0, f[k+1] <= capM[k] (when t < 0)
            #               hi: f[k+1] >= 0 (cannot undo more carry than exists), xm(k+1) >= 0, arrival cap
            if ak > 0 and k <= N - 2 and tx[k]:
                xa, xb = xm(k + 1), xs(k)
                t = step(th1, h[k + 1], xa, 1, th2, hb[k], xb, ak)
                lo = max(-d[k], -xb / ak, -(capM[k] - f[k + 1]))
                hi = min(f[k + 1], xa, (W - g[k]) / ak - d[k])
                t = min(max(t, lo), hi)
                f[k + 1] -= t; d[k] += t; big = max(big, abs(t))
            # (5) SLAVE pass-through  M_k -> S_{k+1} across slave slot k
            #     effect  : d[k] += t, g[k+1] += a_k t ; xm(k) -= t ; slave slot k unchanged
            #               (arrival +a_k t, carry-out +a_k t) ; xs(k+1) += a_k*t
            #     equation: m_k = a_k * s_{k+1}
            #     bounds  : lo: delta_k >= 0, g[k+1] >= 0, xs(k+1) >= 0
            #               hi: xm(k) >= 0, arrival cap at slot k AND at slot k+1
            if ak > 0 and k <= N - 2 and tx[k + 1]:
                xa, xb = xm(k), xs(k + 1)
                t = step(th1, h[k], xa, 1, th2, hb[k + 1], xb, ak)
                lo = max(-d[k], -g[k + 1] / ak, -xb / ak)
                hi = min(xa, (W - g[k]) / ak - d[k], (W - a[k + 1] * d[k + 1] - g[k + 1]) / ak)
                t = min(max(t, lo), hi)
                d[k] += t; g[k + 1] += ak * t; big = max(big, abs(t))
            if k == N - 1:
                continue                     # the last slot has no right tap: moves (6)-(9) need k+1
            # (6) MASTER right tap k -> k+1 : equalise master levels
            #     effect  : f[k+1] += t ; xm(k) -= t ; xm(k+1) += t
            #     equation: nu_k = nu_{k+1}  with nu = P + 1/h  (lambda_k = mu_k = 0)
            #     bounds  : f >= 0 (closed tap, lambda_k) and f <= capM[k] (full tap, mu_k),
            #               plus xm(k) >= t and xm(k+1) >= -t (no negative power)
            xa, xb_ = xm(k), xm(k + 1)
            t = step(th1, h[k], xa, 1, th1, h[k + 1], xb_, 1)
            t = min(max(t, max(-f[k + 1], -xb_)), min(capM[k] - f[k + 1], xa))
            f[k + 1] += t; big = max(big, abs(t))
            # (7) SLAVE right tap k -> k+1 : equalise slave levels
            #     effect  : g[k+1] += t ; xs(k) -= t ; xs(k+1) += t
            #     equation: nubar_k = nubar_{k+1}  with nubar = Pbar + 1/hbar
            #     bounds  : g >= 0 (slave at the floor, lambda_bar_k) and
            #               g[k+1] + a_{k+1} delta_{k+1} <= W (arrival cap of the NEXT slot, mu_bar_k)
            if tx[k] and tx[k + 1]:
                xa, xb_ = xs(k), xs(k + 1)
                t = step(th2, hb[k], xa, 1, th2, hb[k + 1], xb_, 1)
                t = min(max(t, max(-g[k + 1], -xb_)), min(W - a[k + 1] * d[k + 1] - g[k + 1], xa))
                g[k + 1] += t; big = max(big, abs(t))
            # (8) MASTER carry pass-through  M_{k-1} -> M_{k+1} across master slot k
            #     effect  : f[k] += t, f[k+1] += t ; xm(k-1) -= t ; xm(k+1) += t ; xm(k) unchanged
            #     equation: nu_{k-1} = nu_{k+1}  (slot k is dead, so it is skipped over)
            #     bounds  : both carries in [0, cap], xm(k-1) >= 0, xm(k+1) >= 0
            if 1 <= k <= N - 2:
                xa, xb_ = xm(k - 1), xm(k + 1)
                t = step(th1, h[k - 1], xa, 1, th1, h[k + 1], xb_, 1)
                lo = max(-f[k], -f[k + 1], -xb_)
                hi = min(capM[k - 1] - f[k], capM[k] - f[k + 1], xa)
                t = min(max(t, lo), hi)
                f[k] += t; f[k + 1] += t; big = max(big, abs(t))
            # (9) SLAVE carry pass-through  S_{k-1} -> S_{k+1} across slave slot k
            #     effect  : g[k] += t, g[k+1] += t ; xs(k-1) -= t ; xs(k+1) += t ; xs(k) unchanged
            #     equation: nubar_{k-1} = nubar_{k+1}
            #     bounds  : both carries >= 0, both arrival caps (slot k and slot k+1), xs >= 0 at the ends
            if 1 <= k <= N - 2 and tx[k - 1] and tx[k + 1]:
                xa, xb_ = xs(k - 1), xs(k + 1)
                t = step(th2, hb[k - 1], xa, 1, th2, hb[k + 1], xb_, 1)
                lo = max(-g[k], -g[k + 1], -xb_)
                hi = min(W - a[k] * d[k] - g[k], W - a[k + 1] * d[k + 1] - g[k + 1], xa)
                t = min(max(t, lo), hi)
                g[k] += t; g[k + 1] += t; big = max(big, abs(t))
            # (10) DELTA SHIFT  M_k -> M_{k+1} across slave right tap k (slave spending unchanged)
            #     effect  : d[k] += t, g[k+1] += a_k t, d[k+1] -= (a_k/a_{k+1}) t ;
            #               xm(k) -= t ; xm(k+1) += (a_k/a_{k+1}) t ; xs(k), xs(k+1) unchanged
            #     equation: nu_k = nu_{k+1}  (re-times WHEN the master sends, e.g. into a silent slot)
            #     bounds  : d[k], d[k+1] >= 0 ; g[k+1] >= 0 ; xm(k), xm(k+1) >= 0 (arrival caps unchanged
            #               at k+1, and at k via apply())
            if ak > 0 and a[k + 1] > 0:
                rr = ak / a[k + 1]
                t = step(th1, h[k], xm(k), 1, th1, h[k + 1], xm(k + 1), rr)
                big = max(big, apply([], [(k + 1, ak)], [(k, 1.0), (k + 1, -rr)],
                                     (xm(k), -1.0), (xm(k + 1), rr), t))
        it += 1
        # Stop rule: no plain move exceeds tol*scale -> try the longer pass-through paths; if those
        # do nothing either, the KKT conditions of section 3-4 hold and we are done.
        if big < tol * scale:
            if polish() < tol * scale:        # nothing improves even along longer paths
                converged = True
                break

    # ---- post-processing: flows -> physical quantities ---------------------------------------
    x = np.maximum([xm(i) for i in range(N)], 0.0)      # tau*P_i ; max(.,0) only removes round-off
    xb = np.maximum([xs(i) for i in range(N)], 0.0)     # tau*Pbar_i
    P, Pbar = x / tau, xb / tau                          # powers [W]
    delta = d.copy()
    F = np.cumsum(E - x - delta)                         # F_n = sum_{i<=n}(E_i - tau*P_i - delta_i)  (M1)
    b = jp.B_min + g[1:]                                 # b_n = B_min + Gt_n   (end of slot n)
    b_prev = jp.B_min + g[:-1]                   # level before arrival k
    b_peak = b_prev + a * delta                          # b_{k-1} + a_k delta_k  (left side of S2, <= B_max)
    r_m = float(np.sum(0.5 * tau * np.log1p(h * P)))     # master rate  sum tau/2 ln(1+h P)   [nats]
    r_s = float(np.sum(0.5 * tau * np.log1p(hb * Pbar))) # slave rate   sum tau/2 ln(1+hbar Pbar)
    return JointResult(
        P=P, Pbar=Pbar, delta=delta, F=F, b=b, b_peak=b_peak,
        sent_power=delta / tau, arrived_power=jp.alpha * delta / tau,
        stored_power=a * delta / tau, rx_master=h * P, rx_slave=hb * Pbar,
        rate_master=r_m, rate_slave=r_s, objective=th1 * r_m + th2 * r_s,
        iters=it, converged=converged)


# ----------------------------------------------------------------------------
# Verification helpers
# ----------------------------------------------------------------------------
def check_kkt_joint(jp: JointParams, r: JointResult, tol: float = 1e-6) -> dict:
    """Feasibility + the delta (down-tap) condition  Lambda_k = a_k * Gamma_k.

    m_k = th1*h/(2(1+hP))      master marginal value of a Joule in slot k
    s_k = th2*hb/(2(1+hb*Pb))  slave  marginal value of a Joule spent in slot k
    delta_k > 0  ->  m_k = a_k s_k  (if the slave's arrival cap is slack)
    delta_k = 0  ->  m_k >= a_k s_k

    What it does
    ------------
    Takes a finished solution and reports (i) whether it is feasible (battery inside
    [B_min, B_max], no negative power, all energy used) and (ii) how far it is from the KKT
    condition of the down tap (the only condition not visible from feasibility alone).

    Equations
    ---------
    Stationarity for delta_k:   -Lambda_k + a_k*Gamma_k + rho_k = 0,  rho_k >= 0,  rho_k*delta_k = 0.
    With Lambda_k = m_k and Gamma_k = s_k (arrival cap slack, both powers > 0):
        * delta_k > 0 (rho_k = 0):  m_k = a_k*s_k    -> relative gap (m_k - a_k s_k)/m_k should be 0
        * delta_k = 0 (rho_k >= 0): m_k >= a_k*s_k   -> the gap must not be negative
    Slots where the slave arrival cap is active (mu_bar_{k-1} > 0) are excluded from the
    equality test because there Gamma_k = Lambda_bar_k - mu_bar_{k-1} < s_k.

    Parameters
    ----------
    jp : the problem;   r : the JointResult to test;   tol : (unused, kept for API symmetry)

    Returns a dict (targets in brackets):
        final_master_carry           F_{N-1}                     [~0: master used all energy]
        slave_end_level              b_{N-1}                     [~B_min: slave used all usable energy]
        min_P, min_Pbar, min_delta   smallest values             [>= 0]
        slave_min_level              min_n b_n                   [>= B_min]
        slave_max_level              max_k (b_{k-1}+a_k delta_k) [<= B_max]
        min_master_carry             min_{n<N-1} F_n             [>= 0]
        delta_active_slots           number of slots with delta > 0
        delta_cond_violation_active  max |m_k - a_k s_k|/m_k where delta>0 and cap slack   [~0]
        delta_cond_violation_inactive max of (a_k s_k - m_k)/m_k where delta=0             [~0]
        arrival_cap_active_slots     number of slots where the slave ceiling is reached
    """
    m = jp.master
    h, hb, a = m.h, jp.h_bar, jp.a
    mk = jp.theta1 * h / (2 * (1 + h * r.P))                     # m_k  = master marginal value per Joule
    sk = jp.theta2 * hb / (2 * (1 + hb * r.Pbar))                # s_k  = slave  marginal value per Joule
    gap = (mk - a * sk) / mk                                     # relative gap in  m_k = a_k s_k
    tx = jp.slave_schedule
    on = (r.delta > 1e-7 * max(m.E.mean(), 1.0)) & tx            # shared AND slave transmits (local condition valid)
    gap = np.where(tx, gap, np.inf)                              # silent slots: energy is valued at a later slot, skip
    cap_slack = r.b_peak < jp.B_max - 1e-6                       # slave ceiling NOT reached (mu_bar = 0)
    v_on = np.abs(gap[on & cap_slack]).max() if np.any(on & cap_slack) else 0.0     # equality where delta>0
    v_off = np.maximum(-gap[~on], 0.0).max() if np.any(~on) else 0.0                # inequality where delta=0
    return {
        "final_master_carry": float(r.F[-1]),
        "slave_end_level": float(r.b[-1]),
        "min_P": float(r.P.min()), "min_Pbar": float(r.Pbar.min()),
        "min_delta": float(r.delta.min()),
        "slave_min_level": float(r.b.min()),          # >= B_min
        "slave_max_level": float(r.b_peak.max()),     # <= B_max
        "min_master_carry": float(r.F[:-1].min()),    # >= 0
        "delta_active_slots": int(on.sum()),
        "delta_cond_violation_active": float(v_on),   # ~0
        "delta_cond_violation_inactive": float(v_off),# ~0
        "arrival_cap_active_slots": int((~cap_slack).sum()),
        "slave_silent_slots": int((~tx).sum()),                                              # schedule s_i = 0
        "max_Pbar_in_silent_slots": float(r.Pbar[~tx].max()) if np.any(~tx) else 0.0,       # S4: ~0
    }


def reference_solution_joint(jp: JointParams):
    """Independent solve of the same convex problem with cvxpy (pip install cvxpy).
    Used only to cross-check the DWF; returns (P, Pbar, delta, objective).

    What it does
    ------------
    Writes the problem of the module docstring (sections 1 and 3) directly as a convex program
    over the ENERGY variables  x = tau*P,  dl = delta,  xb = tau*Pbar  (all >= 0) and lets a
    generic interior-point solver (CLARABEL) find the optimum.  It shares no code with joint_dwf(),
    so agreement between the two is a real test of the DWF.  Constraint by constraint (with
    cum(v) = cumulative sum, i.e. sum_{i<=k} v_i):

        master causality   cum(x+dl)_k <= cum(E)_k                    M1  (F_k >= 0)
        master uses all    cum(x+dl)_{N-1} == cum(E)_{N-1}            F_{N-1} = 0
        master ceiling     cum(E)_k - cum(x+dl)_k <= E_max - E_{k+1}  M2  (only if E_max finite)
        slave floor        b0 + cum(a*dl - xb)_k >= B_min             S1  (b_k >= B_min)
        slave ceiling      b0 + cum(a*dl)_k - cum(xb)_{k-1} <= B_max  S2  (b_{k-1} + a_k dl_k <= B_max)

    Objective: th1*sum (tau/2) ln(1 + h*x/tau) + th2*sum (tau/2) ln(1 + hbar*xb/tau).
    Returns P = x/tau, Pbar = xb/tau, delta, and the optimal objective value.
    """
    import cvxpy as cp
    m = jp.master
    N, tau, E, h, hb, a = m.N, m.tau, m.E, m.h, jp.h_bar, jp.a
    x, dl, xb = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
    cumE = np.cumsum(E)                                                 # cumulative harvest sum_{i<=k} E_i
    cum = lambda v: cp.cumsum(v)                                        # cumulative sum of a cvxpy vector
    cons = [cum(x + dl) <= cumE,                                        # master causality
            cum(x + dl)[-1] == cumE[-1],
            jp.b0 + cum(cp.multiply(a, dl) - xb) >= jp.B_min,           # slave floor (end of slot)
            jp.b0 + cum(cp.multiply(a, dl)) - cp.hstack([0, cum(xb)[:-1]]) <= jp.B_max]  # slave ceiling (arrival)
    if not jp.slave_schedule.all():                                     # S4: silent slots carry no slave data
        cons.append(xb[np.flatnonzero(~jp.slave_schedule)] == 0)
    if np.isfinite(m.E_max):                                            # master ceiling (arrival)
        cons.append(cumE[:-1] - cum(x + dl)[:-1] <= m.E_max - E[1:])
    obj = (jp.theta1 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(h, x) / tau))
           + jp.theta2 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(hb, xb) / tau)))
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return x.value / tau, xb.value / tau, dl.value, float(prob.value)


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
    """Per-slot throughput of the master and slave links in bits: (tau/2) ln(1+gain*P) / ln 2."""
    m = jp.master
    return (0.5 * m.tau * np.log1p(m.h * r.P) * BITS_PER_NAT,
            0.5 * m.tau * np.log1p(jp.h_bar * r.Pbar) * BITS_PER_NAT)


def _bound_masks(jp: JointParams, r: JointResult):
    """Slots where a causality / capacity constraint is binding (the water level may jump there).

    master_empty : F_k = 0     master battery empty after slot k   (M1, causality)   [not the last slot]
    master_full  : F_k = cap_k master battery full after slot k    (M2, overflow)    [finite E_max only]
    slave_empty  : b_k = B_min slave battery at its floor          (S1, causality)   [not the last slot]
    slave_full   : b_peak_k = B_max  slave battery full on arrival (S2, overflow)
    """
    m, N = jp.master, jp.master.N
    master_empty = np.zeros(N, bool)
    master_full = np.zeros(N, bool)
    master_empty[:-1] = r.F[:-1] <= EDGE_TOL
    cap = np.asarray(m.cap, float)
    fin = np.isfinite(cap)
    master_full[:-1][fin] = r.F[:-1][fin] >= cap[fin] - EDGE_TOL
    slave_empty = np.zeros(N, bool)
    slave_empty[:-1] = r.b[:-1] <= jp.B_min + EDGE_TOL
    slave_full = r.b_peak >= jp.B_max - EDGE_TOL
    return master_empty, master_full, slave_empty, slave_full


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
    axs.set_ylabel("slave power [W]")
    axs.set_title("Slave: data power P̄_k vs water level", loc="left", color=INK2, fontsize=10)
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


def plot_physical_stored_power(jp: JointParams, r: JointResult, ph: PhysicalStorageResult,
                                path: str, title: str = ""):
    """Solver assumption vs physical reality for the power that ends up in the slave battery.

        arrived optical power    alpha*delta/tau                (dashed)  reaches the photodiode
        solver-assumed stored    beta*alpha*delta/tau            (dotted)  joint_dwf()'s linear a_i model
        physical MPPT output     chain.mppt_output_power         (solid)   photodiode + MPPT, physical_chain.py
    """
    t = np.arange(jp.master.N)
    fig, ax = _new_axis()
    ax.plot(t, r.arrived_power, color=C_LINK, lw=2, ls="--", label="optical power arriving at photodiode, αδ/τ")
    ax.plot(t, r.stored_power, color=C_LINK, lw=2, ls=":", label="stored power assumed by the solver, βαδ/τ")
    ax.plot(t, ph.stored_power_phys, color=C_SLAVE, lw=2, label="stored power from the physical photodiode + MPPT model")
    ax.set_ylabel("average power over the slot [W]")
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
    ax.plot(t, r.b, color=C_SLAVE, lw=2, label="battery level, solver's linear model (β_k α_k δ_k)")
    ax.plot(t, ph.b_phys, color=C_MASTER, lw=2, ls="--", label="battery level, physical photodiode + MPPT arrivals")
    ax.axhline(jp.B_max, color=INK2, lw=1, ls=":")
    ax.axhline(jp.B_min, color=INK2, lw=1, ls=":")
    ax.annotate(f"ceiling B_max = {jp.B_max:g} J", (t[-1], jp.B_max), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.annotate(f"floor B_min = {jp.B_min:g} J", (t[-1], jp.B_min), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.set_ylabel("slave battery energy at end of slot [J]")
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
                    fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color=INK2)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("energy over all slots [J]")
    ax.set_xlabel(" ")                                   # not a per-slot plot: suppress the slot-index label
    ax.text(0.99, 0.97, f"energy sent by master: {sent:.4g} J", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2)
    _finish(fig, ax, path, "Energy lost at each element of the master → slave link", title, None)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run as a script:  python dwf_master_slave.py
    # Test_harvest = np.array([0,12,0])

    # Physical link parameters (given by the user)
    SLAVE_SCHEDULE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]               # 1 = slave transmits, 0 = silent; repeats cyclically over the slots
    LINK_DISTANCE_M = 0.0005             # master<->slave separation [m], shared by both links
    FREQ_THZ = 0.30                      # THz operating frequency chosen for the data link
    OPTICAL_ATTEN_450NM = 0.046         # Beer-Lambert c [1/m], 450 nm (blue) light in water

    # THz data link: symmetric, so the same gain is used for master->x and slave->master
    h_thz = thz_gain(FREQ_THZ, LINK_DISTANCE_M)          # h0 = 1.0 (uncalibrated reference)

    # 1) master: winter harvest array, unlimited master battery, 60 s slots, THz data link
    master = MasterParams(E=Harvested_energy_winter, E_max=np.inf, tau=60.0, h=h_thz)
    # 2) slave: alpha = Beer-Lambert transmittance of the 450 nm optical energy link,
    #    beta = 0.8, battery window [200, 5000] J, return data link = same THz channel
    jp0 = JointParams(master=master, alpha=beer_lambert(OPTICAL_ATTEN_450NM, LINK_DISTANCE_M),
                      beta=0.4, B_min=20.0, B_max=1500.0, h_bar=h_thz,
                      slave_schedule=SLAVE_SCHEDULE)

    # Calibrate beta_i against the real photodiode+MPPT chain (Option B, Instructions.md), then
    # solve with it. joint_dwf() itself is unmodified; only jp.beta becomes per-slot and real.
    jp, r, ph, n_outer = solve_with_physical_chain(jp0, wavelength_nm=532, mppt_efficiency=0.95)
    diagnose(jp, r)
    k = check_kkt_joint(jp, r)              # feasibility + delta-condition report
    print(f"iters={r.iters} converged={r.converged}")
    print(f"objective={r.objective:.4f}  master={r.rate_master:.4f}  slave={r.rate_slave:.4f}")
    print(f"P  mean={r.P.mean():.2f} W   Pbar mean={r.Pbar.mean():.2f} W   "
          f"shared mean={r.sent_power.mean():.2f} W")
    for key, v in k.items():
        print(f"  {key}: {v}")

    # Calibration diagnostics: the solver's battery should now equal the physical one.
    print(f"calibration outer iters = {n_outer}, beta range = "
          f"[{jp.beta.min():.4f}, {jp.beta.max():.4f}]")
    print(f"max |solver battery - physical battery| = {np.max(np.abs(r.b - ph.b_phys)):.3e} J")
    print(f"stored energy  solver={r.stored_power.sum()*master.tau:.2f} J   "
          f"physical={ph.stored_energy_phys.sum():.2f} J")
    print(f"physical battery floor_violations={ph.floor_violations}  "
          f"ceiling_violations={ph.ceiling_violations}  overflow_energy={ph.overflow_energy:.3e} J")

    caption = "Master + slave DWF (THz data link, 450 nm optical energy link)"
    plot_link_power(jp, r, "plot_link_power.png", caption)
    plot_transmit_power(jp, r, "plot_transmit_power.png", caption)
    plot_slave_battery(jp, r, "plot_slave_battery.png", caption)
    plot_throughput_per_slot(jp, r, "plot_throughput_per_slot.png", caption)
    plot_throughput_cumulative(jp, r, "plot_throughput_cumulative.png", caption)
    plot_harvested_energy(jp, r, "plot_harvested_energy.png", caption)
    plot_sharing_energy(jp, r, "plot_sharing_energy.png", caption)
    plot_cumulative_energy(jp, r, "plot_cumulative_energy.png", caption)
    plot_physical_stored_power(jp, r, ph, "plot_physical_stored_power.png", caption)
    plot_physical_slave_battery(jp, r, ph, "plot_physical_slave_battery.png", caption)
    plot_loss_histogram(jp, r, ph, "plot_loss_histogram.png", caption)