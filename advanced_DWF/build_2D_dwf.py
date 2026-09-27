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

from dataclasses import dataclass, field

import numpy as np

from build_dwf import Harvested_energy_summer, MasterParams, Harvested_energy_winter 
from dwf_diagnostics import diagnose


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


def thz_gain(freq_thz: float | np.ndarray, d: float, h0: float = 1.0) -> np.ndarray:
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
    beta : battery-recharging efficiency in [0, 1].                     [PLACEHOLDER 0.8]
        Only a_i*delta_i = beta*alpha_i*delta_i of the shared energy ends up in the slave battery.
    B_min, B_max : slave battery floor / ceiling [J]  (given: 200 and 5000).
        Battery level always satisfies  B_min <= b_n  and  b_{n-1} + a_n*delta_n <= B_max.
    b0 : initial slave battery level [J]. Default (None) = B_min, i.e. the reserve is already
        in the battery and the slave starts with zero usable energy.  [ASSUMPTION]
    h_bar : slave -> master data-link gain over noise hbar_i (scalar or per-slot).  [PLACEHOLDER 0.1]
    theta1, theta2 : weights of the master and slave rate in the objective.
    """
    master: MasterParams
    alpha: np.ndarray | float = 0.368     # PLACEHOLDER  exp(-0.2 1/m * 5 m)
    beta: float = 0.8                     # PLACEHOLDER  recharging efficiency
    B_min: float = 200.0                  # given
    B_max: float = 5000.0                 # given
    b0: float | None = None               # initial slave charge; default = B_min (ASSUMPTION)
    h_bar: np.ndarray | float = 0.1       # PLACEHOLDER  slave -> master link gain (SNR per W)
    theta1: float = 1.0                   # weight of master rate
    theta2: float = 1.0                   # weight of slave rate

    def __post_init__(self):
        """Broadcast scalars to one value per slot and validate the model assumptions."""
        n = self.master.N
        # alpha_i and hbar_i may be given as scalars; the solver needs one value per slot.
        self.alpha = np.broadcast_to(np.asarray(self.alpha, float), (n,)).copy()
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
        if np.any((self.alpha < 0) | (self.alpha > 1)) or not (0 <= self.beta <= 1):
            raise ValueError("alpha and beta must be in [0, 1]")
        if np.any(self.h_bar <= 0):
            raise ValueError("h_bar must be > 0")

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

    # ---- state: the three families of flow variables (all in Joules) -------------------------
    f = np.zeros(N + 1)                           # f[k+1] = master carry over tap k
    #   f[k+1] = F_k = master energy carried from slot k to k+1;  f[0] = f[N] = 0 stay fixed
    #   (nothing arrives before slot 0 and nothing is left after slot N-1).
    g = np.zeros(N + 1)                           # g[k+1] = slave carry above floor
    #   g[k+1] = Gt_k = b_k - B_min;  g[0] = b0 - B_min (initial excess), g[N] = 0 stays fixed
    #   (the slave also uses all its energy by the end).
    g[0] = jp.g_init
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
            so = [(k, [])]                                          # (sink slot j, g changes)
            j = k + 1                                               # sinks AFTER k: g[k+1..j] += a_k*t
            while j <= N - 1:
                so.append((j, [(q, a[k]) for q in range(k + 1, j + 1)]))
                if not dead(xs(j)):
                    break
                j += 1
            j = k - 1                                               # sinks BEFORE k: g[j+1..k] -= a_k*t
            while j >= 0:
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
            if ak > 0:
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
            if ak > 0 and k >= 1:
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
            if ak > 0 and k >= 1:
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
            if ak > 0 and k <= N - 2:
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
            if ak > 0 and k <= N - 2:
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
            if 1 <= k <= N - 2:
                xa, xb_ = xs(k - 1), xs(k + 1)
                t = step(th2, hb[k - 1], xa, 1, th2, hb[k + 1], xb_, 1)
                lo = max(-g[k], -g[k + 1], -xb_)
                hi = min(W - a[k] * d[k] - g[k], W - a[k + 1] * d[k + 1] - g[k + 1], xa)
                t = min(max(t, lo), hi)
                g[k] += t; g[k + 1] += t; big = max(big, abs(t))
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
    on = r.delta > 1e-7 * max(m.E.mean(), 1.0)                   # slots where energy is actually shared
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
    if np.isfinite(m.E_max):                                            # master ceiling (arrival)
        cons.append(cumE[:-1] - cum(x + dl)[:-1] <= m.E_max - E[1:])
    obj = (jp.theta1 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(h, x) / tau))
           + jp.theta2 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(hb, xb) / tau)))
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return x.value / tau, xb.value / tau, dl.value, float(prob.value)


# ----------------------------------------------------------------------------
# Plot: received power over the channel | transmit power | battery
# ----------------------------------------------------------------------------
# Categorical slots 1-3 of the reference palette (light surface), fixed order.
C_MASTER, C_SLAVE, C_LINK = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def plot_joint(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Draw the three-panel result figure and save it to `path` (PNG).

    Panels (x-axis = slot index, one slot = one minute)
    ---------------------------------------------------
    1. Power over the underwater energy link, three stages of the same flow:
           sent by the master      delta/tau                     (solid)
           received after channel  alpha*delta/tau               (dashed)  Beer-Lambert loss
           stored after recharge   beta*alpha*delta/tau = a*delta/tau (dotted)  recharging loss
    2. Transmit (data) powers:  master P_i = xm/tau  and slave Pbar_i = xs/tau.
    3. Slave battery: level after each slot b_n = B_min + Gt_n (solid) and level right after
       each arrival b_{n-1} + a_n delta_n (dashed), with the limits B_min and B_max as dotted lines.

    Colours follow the entity: blue = master, orange = slave, green = the energy link.
    """
    import matplotlib
    matplotlib.use("Agg")                                   # draw to a file, no window
    import matplotlib.pyplot as plt

    m, t = jp.master, np.arange(jp.master.N)                # t = slot index 0..N-1
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True, facecolor="white")
    for a_ in ax:                                           # common, low-key styling
        a_.set_facecolor("white")
        a_.grid(True, axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.tick_params(colors=INK2)

    def label(a_, x, y, text, color_line):
        """(Unused helper) write a short text label next to a point."""
        a_.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)

    # 1) power over the energy channel (same entity, three stages -> one hue, three dashes)
    ax[0].plot(t, r.sent_power, color=C_LINK, lw=2, ls="-", label="sent by master, δ/τ")
    ax[0].plot(t, r.arrived_power, color=C_LINK, lw=2, ls="--", label="received after channel, αδ/τ")
    ax[0].plot(t, r.stored_power, color=C_LINK, lw=2, ls=":", label="stored after recharge, βαδ/τ")
    ax[0].set_ylabel("power over the energy link [W]")
    ax[0].set_ylim(bottom=0)
    ax[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=9, labelcolor=INK2)
    ax[0].set_title("Received power over the underwater channel", loc="left", color=INK, fontsize=11)

    # 2) transmit powers
    ax[1].plot(t, r.P, color=C_MASTER, lw=2, label="master data power P")
    ax[1].plot(t, r.Pbar, color=C_SLAVE, lw=2, label="slave data power P̄")
    ax[1].set_ylabel("transmit power [W]")
    ax[1].set_ylim(bottom=0)
    ax[1].legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)
    ax[1].set_title("Transmit power", loc="left", color=INK, fontsize=11)

    # 3) slave battery inside [B_min, B_max]
    ax[2].plot(t, r.b, color=C_SLAVE, lw=2, label="slave battery after slot")
    ax[2].plot(t, r.b_peak, color=C_SLAVE, lw=1.2, ls="--", label="slave battery just after arrival")
    ax[2].axhline(jp.B_max, color=INK2, lw=1, ls=":")       # ceiling B_max (S2)
    ax[2].axhline(jp.B_min, color=INK2, lw=1, ls=":")       # floor   B_min (S1)
    ax[2].annotate(f"B_max = {jp.B_max:g} J", (t[-1], jp.B_max), xytext=(0, 5),
                   textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax[2].annotate(f"B_min = {jp.B_min:g} J", (t[-1], jp.B_min), xytext=(0, 5),
                   textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax[2].set_ylim(0, jp.B_max * 1.08)
    ax[2].set_ylabel("slave battery [J]")
    ax[2].set_xlabel("slot index (1 slot = 1 min)")
    ax[2].legend(frameon=False, loc="center right", fontsize=9, labelcolor=INK2)
    ax[2].set_title("Slave battery level", loc="left", color=INK, fontsize=11)

    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_throughput(jp: JointParams, r: JointResult, path: str, title: str = ""):
    """Draw the per-slot and cumulative throughput (data rate) of master and slave.

    Instantaneous throughput per slot [nats]:
        master  rate_i = (tau/2) ln(1 + h_i    P_i)
        slave   rate_i = (tau/2) ln(1 + hbar_i Pbar_i)
    These are the per-slot terms that sum to r.rate_master / r.rate_slave (section 1 objective,
    unweighted by theta1/theta2). Panel 1 shows them per slot; panel 2 shows the running total
    (cumulative sum), i.e. data delivered so far.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, t = jp.master, np.arange(jp.master.N)
    rate_m = 0.5 * m.tau * np.log1p(m.h * r.P)          # per-slot master throughput [nats]
    rate_s = 0.5 * m.tau * np.log1p(jp.h_bar * r.Pbar)  # per-slot slave  throughput [nats]

    fig, ax = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, facecolor="white")
    for a_ in ax:
        a_.set_facecolor("white")
        a_.grid(True, axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.tick_params(colors=INK2)

    ax[0].plot(t, rate_m, color=C_MASTER, lw=2, label="master throughput")
    ax[0].plot(t, rate_s, color=C_SLAVE, lw=2, label="slave throughput")
    ax[0].set_ylabel("throughput per slot [nats]")
    ax[0].set_ylim(bottom=0)
    ax[0].legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)
    ax[0].set_title("Per-slot throughput", loc="left", color=INK, fontsize=11)

    ax[1].plot(t, np.cumsum(rate_m), color=C_MASTER, lw=2, label=f"master total = {r.rate_master:.3f} nats")
    ax[1].plot(t, np.cumsum(rate_s), color=C_SLAVE, lw=2, label=f"slave total = {r.rate_slave:.3f} nats")
    ax[1].set_ylabel("cumulative throughput [nats]")
    ax[1].set_xlabel("slot index (1 slot = 1 min)")
    ax[1].set_ylim(bottom=0)
    ax[1].legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    ax[1].set_title("Cumulative throughput", loc="left", color=INK, fontsize=11)

    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run as a script:  python dwf_master_slave.py
    # Test_harvest = np.array([0,12,0])

    # Physical link parameters (given by the user)
    LINK_DISTANCE_M = 0.001              # master<->slave separation [m], shared by both links
    FREQ_THZ = 0.30                      # THz operating frequency chosen for the data link
    OPTICAL_ATTEN_450NM = 0.0399         # Beer-Lambert c [1/m], 450 nm (blue) light in water

    # THz data link: symmetric, so the same gain is used for master->x and slave->master
    h_thz = thz_gain(FREQ_THZ, LINK_DISTANCE_M)          # h0 = 1.0 (uncalibrated reference)

    # 1) master: winter harvest array, unlimited master battery, 60 s slots, THz data link
    master = MasterParams(E=Harvested_energy_winter, E_max=np.inf, tau=60.0, h=h_thz)
    # 2) slave: alpha = Beer-Lambert transmittance of the 450 nm optical energy link,
    #    beta = 0.8, battery window [200, 5000] J, return data link = same THz channel
    jp = JointParams(master=master, alpha=beer_lambert(OPTICAL_ATTEN_450NM, LINK_DISTANCE_M),
                     beta=0.5, B_min=20.0, B_max=1700.0, h_bar=h_thz)
    r = joint_dwf(jp)                       # solve
    diagnose(jp, r)
    k = check_kkt_joint(jp, r)              # feasibility + delta-condition report
    print(f"iters={r.iters} converged={r.converged}")
    print(f"objective={r.objective:.4f}  master={r.rate_master:.4f}  slave={r.rate_slave:.4f}")
    print(f"P  mean={r.P.mean():.2f} W   Pbar mean={r.Pbar.mean():.2f} W   "
          f"shared mean={r.sent_power.mean():.2f} W")
    for key, v in k.items():
        print(f"  {key}: {v}")
    plot_joint(jp, r, "test_THz_optics_2.png",
               "Master + slave DWF (THz data link, 450 nm optical energy link)")
    plot_throughput(jp, r, "throughput_2.png",
                    "Master + slave throughput (THz data link, 450 nm optical energy link)")