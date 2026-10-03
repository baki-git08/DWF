"""Problem inputs and solver output: JointParams, JointResult.

Notation follows sections 1-3 of the build_2D_dwf.py module docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from build_dwf import MasterParams


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
    slave_overhead_J : c_i >= 0 [J] per slot, scalar or length N. Non-TX electronics energy the slave
        burns in slot i (camera + MCU + memory + MPPT), see slave_load.attach_slave_load(). Enters the
        slave balance as  tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i - c_i.  None (default) = 0.
    mppt_cap_J : S5, stored-energy cap per slot [J] (scalar or length N): a_k*delta_k <= capA_k
        (MPPT input rating).  None (default) = no cap.
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
    slave_overhead_J: np.ndarray | float | None = None   # c_i: non-TX slave electronics energy per slot (None = 0)
    mppt_cap_J: np.ndarray | float | None = None         # S5: a_k*delta_k <= capA_k, stored energy per slot (None = inf)

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
        # Slave electronics drain c_i >= 0 and MPPT stored-energy cap capA_k > 0 (None keeps today's model).
        if self.slave_overhead_J is not None:
            c = np.asarray(self.slave_overhead_J, float)
            if c.ndim and c.size != n:
                raise ValueError(f"slave_overhead_J needs 1 or N = {n} entries, got {c.size}")
            self.slave_overhead_J = np.broadcast_to(c, (n,)).copy()
            if not np.all(np.isfinite(self.slave_overhead_J)) or np.any(self.slave_overhead_J < 0):
                raise ValueError("slave_overhead_J must be finite and >= 0")
        if self.mppt_cap_J is not None:
            cp = np.asarray(self.mppt_cap_J, float)
            if cp.ndim and cp.size != n:
                raise ValueError(f"mppt_cap_J needs 1 or N = {n} entries, got {cp.size}")
            self.mppt_cap_J = np.broadcast_to(cp, (n,)).copy()
            if np.any(np.isnan(self.mppt_cap_J)) or np.any(self.mppt_cap_J < 0):
                raise ValueError("mppt_cap_J must be >= 0 (use None or inf for no cap)")

    @property
    def a(self) -> np.ndarray:            # net gain of the down tap
        """Net gain of the master->slave energy tap:  a_i = beta * alpha_i  (< 1).

        One Joule sent by the master (delta_i) becomes a_i Joules in the slave battery.
        """
        return self.beta * self.alpha   # <- 

    @property
    def overhead(self) -> np.ndarray:     # c_i, always an array
        """Slave non-TX electronics energy c_i [J] per slot (zeros when no load is attached)."""
        return np.zeros(self.master.N) if self.slave_overhead_J is None else self.slave_overhead_J

    @property
    def capA(self) -> np.ndarray:         # S5 cap, always an array
        """MPPT stored-energy cap per slot [J] (inf when no cap)."""
        return np.full(self.master.N, np.inf) if self.mppt_cap_J is None else self.mppt_cap_J

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
    overhead: np.ndarray | None = None   # c_i  slave non-TX electronics energy per slot [J] (zeros if no load)
    start: str = "share-nothing"         # how the start point was built: share-nothing | just-in-time | lp-fallback
