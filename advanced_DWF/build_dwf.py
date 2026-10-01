"""
dwf_master.py
=============
Directional water-filling (DWF) for a SINGLE energy-harvesting master node.

Step 1 of the master/slave underwater project. Only the master exists here:
no delta_i (energy sharing), no slave. The structure is chosen so those can
be added later without rewriting the core.

Problem (slots i = 0..N-1, slot length tau)
-------------------------------------------
    max   sum_i  (tau/2) * ln(1 + h_i * P_i)
    s.t.  causality : F_n >= 0                          (lambda_n)
          capacity  : F_n <= E_max - E_{n+1}            (mu_n)
          P_i >= 0                                      (eta_i)

    F_n = sum_{i<=n} (E_i - tau*P_i)   battery energy carried from slot n to n+1

DWF view
--------
    water level      nu_i = P_i + 1/h_i
    base (floor)     b_i  = 1/h_i
    tap n -> n+1     right-permeable (only forward flow, F_n >= 0)
                     capacity  cap_n = E_max - E_{n+1}
    Stationarity:    P_i = [nu_i - 1/h_i]^+ ,  nu_i = 1 / (2 * Lambda_i)
                     Lambda_i = sum_{n>=i} lambda_n - sum_{n>=i} mu_n

Algorithm
---------
Work with flows f_n = F_n (energy carried over tap n). Spent energy in slot i:
    x_i = tau * P_i = E_i + f_{i-1} - f_i ,   f_{-1} = f_{N-1} = 0.
Start from "spend what you harvest" (f = 0). Repeatedly visit each tap and move
the amount of energy that equalises the two adjacent water levels, clipped by:
    * tap not saturated:      0 <= f_n <= cap_n   (direction + capacity)
    * no negative power:      x_i >= 0, x_{i+1} >= 0
Sweeps alternate forward/backward until no tap moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# GLOBAL VARIABLES
# 60 samples collected over a 1-hour span -> 1 minute per sample.
Harvested_energy_summer = np.array([
    3259.018904543642, 3225.811176367269, 3219.7400798976305,
    3209.669758246244, 3249.4906451390025, 3236.380071665001,
    3251.82378773598, 3223.610002469285, 3237.0895732725335,
    3240.0733948883803, 3164.2113780586687, 3230.8492020193803,
    3215.2177647517046, 3218.005513521021, 3241.167094917427,
    3255.5480745906307, 3232.1343320405863, 3225.884769539228,
    3240.9316782395417, 3261.510851296834, 3223.0173786785344,
    3224.9412380291024, 3231.2154439465958, 3211.0559205948966,
    3214.8859392018485, 3203.4464715152308, 3180.0183280143983,
    3243.8589491804246, 3228.189395874523, 3221.3209687843005,
    3207.9279279447605, 3200.2464371416077, 3219.81254236639,
    3213.9217089027075, 3207.2197442112565, 3224.569901584549,
    3212.609684832128, 3218.96989164818, 3198.0392316207203,
    3158.4607204751087, 3175.6571959014686, 3180.108026137462,
    3197.772281572488, 3188.5012793965257, 3146.3846301963827,
    3156.9431911679744, 3183.102707356057, 3070.5187945800335,
    3193.3999650977457, 3209.8285500632924, 3174.015613864292,
    3133.6307383893263, 3086.283821223546, 3141.097213839667,
    3204.4792179895153, 3170.440171377065, 3166.6525519388197,
    3163.3956233837544, 3149.005419046985, 3182.9536654525896,
])

Harvested_energy_winter = np.array([
    3882.8349011599184, 3837.293431555212, 3835.3170898801413,
    3920.8231356691927, 3894.4475404226796, 3877.354349347581,
    3866.449677993952, 3823.584241825438, 3776.407331600007,
    3857.9808685943804, 3901.259988840484, 3837.4424388264115,
    3854.3690144848724, 3882.1749339676126, 3815.8288850220706,
    3828.9741072716774, 3862.909961559958, 3902.359874998539,
    3884.7796174373325, 3823.7237247604344, 3819.714150493076,
    3816.183538232225, 3835.5080109932615, 3816.5909621360556,
    3837.334554902968, 3766.4313610431277, 3819.2342893451278,
    3849.273720346089, 3878.2345580753763, 3862.5094765334616,
    3785.1661161161624, 3706.7962345387946, 3805.6533307471823,
    3784.1497821020316, 3724.5139847755213, 3792.576717510211,
    3741.669128362503, 3779.3356757828283, 3757.696117792752,
    3776.1303434499246, 3777.799610672603, 3745.1270236784367,
    3761.9465324564108, 3737.151432284734, 3707.6360414115748,
    3740.4737100200364, 3758.3556088429605, 3733.2355241181845,
    3730.580947514336, 3772.6296246640923, 3728.5877054118314,
    3683.5202696341644, 3690.252522097174, 3717.873422291475,
    3791.3408062642325, 3670.4284093365513, 3683.428556830756,
    3746.7250704655835, 3675.7214976087707, 3659.1246662885324,
])


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
@dataclass
class MasterParams:
    """Everything the master-only problem needs.

    E      : energy harvested at the start of each slot            [J]
    E_max  : battery capacity (np.inf recovers the unconstrained case) [J]
    tau    : slot length                                            [s]
    h      : channel gain (scalar or per-slot array). Only enters via the
             base level 1/h_i. Keep 1.0 if you only care about energy shaping.
    """
    E: np.ndarray
    E_max: float = np.inf
    tau: float = 60.0
    h: np.ndarray | float = 1.0

    def __post_init__(self):
        self.E = np.asarray(self.E, dtype=float).copy()
        if np.any(self.E < 0):
            raise ValueError("Harvested energy must be non-negative.")
        n = self.E.size
        self.h = np.broadcast_to(np.asarray(self.h, dtype=float), (n,)).copy()
        if np.any(self.h <= 0):
            raise ValueError("Channel gains h must be > 0.")
        if self.tau <= 0:
            raise ValueError("tau must be > 0.")
        # Model assumption: E_i <= E_max, otherwise the excess is lost on arrival.
        if np.any(self.E > self.E_max):
            lost = float(np.sum(np.maximum(self.E - self.E_max, 0.0)))
            print(f"[MasterParams] {lost:.1f} J exceed E_max on arrival and are clipped.")
            self.E = np.minimum(self.E, self.E_max)

    @property
    def N(self) -> int:
        return self.E.size

    @property
    def base(self) -> np.ndarray:
        """Base (floor) water level 1/h_i."""
        return 1.0 / self.h

    @property
    def cap(self) -> np.ndarray:
        """Tap capacities cap_n = E_max - E_{n+1}, n = 0..N-2 (inf if E_max=inf)."""
        return self.E_max - self.E[1:]


@dataclass
class DWFResult:
    P: np.ndarray            # transmit power per slot                  [W]
    x: np.ndarray            # energy spent per slot = tau * P          [J]
    F: np.ndarray            # battery energy left after slot n (F[-1]=0) [J]
    nu: np.ndarray           # water level nu_i = P_i + 1/h_i
    lam: np.ndarray          # lambda_n  (causality multiplier, tap n)
    mu: np.ndarray           # mu_n      (capacity multiplier, tap n)
    throughput: float        # sum tau/2 ln(1+hP)  [nats]
    iters: int
    converged: bool
    history: list = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------
# Core: directional water-filling
# --------------------------------------------------------------------------
def directional_water_filling(p: MasterParams, tol: float = 1e-10,
                              max_iter: int = 200_000,
                              record_history: bool = False) -> DWFResult:
    N, tau, base, cap, E = p.N, p.tau, p.base, p.cap, p.E
    f = np.zeros(N + 1)          # f[k+1] = flow over tap k ; f[0] = f[N] = 0
    #                              (padded so x_i = E_i + f[i] - f[i+1])
    def spent(i):
        return E[i] + f[i] - f[i + 1]

    history = []
    it, converged = 0, False
    scale = max(float(E.mean()), 1.0)

    while it < max_iter:
        max_move = 0.0
        order = range(N - 1) if it % 2 == 0 else range(N - 2, -1, -1)
        for n in order:                      # tap n between slot n and n+1
            xa, xb = spent(n), spent(n + 1)
            nu_a = xa / tau + base[n]
            nu_b = xb / tau + base[n + 1]
            # energy moved forward that equalises the two levels
            d = 0.5 * tau * (nu_a - nu_b)
            # tap: 0 <= f <= cap (only forward, limited capacity)
            lo = -f[n + 1]
            hi = cap[n] - f[n + 1]
            # no negative power on either side
            lo = max(lo, -xb)
            hi = min(hi, xa)
            d = min(max(d, lo), hi)
            f[n + 1] += d
            max_move = max(max_move, abs(d))
        it += 1
        if record_history and it % 50 == 0:
            history.append(f.copy())
        if max_move < tol * scale:
            converged = True
            break

    x = np.array([spent(i) for i in range(N)])
    x = np.maximum(x, 0.0)
    P = x / tau
    nu = P + base
    F = np.cumsum(E - x)                 # F[-1] should be ~0
    lam, mu = _multipliers(nu)
    thr = float(np.sum(0.5 * tau * np.log1p(p.h * P)))
    return DWFResult(P, x, F, nu, lam, mu, thr, it, converged, history)


def _multipliers(nu: np.ndarray):
    """Recover (lambda_n, mu_n) from water levels.

    Lambda_i = 1/(2 nu_i) and Lambda_n - Lambda_{n+1} = lambda_n - mu_n
    (with Lambda_{N} = 0 beyond the last slot). lambda/mu are not both > 0
    except in the degenerate E_{n+1} = E_max case, so we split by sign.
    Slots with P=0 sit on their floor; their Lambda is only bounded, not
    determined, so treat these values as indicative there.
    """
    Lam = 0.5 / nu
    d = Lam - np.append(Lam[1:], 0.0)
    return np.maximum(d, 0.0), np.maximum(-d, 0.0)


# --------------------------------------------------------------------------
# Verification helpers
# --------------------------------------------------------------------------
def check_kkt(p: MasterParams, r: DWFResult, tol: float = 1e-6) -> dict:
    """Check feasibility + tap complementary slackness (equal levels unless
    a tap is closed (F=0) or saturated (F=cap))."""
    scale = max(float(p.E.mean()), 1.0)
    F, nu, cap = r.F, r.nu, p.cap
    viol = []
    for n in range(p.N - 1):
        if r.P[n] <= 1e-12 or r.P[n + 1] <= 1e-12:
            continue                          # floor slots: skip (see _multipliers)
        closed = F[n] <= tol * scale
        full = F[n] >= cap[n] - tol * scale
        dn = nu[n] - nu[n + 1]
        if dn > tol * max(nu.mean(), 1.0) and not full:      # water wants to go right
            viol.append((n, "level drops but tap not saturated", dn))
        if dn < -tol * max(nu.mean(), 1.0) and not closed:   # would need to flow left
            viol.append((n, "level rises but tap not closed", dn))
    return {
        "min_F": float(F[:-1].min()) if p.N > 1 else 0.0,               # >= 0 (causality)
        "max_capacity_violation": float(np.max(F[:-1] - cap)) if p.N > 1 else 0.0,  # <= 0
        "final_F": float(F[-1]),                                         # ~ 0 (all energy used)
        "min_P": float(r.P.min()),                                       # >= 0
        "tap_violations": viol,
    }


def reference_solution(p: MasterParams):
    """Independent solve with scipy SLSQP, for cross-checking DWF only."""
    from scipy.optimize import minimize
    N, tau = p.N, p.tau
    s = float(p.E.mean())                       # variable scaling
    L = np.tril(np.ones((N, N)))                # cumulative-sum matrix
    cumE = L @ p.E

    def obj(z):                                 # z = tau*P / s
        return -np.sum(0.5 * tau * np.log1p(p.h * z * s / tau))

    def grad(z):
        return -0.5 * tau * (p.h * s / tau) / (1.0 + p.h * z * s / tau)

    cons = [{"type": "ineq", "fun": lambda z: cumE - L @ (z * s), "jac": lambda z: -L * s}]
    if np.isfinite(p.E_max):
        A = L[:-1]
        capv = p.E_max - p.E[1:]
        cons.append({"type": "ineq", "fun": lambda z: capv - (A @ p.E - A @ (z * s)),
                     "jac": lambda z: A * s})
    cons.append({"type": "eq", "fun": lambda z: np.sum(z * s) - p.E.sum(),
                 "jac": lambda z: np.full((1, N), s)})
    z0 = p.E / s
    res = minimize(obj, z0, jac=grad, bounds=[(0, None)] * N, constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    P = res.x * s / tau
    return P, float(np.sum(0.5 * tau * np.log1p(p.h * P)))


def plot_result(p: MasterParams, r: DWFResult, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(p.N)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].step(t, p.E / p.tau, where="post", label="harvest E_i/tau", color="tab:orange")
    ax[0].step(t, r.P, where="post", label="DWF power P_i", color="tab:blue")
    ax[0].set_ylabel("W"); ax[0].legend(); ax[0].set_title(title)
    ax[1].step(t, r.nu, where="post", color="tab:green", label="water level nu_i")
    ax[1].step(t, p.base, where="post", color="grey", ls="--", label="base 1/h_i")
    ax[1].legend()
    ax[2].plot(t, r.F, color="tab:red", label="battery carried F_n")
    if np.isfinite(p.E_max):
        ax[2].plot(t[:-1], p.cap, color="k", ls=":", label="tap cap E_max - E_{n+1}")
    ax[2].set_xlabel("slot (1 min)"); ax[2].set_ylabel("J"); ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    E = Harvested_energy_winter
    channel = np.exp(-0.005*123)
    for label, Emax in [("E_max = inf", np.inf), ("E_max = 4000 J", 4000.0)]:
        p = MasterParams(E=E, E_max=Emax, tau=60.0, h=channel)
        r = directional_water_filling(p)
        P_ref, thr_ref = reference_solution(p)
        k = check_kkt(p, r)
        print(f"\n=== {label} ===")
        print(f"iters={r.iters} converged={r.converged}")
        print(f"throughput DWF={r.throughput:.6f}  reference={thr_ref:.6f}  "
              f"max|P-P_ref|={np.max(np.abs(r.P - P_ref)):.2e} W")
        print(f"final_F={k['final_F']:.2e}  min_F={k['min_F']:.2e}  "
              f"cap_viol={k['max_capacity_violation']:.2e}  min_P={k['min_P']:.2e}")
        print(f"tap KKT violations: {len(k['tap_violations'])}")
        plot_result(p, r, f"dwf_master_{'inf' if np.isinf(Emax) else int(Emax)}.png", label)