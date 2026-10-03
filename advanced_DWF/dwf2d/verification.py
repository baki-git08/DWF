"""Verification helpers: KKT / feasibility report and the cvxpy reference solution.
"""
from __future__ import annotations

import numpy as np

from .params import JointParams, JointResult
from .solver import InfeasibleSlaveBudget


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
        slave_overhead_total_J       sum c_i                      [J, 0 without a load]
        max_cap_violation_J          max(a_k delta_k - capA_k, 0) [~0, S5]
        mppt_cap_active_slots        slots where S5 binds (delta condition skipped there)
        min_slave_level_margin_J     min(min b - B_min, B_max - max b_peak)   [>= 0]
        slave_ledger_error_J         (b0-B_min) + sum a delta - sum c - sum tau Pbar - (b_N-1 - B_min)  [~1e-9]
    """
    m = jp.master
    jm_tau = m.tau
    h, hb, a = m.h, jp.h_bar, jp.a
    mk = jp.theta1 * h / (2 * (1 + h * r.P))                     # m_k  = master marginal value per Joule
    sk = jp.theta2 * hb / (2 * (1 + hb * r.Pbar))                # s_k  = slave  marginal value per Joule
    gap = (mk - a * sk) / mk                                     # relative gap in  m_k = a_k s_k
    tx = jp.slave_schedule
    on = (r.delta > 1e-7 * max(m.E.mean(), 1.0)) & tx            # shared AND slave transmits (local condition valid)
    gap = np.where(tx, gap, np.inf)                              # silent slots: energy is valued at a later slot, skip
    cap_slack = r.b_peak < jp.B_max - 1e-6                       # slave ceiling NOT reached (mu_bar = 0)
    capA = jp.capA
    stored = a * r.delta
    s5_bind = np.isfinite(capA) & (stored >= capA - 1e-6 * np.maximum(np.where(np.isfinite(capA), capA, 1.0), 1.0))
    cap_slack = cap_slack & ~s5_bind                             # S5 binds -> its multiplier is nonzero, skip the equality
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
        "arrival_cap_active_slots": int((r.b_peak >= jp.B_max - 1e-6).sum()),                # slave ceiling reached
        "slave_silent_slots": int((~tx).sum()),                                              # schedule s_i = 0
        "max_Pbar_in_silent_slots": float(r.Pbar[~tx].max()) if np.any(~tx) else 0.0,       # S4: ~0
        # slave electronics load (c_i) and MPPT cap (S5)
        "slave_overhead_total_J": float(jp.overhead.sum()),
        "max_cap_violation_J": float(max(np.max(np.where(np.isfinite(capA), stored - capA, -np.inf)), 0.0)) if np.isfinite(capA).any() else 0.0,   # S5: ~0
        "mppt_cap_active_slots": int(s5_bind.sum()),
        "min_slave_level_margin_J": float(min(r.b.min() - jp.B_min, jp.B_max - r.b_peak.max())),         # >= 0
        "slave_ledger_error_J": float((jp.b0 - jp.B_min) + stored.sum() - jp.overhead.sum()
                                       - jm_tau * r.Pbar.sum() - (r.b[-1] - jp.B_min)),                # ~0 (1e-9)
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
        slave floor        b0 + cum(a*dl - xb - c)_k >= B_min         S1  (b_k >= B_min)
        slave ceiling      b0 + cum(a*dl)_k - cum(xb + c)_{k-1} <= B_max  S2  (b_{k-1} + a_k dl_k <= B_max)
        slave terminal     b0 + cum(a*dl - xb - c)_{N-1} == B_min    g[N] = 0 (same as joint_dwf)
        MPPT input cap     a_k*dl_k <= capA_k                         S5  (only if jp.mppt_cap_J is set)

    c_i = jp.overhead (slave non-TX electronics energy per slot; zeros when no load is attached).

    Objective: th1*sum (tau/2) ln(1 + h*x/tau) + th2*sum (tau/2) ln(1 + hbar*xb/tau).
    Returns P = x/tau, Pbar = xb/tau, delta, and the optimal objective value.
    """
    import cvxpy as cp
    m = jp.master
    N, tau, E, h, hb, a = m.N, m.tau, m.E, m.h, jp.h_bar, jp.a
    cS, capA = jp.overhead, jp.capA
    x, dl, xb = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
    cumE = np.cumsum(E)                                                 # cumulative harvest sum_{i<=k} E_i
    cum = lambda v: cp.cumsum(v)                                        # cumulative sum of a cvxpy vector
    cons = [cum(x + dl) <= cumE,                                        # master causality
            cum(x + dl)[-1] == cumE[-1],
            jp.b0 + cum(cp.multiply(a, dl) - xb - cS) >= jp.B_min,      # slave floor (end of slot)
            jp.b0 + cum(cp.multiply(a, dl)) - cp.hstack([0, cum(xb + cS)[:-1]]) <= jp.B_max,  # slave ceiling (arrival)
            jp.b0 + cum(cp.multiply(a, dl) - xb - cS)[-1] == jp.B_min]  # slave terminal: g[N] = 0
    fin_cap = np.isfinite(capA)
    if fin_cap.any():                                                   # S5: stored energy per slot <= capA
        cons.append(cp.multiply(a, dl)[np.flatnonzero(fin_cap)] <= capA[fin_cap])
    if not jp.slave_schedule.all():                                     # S4: silent slots carry no slave data
        cons.append(xb[np.flatnonzero(~jp.slave_schedule)] == 0)
    if np.isfinite(m.E_max):                                            # master ceiling (arrival)
        cons.append(cumE[:-1] - cum(x + dl)[:-1] <= m.E_max - E[1:])
    obj = (jp.theta1 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(h, x) / tau))
           + jp.theta2 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(hb, xb) / tau)))
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if x.value is None:                                                  # status infeasible / unbounded
        raise InfeasibleSlaveBudget(f"reference problem is {prob.status}: S1-S5 with g[N] = 0 cannot be satisfied")
    return x.value / tau, xb.value / tau, dl.value, float(prob.value)
