"""Verification helpers: KKT / feasibility report and the cvxpy reference solution.
"""
from __future__ import annotations

import numpy as np

from .params import JointParams, JointResult
from .solver import InfeasibleSlaveBudget, master_floor_profile


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
        final_master_carry           F_{N-1}                     [~0: master used all energy; = fL[N] with a floor]
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
                                     (b_N-1 may end above B_min after the maximal-sharing post-pass)

    Stage 6 (keys present only when the feature is on):
        slave_rate_window set  ->  s_k = th2*hb/(2(1 + hb*x_k/T_k)),  x_k = tau*Pbar_k (burst rate)
        theta1 = 0             ->  m_k = 0: the gap is ABSOLUTE (m_k - a_k s_k), every slot with delta > 0 must be
                                   stopped by a bound (ceiling, S5, S6, M5 or the master's own energy); the delta test
                                   runs only on slots where none binds.  master_has_no_data, unused_spill_J.
        tx_cap_active_slots / max_tx_cap_violation_J          S6  (delta test skipped where S6 binds)
        tx_floor_active_slots / max_tx_floor_violation_J      S7
        delta_cap_active_slots / max_delta_cap_violation_J    M5  (delta test skipped where M5 binds)
        master_floor_binding_slots / min_master_floor_margin_J  M6  F_k = fL[k+1] (k < N-1) / min(F_k - fL[k+1])  [>= 0]
        master_full_slots                                       M2  F_k = E_max - E_{k+1}  (finite E_max only)
    """
    m = jp.master
    jm_tau = m.tau
    h, hb, a = m.h, jp.h_bar, jp.a
    mk = jp.theta1 * h / (2 * (1 + h * r.P))                     # m_k  = master marginal value per Joule
    if jp.slave_rate_window is None:
        sk = jp.theta2 * hb / (2 * (1 + hb * r.Pbar))            # s_k  = slave  marginal value per Joule
    else:                                                        # burst rate: (T/2) ln(1 + hb x/T), x = tau*Pbar
        sk = jp.theta2 * hb / (2 * (1 + hb * (jm_tau * r.Pbar) / jp.Twin))
    no_data = jp.theta1 == 0
    gap = (mk - a * sk) / mk if not no_data else mk - a * sk     # relative gap in  m_k = a_k s_k  (absolute if m_k = 0)
    tx = jp.slave_schedule
    on = (r.delta > 1e-7 * max(m.E.mean(), 1.0)) & tx            # shared AND slave transmits (local condition valid)
    gap = np.where(tx, gap, np.inf)                              # silent slots: energy is valued at a later slot, skip
    cap_slack = r.b_peak < jp.B_max - 1e-6                       # slave ceiling NOT reached (mu_bar = 0)
    capA = jp.capA
    stored = a * r.delta
    s5_bind = np.isfinite(capA) & (stored >= capA - 1e-6 * np.maximum(np.where(np.isfinite(capA), capA, 1.0), 1.0))
    cap_slack = cap_slack & ~s5_bind                             # S5 binds -> its multiplier is nonzero, skip the equality
    extra = {}
    xb = jm_tau * r.Pbar                                         # slave burst energy x_k
    if jp.tx_cap_J is not None:                                  # S6 binds -> the slave cannot spend more, skip
        tc = jp.txcap
        s6_bind = np.isfinite(tc) & (xb >= tc - 1e-6 * np.maximum(np.where(np.isfinite(tc), tc, 1.0), 1e-12))
        cap_slack = cap_slack & ~s6_bind
        extra["tx_cap_active_slots"] = int((s6_bind & tx).sum())
        extra["max_tx_cap_violation_J"] = float(max(np.max(np.where(np.isfinite(tc), xb - tc, -np.inf)), 0.0))
    if jp.tx_floor_J is not None:                                # S7
        tf = jp.txfloor
        extra["tx_floor_active_slots"] = int((tx & (xb <= tf + 1e-6 * np.maximum(tf, 1e-12)) & (tf > 0)).sum())
        extra["max_tx_floor_violation_J"] = float(max(np.max(np.where(tx, tf - xb, -np.inf)), 0.0))
    if jp.delta_cap_J is not None:                               # M5 binds -> the master cannot send more, skip
        dc = jp.dcap
        m5_bind = np.isfinite(dc) & (r.delta >= dc - 1e-6 * np.maximum(np.where(np.isfinite(dc), dc, 1.0), 1e-12))
        cap_slack = cap_slack & ~m5_bind
        extra["delta_cap_active_slots"] = int(m5_bind.sum())
        extra["max_delta_cap_violation_J"] = float(max(np.max(np.where(np.isfinite(dc), r.delta - dc, -np.inf)), 0.0))
    etol = 1e-9 * max(m.E.mean(), 1.0)
    if jp.master_floor_J is not None:                            # M6: master reserve floor
        fL = master_floor_profile(jp)
        extra["master_floor_binding_slots"] = int((r.F[:-1] <= fL[1:-1] + etol).sum())
        extra["min_master_floor_margin_J"] = float((r.F[:-1] - fL[1:-1]).min()) if m.N > 1 else 0.0
    if np.isfinite(m.E_max) and m.N > 1:                         # M2: master battery full on the next arrival
        extra["master_full_slots"] = int((r.F[:-1] >= m.cap - etol).sum())
    if no_data:                                                  # master out of energy (xm = 0) also stops delta
        xm = jm_tau * r.P
        free = cap_slack & (xm > 1e-9 * max(m.E.mean(), 1.0))
        v_on = np.abs(gap[on & free]).max() if np.any(on & free) else 0.0
        v_off = np.maximum(-gap[~on & free], 0.0).max() if np.any(~on & free) else 0.0
        extra["master_has_no_data"] = True
        extra["unused_spill_J"] = float(xm.sum())
    else:
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
        **extra,
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
        master floor       cum(E)_k - cum(x+dl)_k >= fL[k+1], and fL[N] instead of 0 at the end
                                                                      M6  (only if jp.master_floor_J is set)
        slave floor        b0 + cum(a*dl - xb - c)_k >= B_min         S1  (b_k >= B_min)
        slave ceiling      b0 + cum(a*dl)_k - cum(xb + c)_{k-1} <= B_max  S2  (b_{k-1} + a_k dl_k <= B_max)
        slave terminal     b0 + cum(a*dl - xb - c)_{N-1} == B_min    g[N] = 0 (same as joint_dwf)
        MPPT input cap     a_k*dl_k <= capA_k                         S5  (only if jp.mppt_cap_J is set)
        burst cap          xb_i <= txcap_i                            S6  (only if jp.tx_cap_J is set)
        burst floor        xb_i >= txfloor_i  (transmit slots)        S7  (only if jp.tx_floor_J is set)
        laser cap          dl_k <= dcap_k                             M5  (only if jp.delta_cap_J is set)

    c_i = jp.overhead (slave non-TX electronics energy per slot; zeros when no load is attached).

    Objective: th1*sum (tau/2) ln(1 + h*x/tau) + th2*sum (tau/2) ln(1 + hbar*xb/tau);  with
    jp.slave_rate_window set the slave term is th2*sum (T_i/2) ln(1 + hbar_i*xb_i/T_i) (burst rate).
    Returns P = x/tau, Pbar = xb/tau, delta, and the optimal objective value.
    """
    import cvxpy as cp
    m = jp.master
    N, tau, E, h, hb, a = m.N, m.tau, m.E, m.h, jp.h_bar, jp.a
    cS, capA = jp.overhead, jp.capA
    x, dl, xb = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
    cumE = np.cumsum(E)                                                 # cumulative harvest sum_{i<=k} E_i
    cum = lambda v: cp.cumsum(v)                                        # cumulative sum of a cvxpy vector
    fL = master_floor_profile(jp)                                       # M6 floor (zeros when off)
    cons = [cum(x + dl) <= cumE - fL[1:],                               # master causality (+ floor)
            cum(x + dl)[-1] == cumE[-1] - fL[-1],
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
    if jp.tx_cap_J is not None and np.isfinite(jp.txcap).any():        # S6: burst energy cap
        fin = np.flatnonzero(np.isfinite(jp.txcap))
        cons.append(xb[fin] <= jp.txcap[fin])
    if jp.tx_floor_J is not None:                                       # S7: burst energy floor (transmit slots)
        on = np.flatnonzero(jp.slave_schedule)
        cons.append(xb[on] >= jp.txfloor[on])
    if jp.delta_cap_J is not None and np.isfinite(jp.dcap).any():      # M5: laser cap on delta
        fin = np.flatnonzero(np.isfinite(jp.dcap))
        cons.append(dl[fin] <= jp.dcap[fin])
    if jp.slave_rate_window is None:
        slave_term = cp.sum(0.5 * tau * cp.log(1 + cp.multiply(hb, xb) / tau))
    else:                                                               # burst rate (T_i/2) ln(1 + hbar x / T_i)
        T = jp.Twin
        slave_term = cp.sum(cp.multiply(0.5 * T, cp.log(1 + cp.multiply(hb / T, xb))))
    obj = (jp.theta1 * cp.sum(0.5 * tau * cp.log(1 + cp.multiply(h, x) / tau))
           + jp.theta2 * slave_term)
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if x.value is None:                                                  # status infeasible / unbounded
        raise InfeasibleSlaveBudget(f"reference problem is {prob.status}: S1-S5 with g[N] = 0 cannot be satisfied")
    return x.value / tau, xb.value / tau, dl.value, float(prob.value)


# ----------------------------------------------------------------------------
# Randomised cross-check of joint_dwf against reference_solution_joint (stage-6 gate)
# ----------------------------------------------------------------------------
PARITY_FAMILIES = ("default", "tight_window", "per_slot_beta", "finite_E_max", "s6_binding",
                   "s6_s5_binding", "rate_window", "theta1_zero", "integrated_like", "master_floor")


def random_parity_case(rng: np.random.Generator, family: str) -> JointParams:
    """One random problem of `family` (N = 8..14, tau = 10 s, random per-slot h, hbar, alpha, E, weights).

    default        generous slave window, all slots transmit
    tight_window   B_max - B_min in [1, 10] J
    per_slot_beta  beta_i in [0.3, 0.95]
    finite_E_max   E_max in [1.05, 2.5] x max(E)
    s6_binding     S6 burst cap tx_cap_J in [0.5, 4] J per slot
    s6_s5_binding  S6 as above + S5 MPPT cap in [1, 6] J per slot
    rate_window    burst rate window T_i in [0.005, 1] x tau
    theta1_zero    master has no data (theta1 = 0), window in [2, 20] J
    integrated_like  theta1 = 0, random event schedule, per-slot drain c_i, S6 in event slots, S5,
                   T_i = 0.01 tau, small window: the stage-6 structure at test scale
    master_floor   M6: E_max in [1.5, 3] x max(E), floor in [0.2, 0.8] x (E_max - max E), theta1 = 0 in half the cases,
                   per-slot slave drain c_i (just-in-time start, floor ramp after the slave's need)
    """
    from build_dwf import MasterParams
    N = int(rng.integers(8, 15))
    tau = 10.0
    E = rng.uniform(2.0, 40.0, N)
    h = rng.uniform(0.02, 0.3, N)
    hb = h * rng.uniform(1.5, 6.0, N)
    kw = dict(alpha=rng.uniform(0.6, 1.0, N), beta=0.8, B_min=1.0, B_max=1.0 + 2 * E.sum(), h_bar=hb,
              theta1=float(rng.uniform(0.5, 2.0)), theta2=float(rng.uniform(1.0, 4.0)))
    E_max = np.inf
    if family == "tight_window":
        kw["B_max"] = 1.0 + float(rng.uniform(1.0, 10.0))
    elif family == "per_slot_beta":
        kw["beta"] = rng.uniform(0.3, 0.95, N)
    elif family == "finite_E_max":
        E_max = float(rng.uniform(1.05, 2.5) * E.max())
    elif family == "s6_binding":
        kw["tx_cap_J"] = rng.uniform(0.5, 4.0, N)
    elif family == "s6_s5_binding":
        kw["tx_cap_J"] = rng.uniform(0.5, 4.0, N)
        kw["mppt_cap_J"] = rng.uniform(1.0, 6.0, N)
    elif family == "rate_window":
        kw["slave_rate_window"] = tau * rng.uniform(0.005, 1.0, N)
    elif family == "theta1_zero":
        kw["theta1"] = 0.0
        kw["B_max"] = 1.0 + float(rng.uniform(2.0, 20.0))
    elif family == "integrated_like":
        sched = (rng.random(N) < 0.4).astype(int)
        sched[int(rng.integers(0, N))] = 1
        c = np.where(sched == 1, rng.uniform(0.2, 0.6, N), rng.uniform(0.01, 0.05, N))
        kw.update(theta1=0.0, slave_schedule=sched, slave_overhead_J=c,
                  tx_cap_J=np.where(sched == 1, rng.uniform(0.05, 0.5, N), 0.0),
                  mppt_cap_J=float(rng.uniform(2.0, 6.0)), slave_rate_window=0.01 * tau,
                  B_max=1.0 + float(rng.uniform(1.5, 4.0)) * c.max() + 0.5)
    elif family == "master_floor":
        E_max = float(rng.uniform(1.5, 3.0) * E.max())
        kw["master_floor_J"] = float(rng.uniform(0.2, 0.8) * (E_max - E.max()))
        kw["slave_overhead_J"] = rng.uniform(0.05, 0.5, N)
        if rng.random() < 0.5:
            kw["theta1"] = 0.0
            kw["B_max"] = 1.0 + float(rng.uniform(2.0, 20.0))
    elif family != "default":
        raise ValueError(f"unknown family {family!r} (known: {PARITY_FAMILIES})")
    return JointParams(master=MasterParams(E=E, E_max=E_max, tau=tau, h=h), **kw)


def parity_matrix(n_cases: int = 25, seed: int = 0, families=PARITY_FAMILIES, rtol: float = 1e-6,
                  verbose: bool = True) -> dict:
    """joint_dwf vs reference_solution_joint on n_cases random problems per family.

    A case is "exact" if |ref - dwf| <= rtol*|ref|, "below" if the DWF is lower by more, "above" if it is
    higher by more (must never happen: it would mean the DWF broke a constraint or cvxpy failed).
    Returns {family: dict(exact, below, above, infeasible, nonconverged, gaps=[(case, rel_gap)], binding)}.
    """
    from .solver import joint_dwf
    out = {}
    for fi, fam in enumerate(families):
        rng = np.random.default_rng([seed, fi])
        row = dict(exact=0, below=0, above=0, infeasible=0, nonconverged=0, gaps=[], above_cases=[], binding={})
        for ci in range(n_cases):
            jp = random_parity_case(rng, fam)
            try:
                r = joint_dwf(jp)
                _, _, _, ref = reference_solution_joint(jp)
            except InfeasibleSlaveBudget:
                row["infeasible"] += 1
                continue
            if not r.converged:
                row["nonconverged"] += 1
            k = check_kkt_joint(jp, r)
            for key in ("tx_cap_active_slots", "mppt_cap_active_slots", "arrival_cap_active_slots",
                        "master_floor_binding_slots", "master_full_slots"):
                if key in k:
                    row["binding"][key] = row["binding"].get(key, 0) + int(k[key] > 0)
            gap = (ref - r.objective) / max(abs(ref), 1e-12)
            if abs(gap) <= rtol:
                row["exact"] += 1
            elif gap > 0:
                row["below"] += 1
                row["gaps"].append((ci, float(gap)))
            else:
                row["above"] += 1
                row["above_cases"].append((ci, float(gap)))
        out[fam] = row
        if verbose:
            g = ", ".join(f"#{c}: {v:.2e}" for c, v in row["gaps"]) or "-"
            b = ", ".join(f"{k.replace('_active_slots', '')} {v}/{n_cases}" for k, v in row["binding"].items())
            print(f"{fam:15s} exact {row['exact']:2d}  below {row['below']:2d}  above {row['above']:2d}  "
                  f"infeasible {row['infeasible']}  nonconv {row['nonconverged']}  | binds in: {b}\n"
                  f"{'':15s} below-optimum gaps (relative): {g}"
                  + (f"\n{'':15s} ABOVE: {row['above_cases']}" if row["above"] else ""))
    return out


# ----------------------------------------------------------------------------
# Stage-6 refusal loop and maximal-sharing checks (gate of phase 5)
# ----------------------------------------------------------------------------
EVENT_KINDS = ("scarce", "zero_harvest", "tiny_supercap", "all_on", "single_on", "fixed_bias")


def small_event_problem(rng: np.random.Generator, N: int, kind: str = "scarce", n_events: int = 3):
    """Integrated-scale problem (tau = 60 s, event load and supercap from slave_config, theta1 = 0) whose master
    energy is SCARCE (fractions of a joule per slot), so events can be refused.  Returns (build, schedule):
    build(effective) -> JointParams with the event load (slave_load.attach_event_load) for that schedule.

    scarce        E_k ~ U(0, 0.6) J, about a third of the slots zero; n_events random events
    zero_harvest  everything harvested in slot 0 (E_0 ~ U(0.7, 1.5) J), nothing afterwards
    tiny_supercap 0.01 F supercap (W = 0.111 J < one event's 0.198 J): every event must be dropped
    all_on        an event in every slot, E_k ~ U(0, 0.6) J
    single_on     one event, E_k ~ U(0, 0.3) J
    fixed_bias    as scarce, photomixer in "fixed_bias" mode
    """
    import dataclasses as _dc
    import slave_load
    from build_dwf import MasterParams
    E = rng.uniform(0.0, 0.6, N) * (rng.random(N) > 0.33)
    E[0] = max(E[0], rng.uniform(0.2, 0.6))          # the supercap starts at B_min: slot 0's sleep load needs a harvest
    sched = np.zeros(N, dtype=int)
    sched[rng.choice(N, size=min(n_events, N), replace=False)] = 1
    kw = {}
    if kind == "zero_harvest":
        E = np.zeros(N)
        E[0] = rng.uniform(0.7, 1.5)                  # >= the sleep load of 10 slots (0.16 J) / a (0.36) with margin
    elif kind == "tiny_supercap":
        kw["supercap_f"] = 0.01
    elif kind == "all_on":
        sched[:] = 1
    elif kind == "single_on":
        E = rng.uniform(0.0, 0.3, N)
        E[0] = max(E[0], rng.uniform(0.2, 0.6))      # slot 0's sleep load needs a harvest (see above)
        sched[:] = 0
        sched[int(rng.integers(0, N))] = 1
    elif kind == "fixed_bias":
        kw["pm_mode"] = "fixed_bias"
    elif kind != "scarce":
        raise ValueError(f"unknown kind {kind!r} (known: {EVENT_KINDS})")
    jp0 = JointParams(master=MasterParams(E=E, E_max=np.inf, tau=60.0, h=1.0), alpha=1.0, beta=0.36,
                      theta1=0.0, slave_schedule=sched)

    def build(effective):
        return slave_load.attach_event_load(_dc.replace(jp0, slave_schedule=np.asarray(effective, int)), **kw)
    return build, sched


def brute_force_events(build, schedule, x_min: float, max_delay: int, mode: str = "proportional") -> dict:
    """Best placement of the released events over ALL choices: event j runs in one slot of
    [release_j, release_j + max_delay] (at most one event per slot) or is dropped.  A placement is feasible if
    joint_dwf solves it and (proportional) every placed event has x >= x_min.  Score = (events run, objective).
    Returns dict(n_run, objective, placement {release: slot or None}, n_tried)."""
    import itertools
    from .solver import joint_dwf
    sch = np.asarray(schedule, int)
    N = sch.size
    rel = [int(i) for i in np.flatnonzero(sch)]
    opts = [[None] + list(range(r0, min(N, r0 + max_delay + 1))) for r0 in rel]
    best = dict(n_run=-1, objective=-np.inf, placement=None, n_tried=0)
    for combo in itertools.product(*opts):
        slots = [s for s in combo if s is not None]
        if len(set(slots)) < len(slots):
            continue
        best["n_tried"] += 1
        eff = np.zeros(N, int)
        eff[slots] = 1
        try:
            jp = build(eff)
            r = joint_dwf(jp)
        except InfeasibleSlaveBudget:
            continue
        x = jp.master.tau * r.Pbar
        if mode == "proportional" and any(x[s] < x_min * (1 - 1e-9) for s in slots):
            continue
        score = (len(slots), r.objective)
        if score > (best["n_run"], best["objective"]):
            best.update(n_run=len(slots), objective=r.objective, placement=dict(zip(rel, combo)))
    return best


def max_level_lp(jp: JointParams, x) -> np.ndarray:
    """Pointwise-maximal slave level check: maximise sum_k b_k over delta with the bursts x fixed (scipy LP).
        b_k = b0 + cum(a delta - c - x)_k >= B_min ;  b_{k-1} + a_k delta_k <= B_max ;  a delta <= capA ;
        delta <= Dc ;  cum(delta) <= cum(E)   (E_max = inf; the master may spill).  Returns b (end of each slot)."""
    from scipy.optimize import linprog
    m = jp.master
    N, a = m.N, jp.a
    if np.isfinite(m.E_max):
        raise ValueError("max_level_lp assumes E_max = inf")
    cx = np.cumsum(jp.overhead + np.asarray(x, float))
    L = np.tril(np.ones((N, N)))
    La = L * a
    Lm = np.tril(np.ones((N, N)), -1)
    A_ub = np.vstack([-La, La, L])
    b_ub = np.concatenate([jp.b0 - jp.B_min - cx, jp.B_max - jp.b0 + Lm @ (jp.overhead + np.asarray(x, float)), np.cumsum(m.E)])
    ub = np.minimum(np.where(a > 0, jp.capA / np.where(a > 0, a, 1.0), 0.0), jp.dcap)
    res = linprog(-(La.sum(axis=0)), A_ub=A_ub, b_ub=b_ub, bounds=list(zip(np.zeros(N), ub)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"max_level_lp: {res.message}")
    return jp.b0 + La @ res.x - cx


def check_maximal_sharing(jp: JointParams, r: JointResult, r2: JointResult, info: dict) -> dict:
    """Gate checks of the post-pass: objective and every x_i unchanged, each slot stopped by exactly one bound,
    delta_k = 0 where the arrival is already full, level inside [B_min, B_max], and the LP comparison."""
    from .solver import STOP_REASONS
    tau = jp.master.tau
    out = dict(applied=info["applied"])
    if not info["applied"]:
        out["reason"] = info["reason"]
        return out
    stop = info["stop"]
    full = np.array([s == "full" for s in stop])
    out["objective_change"] = abs(r2.objective - r.objective)
    out["max_x_change_J"] = float(np.max(np.abs(tau * (r2.Pbar - r.Pbar))))
    out["every_slot_stopped"] = bool(all(s in STOP_REASONS + ("full",) for s in stop) and len(stop) == jp.master.N)
    out["stop_counts"] = {s: stop.count(s) for s in sorted(set(stop))}
    out["max_delta_in_full_slots_J"] = float(np.max(r2.delta[full])) if full.any() else 0.0
    out["min_level_margin_J"] = float(r2.b.min() - jp.B_min)
    out["max_peak_over_B_max_J"] = float(r2.b_peak.max() - jp.B_max)
    out["ledger_error_J"] = float(jp.b0 + np.sum(jp.a * r2.delta) - jp.overhead.sum() - tau * r2.Pbar.sum() - r2.b[-1])
    try:
        b_lp = max_level_lp(jp, tau * r.Pbar)
        out["lp_sum_b_minus_greedy_J"] = float(b_lp.sum() - r2.b.sum())
        out["min_greedy_minus_lp_J"] = float(np.min(r2.b - b_lp))
    except (RuntimeError, ValueError) as e:
        out["lp"] = str(e)
    return out


def refusal_gate(n_cases: int = 20, seed: int = 0, max_delay: int = 3, verbose: bool = True) -> dict:
    """Phase-5 gate over EVENT_KINDS (n_cases each, N = 7..10): greedy refusal loop vs brute force (all kinds but
    all_on, whose brute force is combinatorial), event bookkeeping (run + rescheduled + dropped = released, delay <=
    max_delay), and the maximal-sharing post-pass checks.  Returns {kind: summary}; summary["failures"] lists every
    broken invariant (the gate passes when all are empty; greedy-vs-brute-force gaps are reported, not failures)."""
    import slave_load
    from .physical import solve_with_refusals
    from .solver import maximal_sharing
    x_min = slave_load.tx_bounds_J()["x_min"]
    out = {}
    for ki, kind in enumerate(EVENT_KINDS):
        rng = np.random.default_rng([seed, ki])
        s = dict(cases=0, converged=0, released=0, run=0, rescheduled=0, dropped=0, bf_match=0, bf_more=0, bf_gaps=[],
                 postpass_applied=0, max_obj_change=0.0, max_x_change=0.0, max_full_delta=0.0, min_margin=np.inf,
                 max_over=-np.inf, max_lp_gap=0.0, failures=[])
        for ci in range(n_cases):
            N = int(rng.integers(7, 11))
            build, sch = small_event_problem(rng, N, kind)
            mode = "fixed_bias" if kind == "fixed_bias" else "proportional"
            g = solve_with_refusals(build, sch, x_min, max_delay, mode=mode, calibrate=False)
            s["cases"] += 1
            c = g.counts
            for o in ("released", "run", "rescheduled", "dropped"):
                s[o] += c[o]
            if not g.converged:
                s["failures"].append(f"#{ci}: not converged ({g.history[-1]['reason'][:80]})")
                continue
            s["converged"] += 1
            if c["run"] + c["rescheduled"] + c["dropped"] != c["released"]:
                s["failures"].append(f"#{ci}: bookkeeping {c}")
            if any(e["delay"] is not None and e["delay"] > max_delay for e in g.events):
                s["failures"].append(f"#{ci}: delay > max_delay")
            if any(e["outcome"] in ("run", "rescheduled") and mode == "proportional" and e["x_J"] < x_min * (1 - 1e-9)
                   for e in g.events):
                s["failures"].append(f"#{ci}: a run event is below x_min")
            if kind != "all_on":
                bf = brute_force_events(build, sch, x_min, max_delay, mode)
                n_g = c["run"] + c["rescheduled"]
                if bf["n_run"] == n_g and abs(bf["objective"] - g.r.objective) <= 1e-9 * max(1.0, abs(bf["objective"])):
                    s["bf_match"] += 1
                else:
                    s["bf_more"] += 1
                    s["bf_gaps"].append((ci, bf["n_run"] - n_g, float(bf["objective"] - g.r.objective)))
            r2, info = maximal_sharing(g.jp, g.r)
            ck = check_maximal_sharing(g.jp, g.r, r2, info)
            if not ck["applied"]:
                s["failures"].append(f"#{ci}: post-pass not applied ({ck['reason']})")
                continue
            s["postpass_applied"] += 1
            s["max_obj_change"] = max(s["max_obj_change"], ck["objective_change"])
            s["max_x_change"] = max(s["max_x_change"], ck["max_x_change_J"])
            s["max_full_delta"] = max(s["max_full_delta"], ck["max_delta_in_full_slots_J"])
            s["min_margin"] = min(s["min_margin"], ck["min_level_margin_J"])
            s["max_over"] = max(s["max_over"], ck["max_peak_over_B_max_J"])
            s["max_lp_gap"] = max(s["max_lp_gap"], abs(ck.get("lp_sum_b_minus_greedy_J", 0.0)), -ck.get("min_greedy_minus_lp_J", 0.0))
            if (ck["objective_change"] > 1e-9 or ck["max_x_change_J"] > 1e-9 or not ck["every_slot_stopped"]
                    or ck["max_delta_in_full_slots_J"] > 0 or ck["min_level_margin_J"] < -1e-12
                    or ck["max_peak_over_B_max_J"] > 1e-12 or abs(ck.get("lp_sum_b_minus_greedy_J", 0.0)) > 1e-9
                    or ck.get("min_greedy_minus_lp_J", 0.0) < -1e-9):
                s["failures"].append(f"#{ci}: post-pass check {ck}")
        out[kind] = s
        if verbose:
            bf = "n/a (combinatorial)" if kind == "all_on" else f"{s['bf_match']}/{s['converged']} match" + (
                f", gaps (case, extra events, objective) {s['bf_gaps']}" if s["bf_gaps"] else "")
            print(f"{kind:13s} converged {s['converged']}/{s['cases']}  events released {s['released']}: run {s['run']}, "
                  f"rescheduled {s['rescheduled']}, dropped {s['dropped']} | brute force {bf}\n"
                  f"{'':13s} post-pass {s['postpass_applied']}/{s['converged']}: |d obj| <= {s['max_obj_change']:.1e}, "
                  f"|d x| <= {s['max_x_change']:.1e} J, delta in full slots <= {s['max_full_delta']:.1e} J, "
                  f"level margin >= {s['min_margin']:.1e} J, peak - B_max <= {s['max_over']:.1e} J, LP gap <= {s['max_lp_gap']:.1e} J"
                  + "".join(f"\n{'':13s} FAIL {f}" for f in s["failures"]))
    return out


def stage6_checks(n_cases: int = 20, seed: int = 0, solve_sizes=(1, 10), on_sizes=(1, 10, 100), verbose: bool = True) -> dict:
    """Phase-7 checks of the stage-6 model (implementation.md section 7).

    presets      THZ_PRESET "0.3THz" vs "0.1THz": the attached JointParams differ ONLY in h_bar.
    invariants   n_cases scarce random event problems (joint_dwf, fixed schedule): objective non-decreasing in
                 x_cap, B_max (supercap), a (beta) and ETA_SC (through beta); non-increasing in d_thz and c;
                 slave ledger closes; Pbar = 0 in sleep slots; x <= x_cap; supercap inside [B_min, B_max].
    scalability  integrated problem repeated n_periods times: joint_dwf runtime and objective per period
                 (solve_sizes), and the O(N) parts (event load, queue, fullest-trajectory pass, link table)
                 timed separately (on_sizes).
    Returns dict(presets, invariants, scalability); "failures" lists every broken check."""
    import dataclasses as _dc
    import time
    import types
    import slave_load
    from dwf_diagnostics import link_table
    from .demos import _integrated_problem
    from .solver import fullest_trajectory, joint_dwf
    out = dict(failures=[])

    # ---- presets: only hbar changes ----
    jps = {}
    for p in ("0.3THz", "0.1THz"):
        jp0, build, link = _integrated_problem(preset=p)
        jps[p] = (build(jp0.slave_schedule), link)
    a, b = jps["0.3THz"][0], jps["0.1THz"][0]
    differ = []
    for f in _dc.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if _dc.is_dataclass(va):
            continue
        same = (np.array_equal(np.asarray(va), np.asarray(vb), equal_nan=True) if va is not None and vb is not None
                else va is vb)
        if not same:
            differ.append(f.name)
    out["presets"] = dict(fields_that_differ=differ, hbar={p: float(np.mean(v[0].h_bar)) for p, v in jps.items()},
                          clamped={p: v[1]["clamped"] for p, v in jps.items()})
    if differ != ["h_bar"]:
        out["failures"].append(f"presets: fields that differ {differ} (expected only h_bar)")

    # ---- invariants on scarce random event problems ----
    rng = np.random.default_rng(seed)
    tb = slave_load.tx_bounds_J()
    inv = dict(cases=0, infeasible=0, mono_fail=[], ledger_max=0.0, silent_pbar_max=0.0, cap_excess_max=0.0,
               level_violation_max=0.0)

    def solve(jp):
        try:
            return joint_dwf(jp)
        except InfeasibleSlaveBudget:
            return None

    for ci in range(n_cases):
        N = int(rng.integers(7, 11))
        build, sch = small_event_problem(rng, N, "scarce")
        jp = build(sch)
        r = solve(jp)
        if r is None:
            inv["infeasible"] += 1
            continue
        inv["cases"] += 1
        tau = jp.master.tau
        x = tau * r.Pbar
        inv["ledger_max"] = max(inv["ledger_max"], abs(check_kkt_joint(jp, r)["slave_ledger_error_J"]))
        inv["silent_pbar_max"] = max(inv["silent_pbar_max"], float(np.max(np.abs(x[~jp.slave_schedule]), initial=0.0)))
        inv["cap_excess_max"] = max(inv["cap_excess_max"], float(np.max(x - jp.txcap)))
        inv["level_violation_max"] = max(inv["level_violation_max"], float(max(jp.B_min - r.b.min(), r.b_peak.max() - jp.B_max)))
        lk = slave_load.thz_link()
        far = slave_load.thz_link(d_m=lk["d_m"] * 1.2)
        variants = {   # (direction, modified problem):  +1 objective must not fall, -1 must not rise
            "x_cap x2": (+1, _dc.replace(jp, tx_cap_J=jp.txcap * 2)),
            "B_max (0.1 F)": (+1, slave_load.attach_event_load(_dc.replace(jp, slave_schedule=sch), supercap_f=0.1,
                                                               link=lk)),
            "a: beta x1.1": (+1, _dc.replace(jp, beta=np.minimum(jp.beta * 1.1, 1.0))),
            "ETA_SC 0.95 -> 1": (+1, _dc.replace(jp, beta=np.minimum(jp.beta / 0.95, 1.0))),
            "d_thz x1.2": (-1, _dc.replace(jp, h_bar=far["hbar"])),
            "c x1.2": (-1, _dc.replace(jp, slave_overhead_J=jp.overhead * 1.2)),
        }
        for name, (sign, jv) in variants.items():
            rv = solve(jv)
            ov = -np.inf if rv is None else rv.objective
            if sign * (ov - r.objective) < -1e-9 * max(1.0, abs(r.objective)):
                inv["mono_fail"].append((ci, name, float(r.objective), float(ov)))
    out["invariants"] = inv
    if inv["mono_fail"]:
        out["failures"].append(f"invariants: monotonicity broken {inv['mono_fail']}")
    for key, lim in (("ledger_max", 1e-11), ("silent_pbar_max", 1e-12), ("cap_excess_max", 1e-12), ("level_violation_max", 1e-9)):
        if inv[key] > lim:
            out["failures"].append(f"invariants: {key} = {inv[key]:.3g} > {lim:g}")

    # ---- scalability ----
    sc = dict(solve=[], on=[])
    for nper in solve_sizes:
        jp0, build, link = _integrated_problem(n_periods=nper)
        jp = build(jp0.slave_schedule)
        t = time.perf_counter()
        r = joint_dwf(jp, tol=1e-15)
        sc["solve"].append(dict(N=jp.master.N, seconds=time.perf_counter() - t, iters=r.iters, converged=r.converged,
                                objective_per_period=r.objective / nper))
    per = [s["objective_per_period"] for s in sc["solve"]]
    if max(per) - min(per) > 1e-9 * max(1.0, abs(per[0])):
        out["failures"].append(f"scalability: objective per period differs {per}")
    for nper in on_sizes:
        jp0, build, link = _integrated_problem(n_periods=nper)
        sch = np.asarray(jp0.slave_schedule, int)
        N = sch.size
        t0 = time.perf_counter()
        jp = build(sch)                                         # event load + attach
        t1 = time.perf_counter()
        rel = {i: i for i in np.flatnonzero(sch)}
        refused = list(np.flatnonzero(sch))[::10]
        slave_load.reschedule(sch, refused, rel, 5)             # queue
        t2 = time.perf_counter()
        x = np.where(sch == 1, tb["x_cap"], 0.0)
        ft = fullest_trajectory(jp, x)                          # post-pass forward pass
        t3 = time.perf_counter()
        fake = types.SimpleNamespace(Pbar=x / jp.master.tau)
        events = [dict(release=int(i), run=int(i), delay=0, x_J=float(x[i]), outcome="run") for i in np.flatnonzero(sch)]
        link_table(jp, fake, events, link, tb["t_tx"], tb["x_min"], tb["x_cap"])
        t4 = time.perf_counter()
        sc["on"].append(dict(N=N, load_s=t1 - t0, queue_s=t2 - t1, postpass_s=t3 - t2, link_table_s=t4 - t3,
                             feasible=ft["fail_slot"] is None))
    out["scalability"] = sc

    if verbose:
        p = out["presets"]
        print(f"presets     : fields that differ between 0.3THz and 0.1THz: {p['fields_that_differ']} | hbar "
              + ", ".join(f"{k} {v:.5g}" for k, v in p["hbar"].items()) + f" | near-field clamp {p['clamped']}")
        print(f"invariants  : {inv['cases']} cases ({inv['infeasible']} infeasible skipped), monotonicity failures "
              f"{len(inv['mono_fail'])} | ledger <= {inv['ledger_max']:.1e} J | Pbar in sleep slots <= "
              f"{inv['silent_pbar_max']:.1e} J | x - x_cap <= {inv['cap_excess_max']:.1e} J | level outside window "
              f"<= {inv['level_violation_max']:.1e} J")
        for s in sc["solve"]:
            print(f"scalability : N = {s['N']:5d}  joint_dwf {s['seconds']:8.2f} s, iters {s['iters']}, converged "
                  f"{s['converged']}, objective per 60-slot period {s['objective_per_period']:.9f}")
        for s in sc["on"]:
            print(f"O(N) parts  : N = {s['N']:5d}  event load {s['load_s'] * 1e3:8.2f} ms | queue {s['queue_s'] * 1e3:7.2f} ms | "
                  f"post-pass {s['postpass_s'] * 1e3:7.2f} ms | link table {s['link_table_s'] * 1e3:7.2f} ms")
        for f in out["failures"]:
            print("FAIL", f)
    return out
