"""The joint master + slave directional water-filling solver, joint_dwf(), and its start points.

Algorithm: sections 4-6 of the build_2D_dwf.py module docstring.
"""
from __future__ import annotations

import numpy as np

from .params import JointParams, JointResult


# ----------------------------------------------------------------------------
# Feasible start when the slave has a known drain c_i > 0
# ----------------------------------------------------------------------------
class InfeasibleSlaveBudget(ValueError):
    """No schedule can keep the slave alive: the slave drain c_i, the battery window W, the MPPT cap
    or the master's harvest make S1-S5 (with g[N] = 0) unsatisfiable.  Constraints are never relaxed.

    .slots : the slots named by the just-in-time start as breaking a bound (sorted; () if unknown).  The
             stage-6 refusal loop (physical.solve_with_refusals) refuses the event at or before the first one."""
    slots: tuple = ()


def _jit_start(jp: "JointParams"):
    """Just-in-time start: the master sends in slot k exactly what the slave is short of there.

        f = 0 (master spends what it harvests),  xs = 0 (no slave data energy yet)
        d[k]   = max(0, c_k - g[k]) / a_k
        g[k+1] = g[k] + a_k d[k] - c_k          (>= 0 by construction)
    Any initial excess beyond the total drain (d == 0 everywhere) is spent as data in the first ON slot.
    With S7 (jp.tx_floor_J) every transmit slot starts AT its floor (the floor is fed like a drain, so
    xs = floor); S6 (burst cap) and M5 (laser cap) are checked like the other bounds.
    Returns (d, g, violations); violations = [(slot, reason)] for bounds the start breaks (empty = feasible).
    """
    m = jp.master
    N, E, a, cS, capA, W = m.N, m.E, jp.a, jp.overhead, jp.capA, jp.W
    if jp.tx_floor_J is not None:                    # S7: feed the floor like a drain -> xs(k) = floor_k
        cS = cS + np.where(jp.slave_schedule, jp.txfloor, 0.0)
    d = np.zeros(N)
    g = np.zeros(N + 1)
    g[0] = jp.g_init
    viol = []
    for k in range(N):
        short = max(0.0, cS[k] - g[k])
        if short > 0.0:
            if a[k] <= 0.0:
                viol.append((k, f"a_{k} = 0 but the slave is short {short:.6g} J (c_{k} = {cS[k]:.6g} J, carry {g[k]:.6g} J): no energy can land"))
                g[k + 1] = 0.0
                continue
            d[k] = short / a[k]
            g[k + 1] = 0.0                                   # g[k] + a_k d_k - c_k = 0 exactly
        else:
            g[k + 1] = g[k] - cS[k]
        if d[k] > E[k] * (1 + 1e-12) + 1e-12:
            viol.append((k, f"harvest too small: the slave needs delta_{k} = {d[k]:.6g} J but the master harvests E_{k} = {E[k]:.6g} J "
                            f"(with f = 0; earlier carry is tried by the LP fallback)"))
        if g[k] + a[k] * d[k] > W * (1 + 1e-12) + 1e-12:
            viol.append((k, f"window W = {W:.6g} J is below the energy needed on arrival in slot {k}: {g[k] + a[k] * d[k]:.6g} J"
                            f" (c_{k} = {cS[k]:.6g} J)"))
        if a[k] * d[k] > capA[k] * (1 + 1e-12) + 1e-12:
            viol.append((k, f"MPPT cap capA_{k} = {capA[k]:.6g} J is below the stored energy needed {a[k] * d[k]:.6g} J (c_{k} = {cS[k]:.6g} J)"))
        if jp.delta_cap_J is not None and d[k] > jp.dcap[k] * (1 + 1e-12) + 1e-12:
            viol.append((k, f"laser cap delta_cap_{k} = {jp.dcap[k]:.6g} J is below the energy needed {d[k]:.6g} J (c_{k} = {cS[k]:.6g} J)"))
    left = g[N]
    if left > 1e-12 * max(1.0, jp.g_init):
        j = int(np.argmax(jp.slave_schedule)) if jp.slave_schedule.any() else -1
        if j < 0:
            viol.append((N - 1, f"initial charge exceeds the total drain by {left:.6g} J and no ON slot can spend it"))
        elif jp.tx_cap_J is not None and jp.txfloor[j] + left > jp.txcap[j] * (1 + 1e-12) + 1e-12:
            viol.append((j, f"initial excess {left:.6g} J does not fit under the burst cap tx_cap_{j} = {jp.txcap[j]:.6g} J"))
        else:
            g[j + 1:] -= left                               # xs(j) = left: spend it as data in the first ON slot
            g[N] = 0.0
    else:
        g[N] = 0.0
    return d, g, viol


def master_floor_profile(jp: "JointParams") -> np.ndarray:
    """M6 floor of the master carry, fL (length N+1):  f[k+1] = F_k >= fL[k+1],  f[N] = fL[N] (fixed end).

        fL[0]   = 0                                                      (nothing arrives before slot 0)
        fL[k+1] = L_k = min(F_min, max(0, L_{k-1} + E_k - need_k))       (charge up to F_min, then hold it)
        fL[N]   = L_{N-1}                                                (the master ENDS holding its reserve)

    need_k = just-in-time delta of _jit_start (what the slave is short of in slot k, clipped to what the master
    has): the slave's minimum need is served first, the master's reserve second, sharing / spill come from what
    is above the floor.  L_k <= E_max - E_{k+1} holds because JointParams checks F_min <= min(cap).
    jp.master_floor_J None -> all zeros (the original f >= 0, f[N] = 0: bit-identical solver path)."""
    N = jp.master.N
    fL = np.zeros(N + 1)
    if jp.master_floor_J is None:
        return fL
    E, Fm = jp.master.E, jp.master_floor_J
    need = _jit_start(jp)[0]
    for k in range(N):
        avail = fL[k] + E[k]
        fL[k + 1] = min(Fm, max(0.0, avail - min(need[k], avail)))
    return fL


def _lp_start(jp: "JointParams"):
    """Feasibility LP (scipy) with the constraints of reference_solution_joint: a start that may borrow
    against future harvest through the master carry f.  Returns (d, f, g) or None if infeasible."""
    try:
        from scipy.optimize import linprog
    except ImportError:                                        # pragma: no cover
        return None
    m = jp.master
    N, E, a, cS, capA = m.N, m.E, jp.a, jp.overhead, jp.capA
    fL = master_floor_profile(jp)                             # M6 floor (zeros when off)
    tx = jp.slave_schedule
    nv = 3 * N                                               # variables: x (master data), d, xb (slave data)
    ix, idl, ixb = np.arange(N), N + np.arange(N), 2 * N + np.arange(N)
    L = np.tril(np.ones((N, N)))                              # cumulative-sum operator
    Lm = np.tril(np.ones((N, N)), -1)                         # strictly lower: cum up to k-1
    Z = np.zeros((N, N))
    A_ub = [np.hstack([L, L, Z]),                             # M1/M6: cum(x+d) <= cum(E) - floor
            np.hstack([Z, -L * a, L]),                        # S1: b0 + cum(a d - xb - c) >= B_min
            np.hstack([Z, L * a, -Lm])]                       # S2: b0 + cum(a d)_k - cum(xb + c)_{k-1} <= B_max
    b_ub = [np.cumsum(E) - fL[1:], jp.b0 - jp.B_min - np.cumsum(cS), jp.B_max - jp.b0 + Lm @ cS]
    if np.isfinite(m.E_max):                                  # M2: cum(E) - cum(x+d) <= E_max - E_{k+1}
        A_ub.append(np.hstack([-L[:-1], -L[:-1], Z[:-1]]))
        b_ub.append(m.E_max - E[1:] - np.cumsum(E)[:-1])
    A_eq = np.vstack([np.concatenate([np.ones(N), np.ones(N), np.zeros(N)]),     # master uses all but its end floor
                      np.concatenate([np.zeros(N), a, -np.ones(N)])])              # slave terminal g[N] = 0
    b_eq = np.array([E.sum() - fL[N], jp.B_min - jp.b0 + cS.sum()])
    bounds = [(0, None)] * (2 * N) + [(0, 0) if not tx[i] else (0, None) for i in range(N)]
    for i in range(N):                                        # S5 as an upper bound on d
        if np.isfinite(capA[i]):
            bounds[N + i] = (0, capA[i] / a[i] if a[i] > 0 else 0.0)
    if jp.delta_cap_J is not None:                            # M5: delta_k <= laser cap
        for i in range(N):
            if np.isfinite(jp.dcap[i]):
                ub = bounds[N + i][1]
                bounds[N + i] = (0, jp.dcap[i] if ub is None else min(ub, jp.dcap[i]))
    if jp.tx_cap_J is not None or jp.tx_floor_J is not None:  # S6 / S7 on the slave burst energy
        for i in np.flatnonzero(tx):
            bounds[2 * N + i] = (float(jp.txfloor[i]), float(jp.txcap[i]) if np.isfinite(jp.txcap[i]) else None)
    res = linprog(np.zeros(nv), A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    x, dl, xb = res.x[ix], res.x[idl], res.x[ixb]
    d = np.maximum(dl, 0.0)
    f = np.zeros(N + 1)
    f[1:] = np.maximum(np.cumsum(E - x - dl), fL[1:])
    f[N] = fL[N]
    g = np.zeros(N + 1)
    g[0] = jp.g_init
    g[1:] = np.maximum(jp.g_init + np.cumsum(a * dl - xb - cS), 0.0)
    g[N] = 0.0
    return d, f, g


# ----------------------------------------------------------------------------
# Core solver
# ----------------------------------------------------------------------------
def warm_start_from(jp: JointParams, r: JointResult):
    """(f, g, d) flows of a previous result, for joint_dwf(..., x0=...):  f[k+1] = F_k, g[k+1] = b_k - B_min,
    with f[0] = 0, f[N] = fL[N] (0 without a master floor), g[0] = b0 - B_min, g[N] = 0 (the solver's fixed ends)."""
    N = jp.master.N
    f = np.zeros(N + 1)
    f[1:N] = r.F[:-1]
    f[N] = master_floor_profile(jp)[N]
    g = np.zeros(N + 1)
    g[0] = jp.g_init
    g[1:N] = r.b[:-1] - jp.B_min
    return f, g, np.asarray(r.delta, float).copy()


def _warm_start_violation(jp: JointParams, f, g, d, tol: float):
    """None if (f, g, d) satisfies every bound of joint_dwf to `tol` [J], else the first violated bound (text)."""
    m = jp.master
    N, E, a, cS = m.N, m.E, jp.a, jp.overhead
    if len(f) != N + 1 or len(g) != N + 1 or len(d) != N:
        return "wrong lengths"
    fL = master_floor_profile(jp)
    if abs(f[0]) > tol or abs(f[N] - fL[N]) > tol or abs(g[0] - jp.g_init) > tol or abs(g[N]) > tol:
        return "fixed ends f[0], f[N], g[0], g[N]"
    xm = E + f[:-1] - f[1:] - d
    xs = a * d + g[:-1] - g[1:] - cS
    tx = jp.slave_schedule
    checks = [("f >= 0", -f.min()), ("f >= master floor", np.max(fL - f)), ("f <= E_max - E_next", np.max(f[1:N] - m.cap) if N > 1 else -1.0),
              ("g >= 0", -g.min()), ("d >= 0", -d.min()), ("P >= 0", -xm.min()), ("Pbar >= 0", -xs.min()),
              ("Pbar = 0 in silent slots", np.max(np.abs(xs[~tx])) if (~tx).any() else -1.0),
              ("arrival cap", np.max(g[:-1] + a * d - jp.W)), ("S5", np.max(a * d - jp.capA)),
              ("S6", np.max(xs - jp.txcap)), ("S7", np.max(np.where(tx, jp.txfloor - xs, -1.0))),
              ("M5", np.max(d - jp.dcap))]
    for name, v in checks:
        if v > tol:
            return f"{name} broken by {v:.3g} J"
    return None


def joint_dwf(jp: JointParams, tol: float = 1e-10, max_iter: int = 1000,
              cross_run: bool | None = None, x0=None) -> JointResult:
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
    cross_run : add polish() family D (delta shift between NON-adjacent slots through the slave carry).
        None (default) = on only when a slave load c_i > 0 is attached, so the no-load path stays
        bit-identical to the original solver.
    x0 : optional warm start (f, g, d) (e.g. warm_start_from(jp_prev, r_prev)).  Used if it satisfies every
        bound of this jp to 1e-9*scale (tiny negatives are clipped); otherwise the cold start is used and
        JointResult.start says why.  None (default) = cold start, unchanged.

    Stage 6 options in jp (S6 tx_cap_J, S7 tx_floor_J, M5 delta_cap_J, slave_rate_window) add bounds to the
    moves and to apply(), and the rate window T_i to step(); see the comments at each move.

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
    cS = jp.overhead                              # c_i: slave non-TX electronics energy per slot [J] (zeros = no load)
    capA = jp.capA                                # S5: a_k*d_k <= capA_k (inf = no cap)
    has_load = bool(np.any(cS > 0))
    use_cross = has_load if cross_run is None else bool(cross_run)
    # ---- stage 6 options (all off by default; off = not one extra operation on the legacy path) -------
    has_s6 = jp.tx_cap_J is not None              # S6: xs(i) <= X[i]          (burst energy cap)
    has_s7 = jp.tx_floor_J is not None            # S7: xs(i) >= Lf[i], tx[i]  (burst energy floor)
    has_m5 = jp.delta_cap_J is not None           # M5: d[k]  <= Dc[k]         (laser cap)
    has_T = jp.slave_rate_window is not None      # slave rate (T_i/2) ln(1 + hbar x/T_i)
    X, Lf, Dc, Tw = jp.txcap, jp.txfloor, jp.dcap, jp.Twin
    Ts = (lambda i: Tw[i]) if has_T else (lambda i: None)   # rate window of SLAVE node i for step()
    has_s67 = has_s6 or has_s7

    # ---- state: the three families of flow variables (all in Joules) -------------------------
    fL = master_floor_profile(jp)                 # M6: f[q] >= fL[q] (all zeros without a master floor)
    f = fL.copy()                                 # f[k+1] = master carry over tap k
    #   f[k+1] = F_k = master energy carried from slot k to k+1;  f[0] = 0 and f[N] = fL[N] stay fixed
    #   (nothing arrives before slot 0; after slot N-1 only the master's reserve is left, 0 without a floor).
    #   Start "on the floor": the master holds its reserve, xm(k) = E_k + fL[k] - fL[k+1] >= need_k >= 0.
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
    start = "share-nothing"
    warm_note = ""
    if x0 is not None:
        f0, g0_, d0_ = (np.asarray(v, float) for v in x0)
        why = _warm_start_violation(jp, f0, g0_, d0_, 1e-9 * max(float(E.mean()), 1.0))
        if why is None:
            f[:], g[:], d[:] = np.maximum(f0, fL), np.maximum(g0_, 0.0), np.maximum(d0_, 0.0)
            f[0], f[N] = 0.0, fL[N]
            g[0], g[N] = jp.g_init, 0.0
        else:
            warm_note = f" (warm start rejected: {why})"
    if x0 is not None and not warm_note:
        start = "warm"
    elif has_load or has_s7 or (has_s6 and jp.g_init > 0):
        #   (S7 needs xs >= floor and S6 may not take the whole initial excess in one slot: same start path)
        #   With c > 0 "share nothing" is infeasible (g[N] = 0 is fixed): the slave must be fed to cover
        #   its drain.  Just-in-time start (module docstring section 5); if it breaks a bound (e.g. harvest
        #   too small in a slot with f = 0) try an LP that may borrow through f; otherwise raise.
        d0, g0, viol = _jit_start(jp)
        if not viol:
            d[:], g[:], start = d0, g0, "just-in-time"
        else:
            lp = _lp_start(jp)
            if lp is None:
                err = InfeasibleSlaveBudget(
                    "no feasible schedule keeps the slave alive (S1-S5, g[N] = 0): "
                    + "; ".join(f"slot {k}: {why}" for k, why in viol[:3])
                    + (f" (+{len(viol) - 3} more)" if len(viol) > 3 else "")
                    + ".  The LP feasibility check (borrowing via the master carry) is infeasible too.")
                err.slots = tuple(sorted({k for k, _ in viol}))
                raise err
            d[:], f[:], g[:] = lp
            start = "lp-fallback"
    start += warm_note

    # ---- energy balance: these two lambdas DEFINE the data energies -----------------------------
    xm = lambda i: E[i] + f[i] - f[i + 1] - d[i]          # master data energy
    #   tau*P_i = E_i + F_{i-1} - F_i - delta_i    (harvest + carry in - carry out - shared)
    xs = lambda i: a[i] * d[i] + g[i] - g[i + 1] - cS[i]  # slave  data energy
    #   tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i - c_i (arrival + carry in - carry out - electronics drain)

    scale = max(float(E.mean()), 1.0)             # typical energy per slot [J]; makes tol relative

    def step(thu, hu, xu, s_, thw, hw, xw, r_, Tu=None, Tw=None):
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

        Rate window (stage 6, slave_rate_window)
        ----------------------------------------
        With U(x) = (th*T/2) ln(1 + h*x/T) at a node (T = Tu, Tw; a burst of length T instead of tau) the
        marginal value is th h / (2(1 + h x/T)), so the balance is
            th_u h_u (1 + h_w (x_w + r t)/T_w) = r th_w h_w (1 + h_u (x_u - s t)/T_u),
        still linear in t:
            t* = [ r th_w h_w (1 + h_u x_u/T_u) - th_u h_u (1 + h_w x_w/T_w) ] / [ h_u h_w r (th_u/T_w + th_w s/T_u) ].
        With T_u = T_w = tau this is the expression above.  Tu = Tw = None (no window) keeps the original
        arithmetic exactly; a None end of a windowed call is a master end (T = tau).

        Parameters (per node): th = rate weight, h = link gain, x = current data energy [J],
        s_ / r_ = how many Joules of x the node loses / gains per Joule moved, Tu / Tw = rate window [s].
        Returns t* (positive = from u to w), UNCLIPPED. Returns 0 if both ends are worthless.
        """
        if Tu is not None or Tw is not None:
            Tu_ = tau if Tu is None else Tu
            Tw_ = tau if Tw is None else Tw
            den = hu * hw * r_ * (thu / Tw_ + thw * s_ / Tu_)
            if den <= 0:                   # both ends worthless (e.g. weight 0): nothing to gain
                return 0.0
            num = r_ * thw * hw * (1 + hu * xu / Tu_) - thu * hu * (1 + hw * xw / Tw_)
            return num / den
        den = hu * hw * (thu * r_ + r_ * thw * s_)
        if den <= 0:                       # both ends worthless (e.g. weight 0): nothing to gain
            return 0.0
        num = r_ * thw * hw * (1 + hu * xu / tau) - thu * hu * (1 + hw * xw / tau)
        return tau * num / den

    def slave_lim(lo, hi, j, xv, c):
        """S6 / S7 on the slave data energy of slot j:  Lf[j] <= xs(j) = xv + c*t <= X[j]  (S7 only where tx[j]).

        S6  xs(j) <= X[j]   burst at the photomixer cap      multiplier: kappa_j  (>= 0)
        S7  xs(j) >= Lf[j]  burst floor (transmit slots)      multiplier: iota_j   (>= 0)
        Both enter the slave stationarity as  s_j - kappa_j + iota_j = Lambda_bar_j, so a binding S6 lowers the value
        of a Joule landing in slot j to what the slave battery can carry on.  Returns the tightened [lo, hi]."""
        if c > 0:
            if has_s6: hi = min(hi, (X[j] - xv) / c)
            if has_s7 and tx[j]: lo = max(lo, (Lf[j] - xv) / c)
        elif c < 0:
            if has_s6: lo = max(lo, (X[j] - xv) / c)
            if has_s7 and tx[j]: hi = min(hi, (Lf[j] - xv) / c)
        return lo, hi

    def sx(j, coef):
        """apply() end-node tuple of SLAVE slot j: (xs(j), coef), plus its own [L, U] when S6 / S7 are on."""
        if not has_s67:
            return (xs(j), coef)
        return (xs(j), coef, Lf[j] if (has_s7 and tx[j]) else 0.0, X[j] if has_s6 else np.inf)

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
        * f[q]              in [fL[q], capM[q-1]]        M1/M6, M2  (master right tap q-1; fL = 0 without floor)
        * g[q]              >= 0                         S1      (slave floor)
        * d[q]              >= 0                         M3      (delta >= 0)
        * g[q] + a[q]*d[q]  <= W                         S2      (slave arrival cap at slot q)
        * a[q]*d[q]         <= capA[q]                   S5      (MPPT input rating, only if a cap is set)
        * d[q]              <= Dc[q]                     M5      (laser cap, only if jp.delta_cap_J is set)
        * src / snk data energy >= 0                     M3, S3  (P >= 0, Pbar >= 0)
          (a slave end may carry its own [L, U]:  S7 floor / S6 cap, see sx())

        Parameters
        ----------
        fch, gch, dch : lists of (index, coefficient c) for the f, g, d variables that move
        src, snk : (current data energy x, coefficient) of the SOURCE (loses) and SINK (gains) node,
                   or (x, coefficient, L, U) when that node has its own bounds (slave end with S6/S7)
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
            lim(f[q], c, fL[q], capM[q - 1])
        for q, c in gch:
            lim(g[q], c, 0.0, np.inf); arr[q] = arr.get(q, 0.0) + c
        for q, c in dch:
            lim(d[q], c, 0.0, np.inf); arr[q] = arr.get(q, 0.0) + a[q] * c
        for q, c in arr.items():
            lim(g[q] + a[q] * d[q], c, -np.inf, W)          # slave arrival cap
        for q, c in dch:
            lim(a[q] * d[q], a[q] * c, -np.inf, capA[q])    # S5: stored energy per slot <= capA_q
        if has_m5:
            for q, c in dch:
                lim(d[q], c, -np.inf, Dc[q])                # M5: delta_q <= laser cap
        lim(src[0], src[1], *(src[2:] if len(src) > 2 else (0.0, np.inf)))
        lim(snk[0], snk[1], *(snk[2:] if len(snk) > 2 else (0.0, np.inf)))
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

        Three families of paths are tried (plus family D when a slave load is attached)
        ---------------------------------
        A. master chain   master slot i -> master slot j   (through dead master slots)   f[i+1..j] += t
        B. slave chain    slave slot i  -> slave slot j    (through dead slave slots)    g[i+1..j] += t
        C. cross paths    master slot i -> master chain -> down tap k -> slave chain -> slave slot j
                          (delta_k += t, master carries and slave carries adjusted so that every
                          intermediate slot keeps its spending)

        D. delta shift   d[k] <-> d[j] (k+1 < j) through the slave carry g[k+1..j]  (cross_run only)
                          effect  : d[k] += t, g[k+1..j] += a_k t, d[j] -= (a_k/a_j) t ;
                                    xm(k) -= t, xm(j) += (a_k/a_j) t ; every xs(q) unchanged
                          equation: nu_k = (a_k/a_j)-weighted nu_j  (the same step() as move (10))
                          bounds  : d[k], d[j] >= 0, g >= 0, arrival caps S2 at k..j, S5, xm >= 0 (via apply)
                          why     : with a slave load, silent slots now drain, so energy is routed through runs
                                    of silent slots.  Move (10) only shifts between ADJACENT slots; an
                                    intermediate slot whose master level is high blocks the adjacent chain
                                    although the end-to-end shift pays (README section 10, same idea as A-C).

        Returns the largest |t| applied, so the caller knows whether anything moved.
        """
        big = 0.0
        if use_cross:
            for k in range(N - 2):                                  # D. delta shift across a run (k, j = k+2 ..)
                if a[k] <= 0:
                    continue
                for j in range(k + 2, N):
                    if a[j] <= 0:
                        continue
                    rr = a[k] / a[j]
                    t = step(th1, h[k], xm(k), 1, th1, h[j], xm(j), rr)
                    big = max(big, apply([], [(q, a[k]) for q in range(k + 1, j + 1)],
                                         [(k, 1.0), (j, -rr)], (xm(k), -1.0), (xm(j), rr), t))
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
                    t = step(th2, hb[i], xs(i), 1, th2, hb[j], xs(j), 1, Ts(i), Ts(j))  # equalise nubar_i, nubar_j
                    big = max(big, apply([], [(q, 1.0) for q in range(i + 1, j + 1)], [],
                                         sx(i, -1.0), sx(j, 1.0), t))
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
                    t = step(th1, h[i], xm(i), 1, th2, hb[j], xs(j), a[k], None, Ts(j))
                    big = max(big, apply(fch, gch, [(k, 1.0)],
                                         (xm(i), -1.0), sx(j, a[k]), t))
        return big

    # ---- main loop: sweeps over the slots -------------------------------------------------------
    it, converged = 0, False
    while it < max_iter:
        print(f"joint_dwf sweep {it} (start = {start})")
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
            #     stage 6 : xs(k) + a_k t <= X_k (S6, hi), >= Lf_k (S7, lo);  d[k] + t <= Dc_k (M5, hi);
            #               slave rate window T_k inside step() (burst rate)
            if ak > 0 and tx[k]:
                x, xb = xm(k), xs(k)
                t = step(th1, h[k], x, 1, th2, hb[k], xb, ak, None, Ts(k))
                lo, hi = max(-d[k], -xb / ak), min(x, (W - g[k]) / ak - d[k], capA[k] / ak - d[k])
                if has_s67: lo, hi = slave_lim(lo, hi, k, xb, ak)
                if has_m5: hi = min(hi, Dc[k] - d[k])
                t = min(max(t, lo), hi)
                d[k] += t; big = max(big, abs(t))
            # (2) IN-SWAP  M_k -> S_{k-1}: delta_k += t, carry g_{k-1} -= a_k t
            #     (needed when the slave arrival cap g_{k-1}+a_k delta_k <= W is tight)
            #     effect  : xm(k) -= t ; slave slot k unchanged (arrival +a_k t, carry-in -a_k t);
            #               slave slot k-1 spends a_k*t more (its carry-out g[k] fell by a_k t)
            #     equation: m_k = a_k * s_{k-1}   (value of the joule at the slave's EARLIER slot)
            #     bounds  : delta_k >= 0 and xs(k-1) >= 0 -> lo ;  xm(k) >= 0 and g[k] >= 0 -> hi
            #     the arrival cap g[k] + a_k delta_k is unchanged, so it never blocks this move
            #     stage 6 : S6/S7 on xs(k-1) (+a_k t), M5 on d[k]
            if ak > 0 and k >= 1 and tx[k - 1]:
                x, xb1 = xm(k), xs(k - 1)
                t = step(th1, h[k], x, 1, th2, hb[k - 1], xb1, ak, None, Ts(k - 1))
                lo, hi = max(-d[k], -xb1 / ak), min(x, g[k] / ak, capA[k] / ak - d[k])
                if has_s67: lo, hi = slave_lim(lo, hi, k - 1, xb1, ak)
                if has_m5: hi = min(hi, Dc[k] - d[k])
                t = min(max(t, lo), hi)
                d[k] += t; g[k] -= ak * t; big = max(big, abs(t))
            # (3) MASTER pass-through  M_{k-1} -> S_k  (f_{k-1}, delta_k up together)
            #     lets energy cross a master slot that has P_k = 0 on its way to the slave
            #     effect  : f[k] += t, d[k] += t ;  xm(k-1) -= t ; master slot k unchanged
            #               (carry-in +t, shared +t) ;  xs(k) += a_k*t
            #     equation: m_{k-1} = a_k * s_k
            #     bounds  : lo: f[k] >= fL[k] (M1/M6), delta_k >= 0, xs(k) >= 0
            #               hi: f[k] <= capM[k-1] (M2), xm(k-1) >= 0, slave arrival cap
            #     stage 6 : S6/S7 on xs(k) (+a_k t), M5 on d[k]
            if ak > 0 and k >= 1 and tx[k]:
                xa, xb = xm(k - 1), xs(k)
                t = step(th1, h[k - 1], xa, 1, th2, hb[k], xb, ak, None, Ts(k))
                lo = max(-(f[k] - fL[k]), -d[k], -xb / ak)
                hi = min(capM[k - 1] - f[k], xa, (W - g[k]) / ak - d[k], capA[k] / ak - d[k])
                if has_s67: lo, hi = slave_lim(lo, hi, k, xb, ak)
                if has_m5: hi = min(hi, Dc[k] - d[k])
                t = min(max(t, lo), hi)
                f[k] += t; d[k] += t; big = max(big, abs(t))
            # (4) MASTER out-swap  M_{k+1} -> S_k  (carry f_k down, delta_k up)
            #     effect  : f[k+1] -= t (undo carry), d[k] += t ; xm(k) unchanged ;
            #               xm(k+1) -= t ; xs(k) += a_k*t
            #     equation: m_{k+1} = a_k * s_k
            #     bounds  : lo: delta_k >= 0, xs(k) >= 0, f[k+1] <= capM[k] (when t < 0)
            #               hi: f[k+1] >= fL[k+1] (cannot undo carry below the floor), xm(k+1) >= 0, arrival cap
            #     stage 6 : S6/S7 on xs(k) (+a_k t), M5 on d[k]
            if ak > 0 and k <= N - 2 and tx[k]:
                xa, xb = xm(k + 1), xs(k)
                t = step(th1, h[k + 1], xa, 1, th2, hb[k], xb, ak, None, Ts(k))
                lo = max(-d[k], -xb / ak, -(capM[k] - f[k + 1]))
                hi = min(f[k + 1] - fL[k + 1], xa, (W - g[k]) / ak - d[k], capA[k] / ak - d[k])
                if has_s67: lo, hi = slave_lim(lo, hi, k, xb, ak)
                if has_m5: hi = min(hi, Dc[k] - d[k])
                t = min(max(t, lo), hi)
                f[k + 1] -= t; d[k] += t; big = max(big, abs(t))
            # (5) SLAVE pass-through  M_k -> S_{k+1} across slave slot k
            #     effect  : d[k] += t, g[k+1] += a_k t ; xm(k) -= t ; slave slot k unchanged
            #               (arrival +a_k t, carry-out +a_k t) ; xs(k+1) += a_k*t
            #     equation: m_k = a_k * s_{k+1}
            #     bounds  : lo: delta_k >= 0, g[k+1] >= 0, xs(k+1) >= 0
            #               hi: xm(k) >= 0, arrival cap at slot k AND at slot k+1
            #     stage 6 : S6/S7 on xs(k+1) (+a_k t), M5 on d[k]
            if ak > 0 and k <= N - 2 and tx[k + 1]:
                xa, xb = xm(k), xs(k + 1)
                t = step(th1, h[k], xa, 1, th2, hb[k + 1], xb, ak, None, Ts(k + 1))
                lo = max(-d[k], -g[k + 1] / ak, -xb / ak)
                hi = min(xa, (W - g[k]) / ak - d[k], (W - a[k + 1] * d[k + 1] - g[k + 1]) / ak,
                         capA[k] / ak - d[k])
                if has_s67: lo, hi = slave_lim(lo, hi, k + 1, xb, ak)
                if has_m5: hi = min(hi, Dc[k] - d[k])
                t = min(max(t, lo), hi)
                d[k] += t; g[k + 1] += ak * t; big = max(big, abs(t))
            if k == N - 1:
                continue                     # the last slot has no right tap: moves (6)-(9) need k+1
            # (6) MASTER right tap k -> k+1 : equalise master levels
            #     effect  : f[k+1] += t ; xm(k) -= t ; xm(k+1) += t
            #     equation: nu_k = nu_{k+1}  with nu = P + 1/h  (lambda_k = mu_k = 0)
            #     bounds  : f >= fL (closed tap / master at its floor, lambda_k) and f <= capM[k] (full tap, mu_k),
            #               plus xm(k) >= t and xm(k+1) >= -t (no negative power)
            xa, xb_ = xm(k), xm(k + 1)
            t = step(th1, h[k], xa, 1, th1, h[k + 1], xb_, 1)
            t = min(max(t, max(-(f[k + 1] - fL[k + 1]), -xb_)), min(capM[k] - f[k + 1], xa))
            f[k + 1] += t; big = max(big, abs(t))
            # (7) SLAVE right tap k -> k+1 : equalise slave levels
            #     effect  : g[k+1] += t ; xs(k) -= t ; xs(k+1) += t
            #     equation: nubar_k = nubar_{k+1}  with nubar = Pbar + 1/hbar
            #     bounds  : g >= 0 (slave at the floor, lambda_bar_k) and
            #               g[k+1] + a_{k+1} delta_{k+1} <= W (arrival cap of the NEXT slot, mu_bar_k)
            #     stage 6 : S6/S7 on xs(k) (-t) and xs(k+1) (+t)
            if tx[k] and tx[k + 1]:
                xa, xb_ = xs(k), xs(k + 1)
                t = step(th2, hb[k], xa, 1, th2, hb[k + 1], xb_, 1, Ts(k), Ts(k + 1))
                lo, hi = max(-g[k + 1], -xb_), min(W - a[k + 1] * d[k + 1] - g[k + 1], xa)
                if has_s67:
                    lo, hi = slave_lim(lo, hi, k, xa, -1.0)
                    lo, hi = slave_lim(lo, hi, k + 1, xb_, 1.0)
                t = min(max(t, lo), hi)
                g[k + 1] += t; big = max(big, abs(t))
            # (8) MASTER carry pass-through  M_{k-1} -> M_{k+1} across master slot k
            #     effect  : f[k] += t, f[k+1] += t ; xm(k-1) -= t ; xm(k+1) += t ; xm(k) unchanged
            #     equation: nu_{k-1} = nu_{k+1}  (slot k is dead, so it is skipped over)
            #     bounds  : both carries in [fL, cap], xm(k-1) >= 0, xm(k+1) >= 0
            if 1 <= k <= N - 2:
                xa, xb_ = xm(k - 1), xm(k + 1)
                t = step(th1, h[k - 1], xa, 1, th1, h[k + 1], xb_, 1)
                lo = max(-(f[k] - fL[k]), -(f[k + 1] - fL[k + 1]), -xb_)
                hi = min(capM[k - 1] - f[k], capM[k] - f[k + 1], xa)
                t = min(max(t, lo), hi)
                f[k] += t; f[k + 1] += t; big = max(big, abs(t))
            # (9) SLAVE carry pass-through  S_{k-1} -> S_{k+1} across slave slot k
            #     effect  : g[k] += t, g[k+1] += t ; xs(k-1) -= t ; xs(k+1) += t ; xs(k) unchanged
            #     equation: nubar_{k-1} = nubar_{k+1}
            #     bounds  : both carries >= 0, both arrival caps (slot k and slot k+1), xs >= 0 at the ends
            #     stage 6 : S6/S7 on xs(k-1) (-t) and xs(k+1) (+t)
            if 1 <= k <= N - 2 and tx[k - 1] and tx[k + 1]:
                xa, xb_ = xs(k - 1), xs(k + 1)
                t = step(th2, hb[k - 1], xa, 1, th2, hb[k + 1], xb_, 1, Ts(k - 1), Ts(k + 1))
                lo = max(-g[k], -g[k + 1], -xb_)
                hi = min(W - a[k] * d[k] - g[k], W - a[k + 1] * d[k + 1] - g[k + 1], xa)
                if has_s67:
                    lo, hi = slave_lim(lo, hi, k - 1, xa, -1.0)
                    lo, hi = slave_lim(lo, hi, k + 1, xb_, 1.0)
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
    if has_T:                                            # burst rate   sum T_i/2 ln(1 + hbar x_i/T_i)
        r_s = float(np.sum(0.5 * Tw * np.log1p(hb * xb / Tw)))
    return JointResult(
        P=P, Pbar=Pbar, delta=delta, F=F, b=b, b_peak=b_peak,
        sent_power=delta / tau, arrived_power=jp.alpha * delta / tau,
        stored_power=a * delta / tau, rx_master=h * P, rx_slave=hb * Pbar,
        rate_master=r_m, rate_slave=r_s, objective=th1 * r_m + th2 * r_s,
        iters=it, converged=converged, overhead=cS.copy(), start=start)


# ----------------------------------------------------------------------------
# Stage 6: maximal-sharing tie-break (post-pass)
# ----------------------------------------------------------------------------
STOP_REASONS = ("ceiling", "S5", "M5", "master", "floor")


def fullest_trajectory(jp: JointParams, x, tol: float = 1e-12) -> dict:
    """Forward pass of maximal_sharing(): with bursts x [J] fixed, the master sends in every slot as much as
    the slave can take (ceiling S2, S5, M5) and it has (E_k + F_{k-1}); see maximal_sharing for the equations.
    With E_max = inf this trajectory is pointwise the fullest one, so it is also an exact feasibility test:
    fail_slot = first slot where even it falls below B_min (None = feasible).
    With a master floor (M6, jp.master_floor_J) the master only shares what is above fL[k+1]
    (stop reason "floor") and the last slot keeps fL[N] instead of spilling it.
    Returns dict(delta, spill, b, b_peak, stop, fail_slot, short_J)."""
    m = jp.master
    N, E, a = m.N, m.E, jp.a
    cS, capA, Dc = jp.overhead, jp.capA, jp.dcap
    fL = master_floor_profile(jp)
    has_floor = jp.master_floor_J is not None
    x = np.asarray(x, float)
    delta, spill, b, b_peak = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    stop = []
    level, carry = jp.b0, 0.0
    for k in range(N):
        avail = E[k] + carry
        head = jp.B_max - level
        cands = {"ceiling": head / a[k] if a[k] > 0 else 0.0,
                 "S5": capA[k] / a[k] if a[k] > 0 else 0.0,
                 "M5": Dc[k], "master": avail}
        if has_floor:                                      # M6: keep fL[k+1] for the master itself
            cands["floor"] = avail - fL[k + 1]
            del cands["master"]
        reason = min(cands, key=cands.get)
        dk = max(0.0, cands[reason])
        if head <= tol * max(jp.B_max, 1.0):
            reason, dk = "full", 0.0                       # arrived full: the master stops
        delta[k] = dk
        b_peak[k] = min(jp.B_max, level + a[k] * dk)
        level = b_peak[k] - cS[k] - x[k]
        if level < jp.B_min - 1e-9 * max(jp.B_max, 1.0):
            return dict(delta=delta, spill=spill, b=b, b_peak=b_peak, stop=stop, fail_slot=k, short_J=level - jp.B_min)
        b[k] = level
        left = avail - dk
        if k == N - 1:
            spill[k], carry = left - fL[N], 0.0           # the master ends holding its reserve fL[N] (0 = off)
        else:
            carry = min(left, m.E_max - E[k + 1])
            spill[k] = left - carry
        stop.append(reason)
    return dict(delta=delta, spill=spill, b=b, b_peak=b_peak, stop=stop, fail_slot=None, short_J=0.0)


def maximal_sharing(jp: JointParams, r: JointResult, tol: float = 1e-12):
    """Tie-break for theta1 = 0: the master shares as early and as much as the slave can take.

    With master Joules worth nothing many delta trajectories are optimal.  This forward pass keeps every
    slave burst x_i = tau*Pbar_i (so the objective, the events and Pbar do not change) and picks the delta
    trajectory with the fullest slave supercap; the master stops only at B_max (or S5 / M5 / its own energy):

        for k = 0 .. N-1:
            delta_k = max(0, min((B_max - b_{k-1})/a_k,  capA_k/a_k,  Dc_k,  E_k + F_{k-1}))   (S2, S5, M5, master)
            b_peak_k = b_{k-1} + a_k delta_k            (<= B_max)
            b_k      = b_peak_k - c_k - x_k              (>= B_min: the fullest trajectory dominates the optimum)
            F_k      = E_k + F_{k-1} - delta_k - spill_k,  spill_k = max(0, F_k_unclipped - (E_max - E_{k+1}));
                       F_{N-1} = 0 (what is left in the last slot is spilled)
            with a master floor (M6): delta_k <= E_k + F_{k-1} - fL[k+1] (stop "floor") and F_{N-1} = fL[N]

    "spill" is the master's unused energy (tau*P_i in the result).  The terminal condition g[N] = 0 of the
    solver is dropped: the slave may end above B_min.  If the master's energy cannot keep the fullest trajectory
    above B_min (it gave energy earlier that the optimum kept for later), the pass is NOT applied.

    Returns (r_new, info): info = dict(applied, stop (per-slot STOP_REASONS entry or "full"), reason).
    r_new is r itself when not applied.
    """
    if jp.theta1 != 0:
        raise ValueError("maximal_sharing is a tie-break for theta1 = 0 only (master energy must be worthless)")
    m = jp.master
    tau, E, a = m.tau, m.E, jp.a
    ft = fullest_trajectory(jp, tau * r.Pbar, tol)
    if ft["fail_slot"] is not None:
        k = ft["fail_slot"]
        return r, dict(applied=False, stop=ft["stop"], reason=f"slot {k}: the master's energy cannot keep the "
                                                               f"fullest trajectory above B_min ({ft['short_J']:.3g} J)")
    delta, spill, b, b_peak, stop = ft["delta"], ft["spill"], ft["b"], ft["b_peak"], ft["stop"]
    P = spill / tau
    F = np.cumsum(E - spill - delta)
    r_m = float(np.sum(0.5 * tau * np.log1p(m.h * P)))
    r_new = JointResult(
        P=P, Pbar=r.Pbar.copy(), delta=delta, F=F, b=b, b_peak=b_peak,
        sent_power=delta / tau, arrived_power=jp.alpha * delta / tau, stored_power=a * delta / tau,
        rx_master=m.h * P, rx_slave=r.rx_slave.copy(), rate_master=r_m, rate_slave=r.rate_slave,
        objective=jp.theta1 * r_m + jp.theta2 * r.rate_slave, iters=r.iters, converged=r.converged,
        overhead=None if r.overhead is None else r.overhead.copy(), start=r.start + " + maximal-sharing")
    return r_new, dict(applied=True, stop=stop, reason="")
