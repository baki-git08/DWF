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
    or the master's harvest make S1-S5 (with g[N] = 0) unsatisfiable.  Constraints are never relaxed."""


def _jit_start(jp: "JointParams"):
    """Just-in-time start: the master sends in slot k exactly what the slave is short of there.

        f = 0 (master spends what it harvests),  xs = 0 (no slave data energy yet)
        d[k]   = max(0, c_k - g[k]) / a_k
        g[k+1] = g[k] + a_k d[k] - c_k          (>= 0 by construction)
    Any initial excess beyond the total drain (d == 0 everywhere) is spent as data in the first ON slot.
    Returns (d, g, violations); violations = [(slot, reason)] for bounds the start breaks (empty = feasible).
    """
    m = jp.master
    N, E, a, cS, capA, W = m.N, m.E, jp.a, jp.overhead, jp.capA, jp.W
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
    left = g[N]
    if left > 1e-12 * max(1.0, jp.g_init):
        j = int(np.argmax(jp.slave_schedule)) if jp.slave_schedule.any() else -1
        if j < 0:
            viol.append((N - 1, f"initial charge exceeds the total drain by {left:.6g} J and no ON slot can spend it"))
        else:
            g[j + 1:] -= left                               # xs(j) = left: spend it as data in the first ON slot
            g[N] = 0.0
    else:
        g[N] = 0.0
    return d, g, viol


def _lp_start(jp: "JointParams"):
    """Feasibility LP (scipy) with the constraints of reference_solution_joint: a start that may borrow
    against future harvest through the master carry f.  Returns (d, f, g) or None if infeasible."""
    try:
        from scipy.optimize import linprog
    except ImportError:                                        # pragma: no cover
        return None
    m = jp.master
    N, E, a, cS, capA = m.N, m.E, jp.a, jp.overhead, jp.capA
    tx = jp.slave_schedule
    nv = 3 * N                                                # variables: x (master data), d, xb (slave data)
    ix, idl, ixb = np.arange(N), N + np.arange(N), 2 * N + np.arange(N)
    L = np.tril(np.ones((N, N)))                              # cumulative-sum operator
    Lm = np.tril(np.ones((N, N)), -1)                         # strictly lower: cum up to k-1
    Z = np.zeros((N, N))
    A_ub = [np.hstack([L, L, Z]),                             # M1: cum(x+d) <= cum(E)
            np.hstack([Z, -L * a, L]),                        # S1: b0 + cum(a d - xb - c) >= B_min
            np.hstack([Z, L * a, -Lm])]                       # S2: b0 + cum(a d)_k - cum(xb + c)_{k-1} <= B_max
    b_ub = [np.cumsum(E), jp.b0 - jp.B_min - np.cumsum(cS), jp.B_max - jp.b0 + Lm @ cS]
    if np.isfinite(m.E_max):                                  # M2: cum(E) - cum(x+d) <= E_max - E_{k+1}
        A_ub.append(np.hstack([-L[:-1], -L[:-1], Z[:-1]]))
        b_ub.append(m.E_max - E[1:] - np.cumsum(E)[:-1])
    A_eq = np.vstack([np.concatenate([np.ones(N), np.ones(N), np.zeros(N)]),     # master uses all its energy
                      np.concatenate([np.zeros(N), a, -np.ones(N)])])              # slave terminal g[N] = 0
    b_eq = np.array([E.sum(), jp.B_min - jp.b0 + cS.sum()])
    bounds = [(0, None)] * (2 * N) + [(0, 0) if not tx[i] else (0, None) for i in range(N)]
    for i in range(N):                                        # S5 as an upper bound on d
        if np.isfinite(capA[i]):
            bounds[N + i] = (0, capA[i] / a[i] if a[i] > 0 else 0.0)
    res = linprog(np.zeros(nv), A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    x, dl, xb = res.x[ix], res.x[idl], res.x[ixb]
    d = np.maximum(dl, 0.0)
    f = np.zeros(N + 1)
    f[1:] = np.maximum(np.cumsum(E - x - dl), 0.0)
    f[N] = 0.0
    g = np.zeros(N + 1)
    g[0] = jp.g_init
    g[1:] = np.maximum(jp.g_init + np.cumsum(a * dl - xb - cS), 0.0)
    g[N] = 0.0
    return d, f, g


# ----------------------------------------------------------------------------
# Core solver
# ----------------------------------------------------------------------------
def joint_dwf(jp: JointParams, tol: float = 1e-10, max_iter: int = 100_000,
              cross_run: bool | None = None) -> JointResult:
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
    start = "share-nothing"
    if has_load:
        #   With c > 0 "share nothing" is infeasible (g[N] = 0 is fixed): the slave must be fed to cover
        #   its drain.  Just-in-time start (module docstring section 5); if it breaks a bound (e.g. harvest
        #   too small in a slot with f = 0) try an LP that may borrow through f; otherwise raise.
        d0, g0, viol = _jit_start(jp)
        if not viol:
            d[:], g[:], start = d0, g0, "just-in-time"
        else:
            lp = _lp_start(jp)
            if lp is None:
                raise InfeasibleSlaveBudget(
                    "no feasible schedule keeps the slave alive (S1-S5, g[N] = 0): "
                    + "; ".join(f"slot {k}: {why}" for k, why in viol[:3])
                    + (f" (+{len(viol) - 3} more)" if len(viol) > 3 else "")
                    + ".  The LP feasibility check (borrowing via the master carry) is infeasible too.")
            d[:], f[:], g[:] = lp
            start = "lp-fallback"

    # ---- energy balance: these two lambdas DEFINE the data energies -----------------------------
    xm = lambda i: E[i] + f[i] - f[i + 1] - d[i]          # master data energy
    #   tau*P_i = E_i + F_{i-1} - F_i - delta_i    (harvest + carry in - carry out - shared)
    xs = lambda i: a[i] * d[i] + g[i] - g[i + 1] - cS[i]  # slave  data energy
    #   tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i - c_i (arrival + carry in - carry out - electronics drain)

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
        * a[q]*d[q]         <= capA[q]                   S5      (MPPT input rating, only if a cap is set)
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
        for q, c in dch:
            lim(a[q] * d[q], a[q] * c, -np.inf, capA[q])    # S5: stored energy per slot <= capA_q
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
                t = min(max(t, max(-d[k], -xb / ak)), min(x, (W - g[k]) / ak - d[k], capA[k] / ak - d[k]))
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
                t = min(max(t, max(-d[k], -xb1 / ak)), min(x, g[k] / ak, capA[k] / ak - d[k]))
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
                hi = min(capM[k - 1] - f[k], xa, (W - g[k]) / ak - d[k], capA[k] / ak - d[k])
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
                hi = min(f[k + 1], xa, (W - g[k]) / ak - d[k], capA[k] / ak - d[k])
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
                hi = min(xa, (W - g[k]) / ak - d[k], (W - a[k + 1] * d[k + 1] - g[k + 1]) / ak,
                         capA[k] / ak - d[k])
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
        iters=it, converged=converged, overhead=cS.copy(), start=start)
