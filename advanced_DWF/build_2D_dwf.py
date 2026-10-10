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
         Optional slave load (off by default): its electronics (camera, MCU, memory, MPPT) burn a known
         c_i >= 0 Joules per slot, ON or silent, and the MPPT accepts at most capA_k stored Joules per
         slot (JointParams.slave_overhead_J, mppt_cap_J; see slave_load.attach_slave_load()).

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
    c_i            cS[i]       slave electronics energy in slot i (jp.overhead, zeros without a load)
    capA_k         capA[k]     MPPT stored-energy cap of slot k (jp.capA, inf without a cap)

Energy balance (this is how xm and xs are DEFINED in the code):

    tau*P_i    = E_i + F_{i-1} - F_i - delta_i              (master: harvest + carry in - carry out - shared)
    tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i - c_i         (slave : arrival  + carry in - carry out - electronics)

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
         A silent slot still receives (delta_i) and stores energy; with a slave load it also DRAINS c_i,
         so the battery falls through a silent run (Gt_i = Gt_{i-1} + a_i delta_i - c_i).
         Moves whose slave end is a silent slot are skipped in joint_dwf().
    (S5) MPPT input rating  a_k*delta_k <= capA_k      stored energy per slot       (JointParams.mppt_cap_J)
         One more linear bound on d[k] (inf = off); added through apply() and in moves 1-5.
    The ceiling is checked at ARRIVAL and the floor at the END of the slot (arrival first, drain after).

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
Start at "spend what you harvest, share nothing" (f = 0, g = 0, d = 0); with a slave load c > 0 that start is
infeasible (g[N] = 0 is fixed), so a just-in-time start is built instead (d[k] = max(0, c_k - g[k])/a_k; if it
breaks a bound, an LP start; else InfeasibleSlaveBudget).  Repeatedly sweep
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

# These imports are unused here but kept: they were always importable from build_2D_dwf.
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from build_dwf import Harvested_energy_summer, MasterParams, Harvested_energy_winter
from dwf_diagnostics import diagnose
from physical_chain import optical_to_electrical, OpticalReceiveChainResult

# ----------------------------------------------------------------------------
# The implementation lives in the dwf2d package (one module per concern); this module re-exports every
# name it has always defined, so `from build_2D_dwf import ...` and `python build_2D_dwf.py` are unchanged.
#   dwf2d/channels.py      beer_lambert, THz absorption table, thz_gain; stage 6: THz photomixer link budget
#                          (thz_hbar, link_budget, near-field limit), uowc_alpha, modulation engine
#   dwf2d/params.py        JointParams, JointResult
#   dwf2d/solver.py        joint_dwf (+ feasible start points, InfeasibleSlaveBudget)
#   dwf2d/physical.py      physical receive chain post-processing and beta calibration
#   dwf2d/verification.py  check_kkt_joint, reference_solution_joint (cvxpy)
#   dwf2d/plotting/        style helpers, joint plots, physical-chain plots, slave-load panels + CSV
#   dwf2d/demos.py         legacy_demo, physical_demo, integrated_demo (stage 6)
# ----------------------------------------------------------------------------
from dwf2d.channels import (  # noqa: F401
    beer_lambert,
    thz_absorption,
    thz_alpha_cm,
    thz_alpha_m,
    thz_freq,
    thz_gain,
    # stage 6
    link_budget,
    near_field_limit_m,
    noise_floor_w,
    thz_hbar,
    uowc_alpha,
    make_link_simulator,
    qam_scheme,
)
from dwf2d.params import (  # noqa: F401
    JointParams,
    JointResult,
)
from dwf2d.solver import (  # noqa: F401
    InfeasibleSlaveBudget,
    joint_dwf,
    _jit_start,
    _lp_start,
    # stage 6
    fullest_trajectory,
    maximal_sharing,
    warm_start_from,
)
from dwf2d.physical import (  # noqa: F401
    physical_stored_power,
    PhysicalStorageResult,
    solve_with_physical_chain,
    # stage 6
    RefusalResult,
    solve_with_refusals,
)
from dwf2d.verification import (  # noqa: F401
    check_kkt_joint,
    reference_solution_joint,
    # stage 6
    brute_force_events,
    check_maximal_sharing,
    parity_matrix,
    refusal_gate,
)
from dwf2d.plotting.style import (  # noqa: F401
    BITS_PER_NAT,
    C_LINK,
    C_MASTER,
    C_SLAVE,
    EDGE_TOL,
    GRID,
    INK,
    INK2,
    _bound_masks,
    _finish,
    _legend,
    _mark,
    _new_axes,
    _new_axis,
    _rates_bits,
    _shade_silent,
    _style,
)
from dwf2d.plotting.joint import (  # noqa: F401
    plot_cumulative_energy,
    plot_harvested_energy,
    plot_link_power,
    plot_sharing_energy,
    plot_slave_battery,
    plot_throughput_cumulative,
    plot_throughput_per_slot,
    plot_transmit_power,
)
from dwf2d.plotting.physical import (  # noqa: F401
    energy_loss_breakdown,
    plot_loss_histogram,
    plot_physical_slave_battery,
    plot_physical_stored_power,
)
from dwf2d.plotting.slave_panels import (  # noqa: F401
    C_MCU,
    C_MEM,
    C_MPPT,
    LOAD_COLORS,
    LOAD_LABELS,
    plot_event_bursts,
    plot_slave_battery_window,
    plot_slave_cumulative_energy,
    plot_slave_energy_breakdown,
    plot_slave_load_per_slot,
    slave_load_components,
    write_refusal_history_csv,
    write_slave_energy_csv,
)
from dwf2d.plotting.thz_link import (  # noqa: F401   (stage 6: the notebook's THz-link figures)
    ThzLinkSetup,
    notebook_schemes,
    plot_ber_sweep,
    plot_ber_vs_distance_gains,
    plot_constellations,
    plot_eye_diagrams,
    plot_gain_vs_aperture,
    plot_link_characteristics,
    plot_max_range_heatmap,
    plot_max_range_vs_gain,
    plot_received_power_vs_distance,
    plot_sim_vs_theory,
    plot_snr_power_vs_distance,
    setup_from_slave_config,
    simulate_demo,
    simulate_validation,
    write_thz_link_plots,
)
from dwf2d.demos import (  # noqa: F401
    legacy_demo,
    physical_demo,
    integrated_demo,
    _demo_problem,
    _integrated_problem,
    _run_demo,
)

# ----------------------------------------------------------------------------
# Demos.  DEMO selects one (or: python build_2D_dwf.py physical_demo)
#   legacy_demo   : no slave load attached; reproduces the README numbers (objective 1902.8194, 1410 iters).
#   physical_demo : dummy slave timing (slave_config.py), MPPT cap S5 on, window from size_storage.
#                   The result is at DUMMY durations and placeholder h0 / theta2; it differs from legacy_demo
#                   by orders of magnitude in slave data energy (the real electronics cannot absorb the
#                   legacy stage-4 flows).  That is expected, not a solver bug: inputs are NOT retuned.
#   integrated_demo : stage 6, master = UOWC energy only, slave = THz bursts in events (slave_config.py STAGE 6
#                   block); writes to out_integrated_demo/.
# ----------------------------------------------------------------------------
# DEMO = "legacy_demo"
DEMO = "integrated_demo"


if __name__ == "__main__":
    import sys
    # Run as a script:  python build_2D_dwf.py [legacy_demo | physical_demo | integrated_demo]   (or edit DEMO above)
    which = sys.argv[1] if len(sys.argv) > 1 else DEMO
    demos = {"legacy_demo": legacy_demo, "physical_demo": physical_demo, "integrated_demo": integrated_demo}
    if which not in demos:
        raise SystemExit("usage: python build_2D_dwf.py [legacy_demo | physical_demo | integrated_demo]")
    demos[which]()
