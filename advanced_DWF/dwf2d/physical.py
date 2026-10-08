"""Physical receive chain post-processing (photodiode + MPPT) and the beta calibration loop.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from physical_chain import OpticalReceiveChainResult, optical_to_electrical

from .params import JointParams, JointResult
from .solver import joint_dwf


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
    overflow_mppt: np.ndarray = None   # [J] per slot stored energy above the MPPT input cap capA (clipped; 0 without a cap)
    overflow_mppt_energy: float = 0.0  # [J] sum(overflow_mppt); 0 at a calibrated fixed point with S5 active in the solver
    overhead: np.ndarray = None        # [J] c_i subtracted from the re-simulated battery (zeros without a load)


def physical_stored_power(
    jp: "JointParams",
    r: JointResult,
    wavelength_nm: float = 532,
    mppt_efficiency: float = 0.95,
    fill_factor: float = 1.0,
    temp_K: float = 300.0,
    battery_tol_J: float = 1e-6,
    supercap_eff: float = 1.0,
) -> PhysicalStorageResult:
    """Run the solver's delta schedule through the real photodiode+MPPT chain.

    Takes r.arrived_power (alpha*delta/tau, the optical power actually reaching the slave --
    exactly the "P_transmitted*alpha" input physical_chain.py's optical_to_electrical() expects)
    and reports the power that chain would really deliver into the battery
    (chain.mppt_output_power), alongside a re-simulation of the slave battery using that
    physical arrival instead of the solver's linear a_i*delta_i assumption.

    Pure post-processing: does not modify jp or r, and never feeds back into joint_dwf().

    With a slave load attached (jp.slave_overhead_J, jp.mppt_cap_J) the re-simulated battery also
    subtracts c_i, and the stored energy is clipped to the MPPT cap (overflow_mppt).

    battery_tol_J : numerical tolerance [J] used for the overflow clip and the floor/ceiling
        violation counts below. joint_dwf() only converges to within `tol * scale` (its own `tol`
        argument, default 1e-10 * mean harvested energy per slot) -- typically ~1e-7 to 1e-6 J at
        the energy scales this model runs at -- so slots that sit exactly ON a bound (common: S1/
        S2 are frequently active) will show a residual of about that size even at an exact beta
        calibration. This must stay looser than that residual, or every such slot reads as a false
        "violation" instead of a real infeasibility.

    supercap_eff : storage efficiency of the slave supercap (slave_config.ETA_SC).  Stored energy =
        supercap_eff x MPPT output, so eff_phys (and the beta that solve_with_physical_chain calibrates to it)
        is the full secant gain  R x MPPT_EFF x ETA_SC  of PV array -> MPPT -> supercap.  1.0 (default) = off.
    """
    m = jp.master
    tau = m.tau
    chain = optical_to_electrical(
        r.arrived_power, wavelength_nm, mppt_efficiency,
        fill_factor=fill_factor, temp_K=temp_K,
    )
    stored_raw = chain.mppt_output_power                          # what the photodiode + MPPT chain would deliver
    if supercap_eff != 1.0:                                       # supercap storage loss (ETA_SC), inside beta
        stored_raw = supercap_eff * stored_raw
    with np.errstate(divide="ignore", invalid="ignore"):
        eff_phys = np.where(r.arrived_power > 0, stored_raw / np.where(r.arrived_power > 0, r.arrived_power, 1.0), np.nan)
    # MPPT input rating (S5): physical_chain.py does not saturate, so clip here.  The energy above
    # capA is rejected (overflow_mppt); eff_phys above stays the UNCLIPPED chain gain so calibration is stable.
    capA = jp.capA
    excess = tau * stored_raw - capA                              # inf cap -> -inf -> no clip
    overflow_mppt = np.where(excess > battery_tol_J, excess, 0.0)
    stored_energy_phys = np.where(overflow_mppt > 0, capA, tau * stored_raw)
    stored_power_phys = stored_energy_phys / tau
    cS = jp.overhead                                              # slave electronics drain c_i

    # Re-simulate the slave battery: same spending (tau*Pbar) and electronics drain c_i as the solver's
    # solution, but the PHYSICAL arrival instead of a_i*delta_i.  Starts at b0 (same initial charge).
    tau_pbar = r.Pbar * tau                                       # tau*Pbar_i, the slave's actual spend (unchanged)
    b_prev_phys = np.empty(m.N)
    b_phys = np.empty(m.N)
    overflow = np.zeros(m.N)
    level = jp.b0                                                 # running battery level, starts at b0
    for i in range(m.N):
        b_prev_phys[i] = level
        level = level + stored_energy_phys[i] - tau_pbar[i] - cS[i]
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
        overflow_mppt=overflow_mppt,
        overflow_mppt_energy=float(overflow_mppt.sum()),
        overhead=cS.copy(),
    )


def solve_with_physical_chain(
    jp: JointParams,
    wavelength_nm: float = 532,
    mppt_efficiency: float = 0.95,
    fill_factor: float = 1.0,
    temp_K: float = 300.0,
    tol: float = 1e-3,
    max_outer: int = 30,
    supercap_eff: float = 1.0,
    solver_kw: dict | None = None,
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

    The slave load (jp.slave_overhead_J) and the MPPT cap (jp.mppt_cap_J) are carried unchanged through
    the loop (they do not depend on beta); only beta_i is calibrated.

    Slots with delta_i == 0 have nothing to calibrate on (arrived_power_i == 0 -> eff_phys_i is
    NaN); their beta is left unchanged from the previous iterate.

    supercap_eff (slave_config.ETA_SC) is passed to physical_stored_power, so the calibrated beta_i is the
    full chain gain R x MPPT_EFF x ETA_SC (mppt_efficiency carries MPPT_EFF).  1.0 (default) = off.

    solver_kw : extra joint_dwf() arguments (tol, max_iter, cross_run, x0); None = joint_dwf defaults.  An x0 warm
    start is used for the first outer iteration only (later ones start from joint_dwf's own start point).

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
        if solver_kw:
            kw = dict(solver_kw)
            if n_outer > 1:
                kw.pop("x0", None)
            r = joint_dwf(jp_i, **kw)
        else:
            r = joint_dwf(jp_i)
        ph = physical_stored_power(jp_i, r, wavelength_nm, mppt_efficiency, fill_factor, temp_K,
                                   supercap_eff=supercap_eff)
        on = r.delta > 1e-9 * max(jp.master.E.mean(), 1.0)         # only calibrate where energy actually moves
        beta_new = beta.copy()
        beta_new[on] = ph.eff_phys[on]
        moved = float(np.max(np.abs(beta_new - beta))) if np.any(on) else 0.0
        beta = beta_new
        if moved < tol:
            break
    return jp_i, r, ph, n_outer


# ----------------------------------------------------------------------------
# Stage 6: event refusal and rescheduling (outer loop around the calibrated solve)
# ----------------------------------------------------------------------------
@dataclass
class RefusalResult:
    """Outcome of solve_with_refusals().

    jp, r, ph : the final calibrated problem, its JointResult and PhysicalStorageResult (effective schedule)
    events    : one dict per RELEASED event: release, run (slot or None), delay, x_J (burst energy), outcome
                ("run" at its release slot, "rescheduled" = ran later, "dropped", or "unresolved" when the loop did
                not converge: placed, but no consistent solution exists for that placement)
    effective : final effective schedule (1 = an event ran there)
    passes    : number of solves;  converged : False if the pass cap was hit (the schedule is then NOT consistent)
    history   : per pass dict(refused, moves, dropped, start, reason)
    """
    jp: JointParams
    r: JointResult
    ph: PhysicalStorageResult
    events: list
    effective: np.ndarray
    passes: int
    converged: bool
    history: list
    n_outer: int = 0

    @property
    def counts(self) -> dict:
        c = {"released": len(self.events)}
        for o in ("run", "rescheduled", "dropped", "unresolved"):
            c[o] = sum(e["outcome"] == o for e in self.events)
        return c


def solve_with_refusals(build_jp, schedule, x_min: float, max_delay: int, *, mode: str = "proportional",
                        max_passes: int | None = None, warm: bool = True, calib_kw: dict | None = None,
                        solver_kw: dict | None = None, calibrate: bool = True) -> RefusalResult:
    """Greedy refusal / reschedule loop of the stage-6 slave (mirrors the hardware: wake, check, refuse).

        effective = schedule (events released);  queue empty
        repeat (at most #events * (max_delay + 1) + 1 passes):
            solve_with_physical_chain(build_jp(effective))           (beta carried over from the previous pass)
            refused = run slots with x_i < x_min                     ("proportional", x_i = tau*Pbar_i, incl. x_i = 0)
                      or, if the solve raises InfeasibleSlaveBudget ("fixed_bias", or an unaffordable event load),
                      the latest run event at or before the first slot where even the fullest supercap
                      trajectory (solver.fullest_trajectory, bursts 0) drops below B_min
            none refused -> done
            each refused event: its slot goes back to sleep load; it moves to the first later slot with no event
            of its own and at most max_delay slots after its RELEASE, else it is dropped (slave_load.reschedule)

    build_jp(effective) -> JointParams with the event load for that effective schedule (slave_load.attach_event_load);
    physical.py never imports slave_load.  Warm start: the previous pass's flows when they satisfy the new bounds.
    Termination: every pass moves an event strictly later or drops it, and delays are bounded.  If the pass cap is
    hit, converged = False.  This is a greedy rule, NOT a combinatorial optimum (see the brute-force check).
    calibrate = False solves joint_dwf directly (beta as given, ph = None): used by the brute-force comparison.
    """
    import slave_load                                        # queue logic only (lazy: keeps the import graph acyclic)
    from .solver import InfeasibleSlaveBudget, fullest_trajectory, warm_start_from
    sch = np.asarray(schedule, dtype=int).ravel()
    released = [int(i) for i in np.flatnonzero(sch)]
    release_of = {i: i for i in released}
    effective = sch.copy()
    #   each event can be refused max_delay times (moved) plus once more (dropped); one last pass confirms
    cap = len(released) * (max(int(max_delay), 0) + 1) + 1 if max_passes is None else int(max_passes)
    calib_kw = dict(calib_kw or {})
    history, dropped_all = [], []
    beta, x0 = None, None
    jp_c = r = ph = None
    n_outer = 0
    converged = False
    passes = 0
    for passes in range(1, cap + 1):
        jp = build_jp(effective)
        if beta is not None:
            jp = dataclasses.replace(jp, beta=beta)
        skw = dict(solver_kw or {})
        if warm and x0 is not None:
            skw["x0"] = x0
        try:
            if calibrate:
                jp_c, r, ph, n_outer = solve_with_physical_chain(jp, solver_kw=skw, **calib_kw)
            else:
                jp_c, r, ph, n_outer = jp, joint_dwf(jp, **skw), None, 0
        except InfeasibleSlaveBudget as e:
            # Wake, check, refuse: the first slot where even the FULLEST supercap trajectory (master sharing all
            # it can, bursts at 0) falls below B_min.  An event there is refused; in a sleep slot the latest event
            # before it is.  No event to blame -> the base (sleep) load itself is unaffordable: stop, unresolved.
            ft = fullest_trajectory(jp, np.zeros(jp.master.N))
            k0 = ft["fail_slot"]
            if k0 is None:                                   # floor reachable: another bound (S5/M5/terminal)
                k0 = min(e.slots) if e.slots else jp.master.N - 1
            before = [i for i in release_of if i <= k0]
            if not before:
                history.append(dict(refused=[], moves=[], dropped=[], start="infeasible",
                                    reason=f"infeasible at slot {k0} with no event at or before it: {e}"))
                jp_c = r = ph = None
                break
            refused, reason = [max(before)], f"supercap short at slot {k0}"
            jp_c = r = ph = None
        else:
            beta = jp_c.beta.copy()
            x = jp_c.master.tau * r.Pbar
            refused = [i for i in sorted(release_of) if x[i] < x_min * (1 - 1e-9)] if mode == "proportional" else []
            #   (x_i < x_min includes x_i = 0: no event spends its camera/MCU/PSRAM energy without transmitting)
            reason = "x < x_min"
            if not refused:
                converged = True
                history.append(dict(refused=[], moves=[], dropped=[], start=r.start, reason="all events affordable"))
                break
            x0 = warm_start_from(jp_c, r)
        effective, release_of, moves, dropped = slave_load.reschedule(effective, refused, release_of, max_delay)
        dropped_all += dropped
        history.append(dict(refused=refused, moves=moves, dropped=dropped, start=r.start if r is not None else "infeasible",
                            reason=reason))
    events = []
    run_of = {rel: run for run, rel in release_of.items()}
    xfin = jp_c.master.tau * r.Pbar if (converged and r is not None) else None
    for rel in released:
        run = run_of.get(rel)
        if run is None:
            events.append(dict(release=rel, run=None, delay=None, x_J=0.0, outcome="dropped"))
        elif not converged:                                  # no consistent solution for this placement
            events.append(dict(release=rel, run=run, delay=run - rel, x_J=np.nan, outcome="unresolved"))
        else:
            events.append(dict(release=rel, run=run, delay=run - rel, x_J=float(xfin[run]),
                               outcome="run" if run == rel else "rescheduled"))
    n = {o: sum(e["outcome"] == o for e in events) for o in ("run", "rescheduled", "dropped", "unresolved")}
    if sum(n.values()) != len(released) or (converged and n["unresolved"]):   # no event may disappear
        raise AssertionError(f"event bookkeeping: {n} vs released {len(released)}")
    return RefusalResult(jp=jp_c, r=r, ph=ph, events=events, effective=effective, passes=passes,
                         converged=converged, history=history, n_outer=n_outer)
