"""Scenario test suite for the Master + Slave DWF (integrated_demo configuration) with execution-time recording.

    python tests/scenarios.py [--only 3.4] [--repeat 3]

Every scenario is a small OVERRIDE DICT on top of the default config (slave_config.py defaults, SLAVE_SCHEDULE,
THZ_PRESET, D_THZ_M, D_UOWC_M, MPPT_EFF, ETA_SC, ...).  Nothing outside tests/ is touched.

Conventions: 0-indexed slots, units J / s / W.
    default harvest     : the full 60-slot Harvested_energy_winter (NOT the demo's `_half`, which is zero in slots 1-59)
    sharing policies    : delta_cap_J[k] = 0 in non-sharing slots (existing M5 hook); the existing maximal-sharing
                          post-pass then shares the most the constraints allow in the allowed slots.
Override keys:
    E_max           master battery upper bound (M2) [J], inf allowed        (default DEFAULT_E_MAX_J = 5000; the code's 4900 is invalid with full winter harvest)
    floor           master reserve floor (M6) [J]                           (default slave_config.MASTER_FLOOR_J)
    harvest         "all" | "first_slot" | "first_half" | array of length N (default "all")
    harvest_scale   multiplier on the harvest array
    attenuation     callable(slot, N) -> c [1/m]  or array of length N      (default: constant UOWC_C_PER_M)
    share_slots     "once_first_event" | "constant" | ("every", n)          (default: policy off = DWF optimum)
    b0_full         True: slave supercap starts at B_max  (NOT used by default; see REDESIGN_HINT)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import platform
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.environ.setdefault("MPLBACKEND", "Agg")

import slave_config as cfg                                            # noqa: E402
import slave_load                                                     # noqa: E402
from build_dwf import Harvested_energy_winter, MasterParams          # noqa: E402
import importlib                                                      # noqa: E402
from dwf2d import demos                                               # noqa: E402
phys_mod = importlib.import_module("dwf2d.physical")                   # (dwf2d.physical attr is shadowed)
from dwf2d.channels import uowc_alpha                                 # noqa: E402
from dwf2d.solver import maximal_sharing, master_floor_profile       # noqa: E402
from dwf2d.verification import check_kkt_joint, reference_solution_joint                       # noqa: E402

INF = float("inf")
# Scenario default master battery.  slave_config.MASTER_E_MAX_J = 4900 J is INVALID with the full 60-slot winter harvest
# (JointParams needs floor <= E_max - max(E[1:]) = 979 J); 5000 J is the user-approved default for these scenarios.
DEFAULT_E_MAX_J = 5000.0
LEDGER_TOL_J = 1e-9          # "about 1e-11" in the brief; the solver stops near 4e-12 J scale.  Tighten if desired.
LEVEL_TOL_J = 1e-9
DUMMY = ("water_attenuation stub (USER-SUPPLIED)", "3.5 ramp 0.046 -> 0.20 1/m (plumbing check, not a physical model)")
REDESIGN_HINT = ("NEEDS REDESIGN: the slave starts at B_min and its sleep load needs energy before the first share. "
                 "Start the supercap full (override {'b0_full': True} -> b0 = B_max) and make sure the master "
                 "does not need to harvest at that moment, then rerun.")


# --------------------------------------------------------------------------------------------------------------
# USER-SUPPLIED: dynamic water attenuation.  Swap in your function (or pass a length-N array) without editing tests.
# --------------------------------------------------------------------------------------------------------------
def water_attenuation(slot_index: int, N: int) -> float:   # 1/m, c in exp(-c*d)
    """USER-SUPPLIED.  Placeholder returns the constant code default c = slave_config.UOWC_C_PER_M (0.315 1/m)."""
    return float(cfg.UOWC_C_PER_M)


def _ramp(slot_index: int, N: int) -> float:
    """Deterministic plumbing check: c ramps 0.046 -> 0.20 1/m over the N slots.  NOT a physical model."""
    return 0.046 + (0.20 - 0.046) * slot_index / max(N - 1, 1)


# --------------------------------------------------------------------------------------------------------------
# Scenario list: (scenario_id, case, overrides).  Adding a scenario = adding one line.
# --------------------------------------------------------------------------------------------------------------
SCENARIOS = [
    ("3.1", "inf master battery",  {"E_max": INF}),
    ("3.2", "Bmax=5000 Bmin=1000", {"E_max": 5000.0, "floor": 1000.0}),
    ("3.3", "a: slot 0 only",      {"harvest": "first_slot"}),
    ("3.3", "b: first half",       {"harvest": "first_half"}),
    ("3.3", "c: all slots",        {}),
    ("3.4", "n=1",                 {"harvest_scale": 1e-1}),
    ("3.4", "n=2",                 {"harvest_scale": 1e-2}),
    ("3.4", "n=3",                 {"harvest_scale": 1e-3}),
    ("3.5", "stub (neutral)",      {"attenuation": water_attenuation}),
    ("3.5", "ramp (plumbing)",     {"attenuation": _ramp}),
    ("3.6", "share once",          {"share_slots": "once_first_event"}),
    ("3.7", "n=2",                 {"share_slots": ("every", 2)}),
    ("3.7", "n=5",                 {"share_slots": ("every", 5)}),
    ("3.7", "n=10",                {"share_slots": ("every", 10)}),
    ("3.8", "share constantly",    {"share_slots": "constant"}),
]
BASELINES = {"3.6", "3.7", "3.8"}


# --------------------------------------------------------------------------------------------------------------
# Problem builder
# --------------------------------------------------------------------------------------------------------------
def _harvest(ov: dict) -> np.ndarray:
    base = Harvested_energy_winter.copy()
    h = ov.get("harvest", "all")
    n = base.size
    if isinstance(h, str):
        if h == "first_slot":
            base[1:] = 0.0
        elif h == "first_half":
            base[n // 2:] = 0.0
        elif h != "all":
            raise ValueError(f"unknown harvest {h!r}")
    else:
        base = np.asarray(h, float).copy()
    return base * float(ov.get("harvest_scale", 1.0))


def _attenuation(ov: dict, n: int):
    a = ov.get("attenuation")
    if a is None:
        return None
    c = np.array([a(i, n) for i in range(n)], float) if callable(a) else np.asarray(a, float)
    if c.size != n:
        raise ValueError(f"attenuation needs {n} entries, got {c.size}")
    return c


def _share_mask(ov: dict, schedule: np.ndarray):
    p = ov.get("share_slots")
    n = schedule.size
    if p is None:
        return None, ""
    m = np.zeros(n, bool)
    if p == "constant":
        m[:] = True
        return m, "all slots"
    if p == "once_first_event":
        k = int(np.flatnonzero(schedule)[0])
        m[k] = True
        return m, f"slot {k} only (first scheduled event)"
    if isinstance(p, tuple) and p[0] == "every":
        m[::int(p[1])] = True
        return m, f"k % {p[1]} == 0"
    raise ValueError(f"unknown share_slots {p!r}")


def make_problem(ov: dict):
    """(jp0, build, mask_desc): jp0 with master / alpha / floor overrides; build(effective) attaches the event load
    and applies the sharing-policy cap AFTER attach_event_load (which sets delta_cap_J itself)."""
    jp0, _, link = demos._integrated_problem()
    E = _harvest(ov)
    n = E.size
    e_max = float(ov.get("E_max", DEFAULT_E_MAX_J))
    floor = ov.get("floor", cfg.MASTER_FLOOR_J)
    master = MasterParams(E=E, E_max=e_max, tau=jp0.master.tau, h=1.0)
    c = _attenuation(ov, n)
    alpha = jp0.alpha if c is None else uowc_alpha(cfg.D_UOWC_M, c, cfg.ETA_LD, cfg.UOWC_GEOM)
    jp0 = dataclasses.replace(jp0, master=master, alpha=alpha, master_floor_J=floor)
    mask, desc = _share_mask(ov, np.resize(np.asarray(cfg.SLAVE_SCHEDULE, int), n))

    def build(effective):
        jp = slave_load.attach_event_load(dataclasses.replace(jp0, slave_schedule=np.asarray(effective, int)), link=link)
        kw = {}
        if mask is not None:
            kw["delta_cap_J"] = np.where(mask, jp.dcap, 0.0)
        if ov.get("b0_full"):
            kw["b0"] = jp.B_max
        return dataclasses.replace(jp, **kw) if kw else jp
    return jp0, build, desc


# --------------------------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------------------------
class _SolveTimer:
    """Wraps physical.solve_with_physical_chain (called by solve_with_refusals) to time and count solver passes."""
    def __init__(self):
        self.t, self.n = 0.0, 0
        self.first = None          # (converged, objective) of the FIRST joint solve (before any refusal)

    def __enter__(self):
        self.orig = phys_mod.solve_with_physical_chain

        def wrapped(*a, **k):
            t0 = time.perf_counter()
            self.n += 1
            try:
                out = self.orig(*a, **k)
                if self.first is None:
                    self.first = (bool(out[1].converged), float(out[1].objective), int(out[1].iters))
                return out
            finally:
                self.t += time.perf_counter() - t0
        phys_mod.solve_with_physical_chain = wrapped
        return self

    def __exit__(self, *exc):
        phys_mod.solve_with_physical_chain = self.orig


def _jsonable(ov: dict) -> str:
    return json.dumps({k: (v.__name__ if callable(v) else (v.tolist() if isinstance(v, np.ndarray) else v))
                       for k, v in ov.items()}, default=str)


def run_case(sid: str, case: str, ov: dict) -> dict:
    row = dict(scenario_id=sid, case=case, overrides=_jsonable(ov), status="ERROR", reason="", redesign_hint="")
    res = {}
    t_start = time.perf_counter()
    timer = _SolveTimer()
    try:
        with timer:
            jp0, build, desc = make_problem(ov)
            tb = slave_load.tx_bounds_J()
            calib = dict(wavelength_nm=cfg.PV_CAL_WAVELENGTH_NM, mppt_efficiency=cfg.MPPT_EFF,
                         supercap_eff=cfg.ETA_SC, fill_idle_beta=True)
            rr = phys_mod.solve_with_refusals(build, jp0.slave_schedule, tb["x_min"], cfg.MAX_DELAY_SLOTS,
                                              mode=cfg.PM_DC_MODE, calib_kw=calib, solver_kw=dict(tol=1e-15))
        row["n_passes"] = rr.passes
        row["converged"] = rr.converged
        if not rr.converged or rr.r is None:
            last = rr.history[-1]["reason"] if rr.history else "no history"
            row["reason"] = f"INFEASIBLE: refusal loop did not converge ({last})"
            if ov.get("share_slots") is not None:
                row["redesign_hint"] = REDESIGN_HINT
            row["objective"] = float("nan")
        else:
            res = _evaluate(row, jp0, rr, tb, ov, timer.first)
    except Exception as e:                                           # noqa: BLE001  (suite must continue)
        row["reason"] = f"{type(e).__name__}: {e}"
        if ov.get("share_slots") is not None and "Infeasible" in type(e).__name__:
            row["redesign_hint"] = REDESIGN_HINT
        row["trace"] = traceback.format_exc(limit=3)
    row["t_total_s"] = time.perf_counter() - t_start
    row["t_solve_s"] = timer.t
    row["t_ref_s"] = row.get("t_ref_s", 0.0)
    row["t_other_s"] = row["t_total_s"] - timer.t - row["t_ref_s"]
    row.setdefault("n_passes", timer.n)
    return dict(row=row, res=res)


def _evaluate(row: dict, jp0, rr, tb: dict, ov: dict, first=None) -> dict:
    jp, r = rr.jp, rr.r
    m = jp.master
    tau, E, N = m.tau, m.E, m.N
    r_rep = r
    stop_note = ""
    if cfg.MAXIMAL_SHARING:
        r2, info = maximal_sharing(jp, r)
        if info["applied"]:
            r_rep = r2
        else:
            stop_note = f"post-pass not applied: {info['reason']}"
    kk = check_kkt_joint(jp, r_rep)
    x = tau * r_rep.Pbar
    a = jp.a
    delta = r_rep.delta
    eff = np.asarray(rr.effective, bool)
    spill = tau * r_rep.P
    ledger_m = float(E.sum() - delta.sum() - spill.sum() - r_rep.F[-1])
    ledger_s = float(jp.b0 + np.sum(a * delta) - jp.overhead.sum() - x.sum() - r_rep.b[-1])
    ev = rr.events
    cnt = dict(released=len(ev), run=sum(e["outcome"] == "run" for e in ev),
               resch=sum(e["outcome"] == "rescheduled" for e in ev), dropped=sum(e["outcome"] == "dropped" for e in ev))
    t0 = time.perf_counter()
    try:                                                   # independent cvxpy optimum of the FINAL problem (not timed in t_solve)
        ref_obj = float(reference_solution_joint(jp)[-1])
    except Exception as e:                                  # noqa: BLE001
        ref_obj = float("nan")
        row["reason"] = f"cvxpy reference unavailable: {type(e).__name__}"
    row["t_ref_s"] = time.perf_counter() - t0
    row.update(iterations=r.iters, solver_converged=bool(r.converged), cvxpy_objective=ref_obj,
               cvxpy_gap=ref_obj - float(r.objective), converged=rr.converged, objective=float(r_rep.objective),
               energy_shared_stored_J=float(np.sum(a * delta)), energy_sent_J=float(delta.sum()),
               master_final_level_J=float(r_rep.F[-1]), slave_final_level_J=float(r_rep.b[-1]),
               events_released=cnt["released"], events_run=cnt["run"] + cnt["resch"],
               events_rescheduled=cnt["resch"], events_dropped=cnt["dropped"],
               ledger_error_master_J=ledger_m, ledger_error_slave_J=ledger_s)
    # ---- checks ----------------------------------------------------------------------------------------
    fails = []
    if r_rep.b.min() < jp.B_min - LEVEL_TOL_J or r_rep.b_peak.max() > jp.B_max + LEVEL_TOL_J:
        fails.append(f"slave level outside [{jp.B_min:.4g}, {jp.B_max:.4g}] "
                     f"(min {r_rep.b.min():.6g}, peak {r_rep.b_peak.max():.6g})")
    fL = master_floor_profile(jp)
    capF = m.cap if np.isfinite(m.E_max) else np.full(N - 1, INF)
    if np.any(r_rep.F[:-1] < fL[1:N] - LEVEL_TOL_J) or r_rep.F[-1] < fL[N] - LEVEL_TOL_J or np.any(r_rep.F[:-1] > capF + LEVEL_TOL_J):
        fails.append("master level outside its bounds")
    prev = np.concatenate([[jp.b0], r_rep.b[:-1]])
    full = prev >= jp.B_max - LEVEL_TOL_J
    if np.any(delta[full] > 1e-9):
        fails.append("delta_k > 0 although the slave arrival is already full")
    if abs(ledger_m) > LEDGER_TOL_J or abs(ledger_s) > LEDGER_TOL_J:
        fails.append(f"ledger master {ledger_m:.2e} / slave {ledger_s:.2e} J")
    if np.any(x[eff] > tb["x_cap"] * (1 + 1e-9) + 1e-12):
        fails.append("x_i > x_cap in a run event")
    if np.any(x[eff] < tb["x_min"] * (1 - 1e-9)):
        fails.append("x_i < x_min in a run event")
    if np.any(np.abs(r_rep.Pbar[~eff]) > 1e-12):
        fails.append("Pbar != 0 in a sleep slot")
    if cnt["run"] + cnt["resch"] + cnt["dropped"] != cnt["released"]:
        fails.append("run + dropped != released")
    delays = [e["delay"] for e in ev if e["delay"] is not None]
    if delays and max(delays) > cfg.MAX_DELAY_SLOTS:
        fails.append(f"delay {max(delays)} > MAX_DELAY_SLOTS")
    if first is not None and not first[0]:
        fails.append(f"FIRST-pass joint_dwf did not converge ({first[2]} sweeps, objective {first[1]:.3g}): "
                     f"events dropped by the refusal cascade may be an optimiser stall, not infeasibility")
    if not r.converged:
        fails.append(f"joint_dwf did not converge ({r.iters} sweeps)")
    if np.isfinite(ref_obj) and ref_obj - r.objective > 1e-6 * max(1.0, abs(ref_obj)):
        fails.append(f"optimiser below cvxpy optimum by {ref_obj - r.objective:.3g} nats (cvxpy {ref_obj:.6g}, DWF {r.objective:.6g})")
    if stop_note:
        row["reason"] = stop_note
    if fails:
        row["status"], row["reason"] = "FAIL", "; ".join(fails) + (" | " + stop_note if stop_note else "")
    else:
        row["status"] = "PASS"
    return dict(objective=row["objective"], delta=delta.copy(), x=x.copy())


# --------------------------------------------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------------------------------------------
COLUMNS = ["scenario_id", "case", "overrides", "t_total_s", "t_solve_s", "t_other_s", "t_ref_s", "n_passes", "converged",
           "iterations", "solver_converged", "objective", "cvxpy_objective", "cvxpy_gap", "energy_shared_stored_J", "energy_sent_J", "master_final_level_J",
           "slave_final_level_J", "events_released", "events_run", "events_rescheduled", "events_dropped",
           "ledger_error_master_J", "ledger_error_slave_J", "status", "reason", "redesign_hint"]


def hardware() -> str:
    try:
        import scipy
        sp = scipy.__version__
    except Exception:                                                # noqa: BLE001
        sp = "n/a"
    return (f"{platform.platform()} | {platform.processor() or platform.machine()} | {os.cpu_count()} logical CPUs | "
            f"Python {platform.python_version()} | numpy {np.__version__} | scipy {sp}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="scenario id prefix, e.g. 3.4")
    ap.add_argument("--repeat", type=int, default=3, help="runs of the FIRST case of each scenario (noise estimate)")
    args = ap.parse_args(argv)
    todo = [s for s in SCENARIOS if args.only is None or s[0] == args.only or s[0].startswith(args.only)]
    if not todo:
        raise SystemExit(f"no scenario matches {args.only!r}")
    print("hardware:", hardware())
    print("DUMMY / placeholders:", "; ".join(DUMMY))
    t_suite = time.perf_counter()
    rows, results, cache, noise = [], {}, {}, {}

    def reference():
        key = "{}"
        if key not in cache:
            out = run_case("ref", "default (DWF optimum)", {})
            rows.append(out["row"])
            cache[key] = out
        return cache[key]

    seen_first = set()
    for sid, case, ov in todo:
        out = run_case(sid, case, ov)
        row, res = out["row"], out["res"]
        if not ov:
            cache["{}"] = out
        if sid not in seen_first:
            seen_first.add(sid)
            ts = [row["t_total_s"]]
            for _ in range(max(args.repeat, 1) - 1):
                ts.append(run_case(sid, case, ov)["row"]["t_total_s"])
            noise[sid] = (case, ts)
        rows.append(row)
        results[(sid, case)] = (row, res)
        print(f"[{sid}] {case:<22} {row['status']:<5} t={row['t_total_s']:7.1f}s obj={row.get('objective', float('nan')):.6g} "
              f"{row['reason'][:90]}", flush=True)

    # ---- cross-scenario checks -------------------------------------------------------------------------------
    notes = []
    obj = lambda k: results[k][0].get("objective", float("nan")) if k in results else float("nan")   # noqa: E731
    default = reference() if any(s[0] in BASELINES or s == ("3.5",) or s[0] == "3.5" for s in todo) else cache.get("{}")
    if default is not None:
        dres = default["res"]
        if ("3.5", "stub (neutral)") in results and dres:
            r5 = results[("3.5", "stub (neutral)")][1]
            if r5:
                d_obj = abs(r5["objective"] - dres["objective"])
                d_del = float(np.max(np.abs(r5["delta"] - dres["delta"])))
                ok = d_obj <= 1e-12 * max(1.0, abs(dres["objective"])) and d_del <= 1e-12 * max(1.0, np.max(dres["delta"]))
                notes.append(f"3.5 neutrality: |d objective| = {d_obj:.2e}, max|d delta| = {d_del:.2e} -> {'OK' if ok else 'VIOLATED'}")
                if not ok:
                    results[("3.5", "stub (neutral)")][0].update(status="FAIL", reason="stub differs from default run")
        for k in [k for k in results if k[0] in BASELINES]:
            b = obj(k)
            d = dres["objective"] if dres else float("nan")
            if np.isnan(b):
                notes.append(f"{k[0]} {k[1]}: baseline infeasible under the policy (DWF {d:.6g}) -> NEEDS REDESIGN")
            elif b > d * (1 + 1e-9) + 1e-12:
                notes.append(f"!!! {k[0]} {k[1]}: BASELINE {b:.6g} BEATS DWF {d:.6g} -> optimiser or modelling bug !!!")
            else:
                notes.append(f"{k[0]} {k[1]}: baseline {b:.6g} / DWF {d:.6g} = {b / d:.4f}  (gap {d - b:.4g} nats)")
    chain = [("3.4 n=1 >= n=2 >= n=3 (objective non-decreasing with harvest scale)",
              [("3.4", "n=3"), ("3.4", "n=2"), ("3.4", "n=1")]),
             ("3.3 slot0 <= first half <= all slots", [("3.3", "a: slot 0 only"), ("3.3", "b: first half"), ("3.3", "c: all slots")]),
             ("3.2 finite <= 3.1 infinite", [("3.2", "Bmax=5000 Bmin=1000"), ("3.1", "inf master battery")])]
    for label, keys in chain:
        if all(k in results for k in keys):
            v = [0.0 if np.isnan(obj(k)) else obj(k) for k in keys]   # infeasible = no nats
            ok = all(v[i] <= v[i + 1] * (1 + 1e-9) + 1e-12 for i in range(len(v) - 1))
            notes.append(f"monotonicity {label}: {['%.6g' % x for x in v]} -> {'OK' if ok else 'VIOLATED'}")

    # ---- output ----------------------------------------------------------------------------------------------
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HERE, "results", f"scenarios_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n" + "=" * 118)
    print(f"{'id':<5}{'case':<22}{'t_total':>9}{'t_solve':>9}{'passes':>7}{'objective':>14}{'shared[J]':>11}"
          f"{'run/rel':>8}{'drop':>5}  status")
    for r in rows:
        print(f"{r['scenario_id']:<5}{r['case']:<22}{r['t_total_s']:9.1f}{r['t_solve_s']:9.1f}{r.get('n_passes', 0):7d}"
              f"{r.get('objective', float('nan')):14.6g}{r.get('energy_shared_stored_J', float('nan')):11.4g}"
              f"{str(r.get('events_run', '-')) + '/' + str(r.get('events_released', '-')):>8}"
              f"{str(r.get('events_dropped', '-')):>5}  {r['status']}  {r['reason'][:60]}")
    print("\ncross-scenario notes:")
    for n in notes:
        print("  " + n)
    print("\ntiming noise (first case of each scenario):")
    for sid, (case, ts) in noise.items():
        print(f"  {sid} {case}: n={len(ts)} mean {np.mean(ts):.2f}s  spread (max-min) {max(ts) - min(ts):.2f}s")
    slow = sorted(rows, key=lambda r: -r["t_total_s"])[:3]
    print(f"\ntotal suite time {time.perf_counter() - t_suite:.1f}s; slowest: " +
          ", ".join(f"{r['scenario_id']} {r['case']} ({r['t_total_s']:.1f}s)" for r in slow))
    print(f"wrote {path}")
    bad = [r for r in rows if r["status"] == "FAIL"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
