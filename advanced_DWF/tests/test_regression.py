"""pytest entry point for the golden-master check (see regression.py).  Run:  python -m pytest tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression  # noqa: E402


def test_matches_golden_master():
    diffs = regression.check()
    assert not diffs, "\n".join(diffs)


# ---- stage 6: fast checks of the gate numbers (implementation.md sections 6 and 8) ----
sys.path.insert(0, regression.PROJECT)


def test_event_load_numbers():
    import slave_config as cfg
    import slave_load
    p = slave_load.event_profile(cfg.SLAVE_SCHEDULE)
    ev, sl = int(p["is_event"].argmax()), int(p["is_event"].argmin())
    assert abs(p["c"][ev] - 0.19841) < 1e-5 and abs(p["c"][sl] - 0.015962) < 1e-5
    assert abs(slave_load.event_profile(cfg.SLAVE_SCHEDULE, psram_state="dpd")["c"][sl] - 0.003218) < 1e-5
    assert int(p["is_event"].sum()) == 7 and abs(p["c"].sum() - 1.7560) < 1e-4
    assert abs(p["t_tx"] - 0.016777216) < 1e-12 and abs(p["x_cap"] - 3.3554e-3) < 1e-7 and abs(p["x_min"] - 1.6777e-4) < 1e-8


def test_thz_link_numbers():
    import numpy as np
    from dwf2d import channels as ch
    assert abs(ch.noise_floor_w() - 4.0039e-11) < 1e-14
    assert abs(ch.near_field_limit_m(0.3e12, 10, 10) - 0.7958e-3) < 1e-7
    for f, want in ((0.3e12, 7.2e3), (0.1e12, 6.48e3)):
        hb = ch.thz_hbar(f, 1e-3, eta_tx=0.1)["hbar"]
        assert abs(hb - want) / want < 2e-3
    hb = ch.thz_hbar(0.3e12, 1e-3, eta_tx=0.1)["hbar"]
    assert abs(0.5 * 0.016777216 * np.log1p(hb * 0.2) - 0.0610) < 1e-4
    assert ch.thz_hbar(0.3e12, 0.5e-3, eta_tx=0.1, warn=False)["clamped"]


def test_reschedule_and_bookkeeping():
    import slave_load
    sch = [1, 1, 0, 0, 1, 0]
    eff, rel, moves, dropped = slave_load.reschedule(sch, [0, 4], {0: 0, 1: 1, 4: 4}, max_delay=1)
    assert moves == [(4, 5, 4)] and dropped == [(0, 0)] and list(eff) == [0, 1, 0, 0, 0, 1]


def test_integrated_problem_exact_and_postpass():
    import numpy as np
    from dwf2d.demos import _integrated_problem
    from dwf2d.solver import joint_dwf, maximal_sharing
    from dwf2d.verification import check_maximal_sharing, reference_solution_joint
    jp0, build, _ = _integrated_problem()
    jp = build(jp0.slave_schedule)
    r = joint_dwf(jp, tol=1e-15)
    _, _, _, ref = reference_solution_joint(jp)
    assert abs(ref - r.objective) <= 1e-6 * abs(ref)
    r2, info = maximal_sharing(jp, r)
    ck = check_maximal_sharing(jp, r, r2, info)
    assert ck["applied"] and ck["objective_change"] == 0 and ck["max_x_change_J"] == 0 and ck["every_slot_stopped"]
    assert ck["min_level_margin_J"] >= -1e-12 and ck["max_peak_over_B_max_J"] <= 1e-12
    assert np.all(60 * r.Pbar[jp.slave_schedule] >= 1.6777e-4)
