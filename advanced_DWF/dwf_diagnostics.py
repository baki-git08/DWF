"""
dwf_diagnostics.py
==================
Post-hoc diagnostics for dwf_master_slave.py. It only READS a JointParams and a JointResult;
the solver is not touched.

Usage
-----
    from dwf_diagnostics import diagnose
    r = joint_dwf(jp)
    d = diagnose(jp, r)                       # prints a report, returns the numbers as a dict

    # optional: pass the physical parameters behind alpha, h, hbar so they can be cross-checked
    meta = dict(optical_wavelength_nm=450, optical_atten_per_m=0.0399, optical_distance_m=0.001,
                thz_freq_THz=0.30, thz_alpha_m=12300, thz_distance_m=0.001)
    d = diagnose(jp, r, meta=meta)

Tweak ONE variable, run diagnose() again, and compare the two reports section by section.

Report layout
-------------
    1. CHANNEL CONDITIONS         optical energy link (alpha, beta, a = beta*alpha, losses, slave battery)
    2. COMMUNICATION LINKS        THz data links (h, hbar, SNR, regime, rates, implied reference gain h0)
    3. SHARING DECISION           does the master share, and WHY it stops (or never starts)
    4. ENERGY LEDGER              where every Joule went
    5. ACTIVE CONSTRAINTS         which limits are binding
    6. SOLVER                     objective, iterations, convergence

Notation (same as dwf_master_slave.py): a = beta*alpha is the net gain of the energy tap,
m_k = th1*h/(2(1+h*P)) is the master's marginal value of a Joule (nats per Joule),
s_k = th2*hbar/(2(1+hbar*Pbar)) is the slave's marginal value, and the master shares only while
a*s_k > m_k (see the docstring of diagnose()).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _dB(x):
    """10*log10(x), safe for zeros."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def _rng(v, fmt: str = ".4g") -> str:
    """'min .. | mean .. | max ..' summary of an array."""
    v = np.asarray(v, float)
    return f"min {v.min():{fmt}} | mean {v.mean():{fmt}} | max {v.max():{fmt}}"


def _regime(snr: float) -> str:
    """Plain-language label for the operating point on the log rate curve ln(1 + SNR)."""
    if snr < 0.1:
        return "LOW SNR: rate ~ linear in power, the log curve has not started to bend"
    if snr < 10:
        return "MID SNR: the log curve is bending (diminishing returns matter)"
    return "HIGH SNR: the log curve is strongly saturated"


# ----------------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------------
def diagnose(jp, r, tol: float = 1e-6, verbose: bool = True, meta: dict | None = None, load: dict | None = None) -> dict:
    """Explain WHY a solution looks the way it does. Prints a six-section report, returns a dict.

    Parameters
    ----------
    jp : JointParams          the problem that was solved
    r : JointResult           the result of joint_dwf(jp)
    tol : relative tolerance used to decide "zero" / "active" (relative to mean harvest)
    verbose : print the report (set False to only get the dict)
    meta : optional dict with the physical numbers behind alpha, h, hbar. Recognised keys:
        optical_wavelength_nm, optical_atten_per_m, optical_distance_m   (energy link)
        thz_freq_THz, thz_alpha_m, thz_distance_m                         (data links)
        They are only used for cross-checks (does exp(-c*d) equal alpha? what h0 is implied?).
    load : optional dict from slave_load.make_load(jp) (components, c, sizing, timing banner). When given,
        the report also shows the dummy-timing banner, the slave electronics energy per component in the
        ledger, the MPPT-cap (S5) status, the battery-window check against the profile's W_req, and the
        supercap sizing summary.  Without it the report is exactly as before.

    1. CHANNEL CONDITIONS (optical energy link, master -> slave)
    ------------------------------------------------------------
    alpha        Beer-Lambert transmittance exp(-c*d): fraction of the sent power reaching the slave.
    beta         battery-recharging efficiency (a second, separate loss).
    a            a = beta*alpha, the net gain of the tap; loss in dB = -10*log10(a). Only a*delta lands.
    per Joule    of every Joule sent: (1-alpha) is lost in the channel, alpha*(1-beta) in recharging,
                 a = beta*alpha is stored. Tells you WHICH loss dominates.
    battery      slave window [B_min, B_max] and initial charge b0.

    2. COMMUNICATION LINKS (data links; noise-normalised, received power = h*P)
    ----------------------------------------------------------------------------
    h, hbar      SNR per watt of the master link and of the slave (return) link; 1/h is the "floor"
                 of the water-filling picture (the power you must supply before the rate moves).
    hbar/h       link ratio: >1 means the slave's own link is better than the master's.
    SNR          mean h*P and hbar*Pbar at the optimum, in linear and dB, with a regime label:
                 LOW (<0.1) the rate is almost linear in power, MID the log curve bends, HIGH saturates.
                 In the LOW regime marginal values are almost constant (about th*h/2), so the split
                 is decided only by ratios (a, hbar/h, th2/th1), not by absolute power.
    h0           if meta has the absorption coefficient and distance: implied reference gain
                 h0 = h / exp(-alpha_thz*d). h0 = 1 means the link budget (antenna gains, spreading,
                 noise power) has NOT been calibrated, so absolute rates are not meaningful.

    3. SHARING DECISION
    -------------------
    Marginal values (nats per Joule):  m_k = th1*h/(2(1+h*P)),  s_k = th2*hbar/(2(1+hbar*Pbar)).
    Sending one Joule costs the master m_k and gives the slave a*s_k, so the master shares while
    a*s_k > m_k and stops where a*s_k = m_k (interior optimum) or when a constraint blocks it.
    Sharing test at the "share nothing" start, per slot:
        R = a * (th2/th1) * (hbar/h) * (1 + h*P0)        with P0 = harvest power E/tau
    R > 1 -> sharing pays; R <= 1 -> it does not. The four factors are printed separately so you
    can see which one pulls against sharing, plus the BREAK-EVEN value each factor would need
    (others fixed). Per-slot outcome at the optimum is classified as:
        BALANCED     delta > 0 and m = a*s: the master gave energy until the marginals met
        CAP          delta > 0 but the slave battery is full on arrival (B_max) - more would be wasted
        UNBALANCED   delta > 0 but m != a*s and no cap: not converged / another constraint
        NOT_WORTH    delta = 0 and m >= a*s: a Joule is worth more to the master than to the slave
        CAP_BLOCKED  delta = 0, a*s > m, but the slave battery is full
        BLOCKED      delta = 0, a*s > m and no cap: another constraint (or not converged)
    A one-line HEADLINE says whether and why the master stops sharing.

    4-6. energy ledger, active constraints, solver status (as before).
    Note: the master carry F_n and slave level b_n are not unique when both batteries have slack
    (only P and Pbar are), so "master taps closed 59/59" can just mean the smoothing moved to the slave.
    """
    m, tau, h, hb, a = jp.master, jp.master.tau, jp.master.h, jp.h_bar, jp.a
    th1, th2 = jp.theta1, jp.theta2
    N, E = m.N, m.E
    meta = meta or {}
    out: dict = {}
    sc = max(E.mean(), 1.0) * tol                       # "zero" threshold in Joules
    P0 = E / tau                                        # harvest power = master power if nothing is shared

    # ---------------- 1. channel conditions ------------------------------------------------
    alpha, beta = jp.alpha, jp.beta
    ch = {
        "alpha": alpha, "beta": beta, "a": a,
        "alpha_dB": float(_dB(alpha.mean())), "a_dB": float(_dB(a.mean())),
        "per_joule_channel_loss": float((1 - alpha).mean()),
        "per_joule_recharge_loss": float((alpha * (1 - beta)).mean()),
        "per_joule_stored": float(a.mean()),
        "B_min": jp.B_min, "B_max": jp.B_max, "b0": jp.b0,
    }
    if "optical_atten_per_m" in meta and "optical_distance_m" in meta:
        ch["alpha_from_meta"] = float(np.exp(-meta["optical_atten_per_m"] * meta["optical_distance_m"]))
        ch["alpha_matches_meta"] = bool(abs(ch["alpha_from_meta"] - alpha.mean()) <= 1e-3 * max(alpha.mean(), 1e-12))
    out["channel"] = ch

    # ---------------- 2. communication links -------------------------------------------------
    snr_m, snr_s = float(np.mean(h * r.P)), float(np.mean(hb * r.Pbar))
    lk = {
        "h_mean": float(h.mean()), "hbar_mean": float(hb.mean()),
        "floor_master": float((1 / h).mean()), "floor_slave": float((1 / hb).mean()),
        "link_ratio": float((hb / h).mean()),
        "P_mean": float(r.P.mean()), "Pbar_mean": float(r.Pbar.mean()),
        "snr_master": snr_m, "snr_slave": snr_s,
        "snr_master_dB": float(_dB(snr_m)), "snr_slave_dB": float(_dB(snr_s)),
        "regime_master": _regime(snr_m), "regime_slave": _regime(snr_s) if snr_s > 0 else "slave transmits nothing",
        "rate_master": r.rate_master, "rate_slave": r.rate_slave,
    }
    if "thz_alpha_m" in meta and "thz_distance_m" in meta:
        t_thz = float(np.exp(-meta["thz_alpha_m"] * meta["thz_distance_m"]))
        lk["thz_transmittance"] = t_thz
        lk["implied_h0_master"] = float(h.mean() / t_thz)
        lk["implied_h0_slave"] = float(hb.mean() / t_thz)
        lk["h0_uncalibrated"] = bool(abs(lk["implied_h0_master"] - 1.0) < 1e-6 and abs(lk["implied_h0_slave"] - 1.0) < 1e-6)
    out["links"] = lk

    # ---------------- 3. sharing decision -----------------------------------------------------
    mk = th1 * h / (2 * (1 + h * r.P))                  # master marginal value at the optimum
    sk = th2 * hb / (2 * (1 + hb * r.Pbar))             # slave  marginal value at the optimum
    m0 = th1 * h / (2 * (1 + h * P0))                   # master marginal value if it shared nothing
    s0 = th2 * hb / 2                                   # slave  marginal value at Pbar = 0
    R_slot = a * s0 / m0                                # sharing test per slot
    f_a, f_w = float(a.mean()), th2 / th1               # the four factors of R
    f_h, f_sat = float((hb / h).mean()), float((1 + h * P0).mean())
    factors = {"a = beta*alpha (tap gain)": f_a, "th2/th1 (rate weight ratio)": f_w,
               "hbar/h (slave vs master link)": f_h, "1+h*P0 (master log saturation)": f_sat}
    breakeven = {
        "a needed": 1.0 / (f_w * f_h * f_sat),
        "th2/th1 needed": 1.0 / (f_a * f_h * f_sat),
        "hbar/h needed": 1.0 / (f_a * f_w * f_sat),
        "master SNR h*P0 needed": max(1.0 / (f_a * f_w * f_h) - 1.0, 0.0),
    }
    gap_rel = (mk - a * sk) / mk                        # >0: master keeps; 0: balanced; <0: sending would pay
    on = r.delta > sc
    cap = r.b_peak >= jp.B_max - sc
    capA = jp.capA
    s5 = np.isfinite(capA) & (a * r.delta >= capA - sc)            # MPPT input rating binds (S5)
    silent = (~jp.slave_schedule) & bool(np.any(jp.overhead > 0))   # FEED label only when a slave load is attached
    reason = np.empty(N, dtype=object)
    for k in range(N):
        if on[k]:
            if silent[k]:
                reason[k] = "FEED"                                 # silent slot: energy stored for later / electronics drain
            elif cap[k]:
                reason[k] = "CAP"
            elif s5[k]:
                reason[k] = "MPPT_CAP"
            else:
                reason[k] = "BALANCED" if abs(gap_rel[k]) < 1e-4 else "UNBALANCED"
        else:
            reason[k] = "NOT_WORTH" if gap_rel[k] >= -1e-9 else ("CAP_BLOCKED" if cap[k] else "BLOCKED")
    counts = {k: int((reason == k).sum()) for k in ("BALANCED", "CAP", "UNBALANCED", "NOT_WORTH", "CAP_BLOCKED", "BLOCKED",
                                                     "MPPT_CAP", "FEED")}

    sent, etot = float(r.delta.sum()), float(E.sum())
    share = sent / etot if etot else 0.0
    if counts["BALANCED"] + counts["CAP"] + counts["UNBALANCED"] + counts["MPPT_CAP"] + counts["FEED"] == 0:
        if counts["NOT_WORTH"] == N:
            worst = min(factors, key=factors.get)
            headline = (f"MASTER DOES NOT SHARE. A Joule is worth {m0.mean():.4g} to the master but only "
                        f"{(a * s0).mean():.4g} to the slave after the tap loss, so R = {R_slot.mean():.3g} <= 1. "
                        f"Main obstacle: {worst} = {factors[worst]:.3g}.")
        else:
            headline = (f"MASTER DOES NOT SHARE, but not only because it is unprofitable: NOT_WORTH {counts['NOT_WORTH']}, "
                        f"CAP_BLOCKED {counts['CAP_BLOCKED']}, BLOCKED {counts['BLOCKED']} of {N} slots.")
    else:
        headline = (f"MASTER SHARES {100 * share:.1f}% of its harvest. Master power {P0.mean():.4g} -> {r.P.mean():.4g} W "
                    f"(its marginal value rose {m0.mean():.3g} -> {mk.mean():.3g}); slave power 0 -> {r.Pbar.mean():.4g} W "
                    f"(a*s fell {(a * s0).mean():.3g} -> {(a * sk).mean():.3g}). ")
        if counts["BALANCED"]:
            headline += f"It STOPS where the marginals balance (m = a*s) in {counts['BALANCED']} of {N} slots. "
        if counts["CAP"]:
            headline += f"In {counts['CAP']} slots the slave battery is full on arrival (B_max), so more would be wasted. "
        if counts["MPPT_CAP"]:
            headline += f"In {counts['MPPT_CAP']} slots the MPPT input rating (S5) limits the stored energy. "
        if counts["FEED"]:
            headline += f"{counts['FEED']} silent slots receive energy to cover the slave electronics drain. "
        if counts["UNBALANCED"]:
            headline += f"{counts['UNBALANCED']} slots are unbalanced (not converged or another constraint)."
    out["decision"] = {
        "R_mean": float(R_slot.mean()), "factors": factors, "breakeven": breakeven,
        "m0_mean": float(m0.mean()), "a_s0_mean": float((a * s0).mean()),
        "m_mean": float(mk.mean()), "a_s_mean": float((a * sk).mean()),
        "P0_mean": float(P0.mean()), "P_mean": float(r.P.mean()), "Pbar_mean": float(r.Pbar.mean()),
        "reason_counts": counts, "reason_per_slot": [str(x) for x in reason], "headline": headline.strip(),
    }

    # ---------------- 4-6. ledger, active constraints, solver --------------------------------
    out["ledger"] = {
        "harvested": etot, "master_data": float(tau * r.P.sum()), "sent": sent,
        "master_leftover": float(r.F[-1]),
        "channel_loss": float(((1 - alpha) * r.delta).sum()),
        "recharge_loss": float((alpha * (1 - beta) * r.delta).sum()),
        "stored": float((a * r.delta).sum()), "slave_data": float(tau * r.Pbar.sum()),
        "battery_change": float(r.b[-1] - jp.b0),
        "slave_overhead": float(jp.overhead.sum()),
    }
    out["active"] = {
        "master_taps_closed": int((r.F[:-1] <= sc).sum()), "master_taps": N - 1,
        "slave_at_floor": int((r.b <= jp.B_min + sc).sum()),
        "slave_at_ceiling": int((r.b_peak >= jp.B_max - sc).sum()),
        "P_zero_slots": int((r.P <= sc / tau).sum()), "Pbar_zero_slots": int((r.Pbar <= sc / tau).sum()),
        "delta_zero_slots": int((r.delta <= sc).sum()), "slots": N,
    }
    out["shared_fraction"] = share
    # ---- slave load (only reported when attached) ----
    stored_cap = np.where(np.isfinite(capA), a * r.delta, 0.0)
    out["mppt_cap"] = {
        "active": bool(np.isfinite(capA).any()),
        "binding_slots": int(s5.sum()),
        "max_stored_J": float((a * r.delta).max()),
        "capA_J": float(capA[np.isfinite(capA)].min()) if np.isfinite(capA).any() else float("inf"),
        "max_fraction_of_cap": float(np.max(stored_cap / np.where(np.isfinite(capA), capA, 1.0))) if np.isfinite(capA).any() else 0.0,
    }
    out["window"] = {"W": jp.W, "max_b_peak_minus_Bmin": float(r.b_peak.max() - jp.B_min)}
    if load is not None:
        sz = load["sizing"]
        out["window"].update(W_req=sz["W_req"], W_design=sz["W_design"],
                             ok=bool(jp.W + 1e-9 >= out["window"]["max_b_peak_minus_Bmin"]),
                             covers_W_req=bool(jp.W + 1e-9 >= sz["W_req"]))
        out["load"] = {"banner": load.get("banner", ""),
                       "components_J": {k: float(load[k].sum()) for k in ("camera", "mcu", "flash", "mppt")},
                       "sizing": dict(sz)}
    # backwards-compatible keys
    out["sharing_ratio"], out["m_mean"], out["a_s_mean"] = out["decision"]["R_mean"], float(mk.mean()), float((a * sk).mean())
    out["snr_master"], out["snr_slave"] = snr_m, snr_s

    if not verbose:
        return out

    # ================================== printing ==============================================
    L, A = out["ledger"], out["active"]
    line = "=" * 92
    if load is not None and load.get("banner"):
        print(load["banner"])
    print(line)
    print("1. CHANNEL CONDITIONS  (optical energy link, master -> slave)")
    print(line)
    print(f"  alpha (transmittance)  : {_rng(alpha)}   -> {-ch['alpha_dB']:.3g} dB channel loss")
    print(f"  beta  (recharge eff.)  : {_rng(beta)}          -> {-float(_dB(np.mean(beta))):.3g} dB recharge loss (mean)")
    print(f"  a = beta*alpha         : {_rng(a)}   -> net tap loss {-ch['a_dB']:.3g} dB")
    print(f"  per Joule sent         : {100 * ch['per_joule_channel_loss']:.2f}% lost in the channel, "
          f"{100 * ch['per_joule_recharge_loss']:.2f}% lost in recharging, {100 * ch['per_joule_stored']:.2f}% stored")
    dom = "recharging efficiency (beta)" if ch["per_joule_recharge_loss"] > ch["per_joule_channel_loss"] else "channel attenuation (alpha)"
    print(f"                           dominant loss: {dom}")
    print(f"  slave battery          : B_min {jp.B_min:g} J, B_max {jp.B_max:g} J (window {jp.W:g} J), initial b0 {jp.b0:g} J")
    if "alpha_from_meta" in ch:
        ok = "matches" if ch["alpha_matches_meta"] else "DOES NOT MATCH"
        print(f"  [meta] optical         : {meta.get('optical_wavelength_nm', '?')} nm, c = {meta['optical_atten_per_m']:g} 1/m, "
              f"d = {meta['optical_distance_m']:g} m -> exp(-c*d) = {ch['alpha_from_meta']:.6g}  ({ok} alpha)")
    print()
    print(line)
    print("2. COMMUNICATION LINKS  (data links; noise-normalised, received power = h*P)")
    print(line)
    print(f"  master link  h         : {_rng(h)}   floor 1/h = {lk['floor_master']:.4g} W   value of first Joule th1*h/2 = {th1 * h.mean() / 2:.4g}")
    print(f"  slave link   hbar      : {_rng(hb)}   floor 1/hbar = {lk['floor_slave']:.4g} W   value of first Joule th2*hbar/2 = {th2 * hb.mean() / 2:.4g}")
    rel = "slave link is BETTER" if lk["link_ratio"] > 1.0 + 1e-9 else ("links are EQUAL" if abs(lk["link_ratio"] - 1) <= 1e-9 else "slave link is WORSE")
    print(f"  link ratio hbar/h      : {lk['link_ratio']:.4g}  ({rel})")
    print(f"  master link at optimum : P = {lk['P_mean']:.4g} W, SNR h*P = {snr_m:.4g} ({lk['snr_master_dB']:.3g} dB), rate {r.rate_master:.4g} nats")
    print(f"                           {lk['regime_master']}")
    print(f"  slave  link at optimum : Pbar = {lk['Pbar_mean']:.4g} W, SNR hbar*Pbar = {snr_s:.4g} "
          f"({(str(round(lk['snr_slave_dB'], 3)) + ' dB') if snr_s > 0 else 'n/a'}), rate {r.rate_slave:.4g} nats")
    print(f"                           {lk['regime_slave']}")
    if "implied_h0_master" in lk:
        print(f"  [meta] THz             : f = {meta.get('thz_freq_THz', '?')} THz, alpha = {meta['thz_alpha_m']:g} 1/m, "
              f"d = {meta['thz_distance_m']:g} m -> transmittance exp(-alpha*d) = {lk['thz_transmittance']:.4g}")
        print(f"                           implied reference gain h0 = h/transmittance: master {lk['implied_h0_master']:.4g}, slave {lk['implied_h0_slave']:.4g}")
        if lk["h0_uncalibrated"]:
            print("                           WARNING: h0 = 1 -> link budget (antenna gains, spreading loss, noise power) is NOT calibrated;"
                  " absolute rates are not meaningful")
    print()
    print(line)
    print("3. SHARING DECISION  (does the master send energy, and why does it stop?)")
    print(line)
    print(f"  HEADLINE: {out['decision']['headline']}")
    print()
    print(f"  Marginal value of one Joule (nats/J)      before sharing        at the optimum")
    print(f"    master  m = th1*h/(2(1+hP))             {m0.mean():<20.4g}  {mk.mean():<.4g}")
    print(f"    slave   a*s  (after the tap loss)       {(a * s0).mean():<20.4g}  {(a * sk).mean():<.4g}")
    print(f"    net per Joule sent  (a*s - m)           {(a * s0 - m0).mean():<20.4g}  {(a * sk - mk).mean():<.4g}   (sends while > 0)")
    print(f"    powers: master {P0.mean():.4g} -> {r.P.mean():.4g} W ;  slave 0 -> {r.Pbar.mean():.4g} W")
    print()
    print(f"  Sharing test R = a * (th2/th1) * (hbar/h) * (1 + h*P0) = {R_slot.mean():.4g}   (>1 -> sharing pays)")
    for name, val in factors.items():
        tag = "  <-- pulls AGAINST sharing" if val < 1.0 - 1e-3 else ("  <-- helps sharing" if val > 1.0 + 1e-3 else "  (neutral)")
        print(f"    {name:<34s} x {val:<10.4g}{tag}")
    if R_slot.mean() <= 1.0:
        print("  To make sharing pay (change ONE, others fixed):")
        for name, val in breakeven.items():
            print(f"    {name:<24s} > {val:.4g}")
    print()
    print(f"  Per-slot outcome ({N} slots):  " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    expl = {"BALANCED": "master gave energy until m = a*s", "CAP": "slave battery full on arrival",
            "UNBALANCED": "not converged or another constraint", "NOT_WORTH": "m >= a*s, keeping the Joule is better",
            "CAP_BLOCKED": "would pay but slave battery is full", "BLOCKED": "would pay but another constraint binds",
            "MPPT_CAP": "delta > 0 but the MPPT input rating (S5) limits the stored energy",
            "FEED": "delta > 0 in a silent slot: energy stored for later / electronics drain"}
    for k, v in counts.items():
        if v:
            print(f"    {k:<12s} {v:>3d} slots : {expl[k]}")
    print()
    print(line)
    print("4. ENERGY LEDGER  (Joules)")
    print(line)
    print("  harvested %.1f = master data %.1f + sent %.1f (+ master leftover %.2g)" % (L["harvested"], L["master_data"], L["sent"], L["master_leftover"]))
    print("  sent %.1f -> channel loss %.1f, recharge loss %.1f, stored %.1f" % (L["sent"], L["channel_loss"], L["recharge_loss"], L["stored"]))
    if L["slave_overhead"] > 0:
        print("  stored %.1f + initial excess %.1f = slave data %.1f + slave electronics %.4g + final level above floor %.3g"
              % (L["stored"], jp.g_init, L["slave_data"], L["slave_overhead"], r.b[-1] - jp.B_min))
        if load is not None:
            comp = out["load"]["components_J"]
            print("  slave electronics (non-TX) %.4g J = " % L["slave_overhead"]
                  + ", ".join(f"{k} {v:.4g}" for k, v in comp.items()))
    else:
        print("  stored %.1f = slave data %.1f + battery change %.1f" % (L["stored"], L["slave_data"], L["battery_change"]))
    print()
    print(line)
    print("5. ACTIVE CONSTRAINTS")
    print(line)
    print(f"  master taps closed (F=0): {A['master_taps_closed']}/{A['master_taps']} | slave at floor: {A['slave_at_floor']}/{A['slots']}"
          f" | slave at ceiling on arrival: {A['slave_at_ceiling']}/{A['slots']}")
    print(f"  P=0 slots: {A['P_zero_slots']} | Pbar=0 slots: {A['Pbar_zero_slots']} | delta=0 slots: {A['delta_zero_slots']}")
    if load is not None:
        mc, wn = out["mppt_cap"], out["window"]
        if mc["active"]:
            print(f"  MPPT cap (S5): binds in {mc['binding_slots']}/{N} slots | largest stored energy {mc['max_stored_J']:.4g} J "
                  f"= {100 * mc['max_fraction_of_cap']:.1f}% of capA = {mc['capA_J']:.4g} J")
        else:
            print("  MPPT cap (S5): not set")
        print(f"  battery window: W = {wn['W']:.4g} J | solution needs max(b_peak) - B_min = {wn['max_b_peak_minus_Bmin']:.4g} J "
              f"| profile W_req = {wn['W_req']:.4g} J (design x margin {wn['W_design']:.4g} J) -> "
              + ("window covers W_req" if wn["covers_W_req"] else "WINDOW BELOW W_req (the placeholder-TX profile would not fit)"))
        sz = out["load"]["sizing"]
        print(f"  supercap sizing: C >= {sz['C_req']:.3f} F for the {load['hw'].v_store_min}-{load['hw'].v_store_max} V range "
              f"-> nearest standard {sz['C_std']} F (window {sz['W_std']:.3f} J)")
    print()
    print(line)
    print("6. SOLVER")
    print(line)
    print(f"  objective {r.objective:.6g} nats = th1*{r.rate_master:.4g} + th2*{r.rate_slave:.4g} | shared fraction of harvest {100 * share:.1f}%"
          f" | iters {r.iters} | converged {r.converged}")
    return out


# ----------------------------------------------------------------------------------------------
# Stage 6: the master has no data link (theta1 = 0); the slave sends THz bursts in events
# ----------------------------------------------------------------------------------------------
def link_table(jp, r, events, link: dict, t_tx: float, x_min: float, x_cap: float,
               ber_target: float = 1e-3, modulation: str = "16-QAM", mode: str = "proportional") -> list[dict]:
    """One row per released event (spec 4.8):
        x [J]        burst energy (proportional: tau*Pbar at the run slot; fixed_bias: x_cap)
        P_pk = x/t_tx [W] electrical peak,  radiated = eta_tx * P_pk,  SNR_burst = hbar * P_pk
        BER          theoretical, ber_func(SNR/bits_per_symbol) of the modulation (Gray-coded QAM)
        pass         BER <= ber_target ;  nats  (t_tx/2) ln(1 + SNR_burst)  (unweighted by theta2)
        at_cap       x >= x_cap (1e-9 relative) ;  outcome / release / run / delay from the refusal loop."""
    from dwf2d.channels import qam_scheme
    sch = qam_scheme(modulation)
    k = sch["bits_per_symbol"]
    hb = float(np.mean(jp.h_bar))
    rows = []
    for e in events:
        row = dict(e)
        if e["outcome"] in ("run", "rescheduled"):
            x = x_cap if mode == "fixed_bias" else float(jp.master.tau * r.Pbar[e["run"]])
            p_pk = x / t_tx
            snr = hb * p_pk
            ber = float(sch["ber_func"](snr / k))
            row.update(x_J=x, P_pk_W=p_pk, radiated_W=link["eta_tx"] * p_pk, snr=snr,
                       snr_dB=float(10 * np.log10(snr)) if snr > 0 else -np.inf, ber=ber, ber_pass=bool(ber <= ber_target),
                       nats=0.5 * t_tx * float(np.log1p(snr)), at_cap=bool(x >= x_cap * (1 - 1e-9)),
                       below_floor=bool(x < x_min * (1 - 1e-9)))
        else:
            row.update(x_J=0.0, P_pk_W=0.0, radiated_W=0.0, snr=0.0, snr_dB=-np.inf, ber=np.nan, ber_pass=False,
                       nats=0.0, at_cap=False, below_floor=False)
        rows.append(row)
    return rows


def link_report(jp, rr, link: dict, rows: list[dict], profile: dict, sizing: dict | None = None,
                ber_target: float = 1e-3, banner: str = "", verbose: bool = True) -> dict:
    """Print the stage-6 THz link report: per-event table (x, P_pk, radiated power, SNR, BER pass/fail, nats),
    counts (at cap, refused, rescheduled, dropped), supercap sizing (W_req vs W), DUMMY banner, near-field flag.
    rr is the RefusalResult of physical.solve_with_refusals.  Returns the summary dict."""
    c = rr.counts
    refused = sum(len(h["refused"]) for h in rr.history)
    run_rows = [x for x in rows if x["outcome"] in ("run", "rescheduled")]
    out = dict(counts=c, refusals=refused, passes=rr.passes, converged=rr.converged,
               at_cap=sum(x["at_cap"] for x in run_rows), ber_fail=sum(not x["ber_pass"] for x in run_rows),
               nats_total=float(sum(x["nats"] for x in run_rows)), hbar=float(np.mean(jp.h_bar)),
               near_field_clamped=bool(link["clamped"]), W=jp.W)
    if sizing is not None:
        out.update(W_req=sizing["W_req"], W_design=sizing["W_design"])
    if not verbose:
        return out
    line = "=" * 92
    if banner:
        print(banner)
    print(line)
    print("THz LINK REPORT  (slave -> third-party receiver; one burst per event)")
    print(line)
    clamp = ", CLAMPED to d_nf: d is inside the Friis-invalid region" if link["clamped"] else ""
    print(f"  preset {link['preset']} ({link['freq_label']}), G_tx/G_rx = {link['g_tx_dbi']:g}/{link['g_rx_dbi']:g} dBi, "
          f"d = {link['d_m'] * 1e3:.4g} mm (Friis valid beyond {link['d_nf'] * 1e3:.4g} mm{clamp})")
    print(f"  noise N = {link['noise_w']:.5g} W ({10 * np.log10(link['noise_w']):.2f} dBW), coupling loss {link['coupling_db']:.2f} dB, "
          f"eta_tx = {link['eta_tx']:g}  ->  hbar = {out['hbar']:.5g} 1/W (electrical burst power -> SNR)")
    print(f"  t_tx = {profile['t_tx'] * 1e3:.6f} ms, x_min = {profile['x_min'] * 1e3:.5f} mJ (1 mW radiated), "
          f"x_cap = {profile['x_cap'] * 1e3:.4f} mJ (20 mW radiated), BER target {ber_target:g}")
    unres = f" | UNRESOLVED {c['unresolved']}" if c.get("unresolved") else ""
    conv = "" if rr.converged else "  (NOT CONVERGED)"
    print(f"  events: released {c['released']} | run {c['run']} | rescheduled {c['rescheduled']} | dropped {c['dropped']}"
          f"{unres} | refusals {refused} in {rr.passes} pass(es){conv}")
    print(f"  bursts at the cap: {out['at_cap']}/{len(run_rows)} | BER above target: {out['ber_fail']} | "
          f"nats credited {out['nats_total']:.4g} (x theta2 = {jp.theta2 * out['nats_total']:.4g})")
    if sizing is not None:
        ok = "covers" if jp.W + 1e-12 >= sizing["W_req"] else "is BELOW"
        print(f"  supercap: window W = {jp.W:.4g} J [{jp.B_min:.4g}, {jp.B_max:.4g}] {ok} W_req = {sizing['W_req']:.4g} J "
              f"(largest event slot {sizing['on_peak']:.4g} J + longest sleep-run drain {sizing['drain_max']:.4g} J)")
    print()
    print(f"  {'release':>7s} {'run':>4s} {'delay':>5s} {'outcome':>11s} {'x [mJ]':>8s} {'P_pk [mW]':>9s} {'rad [mW]':>8s} "
          f"{'SNR [dB]':>8s} {'BER':>9s} {'pass':>4s} {'nats':>7s}")
    for x in rows:
        run = "-" if x["run"] is None else str(x["run"])
        dly = "-" if x["delay"] is None else str(x["delay"])
        if x["outcome"] in ("run", "rescheduled"):
            ok = "yes" if x["ber_pass"] else "NO"
            print(f"  {x['release']:>7d} {run:>4s} {dly:>5s} {x['outcome']:>11s} {x['x_J'] * 1e3:>8.4f} {x['P_pk_W'] * 1e3:>9.2f} "
                  f"{x['radiated_W'] * 1e3:>8.3f} {x['snr_dB']:>8.2f} {x['ber']:>9.2e} {ok:>4s} {x['nats']:>7.4f}")
        else:
            print(f"  {x['release']:>7d} {run:>4s} {dly:>5s} {x['outcome']:>11s}")
    print()
    return out


def diagnose_no_data(jp, r, stop: list | None = None, profile: dict | None = None, verbose: bool = True) -> dict:
    """Compact report when the master has no data link (theta1 = 0): diagnose()'s sharing test divides by
    theta1, so this replaces it.  tau*P is the master's UNUSED / SPILL energy; master throughput and master nats
    are not reported.  stop = per-slot stopping bound of the maximal-sharing post-pass (solver.maximal_sharing)."""
    m, tau, a = jp.master, jp.master.tau, jp.a
    sent, etot = float(r.delta.sum()), float(m.E.sum())
    out = dict(harvested=etot, sent=sent, unused_spill=float(tau * r.P.sum()), stored=float((a * r.delta).sum()),
               slave_overhead=float(jp.overhead.sum()), slave_bursts=float(tau * r.Pbar.sum()),
               final_level_above_floor=float(r.b[-1] - jp.B_min), rate_slave=r.rate_slave, objective=r.objective,
               shared_fraction=sent / etot if etot else 0.0, master_kept=float(r.F[-1]))
    if stop is not None:
        out["stop_counts"] = {s: stop.count(s) for s in sorted(set(stop))}
    if not verbose:
        return out
    line = "=" * 92
    print(line)
    print("MASTER (no data link, theta1 = 0) -> SLAVE ENERGY SHARING")
    print(line)
    print(f"  tap a = beta*alpha: {_rng(a)}  (alpha {_rng(jp.alpha)}, beta {_rng(jp.beta)})")
    print(f"  slave supercap: B_min {jp.B_min:.4g} J, B_max {jp.B_max:.4g} J (window {jp.W:.4g} J), b0 {jp.b0:.4g} J")
    if np.isfinite(m.E_max) or jp.master_floor_J is not None:
        fl = "off" if jp.master_floor_J is None else f"{jp.master_floor_J:.6g} J"
        print(f"  master battery: capacity E_max {m.E_max:.6g} J, reserve floor F_min {fl}, "
              f"level in [{float(r.F.min()):.6g}, {float(r.F.max()):.6g}] J")
        print(f"  harvested {etot:.6g} J = sent {sent:.6g} J ({100 * out['shared_fraction']:.4f}%) + kept by master "
              f"{out['master_kept']:.6g} J ({100 * out['master_kept'] / etot if etot else 0.0:.4f}%) + unused / spill "
              f"{out['unused_spill']:.6g} J")
    else:
        print(f"  harvested {etot:.6g} J = sent {sent:.6g} J ({100 * out['shared_fraction']:.4f}%) + unused / spill "
              f"{out['unused_spill']:.6g} J (+ master leftover {float(r.F[-1]):.2g})")
    print(f"  stored {out['stored']:.6g} J + initial excess {jp.g_init:.4g} J = slave electronics {out['slave_overhead']:.6g} J "
          f"+ THz bursts {out['slave_bursts']:.6g} J + final level above floor {out['final_level_above_floor']:.6g} J")
    if stop is not None:
        expl = {"ceiling": "supercap reaches B_max on arrival", "S5": "MPPT input rating", "M5": "laser cap",
                "master": "master's own energy", "floor": "master's reserve floor",
                "full": "arrived full: delta = 0"}
        print("  master stops sharing (per slot): " + ", ".join(f"{k} {v} ({expl.get(k, k)})" for k, v in out["stop_counts"].items()))
    if profile is not None:
        ks = profile["overhead_keys"]
        print("  slave electronics (non-TX) per component [J]: " + ", ".join(f"{k} {float(np.sum(profile[k])):.4g}" for k in ks))
    print(f"  objective {r.objective:.6g} nats = theta2 x slave burst rate {r.rate_slave:.6g} nats | iters {r.iters} | "
          f"converged {r.converged} | start {r.start}")
    print()
    return out
