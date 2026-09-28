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
def diagnose(jp, r, tol: float = 1e-6, verbose: bool = True, meta: dict | None = None) -> dict:
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
    reason = np.empty(N, dtype=object)
    for k in range(N):
        if on[k]:
            reason[k] = "CAP" if cap[k] else ("BALANCED" if abs(gap_rel[k]) < 1e-4 else "UNBALANCED")
        else:
            reason[k] = "NOT_WORTH" if gap_rel[k] >= -1e-9 else ("CAP_BLOCKED" if cap[k] else "BLOCKED")
    counts = {k: int((reason == k).sum()) for k in ("BALANCED", "CAP", "UNBALANCED", "NOT_WORTH", "CAP_BLOCKED", "BLOCKED")}

    sent, etot = float(r.delta.sum()), float(E.sum())
    share = sent / etot if etot else 0.0
    if counts["BALANCED"] + counts["CAP"] + counts["UNBALANCED"] == 0:
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
    }
    out["active"] = {
        "master_taps_closed": int((r.F[:-1] <= sc).sum()), "master_taps": N - 1,
        "slave_at_floor": int((r.b <= jp.B_min + sc).sum()),
        "slave_at_ceiling": int((r.b_peak >= jp.B_max - sc).sum()),
        "P_zero_slots": int((r.P <= sc / tau).sum()), "Pbar_zero_slots": int((r.Pbar <= sc / tau).sum()),
        "delta_zero_slots": int((r.delta <= sc).sum()), "slots": N,
    }
    out["shared_fraction"] = share
    # backwards-compatible keys
    out["sharing_ratio"], out["m_mean"], out["a_s_mean"] = out["decision"]["R_mean"], float(mk.mean()), float((a * sk).mean())
    out["snr_master"], out["snr_slave"] = snr_m, snr_s

    if not verbose:
        return out

    # ================================== printing ==============================================
    L, A = out["ledger"], out["active"]
    line = "=" * 92
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
            "CAP_BLOCKED": "would pay but slave battery is full", "BLOCKED": "would pay but another constraint binds"}
    for k, v in counts.items():
        if v:
            print(f"    {k:<12s} {v:>3d} slots : {expl[k]}")
    print()
    print(line)
    print("4. ENERGY LEDGER  (Joules)")
    print(line)
    print("  harvested %.1f = master data %.1f + sent %.1f (+ master leftover %.2g)" % (L["harvested"], L["master_data"], L["sent"], L["master_leftover"]))
    print("  sent %.1f -> channel loss %.1f, recharge loss %.1f, stored %.1f" % (L["sent"], L["channel_loss"], L["recharge_loss"], L["stored"]))
    print("  stored %.1f = slave data %.1f + battery change %.1f" % (L["stored"], L["slave_data"], L["battery_change"]))
    print()
    print(line)
    print("5. ACTIVE CONSTRAINTS")
    print(line)
    print(f"  master taps closed (F=0): {A['master_taps_closed']}/{A['master_taps']} | slave at floor: {A['slave_at_floor']}/{A['slots']}"
          f" | slave at ceiling on arrival: {A['slave_at_ceiling']}/{A['slots']}")
    print(f"  P=0 slots: {A['P_zero_slots']} | Pbar=0 slots: {A['Pbar_zero_slots']} | delta=0 slots: {A['delta_zero_slots']}")
    print()
    print(line)
    print("6. SOLVER")
    print(line)
    print(f"  objective {r.objective:.6g} nats = th1*{r.rate_master:.4g} + th2*{r.rate_slave:.4g} | shared fraction of harvest {100 * share:.1f}%"
          f" | iters {r.iters} | converged {r.converged}")
    return out
