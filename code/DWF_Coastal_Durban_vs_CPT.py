"""
THERMAL — COASTAL CAPE TOWN vs DURBAN
========================================

Coastal deployment: cold-water intake at ~200 m depth (shorter pipe,
cheaper than deep-sea OTEC at 1000 m).

Characterised by:
    • Smaller ΔT than deep sea (warmer cold side)
    • Higher biofouling (shallower, more life)
    • More human activity (fishing, shipping, theft)
    • Land runoff / sediment
    • Shipping noise / vibration

Compares Cape Town vs Durban coastal thermal harvesting,
picks the best technology at each, runs DWF, recommends a site.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


# =============================================================================
# SITE CONFIGURATION — COASTAL
# =============================================================================

SEASONS = ['Summer', 'Autumn', 'Winter', 'Spring']

SITES = {
    'Cape Town Coastal': {
        # Benguela current + Southern Ocean influence
        'surface_temps': {
            'Summer': 20.0,
            'Autumn': 17.5,
            'Winter': 15.5,
            'Spring': 17.0,
        },
        # Cold side: coastal shelf at ~200 m depth
        'cold_temp': 6.0,
        # Coastal noise profile
        'noise': {
            'cold_fronts':   0.08,
            'clouds':        0.04,
            'theft_prob':    0.003,   # higher: coastal human activity
            'biofouling':    0.07,    # higher: shallower, nutrient-rich
            'thermal_drift': 0.02,
            'upwelling':     0.06,    # Benguela upwelling
            'shipping_noise':0.03,    # port traffic
            'runoff':        0.03,    # river runoff
        },
    },
    'Durban Coastal': {
        # Agulhas current, subtropical
        'surface_temps': {
            'Summer': 26.5,
            'Autumn': 24.5,
            'Winter': 21.5,
            'Spring': 23.5,
        },
        # Cold side: coastal shelf at ~200 m depth
        # Warmer than Cape Town coastal because Agulhas pushes warm water south
        'cold_temp': 8.0,
        'noise': {
            'cold_fronts':   0.05,
            'clouds':        0.03,
            'theft_prob':    0.004,   # higher: busier port
            'biofouling':    0.08,    # higher: tropical, more life
            'thermal_drift': 0.02,
            'upwelling':     0.03,
            'shipping_noise':0.04,    # Durban is Africa's busiest port
            'runoff':        0.04,
        },
    },
}


# =============================================================================
# TECHNOLOGIES
# =============================================================================

TECHNOLOGIES = {
    'teg_submersible':  {'name': 'TEG Submersible',  'ref_delta_T': 21.0,  'rated_power': 0.35},
    'teg_otec':         {'name': 'TEG OTEC',         'ref_delta_T': 21.0,  'rated_power': 3.01},
    'teg_submarine':    {'name': 'TEG Submarine',    'ref_delta_T': 15.0,  'rated_power': 0.35},
    'teg_hydrothermal': {'name': 'TEG Hydrothermal', 'ref_delta_T': 246.0, 'rated_power': 3.25},
    'micro_otec':       {'name': 'Micro-OTEC',       'ref_delta_T': 21.0,  'rated_power': 714.6},
    'rtg':              {'name': 'RTG',              'ref_delta_T': 21.0,  'rated_power': 0.00233},
}


# =============================================================================
# TEMPERATURE MODEL
# =============================================================================

def surface_temp_at_minute(hour, minute, season, site):
    site_data = SITES[site]
    t = hour + minute / 60.0
    np.random.seed(42 + int(t * 60) + hash(season) % 1000 + hash(site) % 100)

    base = site_data['surface_temps'][season]
    diurnal = 1.5 * np.sin(2 * np.pi * (t - 9) / 24)
    noise = np.random.normal(0, 0.05)
    return base + diurnal + noise


def delta_T(surface_temp, site):
    return max(0.0, surface_temp - SITES[site]['cold_temp'])


# =============================================================================
# COASTAL NOISE MODEL
# =============================================================================

def noise_factor(hour, season, site):
    """Multiplicative loss factor for coastal deployment."""
    n = SITES[site]['noise']
    np.random.seed(99 + hour + hash(season) % 100 + hash(site) % 10)

    cf_mult = 1.8 if season == 'Winter' else 1.2 if season == 'Autumn' else 0.4
    cold_fronts     = n['cold_fronts'] * cf_mult
    clouds          = n['clouds']
    biofouling      = n['biofouling']
    drift           = n['thermal_drift']
    upwelling       = n['upwelling'] if np.random.random() < 0.08 else 0.0
    shipping_noise  = n['shipping_noise']
    runoff          = n['runoff'] if np.random.random() < 0.10 else 0.0

    total = min(1.0, cold_fronts + clouds + biofouling + drift +
                     upwelling + shipping_noise + runoff)
    return 1.0 - total


def theft_incident(site):
    np.random.seed(777 + hash(site) % 100)
    return np.random.random() < SITES[site]['noise']['theft_prob']


# =============================================================================
# POWER CURVE
# =============================================================================

def power_output(dT, tech_key):
    p = TECHNOLOGIES[tech_key]
    if dT <= 0:
        return 0.0
    return min(p['rated_power'] * (dT / p['ref_delta_T']), p['rated_power'])


def power_with_noise(dT, tech_key, hour, season, site):
    return power_output(dT, tech_key) * noise_factor(hour, season, site)


# =============================================================================
# HOURLY HARVEST
# =============================================================================

def harvest_hourly(site, season, tech_key):
    hourly = []
    for hour in range(24):
        total_kj = 0.0
        peak_w = 0.0
        for minute in range(60):
            surf = surface_temp_at_minute(hour, minute, season, site)
            dT = delta_T(surf, site)
            p = power_with_noise(dT, tech_key, hour, season, site)
            total_kj += p * 60 / 1000
            if p > peak_w:
                peak_w = p
        hourly.append({
            'hour': hour,
            'total_energy_kj': total_kj,
            'peak_power_w': peak_w,
        })
    return hourly


# =============================================================================
# DWF ALGORITHM
# =============================================================================

def dwf(E, L):
    E = np.asarray(E, dtype=float)
    L = np.asarray(L, dtype=float)
    P = np.where(L > 0, E / np.where(L == 0, 1, L), 0.0)
    changed = True
    while changed:
        changed = False
        for i in range(len(P) - 1):
            if P[i] > P[i + 1]:
                s, e = i, i + 1
                while s > 0 and P[s - 1] >= P[s]:
                    s -= 1
                while e < len(P) - 1 and P[e] >= P[e + 1]:
                    e += 1
                avg = np.sum(E[s:e + 1]) / np.sum(L[s:e + 1])
                P[s:e + 1] = avg
                changed = True
                break
    return P


# =============================================================================
# STEP 1 — TECHNOLOGY EVALUATION PER SITE
# =============================================================================

def evaluate_technologies():
    print("=" * 110)
    print("STEP 1: COASTAL THERMAL TECHNOLOGY EVALUATION")
    print("=" * 110)

    print(f"\n{'Site':<20} | {'Surface (Sum/Win)':<20} | {'Cold side':<12} | "
          f"{'ΔT (Sum/Win)':<16}")
    print("-" * 110)
    for site, data in SITES.items():
        s_sum = data['surface_temps']['Summer']
        s_win = data['surface_temps']['Winter']
        cold = data['cold_temp']
        print(f"{site:<20} | {s_sum:.1f} / {s_win:.1f} °C".ljust(22) + "| "
              f"{cold:.1f} °C".ljust(14) + "| "
              f"{(s_sum-cold):.1f} / {(s_win-cold):.1f} °C")
    print("-" * 110)

    results = {}
    for site in SITES:
        results[site] = {}
        print(f"\n{'─' * 110}")
        print(f"SITE: {site.upper()}")
        print(f"{'─' * 110}")
        print(f"{'Technology':<18} | {'Summer':<11} | {'Autumn':<11} | "
              f"{'Winter':<11} | {'Spring':<11} | {'Annual':<11} | {'Worst':<11}")
        print("-" * 110)

        for tech_key, params in TECHNOLOGIES.items():
            season_totals = {}
            for s in SEASONS:
                h = harvest_hourly(site, s, tech_key)
                season_totals[s] = float(np.sum([r['total_energy_kj'] for r in h]))

            annual = sum(season_totals.values())
            worst = min(season_totals.values())

            results[site][tech_key] = {
                'season_totals': season_totals,
                'annual': annual,
                'worst': worst,
            }

            print(f"{params['name']:<18} | "
                  f"{season_totals['Summer']:>9.1f} | "
                  f"{season_totals['Autumn']:>9.1f} | "
                  f"{season_totals['Winter']:>9.1f} | "
                  f"{season_totals['Spring']:>9.1f} | "
                  f"{annual:>9.1f} | {worst:>9.1f}")

    return results


def pick_best_per_site(scores):
    print("\n" + "=" * 110)
    print("BEST TECHNOLOGY PER SITE (excluding RTG)")
    print("=" * 110)

    best = {}
    for site in SITES:
        candidates = {k: v for k, v in scores[site].items() if k != 'rtg'}
        best_key = max(candidates,
                       key=lambda k: candidates[k]['season_totals']['Winter'])
        best[site] = best_key
        print(f"  {site:<22}: {TECHNOLOGIES[best_key]['name']:<15} "
              f"(Winter = {candidates[best_key]['season_totals']['Winter']:.1f} kJ, "
              f"Annual = {candidates[best_key]['annual']:.1f} kJ)")
    return best


# =============================================================================
# STEP 2 — DWF AT EACH SITE FOR BEST TECH
# =============================================================================

def run_dwf_all_seasons(site, tech_key):
    print(f"\n{'=' * 110}")
    print(f"DWF — {site} — {TECHNOLOGIES[tech_key]['name']}")
    print("=" * 110)

    out = {}
    for season in SEASONS:
        hourly = harvest_hourly(site, season, tech_key)
        E = np.array([h['total_energy_kj'] for h in hourly])
        L = np.ones(24)
        P = dwf(E, L)

        B = np.zeros(24)
        b = 0.0
        for i in range(24):
            b = b + E[i] - P[i]
            B[i] = b

        out[season] = {
            'E': E,
            'P_dwf': P,
            'battery': B,
            'total_kj': float(np.sum(E)),
            'P_const_w': float(P[0] / 3.6),
        }

        print(f"  {season:<10} : harvest = {np.sum(E):>9.1f} kJ/day | "
              f"DWF = {P[0]/3.6:>7.3f} W | "
              f"THz bursts = {np.sum(E)/0.0117:>13,.0f}")

    return out


# =============================================================================
# STEP 3 — SIDE-BY-SIDE COMPARISON
# =============================================================================

def compare_sites(dwf_results, best_techs):
    print("\n" + "=" * 110)
    print("SIDE-BY-SIDE COMPARISON — COASTAL CAPE TOWN vs DURBAN")
    print("=" * 110)
    print(f"{'Season':<10} | {'Cape Town (kJ)':<18} | {'Durban (kJ)':<18} | "
          f"{'Winner':<10} | {'Ratio (D/CT)':<14}")
    print("-" * 110)

    for s in SEASONS:
        ct = dwf_results['Cape Town Coastal'][s]['total_kj']
        db = dwf_results['Durban Coastal'][s]['total_kj']
        winner = 'Durban' if db > ct else 'Cape Town'
        ratio = db / ct if ct > 0 else 0.0
        print(f"{s:<10} | {ct:>16.1f} | {db:>16.1f} | {winner:<10} | {ratio:>12.2f}")

    ct_annual = sum(dwf_results['Cape Town Coastal'][s]['total_kj'] for s in SEASONS)
    db_annual = sum(dwf_results['Durban Coastal'][s]['total_kj'] for s in SEASONS)
    print("-" * 110)
    print(f"{'ANNUAL':<10} | {ct_annual:>16.1f} | {db_annual:>16.1f} | "
          f"{'Durban':<10} | {db_annual/ct_annual:>12.2f}")

    print("\n" + "=" * 110)
    print("DWF CONSTANT POWER COMPARISON")
    print("=" * 110)
    print(f"{'Season':<10} | {'Cape Town (W)':<18} | {'Durban (W)':<18} | "
          f"{'Winner':<10}")
    print("-" * 80)
    for s in SEASONS:
        ct_w = dwf_results['Cape Town Coastal'][s]['P_const_w']
        db_w = dwf_results['Durban Coastal'][s]['P_const_w']
        winner = 'Durban' if db_w > ct_w else 'Cape Town'
        print(f"{s:<10} | {ct_w:>16.3f} | {db_w:>16.3f} | {winner:<10}")

    print("\n" + "=" * 110)
    print("RECOMMENDATION")
    print("=" * 110)
    if db_annual > ct_annual:
        print(f"Durban Coastal is the better thermal site.")
        print(f"  → Warm Agulhas current → higher surface temp")
        print(f"  → Coastal cold side at 8°C → ΔT of 13.5–18.5 °C")
        print(f"  → Annual thermal energy: {db_annual/1000:.1f} MJ "
              f"vs {ct_annual/1000:.1f} MJ at Cape Town")
        print(f"  → Durban produces {db_annual/ct_annual:.2f}× "
              f"more than Cape Town")
    else:
        print("Cape Town Coastal is the better site.")


# =============================================================================
# STEP 4 — COASTAL vs DEEP SEA CONTEXT
# =============================================================================

def coastal_vs_deepsea_context():
    print("\n" + "=" * 110)
    print("CONTEXT — COASTAL vs DEEP-SEA THERMAL HARVESTING")
    print("=" * 110)
    print(f"{'Aspect':<22} | {'Coastal (200 m)':<28} | {'Deep Sea (1000 m)':<28}")
    print("-" * 100)
    print(f"{'Cold-side temperature':<22} | {'6–8 °C':<28} | {'4 °C':<28}")
    print(f"{'Typical ΔT (Summer)':<22} | {'14–18.5 °C':<28} | {'16.5–22.5 °C':<28}")
    print(f"{'Typical ΔT (Winter)':<22} | {'9.5–13.5 °C':<28} | {'11.5–17.5 °C':<28}")
    print(f"{'Pipe length':<22} | {'200 m (shorter, cheaper)':<28} | {'1000 m (long, costly)':<28}")
    print(f"{'Biofouling':<22} | {'HIGH (nutrients)':<28} | {'MODERATE':<28}")
    print(f"{'Human interference':<22} | {'HIGH (port, fishing)':<28} | {'LOW':<28}")
    print(f"{'Maintenance cost':<22} | {'HIGHER':<28} | {'LOWER':<28}")
    print(f"{'Typical power':<22} | {'COASTAL = ~60–80% of deep':<28} | {'DEEP = baseline':<28}")
    print("=" * 110)


# =============================================================================
# STEP 5 — PLOTS
# =============================================================================

def plot_comparison(dwf_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Thermal Harvesting — Coastal Cape Town vs Durban',
                 fontsize=15, fontweight='bold')
    colors = {'Summer': '#f1c40f', 'Autumn': '#e67e22',
              'Winter': '#3498db', 'Spring': '#2ecc71'}
    hours = np.arange(24)

    # Plot 1 — Hourly energy (Cape Town coastal)
    ax = axes[0, 0]
    for s in SEASONS:
        ax.plot(hours, dwf_results['Cape Town Coastal'][s]['E'], 'o-',
                label=s, color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Energy (kJ)')
    ax.set_title('Cape Town Coastal — Hourly Harvest')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 2 — Hourly energy (Durban coastal)
    ax = axes[0, 1]
    for s in SEASONS:
        ax.plot(hours, dwf_results['Durban Coastal'][s]['E'], 'o-',
                label=s, color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Energy (kJ)')
    ax.set_title('Durban Coastal — Hourly Harvest')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 3 — DWF power (both sites)
    ax = axes[1, 0]
    x = np.arange(len(SEASONS))
    width = 0.35
    ct = [dwf_results['Cape Town Coastal'][s]['P_const_w'] for s in SEASONS]
    db = [dwf_results['Durban Coastal'][s]['P_const_w'] for s in SEASONS]
    ax.bar(x - width/2, ct, width, label='Cape Town', color='#e74c3c')
    ax.bar(x + width/2, db, width, label='Durban', color='#3498db')
    ax.set_xticks(x); ax.set_xticklabels(SEASONS)
    ax.set_ylabel('DWF Constant Power (W)')
    ax.set_title('DWF Power — Coastal Comparison')
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 4 — Daily energy (both sites)
    ax = axes[1, 1]
    ct = [dwf_results['Cape Town Coastal'][s]['total_kj'] for s in SEASONS]
    db = [dwf_results['Durban Coastal'][s]['total_kj'] for s in SEASONS]
    ax.bar(x - width/2, ct, width, label='Cape Town', color='#e74c3c')
    ax.bar(x + width/2, db, width, label='Durban', color='#3498db')
    ax.set_xticks(x); ax.set_xticklabels(SEASONS)
    ax.set_ylabel('Daily Energy (kJ)')
    ax.set_title('Daily Thermal Energy — Coastal Comparison')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_csv(dwf_results):
    with open('thermal_coastal_ct_vs_durban.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Site', 'Season', 'Hour', 'Energy_kJ',
                    'DWF_Power_W', 'Battery_kJ'])
        for site in SITES:
            for s in SEASONS:
                r = dwf_results[site][s]
                for h in range(24):
                    w.writerow([site, s, h,
                                f"{r['E'][h]:.4f}",
                                f"{r['P_dwf'][h]/3.6:.4f}",
                                f"{r['battery'][h]:.4f}"])

    with open('thermal_coastal_ct_vs_durban_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Site', 'Season', 'Total_kJ', 'DWF_Power_W',
                    'THz_Bursts', 'Peak_Battery_kJ'])
        for site in SITES:
            for s in SEASONS:
                r = dwf_results[site][s]
                w.writerow([site, s,
                            f"{r['total_kj']:.2f}",
                            f"{r['P_const_w']:.4f}",
                            f"{r['total_kj']/0.0117:.0f}",
                            f"{np.max(r['battery']):.3f}"])

    print("\n✅ Saved: thermal_coastal_ct_vs_durban.csv")
    print("✅ Saved: thermal_coastal_ct_vs_durban_summary.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("THERMAL — COASTAL CAPE TOWN vs DURBAN")
    print("=" * 110)

    # Noise theft check
    for site in SITES:
        theft_incident(site)

    # STEP 1 — Evaluate technologies at both sites
    scores = evaluate_technologies()

    # STEP 2 — Pick best at each site
    best_techs = pick_best_per_site(scores)

    # STEP 3 — DWF for each site, all seasons
    dwf_results = {}
    for site in SITES:
        dwf_results[site] = run_dwf_all_seasons(site, best_techs[site])

    # STEP 4 — Side-by-side
    compare_sites(dwf_results, best_techs)

    # STEP 5 — Context vs deep sea
    coastal_vs_deepsea_context()

    # STEP 6 — Plots
    plot_comparison(dwf_results)

    # STEP 7 — CSV
    export_csv(dwf_results)

    print("\n" + "=" * 110)
    print("✅ COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()