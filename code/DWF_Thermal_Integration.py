"""
THERMAL — SCENARIO 3 — DURBAN DEEP SEA
========================================

Location: Durban offshore (KwaZulu-Natal), deep-sea (~1000 m).
Source: Ocean thermal gradient (warm surface vs cold deep water).
Technology: Micro-OTEC (best technology from evaluation).

Structure mirrors solar / wave scenarios:
    • Hourly temperature and ΔT
    • Peak / Moderate / Low hour classification
    • Minute-level detail within each hour
    • Seasonal comparison
    • DWF allocation across all seasons

Noise sources (per design notes):
    • Cold fronts      → surface cooling, reduced ΔT
    • Clouds           → reduced surface warming
    • Theft            → rare catastrophic loss
    • Biofouling       → heat-exchanger efficiency drop
    • Thermal drift    → sensor / material drift
    • Upwelling        → cold water intrusion
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


# =============================================================================
# CONFIGURATION — SCENARIO 3 (DURBAN DEEP SEA)
# =============================================================================

SEASONS = ['Summer', 'Autumn', 'Winter', 'Spring']

# Durban sea-surface temperatures (°C)
# Warmer than Cape Town due to the Agulhas current.
SEASONAL_SURFACE_TEMPS = {
    'Summer': 26.5,
    'Autumn': 24.5,
    'Winter': 21.5,
    'Spring': 23.5,
}

# Deep-sea temperature at ~1000 m depth (°C)
DEEP_TEMP = 4.0

# Technologies evaluated
TECHNOLOGIES = {
    'teg_submersible': {
        'name': 'TEG Submersible',
        'ref_delta_T': 21.0,
        'rated_power': 0.35,
    },
    'teg_otec': {
        'name': 'TEG OTEC',
        'ref_delta_T': 21.0,
        'rated_power': 3.01,
    },
    'teg_submarine': {
        'name': 'TEG Submarine',
        'ref_delta_T': 15.0,
        'rated_power': 0.35,
    },
    'teg_hydrothermal': {
        'name': 'TEG Hydrothermal',
        'ref_delta_T': 246.0,
        'rated_power': 3.25,
    },
    'micro_otec': {
        'name': 'Micro-OTEC',
        'ref_delta_T': 21.0,
        'rated_power': 714.6,
    },
    'rtg': {
        'name': 'RTG',
        'ref_delta_T': 21.0,
        'rated_power': 0.00233,
    },
}

# Scenario-3 noise profile (Durban deep sea)
NOISE_PROFILE = {
    'cold_fronts':     0.06,    # 6% ΔT reduction during cold fronts
    'clouds':          0.03,    # 3% reduction from reduced surface warming
    'theft_prob':      0.001,   # 0.1% per day (rare)
    'biofouling':      0.04,    # 4% heat-exchanger loss
    'thermal_drift':   0.02,    # 2% sensor/material drift
    'upwelling':       0.05,    # 5% cold-water intrusion
}

# Battery carried from previous day (kJ)
INITIAL_BATTERY_KJ = 0.0


# =============================================================================
# TEMPERATURE MODEL
# =============================================================================

def surface_temp_at_minute(hour, minute, season, noise_level=0.05, seed=42):
    """Durban surface temperature at a given minute."""
    t = hour + minute / 60.0
    np.random.seed(seed + int(t * 60) + hash(season) % 1000)

    base = SEASONAL_SURFACE_TEMPS[season]

    # Diurnal variation ±1.5°C, peaks ~15:00
    diurnal = 1.5 * np.sin(2 * np.pi * (t - 9) / 24)

    # Small stochastic noise
    noise = np.random.normal(0, noise_level)

    return base + diurnal + noise


def temperature_difference(surface_temp):
    return max(0.0, surface_temp - DEEP_TEMP)


# =============================================================================
# SCENARIO-3 NOISE MODEL
# =============================================================================

def scenario3_noise(hour, season, seed=42):
    """
    Multiplicative loss factor from Scenario-3 noise sources.
    Returns a factor in [0, 1] applied to the raw TEG power.
    """
    np.random.seed(seed + 2000 + hour + hash(season) % 1000)

    # 1. Cold fronts — heavier in Winter
    cf_mult = 1.8 if season == 'Winter' else 1.2 if season == 'Autumn' else 0.4
    cold_fronts = NOISE_PROFILE['cold_fronts'] * cf_mult

    # 2. Clouds — small
    clouds = NOISE_PROFILE['clouds']

    # 3. Biofouling — constant, slowly accumulating
    biofouling = NOISE_PROFILE['biofouling']

    # 4. Thermal drift — constant
    drift = NOISE_PROFILE['thermal_drift']

    # 5. Upwelling — random, ~8% of hours
    upwelling = (NOISE_PROFILE['upwelling']
                 if np.random.random() < 0.08 else 0.0)

    total = min(1.0, cold_fronts + clouds + biofouling + drift + upwelling)
    return 1.0 - total


def theft_incident(seed=42):
    np.random.seed(seed + 7777)
    return np.random.random() < NOISE_PROFILE['theft_prob']


# =============================================================================
# POWER CURVE
# =============================================================================

def power_output(delta_T, tech_key):
    """Linear scaling with ΔT, saturating at rated power."""
    p = TECHNOLOGIES[tech_key]
    if delta_T <= 0:
        return 0.0
    return min(p['rated_power'] * (delta_T / p['ref_delta_T']),
               p['rated_power'])


def power_output_with_noise(delta_T, tech_key, hour, season):
    raw = power_output(delta_T, tech_key)
    return raw * scenario3_noise(hour, season)


# =============================================================================
# MINUTE-LEVEL HARVEST
# =============================================================================

def harvest_minute_hour(hour, season, tech_key):
    minutes_data = []
    total_kj = 0.0
    peak_w = 0.0
    peak_min = 0

    for minute in range(60):
        surf = surface_temp_at_minute(hour, minute, season)
        dT = temperature_difference(surf)
        power_w = power_output_with_noise(dT, tech_key, hour, season)
        energy_kj = power_w * 60 / 1000

        minutes_data.append({
            'minute': minute,
            'time': f"{hour:02d}:{minute:02d}",
            'surface_temp': surf,
            'delta_T': dT,
            'power_watts': power_w,
            'energy_kj': energy_kj,
        })

        total_kj += energy_kj
        if power_w > peak_w:
            peak_w = power_w
            peak_min = minute

    return {
        'hour': hour,
        'season': season,
        'minutes_data': minutes_data,
        'total_energy_kj': total_kj,
        'peak_power_w': peak_w,
        'peak_minute': peak_min,
    }


def harvest_hourly(season, tech_key):
    return [harvest_minute_hour(h, season, tech_key) for h in range(24)]


# =============================================================================
# PEAK / MODERATE / LOW CLASSIFICATION
# =============================================================================

def classify_hours(hourly_results):
    energies = [r['total_energy_kj'] for r in hourly_results]
    return {
        'Peak':     int(np.argmax(energies)),
        'Moderate': int(np.argsort(energies)[::-1][len(energies) // 2]),
        'Low':      int(np.argmin(energies)),
    }


# =============================================================================
# DWF ALGORITHM
# =============================================================================

def directional_water_filling(E, L):
    E = np.asarray(E, dtype=float)
    L = np.asarray(L, dtype=float)
    P = np.where(L > 0, E / np.where(L == 0, 1, L), 0.0)
    N = len(P)

    changed = True
    while changed:
        changed = False
        for i in range(N - 1):
            if P[i] > P[i + 1]:
                start, end = i, i + 1
                while start > 0 and P[start - 1] >= P[start]:
                    start -= 1
                while end < N - 1 and P[end] >= P[end + 1]:
                    end += 1
                avg = np.sum(E[start:end + 1]) / np.sum(L[start:end + 1])
                P[start:end + 1] = avg
                changed = True
                break
    return P


def battery_profile(P, E, L, B_init=0.0):
    B = np.zeros(len(E))
    b = B_init
    for i in range(len(E)):
        b = b + E[i] - P[i] * L[i]
        B[i] = b
    return B


# =============================================================================
# STEP 1 — TECHNOLOGY EVALUATION
# =============================================================================

def evaluate_technologies():
    print("=" * 110)
    print("STEP 1: THERMAL TECHNOLOGY EVALUATION — SCENARIO 3 (DURBAN DEEP SEA)")
    print("=" * 110)
    print(f"Surface temps : " +
          " | ".join(f"{s}={SEASONAL_SURFACE_TEMPS[s]}°C" for s in SEASONS))
    print(f"Deep temp     : {DEEP_TEMP}°C")
    print()
    print(f"{'Technology':<20} | {'Ref ΔT':<8} | {'Rated (W)':<10} | "
          f"{'Summer kJ':<12} | {'Winter kJ':<12} | {'Annual kJ':<12}")
    print("-" * 110)

    scores = {}
    for tech_key, params in TECHNOLOGIES.items():
        season_totals = {}
        for s in SEASONS:
            E = np.array([h['total_energy_kj']
                          for h in harvest_hourly(s, tech_key)])
            season_totals[s] = float(np.sum(E))

        annual = sum(season_totals.values())
        scores[tech_key] = {
            'season_totals': season_totals,
            'annual': annual,
            'worst': min(season_totals.values()),
        }

        print(f"{params['name']:<20} | {params['ref_delta_T']:>6.1f} | "
              f"{params['rated_power']:>8.4f} | "
              f"{season_totals['Summer']:>10.2f} | "
              f"{season_totals['Winter']:>10.2f} | "
              f"{annual:>10.2f}")

    print("-" * 110)
    return scores


def pick_best_technology(scores):
    """
    Best thermal technology = highest Winter energy.
    Also excludes RTG (radioisotope — not renewable).
    """
    candidates = {k: v for k, v in scores.items() if k != 'rtg'}
    best = max(candidates, key=lambda k: candidates[k]['season_totals']['Winter'])

    print("\n" + "=" * 110)
    print("BEST THERMAL TECHNOLOGY SELECTED")
    print("=" * 110)
    print(f"  Name        : {TECHNOLOGIES[best]['name']}")
    print(f"  Reason      : highest Winter energy (worst-case season)")
    print(f"  Ref ΔT      : {TECHNOLOGIES[best]['ref_delta_T']}°C")
    print(f"  Rated power : {TECHNOLOGIES[best]['rated_power']} W")
    print(f"  Winter (kJ) : {scores[best]['season_totals']['Winter']:.2f}")
    print(f"  Annual (kJ) : {scores[best]['annual']:.2f}")
    print("=" * 110)

    return best


# =============================================================================
# SECTION 1 — TEMPERATURE AND ΔT (HOURLY)
# =============================================================================

def section_1_temperatures():
    print("\n" + "=" * 110)
    print("SECTION 1: DURBAN SURFACE TEMPERATURE AND ΔT (HOURLY)")
    print("=" * 110)
    print(f"Deep sea temperature: {DEEP_TEMP}°C")

    temp_data = {}
    delta_data = {}
    for season in SEASONS:
        temps = []
        deltas = []
        for hour in range(24):
            mins = [surface_temp_at_minute(hour, m, season) for m in range(60)]
            t_avg = float(np.mean(mins))
            temps.append(t_avg)
            deltas.append(temperature_difference(t_avg))
        temp_data[season] = temps
        delta_data[season] = deltas

    # Surface temperature table
    print(f"\nSurface Temperature (°C):")
    print(f"{'Hour':<8} | " + " | ".join(f"{s:<14}" for s in SEASONS))
    print("-" * 110)
    for hour in range(24):
        row = f"{hour:>2}:00    | "
        for s in SEASONS:
            row += f"{temp_data[s][hour]:>12.3f} | "
        print(row)

    # ΔT table
    print(f"\nTemperature Difference ΔT (°C):")
    print(f"{'Hour':<8} | " + " | ".join(f"{s:<14}" for s in SEASONS))
    print("-" * 110)
    for hour in range(24):
        row = f"{hour:>2}:00    | "
        for s in SEASONS:
            row += f"{delta_data[s][hour]:>12.3f} | "
        print(row)

    return temp_data, delta_data


# =============================================================================
# SECTION 2 — MINUTE-LEVEL POWER (PEAK / MOD / LOW)
# =============================================================================

def section_2_minute_detail(hourly_all, tech_key):
    tech_name = TECHNOLOGIES[tech_key]['name']
    print("\n" + "=" * 110)
    print(f"SECTION 2: {tech_name.upper()} MINUTE-LEVEL POWER — PEAK / MOD / LOW")
    print("=" * 110)

    for season in SEASONS:
        hours = classify_hours(hourly_all[season])
        print(f"\n{'─' * 110}")
        print(f"SEASON: {season}")
        print(f"  Peak     : {hours['Peak']:02d}:00")
        print(f"  Moderate : {hours['Moderate']:02d}:00")
        print(f"  Low      : {hours['Low']:02d}:00")
        print(f"{'─' * 110}")
        print(f"{'Minute':<8} | {'Peak (W)':<12} | {'Moderate (W)':<14} | {'Low (W)':<12}")
        print("-" * 60)
        for minute in range(0, 60, 10):
            p = hourly_all[season][hours['Peak']]['minutes_data'][minute]['power_watts']
            m = hourly_all[season][hours['Moderate']]['minutes_data'][minute]['power_watts']
            l = hourly_all[season][hours['Low']]['minutes_data'][minute]['power_watts']
            print(f"min {minute:02d}   | {p:>10.2f} | {m:>12.2f} | {l:>10.2f}")


# =============================================================================
# SECTION 3 — HOURLY ENERGY SUMMARY
# =============================================================================

def section_3_hourly_summary(hourly_all):
    print("\n" + "=" * 110)
    print("SECTION 3: HOURLY ENERGY SUMMARY (kJ)")
    print("=" * 110)
    print(f"{'Hour':<8} | " + " | ".join(f"{s:<12}" for s in SEASONS))
    print("-" * 100)

    for hour in range(24):
        row = f"{hour:>2}:00    | "
        for s in SEASONS:
            row += f"{hourly_all[s][hour]['total_energy_kj']:>10.2f} | "
        print(row)

    print("-" * 100)
    row = f"{'TOTAL':<8} | "
    for s in SEASONS:
        total = sum(h['total_energy_kj'] for h in hourly_all[s])
        row += f"{total:>10.1f} | "
    print(row)


# =============================================================================
# SECTION 4 — DWF FOR ALL SEASONS
# =============================================================================

def section_4_dwf_all_seasons(hourly_all):
    print("\n" + "=" * 110)
    print("SECTION 4: DWF ALLOCATION — ALL SEASONS")
    print("=" * 110)

    all_dwf = {}
    for season in SEASONS:
        E = np.array([h['total_energy_kj'] for h in hourly_all[season]])
        L = np.ones(24)
        P_init = E / L
        P_dwf = directional_water_filling(E, L)
        B = battery_profile(P_dwf, E, L, B_init=INITIAL_BATTERY_KJ)

        all_dwf[season] = {
            'E': E, 'P_init': P_init, 'P_dwf': P_dwf, 'battery': B,
        }

        print(f"\n{'─' * 110}")
        print(f"SEASON: {season}")
        print(f"{'Hour':<6} | {'E[i] (kJ)':<12} | {'P_init (W)':<12} | "
              f"{'P_dwf (W)':<12} | {'Battery (kJ)':<14}")
        print("-" * 70)
        for h in range(24):
            print(f"{h:>2}:00  | {E[h]:>10.3f} | {P_init[h]/3.6:>10.4f} | "
                  f"{P_dwf[h]/3.6:>10.4f} | {B[h]:>12.3f}")
        print("-" * 70)
        print(f"  Total harvested   : {np.sum(E):.3f} kJ")
        print(f"  Constant DWF power: {P_dwf[0]/3.6:.4f} W")
        print(f"  Peak battery      : {np.max(B):.3f} kJ")
        print(f"  THz bursts/day    : {np.sum(P_dwf * L) / 0.0117:,.0f}")

    return all_dwf


# =============================================================================
# SECTION 5 — ALL-SEASON SUMMARY
# =============================================================================

def section_5_summary(hourly_all, all_dwf, tech_key):
    print("\n" + "=" * 110)
    print(f"SECTION 5: {TECHNOLOGIES[tech_key]['name'].upper()} — ALL-SEASON SUMMARY")
    print("=" * 110)
    print(f"{'Season':<10} | {'Total (kJ)':<12} | {'Total (kWh)':<12} | "
          f"{'DWF Power (W)':<14} | {'Peak Bat (kJ)':<14} | {'THz bursts':<14}")
    print("-" * 110)

    for s in SEASONS:
        E = all_dwf[s]['E']
        total = float(np.sum(E))
        p_const = all_dwf[s]['P_dwf'][0] / 3.6
        peak_b = float(np.max(all_dwf[s]['battery']))
        bursts = total / 0.0117

        print(f"{s:<10} | {total:>10.2f} | {total/3600:>10.4f} | "
              f"{p_const:>12.4f} | {peak_b:>12.3f} | {bursts:>12,.0f}")


# =============================================================================
# SECTION 6 — NOISE IMPACT
# =============================================================================

def section_6_noise():
    print("\n" + "=" * 110)
    print("SECTION 6: SCENARIO-3 NOISE IMPACT ANALYSIS (DURBAN DEEP SEA)")
    print("=" * 110)
    print(f"{'Noise Source':<20} | {'Typical Loss':<14} | {'Notes':<60}")
    print("-" * 110)
    print(f"{'Cold fronts':<20} | "
          f"{NOISE_PROFILE['cold_fronts']*100:>10.1f}%   | "
          f"Heavier in Winter (×1.8)")
    print(f"{'Clouds':<20} | "
          f"{NOISE_PROFILE['clouds']*100:>10.1f}%   | "
          f"Reduced surface warming")
    print(f"{'Theft':<20} | "
          f"{NOISE_PROFILE['theft_prob']*100:>10.2f}%   | "
          f"Rare, catastrophic")
    print(f"{'Biofouling':<20} | "
          f"{NOISE_PROFILE['biofouling']*100:>10.1f}%   | "
          f"Heat-exchanger efficiency drop")
    print(f"{'Thermal drift':<20} | "
          f"{NOISE_PROFILE['thermal_drift']*100:>10.1f}%   | "
          f"Sensor / material drift")
    print(f"{'Upwelling':<20} | "
          f"{NOISE_PROFILE['upwelling']*100:>10.1f}%   | "
          f"Random, ~8% of hours")

    if theft_incident():
        print("\n⚠  THEFT EVENT TODAY — total loss of harvested energy.")
    else:
        print("\n✅ No theft event today.")


# =============================================================================
# SECTION 7 — PLOTS
# =============================================================================

def section_7_plots(temp_data, delta_data, hourly_all, all_dwf, tech_key):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Scenario 3 — Durban Deep Sea — {TECHNOLOGIES[tech_key]["name"]}',
                 fontsize=15, fontweight='bold')
    colors = {'Summer': '#f1c40f', 'Autumn': '#e67e22',
              'Winter': '#3498db', 'Spring': '#2ecc71'}
    hours = np.arange(24)

    # Plot 1 — ΔT
    ax = axes[0, 0]
    for s in SEASONS:
        ax.plot(hours, delta_data[s], 'o-', label=s,
                color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('ΔT (°C)')
    ax.set_title('Temperature Difference — All Seasons')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 2 — Hourly energy
    ax = axes[0, 1]
    for s in SEASONS:
        E = [h['total_energy_kj'] for h in hourly_all[s]]
        ax.plot(hours, E, 'o-', label=s, color=colors[s],
                linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Energy (kJ)')
    ax.set_title(f'{TECHNOLOGIES[tech_key]["name"]} — Hourly Energy')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 3 — DWF allocation
    ax = axes[1, 0]
    for s in SEASONS:
        ax.plot(hours, all_dwf[s]['P_dwf'] / 3.6, 'o-',
                label=s, color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('DWF Power (W)')
    ax.set_title('DWF Allocation — All Seasons')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 4 — Battery
    ax = axes[1, 1]
    for s in SEASONS:
        ax.plot(hours, all_dwf[s]['battery'], 'o-',
                label=s, color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Battery (kJ)')
    ax.set_title('Battery Level — All Seasons')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.show()


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_csv(hourly_all, all_dwf, tech_key):
    with open('durban_thermal_hourly.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Season', 'Hour', 'Energy_kJ', 'Peak_W'])
        for s in SEASONS:
            for h in range(24):
                r = hourly_all[s][h]
                w.writerow([s, h, f"{r['total_energy_kj']:.4f}",
                            f"{r['peak_power_w']:.4f}"])

    with open('durban_thermal_dwf.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Season', 'Hour', 'Harvested_kJ', 'P_dwf_W', 'Battery_kJ'])
        for s in SEASONS:
            for i in range(24):
                w.writerow([s, i,
                            f"{all_dwf[s]['E'][i]:.4f}",
                            f"{all_dwf[s]['P_dwf'][i]/3.6:.4f}",
                            f"{all_dwf[s]['battery'][i]:.4f}"])

    with open('durban_thermal_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Season', 'Total_kJ', 'Total_kWh',
                    'DWF_Power_W', 'Peak_Battery_kJ', 'THz_Bursts'])
        for s in SEASONS:
            total = float(np.sum(all_dwf[s]['E']))
            w.writerow([s,
                        f"{total:.2f}",
                        f"{total/3600:.4f}",
                        f"{all_dwf[s]['P_dwf'][0]/3.6:.4f}",
                        f"{np.max(all_dwf[s]['battery']):.3f}",
                        f"{total/0.0117:.0f}"])

    print("\n✅ Saved: durban_thermal_hourly.csv")
    print("✅ Saved: durban_thermal_dwf.csv")
    print("✅ Saved: durban_thermal_summary.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("SCENARIO 3 — DURBAN DEEP SEA — THERMAL + DWF (ALL SEASONS)")
    print("=" * 110)

    # STEP 1 — Technology evaluation
    scores = evaluate_technologies()
    best_tech = pick_best_technology(scores)

    # STEP 2 — Hourly harvest for best technology
    hourly_all = {s: harvest_hourly(s, best_tech) for s in SEASONS}

    # STEP 3 — Surface temp / ΔT
    temp_data, delta_data = section_1_temperatures()

    # STEP 4 — Minute detail
    section_2_minute_detail(hourly_all, best_tech)

    # STEP 5 — Hourly summary
    section_3_hourly_summary(hourly_all)

    # STEP 6 — DWF for all seasons
    all_dwf = section_4_dwf_all_seasons(hourly_all)

    # STEP 7 — All-season summary
    section_5_summary(hourly_all, all_dwf, best_tech)

    # STEP 8 — Noise impact
    section_6_noise()

    # STEP 9 — Plots
    section_7_plots(temp_data, delta_data, hourly_all, all_dwf, best_tech)

    # STEP 10 — CSV
    export_csv(hourly_all, all_dwf, best_tech)

    print("\n" + "=" * 110)
    print("✅ COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()