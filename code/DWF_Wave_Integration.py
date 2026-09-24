"""
WAVE — SCENARIO 2 — TIGERRAY + DWF (ALL SEASONS)
==================================================

Cape Town Coastal (sea), Master on surface.
Technology: TigerRAY (best WEC).

Structure mirrors solar_for_hour.py:
    • Hourly harvesting across 24 hours
    • Peak / Moderate / Low hour classification
    • Minute-level detail within each hour
    • Seasonal comparison
    • DWF allocation for EVERY season
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


# =============================================================================
# CONFIGURATION — SCENARIO 2
# =============================================================================

SEASONS = ['Summer', 'Autumn', 'Winter', 'Spring']

# Cape Town coastal significant wave heights (m)
SEASONAL_WAVE_HEIGHTS = {
    'Summer': 1.8,
    'Autumn': 2.3,
    'Winter': 3.0,
    'Spring': 2.2,
}

# TigerRAY WEC parameters (selected best technology)
TIGERRAY = {
    'name': 'TigerRAY',
    'cut_in': 0.5,
    'rated_height': 1.8,
    'rated_power': 55.0,
    'cut_out': 4.5,
}

# Battery carried over from previous day (kJ)
INITIAL_BATTERY_KJ = 0.0


# =============================================================================
# WAVE HEIGHT MODEL — minute resolution
# =============================================================================

def wave_height_at_minute(hour, minute, season, noise_level=0.12, seed=42):
    """Wave height at a specific minute of the day."""
    t = hour + minute / 60.0
    np.random.seed(seed + int(t * 60) + hash(season) % 1000)

    base = SEASONAL_WAVE_HEIGHTS[season]
    tidal = 0.15 * np.sin(2 * np.pi * (t - 3) / 12.4)
    noise = np.random.normal(0, noise_level)

    return max(0.1, base * (1 + tidal + noise))


# =============================================================================
# TIGERRAY POWER CURVE
# =============================================================================

def tigerray_power(wave_height):
    p = TIGERRAY
    if wave_height < p['cut_in'] or wave_height > p['cut_out']:
        return 0.0
    if wave_height <= p['rated_height']:
        return p['rated_power'] * (wave_height / p['rated_height']) ** 2
    return p['rated_power']


# =============================================================================
# MINUTE-LEVEL HARVEST
# =============================================================================

def harvest_minute_hour(hour, season):
    minutes_data = []
    total_kj = 0.0
    peak_w = 0.0
    peak_min = 0

    for minute in range(60):
        wave_h = wave_height_at_minute(hour, minute, season)
        power_w = tigerray_power(wave_h)
        energy_kj = power_w * 60 / 1000

        minutes_data.append({
            'minute': minute,
            'time': f"{hour:02d}:{minute:02d}",
            'wave_height': wave_h,
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


def harvest_hourly(season):
    return [harvest_minute_hour(h, season) for h in range(24)]


# =============================================================================
# PEAK / MODERATE / LOW CLASSIFICATION
# =============================================================================

def classify_hours(hourly_results):
    energies = [r['total_energy_kj'] for r in hourly_results]
    peak_hour = int(np.argmax(energies))
    sorted_idx = np.argsort(energies)[::-1]
    moderate_hour = int(sorted_idx[len(sorted_idx) // 2])
    low_hour = int(np.argmin(energies))

    return {
        'Peak':     peak_hour,
        'Moderate': moderate_hour,
        'Low':      low_hour,
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
# SECTION 1 — HOURLY WAVE HEIGHTS
# =============================================================================

def section_1_hourly_heights():
    print("\n" + "=" * 110)
    print("SECTION 1: HOURLY WAVE HEIGHTS — CAPE TOWN COASTAL (m)")
    print("=" * 110)

    wave_data = {}
    for season in SEASONS:
        heights = []
        for hour in range(24):
            minutes = [wave_height_at_minute(hour, m, season) for m in range(60)]
            heights.append(float(np.mean(minutes)))
        wave_data[season] = heights

    print(f"\n{'Hour':<8} | " + " | ".join(f"{s:<14}" for s in SEASONS))
    print("-" * 110)
    for hour in range(24):
        row = f"{hour:>2}:00    | "
        for s in SEASONS:
            row += f"{wave_data[s][hour]:>12.3f} | "
        print(row)

    return wave_data


# =============================================================================
# SECTION 2 — MINUTE-LEVEL DETAIL (Peak / Moderate / Low)
# =============================================================================

def section_2_minute_detail(hourly_all):
    print("\n" + "=" * 110)
    print("SECTION 2: TIGERRAY MINUTE-LEVEL POWER — PEAK / MODERATE / LOW")
    print("=" * 110)

    for season in SEASONS:
        hours = classify_hours(hourly_all[season])
        print(f"\n{'─' * 110}")
        print(f"SEASON: {season}")
        print(f"  Peak hour     : {hours['Peak']:02d}:00")
        print(f"  Moderate hour : {hours['Moderate']:02d}:00")
        print(f"  Low hour      : {hours['Low']:02d}:00")
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
    print("SECTION 3: TIGERRAY HOURLY ENERGY SUMMARY (kJ)")
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
# SECTION 4 — DWF ALLOCATION FOR EVERY SEASON
# =============================================================================

def section_4_dwf_all_seasons(hourly_all):
    """
    Run DWF for each season and return a dict of results.
    """
    print("\n" + "=" * 110)
    print("SECTION 4: DWF ALLOCATION — TIGERRAY (ALL SEASONS)")
    print("=" * 110)

    all_dwf = {}

    for season in SEASONS:
        E = np.array([h['total_energy_kj'] for h in hourly_all[season]])
        L = np.ones(24)

        P_init = E / L
        P_dwf  = directional_water_filling(E, L)
        B      = battery_profile(P_dwf, E, L, B_init=INITIAL_BATTERY_KJ)

        all_dwf[season] = {
            'E': E,
            'P_init': P_init,
            'P_dwf': P_dwf,
            'battery': B,
        }

        # Print per-season table
        print(f"\n{'─' * 110}")
        print(f"SEASON: {season}")
        print(f"{'─' * 110}")
        print(f"{'Hour':<6} | {'E[i] (kJ)':<12} | {'P_init (W)':<12} | "
              f"{'P_dwf (W)':<12} | {'Battery (kJ)':<14}")
        print("-" * 70)
        for h in range(24):
            print(f"{h:>2}:00  | {E[h]:>10.3f} | {P_init[h]/3.6:>10.4f} | "
                  f"{P_dwf[h]/3.6:>10.4f} | {B[h]:>12.3f}")

        print("-" * 70)
        print(f"  Total harvested   : {np.sum(E):.3f} kJ")
        print(f"  Total delivered   : {np.sum(P_dwf * L):.3f} kJ")
        print(f"  Constant DWF power: {P_dwf[0]/3.6:.4f} W")
        print(f"  Peak battery      : {np.max(B):.3f} kJ")
        print(f"  Min battery       : {np.min(B):.3f} kJ")
        print(f"  THz bursts/day    : {np.sum(P_dwf * L) / 0.0117:,.0f}")

    return all_dwf


# =============================================================================
# SECTION 5 — ALL-SEASON SUMMARY
# =============================================================================

def section_5_all_seasons_summary(hourly_all, all_dwf):
    print("\n" + "=" * 110)
    print("SECTION 5: ALL-SEASON TIGERRAY SUMMARY")
    print("=" * 110)
    print(f"{'Season':<10} | {'Total (kJ)':<12} | {'Total (kWh)':<12} | "
          f"{'DWF Power (W)':<14} | {'Peak Bat (kJ)':<14} | {'THz bursts':<14}")
    print("-" * 110)

    for s in SEASONS:
        E = all_dwf[s]['E']
        total = float(np.sum(E))
        P_const = all_dwf[s]['P_dwf'][0] / 3.6
        peak_b = float(np.max(all_dwf[s]['battery']))
        bursts = total / 0.0117

        print(f"{s:<10} | {total:>10.1f} | {total/3600:>10.4f} | "
              f"{P_const:>12.4f} | {peak_b:>12.3f} | {bursts:>12,.0f}")


# =============================================================================
# SECTION 6 — PLOTS
# =============================================================================

def section_6_plots(wave_data, hourly_all, all_dwf):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Scenario 2 — Cape Town Coastal — TigerRAY (All Seasons)',
                 fontsize=15, fontweight='bold')
    colors = {'Summer': '#f1c40f', 'Autumn': '#e67e22',
              'Winter': '#3498db', 'Spring': '#2ecc71'}
    hours = np.arange(24)

    # Plot 1 — Wave heights
    ax = axes[0, 0]
    for s in SEASONS:
        ax.plot(hours, wave_data[s], 'o-', label=s,
                color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Wave Height (m)')
    ax.set_title('Hourly Wave Heights')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 2 — Hourly energy by season
    ax = axes[0, 1]
    for s in SEASONS:
        E = [h['total_energy_kj'] for h in hourly_all[s]]
        ax.plot(hours, E, 'o-', label=s, color=colors[s],
                linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Energy (kJ)')
    ax.set_title('TigerRAY Hourly Energy')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 3 — DWF allocation for all seasons
    ax = axes[1, 0]
    for s in SEASONS:
        ax.plot(hours, all_dwf[s]['P_dwf'] / 3.6, 'o-',
                label=s, color=colors[s], linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('DWF Power (W)')
    ax.set_title('DWF Allocation — All Seasons')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 4 — Battery level for all seasons
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

def export_csv(hourly_all, all_dwf):
    # Hourly energy for all seasons
    with open('tigerray_hourly.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Season', 'Hour', 'Energy_kJ', 'Peak_W'])
        for s in SEASONS:
            for h in range(24):
                r = hourly_all[s][h]
                w.writerow([s, h, f"{r['total_energy_kj']:.4f}",
                            f"{r['peak_power_w']:.4f}"])

    # DWF schedule for all seasons
    with open('tigerray_dwf_all_seasons.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Season', 'Hour', 'Harvested_kJ', 'P_dwf_W', 'Battery_kJ'])
        for s in SEASONS:
            for i in range(24):
                w.writerow([s, i,
                            f"{all_dwf[s]['E'][i]:.4f}",
                            f"{all_dwf[s]['P_dwf'][i]/3.6:.4f}",
                            f"{all_dwf[s]['battery'][i]:.4f}"])

    # Seasonal summary
    with open('tigerray_summary.csv', 'w', newline='') as f:
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

    print("\n✅ Saved: tigerray_hourly.csv")
    print("✅ Saved: tigerray_dwf_all_seasons.csv")
    print("✅ Saved: tigerray_summary.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 110)
    print("SCENARIO 2 — CAPE TOWN COASTAL — TIGERRAY + DWF (ALL SEASONS)")
    print("=" * 110)
    print(f"Technology  : {TIGERRAY['name']}")
    print(f"Cut-in      : {TIGERRAY['cut_in']} m")
    print(f"Rated       : {TIGERRAY['rated_height']} m @ {TIGERRAY['rated_power']} W")
    print(f"Cut-out     : {TIGERRAY['cut_out']} m")

    # 1. Hourly wave heights
    wave_data = section_1_hourly_heights()

    # 2. Harvest for all hours / seasons
    hourly_all = {s: harvest_hourly(s) for s in SEASONS}

    # 3. Minute detail at peak / moderate / low
    section_2_minute_detail(hourly_all)

    # 4. Hourly energy summary
    section_3_hourly_summary(hourly_all)

    # 5. DWF allocation for ALL seasons
    all_dwf = section_4_dwf_all_seasons(hourly_all)

    # 6. All-season summary
    section_5_all_seasons_summary(hourly_all, all_dwf)

    # 7. Plots
    section_6_plots(wave_data, hourly_all, all_dwf)

    # 8. CSV
    export_csv(hourly_all, all_dwf)

    print("\n" + "=" * 110)
    print("✅ COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()