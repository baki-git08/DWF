import numpy as np
import matplotlib.pyplot as plt
import csv

from solar_for_hour import MinuteSolarHarvester


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASONS = ['summer', 'autumn', 'winter', 'spring']

SCENARIOS = {
    '1h_peak':        [12],
    '3h_around_peak': [11, 12, 13],
    '5h_around_peak': [10, 11, 12, 13, 14],
    'morning_noon':   [9, 12],
    'dawn_peak_dusk': [8, 12, 16],
}

PANEL_WIDTH_M   = 5.0
PANEL_HEIGHT_M  = 1.0     # → 5 m²
BASE_EFFICIENCY = 0.20
NOISE_LEVEL     = 0.15
SEED            = 42

# Battery carried over from the previous day (kJ)
# Set to 0 for strict same-day causality
INITIAL_BATTERY_KJ = 0.0


# =============================================================================
# DWF ALGORITHM
# =============================================================================

def compute_initial_power(E, L):
    E = np.asarray(E, dtype=float)
    L = np.asarray(L, dtype=float)
    P = np.zeros_like(E)
    for i in range(len(E)):
        if L[i] > 0:
            P[i] = E[i] / L[i]
    return P


def directional_water_filling(E, L):
    E = np.asarray(E, dtype=float)
    L = np.asarray(L, dtype=float)
    P = compute_initial_power(E, L)
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


def build_master_schedule(season, harvest_hours,
                          B_init=INITIAL_BATTERY_KJ):
    harvester = MinuteSolarHarvester(
        panel_width_m=PANEL_WIDTH_M,
        panel_height_m=PANEL_HEIGHT_M,
        base_efficiency=BASE_EFFICIENCY,
        location='Johannesburg',
        noise_level=NOISE_LEVEL,
        seed=SEED,
    )

    info = harvester.seasons[season]

    E = np.zeros(24)
    for h in harvest_hours:
        res = harvester.harvest_minute_hour(
            h, season,
            sunrise=info['sunrise'], sunset=info['sunset'],
            add_noise=True,
        )
        E[h] = res['total_energy_kj']

    L = np.ones(24)   # one-hour blocks

    # --- DWF allocation ---
    P_dwf = directional_water_filling(E, L)
    P_init = compute_initial_power(E, L)

    # --- Battery ---
    B = battery_profile(P_dwf, E, L, B_init=B_init)

    return {
        'season': season,
        'hours': list(range(24)),
        'harvest_hours': harvest_hours,
        'E': E,
        'L': L,
        'P_initial': P_init,
        'P_dwf': P_dwf,
        'battery': B,
        'total_harvested_kj': float(np.sum(E)),
        'total_delivered_kj': float(np.sum(P_dwf * L)),
        'peak_power_w': float(np.max(P_dwf) / 3.6),
    }


# PRINTING

def print_summary_table(season, results):
    print(f"\n{'=' * 110}")
    print(f"SEASON: {season.upper()}   |   Panel: {PANEL_WIDTH_M} m × "
          f"{PANEL_HEIGHT_M} m = {PANEL_WIDTH_M * PANEL_HEIGHT_M:.1f} m²")
    print(f"{'=' * 110}")
    print(f"{'Scenario':<18} | {'Harvest (h)':<14} | {'Harvested (kJ)':<16} | "
          f"{'Delivered (kJ)':<16} | {'Constant Power (W)':<20} | {'Delivery Window':<15}")
    print("-" * 110)

    for label, r in results.items():
        P_w = r['P_dwf'] / 3.6
        active = [i for i in range(24) if r['P_dwf'][i] > 0]
        if active:
            p_constant = P_w[active[0]]
            window = f"{active[0]:02d}:00–{active[-1]:02d}:00"
        else:
            p_constant = 0.0
            window = "—"

        print(f"{label:<18} | {str(r['harvest_hours']):<14} | "
              f"{r['total_harvested_kj']:>14.2f} | "
              f"{r['total_delivered_kj']:>14.2f} | "
              f"{p_constant:>18.4f} | {window:<15}")

    print("-" * 110)


def print_detailed_table(r):
    """Print per-hour breakdown for one scenario."""
    season = r['season']
    hours = r['hours']
    E = r['E']
    P = r['P_dwf'] / 3.6
    B = r['battery']

    print(f"\n{'─' * 90}")
    print(f"Hourly breakdown — {season} | harvest at {r['harvest_hours']}")
    print(f"{'─' * 90}")
    print(f"{'Hour':<6} | {'E[i] (kJ)':<12} | {'P_dwf (W)':<12} | "
          f"{'Battery (kJ)':<14} | {'Status':<15}")
    print("-" * 90)

    for h in hours:
        if E[h] > 0:
            status = "HARVEST"
        elif P[h] > 0:
            status = "deliver"
        else:
            status = "idle"
        print(f"{h:>2}:00  | {E[h]:>10.2f} | {P[h]:>10.4f} | "
              f"{B[h]:>12.2f} | {status:<15}")


# PLOTTING

def plot_season(season, results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"DWF + Solar (5 m²) — {season.capitalize()}",
                 fontsize=15, fontweight='bold')
    hours = np.arange(24)

    # Plot 1 — Harvested energy
    ax = axes[0, 0]
    for i, (label, r) in enumerate(results.items()):
        ax.bar(hours + 0.15 * i - 0.3, r['E'], width=0.15,
               label=label, edgecolor='black')
    ax.set_xlabel('Hour'); ax.set_ylabel('Harvested (kJ)')
    ax.set_title('Harvested Energy by Scenario')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 2 — DWF allocation
    ax = axes[0, 1]
    for label, r in results.items():
        ax.plot(hours, r['P_dwf'] / 3.6, 'o-',
                label=label, linewidth=1.5, markersize=3)
    ax.set_xlabel('Hour'); ax.set_ylabel('Power (W)')
    ax.set_title('DWF Power Delivery')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 3 — Battery
    ax = axes[1, 0]
    for label, r in results.items():
        ax.plot(hours, r['battery'], 'o-',
                label=label, linewidth=1.5, markersize=3)
    ax.set_xlabel('Hour'); ax.set_ylabel('Battery (kJ)')
    ax.set_title('Battery Level')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 4 — Delivered energy
    ax = axes[1, 1]
    labels = list(results.keys())
    daily = [r['total_delivered_kj'] for r in results.values()]
    bars = ax.bar(labels, daily, color='#2ecc71', edgecolor='black')
    for b, v in zip(bars, daily):
        ax.text(b.get_x() + b.get_width() / 2, v + 3,
                f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Delivered (kJ/day)')
    ax.set_title('Total Delivered Energy')
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    plt.show()


# MAIN

def main():
    print("=" * 110)
    print("DWF + SOLAR INTEGRATION — MASTER NODE (5 m² PANEL)")
    print("=" * 110)
    print(f"  Panel     : {PANEL_WIDTH_M} m × {PANEL_HEIGHT_M} m "
          f"= {PANEL_WIDTH_M * PANEL_HEIGHT_M:.1f} m²")
    print(f"  Efficiency: {BASE_EFFICIENCY * 100:.0f}%")
    print(f"  Location  : Johannesburg")
    print(f"  DWF mode  : forward-only (causality-respecting)")
    print(f"  B_init    : {INITIAL_BATTERY_KJ:.1f} kJ")

    all_season_results = {}

    for season in SEASONS:
        results = {}
        for label, hours in SCENARIOS.items():
            r = build_master_schedule(season, hours)
            results[label] = r
        all_season_results[season] = results

        print_summary_table(season, results)

    # Detailed breakdown for summer, 3-hour scenario
    print("\n" + "=" * 110)
    print("DETAILED HOURLY BREAKDOWN (Summer, 3h around peak)")
    print("=" * 110)
    print_detailed_table(all_season_results['summer']['3h_around_peak'])

    # Plots
    for season in SEASONS:
        plot_season(season, all_season_results[season])

    print("\n" + "=" * 110)
    print("✅ DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()