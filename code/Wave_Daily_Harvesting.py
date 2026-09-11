import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

SEASONS = ['Summer', 'Autumn', 'Winter', 'Spring']

# Seasonal base wave heights (metres)
SEASONAL_WAVE_HEIGHTS = {
    'Summer': 1.5,
    'Autumn': 2.0,
    'Winter': 2.5,
    'Spring': 1.8
}

# WEC technologies — realistic parameters
TECHNOLOGIES = {
    'pendulum': {
        'name': 'Pendulum',
        'cut_in': 0.2,
        'rated_height': 0.6,
        'rated_power': 5.0,
        'cut_out': 2.0,
    },
    'point_absorber': {
        'name': 'Point Absorber',
        'cut_in': 0.5,
        'rated_height': 1.5,
        'rated_power': 6.8,
        'cut_out': 3.5,
    },
    'gyroscopic': {
        'name': 'Gyroscopic',
        'cut_in': 0.3,
        'rated_height': 1.0,
        'rated_power': 3.0,
        'cut_out': 3.0,
    },
    'hybrid': {
        'name': 'Hybrid',
        'cut_in': 0.2,
        'rated_height': 1.2,
        'rated_power': 10.0,
        'cut_out': 4.0,
    },
    'tigerray': {
        'name': 'TigerRAY',
        'cut_in': 0.5,
        'rated_height': 1.8,
        'rated_power': 55.0,
        'cut_out': 4.0,
    }
}


# ============================================================================
# WAVE HEIGHT MODEL
# ============================================================================

def wave_height_at_hour(hour, season, noise_level=0.12, seed=42):
    np.random.seed(seed + hour + hash(season) % 1000)
    
    base_height = SEASONAL_WAVE_HEIGHTS[season]
    
    # Tidal variation (semi-diurnal, peaks at hours 6 and 18)
    tidal = 0.15 * np.sin(2 * np.pi * (hour - 3) / 12.4)
    
    # Sea state noise (persistent random walk)
    noise = np.random.normal(0, noise_level)
    
    # Combined
    wave_h = base_height * (1 + tidal + noise)
    
    return max(0.1, wave_h)


def generate_wave_heights(season, seed=42):
    """Generate 24 hours of wave heights for a given season."""
    return [wave_height_at_hour(h, season, seed=seed) for h in range(24)]


# ============================================================================
# POWER CURVE
# ============================================================================

def power_output(wave_height, tech_key):
    p = TECHNOLOGIES[tech_key]
    
    if wave_height < p['cut_in']:
        return 0.0
    if wave_height > p['cut_out']:
        return 0.0
    if wave_height <= p['rated_height']:
        return p['rated_power'] * (wave_height / p['rated_height'])**2
    return p['rated_power']


# ============================================================================
# SECTION 1: WAVE HEIGHT FOR EACH SEASON (HOURLY)
# ============================================================================

def section_1_wave_heights():
    """Compare wave height across all four seasons, hour by hour."""
    
    print("\n" + "=" * 110)
    print("SECTION 1: WAVE HEIGHT FOR EACH SEASON (HOURLY)")
    print("=" * 110)
    
    # Generate wave heights for all seasons
    wave_data = {}
    for season in SEASONS:
        wave_data[season] = generate_wave_heights(season)
    
    # Print table
    print(f"\n{'Hour':<8} | {'Summer (m)':<14} | {'Autumn (m)':<14} | {'Winter (m)':<14} | {'Spring (m)':<14}")
    print("-" * 110)
    
    for hour in range(24):
        print(f"{hour:>2}:00    | "
              f"{wave_data['Summer'][hour]:>12.3f} | "
              f"{wave_data['Autumn'][hour]:>12.3f} | "
              f"{wave_data['Winter'][hour]:>12.3f} | "
              f"{wave_data['Spring'][hour]:>12.3f}")
    
    # Summary
    print("\n" + "-" * 110)
    print(f"{'STATS':<8} | {'Summer':<14} | {'Autumn':<14} | {'Winter':<14} | {'Spring':<14}")
    print("-" * 110)
    
    for label, func in [('Mean', np.mean), ('Min', np.min), ('Max', np.max), ('Std', np.std)]:
        row = f"{label:<8} | "
        for season in SEASONS:
            row += f"{func(wave_data[season]):>12.3f} | "
        print(row)
    
    return wave_data


# ============================================================================
# SECTION 2: POWER GENERATION FOR EACH TECHNOLOGY (HOURLY)
# ============================================================================

def section_2_power_by_technology():
    """For each technology, show hourly power for all four seasons."""
    
    print("\n" + "=" * 110)
    print("SECTION 2: POWER GENERATION FOR EACH TECHNOLOGY (HOURLY, W)")
    print("=" * 110)
    
    all_results = {}
    
    for tech_key, tech_params in TECHNOLOGIES.items():
        print(f"\n{'─' * 110}")
        print(f"TECHNOLOGY: {tech_params['name']}")
        print(f"  Cut-in: {tech_params['cut_in']} m | "
              f"Rated: {tech_params['rated_height']} m @ {tech_params['rated_power']} W | "
              f"Cut-out: {tech_params['cut_out']} m")
        print(f"{'─' * 110}")
        
        # Generate wave heights for each season
        wave_data = {}
        for season in SEASONS:
            wave_data[season] = generate_wave_heights(season)
        
        # Calculate power
        print(f"\n{'Hour':<8} | {'Summer (W)':<14} | {'Autumn (W)':<14} | {'Winter (W)':<14} | {'Spring (W)':<14}")
        print("-" * 110)
        
        season_powers = {s: [] for s in SEASONS}
        
        for hour in range(24):
            row = f"{hour:>2}:00    | "
            for season in SEASONS:
                p = power_output(wave_data[season][hour], tech_key)
                season_powers[season].append(p)
                row += f"{p:>12.3f} | "
            print(row)
        
        # Daily summary
        print("-" * 110)
        row = f"{'TOTAL':<8} | "
        for season in SEASONS:
            daily_kj = sum(season_powers[season]) * 3.6  # W·h to kJ: ×3.6
            row += f"{daily_kj:>10.1f}kJ | "
        print(row)
        
        all_results[tech_key] = season_powers
    
    return all_results


# ============================================================================
# SECTION 3: CUMULATIVE ENERGY
# ============================================================================

def section_3_cumulative(all_results):
    """Show cumulative energy across the day for each technology."""
    
    print("\n" + "=" * 110)
    print("SECTION 3: CUMULATIVE ENERGY ACROSS THE DAY (kJ)")
    print("=" * 110)
    
    for tech_key, season_powers in all_results.items():
        tech_name = TECHNOLOGIES[tech_key]['name']
        print(f"\n{tech_name}:")
        print(f"{'Hour':<8} | {'Summer':<12} | {'Autumn':<12} | {'Winter':<12} | {'Spring':<12}")
        print("-" * 70)
        
        # Calculate cumulative
        cumulative = {s: [] for s in SEASONS}
        for season in SEASONS:
            cum = 0
            for hour in range(24):
                cum += season_powers[season][hour] * 3.6  # W·h to kJ
                cumulative[season].append(cum)
        
        # Print every 4 hours
        for hour in range(0, 24, 4):
            print(f"{hour:>2}:00    | "
                  f"{cumulative['Summer'][hour]:>10.1f} | "
                  f"{cumulative['Autumn'][hour]:>10.1f} | "
                  f"{cumulative['Winter'][hour]:>10.1f} | "
                  f"{cumulative['Spring'][hour]:>10.1f}")
        
        # Final totals
        print(f"{'TOTAL':<8} | "
              f"{cumulative['Summer'][-1]:>10.1f} | "
              f"{cumulative['Autumn'][-1]:>10.1f} | "
              f"{cumulative['Winter'][-1]:>10.1f} | "
              f"{cumulative['Spring'][-1]:>10.1f}")


# ============================================================================
# SECTION 4: WINTER FOCUS — ALL TECHNOLOGIES COMPARED
# ============================================================================

def section_4_winter_focus(all_results):
    """Focus on Winter and compare all technologies hour by hour."""
    
    print("\n" + "=" * 110)
    print("SECTION 4: WINTER FOCUS — ALL TECHNOLOGIES COMPARED")
    print("=" * 110)
    
    # Generate winter wave heights
    winter_waves = generate_wave_heights('Winter')
    
    print(f"\nWinter Wave Height (m):")
    print(f"{'Hour':<8} | " + " | ".join([f"{h:>6}:00" for h in range(0, 24, 2)]))
    print("-" * 110)
    row = f"{'Height':<8} | "
    for hour in range(0, 24, 2):
        row += f"{winter_waves[hour]:>10.2f} | "
    print(row)
    
    # Table of all technologies for winter
    print(f"\n\nHourly Power Output — Winter (W):")
    print("-" * 110)
    header = f"{'Hour':<8} | "
    for tech_key in TECHNOLOGIES:
        header += f"{TECHNOLOGIES[tech_key]['name']:<13} | "
    print(header)
    print("-" * 110)
    
    for hour in range(24):
        row = f"{hour:>2}:00    | "
        for tech_key in TECHNOLOGIES:
            p = all_results[tech_key]['Winter'][hour]
            row += f"{p:>11.3f} | "
        print(row)
    
    # Daily totals
    print("-" * 110)
    row = f"{'TOTAL':<8} | "
    for tech_key in TECHNOLOGIES:
        daily_kj = sum(all_results[tech_key]['Winter']) * 3.6
        row += f"{daily_kj:>8.1f}kJ | "
    print(row)
    
    # Summary table
    print(f"\n\nWinter Summary:")
    print("-" * 90)
    print(f"{'Technology':<18} | {'Rated (m)':<12} | {'Rated (W)':<12} | {'Peak (W)':<12} | {'Daily (kJ)':<12} | {'Status':<15}")
    print("-" * 90)
    
    for tech_key, tech_params in TECHNOLOGIES.items():
        powers = all_results[tech_key]['Winter']
        peak = max(powers)
        daily_kj = sum(powers) * 3.6
        avg_wave = np.mean(generate_wave_heights('Winter'))
        
        if avg_wave < tech_params['cut_in']:
            status = "CUT-IN"
        elif avg_wave > tech_params['cut_out']:
            status = "CUT-OUT"
        elif avg_wave > tech_params['rated_height']:
            status = "SATURATED"
        else:
            status = "Scaling (H²)"
        
        print(f"{tech_params['name']:<18} | {tech_params['rated_height']:>10.1f} | "
              f"{tech_params['rated_power']:>10.1f} | {peak:>10.3f} | "
              f"{daily_kj:>10.1f} | {status:<15}")


# ============================================================================
# SECTION 5: PLOTTING
# ============================================================================

def section_5_plots(wave_data, all_results):
    """Generate plots for all sections."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Wave Energy Harvesting — Complete Analysis', fontsize=16, fontweight='bold')
    
    season_colors = {
        'Summer': '#f1c40f',
        'Autumn': '#e67e22',
        'Winter': '#3498db',
        'Spring': '#2ecc71'
    }
    
    hours = range(24)
    
    # Plot 1: Wave height for each season
    ax = axes[0, 0]
    for season in SEASONS:
        ax.plot(hours, wave_data[season], 'o-', label=season,
                color=season_colors[season], linewidth=2, markersize=4)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Wave Height (m)')
    ax.set_title('Wave Height for Each Season')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 2: Power for each technology (Winter context)
    ax = axes[0, 1]
    tech_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']
    for i, (tech_key, tech_params) in enumerate(TECHNOLOGIES.items()):
        powers = all_results[tech_key]['Winter']
        ax.plot(hours, powers, 'o-', label=tech_params['name'],
                color=tech_colors[i], linewidth=2, markersize=3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power Output (W)')
    ax.set_title('Power by Technology — Winter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 3: Cumulative energy (Winter)
    ax = axes[1, 0]
    for i, (tech_key, tech_params) in enumerate(TECHNOLOGIES.items()):
        powers = all_results[tech_key]['Winter']
        cumulative = np.cumsum(powers) * 3.6
        ax.plot(hours, cumulative, 'o-', label=tech_params['name'],
                color=tech_colors[i], linewidth=2, markersize=3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Cumulative Energy (kJ)')
    ax.set_title('Cumulative Energy — Winter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 4: Daily totals by technology and season
    ax = axes[1, 1]
    x = np.arange(len(TECHNOLOGIES))
    width = 0.2
    
    for i, season in enumerate(SEASONS):
        totals = []
        for tech_key in TECHNOLOGIES:
            daily_kj = sum(all_results[tech_key][season]) * 3.6
            totals.append(daily_kj)
        
        offset = (i - len(SEASONS)/2 + 0.5) * width
        ax.bar(x + offset, totals, width, label=season, color=season_colors[season])
    
    ax.set_xlabel('Technology')
    ax.set_ylabel('Daily Energy (kJ)')
    ax.set_title('Daily Energy by Technology and Season')
    ax.set_xticks(x)
    ax.set_xticklabels([TECHNOLOGIES[t]['name'] for t in TECHNOLOGIES], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 110)
    print("WAVE ENERGY HARVESTING — COMPLETE ANALYSIS")
    print("For Underwater THz Communication Investigation")
    print("=" * 110)
    
    # Section 1: Wave height for each season
    wave_data = section_1_wave_heights()
    
    # Section 2: Power for each technology
    all_results = section_2_power_by_technology()
    
    # Section 3: Cumulative energy
    section_3_cumulative(all_results)
    
    # Section 4: Winter focus
    section_4_winter_focus(all_results)
    
    # Section 5: Plots
    section_5_plots(wave_data, all_results)
    
    # Section 6: CSV export
    section_6_export(wave_data, all_results)
    
    print("\n" + "=" * 110)
    print("✅ COMPLETE")
    print("=" * 110)


def run_wave_profile(show_plots=True, export_csv=True, seed=42):
    """Run the wave harvesting profile and return (wave_data, all_results).

    - show_plots: when True, will call section_5_plots
    - export_csv: when True, will call section_6_export
    """
    # Section 1
    wave_data = section_1_wave_heights()

    # Section 2
    all_results = section_2_power_by_technology()

    # Section 3 (prints)
    section_3_cumulative(all_results)

    # Section 4 (prints)
    section_4_winter_focus(all_results)

    # Section 5: plots
    if show_plots:
        try:
            section_5_plots(wave_data, all_results)
        except Exception:
            pass

    return wave_data, all_results