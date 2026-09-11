import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

SEASONS = ['Summer', 'Autumn', 'Winter', 'Spring']

# Monthly surface temperatures for South African coast (°C)
SEASONAL_SURFACE_TEMPS = {
    'Summer': 26.0,
    'Autumn': 22.5,
    'Winter': 20.5,
    'Spring': 22.0
}

# Deep sea temperature (constant at ~1000m depth, °C)
DEEP_TEMP = 4.0

# TEG technologies — realistic parameters
TECHNOLOGIES = {
    'teg_submersible': {
        'name': 'TEG Submersible',
        'ref_delta_T': 21.0,
        'rated_power': 0.35,
        'description': 'Buoyancy-driven TEG with PCM storage'
    },
    'teg_otec': {
        'name': 'TEG OTEC',
        'ref_delta_T': 21.0,
        'rated_power': 3.01,
        'description': 'Bi₂Te₃-based TEG OTEC system'
    },
    'teg_submarine': {
        'name': 'TEG Submarine',
        'ref_delta_T': 15.0,
        'rated_power': 0.35,
        'description': 'TEG for AUV/submarine applications'
    },
    'teg_hydrothermal': {
        'name': 'TEG Hydrothermal',
        'ref_delta_T': 246.0,
        'rated_power': 3.25,
        'description': 'Hydrothermal vent TEG system'
    },
    'micro_otec': {
        'name': 'Micro-OTEC',
        'ref_delta_T': 21.0,
        'rated_power': 714.6,
        'description': 'Micro-OTEC unit (kW scale)'
    },
    'rtg': {
        'name': 'RTG',
        'ref_delta_T': 21.0,
        'rated_power': 0.00233,
        'description': 'Radioisotope TEG for WSN nodes'
    }
}


# ============================================================================
# TEMPERATURE MODEL
# ============================================================================

def surface_temp_at_hour(hour, season, noise_level=0.05, seed=42):

    np.random.seed(seed + hour + hash(season) % 1000)
    
    base_temp = SEASONAL_SURFACE_TEMPS[season]
    
    # Diurnal variation (±1.5°C, peak at 15:00)
    diurnal = 1.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    
    # Small noise
    noise = np.random.normal(0, noise_level)
    
    return base_temp + diurnal + noise


def generate_surface_temps(season, seed=42):
    return [surface_temp_at_hour(h, season, seed=seed) for h in range(24)]


def temperature_difference(surface_temp, deep_temp=DEEP_TEMP):
    return max(0, surface_temp - deep_temp)


# ============================================================================
# POWER CURVE
# ============================================================================

def power_output(delta_T, tech_key):
    p = TECHNOLOGIES[tech_key]
    
    if delta_T <= 0:
        return 0.0
    
    # Linear scaling with ΔT
    power = p['rated_power'] * (delta_T / p['ref_delta_T'])
    
    # Saturate at rated power
    return min(power, p['rated_power'])


# ============================================================================
# SECTION 1: TEMPERATURE GRADIENT FOR EACH SEASON (HOURLY)
# ============================================================================

def section_1_temperatures():
    print("\n" + "=" * 110)
    print("SECTION 1: TEMPERATURE GRADIENT FOR EACH SEASON (HOURLY)")
    print("=" * 110)
    print(f"Deep sea temperature (constant): {DEEP_TEMP}°C")
    
    # Generate surface temperatures for all seasons
    temp_data = {}
    delta_data = {}
    for season in SEASONS:
        temp_data[season] = generate_surface_temps(season)
        delta_data[season] = [temperature_difference(t) for t in temp_data[season]]
    
    # Print surface temperature table
    print(f"\nSurface Temperature (°C):")
    print(f"{'Hour':<8} | {'Summer':<14} | {'Autumn':<14} | {'Winter':<14} | {'Spring':<14}")
    print("-" * 110)
    
    for hour in range(24):
        print(f"{hour:>2}:00    | "
              f"{temp_data['Summer'][hour]:>12.3f} | "
              f"{temp_data['Autumn'][hour]:>12.3f} | "
              f"{temp_data['Winter'][hour]:>12.3f} | "
              f"{temp_data['Spring'][hour]:>12.3f}")
    
    # Print ΔT table
    print(f"\nTemperature Difference ΔT (°C):")
    print(f"{'Hour':<8} | {'Summer':<14} | {'Autumn':<14} | {'Winter':<14} | {'Spring':<14}")
    print("-" * 110)
    
    for hour in range(24):
        print(f"{hour:>2}:00    | "
              f"{delta_data['Summer'][hour]:>12.3f} | "
              f"{delta_data['Autumn'][hour]:>12.3f} | "
              f"{delta_data['Winter'][hour]:>12.3f} | "
              f"{delta_data['Spring'][hour]:>12.3f}")
    
    # Summary
    print("\n" + "-" * 110)
    print(f"{'STATS':<12} | {'Summer':<14} | {'Autumn':<14} | {'Winter':<14} | {'Spring':<14}")
    print("-" * 110)
    
    for label, func in [('Mean ΔT', np.mean), ('Min ΔT', np.min), ('Max ΔT', np.max), ('Std ΔT', np.std)]:
        row = f"{label:<12} | "
        for season in SEASONS:
            row += f"{func(delta_data[season]):>12.3f} | "
        print(row)
    
    return temp_data, delta_data


# ============================================================================
# SECTION 2: POWER GENERATION FOR EACH TECHNOLOGY (HOURLY)
# ============================================================================

def section_2_power_by_technology(delta_data):
    """For each technology, show hourly power for all four seasons."""
    
    print("\n" + "=" * 110)
    print("SECTION 2: POWER GENERATION FOR EACH TECHNOLOGY (HOURLY, W)")
    print("=" * 110)
    
    all_results = {}
    
    for tech_key, tech_params in TECHNOLOGIES.items():
        print(f"\n{'─' * 110}")
        print(f"TECHNOLOGY: {tech_params['name']}")
        print(f"  Reference ΔT: {tech_params['ref_delta_T']}°C | "
              f"Rated Power: {tech_params['rated_power']} W")
        print(f"  {tech_params['description']}")
        print(f"{'─' * 110}")
        
        print(f"\n{'Hour':<8} | {'Summer (W)':<14} | {'Autumn (W)':<14} | {'Winter (W)':<14} | {'Spring (W)':<14}")
        print("-" * 110)
        
        season_powers = {s: [] for s in SEASONS}
        
        for hour in range(24):
            row = f"{hour:>2}:00    | "
            for season in SEASONS:
                p = power_output(delta_data[season][hour], tech_key)
                season_powers[season].append(p)
                row += f"{p:>12.6f} | "
            print(row)
        
        # Daily summary
        print("-" * 110)
        row = f"{'TOTAL':<8} | "
        for season in SEASONS:
            daily_kj = sum(season_powers[season]) * 3.6
            row += f"{daily_kj:>10.4f}kJ | "
        print(row)
        
        all_results[tech_key] = season_powers
    
    return all_results


# ============================================================================
# SECTION 3: CUMULATIVE ENERGY
# ============================================================================

def section_3_cumulative(all_results):
    print("\n" + "=" * 110)
    print("SECTION 3: CUMULATIVE ENERGY ACROSS THE DAY (kJ)")
    print("=" * 110)
    
    for tech_key, season_powers in all_results.items():
        tech_name = TECHNOLOGIES[tech_key]['name']
        print(f"\n{tech_name}:")
        print(f"{'Hour':<8} | {'Summer':<12} | {'Autumn':<12} | {'Winter':<12} | {'Spring':<12}")
        print("-" * 70)
        
        cumulative = {s: [] for s in SEASONS}
        for season in SEASONS:
            cum = 0
            for hour in range(24):
                cum += season_powers[season][hour] * 3.6
                cumulative[season].append(cum)
        
        for hour in range(0, 24, 4):
            print(f"{hour:>2}:00    | "
                  f"{cumulative['Summer'][hour]:>10.4f} | "
                  f"{cumulative['Autumn'][hour]:>10.4f} | "
                  f"{cumulative['Winter'][hour]:>10.4f} | "
                  f"{cumulative['Spring'][hour]:>10.4f}")
        
        print(f"{'TOTAL':<8} | "
              f"{cumulative['Summer'][-1]:>10.4f} | "
              f"{cumulative['Autumn'][-1]:>10.4f} | "
              f"{cumulative['Winter'][-1]:>10.4f} | "
              f"{cumulative['Spring'][-1]:>10.4f}")


# ============================================================================
# SECTION 4: SUMMER FOCUS — ALL TECHNOLOGIES COMPARED
# ============================================================================

def section_4_summer_focus(all_results, delta_data):
    print("\n" + "=" * 110)
    print("SECTION 4: SUMMER FOCUS — ALL TECHNOLOGIES COMPARED")
    print("=" * 110)
    print("Summer has the HIGHEST ΔT (22.0°C) → Maximum thermal power")
    
    print(f"\nSummer ΔT (°C):")
    print(f"{'Hour':<8} | " + " | ".join([f"{h:>8}:00" for h in range(0, 24, 2)]))
    print("-" * 110)
    row = f"{'ΔT':<8} | "
    for hour in range(0, 24, 2):
        row += f"{delta_data['Summer'][hour]:>12.2f} | "
    print(row)
    
    # Table of all technologies for summer
    print(f"\n\nHourly Power Output — Summer (W):")
    print("-" * 120)
    header = f"{'Hour':<8} | "
    for tech_key in TECHNOLOGIES:
        header += f"{TECHNOLOGIES[tech_key]['name']:<16} | "
    print(header)
    print("-" * 120)
    
    for hour in range(24):
        row = f"{hour:>2}:00    | "
        for tech_key in TECHNOLOGIES:
            p = all_results[tech_key]['Summer'][hour]
            row += f"{p:>14.6f} | "
        print(row)
    
    # Daily totals
    print("-" * 120)
    row = f"{'TOTAL':<8} | "
    for tech_key in TECHNOLOGIES:
        daily_kj = sum(all_results[tech_key]['Summer']) * 3.6
        row += f"{daily_kj:>11.4f}kJ | "
    print(row)
    
    # Summary table
    print(f"\n\nSummer Summary:")
    print("-" * 100)
    print(f"{'Technology':<18} | {'Ref ΔT (°C)':<14} | {'Rated (W)':<12} | {'Peak (W)':<12} | {'Daily (kJ)':<12}")
    print("-" * 100)
    
    for tech_key, tech_params in TECHNOLOGIES.items():
        powers = all_results[tech_key]['Summer']
        peak = max(powers)
        daily_kj = sum(powers) * 3.6
        print(f"{tech_params['name']:<18} | {tech_params['ref_delta_T']:>12.1f} | "
              f"{tech_params['rated_power']:>10.4f} | {peak:>10.4f} | "
              f"{daily_kj:>10.4f}")


# ============================================================================
# SECTION 5: PLOTTING
# ============================================================================

def section_5_plots(temp_data, delta_data, all_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Thermal Energy Harvesting — Complete Analysis', fontsize=16, fontweight='bold')
    
    season_colors = {
        'Summer': '#f1c40f',
        'Autumn': '#e67e22',
        'Winter': '#3498db',
        'Spring': '#2ecc71'
    }
    
    tech_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']
    hours = range(24)
    
    # Plot 1: Surface temperature for each season
    ax = axes[0, 0]
    for season in SEASONS:
        ax.plot(hours, temp_data[season], 'o-', label=season,
                color=season_colors[season], linewidth=2, markersize=4)
    ax.axhline(y=DEEP_TEMP, color='navy', linestyle='--', label=f'Deep ({DEEP_TEMP}°C)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Surface Temperature by Season')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 2: ΔT for each season
    ax = axes[0, 1]
    for season in SEASONS:
        ax.plot(hours, delta_data[season], 'o-', label=season,
                color=season_colors[season], linewidth=2, markersize=4)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Temperature Difference ΔT (°C)')
    ax.set_title('Temperature Difference by Season')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 3: Power by technology (SUMMER)
    ax = axes[1, 0]
    for i, (tech_key, tech_params) in enumerate(TECHNOLOGIES.items()):
        powers = all_results[tech_key]['Summer']
        ax.plot(hours, powers, 'o-', label=tech_params['name'],
                color=tech_colors[i % len(tech_colors)], linewidth=2, markersize=3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power Output (W)')
    ax.set_title('Power by Technology — Summer')
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
    print("THERMAL ENERGY HARVESTING — COMPLETE ANALYSIS")
    print("For Underwater THz Communication Investigation")
    print("=" * 110)
    
    # Section 1: Temperature gradient for each season
    temp_data, delta_data = section_1_temperatures()
    
    # Section 2: Power for each technology
    all_results = section_2_power_by_technology(delta_data)
    
    # Section 3: Cumulative energy
    section_3_cumulative(all_results)
    
    # Section 4: SUMMER focus
    section_4_summer_focus(all_results, delta_data)
    
    # Section 5: Plots
    section_5_plots(temp_data, delta_data, all_results)
    
    
    print("\n" + "=" * 110)
    print("✅ COMPLETE")
    print("=" * 110)