import numpy as np
import matplotlib.pyplot as plt


class MinuteSolarHarvester:
    def __init__(self, panel_area=5.0, base_efficiency=0.20, location='Johannesburg',
                 noise_level=0.15, seed=42):

        self.panel_area = panel_area
        self.base_efficiency = base_efficiency
        self.location = location
        self.noise_level = noise_level
        self.seed = seed
        
        np.random.seed(seed)
        
        # Location data - surface irradiance (kWh/m²/day)
        self.locations = {
            'Johannesburg': {'summer': 6.38, 'autumn': 5.81, 'winter': 4.80, 'spring': 7.23},
            'Cape Town': {'summer': 8.98, 'autumn': 5.03, 'winter': 3.56, 'spring': 7.22},
            'Durban': {'summer': 5.5, 'autumn': 4.5, 'winter': 3.8, 'spring': 5.0}
        }
        
        self.location_data = self.locations.get(location, self.locations['Johannesburg'])
        
        # Seasonal efficiency factors
        self.seasonal_efficiency = {
            'summer': 0.90,
            'autumn': 0.97,
            'winter': 1.02,
            'spring': 0.95
        }
        
        # Seasonal noise multipliers
        self.seasonal_noise = {
            'summer': 1.0,
            'autumn': 0.8,
            'winter': 1.2,
            'spring': 0.9
        }
        
        # Seasonal mapping with sunrise/sunset
        self.seasons = {
            'summer': {'label': 'Summer', 'sunrise': 5, 'sunset': 19, 'eff_factor': 0.90},
            'autumn': {'label': 'Autumn', 'sunrise': 6, 'sunset': 18, 'eff_factor': 0.97},
            'winter': {'label': 'Winter', 'sunrise': 7, 'sunset': 17, 'eff_factor': 1.02},
            'spring': {'label': 'Spring', 'sunrise': 6, 'sunset': 18, 'eff_factor': 0.95}
        }
    
    def get_efficiency(self, season):
        eff_factor = self.seasonal_efficiency.get(season, 1.0)
        return self.base_efficiency * eff_factor
    
    def irradiance_at_hour(self, hour, season, sunrise=6, sunset=18):
        total_irradiance = self.location_data[season]
        daylight_hours = sunset - sunrise
        
        if hour < sunrise or hour > sunset:
            return 0.0
        
        t = (hour - sunrise) / daylight_hours
        normalized_irradiance = np.sin(np.pi * t)
        scaling_factor = (np.pi / 2) * (total_irradiance / daylight_hours)
        irradiance = normalized_irradiance * scaling_factor
        
        return max(0, irradiance)
    
    def irradiance_at_minute(self, hour, minute, season, sunrise=6, sunset=18):
        fractional_hour = hour + minute / 60.0
        return self.irradiance_at_hour(fractional_hour, season, sunrise, sunset)
    
    def apply_minute_noise(self, irradiance, season, minute, prev_noise=None):
        if irradiance <= 0:
            return 0.0, 0.0
        
        noise_mult = self.seasonal_noise.get(season, 1.0)
        base_noise = self.noise_level * noise_mult
        
        # 1. Cloud movement noise (rapid fluctuations)
        cloud_factor = np.random.beta(2, 5)
        cloud_noise = (cloud_factor - 0.2) * 0.6
        
        # 2. Atmospheric turbulence (high-frequency)
        turbulence_noise = np.random.normal(0, 0.06)
        
        # 3. Occasional passing shadow (rare, sudden drop)
        shadow_noise = 0.0
        if np.random.random() < 0.03:  # 3% chance per minute
            shadow_noise = -np.random.uniform(0.1, 0.4)
        
        # Combine noise sources
        total_noise = cloud_noise + turbulence_noise + shadow_noise
        
        # Add persistence from previous minute (60% persistence)
        if prev_noise is not None:
            total_noise = 0.6 * prev_noise + 0.4 * total_noise
        
        # Scale by base noise level
        scaled_noise = total_noise * base_noise
        
        # Apply noise
        noisy_irradiance = irradiance * (1 + scaled_noise)
        
        return max(0, noisy_irradiance), scaled_noise
    
    def harvest_minute_hour(self, hour, season, sunrise=6, sunset=18, add_noise=True):
        minutes_data = []
        total_energy_kj = 0
        total_clean_energy_kj = 0
        peak_power_w = 0
        peak_minute = 0
        prev_noise = None
        
        eff = self.get_efficiency(season)
        
        for minute in range(60):
            # Base irradiance (smooth)
            base_irradiance = self.irradiance_at_minute(hour, minute, season, sunrise, sunset)
            
            # Apply noise if requested
            if add_noise:
                irradiance, noise = self.apply_minute_noise(
                    base_irradiance, season, minute, prev_noise
                )
                prev_noise = noise
            else:
                irradiance = base_irradiance
                noise = 0.0
            
            # Power output at this minute (Watts)
            power_w = irradiance * self.panel_area * eff * 1000
            
            # Energy in this minute (kJ)
            energy_kj = power_w * 60 / 1000  # 60 seconds per minute
            
            # Base energy (no noise)
            base_power_w = base_irradiance * self.panel_area * eff * 1000
            base_energy_kj = base_power_w * 60 / 1000
            
            minutes_data.append({
                'minute': minute,
                'time': f"{hour:02d}:{minute:02d}",
                'irradiance': irradiance,
                'base_irradiance': base_irradiance,
                'power_watts': power_w,
                'base_power_watts': base_power_w,
                'energy_kj': energy_kj,
                'base_energy_kj': base_energy_kj,
                'noise_factor': noise
            })
            
            total_energy_kj += energy_kj
            total_clean_energy_kj += base_energy_kj
            if power_w > peak_power_w:
                peak_power_w = power_w
                peak_minute = minute
        
        return {
            'hour': hour,
            'season': season,
            'minutes_data': minutes_data,
            'total_energy_kj': total_energy_kj,
            'total_clean_energy_kj': total_clean_energy_kj,
            'peak_power_w': peak_power_w,
            'peak_minute': peak_minute,
            'efficiency': eff,
            'has_noise': add_noise
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    harvester = MinuteSolarHarvester(
        panel_area=5.0,
        base_efficiency=0.20,
        location='Johannesburg',
        noise_level=0.15,
        seed=42
    )
    
    print("=" * 100)
    print("SOLAR ENERGY HARVESTING - MINUTE-LEVEL DATA WITHIN AN HOUR")
    print(f"Location: {harvester.location}")
    print(f"Panel Area: {harvester.panel_area} m²")
    print(f"Base Panel Efficiency: {harvester.base_efficiency * 100:.0f}%")
    print(f"Noise Level: {harvester.noise_level * 100:.0f}%")
    print("=" * 100)
    
    # Select peak hour for each season (around solar noon)
    peak_hours = {
        'summer': 12,   # Solar noon in summer
        'autumn': 12,
        'winter': 12,
        'spring': 12
    }
    
    season_labels = {
        'summer': 'Summer',
        'autumn': 'Autumn',
        'winter': 'Winter',
        'spring': 'Spring'
    }
    
    # Store results
    all_results = {}
    
    for season_key, hour in peak_hours.items():
        season_info = harvester.seasons[season_key]
        result = harvester.harvest_minute_hour(
            hour, season_key,
            sunrise=season_info['sunrise'],
            sunset=season_info['sunset'],
            add_noise=True
        )
        all_results[season_key] = result
    
    # ============================================================================
    # PRINT MINUTE-LEVEL TABLE (every 5 minutes)
    # ============================================================================
    
    print(f"\nMINUTE-LEVEL SOLAR POWER OUTPUT (W) - PEAK HOUR (12:00):")
    print("-" * 95)
    print(f"{'Time':<8} | {'Summer (W)':<14} | {'Autumn (W)':<14} | {'Winter (W)':<14} | {'Spring (W)':<14}")
    print("-" * 95)
    
    for minute in range(0, 60, 5):  # Every 5 minutes
        summer_power = all_results['summer']['minutes_data'][minute]['power_watts']
        autumn_power = all_results['autumn']['minutes_data'][minute]['power_watts']
        winter_power = all_results['winter']['minutes_data'][minute]['power_watts']
        spring_power = all_results['spring']['minutes_data'][minute]['power_watts']
        
        time_str = f"12:{minute:02d}"
        print(f"{time_str:<8} | {summer_power:>12.1f} | {autumn_power:>12.1f} | {winter_power:>12.1f} | {spring_power:>12.1f}")
    
    print("=" * 95)
    
    # ============================================================================
    # PRINT MINUTE-LEVEL ENERGY TABLE (every 5 minutes)
    # ============================================================================
    
    print(f"\nMINUTE-LEVEL ENERGY GENERATION (kJ) - PEAK HOUR (12:00):")
    print("-" * 95)
    print(f"{'Time':<8} | {'Summer (kJ)':<14} | {'Autumn (kJ)':<14} | {'Winter (kJ)':<14} | {'Spring (kJ)':<14}")
    print("-" * 95)
    
    for minute in range(0, 60, 5):
        summer_energy = all_results['summer']['minutes_data'][minute]['energy_kj']
        autumn_energy = all_results['autumn']['minutes_data'][minute]['energy_kj']
        winter_energy = all_results['winter']['minutes_data'][minute]['energy_kj']
        spring_energy = all_results['spring']['minutes_data'][minute]['energy_kj']
        
        time_str = f"12:{minute:02d}"
        print(f"{time_str:<8} | {summer_energy:>12.3f} | {autumn_energy:>12.3f} | {winter_energy:>12.3f} | {spring_energy:>12.3f}")
    
    print("=" * 95)
    
    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    
    print("\nSUMMARY - ONE HOUR AT PEAK (12:00):")
    print("-" * 100)
    print(f"{'Season':<10} | {'Efficiency':<12} | {'Total (kJ)':<14} | {'Total (kWh)':<14} | {'Peak Power (W)':<16} | {'Peak Minute':<12}")
    print("-" * 100)
    
    for season_key, result in all_results.items():
        label = season_labels[season_key]
        eff = result['efficiency'] * 100
        total_kj = result['total_energy_kj']
        peak_w = result['peak_power_w']
        peak_min = result['peak_minute']
        
        print(f"{label:<10} | {eff:>10.1f}% | {total_kj:>12.3f} | {total_kj/3600:>12.5f} | {peak_w:>14.1f} | {peak_min:>3} min")
    
    print("=" * 100)
    
    # ============================================================================
    # FLUCTUATION ANALYSIS
    # ============================================================================
    
    print("\nFLUCTUATION ANALYSIS (Variation within the hour):")
    print("-" * 90)
    print(f"{'Season':<10} | {'Mean Power (W)':<16} | {'Std Dev (W)':<14} | {'Min (W)':<12} | {'Max (W)':<12} | {'Range (W)':<12} | {'CV (%)':<10}")
    print("-" * 90)
    
    for season_key, result in all_results.items():
        label = season_labels[season_key]
        powers = [data['power_watts'] for data in result['minutes_data']]
        
        mean_p = np.mean(powers)
        std_p = np.std(powers)
        min_p = np.min(powers)
        max_p = np.max(powers)
        range_p = max_p - min_p
        cv = (std_p / mean_p * 100) if mean_p > 0 else 0
        
        print(f"{label:<10} | {mean_p:>14.1f} | {std_p:>12.1f} | {min_p:>10.1f} | {max_p:>10.1f} | {range_p:>10.1f} | {cv:>8.1f}%")
    
    print("=" * 90)
    
    # ============================================================================
    # PLOTTING
    # ============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Minute-Level Solar Power Within an Hour ({harvester.location})', 
                 fontsize=16, fontweight='bold')
    
    season_colors = {
        'summer': '#f1c40f',
        'autumn': '#e67e22',
        'winter': '#3498db',
        'spring': '#2ecc71'
    }
    
    minutes = np.arange(60)
    
    # Plot 1: Power vs Minute (All Seasons)
    ax = axes[0, 0]
    for season_key, result in all_results.items():
        powers = [data['power_watts'] for data in result['minutes_data']]
        ax.plot(minutes, powers, '-', label=season_labels[season_key],
                color=season_colors[season_key], linewidth=1.5)
    ax.set_xlabel('Minute')
    ax.set_ylabel('Power Output (W)')
    ax.set_title('Minute-Level Power Fluctuations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Clean vs Noisy (Summer)
    ax = axes[0, 1]
    summer_clean = [data['base_power_watts'] for data in all_results['summer']['minutes_data']]
    summer_noisy = [data['power_watts'] for data in all_results['summer']['minutes_data']]
    ax.plot(minutes, summer_clean, 'g-', label='Clean (No Noise)', linewidth=2)
    ax.plot(minutes, summer_noisy, 'b-', label='Noisy', linewidth=1, alpha=0.7)
    ax.fill_between(minutes, summer_clean, summer_noisy, alpha=0.3, color='gray')
    ax.set_xlabel('Minute')
    ax.set_ylabel('Power Output (W)')
    ax.set_title('Clean vs Noisy - Summer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy per Minute
    ax = axes[1, 0]
    for season_key, result in all_results.items():
        energies = [data['energy_kj'] for data in result['minutes_data']]
        ax.plot(minutes, energies, '-', label=season_labels[season_key],
                color=season_colors[season_key], linewidth=1.5)
    ax.set_xlabel('Minute')
    ax.set_ylabel('Energy per Minute (kJ)')
    ax.set_title('Energy Generation per Minute')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Energy within the Hour
    ax = axes[1, 1]
    for season_key, result in all_results.items():
        cumulative = np.cumsum([data['energy_kj'] for data in result['minutes_data']])
        ax.plot(minutes, cumulative, '-', label=season_labels[season_key],
                color=season_colors[season_key], linewidth=2)
    ax.set_xlabel('Minute')
    ax.set_ylabel('Cumulative Energy (kJ)')
    ax.set_title('Cumulative Energy Within the Hour')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    