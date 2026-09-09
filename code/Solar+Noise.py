"""
SOLAR ENERGY HARVESTING - HOURLY GENERATION BY SEASON WITH NOISE
For Underwater THz Communication Investigation

This model shows hourly solar energy fluctuations throughout a single day
for each season (Summer, Autumn, Winter, Spring) with realistic noise.

Noise sources:
- Cloud cover: Random fluctuations in irradiance (short-term)
- Atmospheric conditions: Variation in clearness index
- Turbidity: Particulate matter and pollution effects
- Seasonal weather patterns: More clouds in certain seasons
"""

import numpy as np
import matplotlib.pyplot as plt


class SolarHarvester:
    """
    Solar energy harvesting model with daily fluctuations, seasonal efficiency,
    and realistic noise.
    """
    
    def __init__(self, panel_area=5.0, base_efficiency=0.20, location='Johannesburg',
                 noise_level=0.15, seed=42):
        """
        Parameters:
        - panel_area: Solar panel area in m²
        - base_efficiency: Base panel efficiency (0.20 = 20%)
        - location: City name for irradiance data
        - noise_level: Standard deviation of noise relative to irradiance (0.15 = 15%)
        - seed: Random seed for reproducibility
        """
        self.panel_area = panel_area
        self.base_efficiency = base_efficiency
        self.location = location
        self.noise_level = noise_level
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Initialize noise persistence storage
        self._last_noise = {}
        self._cloud_states = {}
        
        # Location data - surface irradiance (kWh/m²/day)
        self.locations = {
            'Upington': {'annual': 2400, 'peak_hours': 6.6,
                         'summer': 8.0, 'autumn': 6.5, 'winter': 5.5, 'spring': 7.5},
            'Kimberley': {'annual': 2200, 'peak_hours': 6.0,
                          'summer': 7.5, 'autumn': 6.0, 'winter': 5.0, 'spring': 7.0},
            'Johannesburg': {'annual': 1950, 'peak_hours': 5.3,
                             'summer': 6.38, 'autumn': 5.81, 'winter': 4.80, 'spring': 7.23},
            'Pretoria': {'annual': 1900, 'peak_hours': 5.2,
                         'summer': 6.2, 'autumn': 5.6, 'winter': 4.7, 'spring': 7.0},
            'Cape Town': {'annual': 1750, 'peak_hours': 4.8,
                          'summer': 8.98, 'autumn': 5.03, 'winter': 3.56, 'spring': 7.22},
            'Durban': {'annual': 1650, 'peak_hours': 4.5,
                       'summer': 5.5, 'autumn': 4.5, 'winter': 3.8, 'spring': 5.0}
        }
        
        self.location_data = self.locations.get(location, self.locations['Johannesburg'])
        
        # Seasonal efficiency factors
        self.seasonal_efficiency = {
            'summer': 0.90,      # 10% loss due to high temperatures
            'autumn': 0.97,      # 3% loss due to moderate temperatures
            'winter': 1.02,      # 2% gain due to cooler temperatures
            'spring': 0.95       # 5% loss due to moderate warming
        }
        
        # Seasonal noise multipliers (more clouds in certain seasons)
        self.seasonal_noise = {
            'summer': 1.0,       # Summer: moderate cloud cover
            'autumn': 0.8,       # Autumn: clearer skies
            'winter': 1.2,       # Winter: more clouds
            'spring': 0.9        # Spring: moderate cloud cover
        }
        
        # Seasonal mapping with sunrise/sunset
        self.seasons = {
            'summer': {'months': [11, 12, 1, 2], 'label': 'Summer', 
                       'sunrise': 5, 'sunset': 19, 'temp': 30, 'eff_factor': 0.90,
                       'noise_factor': 1.0},
            'autumn': {'months': [3, 4, 5], 'label': 'Autumn',
                       'sunrise': 6, 'sunset': 18, 'temp': 22, 'eff_factor': 0.97,
                       'noise_factor': 0.8},
            'winter': {'months': [6, 7, 8], 'label': 'Winter',
                       'sunrise': 7, 'sunset': 17, 'temp': 15, 'eff_factor': 1.02,
                       'noise_factor': 1.2},
            'spring': {'months': [9, 10], 'label': 'Spring',
                       'sunrise': 6, 'sunset': 18, 'temp': 25, 'eff_factor': 0.95,
                       'noise_factor': 0.9}
        }
    
    def get_efficiency(self, season):
        """Get the effective efficiency for a given season."""
        eff_factor = self.seasonal_efficiency.get(season, 1.0)
        return self.base_efficiency * eff_factor
    
    def get_daily_irradiance(self, season):
        """Get total daily irradiance for a given season."""
        return self.location_data[season]  # kWh/m²/day
    
    def irradiance_at_hour(self, hour, season, sunrise=6, sunset=18):
        """
        Calculate solar irradiance at a specific hour using a sine curve.
        No noise applied here - this is the base irradiance.
        """
        total_irradiance = self.get_daily_irradiance(season)
        daylight_hours = sunset - sunrise
        
        if hour < sunrise or hour > sunset:
            return 0.0
        
        t = (hour - sunrise) / daylight_hours
        normalized_irradiance = np.sin(np.pi * t)
        scaling_factor = (np.pi / 2) * (total_irradiance / daylight_hours)
        irradiance = normalized_irradiance * scaling_factor
        
        return max(0, irradiance)
    
    def apply_noise(self, irradiance, season, hour):
        """
        Apply realistic noise to irradiance.
        
        Noise sources:
        1. Cloud cover: Beta distribution with persistence
        2. Atmospheric conditions: Gaussian noise
        3. Seasonal variation: More noise in winter
        """
        if irradiance <= 0:
            return 0.0
        
        # Get seasonal noise multiplier
        noise_mult = self.seasonal_noise.get(season, 1.0)
        
        # Base noise level (percentage of irradiance)
        base_noise = self.noise_level * noise_mult
        
        # Generate cloud noise using beta distribution (skewed towards clear skies)
        # Beta(2, 5) gives most values near 0 (clear) with occasional clouds
        cloud_factor = np.random.beta(2, 5)
        cloud_noise = (cloud_factor - 0.2) * 0.6  # Range: -0.12 to 0.48
        
        # Gaussian atmospheric noise
        atm_noise = np.random.normal(0, 0.08)
        
        # Combined noise factor
        total_noise_factor = cloud_noise + atm_noise
        
        # Scale by noise level
        scaled_noise = total_noise_factor * base_noise
        
        # Add persistence: blend with previous noise
        if season in self._last_noise:
            persistence = 0.7  # 70% persistence from previous hour
            scaled_noise = persistence * self._last_noise[season] + (1 - persistence) * scaled_noise
        
        self._last_noise[season] = scaled_noise
        
        # Apply noise
        noisy_irradiance = irradiance * (1 + scaled_noise)
        
        # Ensure non-negative
        return max(0, noisy_irradiance)
    
    def harvest_hourly(self, season, sunrise=6, sunset=18, add_noise=True):
        """
        Calculate hourly energy harvest for a given season.
        
        Parameters:
        - season: 'summer', 'autumn', 'winter', 'spring'
        - sunrise: Hour of sunrise
        - sunset: Hour of sunset
        - add_noise: Whether to add noise to the irradiance
        
        Returns:
        - Dictionary with hourly data and summary
        """
        hourly_data = []
        total_energy_kj = 0
        total_noise_energy_kj = 0
        peak_power_w = 0
        peak_hour = 0
        
        eff = self.get_efficiency(season)
        
        # Reset noise persistence for this season
        if add_noise:
            if season in self._last_noise:
                del self._last_noise[season]
        
        for hour in range(24):
            # Base irradiance (smooth)
            base_irradiance = self.irradiance_at_hour(hour, season, sunrise, sunset)
            
            # Apply noise if requested
            if add_noise:
                irradiance = self.apply_noise(base_irradiance, season, hour)
            else:
                irradiance = base_irradiance
            
            # Power output at this hour (Watts)
            power_w = irradiance * self.panel_area * eff * 1000
            
            # Energy in this hour (kJ)
            energy_kj = power_w * 3600 / 1000
            
            # Calculate base energy (without noise) for comparison
            base_power_w = base_irradiance * self.panel_area * eff * 1000
            base_energy_kj = base_power_w * 3600 / 1000
            
            hourly_data.append({
                'hour': hour,
                'irradiance': irradiance,
                'base_irradiance': base_irradiance,
                'power_watts': power_w,
                'base_power_watts': base_power_w,
                'energy_kj': energy_kj,
                'base_energy_kj': base_energy_kj,
                'noise_factor': (irradiance / base_irradiance - 1) if base_irradiance > 0 else 0
            })
            
            total_energy_kj += energy_kj
            total_noise_energy_kj += base_energy_kj
            if power_w > peak_power_w:
                peak_power_w = power_w
                peak_hour = hour
        
        return {
            'hourly_data': hourly_data,
            'total_energy_kj': total_energy_kj,
            'total_noise_energy_kj': total_noise_energy_kj,
            'peak_power_w': peak_power_w,
            'peak_hour': peak_hour,
            'season': season,
            'efficiency': eff,
            'sunrise': sunrise,
            'sunset': sunset,
            'has_noise': add_noise
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Create harvester with 5 m² panel and noise
    harvester = SolarHarvester(
        panel_area=5.0, 
        base_efficiency=0.20, 
        location='Johannesburg',
        noise_level=0.15,  # 15% noise relative to irradiance
        seed=42
    )
    
    print("=" * 100)
    print("SOLAR ENERGY HARVESTING - HOURLY GENERATION BY SEASON WITH NOISE")
    print(f"Location: {harvester.location}")
    print(f"Panel Area: {harvester.panel_area} m²")
    print(f"Base Panel Efficiency: {harvester.base_efficiency * 100:.0f}%")
    print(f"Noise Level: {harvester.noise_level * 100:.0f}% (relative to irradiance)")
    print("=" * 100)
    
    # Define seasons
    seasons = {
        'summer': {'label': 'Summer'},
        'autumn': {'label': 'Autumn'},
        'winter': {'label': 'Winter'},
        'spring': {'label': 'Spring'}
    }
    
    # Store results with noise and without noise for comparison
    all_results_noisy = {}
    all_results_clean = {}
    
    # Print seasonal efficiencies
    print("\nSEASONAL EFFICIENCIES:")
    print("-" * 90)
    print(f"{'Season':<10} | {'Efficiency Factor':<18} | {'Effective Efficiency':<22} | {'Noise Factor':<15} | {'Sunrise':<10} | {'Sunset':<10} | {'Daylight (hrs)':<13}")
    print("-" * 90)
    
    for season_key, season_info in harvester.seasons.items():
        eff_factor = season_info['eff_factor']
        eff = harvester.base_efficiency * eff_factor
        noise_factor = season_info['noise_factor']
        sunrise = season_info['sunrise']
        sunset = season_info['sunset']
        daylight = sunset - sunrise
        print(f"{season_info['label']:<10} | {eff_factor:>16.2f}x | {eff*100:>20.1f}% | {noise_factor:>13.2f}x | {sunrise:>8}:00 | {sunset:>7}:00 | {daylight:>11}h")
    
    print("=" * 90)
    
    # Harvest for each season (with and without noise)
    for season_key in seasons.keys():
        season_info = harvester.seasons[season_key]
        # With noise
        result_noisy = harvester.harvest_hourly(
            season_key,
            sunrise=season_info['sunrise'],
            sunset=season_info['sunset'],
            add_noise=True
        )
        all_results_noisy[season_key] = result_noisy
        
        # Without noise (clean)
        result_clean = harvester.harvest_hourly(
            season_key,
            sunrise=season_info['sunrise'],
            sunset=season_info['sunset'],
            add_noise=False
        )
        all_results_clean[season_key] = result_clean
    
    # ============================================================================
    # COMPARE NOISY VS CLEAN ENERGY
    # ============================================================================
    
    print("\nCOMPARISON: CLEAN VS NOISY ENERGY:")
    print("-" * 80)
    print(f"{'Season':<10} | {'Clean (kJ)':<15} | {'Noisy (kJ)':<15} | {'Difference (kJ)':<18} | {'Difference (%)':<15}")
    print("-" * 80)
    
    for season_key in seasons.keys():
        label = harvester.seasons[season_key]['label']
        clean = all_results_clean[season_key]['total_energy_kj']
        noisy = all_results_noisy[season_key]['total_energy_kj']
        diff = noisy - clean
        diff_pct = (diff / clean) * 100 if clean > 0 else 0
        print(f"{label:<10} | {clean:>13.1f} | {noisy:>13.1f} | {diff:>16.1f} | {diff_pct:>13.1f}%")
    
    print("=" * 80)
    
    # ============================================================================
    # HOURLY ENERGY WITH NOISE
    # ============================================================================
    
    print("\nHOURLY ENERGY GENERATION WITH NOISE (kJ per hour):")
    print("-" * 100)
    print(f"{'Hour':<10} | {'Summer (kJ)':<14} | {'Autumn (kJ)':<14} | {'Winter (kJ)':<14} | {'Spring (kJ)':<14} | {'Cloud Factor':<14}")
    print("-" * 100)
    
    for hour in range(24):
        summer_energy = 0
        autumn_energy = 0
        winter_energy = 0
        spring_energy = 0
        cloud_factor = 0
        
        for data in all_results_noisy['summer']['hourly_data']:
            if data['hour'] == hour:
                summer_energy = data['energy_kj']
                cloud_factor = data['noise_factor'] * 100
        for data in all_results_noisy['autumn']['hourly_data']:
            if data['hour'] == hour:
                autumn_energy = data['energy_kj']
        for data in all_results_noisy['winter']['hourly_data']:
            if data['hour'] == hour:
                winter_energy = data['energy_kj']
        for data in all_results_noisy['spring']['hourly_data']:
            if data['hour'] == hour:
                spring_energy = data['energy_kj']
        
        summer_str = f"{summer_energy:>10.2f}" if summer_energy > 0 else "    0.00"
        autumn_str = f"{autumn_energy:>10.2f}" if autumn_energy > 0 else "    0.00"
        winter_str = f"{winter_energy:>10.2f}" if winter_energy > 0 else "    0.00"
        spring_str = f"{spring_energy:>10.2f}" if spring_energy > 0 else "    0.00"
        
        if summer_energy > 0 or autumn_energy > 0 or winter_energy > 0 or spring_energy > 0:
            print(f"{hour:>2}:00 - {hour+1}:00 | {summer_str} | {autumn_str} | {winter_str} | {spring_str} | {cloud_factor:>11.1f}%")
    
    print("=" * 100)
    
    # ============================================================================
    # SEASONAL SUMMARIES WITH NOISE
    # ============================================================================
    
    print("\nSEASONAL SUMMARIES (With Noise):")
    print("-" * 100)
    print(f"{'Season':<10} | {'Efficiency':<12} | {'Total (kJ)':<15} | {'Total (kWh)':<15} | {'Peak Power (W)':<16} | {'Peak Hour':<12} | {'Daylight (hrs)':<13}")
    print("-" * 100)
    
    for season_key, result in all_results_noisy.items():
        label = harvester.seasons[season_key]['label']
        eff = result['efficiency'] * 100
        total_kj = result['total_energy_kj']
        peak_w = result['peak_power_w']
        peak_hour = result['peak_hour']
        daylight = result['sunset'] - result['sunrise']
        
        print(f"{label:<10} | {eff:>10.1f}% | {total_kj:>13.1f} | {total_kj/3600:>13.3f} | {peak_w:>14.1f} | {peak_hour:>2}:00 - {peak_hour+1}:00 | {daylight:>11}h")
    
    print("=" * 100)
    
    # ============================================================================
    # PLOTTING
    # ============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Solar Energy Harvesting - Hourly Generation with Noise ({harvester.location})', 
                 fontsize=16, fontweight='bold')
    
    season_colors = {
        'summer': '#f1c40f',
        'autumn': '#e67e22',
        'winter': '#3498db',
        'spring': '#2ecc71'
    }
    season_labels = {
        'summer': 'Summer',
        'autumn': 'Autumn', 
        'winter': 'Winter',
        'spring': 'Spring'
    }
    
    hours = range(24)
    
    # Plot 1: Energy vs Hour with Noise (All Seasons)
    ax = axes[0, 0]
    for season_key, result in all_results_noisy.items():
        energies = [data['energy_kj'] for data in result['hourly_data']]
        ax.bar(hours, energies, alpha=0.6, label=f"{season_labels[season_key]} ({result['efficiency']*100:.1f}%)", 
               color=season_colors[season_key], width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy (kJ)')
    ax.set_title('Hourly Energy Generation (With Noise)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 2: Clean vs Noisy (Summer example)
    ax = axes[0, 1]
    summer_clean = [data['base_energy_kj'] for data in all_results_clean['summer']['hourly_data']]
    summer_noisy = [data['energy_kj'] for data in all_results_noisy['summer']['hourly_data']]
    ax.plot(hours, summer_clean, 'g-', label='Clean (No Noise)', linewidth=2)
    ax.plot(hours, summer_noisy, 'b-', label='Noisy', linewidth=1.5)
    ax.fill_between(hours, summer_clean, summer_noisy, alpha=0.3, color='gray')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy (kJ)')
    ax.set_title('Clean vs Noisy - Summer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 3: Power vs Hour (All Seasons)
    ax = axes[1, 0]
    for season_key, result in all_results_noisy.items():
        powers = [data['power_watts'] for data in result['hourly_data']]
        ax.plot(hours, powers, 'o-', label=f"{season_labels[season_key]} ({result['efficiency']*100:.1f}%)", 
                color=season_colors[season_key], linewidth=2, markersize=4)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power Output (W)')
    ax.set_title('Power Profile with Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 4: Cumulative Energy (Summer comparison)
    ax = axes[1, 1]
    # Summer cumulative
    summer_cum_noisy = np.cumsum([data['energy_kj'] for data in all_results_noisy['summer']['hourly_data']])
    summer_cum_clean = np.cumsum([data['base_energy_kj'] for data in all_results_clean['summer']['hourly_data']])
    ax.plot(hours, summer_cum_clean, 'g-', label='Clean (No Noise)', linewidth=2)
    ax.plot(hours, summer_cum_noisy, 'b-', label='Noisy', linewidth=2)
    ax.fill_between(hours, summer_cum_clean, summer_cum_noisy, alpha=0.3, color='gray')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Cumulative Energy (kJ)')
    ax.set_title('Cumulative Energy - Summer (Clean vs Noisy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # NOISE STATISTICS
    # ============================================================================
    
    print("\nNOISE STATISTICS:")
    print("-" * 80)
    print(f"{'Season':<10} | {'Mean Noise (%)':<16} | {'Std Dev (%)':<16} | {'Max Noise (%)':<16} | {'Min Noise (%)':<16}")
    print("-" * 80)
    
    for season_key, result in all_results_noisy.items():
        label = harvester.seasons[season_key]['label']
        noise_factors = [data['noise_factor'] * 100 for data in result['hourly_data'] 
                        if data['noise_factor'] != 0]
        
        if noise_factors:
            mean_noise = np.mean(noise_factors)
            std_noise = np.std(noise_factors)
            max_noise = np.max(noise_factors)
            min_noise = np.min(noise_factors)
        else:
            mean_noise = std_noise = max_noise = min_noise = 0
        
        print(f"{label:<10} | {mean_noise:>14.1f}% | {std_noise:>14.1f}% | {max_noise:>14.1f}% | {min_noise:>14.1f}%")
    
    print("=" * 80)