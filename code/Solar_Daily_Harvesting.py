"""
SOLAR ENERGY HARVESTING - DAILY FLUCTUATION MODEL
With Seasonal Efficiency Variations

For Underwater THz Communication Investigation

This model shows solar energy fluctuations throughout a single day
for each season (Summer, Autumn, Winter, Spring).

Seasonal efficiency accounts for:
- Temperature effects: Higher temperatures reduce panel efficiency
- Sun angle: Lower sun angle in winter reduces effective efficiency
- Seasonal degradation: Panel performance varies with conditions
"""

import numpy as np
import matplotlib.pyplot as plt


class SolarHarvester:
    """
    Solar energy harvesting model with daily fluctuations and seasonal efficiency.
    """
    
    def __init__(self, panel_area=5.0, base_efficiency=0.20, location='Johannesburg'):
        self.panel_area = panel_area
        self.base_efficiency = base_efficiency
        self.location = location
        
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
        # Based on temperature and sun angle effects
        # Summer: Hot temperatures reduce efficiency by ~8-10%
        # Winter: Cooler temperatures improve efficiency by ~2-3%
        # Spring/Autumn: Moderate temperatures, near base efficiency
        self.seasonal_efficiency = {
            'summer': 0.90,      # 10% loss due to high temperatures
            'autumn': 0.97,      # 3% loss due to moderate temperatures
            'winter': 1.02,      # 2% gain due to cooler temperatures
            'spring': 0.95       # 5% loss due to moderate warming
        }
        
        # Seasonal mapping
        self.seasons = {
            'summer': {'months': [11, 12, 1, 2], 'label': 'Summer', 
                       'temp': 30, 'efficiency_factor': 0.90},
            'autumn': {'months': [3, 4, 5], 'label': 'Autumn',
                       'temp': 22, 'efficiency_factor': 0.97},
            'winter': {'months': [6, 7, 8], 'label': 'Winter',
                       'temp': 15, 'efficiency_factor': 1.02},
            'spring': {'months': [9, 10], 'label': 'Spring',
                       'temp': 25, 'efficiency_factor': 0.95}
        }
    
    def get_season(self, month):
        for season, data in self.seasons.items():
            if month in data['months']:
                return season
        return 'summer'
    
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
        
        Parameters:
        - hour: Hour of the day (0-23)
        - season: 'summer', 'autumn', 'winter', 'spring'
        - sunrise: Hour of sunrise (default 6:00 AM)
        - sunset: Hour of sunset (default 6:00 PM)
        
        Returns:
        - Irradiance in kW/m²
        """
        # Get total daily irradiance for the season
        total_irradiance = self.get_daily_irradiance(season)
        
        # Effective daylight hours (approximately 12 hours from sunrise to sunset)
        daylight_hours = sunset - sunrise
        
        # Calculate the sine curve
        # Peak at solar noon (sunrise + daylight_hours/2)
        # Use a sine function that goes from 0 at sunrise to peak at noon to 0 at sunset
        
        # Normalize hour to 0-1 range within daylight hours
        if hour < sunrise or hour > sunset:
            return 0.0
        
        # Position within daylight (0 at sunrise, 1 at sunset)
        t = (hour - sunrise) / daylight_hours
        
        # Sine curve: sin(pi * t) peaks at t=0.5 (solar noon)
        # Use a smooth sine wave
        normalized_irradiance = np.sin(np.pi * t)
        
        # Scale to match the total daily irradiance
        # The integral of sin(pi*t) from 0 to 1 = 2/pi
        # So to get the right total energy, we need to scale by pi/2
        scaling_factor = (np.pi / 2) * (total_irradiance / daylight_hours)
        
        irradiance = normalized_irradiance * scaling_factor
        
        # Ensure no negative values (shouldn't happen)
        return max(0, irradiance)
    
    def harvest_hourly(self, season, hours=24, sunrise=6, sunset=18):
        """
        Calculate hourly energy harvest for a given season.
        
        Parameters:
        - season: 'summer', 'autumn', 'winter', 'spring'
        - hours: Number of hours in the day (default 24)
        - sunrise: Hour of sunrise (default 6:00 AM)
        - sunset: Hour of sunset (default 6:00 PM)
        
        Returns:
        - Dictionary with hourly data
        """
        hourly_data = []
        total_energy_kj = 0
        peak_power_w = 0
        
        # Get seasonal efficiency
        eff = self.get_efficiency(season)
        
        for hour in range(hours):
            irradiance = self.irradiance_at_hour(hour, season, sunrise, sunset)
            
            # Power output at this hour (Watts)
            # P = Irradiance (kW/m²) * Area (m²) * Efficiency * 1000 (W/kW)
            power_w = irradiance * self.panel_area * eff * 1000
            
            # Energy in this hour (kJ)
            energy_kj = power_w * 3600 / 1000
            
            hourly_data.append({
                'hour': hour,
                'irradiance': irradiance,
                'power_watts': power_w,
                'energy_kj': energy_kj,
                'efficiency': eff
            })
            
            total_energy_kj += energy_kj
            if power_w > peak_power_w:
                peak_power_w = power_w
        
        return {
            'hourly_data': hourly_data,
            'total_energy_kj': total_energy_kj,
            'peak_power_w': peak_power_w,
            'season': season,
            'efficiency': eff
        }
    
    def harvest_season_day(self, season, sunrise=6, sunset=18):
        """
        Get a full day's harvest for a specific season.
        """
        return self.harvest_hourly(season, sunrise=sunrise, sunset=sunset)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Create harvester with 5 m² panel
    harvester = SolarHarvester(
        panel_area=5.0, 
        base_efficiency=0.20, 
        location='Johannesburg'
    )
    
    print("=" * 80)
    print("SOLAR ENERGY HARVESTING - DAILY FLUCTUATION MODEL")
    print(f"Location: {harvester.location}")
    print(f"Panel Area: {harvester.panel_area} m²")
    print(f"Base Panel Efficiency: {harvester.base_efficiency * 100:.0f}%")
    print("=" * 80)
    
    # Define seasons and their representative months
    seasons = {
        'summer': {'month': 1, 'label': 'Summer', 'sunrise': 5, 'sunset': 19},
        'autumn': {'month': 4, 'label': 'Autumn', 'sunrise': 6, 'sunset': 18},
        'winter': {'month': 7, 'label': 'Winter', 'sunrise': 7, 'sunset': 17},
        'spring': {'month': 10, 'label': 'Spring', 'sunrise': 6, 'sunset': 18}
    }
    
    # Store results for plotting
    all_results = {}
    
    # Print seasonal summary with efficiencies
    print("\nSEASONAL EFFICIENCIES:")
    print("-" * 60)
    print(f"{'Season':<10} | {'Efficiency Factor':<18} | {'Effective Efficiency':<22} | {'Temperature':<12}")
    print("-" * 60)
    
    for season_key, season_info in harvester.seasons.items():
        eff_factor = season_info['efficiency_factor']
        eff = harvester.base_efficiency * eff_factor
        temp = season_info['temp']
        print(f"{season_info['label']:<10} | {eff_factor:>16.2f}x | {eff*100:>20.1f}% | {temp:>10}°C")
    
    print("=" * 60)
    
    # Print seasonal daily summaries
    print("\nSEASONAL DAILY SUMMARY:")
    print("-" * 90)
    print(f"{'Season':<10} | {'Efficiency':<12} | {'Total Energy (kJ)':<18} | {'Total (kWh)':<14} | {'Peak Power (W)':<15} | {'Peak Hour':<10}")
    print("-" * 90)
    
    for season_key, season_info in seasons.items():
        result = harvester.harvest_season_day(
            season_key, 
            sunrise=season_info['sunrise'], 
            sunset=season_info['sunset']
        )
        all_results[season_key] = result
        
        # Find peak hour
        peak_hour = 0
        peak_power = 0
        for data in result['hourly_data']:
            if data['power_watts'] > peak_power:
                peak_power = data['power_watts']
                peak_hour = data['hour']
        
        eff = result['efficiency'] * 100
        print(f"{season_info['label']:<10} | {eff:>10.1f}% | {result['total_energy_kj']:>16.1f} | {result['total_energy_kj']/3600:>12.3f} | {result['peak_power_w']:>13.1f} | {peak_hour:>2}:00 - {peak_hour+1}:00")
    
    print("=" * 90)
    
    # Print hourly data for each season
    print("\nHOURLY ENERGY GENERATION (kJ per hour):")
    print("-" * 85)
    print(f"{'Hour':<6} | {'Summer (kJ)':<12} | {'Autumn (kJ)':<12} | {'Winter (kJ)':<12} | {'Spring (kJ)':<12} | {'Eff (Summer)':<12}")
    print("-" * 85)
    
    for hour in range(24):
        summer_energy = 0
        autumn_energy = 0
        winter_energy = 0
        spring_energy = 0
        summer_eff = 0
        
        for data in all_results['summer']['hourly_data']:
            if data['hour'] == hour:
                summer_energy = data['energy_kj']
                summer_eff = data['efficiency'] * 100
        for data in all_results['autumn']['hourly_data']:
            if data['hour'] == hour:
                autumn_energy = data['energy_kj']
        for data in all_results['winter']['hourly_data']:
            if data['hour'] == hour:
                winter_energy = data['energy_kj']
        for data in all_results['spring']['hourly_data']:
            if data['hour'] == hour:
                spring_energy = data['energy_kj']
        
        # Only print hours with any generation
        if summer_energy > 0 or autumn_energy > 0 or winter_energy > 0 or spring_energy > 0:
            print(f"{hour:>2}:00  | {summer_energy:>10.2f} | {autumn_energy:>10.2f} | {winter_energy:>10.2f} | {spring_energy:>10.2f} | {summer_eff:>10.1f}%")
    
    print("=" * 85)
    
    # ============================================================================
    # PLOTTING
    # ============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Solar Energy Harvesting - Daily Fluctuation ({harvester.location})', 
                 fontsize=16, fontweight='bold')
    
    # Colour mapping for seasons
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
    
    # Plot 1: Power vs Hour (All Seasons)
    ax = axes[0, 0]
    for season_key, result in all_results.items():
        powers = [data['power_watts'] for data in result['hourly_data']]
        ax.plot(hours, powers, 'o-', label=f"{season_labels[season_key]} ({result['efficiency']*100:.1f}%)", 
                color=season_colors[season_key], linewidth=2, markersize=6)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power Output (W)')
    ax.set_title('Power Profile Throughout the Day')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 2: Energy vs Hour (All Seasons)
    ax = axes[0, 1]
    for season_key, result in all_results.items():
        energies = [data['energy_kj'] for data in result['hourly_data']]
        ax.bar(hours, energies, alpha=0.3, label=season_labels[season_key], 
               color=season_colors[season_key], width=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy (kJ)')
    ax.set_title('Hourly Energy Generation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 3: Cumulative Energy
    ax = axes[1, 0]
    for season_key, result in all_results.items():
        cumulative = np.cumsum([data['energy_kj'] for data in result['hourly_data']])
        ax.plot(hours, cumulative, label=f"{season_labels[season_key]} ({result['efficiency']*100:.1f}%)", 
                color=season_colors[season_key], linewidth=2)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Cumulative Energy (kJ)')
    ax.set_title('Cumulative Energy Throughout the Day')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 4: Solar Irradiance
    ax = axes[1, 1]
    for season_key, result in all_results.items():
        irradiance = [data['irradiance'] for data in result['hourly_data']]
        ax.plot(hours, irradiance, label=season_labels[season_key], 
                color=season_colors[season_key], linewidth=2)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Irradiance (kW/m²)')
    ax.set_title('Solar Irradiance Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # SUMMER VS WINTER COMPARISON
    # ============================================================================
    
    print("\nSUMMER vs WINTER COMPARISON:")
    print("-" * 70)
    summer_total = all_results['summer']['total_energy_kj']
    winter_total = all_results['winter']['total_energy_kj']
    summer_peak = all_results['summer']['peak_power_w']
    winter_peak = all_results['winter']['peak_power_w']
    summer_eff = all_results['summer']['efficiency'] * 100
    winter_eff = all_results['winter']['efficiency'] * 100
    
    print(f"Summer Efficiency:       {summer_eff:.1f}%")
    print(f"Winter Efficiency:       {winter_eff:.1f}%")
    print(f"Efficiency Difference:   {summer_eff - winter_eff:.1f}% (Summer loses {abs(summer_eff - winter_eff):.1f}% due to heat)")
    print()
    print(f"Summer Total Energy:     {summer_total:.1f} kJ/day ({summer_total/3600:.3f} kWh/day)")
    print(f"Winter Total Energy:     {winter_total:.1f} kJ/day ({winter_total/3600:.3f} kWh/day)")
    print(f"Summer/Winter Ratio:     {summer_total/winter_total:.2f}x")
    print(f"Summer Peak Power:       {summer_peak:.1f} W")
    print(f"Winter Peak Power:       {winter_peak:.1f} W")
    print("-" * 70)
    
    # ============================================================================
    # COMPARISON: WITH AND WITHOUT SEASONAL EFFICIENCY
    # ============================================================================
    
    # Calculate what energy would be without seasonal efficiency
    # (Using base efficiency = 20% for all seasons)
    print("\nCOMPARISON: With vs Without Seasonal Efficiency:")
    print("-" * 70)
    print(f"{'Season':<10} | {'Irradiance (kWh)':<15} | {'With Eff (kJ)':<15} | {'No Eff (kJ)':<15} | {'Difference (%)':<15}")
    print("-" * 70)
    
    # Re-calculate without seasonal efficiency for comparison
    harvester_no_eff = SolarHarvester(
        panel_area=5.0, 
        base_efficiency=0.20, 
        location='Johannesburg'
    )
    # Override seasonal efficiency to 1.0 for all seasons
    harvester_no_eff.seasonal_efficiency = {
        'summer': 1.0,
        'autumn': 1.0,
        'winter': 1.0,
        'spring': 1.0
    }
    
    for season_key, season_info in seasons.items():
        result_with = all_results[season_key]
        result_no_eff = harvester_no_eff.harvest_season_day(
            season_key,
            sunrise=season_info['sunrise'],
            sunset=season_info['sunset']
        )
        irradiance = harvester.get_daily_irradiance(season_key)
        diff_pct = (result_with['total_energy_kj'] - result_no_eff['total_energy_kj']) / result_no_eff['total_energy_kj'] * 100
        
        print(f"{season_info['label']:<10} | {irradiance:>13.2f} | {result_with['total_energy_kj']:>13.1f} | {result_no_eff['total_energy_kj']:>13.1f} | {diff_pct:>13.1f}%")
    
    print("=" * 70)
    
    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    
    print("\nSUMMARY TABLE (5m² Panel with Seasonal Efficiency):")
    print("=" * 90)
    print(f"{'Season':<10} | {'Efficiency':<12} | {'Total (kJ)':<12} | {'Peak (W)':<12} | {'Energy (kWh)':<14} | {'% of Summer':<12}")
    print("-" * 90)
    
    for season_key, result in all_results.items():
        season_label = season_labels[season_key]
        eff = result['efficiency'] * 100
        total_kj = result['total_energy_kj']
        peak_w = result['peak_power_w']
        percent = (total_kj / all_results['summer']['total_energy_kj']) * 100 if season_key != 'summer' else 100
        
        print(f"{season_label:<10} | {eff:>10.1f}% | {total_kj:>10.1f} | {peak_w:>10.1f} | {total_kj/3600:>12.3f} | {percent:>10.1f}%")
    
    print("=" * 90)
    
    # ============================================================================
    # EFFECTIVE SUNLIGHT HOURS ANALYSIS
    # ============================================================================
    
    print("\nEFFECTIVE SUNLIGHT HOURS ANALYSIS (5m² Panel):")
    print("-" * 70)
    print("Effective hours are when power output > 10% of peak power.")
    print()
    
    for season_key, result in all_results.items():
        season_label = season_labels[season_key]
        peak = result['peak_power_w']
        effective_hours = 0
        total_energy = 0
        
        for data in result['hourly_data']:
            if data['power_watts'] > peak * 0.10:
                effective_hours += 1
                total_energy += data['energy_kj']
        
        print(f"{season_label}: {effective_hours} hours at >10% of peak, producing {total_energy:.1f} kJ ({total_energy/result['total_energy_kj']*100:.1f}% of total)")
    
    print("=" * 70)