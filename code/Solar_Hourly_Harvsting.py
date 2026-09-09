"""
SOLAR ENERGY HARVESTING - HOURLY GENERATION BY SEASON
For Underwater THz Communication Investigation

This model shows hourly solar energy fluctuations throughout a single day
for each season (Summer, Autumn, Winter, Spring).

The solar irradiance follows a sine curve peaking at solar noon,
with seasonal efficiency variations.
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
        self.seasonal_efficiency = {
            'summer': 0.90,      # 10% loss due to high temperatures
            'autumn': 0.97,      # 3% loss due to moderate temperatures
            'winter': 1.02,      # 2% gain due to cooler temperatures
            'spring': 0.95       # 5% loss due to moderate warming
        }
        
        # Seasonal mapping with sunrise/sunset
        self.seasons = {
            'summer': {'months': [11, 12, 1, 2], 'label': 'Summer', 
                       'sunrise': 5, 'sunset': 19, 'temp': 30, 'eff_factor': 0.90},
            'autumn': {'months': [3, 4, 5], 'label': 'Autumn',
                       'sunrise': 6, 'sunset': 18, 'temp': 22, 'eff_factor': 0.97},
            'winter': {'months': [6, 7, 8], 'label': 'Winter',
                       'sunrise': 7, 'sunset': 17, 'temp': 15, 'eff_factor': 1.02},
            'spring': {'months': [9, 10], 'label': 'Spring',
                       'sunrise': 6, 'sunset': 18, 'temp': 25, 'eff_factor': 0.95}
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
        
        Parameters:
        - hour: Hour of the day (0-23)
        - season: 'summer', 'autumn', 'winter', 'spring'
        - sunrise: Hour of sunrise (default 6:00 AM)
        - sunset: Hour of sunset (default 6:00 PM)
        
        Returns:
        - Irradiance in kW/m²
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
    
    def harvest_hourly(self, season, sunrise=6, sunset=18):
        """
        Calculate hourly energy harvest for a given season.
        
        Returns:
        - Dictionary with hourly data and summary
        """
        hourly_data = []
        total_energy_kj = 0
        peak_power_w = 0
        peak_hour = 0
        
        eff = self.get_efficiency(season)
        
        for hour in range(24):
            irradiance = self.irradiance_at_hour(hour, season, sunrise, sunset)
            
            # Power output at this hour (Watts)
            power_w = irradiance * self.panel_area * eff * 1000
            
            # Energy in this hour (kJ)
            energy_kj = power_w * 3600 / 1000
            
            hourly_data.append({
                'hour': hour,
                'irradiance': irradiance,
                'power_watts': power_w,
                'energy_kj': energy_kj
            })
            
            total_energy_kj += energy_kj
            if power_w > peak_power_w:
                peak_power_w = power_w
                peak_hour = hour
        
        return {
            'hourly_data': hourly_data,
            'total_energy_kj': total_energy_kj,
            'peak_power_w': peak_power_w,
            'peak_hour': peak_hour,
            'season': season,
            'efficiency': eff,
            'sunrise': sunrise,
            'sunset': sunset
        }


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
    
    print("=" * 100)
    print("SOLAR ENERGY HARVESTING - HOURLY GENERATION BY SEASON")
    print(f"Location: {harvester.location}")
    print(f"Panel Area: {harvester.panel_area} m²")
    print(f"Base Panel Efficiency: {harvester.base_efficiency * 100:.0f}%")
    print("=" * 100)
    
    # Define seasons
    seasons = {
        'summer': {'label': 'Summer'},
        'autumn': {'label': 'Autumn'},
        'winter': {'label': 'Winter'},
        'spring': {'label': 'Spring'}
    }
    
    # Store all results
    all_results = {}
    
    # Print seasonal efficiencies
    print("\nSEASONAL EFFICIENCIES:")
    print("-" * 80)
    print(f"{'Season':<10} | {'Efficiency Factor':<18} | {'Effective Efficiency':<22} | {'Sunrise':<12} | {'Sunset':<12} | {'Daylight (hrs)':<15}")
    print("-" * 80)
    
    for season_key, season_info in harvester.seasons.items():
        eff_factor = season_info['eff_factor']
        eff = harvester.base_efficiency * eff_factor
        sunrise = season_info['sunrise']
        sunset = season_info['sunset']
        daylight = sunset - sunrise
        print(f"{season_info['label']:<10} | {eff_factor:>16.2f}x | {eff*100:>20.1f}% | {sunrise:>10}:00 | {sunset:>9}:00 | {daylight:>13}h")
    
    print("=" * 80)
    
    # Collect hourly data for each season
    print("\nHOURLY ENERGY GENERATION (kJ per hour):")
    print("-" * 90)
    print(f"{'Hour':<8} | {'Summer (kJ)':<14} | {'Autumn (kJ)':<14} | {'Winter (kJ)':<14} | {'Spring (kJ)':<14}")
    print("-" * 90)
    
    # First, harvest for each season and store results
    for season_key in seasons.keys():
        season_info = harvester.seasons[season_key]
        result = harvester.harvest_hourly(
            season_key,
            sunrise=season_info['sunrise'],
            sunset=season_info['sunset']
        )
        all_results[season_key] = result
    
    # Print hourly table
    for hour in range(24):
        summer_energy = 0
        autumn_energy = 0
        winter_energy = 0
        spring_energy = 0
        
        for data in all_results['summer']['hourly_data']:
            if data['hour'] == hour:
                summer_energy = data['energy_kj']
        for data in all_results['autumn']['hourly_data']:
            if data['hour'] == hour:
                autumn_energy = data['energy_kj']
        for data in all_results['winter']['hourly_data']:
            if data['hour'] == hour:
                winter_energy = data['energy_kj']
        for data in all_results['spring']['hourly_data']:
            if data['hour'] == hour:
                spring_energy = data['energy_kj']
        
        # Format with commas for thousands
        summer_str = f"{summer_energy:>10.2f}" if summer_energy > 0 else "    0.00"
        autumn_str = f"{autumn_energy:>10.2f}" if autumn_energy > 0 else "    0.00"
        winter_str = f"{winter_energy:>10.2f}" if winter_energy > 0 else "    0.00"
        spring_str = f"{spring_energy:>10.2f}" if spring_energy > 0 else "    0.00"
        
        # Only print hours with any generation
        if summer_energy > 0 or autumn_energy > 0 or winter_energy > 0 or spring_energy > 0:
            print(f"{hour:>2}:00 - {hour+1}:00 | {summer_str} | {autumn_str} | {winter_str} | {spring_str}")
    
    print("=" * 90)
    
    # ============================================================================
    # SEASONAL SUMMARIES
    # ============================================================================
    
    print("\nSEASONAL SUMMARIES:")
    print("-" * 90)
    print(f"{'Season':<10} | {'Efficiency':<12} | {'Total (kJ)':<14} | {'Total (kWh)':<14} | {'Peak Power (W)':<16} | {'Peak Hour':<12} | {'Daylight (hrs)':<13}")
    print("-" * 90)
    
    for season_key, result in all_results.items():
        label = harvester.seasons[season_key]['label']
        eff = result['efficiency'] * 100
        total_kj = result['total_energy_kj']
        peak_w = result['peak_power_w']
        peak_hour = result['peak_hour']
        daylight = result['sunset'] - result['sunrise']
        
        print(f"{label:<10} | {eff:>10.1f}% | {total_kj:>12.1f} | {total_kj/3600:>12.3f} | {peak_w:>14.1f} | {peak_hour:>2}:00 - {peak_hour+1}:00 | {daylight:>11}h")
    
    print("=" * 90)
    
    # ============================================================================
    # HOURLY POWER OUTPUT TABLE
    # ============================================================================
    
    print("\nHOURLY POWER OUTPUT (Watts):")
    print("-" * 90)
    print(f"{'Hour':<8} | {'Summer (W)':<14} | {'Autumn (W)':<14} | {'Winter (W)':<14} | {'Spring (W)':<14}")
    print("-" * 90)
    
    for hour in range(24):
        summer_power = 0
        autumn_power = 0
        winter_power = 0
        spring_power = 0
        
        for data in all_results['summer']['hourly_data']:
            if data['hour'] == hour:
                summer_power = data['power_watts']
        for data in all_results['autumn']['hourly_data']:
            if data['hour'] == hour:
                autumn_power = data['power_watts']
        for data in all_results['winter']['hourly_data']:
            if data['hour'] == hour:
                winter_power = data['power_watts']
        for data in all_results['spring']['hourly_data']:
            if data['hour'] == hour:
                spring_power = data['power_watts']
        
        summer_str = f"{summer_power:>10.1f}" if summer_power > 0 else "    0.0"
        autumn_str = f"{autumn_power:>10.1f}" if autumn_power > 0 else "    0.0"
        winter_str = f"{winter_power:>10.1f}" if winter_power > 0 else "    0.0"
        spring_str = f"{spring_power:>10.1f}" if spring_power > 0 else "    0.0"
        
        if summer_power > 0 or autumn_power > 0 or winter_power > 0 or spring_power > 0:
            print(f"{hour:>2}:00 - {hour+1}:00 | {summer_str} | {autumn_str} | {winter_str} | {spring_str}")
    
    print("=" * 90)
    
    # ============================================================================
    # COMPARISON TABLE
    # ============================================================================
    
    print("\nCOMPARISON TABLE:")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Summer':<12} | {'Autumn':<12} | {'Winter':<12} | {'Spring':<12}")
    print("-" * 80)
    
    summer = all_results['summer']
    autumn = all_results['autumn']
    winter = all_results['winter']
    spring = all_results['spring']
    
    # Calculate peak hours
    summer_peak_hour = summer['peak_hour']
    autumn_peak_hour = autumn['peak_hour']
    winter_peak_hour = winter['peak_hour']
    spring_peak_hour = spring['peak_hour']
    
    # Calculate energy in first 6 hours, middle 6 hours, last 6 hours
    def get_energy_in_window(result, start_hour, end_hour):
        total = 0
        for data in result['hourly_data']:
            if start_hour <= data['hour'] < end_hour:
                total += data['energy_kj']
        return total
    
    summer_morning = get_energy_in_window(summer, 6, 12)
    summer_afternoon = get_energy_in_window(summer, 12, 18)
    autumn_morning = get_energy_in_window(autumn, 6, 12)
    autumn_afternoon = get_energy_in_window(autumn, 12, 18)
    winter_morning = get_energy_in_window(winter, 6, 12)
    winter_afternoon = get_energy_in_window(winter, 12, 18)
    spring_morning = get_energy_in_window(spring, 6, 12)
    spring_afternoon = get_energy_in_window(spring, 12, 18)
    
    print(f"{'Total Energy (kJ)':<25} | {summer['total_energy_kj']:>10.1f} | {autumn['total_energy_kj']:>10.1f} | {winter['total_energy_kj']:>10.1f} | {spring['total_energy_kj']:>10.1f}")
    print(f"{'Peak Power (W)':<25} | {summer['peak_power_w']:>10.1f} | {autumn['peak_power_w']:>10.1f} | {winter['peak_power_w']:>10.1f} | {spring['peak_power_w']:>10.1f}")
    print(f"{'Peak Hour':<25} | {summer_peak_hour:>2}:00 - {summer_peak_hour+1}:00 | {autumn_peak_hour:>2}:00 - {autumn_peak_hour+1}:00 | {winter_peak_hour:>2}:00 - {winter_peak_hour+1}:00 | {spring_peak_hour:>2}:00 - {spring_peak_hour+1}:00")
    print(f"{'Daylight Hours':<25} | {summer['sunset'] - summer['sunrise']:>11}h | {autumn['sunset'] - autumn['sunrise']:>11}h | {winter['sunset'] - winter['sunrise']:>11}h | {spring['sunset'] - spring['sunrise']:>11}h")
    print(f"{'Morning Energy (6-12h) kJ':<25} | {summer_morning:>10.1f} | {autumn_morning:>10.1f} | {winter_morning:>10.1f} | {spring_morning:>10.1f}")
    print(f"{'Afternoon Energy (12-18h) kJ':<25} | {summer_afternoon:>10.1f} | {autumn_afternoon:>10.1f} | {winter_afternoon:>10.1f} | {spring_afternoon:>10.1f}")
    print(f"{'Morning/Afternoon Ratio':<25} | {summer_morning/summer_afternoon:>10.2f} | {autumn_morning/autumn_afternoon:>10.2f} | {winter_morning/winter_afternoon:>10.2f} | {spring_morning/spring_afternoon:>10.2f}")
    
    print("=" * 80)
    
    # ============================================================================
    # PLOTTING
    # ============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Solar Energy Harvesting - Hourly Generation by Season ({harvester.location})', 
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
    
    # Plot 1: Energy vs Hour (All Seasons)
    ax = axes[0, 0]
    for season_key, result in all_results.items():
        energies = [data['energy_kj'] for data in result['hourly_data']]
        ax.bar(hours, energies, alpha=0.7, label=f"{season_labels[season_key]} ({result['efficiency']*100:.1f}%)", 
               color=season_colors[season_key], width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy (kJ)')
    ax.set_title('Hourly Energy Generation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    
    # Plot 2: Power vs Hour (All Seasons)
    ax = axes[0, 1]
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
    # EFFECTIVE SUNLIGHT HOURS
    # ============================================================================
    
    print("\nEFFECTIVE SUNLIGHT HOURS ANALYSIS (Power > 10% of Peak):")
    print("-" * 80)
    
    for season_key, result in all_results.items():
        label = harvester.seasons[season_key]['label']
        peak = result['peak_power_w']
        threshold = peak * 0.10
        effective_hours = 0
        total_energy = 0
        hours_list = []
        
        for data in result['hourly_data']:
            if data['power_watts'] > threshold:
                effective_hours += 1
                total_energy += data['energy_kj']
                hours_list.append(data['hour'])
        
        first_hour = hours_list[0] if hours_list else 0
        last_hour = hours_list[-1] if hours_list else 0
        
        print(f"{label}:")
        print(f"  Effective hours: {effective_hours} hours ({first_hour}:00 - {last_hour+1}:00)")
        print(f"  Energy in effective hours: {total_energy:.1f} kJ ({total_energy/result['total_energy_kj']*100:.1f}% of total)")
        print(f"  Threshold: {threshold:.1f} W (10% of peak {peak:.1f} W)")
    
    print("=" * 80)