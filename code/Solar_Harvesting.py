import numpy as np
import matplotlib.pyplot as plt

class SolarHarvester:
    """
    Solar energy harvesting model for Master node on water surface.
    """
    
    def __init__(self, panel_area=1.0, panel_efficiency=0.20, location='Johannesburg'):
        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency
        
        # Location data - surface irradiance (kWh/m²/day) [5†L13-L15][6†L3-L9]
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
        
        # Seasonal mapping
        self.seasons = {
            'summer': {'months': [11, 12, 1, 2], 'label': 'Summer'},
            'autumn': {'months': [3, 4, 5], 'label': 'Autumn'},
            'winter': {'months': [6, 7, 8], 'label': 'Winter'},
            'spring': {'months': [9, 10], 'label': 'Spring'}
        }
    
    def get_season(self, month):
        for season, data in self.seasons.items():
            if month in data['months']:
                return season
        return 'summer'
    
    def harvest_energy(self, month, duration_hours=24):
        """
        Calculate harvestable energy at surface.
        """
        season = self.get_season(month)
        I_surface = self.location_data[season]  # kWh/m²/day
        
        if duration_hours >= 24:
            eff_hours = self.location_data['peak_hours']
        else:
            eff_hours = self.location_data['peak_hours'] * (duration_hours / 24)
        
        # Energy harvested (kWh)
        energy_kwh = I_surface * self.panel_area * self.panel_efficiency * (eff_hours / 24)
        
        # Convert to Joules
        energy_joules = energy_kwh * 3.6e6
        
        return {
            'season': season,
            'surface_irradiance': I_surface,
            'eff_hours': eff_hours,
            'energy_kwh': energy_kwh,
            'energy_joules': energy_joules,
            'power_watts': energy_joules / (eff_hours * 3600) if eff_hours > 0 else 0
        }
    
    def harvest_over_year(self):
        """Calculate monthly harvested energy over a full year."""
        results = []
        for month in range(1, 13):
            result = self.harvest_energy(month)
            results.append({
                'month': month,
                'season': result['season'],
                'energy_joules': result['energy_joules'],
                'power_watts': result['power_watts'],
                'surface_irradiance': result['surface_irradiance']
            })
        return results
    
    def compare_panel_sizes(self, month):
        """Compare energy for different panel sizes."""
        sizes = [0.5, 1.0, 1.5, 2.0]  # m²
        results = {}
        for area in sizes:
            harvester = SolarHarvester(panel_area=area, location='Johannesburg')
            result = harvester.harvest_energy(month)
            results[area] = {
                'energy_joules': result['energy_joules'],
                'power_watts': result['power_watts']
            }
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    harvester = SolarHarvester(panel_area=1.0, panel_efficiency=0.20, location='Johannesburg')
    
    print("=" * 60)
    print("SOLAR ENERGY HARVESTING - MASTER ON WATER SURFACE")
    print("=" * 60)
    
    # 1. Seasonal comparison
    print("\n1. SEASONAL ENERGY (Johannesburg, 1m² panel, 20% efficiency)")
    print("-" * 60)
    print(f"{'Season':<10} | {'Irradiance':<12} | {'Energy':<12} | {'Power':<10}")
    print("-" * 60)
    
    seasons = {'summer': 1, 'autumn': 4, 'winter': 7, 'spring': 10}
    for season, month in seasons.items():
        result = harvester.harvest_energy(month)
        print(f"{season.capitalize():<10} | {result['surface_irradiance']:>6.2f} kWh/m²/day | "
              f"{result['energy_joules']/1000:>8.1f} kJ/day | "
              f"{result['power_watts']:>6.1f} W")
    
    # 2. Location comparison (Winter)
    print("\n2. LOCATION COMPARISON (Winter, 1m² panel, 20% efficiency)")
    print("-" * 60)
    
    for loc in ['Upington', 'Kimberley', 'Johannesburg', 'Pretoria', 'Cape Town', 'Durban']:
        h = SolarHarvester(panel_area=1.0, panel_efficiency=0.20, location=loc)
        result = h.harvest_energy(month=7)
        print(f"{loc:<12} | {result['energy_joules']/1000:>6.1f} kJ/day | {result['power_watts']:>5.1f} W")
    
    # 3. Panel size comparison
    print("\n3. PANEL SIZE COMPARISON (Johannesburg, Summer)")
    print("-" * 60)
    
    sizes = harvester.compare_panel_sizes(month=1)
    for area, data in sizes.items():
        print(f"{area:>4.1f} m² | {data['energy_joules']/1000:>6.1f} kJ/day | {data['power_watts']:>5.1f} W")
    
    # 4. Yearly profile
    print("\n4. YEARLY ENERGY PROFILE (Johannesburg, 1m² panel)")
    print("-" * 60)
    
    yearly = harvester.harvest_over_year()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, r in enumerate(yearly):
        print(f"{months[i]:4} | {r['season']:8} | {r['energy_joules']/1000:6.1f} kJ/day | {r['power_watts']:5.1f} W")
    
    # 5. Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Seasonal comparison
    ax = axes[0, 0]
    seasons_list = ['Summer', 'Autumn', 'Winter', 'Spring']
    energies = []
    for season, month in seasons.items():
        result = harvester.harvest_energy(month)
        energies.append(result['energy_joules'] / 1000)
    ax.bar(seasons_list, energies, color=['gold', 'orange', 'lightblue', 'lightgreen'])
    ax.set_xlabel('Season')
    ax.set_ylabel('Energy (kJ/day)')
    ax.set_title('Seasonal Energy Harvest (Surface, Johannesburg)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Location comparison (Winter)
    ax = axes[0, 1]
    locs = ['Upington', 'Kimberley', 'Johannesburg', 'Pretoria', 'Cape Town', 'Durban']
    vals = []
    for loc in locs:
        h = SolarHarvester(panel_area=1.0, panel_efficiency=0.20, location=loc)
        result = h.harvest_energy(month=7)
        vals.append(result['energy_joules'] / 1000)
    ax.barh(locs, vals, color=['#2ecc71' if v > 120 else '#f1c40f' if v > 90 else '#e74c3c' for v in vals])
    ax.set_xlabel('Energy (kJ/day)')
    ax.set_title('Location Comparison (Winter, Surface)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Panel size comparison
    ax = axes[1, 0]
    sizes = [0.5, 1.0, 1.5, 2.0]
    energies_size = []
    for area in sizes:
        h = SolarHarvester(panel_area=area, panel_efficiency=0.20, location='Johannesburg')
        result = h.harvest_energy(month=1)
        energies_size.append(result['energy_joules'] / 1000)
    ax.bar([f"{s:.1f}m²" for s in sizes], energies_size, color='skyblue')
    ax.set_xlabel('Panel Area')
    ax.set_ylabel('Energy (kJ/day)')
    ax.set_title('Panel Size vs Energy (Summer)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Yearly profile
    ax = axes[1, 1]
    monthly_energy = [r['energy_joules'] / 1000 for r in yearly]
    ax.bar(months, monthly_energy, color='skyblue')
    ax.axhline(y=np.mean(monthly_energy), color='red', linestyle='--', label=f"Average: {np.mean(monthly_energy):.1f} kJ/day")
    ax.set_xlabel('Month')
    ax.set_ylabel('Energy (kJ/day)')
    ax.set_title('Monthly Energy Profile (Surface)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()