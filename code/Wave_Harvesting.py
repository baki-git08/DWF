"""
WAVE ENERGY GENERATION - MONTHLY AND ANNUAL ANALYSIS
For Underwater THz Communication Investigation

This module calculates wave energy generation across all months and seasons
for multiple WEC technologies. Output is in kJ/day and kJ/year.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WECParameters:
    """Parameters for a Wave Energy Converter."""
    name: str
    cut_in: float          # Cut-in wave height (m)
    rated_height: float    # Rated wave height (m)
    rated_power: float     # Rated power (W)
    cut_out: float         # Cut-out wave height (m)
    description: str

@dataclass
class MonthlyResult:
    """Monthly energy generation result."""
    month: str
    season: str
    wave_height: float
    power_watts: float
    energy_kj: float
    energy_kwh: float

# ============================================================================
# WAVE ENERGY GENERATOR
# ============================================================================

class WaveEnergyGenerator:
    """
    Wave energy generation calculator for multiple WEC technologies.
    Focuses on monthly and annual generation only.
    """
    
    # South African monthly wave data (significant wave height in meters)
    MONTHLY_WAVE_DATA = {
        'Jan': 1.8, 'Feb': 1.7, 'Mar': 1.9, 'Apr': 2.1,
        'May': 2.3, 'Jun': 2.6, 'Jul': 2.7, 'Aug': 2.5,
        'Sep': 2.2, 'Oct': 1.8, 'Nov': 1.5, 'Dec': 1.4
    }

    # Number of days used to convert representative daily energy
    # into monthly and annual energy.
    DAYS_IN_MONTH = {
        'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30,
        'May': 31, 'Jun': 30, 'Jul': 31, 'Aug': 31,
        'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31
    }
    
    # Season mapping
    SEASON_MAP = {
        'Jan': 'Summer', 'Feb': 'Summer', 'Mar': 'Autumn',
        'Apr': 'Autumn', 'May': 'Autumn', 'Jun': 'Winter',
        'Jul': 'Winter', 'Aug': 'Winter', 'Sep': 'Spring',
        'Oct': 'Spring', 'Nov': 'Summer', 'Dec': 'Summer'
    }
    
    # Season colours for plotting
    SEASON_COLORS = {
        'Summer': '#f1c40f',
        'Autumn': '#e67e22',
        'Winter': '#3498db',
        'Spring': '#2ecc71'
    }
    
    # WEC technologies
    WEC_TECHNOLOGIES = {
        'pendulum': WECParameters(
            name='Pendulum',
            cut_in=0.2,
            rated_height=0.6,
            rated_power=5.0,
            cut_out=2.0,
            description='Simple pendulum-based WEC for small buoys'
        ),
        'point_absorber': WECParameters(
            name='Point Absorber',
            cut_in=0.5,
            rated_height=1.5,
            rated_power=6.8,
            cut_out=3.5,
            description='Linear generator point absorber, most common type'
        ),
        'gyroscopic': WECParameters(
            name='Gyroscopic',
            cut_in=0.3,
            rated_height=1.0,
            rated_power=3.0,
            cut_out=3.0,
            description='Gyroscopic precession WEC (ISWEC type)'
        ),
        'hybrid': WECParameters(
            name='Hybrid',
            cut_in=0.2,
            rated_height=1.2,
            rated_power=10.0,
            cut_out=4.0,
            description='Hybrid electromagnetic-piezoelectric WEC'
        ),
        'tigerray': WECParameters(
            name='TigerRAY',
            cut_in=0.5,
            rated_height=1.8,
            rated_power=55.0,
            cut_out=4.0,
            description='High-power WEC for AUV docking applications'
        )
    }
    
    def __init__(self, wec_type: str = 'point_absorber', buoy_diameter: float = 1.0):
        """
        Initialize the wave energy generator.
        
        Parameters:
        - wec_type: Type of WEC ('pendulum', 'point_absorber', 'gyroscopic', 'hybrid', 'tigerray')
        - buoy_diameter: Diameter of the buoy in meters
        """
        if wec_type not in self.WEC_TECHNOLOGIES:
            raise ValueError(f"Unknown WEC type: {wec_type}. Available: {list(self.WEC_TECHNOLOGIES.keys())}")
        
        self.wec_type = wec_type
        self.params = self.WEC_TECHNOLOGIES[wec_type]
        self.buoy_diameter = buoy_diameter
        
        self.months = list(self.MONTHLY_WAVE_DATA.keys())
    
    # ========================================================================
    # POWER CALCULATION
    # ========================================================================
    
    def power_output(self, wave_height: float) -> float:
        """
        Calculate electrical power output for a given wave height.
        
        Power curve regions:
        1. Cut-in: P = 0 (wave too small)
        2. Rated region: P = P_rated * (H/H_rated)² (quadratic scaling)
        3. Curtailment: P = P_rated (active power limiting)
        4. Cut-out: P = 0 (safety shutdown)
        """
        p = self.params
        
        # Region 1: Below cut-in
        if wave_height < p.cut_in:
            return 0.0
        
        # Region 4: Above cut-out (safety shutdown)
        if wave_height > p.cut_out:
            return 0.0
        
        # Region 2: Rated region - quadratic scaling with wave height
        if wave_height <= p.rated_height:
            return p.rated_power * (wave_height / p.rated_height)**2
        
        # Region 3: Power curtailment - constant rated power
        return p.rated_power
    
    def energy_per_day(self, wave_height: float) -> float:
        """
        Calculate energy generated per day in kJ.
        """
        power = self.power_output(wave_height)
        return power * 24 * 3600 / 1000
    
    # ========================================================================
    # MONTHLY GENERATION
    # ========================================================================
    
    def generate_month(self, month: str) -> MonthlyResult:
        """
        Calculate energy generation for a specific month.
        """
        wave_height = self.MONTHLY_WAVE_DATA[month]
        season = self.SEASON_MAP[month]
        power = self.power_output(wave_height)
        energy_kj = self.energy_per_day(wave_height)
        energy_kwh = energy_kj / 3600
        
        return MonthlyResult(
            month=month,
            season=season,
            wave_height=wave_height,
            power_watts=power,
            energy_kj=energy_kj,
            energy_kwh=energy_kwh
        )
    
    def generate_year(self) -> List[MonthlyResult]:
        """
        Calculate energy generation for all 12 months.
        """
        return [self.generate_month(month) for month in self.months]
    
    # ========================================================================
    # COMPARISON METHODS
    # ========================================================================
    
    @classmethod
    def compare_all_technologies(cls) -> Dict:
        """
        Compare all WEC technologies across all months.
        Returns a dictionary with technology names as keys and monthly results as values.
        """
        results = {}
        
        for tech_name in cls.WEC_TECHNOLOGIES.keys():
            generator = cls(wec_type=tech_name)
            monthly = generator.generate_year()
            # Correct annual integration: daily energy × days in each month.
            annual_energy = sum(
                r.energy_kj * cls.DAYS_IN_MONTH[r.month] for r in monthly
            )

            # Time-weighted average electrical power over the 365-day year.
            avg_power = annual_energy * 1000 / (365 * 24 * 3600)
            
            results[tech_name] = {
                'name': cls.WEC_TECHNOLOGIES[tech_name].name,
                'monthly': monthly,
                'monthly_energy': [r.energy_kj for r in monthly],
                'monthly_power': [r.power_watts for r in monthly],
                'annual_energy_kj': annual_energy,
                'annual_energy_mj': annual_energy / 1000,
                'annual_energy_kwh': annual_energy / 3600,
                'avg_power_w': avg_power,
                'min_energy': min(r.energy_kj for r in monthly),
                'max_energy': max(r.energy_kj for r in monthly)
            }
        
        return results
    
    @classmethod
    def compare_by_month(cls, month: str) -> Dict:
        """
        Compare all WEC technologies for a specific month.
        """
        results = {}
        
        for tech_name in cls.WEC_TECHNOLOGIES.keys():
            generator = cls(wec_type=tech_name)
            result = generator.generate_month(month)
            results[tech_name] = {
                'name': generator.params.name,
                'power_watts': result.power_watts,
                'energy_kj': result.energy_kj,
                'energy_kwh': result.energy_kwh,
                'wave_height': result.wave_height
            }
        
        return results
    
    # ========================================================================
    # PRINTING METHODS
    # ========================================================================
    
    def print_monthly(self) -> None:
        """
        Print monthly generation results for the selected technology.
        """
        results = self.generate_year()
        annual_energy = sum(
            r.energy_kj * self.DAYS_IN_MONTH[r.month] for r in results
        )
        
        print("=" * 80)
        print(f"WAVE ENERGY GENERATION - {self.params.name}")
        print("=" * 80)
        print(f"Cut-in: {self.params.cut_in} m, Rated: {self.params.rated_height} m, Cut-out: {self.params.cut_out} m")
        print("=" * 80)
        
        print(f"{'Month':<6} | {'Season':<8} | {'Wave (m)':<10} | {'Power (W)':<12} | {'Energy (kJ)':<14} | {'Energy (kWh)':<12}")
        print("-" * 80)
        
        for r in results:
            print(f"{r.month:<6} | {r.season:<8} | {r.wave_height:>8.2f} | {r.power_watts:>10.2f} | {r.energy_kj:>12.1f} | {r.energy_kwh:>10.3f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<6} | {'':<8} | {'':<10} | {'':<12} | {annual_energy:>12.1f} | {annual_energy/3600:>10.2f}")
        print("=" * 80)
        print(f"\nANNUAL SUMMARY:")
        print(f"  Total: {annual_energy:.1f} kJ/year")
        print(f"        = {annual_energy/1000:.2f} MJ/year")
        print(f"        = {annual_energy/3600:.2f} kWh/year")
        print(f"  Average Power: {np.mean([r.power_watts for r in results]):.2f} W")
        print(f"  Daily Energy Range: {min(r.energy_kj for r in results):.1f} - {max(r.energy_kj for r in results):.1f} kJ/day")
        print("=" * 80)
    
    @classmethod
    def print_all_technologies(cls) -> None:
        """
        Print a comparison of all WEC technologies.
        """
        results = cls.compare_all_technologies()
        
        print("=" * 90)
        print("WAVE ENERGY GENERATION - ALL TECHNOLOGIES")
        print("=" * 90)
        
        # Monthly header
        months = list(cls.MONTHLY_WAVE_DATA.keys())
        print(f"{'Technology':<15} | " + " | ".join([f"{m:>6}" for m in months]) + " | {'Total (kJ)':>12} | {'Total (MJ)':>12} | {'Avg (W)':>10}")
        print("-" * 90)
        
        for tech_name, data in results.items():
            monthly_energy = data['monthly_energy']
            energy_str = " | ".join([f"{e:>6.1f}" for e in monthly_energy])
            print(f"{data['name']:<15} | {energy_str} | {data['annual_energy_kj']:>10.1f} | {data['annual_energy_mj']:>10.2f} | {data['avg_power_w']:>8.2f}")
        
        print("=" * 90)
        
        # Print seasonal summaries
        print("\nSEASONAL AVERAGES (kJ/day):")
        print("-" * 60)
        
        seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
        season_months = {
            'Summer': ['Jan', 'Feb', 'Nov', 'Dec'],
            'Autumn': ['Mar', 'Apr', 'May'],
            'Winter': ['Jun', 'Jul', 'Aug'],
            'Spring': ['Sep', 'Oct']
        }
        
        print(f"{'Technology':<15} | " + " | ".join([f"{s:>8}" for s in seasons]))
        print("-" * 60)
        
        for tech_name, data in results.items():
            season_vals = []
            for season in seasons:
                months_in_season = season_months[season]
                avg = np.mean([data['monthly_energy'][list(cls.MONTHLY_WAVE_DATA.keys()).index(m)] 
                              for m in months_in_season if m in data['monthly_energy']])
                # Recalculate properly
                energies = []
                for m in months_in_season:
                    idx = list(cls.MONTHLY_WAVE_DATA.keys()).index(m)
                    energies.append(data['monthly_energy'][idx])
                season_vals.append(np.mean(energies))
            
            print(f"{data['name']:<15} | " + " | ".join([f"{v:>8.1f}" for v in season_vals]))
        
        print("=" * 60)
    
    @classmethod
    def print_month_comparison(cls, month: str) -> None:
        """
        Print a comparison of all technologies for a specific month.
        """
        results = cls.compare_by_month(month)
        
        print("=" * 60)
        print(f"WEC TECHNOLOGY COMPARISON - {month}")
        print(f"Wave Height: {cls.MONTHLY_WAVE_DATA[month]:.1f} m")
        print("=" * 60)
        
        print(f"{'Technology':<18} | {'Power (W)':<12} | {'Energy (kJ)':<14} | {'Energy (kWh)':<12}")
        print("-" * 60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['energy_kj'], reverse=True)
        
        for tech_name, data in sorted_results:
            print(f"{data['name']:<18} | {data['power_watts']:>10.2f} | {data['energy_kj']:>12.1f} | {data['energy_kwh']:>10.3f}")
        
        print("=" * 60)
    
    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================
    
    def plot_results(self) -> None:
        """
        Plot monthly generation results for the selected technology.
        """
        results = self.generate_year()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Wave Energy Generation - {self.params.name}", fontsize=16, fontweight='bold')
        
        months = [r.month for r in results]
        energies = [r.energy_kj for r in results]
        powers = [r.power_watts for r in results]
        wave_heights = [r.wave_height for r in results]
        
        # Colours by season
        colors = [self.SEASON_COLORS.get(r.season, 'gray') for r in results]
        
        # Plot 1: Monthly energy
        ax = axes[0, 0]
        bars = ax.bar(months, energies, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=np.mean(energies), color='red', linestyle='--', 
                  label=f'Average: {np.mean(energies):.1f} kJ/day')
        ax.set_xlabel('Month')
        ax.set_ylabel('Energy (kJ/day)')
        ax.set_title('Monthly Wave Energy Generation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, energy in zip(bars, energies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{energy:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Power and wave height
        ax = axes[0, 1]
        ax2 = ax.twinx()
        
        ax.bar(months, powers, color='#3498db', alpha=0.7, label='Power (W)')
        ax2.plot(months, wave_heights, 'r-o', linewidth=2, label='Wave Height (m)')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Power (W)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylabel('Wave Height (m)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title('Power and Wave Height')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Plot 3: Energy by season
        ax = axes[1, 0]
        seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
        season_energies = []
        for season in seasons:
            season_results = [r for r in results if r.season == season]
            season_energies.append(np.mean([r.energy_kj for r in season_results]))
        
        season_colors = [self.SEASON_COLORS[s] for s in seasons]
        ax.bar(seasons, season_energies, color=season_colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Season')
        ax.set_ylabel('Average Energy (kJ/day)')
        ax.set_title('Seasonal Energy Generation')
        ax.grid(True, alpha=0.3)
        
        for i, val in enumerate(season_energies):
            ax.text(i, val + 5, f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Power curve
        ax = axes[1, 1]
        wave_heights_range = np.linspace(0, 4.5, 100)
        power_curve = [self.power_output(h) for h in wave_heights_range]
        
        ax.plot(wave_heights_range, power_curve, 'b-', linewidth=2.5)
        ax.axvline(x=self.params.cut_in, color='gray', linestyle='--', label='Cut-in')
        ax.axvline(x=self.params.rated_height, color='green', linestyle='--', label='Rated')
        ax.axvline(x=self.params.cut_out, color='red', linestyle='--', label='Cut-out')
        ax.axhline(y=self.params.rated_power, color='orange', linestyle=':', alpha=0.7, label='Rated Power')
        
        # Mark actual operating points
        actual_heights = list(set(r.wave_height for r in results))
        actual_powers = [self.power_output(h) for h in actual_heights]
        ax.scatter(actual_heights, actual_powers, color='red', s=80, zorder=5, 
                  label='Operating Points')
        
        # Add month labels to operating points
        for r in results:
            ax.annotate(r.month, (r.wave_height, self.power_output(r.wave_height)),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Wave Height (m)')
        ax.set_ylabel('Power Output (W)')
        ax.set_title('Power Curve with Operating Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @classmethod
    def plot_comparison(cls) -> None:
        """
        Plot a comparison of all WEC technologies.
        """
        results = cls.compare_all_technologies()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Wave Energy Generation - All Technologies", fontsize=16, fontweight='bold')
        
        months = list(cls.MONTHLY_WAVE_DATA.keys())
        tech_names = list(results.keys())
        tech_labels = [results[t]['name'] for t in tech_names]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
        
        # Plot 1: Monthly energy (grouped bar chart)
        ax = axes[0, 0]
        x = np.arange(len(months))
        width = 0.15
        
        for i, (tech_name, data) in enumerate(results.items()):
            offset = (i - len(tech_names)/2 + 0.5) * width
            ax.bar(x + offset, data['monthly_energy'], width, 
                  label=data['name'], color=colors[i % len(colors)])
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Energy (kJ/day)')
        ax.set_title('Monthly Energy Generation by Technology')
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Annual energy (bar chart)
        ax = axes[0, 1]
        annual_energies = [results[t]['annual_energy_kj'] for t in tech_names]
        bars = ax.bar(tech_labels, annual_energies, color=colors[:len(tech_labels)])
        ax.set_xlabel('Technology')
        ax.set_ylabel('Annual Energy (kJ/year)')
        ax.set_title('Annual Energy Generation by Technology')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, annual_energies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{val/1000:.2f} MJ', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Average power
        ax = axes[1, 0]
        avg_powers = [results[t]['avg_power_w'] for t in tech_names]
        bars = ax.bar(tech_labels, avg_powers, color=colors[:len(tech_labels)])
        ax.set_xlabel('Technology')
        ax.set_ylabel('Average Power (W)')
        ax.set_title('Average Power by Technology')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, avg_powers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f} W', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Minimum and maximum
        ax = axes[1, 1]
        x = np.arange(len(tech_labels))
        mins = [results[t]['min_energy'] for t in tech_names]
        maxs = [results[t]['max_energy'] for t in tech_names]
        
        for i, (tech_name, data) in enumerate(results.items()):
            ax.plot([i, i], [data['min_energy'], data['max_energy']], 
                   'o-', color=colors[i % len(colors)], markersize=8)
        
        ax.set_xlabel('Technology')
        ax.set_ylabel('Energy (kJ/day)')
        ax.set_title('Energy Range by Technology')
        ax.set_xticks(x)
        ax.set_xticklabels(tech_labels)
        ax.legend(['Min-Max Range' for _ in tech_names])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# MODEL LIMITATIONS / INTERPRETATION
# ============================================================================
#
# This is a simplified screening model. It uses one representative significant
# wave height per month and an assumed quadratic WEC power curve. It does not
# model wave period, wave spectrum, PTO/generator efficiency, electrical losses,
# availability/downtime, or a manufacturer's power matrix.
#
# Therefore, the numerical results should be presented as estimates based on
# the stated assumptions, not as experimentally validated WEC output.
#
# A higher-fidelity model should use measured/reanalysis Hs-Te data together
# with the selected WEC's manufacturer power matrix or a validated
# hydrodynamic/PTO model.
#
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 90)
    print("WAVE ENERGY GENERATION - MONTHLY AND ANNUAL ANALYSIS")
    print("For Underwater THz Communication Investigation")
    print("=" * 90)
    
    # ========================================================================
    # 1. SINGLE TECHNOLOGY - MONTHLY GENERATION
    # ========================================================================
    
    print("\n[1] Single Technology - Monthly Generation")
    print("-" * 80)
    
    generator = WaveEnergyGenerator(wec_type='point_absorber')
    generator.print_monthly()
    generator.plot_results()
    
    # ========================================================================
    # 2. ALL TECHNOLOGIES - MONTHLY COMPARISON
    # ========================================================================
    
    print("\n[2] All Technologies - Monthly Comparison")
    print("-" * 80)
    
    WaveEnergyGenerator.print_all_technologies()
    WaveEnergyGenerator.plot_comparison()
    
    # ========================================================================
    # 3. SPECIFIC MONTH COMPARISON
    # ========================================================================
    
    print("\n[3] Specific Month Comparison")
    print("-" * 80)
    
    for month in ['Jan', 'Jul', 'Nov']:
        WaveEnergyGenerator.print_month_comparison(month)
    
    # ========================================================================
    # 4. SUMMARY TABLE FOR REPORT
    # ========================================================================
    
    print("\n[4] Summary Table for Report")
    print("-" * 80)
    
    results = WaveEnergyGenerator.compare_all_technologies()
    
    print("\nSUMMARY TABLE (kJ/day):")
    print("=" * 100)
    print(f"{'Technology':<15} | {'Summer':>8} | {'Autumn':>8} | {'Winter':>8} | {'Spring':>8} | {'Annual (kJ)':>12} | {'Annual (MJ)':>10}")
    print("-" * 100)
    
    seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
    season_months = {
        'Summer': ['Jan', 'Feb', 'Nov', 'Dec'],
        'Autumn': ['Mar', 'Apr', 'May'],
        'Winter': ['Jun', 'Jul', 'Aug'],
        'Spring': ['Sep', 'Oct']
    }
    
    months_list = list(WaveEnergyGenerator.MONTHLY_WAVE_DATA.keys())
    
    for tech_name, data in results.items():
        season_vals = []
        for season in seasons:
            energies = []
            for m in season_months[season]:
                idx = months_list.index(m)
                energies.append(data['monthly_energy'][idx])
            season_vals.append(np.mean(energies))
        
        print(f"{data['name']:<15} | {season_vals[0]:>8.1f} | {season_vals[1]:>8.1f} | {season_vals[2]:>8.1f} | {season_vals[3]:>8.1f} | {data['annual_energy_kj']:>12.1f} | {data['annual_energy_mj']:>10.2f}")
    
    print("=" * 100)
    
    # ========================================================================
    # 5. THZ COMMUNICATION REQUIREMENT ANALYSIS
    # ========================================================================
    
    print("\n[5] THz Communication Requirement Analysis")
    print("-" * 80)
    
    # From your design: THz burst uses 11.7 J = 0.0117 kJ
    THZ_BURST_ENERGY_KJ = 0.0117
    
    print(f"THz burst energy: {THZ_BURST_ENERGY_KJ} kJ ({THZ_BURST_ENERGY_KJ*1000:.1f} J)")
    print()
    
    for tech_name, data in results.items():
        # Calculate number of bursts per day for each month
        bursts_per_month = [e / THZ_BURST_ENERGY_KJ for e in data['monthly_energy']]
        
        print(f"{data['name']}:")
        print(f"  Worst month: {min(bursts_per_month):,.0f} bursts/day")
        print(f"  Best month: {max(bursts_per_month):,.0f} bursts/day")
        print(f"  Annual average: {sum(bursts_per_month)/12:,.0f} bursts/day")
        print()
    
    print("=" * 80)
    print("CONCLUSION: The model estimates THz burst capability from the")
    print("assumed monthly wave heights and WEC power curves.")
    print("Actual WEC performance must be validated using measured wave spectra,")
    print("device power matrices, conversion efficiency, and system losses.")
    print("=" * 80)