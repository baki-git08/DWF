import numpy as np
import matplotlib.pyplot as plt


# PART 1 — SYSTEM DOCUMENTATION

def print_system_documentation():
    """Answer the 9 design questions from the project notes."""

    print("=" * 110)
    print("SOLAR ENERGY HARVESTING MODEL — MASTER NODE")
    print("Optical-to-Electrical Energy Conversion")
    print("=" * 110)

    print("""
1. HOW DOES SOLAR HARVEST ENERGY FROM THE ENVIRONMENT?
   ─────────────────────────────────────────────────────
   Sunlight (photons) strikes the solar panel's photovoltaic (PV) cells.
   The photons excite electrons across the semiconductor P-N junction,
   creating a direct current (DC). This is the photovoltaic effect.
   The panel converts optical energy (sunlight) into electrical energy (DC).

2. WHAT AREA OF SOLAR PANEL DO WE NEED?
   ──────────────────────────────────────
   Master node panel: 5 m × 5 m = 25 m² (as per design sketch).
   Larger area → more harvested power. Power scales linearly with area.

3. WHAT COMPONENTS DO WE NEED?
   ─────────────────────────────
   • Solar Panel (5 m × 5 m array of PV cells)
   • Charge Controller (regulates DC voltage, protects battery)
   • Battery Cell 1 (energy storage)
   • Inverter (DC → AC for system components)
   • AC-to-AC Converter (conditions AC voltage)
   • System Components (microprocessor, sensors, THz transceiver)

4. WHAT HOURS OF THE DAY CAN WE HARVEST?
   ─────────────────────────────────────
   Only during daylight hours (sunrise → sunset). Zero harvesting at night.
   Daylight window depends on season:
     • Summer: 05:00 – 19:00  (14 hours)
     • Autumn: 06:00 – 18:00  (12 hours)
     • Winter: 07:00 – 17:00  (10 hours)
     • Spring: 06:00 – 18:00  (12 hours)

5. PEAK / MODERATE / LOW HOURS BY SEASON
   ─────────────────────────────────────
              Peak (noon)   Moderate       Low
     Summer      12:00        09:00 & 15:00   06:00 & 18:00
     Autumn      12:00        09:00 & 15:00   07:00 & 17:00
     Winter      12:00        10:00 & 14:00   08:00 & 16:00
     Spring      12:00        09:00 & 15:00   07:00 & 17:00

6. OPTICAL-TO-ELECTRICAL EFFICIENCY BY SEASON
   ──────────────────────────────────────────
   Base panel efficiency: 20% (typical commercial PV)
   Temperature-adjusted efficiency:
     • Summer: 18.0%  (heat reduces efficiency)
     • Autumn: 19.4%
     • Winter: 20.4%  (cooler = better efficiency)
     • Spring: 19.0%

7. HOW MUCH ENERGY CAN WE HARVEST WITHIN A DAY?
   ─────────────────────────────────────────────
   Computed for a 25 m² panel at each season (see Section 3 of results).
   Total daily energy = ∫ P(t) dt over daylight hours.

8. FACTORS AFFECTING HARVESTING
   ──────────────────────────────
   Environmental:
     • Solar irradiance (latitude, season, time of day)
     • Cloud cover and weather
     • Temperature (heat reduces PV efficiency)
     • Dust and panel soiling
   System:
     • Panel area and tilt angle
     • PV cell technology (mono-Si, poly-Si, thin-film)
     • Charge controller and battery efficiency
     • Cable and conversion losses
   Underwater-specific (Master is surface buoy):
     • Biofouling on the panel
     • Salt spray and corrosion
     • Wave-induced motion (reduces effective area)

9. SIGNAL FLOW
   ────────────
   See architecture diagram below.
""")


def print_signal_flow():
    """Print the system architecture diagram (from the sketch)."""

    print("=" * 110)
    print("SIGNAL FLOW — OPTICAL-TO-ELECTRICAL CONVERSION CHAIN")
    print("=" * 110)
    print("""
                          ☀  Sunlight (Optical Input)
                          │
                          ▼
                ┌─────────────────────┐
                │   SOLAR PANEL       │  ← 5 m × 5 m (25 m²)
                │  [Optical→Electrical]│
                └─────────┬───────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ DC Voltage│
                    └─────┬─────┘
                          │
              ┌───────────┴────────────┐
              │                        │
              ▼                        ▼
    ┌────────────────────┐    ┌────────────────────┐
    │ AC-to-AC Converter │    │  Charge Controller │
    └──────────┬─────────┘    └──────────┬─────────┘
               │                         │
               │                         ▼
               │                  ┌─────────────┐
               │                  │ Battery     │
               │                  │ Cell 1      │
               │                  └──────┬──────┘
               │                         │
               │                         ▼
               │                  ┌─────────────┐
               │                  │  Inverter   │
               │                  └──────┬──────┘
               │                         │
               ▼                         ▼
    ┌─────────────────────────────────────────────┐
    │       ALL SYSTEM COMPONENTS (AC Load)       │
    │  Microprocessor · THz Transceiver · Sensors │
    └─────────────────────────────────────────────┘

    Power path:   Sunlight → Panel → DC → [Battery + Inverter] or [AC-to-AC]
    Storage path: DC → Charge Controller → Battery
    Load path:    Battery → Inverter → AC Load
""")


# PART 2 — SOLAR HARVESTER CLASS

class MinuteSolarHarvester:
    """
    Solar energy harvesting model.
    Panel: 5 m × 1 m = 5 m² as per design sketch.
    """

    # Panel dimensions from design sketch
    PANEL_WIDTH_M  = 5.0
    PANEL_HEIGHT_M = 1.0

    def __init__(self,
                 panel_width_m=PANEL_WIDTH_M,
                 panel_height_m=PANEL_HEIGHT_M,
                 base_efficiency=0.20,
                 location='Johannesburg',
                 noise_level=0.15,
                 seed=42):

        self.panel_width_m  = panel_width_m
        self.panel_height_m = panel_height_m
        self.panel_area     = panel_width_m * panel_height_m   # 25 m²

        self.base_efficiency = base_efficiency
        self.location        = location
        self.noise_level     = noise_level
        self.seed            = seed

        np.random.seed(seed)

        # Surface irradiance (kWh/m²/day) — averaged monthly per season
        self.locations = {
            'Johannesburg': {'summer': 6.38, 'autumn': 5.81, 'winter': 4.80, 'spring': 7.23},
            'Cape Town':    {'summer': 8.98, 'autumn': 5.03, 'winter': 3.56, 'spring': 7.22},
            'Durban':       {'summer': 5.50, 'autumn': 4.50, 'winter': 3.80, 'spring': 5.00},
        }
        self.location_data = self.locations.get(location, self.locations['Johannesburg'])

        # Seasonal efficiency multipliers (temperature effect on PV)
        self.seasonal_efficiency = {
            'summer': 0.90,
            'autumn': 0.97,
            'winter': 1.02,
            'spring': 0.95,
        }

        # Seasonal noise multipliers (clouds)
        self.seasonal_noise = {
            'summer': 1.0,
            'autumn': 0.8,
            'winter': 1.2,
            'spring': 0.9,
        }

        # Sunrise / sunset by season
        self.seasons = {
            'summer': {'label': 'Summer', 'sunrise': 5, 'sunset': 19},
            'autumn': {'label': 'Autumn', 'sunrise': 6, 'sunset': 18},
            'winter': {'label': 'Winter', 'sunrise': 7, 'sunset': 17},
            'spring': {'label': 'Spring', 'sunrise': 6, 'sunset': 18},
        }

    # Efficiency
    def get_efficiency(self, season):
        return self.base_efficiency * self.seasonal_efficiency.get(season, 1.0)

    # Irradiance (smooth sine curve across daylight hours)
    def irradiance_at_hour(self, hour, season, sunrise=6, sunset=18):
        total = self.location_data[season]
        daylight = sunset - sunrise
        if hour < sunrise or hour > sunset:
            return 0.0
        t = (hour - sunrise) / daylight
        return max(0, np.sin(np.pi * t) * (np.pi / 2) * (total / daylight))

    def irradiance_at_minute(self, hour, minute, season, sunrise=6, sunset=18):
        return self.irradiance_at_hour(hour + minute / 60.0, season, sunrise, sunset)

    # -------------------------------------------------------------------------
    # Minute-level noise (clouds, turbulence, shadows)
    # -------------------------------------------------------------------------
    def apply_minute_noise(self, irradiance, season, minute, prev_noise=None):
        if irradiance <= 0:
            return 0.0, 0.0

        noise_mult = self.seasonal_noise.get(season, 1.0)
        base_noise = self.noise_level * noise_mult

        cloud_factor    = np.random.beta(2, 5)
        cloud_noise     = (cloud_factor - 0.2) * 0.6
        turbulence      = np.random.normal(0, 0.06)
        shadow          = -np.random.uniform(0.1, 0.4) if np.random.random() < 0.05 else 0.0

        total = cloud_noise + turbulence + shadow
        if prev_noise is not None:
            total = 0.6 * prev_noise + 0.4 * total

        scaled = total * base_noise
        return max(0, irradiance * (1 + scaled)), scaled

    # -------------------------------------------------------------------------
    # One-hour harvest at minute resolution
    # -------------------------------------------------------------------------
    def harvest_minute_hour(self, hour, season, sunrise=6, sunset=18, add_noise=True):
        minutes_data = []
        total_kj = 0.0
        clean_kj = 0.0
        peak_w = 0.0
        peak_min = 0
        prev_noise = None
        eff = self.get_efficiency(season)

        for minute in range(60):
            base_irr = self.irradiance_at_minute(hour, minute, season, sunrise, sunset)

            if add_noise:
                irr, noise = self.apply_minute_noise(base_irr, season, minute, prev_noise)
                prev_noise = noise
            else:
                irr, noise = base_irr, 0.0

            power_w     = irr * self.panel_area * eff * 1000
            energy_kj   = power_w * 60 / 1000
            base_power  = base_irr * self.panel_area * eff * 1000
            base_energy = base_power * 60 / 1000

            minutes_data.append({
                'minute': minute, 'time': f"{hour:02d}:{minute:02d}",
                'irradiance': irr, 'base_irradiance': base_irr,
                'power_watts': power_w, 'base_power_watts': base_power,
                'energy_kj': energy_kj, 'base_energy_kj': base_energy,
                'noise_factor': noise,
            })

            total_kj += energy_kj
            clean_kj += base_energy
            if power_w > peak_w:
                peak_w = power_w
                peak_min = minute

        return {
            'hour': hour, 'season': season,
            'minutes_data': minutes_data,
            'total_energy_kj': total_kj,
            'total_clean_energy_kj': clean_kj,
            'peak_power_w': peak_w,
            'peak_minute': peak_min,
            'efficiency': eff,
        }


# =============================================================================
# PART 3 — ANALYSIS (Peak / Moderate / Low hours by season)
# =============================================================================

def classify_hours(season, sunrise, sunset):
    """Return a dict of peak / moderate / low hours for a season."""
    noon = (sunrise + sunset) // 2
    return {
        'Peak':     noon,
        'Moderate': noon - 3,
        'Low':      sunrise + 1,
    }


def run_analysis():
    """Run the full seasonal analysis."""

    harvester = MinuteSolarHarvester(
        panel_width_m=5.0,
        panel_height_m=5.0,
        base_efficiency=0.20,
        location='Johannesburg',
        noise_level=0.15,
        seed=42,
    )

    print("=" * 110)
    print("SOLAR HARVESTING — CONFIGURATION")
    print("=" * 110)
    print(f"  Panel dimensions : {harvester.panel_width_m} m × {harvester.panel_height_m} m"
          f" = {harvester.panel_area:.1f} m²")
    print(f"  Base efficiency  : {harvester.base_efficiency * 100:.0f}%")
    print(f"  Location         : {harvester.location}")
    print(f"  Noise level      : {harvester.noise_level * 100:.0f}%")

    # -------------------------------------------------------------------------
    # Seasonal efficiencies
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("OPTICAL-TO-ELECTRICAL EFFICIENCY BY SEASON")
    print("=" * 110)
    print(f"{'Season':<10} | {'Multiplier':<12} | {'Effective Efficiency':<22}")
    print("-" * 110)
    for s in ['summer', 'autumn', 'winter', 'spring']:
        mult = harvester.seasonal_efficiency[s]
        eff  = harvester.get_efficiency(s)
        print(f"{s.capitalize():<10} | {mult:>10.2f}x | {eff * 100:>18.1f}%")

    # -------------------------------------------------------------------------
    # Harvest each season at Peak / Moderate / Low hours
    # -------------------------------------------------------------------------
    all_results = {}   # {season: {category: result}}

    for season_key in ['summer', 'autumn', 'winter', 'spring']:
        info = harvester.seasons[season_key]
        hours = classify_hours(season_key, info['sunrise'], info['sunset'])
        all_results[season_key] = {}

        for category, hour in hours.items():
            result = harvester.harvest_minute_hour(
                hour, season_key,
                sunrise=info['sunrise'],
                sunset=info['sunset'],
                add_noise=True,
            )
            all_results[season_key][category] = result

    # -------------------------------------------------------------------------
    # SECTION 1 — Minute-level power by hour category
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SECTION 1: MINUTE-LEVEL POWER BY HOUR CATEGORY (W)")
    print("=" * 110)

    season_labels = {'summer': 'Summer', 'autumn': 'Autumn',
                     'winter': 'Winter', 'spring': 'Spring'}

    for season_key in ['summer', 'autumn', 'winter', 'spring']:
        print(f"\n{'─' * 110}")
        print(f"SEASON: {season_labels[season_key]}  "
              f"(sunrise {harvester.seasons[season_key]['sunrise']}:00, "
              f"sunset {harvester.seasons[season_key]['sunset']}:00)")
        print(f"{'─' * 110}")
        print(f"{'Time':<8} | {'Peak (W)':<14} | {'Moderate (W)':<14} | {'Low (W)':<14}")
        print("-" * 110)

        for minute in range(0, 60, 10):
            p = all_results[season_key]['Peak']['minutes_data'][minute]['power_watts']
            m = all_results[season_key]['Moderate']['minutes_data'][minute]['power_watts']
            l = all_results[season_key]['Low']['minutes_data'][minute]['power_watts']
            time_str = f"min {minute:02d}"
            print(f"{time_str:<8} | {p:>12.1f} | {m:>12.1f} | {l:>12.1f}")

    # -------------------------------------------------------------------------
    # SECTION 2 — Summary table: Peak / Moderate / Low by season
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SECTION 2: HOURLY ENERGY SUMMARY — PEAK vs MODERATE vs LOW")
    print("=" * 110)
    print(f"{'Season':<10} | {'Category':<10} | {'Hour':<6} | {'Efficiency':<11} | "
          f"{'Total (kJ)':<12} | {'Peak (W)':<10} | {'Peak min':<8}")
    print("-" * 110)

    for season_key in ['summer', 'autumn', 'winter', 'spring']:
        for category in ['Peak', 'Moderate', 'Low']:
            r = all_results[season_key][category]
            hour = r['hour']
            eff  = r['efficiency'] * 100
            kj   = r['total_energy_kj']
            pw   = r['peak_power_w']
            pm   = r['peak_minute']
            print(f"{season_labels[season_key]:<10} | {category:<10} | "
                  f"{hour:>2}:00 | {eff:>9.1f}% | {kj:>10.2f} | {pw:>8.1f} | {pm:>6}")
        print("-" * 110)

    # -------------------------------------------------------------------------
    # SECTION 3 — Daily energy total by season
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SECTION 3: DAILY ENERGY HARVEST (integrating across all daylight hours)")
    print("=" * 110)
    print(f"{'Season':<10} | {'Daylight':<10} | {'Total (kJ)':<14} | {'Total (kWh)':<14} | {'Total (MJ)':<12}")
    print("-" * 110)

    daily_totals = {}
    for season_key in ['summer', 'autumn', 'winter', 'spring']:
        info = harvester.seasons[season_key]
        total_kj = 0.0
        for hour in range(info['sunrise'], info['sunset'] + 1):
            res = harvester.harvest_minute_hour(hour, season_key,
                                                sunrise=info['sunrise'],
                                                sunset=info['sunset'])
            total_kj += res['total_energy_kj']

        daily_totals[season_key] = total_kj
        daylight = info['sunset'] - info['sunrise']
        print(f"{season_labels[season_key]:<10} | {daylight:>6} hrs | "
              f"{total_kj:>12.1f} | {total_kj/3600:>12.4f} | {total_kj/1000:>10.3f}")

    # -------------------------------------------------------------------------
    # SECTION 4 — Category ratios (per season)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SECTION 4: CATEGORY RATIOS (relative to Peak)")
    print("=" * 110)
    print(f"{'Season':<10} | {'Peak (kJ)':<12} | {'Moderate (kJ)':<15} | "
          f"{'Low (kJ)':<12} | {'Mod/Peak':<10} | {'Low/Peak':<10}")
    print("-" * 110)

    for season_key in ['summer', 'autumn', 'winter', 'spring']:
        pk = all_results[season_key]['Peak']['total_energy_kj']
        md = all_results[season_key]['Moderate']['total_energy_kj']
        lw = all_results[season_key]['Low']['total_energy_kj']
        r_md = md / pk if pk > 0 else 0
        r_lw = lw / pk if pk > 0 else 0
        print(f"{season_labels[season_key]:<10} | {pk:>10.2f} | {md:>13.2f} | "
              f"{lw:>10.2f} | {r_md:>8.2f} | {r_lw:>8.2f}")

    # -------------------------------------------------------------------------
    # SECTION 6 — Plotting
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Solar Energy Harvesting — Master Node (5 m² Panel)",
                 fontsize=16, fontweight='bold')

    colors = {'summer': '#f1c40f', 'autumn': '#e67e22',
              'winter': '#3498db', 'spring': '#2ecc71'}
    minutes = np.arange(60)

    # Plot 1 — Peak hour, all seasons
    ax = axes[0, 0]
    for s in ['summer', 'autumn', 'winter', 'spring']:
        powers = [d['power_watts'] for d in all_results[s]['Peak']['minutes_data']]
        ax.plot(minutes, powers, '-', label=season_labels[s],
                color=colors[s], linewidth=1.5)
    ax.set_title('Peak Hour — All Seasons')
    ax.set_xlabel('Minute'); ax.set_ylabel('Power (W)')
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 2 — Moderate hour, all seasons
    ax = axes[0, 1]
    for s in ['summer', 'autumn', 'winter', 'spring']:
        powers = [d['power_watts'] for d in all_results[s]['Moderate']['minutes_data']]
        ax.plot(minutes, powers, '-', label=season_labels[s],
                color=colors[s], linewidth=1.5)
    ax.set_title('Moderate Hour — All Seasons')
    ax.set_xlabel('Minute'); ax.set_ylabel('Power (W)')
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 3 — Low hour, all seasons
    ax = axes[1, 0]
    for s in ['summer', 'autumn', 'winter', 'spring']:
        powers = [d['power_watts'] for d in all_results[s]['Low']['minutes_data']]
        ax.plot(minutes, powers, '-', label=season_labels[s],
                color=colors[s], linewidth=1.5)
    ax.set_title('Low Hour — All Seasons')
    ax.set_xlabel('Minute'); ax.set_ylabel('Power (W)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_system_documentation()
    print_signal_flow()
    run_analysis()