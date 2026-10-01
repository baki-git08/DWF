"""
SOLAR ENERGY HARVESTING — SUBMERGED MASTER AT 3 m DEPTH
=========================================================

Two sea environments compared:
    • DEEP SEA   — open ocean, oligotrophic, very clear
    • COASTAL    — turbid, nutrient-rich, sediment-rich

First-principles model:
    Beer–Lambert:  I(z) = I(0) · (1−R) · exp(−k·z)
    Spectral correction:  η_spec depends on which wavelengths survive
    Salinity adjustment:  n_water = 1.34 for seawater
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASONS = ['Summer', 'Autumn', 'Winter', 'Spring']

# Surface irradiance (kWh/m²/day) for sea locations
# Durban reference for coastal; offshore reference for deep sea
SURFACE_IRRADIANCE = {
    'Coastal': {
        'Summer': 6.20,
        'Autumn': 5.50,
        'Winter': 4.60,
        'Spring': 6.80,
    },
    'Deep Sea': {
        'Summer': 7.00,
        'Autumn': 6.30,
        'Winter': 5.40,
        'Spring': 7.50,
    },
}

# Panel
PANEL_WIDTH_M   = 5.0
PANEL_HEIGHT_M  = 1.0
PANEL_AREA_M2   = PANEL_WIDTH_M * PANEL_HEIGHT_M

# Submersion depth
DEPTH_M = 3.0

# -----------------------------------------------------------------------------
# WATER OPTICAL PROPERTIES
# -----------------------------------------------------------------------------
# Attenuation coefficient k (m⁻¹) — broadband PAR, includes absorption+scattering
# Sources: Jerlov water types, Mobley (1994), Morel (1988)

WATER_PROFILES = {
    'Deep Sea': {
        'k_par':            0.04,    # Jerlov I — clearest oceanic water
        'n_water':          1.34,    # seawater refractive index
        'eta_spectral':     0.88,    # blue-green dominated, good for Si PV
        'turbidity':        'Low',
        'chlorophyll':      'Very low',
        'suspended_solids': 'Minimal',
        'description':      'Oligotrophic open ocean — Jerlov Type I',
    },
    'Coastal': {
        'k_par':            0.35,    # Jerlov 5–7 — turbid coastal
        'n_water':          1.34,
        'eta_spectral':     0.82,    # green-shifted, some Si mismatch
        'turbidity':        'High',
        'chlorophyll':      'High',
        'suspended_solids': 'Significant',
        'description':      'Turbid coastal — Jerlov Type 5–7',
    },
    # Reference: lake from previous version
    'Lake (reference)': {
        'k_par':            0.80,
        'n_water':          1.33,
        'eta_spectral':     0.85,
        'turbidity':        'Moderate',
        'chlorophyll':      'Moderate',
        'suspended_solids': 'Moderate',
        'description':      'Freshwater lake — reference',
    },
}

# Sunrise / sunset (used for hourly shape)
SUNRISE = {'Summer': 5, 'Autumn': 6, 'Winter': 7, 'Spring': 6}
SUNSET  = {'Summer': 19, 'Autumn': 18, 'Winter': 17, 'Spring': 18}

BASE_EFFICIENCY = 0.20


# =============================================================================
# FIRST-PRINCIPLES DERIVATION
# =============================================================================

def print_derivation():
    print("=" * 100)
    print("FIRST-PRINCIPLES DERIVATION — UNDERWATER SOLAR IRRADIANCE")
    print("Deep Sea vs Coastal Sea at 3 m Depth")
    print("=" * 100)
    print("""
STEP 1 — Solar emission (Stefan–Boltzmann)
    E_sun = σ·T⁴ = 5.67e-8 · (5778)⁴ ≈ 6.33 × 10⁷ W/m²

STEP 2 — Inverse-square spreading to Earth
    S₀ = E_sun · (R_sun / d_earth)² = 1361 W/m²  (solar constant)

STEP 3 — Atmospheric transmission (~30% loss)
    I_surface ≈ 1000 W/m² at clear noon

STEP 4 — Air–water interface (Fresnel)
    R = ((n_air − n_water) / (n_air + n_water))²
    For seawater n = 1.34:  R ≈ 0.0215 → 97.9% enters

STEP 5 — Underwater attenuation (Beer–Lambert)
    dI/dz = −k(λ)·I(z)
    I(z, λ) = I(0, λ) · exp(−k(λ)·z)

STEP 6 — Spectral dependence BY WATER TYPE

    Wavelength    Deep Sea        Coastal Sea      Lake
    ───────────  ──────────────   ──────────────   ──────────────
    Blue  450 nm   k ≈ 0.02 m⁻¹     k ≈ 0.15 m⁻¹    k ≈ 0.40 m⁻¹
    Green 550 nm   k ≈ 0.04 m⁻¹     k ≈ 0.35 m⁻¹    k ≈ 0.60 m⁻¹
    Red   650 nm   k ≈ 0.30 m⁻¹     k ≈ 0.60 m⁻¹    k ≈ 1.50 m⁻¹

    Broadband PAR coefficient:
        Deep Sea    k ≈ 0.04 m⁻¹  (Jerlov Type I)
        Coastal     k ≈ 0.35 m⁻¹  (Jerlov Type 5–7)

STEP 7 — Irradiance at 3 m

    DEEP SEA:
        I(3) = I_surface · 0.979 · exp(−0.04 × 3)
             = I_surface · 0.979 · 0.887
             = 0.868 · I_surface   → 87% survives

    COASTAL:
        I(3) = I_surface · 0.979 · exp(−0.35 × 3)
             = I_surface · 0.979 · 0.350
             = 0.343 · I_surface   → 34% survives

    LAKE (reference):
        I(3) = I_surface · 0.98 · exp(−0.80 × 3)
             = 0.091 · I_surface   → 9.1% survives

STEP 8 — Spectral mismatch factor
    Deep Sea    → blue-green spectrum preserved, silicon PV works well → 0.88
    Coastal     → green-red dominant, some Si mismatch → 0.82
    Lake        → green-yellow dominant → 0.85

STEP 9 — Salinity effect
    Seawater n = 1.34 → Fresnel R ≈ 2.15% (vs 2.0% for freshwater)
    Minor effect — included in Step 4

STEP 10 — Slant-path correction
    Accounts for oblique sunrise/sunset rays
    Folded into k_eff = k_par × 1.05

FINAL FORMULA
    E_daily(z) = I_surface · (1 − R_fresnel) · exp(−k_par · z)
                 · A · η_PV · η_spectral
""")


# =============================================================================
# CORE PHYSICS
# =============================================================================

def fresnel_reflection(n_water):
    """Fresnel reflection coefficient at normal incidence."""
    return ((1.0 - n_water) / (1.0 + n_water)) ** 2


def transmission_at_depth(water_type, depth_m=DEPTH_M):
    """Fraction of surface irradiance surviving to depth."""
    w = WATER_PROFILES[water_type]
    R = fresnel_reflection(w['n_water'])
    return (1.0 - R) * np.exp(-w['k_par'] * depth_m)


def underwater_irradiance(surface_irr, water_type, depth_m=DEPTH_M):
    """Irradiance at depth (kWh/m²/day)."""
    return surface_irr * transmission_at_depth(water_type, depth_m)


def underwater_energy_kj_per_day(surface_irr, water_type, depth_m=DEPTH_M):
    """Daily electrical energy (kJ) from a 1 m² panel at depth."""
    I_z = underwater_irradiance(surface_irr, water_type, depth_m)
    w = WATER_PROFILES[water_type]
    return I_z * 3600 * BASE_EFFICIENCY * w['eta_spectral']


# =============================================================================
# HOURLY HARVEST
# =============================================================================

def irradiance_at_hour(hour, season, water_type, depth_m=DEPTH_M):
    """Hourly underwater irradiance (kW/m²)."""
    sunrise = SUNRISE[season]
    sunset  = SUNSET[season]
    if hour < sunrise or hour > sunset:
        return 0.0

    daylight = sunset - sunrise
    t = (hour - sunrise) / daylight
    shape = np.sin(np.pi * t)

    # Scale so daily integral matches surface irradiance
    surface_kw = SURFACE_IRRADIANCE[water_type][season] / 24.0
    scaling = (np.pi / 2) * (surface_kw * 24 / daylight)

    # Underwater transmission
    trans = transmission_at_depth(water_type, depth_m)
    return max(0.0, shape * scaling * trans)


def harvest_minute_hour(hour, season, water_type, depth_m=DEPTH_M):
    minutes = []
    total_kj = 0.0
    peak_w = 0.0
    w = WATER_PROFILES[water_type]

    for minute in range(60):
        t = hour + minute / 60.0
        irr = irradiance_at_hour(t, season, water_type, depth_m)
        power_w = irr * PANEL_AREA_M2 * BASE_EFFICIENCY * w['eta_spectral'] * 1000
        energy_kj = power_w * 60 / 1000

        minutes.append({
            'minute': minute,
            'irradiance': irr,
            'power_watts': power_w,
            'energy_kj': energy_kj,
        })
        total_kj += energy_kj
        if power_w > peak_w:
            peak_w = power_w

    return {
        'hour': hour, 'season': season,
        'minutes_data': minutes,
        'total_energy_kj': total_kj,
        'peak_power_w': peak_w,
    }


def harvest_hourly(season, water_type, depth_m=DEPTH_M):
    return [harvest_minute_hour(h, season, water_type, depth_m) for h in range(24)]


# =============================================================================
# SECTION 1 — WATER PROPERTIES
# =============================================================================

def section_1_water_properties():
    print("\n" + "=" * 100)
    print("SECTION 1: WATER OPTICAL PROPERTIES")
    print("=" * 100)
    print(f"{'Water Type':<18} | {'k_par':<8} | {'n':<6} | {'η_spec':<8} | "
          f"{'Turbidity':<10} | {'Chlorophyll':<12} | {'Suspended':<12}")
    print("-" * 100)

    for wt, data in WATER_PROFILES.items():
        print(f"{wt:<18} | {data['k_par']:<8.3f} | {data['n_water']:<6.2f} | "
              f"{data['eta_spectral']:<8.2f} | {data['turbidity']:<10} | "
              f"{data['chlorophyll']:<12} | {data['suspended_solids']:<12}")

    print("\nDescriptions:")
    for wt, data in WATER_PROFILES.items():
        print(f"  {wt:<18} : {data['description']}")
    print("=" * 100)


# =============================================================================
# SECTION 2 — DEPTH PROFILE
# =============================================================================

def section_2_depth_profile():
    print("\n" + "=" * 100)
    print("SECTION 2: IRRADIANCE VS DEPTH — ALL WATER TYPES")
    print("=" * 100)

    depths = [0, 0.5, 1, 2, 3, 5, 10, 20]

    # Print transmission tables
    header = f"{'Depth (m)':<10} | "
    for wt in WATER_PROFILES:
        header += f"{wt:<20} | "
    print(header)
    print("-" * 100)

    for z in depths:
        row = f"{z:<10.1f} | "
        for wt in WATER_PROFILES:
            trans = transmission_at_depth(wt, z)
            row += f"{trans*100:>18.3f}% | "
        print(row)

    print("=" * 100)
    print("\nIrradiance (kWh/m²/day) — Summer surface irradiance as reference:")

    header = f"{'Depth (m)':<10} | "
    for wt in WATER_PROFILES:
        header += f"{wt:<20} | "
    print(header)
    print("-" * 100)

    for z in depths:
        row = f"{z:<10.1f} | "
        for wt in WATER_PROFILES:
            # Use the water-type-specific summer irradiance
            I_0 = SURFACE_IRRADIANCE[wt]['Summer']
            I_z = underwater_irradiance(I_0, wt, z)
            row += f"{I_z:>18.4f} | "
        print(row)

    print("=" * 100)


# =============================================================================
# SECTION 3 — SURFACE vs 3 m COMPARISON
# =============================================================================

def section_3_surface_vs_3m():
    print("\n" + "=" * 100)
    print("SECTION 3: SURFACE vs 3 m COMPARISON — ALL WATER TYPES")
    print("=" * 100)
    print(f"{'Water Type':<18} | {'Season':<10} | {'Surface (kJ/day)':<18} | "
          f"{'3 m (kJ/day)':<16} | {'Ratio':<10} | {'Reduction':<12}")
    print("-" * 100)

    for wt in WATER_PROFILES:
        for s in SEASONS:
            surf_irr = SURFACE_IRRADIANCE[wt][s]
            surf_kj = surf_irr * PANEL_AREA_M2 * BASE_EFFICIENCY * 3600

            deep_kj = (underwater_energy_kj_per_day(surf_irr, wt, DEPTH_M)
                       * PANEL_AREA_M2)
            ratio = deep_kj / surf_kj if surf_kj > 0 else 0

            print(f"{wt:<18} | {s:<10} | {surf_kj:>16,.0f} | "
                  f"{deep_kj:>14,.0f} | {ratio:>8.4f} | "
                  f"{(1-ratio)*100:>10.2f}%")
        print("-" * 100)
    print("=" * 100)


# =============================================================================
# SECTION 4 — HOURLY HARVEST
# =============================================================================

def section_4_hourly(hourly_all):
    print("\n" + "=" * 100)
    print("SECTION 4: HOURLY HARVEST AT 3 m — DEEP SEA vs COASTAL")
    print("=" * 100)

    for wt in ['Deep Sea', 'Coastal']:
        print(f"\n{wt.upper()} — Energy per hour (kJ):")
        print(f"{'Hour':<8} | " + " | ".join(f"{s:<12}" for s in SEASONS))
        print("-" * 90)
        for hour in range(24):
            row = f"{hour:>2}:00    | "
            for s in SEASONS:
                row += f"{hourly_all[wt][s][hour]['total_energy_kj']:>10.3f} | "
            print(row)
        print("-" * 90)
        row = f"{'TOTAL':<8} | "
        for s in SEASONS:
            total = sum(h['total_energy_kj'] for h in hourly_all[wt][s])
            row += f"{total:>10.1f} | "
        print(row)
        print("=" * 90)


# =============================================================================
# SECTION 5 — SEASONAL SUMMARY
# =============================================================================

def section_5_summary(hourly_all):
    print("\n" + "=" * 100)
    print("SECTION 5: SEASONAL SUMMARY AT 3 m")
    print("=" * 100)
    print(f"{'Water Type':<12} | {'Season':<10} | {'Total (kJ)':<14} | "
          f"{'Total (kWh)':<12} | {'Peak W':<10} | {'Avg W':<10} | {'THz bursts':<14}")
    print("-" * 110)

    for wt in ['Deep Sea', 'Coastal', 'Lake (reference)']:
        for s in SEASONS:
            E = np.array([h['total_energy_kj'] for h in hourly_all[wt][s]])
            total = float(np.sum(E))
            peak = float(max(h['peak_power_w'] for h in hourly_all[wt][s]))
            avg = total / 24 / 3.6
            bursts = total / 0.0117

            print(f"{wt:<12} | {s:<10} | {total:>12.2f} | "
                  f"{total/3600:>10.4f} | {peak:>8.2f} | "
                  f"{avg:>8.2f} | {bursts:>12,.0f}")
        print("-" * 110)
    print("=" * 100)


# =============================================================================
# SECTION 6 — CROSS-WATER COMPARISON
# =============================================================================

def section_6_comparison(hourly_all):
    print("\n" + "=" * 100)
    print("SECTION 6: CROSS-WATER COMPARISON AT 3 m (SUMMER)")
    print("=" * 100)

    deep_total = sum(h['total_energy_kj'] for h in hourly_all['Deep Sea']['Summer'])
    coast_total = sum(h['total_energy_kj'] for h in hourly_all['Coastal']['Summer'])
    lake_total = sum(h['total_energy_kj'] for h in hourly_all['Lake (reference)']['Summer'])

    print(f"Deep Sea         : {deep_total:>10.2f} kJ/day  (baseline)")
    print(f"Coastal          : {coast_total:>10.2f} kJ/day  "
          f"({coast_total/deep_total*100:.1f}% of deep sea)")
    print(f"Lake             : {lake_total:>10.2f} kJ/day  "
          f"({lake_total/deep_total*100:.1f}% of deep sea)")
    print()
    print(f"Deep Sea / Coastal ratio : {deep_total/coast_total:.2f}×")
    print(f"Deep Sea / Lake ratio    : {deep_total/lake_total:.2f}×")
    print()
    print(f"Deep Sea peak power      : "
          f"{max(h['peak_power_w'] for h in hourly_all['Deep Sea']['Summer']):.2f} W")
    print(f"Coastal peak power       : "
          f"{max(h['peak_power_w'] for h in hourly_all['Coastal']['Summer']):.2f} W")
    print(f"Lake peak power          : "
          f"{max(h['peak_power_w'] for h in hourly_all['Lake (reference)']['Summer']):.2f} W")
    print("=" * 100)


# =============================================================================
# SECTION 7 — PLOTS
# =============================================================================

def section_7_plots(hourly_all):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Solar Harvesting at 3 m — Deep Sea vs Coastal',
                 fontsize=15, fontweight='bold')

    colors = {'Summer': '#f1c40f', 'Autumn': '#e67e22',
              'Winter': '#3498db', 'Spring': '#2ecc71'}
    hours = np.arange(24)

    # Plot 1 — Irradiance vs depth (all water types)
    ax = axes[0, 0]
    depths = np.linspace(0, 10, 100)
    for wt, data in WATER_PROFILES.items():
        I_0 = SURFACE_IRRADIANCE[wt]['Summer']
        irr = [underwater_irradiance(I_0, wt, z) for z in depths]
        ax.plot(depths, irr, label=wt, linewidth=2)
    ax.axvline(x=DEPTH_M, color='red', linestyle='--',
               label=f'Master depth ({DEPTH_M} m)')
    ax.set_xlabel('Depth (m)'); ax.set_ylabel('Irradiance (kWh/m²/day)')
    ax.set_title('Irradiance vs Depth — All Water Types')
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 2 — Hourly harvest (Deep Sea)
    ax = axes[0, 1]
    for s in SEASONS:
        E = [h['total_energy_kj'] for h in hourly_all['Deep Sea'][s]]
        ax.plot(hours, E, 'o-', label=s, color=colors[s],
                linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Energy (kJ)')
    ax.set_title('Deep Sea — Hourly Harvest at 3 m')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 3 — Hourly harvest (Coastal)
    ax = axes[1, 0]
    for s in SEASONS:
        E = [h['total_energy_kj'] for h in hourly_all['Coastal'][s]]
        ax.plot(hours, E, 'o-', label=s, color=colors[s],
                linewidth=2, markersize=4)
    ax.set_xlabel('Hour'); ax.set_ylabel('Energy (kJ)')
    ax.set_title('Coastal — Hourly Harvest at 3 m')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Plot 4 — Daily totals comparison
    ax = axes[1, 1]
    x = np.arange(len(SEASONS))
    width = 0.25

    deep_vals = [sum(h['total_energy_kj'] for h in hourly_all['Deep Sea'][s])
                 for s in SEASONS]
    coast_vals = [sum(h['total_energy_kj'] for h in hourly_all['Coastal'][s])
                  for s in SEASONS]
    lake_vals = [sum(h['total_energy_kj'] for h in hourly_all['Lake (reference)'][s])
                 for s in SEASONS]

    ax.bar(x - width, deep_vals, width, label='Deep Sea', color='#3498db')
    ax.bar(x, coast_vals, width, label='Coastal', color='#e67e22')
    ax.bar(x + width, lake_vals, width, label='Lake (ref)', color='#7f8c8d')

    ax.set_xticks(x); ax.set_xticklabels(SEASONS)
    ax.set_ylabel('Daily Energy (kJ)')
    ax.set_title('Daily Harvest at 3 m — Water Type Comparison')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_derivation()
    section_1_water_properties()
    section_2_depth_profile()
    section_3_surface_vs_3m()

    hourly_all = {}
    for wt in WATER_PROFILES:
        hourly_all[wt] = {s: harvest_hourly(s, wt, DEPTH_M) for s in SEASONS}

    section_4_hourly(hourly_all)
    section_5_summary(hourly_all)
    section_6_comparison(hourly_all)
    section_7_plots(hourly_all)

    print("\n" + "=" * 100)
    print("✅ COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()