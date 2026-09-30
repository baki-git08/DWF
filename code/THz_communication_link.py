"""
UNDERWATER mmWAVE LINK — HORN ANTENNAS
========================================

Location-agnostic model.

Link budget:
    P_rx(d) = P_tx + G_tx + G_rx − FSPL(d) − α·d

Parameters you can vary:
    - frequency (30–300 GHz)
    - water type (fresh / coastal / sea)
    - horn gain (dBi)
    - Tx power (dBm)
    - receiver sensitivity (dBm)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

C = 3e8                     # speed of light (m/s)

# Frequencies of interest (mmWave band)
FREQ_CHOICES = [30e9, 60e9, 100e9, 300e9]

# Attenuation table (dB/m) — indicative values
WATER_ATTENUATION = {
    'Freshwater': {30e9: 3,   60e9: 10,  100e9: 30,   300e9: 200},
    'Coastal':    {30e9: 50,  60e9: 120, 100e9: 300,  300e9: 1500},
    'Seawater':   {30e9: 150, 60e9: 400, 100e9: 1000, 300e9: 5000},
}

# Link parameters
P_TX_DBM       = 10.0       # transmitted power
GAIN_DBI       = 20.0       # horn gain (each end)
SENSITIVITY    = -70.0      # receiver sensitivity (dBm)

# Distance sweep (cm scale for realistic mmWave underwater)
DISTANCES_M = np.linspace(0.01, 5.0, 500)


# -----------------------------------------------------------------------------
# PHYSICS
# -----------------------------------------------------------------------------

def wavelength(freq_hz):
    return C / freq_hz


def horn_gain_from_aperture(freq_hz, A_m, B_m, eta_ap=0.51):
    """Optional — compute gain from physical horn size."""
    lam = wavelength(freq_hz)
    G = (4 * np.pi / lam**2) * eta_ap * A_m * B_m
    return 10 * np.log10(G)


def fspl_db(d_m, freq_hz):
    """Free-space path loss (dB)."""
    d_m = np.maximum(d_m, 1e-9)
    return 20 * np.log10(4 * np.pi * d_m * freq_hz / C)


def received_power_dbm(d_m, freq_hz, alpha_db_per_m,
                       p_tx_dbm=P_TX_DBM, g_tx=GAIN_DBI, g_rx=GAIN_DBI):
    """Full link budget (dBm)."""
    return (p_tx_dbm + g_tx + g_rx
            - fspl_db(d_m, freq_hz)
            - alpha_db_per_m * d_m)


def max_range_m(freq_hz, alpha_db_per_m,
                p_tx_dbm=P_TX_DBM, g_tx=GAIN_DBI, g_rx=GAIN_DBI,
                sens_dbm=SENSITIVITY):
    """Largest distance where P_rx ≥ sensitivity."""
    p_rx = received_power_dbm(DISTANCES_M, freq_hz, alpha_db_per_m,
                              p_tx_dbm, g_tx, g_rx)
    valid = np.where(p_rx >= sens_dbm)[0]
    return DISTANCES_M[valid[-1]] if len(valid) else 0.0


# -----------------------------------------------------------------------------
# SECTION 1 — HORN GAIN EXAMPLES
# -----------------------------------------------------------------------------

def section_1_horn_examples():
    print("=" * 90)
    print("SECTION 1: HORN ANTENNA GAIN EXAMPLES")
    print("=" * 90)
    print(f"{'Frequency':<12} | {'Wavelength':<14} | {'Aperture':<16} | "
          f"{'Gain (dBi)':<12}")
    print("-" * 90)

    apertures = [
        ('2 cm × 1.5 cm', 0.02, 0.015),
        ('5 mm × 5 mm',   0.005, 0.005),
        ('2 mm × 2 mm',   0.002, 0.002),
    ]

    for f in FREQ_CHOICES:
        lam = wavelength(f)
        for name, A, B in apertures:
            G = horn_gain_from_aperture(f, A, B)
            print(f"{f/1e9:>6.0f} GHz   | {lam*1000:>10.3f} mm  | "
                  f"{name:<16} | {G:>10.2f}")
        print("-" * 90)


# -----------------------------------------------------------------------------
# SECTION 2 — MAX RANGE TABLE
# -----------------------------------------------------------------------------

def section_2_max_range():
    print("\n" + "=" * 90)
    print("SECTION 2: MAXIMUM RANGE (cm) FOR EACH FREQUENCY AND WATER TYPE")
    print("=" * 90)
    print(f"Assumptions: P_tx = {P_TX_DBM} dBm, "
          f"G_tx = G_rx = {GAIN_DBI} dBi, sensitivity = {SENSITIVITY} dBm")
    print()
    print(f"{'Water':<12} | " + " | ".join(f"{f/1e9:>6.0f} GHz" for f in FREQ_CHOICES))
    print("-" * 90)

    for water, alphas in WATER_ATTENUATION.items():
        row = f"{water:<12} | "
        for f in FREQ_CHOICES:
            r = max_range_m(f, alphas[f])
            row += f"{r*100:>10.2f} cm | "
        print(row)
    print("=" * 90)


# -----------------------------------------------------------------------------
# SECTION 3 — RECEIVED POWER VS DISTANCE
# -----------------------------------------------------------------------------

def section_3_plot_received():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('mmWave Underwater Link — Received Power vs Distance',
                 fontsize=14, fontweight='bold')

    for ax, water in zip(axes, WATER_ATTENUATION):
        for f in FREQ_CHOICES:
            p_rx = received_power_dbm(DISTANCES_M, f,
                                      WATER_ATTENUATION[water][f])
            ax.plot(DISTANCES_M * 100, p_rx, label=f"{f/1e9:.0f} GHz")
        ax.axhline(SENSITIVITY, color='red', linestyle=':',
                   label=f'Sensitivity ({SENSITIVITY} dBm)')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Received Power (dBm)')
        ax.set_title(f'{water}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-150, 10)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# SECTION 4 — MAX RANGE BAR CHART
# -----------------------------------------------------------------------------

def section_4_plot_range():
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(WATER_ATTENUATION))
    width = 0.2
    for i, f in enumerate(FREQ_CHOICES):
        ranges = [max_range_m(f, WATER_ATTENUATION[w][f]) * 100
                  for w in WATER_ATTENUATION]
        ax.bar(x + i * width, ranges, width, label=f"{f/1e9:.0f} GHz")

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(WATER_ATTENUATION.keys())
    ax.set_ylabel('Maximum range (cm)')
    ax.set_title(f'mmWave Max Range — Sensitivity {SENSITIVITY} dBm, '
                 f'Gain {GAIN_DBI} dBi')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    section_1_horn_examples()
    section_2_max_range()
    section_3_plot_received()
    section_4_plot_range()