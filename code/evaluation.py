import numpy as np

def calculate_total_noise(P_r=0.01, R=0.5, B=None, I_dark=1e-9, rise_time=1e-9):
    """
    Calculate total noise variance in an underwater optical communication system.
    
    Parameters:
    -----------
    P_r : float
        Received optical power (Watts)
    R : float
        Photodetector responsivity (A/W) - from datasheet quantum efficiency
    B : float
        Electrical bandwidth (Hz)
    I_dark : float
        Dark current (Amperes) - from datasheet
    rise_time : float, optional
        Rise time (seconds) - if provided, used to estimate bandwidth
    
    Returns:
    --------
    dict : {
        'total_noise': float (A²),
        'shot_noise': float (A²),
        'dark_noise': float (A²),
        'thermal_noise': float (A²),
        'snr_linear': float,
        'snr_db': float
    }
    """

    # Example 1: Basic calculation with dark current and rise time
    print("=" * 50)
    print("AD1900-11 APD Noise Calculation")
    print("=" * 50)
    
    # Constants
    q = 1.6e-19      # Electron charge (Coulombs)
    k = 1.38e-23     # Boltzmann constant (J/K)
    T = 300          # Temperature (Kelvin) - room temperature
    R_L = 50         # Load resistance (Ohms) - typical value
    
    # If rise_time provided, estimate bandwidth using: B = 0.35 / rise_time
    if rise_time is not None:
        B = 0.35 / rise_time
    
    # 1. Shot Noise (from signal current)
    I_signal = R * P_r
    shot_noise = 2 * q * I_signal * B
    
    # 2. Dark Current Noise (from datasheet)
    dark_noise = 2 * q * I_dark * B
    
    # 3. Thermal Noise (Johnson noise from load resistor)
    thermal_noise = 4 * k * T * B / R_L
    
    # 4. Total Noise (sum of all components)
    total_noise = shot_noise + dark_noise + thermal_noise
    
    # 5. SNR Calculation
    signal_power = I_signal ** 2
    snr_linear = signal_power / total_noise if total_noise > 0 else 0
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
    
    return {
        'total_noise': total_noise,
        'shot_noise': shot_noise,
        'dark_noise': dark_noise,
        'thermal_noise': thermal_noise,
        'snr_linear': snr_linear,
        'snr_db': snr_db,
        'signal_current': I_signal,
        'bandwidth': B
    }


# ============================================================================
# Example usage with the AD1900-11 APD datasheet
# ============================================================================

if __name__ == "__main__":
    
    # From the datasheet:
    # - Active area: 1900 μm diameter
    # - Dark current: ~1 nA (typical at low gain)
    # - Rise time: ~1 ns (typical for high-speed APDs)
    
    # Example 1: Basic calculation with dark current and rise time
    print("=" * 50)
    print("AD1900-11 APD Noise Calculation")
    print("=" * 50)
    
    # Assumptions for a typical underwater optical link
    P_r = 0.01          # 10 mW received power
    R = 0.5             # ~0.5 A/W responsivity at 500nm (from datasheet QE curve)
    I_dark = 1e-9       # 1 nA dark current (from datasheet)
    rise_time = 1e-9    # 1 ns rise time (from datasheet)
    
    results = calculate_total_noise(P_r, R, None, I_dark, rise_time)
    
    print(f"Input Parameters:")
    print(f"  Received Power: {P_r*1000:.1f} mW")
    print(f"  Responsivity: {R:.2f} A/W")
    print(f"  Dark Current: {I_dark*1e9:.1f} nA")
    print(f"  Rise Time: {rise_time*1e9:.1f} ns")
    print(f"  Bandwidth: {results['bandwidth']/1e6:.1f} MHz")
    print()
    print(f"Noise Components:")
    print(f"  Shot Noise: {results['shot_noise']:.2e} A²")
    print(f"  Dark Noise: {results['dark_noise']:.2e} A²")
    print(f"  Thermal Noise: {results['thermal_noise']:.2e} A²")
    print(f"  Total Noise: {results['total_noise']:.2e} A²")
    print()
    print(f"SNR:")
    print(f"  Linear: {results['snr_linear']:.2e}")
    print(f"  dB: {results['snr_db']:.2f} dB")
    print(f"  Signal Current: {results['signal_current']:.2e} A")
    
    
    # Example 2: Compare different received powers
    print("\n" + "=" * 50)
    print("SNR vs Received Power")
    print("=" * 50)
    
    print(f"{'P_r (mW)':>10} | {'SNR (dB)':>10}")
    print("-" * 25)
    
    for P_r_mW in [1, 5, 10, 20, 50, 100]:
        P_r = P_r_mW / 1000  # Convert mW to W
        results = calculate_total_noise(P_r, R, None, I_dark, rise_time)
        print(f"{P_r_mW:>10.1f} | {results['snr_db']:>10.2f}")