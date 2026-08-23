import numpy as np

# DIRECTIONAL WATER FILLING WITH VALIDATION
#
# This code implements Directional Water Filling with battery feasibility validation,
# fixing the battery level calculation bug found in the original implementation.
#
# FIX (added): floating-point comparisons against exact zero were rejecting valid
# merges at pool boundaries, where the true battery level is 0 but floating-point
# arithmetic produces tiny noise like -1e-15. Added EPS tolerance so these near-zero
# values are treated as zero, matching the real (exact) mathematical result.

# Set global display precision for all NumPy outputs
np.set_printoptions(precision=2, suppress=True)

## GLOBAL VARIABLES
N = 6
E = np.array([10, 6, 9, 15, 16, 1])
L = np.array([2, 3, 3, 3, 2, 1])
LOWER_BATTERY_LIMIT = 0  # Minimum battery level allowed
EPS = 1e-9  # FIX: floating-point tolerance for near-zero comparisons


def compute_power(energy_harvested, transmission_duration):
    '''Compute per-block power from harvested energy and transmission duration.'''
    n = len(energy_harvested)  # FIXED: Use len() instead of global N
    power = np.zeros(n)
    for i in range(n):
        if transmission_duration[i] > 0:
            power[i] = energy_harvested[i] / transmission_duration[i]
        else:
            print(f"Warning: Transmission duration for block {i} is zero. Setting power to 0.")
            power[i] = 0
    return power


def compute_battery_level(power, energy_harvested, transmission_duration):
    '''Compute per-block cumulative battery level.'''
    n = len(energy_harvested)  # FIXED: Use len() instead of global N
    battery_level = np.zeros(n)
    cumulative_battery = 0

    for i in range(n):
        consumed_energy = power[i] * transmission_duration[i]
        cumulative_battery = cumulative_battery + energy_harvested[i] - consumed_energy
        battery_level[i] = cumulative_battery
    return battery_level


def compute_throughput(power, transmission_duration):
    '''Compute per-block throughput.'''
    n = len(power)  # FIXED: Use len() instead of global N
    throughput = np.zeros(n)
    for i in range(n):
        throughput[i] = transmission_duration[i] / 2 * np.log2(1 + power[i])
    return throughput


def is_feasible(power, energy_harvested, transmission_duration):
    '''Check if a power allocation is feasible (battery never negative).'''
    n = len(energy_harvested)  # FIXED: Use len() instead of global N
    battery = 0
    for i in range(n):
        battery = battery + energy_harvested[i] - power[i] * transmission_duration[i]
        if battery < -EPS:  # FIX: was `battery < 0`
            return False
    return True


def directional_water_filling_with_validation(E, L):
    '''DWF with battery feasibility validation.'''
    n = len(E)
    power = compute_power(E, L)
    print(f"Initial Power: {power}")

    iteration = 0
    changed = True
    while changed:
        changed = False
        iteration += 1
        print(f"\nIteration {iteration}: {power}")

        for i in range(n - 1):
            if power[i] > power[i + 1] + EPS:  # FIX: was `power[i] > power[i + 1]`
                print(f"  Detected flow at i={i}: {power[i]:.3f} > {power[i+1]:.3f}")
                start, end = i, i + 1

                # Extend left while powers are non-decreasing going left
                while start > 0 and power[start - 1] >= power[start] - EPS:  # FIX: added -EPS
                    start -= 1

                # Extend right while powers are strictly decreasing
                # FIXED: Use > instead of >= to only include blocks where power is actually decreasing
                while end < n - 1 and power[end] > power[end + 1] + EPS:  # FIX: added +EPS
                    end += 1

                print(f"  Hill segment: blocks {start} to {end}")

                # Compute average power
                avg_power = np.sum(E[start:end + 1]) / np.sum(L[start:end + 1])
                print(f"  Average power: {avg_power:.4f}")

                # Test if this allocation is feasible
                test_power = power.copy()
                test_power[start:end + 1] = avg_power

                if is_feasible(test_power, E, L):
                    # It's feasible! Apply the change
                    power[start:end + 1] = avg_power
                    print(f"  Applied: {power}")
                    changed = True
                    break
                else:
                    # Not feasible - this means the hill should be smaller
                    print("  Not feasible, trying smaller hill")
                    if end > start + 1:
                        end -= 1
                        avg_power = np.sum(E[start:end + 1]) / np.sum(L[start:end + 1])
                        test_power = power.copy()
                        test_power[start:end + 1] = avg_power

                        if is_feasible(test_power, E, L):
                            power[start:end + 1] = avg_power
                            print(f"  Applied smaller hill: {power}")
                            changed = True
                            break
                    else:
                        print("  Cannot reduce further, skipping")

    return power


def main():
    # Run DWF with validation
    print("Running Directional Water Filling with Feasibility Validation")
    print("=" * 60)

    power = directional_water_filling_with_validation(E, L)
    print(f"\nFinal Power: {power}")

    battery_level = compute_battery_level(power, E, L)
    print(f"Battery Level: {battery_level}")

    throughput = compute_throughput(power, L)
    print(f"Throughput: {throughput}")

    # Verify feasibility
    print(f"\nIs feasible? {is_feasible(power, E, L)}")

    # Additional verification
    min_battery = np.min(battery_level)
    print(f"Minimum battery level: {min_battery}")
    print(f"All battery levels non-negative: {min_battery >= -EPS}")  # FIX: was `>= 0`


if __name__ == "__main__":
    main()