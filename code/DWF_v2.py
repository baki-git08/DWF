import numpy as np

# DIRECTIONAL WATER FILLING WITH VALIDATION


# Set global display precision for all NumPy outputs
np.set_printoptions(precision=2, suppress=True)

# GLOBAL VARIABLES
Harvested_energy = np.array([10, 6, 9, 15, 16, 1])
Transmission_duration = np.array([2, 3, 3, 3, 2, 1])
LOWER_BATTERY_LIMIT = 2  # Minimum battery level allowed
INIT_BATTERY_LEVEL = 0 
EPS = 1e-9 
    

def compute_power(energy_harvested, transmission_duration):
    '''Compute per-block power from harvested energy and transmission duration.'''
    transmission_blocks = len(energy_harvested)  
    power = np.zeros(transmission_blocks)           # initialize power array
    for i in range(transmission_blocks):
        if transmission_duration[i] > 0:
            power[i] = energy_harvested[i] / transmission_duration[i]
        else:
            print(f"Warning: Transmission duration for block {i} is zero. Setting power to 0.")
            power[i] = 0
    return power


def compute_battery_level(power, energy_harvested, transmission_duration):
    '''Compute per-block cumulative battery level.'''
    transmission_blocks = len(energy_harvested)
    battery_level = np.zeros(transmission_blocks)       # Initializing battery level array
    cumulative_battery = 0

    for i in range(transmission_blocks):
        consumed_energy = power[i] * transmission_duration[i]
        cumulative_battery = cumulative_battery + energy_harvested[i] - consumed_energy
        print(f"Block {i}: Harvested={energy_harvested[i]}, Consumed={consumed_energy}, Cumulative Battery={cumulative_battery}")

        # if cumulative_battery < LOWER_BATTERY_LIMIT - EPS:
        #     print(f"Warning: Battery level below lower limit at block {i}. Cumulative battery at {cumulative_battery}")
            # cumulative_battery = LOWER_BATTERY_LIMIT  # Enforce lower battery limit
        
        battery_level[i] = cumulative_battery
    return battery_level


def compute_throughput(power, transmission_duration):
    '''Compute per-block throughput.'''
    transmission_blocks = len(power)  
    throughput = np.zeros(transmission_blocks)
    for i in range(transmission_blocks):
        throughput[i] = transmission_duration[i] / 2 * np.log2(1 + power[i])
    return throughput


def is_feasible(power, energy_harvested, transmission_duration):
    '''Check if a power allocation is feasible (battery never negative).'''
    transmission_blocks = len(energy_harvested)  
    battery = INIT_BATTERY_LEVEL
    for i in range(transmission_blocks):
        battery = battery + energy_harvested[i] - power[i] * transmission_duration[i]
        print(f"Block {i}: Battery level = {battery}")
        if battery < LOWER_BATTERY_LIMIT - EPS:
            print(f"Feasibility check failed at block {i}: Battery level {battery} below lower limit {LOWER_BATTERY_LIMIT}")
            return False
    return True


def directional_water_filling_with_validation(Harvested_energy, Transmission_duration):
    '''DWF with battery feasibility validation.'''
    transmission_blocks = len(Harvested_energy)
    power = compute_power(Harvested_energy, Transmission_duration)
    print(f"Initial Power: {power}")

    iteration = 0
    changed = True
    while changed:
        changed = False
        iteration += 1
        print(f"\nIteration {iteration}: {power}")

        for i in range(transmission_blocks - 1):
            if power[i] > power[i + 1] + EPS: 
                print(f"  Detected flow at i={i}: {power[i]:.3f} > {power[i+1]:.3f}")
                start, end = i, i + 1

                # Extend left while powers are non-decreasing going left
                while start > 0 and power[start - 1] >= power[start] - EPS: 
                    start -= 1

                # Extend right while powers are strictly decreasing
                while end < transmission_blocks - 1 and power[end] > power[end + 1] + EPS:  # FIX: added +EPS
                    end += 1

                print(f"  Hill segment: blocks {start} to {end}")

                # Compute average power
                avg_power = np.sum(Harvested_energy[start:end + 1]) / np.sum(Transmission_duration[start:end + 1])
                print(f"  Average power: {avg_power:.4f}")

                # Test if this allocation is feasible
                temp_power = power.copy()
                temp_power[start:end + 1] = avg_power

                if is_feasible(temp_power, Harvested_energy, Transmission_duration):
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
                        avg_power = np.sum(Harvested_energy[start:end + 1]) / np.sum(Transmission_duration[start:end + 1])
                        temp_power = power.copy()
                        temp_power[start:end + 1] = avg_power

                        if is_feasible(temp_power, Harvested_energy, Transmission_duration):
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

    power = directional_water_filling_with_validation(Harvested_energy, Transmission_duration)
    print(f"\nFinal Power: {power}")

    battery_level = compute_battery_level(power, Harvested_energy, Transmission_duration)
    print(f"Battery Level: {battery_level}")

    throughput = compute_throughput(power, Transmission_duration)
    print(f"Throughput: {throughput}")

    # Verify feasibility
    print(f"\nIs feasible? {is_feasible(power, Harvested_energy, Transmission_duration)}")

    # Additional verification
    min_battery = np.min(battery_level)
    print(f"Minimum battery level: {min_battery}")
    print(f"All battery levels non-negative: {min_battery >= -EPS}")  # FIX: was `>= 0`


if __name__ == "__main__":
    main()