import numpy as np

# DIRECTIONAL WATER FILLING WITH VALIDATION AND MINIMUM BATTERY RESERVE


np.set_printoptions(precision=2, suppress=True)

# GLOBAL VARIABLES
Harvested_energy = np.array([10, 6, 9, 15, 16, 1])
Transmission_duration = np.array([2, 3, 3, 3, 2, 1])
LOWER_BATTERY_LIMIT = 1
INIT_BATTERY_LEVEL = 0
EPS = 1e-9


def compute_operation_energy(energy_harvested, lower_battery_limit):
    '''
    Carve the reserve requirement off the FRONT of the harvested-energy
    sequence, across as many epochs as needed. Returns the energy actually
    available for transmission each epoch (the OPERATION_ENERGY).

    Example matching the spec: battery starts at 0J, harvested = 5J,
    LOWER_BATTERY_LIMIT = 2J -> 2J reserved, 3J returned as operation energy.
    '''
    operation_energy = energy_harvested.astype(float).copy()
    remaining_reserve = float(lower_battery_limit) - INIT_BATTERY_LEVEL
    for i in range(len(operation_energy)):
        if remaining_reserve <= 0:
            break
        take = min(remaining_reserve, operation_energy[i])
        operation_energy[i] -= take
        remaining_reserve -= take
    if remaining_reserve > 0:
        print(f"Warning: total harvested energy ({energy_harvested.sum()}J) is insufficient "
              f"to ever meet the {lower_battery_limit}J reserve. No transmission is possible.")
    return operation_energy


def compute_power(energy_harvested, transmission_duration):
    '''Compute per-block power from harvested energy and transmission duration.'''
    transmission_blocks = len(energy_harvested)
    power = np.zeros(transmission_blocks)
    for i in range(transmission_blocks):
        if transmission_duration[i] > 0:
            power[i] = energy_harvested[i] / transmission_duration[i]
        else:
            print(f"Warning: Transmission duration for block {i} is zero. Setting power to 0.")
            power[i] = 0
    return power


def compute_battery_level(power, energy_harvested, transmission_duration):
    '''Compute per-block cumulative PHYSICAL battery level (uses real harvested energy).'''
    transmission_blocks = len(energy_harvested)
    battery_level = np.zeros(transmission_blocks)
    cumulative_battery = INIT_BATTERY_LEVEL

    for i in range(transmission_blocks):
        consumed_energy = power[i] * transmission_duration[i]
        cumulative_battery = cumulative_battery + energy_harvested[i] - consumed_energy
        print(f"Block {i}: Harvested={energy_harvested[i]}, Consumed={consumed_energy:.4f}, "
              f"Cumulative Battery={cumulative_battery:.4f}")
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
    '''Check if a power allocation is feasible against the PHYSICAL battery/reserve.'''
    transmission_blocks = len(energy_harvested)
    battery = INIT_BATTERY_LEVEL
    reserve_reached = False
    for i in range(transmission_blocks):
        battery = battery + energy_harvested[i] - power[i] * transmission_duration[i]
        if battery >= LOWER_BATTERY_LIMIT - EPS:
            reserve_reached = True
        # only enforce the floor once the reserve has been reached at least once;
        # before that, a climbing battery below LIMIT is expected, not a violation
        if reserve_reached and battery < LOWER_BATTERY_LIMIT - EPS:
            print(f"Feasibility check failed at block {i}: Battery level {battery:.4f} "
                  f"dropped back below reserve {LOWER_BATTERY_LIMIT} after being reached")
            return False
    return True


def directional_water_filling_with_validation(Harvested_energy, transmission_duration):
    '''DWF with battery feasibility validation. Operates on OPERATION energy
    (already reserve-adjusted), so the pooling logic itself is unchanged.'''
    # transmission_blocks = len(Harvested_energy)
    n = len(Harvested_energy)
    power = compute_power(Harvested_energy, transmission_duration)
    print(f"Initial Power: {power}")

    iteration = 0
    changed = True
    while changed:
        changed = False
        iteration += 1
        print(f"\nIteration {iteration}: {power}")


        for i in range(n - 1):
            if power[i] > power[i + 1] + EPS:
                print(f"  Detected flow at i={i}: {power[i]:.3f} > {power[i+1]:.3f}, pooling...")
                start, end = i, i + 1
                while start > 0 and power[start - 1] >= power[start] - EPS:
                    start -= 1
                while end < n - 1 and power[end] > power[end + 1] + EPS:
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

                avg_power = (np.sum(Harvested_energy[start:end + 1])/np.sum(transmission_duration[start:end + 1]))
                power[start:end + 1] = avg_power
                changed = True
                break
    return power


def main():
    # Run DWF with validation
    print("Running Directional Water Filling with Reserve + Feasibility Validation")
    print("=" * 60)

    operation_energy = compute_operation_energy(Harvested_energy, LOWER_BATTERY_LIMIT)
    print(f"Harvested energy:   {Harvested_energy}")
    print(f"Operation energy:   {operation_energy}\n")

    power = directional_water_filling_with_validation(operation_energy, Transmission_duration)
    print(f"\nFinal Power: {power}")

    battery_level = compute_battery_level(power, Harvested_energy, Transmission_duration)
    print(f"Battery Level: {battery_level}")

    throughput = compute_throughput(power, Transmission_duration)
    print(f"Throughput: {throughput}")

    print(f"\nIs feasible? {is_feasible(power, Harvested_energy, Transmission_duration)}")
    min_battery = np.min(battery_level)
    print(f"Minimum battery level: {min_battery}")

    return power 

operation_energy = compute_operation_energy(Harvested_energy, LOWER_BATTERY_LIMIT)
power = directional_water_filling_with_validation(operation_energy, Transmission_duration)


if __name__ == "__main__":
    main()