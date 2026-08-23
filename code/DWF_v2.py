import numpy as np

## GLOBAL VARIABLES
N = 6
E = np.array([10, 6, 9, 15, 16, 1])
L = np.array([2, 3, 3, 3, 2, 1])
LOWER_BATTERY_LIMIT = 0  # Minimum battery level allowed

def compute_power(energy_harvested, transmission_duration):
    '''Compute per-block power from harvested energy and transmission duration.'''
    power = np.zeros(N)
    for i in range(N):
        if transmission_duration[i] > 0:
            power[i] = energy_harvested[i] / transmission_duration[i]
        else:
            print(f"Warning: Transmission duration for block {i} is zero. Setting power to 0.")
            exit(1)
    return power

def compute_battery_level(power, energy_harvested, transmission_duration):
    '''Compute per-block cumulative battery level.'''
    battery_level = np.zeros(N)
    cumulative_battery = 0
    
    for i in range(N):
        consumed_energy = power[i] * transmission_duration[i]
        cumulative_battery = cumulative_battery + energy_harvested[i] - consumed_energy
        battery_level[i] = cumulative_battery
    return battery_level

def compute_throughput(power, transmission_duration):
    '''Compute per-block throughput.'''
    throughput = np.zeros(N)
    for i in range(N):
        throughput[i] = transmission_duration[i] / 2 * np.log2(1 + power[i])
    return throughput

def is_feasible(power, energy_harvested, transmission_duration):
    '''Check if a power allocation is feasible (battery never negative).'''
    battery = 0
    for i in range(N):
        battery = battery + energy_harvested[i] - power[i] * transmission_duration[i]
        if battery < 0:
            return False
    return True

def directional_water_filling_with_validation(E, L):
    '''DWF with battery feasibility validation.'''
    N = len(E)
    power = compute_power(E, L)
    print(f"Initial Power: {power}")
    
    changed = True
    while changed:
        changed = False
        
        for i in range(N - 1):
            if power[i] > power[i + 1]:
                start, end = i, i + 1
                
                # Extend left while powers are non-decreasing going left
                while start > 0 and power[start - 1] >= power[start]:
                    start -= 1
                
                # Extend right while powers are non-increasing going right
                while end < N - 1 and power[end] >= power[end + 1]:
                    end += 1
                
                # Compute average power
                avg_power = np.sum(E[start:end + 1])/ np.sum(L[start:end + 1])
                
                # Test if this allocation is feasible
                test_power = power.copy()
                test_power[start:end + 1] = avg_power
                
                if is_feasible(test_power, E, L):
                    # It's feasible! Apply the change
                    power[start:end + 1] = avg_power
                    changed = True
                    break
                else:
                    # Not feasible - this means the hill should be smaller
                    # Try reducing the hill by one block on the right
                    if end > start + 1:
                        end -= 1
                        avg_power = np.sum(E[start:end + 1]) / np.sum(L[start:end + 1])
                        test_power = power.copy()
                        test_power[start:end + 1] = avg_power
                        
                        if is_feasible(test_power, E, L):
                            power[start:end + 1] = avg_power
                            changed = True
                            break
                    else:
                        # Can't reduce further, skip this flow opportunity
                        pass
    
    return power

def main():
    # Run DWF with validation
    power = directional_water_filling_with_validation(E, L)
    print(f"Final Power: {power}")
    
    battery_level = compute_battery_level(power, E, L)
    print(f"Battery Level: {battery_level}")
    
    throughput = compute_throughput(power, L)
    print(f"Throughput: {throughput}")
    
    # Verify feasibility
    print(f"\nIs feasible? {is_feasible(power, E, L)}")

main()