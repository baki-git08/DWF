import numpy as np

# DIRECTIONAL WATER FILLING
#
# This code breaks down "Directional Water Filling"
# into small functions that can be incorporated within the MAIN loop.

# Define parameters

N = 3  # Number of transmission blocks

E = np.array([10, 0, 5])  # Harvested energy per block
L = np.array([2, 1, 2])   # Transmission duration per block


def compute_power(energy_harvested, transmission_duration):
    '''Compute per-block power from harvested energy and transmission duration.'''
    power = np.zeros(N)

    for i in range(N):
        if transmission_duration[i] > 0:
            power[i] = energy_harvested[i] / transmission_duration[i]
        else:
            print(f"Warning: Transmission duration for block {i} is zero. Setting power to 0.")
            exit(1)  # Exit the program if transmission duration is zero
    return power


def compute_battery_level(power, energy_harvested, transmission_duration):
    '''Compute per-block battery level from the (merged) power allocation.'''
    battery_level = np.zeros(N)

    for i in range(N):
        consumed_energy = power[i] * transmission_duration[i]
        battery_level[i] = battery_level[i - 1] + energy_harvested[i] - consumed_energy
    return battery_level


def compute_throughput(power, transmission_duration):
    '''Compute per-block throughput from power and transmission duration.'''
    throughput = np.zeros(N)

    for i in range(N):
        throughput[i] = transmission_duration[i] / 2 * np.log2(1 + power[i])
    return throughput


def main():
    # Compute initial per-block power based on harvested energy and transmission duration
    power = compute_power(E, L)
    print(f"Initial Power: {power}")

    # Directional water-filling: whenever power decreases from one block to the
    # next, merge the offending run of blocks into their average power, and
    # repeat until the power sequence is non-decreasing.
    changed = True
    while changed:
        changed = False

        for i in range(N - 1):
            if power[i] > power[i + 1]:
                """Checking causality: if power[i] > power[i + 1], then energy can flow from blocks i to i+1."""
                start, end = i, i + 1

                while start > 0 and power[start - 1] >= power[start]:
                    start -= 1
                while end < N - 1 and power[end] >= power[end + 1]:
                    end += 1

                average_power = np.sum(E[start:end + 1]) / np.sum(L[start:end + 1])
                power[start:end + 1] = average_power
                changed = True
                break  # restart the scan after every merge

    print(f"Final Power: {power}")

    battery_level = compute_battery_level(power, E, L)
    print(f"Battery Level: {battery_level}")

    throughput = compute_throughput(power, L)
    print(f"Throughput: {throughput}")


main()  # Call the main function to execute the code
