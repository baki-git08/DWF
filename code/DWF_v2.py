import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line if running interactively with a display
import matplotlib.pyplot as plt

# DIRECTIONAL WATER FILLING WITH VALIDATION AND MINIMUM BATTERY RESERVE


np.set_printoptions(precision=2, suppress=True)

# GLOBAL VARIABLES
# 60 samples collected over a 1-hour span -> 1 minute per sample.
Harvested_energy = np.array([
    4660.579500901392, 4512.8425041569935, 4493.455981638057,
    4492.731098105154, 4543.257333460804, 4546.982641736262,
    4482.832584074914, 4584.158867460887, 4568.930363429445,
    4545.047171620674, 4473.302476681726, 4479.605684014643,
    4496.13399656337, 4508.062486773566, 4492.972108268132,
    4480.788695302276, 4511.296104910635, 4478.5973313381355,
    4482.266863496713, 4446.792438067549, 4440.490614286565,
    4420.270124463221, 4484.450517387264, 4476.797167482416,
    4490.003822630348, 4492.056543657553, 4479.238432024654,
    4480.155225158745, 4505.10474190502, 4475.051560473867,
    4499.25508960422, 4493.133780764153, 4495.870299049443,
    4417.975409293113, 4491.183764351152, 4427.62818658292,
    4433.766382521933, 4442.956944971574, 4414.195222620365,
    4432.869190909629, 4353.948448673094, 4454.4521426521205,
    4444.736560801864, 4413.08049824759, 4390.7935061389835,
    4427.674738692369, 4382.891564664771, 4373.214266250408,
    4361.853912731085, 4384.632438803332, 4361.569090584382,
    4424.171916484096, 4458.714213778103, 4365.822212708611,
    4379.731080987091, 4388.843836357234, 4331.800646342195,
    4286.445636813736, 4381.7154802308, 4353.305733462656,
])
Transmission_duration = np.full(len(Harvested_energy), 60.0)  # seconds; 60 x 60s = 1 hour
LOWER_BATTERY_LIMIT = 450
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


def is_causal(power, operation_energy, transmission_duration):
    '''Battery built from operation energy (reserve already set aside up front)
    must never go negative. Used while pooling on operation energy, where the
    reserve has already been removed and should NOT be demanded again.'''
    battery = np.cumsum(operation_energy - power * transmission_duration)
    return np.all(battery >= -EPS)


def directional_water_filling_with_validation(operation_energy, transmission_duration):
    '''DWF with battery feasibility validation. Operates on OPERATION energy
    (already reserve-adjusted), so the pooling logic itself is unchanged.'''
    n = len(operation_energy)
    power = compute_power(operation_energy, transmission_duration)
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

                applied = False
                while end > start:
                    avg_power = (np.sum(operation_energy[start:end + 1])
                                 / np.sum(transmission_duration[start:end + 1]))
                    temp_power = power.copy()
                    temp_power[start:end + 1] = avg_power

                    if is_causal(temp_power, operation_energy, transmission_duration):
                        if not np.allclose(power[start:end + 1], avg_power, atol=EPS):
                            power[start:end + 1] = avg_power
                            print(f"  Applied hill [{start}:{end}]: {power}")
                            changed = True
                            applied = True
                        break

                    print(f"  Hill [{start}:{end}] not causal, trying smaller hill")
                    end -= 1

                if not applied:
                    print("  Cannot reduce further, skipping")
                    continue
                break
    return power


def plot_results(harvested_energy, operational_power, final_power, battery_level,
                  transmission_duration, filename="dwf_v2_results.png"):
    '''Plot harvested energy, operational (pre-pooling) power, final (pooled)
    power, and battery level over time (minutes).'''
    minutes = np.cumsum(transmission_duration) / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(minutes, harvested_energy, color="tab:green", marker=".")
    axes[0].set_ylabel("Harvested energy (J)")
    axes[0].set_title("Harvested Energy")

    axes[1].plot(minutes, operational_power, color="tab:orange", marker=".")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_title("Operational Power (pre-pooling, from operation energy)")

    axes[2].plot(minutes, final_power, color="tab:blue", marker=".")
    axes[2].set_ylabel("Power (W)")
    axes[2].set_title("Final Power (after DWF pooling)")

    axes[3].plot(minutes, battery_level, color="tab:red", marker=".")
    axes[3].axhline(LOWER_BATTERY_LIMIT, color="black", linestyle="--", linewidth=1,
                     label=f"Reserve ({LOWER_BATTERY_LIMIT} J)")
    axes[3].set_ylabel("Battery level (J)")
    axes[3].set_title("Battery Level")
    axes[3].legend()
    axes[3].set_xlabel("Time (minutes)")

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    print(f"saved {filename}")


def main():
    # Run DWF with validation
    print("Running Directional Water Filling with Reserve + Feasibility Validation")
    print("=" * 60)

    operation_energy = compute_operation_energy(Harvested_energy, LOWER_BATTERY_LIMIT)
    print(f"Harvested energy:   {Harvested_energy}")
    print(f"Operation energy:   {operation_energy}\n")

    operational_power = compute_power(operation_energy, Transmission_duration)
    power = directional_water_filling_with_validation(operation_energy, Transmission_duration)
    print(f"\nFinal Power: {power}")

    battery_level = compute_battery_level(power, Harvested_energy, Transmission_duration)
    print(f"Battery Level: {battery_level}")

    throughput = compute_throughput(power, Transmission_duration)
    print(f"Throughput: {throughput}")

    print(f"\nIs feasible? {is_feasible(power, Harvested_energy, Transmission_duration)}")
    min_battery = np.min(battery_level)
    print(f"Minimum battery level: {min_battery}")

    plot_results(Harvested_energy, operational_power, power, battery_level, Transmission_duration)

    return power


if __name__ == "__main__":
    main()