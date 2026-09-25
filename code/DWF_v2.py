import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line if running interactively with a display
import matplotlib.pyplot as plt

# DIRECTIONAL WATER FILLING WITH VALIDATION AND MINIMUM BATTERY RESERVE


np.set_printoptions(precision=2, suppress=True)

# GLOBAL VARIABLES
# 60 samples collected over a 1-hour span -> 1 minute per sample.

Harvested_energy_summer = np.array([
    3259.018904543642, 3225.811176367269, 3219.7400798976305,
    3209.669758246244, 3249.4906451390025, 3236.380071665001,
    3251.82378773598, 3223.610002469285, 3237.0895732725335,
    3240.0733948883803, 3164.2113780586687, 3230.8492020193803,
    3215.2177647517046, 3218.005513521021, 3241.167094917427,
    3255.5480745906307, 3232.1343320405863, 3225.884769539228,
    3240.9316782395417, 3261.510851296834, 3223.0173786785344,
    3224.9412380291024, 3231.2154439465958, 3211.0559205948966,
    3214.8859392018485, 3203.4464715152308, 3180.0183280143983,
    3243.8589491804246, 3228.189395874523, 3221.3209687843005,
    3207.9279279447605, 3200.2464371416077, 3219.81254236639,
    3213.9217089027075, 3207.2197442112565, 3224.569901584549,
    3212.609684832128, 3218.96989164818, 3198.0392316207203,
    3158.4607204751087, 3175.6571959014686, 3180.108026137462,
    3197.772281572488, 3188.5012793965257, 3146.3846301963827,
    3156.9431911679744, 3183.102707356057, 3070.5187945800335,
    3193.3999650977457, 3209.8285500632924, 3174.015613864292,
    3133.6307383893263, 3086.283821223546, 3141.097213839667,
    3204.4792179895153, 3170.440171377065, 3166.6525519388197,
    3163.3956233837544, 3149.005419046985, 3182.9536654525896,
])

Harvested_energy_autumn = np.array([
    3759.0463169446766, 3683.0366516822264, 3667.788318463284,
    3727.0513546050124, 3716.009359223653, 3707.671058910965,
    3691.117048289839, 3674.946587776859, 3698.5651482368194,
    3710.8398559839316, 3693.6550855762175, 3709.6145481210106,
    3674.929542059389, 3711.0879488725036, 3677.3271601932415,
    3673.2800149529535, 3690.8936744282905, 3733.851488491906,
    3706.904060958208, 3683.583407350078, 3674.0359991085966,
    3703.999712289754, 3680.962875758944, 3670.0254909162545,
    3669.393300864698, 3677.5557118243078, 3694.733398340336,
    3626.3102523686616, 3651.2880353882274, 3642.2574212729673,
    3644.1647431652323, 3681.3421829615377, 3658.6365454221686,
    3652.9852711636754, 3678.8478045348497, 3647.265935235369,
    3632.6049417989993, 3655.069898912205, 3654.016622447117,
    3625.4691253970173, 3629.472511799474, 3645.4260635223886,
    3604.631305170355, 3618.2418198143814, 3654.9726252532078,
    3623.080493773589, 3626.0100646482083, 3601.4793268373737,
    3675.1168376513506, 3554.9239861557962, 3599.1105722210555,
    3667.36272407978, 3594.6417856659987, 3574.569792894903,
    3595.1213124737064, 3586.883086179849, 3595.5199482698463,
    3580.3322831017585, 3566.9230946324715, 3553.973997200766,
])

Harvested_energy_winter = np.array([
    3882.8349011599184, 3837.293431555212, 3835.3170898801413,
    3920.8231356691927, 3894.4475404226796, 3877.354349347581,
    3866.449677993952, 3823.584241825438, 3776.407331600007,
    3857.9808685943804, 3901.259988840484, 3837.4424388264115,
    3854.3690144848724, 3882.1749339676126, 3815.8288850220706,
    3828.9741072716774, 3862.909961559958, 3902.359874998539,
    3884.7796174373325, 3823.7237247604344, 3819.714150493076,
    3816.183538232225, 3835.5080109932615, 3816.5909621360556,
    3837.334554902968, 3766.4313610431277, 3819.2342893451278,
    3849.273720346089, 3878.2345580753763, 3862.5094765334616,
    3785.1661161161624, 3706.7962345387946, 3805.6533307471823,
    3784.1497821020316, 3724.5139847755213, 3792.576717510211,
    3741.669128362503, 3779.3356757828283, 3757.696117792752,
    3776.1303434499246, 3777.799610672603, 3745.1270236784367,
    3761.9465324564108, 3737.151432284734, 3707.6360414115748,
    3740.4737100200364, 3758.3556088429605, 3733.2355241181845,
    3730.580947514336, 3772.6296246640923, 3728.5877054118314,
    3683.5202696341644, 3690.252522097174, 3717.873422291475,
    3791.3408062642325, 3670.4284093365513, 3683.428556830756,
    3746.7250704655835, 3675.7214976087707, 3659.1246662885324,
])

Harvested_energy_spring = np.array([
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

# Manually switch which season is active by changing this assignment.
Harvested_energy = Harvested_energy_winter
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


def plot_results(harvested_energy, initial_power, operational_power, final_power,
                  battery_level, transmission_duration, filename="dwf_v2_results_winter.png"):
    '''Plot harvested energy, initial (raw, pre-reserve) power, operational
    (pre-pooling) power, final (pooled) power, and battery level over time
    (minutes).'''
    minutes = np.cumsum(transmission_duration) / 60.0

    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    axes[0].plot(minutes, harvested_energy, color="tab:green", marker=".")
    axes[0].set_ylabel("Harvested energy (J)")
    axes[0].set_title("Harvested Energy")

    axes[1].plot(minutes, initial_power, color="tab:purple", marker=".")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_title("Initial Power (raw harvested energy, before reserve set-aside)")

    axes[2].plot(minutes, operational_power, color="tab:orange", marker=".")
    axes[2].set_ylabel("Power (W)")
    axes[2].set_title("Operational Power (pre-pooling, from operation energy)")

    axes[3].plot(minutes, final_power, color="tab:blue", marker=".")
    axes[3].set_ylabel("Power (W)")
    axes[3].set_title("Final Power (after DWF pooling)")

    axes[4].plot(minutes, battery_level, color="tab:red", marker=".")
    axes[4].axhline(LOWER_BATTERY_LIMIT, color="black", linestyle="--", linewidth=1,
                     label=f"Reserve ({LOWER_BATTERY_LIMIT} J)")
    axes[4].set_ylabel("Battery level (J)")
    axes[4].set_title("Battery Level")
    axes[4].legend()
    axes[4].set_xlabel("Time (minutes)")

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

    initial_power = compute_power(Harvested_energy, Transmission_duration)
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

    plot_results(Harvested_energy, initial_power, operational_power, power, battery_level, Transmission_duration)

    return power


if __name__ == "__main__":
    main()