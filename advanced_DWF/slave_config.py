# slave_config.py  --  EDIT THIS FILE ONLY.   Units: seconds, watts, joules, volts.
#
# Pure data.  slave_load.py reads it; nothing else in the repo hard-codes a slave duration.
# Replacing a dummy with a measured value never needs a code change:
#   * every duration may be a scalar OR a list/array of length N (one value per slot),
#   * a sleep/retain time of None means "the rest of the slot" (tau - the active times).
# ON_SLOT durations are used in slots where jp.slave_schedule == 1, SILENT_SLOT durations where it is 0.

TIMING_IS_DUMMY = False        # flip to False once every duration below is a measured value

TAU_S = 60.0                  # must equal MasterParams.tau; validated at attach time

# ---------------- REAL: datasheet powers (change only if the part changes) ----------------
POWERS = dict(
    camera_active_W=0.30,  camera_sleep_W=10e-6,            # energy equation uses 0.3 W (2.8 V x 100 mA = 0.28 W)
    mcu_active_W=0.060,    mcu_sleep_W=0.016e-3,            # stated 60 mW (3.3 V x 15 mA = 49.5 mW); 3.3 V x 5 uA
    memory_active_W=0.080, memory_retain_W=0.200e-3,        # stated 80 mW (3.3 V x 30 mA = 99 mW); 60 uA x 3.3 V
    mppt_quiescent_W=320e-9 * 3.0,                          # operating, at 3 V storage
    mppt_sleep_W=480e-9 * 3.0,                              # NOTE: sleep current > operating current looks backwards; kept as supplied
    mppt_in_V=3.3, mppt_in_I_max_A=0.100,                   # input rating -> 0.33 W
    mppt_eff=0.95,                                          # same value as the physical_chain.py demo
    v_store_min=2.2, v_store_max=5.2,                       # storage voltage range
    memory_MB=8.0,                                          # external memory size
)

# ---------------- DUMMY: durations per slot type (scalar, or array of length N) ----------------
# *_sleep_s / memory_retain_s = None  ->  remainder of the slot (tau - the active times).
ON_SLOT = dict(                      # a slot in which the slave transmits (schedule = 1)
    camera_active_s=0.05,   camera_sleep_s=None,     # DUMMY: capture, per image
    mcu_active_s=1.5,      mcu_sleep_s=None,        # DUMMY: capture + write + read/TX window
    memory_write_s=0.20,    memory_read_s=0.20,       # DUMMY
    memory_retain_s=None,
    mppt_active_s=0.50,    mppt_sleep_s=None,       # DUMMY: quiescent time (whole slot)
    tx_window_s=0.80,                                # DUMMY: time the THz TX is on (inside the read window); None = memory_read_s
)
SILENT_SLOT = dict(                  # a slot in which the slave does not transmit (schedule = 0)
    camera_active_s=0.0,   camera_sleep_s=None,
    mcu_active_s=0.0,      mcu_sleep_s=None,
    memory_write_s=0.0,    memory_read_s=0.0,
    memory_retain_s=None,
    mppt_active_s=0.0,     mppt_sleep_s=None,
    tx_window_s=0.0,
)

# ---------------- DUMMY: other assumptions ----------------
ETA_TX = 1.0                         # DUMMY: THz TX-stage efficiency (radiated / electrical); hbar_eff = ETA_TX * hbar
STORAGE_MARGIN = 2.0                 # DUMMY: window margin used by size_storage()

# ---------------- DUMMY: image / memory / link assumptions ----------------
# Used ONLY by derive_dummy_timing() (regenerates the ON_SLOT defaults) and by the standalone
# `python slave_load.py` placeholder TX energy.  Never on the solver path.
STANDALONE = dict(
    image_MB=2.0,                    # DUMMY: 13 MP JPEG, 1 MB = 1e6 bytes
    memory_MBps=1.0,                 # DUMMY: write and read throughput
    link_bps=10e6,                   # DUMMY: THz data-link bit rate
    p_tx_W=0.5,                      # DUMMY: electrical power of the THz TX stage
    t_capture_s=0.05,                 # DUMMY: wake + settle + exposure + readout, per capture
    images_per_on_slot=1,
)


# ==========================================================================================================
# STAGE 6: EVENT-BASED SLAVE (integrated_demo).  Read only by the "event" model
# (slave_load.event_profile, attach_slave_load(..., model="event")).  The blocks above stay the inputs of
# legacy_demo / physical_demo and are not used here.
#   Roles: master = harvest + UOWC energy sharing only (no data, theta1 = 0); slave = PV/MPPT/supercap,
#   transmits ONLY over the THz photomixer, one event (camera -> PSRAM -> THz burst) per scheduled slot.
# ==========================================================================================================

# ---------------- schedule / events ----------------
SLAVE_SCHEDULE = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
#   1 = one transmission EVENT released in that slot, 0 = sleep slot; repeats cyclically (7 events / 30 slots).
EVENTS_PER_SLOT_MAX = 1              # at most one event per slot
MAX_DELAY_SLOTS = 5                  # PLACEHOLDER: a refused event may run at most this many slots after its release

# ---------------- THz link (notebook values; REAL unless marked) ----------------
THZ_PRESET = "0.3THz"                # "0.3THz" -> 0.3 THz, 10/10 dBi ;  "0.1THz" -> 0.1 THz, 5/5 dBi
THZ_FREQS_HZ = {"0.1 THz": 0.1e12, "0.3 THz": 0.3e12}
THZ_PRESETS = {"0.3THz": "0.3 THz", "0.1THz": "0.1 THz"}            # preset -> THZ_FREQS_HZ / gain key
GAIN_DBI_BY_FREQ = {"0.1 THz": (5.0, 5.0), "0.3 THz": (10.0, 10.0)}  # (G_tx, G_rx) dBi
LEGACY_GAINS = False                 # True: 5/5 dBi at both frequencies (the earlier notebook outputs)
D_THZ_M = 1.0e-3                     # slave -> receiver THz distance [m] (0.5 mm is inside the Friis-invalid region)
NOISE_MODE = "thermal"               # receiver noise of the third-party receiver: k_B T F R_s
T_NOISE_K = 290.0
NOISE_FIGURE_DB = 10.0
R_s = 1e9                            # symbol rate [sym/s] = noise bandwidth
ALPHA_THZ = 12300.0                  # molecular extinction [1/m]
PT_W = 20e-3                         # photomixer radiated power used by the notebook tables [W]
BER_TARGET = 1e-6
PAYLOAD_BYTES = 8 * 1024 * 1024      # 8 MiB image per event
MODULATION = "16-QAM"                # t_tx = 8*PAYLOAD_BYTES / (R_s * bits_per_symbol) = 16.777216 ms

# ---------------- photomixer (THz source) ----------------
P_PM_RAD_MAX_W = 0.020               # radiated power cap (antenna 50-100 ohm)
P_PM_ELEC_W = 0.200                  # electrical power while transmitting at the cap
P_PM_RAD_MIN_W = 0.005               # radiated floor: a burst below it is refused
ETA_TX_PM = P_PM_RAD_MAX_W / P_PM_ELEC_W   # = 0.1, radiated / electrical  (legacy ETA_TX above is a different, DUMMY knob)
PM_DC_MODE = "proportional"          # "proportional": burst energy is a DWF variable in [x_min, x_cap]
#                                      "fixed_bias"  : every event costs P_PM_ELEC_W * t_tx (fixed, in c_i)

# ---------------- energy chain: UOWC -> PV -> MPPT -> supercap ----------------
D_UOWC_M = 15                         # master -> slave optical (energy) distance [m]; the link works over 1 - 5 m,
#                                      5 m = the far end (conservative).  NOT the THz distance (D_THZ_M, mm range).
UOWC_C_PER_M = 0.315                 # water attenuation at 450 nm [1/m] (given)
UOWC_C_PER_M_SWEEP = [0.315, 0.5, 0.8]   # attenuation values for integrated_demo_uowc_sweep [1/m]; each run sets
#                                      UOWC_C_PER_M to one value and writes to out_uowc_sweep/c_<value>/
UOWC_GEOM = 1.0                      # DUMMY: fraction of the laser beam the slave's PV array captures at D_UOWC_M
#                                      (beam divergence vs PV aperture).  1.0 = no spreading loss; at metres a real
#                                      beam spreads, so this is optimistic until a beam/aperture value is set.
ETA_LD = 0.4                         # DUMMY: laser electrical -> optical efficiency (inside alpha)
LASER_P_OPT_MAX_W = 1                # DUMMY: laser optical power cap [W]; None = M5 off
PV_CAL_WAVELENGTH_NM = 532           # wavelength at which beta is calibrated against physical_chain.py (legacy demos: 532)
MPPT_EFF = 0.95                      # MPPT conversion efficiency (0.90 - 0.95), inside beta
ETA_SC = 0.95                        # DUMMY: supercap storage efficiency, inside beta
SUPERCAP_F = 0.047                   # supercap [F] over POWERS v_store_min..v_store_max: B_min = C V_min^2/2, B_max = C V_max^2/2
PSRAM_SILENT_STATE = "standby"       # PSRAM state outside its active window: "standby" (data kept) or "dpd"
MAXIMAL_SHARING = True               # post-pass: master shares as early / as much as the supercap can take

# ---------------- master battery (integrated_demo) ----------------
#   Priority of the master's harvest each slot: the slave's minimum (just-in-time) need -> the master's own reserve
#   up to MASTER_FLOOR_J -> sharing (maximal-sharing post-pass, stopped by B_max / S5 / M5) -> stored up to
#   MASTER_E_MAX_J -> spilled.  Sized for the winter harvest (3.66 - 3.92 kJ per 60 s slot).
MASTER_E_MAX_J = 5000.0             # M2: master battery capacity [J]; float("inf") = no cap (the old behaviour)
MASTER_FLOOR_J = 1000.0              # M6: reserve the master keeps for itself [J]; None = off (master may end empty).
#                                      Must be <= MASTER_E_MAX_J - max harvest per slot (= 5000 - 3920.8 = 1079 J with the full winter harvest).

# ---------------- event timing (per EVENT, seconds; scalar or length-N array) ----------------
EVENT_DURATIONS = dict(
    t_cam=0.500,                     # camera readout
    t_write=0.0105,                  # PSRAM write, 8 MB at 200 MHz
    t_read=0.0105,                   # PSRAM read, inside the TX window
)                                    # t_tx is derived from PAYLOAD_BYTES, R_s and MODULATION (not set here)

# ---------------- component table (adding a component = adding a row) ----------------
#   energy in an event slot : active_W * t_active + sleep_W * (tau - t_active),  t_active = sum of active_terms
#   energy in a sleep slot  : sleep_W * tau
#   in_c = False : the ACTIVE term is not part of c_i (it is the DWF's burst energy); the sleep term still is.
#   sleep_W may be a dict keyed by PSRAM_SILENT_STATE.
EVENT_COMPONENTS = [
    dict(key="camera",     active_W=0.300,  active_terms=("t_cam",),                    sleep_W=10e-6),
    dict(key="mcu",        active_W=0.060,  active_terms=("t_cam", "t_write", "t_tx"),  sleep_W=16e-6),
    dict(key="flash",      active_W=0.0396, active_terms=("t_write", "t_read"),         # PSRAM (APS12808L)
         sleep_W={"standby": 237.6e-6, "dpd": 25.2e-6}),
    dict(key="mppt",       active_W=0.0,    active_terms=(),                            sleep_W=1.44e-6),
    #   MPPT: the 330 mW is its INPUT rating (S5 cap), not a load; only its sleep/quiescent draw is a load.
    dict(key="photomixer", active_W=P_PM_ELEC_W, active_terms=("t_tx",),                sleep_W=1e-6, in_c=False),
]

# Values that are not measured yet: listed in the banner of every event-model run.  Remove a name once real.
DUMMY_VALUES = ("ETA_SC", "ETA_LD", "LASER_P_OPT_MAX_W", "MAX_DELAY_SLOTS", "UOWC_GEOM")
