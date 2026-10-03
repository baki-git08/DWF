# slave_config.py  --  EDIT THIS FILE ONLY.   Units: seconds, watts, joules, volts.
#
# Pure data.  slave_load.py reads it; nothing else in the repo hard-codes a slave duration.
# Replacing a dummy with a measured value never needs a code change:
#   * every duration may be a scalar OR a list/array of length N (one value per slot),
#   * a sleep/retain time of None means "the rest of the slot" (tau - the active times).
# ON_SLOT durations are used in slots where jp.slave_schedule == 1, SILENT_SLOT durations where it is 0.

TIMING_IS_DUMMY = True        # flip to False once every duration below is a measured value

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
