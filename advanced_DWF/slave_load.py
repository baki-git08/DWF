"""
slave_load.py  --  electronics, consumption and energy profile of the SLAVE node.
(The task text calls this module slave_load_profile.py; the file in the repo is slave_load.py.)

Companion to the DWF solver.  Powers and durations come from slave_config.py (the only file the user
edits).  Same time grid as the README: N = 60 slots, tau = 60 s, same S4 slave schedule (1 = slave may
transmit, 0 = silent).

What it does
------------
1. Hardware model: camera (13 MP, JPEG), MCU, 8 MB external memory ("flash" in the CSV/plot keys),
   MPPT, and a THz transmit stage.  Every component has an ACTIVE and a SLEEP power;
   energy = power x time in state, with the times taken from slave_config.py (SlotTiming).
2. overhead_profile(): the NON-TX energy c_i = camera + mcu + memory + mppt of each slot.  This is the
   known per-slot drain the DWF solver sees (JointParams.slave_overhead_J).  The THz transmit stage is
   NOT in c_i: in the integrated flow its energy is the solver's slave data energy tau*Pbar_i.
3. attach_slave_load(jp, ...): returns a copy of a JointParams with the load, the MPPT input cap (S5)
   and the TX-stage efficiency attached.
4. Storage sizing: usable window W = B_max - B_min and supercap capacitance for the MPPT storage
   voltage range.
5. Standalone demo (python slave_load.py): a placeholder TX energy and a look-ahead arrival policy
   illustrate the numbers.  They are NOT used by the solver.

Energy conventions (ASCII):  E_slot = sum_state P_state * t_state ;  c_i = E_camera + E_mcu
+ E_memory + E_mppt ;  load_i = c_i + tau*Pbar_i ;  b_i = b_{i-1} + stored_i - load_i.

Every duration / link number is a DUMMY (slave_config.py).  Run:  python slave_load.py
"""
from __future__ import annotations

import csv
import dataclasses
import warnings
from dataclasses import dataclass, replace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import slave_config as cfg

TAU = cfg.TAU_S
OVERHEAD = ("camera", "mcu", "flash", "mppt")           # non-TX components, c_i = their sum
COMPONENTS = OVERHEAD + ("tx",)                          # "flash" = external memory (key kept for CSV/plots)

TIMING_FIELDS = ("camera_active_s", "camera_sleep_s", "mcu_active_s", "mcu_sleep_s",
                 "memory_write_s", "memory_read_s", "memory_retain_s",
                 "mppt_active_s", "mppt_sleep_s", "tx_window_s")
_REST = ("camera_sleep_s", "mcu_sleep_s", "memory_retain_s", "mppt_sleep_s")   # None -> rest of slot

_BANNER_SHOWN = False


def timing_banner() -> str:
    """One-line banner while slave_config.TIMING_IS_DUMMY is True, else ''."""
    return "slave timing: DUMMY values (edit slave_config.py)" if cfg.TIMING_IS_DUMMY else ""


def print_banner(once: bool = True) -> None:
    """Print the dummy-timing banner (once per process unless once=False)."""
    global _BANNER_SHOWN
    b = timing_banner()
    if b and not (once and _BANNER_SHOWN):
        print(b)
    _BANNER_SHOWN = _BANNER_SHOWN or bool(b)


@dataclass(frozen=True)
class SlaveHardware:
    """Powers (REAL, datasheet) and the standalone-only placeholders (DUMMY). All from slave_config."""
    cam_active_W: float = cfg.POWERS["camera_active_W"]
    cam_sleep_W: float = cfg.POWERS["camera_sleep_W"]
    mcu_active_W: float = cfg.POWERS["mcu_active_W"]
    mcu_sleep_W: float = cfg.POWERS["mcu_sleep_W"]
    flash_active_W: float = cfg.POWERS["memory_active_W"]
    flash_retain_W: float = cfg.POWERS["memory_retain_W"]
    flash_MB: float = cfg.POWERS["memory_MB"]
    mppt_in_V: float = cfg.POWERS["mppt_in_V"]
    mppt_in_I_max_A: float = cfg.POWERS["mppt_in_I_max_A"]
    mppt_quiescent_W: float = cfg.POWERS["mppt_quiescent_W"]
    mppt_sleep_W: float = cfg.POWERS["mppt_sleep_W"]
    mppt_eff: float = cfg.POWERS["mppt_eff"]
    v_store_min: float = cfg.POWERS["v_store_min"]
    v_store_max: float = cfg.POWERS["v_store_max"]
    # --- DUMMY, standalone only (placeholder TX energy and sensitivity study) ---
    image_MB: float = cfg.STANDALONE["image_MB"]
    flash_rate_MBps: float = cfg.STANDALONE["memory_MBps"]
    link_rate_bps: float = cfg.STANDALONE["link_bps"]
    p_tx_W: float = cfg.STANDALONE["p_tx_W"]
    t_capture_s: float = cfg.STANDALONE["t_capture_s"]
    images_per_on_slot: int = cfg.STANDALONE["images_per_on_slot"]

    @property
    def mppt_cap_J(self) -> float:
        """S5: stored energy the MPPT can accept in one slot = V_in * I_max * eta * tau  (18.81 J at defaults)."""
        return self.mppt_in_V * self.mppt_in_I_max_A * self.mppt_eff * TAU


# ----------------------------------------------------------------------------------------
# datasheet consistency
# ----------------------------------------------------------------------------------------
def datasheet_check() -> list[tuple[str, float, float]]:
    """(name, stated W, V x I W).  Differences are reported, not silently fixed."""
    return [
        ("camera active", 0.30, 2.8 * 0.100),
        ("MCU active", 0.060, 3.3 * 0.015),
        ("memory active", 0.080, 3.3 * 0.030),
        ("MCU sleep", 0.016e-3, 3.3 * 5e-6),
        ("memory retention", 0.200e-3, 60e-6 * 3.3),
    ]


# ----------------------------------------------------------------------------------------
# timing: SlotTiming + loader with validation
# ----------------------------------------------------------------------------------------
@dataclass
class SlotTiming:
    """Resolved per-slot durations [s]: every field in TIMING_FIELDS is an array of length N (the ON
    values in slots where is_on, the SILENT values elsewhere).  Sleep/retain times are already
    resolved ('rest of the slot' filled in)."""
    is_on: np.ndarray
    tau: float
    is_dummy: bool
    camera_active_s: np.ndarray
    camera_sleep_s: np.ndarray
    mcu_active_s: np.ndarray
    mcu_sleep_s: np.ndarray
    memory_write_s: np.ndarray
    memory_read_s: np.ndarray
    memory_retain_s: np.ndarray
    mppt_active_s: np.ndarray
    mppt_sleep_s: np.ndarray
    tx_window_s: np.ndarray

    @property
    def N(self) -> int:
        return len(self.is_on)


def _as_slot_array(kind: str, name: str, val, n: int) -> np.ndarray:
    """Scalar or length-n sequence -> float array (None -> NaN = 'rest of slot')."""
    if val is None:
        return np.full(n, np.nan)
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    arr = arr.ravel()
    if arr.size != n:
        raise ValueError(f"{kind}.{name}: array has {arr.size} entries, need 1 or N = {n}")
    return arr.copy()


def load_timing(schedule, on_slot: dict | None = None, silent_slot: dict | None = None,
                tau: float | None = None, quiet: bool = False) -> SlotTiming:
    """Read the durations from slave_config.py (or the given dicts), broadcast, resolve 'rest of
    slot', validate.  Raises ValueError naming the field and the slot index.

    schedule : 0/1 per slot (a shorter pattern repeats cyclically only via overhead_profile's n_slots).
    """
    sch = np.asarray(schedule).astype(bool).ravel()
    n = sch.size
    tau = cfg.TAU_S if tau is None else float(tau)
    on_d = dict(cfg.ON_SLOT if on_slot is None else on_slot)
    si_d = dict(cfg.SILENT_SLOT if silent_slot is None else silent_slot)
    res: dict[str, np.ndarray] = {}
    for name in TIMING_FIELDS:
        for kind, d in (("ON_SLOT", on_d), ("SILENT_SLOT", si_d)):
            if name not in d:
                raise ValueError(f"{kind} is missing the field {name!r}")
        res[name] = np.where(sch, _as_slot_array("ON_SLOT", name, on_d[name], n),
                             _as_slot_array("SILENT_SLOT", name, si_d[name], n))

    def where(i):
        return f"{'ON_SLOT' if sch[i] else 'SILENT_SLOT'}, slot {i}"

    for name in TIMING_FIELDS:
        v = res[name]
        bad = np.flatnonzero(~np.isnan(v) & ((v < 0) | ~np.isfinite(v)))
        if bad.size:
            raise ValueError(f"{name} must be >= 0 and finite ({where(bad[0])}: {v[bad[0]]})")
    # tx window defaults to the memory read window
    tw = res["tx_window_s"]
    res["tx_window_s"] = np.where(np.isnan(tw), res["memory_read_s"], tw)
    for name in ("camera_active_s", "mcu_active_s", "memory_write_s", "memory_read_s", "mppt_active_s"):
        if np.any(np.isnan(res[name])):
            raise ValueError(f"{name} cannot be None (only sleep/retain times may be 'rest of slot')")
    tol = 1e-12 * max(tau, 1.0)
    used = {"camera_sleep_s": res["camera_active_s"], "mcu_sleep_s": res["mcu_active_s"],
            "memory_retain_s": res["memory_write_s"] + res["memory_read_s"],
            "mppt_sleep_s": res["mppt_active_s"]}
    pair = {"camera_sleep_s": "camera_active_s + camera_sleep_s", "mcu_sleep_s": "mcu_active_s + mcu_sleep_s",
            "memory_retain_s": "memory_write_s + memory_read_s + memory_retain_s",
            "mppt_sleep_s": "mppt_active_s + mppt_sleep_s"}
    off_note = []
    for name in _REST:
        act = used[name]
        over = np.flatnonzero(act > tau + tol)
        if over.size:
            raise ValueError(f"{pair[name]} exceeds tau = {tau:g} s ({where(over[0])}: active part {act[over[0]]:g} s)")
        res[name] = np.where(np.isnan(res[name]), tau - act, res[name])
        tot = act + res[name]
        over = np.flatnonzero(tot > tau + tol)
        if over.size:
            raise ValueError(f"{pair[name]} = {tot[over[0]]:g} s exceeds tau = {tau:g} s ({where(over[0])})")
        short = tot < tau - tol
        if short.any():
            off_note.append(f"{name.split('_')[0]} in {int(short.sum())} slot(s)")
    # the TX can only be on while the MCU is awake
    over = np.flatnonzero(res["tx_window_s"] > res["mcu_active_s"] + tol)
    if over.size:
        raise ValueError(f"tx_window_s ({res['tx_window_s'][over[0]]:g} s) exceeds mcu_active_s "
                         f"({res['mcu_active_s'][over[0]]:g} s) ({where(over[0])})")
    if off_note and not quiet:
        print("slave timing: explicit sleep times leave the remainder of the slot powered off (0 W): "
              + ", ".join(off_note))
    return SlotTiming(is_on=sch, tau=tau, is_dummy=bool(cfg.TIMING_IS_DUMMY),
                      **{k: res[k] for k in TIMING_FIELDS})


def _silent_scalars() -> dict:
    """Scalar silent-slot timing (all active/window times 0, sleep = rest of slot): used only by the dummy
    generator / sensitivity study, which model ONE slot and must not depend on per-slot arrays in the config."""
    return {k: (None if k in _REST else 0.0) for k in TIMING_FIELDS if k != "mppt_active_s"} | {"mppt_active_s": 0.0}


def derive_dummy_timing(image_MB: float, memory_MBps: float, link_bps: float,
                        t_capture_s: float = cfg.STANDALONE["t_capture_s"], images: int = 1,
                        memory_MB: float = cfg.POWERS["memory_MB"]) -> tuple[dict, dict]:
    """Dummy generator ONLY (regenerates slave_config.ON_SLOT from image / memory / link assumptions;
    never on the solver path).  Returns (ON_SLOT dict, SILENT_SLOT dict)."""
    if images * image_MB > memory_MB:
        raise ValueError("images x image_MB exceeds the external memory")
    t_cap = images * t_capture_s
    t_wr = images * image_MB / memory_MBps
    t_rd = t_wr
    t_air = images * image_MB * 8e6 / link_bps
    t_tx = max(t_rd, t_air)                              # memory read is streamed into the TX
    t_mcu = t_cap + t_wr + t_tx
    if t_mcu > cfg.TAU_S:
        raise ValueError("cycle longer than one slot")
    on = dict(camera_active_s=t_cap, camera_sleep_s=None, mcu_active_s=t_mcu, mcu_sleep_s=None,
              memory_write_s=t_wr, memory_read_s=t_rd, memory_retain_s=None,
              mppt_active_s=cfg.TAU_S, mppt_sleep_s=None, tx_window_s=t_tx)
    return on, _silent_scalars()


# ----------------------------------------------------------------------------------------
# energies
# ----------------------------------------------------------------------------------------
def slot_energy(hw: SlaveHardware, timing: SlotTiming, i: int,
                tx_energy_J: float | None = None) -> dict[str, float]:
    """Energy [J] per component in slot i:  P_active*t_active + P_sleep*t_sleep
    (memory: P_active*(t_write + t_read) + P_retain*t_retain).  TX: the given energy, else the
    standalone placeholder p_tx_W * tx_window (ON slots only)."""
    t = timing
    tx = 0.0
    if t.is_on[i]:
        tx = hw.p_tx_W * float(t.tx_window_s[i]) if tx_energy_J is None else float(tx_energy_J)
    return dict(
        camera=hw.cam_active_W * t.camera_active_s[i] + hw.cam_sleep_W * t.camera_sleep_s[i],
        mcu=hw.mcu_active_W * t.mcu_active_s[i] + hw.mcu_sleep_W * t.mcu_sleep_s[i],
        flash=hw.flash_active_W * (t.memory_write_s[i] + t.memory_read_s[i]) + hw.flash_retain_W * t.memory_retain_s[i],
        mppt=hw.mppt_quiescent_W * t.mppt_active_s[i] + hw.mppt_sleep_W * t.mppt_sleep_s[i],
        tx=tx,
    )


def overhead_profile(timing: SlotTiming | None, schedule, n_slots: int | None = None,
                     hw: SlaveHardware | None = None) -> dict:
    """Per-slot NON-TX energy [J]: arrays of length N for camera, mcu, flash, mppt and
    c = camera + mcu + flash + mppt.  No TX here.

    timing   : a resolved SlotTiming, or None to read slave_config.py.
    schedule : 0/1 per slot; n_slots (default len(schedule)) repeats a shorter pattern cyclically.
    """
    hw = hw or SlaveHardware()
    sch = np.asarray(schedule).astype(bool).ravel()
    if n_slots is not None and sch.size != n_slots:
        sch = np.resize(sch, n_slots)
    if timing is None:
        timing = load_timing(sch)
    if timing.N != sch.size or not np.array_equal(timing.is_on, sch):
        raise ValueError("timing was resolved for a different schedule / length than the one given")
    comp = {k: np.zeros(sch.size) for k in OVERHEAD}
    for i in range(sch.size):
        e = slot_energy(hw, timing, i)
        for k in OVERHEAD:
            comp[k][i] = e[k]
    comp["c"] = sum(comp[k] for k in OVERHEAD)
    comp["timing"] = timing
    return comp


def build_profile(hw: SlaveHardware, schedule, Pbar=None, timing: SlotTiming | None = None) -> dict[str, np.ndarray]:
    """Per-slot energy of each component.  With Pbar (W, per slot, from the solver) tx = tau*Pbar_i in
    every slot and load = c + tx.  Without Pbar the placeholder TX energy is used (standalone only)."""
    sch = np.asarray(schedule, dtype=int)
    prof = overhead_profile(timing, sch, hw=hw)
    comp = {k: prof[k] for k in OVERHEAD}
    t = prof["timing"]
    if Pbar is not None:
        comp["tx"] = np.asarray(Pbar, dtype=float) * t.tau
    else:
        comp["tx"] = np.array([slot_energy(hw, t, i)["tx"] for i in range(sch.size)])
    comp["c"] = prof["c"]
    comp["load"] = comp["c"] + comp["tx"]
    return comp


# ----------------------------------------------------------------------------------------
# storage sizing, and the hook into the solver
# ----------------------------------------------------------------------------------------
def silent_run_drain(load: np.ndarray, sch: np.ndarray, i: int) -> float:
    """Energy drawn in the run of silent slots that starts right after slot i."""
    j, e = i + 1, 0.0
    while j < len(sch) and not sch[j]:
        e += load[j]
        j += 1
    return e


def size_storage(hw: SlaveHardware, load: np.ndarray, schedule, margin: float | None = None) -> dict[str, float]:
    margin = cfg.STORAGE_MARGIN if margin is None else margin
    sch = np.asarray(schedule, dtype=int)
    on_peak = float(load[sch == 1].max()) if sch.any() else 0.0
    drains = [silent_run_drain(load, sch, i) for i in range(len(sch)) if sch[i]]
    lead = 0.0
    for i in range(len(sch)):
        if sch[i]:
            break
        lead += load[i]
    drain_max = max(max(drains, default=0.0), lead)
    W_req = on_peak + drain_max
    W = margin * W_req
    span = hw.v_store_max ** 2 - hw.v_store_min ** 2
    C = 2.0 * W / span
    std = [0.047, 0.1, 0.22, 0.47, 1.0, 1.5, 2.2, 3.3, 5.0, 10.0]
    C_std = next((c for c in std if c >= C), std[-1])
    return dict(on_peak=on_peak, drain_max=drain_max, W_req=W_req, W_design=W, C_req=C, C_std=C_std,
                W_std=0.5 * C_std * span, lead=lead)


def make_load(jp, timing: SlotTiming | None = None, hw: SlaveHardware | None = None) -> dict:
    """Reporting context for plots / CSV / diagnose(): components, c, and the sizing summary.
    Uses jp.slave_schedule (never slave_schedule_demo).  Sizing uses the placeholder TX energy (the
    true TX energy is the solver's output); flagged in the dict."""
    hw = hw or SlaveHardware()
    sch = jp.slave_schedule
    prof = overhead_profile(timing, sch, hw=hw)
    placeholder = build_profile(hw, sch, timing=prof["timing"])
    sizing = size_storage(hw, placeholder["load"], sch)
    return dict(prof, hw=hw, sizing=sizing, schedule=np.asarray(sch, dtype=int),
                timing_is_dummy=bool(cfg.TIMING_IS_DUMMY), banner=timing_banner(), tau=jp.master.tau)


def attach_slave_load(jp, timing: SlotTiming | None = None, *, size_storage: bool = False,
                      use_mppt_cap: bool = True, eta_tx: float | None = None,
                      hw: SlaveHardware | None = None, model: str = "slot"):
    """Return a COPY of jp with the slave electronics attached.

    model = "slot" (default): the stage-5 per-slot model below (legacy_demo / physical_demo).
    model = "event": the stage-6 event-based slave, see attach_event_load() (timing / size_storage / use_mppt_cap /
            eta_tx do not apply there; everything comes from the STAGE 6 block of slave_config.py).

    Slot model:

        slave_overhead_J = c  (camera + MCU + memory + MPPT; no TX)
        mppt_cap_J       = V_in * I_max * eta * tau   (S5), or None if use_mppt_cap is False
        h_bar            = eta_tx * h_bar             (radiated = eta_tx * electrical TX power)
        size_storage=True: B_max = B_min + W_std and b0 - B_min = leading-silent-run drain
                           (from the placeholder-TX sizing of the profile).

    The schedule is read from jp.slave_schedule.  Not idempotent: attaching twice scales h_bar twice.
    """
    if model == "event":
        return attach_event_load(jp, hw=hw)
    if model != "slot":
        raise ValueError(f"model must be 'slot' or 'event', got {model!r}")
    hw = hw or SlaveHardware()
    n = jp.master.N
    if abs(cfg.TAU_S - jp.master.tau) > 1e-9:
        raise ValueError(f"slave_config.TAU_S = {cfg.TAU_S:g} s but MasterParams.tau = {jp.master.tau:g} s")
    sch = jp.slave_schedule
    if len(sch) != n:
        raise ValueError("slave_schedule length differs from N")
    print_banner()
    prof = overhead_profile(timing, sch, hw=hw)
    t = prof["timing"]
    eta = cfg.ETA_TX if eta_tx is None else float(eta_tx)
    if eta <= 0:
        raise ValueError("eta_tx must be > 0")
    on = t.is_on
    if on.any() and np.any(t.tx_window_s[on] < t.tau - 1e-9):
        warnings.warn(
            f"TX window ({t.tx_window_s[on].min():g}..{t.tx_window_s[on].max():g} s) is shorter than tau = {t.tau:g} s: "
            "the DWF credits tau/2 ln(1+hbar*Pbar) with Pbar averaged over the whole slot; a real burst "
            "radiates xs/t_tx for t_tx < tau and, by concavity, delivers fewer nats than credited.",
            stacklevel=2)
    kw = dict(slave_overhead_J=prof["c"].copy(),
              mppt_cap_J=hw.mppt_cap_J if use_mppt_cap else None,
              h_bar=np.asarray(jp.h_bar, float) * eta)
    if size_storage:
        placeholder = build_profile(hw, sch, timing=t)
        sz = globals()["size_storage"](hw, placeholder["load"], sch)
        kw["B_max"] = jp.B_min + sz["W_std"]
        kw["b0"] = jp.B_min + sz["lead"]
    return replace(jp, **kw)


# ----------------------------------------------------------------------------------------
# STAGE 6: event-based slave model (slave_config.py "STAGE 6" block; used by integrated_demo)
# ----------------------------------------------------------------------------------------
#   An EVENT = camera readout -> PSRAM write -> PSRAM read + THz burst, released by SLAVE_SCHEDULE.
#   Durations are per event (not per slot).  For slot i with the EFFECTIVE schedule s_i (after refusals):
#       event slot : c_i = sum_comp [P_active * t_active + P_sleep * (tau - t_active)]   (photomixer active term excluded)
#       sleep slot : c_i = sum_comp  P_sleep * tau
#   The photomixer's active energy is the DWF burst variable x_i = tau*Pbar_i in [x_min, x_cap] ("proportional"),
#   or the fixed P_PM_ELEC_W * t_tx added to c_i ("fixed_bias").
EVENT_REQUIRED = ("key", "active_W", "active_terms", "sleep_W")


def dummy_banner() -> str:
    """One line naming the event-model values that are still DUMMY / PLACEHOLDER (slave_config.DUMMY_VALUES)."""
    names = tuple(getattr(cfg, "DUMMY_VALUES", ()))
    return ("slave model: DUMMY values in use (edit slave_config.py): " + ", ".join(names)) if names else ""


def bits_per_symbol(modulation: str | None = None) -> int:
    """Bits per symbol of slave_config.MODULATION (16-QAM -> 4)."""
    from dwf2d.channels import QAM_SPLIT
    m = cfg.MODULATION if modulation is None else modulation
    if m not in QAM_SPLIT:
        raise ValueError(f"MODULATION {m!r} is not one of {sorted(QAM_SPLIT)}")
    return sum(QAM_SPLIT[m])


def tx_time_s(payload_bytes: float | None = None, r_s: float | None = None, modulation: str | None = None) -> float:
    """THz burst duration t_tx = payload_bits / (R_s * bits_per_symbol)  [s]  (16.777216 ms for 8 MiB, 1 Gsps, 16-QAM)."""
    payload = cfg.PAYLOAD_BYTES if payload_bytes is None else payload_bytes
    rs = cfg.R_s if r_s is None else r_s
    if payload <= 0 or rs <= 0:
        raise ValueError("PAYLOAD_BYTES and R_s must be > 0")
    return payload * 8 / (rs * bits_per_symbol(modulation))


def tx_bounds_J(t_tx: float | None = None) -> dict[str, float]:
    """Electrical burst-energy bounds of one event [J]:
        x_cap = P_PM_ELEC_W * t_tx                    (20 mW radiated, 3.3554 mJ)
        x_min = (P_PM_RAD_MIN_W / ETA_TX_PM) * t_tx   (1 mW radiated,  0.16777 mJ)"""
    t = tx_time_s() if t_tx is None else t_tx
    return dict(t_tx=t, x_cap=cfg.P_PM_ELEC_W * t, x_min=cfg.P_PM_RAD_MIN_W / cfg.ETA_TX_PM * t)


def event_profile(schedule, tau: float | None = None, psram_state: str | None = None,
                  components: list | None = None, durations: dict | None = None,
                  pm_mode: str | None = None, n_slots: int | None = None) -> dict:
    """Per-slot NON-TX energy [J] of the event model, one array per component plus c = their sum.

    schedule  : effective 0/1 per slot (1 = an event runs there); n_slots repeats a shorter pattern cyclically.
    pm_mode   : "proportional" (default from config): the photomixer's active energy is NOT in c (it is x_i);
                "fixed_bias": it is, as P_active * t_tx per event.
    Raises ValueError naming the field and slot for a negative / non-finite duration, a per-slot array of the
    wrong length, or an active time longer than tau.
    Returns dict(<component key>: array, c, overhead_keys, is_event, t_active, durations, t_tx, x_cap, x_min,
                 tau, psram_state, pm_mode, pm_fixed_J).
    """
    sch = np.asarray(schedule).ravel()
    if sch.size == 0 or not np.all((sch == 0) | (sch == 1)):
        raise ValueError("event schedule must be a non-empty array of 0s and 1s (EVENTS_PER_SLOT_MAX = 1)")
    if getattr(cfg, "EVENTS_PER_SLOT_MAX", 1) != 1:
        raise ValueError("only EVENTS_PER_SLOT_MAX = 1 is modelled")
    if n_slots is not None and sch.size != n_slots:
        sch = np.resize(sch, n_slots)
    sch = sch.astype(bool)
    n = sch.size
    tau = cfg.TAU_S if tau is None else float(tau)
    state = cfg.PSRAM_SILENT_STATE if psram_state is None else psram_state
    mode = cfg.PM_DC_MODE if pm_mode is None else pm_mode
    if mode not in ("proportional", "fixed_bias"):
        raise ValueError(f"PM_DC_MODE must be 'proportional' or 'fixed_bias', got {mode!r}")
    comps = cfg.EVENT_COMPONENTS if components is None else components
    tb = tx_bounds_J()
    dur = {k: _as_slot_array("EVENT_DURATIONS", k, v, n) for k, v in dict(cfg.EVENT_DURATIONS if durations is None else durations).items()}
    if "t_tx" in dur:
        raise ValueError("EVENT_DURATIONS.t_tx is derived from PAYLOAD_BYTES, R_s and MODULATION; do not set it")
    dur["t_tx"] = np.full(n, tb["t_tx"])
    for k, v in dur.items():
        bad = np.flatnonzero(~np.isfinite(v) | (v < 0))
        if bad.size:
            raise ValueError(f"EVENT_DURATIONS.{k} must be >= 0 and finite (slot {bad[0]}: {v[bad[0]]})")
    tol = 1e-12 * max(tau, 1.0)
    out: dict = {}
    t_active: dict = {}
    keys = []
    for row in comps:
        miss = [f for f in EVENT_REQUIRED if f not in row]
        if miss:
            raise ValueError(f"EVENT_COMPONENTS row {row.get('key', '?')!r} is missing {miss}")
        key = row["key"]
        for term in row["active_terms"]:
            if term not in dur:
                raise ValueError(f"EVENT_COMPONENTS.{key}: unknown duration {term!r} (known: {sorted(dur)})")
        ta = sum((dur[t] for t in row["active_terms"]), np.zeros(n))
        over = np.flatnonzero(ta > tau + tol)
        if over.size:
            raise ValueError(f"EVENT_COMPONENTS.{key}: active time {ta[over[0]]:g} s exceeds tau = {tau:g} s (slot {over[0]})")
        p_s = row["sleep_W"]
        if isinstance(p_s, dict):
            if state not in p_s:
                raise ValueError(f"EVENT_COMPONENTS.{key}: sleep_W has no entry for PSRAM_SILENT_STATE = {state!r}")
            p_s = p_s[state]
        p_a = float(row["active_W"])
        in_c = bool(row.get("in_c", True))
        active_in_c = in_c or mode == "fixed_bias"
        e_event = (p_a * ta if active_in_c else 0.0) + p_s * (tau - ta)
        out[key] = np.where(sch, e_event, p_s * tau)
        t_active[key] = np.where(sch, ta, 0.0)
        keys.append(key)
    out["c"] = sum(out[k] for k in keys)
    pm = next((r for r in comps if not r.get("in_c", True)), None)
    out.update(overhead_keys=tuple(keys), is_event=sch, t_active=t_active, durations=dur, tau=tau,
               psram_state=state, pm_mode=mode,
               pm_fixed_J=(float(pm["active_W"]) * tb["t_tx"] if (pm is not None and mode == "fixed_bias") else 0.0),
               **tb)
    return out


def thz_link(preset: str | None = None, d_m: float | None = None, legacy_gains: bool | None = None) -> dict:
    """The slave's THz link from slave_config (THZ_PRESET, D_THZ_M, gains, noise, ETA_TX_PM):
    dwf2d.channels.thz_hbar() plus the labels.  hbar [1/W] maps the ELECTRICAL burst power to SNR and already
    contains ETA_TX_PM.  Returns dict(preset, freq_label, freq_hz, hbar, d_m, d_eff, d_nf, clamped, coupling_db,
    g_tx_dbi, g_rx_dbi, noise_w, eta_tx)."""
    from dwf2d import channels as ch
    p = cfg.THZ_PRESET if preset is None else preset
    if p not in cfg.THZ_PRESETS:
        raise ValueError(f"THZ_PRESET {p!r} is not one of {sorted(cfg.THZ_PRESETS)}")
    label = cfg.THZ_PRESETS[p]
    f = cfg.THZ_FREQS_HZ[label]
    leg = cfg.LEGACY_GAINS if legacy_gains is None else legacy_gains
    g_tx, g_rx = ch.LEGACY_GAIN_DBI if leg else cfg.GAIN_DBI_BY_FREQ[label]
    d = cfg.D_THZ_M if d_m is None else d_m
    n_w = ch.noise_floor_w(cfg.NOISE_MODE, cfg.T_NOISE_K, cfg.NOISE_FIGURE_DB, cfg.R_s)
    res = ch.thz_hbar(f, d, g_tx, g_rx, eta_tx=cfg.ETA_TX_PM, noise_w=n_w, alpha=cfg.ALPHA_THZ)
    return dict(res, preset=p, freq_label=label, freq_hz=f, d_m=d, eta_tx=cfg.ETA_TX_PM)


def supercap_window(c_f: float | None = None, hw: "SlaveHardware | None" = None) -> tuple[float, float]:
    """(B_min, B_max) [J] of a supercap of c_f farads over POWERS v_store_min..v_store_max:  E = C V^2 / 2."""
    hw = hw or SlaveHardware()
    c = cfg.SUPERCAP_F if c_f is None else float(c_f)
    if c <= 0:
        raise ValueError("SUPERCAP_F must be > 0")
    return 0.5 * c * hw.v_store_min ** 2, 0.5 * c * hw.v_store_max ** 2


def attach_event_load(jp, *, link: dict | None = None, pm_mode: str | None = None, psram_state: str | None = None,
                      supercap_f: float | None = None, hw: SlaveHardware | None = None):
    """Return a COPY of jp with the stage-6 event-based slave attached (attach_slave_load(..., model="event")).

    The EFFECTIVE event schedule is jp.slave_schedule (1 = an event runs in that slot).  Sets
        slave_overhead_J  = c_i of event_profile()  (non-TX; fixed_bias adds P_PM_ELEC_W*t_tx per event)
        mppt_cap_J        = mppt_in_V * mppt_in_I_max * MPPT_EFF * tau        (S5, 18.81 J)
        h_bar             = thz_link()["hbar"]  (already contains ETA_TX_PM: scaled ONCE, here)
        tx_cap_J          = x_cap in event slots, 0 elsewhere  ("proportional");  0 everywhere ("fixed_bias")
        slave_rate_window = t_tx                                                (burst rate)
        delta_cap_J       = LASER_P_OPT_MAX_W * tau / ETA_LD   (M5; None when LASER_P_OPT_MAX_W is None)
        B_min, B_max      = supercap_window(SUPERCAP_F),  b0 = B_min
    The x_min floor is NOT a bound: an event below it is refused (physical.solve_with_refusals).
    Idempotent (every field is set from slave_config, nothing is scaled in place).
    """
    hw = hw or SlaveHardware()
    if abs(cfg.TAU_S - jp.master.tau) > 1e-9:
        raise ValueError(f"slave_config.TAU_S = {cfg.TAU_S:g} s but MasterParams.tau = {jp.master.tau:g} s")
    sch = np.asarray(jp.slave_schedule, dtype=int)
    prof = event_profile(sch, jp.master.tau, psram_state=psram_state, pm_mode=pm_mode)
    lk = thz_link() if link is None else link
    b_min, b_max = supercap_window(supercap_f, hw)
    tau = jp.master.tau
    fixed = prof["pm_mode"] == "fixed_bias"
    kw = dict(slave_overhead_J=prof["c"].copy(),
              mppt_cap_J=hw.mppt_in_V * hw.mppt_in_I_max_A * cfg.MPPT_EFF * tau,
              h_bar=float(lk["hbar"]),
              tx_cap_J=np.zeros(sch.size) if fixed else np.where(sch == 1, prof["x_cap"], 0.0),
              slave_rate_window=prof["t_tx"],
              delta_cap_J=(None if cfg.LASER_P_OPT_MAX_W is None else cfg.LASER_P_OPT_MAX_W * tau / cfg.ETA_LD),
              B_min=b_min, B_max=b_max, b0=b_min)
    return replace(jp, **kw)


def make_event_load(jp, hw: SlaveHardware | None = None, pm_mode: str | None = None,
                    psram_state: str | None = None) -> dict:
    """Reporting context of the event model (plots / CSV / link report), like make_load() for the slot model:
    event_profile() of jp.slave_schedule (the EFFECTIVE schedule) + the supercap sizing with every burst at x_cap:
        W_req = largest event-slot load (c + x_cap) + longest sleep-run drain   (size_storage, STORAGE_MARGIN for W_design)
    plus the DUMMY banner (slave_config.DUMMY_VALUES)."""
    hw = hw or SlaveHardware()
    sch = np.asarray(jp.slave_schedule, dtype=int)
    prof = event_profile(sch, jp.master.tau, psram_state=psram_state, pm_mode=pm_mode)
    burst = 0.0 if prof["pm_mode"] == "fixed_bias" else prof["x_cap"]
    sizing = size_storage(hw, prof["c"] + burst * sch, sch)
    return dict(prof, hw=hw, sizing=sizing, schedule=sch, timing_is_dummy=bool(getattr(cfg, "DUMMY_VALUES", ())),
                banner=dummy_banner(), tau=jp.master.tau)


@dataclass
class SlaveState:
    """Run-time state of one slave (indexed by slave id; one slave for now).

    level_J  : supercap level [J]           pending : FIFO of (release_slot, age_slots) waiting to run
    counters : released / run / rescheduled / dropped events"""
    slave_id: int = 0
    level_J: float = 0.0
    pending: list = dataclasses.field(default_factory=list)
    released: int = 0
    run: int = 0
    rescheduled: int = 0
    dropped: int = 0


def new_slave_states(n_slaves: int = 1, level_J: float = 0.0) -> dict[int, SlaveState]:
    """{slave_id: SlaveState} for n_slaves slaves (arrays elsewhere are sized from the schedule length)."""
    return {i: SlaveState(slave_id=i, level_J=level_J) for i in range(n_slaves)}


def reschedule(effective, refused, release_of: dict, max_delay: int | None = None):
    """Move refused events to the next available slot (greedy; mirrors the hardware: wake, check, refuse).

    effective  : 0/1 per slot (1 = an event runs there), modified copy returned
    refused    : run slots whose event is refused this pass (processed in ascending order)
    release_of : {run_slot: release_slot} for every event currently placed (updated copy returned)
    For each refused slot i: effective[i] = 0 (nothing powers up there); the event goes to the FIRST slot
    j > i with effective[j] == 0 (no event of its own) and j - release <= max_delay; else it is dropped.
    Returns (effective, release_of, moves [(from, to, release)], dropped [(from, release)]).
    """
    eff = np.asarray(effective, dtype=int).copy()
    rel = dict(release_of)
    md = cfg.MAX_DELAY_SLOTS if max_delay is None else int(max_delay)
    moves, dropped = [], []
    for i in sorted(int(k) for k in refused):
        if not eff[i]:
            raise ValueError(f"slot {i} is refused but carries no event")
        r0 = rel.pop(i)
        eff[i] = 0
        j = next((j for j in range(i + 1, min(len(eff), r0 + md + 1)) if eff[j] == 0), None)
        if j is None:
            dropped.append((i, r0))
        else:
            eff[j] = 1
            rel[j] = r0
            moves.append((i, j, r0))
    return eff, rel, moves, dropped


# ----------------------------------------------------------------------------------------
# standalone illustration (NOT used by the solver)
# ----------------------------------------------------------------------------------------
# DEMO-ONLY constants for the scale comparison at the end of the standalone run.
DEMO_SLAVE_DATA_J = 29600.0
DEMO_SENT_J = 77945.0
DEMO_HARVEST_J = 227633.0
DEMO_ON_SLOTS = 20


def cycle_times(timing: SlotTiming) -> dict[str, float]:
    """Durations of the first ON slot (for the standalone printout)."""
    i = int(np.argmax(timing.is_on))
    return dict(cap=float(timing.camera_active_s[i]), wr=float(timing.memory_write_s[i]),
                tx=float(timing.tx_window_s[i]), mcu=float(timing.mcu_active_s[i]))


def slave_schedule_demo(n_slots: int = 60) -> np.ndarray:
    """README S4 test pattern: 13 off, 4 on, 4 off, 3 on, 1 off, 3 on, 2 off (30 slots), repeated."""
    pat = [0] * 13 + [1] * 4 + [0] * 4 + [1] * 3 + [0] * 1 + [1] * 3 + [0] * 2
    return np.array([pat[i % len(pat)] for i in range(n_slots)], dtype=int)


def arrival_profile(hw: SlaveHardware, load: np.ndarray, schedule, a) -> dict[str, np.ndarray]:
    """Look-ahead sharing policy (standalone illustration only; in the integrated flow the solver
    chooses delta).  stored_i covers that slot's load, plus (last ON slot of a run) the following
    silent run.  Limited by the MPPT input rating."""
    sch = np.asarray(schedule, dtype=int)
    N = len(sch)
    a = np.broadcast_to(np.asarray(a, dtype=float), (N,))
    cap_J = hw.mppt_cap_J
    stored = np.zeros(N)
    for i in range(N):
        if sch[i]:
            need = load[i]
            if i == N - 1 or not sch[i + 1]:
                need += silent_run_drain(load, sch, i)
            stored[i] = min(need, cap_J)
    delta = stored / a
    lead = 0.0                                   # initial charge covers the leading silent run
    j = 0
    while j < N and not sch[j]:
        lead += load[j]
        j += 1
    b = np.zeros(N + 1)
    b[0] = lead
    for i in range(N):
        b[i + 1] = b[i] + stored[i] - load[i]
    return dict(stored=stored, delta=delta, b=b, b0=lead, cap_J=cap_J,
                b_peak=b[:-1] + stored)


def sensitivity(hw: SlaveHardware) -> list[tuple[str, float]]:
    """Energy of one ON slot [J] under single-parameter changes (rebuilds the dummy timing)."""
    def e(h):
        on, si = derive_dummy_timing(h.image_MB, h.flash_rate_MBps, h.link_rate_bps, h.t_capture_s,
                                     h.images_per_on_slot, h.flash_MB)
        t = load_timing([1], on_slot=on, silent_slot=si, quiet=True)
        return sum(slot_energy(h, t, 0).values())
    rows = [("baseline", e(hw))]
    for name, h in [
        ("TX power 0.1 W", replace(hw, p_tx_W=0.1)),
        ("TX power 2 W", replace(hw, p_tx_W=2.0)),
        ("image 1 MB", replace(hw, image_MB=1.0)),
        ("image 4 MB", replace(hw, image_MB=4.0)),
        ("link 1 Mbit/s", replace(hw, link_rate_bps=1e6)),
        ("link 100 Mbit/s", replace(hw, link_rate_bps=100e6)),
        ("memory 10 MB/s", replace(hw, flash_rate_MBps=10.0)),
        ("memory 0.25 MB/s", replace(hw, flash_rate_MBps=0.25)),
        ("V x I powers (49.5/99 mW, 0.28 W)",
         replace(hw, cam_active_W=0.28, mcu_active_W=0.0495, flash_active_W=0.099)),
    ]:
        rows.append((name, e(h)))
    return rows


# ----------------------------------------------------------------------------------------
# standalone plots (one figure per panel, like build_2D_dwf.py)
# ----------------------------------------------------------------------------------------
def _fig(title: str, xlabel: str = "slot", ylabel: str = "energy [J]"):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _shade_silent(ax, sch):
    for i, s in enumerate(sch):
        if not s:
            ax.axvspan(i - 0.5, i + 0.5, color="0.9", zorder=0)


def plot_load_per_slot(comp, sch, path):
    fig, ax = _fig("Slave load per slot by component (silent slots are about 13.5 mJ, not visible at this scale)")
    x = np.arange(len(sch))
    bottom = np.zeros(len(sch))
    for k in COMPONENTS:
        ax.bar(x, comp[k], bottom=bottom, label=k, zorder=2)
        bottom += comp[k]
    _shade_silent(ax, sch)
    ax.legend(ncol=5, fontsize=8)
    _save(fig, path)


def plot_breakdown(comp, path):
    fig, ax = _fig("Slave energy over the horizon by component", xlabel="energy [J]", ylabel="")
    tot = {k: comp[k].sum() for k in COMPONENTS}
    T = sum(tot.values())
    ks = list(COMPONENTS)
    ax.barh(ks, [tot[k] for k in ks], zorder=2)
    for i, k in enumerate(ks):
        ax.text(tot[k], i, f"  {tot[k]:.2f} J ({100 * tot[k] / T:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, 1.25 * max(tot.values()))
    _save(fig, path)


def plot_cumulative(comp, arr, sch, path):
    fig, ax = _fig("Cumulative energy: consumed by the slave vs stored from the energy link")
    x = np.arange(1, len(sch) + 1)
    ax.step(x, np.cumsum(comp["load"]), where="post", label="consumed (load)")
    ax.step(x, np.cumsum(arr["stored"]) + arr["b0"], where="post", label="stored + initial charge")
    _shade_silent(ax, sch)
    ax.legend()
    _save(fig, path)


def plot_battery_window(arr, sizing, sch, path):
    fig, ax = _fig("Slave storage: usable energy above the floor (b - B_min)")
    ax.step(np.arange(len(arr["b"])), arr["b"], where="post", label="end of slot")
    ax.step(np.arange(len(arr["b_peak"])), arr["b_peak"], where="post", alpha=0.6, label="right after arrival")
    ax.axhline(sizing["W_req"], ls="--", color="C2", label=f"W_req = {sizing['W_req']:.2f} J")
    ax.axhline(sizing["W_design"], ls="--", color="C3", label=f"W_design = {sizing['W_design']:.2f} J")
    _shade_silent(ax, sch)
    ax.legend(fontsize=8)
    _save(fig, path)


CSV_HEADER = ["slot", "on", *[f"{k}_J" for k in COMPONENTS], "load_J", "stored_J", "delta_sent_J", "battery_end_J"]


def write_csv(path, comp, arr, sch):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for i in range(len(sch)):
            w.writerow([i, int(sch[i]), *[f"{comp[k][i]:.6g}" for k in COMPONENTS],
                        f"{comp['load'][i]:.6g}", f"{arr['stored'][i]:.6g}",
                        f"{arr['delta'][i]:.6g}", f"{arr['b'][i + 1]:.6g}"])


# ----------------------------------------------------------------------------------------
# standalone demo
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    print_banner()
    hw = SlaveHardware()
    sch = slave_schedule_demo(60)
    N = len(sch)
    timing = load_timing(sch)
    beta = 0.38                               # 532 nm GaAs + MPPT 0.95 (physical_chain.py doc)
    alpha = np.exp(-0.046 * 5)                # README: c = 0.046 1/m, d = 5 m
    a = beta * alpha

    print("== datasheet consistency (stated vs V x I) ==")
    for name, stated, vi in datasheet_check():
        print(f"  {name:18s} stated {stated * 1e3:8.3f} mW   V x I {vi * 1e3:8.3f} mW   "
              f"({100 * (vi - stated) / stated:+.1f}%)")

    t = cycle_times(timing)
    print("\n== ON-slot cycle (ASSUMPTIONS: image %.1f MB, memory %.1f MB/s, link %.0f Mbit/s, TX %.2f W) =="
          % (hw.image_MB, hw.flash_rate_MBps, hw.link_rate_bps / 1e6, hw.p_tx_W))
    print("  capture %.2f s | write %.2f s | read+TX window %.2f s | MCU awake %.2f s of %.0f s"
          % (t["cap"], t["wr"], t["tx"], t["mcu"], TAU))
    sl = slot_energy(hw, timing, int(np.argmin(timing.is_on)))
    on = slot_energy(hw, timing, int(np.argmax(timing.is_on)))
    print("  silent slot: %.3f mJ  (floor %.1f uW; memory retention is %.0f%% of it)"
          % (1e3 * sum(sl.values()), 1e6 * sum(sl.values()) / TAU, 100 * sl["flash"] / sum(sl.values())))
    print("  ON slot    : " + ", ".join(f"{k} {v:.3f} J" for k, v in on.items())
          + f"  | total {sum(on.values()):.3f} J")

    comp = build_profile(hw, sch, timing=timing)
    arr = arrival_profile(hw, comp["load"], sch, a)
    sz = size_storage(hw, comp["load"], sch)

    tot = comp["load"].sum()
    print("\n== horizon (%d slots = %.0f min, %d ON) ==" % (N, N * TAU / 60, sch.sum()))
    for k in COMPONENTS:
        print(f"  {k:7s} {comp[k].sum():8.3f} J  {100 * comp[k].sum() / tot:5.1f}%")
    print(f"  total   {tot:8.3f} J   mean power {1e3 * tot / (N * TAU):.2f} mW   peak ON-slot power "
          f"{1e3 * comp['load'].max() / TAU:.1f} mW (slot mean)")
    peak_inst = hw.cam_active_W + hw.mcu_active_W + hw.flash_active_W + hw.p_tx_W
    print(f"  instantaneous worst case (all active) {peak_inst:.2f} W -> {peak_inst / 3.0:.2f} A at 3 V")
    nimg = sch.sum() * hw.images_per_on_slot
    print(f"  images {nimg} | energy/image {tot / nimg:.3f} J | {tot / (nimg * hw.image_MB * 8e6) * 1e6:.3f} uJ/bit")

    print("\n== arrival through the MPPT ==")
    print(f"  MPPT stored-energy cap per slot: {arr['cap_J']:.2f} J  ({hw.mppt_in_V * hw.mppt_in_I_max_A:.2f} W in x "
          f"{hw.mppt_eff} x {TAU:.0f} s); max per-slot arrival used {arr['stored'].max():.3f} J")
    print(f"  a = beta*alpha = {a:.4f};  master must send sum(delta) = {arr['delta'].sum():.1f} J "
          f"(max {arr['delta'].max():.2f} J in one slot)")
    ledger_err = arr["b0"] + arr["stored"].sum() - tot - arr["b"][-1]
    print(f"  ledger: b0 + stored - load - b_end = {ledger_err:.2e} J ;  min usable level {arr['b'].min():.4f} J ;"
          f" max {arr['b_peak'].max():.3f} J")

    print("\n== storage ==")
    print(f"  largest ON-slot load {sz['on_peak']:.3f} J + longest silent drain {sz['drain_max']:.3f} J "
          f"-> W_req {sz['W_req']:.3f} J ; design (x{cfg.STORAGE_MARGIN:g}) {sz['W_design']:.3f} J")
    print(f"  supercap over {hw.v_store_min}-{hw.v_store_max} V: C >= {sz['C_req']:.3f} F -> standard {sz['C_std']} F "
          f"(window {sz['W_std']:.2f} J)")
    print(f"  DWF settings suggested: B_max - B_min = {sz['W_std']:.2f} J, b0 - B_min = {arr['b0']:.3f} J")

    print("\n== sensitivity of one ON slot [J] ==")
    for name, e in sensitivity(hw):
        print(f"  {name:36s} {e:7.3f}")

    print("\n== scale check against the README stage-4 demo ==")
    demo_p = DEMO_SLAVE_DATA_J / DEMO_ON_SLOTS / TAU
    print(f"  demo slave data energy {DEMO_SLAVE_DATA_J:.0f} J vs this load {tot:.1f} J  (x{DEMO_SLAVE_DATA_J / tot:.0f})")
    print(f"  demo stored power in an ON slot {demo_p:.1f} W vs MPPT stored-power rating "
          f"{hw.mppt_in_V * hw.mppt_in_I_max_A * hw.mppt_eff:.3f} W  (x{demo_p / (hw.mppt_in_V * hw.mppt_in_I_max_A * hw.mppt_eff):.0f})")
    print(f"  demo sent {DEMO_SENT_J:.0f} J ({100 * DEMO_SENT_J / DEMO_HARVEST_J:.1f}% of harvest) vs "
          f"{arr['delta'].sum():.0f} J here ({100 * arr['delta'].sum() / DEMO_HARVEST_J:.3f}% of the demo harvest)")

    plot_load_per_slot(comp, sch, "plot_slave_load_per_slot.png")
    plot_breakdown(comp, "plot_slave_energy_breakdown.png")
    plot_cumulative(comp, arr, sch, "plot_slave_cumulative_energy.png")
    plot_battery_window(arr, sz, sch, "plot_slave_battery_window.png")
    write_csv("slave_energy_profile.csv", comp, arr, sch)
    print("\nwrote 4 PNG files and slave_energy_profile.csv")
