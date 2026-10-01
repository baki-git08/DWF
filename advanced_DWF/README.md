# Master + Slave Directional Water-Filling (DWF)

Context file for Claude Code. Read this before touching the code. It explains what the
project is, what the algorithm does, how it maps to the math, what has been verified, and what
is still open. Everything here reflects the state of the code as delivered; nothing is aspirational
unless it is under "Open items".

---------------------------------------------------------------------------------------------

## 1. Project goal

Build, step by step, a **directional water-filling (DWF)** power/energy-sharing algorithm for an
**underwater energy-harvesting pair**:

* **Master** (node 1) harvests energy `E_i` from the environment every slot, transmits data at
  power `P_i`, and **shares** energy `delta_i >= 0` with the slave.
* **Slave** (node 2) harvests **nothing**. Its only energy is what the master shares, reduced by
  two losses in series: channel attenuation `alpha_i` (Beer-Lambert: `alpha = exp(-c*d)`) and
  battery recharging efficiency `beta`. Only `a_i*delta_i` lands in its battery, `a_i = beta*alpha_i`.
* The slave draws on its **finite battery** (`B_min <= level <= B_max`) to send data back to the
  master at power `Pbar_i`.
* The master battery capacity `E_max` can be taken to infinity to recover the simple case.
* Two structural properties: **right taps** (a node's own energy flowing forward in time, i.e.
  temporal / energy causality) and **down taps** (master -> slave energy transfer, i.e. spatial).
  Terminology follows Gurakan et al. (right = temporal, down = spatial). The original brief had
  these swapped; the user did not object to using the paper's convention.

The work has been done in stages: (1) master alone, (2) master + energy sharing `delta`, (3) master +
slave with a finite battery window, (4) slave **transmission schedule** (the slave may only send data in
some slots) and a **physical receive-chain calibration** of the down tap (photodiode/GaAs cell + MPPT).
Stage 4 is the current state; `joint_dwf()` stays the single solver for all of it.

---------------------------------------------------------------------------------------------

## 2. Files (all in `advanced_DWF/`)

| File | What it is |
|---|---|
| `build_dwf.py` | Stage 1: master-only DWF. `MasterParams`, `directional_water_filling`, `check_kkt`, `reference_solution` (scipy), `plot_result`. Holds `Harvested_energy_summer` and `Harvested_energy_winter` (60 samples, J per 1-minute slot). |
| `build_2D_dwf.py` | Stages 3-4: joint master + slave DWF, slave schedule (S4), physical-chain post-processing and calibration. Imports `Harvested_energy_summer`/`Harvested_energy_winter` and `MasterParams` from `build_dwf.py`, `diagnose` from `dwf_diagnostics.py`, and `optical_to_electrical`, `OpticalReceiveChainResult` from `physical_chain.py`. Fully commented (module docstring is a reading guide). |
| `physical_chain.py` | Receive side of the optical energy link: optical power at the slave -> GaAs/Ge cell array (Spectrolab datasheet: Voc = 1.025 V, FF = 0.82, QE-derived responsivity at 450/532 nm) -> MPPT -> electrical power into the battery. `optical_to_electrical()`, `photodiode_stage()`, `mppt_stage()`, `size_array_by_intensity()`. Never modifies the solver. |
| `dwf_diagnostics.py` | `diagnose(jp, r)`: human-readable report (sharing decision, energy ledger, active constraints, solver status). Called by the demo. |
| `Instructions.md` | Design notes for replacing the linear slave-battery model with the physical chain: Option B (per-slot calibrated `beta`, **implemented**) and Option C (nonlinear tap inside the optimiser, **not implemented**). |
| `PD cell/` | Datasheets (photodiodes, APD, STM32L010, ODD-5W) used for the receive-chain parameters. |
| `plot_link_power.png`, `plot_transmit_power.png`, `plot_slave_battery.png`, `plot_throughput_per_slot.png`, `plot_throughput_cumulative.png`, `plot_harvested_energy.png`, `plot_sharing_energy.png`, `plot_physical_stored_power.png`, `plot_physical_slave_battery.png` | Output of the demo: one PNG per panel (see section 4's "Public API" for what each shows). Each function saves exactly one figure, so panels can be read, sized and shared independently. |
| `dwf_master_inf.png`, `dwf_master_3300.png` | Stage 1 plots (E_max = inf and 3300 J). |
| Reference papers (PDF, provided by the user) | Gurakan et al., *Energy Cooperation in Energy Harvesting Communications* (2-D DWF, right/down taps); Ozel et al., *Transmission with Energy Harvesting Nodes in Fading Wireless Channels* (and the INFOCOM'11 version); Ozel et al., *Optimal Broadcast Scheduling ... Finite Capacity Battery*; *Sum-rate optimal power policies ... interference channel*. |

`build_dwf.py` and `physical_chain.py` must not be modified as a side effect of solver work; `build_2D_dwf.py` only imports from them.

Dependencies: `numpy`, `matplotlib` (plots), plus the local modules above. `cvxpy` (with the CLARABEL solver) only for
`reference_solution_joint()`, which is a cross-check, not part of the algorithm.
`scipy` is used by the stage-1 `reference_solution()`.

Run the demo: `cd advanced_DWF && python build_2D_dwf.py` (prints the `diagnose()` report, the KKT and calibration summaries, and writes the nine `plot_*.png` files above).

---------------------------------------------------------------------------------------------

## 3. The problem being solved (stage 3)

Slots `i = 0..N-1`, slot length `tau` (60 s), energies in Joules, powers in W. Rates in nats.

```
max   sum_i (th1*tau/2) ln(1 + h_i * P_i)   +   sum_i (th2*tau/2) ln(1 + hbar_i * Pbar_i)
```

* `h_i`: master link gain over noise. `hbar_i`: slave return-link gain over noise.
* `th1, th2`: rate weights (1 = plain sum-rate).
* "Received power" in plots/results is in noise-normalised units (N0 = 1 W): `h*P` is the SNR
  inside the log. The energy-link power is `delta/tau` (sent), `alpha*delta/tau` (after channel),
  `beta*alpha*delta/tau` (stored).

### Variables and energy balance

Free variables (all Joules): master carry `F_k`, slave carry above the floor `Gt_k`, and `delta_k`.
The data energies are *defined* by the energy balance:

```
tau*P_i    = E_i + F_{i-1} - F_i - delta_i            (master)
tau*Pbar_i = a_i*delta_i + Gt_{i-1} - Gt_i            (slave)
F_{-1} = F_{N-1} = 0 ;  Gt_{-1} = b0 - B_min ,  Gt_{N-1} = 0
```

`Gt_k = b_k - B_min`, where `b_k` is the slave battery level at the end of slot k.

### Constraints (multipliers in brackets)

```
M1  F_n >= 0                               master causality            [lambda_n]
M2  F_n <= E_max - E_{n+1}                 master capacity at arrival  [mu_n]
M3  P_i >= 0, delta_i >= 0                                             [eta_i, rho_i]
S1  b_n >= B_min                           slave floor, end of slot    [lambda_bar_n]
S2  b_{n-1} + a_n*delta_n <= B_max         slave ceiling, at ARRIVAL   [mu_bar_{n-1}]
S3  Pbar_i >= 0                                                        [eta_bar_i]
S4  Pbar_i = 0 in every slot with schedule s_i = 0     (JointParams.slave_schedule)
```

**S4 (slave schedule).** `s_i in {0,1}`; a shorter pattern repeats cyclically (`[1,0]` = every other
slot). A silent slot (`s_i = 0`) still *receives* `delta_i` and stores it; it is a pure pass-through for
the slave battery (carry in = carry out + arrival). Consequence: energy sent in a silent slot has no local
marginal value; it is valued at the next transmitting slot, so the interior condition `m_k = a_k s_k` is
only tested at transmitting slots. The initial excess `b0 - B_min` can only be spent from the first ON
slot on. Validation: the schedule must be 0/1, and `b0 > B_min` with no ON slot is rejected.

S1 and S2 combine into one two-sided bound on the slave carry:
`B_min <= b_n <= B_max - a_{n+1}*delta_{n+1}`. The slave's right-tap capacity therefore depends on a
decision variable (`delta_{n+1}`); this is the only place the down tap feeds back into the slave's
temporal chain. Within a slot the level only falls after the arrival, so the floor is checked at the
end of the slot and the ceiling at the arrival.

### Stationarity (what the moves are derived from)

```
m_k = th1*h_k / (2(1 + h_k P_k))            master marginal value of a Joule   = Lambda_k      (P_k > 0)
s_k = th2*hbar_k / (2(1 + hbar_k Pbar_k))   slave  marginal value of a Joule   = Lambda_bar_k  (Pbar_k > 0)

P_k    = [ nu_k - 1/h_k ]^+ ,   nu_k = th1 / (2 Lambda_k)         (water level nu = P + 1/h)
Lambda_k     = sum_{n>=k} lambda_n - sum_{n>=k} mu_n              (master tap term)
Lambda_bar_k = sum_{n>=k} lambda_bar_n - sum_{n>=k} mu_bar_n      (slave tap term)
Gamma_k      = Lambda_bar_k - mu_bar_{k-1}                         (value of a Joule LANDING at slave slot k)

delta_k :  -Lambda_k + a_k * Gamma_k + rho_k = 0 ,  rho_k >= 0 ,  rho_k*delta_k = 0
           interior (delta_k > 0, arrival cap slack):   m_k = a_k * s_k
           water levels then satisfy:   nubar_k = (a_k*th2/th1) * nu_k
```

Right-tap behaviour (complementary slackness): a tap that is open (`lambda = mu = 0`) equalises the
two water levels; a closed tap (`F = 0`, battery empty / at floor) allows only a level *rise*
(`nu_{n+1} >= nu_n`); a full tap (`F = cap`) allows only a level *fall*. With `E_max = inf` all
`mu = 0`, so `nu` is non-decreasing in time.

Important facts that follow:

* With infinite batteries and a harvest that is high early and lower later (as in the summer array),
  no tap closes before the last slot, so `nu` (and `P`) is flat.
* `P[i] > P[i+1]` is **not** implied by `E[i] > E[i+1]`. It can occur only where the tap is full
  (battery ceiling reached). The rule is about water levels and battery state, not single harvests.
* With `S(delta) = 0` the master never shares (`rho_k = Lambda_k > 0`).
* Sharing only happens when `a_k*th2*hbar_k > th1*h_k / (1 + h_k P_k)`. With `hbar = h` the master
  shares nothing (verified).

---------------------------------------------------------------------------------------------

## 4. The algorithm (`joint_dwf`)

**Block coordinate ascent over "taps".** Not a closed form, not a solver: repeated exact 1-D
maximisations, each clipped to the feasible interval, so the objective never decreases.

Network picture (each arrow is a flow variable):

```
 master   M_0 --f--> M_1 --f--> M_2 --f--> ...      f[k+1] = F_k          (right taps, temporal)
           |          |          |
           | delta    | delta    | delta            d[k]   = delta_k      (down taps, spatial, gain a_k)
           v (x a_k)  v          v
 slave    S_0 --g--> S_1 --g--> S_2 --g--> ...      g[k+1] = Gt_k         (right taps, temporal)
```

### State in the code (0-indexed)

| Code | Math |
|---|---|
| `E[i]`, `h[i]`, `hb[i]`, `a[k]` | `E_i`, `h_i`, `hbar_i`, `a_k = beta*alpha_k` |
| `f[k+1]` | `F_k` master carry k -> k+1 (`f[0] = f[N] = 0` fixed) |
| `capM[k]` | `E_max - E_{k+1}` capacity of master right tap k |
| `g[k+1]` | `Gt_k` slave carry above floor (`g[0] = b0 - B_min`, `g[N] = 0` fixed) |
| `d[k]` | `delta_k` |
| `W` | `B_max - B_min` |
| `xm(i)` | `tau*P_i = E[i] + f[i] - f[i+1] - d[i]` |
| `xs(i)` | `tau*Pbar_i = a[i]*d[i] + g[i] - g[i+1]` |

Because the flows are the variables, the energy-balance equalities hold by construction and the
constraints become bounds: `f >= 0`, `f[k+1] <= capM[k]`, `g >= 0`, `d >= 0`,
`g[k] + a[k]*d[k] <= W` (slave arrival cap), `xm >= 0`, `xs >= 0`.

### Start point
`f = 0`, `g = 0` (except `g[0]`), `d = 0`: "spend what you harvest, share nothing". If the first slots are
silent (S4), the initial slave excess `g[0]` is carried unchanged up to the first ON slot so the start is feasible.

### The one equation: `step()`

Moving `t` Joules from node u (loses `s*t`) to node w (gains `r*t`) has derivative zero when the
marginal values balance: `th_u h_u (1 + h_w P_w) = r th_w h_w (1 + h_u P_u)`. This is linear in
`t`:

```
t* = tau * [ r*th_w*h_w*(1 + h_u*x_u/tau) - th_u*h_u*(1 + h_w*x_w/tau) ] / [ h_u*h_w*(th_u*r + r*th_w*s) ]
```

* Right taps (`s = r = 1`, same theta): reduces to `t* = 0.5*tau*(nu_u - nu_w)` (equalise levels).
* Down tap (`s = 1`, `r = a_k`): the interior condition `m_k = a_k s_k`.
* `step` returns 0 if the denominator is not positive (e.g. both weights 0).

The applied move is `t = clip(t*, lo, hi)`, where `[lo, hi]` keeps every constraint of section 3 valid.
A clipped move means that constraint is active, i.e. its multiplier is nonzero (complementary
slackness in code form). **Multipliers are never computed explicitly**; optimality is recognised by
"no move can improve".

### The ten moves at slot k (all inside `joint_dwf`, in this order)

Let `tx[i] = slave_schedule[i]`. Any move whose **slave end** is a silent slot is skipped (that end would
get `Pbar > 0`): move 1 needs `tx[k]`, 2 needs `tx[k-1]`, 3 and 4 need `tx[k]`, 5 needs `tx[k+1]`, 7 needs
`tx[k]` and `tx[k+1]`, 9 needs `tx[k-1]` and `tx[k+1]`. Moves 6, 8 and 10 only touch master data energy and
are never gated.

| # | Path (source -> sink) | Variables that move | Equation enforced |
|---|---|---|---|
| 1 | master k -> slave k (down tap) | `d[k] += t` | `m_k = a_k s_k` |
| 2 | master k -> slave k-1 (in-swap) | `d[k] += t`, `g[k] -= a_k t` | `m_k = a_k s_{k-1}` (used when slave arrival cap is tight) |
| 3 | master k-1 -> slave k (master pass-through) | `f[k] += t`, `d[k] += t` | `m_{k-1} = a_k s_k` |
| 4 | master k+1 -> slave k (master out-swap) | `f[k+1] -= t`, `d[k] += t` | `m_{k+1} = a_k s_k` |
| 5 | master k -> slave k+1 (slave pass-through) | `d[k] += t`, `g[k+1] += a_k t` | `m_k = a_k s_{k+1}` |
| 6 | master k -> master k+1 | `f[k+1] += t` | `nu_k = nu_{k+1}` |
| 7 | slave k -> slave k+1 | `g[k+1] += t` | `nubar_k = nubar_{k+1}` |
| 8 | master k-1 -> master k+1 (carry pass-through) | `f[k] += t`, `f[k+1] += t` | `nu_{k-1} = nu_{k+1}` |
| 9 | slave k-1 -> slave k+1 (carry pass-through) | `g[k] += t`, `g[k+1] += t` | `nubar_{k-1} = nubar_{k+1}` |
| 10 | master k -> master k+1 across slave right tap k (delta shift) | `d[k] += t`, `g[k+1] += a_k t`, `d[k+1] -= (a_k/a_{k+1}) t` | `nu_k = nu_{k+1}` (slave spending `xs(k)`, `xs(k+1)` unchanged) |

Moves 1, 6, 7 are the plain equations. Move 10 is new with the schedule: it **re-times when the master
sends** (e.g. moves `delta` out of a slot and into a later one through the slave battery) without changing what
the slave spends; `step()` is called with sink gain `r = a_k/a_{k+1}` and the move is applied through `apply()`.
Moves 2-5, 8, 9 are the same equations with two variables
moving together. They exist **because single-variable moves can get stuck**: a slot with zero power
(`P = 0` or `Pbar = 0`, a "dead" slot, water level sitting on the floor `1/h`) cannot supply energy,
so no single tap looks profitable although energy should pass *through* that slot; likewise the
slave arrival cap couples `delta_k` with the incoming slave carry. Moves 6-10 need `k+1`, so the last
slot skips them.

### `apply()` and `polish()`

* `apply(fch, gch, dch, src, snk, t_star)` is the generic path mover. Every constraint is linear in
  the flows, so each affected quantity is `v + c*t` and must stay in `[L, U]`; the tightest bounds
  form `[lo, hi]`; the move is `clip(t_star, lo, hi)`.
* `polish()` extends the pass-through idea to **runs of consecutive dead slots** (paths across two or
  more dead slots): master chains, slave chains, and cross paths master -> down tap -> slave.
  A slot is "dead" if its data energy `<= 1e-6*scale`. It is only called when the fast sweeps have
  stopped moving. Under S4 the slave chain is attempted only if both end slots transmit, and silent slots are
  excluded as slave sinks of the cross paths (they may still be crossed).

### Loop and stopping rule

1. One **sweep** = one pass over all slots trying moves 1-10 (forward on even sweeps, backward on odd).
2. If the largest move in the sweep is `< tol*scale` (`tol = 1e-10`, `scale = mean(E)`), call
   `polish()`. If `polish()` also moves less than `tol*scale`, the KKT conditions hold: converged.
3. `max_iter = 100000` sweeps is only a safety cap. The result reports `iters` and `converged`.

### Post-processing
`P = xm/tau`, `Pbar = xs/tau` (with `max(., 0)` only to remove round-off), `F = cumsum(E - x - delta)`,
`b = B_min + g[1:]`, `b_peak = B_min + g[:-1] + a*delta` (battery right after each arrival, must be
`<= B_max`), rates `sum tau/2 ln(1+hP)` and `sum tau/2 ln(1+hbar*Pbar)`.

### Public API of `build_2D_dwf.py`
`beer_lambert(c, d)`, `thz_absorption(freq_thz)`, `thz_gain(freq_thz, d, h0)`, `JointParams`
(now with `slave_schedule` and per-slot `beta`), `JointResult`, `PhysicalStorageResult`,
`joint_dwf(jp, tol, max_iter)`, `physical_stored_power(jp, r, ...)`, `solve_with_physical_chain(jp, ...)`,
`check_kkt_joint(jp, r)`, `reference_solution_joint(jp)` (cvxpy), and one plotting function per panel (each saves a single
PNG to `path`; `title` is an optional figure-level caption):

| Function | Panel |
|---|---|
| `plot_link_power(jp, r, path, title)` | power over the energy link: sent / received after channel / stored after recharge |
| `plot_transmit_power(jp, r, path, title)` | master vs slave transmit (data) power |
| `plot_slave_battery(jp, r, path, title)` | slave battery level after each slot and right after each arrival, with `B_min`/`B_max` |
| `plot_throughput_per_slot(jp, r, path, title)` | master vs slave throughput per slot [nats] |
| `plot_throughput_cumulative(jp, r, path, title)` | running total of that throughput |
| `plot_harvested_energy(jp, r, path, title)` | energy the master harvests per slot, `E_i` |
| `plot_sharing_energy(jp, r, path, title)` | energy balance of sharing: slave gain `a*delta`, master loss `delta`, net `a*delta - delta` |
| `plot_physical_stored_power(jp, r, ph, path, title)` | arrived optical power vs solver-assumed stored power vs physical MPPT output |
| `plot_physical_slave_battery(jp, r, ph, path, title)` | slave battery per the solver vs re-simulated with the physical chain's arrivals |

`plot_transmit_power` and `plot_throughput_per_slot` shade the silent slots (`_shade_silent`).
`check_kkt_joint` now also returns `slave_silent_slots` and `max_Pbar_in_silent_slots` (S4, should be ~0), and
tests the `delta` condition only at transmitting slots. `reference_solution_joint` adds `xb[silent] == 0`.
`plot_harvested_energy` and `plot_sharing_energy` used to be one combined plot (`plot_energy_balance`),
but `E_i` is normally one to three orders of magnitude larger than the sharing quantities, which
flattened the sharing curves to the baseline; they are now separate figures with independent scales.
Likewise the old three-panel `plot_joint` and two-panel `plot_throughput` were split so each panel is
its own figure (a shared `_new_axis()`/`_finish()` helper pair keeps the styling consistent).

### Physical receive chain (stage 4: post-processing + calibration)

`joint_dwf()` assumes a **linear** tap, `a_i = beta*alpha_i`. The real receive chain is slightly nonlinear
and its efficiency differs from the placeholder `beta`. Two functions bridge this without touching the solver
(Option B of `Instructions.md`):

* `physical_stored_power(jp, r, wavelength_nm=532, mppt_efficiency=0.95, ...)`: feeds `r.arrived_power`
  (`alpha*delta/tau`) through `optical_to_electrical()` and returns a `PhysicalStorageResult`: MPPT output
  power/energy, `eff_phys = P_mppt / P_arrived` (NaN where nothing arrives), the slave battery re-simulated
  with the physical arrivals (`b_phys`, `b_peak_phys`), `overflow` above `B_max`, and floor/ceiling violation
  counts (tolerance `battery_tol_J = 1e-6` J, looser than the solver's own residual). Pure post-processing.
* `solve_with_physical_chain(jp, ..., tol=1e-3, max_outer=30)`: fixed-point loop. Solve, measure `eff_phys`
  on slots where `delta` is non-negligible, set per-slot `beta_i = eff_phys_i`, re-solve, until `beta` moves
  less than `tol`. Slots with `delta = 0` keep their previous `beta`. At the fixed point
  `a_i*delta_i/tau == mppt_output_power_i`, so `r.b` equals `ph.b_phys` and B_min/B_max are enforced against
  the real energy. Returns `(jp_calibrated, r, ph, n_outer)`.

Option C (nonlinear tap inside the optimiser) is described in `Instructions.md` and is **not** built.

`thz_absorption` interpolates the measured molecular-absorption table (`thz_freq` [THz] against
`thz_alpha_m` [1/m], the latter converted from the given `thz_alpha_cm`). `thz_gain` turns that into
a channel gain via `h = h0 * exp(-alpha_m*d)`, the same shape as `beer_lambert`'s transmittance but
with a free reference constant `h0` (default now 10.0) standing in for whatever the absorption coefficient
alone does not capture.

---------------------------------------------------------------------------------------------

## 5. Stage 1 (master only, `build_dwf.py`) in brief

Same tap idea with a single node: flows `f_n`, `x_i = E_i + f_{i-1} - f_i`, one move per tap
(equalise `nu_i = P_i + 1/h_i`) clipped by `0 <= f <= E_max - E_{n+1}` and `x >= 0`. No pass-through
moves. Converges slowly for `E_max = inf` (about 5700 sweeps on the summer array) but exactly.
Summer array, `tau = 60`, `h = 1/50`: `E_max = inf` gives flat `P = 53.38 W`, throughput 1307.4416 nats;
`E_max = 3300 J` gives 1307.4255 nats. It matched an independent solver in these cases and on 8 random
per-slot-`h` cases. Its scipy SLSQP reference stalls for `E_max = inf` (use trust-constr or cvxpy).

---------------------------------------------------------------------------------------------

## 6. Placeholders and assumptions (NOT from the user's data)

The demo uses these; replace them when real values exist.

| Item | Value | Status |
|---|---|---|
| `B_min`, `B_max` | 20 J, 1500 J (earlier demos: 200/5000, then 200/3000) | chosen for the current demo; the slave battery is small enough that the ceiling binds |
| `E` | `Harvested_energy_winter` (J per 1-min slot; the summer array is still available) | given by the user |
| `LINK_DISTANCE_M` | 0.0005 m, shared by the THz data link and the optical energy link (was 0.001) | given by the user |
| `alpha` | `beer_lambert(0.046, 0.0005)`: 450 nm blue, c = 0.046 1/m (was 0.0399) | given by the user |
| `beta` | initial 0.4, then **per-slot, calibrated** by `solve_with_physical_chain()` to the photodiode+MPPT secant gain (about 0.380 to 0.400 in the demo) | the 0.4 start is a PLACEHOLDER; calibrated values come from `physical_chain.py` (532 nm, MPPT 0.95) |
| `h`, `hbar` (master/slave THz data link, symmetric) | `thz_gain(0.30, 0.0005)` = h0 * exp(-12300 * 0.0005) | alpha_m given by the user (0.30 THz, 12300 1/m); **h0 = 10.0 is a PLACEHOLDER** (default changed from 1.0) |
| `theta1`, `theta2` | 1 and **3** (default changed; slave rate weighted 3x) | assumption (weighted sum-rate) |
| `slave_schedule` | 30-slot pattern (13 silent, 4 ON, 4 silent, 3 ON, 1 silent, 3 ON, 2 silent) repeated over 60 slots: 20 ON, 40 silent | assumption / test pattern (slave transmits in 1/3 of the slots) |
| `E_max` (master) | inf | assumption (user said it can go to infinity) |
| `b0` (initial slave charge) | `B_min` | ASSUMPTION, asked, not yet answered |
| Receive chain | GaAs/Ge cell (Spectrolab datasheet), FF 0.82, Voc 1.025 V constant, array sized by intensity | datasheet numbers; Voc fall-off at low illumination is NOT modelled (see `physical_chain.py` docstring) |
| Arrival timing | energy sent in slot k is usable by the slave in slot k | ASSUMPTION (as in Gurakan; acoustic/optical delay negligible vs 60 s) |

The sharing test is `R = a*(th2/th1)*(hbar/h)*(1+h*P0)`. With `th2/th1 = 3` and `h0 = 10` it is about 2.75
(> 1), so sharing pays in the current demo, unlike the earlier `th2 = th1`, `h0 = 1` demo where `R < 1` forced
`delta = 0` everywhere. `h0` and `theta2` are therefore the knobs that decide whether anything is shared; do
not present demo numbers as results about a real system.

---------------------------------------------------------------------------------------------

## 7. Current results (demo: winter array, schedule, calibrated beta)

`iters = 1410`, converged, objective **1902.8194 nats** (master 1141.7442, slave 253.6917; objective =
th1*master + th2*slave). Master data power 41.58 W mean (min 40.91), slave data power 8.22 W mean (zero in
the 40 silent slots, about 7e-16 per `check_kkt_joint`), shared 21.65 W mean = **34.2% of the harvest**
(77945 J sent of 227633 J). Ledger: sent 77945 J -> channel loss 1.8 J, recharge loss 48344 J, stored 29600 J
= slave data 29600 J (battery change 0). All 20 transmitting slots hit the slave ceiling on arrival (`B_max`);
`delta_active_slots = 20`, slave at the floor in 57/60 slots, master taps closed in 1/59.
`check_kkt_joint`: final master carry 0, slave end level 20 J, `delta`-condition violations 0.
Calibration: 4 outer iterations, `beta` in [0.3798, 0.4000], max |solver battery - physical battery| =
3.9e-12 J, stored energy solver = physical = 29600 J, no floor/ceiling violations, overflow 0.

**The slave battery trajectory is not unique.** `(P, Pbar)` are unique (strict concavity), but the
split of carried energy between the master battery and the slave battery is not when both have slack
(same objective, different `F` and `b`). Do not write tests that pin `F` or `b` exactly unless a
tie-breaker is added.

---------------------------------------------------------------------------------------------

## 8. What has been verified

* **Reduction to master-only:** `theta2 = 0`, or `beta = 0`, gives the `build_dwf.py` result (max
  difference about 1e-8 W).
* **Independent solver (cvxpy, `reference_solution_joint`):** default, no-sharing (`hbar = h`) and a
  tight-battery case (window 200-900 J, `b0 = 500`, time-varying `alpha`, `hbar`, finite `E_max`) all
  match to about 1e-6 in objective.
* **Randomised comparison, 25 extreme cases** (per-slot random `h`, `hbar`, `alpha`, `E`, weights, tight
  batteries, finite `E_max`): 22 exact, 3 ended about 0.1% below optimum (gaps 0.3-1.9 of about 2000).
  Those need joint rerouting over several slots that no implemented move covers. Realistic smooth
  inputs (constant `h`, smooth `alpha`) have not shown this.
* **Invariants:** objective non-decreasing in `B_max` and saturating once `B_max` exceeds the peak battery
  level; `B_max = 5000` vs `1e6` identical; scaling `E, B` by c and `h, hbar` by 1/c scales `P` by c;
  flat harvest gives flat powers; energy accounting closes to about 1e-11 J on both nodes.
* **Published example (Gurakan Sec. V-B):** E = [0,12,0], slave harvest Ebar = [6,6,0], alpha = 1,
  theta1 = theta2, infinite batteries: optimum P = [0,4.8,4.8], Pbar = [4.8,4.8,4.8], total delta = 2.4
  (the split of delta across slots is not unique). A scratch copy of the solver (one-line change adding
  slave harvest) reaches this: P within 0.002 after 3 sweeps, converged in 10 sweeps. The paper's
  transient values from its tap orderings are *not* expected to match the sweeps (the Lagrangian gives
  the optimum, not a path).
* **Gurakan Sec. VIII-A relay example 1** as printed in the provided PDF is inconsistent (cumulative
  `P + delta` exceeds cumulative `E`); example 2 is consistent but is a relay-throughput problem with
  data causality, which this code does not model. Do not use example 1 as ground truth.
* **Schedule (S4):** the demo's `max_Pbar_in_silent_slots` is about 7e-16 and the `delta` condition holds at
  every transmitting slot. `reference_solution_joint` includes the S4 constraint, but a systematic cvxpy
  comparison over random schedules (and for move 10) has NOT been run yet.
* **Physical calibration:** at the fixed point the solver battery and the re-simulated physical battery agree to
  about 4e-12 J (section 7).
* Comment-only edits were verified by comparing the AST with docstrings stripped and re-running the demo (done for the stage-3 code; redo
  it against the current outputs, see section 10).

---------------------------------------------------------------------------------------------

## 9. Known limits and gotchas

* **Not guaranteed optimal on extreme inputs** (section 8). If an unusual configuration matters, check
  the objective against `reference_solution_joint`; there is no built-in optimality certificate other
  than `check_kkt_joint` (which only tests feasibility and the `delta` condition).
* **Speed:** pure-Python loops. About 10 s for N = 60 with `E_max = inf` (slow diffusion of energy along
  the master chain). Larger N or tighter `tol` will be slower; vectorising the sweep would be the fix.
* **No multipliers are returned.** They can be recovered from the water levels (`Lambda = th/(2 nu)`),
  but only for slots with positive power.
* **Slave has no harvest input** (`Ebar`); the published examples all have one. Adding it is small:
  `xs(i)` gains `+ Ebar[i]` and the arrival cap becomes `g[k] + a[k]*d[k] + Ebar[k] <= W` everywhere
  (`apply`, moves 1-5, 7, 9, `check_kkt_joint`, `reference_solution_joint`, `b_peak`).
* `check_kkt_joint(..., tol)`: `tol` is unused.
* `dwf_diagnostics.diagnose()` assumed a scalar `jp.beta` in a couple of print lines (noted in
  `Instructions.md`); beta is now per-slot after calibration, so check its output if it looks off.
* `solve_with_physical_chain` calibrates only slots where `delta` moves energy; other slots keep the initial
  `beta`, which is harmless because they carry no energy. Whether the secant gain is a good tap model at other
  operating points is untested (that is Option C's job).
* `B_min` only matters when `B_max` is tight or `b0` is small; with `B_max = 5000` the objective is
  identical for `B_min = 0` and `200`. Do not read that as a bug.
* Units: `delta` is **energy** (J), not power. `E` values are J per slot; `E/tau` is watts.
* Do not assume `P[i] > P[i+1]` follows from `E[i] > E[i+1]` (see section 3).
* The moves rely on `a[k] > 0`; slots with `a[k] = 0` skip moves 1-5.

---------------------------------------------------------------------------------------------

## 10. Conventions for changing the code

* **Functionality-preserving edits (comments/docstrings/renames):** verify with the AST check
  `ast.dump` of old vs new with docstrings removed, and re-run the demo (objective must stay
  1902.8194, `iters = 1410`, with the current demo settings of section 6).
* Keep 0-indexed slots and the `f[k+1]`, `g[k+1]` index convention; `f[0]`, `f[N]`, `g[N]` are fixed.
* Every move must remain "exact 1-D maximiser of a concave function, clipped to feasibility", so the
  objective can never decrease. A new move needs: effect on `xm`/`xs`, the marginal-balance equation,
  and every bound (which multiplier each corresponds to).
* Keep math in docstrings/comments as plain ASCII (`nu = P + 1/h`, `tau*P_i = E_i + F_{i-1} - F_i - delta_i`).
* Any change to the solver should be re-checked against `reference_solution_joint` on the default
  case, the no-sharing case, the tight-battery case and a batch of random cases; report failure rates,
  not just single successes.
* Minimal cross-check pattern:

```python
import numpy as np
from build_dwf import MasterParams, Harvested_energy_summer
from build_2D_dwf import JointParams, beer_lambert, joint_dwf, reference_solution_joint

m  = MasterParams(E=Harvested_energy_summer, E_max=np.inf, tau=60.0, h=1/50)
jp = JointParams(master=m, alpha=beer_lambert(0.2, 5.0), beta=0.8, B_min=200.0, B_max=5000.0, h_bar=0.1)
r  = joint_dwf(jp)
_, _, _, ref = reference_solution_joint(jp)
assert r.converged and abs(ref - r.objective) < 1e-5 * abs(ref)
```

---------------------------------------------------------------------------------------------

## 11. Open items (suggested next steps, none started unless noted)

1. Add an optional slave-harvest input `Ebar` and turn the Gurakan Sec. V-B example into a test
   (expected: P = [0,4.8,4.8], Pbar = [4.8,4.8,4.8], sum(delta) = 2.4).
2. Write a pytest suite from section 8 (reduction tests, cvxpy cross-checks over many seeds, invariants,
   monotonicity in `B_max`, `beta`, `alpha`, `E`; edge cases N = 1, zero-harvest slots, `max_iter` reached).
3. Replace placeholders with real `alpha_i` (Beer-Lambert with a water-type `c`; e.g. one survey table
   gives c at 520 nm of 0.15 pure sea, 0.25 clear ocean, 0.50 coastal, 1.50 turbid harbor 1/m; other
   sources differ by wavelength), real `h_i`, `hbar_i`, and decide `b0`.
4. Public data for realistic tests: NREL MIDC 1-minute solar irradiance (`E_i = G_i * A * eta * tau`),
   NSRDB / PVGIS hourly, CRAWDAD columbia/enhants light traces; Watermark / ACommSet for acoustic
   channel gains (not Beer-Lambert). None has been downloaded or used yet.
5. Load the other seasons' arrays ("similar arrays" to be supplied by the user).
6. Compute the minimum useful slave capacity `B*` (peak battery of the unconstrained optimum) and the
   sensitivity `d(objective)/d(B_max) = sum(mu_bar)`.
7. Optionally add a battery tie-breaker so `F` and `b` become unique; optionally a master floor `E_min`.
8. Extensions discussed but not built: one-slot energy-transfer delay, data-causality (relay) objective,
   max-min or master-only objectives, time-varying `c` per slot.
9. Performance: vectorise the sweeps.
10. Calibrate `h0` in `thz_gain` (or replace pure absorption with a full THz path-loss model:
    spreading loss + molecular absorption + antenna/beam gains). `h0 = 10` and `theta2 = 3` currently make
    sharing profitable; until `h0` is real, the THz/optical demo numbers (section 6) show the link model
    wired in correctly, not a realistic operating point.
11. Verify S4 and move 10 against `reference_solution_joint` over many random schedules (all ON, a single ON
    slot, leading/trailing silent runs, per-slot `beta`), reporting failure rates as in section 10.
12. Implement Option C of `Instructions.md` (nonlinear photodiode/MPPT tap inside the optimiser) if the
    secant-gain calibration proves inadequate; add Voc fall-off at low illumination to `physical_chain.py`.
13. The schedule is a fixed input. Choosing which slots the slave transmits in would be a combinatorial
    extension and is not attempted.

---------------------------------------------------------------------------------------------

## 12. Notation cheat sheet

`tau` slot length; `E_i` harvest; `E_max` master capacity; `P_i`, `Pbar_i` master/slave data power;
`delta_i` shared energy; `alpha_i = exp(-c d)`; `beta` recharge efficiency; `a_i = beta alpha_i`;
`h_i`, `hbar_i` link gains over noise; `th1`, `th2` rate weights; `F_n` master carry; `b_n` slave level;
`Gt_n = b_n - B_min`; `W = B_max - B_min`; `nu = P + 1/h` master water level; `nubar = Pbar + 1/hbar`;
`lambda_n, mu_n` master causality/capacity multipliers; `lambda_bar_n, mu_bar_n` slave floor/ceiling
multipliers; `eta, eta_bar, rho` non-negativity multipliers for `P, Pbar, delta`;
`Lambda_k` master tap term; `Lambda_bar_k` slave tap term; `Gamma_k = Lambda_bar_k - mu_bar_{k-1}`.
