# Wuhan RTC v0.6.9 — TFV-first methodology testbed with temporally continuous control

This repository implements a **model-based urban-drainage real-time-control methodology test** on an idealized, simplified Wuhan SWMM network. It is deliberately **not** a field-calibrated Wuhan digital twin and does not certify field actuator capability.

v0.6.9 keeps the v0.6.7 physical/rainfall testbed and v0.6.8 simulation-asset lineage, and freezes the runtime semantics that must not drift during later Codex runs:

```text
model/observation step                 = 5 min
supervisory control update             = 10 min
causal Step1 history                   = 13 frames = 0,5,...,60 min
first Proposed control                 = simulation elapsed 60 min
prediction horizon                     = 360 min = 72 model steps = 36 control blocks
effective pre-rain warm-up             = 120 min
max supervisory setting change         = 0.5 per 10-min update
post-rain Formal evaluation tail       = 360 min
```

The 120-min effective warm-up and the 60-min first-control epoch are **different clocks**. For the Formal prepared event, SWMM starts at `t=0`, Proposed has a complete causal history at `t=60 min` and may issue its first command then, while positive design rainfall begins at `t=120 min`.

```text
causal sparse observations + realised rainfall + actuator readback
                              |
                              v
              Step1 current-state reconstruction
                              |
                              v
          Step2 differentiable hydraulic/action model
                              |
                              v
        360-min continuous receding-horizon MPC
                              |
                  execute first 10-min move
                              |
                              v
                       authoritative SWMM
                              |
                 observe actual readback
                              |
                        re-solve at t+10
```

## Scientific objective and claim

Primary objective: minimize authoritative-SWMM system-wide cumulative **Total Flood Volume (TFV)**. PFV at the eight frozen priority nodes is a soft secondary/diagnostic quantity. Priority depth and Global Peak are report-only.

The intended claim is:

```text
IDEALIZED_METHODOLOGY_TESTBED_NOT_FIELD_DIGITAL_TWIN
```

The supplied DWF is retained as **idealized background loading**. No claim is made that the DWF, continuous actuator capability or controller timing represents a field-calibrated Wuhan installation.

## Active physical/rainfall inputs

The source physical/rainfall bundle remains v0.6.7:

- 932 hydraulic nodes;
- 109 writable actuators;
- 57 pump curves migrated `PUMP2 -> PUMP4`, endpoint-preserving;
- 42 continuously modelled orifices with 10-min full-travel assumption;
- 41 FREE outfalls;
- five retrofit storage curves and directional flap gates;
- supplied DWF retained;
- source SUBAREAS/infiltration retained.

The design rainfall library remains exactly 30 Chicago events:

```text
return periods: 5, 10, 20, 50, 100 years
durations:      60, 120, 180, 240, 300, 360 min
Chicago r:      0.39
rain step:      5 min
spatial mode:   one uniform design gage
```

Wuhan DB4201/T 641-2020 formula:

```text
i = 9.686 * (1 + 0.887 * log10(P)) / (t + 11.23)^0.658   [mm/min]
```

## Event clock and effective warm-up

The v0.6.7 source bundle carries a 60-min pre-rain prefix. Production/formal events must be prepared to a **total effective 120-min pre-rain prefix**:

```powershell
rtc-prepare-event-suite `
  --events E:\RTC_sewer\Project7\inputs\contracts\events_with_splits.csv `
  --out-dir E:\RTC_sewer\Project7\study_v069\prepared_events\events `
  --out-registry E:\RTC_sewer\Project7\study_v069\prepared_events\events_with_splits.csv `
  --target-effective-warmup-minutes 120 `
  --post-rain-tail-minutes 360
```

For the current source bundle, preparation adds only the missing 60 min:

```text
source_pre_rain_prefix_minutes = 60
additional_warmup_minutes      = 60
effective_warmup_minutes       = 120
```

The production runner independently inspects the prepared INP and refuses a Formal Project7 policy run if the actual positive-rainfall onset is not 120 min after SWMM START.

The 360-min post-rain tail is the common Formal evaluation window. Longer END extensions remain Phase-0 diagnostics only and never redefine Final TFV.

## Temporal control continuity — mandatory

A real-time controller must evolve actuator commands through time; opening a new MPC horizon must never reset a facility to its original/default state.

v0.6.9 enforces three layers:

1. **Between decision epochs** — the last issued `target_setting` is held until the next 10-min supervisory decision.
2. **Inside every 360-min horizon** — all 36 control blocks form one sequential path. Block 0 is bounded from the actual current readback; every later block is bounded from the preceding block by `|Δsetting| <= 0.5`.
3. **Across rolling horizons** — the next executed first move must be feasible relative to both the current SWMM `current_setting` **and the previous issued supervisory target**. A device that is still travelling toward a target therefore cannot be abruptly commanded back because a new horizon was solved.

Reversal is allowed when hydraulically useful, but it must happen progressively. For example, a target may evolve `1.0 -> 0.5 -> 0.0` over successive decisions; it cannot jump `1.0 -> 0.0` in one update under the frozen 0.5-per-10-min methodology assumption.

The same Python-side continuity guard is applied to Proposed, Auto-RBC, EFD and the All-open/All-closed diagnostic extremes. All-open/All-closed therefore **ramp toward** their extreme target rather than teleporting all facilities at the first decision. No-control has no Python writes. Internal RTC is intentionally left under its frozen native SWMM `[CONTROLS]` semantics and is not overwritten by the Python continuity guard.

`max_setting_delta_per_update = 0.5` is an **idealized simulation engineering-smoothing assumption**, not an EPA field-actuator standard.

## Formal information budgets

Exactly seven strategies remain:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Competitive comparators are No-control, Internal RTC, Auto-RBC and EFD. All-open/all-closed are diagnostic extremes.

The information advantage of the rule/native baselines is **accepted and disclosed rather than artificially removed**:

- Proposed directly observes only the frozen sparse sensor layout, realised rainfall and actuator readback;
- Internal RTC may use the true native SWMM variables used by its frozen rules;
- Auto-RBC may use current true actuator-adjacent node depths;
- EFD may use current true controlled-storage depths.

No strategy may use future realised rainfall, future SWMM state/flooding, future Internal trajectory or untouched Final truth.

## 360-min prediction horizon

The Project7 production horizon is now a **pre-registered methodology design choice**:

```text
horizon = 360 min = 72 x 5-min model steps = 36 x 10-min control blocks
```

Phase-0 sustained-step censoring and pulse/release recovery remain scientific diagnostics and must still be reported. They no longer automatically lengthen or shorten the production horizon. In particular, a late depth response in h360 must not be hidden, but it also must not silently rewrite the user-frozen 360-min controller contract.

Use `rtc-freeze-phase0-timing` to bind diagnostics to the fixed runtime grid.

## Data roles and authoritative truth

```text
D0  controls-disabled reference hydraulic trajectories
D1  development/train-only controlled state exploration
D2  exact same-prefix local sustained actuator counterfactuals
Phase0 pulse  one-block action then base-action release/recovery diagnostic
D3  joint multi-actuator, multi-control-block sequences
Step1 sparse causal history -> current full hydraulic state
Step2 current state + future setting/rainfall sequence -> future hydraulic/action consequence
MPC  optimize a 360-min future path, execute the first 10-min move only, re-observe and re-solve
```

Final TFV/PFV truth comes from authoritative SWMM cumulative node statistics. Global Peak is obtained by routing-step replay of the frozen executed decision schedule.

## Simulation identity and local large-data reuse

Large hydraulic data stays on the local data disk. Use a persistent asset root such as:

```text
E:\RTC_sewer\Project7\data_assets_v069
```

D2/D3 assets are keyed by physical event/checkpoint state, complete action/sequence, engine and timing semantics rather than by directory name. Before SWMM starts, runners deduplicate requests, verify endpoint executability, audit the local cache and write `REQUEST_CENSUS.json`.

Existing successful v0.6.7/v0.6.8 authoritative branches may be reused only after simulation-identity and artifact-hash verification. File names or folder similarity are never sufficient. See `docs/SIMULATION_ASSET_MANAGEMENT_V068.md`.

## Fresh acceptance flow

```text
0  adopt/build v0.6.7 physical inputs / graph / audit
1  prepare effective-120-min events + high-frequency D0 / Phase-0 D2
2  sustained timing + pulse/recovery + exact control leverage + bind fixed 360-min timing
3  production D0/D1 + Step1 train/held-out acceptance
4  production D2/D3 + Step2 train/held-out acceptance
5  gradient + ranking/regret acceptance
6  development closed-loop comparison with temporally continuous commands
7  runtime/readback/deadline + cross-decision/horizon-continuity acceptance
8  Policy Lock
9  untouched seven-strategy authoritative-SWMM Final
```

Policy Lock refuses the legacy V4 controller contract and the legacy runtime-acceptance V1 evidence. It binds the V5 controller, effective-120 event clock, fixed 60-min first-control epoch, 360-min horizon and zero continuity violations.

## Start here

- `docs/METHOD_TESTBED_V067.md` — frozen physical/rainfall testbed.
- `FORMAL_PIPELINE_LATEST.md` — active v0.6.9 fail-closed evidence contract.
- `docs/SIMULATION_ASSET_MANAGEMENT_V068.md` — local large-data identity/reuse/invalidation contract.
- `configs/formal_controller_v5.template.json` — active TFV-first 60/120/360 temporal-continuity controller template.
- `scripts/bootstrap_project7_v067.ps1` — source-input adoption helper only.

Historical controller V4 and earlier learned evidence remain provenance only for the v0.6.9 runtime contract.
