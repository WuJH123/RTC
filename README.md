# Wuhan RTC v0.6.6 — event-paired baselines, causal dry prefix and TFV-first MPC

This repository implements a **model-based** urban-drainage real-time-control workflow:

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
             continuous receding-horizon MPC
                              |
                       first move only
                              |
                              v
                       authoritative SWMM
```

The primary objective is system-wide cumulative **Total Flood Volume (TFV)**. Priority Flood Volume (PFV) and priority depth are soft/reporting quantities rather than hard admission gates. Global Peak is report-only. Final truth comes from authoritative SWMM cumulative node statistics and routing-step replay.

## Start here

Use:

- `docs/LEAN_FRESH_RUN_V066.md` — current fresh-run contract;
- `FORMAL_PIPELINE_LATEST.md` — scientific/Policy-Lock/Final evidence contract;
- `configs/formal_controller_v4.template.json` — production-controller template;
- `configs/formal_baseline_plan.v3.json` — seven-strategy event-paired comparison contract.

Historical v0.6.2–v0.6.5 runbooks remain useful provenance but are not the active contract.

## v0.6.6 corrections from the Project7 pre-training audit

The pre-training audit found that copied event INPs carried the desired event forcing/DWF but no executable native `[CONTROLS]`. Using those files directly for `internal_rtc` would therefore mislabel a no-native-control event as the native comparator. It also found no pre-rain causal history, a 3 h recovery tail that was right-censored in all eight high-frequency pilot events, and no field metadata proving that 57 PUMP2 facilities are real VFD assets.

v0.6.6 changes the contract accordingly:

```text
rtc-prepare-event-suite
    -> preserves the absolute storm clock and DWF phase,
       moves simulation start earlier to create a dry/DWF causal history,
       and extends the post-rain recovery tail.

rtc-check-study-readiness
    -> fails closed on insufficient warm-up/tail, unresolved sensor/rainfall provenance,
       or an invalid actuator-claim scope.

Internal RTC runtime
    -> exact prepared event forcing + DWF + initial conditions
       plus ONLY the frozen network [CONTROLS] payload.

Formal Final
    -> hashes and verifies the native controls payload against Policy Lock.
```

## Formal baseline matrix

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

The scientifically important competitive baselines are No-control, Internal RTC, Auto-RBC and EFD. All-open and All-closed are diagnostic extremes and are reported separately.

- **No-control** — native supervisory `[CONTROLS]` disabled and no Python setting writes; exact event forcing, DWF, initial conditions and intrinsic/local equipment physics remain.
- **Internal RTC** — a **strong native engineering comparator**. The same event forcing/DWF/initial state is paired at runtime with the frozen network native rules. It may access true SWMM rule variables on the native rule clock; this information/frequency advantage is disclosed rather than artificially removed.
- **Auto-RBC** — causal local rule control from current actuator-adjacent true SWMM depths. Its direct local sensor budget may exceed the Proposed sparse layout and is disclosed.
- **EFD** — causal storage-aware Equal Filling Degree from current controlled-storage depths.
- **All-open / All-closed** — diagnostic extremes.

No strategy may use future realised rainfall, future SWMM state/flooding, future Internal trajectory or untouched Final truth.

## Actuation scope

All 109 discovered SWMM-writable settings remain eligible in the simulation MPC. The default study contract is:

```text
SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY
```

This is **not** a field certification that every pump has a VFD or that every regulator can move continuously. No artificial binary mask is introduced without engineering metadata. A field-deployment claim requires a separate hashed capability map describing control mode, bounds, ramp, dwell, interlock, readback and fail-safe behavior.

## Rainfall scope

The currently audited main library is an exact 180-event factorial over:

```text
return periods: 10, 20, 30, 50, 75, 100 years
durations:      75, 105, 150, 210, 240, 300 min
patterns:       block, chicago_center, chicago_early, chicago_late, double_peak
spatial mode:   one uniform design gage across all subcatchments
```

The repository binds these actual event bytes but does **not** claim independent regeneration from an official Wuhan IDF standard unless an authoritative formula/generator provenance file is supplied. Missing 2–5 year, <75 min, >300 min and spatially heterogeneous storms are robustness/sensitivity gaps, not silently invented data.

## Fresh-data execution logic

Recommended directory separation:

```text
E:\RTC_sewer\Project7\repo
E:\RTC_sewer\Project7\inputs
E:\RTC_sewer\Project7\study_v066
E:\RTC_sewer\Project7\logs
```

Do not overwrite the previous v0.6.5 study. v0.6.6 changes the scientific implementation contract, so D0/D1/D2/D3/models/acceptance/development/Policy Lock/Final evidence must be regenerated under the new contract.

The ten-step flow is:

```text
0. prepared inputs / lineage / graph / actuator / split / readiness audit
1. forcing-only Phase-0 cohort -> high-frequency No-control D0 -> checkpoints -> small D2
2. exact-SWMM response timing + control leverage -> non-censored timing freeze
3. production D0 + D1 -> Step1 train and held-out acceptance
4. production D2 + D3 -> Step2 train and held-out acceptance
5. D2 gradient + D2/D3 ranking/regret acceptance
6. development closed-loop Proposed vs competitive baselines
7. runtime/readback/deadline acceptance
8. Policy Lock
9. untouched seven-strategy authoritative-SWMM Final
```

Do not scale Step2 before exact SWMM demonstrates useful action-dependent TFV variation, and do not enter closed-loop MPC if Step2 action ranking/gradient acceptance fails.

## Data roles

- **D0:** controls-disabled reference hydraulic trajectories, including the causal dry/DWF prefix.
- **D1:** development/train-only continuous controlled exploration for Step1 state coverage.
- **Step1:** sparse causal history -> current full six-channel hydraulic state estimate.
- **D2:** exact same-prefix single-actuator/local counterfactual effects; sampling budget does not shrink online action space.
- **D3:** joint multi-actuator, multi-control-block sequences for interaction/ranking learning.
- **Step2:** current state + future setting/rainfall sequence -> future hydraulic/actuator-flow/flood-volume trajectory.
- **MPC:** optimize the future setting sequence online, retain best-so-far, require predicted hold dominance, execute only the first move and re-solve.

## Metrics

```text
TFV = sum over all hydraulic nodes of SWMM cumulative flooding-volume delta [m3]
PFV = the same cumulative volume delta summed over the eight frozen priority nodes [m3]
Global Peak = max_t sum_i max(flooding_rate_i(t), 0) [m3/s]
```

Formal aggregation gives each independent rainfall group equal weight.

## Workstation defaults

For a 16 GB RAM / RTX 4060 8 GB workstation:

```text
SWMM generation: 16 independent processes x 1 SWMM thread
reduce to 12 workers if Windows begins paging
Step1: batch 8, grad accumulation 2, AMP
Step2: batch 2, grad accumulation 4, AMP
Step2 shard size: about 128
```

## Install and self-test

```powershell
cd E:\RTC_sewer\Project7\repo
git checkout main
git pull --ff-only
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print(importlib.metadata.version('wuhan-rtc'))"
```

Expected release:

```text
0.6.6
```

Code and CI establish execution contracts, not Wuhan performance. Whether Proposed reduces TFV must be demonstrated by fresh authoritative-SWMM development and untouched Final runs.
