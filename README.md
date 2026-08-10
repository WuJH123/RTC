# Wuhan RTC v0.6.7 — TFV-first methodology testbed

This repository implements a **model-based urban-drainage real-time-control methodology test** on an idealized, simplified Wuhan SWMM network.

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

## Scientific claim

The study is designed to demonstrate that the proposed sparse-sensing surrogate MPC can reduce **sewer-node overflow** in the frozen SWMM testbed. It is not a field-calibrated Wuhan digital twin and does not certify field actuator capability.

Primary objective: minimize authoritative-SWMM system-wide cumulative **Total Flood Volume (TFV)**. PFV at the eight frozen priority nodes is a soft secondary/diagnostic quantity. Priority depth and Global Peak are report-only.

## Active v0.6.7 inputs

Fresh Project7 studies must be rebuilt from source-only assets with:

```powershell
rtc-build-method-testbed-v067 `
  --source-inp E:\RTC_sewer\Project7\source\wuhan_with_controls.inp `
  --sensors E:\RTC_sewer\Project7\source\sensor_nodes.txt `
  --priority E:\RTC_sewer\Project7\source\priority_nodes.txt `
  --out-root E:\RTC_sewer\Project7\inputs `
  --warmup-minutes 60 `
  --recession-minutes 360 `
  --orifice-travel-minutes 10
```

Or use `scripts/bootstrap_project7_v067.ps1`, which syncs GitHub and refuses to mix a non-empty historical `inputs` or `study_v067` directory into the fresh study.

Do **not** reuse historical D0/D1/D2/D3 trajectories, Step1/Step2 checkpoints, training shards, development runs, Policy Locks or Final evidence.

## Rainfall contract

The active design library is exactly **30 events**:

```text
return periods: 5, 10, 20, 50, 100 years
durations:      60, 120, 180, 240, 300, 360 min
pattern:        Chicago only
Chicago r:      0.39
rain step:      5 min
spatial mode:   one uniform design gage across all subcatchments
```

The generator uses Wuhan DB4201/T 641-2020 directly:

```text
i = 9.686 * (1 + 0.887 * log10(P)) / (t + 11.23)^0.658   [mm/min]
```

No historical rainfall file is required. Five-minute block depths are analytically integrated from the Chicago curve and checked against the standard duration depth.

## Event clock

The fresh methodology-testbed event contract is:

```text
pre-rain causal warm-up/history = 60 min
post-rain evaluation tail       = 360 min
```

The 60-min prefix supplies the default 13-frame `0,5,...,60 min` causal history. It is not a claim of complete dry-weather hydraulic equilibrium; a small development-only 60/120-min onset-state sensitivity should be checked before scaling production data.

The 360-min tail is a **fixed evaluation window**, not evidence that flooding has returned to zero. Formal TFV is cumulative SWMM node flooding volume through the common event endpoint.

## Physical network contract

v0.6.7 preserves the user-frozen idealizations:

- supplied DWF is retained as idealized background hydraulic loading;
- all 41 source outfalls remain `FREE`;
- SUBAREAS/infiltration source values are not recalibrated;
- the five retrofit storage curves are preserved.

Actuation is explicitly `SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY`:

- 57 source two-point `PUMP2` depth-flow curves are migrated to `PUMP4` without changing their endpoints;
- all 42 orifices remain continuously controllable on `[0,1]` and receive a 10-min full travel time;
- `RTC_IN_01..05` and `RTC_OUT_01..05` receive flap gates so each storage connection is one-way in its declared direction;
- the candidate 10-min supervisory contract uses `max_setting_delta_per_update = 0.5`;
- four known copied OFF rules (`VP0600010.3/.4/.5`, `add300.1`) are repaired from `SETTING=1` to `SETTING=0`.

## Graph/model physics

The v0.6.7 graph no longer exposes only invert/max depth and node type. The node-static vector now includes storage capacity/area, incident conduit length/roughness/section scale, contributing subcatchment area and impervious area, area-weighted width/slope, and Horton infiltration rates. These are frozen INP properties and therefore preserve online causality.

## Formal baselines

Exactly seven Formal strategies remain:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Competitive comparators are No-control, Internal RTC, Auto-RBC and EFD. All-open/all-closed are diagnostic extremes. No strategy may use future realised rainfall, future SWMM state/flooding, future Internal trajectory or untouched Final truth.

## Data roles

```text
D0  controls-disabled reference hydraulic trajectories
D1  development/train-only controlled state exploration
D2  exact same-prefix local actuator counterfactuals
D3  joint multi-actuator, multi-control-block sequences
Step1 sparse causal history -> current full hydraulic state
Step2 current state + future setting/rainfall sequence -> future hydraulic/action consequence
MPC  optimize a future setting sequence, execute first move only, re-solve
```

Final TFV/PFV truth comes from authoritative SWMM cumulative node statistics. Global Peak is obtained by routing-step replay of the frozen executed decision schedule.

## Fresh acceptance flow

```text
0  build v0.6.7 inputs / readiness / graph / physical audit
1  small high-frequency No-control D0 + exact-prefix D2
2  response timing + exact SWMM control-leverage evidence
3  production D0/D1 + Step1 train/held-out acceptance
4  production D2/D3 + Step2 train/held-out acceptance
5  gradient + ranking/regret acceptance
6  development closed-loop comparison
7  runtime/readback/deadline acceptance
8  Policy Lock
9  untouched seven-strategy authoritative-SWMM Final
```

Do not scale learned models when a physical/source-data problem is unresolved.

## Start here

- `docs/METHOD_TESTBED_V067.md` — active scientific/physical contract.
- `FORMAL_PIPELINE_LATEST.md` — active fail-closed evidence contract.
- `scripts/bootstrap_project7_v067.ps1` — fresh Windows Project7 bootstrap.
- `configs/formal_controller_v4.template.json` — TFV-first production-controller template.

Historical v0.6.6 runbooks remain provenance only and must not be treated as the active fresh-study input contract.
