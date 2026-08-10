# Wuhan RTC v0.6.8 — TFV-first methodology testbed with simulation-asset lineage

This repository implements a **model-based urban-drainage real-time-control methodology test** on an idealized, simplified Wuhan SWMM network. v0.6.8 keeps the v0.6.7 physical/rainfall testbed unchanged and hardens event-time semantics, reusable SWMM simulation identity, endpoint preflight and response-timing diagnosis.

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

## Active physical inputs

The source physical/rainfall bundle remains v0.6.7. It can be built from source-only assets with:

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

The 60 min above is the **source-bundle pre-rain prefix**, not the complete v0.6.8 production initialization decision. `scripts/bootstrap_project7_v067.ps1` remains the source-bundle bootstrap/adoption helper.

Do not copy historical model checkpoints, training shards, acceptance evidence, Policy Locks or Final evidence into a fresh study. Existing authoritative v0.6.7 D2 computations may, however, be indexed and reused under the v0.6.8 simulation-identity/hash audit described below; this is not the same as treating old derived files as trusted merely because their paths exist.

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

## Event clock: source prefix != effective warm-up

The source bundle carries:

```text
source pre-rain prefix          = 60 min
fixed whole-event evaluation tail = 360 min
```

The completed v0.6.7 development sensitivity showed that initialization at the rainfall onset is materially different when the effective prefix is increased. The active production-data preparation target is therefore:

```text
effective pre-rain warm-up = 120 min
```

Use the explicit v0.6.8 interface:

```powershell
rtc-prepare-event-suite `
  --events E:\RTC_sewer\Project7\inputs\contracts\events_with_splits.csv `
  --out-dir E:\RTC_sewer\Project7\study_v068\prepared_events\events `
  --out-registry E:\RTC_sewer\Project7\study_v068\prepared_events\events_with_splits.csv `
  --target-effective-warmup-minutes 120 `
  --post-rain-tail-minutes 360
```

The prepared registry records separately:

```text
source_pre_rain_prefix_minutes
additional_warmup_minutes
effective_warmup_minutes
```

For the current bundle this resolves to 60 + 60 = 120 min. The 120-min initialization is not a claim of complete dry-weather equilibrium.

The 360-min post-rain tail remains the **Formal common evaluation window**. Longer 480/600-min event END extensions are allowed only for Phase-0 checkpoint+horizon executability and do not redefine Final TFV.

## Physical network contract

v0.6.7/v0.6.8 preserves the user-frozen idealizations:

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

The graph exposes the v0.6.7 26-D node-static physical/hydrologic vector, including storage capacity/area, incident conduit length/roughness/section scale, contributing subcatchment/impervious area, area-weighted width/slope and Horton infiltration rates. These are frozen INP properties and preserve online causality.

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
D2  exact same-prefix local sustained actuator counterfactuals
Phase0 pulse  one-block action then base-action release/recovery diagnostic
D3  joint multi-actuator, multi-control-block sequences
Step1 sparse causal history -> current full hydraulic state
Step2 current state + future setting/rainfall sequence -> future hydraulic/action consequence
MPC  optimize a future setting sequence, execute first move only, re-solve
```

Final TFV/PFV truth comes from authoritative SWMM cumulative node statistics. Global Peak is obtained by routing-step replay of the frozen executed decision schedule.

## Simulation identity and local data reuse

Large hydraulic data stays on the local data disk. Use a persistent asset root, for example:

```text
E:\RTC_sewer\Project7\data_assets_v068
```

D2/D3 assets are keyed by physical event/checkpoint state, complete action/sequence, engine and timing semantics rather than by directory name. Before SWMM starts, runners deduplicate requests, verify `checkpoint + horizon <= event END`, audit the local cache and write `REQUEST_CENSUS.json`.

Example max-horizon Phase-0 D2:

```powershell
rtc-run-probes `
  --manifest <d2_manifest.csv> `
  --out-dir <phase0\d2_h360> `
  --horizon-minutes 360 `
  --snapshot-horizons-minutes 210,240,300,360 `
  --stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1 `
  --asset-root E:\RTC_sewer\Project7\data_assets_v068
```

The same h360 compact trajectory can provide h210/h240/h300 **timing views** without three repeated SWMM simulations:

```powershell
rtc-phase0-timescale ... --analysis-horizon-minutes 210
rtc-phase0-timescale ... --analysis-horizon-minutes 240
rtc-phase0-timescale ... --analysis-horizon-minutes 300
rtc-phase0-timescale ... --analysis-horizon-minutes 360
```

For exact shorter-horizon TFV, only stored cumulative SWMM endpoint snapshots are valid. A long branch's final h360 cumulative statistics must never be relabeled as h210/h240/h300 truth.

Existing successful v0.6.7 D2 branches can be indexed in place with `rtc-index-existing-d2-assets`; the first endpoint-failed h240 partial lineage must remain invalid/quarantined. Audit the store with `rtc-audit-simulation-assets`.

See `docs/SIMULATION_ASSET_MANAGEMENT_V068.md` for identity, qualification, invalidation and Git/local-storage rules.

## Response timing

The prior sustained-step Phase-0 evidence showed long hydraulic response tails. v0.6.8 does not lower the existing near-horizon censor guard. Instead:

1. run the largest executable sustained D2 horizon once;
2. derive shorter timing views from that trajectory;
3. use a separate development-only pulse/release experiment to measure decay after one 10-min control block;
4. freeze production timing only after sustained-response, post-release recovery, control-leverage and runtime evidence are jointly interpretable.

Pulse sequences reconstruct and verify the exact complete base-action SHA and enforce `max_setting_delta_per_update = 0.5` before SWMM execution.

## Fresh acceptance flow

```text
0  adopt/build v0.6.7 physical inputs / readiness / graph / audit
1  high-frequency D0 + explicit warm-up sensitivity + longest-horizon exact-prefix D2
2  sustained timing + pulse/recovery + exact control leverage + timing freeze
3  production D0/D1 + Step1 train/held-out acceptance
4  production D2/D3 + Step2 train/held-out acceptance
5  gradient + ranking/regret acceptance
6  development closed-loop comparison
7  runtime/readback/deadline acceptance
8  Policy Lock
9  untouched seven-strategy authoritative-SWMM Final
```

Do not scale learned models when a physical/source-data/timing problem is unresolved.

## Start here

- `docs/METHOD_TESTBED_V067.md` — frozen physical/rainfall testbed.
- `FORMAL_PIPELINE_LATEST.md` — active v0.6.8 fail-closed evidence contract.
- `docs/SIMULATION_ASSET_MANAGEMENT_V068.md` — local large-data identity/reuse/invalidation contract.
- `scripts/bootstrap_project7_v067.ps1` — v0.6.7 source-input bootstrap/adoption helper.
- `configs/formal_controller_v4.template.json` — TFV-first production-controller template.

Historical v0.6.6 and earlier learned evidence remains provenance only. v0.6.7 authoritative hydraulic branches are reusable in v0.6.8 only through explicit identity/hash verification; path/name similarity is never sufficient.
