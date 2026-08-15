# Project7 — TFV-first continuous real-time control methodology testbed

This repository implements an **idealized EPA-SWMM urban-drainage RTC methodology test** on a simplified Wuhan network. It is **not** a field-calibrated digital twin. SWMM is the authoritative hydraulic truth.

## Frozen control problem

Every 600 s:

```text
causal sparse sensing
 -> Step1 full-network current-state reconstruction
 -> differentiable action-conditioned Step2 surrogate
 -> H360 prediction / H120 free control / 109 continuous actuators
 -> engineering envelope inside differentiable decoder
 -> execute only first 10-min target
 -> SWMM write/readback
 -> re-observe and re-optimize
```

Scientific hierarchy: **system-wide cumulative TFV primary**, Priority8 PFV **one-sided soft secondary**, Global Peak **report-only**. Model/observation step is 300 s; control update 600 s; 12 x 109 = 1308 free MPC variables.

## Debug first, full last

The current development priority is to find weak Proposal components quickly and reject bad ideas before a multi-hour full run. `scripts/run_step2_current.py` therefore has **no default training cost**: the caller must choose one of:

```text
--profile smoke   tiny deterministic Development subset; nonfinal
--profile dev     larger deterministic Development proxy; nonfinal
--profile full    canonical scientific Development training
```

`smoke` and `dev` preserve the 109-actuator/H360/V128/exact-pairwise code path but reduce data coverage and repetitions. They cannot create a strict final checkpoint and cannot enter D5/runtime/Policy Lock. Only explicit `--profile full` can create the strict V6 V128 base checkpoint.

Recommended funnel:

```text
unit/preflight
 -> smoke one-group profiler
 -> smoke
 -> spatial/ranking/gradient diagnosis
 -> dev
 -> reject or promote
 -> full once
 -> D5 + authoritative Development closed loop
 -> seven-strategy Development comparison
 -> Policy Lock only after gates
```

Stage checkpoints (`stage_a.pt`, `stage_b0.pt`, `stage_objective.pt`) make Stage-A/B0 recomputation avoidable after compatible interruptions. They are explicitly NONFINAL and fail closed on profile/graph/data/design mismatch.

## Current P0 correctness fixes

The exact H360 objective now uses one **canonical float32 SWMM candidate TFV delta** for informative-pair census, reported pair loss and live directed gradient. This eliminates threshold-partition drift caused by mixing NumPy float64 census sums with float32 CUDA reductions around the frozen 1 m3 action-effect floor.

Current strict checkpoint:

```text
PROJECT7_V128_STEP2_CHECKPOINT_V6_CURRENT_PROFILE_TRAINING_SOURCE_STRICT
```

It rejects smoke/dev artifacts, stale model/training source, graph/schema/time mismatch and old V127/V128 checkpoint contracts.

## Spatial/action-effect diagnostics

A control facility can influence nodes many graph hops away. Current V128 already propagates effects recursively, but long-range accuracy must be measured rather than assumed.

Current Development diagnostics:

```text
scripts/audit_step2_spatial_current.py
    held-out D2 action-effect sign/magnitude at 1-3 / 4-6 / 7-12 / 13+ hops

scripts/audit_step1_global_attention_current.py
    frozen Step1 vs V122 sensor-to-all-node attention on identical held-out windows

scripts/build_edge_physics_current.py
scripts/run_step2_edge_aware_dev.py
scripts/audit_step2_edge_spatial_current.py
    Development-only V128 edge-aware ablation using SWMM link physics + dynamic head gradients

scripts/build_hydraulic_influence_current.py
    Development-only sparse remote influence shortcut candidates from D2 TrainFit only
```

These experiments are not automatically promoted. Global-attention Step1 requires a rebuilt causal state store and complete Step2 retraining if selected. Edge-aware/influence variants must first improve held-out spatial/ranking/gradient evidence.

## Physics diagnostics

V128 actuator injection is locally conservative, but the current Step2 state does not expose every SWMM node-loss term or ordinary conduit dynamic flow. `src/rtc/physics_diagnostics_v128.py` therefore provides a diagnostic continuity proxy and a conduit-flow-label readiness gate; it deliberately does **not** enable an incomplete physics loss or fabricate link-flow labels.

## One current user surface

Start from:

```text
CODEX_START_HERE.md
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
```

Stable entrypoints:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

The obsolete full-only `scripts/run_step2_v128_control_4060.py` and detached `src/rtc/step2_train_v128.py` are deleted. Historical V127 modules that remain are archival or still-used shared implementations; they are not user entrypoints.

## Boundaries

No future realised rainfall/state/Internal trajectory online. No Validation/Final/Formal/Policy Lock during development. Do not claim robust/stochastic MPC for the default deterministic persistence/decay rainfall forecast. Do not project an action after scoring. Ranking, D2 and D5 evidence for promotion must reference the identical final Step2 SHA256. A real-time claim additionally requires every guarded decision <600 s plus explicit score==execute, continuity and same-epoch target readback.

See `CODEX_START_HERE.md` for the exact debug-first commands and stop rules.
