# Project7 — TFV-first continuous real-time control methodology testbed

This repository implements an **idealized EPA-SWMM urban-drainage real-time-control methodology test** on a simplified Wuhan network. It is **not** a field-calibrated Wuhan digital twin and does not certify field actuator capability.

## Current research target

Every 600 s, the current method:

```text
causal sparse sensing
  -> Step1 current full-network state reconstruction
  -> typed/physics-aware differentiable action-conditioned Step2 surrogate
  -> H360 prediction + 12 x 109 continuous 10-min target fractions over H120
  -> per-actuator engineering envelope inside the differentiable decoder
  -> execute only the first 10-min target
  -> authoritative SWMM target write/readback
  -> re-observe and re-optimize
```

Scientific hierarchy:

- **system-wide cumulative TFV is the primary objective**;
- Priority8 PFV is a **one-sided soft secondary** quantity;
- Global Peak is **report-only**;
- authoritative truth is **SWMM**.

The default online rainfall forecast is causal deterministic persistence/decay. It must not be described as robust/stochastic MPC unless a separate multi-scenario forecast/evidence contract is frozen.

Frozen timing:

```text
model / observation step       300 s
supervisory control update     600 s
prediction horizon             72 x 5 min = 360 min
free control horizon           12 x 10 min = 120 min
execute                         first 10 min target only
```

## One current execution surface

Do not select scripts from historical version numbers. Start only from:

- `CODEX_START_HERE.md` — complete current execution order;
- `configs/step2_current_contract.json` — machine-readable research/method contract;
- `configs/project7_execution_registry.json` — canonical routing;
- `configs/v128_control_execution.json` — selected implementation/hardware contract.

Stable user/Codex entrypoints:

```text
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Current internal Step2 implementation uses the V128 typed actuator-message architecture and the two-pass exact full within-group pairwise first-order training objective:

```text
src/rtc/step2_differentiable_v128.py
src/rtc/step2_train_v128_hydraulic.py
src/rtc/step2_train_v128_exact.py
src/rtc/checkpoint_v128.py
```

Versioned V127 modules that remain in the repository are either archival implementations or audited shared orchestration used internally. They are not current user entrypoints.

## Control/action semantics

All 109 writable actuators are eligible. MPC optimizes 1308 free variables: 12 future 10-min target fractions x 109 actuators.

Engineering bounds and target-rate limits are applied **inside the differentiable fraction-to-target decoder before scoring**, so the scored action is the action that can be executed. Post-score projection is not allowed.

The historical default envelope uses graph min/max bounds plus a 0.5 target change per 10 min. This is an **idealized methodology assumption**, not a field-device claim. A custom per-actuator envelope requires matching decoder-space D5 evidence.

The supervisory slew anchor is the previous issued `target_setting`. Realised `current_setting` is hydraulic state/tracking diagnostic and may lag the target.

## Step2 scientific role

Step2 must learn decision-relevant hydraulic/action consequences, not just low average state error. Current V128 removes two previous failure modes:

1. actuator settings are no longer collapsed into lossy per-node scalar sums; each actuator contributes a typed/physics-aware, direction-aware message using endpoint state, setting, previous/predicted managed flow, responsiveness, physical/type features and actuator identity;
2. H360 candidate ranking uses a two-pass same-parameter-snapshot construction so full within-state pairwise **first-order gradients** are recovered while only one GPU microbatch autograd graph is resident at a time.

The training ranking floor is a fixed 1 m3 SWMM action-effect threshold; event magnitude does not enlarge the training deadband.

## Causality and data boundaries

Allowed online information:

- frozen sparse sensor observations/history;
- realised rainfall observed up to the current time;
- actuator current/target readback;
- causal rainfall forecast derived from observed history.

Forbidden online information:

- future realised rainfall;
- future SWMM hydraulic/flooding state;
- future Internal-RTC trajectory;
- Validation/Final/Formal truth.

Development training must not use InternalHoldout, D4-AUDIT, D5-AUDIT or D2 development-validation outcomes. Ranking, D2 and D5 evidence must all reference the identical final Step2 SHA256.

## Checkpoint and runtime correctness

A current Step2 checkpoint is fail-closed against:

- model-source semantic changes;
- exact-training-source semantic changes;
- graph/schema/time-contract mismatch;
- V127/older-V128 checkpoint mixing.

A real-time claim requires every complete continuity-guarded supervisory decision callback to finish in **less than 600 s**, with explicit score==execute, continuity and same-epoch SWMM target-write/readback evidence.

## Authoritative comparison

Exactly seven strategies are compared on the same event/clock/engine:

1. Proposed current continuous differentiable MPC;
2. No-control;
3. Internal RTC;
4. Auto-RBC;
5. storage-volume EFD;
6. All-open;
7. All-closed.

TFV/PFV are recomputed from authoritative node statistics. Global Peak is obtained by routing-step frozen-decision replay and remains report-only.

## Development versus scientific promotion

The current implementation is merged and maintained on `main` to remove repository routing ambiguity, but code/CI success is **not** Policy Lock or Final evidence. Promotion still requires same-checkpoint ranking/gradient evidence, acceptable H30-H360 hydraulic behavior, authoritative development TFV/PFV, and measured sub-600-s decisions without execution violations.

See `CODEX_START_HERE.md` for the exact workstation commands and stop rules.
