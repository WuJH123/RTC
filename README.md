# Project7 — TFV-first continuous real-time control methodology testbed

This repository implements an **idealized EPA-SWMM urban-drainage RTC methodology test** on a simplified Wuhan network. It is **not** a field-calibrated digital twin. SWMM remains the authoritative hydraulic truth.

## Frozen control problem

Every 600 s:

```text
causal sparse sensing
 -> Step1 current full-network state reconstruction
 -> differentiable action-conditioned Step2 hydraulic surrogate
 -> H360 prediction / H120 free control / 109 continuous actuators
 -> engineering envelope inside differentiable decoder
 -> execute only first 10-min target
 -> authoritative SWMM write/readback
 -> re-observe and re-optimize
```

Scientific hierarchy: **system-wide cumulative TFV primary**, Priority8 PFV **one-sided soft secondary**, Global Peak **report-only**. The model/observation step is 300 s; the control update is 600 s; 12 x 109 = 1308 free MPC variables.

## Current Development status: repair the action chain before full

The latest same-checkpoint diagnostics localized a mixed failure in:

```text
setting -> managed actuator flow -> node hydraulics/flooding -> TFV
```

The old V128 Stage A already had severely attenuated TFV gradients. A later read-only decomposition showed that held-out action-to-managed-flow **direction** was often sensible while its **magnitude** was almost absent, and that injecting authoritative SWMM actuator flows into the old node transition did not repair hydraulic/flood-volume action effects.

The current smoke/dev implementation therefore uses a diagnostic-driven repair rather than more epochs or a larger hidden layer:

1. **FIT-only hybrid flow residual scale** — per actuator, the numerical residual scale is the maximum of the historical temporal-flow-change q99.5 and the TrainFit candidate-minus-reference action-flow q99.5. Holdout/AUDIT/Validation/Final/Formal outcomes never define this scale. It is not an engineering ramp constraint.
2. **Explicit setting response** — current setting enters an explicit differentiable linear/quadratic basis. Previous flow remains causal context, but a learned responsiveness gate can no longer suppress the entire action delta.
3. **Counterfactual effect supervision** — Stage A and B0 train both absolute hydraulics and candidate-minus-reference state/managed-flow effects, so a good teacher-forced forecast cannot hide an unusable action Jacobian.
4. **Edge-physics propagation** — the current smoke/dev model uses the graph-edge-aligned frozen SWMM link artifact and dynamic head-difference/head-gradient messages. Missing ordinary-conduit dynamic-flow labels are not fabricated.
5. **Exact H360 TFV objective retained** — the canonical float32, two-pass full within-group pairwise first-order objective is unchanged; a small FIT-only H360 action-effect rehearsal follows it to reduce sensitivity forgetting.

Primary implementation files:

```text
scripts/run_step2_current.py
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_action_identifiable_v128.py
src/rtc/step2_current_dev_context_v128.py
src/rtc/step2_differentiable_v128_edge.py
src/rtc/edge_physics_current_v128.py
src/rtc/step2_train_v128_exact.py
```

## Debug-first execution

Current user entrypoint:

```text
scripts/run_step2_current.py
```

The repaired candidate currently enables only:

```text
--profile smoke
--profile dev
```

`--profile full` intentionally fails closed until held-out Development action-flow, gradient, ranking and spatial evidence improves and a later GitHub change explicitly promotes the architecture. Smoke/dev are NONFINAL and cannot enter D5/runtime/Policy Lock.

The recommended loop is deliberately stage-gated:

```text
cheap gates + preflight
 -> build/reuse edge-physics artifact
 -> smoke Stage A only
 -> action-to-flow audit + Stage-A TFV-gradient audit
 -> STOP / reject or explicitly continue
 -> resume B0
 -> repeat action-flow + gradient audit
 -> STOP / reject or explicitly continue
 -> resume H360 exact objective
 -> ranking + spatial + final smoke gradient
 -> dev only if coherent improvement
 -> full only after later explicit promotion
```

Stage checkpoints (`stage_a.pt`, `stage_b0.pt`, `stage_objective.pt`) are source/graph/data/design strict. The repaired runner also fingerprints the edge-physics artifact and enhanced model/training source so a baseline checkpoint cannot masquerade as a compatible repaired checkpoint.

## Current Development diagnostics

```text
scripts/audit_step2_actuator_flow_effect_current.py
    cheap action -> managed-flow effect magnitude/sign audit for Stage A/B0/objective

scripts/audit_step2_gradient_stage_current_dev.py
    source-strict TFV directional-gradient decomposition at Stage A/B0/objective

scripts/audit_step2_gradient_current_dev.py
    final smoke/dev objective-stage D2 TFV gradient audit

scripts/audit_step2_spatial_current.py
    held-out D2 action-effect sign/magnitude at 1-3 / 4-6 / 7-12 / 13+ hops

scripts/build_edge_physics_current.py
    deterministic edge artifact from frozen INP + graph; required by current smoke/dev

scripts/train_step1_global_attention_dev.py
scripts/audit_step1_global_attention_current.py
    separate Step1 attention ablation; never hot-swap a different Step1 into an old Step2 lineage
```

Hydraulic-influence shortcuts remain Development-only and are **not** promoted by this change because the measured failure was not a simple monotonic far-field-only collapse.

## Physics boundary

The current data do not expose every ordinary-conduit dynamic-flow label or every SWMM node loss term required for a complete continuity residual. The repository therefore does **not** fabricate missing link-flow supervision and does **not** enable the incomplete continuity proxy as a training loss. Edge physics and available authoritative node/managed-flow counterfactuals are used instead.

## One current user surface

Start from:

```text
CODEX_START_HERE.md
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Stable entrypoints:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Runtime/seven-strategy entrypoints remain in the repository for the future promoted model but are **not authorized for the current repaired smoke/dev candidate**.

The obsolete detached `src/rtc/step2_train_v128.py` and obsolete full-only current runner remain deleted. Historical V127/V128 files that remain are archival or shared internal implementations, not alternative user entrypoints.

## Boundaries

No future realised rainfall, future SWMM state or future Internal trajectory online. No Validation/Final/Formal/Policy Lock during Development. Do not call the default persistence/decay rainfall forecast robust/stochastic MPC. Do not project an action after scoring. A later real-time claim still requires every guarded decision below 600 s together with score==execute, continuity and same-epoch target readback.

See **`CODEX_START_HERE.md`** for the exact clean-sync, stage-stop, resume and diagnostic commands.
