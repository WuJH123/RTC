# Project7 — TFV-first continuous real-time control methodology testbed

This repository implements an **idealized EPA-SWMM urban-drainage RTC methodology test** on a
simplified Wuhan network. It is **not** a field-calibrated digital twin. SWMM remains the
authoritative hydraulic truth.

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

Scientific hierarchy: **system-wide cumulative TFV primary**, Priority8 PFV **one-sided soft
secondary**, Global Peak **report-only**. Model/observation step = 300 s, control update = 600 s,
and 12 x 109 = 1308 free MPC variables.

## Current Development status — counterfactual trajectory first

PR #77 recovered the magnitude of the historical full-H72 managed-flow effect, but that metric
mixes local actuator response with later hydraulic feedback and can be dominated by one high-energy
actuator. Stage-A TFV autograd also remained weak in direction. The current Development surface
does **not** train toward a larger SWMM objective gradient. It first repairs the causal trajectory
chain:

1. **A0 — direct same-prefix setting -> managed flow.** Only the first setting-divergence
   transition is a local actuator label. Temporal flow variation and direct action-response scales
   remain separate.
2. **A1 — managed flow -> hydraulics.** The actuator submodel is frozen and authoritative SWMM
   managed flow is injected. For direct reference/candidate effects, both branches share the
   reference setting inside the typed action context; authoritative managed flow is the only
   branch-varying control signal. This blocks a hidden `setting -> message -> state` bypass.
3. **A2 — joint direct counterfactual teacher forcing.** Predicted flow and hydraulic transition are
   trained together with absolute state/flow targets plus response-weighted same-prefix effects.
4. **B0 — autoregressive network feedback.** Full candidate/reference hydraulic feedback belongs
   here, not in the local `dq/du` label. Feedback-flow effects use actuator flow standard deviation
   for normalization rather than the local direct-action scale.
5. **B0 evidence before H360.** A source-strict nonfinal ranking+horizon audit and a selectable B0
   spatial audit run directly on `stage_b0.pt`; no production checkpoint is required.
6. **H360 exact TFV objective downstream.** The source-strict two-pass within-group pairwise TFV
   objective remains intact. SWMM action-gradient labels are never training targets; autograd is a
   Development diagnostic and future online optimization signal after trajectory/ranking fidelity.

Large authoritative state/flow truth arrays stay mmap-backed. A0/A1/A2, B0 and the post-objective
trajectory anchor call the V128 lazy helpers explicitly; direct-pair extraction materializes only
the reference/candidate slices needed through first divergence.

Current implementation:

```text
scripts/run_step2_current.py
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_counterfactual_first_v128.py
src/rtc/step2_oracle_isolation_v128.py
src/rtc/step2_counterfactual_training_v5.py
src/rtc/step2_differentiable_v128_edge.py
src/rtc/step2_train_v128_exact.py
```

Current Development diagnostics:

```text
scripts/audit_step2_actuator_flow_effect_current.py
    direct same-prefix + actuator-balanced flow metrics; full-H72 feedback reported separately

scripts/audit_step2_direct_hydraulic_effect_current.py
    normal predicted-flow vs strict q-only authoritative-flow hydraulic isolation

scripts/audit_step2_stage_ranking_current.py
    source-strict ranking + H30-H360 trajectory audit for stage_b0 or objective checkpoints

scripts/audit_step2_spatial_current.py
    source-strict held-out action effect by graph distance for stage_b0 or objective checkpoints

scripts/audit_step2_gradient_stage_current_dev.py
    TFV autograd diagnostic at Stage A/B0/objective
```

## Development funnel

Only `--profile smoke` and `--profile dev` are enabled. `--profile full` intentionally fails
closed. Smoke/dev checkpoints are NONFINAL.

```text
cheap gates + preflight
 -> edge-physics lineage
 -> smoke Stage A (A0 -> q-only A1 -> A2)
 -> direct same-prefix/actuator-balanced flow audit
 -> strict q-only direct hydraulic audit
 -> Stage-A TFV-gradient diagnostic
 -> STOP / reject or explicitly resume B0
 -> B0 source-strict ranking+horizon + spatial + gradient audits
 -> only then exact H360 TFV objective
 -> repeat same-checkpoint ranking+horizon + spatial + gradient diagnostics
 -> deterministic dev confirmation
 -> future explicit production checkpoint/loader/runtime promotion
 -> D5 / runtime / seven-strategy / Policy Lock only after that promotion
```

A large gradient does not rescue a surrogate with wrong trajectory/ranking. Conversely, weak
Stage-A gradient alone is not a reason to reject a candidate whose direct chain has improved;
autoregressive trajectory and ranking evidence are checked before the final gradient decision.

## Physics and evidence boundary

Current smoke/dev uses frozen SWMM edge descriptors and dynamic head-difference/head-gradient
messages. Missing ordinary-conduit dynamic-flow labels are not fabricated, and the incomplete
continuity proxy is not enabled as a training loss. Hydraulic-influence shortcuts/deeper GNNs are
not promoted until held-out spatial evidence isolates a distance-dependent failure.

No future realised rainfall, future SWMM state or future Internal trajectory online. No
Validation/Final/Formal/Policy Lock during Development.

## Production is intentionally fail-closed

The current counterfactual-first model is a Development subclass and has no promoted production
checkpoint factory/loader yet. Therefore:

```text
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

support `--help` / `--promotion-status` but reject ordinary execution. This prevents an older
base-V128 checkpoint from silently masquerading as the current Proposed controller. Full, D5,
runtime and seven-strategy evaluation are enabled only by a later explicit production-promotion
source change after Development evidence passes.

## One current user surface

Start from:

```text
CODEX_START_HERE.md
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Stable user entrypoints:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/audit_step2_stage_ranking_current.py
scripts/run_policy_current.py           # currently fail-closed
scripts/run_seven_strategies_current.py # currently fail-closed
```

See **`CODEX_START_HERE.md`** for the exact clean-sync, stage-stop, audit and conditional-resume
commands.
