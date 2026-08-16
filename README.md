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

Recent source-strict smoke diagnostics showed that simply making the surrogate more sensitive was
not enough. The previous repair recovered full-H72 managed-flow magnitude but the historical
teacher-forced candidate/reference flow difference mixed direct actuator response with later
hydraulic feedback and could be dominated by one high-energy actuator. Stage-A TFV autograd
remained weak in direction.

The current smoke/dev method therefore separates the causal chain:

1. **direct same-prefix action response** — only the first setting-divergence transition is a local
   `setting -> managed flow` label;
2. **separate numerical scales** — temporal `q(t)-q(t-1)` scale and direct setting-response scale
   are distinct; later feedback cannot inflate the local action scale;
3. **A0 actuator pretraining** — absolute flow plus direct same-prefix magnitude/direction;
4. **A1 oracle-flow hydraulic pretraining** — actuator frozen, authoritative managed flow injected
   to teach `managed flow -> next hydraulic state`;
5. **A2 joint training** — predicted managed flow plus response-weighted direct flow/state effects;
6. **B0 autoregressive trajectory learning** — later network-feedback effects belong here;
7. **H360 exact TFV objective downstream** — no SWMM action-gradient labels are trained.

The differentiable TFV gradient is now a **Development diagnostic and eventual online solver
signal**, not the primary surrogate training target. Candidate-reference trajectory fidelity and
ranking must be useful first.

Current implementation:

```text
scripts/run_step2_current.py
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_counterfactual_first_v128.py
src/rtc/step2_counterfactual_training_v4.py
src/rtc/step2_action_identifiable_v128.py
src/rtc/step2_differentiable_v128_edge.py
src/rtc/edge_physics_current_v128.py
src/rtc/step2_train_v128_exact.py
```

## Debug-first execution

Only `--profile smoke` and `--profile dev` are currently enabled. `--profile full` intentionally
fails closed. Smoke/dev checkpoints are NONFINAL and cannot enter D5/runtime/Policy Lock.

```text
cheap gates + preflight
 -> edge-physics artifact
 -> Stage A only (A0 -> A1 -> A2)
 -> direct same-prefix flow audit
 -> direct normal-vs-oracle hydraulic audit
 -> Stage-A TFV gradient diagnostic
 -> STOP / reject or explicitly resume B0
 -> B0 trajectory/ranking/spatial checks
 -> only then consider H360 exact objective
 -> dev confirmation
 -> future explicit full promotion
```

Current Development diagnostics:

```text
scripts/audit_step2_actuator_flow_effect_current.py
    direct same-prefix flow response + separate full-H72 feedback metrics + actuator-balanced macro

scripts/audit_step2_direct_hydraulic_effect_current.py
    same-prefix normal predicted-flow vs authoritative-oracle-flow one-step hydraulic response

scripts/audit_step2_gradient_stage_current_dev.py
    TFV autograd diagnostic at Stage A/B0/objective

scripts/audit_step2_spatial_current.py
    held-out action-effect by graph distance after trajectory stages
```

Large authoritative truth arrays remain mmap-backed. Direct-pair extraction materializes only the
two required branches up to the first divergence step.

## Physics and evidence boundary

Current smoke/dev uses frozen SWMM edge descriptors and dynamic head-difference/head-gradient
messages. Missing ordinary-conduit dynamic-flow labels are not fabricated, and the incomplete
continuity proxy is not enabled as a training loss. Hydraulic-influence shortcuts/deeper GNNs are
not promoted because current evidence did not isolate a far-field-only failure.

No future realised rainfall, future SWMM state or future Internal trajectory online. No
Validation/Final/Formal/Policy Lock during Development. A future real-time claim still requires
every guarded decision below 600 s together with score==execute and target readback.

## One current user surface

Start from:

```text
CODEX_START_HERE.md
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Stable user entrypoints remain:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Runtime/seven-strategy entrypoints are retained for a future explicitly promoted model but are not
authorized for the current smoke/dev candidate. See **`CODEX_START_HERE.md`** for exact commands.
