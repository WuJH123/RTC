# Project7 current execution guide

GitHub `main` is the code source of truth after PR #95 is merged. While PR #95 is under Development review, use branch `agent/direct-tfv-optimizer-consistency-v7`. Project7 is an **idealized EPA-SWMM methodology testbed**, not a field digital twin.

## Frozen research problem

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 Direct-TFV action-value model for all 109 facilities
 -> Step3 screen all 109 facilities relative to HOLD
 -> optimize a q95-supported H120 action sequence for H360 TFV
 -> contract the decoded sequence inside D3-HOLD joint temporal support
 -> admit only when optimizer-aware one-sided TFV residual upper bound < 0
 -> execute first 10-minute target only
 -> re-observe and repeat
```

Primary objective remains **system-wide cumulative TFV minimization**. SWMM is authoritative offline truth. No PFV/Global Peak objective or gate is added. No future realised rainfall, future SWMM state, future flooding, Validation, Final or Formal truth is available online.

Frozen clock/action contract:
- model/state step = 300 s;
- control update = 600 s;
- H360 prediction = 72 five-minute steps;
- H120 free control = 12 ten-minute blocks;
- writable facilities = 109;
- all 109 are screened every decision;
- setting movement <= 0.5 per 10-minute update;
- execute first 10-minute target only, then re-observe.

## Current evidence and scientific bottleneck

Step2 V5 has useful D3 HOLD-reference cached-candidate ranking. The old threshold-free Step3 V5/q95 closed-loop evidence was inconsistent:

```text
T5_D120   TFV reduction vs No-control  +11.547%
T10_D180                              +0.615%
T20_D300                              -1.396%
```

Exact same-prefix T5 H360 replay had zero prefix/routing mismatch but only 3/6 correct signs and 3/6 false-beneficial optimizer-selected plans. Therefore the active scientific problem is not runtime execution, 109-facility coverage or simply increasing K. It is **optimizer-induced distribution shift**: a continuous optimizer can exploit action-value error and combine individually supported actuator moves into joint temporal sequences not represented by authoritative SWMM training branches.

Current classification:

```text
DEVELOPMENT_OPTIMIZER_CONSISTENCY_UNDER_TEST
```

## Step2 — keep accepted V5 weights frozen for this experiment

Current training contract:

```text
PROJECT7_DIRECT_TFV_CORE_TRAINING_V5
```

Do not retrain Step2 merely to attach action-support geometry. The accepted pairwise `V(candidate)-V(reference)` model remains the current evaluator. If the new optimizer-consistency experiment fails, the next Step2 change may be optimizer-aware hard-negative fine-tuning, but only after exact same-prefix SWMM evidence identifies that need.

## Step3 V6 — D3-HOLD joint temporal trust region + V5 residual admission

Canonical contracts:

```text
PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V6
PROJECT7_DIRECT_TFV_D3_HOLD_JOINT_SEQUENCE_SUPPORT_V1
PROJECT7_DIRECT_TFV_OPTIMIZER_AWARE_ONE_SIDED_ADMISSION_V1
```

Existing support is retained:
- per-facility TrainFit q95 first-move/sequence radii;
- D3-HOLD q95 changed-facility count ceiling;
- q90 conservative ablation; q99 diagnostic only.

V6 adds label-independent support for the **complete H120 joint temporal sequence**, derived only from D3 TrainFit HOLD-reference multi-facility branches:
- first-block L1 action mass;
- cumulative H120 L1 action mass;
- H120 temporal total variation, including departure from HOLD at the first block.

For every differentiable decoder call, the decoded sequence is radially contracted toward the current HOLD target until all three metrics are inside the selected D3 support quantile. The contracted sequence is the sequence scored by Step2 and the sequence sent to runtime, so `score == execute` is preserved. This is a support/trust-region constraint, not a TFV-performance threshold.

After optimization, V5 optimizer-aware one-sided admission is still applied:

```text
upper = predicted_delta_TFV + calibrated_one_sided_residual_margin
if upper < 0 and first 10-minute block changes:
    execute first block
else:
    HOLD
```

The residual margin combines rainfall-disjoint D3 HOLD cached residuals and exact same-prefix optimizer-selected Development replay residuals. The small optimizer replay sample contributes only its empirical maximum residual and makes no coverage claim.

## Counterfactual replay

Runtime telemetry intentionally keeps `PROJECT7_DIRECT_TFV_COUNTERFACTUAL_PLAN_TELEMETRY_V2` for compatibility with the existing V2 selector. New joint-support fields are additive. The selector must retain both accepted and rejected raw optimizer plans for exact same-prefix H360 replay.

The replay should test:
- true-beneficial fraction among admitted plans;
- true-beneficial / false-beneficial fraction among rejected plans;
- prediction-vs-SWMM sign accuracy and MAE;
- prefix state/target/current/statistics equality;
- HOLD target equality;
- routing error.

Any prefix mismatch remains `COUNTERFACTUAL_REPLAY_P0`.

## Frozen Development assets

```text
GRAPH
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

BASE_CACHE
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

RAIN
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

STATE
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz

STEP1
E:\RTC_sewer\Project7\study_v069\models\step1\step1_model.pt

SENSORS
E:\RTC_sewer\Project7\inputs\contracts\sensor_nodes.txt

CONFIG
E:\RTC_sewer\Project7\study_v069\contracts\controller_resolved.json
```

Auto-discover and SHA/contract-validate the accepted Step2 V5 checkpoint and the existing PR #94 admission artifact. Do not guess lineage and do not regenerate large SWMM datasets just because a filename contains an older version tag.

## Canonical Development order

1. Protect local uncommitted replay-runner work before syncing GitHub. Do not blindly `git clean` or pop a stash over a changed tree.
2. Sync the PR #95 branch (or merged descendant), record HEAD, install, run full pytest and current Ruff gate.
3. Verify CUDA, but do **not** retrain Step2.
4. Build the checkpoint-bound D3-HOLD joint-sequence support artifact with `scripts/build_direct_tfv_sequence_support_current.py`.
5. Reuse the existing optimizer-aware admission artifact only if its Step2 checkpoint SHA and split contract match; otherwise rebuild it from the existing exact T5 H360 replay plus rainfall-disjoint D3 residuals.
6. Run `scripts/run_step3_direct_tfv_solver_calibrated_current.py` with both `--admission-calibration` and `--sequence-support`, canonical q95.
7. Require zero engineering, per-facility support and joint-sequence-support violations. Report how often the new joint-sequence support is binding.
8. T5 may be rerun only as calibration/engineering consistency smoke because its prior optimizer replay contributed to admission calibration.
9. Run T10_D180 and T20_D300 as fresh post-calibration Development closed loops.
10. Audit each closed loop with `scripts/audit_direct_tfv_calibrated_closed_loop_current.py`.
11. Generate V2 counterfactual manifests from fresh T10/T20 metadata and exact same-prefix H360 SWMM replay for accepted and rejected optimizer plans.
12. Primary scientific gate: both fresh events must be non-inferior to No-control in TFV, with zero execution/support violations and materially reduced false-beneficial optimizer actions.
13. Only after No-control consistency is restored should Internal/Auto-RBC competitiveness be reassessed.
14. Stop before Validation/Final/Formal/Policy Lock.

## If V6 still fails

Do not add PFV, Global Peak, q99 canonical, unrestricted K, future rainfall truth, RBC imitation/fallback, or weaker baselines. First classify the remaining failure:

- if support contraction is frequently binding and fresh exact replay becomes reliable but event TFV remains poor: inspect H360 value-estimand vs 10-minute receding-control mismatch and run a preregistered horizon ablation;
- if admitted optimizer plans remain false-beneficial inside support: create a small optimizer-aware hard-negative SWMM dataset from fresh Development replays and fine-tune Step2 against those optimizer-query examples;
- if both are reliable but Auto-RBC still wins: classify baseline competitiveness separately rather than changing the objective to force a win.

## Scientific boundaries

- TFV remains the only optimization objective in this Development experiment;
- PFV and Global Peak remain diagnostics/reporting only;
- no online SWMM candidate evaluation;
- no future realised rainfall online;
- no q99 canonical promotion from surrogate predictions alone;
- no event-wise Auto-RBC/EFD tuning;
- no use of Validation/Final/Formal/Policy Lock for support/admission calibration;
- no production promotion until fresh Development evidence supports optimizer consistency and No-control non-inferiority.
