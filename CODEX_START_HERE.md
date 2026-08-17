# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. GitHub `main` is the
code source of truth. Do not infer the active method from the highest V-number in a filename.

## 1. Frozen research problem

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 learn 109-facility ACTION -> future delta TFV
 -> HOLD/action selection guard
 -> Step3 choose lower-TFV 109-facility action sequence every 10 min
 -> execute first 10 min target only
 -> observe again and repeat
```

Primary objective: **system-wide cumulative TFV minimization**. SWMM is authoritative truth for
offline labels and final evaluation, not the online candidate evaluator.

Frozen timing/action contract:

- model/state step = 300 s;
- control update = 600 s;
- prediction horizon = H360 = 72 model steps;
- free-control horizon = H120 = 12 ten-minute blocks;
- writable continuous facilities = 109;
- decision variables = 12 x 109 = 1308;
- execute first 10 minutes only.

## 2. Current Step2 — pairwise Direct-TFV V2

The learned model receives causal Step1 current state, causal rainfall forecast, previous managed
flow, and the complete reference and complete candidate H360 sequences. It predicts

```text
delta TFV = V(candidate) - V(reference)
```

with

```text
sum(109 facility value differences) + multi-facility interaction value difference.
```

The complete reference and complete candidate H360 sequences use the same network. Candidate ==
reference is exactly zero and swapping candidate/reference negates the predicted delta exactly.
Full hydraulic trajectory prediction remains auxiliary/ablation only.

Development reference families are:

```text
D2       base-action/single-actuator reference
D3       HOLD reference
D4 FIT   causal Sparse-RBC anchor
D4 AUDIT causal Sparse-RBC anchor
future Step3 HOLD reference
```

## 3. Direct-TFV V2 evidence is complete enough to move the bottleneck

The full V2 DEV run used every admitted existing Development group:

```text
112 D2 FIT
112 D3 FIT
32 D2 held-out
32 D3 held-out
33 D4 FIT
15 D4 AUDIT
```

and preserved useful action ordering without generating new SWMM data.

Key full-DEV metrics:

```text
split       rank   pairwise   selected true dTFV   harmful
D2 holdout  0.299  0.614      -6,028.9 m3          28.1%
D3 holdout  0.340  0.623     -30,221.8 m3           9.4%
D4 FIT      0.419  0.659        +311.6 m3           54.5%
D4 AUDIT    0.326  0.640        +315.9 m3           46.7%
```

This is materially better than legacy V128 ordering, especially D3/D4. Therefore **do not return
to the hydraulic world-model or add model capacity merely because raw selection is imperfect**.

The current bottleneck is narrower: **HOLD/action selection calibration**.

Evidence:

- every evaluated split had `hold_selected_fraction = 0`;
- D4 FIT contained 9 oracle-HOLD groups and raw selection acted on all 9;
- D4 FIT and D4 AUDIT raw selected true delta TFV were slightly positive on average;
- D4 ranking itself is now useful, so this is not primarily a reference-encoder failure.

## 4. Current selection-calibration design

The value model is frozen. The current next runner is:

```text
scripts/run_step2_selection_calibration_current.py
```

It does **not** retrain Step2 and does **not** launch SWMM.

It takes the frozen full-DEV V2 checkpoint and reuses the existing D3 internal holdout because D3
has the same HOLD reference family as future Step3. The D3 holdout is split again by rainfall group
into disjoint selection-calibration and selection-audit subsets.

For each calibration group, keep the model-best non-HOLD candidate and define

```text
selected residual = authoritative true delta TFV - predicted delta TFV.
```

A finite-sample one-sided upper residual quantile gives `margin_m3`. Future selection admits the
model-best action only if

```text
predicted delta TFV + margin_m3 < 0.
```

Otherwise return HOLD.

Important boundaries:

- D4 AUDIT never enters model training, target-scale fitting, or margin fitting;
- the D4 results are stress tests because D4 uses Sparse-RBC reference, not HOLD;
- this Development margin is not a formal coverage claim under D3->D4 distribution shift;
- no Validation/Final/Formal/Policy Lock is accessed;
- no new SWMM is generated;
- runtime and seven-strategy execution remain disabled.

## 5. Current code surface

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/project7_current_lint_surface.json
scripts/run_step2_current.py
scripts/run_step2_tfv_value_current.py
scripts/run_step2_selection_calibration_current.py
src/rtc/step2_tfv_value.py
src/rtc/step2_tfv_value_training.py
src/rtc/step2_tfv_selection.py
src/rtc/step3_tfv_value_mpc.py
scripts/audit_facility_tfv_influence_current.py
```

Legacy V128 remains reproducibility/ablation only.

## 6. Frozen Development assets

```text
<GRAPH>
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

<BASE_CACHE>
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

<D4_FIT>
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json

<D4_AUDIT>
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json

<RAIN>
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

<STATE_V2>
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz

<V2_DEV_CHECKPOINT>
E:\RTC_sewer\Project7\study_v069\direct_tfv_pairwise_v2_dev_20260817_010351\step2_direct_tfv_value_dev.pt
```

Expected V2 DEV checkpoint SHA256:

```text
A0D0C44CC8445C96D29E3BF76C83A781098B9055708F10A8FD3803FFCE91B314
```

## 7. Hard-sync and cheap gates

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git reset --hard origin/main
git clean -fd

python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python scripts/lint_current_surface.py
python scripts/run_step2_selection_calibration_current.py --help
python scripts/run_policy_current.py --promotion-status
python scripts/run_seven_strategies_current.py --promotion-status
```

Runtime and seven-strategy execution must remain disabled.

## 8. Current expensive action — selection calibration only

Do not rerun Step2 training by default. Use the existing full-DEV checkpoint:

```powershell
$Study="E:\RTC_sewer\Project7\study_v069"
$Checkpoint="$Study\direct_tfv_pairwise_v2_dev_20260817_010351\step2_direct_tfv_value_dev.pt"
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Run="$Study\direct_tfv_selection_guard_$Stamp"

python scripts/run_step2_selection_calibration_current.py `
  --checkpoint $Checkpoint `
  --graph "$Study\formal_assets\graph_schema.npz" `
  --cache-manifest "$Study\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json" `
  --d4-fit-cache "$Study\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json" `
  --d4-audit-cache "$Study\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json" `
  --causal-store "$Study\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz" `
  --causal-state-store "$Study\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz" `
  --out-dir $Run `
  --device cuda `
  --alpha 0.10
```

Expected output:

```text
STEP2_DIRECT_TFV_SELECTION_REPORT.json
```

## 9. Read raw versus guarded metrics

For each of

```text
d3_selection_calibration
d3_selection_audit
d2_holdout_stress
d4_fit_stress
d4_audit_stress
```

compare `raw` and `guarded`:

```text
action_selected_fraction
hold_selected_fraction
oracle_hold_groups
false_action_when_hold_oracle_fraction
selected_beneficial_fraction
selected_harmful_fraction
selected_true_delta_tfv_m3
selected_regret_m3
```

Also report:

```text
calibration.margin_m3
calibration.calibration_groups
calibration.calibration_action_groups
calibration.residual_quantile_rank
D3 calibration rainfall groups
D3 audit rainfall groups
```

## 10. Decision after the guard

Do not judge the guard by rank; it intentionally does not change candidate ranking.

A useful guard should:

1. keep nonzero action rate on the disjoint D3 HOLD-reference audit;
2. keep D3 selected true delta TFV beneficial;
3. reduce harmful/false-action behavior rather than merely choosing HOLD everywhere;
4. make D4 stress-test selected true delta TFV non-harmful or clearly closer to zero without using D4 AUDIT to fit the margin.

If those conditions hold, the next code task is a **Development Step3 solver test** using the
calibrated margin through the already existing
`DirectTFVMPCDesign.minimum_predicted_improvement_m3` hook. Do not run a SWMM closed loop yet.

If the only way to avoid D4 harm is to select HOLD almost everywhere, stop. The next missing evidence
is then a **small targeted HOLD-reference first-move panel** rather than another large random SWMM
dataset or a larger neural network.

If D3 HOLD-reference audit itself remains poorly calibrated, then consider an explicit HOLD-focused
training/selection loss. Do not modify the value architecture first.
