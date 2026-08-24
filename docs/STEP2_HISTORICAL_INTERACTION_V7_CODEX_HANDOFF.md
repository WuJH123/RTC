# Project7 Step2 V7 — Codex handoff

## Why V7 exists

V6 tested one precise hypothesis: preserve the MAIN single-facility Direct-TFV backbone during JOINT/CONTROL.  Existing-data DEV evidence showed a real but limited D2 improvement (rank +0.0213, pairwise +0.0082, Top1 +0.125, regret -1203.9 m3), while D3 worsened (rank -0.0381, pairwise -0.0129, regret +637.6 m3, harmful selection 1/32 -> 3/32).  Therefore the remaining blocker is not MAIN forgetting alone; it is the representation of joint action interactions across states.

V7 is the single bounded follow-up already allowed by the V6 experiment contract.  It ports only the historically supported interaction ideas from V4.1 into the current Direct-TFV value architecture:

- active-aware latent pooling;
- actuator-identity-weighted signed action moment;
- elementwise second-order latent pair moment;
- signed / absolute / quadratic action moments;
- explicit candidate/reference antisymmetrization;
- exact zero interaction for HOLD and single-facility perturbations.

The V5 facility main pathway, causal inputs, exact SWMM delta-TFV labels, split, seed, losses, epochs, learning rates and target scale remain unchanged.

## Git / scientific boundary

- branch: `agent/step2-historical-retrain-v6`
- PR: `#125`, keep OPEN + DRAFT
- base main: `f7eb9bfc91967528dfd218d620f4c427f186f913`
- frozen V5 checkpoint SHA: `3a05704812a07a914d0ce9d8d026f6c84a4dbed646743f95d27726b29c3a544a`
- rejected V6 checkpoint SHA: `34d32c52ab6f136cc7f2438748e57c8af9d87659bef873423cc82a6077c73812`

Do not access Validation, Final, Policy Lock or old Formal outcomes.  Do not run SWMM.  Do not change seed, epochs, losses, split or learning rates.  V7 is the last existing-data architecture follow-up authorized by this branch.

## Cheap gates

Run from repo root:

```powershell
python -m py_compile src/rtc/step2_tfv_value_historical_v7.py src/rtc/step2_tfv_value_training_historical_v7.py scripts/run_step2_tfv_value_historical_interaction_v7.py
python -m pytest -q tests/test_step2_historical_interaction_v7.py tests/test_step2_historical_retrain_v6.py tests/test_compare_step2_historical_retrain_v6.py tests/test_direct_tfv_core_training.py tests/test_direct_tfv_value.py
python scripts/run_step2_tfv_value_historical_interaction_v7.py --help
python scripts/lint_current_surface.py
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Any failure: STOP and return evidence to Web-GPT.  Do not patch scientific logic locally.

## Run V7 once

Recover the exact graph/cache/causal-store paths and V5 training CLI values from the successful V6 invocation / V5 report.  Use the same DEV profile and seed 42.  Output to a new directory, for example:

`E:\RTC_sewer\Project7\study_v069\direct_tfv_historical_interaction_v7_dev_<timestamp>`

Run exactly once:

```powershell
python scripts/run_step2_tfv_value_historical_interaction_v7.py `
  --profile dev `
  --graph "<SAME V5/V6 GRAPH>" `
  --cache-manifest "<SAME V5/V6 BASE CACHE>" `
  --d4-fit-cache "<SAME V5/V6 D4 FIT CACHE>" `
  --d4-audit-cache "<SAME V5/V6 D4 AUDIT CACHE>" `
  --causal-store "<SAME V5/V6 CAUSAL RAINFALL STORE>" `
  --causal-state-store "<SAME V5/V6 CAUSAL STATE STORE>" `
  --out-dir "<NEW V7 OUTPUT DIR>" `
  --device cuda `
  <THE SAME EXPLICIT EPOCH/LR FLAGS USED BY V5/V6, IF ANY>
```

No sweep and no retry with changed hyperparameters.

## Compare against V5

Reuse the existing comparator:

```powershell
python scripts/compare_step2_historical_retrain_v6.py `
  --baseline-report "<V5 STEP2_DIRECT_TFV_VALUE_REPORT.json>" `
  --candidate-report "<V7 STEP2_DIRECT_TFV_VALUE_REPORT.json>" `
  --out "<V7 OUTPUT DIR>\STEP2_HISTORICAL_INTERACTION_V7_COMPARISON.json"
```

Interpret the comparator as the same preregistered gate:

- D2 rank >= V5;
- D2 pairwise >= V5;
- D2 harmful fraction <= V5;
- D3 pairwise > V5;
- D3 sign >= V5;
- D3 selected regret < V5;
- D3 harmful fraction <= V5;
- D3 rank >= 0.70 is diagnostic only.

If the comparator uses a nonzero reject exit code, treat the JSON verdict as authoritative and STOP.

## Required report

Return:

- Git HEAD / origin branch / PR state;
- all cheap-gate results;
- V5 SHA and V7 SHA;
- exact selected group counts and proof of zero role overlap;
- V5 and V7 D2/D3 rank, pairwise, sign, Top1, harmful fraction, regret and MAE;
- deltas V7-V5;
- training history MAIN/JOINT/CONTROL final metrics;
- structural test status: HOLD zero, single-action interaction zero, swap antisymmetry, JOINT/CONTROL interaction-only trainability;
- comparator JSON path and gate verdict;
- `NEW_SWMM_RUNS=0`;
- `VALIDATION_ACCESSED=false`;
- `FINAL_ACCESSED=false`;
- `FORMAL_USED_FOR_TUNING=false`.

If V7 fails, final verdict must be:

`EXISTING_DATA_STEP2_ARCHITECTURE_FOLLOWUP_NOT_JUSTIFIED`

and STOP.  Do not start V8, do not generate data, do not change Step3.

If V7 passes, STOP and return to Web-GPT before any Step3 work.  A passing Step2 still requires a new Development-only Step3 rank/boundary lineage before closed-loop evaluation.
