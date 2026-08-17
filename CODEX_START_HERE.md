# Project7 current execution guide

GitHub `main` is the code source of truth. Do not infer the active method from the highest V-number
in a filename.

## 1. Frozen research problem

Project7 is an **idealized EPA-SWMM methodology testbed**, not a field digital twin.

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 learn 109-facility ACTION -> future delta TFV
 -> explicit HOLD/action decision calibration
 -> Step3 screen all 109 facilities
 -> optimize a TrainFit-supported dynamic active set
 -> execute first 10-minute target only
 -> observe again and repeat
```

Primary objective: **system-wide cumulative TFV minimization**. SWMM is authoritative truth for
offline labels and authoritative control evaluation; SWMM is not the online candidate evaluator.

Frozen clock/action contract:

- model/state step = 300 s;
- control update = 600 s;
- H360 prediction = 72 model steps;
- H120 free control = 12 ten-minute blocks;
- writable facilities = 109;
- every facility is eligible and screened every decision;
- execute only the first 10-minute target, then re-observe.

## 2. What the completed V2 evidence means

Direct exact-delta-TFV learning is the correct core target. Full V2 Development used
112/112/32/32/33/15 D2/D3/D4 groups and produced useful ordering:

```text
D2 holdout  rank 0.299  pairwise 0.614  selected true dTFV  -6,028.9 m3
D3 holdout  rank 0.340  pairwise 0.623  selected true dTFV -30,221.8 m3
D4 FIT      rank 0.419  pairwise 0.659  selected true dTFV    +311.6 m3
D4 AUDIT    rank 0.326  pairwise 0.640  selected true dTFV    +315.9 m3
```

The model therefore learned useful action ordering, especially D3/D4, but raw argmin selection was
too eager. The first scalar residual guard fixed harmful actions only by becoming overconservative:
with 16 D3 calibration groups and alpha=0.10 its finite-sample rank was 16/16, so the margin became
the maximum residual (27,986 m3) and D2/D4 collapsed to HOLD.

Do **not** return to the V128 hydraulic world model. The remaining problem is control-oriented:
learn the absolute HOLD boundary and prevent Step3 from exploiting unsupported action sequences.

## 3. Current Step2 — pairwise value model + selection-aware training

The model architecture remains:

```text
causal Step1 current state
+ causal rainfall forecast
+ previous managed flow
+ complete H360 reference sequence
+ complete H360 candidate sequence
 -> delta TFV = V(candidate) - V(reference)
```

with 109 facility value differences plus a multi-facility interaction value difference.

Structural contracts remain:

```text
candidate == reference -> exact zero
swap(candidate, reference) -> exact sign reversal
single changed facility -> interaction residual exactly zero
```

Current training is `PROJECT7_DIRECT_TFV_SELECTION_AWARE_TRAINING_V3`:

1. **MAIN** — exact single-facility branches train the 109 facility effects.
2. **JOINT** — multi-facility branches train the interaction residual.
3. **SELECTION** — low-learning-rate late-representation/head fine-tuning on all TrainFit branches
   adds explicit HOLD-vs-action sign loss and oracle-choice loss.

Harmful actions receive extra sign-loss weight because a false-beneficial prediction can trigger a
real control move; a missed benefit falls back to HOLD.

Full DEV now fails closed unless TrainFit contains exact single-facility evidence for **109/109**
facilities. The checkpoint also stores TrainFit action support:

- per-facility first-move absolute q95;
- per-facility sequence absolute q95;
- joint changed-facility count quantiles;
- per-facility single-branch/rainfall coverage;
- TrainFit delta-TFV magnitude range.

D4 AUDIT never enters model training, target scaling, support derivation, or decision calibration.

## 4. Current HOLD/action calibration

The old max-residual scalar guard is history/ablation only.

Current runner:

```text
scripts/run_step2_selection_threshold_current.py
```

uses the same rainfall-disjoint D3 HOLD-reference calibration/audit split, but directly calibrates
the online decision variable: the **minimum predicted improvement** needed before ACTION is admitted.

Threshold selection is lexicographic on the calibration subset:

1. minimise harmful selected actions;
2. minimise false actions when HOLD is authoritative oracle;
3. among equally safe thresholds, minimise authoritative selected delta TFV;
4. minimise regret;
5. prefer useful action rate.

This is Development calibration, not a formal probabilistic guarantee.

## 5. Current Step3 — all-109 screening + trust-region MPC

Canonical module:

```text
src/rtc/step3_tfv_value_mpc_v2.py
```

Canonical solver audit:

```text
scripts/run_step3_direct_tfv_solver_current.py
```

At every 10-minute decision Step3:

1. builds a HOLD H360 reference;
2. evaluates small first-move up/down probes for **all 109 facilities** inside each facility's
   TrainFit first-move q95 support;
3. ranks all facilities by learned first-move delta TFV;
4. forms a dynamic active set. Default active-set size is the TrainFit median number of changed
   facilities among joint counterfactuals;
5. runs L-BFGS-B only for that active set over 12 free 10-minute blocks;
6. constrains every target to physical min/max, <=0.5 update movement, and per-facility TrainFit q95
   action support;
7. accepts the solution only if predicted improvement exceeds the calibrated D3 HOLD-reference
   minimum-improvement threshold;
8. otherwise returns HOLD;
9. executes only the first 10-minute target and re-observes.

This preserves the scientific claim that all 109 facilities are learned and considered, while
preventing an unrestricted 1308-dimensional optimizer from inventing unsupported joint actions.

## 6. Frozen Development assets

```text
GRAPH
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

BASE_CACHE
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

D4_FIT
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json

D4_AUDIT
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json

RAIN
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

STATE
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz
```

## 7. Development execution order

Always hard-sync and run cheap gates first:

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git reset --hard origin/main
git clean -fd
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python scripts/lint_current_surface.py
python scripts/run_step2_current.py --help
python scripts/run_step2_selection_threshold_current.py --help
python scripts/run_step3_direct_tfv_solver_current.py --help
python scripts/run_policy_current.py --promotion-status
python scripts/run_seven_strategies_current.py --promotion-status
```

Then, in order:

```text
A. selection-aware Direct-TFV smoke
B. if no regression, full selection-aware DEV
C. require 109/109 single-facility TrainFit coverage
D. calibrate rainfall-disjoint D3 HOLD/action minimum-improvement threshold
E. run all-109 screened trust-region Step3 solver-only audit on D3 selection-audit states
F. if solver audit passes, run a small authoritative SWMM first-move probe
G. only then wire one causal 10-minute closed loop
```

Do not skip directly to Formal/Final/Policy Lock.

## 8. Current scientific boundaries

- no future realised rainfall online;
- no future SWMM state/flood truth online;
- no D4 AUDIT fitting of any kind;
- no new gradient labels;
- no return to full future-hydraulic trajectory as the primary target;
- no unrestricted 12x109 online search outside TrainFit support;
- runtime/seven-strategy production promotion remains fail-closed;
- Validation/Final/Formal/Policy Lock remain untouched during Development.

## 9. What decides success

Step2 is not judged only by rank. Read together:

```text
rank
pairwise
sign
selected_harmful_fraction
selected_true_delta_tfv_m3
selected_regret_m3
false_action_when_hold_oracle_fraction
109/109 single-facility TrainFit coverage
```

Step3 solver-only success requires:

```text
screened_facility_count = 109 for every group
support_violation_count = 0
engineering_violation_count = 0
action_selected_count > 0
solver time < one 600-s control interval
```

The next authoritative scientific gate is SWMM truth for the proposed first moves. Solver prediction
alone is not final evidence.
