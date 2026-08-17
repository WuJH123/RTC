# Project7 current execution guide

GitHub `main` is the code source of truth. Do not infer the active method from the highest V-number
in a filename.

## 1. Frozen research problem

Project7 is an **idealized EPA-SWMM methodology testbed**, not a field digital twin.

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 learn 109-facility ACTION -> future delta TFV
 -> Step3 evaluate all 109 facilities relative to HOLD
 -> optimise the predicted-beneficial multi-facility subset
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

## 2. What the completed V3 evidence means

Full V3 Development used all 112/112/32/32/33/15 D2/D3/D4 groups and proved complete
single-facility learning coverage: **109/109** facilities, 14 rainfall groups per facility.

The runtime-reference-aligned D3 holdout is already useful:

```text
rank                    0.367
pairwise                0.637
selected harmful        3.1%
selected true delta TFV -31,396.6 m3
regret                   4,319.7 m3
```

D4 FIT/AUDIT still shows about 50% harmful raw selection, but D4 uses a Sparse-RBC reference whereas
future Step3 uses HOLD. D4 is therefore a **reference-shift stress diagnostic**, not a reason to stop
the HOLD-reference control path.

Do not return to the V128 hydraulic world model and do not generate large new datasets before the
current HOLD-reference Step3 and authoritative SWMM path is tested.

## 3. Current Step2 — core Direct-TFV V4 training

Model architecture remains:

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

Current training is `PROJECT7_DIRECT_TFV_CORE_TRAINING_V4`:

1. **MAIN** — exact single-facility branches learn the 109 facility effects. Regression is averaged
   per facility before the global mean so a few facilities with 128-145 branches cannot dominate
   facilities that have about 22-27 branches.
2. **JOINT** — multi-facility branches fit the interaction residual. No HOLD/action decision bias is
   added in this stage.
3. **CONTROL** — only D3 HOLD-reference TrainFit groups fine-tune the late value representation with
   a symmetric sign loss. There is no extra harmful-action safety weight and no oracle-top1 loss in
   the canonical objective.

Exact cached-candidate `top1_fraction` remains a diagnostic only. The actual continuous Step3 needs
correct value/sign and low selected regret; it does not need to reproduce the identity of one cached
branch exactly.

Full DEV fails closed only if single-facility coverage is below **109/109**.

## 4. Current Step3 — all-109 receding MPC V3

Canonical module:

```text
src/rtc/step3_tfv_value_mpc_v3.py
```

Canonical solver audit:

```text
scripts/run_step3_direct_tfv_solver_current.py
```

At every 10-minute decision Step3:

1. builds a HOLD H360 reference;
2. for **all 109 facilities**, evaluates up/down probes at half and full supported first-move radius
   under both a one-block **pulse** pattern and a **persistent** pattern; the latter preserves
   facilities whose TFV benefit is delayed and would be missed by a single 10-minute pulse;
3. keeps facilities whose best pulse/persistent probe has predicted delta TFV < 0;
4. ranks those predicted-beneficial facilities;
5. forms a dynamic active set. Default ceiling is the TrainFit q90 joint changed-facility count,
   preserving more control freedom than the previous q50=8 default;
6. runs L-BFGS-B for that active set over 12 free 10-minute blocks;
7. enforces physical min/max, <=0.5 change per update, and the action magnitudes represented by
   TrainFit data;
8. if the optimised predicted delta TFV is < 0, execute the first 10-minute target; otherwise HOLD;
9. re-observe and solve again after 10 minutes.

There is **no separate selection-threshold calibration stage** in the canonical path. HOLD is exact
zero, so `predicted delta TFV < 0` is the direct decision implied by the scientific objective.

All 109 facilities are learned and evaluated every decision. This does **not** mean all 109 must be
changed simultaneously; the current hydraulic state and rainfall determine the useful subset.

## 5. Frozen Development assets

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

## 6. Development execution order

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
python scripts/run_step3_direct_tfv_solver_current.py --help
python scripts/run_policy_direct_tfv_development.py --help
python scripts/run_policy_current.py --promotion-status
python scripts/run_seven_strategies_current.py --promotion-status
```

Then:

```text
A. core Direct-TFV smoke
B. if no catastrophic D2/D3 regression, full core DEV
C. require 109/109 single-facility TrainFit coverage
D. judge Step2 mainly from D3 HOLD-reference held-out selected true dTFV, harmful fraction, regret,
   pairwise and sign; D4 remains diagnostic
E. run Step3 solver audit directly from the Step2 checkpoint
F. if solver generates executable non-HOLD actions inside one control period, run one authoritative
   Development SWMM closed loop
G. compare authoritative TFV against a matched baseline
```

Do not skip directly to Formal/Final/Policy Lock.

## 7. Scientific boundaries

- no future realised rainfall online;
- no future SWMM state/flood truth online;
- no D4 AUDIT fitting;
- no gradient labels;
- no return to full future-hydraulic trajectory as the primary target;
- no extra calibrated threshold that suppresses otherwise predicted-beneficial actions;
- production/seven-strategy promotion remains fail-closed;
- Validation/Final/Formal/Policy Lock remain untouched during Development.

## 8. What decides success

Step2 primary evidence:

```text
109/109 single-facility TrainFit coverage
D3 HOLD-reference pairwise/sign
D3 selected_harmful_fraction
D3 selected_true_delta_tfv_m3
D3 selected_regret_m3
```

`top1_fraction` and D4 are diagnostics, not promotion gates.

Step3 solver evidence:

```text
screened_facility_count = 109 for every group
predicted-beneficial facility count > 0 on useful states
action_selected_count > 0
engineering/support violations = 0
solver time < 600 s
```

The decisive scientific gate is authoritative SWMM:

```text
TFV_proposed < TFV_matched_baseline
```

while the exact scored first 10-minute target is the one actually written and read back.
