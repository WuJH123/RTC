# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. GitHub `main` is the
code source of truth. Do not infer the active method from the highest V-number in a filename.

## 1. Core research problem

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

The shortest valid research pipeline is:

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 learn 109-facility ACTION -> future delta TFV
 -> Step3 choose lower-TFV 109-facility action sequence every 10 min
 -> execute first 10 min target only
 -> observe again and repeat
```

The primary objective is **system-wide cumulative TFV minimization**. SWMM remains authoritative
truth for offline labels and final evaluation, but it must not be called repeatedly inside online
candidate search.

Frozen clock/action contract:

- state/model step = 300 s;
- control update = 600 s;
- prediction horizon = H360 = 72 model steps;
- free-control horizon = H120 = 12 ten-minute blocks;
- writable continuous facilities = 109;
- decision variables = 12 x 109 = 1308;
- execute first 10 minutes only.

## 2. What Step1, Step2 and Step3 are responsible for

### Step1 — see the current system cheaply

Step1 reconstructs the current full-network hydraulic state from sparse causal observations.
It replaces the impossible assumption that every node is measured online. Step1 does **not** choose
control actions and does not predict TFV value by itself.

### Step2 — learn control value directly

Step2 receives:

```text
causal Step1 current state
+ causal rainfall forecast
+ current/reference actuator settings and managed flow
+ candidate H360 sequence for 109 facilities
```

and predicts:

```text
delta TFV = TFV(candidate) - TFV(reference)
```

The current direct model decomposes this into:

```text
sum(109 facility main effects) + multi-facility interaction residual
```

Single-actuator same-prefix SWMM counterfactuals supervise facility main effects. Multi-actuator
counterfactuals supervise only the residual interaction term after the main-effect stage.

Full H360 hydraulic trajectory prediction is now an **auxiliary diagnostic/ablation**, not a
prerequisite for learning the TFV control objective.

### Step3 — optimize the learned value

Step3 directly minimizes Step2 predicted delta TFV over the same 12 x 109 bounded targets that
would be executed. The existing differentiable engineering decoder preserves min/max and update
bounds. If no finite predicted improvement is found, Development Step3 returns HOLD.

## 3. Why the architecture changed

The authoritative facility audit established physical controllability from existing data:

- 109/109 facilities have exact same-prefix single-actuator evidence;
- all 18 rainfall x 109 facility cells are tested;
- 82 facilities show at least one meaningful TFV influence;
- 37 facilities have sampled beneficial evidence;
- delayed H60/H120/H360 benefits exist;
- multi-actuator candidates can have large joint TFV effects.

At the same time the legacy V128 B0 world-model path executed successfully but did not learn useful
TFV ordering: D2 was near random and D3/D4 ranking was worse. Therefore the bottleneck is no longer
"prove the full hydraulic rollout can run". The immediate task is **learn the SWMM-labelled action
value that Step3 actually optimizes**.

Do not interpret this change as saying hydraulics are irrelevant. Current hydraulic state remains a
critical conditioning input. The change removes an unnecessary requirement that thousands of
future node states must all be predicted accurately before a candidate can be ranked by TFV.

## 4. Data roles

```text
D0  baseline hydraulic/flood opportunity context; not facility attribution
D1  broad state/action coverage; not direct facility credit assignment
D2  primary long-duration single-facility main-effect supervision
D3  temporally diverse single-facility supervision + multi-facility interactions
D4 FIT  local/first-move operating-region supervision
D4 AUDIT untouched Development audit; NEVER training
D5  optional later solver-sensitivity diagnostic; not a prerequisite
```

The historical `1% of reference TFV` practical threshold is **reporting only**. Do not convert
continuous exact delta-TFV labels to zero during training. Many physically informative effects are
smaller than 1% of a multi-million-m3 event total.

## 5. Current code surface

Canonical current implementation:

```text
configs/step2_current_contract.json
scripts/run_step2_current.py
scripts/run_step2_tfv_value_current.py
src/rtc/step2_tfv_value.py
src/rtc/step2_tfv_value_training.py
src/rtc/step3_tfv_value_mpc.py
scripts/audit_facility_tfv_influence_current.py
```

Legacy V128 world-model files remain in the repository for ablation/history, including:

```text
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_counterfactual_first_v128.py
src/rtc/step2_counterfactual_training_v5.py
src/rtc/step2_differentiable_v128_edge.py
```

They are **not** the canonical current Step2 entrypoint.

## 6. Scientific boundaries

Until explicit direct-value promotion:

- no Validation, Final, Formal or Policy Lock;
- no D4-AUDIT outcomes in training or target-scale fitting;
- no future realised rainfall online;
- no future SWMM state/flood truth online;
- no new SWMM data merely because architecture changed;
- no runtime/seven-strategy promotion before direct-value held-out ranking is useful;
- do not return to gradient tuning if TFV candidate ordering is poor.

## 7. Frozen Development assets

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
```

## 8. Hard-sync and cheap gates

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
python scripts/audit_facility_tfv_influence_current.py --help
```

Do not build the legacy edge-physics artifact for the direct-value run. It is not an input to the
current Step2 model.

## 9. First expensive action — direct-value smoke

```powershell
$Study="E:\RTC_sewer\Project7\study_v069"
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Run="$Study\direct_tfv_value_smoke_$Stamp"

python scripts/run_step2_current.py `
  --profile smoke `
  --graph "$Study\formal_assets\graph_schema.npz" `
  --cache-manifest "$Study\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json" `
  --d4-fit-cache "$Study\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json" `
  --d4-audit-cache "$Study\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json" `
  --causal-store "$Study\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz" `
  --causal-state-store "$Study\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz" `
  --out-dir $Run `
  --device cuda
```

Expected outputs:

```text
STEP2_DIRECT_TFV_VALUE_REPORT.json
step2_direct_tfv_value_dev.pt
```

## 10. What decides whether the new Step2 is working

Read these metrics first:

```text
rank
pairwise
top1_fraction
delta_tfv_mae_m3
selected_regret_m3
```

for:

```text
trainfit_d2
trainfit_d3
internal_holdout_d2
internal_holdout_d3
d4_fit
d4_audit
```

The immediate comparison is the failed legacy B0 Development evidence, approximately:

```text
D2 rank       0.056, pairwise 0.518
D3 rank      -0.160, pairwise 0.443
D4 FIT rank  -0.411, pairwise 0.327
D4 AUDIT     -0.476, pairwise 0.319
```

Do not invent a universal hard score such as rank > 0.8. The first question is whether direct SWMM
value supervision produces a clear, reproducible improvement in candidate ordering and selected
regret over the old indirect trajectory path.

If smoke improves materially, run the same command with `--profile dev`.

If direct-value holdout ranking remains random, inspect **state/action representation and temporal
coverage**. Do not return to the old A0/A1/A2/B0 hydraulic curriculum by default.

## 11. Step3 status

`src/rtc/step3_tfv_value_mpc.py` already implements the minimal direct-value optimization surface:

```text
Step1 state + causal rain + HOLD reference
 -> bounded 12 x 109 candidate
 -> Step2 predicted delta TFV
 -> L-BFGS-B
 -> accept only predicted improvement, otherwise HOLD
```

This is not yet production-promoted. First prove Step2 held-out value ordering. Then wire one
Development closed loop, log the actual 10-min target vectors, and compare authoritative SWMM TFV.

## 12. Current priority

Do not overcomplicate the project. Work in this order:

1. prove direct Step2 learns exact SWMM action -> delta-TFV ordering;
2. run one causal Development 10-min closed loop with direct Step3;
3. verify multiple facilities change and authoritative SWMM TFV improves;
4. only then improve robustness, PFV soft protection, uncertainty, gradient efficiency and formal
   publication evidence.
