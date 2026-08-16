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

Primary objective: **system-wide cumulative TFV minimization**. SWMM is authoritative truth for
offline labels and final evaluation, but is not the online candidate evaluator.

Frozen clock/action contract:

- state/model step = 300 s;
- control update = 600 s;
- prediction horizon = H360 = 72 model steps;
- free-control horizon = H120 = 12 ten-minute blocks;
- writable continuous facilities = 109;
- decision variables = 12 x 109 = 1308;
- execute first 10 minutes only.

## 2. Step responsibilities

### Step1 — reconstruct the current state

Step1 estimates the current full-network hydraulic state from sparse causal observations. It does
not choose actions and does not learn future TFV by itself.

### Step2 — learn pairwise TFV control value

The current model receives:

```text
causal Step1 current state
+ causal rainfall forecast
+ previous managed flow
+ complete H360 reference sequence
+ complete H360 candidate sequence
```

and predicts:

```text
delta TFV = V(candidate) - V(reference)
```

The shared value network is decomposed into:

```text
sum(109 facility value differences) + multi-facility interaction value difference
```

This V2 pairwise representation is deliberate. Development data contain several valid reference
families:

```text
D2      base-action/single-actuator counterfactual reference
D3      HOLD reference
D4 FIT  causal Sparse-RBC anchor reference
D4 AUDIT causal Sparse-RBC anchor reference
future Step3 HOLD reference
```

The previous V1 direct model encoded the candidate-reference sequence difference but only the first
reference setting explicitly. Smoke improved D2/D3 strongly over legacy V128 but D4 remained weak.
V2 therefore encodes the **complete reference and complete candidate H360 sequences with the same
network**. Candidate==reference is exactly zero, and swapping candidate/reference negates the
prediction exactly.

Single-actuator branches supervise facility main-value differences. Multi-actuator branches train
only the interaction value difference after the main stage. The historical 1% practical-effect
threshold remains reporting-only and never zeroes continuous exact delta-TFV labels.

Full H360 hydraulic trajectory prediction remains an auxiliary diagnostic/ablation, not a
prerequisite for the TFV control objective.

### Step3 — optimize the learned value

Step3 minimizes predicted delta TFV over the bounded 12 x 109 decision tensor. It uses HOLD as the
online reference and returns HOLD when no finite predicted improvement is found. Production runtime
remains fail-closed until full Development evidence supports the V2 pairwise model.

## 3. Evidence that motivated V2

The authoritative facility audit established physical controllability:

- 109/109 facilities have exact same-prefix single-actuator evidence;
- all 18 rainfall x 109 facility cells are tested;
- 82 facilities show at least one meaningful TFV influence under the reporting threshold;
- 37 facilities have sampled beneficial evidence;
- 26 facilities have delayed beneficial evidence;
- multi-actuator candidates show additional joint leverage.

The first Direct-TFV smoke (V1) produced clear D2/D3 improvement over the legacy V128 indirect
hydraulic-world-model path:

```text
D2 holdout rank      0.056 -> 0.403
D2 pairwise          0.518 -> 0.646
D3 holdout rank     -0.160 -> 0.301
D3 pairwise          0.443 -> 0.602
D4 FIT rank         -0.411 -> 0.012
D4 AUDIT rank       -0.476 -> -0.106
```

This supports direct exact-delta-TFV learning, but the remaining bottleneck is local first-move /
reference-family robustness. Smoke was also not facility-complete: only 81/109 facilities had
single-actuator training coverage.

## 4. Current data roles

```text
D0  baseline hydraulic/flood opportunity context; not facility attribution
D1  broad state/action coverage; not direct facility credit assignment
D2  primary long-duration single-facility supervision
D3  HOLD-reference temporal single-facility + multi-facility supervision
D4 FIT  Sparse-RBC-reference first-move/local-operating-region supervision
D4 AUDIT untouched Sparse-RBC-reference Development audit; NEVER training
D5  optional later solver-sensitivity diagnostic; not a prerequisite
```

## 5. Direct Development profiles

`smoke` remains a deterministic fast signal check and is **not facility-complete**.

`dev` no longer reuses the small legacy V128 Development subset. It must consume **all admitted
existing Development groups** after the frozen rainfall split:

```text
112 D2 FIT
112 D3 FIT
32 D2 held-out
32 D3 held-out
33 D4 FIT
15 D4 AUDIT
```

No new SWMM data are generated merely to run this Development profile.

## 6. Current code surface

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
scripts/run_step2_current.py
scripts/run_step2_tfv_value_current.py
src/rtc/step2_tfv_value.py
src/rtc/step2_tfv_value_training.py
src/rtc/step3_tfv_value_mpc.py
scripts/audit_facility_tfv_influence_current.py
```

Legacy V128 files remain for ablation/history only, including:

```text
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_counterfactual_first_v128.py
src/rtc/step2_counterfactual_training_v5.py
src/rtc/step2_differentiable_v128_edge.py
```

## 7. Scientific boundaries

Until explicit production promotion:

- no Validation, Final, Formal or Policy Lock;
- no D4-AUDIT outcomes in training or target-scale fitting;
- no future realised rainfall online;
- no future SWMM state/flood truth online;
- no new SWMM data merely because architecture changed;
- no legacy A0/A1/A2/B0 rerun by default;
- no gradient audit as the primary gate;
- no runtime/seven-strategy promotion before V2 full Development evidence;
- do not interpret smoke as proof that all 109 facilities are learned.

## 8. Frozen Development assets

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

## 9. Hard-sync and cheap gates

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
python scripts/run_policy_current.py --promotion-status
python scripts/run_seven_strategies_current.py --promotion-status
```

Runtime and seven-strategy execution must remain disabled.

## 10. V2 smoke before full Development

Because the pairwise representation changed, first rerun one smoke to reject regressions cheaply:

```powershell
$Study="E:\RTC_sewer\Project7\study_v069"
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Run="$Study\direct_tfv_pairwise_v2_smoke_$Stamp"

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

Read first:

```text
rank
pairwise
sign
top1_fraction
hold_selected_fraction
delta_tfv_mae_m3
selected_regret_m3
```

for D2/D3 holdout and D4 FIT/AUDIT. Do not promote V2 merely because training loss falls.

If V2 smoke preserves the clear D2/D3 improvement and does not materially worsen D4, proceed to
`--profile dev`. If V2 materially improves D4 while preserving D2/D3, that is stronger evidence.

## 11. Full Development gate

The full Direct-TFV Development run must use all admitted existing groups. The report must state:

```text
dev_uses_all_existing_development_groups = true
complete_reference_sequence_encoded = true
candidate_reference_antisymmetry_by_construction = true
selected_group_counts = 112/112/32/32/33/15
```

Only after full Development evidence is available should one Development closed loop be wired.
The next question is then operational: can Step3 produce bounded multi-facility 10-min targets that
improve authoritative SWMM TFV relative to HOLD without online SWMM candidate search?
