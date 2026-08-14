# Codex start here — Project7 v0.6.9 V126 development contract

This is the **single current Project7 entrypoint**. Read this file and `configs/step2_current_contract.json` before running any command. Do not infer the active method from old PR numbers, branch names, V120/V123/V125 prompts, or historical Hydraulic V8–V113 code.

## 1. Frozen research question

Project7 is an idealized EPA-SWMM methodology testbed, not a field digital twin.

Every 10 minutes Proposed must use sparse causal sensing to reconstruct current hydraulic state, command all 109 writable pumps/orifices/weirs, and minimize authoritative whole-system cumulative sewer-node flooding volume (`TFV`). Frozen Priority8 `PFV` is one-sided **soft deterioration protection** only. PFV improvement cannot buy worse TFV. Global Peak is report-only.

No future SWMM hydraulic truth or future realized rainfall is available online.

## 2. What V125 proved

The completed V125 D4→T5 development run is diagnostic evidence, not a successful control result:

- D4 V2 authoritative execution: **390 branches**, 48 checkpoints, 14 rainfall groups. Do not call this 392 unless a new local census proves otherwise.
- D4 FIT: 269 branches / 33 groups / 10 rainfall groups.
- D4 AUDIT: 121 branches / 15 groups / 4 rainfall groups.
- TFV Value: InternalHoldout D3 rank ≈ **0.4131**.
- D4-FIT anchor-neighbourhood rank ≈ **0.5020**.
- D4-AUDIT anchor-neighbourhood rank ≈ **0.4793**.
- FIT-only TFV false-benefit margin ≈ **6018.56 m3**.
- FIT-only PFV model-error margin ≈ **175.66 m3**.
- T5 learned overrides: **0/60**.
- T5 Proposed TFV reduction vs No-control ≈ **9.274%**.
- T5 Sparse-RBC anchor-only reduction ≈ **10.167%**.
- T5 Auto-RBC reference reduction ≈ **22.022%**.
- Continuous MPC remains blocked.

Two V125 correctness bugs were discovered during the local run and are now part of the remote V126 branch:

1. anchor-advantage evidence must exclude the D4 reference row;
2. tiny float32 executable-projection drift must not destroy the exact candidate==anchor zero reference.

## 3. Critical interpretation: data count is not independent supervision

The current data populations have different scientific roles.

### D2

Authoritative source census:

- 4,800 branches / 192 groups.
- only 3,600 branches / 144 groups are frozen train-eligible;
- the other 1,200 branches / 48 groups are development-validation and **must not train**.

Therefore “use all 4,800 D2 branches for training” is leakage, not fuller training.

The canonical 80/20 TrainFit split currently gives:

- D2 TrainFit: 112 groups;
- D2 InternalHoldout: 32 groups.

Historical audit also showed D2 mainly contains single-actuator, temporally low-rank perturbations. It is useful for broad actuator sensitivity but insufficient for coordinated 10-minute control by itself.

### targeted D3

Canonical targeted D3:

- 3,600 branches / 144 groups;
- 1 reference + 24 candidates per group;
- TrainFit 112 groups;
- InternalHoldout 32 groups.

It remains broad coordinated-action supervision and generic state/rainfall generalization evidence.

### legacy D3

Local project history reports roughly 1,318 older D3 branches. They are **not automatically train-eligible**. Before any use, run the V126 data census against the actual legacy cache and verify reference semantics, causal lineage, rainfall overlap, action geometry and duplicate identities.

The first V126 training run does **not** use legacy D3. If it is later admitted, it may be used only as auxiliary broad representation support, never by silently merging it into D4 decision labels.

### D4

D4 is direct candidate-minus-causal-Sparse-RBC-anchor supervision. It is the most decision-relevant data, but V125 appended D4-FIT into the generic D3 stage. That mixed reference/action distributions and used the broad D3 scale/loss for a much smaller local advantage problem.

V126 fixes training semantics instead of generating another random D4 batch.

## 4. V126 scientific hypothesis

**The dominant current blocker is not raw branch count. It is source/target mismatch and weak local decision supervision.**

First V126 experiment keeps the V124 interaction-aware network, seed 42 and broad causal inputs fixed. It changes only the curriculum:

1. **Broad representation stage:** D2 then targeted D3 using the established V124 objective.
2. **Decision fine-tune stage:** D4-FIT only, with a local anchor-advantage scale and a loss built around:
   - physical candidate-minus-anchor TFV error;
   - pairwise magnitude;
   - pairwise ordering sign;
   - beneficial-vs-harmful sign around zero;
   - listwise ordering;
   - true-best candidate margin.
3. D4-AUDIT and InternalHoldout stay read-only.

Do not re-introduce the historical full-hydraulic V8–V113 joint-loss route. Earlier Project7 gradient audits showed trajectory/state auxiliary gradients can conflict with direct TFV/ranking.

## 5. Canonical code surface

Read in this order:

1. `configs/step2_current_contract.json`
2. `src/rtc/step2_current.py`
3. `scripts/audit_step2_v126_data_inventory.py`
4. `src/rtc/step2_curriculum_v126.py`
5. `scripts/run_step2_v126_value.py`
6. `scripts/audit_v126_anchor_equivalence.py`
7. `src/rtc/step2_policy_v125.py` — corrected runtime selector, retained until V126 Value is accepted
8. `src/rtc/controller_v125.py`
9. `scripts/run_policy_v125.py`
10. `scripts/build_step2_v125_anchor_override_evidence.py` — corrected non-reference evidence
11. `scripts/calibrate_step3_v125_anchor_override.py`

Historical V125 D4 generation/cache code remains authoritative for the already-generated D4 data, but **do not generate new D4/D5 SWMM in the first V126 run**.

## 6. Mandatory execution order

### Gate A — Git and tests

Start only from the merged V126 `main` SHA supplied in the supervising prompt. Working tree must be clean. Run full `pytest -q`, `py_compile` for V126 entrypoints and `git diff --check`.

### Gate B — data usage census

Run `scripts/audit_step2_v126_data_inventory.py` before training.

The report must distinguish:

- branch count versus independent group count;
- D2 source 4,800 versus frozen train-eligible 3,600;
- canonical targeted D3 3,600;
- actual legacy D3 count from its local cache if supplied;
- D4 FIT/AUDIT physical separation;
- TrainFit versus InternalHoldout rainfall groups.

No model training occurs in this gate.

### Gate C — anchor equivalence

Before blaming Step2 for the T5 gap, run `scripts/audit_v126_anchor_equivalence.py` on the existing Sparse-RBC anchor-only and corrected V125 no-override T5 artifacts.

If all Proposed decisions report `learned_override_admitted=false`, then elapsed decision times, all 109 target settings, authoritative TFV and Priority8 PFV must match the anchor-only run within tolerance.

If this fails: verdict is `V126_STEP3_ANCHOR_EQUIVALENCE_BLOCKED`. Do not train V126 until the run-lineage/runtime mismatch is understood.

### Gate D — V126 TFV curriculum

If Gate C passes, run `scripts/run_step2_v126_value.py` with the frozen graph, canonical D2/D3 cache, existing D4 FIT/AUDIT caches and causal rainfall store.

No seed/hidden/head/loss/epoch sweep in the first run.

Report metrics **before and after D4 fine-tuning** for:

- D4-AUDIT rank, pairwise, sign, top1, regret, MAE;
- InternalHoldout D3 rank, pairwise, sign, top1, regret, MAE.

This separates local decision identification from generic state/rainfall generalization.

### Gate E — evidence and calibration

Only if D4-AUDIT materially improves without catastrophic InternalHoldout regression:

- rebuild direct anchor-relative TFV/PFV evidence;
- calibrate from FIT only;
- inspect the new false-benefit margin;
- keep D4-AUDIT read-only.

A smaller margin is useful only if AUDIT false-benefit precision also improves; never lower the margin manually to force overrides.

### Gate F — fixed T5

Only after Gate E assets are frozen, run one fixed T5 Proposed evaluation. Compare against exactly the frozen no-control, Internal RTC, Auto-RBC, EFD, All-open, All-closed and Sparse-RBC anchor-only evidence.

The immediate scientific success question is:

> Does learned control add **positive incremental TFV value over the Sparse-RBC anchor** without material Priority8 PFV deterioration?

Beating No-control alone is not sufficient. Beating Auto-RBC is a later competitiveness target, not permission to tune on T5.

## 7. When new SWMM is justified

Do not produce more random/global candidate branches now. Existing D2/D3/D4 data must first be audited and trained with the corrected curriculum.

A new D5 is authorized only after an outcome-blind support diagnosis shows where the accepted model remains uncertain. D5 selection may use TrainFit-only model disagreement, current causal Step1 state and action geometry, but not holdout outcomes. D5 must preserve common continuation and freeze FIT/AUDIT rainfall groups before SWMM outcomes.

## 8. Continuous MPC

Still disabled. Do not run L-BFGS-B unless the frozen Project7 gate passes:

- rank >= 0.70;
- top1 >= 0.50;
- TFV gradient sign >= 0.70;
- gradient cosine >= 0.60.

These are Project7 preregistered engineering/scientific gates, not universal literature thresholds.

## 9. Forbidden

Until development gates pass, do not access or tune on:

- the 1,200 D2 development-validation branches;
- D4-AUDIT training labels;
- Validation outcomes;
- Final outcomes;
- Formal;
- Policy Lock;
- future realized rainfall online;
- future SWMM state online;
- continuous MPC;
- Global Peak as objective/gate;
- legacy D3 by silent concatenation;
- a new full-hydraulic joint-loss architecture;
- new random/global SWMM expansion.

When a gate fails, stop at that scientific layer and report the failure class rather than changing multiple variables at once.
