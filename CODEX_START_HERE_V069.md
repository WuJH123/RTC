# Codex start here — Project7 v0.6.9 execution freeze

This is the single active entrypoint for the Project7 TFV-first methodology testbed. The v0.6.7 and v0.6.8 names that remain in physical-input and simulation-asset documents are active source/identity contracts, not alternative controller versions.

Before running SWMM or training, read exactly these active files:

1. `FORMAL_PIPELINE_LATEST.md` — Step0→Step9 scientific sequence and fail-closed gates.
2. `configs/project7_v069_parameter_register.json` — original execution freeze and all unchanged scientific/runtime choices.
3. `configs/step2_counterfactual_training_amendment_v1.json` — **active pre-Formal Step2 amendment** created after the Train-only mechanism diagnostic exposed `ACTION_LOSS_PATHWAY`; it supersedes only the original Step2 training pathway/defaults and does not change data generation, splits, controller objective or acceptance gates.
4. `configs/project7_v069_split_contract.json` — preregistered 18 Train / 6 Validation / 6 Final forcing-only allocation.
5. `configs/project7_v069_events_with_splits.csv` — active portable 30-event registry.
6. `configs/formal_controller_v5.json` — resolved controller/runtime configuration; no `REPLACE_*` values remain.
7. `configs/formal_baseline_plan.v6.json` — active seven-strategy comparison contract.
8. `configs/model_acceptance_contract_v4.json` — preregistered dimensionless acceptance gates; unchanged by the Step2 amendment.
9. `docs/METHOD_TESTBED_V067.md` — still-active frozen physical/rainfall source contract.
10. `docs/SIMULATION_ASSET_MANAGEMENT_V068.md` — still-active large-data identity/reuse contract.

Do not search old versioned controller/split/acceptance files to decide current semantics. The active split contains only `development/train`, `development/validation`, and untouched `final`; calibration/safety-audit are no longer active roles.

## Step0 event-clock lineage

The authoritative Step0 bootstrap is `scripts/adopt_and_step0_project7_v067.ps1`. It must:

1. verify the extracted v0.6.7 physical/rainfall source bundle and the frozen 18/6/6 source registry;
2. deterministically prepare all 30 event INPs to **effective pre-rain warm-up = 120 min** with **post-rain tail = 360 min**;
3. validate the prepared 18/6/6 registry;
4. initialize `study_v069` against that **prepared effective-120 registry**, not against the original 60-min source registry;
5. write canonical preflight/readiness evidence at the paths later consumed by Policy Lock.

This ordering is mandatory. Initializing the fresh workspace against the original 60-min source registry and later replacing it with the prepared registry would create a registry-SHA mismatch at Policy Lock.

## Frozen choices — do not ask Codex to rediscover them

- 30 events = **18 Train + 6 Validation + 6 Final**.
- D1 first action = **elapsed 60 min**.
- Model/record step = **300 s**; control update = **600 s**; history = **13 frames**.
- First Proposed control = **elapsed 60 min**; effective warm-up = **120 min**.
- MPC prediction horizon = **360 min / 72 model steps / 36 control blocks**.
- Max setting change = **0.5 per 10-min update**.
- MPC controller/forecast parameters = the concrete values in `configs/formal_controller_v5.json`; **no sweep**.
- Phase-0 = **6 development/train events × 4 checkpoints/event × 12 actuators/checkpoint, epsilon=0.15, one authoritative h360 run per unique action**.
- D1 = control start 60 min, perturbation std 0.12, change probability 0.35, exploration max delta 0.20.
- D3 = 8 sequences/checkpoint, std 0.20, change probability 0.25; **no sweep**.
- Step1 remains frozen by the original parameter register. Step2 uses the **single fixed diagnostic-driven amendment** in `configs/step2_counterfactual_training_amendment_v1.json`; do not broad-sweep either model.
- The Step2 amendment explicitly trains same-prefix `Δflow`, `Δstate`, `ΔTFV` and same-checkpoint ranking, uses a H1→H2→H6→H12→H24→H72 curriculum, adds direct node-local setting context, and retains authoritative exact-H360 flood supervision.
- Hard dimensionless gates remain unchanged: NSE/rank/sign = **0.70**, gradient cosine = **0.60**, D2/D3 top-1 = **0.50**.
- Absolute RMSE/MAE/regret metrics remain mandatory diagnostics but are not post-hoc hard thresholds.
- Runtime budget = **300 s**; target readback tolerance = **1e-6**; current-setting tolerance = **0.05**.
- Exactly one small development runtime/readback benchmark is allowed to verify those fixed values. Failure is fail-closed and requires human review; do not auto-retune.
- Auto-RBC and EFD use their fixed defaults; **never event-tune them**.
- Phase-0 pulse/recovery is conditional diagnostic only and is not an automatic batch.

## Cohort scope — mandatory

- Phase-0 high-frequency D0/D2 uses only the frozen **6 selected Train events**.
- Step3 production D0 at the frozen 300-s grid must cover **all 24 development events = 18 Train + 6 Validation**, because Step1 held-out acceptance requires real Validation trajectories.
- D1 controlled exploration is **18 Train only**; no D1 on Validation or Final.
- Step4 production D2/D3 must cover **Train and Validation development cohorts** sufficiently for Step2 training and held-out action-effect/ranking evidence; no Final rows are allowed pre-lock.
- Step1 and Step2 fit only on **18 Train** and accept only on **6 Validation**.
- Step6 development closed-loop comparison should use the **6 Validation** events as the held-out development comparison set; do not tune from those outcomes.
- Step9 uses only the **6 untouched Final** events after Policy Lock.

## Step2 post-diagnostic execution rule — mandatory

The original absolute-trajectory Step2 training path is no longer active because the development-only `STEP2_MECHANISM_DIAG_V1` found that same-state action sensitivity/ranking remained weak even at H60. This is a mechanism-driven amendment made **before Formal Step2 fitting**; it is not permission to tune Validation.

After D3 completes normally:

1. leave all existing D2/D3 SWMM assets untouched; **do not regenerate D2 merely because the trainer changed**;
2. build the final D2+D3 Step2 run index with `source_kind` and `base_action_sha256` provenance where available; reject Final;
3. compile fresh **V6 counterfactual-group-preserving** Train18 and Validation6 Step2 shards so a same checkpoint group is never split across shard boundaries;
4. run one bounded **Train-only** action-sensitivity smoke using the fixed amendment; this may detect non-finite gradients/action collapse but may not tune against Validation;
5. if the Train-only smoke is healthy, fit exactly one Formal Step2 model on Train18 with the amendment;
6. run the unchanged Validation6 Step2 exact-truth, gradient and D2/D3 ranking gates;
7. if any held-out gate fails, stop and diagnose the failed mechanism. Do not lower a threshold, reshuffle events or inspect Final.

The Formal public CLI `rtc-train-step2-large` is routed to the counterfactual action-sensitive trainer after this amendment.

## CLI scope — do not propagate the Phase-0 budget into production D2

The small `4 checkpoints/event × 12 actuators/checkpoint` budget is **Phase-0 only**.

Use these guarded commands only for Phase-0:

```text
rtc-design-phase0-events
rtc-design-phase0-checkpoints
rtc-design-phase0-probes
```

For Step4 production D2 generation, use the general commands:

```text
rtc-design-checkpoints
rtc-design-probes-efficient
```

The general Step4 commands are deliberately not forced to the Phase-0 4/12 budget. Their production design must provide sufficient Train/Validation coverage of the 109-actuator action space, use the current normal production defaults once unless the frozen workflow explicitly specifies otherwise, and must not be hyperparameter-swept.

D1 and D3 values remain globally frozen by their guarded public CLIs because the user explicitly froze those settings for the study.

## Expensive-compute rules

- Never grid-search controller, forecast, model architecture, training, D1/D2/D3 or baseline parameters unless the frozen contract is explicitly revised by the user before Final.
- The single Step2 amendment above is already that explicit pre-Formal revision; do not create a second automatic amendment from Validation outcomes.
- Never reshuffle Train/Validation/Final after any hydraulic outcome is observed.
- Never use Final data for training, acceptance, controller tuning, baseline tuning, threshold selection or debugging decisions.
- Never run h210/h240/h300 as separate SWMM branches merely to inspect timing when an identity-equivalent h360 trajectory can be sliced; shorter-window exact TFV still requires an exact endpoint statistics snapshot.
- Before every large D2/D3 batch, write a branch-count/disk/runtime census, deduplicate actions, check endpoint executability, query the simulation-asset registry, then execute only required new branches.
- Reuse authoritative SWMM results only through the v0.6.8 simulation-identity/hash contract.
- When a gate fails, diagnose the specific data/model/runtime mechanism before increasing data volume. Do not lower the gate.

## Current public CLI names

Use `rtc-policy-lock-v6` and `rtc-compile-final-v5`. Historical mismatched aliases are not part of the active interface.
