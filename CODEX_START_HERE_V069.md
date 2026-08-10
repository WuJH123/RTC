# Codex start here — Project7 v0.6.9 execution freeze

This is the single active entrypoint for the Project7 TFV-first methodology testbed. The v0.6.7 and v0.6.8 names that remain in physical-input and simulation-asset documents are active source/identity contracts, not alternative controller versions.

Before running SWMM or training, read exactly these active files:

1. `FORMAL_PIPELINE_LATEST.md` — Step0→Step9 scientific sequence and fail-closed gates.
2. `configs/project7_v069_parameter_register.json` — fully frozen execution parameters.
3. `configs/project7_v069_split_contract.json` — preregistered 18 Train / 6 Validation / 6 Final forcing-only allocation.
4. `configs/project7_v069_events_with_splits.csv` — active portable 30-event registry.
5. `configs/formal_controller_v5.json` — resolved controller/runtime configuration; no `REPLACE_*` values remain.
6. `configs/formal_baseline_plan.v6.json` — active seven-strategy comparison contract.
7. `configs/model_acceptance_contract_v4.json` — preregistered dimensionless acceptance gates.
8. `docs/METHOD_TESTBED_V067.md` — still-active frozen physical/rainfall source contract.
9. `docs/SIMULATION_ASSET_MANAGEMENT_V068.md` — still-active large-data identity/reuse contract.

Do not search old versioned controller/split/acceptance files to decide current semantics. The active split contains only `development/train`, `development/validation`, and untouched `final`; calibration/safety-audit are no longer active roles.

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
- Step1/Step2 = first-run architecture/training defaults frozen in the parameter register; **no broad hyperparameter sweep**.
- Hard dimensionless gates: NSE/rank/sign = **0.70**, gradient cosine = **0.60**, D2/D3 top-1 = **0.50**.
- Absolute RMSE/MAE/regret metrics remain mandatory diagnostics but are not post-hoc hard thresholds.
- Runtime budget = **300 s**; target readback tolerance = **1e-6**; current-setting tolerance = **0.05**.
- Exactly one small development runtime/readback benchmark is allowed to verify those fixed values. Failure is fail-closed and requires human review; do not auto-retune.
- Auto-RBC and EFD use their fixed defaults; **never event-tune them**.
- Phase-0 pulse/recovery is conditional diagnostic only and is not an automatic batch.

## Expensive-compute rules

- Never grid-search controller, forecast, model architecture, training, D1/D2/D3 or baseline parameters unless the frozen contract is explicitly revised by the user before Final.
- Never reshuffle Train/Validation/Final after any hydraulic outcome is observed.
- Never use Final data for training, acceptance, controller tuning, baseline tuning, threshold selection or debugging decisions.
- Never run h210/h240/h300 as separate SWMM branches merely to inspect timing when an identity-equivalent h360 trajectory can be sliced; shorter-window exact TFV still requires an exact endpoint statistics snapshot.
- Before every large D2/D3 batch, write a branch-count/disk/runtime census, deduplicate actions, check endpoint executability, query the simulation-asset registry, then execute only required new branches.
- Reuse authoritative SWMM results only through the v0.6.8 simulation-identity/hash contract.
- When a gate fails, diagnose the specific data/model/runtime mechanism before increasing data volume. Do not lower the gate.

## Current public CLI names

Use `rtc-policy-lock-v6` and `rtc-compile-final-v5`. Historical mismatched aliases are not part of the active interface.
