# Codex start here — Project7 v0.6.9

This repository intentionally keeps physical-input contract names from v0.6.7 and the simulation-asset contract name from v0.6.8 because they remain scientifically active. Do **not** infer the active controller version from those source-contract filenames.

Before running SWMM or training, read exactly these active files:

1. `FORMAL_PIPELINE_LATEST.md` — scientific sequence and fail-closed gates.
2. `configs/project7_v069_parameter_register.json` — frozen values, pending human decisions, one-time benchmarks and diagnostic-only thresholds.
3. `configs/formal_controller_v5.template.json` — active runtime/controller template.
4. `configs/formal_baseline_plan.v6.json` — active seven-strategy comparison contract.
5. `configs/model_acceptance_contract_v3.template.json` — active model/gradient/ranking acceptance threshold template.
6. `docs/METHOD_TESTBED_V067.md` — still-active frozen physical/rainfall source contract.
7. `docs/SIMULATION_ASSET_MANAGEMENT_V068.md` — still-active large-data identity/reuse contract.

Do not search old versioned controller/baseline configs to decide current semantics. Obsolete `formal_controller_v3`, `formal_controller_v4` and `formal_baseline_plan.v2/v3` files have been removed from the active tree.

## Expensive-compute rule

- Never grid-search every `REPLACE_*` value.
- Never use Final data to choose any parameter or threshold.
- First resolve `HUMAN_DECISION_PENDING` values once.
- `ONE_TIME_BENCHMARK` means a small runtime/readback benchmark only.
- `DIAGNOSTIC_ONLY` values are reported and must not be tuned to obtain a favorable result.
- Reuse authoritative SWMM branches only through the simulation-identity/hash contract.
- When a scientific gate fails, diagnose the failed model/data mechanism before increasing sample size or sweeping hyperparameters.
- The source registry's `calibration` and `safety_audit` roles are optional under the active v0.6.9 rainfall-design code. Do not simulate those eight events merely because the legacy role labels exist; wait for the explicit split decision in the parameter register.
- The `160 rainfall groups` value is a paper-strength recommendation only, never an instruction to generate 130 additional storms for the current 30-event methodology testbed.

## Current public CLI names

Use `rtc-policy-lock-v6` and `rtc-compile-final-v5`. Historical mismatched public aliases are not part of the active v0.6.9 interface.
