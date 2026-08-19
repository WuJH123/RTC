# V12 admission lineage refresh

Use this recovery path only when the frozen V12 continuation factory rejects a first-move admission because its behavioral fingerprint differs from current source.

## Scientific rule

Do **not** edit the old admission JSON, whitelist a stale behavioral SHA, substitute the full-byte provenance SHA, or weaken the runtime validator. A behavioral mismatch without component-level proof remains incompatible.

A source-lineage refresh is **not** a reason to synthesize new rainfall or regenerate generic D3. Reuse the same dedicated V12 calibration rainfall groups when they remain calibration-only and their branch truth has not been used to tune the current implementation.

## Required recovery

1. Run `scripts/audit_direct_tfv_v12_admission_lineage_current.py` on the historical V12 admission, frozen Step2 checkpoint and q95 sequence-support artifact.
2. If the report says `RECALIBRATE_CURRENT_MAIN_WITH_EXISTING_V12_CALIBRATION_GROUPS`, locate the exact historical V12 calibration rainfall groups and verify their scientific role remains calibration-only.
3. Rebuild candidate-free D0 first-move contexts for those same groups only when the existing context is not provably compatible with current input/state lineage.
4. Regenerate the **current-main** V12 scenario-mean panel. It must contain exactly one previous-target HOLD and one refined/pruned target-latch candidate per rainfall group. No generic D3 candidate rows are permitted.
5. Run authoritative SWMM for 24 HOLD + 24 candidate branches (48 total for the minimum 24 groups). Historical SWMM truth is not automatically reusable; reuse requires exact prefix, action-sequence, INP and branch-identity proof.
6. Recalibrate V12 first-move admission with `scripts/calibrate_direct_tfv_robust_rainfall_first_move_admission_current.py`.
7. Re-run the lineage audit with `--require-compatible`. It must pass before V12 is used as the frozen continuation policy for policy-return labels.
8. Only then repeat one policy-return CANDIDATE/HOLD single-pair smoke. Bulk policy-return data generation remains closed until same-prefix and same-continuation verification pass.

## Frozen boundaries

- Step1 and base Step2 V5 stay frozen.
- q95 support stays canonical.
- V12 rainfall scenarios stay `(0.8, 1.0, 1.2)`, history 3, decay 0.92, mean predicted delta TFV.
- Online objective remains system-wide cumulative TFV only.
- HOLD remains previous supervisory target latch.
- No future realized rainfall is available online.
- Validation, Final, Formal and Policy Lock remain closed.
