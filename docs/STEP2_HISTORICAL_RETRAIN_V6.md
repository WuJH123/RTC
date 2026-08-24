# Project7 Step2 historical-retrain V6 (Development only)

This branch tests one bounded hypothesis using **existing Development truth only**. It does not replace
V23, does not relabel old Formal evidence, and does not authorize Validation/Final access.

## Hypothesis

Historical V4.1/V4.2 showed that the single-facility response pathway could rank actions well, while
joint-action training/generalization remained the persistent failure mode. The current Direct-TFV V5
trainer deliberately lets JOINT and CONTROL update the shared facility representation. This experiment
asks whether preserving the MAIN single-facility backbone and training only the interaction head after
MAIN improves internal-holdout D3 decision quality without sacrificing D2.

## What changes

- V5 model architecture: unchanged.
- Exact authoritative SWMM delta-TFV target: unchanged.
- Causal state/rainfall inputs: unchanged.
- 109 actuators, H360 prediction, H120 free search: unchanged.
- Deterministic existing-data split: unchanged.
- MAIN: unchanged, all model parameters trainable.
- JOINT: only `interaction_head` trainable.
- CONTROL: only `interaction_head` trainable.

## What must remain frozen

`src/rtc/project7_v23_step2_lineage.py`, the V23 V5 checkpoint, V15 rank checkpoint, V21 boundary
checkpoint, existing Policy Lock and all existing Formal evidence remain untouched.

## Required order

1. Run unit tests and current lint.
2. Reproduce/read the frozen V5 DEV report.
3. Run `scripts/run_step2_tfv_value_historical_retrain_v6.py --profile dev ...` using exactly the same
   graph, caches, causal rainfall store and causal state store as the V5 DEV run.
4. Run `scripts/compare_step2_historical_retrain_v6.py` on the V5 and candidate reports.
5. If the comparator exits 2, stop and reject this arm.
6. If it passes, retrain/recalibrate downstream Step3 on Development data before any closed-loop claim.
7. Only a newly frozen end-to-end policy may later enter a new Formal protocol.

The standalone `rank >= 0.70` value is retained as a diagnostic, not a sufficient promotion rule.
