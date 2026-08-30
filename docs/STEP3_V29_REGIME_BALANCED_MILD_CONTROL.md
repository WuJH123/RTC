# Project7 Step3 V29 — regime-balanced mild-control Development lane

## Scientific problem

V28R1 fixed train/deployment residual feature parity but selected residual shrinkage alpha=0, so the deployed value surface reverted to frozen Q27. The remaining error is regime dependent: the mild/short RP006-D105 event executes far fewer actions and remains substantially worse than Auto-RBC, while moderate/high-load events are competitive. Return period and event duration are therefore treated as reporting strata, not policy inputs.

## Frozen study objective

The sole online objective remains cumulative system-wide Total Flood Volume (TFV). Priority8 PFV is a secondary non-inferiority claim, `PFV_proposed <= 100 + 1.05 * PFV_no_control`. Global Peak is report-only. Step1, Step2 and Q27 remain frozen. Every 10 minutes the controller re-observes, regenerates engineering-feasible H10 candidates, ranks them against HOLD=0 and executes only the first H10.

## V29 hypotheses

### H1 — group-balanced regime value

The existing exact-return bank contains repeated records within event/context leakage groups and large-return regimes can dominate pointwise regression. V29 gives every Train leakage group equal total regression weight and adds continuous interactions between Q27, hydraulic stress and action magnitude. It never accepts event ID, return period or event duration as a feature. Ridge and residual shrinkage are selected by Train leakage-group CV only. `alpha=0` is an exact Q27 fallback. Validation and Test are report-only.

### H2 — hydraulic-utility Auto-RBC shadow

The successful Auto-RBC baseline can make many small coordinated adjustments in mild events. The existing shadow must compress such a broad action to the frozen changed-facility ceiling, historically using absolute SWMM SETTING change. Because SETTING has actuator-type-specific semantics, V29 introduces a second candidate that retains the K facilities with greatest causal hydraulic release utility: release mismatch × upstream filling × downstream headroom. It still obeys the 82/109 mask, first-move radius, actuator bounds, 0.5 slew, changed-facility ceiling and the common q95 sequence support. It has no unconditional ACTION authority.

## Development evaluation

Use the existing 510-row V28 augmented exact-return dataset. Do not generate new training truth. Report Train-CV, Validation and Test metrics overall and within Train-defined hydraulic-stress quartiles. Then run Proposed V29 exactly once on the frozen five-event Operational Benchmark5 and reuse immutable baselines. Benchmark5 is Development-only and cannot be used for post-hoc retuning.

Primary closed-loop comparisons are V29 vs V28R1, V28, V27, Auto-RBC, Internal RTC, No-control and EFD. Report event-balanced mean, median, aggregate-volume reduction, PFV safety, ACTION/HOLD mix, candidate sources, hydraulic-utility shadow retention, engineering execution and runtime.

## Publication transition

After the single Development Benchmark5 run, do not modify V29 hyperparameters, candidate thresholds or support limits. Freeze the exact V29 checkpoint, runtime source SHA, asset manifest, dataset manifest, q95 support and supervisory mask. Publication evaluation must use authoritative SWMM with a predeclared event panel that is disjoint from V29 Train/Validation/Test leakage groups and Benchmark5 development events. Selection of that panel may use forcing metadata only; hydraulic outcomes cannot be used for selection. Baselines are computed once and cached. Final reporting is event-balanced and includes bootstrap confidence intervals for TFV reductions.

The publication evaluation may falsify the V29 hypothesis. The pipeline is designed to produce defensible evidence, not to force a positive result.
