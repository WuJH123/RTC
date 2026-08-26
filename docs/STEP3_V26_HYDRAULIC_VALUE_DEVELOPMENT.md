# Project7 Step3 V26 — hydraulic exact-return value selection

## Why V25R2 stopped too early

V25R2 corrected the supervised target from the H120 diagnostic window to exact system-wide policy
return and improved OOF AUC from 0.365 to 0.725.  However, its controller factory still treated a
Development statistic (`harmful admitted ACTION count == 0`) as permission to start SWMM.  One
borderline OOF error therefore prevented the entire five-event Benchmark5 experiment.

That is not an engineering feasibility condition.  It is a model-quality diagnostic that should be
measured, not used to suppress the very closed-loop experiment needed to judge the controller.

## V26 consensus

V26 returns Step3 to a direct value-selection problem:

1. Build the existing three-family V23 engineering-feasible candidate portfolio.
2. Represent **every** candidate with an action-conditioned hydraulic feature.
3. Predict exact system-wide cumulative TFV return relative to HOLD for every candidate.
4. Add HOLD with value exactly zero.
5. Execute the candidate with minimum predicted return when that value is negative; otherwise HOLD.
6. Execute only the first 10 minutes and repeat causally at the next decision.

There is no V15 Top-1 selection in the V26 decision path, no V21 ACTION/HOLD classifier, and no V25
one-sided UCB admission.  The old V15/V21 checkpoints are temporarily loaded only because the tested
V23 parent factory owns common Step1/Step2/controller construction; V26 `optimize()` does not consult
them for selection or admission.

## What V26 learns from Auto-RBC

Auto-RBC has been consistently stronger than V23/V24 on Operational Benchmark5.  Its useful prior is
not a magic threshold.  It is the hydraulic structure of the decision: upstream filling creates
release pressure; downstream congestion removes release headroom; actuator type changes the meaning
of a positive SETTING move; rainfall, flooding and action magnitude modify the consequence.

V26 therefore augments the V20 facility-resolved representation with explicit causal quantities for:

- upstream depth/volume filling and composite pressure;
- downstream filling, congestion and headroom;
- upstream/downstream flooding;
- actuator-type-aware release-direction delta;
- alignment between release direction and local hydraulic pressure gradient;
- rainfall loading, network stress and action magnitude.

These are model inputs, **not hard rules**.  Exact SWMM policy-return truth decides their learned
weights.

## Data policy

Valid historical exact-return truth may be reused.  A record is not excluded merely because an older
Step3 version saw it.  `build_project7_v26_exact_return_dataset.py` pools explicitly supplied
candidate-return JSONL files, retains distinct actions from the same causal query, removes only exact
duplicate state/action records, and makes one deterministic rainfall-group-disjoint
Train/Validation/Test split (default 70/15/15).

The learning target is always:

`true_policy_return_delta_tfv_m3`

H120 remains diagnostic only.  A row is rejected only for a real data-integrity problem such as a
missing candidate action/context, non-finite exact return, failed recorded readback/prefix/continuation
verification, or contradictory duplicate truth.

## Model selection and Test

The compact ridge value model uses an asinh-scaled exact-return target.  A small fixed ridge grid is
selected on Validation by the downstream V26 decision objective: the realized exact TFV of
`argmin(predicted candidate return, HOLD=0)`.  Candidate RMSE and ridge are deterministic tie-breakers.

Test metrics are written once and are never used to select ridge.  AUC, sign accuracy, harmful action
count, precision and regret remain visible diagnostics.  None of them prevents the Development
controller from running.

## What remains hard

V26 does **not** remove engineering or reproducibility constraints.  The following remain fail-closed:

- causal state/rainfall inputs;
- 82/109 supervisory mask and passive channels unchanged;
- q95 sequence support;
- 0.5 maximum setting move per update;
- target write/readback verification;
- exact checkpoint/data/asset lineage.

Those constraints determine whether an action is physically executable and whether a result is
reproducible.  They are categorically different from an offline AUC threshold.

## Development evaluation

After Train/Validation/Test reporting, V26 is allowed to run all five frozen Operational Benchmark5
Proposed events regardless of whether offline metrics are aesthetically strong.  Existing
No-control/Internal/Auto-RBC/EFD results are reused from the immutable baseline cache; only Proposed
is rerun.

Paper-facing Development judgement then uses authoritative SWMM:

- primary: system-wide TFV;
- PFV safety: `PFV_proposed <= 100 + 1.05 * PFV_no_control`;
- engineering violations/readback/support;
- event-balanced and aggregate comparison against V23 and Auto-RBC;
- Global Peak: report only.

A poor Benchmark5 result is a scientific result and should trigger another model/candidate revision,
not be hidden behind a pre-Benchmark gate.
