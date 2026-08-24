# Project7 publication-final protocol V1

## Research objective

Project7 targets an engineering-deployable learning-assisted real-time controller for a large urban
drainage network under sparse causal sensing.  The primary control objective is to reduce system-wide
Total Flood Volume (TFV).  Priority8 flooding is a pre-registered non-inferiority safety outcome:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

Global Peak is reported only and is not optimized or gated.

## Final scientific disposition after V6/V7

The bounded V6 and V7 Step2 follow-ups were rejected on the existing Development holdout.  V6 modestly
improved single-facility D2 metrics but degraded D3 joint-action quality.  V7 reduced D3 selected regret
but still regressed D3 rank/pairwise and increased harmful selection.  Therefore no V8 architecture
search is scientifically authorized for the publication controller.  The frozen Direct-TFV V5
checkpoint remains the publication Step2 and its failed standalone exact-TFV diagnostic must be
reported as a component limitation.

## Step 1 — sparse causal state reconstruction

Purpose: reconstruct the network hydraulic state from sparse causal sensor observations for online
control.  The publication-facing acceptance metric is unobserved-node depth NSE with the frozen
minimum 0.70.  The accepted evidence value is approximately 0.9244.  Step1 is frozen and is not
retrained during publication evaluation.

## Step 2 — causal action-conditioned TFV representation

Purpose: provide a causal, action-conditioned representation of candidate-minus-reference system-wide
TFV response for downstream ranking and HOLD decisions.  The publication controller uses Direct-TFV
V5, SHA256 `3a05704812a07a914d0ce9d8d026f6c84a4dbed646743f95d27726b29c3a544a`.

Step2 is not described as an independently accurate TFV surrogate because its standalone Formal
ranking criterion failed.  The paper must distinguish component-level surrogate accuracy from
end-to-end closed-loop controller efficacy.

## Step 3 — finite-candidate receding-horizon learning-assisted RTC

Step3 is not continuous gradient MPC.  Historical L-BFGS-B is archival and projected-gradient H10 is a
Development ablation only because prior gradient-fidelity and candidate-truth audits did not justify
production use.

Every 10 minutes the controller:

1. receives causal sparse observations;
2. reconstructs the current state with frozen Step1;
3. evaluates at most three engineering-supported H10 candidate families with the frozen Step2/V15/V21
   policy-return stack;
4. selects the best generated candidate;
5. executes ACTION only when the frozen selected-candidate/HOLD boundary predicts benefit, otherwise
   latches HOLD;
6. executes only the first H10 action, then re-observes and replans.

The candidate families remain exactly:

- `STEP2_H10_PROBE_SCALE_0.50`;
- `STEP2_H10_PROBE_SCALE_1.00`;
- `TYPE_AWARE_HYDRAULIC_PRESSURE`.

The online objective is system-wide TFV only.  PFV is not converted into an unvalidated learned online
penalty; instead the frozen Priority8 non-inferiority envelope is a hard end-to-end Validation/Final
safety requirement under authoritative SWMM.

## Engineering contract

- 109 model action channels;
- 82 native supervisory controls;
- 27 passive/reference channels;
- settings in [0,1];
- maximum single-update setting change 0.5;
- q95 action-support contract;
- target write/readback required;
- continuity guard required;
- no online SWMM candidate search.

## Formal evaluation

The publication mode is `FIXED_POLICY_NO_RETRAIN`:

Development/acceptance -> independent Validation -> Policy Lock -> complete Final comparison.

Competitive baselines are No-control, Internal native SWMM RTC, Auto-RBC and EFD.  Rainfall event is
the independent statistical unit.  Report event-balanced mean and median TFV reduction, aggregate
volume reduction, event wins, bootstrap intervals and exact sign tests.  Final outcomes never train or
tune the controller.

## Claim boundaries

The paper may claim an engineering-executable, sparse-sensing, causal, receding-horizon learning-assisted
RTC framework when supported by locked end-to-end SWMM evidence.  It must not claim continuous-gradient
MPC, a global control optimum, Step2 standalone high-accuracy TFV prediction, Global Peak optimization,
or universal superiority over every comparator.  Comparator-specific claims must follow the locked
Final event statistics.
