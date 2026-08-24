# Project7 Step2 historical-retrain branch

This Draft branch is a bounded existing-data Step2 mechanism study. It does not replace the frozen V23/V5 policy or alter existing Formal evidence.

## V6 result: preservation alone is insufficient

V6 tested whether protecting the MAIN single-facility value backbone during JOINT/CONTROL would recover generalization. The result was informative but negative overall:

- D2 rank +0.021334;
- D2 pairwise +0.008219;
- D2 Top1 +0.125000;
- D2 selected regret -1203.895 m3;
- D3 rank -0.038060;
- D3 pairwise -0.012875;
- D3 selected regret +637.624 m3;
- D3 harmful selection increased from 1/32 to 3/32.

Thus `MAIN forgetting alone` is falsified as the primary blocker. The current bottleneck is joint-action cross-state representation/generalization.

## V7: single bounded follow-up

V7 changes only the joint residual while keeping the current Direct-TFV main-value pathway, exact SWMM delta-TFV targets, causal inputs, canonical split, seed 42, losses, epochs, learning rates and target scale unchanged.

Historically supported mechanisms ported from V4.1 are:

- active-aware latent pooling;
- actuator-identity-weighted signed action moment;
- elementwise second-order latent pair moment;
- signed / absolute / quadratic action moments;
- explicit candidate/reference antisymmetrization;
- exact zero interaction for HOLD and single-facility D2 actions.

JOINT and CONTROL may train only the new `historical_interaction_head`; the MAIN backbone remains protected after MAIN.

The experiment contract is `configs/step2_historical_interaction_v7_experiment.json` and the executable handoff is `docs/STEP2_HISTORICAL_INTERACTION_V7_CODEX_HANDOFF.md`.

## Scientific firewall

Until the V7 existing-data offline gate passes:

- new SWMM = 0;
- Validation access = forbidden;
- Final access = forbidden;
- old Formal outcomes may not be used for tuning;
- V23/V5/V15/V21/Policy Lock remain unchanged;
- no seed/epoch/LR/grid tuning;
- no V8 architecture search.

If V7 fails the preregistered V5-vs-V7 gate, the branch must stop with `EXISTING_DATA_STEP2_ARCHITECTURE_FOLLOWUP_NOT_JUSTIFIED`.

If V7 passes, downstream Step3 still must be retrained/recalibrated on Development-only data with a new Step2 SHA-bound lineage before any closed-loop or new Formal evaluation.
