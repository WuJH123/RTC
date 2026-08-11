# Step2 response-calibration audit V4.1

Status: **AUDIT COMPLETE — NO TRAINING**

## Boundary

- Existing development/train tiny and frozen 12-group cohorts only.
- SWMM launched: **NO**; D2/D3 regenerated: **NO**.
- Validation outcomes: **NOT ACCESSED**; Final: **NOT ACCESSED**.
- Formal Step2 / closed-loop: **NOT RUN**.
- Frozen V4 checkpoint SHA-256: `cf6c5a15676cdaaaff8fb932556f59ac8c77831d0f0425d28531d25c6cd1a711`.

## Root cause

The V4 action pathway is nonzero, but the response objective is miscalibrated:

- Counterfactual state and flow losses use absolute-state/flow standard deviations. The loaded Train-only `state_delta_scale` and `flow_delta_scale` do not normalize those effect losses.
- The V4 TFV target is a rectangle integral of future flooding-rate states, not authoritative cumulative SWMM node flood volume.
- D2 rate-surrogate versus authoritative mean group Spearman is `0.1423`; mean spread ratio is `4.8662`; Top-1 is `0/6`.
- D3 rate-surrogate versus authoritative mean group Spearman is `0.8651`; mean spread ratio is `0.9388`; Top-1 is `4/6`.
- D3 authoritative delta-TFV RMS is `22.52` times D2, but both sources train the same effect head.
- The ranking term is candidate-vs-reference sign classification, not within-group candidate-pair ranking.
- The largest current effect-parameter gradient is D2 `delta_tfv_rate_rectangle` on the trajectory effect head (`104.384`).
- D2 `delta_flow` and effect-energy gradients have cosine `0.9762`, so minimizing that term locally drives the already small response toward zero.
- There is no direct delta-TFV head (`0` parameters).
- Pair stacking evaluates `48` reference rows for every D2 24-candidate group and `16` for every D3 8-candidate group; a group model needs one shared reference encoding.

## Physical semantics

- `head = invert elevation + depth` holds across the fixed cohort with maximum absolute residual `1.90735e-6 m`.
- The graph contains invert elevation, maximum depth, surcharge depth and ponded area.
- Neither `depth > max_depth` nor `depth > max_depth + surcharge_depth` matched flooding occurrence in the fixed cohort. No guessed differentiable occurrence gate is authorized.
- V4.1 should enforce non-negative reference/candidate flooding rates structurally while retaining signed counterfactual delta flooding.

## Decision

Primary failure: **TARGET_AND_LOSS_MISALIGNMENT**.

Proceed only to bounded development/train V4.1 architecture and loss changes. Do not run the 12-group micro until D2 tiny, D3 tiny and combined tiny pass in order. Need new SWMM: **NO**.
