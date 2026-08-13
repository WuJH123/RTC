# Project7 Step2 V11.2 — Influence-Atlas-First Hydraulic Learning

## Why this branch exists

V11.1 fixed real correctness/optimization defects in the dense signed-effect path: signed effects are no longer physically clipped, exact-zero is preserved for unchanged actions, inactive leakage gradients are bounded, and TrainFit tiny/micro diagnostics are reproducible. The remaining failure is different: local endpoint diagnostics can learn D2 depth/storage effects, while the full-network action-to-node decoder still collapses toward the zero predictor.

The next step is therefore **not** another capacity/epoch/loss sweep. D2 is first used as an intervention census to learn where and when each actuator can have a meaningful hydraulic effect.

## D2 lineage — two populations must not be conflated

Original v0.6.9 source census:

- 9,216 requested probes;
- 4,800 unique authoritative SWMM branches;
- 24 generation-development events;
- 192 checkpoint states;
- 109/109 actuator coverage;
- 360 min source horizon, 300 s stride.

The current V60/V11 training cache is a later derived view (for example 144 checkpoint groups / about 3,600 branches after the current event split). It is scientifically wrong to rename that cache "the complete D2 dataset". It is equally wrong to restore all 4,800 source branches to training merely because they exist: the **current scientific split remains authoritative**. Validation/Final outcomes stay untouched and the internal holdout remains independent.

## New D2 causal decomposition

For a single-actuator D2 probe, V11.2 records:

`Delta setting -> realized Delta facility flow -> response support -> signed response`

This separates an ineffective setting change from an actual hydraulic intervention. A changed setting with negligible realized facility-flow response is valid negative evidence, not missing/bad data.

Response support is indexed by:

- source actuator;
- current causal hydraulic phase;
- retained response time;
- node;
- hydraulic channel (depth, flooding rate, storage volume, inflow, outflow).

The stored delta remains signed and unclipped.

## State-conditioned Influence Atlas

`audit_step2_influence_atlas_v112.py` derives meaningful-effect thresholds from **TrainFit D2 only**, then forms overall and low/mid/high checkpoint-load support frequencies. Hydraulic phase uses the causal checkpoint P90 depth/max-depth ratio; tertile boundaries are learned from TrainFit only.

Support is a **soft prior**, not a physical reachability mask. Jeffreys smoothing retains a non-zero global escape for observed actuator/node pairs. Do not add a hard 4/8/16-hop cutoff: backwater, storage redistribution, diversions and surcharge can create delayed non-local responses.

## Multi-actuator rule — do not repeat V4.5

D2 may answer **where to look** for a joint candidate. It may not answer the joint signed magnitude by superposition.

Allowed:

`support_joint = soft_union(support_a1, support_a2, ...)`

Forbidden:

`DeltaY_joint = DeltaY_a1 + DeltaY_a2 + ...`

The latter is the rejected `SUM D2 + small interaction residual` route. D3 must directly supervise the authoritative joint-action signed response.

## Local execution

Use the existing cache and graph only; do not start SWMM:

```powershell
cd E:\RTC_sewer\Project7\repo
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
python -u scripts\audit_step2_influence_atlas_v112.py `
  --cache-manifest E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json `
  --graph E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz `
  --out-dir E:\RTC_sewer\Project7\study_v069\step2_v112_influence_atlas
```

A bounded `--max-fit-groups` run is smoke-only. Scientific interpretation requires the uncapped TrainFit atlas.

## What must be inspected before another Hydraulic model is trained

1. Current cache must report `DERIVED_D2_VIEW`, while preserving the frozen 4,800-branch source lineage.
2. TrainFit actuator exposure and any missing actuator must be reported; missing support cannot be silently imputed.
3. Report the fraction of setting probes that produce a meaningful source facility-flow response.
4. Report response sparsity and onset/peak timing by channel and hydraulic phase.
5. Report local versus delayed/non-local support; do not turn that observation into a hard reachability rule.
6. Verify that zero action still implies exact-zero effect.
7. Verify that the atlas never reads internal-holdout outcomes during fitting.

## Next model only after atlas review

If the atlas shows stable state-conditioned support, the next Hydraulic model should retain the V11.1 signed direct target but replace blind dense action-to-all-node decoding with:

1. physical actuator endpoint injection;
2. source-flow-effect representation;
3. soft atlas support as an additional proposal/context feature;
4. learned graph propagation with global escape;
5. direct signed magnitude on response support plus bounded inactive leakage.

Do not use the atlas as future SWMM truth online. Online state must come from the frozen causal Step1 reconstruction and current/known action sequence. D2 support is frozen training knowledge only.

## Gate order

`source lineage -> TrainFit atlas -> local representation sanity -> tiny/micro -> canonical TrainFit D2 -> freeze -> one internal-holdout read -> direct D3 joint training -> forecast shift -> runtime gradient/MPC smoke`

Frozen V7 Value remains unchanged. No new SWMM, Validation, Final, Formal, Policy Lock or production evaluation is authorized by V11.2.
