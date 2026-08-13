# Project7 V11 — D2/D3 mechanism semantics and V7/V11 rolling-MPC wiring

## Scope

This patch does **not** regenerate SWMM data and does **not** redesign or retrain the frozen V7 Value model. It adds two bounded capabilities around the current Step2 V11 development contract:

1. mine more control-relevant structure from the existing Train-only D2/D3 counterfactual groups;
2. provide a fail-closed development runtime path from Step1 through frozen V7 Value + V11 Hydraulic to the existing first-move rolling controller.

Validation, Final, Formal and Policy Lock remain outside this patch.

## The 7,200 authoritative branches are grouped experiments, not IID rows

The current cache contains 144 D2 groups and 144 targeted D3-v2 groups. Each group preserves a common checkpoint/prefix/reference and multiple counterfactual actions. The correct information unit is therefore the **counterfactual group**:

`shared initial hydraulic state + shared forcing/reference + different candidate actions -> different authoritative responses`.

Treating these branches as exchangeable IID rows throws away the strongest causal structure already paid for with SWMM.

### D2 role

D2 is single-actuator mechanism supervision. It is especially informative for:

- actuator identity/type sensitivity;
- setting dose-response;
- response onset/peak delay;
- upstream/downstream and remote propagation;
- hydraulic phase dependence;
- sign and magnitude of local counterfactual effects.

A D2 action may lie away from the grouped online MPC manifold. That is not a data error: it is an intervention designed to identify a local mechanism.

### D3 role

Targeted D3-v2 is joint-action/control-manifold supervision. It was generated from the frozen low-dimensional temporal/group control basis, so it is especially informative for:

- candidate discrimination on the action space the MPC will actually search;
- same-zone and cross-zone interaction;
- temporal-basis interaction;
- nonlinear interaction residuals;
- group-wise ranking/regret;
- robustness of D2 mechanism knowledge under joint actions.

D2 and D3 should therefore be complementary, not pooled as if they had identical sampling semantics.

## New TrainFit-only mechanism descriptors

`rtc.step2_data_semantics_v110` extracts one record per non-reference branch.

### Causal/action descriptors

These may be computed without future SWMM outcomes:

- changed actuator count;
- L1/L2 setting exposure;
- maximum setting change;
- first/last change time;
- active duration;
- action transition and sign-reversal counts;
- peak number of simultaneously changed actuators;
- current reference depth fill ratio;
- current near-surcharge fraction;
- current flood-active fraction;
- current storage fill / near-capacity fraction;
- current mean net inflow.

### Target-side mechanism labels

These use future authoritative SWMM response and are **offline TrainFit diagnostic/stratification labels only**:

- normalized hydraulic-effect energy;
- response onset time;
- peak-response time;
- fraction of state-effect energy occurring more than eight graph hops from the changed actuator set;
- authoritative signed Delta-TFV.

They must never be fed to online MPC as observed features.

## MPC-manifold descriptors

`rtc.step2_manifold_semantics_v110` projects each candidate-reference action difference onto the frozen V6/V7 control basis and reports:

- coefficient L2/Linf magnitude;
- active coefficient count;
- active control-group count;
- active temporal-basis count;
- setting-space projection RMSE;
- residual ratio to the frozen MPC manifold.

This makes a critical distinction explicit:

- **D2:** mechanism-rich and potentially off-manifold;
- **D3:** MPC-relevant and expected to be near-manifold.

A future weighting/sampling change should be justified by these diagnostics rather than by Validation performance. This patch does not change the frozen V11 loss weights.

## Read-only audits

Run:

```powershell
python scripts\audit_step2_data_semantics_v110.py `
  --graph <graph_schema.npz> `
  --cache-manifest <CACHE_MANIFEST.json> `
  --out-dir <audit_dir>

python scripts\audit_step2_manifold_v110.py `
  --graph <graph_schema.npz> `
  --cache-manifest <CACHE_MANIFEST.json> `
  --out <audit_dir>\STEP2_V110_MANIFOLD_AUDIT.json
```

Both audits are TrainFit-only and do not run SWMM or train a model.

## Runtime bundle

The V11 training checkpoint previously did not carry the TrainFit input normalization needed to convert the physical-SI Step1/rainfall/managed-flow runtime boundary into the normalized V7/V11 model boundary.

`compile_step2_v110_runtime_bundle.py` reconstructs exactly the same deterministic TrainFit split, normalization and local effect scales from the frozen cache, validates V7/V11 checkpoint/report lineage, requires all four D3 held-out primary hydraulic skills to be finite and positive, and writes a self-describing **development-only** bundle.

The bundle is:

- `runtime_compatible = true`;
- `production_compatible = false`.

It is not a Policy-Lock promotion artifact.

## V7/V11 MPC division of labour

The new `V7V11RollingMPC` keeps the dual-timescale scientific contract:

1. **V7 Value, 0-360 min:** primary robust signed Delta-TFV objective and anti-myopia signal;
2. **V11 Hydraulic, 0-120 min:** secondary positive local flood-deterioration discriminator only inside the V7 near-optimal TFV envelope.

Thus short-horizon V11 hydraulics cannot replace the long-horizon TFV objective.

The MPC searches the frozen low-dimensional control basis rather than unrestricted 109 x 36 raw settings.

## Receding-horizon execution

The new path reuses `TorchMPCController` and the existing authoritative closed-loop runner:

`new observation -> Step1 physical state -> causal rainfall forecast -> V7/V11 optimization -> engineering projection -> execute first 10-min move -> SWMM readback -> repeat`.

The 360-min action sequence is never written open-loop. At the next 10-min decision, the state, rainfall forecast and optimization are recomputed.

## Production routing

The existing public router now recognizes the explicit V11 flags:

- `--step2-value <V7 Value>`
- `--step2-reference <V7 Hydraulic reference>`
- `--step2-v110 <self-describing V11 runtime bundle>`

Only that explicit three-checkpoint surface enters the new Proposed path. Historical `--step2` continues to route to the legacy runner until a later deliberate Formal promotion removes compatibility.

A bounded development smoke may additionally pass `--allow-development-v110`. Without that flag, the loader requires a future explicitly promoted `production_compatible` bundle.

## Remaining scientific gates

Code wiring is not scientific acceptance. Before any Formal claim:

1. D2 held-out depth/flood/volume/managed-flow skill-vs-zero all > 0;
2. lag diagnostics: 0-30, 30-60, 60-120 min plus onset/peak timing;
3. non-local diagnostics by graph-distance bin, especially >8 hops;
4. D3 held-out primary skills all > 0 and no D2 mechanism collapse;
5. one bounded development runtime smoke verifies normalization, action gradients, first-move execution, readback and compute budget;
6. only then consider Policy-Lock/production promotion.

Authoritative SWMM remains the source of final control-performance claims.
