# Project7 Step2 V11 — canonical architecture

## Why V11 exists

Train-only forensics established that single-actuator D2 hydraulic effects are
learnable, while the prior full-network finite-hop Hydraulic models failed to
represent where and when those effects appear.  V11 therefore changes the task,
not merely the GNN depth.

The mainline Step2 is now deliberately dual-timescale:

- **V7 Value (frozen, supported):** 0–360 min direct signed `Delta-TFV` for MPC
  candidate ranking and long-horizon anti-myopia.
- **V11 Hydraulic Effect (development):** 0–120 min direct signed hydraulic
  response for control interpretation, short/medium-horizon context and
  feasibility diagnostics.

A 10-minute receding-horizon controller does not need centimetre-accurate
nodewise hydraulics six hours ahead.  It does need the near-term direction,
support, magnitude and timing of control consequences, while retaining a
long-horizon system objective.

## Target semantics

The authoritative target remains a same-prefix SWMM counterfactual:

`Delta x(t+tau) = x_candidate(t+tau) - x_reference(t+tau)`.

Lag is represented by the *time-indexed response function*, not by replacing
this target with `x(t+1)-x(t)`.  For response time `tau`, V11 constructs features
from **only the action prefix that has occurred by tau**.  A setting change
scheduled after tau is structurally forbidden from changing the response at tau.

## Lag-aware action exposure

For each changed actuator and response time, the token includes:

- reference/candidate setting at that time,
- current signed action delta,
- prefix mean and mean-absolute delta,
- prefix maximum absolute delta,
- 10/30/60-min recent action exposure,
- time since first and last action change,
- whether the actuator has changed at all by that response time.

This makes delayed effects learnable without recurrent free-run rollout.

## Variable-size actuator set

At each candidate and response time, only actuators that have changed by that
time are active keys in a Set-Transformer-style self-attention block.  Unchanged
facilities are masked.  Thus K=1 D2 and arbitrary feasible K>1 D3 candidates use
one shared model.  The model does **not** enumerate combinations and does **not**
sum predicted D2 effects.

D2 trains the local mechanism representation.  D3 trains the joint set
interaction with a fixed 0.25 authoritative-D2 anchor.

## Nonlocal response

Each node/time pair forms a query from:

- predicted causal V7 reference hydraulics,
- current reconstructed state,
- node static properties and node identity,
- causal rainfall prefix,
- response time.

The query attends to the entire changed-actuator set.  Static actuator-node
relation bias includes endpoint identity and all-range graph proximity.  There
is no finite H-hop cutoff.

## Active / sign / magnitude

A hydraulic effect is decomposed into three learned questions:

1. **Active:** is the candidate-reference effect locally meaningful?
2. **Sign:** if active, is it positive or negative?
3. **Magnitude:** if active, how large is it?

The reconstructed signed response is based on the three heads, with
candidate==reference and pre-action response times forced to exact zero.

"Active" is no longer based on one global channel scale.  TrainFit D2 freezes
node/actuator-local thresholds from `0.25 × P90(|Delta|)` with physical floors:

- depth: at least 0.01 m and 1% of local maximum depth,
- flooding: at least 1e-5 m3/s,
- volume: at least 1e-3 m3 and, for storage, 0.5% of capacity,
- node inflow/outflow: at least 1e-4 m3/s,
- managed flow: at least 1e-4 m3/s.

Holdout data never selects these thresholds.

## Time-domain supervision

V11 predicts retained responses directly:

- 5, 10, 15, 20, 25, 30 min,
- then every 10 min through 120 min.

The loss includes authoritative finite-difference response changes between
retained times.  This supervises rise/decay/timing without requiring recursive
state rollout or imposing an artificial generic smoothness prior.

## Training gates

### Stage D2
- TrainFit D2 only.
- 4 epochs, seed 42, FP32, AdamW.
- D3 is blocked until all primary InternalHoldout skills vs zero are > 0:
  depth, flooding, volume and managed flow.

### Stage D3
- Targeted D3 only after accepted D2 checkpoint/report.
- 10 epochs.
- D3/D2 authoritative losses = 0.75/0.25.
- D2 outputs are never added to create a D3 target.

No stage automatically accesses Validation, Final, Formal, Policy Lock or new
SWMM data.

## Canonical import rule

New code must import Step2 through:

`rtc.step2_current`

The active contract is:

- Value = V7.
- Hydraulic Effect = V11.

V4–V10 Hydraulic files remain historical provenance only and must not be
imported by the canonical runtime or new development code.
