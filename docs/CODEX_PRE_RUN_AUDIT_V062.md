# Pre-Codex scientific code audit — Wuhan RTC v0.6.2

This audit was completed before starting the expensive full Wuhan workflow. It distinguishes implementation correctness from results that require authoritative local SWMM execution.

## Final scientific contract

PASS:

- Step1 reconstructs the current full hydraulic state from causal sparse observations, realised rainfall and actuator context.
- Step2 is a differentiable setting→flow→hydraulic trajectory model.
- all 109 writable actuators remain continuously eligible;
- no Engineering36/fixed active set/runtime Top-K/artificial binary-pump mask;
- rolling MPC is TFV-first;
- eight-site PFV/depth are soft/diagnostic only;
- Global Peak is report-only;
- Final truth is SWMM only.

## Metric implementation

PASS:

```text
TFV = cumulative SWMM flooding-volume delta summed over all nodes
PFV = cumulative flooding-volume delta summed over the eight priority nodes
Global Peak = routing-step max of simultaneous network positive flooding-rate sum
```

PFV is not peak flooding rate.

## No-control / Internal-RTC

PASS:

- No-control removes executable `[CONTROLS]` and makes no Python writes.
- Internal-RTC retains native `[CONTROLS]` and makes no Python writes.
- pump curves/status, intrinsic Startup/Shutoff and regulator physical behaviour remain intact.
- All-open/All-closed are separate explicit diagnostic baselines.

## P0 issue found and fixed — D0 time origin

Previous D0 generation used `Simulation.step_advance()` and first saved inside iteration, so its first frame could be `t=Δt`, while closed-loop/D1 explicitly included `t=0`.

v0.6.2:

- saves the initial hydraulic/rainfall/readback frame before SWMM iteration;
- requires `elapsed_seconds[0] == 0`;
- requires every interval equal the frozen model step;
- invalidates old D0-derived evidence through the semantic implementation fingerprint.

## P0 issue found and fixed — counterfactual same-prefix validity

Previous D2/D3 design selected checkpoints from saved No-control trajectories but execution did not numerically prove the rerun reached the same hydraulic state before action.

v0.6.2 now compares, before any candidate write:

```text
checkpoint time
all node IDs/order
all actuator IDs/order
all six state channels at every node
all actuator current settings/readback
SWMM engine version
```

Reference metadata and compact trajectory hashes enter the branch generation key. A mismatch aborts before action.

## P0 issue found and fixed — SWMM engine lineage

Previously engine version was recorded but not fully enforced across the learned operator.

v0.6.2 requires one engine lineage through:

```text
Step1 train/validation
Step2 D2/D3 train/validation shards
Step1/Step2 locked models
Proposed development/Final runtime
Final main run and Global Peak replay
paired Final strategy comparison
```

## P0 issue found and fixed — D3 engineering feasibility

Previous D3 sampling could generate jumps larger than a frozen production `max_setting_delta_per_update`.

v0.6.2 samples sequential D3 sequences using the same rate limit when configured. All actuators remain eligible; no Top-K is introduced.

## P0 issue found and fixed — Final event forcing identity

Previously Formal Final primarily verified physical-network identity. A different rain event on the same physical network could theoretically be mislabeled with the same `event_id`.

v0.6.2 defines a scientific event hash that binds:

- event INP scientific contents/options/forcing;
- external `FILE` forcing bytes;

while intentionally ignoring:

- `[CONTROLS]` policy differences;
- execution-only `THREADS`.

Runtime INP relocation preserves relative external `FILE` inputs by converting them to absolute source paths; event identity is based on file bytes rather than path spelling.

## P0 issue found and fixed — complete Final event set

Previous compiler required all five strategies for every event present in the Final index, but did not prove every locked Final event was present.

v0.6.2 requires:

```text
Final run-index event set == locked Final event set
exact five strategies per event
registry-matched rainfall_group
registry-matched scientific event forcing
one SWMM engine lineage
```

Thus a poor-performing Final event cannot be silently omitted.

## Step1 dataset sufficiency

Implementation PASS. Formal Step1 records provide:

```text
causal sparse depth/head history
observation mask
causal node-local rainfall + actuator setting/flow context
static graph features
current full six-channel target state
rainfall group/split lineage
fixed time grid
SWMM engine lineage
```

Scientific accuracy still requires the actual Wuhan held-out acceptance run.

## Step2 dataset sufficiency

Implementation PASS. Each branch/shard provides:

```text
x_t
rainfall sequence
continuous setting sequence
previous actuator flow
device physics via locked graph
target x_(t+1)...x_(t+H)
target actuator-flow trajectory
exact cumulative SWMM node flood-volume label
event/rainfall/checkpoint/action lineage
fixed model step/horizon
SWMM engine lineage
```

Interval alignment is explicitly `u_t/rain_t : x_t → x_(t+1)`.

Scientific predictive quality still requires Step2 held-out, D2 gradient and D2+D3 ranking acceptance.

## Data persistence / resume

PASS:

- compact NPZ is primary large-data trajectory format;
- exact node flood statistics are saved separately;
- metadata records scientific/data/runtime lineage;
- generated artifacts are hashed;
- file existence alone does not authorize reuse;
- Step1/Step2 epoch state includes model/optimizer/scaler/RNG;
- unrelated documentation edits do not invalidate data because implementation hashing is semantic rather than whole-source byte hashing.

## Time causality

PASS at implementation level:

```text
observe current state
→ reconstruct current full state
→ causal rainfall forecast
→ optimize MPC
→ write first move
→ SWMM evolves
→ observe next model step/readback
```

No future realised rainfall/runoff/state/flooding/Internal/Final truth is an online policy input.

Production 1/5/10-min choices remain empirical Phase-0 decisions rather than assumptions.

## What GitHub CI does not prove

The repository is code-ready only. The following remain local scientific evidence tasks:

- Phase-0 selects defensible model/control/horizon clocks;
- Step1 meets the frozen acceptance thresholds;
- Step2 trajectory/TFV/action-effect accuracy passes;
- Proposed runs within real-time compute budget;
- Proposed actually lowers TFV versus baselines on untouched Final events;
- PFV/depth/Global Peak remain scientifically interpretable.

Do not claim control superiority until the complete untouched Final SWMM experiment supports it.

## Codex instruction

Start from `docs/LOCAL_RUNBOOK_V062.md` and a new `RTC_fresh_v062` workspace. Do not reuse v0.6.1-derived RTC data/models/evidence.
