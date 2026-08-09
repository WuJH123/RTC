# Wuhan RTC v0.6.3 — sparse-state reconstruction, differentiable hydraulic action model and realtime TFV MPC

This repository implements a model-based urban-drainage realtime-control pipeline:

```text
causal sparse observations + realised rainfall + actuator readback
                              |
                              v
              Step1 current full-state reconstruction
                              |
                              v
          Step2 differentiable hydraulic/action world model
                              |
                              v
             continuous receding-horizon TFV MPC
                              |
                        first move only
                              |
                              v
                       authoritative SWMM
```

The executable code is designed to minimize predicted cumulative system-wide **Total Flood Volume (TFV)**. Priority-site flood volume/depth is a soft secondary diagnostic rather than a hard admission constraint. Final hydraulic truth is SWMM.

## What v0.6.3 changes

v0.6.3 is an independent engineering audit focused on avoiding expensive data generation before controllability has been demonstrated.

It fixes a production-breaking Step2 checkpoint mismatch: the current trainer writes `STEP2_FIXED_DISCRETE_TIME_ENGINE_V2`, and the production loader now consumes that exact contract and preserves the SWMM-engine lineage.

It also adds three lean engineering improvements:

1. **Exact-SWMM control-leverage pilot** — determine whether actuator changes measurably alter future TFV before large Step2 training.
2. **Budgeted rotating D2 probes** — reduce expensive local SWMM branches while keeping every discovered actuator in the online MPC action space.
3. **Robust production MPC selection** — retain the best optimization iterate and execute an optimized action only when predicted TFV beats the explicit hold/fallback sequence.

The control-leverage report is diagnostic only; it is intentionally not another Policy-Lock gate.

## Start here for a completely fresh study

If previous RTC-derived data cannot be trusted, regenerate every hydraulic trajectory and model from the current code.

Use:

- `docs/LEAN_FRESH_RUN_V063.md` — recommended staged fresh-data workflow and workstation settings.
- `docs/INDEPENDENT_CODE_AUDIT_V063.md` — what the current code can actually do and the failure modes found by the independent audit.
- `FORMAL_PIPELINE_LATEST.md` — publication/Policy-Lock evidence machinery after development control value has been demonstrated.

The recommended order is deliberately progressive:

```text
fresh input/graph audit
        |
high-frequency D0 timing pilot
        |
small exact-SWMM D2 control-leverage pilot
        |
freeze production time scales
        |
production D0 + D1 -> Step1
        |
budgeted D2 + small D3 -> Step2
        |
development closed-loop SWMM
        |
only if useful -> full baselines / Policy Lock / untouched Final
```

Do not start by generating the full D2/D3 or Final matrix.

## Fresh-data rule

A new v0.6.3 study may reuse source assets only when they are still the intended inputs:

```text
event/source INPs and their external rainfall files
event/rainfall-group registry
sensor-node definitions
priority-node definitions
```

Regenerate from the current code:

```text
D0 trajectories
D1 exploration trajectories
D2 counterfactual branches
D3 joint sequences
Step1 indexes/model
Step2 indexes/shards/model
development closed-loop runs
runtime/acceptance evidence
Policy Lock
Final strategy runs
```

Use a new root such as `E:\RTC_sewer\RTC_fresh_v063`.

## Data meaning

### D0

Controls-disabled/reference full-event hydraulic trajectories. Used for current-state coverage, timing analysis and replayable action-effect checkpoints.

### D1

Development/train controlled exploration trajectories. Used to expose Step1 to hydraulically reachable controlled states rather than only passive states.

### D2

Same-checkpoint local setting perturbations. Used to learn and validate local action effects and TFV gradient direction.

The efficient public design command is:

```text
rtc-design-probes-efficient
```

It samples only a budget of perturbed actuator IDs per checkpoint, rotates that budget across checkpoints, and still writes the **complete all-actuator setting vector** for every SWMM candidate. This is data-efficiency sampling, not runtime Top-K control.

### D3

Joint multi-actuator, multi-control-block sequences. Used to learn interactions and validate action ranking/regret beyond independent local perturbations.

## Early controllability diagnosis

Before full Step2 generation, run:

```text
rtc-control-leverage-audit
```

It uses authoritative SWMM cumulative node flood-volume deltas to compare the hold/reference action against D2/D3 alternatives and reports:

```text
best exact TFV improvement
action TFV spread
fraction of checkpoints with meaningful improvement
sampled actuator effect coverage
```

If SWMM itself shows essentially no action-dependent TFV variation at representative flooded states, generating a much larger neural-network dataset is unlikely to solve the problem.

## Realtime feasibility logic

The current mathematical action set is not expected to become empty: hold-current is always a valid projected sequence when current settings are valid. The important questions are instead:

```text
1. Does the physical SWMM system have useful control leverage?
2. Can Step1 reconstruct enough current state for control?
3. Can Step2 learn action ranking/gradient, not just average state trajectory?
4. Can MPC find a better-than-hold action before the control deadline?
5. Does authoritative development SWMM confirm the predicted improvement?
```

v0.6.3 production MPC explicitly refuses a finite but predicted-worse-than-hold optimized candidate.

## Metric definitions

```text
TFV = sum over all hydraulic nodes of SWMM cumulative flooding-volume delta [m3]
PFV = the same volume delta summed over the configured priority nodes [m3]
Global Peak = maximum synchronous network sum of positive instantaneous flooding rates [m3/s]
```

`Node.flooding` and node `peak_flooding_rate` are rates, not TFV/PFV.

## Causal boundary

Online Proposed control may use only information available at or before decision time:

```text
sparse depth/head observations
realised rainfall history
actuator target/current-setting/flow readback
static graph/device features
causal rainfall scenarios derived from observed history
```

It must not use event ID as a policy feature, future realised rainfall/runoff, future SWMM hydraulic states/flooding, future Internal-RTC trajectories or Final truth.

## Hardware settings

Target workstation:

```text
RAM: 16 GB
GPU: RTX 4060 8 GB
SWMM: up to 16 independent simulations
```

Recommended starting settings:

```text
SWMM generation: --workers 16 --swmm-threads-per-process 1
Step1: batch-size 8, grad-accum 2, AMP
Step2: batch-size 2, grad-accum 4, AMP
Step2 shards: about 128 branches/shard
```

If system RAM begins paging, reduce SWMM workers from 16 to 12. If Step2 has GPU memory headroom, test batch 4 / grad-accum 2; if it OOMs, use batch 1 / grad-accum 8.

## Install and self-test

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print(importlib.metadata.version('wuhan-rtc'))"
```

Expected package version after the v0.6.3 audit merge:

```text
0.6.3
```

Code/CI cannot establish that the controller reduces flooding in the Wuhan model. That conclusion requires the staged fresh authoritative-SWMM experiment described in `docs/LEAN_FRESH_RUN_V063.md`.
