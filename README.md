# Wuhan RTC

A clean-slate research codebase for **state-adaptive real-time control (RTC) of a large urban drainage system** using `wuhan_v8_storage_retrofit.inp`.

## Scientific goal

The controller must **not** depend on rainfall-event IDs, rainfall-specific control schedules, a fixed actuator subset, or a pre-enumerated finite action library. At every control update it should:

1. reconstruct the current network-wide hydraulic state from sparse observations;
2. predict future hydraulics under causal rainfall information and proposed continuous actuator settings;
3. protect the eight observed real-world priority ponding locations from material deterioration relative to the fallback operation;
4. minimize system-wide total flood volume (TFV) among safe actions;
5. execute only the first control move, verify readback, then re-observe and re-optimize.

## Core architecture

```text
Sparse sensors + realised rainfall + actuator readback
                    |
                    v
          Step 1: state estimator
                    |
                    v
       reconstructed hydraulic state
                    |
       causal rainfall forecast ensemble
                    |
                    v
 Step 2: differentiable hydraulic world model
      setting -> facility flow -> network state
                    |
                    v
 Step 3: continuous, receding-horizon MPC
   all writable actuators are eligible at runtime
                    |
       priority-site safety admission
                    |
        minimise risk-adjusted TFV
                    |
                    v
        execute first move + readback
                    |
                    +---- repeat at next update
```

## Deliberate departures from the previous Project6 design

- **No Engineering36 / fixed active actuator list.** Actuators are discovered from the INP (`PUMPS`, `ORIFICES`, `WEIRS`, `OUTLETS`).
- **No binary pump assumption.** SWMM settings are modelled as continuous controls in `[0, 1]`; no code path hard-binarises pumps.
- **No rainfall-ID policy.** Rainfall event names are metadata only, never policy inputs.
- **No direct `state + action -> PFV/TFV` primary surrogate.** PFV/TFV are derived from predicted hydraulic trajectories.
- **No fixed K/Top-K control constraint.** Runtime actionability should emerge from hydraulic state and the differentiable model, not a preselected subset.
- **The eight priority locations remain first-class safety sites** because they originate from observed ponding information, not model convenience.
- **Time horizons are configuration parameters to be identified from hydraulic-response experiments**, not immutable scientific constants.

## Priority ponding nodes

The migrated eight observed priority locations are stored in `data/priority_nodes.txt`.

## Safety / optimisation contract

The default policy is lexicographic:

1. all settings must be physically executable and remain within their continuous bounds;
2. an upper confidence bound on deterioration at the eight priority sites must remain within independently calibrated tolerances relative to the fallback operation;
3. optional risk-transfer protection prevents creation of material new flooding outside the eight priority sites;
4. among safe controls, minimise rainfall-ensemble risk-adjusted TFV (CVaR is supported conceptually by the contract);
5. if controls are practically equivalent in flood performance, prefer the smaller control movement as a tie-breaker.

The safety tolerances are **not hard-coded scientific constants**. They must be frozen from an independent calibration set before formal evaluation.

## Data strategy

The codebase is designed around four information-efficient datasets instead of one huge candidate bank:

- **D0/D1 hydraulic-state trajectories:** diverse rainfall and hydraulic regimes for Step 1 and uncontrolled/base dynamics.
- **D2 independent actuator probes:** each actuator is perturbed from the *same checkpoint* while all other settings are held fixed; this learns actionability and `setting -> flow` without sequential-pulse contamination.
- **D3 multi-actuator rollouts:** full-horizon trajectories for interaction and multi-step hydraulic propagation.
- **D4 active-learning additions:** new SWMM runs only where the current model is uncertain, near thresholds, or has poor rollout / gradient fidelity.

## Quick start

The large Wuhan INP is intentionally not duplicated in this repository. Place or reference the frozen file locally, for example:

```text
data/wuhan_v8_storage_retrofit.inp
```

Install:

```bash
python -m pip install -e .[dev]
```

Audit the INP and automatically discover all writable actuators:

```bash
rtc-audit-inp --inp data/wuhan_v8_storage_retrofit.inp --priority data/priority_nodes.txt
```

Create an independent single-actuator probe design (manifest only; no SWMM is run):

```bash
rtc-design-probes \
  --inp data/wuhan_v8_storage_retrofit.inp \
  --checkpoints checkpoint_ids.txt \
  --out outputs/probe_manifest.csv
```

Run tests:

```bash
pytest -q
```

## Status

This repository starts a new V1 scientific contract and intentionally does not import the old Project6 training artifacts as authority. Old assets may later be used only after explicit compatibility and leakage audits.
