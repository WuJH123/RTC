# Wuhan RTC v0.6.4 — sparse-state reconstruction, differentiable action model and realtime TFV MPC

The repository implements a **model-based** urban-drainage realtime-control workflow:

```text
causal sparse observations + realised rainfall + actuator readback
                              |
                              v
              Step1 current-state reconstruction
                              |
                              v
          Step2 differentiable hydraulic/action model
                              |
                              v
             continuous receding-horizon MPC
                              |
                       first move only
                              |
                              v
                       authoritative SWMM
```

The primary optimization variable is the **time sequence of writable facility settings**. The primary objective is predicted cumulative system-wide **Total Flood Volume (TFV)**. Priority Flood Volume (PFV) and priority depth are soft/reporting quantities rather than hard admission gates. Final truth comes from SWMM cumulative node statistics.

## v0.6.4 baseline audit

The Formal comparison matrix is now:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

`auto_rrc` / `Auto-RRC` are accepted aliases and canonicalized to `auto_rbc`.

The strategies mean:

- **No-control** — native supervisory `[CONTROLS]` disabled and no Python setting writes.
- **Internal RTC** — original SWMM `[CONTROLS]` are retained; Python makes no setting writes.
- **Auto-RBC** — automatically parameterized causal rule-based control. Current actuator-adjacent node depths are normalized by design max depth; upstream filling opens a link and severe downstream filling suppresses discharge. No future rainfall/state is used and thresholds are not tuned per event.
- **EFD** — storage-aware Equal Filling Degree. Current normalized storage levels are compared; more-filled storages receive stronger outgoing settings so available storage is used more evenly. No future information is used.
- **All-open / All-closed** — extreme diagnostic policies, not the main evidence of superiority.

Auto-RBC and EFD execute on the same controls-disabled physical model and the same supervisory control cadence as Proposed. If the controller config contains `max_setting_delta_per_update`, the same per-update movement limit is applied to these rule baselines.

## What TFV/PFV comparisons mean

For each independent rainfall group, Final reporting computes Proposed minus each reference separately:

```text
delta_TFV = TFV_proposed - TFV_reference
TFV reduction % = 100 * (TFV_reference - TFV_proposed) / TFV_reference

delta_PFV = PFV_proposed - PFV_reference
PFV reduction % = 100 * (PFV_reference - PFV_proposed) / PFV_reference
```

Therefore there is no single hidden TFV/PFV reference. v0.6.4 produces pairwise comparisons against:

```text
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

The scientifically important comparisons are primarily No-control, Internal RTC, Auto-RBC and EFD. All-open/All-closed show where the physical extremes lie.

## Fresh-data execution logic

If previous RTC-derived data are not trusted, use a new root such as:

```text
E:\RTC_sewer\RTC_fresh_v064
```

Reuse only intended source assets (event INPs/external rainfall files, event registry, sensor definition, priority definition). Regenerate D0/D1/D2/D3, Step1/Step2 models, development runs, Policy Lock and Final runs.

The efficient order is:

```text
0. input / graph / actuator audit
1. high-frequency D0 timing pilot
2. small exact-SWMM D2 control-leverage pilot
3. freeze model/control/horizon clocks
4. production D0 + D1 -> Step1
5. budgeted D2 + small D3 -> Step2
6. development closed-loop Proposed + meaningful baselines
7. only if control value exists -> Policy Lock + untouched seven-strategy Final
```

Do not generate a huge Step2 dataset before SWMM itself shows that actuator changes have useful TFV leverage.

## Data roles

### D0 — reference hydraulic trajectories

Full-event controls-disabled trajectories. They provide passive hydraulic-state coverage, response timing and exact replay prefixes for counterfactual actions.

### D1 — controlled exploration trajectories

Development/train-only exploratory facility settings. They expose Step1 to controlled hydraulic states rather than only No-control states.

### Step1 — sparse history to current full state

Step1 learns:

```text
sparse depth/head history
+ observation mask
+ causal rainfall / actuator context
+ graph/static features
-> current full six-channel hydraulic state
```

It is a state observer, not the control policy itself.

### D2 — local same-state action effects

At one replayed hydraulic checkpoint, change one sampled actuator up/down while all other settings are held in the complete action vector. D2 answers questions such as:

```text
At the same state, does opening this pump more make future TFV rise or fall?
```

`rtc-design-probes-efficient` rotates a limited local-probe budget across checkpoints while keeping the online action space unchanged.

### D3 — joint multi-facility time sequences

D3 changes several facilities over several control blocks. It supplies interaction data that independent D2 probes cannot provide.

### Step2 — differentiable hydraulic/action world model

Step2 learns:

```text
current full state
+ future facility setting sequence
+ rainfall sequence
+ previous actuator flow / device physics
-> future hydraulic state trajectory
-> future actuator-flow trajectory
-> cumulative flooding consequence
```

Training includes exact SWMM cumulative node flood-volume supervision. For control usefulness, action ranking and gradient direction matter more than a cosmetically small average state RMSE.

### MPC — learn the system, optimize the action online

The project does **not** train a neural policy that directly says “set pump P1 to 0.73”. Instead:

1. Step1 estimates the current whole-network state.
2. Step2 acts as a fast differentiable virtual drainage system.
3. MPC starts from the current settings and changes the future setting sequence by gradient optimization.
4. The best-so-far sequence is retained.
5. An optimized sequence is executed only if predicted TFV beats the explicit hold-current sequence.
6. Only the first control move is written to SWMM; the problem is solved again at the next decision time.

This is how the time-varying facility control strategy is obtained.

## Early control-value screen

Before expensive Step2 training, run:

```text
rtc-control-leverage-audit
```

It uses exact SWMM branches to report whether different facility actions actually create meaningful TFV spread and whether better-than-hold actions exist.

If exact SWMM actions are almost indistinguishable, generating ten times more neural-network data is unlikely to fix the project.

## Development success criterion

The first engineering objective is not “Proposed must already beat every baseline”. It is:

```text
1. Proposed produces real time-varying facility actions rather than permanent hold;
2. decision runtime fits inside the control interval;
3. authoritative SWMM shows that those actions alter TFV in the expected direction;
4. Proposed is not the worst group-balanced TFV strategy among
   No-control / Internal RTC / Auto-RBC / EFD;
5. preferably it beats No-control and at least one dynamic rule baseline.
```

All-open and All-closed are diagnostics and should not be used to claim success simply because Proposed beats an extreme policy.

## Metric definitions

```text
TFV = sum over all hydraulic nodes of SWMM cumulative flooding-volume delta [m3]
PFV = the same cumulative volume delta summed over configured priority nodes [m3]
Global Peak = max_t sum_i max(flooding_rate_i(t), 0) [m3/s]
```

`Node.flooding` and `peak_flooding_rate` are rates, not TFV/PFV.

## Causal boundary

Online Proposed, Auto-RBC and EFD use only information available at or before the current decision. Proposed may use causal rainfall scenarios derived from observed history. None of these strategies may use future realised rainfall, future SWMM state/flooding or Final truth.

## Workstation defaults

For a 16 GB RAM / RTX 4060 8 GB workstation:

```text
SWMM generation: 16 independent processes x 1 SWMM thread
reduce workers to 12 if Windows begins paging

Step1: batch 8, grad accumulation 2, AMP
Step2: batch 2, grad accumulation 4, AMP
Step2 shard size: about 128
```

## Install and self-test

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print(importlib.metadata.version('wuhan-rtc'))"
```

Expected version after the v0.6.4 merge:

```text
0.6.4
```

Code and CI establish execution contracts, not Wuhan performance. Whether Proposed actually reduces TFV must still be demonstrated by fresh authoritative-SWMM development and untouched Final runs.
