# Wuhan RTC — large-system TFV-first framework

Research code for **sparse-sensing, differentiable-surrogate, state-adaptive real-time control of a large SWMM drainage network**.

The controller does **not** use rainfall-event IDs, rainfall-specific lookup schedules, a fixed active actuator subset, a pre-enumerated finite action library, or hard-binary pump assumptions. At each control update it should infer the current hydraulic state and decide online **which facilities are worth controlling, by how much, and when**.

## Current scientific goal

```text
Sparse hydraulic sensors + realised causal rainfall + actuator readback
                           |
                           v
                 Step 1: state reconstruction
                           |
                           v
                 full current hydraulic state
                           |
                 causal rainfall scenarios
                           |
                           v
       Step 2: differentiable hydraulic world model
       setting -> facility flow -> network trajectory
                           |
                           v
          Step 3: continuous receding-horizon MPC
          every writable actuator remains eligible
                           |
               PRIMARY: minimise cumulative TFV
                           |
          SECONDARY: reduce priority-site deterioration
            within a TFV-near-optimal solution set
                           |
                           v
              execute first block + readback
                           |
                           +---- observe and repeat
```

### TFV/PFV contract

- **TFV is the primary control objective.** It is cumulative flooding volume over the
  defined prediction horizon/event, summed over all nodes.
- **PFV is not a hard admission constraint.** The eight observed ponding locations remain
  important site-wise diagnostics and a soft lexicographic secondary preference. Some PFV
  deterioration may be accepted when necessary for a meaningful system-wide TFV reduction.
- `Node.flooding` is an instantaneous flow rate. It is never labelled PFV/TFV by itself.
- Formal PFV/TFV truth comes from cumulative SWMM node statistics over the exact
  event/horizon; sampled-rate integration is surrogate/diagnostic only.
- Global Peak is reported separately and is not a hard MPC condition.

## Policy semantics: do not mix these

- **Internal-RTC**: original frozen event INP, native `[CONTROLS]` enabled, **no Python
  setting writes**.
- **No-control**: same physical network and forcing, native `[CONTROLS]` disabled, **no
  Python setting writes**. It is not all-open.
- **Proposed / Hold / D1 / D2 / D3 / All-open / All-closed**: run from simulation start on
  the controls-disabled physical base.

This isolation is essential for the uploaded Wuhan V8 model because native rules execute on
an internal rule step that is much shorter than a normal RTC update; allowing native rules
to coexist with Python branch actions would contaminate actuator-effect labels.

## Uploaded Wuhan V8 compatibility audit

The user-supplied `wuhan_v8_storage_retrofit(2).inp` was audited directly on 2026-08-09.
See [`docs/WUHAN_V8_UPLOADED_INP_AUDIT_2026-08-09.md`](docs/WUHAN_V8_UPLOADED_INP_AUDIT_2026-08-09.md).

Key facts from that exact file:

- `FLOW_UNITS = CMS`, `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes, 1,167 conduits, 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = **109 eligible continuous actuators**;
- 82/109 actuators are affected by native `[CONTROLS]`, including all 57 pumps;
- source `ROUTING_STEP = 15 s`, `RULE_STEP = 10 s`, `REPORT/WET_STEP = 5 min`;
- the eight node IDs currently stored in `data/priority_nodes.txt` are **0/8 present in
  this uploaded INP**.

Therefore Formal priority/PFV evidence is blocked until the eight observed ponding sites are
mapped to valid nodes in this exact model. The code fails fast instead of guessing IDs.

## Data programme

- **D0** — compact No-control / Internal-RTC full-event trajectories for hydraulic-state
  coverage and baseline dynamics.
- **D1** — development-only continuous exploration on the controls-disabled network to
  expose Step1 to controlled states; every actuator remains eligible.
- **D2** — same-checkpoint `u-epsilon / u / u+epsilon` actuator counterfactuals. Every
  branch starts from the same prefix and native rules are disabled.
- **D3** — multi-actuator continuous sequences for interactions and long-horizon rollout.
- **D4** — active additions only where held-out rollout/gradient/ranking evidence shows a
  coverage gap.

### Compact storage contract

Formal new data is stored as compressed SI arrays plus exact node statistics and JSON
lineage. Raw node/actuator CSV and SWMM `.out/.rpt` are debug-only and disabled by default.
Per-subcatchment realised runoff is not persisted as a formal model input.

Compact state channels are:

`depth_m, head_m, flooding_m3s, volume_m3, total_inflow_m3s, total_outflow_m3s`.

Step1 windows are sliced lazily from compact trajectories instead of materialized to disk.
Step2 branches are compiled into bounded-size shards. This permits changing causal history
or batch configuration without rerunning authoritative SWMM.

## Hardware path: 16 CPU workers + RTX 4060 8GB

For SWMM generation use independent processes, normally:

```text
--workers 16 --swmm-threads-per-process 1
```

The source INP has `THREADS=2`; using 16 SWMM processes without overriding that would
oversubscribe the CPU. The data runners create a policy-isolated runtime INP with one engine
thread per process and support content-complete resume.

For GPU training use the `*-large` commands. Defaults use AMP, small micro-batches and
gradient accumulation. Step1 uses lazy trajectory-local batches; Step2 streams shards so
an 8GB GPU does not require loading the complete Wuhan dataset at once.

## Recommended entry point

Read [`FORMAL_PIPELINE_LATEST.md`](FORMAL_PIPELINE_LATEST.md). It is the only supported
new-Formal workflow. In particular, do not revive the old materialized-window/monolithic
Step2 commands for this large model.

Typical beginning:

```powershell
rtc-inp-audit-v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --out outputs/preflight/inp_audit.json
```

If the priority audit fails, resolve the observed points before Formal claims. Then proceed
with rainfall-group splitting, compact D0/D1, stratified checkpoints, compact D2, Phase-0
time-scale analysis, frozen graph/assets, lazy Step1, D3 + Step2 shards, held-out
acceptance/gradient/ranking, development closed-loop, Policy Lock V5 and untouched Final V4.

## Evidence boundary

Only authoritative SWMM runs can support final PFV/TFV/Global-Peak claims. Surrogate outputs
are used for state reconstruction, prediction, gradient search and online decision-making;
they do not replace Final SWMM truth. Final rainfall groups are never used for fitting,
priority mapping, sensor selection, time-scale selection, uncertainty calibration or
hyperparameter tuning.
