# Real PySWMM smoke on uploaded Wuhan V8 INP — 2026-08-09

This is an execution-interface smoke test, **not** a model-performance claim.

The current PR branch was installed with the `swmm` extra and executed against the exact
user-uploaded `wuhan_v8_storage_retrofit(2).inp` physical model.

## Procedure

1. Build a runtime copy with native `[CONTROLS]` disabled and `THREADS=1`.
2. Replay from simulation start to a 5-minute checkpoint with no Python setting writes.
3. At the checkpoint, snapshot cumulative node statistics.
4. Command all 109 discovered continuous actuators to a valid continuous smoke setting.
5. Run one 5-minute post-action interval.
6. Snapshot exact endpoint cumulative node statistics before termination.
7. Save compact evidence, compile it through the same `compile_branch_tensors` path used by
   Step2, and verify engine debug files are removed by default.

## Verified outputs

The real branch compiled successfully to:

- current state: `932 nodes x 6 SI state channels`;
- continuous action: `109 actuator settings`;
- next-state truth: `932 nodes x 6 SI state channels`;
- causal rainfall forcing: `932 nodes x 1 rainfall channel`;
- exact cumulative flooding-volume truth: one value for each of the 932 nodes;
- finite hydraulic arrays;
- controls-disabled metadata contract;
- `.rpt/.out` removed after successful completion.

The 6 state channels are depth, head, instantaneous flooding rate, node volume, total
inflow and total outflow. The exact cumulative flooding-volume statistics are stored
separately and are the authoritative branch KPI truth.

## What this smoke does and does not establish

It establishes that the uploaded large INP is executable through the revised D2 compact
data path and that the generated branch can be consumed directly by Step2 data compilation.
It does **not** establish Step1 accuracy, Step2 free-rollout accuracy, gradient fidelity,
closed-loop benefit, Policy Lock readiness or Final performance. Those remain fail-closed
until the full development/validation/calibration/final workflow is executed locally.
