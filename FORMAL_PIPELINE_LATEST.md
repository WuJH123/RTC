# Formal Pipeline — latest mandatory entrypoints

`FORMAL_PIPELINE_V2.md` remains the detailed stage-by-stage scientific explanation. The following entrypoints supersede three earlier commands and are mandatory for new Formal runs.

## 1. Frozen assets

Use the parser-field-independent asset compiler:

```powershell
python -m rtc.formal_assets_v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --sensors data/sensor_nodes.txt `
  --out-dir outputs/formal_assets
```

## 2. Production controller configuration

Start from `configs/formal_controller_v3.template.json`, not the older template. After development-only tuning and Phase-0 time-scale evidence, replace every placeholder and freeze a numeric copy.

The formal main run MUST set:

```json
"exact_global_peak": false
```

This preserves the exact frozen model-step observation cadence and control-update cadence.

## 3. Main causal closed-loop run

Run `rtc-run-policy` with the locked V3 controller configuration. This run produces:

- the authoritative cumulative SWMM node statistics used for PFV/TFV;
- the exact executed decision log;
- sparse-sensing causal controller evidence;
- fixed model-step/control-step timing.

Do not use the sampled `global_peak_flood_rate_m3s` from this main run as the Formal Global Peak.

## 4. Bind an exact routing-step Global Peak replay

Immediately after each main run:

```powershell
python -m rtc.formalize_run `
  --main-metadata <run.json> `
  --strategy <strategy_id> `
  --event-id <event_id> `
  --rainfall-group <rainfall_group> `
  --out <formal_run_manifest.json>
```

`formalize_run` reruns fresh SWMM at every routing callback but **never calls Step1, Step2, the optimizer or rainfall forecast**. It only replays the already executed decision schedule. The resulting peak-replay SHA, decision-log SHA, main-run SHA and node-statistics SHA are bound into `FORMAL_CLOSED_LOOP_RUN_MANIFEST_V3`.

This separation prevents a reporting metric from changing the online observation/control cadence.

## 5. Strict Policy Lock

Use:

```powershell
python -m rtc.formal_lock_v3 `
  --ledger outputs/evidence/pipeline_ledger.json `
  --artifacts outputs/contracts/formal_policy_artifacts.json `
  --out outputs/policy_lock/policy_lock.json
```

Do not use the older permissive lock helper for Formal claims.

## 6. Untouched Final

The Final run index now contains:

```text
event_id,rainfall_group,strategy,formal_manifest_path
```

Compile only with:

```powershell
python -m rtc.formal_final_v3 `
  --policy-lock outputs/policy_lock/policy_lock.json `
  --run-index outputs/final/run_index.csv `
  --out-dir outputs/final/evidence
```

Formal V3 uses:

- PFV/TFV: cumulative SWMM node statistics from the fixed-cadence main run;
- Global Peak: routing-step replay of that exact run's frozen executed decision log;
- physical-network equality: forcing/control-independent SWMM physical contract hash;
- eight priority nodes: node-specific flooding volumes and maximum depths retained in detail evidence.

Any altered lock artifact, run evidence, decision log, physical network, split role, timing mismatch, incomplete event×strategy matrix, or unbound peak replay fails closed.
