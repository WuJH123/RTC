# Project7 simulation asset management — v0.6.8

This document defines how expensive local SWMM evidence is identified, reused, invalidated and audited. Large trajectories stay on the user's local disk; Git stores code, contracts, schemas, manifests and small evidence summaries only.

## 1. Why this exists

A directory name is not a scientific identity. `d2_h210`, `d2_h240_tail480` and `d2_h360_tail600` may contain the same event/checkpoint/action family with different observation horizons or recovery-tail allowances. Conversely, two files with the same friendly name may be scientifically different after a warm-up, forcing, network, action, engine or checkpoint-state change.

Project7 therefore uses **simulation identity**, not file/folder identity.

## 2. Local layout

Recommended local root:

```text
E:\RTC_sewer\Project7\data_assets_v068\
    _registry\
        simulations.sqlite3
    authoritative\
        d0\
        d2\
        d3\
    derived_views\
        step1\
        step2\
        phase0_timing\
    quarantine\
        invalid\
        stale\
        failed\
```

The SQLite registry stores paths and hashes. It does **not** copy large `.npz`, `.out`, `.rpt` or runtime INPs into Git.

## 3. D2 identity

D2 uses two keys.

### Family identity

Bound to:

```text
physical network SHA
scientific event/forcing/start-clock family SHA
exact checkpoint elapsed time
exact checkpoint six-channel hydraulic state SHA
exact checkpoint actuator readback SHA
candidate full-action SHA
native-controls-disabled semantics
SWMM engine version
record stride
```

The event family deliberately ignores only recovery-tail `END/REPORT_END` options. Therefore extending a prepared event from tail480 to tail600 does not make an otherwise identical physical checkpoint/action a new family. START/REPORT_START, rainfall, DWF, network physics and all other scientific content remain bound.

### Simulation identity

```text
family identity + requested horizon
```

Changing the horizon creates a new simulation identity while retaining the same family identity.

## 4. Warm-up identity

Never describe one number as simply `warmup_minutes` without its role.

Every prepared registry must record:

```text
source_pre_rain_prefix_minutes
additional_warmup_minutes
effective_warmup_minutes
```

New Formal preparation uses `--target-effective-warmup-minutes`. Example: if a source event already contains 60 min before rainfall and the target is 120 min, only another 60 min is added.

Changing effective warm-up changes START and the checkpoint state, so simulation identity changes automatically.

## 5. Endpoint preflight

Before launching any D2/D3 SWMM process:

```text
required_end = checkpoint_elapsed + requested_horizon
available_end = event END - event START
```

If `required_end > available_end`, the request is invalid **before execution**. Do not run hundreds of branches and discover the problem at the end.

`rtc-run-probes` writes `REQUEST_CENSUS.json` containing at least:

```text
requested_rows
unique_projected_actions
deduplicated_rows
endpoint_invalid
exact_asset_hits
covering_trajectory_hits
need_execution_before_local_resume
```

D3 performs the same endpoint preflight for the complete sequence duration.

## 6. Dedup before execution

Scientific request manifests and execution manifests are different concepts.

A D2 request may contain multiple rows that project to the same complete 109-setting action. SWMM execution is deduplicated by event/checkpoint/action identity before processes are launched. The census records both the requested and unique counts.

## 7. Exact cache reuse

Use a persistent local asset root:

```powershell
rtc-run-probes ... `
  --asset-root E:\RTC_sewer\Project7\data_assets_v068
```

An exact registry hit is reused only after the registry verifies:

```text
simulation identity
metadata SHA
all referenced compact/statistics artifact SHA values
qualification == VALID_REUSABLE
```

A changed/missing file is a cache miss, never a silent success.

## 8. Reusing existing v0.6.7 D2 evidence

Existing local exact-prefix branches can be indexed without copying the large data:

```powershell
rtc-index-existing-d2-assets `
  --asset-root E:\RTC_sewer\Project7\data_assets_v068 `
  --run-summary E:\RTC_sewer\Project7\study_v067\phase0\d2_h210\RUN_SUMMARY.csv `
  --run-summary E:\RTC_sewer\Project7\study_v067\phase0\d2_h240_tail480\RUN_SUMMARY.csv `
  --run-summary E:\RTC_sewer\Project7\study_v067\phase0\d2_h300_tail480\RUN_SUMMARY.csv `
  --run-summary E:\RTC_sewer\Project7\study_v067\phase0\d2_h360_tail600\RUN_SUMMARY.csv `
  --qualification VALID_REUSABLE `
  --out E:\RTC_sewer\Project7\logs\asset_import_v068.json
```

Do not index the first endpoint-failed h240 partial directory as reusable evidence.

`VALID_REUSABLE` means the hydraulic computation is identity/hash-valid. It does **not** mean its timing gate passed. The h210/h240/h300/h360 censor conclusions remain attached to their scientific reports.

Audit later with:

```powershell
rtc-audit-simulation-assets `
  --asset-root E:\RTC_sewer\Project7\data_assets_v068 `
  --out E:\RTC_sewer\Project7\logs\asset_audit_v068.json
```

## 9. One long timing trajectory, multiple horizon views

For response-timing analysis, a longer compact D2 trajectory may be sliced exactly at a shorter horizon:

```powershell
rtc-phase0-timescale ... `
  --run-summary <h360 RUN_SUMMARY.csv> `
  --analysis-horizon-minutes 210
```

Repeat with 240/300/360 as required. This avoids rerunning the same physical branch only to observe a shorter prefix.

This reuse is restricted to time-series/timing analysis. A long branch's final cumulative node statistics must **not** be presented as shorter-horizon TFV truth.

For new max-horizon D2 generation, intermediate cumulative node-statistics snapshots should be requested so exact shorter-horizon TFV can be derived from the same SWMM run. Existing old branches without those snapshots cannot be retroactively upgraded by integrating coarse sampled flooding rates.

## 10. Sustained step vs pulse recovery

D2 is a sustained step-action experiment. It is appropriate for control leverage and step-response timing, but it cannot by itself identify how quickly the system recovers after the action is released.

If sustained-step depth remains near-horizon while flow/flooding have largely converged, do not automatically extend to ever-longer horizons and do not weaken the 5% censor guard. Use development-only pulse/release evidence:

```powershell
rtc-design-phase0-pulses ...
rtc-run-d3-batch ...
rtc-analyse-phase0-pulses ...
```

The pulse applies a candidate for one control block, then restores the complete base action for the remaining blocks. This measures post-release decay separately from sustained D2.

## 11. Qualification states

```text
VALID_REUSABLE     computation/artifacts pass identity+hash checks and may satisfy matching requests
DIAGNOSTIC_ONLY    intentionally retained for diagnosis but not selectable as reusable Formal input
INVALID            known scientific/time/endpoint contract violation
STALE              dependency/contract changed
FAILED             execution failed
PENDING            incomplete/not yet audited
REBUILDABLE_CACHE  derivative that can be rebuilt from authoritative assets
```

Qualification is an asset-management property. Scientific gates (censoring, Step1/Step2 acceptance, Policy Lock, Final) remain separate.

## 12. What invalidates reuse

Examples that change identity or qualification:

```text
network physics changes
rainfall/DWF changes
START/effective warm-up changes
checkpoint hydraulic/readback state changes
candidate complete action changes
SWMM engine changes
record stride changes when required by the evidence contract
prefix verification fails
artifact bytes change or disappear
```

Merely moving from a 480-min to a 600-min recovery tail does not split the D2 *family* if everything through the requested endpoint is otherwise identical; exact simulation identity still includes the requested horizon.

## 13. Git boundary

Commit to Git:

```text
source code
scientific contracts
small portable registries
identity schemas
asset audits/census summaries
Policy Lock manifests
Final CSV/JSON/Markdown summaries
```

Keep local/remote object storage outside normal Git:

```text
large prepared INPs
compact trajectories
raw SWMM output/report files
training shards
model checkpoints
large caches
```

## 14. Acceleration not yet authorized for Formal production

PySWMM exposes SWMM hot-start APIs, and SWMM supports reusable interface files, but Project7 must not switch Formal D2/D3 from exact-prefix replay to hot-start/runoff reuse merely for speed. First run a separately registered equivalence audit against exact-prefix replay across multiple events/checkpoints/actions. Any feature that SWMM does not preserve in a hot start must be accounted for before such an optimization can enter the Formal execution contract.
