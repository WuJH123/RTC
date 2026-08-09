# Uploaded Wuhan V8 INP audit — 2026-08-09

This audit records the frozen Wuhan V8 physical lineage and the subsequent reconciliation of the verified observed-site PFV node set. The large INP itself is not duplicated in Git.

## File lineage

- SHA-256: `2a59ca3395f549a14d5ccd22a744f8ef9d6a360864c0a2fb563b7917e38372c3`
- Size: 25,887,546 bytes
- `FLOW_UNITS = CMS`
- `FLOW_ROUTING = DYNWAVE`
- `ALLOW_PONDING = YES`
- `REPORT_STEP = 00:05:00`
- `WET_STEP = 00:05:00`
- `ROUTING_STEP = 00:00:15`
- `RULE_STEP = 00:00:10`
- `THREADS = 2` in the source INP
- Simulation period encoded in this file: 2022-08-11 00:00 to 2022-08-12 03:00

## Network census

- Junctions: 881
- Outfalls: 41
- Storage nodes: 10
- Hydraulic nodes total: 932
- Conduits: 1,167
- Subcatchments: 3,731
- Rain gages: 1
- Pumps: 57
- Orifices: 42
- Weirs: 10
- Outlets: 0
- Writable SWMM actuator links total: 109
- Cross sections: 1,219
- Curves: 144

All 57 pumps reference `PUMP2` curves. Their encoded maximum flow ordinates are heterogeneous, approximately 0.2–6.0 CMS. Orifice/weir geometry and coefficients are also heterogeneous. Step2 therefore stores physical device features and a locked actuator identity instead of treating all links of one type as interchangeable.

## Verified observed-site PFV_CORE8 reconciliation

The earlier repository copy of `data/priority_nodes.txt` contained eight stale IDs that did not belong to this exact Wuhan V8 lineage. A later recovery of the historical Project4/5 facility-leverage/PFV evidence tied to the **same physical INP SHA-256 above**, the same 932-node network and the same 109-actuator catalogue identified the frozen waterlogging-matched PFV core as:

```text
MSLBZW001
HS1316314
YS2530050
HS2529198
MH0200773
HS1330349
HS2529139
HS2529052
```

The historical definition was `project5_waterlogging_matched_hydraulic_active_pfv_core`; sentinel nodes were a separate diagnostic set and are not substituted into PFV_CORE8.

`data/priority_nodes.txt` has now been overwritten with exactly these eight IDs. New Formal preflight remains fail-closed: all eight must be present in the frozen INP/graph at execution time, so the recovered definition cannot bypass current-model membership checking.

## Native Internal-RTC versus No-supervisory-RTC

`[CONTROLS]` contains 536 executable/non-comment lines. Parsing action lines shows native rules can manipulate 82 of the 109 actuator links:

- pumps: 57 / 57
- orifices: 16 / 42
- weirs: 9 / 10

Because `RULE_STEP = 10 s`, a Python controller writing every few minutes must not compete with native supervisory rules in the same runtime.

Formal semantics are therefore:

- **Internal-RTC:** original event INP, native `[CONTROLS]` enabled, no Python writes;
- **No-control / No-supervisory-RTC:** same physical network/forcing, executable `[CONTROLS]` removed, no Python writes;
- **Proposed/D1/D2/D3/All-open/All-closed:** controls-disabled base from simulation start; Python writes only where the strategy requires them.

Removing `[CONTROLS]` does **not** mean erasing all local equipment behaviour. Pump Startup/Shutoff depth fields live in `[PUMPS]` and are preserved, together with pump curves, initial status and regulator physics. The current `rtc-inp-audit-v2` reports all nonzero intrinsic pump Startup/Shutoff settings explicitly under contract `NO_SUPERVISORY_RTC_V2`.

A controls-disabled runtime must preserve the physical-network hash and forcing while reducing executable supervisory `[CONTROLS]` to zero.

## Timing interpretation

The 15 s `ROUTING_STEP` is the Dynamic-Wave hydraulic integration scale. A Python `step_advance` callback does not replace that routing scale.

Production 5-min observation/model and 10-min supervisory-control cadences are **candidate higher-level clocks**, not values inferred from `REPORT_STEP`/`WET_STEP`. They must be frozen only after <=60 s Phase-0 response/readback experiments and runtime-latency acceptance.

If 300 s observation + 600 s control is accepted, the corrected t=0-inclusive causal history permits:

```text
0,5,10,...,60 min = 13 frames
first real MPC at 60 min
then 70,80,... min
```

## Units and flood metrics

With `FLOW_UNITS = CMS`:

- instantaneous `Node.flooding` is a flow rate in m3/s;
- it is not PFV or TFV;
- authoritative event/horizon TFV is the sum of cumulative SWMM node flooding-volume change over all hydraulic nodes;
- authoritative PFV is the same cumulative-volume change restricted to the verified eight PFV_CORE8 nodes;
- D2/D3 horizon truth subtracts cumulative statistics at the exact checkpoint from the exact aligned horizon endpoint;
- Global Peak is the maximum over routing time of the simultaneous network sum of positive flooding rates.

## Continuous SWMM command versus field hardware

SWMM permits fractional pump/orifice/weir settings. For non-Type5 pumps, the pump setting scales the pump-curve flow; this makes continuous-setting optimisation a valid SWMM experiment even though the physical pump curve is Type2.

This does not prove that every physical Wuhan pump has a VFD or that every regulator is continuously remotely actuated. A field-deployment claim additionally requires a per-facility SCADA/operability catalogue covering discrete/continuous mode, remote command availability, ramps/rates, dwell, interlocks, readback/communications latency, local fail-safe and manual override. Those properties must not be guessed from the INP.

## Compute/storage recommendation

For 16 concurrent CPU workers plus an RTX 4060 8GB GPU:

- SWMM generation: independent Python processes, normally `--workers 16 --swmm-threads-per-process 1`;
- GPU training: separate phase, AMP enabled, small micro-batches plus gradient accumulation;
- all new RTC-derived outputs belong inside the new Fresh Workspace;
- compact/resumable evidence is preferred; successful `.rpt/.out` and raw row-wise CSV remain debug-only by default.
