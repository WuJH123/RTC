# Uploaded Wuhan V8 INP audit — 2026-08-09

This audit records facts read directly from the user-supplied
`wuhan_v8_storage_retrofit(2).inp`. The large INP itself is not duplicated in Git.

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
- Eligible continuous actuators total: 109
- Cross sections: 1,219
- Curves: 144

All 57 pumps reference `PUMP2` curves. The pump curves are heterogeneous; their encoded
maximum flow ordinate spans approximately 0.2 to 6.0 CMS. Orifice/weir geometry and
coefficients are also heterogeneous, which is why Step2 V2 stores physical device features
and a locked actuator-identity embedding rather than only a device-type one-hot vector.

## Native Internal-RTC

`[CONTROLS]` contains 536 executable/non-comment lines. Parsing action lines shows native
rules can manipulate **82 of the 109 eligible actuators**:

- pumps: 57 / 57
- orifices: 16 / 42
- weirs: 9 / 10

Because `RULE_STEP = 10 s`, a Python controller that writes settings only every few minutes
must not share the same runtime with native rules. New policy isolation therefore uses:

- **Internal-RTC:** original event INP, native `[CONTROLS]` enabled, no Python setting writes;
- **No-control:** same physical network/forcing, executable `[CONTROLS]` removed, no Python
  writes;
- **Proposed/D1/D2/D3/Hold/diagnostics:** controls-disabled base from simulation start.

A direct text audit of the controls-disabled copy preserved counts for junctions, outfalls,
storage, conduits, pumps, orifices, weirs, cross sections, subcatchments, gages and time
series while reducing executable `[CONTROLS]` lines from 536 to 0.

## Units and flood metrics

With `FLOW_UNITS = CMS`:

- instantaneous `Node.flooding` is a **flow rate in m3/s**;
- it is not PFV or TFV;
- authoritative event/horizon TFV is the sum of cumulative SWMM node flooding volume over
  all nodes;
- authoritative PFV is the sum of cumulative SWMM node flooding volume over the verified
  priority-site nodes;
- branch-level D2/D3 truth subtracts the cumulative statistic at the exact checkpoint from
  the statistic at the exact aligned horizon endpoint;
- Global Peak is the maximum over routing time of the simultaneous network sum of flooding
  rates.

## Priority-site blocker

The current repository `data/priority_nodes.txt` contains:

`JYS0814052`, `PS010004`, `HMT136`, `JPQM12306`, `GZLT_CFDP`, `MW5`, `GCLC_CFDP`, `LWS054702`.

**None of these eight IDs exists in this uploaded INP (0/8).** They must not be silently
reused or replaced by guessed IDs. Formal priority/PFV evidence is blocked until the eight
observed ponding locations are mapped to nodes in this exact INP. `rtc-resolve-priority`
provides an auditable coordinate-to-nearest-node mapping when observed-point coordinates are
available and reports mapping distance/duplicates for manual review.

## Local hardware execution recommendation

For a workstation capable of 16 concurrent CPU workers plus an RTX 4060 8GB GPU:

- SWMM data generation: 16 independent Python processes, normally one SWMM engine thread per
  process (`--workers 16 --swmm-threads-per-process 1`) to avoid 32-thread oversubscription;
- GPU model training: separate phase, AMP enabled, small micro-batches plus gradient
  accumulation;
- compact/resumable evidence prevents interrupted runs from forcing full regeneration.
