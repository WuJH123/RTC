# Wuhan RTC v0.6.7 — TFV-first methodology testbed

This document is the v0.6.7 scientific override for the fresh Project7 study.

## 1. Claim scope

The study is a **methodology test on an idealized, simplified Wuhan SWMM network**. It does not claim a field-calibrated digital twin or field-certified actuator capability. The physical outcome claim is limited to reducing **sewer-node overflow** in authoritative SWMM.

Primary objective: minimize system-wide cumulative TFV. PFV is a soft secondary/diagnostic quantity. Global Peak is report-only.

## 2. Fresh source rule

Use GitHub `main` as the only code source after v0.6.7 is merged. Do not copy historical D0/D1/D2/D3, Step1/Step2 models, shards, acceptance evidence, Policy Locks or Final outputs into the new study.

The only source assets needed to reconstruct Project7 inputs are:

- the user-frozen source network INP;
- `sensor_nodes.txt` (89 nodes; existing Project7 provenance hash must match);
- `priority_nodes.txt` (8 nodes).

Run `rtc-build-method-testbed-v067` to create the new network and all 30 event INPs from those source-only assets.

## 3. Rainfall design

The v0.6.7 library is exactly 30 uniform design storms:

- return periods: 5, 10, 20, 50, 100 years;
- durations: 60, 120, 180, 240, 300, 360 min;
- pattern: Chicago only;
- rainfall step: 5 min;
- spatial mode: one uniform design gage across all subcatchments.

Wuhan DB4201/T 641-2020 is used directly:

`i = 9.686 * (1 + 0.887 * log10(P)) / (t + 11.23)^0.658`  [mm/min]

Chicago peak-position coefficient: `r = 0.39`.

Every 5-min block is integrated analytically from the Chicago intensity curve, and the generator asserts that block depths sum to the IDF depth for the requested duration.

No historical Project5/Project6/old-Project7 rainfall file is reused.

## 4. Event clock and evaluation window

Default pre-rain warm-up: **60 min**. This is intentionally the shortest default that supplies the 13-frame `0,5,...,60 min` causal history on a 5-min model grid. It is not described as proof of full DWF equilibrium.

Before production scaling, run a small development-only onset-state sensitivity comparing 60 min with 120 min (and 180 min only if needed). Increase the fresh-study warm-up only if storm-onset hydraulics are materially warm-up dependent.

Default post-rain tail: **360 min**. This is a fixed comparison/evaluation window, not a claim that flooding has returned to zero. TFV is authoritative SWMM cumulative flooding volume through the common event endpoint.

## 5. DWF, outfalls and hydrology

- Retain the supplied DWF as an **idealized background hydraulic load**.
- Preserve all 41 source outfalls as `FREE`.
- Preserve SUBAREAS and infiltration source values; v0.6.7 does not recalibrate them.
- Do not claim that the resulting model is a field digital twin.

## 6. Pump semantics

The source network contains 57 two-point `PUMP2` depth-flow curves. v0.6.7 migrates these curve declarations to `PUMP4` without changing either endpoint. This removes the stepwise Type-2 depth lookup and uses SWMM's continuous variable-depth interpolation.

The study treats pump `SETTING` continuously on `[0,1]` under the simulation-only actuator claim. This is a modeling assumption for the methodology testbed, not evidence that every physical Wuhan pump has a VFD.

## 7. Orifice semantics

All 42 orifices remain continuously controllable on `[0,1]` and receive a 10-min full opening/closing travel time in the generated network.

The controller additionally uses a common `max_setting_delta_per_update = 0.5` on the candidate 10-min supervisory grid. Thus the optimizer cannot request an instantaneous 0-to-1 supervisory jump.

## 8. Retrofit storage directionality

The five added storages retain their supplied storage curves. Their ten connecting links are made explicitly directional:

- `RTC_IN_01..05`: main network -> storage, `FlapGate=YES`;
- `RTC_OUT_01..05`: storage -> main network, `FlapGate=YES`.

The inlet and outlet are still eligible continuous regulators, but neither can reverse through its declared direction.

## 9. Native-rule repair

The source INP contains four copied OFF rules that command `SETTING=1`, identical to their ON rule:

- `VP0600010.3`;
- `VP0600010.4`;
- `VP0600010.5`;
- `add300.1`.

Only these known migration defects are repaired: the OFF action becomes `SETTING=0`. Other native thresholds are preserved so Internal RTC remains the source engineering-rule comparator rather than a newly redesigned baseline.

## 10. Graph/static features

v0.6.6 used only invert elevation, maximum depth and node type as node-static physics. v0.6.7 expands the static vector to include:

- initial/surcharge depth and ponded area;
- storage capacity and full-depth area;
- incident conduit counts, length, roughness and primary section scale;
- contributing subcatchment count, area and impervious area;
- area-weighted width and slope;
- area-weighted Horton maximum/minimum infiltration rates.

The actuator feature vector continues to expose pump capacity, regulator geometry, coefficient and flap-gate state. These are static INP properties and do not violate online causality.

## 11. Fresh Windows bootstrap

Use `scripts/bootstrap_project7_v067.ps1`. It refuses to reuse a non-empty `inputs` or `study_v067` directory. The intended layout is:

```text
E:\RTC_sewer\Project7\repo
E:\RTC_sewer\Project7\source
E:\RTC_sewer\Project7\inputs
E:\RTC_sewer\Project7\study_v067
E:\RTC_sewer\Project7\logs
```

The bootstrap clones/pulls GitHub, installs the package, runs pytest, builds the network and 30-event library, copies the existing sensor provenance, validates rainfall design, initializes a fresh workspace, compiles formal assets and runs the readiness gate.

Do not begin large D0/D1/D2/D3 generation until this fresh Step 0 passes.
