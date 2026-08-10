# Wuhan RTC v0.6.7 source-input snapshot

This directory pins the **source-only physical/rainfall inputs** for `WUHAN_RTC_METHOD_TESTBED_V067` without committing ~754 MB of duplicated full-network event INPs.

## Authoritative local bundle

Expected Windows root:

`E:/RTC_sewer/Project7/inputs`

The local bundle contains the corrected network INP plus 30 generated event INPs. Each event INP repeats almost the entire ~25.5 MB network text, so committing all 31 full INPs would add roughly 754 MB of highly duplicated text to the code repository.

GitHub therefore stores:

- the exact corrected-network SHA256;
- SHA256 and byte size for every one of the 30 full event INPs;
- the complete source-event/rainfall forcing provenance;
- priority/sensor/actuator/rainfall/method contracts;
- the deterministic `rtc-build-method-testbed-v067` generator in `src/rtc/method_testbed_v067.py`.

`SOURCE_INPUT_BUNDLE_MANIFEST.json` is provenance for the originally distributed v0.6.7 source bundle. Its legacy `contracts/events_with_splits.csv` entry records what shipped in that bundle and is **not** the active Project7 split.

The active scientific split is now exclusively:

```text
configs/project7_v069_split_contract.json
configs/project7_v069_events_with_splits.csv
```

`rtc-adopt-method-testbed-v067` verifies all physical/rainfall hashes and then overwrites the extracted bundle's local `contracts/events_with_splits.csv` with the frozen v0.6.9 **18 Train / 6 Validation / 6 Final** registry. Calibration/safety-audit roles from the source bundle are historical provenance only and must not drive new simulations.

The local run MUST verify all hashes before Step 0. If the corrected network or any of the 30 event/rainfall sources differs from the pinned physical/forcing lineage, fail closed. Do not substitute historical Project5/6/7 derived data.

Scientific scope: TFV-first methodology evaluation on an idealized simplified Wuhan SWMM; claim reduction of sewer-node overflow, not field digital-twin deployment.
