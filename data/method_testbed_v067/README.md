# Wuhan RTC v0.6.7 source-input snapshot

This directory pins the **source-only scientific inputs** for `WUHAN_RTC_METHOD_TESTBED_V067` without committing ~754 MB of duplicated full-network event INPs.

## Authoritative local bundle

Expected Windows root:

`E:/RTC_sewer/Project7/inputs`

The local bundle contains the corrected network INP plus 30 generated event INPs. Each event INP repeats almost the entire ~25.5 MB network text, so committing all 31 full INPs would add roughly 754 MB of highly duplicated text to the code repository.

GitHub therefore stores:

- the exact corrected-network SHA256;
- SHA256 and byte size for every one of the 30 full event INPs;
- portable event registries;
- the complete 30-event rainfall forcing library;
- priority/sensor/actuator/rainfall/method contracts;
- the deterministic `rtc-build-method-testbed-v067` generator in `src/rtc/method_testbed_v067.py`.

The local run MUST verify all hashes before Step 0. If the corrected network or any of the 30 event INPs differs from the pinned manifest, fail closed and regenerate from the v0.6.7 generator. Do not substitute historical Project5/6/7 derived data.

Scientific scope: TFV-first methodology evaluation on an idealized simplified Wuhan SWMM; claim reduction of sewer-node overflow, not field digital-twin deployment.
