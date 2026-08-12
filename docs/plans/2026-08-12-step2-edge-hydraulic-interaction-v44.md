# Step2 V4.4 Edge-Hydraulic Interaction Implementation Plan

**Goal:** Establish an auditable SWMM physical-link lineage and, only after it passes, add a zero-initialized edge-hydraulic residual to the frozen V4.3.3 D3 interaction path.

**Architecture:** Parse the frozen `wuhan_method_testbed_v067.inp` without executing SWMM, retain one forward/reverse directed edge per physical hydraulic link, and expose finite static plus causal head-gradient features. Keep the V4.3.3 baseline frozen; add only a zero-initialized edge residual and train only its parameters in the fixed tiny and 12-group in-sample mechanism cohorts.

**Tech Stack:** Python, PyTorch, NumPy, pytest, frozen SWMM INP/graph artifacts, FP32.

---

### Task 1: Establish physical-link lineage

**Files:**
- Create: `src/rtc/step2_edge_hydraulic_v44.py`
- Create: `tests/test_step2_edge_hydraulic_interaction_v44.py`
- Create: `scripts/audit_step2_edge_hydraulic_lineage_v44.py`
- Create: `docs/STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.json`
- Create: `docs/STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.md`

Parse all required static INP link sections, preserve physical link identity and orientation, compare the 2,420-edge legacy graph against the physical-link census, and fail closed if node IDs or link endpoints cannot be resolved.

### Task 2: Write failing contract tests

Add tests for one directed pair per physical link, reverse-edge identity/orientation, parallel-link retention, finite static normalization, causal dynamic features, unavailable link-flow rejection, and zero-action/single-action contracts.

### Task 3: Implement the residual-only V4.4 path

**Files:**
- Modify: `src/rtc/step2_control_response_v433.py`
- Create: `src/rtc/step2_control_response_v44.py`
- Create: `src/rtc/step2_train_response_v44.py`

Keep the existing V4.3.3 response as baseline, add a zero-initialized edge message residual using physical static features and causal reference head/depth context, and freeze Reference/D2/old D3 parameters.

### Task 4: Run bounded mechanism experiments

**Files:**
- Create: `scripts/run_step2_edge_hydraulic_interaction_v44.py`
- Create: `docs/STEP2_EDGE_HYDRAULIC_INTERACTION_AUDIT_V44.json`
- Create: `docs/STEP2_EDGE_HYDRAULIC_INTERACTION_AUDIT_V44.md`
- Create: `docs/STEP2_EDGE_HYDRAULIC_INTERACTION_V44_REPORT.json`
- Create: `docs/STEP2_EDGE_HYDRAULIC_INTERACTION_V44_REPORT.md`

Run lineage audit first, then the frozen tiny A/B experiment, and only if tiny passes run the identical 12-group in-sample mechanism micro. Record D2 invariance, edge ablation, magnitude strata, interaction decomposition, causality, and performance.

### Task 5: Verify and publish

Run full pytest, py_compile, and `git diff --check`; commit only source/tests/small reports/lineage manifest; push the branch and create a Draft PR based on `agent/step2-nodewise-tfv-correctness-v433`. Do not run SWMM, Formal, Validation, Final, full smoke, or production wiring.
