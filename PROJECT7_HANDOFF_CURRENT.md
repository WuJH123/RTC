# Project7 current handoff — three-family 82-control / 109-representation RTC

## Repository and state

- Repository: `WuJH123/RTC`
- Development branch: `agent/core-policy-return-portfolio-v14-r1`
- PR: #106, intentionally Draft until independent Development evidence is complete.
- Local repository: `E:\RTC_sewer\Project7\repo`
- Preserve the user's local documentation edits. Never use destructive reset/clean/restore workflows.
- `READY_FOR_POLICY_LOCK=false`.

## Frozen research question

Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable learning
controller reduce **system-wide cumulative total flooding volume (TFV)** in the Wuhan SWMM methodology
testbed while avoiding material deterioration of flooding at the frozen Priority8 nodes?

Online objective hierarchy:

1. sole optimization objective: system-wide cumulative TFV;
2. authoritative secondary Priority8 PFV safety: `PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`;
3. Global Peak: report-only.

No future realised rainfall/state/flooding/Internal trajectory, online SWMM candidate search, PFV/Peak
action penalties, baseline imitation or baseline warm start is allowed.

## Physical/model contract

- model/observation grid: 300 s;
- supervisory decision: 600 s;
- critic context: H360;
- executable first action: H10;
- critic token: H10 candidate then H350 HOLD;
- HOLD = latch the previous supervisory target;
- SWMM is authoritative truth.

Hydraulic representation and controllability are deliberately separated:

- frozen Step2 representation: 109 action channels;
- native supervisory controls: 82 = 57 pumps + 16 orifices + 9 weirs;
- passive setting channels: 27, always candidate == HOLD/reference;
- five added `RTC_ST_*` Storage nodes remain frozen hydraulic state/capacity features;
- Step1 and base Step2 are not retrained because of the 82-control mask.

Current local artifacts from the completed mask migration:

- canonical source INP SHA: `75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea`;
- supervisory mask SHA: `333d8744a053cc823252d60df0388a34e11124a8f9fc1f28bf6296212ffd3ebe`;
- native-control artifact SHA: `0e036ba08b2cb3eb497c48328ea8409d62218f50d7c3a1fcd5d30b9b90149140`;
- masked sequence-support artifact SHA: `e43b5e1956c0bfffb230f45b5d8223446cd6b13b07a83b4c1e16def5f099e49c`;
- Practical asset-manifest SHA: `4151ed3505b9119d892543392dd948a088ff184a534ef9414f1aa2c060f8ea23`.

Masked q95 support: changed-K=20, first-block L1=4.55, H120 L1=32.00, H120 total variation=11.15.

## Completed seen mechanism evidence

The 82-control four-family Development mechanism panel froze six query points label-blind before
truth: T8 decisions 0/25, T30 decisions 0/10, T80 decisions 0/9. All same-prefix, continuation,
causality, passive-channel, engineering/readback/support gates passed.

Key result: **6/6 queries had at least one truly beneficial generated action**. Candidate coverage is
therefore not the principal bottleneck.

Base Step2 as a deployed return/rank oracle was weak:

- sign accuracy 0.45;
- false-beneficial 0.30;
- false-reject 0.25;
- within-query pairwise rank 0.4167;
- candidate top1 0.50.

Family mechanism evidence:

- Step2 scale 0.50: beneficial 3/6;
- Step2 scale 1.00: beneficial 2/3 when distinct;
- type-aware hydraulic pressure: beneficial 5/6, repeatedly oracle-best;
- projected gradient: beneficial 3/5, oracle-best 0/5.

Current scientific bottleneck:

`EXACT_RECEDING_POLICY_RETURN_SIGN_AND_SAME_PREFIX_RANKING_GENERALIZATION`.

## Current best simplification

The paper-facing online candidate portfolio is now **three-family only**:

1. `STEP2_H10_PROBE_SCALE_0.50`;
2. `STEP2_H10_PROBE_SCALE_1.00`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`.

Projected gradient remains code-level Development ablation only. It is excluded from current pi0, pi1,
matched policy-return datasets/calibration and online execution. Historical L-BFGS-B remains archival.

Base Step2 is retained as a pretrained action-effect representation and Step2-direction generator. It
is **not** trusted as the deployed first-action value oracle. The next learned component is the exact
receding-policy-return critic, initialized from Step2 and initially fine-tuned only in control/action
and interaction heads.

## Important estimand rule after the simplification

Removing gradient changes pi0 continuation. Therefore the completed four-family exact-return records
are mechanism evidence only and cannot become current three-family train/validation/calibration labels.

Before bulk learning, run a minimal continuation-specific recheck:

- freeze two three-family query points per seen T8/T30/T80 event before reading truth;
- first wave evaluates one query/event = 3 total;
- if all three have a beneficial candidate and technical gates pass, stop the recheck;
- otherwise evaluate the already-frozen reserve queries before changing architecture.

## Data-role boundary and next compute-saving step

The previous role-plan target remains 48 train / 12 model-selection validation / 24 conformal
calibration / 3 new Development groups. The prior inventory had not confirmed any eligible untouched
pool and therefore correctly did not start bulk.

Next action is **zero-SWMM forcing inventory first**. Search existing forcing/event metadata and exclude
all seen Development families, Step2/D3 TrainFit role groups and Validation/Final/Formal/PolicyLock
resources. Create only the forcing deficit after this inventory. Selection is deterministic,
forcing-only and label-blind.

Only after the three-family mechanism recheck and role assignments are frozen should exact policy-
return bulk generation start. Initially use one deterministic causal query per group with one shared
HOLD plus all distinct three-family candidates.

## Critic, calibration and promotion

The actual decision set is `{HOLD=0 + candidates}`. Model selection prioritizes HOLD-aware false-
beneficial, false-reject, same-query rank and selected regret before scalar MAE. Matched calibration
contains only current three-family rows and uses rainfall group as the independent split-conformal
unit.

New pi1 Development closed loops must compare unchanged No-control, Internal RTC, Auto-RBC and EFD on
identical forcing. All-max/min SETTING remain diagnostics. Positive method claims additionally require
Priority8 PFV safety. If pi1 materially changes the policy distribution, collect a role-disjoint Q^pi1
round before Policy Lock.

## Resource execution policy

Machine-resource measurements are telemetry only. The current Project7 workflow must **not** stop,
pause, skip a planned rainfall group, or change scientific execution because of free-RAM, paging,
pagefile, GPU-memory, GPU-utilization, CPU-utilization or similar threshold values. No host-memory or
GPU safety reserve is part of the current execution contract.

Authoritative candidate/HOLD branches remain sequential because of the PySWMM/current scientific
execution semantics, not because of a resource threshold. Real CUDA OOM, host allocation failure,
PySWMM failure, process crash, or scientific/lineage fail-closed exceptions remain real errors and
should surface normally. The canonical code contract is
`PROJECT7_RESOURCE_TELEMETRY_ONLY_NO_PREEMPTIVE_STOP_V1` in
`src/rtc/resource_execution_policy.py`.

## Do not do

Do not add pump-energy objectives, PID setpoints, new level penalties, online PFV surrogate, q99/K
relaxation, more candidate families, L-BFGS-B, gradient tuning, or Step1/base-Step2 retraining without
new independent evidence. Do not use Validation/Final/Formal/PolicyLock for tuning. Do not merge PR
#106 just because CI is green.
