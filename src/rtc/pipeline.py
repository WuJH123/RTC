from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


STAGES = (
    "phase0_inp_timescale",
    "d0_d1_hydraulic_coverage",
    "d2_actuator_probes",
    "step1_training_acceptance",
    "step2_flow_acceptance",
    "step2_transition_acceptance",
    "step2_short_rollout_acceptance",
    "step2_horizon_rollout_acceptance",
    "candidate_ranking_acceptance",
    "gradient_truth_acceptance",
    "safety_calibration",
    "independent_safety_audit",
    "closed_loop_development",
    "policy_lock",
    "final_closed_loop_swmm",
)


@dataclass
class StageEvidence:
    stage: str
    passed: bool
    evidence_paths: tuple[str, ...] = ()
    evidence_sha256: tuple[str, ...] = ()
    notes: str = ""

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown pipeline stage: {self.stage}")
        if len(self.evidence_paths) != len(self.evidence_sha256):
            raise ValueError("evidence paths and hashes must align")
        if self.passed and not self.evidence_paths:
            raise ValueError("a passed stage requires hashed evidence")


@dataclass
class PipelineLedger:
    contract: str = "WUHAN_RTC_ACTUATOR_AGNOSTIC_V1"
    stages: dict[str, StageEvidence] = field(default_factory=dict)

    def record(self, evidence: StageEvidence) -> None:
        evidence.validate()
        index = STAGES.index(evidence.stage)
        if evidence.passed:
            for prior in STAGES[:index]:
                prior_ev = self.stages.get(prior)
                if prior_ev is None or not prior_ev.passed:
                    raise ValueError(
                        f"cannot pass {evidence.stage}: prerequisite {prior} is not passed"
                    )
        self.stages[evidence.stage] = evidence

    def passed(self, stage: str) -> bool:
        return bool(self.stages.get(stage) and self.stages[stage].passed)

    def require_ready_for(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        for prior in STAGES[: STAGES.index(stage)]:
            if not self.passed(prior):
                raise RuntimeError(f"pipeline is blocked before {stage}: {prior} has not passed")

    def verify_integrity(self) -> None:
        """Fail if any previously recorded evidence file has changed or disappeared."""

        for stage, evidence in self.stages.items():
            evidence.validate()
            for path, expected in zip(evidence.evidence_paths, evidence.evidence_sha256, strict=True):
                p = Path(path)
                if not p.is_file():
                    raise RuntimeError(f"pipeline evidence disappeared: {stage}: {path}")
                current = sha256_file(p)
                if current != expected:
                    raise RuntimeError(f"pipeline evidence hash changed: {stage}: {path}")

    def to_json(self, path: str | Path) -> None:
        payload = {
            "contract": self.contract,
            "stages": {name: asdict(ev) for name, ev in self.stages.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger = cls(contract=payload.get("contract", "WUHAN_RTC_ACTUATOR_AGNOSTIC_V1"))
        for stage, raw in payload.get("stages", {}).items():
            ledger.stages[stage] = StageEvidence(
                stage=stage,
                passed=bool(raw["passed"]),
                evidence_paths=tuple(raw.get("evidence_paths", [])),
                evidence_sha256=tuple(raw.get("evidence_sha256", [])),
                notes=str(raw.get("notes", "")),
            )
        return ledger


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_from_files(stage: str, paths: list[str | Path], *, passed: bool, notes: str = "") -> StageEvidence:
    normalized = tuple(str(Path(p)) for p in paths)
    hashes = tuple(sha256_file(p) for p in paths)
    return StageEvidence(stage=stage, passed=passed, evidence_paths=normalized, evidence_sha256=hashes, notes=notes)


def _require_passed_json(path: str | Path, name: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError(f"policy lock artifact {name} is not a passed evidence JSON")


def create_policy_lock(
    *,
    ledger: PipelineLedger,
    artefacts: dict[str, str | Path],
    output_path: str | Path,
) -> dict[str, object]:
    """Freeze the complete production/evidence lineage before untouched Final."""

    ledger.require_ready_for("policy_lock")
    ledger.verify_integrity()
    required = {
        "step1_model",
        "step2_model",
        "graph_schema",
        "state_schema",
        "actuator_catalog",
        "split_registry",
        "model_acceptance_contract",
        "step1_acceptance",
        "step2_acceptance",
        "gradient_acceptance",
        "candidate_ranking_acceptance",
        "safety_calibration",
        "safety_audit",
        "controller_config",
        "rainfall_forecast_config",
        "fallback_policy",
        "baseline_plan",
    }
    missing = sorted(required - set(artefacts))
    if missing:
        raise ValueError(f"policy lock missing required artefacts: {missing}")
    for name, path in artefacts.items():
        if not Path(path).is_file():
            raise ValueError(f"policy lock artifact does not exist: {name}: {path}")
    for name in ("step1_acceptance", "step2_acceptance", "gradient_acceptance", "candidate_ranking_acceptance", "safety_audit"):
        _require_passed_json(artefacts[name], name)
    hashes = {name: sha256_file(path) for name, path in sorted(artefacts.items())}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    policy_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "contract": "WUHAN_RTC_POLICY_LOCK_V1",
        "scientific_contract": ledger.contract,
        "policy_sha256": policy_sha,
        "artefacts": {name: str(Path(path)) for name, path in sorted(artefacts.items())},
        "sha256": hashes,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
