from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


TFV_STAGES = (
    "inp_preflight",
    "rainfall_split",
    "phase0_timescale",
    "d0_d1_coverage",
    "d2_d3_generation",
    "step1_acceptance",
    "step2_acceptance",
    "gradient_acceptance",
    "candidate_ranking_acceptance",
    "closed_loop_development",
    "policy_lock",
    "final_closed_loop_swmm",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TFVStageEvidence:
    stage: str
    passed: bool
    paths: tuple[str, ...]
    sha256: tuple[str, ...]
    notes: str = ""

    def validate(self) -> None:
        if self.stage not in TFV_STAGES:
            raise ValueError(f"unknown TFV-first stage: {self.stage}")
        if len(self.paths) != len(self.sha256):
            raise ValueError("evidence paths/hashes do not align")
        if self.passed and not self.paths:
            raise ValueError("passed stage requires hashed evidence")


@dataclass
class TFVPipelineLedger:
    contract: str = "WUHAN_RTC_TFV_FIRST_PIPELINE_V2"
    stages: dict[str, TFVStageEvidence] = field(default_factory=dict)

    def record_files(self, stage: str, paths: list[str | Path], *, passed: bool, notes: str = "") -> None:
        index = TFV_STAGES.index(stage)
        if passed:
            for prior in TFV_STAGES[:index]:
                if prior in {"policy_lock", "final_closed_loop_swmm"}:
                    continue
                evidence = self.stages.get(prior)
                if evidence is None or not evidence.passed:
                    raise ValueError(f"cannot pass {stage}: prerequisite {prior} has not passed")
        normalized = tuple(str(Path(p)) for p in paths)
        ev = TFVStageEvidence(
            stage=stage,
            passed=bool(passed),
            paths=normalized,
            sha256=tuple(sha256_file(p) for p in paths),
            notes=str(notes),
        )
        ev.validate()
        self.stages[stage] = ev

    def require_ready_for_lock(self) -> None:
        for stage in TFV_STAGES[: TFV_STAGES.index("policy_lock")]:
            ev = self.stages.get(stage)
            if ev is None or not ev.passed:
                raise RuntimeError(f"TFV-first pipeline blocked before Policy Lock: {stage}")
        self.verify_integrity()

    def verify_integrity(self) -> None:
        for stage, ev in self.stages.items():
            ev.validate()
            for path, expected in zip(ev.paths, ev.sha256, strict=True):
                p = Path(path)
                if not p.is_file() or sha256_file(p) != expected:
                    raise RuntimeError(f"TFV-first evidence missing/changed: {stage}: {p}")

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "contract": self.contract,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "TFVPipelineLedger":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(contract=str(raw.get("contract", "WUHAN_RTC_TFV_FIRST_PIPELINE_V2")))
        for name, item in raw.get("stages", {}).items():
            obj.stages[name] = TFVStageEvidence(
                stage=name,
                passed=bool(item["passed"]),
                paths=tuple(item.get("paths", [])),
                sha256=tuple(item.get("sha256", [])),
                notes=str(item.get("notes", "")),
            )
        return obj
