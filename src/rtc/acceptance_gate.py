from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .acceptance import apply_metric_thresholds


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_gate(
    *,
    metrics_path: str | Path,
    contract_path: str | Path,
    section: str,
    output_path: str | Path,
) -> dict[str, object]:
    metrics_payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    raw = contract.get(section)
    if not isinstance(raw, dict):
        raise ValueError(f"acceptance contract lacks section {section}")
    minimum = {str(k): float(v) for k, v in raw.get("minimum", {}).items()}
    maximum = {str(k): float(v) for k, v in raw.get("maximum", {}).items()}
    metrics = metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("metrics evidence lacks metrics object")
    result = apply_metric_thresholds(
        {str(k): float(v) for k, v in metrics.items()}, minimum=minimum, maximum=maximum
    )
    payload: dict[str, object] = {
        "contract": "PREREGISTERED_ACCEPTANCE_GATE_V3",
        "section": section,
        "passed": result.passed,
        "failed_metrics": list(result.failed_metrics),
        "metrics": result.metrics,
        "thresholds": {"minimum": minimum, "maximum": maximum},
        "source_metrics_path": str(Path(metrics_path)),
        "source_metrics_sha256": _sha(metrics_path),
        "acceptance_contract_path": str(Path(contract_path)),
        "acceptance_contract_sha256": _sha(contract_path),
    }
    for key in ("model_sha256", "step2_sha256", "manifest_sha256"):
        if key in metrics_payload:
            payload[key] = metrics_payload[key]
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen preregistered acceptance threshold section")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--section", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = create_gate(
        metrics_path=args.metrics, contract_path=args.contract,
        section=args.section, output_path=args.out,
    )
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
