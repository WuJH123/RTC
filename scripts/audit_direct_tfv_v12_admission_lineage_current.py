"""Audit whether a V12 admission is behaviorally valid for current source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.direct_tfv_v12_lineage_audit import audit_v12_admission_lineage


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--step2-checkpoint", required=True)
    parser.add_argument("--sequence-support", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--require-compatible",
        action="store_true",
        help="Exit nonzero when the admission is not reusable by current V12 source.",
    )
    args = parser.parse_args()

    admission_path = Path(args.admission).resolve()
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if not isinstance(admission, dict):
        raise ValueError("V12 admission must be a JSON object")

    payload = audit_v12_admission_lineage(
        admission,
        step2_checkpoint_sha256=_sha(args.step2_checkpoint),
        sequence_support_sha256=_sha(args.sequence_support),
    )
    payload["admission_path"] = str(admission_path)
    payload["admission_sha256"] = _sha(admission_path)
    payload["step2_checkpoint_path"] = str(Path(args.step2_checkpoint).resolve())
    payload["sequence_support_path"] = str(Path(args.sequence_support).resolve())

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")

    if args.require_compatible and not payload["safe_to_reuse_admission"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
