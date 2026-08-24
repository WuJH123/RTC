"""Compare an NG1 Development report with the frozen V5 Development frontier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.step2_ng1_comparator import compare_frontier_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    v5 = json.loads(Path(args.v5_report).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_report).read_text(encoding="utf-8"))
    result = compare_frontier_reports(v5, candidate)
    result["v5_report"] = str(Path(args.v5_report))
    result["candidate_report"] = str(Path(args.candidate_report))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result["PARETO_CORE_PASS"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
