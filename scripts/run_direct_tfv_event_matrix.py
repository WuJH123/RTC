"""Run independent Project7 Development events concurrently with strict output isolation.

This is a throughput utility, not a source of real-time latency evidence.  Each child process owns one
SWMM/PySWMM instance and one output directory.  PySWMM intentionally prevents multiple simulations
inside one Python process, so concurrency is process based.  The default is two concurrent events;
three is allowed for workstation throughput experiments when RAM/VRAM headroom has been checked.

The runner never changes scientific CLI arguments.  It only dispatches already-frozen event commands.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


MATRIX_CONTRACT = "PROJECT7_DIRECT_TFV_DEVELOPMENT_EVENT_MATRIX_V1"


def _flag(name: str) -> str:
    return "--" + str(name).strip().replace("_", "-")


def _append_argument(command: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(_flag(name))
        return
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            command.extend((_flag(name), str(item)))
        return
    command.extend((_flag(name), str(value)))


def _load_events(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list) or not events:
        raise ValueError("event matrix requires a non-empty events list")
    result: list[dict[str, Any]] = []
    seen_event: set[str] = set()
    seen_out: set[str] = set()
    for raw in events:
        if not isinstance(raw, dict):
            raise ValueError("every event matrix entry must be a mapping")
        required = ("event_id", "inp", "out_dir", "run_id")
        missing = [name for name in required if not str(raw.get(name, "")).strip()]
        if missing:
            raise ValueError(f"event matrix entry misses {missing}")
        event_id = str(raw["event_id"])
        out_dir = str(Path(str(raw["out_dir"])).resolve())
        if event_id in seen_event:
            raise ValueError(f"duplicate event_id in event matrix: {event_id}")
        if out_dir in seen_out:
            raise ValueError(f"duplicate output directory in event matrix: {out_dir}")
        seen_event.add(event_id)
        seen_out.add(out_dir)
        inp = Path(str(raw["inp"]))
        if not inp.is_file():
            raise FileNotFoundError(f"event input does not exist: {inp}")
        result.append(dict(raw))
    return result


def _load_common_args(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("common args JSON must be a mapping from CLI name to value")
    forbidden = {"inp", "out_dir", "run_id"}
    overlap = sorted(forbidden & set(payload))
    if overlap:
        raise ValueError(f"common args cannot override per-event fields: {overlap}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events-json", required=True)
    p.add_argument("--common-args-json")
    p.add_argument(
        "--runtime-script",
        default="scripts/run_policy_direct_tfv_first_move_development.py",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--summary-out", required=True)
    p.add_argument(
        "--parallel-throughput-only",
        action="store_true",
        help="Required acknowledgement: child latency is not formal real-time latency evidence.",
    )
    args = p.parse_args()

    if not args.parallel_throughput_only:
        raise ValueError(
            "parallel event execution is throughput-only; pass --parallel-throughput-only explicitly"
        )
    if not 1 <= int(args.max_parallel) <= 3:
        raise ValueError("Development event matrix max-parallel must lie in [1,3]")
    runtime = Path(args.runtime_script)
    if not runtime.is_file():
        raise FileNotFoundError(f"runtime script does not exist: {runtime}")
    events = _load_events(args.events_json)
    common = _load_common_args(args.common_args_json)

    # Independent processes are mandatory: PySWMM's Simulation state manager rejects multiple open
    # SWMM simulations inside one Python process.  One thread per child also prevents BLAS/OpenMP
    # oversubscription while SWMM/event-level parallelism is used.
    child_env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env[key] = "1"

    pending = list(events)
    running: dict[subprocess.Popen[str], dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    while pending or running:
        while pending and len(running) < int(args.max_parallel):
            event = pending.pop(0)
            out = Path(str(event["out_dir"]))
            out.mkdir(parents=True, exist_ok=True)
            stdout_path = out / "matrix.stdout.log"
            stderr_path = out / "matrix.stderr.log"
            command = [
                str(args.python),
                str(runtime),
                "--inp", str(event["inp"]),
                "--out-dir", str(out),
                "--run-id", str(event["run_id"]),
            ]
            merged = dict(common)
            for key, value in event.items():
                if key not in {"event_id", "inp", "out_dir", "run_id"}:
                    merged[key] = value
            for name, value in merged.items():
                _append_argument(command, name, value)
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=child_env,
            )
            running[process] = {
                "event_id": str(event["event_id"]),
                "out_dir": str(out),
                "run_id": str(event["run_id"]),
                "command": command,
                "started": time.perf_counter(),
                "stdout_handle": stdout,
                "stderr_handle": stderr,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }

        completed = [process for process in running if process.poll() is not None]
        if not completed:
            time.sleep(0.5)
            continue
        for process in completed:
            record = running.pop(process)
            record["stdout_handle"].close()
            record["stderr_handle"].close()
            record["returncode"] = int(process.returncode or 0)
            record["elapsed_seconds"] = float(time.perf_counter() - record.pop("started"))
            record.pop("stdout_handle", None)
            record.pop("stderr_handle", None)
            finished.append(record)

    payload = {
        "contract": MATRIX_CONTRACT,
        "development_only": True,
        "parallel_throughput_only": True,
        "valid_for_real_time_latency_claim": False,
        "max_parallel": int(args.max_parallel),
        "event_count": len(events),
        "all_passed": all(int(item["returncode"]) == 0 for item in finished),
        "wall_elapsed_seconds": float(time.perf_counter() - started_all),
        "records": sorted(finished, key=lambda item: str(item["event_id"])),
    }
    summary = Path(args.summary_out)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
