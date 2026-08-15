"""Low-overhead JSONL timing/resource telemetry for Project7 development runs."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch

V128_DEVELOPMENT_TELEMETRY_CONTRACT = "PROJECT7_V128_DEVELOPMENT_TELEMETRY_V1"


def resource_snapshot(device: torch.device | str) -> dict[str, Any]:
    target = torch.device(device)
    payload: dict[str, Any] = {
        "wall_time_unix": time.time(),
        "device": str(target),
    }
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        basic = process.memory_info()
        full = process.memory_full_info()
        vm = psutil.virtual_memory()
        payload.update(
            {
                "psutil_available": True,
                "process_rss_gb": float(basic.rss / 1024**3),
                "process_vms_gb": float(basic.vms / 1024**3),
                "system_ram_available_gb": float(vm.available / 1024**3),
                "system_ram_percent": float(vm.percent),
            }
        )
        for source_name, output_name in (
            ("uss", "process_uss_gb"),
            ("private", "process_private_gb"),
        ):
            value = getattr(full, source_name, None)
            if value is not None:
                payload[output_name] = float(value / 1024**3)
        swap = psutil.swap_memory()
        payload.update(
            {
                "swap_used_gb": float(swap.used / 1024**3),
                "swap_percent": float(swap.percent),
            }
        )
    except Exception:
        payload["psutil_available"] = False
    if target.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(target)
        payload.update(
            {
                "cuda_allocated_gb": float(torch.cuda.memory_allocated(target) / 1024**3),
                "cuda_reserved_gb": float(torch.cuda.memory_reserved(target) / 1024**3),
                "cuda_peak_allocated_gb": float(torch.cuda.max_memory_allocated(target) / 1024**3),
                "cuda_peak_reserved_gb": float(torch.cuda.max_memory_reserved(target) / 1024**3),
                "cuda_free_gb": float(free / 1024**3),
                "cuda_total_gb": float(total / 1024**3),
            }
        )
    return payload


class JsonlTelemetry:
    def __init__(self, path: str | Path, *, device: torch.device | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

    def emit(self, event: str, **fields: Any) -> None:
        row = {
            "contract": V128_DEVELOPMENT_TELEMETRY_CONTRACT,
            "event": str(event),
            **resource_snapshot(self.device),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


class TimedStage:
    def __init__(self, telemetry: JsonlTelemetry, stage: str):
        self.telemetry = telemetry
        self.stage = str(stage)
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        self.telemetry.emit("stage_start", stage=self.stage)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        self.telemetry.emit(
            "stage_end" if exc is None else "stage_error",
            stage=self.stage,
            elapsed_seconds=float(elapsed),
            error="" if exc is None else f"{type(exc).__name__}: {exc}",
        )
        return False


__all__ = [
    "JsonlTelemetry",
    "TimedStage",
    "V128_DEVELOPMENT_TELEMETRY_CONTRACT",
    "resource_snapshot",
]
