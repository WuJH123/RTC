"""Fail-closed V128 preflight before expensive training/evidence/runtime work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import torch

from .checkpoint_v128 import V128_CHECKPOINT_CONTRACT, load_step2_v128
from .engineering_v128 import (
    idealized_engineering_envelope_v128,
    load_engineering_envelope_v128,
)
from .production_cli import _load_graph
from .step2_differentiable_v128 import V128_STEP2_CONTRACT
from .v128_control_profile import configure_v128_cuda_matmul_precision

V128_PREFLIGHT_CONTRACT = "PROJECT7_V128_FAIL_CLOSED_PREFLIGHT_V1"
V128_EVIDENCE_CONTRACT = "PROJECT7_V128_CONTINUOUS_MPC_EVIDENCE_V1_TYPED_SAME_CHECKPOINT"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _system_ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return float(status.ullTotalPhys / (1024**3))
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * size / (1024**3))
    except Exception:
        return None


def inspect_v128_preflight(
    *,
    graph_path: str | Path,
    step2_path: str | Path | None = None,
    evidence_path: str | Path | None = None,
    engineering_envelope_path: str | Path | None = None,
    device_text: str = "cuda",
) -> dict[str, Any]:
    matmul_profile = configure_v128_cuda_matmul_precision()
    graph = _load_graph(graph_path)
    if len(graph.actuator_ids) != 109:
        raise ValueError("V128 requires exactly 109 ordered writable actuators")
    if len(set(graph.actuator_ids)) != 109:
        raise ValueError("V128 graph actuator IDs are not unique")

    device = torch.device(
        device_text if device_text == "cuda" and torch.cuda.is_available() else "cpu"
    )
    hardware: dict[str, Any] = {
        "requested_device": device_text,
        "resolved_device": str(device),
        "system_ram_gb": _system_ram_gb(),
        "float32_matmul_precision_before": matmul_profile["float32_matmul_precision_before"],
        "float32_matmul_precision": matmul_profile["float32_matmul_precision"],
        "amp_enabled": False,
    }
    if device_text == "cuda" and device.type != "cuda":
        raise RuntimeError("V128 CUDA preflight requested but CUDA is unavailable")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        hardware.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_total_gb": float(total / (1024**3)),
                "cuda_free_gb": float(free / (1024**3)),
            }
        )
        if total < 7.0 * 1024**3:
            raise RuntimeError("V128 4060 profile expects approximately 8 GB or more CUDA VRAM")

    envelope = (
        load_engineering_envelope_v128(engineering_envelope_path, graph=graph)
        if engineering_envelope_path
        else idealized_engineering_envelope_v128(graph)
    )
    payload: dict[str, Any] = {
        "contract": V128_PREFLIGHT_CONTRACT,
        "passed": True,
        "graph_sha256": _sha(graph_path),
        "actuator_count": len(graph.actuator_ids),
        "hardware": hardware,
        "engineering_envelope_source": envelope.source,
        "engineering_envelope_semantic_sha256": envelope.semantic_sha256,
        "engineering_envelope_is_idealized_default": envelope.is_idealized_default,
        "step2_checked": False,
        "evidence_checked": False,
    }

    if step2_path is not None:
        _, checkpoint = load_step2_v128(step2_path, graph=graph, device=device)
        if checkpoint.get("checkpoint_contract") != V128_CHECKPOINT_CONTRACT:
            raise ValueError("V128 preflight loaded the wrong checkpoint contract")
        if checkpoint.get("step2_contract") != V128_STEP2_CONTRACT:
            raise ValueError("V128 preflight loaded the wrong Step2 contract")
        payload.update(
            {
                "step2_checked": True,
                "step2_sha256": _sha(step2_path),
                "step2_contract": V128_STEP2_CONTRACT,
                "checkpoint_contract": V128_CHECKPOINT_CONTRACT,
            }
        )

    if evidence_path is not None:
        if step2_path is None:
            raise ValueError("V128 evidence preflight requires --step2 for SHA binding")
        evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        if evidence.get("contract") != V128_EVIDENCE_CONTRACT or evidence.get("passed") is not True:
            raise ValueError("V128 continuous evidence contract/pass flag is invalid")
        if evidence.get("step2_contract") != V128_STEP2_CONTRACT:
            raise ValueError("V128 evidence is not typed-Step2 evidence")
        if evidence.get("checkpoint_contract") != V128_CHECKPOINT_CONTRACT:
            raise ValueError("V128 evidence checkpoint contract mismatch")
        if str(evidence.get("step2_sha256", "")).lower() != _sha(step2_path).lower():
            raise ValueError("V128 evidence SHA does not match selected Step2 checkpoint")
        if not envelope.is_idealized_default and not bool(
            evidence.get("custom_engineering_envelope_supported_by_this_d5_evidence", False)
        ):
            raise ValueError("custom engineering envelope lacks matching V128 D5 evidence")
        payload.update(
            {
                "evidence_checked": True,
                "evidence_sha256": _sha(evidence_path),
                "evidence_contract": V128_EVIDENCE_CONTRACT,
            }
        )
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--step2")
    p.add_argument("--continuous-evidence")
    p.add_argument("--engineering-envelope")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out")
    args = p.parse_args()
    payload = inspect_v128_preflight(
        graph_path=args.graph,
        step2_path=args.step2,
        evidence_path=args.continuous_evidence,
        engineering_envelope_path=args.engineering_envelope,
        device_text=args.device,
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
