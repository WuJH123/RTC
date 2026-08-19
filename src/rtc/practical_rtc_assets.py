"""Strict file manifest for the Project7 Practical RTC execution surface.

Current Project7 keeps the frozen 109-channel Step2 representation but requires one additional
label-independent artifact: the native supervisory-control mask derived from the source INP
``[CONTROLS]``. The current Wuhan testbed resolves to 82 online control freedoms; passive setting
channels remain in the hydraulic representation but cannot differ between candidate and reference.

Every current script verifies absolute paths and SHA-256. Silent fallback to historical V* artifacts
or legacy V12 admissions remains forbidden.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


PRACTICAL_RTC_ASSET_MANIFEST_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_ABSOLUTE_ASSET_MANIFEST_V3_NATIVE_CONTROL_SUBSPACE"
)
PRACTICAL_RTC_REQUIRED_ASSETS = (
    "graph",
    "sensors",
    "config",
    "step1",
    "step2",
    "supervisory_control",
    "sequence_support",
    "priority8",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_practical_rtc_asset_manifest(paths: Mapping[str, str | Path]) -> dict:
    missing_keys = sorted(set(PRACTICAL_RTC_REQUIRED_ASSETS) - set(paths))
    extra_keys = sorted(set(paths) - set(PRACTICAL_RTC_REQUIRED_ASSETS))
    if missing_keys or extra_keys:
        raise ValueError(f"asset manifest keys mismatch; missing={missing_keys}, extra={extra_keys}")
    assets: dict[str, dict[str, str]] = {}
    for key in PRACTICAL_RTC_REQUIRED_ASSETS:
        path = Path(paths[key]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Practical RTC asset {key} does not exist: {path}")
        assets[key] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "contract": PRACTICAL_RTC_ASSET_MANIFEST_CONTRACT,
        "absolute_paths_only": True,
        "silent_path_fallback_allowed": False,
        "legacy_v12_admissions_required": False,
        "model_action_channel_count": 109,
        "online_control_dimension_expected": 82,
        "step1_retraining_required": False,
        "base_step2_retraining_required": False,
        "assets": assets,
    }


def load_practical_rtc_asset_manifest(path: str | Path) -> dict:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != PRACTICAL_RTC_ASSET_MANIFEST_CONTRACT:
        raise ValueError("wrong Practical RTC asset-manifest contract")
    if payload.get("absolute_paths_only") is not True or payload.get("silent_path_fallback_allowed") is not False:
        raise ValueError("Practical RTC asset manifest does not prohibit path fallback")
    if payload.get("legacy_v12_admissions_required") is not False:
        raise ValueError("current Practical RTC manifest must not require legacy V12 admissions")
    if int(payload.get("model_action_channel_count", -1)) != 109:
        raise ValueError("current Practical RTC manifest must retain the 109-channel model representation")
    if int(payload.get("online_control_dimension_expected", -1)) != 82:
        raise ValueError("current Practical RTC manifest must freeze the 82-control online subspace")
    if payload.get("step1_retraining_required") is not False or payload.get("base_step2_retraining_required") is not False:
        raise ValueError("current Practical asset contract must reuse frozen Step1/base Step2")
    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(PRACTICAL_RTC_REQUIRED_ASSETS):
        raise ValueError("Practical RTC asset manifest has incomplete/extra assets")
    for key in PRACTICAL_RTC_REQUIRED_ASSETS:
        row = assets[key]
        if not isinstance(row, dict):
            raise ValueError(f"asset manifest row {key} is invalid")
        item = Path(str(row.get("path", "")))
        if not item.is_absolute() or not item.is_file():
            raise FileNotFoundError(f"Practical RTC asset path invalid for {key}: {item}")
        expected = str(row.get("sha256", "")).lower()
        actual = sha256_file(item)
        if actual != expected:
            raise ValueError(f"Practical RTC asset SHA drift for {key}: {item}")
    payload["manifest_path"] = str(manifest_path)
    return payload


def practical_asset_path(manifest: Mapping, key: str) -> str:
    if key not in PRACTICAL_RTC_REQUIRED_ASSETS:
        raise KeyError(f"unknown Practical RTC asset: {key}")
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("Practical RTC manifest lacks assets")
    row = assets[key]
    if not isinstance(row, Mapping):
        raise ValueError(f"Practical RTC manifest row invalid: {key}")
    return str(row["path"])


__all__ = [
    "PRACTICAL_RTC_ASSET_MANIFEST_CONTRACT",
    "PRACTICAL_RTC_REQUIRED_ASSETS",
    "build_practical_rtc_asset_manifest",
    "load_practical_rtc_asset_manifest",
    "practical_asset_path",
    "sha256_file",
]
