from __future__ import annotations

from pathlib import Path

import pytest

from rtc.practical_rtc_assets import (
    PRACTICAL_RTC_REQUIRED_ASSETS,
    build_practical_rtc_asset_manifest,
    load_practical_rtc_asset_manifest,
)


def _files(tmp_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key in PRACTICAL_RTC_REQUIRED_ASSETS:
        path = tmp_path / f"{key}.dat"
        path.write_text(f"asset={key}\n", encoding="utf-8")
        result[key] = path
    return result


def test_asset_manifest_freezes_absolute_paths_and_sha(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    payload = build_practical_rtc_asset_manifest(paths)
    manifest = tmp_path / "PRACTICAL_RTC_ASSETS.json"
    import json

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_practical_rtc_asset_manifest(manifest)
    assert loaded["silent_path_fallback_allowed"] is False
    assert loaded["absolute_paths_only"] is True
    assert set(loaded["assets"]) == set(PRACTICAL_RTC_REQUIRED_ASSETS)
    assert all(Path(row["path"]).is_absolute() for row in loaded["assets"].values())


def test_asset_manifest_fails_when_recorded_file_changes(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    payload = build_practical_rtc_asset_manifest(paths)
    manifest = tmp_path / "PRACTICAL_RTC_ASSETS.json"
    import json

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    paths["step2"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA drift for step2"):
        load_practical_rtc_asset_manifest(manifest)


def test_asset_manifest_never_replaces_a_missing_path(tmp_path: Path) -> None:
    paths = _files(tmp_path)
    payload = build_practical_rtc_asset_manifest(paths)
    manifest = tmp_path / "PRACTICAL_RTC_ASSETS.json"
    import json

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    paths["graph"].unlink()
    with pytest.raises(FileNotFoundError, match="graph"):
        load_practical_rtc_asset_manifest(manifest)
