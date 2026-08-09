from __future__ import annotations

import hashlib
from pathlib import Path


CODE_CONTRACT = "RTC_PYTHON_SOURCE_TREE_V1"


def rtc_source_tree_sha256() -> str:
    """Hash the installed RTC Python source tree deterministically.

    The hash binds data generation/training to the actual Python implementation rather than
    to a human-maintained version string. Relative path, file length and content are hashed
    for every ``src/rtc/*.py`` module. Generated caches/bytecode are excluded.
    """

    root = Path(__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if p.is_file())
    if not files:
        raise RuntimeError(f"RTC source tree is empty: {root}")
    digest = hashlib.sha256()
    digest.update((CODE_CONTRACT + "\n").encode("utf-8"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()
