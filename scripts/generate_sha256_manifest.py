#!/usr/bin/env python3
"""Generate the deterministic repository SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records: list[tuple[str, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
            or path.suffix in {".pyc", ".pyo"}
            or path == OUTPUT
        ):
            continue
        records.append((sha256(path), path.relative_to(ROOT).as_posix()))
    OUTPUT.write_text("".join(f"{digest}  {path}\n" for digest, path in records), encoding="utf-8")
    print(f"wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
