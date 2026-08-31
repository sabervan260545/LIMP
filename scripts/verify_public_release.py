#!/usr/bin/env python3
"""Verify the standalone LIMP release candidate without network access."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = "c25b229bd85f321926d83c0976d4c13372bff6f23e5f645a6535a1f6074562d8"
EXPECTED_FIELDS = ["sequence_sha256", "length", "label", "source_class", "cluster_id", "split"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> int:
    manifest = ROOT / "SHA256SUMS.txt"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"manifest mismatch: {relative}")
        checked += 1
    return checked


def verify_split() -> int:
    path = ROOT / "data_split/split_manifest_hash_only.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_FIELDS:
            raise RuntimeError(f"unexpected split fields: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 5652:
        raise RuntimeError(f"unexpected split row count: {len(rows)}")
    if any(not re.fullmatch(r"[0-9a-f]{64}", row["sequence_sha256"]) for row in rows):
        raise RuntimeError("invalid sequence SHA-256 in split manifest")
    return len(rows)


def verify_checkpoint() -> dict[str, object]:
    path = ROOT / "checkpoint/LIMP-AR_generator.pt"
    if sha256(path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    parameters = sum(tensor.numel() for tensor in checkpoint["state_dict"].values())
    if checkpoint["architecture"] != "liup_amp_autoregressive_transformer_v2":
        raise RuntimeError("unexpected checkpoint architecture")
    if parameters != 407703:
        raise RuntimeError(f"unexpected parameter count: {parameters}")
    return {
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "parameters": parameters,
        "model_config": checkpoint["model_config"],
        "best_validation_nll": checkpoint["best_validation_nll"],
    }


def verify_formal_config() -> dict[str, object]:
    path = ROOT / "configs/formal_100k_generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "requested_unique_sequences": 100000,
        "seed": 20260727,
        "temperature": 1.0,
        "top_k": 10,
        "top_p": 1.0,
        "min_length": 12,
        "max_length": 28,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"formal configuration mismatch: {mismatches}")
    return expected


def verify_no_raw_training_sequences() -> None:
    forbidden_suffixes = {".fasta", ".fa", ".faa"}
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
    if forbidden:
        raise RuntimeError(f"raw FASTA files present: {forbidden}")


def main() -> int:
    report = {
        "schema_version": "limp-public-release-verification-v1",
        "manifest_files_verified": verify_manifest(),
        "split_rows_verified": verify_split(),
        "checkpoint": verify_checkpoint(),
        "formal_generation": verify_formal_config(),
        "raw_training_fasta_present": False,
        "technical_release_ready": True,
        "public_upload_ready": (ROOT / "LICENSE").is_file(),
        "remaining_gate": "replace neutral author label with approved metadata when available",
    }
    verify_no_raw_training_sequences()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
