#!/usr/bin/env python3
"""Run the first upgrade benchmark and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.benchmark_data import compare_random_and_cluster_splits
from liup_generator.data import read_amp_fasta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-fasta", type=Path, required=True)
    parser.add_argument("--non-amp-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    args = parser.parse_args()

    amp = read_amp_fasta(args.amp_fasta, label="amp")
    non_amp = read_amp_fasta(args.non_amp_fasta, label="non_amp")
    legacy, similarity_aware = compare_random_and_cluster_splits(
        amp, non_amp, identity_threshold=args.identity_threshold
    )
    report = {
        "identity_metric": "same_length_ungapped_identity",
        "identity_threshold": args.identity_threshold,
        "acceptance": {
            "similarity_aware_near_neighbor_rate_must_equal": 0.0,
            "similarity_aware_near_neighbor_rate": similarity_aware.near_neighbor_rate,
            "passed": similarity_aware.near_neighbor_rate == 0.0,
        },
        "legacy_random": asdict(legacy),
        "similarity_aware": asdict(similarity_aware),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
