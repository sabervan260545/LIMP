#!/usr/bin/env python3
"""Prepare an AMP-only, similarity-aware dataset for LIMP-AR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.data import (
    cluster_same_length_identity,
    read_amp_fasta,
    split_by_cluster,
    write_generator_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_amp_fasta(args.amp_fasta, label="amp")
    clustered = cluster_same_length_identity(records, args.identity_threshold)
    split_records = split_by_cluster(
        clustered,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    metadata = write_generator_dataset(
        split_records,
        args.output_dir,
        args.amp_fasta,
        args.identity_threshold,
        args.seed,
    )
    print(f"Prepared {metadata['records']['count']} AMP sequences in {args.output_dir}")
    print(f"Clusters: {metadata['records']['clusters']}; splits: {metadata['records']['splits']}")


if __name__ == "__main__":
    main()
