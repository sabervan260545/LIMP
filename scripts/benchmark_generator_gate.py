#!/usr/bin/env python3
"""Compare legacy and v2 generator artifacts against explicit acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-exact-train-overlap", type=float, default=0.01)
    parser.add_argument("--min-unique-rate", type=float, default=0.95)
    args = parser.parse_args()

    legacy, candidate = load(args.legacy), load(args.candidate)
    legacy_metrics = legacy["metrics"]
    candidate_metrics = candidate["metrics"]
    gates = {
        "same_sample_count": candidate_metrics["count"] == legacy_metrics["count"],
        "all_tokens_are_valid": candidate_metrics["validity_rate"] == 1.0,
        "all_lengths_are_compliant": candidate_metrics["length_compliance_rate"] == 1.0,
        "unique_rate": candidate_metrics["unique_rate"] >= args.min_unique_rate,
        "exact_train_overlap": candidate_metrics["exact_train_overlap_rate"] <= args.max_exact_train_overlap,
        "composition_closer_than_legacy": (
            candidate_metrics["amino_acid_l1_distance"] < legacy_metrics["amino_acid_l1_distance"]
        ),
        "faster_than_legacy": candidate["elapsed_seconds"] < legacy["elapsed_seconds"],
        "homopolymer_rate_matches_training_distribution": (
            candidate_metrics["homopolymer_run_ge4_rate"]
            <= candidate_metrics["training_homopolymer_run_ge4_rate"] + 0.03
        ),
    }
    report = {
        "legacy": {
            "generator": legacy["generator"],
            "elapsed_seconds": legacy["elapsed_seconds"],
            "metrics": legacy_metrics,
        },
        "candidate": {
            "generator": candidate["generator"],
            "elapsed_seconds": candidate["elapsed_seconds"],
            "metrics": candidate_metrics,
        },
        "relative": {
            "speedup": legacy["elapsed_seconds"] / candidate["elapsed_seconds"],
            "composition_l1_reduction": 1.0
            - candidate_metrics["amino_acid_l1_distance"] / legacy_metrics["amino_acid_l1_distance"],
        },
        "acceptance": {"gates": gates, "passed": all(gates.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
