#!/usr/bin/env python3
"""Aggregate the immutable v2 release gates into one auditable report.

This script deliberately checks only generator-intrinsic and packaging facts.
It never calls MIC, toxicity, hemolysis, novelty, or Pareto services; those are
owned by the Agent Harness after candidate generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import torch


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _health(url: str) -> dict:
    with urlopen(f"{url.rstrip('/')}/health", timeout=15) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-gate", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-training", type=Path, required=True)
    parser.add_argument("--generator-gate", type=Path, required=True)
    parser.add_argument("--sampling-stability", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_gate = _load(args.data_gate)
    legacy_checkpoint = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True)
    candidate_checkpoint = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=True)
    candidate_training = _load(args.candidate_training)
    generator_gate = _load(args.generator_gate)
    stability = _load(args.sampling_stability)
    health = _health(args.service_url)

    legacy_parameters = sum(tensor.numel() for tensor in legacy_checkpoint["state_dict"].values())
    candidate_parameters = sum(tensor.numel() for tensor in candidate_checkpoint["state_dict"].values())
    checks = {
        "data_similarity_isolation": data_gate["acceptance"]["passed"],
        "candidate_is_lightweight_at_most_500k_parameters": candidate_parameters <= 500_000,
        "candidate_is_smaller_than_legacy": candidate_parameters < legacy_parameters,
        "candidate_validation_nll_at_most_2_30": candidate_training["best_validation_nll"] <= 2.30,
        "legacy_comparison_gate": generator_gate["acceptance"]["passed"],
        "three_seed_sampling_stability": stability["acceptance"]["passed"],
        "service_is_ready": health.get("status") == "ready",
        "service_is_sequence_only": health.get("sequence_only") is True,
    }
    report = {
        "release": "LIMP AMP Generator v2",
        "scope": "AMP sequence generation only; downstream predictors and Pareto ranking remain in the Harness.",
        "parameters": {
            "legacy_classifier_guided": legacy_parameters,
            "candidate_autoregressive": candidate_parameters,
            "candidate_to_legacy_ratio": candidate_parameters / legacy_parameters,
        },
        "candidate_training": {
            "best_validation_nll": candidate_training["best_validation_nll"],
            "test_nll": candidate_training["test_nll"],
            "test_perplexity": candidate_training["test_perplexity"],
        },
        "generator_comparison": generator_gate["relative"],
        "sampling_stability": stability["acceptance"],
        "service_health": health,
        "acceptance": {"checks": checks, "passed": all(checks.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
