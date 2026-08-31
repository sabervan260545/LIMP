#!/usr/bin/env python3
"""Generate an intrinsic-metrics baseline using the legacy gradient method."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.data import read_amp_fasta
from liup_generator.legacy import LegacyLIUPClassifier, generate_with_legacy_gradient
from liup_generator.metrics import evaluate_generation, metrics_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = LegacyLIUPClassifier().to(device)
    model.load_state_dict(saved["state_dict"])
    training_sequences = [record.sequence for record in read_amp_fasta(args.training_fasta)]
    started = time.perf_counter()
    generated = generate_with_legacy_gradient(
        model, count=args.count, iterations=args.iterations, seed=args.seed, device=device
    )
    sequences = [item.sequence for item in generated]
    metrics = evaluate_generation(sequences, training_sequences, min_length=10, max_length=29)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {
        "generator": "legacy_classifier_guided",
        "configuration": configuration,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics_dict(metrics),
        "sequences": [item.__dict__ for item in generated],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sequences"}, indent=2))


if __name__ == "__main__":
    main()
