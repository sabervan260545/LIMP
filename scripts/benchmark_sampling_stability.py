#!/usr/bin/env python3
"""Verify intrinsic generation gates across multiple deterministic sampling seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.autoregressive import AMPAutoregressiveModel, sample_sequences
from liup_generator.data import read_amp_fasta
from liup_generator.metrics import evaluate_generation, metrics_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=29)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AMPAutoregressiveModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    training = [record.sequence for record in read_amp_fasta(args.training_fasta)]
    results = []
    for seed in args.seeds:
        generated = sample_sequences(
            model,
            count=args.count,
            min_length=args.min_length,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=seed,
            device=device,
        )
        results.append({"seed": seed, "metrics": metrics_dict(evaluate_generation(
            [item.sequence for item in generated], training, args.min_length, args.max_length
        ))})
    max_run_rate = max(item["metrics"]["homopolymer_run_ge4_rate"] for item in results)
    max_overlap = max(item["metrics"]["exact_train_overlap_rate"] for item in results)
    min_unique = min(item["metrics"]["unique_rate"] for item in results)
    training_rate = results[0]["metrics"]["training_homopolymer_run_ge4_rate"]
    report = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "results": results,
        "acceptance": {
            "minimum_unique_rate": min_unique,
            "maximum_exact_train_overlap": max_overlap,
            "maximum_homopolymer_run_ge4_rate": max_run_rate,
            "training_homopolymer_run_ge4_rate": training_rate,
            "passed": (
                min_unique >= 0.95
                and max_overlap <= 0.01
                and max_run_rate <= training_rate + 0.03
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
