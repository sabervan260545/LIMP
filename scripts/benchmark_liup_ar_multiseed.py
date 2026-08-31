#!/usr/bin/env python3
"""Generate three exact LIMP-AR pools with the formal sampling contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.autoregressive import AMPAutoregressiveModel, sample_sequences
from liup_generator.benchmarking import (
    identity_cluster_stats,
    max_same_length_identity,
    sampled_pairwise_identity,
    sequence_properties,
    sequence_sha256,
)
from liup_generator.data import AMINO_ACIDS, sha256_file


def read_fasta(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line.upper())
    if current:
        sequences.append("".join(current))
    return sequences


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: Path, device: torch.device) -> tuple[AMPAutoregressiveModel, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = AMPAutoregressiveModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def generate_exact(
    model: AMPAutoregressiveModel,
    target: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    min_length: int,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    raw_draws = duplicates = invalid = rounds = 0
    gpu_milliseconds = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    while len(rows) < target:
        rounds += 1
        requested = min(batch_size, target - len(rows))
        round_seed = seed + (rounds - 1) * 1_000_003
        before = after = None
        if device.type == "cuda":
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
        generated = sample_sequences(
            model,
            count=requested,
            device=device,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=round_seed,
        )
        if after is not None and before is not None:
            after.record()
            torch.cuda.synchronize(device)
            gpu_milliseconds += float(before.elapsed_time(after))
        for local_index, generated_row in enumerate(generated):
            raw_draws += 1
            sequence = generated_row.sequence
            valid = bool(sequence) and set(sequence).issubset(set(AMINO_ACIDS)) and min_length <= len(sequence) <= max_length
            if not valid:
                invalid += 1
                continue
            if sequence in seen:
                duplicates += 1
                continue
            seen.add(sequence)
            rows.append(
                {
                    "pool_seed": seed,
                    "sequence_index": len(rows),
                    "sequence_sha256": sequence_sha256(sequence),
                    "sequence": sequence,
                    "length": len(sequence),
                    "log_probability": generated_row.log_probability,
                    "length_normalized_nll": -generated_row.log_probability / (len(sequence) + 1),
                    "generation_round": rounds,
                    "round_seed": round_seed,
                    "round_local_index": local_index,
                }
            )
            if len(rows) == target:
                break
    wall_seconds = time.perf_counter() - started
    summary = {
        "seed": seed,
        "target_unique": target,
        "final_unique": len(rows),
        "exact_count_complete": len(rows) == target,
        "raw_draws": raw_draws,
        "raw_duplicates": duplicates,
        "raw_duplicate_rate": duplicates / raw_draws,
        "invalid_draws": invalid,
        "validity_rate": 1.0 - invalid / raw_draws,
        "refill_rounds": max(0, rounds - int(np.ceil(target / batch_size))),
        "generation_batches": rounds,
        "generation_wall_seconds": wall_seconds,
        "generation_gpu_seconds": gpu_milliseconds / 1000.0,
        "valid_unique_per_wall_second": len(rows) / wall_seconds,
        "valid_unique_per_gpu_second": len(rows) / (gpu_milliseconds / 1000.0) if gpu_milliseconds else float("nan"),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260727, 20260728, 20260729])
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=28)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    training_sequences = read_fasta(args.training_fasta)
    training_set = set(training_sequences)
    summary_rows: list[dict[str, object]] = []
    property_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        rows, summary = generate_exact(
            model,
            args.target,
            seed,
            device,
            args.batch_size,
            args.min_length,
            args.max_length,
            args.temperature,
            args.top_k,
            args.top_p,
        )
        sequences = [str(row["sequence"]) for row in rows]
        nearest = max_same_length_identity(training_sequences, sequences)
        clusters = identity_cluster_stats(sequences, threshold=0.80)
        pairwise = sampled_pairwise_identity(sequences, pairs=100000, seed=seed)
        for index, (row, identity) in enumerate(zip(rows, nearest)):
            row["exact_training_overlap"] = str(row["sequence"]) in training_set
            row["nearest_training_identity"] = float(identity)
            properties = sequence_properties(str(row["sequence"]))
            property_rows.append({"implementation": "LIMP-AR", "seed": seed, "sequence_sha256": row["sequence_sha256"], **properties})
        summary.update(
            {
                "exact_training_overlap_count": sum(bool(row["exact_training_overlap"]) for row in rows),
                "exact_training_overlap_rate": float(np.mean([bool(row["exact_training_overlap"]) for row in rows])),
                "nearest_training_identity_mean": float(nearest.mean()),
                "nearest_training_identity_median": float(np.median(nearest)),
                "nearest_training_identity_p95": float(np.quantile(nearest, 0.95)),
                "length_normalized_nll_mean": float(np.mean([float(row["length_normalized_nll"]) for row in rows])),
                "length_normalized_nll_median": float(np.median([float(row["length_normalized_nll"]) for row in rows])),
                **clusters,
                **pairwise,
            }
        )
        summary_rows.append(summary)
        seed_dir = args.output_dir / f"seed_{seed}"
        write_csv(seed_dir / "sequences.csv", rows)
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    write_csv(args.output_dir / "generation_seed_metrics.csv", summary_rows)
    write_csv(args.output_dir / "sequence_property_distributions.csv", property_rows)
    config = {
        "status": "complete",
        "implementation": "LIMP-AR",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "training_fasta": str(args.training_fasta),
        "training_fasta_sha256": sha256_file(args.training_fasta),
        "checkpoint_metadata": {key: value for key, value in checkpoint.items() if key != "state_dict"},
        "seeds": args.seeds,
        "target_unique_per_seed": args.target,
        "sampling": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p, "min_length": args.min_length, "max_length": args.max_length},
        "device": str(device),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
