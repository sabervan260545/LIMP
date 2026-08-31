#!/usr/bin/env python3
"""Generate LIMP-DI-corrected pools and assemble the DI-to-AR bridge data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.benchmarking import (
    identity_cluster_stats,
    max_same_length_identity,
    sampled_pairwise_identity,
    sequence_properties,
    sequence_sha256,
)
from liup_generator.data import AMINO_ACIDS, sha256_file
from liup_generator.discriminative import CorrectedLIUPClassifier, generate_with_corrected_gradient


def read_fasta(path: Path) -> list[str]:
    output: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if current:
                output.append("".join(current))
                current = []
        elif line.strip():
            current.append(line.strip().upper())
    if current:
        output.append("".join(current))
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_corrected(path: Path, device: torch.device) -> CorrectedLIUPClassifier:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") not in {"LIUP-DI-corrected", "LIMP-DI-corrected"}:
        raise ValueError(f"Unexpected checkpoint architecture: {checkpoint.get('architecture')}")
    model = CorrectedLIUPClassifier()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def generate_exact_di(model: CorrectedLIUPClassifier, target: int, seed: int, device: torch.device, batch_size: int, min_length: int, max_length: int, iterations: int, temperature: float, learning_rate: float) -> tuple[list[dict[str, object]], dict[str, object]]:
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
        generated = generate_with_corrected_gradient(
            model,
            count=requested,
            device=device,
            min_length=min_length,
            max_length=max_length,
            iterations=iterations,
            temperature=temperature,
            learning_rate=learning_rate,
            seed=round_seed,
        )
        if before is not None and after is not None:
            after.record()
            torch.cuda.synchronize(device)
            gpu_milliseconds += float(before.elapsed_time(after))
        for local_index, generated_row in enumerate(generated):
            raw_draws += 1
            sequence = generated_row.sequence
            if not sequence or not set(sequence).issubset(set(AMINO_ACIDS)) or not min_length <= len(sequence) <= max_length:
                invalid += 1
                continue
            if sequence in seen:
                duplicates += 1
                continue
            seen.add(sequence)
            rows.append({
                "implementation": "LIMP-DI-corrected",
                "seed": seed,
                "sequence_index": len(rows),
                "sequence_sha256": sequence_sha256(sequence),
                "sequence": sequence,
                "length": len(sequence),
                "liup_di_self_score_diagnostic_only": generated_row.self_score,
                "generation_round": rounds,
                "round_seed": round_seed,
                "round_local_index": local_index,
            })
            if len(rows) == target:
                break
    wall = time.perf_counter() - started
    gpu = gpu_milliseconds / 1000.0
    summary = {
        "implementation": "LIMP-DI-corrected",
        "seed": seed,
        "target_unique": target,
        "final_unique": len(rows),
        "exact_count_complete": len(rows) == target,
        "raw_draws": raw_draws,
        "raw_duplicates": duplicates,
        "raw_duplicate_rate": duplicates / raw_draws,
        "invalid_draws": invalid,
        "validity_rate": 1.0 - invalid / raw_draws,
        "generation_batches": rounds,
        "generation_wall_seconds": wall,
        "generation_gpu_seconds": gpu,
        "valid_unique_per_wall_second": len(rows) / wall,
        "valid_unique_per_gpu_second": len(rows) / gpu if gpu else float("nan"),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--di-checkpoint", type=Path, required=True)
    parser.add_argument("--training-fasta", type=Path, required=True)
    parser.add_argument("--ar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260727, 20260728, 20260729])
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=6e-3)
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=28)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_corrected(args.di_checkpoint, device)
    training_sequences = read_fasta(args.training_fasta)
    training_set = set(training_sequences)
    bridge_summary: list[dict[str, object]] = []
    property_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        rows, summary = generate_exact_di(model, args.target, seed, device, args.batch_size, args.min_length, args.max_length, args.iterations, args.temperature, args.learning_rate)
        sequences = [str(row["sequence"]) for row in rows]
        nearest = max_same_length_identity(training_sequences, sequences)
        clusters = identity_cluster_stats(sequences, 0.80)
        pairwise = sampled_pairwise_identity(sequences, 100000, seed)
        for row, identity in zip(rows, nearest):
            row["exact_training_overlap"] = str(row["sequence"]) in training_set
            row["nearest_training_identity"] = float(identity)
            property_rows.append({"implementation": "LIMP-DI-corrected", "seed": seed, "sequence_sha256": row["sequence_sha256"], **sequence_properties(str(row["sequence"]))})
        summary.update({
            "exact_training_overlap_count": sum(bool(row["exact_training_overlap"]) for row in rows),
            "exact_training_overlap_rate": float(np.mean([bool(row["exact_training_overlap"]) for row in rows])),
            "nearest_training_identity_mean": float(nearest.mean()),
            "nearest_training_identity_median": float(np.median(nearest)),
            "nearest_training_identity_p95": float(np.quantile(nearest, 0.95)),
            **clusters,
            **pairwise,
        })
        bridge_summary.append(summary)
        seed_dir = args.output_dir / "liup_di_corrected" / f"seed_{seed}"
        write_csv(seed_dir / "sequences.csv", rows)
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    # Reuse immutable AR-1 pools and summaries without selective regeneration.
    for seed in args.seeds:
        ar_rows = pd.read_csv(args.ar_dir / f"seed_{seed}/sequences.csv")
        ar_summary = json.loads((args.ar_dir / f"seed_{seed}/summary.json").read_text(encoding="utf-8"))
        ar_summary["implementation"] = "LIMP-AR"
        bridge_summary.append(ar_summary)
        for row in ar_rows.itertuples(index=False):
            property_rows.append({"implementation": "LIMP-AR", "seed": seed, "sequence_sha256": row.sequence_sha256, **sequence_properties(row.sequence)})

    # Cross-seed reproducibility is overlap, not an expectation of identical pools.
    overlap_rows: list[dict[str, object]] = []
    for implementation, loader in (
        ("LIMP-DI-corrected", lambda seed: pd.read_csv(args.output_dir / f"liup_di_corrected/seed_{seed}/sequences.csv")),
        ("LIMP-AR", lambda seed: pd.read_csv(args.ar_dir / f"seed_{seed}/sequences.csv")),
    ):
        sets = {seed: set(loader(seed)["sequence"].astype(str)) for seed in args.seeds}
        for index, left in enumerate(args.seeds):
            for right in args.seeds[index + 1 :]:
                overlap = sets[left] & sets[right]
                union = sets[left] | sets[right]
                overlap_rows.append({"implementation": implementation, "seed_left": left, "seed_right": right, "intersection_n": len(overlap), "jaccard": len(overlap) / len(union)})

    write_csv(args.output_dir / "bridge_generation_summary.csv", bridge_summary)
    write_csv(args.output_dir / "bridge_sequence_properties.csv", property_rows)
    write_csv(args.output_dir / "cross_seed_overlap.csv", overlap_rows)
    manifest = {
        "status": "generation_complete_predictor_panel_pending",
        "di_checkpoint": str(args.di_checkpoint),
        "di_checkpoint_sha256": sha256_file(args.di_checkpoint),
        "training_fasta": str(args.training_fasta),
        "training_fasta_sha256": sha256_file(args.training_fasta),
        "ar_multiseed_dir": str(args.ar_dir),
        "seeds": args.seeds,
        "target_unique_per_seed": args.target,
        "output_contract": {"min_length": args.min_length, "max_length": args.max_length, "canonical_only": True, "exact_unique": True},
        "di_native_optimization": {"iterations": args.iterations, "temperature": args.temperature, "learning_rate": args.learning_rate},
        "self_score_is_diagnostic_only": True,
        "device": str(device),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
