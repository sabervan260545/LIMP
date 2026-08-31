#!/usr/bin/env python3
"""Train the compact AMP-only autoregressive generator on prepared splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.autoregressive import AMPAutoregressiveModel, collate_autoregressive, encode_sequence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_manifest(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def make_loader(rows: list[dict[str, str]], batch_size: int, shuffle: bool, cluster_balanced: bool) -> DataLoader:
    encoded = [encode_sequence(row["sequence"]) for row in rows]
    if cluster_balanced:
        sizes = Counter(row["cluster_id"] for row in rows)
        weights = torch.DoubleTensor([1.0 / sizes[row["cluster_id"]] for row in rows])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(encoded, batch_size=batch_size, sampler=sampler, collate_fn=collate_autoregressive)
    return DataLoader(encoded, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_autoregressive)


@torch.no_grad()
def evaluate(model: AMPAutoregressiveModel, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item()
        total_tokens += targets.ne(0).sum().item()
    nll = total_loss / total_tokens
    return nll, math.exp(min(nll, 20.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    train_rows = read_manifest(args.manifest, "train")
    validation_rows = read_manifest(args.manifest, "validation")
    test_rows = read_manifest(args.manifest, "test")
    train_loader = make_loader(train_rows, args.batch_size, shuffle=True, cluster_balanced=True)
    validation_loader = make_loader(validation_rows, args.batch_size, shuffle=False, cluster_balanced=False)
    test_loader = make_loader(test_rows, args.batch_size, shuffle=False, cluster_balanced=False)

    model = AMPAutoregressiveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    best_state: dict[str, torch.Tensor] | None = None
    best_nll, no_improvement = float("inf"), 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = train_criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        validation_nll, validation_perplexity = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "validation_nll": validation_nll,
            "validation_perplexity": validation_perplexity,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_nll < best_nll:
            best_nll, no_improvement = validation_nll, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not create a checkpoint")
    model.load_state_dict(best_state)
    test_nll, test_perplexity = evaluate(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "liup_amp_autoregressive_transformer_v2",
        "model_config": model.config_dict(),
        "seed": args.seed,
        "state_dict": best_state,
        "best_validation_nll": best_nll,
    }
    torch.save(checkpoint, args.output_dir / "generator.pt")
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {
        "configuration": configuration,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_nll": best_nll,
        "test_nll": test_nll,
        "test_perplexity": test_perplexity,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
