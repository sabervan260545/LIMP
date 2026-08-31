#!/usr/bin/env python3
"""Train the historical LIMP-DI classifier architecture as a benchmark artifact."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.data import read_amp_fasta
from liup_generator.legacy import LegacyLIUPClassifier, encode_one_hot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-fasta", type=Path, required=True)
    parser.add_argument("--non-amp-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    positives = [record.sequence for record in read_amp_fasta(args.amp_fasta, label="amp")]
    negatives = [record.sequence for record in read_amp_fasta(args.non_amp_fasta, label="non_amp")]
    sequences = positives + negatives
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.float32)
    max_length = max(map(len, sequences))
    one_hot, mask = encode_one_hot(sequences, max_length=max_length)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_indices, validation_indices = next(splitter.split(np.zeros(len(labels)), labels))
    train_data = TensorDataset(one_hot[train_indices], mask[train_indices], torch.from_numpy(labels[train_indices]))
    validation_data = TensorDataset(
        one_hot[validation_indices], mask[validation_indices], torch.from_numpy(labels[validation_indices])
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    model = LegacyLIUPClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy, no_improvement = -float("inf"), 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_count = 0.0, 0, 0
        for batch_one_hot, batch_mask, batch_labels in train_loader:
            batch_one_hot, batch_mask, batch_labels = (
                batch_one_hot.to(device), batch_mask.to(device), batch_labels.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_one_hot, batch_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_labels)
            train_correct += ((torch.sigmoid(logits) >= 0.5) == batch_labels.bool()).sum().item()
            train_count += len(batch_labels)

        model.eval()
        validation_logits: list[torch.Tensor] = []
        validation_labels: list[torch.Tensor] = []
        validation_loss = 0.0
        with torch.no_grad():
            for batch_one_hot, batch_mask, batch_labels in validation_loader:
                batch_one_hot, batch_mask, batch_labels = (
                    batch_one_hot.to(device), batch_mask.to(device), batch_labels.to(device)
                )
                logits = model(batch_one_hot, batch_mask)
                validation_loss += criterion(logits, batch_labels).item() * len(batch_labels)
                validation_logits.append(logits.cpu())
                validation_labels.append(batch_labels.cpu())
        probability = torch.sigmoid(torch.cat(validation_logits)).numpy()
        truth = torch.cat(validation_labels).numpy().astype(int)
        validation_accuracy = float(accuracy_score(truth, probability >= 0.5))
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "train_accuracy": train_correct / train_count,
            "validation_loss": validation_loss / len(truth),
            "validation_accuracy": validation_accuracy,
            "validation_auroc": float(roc_auc_score(truth, probability)),
            "validation_auprc": float(average_precision_score(truth, probability)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_accuracy > best_accuracy:
            best_accuracy, no_improvement = validation_accuracy, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Legacy benchmark did not produce a checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "legacy_liup_classifier_torch_reproduction",
        "max_length": max_length,
        "seed": args.seed,
        "state_dict": best_state,
        "best_validation_accuracy": best_accuracy,
    }
    torch.save(checkpoint, args.output_dir / "legacy_classifier.pt")
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {
        "configuration": configuration,
        "max_length": max_length,
        "best_validation_accuracy": best_accuracy,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", **{key: report[key] for key in report if key != "history"}}, indent=2))


if __name__ == "__main__":
    main()
