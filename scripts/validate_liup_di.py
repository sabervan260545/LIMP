#!/usr/bin/env python3
"""Five-seed, cluster-disjoint validation of LIMP-DI legacy and corrected."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.benchmarking import max_same_length_identity, property_matrix, sequence_properties, sequence_sha256
from liup_generator.data import SequenceRecord, cluster_same_length_identity, read_amp_fasta, sha256_file
from liup_generator.discriminative import CorrectedLIUPClassifier
from liup_generator.legacy import LegacyLIUPClassifier, encode_one_hot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_joint_records(amp_fasta: Path, non_amp_fasta: Path, threshold: float) -> tuple[list[SequenceRecord], list[dict[str, object]]]:
    raw = read_amp_fasta(amp_fasta, "amp") + read_amp_fasta(non_amp_fasta, "non_amp")
    labels: dict[str, set[str]] = defaultdict(set)
    rows: dict[tuple[str, str], SequenceRecord] = {}
    for record in raw:
        labels[record.sequence].add(record.label)
        rows.setdefault((record.sequence, record.label), record)
    conflicts = [
        {"sequence_sha256": sequence_sha256(sequence), "sequence": sequence, "labels": ";".join(sorted(values))}
        for sequence, values in labels.items()
        if len(values) > 1
    ]
    conflict_sequences = {row["sequence"] for row in conflicts}
    unique = [record for (sequence, _), record in rows.items() if sequence not in conflict_sequences]
    clustered = cluster_same_length_identity(unique, identity_threshold=threshold)
    return clustered, conflicts


def make_split(labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val, test = next(outer.split(indices, labels, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1000)
    train_rel, val_rel = next(inner.split(train_val, labels[train_val], groups[train_val]))
    train, validation = train_val[train_rel], train_val[val_rel]
    return train, validation, test


def loaders(one_hot: torch.Tensor, mask: torch.Tensor, labels: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(one_hot[indices], mask[indices], torch.from_numpy(labels[indices].astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, pin_memory=True)


@torch.no_grad()
def predict(model: nn.Module, one_hot: torch.Tensor, mask: torch.Tensor, indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        logits = model(one_hot[selected].to(device), mask[selected].to(device))
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def train_model(model: nn.Module, train_loader: DataLoader, val_one_hot: torch.Tensor, val_mask: torch.Tensor, val_indices: np.ndarray, labels: np.ndarray, device: torch.device, epochs: int, patience: int, learning_rate: float, batch_size: int) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]], float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stalled = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_one_hot, batch_mask, batch_labels in train_loader:
            batch_one_hot = batch_one_hot.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_one_hot, batch_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            count += len(batch_labels)
        probability = predict(model, val_one_hot, val_mask, val_indices, device, batch_size)
        truth = labels[val_indices]
        score = float(balanced_accuracy_score(truth, probability >= 0.5))
        row = {"epoch": epoch, "train_loss": total_loss / count, "validation_balanced_accuracy": score}
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    return best_state, history, time.perf_counter() - started


def expected_calibration_error(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assigned == index
        if selected.any():
            ece += selected.mean() * abs(truth[selected].mean() - probability[selected].mean())
    return float(ece)


def metric_row(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    if len(np.unique(truth)) < 2:
        return {key: float("nan") for key in ("auroc", "auprc", "mcc", "balanced_accuracy", "accuracy", "brier", "ece")}
    return {
        "auroc": float(roc_auc_score(truth, probability)),
        "auprc": float(average_precision_score(truth, probability)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "brier": float(brier_score_loss(truth, probability)),
        "ece": expected_calibration_error(truth, probability),
    }


def matched_indices(train_sequences: list[str], test_sequences: list[str], truth: np.ndarray) -> np.ndarray:
    train_x, _ = property_matrix(train_sequences)
    test_x, _ = property_matrix(test_sequences)
    columns = [0, 1, 2]
    mean, scale = train_x[:, columns].mean(axis=0), train_x[:, columns].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (test_x[:, columns] - mean) / scale
    positive = np.where(truth == 1)[0]
    negative = np.where(truth == 0)[0]
    distance = np.linalg.norm(
        standardized[positive][:, None, :] - standardized[negative][None, :, :],
        axis=2,
    )
    left, right = linear_sum_assignment(distance)
    return np.sort(np.concatenate([positive[left], negative[right]]))


def shuffled_sequences(sequences: list[str], seed: int) -> list[str]:
    output: list[str] = []
    for index, sequence in enumerate(sequences):
        residues = list(sequence)
        rng = random.Random(seed + index * 104729)
        candidate = residues[:]
        for _ in range(10):
            rng.shuffle(candidate)
            if candidate != residues:
                break
        output.append("".join(candidate))
    return output


def score_sequences(model: nn.Module, sequences: list[str], device: torch.device, batch_size: int, max_length: int = 30) -> np.ndarray:
    one_hot, mask = encode_one_hot(sequences, max_length=max_length)
    return predict(model, one_hot, mask, np.arange(len(sequences)), device, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-fasta", type=Path, required=True)
    parser.add_argument("--non-amp-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260820, 20260821, 20260822, 20260823, 20260824])
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    records, conflicts = prepare_joint_records(args.amp_fasta, args.non_amp_fasta, args.identity_threshold)
    write_csv(args.output_dir / "conflicting_exact_duplicates.csv", conflicts)
    sequences = [record.sequence for record in records]
    labels = np.asarray([record.label == "amp" for record in records], dtype=int)
    groups = np.asarray([record.cluster_id for record in records], dtype=int)
    one_hot, mask = encode_one_hot(sequences, max_length=30)
    property_x, property_names = property_matrix(sequences)

    seed_metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    stratified: list[dict[str, object]] = []
    shuffle_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        train, validation, test = make_split(labels, groups, seed)
        split_by_index = {int(index): name for name, selected in (("train", train), ("validation", validation), ("test", test)) for index in selected}
        for index, record in enumerate(records):
            split_rows.append({"seed": seed, "sequence_sha256": sequence_sha256(record.sequence), "label": record.label, "cluster_id": record.cluster_id, "split": split_by_index[index], "length": len(record.sequence)})
        train_sequences = [sequences[index] for index in train]
        test_sequences = [sequences[index] for index in test]
        nearest_identity = max_same_length_identity(train_sequences, test_sequences)
        matched = matched_indices(train_sequences, test_sequences, labels[test])

        logistic = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))])
        logistic.fit(property_x[train], labels[train])
        logistic_probability = logistic.predict_proba(property_x[test])[:, 1]
        for subset_name, selected in (("all", np.arange(len(test))), ("matched", matched)):
            seed_metrics.append({"seed": seed, "implementation": "physicochemical_logistic", "subset": subset_name, "n": len(selected), **metric_row(labels[test][selected], logistic_probability[selected])})

        for name, factory in (("LIMP-DI-legacy", LegacyLIUPClassifier), ("LIMP-DI-corrected", CorrectedLIUPClassifier)):
            set_seed(seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = factory().to(device)
            train_loader = loaders(one_hot, mask, labels, train, args.batch_size, True, seed)
            state, history, elapsed = train_model(model, train_loader, one_hot, mask, validation, labels, device, args.epochs, args.patience, args.learning_rate, args.batch_size)
            model.load_state_dict(state)
            probability = predict(model, one_hot, mask, test, device, args.batch_size)
            peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            for subset_name, selected in (("all", np.arange(len(test))), ("matched", matched)):
                seed_metrics.append({"seed": seed, "implementation": name, "subset": subset_name, "n": len(selected), "epochs": len(history), "training_seconds": elapsed, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "peak_gpu_bytes": peak_memory, **metric_row(labels[test][selected], probability[selected])})
            for local, index in enumerate(test):
                props = sequence_properties(sequences[index])
                predictions.append({"seed": seed, "implementation": name, "sequence_sha256": sequence_sha256(sequences[index]), "label": int(labels[index]), "probability": float(probability[local]), "nearest_train_identity": float(nearest_identity[local]), **props})

            strata = {
                "length:<12": np.asarray([len(row) < 12 for row in test_sequences]),
                "length:12-15": np.asarray([12 <= len(row) <= 15 for row in test_sequences]),
                "length:16-19": np.asarray([16 <= len(row) <= 19 for row in test_sequences]),
                "length:20-23": np.asarray([20 <= len(row) <= 23 for row in test_sequences]),
                "length:>=24": np.asarray([len(row) >= 24 for row in test_sequences]),
                "identity:<0.4": nearest_identity < 0.4,
                "identity:0.4-0.6": (nearest_identity >= 0.4) & (nearest_identity < 0.6),
                "identity:0.6-0.8": (nearest_identity >= 0.6) & (nearest_identity < 0.8),
            }
            charges = property_x[test, property_names.index("net_charge_proxy")]
            strata.update({
                "charge:<=0": charges <= 0,
                "charge:0-2": (charges > 0) & (charges <= 2),
                "charge:2-5": (charges > 2) & (charges <= 5),
                "charge:>5": charges > 5,
            })
            for stratum, selected_mask in strata.items():
                if selected_mask.sum() >= 10:
                    stratified.append({"seed": seed, "implementation": name, "stratum": stratum, "n": int(selected_mask.sum()), "positive_n": int(labels[test][selected_mask].sum()), **metric_row(labels[test][selected_mask], probability[selected_mask])})

            positive_local = np.where(labels[test] == 1)[0]
            positives = [test_sequences[index] for index in positive_local]
            shuffled = shuffled_sequences(positives, seed)
            shuffled_probability = score_sequences(model, shuffled, device, args.batch_size)
            original_probability = probability[positive_local]
            for label_name, selected in (("all", np.arange(len(test))), ("amp", positive_local), ("non_amp", np.where(labels[test] == 0)[0])):
                rho = spearmanr(np.asarray([len(test_sequences[index]) for index in selected]), probability[selected]).statistic
                shuffle_rows.append({"seed": seed, "implementation": name, "analysis": "score_length_correlation", "subset": label_name, "n": len(selected), "value": float(rho)})
            deltas = original_probability - shuffled_probability
            shuffle_rows.extend([
                {"seed": seed, "implementation": name, "analysis": "original_minus_shuffled", "subset": "amp", "n": len(deltas), "value": float(np.mean(deltas)), "statistic": "mean"},
                {"seed": seed, "implementation": name, "analysis": "original_minus_shuffled", "subset": "amp", "n": len(deltas), "value": float(np.median(deltas)), "statistic": "median"},
                {"seed": seed, "implementation": name, "analysis": "decision_flip_fraction", "subset": "amp", "n": len(deltas), "value": float(np.mean((original_probability >= 0.5) != (shuffled_probability >= 0.5)))},
            ])
            checkpoint_dir = args.output_dir / "checkpoints" / name / f"seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"architecture": name, "seed": seed, "max_length": 30, "state_dict": state, "best_validation_balanced_accuracy": max(row["validation_balanced_accuracy"] for row in history)}, checkpoint_dir / "classifier.pt")
            (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    write_csv(args.output_dir / "seed_metrics.csv", seed_metrics)
    write_csv(args.output_dir / "test_predictions.csv", predictions)
    write_csv(args.output_dir / "stratified_metrics.csv", stratified)
    write_csv(args.output_dir / "shuffle_sensitivity.csv", shuffle_rows)
    write_csv(args.output_dir / "split_manifest.csv", split_rows)
    manifest = {
        "status": "complete",
        "terminology": ["LIMP-DI-legacy", "LIMP-DI-corrected"],
        "inputs": {str(args.amp_fasta): sha256_file(args.amp_fasta), str(args.non_amp_fasta): sha256_file(args.non_amp_fasta)},
        "seeds": args.seeds,
        "identity_threshold": args.identity_threshold,
        "records_after_dedup_and_conflict_quarantine": len(records),
        "conflicting_sequences": len(conflicts),
        "clusters": len(set(groups.tolist())),
        "device": str(device),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
