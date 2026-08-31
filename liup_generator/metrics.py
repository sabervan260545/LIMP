"""Intrinsic generator metrics; no downstream property predictor is called here."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .data import AMINO_ACIDS


@dataclass(frozen=True)
class GenerationMetrics:
    count: int
    validity_rate: float
    length_compliance_rate: float
    unique_rate: float
    exact_train_overlap_rate: float
    mean_max_train_identity: float
    median_max_train_identity: float
    amino_acid_l1_distance: float
    homopolymer_run_ge4_rate: float
    max_homopolymer_run: int
    training_homopolymer_run_ge4_rate: float


def _max_homopolymer_run(sequence: str) -> int:
    if not sequence:
        return 0
    maximum, current = 1, 1
    for previous, residue in zip(sequence, sequence[1:]):
        current = current + 1 if residue == previous else 1
        maximum = max(maximum, current)
    return maximum


def _composition(sequences: Sequence[str]) -> np.ndarray:
    counts = Counter("".join(sequences))
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(AMINO_ACIDS), dtype=float)
    return np.asarray([counts[aa] / total for aa in AMINO_ACIDS], dtype=float)


def _max_same_length_identity(reference: Sequence[str], generated: Sequence[str]) -> np.ndarray:
    by_length: dict[int, list[str]] = {}
    for sequence in reference:
        by_length.setdefault(len(sequence), []).append(sequence)
    values: list[float] = []
    for sequence in generated:
        candidates = by_length.get(len(sequence), [])
        if not candidates:
            values.append(0.0)
            continue
        values.append(max(sum(left == right for left, right in zip(sequence, candidate)) / len(sequence) for candidate in candidates))
    return np.asarray(values, dtype=float)


def evaluate_generation(
    generated: Sequence[str],
    training_sequences: Sequence[str],
    min_length: int,
    max_length: int,
) -> GenerationMetrics:
    if not generated:
        raise ValueError("Cannot evaluate an empty generated sequence collection")
    valid = [set(sequence).issubset(set(AMINO_ACIDS)) and len(sequence) > 0 for sequence in generated]
    length_ok = [min_length <= len(sequence) <= max_length for sequence in generated]
    max_identity = _max_same_length_identity(training_sequences, generated)
    train_set = set(training_sequences)
    generated_runs = np.asarray([_max_homopolymer_run(sequence) for sequence in generated])
    training_runs = np.asarray([_max_homopolymer_run(sequence) for sequence in training_sequences])
    return GenerationMetrics(
        count=len(generated),
        validity_rate=float(np.mean(valid)),
        length_compliance_rate=float(np.mean(length_ok)),
        unique_rate=len(set(generated)) / len(generated),
        exact_train_overlap_rate=float(np.mean([sequence in train_set for sequence in generated])),
        mean_max_train_identity=float(max_identity.mean()),
        median_max_train_identity=float(np.median(max_identity)),
        amino_acid_l1_distance=float(np.abs(_composition(generated) - _composition(training_sequences)).sum()),
        homopolymer_run_ge4_rate=float(np.mean(generated_runs >= 4)),
        max_homopolymer_run=int(generated_runs.max()),
        training_homopolymer_run_ge4_rate=float(np.mean(training_runs >= 4)),
    )


def metrics_dict(metrics: GenerationMetrics) -> dict[str, float | int]:
    return asdict(metrics)
