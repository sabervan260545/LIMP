"""Data-gate benchmark: legacy random split versus similarity-aware split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import AMINO_ACIDS, SequenceRecord, cluster_same_length_identity


@dataclass(frozen=True)
class SplitBenchmark:
    name: str
    train_count: int
    test_count: int
    accuracy: float
    balanced_accuracy: float
    auroc: float
    auprc: float
    mcc: float
    near_neighbor_rate: float


def composition_features(sequences: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[len(sequence)] + [sequence.count(aa) / len(sequence) for aa in AMINO_ACIDS] for sequence in sequences],
        dtype=np.float32,
    )


def _max_same_length_identity(train: Sequence[str], test: Sequence[str]) -> np.ndarray:
    """Return each test sequence's max train identity under the data-gate metric."""

    lookup: dict[int, np.ndarray] = {}
    aa_to_index = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
    for length in range(1, max(map(len, train + test)) + 1):
        matching = [sequence for sequence in train if len(sequence) == length]
        if matching:
            lookup[length] = np.asarray(
                [[aa_to_index[aa] for aa in sequence] for sequence in matching], dtype=np.uint8
            )

    maximums: list[float] = []
    for length in range(1, max(map(len, test)) + 1):
        matching = [sequence for sequence in test if len(sequence) == length]
        if not matching:
            continue
        train_encoded = lookup.get(length)
        if train_encoded is None:
            maximums.extend([0.0] * len(matching))
            continue
        test_encoded = np.asarray(
            [[aa_to_index[aa] for aa in sequence] for sequence in matching], dtype=np.uint8
        )
        for start in range(0, len(test_encoded), 256):
            identity = (test_encoded[start : start + 256, None, :] == train_encoded[None, :, :]).mean(axis=2)
            maximums.extend(identity.max(axis=1).tolist())
    return np.asarray(maximums, dtype=float)


def _evaluate(name: str, sequences: Sequence[str], labels: np.ndarray, train: np.ndarray, test: np.ndarray, threshold: float) -> SplitBenchmark:
    features = composition_features(sequences)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)
    )
    model.fit(features[train], labels[train])
    probability = model.predict_proba(features[test])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    identity = _max_same_length_identity(
        [sequences[index] for index in train], [sequences[index] for index in test]
    )
    return SplitBenchmark(
        name=name,
        train_count=len(train),
        test_count=len(test),
        accuracy=float(accuracy_score(labels[test], prediction)),
        balanced_accuracy=float(balanced_accuracy_score(labels[test], prediction)),
        auroc=float(roc_auc_score(labels[test], probability)),
        auprc=float(average_precision_score(labels[test], probability)),
        mcc=float(matthews_corrcoef(labels[test], prediction)),
        near_neighbor_rate=float((identity >= threshold).mean()),
    )


def compare_random_and_cluster_splits(
    amp_records: Sequence[SequenceRecord],
    non_amp_records: Sequence[SequenceRecord],
    identity_threshold: float = 0.80,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[SplitBenchmark, SplitBenchmark]:
    records = list(amp_records) + list(non_amp_records)
    sequences = [record.sequence for record in records]
    labels = np.asarray([1] * len(amp_records) + [0] * len(non_amp_records), dtype=int)
    indices = np.arange(len(records))

    random_split = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    random_train, random_test = next(random_split.split(indices, labels))
    random_result = _evaluate("legacy_random", sequences, labels, random_train, random_test, identity_threshold)

    clustered = cluster_same_length_identity(records, identity_threshold=identity_threshold)
    groups = np.asarray([record.cluster_id for record in clustered])
    grouped_split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    grouped_train, grouped_test = next(grouped_split.split(indices, labels, groups=groups))
    grouped_result = _evaluate("similarity_aware", sequences, labels, grouped_train, grouped_test, identity_threshold)
    return random_result, grouped_result
