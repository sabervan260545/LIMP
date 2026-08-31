"""Shared, deterministic metrics for LIMP completion experiments."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from .data import AMINO_ACIDS, AA_TO_INDEX


KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
HYDROPHOBIC = frozenset("AILMFWVYC")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def net_charge_proxy(sequence: str) -> float:
    return float(sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H") - sequence.count("D") - sequence.count("E"))


def mean_hydropathy(sequence: str) -> float:
    return float(np.mean([KYTE_DOOLITTLE[residue] for residue in sequence]))


def sequence_properties(sequence: str) -> dict[str, float | int]:
    counts = Counter(sequence)
    row: dict[str, float | int] = {
        "length": len(sequence),
        "net_charge_proxy": net_charge_proxy(sequence),
        "mean_hydropathy": mean_hydropathy(sequence),
        "hydrophobic_fraction": sum(residue in HYDROPHOBIC for residue in sequence) / len(sequence),
    }
    row.update({f"fraction_{aa}": counts[aa] / len(sequence) for aa in AMINO_ACIDS})
    return row


def property_matrix(sequences: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    names = ["length", "net_charge_proxy", "mean_hydropathy", "hydrophobic_fraction"] + [
        f"fraction_{aa}" for aa in AMINO_ACIDS
    ]
    rows = [sequence_properties(sequence) for sequence in sequences]
    return np.asarray([[float(row[name]) for name in names] for row in rows]), names


def max_same_length_identity(reference: Sequence[str], query: Sequence[str], batch_size: int = 256) -> np.ndarray:
    by_length: dict[int, list[str]] = defaultdict(list)
    for sequence in reference:
        by_length[len(sequence)].append(sequence)
    result = np.zeros(len(query), dtype=np.float32)
    query_by_length: dict[int, list[int]] = defaultdict(list)
    for index, sequence in enumerate(query):
        query_by_length[len(sequence)].append(index)
    for length, indices in query_by_length.items():
        candidates = by_length.get(length)
        if not candidates:
            continue
        reference_array = np.asarray([[AA_TO_INDEX[aa] for aa in sequence] for sequence in candidates], dtype=np.uint8)
        query_array = np.asarray([[AA_TO_INDEX[aa] for aa in query[index]] for index in indices], dtype=np.uint8)
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            identity = (query_array[start:stop, None, :] == reference_array[None, :, :]).mean(axis=2)
            result[np.asarray(indices[start:stop])] = identity.max(axis=1)
    return result


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def identity_cluster_stats(sequences: Sequence[str], threshold: float = 0.80, batch_size: int = 256) -> dict[str, float | int]:
    union = _UnionFind(len(sequences))
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        by_length[len(sequence)].append(index)
    for indices in by_length.values():
        if len(indices) < 2:
            continue
        encoded = np.asarray([[AA_TO_INDEX[aa] for aa in sequences[index]] for index in indices], dtype=np.uint8)
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            identities = (encoded[start:stop, None, :] == encoded[None, :, :]).mean(axis=2)
            rows, columns = np.where(identities >= threshold)
            for row, column in zip(rows.tolist(), columns.tolist()):
                left = start + row
                if left < column:
                    union.union(indices[left], indices[column])
    sizes = Counter(union.find(index) for index in range(len(sequences)))
    return {
        "cluster_count": len(sizes),
        "largest_cluster_size": max(sizes.values()),
        "largest_cluster_fraction": max(sizes.values()) / len(sequences),
    }


def sampled_pairwise_identity(sequences: Sequence[str], pairs: int = 100000, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    values: list[float] = []
    by_length: dict[int, list[str]] = defaultdict(list)
    for sequence in sequences:
        by_length[len(sequence)].append(sequence)
    eligible = [length for length, rows in by_length.items() if len(rows) >= 2]
    weights = [len(by_length[length]) * (len(by_length[length]) - 1) for length in eligible]
    if not eligible:
        return {"pairwise_identity_mean": float("nan"), "pairwise_identity_median": float("nan")}
    for _ in range(pairs):
        length = rng.choices(eligible, weights=weights, k=1)[0]
        first, second = rng.sample(by_length[length], 2)
        values.append(sum(a == b for a, b in zip(first, second)) / length)
    return {
        "pairwise_identity_mean": float(np.mean(values)),
        "pairwise_identity_median": float(np.median(values)),
    }
