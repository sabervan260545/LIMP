"""Deterministic FASTA preparation and similarity-aware splitting utilities.

The historical notebook mixed source datasets and performed a random split.
This module keeps provenance, produces globally unique IDs, and clusters short
peptides by same-length ungapped identity before any split is made.  The identity
metric is deliberately explicit: it is fast and deterministic for the supplied
4--30 aa dataset.  A future alignment-based clustering backend can replace it
without changing the manifest schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class SequenceRecord:
    """One canonical peptide with immutable provenance fields."""

    sequence_id: str
    sequence: str
    label: str
    source: str
    source_record_id: str
    original_index: int
    cluster_id: int | None = None
    split: str | None = None


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_boundary(ids: Sequence[str]) -> int | None:
    """Return the first reset boundary in an otherwise sequential FASTA ID list."""

    first_seen: set[str] = set()
    for index, record_id in enumerate(ids):
        if record_id in first_seen:
            return index
        first_seen.add(record_id)
    return None


def read_amp_fasta(path: Path, label: str = "amp") -> list[SequenceRecord]:
    """Read, validate, and provenance-tag the supplied AMP FASTA.

    The current input repeats ``seq_0`` when the original main dataset begins.
    Records before that reset are tagged ``independent`` and records after it
    ``main``.  If a future file has globally unique IDs, every record is tagged
    ``unspecified`` rather than silently inventing provenance.
    """

    parsed = list(SeqIO.parse(str(path), "fasta"))
    if not parsed:
        raise ValueError(f"No FASTA records found in {path}")

    ids = [record.id for record in parsed]
    boundary = _source_boundary(ids)
    records: list[SequenceRecord] = []
    for index, fasta_record in enumerate(parsed):
        sequence = str(fasta_record.seq).upper().strip()
        invalid = sorted(set(sequence).difference(AMINO_ACIDS))
        if invalid:
            raise ValueError(
                f"{path.name} record {fasta_record.id!r} contains invalid residues: {invalid}"
            )
        source = "unspecified"
        if boundary is not None:
            source = "independent" if index < boundary else "main"
        sequence_id = f"{source}_{label}_{index:06d}"
        records.append(
            SequenceRecord(
                sequence_id=sequence_id,
                sequence=sequence,
                label=label,
                source=source,
                source_record_id=fasta_record.id,
                original_index=index,
            )
        )
    return records


def remove_exact_duplicates(records: Iterable[SequenceRecord]) -> tuple[list[SequenceRecord], int]:
    """Keep first occurrence of each sequence and return the number discarded."""

    seen: set[str] = set()
    unique: list[SequenceRecord] = []
    discarded = 0
    for record in records:
        if record.sequence not in seen:
            unique.append(record)
            seen.add(record.sequence)
        else:
            discarded += 1
    return unique, discarded


def cluster_same_length_identity(
    records: Sequence[SequenceRecord], identity_threshold: float = 0.80, batch_size: int = 256
) -> list[SequenceRecord]:
    """Cluster peptide records connected by same-length ungapped identity.

    A pair is connected when its matching-residue fraction is at least
    ``identity_threshold``.  Connected components, rather than only direct
    pairs, become clusters.  This prevents a chain of close variants from being
    split between training and validation.
    """

    if not 0 < identity_threshold <= 1:
        raise ValueError("identity_threshold must be in (0, 1]")
    union_find = _UnionFind(len(records))
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_length[len(record.sequence)].append(index)

    for length, global_indices in by_length.items():
        if len(global_indices) < 2:
            continue
        encoded = np.asarray(
            [[AA_TO_INDEX[residue] for residue in records[index].sequence] for index in global_indices],
            dtype=np.uint8,
        )
        count = len(global_indices)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            identity = (encoded[start:stop, None, :] == encoded[None, :, :]).mean(axis=2)
            row_indices, column_indices = np.where(identity >= identity_threshold)
            for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
                left = start + row_index
                if left < column_index:
                    union_find.union(global_indices[left], global_indices[column_index])

    roots = [union_find.find(index) for index in range(len(records))]
    root_to_cluster: dict[int, int] = {}
    cluster_ids: list[int] = []
    for root in roots:
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_ids.append(root_to_cluster[root])
    return [replace(record, cluster_id=cluster_id) for record, cluster_id in zip(records, cluster_ids)]


def split_by_cluster(
    records: Sequence[SequenceRecord],
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> list[SequenceRecord]:
    """Assign whole clusters to train/validation/test with deterministic seeds."""

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    if any(record.cluster_id is None for record in records):
        raise ValueError("Records must be clustered before splitting")

    groups = np.asarray([record.cluster_id for record in records])
    indices = np.arange(len(records))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_validation_indices, test_indices = next(outer.split(indices, groups=groups))

    remaining_groups = groups[train_validation_indices]
    inner_fraction = validation_fraction / (1.0 - test_fraction)
    inner = GroupShuffleSplit(n_splits=1, test_size=inner_fraction, random_state=seed + 1)
    train_relative, validation_relative = next(
        inner.split(train_validation_indices, groups=remaining_groups)
    )
    train_indices = train_validation_indices[train_relative]
    validation_indices = train_validation_indices[validation_relative]

    assignments: dict[int, str] = {}
    assignments.update({index: "train" for index in train_indices.tolist()})
    assignments.update({index: "validation" for index in validation_indices.tolist()})
    assignments.update({index: "test" for index in test_indices.tolist()})
    split_records = [replace(record, split=assignments[index]) for index, record in enumerate(records)]
    assert_cluster_disjoint(split_records)
    return split_records


def assert_cluster_disjoint(records: Sequence[SequenceRecord]) -> None:
    """Raise when a cluster appears in more than one assigned split."""

    cluster_splits: dict[int, set[str]] = defaultdict(set)
    for record in records:
        if record.cluster_id is None or record.split is None:
            raise ValueError("Cluster and split must be assigned")
        cluster_splits[record.cluster_id].add(record.split)
    leaked = [cluster_id for cluster_id, splits in cluster_splits.items() if len(splits) != 1]
    if leaked:
        raise AssertionError(f"Clusters leaked across splits: {leaked[:10]}")


def _records_stats(records: Sequence[SequenceRecord]) -> dict[str, object]:
    lengths = np.asarray([len(record.sequence) for record in records], dtype=float)
    return {
        "count": len(records),
        "unique_sequences": len({record.sequence for record in records}),
        "length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        },
        "sources": dict(Counter(record.source for record in records)),
        "clusters": len({record.cluster_id for record in records}),
        "splits": dict(Counter(record.split for record in records)),
    }


def write_generator_dataset(
    records: Sequence[SequenceRecord],
    output_dir: Path,
    input_fasta: Path,
    identity_threshold: float,
    seed: int,
) -> dict[str, object]:
    """Persist a versioned manifest, split FASTAs, and auditable metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    for split in ("train", "validation", "test"):
        split_path = output_dir / f"{split}.fasta"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.split == split:
                    handle.write(f">{record.sequence_id}\n{record.sequence}\n")

    metadata = {
        "input_fasta": str(input_fasta.resolve()),
        "input_sha256": sha256_file(input_fasta),
        "identity_metric": "same_length_ungapped_identity",
        "identity_threshold": identity_threshold,
        "split_seed": seed,
        "records": _records_stats(records),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
