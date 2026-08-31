from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.data import SequenceRecord, assert_cluster_disjoint, cluster_same_length_identity, split_by_cluster


def _record(index: int, sequence: str) -> SequenceRecord:
    return SequenceRecord(
        sequence_id=f"amp_{index}",
        sequence=sequence,
        label="amp",
        source="test",
        source_record_id=f"seq_{index}",
        original_index=index,
    )


def test_close_sequences_share_cluster_and_split() -> None:
    records = [
        _record(0, "KKKKAAAAAA"),
        _record(1, "KKKKAAAAAT"),  # 90% same-length identity
        _record(2, "VVVVVVVVVV"),
        _record(3, "LLLLLLLLLL"),
        _record(4, "RRRRRRRRRR"),
    ]
    clustered = cluster_same_length_identity(records, identity_threshold=0.8)
    assert clustered[0].cluster_id == clustered[1].cluster_id
    split = split_by_cluster(clustered, validation_fraction=0.2, test_fraction=0.2, seed=7)
    assert split[0].split == split[1].split
    assert_cluster_disjoint(split)
