from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.autoregressive import AMPAutoregressiveModel, decode_tokens, encode_sequence, sample_sequences


def test_encode_decode_round_trip() -> None:
    sequence = "GIGKFLHAAKKFAKAF"
    assert decode_tokens(encode_sequence(sequence)[1:]) == sequence


def test_sampling_is_seed_reproducible_and_length_bounded() -> None:
    torch.manual_seed(1)
    model = AMPAutoregressiveModel(num_layers=1, d_model=32, num_heads=4, feedforward_dim=64, dropout=0.0)
    first = sample_sequences(model, count=4, device=torch.device("cpu"), min_length=8, max_length=12, seed=9)
    second = sample_sequences(model, count=4, device=torch.device("cpu"), min_length=8, max_length=12, seed=9)
    assert first == second
    assert all(8 <= len(item.sequence) <= 12 for item in first)
