"""PyTorch reproduction of the historical LIMP-DI classifier-guided generator.

This is a benchmark implementation, not the production design.  It preserves
the consequential choices in ``gen_20250108.ipynb``: one-hot input, three
attention stages, Keras-style per-head dimensions, random-length input-logit
optimisation, continuous Softmax relaxation, and the historical L1 term.  The
last two choices are deliberately retained here so later generator upgrades are
compared with a real legacy baseline rather than a repaired version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .data import AMINO_ACIDS


def sine_position_encoding(length: int, dimension: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros((length, dimension), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class KerasStyleMultiHeadAttention(nn.Module):
    """Attention matching Keras ``MultiHeadAttention(num_heads, key_dim)``.

    In the notebook ``key_dim`` was supplied as 256/128/32, i.e. it is the
    per-head width rather than the total representation width.  This class keeps
    that historical parameterisation for an honest baseline.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.query = nn.Linear(embed_dim, inner_dim)
        self.key = nn.Linear(embed_dim, inner_dim)
        self.value = nn.Linear(embed_dim, inner_dim)
        self.output = nn.Linear(inner_dim, embed_dim)

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        batch, length, _ = inputs.shape
        def project(layer: nn.Linear) -> Tensor:
            return layer(inputs).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = project(self.query), project(self.key), project(self.value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output(attended)


class LegacyAttentionStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU())
        self.attention = KerasStyleMultiHeadAttention(output_dim, num_heads=4, head_dim=output_dim)

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        projected = self.projection(inputs)
        projected = projected + sine_position_encoding(
            projected.shape[1], projected.shape[2], projected.device, projected.dtype
        )
        attended = self.attention(projected, valid_mask)
        return attended * valid_mask.unsqueeze(-1).to(attended.dtype)


class LegacyReadout(nn.Module):
    """Historical readout including its missing final output mask."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        self.attention = KerasStyleMultiHeadAttention(dimension, num_heads=4, head_dim=dimension)
        self.layernorm_1 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False), nn.LeakyReLU(), nn.Linear(dimension, dimension, bias=False)
        )
        self.layernorm_2 = nn.LayerNorm(dimension)

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        attention_output = self.attention(inputs, valid_mask)
        projected = self.layernorm_1(inputs + attention_output)
        projected = self.layernorm_2(projected + self.ffn(projected))
        # This intentionally reproduces the legacy notebook bug: padded readout
        # states are summed but the denominator is only the valid length.
        denominator = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(projected.dtype)
        return projected.sum(dim=1) / denominator


class LegacyLIUPClassifier(nn.Module):
    """Torch baseline structurally equivalent to the notebook's classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.stage_1 = LegacyAttentionStage(20, 256)
        self.stage_2 = LegacyAttentionStage(256, 128)
        self.stage_3 = LegacyAttentionStage(128, 32)
        self.readout = LegacyReadout(32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, one_hot: Tensor, valid_mask: Tensor) -> Tensor:
        outputs = self.stage_1(one_hot, valid_mask)
        outputs = self.stage_2(outputs, valid_mask)
        outputs = self.stage_3(outputs, valid_mask)
        return self.classifier(self.readout(outputs, valid_mask)).squeeze(-1)


def encode_one_hot(sequences: Sequence[str], max_length: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """Encode canonical peptide strings into one-hot tensors and boolean masks."""

    index = {residue: position for position, residue in enumerate(AMINO_ACIDS)}
    array = torch.zeros((len(sequences), max_length, len(AMINO_ACIDS)), dtype=torch.float32)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        if len(sequence) > max_length:
            raise ValueError(f"Sequence length {len(sequence)} exceeds max_length {max_length}")
        for column, residue in enumerate(sequence):
            array[row, column, index[residue]] = 1.0
            mask[row, column] = True
    if device is not None:
        array, mask = array.to(device), mask.to(device)
    return array, mask


@dataclass(frozen=True)
class LegacyGeneration:
    sequence: str
    self_score: float
    requested_length: int


@torch.no_grad()
def score_legacy_sequences(model: LegacyLIUPClassifier, sequences: Sequence[str], device: torch.device) -> Tensor:
    one_hot, mask = encode_one_hot(sequences, max(map(len, sequences)), device=device)
    return torch.sigmoid(model(one_hot, mask))


def generate_with_legacy_gradient(
    model: LegacyLIUPClassifier,
    count: int,
    device: torch.device,
    min_length: int = 10,
    max_length: int = 30,
    iterations: int = 500,
    temperature: float = 0.3,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> list[LegacyGeneration]:
    """Run the historical continuous gradient input optimisation once per batch."""

    if not 1 <= min_length < max_length:
        raise ValueError("Expected 1 <= min_length < max_length")
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    requested_lengths = torch.randint(
        low=min_length,
        high=max_length,
        size=(count,),
        device=device,
        generator=generator,
    )
    valid_mask = torch.arange(max_length, device=device)[None, :] < requested_lengths[:, None]
    logits = torch.randn((count, max_length, len(AMINO_ACIDS)), device=device, generator=generator, requires_grad=True)

    for _ in range(iterations):
        relaxed = torch.softmax(logits / temperature, dim=-1)
        probability = torch.sigmoid(model(relaxed, valid_mask))
        # Kept exactly for legacy reproduction: after Softmax this L1 term is
        # constant, and broadcasting repeats the classification term by length.
        objective = -torch.log(probability.clamp_min(1e-8)).unsqueeze(1) + 0.1 * relaxed.abs().sum(dim=-1)
        gradient = torch.autograd.grad(objective.sum(), logits, only_inputs=True)[0]
        with torch.no_grad():
            logits.sub_(learning_rate * gradient)

    with torch.no_grad():
        relaxed = torch.softmax(logits / temperature, dim=-1)
        indices = relaxed.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=len(AMINO_ACIDS)).to(torch.float32)
        self_scores = torch.sigmoid(model(one_hot, valid_mask)).cpu().tolist()

    generated: list[LegacyGeneration] = []
    for row, length in enumerate(requested_lengths.cpu().tolist()):
        sequence = "".join(AMINO_ACIDS[index] for index in indices[row, :length].cpu().tolist())
        generated.append(LegacyGeneration(sequence, float(self_scores[row]), int(length)))
    return generated
