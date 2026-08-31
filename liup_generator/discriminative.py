"""Corrected LIMP discriminative inverse-design implementation.

LIMP-DI-corrected preserves the historical classifier capacity while repairing
two implementation defects: padded readout states are excluded from pooling,
and inverse design optimizes only the discriminative objective instead of a
post-Softmax L1 term that is mathematically constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .data import AMINO_ACIDS
from .legacy import KerasStyleMultiHeadAttention, LegacyAttentionStage


class CorrectedReadout(nn.Module):
    """Attention readout with an explicit valid-position mask after the FFN."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        self.attention = KerasStyleMultiHeadAttention(dimension, num_heads=4, head_dim=dimension)
        self.layernorm_1 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False),
            nn.LeakyReLU(),
            nn.Linear(dimension, dimension, bias=False),
        )
        self.layernorm_2 = nn.LayerNorm(dimension)

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        attention_output = self.attention(inputs, valid_mask)
        projected = self.layernorm_1(inputs + attention_output)
        projected = self.layernorm_2(projected + self.ffn(projected))
        valid = valid_mask.unsqueeze(-1).to(projected.dtype)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        return (projected * valid).sum(dim=1) / denominator


class CorrectedLIUPClassifier(nn.Module):
    """LIMP-DI classifier with corrected masked pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.stage_1 = LegacyAttentionStage(20, 256)
        self.stage_2 = LegacyAttentionStage(256, 128)
        self.stage_3 = LegacyAttentionStage(128, 32)
        self.readout = CorrectedReadout(32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, one_hot: Tensor, valid_mask: Tensor) -> Tensor:
        outputs = self.stage_1(one_hot, valid_mask)
        outputs = self.stage_2(outputs, valid_mask)
        outputs = self.stage_3(outputs, valid_mask)
        return self.classifier(self.readout(outputs, valid_mask)).squeeze(-1)


@dataclass(frozen=True)
class DiscriminativeGeneration:
    sequence: str
    self_score: float
    requested_length: int


def generate_with_corrected_gradient(
    model: CorrectedLIUPClassifier,
    count: int,
    device: torch.device,
    min_length: int = 12,
    max_length: int = 28,
    iterations: int = 500,
    temperature: float = 0.3,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> list[DiscriminativeGeneration]:
    """Optimize continuous residue logits against the corrected classifier.

    The classification loss is summed across sequences so each independent
    candidate receives the same gradient scale regardless of batch size.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if not 1 <= min_length <= max_length:
        raise ValueError("Expected 1 <= min_length <= max_length")
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    width = max_length
    requested_lengths = torch.randint(
        low=min_length,
        high=max_length + 1,
        size=(count,),
        device=device,
        generator=generator,
    )
    valid_mask = torch.arange(width, device=device)[None, :] < requested_lengths[:, None]
    logits = torch.randn(
        (count, width, len(AMINO_ACIDS)),
        device=device,
        generator=generator,
        requires_grad=True,
    )

    for _ in range(iterations):
        relaxed = torch.softmax(logits / temperature, dim=-1)
        probability = torch.sigmoid(model(relaxed, valid_mask))
        objective = -torch.log(probability.clamp_min(1e-8))
        gradient = torch.autograd.grad(objective.sum(), logits, only_inputs=True)[0]
        with torch.no_grad():
            logits.sub_(learning_rate * gradient)

    with torch.no_grad():
        indices = torch.softmax(logits / temperature, dim=-1).argmax(dim=-1)
        discrete = torch.nn.functional.one_hot(indices, num_classes=len(AMINO_ACIDS)).to(torch.float32)
        scores = torch.sigmoid(model(discrete, valid_mask)).cpu().tolist()

    rows: list[DiscriminativeGeneration] = []
    for row, length in enumerate(requested_lengths.cpu().tolist()):
        sequence = "".join(AMINO_ACIDS[index] for index in indices[row, :length].cpu().tolist())
        rows.append(DiscriminativeGeneration(sequence, float(scores[row]), int(length)))
    return rows


@torch.no_grad()
def score_discriminative_sequences(
    model: CorrectedLIUPClassifier,
    sequences: Sequence[str],
    device: torch.device,
    max_length: int = 30,
) -> Tensor:
    from .legacy import encode_one_hot

    one_hot, mask = encode_one_hot(sequences, max_length=max_length, device=device)
    return torch.sigmoid(model(one_hot, mask))
