"""Compact autoregressive language model for AMP-only sequence generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .data import AMINO_ACIDS


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
AA_OFFSET = 3
VOCAB_SIZE = AA_OFFSET + len(AMINO_ACIDS)
AA_TO_TOKEN = {aa: AA_OFFSET + index for index, aa in enumerate(AMINO_ACIDS)}
TOKEN_TO_AA = {token: aa for aa, token in AA_TO_TOKEN.items()}


def encode_sequence(sequence: str) -> list[int]:
    return [BOS_ID] + [AA_TO_TOKEN[residue] for residue in sequence] + [EOS_ID]


def decode_tokens(tokens: Sequence[int]) -> str:
    residues: list[str] = []
    for token in tokens:
        if token == EOS_ID:
            break
        if token in TOKEN_TO_AA:
            residues.append(TOKEN_TO_AA[token])
    return "".join(residues)


def collate_autoregressive(batch: Sequence[list[int]]) -> tuple[Tensor, Tensor]:
    """Create padded teacher-forcing inputs and targets from encoded sequences."""

    width = max(len(item) - 1 for item in batch)
    inputs = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
    targets = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
    for row, tokens in enumerate(batch):
        inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1], dtype=torch.long)
        targets[row, : len(tokens) - 1] = torch.tensor(tokens[1:], dtype=torch.long)
    return inputs, targets


class AMPAutoregressiveModel(nn.Module):
    """A small causal Transformer suitable for short peptide inference."""

    def __init__(
        self,
        max_tokens: int = 32,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.max_tokens = max_tokens
        self.d_model = d_model
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, token_ids: Tensor) -> Tensor:
        batch, length = token_ids.shape
        if length > self.max_tokens:
            raise ValueError(f"Input length {length} exceeds model limit {self.max_tokens}")
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0).expand(batch, length)
        hidden = self.token_embedding(token_ids) * math.sqrt(self.d_model) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((length, length), device=token_ids.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = token_ids.eq(PAD_ID)
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output(self.final_norm(hidden))

    def config_dict(self) -> dict[str, int | float]:
        first_layer = self.transformer.layers[0]
        return {
            "max_tokens": self.max_tokens,
            "d_model": self.d_model,
            "num_layers": len(self.transformer.layers),
            "num_heads": first_layer.self_attn.num_heads,
            "feedforward_dim": first_layer.linear1.out_features,
            "dropout": float(first_layer.dropout.p),
        }


def _top_k_top_p_filter(logits: Tensor, top_k: int, top_p: float) -> Tensor:
    filtered = logits.clone()
    if top_k > 0:
        top_k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, -torch.inf))
    return filtered


@dataclass(frozen=True)
class GeneratedSequence:
    sequence: str
    log_probability: float


@torch.no_grad()
def sample_sequences(
    model: AMPAutoregressiveModel,
    count: int,
    device: torch.device,
    min_length: int = 8,
    max_length: int = 30,
    temperature: float = 1.3,
    top_k: int = 0,
    top_p: float = 0.90,
    seed: int = 42,
) -> list[GeneratedSequence]:
    """Sample an exactly-sized batch without calling any external predictor."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not 1 <= min_length <= max_length < model.max_tokens:
        raise ValueError("Requested lengths must fit the model token limit")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    token_ids = torch.full((count, 1), BOS_ID, device=device, dtype=torch.long)
    finished = torch.zeros(count, device=device, dtype=torch.bool)
    log_probability = torch.zeros(count, device=device)

    for produced_length in range(max_length + 1):
        logits = model(token_ids)[:, -1, :] / temperature
        logits[:, PAD_ID] = -torch.inf
        logits[:, BOS_ID] = -torch.inf
        if produced_length < min_length:
            logits[:, EOS_ID] = -torch.inf
        if produced_length >= max_length:
            logits.fill_(-torch.inf)
            logits[:, EOS_ID] = 0.0
        logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        log_probs = torch.log_softmax(logits, dim=-1)
        next_ids = torch.multinomial(log_probs.exp(), num_samples=1, generator=rng).squeeze(1)
        active = ~finished
        log_probability = log_probability + torch.where(active, log_probs.gather(1, next_ids[:, None]).squeeze(1), 0.0)
        finished = finished | next_ids.eq(EOS_ID)
        next_ids = torch.where(finished, next_ids, next_ids)
        token_ids = torch.cat((token_ids, next_ids[:, None]), dim=1)
        if bool(finished.all()):
            break

    sequences = [decode_tokens(row.tolist()[1:]) for row in token_ids.cpu()]
    return [GeneratedSequence(sequence, float(score)) for sequence, score in zip(sequences, log_probability.cpu())]
