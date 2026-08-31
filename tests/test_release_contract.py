import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_formal_checkpoint_contract() -> None:
    path = ROOT / "checkpoint/LIMP-AR_generator.pt"
    assert _sha256(path) == "c25b229bd85f321926d83c0976d4c13372bff6f23e5f645a6535a1f6074562d8"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["architecture"] == "liup_amp_autoregressive_transformer_v2"
    assert checkpoint["model_config"] == {
        "max_tokens": 32,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "feedforward_dim": 256,
        "dropout": 0.15,
    }


def test_formal_generation_contract() -> None:
    config = json.loads((ROOT / "configs/formal_100k_generation_config.json").read_text())
    assert config["requested_unique_sequences"] == 100000
    assert config["count_policy"] == "exact_n_with_invalid_and_duplicate_refill"
    assert config["seed"] == 20260727
    assert config["temperature"] == 1.0
    assert config["top_k"] == 10
    assert config["top_p"] == 1.0
