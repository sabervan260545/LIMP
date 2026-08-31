from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liup_generator.autoregressive import AMPAutoregressiveModel


def test_service_returns_only_sequences(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "generator.pt"
    torch.manual_seed(3)
    model = AMPAutoregressiveModel(num_layers=1, d_model=32, num_heads=4, feedforward_dim=64, dropout=0.0)
    torch.save(
        {
            "architecture": "test",
            "model_config": model.config_dict(),
            "seed": 3,
            "best_validation_nll": 0.0,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    # Directly test the API's sampling core; importing an ASGI client is avoided
    # because production pins its own FastAPI/httpx versions in Docker.
    import app as service

    original_path, original_model = service.MODEL_PATH, service.model
    service.MODEL_PATH, service.model = checkpoint_path, None
    service.load_model()
    request = service.GenerateRequest(n=3, min_length=8, max_length=12, seed=12)
    first = service.generate(request)
    second = service.generate(request)
    assert first["sequences"] == second["sequences"]
    assert len(first["sequences"]) == 3
    assert all(8 <= len(sequence) <= 12 for sequence in first["sequences"])
    service.MODEL_PATH, service.model = original_path, original_model
