"""Sequence-only API for the accepted LIMP-AR generator."""

from __future__ import annotations

import os
import hashlib
from functools import lru_cache
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from liup_generator.autoregressive import AMPAutoregressiveModel, sample_sequences


MODEL_PATH = Path(os.environ.get("LIMP_MODEL_PATH", os.environ.get("LIUP_MODEL_PATH", "/app/models/generator.pt")))
DEVICE_NAME = os.environ.get("LIMP_DEVICE", os.environ.get("LIUP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device(DEVICE_NAME)
model: AMPAutoregressiveModel | None = None
model_metadata: dict = {}


@lru_cache(maxsize=8)
def _artifact_fingerprint(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "size_bytes": None, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "exists": True, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_model() -> None:
    global model, model_metadata
    if model is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LIMP-AR checkpoint not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    loaded = AMPAutoregressiveModel(**checkpoint["model_config"]).to(device)
    loaded.load_state_dict(checkpoint["state_dict"])
    loaded.eval()
    model = loaded
    model_metadata = {
        "architecture": checkpoint["architecture"],
        "model_config": checkpoint["model_config"],
        "training_seed": checkpoint["seed"],
        "best_validation_nll": checkpoint["best_validation_nll"],
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except FileNotFoundError:
        # Health exposes the cause and generation returns a 503.  This lets the
        # orchestrator start the container before a model volume is mounted.
        pass
    yield


app = FastAPI(title="LIMP-AR AMP Generator", version="1.0.0-rc1", lifespan=lifespan)


class GenerateRequest(BaseModel):
    # ``n`` keeps compatibility with the Agent's existing low-level generator
    # client. ``num_samples`` supports direct API callers.
    n: Optional[int] = Field(default=None, ge=1, le=2048)
    num_samples: Optional[int] = Field(default=None, ge=1, le=2048)
    min_length: int = Field(default=8, ge=1, le=30)
    max_length: int = Field(default=30, ge=1, le=30)
    temperature: float = Field(default=1.3, gt=0, le=2.0)
    top_k: int = Field(default=0, ge=0, le=23)
    top_p: float = Field(default=0.90, gt=0, le=1.0)
    seed: Optional[int] = None
    # Accepted but intentionally unused: target-specific optimisation belongs
    # to the Agent Harness after this service returns candidate sequences.
    target: Optional[str] = None
    prompt: Optional[str] = None


def _requested_count(request: GenerateRequest) -> int:
    if request.n is not None and request.num_samples is not None and request.n != request.num_samples:
        raise HTTPException(status_code=422, detail="n and num_samples must agree when both are supplied")
    return request.n if request.n is not None else request.num_samples or 5


def _sample_unique(request: GenerateRequest, count: int, seed: int) -> list[dict[str, object]]:
    assert model is not None
    if request.min_length > request.max_length:
        raise HTTPException(status_code=422, detail="min_length cannot exceed max_length")
    collected: list[dict[str, object]] = []
    seen: set[str] = set()
    # Exact deduplication is a basic generator guarantee.  It does not query a
    # training database and is not a replacement for Harness novelty screening.
    for attempt in range(8):
        needed = count - len(collected)
        if needed <= 0:
            break
        samples = sample_sequences(
            model,
            count=max(needed * 2, needed),
            min_length=request.min_length,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=seed + attempt,
            device=device,
        )
        for sample in samples:
            if sample.sequence and sample.sequence not in seen:
                seen.add(sample.sequence)
                collected.append({"sequence": sample.sequence, "log_probability": sample.log_probability})
            if len(collected) == count:
                break
    if len(collected) != count:
        raise HTTPException(status_code=503, detail="Unable to produce the requested unique sequence count")
    return collected


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ready" if model is not None else "not_ready",
        "model": "LIMP-AR",
        "device": str(device),
        "sequence_only": True,
        "metadata": model_metadata,
        "model_artifact": _artifact_fingerprint(MODEL_PATH),
        "service_code": _artifact_fingerprint(Path(__file__)),
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, object]:
    if model is None:
        try:
            load_model()
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    count = _requested_count(request)
    seed = request.seed if request.seed is not None else secrets.randbits(31)
    generated = _sample_unique(request, count, seed)
    return {
        "generator": "LIMP-AR",
        "version": "1.0.0-rc1",
        "seed": seed,
        "sequences": [item["sequence"] for item in generated],
        "metadata": generated,
        "generated_count": len(generated),
        "sampling": {
            "min_length": request.min_length,
            "max_length": request.max_length,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
        },
    }
