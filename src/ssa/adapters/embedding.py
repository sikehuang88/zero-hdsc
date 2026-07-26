"""Embedding service — wraps sentence-transformers behind a stable interface.

Pipeline §13.2 / §13.4:
- `EmbeddingVector`: immutable vector object with model + dim metadata.
- `EmbeddingService` protocol: `embed_one` / `embed_many`.
- Embedding pipeline: length check → encode → normalize → NaN/Inf/dim check → cache → return.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, field_validator

from ssa.config import EmbeddingConfig


class EmbeddingVector(BaseModel):
    """An immutable embedding vector with provenance metadata."""

    values: list[float]
    model: str
    dimension: int
    normalized: bool = True

    @field_validator("dimension")
    @classmethod
    def _check_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("dimension must be positive")
        return v

    def as_array(self) -> np.ndarray:
        """Return as a numpy float32 array."""
        return np.array(self.values, dtype=np.float32)  # type: ignore[no-any-return]

    def as_bytes(self) -> bytes:
        """Return as raw bytes for sqlite-vec BLOB storage."""
        return self.as_array().tobytes()


@runtime_checkable
class EmbeddingService(Protocol):
    """Stable embedding interface (pipeline §13.2)."""

    def embed_one(self, text: str) -> EmbeddingVector: ...

    def embed_many(self, texts: list[str]) -> list[EmbeddingVector]: ...


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SentenceTransformerEmbeddingService:
    """Production embedding service backed by sentence-transformers.

    Uses bge-small-zh-v1.5 by default (512-dim, CPU-capable).
    Caches results by (model, content_hash) to avoid re-encoding.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None  # SentenceTransformer, loaded lazily
        self._cache: dict[str, EmbeddingVector] = {}

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    @property
    def dimension(self) -> int:
        return self._config.dim

    @property
    def model_name(self) -> str:
        return self._config.model

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._config.model,
                cache_folder=self._config.cache_dir,
            )
        return self._model

    def embed_one(self, text: str) -> EmbeddingVector:
        if not text:
            raise ValueError("cannot embed empty text")
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []

        # Check for empty strings.
        for i, t in enumerate(texts):
            if not t:
                raise ValueError(f"text at index {i} is empty")

        # Batch encode, checking cache first.
        results: list[EmbeddingVector | None] = [None] * len(texts)
        to_encode: list[str] = []
        encode_indices: list[int] = []

        for i, text in enumerate(texts):
            key = f"{self._config.model}:{_content_hash(text)}"
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_encode.append(text)
                encode_indices.append(i)

        if to_encode:
            model = self._ensure_model()
            embeddings = model.encode(
                to_encode,
                batch_size=self._config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for idx, text, emb in zip(encode_indices, to_encode, embeddings, strict=True):
                vec = self._validate_and_build(emb, text)
                results[idx] = vec
                key = f"{self._config.model}:{_content_hash(text)}"
                self._cache[key] = vec

        return [r for r in results if r is not None]

    def _validate_and_build(self, emb: np.ndarray, text: str) -> EmbeddingVector:
        """Validate NaN/Inf/dimension and build an EmbeddingVector."""
        arr = np.asarray(emb, dtype=np.float32)

        if arr.ndim != 1:
            raise ValueError(
                f"embedding for text has wrong shape: {arr.shape}, expected 1-D"
            )

        if arr.shape[0] != self._config.dim:
            raise ValueError(
                f"embedding dimension mismatch: got {arr.shape[0]}, "
                f"expected {self._config.dim} (model={self._config.model})"
            )

        if math.isnan(float(np.any(arr))) or math.isinf(float(np.any(arr))):
            raise ValueError(f"embedding contains NaN or Inf for text: {text[:50]!r}")

        values = arr.tolist()

        # Re-normalize defensively (sentence-transformers should already do this).
        norm = float(np.linalg.norm(arr))
        if norm > 0 and abs(norm - 1.0) > 0.01:
            arr = arr / norm
            values = arr.tolist()

        return EmbeddingVector(
            values=values,
            model=self._config.model,
            dimension=self._config.dim,
            normalized=True,
        )


class FakeEmbeddingService:
    """Deterministic fake for tests — no model download, no torch.

    Produces a hash-based pseudo-embedding of the configured dimension.
    Same text always produces the same vector, so caching and dedup
    tests work correctly.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._call_count = 0

    @property
    def dimension(self) -> int:
        return self._config.dim

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def call_count(self) -> int:
        return self._call_count

    def embed_one(self, text: str) -> EmbeddingVector:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        self._call_count += 1
        results = []
        for text in texts:
            if not text:
                raise ValueError("cannot embed empty text")
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # Expand hash to fill the required dimension.
            raw: np.ndarray = np.frombuffer(
                h * (self._config.dim // 32 + 1), dtype=np.uint8
            )
            arr: np.ndarray = raw[: self._config.dim].astype(np.float32)
            # Normalize to unit vector.
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            results.append(
                EmbeddingVector(
                    values=arr.tolist(),
                    model=self._config.model,
                    dimension=self._config.dim,
                    normalized=True,
                )
            )
        return results


__all__ = [
    "EmbeddingService",
    "EmbeddingVector",
    "FakeEmbeddingService",
    "SentenceTransformerEmbeddingService",
]
