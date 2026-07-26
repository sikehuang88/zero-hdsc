"""Tests for embedding service — FakeEmbeddingService.

Tests use the fake implementation to avoid downloading model weights.
The SentenceTransformerEmbeddingService is covered by contract tests
in tests/contract/ (not run in every CI cycle).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ssa.adapters.embedding import (
    EmbeddingService,
    EmbeddingVector,
    FakeEmbeddingService,
)
from ssa.config import EmbeddingConfig


@pytest.fixture
def emb_config() -> EmbeddingConfig:
    return EmbeddingConfig(model="fake-model", dim=64, cache_dir="/tmp/fake")


@pytest.fixture
def emb(emb_config: EmbeddingConfig) -> FakeEmbeddingService:
    return FakeEmbeddingService(emb_config)


# ---------------------------------------------------------------------------
# EmbeddingVector
# ---------------------------------------------------------------------------


def test_embedding_vector_is_valid():
    v = EmbeddingVector(values=[0.1, 0.2, 0.3], model="test", dimension=3)
    assert v.dimension == 3
    assert v.normalized is True
    assert len(v.values) == 3


def test_embedding_vector_as_array():
    v = EmbeddingVector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    arr = v.as_array()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (3,)


def test_embedding_vector_as_bytes():
    v = EmbeddingVector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    b = v.as_bytes()
    assert isinstance(b, bytes)
    assert len(b) == 3 * 4  # 3 floats * 4 bytes each


def test_embedding_vector_rejects_zero_dim():
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingVector(values=[], model="test", dimension=0)


# ---------------------------------------------------------------------------
# FakeEmbeddingService
# ---------------------------------------------------------------------------


def test_fake_service_is_an_embedding_service(emb: FakeEmbeddingService):
    assert isinstance(emb, EmbeddingService)


def test_fake_embed_one_returns_vector(emb: FakeEmbeddingService):
    vec = emb.embed_one("hello world")
    assert isinstance(vec, EmbeddingVector)
    assert vec.dimension == 64
    assert vec.model == "fake-model"
    assert vec.normalized is True


def test_fake_embed_one_is_deterministic(emb: FakeEmbeddingService):
    v1 = emb.embed_one("hello")
    v2 = emb.embed_one("hello")
    assert v1.values == v2.values


def test_fake_embed_different_texts_differ(emb: FakeEmbeddingService):
    v1 = emb.embed_one("hello")
    v2 = emb.embed_one("world")
    assert v1.values != v2.values


def test_fake_embed_many_returns_list(emb: FakeEmbeddingService):
    vecs = emb.embed_many(["hello", "world", "foo"])
    assert len(vecs) == 3
    for v in vecs:
        assert v.dimension == 64


def test_fake_embed_many_batch_consistent(emb: FakeEmbeddingService):
    texts = ["hello", "world", "foo"]
    batch = emb.embed_many(texts)
    single = [emb.embed_one(t) for t in texts]
    for b, s in zip(batch, single, strict=True):
        assert b.values == s.values


def test_fake_embed_rejects_empty_string(emb: FakeEmbeddingService):
    with pytest.raises(ValueError, match="empty"):
        emb.embed_one("")


def test_fake_embed_many_rejects_empty_in_list(emb: FakeEmbeddingService):
    with pytest.raises(ValueError, match="empty"):
        emb.embed_many(["hello", "", "world"])


def test_fake_embed_many_empty_list(emb: FakeEmbeddingService):
    assert emb.embed_many([]) == []


def test_fake_vector_is_normalized(emb: FakeEmbeddingService):
    vec = emb.embed_one("test normalization")
    norm = math.sqrt(sum(x * x for x in vec.values))
    assert abs(norm - 1.0) < 0.01


def test_fake_call_count(emb: FakeEmbeddingService):
    assert emb.call_count == 0
    emb.embed_one("hello")
    assert emb.call_count == 1
    emb.embed_many(["a", "b"])
    assert emb.call_count == 2


def test_fake_custom_dimension():
    svc = FakeEmbeddingService(EmbeddingConfig(model="fake", dim=128))
    vec = svc.embed_one("test")
    assert vec.dimension == 128
    assert len(vec.values) == 128
