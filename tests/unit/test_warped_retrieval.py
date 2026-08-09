"""Adaptive scheduling and storage-free warped retrieval adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from ssa.hdsc.transport import HDSCInvariantError
from ssa.hdsc.warped_retrieval import (
    AdaptiveWarpedRetriever,
    WarpedRetrievalConfig,
    WarpedRetrievalRequest,
    adaptive_warp_parameters,
    query_content_information,
)


def _request(seed: int = 7) -> WarpedRetrievalRequest:
    rng = np.random.default_rng(seed)
    count = 8
    dimension = 12
    return WarpedRetrievalRequest(
        trace_ids=tuple(f"trace-{index}" for index in range(count)),
        embeddings=rng.normal(size=(count, dimension)),
        affect=np.column_stack(
            (
                rng.uniform(-1.0, 1.0, count),
                rng.uniform(0.05, 1.0, count),
                np.zeros(count),
            )
        ),
        spread=rng.uniform(0.05, 0.9, count),
        query_embedding=rng.normal(size=dimension),
        query_affect=(0.2, 0.4, 0.0),
        semantic_similarity=np.asarray([0.95, 0.2, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]),
    )


def test_content_information_detects_a_distinct_semantic_peak() -> None:
    flat = query_content_information(np.full(8, 0.2))
    peaked = query_content_information(np.asarray([0.95, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))

    assert flat == pytest.approx(0.0)
    assert peaked > 0.7


def test_adaptive_schedule_suppresses_affect_for_informative_queries() -> None:
    config = WarpedRetrievalConfig(low_information=0.1, high_information=0.4)
    vague, vague_mix = adaptive_warp_parameters(0.0, config)
    precise, precise_mix = adaptive_warp_parameters(0.9, config)

    assert vague_mix == pytest.approx(1.0)
    assert vague.beta == pytest.approx(config.warp_beta)
    assert precise_mix == pytest.approx(0.0)
    assert precise.beta == pytest.approx(0.0)
    assert precise.affect_scale > vague.affect_scale


def test_adapter_is_deterministic_and_returns_only_archive_ids() -> None:
    request = _request()
    retriever = AdaptiveWarpedRetriever(
        WarpedRetrievalConfig(surfacing_detuning_cap=1e9, recall_count=4)
    )

    first = retriever.scan(request)
    second = retriever.scan(request)

    assert first == second
    assert first.selected_trace_ids
    assert set(first.selected_trace_ids) <= set(request.trace_ids)
    assert len(first.detuning) == len(request.trace_ids)
    assert len(first.probability) == len(request.trace_ids)


def test_adapter_rejects_misaligned_inputs() -> None:
    request = _request()
    broken = WarpedRetrievalRequest(
        trace_ids=request.trace_ids,
        embeddings=request.embeddings[:-1],
        affect=request.affect,
        spread=request.spread,
        query_embedding=request.query_embedding,
        query_affect=request.query_affect,
        semantic_similarity=request.semantic_similarity,
    )

    with pytest.raises(HDSCInvariantError, match="align"):
        AdaptiveWarpedRetriever().scan(broken)
