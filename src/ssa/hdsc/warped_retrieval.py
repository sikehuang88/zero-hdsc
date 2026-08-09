"""Adaptive, storage-agnostic adapter for warped free-energy retrieval.

The online service supplies immutable arrays and receives an audit value.  This
module owns neither trace persistence nor prompt construction, so the metric
experiment can evolve without coupling the certified H1D serving path to its
data model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from ssa.hdsc.free_energy_resonance import (
    ResonanceConfig as FreeEnergyConfig,
)
from ssa.hdsc.free_energy_resonance import resonance_scan
from ssa.hdsc.transport import HDSCInvariantError
from ssa.hdsc.warped import WarpParameters, build_warped_conductance

MODEL_ID: Final = "hdsc-adaptive-warped-free-energy-v1"
_EPS: Final = 1e-12


@dataclass(frozen=True, slots=True)
class WarpedRetrievalConfig:
    """Bounded experiment settings; defaults remain shadow-only in serving."""

    low_information: float = 0.12
    high_information: float = 0.45
    warp_beta: float = 4.0
    bandwidth: float = 1.0
    diffusion_time: float = 0.35
    temperature: float = 0.15
    recall_count: int = 5
    surfacing_detuning_cap: float = 6.0
    surfacing_margin: float = 0.25

    def __post_init__(self) -> None:
        numeric = (
            self.low_information,
            self.high_information,
            self.warp_beta,
            self.bandwidth,
            self.diffusion_time,
            self.temperature,
            self.surfacing_detuning_cap,
            self.surfacing_margin,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("warped retrieval settings must be finite")
        if not 0.0 <= self.low_information < self.high_information <= 1.0:
            raise ValueError("warped information thresholds must satisfy 0 <= low < high <= 1")
        if self.warp_beta < 0.0:
            raise ValueError("warp_beta must be non-negative")
        if self.bandwidth <= 0.0 or self.diffusion_time <= 0.0 or self.temperature <= 0.0:
            raise ValueError("warped bandwidth, diffusion time, and temperature must be positive")
        if self.recall_count < 1:
            raise ValueError("warped recall_count must be positive")
        if self.surfacing_detuning_cap <= 0.0:
            raise ValueError("warped surfacing_detuning_cap must be positive")
        if self.surfacing_margin < 0.0:
            raise ValueError("warped surfacing_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class WarpedRetrievalRequest:
    trace_ids: tuple[str, ...]
    embeddings: NDArray[np.float64]
    affect: NDArray[np.float64]
    spread: NDArray[np.float64]
    query_embedding: NDArray[np.float64]
    query_affect: tuple[float, float, float]
    semantic_similarity: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WarpedRetrievalResult:
    model_id: str
    content_information: float
    affect_mix: float
    temperature: float
    selected_trace_ids: tuple[str, ...]
    detuning: tuple[float, ...]
    probability: tuple[float, ...]
    surfaced: bool
    reason: str
    parameters: tuple[tuple[str, float], ...]


class WarpedRetrievalStrategy(Protocol):
    """Injection boundary used by the trace-space service."""

    def scan(self, request: WarpedRetrievalRequest) -> WarpedRetrievalResult: ...


def query_content_information(similarities: np.ndarray) -> float:
    """Estimate whether content provides a distinct retrieval peak."""
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise HDSCInvariantError("semantic similarities must be a non-empty vector")
    if not bool(np.all(np.isfinite(values))):
        raise HDSCInvariantError("semantic similarities must be finite")
    clipped = np.clip(values, 0.0, 1.0)
    if clipped.size == 1:
        return float(clipped[0])
    peak = float(np.max(clipped))
    background = float(np.median(clipped))
    prominence = max(0.0, peak - background) / max(_EPS, 1.0 - background)
    dispersion = min(1.0, 2.0 * float(np.std(clipped)))
    return float(np.clip(0.75 * prominence + 0.25 * dispersion, 0.0, 1.0))


def adaptive_warp_parameters(
    content_information: float,
    config: WarpedRetrievalConfig,
) -> tuple[WarpParameters, float]:
    """Suppress affect geometry as query content becomes informative."""
    position = np.clip(
        (content_information - config.low_information)
        / (config.high_information - config.low_information),
        0.0,
        1.0,
    )
    smooth = float(position * position * (3.0 - 2.0 * position))
    affect_mix = 1.0 - smooth
    parameters = WarpParameters(
        beta=config.warp_beta * affect_mix,
        affect_scale=1.0 / max(0.05, affect_mix),
        bandwidth=config.bandwidth,
    )
    return parameters, affect_mix


class AdaptiveWarpedRetriever:
    """Pure NumPy implementation of the adaptive experimental strategy."""

    def __init__(self, config: WarpedRetrievalConfig | None = None) -> None:
        self.config = config or WarpedRetrievalConfig()

    def scan(self, request: WarpedRetrievalRequest) -> WarpedRetrievalResult:
        count = len(request.trace_ids)
        embeddings = np.asarray(request.embeddings, dtype=np.float64)
        affect = np.asarray(request.affect, dtype=np.float64)
        spread = np.asarray(request.spread, dtype=np.float64)
        query = np.asarray(request.query_embedding, dtype=np.float64)
        similarities = np.asarray(request.semantic_similarity, dtype=np.float64)
        if (
            count == 0
            or len(set(request.trace_ids)) != count
            or any(not trace_id.strip() for trace_id in request.trace_ids)
        ):
            raise HDSCInvariantError("warped retrieval requires unique archive trace IDs")
        if embeddings.ndim != 2 or embeddings.shape[0] != count:
            raise HDSCInvariantError("warped embeddings must align with trace IDs")
        if query.ndim != 1 or query.size != embeddings.shape[1]:
            raise HDSCInvariantError("warped query embedding dimension must match the archive")
        if affect.shape != (count, 3):
            raise HDSCInvariantError("warped affect must have one three-axis row per trace")
        if spread.shape != (count,) or similarities.shape != (count,):
            raise HDSCInvariantError("warped spread and similarities must align with trace IDs")
        arrays = (embeddings, affect, spread, query, similarities)
        if not all(bool(np.all(np.isfinite(values))) for values in arrays):
            raise HDSCInvariantError("warped retrieval inputs must be finite")
        if bool(np.any(spread < 0.0)):
            raise HDSCInvariantError("warped memory spread must be non-negative")
        query_affect = np.asarray(request.query_affect, dtype=np.float64)
        if query_affect.shape != (3,) or not bool(np.all(np.isfinite(query_affect))):
            raise HDSCInvariantError("warped query affect must be a finite three-axis vector")

        content_information = query_content_information(similarities)
        parameters, affect_mix = adaptive_warp_parameters(content_information, self.config)
        extended_embeddings = np.vstack((embeddings, query[None, :]))
        extended_affect = np.vstack((affect, query_affect[None, :]))
        conductance = build_warped_conductance(
            extended_embeddings,
            extended_affect,
            parameters,
        )
        seed: NDArray[np.float64] = np.zeros(count + 1, dtype=np.float64)
        seed[-1] = 1.0
        extended_spread: NDArray[np.float64] = np.append(spread, float(np.mean(spread)))
        result = resonance_scan(
            conductance,
            seed,
            extended_spread,
            temperature=self.config.temperature,
            content_dimension=embeddings.shape[1],
            config=FreeEnergyConfig(
                diffusion_time=self.config.diffusion_time,
                recall_count=self.config.recall_count,
                surfacing_detuning_cap=self.config.surfacing_detuning_cap,
                surfacing_margin=self.config.surfacing_margin,
            ),
        )
        selected = tuple(request.trace_ids[index] for index in result.selected if index < count)
        return WarpedRetrievalResult(
            model_id=MODEL_ID,
            content_information=content_information,
            affect_mix=affect_mix,
            temperature=self.config.temperature,
            selected_trace_ids=selected,
            detuning=tuple(result.detuning[:count]),
            probability=tuple(result.probability[:count]),
            surfaced=result.surfaced and bool(selected),
            reason=result.reason,
            parameters=(
                ("beta", parameters.beta),
                ("affect_scale", parameters.affect_scale),
                ("bandwidth", parameters.bandwidth),
                ("diffusion_time", self.config.diffusion_time),
                ("detuning_cap", self.config.surfacing_detuning_cap),
                ("surfacing_margin", self.config.surfacing_margin),
            ),
        )


__all__ = [
    "MODEL_ID",
    "AdaptiveWarpedRetriever",
    "WarpedRetrievalConfig",
    "WarpedRetrievalRequest",
    "WarpedRetrievalResult",
    "WarpedRetrievalStrategy",
    "adaptive_warp_parameters",
    "query_content_information",
]
