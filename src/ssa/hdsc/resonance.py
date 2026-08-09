"""State-modulated typed transport and evidence-gated resonance selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import cast

import numpy as np

from ssa.domain.traces import Trace, TraceLink, TraceLinkType
from ssa.hdsc.directed_transport import (
    HDSCH1DAudit,
    HDSCH1DParameters,
    build_directed_rates,
    propagate_directed,
)


@dataclass(frozen=True, slots=True)
class RecallState:
    valence: float = 0.0
    arousal: float = 0.2
    energy: float = 0.7
    connection_need: float = 0.4
    tension: float = 0.0
    situation_mode: str = "answer"
    origin_intent: bool = False


@dataclass(frozen=True, slots=True)
class ResonanceConfig:
    hop_budget: int = 3
    capacity: int = 12
    emergence_ratio: float = 1.35
    background_floor: float = 0.02
    min_semantic_support: float = 0.12
    content_weight: float = 0.35
    affect_weight: float = 0.25
    relation_weight: float = 0.15
    situation_weight: float = 0.15
    arc_weight: float = 0.10

    def normalized_weights(self) -> tuple[float, float, float, float, float]:
        raw = (
            self.content_weight,
            self.affect_weight,
            self.relation_weight,
            self.situation_weight,
            self.arc_weight,
        )
        total = sum(raw)
        if total <= 0.0:
            raise ValueError("resonance weights must have positive total")
        return cast(
            tuple[float, float, float, float, float],
            tuple(value / total for value in raw),
        )


@dataclass(frozen=True, slots=True)
class ResonanceMetrics:
    content_distance: float
    affect_distance: float
    relation_distance: float
    situation_distance: float
    arc_distance: float
    detuning: float
    coupling_mass: float
    damping: float
    amplitude: float
    background_ratio: float = 0.0


def state_modulated_gains(state: RecallState) -> dict[TraceLinkType, float]:
    """Return bounded conductance gains for the current internal state."""
    reminisce = state.situation_mode == "reminisce" or state.origin_intent
    low_energy = 1.0 - state.energy
    return {
        TraceLinkType.SEMANTIC: _clip(0.75 + 0.45 * low_energy, 0.25, 1.5),
        TraceLinkType.TEMPORAL: _clip(
            0.45 + 0.45 * state.arousal + (0.65 if reminisce else 0.0),
            0.2,
            1.8,
        ),
        TraceLinkType.AFFECTIVE: _clip(0.45 + 0.80 * state.arousal, 0.2, 1.6),
        TraceLinkType.ENTITY: _clip(0.45 + 0.90 * state.connection_need, 0.2, 1.6),
        TraceLinkType.CAUSAL: 1.0,
    }


def effective_hop_budget(config: ResonanceConfig, state: RecallState) -> int:
    budget = config.hop_budget
    if state.energy < 0.30:
        budget = min(budget, 1)
    elif state.energy < 0.50:
        budget = min(budget, 2)
    if state.situation_mode == "reminisce" or state.origin_intent:
        budget = min(8, budget + 1)
    return max(1, budget)


def modulated_rates(
    traces: Sequence[Trace],
    links: Sequence[TraceLink],
    state: RecallState,
) -> tuple[np.ndarray, dict[TraceLinkType, float]]:
    ids = [trace.id for trace in traces]
    by_id = {trace.id: trace for trace in traces}
    gains = state_modulated_gains(state)
    edges: list[tuple[str, str, float]] = []
    for link in links:
        gain = gains[link.link_type]
        direction_gain = 1.0
        if link.link_type == TraceLinkType.TEMPORAL and state.origin_intent:
            source = by_id[link.source_trace_id]
            target = by_id[link.target_trace_id]
            direction_gain = 1.35 if target.created_at_ms < source.created_at_ms else 0.55
        edges.append(
            (
                link.source_trace_id,
                link.target_trace_id,
                min(1.0, link.weight * gain * direction_gain),
            )
        )
    return build_directed_rates(ids, edges), gains


def transport_coupling(
    source_mass: np.ndarray,
    rates: np.ndarray,
    hop_budget: int,
) -> tuple[np.ndarray, HDSCH1DAudit]:
    result = propagate_directed(
        source_mass,
        rates,
        HDSCH1DParameters(transport_time=0.55 * hop_budget),
    )
    return np.asarray(result.output_mass, dtype=np.float64), result.audit


def arc_signature(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Return endpoint motion plus curvature, preserving rise-then-fall shape."""
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    dv = points[-1][0] - points[0][0]
    da = points[-1][1] - points[0][1]
    if len(points) < 3:
        return dv, da, 0.0
    curvature = sum(
        math.hypot(
            points[index - 1][0] - 2.0 * points[index][0] + points[index + 1][0],
            points[index - 1][1] - 2.0 * points[index][1] + points[index + 1][1],
        )
        for index in range(1, len(points) - 1)
    ) / (len(points) - 2)
    return dv, da, curvature


def arc_shape_distance(
    candidate: Sequence[tuple[float, float]],
    query: Sequence[tuple[float, float]],
) -> float:
    left = arc_signature(candidate)
    right = arc_signature(query)
    return min(1.0, math.dist(left, right) / math.sqrt(6.0))


def state_metric_tensor(
    state: RecallState,
    config: ResonanceConfig,
) -> np.ndarray:
    """Build a positive-definite local metric with state-dependent cross-axis coupling."""
    weights = np.asarray(config.normalized_weights(), dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    root = np.diag(np.sqrt(weights))
    arousal = _clip(state.arousal, 0.0, 1.0)
    connection = _clip(state.connection_need, 0.0, 1.0)
    reminisce = 1.0 if state.situation_mode == "reminisce" or state.origin_intent else 0.0

    # Affect bends content; relation bends situation; affective arcs bend one another.
    root[1, 0] = 0.35 * arousal * math.sqrt(weights[1])
    root[3, 2] = 0.30 * connection * math.sqrt(weights[3])
    root[4, 1] = 0.25 * (0.5 * arousal + 0.5 * reminisce) * math.sqrt(weights[4])
    metric = root.T @ root
    metric /= float(np.trace(metric))
    return cast(np.ndarray, metric)


def geodesic_detuning(
    distances: Sequence[float],
    metric_tensor: np.ndarray,
) -> float:
    """Evaluate the local Riemannian line element for one detuning vector."""
    vector = np.asarray(tuple(distances), dtype=np.float64)
    metric = np.asarray(metric_tensor, dtype=np.float64)
    if vector.shape != (5,) or metric.shape != (5, 5):
        raise ValueError("resonance detuning requires a five-axis metric")
    if not bool(np.all(np.isfinite(vector))) or not bool(np.all(np.isfinite(metric))):
        raise ValueError("resonance detuning inputs must be finite")
    if not bool(np.allclose(metric, metric.T, atol=1e-12)):
        raise ValueError("resonance metric must be symmetric")
    if float(np.min(np.linalg.eigvalsh(metric))) <= 0.0:
        raise ValueError("resonance metric must be positive definite")
    squared = float(vector @ metric @ vector)
    return math.sqrt(max(0.0, squared))


def resonance_metrics(
    *,
    content_distance: float,
    affect_distance: float,
    relation_distance: float,
    situation_distance: float,
    arc_distance: float,
    coupling_mass: float,
    damping: float,
    config: ResonanceConfig,
    metric_tensor: np.ndarray | None = None,
) -> ResonanceMetrics:
    axes = (
        content_distance,
        affect_distance,
        relation_distance,
        situation_distance,
        arc_distance,
    )
    metric = (
        metric_tensor
        if metric_tensor is not None
        else np.diag(np.asarray(config.normalized_weights(), dtype=np.float64))
    )
    detuning = geodesic_detuning(axes, metric)
    amplitude = coupling_mass / math.sqrt(detuning * detuning + damping * damping)
    return ResonanceMetrics(
        content_distance=content_distance,
        affect_distance=affect_distance,
        relation_distance=relation_distance,
        situation_distance=situation_distance,
        arc_distance=arc_distance,
        detuning=detuning,
        coupling_mass=coupling_mass,
        damping=damping,
        amplitude=amplitude,
    )


def gate_resonance(
    metrics: Mapping[str, ResonanceMetrics],
    config: ResonanceConfig,
) -> tuple[dict[str, ResonanceMetrics], float]:
    if not metrics:
        return {}, config.background_floor
    background = max(
        config.background_floor, float(median(item.amplitude for item in metrics.values()))
    )
    gated: dict[str, ResonanceMetrics] = {}
    for trace_id, item in metrics.items():
        ratio = item.amplitude / background
        gated[trace_id] = ResonanceMetrics(
            content_distance=item.content_distance,
            affect_distance=item.affect_distance,
            relation_distance=item.relation_distance,
            situation_distance=item.situation_distance,
            arc_distance=item.arc_distance,
            detuning=item.detuning,
            coupling_mass=item.coupling_mass,
            damping=item.damping,
            amplitude=item.amplitude,
            background_ratio=ratio,
        )
    return gated, background


def strongest_paths(
    node_ids: Sequence[str],
    rates: np.ndarray,
    source_mass: np.ndarray,
    hop_budget: int,
) -> dict[str, tuple[str, ...]]:
    """Return the strongest bounded-hop explanatory path to every node."""
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    best: dict[int, tuple[float, tuple[int, ...]]] = {
        position: (float(mass), (position,))
        for position, mass in enumerate(source_mass)
        if mass > 0.0
    }
    frontier = dict(best)
    for _hop in range(hop_budget):
        next_frontier: dict[int, tuple[float, tuple[int, ...]]] = {}
        for source, (path_score, path) in frontier.items():
            for target in range(len(node_ids)):
                rate = float(rates[target, source])
                if rate <= 0.0 or target in path:
                    continue
                candidate = (path_score * rate, (*path, target))
                if candidate[0] > next_frontier.get(target, (0.0, ()))[0]:
                    next_frontier[target] = candidate
                if candidate[0] > best.get(target, (0.0, ()))[0]:
                    best[target] = candidate
        frontier = next_frontier
        if not frontier:
            break
    reverse = {position: node_id for node_id, position in index.items()}
    return {
        reverse[position]: tuple(reverse[item] for item in path)
        for position, (_score, path) in best.items()
    }


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = [
    "RecallState",
    "ResonanceConfig",
    "ResonanceMetrics",
    "arc_shape_distance",
    "arc_signature",
    "effective_hop_budget",
    "gate_resonance",
    "geodesic_detuning",
    "modulated_rates",
    "resonance_metrics",
    "state_metric_tensor",
    "state_modulated_gains",
    "strongest_paths",
    "transport_coupling",
]
