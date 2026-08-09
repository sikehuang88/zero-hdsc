"""Deterministic contracts for typed, state-modulated resonance recall."""

from __future__ import annotations

import numpy as np
import pytest

from ssa.domain.traces import TraceLinkType
from ssa.hdsc.resonance import (
    RecallState,
    ResonanceConfig,
    ResonanceMetrics,
    arc_shape_distance,
    arc_signature,
    effective_hop_budget,
    gate_resonance,
    geodesic_detuning,
    state_metric_tensor,
    state_modulated_gains,
    strongest_paths,
)


def test_state_modulates_relation_conductance_and_hop_budget() -> None:
    baseline = RecallState(arousal=0.1, energy=0.8, connection_need=0.1)
    activated = RecallState(
        arousal=0.9,
        energy=0.8,
        connection_need=0.9,
        situation_mode="reminisce",
        origin_intent=True,
    )
    baseline_gains = state_modulated_gains(baseline)
    activated_gains = state_modulated_gains(activated)

    assert activated_gains[TraceLinkType.AFFECTIVE] > baseline_gains[TraceLinkType.AFFECTIVE]
    assert activated_gains[TraceLinkType.TEMPORAL] > baseline_gains[TraceLinkType.TEMPORAL]
    assert activated_gains[TraceLinkType.ENTITY] > baseline_gains[TraceLinkType.ENTITY]
    assert effective_hop_budget(ResonanceConfig(hop_budget=4), RecallState(energy=0.2)) == 1
    assert effective_hop_budget(ResonanceConfig(hop_budget=3), activated) == 4


def test_arc_signature_distinguishes_curvature_with_equal_endpoints() -> None:
    flat = ((0.0, 0.2), (0.0, 0.2), (0.0, 0.2))
    rise_then_fall = ((0.0, 0.2), (0.8, 0.9), (0.0, 0.2))

    assert arc_signature(flat)[:2] == arc_signature(rise_then_fall)[:2]
    assert arc_signature(rise_then_fall)[2] > arc_signature(flat)[2]
    assert arc_shape_distance(flat, rise_then_fall) > 0.0


def test_state_metric_is_positive_definite_and_couples_axes() -> None:
    config = ResonanceConfig()
    calm = state_metric_tensor(RecallState(arousal=0.0, connection_need=0.0), config)
    activated = state_metric_tensor(
        RecallState(
            arousal=0.9,
            connection_need=0.9,
            situation_mode="reminisce",
        ),
        config,
    )
    distances = (0.7, 0.6, 0.5, 0.4, 0.3)

    assert np.min(np.linalg.eigvalsh(activated)) > 0.0
    assert activated[0, 1] != pytest.approx(0.0)
    assert activated[2, 3] != pytest.approx(0.0)
    assert geodesic_detuning(distances, activated) != pytest.approx(
        geodesic_detuning(distances, calm)
    )


def test_background_gate_preserves_explicit_empty_result() -> None:
    config = ResonanceConfig(emergence_ratio=1.35, background_floor=0.02)
    metrics = {
        "left": ResonanceMetrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.8, 0.10),
        "right": ResonanceMetrics(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.8, 0.11),
    }

    gated, background = gate_resonance(metrics, config)

    assert background == pytest.approx(0.105)
    assert all(item.background_ratio < config.emergence_ratio for item in gated.values())


def test_strongest_paths_records_bounded_multi_hop_explanation() -> None:
    node_ids = ("origin", "middle", "query")
    rates = np.zeros((3, 3), dtype=np.float64)
    rates[1, 2] = 0.9
    rates[0, 1] = 0.8
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    paths = strongest_paths(node_ids, rates, source, hop_budget=2)

    assert paths["origin"] == ("query", "middle", "origin")
