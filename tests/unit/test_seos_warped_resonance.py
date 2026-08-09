"""Affect-warped metric and free-energy resonance retrieval."""

# Mathematical notation in docstrings intentionally uses Unicode symbols.
# ruff: noqa: RUF002

from __future__ import annotations

import numpy as np
import pytest

from ssa.hdsc.free_energy_resonance import (
    ResonanceConfig,
    TemperatureParameters,
    gaussian_entropy,
    geodesic_detuning,
    heat_kernel,
    resonance_scan,
    retrieval_temperature,
)
from ssa.hdsc.transport import HDSCInvariantError
from ssa.hdsc.warped import (
    FLAT,
    WarpParameters,
    affect_distance,
    build_warped_conductance,
    conformal_factor,
    content_distance,
    warped_edge_length,
)

DIMENSION = 12


def _archive(count: int = 24, seed: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(count, DIMENSION))
    affect = np.column_stack(
        [
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(0.05, 1.0, count),
            rng.uniform(0.0, 1.0, count),
        ]
    )
    spread = rng.uniform(0.05, 0.9, count)
    return content, affect, spread


# -- metric ------------------------------------------------------------


def test_content_distance_is_a_metric_shape() -> None:
    content, _, _ = _archive()
    distance = content_distance(content)
    assert distance.shape == (content.shape[0], content.shape[0])
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert bool(np.all(distance >= 0.0))


def test_arousal_is_compared_as_a_ratio_not_a_difference() -> None:
    """An octave is a multiplicative statement, so 0.1→0.2 must equal 0.4→0.8."""

    affect = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.0, 0.4, 0.0], [0.0, 0.8, 0.0]])
    parameters = WarpParameters(valence_weight=0.0, tension_weight=0.0, arousal_floor=1e-9)
    distance = affect_distance(affect, parameters)
    assert distance[0, 1] == pytest.approx(distance[2, 3], rel=1e-6)


def test_crossing_zero_valence_costs_a_fixed_penalty() -> None:
    """A joyful and a grieving memory are categorically apart, not two units."""

    affect = np.array([[0.4, 0.5, 0.0], [0.5, 0.5, 0.0], [-0.4, 0.5, 0.0]])
    parameters = WarpParameters(sign_flip_penalty=1.0, valence_deadband=0.05)
    distance = affect_distance(affect, parameters)
    assert distance[0, 2] > distance[0, 1]


def test_flat_metric_has_no_position_dependence() -> None:
    _, affect, _ = _archive()
    assert FLAT.is_flat
    assert np.allclose(conformal_factor(affect, FLAT), 1.0)


def test_curvature_separates_pairs_that_a_diagonal_metric_cannot() -> None:
    """The claim in one assertion: identical content and affect gaps, different
    affect position, and only the warped metric tells them apart."""

    content = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    affect = np.array([[0.0, 0.35, 0.0], [0.0, 0.35, 0.0], [0.9, 0.95, 0.9], [0.9, 0.95, 0.9]])
    flat = warped_edge_length(content, affect, FLAT)
    warped = warped_edge_length(content, affect, WarpParameters(beta=4.0))
    assert flat[0, 1] == pytest.approx(flat[2, 3])
    assert warped[2, 3] > 2.0 * warped[0, 1]


def test_conductance_is_symmetric_and_non_negative() -> None:
    content, affect, _ = _archive()
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=2.0))
    assert np.allclose(conductance, conductance.T)
    assert bool(np.all(conductance >= 0.0))
    assert np.allclose(np.diag(conductance), 0.0)


def test_affect_requires_three_columns() -> None:
    content, _, _ = _archive()
    with pytest.raises(HDSCInvariantError, match="three columns"):
        affect_distance(np.zeros((content.shape[0], 2)))


# -- entropy and temperature -------------------------------------------


def test_entropy_grows_with_the_log_of_spread() -> None:
    """S ~ d·log σ, which is why callers must standardise before mixing it
    with a quadratic detuning term."""

    entropy = gaussian_entropy(np.array([0.1, 1.0, 10.0]), DIMENSION)
    first = entropy[1] - entropy[0]
    second = entropy[2] - entropy[1]
    assert first == pytest.approx(second, rel=1e-9)
    assert first == pytest.approx(DIMENSION * np.log(10.0), rel=1e-9)


def test_temperature_rises_with_arousal_and_falls_with_energy() -> None:
    calm = retrieval_temperature(arousal=0.1, tension=0.0, energy=0.9)
    alarmed = retrieval_temperature(arousal=0.95, tension=0.9, energy=0.2)
    assert alarmed > 5.0 * calm
    tired = retrieval_temperature(arousal=0.5, energy=0.0)
    rested = retrieval_temperature(arousal=0.5, energy=1.0)
    assert tired > rested


def test_temperature_is_bounded() -> None:
    parameters = TemperatureParameters(minimum=0.01, maximum=3.0)
    assert retrieval_temperature(arousal=50.0, parameters=parameters) == 3.0
    assert retrieval_temperature(arousal=-50.0, parameters=parameters) == 0.01


# -- resonance ---------------------------------------------------------


def test_detuning_is_non_negative_and_grows_with_graph_distance() -> None:
    content, affect, _ = _archive()
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=1.0))
    kernel = heat_kernel(conductance, 0.3)
    seed = np.zeros(content.shape[0])
    seed[0] = 1.0
    detuning = geodesic_detuning(kernel, seed, 0.3)
    assert bool(np.all(detuning >= 0.0))
    assert detuning[0] == pytest.approx(min(detuning))


def test_cold_search_returns_exactly_one_sharp_memory() -> None:
    content, affect, spread = _archive()
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=2.0))
    seed = np.zeros(content.shape[0])
    seed[0] = 1.0
    result = resonance_scan(
        conductance,
        seed,
        spread,
        temperature=0.01,
        content_dimension=DIMENSION,
        config=ResonanceConfig(recall_count=5, surfacing_detuning_cap=1e9),
        rng=np.random.default_rng(0),
    )
    assert len(result.selected) == 1
    assert result.distribution_entropy < 0.5


def test_hot_search_widens_and_is_driven_by_entropy_not_detuning() -> None:
    content, affect, spread = _archive(count=40)
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=2.0))
    seed = np.zeros(content.shape[0])
    seed[0] = 1.0
    config = ResonanceConfig(recall_count=5, surfacing_detuning_cap=1e9)

    cold = resonance_scan(
        conductance,
        seed,
        spread,
        temperature=0.02,
        content_dimension=DIMENSION,
        config=config,
        rng=np.random.default_rng(0),
    )
    hot = resonance_scan(
        conductance,
        seed,
        spread,
        temperature=50.0,
        content_dimension=DIMENSION,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert hot.distribution_entropy > cold.distribution_entropy
    assert len(hot.selected) > len(cold.selected)

    probability = np.asarray(hot.probability)
    mask = probability > 0.0
    entropy_corr = np.corrcoef(probability[mask], np.asarray(hot.entropy)[mask])[0, 1]
    detune_corr = np.corrcoef(probability[mask], -np.asarray(hot.detuning)[mask])[0, 1]
    assert entropy_corr > detune_corr


def test_a_hot_search_still_refuses_to_invent_a_memory() -> None:
    """Temperature widens what surfaces; it must never lower the evidence bar.

    A companion that confabulates harder as it gets more agitated is the exact
    failure this gate exists to prevent.
    """

    content, affect, spread = _archive(count=30)
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=2.0, bandwidth=0.2))
    seed = np.zeros(content.shape[0])
    seed[0] = 1.0
    config = ResonanceConfig(surfacing_detuning_cap=1.0)
    for temperature in (0.05, 1.0, 90.0):
        result = resonance_scan(
            conductance,
            seed,
            spread,
            temperature=temperature,
            content_dimension=DIMENSION,
            config=config,
            rng=np.random.default_rng(0),
        )
        assert not result.surfaced
        assert result.selected == ()


def test_scan_requires_a_candidate_to_stand_out_from_background() -> None:
    count = 8
    conductance = np.ones((count, count), dtype=np.float64) - np.eye(count)
    seed = np.zeros(count)
    seed[-1] = 1.0
    result = resonance_scan(
        conductance,
        seed,
        np.full(count, 0.2),
        temperature=1.0,
        content_dimension=DIMENSION,
        config=ResonanceConfig(
            recall_count=5,
            surfacing_detuning_cap=1e9,
            surfacing_margin=0.25,
        ),
    )

    assert not result.surfaced
    assert result.selected == ()
    assert "background" in result.reason


def test_the_query_itself_is_never_recalled() -> None:
    content, affect, spread = _archive()
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=1.0))
    seed = np.zeros(content.shape[0])
    seed[3] = 1.0
    result = resonance_scan(
        conductance,
        seed,
        spread,
        temperature=1.0,
        content_dimension=DIMENSION,
        config=ResonanceConfig(recall_count=5, surfacing_detuning_cap=1e9),
        rng=np.random.default_rng(0),
    )
    assert 3 not in result.selected
    assert result.probability[3] == 0.0


def test_scan_is_deterministic_for_a_fixed_generator() -> None:
    content, affect, spread = _archive()
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=1.5))
    seed = np.zeros(content.shape[0])
    seed[0] = 1.0
    config = ResonanceConfig(recall_count=4, surfacing_detuning_cap=1e9)
    first = resonance_scan(
        conductance,
        seed,
        spread,
        temperature=1.0,
        content_dimension=DIMENSION,
        config=config,
        rng=np.random.default_rng(11),
    )
    second = resonance_scan(
        conductance,
        seed,
        spread,
        temperature=1.0,
        content_dimension=DIMENSION,
        config=config,
        rng=np.random.default_rng(11),
    )
    assert first == second
