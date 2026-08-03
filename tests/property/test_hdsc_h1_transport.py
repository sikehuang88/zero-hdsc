"""Invariant and numerical tests for the HDSC-H1 shadow kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ssa.hdsc.transport import (
    CLAIM_LEVEL,
    MICROSTATE_STATUS,
    MODEL_ID,
    MODELED_SCALE,
    HDSCH1Parameters,
    HDSCInvariantError,
    allocate_source_mass,
    build_semantic_conductance,
    graph_laplacian,
    propagate,
    run_shadow,
)


@st.composite
def _passive_systems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=1, max_value=8))
    entries = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=size * size,
            max_size=size * size,
        )
    )
    masses = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    raw = np.asarray(entries, dtype=np.float64).reshape(size, size)
    conductance = 0.5 * (raw + raw.T)
    np.fill_diagonal(conductance, 0.0)
    source = np.asarray(masses, dtype=np.float64)
    if float(np.sum(source)) == 0.0:
        source[0] = 1.0
    return source, conductance


@given(_passive_systems())
@settings(max_examples=60, deadline=None)
def test_passive_system_preserves_declared_invariants(
    system: tuple[np.ndarray, np.ndarray],
) -> None:
    source, conductance = system
    parameters = HDSCH1Parameters(diffusion_time=0.75, sink_rate=0.2)

    result = propagate(source, conductance, parameters)
    audit = result.audit

    assert audit.model_id == MODEL_ID
    assert audit.claim_level == CLAIM_LEVEL
    assert audit.modeled_scale == MODELED_SCALE
    assert audit.microstate_status == MICROSTATE_STATUS
    assert audit.minimum_output_mass >= 0.0
    assert abs(audit.conservation_residual) <= 1e-10
    assert audit.reciprocity_residual <= 1e-10
    assert audit.laplacian_row_sum_residual <= 1e-10
    assert audit.dirichlet_after <= audit.dirichlet_before + 1e-10
    assert audit.shannon_entropy_after + 1e-10 >= audit.shannon_entropy_before
    assert audit.output_mass == pytest.approx(
        audit.input_mass * math.exp(-0.2 * 0.75),
        abs=1e-10,
    )


def test_source_budget_does_not_grow_with_duplicate_nodes() -> None:
    parameters = HDSCH1Parameters(source_budget=3.0, max_sources=3)
    source = allocate_source_mass(
        np.asarray([0.9, 0.9, 0.7, 0.6]),
        np.ones(4),
        np.ones(4),
        parameters,
    )

    assert float(np.sum(source)) == pytest.approx(3.0, abs=1e-12)
    assert np.count_nonzero(source) == 3


def test_semantic_conductance_is_reciprocal_and_nonnegative() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    conductance = build_semantic_conductance(
        embeddings,
        HDSCH1Parameters(conductance_threshold=0.2),
    )

    assert np.allclose(conductance, conductance.T)
    assert np.all(conductance >= 0.0)
    assert np.allclose(np.diag(conductance), 0.0)
    assert np.allclose(graph_laplacian(conductance).sum(axis=1), 0.0)


def test_disconnected_components_do_not_exchange_mass() -> None:
    conductance = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    result = propagate(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        conductance,
        HDSCH1Parameters(diffusion_time=2.0),
    )

    assert sum(result.output_mass[:2]) == pytest.approx(1.0, abs=1e-12)
    assert sum(result.output_mass[2:]) == pytest.approx(0.0, abs=1e-12)


def test_ring_diffusion_reaches_uniform_fixed_point() -> None:
    conductance = np.zeros((4, 4), dtype=np.float64)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 0)):
        conductance[left, right] = 1.0
        conductance[right, left] = 1.0

    result = propagate(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        conductance,
        HDSCH1Parameters(diffusion_time=100.0),
    )

    assert result.diffused_mass == pytest.approx((0.25, 0.25, 0.25, 0.25), abs=1e-10)


def test_permutation_does_not_change_physics() -> None:
    source = np.asarray([0.7, 0.2, 0.1])
    conductance = np.asarray(
        [
            [0.0, 1.0, 0.2],
            [1.0, 0.0, 0.4],
            [0.2, 0.4, 0.0],
        ]
    )
    permutation = np.asarray([2, 0, 1])
    parameters = HDSCH1Parameters(diffusion_time=0.9, sink_rate=0.15)

    original = np.asarray(propagate(source, conductance, parameters).output_mass)
    permuted = np.asarray(
        propagate(
            source[permutation],
            conductance[np.ix_(permutation, permutation)],
            parameters,
        ).output_mass
    )

    assert np.allclose(permuted, original[permutation], atol=1e-12)


def test_continuous_time_semigroup_composes() -> None:
    source = np.asarray([1.0, 0.0, 0.0])
    conductance = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    first_parameters = HDSCH1Parameters(diffusion_time=0.3, sink_rate=0.2)
    second_parameters = HDSCH1Parameters(diffusion_time=0.8, sink_rate=0.2)
    combined_parameters = HDSCH1Parameters(diffusion_time=1.1, sink_rate=0.2)

    first = propagate(source, conductance, first_parameters)
    composed = propagate(np.asarray(first.output_mass), conductance, second_parameters)
    combined = propagate(source, conductance, combined_parameters)

    assert np.allclose(composed.output_mass, combined.output_mass, atol=1e-12)


def test_shadow_path_handles_singleton_without_artificial_loss() -> None:
    result = run_shadow(
        similarities=np.asarray([0.9]),
        freshness=np.asarray([1.0]),
        importance=np.asarray([0.8]),
        embeddings=np.asarray([[1.0, 0.0]]),
        parameters=HDSCH1Parameters(source_budget=2.0),
    )

    assert result.source_mass == pytest.approx((2.0,))
    assert result.output_mass == pytest.approx((2.0,))
    assert result.audit.dirichlet_before == 0.0
    assert result.audit.dirichlet_after == 0.0


@pytest.mark.parametrize(
    "conductance",
    [
        np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        np.asarray([[0.0, -1.0], [-1.0, 0.0]]),
        np.asarray([[0.1, 1.0], [1.0, 0.0]]),
    ],
)
def test_non_passive_conductance_is_rejected(conductance: np.ndarray) -> None:
    with pytest.raises(HDSCInvariantError):
        propagate(np.asarray([1.0, 0.0]), conductance, HDSCH1Parameters())


def test_no_positive_source_evidence_is_explicit() -> None:
    with pytest.raises(HDSCInvariantError, match="no trace has positive"):
        allocate_source_mass(
            np.asarray([-0.5, -0.2]),
            np.ones(2),
            np.ones(2),
            HDSCH1Parameters(affinity_threshold=0.0),
        )
