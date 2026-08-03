"""Properties for HDSC-H1D directed nonreversible transport."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ssa.hdsc.directed_transport import (
    CLAIM_LEVEL,
    MODEL_ID,
    PHYSICAL_THERMODYNAMIC_STATUS,
    RATE_CONVENTION,
    TOPOLOGY_CLASS,
    HDSCH1DParameters,
    build_directed_rates,
    directed_generator,
    propagate_directed,
)
from ssa.hdsc.transport import HDSCH1Parameters, HDSCInvariantError, propagate


@st.composite
def _directed_systems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=1, max_value=6))
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
    rates = np.asarray(entries, dtype=np.float64).reshape(size, size)
    np.fill_diagonal(rates, 0.0)
    source = np.asarray(masses, dtype=np.float64)
    if float(np.sum(source)) == 0.0:
        source[0] = 1.0
    return source, rates


def _cycle_rates(size: int, *, clockwise: float, counterclockwise: float) -> np.ndarray:
    edges: list[tuple[str, str, float]] = []
    node_ids = [str(index) for index in range(size)]
    for source in range(size):
        target = (source + 1) % size
        edges.append((str(source), str(target), clockwise))
        if counterclockwise > 0.0:
            edges.append((str(target), str(source), counterclockwise))
    return build_directed_rates(node_ids, edges)


def test_directed_rate_builder_preserves_edge_orientation() -> None:
    rates = build_directed_rates(
        ["past", "present", "future"],
        [
            ("past", "present", 2.0),
            ("present", "future", 3.0),
            ("past", "present", 0.5),
        ],
    )

    assert rates[1, 0] == pytest.approx(2.5)
    assert rates[0, 1] == 0.0
    assert rates[2, 1] == pytest.approx(3.0)
    assert rates[1, 2] == 0.0


def test_one_way_transport_matches_two_node_master_equation() -> None:
    rate = 1.7
    duration = 0.8
    rates = np.asarray([[0.0, 0.0], [rate, 0.0]])
    result = propagate_directed(
        np.asarray([1.0, 0.0]),
        rates,
        HDSCH1DParameters(transport_time=duration),
    )

    remaining = math.exp(-rate * duration)
    assert result.transported_mass == pytest.approx((remaining, 1.0 - remaining), abs=1e-12)
    assert result.audit.asymmetry_index == pytest.approx(1.0)
    assert result.audit.directed_excess_rate_mass == pytest.approx(rate)
    assert result.audit.reciprocal_rate_mass == 0.0


@given(_directed_systems())
@settings(max_examples=60, deadline=None)
def test_directed_markov_transport_preserves_mass_and_positivity(
    system: tuple[np.ndarray, np.ndarray],
) -> None:
    source, rates = system
    result = propagate_directed(
        source,
        rates,
        HDSCH1DParameters(transport_time=0.6, sink_rate=0.2),
    )
    audit = result.audit

    assert audit.model_id == MODEL_ID
    assert audit.claim_level == CLAIM_LEVEL
    assert audit.topology_class == TOPOLOGY_CLASS
    assert audit.rate_convention == RATE_CONVENTION
    assert audit.physical_thermodynamic_status == PHYSICAL_THERMODYNAMIC_STATUS
    assert audit.minimum_kernel_entry >= -1e-10
    assert audit.minimum_output_mass >= 0.0
    assert audit.generator_column_sum_residual <= 1e-10
    assert audit.kernel_column_sum_residual <= 1e-10
    assert abs(audit.conservation_residual) <= 1e-10
    assert audit.l1_operator_norm == pytest.approx(1.0, abs=1e-10)
    assert audit.output_mass == pytest.approx(
        audit.input_mass * math.exp(-0.2 * 0.6),
        abs=1e-10,
    )


def test_transition_kernel_is_l1_nonexpansive() -> None:
    rates = np.asarray(
        [
            [0.0, 0.2, 0.0],
            [1.0, 0.0, 0.3],
            [0.0, 0.8, 0.0],
        ]
    )
    left = np.asarray([0.7, 0.2, 0.1])
    right = np.asarray([0.1, 0.3, 0.6])
    result = propagate_directed(left, rates, HDSCH1DParameters(transport_time=1.3))
    kernel = np.asarray(result.transition_kernel)

    before = float(np.sum(np.abs(left - right)))
    after = float(np.sum(np.abs(kernel @ left - kernel @ right)))
    assert after <= before + 1e-12


def test_directed_semigroup_composes() -> None:
    rates = np.asarray(
        [
            [0.0, 0.0, 0.4],
            [1.2, 0.0, 0.0],
            [0.0, 0.7, 0.0],
        ]
    )
    source = np.asarray([1.0, 0.0, 0.0])
    first = propagate_directed(source, rates, HDSCH1DParameters(transport_time=0.3))
    composed = propagate_directed(
        np.asarray(first.transported_mass),
        rates,
        HDSCH1DParameters(transport_time=0.8),
    )
    combined = propagate_directed(source, rates, HDSCH1DParameters(transport_time=1.1))

    assert composed.transported_mass == pytest.approx(combined.transported_mass, abs=1e-12)


def test_asymmetric_topology_is_not_sent_to_symmetric_h1() -> None:
    rates = np.asarray([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(HDSCInvariantError, match="symmetric"):
        propagate(np.asarray([1.0, 0.0]), rates, HDSCH1Parameters())

    directed = propagate_directed(np.asarray([1.0, 0.0]), rates)
    assert directed.output_mass[1] > 0.0
    assert directed.output_mass[0] < 1.0


def test_nonreversible_cycle_uses_stationary_relative_entropy_not_dirichlet_energy() -> None:
    rates = _cycle_rates(3, clockwise=2.0, counterclockwise=1.0)
    stationary = np.full(3, 1.0 / 3.0)
    result = propagate_directed(
        np.asarray([1.0, 0.0, 0.0]),
        rates,
        HDSCH1DParameters(transport_time=0.7),
        stationary_distribution=stationary,
    )
    audit = result.audit

    assert audit.stationary_gate == "pass"
    assert audit.relative_entropy_gate == "pass"
    assert audit.relative_entropy_before is not None
    assert audit.relative_entropy_after is not None
    assert audit.relative_entropy_after < audit.relative_entropy_before
    assert audit.reciprocal_support is True
    assert audit.finite_entropy_production_gate == "pass"
    assert audit.entropy_production_rate is not None
    assert audit.entropy_production_rate > 0.0
    assert audit.detailed_balance_residual is not None
    assert audit.detailed_balance_residual > 0.0


def test_one_way_cycle_abstains_from_finite_entropy_production() -> None:
    rates = _cycle_rates(3, clockwise=1.0, counterclockwise=0.0)
    result = propagate_directed(
        np.asarray([1.0, 0.0, 0.0]),
        rates,
        stationary_distribution=np.full(3, 1.0 / 3.0),
    )

    assert result.audit.stationary_gate == "pass"
    assert result.audit.relative_entropy_gate == "pass"
    assert result.audit.reciprocal_support is False
    assert result.audit.finite_entropy_production_gate == "abstain"
    assert result.audit.entropy_production_rate is None


def test_invalid_stationary_distribution_is_a_failed_gate() -> None:
    rates = np.asarray([[0.0, 1.0], [2.0, 0.0]])
    result = propagate_directed(
        np.asarray([1.0, 0.0]),
        rates,
        stationary_distribution=np.asarray([0.5, 0.5]),
    )

    assert result.audit.stationary_gate == "fail"
    assert result.audit.relative_entropy_gate == "not-evaluated"


def test_directed_transport_is_permutation_equivariant() -> None:
    rates = np.asarray(
        [
            [0.0, 0.1, 0.9],
            [0.8, 0.0, 0.2],
            [0.3, 0.7, 0.0],
        ]
    )
    source = np.asarray([0.6, 0.3, 0.1])
    permutation = np.asarray([2, 0, 1])
    parameters = HDSCH1DParameters(transport_time=0.9, sink_rate=0.1)

    original = np.asarray(propagate_directed(source, rates, parameters).output_mass)
    permuted = np.asarray(
        propagate_directed(
            source[permutation],
            rates[np.ix_(permutation, permutation)],
            parameters,
        ).output_mass
    )
    assert permuted == pytest.approx(original[permutation], abs=1e-12)


@pytest.mark.parametrize(
    "rates",
    [
        np.asarray([[0.0, -1.0], [0.0, 0.0]]),
        np.asarray([[0.1, 0.0], [0.0, 0.0]]),
        np.asarray([[0.0, float("nan")], [0.0, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_invalid_directed_rate_matrices_are_rejected(rates: np.ndarray) -> None:
    with pytest.raises(HDSCInvariantError):
        propagate_directed(np.asarray([1.0, 0.0]), rates)


def test_generator_has_zero_columns_and_keeps_direction() -> None:
    rates = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    generator = directed_generator(rates)

    assert np.sum(generator, axis=0) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert generator[1, 0] == pytest.approx(2.0)
    assert generator[0, 1] == 0.0
