"""Property and boundary tests for the HDSC-H2 bounded active state."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ssa.hdsc.active_space import (
    MODEL_ID,
    SUPPORT_METRIC,
    H2ActiveCluster,
    H2Candidate,
    H2Config,
    H2GainBounds,
    H2State,
    bounded_active_shadow_step,
    state_total_variation,
)
from ssa.hdsc.transport import HDSCInvariantError


def _candidate(
    trace_id: str,
    cluster_id: str,
    *,
    relevance: float = 0.8,
    novelty: float = 1.0,
    fingerprint: str | None = None,
    revision: str = "r1",
    prototype: tuple[float, ...] = (1.0, 0.0),
    cluster_assignment_margin: float | None = 1.0,
) -> H2Candidate:
    return H2Candidate(
        trace_id=trace_id,
        semantic_fingerprint=fingerprint or f"fp-{trace_id}",
        cluster_id=cluster_id,
        cluster_revision=revision,
        prototype=prototype,
        relevance=relevance,
        novelty=novelty,
        cluster_assignment_margin=cluster_assignment_margin,
    )


def _cluster(
    cluster_id: str,
    mass: float,
    *,
    prototype: tuple[float, ...] = (1.0, 0.0),
) -> H2ActiveCluster:
    candidate = _candidate("hash-source", cluster_id, prototype=prototype)
    initialized = bounded_active_shadow_step(
        H2State.initial(H2Config(retention=0.0, injection_rate=1.0)),
        [candidate],
        H2Config(retention=0.0, injection_rate=1.0),
    ).state.clusters[0]
    return H2ActiveCluster(
        cluster_id=cluster_id,
        cluster_revision="r1",
        prototype=prototype,
        prototype_hash=initialized.prototype_hash,
        mass=mass,
    )


def _state(config: H2Config, masses: dict[str, float], null_mass: float) -> H2State:
    return H2State(
        step=3,
        clusters=tuple(_cluster(cluster_id, mass) for cluster_id, mass in sorted(masses.items())),
        null_mass=null_mass,
        config_hash=config.config_hash,
    )


def _complete_gains(**updates: float | str | None) -> H2GainBounds:
    values: dict[str, float | str | None] = {
        "response_signal": 0.1,
        "response_memory": 0.1,
        "user_feedback": 0.1,
        "proposal_signal": 0.1,
        "proposal_response": 0.1,
        "proposal_memory": 0.1,
        "user_gain_source": "paired-replay-v1",
    }
    values.update(updates)
    return H2GainBounds(**values)  # type: ignore[arg-type]


def test_exact_duplicate_candidates_do_not_change_behavior_state() -> None:
    config = H2Config()
    initial = H2State.initial(config)
    one = [_candidate("trace-0", "identity", fingerprint="same")]
    repeated = [_candidate(f"trace-{index}", "identity", fingerprint="same") for index in range(20)]

    single = bounded_active_shadow_step(initial, one, config)
    duplicate = bounded_active_shadow_step(initial, repeated, config)

    assert duplicate.state == single.state
    assert duplicate.audit.next_state_hash == single.audit.next_state_hash
    assert duplicate.audit.duplicate_count == 19
    assert duplicate.audit.duplicate_suppressed_mass > 0.0


def test_zero_novelty_duplicate_round_matches_empty_round() -> None:
    config = H2Config()
    initial = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("first", "identity")],
        config,
    ).state

    empty = bounded_active_shadow_step(initial, [], config)
    duplicate = bounded_active_shadow_step(
        initial,
        [_candidate("again", "identity", novelty=0.0)],
        config,
    )

    assert duplicate.state == empty.state


def test_candidate_order_is_irrelevant() -> None:
    config = H2Config(capacity=4, candidate_top_k=3)
    candidates = [
        _candidate("a", "alpha", relevance=0.9, prototype=(1.0, 0.0)),
        _candidate("b", "beta", relevance=0.7, prototype=(0.0, 1.0)),
        _candidate("c", "gamma", relevance=0.5, prototype=(0.6, 0.8)),
    ]

    forward = bounded_active_shadow_step(H2State.initial(config), candidates, config)
    reverse = bounded_active_shadow_step(
        H2State.initial(config),
        list(reversed(candidates)),
        config,
    )

    assert forward.state == reverse.state
    assert forward.proposal_mass == reverse.proposal_mass


@given(st.lists(st.integers(min_value=0, max_value=40), min_size=0, max_size=80))
@settings(max_examples=35, deadline=None)
def test_archive_growth_never_expands_active_capacity(cluster_numbers: list[int]) -> None:
    config = H2Config(capacity=3, candidate_top_k=2)
    state = H2State.initial(config)

    for index, number in enumerate(cluster_numbers):
        angle = number / 40.0 * math.pi / 2.0
        state = bounded_active_shadow_step(
            state,
            [
                _candidate(
                    f"trace-{index}",
                    f"cluster-{number}",
                    fingerprint=f"fp-{index}",
                    prototype=(math.cos(angle), math.sin(angle)),
                )
            ],
            config,
            archive_trace_count=index + 1,
        ).state
        assert len(state.clusters) <= config.capacity
        masses = [state.null_mass, *(cluster.mass for cluster in state.clusters)]
        assert all(math.isfinite(mass) and mass >= 0.0 for mass in masses)
        assert sum(masses) == pytest.approx(1.0, abs=1e-10)


def test_capacity_overflow_moves_to_null_without_renormalizing_survivor() -> None:
    config = H2Config(
        capacity=1,
        candidate_top_k=1,
        retention=0.5,
        injection_rate=0.4,
        hysteresis=0.0,
    )
    initial = _state(config, {"old": 0.8}, null_mass=0.2)

    result = bounded_active_shadow_step(
        initial,
        [_candidate("new", "new", relevance=1.0, prototype=(0.0, 1.0))],
        config,
    )

    assert result.state.clusters[0].cluster_id == "new"
    assert result.state.clusters[0].mass == pytest.approx(0.4)
    assert result.state.null_mass == pytest.approx(0.6)
    assert result.audit.dropped_mass == pytest.approx(0.4)


def test_fixed_support_empty_update_contracts_by_retention() -> None:
    config = H2Config(capacity=2, candidate_top_k=1, retention=0.6, injection_rate=0.2)
    left = _state(config, {"a": 0.7, "b": 0.1}, null_mass=0.2)
    right = _state(config, {"a": 0.2, "b": 0.6}, null_mass=0.2)
    before = state_total_variation(left, right)

    left_next = bounded_active_shadow_step(left, [], config).state
    right_next = bounded_active_shadow_step(right, [], config).state

    assert state_total_variation(left_next, right_next) == pytest.approx(
        config.retention * before,
        abs=1e-12,
    )


def test_hysteresis_reduces_marginal_support_churn() -> None:
    with_hysteresis = H2Config(
        capacity=1,
        candidate_top_k=1,
        retention=0.5,
        injection_rate=0.45,
        hysteresis=0.1,
    )
    without_hysteresis = H2Config(
        capacity=1,
        candidate_top_k=1,
        retention=0.5,
        injection_rate=0.45,
        hysteresis=0.0,
    )
    candidate = _candidate("new", "new", relevance=1.0, prototype=(0.0, 1.0))

    kept = bounded_active_shadow_step(
        _state(with_hysteresis, {"old": 0.8}, null_mass=0.2),
        [candidate],
        with_hysteresis,
    )
    replaced = bounded_active_shadow_step(
        _state(without_hysteresis, {"old": 0.8}, null_mass=0.2),
        [candidate],
        without_hysteresis,
    )

    assert kept.state.clusters[0].cluster_id == "old"
    assert replaced.state.clusters[0].cluster_id == "new"
    assert kept.audit.support_churn == 0.0
    assert replaced.audit.support_churn == 1.0


def test_top_k_tie_abstains_from_stability_certificate() -> None:
    config = H2Config(capacity=2, candidate_top_k=1)
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [
            _candidate("a", "a", relevance=0.8),
            _candidate("b", "b", relevance=0.8, prototype=(0.0, 1.0)),
        ],
        config,
        gains=_complete_gains(),
    )

    assert result.audit.proposal_boundary_gap == 0.0
    assert result.audit.proposal_tie_count == 2
    assert result.audit.certificate_status == "abstain"
    assert "no positive local margin" in result.audit.certificate_reason


def test_zero_score_entry_boundary_has_zero_certified_radius() -> None:
    config = H2Config(capacity=2, candidate_top_k=2)
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [
            _candidate("a", "a", novelty=1.0),
            _candidate("b", "b", novelty=0.0, prototype=(0.0, 1.0)),
        ],
        config,
        gains=_complete_gains(),
    )

    assert result.audit.proposal_boundary_gap == 0.0
    assert result.audit.certified_radius == 0.0
    assert result.audit.certificate_status == "abstain"


def test_capacity_margin_uses_hysteresis_adjusted_selection_priority() -> None:
    config = H2Config(
        capacity=1,
        candidate_top_k=1,
        retention=0.5,
        injection_rate=0.45,
        hysteresis=0.02,
    )
    result = bounded_active_shadow_step(
        _state(config, {"old": 0.86}, null_mass=0.14),
        [_candidate("new", "new", relevance=1.0, prototype=(0.0, 1.0))],
        config,
        gains=_complete_gains(),
    )

    assert result.audit.capacity_boundary_gap == pytest.approx(0.0, abs=1e-12)
    assert result.audit.capacity_certified_radius == pytest.approx(0.0, abs=1e-12)
    assert result.audit.certified_radius == pytest.approx(0.0, abs=1e-12)
    assert result.audit.certificate_status == "abstain"


def test_combined_radius_is_limited_by_capacity_support_margin() -> None:
    config = H2Config(
        capacity=1,
        candidate_top_k=1,
        retention=0.5,
        injection_rate=0.45,
        hysteresis=0.02,
    )
    result = bounded_active_shadow_step(
        _state(config, {"old": 0.84}, null_mass=0.16),
        [_candidate("new", "new", relevance=1.0, prototype=(0.0, 1.0))],
        config,
        gains=_complete_gains(),
    )

    assert result.audit.capacity_boundary_gap == pytest.approx(0.01)
    expected_gain = config.retention + config.injection_rate
    assert result.audit.capacity_certified_radius == pytest.approx(0.01 / (2 * expected_gain))
    assert result.audit.certified_radius == pytest.approx(0.01 / (2 * expected_gain))
    assert result.audit.certificate_status == "pass"


def test_proposal_normalization_gain_limits_small_score_capacity_radius() -> None:
    config = H2Config(
        capacity=2,
        candidate_top_k=2,
        retention=0.8,
        injection_rate=0.15,
        hysteresis=0.0,
    )
    old_mass = 0.074999 / config.retention
    result = bounded_active_shadow_step(
        _state(config, {"old": old_mass}, null_mass=1.0 - old_mass),
        [
            _candidate("a", "a", relevance=1e-6),
            _candidate("b", "b", relevance=0.999999e-6, prototype=(0.0, 1.0)),
        ],
        config,
        gains=_complete_gains(),
    )

    assert result.audit.proposal_total_score == pytest.approx(1.999999e-6)
    assert result.audit.proposal_normalization_gain == pytest.approx(2 / 1.999999e-6)
    assert result.audit.certified_radius < 4e-7
    assert result.audit.certificate_status == "abstain"
    assert "below tolerance" in result.audit.certificate_reason


@pytest.mark.parametrize("margin", [None, 0.0])
def test_cluster_assignment_boundary_abstains(margin: float | None) -> None:
    config = H2Config()
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("a", "a", cluster_assignment_margin=margin)],
        config,
        gains=_complete_gains(),
    )

    assert result.audit.cluster_assignment_margin == margin
    assert result.audit.certified_radius == 0.0
    assert result.audit.certificate_status == "abstain"


def test_small_gain_certificate_requires_measured_user_gain() -> None:
    config = H2Config()
    candidate = [_candidate("a", "a")]

    missing = bounded_active_shadow_step(
        H2State.initial(config),
        candidate,
        config,
        gains=_complete_gains(user_feedback=None),
    )
    measured = bounded_active_shadow_step(
        H2State.initial(config),
        candidate,
        config,
        gains=_complete_gains(),
    )

    assert missing.audit.certificate_status == "abstain"
    assert measured.audit.certificate_status == "pass"
    assert measured.audit.small_gain_spectral_radius is not None
    assert measured.audit.small_gain_spectral_radius < 1.0 - config.gain_margin


def test_small_gain_certificate_requires_a_measured_user_gain_source() -> None:
    config = H2Config()
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("a", "a")],
        config,
        gains=_complete_gains(user_gain_source="unmeasured"),
    )

    assert result.audit.certificate_status == "abstain"
    assert result.audit.small_gain_matrix is None
    assert "source is unmeasured" in result.audit.certificate_reason


@pytest.mark.parametrize("invalid_gain", [float("nan"), float("inf"), -0.1])
def test_invalid_gain_abstains_without_crashing_shadow(invalid_gain: float) -> None:
    config = H2Config()
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("a", "a")],
        config,
        gains=_complete_gains(response_signal=invalid_gain),
    )

    assert result.audit.certificate_status == "abstain"
    assert result.audit.small_gain_matrix is None
    assert "non-finite or negative" in result.audit.certificate_reason


def test_gain_composition_overflow_abstains_with_component_audit() -> None:
    config = H2Config()
    gains = H2GainBounds(
        response_signal=1e308,
        response_memory=1e308,
        user_feedback=1e308,
        proposal_signal=1e308,
        proposal_response=1e308,
        proposal_memory=1e308,
        user_gain_source="stress-bound-v1",
    )
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("a", "a")],
        config,
        gains=gains,
    )

    assert result.audit.certificate_status == "abstain"
    assert result.audit.small_gain_matrix is None
    assert "overflowed" in result.audit.certificate_reason
    assert result.audit.response_signal_gain == 1e308
    assert result.audit.proposal_memory_gain == 1e308


def test_exhausted_small_gain_margin_fails_certificate() -> None:
    config = H2Config()
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("a", "a")],
        config,
        gains=_complete_gains(response_signal=1.0, user_feedback=2.0),
    )

    assert result.audit.certificate_status == "fail"
    assert result.audit.small_gain_spectral_radius is not None
    assert result.audit.small_gain_spectral_radius >= 1.0 - config.gain_margin


def test_archive_count_is_observational_and_excluded_from_state_hash() -> None:
    config = H2Config()
    candidate = [_candidate("a", "a")]

    small_archive = bounded_active_shadow_step(
        H2State.initial(config), candidate, config, archive_trace_count=1
    )
    large_archive = bounded_active_shadow_step(
        H2State.initial(config), candidate, config, archive_trace_count=1_000_000
    )

    assert small_archive.state.state_hash == large_archive.state.state_hash
    assert small_archive.audit.archive_trace_count == 1
    assert large_archive.audit.archive_trace_count == 1_000_000


def test_behavior_measure_hash_excludes_replay_step() -> None:
    config = H2Config()
    initial = H2State.initial(config)
    advanced = bounded_active_shadow_step(initial, [], config).state

    assert state_total_variation(initial, advanced) == 0.0
    assert advanced.state_hash == initial.state_hash
    assert advanced.replay_hash != initial.replay_hash


def test_semantic_partition_changes_configuration_hash() -> None:
    first = H2Config(semantic_partition_id="partition-a")
    second = H2Config(semantic_partition_id="partition-b")

    assert first.config_hash != second.config_hash


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate("bad", "bad", relevance=float("nan")),
        _candidate("bad", "bad", novelty=-0.1),
        _candidate("bad", "bad", prototype=(2.0, 0.0)),
        _candidate("bad", "bad", prototype=(0.0, 0.0)),
        _candidate("bad", "bad", cluster_assignment_margin=float("nan")),
        _candidate("bad", "bad", cluster_assignment_margin=-0.1),
        _candidate("", "bad"),
    ],
)
def test_invalid_candidate_is_rejected(candidate: H2Candidate) -> None:
    config = H2Config()
    with pytest.raises(HDSCInvariantError):
        bounded_active_shadow_step(H2State.initial(config), [candidate], config)


def test_conflicting_fingerprint_and_prototype_revision_are_rejected() -> None:
    config = H2Config()
    with pytest.raises(HDSCInvariantError, match="fingerprint"):
        bounded_active_shadow_step(
            H2State.initial(config),
            [
                _candidate("a", "a", fingerprint="same"),
                _candidate("b", "b", fingerprint="same", prototype=(0.0, 1.0)),
            ],
            config,
        )
    with pytest.raises(HDSCInvariantError, match="conflicting prototypes"):
        bounded_active_shadow_step(
            H2State.initial(config),
            [
                _candidate("a", "a", fingerprint="one"),
                _candidate("b", "a", fingerprint="two", prototype=(0.0, 1.0)),
            ],
            config,
        )


def test_state_config_and_prototype_dimension_mismatches_are_rejected() -> None:
    first = H2Config(capacity=2, candidate_top_k=1)
    second = H2Config(capacity=3, candidate_top_k=1)
    with pytest.raises(HDSCInvariantError, match="config hash"):
        bounded_active_shadow_step(H2State.initial(first), [], second)
    with pytest.raises(HDSCInvariantError, match="share one dimension"):
        bounded_active_shadow_step(
            H2State.initial(first),
            [
                _candidate("a", "a", prototype=(1.0, 0.0)),
                _candidate("b", "b", prototype=(1.0, 0.0, 0.0)),
            ],
            first,
        )


def test_behavior_measure_hash_is_independent_of_cluster_tuple_order() -> None:
    config = H2Config(capacity=2, candidate_top_k=2)
    canonical = _state(config, {"a": 0.3, "b": 0.4}, null_mass=0.3)
    reversed_state = H2State(
        step=canonical.step,
        clusters=tuple(reversed(canonical.clusters)),
        null_mass=canonical.null_mass,
        config_hash=canonical.config_hash,
    )

    assert state_total_variation(canonical, reversed_state) == 0.0
    assert canonical.state_hash == reversed_state.state_hash


def test_audit_declares_shadow_scope_and_mass_invariants() -> None:
    config = H2Config()
    result = bounded_active_shadow_step(
        H2State.initial(config),
        [_candidate("a", "a")],
        config,
    )

    assert result.audit.model_id == MODEL_ID
    assert result.audit.shadow_only is True
    assert result.audit.semantic_partition_id == config.semantic_partition_id
    assert result.audit.previous_replay_hash != result.audit.next_replay_hash
    assert result.audit.modeled_scale == "macro-coarse-grained"
    assert result.audit.microstate_status == "unmodeled"
    assert result.audit.transport_model == "identity-nonexpansive"
    assert result.audit.support_metric == SUPPORT_METRIC
    assert result.audit.total_mass == pytest.approx(1.0, abs=1e-12)
    assert abs(result.audit.mass_residual) <= 1e-12
    assert result.audit.minimum_mass >= 0.0
