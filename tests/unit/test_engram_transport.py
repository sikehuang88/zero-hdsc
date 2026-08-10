"""Typed non-symmetric transition powers and path provenance."""

from __future__ import annotations

import pytest

from ssa.config import Environment, load_settings
from ssa.domain.engram import EngramEdge, EngramRelation
from ssa.hdsc.engram_transport import EngramTransportConfig, propagate_engram


def _edge(
    edge_id: str,
    src: str,
    dst: str,
    relation: EngramRelation,
    event_id: str,
) -> EngramEdge:
    return EngramEdge(
        edge_id=edge_id,
        src=src,
        dst=dst,
        rel_type=relation,
        support=1.0,
        trust_weight=1.0,
        source_event_id=event_id,
        valid_from_ms=1,
    )


def test_two_hop_activation_retains_typed_path_and_honesty_gate() -> None:
    result = propagate_engram(
        ["A", "B", "C", "D"],
        [
            _edge("edge-ab", "A", "B", EngramRelation.SEMANTIC, "event-ab"),
            _edge("edge-bc", "B", "C", EngramRelation.ENTITY, "event-bc"),
        ],
        {"A": 1.0},
        config=EngramTransportConfig(max_hops=2, epsilon=1e-6),
    )
    activation = {item.node_id: item for item in result.activations}

    assert activation["B"].hops == 1
    assert activation["B"].is_assertable is True
    assert activation["C"].hops == 2
    assert [step.rel_type for step in activation["C"].path] == [
        EngramRelation.SEMANTIC,
        EngramRelation.ENTITY,
    ]
    assert [step.source_event_id for step in activation["C"].path] == [
        "event-ab",
        "event-bc",
    ]
    assert activation["C"].is_assertable is False
    assert "D" not in activation


def test_reverse_reachability_is_not_invented_and_relation_gate_can_close_path() -> None:
    edges = [
        _edge("edge-ab", "A", "B", EngramRelation.SEMANTIC, "event-ab"),
        _edge("edge-bc", "B", "C", EngramRelation.ENTITY, "event-bc"),
    ]
    reverse = propagate_engram(
        ["A", "B", "C"],
        edges,
        {"C": 1.0},
        config=EngramTransportConfig(max_hops=2),
    )
    gated = propagate_engram(
        ["A", "B", "C"],
        edges,
        {"A": 1.0},
        relation_gates={EngramRelation.ENTITY: 0.0},
        config=EngramTransportConfig(max_hops=2),
    )

    assert {item.node_id for item in reverse.activations} == {"C"}
    assert "C" not in {item.node_id for item in gated.activations}


def test_path_provenance_uses_transport_weight_for_parallel_typed_edges() -> None:
    dominant = EngramEdge(
        edge_id="edge-low-gate",
        src="A",
        dst="B",
        rel_type=EngramRelation.SEMANTIC,
        support=10.0,
        trust_weight=1.0,
        source_event_id="semantic-evidence",
        valid_from_ms=1,
    )
    gated = EngramEdge(
        edge_id="edge-high-gate",
        src="A",
        dst="B",
        rel_type=EngramRelation.ENTITY,
        support=2.0,
        trust_weight=1.0,
        source_event_id="entity-evidence",
        valid_from_ms=1,
    )
    result = propagate_engram(
        ["A", "B"],
        [dominant, gated],
        {"A": 1.0},
        relation_gates={EngramRelation.SEMANTIC: 0.1, EngramRelation.ENTITY: 1.0},
        config=EngramTransportConfig(max_hops=1),
    )

    activation = {item.node_id: item for item in result.activations}["B"]
    assert activation.path[0].rel_type is EngramRelation.ENTITY
    assert activation.path[0].source_event_id == "entity-evidence"


def test_ppr_and_bounded_active_account_for_tail_and_dropped_mass() -> None:
    edges = [
        _edge("edge-ab", "A", "B", EngramRelation.SEMANTIC, "event-ab"),
        _edge("edge-ac", "A", "C", EngramRelation.ENTITY, "event-ac"),
        _edge("edge-ad", "A", "D", EngramRelation.CATEGORY, "event-ad"),
    ]
    ppr = propagate_engram(
        ["A", "B", "C", "D"],
        edges,
        {"A": 1.0},
        config=EngramTransportConfig(mode="ppr", max_hops=3, restart_probability=0.25),
    )
    bounded = propagate_engram(
        ["A", "B", "C", "D"],
        edges,
        {"A": 1.0},
        config=EngramTransportConfig(mode="bounded_active", max_hops=2, max_active=1),
    )

    assert ppr.audit.null_mass == pytest.approx(0.75**4)
    assert ppr.audit.conservation_residual == pytest.approx(0.0, abs=1e-10)
    assert bounded.audit.null_mass > 0.0
    assert bounded.audit.conservation_residual == pytest.approx(0.0, abs=1e-10)


def test_engram_switch_is_off_by_default_and_selects_kernel_from_env() -> None:
    defaults = load_settings(Environment.DEVELOPMENT, environ={})
    configured = load_settings(
        Environment.DEVELOPMENT,
        environ={
            "HDSC_ENGRAM_ENABLED": "true",
            "HDSC_ENGRAM_MODE": "ppr",
            "HDSC_ENGRAM_MAX_HOPS": "5",
        },
    )

    assert defaults.engram.enabled is False
    assert configured.engram.enabled is True
    assert configured.engram.mode.value == "ppr"
    assert configured.engram.max_hops == 5
