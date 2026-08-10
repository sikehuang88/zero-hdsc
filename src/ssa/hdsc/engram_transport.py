"""Discrete typed transition-power transport for the ENGRAM graph."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from ssa.domain.engram import (
    EngramActivation,
    EngramEdge,
    EngramPathStep,
    EngramQueryResult,
    EngramRelation,
    EngramTransportAudit,
)


class EngramTransportError(ValueError):
    """Raised when a typed transition graph violates transport invariants."""


TransportMode = Literal["truncated_power", "ppr", "bounded_active"]


@dataclass(frozen=True, slots=True)
class EngramTransportConfig:
    mode: TransportMode = "truncated_power"
    max_hops: int = 3
    restart_probability: float = 0.20
    epsilon: float = 1e-6
    max_active: int = 64
    max_results: int = 32
    action: str = "default"

    def __post_init__(self) -> None:
        if not 1 <= self.max_hops <= 32:
            raise ValueError("engram max_hops must be in [1, 32]")
        if not 0.0 < self.restart_probability <= 1.0:
            raise ValueError("engram restart_probability must be in (0, 1]")
        if not 0.0 <= self.epsilon <= 1.0 or not math.isfinite(self.epsilon):
            raise ValueError("engram epsilon must be finite in [0, 1]")
        if not 1 <= self.max_active <= 4096:
            raise ValueError("engram max_active must be in [1, 4096]")
        if not 1 <= self.max_results <= 256:
            raise ValueError("engram max_results must be in [1, 256]")
        if not self.action.strip():
            raise ValueError("engram action must not be blank")


def propagate_engram(
    node_ids: list[str],
    edges: list[EngramEdge],
    seed_masses: dict[str, float],
    *,
    relation_gates: dict[EngramRelation, float] | None = None,
    config: EngramTransportConfig | None = None,
) -> EngramQueryResult:
    """Propagate seed mass through column-normalized typed directed powers.

    ``matrix[target][source]`` is the convention. No reverse edge is created;
    path state is carried alongside mass and the best supporting path is retained.
    """
    effective = config or EngramTransportConfig()
    _validate_nodes(node_ids)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    seeds = _validate_seeds(seed_masses, index)
    gates = relation_gates or {}
    transitions, transition_edges = _build_transition_columns(
        node_ids,
        edges,
        index,
        gates,
        effective,
    )

    current = dict(seeds)
    current_paths: dict[str, tuple[EngramPathStep, ...]] = dict.fromkeys(seeds, ())
    score_by_node: dict[str, float] = defaultdict(float)
    best_path_by_node: dict[str, tuple[EngramPathStep, ...]] = {}
    best_contribution: dict[str, float] = {}
    null_mass = 0.0
    seed_mass = sum(seeds.values())

    for hop in range(effective.max_hops + 1):
        factor = _hop_factor(effective, hop)
        for node_id, mass in current.items():
            contribution = factor * mass
            score_by_node[node_id] += contribution
            if contribution > best_contribution.get(node_id, -1.0):
                best_contribution[node_id] = contribution
                best_path_by_node[node_id] = current_paths.get(node_id, ())
        if hop == effective.max_hops:
            break
        next_mass, next_paths = _advance(
            current,
            current_paths,
            transitions,
            transition_edges,
            gates,
        )
        if effective.mode == "bounded_active":
            next_mass, next_paths, dropped = _retain_active(
                next_mass,
                next_paths,
                effective.max_active,
            )
            null_mass += dropped
        current, current_paths = next_mass, next_paths

    if effective.mode == "ppr":
        null_mass += seed_mass * (1.0 - effective.restart_probability) ** (effective.max_hops + 1)
        final_mass = sum(score_by_node.values()) + null_mass
    else:
        final_mass = sum(current.values()) + null_mass
    activations = [
        EngramActivation(
            node_id=node_id,
            mass=mass,
            hops=len(best_path_by_node.get(node_id, ())),
            path=list(best_path_by_node.get(node_id, ())),
            is_assertable=_is_assertable(node_id, seeds, best_path_by_node),
        )
        for node_id, mass in sorted(score_by_node.items(), key=lambda item: (-item[1], item[0]))
        if mass > 0.0
    ][: effective.max_results]
    audit = EngramTransportAudit(
        mode=effective.mode,
        node_count=len(node_ids),
        edge_count=len(transition_edges),
        seed_mass=seed_mass,
        propagated_mass=sum(score_by_node.values()),
        null_mass=null_mass,
        conservation_residual=seed_mass - final_mass,
        max_hops=effective.max_hops,
        path_grounded_count=sum(bool(item.path) for item in activations),
    )
    return EngramQueryResult(activations=activations, audit=audit)


def _build_transition_columns(
    node_ids: list[str],
    edges: list[EngramEdge],
    index: dict[str, int],
    gates: dict[EngramRelation, float],
    config: EngramTransportConfig,
) -> tuple[dict[str, list[tuple[str, float]]], list[EngramEdge]]:
    columns: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in node_ids}
    effective_edges: list[EngramEdge] = []
    for edge in edges:
        if edge.action != config.action:
            continue
        if edge.src not in index or edge.dst not in index:
            raise EngramTransportError("edge references a node outside the topology")
        gate = float(gates.get(edge.rel_type, 1.0))
        if gate < 0.0 or not math.isfinite(gate):
            raise EngramTransportError("relation gates must be finite and non-negative")
        weight = edge.transport_weight(gate)
        if weight <= 0.0:
            continue
        columns[edge.src].append((edge.dst, weight))
        effective_edges.append(edge)
    for node_id in node_ids:
        columns[node_id].append((node_id, config.epsilon))
        total = sum(weight for _, weight in columns[node_id])
        columns[node_id] = [(target, weight / total) for target, weight in columns[node_id]]
    return columns, effective_edges


def _advance(
    current: dict[str, float],
    paths: dict[str, tuple[EngramPathStep, ...]],
    transitions: dict[str, list[tuple[str, float]]],
    edges: list[EngramEdge],
    gates: dict[EngramRelation, float],
) -> tuple[dict[str, float], dict[str, tuple[EngramPathStep, ...]]]:
    edge_lookup: dict[tuple[str, str], list[tuple[EngramEdge, float]]] = defaultdict(list)
    for edge in edges:
        edge_lookup[(edge.src, edge.dst)].append(
            (edge, edge.transport_weight(gates.get(edge.rel_type, 1.0)))
        )
    next_mass: dict[str, float] = defaultdict(float)
    next_paths: dict[str, tuple[EngramPathStep, ...]] = {}
    best_path_mass: dict[str, float] = {}
    for source, mass in current.items():
        for target, probability in transitions[source]:
            amount = mass * probability
            next_mass[target] += amount
            if source == target:
                candidate_path = paths.get(source, ())
            else:
                candidate_edges = edge_lookup[(source, target)]
                edge, _weight = max(candidate_edges, key=lambda item: item[1])
                candidate_path = (
                    *paths.get(source, ()),
                    EngramPathStep(
                        edge_src=edge.src,
                        edge_dst=edge.dst,
                        rel_type=edge.rel_type,
                        source_event_id=edge.source_event_id,
                        support=edge.support,
                    ),
                )
            if amount > best_path_mass.get(target, -1.0):
                best_path_mass[target] = amount
                next_paths[target] = candidate_path
    return dict(next_mass), next_paths


def _retain_active(
    masses: dict[str, float],
    paths: dict[str, tuple[EngramPathStep, ...]],
    max_active: int,
) -> tuple[dict[str, float], dict[str, tuple[EngramPathStep, ...]], float]:
    ranked = sorted(masses.items(), key=lambda item: (-item[1], item[0]))
    keep = dict(ranked[:max_active])
    dropped = sum(mass for node_id, mass in ranked[max_active:])
    return keep, {node_id: paths.get(node_id, ()) for node_id in keep}, dropped


def _hop_factor(config: EngramTransportConfig, hop: int) -> float:
    if config.mode != "ppr":
        return 1.0
    return config.restart_probability * (1.0 - config.restart_probability) ** hop


def _is_assertable(
    node_id: str,
    seeds: dict[str, float],
    paths: dict[str, tuple[EngramPathStep, ...]],
) -> bool:
    if node_id in seeds:
        return True
    path = paths.get(node_id, ())
    return len(path) == 1 and path[0].source_event_id is not None


def _validate_nodes(node_ids: list[str]) -> None:
    if not node_ids or any(not node_id.strip() for node_id in node_ids):
        raise EngramTransportError("node_ids must contain at least one non-empty identifier")
    if len(set(node_ids)) != len(node_ids):
        raise EngramTransportError("node_ids must be unique")


def _validate_seeds(seed_masses: dict[str, float], index: dict[str, int]) -> dict[str, float]:
    if not seed_masses:
        raise EngramTransportError("seed_masses must not be empty")
    result: dict[str, float] = {}
    for node_id, mass in seed_masses.items():
        if node_id not in index:
            raise EngramTransportError("seed mass references an unknown node")
        if mass < 0.0 or not math.isfinite(mass):
            raise EngramTransportError("seed masses must be finite and non-negative")
        if mass > 0.0:
            result[node_id] = float(mass)
    if not result:
        raise EngramTransportError("seed_masses must contain positive mass")
    return result


__all__ = ["EngramTransportConfig", "EngramTransportError", "propagate_engram"]
