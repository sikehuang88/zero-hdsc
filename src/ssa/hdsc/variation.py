"""Deterministic, side-effect-free structural variation for SEOS programs."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from ssa.hdsc.operators import PRIMITIVE_REGISTRY, OperatorRegistry, Ty
from ssa.hdsc.program import Node, Program, ProgramTypeError, check_linearity, type_check


class VariationError(ValueError):
    """Raised when a requested variation cannot preserve program invariants."""


def derive_rng(namespace: str, *parts: object) -> np.random.Generator:
    """Return a deterministic generator with no ambient entropy source."""

    if not namespace.strip():
        raise ValueError("namespace must not be empty")
    key = ":".join(("hdsc-seos-v1", namespace, *(str(part) for part in parts)))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _checked(program: Program, registry: OperatorRegistry) -> Program:
    type_check(program, registry)
    check_linearity(program, registry)
    return program


def _find_node(program: Program, node_id: str) -> Node:
    for node in program.nodes:
        if node.node_id == node_id:
            return node
    raise VariationError(f"unknown node: {node_id}")


def _replacement_nodes(program: Program, replacement: Node) -> tuple[Node, ...]:
    return tuple(
        replacement if node.node_id == replacement.node_id else node
        for node in program.nodes
    )


def mutate_point(
    program: Program,
    node_id: str,
    replacement_operator_id: str,
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> Program:
    """Replace one node while preserving its complete signature and purity."""

    current = _find_node(program, node_id)
    old_operator = registry.require(current.operator_id)
    new_operator = registry.require(replacement_operator_id)
    if old_operator.signature != new_operator.signature:
        raise VariationError("point mutation requires identical signatures")
    if old_operator.purity != new_operator.purity:
        raise VariationError("point mutation cannot change purity")
    replacement = replace(current, operator_id=replacement_operator_id)
    return _checked(replace(program, nodes=_replacement_nodes(program, replacement)), registry)


def mutate_insert(
    program: Program,
    target_node_id: str,
    input_index: int,
    operator_id: str,
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> Program:
    """Insert a pure unary T -> T operator on one typed edge."""

    target = _find_node(program, target_node_id)
    target_operator = registry.require(target.operator_id)
    if not 0 <= input_index < len(target.inputs):
        raise VariationError("input_index is outside the target node arity")
    inserted = registry.require(operator_id)
    expected = target_operator.signature.params[input_index]
    if (
        inserted.purity != "pure"
        or inserted.signature.eff_flux != 0
        or len(inserted.signature.params) != 1
        or inserted.signature.params[0] != expected
        or inserted.signature.result != expected
    ):
        raise VariationError("inserted operator must be a pure unary T -> T operator")
    base_id = f"{target_node_id}__insert__{operator_id}"
    existing = {node.node_id for node in program.nodes}
    inserted_id = base_id
    suffix = 2
    while inserted_id in existing:
        inserted_id = f"{base_id}__{suffix}"
        suffix += 1
    source = target.inputs[input_index]
    inserted_node = Node(inserted_id, operator_id, (source,))
    new_inputs = list(target.inputs)
    new_inputs[input_index] = inserted_id
    rewritten_target = replace(target, inputs=tuple(new_inputs))
    nodes = (
        *(
            rewritten_target if node.node_id == target_node_id else node
            for node in program.nodes
        ),
        inserted_node,
    )
    return _checked(replace(program, nodes=nodes), registry)


def mutate_delete(
    program: Program,
    node_id: str,
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> Program:
    """Delete a pure unary identity-shaped node and reconnect its consumers."""

    target = _find_node(program, node_id)
    operator = registry.require(target.operator_id)
    if (
        operator.purity != "pure"
        or operator.signature.eff_flux != 0
        or len(target.inputs) != 1
        or len(operator.signature.params) != 1
        or operator.signature.params[0] != operator.signature.result
    ):
        raise VariationError("only pure unary T -> T nodes can be deleted")
    source = target.inputs[0]
    nodes: list[Node] = []
    for node in program.nodes:
        if node.node_id == node_id:
            continue
        nodes.append(
            replace(node, inputs=tuple(source if value == node_id else value for value in node.inputs))
        )
    outputs = tuple(source if value == node_id else value for value in program.outputs)
    return _checked(replace(program, nodes=tuple(nodes), outputs=outputs), registry)


def mutate_param(
    program: Program,
    node_id: str,
    parameters: tuple[tuple[str, object], ...],
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> Program:
    """Change only a node's parameter payload; topology and signature stay fixed."""

    current = _find_node(program, node_id)
    replacement = replace(current, parameters=parameters)
    return _checked(replace(program, nodes=_replacement_nodes(program, replacement)), registry)


def crossover_subgraph(
    receiver: Program,
    donor: Program,
    receiver_node_id: str,
    donor_node_id: str,
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> Program:
    """Exchange one-node subgraphs with identical typed interfaces.

    The one-node form is the first safe implementation of SEOS crossover.  A
    future multi-node variant can use the same interface check before remapping
    donor ancestors into the receiver DAG.
    """

    receiver_node = _find_node(receiver, receiver_node_id)
    donor_node = _find_node(donor, donor_node_id)
    receiver_operator = registry.require(receiver_node.operator_id)
    donor_operator = registry.require(donor_node.operator_id)
    if receiver_operator.signature != donor_operator.signature:
        raise VariationError("crossover requires identical subgraph signatures")
    if receiver_operator.purity != donor_operator.purity:
        raise VariationError("crossover cannot change subgraph purity")
    replacement = replace(
        receiver_node,
        operator_id=donor_node.operator_id,
        parameters=donor_node.parameters,
    )
    return _checked(replace(receiver, nodes=_replacement_nodes(receiver, replacement)), registry)


def apply_random_variation(
    program: Program,
    *,
    seed: int,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> Program:
    """Apply one deterministic point variation."""

    checked = _checked(program, registry)
    if not checked.nodes:
        return checked
    rng = derive_rng("mutate", seed, len(checked.nodes))
    node_index = int(rng.integers(0, len(checked.nodes)))
    node = checked.nodes[node_index]
    operator = registry.require(node.operator_id)
    alternatives = tuple(
        candidate.operator_id
        for candidate in registry.values()
        if candidate.operator_id != operator.operator_id
        and candidate.signature == operator.signature
        and candidate.purity == operator.purity
    )
    if alternatives:
        replacement = alternatives[int(rng.integers(0, len(alternatives)))]
        return mutate_point(checked, node.node_id, replacement, registry=registry)
    return checked


def ensure_type(value_type: Ty, expected: Ty) -> None:
    """Validate a proposed edge type before constructing a variation."""

    if value_type != expected:
        raise ProgramTypeError(f"expected {expected!r}, got {value_type!r}")
