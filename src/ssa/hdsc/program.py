"""Typed DAG programs and P0 validation for the SEOS operator space.

Programs are immutable data.  This module only validates and fingerprints them;
it does not execute operators or expose effect handles.  That separation keeps
candidate evaluation side-effect free by construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from ssa.hdsc.operators import (
    EFF,
    PRIMITIVE_REGISTRY,
    Base,
    Coll,
    Fn,
    OperatorRegistry,
    Ty,
)

HASH_NAMESPACE: Final = "hdsc-seos-genotype-v1:"
MODEL_ID: Final = "hdsc-seos-v0-shadow"
CLAIM_LEVEL: Final = "C0-A-bounded-program-space"
SEARCH_SCOPE: Final = "typed-dag-over-fixed-primitive-basis"
CAPABILITY_ENVELOPE: Final = "primitive-basis-closure"


class ProgramTypeError(ValueError):
    """Raised when a program is not a well-typed DAG."""


class LinearityError(ProgramTypeError):
    """Raised when Eff values are copied, dropped, or merged."""


@dataclass(frozen=True, slots=True)
class ProgramInput:
    """A named input value available to a program."""

    name: str
    type: Ty

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("program input name must not be empty")


@dataclass(frozen=True, slots=True)
class Node:
    """One immutable operator node in a program DAG.

    ``inputs`` contains names of program inputs or earlier/later node IDs.  The
    checker computes a topological order, so serialized nodes do not need to be
    physically sorted.
    """

    node_id: str
    operator_id: str
    inputs: tuple[str, ...]
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must not be empty")
        if not self.operator_id.strip():
            raise ValueError("operator_id must not be empty")
        names = [name for name, _ in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("node parameter names must be unique")


@dataclass(frozen=True, slots=True)
class Program:
    """An immutable typed DAG and its output references."""

    inputs: tuple[ProgramInput, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]
    schema_version: str = HASH_NAMESPACE

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")

    @property
    def eff_flux(self) -> int:
        return sum(item.type == EFF for item in self.inputs)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        """Return the longest input-to-output path length."""

        depths: dict[str, int] = {item.name: 0 for item in self.inputs}
        for node in _topological_nodes(self):
            depths[node.node_id] = 1 + max(
                (depths.get(reference, 0) for reference in node.inputs),
                default=0,
            )
        return max((depths.get(output, 0) for output in self.outputs), default=0)


def _node_map(program: Program) -> dict[str, Node]:
    result: dict[str, Node] = {}
    input_names = {item.name for item in program.inputs}
    if len(input_names) != len(program.inputs):
        raise ProgramTypeError("duplicate program input name")
    for node in program.nodes:
        if node.node_id in input_names:
            raise ProgramTypeError(f"node/input name collision: {node.node_id}")
        if node.node_id in result:
            raise ProgramTypeError(f"duplicate node id: {node.node_id}")
        result[node.node_id] = node
    return result


def _topological_nodes(program: Program) -> tuple[Node, ...]:
    nodes = _node_map(program)
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[Node] = []

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ProgramTypeError(f"cycle detected at node: {node_id}")
        node = nodes.get(node_id)
        if node is None:
            raise ProgramTypeError(f"unknown node reference: {node_id}")
        visiting.add(node_id)
        for reference in node.inputs:
            if reference in nodes:
                visit(reference)
        visiting.remove(node_id)
        visited.add(node_id)
        ordered.append(node)

    for node in program.nodes:
        visit(node.node_id)
    return tuple(ordered)


def type_check(program: Program, registry: OperatorRegistry = PRIMITIVE_REGISTRY) -> None:
    """Raise ``ProgramTypeError`` unless every edge is type-compatible."""

    nodes = _node_map(program)
    input_types = {item.name: item.type for item in program.inputs}
    value_types: dict[str, Ty] = dict(input_types)
    for node in _topological_nodes(program):
        operator = registry.get(node.operator_id)
        if operator is None:
            raise ProgramTypeError(f"unknown operator: {node.operator_id}")
        signature = operator.signature
        if len(node.inputs) != len(signature.params):
            raise ProgramTypeError(
                f"{node.node_id}: expected {len(signature.params)} inputs, "
                f"got {len(node.inputs)}"
            )
        for index, (reference, expected) in enumerate(
            zip(node.inputs, signature.params, strict=True)
        ):
            actual = value_types.get(reference)
            if actual is None:
                if reference in nodes:
                    raise ProgramTypeError(
                        f"{node.node_id}: dependency {reference} was not resolved"
                    )
                raise ProgramTypeError(f"{node.node_id}: unknown input {reference}")
            if actual != expected:
                raise ProgramTypeError(
                    f"{node.node_id}: input {index} expects {expected!r}, got {actual!r}"
                )
        value_types[node.node_id] = signature.result
    for output in program.outputs:
        if output not in value_types:
            raise ProgramTypeError(f"unknown program output: {output}")


def check_linearity(program: Program, registry: OperatorRegistry = PRIMITIVE_REGISTRY) -> None:
    """Raise ``LinearityError`` unless all ``Eff`` edges form linear chains.

    The current SEOS candidate model permits zero or one effect token per
    program.  Every token must be consumed exactly once by an effecting node;
    candidates therefore have ``eff_flux == 0`` and contain no effecting node.
    """

    type_check(program, registry)
    input_types = {item.name: item.type for item in program.inputs}
    value_types: dict[str, Ty] = dict(input_types)
    eff_values: set[str] = {name for name, value_type in input_types.items() if value_type == EFF}
    eff_consumers: dict[str, int] = {}
    effecting_nodes = 0

    for node in _topological_nodes(program):
        operator = registry.require(node.operator_id)
        if operator.purity == "effecting":
            effecting_nodes += 1
        eff_positions = [
            index for index, expected in enumerate(operator.signature.params) if expected == EFF
        ]
        if len(eff_positions) > 1:
            raise LinearityError(f"{node.node_id}: multiple Eff inputs would merge a chain")
        for index, reference in enumerate(node.inputs):
            actual = value_types[reference]
            if actual == EFF:
                eff_values.add(reference)
                if index not in eff_positions:
                    raise LinearityError(f"{node.node_id}: Eff passed to a non-Eff parameter")
                eff_consumers[reference] = eff_consumers.get(reference, 0) + 1
        value_types[node.node_id] = operator.signature.result
        if operator.signature.result == EFF:
            eff_values.add(node.node_id)

    if not eff_values:
        if effecting_nodes:
            raise LinearityError("effecting operator has no Eff source")
        return
    for value in eff_values:
        if eff_consumers.get(value, 0) != 1:
            raise LinearityError(f"Eff value {value!r} must have exactly one consumer")
    expected_effects = len([name for name, value_type in input_types.items() if value_type == EFF])
    if effecting_nodes != expected_effects:
        raise LinearityError("each Eff source must terminate in one effecting operator")


def _canonical(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest(payload: object) -> str:
    encoded = json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((HASH_NAMESPACE + encoded).encode("utf-8")).hexdigest()


def _type_key(value_type: Ty) -> object:
    if isinstance(value_type, Base):
        return {"base": value_type.name}
    if isinstance(value_type, Coll):
        return {"collection": value_type.kind, "item": _type_key(value_type.item)}
    if isinstance(value_type, Fn):
        return {
            "fn": {
                "params": [_type_key(item) for item in value_type.params],
                "result": _type_key(value_type.result),
            }
        }
    raise TypeError(f"unsupported type AST node: {value_type!r}")


def _input_digests(program: Program) -> dict[str, str]:
    return {
        item.name: _digest({"input": item.name, "type": _type_key(item.type)})
        for item in program.inputs
    }


def node_digest(
    node: Node,
    child_digests: tuple[str, ...],
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> str:
    """Return a canonical digest for one node and its already-digested inputs."""

    operator = registry.require(node.operator_id)
    if len(child_digests) != len(operator.signature.params):
        raise ValueError("child digest count must match operator arity")
    normalized = list(child_digests)
    commutative_positions = [
        index
        for index, enabled in enumerate(operator.signature.commutative)
        if enabled
    ]
    sorted_children = sorted(normalized[index] for index in commutative_positions)
    for index, child in zip(commutative_positions, sorted_children, strict=True):
        normalized[index] = child
    return _digest(
        {
            "operator": node.operator_id,
            "parameters": node.parameters,
            "children": normalized,
        }
    )


def _node_digests(
    program: Program,
    registry: OperatorRegistry,
) -> dict[str, str]:
    type_check(program, registry)
    input_digests = _input_digests(program)
    digests: dict[str, str] = {}
    for node in _topological_nodes(program):
        children = tuple(
            input_digests.get(reference, digests.get(reference, ""))
            for reference in node.inputs
        )
        if any(not child for child in children):
            raise ProgramTypeError(f"unresolved digest dependency in {node.node_id}")
        digests[node.node_id] = node_digest(node, children, registry)
    return digests


def program_digest(program: Program, registry: OperatorRegistry = PRIMITIVE_REGISTRY) -> str:
    """Return a stable genotype hash independent of node IDs and node order."""

    digests = _node_digests(program, registry)
    roots = tuple(digests.get(output, _input_digests(program).get(output, "")) for output in program.outputs)
    if any(not root for root in roots):
        raise ProgramTypeError("program outputs must resolve to values")
    return _digest({"schema": program.schema_version, "outputs": roots})


def subtree_digests(program: Program, registry: OperatorRegistry = PRIMITIVE_REGISTRY) -> dict[str, int]:
    """Count canonical node subtrees, enabling macro extraction by frequency."""

    digests = _node_digests(program, registry)
    counts: dict[str, int] = {}
    for digest in digests.values():
        counts[digest] = counts.get(digest, 0) + 1
    return counts


def static_cost(program: Program, registry: OperatorRegistry = PRIMITIVE_REGISTRY) -> float:
    """Compute the pre-execution cost upper bound for a bounded DAG."""

    type_check(program, registry)
    check_linearity(program, registry)
    return sum(registry.require(node.operator_id).cost for node in _topological_nodes(program))


def iter_nodes(program: Program, registry: OperatorRegistry = PRIMITIVE_REGISTRY) -> Iterable[Node]:
    """Yield nodes in deterministic topological order after type checking."""

    type_check(program, registry)
    return _topological_nodes(program)
