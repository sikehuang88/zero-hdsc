"""Budgeted, side-effect-isolated interpreter for offline SEOS programs."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

from ssa.hdsc.operators import PRIMITIVE_REGISTRY, OperatorRegistry
from ssa.hdsc.program import Program, check_linearity, iter_nodes, static_cost, type_check


class InterpreterError(RuntimeError):
    """Base error for candidate program execution."""


class BudgetExceeded(InterpreterError):
    """Raised when a static or runtime budget boundary is crossed."""


class EffectBoundaryError(InterpreterError):
    """Raised when an offline candidate reaches an effect boundary."""


@dataclass(frozen=True, slots=True)
class ExecBudget:
    max_steps: int = 4096
    max_wall_ms: int = 2_000
    max_llm_calls: int = 4
    max_tokens: int = 8_192
    max_static_cost: float = 1_000.0

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_wall_ms < 1:
            raise ValueError("step and wall budgets must be positive")
        if self.max_llm_calls < 0 or self.max_tokens < 0:
            raise ValueError("LLM and token budgets must be non-negative")
        if not isfinite(self.max_static_cost) or self.max_static_cost <= 0.0:
            raise ValueError("max_static_cost must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExecutionValue:
    """Optional runtime accounting returned by an operator implementation."""

    value: object
    llm_calls: int = 0
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.llm_calls < 0 or self.tokens < 0:
            raise ValueError("runtime accounting must be non-negative")


OperatorImplementation = Callable[
    [tuple[object, ...], tuple[tuple[str, object], ...]],
    object | ExecutionValue,
]


@dataclass(frozen=True, slots=True)
class ExecContext:
    """Runtime boundary for one evaluation.

    Effects are disabled by default. Implementations are injected by the caller
    so this module never reaches the host tool kernel on its own.
    """

    implementations: Mapping[str, OperatorImplementation]
    allow_effects: bool = False
    effect_token: object | None = None
    shared_cache: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class StaticBudgetReport:
    static_cost: float
    node_count: int
    depth: int
    admitted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecResult:
    value: object
    steps: int
    wall_ms: int
    llm_calls: int
    tokens: int
    cache_hits: int
    truncated: bool = False


def static_budget_gate(
    program: Program,
    budget: ExecBudget,
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> StaticBudgetReport:
    """Validate type/effect invariants and admit only bounded programs."""

    type_check(program, registry)
    check_linearity(program, registry)
    cost = static_cost(program, registry)
    if program.node_count > budget.max_steps:
        return StaticBudgetReport(
            cost,
            program.node_count,
            program.depth,
            False,
            "node count exceeds max_steps",
        )
    if cost > budget.max_static_cost:
        return StaticBudgetReport(
            cost,
            program.node_count,
            program.depth,
            False,
            "static cost exceeds max_static_cost",
        )
    return StaticBudgetReport(cost, program.node_count, program.depth, True)


def _value_key(value: object) -> str:
    """Return a collision-free content key for one runtime value.

    ``repr`` is not usable on its own: NumPy truncates large arrays, so two
    different arrays sharing their edge elements render identically and would
    otherwise collide into one memoization entry and return the wrong result.
    """

    buffer = getattr(value, "tobytes", None)
    if callable(buffer):
        shape = getattr(value, "shape", ())
        dtype = getattr(value, "dtype", "")
        digest = hashlib.sha256(buffer()).hexdigest()
        return f"buffer:{dtype}:{shape}:{digest}"
    digest_property = getattr(value, "digest", None)
    if isinstance(digest_property, str):
        return f"digest:{type(value).__name__}:{digest_property}"
    if isinstance(value, (tuple, list)):
        joined = "|".join(_value_key(item) for item in value)
        return f"seq:{len(value)}:{hashlib.sha256(joined.encode('utf-8')).hexdigest()}"
    return f"repr:{type(value).__name__}:{value!r}"


def _cache_key(
    operator_id: str,
    args: tuple[object, ...],
    parameters: tuple[tuple[str, object], ...],
) -> str:
    payload = {
        "operator": operator_id,
        "args": [_value_key(item) for item in args],
        "parameters": [[name, _value_key(item)] for name, item in parameters],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def execute(
    program: Program,
    inputs: Mapping[str, object],
    budget: ExecBudget,
    ctx: ExecContext,
    *,
    registry: OperatorRegistry = PRIMITIVE_REGISTRY,
) -> ExecResult:
    """Execute one bounded DAG with purity-aware memoization."""

    admission = static_budget_gate(program, budget, registry=registry)
    if not admission.admitted:
        raise BudgetExceeded(admission.reason or "program rejected by static budget")
    if program.eff_flux and not ctx.allow_effects:
        raise EffectBoundaryError("offline execution does not provide an Eff capability")
    if program.eff_flux and ctx.effect_token is None:
        raise EffectBoundaryError("effectful execution requires an explicit effect token")
    for item in program.inputs:
        if item.name not in inputs:
            raise InterpreterError(f"missing program input: {item.name}")

    values: dict[str, object] = dict(inputs)
    local_cache: dict[str, object] = {}
    start = time.perf_counter()
    steps = 0
    llm_calls = 0
    tokens = 0
    cache_hits = 0

    for node in iter_nodes(program, registry):
        if steps >= budget.max_steps:
            raise BudgetExceeded("runtime step budget exceeded")
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if elapsed_ms >= budget.max_wall_ms:
            raise BudgetExceeded("runtime wall budget exceeded")
        operator = registry.require(node.operator_id)
        if operator.purity == "effecting" and not ctx.allow_effects:
            raise EffectBoundaryError(f"effect boundary reached at {node.node_id}")
        args = tuple(values[reference] for reference in node.inputs)
        cache = (
            ctx.shared_cache
            if operator.purity == "pure" and ctx.shared_cache is not None
            else local_cache
        )
        key = _cache_key(node.operator_id, args, node.parameters)
        if operator.purity != "effecting" and key in cache:
            result = cache[key]
            cache_hits += 1
        else:
            implementation = ctx.implementations.get(node.operator_id)
            if implementation is None:
                raise InterpreterError(f"no implementation for operator: {node.operator_id}")
            result = implementation(args, node.parameters)
            if operator.purity != "effecting":
                cache[key] = result
        if isinstance(result, ExecutionValue):
            value = result.value
            llm_calls += result.llm_calls
            tokens += result.tokens
        else:
            value = result
        if llm_calls > budget.max_llm_calls:
            raise BudgetExceeded("runtime LLM-call budget exceeded")
        if tokens > budget.max_tokens:
            raise BudgetExceeded("runtime token budget exceeded")
        values[node.node_id] = value
        steps += 1

    outputs = tuple(values[reference] for reference in program.outputs)
    output_value: object = outputs[0] if len(outputs) == 1 else outputs
    wall_ms = int((time.perf_counter() - start) * 1000)
    if wall_ms > budget.max_wall_ms:
        raise BudgetExceeded("runtime wall budget exceeded")
    return ExecResult(output_value, steps, wall_ms, llm_calls, tokens, cache_hits)
