"""Typed primitive operators for the SEOS program space.

The operator registry is metadata only.  Execution is deliberately kept out of
this P0 module so that type checking and effect isolation can be tested without
calling an LLM, touching the filesystem, or mutating the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Literal

Purity = Literal["pure", "read", "effecting"]
Diff = Literal["exact", "surrogate", "none"]


@dataclass(frozen=True, slots=True)
class Base:
    """An atomic value type in the SEOS runtime type AST."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("base type name must not be empty")


@dataclass(frozen=True, slots=True)
class Coll:
    """A homogeneous set or sequence type."""

    kind: Literal["set", "seq"]
    item: Ty


@dataclass(frozen=True, slots=True)
class Fn:
    """A monomorphic function type used by higher-order combinators."""

    params: tuple[Ty, ...]
    result: Ty


Ty = Base | Coll | Fn

EV: Final = Base("Ev")
Z: Final = Base("Z")
HV: Final = Base("Hv")
GR: Final = Base("Gr")
AFF: Final = Base("Aff")
REL: Final = Base("Rel")
CTX: Final = Base("Ctx")
SC: Final = Base("Sc")
MEM: Final = Base("Mem")
EFF: Final = Base("Eff")


@dataclass(frozen=True, slots=True)
class Signature:
    """Operator signature and its effect/commutativity metadata."""

    params: tuple[Ty, ...]
    result: Ty
    eff_flux: int
    commutative: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if self.eff_flux < 0:
            raise ValueError("eff_flux must be non-negative")
        actual = sum(parameter == EFF for parameter in self.params)
        if self.eff_flux != actual:
            raise ValueError("eff_flux must equal the count of Eff parameters")
        if self.commutative and len(self.commutative) != len(self.params):
            raise ValueError("commutative must align with params")


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """A bounded tunable parameter of one operator.

    Parameter mutation is the only variation that applies to every node
    regardless of the registry's signature population, so without a declared
    space most programs have no reachable neighbours at all.
    """

    name: str
    low: float
    high: float
    default: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must not be empty")
        if not (isfinite(self.low) and isfinite(self.high) and isfinite(self.default)):
            raise ValueError("parameter bounds must be finite")
        if self.low >= self.high:
            raise ValueError("parameter low must be below high")
        if not self.low <= self.default <= self.high:
            raise ValueError("parameter default must lie within its bounds")

    def clamp(self, value: float) -> float:
        return min(self.high, max(self.low, value))


@dataclass(frozen=True, slots=True)
class Operator:
    """One named primitive in the typed operator algebra."""

    operator_id: str
    signature: Signature
    purity: Purity
    gradient: Diff
    cost: float = 1.0
    parameters: tuple[ParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.operator_id.strip():
            raise ValueError("operator_id must not be empty")
        names = [spec.name for spec in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("operator parameter names must be unique")
        if not isfinite(self.cost) or self.cost <= 0.0:
            raise ValueError("operator cost must be finite and positive")
        if self.purity == "effecting" and self.signature.eff_flux == 0:
            raise ValueError("effecting operators must consume an Eff token")
        if self.purity != "effecting" and self.signature.eff_flux != 0:
            raise ValueError("only effecting operators may consume Eff")


class OperatorRegistry:
    """Deterministic registry used by type checking and genotype hashing."""

    def __init__(self, operators: tuple[Operator, ...] = ()) -> None:
        self._operators: dict[str, Operator] = {}
        for operator in operators:
            self.register(operator)

    def register(self, operator: Operator) -> None:
        if operator.operator_id in self._operators:
            raise ValueError(f"duplicate operator: {operator.operator_id}")
        self._operators[operator.operator_id] = operator

    def get(self, operator_id: str) -> Operator | None:
        return self._operators.get(operator_id)

    def require(self, operator_id: str) -> Operator:
        operator = self.get(operator_id)
        if operator is None:
            raise KeyError(f"unknown operator: {operator_id}")
        return operator

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))

    def values(self) -> tuple[Operator, ...]:
        return tuple(self._operators[operator_id] for operator_id in self.ids())


def build_primitive_registry() -> OperatorRegistry:
    """Build the side-effect-free P0 primitive registry.

    Effecting operators are intentionally absent: candidate programs are
    evaluated without an ``Eff`` input.  They can be registered by a future
    online-champion registry once the execution boundary is implemented.
    """

    set_hv = Coll("set", HV)
    seq_hv = Coll("seq", HV)
    score_to_score = Fn((SC,), SC)
    registry = OperatorRegistry()
    primitives = (
        Operator("bind", Signature((HV, HV), HV, 0), "pure", "exact"),
        Operator(
            "bundle",
            Signature((set_hv,), HV, 0, (True,)),
            "pure",
            "surrogate",
            cost=2.0,
        ),
        Operator("permute", Signature((HV,), HV, 0), "pure", "exact"),
        Operator("unbind", Signature((HV, HV), HV, 0), "pure", "exact"),
        Operator("embed", Signature((EV,), Z, 0), "read", "none", cost=4.0),
        Operator("quantize", Signature((Z,), HV, 0), "pure", "surrogate"),
        Operator("project", Signature((Z,), Z, 0), "pure", "exact"),
        Operator(
            "activate",
            Signature((HV, Coll("set", EV)), Coll("set", EV), 0),
            "read",
            "none",
            cost=3.0,
            parameters=(
                ParameterSpec("threshold", -1.0, 1.0, 0.0),
                ParameterSpec("max_events", 1.0, 128.0, 64.0),
            ),
        ),
        Operator("conductance", Signature((Coll("set", EV),), GR, 0), "pure", "exact"),
        Operator("laplacian", Signature((GR,), GR, 0), "pure", "exact"),
        Operator(
            "propagate",
            Signature((GR, Z), Z, 0),
            "pure",
            "exact",
            cost=2.0,
            parameters=(ParameterSpec("diffusion_time", 0.0, 2.0, 0.25),),
        ),
        Operator("directed_rates", Signature((Coll("set", EV),), GR, 0), "pure", "exact"),
        Operator(
            "propagate_directed",
            Signature((GR, Z), Z, 0),
            "pure",
            "exact",
            cost=2.0,
            parameters=(ParameterSpec("diffusion_time", 0.0, 2.0, 0.25),),
        ),
        Operator(
            "allocate_mass",
            Signature((Coll("set", EV),), Z, 0),
            "pure",
            "exact",
            parameters=(ParameterSpec("half_life_steps", 0.5, 64.0, 8.0),),
        ),
        Operator("topk", Signature((Coll("set", HV), SC), set_hv, 0), "pure", "exact"),
        Operator("rank_by", Signature((set_hv, score_to_score), seq_hv, 0), "pure", "exact"),
        Operator(
            "leak",
            Signature((Z,), Z, 0),
            "pure",
            "exact",
            parameters=(ParameterSpec("beta", 0.0, 1.0, 0.9),),
        ),
        Operator(
            "liquid",
            Signature((Z, Z), Z, 0),
            "pure",
            "exact",
            cost=2.0,
            parameters=(ParameterSpec("alpha", 0.0, 1.0, 0.3),),
        ),
        Operator("integrate", Signature((Z, Z), Z, 0), "pure", "exact"),
        Operator("appraise", Signature((EV, CTX), AFF, 0), "read", "none", cost=2.0),
        Operator(
            "modulate",
            Signature((AFF, SC), SC, 0),
            "pure",
            "exact",
            parameters=(ParameterSpec("gain", -1.0, 1.0, 0.25),),
        ),
        Operator(
            "blend",
            Signature((AFF, AFF), AFF, 0),
            "pure",
            "exact",
            parameters=(ParameterSpec("weight", 0.0, 1.0, 0.5),),
        ),
        # Scalar sinks. Without an operator whose result is Sc and whose inputs
        # are not already Sc, the type graph has no path from evidence to a
        # score, so no program can emit a decision and there is nothing for
        # selection to act on.
        Operator(
            "similarity",
            Signature((HV, HV), SC, 0, (True, True)),
            "pure",
            "exact",
        ),
        Operator("z_energy", Signature((Z,), SC, 0), "pure", "exact"),
        Operator(
            "calibrate",
            Signature((SC,), SC, 0),
            "pure",
            "exact",
            parameters=(
                ParameterSpec("scale", -12.0, 12.0, 4.0),
                ParameterSpec("bias", -6.0, 6.0, 0.0),
            ),
        ),
        Operator("relation_update", Signature((REL, EV), REL, 0), "read", "none"),
        Operator(
            "rewire",
            Signature((GR, SC), GR, 0),
            "pure",
            "exact",
            cost=2.0,
        ),
        Operator("community", Signature((GR,), Coll("set", GR), 0), "pure", "exact", cost=2.0),
        Operator(
            "decay",
            Signature((GR,), GR, 0),
            "pure",
            "exact",
            parameters=(ParameterSpec("lambda", 0.0, 1.0, 0.9),),
        ),
    )
    for primitive in primitives:
        registry.register(primitive)
    return registry


PRIMITIVE_REGISTRY: Final = build_primitive_registry()
