"""HDSC-H1D directed, nonreversible transport shadow kernel.

Rates use the column convention ``rates[target, source]``. The resulting
continuous-time Markov generator is Metzler and has zero column sums. This
module never symmetrizes the topology: doing so would invent reverse edges and
erase stationary currents.

The kernel proves computational mass, positivity, and L1-contraction
properties. Physical thermodynamic interpretation remains uncalibrated and
requires local-detailed-balance plus reservoir metadata.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
from scipy.linalg import expm

from ssa.hdsc.transport import HDSCInvariantError

MODEL_ID: Final = "hdsc-h1d-directed-markov-shadow"
CLAIM_LEVEL: Final = "C0-M-nonreversible-markov-transport"
MODELED_SCALE: Final = "macro-coarse-grained"
MICROSTATE_STATUS: Final = "unmodeled"
TOPOLOGY_CLASS: Final = "directed-nonreversible"
RATE_CONVENTION: Final = "rates[target,source]"
PHYSICAL_THERMODYNAMIC_STATUS: Final = "not-calibrated"

StationaryGate = Literal["not-evaluated", "pass", "fail"]
EntropyGate = Literal["not-evaluated", "pass", "fail"]
FiniteEntropyProductionGate = Literal["not-evaluated", "pass", "abstain", "fail"]


@dataclass(frozen=True, slots=True)
class HDSCH1DParameters:
    """Versioned limits for directed continuous-time transport."""

    transport_time: float = 1.0
    sink_rate: float = 0.0
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.transport_time < 0.0 or not math.isfinite(self.transport_time):
            raise ValueError("transport_time must be finite and non-negative")
        if self.sink_rate < 0.0 or not math.isfinite(self.sink_rate):
            raise ValueError("sink_rate must be finite and non-negative")
        if self.tolerance <= 0.0 or not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class HDSCH1DAudit:
    """Invariant and non-equilibrium diagnostics for one directed step."""

    model_id: str
    claim_level: str
    modeled_scale: str
    microstate_status: str
    topology_class: str
    rate_convention: str
    physical_thermodynamic_status: str
    node_count: int
    directed_edge_count: int
    input_mass: float
    transported_mass: float
    output_mass: float
    dissipated_mass: float
    conservation_residual: float
    generator_column_sum_residual: float
    kernel_column_sum_residual: float
    minimum_kernel_entry: float
    minimum_output_mass: float
    l1_operator_norm: float
    asymmetry_index: float
    reciprocal_rate_mass: float
    directed_excess_rate_mass: float
    reciprocal_support: bool
    stationary_gate: StationaryGate
    stationary_residual: float | None
    relative_entropy_gate: EntropyGate
    relative_entropy_before: float | None
    relative_entropy_after: float | None
    finite_entropy_production_gate: FiniteEntropyProductionGate
    entropy_production_rate: float | None
    detailed_balance_residual: float | None


@dataclass(frozen=True, slots=True)
class HDSCH1DResult:
    """Directed transport vectors, generator, kernel, and audit."""

    source_mass: tuple[float, ...]
    transported_mass: tuple[float, ...]
    output_mass: tuple[float, ...]
    generator: tuple[tuple[float, ...], ...]
    transition_kernel: tuple[tuple[float, ...], ...]
    audit: HDSCH1DAudit


def build_directed_rates(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
) -> np.ndarray:
    """Build rates[target, source] without adding reverse edges."""
    if not node_ids:
        raise HDSCInvariantError("node_ids must not be empty")
    if any(not node_id.strip() for node_id in node_ids):
        raise HDSCInvariantError("node_ids must contain non-empty identifiers")
    if len(set(node_ids)) != len(node_ids):
        raise HDSCInvariantError("node_ids must be unique")

    index = {node_id: position for position, node_id in enumerate(node_ids)}
    rates: np.ndarray = np.zeros((len(node_ids), len(node_ids)), dtype=np.float64)
    for source_id, target_id, raw_rate in edges:
        if source_id not in index or target_id not in index:
            raise HDSCInvariantError("directed edge references an unknown node")
        if source_id == target_id:
            raise HDSCInvariantError("directed self-edges are not transport rates")
        rate = float(raw_rate)
        if rate < 0.0 or not math.isfinite(rate):
            raise HDSCInvariantError("directed edge rates must be finite and non-negative")
        rates[index[target_id], index[source_id]] += rate

    if not bool(np.all(np.isfinite(rates))):
        raise HDSCInvariantError("accumulated directed rates overflowed")
    return rates


def directed_generator(rates: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return a column-conservative Metzler generator for directed rates."""
    matrix = _rate_matrix(rates, tolerance)
    outflow = np.sum(matrix, axis=0, dtype=np.float64)
    generator = matrix - np.diag(outflow)
    return cast(np.ndarray, generator)


def propagate_directed(
    source_mass: np.ndarray,
    rates: np.ndarray,
    parameters: HDSCH1DParameters | None = None,
    *,
    stationary_distribution: np.ndarray | None = None,
) -> HDSCH1DResult:
    """Propagate mass on the full directed topology without symmetrization."""
    effective = parameters or HDSCH1DParameters()
    source = _mass_vector(source_mass, effective.tolerance)
    rate_matrix = _rate_matrix(rates, effective.tolerance)
    if rate_matrix.shape[0] != source.size:
        raise HDSCInvariantError("source mass and directed rates dimensions must agree")

    generator = directed_generator(rate_matrix, tolerance=effective.tolerance)
    raw_kernel = np.asarray(expm(effective.transport_time * generator), dtype=np.float64)
    if not bool(np.all(np.isfinite(raw_kernel))):
        raise HDSCInvariantError("directed transition kernel contains non-finite values")
    kernel_column_residual = float(
        np.max(np.abs(np.sum(raw_kernel, axis=0, dtype=np.float64) - 1.0))
    )
    minimum_kernel_entry = float(np.min(raw_kernel))
    kernel = _repair_markov_kernel(raw_kernel, effective.tolerance)

    input_total = float(np.sum(source, dtype=np.float64))
    transported = kernel @ source
    transported = _repair_mass(transported, input_total, effective.tolerance)
    retention = math.exp(-effective.sink_rate * effective.transport_time)
    target_output = retention * input_total
    output = _repair_mass(retention * transported, target_output, effective.tolerance)

    transported_total = float(np.sum(transported, dtype=np.float64))
    output_total = float(np.sum(output, dtype=np.float64))
    dissipated = (1.0 - retention) * input_total
    reciprocal = np.minimum(rate_matrix, rate_matrix.T)
    directed_excess = rate_matrix - reciprocal
    total_rate = float(np.sum(rate_matrix, dtype=np.float64))
    asymmetry_index = (
        float(np.sum(np.abs(rate_matrix - rate_matrix.T), dtype=np.float64)) / (2.0 * total_rate)
        if total_rate > 0.0
        else 0.0
    )
    reciprocal_support = _has_reciprocal_support(rate_matrix, effective.tolerance)
    stationary = _stationary_diagnostics(
        source,
        transported,
        rate_matrix,
        generator,
        stationary_distribution,
        effective.tolerance,
    )

    audit = HDSCH1DAudit(
        model_id=MODEL_ID,
        claim_level=CLAIM_LEVEL,
        modeled_scale=MODELED_SCALE,
        microstate_status=MICROSTATE_STATUS,
        topology_class=TOPOLOGY_CLASS,
        rate_convention=RATE_CONVENTION,
        physical_thermodynamic_status=PHYSICAL_THERMODYNAMIC_STATUS,
        node_count=source.size,
        directed_edge_count=int(np.count_nonzero(rate_matrix > effective.tolerance)),
        input_mass=input_total,
        transported_mass=transported_total,
        output_mass=output_total,
        dissipated_mass=dissipated,
        conservation_residual=input_total - output_total - dissipated,
        generator_column_sum_residual=float(
            np.max(np.abs(np.sum(generator, axis=0, dtype=np.float64)))
        ),
        kernel_column_sum_residual=kernel_column_residual,
        minimum_kernel_entry=minimum_kernel_entry,
        minimum_output_mass=float(np.min(output)),
        l1_operator_norm=float(np.max(np.sum(np.abs(kernel), axis=0, dtype=np.float64))),
        asymmetry_index=asymmetry_index,
        reciprocal_rate_mass=float(np.sum(reciprocal, dtype=np.float64)),
        directed_excess_rate_mass=float(np.sum(directed_excess, dtype=np.float64)),
        reciprocal_support=reciprocal_support,
        stationary_gate=stationary.stationary_gate,
        stationary_residual=stationary.stationary_residual,
        relative_entropy_gate=stationary.relative_entropy_gate,
        relative_entropy_before=stationary.relative_entropy_before,
        relative_entropy_after=stationary.relative_entropy_after,
        finite_entropy_production_gate=stationary.finite_entropy_production_gate,
        entropy_production_rate=stationary.entropy_production_rate,
        detailed_balance_residual=stationary.detailed_balance_residual,
    )
    return HDSCH1DResult(
        source_mass=tuple(float(value) for value in source),
        transported_mass=tuple(float(value) for value in transported),
        output_mass=tuple(float(value) for value in output),
        generator=_matrix_tuple(generator),
        transition_kernel=_matrix_tuple(kernel),
        audit=audit,
    )


@dataclass(frozen=True, slots=True)
class _StationaryDiagnostics:
    stationary_gate: StationaryGate
    stationary_residual: float | None
    relative_entropy_gate: EntropyGate
    relative_entropy_before: float | None
    relative_entropy_after: float | None
    finite_entropy_production_gate: FiniteEntropyProductionGate
    entropy_production_rate: float | None
    detailed_balance_residual: float | None


def _stationary_diagnostics(
    source: np.ndarray,
    transported: np.ndarray,
    rates: np.ndarray,
    generator: np.ndarray,
    stationary_distribution: np.ndarray | None,
    tolerance: float,
) -> _StationaryDiagnostics:
    if stationary_distribution is None:
        return _StationaryDiagnostics(
            stationary_gate="not-evaluated",
            stationary_residual=None,
            relative_entropy_gate="not-evaluated",
            relative_entropy_before=None,
            relative_entropy_after=None,
            finite_entropy_production_gate="not-evaluated",
            entropy_production_rate=None,
            detailed_balance_residual=None,
        )

    stationary = _strict_probability(stationary_distribution, tolerance)
    if stationary.size != source.size:
        raise HDSCInvariantError("stationary distribution and topology dimensions must agree")
    residual = float(np.max(np.abs(generator @ stationary)))
    if residual > tolerance:
        return _StationaryDiagnostics(
            stationary_gate="fail",
            stationary_residual=residual,
            relative_entropy_gate="not-evaluated",
            relative_entropy_before=None,
            relative_entropy_after=None,
            finite_entropy_production_gate="not-evaluated",
            entropy_production_rate=None,
            detailed_balance_residual=None,
        )

    source_total = float(np.sum(source, dtype=np.float64))
    if source_total > 0.0:
        before = _relative_entropy(source / source_total, stationary)
        after = _relative_entropy(transported / source_total, stationary)
        entropy_gate: EntropyGate = "pass" if after <= before + tolerance else "fail"
    else:
        before = None
        after = None
        entropy_gate = "not-evaluated"

    flux = rates * stationary[np.newaxis, :]
    balance_residual = float(np.max(np.abs(flux - flux.T)))
    reciprocal_support = _has_reciprocal_support(rates, tolerance)
    if reciprocal_support:
        entropy_production = _entropy_production(flux, tolerance)
        production_gate: FiniteEntropyProductionGate = (
            "pass" if entropy_production >= -tolerance else "fail"
        )
        entropy_production = max(0.0, entropy_production)
    else:
        entropy_production = None
        production_gate = "abstain"

    return _StationaryDiagnostics(
        stationary_gate="pass",
        stationary_residual=residual,
        relative_entropy_gate=entropy_gate,
        relative_entropy_before=before,
        relative_entropy_after=after,
        finite_entropy_production_gate=production_gate,
        entropy_production_rate=entropy_production,
        detailed_balance_residual=balance_residual,
    )


def _rate_matrix(rates: np.ndarray, tolerance: float) -> np.ndarray:
    matrix = np.asarray(rates, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise HDSCInvariantError("directed rates must be a non-empty square matrix")
    if not bool(np.all(np.isfinite(matrix))):
        raise HDSCInvariantError("directed rates must contain only finite values")
    if np.any(matrix < -tolerance):
        raise HDSCInvariantError("directed rates must be non-negative")
    if float(np.max(np.abs(np.diag(matrix)))) > tolerance:
        raise HDSCInvariantError("directed rate diagonal must be zero")
    clean = np.maximum(matrix, 0.0)
    np.fill_diagonal(clean, 0.0)
    return cast(np.ndarray, clean)


def _mass_vector(source_mass: np.ndarray, tolerance: float) -> np.ndarray:
    source = np.asarray(source_mass, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise HDSCInvariantError("source mass must be a non-empty one-dimensional vector")
    if not bool(np.all(np.isfinite(source))):
        raise HDSCInvariantError("source mass must contain only finite values")
    if np.any(source < -tolerance):
        raise HDSCInvariantError("source mass must be non-negative")
    return cast(np.ndarray, np.maximum(source, 0.0))


def _repair_markov_kernel(kernel: np.ndarray, tolerance: float) -> np.ndarray:
    if float(np.min(kernel)) < -tolerance:
        raise HDSCInvariantError("directed transition kernel produced negative probability")
    clean = np.maximum(kernel, 0.0)
    column_sums = np.sum(clean, axis=0, dtype=np.float64)
    if np.any(column_sums <= 0.0) or not bool(np.all(np.isfinite(column_sums))):
        raise HDSCInvariantError("directed transition kernel lost a probability column")
    clean /= column_sums[np.newaxis, :]
    return cast(np.ndarray, clean)


def _repair_mass(values: np.ndarray, target_mass: float, tolerance: float) -> np.ndarray:
    if float(np.min(values)) < -tolerance:
        raise HDSCInvariantError("directed transport produced negative mass")
    repaired = np.maximum(values, 0.0)
    actual = float(np.sum(repaired, dtype=np.float64))
    if target_mass == 0.0:
        return cast(np.ndarray, np.zeros_like(repaired))
    if actual <= 0.0 or not math.isfinite(actual):
        raise HDSCInvariantError("directed transport lost a positive source mass")
    repaired *= target_mass / actual
    return cast(np.ndarray, repaired)


def _strict_probability(values: np.ndarray, tolerance: float) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or probability.size == 0:
        raise HDSCInvariantError("stationary distribution must be one-dimensional")
    if not bool(np.all(np.isfinite(probability))) or np.any(probability <= tolerance):
        raise HDSCInvariantError("stationary distribution must be finite and strictly positive")
    total = float(np.sum(probability, dtype=np.float64))
    if abs(total - 1.0) > tolerance:
        raise HDSCInvariantError("stationary distribution must sum to one")
    return cast(np.ndarray, probability)


def _has_reciprocal_support(rates: np.ndarray, tolerance: float) -> bool:
    support = rates > tolerance
    return bool(np.all(~support | support.T))


def _relative_entropy(probability: np.ndarray, stationary: np.ndarray) -> float:
    positive = probability > 0.0
    return float(
        np.sum(
            probability[positive] * np.log(probability[positive] / stationary[positive]),
            dtype=np.float64,
        )
    )


def _entropy_production(flux: np.ndarray, tolerance: float) -> float:
    production = 0.0
    for left in range(flux.shape[0]):
        for right in range(left + 1, flux.shape[0]):
            forward = float(flux[left, right])
            reverse = float(flux[right, left])
            if forward <= tolerance and reverse <= tolerance:
                continue
            if forward <= tolerance or reverse <= tolerance:
                raise HDSCInvariantError(
                    "finite entropy production requires reciprocal transition support"
                )
            production += (forward - reverse) * math.log(forward / reverse)
    return production


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


__all__ = [
    "CLAIM_LEVEL",
    "MICROSTATE_STATUS",
    "MODELED_SCALE",
    "MODEL_ID",
    "PHYSICAL_THERMODYNAMIC_STATUS",
    "RATE_CONVENTION",
    "TOPOLOGY_CLASS",
    "HDSCH1DAudit",
    "HDSCH1DParameters",
    "HDSCH1DResult",
    "build_directed_rates",
    "directed_generator",
    "propagate_directed",
]
