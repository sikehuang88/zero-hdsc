"""Reference HDSC-H1 passive graph-diffusion kernel.

The state variable is dimensionless computational activation mass. It is not
physical energy, heat, or temperature. The symmetric graph heat equation gives
this kernel a C1 thermodynamic structure analogy; a C2 physical claim requires
an independently calibrated mapping to measured joules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

MODEL_ID: Final = "hdsc-h1-c1-shadow"
CLAIM_LEVEL: Final = "C1-structural-analogy"
MODELED_SCALE: Final = "macro-coarse-grained"
MICROSTATE_STATUS: Final = "unmodeled"


class HDSCInvariantError(ValueError):
    """Raised when inputs violate a declared HDSC-H1 invariant."""


@dataclass(frozen=True, slots=True)
class HDSCH1Parameters:
    """Versioned parameters for source allocation and passive transport."""

    source_budget: float = 1.0
    max_sources: int = 8
    affinity_threshold: float = 0.0
    affinity_power: float = 2.0
    freshness_power: float = 1.0
    importance_power: float = 1.0
    conductance_threshold: float = 0.4
    conductance_power: float = 1.0
    diffusion_time: float = 1.0
    sink_rate: float = 0.0
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not math.isfinite(self.source_budget) or self.source_budget <= 0.0:
            raise ValueError("source_budget must be finite and positive")
        if self.max_sources < 1:
            raise ValueError("max_sources must be positive")
        if not -1.0 <= self.affinity_threshold <= 1.0:
            raise ValueError("affinity_threshold must be in [-1, 1]")
        if not -1.0 <= self.conductance_threshold <= 1.0:
            raise ValueError("conductance_threshold must be in [-1, 1]")
        if self.affinity_power <= 0.0 or not math.isfinite(self.affinity_power):
            raise ValueError("affinity_power must be finite and positive")
        if self.freshness_power < 0.0 or not math.isfinite(self.freshness_power):
            raise ValueError("freshness_power must be finite and non-negative")
        if self.importance_power < 0.0 or not math.isfinite(self.importance_power):
            raise ValueError("importance_power must be finite and non-negative")
        if self.conductance_power <= 0.0 or not math.isfinite(self.conductance_power):
            raise ValueError("conductance_power must be finite and positive")
        if self.diffusion_time < 0.0 or not math.isfinite(self.diffusion_time):
            raise ValueError("diffusion_time must be finite and non-negative")
        if self.sink_rate < 0.0 or not math.isfinite(self.sink_rate):
            raise ValueError("sink_rate must be finite and non-negative")
        if self.tolerance <= 0.0 or not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class HDSCH1Audit:
    """Numerical evidence emitted for every HDSC-H1 propagation."""

    model_id: str
    claim_level: str
    modeled_scale: str
    microstate_status: str
    node_count: int
    input_mass: float
    output_mass: float
    dissipated_mass: float
    conservation_residual: float
    reciprocity_residual: float
    laplacian_row_sum_residual: float
    minimum_output_mass: float
    dirichlet_before: float
    dirichlet_after: float
    shannon_entropy_before: float
    shannon_entropy_after: float


@dataclass(frozen=True, slots=True)
class HDSCH1Result:
    """Immutable propagation vectors and their invariant audit."""

    source_mass: tuple[float, ...]
    diffused_mass: tuple[float, ...]
    output_mass: tuple[float, ...]
    audit: HDSCH1Audit


def allocate_source_mass(
    similarities: np.ndarray,
    freshness: np.ndarray,
    importance: np.ndarray,
    parameters: HDSCH1Parameters,
) -> np.ndarray:
    """Allocate a fixed mass budget across the globally best source traces."""
    similarity = _vector(similarities, "similarities")
    freshness_vector = _vector(freshness, "freshness")
    importance_vector = _vector(importance, "importance")
    if not (similarity.shape == freshness_vector.shape == importance_vector.shape):
        raise HDSCInvariantError("source feature vectors must have identical shapes")
    if np.any((similarity < -1.0) | (similarity > 1.0)):
        raise HDSCInvariantError("similarities must be in [-1, 1]")
    if np.any((freshness_vector <= 0.0) | (freshness_vector > 1.0)):
        raise HDSCInvariantError("freshness must be in (0, 1]")
    if np.any((importance_vector < 0.0) | (importance_vector > 1.0)):
        raise HDSCInvariantError("importance must be in [0, 1]")

    affinity = np.maximum(similarity - parameters.affinity_threshold, 0.0)
    weights = (
        np.power(affinity, parameters.affinity_power)
        * np.power(freshness_vector, parameters.freshness_power)
        * np.power(importance_vector, parameters.importance_power)
    )
    eligible = np.flatnonzero(weights > 0.0)
    if eligible.size == 0:
        raise HDSCInvariantError("no trace has positive source evidence")

    ranked = eligible[np.argsort(-weights[eligible], kind="stable")]
    selected = ranked[: parameters.max_sources]
    selected_total = float(np.sum(weights[selected], dtype=np.float64))
    if not math.isfinite(selected_total) or selected_total <= 0.0:
        raise HDSCInvariantError("source evidence is numerically degenerate")

    source = np.zeros_like(weights, dtype=np.float64)
    source[selected] = parameters.source_budget * weights[selected] / selected_total
    source *= parameters.source_budget / float(np.sum(source, dtype=np.float64))
    return cast(np.ndarray, source)


def build_semantic_conductance(
    embeddings: np.ndarray,
    parameters: HDSCH1Parameters,
) -> np.ndarray:
    """Build reciprocal non-negative conductance from pairwise cosine affinity."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HDSCInvariantError("embeddings must be a non-empty two-dimensional matrix")
    _require_finite(matrix, "embeddings")

    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.zeros_like(matrix, dtype=np.float64)
    nonzero = norms > parameters.tolerance
    normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    affinity = np.maximum(similarities - parameters.conductance_threshold, 0.0)
    conductance = np.power(affinity, parameters.conductance_power)
    np.fill_diagonal(conductance, 0.0)
    # The cosine Gram matrix is symmetric analytically; this removes roundoff only.
    conductance = 0.5 * (conductance + conductance.T)
    return cast(np.ndarray, conductance)


def graph_laplacian(conductance: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Validate a passive conductance matrix and return L = D - K."""
    matrix = np.asarray(conductance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise HDSCInvariantError("conductance must be a non-empty square matrix")
    _require_finite(matrix, "conductance")
    if np.any(matrix < -tolerance):
        raise HDSCInvariantError("conductance must be non-negative")
    if float(np.max(np.abs(matrix - matrix.T))) > tolerance:
        raise HDSCInvariantError("passive conductance must be symmetric")
    if float(np.max(np.abs(np.diag(matrix)))) > tolerance:
        raise HDSCInvariantError("conductance diagonal must be zero")

    clean = np.maximum(0.0, 0.5 * (matrix + matrix.T))
    np.fill_diagonal(clean, 0.0)
    return cast(np.ndarray, np.diag(np.sum(clean, axis=1, dtype=np.float64)) - clean)


def propagate(
    source_mass: np.ndarray,
    conductance: np.ndarray,
    parameters: HDSCH1Parameters,
) -> HDSCH1Result:
    """Apply passive diffusion and an explicitly accounted environment sink.

    The closed-system term is exp(-t L). Uniform leakage to an environment is
    exp(-kappa t). Their product is a positive contraction semigroup.
    """
    source = _vector(source_mass, "source_mass").copy()
    if np.any(source < -parameters.tolerance):
        raise HDSCInvariantError("source mass must be non-negative")
    source[source < 0.0] = 0.0

    laplacian = graph_laplacian(conductance, tolerance=parameters.tolerance)
    if laplacian.shape[0] != source.size:
        raise HDSCInvariantError("source mass and conductance dimensions must agree")

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    if float(np.min(eigenvalues)) < -parameters.tolerance:
        raise HDSCInvariantError("graph Laplacian is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    attenuation = np.exp(-parameters.diffusion_time * eigenvalues)
    heat_kernel = (eigenvectors * attenuation) @ eigenvectors.T
    diffused = heat_kernel @ source

    input_total = float(np.sum(source, dtype=np.float64))
    diffused = _repair_nonnegative_mass(diffused, input_total, parameters.tolerance)
    retention = math.exp(-parameters.sink_rate * parameters.diffusion_time)
    target_output = retention * input_total
    output = _repair_nonnegative_mass(
        retention * diffused,
        target_output,
        parameters.tolerance,
    )

    output_total = float(np.sum(output, dtype=np.float64))
    dissipated = (1.0 - retention) * input_total
    conservation_residual = input_total - output_total - dissipated
    reciprocity_residual = float(np.max(np.abs(conductance - conductance.T)))
    row_sum_residual = float(np.max(np.abs(np.sum(laplacian, axis=1))))
    dirichlet_before = _dirichlet(source, laplacian)
    dirichlet_after = _dirichlet(diffused, laplacian)

    audit = HDSCH1Audit(
        model_id=MODEL_ID,
        claim_level=CLAIM_LEVEL,
        modeled_scale=MODELED_SCALE,
        microstate_status=MICROSTATE_STATUS,
        node_count=source.size,
        input_mass=input_total,
        output_mass=output_total,
        dissipated_mass=dissipated,
        conservation_residual=conservation_residual,
        reciprocity_residual=reciprocity_residual,
        laplacian_row_sum_residual=row_sum_residual,
        minimum_output_mass=float(np.min(output)),
        dirichlet_before=dirichlet_before,
        dirichlet_after=dirichlet_after,
        shannon_entropy_before=_shannon_entropy(source),
        shannon_entropy_after=_shannon_entropy(diffused),
    )
    return HDSCH1Result(
        source_mass=tuple(float(value) for value in source),
        diffused_mass=tuple(float(value) for value in diffused),
        output_mass=tuple(float(value) for value in output),
        audit=audit,
    )


def run_shadow(
    *,
    similarities: np.ndarray,
    freshness: np.ndarray,
    importance: np.ndarray,
    embeddings: np.ndarray,
    parameters: HDSCH1Parameters | None = None,
) -> HDSCH1Result:
    """Run the complete HDSC-H1 reference path without affecting prompts."""
    effective = parameters or HDSCH1Parameters()
    source = allocate_source_mass(similarities, freshness, importance, effective)
    conductance = build_semantic_conductance(embeddings, effective)
    return propagate(source, conductance, effective)


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise HDSCInvariantError(f"{name} must be a non-empty one-dimensional vector")
    _require_finite(vector, name)
    return cast(np.ndarray, vector)


def _require_finite(values: np.ndarray, name: str) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise HDSCInvariantError(f"{name} must contain only finite values")


def _repair_nonnegative_mass(
    values: np.ndarray,
    target_mass: float,
    tolerance: float,
) -> np.ndarray:
    if float(np.min(values)) < -tolerance:
        raise HDSCInvariantError("diffusion produced negative mass beyond tolerance")
    repaired = np.maximum(values, 0.0)
    actual_mass = float(np.sum(repaired, dtype=np.float64))
    if target_mass == 0.0:
        return cast(np.ndarray, np.zeros_like(repaired))
    if actual_mass <= 0.0 or not math.isfinite(actual_mass):
        raise HDSCInvariantError("diffusion lost a positive source budget")
    repaired *= target_mass / actual_mass
    return cast(np.ndarray, repaired)


def _dirichlet(values: np.ndarray, laplacian: np.ndarray) -> float:
    result = 0.5 * float(values @ laplacian @ values)
    return max(0.0, result)


def _shannon_entropy(values: np.ndarray) -> float:
    total = float(np.sum(values, dtype=np.float64))
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0.0]
    return -float(np.sum(positive * np.log(positive), dtype=np.float64))


__all__ = [
    "CLAIM_LEVEL",
    "MICROSTATE_STATUS",
    "MODELED_SCALE",
    "MODEL_ID",
    "HDSCH1Audit",
    "HDSCH1Parameters",
    "HDSCH1Result",
    "HDSCInvariantError",
    "allocate_source_mass",
    "build_semantic_conductance",
    "graph_laplacian",
    "propagate",
    "run_shadow",
]
