"""Affect-warped Riemannian metric over the memory topology.

A weighted distance ``Σ wᵢ dᵢ²`` is a *diagonal, constant* metric: it asserts
that the cost of moving in the content direction does not depend on where you
stand in affect space.  That assertion is what makes flat retrieval fail on a
query like "haven't we met somewhere" — two slices with unrelated content can
belong together because their affect matches, while two nearly identical slices
belong apart when their affect is an octave off.  No choice of weights encodes
that, because the problem is not the weights but the coupling between axes.

This module instead equips the memory space with a warped product metric::

    ds² = f(a)·|dc|² + h·|da|²

``f`` depends on affect *position*, not on the content coordinates, so this is
a genuine Riemannian metric with a single scalar degree of freedom rather than
a general tensor with d(d+1)/2 unknown functions.  It reproduces the required
behaviour and adds one prediction that was not designed in: a geodesic may
first move along the affect axis into a region where ``f`` is small, traverse
content cheaply there, and move back.  That is emotional context reinstatement
— recall by first tuning into the register, then searching within it.

Two clarifications that matter for the implementation:

*Octave* is a statement about coordinates, not curvature.  Arousal is compared
on a log scale, which is a change of chart; in log coordinates that axis is
still flat.  The curvature comes solely from ``f`` varying with position.

The discrete object is a weighted graph, not a manifold.  The archive is a
point cloud, so the manifold is the *justification* and the edge weights are
the *implementation*; the graph Laplacian only approximates the Laplace-Beltrami
operator.  Nothing here builds a manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ssa.hdsc.transport import HDSCInvariantError

MODEL_ID: Final = "hdsc-warped-affect-metric-v0"
CLAIM_LEVEL: Final = "C0-A-warped-graph-metric"
METRIC_FORM: Final = "conformal-warped-product"

_EPS: Final = 1e-12


@dataclass(frozen=True, slots=True)
class WarpParameters:
    """Parameters of the warped product metric.

    All of these are meant to be fitted by the retrodiction loop rather than
    hand-set; a Riemannian metric chosen to taste is unfalsifiable.  The
    defaults only have to be a sane starting point for that search.
    """

    # Conformal factor f(a) = 1 + beta * ||a - a_star||^power
    beta: float = 1.0
    power: float = 2.0
    # Affect position where content channels are widest (f is smallest).
    valence_star: float = 0.0
    arousal_star: float = 0.35
    tension_star: float = 0.0
    # Affect-axis metric weights.
    affect_scale: float = 1.0
    valence_weight: float = 1.0
    arousal_weight: float = 1.0
    tension_weight: float = 0.5
    # Crossing zero valence is categorical, not a small step.
    sign_flip_penalty: float = 0.5
    valence_deadband: float = 0.05
    # Arousal is compared as a ratio; the floor keeps the log finite.
    arousal_floor: float = 0.05
    # Gaussian conductance bandwidth.
    bandwidth: float = 1.0

    def __post_init__(self) -> None:
        numeric = (
            self.beta,
            self.power,
            self.valence_star,
            self.arousal_star,
            self.tension_star,
            self.affect_scale,
            self.valence_weight,
            self.arousal_weight,
            self.tension_weight,
            self.sign_flip_penalty,
            self.valence_deadband,
            self.arousal_floor,
            self.bandwidth,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("warp parameters must be finite")
        if self.beta < 0.0 or not np.isfinite(self.beta):
            raise ValueError("beta must be finite and non-negative")
        if not 0.0 < self.power <= 8.0:
            raise ValueError("power must lie in (0, 8]")
        if self.affect_scale <= 0.0 or not np.isfinite(self.affect_scale):
            raise ValueError("affect_scale must be finite and positive")
        if self.arousal_floor <= 0.0:
            raise ValueError("arousal_floor must be positive")
        if self.bandwidth <= 0.0 or not np.isfinite(self.bandwidth):
            raise ValueError("bandwidth must be finite and positive")
        if self.sign_flip_penalty < 0.0:
            raise ValueError("sign_flip_penalty must be non-negative")
        if self.valence_deadband < 0.0:
            raise ValueError("valence_deadband must be non-negative")
        if min(self.valence_weight, self.arousal_weight, self.tension_weight) < 0.0:
            raise ValueError("affect metric weights must be non-negative")

    @property
    def is_flat(self) -> bool:
        """True when the metric degenerates to the flat, uncoupled case."""

        return self.beta == 0.0


FLAT: Final = WarpParameters(beta=0.0)
# Module-level singleton so the public helpers can default without a call in
# their signature (flake8-bugbear B008); WarpParameters is frozen, so sharing
# one instance is safe.
DEFAULT: Final = WarpParameters()


def _as_matrix(values: NDArray[np.float64] | np.ndarray, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HDSCInvariantError(f"{name} must be a non-empty two-dimensional matrix")
    if not bool(np.all(np.isfinite(matrix))):
        raise HDSCInvariantError(f"{name} must be finite")
    return matrix


def normalized_content(embeddings: np.ndarray) -> NDArray[np.float64]:
    """Row-normalise content vectors, leaving zero rows at the origin."""

    matrix = _as_matrix(embeddings, "embeddings")
    norms = np.linalg.norm(matrix, axis=1)
    result: NDArray[np.float64] = np.zeros_like(matrix)
    nonzero = norms > _EPS
    result[nonzero] = matrix[nonzero] / norms[nonzero, None]
    return result


def content_distance(embeddings: np.ndarray) -> NDArray[np.float64]:
    """Pairwise content distance in [0, 1], derived from cosine similarity."""

    unit = normalized_content(embeddings)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    distance: NDArray[np.float64] = np.sqrt(np.maximum(0.5 * (1.0 - similarity), 0.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def affect_distance(
    affect: np.ndarray,
    parameters: WarpParameters = DEFAULT,
) -> NDArray[np.float64]:
    """Pairwise affect distance with log-scaled arousal and a sign-flip term.

    ``affect`` has columns (valence, arousal, tension).  Valence is a signed
    axis, arousal is compared as a ratio because "an octave apart" is a
    multiplicative statement, and crossing zero valence adds a fixed penalty:
    a joyful and a grieving memory of the same event are categorically apart,
    not merely two units apart.
    """

    matrix = _as_matrix(affect, "affect")
    if matrix.shape[1] != 3:
        raise HDSCInvariantError("affect must have three columns: valence, arousal, tension")

    valence = matrix[:, 0]
    arousal = np.maximum(matrix[:, 1], 0.0) + parameters.arousal_floor
    tension = matrix[:, 2]

    d_valence = valence[:, None] - valence[None, :]
    d_arousal = np.log(arousal)[:, None] - np.log(arousal)[None, :]
    d_tension = tension[:, None] - tension[None, :]

    band = parameters.valence_deadband
    sign = np.zeros_like(valence)
    sign[valence > band] = 1.0
    sign[valence < -band] = -1.0
    flipped = (sign[:, None] * sign[None, :]) < 0.0

    squared = (
        parameters.valence_weight * d_valence**2
        + parameters.arousal_weight * d_arousal**2
        + parameters.tension_weight * d_tension**2
        + parameters.sign_flip_penalty * flipped.astype(np.float64)
    )
    distance: NDArray[np.float64] = np.sqrt(np.maximum(squared, 0.0)) / parameters.affect_scale
    np.fill_diagonal(distance, 0.0)
    return distance


def conformal_factor(
    affect: np.ndarray,
    parameters: WarpParameters = DEFAULT,
) -> NDArray[np.float64]:
    """``f(a)`` per node: how expensive content motion is at that affect.

    Small ``f`` means a wide content channel — from that emotional register the
    system can travel far in content for little cost, which is what lets a
    query reach a slice whose content is unrelated.
    """

    matrix = _as_matrix(affect, "affect")
    centre = np.array(
        [parameters.valence_star, parameters.arousal_star, parameters.tension_star],
        dtype=np.float64,
    )
    offset = np.linalg.norm(matrix - centre[None, :], axis=1)
    factor: NDArray[np.float64] = 1.0 + parameters.beta * np.power(offset, parameters.power)
    return factor


def warped_edge_length(
    embeddings: np.ndarray,
    affect: np.ndarray,
    parameters: WarpParameters = DEFAULT,
) -> NDArray[np.float64]:
    """Squared geodesic edge length under the warped product metric.

    ``ℓ²_ij = f(ā_ij)·d_c(i,j)² + h·d_a(i,j)²`` with ``f`` evaluated at the
    edge midpoint, which is the standard midpoint discretisation of the
    continuous metric.
    """

    content = content_distance(embeddings)
    affect_gap = affect_distance(affect, parameters)
    if content.shape != affect_gap.shape:
        raise HDSCInvariantError("embeddings and affect must describe the same nodes")

    factor = conformal_factor(affect, parameters)
    midpoint = 0.5 * (factor[:, None] + factor[None, :])
    lengths: NDArray[np.float64] = midpoint * content**2 + affect_gap**2
    np.fill_diagonal(lengths, 0.0)
    return lengths


def build_warped_conductance(
    embeddings: np.ndarray,
    affect: np.ndarray,
    parameters: WarpParameters = DEFAULT,
) -> NDArray[np.float64]:
    """Symmetric non-negative conductance induced by the warped metric.

    The output satisfies the same contract as the flat cosine conductance in
    :mod:`ssa.hdsc.transport`, so it drops straight into ``graph_laplacian``
    and ``propagate`` without touching the certified transport path.
    """

    lengths = warped_edge_length(embeddings, affect, parameters)
    conductance: NDArray[np.float64] = np.exp(-lengths / (parameters.bandwidth**2))
    np.fill_diagonal(conductance, 0.0)
    # Analytically symmetric; this removes floating-point asymmetry only, which
    # `propagate` rejects as a reciprocity violation.
    conductance = 0.5 * (conductance + conductance.T)
    return conductance


__all__ = [
    "CLAIM_LEVEL",
    "FLAT",
    "METRIC_FORM",
    "MODEL_ID",
    "WarpParameters",
    "affect_distance",
    "build_warped_conductance",
    "conformal_factor",
    "content_distance",
    "normalized_content",
    "warped_edge_length",
]
