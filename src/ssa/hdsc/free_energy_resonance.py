"""Free-energy resonance retrieval over the warped memory topology.

A query like "haven't we met somewhere" is not a nearest-neighbour lookup.  Its
content matches nothing in particular; what it does is drop a stone on the
topology and ask which nodes the ripple lights up.  This module implements that
scan as a statistical-mechanical selection.

For a candidate memory ``m`` under current state ``q``::

    F(m) = χ(m)² − T·S(m)

``χ²`` is the *geodesic* detuning under the affect-warped metric, ``S`` is the
entropy of the memory read as a distribution rather than a point — a slice you
remember vaguely occupies more phase space — and ``T`` is a retrieval
temperature driven by arousal.  Selecting by ``p(m) ∝ exp(−F/T)`` factorises::

    p(m) ∝ exp(S(m)) · exp(−χ(m)²/T)
             ↑ phase-space volume   ↑ Boltzmann factor

which is what makes the temperature interesting.  The degeneracy term does not
depend on ``T`` at all; only the detuning term is annealed.  So:

* ``T → 0``   the Boltzmann factor collapses onto ``argmin χ²`` — exact,
  rigid recall of one fixed memory.
* ``T ≈ 1``   the two terms compete and blurred memories start participating —
  associative drift, the brainstorming register.
* ``T ≫ 1``   ``exp(−χ²/T) → 1`` and the detuning term stops mattering, leaving
  ``p ∝ exp(S)``: fuzzy clouds light up indiscriminately, which is what a mind
  does under fear.

Combining with :mod:`ssa.hdsc.warped` by Varadhan's formula,
``d_geo(x,y)² = −lim_{t→0} 4t log p_t(x,y)``, the detuning is read directly off
the heat kernel and the whole rule closes::

    p(m) ∝ p_t(q,m)^{4t/T} · exp(S(m))

Temperature is literally an exponent on the arriving diffusion mass.

Two properties are deliberate rather than incidental.  Both ``χ²`` and ``S`` are
standardised across the candidate set, because Gaussian entropy scales as
``log det Σ`` while detuning scales quadratically — without normalisation the
two terms live on incomparable scales and ``T ≈ 1`` would not be the transition
it is described as.  And retrieval can return nothing: if no candidate is close
in absolute terms, raising the temperature must not manufacture a memory.
"""

# Mathematical notation in docstrings intentionally uses Unicode symbols.
# ruff: noqa: RUF002

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ssa.hdsc.transport import HDSCInvariantError, graph_laplacian

MODEL_ID: Final = "hdsc-free-energy-resonance-v0"
CLAIM_LEVEL: Final = "C0-A-boltzmann-retrieval"
SELECTION_RULE: Final = "boltzmann-over-geodesic-detuning-and-entropy"

_EPS: Final = 1e-12
_LOG_FLOOR: Final = 1e-300


@dataclass(frozen=True, slots=True)
class TemperatureParameters:
    """Maps organism/affect state onto a retrieval temperature."""

    base: float = 0.6
    arousal_gain: float = 2.2
    tension_gain: float = 0.8
    energy_gain: float = 0.6
    minimum: float = 1e-3
    maximum: float = 100.0

    def __post_init__(self) -> None:
        numeric = (
            self.base,
            self.arousal_gain,
            self.tension_gain,
            self.energy_gain,
            self.minimum,
            self.maximum,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("temperature parameters must be finite")
        if self.base <= 0.0:
            raise ValueError("base temperature must be finite and positive")
        if not 0.0 < self.minimum < self.maximum:
            raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")


def retrieval_temperature(
    *,
    arousal: float,
    tension: float = 0.0,
    energy: float = 0.5,
    parameters: TemperatureParameters | None = None,
) -> float:
    """Temperature rises with arousal and tension, falls with available energy.

    This is the second, separate role affect plays in retrieval.  Affect
    *position* warps the metric and decides which memories are geodesically
    near; affect *arousal* sets the temperature and decides how sharply the
    system picks among them.  The two compose and do not overlap.
    """

    settings = parameters or TemperatureParameters()
    exponent = (
        settings.arousal_gain * arousal
        + settings.tension_gain * tension
        - settings.energy_gain * energy
    )
    value = settings.base * math.exp(max(-30.0, min(30.0, exponent)))
    return float(min(settings.maximum, max(settings.minimum, value)))


def heat_kernel(
    conductance: np.ndarray,
    diffusion_time: float,
    *,
    tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    """Return ``exp(-t·L)`` for the graph Laplacian of ``conductance``."""

    if diffusion_time <= 0.0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and positive")
    laplacian = graph_laplacian(conductance, tolerance=tolerance)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    if float(np.min(eigenvalues)) < -tolerance:
        raise HDSCInvariantError("graph Laplacian is not positive semidefinite")
    attenuation = np.exp(-diffusion_time * np.maximum(eigenvalues, 0.0))
    kernel: NDArray[np.float64] = (eigenvectors * attenuation) @ eigenvectors.T
    return kernel


def geodesic_detuning(
    kernel: NDArray[np.float64],
    seed_mass: np.ndarray,
    diffusion_time: float,
) -> NDArray[np.float64]:
    """χ² per node, read off the heat kernel via Varadhan's short-time formula.

    ``d(x,y)² = −lim_{t→0} 4t·log p_t(x,y)``.  On a finite graph this is an
    approximation of the manifold statement, not an identity; it is used here
    because it means the multi-hop ripple and the geodesic search are the same
    computation rather than two mechanisms to keep in sync.
    """

    seed = np.asarray(seed_mass, dtype=np.float64)
    if seed.ndim != 1 or seed.size != kernel.shape[0]:
        raise HDSCInvariantError("seed mass must match the node count")
    if np.any(seed < 0.0):
        raise HDSCInvariantError("seed mass must be non-negative")
    total = float(np.sum(seed))
    if total <= _EPS:
        raise HDSCInvariantError("seed mass must be positive somewhere")

    arrived = kernel @ (seed / total)
    arrived = np.maximum(arrived, _LOG_FLOOR)
    detuning: NDArray[np.float64] = -4.0 * diffusion_time * np.log(arrived)
    return np.maximum(detuning, 0.0)


def gaussian_entropy(spread: np.ndarray, dimension: int) -> NDArray[np.float64]:
    """Differential entropy of an isotropic Gaussian with the given spread.

    ``S = ½·log((2πe)^d · det Σ)``.  With ``Σ = σ²I`` this is
    ``d·log σ + const``: entropy grows with the *logarithm* of the spread, not
    proportionally to it.  That is why the caller must standardise before
    combining it with a quadratic detuning term.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    sigma = np.asarray(spread, dtype=np.float64)
    if sigma.ndim != 1:
        raise HDSCInvariantError("spread must be one-dimensional")
    if np.any(sigma < 0.0):
        raise HDSCInvariantError("spread must be non-negative")
    safe = np.maximum(sigma, _EPS)
    entropy: NDArray[np.float64] = dimension * (
        np.log(safe) + 0.5 * math.log(2.0 * math.pi * math.e)
    )
    return entropy


def _standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = values - float(np.mean(values))
    deviation = float(np.std(centred))
    if deviation <= _EPS:
        return np.zeros_like(centred)
    result: NDArray[np.float64] = centred / deviation
    return result


@dataclass(frozen=True, slots=True)
class ResonanceConfig:
    diffusion_time: float = 0.35
    recall_count: int = 3
    # A candidate must be this close in *absolute* geodesic terms before it is
    # allowed to surface, no matter how hot the search is.
    surfacing_detuning_cap: float = 6.0
    # ...and it must stand out from the field by this many standard deviations.
    surfacing_margin: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.diffusion_time) or self.diffusion_time <= 0.0:
            raise ValueError("diffusion_time must be positive")
        if self.recall_count < 1:
            raise ValueError("recall_count must be positive")
        if (
            not math.isfinite(self.surfacing_detuning_cap)
            or self.surfacing_detuning_cap <= 0.0
        ):
            raise ValueError("surfacing_detuning_cap must be positive")
        if not math.isfinite(self.surfacing_margin) or self.surfacing_margin < 0.0:
            raise ValueError("surfacing_margin must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ResonanceResult:
    """The full internal state of one resonance scan.

    Everything here is deliberation, not output.  It belongs in the audit
    ledger and in private prompt context; none of it is user-visible text.
    """

    temperature: float
    detuning: tuple[float, ...]
    entropy: tuple[float, ...]
    free_energy: tuple[float, ...]
    probability: tuple[float, ...]
    selected: tuple[int, ...]
    surfaced: bool
    reason: str

    @property
    def distribution_entropy(self) -> float:
        """Shannon entropy of the selection distribution, in nats.

        This is the observable that separates the three temperature regimes;
        it collapses toward zero as the search becomes rigid.
        """

        probability = np.asarray(self.probability, dtype=np.float64)
        positive = probability[probability > _EPS]
        if positive.size == 0:
            return 0.0
        return float(-np.sum(positive * np.log(positive)))


def resonance_scan(
    conductance: np.ndarray,
    seed_mass: np.ndarray,
    spread: np.ndarray,
    *,
    temperature: float,
    content_dimension: int,
    config: ResonanceConfig | None = None,
    rng: np.random.Generator | None = None,
) -> ResonanceResult:
    """Run one fuzzy resonance scan and return the whole deliberation.

    ``seed_mass`` is the stone: mass placed on whichever nodes the query
    touches directly, which for a vague query may be almost flat.  The ripple
    is the heat kernel; selection is Boltzmann over free energy.
    """

    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    settings = config or ResonanceConfig()

    kernel = heat_kernel(conductance, settings.diffusion_time)
    detuning = geodesic_detuning(kernel, seed_mass, settings.diffusion_time)
    entropy = gaussian_entropy(spread, content_dimension)
    if entropy.size != detuning.size:
        raise HDSCInvariantError("spread and conductance must describe the same nodes")

    # The seeded nodes are the query itself, not something recalled, and their
    # detuning is ~0 by construction.  Leaving them in would let the surfacing
    # gate always pass on the query's own reflection.
    seed = np.asarray(seed_mass, dtype=np.float64)
    candidate = seed <= _EPS
    if not bool(np.any(candidate)):
        raise HDSCInvariantError("every node is seeded; there is nothing to recall")

    # Both terms are standardised so that T ~ 1 is genuinely the transition
    # point; a quadratic detuning and a logarithmic entropy are otherwise on
    # incomparable scales and the temperature axis would be arbitrary.
    detuning_z = _standardize(detuning[candidate])
    entropy_z = _standardize(entropy[candidate])
    free_energy_c = detuning_z - temperature * entropy_z

    exponent = -detuning_z / temperature + entropy_z
    exponent -= float(np.max(exponent))
    weights = np.exp(exponent)
    total = float(np.sum(weights))
    probability_c: NDArray[np.float64] = (
        weights / total if total > _EPS else np.full(weights.shape, 1.0 / weights.size)
    )

    # Scatter the candidate-only vectors back to full node indexing so callers
    # can align them with the topology.
    indices = np.flatnonzero(candidate)
    # Seeded nodes are not candidates.  Infinity reads correctly and, unlike
    # NaN, compares equal to itself, which keeps the result value comparable.
    free_energy = np.full(detuning.shape, np.inf)
    probability = np.zeros(detuning.shape)
    free_energy[indices] = free_energy_c
    probability[indices] = probability_c

    # Anti-confabulation: a hot search may widen what surfaces, but it must not
    # invent a memory when nothing is actually close.  This gate is checked on
    # the raw geodesic distance, before any temperature is applied.
    closest = float(np.min(detuning[candidate]))
    if closest > settings.surfacing_detuning_cap:
        return ResonanceResult(
            temperature=temperature,
            detuning=tuple(float(value) for value in detuning),
            entropy=tuple(float(value) for value in entropy),
            free_energy=tuple(float(value) for value in free_energy),
            probability=tuple(float(value) for value in probability),
            selected=(),
            surfaced=False,
            reason="no candidate within the absolute geodesic cap",
        )
    closest_z = float(np.min(detuning_z))
    if closest_z > -settings.surfacing_margin:
        return ResonanceResult(
            temperature=temperature,
            detuning=tuple(float(value) for value in detuning),
            entropy=tuple(float(value) for value in entropy),
            free_energy=tuple(float(value) for value in free_energy),
            probability=tuple(float(value) for value in probability),
            selected=(),
            surfaced=False,
            reason="no candidate stands out from the detuning background",
        )

    # How many memories surface is itself a function of temperature, and the
    # principled measure is the distribution's perplexity: exp(H) is the
    # effective size of its support.  Cold recall has perplexity 1 and yields
    # exactly one precise memory; a hot scan yields as many as the budget
    # allows.  Drawing a fixed k would pad a near-delta distribution with
    # meaningless tail nodes.
    positive = probability_c[probability_c > _EPS]
    support = int(positive.size)
    shannon = float(-np.sum(positive * np.log(positive)))
    perplexity = round(math.exp(shannon))
    count = max(1, min(settings.recall_count, perplexity, support))

    generator = rng if rng is not None else np.random.default_rng(0)
    ordered: tuple[int, ...]
    if count == 1:
        ordered = (int(indices[int(np.argmax(probability_c))]),)
    else:
        # Sampling rather than argmax is what produces the phenomenology at
        # high temperature: "several things flashed through my mind" is drawing
        # samples from a high-entropy distribution, not picking the single
        # vaguest memory.
        chosen = generator.choice(probability_c.size, size=count, replace=False, p=probability_c)
        ordered = tuple(
            int(indices[position]) for position in sorted(chosen, key=lambda i: -probability_c[i])
        )

    return ResonanceResult(
        temperature=temperature,
        detuning=tuple(float(value) for value in detuning),
        entropy=tuple(float(value) for value in entropy),
        free_energy=tuple(float(value) for value in free_energy),
        probability=tuple(float(value) for value in probability),
        selected=ordered,
        surfaced=True,
        reason="resonance above surfacing floor",
    )


__all__ = [
    "CLAIM_LEVEL",
    "MODEL_ID",
    "SELECTION_RULE",
    "ResonanceConfig",
    "ResonanceResult",
    "TemperatureParameters",
    "gaussian_entropy",
    "geodesic_detuning",
    "heat_kernel",
    "resonance_scan",
    "retrieval_temperature",
]
