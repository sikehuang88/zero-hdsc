"""Deterministic NumPy implementations for the SEOS primitive registry.

Every implementation here is pure or read-only, needs no language model, and
derives all randomness from content digests rather than graph structure.  That
makes offline evaluation bit-reproducible even though the serving model itself
cannot be seeded (the DeepSeek adapter rejects a ``seed`` parameter), and it
keeps inlining semantics-preserving: a node's result never depends on where the
node sits in the DAG.

Runtime representations
-----------------------
``Ev``        :class:`ArchivedEvent`
``set[Ev]``   ``tuple[ArchivedEvent, ...]``, ordered by ``(created_at_ms, id)``
``Z``         ``NDArray[np.float64]`` of length ``dimension``
``Hv``        :class:`~ssa.hdsc.hypervector.PackedHypervector`
``Gr``        ``NDArray[np.float64]`` of shape ``(dimension, dimension)``
``Sc``        ``float`` constrained to ``[0, 1]``
``Aff``       ``tuple[float, float, float, float]``

``set`` types are represented by ordered tuples on purpose.  Python sets have
no reproducible iteration order for these element types, and reproducibility is
a hard requirement of the evolution loop.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ssa.hdsc import hypervector as hv
from ssa.hdsc.hypervector import PackedHypervector
from ssa.hdsc.interpreter import OperatorImplementation

DEFAULT_DIMENSION: Final = 256
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class ArchivedEvent:
    """One immutable archived event as seen by an offline candidate program."""

    event_id: str
    created_at_ms: int
    actor: str
    content: str

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if self.created_at_ms < 0:
            raise ValueError("created_at_ms must be non-negative")

    @property
    def content_digest(self) -> str:
        payload = f"{self.actor}\x1f{self.content}".encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def digest(self) -> str:
        """Identity digest used by replay/cache keys, including event position."""
        payload = (
            f"{self.event_id}\x1f{self.created_at_ms}\x1f{self.actor}\x1f{self.content}"
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def order_events(events: Sequence[ArchivedEvent]) -> tuple[ArchivedEvent, ...]:
    """Return events in the canonical deterministic order."""

    return tuple(sorted(events, key=lambda item: (item.created_at_ms, item.event_id)))


def _param(parameters: tuple[tuple[str, object], ...], name: str, default: float) -> float:
    for key, value in parameters:
        if key == name and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    return float(min(1.0, max(0.0, value)))


class RetrodictionImplementations:
    """Implementation set bound to one hypervector/latent dimension."""

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension
        self._event_vectors: dict[str, PackedHypervector] = {}

    # -- representation -------------------------------------------------

    def event_vector(self, event: ArchivedEvent) -> PackedHypervector:
        cached = self._event_vectors.get(event.content_digest)
        if cached is not None:
            return cached
        vector = hv.semantic_random(
            self.dimension,
            semantic_role="archived-event",
            input_digest=event.content_digest,
        )
        self._event_vectors[event.content_digest] = vector
        return vector

    def _zeros(self) -> NDArray[np.float64]:
        return np.zeros(self.dimension, dtype=np.float64)

    def _to_hv(self, values: NDArray[np.float64]) -> PackedHypervector:
        signs = np.where(values >= 0.0, 1, -1).astype(np.int8)
        return PackedHypervector.from_bipolar(signs)

    # -- operator bodies ------------------------------------------------

    def embed(self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]) -> object:
        event = args[0]
        assert isinstance(event, ArchivedEvent)
        return self.event_vector(event).to_bipolar().astype(np.float64)

    def allocate_mass(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        events = args[0]
        assert isinstance(events, tuple)
        if not events:
            return self._zeros()
        half_life = max(1.0, _param(parameters, "half_life_steps", 8.0))
        total = self._zeros()
        newest = max(item.created_at_ms for item in events)
        for event in events:
            age_steps = (newest - event.created_at_ms) / 1000.0
            weight = float(0.5 ** (age_steps / half_life))
            total += weight * self.event_vector(event).to_bipolar().astype(np.float64)
        return total

    def quantize(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        latent = args[0]
        assert isinstance(latent, np.ndarray)
        return self._to_hv(latent)

    def activate(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        query, events = args[0], args[1]
        assert isinstance(query, PackedHypervector)
        assert isinstance(events, tuple)
        if not events:
            return ()
        threshold = _param(parameters, "threshold", 0.0)
        limit = int(_param(parameters, "max_events", 64.0))
        scored = [
            (hv.similarity(self.event_vector(event), query), event.created_at_ms, event)
            for event in events
        ]
        kept = [item for item in scored if item[0] >= threshold]
        if not kept:
            kept = scored
        kept.sort(key=lambda item: (-item[0], -item[1], item[2].event_id))
        return order_events([item[2] for item in kept[: max(1, limit)]])

    def similarity(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        left, right = args[0], args[1]
        assert isinstance(left, PackedHypervector)
        assert isinstance(right, PackedHypervector)
        return _clip01((hv.similarity(left, right) + 1.0) / 2.0)

    def z_energy(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        latent = args[0]
        assert isinstance(latent, np.ndarray)
        norm = float(np.linalg.norm(latent)) / float(np.sqrt(self.dimension))
        return _clip01(float(np.tanh(norm)))

    def calibrate(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        score = args[0]
        assert isinstance(score, float)
        scale = _param(parameters, "scale", 4.0)
        bias = _param(parameters, "bias", 0.0)
        logit = scale * (score - 0.5) + bias
        return _clip01(float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))))

    def modulate(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        affect, score = args[0], args[1]
        assert isinstance(affect, tuple)
        assert isinstance(score, float)
        gain = _param(parameters, "gain", 0.25)
        valence = float(affect[0]) if affect else 0.0
        return _clip01(score + gain * valence * (1.0 - score))

    def bind(self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]) -> object:
        left, right = args[0], args[1]
        assert isinstance(left, PackedHypervector)
        assert isinstance(right, PackedHypervector)
        return hv.bind(left, right)

    def unbind(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        bound, key = args[0], args[1]
        assert isinstance(bound, PackedHypervector)
        assert isinstance(key, PackedHypervector)
        return hv.unbind(bound, key)

    def permute(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        vector = args[0]
        assert isinstance(vector, PackedHypervector)
        return hv.permute(vector, semantic_role="seos-permute")

    def bundle(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        vectors = args[0]
        assert isinstance(vectors, tuple)
        if not vectors:
            return self._to_hv(self._zeros())
        return hv.bundle(vectors, semantic_role="seos-bundle")

    def topk(self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]) -> object:
        vectors, threshold = args[0], args[1]
        assert isinstance(vectors, tuple)
        assert isinstance(threshold, float)
        limit = max(1, round(threshold * len(vectors))) if vectors else 0
        ordered = sorted(vectors, key=lambda item: item.digest)
        return tuple(ordered[:limit])

    def rank_by(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        vectors = args[0]
        assert isinstance(vectors, tuple)
        return tuple(sorted(vectors, key=lambda item: item.digest))

    # -- latent dynamics ------------------------------------------------

    def leak(self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]) -> object:
        latent = args[0]
        assert isinstance(latent, np.ndarray)
        return float(_param(parameters, "beta", 0.9)) * latent

    def project(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        latent = args[0]
        assert isinstance(latent, np.ndarray)
        rotation = hv.semantic_random(
            self.dimension,
            semantic_role="seos-projection",
            input_digest=hashlib.sha256(b"seos-projection-v1").hexdigest(),
        ).to_bipolar()
        return latent * rotation.astype(np.float64)

    def integrate(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        left, right = args[0], args[1]
        assert isinstance(left, np.ndarray)
        assert isinstance(right, np.ndarray)
        return left + right

    def liquid(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        state, drive = args[0], args[1]
        assert isinstance(state, np.ndarray)
        assert isinstance(drive, np.ndarray)
        alpha = min(1.0, max(0.0, _param(parameters, "alpha", 0.3)))
        return (1.0 - alpha) * state + alpha * drive

    # -- graph ----------------------------------------------------------

    def conductance(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        events = args[0]
        assert isinstance(events, tuple)
        matrix: NDArray[np.float64] = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        if not events:
            return matrix
        for event in events:
            vector = self.event_vector(event).to_bipolar().astype(np.float64)
            matrix += np.outer(vector, vector)
        return matrix / float(len(events) * self.dimension)

    def directed_rates(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        events = args[0]
        assert isinstance(events, tuple)
        matrix: NDArray[np.float64] = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        ordered = order_events(events)
        if len(ordered) < 2:
            return matrix
        for earlier, later in pairwise(ordered):
            source = self.event_vector(earlier).to_bipolar().astype(np.float64)
            target = self.event_vector(later).to_bipolar().astype(np.float64)
            matrix += np.outer(target, source)
        return matrix / float((len(ordered) - 1) * self.dimension)

    def laplacian(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        graph = args[0]
        assert isinstance(graph, np.ndarray)
        weights = np.abs(graph)
        return np.diag(weights.sum(axis=1)) - graph

    def propagate(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        graph, latent = args[0], args[1]
        assert isinstance(graph, np.ndarray)
        assert isinstance(latent, np.ndarray)
        step = _param(parameters, "diffusion_time", 0.25)
        return latent - step * (graph @ latent)

    def propagate_directed(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        return self.propagate(args, parameters)

    def rewire(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        graph, threshold = args[0], args[1]
        assert isinstance(graph, np.ndarray)
        assert isinstance(threshold, float)
        scale = float(np.abs(graph).max(initial=0.0))
        cutoff = threshold * scale
        return np.where(np.abs(graph) >= cutoff, graph, 0.0)

    def community(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        graph = args[0]
        assert isinstance(graph, np.ndarray)
        half = self.dimension // 2
        return (graph[:half, :half].copy(), graph[half:, half:].copy())

    def decay(self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]) -> object:
        graph = args[0]
        assert isinstance(graph, np.ndarray)
        return float(_param(parameters, "lambda", 0.9)) * graph

    # -- read-only state ------------------------------------------------

    def appraise(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        event = args[0]
        assert isinstance(event, ArchivedEvent)
        digest = event.digest
        valence = (int(digest[:4], 16) / 0xFFFF) * 2.0 - 1.0
        arousal = int(digest[4:8], 16) / 0xFFFF
        tension = int(digest[8:12], 16) / 0xFFFF
        uncertainty = int(digest[12:16], 16) / 0xFFFF
        return (valence, arousal, tension, uncertainty)

    def blend(self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]) -> object:
        left, right = args[0], args[1]
        assert isinstance(left, tuple)
        assert isinstance(right, tuple)
        weight = min(1.0, max(0.0, _param(parameters, "weight", 0.5)))
        return tuple(
            (1.0 - weight) * float(a) + weight * float(b) for a, b in zip(left, right, strict=True)
        )

    def relation_update(
        self, args: tuple[object, ...], parameters: tuple[tuple[str, object], ...]
    ) -> object:
        relation = args[0]
        assert isinstance(relation, tuple)
        return relation

    def as_mapping(self) -> Mapping[str, OperatorImplementation]:
        """Return the operator-id to implementation mapping."""

        return {
            "embed": self.embed,
            "allocate_mass": self.allocate_mass,
            "quantize": self.quantize,
            "activate": self.activate,
            "similarity": self.similarity,
            "z_energy": self.z_energy,
            "calibrate": self.calibrate,
            "modulate": self.modulate,
            "bind": self.bind,
            "unbind": self.unbind,
            "permute": self.permute,
            "bundle": self.bundle,
            "topk": self.topk,
            "rank_by": self.rank_by,
            "leak": self.leak,
            "project": self.project,
            "integrate": self.integrate,
            "liquid": self.liquid,
            "conductance": self.conductance,
            "directed_rates": self.directed_rates,
            "laplacian": self.laplacian,
            "propagate": self.propagate,
            "propagate_directed": self.propagate_directed,
            "rewire": self.rewire,
            "community": self.community,
            "decay": self.decay,
            "appraise": self.appraise,
            "blend": self.blend,
            "relation_update": self.relation_update,
        }


__all__ = [
    "DEFAULT_DIMENSION",
    "ArchivedEvent",
    "RetrodictionImplementations",
    "order_events",
]
