"""Write, activate, radiate, and project the persistent HDSC trace space."""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np

from ssa.adapters.embedding import EmbeddingService
from ssa.clock import Clock
from ssa.config import HDSCConfig, RetrievalConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import SourceKind
from ssa.domain.events import Event
from ssa.domain.traces import (
    ActivatedTrace,
    Trace,
    TraceLink,
    TraceNode,
    TraceSpaceSnapshot,
    TraceSpaceStabilityAudit,
)
from ssa.hdsc.active_space import (
    MODEL_ID as H2_MODEL_ID,
)
from ssa.hdsc.active_space import (
    H2Audit,
    H2Candidate,
    H2Config,
    H2Result,
    H2State,
    bounded_active_shadow_step,
)
from ssa.hdsc.directed_transport import build_directed_rates
from ssa.ids import IdGenerator
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid

_DAY_MS = 24 * 60 * 60 * 1000
_H2_PARTITION_VERSION = "simhash-hyperplanes-v1"
_ShadowStatus = Literal["disabled", "awaiting-replay", "awaiting-input", "passed", "failed"]
_AuditPhase = Literal["awaiting-input", "replay", "post-write", "failed"]
_ClosedLoopGate = Literal["not-measured", "conditional-pass", "fail"]


class TraceSpaceService:
    """Reproducible legacy SSA-A0 baseline retained inside HDSC."""

    def __init__(
        self,
        *,
        embedding: EmbeddingService,
        repository: SqliteTraceRepository,
        clock: Clock,
        ids: IdGenerator,
        config: RetrievalConfig,
        embedding_model: str,
        embedding_dim: int,
        hdsc_config: HDSCConfig | None = None,
    ) -> None:
        self._embedding = embedding
        self._repo = repository
        self._clock = clock
        self._ids = ids
        self._config = config
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        experimental = hdsc_config or HDSCConfig()
        self._h2_enabled = experimental.h2_shadow_enabled
        self._h2_cluster_bits = experimental.h2_cluster_bits
        self._h2_hyperplanes = _fixed_hyperplanes(embedding_dim, self._h2_cluster_bits)
        self._h2_partition_id = _semantic_partition_id(
            embedding_model,
            embedding_dim,
            self._h2_cluster_bits,
            self._h2_hyperplanes,
        )
        self._h2_config = H2Config(
            semantic_partition_id=self._h2_partition_id,
            capacity=experimental.h2_active_capacity,
            candidate_top_k=experimental.h2_candidate_top_k,
            retention=experimental.h2_retention,
            injection_rate=experimental.h2_injection_rate,
            hysteresis=experimental.h2_hysteresis,
            score_lipschitz_bound=experimental.h2_score_lipschitz_bound,
            gain_margin=experimental.h2_gain_margin,
            mass_tolerance=experimental.h2_mass_tolerance,
        )
        self._h2_states: dict[str, H2State] = {}
        self._h2_audits: dict[str, H2Audit] = {}
        self._h2_errors: dict[str, str] = {}
        self._h2_phases: dict[str, str] = {}

    def write_turn(
        self,
        user_event: Event,
        agent_event: Event,
        appraisal: AppraisalResult,
        *,
        advance_shadow: bool = True,
    ) -> Trace:
        """Append one immutable episode trace for a completed interaction."""
        content = f"User: {user_event.content}\nAgent: {agent_event.content}"
        vector = self._embedding.embed_one(content)
        neighbors = self._repo.find_similar(
            vector.as_bytes(),
            conversation_id=user_event.conversation_id,
            limit=5,
        )
        now_ms = self._clock.now_ms()
        trace_id = self._ids.new()
        importance = _clip(
            0.20
            + 0.30 * appraisal.novelty
            + 0.25 * appraisal.relationship_relevance
            + 0.15 * appraisal.urgency
            + 0.10 * abs(appraisal.valence_signal)
        )
        trace = Trace(
            id=trace_id,
            conversation_id=user_event.conversation_id,
            correlation_id=user_event.correlation_id,
            input_event_id=user_event.id,
            output_event_id=agent_event.id,
            content=content,
            content_type="episode",
            source_kind=SourceKind.SYSTEM_DERIVED,
            importance=importance,
            valence=appraisal.valence_signal,
            arousal=appraisal.arousal_signal,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            vec_rowid=trace_vec_rowid(trace_id),
            created_at_ms=now_ms,
        )
        self._repo.insert(trace, vector.as_bytes())

        for neighbor, similarity in neighbors:
            if similarity < 0.40:
                continue
            self._repo.add_link(
                TraceLink(
                    source_trace_id=trace.id,
                    target_trace_id=neighbor.id,
                    weight=similarity,
                    created_at_ms=now_ms,
                )
            )
            self._repo.add_link(
                TraceLink(
                    source_trace_id=neighbor.id,
                    target_trace_id=trace.id,
                    weight=similarity,
                    created_at_ms=now_ms,
                )
            )
            self._repo.add_link(
                TraceLink(
                    source_trace_id=neighbor.id,
                    target_trace_id=trace.id,
                    link_type="temporal-forward",
                    weight=similarity,
                    created_at_ms=now_ms,
                )
            )
        if advance_shadow:
            self._advance_h2_shadow(trace)
        return trace

    def replay_h2_shadow(self, conversation_id: str) -> None:
        """Rebuild the path-dependent H2 shadow state from the immutable archive."""
        if not self._h2_enabled:
            return
        self._h2_states[conversation_id] = H2State.initial(self._h2_config)
        self._h2_audits.pop(conversation_id, None)
        self._h2_errors.pop(conversation_id, None)
        self._h2_phases[conversation_id] = "replay"
        try:
            expected_count = self._repo.count(conversation_id)
            archive = self._repo.all_with_embeddings(conversation_id)
            if len(archive) != expected_count:
                raise ValueError(
                    "H2 replay requires one persisted embedding for every archived trace"
                )
            for index, (trace, embedding) in enumerate(archive, 1):
                self._run_h2_step(
                    trace,
                    embedding,
                    archive_count=index,
                    phase="replay",
                )
            if not archive:
                self._h2_phases[conversation_id] = "awaiting-input"
        except Exception as exc:
            self._h2_errors[conversation_id] = f"{type(exc).__name__}: {exc}"
            self._h2_phases[conversation_id] = "failed"

    def activate(
        self,
        query_text: str,
        *,
        conversation_id: str,
        max_main: int = 8,
        max_total: int = 12,
    ) -> list[ActivatedTrace]:
        """Activate top traces, then perform one bounded radiation hop."""
        if self._repo.count(conversation_id) == 0:
            return []
        now_ms = self._clock.now_ms()
        query_vector = self._embedding.embed_one(query_text)
        candidates = [
            (trace, _cosine(query_vector.values, vector))
            for trace, vector in self._repo.all_with_embeddings(conversation_id)
        ]
        scored: list[ActivatedTrace] = []
        for trace, similarity in candidates:
            freshness = _freshness(trace.created_at_ms, now_ms)
            score = similarity * freshness * trace.importance
            scored.append(
                ActivatedTrace(
                    trace=trace,
                    rank=1,
                    score=score,
                    semantic_similarity=similarity,
                    freshness=freshness,
                    importance_factor=trace.importance,
                    activation_kind="main",
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        main = scored[:max_main]

        expanded: dict[str, ActivatedTrace] = {item.trace.id: item for item in main}
        for parent in main:
            for link in self._repo.links_from(
                parent.trace.id,
                link_types=("semantic",),
            ):
                if link.target_trace_id in expanded:
                    continue
                target = self._repo.get(link.target_trace_id)
                if target is None or target.conversation_id != conversation_id:
                    continue
                freshness = _freshness(target.created_at_ms, now_ms)
                radiation_score = parent.score * link.weight * self._config.radiation_decay
                expanded[target.id] = ActivatedTrace(
                    trace=target,
                    rank=1,
                    score=radiation_score,
                    semantic_similarity=min(1.0, parent.semantic_similarity * link.weight),
                    freshness=freshness,
                    importance_factor=target.importance,
                    activation_kind="radiation",
                )

        ordered = sorted(expanded.values(), key=lambda item: item.score, reverse=True)[:max_total]
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(ordered, 1)]

    def record_activation(
        self,
        *,
        conversation_id: str,
        correlation_id: str,
        query_event_id: str,
        activations: list[ActivatedTrace],
    ) -> None:
        self._repo.record_activations(
            activation_ids=[self._ids.new() for _ in activations],
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            query_event_id=query_event_id,
            activations=activations,
            activated_at_ms=self._clock.now_ms(),
        )

    def snapshot(
        self,
        conversation_id: str,
        *,
        latest_query: str = "",
        limit: int = 120,
    ) -> TraceSpaceSnapshot:
        traces = self._repo.recent(conversation_id, limit=limit)
        activations = {
            item.trace.id: item for item in self._repo.latest_activations(conversation_id)
        }
        coordinates, projection = self._project(traces)
        now_ms = self._clock.now_ms()
        nodes = [
            TraceNode(
                trace_id=trace.id,
                content=trace.content,
                content_type=trace.content_type,
                source_kind=trace.source_kind,
                x=coordinates[trace.id][0],
                y=coordinates[trace.id][1],
                importance=trace.importance,
                freshness=_freshness(trace.created_at_ms, now_ms),
                activation_score=(activations[trace.id].score if trace.id in activations else 0.0),
                activation_kind=(
                    activations[trace.id].activation_kind if trace.id in activations else None
                ),
                created_at_ms=trace.created_at_ms,
            )
            for trace in traces
        ]
        trace_ids = [trace.id for trace in traces]
        archive_count = self._repo.count(conversation_id)
        stability_audit, shadow_status = self._h2_snapshot(conversation_id, archive_count)
        return TraceSpaceSnapshot(
            nodes=nodes,
            links=self._repo.links_among(trace_ids),
            total_traces=archive_count,
            latest_query=latest_query,
            projection=projection,
            legacy_activated_count=len(activations),
            shadow_model_id=(H2_MODEL_ID if self._h2_enabled else None),
            shadow_status=shadow_status,
            shadow_affects_prompt=False,
            stability_audit=stability_audit,
        )

    def directed_rate_matrix(
        self,
        conversation_id: str,
        *,
        limit: int = 120,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Return the full multi-relation topology for H1D shadow evaluation."""
        traces = self._repo.recent(conversation_id, limit=limit)
        node_ids = tuple(trace.id for trace in traces)
        if not node_ids:
            return (), np.zeros((0, 0), dtype=np.float64)
        links = self._repo.links_among(list(node_ids))
        rates = build_directed_rates(
            node_ids,
            [(link.source_trace_id, link.target_trace_id, link.weight) for link in links],
        )
        return node_ids, rates

    def _advance_h2_shadow(
        self,
        trace: Trace,
    ) -> None:
        if not self._h2_enabled:
            return
        conversation_id = trace.conversation_id
        try:
            archive_count = self._repo.count(conversation_id)
            state = self._h2_states.get(conversation_id)
            processed_count = state.step if state is not None else 0
            if processed_count != archive_count - 1:
                self.replay_h2_shadow(conversation_id)
                return
            embedding = self._repo.embedding(trace)
            if embedding is None:
                raise ValueError("H2 shadow requires the archived trace embedding")
            self._run_h2_step(
                trace,
                embedding,
                archive_count=archive_count,
                phase="post-write",
            )
        except Exception as exc:
            # H2 is observational: its failure is visible but never blocks serving.
            self._h2_errors[conversation_id] = f"{type(exc).__name__}: {exc}"
            self._h2_phases[conversation_id] = "failed"

    def _run_h2_step(
        self,
        trace: Trace,
        embedding: list[float],
        *,
        archive_count: int,
        phase: str,
    ) -> H2Result:
        state = self._h2_states.get(trace.conversation_id) or H2State.initial(self._h2_config)
        candidate = self._h2_candidate(trace, embedding, state=state)
        result = bounded_active_shadow_step(
            state,
            [candidate],
            self._h2_config,
            archive_trace_count=archive_count,
        )
        self._h2_states[trace.conversation_id] = result.state
        self._h2_audits[trace.conversation_id] = result.audit
        self._h2_errors.pop(trace.conversation_id, None)
        self._h2_phases[trace.conversation_id] = phase
        return result

    def _h2_candidate(
        self,
        trace: Trace,
        embedding: list[float],
        *,
        state: H2State,
    ) -> H2Candidate:
        if (
            trace.embedding_model != self._embedding_model
            or trace.embedding_dim != self._embedding_dim
        ):
            raise ValueError("H2 trace embedding partition does not match the configured codebook")
        vector = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)) or norm == 0.0:
            raise ValueError("H2 requires a finite non-zero trace embedding")
        normalized = vector / norm
        fingerprint_bytes = np.round(normalized, 6).astype("<f4").tobytes()
        fingerprint = hashlib.sha256(fingerprint_bytes).hexdigest()
        projections = self._h2_hyperplanes @ normalized
        code = "".join("1" if value >= 0.0 else "0" for value in projections)
        signs = np.where(projections >= 0.0, 1.0, -1.0)
        prototype = signs @ self._h2_hyperplanes
        prototype_norm = float(np.linalg.norm(prototype))
        if prototype_norm == 0.0:
            raise ValueError("H2 codebook produced a zero prototype")
        prototype /= prototype_norm
        cluster_id = f"simhash-{code}"
        revision = self._h2_partition_id
        already_active = any(
            cluster.cluster_id == cluster_id and cluster.cluster_revision == revision
            for cluster in state.clusters
        )
        return H2Candidate(
            trace_id=trace.id,
            semantic_fingerprint=fingerprint,
            cluster_id=cluster_id,
            cluster_revision=revision,
            prototype=tuple(float(value) for value in prototype),
            relevance=trace.importance,
            novelty=(0.0 if already_active else 1.0),
            cluster_assignment_margin=float(np.min(np.abs(projections))),
        )

    def _h2_snapshot(
        self,
        conversation_id: str,
        archive_count: int,
    ) -> tuple[TraceSpaceStabilityAudit | None, _ShadowStatus]:
        if not self._h2_enabled:
            return None, "disabled"
        error = self._h2_errors.get(conversation_id)
        if error is not None:
            return (
                TraceSpaceStabilityAudit(
                    phase="failed",
                    local_gate="fail",
                    closed_loop_gate="not-measured",
                    evaluated_archive_count=None,
                    active_capacity=self._h2_config.capacity,
                    certificate_reason=error,
                ),
                "failed",
            )
        audit = self._h2_audits.get(conversation_id)
        state = self._h2_states.get(conversation_id)
        if audit is None or state is None:
            status: _ShadowStatus = "awaiting-replay" if archive_count else "awaiting-input"
            return None, status
        if state.step != archive_count or audit.archive_trace_count != archive_count:
            return None, "awaiting-replay"
        closed_loop_gate: _ClosedLoopGate = (
            "conditional-pass"
            if audit.certificate_status == "pass"
            else "fail"
            if audit.certificate_status == "fail"
            else "not-measured"
        )
        phase = self._h2_phases.get(conversation_id, "post-write")
        read_phase: _AuditPhase = "replay" if phase == "replay" else "post-write"
        return (
            TraceSpaceStabilityAudit(
                phase=read_phase,
                local_gate="pass",
                closed_loop_gate=closed_loop_gate,
                evaluated_archive_count=audit.archive_trace_count,
                active_count=len(state.clusters),
                active_capacity=audit.capacity,
                active_mass=1.0 - state.null_mass,
                null_mass=state.null_mass,
                mass_residual=audit.mass_residual,
                contraction_bound=audit.local_memory_contraction_bound,
                active_set_churn=audit.support_churn,
                joint_spectral_radius=audit.small_gain_spectral_radius,
                certified_radius=audit.certified_radius,
                duplicate_count=audit.duplicate_count,
                dropped_mass=audit.dropped_mass,
                certificate_reason=audit.certificate_reason,
            ),
            "passed",
        )

    def _project(self, traces: list[Trace]) -> tuple[dict[str, tuple[float, float]], str]:
        if not traces:
            return {}, "embedding-pca"
        embeddings = [self._repo.embedding(trace) for trace in traces]
        if all(vector is not None for vector in embeddings):
            matrix = np.asarray(embeddings, dtype=np.float64)
            if len(traces) == 1:
                return {traces[0].id: (0.5, 0.5)}, "embedding-pca"
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            _u, _singular, axes = np.linalg.svd(centered, full_matrices=False)
            dimensions = min(2, axes.shape[0])
            projected = centered @ axes[:dimensions].T
            if dimensions == 1:
                projected = np.column_stack((projected[:, 0], np.zeros(len(traces))))
            normalized = _normalize_projection(projected)
            return {
                trace.id: (float(normalized[index, 0]), float(normalized[index, 1]))
                for index, trace in enumerate(traces)
            }, "embedding-pca"
        return {trace.id: _fallback_coordinate(trace.id) for trace in traces}, "hash-fallback"


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _freshness(created_at_ms: int, now_ms: int) -> float:
    age_days = max(0, now_ms - created_at_ms) / _DAY_MS
    return math.exp(-age_days / 30.0)


def _normalize_projection(projected: np.ndarray) -> np.ndarray:
    result = np.full_like(projected, 0.5, dtype=np.float64)
    for axis in range(2):
        low = float(projected[:, axis].min())
        high = float(projected[:, axis].max())
        if high - low > 1e-12:
            result[:, axis] = 0.05 + 0.90 * (projected[:, axis] - low) / (high - low)
    return result  # type: ignore[no-any-return]


def _fallback_coordinate(trace_id: str) -> tuple[float, float]:
    digest = hashlib.sha256(trace_id.encode("utf-8")).digest()
    return (0.05 + digest[0] / 255 * 0.90, 0.05 + digest[1] / 255 * 0.90)


def _fixed_hyperplanes(dimension: int, bits: int) -> np.ndarray:
    """Build a deterministic unit-norm SimHash codebook independent of archive data."""
    planes: np.ndarray = np.empty((bits, dimension), dtype=np.float64)
    bytes_per_plane = (dimension + 7) // 8
    for plane_index in range(bits):
        payload = bytearray()
        counter = 0
        while len(payload) < bytes_per_plane:
            seed = f"hdsc-h2-simhash-v1:{dimension}:{plane_index}:{counter}".encode("ascii")
            payload.extend(hashlib.sha256(seed).digest())
            counter += 1
        unpacked = np.unpackbits(np.frombuffer(bytes(payload[:bytes_per_plane]), dtype=np.uint8))
        planes[plane_index] = np.where(unpacked[:dimension] == 1, 1.0, -1.0)
    planes /= math.sqrt(dimension)
    return planes


def _semantic_partition_id(
    embedding_model: str,
    dimension: int,
    bits: int,
    hyperplanes: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(_H2_PARTITION_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(embedding_model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(dimension).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(bits).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(hyperplanes, dtype="<f8").tobytes())
    return f"{_H2_PARTITION_VERSION}:{digest.hexdigest()}"


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


__all__ = ["TraceSpaceService"]
