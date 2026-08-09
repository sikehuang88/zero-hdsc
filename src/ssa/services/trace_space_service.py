"""Write, activate, radiate, and project the persistent HDSC trace space."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ssa.adapters.embedding import EmbeddingService
from ssa.clock import Clock
from ssa.config import HDSCConfig, RetrievalConfig, WarpedResonanceMode
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import SourceKind
from ssa.domain.events import Event
from ssa.domain.traces import (
    ActivatedTrace,
    ResonanceCandidateAudit,
    ResonanceRecallAudit,
    Trace,
    TraceLink,
    TraceLinkType,
    TraceNode,
    TraceSpaceSnapshot,
    TraceSpaceStabilityAudit,
    WarpedResonanceShadowAudit,
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
from ssa.hdsc.resonance import (
    RecallState,
    ResonanceConfig,
    arc_shape_distance,
    effective_hop_budget,
    gate_resonance,
    modulated_rates,
    resonance_metrics,
    state_metric_tensor,
    state_modulated_gains,
    strongest_paths,
    transport_coupling,
)
from ssa.hdsc.warped_retrieval import (
    MODEL_ID as WARPED_MODEL_ID,
)
from ssa.hdsc.warped_retrieval import (
    AdaptiveWarpedRetriever,
    WarpedRetrievalConfig,
    WarpedRetrievalRequest,
    WarpedRetrievalStrategy,
)
from ssa.ids import IdGenerator
from ssa.storage.trace_repository import SqliteTraceRepository, trace_vec_rowid

_DAY_MS = 24 * 60 * 60 * 1000
_H2_PARTITION_VERSION = "simhash-hyperplanes-v1"
_ShadowStatus = Literal["disabled", "awaiting-replay", "awaiting-input", "passed", "failed"]
_AuditPhase = Literal["awaiting-input", "replay", "post-write", "failed"]
_ClosedLoopGate = Literal["not-measured", "conditional-pass", "fail"]


class TraceSpaceService:
    """Persistent trace substrate with live H1D recall and shadow challengers."""

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
        warped_retriever: WarpedRetrievalStrategy | None = None,
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
        self._resonance_config = ResonanceConfig(
            hop_budget=experimental.resonance_hop_budget,
            capacity=experimental.h2_active_capacity + experimental.h2_candidate_top_k,
            emergence_ratio=experimental.resonance_emergence_ratio,
            background_floor=experimental.resonance_background_floor,
            min_semantic_support=experimental.resonance_min_semantic_support,
            content_weight=experimental.resonance_content_weight,
            affect_weight=experimental.resonance_affect_weight,
            relation_weight=experimental.resonance_relation_weight,
            situation_weight=experimental.resonance_situation_weight,
            arc_weight=experimental.resonance_arc_weight,
        )
        self._resonance_enabled = experimental.resonance_enabled
        self._resonance_audits: dict[str, ResonanceRecallAudit] = {}
        self._warped_mode = experimental.warped_resonance_mode
        self._warped_max_nodes = experimental.warped_resonance_max_nodes
        self._warped_config = WarpedRetrievalConfig(
            low_information=experimental.warped_low_information,
            high_information=experimental.warped_high_information,
            warp_beta=experimental.warped_beta,
            bandwidth=experimental.warped_bandwidth,
            diffusion_time=experimental.warped_diffusion_time,
            temperature=experimental.warped_temperature,
            recall_count=experimental.warped_recall_count,
            surfacing_detuning_cap=experimental.warped_detuning_cap,
            surfacing_margin=experimental.warped_surfacing_margin,
        )
        self._warped_retriever = warped_retriever or (
            AdaptiveWarpedRetriever(self._warped_config)
            if self._warped_mode != WarpedResonanceMode.DISABLED
            else None
        )

    def write_turn(
        self,
        user_event: Event,
        agent_event: Event,
        appraisal: AppraisalResult,
        *,
        tension: float = 0.0,
        advance_shadow: bool = True,
    ) -> Trace:
        """Append one immutable episode trace for a completed interaction."""
        content = f"User: {user_event.content}\nAgent: {agent_event.content}"
        vector = self._embedding.embed_one(content)
        previous_traces = self._repo.recent(user_event.conversation_id, limit=32)
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
            tension=_clip(tension),
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
                    link_type=TraceLinkType.TEMPORAL,
                    weight=similarity,
                    created_at_ms=now_ms,
                )
            )
        self._add_typed_links(trace, previous_traces, user_event, agent_event, now_ms)
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

    def activate_resonant(
        self,
        query_text: str,
        *,
        conversation_id: str,
        state: RecallState | None = None,
        max_total: int = 12,
    ) -> list[ActivatedTrace]:
        """Perform bounded typed multi-hop recall with an explicit empty result."""
        if not self._resonance_enabled or self._repo.count(conversation_id) == 0:
            return []
        recall_state = state or RecallState()
        archive = self._repo.all_with_embeddings(conversation_id)
        if not archive:
            return []
        traces = [item[0] for item in archive]
        embeddings = [np.asarray(item[1], dtype=np.float64) for item in archive]
        query_vector = np.asarray(self._embedding.embed_one(query_text).values, dtype=np.float64)
        similarities = np.asarray(
            [_cosine(query_vector, vector) for vector in embeddings], dtype=np.float64
        )
        semantic_support = np.clip(similarities, 0.0, 1.0)
        warped_shadow = self._run_warped_evaluation(
            traces,
            embeddings,
            query_vector,
            semantic_support,
            recall_state,
        )
        source = semantic_support.copy()
        if recall_state.origin_intent:
            ages = np.asarray(
                [trace.created_at_ms - traces[0].created_at_ms for trace in traces],
                dtype=np.float64,
            )
            span = max(1.0, float(ages[-1]))
            source = np.maximum(source, 0.25 * (1.0 - ages / span))
        else:
            source[semantic_support < self._resonance_config.min_semantic_support] = 0.0
        if float(source.sum()) <= 0.0:
            warped_activated = self._apply_warped_live(
                [],
                warped_shadow,
                traces=traces,
                semantic_support=semantic_support,
                coupling=np.zeros(len(traces), dtype=np.float64),
                paths={},
                now_ms=self._clock.now_ms(),
                max_total=max_total,
            )
            self._store_resonance_audit(
                conversation_id,
                query_text,
                recall_state,
                (),
                tuple(item.trace.id for item in warped_activated),
                background=self._resonance_config.background_floor,
                null_mass=1.0,
                conservation_residual=0.0,
                emerged=bool(warped_activated),
                warped_shadow=warped_shadow,
            )
            return warped_activated
        source /= source.sum()
        links = self._repo.links_among([trace.id for trace in traces])
        rates, gains = modulated_rates(traces, links, recall_state)
        hop_budget = effective_hop_budget(self._resonance_config, recall_state)
        coupling, h1d_audit = transport_coupling(source, rates, hop_budget)
        paths = strongest_paths([trace.id for trace in traces], rates, source, hop_budget)
        now_ms = self._clock.now_ms()
        query_arc = self._query_arc(traces, recall_state)
        metric_tensor = state_metric_tensor(recall_state, self._resonance_config)
        raw_metrics = {}
        trace_indexes = {trace.id: index for index, trace in enumerate(traces)}
        for index, trace in enumerate(traces):
            freshness = _freshness(trace.created_at_ms, now_ms)
            age_ratio = (trace.created_at_ms - traces[0].created_at_ms) / max(
                1.0, float(traces[-1].created_at_ms - traces[0].created_at_ms)
            )
            candidate_arc = self._candidate_arc(traces, index)
            metrics = resonance_metrics(
                content_distance=1.0 - float(semantic_support[index]),
                affect_distance=math.hypot(
                    (trace.valence - recall_state.valence) / 2.0,
                    trace.arousal - recall_state.arousal,
                )
                / math.sqrt(2.0),
                relation_distance=abs(trace.importance - recall_state.connection_need),
                situation_distance=(age_ratio if recall_state.origin_intent else 1.0 - freshness),
                arc_distance=arc_shape_distance(candidate_arc, query_arc),
                coupling_mass=float(coupling[index]),
                damping=1.0 / (1.0 + trace.importance),
                config=self._resonance_config,
                metric_tensor=metric_tensor,
            )
            if (
                semantic_support[index] >= self._resonance_config.min_semantic_support
                or recall_state.origin_intent
            ):
                raw_metrics[trace.id] = metrics
        gated, background = gate_resonance(raw_metrics, self._resonance_config)
        selected_metrics = {
            trace_id: item
            for trace_id, item in gated.items()
            if item.background_ratio >= self._resonance_config.emergence_ratio
        }
        ranked = sorted(
            selected_metrics.items(),
            key=lambda item: (-item[1].amplitude, item[0]),
        )[: min(max_total, self._resonance_config.capacity)]
        by_id = {trace.id: trace for trace in traces}
        activated: list[ActivatedTrace] = []
        audits: list[ResonanceCandidateAudit] = []
        for trace_id, metrics in gated.items():
            audits.append(
                ResonanceCandidateAudit(
                    trace_id=trace_id,
                    content_distance=metrics.content_distance,
                    affect_distance=metrics.affect_distance,
                    relation_distance=metrics.relation_distance,
                    situation_distance=metrics.situation_distance,
                    arc_distance=metrics.arc_distance,
                    detuning=metrics.detuning,
                    coupling_mass=metrics.coupling_mass,
                    damping=metrics.damping,
                    amplitude=metrics.amplitude,
                    background_ratio=metrics.background_ratio,
                    path_trace_ids=paths.get(trace_id, ()),
                    emerged=trace_id in selected_metrics,
                )
            )
        for rank, (trace_id, metrics) in enumerate(ranked, 1):
            trace = by_id[trace_id]
            activated.append(
                ActivatedTrace(
                    trace=trace,
                    rank=rank,
                    score=metrics.amplitude,
                    semantic_similarity=float(semantic_support[trace_indexes[trace_id]]),
                    freshness=_freshness(trace.created_at_ms, now_ms),
                    importance_factor=trace.importance,
                    activation_kind="resonance",
                    coupling_mass=metrics.coupling_mass,
                    detuning=metrics.detuning,
                    damping=metrics.damping,
                    resonance_amplitude=metrics.amplitude,
                    resonance_ratio=metrics.background_ratio,
                    path_trace_ids=paths.get(trace_id, ()),
                    recall_reason=(
                        f"typed {hop_budget}-hop resonance; background ratio "
                        f"{metrics.background_ratio:.2f}"
                    ),
                )
            )
        activated = self._apply_warped_live(
            activated,
            warped_shadow,
            traces=traces,
            semantic_support=semantic_support,
            coupling=coupling,
            paths=paths,
            now_ms=now_ms,
            max_total=max_total,
        )
        selected_ids = tuple(item.trace.id for item in activated)
        null_mass = max(0.0, 1.0 - sum(item.coupling_mass for item in activated))
        self._store_resonance_audit(
            conversation_id,
            query_text,
            recall_state,
            tuple(audits),
            selected_ids,
            background=background,
            null_mass=null_mass,
            conservation_residual=h1d_audit.conservation_residual,
            emerged=bool(activated),
            hop_budget=hop_budget,
            gains=gains,
            warped_shadow=warped_shadow,
        )
        return activated

    def _run_warped_evaluation(
        self,
        traces: list[Trace],
        embeddings: list[np.ndarray],
        query_vector: np.ndarray,
        semantic_support: np.ndarray,
        state: RecallState,
    ) -> WarpedResonanceShadowAudit | None:
        if (
            self._warped_mode == WarpedResonanceMode.DISABLED
            or self._warped_retriever is None
            or len(traces) < 2
        ):
            return None
        indexes = _bounded_archive_indexes(len(traces), self._warped_max_nodes)
        selected_traces = [traces[index] for index in indexes]
        try:
            request = WarpedRetrievalRequest(
                trace_ids=tuple(trace.id for trace in selected_traces),
                embeddings=np.asarray([embeddings[index] for index in indexes], dtype=np.float64),
                affect=np.asarray(
                    [
                        (
                            traces[index].valence,
                            traces[index].arousal,
                            traces[index].tension,
                        )
                        for index in indexes
                    ],
                    dtype=np.float64,
                ),
                spread=np.asarray(
                    [max(0.05, 1.0 - traces[index].importance) for index in indexes],
                    dtype=np.float64,
                ),
                query_embedding=np.asarray(query_vector, dtype=np.float64),
                query_affect=(state.valence, state.arousal, state.tension),
                semantic_similarity=np.asarray(
                    [semantic_support[index] for index in indexes],
                    dtype=np.float64,
                ),
            )
            result = self._warped_retriever.scan(request)
            return WarpedResonanceShadowAudit(
                model_id=result.model_id,
                mode=("live" if self._warped_mode == WarpedResonanceMode.LIVE else "shadow"),
                evaluated_trace_count=len(selected_traces),
                content_information=result.content_information,
                affect_mix=result.affect_mix,
                temperature=result.temperature,
                selected_trace_ids=result.selected_trace_ids,
                detuning_by_trace_id=dict(zip(request.trace_ids, result.detuning, strict=True)),
                probability_by_trace_id=dict(
                    zip(request.trace_ids, result.probability, strict=True)
                ),
                surfaced=result.surfaced,
                reason=result.reason,
                parameters=dict(result.parameters),
            )
        except Exception as exc:
            return WarpedResonanceShadowAudit(
                model_id=WARPED_MODEL_ID,
                mode=("live" if self._warped_mode == WarpedResonanceMode.LIVE else "shadow"),
                evaluated_trace_count=len(selected_traces),
                content_information=0.0,
                affect_mix=0.0,
                temperature=self._warped_config.temperature,
                selected_trace_ids=(),
                detuning_by_trace_id={},
                probability_by_trace_id={},
                surfaced=False,
                reason=f"{type(exc).__name__}: {exc}",
                parameters={},
            )

    def _apply_warped_live(
        self,
        baseline: list[ActivatedTrace],
        evaluation: WarpedResonanceShadowAudit | None,
        *,
        traces: list[Trace],
        semantic_support: np.ndarray,
        coupling: np.ndarray,
        paths: dict[str, tuple[str, ...]],
        now_ms: int,
        max_total: int,
    ) -> list[ActivatedTrace]:
        if (
            self._warped_mode != WarpedResonanceMode.LIVE
            or evaluation is None
            or not evaluation.surfaced
        ):
            return baseline
        trace_indexes = {trace.id: index for index, trace in enumerate(traces)}
        probabilities = [
            value for value in evaluation.probability_by_trace_id.values() if value > 0.0
        ]
        background = max(1e-12, float(np.median(probabilities)) if probabilities else 0.0)
        activated: list[ActivatedTrace] = []
        for trace_id in evaluation.selected_trace_ids[:max_total]:
            index = trace_indexes.get(trace_id)
            if index is None:
                continue
            trace = traces[index]
            probability = evaluation.probability_by_trace_id[trace_id]
            detuning = evaluation.detuning_by_trace_id[trace_id]
            activated.append(
                ActivatedTrace(
                    trace=trace,
                    rank=len(activated) + 1,
                    score=probability,
                    semantic_similarity=float(semantic_support[index]),
                    freshness=_freshness(trace.created_at_ms, now_ms),
                    importance_factor=trace.importance,
                    activation_kind="resonance",
                    coupling_mass=float(coupling[index]),
                    detuning=detuning,
                    damping=1.0 / (1.0 + trace.importance),
                    resonance_amplitude=probability,
                    resonance_ratio=probability / background,
                    path_trace_ids=paths.get(trace_id, ()),
                    recall_reason=(
                        "adaptive warped free-energy live; content information "
                        f"{evaluation.content_information:.3f}"
                    ),
                )
            )
        return activated or baseline

    def _add_typed_links(
        self,
        trace: Trace,
        previous_traces: list[Trace],
        user_event: Event,
        agent_event: Event,
        now_ms: int,
    ) -> None:
        """Create bounded, deterministic relation edges for the new episode."""
        recent = previous_traces[-8:]
        for previous in recent:
            gap = max(0, trace.created_at_ms - previous.created_at_ms)
            temporal_weight = _clip(math.exp(-gap / (7.0 * _DAY_MS)))
            if temporal_weight >= 0.05:
                self._repo.add_link(
                    TraceLink(
                        source_trace_id=trace.id,
                        target_trace_id=previous.id,
                        link_type=TraceLinkType.TEMPORAL,
                        weight=temporal_weight,
                        created_at_ms=now_ms,
                    )
                )
                self._repo.add_link(
                    TraceLink(
                        source_trace_id=previous.id,
                        target_trace_id=trace.id,
                        link_type=TraceLinkType.TEMPORAL,
                        weight=_clip(temporal_weight * 0.9),
                        created_at_ms=now_ms,
                    )
                )

            affective_distance = math.hypot(
                (trace.valence - previous.valence) / 2.0,
                trace.arousal - previous.arousal,
            ) / math.sqrt(2.0)
            affective_weight = _clip(1.0 - affective_distance)
            if affective_weight >= 0.35:
                self._repo.add_link(
                    TraceLink(
                        source_trace_id=trace.id,
                        target_trace_id=previous.id,
                        link_type=TraceLinkType.AFFECTIVE,
                        weight=affective_weight,
                        created_at_ms=now_ms,
                    )
                )

            shared_entities = _entity_keys(trace.content) & _entity_keys(previous.content)
            if shared_entities:
                entity_weight = _clip(0.45 + 0.10 * min(5, len(shared_entities)))
                self._repo.add_link(
                    TraceLink(
                        source_trace_id=trace.id,
                        target_trace_id=previous.id,
                        link_type=TraceLinkType.ENTITY,
                        weight=entity_weight,
                        created_at_ms=now_ms,
                    )
                )

        # Event ancestry is the strongest available evidence-chain relation.
        parent_event_ids = tuple(
            event_id
            for event_id in (user_event.parent_event_id, agent_event.parent_event_id)
            if event_id
        )
        for parent_event_id in parent_event_ids:
            parent_trace = self._repo.find_by_event(parent_event_id)
            if parent_trace is None or parent_trace.id == trace.id:
                continue
            self._repo.add_link(
                TraceLink(
                    source_trace_id=parent_trace.id,
                    target_trace_id=trace.id,
                    link_type=TraceLinkType.CAUSAL,
                    weight=0.85,
                    created_at_ms=now_ms,
                )
            )

    def _query_arc(
        self,
        traces: list[Trace],
        state: RecallState,
    ) -> tuple[tuple[float, float], ...]:
        """Represent the current affect as a short trajectory, not one endpoint."""
        if not traces:
            return ((0.0, 0.2), (state.valence, state.arousal))
        baseline = traces[0]
        return (
            (baseline.valence, baseline.arousal),
            (
                0.5 * baseline.valence + 0.5 * state.valence,
                0.5 * baseline.arousal + 0.5 * state.arousal,
            ),
            (state.valence, state.arousal),
        )

    def _candidate_arc(
        self,
        traces: list[Trace],
        index: int,
    ) -> tuple[tuple[float, float], ...]:
        """Return a local chronological affect arc around one candidate."""
        if not traces:
            return ()
        left = max(0, index - 2)
        right = min(len(traces), index + 3)
        points = [(item.valence, item.arousal) for item in traces[left:right]]
        if len(points) == 1:
            point = points[0]
            points = [(point[0], point[1]), (point[0], point[1]), (point[0], point[1])]
        elif len(points) == 2:
            points.insert(0, points[0])
        return tuple(points)

    def _store_resonance_audit(
        self,
        conversation_id: str,
        query_text: str,
        state: RecallState,
        candidates: tuple[ResonanceCandidateAudit, ...],
        selected_trace_ids: tuple[str, ...],
        *,
        background: float,
        null_mass: float,
        conservation_residual: float,
        emerged: bool,
        hop_budget: int | None = None,
        gains: dict[TraceLinkType, float] | None = None,
        warped_shadow: WarpedResonanceShadowAudit | None = None,
    ) -> None:
        audit = ResonanceRecallAudit(
            id=self._ids.new(),
            conversation_id=conversation_id,
            query_digest=hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
            situation_mode=state.situation_mode,
            hop_budget=hop_budget or effective_hop_budget(self._resonance_config, state),
            edge_gains=gains or state_modulated_gains(state),
            candidates=candidates,
            selected_trace_ids=selected_trace_ids,
            background_median=max(0.0, background),
            emergence_ratio=self._resonance_config.emergence_ratio,
            null_mass=_clip(null_mass),
            conservation_residual=conservation_residual,
            emerged=emerged,
            created_at_ms=self._clock.now_ms(),
            warped_shadow=warped_shadow,
        )
        self._resonance_audits[conversation_id] = audit
        try:
            self._repo.insert_resonance_audit(audit)
        except Exception:
            # Resonance is observable; audit storage failure must not block recall.
            return

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
        resonance_audit = self._resonance_audits.get(conversation_id)
        if resonance_audit is None:
            try:
                resonance_audit = self._repo.latest_resonance_audit(conversation_id)
            except Exception:
                resonance_audit = None
        activation_model = "legacy-ssa-a0"
        serving_model_id = "legacy-ssa-a0"
        if any(item.activation_kind == "resonance" for item in activations.values()):
            activation_model = "hdsc-h1d-resonance-v1"
            serving_model_id = activation_model
        if (
            resonance_audit is not None
            and resonance_audit.warped_shadow is not None
            and resonance_audit.warped_shadow.mode == "live"
            and resonance_audit.warped_shadow.surfaced
        ):
            activation_model = resonance_audit.warped_shadow.model_id
            serving_model_id = activation_model
        return TraceSpaceSnapshot(
            nodes=nodes,
            links=self._repo.links_among(trace_ids),
            total_traces=archive_count,
            latest_query=latest_query,
            projection=projection,
            activation_model=activation_model,
            serving_model_id=serving_model_id,
            legacy_activated_count=len(activations),
            shadow_model_id=(H2_MODEL_ID if self._h2_enabled else None),
            shadow_status=shadow_status,
            shadow_affects_prompt=False,
            warped_mode=self._warped_mode.value,
            warped_affects_prompt=(self._warped_mode == WarpedResonanceMode.LIVE),
            stability_audit=stability_audit,
            resonance_audit=resonance_audit,
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


def _bounded_archive_indexes(count: int, limit: int) -> tuple[int, ...]:
    """Sample the full chronology deterministically for bounded shadow work."""
    if count <= 0 or limit <= 0:
        return ()
    if count <= limit:
        return tuple(range(count))
    positions = np.linspace(0, count - 1, num=limit)
    return tuple(round(position) for position in positions)


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


_ENTITY_TOKEN_RE = re.compile(r"[a-z0-9_\u3400-\u9fff]+", re.IGNORECASE)
_ENTITY_STOPWORDS = frozenset(
    {"user", "agent", "assistant", "the", "and", "的", "了", "是", "我", "你", "我们"}
)


def _entity_keys(content: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in _ENTITY_TOKEN_RE.findall(content)
        if token.casefold() not in _ENTITY_STOPWORDS and len(token.strip()) >= 2
    }
    if "user:" in content.casefold() and "agent:" in content.casefold():
        tokens.add("relationship:dual-actor")
    return tokens


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


__all__ = ["TraceSpaceService"]
