"""Query-time ENGRAM graph projection over the existing trace repository."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ssa.clock import Clock
from ssa.config import EngramConfig
from ssa.domain.engram import EngramQueryResult, EngramRelation, EngramTransportAudit
from ssa.hdsc.engram_transport import EngramTransportConfig, propagate_engram
from ssa.storage.engram_repository import EngramRepository
from ssa.storage.trace_repository import SqliteTraceRepository


class EngramActivationService:
    """Build the typed graph from immutable traces, then run a bounded query."""

    def __init__(
        self,
        engram_repository: EngramRepository,
        trace_repository: SqliteTraceRepository,
        clock: Clock,
        *,
        transport: EngramTransportConfig | None = None,
        graph_limit: int = 512,
    ) -> None:
        if graph_limit < 1:
            raise ValueError("engram graph_limit must be positive")
        self._engram = engram_repository
        self._traces = trace_repository
        self._clock = clock
        self._transport = transport or EngramTransportConfig()
        self._graph_limit = graph_limit

    @classmethod
    def from_config(
        cls,
        engram_repository: EngramRepository,
        trace_repository: SqliteTraceRepository,
        clock: Clock,
        config: EngramConfig,
    ) -> EngramActivationService:
        return cls(
            engram_repository,
            trace_repository,
            clock,
            transport=EngramTransportConfig(
                mode=config.mode.value,
                max_hops=config.max_hops,
                restart_probability=config.restart_probability,
                epsilon=config.epsilon,
                max_active=config.max_active,
                max_results=config.max_results,
            ),
            graph_limit=config.graph_limit,
        )

    def activate(
        self,
        conversation_id: str,
        seed_masses: Mapping[str, float],
        *,
        relation_gates: Mapping[EngramRelation, float] | None = None,
        sync: bool = True,
    ) -> EngramQueryResult:
        traces = self._traces.recent(conversation_id, limit=self._graph_limit)
        trace_by_id = {trace.id: trace for trace in traces}
        for seed_id in seed_masses:
            if seed_id in trace_by_id:
                continue
            seed_trace = self._traces.get(seed_id)
            if seed_trace is not None and seed_trace.conversation_id == conversation_id:
                trace_by_id[seed_trace.id] = seed_trace
        valid_seed_masses = {
            node_id: mass for node_id, mass in seed_masses.items() if node_id in trace_by_id
        }
        traces = sorted(trace_by_id.values(), key=lambda trace: (trace.created_at_ms, trace.id))
        node_ids = [trace.id for trace in traces]
        if not valid_seed_masses:
            return EngramQueryResult(
                activations=[],
                audit=EngramTransportAudit(
                    mode=self._transport.mode,
                    node_count=len(node_ids),
                    edge_count=0,
                    seed_mass=0.0,
                    propagated_mass=0.0,
                    null_mass=0.0,
                    conservation_residual=0.0,
                    max_hops=self._transport.max_hops,
                    path_grounded_count=0,
                ),
            )
        if sync:
            links = self._traces.links_among(node_ids)
            self._engram.sync_trace_subgraph(traces, links)
        edges = self._engram.active_edges(
            conversation_id,
            node_ids,
            now_ms=self._clock.now_ms(),
        )
        return propagate_engram(
            node_ids,
            edges,
            valid_seed_masses,
            relation_gates=dict(relation_gates or {}),
            config=self._transport,
        )

    def activate_from_scores(
        self,
        conversation_id: str,
        seeds: Sequence[tuple[str, float]],
        *,
        relation_gates: Mapping[EngramRelation, float] | None = None,
    ) -> EngramQueryResult:
        """Convenience adapter for vector-retrieval scores used as π0."""
        return self.activate(
            conversation_id,
            dict(seeds),
            relation_gates=relation_gates,
        )


__all__ = ["EngramActivationService"]
