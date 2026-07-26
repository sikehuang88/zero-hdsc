"""Memory retrieval service — finds the memories that matter for this turn.

Pipeline §15 (M06).
"""

from __future__ import annotations

import logging
import math

from ssa.adapters.embedding import EmbeddingService
from ssa.clock import Clock
from ssa.config import RetrievalConfig
from ssa.domain.enums import MemoryType, SourceKind
from ssa.domain.memories import Memory, RetrievedMemory
from ssa.storage.memory_repository import SqliteMemoryRepository

logger = logging.getLogger(__name__)


class MemoryRetrievalService:
    """Stable, explainable memory retrieval (M06).

    Pipeline §15.2 defines the retrieval contract.
    """

    def __init__(
        self,
        embedding: EmbeddingService,
        memory_repo: SqliteMemoryRepository,
        clock: Clock,
        config: RetrievalConfig,
    ) -> None:
        self._embedding = embedding
        self._repo = memory_repo
        self._clock = clock
        self._config = config

    def prefetch_for_appraisal(
        self,
        query_text: str,
        recent_limit: int = 3,
        semantic_limit: int = 2,
    ) -> list[RetrievedMemory]:
        """Lightweight retrieval for appraisal context (pipeline §15.2).

        No radiation, no access count update.
        """
        vec = self._embedding.embed_one(query_text)
        similar = self._repo.find_similar(
            vec.as_bytes(), limit=semantic_limit * 3
        )
        results: list[RetrievedMemory] = []
        for mem, sim in similar[:semantic_limit]:
            evidence = self._repo.get_evidence(mem.id)
            results.append(
                RetrievedMemory(
                    memory_id=mem.id,
                    content=mem.content,
                    source_kind=mem.source_kind,
                    evidence_event_ids=[eid for eid, _ in evidence],
                    score=sim,
                    score_components={"semantic": sim},
                    retrieval_reason="prefetch_semantic",
                )
            )
        return results

    def retrieve(
        self,
        query_text: str,
        *,
        allowed_source_kinds: set[SourceKind] | None = None,
        token_budget: int = 2000,
        max_results: int = 8,
    ) -> list[RetrievedMemory]:
        """Full retrieval with reranking, radiation, and MMR (pipeline §15.3)."""
        now_ms = self._clock.now_ms()
        vec = self._embedding.embed_one(query_text)

        # Step 2: recall semantic candidates.
        similar = self._repo.find_similar(
            vec.as_bytes(), limit=self._config.candidates
        )

        # Step 3: filter.
        if allowed_source_kinds is not None:
            similar = [(m, s) for m, s in similar if m.source_kind in allowed_source_kinds]

        if not similar:
            return []

        # Steps 4-5: compute score components.
        scored: list[tuple[Memory, float, dict[str, float]]] = []
        for mem, semantic_sim in similar:
            components = self._score_components(mem, semantic_sim, now_ms)
            total = sum(components.values())
            scored.append((mem, total, components))

        # Step 6: top-12.
        scored.sort(key=lambda x: x[1], reverse=True)
        main = scored[:12]

        # Steps 7-9: link expansion + dedup.
        expanded: list[tuple[Memory, float, dict[str, float], str]] = []
        seen: set[str] = set()
        for mem, score, comp in main:
            if mem.id not in seen:
                expanded.append((mem, score, comp, "main"))
                seen.add(mem.id)
            for target_id, link_type, weight in self._repo.get_links(mem.id):
                if target_id in seen:
                    continue
                target = self._repo.get(target_id)
                if target is None or target.status != "active":
                    continue
                ns = score * weight * self._config.radiation_decay
                expanded.append((target, ns, {**comp, "link": weight}, f"link:{link_type}"))
                seen.add(target_id)

        # Step 10: MMR.
        mmr_results = self._mmr(expanded)

        # Step 12: truncate.
        final = mmr_results[:max_results]

        # Step 13: update access.
        for r in final:
            self._repo.increment_access(r.memory_id, now_ms)

        logger.info(
            "Retrieval: %d → %d main → %d expanded → %d final",
            len(similar), len(main), len(expanded), len(final),
        )
        return final

    def _score_components(
        self, mem: Memory, semantic_sim: float, now_ms: int
    ) -> dict[str, float]:
        """Pipeline §15.4 scoring."""
        age_days = max(0, now_ms - mem.created_at_ms) / (1000 * 60 * 60 * 24)
        recency = math.exp(-age_days / 30.0)
        relationship = 1.0 if mem.memory_type == MemoryType.RELATIONSHIP else 0.3
        unresolved = 1.0 if mem.memory_type == MemoryType.UNRESOLVED else 0.2
        trust_map = {
            SourceKind.USER_OBSERVED: 1.0, SourceKind.AGENT_OUTPUT: 0.8,
            SourceKind.SYSTEM_DERIVED: 0.6, SourceKind.WORLD_OBSERVED: 0.7,
            SourceKind.MODEL_INFERENCE: 0.4,
        }
        source_trust = trust_map.get(mem.source_kind, 0.5)
        c = self._config
        return {
            "semantic": semantic_sim * c.semantic_weight,
            "recency": recency * c.recency_weight,
            "importance": mem.importance * c.importance_weight,
            "relationship": relationship * c.relationship_weight,
            "unresolved": unresolved * c.unresolved_weight,
            "source_trust": source_trust * c.source_trust_weight,
        }

    def _mmr(
        self,
        candidates: list[tuple[Memory, float, dict[str, float], str]],
        lambda_param: float = 0.7,
    ) -> list[RetrievedMemory]:
        """Maximal Marginal Relevance."""
        if not candidates:
            return []
        contents = [c[0].content for c in candidates]
        embeddings = self._embedding.embed_many(contents)
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < 8:
            best_idx, best_score = -1, -1.0
            for idx in remaining:
                relevance = candidates[idx][1]
                max_sim = 0.0
                for sel_idx in selected:
                    sim = self._cosine(embeddings[idx].values, embeddings[sel_idx].values)
                    max_sim = max(max_sim, sim)
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        results: list[RetrievedMemory] = []
        for idx in selected:
            mem, score, comp, reason = candidates[idx]
            evidence = self._repo.get_evidence(mem.id)
            results.append(RetrievedMemory(
                memory_id=mem.id, content=mem.content,
                source_kind=mem.source_kind,
                evidence_event_ids=[eid for eid, _ in evidence],
                score=score, score_components=comp, retrieval_reason=reason,
            ))
        return results

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


__all__ = ["MemoryRetrievalService"]
