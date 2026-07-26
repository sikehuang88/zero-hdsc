"""Memory write service — extracts, validates, deduplicates and persists memories.

Pipeline §14 (M05).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ssa.adapters.embedding import EmbeddingService
from ssa.adapters.llm import ChatMessage, LLMAdapter, LLMRequest
from ssa.clock import Clock
from ssa.config import RetrievalConfig
from ssa.domain.enums import MemoryType, SourceKind
from ssa.domain.events import Event
from ssa.domain.memories import Memory, MemoryCandidate
from ssa.ids import IdGenerator
from ssa.storage.memory_repository import SqliteMemoryRepository

logger = logging.getLogger(__name__)


@dataclass
class MemoryWriteResult:
    """Summary of a memory write operation."""

    created: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicted: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.created) + len(self.merged) + len(self.skipped) + len(self.conflicted)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class MemoryWriteService:
    """Orchestrates the memory write pipeline (M05)."""

    def __init__(
        self,
        llm: LLMAdapter,
        embedding: EmbeddingService,
        memory_repo: SqliteMemoryRepository,
        event_lookup: Callable[[str], Event | None],
        ids: IdGenerator,
        clock: Clock,
        retrieval_config: RetrievalConfig,
        *,
        prompt_version: str = "memory_extract_v1",
        embedding_model_name: str = "bge-small-zh-v1.5",
        embedding_dim: int = 512,
    ) -> None:
        self._llm = llm
        self._embedding = embedding
        self._repo = memory_repo
        self._event_lookup = event_lookup
        self._ids = ids
        self._clock = clock
        self._config = retrieval_config
        self._prompt_version = prompt_version
        self._emb_model = embedding_model_name
        self._emb_dim = embedding_dim

    async def extract_candidates(
        self,
        user_event: Event,
        agent_event: Event | None = None,
    ) -> list[MemoryCandidate]:
        """Use the LLM to extract memory candidates from this turn's events."""
        events_desc = f"User said: {user_event.content}"
        if agent_event is not None:
            events_desc += f"\nAgent responded: {agent_event.content}"

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_type": {"type": "string"},
                            "content": {"type": "string"},
                            "evidence_event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number"},
                            "importance": {"type": "number"},
                            "valence": {"type": "number"},
                            "arousal": {"type": "number"},
                            "contradiction_query": {"type": ["string", "null"]},
                        },
                        "required": [
                            "memory_type", "content", "evidence_event_ids",
                            "confidence", "importance", "valence", "arousal",
                        ],
                    },
                }
            },
            "required": ["candidates"],
        }

        request = LLMRequest(
            purpose="memory_extract",
            messages=[
                ChatMessage.system(
                    "You are a memory extraction system. Extract memorable facts "
                    "from the conversation below. Only extract things worth "
                    "remembering long-term. Output JSON with a 'candidates' array."
                ),
                ChatMessage.user(events_desc),
            ],
            model="deepseek/deepseek-chat",
            temperature=0.3,
            max_tokens=1024,
            prompt_version=self._prompt_version,
            json_schema=schema,
        )

        response = await self._llm.complete(request)
        if response.parsed is None:
            return []

        raw_candidates = response.parsed.get("candidates", [])
        candidates: list[MemoryCandidate] = []

        for raw in raw_candidates:
            try:
                evidence_ids = raw.get("evidence_event_ids", [])
                valid_evidence = [
                    eid for eid in evidence_ids if self._event_lookup(eid) is not None
                ]
                if not valid_evidence:
                    valid_evidence = [user_event.id]

                # Derive source_kind from context (pipeline §14.3 step 4).
                if user_event.id in valid_evidence:
                    source = SourceKind.USER_OBSERVED
                elif agent_event and agent_event.id in valid_evidence:
                    source = SourceKind.AGENT_OUTPUT
                else:
                    source = SourceKind.MODEL_INFERENCE

                # MODEL_INFERENCE confidence capped at 0.6 (pipeline §14.4).
                raw_conf = float(raw.get("confidence", 0.5))
                if source == SourceKind.MODEL_INFERENCE:
                    raw_conf = min(raw_conf, 0.6)

                candidate = MemoryCandidate(
                    memory_type=MemoryType(raw["memory_type"]),
                    content=raw["content"],
                    source_kind=source,
                    evidence_event_ids=valid_evidence,
                    confidence=raw_conf,
                    importance=float(raw.get("importance", 0.5)),
                    valence=float(raw.get("valence", 0.0)),
                    arousal=float(raw.get("arousal", 0.3)),
                    contradiction_query=raw.get("contradiction_query"),
                )
                candidates.append(candidate)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping invalid memory candidate: %s", exc)
                continue

        return candidates

    def write_candidates(
        self, candidates: list[MemoryCandidate]
    ) -> MemoryWriteResult:
        """Validate, dedup, and persist memory candidates.

        Pipeline §14.3 steps 4-13.
        """
        result = MemoryWriteResult()
        now_ms = self._clock.now_ms()

        for candidate in candidates:
            chash = _content_hash(candidate.content)

            # Exact content hash dedup.
            existing = self._repo.find_by_content_hash(chash)
            if existing is not None:
                for eid in candidate.evidence_event_ids:
                    self._repo.add_evidence(existing.id, eid, "supports")
                result.merged.append(existing.id)
                continue

            # Compute embedding.
            vec = self._embedding.embed_one(candidate.content)

            # Semantic dedup: find similar memories.
            similar = self._repo.find_similar(
                vec.as_bytes(),
                memory_type=candidate.memory_type,
                limit=10,
            )

            # High-similarity merge.
            merged_into: str | None = None
            for mem, sim in similar:
                if sim >= self._config.dedup_similarity_threshold:
                    for eid in candidate.evidence_event_ids:
                        self._repo.add_evidence(mem.id, eid, "supports")
                    merged_into = mem.id
                    break

            if merged_into is not None:
                result.merged.append(merged_into)
                continue

            # Conflict detection.
            if candidate.contradiction_query:
                conflict_vec = self._embedding.embed_one(candidate.contradiction_query)
                conflict_similar = self._repo.find_similar(
                    conflict_vec.as_bytes(),
                    memory_type=candidate.memory_type,
                    limit=5,
                )
                for mem, sim in conflict_similar:
                    if sim >= 0.7:
                        self._repo.add_evidence(
                            mem.id, candidate.evidence_event_ids[0], "contradicts"
                        )
                        self._repo.update_status(mem.id, "conflicted", now_ms)
                        result.conflicted.append(mem.id)
                        break

            # Create new memory.
            memory_id = self._ids.new()
            memory = Memory(
                id=memory_id,
                memory_type=candidate.memory_type,
                source_kind=candidate.source_kind,
                content=candidate.content,
                summary=None,
                confidence=candidate.confidence,
                importance=candidate.importance,
                valence=candidate.valence,
                arousal=candidate.arousal,
                embedding_model=self._emb_model,
                embedding_dim=self._emb_dim,
                content_hash=chash,
                derived_by_model=None,
                prompt_version=self._prompt_version,
                status="active",
                access_count=0,
                last_accessed_at_ms=None,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )

            self._repo.insert(memory, vec.as_bytes())

            # Add evidence links.
            for eid in candidate.evidence_event_ids:
                self._repo.add_evidence(memory_id, eid, "supports")

            # Generate one-hop links to similar memories.
            for mem, sim in similar[:5]:
                if mem.id != memory_id:
                    self._repo.add_link(
                        memory_id, mem.id, "semantic", float(sim), now_ms
                    )

            result.created.append(memory_id)

        logger.info(
            "Memory write: %d created, %d merged, %d skipped, %d conflicted",
            len(result.created),
            len(result.merged),
            len(result.skipped),
            len(result.conflicted),
        )
        return result


__all__ = ["MemoryWriteResult", "MemoryWriteService"]
