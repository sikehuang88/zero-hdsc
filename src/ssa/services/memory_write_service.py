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
from ssa.config import RetrievalConfig, ThinkingMode
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


def _coerce_memory_type(value: object) -> MemoryType:
    normalized = str(value or MemoryType.SEMANTIC.value).strip().casefold().replace("-", "_")
    try:
        return MemoryType(normalized)
    except ValueError:
        if "relationship" in normalized:
            return MemoryType.RELATIONSHIP
        if "promise" in normalized or "commit" in normalized:
            return MemoryType.PROMISE
        if "unresolved" in normalized or "conflict" in normalized:
            return MemoryType.UNRESOLVED
        if "reflection" in normalized:
            return MemoryType.REFLECTION
        if normalized.startswith("self") or "identity" in normalized:
            return MemoryType.SELF
        if "episode" in normalized or "event" in normalized:
            return MemoryType.EPISODIC
        return MemoryType.SEMANTIC


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
        default_model: str = "deepseek/deepseek-v4-flash",
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
        self._default_model = default_model

    async def extract_candidates(
        self,
        user_event: Event,
        agent_event: Event | None = None,
    ) -> list[MemoryCandidate]:
        """Use the LLM to extract memory candidates from this turn's events."""
        events_desc = f"User event id={user_event.id}: {user_event.content}"
        if agent_event is not None:
            events_desc += f"\nAgent event id={agent_event.id}: {agent_event.content}"

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
                            "memory_type",
                            "content",
                            "evidence_event_ids",
                            "confidence",
                            "importance",
                            "valence",
                            "arousal",
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
                    "remembering long-term. Every candidate must cite one or more "
                    "supplied event IDs exactly. Output compact JSON with a 'candidates' "
                    "array. Every item must contain memory_type, content, "
                    "evidence_event_ids, confidence, importance, valence, and arousal."
                ),
                ChatMessage.user(events_desc),
            ],
            model=self._default_model,
            temperature=0.3,
            max_tokens=1024,
            thinking=ThinkingMode.DISABLED,
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
                raw_evidence_ids = raw.get("evidence_event_ids", raw.get("event_ids", []))
                if not isinstance(raw_evidence_ids, list) or any(
                    not isinstance(eid, str) for eid in raw_evidence_ids
                ):
                    logger.warning(
                        "Skipping memory candidate with malformed evidence IDs: %s",
                        raw_evidence_ids,
                    )
                    continue
                allowed_evidence = {user_event.id}
                if agent_event is not None:
                    allowed_evidence.add(agent_event.id)
                if not raw_evidence_ids:
                    evidence_ids = [user_event.id]
                    if agent_event is not None:
                        evidence_ids.append(agent_event.id)
                    logger.info(
                        "Memory candidate omitted evidence IDs; using the supplied turn as "
                        "model-inference evidence"
                    )
                else:
                    evidence_ids = raw_evidence_ids
                if any(eid not in allowed_evidence for eid in evidence_ids):
                    logger.warning(
                        "Skipping memory candidate with invalid evidence IDs: %s",
                        evidence_ids,
                    )
                    continue
                valid_evidence = list(dict.fromkeys(evidence_ids))

                # Derive source_kind from context (pipeline §14.3 step 4).
                evidence_set = set(valid_evidence)
                if evidence_set == {user_event.id}:
                    source = SourceKind.USER_OBSERVED
                elif agent_event is not None and evidence_set == {agent_event.id}:
                    source = SourceKind.AGENT_OUTPUT
                else:
                    source = SourceKind.MODEL_INFERENCE

                # MODEL_INFERENCE confidence capped at 0.6 (pipeline §14.4).
                raw_conf = float(raw.get("confidence", 0.5))
                if source == SourceKind.MODEL_INFERENCE:
                    raw_conf = min(raw_conf, 0.6)

                candidate = MemoryCandidate(
                    memory_type=_coerce_memory_type(raw.get("memory_type")),
                    content=raw.get("content", raw.get("fact", "")),
                    source_kind=source,
                    evidence_event_ids=valid_evidence,
                    confidence=raw_conf,
                    importance=float(raw.get("importance", 0.6)),
                    valence=float(raw.get("valence", 0.0)),
                    arousal=float(raw.get("arousal", 0.3)),
                    contradiction_query=raw.get("contradiction_query"),
                    derived_by_model=response.model,
                )
                candidates.append(candidate)
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("Skipping invalid memory candidate: %s", exc)
                continue

        return candidates

    def write_candidates(self, candidates: list[MemoryCandidate]) -> MemoryWriteResult:
        """Validate, dedup, and persist memory candidates.

        Pipeline §14.3 steps 4-13.
        """
        result = MemoryWriteResult()
        now_ms = self._clock.now_ms()

        for candidate in candidates:
            candidate = self._validated_candidate(candidate)
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
                derived_by_model=candidate.derived_by_model,
                prompt_version=self._prompt_version,
                status="active",
                access_count=0,
                last_accessed_at_ms=None,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )

            links = [
                (mem.id, "semantic", float(sim), now_ms)
                for mem, sim in similar[:5]
                if mem.id != memory_id
            ]
            self._repo.insert_bundle(
                memory,
                vec.as_bytes(),
                candidate.evidence_event_ids,
                links,
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

    def _validated_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        if not candidate.evidence_event_ids:
            raise ValueError("memory candidate requires at least one evidence event")
        events = [self._event_lookup(event_id) for event_id in candidate.evidence_event_ids]
        if any(event is None for event in events):
            raise ValueError("memory candidate references an unknown evidence event")
        present = [event for event in events if event is not None]
        if candidate.source_kind == SourceKind.USER_OBSERVED and any(
            event.actor.value != "user" for event in present
        ):
            raise ValueError("user-observed memory evidence must come from user events")
        if candidate.source_kind == SourceKind.AGENT_OUTPUT and any(
            event.actor.value != "agent" for event in present
        ):
            raise ValueError("agent-output memory evidence must come from agent events")
        if candidate.source_kind == SourceKind.WORLD_OBSERVED and any(
            event.source_kind != SourceKind.WORLD_OBSERVED for event in present
        ):
            raise ValueError("world-observed memory evidence must retain world provenance")
        if candidate.source_kind == SourceKind.MODEL_INFERENCE and candidate.confidence > 0.6:
            return candidate.model_copy(update={"confidence": 0.6})
        return candidate


__all__ = ["MemoryWriteResult", "MemoryWriteService"]
