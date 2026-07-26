"""Memory domain models — Memory, MemoryCandidate, evidence and links.

Pipeline §14.2 / §15.2:
- `MemoryCandidate`: LLM-extracted candidate before persistence.
- `Memory`: persisted, searchable, provenance-tagged memory.
- `RetrievedMemory`: a memory returned from retrieval with score breakdown.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ssa.domain.enums import MemoryType, SourceKind


class MemoryCandidate(BaseModel):
    """LLM-extracted memory candidate (pipeline §14.2).

    The `source_kind` is derived by code from execution context, NOT from
    the model's self-report (pipeline §14.3 step 4).
    """

    memory_type: MemoryType
    content: str
    source_kind: SourceKind
    evidence_event_ids: list[str]
    confidence: float
    importance: float
    valence: float
    arousal: float
    contradiction_query: str | None = None

    @field_validator("confidence", "importance")
    @classmethod
    def _check_0_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {v}")
        return v

    @field_validator("valence")
    @classmethod
    def _check_neg1_1(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"valence must be in [-1, 1], got {v}")
        return v

    @field_validator("arousal")
    @classmethod
    def _check_arousal(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"arousal must be in [0, 1], got {v}")
        return v

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("memory content must not be empty")
        return v


class Memory(BaseModel):
    """A persisted memory (pipeline §8.2 `memories` table)."""

    id: str
    memory_type: MemoryType
    source_kind: SourceKind
    content: str
    summary: str | None = None
    confidence: float
    importance: float
    valence: float
    arousal: float
    embedding_model: str
    embedding_dim: int
    content_hash: str
    derived_by_model: str | None = None
    prompt_version: str | None = None
    status: str = "active"  # active | merged | conflicted | archived
    access_count: int = 0
    last_accessed_at_ms: int | None = None
    created_at_ms: int
    updated_at_ms: int

    @field_validator("confidence", "importance")
    @classmethod
    def _check_0_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {v}")
        return v

    @field_validator("valence")
    @classmethod
    def _check_neg1_1(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"valence must be in [-1, 1], got {v}")
        return v

    @field_validator("arousal")
    @classmethod
    def _check_arousal(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"arousal must be in [0, 1], got {v}")
        return v

    def to_export_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["memory_type"] = self.memory_type.value
        d["source_kind"] = self.source_kind.value
        return d


class RetrievedMemory(BaseModel):
    """A memory returned from retrieval with score breakdown (pipeline §15.2)."""

    memory_id: str
    content: str
    source_kind: SourceKind
    evidence_event_ids: list[str] = Field(default_factory=list)
    score: float
    score_components: dict[str, float] = Field(default_factory=dict)
    retrieval_reason: str = ""


__all__ = ["Memory", "MemoryCandidate", "RetrievedMemory"]
