"""Event domain model — the immutable fact ledger entry.

Pipeline §2.1: any information that influences future behavior must
first become an Event. Events are append-only; corrections are new
events referencing the old one via `parent_event_id`.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ssa.domain.enums import Actor, SourceKind


class IncomingSignal(BaseModel):
    """Raw signal entering the system, before normalization into an Event.

    Pipeline §12.2.
    """

    actor: Actor
    signal_type: str
    content: str
    channel: str
    channel_message_id: str | None = None
    conversation_id: str
    parent_event_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("signal content must not be empty")
        return v

    @field_validator("signal_type")
    @classmethod
    def _type_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("signal_type must not be empty")
        return v


class Event(BaseModel):
    """An immutable, persisted event.

    Once written, an Event is never updated. Corrections create a new
    Event with `parent_event_id` pointing back.
    """

    id: str
    correlation_id: str
    conversation_id: str
    actor: Actor
    event_type: str
    source_kind: SourceKind
    content: str
    parent_event_id: str | None = None
    channel: str | None = None
    channel_message_id: str | None = None
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_ms: int

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("event content must not be empty")
        return v

    def to_jsonl(self) -> str:
        """Serialize for JSONL export (pipeline §27.1)."""
        return self.model_dump_json()

    def to_export_dict(self) -> dict[str, Any]:
        """Flat dict for identity-package export."""
        d = self.model_dump(mode="json")
        # Ensure enums are strings.
        d["actor"] = str(self.actor.value)
        d["source_kind"] = str(self.source_kind.value)
        return d


def compute_content_hash(content: str) -> str:
    """SHA-256 of the raw content (pipeline §12.3 step 4)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_signal(
    signal: IncomingSignal,
    *,
    event_id: str,
    correlation_id: str,
    now_ms: int,
    source_kind: SourceKind,
) -> Event:
    """Convert an IncomingSignal into an immutable Event.

    Pipeline §12.3 normalization steps:
    1. Strip NUL characters.
    2. Preserve newlines.
    3. (Length capping handled by the caller / interface layer.)
    4. Compute SHA-256 content hash.
    5. Use provided UTC ms timestamp.
    6. Use provided event_id and correlation_id.
    7. Actor, source_kind, event_type from signal + caller.
    """
    cleaned = signal.content.replace("\x00", "")
    return Event(
        id=event_id,
        correlation_id=correlation_id,
        conversation_id=signal.conversation_id,
        actor=signal.actor,
        event_type=signal.signal_type,
        source_kind=source_kind,
        content=cleaned,
        parent_event_id=signal.parent_event_id,
        channel=signal.channel,
        channel_message_id=signal.channel_message_id,
        content_hash=compute_content_hash(cleaned),
        metadata=dict(signal.metadata),
        created_at_ms=now_ms,
    )


__all__ = [
    "Event",
    "IncomingSignal",
    "compute_content_hash",
    "normalize_signal",
]
