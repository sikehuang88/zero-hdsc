"""Domain enums — shared across the whole package.

Pipeline §4.1 defines these as the canonical actor / source / type
vocabularies. They are persisted as TEXT in SQLite and validated by
Pydantic on read/write.
"""

from __future__ import annotations

from enum import StrEnum


class Actor(StrEnum):
    """Who produced an event."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    WORLD = "world"


class SourceKind(StrEnum):
    """Provenance label — separates observed fact from model inference.

    Pipeline §2.2: model-generated content must never auto-promote to
    user-observed fact.
    """

    USER_OBSERVED = "user_observed"
    AGENT_OUTPUT = "agent_output"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_DERIVED = "system_derived"
    WORLD_OBSERVED = "world_observed"


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    SELF = "self"
    PROMISE = "promise"
    UNRESOLVED = "unresolved"
    REFLECTION = "reflection"


class ActionIntent(StrEnum):
    """What the agent decided to do this turn (pipeline §4.1)."""

    ACKNOWLEDGE = "acknowledge"
    ANSWER = "answer"
    ASK = "ask"
    COMFORT = "comfort"
    CHALLENGE = "challenge"
    REPAIR = "repair"
    SHARE = "share"
    DEFER = "defer"
    INITIATE = "initiate"
    PROJECT_WORK = "project_work"
    REST = "rest"


__all__ = ["ActionIntent", "Actor", "MemoryType", "SourceKind"]
