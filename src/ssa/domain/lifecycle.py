"""Typed contracts for goals, lifecycle jobs, initiatives, and delivery."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class GoalOwner(StrEnum):
    SELF = "self"
    SHARED = "shared"
    USER = "user"


class GoalStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    BLOCKED = "blocked"
    DONE = "done"
    ARCHIVED = "archived"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    DEAD_LETTER = "dead_letter"


class InitiativeStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    QUEUED = "queued"
    SENT = "sent"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    DEAD_LETTER = "dead_letter"


class EmotionType(StrEnum):
    ATTACHMENT = "attachment"
    CURIOSITY = "curiosity"
    CONCERN = "concern"
    ANTICIPATION = "anticipation"
    FRUSTRATION = "frustration"
    CONTENTMENT = "contentment"
    AMBIVALENCE = "ambivalence"


class EmotionStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class ContactPhase(StrEnum):
    OPENING = "opening"
    WAITING = "waiting"
    FOLLOW_UP = "follow_up"
    ESCALATION = "escalation"
    WITHDRAWAL = "withdrawal"
    RESOLUTION = "resolution"


class ContactStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class OfflineActionKind(StrEnum):
    TRACE_REFLECTION = "trace_reflection"
    EXTERNAL_STUDY = "external_study"


class OfflineEpisodeStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InnerLifeMode(StrEnum):
    ENGAGED = "engaged"
    WAITING = "waiting"
    OFFLINE = "offline"
    QUIET_REST = "quiet_rest"


class Goal(BaseModel):
    id: str
    conversation_id: str
    owner: GoalOwner
    title: str
    motive: str
    success_criteria: list[str]
    priority: float = Field(ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    status: GoalStatus = GoalStatus.PROPOSED
    due_at_ms: int | None = Field(default=None, ge=0)
    next_action_at_ms: int | None = Field(default=None, ge=0)
    correlation_id: str | None = None
    blocked_reason: str | None = None
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)

    @field_validator("id", "conversation_id", "title", "motive")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("success_criteria")
    @classmethod
    def _criteria_required(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("success_criteria must contain non-empty values")
        return value


class GoalStep(BaseModel):
    id: str
    goal_id: str
    action: str
    result: str | None = None
    source_event_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_ms: int = Field(ge=0)


class ScheduledJob(BaseModel):
    id: str
    conversation_id: str
    job_type: str
    dedup_key: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    due_at_ms: int = Field(ge=0)
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=5, ge=1)
    status: JobStatus = JobStatus.PENDING
    last_error: str | None = None
    leased_until_ms: int | None = Field(default=None, ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)


class Initiative(BaseModel):
    id: str
    conversation_id: str
    motive: str
    intent: str
    content_draft: str
    urgency: float = Field(ge=0.0, le=1.0)
    decision_score: float = Field(ge=-1.0, le=1.0)
    status: InitiativeStatus = InitiativeStatus.CANDIDATE
    earliest_send_at_ms: int = Field(ge=0)
    expires_at_ms: int | None = Field(default=None, ge=0)
    correlation_id: str | None = None
    goal_id: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    decision: dict[str, Any] = Field(default_factory=dict)
    channel: str = "local"
    dedup_key: str | None = None
    sent_event_id: str | None = None
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _send_window_is_valid(self) -> Initiative:
        if self.expires_at_ms is not None and self.expires_at_ms < self.earliest_send_at_ms:
            raise ValueError("initiative expiry must follow earliest send time")
        return self


class OutboxMessage(BaseModel):
    id: str
    correlation_id: str
    channel: str
    recipient: str
    payload: dict[str, Any]
    status: OutboxStatus = OutboxStatus.PENDING
    attempt_count: int = Field(default=0, ge=0)
    next_attempt_at_ms: int | None = Field(default=None, ge=0)
    delivered_at_ms: int | None = Field(default=None, ge=0)
    created_at_ms: int = Field(ge=0)
    initiative_id: str | None = None
    dedup_key: str | None = None
    delivered_event_id: str | None = None
    last_error: str | None = None


class CausalTraceRecord(BaseModel):
    id: str
    correlation_id: str
    input_event_id: str
    appraisal_id: str | None = None
    old_state_id: str | None = None
    new_state_id: str | None = None
    old_relationship_id: str | None = None
    new_relationship_id: str | None = None
    memory_ids: list[str] = Field(default_factory=list)
    goal_ids: list[str] = Field(default_factory=list)
    decision: dict[str, Any] = Field(default_factory=dict)
    output_event_id: str | None = None
    created_at_ms: int = Field(ge=0)


class EmotionEpisode(BaseModel):
    id: str
    conversation_id: str
    emotion_type: EmotionType
    target: str
    action_tendency: str
    intensity: float = Field(ge=0.0, le=1.0)
    inhibition: float = Field(ge=0.0, le=1.0)
    half_life_minutes: float = Field(gt=0.0)
    correlation_id: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    resolution_condition: str | None = None
    status: EmotionStatus = EmotionStatus.ACTIVE
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    resolved_at_ms: int | None = Field(default=None, ge=0)


class ContactEpisode(BaseModel):
    id: str
    conversation_id: str
    correlation_id: str
    motive: str
    topic: str
    phase: ContactPhase = ContactPhase.OPENING
    status: ContactStatus = ContactStatus.ACTIVE
    message_count: int = Field(default=0, ge=0)
    max_messages: int = Field(default=3, ge=1, le=8)
    hypotheses: dict[str, float] = Field(default_factory=dict)
    source_initiative_id: str | None = None
    last_sent_at_ms: int | None = Field(default=None, ge=0)
    next_action_at_ms: int | None = Field(default=None, ge=0)
    resolved_by_event_id: str | None = None
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)

    @field_validator("hypotheses")
    @classmethod
    def _hypotheses_are_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if any(weight < 0.0 or weight > 1.0 for weight in value.values()):
            raise ValueError("contact hypotheses must be in [0, 1]")
        return value


class WorldObservation(BaseModel):
    id: str
    conversation_id: str
    observation_type: str
    source: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    salience: float = Field(ge=0.0, le=1.0)
    dedup_key: str
    observed_at_ms: int = Field(ge=0)
    event_id: str | None = None


class OfflineEpisode(BaseModel):
    id: str
    conversation_id: str
    correlation_id: str
    dedup_key: str
    action_kind: OfflineActionKind
    motive: str
    status: OfflineEpisodeStatus = OfflineEpisodeStatus.PLANNED
    goal_id: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    tool_name: str | None = None
    tool_output_hash: str | None = None
    summary: str | None = None
    error: str | None = None
    event_id: str | None = None
    started_at_ms: int = Field(ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)


class OfflineArtifact(BaseModel):
    id: str
    episode_id: str
    artifact_type: str
    title: str
    content: str
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_trace_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_ms: int = Field(ge=0)

    @field_validator("artifact_type", "title", "content")
    @classmethod
    def _artifact_text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("offline artifact text must not be empty")
        return value


class InnerLoopState(BaseModel):
    conversation_id: str
    version: int = Field(ge=1)
    mode: InnerLifeMode
    reply_expectation: float = Field(ge=0.0, le=1.0)
    concern: float = Field(ge=0.0, le=1.0)
    curiosity: float = Field(ge=0.0, le=1.0)
    connection_pressure: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    offline_readiness: float = Field(ge=0.0, le=1.0)
    wait_started_at_ms: int | None = Field(default=None, ge=0)
    wait_deadline_at_ms: int | None = Field(default=None, ge=0)
    last_user_event_id: str | None = None
    last_agent_event_id: str | None = None
    last_heartbeat_at_ms: int = Field(ge=0)
    transition_reason: str
    updated_at_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _wait_window_is_valid(self) -> InnerLoopState:
        if (
            self.wait_started_at_ms is not None
            and self.wait_deadline_at_ms is not None
            and self.wait_deadline_at_ms < self.wait_started_at_ms
        ):
            raise ValueError("inner-loop wait deadline must follow its start")
        return self


__all__ = [
    "CausalTraceRecord",
    "ContactEpisode",
    "ContactPhase",
    "ContactStatus",
    "EmotionEpisode",
    "EmotionStatus",
    "EmotionType",
    "Goal",
    "GoalOwner",
    "GoalStatus",
    "GoalStep",
    "Initiative",
    "InitiativeStatus",
    "InnerLifeMode",
    "InnerLoopState",
    "JobStatus",
    "OfflineActionKind",
    "OfflineArtifact",
    "OfflineEpisode",
    "OfflineEpisodeStatus",
    "OutboxMessage",
    "OutboxStatus",
    "ScheduledJob",
    "WorldObservation",
]
