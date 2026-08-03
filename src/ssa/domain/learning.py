"""Typed contracts for evidence-gated reflection and outcome learning."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ssa.domain.enums import SourceKind


class ReflectionTriggerKind(StrEnum):
    PREDICTION_ERROR = "prediction_error"
    UNRESOLVED = "unresolved"
    GOAL_RELEVANT = "goal_relevant"
    RELATIONSHIP_SALIENT = "relationship_salient"
    NOVEL = "novel"
    SCHEDULED = "scheduled"


class ReflectionRunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class LearningProposalType(StrEnum):
    EPISODE_NOTE = "episode_note"
    MEMORY_CANDIDATE = "memory_candidate"
    BELIEF_REVISION = "belief_revision"
    GOAL_ADJUSTMENT = "goal_adjustment"
    POLICY_PROPOSAL = "policy_proposal"


class LearningProposalStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    UNDER_TEST = "under_test"
    CONFIRMED = "confirmed"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


class EvidenceReferenceKind(StrEnum):
    EVENT = "event"
    TRACE = "trace"
    OFFLINE_ARTIFACT = "offline_artifact"
    OUTCOME_OBSERVATION = "outcome_observation"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class ConsolidationDecisionKind(StrEnum):
    APPROVE = "approve"
    DEFER = "defer"
    REJECT = "reject"
    ROLLBACK = "rollback"


class ExperimentStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class OutcomeVerdict(StrEnum):
    CONFIRMED = "confirmed"
    MIXED = "mixed"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class ReflectionRun(BaseModel):
    id: str
    conversation_id: str
    correlation_id: str
    dedup_key: str
    trigger_kind: ReflectionTriggerKind
    priority_score: float = Field(ge=0.0, le=1.0)
    score_components: dict[str, float] = Field(default_factory=dict)
    status: ReflectionRunStatus = ReflectionRunStatus.PLANNED
    source_episode_id: str | None = None
    source_artifact_id: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    reflection_text: str | None = None
    critique_text: str | None = None
    critic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = None
    prompt_version: str | None = None
    error: str | None = None
    started_at_ms: int = Field(ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _terminal_shape(self) -> ReflectionRun:
        if self.status == ReflectionRunStatus.COMPLETED:
            if not self.reflection_text or not self.critique_text:
                raise ValueError("completed reflection requires reflection and critique text")
            if self.critic_score is None or self.completed_at_ms is None:
                raise ValueError("completed reflection requires critic score and completion time")
        if self.status == ReflectionRunStatus.FAILED and (
            not self.error or self.completed_at_ms is None
        ):
            raise ValueError("failed reflection requires error and completion time")
        return self


class LearningProposal(BaseModel):
    id: str
    run_id: str
    conversation_id: str
    dedup_key: str
    proposal_type: LearningProposalType
    status: LearningProposalStatus = LearningProposalStatus.CANDIDATE
    title: str
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    critic_score: float = Field(ge=0.0, le=1.0)
    target_kind: str | None = None
    target_id: str | None = None
    rollback: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    applied_at_ms: int | None = Field(default=None, ge=0)

    @field_validator("title", "content", "rationale")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("learning proposal text must not be empty")
        return value


class ProposalEvidence(BaseModel):
    proposal_id: str
    reference_kind: EvidenceReferenceKind
    reference_id: str
    relation: EvidenceRelation
    provenance_kind: SourceKind
    trust_weight: float = Field(ge=0.0, le=1.0)
    likelihood_ratio: float = Field(gt=0.0)
    content_hash: str
    excerpt: str | None = None
    recorded_at_ms: int = Field(ge=0)


class ConsolidationDecision(BaseModel):
    id: str
    proposal_id: str
    decision_version: int = Field(ge=1)
    decision: ConsolidationDecisionKind
    gate_score: float = Field(ge=0.0, le=1.0)
    criteria: dict[str, Any] = Field(default_factory=dict)
    reason: str
    rules_version: str
    model: str | None = None
    prompt_version: str | None = None
    event_id: str | None = None
    created_at_ms: int = Field(ge=0)


class BehaviorExperiment(BaseModel):
    id: str
    conversation_id: str
    proposal_id: str
    dedup_key: str
    status: ExperimentStatus = ExperimentStatus.PLANNED
    hypothesis: str
    policy_key: str
    baseline: dict[str, Any] = Field(default_factory=dict)
    treatment: dict[str, Any] = Field(default_factory=dict)
    rollback: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    min_observations: int = Field(default=1, ge=1)
    result_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    conclusion: str | None = None
    started_at_ms: int | None = Field(default=None, ge=0)
    due_at_ms: int = Field(ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)


class OutcomeObservation(BaseModel):
    id: str
    experiment_id: str
    dedup_key: str
    source_event_id: str | None = None
    observation_kind: OutcomeVerdict
    payload: dict[str, Any] = Field(default_factory=dict)
    normalized_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    prediction_error: float = Field(ge=0.0, le=1.0)
    observed_at_ms: int = Field(ge=0)
    created_at_ms: int = Field(ge=0)


__all__ = [
    "BehaviorExperiment",
    "ConsolidationDecision",
    "ConsolidationDecisionKind",
    "EvidenceReferenceKind",
    "EvidenceRelation",
    "ExperimentStatus",
    "LearningProposal",
    "LearningProposalStatus",
    "LearningProposalType",
    "OutcomeObservation",
    "OutcomeVerdict",
    "ProposalEvidence",
    "ReflectionRun",
    "ReflectionRunStatus",
    "ReflectionTriggerKind",
]
