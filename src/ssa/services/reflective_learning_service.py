"""Continuous reflection, evidence criticism, consolidation, and outcome learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ssa.clock import Clock
from ssa.config import ReflectiveLearningConfig
from ssa.domain.enums import Actor, MemoryType, SourceKind
from ssa.domain.events import Event, IncomingSignal, compute_content_hash, normalize_signal
from ssa.domain.learning import (
    BehaviorExperiment,
    ConsolidationDecision,
    ConsolidationDecisionKind,
    EvidenceReferenceKind,
    EvidenceRelation,
    ExperimentStatus,
    LearningProposal,
    LearningProposalStatus,
    LearningProposalType,
    OutcomeObservation,
    OutcomeVerdict,
    ProposalEvidence,
    ReflectionRun,
    ReflectionRunStatus,
    ReflectionTriggerKind,
)
from ssa.domain.lifecycle import Goal, OfflineActionKind, OfflineArtifact, OfflineEpisode
from ssa.domain.memories import MemoryCandidate
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.ids import IdGenerator
from ssa.services.memory_write_service import MemoryWriteService
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.learning_repository import SqliteLearningRepository
from ssa.storage.lifecycle_repository import OfflineAgencyRepository
from ssa.storage.trace_repository import SqliteTraceRepository

_HOUR_MS = 3_600_000
_RULES_VERSION = "reflective-learning-rules-v1"
_PROMPT_VERSION = "offline-learning-v1"

_POSITIVE_CUES = (
    "谢谢",
    "很好",
    "喜欢",
    "有用",
    "有效",
    "记住了",
    "thanks",
    "helpful",
    "works well",
    "good idea",
)
_NEGATIVE_CUES = (
    "不对",
    "不好",
    "没用",
    "别这样",
    "讨厌",
    "错了",
    "not helpful",
    "doesn't work",
    "wrong",
    "stop doing",
)


@dataclass(frozen=True)
class ReflectionScheduleResult:
    runs: tuple[ReflectionRun, ...] = ()
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class ConsolidationResult:
    proposals: tuple[LearningProposal, ...] = ()
    decisions: tuple[ConsolidationDecision, ...] = ()
    target_ids: tuple[str, ...] = ()
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class OutcomeEvaluationResult:
    outcomes: tuple[OutcomeObservation, ...] = ()
    experiments: tuple[BehaviorExperiment, ...] = ()
    events: tuple[Event, ...] = ()


class ReflectiveLearningService:
    """Turn offline artifacts into reversible, evidence-linked learning effects."""

    def __init__(
        self,
        *,
        learning: SqliteLearningRepository,
        offline: OfflineAgencyRepository,
        events: SqliteEventRepository,
        traces: SqliteTraceRepository,
        memory_write: MemoryWriteService,
        clock: Clock,
        ids: IdGenerator,
        config: ReflectiveLearningConfig,
    ) -> None:
        self._learning = learning
        self._offline = offline
        self._events = events
        self._traces = traces
        self._memory_write = memory_write
        self._clock = clock
        self._ids = ids
        self._config = config

    def schedule_reflections(
        self,
        conversation_id: str,
        organism: OrganismState,
        relationship: RelationshipState,
        goals: list[Goal],
    ) -> ReflectionScheduleResult:
        if not self._config.enabled:
            return ReflectionScheduleResult()
        runs: list[ReflectionRun] = []
        emitted: list[Event] = []
        for episode, artifact in self._learning.unprocessed_artifacts(
            conversation_id,
            limit=self._config.max_runs_per_cycle,
        ):
            components = self._priority_components(
                episode,
                artifact,
                organism,
                relationship,
                goals,
            )
            priority = _clip(
                0.12
                + 0.22 * components["surprise"]
                + 0.18 * components["unresolved"]
                + 0.20 * components["goal_relevance"]
                + 0.16 * components["novelty"]
                + 0.16 * components["relationship_salience"]
                - 0.08 * components["cost"]
            )
            critic_components = self._critic_components(episode, artifact)
            critic_score = _clip(
                0.30 * critic_components["coverage"]
                + 0.30 * critic_components["source_trust"]
                + 0.25 * critic_components["consistency"]
                + 0.15 * critic_components["falsifiability"]
            )
            accepted = priority >= self._config.min_reflection_priority
            status = ReflectionRunStatus.COMPLETED if accepted else ReflectionRunStatus.SKIPPED
            trigger = max(
                (
                    (ReflectionTriggerKind.PREDICTION_ERROR, components["surprise"]),
                    (ReflectionTriggerKind.UNRESOLVED, components["unresolved"]),
                    (ReflectionTriggerKind.GOAL_RELEVANT, components["goal_relevance"]),
                    (
                        ReflectionTriggerKind.RELATIONSHIP_SALIENT,
                        components["relationship_salience"],
                    ),
                    (ReflectionTriggerKind.NOVEL, components["novelty"]),
                ),
                key=lambda item: item[1],
            )[0]
            now_ms = self._clock.now_ms()
            critique = (
                f"coverage={critic_components['coverage']:.3f}; "
                f"source_trust={critic_components['source_trust']:.3f}; "
                f"consistency={critic_components['consistency']:.3f}; "
                f"falsifiability={critic_components['falsifiability']:.3f}; "
                f"gate={'accepted' if accepted else 'below_priority'}"
            )
            run = self._learning.insert_run(
                ReflectionRun(
                    id=self._ids.new(),
                    conversation_id=conversation_id,
                    correlation_id=episode.correlation_id,
                    dedup_key=(
                        f"reflection:{conversation_id}:artifact:{artifact.id}:{_RULES_VERSION}"
                    ),
                    trigger_kind=trigger,
                    priority_score=priority,
                    score_components={**components, **critic_components},
                    status=status,
                    source_episode_id=episode.id,
                    source_artifact_id=artifact.id,
                    source_event_ids=list(artifact.evidence_event_ids),
                    source_trace_ids=list(artifact.evidence_trace_ids),
                    reflection_text=artifact.content if accepted else None,
                    critique_text=critique if accepted else None,
                    critic_score=critic_score if accepted else None,
                    model=str(artifact.metadata.get("reflection_source") or "deterministic"),
                    prompt_version=_PROMPT_VERSION,
                    started_at_ms=now_ms,
                    completed_at_ms=now_ms if accepted else None,
                )
            )
            runs.append(run)
            if accepted:
                emitted.append(
                    self._append_event(
                        conversation_id,
                        episode.correlation_id,
                        "learning.reflection_completed",
                        "An offline artifact passed the independent reflection gate.",
                        {
                            "reflection_run_id": run.id,
                            "source_artifact_id": artifact.id,
                            "priority_score": priority,
                            "critic_score": critic_score,
                        },
                    )
                )
        return ReflectionScheduleResult(tuple(runs), tuple(emitted))

    def consolidate(self, conversation_id: str) -> ConsolidationResult:
        if not self._config.enabled:
            return ConsolidationResult()
        touched: dict[str, LearningProposal] = {}
        decisions: list[ConsolidationDecision] = []
        targets: list[str] = []
        emitted: list[Event] = []

        runs = self._learning.recent_runs(
            conversation_id,
            {ReflectionRunStatus.COMPLETED},
            limit=self._config.max_runs_per_cycle * 4,
        )
        for run in reversed(runs):
            existing_types = {
                proposal.proposal_type for proposal in self._learning.proposals_for_run(run.id)
            }
            artifact = (
                self._offline.get_artifact(run.source_artifact_id)
                if run.source_artifact_id is not None
                else None
            )
            if artifact is None:
                continue
            if LearningProposalType.MEMORY_CANDIDATE not in existing_types:
                proposal = self._new_memory_proposal(run, artifact)
                touched[proposal.id] = proposal
                self._attach_evidence(proposal, run, artifact)
            next_action = str(artifact.metadata.get("next_action") or "").strip()
            if (
                next_action
                and self._config.policy_experiments_enabled
                and LearningProposalType.POLICY_PROPOSAL not in existing_types
            ):
                proposal = self._new_policy_proposal(run, artifact, next_action)
                touched[proposal.id] = proposal
                self._attach_evidence(proposal, run, artifact)

        ready = self._learning.list_proposals(
            conversation_id,
            {LearningProposalStatus.CANDIDATE, LearningProposalStatus.APPROVED},
            limit=self._config.max_proposals_per_cycle,
        )
        for proposal in reversed(ready):
            evidence = self._learning.evidence_for_proposal(proposal.id)
            if proposal.status == LearningProposalStatus.CANDIDATE:
                approved = (
                    proposal.critic_score >= self._config.min_critic_score
                    and len(evidence) >= self._config.min_evidence_count
                )
                decision_kind = (
                    ConsolidationDecisionKind.APPROVE
                    if approved
                    else ConsolidationDecisionKind.REJECT
                )
                new_status = (
                    LearningProposalStatus.APPROVED if approved else LearningProposalStatus.REJECTED
                )
                decision = ConsolidationDecision(
                    id=self._ids.new(),
                    proposal_id=proposal.id,
                    decision_version=1,
                    decision=decision_kind,
                    gate_score=proposal.critic_score,
                    criteria={
                        "critic_threshold": self._config.min_critic_score,
                        "evidence_count": len(evidence),
                        "minimum_evidence": self._config.min_evidence_count,
                    },
                    reason=(
                        "Evidence and critic gates passed."
                        if approved
                        else "Proposal did not satisfy deterministic consolidation gates."
                    ),
                    rules_version=_RULES_VERSION,
                    created_at_ms=self._clock.now_ms(),
                )
                proposal = self._learning.append_decision(
                    decision,
                    proposal.version,
                    proposal.model_copy(
                        update={
                            "status": new_status,
                            "version": proposal.version + 1,
                            "updated_at_ms": self._clock.now_ms(),
                        }
                    ),
                )
                decisions.append(decision)
                touched[proposal.id] = proposal
            if proposal.status != LearningProposalStatus.APPROVED:
                continue
            if proposal.proposal_type == LearningProposalType.MEMORY_CANDIDATE:
                applied, target_id = self._apply_memory(proposal)
            elif proposal.proposal_type == LearningProposalType.POLICY_PROPOSAL:
                applied, target_id = self._start_experiment(proposal)
            else:
                applied, target_id = proposal, None
            touched[applied.id] = applied
            if target_id is not None:
                targets.append(target_id)
                emitted.append(
                    self._append_event(
                        conversation_id,
                        run_correlation_id(applied, runs),
                        "learning.consolidated",
                        f"A {applied.proposal_type.value} passed consolidation.",
                        {
                            "learning_proposal_id": applied.id,
                            "target_kind": applied.target_kind,
                            "target_id": target_id,
                            "status": applied.status.value,
                        },
                    )
                )
        return ConsolidationResult(
            tuple(touched.values()),
            tuple(decisions),
            tuple(targets),
            tuple(emitted),
        )

    def evaluate_outcomes(self, conversation_id: str) -> OutcomeEvaluationResult:
        if not self._config.enabled:
            return OutcomeEvaluationResult()
        now_ms = self._clock.now_ms()
        outcomes: list[OutcomeObservation] = []
        experiments: list[BehaviorExperiment] = []
        emitted: list[Event] = []
        recent = self._events.recent_by_conversation(conversation_id, limit=240)
        for experiment in self._learning.list_due_experiments(conversation_id, now_ms):
            user_events = [
                event
                for event in recent
                if event.actor == Actor.USER
                and experiment.started_at_ms is not None
                and event.created_at_ms >= experiment.started_at_ms
            ]
            if len(user_events) < experiment.min_observations:
                postponed = experiment.model_copy(
                    update={
                        "due_at_ms": now_ms + self._config.outcome_interval_minutes * 60_000,
                        "updated_at_ms": now_ms,
                    }
                )
                experiments.append(
                    self._learning.update_experiment(
                        ExperimentStatus.RUNNING,
                        postponed,
                    )
                )
                continue
            signals = [_outcome_signal(event.content) for event in user_events]
            score = max(-1.0, min(1.0, sum(signals) / max(1, len(signals))))
            explicit = sum(signal != 0.0 for signal in signals)
            confidence = _clip(0.30 + 0.20 * len(user_events) + 0.20 * explicit)
            verdict = (
                OutcomeVerdict.CONFIRMED
                if score >= 0.20
                else OutcomeVerdict.CONTRADICTED
                if score <= -0.20
                else OutcomeVerdict.MIXED
                if explicit
                else OutcomeVerdict.INCONCLUSIVE
            )
            latest = user_events[-1]
            outcome = self._learning.insert_outcome(
                OutcomeObservation(
                    id=self._ids.new(),
                    experiment_id=experiment.id,
                    dedup_key=f"outcome:{experiment.id}:{latest.id}:{_RULES_VERSION}",
                    source_event_id=latest.id,
                    observation_kind=verdict,
                    payload={
                        "observed_event_ids": [event.id for event in user_events],
                        "explicit_signal_count": explicit,
                    },
                    normalized_score=score,
                    confidence=confidence,
                    prediction_error=_clip(abs(0.35 - score) / 1.35),
                    observed_at_ms=latest.created_at_ms,
                    created_at_ms=now_ms,
                )
            )
            outcomes.append(outcome)
            proposal = self._learning.get_proposal(experiment.proposal_id)
            final_status = (
                LearningProposalStatus.CONFIRMED
                if verdict == OutcomeVerdict.CONFIRMED
                else LearningProposalStatus.ROLLED_BACK
            )
            if proposal is not None and proposal.status == LearningProposalStatus.UNDER_TEST:
                relation = (
                    EvidenceRelation.SUPPORTS
                    if final_status == LearningProposalStatus.CONFIRMED
                    else EvidenceRelation.CONTRADICTS
                )
                self._learning.add_evidence(
                    ProposalEvidence(
                        proposal_id=proposal.id,
                        reference_kind=EvidenceReferenceKind.OUTCOME_OBSERVATION,
                        reference_id=outcome.id,
                        relation=relation,
                        provenance_kind=SourceKind.SYSTEM_DERIVED,
                        trust_weight=confidence,
                        likelihood_ratio=max(0.05, 1.0 + score),
                        content_hash=compute_content_hash(
                            f"{outcome.observation_kind.value}:{score:.6f}"
                        ),
                        excerpt=f"Outcome {verdict.value} with score {score:.3f}",
                        recorded_at_ms=now_ms,
                    )
                )
                previous_decisions = self._learning.decisions_for_proposal(proposal.id)
                decision = ConsolidationDecision(
                    id=self._ids.new(),
                    proposal_id=proposal.id,
                    decision_version=len(previous_decisions) + 1,
                    decision=(
                        ConsolidationDecisionKind.APPROVE
                        if final_status == LearningProposalStatus.CONFIRMED
                        else ConsolidationDecisionKind.ROLLBACK
                    ),
                    gate_score=confidence,
                    criteria={"outcome_score": score, "verdict": verdict.value},
                    reason=(
                        "Observed feedback confirmed the bounded policy experiment."
                        if final_status == LearningProposalStatus.CONFIRMED
                        else "Observed feedback did not confirm the policy experiment."
                    ),
                    rules_version=_RULES_VERSION,
                    created_at_ms=now_ms,
                )
                self._learning.append_decision(
                    decision,
                    proposal.version,
                    proposal.model_copy(
                        update={
                            "status": final_status,
                            "version": proposal.version + 1,
                            "updated_at_ms": now_ms,
                        }
                    ),
                )
            completed = experiment.model_copy(
                update={
                    "status": ExperimentStatus.COMPLETED,
                    "result_score": score,
                    "conclusion": verdict.value,
                    "completed_at_ms": now_ms,
                    "updated_at_ms": now_ms,
                }
            )
            experiments.append(
                self._learning.update_experiment(ExperimentStatus.RUNNING, completed)
            )
            emitted.append(
                self._append_event(
                    conversation_id,
                    self._ids.new(),
                    "learning.outcome_evaluated",
                    f"A policy experiment concluded as {verdict.value}.",
                    {
                        "experiment_id": experiment.id,
                        "proposal_id": experiment.proposal_id,
                        "outcome_id": outcome.id,
                        "score": score,
                        "confidence": confidence,
                    },
                )
            )
        return OutcomeEvaluationResult(tuple(outcomes), tuple(experiments), tuple(emitted))

    def context_summary(self, conversation_id: str) -> str:
        active = self._learning.active_experiments(conversation_id, limit=3)
        confirmed = self._learning.list_proposals(
            conversation_id,
            {LearningProposalStatus.CONFIRMED},
            proposal_type=LearningProposalType.POLICY_PROPOSAL,
            limit=3,
        )
        if not active and not confirmed:
            return "Reflective learning policy evidence: no active or confirmed policy guidance."
        lines = ["Reflective learning policy evidence (private soft controls, never user facts):"]
        for experiment in active:
            instruction = str(experiment.treatment.get("instruction") or "")[:400]
            lines.append(
                f"- active experiment {experiment.id}; policy={experiment.policy_key}; "
                f"instruction={instruction}; due_at_ms={experiment.due_at_ms}"
            )
        for proposal in confirmed:
            instruction = str(proposal.payload.get("instruction") or proposal.content)[:400]
            lines.append(
                f"- confirmed proposal {proposal.id}; instruction={instruction}; "
                f"critic_score={proposal.critic_score:.3f}"
            )
        lines.append(
            "Apply guidance lightly and never claim an experiment succeeded before outcome evidence."
        )
        return "\n".join(lines)

    def _new_memory_proposal(
        self,
        run: ReflectionRun,
        artifact: OfflineArtifact,
    ) -> LearningProposal:
        now_ms = self._clock.now_ms()
        return self._learning.insert_proposal(
            LearningProposal(
                id=self._ids.new(),
                run_id=run.id,
                conversation_id=run.conversation_id,
                dedup_key=f"proposal:{run.id}:memory:{_RULES_VERSION}",
                proposal_type=LearningProposalType.MEMORY_CANDIDATE,
                title=artifact.title,
                content=artifact.content,
                payload={
                    "memory_type": MemoryType.REFLECTION.value,
                    "importance": max(0.35, run.priority_score),
                },
                rationale="Preserve an evidence-linked reflection without promoting it to fact.",
                confidence=min(0.60, 0.25 + 0.45 * (run.critic_score or 0.0)),
                critic_score=run.critic_score or 0.0,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
        )

    def _new_policy_proposal(
        self,
        run: ReflectionRun,
        artifact: OfflineArtifact,
        next_action: str,
    ) -> LearningProposal:
        now_ms = self._clock.now_ms()
        return self._learning.insert_proposal(
            LearningProposal(
                id=self._ids.new(),
                run_id=run.id,
                conversation_id=run.conversation_id,
                dedup_key=f"proposal:{run.id}:policy:{_RULES_VERSION}",
                proposal_type=LearningProposalType.POLICY_PROPOSAL,
                title=f"Bounded experiment: {artifact.title}"[:240],
                content=next_action[:800],
                payload={
                    "policy_key": "reflection_guidance",
                    "instruction": next_action[:800],
                },
                rationale="Test a reversible behavioral adjustment before retaining it.",
                confidence=min(0.55, 0.20 + 0.40 * (run.critic_score or 0.0)),
                critic_score=_clip((run.critic_score or 0.0) * 0.95),
                rollback={"instruction": None},
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
        )

    def _attach_evidence(
        self,
        proposal: LearningProposal,
        run: ReflectionRun,
        artifact: OfflineArtifact,
    ) -> None:
        now_ms = self._clock.now_ms()
        provider_reported = artifact.metadata.get("truth_status") == "provider_reported"
        self._learning.add_evidence(
            ProposalEvidence(
                proposal_id=proposal.id,
                reference_kind=EvidenceReferenceKind.OFFLINE_ARTIFACT,
                reference_id=artifact.id,
                relation=EvidenceRelation.SUPPORTS,
                provenance_kind=SourceKind.MODEL_INFERENCE,
                trust_weight=0.55 if provider_reported else 0.40,
                likelihood_ratio=1.20,
                content_hash=compute_content_hash(artifact.content),
                excerpt=artifact.content[:500],
                recorded_at_ms=now_ms,
            )
        )
        for event_id in run.source_event_ids:
            event = self._events.get(event_id)
            if event is None:
                continue
            self._learning.add_evidence(
                ProposalEvidence(
                    proposal_id=proposal.id,
                    reference_kind=EvidenceReferenceKind.EVENT,
                    reference_id=event.id,
                    relation=EvidenceRelation.SUPPORTS,
                    provenance_kind=event.source_kind,
                    trust_weight=_source_trust(event.source_kind),
                    likelihood_ratio=1.35,
                    content_hash=event.content_hash,
                    excerpt=event.content[:500],
                    recorded_at_ms=now_ms,
                )
            )
        for trace_id in run.source_trace_ids:
            trace = self._traces.get(trace_id)
            if trace is None:
                continue
            self._learning.add_evidence(
                ProposalEvidence(
                    proposal_id=proposal.id,
                    reference_kind=EvidenceReferenceKind.TRACE,
                    reference_id=trace.id,
                    relation=EvidenceRelation.SUPPORTS,
                    provenance_kind=trace.source_kind,
                    trust_weight=_source_trust(trace.source_kind),
                    likelihood_ratio=1.25,
                    content_hash=compute_content_hash(trace.content),
                    excerpt=trace.content[:500],
                    recorded_at_ms=now_ms,
                )
            )

    def _apply_memory(self, proposal: LearningProposal) -> tuple[LearningProposal, str | None]:
        evidence = self._learning.evidence_for_proposal(proposal.id)
        event_ids = [
            item.reference_id
            for item in evidence
            if item.reference_kind == EvidenceReferenceKind.EVENT
        ]
        if not event_ids:
            return proposal, None
        result = self._memory_write.write_candidates(
            [
                MemoryCandidate(
                    memory_type=MemoryType.REFLECTION,
                    content=proposal.content,
                    source_kind=SourceKind.MODEL_INFERENCE,
                    evidence_event_ids=list(dict.fromkeys(event_ids)),
                    confidence=min(0.60, proposal.confidence),
                    importance=float(proposal.payload.get("importance", 0.5)),
                    valence=0.0,
                    arousal=0.2,
                    derived_by_model="reflective-learning-consolidator-v1",
                )
            ]
        )
        target_ids = [*result.created, *result.merged]
        if not target_ids:
            return proposal, None
        target_id = target_ids[0]
        applied = self._learning.update_proposal(
            proposal.version,
            proposal.model_copy(
                update={
                    "status": LearningProposalStatus.APPLIED,
                    "target_kind": "memory",
                    "target_id": target_id,
                    "version": proposal.version + 1,
                    "updated_at_ms": self._clock.now_ms(),
                    "applied_at_ms": self._clock.now_ms(),
                }
            ),
        )
        return applied, target_id

    def _start_experiment(
        self,
        proposal: LearningProposal,
    ) -> tuple[LearningProposal, str]:
        now_ms = self._clock.now_ms()
        experiment = self._learning.insert_experiment(
            BehaviorExperiment(
                id=self._ids.new(),
                conversation_id=proposal.conversation_id,
                proposal_id=proposal.id,
                dedup_key=f"experiment:{proposal.id}:{_RULES_VERSION}",
                status=ExperimentStatus.RUNNING,
                hypothesis="The bounded reflection guidance improves later user-observed utility.",
                policy_key=str(proposal.payload.get("policy_key") or "reflection_guidance"),
                baseline={"instruction": None},
                treatment={"instruction": str(proposal.payload.get("instruction") or "")[:800]},
                rollback=dict(proposal.rollback),
                success_criteria={"minimum_normalized_score": 0.20},
                min_observations=self._config.experiment_min_observations,
                started_at_ms=now_ms,
                due_at_ms=now_ms + self._config.experiment_duration_hours * _HOUR_MS,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
        )
        under_test = self._learning.update_proposal(
            proposal.version,
            proposal.model_copy(
                update={
                    "status": LearningProposalStatus.UNDER_TEST,
                    "target_kind": "behavior_experiment",
                    "target_id": experiment.id,
                    "version": proposal.version + 1,
                    "updated_at_ms": now_ms,
                    "applied_at_ms": now_ms,
                }
            ),
        )
        return under_test, experiment.id

    def _priority_components(
        self,
        episode: OfflineEpisode,
        artifact: OfflineArtifact,
        organism: OrganismState,
        relationship: RelationshipState,
        goals: list[Goal],
    ) -> dict[str, float]:
        traces = [
            trace
            for trace_id in artifact.evidence_trace_ids
            if (trace := self._traces.get(trace_id)) is not None
        ]
        surprise = max(
            (
                _clip(0.35 * abs(trace.valence) + 0.35 * trace.arousal + 0.30 * trace.importance)
                for trace in traces
            ),
            default=0.65 if episode.action_kind == OfflineActionKind.EXTERNAL_STUDY else 0.35,
        )
        novelty = max(
            (
                math.exp(-max(0, self._clock.now_ms() - trace.created_at_ms) / (7 * 86_400_000))
                for trace in traces
            ),
            default=0.80 if episode.action_kind == OfflineActionKind.EXTERNAL_STUDY else 0.40,
        )
        return {
            "surprise": surprise,
            "unresolved": _clip(0.55 * relationship.repair_debt + 0.45 * relationship.tension),
            "goal_relevance": max(
                (goal.priority * (1.0 - goal.progress) for goal in goals), default=0.25
            ),
            "novelty": _clip(novelty),
            "relationship_salience": _clip(
                relationship.closeness * max((trace.importance for trace in traces), default=0.35)
            ),
            "cost": _clip(1.0 - organism.energy),
        }

    def _critic_components(
        self,
        episode: OfflineEpisode,
        artifact: OfflineArtifact,
    ) -> dict[str, float]:
        events = [
            event
            for event_id in artifact.evidence_event_ids
            if (event := self._events.get(event_id)) is not None
        ]
        trust = [_source_trust(event.source_kind) for event in events]
        if artifact.metadata.get("truth_status") == "provider_reported":
            trust.append(0.65)
        questions = artifact.metadata.get("questions")
        return {
            "coverage": 1.0 if events or artifact.evidence_trace_ids else 0.45,
            "source_trust": sum(trust) / len(trust) if trust else 0.40,
            "consistency": 1.0 if episode.error is None and episode.completed_at_ms else 0.0,
            "falsifiability": 0.75 if isinstance(questions, list) and questions else 0.45,
        }

    def _append_event(
        self,
        conversation_id: str,
        correlation_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, object],
    ) -> Event:
        event_id = self._ids.new()
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type=event_type,
                    content=content,
                    channel="lifecycle",
                    channel_message_id=event_id,
                    conversation_id=conversation_id,
                    metadata=metadata,
                ),
                event_id=event_id,
                correlation_id=correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )


def run_correlation_id(proposal: LearningProposal, runs: list[ReflectionRun]) -> str:
    return next(
        (run.correlation_id for run in runs if run.id == proposal.run_id),
        proposal.run_id,
    )


def _source_trust(source: SourceKind) -> float:
    return {
        SourceKind.USER_OBSERVED: 1.00,
        SourceKind.AGENT_OUTPUT: 0.75,
        SourceKind.SYSTEM_DERIVED: 0.65,
        SourceKind.WORLD_OBSERVED: 0.70,
        SourceKind.MODEL_INFERENCE: 0.40,
    }.get(source, 0.40)


def _outcome_signal(content: str) -> float:
    text = content.casefold()
    if any(cue in text for cue in _NEGATIVE_CUES):
        return -1.0
    if any(cue in text for cue in _POSITIVE_CUES):
        return 1.0
    return 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "ConsolidationResult",
    "OutcomeEvaluationResult",
    "ReflectionScheduleResult",
    "ReflectiveLearningService",
]
