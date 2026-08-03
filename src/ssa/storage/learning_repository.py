"""SQLite persistence for evidence-gated reflective learning."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ssa.domain.enums import SourceKind
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
from ssa.domain.lifecycle import OfflineArtifact, OfflineEpisode, OfflineEpisodeStatus


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class LearningVersionConflict(RuntimeError):
    """Raised when another worker advanced a learning aggregate first."""


class SqliteLearningRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_run(self, run: ReflectionRun) -> ReflectionRun:
        existing = self.find_run_by_dedup(run.dedup_key)
        if existing is not None:
            return existing
        self._conn.execute(
            """
            INSERT INTO reflection_runs (
                id, conversation_id, correlation_id, dedup_key, trigger_kind,
                priority_score, score_components_json, status, source_episode_id,
                source_artifact_id, source_event_ids_json, source_trace_ids_json,
                reflection_text, critique_text, critic_score, model,
                prompt_version, error, started_at_ms, completed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._run_values(run),
        )
        return run

    def get_run(self, run_id: str) -> ReflectionRun | None:
        row = self._conn.execute("SELECT * FROM reflection_runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row is not None else None

    def find_run_by_dedup(self, dedup_key: str) -> ReflectionRun | None:
        row = self._conn.execute(
            "SELECT * FROM reflection_runs WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_run(row) if row is not None else None

    def update_run(
        self,
        expected_status: ReflectionRunStatus,
        run: ReflectionRun,
    ) -> ReflectionRun:
        cursor = self._conn.execute(
            """
            UPDATE reflection_runs SET
                trigger_kind = ?, priority_score = ?, score_components_json = ?,
                status = ?, source_episode_id = ?, source_artifact_id = ?,
                source_event_ids_json = ?, source_trace_ids_json = ?,
                reflection_text = ?, critique_text = ?, critic_score = ?, model = ?,
                prompt_version = ?, error = ?, started_at_ms = ?, completed_at_ms = ?
            WHERE id = ? AND conversation_id = ? AND status = ?
            """,
            (
                run.trigger_kind.value,
                run.priority_score,
                _json(run.score_components),
                run.status.value,
                run.source_episode_id,
                run.source_artifact_id,
                _json(run.source_event_ids),
                _json(run.source_trace_ids),
                run.reflection_text,
                run.critique_text,
                run.critic_score,
                run.model,
                run.prompt_version,
                run.error,
                run.started_at_ms,
                run.completed_at_ms,
                run.id,
                run.conversation_id,
                expected_status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise LearningVersionConflict(f"reflection run {run.id!r} changed concurrently")
        return run

    def unprocessed_artifacts(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[tuple[OfflineEpisode, OfflineArtifact]]:
        rows = self._conn.execute(
            """
            SELECT
                episode.id AS ep_id, episode.conversation_id AS ep_conversation_id,
                episode.correlation_id AS ep_correlation_id,
                episode.dedup_key AS ep_dedup_key, episode.action_kind AS ep_action_kind,
                episode.motive AS ep_motive, episode.status AS ep_status,
                episode.goal_id AS ep_goal_id,
                episode.source_event_ids_json AS ep_source_event_ids_json,
                episode.source_trace_ids_json AS ep_source_trace_ids_json,
                episode.provider AS ep_provider, episode.tool_name AS ep_tool_name,
                episode.tool_output_hash AS ep_tool_output_hash,
                episode.summary AS ep_summary, episode.error AS ep_error,
                episode.event_id AS ep_event_id, episode.started_at_ms AS ep_started_at_ms,
                episode.completed_at_ms AS ep_completed_at_ms,
                artifact.id AS ar_id, artifact.episode_id AS ar_episode_id,
                artifact.artifact_type AS ar_artifact_type, artifact.title AS ar_title,
                artifact.content AS ar_content,
                artifact.evidence_event_ids_json AS ar_evidence_event_ids_json,
                artifact.evidence_trace_ids_json AS ar_evidence_trace_ids_json,
                artifact.metadata_json AS ar_metadata_json,
                artifact.created_at_ms AS ar_created_at_ms
            FROM offline_artifacts AS artifact
            JOIN offline_episodes AS episode ON episode.id = artifact.episode_id
            LEFT JOIN reflection_runs AS run ON run.source_artifact_id = artifact.id
            WHERE episode.conversation_id = ?
              AND episode.status = 'completed'
              AND run.id IS NULL
            ORDER BY artifact.created_at_ms, artifact.rowid
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [(self._aliased_episode(row), self._aliased_artifact(row)) for row in rows]

    def completed_runs_without_proposals(
        self,
        conversation_id: str,
        *,
        limit: int = 10,
    ) -> list[ReflectionRun]:
        rows = self._conn.execute(
            """
            SELECT run.* FROM reflection_runs AS run
            LEFT JOIN learning_proposals AS proposal ON proposal.run_id = run.id
            WHERE run.conversation_id = ? AND run.status = 'completed'
              AND proposal.id IS NULL
            ORDER BY run.completed_at_ms, run.rowid LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def recent_runs(
        self,
        conversation_id: str,
        statuses: set[ReflectionRunStatus],
        *,
        limit: int = 20,
    ) -> list[ReflectionRun]:
        if not statuses:
            return []
        values = sorted(status.value for status in statuses)
        placeholders = ", ".join("?" for _ in values)
        rows = self._conn.execute(
            f"""
            SELECT * FROM reflection_runs
            WHERE conversation_id = ? AND status IN ({placeholders})
            ORDER BY started_at_ms DESC, rowid DESC LIMIT ?
            """,
            (conversation_id, *values, limit),
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def insert_proposal(self, proposal: LearningProposal) -> LearningProposal:
        existing = self.find_proposal_by_dedup(proposal.dedup_key)
        if existing is not None:
            return existing
        run = self.get_run(proposal.run_id)
        if run is None or run.status != ReflectionRunStatus.COMPLETED:
            raise ValueError("learning proposals require a completed reflection run")
        if run.conversation_id != proposal.conversation_id:
            raise ValueError("proposal conversation must match its reflection run")
        self._conn.execute(
            """
            INSERT INTO learning_proposals (
                id, run_id, conversation_id, dedup_key, proposal_type, status,
                title, content, payload_json, rationale, confidence, critic_score,
                target_kind, target_id, rollback_json, version, created_at_ms,
                updated_at_ms, applied_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._proposal_values(proposal),
        )
        return proposal

    def get_proposal(self, proposal_id: str) -> LearningProposal | None:
        row = self._conn.execute(
            "SELECT * FROM learning_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        return self._row_to_proposal(row) if row is not None else None

    def find_proposal_by_dedup(self, dedup_key: str) -> LearningProposal | None:
        row = self._conn.execute(
            "SELECT * FROM learning_proposals WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_proposal(row) if row is not None else None

    def proposals_for_run(self, run_id: str) -> list[LearningProposal]:
        rows = self._conn.execute(
            "SELECT * FROM learning_proposals WHERE run_id = ? ORDER BY created_at_ms, rowid",
            (run_id,),
        ).fetchall()
        return [self._row_to_proposal(row) for row in rows]

    def list_proposals(
        self,
        conversation_id: str,
        statuses: set[LearningProposalStatus],
        *,
        proposal_type: LearningProposalType | None = None,
        limit: int = 20,
    ) -> list[LearningProposal]:
        if not statuses:
            return []
        values = sorted(status.value for status in statuses)
        placeholders = ", ".join("?" for _ in values)
        sql = (
            "SELECT * FROM learning_proposals WHERE conversation_id = ? "
            f"AND status IN ({placeholders})"
        )
        params: list[object] = [conversation_id, *values]
        if proposal_type is not None:
            sql += " AND proposal_type = ?"
            params.append(proposal_type.value)
        sql += " ORDER BY updated_at_ms DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_proposal(row) for row in rows]

    def update_proposal(
        self,
        expected_version: int,
        proposal: LearningProposal,
    ) -> LearningProposal:
        if proposal.version != expected_version + 1:
            raise ValueError("learning proposal version must increment by one")
        cursor = self._conn.execute(
            """
            UPDATE learning_proposals SET
                status = ?, title = ?, content = ?, payload_json = ?, rationale = ?,
                confidence = ?, critic_score = ?, target_kind = ?, target_id = ?,
                rollback_json = ?, version = ?, updated_at_ms = ?, applied_at_ms = ?
            WHERE id = ? AND conversation_id = ? AND version = ?
            """,
            (
                proposal.status.value,
                proposal.title,
                proposal.content,
                _json(proposal.payload),
                proposal.rationale,
                proposal.confidence,
                proposal.critic_score,
                proposal.target_kind,
                proposal.target_id,
                _json(proposal.rollback),
                proposal.version,
                proposal.updated_at_ms,
                proposal.applied_at_ms,
                proposal.id,
                proposal.conversation_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise LearningVersionConflict(f"learning proposal {proposal.id!r} changed concurrently")
        return proposal

    def add_evidence(self, evidence: ProposalEvidence) -> ProposalEvidence:
        proposal = self.get_proposal(evidence.proposal_id)
        if proposal is None:
            raise KeyError(f"unknown proposal {evidence.proposal_id!r}")
        self._verify_reference(proposal.conversation_id, evidence)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO learning_proposal_evidence (
                proposal_id, reference_kind, reference_id, relation,
                provenance_kind, trust_weight, likelihood_ratio, content_hash,
                excerpt, recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.proposal_id,
                evidence.reference_kind.value,
                evidence.reference_id,
                evidence.relation.value,
                evidence.provenance_kind.value,
                evidence.trust_weight,
                evidence.likelihood_ratio,
                evidence.content_hash,
                evidence.excerpt,
                evidence.recorded_at_ms,
            ),
        )
        return evidence

    def evidence_for_proposal(self, proposal_id: str) -> list[ProposalEvidence]:
        rows = self._conn.execute(
            """
            SELECT * FROM learning_proposal_evidence
            WHERE proposal_id = ? ORDER BY recorded_at_ms, reference_kind, reference_id
            """,
            (proposal_id,),
        ).fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def append_decision(
        self,
        decision: ConsolidationDecision,
        expected_version: int,
        proposal: LearningProposal,
    ) -> LearningProposal:
        with self._savepoint("learning_decision"):
            self._conn.execute(
                """
                INSERT INTO consolidation_decisions (
                    id, proposal_id, decision_version, decision, gate_score,
                    criteria_json, reason, rules_version, model, prompt_version,
                    event_id, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.proposal_id,
                    decision.decision_version,
                    decision.decision.value,
                    decision.gate_score,
                    _json(decision.criteria),
                    decision.reason,
                    decision.rules_version,
                    decision.model,
                    decision.prompt_version,
                    decision.event_id,
                    decision.created_at_ms,
                ),
            )
            self.update_proposal(expected_version, proposal)
        return proposal

    def decisions_for_proposal(self, proposal_id: str) -> list[ConsolidationDecision]:
        rows = self._conn.execute(
            """
            SELECT * FROM consolidation_decisions
            WHERE proposal_id = ? ORDER BY decision_version
            """,
            (proposal_id,),
        ).fetchall()
        return [self._row_to_decision(row) for row in rows]

    def insert_experiment(self, experiment: BehaviorExperiment) -> BehaviorExperiment:
        existing = self.find_experiment_by_dedup(experiment.dedup_key)
        if existing is not None:
            return existing
        proposal = self.get_proposal(experiment.proposal_id)
        if proposal is None or proposal.proposal_type != LearningProposalType.POLICY_PROPOSAL:
            raise ValueError("only policy proposals can create behavior experiments")
        if proposal.conversation_id != experiment.conversation_id:
            raise ValueError("experiment conversation must match its policy proposal")
        self._conn.execute(
            """
            INSERT INTO behavior_experiments (
                id, conversation_id, proposal_id, dedup_key, status, hypothesis,
                policy_key, baseline_json, treatment_json, rollback_json,
                success_criteria_json, min_observations, result_score, conclusion,
                started_at_ms, due_at_ms, completed_at_ms, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._experiment_values(experiment),
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> BehaviorExperiment | None:
        row = self._conn.execute(
            "SELECT * FROM behavior_experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return self._row_to_experiment(row) if row is not None else None

    def find_experiment_by_dedup(self, dedup_key: str) -> BehaviorExperiment | None:
        row = self._conn.execute(
            "SELECT * FROM behavior_experiments WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_experiment(row) if row is not None else None

    def list_due_experiments(
        self,
        conversation_id: str,
        now_ms: int,
        *,
        limit: int = 10,
    ) -> list[BehaviorExperiment]:
        rows = self._conn.execute(
            """
            SELECT * FROM behavior_experiments
            WHERE conversation_id = ? AND status = 'running' AND due_at_ms <= ?
            ORDER BY due_at_ms, rowid LIMIT ?
            """,
            (conversation_id, now_ms, limit),
        ).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def active_experiments(
        self,
        conversation_id: str,
        *,
        limit: int = 8,
    ) -> list[BehaviorExperiment]:
        rows = self._conn.execute(
            """
            SELECT * FROM behavior_experiments
            WHERE conversation_id = ? AND status = 'running'
            ORDER BY updated_at_ms DESC, rowid DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def recent_experiments(
        self,
        conversation_id: str,
        *,
        limit: int = 12,
    ) -> list[BehaviorExperiment]:
        rows = self._conn.execute(
            """
            SELECT * FROM behavior_experiments
            WHERE conversation_id = ?
            ORDER BY updated_at_ms DESC, rowid DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def update_experiment(
        self,
        expected_status: ExperimentStatus,
        experiment: BehaviorExperiment,
    ) -> BehaviorExperiment:
        cursor = self._conn.execute(
            """
            UPDATE behavior_experiments SET
                status = ?, hypothesis = ?, policy_key = ?, baseline_json = ?,
                treatment_json = ?, rollback_json = ?, success_criteria_json = ?,
                min_observations = ?, result_score = ?, conclusion = ?,
                started_at_ms = ?, due_at_ms = ?, completed_at_ms = ?, updated_at_ms = ?
            WHERE id = ? AND conversation_id = ? AND status = ?
            """,
            (
                experiment.status.value,
                experiment.hypothesis,
                experiment.policy_key,
                _json(experiment.baseline),
                _json(experiment.treatment),
                _json(experiment.rollback),
                _json(experiment.success_criteria),
                experiment.min_observations,
                experiment.result_score,
                experiment.conclusion,
                experiment.started_at_ms,
                experiment.due_at_ms,
                experiment.completed_at_ms,
                experiment.updated_at_ms,
                experiment.id,
                experiment.conversation_id,
                expected_status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise LearningVersionConflict(
                f"behavior experiment {experiment.id!r} changed concurrently"
            )
        return experiment

    def insert_outcome(self, outcome: OutcomeObservation) -> OutcomeObservation:
        existing = self.find_outcome_by_dedup(outcome.dedup_key)
        if existing is not None:
            return existing
        self._conn.execute(
            """
            INSERT INTO outcome_observations (
                id, experiment_id, dedup_key, source_event_id, observation_kind,
                payload_json, normalized_score, confidence, prediction_error,
                observed_at_ms, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.id,
                outcome.experiment_id,
                outcome.dedup_key,
                outcome.source_event_id,
                outcome.observation_kind.value,
                _json(outcome.payload),
                outcome.normalized_score,
                outcome.confidence,
                outcome.prediction_error,
                outcome.observed_at_ms,
                outcome.created_at_ms,
            ),
        )
        return outcome

    def find_outcome_by_dedup(self, dedup_key: str) -> OutcomeObservation | None:
        row = self._conn.execute(
            "SELECT * FROM outcome_observations WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_outcome(row) if row is not None else None

    def outcomes_for_experiment(self, experiment_id: str) -> list[OutcomeObservation]:
        rows = self._conn.execute(
            """
            SELECT * FROM outcome_observations
            WHERE experiment_id = ? ORDER BY observed_at_ms, rowid
            """,
            (experiment_id,),
        ).fetchall()
        return [self._row_to_outcome(row) for row in rows]

    def _verify_reference(self, conversation_id: str, evidence: ProposalEvidence) -> None:
        queries = {
            EvidenceReferenceKind.EVENT: (
                "SELECT 1 FROM events WHERE id = ? AND conversation_id = ?",
                (evidence.reference_id, conversation_id),
            ),
            EvidenceReferenceKind.TRACE: (
                "SELECT 1 FROM traces WHERE id = ? AND conversation_id = ?",
                (evidence.reference_id, conversation_id),
            ),
            EvidenceReferenceKind.OFFLINE_ARTIFACT: (
                """
                SELECT 1 FROM offline_artifacts AS artifact
                JOIN offline_episodes AS episode ON episode.id = artifact.episode_id
                WHERE artifact.id = ? AND episode.conversation_id = ?
                """,
                (evidence.reference_id, conversation_id),
            ),
            EvidenceReferenceKind.OUTCOME_OBSERVATION: (
                """
                SELECT 1 FROM outcome_observations AS outcome
                JOIN behavior_experiments AS experiment ON experiment.id = outcome.experiment_id
                WHERE outcome.id = ? AND experiment.conversation_id = ?
                """,
                (evidence.reference_id, conversation_id),
            ),
        }
        sql, params = queries[evidence.reference_kind]
        if self._conn.execute(sql, params).fetchone() is None:
            raise ValueError(
                f"unknown {evidence.reference_kind.value} evidence {evidence.reference_id!r}"
            )

    @contextmanager
    def _savepoint(self, name: str) -> Iterator[None]:
        self._conn.execute(f"SAVEPOINT {name}")
        try:
            yield
            self._conn.execute(f"RELEASE SAVEPOINT {name}")
        except Exception:
            self._conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            self._conn.execute(f"RELEASE SAVEPOINT {name}")
            raise

    @staticmethod
    def _run_values(run: ReflectionRun) -> tuple[object, ...]:
        return (
            run.id,
            run.conversation_id,
            run.correlation_id,
            run.dedup_key,
            run.trigger_kind.value,
            run.priority_score,
            _json(run.score_components),
            run.status.value,
            run.source_episode_id,
            run.source_artifact_id,
            _json(run.source_event_ids),
            _json(run.source_trace_ids),
            run.reflection_text,
            run.critique_text,
            run.critic_score,
            run.model,
            run.prompt_version,
            run.error,
            run.started_at_ms,
            run.completed_at_ms,
        )

    @staticmethod
    def _proposal_values(proposal: LearningProposal) -> tuple[object, ...]:
        return (
            proposal.id,
            proposal.run_id,
            proposal.conversation_id,
            proposal.dedup_key,
            proposal.proposal_type.value,
            proposal.status.value,
            proposal.title,
            proposal.content,
            _json(proposal.payload),
            proposal.rationale,
            proposal.confidence,
            proposal.critic_score,
            proposal.target_kind,
            proposal.target_id,
            _json(proposal.rollback),
            proposal.version,
            proposal.created_at_ms,
            proposal.updated_at_ms,
            proposal.applied_at_ms,
        )

    @staticmethod
    def _experiment_values(experiment: BehaviorExperiment) -> tuple[object, ...]:
        return (
            experiment.id,
            experiment.conversation_id,
            experiment.proposal_id,
            experiment.dedup_key,
            experiment.status.value,
            experiment.hypothesis,
            experiment.policy_key,
            _json(experiment.baseline),
            _json(experiment.treatment),
            _json(experiment.rollback),
            _json(experiment.success_criteria),
            experiment.min_observations,
            experiment.result_score,
            experiment.conclusion,
            experiment.started_at_ms,
            experiment.due_at_ms,
            experiment.completed_at_ms,
            experiment.created_at_ms,
            experiment.updated_at_ms,
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row | dict[str, Any]) -> ReflectionRun:
        data = dict(row)
        return ReflectionRun(
            id=data["id"],
            conversation_id=data["conversation_id"],
            correlation_id=data["correlation_id"],
            dedup_key=data["dedup_key"],
            trigger_kind=ReflectionTriggerKind(data["trigger_kind"]),
            priority_score=data["priority_score"],
            score_components=json.loads(data["score_components_json"] or "{}"),
            status=ReflectionRunStatus(data["status"]),
            source_episode_id=data["source_episode_id"],
            source_artifact_id=data["source_artifact_id"],
            source_event_ids=json.loads(data["source_event_ids_json"] or "[]"),
            source_trace_ids=json.loads(data["source_trace_ids_json"] or "[]"),
            reflection_text=data["reflection_text"],
            critique_text=data["critique_text"],
            critic_score=data["critic_score"],
            model=data["model"],
            prompt_version=data["prompt_version"],
            error=data["error"],
            started_at_ms=data["started_at_ms"],
            completed_at_ms=data["completed_at_ms"],
        )

    @staticmethod
    def _row_to_proposal(row: sqlite3.Row | dict[str, Any]) -> LearningProposal:
        data = dict(row)
        return LearningProposal(
            id=data["id"],
            run_id=data["run_id"],
            conversation_id=data["conversation_id"],
            dedup_key=data["dedup_key"],
            proposal_type=LearningProposalType(data["proposal_type"]),
            status=LearningProposalStatus(data["status"]),
            title=data["title"],
            content=data["content"],
            payload=json.loads(data["payload_json"] or "{}"),
            rationale=data["rationale"],
            confidence=data["confidence"],
            critic_score=data["critic_score"],
            target_kind=data["target_kind"],
            target_id=data["target_id"],
            rollback=json.loads(data["rollback_json"] or "{}"),
            version=data["version"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
            applied_at_ms=data["applied_at_ms"],
        )

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row | dict[str, Any]) -> ProposalEvidence:
        data = dict(row)
        return ProposalEvidence(
            proposal_id=data["proposal_id"],
            reference_kind=EvidenceReferenceKind(data["reference_kind"]),
            reference_id=data["reference_id"],
            relation=EvidenceRelation(data["relation"]),
            provenance_kind=SourceKind(data["provenance_kind"]),
            trust_weight=data["trust_weight"],
            likelihood_ratio=data["likelihood_ratio"],
            content_hash=data["content_hash"],
            excerpt=data["excerpt"],
            recorded_at_ms=data["recorded_at_ms"],
        )

    @staticmethod
    def _row_to_decision(row: sqlite3.Row | dict[str, Any]) -> ConsolidationDecision:
        data = dict(row)
        return ConsolidationDecision(
            id=data["id"],
            proposal_id=data["proposal_id"],
            decision_version=data["decision_version"],
            decision=ConsolidationDecisionKind(data["decision"]),
            gate_score=data["gate_score"],
            criteria=json.loads(data["criteria_json"] or "{}"),
            reason=data["reason"],
            rules_version=data["rules_version"],
            model=data["model"],
            prompt_version=data["prompt_version"],
            event_id=data["event_id"],
            created_at_ms=data["created_at_ms"],
        )

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row | dict[str, Any]) -> BehaviorExperiment:
        data = dict(row)
        return BehaviorExperiment(
            id=data["id"],
            conversation_id=data["conversation_id"],
            proposal_id=data["proposal_id"],
            dedup_key=data["dedup_key"],
            status=ExperimentStatus(data["status"]),
            hypothesis=data["hypothesis"],
            policy_key=data["policy_key"],
            baseline=json.loads(data["baseline_json"] or "{}"),
            treatment=json.loads(data["treatment_json"] or "{}"),
            rollback=json.loads(data["rollback_json"] or "{}"),
            success_criteria=json.loads(data["success_criteria_json"] or "{}"),
            min_observations=data["min_observations"],
            result_score=data["result_score"],
            conclusion=data["conclusion"],
            started_at_ms=data["started_at_ms"],
            due_at_ms=data["due_at_ms"],
            completed_at_ms=data["completed_at_ms"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )

    @staticmethod
    def _row_to_outcome(row: sqlite3.Row | dict[str, Any]) -> OutcomeObservation:
        data = dict(row)
        return OutcomeObservation(
            id=data["id"],
            experiment_id=data["experiment_id"],
            dedup_key=data["dedup_key"],
            source_event_id=data["source_event_id"],
            observation_kind=OutcomeVerdict(data["observation_kind"]),
            payload=json.loads(data["payload_json"] or "{}"),
            normalized_score=data["normalized_score"],
            confidence=data["confidence"],
            prediction_error=data["prediction_error"],
            observed_at_ms=data["observed_at_ms"],
            created_at_ms=data["created_at_ms"],
        )

    @staticmethod
    def _aliased_episode(row: sqlite3.Row | dict[str, Any]) -> OfflineEpisode:
        data = dict(row)
        from ssa.domain.lifecycle import OfflineActionKind

        return OfflineEpisode(
            id=data["ep_id"],
            conversation_id=data["ep_conversation_id"],
            correlation_id=data["ep_correlation_id"],
            dedup_key=data["ep_dedup_key"],
            action_kind=OfflineActionKind(data["ep_action_kind"]),
            motive=data["ep_motive"],
            status=OfflineEpisodeStatus(data["ep_status"]),
            goal_id=data["ep_goal_id"],
            source_event_ids=json.loads(data["ep_source_event_ids_json"] or "[]"),
            source_trace_ids=json.loads(data["ep_source_trace_ids_json"] or "[]"),
            provider=data["ep_provider"],
            tool_name=data["ep_tool_name"],
            tool_output_hash=data["ep_tool_output_hash"],
            summary=data["ep_summary"],
            error=data["ep_error"],
            event_id=data["ep_event_id"],
            started_at_ms=data["ep_started_at_ms"],
            completed_at_ms=data["ep_completed_at_ms"],
        )

    @staticmethod
    def _aliased_artifact(row: sqlite3.Row | dict[str, Any]) -> OfflineArtifact:
        data = dict(row)
        return OfflineArtifact(
            id=data["ar_id"],
            episode_id=data["ar_episode_id"],
            artifact_type=data["ar_artifact_type"],
            title=data["ar_title"],
            content=data["ar_content"],
            evidence_event_ids=json.loads(data["ar_evidence_event_ids_json"] or "[]"),
            evidence_trace_ids=json.loads(data["ar_evidence_trace_ids_json"] or "[]"),
            metadata=json.loads(data["ar_metadata_json"] or "{}"),
            created_at_ms=data["ar_created_at_ms"],
        )


__all__ = ["LearningVersionConflict", "SqliteLearningRepository"]
