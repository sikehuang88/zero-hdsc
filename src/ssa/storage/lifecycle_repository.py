"""SQLite repositories for the persistent autonomous lifecycle kernel."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ssa.domain.lifecycle import (
    CausalTraceRecord,
    ContactEpisode,
    ContactPhase,
    ContactStatus,
    EmotionEpisode,
    EmotionStatus,
    EmotionType,
    Goal,
    GoalOwner,
    GoalStatus,
    GoalStep,
    Initiative,
    InitiativeStatus,
    InnerLifeMode,
    InnerLoopState,
    JobStatus,
    OfflineActionKind,
    OfflineArtifact,
    OfflineEpisode,
    OfflineEpisodeStatus,
    OutboxMessage,
    OutboxStatus,
    ScheduledJob,
    WorldObservation,
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return dict(row)


class GoalRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, goal: Goal) -> Goal:
        self._conn.execute(
            """
            INSERT INTO goals (
                id, owner, title, motive, success_criteria, priority, progress,
                status, due_at_ms, next_action_at_ms, created_at_ms, updated_at_ms,
                conversation_id, correlation_id, blocked_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goal.id,
                goal.owner.value,
                goal.title,
                goal.motive,
                _json(goal.success_criteria),
                goal.priority,
                goal.progress,
                goal.status.value,
                goal.due_at_ms,
                goal.next_action_at_ms,
                goal.created_at_ms,
                goal.updated_at_ms,
                goal.conversation_id,
                goal.correlation_id,
                goal.blocked_reason,
            ),
        )
        return goal

    def get(self, goal_id: str) -> Goal | None:
        row = self._conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
        return self._row_to_goal(row) if row is not None else None

    def list_active(
        self,
        conversation_id: str,
        *,
        owner: GoalOwner | None = None,
    ) -> list[Goal]:
        params: list[object] = [conversation_id, GoalStatus.ACTIVE.value]
        owner_clause = ""
        if owner is not None:
            owner_clause = " AND owner = ?"
            params.append(owner.value)
        rows = self._conn.execute(
            f"""
            SELECT * FROM goals
            WHERE conversation_id = ? AND status = ?{owner_clause}
            ORDER BY priority DESC, created_at_ms
            """,
            params,
        ).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def list_open(
        self,
        conversation_id: str,
        *,
        owner: GoalOwner | None = None,
    ) -> list[Goal]:
        params: list[object] = [conversation_id]
        owner_clause = ""
        if owner is not None:
            owner_clause = " AND owner = ?"
            params.append(owner.value)
        rows = self._conn.execute(
            f"""
            SELECT * FROM goals
            WHERE conversation_id = ? AND status IN ('proposed', 'active', 'blocked')
              {owner_clause}
            ORDER BY priority DESC, created_at_ms
            """,
            params,
        ).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def list_due(self, conversation_id: str, now_ms: int) -> list[Goal]:
        rows = self._conn.execute(
            """
            SELECT * FROM goals
            WHERE conversation_id = ? AND status IN ('active', 'blocked')
              AND next_action_at_ms IS NOT NULL AND next_action_at_ms <= ?
            ORDER BY priority DESC, next_action_at_ms
            """,
            (conversation_id, now_ms),
        ).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def update(self, goal: Goal) -> Goal:
        cursor = self._conn.execute(
            """
            UPDATE goals SET
                owner = ?, title = ?, motive = ?, success_criteria = ?,
                priority = ?, progress = ?, status = ?, due_at_ms = ?,
                next_action_at_ms = ?, updated_at_ms = ?, correlation_id = ?,
                blocked_reason = ?
            WHERE id = ? AND conversation_id = ?
            """,
            (
                goal.owner.value,
                goal.title,
                goal.motive,
                _json(goal.success_criteria),
                goal.priority,
                goal.progress,
                goal.status.value,
                goal.due_at_ms,
                goal.next_action_at_ms,
                goal.updated_at_ms,
                goal.correlation_id,
                goal.blocked_reason,
                goal.id,
                goal.conversation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown goal {goal.id!r}")
        return goal

    def add_step(self, step: GoalStep) -> GoalStep:
        self._conn.execute(
            """
            INSERT INTO goal_steps (
                id, goal_id, action, result, source_event_id, created_at_ms, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.id,
                step.goal_id,
                step.action,
                step.result,
                step.source_event_id,
                step.created_at_ms,
                _json(step.metadata),
            ),
        )
        return step

    def steps(self, goal_id: str) -> list[GoalStep]:
        rows = self._conn.execute(
            "SELECT * FROM goal_steps WHERE goal_id = ? ORDER BY created_at_ms, rowid",
            (goal_id,),
        ).fetchall()
        return [
            GoalStep(
                id=row["id"],
                goal_id=row["goal_id"],
                action=row["action"],
                result=row["result"],
                source_event_id=row["source_event_id"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at_ms=row["created_at_ms"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_goal(row: sqlite3.Row | dict[str, Any]) -> Goal:
        data = _dict(row)
        return Goal(
            id=data["id"],
            conversation_id=data["conversation_id"],
            owner=GoalOwner(data["owner"]),
            title=data["title"],
            motive=data["motive"],
            success_criteria=json.loads(data["success_criteria"]),
            priority=data["priority"],
            progress=data["progress"],
            status=GoalStatus(data["status"]),
            due_at_ms=data["due_at_ms"],
            next_action_at_ms=data["next_action_at_ms"],
            correlation_id=data["correlation_id"],
            blocked_reason=data["blocked_reason"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )


class ScheduledJobRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def enqueue(self, job: ScheduledJob) -> ScheduledJob:
        if job.dedup_key is not None:
            existing = self.find_by_dedup(job.dedup_key)
            if existing is not None:
                return existing
        self._conn.execute(
            """
            INSERT INTO scheduled_jobs (
                id, job_type, dedup_key, payload_json, due_at_ms, attempt_count,
                status, last_error, leased_until_ms, completed_at_ms, created_at_ms,
                conversation_id, max_attempts, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.job_type,
                job.dedup_key,
                _json(job.payload),
                job.due_at_ms,
                job.attempt_count,
                job.status.value,
                job.last_error,
                job.leased_until_ms,
                job.completed_at_ms,
                job.created_at_ms,
                job.conversation_id,
                job.max_attempts,
                job.updated_at_ms,
            ),
        )
        return job

    def get(self, job_id: str) -> ScheduledJob | None:
        row = self._conn.execute("SELECT * FROM scheduled_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    def find_by_dedup(self, dedup_key: str) -> ScheduledJob | None:
        row = self._conn.execute(
            "SELECT * FROM scheduled_jobs WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def revive_missing_handler_dead_letter(
        self,
        job_id: str,
        due_at_ms: int,
        now_ms: int,
    ) -> ScheduledJob | None:
        """Revive a recurring job that died only because its handler was absent."""
        self._conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'pending', due_at_ms = ?, attempt_count = 0,
                last_error = NULL, leased_until_ms = NULL, completed_at_ms = NULL,
                updated_at_ms = ?
            WHERE id = ? AND status = 'dead_letter'
              AND last_error LIKE 'LookupError: no lifecycle handler registered%'
            """,
            (due_at_ms, now_ms, job_id),
        )
        return self.get(job_id)

    def expedite(self, dedup_key: str, due_at_ms: int, now_ms: int) -> ScheduledJob | None:
        """Move a pending recurring job earlier without creating a duplicate row."""
        self._conn.execute(
            """
            UPDATE scheduled_jobs
            SET due_at_ms = MIN(due_at_ms, ?), updated_at_ms = ?
            WHERE dedup_key = ? AND status = 'pending'
            """,
            (due_at_ms, now_ms, dedup_key),
        )
        return self.find_by_dedup(dedup_key)

    def lease_due(
        self,
        now_ms: int,
        *,
        lease_ms: int = 60_000,
        limit: int = 8,
    ) -> list[ScheduledJob]:
        rows = self._conn.execute(
            """
            SELECT id FROM scheduled_jobs
            WHERE (
                (status = 'pending' AND due_at_ms <= ?)
                OR (status = 'running' AND leased_until_ms <= ?)
            ) AND attempt_count < max_attempts
            ORDER BY due_at_ms, created_at_ms
            LIMIT ?
            """,
            (now_ms, now_ms, limit),
        ).fetchall()
        leased: list[ScheduledJob] = []
        for row in rows:
            cursor = self._conn.execute(
                """
                UPDATE scheduled_jobs
                SET status = 'running', leased_until_ms = ?,
                    attempt_count = attempt_count + 1, updated_at_ms = ?
                WHERE id = ? AND (
                    (status = 'pending' AND due_at_ms <= ?)
                    OR (status = 'running' AND leased_until_ms <= ?)
                )
                """,
                (now_ms + lease_ms, now_ms, row["id"], now_ms, now_ms),
            )
            if cursor.rowcount == 1:
                current = self.get(str(row["id"]))
                if current is not None:
                    leased.append(current)
        return leased

    def reschedule(self, job_id: str, due_at_ms: int, now_ms: int) -> ScheduledJob:
        self._conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'pending', due_at_ms = ?, leased_until_ms = NULL,
                attempt_count = 0, last_error = NULL, completed_at_ms = NULL,
                updated_at_ms = ?
            WHERE id = ?
            """,
            (due_at_ms, now_ms, job_id),
        )
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id!r}")
        return job

    def complete(self, job_id: str, now_ms: int) -> ScheduledJob:
        self._conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = 'done', completed_at_ms = ?, leased_until_ms = NULL,
                updated_at_ms = ?
            WHERE id = ?
            """,
            (now_ms, now_ms, job_id),
        )
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id!r}")
        return job

    def fail(self, job_id: str, error: str, now_ms: int, retry_at_ms: int) -> ScheduledJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id!r}")
        terminal = job.attempt_count >= job.max_attempts
        self._conn.execute(
            """
            UPDATE scheduled_jobs
            SET status = ?, last_error = ?, due_at_ms = ?, leased_until_ms = NULL,
                updated_at_ms = ?
            WHERE id = ?
            """,
            (
                JobStatus.DEAD_LETTER.value if terminal else JobStatus.PENDING.value,
                error,
                retry_at_ms,
                now_ms,
                job_id,
            ),
        )
        failed = self.get(job_id)
        assert failed is not None
        return failed

    @staticmethod
    def _row_to_job(row: sqlite3.Row | dict[str, Any]) -> ScheduledJob:
        data = _dict(row)
        return ScheduledJob(
            id=data["id"],
            conversation_id=data["conversation_id"],
            job_type=data["job_type"],
            dedup_key=data["dedup_key"],
            payload=json.loads(data["payload_json"] or "{}"),
            due_at_ms=data["due_at_ms"],
            attempt_count=data["attempt_count"],
            max_attempts=data["max_attempts"],
            status=JobStatus(data["status"]),
            last_error=data["last_error"],
            leased_until_ms=data["leased_until_ms"],
            completed_at_ms=data["completed_at_ms"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )


class InitiativeRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, initiative: Initiative) -> Initiative:
        if initiative.dedup_key is not None:
            existing = self.find_by_dedup(initiative.dedup_key)
            if existing is not None:
                return existing
        self._conn.execute(
            """
            INSERT INTO initiatives (
                id, motive, intent, content_draft, urgency, status,
                earliest_send_at_ms, expires_at_ms, sent_event_id, created_at_ms,
                updated_at_ms, conversation_id, correlation_id, goal_id,
                source_event_ids_json, source_trace_ids_json, decision_json,
                decision_score, channel, dedup_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                initiative.id,
                initiative.motive,
                initiative.intent,
                initiative.content_draft,
                initiative.urgency,
                initiative.status.value,
                initiative.earliest_send_at_ms,
                initiative.expires_at_ms,
                initiative.sent_event_id,
                initiative.created_at_ms,
                initiative.updated_at_ms,
                initiative.conversation_id,
                initiative.correlation_id,
                initiative.goal_id,
                _json(initiative.source_event_ids),
                _json(initiative.source_trace_ids),
                _json(initiative.decision),
                initiative.decision_score,
                initiative.channel,
                initiative.dedup_key,
            ),
        )
        return initiative

    def get(self, initiative_id: str) -> Initiative | None:
        row = self._conn.execute(
            "SELECT * FROM initiatives WHERE id = ?", (initiative_id,)
        ).fetchone()
        return self._row_to_initiative(row) if row is not None else None

    def find_by_dedup(self, dedup_key: str) -> Initiative | None:
        row = self._conn.execute(
            "SELECT * FROM initiatives WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_initiative(row) if row is not None else None

    def list_ready(self, conversation_id: str, now_ms: int) -> list[Initiative]:
        rows = self._conn.execute(
            """
            SELECT * FROM initiatives
            WHERE conversation_id = ? AND status IN ('approved', 'queued')
              AND earliest_send_at_ms <= ?
              AND (expires_at_ms IS NULL OR expires_at_ms > ?)
            ORDER BY urgency DESC, earliest_send_at_ms
            """,
            (conversation_id, now_ms, now_ms),
        ).fetchall()
        return [self._row_to_initiative(row) for row in rows]

    def list_active(self, conversation_id: str) -> list[Initiative]:
        rows = self._conn.execute(
            """
            SELECT * FROM initiatives
            WHERE conversation_id = ?
              AND status IN ('candidate', 'approved', 'queued')
            ORDER BY created_at_ms
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_initiative(row) for row in rows]

    def count_sent_since(self, conversation_id: str, since_ms: int) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS c FROM initiatives
            WHERE conversation_id = ? AND status = 'sent' AND updated_at_ms >= ?
            """,
            (conversation_id, since_ms),
        ).fetchone()
        return int(row["c"]) if row is not None else 0

    def latest_sent(self, conversation_id: str) -> Initiative | None:
        row = self._conn.execute(
            """
            SELECT * FROM initiatives
            WHERE conversation_id = ? AND status = 'sent'
            ORDER BY updated_at_ms DESC LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        return self._row_to_initiative(row) if row is not None else None

    def update_status(
        self,
        initiative_id: str,
        status: InitiativeStatus,
        now_ms: int,
        *,
        sent_event_id: str | None = None,
    ) -> Initiative:
        self._conn.execute(
            """
            UPDATE initiatives
            SET status = ?, updated_at_ms = ?,
                sent_event_id = COALESCE(?, sent_event_id)
            WHERE id = ?
            """,
            (status.value, now_ms, sent_event_id, initiative_id),
        )
        initiative = self.get(initiative_id)
        if initiative is None:
            raise KeyError(f"unknown initiative {initiative_id!r}")
        return initiative

    def expire_due(self, conversation_id: str, now_ms: int) -> int:
        cursor = self._conn.execute(
            """
            UPDATE initiatives
            SET status = 'expired', updated_at_ms = ?
            WHERE conversation_id = ? AND status IN ('candidate', 'approved', 'queued')
              AND expires_at_ms IS NOT NULL AND expires_at_ms <= ?
            """,
            (now_ms, conversation_id, now_ms),
        )
        return max(0, cursor.rowcount)

    @staticmethod
    def _row_to_initiative(row: sqlite3.Row | dict[str, Any]) -> Initiative:
        data = _dict(row)
        return Initiative(
            id=data["id"],
            conversation_id=data["conversation_id"],
            motive=data["motive"],
            intent=data["intent"],
            content_draft=data["content_draft"],
            urgency=data["urgency"],
            decision_score=data["decision_score"],
            status=InitiativeStatus(data["status"]),
            earliest_send_at_ms=data["earliest_send_at_ms"],
            expires_at_ms=data["expires_at_ms"],
            correlation_id=data["correlation_id"],
            goal_id=data["goal_id"],
            source_event_ids=json.loads(data["source_event_ids_json"] or "[]"),
            source_trace_ids=json.loads(data["source_trace_ids_json"] or "[]"),
            decision=json.loads(data["decision_json"] or "{}"),
            channel=data["channel"],
            dedup_key=data["dedup_key"],
            sent_event_id=data["sent_event_id"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )


class OutboxRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def enqueue(self, message: OutboxMessage) -> OutboxMessage:
        if message.dedup_key is not None:
            existing = self.find_by_dedup(message.dedup_key)
            if existing is not None:
                return existing
        self._conn.execute(
            """
            INSERT INTO outbox (
                id, correlation_id, channel, recipient, payload_json, status,
                attempt_count, next_attempt_at_ms, delivered_at_ms, created_at_ms,
                initiative_id, dedup_key, delivered_event_id, last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.correlation_id,
                message.channel,
                message.recipient,
                _json(message.payload),
                message.status.value,
                message.attempt_count,
                message.next_attempt_at_ms,
                message.delivered_at_ms,
                message.created_at_ms,
                message.initiative_id,
                message.dedup_key,
                message.delivered_event_id,
                message.last_error,
            ),
        )
        return message

    def get(self, message_id: str) -> OutboxMessage | None:
        row = self._conn.execute("SELECT * FROM outbox WHERE id = ?", (message_id,)).fetchone()
        return self._row_to_message(row) if row is not None else None

    def find_by_dedup(self, dedup_key: str) -> OutboxMessage | None:
        row = self._conn.execute(
            "SELECT * FROM outbox WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_message(row) if row is not None else None

    def claim_due(
        self,
        now_ms: int,
        *,
        lease_ms: int = 60_000,
        limit: int = 8,
    ) -> list[OutboxMessage]:
        if lease_ms < 1:
            raise ValueError("lease_ms must be positive")
        rows = self._conn.execute(
            """
            SELECT id FROM outbox
            WHERE (
                (status = 'pending' AND (next_attempt_at_ms IS NULL OR next_attempt_at_ms <= ?))
                OR
                (status = 'delivering' AND (next_attempt_at_ms IS NULL OR next_attempt_at_ms <= ?))
            )
            ORDER BY created_at_ms LIMIT ?
            """,
            (now_ms, now_ms, limit),
        ).fetchall()
        messages: list[OutboxMessage] = []
        for row in rows:
            cursor = self._conn.execute(
                """
                UPDATE outbox
                SET status = 'delivering', attempt_count = attempt_count + 1,
                    next_attempt_at_ms = ?
                WHERE id = ? AND (
                    (status = 'pending' AND (next_attempt_at_ms IS NULL OR next_attempt_at_ms <= ?))
                    OR
                    (status = 'delivering' AND (next_attempt_at_ms IS NULL OR next_attempt_at_ms <= ?))
                )
                """,
                (now_ms + lease_ms, row["id"], now_ms, now_ms),
            )
            if cursor.rowcount == 1:
                message = self.get(str(row["id"]))
                if message is not None:
                    messages.append(message)
        return messages

    def mark_delivered(
        self,
        message_id: str,
        now_ms: int,
        delivered_event_id: str | None,
    ) -> OutboxMessage:
        self._conn.execute(
            """
            UPDATE outbox
            SET status = 'delivered', delivered_at_ms = ?, delivered_event_id = ?,
                last_error = NULL
            WHERE id = ?
            """,
            (now_ms, delivered_event_id, message_id),
        )
        message = self.get(message_id)
        if message is None:
            raise KeyError(f"unknown outbox message {message_id!r}")
        return message

    def fail(
        self,
        message_id: str,
        error: str,
        retry_at_ms: int,
        *,
        max_attempts: int = 5,
    ) -> OutboxMessage:
        message = self.get(message_id)
        if message is None:
            raise KeyError(f"unknown outbox message {message_id!r}")
        status = (
            OutboxStatus.DEAD_LETTER
            if message.attempt_count >= max_attempts
            else OutboxStatus.PENDING
        )
        self._conn.execute(
            """
            UPDATE outbox
            SET status = ?, last_error = ?, next_attempt_at_ms = ?
            WHERE id = ?
            """,
            (status.value, error, retry_at_ms, message_id),
        )
        failed = self.get(message_id)
        assert failed is not None
        return failed

    @staticmethod
    def _row_to_message(row: sqlite3.Row | dict[str, Any]) -> OutboxMessage:
        data = _dict(row)
        return OutboxMessage(
            id=data["id"],
            correlation_id=data["correlation_id"],
            channel=data["channel"],
            recipient=data["recipient"],
            payload=json.loads(data["payload_json"] or "{}"),
            status=OutboxStatus(data["status"]),
            attempt_count=data["attempt_count"],
            next_attempt_at_ms=data["next_attempt_at_ms"],
            delivered_at_ms=data["delivered_at_ms"],
            created_at_ms=data["created_at_ms"],
            initiative_id=data["initiative_id"],
            dedup_key=data["dedup_key"],
            delivered_event_id=data["delivered_event_id"],
            last_error=data["last_error"],
        )


class EmotionEpisodeRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, episode: EmotionEpisode) -> EmotionEpisode:
        self._conn.execute(
            """
            INSERT INTO emotion_episodes (
                id, conversation_id, correlation_id, emotion_type, target,
                action_tendency, intensity, inhibition, half_life_minutes,
                source_event_ids_json, source_trace_ids_json, resolution_condition,
                status, created_at_ms, updated_at_ms, resolved_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.conversation_id,
                episode.correlation_id,
                episode.emotion_type.value,
                episode.target,
                episode.action_tendency,
                episode.intensity,
                episode.inhibition,
                episode.half_life_minutes,
                _json(episode.source_event_ids),
                _json(episode.source_trace_ids),
                episode.resolution_condition,
                episode.status.value,
                episode.created_at_ms,
                episode.updated_at_ms,
                episode.resolved_at_ms,
            ),
        )
        return episode

    def active(self, conversation_id: str) -> list[EmotionEpisode]:
        rows = self._conn.execute(
            """
            SELECT * FROM emotion_episodes
            WHERE conversation_id = ? AND status = 'active'
            ORDER BY intensity DESC, updated_at_ms DESC
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def resolve_all(self, conversation_id: str, now_ms: int) -> int:
        cursor = self._conn.execute(
            """
            UPDATE emotion_episodes
            SET status = 'resolved', resolved_at_ms = ?, updated_at_ms = ?
            WHERE conversation_id = ? AND status = 'active'
            """,
            (now_ms, now_ms, conversation_id),
        )
        return max(0, cursor.rowcount)

    def expire_decayed(
        self,
        conversation_id: str,
        now_ms: int,
        *,
        threshold: float = 0.05,
    ) -> int:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        expired_ids: list[str] = []
        for episode in self.active(conversation_id):
            age_minutes = max(0.0, (now_ms - episode.updated_at_ms) / 60_000.0)
            effective = episode.intensity * (0.5 ** (age_minutes / episode.half_life_minutes))
            if effective < threshold:
                expired_ids.append(episode.id)
        if not expired_ids:
            return 0
        placeholders = ", ".join("?" for _ in expired_ids)
        cursor = self._conn.execute(
            f"""
            UPDATE emotion_episodes
            SET status = 'expired', resolved_at_ms = ?, updated_at_ms = ?
            WHERE id IN ({placeholders}) AND status = 'active'
            """,
            (now_ms, now_ms, *expired_ids),
        )
        return max(0, cursor.rowcount)

    @staticmethod
    def _row_to_episode(row: sqlite3.Row | dict[str, Any]) -> EmotionEpisode:
        data = _dict(row)
        return EmotionEpisode(
            id=data["id"],
            conversation_id=data["conversation_id"],
            correlation_id=data["correlation_id"],
            emotion_type=EmotionType(data["emotion_type"]),
            target=data["target"],
            action_tendency=data["action_tendency"],
            intensity=data["intensity"],
            inhibition=data["inhibition"],
            half_life_minutes=data["half_life_minutes"],
            source_event_ids=json.loads(data["source_event_ids_json"] or "[]"),
            source_trace_ids=json.loads(data["source_trace_ids_json"] or "[]"),
            resolution_condition=data["resolution_condition"],
            status=EmotionStatus(data["status"]),
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
            resolved_at_ms=data["resolved_at_ms"],
        )


class ContactEpisodeRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, episode: ContactEpisode) -> ContactEpisode:
        self._conn.execute(
            """
            INSERT INTO contact_episodes (
                id, conversation_id, correlation_id, motive, topic, phase, status,
                message_count, max_messages, hypothesis_json, source_initiative_id,
                last_sent_at_ms, next_action_at_ms, resolved_by_event_id,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.conversation_id,
                episode.correlation_id,
                episode.motive,
                episode.topic,
                episode.phase.value,
                episode.status.value,
                episode.message_count,
                episode.max_messages,
                _json(episode.hypotheses),
                episode.source_initiative_id,
                episode.last_sent_at_ms,
                episode.next_action_at_ms,
                episode.resolved_by_event_id,
                episode.created_at_ms,
                episode.updated_at_ms,
            ),
        )
        return episode

    def get(self, episode_id: str) -> ContactEpisode | None:
        row = self._conn.execute(
            "SELECT * FROM contact_episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def find_by_source_initiative(self, initiative_id: str) -> ContactEpisode | None:
        row = self._conn.execute(
            """
            SELECT * FROM contact_episodes
            WHERE source_initiative_id = ?
            ORDER BY created_at_ms DESC LIMIT 1
            """,
            (initiative_id,),
        ).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def active(self, conversation_id: str) -> list[ContactEpisode]:
        rows = self._conn.execute(
            """
            SELECT * FROM contact_episodes
            WHERE conversation_id = ? AND status = 'active'
            ORDER BY created_at_ms
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def due(self, conversation_id: str, now_ms: int) -> list[ContactEpisode]:
        rows = self._conn.execute(
            """
            SELECT * FROM contact_episodes
            WHERE conversation_id = ? AND status = 'active'
              AND next_action_at_ms IS NOT NULL AND next_action_at_ms <= ?
            ORDER BY next_action_at_ms
            """,
            (conversation_id, now_ms),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def update(self, episode: ContactEpisode) -> ContactEpisode:
        cursor = self._conn.execute(
            """
            UPDATE contact_episodes SET
                motive = ?, topic = ?, phase = ?, status = ?, message_count = ?,
                max_messages = ?, hypothesis_json = ?, last_sent_at_ms = ?,
                next_action_at_ms = ?, resolved_by_event_id = ?, updated_at_ms = ?
            WHERE id = ? AND conversation_id = ?
            """,
            (
                episode.motive,
                episode.topic,
                episode.phase.value,
                episode.status.value,
                episode.message_count,
                episode.max_messages,
                _json(episode.hypotheses),
                episode.last_sent_at_ms,
                episode.next_action_at_ms,
                episode.resolved_by_event_id,
                episode.updated_at_ms,
                episode.id,
                episode.conversation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown contact episode {episode.id!r}")
        return episode

    @staticmethod
    def _row_to_episode(row: sqlite3.Row | dict[str, Any]) -> ContactEpisode:
        data = _dict(row)
        return ContactEpisode(
            id=data["id"],
            conversation_id=data["conversation_id"],
            correlation_id=data["correlation_id"],
            motive=data["motive"],
            topic=data["topic"],
            phase=ContactPhase(data["phase"]),
            status=ContactStatus(data["status"]),
            message_count=data["message_count"],
            max_messages=data["max_messages"],
            hypotheses=json.loads(data["hypothesis_json"] or "{}"),
            source_initiative_id=data["source_initiative_id"],
            last_sent_at_ms=data["last_sent_at_ms"],
            next_action_at_ms=data["next_action_at_ms"],
            resolved_by_event_id=data["resolved_by_event_id"],
            created_at_ms=data["created_at_ms"],
            updated_at_ms=data["updated_at_ms"],
        )


class WorldObservationRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, observation: WorldObservation) -> WorldObservation:
        existing = self.find_by_dedup(observation.dedup_key)
        if existing is not None:
            return existing
        self._conn.execute(
            """
            INSERT INTO world_observations (
                id, conversation_id, observation_type, source, summary,
                payload_json, salience, dedup_key, observed_at_ms, event_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.id,
                observation.conversation_id,
                observation.observation_type,
                observation.source,
                observation.summary,
                _json(observation.payload),
                observation.salience,
                observation.dedup_key,
                observation.observed_at_ms,
                observation.event_id,
            ),
        )
        return observation

    def find_by_dedup(self, dedup_key: str) -> WorldObservation | None:
        row = self._conn.execute(
            "SELECT * FROM world_observations WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_observation(row) if row is not None else None

    def recent(self, conversation_id: str, *, limit: int = 20) -> list[WorldObservation]:
        rows = self._conn.execute(
            """
            SELECT * FROM world_observations
            WHERE conversation_id = ?
            ORDER BY observed_at_ms DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    @staticmethod
    def _row_to_observation(row: sqlite3.Row | dict[str, Any]) -> WorldObservation:
        data = _dict(row)
        return WorldObservation(
            id=data["id"],
            conversation_id=data["conversation_id"],
            observation_type=data["observation_type"],
            source=data["source"],
            summary=data["summary"],
            payload=json.loads(data["payload_json"] or "{}"),
            salience=data["salience"],
            dedup_key=data["dedup_key"],
            observed_at_ms=data["observed_at_ms"],
            event_id=data["event_id"],
        )


class CausalTraceRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, trace: CausalTraceRecord) -> CausalTraceRecord:
        self._conn.execute(
            """
            INSERT INTO causal_traces (
                id, correlation_id, input_event_id, appraisal_id, old_state_id,
                new_state_id, old_relationship_id, new_relationship_id,
                memory_ids_json, goal_ids_json, decision_json, output_event_id,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                trace.correlation_id,
                trace.input_event_id,
                trace.appraisal_id,
                trace.old_state_id,
                trace.new_state_id,
                trace.old_relationship_id,
                trace.new_relationship_id,
                _json(trace.memory_ids),
                _json(trace.goal_ids),
                _json(trace.decision),
                trace.output_event_id,
                trace.created_at_ms,
            ),
        )
        return trace


class OfflineAgencyRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_episode(self, episode: OfflineEpisode) -> OfflineEpisode:
        existing = self.find_by_dedup(episode.dedup_key)
        if existing is not None:
            return existing
        self._conn.execute(
            """
            INSERT INTO offline_episodes (
                id, conversation_id, correlation_id, dedup_key, action_kind,
                motive, status, goal_id, source_event_ids_json,
                source_trace_ids_json, provider, tool_name, tool_output_hash,
                summary, error, event_id, started_at_ms, completed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.conversation_id,
                episode.correlation_id,
                episode.dedup_key,
                episode.action_kind.value,
                episode.motive,
                episode.status.value,
                episode.goal_id,
                _json(episode.source_event_ids),
                _json(episode.source_trace_ids),
                episode.provider,
                episode.tool_name,
                episode.tool_output_hash,
                episode.summary,
                episode.error,
                episode.event_id,
                episode.started_at_ms,
                episode.completed_at_ms,
            ),
        )
        return episode

    def get_episode(self, episode_id: str) -> OfflineEpisode | None:
        row = self._conn.execute(
            "SELECT * FROM offline_episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def find_by_dedup(self, dedup_key: str) -> OfflineEpisode | None:
        row = self._conn.execute(
            "SELECT * FROM offline_episodes WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def update_episode(self, episode: OfflineEpisode) -> OfflineEpisode:
        cursor = self._conn.execute(
            """
            UPDATE offline_episodes SET
                action_kind = ?, motive = ?, status = ?, goal_id = ?,
                source_event_ids_json = ?, source_trace_ids_json = ?,
                provider = ?, tool_name = ?, tool_output_hash = ?, summary = ?,
                error = ?, event_id = ?, completed_at_ms = ?
            WHERE id = ? AND conversation_id = ?
            """,
            (
                episode.action_kind.value,
                episode.motive,
                episode.status.value,
                episode.goal_id,
                _json(episode.source_event_ids),
                _json(episode.source_trace_ids),
                episode.provider,
                episode.tool_name,
                episode.tool_output_hash,
                episode.summary,
                episode.error,
                episode.event_id,
                episode.completed_at_ms,
                episode.id,
                episode.conversation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown offline episode {episode.id!r}")
        return episode

    def recent_episodes(self, conversation_id: str, *, limit: int = 20) -> list[OfflineEpisode]:
        rows = self._conn.execute(
            """
            SELECT * FROM offline_episodes
            WHERE conversation_id = ?
            ORDER BY started_at_ms DESC, rowid DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def count_since(self, conversation_id: str, since_ms: int) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS c FROM offline_episodes
            WHERE conversation_id = ? AND started_at_ms >= ?
            """,
            (conversation_id, since_ms),
        ).fetchone()
        return int(row["c"]) if row is not None else 0

    def used_trace_ids(self, conversation_id: str, *, limit: int = 200) -> set[str]:
        rows = self._conn.execute(
            """
            SELECT source_trace_ids_json FROM offline_episodes
            WHERE conversation_id = ? AND status = 'completed'
            ORDER BY started_at_ms DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return {
            str(trace_id)
            for row in rows
            for trace_id in json.loads(row["source_trace_ids_json"] or "[]")
        }

    def insert_artifact(self, artifact: OfflineArtifact) -> OfflineArtifact:
        self._conn.execute(
            """
            INSERT INTO offline_artifacts (
                id, episode_id, artifact_type, title, content,
                evidence_event_ids_json, evidence_trace_ids_json,
                metadata_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.episode_id,
                artifact.artifact_type,
                artifact.title,
                artifact.content,
                _json(artifact.evidence_event_ids),
                _json(artifact.evidence_trace_ids),
                _json(artifact.metadata),
                artifact.created_at_ms,
            ),
        )
        return artifact

    def artifacts_for_episode(self, episode_id: str) -> list[OfflineArtifact]:
        rows = self._conn.execute(
            """
            SELECT * FROM offline_artifacts
            WHERE episode_id = ? ORDER BY created_at_ms, rowid
            """,
            (episode_id,),
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> OfflineArtifact | None:
        row = self._conn.execute(
            "SELECT * FROM offline_artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return self._row_to_artifact(row) if row is not None else None

    def recent_artifacts(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[OfflineArtifact]:
        rows = self._conn.execute(
            """
            SELECT artifact.* FROM offline_artifacts AS artifact
            JOIN offline_episodes AS episode ON episode.id = artifact.episode_id
            WHERE episode.conversation_id = ?
              AND episode.status = 'completed'
            ORDER BY artifact.created_at_ms DESC, artifact.rowid DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    @staticmethod
    def _row_to_episode(row: sqlite3.Row | dict[str, Any]) -> OfflineEpisode:
        data = _dict(row)
        return OfflineEpisode(
            id=data["id"],
            conversation_id=data["conversation_id"],
            correlation_id=data["correlation_id"],
            dedup_key=data["dedup_key"],
            action_kind=OfflineActionKind(data["action_kind"]),
            motive=data["motive"],
            status=OfflineEpisodeStatus(data["status"]),
            goal_id=data["goal_id"],
            source_event_ids=json.loads(data["source_event_ids_json"] or "[]"),
            source_trace_ids=json.loads(data["source_trace_ids_json"] or "[]"),
            provider=data["provider"],
            tool_name=data["tool_name"],
            tool_output_hash=data["tool_output_hash"],
            summary=data["summary"],
            error=data["error"],
            event_id=data["event_id"],
            started_at_ms=data["started_at_ms"],
            completed_at_ms=data["completed_at_ms"],
        )

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row | dict[str, Any]) -> OfflineArtifact:
        data = _dict(row)
        return OfflineArtifact(
            id=data["id"],
            episode_id=data["episode_id"],
            artifact_type=data["artifact_type"],
            title=data["title"],
            content=data["content"],
            evidence_event_ids=json.loads(data["evidence_event_ids_json"] or "[]"),
            evidence_trace_ids=json.loads(data["evidence_trace_ids_json"] or "[]"),
            metadata=json.loads(data["metadata_json"] or "{}"),
            created_at_ms=data["created_at_ms"],
        )


class InnerLoopRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, conversation_id: str) -> InnerLoopState | None:
        row = self._conn.execute(
            "SELECT * FROM inner_loop_states WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return self._row_to_state(row) if row is not None else None

    def upsert(self, state: InnerLoopState) -> InnerLoopState:
        self._conn.execute(
            """
            INSERT INTO inner_loop_states (
                conversation_id, version, mode, reply_expectation, concern,
                curiosity, connection_pressure, uncertainty, offline_readiness,
                wait_started_at_ms, wait_deadline_at_ms, last_user_event_id,
                last_agent_event_id, last_heartbeat_at_ms, transition_reason,
                updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                version = excluded.version,
                mode = excluded.mode,
                reply_expectation = excluded.reply_expectation,
                concern = excluded.concern,
                curiosity = excluded.curiosity,
                connection_pressure = excluded.connection_pressure,
                uncertainty = excluded.uncertainty,
                offline_readiness = excluded.offline_readiness,
                wait_started_at_ms = excluded.wait_started_at_ms,
                wait_deadline_at_ms = excluded.wait_deadline_at_ms,
                last_user_event_id = excluded.last_user_event_id,
                last_agent_event_id = excluded.last_agent_event_id,
                last_heartbeat_at_ms = excluded.last_heartbeat_at_ms,
                transition_reason = excluded.transition_reason,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                state.conversation_id,
                state.version,
                state.mode.value,
                state.reply_expectation,
                state.concern,
                state.curiosity,
                state.connection_pressure,
                state.uncertainty,
                state.offline_readiness,
                state.wait_started_at_ms,
                state.wait_deadline_at_ms,
                state.last_user_event_id,
                state.last_agent_event_id,
                state.last_heartbeat_at_ms,
                state.transition_reason,
                state.updated_at_ms,
            ),
        )
        return state

    @staticmethod
    def _row_to_state(row: sqlite3.Row | dict[str, Any]) -> InnerLoopState:
        data = _dict(row)
        return InnerLoopState(
            conversation_id=data["conversation_id"],
            version=data["version"],
            mode=InnerLifeMode(data["mode"]),
            reply_expectation=data["reply_expectation"],
            concern=data["concern"],
            curiosity=data["curiosity"],
            connection_pressure=data["connection_pressure"],
            uncertainty=data["uncertainty"],
            offline_readiness=data["offline_readiness"],
            wait_started_at_ms=data["wait_started_at_ms"],
            wait_deadline_at_ms=data["wait_deadline_at_ms"],
            last_user_event_id=data["last_user_event_id"],
            last_agent_event_id=data["last_agent_event_id"],
            last_heartbeat_at_ms=data["last_heartbeat_at_ms"],
            transition_reason=data["transition_reason"],
            updated_at_ms=data["updated_at_ms"],
        )


class BackgroundUsageRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def try_consume(self, usage_day: str, now_ms: int, *, limit: int, amount: int = 1) -> bool:
        if amount < 1:
            raise ValueError("amount must be positive")
        self._conn.execute(
            """
            INSERT OR IGNORE INTO background_usage (usage_day, llm_calls, updated_at_ms)
            VALUES (?, 0, ?)
            """,
            (usage_day, now_ms),
        )
        cursor = self._conn.execute(
            """
            UPDATE background_usage
            SET llm_calls = llm_calls + ?, updated_at_ms = ?
            WHERE usage_day = ? AND llm_calls + ? <= ?
            """,
            (amount, now_ms, usage_day, amount, limit),
        )
        return cursor.rowcount == 1

    def used(self, usage_day: str) -> int:
        row = self._conn.execute(
            "SELECT llm_calls FROM background_usage WHERE usage_day = ?",
            (usage_day,),
        ).fetchone()
        return int(row["llm_calls"]) if row is not None else 0


__all__ = [
    "BackgroundUsageRepository",
    "CausalTraceRepository",
    "ContactEpisodeRepository",
    "EmotionEpisodeRepository",
    "GoalRepository",
    "InitiativeRepository",
    "InnerLoopRepository",
    "OfflineAgencyRepository",
    "OutboxRepository",
    "ScheduledJobRepository",
    "WorldObservationRepository",
]
