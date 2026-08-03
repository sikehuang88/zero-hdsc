"""Persistent self/shared goal state machine for background activity."""

from __future__ import annotations

from ssa.clock import Clock
from ssa.config import GoalConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import Goal, GoalOwner, GoalStatus, GoalStep
from ssa.domain.state import OrganismState
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import GoalRepository
from ssa.storage.trace_repository import SqliteTraceRepository


class GoalService:
    def __init__(
        self,
        *,
        goals: GoalRepository,
        traces: SqliteTraceRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
        config: GoalConfig,
    ) -> None:
        self._goals = goals
        self._traces = traces
        self._events = events
        self._clock = clock
        self._ids = ids
        self._config = config

    def ensure_self_project(
        self,
        conversation_id: str,
        organism: OrganismState,
    ) -> Goal | None:
        open_goals = self._goals.list_open(conversation_id, owner=GoalOwner.SELF)
        if open_goals or organism.curiosity < 0.45:
            return open_goals[0] if open_goals else None
        now_ms = self._clock.now_ms()
        correlation_id = self._ids.new()
        event = self._append_event(
            conversation_id,
            correlation_id,
            "goal.created",
            "A self-directed continuity journal project was created.",
        )
        goal = Goal(
            id=self._ids.new(),
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            owner=GoalOwner.SELF,
            title="Maintain the shared continuity journal",
            motive="Review meaningful episodes and preserve relationship continuity over time",
            success_criteria=[
                "Review at least seven distinct episodes",
                "Create one evidence-linked continuity summary",
            ],
            priority=_clip(0.45 + 0.35 * organism.curiosity + 0.20 * organism.autonomy_need),
            status=GoalStatus.ACTIVE,
            due_at_ms=now_ms + self._config.min_project_days * 86_400_000,
            next_action_at_ms=now_ms,
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        stored = self._goals.insert(goal)
        self._goals.add_step(
            GoalStep(
                id=self._ids.new(),
                goal_id=stored.id,
                action="Initialize the continuity journal",
                result="Project initialized from the persistent trace archive",
                source_event_id=event.id,
                created_at_ms=now_ms,
            )
        )
        return stored

    def advance_due(self, conversation_id: str) -> Goal | None:
        due = self._goals.list_due(conversation_id, self._clock.now_ms())
        if not due:
            return None
        goal = max(due, key=self.urgency)
        traces = self._traces.recent(conversation_id, limit=120)
        steps = self._goals.steps(goal.id)
        reviewed_ids = {
            str(trace_id) for step in steps for trace_id in step.metadata.get("trace_ids", [])
        }
        candidate = next(
            (trace for trace in reversed(traces) if trace.id not in reviewed_ids),
            None,
        )
        now_ms = self._clock.now_ms()
        correlation_id = self._ids.new()
        if candidate is None:
            return self._goals.update(
                goal.model_copy(
                    update={
                        "status": GoalStatus.BLOCKED,
                        "blocked_reason": "No unreviewed episode is available",
                        "next_action_at_ms": now_ms + 6 * 60 * 60 * 1_000,
                        "updated_at_ms": now_ms,
                    }
                )
            )

        event = self._append_event(
            conversation_id,
            correlation_id,
            "goal.step",
            f"Reviewed one continuity episode: {candidate.content[:180]}",
            metadata={"goal_id": goal.id, "trace_ids": [candidate.id]},
        )
        self._goals.add_step(
            GoalStep(
                id=self._ids.new(),
                goal_id=goal.id,
                action="Review one archived episode",
                result=f"Reviewed trace {candidate.id}",
                source_event_id=event.id,
                metadata={"trace_ids": [candidate.id]},
                created_at_ms=now_ms,
            )
        )
        reviewed_count = len(reviewed_ids) + 1
        progress = min(1.0, reviewed_count / 7.0)
        status = GoalStatus.DONE if progress >= 1.0 else GoalStatus.ACTIVE
        return self._goals.update(
            goal.model_copy(
                update={
                    "progress": progress,
                    "status": status,
                    "next_action_at_ms": None if status == GoalStatus.DONE else now_ms + 86_400_000,
                    "blocked_reason": None,
                    "updated_at_ms": now_ms,
                    "correlation_id": correlation_id,
                }
            )
        )

    def urgency(self, goal: Goal) -> float:
        now_ms = self._clock.now_ms()
        due_pressure = 0.0
        if goal.due_at_ms is not None:
            remaining = goal.due_at_ms - now_ms
            due_pressure = _clip(
                1.0 - remaining / max(1.0, self._config.min_project_days * 86_400_000)
            )
        opportunity = (
            1.0 if goal.next_action_at_ms is not None and goal.next_action_at_ms <= now_ms else 0.0
        )
        continuity = min(1.0, len(self._goals.steps(goal.id)) / 7.0)
        return _clip(
            goal.priority * self._config.urgency_priority_weight
            + due_pressure * self._config.urgency_due_weight
            + (1.0 - goal.progress) * self._config.urgency_need_weight
            + opportunity * self._config.urgency_opportunity_weight
            + continuity * self._config.urgency_continuity_weight
        )

    def _append_event(
        self,
        conversation_id: str,
        correlation_id: str,
        event_type: str,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
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
                    metadata=metadata or {},
                ),
                event_id=event_id,
                correlation_id=correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["GoalService"]
