"""Persistent contact episodes and bounded silence-aware follow-up behavior."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from ssa.clock import Clock
from ssa.config import InitiativeConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import (
    ContactEpisode,
    ContactPhase,
    ContactStatus,
    Initiative,
    InitiativeStatus,
    OutboxMessage,
)
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import (
    ContactEpisodeRepository,
    InitiativeRepository,
    OutboxRepository,
)

_HOUR_MS = 3_600_000


@dataclass(frozen=True)
class ContactEvaluationResult:
    observed_silence: int
    follow_ups_queued: int
    withdrawn: int
    initiative_ids: tuple[str, ...]


class ContactEpisodeService:
    """Advance proactive contact without equating silence with rejection."""

    def __init__(
        self,
        *,
        contacts: ContactEpisodeRepository,
        initiatives: InitiativeRepository,
        outbox: OutboxRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
        config: InitiativeConfig,
        timezone: str,
        wait_hours: int = 6,
        max_messages: int = 3,
    ) -> None:
        if wait_hours < 1:
            raise ValueError("wait_hours must be positive")
        self._contacts = contacts
        self._initiatives = initiatives
        self._outbox = outbox
        self._events = events
        self._clock = clock
        self._ids = ids
        self._config = config
        self._timezone = ZoneInfo(timezone)
        self._wait_ms = wait_hours * _HOUR_MS
        self._max_messages = max_messages

    def on_delivered(self, initiative: Initiative, event: Event) -> ContactEpisode:
        """Open a contact episode or advance the episode named by a follow-up."""
        now_ms = self._clock.now_ms()
        episode_id = initiative.decision.get("contact_episode_id")
        if isinstance(episode_id, str):
            episode = self._contacts.get(episode_id)
            if episode is None:
                raise LookupError(f"contact episode {episode_id!r} does not exist")
            updated = episode.model_copy(
                update={
                    "phase": ContactPhase.WAITING,
                    "message_count": episode.message_count + 1,
                    "last_sent_at_ms": now_ms,
                    "next_action_at_ms": now_ms + self._wait_ms,
                    "updated_at_ms": now_ms,
                }
            )
            return self._contacts.update(updated)

        existing = self._contacts.find_by_source_initiative(initiative.id)
        if existing is not None:
            return existing
        opened = self._contacts.insert(
            ContactEpisode(
                id=self._ids.new(),
                conversation_id=initiative.conversation_id,
                correlation_id=initiative.correlation_id or event.correlation_id,
                motive=initiative.motive,
                topic=self._topic(initiative),
                phase=ContactPhase.OPENING,
                message_count=0,
                max_messages=self._max_messages,
                hypotheses={"busy": 0.50, "unavailable": 0.35, "avoidance": 0.15},
                source_initiative_id=initiative.id,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
        )
        waiting = opened.model_copy(
            update={
                "phase": ContactPhase.WAITING,
                "message_count": 1,
                "last_sent_at_ms": now_ms,
                "next_action_at_ms": now_ms + self._wait_ms,
                "updated_at_ms": now_ms,
            }
        )
        return self._contacts.update(waiting)

    def observe_user_event(self, event: Event) -> list[ContactEpisode]:
        if event.actor != Actor.USER:
            return []
        resolved: list[ContactEpisode] = []
        for episode in self._contacts.active(event.conversation_id):
            if episode.last_sent_at_ms is None or event.created_at_ms < episode.last_sent_at_ms:
                continue
            updated = episode.model_copy(
                update={
                    "phase": ContactPhase.RESOLUTION,
                    "status": ContactStatus.RESOLVED,
                    "next_action_at_ms": None,
                    "resolved_by_event_id": event.id,
                    "updated_at_ms": event.created_at_ms,
                }
            )
            resolved.append(self._contacts.update(updated))
            self._append_event(
                updated,
                "contact.resolved",
                "The user returned after a proactive contact episode.",
                parent_event_id=event.id,
            )
        return resolved

    def evaluate_due(self, conversation_id: str) -> ContactEvaluationResult:
        now_ms = self._clock.now_ms()
        observed_silence = 0
        queued_ids: list[str] = []
        withdrawn = 0
        for episode in self._contacts.due(conversation_id, now_ms):
            observed_silence += 1
            hypotheses = self._silence_hypotheses(episode.hypotheses)
            self._append_event(
                episode,
                "contact.silence_observed",
                "No reply was observed yet; several explanations remain plausible.",
                metadata={"hypotheses": hypotheses, "message_count": episode.message_count},
            )
            if episode.message_count >= episode.max_messages:
                self._withdraw(episode, hypotheses)
                withdrawn += 1
                continue
            gate_delay = self._delivery_gate_delay(conversation_id, now_ms)
            if gate_delay is not None:
                self._contacts.update(
                    episode.model_copy(
                        update={
                            "hypotheses": hypotheses,
                            "next_action_at_ms": now_ms + gate_delay,
                            "updated_at_ms": now_ms,
                        }
                    )
                )
                continue
            follow_up = self._queue_follow_up(episode, hypotheses)
            queued_ids.append(follow_up.id)
        return ContactEvaluationResult(
            observed_silence=observed_silence,
            follow_ups_queued=len(queued_ids),
            withdrawn=withdrawn,
            initiative_ids=tuple(queued_ids),
        )

    def _queue_follow_up(
        self,
        episode: ContactEpisode,
        hypotheses: dict[str, float],
    ) -> Initiative:
        now_ms = self._clock.now_ms()
        next_number = episode.message_count + 1
        phase = ContactPhase.FOLLOW_UP if next_number == 2 else ContactPhase.ESCALATION
        dedup_key = f"contact:{episode.id}:message:{next_number}"
        existing = self._initiatives.find_by_dedup(dedup_key)
        if existing is not None:
            return existing
        draft = (
            f"No rush to reply. I just wanted to leave a small thought beside {episode.topic}."
            if phase == ContactPhase.FOLLOW_UP
            else "I will give you some quiet space now. I am still here when you return."
        )
        candidate = self._initiatives.insert(
            Initiative(
                id=self._ids.new(),
                conversation_id=episode.conversation_id,
                motive=f"contact_follow_up:{episode.motive}",
                intent="initiate",
                content_draft=draft,
                urgency=max(0.7, 1.0 - 0.1 * episode.message_count),
                decision_score=max(0.3, 0.7 - 0.1 * episode.message_count),
                status=InitiativeStatus.CANDIDATE,
                earliest_send_at_ms=now_ms,
                expires_at_ms=now_ms + self._wait_ms,
                correlation_id=episode.correlation_id,
                source_event_ids=self._episode_evidence(episode),
                decision={
                    "gate": "contact_episode",
                    "contact_episode_id": episode.id,
                    "phase": phase.value,
                    "hypotheses": hypotheses,
                    "silence_is_uncertain": True,
                },
                channel="local",
                dedup_key=dedup_key,
                created_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
        )
        approved = self._initiatives.update_status(
            candidate.id,
            InitiativeStatus.APPROVED,
            now_ms,
        )
        self._outbox.enqueue(
            OutboxMessage(
                id=self._ids.new(),
                correlation_id=episode.correlation_id,
                channel=approved.channel,
                recipient=episode.conversation_id,
                payload={"content": approved.content_draft, "contact_episode_id": episode.id},
                next_attempt_at_ms=now_ms,
                created_at_ms=now_ms,
                initiative_id=approved.id,
                dedup_key=f"outbox:{approved.id}",
            )
        )
        queued = self._initiatives.update_status(
            approved.id,
            InitiativeStatus.QUEUED,
            now_ms,
        )
        self._contacts.update(
            episode.model_copy(
                update={
                    "phase": phase,
                    "hypotheses": hypotheses,
                    "next_action_at_ms": None,
                    "updated_at_ms": now_ms,
                }
            )
        )
        return queued

    def _withdraw(self, episode: ContactEpisode, hypotheses: dict[str, float]) -> None:
        now_ms = self._clock.now_ms()
        withdrawn = episode.model_copy(
            update={
                "phase": ContactPhase.WITHDRAWAL,
                "status": ContactStatus.CANCELLED,
                "hypotheses": hypotheses,
                "next_action_at_ms": None,
                "updated_at_ms": now_ms,
            }
        )
        self._contacts.update(withdrawn)
        self._append_event(
            withdrawn,
            "contact.withdrawn",
            "The contact episode reached its message bound and returned to quiet waiting.",
        )

    def _delivery_gate_delay(self, conversation_id: str, now_ms: int) -> int | None:
        if self._is_quiet(now_ms):
            return _HOUR_MS
        local = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone)
        midnight = datetime.combine(local.date(), time.min, tzinfo=self._timezone)
        day_start_ms = int(midnight.timestamp() * 1000)
        if (
            self._initiatives.count_sent_since(conversation_id, day_start_ms)
            >= self._config.daily_limit
        ):
            return _HOUR_MS
        latest = self._initiatives.latest_sent(conversation_id)
        cooldown_ms = self._config.cooldown_minutes * 60_000
        if latest is not None:
            remaining = cooldown_ms - (now_ms - latest.updated_at_ms)
            if remaining > 0:
                return remaining
        return None

    def _is_quiet(self, now_ms: int) -> bool:
        local_time = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone).time()
        start = time.fromisoformat(self._config.quiet_hours_start)
        end = time.fromisoformat(self._config.quiet_hours_end)
        if start == end:
            return False
        if start < end:
            return start <= local_time < end
        return local_time >= start or local_time < end

    @staticmethod
    def _silence_hypotheses(current: dict[str, float]) -> dict[str, float]:
        raw = {
            "busy": current.get("busy", 0.5) + 0.04,
            "unavailable": current.get("unavailable", 0.35) + 0.03,
            "avoidance": current.get("avoidance", 0.15) + 0.01,
        }
        total = sum(raw.values())
        return {name: round(weight / total, 6) for name, weight in raw.items()}

    def _episode_evidence(self, episode: ContactEpisode) -> list[str]:
        if episode.source_initiative_id is None:
            return []
        initiative = self._initiatives.get(episode.source_initiative_id)
        if initiative is None:
            return []
        evidence = list(initiative.source_event_ids)
        if initiative.sent_event_id is not None:
            evidence.append(initiative.sent_event_id)
        return list(dict.fromkeys(evidence))

    @staticmethod
    def _topic(initiative: Initiative) -> str:
        value = initiative.decision.get("topic")
        if isinstance(value, str) and value.strip():
            return value.strip()
        digest = hashlib.sha256(initiative.content_draft.encode("utf-8")).hexdigest()[:8]
        return f"shared thread {digest}"

    def _append_event(
        self,
        episode: ContactEpisode,
        event_type: str,
        content: str,
        *,
        parent_event_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Event:
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.SYSTEM,
                    signal_type=event_type,
                    content=content,
                    channel="lifecycle",
                    channel_message_id=f"{event_type}:{episode.id}:{self._ids.new()}",
                    conversation_id=episode.conversation_id,
                    parent_event_id=parent_event_id,
                    metadata={"contact_episode_id": episode.id, **(metadata or {})},
                ),
                event_id=self._ids.new(),
                correlation_id=episode.correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.SYSTEM_DERIVED,
            )
        )


__all__ = ["ContactEpisodeService", "ContactEvaluationResult"]
