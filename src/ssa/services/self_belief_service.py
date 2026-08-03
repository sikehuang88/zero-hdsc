"""Evidence review and lifecycle rules for self beliefs (M10)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from ssa.adapters.llm import ChatMessage, LLMAdapter, LLMRequest
from ssa.clock import Clock
from ssa.config import IdentityConfig, ReasoningEffort, ThinkingMode
from ssa.domain.enums import Actor
from ssa.domain.events import Event
from ssa.domain.self_belief import (
    SelfBelief,
    SelfBeliefCandidate,
    SelfBeliefStatus,
)
from ssa.ids import IdGenerator
from ssa.storage.self_belief_repository import SqliteSelfBeliefRepository

logger = logging.getLogger(__name__)

_DAY_MS = 24 * 60 * 60 * 1000


class SelfBeliefService:
    """Turns reviewed evidence into conservative, versioned identity claims."""

    def __init__(
        self,
        llm: LLMAdapter,
        repository: SqliteSelfBeliefRepository,
        event_lookup: Callable[[str], Event | None],
        ids: IdGenerator,
        clock: Clock,
        config: IdentityConfig,
        *,
        prompt_version: str = "identity_review_v1",
        rules_version: str = "self_belief_rules_v1",
        default_model: str = "deepseek/deepseek-v4-pro",
    ) -> None:
        self._llm = llm
        self._repo = repository
        self._event_lookup = event_lookup
        self._ids = ids
        self._clock = clock
        self._config = config
        self._prompt_version = prompt_version
        self._rules_version = rules_version
        self._default_model = default_model

    async def extract_candidates(self, events: list[Event]) -> list[SelfBeliefCandidate]:
        """Extract candidates from eligible events in the most recent week.

        The model proposes claims; code verifies provenance and the two-event
        minimum before returning anything that can be persisted.
        """
        now_ms = self._clock.now_ms()
        cutoff_ms = now_ms - (7 * _DAY_MS)
        eligible = [
            event
            for event in events
            if cutoff_ms <= event.created_at_ms <= now_ms and self._is_identity_source(event)
        ][-8:]
        if len({event.id for event in eligible}) < 2:
            return []

        event_text = "\n".join(
            f"- id={event.id}; at_ms={event.created_at_ms}; "
            f"type={event.event_type}; actor={event.actor.value}; "
            f"content={event.content.replace(chr(10), ' ')[:600]}"
            for event in eligible
        )
        current_beliefs = self._repo.list_current(limit=20)
        current_text = "\n".join(f"- {belief.claim}" for belief in current_beliefs) or "- none"
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "confidence": {"type": "number"},
                            "evidence_event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "change_reason": {"type": "string"},
                        },
                        "required": [
                            "claim",
                            "confidence",
                            "evidence_event_ids",
                            "change_reason",
                        ],
                    },
                }
            },
            "required": ["candidates"],
        }
        request = LLMRequest(
            purpose="identity_review",
            messages=[
                ChatMessage.system(
                    "Review recent actions and choices for cautious claims about the "
                    "digital lifeform's own tendencies. Every claim must cite at least "
                    "two distinct supplied event IDs. Do not turn a single response into "
                    "identity. Treat claims as revisable and keep confidence at or below 0.6. "
                    "When new evidence supports an existing claim, reuse that claim verbatim. "
                    "Return at most three concise claims. Output compact JSON with a "
                    "'candidates' array. Every item must contain claim, confidence, "
                    "evidence_event_ids, and change_reason."
                ),
                ChatMessage.user(
                    f"Current claims (reuse exact text when supported):\n{current_text}\n\n"
                    f"Eligible events from the last seven days:\n{event_text}"
                ),
            ],
            model=self._default_model,
            temperature=None,
            max_tokens=2048,
            thinking=ThinkingMode.ENABLED,
            reasoning_effort=ReasoningEffort.HIGH,
            prompt_version=self._prompt_version,
            json_schema=schema,
        )

        try:
            response = await self._llm.complete(request)
        except Exception as exc:
            logger.warning("Identity review LLM call failed: %s", exc)
            return []
        if response.parsed is None:
            logger.warning("Identity review returned no parsed JSON")
            return []

        raw_candidates = response.parsed.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return []

        eligible_ids = {event.id for event in eligible}
        seen_claims: set[str] = set()
        candidates: list[SelfBeliefCandidate] = []
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                continue
            raw_ids = raw.get("evidence_event_ids", raw.get("cited_event_ids", []))
            if (
                not isinstance(raw_ids, list)
                or not raw_ids
                or any(
                    not isinstance(event_id, str) or event_id not in eligible_ids
                    for event_id in raw_ids
                )
            ):
                continue
            evidence_ids = list(dict.fromkeys(raw_ids))
            if len(evidence_ids) < 2:
                continue
            try:
                claim = str(raw["claim"]).strip()
                normalized_claim = claim.casefold()
                if normalized_claim in seen_claims:
                    continue
                candidate = SelfBeliefCandidate(
                    claim=claim,
                    confidence=min(
                        float(raw.get("confidence", 0.5)),
                        self._config.candidate_confidence_cap,
                        0.6,
                    ),
                    evidence_event_ids=evidence_ids,
                    change_reason=str(raw.get("change_reason", "weekly identity review")),
                    model=response.model,
                    prompt_version=self._prompt_version,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping invalid self-belief candidate: %s", exc)
                continue
            seen_claims.add(normalized_claim)
            candidates.append(candidate)
        return candidates

    def create_candidate(
        self,
        candidate: SelfBeliefCandidate,
        *,
        cause_event_id: str | None = None,
    ) -> SelfBelief:
        """Persist a version-one candidate after deterministic source checks."""
        self._validate_candidate_sources(candidate.evidence_event_ids)
        cause_id = cause_event_id or candidate.evidence_event_ids[-1]
        self._require_event(cause_id)
        now_ms = self._clock.now_ms()
        belief_id = self._ids.new()
        belief = SelfBelief(
            id=belief_id,
            lineage_id=belief_id,
            claim=candidate.claim,
            confidence=min(
                candidate.confidence,
                self._config.candidate_confidence_cap,
                0.6,
            ),
            status=SelfBeliefStatus.CANDIDATE,
            version=1,
            evidence_event_ids=list(candidate.evidence_event_ids),
            counterevidence_event_ids=[],
            previous_id=None,
            cause_event_id=cause_id,
            change_reason=candidate.change_reason,
            model=candidate.model,
            prompt_version=candidate.prompt_version,
            candidate_since_ms=now_ms,
            activated_at_ms=None,
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        return self._repo.insert_initial(belief)

    def create_candidates(
        self,
        candidates: Iterable[SelfBeliefCandidate],
    ) -> list[SelfBelief]:
        """Persist a reviewed batch, skipping claims already represented."""
        created: list[SelfBelief] = []
        for candidate in candidates:
            if self._repo.find_current_by_claim(candidate.claim) is None:
                created.append(self.create_candidate(candidate))
        return created

    def integrate_candidates(
        self,
        candidates: Iterable[SelfBeliefCandidate],
        *,
        support_confidence_delta: float = 0.05,
    ) -> list[SelfBelief]:
        """Create new claims and append unseen support to existing lineages."""
        self._validate_delta(support_confidence_delta)
        changed: list[SelfBelief] = []
        for candidate in candidates:
            current = self._repo.find_current_by_claim(candidate.claim)
            if current is None:
                changed.append(self.create_candidate(candidate))
                continue

            updated = current
            known_evidence = {
                *current.evidence_event_ids,
                *current.counterevidence_event_ids,
            }
            for event_id in candidate.evidence_event_ids:
                if event_id in known_evidence:
                    continue
                updated = self.record_support(
                    updated.lineage_id,
                    event_id,
                    confidence_delta=support_confidence_delta,
                    change_reason=candidate.change_reason,
                )
                known_evidence.add(event_id)
            if updated.id != current.id:
                changed.append(updated)
        return changed

    def record_support(
        self,
        lineage_id: str,
        event_id: str,
        *,
        confidence_delta: float = 0.1,
        change_reason: str = "new supporting evidence",
    ) -> SelfBelief:
        """Append supporting evidence and promote mature candidates when eligible."""
        self._validate_delta(confidence_delta)
        current = self._current(lineage_id)
        self._ensure_mutable(current)
        self._require_identity_source(event_id)
        if event_id in current.counterevidence_event_ids:
            raise SelfBeliefEvidenceError(f"event {event_id!r} already contradicts this belief")
        if event_id in current.evidence_event_ids:
            return current

        now_ms = self._clock.now_ms()
        evidence = [*current.evidence_event_ids, event_id]
        confidence = min(1.0, current.confidence + confidence_delta)
        status = current.status
        activated_at_ms = current.activated_at_ms
        if current.status == SelfBeliefStatus.CANDIDATE and self._eligible_for_activation(
            confidence,
            evidence,
            current.candidate_since_ms,
            now_ms,
        ):
            status = SelfBeliefStatus.ACTIVE
            activated_at_ms = now_ms

        return self._append(
            current,
            cause_event_id=event_id,
            change_reason=change_reason,
            confidence=confidence,
            status=status,
            evidence_event_ids=evidence,
            activated_at_ms=activated_at_ms,
        )

    def review_candidate(
        self,
        lineage_id: str,
        cause_event_id: str,
        *,
        change_reason: str = "scheduled identity review",
    ) -> SelfBelief:
        """Activate an old-enough, sufficiently supported candidate."""
        current = self._current(lineage_id)
        if current.status != SelfBeliefStatus.CANDIDATE:
            raise SelfBeliefLifecycleError("only candidate beliefs can be activated")
        self._require_event(cause_event_id)
        now_ms = self._clock.now_ms()
        if not self._eligible_for_activation(
            current.confidence,
            current.evidence_event_ids,
            current.candidate_since_ms,
            now_ms,
        ):
            return current
        return self._append(
            current,
            cause_event_id=cause_event_id,
            change_reason=change_reason,
            status=SelfBeliefStatus.ACTIVE,
            activated_at_ms=now_ms,
        )

    def record_counterevidence(
        self,
        lineage_id: str,
        event_id: str,
        *,
        confidence_delta: float = 0.2,
        major: bool = True,
        change_reason: str = "new counterevidence",
    ) -> SelfBelief:
        """Retain a counterexample and challenge a stable claim when major."""
        self._validate_delta(confidence_delta)
        current = self._current(lineage_id)
        self._ensure_mutable(current)
        self._require_identity_source(event_id)
        if event_id in current.evidence_event_ids:
            raise SelfBeliefEvidenceError(f"event {event_id!r} already supports this belief")
        if event_id in current.counterevidence_event_ids:
            return current

        status = current.status
        if major and current.status in {
            SelfBeliefStatus.ACTIVE,
            SelfBeliefStatus.REVISED,
        }:
            status = SelfBeliefStatus.CHALLENGED
        return self._append(
            current,
            cause_event_id=event_id,
            change_reason=change_reason,
            confidence=max(0.0, current.confidence - confidence_delta),
            status=status,
            counterevidence_event_ids=[
                *current.counterevidence_event_ids,
                event_id,
            ],
        )

    def revise(
        self,
        lineage_id: str,
        new_claim: str,
        cause_event_id: str,
        *,
        change_reason: str,
        confidence: float | None = None,
        model: str = "deterministic",
        prompt_version: str | None = None,
    ) -> SelfBelief:
        """Replace a challenged claim while preserving its full predecessor."""
        current = self._current(lineage_id)
        if current.status != SelfBeliefStatus.CHALLENGED:
            raise SelfBeliefLifecycleError("only challenged beliefs can be revised")
        self._require_event(cause_event_id)
        new_claim = new_claim.strip()
        if not new_claim or new_claim == current.claim:
            raise ValueError("a revision must provide a different non-empty claim")
        new_confidence = current.confidence if confidence is None else confidence
        self._validate_confidence(new_confidence)
        return self._append(
            current,
            cause_event_id=cause_event_id,
            change_reason=change_reason,
            claim=new_claim,
            confidence=new_confidence,
            status=SelfBeliefStatus.REVISED,
            model=model,
            prompt_version=prompt_version or self._prompt_version,
        )

    def archive(
        self,
        lineage_id: str,
        cause_event_id: str,
        *,
        change_reason: str,
    ) -> SelfBelief:
        """Append a terminal archived version instead of deleting history."""
        current = self._current(lineage_id)
        self._ensure_mutable(current)
        self._require_event(cause_event_id)
        return self._append(
            current,
            cause_event_id=cause_event_id,
            change_reason=change_reason,
            status=SelfBeliefStatus.ARCHIVED,
        )

    def relevant_for_prompt(
        self,
        lineage_ids: Iterable[str] | None = None,
        *,
        limit: int = 8,
    ) -> list[SelfBelief]:
        """Return only current stable/revised beliefs selected for this prompt."""
        beliefs = self._repo.list_current({SelfBeliefStatus.ACTIVE, SelfBeliefStatus.REVISED})
        if lineage_ids is not None:
            selected = set(lineage_ids)
            beliefs = [belief for belief in beliefs if belief.lineage_id in selected]
        return beliefs[:limit]

    @staticmethod
    def render_prompt_context(beliefs: Iterable[SelfBelief]) -> str:
        """Render confidence and evidence summaries without injecting history."""
        lines = [
            f"- {belief.claim} (confidence={belief.confidence:.2f}; "
            f"evidence={','.join(belief.evidence_event_ids)})"
            for belief in beliefs
        ]
        return "\n".join(lines)

    def _append(
        self,
        current: SelfBelief,
        *,
        cause_event_id: str,
        change_reason: str,
        claim: str | None = None,
        confidence: float | None = None,
        status: SelfBeliefStatus | None = None,
        evidence_event_ids: list[str] | None = None,
        counterevidence_event_ids: list[str] | None = None,
        activated_at_ms: int | None = None,
        model: str = "deterministic",
        prompt_version: str | None = None,
    ) -> SelfBelief:
        now_ms = self._clock.now_ms()
        successor = SelfBelief(
            id=self._ids.new(),
            lineage_id=current.lineage_id,
            claim=claim if claim is not None else current.claim,
            confidence=confidence if confidence is not None else current.confidence,
            status=status if status is not None else current.status,
            version=current.version + 1,
            evidence_event_ids=(
                evidence_event_ids
                if evidence_event_ids is not None
                else list(current.evidence_event_ids)
            ),
            counterevidence_event_ids=(
                counterevidence_event_ids
                if counterevidence_event_ids is not None
                else list(current.counterevidence_event_ids)
            ),
            previous_id=current.id,
            cause_event_id=cause_event_id,
            change_reason=change_reason,
            model=model,
            prompt_version=prompt_version or self._rules_version,
            candidate_since_ms=current.candidate_since_ms,
            activated_at_ms=(
                activated_at_ms if activated_at_ms is not None else current.activated_at_ms
            ),
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        return self._repo.append_if_version(current.version, successor)

    def _current(self, lineage_id: str) -> SelfBelief:
        current = self._repo.latest(lineage_id)
        if current is None:
            raise SelfBeliefNotFoundError(f"Unknown self-belief lineage {lineage_id!r}")
        return current

    def _validate_candidate_sources(self, event_ids: list[str]) -> None:
        now_ms = self._clock.now_ms()
        cutoff_ms = now_ms - (7 * _DAY_MS)
        if len(set(event_ids)) < 2:
            raise SelfBeliefEvidenceError("a candidate needs two independent events")
        for event_id in event_ids:
            event = self._require_identity_source(event_id)
            if not cutoff_ms <= event.created_at_ms <= now_ms:
                raise SelfBeliefEvidenceError(
                    f"candidate evidence event {event_id!r} is outside the recent week"
                )

    def _require_event(self, event_id: str) -> Event:
        event = self._event_lookup(event_id)
        if event is None:
            raise SelfBeliefEvidenceError(f"Unknown evidence event {event_id!r}")
        return event

    def _require_identity_source(self, event_id: str) -> Event:
        event = self._require_event(event_id)
        if not self._is_identity_source(event):
            raise SelfBeliefEvidenceError(
                f"event {event_id!r} is not an agent action, goal choice, or relationship event"
            )
        return event

    @staticmethod
    def _is_identity_source(event: Event) -> bool:
        return event.actor == Actor.AGENT or event.event_type.startswith(("goal.", "relationship."))

    def _eligible_for_activation(
        self,
        confidence: float,
        evidence_event_ids: list[str],
        candidate_since_ms: int,
        now_ms: int,
    ) -> bool:
        minimum_age_ms = self._config.active_min_age_days * _DAY_MS
        return (
            len(set(evidence_event_ids)) >= 2
            and confidence >= self._config.active_confidence_threshold
            and now_ms - candidate_since_ms >= minimum_age_ms
        )

    @staticmethod
    def _ensure_mutable(belief: SelfBelief) -> None:
        if belief.status == SelfBeliefStatus.ARCHIVED:
            raise SelfBeliefLifecycleError("archived beliefs are terminal")

    @staticmethod
    def _validate_delta(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence delta must be in [0, 1], got {value}")

    @staticmethod
    def _validate_confidence(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value}")


class SelfBeliefEvidenceError(ValueError):
    """Raised when evidence provenance or independence checks fail."""


class SelfBeliefLifecycleError(ValueError):
    """Raised when a requested lifecycle transition is invalid."""


class SelfBeliefNotFoundError(LookupError):
    """Raised when a logical self-belief lineage does not exist."""


__all__ = [
    "SelfBeliefEvidenceError",
    "SelfBeliefLifecycleError",
    "SelfBeliefNotFoundError",
    "SelfBeliefService",
]
