"""Appraisal service — evaluates events into structured significance.

Pipeline §16 (M07):
1. Input: current event, state summary, active goals, <=5 related memories.
2. LLM outputs Appraisal JSON.
3. Pydantic validates ranges and IDs.
4. Remove unsupported evidence references.
5. Low certainty when no supporting evidence (and not trivial greeting).
6. Save appraisal with prompt/model version.
7. Pass result to deterministic state engine.

Failure: LLM timeout, JSON error, or evidence validation failure → neutral appraisal.
"""

from __future__ import annotations

import logging
from typing import Any

from ssa.adapters.llm import ChatMessage, LLMAdapter, LLMRequest
from ssa.clock import Clock
from ssa.config import ThinkingMode
from ssa.domain.appraisal import AppraisalResult, StoredAppraisal
from ssa.domain.events import Event
from ssa.domain.memories import RetrievedMemory
from ssa.ids import IdGenerator
from ssa.storage.appraisal_repository import SqliteAppraisalRepository

logger = logging.getLogger(__name__)

_REQUIRED_NUMERIC_FIELDS = (
    "novelty",
    "goal_congruence",
    "controllability",
    "certainty",
    "self_agency",
    "user_agency",
    "external_agency",
    "relationship_relevance",
    "urgency",
    "valence_signal",
    "arousal_signal",
)


class AppraisalService:
    """Evaluates events into structured appraisals (M07)."""

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        repository: SqliteAppraisalRepository | None = None,
        ids: IdGenerator | None = None,
        clock: Clock | None = None,
        prompt_version: str = "appraisal_v1",
        default_model: str = "deepseek/deepseek-v4-flash",
    ) -> None:
        self._llm = llm
        persistence_parts = (repository, ids, clock)
        if any(part is not None for part in persistence_parts) and not all(
            part is not None for part in persistence_parts
        ):
            raise ValueError("repository, ids, and clock must be provided together")
        self._repository = repository
        self._ids = ids
        self._clock = clock
        self._prompt_version = prompt_version
        self._default_model = default_model

    async def evaluate(
        self,
        event: Event,
        state_summary: str,
        active_goals: list[str],
        memories: list[RetrievedMemory],
    ) -> AppraisalResult:
        """Generate an appraisal for the given event.

        Pipeline §16.3.
        """
        # Build the context description.
        mem_desc = (
            "\n".join(f"- [{m.source_kind.value}] {m.content}" for m in memories[:5])
            or "(no relevant memories)"
        )

        goals_desc = "\n".join(f"- {g}" for g in active_goals) or "(no active goals)"

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "novelty": {"type": "number"},
                "goal_congruence": {"type": "number"},
                "controllability": {"type": "number"},
                "certainty": {"type": "number"},
                "self_agency": {"type": "number"},
                "user_agency": {"type": "number"},
                "external_agency": {"type": "number"},
                "relationship_relevance": {"type": "number"},
                "urgency": {"type": "number"},
                "valence_signal": {"type": "number"},
                "arousal_signal": {"type": "number"},
                "supported_event_ids": {"type": "array", "items": {"type": "string"}},
                "supported_memory_ids": {"type": "array", "items": {"type": "string"}},
                "explanation": {"type": "string"},
            },
            "required": [
                "novelty",
                "goal_congruence",
                "controllability",
                "certainty",
                "self_agency",
                "user_agency",
                "external_agency",
                "relationship_relevance",
                "urgency",
                "valence_signal",
                "arousal_signal",
                "explanation",
            ],
        }

        request = LLMRequest(
            purpose="appraisal",
            messages=[
                ChatMessage.system(
                    "You are an appraisal system for a digital lifeform. "
                    "Evaluate the significance of the incoming event. "
                    "Output JSON with the appraisal fields. "
                    "All values in [0,1] except goal_congruence and valence_signal in [-1,1]."
                ),
                ChatMessage.user(
                    f"Current event:\n{event.content}\n\n"
                    f"State summary:\n{state_summary}\n\n"
                    f"Active goals:\n{goals_desc}\n\n"
                    f"Related memories:\n{mem_desc}"
                ),
            ],
            model=self._default_model,
            temperature=0.3,
            max_tokens=512,
            thinking=ThinkingMode.DISABLED,
            prompt_version=self._prompt_version,
            json_schema=schema,
        )

        try:
            response = await self._llm.complete(request)
        except Exception as exc:
            logger.warning("Appraisal LLM call failed, using neutral: %s", exc)
            result = AppraisalResult.neutral()
            self._persist(result, event, provider="fallback", model=self._default_model)
            return result

        if response.parsed is None:
            logger.warning("Appraisal LLM returned no parsed JSON, using neutral")
            result = AppraisalResult.neutral()
            self._persist(result, event, provider="fallback", model=response.model)
            return result

        validated = self._validate_and_build(response.parsed, event, memories)
        if validated is None:
            result = AppraisalResult.neutral()
            self._persist(result, event, provider="fallback", model=response.model)
            return result

        self._persist(validated, event, provider=response.provider, model=response.model)
        return validated

    def _persist(
        self,
        result: AppraisalResult,
        event: Event,
        *,
        provider: str,
        model: str,
    ) -> None:
        if self._repository is None or self._ids is None or self._clock is None:
            return
        self._repository.insert(
            StoredAppraisal(
                id=self._ids.new(),
                correlation_id=event.correlation_id,
                cause_event_id=event.id,
                result=result,
                provider=provider,
                model=model,
                prompt_version=self._prompt_version,
                created_at_ms=self._clock.now_ms(),
            )
        )

    def _validate_and_build(
        self,
        data: dict[str, Any],
        event: Event,
        memories: list[RetrievedMemory],
    ) -> AppraisalResult | None:
        """Validate ranges and evidence references (pipeline §16.3 steps 3-5)."""
        try:
            missing = [name for name in _REQUIRED_NUMERIC_FIELDS if name not in data]
            if missing or "explanation" not in data:
                raise ValueError(f"missing required fields: {missing}")

            event_ids = _read_string_list(data, "supported_event_ids")
            memory_ids = _read_string_list(data, "supported_memory_ids")
            valid_event_ids = {event.id}
            valid_memory_ids = {memory.memory_id for memory in memories}
            supported_events = [eid for eid in event_ids if eid in valid_event_ids]
            supported_memories = [mid for mid in memory_ids if mid in valid_memory_ids]

            # Step 5: lower certainty only when neither event nor memory evidence exists.
            trivial_messages = {"hi", "hey", "hello", "ok", "嗯", "好", "你好"}
            is_trivial = event.content.strip().lower() in trivial_messages
            certainty = float(data["certainty"])
            if not supported_events and not supported_memories and not is_trivial:
                certainty = min(certainty, 0.3)

            return AppraisalResult(
                novelty=float(data["novelty"]),
                goal_congruence=float(data["goal_congruence"]),
                controllability=float(data["controllability"]),
                certainty=certainty,
                self_agency=float(data["self_agency"]),
                user_agency=float(data["user_agency"]),
                external_agency=float(data["external_agency"]),
                relationship_relevance=float(data["relationship_relevance"]),
                urgency=float(data["urgency"]),
                valence_signal=float(data["valence_signal"]),
                arousal_signal=float(data["arousal_signal"]),
                supported_event_ids=supported_events,
                supported_memory_ids=supported_memories,
                explanation=str(data["explanation"]),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Appraisal validation failed: %s, using neutral", exc)
            return None


def _read_string_list(data: dict[str, Any], field_name: str) -> list[str]:
    value = data.get(field_name, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings")
    return value


__all__ = ["AppraisalService"]
