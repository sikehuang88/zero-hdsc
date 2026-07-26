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
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.events import Event
from ssa.domain.memories import RetrievedMemory

logger = logging.getLogger(__name__)


class AppraisalService:
    """Evaluates events into structured appraisals (M07)."""

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        prompt_version: str = "appraisal_v1",
        default_model: str = "deepseek/deepseek-chat",
    ) -> None:
        self._llm = llm
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
        mem_desc = "\n".join(
            f"- [{m.source_kind.value}] {m.content}" for m in memories[:5]
        ) or "(no relevant memories)"

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
                "novelty", "goal_congruence", "controllability", "certainty",
                "self_agency", "user_agency", "external_agency",
                "relationship_relevance", "urgency", "valence_signal",
                "arousal_signal", "explanation",
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
            prompt_version=self._prompt_version,
            json_schema=schema,
        )

        try:
            response = await self._llm.complete(request)
        except Exception as exc:
            logger.warning("Appraisal LLM call failed, using neutral: %s", exc)
            return AppraisalResult.neutral()

        if response.parsed is None:
            logger.warning("Appraisal LLM returned no parsed JSON, using neutral")
            return AppraisalResult.neutral()

        return self._validate_and_build(response.parsed, event, memories)

    def _validate_and_build(
        self,
        data: dict[str, Any],
        event: Event,
        memories: list[RetrievedMemory],
    ) -> AppraisalResult:
        """Validate ranges and evidence references (pipeline §16.3 steps 3-5)."""
        valid_event_ids = {event.id}
        valid_memory_ids = {m.memory_id for m in memories}

        # Filter supported IDs to only valid ones.
        supported_events = [
            eid for eid in data.get("supported_event_ids", [])
            if eid in valid_event_ids
        ]
        supported_memories = [
            mid for mid in data.get("supported_memory_ids", [])
            if mid in valid_memory_ids
        ]

        # Step 5: low certainty when no supporting evidence and not trivial.
        is_trivial = len(event.content) < 10 and any(
            word in event.content.lower()
            for word in ["hi", "hey", "hello", "ok", "嗯", "好"]
        )
        certainty = float(data.get("certainty", 0.5))
        if not supported_events and not is_trivial:
            certainty = min(certainty, 0.3)

        try:
            return AppraisalResult(
                novelty=float(data.get("novelty", 0.5)),
                goal_congruence=float(data.get("goal_congruence", 0.0)),
                controllability=float(data.get("controllability", 0.5)),
                certainty=certainty,
                self_agency=float(data.get("self_agency", 0.3)),
                user_agency=float(data.get("user_agency", 0.3)),
                external_agency=float(data.get("external_agency", 0.3)),
                relationship_relevance=float(data.get("relationship_relevance", 0.3)),
                urgency=float(data.get("urgency", 0.2)),
                valence_signal=float(data.get("valence_signal", 0.0)),
                arousal_signal=float(data.get("arousal_signal", 0.2)),
                supported_event_ids=supported_events,
                supported_memory_ids=supported_memories,
                explanation=str(data.get("explanation", "")),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Appraisal validation failed: %s, using neutral", exc)
            return AppraisalResult.neutral()


__all__ = ["AppraisalService"]
