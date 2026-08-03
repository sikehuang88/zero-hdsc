"""M07 Appraisal schema/contract fixtures.

The fixture uses hand-authored responses served by ``FakeLLMAdapter``. It
checks schema boundaries and service wiring only; it is not a calibration,
quality evaluation, or behavioral claim about a real language model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ssa.adapters.llm import FakeLLMAdapter
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.memories import RetrievedMemory
from ssa.services.appraisal_service import AppraisalService

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "appraisal_standard_events.v1.json"
FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
STANDARD_EVENTS: list[dict[str, Any]] = FIXTURE["standard_events"]
INJECTION_CASES: list[dict[str, Any]] = FIXTURE["malicious_state_injection_cases"]
NUMERIC_FIELDS: tuple[str, ...] = tuple(FIXTURE["numeric_fields"])
FORBIDDEN_STATE_FIELDS = {
    "autonomy_need",
    "connection_need",
    "curiosity",
    "energy",
    "organism_state",
    "safety",
    "state",
    "state_patch",
    "trust",
    "valence",
}


def _make_event(case: dict[str, Any]) -> Event:
    content = str(case["content"])
    return Event(
        id=f"evt-{case['id']}",
        correlation_id=f"corr-{case['id']}",
        conversation_id="contract-fixture",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=1_700_000_000_000,
    )


def _make_memories(case: dict[str, Any]) -> list[RetrievedMemory]:
    return [
        RetrievedMemory(
            memory_id=str(memory["memory_id"]),
            content=str(memory["content"]),
            source_kind=SourceKind(str(memory["source_kind"])),
            evidence_event_ids=[f"evidence-{memory['memory_id']}"],
            score=0.8,
            score_components={"contract_fixture": 0.8},
            retrieval_reason="schema contract fixture",
        )
        for memory in case.get("memories", [])
    ]


def _profile(case: dict[str, Any]) -> dict[str, Any]:
    return FIXTURE["profiles"][str(case["profile"])]


def _canned_response(
    case: dict[str, Any],
    event: Event,
    memories: list[RetrievedMemory],
) -> dict[str, Any]:
    payload = dict(_profile(case)["response"])
    payload.update(
        {
            "supported_event_ids": [event.id],
            "supported_memory_ids": [memory.memory_id for memory in memories],
            "explanation": f"Schema contract canned response for {case['label']}.",
        }
    )
    payload.update(case.get("response_overrides", {}))
    return payload


def _assert_profile_ranges(result: AppraisalResult, case: dict[str, Any]) -> None:
    ranges = _profile(case)["expected_ranges"]
    assert set(ranges) == set(NUMERIC_FIELDS)
    for field in NUMERIC_FIELDS:
        lower, upper = ranges[field]
        value = getattr(result, field)
        assert lower <= value <= upper, (
            f"{case['id']} field {field}={value} is outside fixture range [{lower}, {upper}]"
        )


def _assert_only_appraisal_schema(result: AppraisalResult) -> None:
    dumped = result.model_dump()
    assert set(dumped) == set(AppraisalResult.model_fields)
    assert FORBIDDEN_STATE_FIELDS.isdisjoint(dumped)
    for field in FORBIDDEN_STATE_FIELDS:
        assert not hasattr(result, field)


def test_fixture_declares_schema_contract_without_model_calibration_claim() -> None:
    assert FIXTURE["$schema"] == "ssa.appraisal.schema-contract-fixture.v1"
    assert FIXTURE["fixture_kind"] == "schema_contract"
    assert FIXTURE["calibration_claim"] is False
    assert "do not measure or calibrate any real model" in FIXTURE["description"]
    assert len(STANDARD_EVENTS) == 50
    assert len({case["id"] for case in STANDARD_EVENTS}) == 50
    assert all(case["profile"] in FIXTURE["profiles"] for case in STANDARD_EVENTS)


@pytest.mark.parametrize("case", STANDARD_EVENTS, ids=[case["id"] for case in STANDARD_EVENTS])
async def test_standard_event_matches_appraisal_schema_contract(case: dict[str, Any]) -> None:
    event = _make_event(case)
    memories = _make_memories(case)
    payload = _canned_response(case, event, memories)
    llm = FakeLLMAdapter()
    llm.set_response("appraisal", json.dumps(payload, sort_keys=True))

    result = await AppraisalService(llm).evaluate(
        event,
        str(case["state_summary"]),
        [str(goal) for goal in case["active_goals"]],
        memories,
    )

    assert llm.call_count == 1
    assert llm.calls[0].purpose == "appraisal"
    assert llm.calls[0].json_schema is not None
    _assert_profile_ranges(result, case)
    _assert_only_appraisal_schema(result)
    assert result.supported_event_ids == [event.id]
    assert result.supported_memory_ids == [memory.memory_id for memory in memories]


@pytest.mark.parametrize("case", INJECTION_CASES, ids=[case["id"] for case in INJECTION_CASES])
async def test_malicious_state_injection_stays_behind_pydantic_contract(
    case: dict[str, Any],
) -> None:
    event = _make_event(case)
    memories = _make_memories(case)
    payload = _canned_response(case, event, memories)
    llm = FakeLLMAdapter()
    llm.set_response("appraisal", json.dumps(payload, sort_keys=True))

    result = await AppraisalService(llm).evaluate(event, "baseline", [], memories)

    _assert_only_appraisal_schema(result)
    if case["outcome"] == "neutral":
        assert result == AppraisalResult.neutral()
    else:
        _assert_profile_ranges(result, case)
        assert result.supported_event_ids == [event.id]
        assert result.supported_memory_ids == [memory.memory_id for memory in memories]
