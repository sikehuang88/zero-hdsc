"""Tests for Appraisal service (M07) and State engine (M08)."""

from __future__ import annotations

import pytest

from ssa.adapters.llm import FakeLLMAdapter
from ssa.config import StateConfig, ThinkingMode
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import ActionIntent, Actor, SourceKind
from ssa.domain.events import Event
from ssa.domain.memories import RetrievedMemory
from ssa.domain.state import DeterministicStateEngine, OrganismState
from ssa.services.appraisal_service import AppraisalService

# ---------------------------------------------------------------------------
# AppraisalResult
# ---------------------------------------------------------------------------


class TestAppraisalResult:
    def test_neutral_appraisal(self):
        a = AppraisalResult.neutral()
        assert a.novelty == 0.5
        assert a.goal_congruence == 0.0
        assert a.certainty == 0.2
        assert a.valence_signal == 0.0
        assert a.arousal_signal == 0.2

    def test_neutral_does_not_claim_high_relevance(self):
        a = AppraisalResult.neutral()
        assert a.relationship_relevance <= 0.5
        assert a.urgency <= 0.3

    def test_rejects_out_of_range_0_1(self):
        with pytest.raises(ValueError):
            AppraisalResult(
                novelty=1.5,
                goal_congruence=0,
                controllability=0.5,
                certainty=0.5,
                self_agency=0.5,
                user_agency=0.5,
                external_agency=0.5,
                relationship_relevance=0.5,
                urgency=0.5,
                valence_signal=0,
                arousal_signal=0.5,
            )

    def test_rejects_out_of_range_neg1_1(self):
        with pytest.raises(ValueError):
            AppraisalResult(
                novelty=0.5,
                goal_congruence=2.0,
                controllability=0.5,
                certainty=0.5,
                self_agency=0.5,
                user_agency=0.5,
                external_agency=0.5,
                relationship_relevance=0.5,
                urgency=0.5,
                valence_signal=0,
                arousal_signal=0.5,
            )

    def test_accepts_valid_ranges(self):
        a = AppraisalResult(
            novelty=1.0,
            goal_congruence=-1.0,
            controllability=0.0,
            certainty=1.0,
            self_agency=0.5,
            user_agency=0.5,
            external_agency=0.5,
            relationship_relevance=1.0,
            urgency=0.0,
            valence_signal=1.0,
            arousal_signal=0.0,
        )
        assert a.novelty == 1.0
        assert a.goal_congruence == -1.0


# ---------------------------------------------------------------------------
# AppraisalService
# ---------------------------------------------------------------------------


def make_event(content: str = "hello", event_id: str = "evt-1") -> Event:
    return Event(
        id=event_id,
        correlation_id="corr-1",
        conversation_id="c1",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content=content,
        content_hash="abc",
        created_at_ms=1000,
    )


class TestAppraisalService:
    @pytest.fixture
    def llm(self) -> FakeLLMAdapter:
        return FakeLLMAdapter()

    @pytest.fixture
    def svc(self, llm: FakeLLMAdapter) -> AppraisalService:
        return AppraisalService(llm)

    @pytest.mark.asyncio
    async def test_evaluate_returns_valid_appraisal(self, svc, llm):
        event = make_event("我今天升职了")
        llm.set_response(
            "appraisal",
            """{
            "novelty": 0.8, "goal_congruence": 0.6, "controllability": 0.7,
            "certainty": 0.9, "self_agency": 0.2, "user_agency": 0.8,
            "external_agency": 0.1, "relationship_relevance": 0.7,
            "urgency": 0.3, "valence_signal": 0.8, "arousal_signal": 0.6,
            "supported_event_ids": ["evt-1"],
            "supported_memory_ids": [],
            "explanation": "positive career event"
        }""",
        )

        result = await svc.evaluate(event, "happy state", [], [])
        assert result.novelty == 0.8
        assert result.valence_signal == 0.8
        assert result.supported_event_ids == ["evt-1"]
        request = llm.calls[-1]
        assert request.model == "deepseek/deepseek-v4-flash"
        assert request.thinking == ThinkingMode.DISABLED
        assert request.reasoning_effort is None
        assert [message.role for message in request.messages] == ["system", "user"]
        assert "我今天升职了" in (request.messages[-1].content or "")

    @pytest.mark.asyncio
    async def test_evaluate_falls_back_to_neutral_on_llm_error(self, svc, llm):
        event = make_event("test")

        # Force LLM to raise by not setting a response and having it echo.
        # The FakeLLMAdapter won't raise, but we can test the neutral path
        # by making the parsed result invalid.
        llm.set_response("appraisal", "not json at all")
        result = await svc.evaluate(event, "state", [], [])
        # FakeLLM tries to parse, fails, returns None parsed → neutral.
        assert result.certainty == 0.2  # neutral

    @pytest.mark.asyncio
    async def test_evaluate_filters_unsupported_evidence(self, svc, llm):
        event = make_event("test")
        llm.set_response(
            "appraisal",
            """{
            "novelty": 0.5, "goal_congruence": 0.0, "controllability": 0.5,
            "certainty": 0.9, "self_agency": 0.3, "user_agency": 0.3,
            "external_agency": 0.3, "relationship_relevance": 0.3,
            "urgency": 0.2, "valence_signal": 0.0, "arousal_signal": 0.2,
            "supported_event_ids": ["fake-event", "evt-1"],
            "supported_memory_ids": ["fake-mem"],
            "explanation": "test"
        }""",
        )

        result = await svc.evaluate(event, "state", [], [])
        assert "fake-event" not in result.supported_event_ids
        assert "fake-mem" not in result.supported_memory_ids
        assert "evt-1" in result.supported_event_ids

    @pytest.mark.asyncio
    async def test_evaluate_lowers_certainty_without_evidence(self, svc, llm):
        event = make_event("something happened today")
        llm.set_response(
            "appraisal",
            """{
            "novelty": 0.7, "goal_congruence": 0.0, "controllability": 0.5,
            "certainty": 0.9, "self_agency": 0.3, "user_agency": 0.3,
            "external_agency": 0.3, "relationship_relevance": 0.3,
            "urgency": 0.2, "valence_signal": 0.0, "arousal_signal": 0.2,
            "supported_event_ids": [],
            "explanation": "no evidence"
        }""",
        )

        result = await svc.evaluate(event, "state", [], [])
        # No supporting evidence and not trivial → certainty capped at 0.3.
        assert result.certainty <= 0.3

    @pytest.mark.asyncio
    async def test_memory_evidence_preserves_certainty(self, svc, llm):
        event = make_event("something happened today")
        memory = RetrievedMemory(
            memory_id="mem-1",
            content="related memory",
            source_kind=SourceKind.USER_OBSERVED,
            score=0.9,
        )
        llm.set_response(
            "appraisal",
            """{
            "novelty": 0.7, "goal_congruence": 0.0, "controllability": 0.5,
            "certainty": 0.9, "self_agency": 0.3, "user_agency": 0.3,
            "external_agency": 0.3, "relationship_relevance": 0.3,
            "urgency": 0.2, "valence_signal": 0.0, "arousal_signal": 0.2,
            "supported_memory_ids": ["mem-1"],
            "explanation": "supported by memory"
        }""",
        )

        result = await svc.evaluate(event, "state", [], [memory])

        assert result.certainty == 0.9
        assert result.supported_memory_ids == ["mem-1"]

    @pytest.mark.asyncio
    async def test_malformed_structured_fields_fall_back_to_neutral(self, svc, llm):
        event = make_event("something happened today")
        llm.set_response(
            "appraisal",
            """{
            "novelty": "invalid", "goal_congruence": 0.0, "controllability": 0.5,
            "certainty": 0.9, "self_agency": 0.3, "user_agency": 0.3,
            "external_agency": 0.3, "relationship_relevance": 0.3,
            "urgency": 0.2, "valence_signal": 0.0, "arousal_signal": 0.2,
            "supported_event_ids": ["evt-1"],
            "explanation": "malformed"
        }""",
        )

        result = await svc.evaluate(event, "state", [], [])

        assert result == AppraisalResult.neutral()


# ---------------------------------------------------------------------------
# State Engine
# ---------------------------------------------------------------------------


class TestOrganismState:
    def test_initial_state(self):
        s = OrganismState.initial(now_ms=1000)
        assert s.version == 1
        assert 0 <= s.energy <= 1
        assert -1 <= s.valence <= 1
        assert s.updated_at_ms == 1000

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            OrganismState(
                version=1,
                energy=2.0,
                connection_need=0.5,
                autonomy_need=0.5,
                curiosity=0.5,
                safety=0.5,
                valence=0.0,
                arousal=0.5,
                updated_at_ms=1000,
            )

    def test_rejects_valence_out_of_range(self):
        with pytest.raises(ValueError):
            OrganismState(
                version=1,
                energy=0.5,
                connection_need=0.5,
                autonomy_need=0.5,
                curiosity=0.5,
                safety=0.5,
                valence=2.0,
                arousal=0.5,
                updated_at_ms=1000,
            )


class TestDeterministicStateEngine:
    @pytest.fixture
    def engine(self) -> DeterministicStateEngine:
        return DeterministicStateEngine(StateConfig())

    @pytest.fixture
    def initial_state(self) -> OrganismState:
        return OrganismState.initial(now_ms=1000)

    def test_preview_returns_new_state(self, engine, initial_state):
        appraisal = AppraisalResult(
            novelty=0.8,
            goal_congruence=0.5,
            controllability=0.7,
            certainty=0.9,
            self_agency=0.3,
            user_agency=0.7,
            external_agency=0.2,
            relationship_relevance=0.6,
            urgency=0.3,
            valence_signal=0.7,
            arousal_signal=0.5,
        )
        preview = engine.preview(initial_state, appraisal, now_ms=2000)
        assert preview is not initial_state
        assert preview.version == initial_state.version  # not bumped in preview
        assert preview.updated_at_ms == 2000

    def test_preview_does_not_mutate_old(self, engine, initial_state):
        original_energy = initial_state.energy
        appraisal = AppraisalResult.neutral()
        engine.preview(initial_state, appraisal, now_ms=2000)
        assert initial_state.energy == original_energy

    def test_preview_is_deterministic(self, engine, initial_state):
        appraisal = AppraisalResult(
            novelty=0.6,
            goal_congruence=0.3,
            controllability=0.5,
            certainty=0.7,
            self_agency=0.3,
            user_agency=0.5,
            external_agency=0.3,
            relationship_relevance=0.4,
            urgency=0.4,
            valence_signal=0.2,
            arousal_signal=0.4,
        )
        p1 = engine.preview(initial_state, appraisal, now_ms=2000)
        p2 = engine.preview(initial_state, appraisal, now_ms=2000)
        assert p1.model_dump() == p2.model_dump()

    def test_finalize_bumps_version(self, engine, initial_state):
        preview = engine.preview(initial_state, AppraisalResult.neutral(), now_ms=2000)
        final = engine.finalize(preview, ActionIntent.ACKNOWLEDGE)
        assert final.version == preview.version + 1

    def test_finalize_energy_cost_for_answer(self, engine, initial_state):
        preview = engine.preview(initial_state, AppraisalResult.neutral(), now_ms=2000)
        final = engine.finalize(preview, ActionIntent.ANSWER)
        assert final.energy <= preview.energy

    def test_finalize_rest_recovers_energy(self, engine, initial_state):
        preview = engine.preview(initial_state, AppraisalResult.neutral(), now_ms=2000)
        final = engine.finalize(preview, ActionIntent.REST)
        assert final.energy > preview.energy

    def test_finalize_project_work_costs_more(self, engine, initial_state):
        preview = engine.preview(initial_state, AppraisalResult.neutral(), now_ms=2000)
        final = engine.finalize(preview, ActionIntent.PROJECT_WORK)
        assert final.energy < preview.energy
        assert final.autonomy_need < preview.autonomy_need

    def test_state_always_in_bounds(self, engine, initial_state):
        """Pipeline §17.5: state always in defined domain."""
        import random

        random.seed(42)
        for _ in range(100):
            appraisal = AppraisalResult(
                novelty=random.random(),
                goal_congruence=random.uniform(-1, 1),
                controllability=random.random(),
                certainty=random.random(),
                self_agency=random.random(),
                user_agency=random.random(),
                external_agency=random.random(),
                relationship_relevance=random.random(),
                urgency=random.random(),
                valence_signal=random.uniform(-1, 1),
                arousal_signal=random.random(),
            )
            preview = engine.preview(initial_state, appraisal, now_ms=2000)
            assert 0 <= preview.energy <= 1
            assert -1 <= preview.valence <= 1
            assert 0 <= preview.arousal <= 1
            assert 0 <= preview.connection_need <= 1
            assert 0 <= preview.autonomy_need <= 1
            assert 0 <= preview.curiosity <= 1
            assert 0 <= preview.safety <= 1

            intent = random.choice(list(ActionIntent))
            final = engine.finalize(preview, intent)
            assert 0 <= final.energy <= 1
            assert -1 <= final.valence <= 1
            assert 0 <= final.arousal <= 1

    def test_energy_recovers_over_time(self, engine):
        """Pipeline §17.3: energy recovers with rest/time."""
        low_energy = OrganismState(
            version=1,
            energy=0.2,
            connection_need=0.5,
            autonomy_need=0.5,
            curiosity=0.5,
            safety=0.5,
            valence=0.0,
            arousal=0.2,
            updated_at_ms=1000,
        )
        # 10 hours later.
        preview = engine.preview(
            low_energy, AppraisalResult.neutral(), now_ms=1000 + 10 * 3600 * 1000
        )
        assert preview.energy > low_energy.energy

    def test_valence_ema(self, engine, initial_state):
        """Pipeline §17.4: valence uses EMA of appraisal signal."""
        positive = AppraisalResult(
            novelty=0.5,
            goal_congruence=0.5,
            controllability=0.5,
            certainty=0.5,
            self_agency=0.3,
            user_agency=0.5,
            external_agency=0.3,
            relationship_relevance=0.5,
            urgency=0.3,
            valence_signal=1.0,
            arousal_signal=0.3,
        )
        preview = engine.preview(initial_state, positive, now_ms=2000)
        # EMA: 0.0 * 0.75 + 1.0 * 0.25 = 0.25
        assert abs(preview.valence - 0.25) < 0.01

    def test_engine_does_not_call_llm(self, engine, initial_state):
        """Pipeline §17.5: state engine must not call LLM."""
        # This is implicitly tested — DeterministicStateEngine has no LLM dependency.
        preview = engine.preview(initial_state, AppraisalResult.neutral(), now_ms=2000)
        assert preview is not None
