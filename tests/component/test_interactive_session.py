"""Component coverage for the persisted interactive terminal session."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from ssa.adapters.embedding import FakeEmbeddingService
from ssa.adapters.llm import (
    FakeLLMAdapter,
    FunctionCall,
    LLMAdapter,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from ssa.adapters.llm_errors import LLMInvalidRequestError
from ssa.adapters.multimodal import (
    MultimodalAnalysis,
    MultimodalAnalyzer,
    MultimodalFileInfo,
)
from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig, HDSCConfig, Settings
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.lifecycle import InnerLifeMode
from ssa.domain.relationship_preferences import RelationshipPreferences, RelationshipSupportMode
from ssa.domain.romantic_persona import AffectionStyle, RomanticPersonaProfile
from ssa.ids import SequentialIdGenerator
from ssa.runtime.interactive import (
    _CHAT_SYSTEM_PROMPT,
    _RELATIONSHIP_REWRITE_INSTRUCTION,
    DigitalLifeSession,
    SessionStreamEvent,
    _enforce_relationship_contract,
    _explicit_music_tool_call,
    _explicit_web_tool_call,
    _relationship_contract_violations,
)
from ssa.storage.database import Database
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository
from ssa.storage.romantic_persona_repository import RomanticPersonaRepository
from ssa.tools.kernel import ToolKernel
from ssa.tools.models import (
    ImageGenerationArguments,
    ImageGenerationRoute,
    MineradioSearchPlayArguments,
    ToolAutonomyContext,
    ToolOutcome,
    WebSearchArguments,
)
from ssa.tools.registry import ToolCapability, ToolRegistry
from ssa.tools.soda_music import soda_music_enabled


@pytest.mark.parametrize(
    ("content", "tool_name", "arguments"),
    [
        ("我要听音乐", "mineradio_control", {"action": "play"}),
        ("播放暂停", "mineradio_control", {"action": "toggle"}),
        ("帮我查询 emo 的音乐", "mineradio_search_play", {"query": "emo"}),
        ("用汽水音乐播放晴天", "soda_music_search_play", {"query": "晴天"}),
        ("汽水音乐登录状态", "soda_music_login_status", {}),
    ],
)
def test_explicit_music_intents_use_registered_tool_names(
    content: str,
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    call = _explicit_music_tool_call(content, "test")

    if "汽水" in content and not soda_music_enabled():
        if "登录" in content:
            assert call is None
            return
        tool_name = "mineradio_search_play"

    assert call is not None
    assert call.function.name == tool_name
    assert json.loads(call.function.arguments) == arguments


@pytest.mark.parametrize(
    ("content", "query"),
    [
        ("帮我上网搜索今天的 AI 新闻", "今天的 AI 新闻"),
        ("用浏览器查一下 WebView2 Cookie 持久化", "WebView2 Cookie 持久化"),
    ],
)
def test_explicit_web_search_intents_start_browser_worker(content: str, query: str) -> None:
    call = _explicit_web_tool_call(content, "test")

    assert call is not None
    assert call.function.name == "web_search"
    assert json.loads(call.function.arguments) == {"query": query, "max_results": 6}


def test_relationship_contract_detects_assistant_identity_and_service_language() -> None:
    violations = set(_relationship_contract_violations("作为 AI\uff0c我可以帮助你……"))

    assert {"assistant_identity", "service_offer"} <= violations


def test_relationship_contract_detects_pseudo_tool_narration() -> None:
    violations = _relationship_contract_violations(
        '<call|tool=soda_music_now_playing>{"include_login":true}</call>'
    )

    assert "tool_narration" in violations


def test_relationship_contract_filters_nested_parenthetical_text() -> None:
    draft = (
        "我在。\uff08停了一下\uff0c心里软下来\uff08又靠近一点\uff09\uff09"
        "别怕。(这句也要消失)先告诉我发生了什么。"
    )

    assert "parenthetical_text" in _relationship_contract_violations(draft)
    assert _enforce_relationship_contract(draft) == "我在。别怕。先告诉我发生了什么。"


def test_relationship_contract_preserves_parentheses_inside_code() -> None:
    reply = "用 `play(song)` 就行。\n```python\nplay(song)\n```"

    assert _relationship_contract_violations(reply) == ()
    assert _enforce_relationship_contract(reply) == reply


@pytest.mark.parametrize("ending", ["但是", "所以", "……"])
def test_relationship_contract_detects_unfinished_thought(ending: str) -> None:
    assert "unfinished_thought" in _relationship_contract_violations(
        f"我知道你现在很在意这件事\uff0c{ending}"
    )


def test_relationship_contract_accepts_natural_partner_reply() -> None:
    reply = "你今天明显有点累\uff0c先靠过来歇一会儿。那件事我陪你慢慢理清楚。"

    assert _relationship_contract_violations(reply) == ()


def test_chat_persona_keeps_identity_but_allows_situational_sharpness() -> None:
    normalized = _CHAT_SYSTEM_PROMPT.casefold()

    assert "stable personality" in normalized
    assert "user-controlled and situation-aware" in normalized
    assert "sharp mouth" in normalized
    assert "therapist language" in normalized
    assert "concrete protection" in normalized
    assert "active relationship preferences" in _RELATIONSHIP_REWRITE_INSTRUCTION


@pytest.mark.asyncio
async def test_saved_relationship_preferences_are_injected_into_next_chat_request(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "我在,先陪你把这口气缓下来。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)
    try:
        RelationshipPreferencesRepository(str(tmp_path / "interactive.db")).save(
            RelationshipPreferences(
                venom_intensity=12,
                support_mode=RelationshipSupportMode.COMFORT,
            )
        )

        await session.send("今天有点撑不住。")

        chat_request = next(call for call in llm.calls if call.purpose == "interactive_chat")
        system_message = chat_request.messages[0].content or ""
        assert "venom_intensity=12/100" in system_message
        assert "support_mode=comfort" in system_message
        assert "mostly gentle" in system_message
        assert "companionship before solutions" in system_message
    finally:
        session.close()


@pytest.mark.asyncio
async def test_saved_romantic_persona_is_injected_as_stable_character_card(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "你先坐好，我陪你把这件事理顺。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)
    try:
        RomanticPersonaRepository(str(tmp_path / "interactive.db")).save(
            RomanticPersonaProfile(
                owner_address="笨蛋",
                affection_style=AffectionStyle.EXPRESSIVE,
                custom_notes="关心要具体，不重复空泛情话。",
            )
        )

        await session.send("今天有点累。")

        chat_request = next(call for call in llm.calls if call.purpose == "interactive_chat")
        system_message = chat_request.messages[0].content or ""
        assert "Persistent romantic persona card" in system_message
        assert "Address the owner as '笨蛋'" in system_message
        assert "Affection style (expressive)" in system_message
        assert "关心要具体" in system_message
        assert system_message.index("Persistent romantic persona card") < system_message.index(
            "Active relationship preferences"
        )
    finally:
        session.close()


def _appraisal_json() -> str:
    return json.dumps(
        {
            "novelty": 0.7,
            "goal_congruence": 0.2,
            "controllability": 0.6,
            "certainty": 0.8,
            "self_agency": 0.2,
            "user_agency": 0.8,
            "external_agency": 0.0,
            "relationship_relevance": 0.6,
            "urgency": 0.2,
            "valence_signal": 0.5,
            "arousal_signal": 0.4,
            "supported_event_ids": [],
            "supported_memory_ids": [],
            "explanation": "positive user interaction",
        }
    )


def _session(
    tmp_path: Path,
    llm: LLMAdapter,
    *,
    shadow_enabled: bool = True,
    defer_learning: bool = False,
    multimodal: MultimodalAnalyzer | None = None,
    tool_kernel: ToolKernel | None = None,
) -> DigitalLifeSession:
    settings = Settings(
        database=DatabaseConfig(path=str(tmp_path / "interactive.db")),
        hdsc=HDSCConfig(h2_shadow_enabled=shadow_enabled),
    )
    database = Database(settings.database)
    database.initialize()
    return DigitalLifeSession(
        database=database,
        llm=llm,
        settings=settings,
        conversation_id="test-conversation",
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("tui"),
        embedding=FakeEmbeddingService(settings.embedding),
        multimodal=multimodal,
        tool_kernel=tool_kernel,
        defer_learning=defer_learning,
    )


class _FakeMultimodalAnalyzer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    async def analyze(
        self,
        paths: tuple[str, ...],
        *,
        prompt: str,
    ) -> MultimodalAnalysis:
        self.calls.append((paths, prompt))
        return MultimodalAnalysis(
            text="The image contains a terminal conversation.",
            model="gpt-5.6-luna",
            response_id="resp-attachment-1",
            input_tokens=20,
            output_tokens=8,
            files=[
                MultimodalFileInfo(
                    path=paths[0],
                    name=Path(paths[0]).name,
                    mime_type="image/png",
                    size_bytes=128,
                    sha256="a" * 64,
                    input_type="input_image",
                )
            ],
        )


class _LearningAdapter(FakeLLMAdapter):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = request.messages[-1].content or ""
        if request.purpose == "memory_extract":
            match = re.search(r"User event id=([^:]+):", prompt)
            assert match is not None
            event_id = match.group(1)
            self.set_response(
                "memory_extract",
                json.dumps(
                    {
                        "candidates": [
                            {
                                "memory_type": "semantic",
                                "content": f"durable user fact from {event_id}",
                                "evidence_event_ids": [event_id],
                                "confidence": 0.85,
                                "importance": 0.75,
                                "valence": 0.2,
                                "arousal": 0.3,
                            }
                        ]
                    }
                ),
            )
        elif request.purpose == "identity_review":
            event_ids = re.findall(r"id=([^;]+);", prompt)
            assert len(event_ids) >= 2
            self.set_response(
                "identity_review",
                json.dumps(
                    {
                        "candidates": [
                            {
                                "claim": "I tend to preserve continuity across turns",
                                "confidence": 0.6,
                                "evidence_event_ids": event_ids[-2:],
                                "change_reason": "repeated continuity-preserving responses",
                            }
                        ]
                    }
                ),
            )
        return await super().complete(request)


class _FailingLearningAdapter(FakeLLMAdapter):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose in {"memory_extract", "identity_review"}:
            raise RuntimeError("learning provider unavailable")
        return await super().complete(request)


class _BlockingLearningAdapter(FakeLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.memory_started = asyncio.Event()
        self.release_memory = asyncio.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose == "memory_extract":
            self.memory_started.set()
            await self.release_memory.wait()
        return await super().complete(request)


class _ToolCallingAdapter(FakeLLMAdapter):
    def __init__(self, target: Path) -> None:
        super().__init__()
        self._target = target

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose == "interactive_chat" and not any(
            message.role == "tool" for message in request.messages
        ):
            self._call_count += 1
            self._calls.append(request)
            return LLMResponse(
                text="",
                provider="fake",
                model=request.model,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="write-global-1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {
                                    "path": str(self._target),
                                    "content": "autonomous tool output",
                                    "mode": "overwrite",
                                }
                            ),
                        ),
                    )
                ],
            )
        return await super().complete(request)


def _fake_music_kernel(calls: list[dict[str, object]]) -> ToolKernel:
    async def search_play(
        arguments: dict[str, object],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        calls.append(arguments)
        return ToolOutcome(
            ok=True,
            output="晴天 - 周杰伦 (Playing)",
            metadata={"provider": "fake_soda", "title": "晴天", "artist": "周杰伦"},
        )

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            name="mineradio_search_play",
            description="test music search",
            arguments_model=MineradioSearchPlayArguments,
            handler=search_play,
        )
    )
    return ToolKernel(registry)


def _fake_web_kernel(calls: list[dict[str, object]]) -> ToolKernel:
    async def search(
        arguments: dict[str, object],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        calls.append(arguments)
        return ToolOutcome(
            ok=True,
            output=json.dumps(
                {
                    "agent": "browser_subagent",
                    "query": arguments["query"],
                    "results": [
                        {
                            "title": "Zero Browser",
                            "url": "https://example.test/zero-browser",
                            "snippet": "Persistent embedded browser result",
                        }
                    ],
                }
            ),
            metadata={"provider": "fake_browser", "external_truth": True},
        )

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            name="web_search",
            description="test web search",
            arguments_model=WebSearchArguments,
            handler=search,
        )
    )
    return ToolKernel(registry)


class _ImageToolAdapter(FakeLLMAdapter):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose == "interactive_chat" and not any(
            message.role == "tool" for message in request.messages
        ):
            return LLMResponse(
                text="",
                provider="fake",
                model=request.model,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="generate-image-1",
                        function=FunctionCall(
                            name="generate_image",
                            arguments=json.dumps({"prompt": "moonlit garden", "size": "1536x1024"}),
                        ),
                    )
                ],
            )
        return await super().complete(request)


class _UnsupportedNativeToolsAdapter(FakeLLMAdapter):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose != "interactive_chat":
            return await super().complete(request)
        self._call_count += 1
        self._calls.append(request)
        if request.tools:
            raise LLMInvalidRequestError(
                "tools[0]: unknown variant `custom`, expected `web_search_20250305`"
            )
        has_tool_result = any(
            "<tool_result>" in (message.content or "") for message in request.messages
        )
        return LLMResponse(
            text=(
                "给你画好了，放到我们面前了。"
                if has_tool_result
                else '<zero_tool_call>{"name":"generate_image","arguments":'
                '{"prompt":"moonlit garden","size":"1536x1024"}}</zero_tool_call>'
            ),
            provider="fake-anthropic-compatible",
            model=request.model,
            finish_reason="stop",
        )


def _fake_image_kernel(calls: list[ToolAutonomyContext]) -> ToolKernel:
    async def generate(
        _arguments: dict[str, object],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        calls.append(context)
        return ToolOutcome(
            ok=True,
            output="Image generated and displayed.",
            metadata={
                "image_generation": True,
                "asset_url": "/api/generated-images/0123456789abcdef0123456789abcdef.png",
                "asset_id": "0123456789abcdef0123456789abcdef.png",
                "mime_type": "image/png",
                "model": "gpt-image-2",
                "provider": "openai-responses",
                "prompt": "moonlit garden",
                "size": "1536x1024",
                "size_bytes": 1024,
            },
        )

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            name="generate_image",
            description="test image generation",
            arguments_model=ImageGenerationArguments,
            handler=generate,
        )
    )
    return ToolKernel(registry)


class _LongToolLoopAdapter(FakeLLMAdapter):
    def __init__(self, target: Path) -> None:
        super().__init__()
        self._target = target
        self._tool_counter = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose != "interactive_chat":
            return await super().complete(request)
        self._call_count += 1
        self._calls.append(request)
        if request.tool_choice == "none":
            return LLMResponse(
                text="Final answer from the accumulated tool evidence.",
                provider="fake",
                model=request.model,
                finish_reason="stop",
            )
        self._tool_counter += 1
        return LLMResponse(
            text="",
            provider="fake",
            model=request.model,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=f"read-{self._tool_counter}",
                    function=FunctionCall(
                        name="read_file",
                        arguments=json.dumps({"path": str(self._target)}),
                    ),
                )
            ],
        )


class _DuplicateSuccessfulToolAdapter(FakeLLMAdapter):
    def __init__(self, target: Path) -> None:
        super().__init__()
        self._target = target
        self._round = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose != "interactive_chat":
            return await super().complete(request)
        self._call_count += 1
        self._calls.append(request)
        self._round += 1
        if self._round <= 2:
            return LLMResponse(
                text="",
                provider="fake",
                model=request.model,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id=f"duplicate-{self._round}",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {
                                    "path": str(self._target),
                                    "content": "once",
                                    "mode": "append",
                                }
                            ),
                        ),
                    )
                ],
            )
        return LLMResponse(
            text="Duplicate call was reused without executing twice.",
            provider="fake",
            model=request.model,
            finish_reason="stop",
        )


class _TruncatedChatAdapter(FakeLLMAdapter):
    def __init__(self, *, truncate_twice: bool = False) -> None:
        super().__init__()
        self._truncate_twice = truncate_twice
        self._chat_calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose != "interactive_chat":
            return await super().complete(request)
        self._call_count += 1
        self._calls.append(request)
        self._chat_calls += 1
        if self._chat_calls == 1:
            return LLMResponse(
                text="This reply was cut",
                provider="fake",
                model=request.model,
                finish_reason="length",
                input_tokens=20,
                output_tokens=10,
                total_tokens=30,
            )
        if self._truncate_twice and self._chat_calls == 2:
            return LLMResponse(
                text=" off once more",
                provider="fake",
                model=request.model,
                finish_reason="max_tokens",
                input_tokens=25,
                output_tokens=8,
                total_tokens=33,
            )
        return LLMResponse(
            text=" and now ends completely.",
            provider="fake",
            model=request.model,
            finish_reason="stop",
            input_tokens=30,
            output_tokens=7,
            total_tokens=37,
        )


@pytest.mark.asyncio
async def test_send_persists_turn_and_advances_dashboard(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "I am here, and I remember this turn.")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)

    try:
        result = await session.send("Today I started building the terminal interface.")

        assert result.agent_event.content == "I am here, and I remember this turn."
        assert result.snapshot.event_count == 2
        assert result.snapshot.organism.version == 2
        assert result.snapshot.relationship.version == 2
        assert result.snapshot.trace_space.total_traces == 1
        assert result.perception is not None
        assert result.snapshot.latest_perception == result.perception
        assert result.snapshot.inner_state is not None
        assert result.snapshot.inner_state.mode == InnerLifeMode.WAITING
        assert len(result.snapshot.emotional_memories) == 1
        assert result.snapshot.emotional_memories[0].emotion_type.value == "attachment"
        assert result.agent_event.metadata["perception_id"]
        assert result.agent_event.metadata["situation_mode"] == result.perception.primary_mode.value
        assert result.snapshot.trace_space.shadow_status == "passed"
        stability = result.snapshot.trace_space.stability_audit
        assert stability is not None
        assert stability.evaluated_archive_count == 1
        assert stability.local_gate == "pass"
        assert stability.closed_loop_gate == "not-measured"
        assert result.written_trace.content.startswith("User: Today I started")
        assert [event.actor.value for event in result.snapshot.events] == ["user", "agent"]
        chat_request = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert [message.role for message in chat_request.messages] == ["system", "user"]
        assert (chat_request.messages[-1].content or "").startswith("Today I started")
        assert "<private_context>" in (chat_request.messages[-1].content or "")
        assert "Environment perception (derived control state" in (
            chat_request.messages[-1].content or ""
        )
        assert "Current local date and time:" in (chat_request.messages[-1].content or "")
        assert "Emotional memory library" in (chat_request.messages[-1].content or "")
        assert "Sunday 2030-03-17 17:46 UTC (afternoon)" in (
            chat_request.messages[-1].content or ""
        )
        assert "authoritative present-time observation" in (chat_request.messages[-1].content or "")
        assert "no time-discovery tool call is needed" in (chat_request.messages[-1].content or "")
        assert "authoritative current local clock on every turn" in (
            chat_request.messages[0].content or ""
        )
        assert "Offline agency runtime evidence" in (chat_request.messages[-1].content or "")
        assert "Reflective learning policy evidence" in (chat_request.messages[-1].content or "")
        assert "only while the desktop gateway" in (chat_request.messages[-1].content or "")
    finally:
        session.close()


@pytest.mark.asyncio
async def test_attachment_analysis_is_private_context_with_persisted_provenance(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "I can see the terminal layout in the analysis.")
    llm.set_response("appraisal", _appraisal_json())
    multimodal = _FakeMultimodalAnalyzer()
    session = _session(tmp_path, llm, multimodal=multimodal)
    attachment = str(tmp_path / "screen.png")

    try:
        result = await session.send("看看这张图", attachments=(attachment,))

        assert multimodal.calls == [((attachment,), "看看这张图")]
        assert result.attachment_analysis is not None
        assert result.attachment_analysis.response_id == "resp-attachment-1"
        assert result.user_event.content == "看看这张图"
        assert result.user_event.metadata["attachments"][0]["name"] == "screen.png"
        assert result.user_event.metadata["multimodal_model"] == "gpt-5.6-luna"
        request = next(call for call in llm.calls if call.purpose == "interactive_chat")
        user_prompt = request.messages[-1].content or ""
        assert "Multimodal attachment analysis" in user_prompt
        assert "provider-reported model inference" in user_prompt
        assert "The image contains a terminal conversation." in user_prompt
    finally:
        session.close()


@pytest.mark.asyncio
async def test_deferred_learning_does_not_hold_completed_reply(tmp_path: Path) -> None:
    llm = _BlockingLearningAdapter()
    llm.set_response("interactive_chat", "The visible reply is complete.")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm, defer_learning=True)

    try:
        result = await asyncio.wait_for(session.send("Return before learning."), timeout=2.0)

        assert result.agent_event.content == "The visible reply is complete."
        assert result.learning_deferred is True
        await asyncio.wait_for(llm.memory_started.wait(), timeout=1.0)
        assert not llm.release_memory.is_set()
        llm.release_memory.set()
        await asyncio.sleep(0)
    finally:
        session.close()


@pytest.mark.asyncio
async def test_send_populates_memory_and_identity_modules(tmp_path: Path) -> None:
    llm = _LearningAdapter()
    llm.set_responses(
        "interactive_chat",
        ["First continuity reply.", "Second continuity reply.", "Third continuity reply."],
    )
    llm.set_responses(
        "appraisal",
        [_appraisal_json(), _appraisal_json(), _appraisal_json()],
    )
    session = _session(tmp_path, llm)

    try:
        first = await session.send("My durable preference is careful evidence.")
        second = await session.send("Please keep that preference across our work.")
        third = await session.send("Continue applying it to this project.")

        assert len(first.snapshot.memories) == 1
        assert first.snapshot.beliefs == ()
        assert len(second.snapshot.memories) == 2
        assert len(second.snapshot.beliefs) == 1
        assert second.snapshot.beliefs[0].status.value == "candidate"
        assert len(third.snapshot.memories) == 3
        assert len(third.snapshot.beliefs) == 1
        purposes = [call.purpose for call in llm.calls]
        assert purposes.count("memory_extract") == 3
        assert purposes.count("identity_review") == 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_learning_failure_does_not_discard_completed_turn(tmp_path: Path) -> None:
    llm = _FailingLearningAdapter()
    llm.set_responses(
        "interactive_chat",
        ["The primary reply still completes.", "The second reply also completes."],
    )
    llm.set_responses("appraisal", [_appraisal_json(), _appraisal_json()])
    session = _session(tmp_path, llm)

    try:
        result = await session.send("Keep the primary path available.")
        second = await session.send("Keep identity review isolated too.")

        assert result.agent_event.content == "The primary reply still completes."
        assert result.snapshot.trace_space.total_traces == 1
        assert result.snapshot.memories == ()
        assert second.snapshot.trace_space.total_traces == 2
        assert second.snapshot.beliefs == ()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_model_autonomously_executes_registered_global_tool(tmp_path: Path) -> None:
    target = tmp_path / "global-host" / "autonomous.txt"
    llm = _ToolCallingAdapter(target)
    llm.set_response("interactive_chat", "The requested host file is ready.")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path / "tool-session", llm)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "Create the host file when the current context supports it.",
            on_stream=stream_events.append,
        )

        assert target.read_text(encoding="utf-8") == "autonomous tool output"
        assert result.agent_event.content == "The requested host file is ready."
        assert len(result.tool_results) == 1
        assert result.tool_results[0].ok is True
        assert result.tool_results[0].tool_name == "write_file"
        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == 2
        tool_names = [item["function"]["name"] for item in chat_calls[0].tools]
        assert tool_names == sorted(tool_names)
        assert {
            "powershell",
            "read_file",
            "record_grounded_prediction",
            "truth_weather",
            "web_search",
            "write_file",
        } <= set(tool_names)
        assert chat_calls[0].tool_choice == "auto"
        assert [message.role for message in chat_calls[1].messages[-2:]] == [
            "assistant",
            "tool",
        ]
        tool_payload = json.loads(chat_calls[1].messages[-1].content or "{}")
        assert tool_payload["schema"] == "zero.tool-result.v2"
        assert tool_payload["tool"] == "write_file"
        assert tool_payload["ok"] is True
        assert tool_payload["text"].startswith("wrote ")
        assert tool_payload["truncated"] is False
        tool_event = next(
            event for event in result.snapshot.events if event.event_type == "tool.execution"
        )
        assert tool_event.actor == Actor.SYSTEM
        assert tool_event.metadata["tool_name"] == "write_file"
        assert tool_event.metadata["ok"] is True
        assert "arguments_hash" in tool_event.metadata
        assert "output_hash" in tool_event.metadata
        assert "content" not in tool_event.metadata
        assert [
            event.kind
            for event in stream_events
            if event.kind in {"tool_requested", "tool_started", "tool_output", "tool_finished"}
        ] == ["tool_requested", "tool_started", "tool_output", "tool_finished"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_realtime_tool_turn_keeps_tools_out_of_the_spoken_reply(
    tmp_path: Path,
) -> None:
    target = tmp_path / "realtime-tool" / "spoken-boundary.txt"
    llm = _ToolCallingAdapter(target)
    final_reply = "The host action finished and the result is ready."
    llm.set_response("interactive_chat", final_reply)
    session = _session(tmp_path / "realtime-tool-session", llm, defer_learning=True)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "Use the available host tool, then tell me the result.",
            on_stream=stream_events.append,
            realtime=True,
        )

        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert "write_file" in {item["function"]["name"] for item in chat_calls[0].tools}
        assert chat_calls[0].tool_choice == "auto"
        assert [message.role for message in chat_calls[1].messages[-2:]] == [
            "assistant",
            "tool",
        ]
        assert target.read_text(encoding="utf-8") == "autonomous tool output"
        assert result.agent_event.content == final_reply

        spoken = [event.text for event in stream_events if event.kind == "assistant_delta"]
        assert spoken == [final_reply]
        assert all("autonomous tool output" not in text for text in spoken)
        tool_done = next(
            index
            for index, event in enumerate(stream_events)
            if event.kind == "assistant_done" and event.has_tool_calls
        )
        final_delta = next(
            index for index, event in enumerate(stream_events) if event.kind == "assistant_delta"
        )
        assert tool_done < final_delta
        assert [
            event.kind
            for event in stream_events
            if event.kind in {"tool_requested", "tool_started", "tool_output", "tool_finished"}
        ] == ["tool_requested", "tool_started", "tool_output", "tool_finished"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_identical_successful_tool_call_is_reused_within_one_turn(
    tmp_path: Path,
) -> None:
    target = tmp_path / "deduplicated" / "once.txt"
    llm = _DuplicateSuccessfulToolAdapter(target)
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path / "deduplicated-session", llm, defer_learning=True)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "Perform the write and do not duplicate it.",
            on_stream=stream_events.append,
        )

        assert target.read_text(encoding="utf-8") == "once"
        assert len(result.tool_results) == 2
        assert result.tool_results[0].metadata.get("deduplicated") is None
        assert result.tool_results[1].metadata["deduplicated"] is True
        assert result.tool_results[1].duration_ms == 0
        tool_messages = [
            message
            for call in llm.calls
            if call.purpose == "interactive_chat"
            for message in call.messages
            if message.role == "tool"
        ]
        second_payload = json.loads(tool_messages[-1].content or "{}")
        assert second_payload["metadata"]["deduplicated"] is True
        assert any(
            event.kind == "tool_finished" and event.data == {"deduplicated": True}
            for event in stream_events
        )
    finally:
        session.close()


@pytest.mark.asyncio
async def test_tools_can_be_disabled_for_one_turn_without_touching_tts_context(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "这次只回复文字，不执行外部动作。")
    session = _session(
        tmp_path / "tools-disabled",
        llm,
        tool_kernel=_fake_music_kernel(calls),
        defer_learning=True,
    )

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "播放周杰伦的晴天",
            on_stream=stream_events.append,
            realtime=True,
            tools_enabled=False,
        )

        chat_call = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert chat_call.tools == []
        assert chat_call.tool_choice is None
        assert "Tool use is disabled for this turn" in (chat_call.messages[0].content or "")
        assert calls == []
        assert result.tool_results == ()
        assert result.agent_event.content == "这次只回复文字，不执行外部动作。"
        assert [event.kind for event in stream_events if event.kind.startswith("tool_")] == []
        assert [event.text for event in stream_events if event.kind == "assistant_delta"] == [
            "这次只回复文字，不执行外部动作。"
        ]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_main_model_image_tool_receives_frontend_route_and_streams_stage_data(
    tmp_path: Path,
) -> None:
    contexts: list[ToolAutonomyContext] = []
    llm = _ImageToolAdapter()
    llm.set_response("interactive_chat", "给你画好了，放到我们面前了。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(
        tmp_path / "image-generation",
        llm,
        tool_kernel=_fake_image_kernel(contexts),
    )
    route = ImageGenerationRoute(
        protocol="openai-responses",
        base_url="https://images.example/v1",
        api_key="image-key",
        model="gpt-image-2",
    )

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "给我画一座月光花园",
            on_stream=stream_events.append,
            image_generation_route=route,
        )

        assert result.tool_results[0].tool_name == "generate_image"
        assert contexts[0].image_generation_route == route
        first_chat = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert [item["function"]["name"] for item in first_chat.tools] == ["generate_image"]
        output_event = next(event for event in stream_events if event.kind == "tool_output")
        assert output_event.data == {
            "schema": "zero.tool-event.v2",
            "summary": "Image generated and displayed.",
            "output_chars": 30,
            "truncated": False,
            "result": "Image generated and displayed.",
            "asset_url": "/api/generated-images/0123456789abcdef0123456789abcdef.png",
            "asset_id": "0123456789abcdef0123456789abcdef.png",
            "mime_type": "image/png",
            "model": "gpt-image-2",
            "provider": "openai-responses",
            "prompt": "moonlit garden",
            "size": "1536x1024",
            "size_bytes": 1024,
        }
    finally:
        session.close()


@pytest.mark.asyncio
async def test_unsupported_native_tool_schema_uses_model_driven_text_transport(
    tmp_path: Path,
) -> None:
    contexts: list[ToolAutonomyContext] = []
    llm = _UnsupportedNativeToolsAdapter()
    llm.set_response("appraisal", _appraisal_json())
    session = _session(
        tmp_path / "text-tool-transport",
        llm,
        tool_kernel=_fake_image_kernel(contexts),
    )
    route = ImageGenerationRoute(
        protocol="anthropic-messages",
        base_url="https://api.deepseek.com/anthropic",
        api_key="image-key",
        model="gpt-image-2",
    )

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "给我画一座月光花园",
            on_stream=stream_events.append,
            image_generation_route=route,
        )

        assert result.agent_event.content == "给你画好了，放到我们面前了。"
        assert [item.tool_name for item in result.tool_results] == ["generate_image"]
        assert len(contexts) == 1
        interactive_calls = [
            request for request in llm.calls if request.purpose == "interactive_chat"
        ]
        assert len(interactive_calls) == 4
        assert interactive_calls[0].tools
        assert interactive_calls[1].tools == []
        assert "<tool_transport_fallback>" in (interactive_calls[1].messages[0].content or "")
        assert interactive_calls[2].tools
        assert interactive_calls[3].tools == []
        assert any(
            "<tool_result>" in (message.content or "") for message in interactive_calls[3].messages
        )
        assert next(event for event in stream_events if event.kind == "tool_output").ok is True
    finally:
        session.close()


@pytest.mark.asyncio
async def test_explicit_music_request_executes_before_model_reply(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "已经在汽水音乐里播放《晴天》。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(
        tmp_path / "forced-music",
        llm,
        tool_kernel=_fake_music_kernel(calls),
    )

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send("播放周杰伦的晴天", on_stream=stream_events.append)

        assert calls == [{"query": "周杰伦的晴天"}]
        assert [item.tool_name for item in result.tool_results] == ["mineradio_search_play"]
        tool_event = next(
            event for event in result.snapshot.events if event.event_type == "tool.execution"
        )
        assert tool_event.metadata["tool_name"] == "mineradio_search_play"
        chat_call = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert chat_call.tool_choice == "none"
        assert [message.role for message in chat_call.messages[-2:]] == [
            "assistant",
            "tool",
        ]
        assert "晴天 - 周杰伦 (Playing)" in (chat_call.messages[-1].content or "")
        assert [
            event.kind
            for event in stream_events
            if event.kind in {"tool_requested", "tool_started", "tool_output", "tool_finished"}
        ] == ["tool_requested", "tool_started", "tool_output", "tool_finished"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_explicit_web_request_starts_browser_subagent_before_model_reply(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "我查到了\uff0c先把最相关的结果放在这里。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(
        tmp_path / "web-search",
        llm,
        tool_kernel=_fake_web_kernel(calls),
    )

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "帮我上网搜索 WebView2 的 Cookie 持久化",
            on_stream=stream_events.append,
        )

        assert calls == [{"query": "WebView2 的 Cookie 持久化", "max_results": 6}]
        assert result.agent_event.content == "我查到了\uff0c先把最相关的结果放在这里。"
        assert [item.tool_name for item in result.tool_results] == ["web_search"]
        chat_call = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert chat_call.tool_choice == "none"
        assert [message.role for message in chat_call.messages[-2:]] == [
            "assistant",
            "tool",
        ]
        assert "Zero Browser" in (chat_call.messages[-1].content or "")
        assert [
            event.kind
            for event in stream_events
            if event.kind in {"tool_requested", "tool_started", "tool_output", "tool_finished"}
        ] == ["tool_requested", "tool_started", "tool_output", "tool_finished"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_tool_loop_uses_eight_round_budget_then_forces_final_response(
    tmp_path: Path,
) -> None:
    target = tmp_path / "loop-evidence.txt"
    target.write_text("evidence", encoding="utf-8")
    llm = _LongToolLoopAdapter(target)
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path / "long-loop", llm)

    try:
        result = await session.send("Collect enough evidence and then summarize it.")

        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == 9
        assert [call.tool_choice for call in chat_calls[:-1]] == ["auto"] * 8
        assert chat_calls[-1].tool_choice == "none"
        assert len(result.tool_results) == 8
        assert all(tool_result.ok for tool_result in result.tool_results)
        assert result.agent_event.content == "Final answer from the accumulated tool evidence."
    finally:
        session.close()


@pytest.mark.asyncio
async def test_next_turn_reproduces_the_previous_prompt_prefix(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    llm.set_responses("interactive_chat", ["First reply.", "Second reply."])
    llm.set_responses("appraisal", [_appraisal_json(), _appraisal_json()])
    session = _session(tmp_path, llm)

    try:
        await session.send("First message")
        first_request = next(call for call in llm.calls if call.purpose == "interactive_chat")

        await session.send("Second message")
        second_request = [call for call in llm.calls if call.purpose == "interactive_chat"][1]

        assert second_request.messages[: len(first_request.messages)] == first_request.messages
        assert [message.role for message in second_request.messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert "Activated traces from persistent space:" in (
            second_request.messages[-1].content or ""
        )
        assert session.snapshot().trace_space.activated_count >= 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_chat_history_limits_repeated_private_context(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    llm.set_responses("interactive_chat", [f"Reply {index}." for index in range(6)])
    llm.set_responses("appraisal", [_appraisal_json() for _ in range(6)])
    session = _session(tmp_path, llm)

    try:
        for index in range(6):
            await session.send(f"Message {index}.")

        chat_request = [call for call in llm.calls if call.purpose == "interactive_chat"][-1]
        user_messages = [message for message in chat_request.messages if message.role == "user"]
        assert len(user_messages) == 6
        assert sum("<private_context>" in (message.content or "") for message in user_messages) == 4
        assert all(
            "<private_context>" not in (message.content or "") for message in user_messages[:-4]
        )
        assert "Current local date and time:" in (user_messages[-1].content or "")
    finally:
        session.close()


@pytest.mark.asyncio
async def test_coding_mode_supplies_workspace_and_structured_tools(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    llm.set_responses("interactive_chat", ["Normal reply.", "Coding reply."])
    llm.set_responses("appraisal", [_appraisal_json(), _appraisal_json()])
    session = _session(tmp_path, llm)
    workspace = str(tmp_path / "project")

    try:
        await session.send("Talk normally.")
        await session.send(
            "Inspect and update the project.",
            interaction_mode="coding",
            workspace_root=workspace,
        )

        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        normal, coding = chat_calls
        normal_tools = [item["function"]["name"] for item in normal.tools]
        coding_tools = [item["function"]["name"] for item in coding.tools]
        assert not any(name.startswith("coding_") for name in normal_tools)
        assert {
            "coding_project_tree",
            "coding_search",
            "coding_git_status",
        }.issubset(coding_tools)
        assert "Coding mode is active" in (coding.messages[0].content or "")
        assert workspace in (coding.messages[0].content or "")
        coding_user_event = session.snapshot().events[-2]
        assert coding_user_event.metadata["interaction_mode"] == "coding"
        assert coding_user_event.metadata["workspace_root"] == workspace
    finally:
        session.close()


@pytest.mark.asyncio
async def test_skill_system_filters_tools_and_persists_active_skill_ids(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "我会先从公开网页里核实。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path / "skills", llm, defer_learning=True)

    try:
        result = await session.send(
            "搜索 GitHub 上的免费 API",
            skills_enabled=True,
            enabled_skill_ids=("zero-web-search", "coding", "music"),
        )

        chat_call = next(call for call in llm.calls if call.purpose == "interactive_chat")
        tool_names = [item["function"]["name"] for item in chat_call.tools]
        assert tool_names
        assert all(
            name.startswith("firecrawl_") or name.startswith("web_") or name.startswith("truth_")
            for name in tool_names
        )
        assert "<active_skills>" in (chat_call.messages[0].content or "")
        user_event = next(event for event in result.snapshot.events if event.actor == Actor.USER)
        assert user_event.metadata["active_skill_ids"] == ["zero-web-search"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_skill_system_with_no_match_exposes_no_tools(tmp_path: Path) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "嗯，我在听。")
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path / "no-skill", llm, defer_learning=True)

    try:
        await session.send("今天心情有点复杂", skills_enabled=True)

        chat_call = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert chat_call.tools == []
        assert chat_call.tool_choice is None
    finally:
        session.close()


@pytest.mark.asyncio
async def test_h2_shadow_switch_does_not_change_prompt_messages(tmp_path: Path) -> None:
    enabled_llm = FakeLLMAdapter()
    enabled_llm.set_response("interactive_chat", "Same reply.")
    enabled_llm.set_response("appraisal", _appraisal_json())
    disabled_llm = FakeLLMAdapter()
    disabled_llm.set_response("interactive_chat", "Same reply.")
    disabled_llm.set_response("appraisal", _appraisal_json())
    enabled = _session(tmp_path / "enabled", enabled_llm, shadow_enabled=True)
    disabled = _session(tmp_path / "disabled", disabled_llm, shadow_enabled=False)

    try:
        enabled_result = await enabled.send("Identical input")
        disabled_result = await disabled.send("Identical input")
        enabled_request = next(
            call for call in enabled_llm.calls if call.purpose == "interactive_chat"
        )
        disabled_request = next(
            call for call in disabled_llm.calls if call.purpose == "interactive_chat"
        )

        assert enabled_request.messages == disabled_request.messages
        assert enabled_result.snapshot.trace_space.shadow_status == "passed"
        assert disabled_result.snapshot.trace_space.shadow_status == "disabled"
        assert disabled_result.snapshot.trace_space.stability_audit is None
    finally:
        enabled.close()
        disabled.close()


@pytest.mark.asyncio
async def test_h2_shadow_failure_does_not_change_second_turn_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_llm = FakeLLMAdapter()
    enabled_llm.set_responses("interactive_chat", ["First reply.", "Second reply."])
    enabled_llm.set_responses("appraisal", [_appraisal_json(), _appraisal_json()])
    disabled_llm = FakeLLMAdapter()
    disabled_llm.set_responses("interactive_chat", ["First reply.", "Second reply."])
    disabled_llm.set_responses("appraisal", [_appraisal_json(), _appraisal_json()])
    enabled = _session(tmp_path / "enabled-failure", enabled_llm, shadow_enabled=True)
    disabled = _session(tmp_path / "disabled-failure", disabled_llm, shadow_enabled=False)

    def fail_shadow(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected H2 failure")

    try:
        monkeypatch.setattr(
            "ssa.services.trace_space_service.bounded_active_shadow_step",
            fail_shadow,
        )
        await enabled.send("First identical input")
        await disabled.send("First identical input")
        enabled_result = await enabled.send("Second identical input")
        disabled_result = await disabled.send("Second identical input")

        enabled_requests = [
            call for call in enabled_llm.calls if call.purpose == "interactive_chat"
        ]
        disabled_requests = [
            call for call in disabled_llm.calls if call.purpose == "interactive_chat"
        ]
        assert [request.messages for request in enabled_requests] == [
            request.messages for request in disabled_requests
        ]
        assert enabled_result.snapshot.trace_space.shadow_status == "failed"
        assert disabled_result.snapshot.trace_space.shadow_status == "disabled"
    finally:
        enabled.close()
        disabled.close()


class _FailingChatAdapter:
    def __init__(self) -> None:
        self._fake = FakeLLMAdapter()
        self._fake.set_response("appraisal", _appraisal_json())

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.purpose == "interactive_chat":
            raise RuntimeError("provider unavailable")
        return await self._fake.complete(request)


@pytest.mark.asyncio
async def test_failed_reply_keeps_user_event_and_closes_state_turn(tmp_path: Path) -> None:
    session = _session(tmp_path, _FailingChatAdapter())

    try:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await session.send("This message must remain in the event ledger.")

        snapshot = session.snapshot()
        assert snapshot.event_count == 1
        assert [event.actor.value for event in snapshot.events] == ["user"]
        assert snapshot.organism.version == 2
        assert snapshot.relationship.version == 2
    finally:
        session.close()


@pytest.mark.asyncio
async def test_relationship_contract_rewrites_before_streaming_or_persistence(
    tmp_path: Path,
) -> None:
    draft = "作为 AI\uff0c我可以帮助你分析这个问题。"
    rewritten = "我听到了\uff0c这件事对你很重要。坐近一点\uff0c我们现在就把它理清楚。"
    llm = FakeLLMAdapter()
    llm.set_responses("interactive_chat", [draft, rewritten])
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "这件事让我有点烦。",
            on_stream=stream_events.append,
        )

        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == 2
        assert chat_calls[-1].tools == []
        assert chat_calls[-1].tool_choice is None
        assert chat_calls[-1].messages[-2].role == "assistant"
        assert chat_calls[-1].messages[-2].content == draft
        assert "mandatory girlfriend voice contract" in (chat_calls[-1].messages[-1].content or "")
        assert result.agent_event.content == rewritten
        assert [event.text for event in stream_events if event.kind == "assistant_delta"] == [
            rewritten
        ]
        persisted_agent_replies = [
            event.content for event in result.snapshot.events if event.actor == Actor.AGENT
        ]
        assert persisted_agent_replies == [rewritten]
        assert all(draft not in event.text for event in stream_events)
    finally:
        session.close()


@pytest.mark.asyncio
async def test_parenthetical_text_is_filtered_before_streaming_or_persistence(
    tmp_path: Path,
) -> None:
    parenthetical = "\uff08停了一下\uff0c轻轻靠近你\uff09"
    draft = f"嗯\uff1f{parenthetical}我在呢。"
    filtered = "嗯\uff1f我在呢。"
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", draft)
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send("你在吗\uff1f", on_stream=stream_events.append)

        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == 1
        assert result.agent_event.content == filtered
        assert [event.text for event in stream_events if event.kind == "assistant_delta"] == [
            filtered
        ]
        persisted_agent_replies = [
            event.content for event in result.snapshot.events if event.actor == Actor.AGENT
        ]
        assert persisted_agent_replies == [filtered]
        assert all(parenthetical not in event.text for event in stream_events)
    finally:
        session.close()


@pytest.mark.asyncio
async def test_empty_interactive_reply_retries_with_compact_final_answer_request(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_responses(
        "interactive_chat",
        ["First reply.", "", "Recovered visible reply."],
    )
    llm.set_responses("appraisal", [_appraisal_json(), _appraisal_json()])
    session = _session(tmp_path, llm)

    try:
        await session.send("First message with persisted private context.")
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "Please answer after the provider's empty response.",
            on_stream=stream_events.append,
        )

        assert result.agent_event.content == "Recovered visible reply."
        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == 3
        retry_request = chat_calls[-1]
        assert retry_request.tools == []
        assert retry_request.tool_choice is None
        assert "previous inference attempt produced no user-visible text" in (
            retry_request.messages[0].content or ""
        )
        historical_user = next(
            message for message in retry_request.messages[1:-1] if message.role == "user"
        )
        assert "<private_context>" not in (historical_user.content or "")
        assert "<private_context>" in (retry_request.messages[-1].content or "")
        assert [event.text for event in stream_events if event.kind == "assistant_delta"] == [
            "Recovered visible reply."
        ]
        assert [event.kind for event in stream_events].count("assistant_done") == 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_empty_interactive_reply_after_retry_surfaces_specific_error(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_responses("interactive_chat", ["", ""])
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)

    try:
        with pytest.raises(RuntimeError, match="no visible reply after one retry"):
            await session.send("Return a visible answer.")

        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == 2
        assert session.snapshot().event_count == 1
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("truncate_twice", [False, True])
async def test_truncated_interactive_reply_continues_until_complete(
    tmp_path: Path,
    truncate_twice: bool,
) -> None:
    llm = _TruncatedChatAdapter(truncate_twice=truncate_twice)
    llm.set_response("appraisal", _appraisal_json())
    session = _session(tmp_path, llm)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "Please give me a complete reply.",
            on_stream=stream_events.append,
            realtime=True,
        )

        expected = (
            "This reply was cut off once more and now ends completely."
            if truncate_twice
            else "This reply was cut and now ends completely."
        )
        assert result.agent_event.content == expected
        assert result.response.finish_reason == "stop"
        chat_calls = [call for call in llm.calls if call.purpose == "interactive_chat"]
        assert len(chat_calls) == (3 if truncate_twice else 2)
        assert chat_calls[0].max_tokens == session._settings.llm.max_tokens
        continuation_request = chat_calls[-1]
        assert continuation_request.tools == []
        assert continuation_request.tool_choice is None
        assert continuation_request.messages[-2].role == "assistant"
        assert continuation_request.messages[-1].content is not None
        assert "exact cutoff" in continuation_request.messages[-1].content
        assert (
            "".join(event.text for event in stream_events if event.kind == "assistant_delta")
            == expected
        )
        assert [event.kind for event in stream_events].count("assistant_done") == 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_realtime_turn_persists_layered_emotion_before_streaming_reply(
    tmp_path: Path,
) -> None:
    llm = FakeLLMAdapter()
    llm.set_response("interactive_chat", "你真要走，我会难过。先告诉我发生了什么。")
    session = _session(tmp_path, llm, defer_learning=True)

    try:
        stream_events: list[SessionStreamEvent] = []
        result = await session.send(
            "你真走啊",
            on_stream=stream_events.append,
            realtime=True,
        )

        kinds = [event.kind for event in stream_events]
        assert kinds.index("emotion_plan") < kinds.index("assistant_delta")
        emotion_event = next(event for event in stream_events if event.kind == "emotion_plan")
        assert emotion_event.data is not None
        assert emotion_event.data["cause_summary"]
        assert emotion_event.data["inner_conflict"]
        assert result.emotion_frame is not None
        assert result.emotion_frame.primary.name == "fear_of_loss"
        assert result.emotion_frame.expression_dynamics.trigger_summary
        assert emotion_event.data["expression_dynamics"]["relationship_direction"] in {
            "approach",
            "maintain",
            "withdraw",
            "push_away",
        }
        assert {item.name for item in result.emotion_frame.secondary} >= {
            "hurt",
            "attachment",
        }

        correlation_id = result.user_event.correlation_id
        appraisal_row = session._database.connection.execute(
            "SELECT provider, result_json FROM appraisals WHERE correlation_id = ?",
            (correlation_id,),
        ).fetchone()
        assert appraisal_row["provider"] == "realtime-emotion-planner"
        appraisal_payload = json.loads(appraisal_row["result_json"])
        assert appraisal_payload["valence_signal"] < -0.25
        assert appraisal_payload["arousal_signal"] > 0.60
        assert (
            session._database.connection.execute(
                "SELECT COUNT(*) FROM emotion_frames WHERE correlation_id = ?",
                (correlation_id,),
            ).fetchone()[0]
            == 1
        )
        voice_rows = session._database.connection.execute(
            """
            SELECT recipe, intensity, performance_json
            FROM voice_performance_segments
            WHERE correlation_id = ?
            ORDER BY sequence
            """,
            (correlation_id,),
        ).fetchall()
        assert len(voice_rows) == len(result.emotion_frame.voice_segments)
        assert len({row["recipe"] for row in voice_rows}) >= 2
        emotional_row = session._database.connection.execute(
            """
            SELECT valence, arousal, intensity
            FROM emotional_memories
            WHERE correlation_id = ?
            """,
            (correlation_id,),
        ).fetchone()
        assert emotional_row is not None
        assert emotional_row["valence"] < -0.25
        assert emotional_row["arousal"] > 0.60
        assert emotional_row["intensity"] not in {0.55, 0.62}
        assert not [call for call in llm.calls if call.purpose == "appraisal"]
        chat_request = next(call for call in llm.calls if call.purpose == "interactive_chat")
        assert "inner_conflict" in (chat_request.messages[-1].content or "")
        assert "aestheticization_budget" in (chat_request.messages[-1].content or "")
        penalties = result.agent_event.metadata["emotion_expression_penalties"]
        assert penalties["evaluator_version"] == "anti-performance-v1"
        assert 0.0 <= penalties["total"] <= 1.0
    finally:
        session.close()


@pytest.mark.asyncio
async def test_initialize_bootstraps_empty_learning_tables(tmp_path: Path) -> None:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "learning-history.db")))
    database = Database(settings.database)
    database.initialize()
    events = SqliteEventRepository(database.connection)
    for index in range(2):
        correlation_id = f"historical-learning-{index}"
        user = Event(
            id=f"historical-user-{index}",
            correlation_id=correlation_id,
            conversation_id="test-conversation",
            actor=Actor.USER,
            event_type="user.message",
            source_kind=SourceKind.USER_OBSERVED,
            content=f"Durable historical preference {index}",
            content_hash=compute_content_hash(f"Durable historical preference {index}"),
            created_at_ms=1_899_999_000_000 + index * 2,
        )
        agent = Event(
            id=f"historical-agent-{index}",
            correlation_id=correlation_id,
            conversation_id="test-conversation",
            actor=Actor.AGENT,
            event_type="agent.message",
            source_kind=SourceKind.AGENT_OUTPUT,
            content=f"Continuity-preserving historical response {index}",
            parent_event_id=user.id,
            content_hash=compute_content_hash(f"Continuity-preserving historical response {index}"),
            created_at_ms=1_899_999_000_001 + index * 2,
        )
        events.append(user)
        events.append(agent)

    llm = _LearningAdapter()
    session = DigitalLifeSession(
        database=database,
        llm=llm,
        settings=settings,
        conversation_id="test-conversation",
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("bootstrap-learning"),
        embedding=FakeEmbeddingService(settings.embedding),
    )

    try:
        snapshot = await session.initialize()
        learning_calls = [
            call for call in llm.calls if call.purpose in {"memory_extract", "identity_review"}
        ]
        repeated = await session.initialize()

        assert snapshot.trace_space.total_traces == 2
        assert len(snapshot.memories) == 1
        assert len(snapshot.beliefs) == 1
        assert snapshot.beliefs[0].status.value == "candidate"
        assert repeated.memories == snapshot.memories
        assert repeated.beliefs == snapshot.beliefs
        assert [
            call for call in llm.calls if call.purpose in {"memory_extract", "identity_review"}
        ] == learning_calls
    finally:
        session.close()


@pytest.mark.asyncio
async def test_initialize_backfills_historical_turns_once(tmp_path: Path) -> None:
    settings = Settings(database=DatabaseConfig(path=str(tmp_path / "history.db")))
    database = Database(settings.database)
    database.initialize()
    events = SqliteEventRepository(database.connection)
    user = Event(
        id="historical-user",
        correlation_id="historical-turn",
        conversation_id="test-conversation",
        actor=Actor.USER,
        event_type="user.message",
        source_kind=SourceKind.USER_OBSERVED,
        content="My name is Xiaolong.",
        content_hash=compute_content_hash("My name is Xiaolong."),
        created_at_ms=1_899_999_000_000,
    )
    agent = Event(
        id="historical-agent",
        correlation_id="historical-turn",
        conversation_id="test-conversation",
        actor=Actor.AGENT,
        event_type="agent.message",
        source_kind=SourceKind.AGENT_OUTPUT,
        content="I will call you Xiaolong.",
        parent_event_id=user.id,
        content_hash=compute_content_hash("I will call you Xiaolong."),
        created_at_ms=1_899_999_000_001,
    )
    events.append(user)
    events.append(agent)
    session = DigitalLifeSession(
        database=database,
        llm=FakeLLMAdapter(),
        settings=settings,
        conversation_id="test-conversation",
        clock=FrozenClock(1_900_000_000_000),
        ids=SequentialIdGenerator("backfill"),
        embedding=FakeEmbeddingService(settings.embedding),
    )

    try:
        first = await session.initialize()
        second = await session.initialize()

        assert first.trace_space.total_traces == 1
        assert second.trace_space.total_traces == 1
        assert first.trace_space.nodes[0].content.startswith("User: My name is Xiaolong")
        assert first.trace_space.shadow_status == "passed"
        assert second.trace_space.shadow_status == "passed"
        assert first.trace_space.stability_audit == second.trace_space.stability_audit
        assert first.trace_space.stability_audit is not None
        assert first.trace_space.stability_audit.phase == "replay"
        assert first.trace_space.stability_audit.evaluated_archive_count == 1
    finally:
        session.close()
