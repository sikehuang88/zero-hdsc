"""Textual pilot tests for the interactive terminal workspace."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from io import StringIO

import pytest
from rich.console import Console
from textual.widgets import Footer, Input, OptionList, Static, TabbedContent

from ssa.adapters.llm import LLMResponse
from ssa.config import PerceptionConfig
from ssa.domain.appraisal import AppraisalResult
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, compute_content_hash
from ssa.domain.learning import (
    BehaviorExperiment,
    EvidenceReferenceKind,
    EvidenceRelation,
    ExperimentStatus,
    LearningProposal,
    LearningProposalStatus,
    LearningProposalType,
    ProposalEvidence,
    ReflectionRun,
    ReflectionRunStatus,
    ReflectionTriggerKind,
)
from ssa.domain.lifecycle import (
    Goal,
    GoalOwner,
    GoalStatus,
    InnerLifeMode,
    InnerLoopState,
    OfflineActionKind,
    OfflineArtifact,
    OfflineEpisode,
    OfflineEpisodeStatus,
)
from ssa.domain.relationship import RelationshipState
from ssa.domain.state import OrganismState
from ssa.domain.traces import (
    Trace,
    TraceLink,
    TraceNode,
    TraceSpaceSnapshot,
    TraceSpaceStabilityAudit,
)
from ssa.interfaces.tui import (
    ConversationView,
    DigitalLifeApp,
    MessageBlock,
    OrganismStatePanel,
    RelationshipStatePanel,
    StreamActivity,
    ThinkingWaterfall,
)
from ssa.interfaces.tui_format import (
    format_activity_panel,
    format_relationship_panel,
    format_state_panel,
    format_trace_map,
)
from ssa.runtime.interactive import (
    ChatTurnResult,
    DashboardSnapshot,
    SessionStreamCallback,
    SessionStreamEvent,
)
from ssa.services.perception_service import EnvironmentPerceptionService


def _event(event_id: str, actor: Actor, content: str) -> Event:
    return Event(
        id=event_id,
        correlation_id="corr-1",
        conversation_id="test",
        actor=actor,
        event_type=f"{actor.value}.message",
        source_kind=(SourceKind.USER_OBSERVED if actor == Actor.USER else SourceKind.AGENT_OUTPUT),
        content=content,
        content_hash=compute_content_hash(content),
        created_at_ms=1_900_000_000_000,
    )


def _render_plain(renderable: object, *, width: int = 120) -> str:
    console = Console(file=StringIO(), width=width, color_system=None, record=True)
    console.print(renderable)
    return console.export_text()


def _trace() -> Trace:
    return Trace(
        id="trace-1",
        conversation_id="test",
        correlation_id="corr-1",
        input_event_id="user-1",
        output_event_id="agent-1",
        content="User: hello\nAgent: The terminal is alive and responsive.",
        source_kind=SourceKind.SYSTEM_DERIVED,
        importance=0.7,
        valence=0.2,
        arousal=0.3,
        embedding_model="fake",
        embedding_dim=512,
        vec_rowid=1,
        created_at_ms=1_900_000_000_000,
    )


def _snapshot(*events: Event, with_trace: bool = False) -> DashboardSnapshot:
    trace = _trace()
    return DashboardSnapshot(
        organism=OrganismState.initial(1_900_000_000_000),
        relationship=RelationshipState.initial(1_900_000_000_000),
        memories=(),
        beliefs=(),
        events=events,
        trace_space=TraceSpaceSnapshot(
            nodes=(
                [
                    TraceNode(
                        trace_id=trace.id,
                        content=trace.content,
                        content_type=trace.content_type,
                        source_kind=trace.source_kind,
                        x=0.5,
                        y=0.5,
                        importance=trace.importance,
                        freshness=1.0,
                        activation_score=0.8,
                        activation_kind="main",
                        created_at_ms=trace.created_at_ms,
                    )
                ]
                if with_trace
                else []
            ),
            total_traces=1 if with_trace else 0,
            latest_query="hello" if with_trace else "",
            legacy_activated_count=1 if with_trace else 0,
            shadow_status="passed" if with_trace else "awaiting-input",
            stability_audit=(
                TraceSpaceStabilityAudit(
                    phase="post-write",
                    local_gate="pass",
                    closed_loop_gate="not-measured",
                    evaluated_archive_count=1,
                    active_count=1,
                    active_capacity=8,
                    active_mass=0.15,
                    null_mass=0.85,
                    mass_residual=0.0,
                    contraction_bound=0.8,
                    active_set_churn=1.0,
                    certified_radius=0.5,
                )
                if with_trace
                else None
            ),
        ),
        event_count=len(events),
        model="deepseek/deepseek-v4-flash",
    )


@dataclass
class _StubSession:
    conversation_id: str = "test"
    closed: bool = False
    sent: list[str] | None = None
    attachments_sent: list[tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        self.sent = []
        self.attachments_sent = []

    def snapshot(self) -> DashboardSnapshot:
        return _snapshot()

    async def initialize(self) -> DashboardSnapshot:
        return self.snapshot()

    async def poll_lifecycle(self) -> tuple[Event, ...]:
        return ()

    async def send(
        self,
        content: str,
        *,
        attachments: tuple[str, ...] = (),
        on_stream: SessionStreamCallback | None = None,
    ) -> ChatTurnResult:
        assert self.sent is not None
        assert self.attachments_sent is not None
        self.sent.append(content)
        self.attachments_sent.append(attachments)
        await _emit(on_stream, SessionStreamEvent(kind="assistant_start", round_index=1))
        await _emit(
            on_stream,
            SessionStreamEvent(
                kind="assistant_delta",
                round_index=1,
                text="The terminal is alive and responsive.",
            ),
        )
        await _emit(
            on_stream,
            SessionStreamEvent(kind="assistant_done", round_index=1),
        )
        user = _event("user-1", Actor.USER, content)
        agent = _event("agent-1", Actor.AGENT, "The terminal is alive and responsive.")
        response = LLMResponse(
            text=agent.content,
            provider="fake",
            model="deepseek/deepseek-v4-flash",
            input_tokens=80,
            output_tokens=20,
            prompt_cache_hit_tokens=60,
            prompt_cache_miss_tokens=20,
            latency_ms=42,
        )
        return ChatTurnResult(
            user_event=user,
            agent_event=agent,
            response=response,
            appraisal=AppraisalResult.neutral(),
            written_trace=_trace(),
            snapshot=_snapshot(user, agent, with_trace=True),
        )

    def close(self) -> None:
        self.closed = True


class _BlockingSession(_StubSession):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def send(
        self,
        content: str,
        *,
        attachments: tuple[str, ...] = (),
        on_stream: SessionStreamCallback | None = None,
    ) -> ChatTurnResult:
        self.started.set()
        await self.release.wait()
        return await super().send(content, attachments=attachments, on_stream=on_stream)


class _StreamingSession(_StubSession):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.visible = asyncio.Event()
        self.release = asyncio.Event()

    async def send(
        self,
        content: str,
        *,
        attachments: tuple[str, ...] = (),
        on_stream: SessionStreamCallback | None = None,
    ) -> ChatTurnResult:
        await _emit(on_stream, SessionStreamEvent(kind="assistant_start", round_index=1))
        await _emit(
            on_stream,
            SessionStreamEvent(kind="assistant_delta", round_index=1, text="Hel"),
        )
        await _emit(
            on_stream,
            SessionStreamEvent(
                kind="tool_delta",
                round_index=1,
                tool_call_index=0,
                tool_name="read_file",
            ),
        )
        await _emit(
            on_stream,
            SessionStreamEvent(
                kind="tool_requested",
                round_index=1,
                tool_call_id="call-1",
                tool_name="read_file",
            ),
        )
        await _emit(
            on_stream,
            SessionStreamEvent(
                kind="tool_started",
                round_index=1,
                tool_call_id="call-1",
                tool_name="read_file",
            ),
        )
        self.visible.set()
        await self.release.wait()
        await _emit(
            on_stream,
            SessionStreamEvent(
                kind="tool_output",
                round_index=1,
                text="tool output",
                tool_call_id="call-1",
                tool_name="read_file",
                ok=True,
                duration_ms=12,
            ),
        )
        await _emit(
            on_stream,
            SessionStreamEvent(
                kind="tool_finished",
                round_index=1,
                tool_call_id="call-1",
                tool_name="read_file",
                ok=True,
                duration_ms=12,
            ),
        )
        await _emit(on_stream, SessionStreamEvent(kind="assistant_start", round_index=2))
        return await super().send(content, attachments=attachments, on_stream=on_stream)


async def _emit(
    callback: SessionStreamCallback | None,
    event: SessionStreamEvent,
) -> None:
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


def test_state_panel_renders_environment_perception() -> None:
    user = _event("user-clock", Actor.USER, "What time is it?")
    snapshot = _snapshot(user)
    perception = EnvironmentPerceptionService(
        config=PerceptionConfig(),
        timezone_name="UTC",
        quiet_hours_start="23:00",
        quiet_hours_end="07:00",
    ).perceive(
        user,
        AppraisalResult.neutral(),
        snapshot.organism,
        snapshot.relationship,
        [],
        [user],
    )

    perceived_snapshot = replace(snapshot, latest_perception=perception)
    rendered = format_state_panel(perceived_snapshot, animation_frame=0)
    next_frame = format_state_panel(perceived_snapshot, animation_frame=1)

    assert "ENVIRONMENT" in rendered
    assert "hdsc-p1-environment-perception" in rendered
    assert perception.primary_mode.value in rendered
    assert "first_contact" in rendered
    assert "╞" in rendered
    assert "╡" in rendered
    assert "▌" in rendered
    assert rendered != next_frame

    relationship = format_relationship_panel(snapshot, animation_frame=0)
    next_relationship_frame = format_relationship_panel(snapshot, animation_frame=1)
    assert "RELATIONSHIP" in relationship
    assert "╞" in relationship
    assert "╡" in relationship
    assert "▌" in relationship
    assert relationship != next_relationship_frame


def test_activity_panel_renders_persisted_offline_evidence() -> None:
    snapshot = replace(
        _snapshot(),
        offline_runtime_enabled=True,
        inner_state=InnerLoopState(
            conversation_id="test",
            version=3,
            mode=InnerLifeMode.WAITING,
            reply_expectation=0.72,
            concern=0.31,
            curiosity=0.64,
            connection_pressure=0.55,
            uncertainty=0.40,
            offline_readiness=0.28,
            wait_started_at_ms=1_900_000_000_000,
            wait_deadline_at_ms=1_900_003_600_000,
            last_user_event_id="user-1",
            last_agent_event_id="agent-1",
            last_heartbeat_at_ms=1_900_000_030_000,
            transition_reason="reply_window_open",
            updated_at_ms=1_900_000_030_000,
        ),
        reflection_runs=(
            ReflectionRun(
                id="reflection-1",
                conversation_id="test",
                correlation_id="corr-offline-1",
                dedup_key="reflection:test:artifact-1",
                trigger_kind=ReflectionTriggerKind.GOAL_RELEVANT,
                priority_score=0.78,
                status=ReflectionRunStatus.COMPLETED,
                source_episode_id="episode-1",
                source_artifact_id="artifact-1",
                source_event_ids=["user-1"],
                source_trace_ids=["trace-1"],
                reflection_text="Evidence-linked reflection",
                critique_text="Evidence coverage passed",
                critic_score=0.82,
                started_at_ms=1_900_000_001_000,
                completed_at_ms=1_900_000_002_000,
            ),
        ),
        learning_proposals=(
            LearningProposal(
                id="proposal-1",
                run_id="reflection-1",
                conversation_id="test",
                dedup_key="proposal:reflection-1:memory",
                proposal_type=LearningProposalType.MEMORY_CANDIDATE,
                status=LearningProposalStatus.APPLIED,
                title="Evidence before self-description",
                content="Keep reflection separate from user facts.",
                rationale="Preserve the reviewed note.",
                confidence=0.55,
                critic_score=0.82,
                target_kind="memory",
                target_id="memory-1",
                version=3,
                created_at_ms=1_900_000_002_000,
                updated_at_ms=1_900_000_003_000,
                applied_at_ms=1_900_000_003_000,
            ),
        ),
        learning_evidence=(
            ProposalEvidence(
                proposal_id="proposal-1",
                reference_kind=EvidenceReferenceKind.EVENT,
                reference_id="user-1",
                relation=EvidenceRelation.SUPPORTS,
                provenance_kind=SourceKind.USER_OBSERVED,
                trust_weight=1.0,
                likelihood_ratio=1.5,
                content_hash="evidence-hash",
                recorded_at_ms=1_900_000_003_000,
            ),
        ),
        behavior_experiments=(
            BehaviorExperiment(
                id="experiment-1",
                conversation_id="test",
                proposal_id="proposal-policy-1",
                dedup_key="experiment:proposal-policy-1",
                status=ExperimentStatus.RUNNING,
                hypothesis="Evidence citations improve continuity replies.",
                policy_key="reflection_guidance",
                treatment={"instruction": "Cite an activated trace."},
                min_observations=1,
                started_at_ms=1_900_000_003_000,
                due_at_ms=1_900_003_603_000,
                created_at_ms=1_900_000_003_000,
                updated_at_ms=1_900_000_003_000,
            ),
        ),
        goals=(
            Goal(
                id="goal-1",
                conversation_id="test",
                owner=GoalOwner.SELF,
                title="Build evidence-grounded continuity",
                motive="Turn offline intent into inspectable work",
                success_criteria=["Complete one evidence-linked reflection"],
                priority=0.8,
                progress=0.25,
                status=GoalStatus.ACTIVE,
                created_at_ms=1_900_000_000_000,
                updated_at_ms=1_900_000_000_000,
            ),
        ),
        offline_episodes=(
            OfflineEpisode(
                id="episode-1",
                conversation_id="test",
                correlation_id="corr-offline-1",
                dedup_key="offline:test:trace:trace-1",
                action_kind=OfflineActionKind.TRACE_REFLECTION,
                motive="Advance continuity",
                status=OfflineEpisodeStatus.COMPLETED,
                source_event_ids=["user-1"],
                source_trace_ids=["trace-1"],
                summary="Reviewed one archived interaction.",
                event_id="event-offline-1",
                started_at_ms=1_900_000_000_000,
                completed_at_ms=1_900_000_001_000,
            ),
        ),
        offline_artifacts=(
            OfflineArtifact(
                id="artifact-1",
                episode_id="episode-1",
                artifact_type="trace_reflection",
                title="Evidence before self-description",
                content="Offline growth claims require a completed, inspectable artifact.",
                evidence_event_ids=["user-1"],
                evidence_trace_ids=["trace-1"],
                created_at_ms=1_900_000_001_000,
            ),
        ),
    )

    rendered = format_activity_panel(snapshot)

    assert "OFFLINE AGENCY" in rendered
    assert "PROCESS LIVE" in rendered
    assert "INNER LOOP" in rendered
    assert "WAITING" in rendered
    assert "reply 72% // concern 31%" in rendered
    assert "Build evidence-grounded continuity" in rendered
    assert "COMPLETED" in rendered
    assert "Evidence before self-description" in rendered
    assert "Offline growth claims require" in rendered
    assert "events 1 // traces 1 // id artifact-1" in rendered
    assert "LEARNING LOOP" in rendered
    assert "APPLIED" in rendered
    assert "memory:memory-1" in rendered
    assert "evidence 1" in rendered
    assert "REFLECTION" in rendered
    assert "EXPERIMENT" in rendered
    assert "Evidence citations improve continuity replies" in rendered


@pytest.mark.asyncio
async def test_dashboard_batteries_animate_without_changing_snapshot() -> None:
    app = DigitalLifeApp(_StubSession())

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        panels = [app.query_one(OrganismStatePanel), app.query_one(RelationshipStatePanel)]
        snapshots = [panel._snapshot for panel in panels]
        first_frames = [str(panel.content) for panel in panels]

        await pilot.pause(0.25)

        for panel, snapshot, first_frame in zip(panels, snapshots, first_frames, strict=True):
            assert panel._snapshot is snapshot
            assert str(panel.content) != first_frame


@pytest.mark.asyncio
async def test_thinking_waterfall_animates_only_while_request_is_active() -> None:
    session = _BlockingSession()
    app = DigitalLifeApp(session)

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        composer = app.query_one("#composer", Input)
        composer_shell = app.query_one("#composer-shell")
        status = app.query_one("#status-line", Static)
        conversation_column = app.query_one("#conversation-column")
        waterfall = app.query_one(ThinkingWaterfall)

        assert len(app.query(Footer)) == 0
        assert composer_shell.region.height == 4
        assert composer.region.height == 2
        assert status.region.height == 1
        assert status.region.right <= conversation_column.region.right

        composer.value = "hold the response"
        await pilot.press("enter")
        await asyncio.wait_for(session.started.wait(), timeout=1.0)
        await pilot.pause()

        assert waterfall.has_class("active")
        assert composer_shell.region.height == 7
        assert waterfall.region.bottom <= composer.region.y
        assert composer.region.bottom <= status.region.y
        first_frame = waterfall.render().plain
        await pilot.pause(0.3)
        second_frame = waterfall.render().plain
        assert second_frame != first_frame

        session.release.set()
        await pilot.pause(0.2)
        assert not waterfall.has_class("active")
        assert composer_shell.region.height == 4


@pytest.mark.asyncio
async def test_stream_activity_updates_answer_and_tool_lifecycle_live() -> None:
    session = _StreamingSession()
    app = DigitalLifeApp(session)

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        composer = app.query_one("#composer", Input)
        composer.value = "stream this"
        await pilot.press("enter")
        await asyncio.wait_for(session.visible.wait(), timeout=1.0)
        await pilot.pause()

        activity = app.query_one(StreamActivity)
        rendered = _render_plain(activity.content)
        assert activity.has_class("active")
        assert activity.answer == "Hel"
        assert "read_file" in rendered
        assert "running" in rendered

        session.release.set()
        await pilot.pause(0.2)
        assert not activity.has_class("active")
        assert composer.disabled is False


@pytest.mark.asyncio
async def test_assistant_done_switches_status_to_trace_finalization() -> None:
    app = DigitalLifeApp(_StubSession())

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        activity = app.query_one(ConversationView).append_stream()
        app._active_stream = activity
        await pilot.pause()
        app._handle_stream_event(
            SessionStreamEvent(kind="assistant_done", round_index=1, has_tool_calls=False)
        )

        status = app.query_one("#status-line", Static)
        assert "FINALIZING // saving trace" in str(status.content)
        assert not app.query_one(ThinkingWaterfall).has_class("active")


@pytest.mark.asyncio
async def test_tui_submits_message_and_refreshes_status() -> None:
    session = _StubSession()
    app = DigitalLifeApp(session)

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        composer = app.query_one("#composer", Input)
        composer.value = "hello"
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert session.sent == ["hello"]
        assert composer.disabled is False
        messages = list(app.query(MessageBlock))
        assert len(messages) == 2
        assert sum(
            "The terminal is alive and responsive." in _render_plain(item.content)
            for item in messages
        ) == 1
        assert not app.query_one(StreamActivity).has_class("active")
        status = app.query_one("#status-line", Static)
        assert "cache 75%" in str(status.content)
        assert app.query_one("#trace-list", OptionList).option_count == 1
        assert "TRACE SPACE" in str(app.query_one("#space-map", Static).content)
        rendered_space = str(app.query_one("#space-map", Static).content)
        assert "archive 1 // view 1" in rendered_space
        assert "serving legacy-ssa-a0" in rendered_space
        assert "shadow hdsc-h2-bounded-active-shadow" in rendered_space
        assert "active 1/8 clusters" in rendered_space
        assert "loop not measured" in rendered_space

        composer.value = "/identity"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(TabbedContent).active == "identity-tab"

        composer.value = "/activity"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(TabbedContent).active == "activity-tab"

    assert session.closed is True


def test_message_block_renders_markdown_and_stage_direction() -> None:
    message = MessageBlock(
        "SSA",
        "**重点**\n\n1. 第一项\n2. 第二项\n\n\uff08轻轻眨了眨眼\uff09",
        "#e8bd67",
    )

    rendered = _render_plain(message.content)
    assert "重点" in rendered
    assert "第一项" in rendered and "第二项" in rendered
    assert "**" not in rendered
    assert "\uff08轻轻眨了眨眼\uff09" in rendered


@pytest.mark.asyncio
async def test_attachment_queue_is_sent_with_next_message(tmp_path) -> None:
    attachment = tmp_path / "visual note.png"
    attachment.write_bytes(b"not-a-real-image")
    session = _StubSession()
    app = DigitalLifeApp(session)

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        composer = app.query_one("#composer", Input)
        composer.value = f'/attach "{attachment}"'
        await pilot.press("enter")
        await pilot.pause()

        strip = app.query_one("#attachment-strip", Static)
        assert strip.has_class("active")
        assert attachment.name in str(strip.content)

        composer.value = "看看这张图"
        await pilot.press("enter")
        await pilot.pause(0.15)

        assert session.attachments_sent == [(str(attachment.resolve()),)]
        assert not strip.has_class("active")


@pytest.mark.asyncio
async def test_slash_command_palette_filters_completes_and_executes() -> None:
    app = DigitalLifeApp(_StubSession())

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.1)
        composer = app.query_one("#composer", Input)
        palette = app.query_one("#command-palette", OptionList)

        composer.value = "/"
        await pilot.pause()
        assert palette.has_class("active")
        assert palette.option_count == 8
        assert palette.highlighted == 0
        await pilot.press("down")
        assert palette.highlighted == 1
        await pilot.press("up")
        assert palette.highlighted == 0

        composer.value = "/act"
        await pilot.pause()
        assert palette.option_count == 1
        assert "/activity" in _render_plain(palette.get_option_at_index(0).prompt)

        await pilot.press("enter")
        await pilot.pause()
        assert composer.value == "/activity"
        assert app.query_one(TabbedContent).active == "space-tab"

        await pilot.press("enter")
        await pilot.pause()
        assert composer.value == ""
        assert app.query_one(TabbedContent).active == "activity-tab"
        assert not palette.has_class("active")

        composer.value = "/att"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert composer.value == "/attach "
        assert not palette.has_class("active")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 30), (120, 42)])
async def test_message_blocks_stay_inside_conversation_width(
    size: tuple[int, int],
) -> None:
    app = DigitalLifeApp(_StubSession())

    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        app._write_message("YOU", "这是一条用于验证窄屏布局的较长消息。", "#77c8b9")
        app._write_message(
            "SSA",
            "\uff08轻轻眨了眨眼\uff09\n我会待在对话空间里面\uff0c不挤到旁边的状态面板。",
            "#e8bd67",
        )
        await pilot.pause()

        conversation = app.query_one(ConversationView)
        for message in app.query(MessageBlock):
            assert message.region.x >= conversation.region.x
            assert message.region.right <= conversation.region.right
        assert conversation.max_scroll_x == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "map_height", "show_trace_detail"),
    [
        ((80, 30), 14, False),
        ((120, 42), 18, True),
    ],
)
async def test_trace_space_layout_stays_inside_supported_terminal_sizes(
    size: tuple[int, int],
    map_height: int,
    show_trace_detail: bool,
) -> None:
    app = DigitalLifeApp(_StubSession())

    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        snapshot = _snapshot(
            _event("user-1", Actor.USER, "hello"),
            _event("agent-1", Actor.AGENT, "still here"),
            with_trace=True,
        )
        app._refresh_snapshot(snapshot)
        await pilot.pause()

        space_map = app.query_one("#space-map", Static)
        trace_list = app.query_one("#trace-list", OptionList)
        trace_detail = app.query_one("#trace-detail", Static)
        assert space_map.region.height == map_height
        assert space_map.region.intersection(app.screen.region) == space_map.region
        assert trace_list.display is show_trace_detail
        assert trace_detail.display is show_trace_detail
        if show_trace_detail:
            assert trace_list.region.intersection(app.screen.region) == trace_list.region
            assert trace_detail.region.intersection(app.screen.region) == trace_detail.region

        canvas_height = 4 if size[0] < 100 else 8
        rendered_lines = format_trace_map(snapshot, height=canvas_height).splitlines()
        assert len(rendered_lines) == map_height

    assert app._session.closed is True


def test_trace_map_distinguishes_directed_and_symmetric_links() -> None:
    snapshot = _snapshot(with_trace=True)
    first = snapshot.trace_space.nodes[0]
    second = first.model_copy(
        update={
            "trace_id": "trace-2",
            "content": "Later episode",
            "x": 0.9,
            "created_at_ms": first.created_at_ms + 1,
        }
    )
    trace_space = snapshot.trace_space.model_copy(
        update={
            "nodes": [first.model_copy(update={"x": 0.1}), second],
            "links": [
                TraceLink(
                    source_trace_id=first.trace_id,
                    target_trace_id=second.trace_id,
                    link_type="temporal-forward",
                    weight=0.8,
                    created_at_ms=second.created_at_ms,
                )
            ],
        }
    )
    rendered = format_trace_map(
        snapshot.__class__(**{**snapshot.__dict__, "trace_space": trace_space}),
        width=20,
        height=4,
    )

    assert "[bold #e06c75]>[/]" in rendered
    assert "[dim]:[/] sym" in rendered
