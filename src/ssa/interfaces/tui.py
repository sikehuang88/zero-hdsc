"""Textual terminal interface for an interactive SSA session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import (
    Header,
    Input,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option

from ssa.commands import (
    CommandCapability,
    CommandRegistry,
    ParsedCommand,
    build_default_command_registry,
)
from ssa.config import Settings
from ssa.domain.events import Event
from ssa.domain.traces import TraceNode
from ssa.interfaces.tui_format import (
    format_activity_panel,
    format_identity_panel,
    format_memory_panel,
    format_relationship_panel,
    format_state_panel,
    format_trace_detail,
    format_trace_map,
)
from ssa.runtime.interactive import (
    ChatTurnResult,
    DashboardSnapshot,
    InteractiveSession,
    SessionStreamEvent,
    build_interactive_session,
)


class ThinkingWaterfall(Static):
    """A compact abstract activity pulse; it never renders model reasoning."""

    _timer: Timer | None = None
    _tick: int = 0

    def on_mount(self) -> None:
        self._timer = self.set_interval(
            0.12,
            self._advance,
            name="thinking-waterfall",
            pause=True,
        )

    def start(self) -> None:
        self._tick = 0
        self.add_class("active")
        if self._timer is not None:
            self._timer.resume()
        self.refresh()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.pause()
        self.remove_class("active")

    def _advance(self) -> None:
        self._tick = (self._tick + 1) % 420
        self.refresh()

    def render(self) -> Text:
        width = max(18, min(42, self.size.width - 12))
        column_count = max(6, width // 3)
        rows = [[" " for _ in range(column_count * 3)] for _ in range(3)]
        styles: dict[tuple[int, int], str] = {}
        glyphs = ((0, "█", "bold #77c8b9"), (1, "▓", "#66b3a6"), (2, "▒", "#4f8f83"))

        for column in range(column_count):
            speed = 1 + column % 3
            head = ((self._tick * speed + column * 2) % 8) - 3
            x = column * 3
            for distance, glyph, style in glyphs:
                y = head - distance
                if 0 <= y < len(rows):
                    rows[y][x] = glyph
                    styles[(y, x)] = style

        output = Text()
        for row_index, row in enumerate(rows):
            output.append("THINKING  " if row_index == 0 else "          ", style="bold #77c8b9")
            for column_index, glyph in enumerate(row):
                output.append(glyph, style=styles.get((row_index, column_index), "#263238"))
            if row_index < len(rows) - 1:
                output.append("\n")
        return output


@dataclass
class _ToolVisual:
    name: str
    status: str
    round_index: int
    summary: str = ""
    ok: bool | None = None
    duration_ms: int | None = None
    exit_code: int | None = None


class MessageBlock(Static):
    """A softly animated, independently styled conversation message."""

    def __init__(
        self,
        author: str,
        content: str,
        color: str,
        *,
        stamp: str | None = None,
        role: str = "assistant",
        **kwargs: Any,
    ) -> None:
        self.author = author
        self.message = content
        self.color = color
        self.stamp = stamp or datetime.now(UTC).strftime("%H:%M")
        super().__init__(classes=f"message-block {role}", **kwargs)
        self.update(self._message_renderable())

    def on_mount(self) -> None:
        self.styles.opacity = 0.35
        self.styles.animate(
            "opacity",
            1.0,
            duration=0.22,
            easing="out_cubic",
        )

    def set_message(self, content: str) -> None:
        self.message = content
        self.update(self._message_renderable())

    def _message_renderable(self) -> Group:
        heading = Text()
        marker = "◆" if self.author == "SSA" else "●" if self.author == "YOU" else "◇"
        heading.append(f"{marker} {self.author}", style=f"bold {self.color}")
        heading.append(f"  {self.stamp}", style="#6f7a82")
        return Group(heading, self._markdown_body(self.message))

    @staticmethod
    def _markdown_body(content: str) -> Padding:
        normalized: list[str] = []
        lines = content.splitlines() or [""]
        for line in lines:
            stripped = line.strip()
            is_stage_direction = len(stripped) >= 2 and (
                (stripped.startswith("\uff08") and stripped.endswith("\uff09"))
                or (stripped.startswith("(") and stripped.endswith(")"))
            )
            normalized.append(f"*{line}*" if is_stage_direction else line)
        markdown = Markdown(
            "\n".join(normalized),
            code_theme="monokai",
            inline_code_theme="monokai",
            style="#e1e5e8",
        )
        return Padding(markdown, (0, 0, 0, 2))


class ConversationView(VerticalScroll):
    """Scrollable message timeline with stable per-message identity."""

    def append_message(
        self,
        author: str,
        content: str,
        color: str,
        *,
        role: str,
    ) -> MessageBlock:
        message = MessageBlock(author, content, color, role=role)
        self.mount(message)
        self.call_after_refresh(self.scroll_end, animate=False)
        return message

    def append_stream(self) -> StreamActivity:
        activity = StreamActivity()
        self.mount(activity)
        activity.begin()
        self.call_after_refresh(self.scroll_end, animate=False)
        return activity

    def keep_latest_visible(self) -> None:
        self.call_after_refresh(self.scroll_end, animate=False)

    def clear_messages(self) -> None:
        self.remove_children()


class StreamActivity(MessageBlock):
    """Live assistant message with a compact tool lifecycle trail."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            "SSA",
            "",
            "#e8bd67",
            *args,
            role="assistant streaming",
            **kwargs,
        )
        self._round_index = 0
        self._answer = ""
        self._phase = "CONNECTING"
        self._tools: dict[str, _ToolVisual] = {}
        self._pulse_timer: Timer | None = None
        self._pulse_tick = 0

    def on_mount(self) -> None:
        super().on_mount()
        self._pulse_timer = self.set_interval(
            0.42,
            self._advance_pulse,
            name="message-breathing-dot",
            pause=not self.has_class("active"),
        )

    def _advance_pulse(self) -> None:
        self._pulse_tick = (self._pulse_tick + 1) % 4
        self._render_activity()

    @property
    def answer(self) -> str:
        return self._answer

    def begin(self) -> None:
        self._round_index = 0
        self._answer = ""
        self._phase = "CONNECTING"
        self._tools.clear()
        self.add_class("active")
        if self._pulse_timer is not None:
            self._pulse_timer.resume()
        self._render_activity()

    def finish(self) -> None:
        self.remove_class("active")
        if self._pulse_timer is not None:
            self._pulse_timer.pause()
        self._render_activity()

    def complete(self, content: str) -> None:
        self._answer = content
        self._phase = "SETTLED"
        self.finish()

    def analyzing_files(self, count: int) -> None:
        self._phase = f"ANALYZING {count} FILE{'S' if count != 1 else ''}"
        self._render_activity()

    def handle(self, event: SessionStreamEvent) -> None:
        self.add_class("active")
        self._round_index = event.round_index
        if event.kind == "assistant_start":
            self._answer = ""
            self._phase = "GENERATING"
        elif event.kind == "assistant_delta":
            self._answer += event.text
            self._phase = "STREAMING"
        elif event.kind == "tool_delta":
            key = self._pending_key(event)
            visual = self._tools.setdefault(
                key,
                _ToolVisual(name="", status="ASSEMBLING", round_index=event.round_index),
            )
            visual.name += event.tool_name
            self._phase = "TOOL CALL"
        elif event.kind in {"tool_requested", "tool_started"}:
            self._remove_matching_pending(event)
            self._tools[event.tool_call_id] = _ToolVisual(
                name=event.tool_name,
                status="RUNNING" if event.kind == "tool_started" else "QUEUED",
                round_index=event.round_index,
            )
            self._phase = "TOOL RUN"
        elif event.kind in {"tool_output", "tool_finished"}:
            visual = self._tools.setdefault(
                event.tool_call_id,
                _ToolVisual(
                    name=event.tool_name,
                    status="RUNNING",
                    round_index=event.round_index,
                ),
            )
            if event.kind == "tool_output":
                visual.summary = " ".join(event.text.split())[:120]
            else:
                visual.status = "SUCCESS" if event.ok else "FAILED"
                visual.ok = event.ok
                visual.duration_ms = event.duration_ms
                visual.exit_code = event.exit_code
            self._phase = "TOOL RUN"
        elif event.kind == "assistant_done":
            self._phase = "TOOL PENDING" if event.has_tool_calls else "FINALIZING"
        self._render_activity()

    def _pending_key(self, event: SessionStreamEvent) -> str:
        return f"pending:{event.round_index}:{event.tool_call_index or 0}"

    def _remove_matching_pending(self, event: SessionStreamEvent) -> None:
        pending = [
            key
            for key, visual in self._tools.items()
            if key.startswith(f"pending:{event.round_index}:")
            and (not visual.name or visual.name == event.tool_name)
        ]
        if pending:
            del self._tools[pending[0]]

    def _render_activity(self) -> None:
        heading = Text()
        marker = "◉" if self._pulse_tick % 2 == 0 else "◇"
        if not self.has_class("active"):
            marker = "◆"
        heading.append(f"{marker} SSA", style="bold #e8bd67")
        heading.append(f"  {self.stamp}", style="#6f7a82")
        if self.has_class("active"):
            heading.append(
                f"  ·  R{self._round_index:02d} {self._phase.casefold()}",
                style="#8a969f",
            )
        renderables: list[Any] = [heading]
        if self._answer:
            renderables.append(self._markdown_body(self._answer))
        tool_trail = Text()
        visuals = list(self._tools.values())
        for index, visual in enumerate(visuals):
            status_style = (
                "bold #77c8b9"
                if visual.status == "SUCCESS"
                else "bold #e06c75"
                if visual.status == "FAILED"
                else "bold #f0c674"
            )
            branch = "╰─" if index == len(visuals) - 1 else "├─"
            state_glyph = "✓" if visual.status == "SUCCESS" else "!" if visual.status == "FAILED" else "◌"
            if index:
                tool_trail.append("\n")
            tool_trail.append(f"{branch} {state_glyph} ", style=status_style)
            tool_trail.append(visual.name or "function", style="bold #d8dee9")
            tool_trail.append(f"  {visual.status.casefold()}", style=status_style)
            if visual.duration_ms is not None:
                tool_trail.append(f"  {visual.duration_ms}ms", style="dim")
            if visual.exit_code is not None:
                tool_trail.append(f"  exit {visual.exit_code}", style="dim")
            if visual.summary:
                tool_trail.append(f"\n   {visual.summary}", style="#9fb0bd")
        if visuals:
            renderables.append(Padding(tool_trail, (0, 0, 0, 2)))
        self.update(Group(*renderables))


class _AnimatedBatteryPanel(Static):
    """Shared animation lifecycle for snapshot-backed battery panels."""

    _snapshot: DashboardSnapshot | None = None
    _animation_frame: int = 0
    _timer_name: ClassVar[str] = "dashboard-batteries"

    def on_mount(self) -> None:
        self.set_interval(0.18, self._advance_batteries, name=self._timer_name)

    def set_snapshot(self, snapshot: DashboardSnapshot) -> None:
        self._snapshot = snapshot
        self._render_batteries()

    def _advance_batteries(self) -> None:
        if self._snapshot is None:
            return
        self._animation_frame = (self._animation_frame + 1) % 420
        self._render_batteries()

    def on_resize(self) -> None:
        self._render_batteries()

    def _render_batteries(self) -> None:
        if self._snapshot is not None:
            battery_width = max(8, min(14, self.size.width - 20))
            self.update(self._format_snapshot(self._snapshot, battery_width=battery_width))

    def _format_snapshot(self, snapshot: DashboardSnapshot, *, battery_width: int) -> str:
        raise NotImplementedError


class OrganismStatePanel(_AnimatedBatteryPanel):
    """Animated battery view over the latest persisted organism snapshot."""

    _timer_name = "organism-batteries"

    def _format_snapshot(self, snapshot: DashboardSnapshot, *, battery_width: int) -> str:
        return format_state_panel(
            snapshot,
            animation_frame=self._animation_frame,
            battery_width=battery_width,
        )


class RelationshipStatePanel(_AnimatedBatteryPanel):
    """Animated battery view over the latest persisted relationship snapshot."""

    _timer_name = "relationship-batteries"

    def _format_snapshot(self, snapshot: DashboardSnapshot, *, battery_width: int) -> str:
        return format_relationship_panel(
            snapshot,
            animation_frame=self._animation_frame,
            battery_width=battery_width,
        )


class DigitalLifeApp(App[None]):
    """Full-screen terminal workspace for the local digital lifeform."""

    TITLE = "HDSC // 超维度空间计算"
    SUB_TITLE = "hyperdimensional space console"

    CSS = """
    Screen {
        background: #111417;
        color: #d8dee9;
    }

    Header {
        background: #1a2026;
        color: #f0f3f5;
        height: 1;
    }

    #workspace {
        height: 1fr;
    }

    #conversation-column {
        width: 2fr;
        min-width: 48;
        border-right: solid #36414a;
    }

    #conversation {
        height: 1fr;
        padding: 1 1 0 1;
        scrollbar-size: 1 1;
        scrollbar-color: #3d6e66;
        scrollbar-color-hover: #66b3a6;
    }

    MessageBlock {
        width: 1fr;
        height: auto;
        min-height: 3;
        margin: 0 1 1 1;
        padding: 0 1 1 1;
        color: #e1e5e8;
    }

    MessageBlock.assistant {
        background: #181816;
        border-left: tall #755d31;
    }

    MessageBlock.user {
        width: 1fr;
        margin-left: 7;
        background: #14201f;
        border-left: tall #376d65;
    }

    MessageBlock.system {
        background: #171b1f;
        border-left: tall #49545d;
        color: #aeb7be;
    }

    MessageBlock.streaming {
        background: #1b1a16;
        border-left: heavy #e8bd67;
    }

    MessageBlock.streaming.active {
        background: #1d1c17;
    }

    #composer-shell {
        height: auto;
        min-height: 3;
        padding: 0 2;
        background: #181d22;
        border-top: solid #36414a;
    }

    #attachment-strip {
        display: none;
        height: 2;
        padding: 0 1;
        color: #b9c5cc;
        background: #151b1f;
        border-left: tall #6e8fa1;
    }

    #attachment-strip.active {
        display: block;
    }

    #command-palette {
        display: none;
        height: auto;
        max-height: 9;
        margin: 0 0 0 1;
        border-left: tall #4f8f83;
        border-bottom: solid #36414a;
        background: #151b1f;
        scrollbar-size: 1 1;
        scrollbar-color: #3d6e66;
    }

    #command-palette.active {
        display: block;
    }

    #thinking-waterfall {
        height: 3;
        display: none;
        color: #77c8b9;
    }

    #thinking-waterfall.active {
        display: block;
    }

    #composer {
        height: 2;
        border: none;
        border-left: heavy #4f8f83;
        padding: 0 1;
        background: #181d22;
    }

    #composer:focus {
        border-left: heavy #77c8b9;
        background: #181d22;
    }

    #inspector {
        width: 1fr;
        min-width: 36;
        padding: 0 1;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1;
        background: #14191d;
    }

    .panel-content {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }

    #space-map {
        height: 18;
        padding: 0 1;
    }

    #trace-list {
        height: 1fr;
        min-height: 5;
        border-top: solid #36414a;
        border-bottom: solid #36414a;
        scrollbar-size: 1 1;
        scrollbar-color: #3d6e66;
    }

    #trace-detail {
        height: 8;
        padding: 1;
        overflow-y: auto;
    }

    #status-line {
        height: 1;
        padding: 0 1;
        background: #181d22;
        color: #9fb0bd;
    }

    #status-line.busy {
        color: #f0c674;
    }

    #status-line.error {
        color: #e06c75;
    }

    Screen.narrow #workspace {
        layout: vertical;
    }

    Screen.narrow #conversation-column {
        width: 1fr;
        height: 12;
        min-width: 0;
        border-right: none;
        border-bottom: solid #36414a;
    }

    Screen.narrow #inspector {
        width: 1fr;
        height: 1fr;
        min-width: 0;
    }

    Screen.narrow #space-map {
        height: 14;
    }

    Screen.narrow #trace-list,
    Screen.narrow #trace-detail {
        display: none;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("f1", "show_help", "Commands"),
    ]

    def __init__(
        self,
        session: InteractiveSession,
        *,
        command_registry: CommandRegistry | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._history_loaded = False
        self._trace_nodes: dict[str, TraceNode] = {}
        self._last_snapshot: DashboardSnapshot | None = None
        self._lifecycle_ready = False
        self._lifecycle_polling = False
        self._rendered_event_ids: set[str] = set()
        self._active_stream: StreamActivity | None = None
        self._pending_attachments: list[Path] = []
        self._commands = command_registry or build_default_command_registry()
        self._command_handlers: dict[str, Callable[[ParsedCommand], None]] = {
            "attachment.add": self._command_attachment_add,
            "attachment.list": self._command_attachment_list,
            "attachment.remove": self._command_attachment_remove,
            "attachment.clear": self._command_attachment_clear,
            "tab.open": self._command_tab_open,
            "dashboard.refresh": self._command_refresh,
            "conversation.clear": self._command_clear,
            "commands.help": self._command_help,
            "app.quit": self._command_quit,
        }

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon="", time_format="%H:%M")
        with Horizontal(id="workspace"):
            with Vertical(id="conversation-column"):
                yield ConversationView(id="conversation")
                with Vertical(id="composer-shell"):
                    yield OptionList(id="command-palette", compact=True)
                    yield Static(id="attachment-strip")
                    yield ThinkingWaterfall(id="thinking-waterfall")
                    yield Input(
                        id="composer",
                        placeholder="Message the digital lifeform...",
                        max_length=16_000,
                    )
                    yield Static("READY", id="status-line")
            with Vertical(id="inspector"), TabbedContent(initial="space-tab"):
                with TabPane("SPACE", id="space-tab"):
                    yield Static(id="space-map")
                    yield OptionList(id="trace-list", compact=True)
                    yield Static(id="trace-detail")
                with TabPane("STATE", id="state-tab"):
                    yield OrganismStatePanel(id="state-panel", classes="panel-content")
                with TabPane("RELATION", id="relation-tab"):
                    yield RelationshipStatePanel(id="relationship-panel", classes="panel-content")
                with TabPane("MEMORY", id="memory-tab"):
                    yield Static(id="memory-panel", classes="panel-content")
                with TabPane("IDENTITY", id="identity-tab"):
                    yield Static(id="identity-panel", classes="panel-content")
                with TabPane("ACTIVITY", id="activity-tab"):
                    yield Static(id="activity-panel", classes="panel-content")

    def on_mount(self) -> None:
        self.screen.set_class(self.size.width < 100, "narrow")
        self._refresh_snapshot(self._session.snapshot(), load_history=True)
        composer = self.query_one("#composer", Input)
        composer.disabled = True
        self._set_busy(True, "INDEXING // building persistent trace space")
        self.run_worker(self._initialize_session(), group="initialize", exclusive=True)
        self.set_interval(2.0, self._schedule_lifecycle_poll, name="lifecycle-poll")

    async def _initialize_session(self) -> None:
        try:
            snapshot = await self._session.initialize()
        except Exception as exc:
            self._set_status(f"TRACE INDEX ERROR // {escape(str(exc))}", "error")
        else:
            self._refresh_snapshot(snapshot)
            self._set_status(f"READY   {snapshot.trace_space.total_traces} traces indexed")
            self._lifecycle_ready = True
        finally:
            composer = self.query_one("#composer", Input)
            composer.disabled = False
            composer.focus()

    def _schedule_lifecycle_poll(self) -> None:
        if not self._lifecycle_ready or self._lifecycle_polling:
            return
        self._lifecycle_polling = True
        self.run_worker(self._poll_lifecycle(), group="lifecycle")

    async def _poll_lifecycle(self) -> None:
        try:
            events = await self._session.poll_lifecycle()
            for event in events:
                if event.id not in self._rendered_event_ids:
                    self._write_event(event)
            if events:
                self._refresh_snapshot(self._session.snapshot())
                self._set_status(f"READY   {len(events)} proactive event delivered")
        except Exception as exc:
            self._set_status(f"LIFECYCLE ERROR // {escape(str(exc))}", "error")
        finally:
            self._lifecycle_polling = False

    def on_resize(self) -> None:
        self.screen.set_class(self.size.width < 100, "narrow")
        if self._last_snapshot is not None:
            self._refresh_trace_space(self._last_snapshot)

    def on_unmount(self) -> None:
        self._session.close()

    @on(Input.Submitted, "#composer")
    def submit_message(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            event.input.value = ""
            return
        if value.startswith("/"):
            if self._complete_command_from_palette(value):
                return
            event.input.value = ""
            self._hide_command_palette()
            self._run_command(value)
            return
        event.input.value = ""
        self._hide_command_palette()
        attachments = tuple(str(path) for path in self._pending_attachments)
        self._write_message(
            "YOU",
            _message_with_attachments(value, self._pending_attachments),
            "#77c8b9",
        )
        self._pending_attachments.clear()
        self._refresh_attachment_strip()
        self._active_stream = self.query_one(ConversationView).append_stream()
        if attachments:
            self._active_stream.analyzing_files(len(attachments))
        self.query_one(ThinkingWaterfall).start()
        self._set_busy(
            True,
            f"FILES // analyzing {len(attachments)}" if attachments else "BUSY",
        )
        event.input.disabled = True
        self.run_worker(self._send(value, attachments), group="chat", exclusive=True)

    @on(Input.Changed, "#composer")
    def update_command_palette(self, event: Input.Changed) -> None:
        matches = self._commands.suggest(event.value)
        palette = self.query_one("#command-palette", OptionList)
        palette.clear_options()
        if not matches:
            palette.remove_class("active")
            return
        palette.add_options(
            [
                Option(
                    Text.assemble(
                        (capability.usage, "bold #77c8b9"),
                        (f"  {capability.description}", "#87949d"),
                    ),
                    id=capability.name,
                )
                for capability in matches
            ]
        )
        palette.highlighted = 0
        palette.add_class("active")

    @on(OptionList.OptionSelected, "#command-palette")
    def select_palette_command(self, event: OptionList.OptionSelected) -> None:
        capability = self._commands.get(event.option.id or "")
        if capability is not None:
            self._apply_command_completion(capability)

    def on_key(self, event: events.Key) -> None:
        composer = self.query_one("#composer", Input)
        palette = self.query_one("#command-palette", OptionList)
        if self.focused is not composer or not palette.has_class("active"):
            return
        if event.key == "escape":
            self._hide_command_palette()
        elif event.key in {"up", "down"}:
            count = palette.option_count
            if count:
                current = palette.highlighted or 0
                delta = -1 if event.key == "up" else 1
                palette.highlighted = (current + delta) % count
        elif event.key in {"tab", "shift+tab"}:
            capability = self._selected_palette_command()
            if capability is not None:
                self._apply_command_completion(capability)
        else:
            return
        event.prevent_default()
        event.stop()

    def _complete_command_from_palette(self, value: str) -> bool:
        parsed = self._commands.parse(value)
        if parsed is not None and (
            not parsed.capability.requires_argument or parsed.argument
        ):
            return False
        capability = self._selected_palette_command()
        if capability is None:
            matches = self._commands.suggest(value, limit=1)
            capability = matches[0] if matches else None
        if capability is None:
            return False
        self._apply_command_completion(capability)
        return True

    def _selected_palette_command(self) -> CommandCapability | None:
        palette = self.query_one("#command-palette", OptionList)
        if not palette.has_class("active") or palette.highlighted is None:
            return None
        option = palette.get_option_at_index(palette.highlighted)
        return self._commands.get(option.id or "")

    def _apply_command_completion(self, capability: CommandCapability) -> None:
        composer = self.query_one("#composer", Input)
        suffix = " " if capability.requires_argument else ""
        composer.value = f"/{capability.name}{suffix}"
        composer.cursor_position = len(composer.value)
        composer.focus()
        if capability.requires_argument:
            self._hide_command_palette()

    def _hide_command_palette(self) -> None:
        palette = self.query_one("#command-palette", OptionList)
        palette.remove_class("active")

    async def _send(self, content: str, attachments: tuple[str, ...] = ()) -> None:
        try:
            result = await self._session.send(
                content,
                attachments=attachments,
                on_stream=self._handle_stream_event,
            )
        except Exception as exc:
            if self._active_stream is not None:
                self._active_stream.finish()
                self._active_stream = None
            self._write_message("SYSTEM", str(exc), "#e06c75")
            for raw_path in attachments:
                path = Path(raw_path)
                if path.exists() and path not in self._pending_attachments:
                    self._pending_attachments.append(path)
            self._refresh_attachment_strip()
            self._set_status(f"ERROR // {escape(str(exc))}", "error")
        else:
            self._render_turn(result)
        finally:
            self.query_one(ThinkingWaterfall).stop()
            composer = self.query_one("#composer", Input)
            composer.disabled = False
            composer.focus()

    def _handle_stream_event(self, event: SessionStreamEvent) -> None:
        activity = self._active_stream
        if activity is None:
            activity = self.query_one(ConversationView).append_stream()
            self._active_stream = activity
        activity.handle(event)
        self.query_one(ConversationView).keep_latest_visible()
        waterfall = self.query_one(ThinkingWaterfall)
        if event.kind == "assistant_start":
            waterfall.start()
            self._set_busy(True, f"STREAM // round {event.round_index}")
        elif event.kind == "assistant_delta":
            waterfall.stop()
            self._set_busy(True, "STREAM // receiving")
        elif event.kind == "assistant_done":
            waterfall.stop()
            phase = "waiting for tools" if event.has_tool_calls else "saving trace"
            self._set_busy(True, f"FINALIZING // {phase}")
        elif event.kind in {
            "tool_delta",
            "tool_requested",
            "tool_started",
            "tool_output",
            "tool_finished",
        }:
            waterfall.stop()
            tool_name = event.tool_name or "function"
            state = (
                "DONE"
                if event.kind == "tool_finished" and event.ok
                else "FAILED"
                if event.kind == "tool_finished"
                else "RUNNING"
            )
            self._set_busy(True, f"TOOL // {tool_name} {state}")

    def _render_turn(self, result: ChatTurnResult) -> None:
        if self._active_stream is None:
            self._write_message("SSA", result.agent_event.content, "#e8bd67")
        else:
            self._active_stream.complete(result.agent_event.content)
            self._active_stream = None
        self._refresh_snapshot(result.snapshot)
        response = result.response
        latency = (
            f"{response.latency_ms}ms"
            if response.latency_ms < 1_000
            else f"{response.latency_ms / 1_000:.1f}s"
        )
        status = Text()
        status.append("READY", style="bold #77c8b9")
        status.append(f"   {latency}", style="#9fb0bd")
        status.append(f"   in {response.input_tokens:,}", style="dim")
        status.append(f"   out {response.output_tokens:,}", style="dim")
        status.append(f"   cache {response.cache_hit_ratio:.0%}", style="dim")
        if result.tool_results:
            status.append(f"   tools {len(result.tool_results)}", style="dim")
        if result.learning_deferred:
            status.append("   learning bg", style="dim")
        if result.attachment_analysis is not None:
            status.append(
                f"   files {len(result.attachment_analysis.files)}",
                style="#8fb7c9",
            )
        self._set_status(status)

    def _refresh_snapshot(
        self,
        snapshot: DashboardSnapshot,
        *,
        load_history: bool = False,
    ) -> None:
        self._last_snapshot = snapshot
        self.query_one("#state-panel", OrganismStatePanel).set_snapshot(snapshot)
        self.query_one("#relationship-panel", RelationshipStatePanel).set_snapshot(snapshot)
        self.query_one("#memory-panel", Static).update(format_memory_panel(snapshot))
        self.query_one("#identity-panel", Static).update(format_identity_panel(snapshot))
        self.query_one("#activity-panel", Static).update(format_activity_panel(snapshot))
        self._refresh_trace_space(snapshot)
        if load_history and not self._history_loaded:
            for event in snapshot.events:
                self._write_event(event)
            self._history_loaded = True
        self.sub_title = (
            f"{snapshot.model} // {snapshot.event_count} events // "
            f"{snapshot.trace_space.total_traces} traces"
        )

    def _refresh_trace_space(self, snapshot: DashboardSnapshot) -> None:
        canvas_height = 4 if self.size.width < 100 else 8
        self.query_one("#space-map", Static).update(
            format_trace_map(snapshot, height=canvas_height)
        )
        trace_list = self.query_one("#trace-list", OptionList)
        trace_list.clear_options()
        ordered = sorted(
            snapshot.trace_space.nodes,
            key=lambda node: (
                node.activation_kind is None,
                -node.activation_score,
                -node.created_at_ms,
            ),
        )
        self._trace_nodes = {node.trace_id: node for node in ordered}
        trace_list.add_options(
            [
                Option(
                    Text.assemble(
                        (
                            "@ "
                            if node.activation_kind == "main"
                            else "+ "
                            if node.activation_kind == "radiation"
                            else ". ",
                            "bold #f0c674"
                            if node.activation_kind == "main"
                            else "bold #77c8b9"
                            if node.activation_kind == "radiation"
                            else "dim",
                        ),
                        (f"{node.activation_score:0.3f} ", "dim"),
                        node.content.replace("\n", " ")[:54],
                    ),
                    id=node.trace_id,
                )
                for node in ordered
            ]
        )
        if ordered:
            trace_list.highlighted = 0
            self.query_one("#trace-detail", Static).update(format_trace_detail(ordered[0]))
        else:
            self.query_one("#trace-detail", Static).update(format_trace_detail(None))

    @on(OptionList.OptionHighlighted, "#trace-list")
    def show_trace_detail(self, event: OptionList.OptionHighlighted) -> None:
        trace_id = event.option.id
        node = self._trace_nodes.get(trace_id) if trace_id is not None else None
        self.query_one("#trace-detail", Static).update(format_trace_detail(node))

    def _write_event(self, event: Event) -> None:
        self._rendered_event_ids.add(event.id)
        if event.actor.value == "user":
            attachment_paths: list[Path] = []
            attachments = event.metadata.get("attachments")
            if isinstance(attachments, list):
                for item in attachments:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        attachment_paths.append(Path(item["path"]))
            self._write_message(
                "YOU",
                _message_with_attachments(event.content, attachment_paths),
                "#77c8b9",
            )
        elif event.actor.value == "agent":
            self._write_message("SSA", event.content, "#f0c674")

    def _write_message(self, author: str, content: str, color: str) -> None:
        role = "user" if author == "YOU" else "assistant" if author == "SSA" else "system"
        self.query_one(ConversationView).append_message(
            author,
            content,
            color,
            role=role,
        )

    def _run_command(self, command: str) -> None:
        parsed = self._commands.parse(command)
        if parsed is None:
            self._set_status(f"UNKNOWN COMMAND // {escape(command)}", "error")
            return
        if parsed.capability.requires_argument and not parsed.argument:
            self._set_status(
                f"COMMAND ARGUMENT REQUIRED // {parsed.capability.usage}",
                "error",
            )
            return
        handler = self._command_handlers.get(parsed.capability.action)
        if handler is None:
            self._set_status(
                f"COMMAND HANDLER MISSING // {escape(parsed.capability.action)}",
                "error",
            )
            return
        handler(parsed)

    def action_refresh(self) -> None:
        self._refresh_snapshot(self._session.snapshot())
        self._set_status("READY   dashboard refreshed")

    def action_clear_log(self) -> None:
        self.query_one(ConversationView).clear_messages()
        self._active_stream = None
        self._set_status("READY   conversation view cleared")

    def action_show_help(self) -> None:
        self._render_command_help()

    def _command_attachment_add(self, command: ParsedCommand) -> None:
        self._queue_attachment(command.argument)

    def _command_attachment_list(self, _command: ParsedCommand) -> None:
        self._refresh_attachment_strip()
        count = len(self._pending_attachments)
        self._set_status(f"ATTACHMENTS // {count} queued")

    def _command_attachment_remove(self, command: ParsedCommand) -> None:
        self._remove_attachment(command.argument)

    def _command_attachment_clear(self, _command: ParsedCommand) -> None:
        self._pending_attachments.clear()
        self._refresh_attachment_strip()
        self._set_status("ATTACHMENTS // queue cleared")

    def _command_tab_open(self, command: ParsedCommand) -> None:
        self.query_one(TabbedContent).active = command.capability.payload

    def _command_refresh(self, _command: ParsedCommand) -> None:
        self.action_refresh()

    def _command_clear(self, _command: ParsedCommand) -> None:
        self.action_clear_log()

    def _command_help(self, _command: ParsedCommand) -> None:
        self._render_command_help()

    def _command_quit(self, _command: ParsedCommand) -> None:
        self.exit()

    def _render_command_help(self) -> None:
        lines = [
            f"- `{capability.usage}` {capability.description}"
            for capability in self._commands.capabilities
        ]
        self._write_message(
            "SYSTEM",
            "**Registered commands**\n\n" + "\n".join(lines),
            "#9fb0bd",
        )

    def _queue_attachment(self, raw_path: str) -> None:
        value = raw_path.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            self._set_status(f"ATTACHMENT ERROR // {escape(str(exc))}", "error")
            return
        if not path.is_file():
            self._set_status(f"ATTACHMENT ERROR // not a file: {escape(str(path))}", "error")
            return
        if path in self._pending_attachments:
            self._set_status(f"ATTACHMENT // already queued: {escape(path.name)}")
            return
        if len(self._pending_attachments) >= 8:
            self._set_status("ATTACHMENT ERROR // at most 8 files per message", "error")
            return
        self._pending_attachments.append(path)
        self._refresh_attachment_strip()
        self._set_status(f"ATTACHMENT // queued {escape(path.name)}")

    def _remove_attachment(self, raw_index: str) -> None:
        try:
            index = int(raw_index.strip()) - 1
            if index < 0:
                raise IndexError
            path = self._pending_attachments.pop(index)
        except (ValueError, IndexError):
            self._set_status("ATTACHMENT ERROR // /detach expects a queued file number", "error")
            return
        self._refresh_attachment_strip()
        self._set_status(f"ATTACHMENT // removed {escape(path.name)}")

    def _refresh_attachment_strip(self) -> None:
        strip = self.query_one("#attachment-strip", Static)
        strip.set_class(bool(self._pending_attachments), "active")
        if not self._pending_attachments:
            strip.update("")
            return
        output = Text("FILES  ", style="bold #8fb7c9")
        for index, path in enumerate(self._pending_attachments, start=1):
            if index > 1:
                output.append("   ", style="dim")
            output.append(f"{index}:", style="#78909c")
            output.append(f"▣ {path.name}", style="#d7e2e8")
        output.append("   send with next message", style="dim")
        strip.update(output)

    def _set_busy(self, busy: bool, message: str) -> None:
        self._set_status(message, "busy" if busy else "")

    def _set_status(self, message: str | Text, status_class: str = "") -> None:
        status = self.query_one("#status-line", Static)
        status.set_classes(status_class)
        status.update(message)


def _message_with_attachments(content: str, paths: list[Path]) -> str:
    if not paths:
        return content
    file_lines = "\n".join(f"- `▣ {path.name}`" for path in paths)
    return f"{content}\n\n{file_lines}"


def run_tui(settings: Settings, *, conversation_id: str = "cli-primary") -> int:
    """Build and run the production terminal application."""
    session = build_interactive_session(settings, conversation_id=conversation_id)
    DigitalLifeApp(session).run()
    return 0


__all__ = [
    "ConversationView",
    "DigitalLifeApp",
    "MessageBlock",
    "OrganismStatePanel",
    "RelationshipStatePanel",
    "StreamActivity",
    "ThinkingWaterfall",
    "run_tui",
]
