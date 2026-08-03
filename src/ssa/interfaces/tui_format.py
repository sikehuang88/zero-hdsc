"""Rich markup formatting for SSA terminal dashboard panels."""

from __future__ import annotations

from datetime import UTC, datetime

from rich.markup import escape

from ssa.domain.traces import TraceNode
from ssa.runtime.interactive import DashboardSnapshot


def _battery(value: float, *, animation_frame: int, width: int = 14) -> str:
    bounded = max(0.0, min(1.0, value))
    filled = round(bounded * width)
    color = "#66b3a6" if bounded >= 0.6 else "#f0c674" if bounded >= 0.3 else "#e06c75"
    outline = "#56616a"
    cells: list[str] = []
    head = animation_frame % filled if filled else -1
    trail = (head - 1) % filled if filled > 1 else -1

    for index in range(width):
        if index >= filled:
            cells.append("[#36414a]▱[/]")
        elif index == head:
            glyph = "█" if filled > 1 or animation_frame % 2 else "▓"
            cells.append(f"[bold #dffcf7]{glyph}[/]")
        elif index == trail:
            cells.append("[#9ddbd0]▓[/]")
        else:
            cells.append(f"[{color}]█[/]")

    return (
        f"[{outline}]╞[/]"
        + "".join(cells)
        + f"[{outline}]╡[/][{color}]▌[/] [bold]{bounded:>4.0%}[/]"
    )


def _battery_metric(
    name: str,
    value: float,
    *,
    animation_frame: int,
    phase: int,
    width: int,
) -> str:
    battery = _battery(value, animation_frame=animation_frame + phase, width=width)
    return f"[bold]{name:<10}[/] {battery}"


def format_trace_map(
    snapshot: DashboardSnapshot,
    *,
    width: int = 38,
    height: int = 8,
) -> str:
    """Render the projected trace geometry and latest activated region."""
    space = snapshot.trace_space
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    positions = {
        node.trace_id: (
            min(width - 1, round(node.x * (width - 1))),
            min(height - 1, round((1.0 - node.y) * (height - 1))),
        )
        for node in space.nodes
    }
    newest_trace_id = (
        max(space.nodes, key=lambda node: node.created_at_ms).trace_id if space.nodes else None
    )
    for link in space.links:
        start = positions.get(link.source_trace_id)
        end = positions.get(link.target_trace_id)
        if start is not None and end is not None:
            _draw_edge(
                canvas,
                start,
                end,
                directed=link.link_type != "semantic",
            )

    priority = {".": 1, "o": 2, "+": 3, "@": 4}
    ordered = sorted(
        space.nodes,
        key=lambda node: priority[_trace_symbol(node, newest_trace_id)],
    )
    for node in ordered:
        x, y = positions[node.trace_id]
        symbol = _trace_symbol(node, newest_trace_id)
        if priority.get(symbol, 0) >= priority.get(canvas[y][x], 0):
            canvas[y][x] = symbol

    shadow = escape(space.shadow_model_id or "disabled")
    lines = [
        f"[bold #77c8b9]TRACE SPACE[/]  [dim]{space.projection}[/]",
        f"[dim]archive {space.total_traces} // view {space.visible_count}[/]",
        f"[dim]serving {escape(space.serving_model_id)} // {space.activated_count} activated[/]",
        f"[dim]shadow {shadow} // {space.shadow_status.upper()} // no prompt effect[/]",
    ]
    audit = space.stability_audit
    if audit is None:
        lines.append("[dim]active n/a // local not run // loop not measured[/]")
    else:
        active_mass = audit.active_mass if audit.active_mass is not None else 0.0
        active_count = audit.active_count if audit.active_count is not None else 0
        lines.append(
            f"[dim]active {active_count}/{audit.active_capacity} clusters // "
            f"mass {active_mass:.3f}/{audit.mass_budget:.3f} // null "
            f"{(audit.null_mass or 0.0):.3f}[/]"
        )
        loop = audit.closed_loop_gate.replace("-", " ")
        lines.append(
            f"[dim]local {audit.local_gate} // loop {loop} // "
            f"beta {audit.contraction_bound or 0.0:.2f} // "
            f"churn {audit.active_set_churn or 0.0:.0%}[/]"
        )
    lines.append("+" + "-" * width + "+")
    color_map = {
        "@": "[bold #f0c674]@[/]",
        "+": "[bold #77c8b9]+[/]",
        "o": "[#d8dee9]o[/]",
        ".": "[dim].[/]",
        ":": "[dim]:[/]",
        ">": "[bold #e06c75]>[/]",
        "<": "[bold #e06c75]<[/]",
        "^": "[bold #e06c75]^[/]",
        "v": "[bold #e06c75]v[/]",
    }
    for row in canvas:
        lines.append("|" + "".join(color_map.get(char, char) for char in row) + "|")
    lines.extend(
        [
            "+" + "-" * width + "+",
            "[bold #f0c674]@[/] main  [bold #77c8b9]+[/] radiation  "
            "[#d8dee9]o[/] new  [dim].[/] trace  [dim]:[/] sym  "
            "[bold #e06c75]>[/] dir",
        ]
    )
    if space.latest_query:
        lines.append(f"[dim]query[/] {escape(space.latest_query[:60])}")
    return "\n".join(lines)


def format_trace_detail(node: TraceNode | None) -> str:
    if node is None:
        return "[dim]Select a trace to inspect its activation evidence.[/]"
    activation = node.activation_kind or "inactive"
    return "\n".join(
        [
            f"[bold]{escape(node.content_type.upper())}[/]  [dim]{escape(node.trace_id[:12])}[/]",
            f"[dim]activation[/] {activation}  "
            f"[dim]score[/] {node.activation_score:.3f}  "
            f"[dim]fresh[/] {node.freshness:.0%}  "
            f"[dim]importance[/] {node.importance:.0%}",
            escape(node.content[:420]),
        ]
    )


def _trace_symbol(node: TraceNode, newest_trace_id: str | None) -> str:
    if node.activation_kind == "main":
        return "@"
    if node.activation_kind == "radiation":
        return "+"
    return "o" if node.trace_id == newest_trace_id else "."


def _draw_edge(
    canvas: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    directed: bool = False,
) -> None:
    """Draw a Bresenham edge and an optional arrow before the target."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    path: list[tuple[int, int]] = []
    while (x0, y0) != (x1, y1):
        path.append((x0, y0))
        if canvas[y0][x0] == " ":
            canvas[y0][x0] = ":"
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += step_x
        if twice_error <= dx:
            error += dx
            y0 += step_y
    path.append((x1, y1))
    if directed and len(path) >= 3:
        arrow_x, arrow_y = path[-2]
        delta_x = x1 - arrow_x
        delta_y = y1 - arrow_y
        arrow = (
            ">"
            if abs(delta_x) >= abs(delta_y) and delta_x > 0
            else "<"
            if abs(delta_x) >= abs(delta_y)
            else "v"
            if delta_y > 0
            else "^"
        )
        canvas[arrow_y][arrow_x] = arrow


def format_state_panel(
    snapshot: DashboardSnapshot,
    *,
    animation_frame: int = 0,
    battery_width: int = 14,
) -> str:
    state = snapshot.organism
    valence = (state.valence + 1.0) / 2.0
    perception = snapshot.latest_perception
    perception_lines = (
        [
            f"[bold #77c8b9]ENVIRONMENT[/]  [dim]{escape(perception.model_id)}[/]",
            f"[dim]clock[/] {escape(perception.time.human_clock)}",
            f"[dim]mode[/] {escape(perception.primary_mode.value)} "
            f"{perception.confidence:.0%}  "
            f"[dim]gap[/] {escape(perception.time.conversation_gap.value)}",
            "",
        ]
        if perception is not None
        else ["[bold #77c8b9]ENVIRONMENT[/]  [dim]awaiting signal[/]", ""]
    )
    return "\n".join(
        [
            *perception_lines,
            f"[bold #77c8b9]ORGANISM[/]  [dim]v{state.version}[/]",
            "",
            _battery_metric(
                "energy",
                state.energy,
                animation_frame=animation_frame,
                phase=0,
                width=battery_width,
            ),
            _battery_metric(
                "connection",
                state.connection_need,
                animation_frame=animation_frame,
                phase=2,
                width=battery_width,
            ),
            _battery_metric(
                "autonomy",
                state.autonomy_need,
                animation_frame=animation_frame,
                phase=4,
                width=battery_width,
            ),
            _battery_metric(
                "curiosity",
                state.curiosity,
                animation_frame=animation_frame,
                phase=6,
                width=battery_width,
            ),
            _battery_metric(
                "safety",
                state.safety,
                animation_frame=animation_frame,
                phase=8,
                width=battery_width,
            ),
            _battery_metric(
                "valence",
                valence,
                animation_frame=animation_frame,
                phase=10,
                width=battery_width,
            ),
            _battery_metric(
                "arousal",
                state.arousal,
                animation_frame=animation_frame,
                phase=12,
                width=battery_width,
            ),
        ]
    )


def format_relationship_panel(
    snapshot: DashboardSnapshot,
    *,
    animation_frame: int = 0,
    battery_width: int = 14,
) -> str:
    relation = snapshot.relationship
    return "\n".join(
        [
            f"[bold #77c8b9]RELATIONSHIP[/]  [dim]v{relation.version}[/]",
            "",
            _battery_metric(
                "trust",
                relation.trust,
                animation_frame=animation_frame,
                phase=1,
                width=battery_width,
            ),
            _battery_metric(
                "closeness",
                relation.closeness,
                animation_frame=animation_frame,
                phase=4,
                width=battery_width,
            ),
            _battery_metric(
                "tension",
                relation.tension,
                animation_frame=animation_frame,
                phase=7,
                width=battery_width,
            ),
            _battery_metric(
                "reciprocity",
                relation.reciprocity,
                animation_frame=animation_frame,
                phase=10,
                width=battery_width,
            ),
            _battery_metric(
                "repair debt",
                relation.repair_debt,
                animation_frame=animation_frame,
                phase=13,
                width=battery_width,
            ),
            "",
            f"[dim]commitments[/] {len(relation.active_commitment_ids)}",
            f"[dim]unresolved[/]  {len(relation.unresolved_memory_ids)}",
            f"[dim]rituals[/]     {len(relation.shared_ritual_ids)}",
        ]
    )


def format_memory_panel(snapshot: DashboardSnapshot) -> str:
    lines = [
        f"[bold #77c8b9]ACTIVE MEMORY[/]  [dim]{len(snapshot.memories)} shown[/]",
        "",
    ]
    if not snapshot.memories:
        lines.append("[dim]No persisted memories yet.[/]")
    for memory in snapshot.memories:
        lines.extend(
            [
                f"[bold]{escape(memory.memory_type.value.upper())}[/] "
                f"[dim]{memory.confidence:.0%} confidence[/]",
                escape(memory.content),
                "",
            ]
        )
    return "\n".join(lines)


def format_identity_panel(snapshot: DashboardSnapshot) -> str:
    lines = [
        f"[bold #77c8b9]SELF BELIEFS[/]  [dim]{len(snapshot.beliefs)} shown[/]",
        "",
    ]
    if not snapshot.beliefs:
        lines.append("[dim]No evidence-backed identity claims yet.[/]")
    for belief in snapshot.beliefs:
        lines.extend(
            [
                f"[bold]{escape(belief.status.value.upper())}[/] "
                f"[dim]v{belief.version} // {belief.confidence:.0%}[/]",
                escape(belief.claim),
                "",
            ]
        )
    return "\n".join(lines)


def format_activity_panel(snapshot: DashboardSnapshot) -> str:
    runtime = "PROCESS LIVE" if snapshot.offline_runtime_enabled else "PROCESS DISABLED"
    artifacts_by_episode = {
        artifact.episode_id: artifact for artifact in snapshot.offline_artifacts
    }
    evidence_counts: dict[str, int] = {}
    for evidence in snapshot.learning_evidence:
        evidence_counts[evidence.proposal_id] = evidence_counts.get(evidence.proposal_id, 0) + 1
    lines = [
        f"[bold #77c8b9]OFFLINE AGENCY[/]  [bold]{runtime}[/]",
        f"[dim]open goals[/] {len(snapshot.goals)}   "
        f"[dim]recent episodes[/] {len(snapshot.offline_episodes)}   "
        f"[dim]artifacts[/] {len(snapshot.offline_artifacts)}",
        f"[dim]reflections[/] {len(snapshot.reflection_runs)}   "
        f"[dim]proposals[/] {len(snapshot.learning_proposals)}   "
        f"[dim]experiments[/] {len(snapshot.behavior_experiments)}",
        "",
    ]
    inner = snapshot.inner_state
    if inner is None:
        lines.extend(["[bold]INNER LOOP[/] [dim]awaiting first heartbeat[/]", ""])
    else:
        heartbeat = datetime.fromtimestamp(inner.last_heartbeat_at_ms / 1_000, UTC).strftime(
            "%m-%d %H:%M:%S"
        )
        deadline = (
            datetime.fromtimestamp(inner.wait_deadline_at_ms / 1_000, UTC).strftime("%m-%d %H:%M")
            if inner.wait_deadline_at_ms is not None
            else "none"
        )
        lines.extend(
            [
                f"[bold]INNER LOOP[/] [bold #f0c674]{inner.mode.value.upper()}[/] "
                f"[dim]// heartbeat {heartbeat} UTC[/]",
                f"[dim]reply {inner.reply_expectation:.0%} // concern {inner.concern:.0%} // "
                f"curiosity {inner.curiosity:.0%} // offline {inner.offline_readiness:.0%}[/]",
                f"[dim]wait deadline {deadline} UTC // {escape(inner.transition_reason)}[/]",
                "",
            ]
        )
    if snapshot.learning_proposals:
        lines.extend(["[bold #77c8b9]LEARNING LOOP[/]", ""])
        for run in snapshot.reflection_runs[:4]:
            lines.extend(
                [
                    f"[bold]REFLECTION[/] {escape(run.status.value.upper())} "
                    f"[dim]// priority {run.priority_score:.0%} // "
                    f"critic {(run.critic_score or 0.0):.0%}[/]",
                    f"[dim]artifact {escape(run.source_artifact_id or 'none')} // "
                    f"trigger {escape(run.trigger_kind.value)}[/]",
                    "",
                ]
            )
        for proposal in snapshot.learning_proposals[:6]:
            target = (
                f" // {proposal.target_kind}:{proposal.target_id}"
                if proposal.target_kind and proposal.target_id
                else ""
            )
            lines.extend(
                [
                    f"[bold]{escape(proposal.status.value.upper())}[/] "
                    f"[dim]{escape(proposal.proposal_type.value)} // "
                    f"critic {proposal.critic_score:.0%} // "
                    f"evidence {evidence_counts.get(proposal.id, 0)}"
                    f"{escape(target)}[/]",
                    escape(proposal.title),
                    "",
                ]
            )
    for experiment in snapshot.behavior_experiments[:4]:
        lines.extend(
            [
                f"[bold]EXPERIMENT[/] {escape(experiment.status.value.upper())} "
                f"[dim]// {escape(experiment.policy_key)}[/]",
                escape(experiment.hypothesis),
                (
                    f"[dim]result {experiment.result_score:+.2f} // "
                    f"{escape(experiment.conclusion or 'pending')}[/]"
                    if experiment.result_score is not None
                    else f"[dim]due_at_ms {experiment.due_at_ms}[/]"
                ),
                "",
            ]
        )
    for goal in snapshot.goals[:4]:
        lines.extend(
            [
                f"[bold]GOAL[/] {escape(goal.title)}",
                f"[dim]{goal.status.value} // progress {goal.progress:.0%} // "
                f"priority {goal.priority:.0%}[/]",
                "",
            ]
        )
    if not snapshot.offline_episodes:
        lines.append("[dim]No offline episode has completed yet.[/]")
    for episode in snapshot.offline_episodes:
        stamp = datetime.fromtimestamp(episode.started_at_ms / 1_000, UTC).strftime("%m-%d %H:%M")
        color = "#77c8b9" if episode.status.value == "completed" else "#e06c75"
        artifact = artifacts_by_episode.get(episode.id)
        lines.extend(
            [
                f"[bold {color}]{escape(episode.status.value.upper())}[/] "
                f"[dim]{stamp} // {escape(episode.action_kind.value)}[/]",
                escape(episode.summary or episode.error or episode.motive),
                (
                    f"[dim]provider {escape(episode.provider)}[/]"
                    if episode.provider
                    else "[dim]source persistent trace[/]"
                ),
            ]
        )
        if artifact is not None:
            evidence_summary = (
                f"events {len(artifact.evidence_event_ids)} // "
                f"traces {len(artifact.evidence_trace_ids)}"
            )
            lines.extend(
                [
                    f"[bold]ARTIFACT[/] {escape(artifact.title)} "
                    f"[dim]// {escape(artifact.artifact_type)}[/]",
                    escape(" ".join(artifact.content.split())[:400]),
                    f"[dim]evidence {evidence_summary} // id {escape(artifact.id)}[/]",
                ]
            )
        elif episode.status.value == "completed":
            lines.append("[bold #e06c75]EVIDENCE GAP // artifact missing[/]")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "format_activity_panel",
    "format_identity_panel",
    "format_memory_panel",
    "format_relationship_panel",
    "format_state_panel",
    "format_trace_detail",
    "format_trace_map",
]
