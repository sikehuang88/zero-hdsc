"""Reliable outbox delivery and local/desktop channel adapters."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ssa.clock import Clock
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import Initiative, InitiativeStatus, OutboxMessage, OutboxStatus
from ssa.ids import IdGenerator
from ssa.services.proactive_contact_policy import ProactiveContactPolicy
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import InitiativeRepository, OutboxRepository


@dataclass(frozen=True)
class DeliveryReceipt:
    channel_message_id: str


@runtime_checkable
class ChannelAdapter(Protocol):
    async def deliver(self, message: OutboxMessage) -> DeliveryReceipt: ...


class LocalEventChannel:
    """A zero-side-effect channel consumed from the local event ledger by the desktop client."""

    async def deliver(self, message: OutboxMessage) -> DeliveryReceipt:
        return DeliveryReceipt(channel_message_id=message.id)


class WindowsDesktopChannel:
    """Best-effort Windows toast delivery through the built-in WinRT PowerShell API."""

    def __init__(self, powershell_executable: str = "powershell.exe") -> None:
        self._powershell = powershell_executable

    async def deliver(self, message: OutboxMessage) -> DeliveryReceipt:
        content = str(message.payload.get("content", "")).strip()
        if not content:
            raise ValueError("desktop notification content is empty")
        title = str(message.payload.get("title", "HDSC"))
        script = _windows_toast_script(title, content)
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        process = await asyncio.create_subprocess_exec(
            self._powershell,
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = (
                stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            )
            raise RuntimeError(f"desktop notification failed ({process.returncode}): {detail}")
        return DeliveryReceipt(channel_message_id=message.id)


@dataclass(frozen=True)
class DeliveryBatchResult:
    claimed: int
    delivered: int
    failed: int
    dead_lettered: int
    events: tuple[Event, ...]


DeliveryHook = Callable[[Initiative, Event], None]


class OutboxDeliveryService:
    """Claim due messages, invoke channels, and commit one agent event per outbox row."""

    def __init__(
        self,
        *,
        outbox: OutboxRepository,
        initiatives: InitiativeRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
        proactive_policy: ProactiveContactPolicy | None = None,
        adapters: Mapping[str, ChannelAdapter] | None = None,
        on_delivered: DeliveryHook | None = None,
        lease_ms: int = 60_000,
        max_attempts: int = 5,
    ) -> None:
        self._outbox = outbox
        self._initiatives = initiatives
        self._events = events
        self._clock = clock
        self._ids = ids
        self._proactive_policy = proactive_policy
        self._adapters = dict(adapters or {"local": LocalEventChannel()})
        self._on_delivered = on_delivered
        self._lease_ms = lease_ms
        self._max_attempts = max_attempts

    async def deliver_due(self, *, limit: int = 8) -> DeliveryBatchResult:
        now_ms = self._clock.now_ms()
        if (
            self._proactive_policy is not None
            and self._proactive_policy.gate(now_ms) is not None
        ):
            return DeliveryBatchResult(0, 0, 0, 0, ())
        claimed = self._outbox.claim_due(now_ms, lease_ms=self._lease_ms, limit=limit)
        delivered_events: list[Event] = []
        failed = 0
        dead_lettered = 0
        for message in claimed:
            try:
                adapter = self._adapters.get(message.channel)
                if adapter is None:
                    raise LookupError(f"no channel adapter registered for {message.channel!r}")
                initiative = self._initiative_for(message)
                await adapter.deliver(message)
                event = self._delivery_event(message, initiative)
                self._outbox.mark_delivered(message.id, self._clock.now_ms(), event.id)
                sent = self._initiatives.update_status(
                    initiative.id,
                    InitiativeStatus.SENT,
                    self._clock.now_ms(),
                    sent_event_id=event.id,
                )
                if self._on_delivered is not None:
                    self._on_delivered(sent, event)
                delivered_events.append(event)
            except Exception as exc:
                failed += 1
                retry_delay_ms = min(
                    60 * 60 * 1_000,
                    30_000 * (2 ** max(0, message.attempt_count - 1)),
                )
                current = self._outbox.fail(
                    message.id,
                    f"{type(exc).__name__}: {exc}",
                    self._clock.now_ms() + retry_delay_ms,
                    max_attempts=self._max_attempts,
                )
                if current.status == OutboxStatus.DEAD_LETTER:
                    dead_lettered += 1
        return DeliveryBatchResult(
            claimed=len(claimed),
            delivered=len(delivered_events),
            failed=failed,
            dead_lettered=dead_lettered,
            events=tuple(delivered_events),
        )

    def _initiative_for(self, message: OutboxMessage) -> Initiative:
        if message.initiative_id is None:
            raise ValueError("outbox message has no initiative")
        initiative = self._initiatives.get(message.initiative_id)
        if initiative is None:
            raise LookupError(f"initiative {message.initiative_id!r} does not exist")
        return initiative

    def _delivery_event(self, message: OutboxMessage, initiative: Initiative) -> Event:
        channel = f"outbox:{message.channel}"
        existing = self._events.find_by_channel_message(channel, message.id)
        if existing is not None:
            return existing
        content = str(message.payload.get("content", "")).strip()
        if not content:
            raise ValueError("outbox content is empty")
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.AGENT,
                    signal_type="agent.initiative",
                    content=content,
                    channel=channel,
                    channel_message_id=message.id,
                    conversation_id=initiative.conversation_id,
                    metadata={
                        "initiative_id": initiative.id,
                        "outbox_id": message.id,
                        "intent": initiative.intent,
                        "motive": initiative.motive,
                        "proactive": True,
                    },
                ),
                event_id=self._ids.new(),
                correlation_id=message.correlation_id,
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.AGENT_OUTPUT,
            )
        )


def _windows_toast_script(title: str, content: str) -> str:
    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = (
        "<toast><visual><binding template='ToastGeneric'>"
        f"<text>{escaped_title}</text><text>{escaped_content}</text>"
        "</binding></visual></toast>"
    )
    return (
        "$ErrorActionPreference='Stop';"
        "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,"
        "ContentType=WindowsRuntime] > $null;"
        "[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,"
        "ContentType=WindowsRuntime] > $null;"
        f"$xml=New-Object Windows.Data.Xml.Dom.XmlDocument;$xml.LoadXml('{xml}');"
        "$toast=[Windows.UI.Notifications.ToastNotification]::new($xml);"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('HDSC')"
        ".Show($toast);"
    )


__all__ = [
    "ChannelAdapter",
    "DeliveryBatchResult",
    "DeliveryReceipt",
    "LocalEventChannel",
    "OutboxDeliveryService",
    "WindowsDesktopChannel",
]
