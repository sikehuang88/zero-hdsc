"""Clock, file, calendar, and foreground-window world observations."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ssa.clock import Clock
from ssa.config import WorldConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, normalize_signal
from ssa.domain.lifecycle import WorldObservation
from ssa.ids import IdGenerator
from ssa.storage.event_repository import SqliteEventRepository
from ssa.storage.lifecycle_repository import WorldObservationRepository


@dataclass(frozen=True)
class WorldObservationResult:
    observations: tuple[WorldObservation, ...]
    events: tuple[Event, ...]
    material_count: int


@dataclass(frozen=True)
class _WorldSignal:
    observation_type: str
    source: str
    summary: str
    payload: dict[str, Any]
    salience: float
    dedup_key: str


class WorldObservationService:
    """Poll deterministic host sources and promote accepted changes into events."""

    def __init__(
        self,
        *,
        observations: WorldObservationRepository,
        events: SqliteEventRepository,
        clock: Clock,
        ids: IdGenerator,
        config: WorldConfig,
        timezone: str,
    ) -> None:
        self._observations = observations
        self._events = events
        self._clock = clock
        self._ids = ids
        self._config = config
        self._timezone = ZoneInfo(timezone)

    def observe(self, conversation_id: str) -> WorldObservationResult:
        if not self._config.enabled:
            return WorldObservationResult((), (), 0)
        now_ms = self._clock.now_ms()
        signals = [self._clock_signal(now_ms)]
        signals.extend(self._file_signals())
        signals.extend(self._calendar_signals(now_ms))
        foreground = self._foreground_signal(now_ms)
        if foreground is not None:
            signals.append(foreground)

        stored: list[WorldObservation] = []
        events: list[Event] = []
        for signal in signals:
            if self._observations.find_by_dedup(signal.dedup_key) is not None:
                continue
            event = self._append_event(conversation_id, signal)
            observation = self._observations.insert(
                WorldObservation(
                    id=self._ids.new(),
                    conversation_id=conversation_id,
                    observation_type=signal.observation_type,
                    source=signal.source,
                    summary=signal.summary,
                    payload=signal.payload,
                    salience=signal.salience,
                    dedup_key=signal.dedup_key,
                    observed_at_ms=now_ms,
                    event_id=event.id,
                )
            )
            stored.append(observation)
            events.append(event)
        return WorldObservationResult(
            observations=tuple(stored),
            events=tuple(events),
            material_count=sum(item.salience >= self._config.material_salience for item in stored),
        )

    def _clock_signal(self, now_ms: int) -> _WorldSignal:
        local = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone)
        hour = local.hour
        if hour < 5:
            phase = "late_night"
        elif hour < 8:
            phase = "early_morning"
        elif hour < 12:
            phase = "morning"
        elif hour < 18:
            phase = "afternoon"
        elif hour < 23:
            phase = "evening"
        else:
            phase = "late_night"
        key = f"clock:{local.date().isoformat()}:{phase}:{self._timezone.key}"
        return _WorldSignal(
            observation_type="clock",
            source="system_clock",
            summary=f"Local time entered the {phase.replace('_', ' ')} phase.",
            payload={
                "local_iso": local.isoformat(),
                "day_phase": phase,
                "timezone": self._timezone.key,
                "weekday": local.strftime("%A"),
            },
            salience=0.20,
            dedup_key=key,
        )

    def _file_signals(self) -> list[_WorldSignal]:
        signals: list[_WorldSignal] = []
        for configured_path in self._config.watched_paths:
            path = Path(configured_path).expanduser()
            try:
                stat = path.stat()
            except OSError:
                digest = _digest(str(path.resolve(strict=False)))
                signals.append(
                    _WorldSignal(
                        observation_type="file_missing",
                        source="filesystem",
                        summary=f"A watched path is currently unavailable: {path.name}",
                        payload={"path": str(path.resolve(strict=False))},
                        salience=0.35,
                        dedup_key=f"file-missing:{digest}",
                    )
                )
                continue
            resolved = str(path.resolve())
            fingerprint = f"{resolved}|{stat.st_mtime_ns}|{stat.st_size}|{stat.st_mode}"
            signals.append(
                _WorldSignal(
                    observation_type="file_change",
                    source="filesystem",
                    summary=f"A watched path changed: {path.name}",
                    payload={
                        "path": resolved,
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                        "is_directory": path.is_dir(),
                    },
                    salience=0.65,
                    dedup_key=f"file:{_digest(fingerprint)}",
                )
            )
        return signals

    def _calendar_signals(self, now_ms: int) -> list[_WorldSignal]:
        signals: list[_WorldSignal] = []
        for configured_path in self._config.calendar_json_paths:
            path = Path(configured_path).expanduser()
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries = raw.get("events", []) if isinstance(raw, dict) else raw
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                title = str(entry.get("title") or entry.get("summary") or "calendar event")
                event_key = str(entry.get("id") or f"{path}:{index}:{title}")
                fingerprint = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                signals.append(
                    _WorldSignal(
                        observation_type="calendar",
                        source="calendar_json",
                        summary=f"Calendar context: {title}",
                        payload={"calendar_path": str(path.resolve()), "event": entry},
                        salience=float(max(0.0, min(1.0, entry.get("salience", 0.75)))),
                        dedup_key=f"calendar:{_digest(event_key + '|' + fingerprint)}",
                    )
                )
        return signals

    def _foreground_signal(self, now_ms: int) -> _WorldSignal | None:
        if not self._config.observe_foreground_window or os.name != "nt":
            return None
        title = _foreground_window_title()
        if not title:
            return None
        local = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone)
        bucket = local.strftime("%Y-%m-%dT%H:%M")
        return _WorldSignal(
            observation_type="foreground_window",
            source="windows_user32",
            summary=f"Foreground application context changed to: {title}",
            payload={"title": title},
            salience=0.30,
            dedup_key=f"foreground:{_digest(title + '|' + bucket)}",
        )

    def _append_event(self, conversation_id: str, signal: _WorldSignal) -> Event:
        event_id = self._ids.new()
        return self._events.append(
            normalize_signal(
                IncomingSignal(
                    actor=Actor.WORLD,
                    signal_type=f"world.{signal.observation_type}",
                    content=signal.summary,
                    channel="world",
                    channel_message_id=signal.dedup_key,
                    conversation_id=conversation_id,
                    metadata={
                        "source": signal.source,
                        "salience": signal.salience,
                        **signal.payload,
                    },
                ),
                event_id=event_id,
                correlation_id=self._ids.new(),
                now_ms=self._clock.now_ms(),
                source_kind=SourceKind.WORLD_OBSERVED,
            )
        )


def _foreground_window_title() -> str:
    user32 = ctypes.windll.user32
    window = user32.GetForegroundWindow()
    if not window:
        return ""
    length = user32.GetWindowTextLengthW(window)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(window, buffer, length + 1)
    return str(buffer.value).strip()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["WorldObservationResult", "WorldObservationService"]
