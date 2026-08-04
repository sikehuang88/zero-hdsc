"""User-controlled limits for proactive relationship contact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from ssa.config import InitiativeConfig
from ssa.domain.relationship_preferences import ProactiveFrequency
from ssa.storage.relationship_preferences_repository import RelationshipPreferencesRepository


@dataclass(frozen=True)
class ProactiveContactLimits:
    frequency: ProactiveFrequency
    daily_limit: int
    cooldown_minutes: int
    min_gap_hours: int
    quiet_hours_enabled: bool
    quiet_hours_start: str
    quiet_hours_end: str

    @property
    def enabled(self) -> bool:
        return self.frequency != ProactiveFrequency.OFF


class ProactiveContactPolicy:
    """Resolve the current preference profile on every background decision."""

    def __init__(
        self,
        *,
        preferences: RelationshipPreferencesRepository,
        config: InitiativeConfig,
        timezone: str,
    ) -> None:
        self._preferences = preferences
        self._config = config
        self._timezone = ZoneInfo(timezone)

    def current(self) -> ProactiveContactLimits:
        preferences = self._preferences.get()
        frequency = preferences.proactive_frequency
        if frequency == ProactiveFrequency.LOW:
            daily_limit, cooldown_minutes, min_gap_hours = 1, 12 * 60, 8
        elif frequency == ProactiveFrequency.HIGH:
            daily_limit, cooldown_minutes, min_gap_hours = 5, 2 * 60, 2
        else:
            daily_limit = self._config.daily_limit
            cooldown_minutes = self._config.cooldown_minutes
            min_gap_hours = self._config.min_gap_hours
        return ProactiveContactLimits(
            frequency=frequency,
            daily_limit=daily_limit,
            cooldown_minutes=cooldown_minutes,
            min_gap_hours=min_gap_hours,
            quiet_hours_enabled=preferences.quiet_hours_enabled,
            quiet_hours_start=preferences.quiet_hours_start,
            quiet_hours_end=preferences.quiet_hours_end,
        )

    def gate(self, now_ms: int, limits: ProactiveContactLimits | None = None) -> str | None:
        active = limits or self.current()
        if not active.enabled:
            return "proactive_disabled"
        if active.quiet_hours_enabled and self._is_quiet(now_ms, active):
            return "quiet_hours"
        return None

    def _is_quiet(self, now_ms: int, limits: ProactiveContactLimits) -> bool:
        local_time = datetime.fromtimestamp(now_ms / 1000, UTC).astimezone(self._timezone).time()
        start = time.fromisoformat(limits.quiet_hours_start)
        end = time.fromisoformat(limits.quiet_hours_end)
        if start == end:
            return False
        if start < end:
            return start <= local_time < end
        return local_time >= start or local_time < end


__all__ = ["ProactiveContactLimits", "ProactiveContactPolicy"]
