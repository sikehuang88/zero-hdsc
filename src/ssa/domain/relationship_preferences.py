"""User-controlled relationship communication preferences."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

_CLOCK_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class RelationshipSupportMode(StrEnum):
    ADAPTIVE = "adaptive"
    LISTEN = "listen"
    COMFORT = "comfort"
    SOLVE = "solve"
    CHALLENGE = "challenge"


class ProactiveFrequency(StrEnum):
    OFF = "off"
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"


class RelationshipPreferences(BaseModel):
    venom_intensity: int = Field(default=58, ge=0, le=100)
    support_mode: RelationshipSupportMode = RelationshipSupportMode.ADAPTIVE
    proactive_frequency: ProactiveFrequency = ProactiveFrequency.BALANCED
    quiet_hours_enabled: bool = True
    quiet_hours_start: str = "23:00"
    quiet_hours_end: str = "08:00"
    updated_at_ms: int = Field(default=0, ge=0)

    @field_validator("quiet_hours_start", "quiet_hours_end")
    @classmethod
    def _valid_clock(cls, value: str) -> str:
        if not _CLOCK_RE.fullmatch(value):
            raise ValueError("quiet hours must use HH:MM")
        return value

    def prompt_context(self) -> str:
        intensity = self.venom_intensity
        if intensity <= 20:
            voice = "mostly gentle and plainspoken; use sharp teasing only when clearly invited"
        elif intensity <= 45:
            voice = "direct with occasional dry teasing; never make sharpness the point of the reply"
        elif intensity <= 70:
            voice = "confident and pointed with a recognizable venomous-older-sister edge"
        else:
            voice = "strongly blunt and cutting toward excuses, while staying specific and useful"

        support = {
            RelationshipSupportMode.ADAPTIVE: (
                "infer whether the owner needs listening, comfort, practical help, challenge, or quiet; "
                "when evidence is weak, ask one short calibration question"
            ),
            RelationshipSupportMode.LISTEN: (
                "when the owner is distressed, listen and reflect first; keep advice brief unless requested"
            ),
            RelationshipSupportMode.COMFORT: (
                "when the owner is distressed, acknowledge the feeling and offer steady companionship before solutions"
            ),
            RelationshipSupportMode.SOLVE: (
                "when the owner is distressed, acknowledge it briefly and move into a concrete next step"
            ),
            RelationshipSupportMode.CHALLENGE: (
                "when the owner is distressed, identify avoidance or self-deception directly, then give one executable step"
            ),
        }[self.support_mode]

        return (
            "Active relationship preferences (user-controlled and immediately effective):\n"
            f"- venom_intensity={intensity}/100: {voice}.\n"
            f"- support_mode={self.support_mode.value}: {support}.\n"
            "- Context overrides baseline sharpness: in support or repair situations, lower the edge, "
            "do not ridicule vulnerability, and prioritize being understood; in ordinary playful or "
            "challenge situations, the configured edge may be more visible.\n"
            "- Preserve a personal stance and concrete care. Do not become a therapist, customer-service "
            "agent, obedient mirror, or generic praise generator."
        )


__all__ = [
    "ProactiveFrequency",
    "RelationshipPreferences",
    "RelationshipSupportMode",
]
