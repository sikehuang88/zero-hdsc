"""Retrodiction self-play over the immutable event archive.

The evolution loop needs a fitness signal that is objective, fast, and
available in unlimited quantity.  Forward predictions are objective but resolve
on the user's timescale, which is far too slow to select programs against.

Retrodiction closes that gap without weakening the ground truth.  Pick a cut
point in the archived past, show a candidate program only what existed before
it, and ask for the probability that a given topic recurs within a horizon.
The archive already holds the answer, so scoring is a substring test over
immutable records: no user in the loop, no model judging itself, and one
episode per (cut point, topic) pair.

The scoring rule is Brier, which is strictly proper: expected loss is uniquely
minimised by stating the true probability, so a program cannot gain by being
systematically overconfident.  Skill is reported against the topic base rate so
that predicting the base rate scores zero and trivial claims earn nothing.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from ssa.hdsc.hypervector import PackedHypervector, semantic_random
from ssa.hdsc.implementations import ArchivedEvent, order_events

MODEL_ID: Final = "hdsc-seos-retrodiction-v0"
CLAIM_LEVEL: Final = "C0-A-archive-backtest"
GROUND_TRUTH: Final = "immutable-archive-substring"

_ASCII_TOKEN = re.compile(r"[a-z0-9]{3,}")
_CJK = re.compile(r"[一-鿿]")


class FutureLeakError(RuntimeError):
    """Raised when evaluation evidence reaches at or past its cut point."""


@dataclass(frozen=True, slots=True)
class RetrodictionEpisode:
    """One self-play episode derived entirely from archived events."""

    episode_id: str
    topic: str
    cut_at_ms: int
    horizon_ms: int
    outcome: bool
    base_rate: float

    def __post_init__(self) -> None:
        if self.horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        if not 0.0 <= self.base_rate <= 1.0:
            raise ValueError("base_rate must be within [0, 1]")
        if not self.topic:
            raise ValueError("topic must not be empty")

    @property
    def query_digest(self) -> str:
        return hashlib.sha256(f"topic:{self.topic}".encode()).hexdigest()

    def query_vector(self, dimension: int) -> PackedHypervector:
        """Return the content-derived hypervector handed to the program."""

        return semantic_random(
            dimension,
            semantic_role="retrodiction-topic",
            input_digest=self.query_digest,
        )

    def brier(self, probability: float) -> float:
        outcome = 1.0 if self.outcome else 0.0
        return float((probability - outcome) ** 2)

    def baseline_brier(self) -> float:
        outcome = 1.0 if self.outcome else 0.0
        return float((self.base_rate - outcome) ** 2)


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Bounds for episode generation."""

    horizon_ms: int = 3_600_000
    max_episodes: int = 128
    min_evidence: int = 4
    min_base_rate: float = 0.05
    max_base_rate: float = 0.70
    topics_per_cut: int = 3

    def __post_init__(self) -> None:
        if self.horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        if self.max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        if not 0.0 <= self.min_base_rate < self.max_base_rate <= 1.0:
            raise ValueError("base-rate band must be ordered within [0, 1]")


def extract_topics(content: str) -> tuple[str, ...]:
    """Extract stable topic tokens: CJK bigrams plus ASCII words."""

    lowered = content.casefold()
    tokens: list[str] = list(_ASCII_TOKEN.findall(lowered))
    cjk = _CJK.findall(content)
    tokens.extend(a + b for a, b in pairwise(cjk))
    seen: dict[str, None] = {}
    for token in tokens:
        seen.setdefault(token, None)
    return tuple(seen)


def _topic_occurs(
    events: Sequence[ArchivedEvent],
    topic: str,
    start_ms: int,
    end_ms: int,
    actor: str,
) -> bool:
    return any(
        event.actor == actor and start_ms <= event.created_at_ms < end_ms and topic in event.content
        for event in events
    )


def build_episodes(
    events: Sequence[ArchivedEvent],
    config: EpisodeConfig | None = None,
    *,
    actor: str = "user",
) -> tuple[RetrodictionEpisode, ...]:
    """Derive deterministic self-play episodes from an archive slice."""

    settings = config or EpisodeConfig()
    ordered = order_events(events)
    if len(ordered) <= settings.min_evidence:
        return ()

    horizon = settings.horizon_ms
    cut_candidates = ordered[settings.min_evidence :]

    # Base rate per topic: the share of cut points at which the topic recurs.
    topic_hits: dict[str, int] = {}
    topic_total: dict[str, int] = {}
    per_cut_topics: list[tuple[int, tuple[str, ...]]] = []
    for event in cut_candidates:
        cut = event.created_at_ms
        past_topics: dict[str, None] = {}
        for earlier in ordered:
            if earlier.created_at_ms >= cut:
                break
            for topic in extract_topics(earlier.content):
                past_topics.setdefault(topic, None)
        candidates = tuple(past_topics)[: max(1, settings.topics_per_cut * 4)]
        per_cut_topics.append((cut, candidates))
        for topic in candidates:
            topic_total[topic] = topic_total.get(topic, 0) + 1
            if _topic_occurs(ordered, topic, cut, cut + horizon, actor):
                topic_hits[topic] = topic_hits.get(topic, 0) + 1

    base_rates = {
        topic: topic_hits.get(topic, 0) / total for topic, total in topic_total.items() if total
    }
    usable = {
        topic: rate
        for topic, rate in base_rates.items()
        if settings.min_base_rate <= rate <= settings.max_base_rate
    }
    if not usable:
        return ()

    episodes: list[RetrodictionEpisode] = []
    for cut, candidates in per_cut_topics:
        selected = [topic for topic in candidates if topic in usable]
        selected.sort(key=lambda item: (abs(usable[item] - 0.5), item))
        for topic in selected[: settings.topics_per_cut]:
            outcome = _topic_occurs(ordered, topic, cut, cut + horizon, actor)
            episode_id = hashlib.sha256(f"episode:{cut}:{horizon}:{topic}".encode()).hexdigest()[
                :32
            ]
            episodes.append(
                RetrodictionEpisode(
                    episode_id=episode_id,
                    topic=topic,
                    cut_at_ms=cut,
                    horizon_ms=horizon,
                    outcome=outcome,
                    base_rate=usable[topic],
                )
            )
            if len(episodes) >= settings.max_episodes:
                return tuple(episodes)
    return tuple(episodes)


def assert_no_future(events: Sequence[ArchivedEvent], cut_at_ms: int) -> None:
    """Raise ``FutureLeakError`` if any event reaches its evaluation cut point.

    Time-ordered splitting is only a convention until something enforces it.
    This turns leakage into a deterministic failure instead of a silent bias
    that would otherwise surface as offline scores that never transfer online.
    """

    for event in events:
        if event.created_at_ms >= cut_at_ms:
            raise FutureLeakError(
                f"event {event.event_id!r} at {event.created_at_ms} "
                f"reaches evaluation cut {cut_at_ms}"
            )


def evidence_before(
    events: Sequence[ArchivedEvent],
    cut_at_ms: int,
    *,
    max_events: int = 256,
) -> tuple[ArchivedEvent, ...]:
    """Return the deterministic evidence window strictly before the cut."""

    window = [event for event in order_events(events) if event.created_at_ms < cut_at_ms]
    trimmed = tuple(window[-max_events:])
    assert_no_future(trimmed, cut_at_ms)
    return trimmed


def time_ordered_split(
    episodes: Sequence[RetrodictionEpisode],
    *,
    train_fraction: float = 0.6,
) -> tuple[tuple[RetrodictionEpisode, ...], tuple[RetrodictionEpisode, ...]]:
    """Split episodes by cut time so validation never precedes training."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be within (0, 1)")
    ordered = sorted(episodes, key=lambda item: (item.cut_at_ms, item.episode_id))
    boundary = int(len(ordered) * train_fraction)
    return tuple(ordered[:boundary]), tuple(ordered[boundary:])


__all__ = [
    "CLAIM_LEVEL",
    "GROUND_TRUTH",
    "MODEL_ID",
    "EpisodeConfig",
    "FutureLeakError",
    "RetrodictionEpisode",
    "assert_no_future",
    "build_episodes",
    "evidence_before",
    "extract_topics",
    "time_ordered_split",
]
