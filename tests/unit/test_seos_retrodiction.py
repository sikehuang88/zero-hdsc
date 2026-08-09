"""Retrodiction self-play: episode construction, leakage guards, and scoring."""

from __future__ import annotations

import random

import pytest

from ssa.hdsc.implementations import ArchivedEvent, RetrodictionImplementations, order_events
from ssa.hdsc.operators import EV, HV, PRIMITIVE_REGISTRY, SC, Coll
from ssa.hdsc.program import Node, Program, ProgramInput, check_linearity, type_check
from ssa.hdsc.retrodiction import (
    EpisodeConfig,
    FutureLeakError,
    RetrodictionEpisode,
    assert_no_future,
    build_episodes,
    evidence_before,
    extract_topics,
    time_ordered_split,
)
from ssa.services.retrodiction_evaluator import EvaluatorConfig, RetrodictionEvaluator

BASE_MS = 1_700_000_000_000


def synthetic_archive(count: int = 160, seed: int = 11) -> list[ArchivedEvent]:
    rng = random.Random(seed)
    topics = ["跑步", "读书", "工作", "睡眠", "做饭", "旅行"]
    events: list[ArchivedEvent] = []
    state = 0
    for index in range(count):
        if rng.random() < 0.35:
            state = rng.randrange(len(topics))
        actor = "user" if rng.random() < 0.55 else "agent"
        events.append(
            ArchivedEvent(
                f"e{index}",
                BASE_MS + index * 600_000,
                actor,
                f"关于{topics[state]}的对话内容{rng.randrange(100)}",
            )
        )
    return events


def seed_program() -> Program:
    return Program(
        inputs=(ProgramInput("evidence", Coll("set", EV)), ProgramInput("query", HV)),
        nodes=(
            Node("a", "activate", ("query", "evidence")),
            Node("m", "allocate_mass", ("a",)),
            Node("h", "quantize", ("m",)),
            Node("s", "similarity", ("h", "query")),
            Node("c", "calibrate", ("s",), (("bias", 0.0), ("scale", 4.0))),
        ),
        outputs=("c",),
    )


def test_seed_program_is_well_typed_and_effect_free() -> None:
    program = seed_program()
    type_check(program)
    check_linearity(program)
    assert program.eff_flux == 0


def test_registry_has_a_scalar_sink() -> None:
    """Without an operator producing Sc from non-Sc inputs no program can score."""

    sinks = [
        operator.operator_id
        for operator in PRIMITIVE_REGISTRY.values()
        if operator.signature.result == SC
        and not any(param == SC for param in operator.signature.params)
    ]
    assert sinks, "the type graph must have a path from evidence to a score"


def test_every_registry_operator_has_an_implementation() -> None:
    implementations = RetrodictionImplementations().as_mapping()
    assert set(PRIMITIVE_REGISTRY.ids()) == set(implementations)


def test_event_identity_and_content_digests_are_separate() -> None:
    first = ArchivedEvent("e1", BASE_MS, "user", "same content")
    second = ArchivedEvent("e2", BASE_MS + 600_000, "user", "same content")
    implementations = RetrodictionImplementations(dimension=64)

    assert first.digest != second.digest
    assert first.content_digest == second.content_digest
    assert implementations.event_vector(first) == implementations.event_vector(second)


def test_extract_topics_covers_cjk_bigrams_and_ascii_words() -> None:
    topics = extract_topics("今天 running 很好")
    assert "running" in topics
    assert "今天" in topics


def test_episodes_carry_objective_outcomes_and_a_usable_base_rate() -> None:
    episodes = build_episodes(
        synthetic_archive(), EpisodeConfig(horizon_ms=1_800_000, max_episodes=200)
    )
    assert episodes
    assert all(0.0 <= episode.base_rate <= 1.0 for episode in episodes)
    outcomes = {episode.outcome for episode in episodes}
    assert outcomes == {True, False}, "a usable game needs both outcomes present"


def test_evidence_before_excludes_the_cut_point() -> None:
    events = synthetic_archive(40)
    cut = events[20].created_at_ms
    evidence = evidence_before(events, cut)
    assert evidence
    assert all(event.created_at_ms < cut for event in evidence)


def test_assert_no_future_rejects_leakage() -> None:
    events = order_events(synthetic_archive(20))
    cut = events[5].created_at_ms
    with pytest.raises(FutureLeakError):
        assert_no_future(events, cut)


def test_time_ordered_split_never_lets_validation_precede_training() -> None:
    episodes = build_episodes(
        synthetic_archive(), EpisodeConfig(horizon_ms=1_800_000, max_episodes=200)
    )
    train, validate = time_ordered_split(episodes)
    assert train and validate
    assert max(item.cut_at_ms for item in train) <= min(item.cut_at_ms for item in validate)


def test_brier_is_minimised_by_the_truthful_probability() -> None:
    """The scoring rule must be strictly proper, or overconfidence pays."""

    episode = RetrodictionEpisode("id", "topic", BASE_MS, 1_000, outcome=True, base_rate=0.4)
    assert episode.brier(1.0) < episode.brier(0.9) < episode.brier(0.5)
    miss = RetrodictionEpisode("id", "topic", BASE_MS, 1_000, outcome=False, base_rate=0.4)
    assert miss.brier(0.0) < miss.brier(0.1) < miss.brier(0.5)


def test_evaluation_is_deterministic_and_free_of_model_calls() -> None:
    events = synthetic_archive()
    episodes = build_episodes(events, EpisodeConfig(horizon_ms=1_800_000, max_episodes=200))
    train, validate = time_ordered_split(episodes)
    evaluator = RetrodictionEvaluator(events, train, validate, EvaluatorConfig(dimension=128))
    program = seed_program()

    first = evaluator.score_episodes(program, validate)
    second = evaluator.score_episodes(program, validate)
    assert first == second

    outcome = evaluator.evaluate(program, "validate")
    assert outcome.llm_calls == 0
    assert outcome.tokens == 0
    assert outcome.evidence_trace_count > 0


def test_evaluator_config_forbids_model_calls() -> None:
    from ssa.hdsc.interpreter import ExecBudget

    with pytest.raises(ValueError, match="must not permit model calls"):
        EvaluatorConfig(budget=ExecBudget(max_llm_calls=1, max_tokens=0))


def test_scores_stay_within_the_unit_interval_for_a_failing_program() -> None:
    """A program that cannot execute must score, not crash the loop."""

    events = synthetic_archive(60)
    episodes = build_episodes(events, EpisodeConfig(horizon_ms=1_800_000, max_episodes=60))
    train, validate = time_ordered_split(episodes)
    evaluator = RetrodictionEvaluator(events, train, validate, EvaluatorConfig(dimension=64))
    broken = Program(
        inputs=(ProgramInput("evidence", Coll("set", EV)), ProgramInput("query", HV)),
        nodes=(Node("s", "similarity", ("query", "query")),),
        outputs=("s",),
    )
    outcome = evaluator.evaluate(broken, "validate")
    for value in (outcome.score.cognitive, outcome.score.emotional, outcome.score.systemic):
        assert 0.0 <= value <= 1.0
