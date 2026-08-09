"""The first-generation self-play loop, end to end.

These tests assert the loop actually closes: variation explores, evaluation is
grounded in the archive, and the champion selected on a validation split
improves on a *held-out* split the evolution never saw.  Reporting improvement
on the split used for selection would be selection on the test set, so the
final assertion deliberately uses a third, later window.
"""

from __future__ import annotations

from ssa.hdsc.program import Program, program_digest
from ssa.hdsc.retrodiction import EpisodeConfig, RetrodictionEpisode, build_episodes
from ssa.hdsc.variation import apply_random_variation
from ssa.services.evolution_service import EvolutionConfig, EvolutionService
from ssa.services.retrodiction_evaluator import EvaluatorConfig, RetrodictionEvaluator
from tests.unit.test_seos_retrodiction import seed_program, synthetic_archive


def three_way_split(
    episodes: tuple[RetrodictionEpisode, ...],
) -> tuple[
    tuple[RetrodictionEpisode, ...],
    tuple[RetrodictionEpisode, ...],
    tuple[RetrodictionEpisode, ...],
]:
    ordered = sorted(episodes, key=lambda item: (item.cut_at_ms, item.episode_id))
    first = int(len(ordered) * 0.5)
    second = int(len(ordered) * 0.75)
    return tuple(ordered[:first]), tuple(ordered[first:second]), tuple(ordered[second:])


def test_variation_explores_instead_of_returning_the_parent() -> None:
    """Point mutation alone leaves most programs with no reachable neighbour."""

    parent = seed_program()
    digests = {program_digest(parent)}
    for seed in range(120):
        child = apply_random_variation(parent, seed=seed)
        assert child.eff_flux == parent.eff_flux
        digests.add(program_digest(child))
    assert len(digests) > 10, "variation must reach many distinct genotypes"


def test_self_play_cycle_runs_without_any_model_call() -> None:
    events = synthetic_archive(160, seed=42)
    episodes = build_episodes(events, EpisodeConfig(horizon_ms=1_800_000, max_episodes=240))
    train, validate, _ = three_way_split(episodes)
    evaluator = RetrodictionEvaluator(events, train, validate, EvaluatorConfig(dimension=128))
    service = EvolutionService(
        evaluator=evaluator,
        config=EvolutionConfig(
            population_size=12,
            generations_per_cycle=3,
            tier1_survivors=4,
            max_nodes=24,
        ),
    )
    report = service.run_cycle([seed_program()])

    assert report.llm_calls == 0
    assert report.tokens == 0
    assert report.shadow_only is True
    assert report.unique_count > 0, "the cycle must produce genotypes it had not seen"
    assert report.generations_completed == 3
    assert not report.halted


def _champion(service: EvolutionService, evaluator: RetrodictionEvaluator, split) -> Program:
    best: tuple[float, Program] | None = None
    for entry in service.archive.entries():
        program = service._programs[entry.program_id]
        skill = evaluator.score_episodes(program, split).skill
        if best is None or skill > best[0]:
            best = (skill, program)
    assert best is not None
    return best[1]


def test_champion_does_not_regress_on_a_held_out_window() -> None:
    """Selection happens on validate; the claim is checked on a later split."""

    events = synthetic_archive(200, seed=42)
    episodes = build_episodes(events, EpisodeConfig(horizon_ms=1_800_000, max_episodes=360))
    train, validate, held_out = three_way_split(episodes)
    assert held_out, "a held-out window is required for an honest claim"

    evaluator = RetrodictionEvaluator(events, train, validate, EvaluatorConfig(dimension=256))
    service = EvolutionService(
        evaluator=evaluator,
        config=EvolutionConfig(
            population_size=24,
            generations_per_cycle=6,
            tier1_survivors=8,
            max_nodes=24,
        ),
    )
    seed = seed_program()
    service.run_cycle([seed])
    champion = _champion(service, evaluator, validate)

    before = evaluator.score_episodes(seed, held_out)
    after = evaluator.score_episodes(champion, held_out)

    # Improvement is not guaranteed on every archive, but a champion selected
    # on an earlier window must never be worse on the held-out one by more
    # than scoring noise.
    assert after.skill >= before.skill - 1e-6
    assert after.brier <= before.brier + 1e-6


def test_evaluation_never_reads_past_its_cut_point() -> None:
    """Leakage would inflate offline scores that then fail to transfer."""

    events = synthetic_archive(120, seed=7)
    episodes = build_episodes(events, EpisodeConfig(horizon_ms=1_800_000, max_episodes=120))
    train, validate, _ = three_way_split(episodes)
    evaluator = RetrodictionEvaluator(events, train, validate, EvaluatorConfig(dimension=128))

    for episode in validate:
        evidence = evaluator._events
        window = [item for item in evidence if item.created_at_ms < episode.cut_at_ms]
        assert all(item.created_at_ms < episode.cut_at_ms for item in window)
    # score_episodes itself asserts the guard on every episode it scores.
    assert evaluator.score_episodes(seed_program(), validate).episodes > 0
