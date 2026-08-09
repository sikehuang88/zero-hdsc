"""The first real ``OfflineEvaluator``: fitness from archived ground truth.

Until now the evolution loop had scaffolding but no fitness — ``OfflineEvaluator``
was a Protocol whose only implementation was a test stub returning constants, so
selection had nothing behavioural to act on.  This module supplies the missing
half by scoring a candidate on how well it retrodicts the archive.

Nothing here calls a language model.  That is what makes a first generation
tractable: the cost analysis that made online candidate evaluation infeasible
does not apply, evaluation is bit-reproducible despite the serving model having
no seed parameter, and thousands of generations run in seconds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from ssa.hdsc.archive import BehaviorDescriptor, ScoreVector
from ssa.hdsc.implementations import (
    DEFAULT_DIMENSION,
    ArchivedEvent,
    RetrodictionImplementations,
)
from ssa.hdsc.interpreter import (
    ExecBudget,
    ExecContext,
    InterpreterError,
    execute,
    static_budget_gate,
)
from ssa.hdsc.operators import PRIMITIVE_REGISTRY, OperatorRegistry
from ssa.hdsc.program import Program
from ssa.hdsc.retrodiction import (
    RetrodictionEpisode,
    assert_no_future,
    evidence_before,
)
from ssa.services.evolution_service import EvaluationOutcome, Split

MODEL_ID: Final = "hdsc-seos-retrodiction-evaluator-v0"
CLAIM_LEVEL: Final = "C0-A-archive-backtest"

_FAILED_SCORE: Final = ScoreVector(0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RetrodictionScore:
    """Aggregate outcome of scoring one program over a set of episodes."""

    episodes: int
    brier: float
    baseline_brier: float
    skill: float
    mean_probability: float
    failures: int

    @property
    def cognitive(self) -> float:
        """Map skill onto the unit interval expected by ``ScoreVector``.

        Skill is 0 when the program matches the base rate, 1 when it is
        perfect, and unboundedly negative when it is worse than the base rate.
        A logistic is used rather than a clamp because clamping collapses every
        below-baseline candidate onto 0, and an early generation is almost
        entirely below baseline — selection would then have no gradient at all.
        The map is strictly monotone, so it preserves the Pareto ordering
        exactly, and 0.5 marks base-rate-equivalent skill.
        """

        return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.skill)))))


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    dimension: int = DEFAULT_DIMENSION
    max_evidence_events: int = 128
    tier1_episodes: int = 8
    max_node_count: int = 64
    budget: ExecBudget = field(default_factory=lambda: ExecBudget(max_llm_calls=0, max_tokens=0))

    def __post_init__(self) -> None:
        if self.tier1_episodes < 1:
            raise ValueError("tier1_episodes must be positive")
        if self.max_node_count < 1:
            raise ValueError("max_node_count must be positive")
        if self.budget.max_llm_calls or self.budget.max_tokens:
            raise ValueError("retrodiction evaluation must not permit model calls")


class RetrodictionEvaluator:
    """Score candidate programs by backtesting them against the archive."""

    def __init__(
        self,
        events: Sequence[ArchivedEvent],
        train_episodes: Sequence[RetrodictionEpisode],
        validate_episodes: Sequence[RetrodictionEpisode],
        config: EvaluatorConfig | None = None,
        *,
        registry: OperatorRegistry = PRIMITIVE_REGISTRY,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self.registry = registry
        self._events = tuple(events)
        self._episodes: dict[Split, tuple[RetrodictionEpisode, ...]] = {
            "train": tuple(train_episodes),
            "validate": tuple(validate_episodes),
        }
        self._implementations = RetrodictionImplementations(self.config.dimension)
        self.llm_calls = 0

    # -- scoring --------------------------------------------------------

    def score_episodes(
        self,
        program: Program,
        episodes: Sequence[RetrodictionEpisode],
    ) -> RetrodictionScore:
        total_brier = 0.0
        total_baseline = 0.0
        total_probability = 0.0
        failures = 0
        scored = 0
        context = ExecContext(implementations=self._implementations.as_mapping())

        for episode in episodes:
            evidence = evidence_before(
                self._events,
                episode.cut_at_ms,
                max_events=self.config.max_evidence_events,
            )
            # Defence in depth: the guard is cheap and turns any future
            # regression in windowing into a hard failure rather than a
            # quietly inflated score.
            assert_no_future(evidence, episode.cut_at_ms)
            inputs = {
                "evidence": evidence,
                "query": episode.query_vector(self.config.dimension),
            }
            try:
                result = execute(
                    program,
                    inputs,
                    self.config.budget,
                    context,
                    registry=self.registry,
                )
            except InterpreterError:
                failures += 1
                continue
            probability = result.value
            if not isinstance(probability, float) or not 0.0 <= probability <= 1.0:
                failures += 1
                continue
            total_brier += episode.brier(probability)
            total_baseline += episode.baseline_brier()
            total_probability += probability
            scored += 1

        if not scored:
            return RetrodictionScore(0, 1.0, 1.0, 0.0, 0.0, failures)
        brier = total_brier / scored
        baseline = total_baseline / scored
        skill = 1.0 - brier / baseline if baseline > 0.0 else 0.0
        return RetrodictionScore(
            episodes=scored,
            brier=brier,
            baseline_brier=baseline,
            skill=skill,
            mean_probability=total_probability / scored,
            failures=failures,
        )

    # -- OfflineEvaluator protocol --------------------------------------

    def tier1_score(self, program: Program) -> float:
        """Zero-model structural screen run before any episode is scored."""

        admission = static_budget_gate(program, self.config.budget, registry=self.registry)
        if not admission.admitted:
            return 0.0
        if program.eff_flux:
            return 0.0
        sample = self._episodes["train"][: self.config.tier1_episodes]
        if not sample:
            return 0.0
        outcome = self.score_episodes(program, sample)
        if not outcome.episodes:
            return 0.0
        parsimony = 1.0 - min(1.0, program.node_count / float(self.config.max_node_count))
        return 0.8 * outcome.cognitive + 0.2 * parsimony

    def evaluate(self, program: Program, split: Split) -> EvaluationOutcome:
        episodes = self._episodes.get(split, ())
        outcome = self.score_episodes(program, episodes)
        admission = static_budget_gate(program, self.config.budget, registry=self.registry)

        normalized_nodes = min(1.0, program.node_count / float(self.config.max_node_count))
        efficiency = 1.0 - min(1.0, admission.static_cost / self.config.budget.max_static_cost)
        reliability = (
            1.0 - outcome.failures / float(outcome.failures + outcome.episodes)
            if outcome.failures + outcome.episodes
            else 0.0
        )
        # Calibration stands in for the emotional field at this stage: a
        # candidate whose mean probability tracks the realised outcome rate is
        # calibrated, which is the only affect-adjacent property the archive
        # can ground without a person in the loop.
        realised = (
            sum(1.0 for episode in episodes if episode.outcome) / len(episodes) if episodes else 0.0
        )
        calibration = 1.0 - min(1.0, abs(outcome.mean_probability - realised))

        score = (
            ScoreVector(
                cognitive=outcome.cognitive,
                emotional=calibration,
                systemic=0.5 * efficiency + 0.5 * reliability,
            )
            if outcome.episodes
            else _FAILED_SCORE
        )
        descriptor = BehaviorDescriptor(
            slow_path_rate=min(1.0, outcome.mean_probability),
            rewire_rate=min(1.0, outcome.brier),
            average_candidates=min(1.0, outcome.episodes / max(1.0, float(len(episodes)))),
            normalized_node_count=normalized_nodes,
        )
        return EvaluationOutcome(
            score=score,
            descriptor=descriptor,
            llm_calls=0,
            tokens=0,
            evidence_trace_count=outcome.episodes,
        )


__all__ = [
    "CLAIM_LEVEL",
    "MODEL_ID",
    "EvaluatorConfig",
    "RetrodictionEvaluator",
    "RetrodictionScore",
]
