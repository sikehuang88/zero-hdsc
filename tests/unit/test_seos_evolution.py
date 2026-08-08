"""Tests for the offline SEOS evolution service."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ssa.hdsc.archive import ArchiveConfig, BehaviorDescriptor, ScoreVector
from ssa.hdsc.operators import HV
from ssa.hdsc.program import Node, Program, ProgramInput
from ssa.services.evolution_service import (
    EvaluationOutcome,
    EvolutionConfig,
    EvolutionService,
)


def _parent() -> Program:
    return Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "bind", ("left", "right")),),
        outputs=("root",),
    )


@dataclass
class FakeEvaluator:
    evaluations: list[str]
    tier1_calls: int = 0

    def tier1_score(self, program: Program) -> float:
        self.tier1_calls += 1
        return 1.0 / program.node_count

    def evaluate(self, program: Program, split: str) -> EvaluationOutcome:
        self.evaluations.append(split)
        score = 0.8 if split == "validate" else 0.7
        return EvaluationOutcome(
            score=ScoreVector(score, 0.6, 0.9),
            descriptor=BehaviorDescriptor(0.1, 0.2, 0.3, 0.1),
            llm_calls=1,
            tokens=4,
            evidence_trace_count=2,
        )


def test_offline_cycle_filters_and_archives_candidates() -> None:
    evaluator = FakeEvaluator([])
    service = EvolutionService(
        evaluator,
        config=EvolutionConfig(
            population_size=4,
            generations_per_cycle=1,
            tier1_survivors=2,
            cycle_llm_call_cap=10,
            archive=ArchiveConfig(bins=(2, 2, 2, 2), cell_capacity=2),
        ),
    )
    report = service.run_cycle((_parent(),))
    assert report.halted is False
    assert report.generations_completed == 1
    assert report.tier1_survivor_count == 2
    assert report.evaluated_count == 4
    assert report.llm_calls == 4
    assert report.archive_filled_cells == 1
    assert evaluator.tier1_calls > 0
    assert evaluator.evaluations == ["train", "validate", "train", "validate"]


def test_cycle_stops_at_llm_budget() -> None:
    evaluator = FakeEvaluator([])
    service = EvolutionService(
        evaluator,
        config=EvolutionConfig(
            population_size=2,
            generations_per_cycle=2,
            tier1_survivors=1,
            cycle_llm_call_cap=0,
        ),
    )
    report = service.run_cycle((_parent(),))
    assert report.halted is True
    assert report.halt_reason == "cycle LLM-call budget exceeded"
    assert report.generations_completed == 0


def test_config_rejects_online_promotion() -> None:
    with pytest.raises(ValueError, match="online promotion"):
        EvolutionConfig(shadow_only=False)
