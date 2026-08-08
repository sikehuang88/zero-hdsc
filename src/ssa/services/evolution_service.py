"""Offline SEOS candidate evolution service.

The service owns no online champion and never receives host-tool handles.  It
keeps the read/compute/write boundary at the API level: callers supply an
offline evaluator, and the service returns append-only candidate records for a
later persistence layer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from ssa.hdsc.archive import (
    ArchiveConfig,
    ArchiveEntry,
    BehaviorDescriptor,
    ParetoArchive,
    ScoreVector,
)
from ssa.hdsc.interpreter import ExecBudget, static_budget_gate
from ssa.hdsc.operators import PRIMITIVE_REGISTRY, OperatorRegistry
from ssa.hdsc.program import Program, ProgramTypeError, program_digest
from ssa.hdsc.variation import apply_random_variation

MODEL_ID: Final = "hdsc-seos-evolution-v0-shadow"
CLAIM_LEVEL: Final = "C0-offline-candidate-search"
Split = Literal["train", "validate"]


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    population_size: int = 64
    generations_per_cycle: int = 4
    tier1_survivors: int = 8
    max_nodes: int = 64
    max_static_cost: float = 1_000.0
    cycle_llm_call_cap: int = 1_024
    archive: ArchiveConfig | None = None
    shadow_only: bool = True

    def __post_init__(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if self.generations_per_cycle < 1:
            raise ValueError("generations_per_cycle must be positive")
        if not 1 <= self.tier1_survivors <= self.population_size:
            raise ValueError("tier1_survivors must be within population_size")
        if self.max_nodes < 1 or self.max_static_cost <= 0.0:
            raise ValueError("program limits must be positive")
        if self.cycle_llm_call_cap < 0:
            raise ValueError("cycle_llm_call_cap must be non-negative")
        if not self.shadow_only:
            raise ValueError("online promotion is not implemented in the offline service")


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    score: ScoreVector
    descriptor: BehaviorDescriptor
    llm_calls: int = 0
    tokens: int = 0
    cost_ms: int = 0
    evidence_trace_count: int = 0

    def __post_init__(self) -> None:
        if self.llm_calls < 0 or self.tokens < 0 or self.cost_ms < 0:
            raise ValueError("evaluation accounting must be non-negative")
        if self.evidence_trace_count < 0:
            raise ValueError("evidence_trace_count must be non-negative")


class OfflineEvaluator(Protocol):
    """Pure replay evaluator; implementations must not perform real actions."""

    def tier1_score(self, program: Program) -> float:
        """Return a zero-LLM proxy score for first-stage filtering."""

    def evaluate(self, program: Program, split: Split) -> EvaluationOutcome:
        """Evaluate a program on a time-bounded replay split."""


@dataclass(frozen=True, slots=True)
class LineageRecord:
    parent_hash: str
    child_hash: str
    variation: str
    seed: int
    generation: int


@dataclass(frozen=True, slots=True)
class EvolutionReport:
    model_id: str
    claim_level: str
    shadow_only: bool
    generations_completed: int
    generated_count: int
    unique_count: int
    tier1_admitted_count: int
    tier1_survivor_count: int
    evaluated_count: int
    llm_calls: int
    tokens: int
    archive_coverage: float
    archive_filled_cells: int
    lineage: tuple[LineageRecord, ...]
    halted: bool
    halt_reason: str | None


def _variation_seed(parent_hash: str, generation: int, index: int) -> int:
    payload = f"hdsc-seos-evolution-v1:{parent_hash}:{generation}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class EvolutionService:
    """Run a bounded, offline-only candidate evolution cycle."""

    def __init__(
        self,
        evaluator: OfflineEvaluator,
        *,
        config: EvolutionConfig | None = None,
        registry: OperatorRegistry = PRIMITIVE_REGISTRY,
        archive: ParetoArchive | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or EvolutionConfig()
        self.registry = registry
        self.archive = archive or ParetoArchive(self.config.archive or ArchiveConfig())
        self._programs: dict[str, Program] = {}

    def run_cycle(self, parents: Sequence[Program]) -> EvolutionReport:
        """Generate, tier-1 filter, evaluate, and archive offline candidates."""

        if not parents:
            raise ValueError("at least one parent program is required")
        pool: list[Program] = list(parents[: self.config.population_size])
        generated_count = 0
        unique_count = 0
        tier1_admitted_count = 0
        tier1_survivor_count = 0
        evaluated_count = 0
        llm_calls = 0
        tokens = 0
        lineage: list[LineageRecord] = []
        seen: set[str] = set()
        halted = False
        halt_reason: str | None = None
        generations_completed = 0

        for parent in pool:
            try:
                digest = program_digest(parent, self.registry)
                admission = static_budget_gate(
                    parent,
                    ExecBudget(
                        max_steps=self.config.max_nodes,
                        max_static_cost=self.config.max_static_cost,
                    ),
                    registry=self.registry,
                )
                if not admission.admitted:
                    raise ValueError(admission.reason or "parent rejected by static budget")
            except (ProgramTypeError, ValueError) as exc:
                raise ValueError(f"invalid parent program: {exc}") from exc
            self._programs[digest] = parent
            seen.add(digest)

        for generation in range(self.config.generations_per_cycle):
            candidates: list[Program] = list(pool)
            for index in range(self.config.population_size):
                parent = pool[index % len(pool)]
                parent_hash = program_digest(parent, self.registry)
                seed = _variation_seed(parent_hash, generation, index)
                child = apply_random_variation(parent, seed=seed, registry=self.registry)
                generated_count += 1
                child_hash = program_digest(child, self.registry)
                if child_hash not in seen:
                    seen.add(child_hash)
                    unique_count += 1
                    self._programs[child_hash] = child
                    lineage.append(
                        LineageRecord(parent_hash, child_hash, "mutate", seed, generation)
                    )
                    candidates.append(child)

            admitted: list[tuple[float, str, Program]] = []
            for candidate in candidates:
                try:
                    report = static_budget_gate(
                        candidate,
                        ExecBudget(
                            max_steps=self.config.max_nodes,
                            max_static_cost=self.config.max_static_cost,
                        ),
                        registry=self.registry,
                    )
                except (ProgramTypeError, ValueError):
                    continue
                if not report.admitted:
                    continue
                score = self.evaluator.tier1_score(candidate)
                if not 0.0 <= score <= 1.0:
                    raise ValueError("tier1_score must be within [0, 1]")
                admitted.append((score, program_digest(candidate, self.registry), candidate))
            tier1_admitted_count += len(admitted)
            admitted.sort(key=lambda item: (-item[0], item[1]))
            survivors = [item[2] for item in admitted[: self.config.tier1_survivors]]
            tier1_survivor_count += len(survivors)

            next_pool: list[Program] = []
            for candidate in survivors:
                for split in ("train", "validate"):
                    outcome = self.evaluator.evaluate(candidate, split)
                    llm_calls += outcome.llm_calls
                    tokens += outcome.tokens
                    evaluated_count += 1
                    if llm_calls > self.config.cycle_llm_call_cap:
                        halted = True
                        halt_reason = "cycle LLM-call budget exceeded"
                        break
                    digest = program_digest(candidate, self.registry)
                    entry = ArchiveEntry(
                        program_id=digest,
                        genotype_hash=digest,
                        score=outcome.score,
                        descriptor=outcome.descriptor,
                        generation=generation,
                    )
                    if split == "validate" and self.archive.update(entry):
                        next_pool.append(candidate)
                if halted:
                    break
            if halted:
                break
            generations_completed += 1
            pool = next_pool[: self.config.population_size] or survivors

        return EvolutionReport(
            model_id=MODEL_ID,
            claim_level=CLAIM_LEVEL,
            shadow_only=self.config.shadow_only,
            generations_completed=generations_completed,
            generated_count=generated_count,
            unique_count=unique_count,
            tier1_admitted_count=tier1_admitted_count,
            tier1_survivor_count=tier1_survivor_count,
            evaluated_count=evaluated_count,
            llm_calls=llm_calls,
            tokens=tokens,
            archive_coverage=self.archive.coverage,
            archive_filled_cells=self.archive.filled_cells,
            lineage=tuple(lineage),
            halted=halted,
            halt_reason=halt_reason,
        )
