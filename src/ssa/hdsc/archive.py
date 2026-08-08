"""Offline Pareto and quality-diversity archive for SEOS candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _unit(value: float, name: str) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class ScoreVector:
    """The three EVF-MACK objectives, kept separate for Pareto selection."""

    cognitive: float
    emotional: float
    systemic: float

    def __post_init__(self) -> None:
        _unit(self.cognitive, "cognitive")
        _unit(self.emotional, "emotional")
        _unit(self.systemic, "systemic")

    def dominates(self, other: ScoreVector) -> bool:
        values = (self.cognitive, self.emotional, self.systemic)
        other_values = (other.cognitive, other.emotional, other.systemic)
        return all(left >= right for left, right in zip(values, other_values, strict=True)) and any(
            left > right for left, right in zip(values, other_values, strict=True)
        )

    @property
    def sum(self) -> float:
        return self.cognitive + self.emotional + self.systemic


@dataclass(frozen=True, slots=True)
class BehaviorDescriptor:
    """Normalized MAP-Elites coordinates for one candidate phenotype."""

    slow_path_rate: float
    rewire_rate: float
    average_candidates: float
    normalized_node_count: float

    def __post_init__(self) -> None:
        _unit(self.slow_path_rate, "slow_path_rate")
        _unit(self.rewire_rate, "rewire_rate")
        _unit(self.average_candidates, "average_candidates")
        _unit(self.normalized_node_count, "normalized_node_count")

    def values(self) -> tuple[float, ...]:
        return (
            self.slow_path_rate,
            self.rewire_rate,
            self.average_candidates,
            self.normalized_node_count,
        )


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One append-only candidate record held by the archive."""

    program_id: str
    genotype_hash: str
    score: ScoreVector
    descriptor: BehaviorDescriptor
    generation: int

    def __post_init__(self) -> None:
        if not self.program_id.strip():
            raise ValueError("program_id must not be empty")
        if not self.genotype_hash.strip():
            raise ValueError("genotype_hash must not be empty")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")


@dataclass(frozen=True, slots=True)
class ArchiveConfig:
    """Fixed grid geometry and per-cell Pareto capacity."""

    bins: tuple[int, ...] = (6, 6, 6, 6)
    cell_capacity: int = 2

    def __post_init__(self) -> None:
        if not self.bins or any(value < 1 for value in self.bins):
            raise ValueError("bins must contain positive values")
        if self.cell_capacity < 1:
            raise ValueError("cell_capacity must be positive")


class ParetoArchive:
    """Deterministic bounded archive with no scalarized fitness ordering."""

    def __init__(self, config: ArchiveConfig | None = None) -> None:
        self.config = config or ArchiveConfig()
        self._cells: dict[tuple[int, ...], tuple[ArchiveEntry, ...]] = {}

    def cell_key(self, descriptor: BehaviorDescriptor) -> tuple[int, ...]:
        values = descriptor.values()
        if len(values) != len(self.config.bins):
            raise ValueError("descriptor dimensionality must match archive bins")
        return tuple(
            min(int(value * bins), bins - 1)
            for value, bins in zip(values, self.config.bins, strict=True)
        )

    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(
            entry
            for key in sorted(self._cells)
            for entry in self._cells[key]
        )

    @property
    def filled_cells(self) -> int:
        return len(self._cells)

    @property
    def coverage(self) -> float:
        total = 1
        for bins in self.config.bins:
            total *= bins
        return self.filled_cells / total

    def update(self, candidate: ArchiveEntry) -> bool:
        """Insert a non-dominated candidate and return whether it was kept."""

        key = self.cell_key(candidate.descriptor)
        current = self._cells.get(key, ())
        if any(entry.score.dominates(candidate.score) for entry in current):
            return False
        survivors = [
            entry for entry in current if not candidate.score.dominates(entry.score)
        ]
        survivors.append(candidate)
        survivors.sort(
            key=lambda entry: (-entry.score.sum, entry.genotype_hash, entry.program_id)
        )
        kept = tuple(survivors[: self.config.cell_capacity])
        if candidate not in kept:
            if kept:
                self._cells[key] = kept
            else:
                self._cells.pop(key, None)
            return False
        self._cells[key] = kept
        return True

    def cell(self, key: tuple[int, ...]) -> tuple[ArchiveEntry, ...]:
        return self._cells.get(key, ())
