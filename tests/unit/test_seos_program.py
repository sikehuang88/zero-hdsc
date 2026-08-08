"""P0 tests for the SEOS typed operator/program kernel."""

from __future__ import annotations

import pytest

from ssa.hdsc.archive import (
    ArchiveConfig,
    ArchiveEntry,
    BehaviorDescriptor,
    ParetoArchive,
    ScoreVector,
)
from ssa.hdsc.interpreter import (
    BudgetExceeded,
    EffectBoundaryError,
    ExecBudget,
    ExecContext,
    ExecutionValue,
    execute,
    static_budget_gate,
)
from ssa.hdsc.operators import EFF, EV, HV, Operator, OperatorRegistry, Signature, Z
from ssa.hdsc.program import (
    LinearityError,
    Node,
    Program,
    ProgramInput,
    ProgramTypeError,
    check_linearity,
    program_digest,
    static_cost,
    subtree_digests,
    type_check,
)
from ssa.hdsc.variation import (
    VariationError,
    apply_random_variation,
    crossover_subgraph,
    mutate_delete,
    mutate_insert,
    mutate_point,
)


def test_signature_checks_effect_flux() -> None:
    with pytest.raises(ValueError, match="eff_flux"):
        Signature((EFF,), HV, eff_flux=0)


def test_effecting_operator_requires_an_eff_input() -> None:
    with pytest.raises(ValueError, match="effecting"):
        Operator("bad", Signature((EV,), EV, 0), "effecting", "none")


def test_pure_program_type_checks() -> None:
    program = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("bound", "bind", ("left", "right")),),
        outputs=("bound",),
    )
    type_check(program)
    check_linearity(program)


def test_type_checker_rejects_wrong_edge_type() -> None:
    program = Program(
        inputs=(ProgramInput("event", EV), ProgramInput("right", HV)),
        nodes=(Node("bound", "bind", ("event", "right")),),
        outputs=("bound",),
    )
    with pytest.raises(ProgramTypeError, match="expects"):
        type_check(program)


def test_type_checker_rejects_cycles() -> None:
    program = Program(
        inputs=(),
        nodes=(
            Node("a", "permute", ("b",)),
            Node("b", "permute", ("a",)),
        ),
        outputs=("a",),
    )
    with pytest.raises(ProgramTypeError, match="cycle"):
        type_check(program)


def _effect_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register(Operator("emit", Signature((EV, EFF), EV, 1), "effecting", "none"))
    return registry


def test_eff_chain_is_consumed_once() -> None:
    program = Program(
        inputs=(ProgramInput("event", EV), ProgramInput("token", EFF)),
        nodes=(Node("emit", "emit", ("event", "token")),),
        outputs=("emit",),
    )
    registry = _effect_registry()
    type_check(program, registry)
    check_linearity(program, registry)


def test_eff_drop_is_rejected() -> None:
    program = Program(
        inputs=(ProgramInput("token", EFF),),
        nodes=(),
        outputs=("token",),
    )
    with pytest.raises(LinearityError, match="exactly one consumer"):
        check_linearity(program)


def test_eff_copy_is_rejected() -> None:
    program = Program(
        inputs=(ProgramInput("event", EV), ProgramInput("token", EFF)),
        nodes=(
            Node("first", "emit", ("event", "token")),
            Node("second", "emit", ("event", "token")),
        ),
        outputs=("first",),
    )
    with pytest.raises(LinearityError):
        check_linearity(program, _effect_registry())


def test_program_digest_ignores_node_ids_and_node_order() -> None:
    first = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "bind", ("left", "right")),),
        outputs=("root",),
    )
    second = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("renamed", "bind", ("left", "right")),),
        outputs=("renamed",),
    )
    assert program_digest(first) == program_digest(second)


def test_commutative_inputs_are_canonicalized() -> None:
    registry = OperatorRegistry()
    registry.register(
        Operator(
            "commute",
            Signature((HV, HV), HV, 0, (True, True)),
            "pure",
            "exact",
        )
    )
    first = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "commute", ("left", "right")),),
        outputs=("root",),
    )
    second = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "commute", ("right", "left")),),
        outputs=("root",),
    )
    assert program_digest(first, registry) == program_digest(second, registry)


def test_subtree_digests_count_nodes() -> None:
    program = Program(
        inputs=(ProgramInput("value", HV),),
        nodes=(
            Node("first", "permute", ("value",)),
            Node("second", "permute", ("first",)),
        ),
        outputs=("second",),
    )
    counts = subtree_digests(program)
    assert sum(counts.values()) == 2
    assert len(counts) == 2


def test_program_depth_and_static_cost_are_precomputed() -> None:
    program = Program(
        inputs=(ProgramInput("value", Z),),
        nodes=(
            Node("first", "project", ("value",)),
            Node("second", "leak", ("first",)),
        ),
        outputs=("second",),
    )
    assert program.depth == 2
    assert static_cost(program) == pytest.approx(2.0)


def test_static_budget_gate_rejects_expensive_program_before_execution() -> None:
    program = Program(
        inputs=(ProgramInput("value", Z),),
        nodes=(
            Node("first", "project", ("value",)),
            Node("second", "leak", ("first",)),
        ),
        outputs=("second",),
    )
    report = static_budget_gate(program, ExecBudget(max_static_cost=1.0))
    assert report.admitted is False
    assert "static cost" in (report.reason or "")


def test_interpreter_executes_and_reuses_pure_cache() -> None:
    program = Program(
        inputs=(ProgramInput("value", Z),),
        nodes=(Node("root", "project", ("value",)),),
        outputs=("root",),
    )
    calls = 0

    def project(args: tuple[object, ...], _: tuple[tuple[str, object], ...]) -> object:
        nonlocal calls
        calls += 1
        return float(args[0]) + 1.0

    cache: dict[str, object] = {}
    context = ExecContext({"project": project}, shared_cache=cache)
    first = execute(program, {"value": 1.0}, ExecBudget(), context)
    second = execute(program, {"value": 1.0}, ExecBudget(), context)
    assert first.value == 2.0
    assert second.value == 2.0
    assert calls == 1
    assert second.cache_hits == 1


def test_interpreter_counts_runtime_budget() -> None:
    program = Program(
        inputs=(ProgramInput("value", Z),),
        nodes=(Node("root", "project", ("value",)),),
        outputs=("root",),
    )

    def project(
        _args: tuple[object, ...],
        _parameters: tuple[tuple[str, object], ...],
    ) -> ExecutionValue:
        return ExecutionValue(1.0, llm_calls=1, tokens=4)

    context = ExecContext({"project": project})
    with pytest.raises(BudgetExceeded, match="LLM-call"):
        execute(program, {"value": 1.0}, ExecBudget(max_llm_calls=0), context)


def test_interpreter_keeps_effects_out_of_offline_context() -> None:
    registry = _effect_registry()
    program = Program(
        inputs=(ProgramInput("event", EV), ProgramInput("token", EFF)),
        nodes=(Node("emit", "emit", ("event", "token")),),
        outputs=("emit",),
    )
    context = ExecContext({"emit": lambda args, _: args[0]})
    with pytest.raises(EffectBoundaryError):
        execute(program, {"event": "x", "token": object()}, ExecBudget(), context, registry=registry)


def test_mutate_point_preserves_type_and_changes_operator() -> None:
    program = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "bind", ("left", "right")),),
        outputs=("root",),
    )
    mutated = mutate_point(program, "root", "unbind")
    assert mutated.nodes[0].operator_id == "unbind"
    type_check(mutated)


def test_mutate_insert_and_delete_are_closed() -> None:
    program = Program(
        inputs=(ProgramInput("value", Z),),
        nodes=(Node("root", "project", ("value",)),),
        outputs=("root",),
    )
    inserted = mutate_insert(program, "root", 0, "leak")
    assert inserted.node_count == 2
    restored = mutate_delete(inserted, "root__insert__leak")
    assert restored == program


def test_mutation_rejects_signature_change() -> None:
    program = Program(
        inputs=(ProgramInput("value", Z),),
        nodes=(Node("root", "project", ("value",)),),
        outputs=("root",),
    )
    with pytest.raises(VariationError, match="identical signatures"):
        mutate_point(program, "root", "permute")


def test_crossover_reuses_donor_operator_with_same_interface() -> None:
    receiver = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "bind", ("left", "right")),),
        outputs=("root",),
    )
    donor = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "unbind", ("left", "right")),),
        outputs=("root",),
    )
    child = crossover_subgraph(receiver, donor, "root", "root")
    assert child.nodes[0].operator_id == "unbind"
    type_check(child)


def test_random_variation_is_seeded() -> None:
    program = Program(
        inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
        nodes=(Node("root", "bind", ("left", "right")),),
        outputs=("root",),
    )
    first = apply_random_variation(program, seed=17)
    second = apply_random_variation(program, seed=17)
    assert first == second


def _archive_entry(
    program_id: str,
    score: tuple[float, float, float],
    descriptor: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1),
) -> ArchiveEntry:
    return ArchiveEntry(
        program_id=program_id,
        genotype_hash=f"hash-{program_id}",
        score=ScoreVector(*score),
        descriptor=BehaviorDescriptor(*descriptor),
        generation=0,
    )


def test_score_vector_uses_pareto_dominance() -> None:
    strong = ScoreVector(0.8, 0.8, 0.8)
    weak = ScoreVector(0.7, 0.8, 0.8)
    mixed = ScoreVector(0.9, 0.6, 0.9)
    assert strong.dominates(weak)
    assert not strong.dominates(mixed)


def test_archive_rejects_dominated_candidate() -> None:
    archive = ParetoArchive(ArchiveConfig(bins=(2, 2, 2, 2), cell_capacity=2))
    assert archive.update(_archive_entry("strong", (0.8, 0.8, 0.8)))
    assert not archive.update(_archive_entry("weak", (0.7, 0.7, 0.7)))
    assert [entry.program_id for entry in archive.entries()] == ["strong"]


def test_archive_keeps_incomparable_candidates_and_tracks_coverage() -> None:
    archive = ParetoArchive(ArchiveConfig(bins=(2, 2, 2, 2), cell_capacity=2))
    assert archive.update(_archive_entry("a", (0.9, 0.4, 0.9)))
    assert archive.update(_archive_entry("b", (0.4, 0.9, 0.9)))
    assert len(archive.entries()) == 2
    assert archive.filled_cells == 1
    assert archive.coverage == pytest.approx(1 / 16)


def test_archive_descriptor_separates_behavior_cells() -> None:
    archive = ParetoArchive(ArchiveConfig(bins=(2, 2, 2, 2), cell_capacity=1))
    assert archive.update(_archive_entry("low", (0.5, 0.5, 0.5), (0.1, 0.1, 0.1, 0.1)))
    assert archive.update(_archive_entry("high", (0.4, 0.4, 0.4), (0.9, 0.9, 0.9, 0.9)))
    assert archive.filled_cells == 2
