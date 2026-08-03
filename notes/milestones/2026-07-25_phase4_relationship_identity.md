# Milestone: Phase 4 - Relationship and Identity Core

**Date:** 2026-07-25
**Phase:** 4
**Status:** Core implementation complete; live-model evaluation pending

## Deliverables

### M09: Relationship Evolution

- `src/ssa/domain/relationship.py` - bounded relationship snapshots and evidence-bearing transition inputs
- `src/ssa/services/relationship_service.py` - deterministic `preview` / `finalize` evolution with whole-turn delta caps
- `src/ssa/storage/relationship_repository.py` - version chain, predecessor links, and atomic optimistic locking
- Promise, unresolved-memory, ritual, conflict, and repair effects use structured event metadata rather than free-text parsing

### M10: Evidence-Backed Self Beliefs

- `src/ssa/domain/self_belief.py` - candidate, active, challenged, revised, and archived lifecycle
- `src/ssa/services/self_belief_service.py` - two-event creation gate, 0.6 candidate cap, seven-day activation gate, counterevidence, revision, and prompt selection
- `src/ssa/storage/self_belief_repository.py` - append-only lineage versions and normalized evidence links
- `migrations/007_self_belief_versions.sql` - predecessor/audit fields plus foreign-key-backed evidence

### Runtime Hardening

- `migrations/008_memory_vec_table.sql` keeps vector-table creation pending when sqlite-vec is temporarily unavailable
- `src/ssa/interfaces/cli.py` provides dependency-free `ssa doctor` and `ssa init-db` commands
- A clean core-only install now exposes a working console entry point

## Tests

- `tests/component/test_relationship.py` - 13 focused relationship tests
- `tests/component/test_self_belief.py` - 9 lifecycle, migration, provenance, and rollback tests
- `tests/component/test_database.py` - fresh schema v8 plus v5/v6 and deferred-vector recovery paths
- `tests/unit/test_cli.py` - 3 installed CLI behavior tests

## Gate Verification

```text
uv run pytest                    -> 232 passed
uv run pytest --cov=ssa          -> 87% total coverage
uv run ruff check .              -> All checks passed
uv run mypy src/ssa              -> Success: 35 source files
core-only uv sync --frozen       -> installed successfully
installed ssa doctor             -> schema v8, sqlite-vec loaded, status ok
```

## Acceptance

| Requirement | Status |
|-------------|--------|
| Ordinary and major relationship changes respect 0.03 / 0.10 caps | Complete |
| Relationship growth requires repeated events | Complete |
| Neutral Appraisal does not raise trust or closeness | Complete |
| Candidate identity needs at least two real events | Complete |
| Candidate confidence is capped and age-gated before activation | Complete |
| Counterevidence remains queryable and can challenge an active belief | Complete |
| Revised claims retain their predecessor and reason | Complete |
| Empty self-belief storage leaves the base system operational | Complete |

## Evaluation Boundary

The component suite establishes deterministic contracts and causal provenance.
It does not establish that a real model exhibits convincing relationship or
identity continuity. The planned 30 relationship trajectories, 40 identity
probe groups, multi-seed runs, and cross-model comparisons remain Phase 7
evaluation work.
