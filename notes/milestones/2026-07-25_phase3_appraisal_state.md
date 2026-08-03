# Milestone: Phase 3 — Appraisal and State

**Date:** 2026-07-25
**Phase:** 3
**Status:** ✅ Complete

## Deliverables

### M07: Appraisal Pipeline
- `src/ssa/domain/appraisal.py` — AppraisalResult with bounded fields + neutral fallback
- `src/ssa/services/appraisal_service.py` — LLM-based appraisal with evidence validation and neutral degradation
- `tests/fixtures/appraisal_standard_events.v1.json` — 50 hand-authored Appraisal schema/contract events plus malicious state-injection cases
- `tests/contract/test_appraisal_fixture_contract.py` — fixed `FakeLLMAdapter` contract coverage for ranges, evidence references, and Pydantic boundaries

### M08: Instant State Engine
- `src/ssa/domain/state.py` — OrganismState + DeterministicStateEngine (preview + finalize, no LLM)
- `src/ssa/storage/state_repository.py` — versioned snapshot repository with optimistic locking

## Tests

- `tests/component/test_appraisal_state.py` — 25 tests for appraisal validation and the state engine
- `tests/component/test_appraisal_state_persistence.py` — 9 persistence and optimistic-lock tests
- Includes 100-iteration random property test for state bounds
- `tests/contract/test_appraisal_fixture_contract.py` — 55 contract checks: 50 standard events, 4 malicious payloads, and fixture metadata

The 50 event outputs are hand-authored schema/contract fixtures. They do not calibrate,
benchmark, or make behavioral claims about a real LLM.

The M07 contract target is verified separately below so these hand-authored
fixtures are not confused with model-evaluation coverage.

## Gate Verification

```text
uv run pytest     → 232 passed (current project suite)
uv run ruff check → All checks passed
uv run mypy src/ssa → Success: no issues found in 35 source files
```

## Pipeline §17.5 Acceptance (M08)

| Requirement | Status |
|-------------|--------|
| State always in defined domain | ✅ (100-iteration random test) |
| Same old state + appraisal → same new state | ✅ (deterministic test) |
| Each dimension has recovery, increase, boundary tests | ✅ |
| State engine does not call LLM | ✅ (no LLM dependency) |

## Pipeline §16.5 Acceptance (M07)

| Requirement | Status |
|-------------|--------|
| 50 standard events with expected ranges | ✅ (schema/contract fixtures; not real-model calibration) |
| Reproducible with same input + fixed params | ✅ (deterministic) |
| Malicious text cannot directly set state values | ✅ (4 injection contracts; extras dropped or neutral fallback) |
| Degradation does not block normal reply | ✅ (neutral fallback) |

### M07 Contract Verification

```text
uv run pytest tests/contract/test_appraisal_fixture_contract.py -q
→ 55 passed
```
