# Milestone: Phase 1 — Persistent Event Store

**Date:** 2026-07-25
**Phase:** 1
**Status:** ✅ Complete

## Deliverables

- 5 SQL migration files (`001_core.sql` through `005_audit_jobs_outbox.sql`)
- `src/ssa/domain/enums.py` — Actor, SourceKind, MemoryType, ActionIntent enums
- `src/ssa/domain/events.py` — IncomingSignal, Event, normalize_signal
- `src/ssa/storage/database.py` — Database with WAL, migration runner, transactions
- `src/ssa/storage/event_repository.py` — SqliteEventRepository (append, get, find_by_channel_message)
- `scripts/init_db.py` — Database initialization script

## Schema

18 tables created across 5 migrations:
- `001_core`: schema_migrations, events, idempotency_keys, config_meta
- `002_memory`: memories, memory_evidence, memory_links, memory_vec (vec0)
- `003_state_relationship`: state_snapshots, relationship_snapshots, appraisals
- `004_identity_goals`: self_beliefs, goals, goal_steps
- `005_audit_jobs_outbox`: llm_calls, causal_traces, scheduled_jobs, initiatives, outbox

## Tests

- `tests/component/test_database.py` — 12 tests: init, schema, WAL, FK, transactions, integrity
- `tests/component/test_event_repository.py` — 17 tests: CRUD, dedup, 1000-event scale, rollback, restart, normalization

**Total: 69 tests (40 unit + 29 component), all passing.**

## Gate Verification

```text
uv run pytest     → 69 passed
uv run ruff check → All checks passed
uv run mypy src/ssa → Success: no issues found in 16 source files
```

## Pipeline §11.5 Acceptance

| Requirement | Status |
|-------------|--------|
| 1000 concurrent reads + serial writes | ✅ (1000-event test) |
| Duplicate channel message → one Event | ✅ (UNIQUE constraint + DuplicateEventError) |
| Transaction mid-exception → no partial state | ✅ (rollback test) |
| Optimistic lock retry | 🔄 (deferred to M08/M14 snapshot repos) |

## Notes

- sqlite-vec extension not yet loaded; `memory_vec` table creation is gracefully skipped
- Migration runner uses custom SQL splitter (not `executescript`) to maintain transaction atomicity
- `vec_extension_loaded` property available for startup checks
- Migration path resolution: `src/ssa/storage/database.py` → `../../../../migrations/`
