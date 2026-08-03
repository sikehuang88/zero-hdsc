# Milestone: Phase 2 — Memory Foundation

**Date:** 2026-07-25
**Phase:** 2
**Status:** ✅ Complete

## Deliverables

### M04: Model Adapter Foundation
- `src/ssa/adapters/llm_errors.py` — 6 unified error types
- `src/ssa/adapters/embedding.py` — EmbeddingVector, EmbeddingService protocol, SentenceTransformer + Fake impls
- `src/ssa/adapters/llm.py` — ChatMessage, LLMRequest/Response, LiteLLMAdapter, FakeLLMAdapter

### M05: Memory Write Pipeline
- `src/ssa/domain/memories.py` — MemoryCandidate, Memory, RetrievedMemory
- `src/ssa/storage/memory_repository.py` — SqliteMemoryRepository (insert, find_similar, evidence, links)
- `src/ssa/services/memory_write_service.py` — extract + write with dedup/conflict/evidence/links

### M06: Memory Retrieval Pipeline
- `src/ssa/services/memory_retrieval_service.py` — prefetch + full retrieve with scoring/MMR/radiation

### Infrastructure
- sqlite-vec extension loaded in Database.initialize()
- `memories` table extended with `vec_rowid` through append-only migration 006
- migration 008 retries `memory_vec` creation when sqlite-vec becomes available later

## Tests

- `tests/unit/test_embedding.py` — 16 tests
- `tests/unit/test_llm.py` — 16 tests
- `tests/component/test_memory_services.py` — 12 tests

**Total: 113 tests, all passing.**

## Gate Verification

```text
uv run pytest     → 113 passed
uv run ruff check → All checks passed
uv run mypy src/ssa → Success: no issues found in 23 source files
```

## Notes

- sqlite-vec 0.1.9 loaded successfully; KNN queries work
- FakeEmbeddingService uses SHA-256 hash (dim=512 to match vec table)
- Runtime and mypy target Python 3.11; numpy imports use a scoped mypy override
- MMR uses cosine similarity on content embeddings for diversity
