# Milestone: Phase 0 — Engineering Baseline

**Date:** 2026-07-25
**Phase:** 0
**Status:** ✅ Complete

## Deliverables

- Git repository initialized at `e:\zerobot\ssa\`
- `pyproject.toml` with uv-based project, dev dependency group (pytest, ruff, mypy, hypothesis, pytest-asyncio, pytest-cov)
- `src/ssa/` package layout (src layout, hatchling build backend)
- `config/defaults.toml`, `config/development.toml`, `config/evaluation.toml` — version-controlled parameter files
- `.env.example` — secret template (env-only secrets)
- `tests/` directory with unit tests
- `README.md` with quickstart commands

## Modules Implemented

| Module | File | Description |
|--------|------|-------------|
| M00 | `pyproject.toml` | Package config, tooling (pytest, ruff, mypy) |
| M00 | `.gitignore` | Python, data, secrets, caches |
| M01 | `src/ssa/clock.py` | `Clock` protocol + `SystemClock` + `FrozenClock` |
| M01 | `src/ssa/ids.py` | `IdGenerator` protocol + `UuidIdGenerator` + `SequentialIdGenerator` |
| M01 | `src/ssa/config.py` | Layered settings (`load_settings`), `Secrets`, Pydantic-validated config bundles |

## Tests

- `tests/unit/test_smoke.py` — package import + version
- `tests/unit/test_clock.py` — 11 tests covering Clock protocol, SystemClock, FrozenClock
- `tests/unit/test_ids.py` — 7 tests covering IdGenerator protocol and impls
- `tests/unit/test_config.py` — 18 tests covering defaults, env overrides, secrets, validation, no-leak

**Total: 40 tests, all passing.**

## Gate Verification

```text
uv run pytest     → 40 passed
uv run ruff check → All checks passed
uv run mypy src/ssa → Success: no issues found in 5 source files
```

## Definition of Done

✅ New machine needs only Python + uv to install and run smoke test within 10 minutes.
✅ All static checks pass.
✅ All unit tests pass.
✅ Configuration is layered, typed, and secret-aware.
✅ Clock and ID abstractions are testable (frozen/sequential).

## Notes

- Python 3.11.15 installed via `uv python install 3.11`
- StrEnum used instead of `(str, Enum)` per ruff UP042
- `tomllib` from stdlib (no `tomli` fallback needed — pyproject requires 3.11+)
- Unused mypy overrides for future modules (litellm, sqlite_vec, etc.) are pre-staged
