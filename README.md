# SSA — Digital Lifeform Prototype

> Stateless Space Activation (SSA) — a single-user, text-first, locally-persisted digital lifeform.

This is a research prototype, not a commercial product. See
[`development_pipeline.md`](development_pipeline.md) for the full
engineering baseline and [`project_plan.md`](project_plan.md) for
technology selection rationale.

---

## Quickstart (Windows)

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.11+ (uv will install it if missing)

### Install

```powershell
cd e:\zerobot\ssa
uv sync                  # install runtime + dev deps
uv sync --extra all      # install everything (telegram, cli, scheduler, dashboard)
```

### Configure

```powershell
copy .env.example .env
# edit .env and fill in SSA_LLM_API_KEY etc.
```

### Initialize the database

```powershell
uv run python scripts/init_db.py
```

### Run checks

```powershell
uv run pytest                    # run tests
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src/ssa              # type check
```

### Run the doctor

```powershell
uv run ssa doctor
```

### Chat (CLI)

```powershell
uv run ssa chat
```

### Inspect

```powershell
uv run ssa inspect state
uv run ssa inspect event EVENT_ID
uv run ssa inspect memory MEMORY_ID
uv run ssa inspect goals
uv run ssa replay CORRELATION_ID
uv run ssa export PATH
```

---

## Project layout

```
ssa/
├── pyproject.toml
├── .env.example
├── config/           # version-controlled defaults
├── migrations/       # SQL migrations (append-only)
├── src/ssa/          # core package
│   ├── domain/       # pure domain models
│   ├── storage/      # repositories + database
│   ├── services/     # business logic
│   ├── runtime/      # orchestrator + scheduler
│   ├── adapters/     # llm, embedding, telegram
│   ├── interfaces/   # cli, telegram_bot, admin_api
│   ├── prompts/      # jinja2 templates
│   └── observability/
├── tests/
├── evals/
├── scripts/
├── data/             # SQLite db (gitignored)
└── notes/            # milestones, decisions, incidents
```

## Baselines (ablation switches)

| Baseline | Stack |
|----------|-------|
| B0 | pure LLM |
| B1 | LLM + naive memory retrieval |
| B2 | B1 + weighted retrieval + source provenance |
| B3 | B2 + appraisal + state |
| B4 | B3 + relationship + identity |
| B5 | B4 + goals + lifecycle + initiative |

Set via `config/defaults.toml` → `[ablation].baseline`.

## License

MIT.
