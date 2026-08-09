# 超维度空间计算（HDSC）

> Hyperdimensional Space Computing — a single-user, text-first digital-life
> research runtime with a local, inspectable high-dimensional trace substrate.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-sqlite--vec-003B57?logo=sqlite&logoColor=white)
![LiteLLM](https://img.shields.io/badge/LLM-DeepSeek%20V4-4D6BFE)
![ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)
![License](https://img.shields.io/badge/license-private-lightgrey)

> ZERO 的"大脑":会话无状态推理核 + append-only 痕迹空间。表现层(官网/桌面/移动端)见
> [`zero`](https://github.com/sikehuang88/zero)。

HDSC succeeds the earlier Stateless Space Activation (SSA) prototype. `ssa`
remains the internal Python package and a compatibility CLI while the research
formalization is rebuilt. Legacy SSA terminology denotes reproducible baseline
models only and carries no thermodynamic-compliance claim.

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
uv sync                  # install core + dev dependencies
uv sync --extra runtime  # add real LLM and local embedding adapters
uv sync --extra all      # install every optional interface and adapter
```

### Configure

```powershell
copy .env.example .env
# edit .env and fill in HDSC_LLM_API_KEY (or DEEPSEEK_API_KEY)
```

HDSC uses the official DeepSeek OpenAI-compatible endpoint and two explicit V4
routes. Model names include the `deepseek/` provider prefix expected by LiteLLM:

| Workload | Model | Thinking |
|----------|-------|----------|
| Frequent structured work (memory extraction and appraisal) | `deepseek/deepseek-v4-flash` | `disabled` |
| Complex identity/evidence review | `deepseek/deepseek-v4-pro` | `enabled`, `high` |

The checked-in defaults can be overridden from `.env`:

```dotenv
HDSC_LLM_MODEL=deepseek/deepseek-v4-flash
HDSC_LLM_REASONING_MODEL=deepseek/deepseek-v4-pro
HDSC_LLM_BASE_URL=https://api.deepseek.com
HDSC_LLM_BETA_BASE_URL=https://api.deepseek.com/beta
HDSC_LLM_THINKING_MODE=disabled
HDSC_LLM_REASONING_EFFORT=high
HDSC_DEEPSEEK_USER_ID=hdsc-primary
```

`HDSC_LLM_REASONING_EFFORT` accepts `high` or `max`. The stable
`HDSC_DEEPSEEK_USER_ID` contains only letters, digits, `_`, or `-` (maximum 512
characters); do not put personal data in it. The adapter selects the Beta
endpoint only for features that require it, such as strict function tools or
assistant prefix completion. See the
[DeepSeek V4 adapter research note](notes/research/2026-07-25_deepseek_v4_adapter.md)
for the complete parameter matrix and operational constraints.

### Initialize the database

```powershell
uv run hdsc init-db
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
uv run hdsc doctor
```

The doctor reports the effective main and reasoning models, endpoint,
thinking mode/effort, multimodal model and endpoint, LiteLLM runtime presence,
and whether each API secret is configured. It never prints secret values or
makes a paid API request.

### Launch the desktop gateway

```powershell
uv run hdsc web
```

The desktop client (`zero-web`) starts this gateway itself as a sidecar; run it
by hand only when driving the API directly. Both paths default to the
`cli-primary` conversation, which holds the accumulated trace space.

The web gateway registers the persistent P0-P6 lifecycle at startup and
advance it every two seconds without overlapping the foreground chat turn. The
web gateway owns this loop independently of browser polling, so minimized or
throttled frontend windows do not pause learning. Proactive messages
are delivered through the durable outbox and appear in the same conversation
as `agent.initiative` events.

For continuous autonomy without keeping the desktop app open, run the standalone
worker in another terminal:

```powershell
uv run hdsc worker
uv run hdsc worker --conversation research-notes
uv run hdsc worker --once
```

The worker advances fourteen idempotent recurring jobs: contextual inner-loop
heartbeat, offline agency, reflection scheduling, evidence-gated learning
consolidation, outcome evaluation, state refresh, health, resting trace
activation, goal progress, initiative evaluation/expiry, contact episodes,
outbox delivery, and world observation. `[world]` in
`config/defaults.toml` controls the human clock, watched paths, structured
calendar JSON files, optional Windows foreground-window observation, and the
material-salience threshold. Run one worker process per database.

Offline agency starts only after the configured user-absence interval and
executes a bounded trace reflection or external-truth study. Every successful
cycle persists an `offline_episodes` row, an evidence-linked
`offline_artifacts` row, a lifecycle event, and the associated energy cost.
Query the activity API to inspect the actual history and
artifact excerpt. A claim about past offline learning is eligible for the chat
prompt only when a completed episode and its artifact both exist. Closing the
desktop gateway and `hdsc worker` stops background action;
on the next launch the clock gap is observed, but no actions are backfilled for
the stopped interval.

After every agent reply, the contextual inner loop estimates reply expectation
from question, support, affect, relationship, pause, and closure signals. It
derives a bounded wait deadline instead of treating every silence alike. The
60-second `inner.heartbeat` advances only a compact persisted latent vector and
mode (`engaged`, `waiting`, `offline`, or `quiet_rest`). It does not generate a
free-form private monologue. Offline study, proactive initiative, and contact
follow-up share this mode gate; cooldown, quiet hours, daily limits, and contact
bounds remain hard downstream constraints.

Continuous learning is a separate evidence-gated loop. A completed offline
artifact becomes one `reflection_run`; the deterministic critic scores source
coverage, provenance trust, consistency, and falsifiability. Approved proposals
are routed to memory or to a reversible behavior experiment. Experiments are
confirmed only by explicit later user feedback and are rolled back when that
feedback is negative; silence is inconclusive. The activity API exposes run,
proposal, target, experiment, and evidence counts so a claimed change can be
traced back to its source.

The emotional-memory library is separate from both factual `memories` and
short-lived `emotion_episodes`. Each salient interaction can create one
`emotional_memories` record containing the trigger, subjective emotion type,
target, valence, arousal, intensity, action tendency, source events, and source
trace. Before a reply, recall combines topic overlap, activated-trace support,
intensity, and recency. Recalled emotion may shape warmth, caution, and attention,
but it is explicitly excluded from user/world factual evidence. Startup performs
an idempotent bounded backfill from existing traces; low-salience history stays out.

The desktop client renders the persistent trace space as a two-dimensional PCA
projection of real embeddings, distinguishing main activation, radiation, the
newest trace, and semantic links, and exposes activation score, freshness,
importance, source, and raw content per trace.

At startup, existing user/agent event pairs are indexed into the trace space
idempotently. The current `legacy-ssa-a0` baseline activates old traces with
`semantic similarity * freshness * importance` and performs one bounded
radiation hop. It is retained for replay and comparison while the HDSC
propagation kernel proceeds through cross-disciplinary validation; it is not
described as a thermodynamic model. After the reply, the complete interaction
is appended as a new immutable episode trace. `MEMORY` remains a separate view for extracted,
deduplicated long-term facts; it is not the trace-space substrate.

Before each main reply, the P1 environment-perception layer projects the UTC
event timestamp onto the configured IANA human clock, derives day phase, quiet
hours, and conversation gap, then crosses deterministic semantic cues with the
current structured appraisal, relationship state, and activated-trace context.
The normalized situation mode and continuous response posture are stored in
`perception_snapshots` and reconstructed into private prompt context. The
desktop state panel shows the latest local clock, mode, confidence, and gap. P1
is a serving control layer; it is not the unimplemented YUANZI-5D active-
inference loop proposed by the paper.

### Scientific status

The memory activation path is still `legacy-ssa-a0`, a reproducible C0
baseline. Software v0.5 adds `hdsc-p1-environment-perception` to the serving
control path without promoting H1D or H2.
`hdsc-h1-c1-shadow` remains the reversible symmetric reference. Real
multi-relation topology is evaluated by `hdsc-h1d-directed-markov-shadow`,
which preserves edge direction with a nonreversible continuous-time Markov
generator and never symmetrizes the graph. The active-state candidate is
`hdsc-h2-bounded-active-shadow`: the append-only
archive is separated from a replay-derived active measure with fixed capacity,
unit total mass, a null reservoir, deterministic semantic buckets, duplicate
suppression, hysteresis, and a conditional small-gain certificate. H2 advances
after each complete episode and is rebuilt by chronological replay at startup.
It has no prompt effect. Promotion requires the validation gates in
[`notes/research/2026-07-26_hdsc_validation_framework.md`](notes/research/2026-07-26_hdsc_validation_framework.md).

The remaining desktop panels expose organism state, relationship state,
persisted long-term memories, and evidence-backed self-beliefs, plus request
latency, token usage, and KV Cache hit ratio.

### Multimodal attachments

Up to eight images or files can be queued for the next message. The
production adapter uses the OpenAI-compatible Responses API at
`https://sky1818.com/v1/responses` with `gpt-5.6-luna`. Configure the secret as
`HDSC_MULTIMODAL_API_KEY` (or `MAGICAI_API_KEY`) in the ignored local `.env`.

Images are sent as `input_image`, PDF/Office documents as `input_file`, and
text/code formats as attributed `input_text`. Multiple files share one
Responses request. Binary Base64 exists only in that request; the event ledger
stores paths, MIME types, sizes, SHA-256 hashes, model/response IDs, and the
provider-reported analysis. The analysis enters DeepSeek's private context as
attributed model inference, while the visible final answer still comes from the
persistent digital-life conversation model.

### Autonomous host-tool kernel

`src/ssa/tools/` is an interface-independent capability kernel. Its registry is
sent to DeepSeek as a stable leading tool definition, while the latest organism,
relationship, appraisal, and environment-perception state remains in the private
decision context. DeepSeek uses `tool_choice=auto` and may complete up to eight
tool rounds before producing the visible reply.

The default registry exposes `powershell`, `read_file`, and `write_file`.
Filesystem paths are resolved against the host and are not constrained to the
project directory. `powershell` accepts any working directory and an
`elevated=true` flag; on Windows the elevation broker relaunches the payload with
`Start-Process -Verb RunAs`, waits for completion, and returns its output to the
model. Windows displays its normal UAC consent surface for that transition.

When `[multimodal]` is enabled and `HDSC_MULTIMODAL_API_KEY` is configured,
the same registry adds `ask_gpt`. DeepSeek may use it with `tool_choice=auto`
to request an independent GPT analysis, second opinion, or image/PDF/file
interpretation. The tool accepts a focused prompt, optional instructions, and
up to eight absolute file paths. GPT receives no tool registry of its own, so
delegation terminates after one bounded Responses call and returns to DeepSeek
for critical integration into the final reply. Audit events retain provider,
model, response ID, token counts, and file hashes without persisting request
Base64 or the complete delegated output.

On Windows, `[win32]` adds a separate read-only native capability registry.
These tools call `user32.dll`, `kernel32.dll`, and `psapi.dll` directly through
`ctypes`; they do not shell through PowerShell and do not require `pywin32`:

- `win32_active_window`: foreground title/process and time since last input.
- `win32_visible_windows`: visible top-level windows and owning processes.
- `win32_processes`: running process IDs, names, and optional executable paths.
- `win32_drives`: drive types, labels, filesystems, capacity, and free space.
- `win32_list_directory`: bounded `FindFirstFileW`/`FindNextFileW` enumeration.
- `win32_clipboard_text`: bounded Unicode clipboard text and sequence number.

DeepSeek receives these definitions in the normal autonomous tool loop and can
select the smallest observation needed for the current conversation. Results
are current host observations rather than user-authored facts. Tool audit events
store only the observation kind, provider, timestamp, record count, and hashes;
window lists, file names, process paths, and clipboard text remain in the
current model turn. Limits and clipboard registration are controlled by
`[win32]` and the corresponding `HDSC_WIN32_*` environment values.

The isolated external-truth registry adds keyless, fixed-provider tools for
Open-Meteo weather, Sunrise-Sunset daylight times, Nager.Date holidays,
Nominatim geocoding, Noozra headlines, Radio Browser stations, MusicBrainz
metadata, MyMemory translation, and official GitHub/Cloudflare/Discord/OpenAI
status pages. These tools accept structured parameters and never accept an
arbitrary URL. Requests use HTTPS host allowlists, timeouts, response-size
limits, per-provider pacing, and an in-memory cache.

Every result uses the `external-observation-v1` envelope with provider URL,
retrieval time, cache state, and `truth_status=provider_reported`. A tool result
is external evidence for the current response; it is not automatically written
as a user fact, relationship fact, or long-term memory. Runtime controls are
under `[external_truth]` in `config/defaults.toml`.

The bundled `zero-web-search` skill adds Firecrawl-backed live discovery and
page extraction through `firecrawl_search` and `firecrawl_scrape`. Configure
`HDSC_FIRECRAWL_API_KEY` (or `FIRECRAWL_API_KEY`) in the ignored local `.env`;
the key is never accepted as a tool argument or written to the event ledger.
The skill searches first, reads primary sources, cross-checks material claims,
and returns direct source URLs. Without a configured key, Zero keeps the skill
available but falls back to its embedded-browser and fixed-provider web tools.
Behavioral limits and the API base URL live under `[firecrawl]` in
`config/defaults.toml`.

Tool arguments and output are returned to the current model loop. The immutable
event ledger stores only tool name, result state, timing, elevation state, exit
code, and argument/output hashes. Runtime controls are under `[tools]` in
`config/defaults.toml` and the corresponding `HDSC_TOOLS_*` environment values.

### Grounded predictions

The live session exposes `record_grounded_prediction` for concrete claims with a
machine-checkable `event_match` predicate. Predictions are append-only records
with a finite verification window; the lifecycle worker resolves them from
immutable events without asking the model to grade its own output. Resolved
claims contribute `(confidence - outcome)^2` to the Brier score, while the
dashboard snapshot includes ten confidence buckets, observed rates, and a
base-rate baseline. External-truth and tool-result verifier shapes are accepted
as explicit `unverifiable` records until their fixed provider adapters are wired.

The `[emotion_library].library_root` setting (or `HDSC_EMOTION_LIBRARY_ROOT`)
controls the optional authored expression corpus. `hdsc doctor` reports its
resolved path and loaded scene/entry counts instead of allowing a missing corpus
to fail silently.

Use a separate persistent conversation when needed:

```powershell
uv run hdsc web --conversation research-notes
```

Traces, state, and events are isolated per `conversation_id`, so a separate
conversation is a separate life. The default `cli-primary` holds the
accumulated history; do not point the gateway at a new ID unless a blank space
is intended.

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
│   ├── hdsc/         # H1 reversible + H1D directed + H2 bounded shadow kernels
│   ├── storage/      # repositories + database
│   ├── services/     # business logic, including P1 environment perception
│   ├── runtime/      # orchestrator + scheduler
│   ├── builtin_skills/ # bundled official-format SKILL.md packages
│   ├── tools/        # autonomous capability registry + global host executors
│   ├── adapters/     # llm, embedding, telegram
│   ├── interfaces/   # cli, web gateway (desktop client API)
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

## SEOS shadow kernel

The first SEOS implementation is a side-effect-free offline kernel. It provides
typed primitive operators, immutable DAG programs, `Eff` linearity checking,
canonical genotype hashes, deterministic point variation and a bounded
Pareto/MAP-Elites-style archive. It is shadow-only and does not participate in
the live conversation path.

```python
from ssa.hdsc import (
    HV,
    Node,
    Program,
    ProgramInput,
    check_linearity,
    program_digest,
    type_check,
)

program = Program(
    inputs=(ProgramInput("left", HV), ProgramInput("right", HV)),
    nodes=(Node("root", "bind", ("left", "right")),),
    outputs=("root",),
)
type_check(program)
check_linearity(program)
print(program_digest(program))
```

Implementation files:

- `src/ssa/hdsc/operators.py` — type AST, signatures and primitive registry.
- `src/ssa/hdsc/program.py` — typed DAG checks, effect linearity, digest and cost.
- `src/ssa/hdsc/variation.py` — deterministic mutation and one-node crossover.
- `src/ssa/hdsc/archive.py` — separated Pareto objectives and behavior archive.
- `tests/unit/test_seos_program.py` — P0/P1 unit coverage.

### Retrodiction self-play

The first offline fitness path now evaluates SEOS programs against immutable
archive slices. `ArchivedEvent` records are split by a historical cut point;
`build_episodes` creates recurrence episodes, `evidence_before` enforces strict
no-future leakage, and `RetrodictionEvaluator` scores probabilities with Brier
skill against a topic base rate. `EvolutionService` can use that evaluator for
zero-LLM tier-1 filtering and validation/archive selection.

```python
from ssa.hdsc import ArchivedEvent, EpisodeConfig, build_episodes
from ssa.services.retrodiction_evaluator import RetrodictionEvaluator

archive = [ArchivedEvent("e1", 1_700_000_000_000, "user", "关于跑步的记录")]
episodes = build_episodes(archive, EpisodeConfig())
```

The loop is shadow-only: it evolves offline retrodiction candidates and does
not promote a champion into the live conversation path. Synthetic archive
benchmarks are regression fixtures, not evidence of online improvement.

## License

MIT.
