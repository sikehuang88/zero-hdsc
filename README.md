# 超维度空间计算（HDSC）

> Hyperdimensional Space Computing — a single-user, text-first digital-life
> research runtime with a local, inspectable high-dimensional trace substrate.

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

### Launch the interactive terminal

```powershell
uv run hdsc tui
```

The TUI and web gateway register the persistent P0-P6 lifecycle at startup and
advance it every two seconds without overlapping the foreground chat turn. The
web gateway owns this loop independently of browser polling, so minimized or
throttled frontend windows do not pause learning. Proactive messages
are delivered through the durable outbox and appear in the same conversation
as `agent.initiative` events.

For continuous autonomy without keeping the TUI open, run the standalone
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
Open the `ACTIVITY` tab or enter `/activity` to inspect the actual history and
artifact excerpt. A claim about past offline learning is eligible for the chat
prompt only when a completed episode and its artifact both exist. Closing every
interactive host (TUI and web gateway) and `hdsc worker` stops background action;
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
feedback is negative; silence is inconclusive. The `ACTIVITY` tab exposes run,
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

The terminal workspace keeps the live conversation on the left. Its default
`SPACE` inspector renders the persistent trace space as a two-dimensional PCA
projection of real embeddings: `@` marks main activation, `+` radiation, `o`
the newest trace, and `:` semantic links. The trace list is keyboard-selectable
and exposes activation score, freshness, importance, source, and raw content.
Below 100 columns the workspace uses a 14-row compact map and collapses the
trace list/detail panes, keeping the complete H2 status block visible in an
`80x30` terminal. The full 18-row map remains visible at `120x42`.

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
`STATE` inspector shows the latest local clock, mode, confidence, and gap. P1
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

The remaining inspector tabs expose organism state, relationship state,
persisted long-term memories, and evidence-backed self-beliefs. The status line
reports request latency, token usage, and KV Cache hit ratio.

### Multimodal attachments

The TUI can queue up to eight images or files for the next message. The
production adapter uses the OpenAI-compatible Responses API at
`https://sky1818.com/v1/responses` with `gpt-5.6-luna`. Configure the secret as
`HDSC_MULTIMODAL_API_KEY` (or `MAGICAI_API_KEY`) in the ignored local `.env`.

```text
/attach "E:\research\diagram.png"
/attach "E:\research\paper.pdf"
/attachments
/detach 1
/clear-attachments
```

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

Use a separate persistent conversation when needed:

```powershell
uv run hdsc tui --conversation research-notes
```

Keyboard controls: `Ctrl+Q` quit, `Ctrl+R` refresh, `Ctrl+L` clear the visible
log, and `F1` show slash commands. The main commands are `/space`, `/state`,
`/relation`, `/memory`, `/identity`, `/activity`, `/attach`, `/attachments`,
`/detach`, `/clear-attachments`, `/refresh`, `/clear`, and `/quit`.

Typing `/` opens the command palette above the composer. Prefix text filters
the central command registry; use `Up`/`Down` to select, `Tab` to complete,
`Enter` to complete a partial command or execute a complete command, and
`Escape` to close the palette. Command names, aliases, argument hints, help
text, and action IDs are registered in `src/ssa/commands/registry.py`, so the
palette and `/help` remain synchronized as capabilities are added.

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
