# P1 Environment Perception Milestone

Date: 2026-07-26
Software: 0.5.0
Model: `hdsc-p1-environment-perception`
Operator: `semantic-affect-context-time-v1`

## Delivered

- UTC epoch events are projected into a configured IANA human clock.
- Time perception records local/UTC ISO values, weekday, day phase, quiet hours,
  circadian sine/cosine, previous-event gap, previous-user gap, and a categorical
  conversation gap.
- The deterministic situation operator crosses semantic controls, structured
  appraisal affect, relationship state, activated-trace context, and time.
- Eight situation modes form a normalized probability distribution:
  `answer`, `act`, `clarify`, `support`, `repair`, `reminisce`, `explore`, and
  `acknowledge`.
- Five response-posture controls enter private prompt context: warmth,
  directness, exploration, caution, and temporal sensitivity.
- Migration 011 persists immutable snapshots in `perception_snapshots`.
- Historical prompts read the stored perception instead of recomputing old
  events against the current clock.
- The TUI `STATE` tab renders the latest human clock, situation mode,
  confidence, and conversation gap.

## Runtime Order

```text
user event
-> legacy trace activation
-> structured appraisal
-> P1 perception
-> perception snapshot
-> stable private prompt
-> main response
-> state/relationship update
-> episode trace
```

Appraisal and main response are intentionally sequential in P1. The current
appraisal must exist before its emotional state can influence the current
response. This adds one LLM dependency to response latency but preserves a
replayable causal order.

## Boundaries

- P1 output is derived control state, not user-observed evidence.
- P1 does not read H1D or H2 shadow output.
- P1 is not the YUANZI-5D active-inference or recursive meta-operator theory.
- Fixed semantic cues and cross weights are versioned and auditable, but they
  are not a language-complete learned semantic model.
- Variational and physical free-energy claims remain outside P1.

## Verification

```text
pytest:       413 passed
coverage:     90%
ruff:         passed
strict mypy:  passed, 49 source files
wheel:        hdsc-0.5.0-py3-none-any.whl
migrations:   11 packaged SQL files
wheel doctor: schema v11, sqlite-vec loaded, status ok
```
