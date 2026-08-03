# P0-P6 Autonomous Lifecycle Delivery

Date: 2026-07-26

## Delivered

- P0: schema v15 domain contracts and repositories for goals, jobs,
  initiatives, outbox, emotions, contact episodes, world observations,
  causal traces, and background LLM usage.
- P1: durable recurring worker with leases, expired-lease recovery,
  exponential retry, dead-letter state, and successful-run retry reset.
- P2: bounded query-free directed trace activation, half-life emotion
  episodes, and a persistent seven-step continuity goal.
- P3: evidence-bearing initiative score, quiet-hour/cooldown/daily/gap/dedup
  gates, and an atomic background drafting budget.
- P4: reliable outbox delivery, local event and Windows desktop adapters,
  TUI polling, and standalone `hdsc worker` operation.
- P5: bounded contact episodes with uncertain silence hypotheses, follow-up,
  escalation, withdrawal, and user-return resolution.
- P6: clock, file metadata, structured calendar JSON, and optional Windows
  foreground-window observations with provenance and deduplication.

## Runtime Jobs

| Job | Interval |
|---|---:|
| `agency.offline_cycle` | 30 minutes |
| `inner.heartbeat` | 1 minute (configurable) |
| `reflection.schedule` | 15 minutes (configurable) |
| `learning.consolidate` | 15 minutes (configurable) |
| `learning.evaluate_outcomes` | 1 hour (configurable) |
| `health.check` | 1 minute |
| `outbox.deliver` | 30 seconds |
| `world.observe` | 5 minutes (configurable) |
| `trace.resting_step` | 30 minutes |
| `initiative.evaluate` | 30 minutes |
| `initiative.expire` | 30 minutes |
| `state.refresh` | 1 hour |
| `goal.advance` | 1 hour |
| `contact.evaluate` | 1 hour |

## Acceptance

`tests/e2e/test_autonomous_24h_soak.py` advances a frozen clock in five-minute
steps for 24 hours and verifies unique recurring jobs, bounded activation mass,
goal progress, bounded contact termination, one event per delivered outbox row,
and restart continuity.
