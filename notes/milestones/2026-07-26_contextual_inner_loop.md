# Contextual Inner Loop Delivery

Date: 2026-07-26

## Delivered

- Schema v14 `inner_loop_states` persistence.
- Context-derived reply expectation and bounded wait horizon.
- Structured latent recurrence for concern, curiosity, connection pressure,
  uncertainty, and offline readiness.
- `engaged / waiting / offline / quiet_rest` transition gate with hysteresis.
- One-minute `inner.heartbeat` with transition-only lifecycle events.
- Shared gating for offline agency, initiative evaluation, and contact follow-up.
- Private prompt context and `ACTIVITY` diagnostic telemetry without free-form
  hidden monologue generation.

## Runtime Truth

The heartbeat proves that the process was running at that timestamp. It does
not prove that the user was present. No heartbeat or action is backfilled for a
period when both the TUI and standalone worker were stopped.
