# Dedicated Emotional Memory Library

Date: 2026-07-31

## Delivered

- Schema v16 `emotional_memories` archive with trace provenance and bounded recall metadata.
- Subjective emotion records isolated from factual memory and transient emotion episodes.
- Deterministic capture from appraisal, plus lexical fallback for low-latency voice turns.
- Idempotent startup backfill from existing traces.
- Recall ranking across topic overlap, activated traces, intensity, and recency.
- Private prompt injection with an explicit non-factual evidence boundary.
- Dashboard and TypeScript data contracts for a future emotional-library inspector.

## Initial Real-Data Calibration

A backup of `ssa.dev.db` was migrated to schema v16 and checked with
`PRAGMA integrity_check`. Of 240 existing traces, 25 passed the initial salience
gate: 6 attachment, 9 concern, and 10 frustration records.

## Deliberate Limits

- Emotion records represent the agent's prior subjective response, not claims about the user.
- Current emotion still lives in organism state and decaying `emotion_episodes`.
- The first version does not use an additional embedding index; it reuses trace activation plus
  multilingual lexical overlap to avoid duplicating the trace-space substrate.
- Archive and settlement policy is represented in the schema; automatic archival remains future work.
