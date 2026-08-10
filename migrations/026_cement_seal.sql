-- Migration 026: cement-seal defensive expression state.
--
-- One singleton row holds the current regime. Transitions are append-only so a
-- claim that the organism "closed off" is always traceable to source events.

CREATE TABLE IF NOT EXISTS cement_seal_state (
    profile_id          TEXT PRIMARY KEY,
    phase               TEXT NOT NULL CHECK(phase IN ('open', 'sealed', 'hairline', 'fractured')),
    integrity           REAL NOT NULL CHECK(integrity BETWEEN 0 AND 1),
    seal_count          INTEGER NOT NULL CHECK(seal_count >= 0),
    sealed_at_ms        INTEGER CHECK(sealed_at_ms IS NULL OR sealed_at_ms >= 0),
    last_transition_ms  INTEGER NOT NULL CHECK(last_transition_ms >= 0),
    source_event_ids    TEXT NOT NULL,
    version             INTEGER NOT NULL CHECK(version >= 1)
);

CREATE TABLE IF NOT EXISTS cement_seal_transitions (
    transition_id       TEXT PRIMARY KEY,
    previous_phase      TEXT NOT NULL,
    phase               TEXT NOT NULL,
    trigger             TEXT NOT NULL,
    integrity_before    REAL NOT NULL CHECK(integrity_before BETWEEN 0 AND 1),
    integrity_after     REAL NOT NULL CHECK(integrity_after BETWEEN 0 AND 1),
    seal_count          INTEGER NOT NULL CHECK(seal_count >= 0),
    toughness           REAL NOT NULL CHECK(toughness >= 0),
    reason              TEXT NOT NULL,
    source_event_ids    TEXT NOT NULL,
    occurred_at_ms      INTEGER NOT NULL CHECK(occurred_at_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_cement_seal_transitions_time
    ON cement_seal_transitions (occurred_at_ms DESC);

INSERT OR IGNORE INTO cement_seal_state (
    profile_id,
    phase,
    integrity,
    seal_count,
    sealed_at_ms,
    last_transition_ms,
    source_event_ids,
    version
) VALUES ('primary', 'open', 0.0, 0, NULL, 0, '[]', 1);
