-- Migration 003: State and relationship snapshots + appraisals

CREATE TABLE state_snapshots (
    id              TEXT    PRIMARY KEY,
    previous_id     TEXT    REFERENCES state_snapshots(id),
    cause_event_id  TEXT    REFERENCES events(id),
    version         INTEGER NOT NULL UNIQUE,
    state_json      TEXT    NOT NULL,
    created_at_ms   INTEGER NOT NULL
);

CREATE INDEX idx_state_version ON state_snapshots(version);

CREATE TABLE relationship_snapshots (
    id                  TEXT    PRIMARY KEY,
    previous_id         TEXT    REFERENCES relationship_snapshots(id),
    cause_event_id      TEXT    REFERENCES events(id),
    version             INTEGER NOT NULL UNIQUE,
    relationship_json   TEXT    NOT NULL,
    created_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_relationship_version ON relationship_snapshots(version);

CREATE TABLE appraisals (
    id              TEXT    PRIMARY KEY,
    correlation_id  TEXT    NOT NULL,
    cause_event_id  TEXT    NOT NULL REFERENCES events(id),
    result_json     TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    prompt_version  TEXT    NOT NULL,
    created_at_ms   INTEGER NOT NULL
);

CREATE INDEX idx_appraisals_correlation ON appraisals(correlation_id);
CREATE INDEX idx_appraisals_event ON appraisals(cause_event_id);
