-- Migration 011: immutable environment-perception snapshots.

CREATE TABLE perception_snapshots (
    id                  TEXT    PRIMARY KEY,
    correlation_id      TEXT    NOT NULL,
    conversation_id     TEXT    NOT NULL,
    query_event_id      TEXT    NOT NULL UNIQUE REFERENCES events(id),
    model_id            TEXT    NOT NULL,
    operator_version    TEXT    NOT NULL,
    result_json         TEXT    NOT NULL,
    created_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_perception_conversation_time
    ON perception_snapshots(conversation_id, created_at_ms DESC);
CREATE INDEX idx_perception_correlation
    ON perception_snapshots(correlation_id);

