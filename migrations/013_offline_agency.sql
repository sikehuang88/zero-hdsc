-- Migration 013: auditable offline agency episodes and learning artifacts

CREATE TABLE offline_episodes (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    correlation_id          TEXT    NOT NULL,
    dedup_key               TEXT    NOT NULL UNIQUE,
    action_kind             TEXT    NOT NULL,
    motive                  TEXT    NOT NULL,
    status                  TEXT    NOT NULL DEFAULT 'planned',
    goal_id                 TEXT    REFERENCES goals(id),
    source_event_ids_json   TEXT    NOT NULL DEFAULT '[]',
    source_trace_ids_json   TEXT    NOT NULL DEFAULT '[]',
    provider                TEXT,
    tool_name               TEXT,
    tool_output_hash        TEXT,
    summary                 TEXT,
    error                   TEXT,
    event_id                TEXT    REFERENCES events(id),
    started_at_ms           INTEGER NOT NULL,
    completed_at_ms         INTEGER
);

CREATE INDEX idx_offline_episode_time
ON offline_episodes(conversation_id, started_at_ms DESC);

CREATE INDEX idx_offline_episode_status
ON offline_episodes(conversation_id, status, started_at_ms DESC);

CREATE TABLE offline_artifacts (
    id                      TEXT    PRIMARY KEY,
    episode_id              TEXT    NOT NULL REFERENCES offline_episodes(id),
    artifact_type           TEXT    NOT NULL,
    title                   TEXT    NOT NULL,
    content                 TEXT    NOT NULL,
    evidence_event_ids_json TEXT    NOT NULL DEFAULT '[]',
    evidence_trace_ids_json TEXT    NOT NULL DEFAULT '[]',
    metadata_json           TEXT    NOT NULL DEFAULT '{}',
    created_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_offline_artifact_episode
ON offline_artifacts(episode_id, created_at_ms);
