-- Migration 012: executable lifecycle, initiative, contact, and world kernel

CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    title TEXT NOT NULL,
    motive TEXT NOT NULL,
    success_criteria TEXT NOT NULL,
    priority REAL NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'proposed',
    due_at_ms INTEGER,
    next_action_at_ms INTEGER,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS goal_steps (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES goals(id),
    action TEXT NOT NULL,
    result TEXT,
    source_event_id TEXT REFERENCES events(id),
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    dedup_key TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    due_at_ms INTEGER NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    last_error TEXT,
    leased_until_ms INTEGER,
    completed_at_ms INTEGER,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS initiatives (
    id TEXT PRIMARY KEY,
    motive TEXT NOT NULL,
    intent TEXT NOT NULL,
    content_draft TEXT NOT NULL,
    urgency REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    earliest_send_at_ms INTEGER NOT NULL,
    expires_at_ms INTEGER,
    sent_event_id TEXT REFERENCES events(id),
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS outbox (
    id TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at_ms INTEGER,
    delivered_at_ms INTEGER,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS causal_traces (
    id TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    input_event_id TEXT NOT NULL REFERENCES events(id),
    appraisal_id TEXT,
    old_state_id TEXT,
    new_state_id TEXT,
    old_relationship_id TEXT,
    new_relationship_id TEXT,
    memory_ids_json TEXT NOT NULL DEFAULT '[]',
    goal_ids_json TEXT NOT NULL DEFAULT '[]',
    decision_json TEXT NOT NULL DEFAULT '{}',
    output_event_id TEXT REFERENCES events(id),
    created_at_ms INTEGER NOT NULL
);

ALTER TABLE goals ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'cli-primary';
ALTER TABLE goals ADD COLUMN correlation_id TEXT;
ALTER TABLE goals ADD COLUMN blocked_reason TEXT;

ALTER TABLE goal_steps ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE scheduled_jobs ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'cli-primary';
ALTER TABLE scheduled_jobs ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 5;
ALTER TABLE scheduled_jobs ADD COLUMN updated_at_ms INTEGER NOT NULL DEFAULT 0;

CREATE UNIQUE INDEX idx_jobs_dedup_unique
ON scheduled_jobs(dedup_key)
WHERE dedup_key IS NOT NULL;

ALTER TABLE initiatives ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'cli-primary';
ALTER TABLE initiatives ADD COLUMN correlation_id TEXT;
ALTER TABLE initiatives ADD COLUMN goal_id TEXT REFERENCES goals(id);
ALTER TABLE initiatives ADD COLUMN source_event_ids_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE initiatives ADD COLUMN source_trace_ids_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE initiatives ADD COLUMN decision_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE initiatives ADD COLUMN decision_score REAL NOT NULL DEFAULT 0.0;
ALTER TABLE initiatives ADD COLUMN channel TEXT NOT NULL DEFAULT 'local';
ALTER TABLE initiatives ADD COLUMN dedup_key TEXT;

CREATE INDEX idx_initiatives_conversation ON initiatives(conversation_id);
CREATE UNIQUE INDEX idx_initiatives_dedup_unique
ON initiatives(dedup_key)
WHERE dedup_key IS NOT NULL;

ALTER TABLE outbox ADD COLUMN initiative_id TEXT REFERENCES initiatives(id);
ALTER TABLE outbox ADD COLUMN dedup_key TEXT;
ALTER TABLE outbox ADD COLUMN delivered_event_id TEXT REFERENCES events(id);
ALTER TABLE outbox ADD COLUMN last_error TEXT;

CREATE UNIQUE INDEX idx_outbox_dedup_unique
ON outbox(dedup_key)
WHERE dedup_key IS NOT NULL;

CREATE TABLE emotion_episodes (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    correlation_id          TEXT,
    emotion_type            TEXT    NOT NULL,
    target                  TEXT    NOT NULL,
    action_tendency         TEXT    NOT NULL,
    intensity               REAL    NOT NULL CHECK(intensity BETWEEN 0 AND 1),
    inhibition              REAL    NOT NULL CHECK(inhibition BETWEEN 0 AND 1),
    half_life_minutes       REAL    NOT NULL,
    source_event_ids_json   TEXT    NOT NULL DEFAULT '[]',
    source_trace_ids_json   TEXT    NOT NULL DEFAULT '[]',
    resolution_condition    TEXT,
    status                  TEXT    NOT NULL DEFAULT 'active',
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    resolved_at_ms          INTEGER
);

CREATE INDEX idx_emotion_active
ON emotion_episodes(conversation_id, status, updated_at_ms);

CREATE TABLE contact_episodes (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    correlation_id          TEXT    NOT NULL,
    motive                  TEXT    NOT NULL,
    topic                   TEXT    NOT NULL,
    phase                   TEXT    NOT NULL DEFAULT 'opening',
    status                  TEXT    NOT NULL DEFAULT 'active',
    message_count           INTEGER NOT NULL DEFAULT 0,
    max_messages            INTEGER NOT NULL DEFAULT 3,
    hypothesis_json         TEXT    NOT NULL DEFAULT '{}',
    source_initiative_id    TEXT    REFERENCES initiatives(id),
    last_sent_at_ms         INTEGER,
    next_action_at_ms       INTEGER,
    resolved_by_event_id    TEXT    REFERENCES events(id),
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_contact_active
ON contact_episodes(conversation_id, status, next_action_at_ms);

CREATE TABLE world_observations (
    id                  TEXT    PRIMARY KEY,
    conversation_id     TEXT    NOT NULL,
    observation_type    TEXT    NOT NULL,
    source              TEXT    NOT NULL,
    summary             TEXT    NOT NULL,
    payload_json        TEXT    NOT NULL DEFAULT '{}',
    salience            REAL    NOT NULL CHECK(salience BETWEEN 0 AND 1),
    dedup_key           TEXT    NOT NULL UNIQUE,
    observed_at_ms      INTEGER NOT NULL,
    event_id            TEXT    REFERENCES events(id)
);

CREATE INDEX idx_world_observation_time
ON world_observations(conversation_id, observed_at_ms);

CREATE TABLE background_usage (
    usage_day       TEXT    PRIMARY KEY,
    llm_calls       INTEGER NOT NULL DEFAULT 0,
    updated_at_ms   INTEGER NOT NULL
);
