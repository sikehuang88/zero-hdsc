-- Migration 005: Audit, jobs, outbox, initiatives

-- LLM call audit log (pipeline §8.2)
CREATE TABLE llm_calls (
    id              TEXT    PRIMARY KEY,
    correlation_id  TEXT    NOT NULL,
    purpose         TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    prompt_version  TEXT    NOT NULL,
    prompt_hash     TEXT,
    response_hash   TEXT,
    temperature     REAL,
    seed            INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    status          TEXT    NOT NULL,
    error_code      TEXT,
    created_at_ms   INTEGER NOT NULL
);

CREATE INDEX idx_llm_calls_correlation ON llm_calls(correlation_id);
CREATE INDEX idx_llm_calls_purpose ON llm_calls(purpose);
CREATE INDEX idx_llm_calls_status ON llm_calls(status);

-- Causal trace: full event -> response linkage (pipeline §8.2, §26.3)
CREATE TABLE causal_traces (
    id                      TEXT    PRIMARY KEY,
    correlation_id          TEXT    NOT NULL,
    input_event_id          TEXT    NOT NULL REFERENCES events(id),
    appraisal_id            TEXT    REFERENCES appraisals(id),
    old_state_id            TEXT    REFERENCES state_snapshots(id),
    new_state_id            TEXT    REFERENCES state_snapshots(id),
    old_relationship_id     TEXT    REFERENCES relationship_snapshots(id),
    new_relationship_id     TEXT    REFERENCES relationship_snapshots(id),
    memory_ids_json         TEXT    NOT NULL DEFAULT '[]',
    goal_ids_json           TEXT    NOT NULL DEFAULT '[]',
    decision_json           TEXT    NOT NULL DEFAULT '{}',
    output_event_id         TEXT    REFERENCES events(id),
    created_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_causal_traces_correlation ON causal_traces(correlation_id);
CREATE INDEX idx_causal_traces_input ON causal_traces(input_event_id);

-- Scheduled background jobs (pipeline §8.2, §24)
CREATE TABLE scheduled_jobs (
    id              TEXT    PRIMARY KEY,
    job_type        TEXT    NOT NULL,
    dedup_key       TEXT,
    payload_json    TEXT    NOT NULL DEFAULT '{}',
    due_at_ms       INTEGER NOT NULL,
    attempt_count   INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'pending',
    last_error      TEXT,
    leased_until_ms INTEGER,
    completed_at_ms INTEGER,
    created_at_ms   INTEGER NOT NULL
);

CREATE INDEX idx_jobs_due ON scheduled_jobs(due_at_ms, status);
CREATE INDEX idx_jobs_dedup ON scheduled_jobs(dedup_key);
CREATE INDEX idx_jobs_type ON scheduled_jobs(job_type);

-- Initiatives: planned proactive messages (pipeline §8.2, §24.4)
CREATE TABLE initiatives (
    id                  TEXT    PRIMARY KEY,
    motive              TEXT    NOT NULL,
    intent              TEXT    NOT NULL,
    content_draft       TEXT    NOT NULL,
    urgency             REAL    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'candidate',
    earliest_send_at_ms INTEGER NOT NULL,
    expires_at_ms       INTEGER,
    sent_event_id       TEXT    REFERENCES events(id),
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_initiatives_status ON initiatives(status);
CREATE INDEX idx_initiatives_earliest ON initiatives(earliest_send_at_ms);

-- Outbox: messages awaiting delivery (pipeline §8.2, §7.2 step 19-20)
CREATE TABLE outbox (
    id                  TEXT    PRIMARY KEY,
    correlation_id      TEXT    NOT NULL,
    channel             TEXT    NOT NULL,
    recipient           TEXT    NOT NULL,
    payload_json        TEXT    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'pending',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    next_attempt_at_ms  INTEGER,
    delivered_at_ms     INTEGER,
    created_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_outbox_status ON outbox(status);
CREATE INDEX idx_outbox_next_attempt ON outbox(next_attempt_at_ms);
CREATE INDEX idx_outbox_correlation ON outbox(correlation_id);
