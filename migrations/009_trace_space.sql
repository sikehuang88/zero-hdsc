-- Migration 009: append-only SSA trace space and activation audit.

CREATE TABLE traces (
    id                  TEXT    PRIMARY KEY,
    conversation_id     TEXT    NOT NULL,
    correlation_id      TEXT    NOT NULL,
    input_event_id      TEXT    NOT NULL REFERENCES events(id),
    output_event_id     TEXT    REFERENCES events(id),
    content             TEXT    NOT NULL,
    content_type        TEXT    NOT NULL,
    source_kind         TEXT    NOT NULL,
    importance          REAL    NOT NULL CHECK(importance BETWEEN 0 AND 1),
    valence             REAL    NOT NULL CHECK(valence BETWEEN -1 AND 1),
    arousal             REAL    NOT NULL CHECK(arousal BETWEEN 0 AND 1),
    is_internal         INTEGER NOT NULL DEFAULT 0 CHECK(is_internal IN (0, 1)),
    embedding_model     TEXT    NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    vec_rowid           INTEGER NOT NULL UNIQUE,
    created_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_traces_conversation_time
    ON traces(conversation_id, created_at_ms DESC);
CREATE INDEX idx_traces_correlation ON traces(correlation_id);
CREATE INDEX idx_traces_content_type ON traces(content_type);

CREATE TABLE trace_links (
    source_trace_id     TEXT    NOT NULL REFERENCES traces(id),
    target_trace_id     TEXT    NOT NULL REFERENCES traces(id),
    link_type           TEXT    NOT NULL,
    weight              REAL    NOT NULL CHECK(weight BETWEEN 0 AND 1),
    created_at_ms       INTEGER NOT NULL,
    PRIMARY KEY(source_trace_id, target_trace_id, link_type),
    CHECK(source_trace_id <> target_trace_id)
);

CREATE INDEX idx_trace_links_source ON trace_links(source_trace_id);
CREATE INDEX idx_trace_links_target ON trace_links(target_trace_id);

CREATE TABLE trace_activations (
    id                  TEXT    PRIMARY KEY,
    conversation_id     TEXT    NOT NULL,
    correlation_id      TEXT    NOT NULL,
    query_event_id      TEXT    NOT NULL REFERENCES events(id),
    trace_id            TEXT    NOT NULL REFERENCES traces(id),
    rank                INTEGER NOT NULL,
    score               REAL    NOT NULL,
    semantic_similarity REAL    NOT NULL,
    freshness           REAL    NOT NULL,
    importance_factor   REAL    NOT NULL,
    activation_kind     TEXT    NOT NULL,
    activated_at_ms     INTEGER NOT NULL,
    UNIQUE(correlation_id, trace_id)
);

CREATE INDEX idx_trace_activations_conversation
    ON trace_activations(conversation_id, activated_at_ms DESC);
CREATE INDEX idx_trace_activations_query ON trace_activations(query_event_id);
CREATE INDEX idx_trace_activations_trace ON trace_activations(trace_id);
