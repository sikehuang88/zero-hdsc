-- Dedicated long-term emotional memory library.
-- These records are subjective, model-derived continuity, not user facts.

CREATE TABLE emotional_memories (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    correlation_id          TEXT    NOT NULL,
    origin_trace_id         TEXT    NOT NULL UNIQUE REFERENCES traces(id),
    emotion_type            TEXT    NOT NULL CHECK(emotion_type IN (
        'attachment', 'curiosity', 'concern', 'anticipation',
        'frustration', 'contentment', 'ambivalence'
    )),
    target                  TEXT    NOT NULL,
    trigger_summary         TEXT    NOT NULL,
    felt_summary            TEXT    NOT NULL,
    valence                 REAL    NOT NULL CHECK(valence BETWEEN -1 AND 1),
    arousal                 REAL    NOT NULL CHECK(arousal BETWEEN 0 AND 1),
    intensity               REAL    NOT NULL CHECK(intensity BETWEEN 0 AND 1),
    action_tendency         TEXT    NOT NULL,
    source_event_ids_json   TEXT    NOT NULL DEFAULT '[]',
    source_trace_ids_json   TEXT    NOT NULL DEFAULT '[]',
    status                  TEXT    NOT NULL DEFAULT 'active'
                            CHECK(status IN ('active', 'settled', 'archived')),
    recall_count            INTEGER NOT NULL DEFAULT 0 CHECK(recall_count >= 0),
    last_recalled_at_ms     INTEGER,
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_emotional_memories_recall
ON emotional_memories(conversation_id, status, intensity DESC, updated_at_ms DESC);

CREATE INDEX idx_emotional_memories_target
ON emotional_memories(conversation_id, target, updated_at_ms DESC);
