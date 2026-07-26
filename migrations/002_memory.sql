-- Migration 002: Memory schema
-- Memories, their evidence links to events, and inter-memory links.
-- The vector table (vec0) is created here but may be loaded lazily
-- depending on sqlite-vec availability.

CREATE TABLE memories (
    id                  TEXT    PRIMARY KEY,
    memory_type         TEXT    NOT NULL,
    source_kind         TEXT    NOT NULL,
    content             TEXT    NOT NULL,
    summary             TEXT,
    confidence          REAL    NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    importance          REAL    NOT NULL CHECK(importance BETWEEN 0 AND 1),
    valence             REAL    NOT NULL CHECK(valence BETWEEN -1 AND 1),
    arousal             REAL    NOT NULL CHECK(arousal BETWEEN 0 AND 1),
    embedding_model     TEXT    NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    content_hash        TEXT    NOT NULL,
    derived_by_model    TEXT,
    prompt_version      TEXT,
    status              TEXT    NOT NULL DEFAULT 'active',
    access_count        INTEGER NOT NULL DEFAULT 0,
    last_accessed_at_ms INTEGER,
    vec_rowid           INTEGER,
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_source ON memories(source_kind);
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_importance ON memories(importance);
CREATE INDEX idx_memories_created_at ON memories(created_at_ms);
CREATE INDEX idx_memories_content_hash ON memories(content_hash);
CREATE INDEX idx_memories_vec_rowid ON memories(vec_rowid);

CREATE TABLE memory_evidence (
    memory_id   TEXT    NOT NULL REFERENCES memories(id),
    event_id    TEXT    NOT NULL REFERENCES events(id),
    relation    TEXT    NOT NULL,
    PRIMARY KEY(memory_id, event_id, relation)
);

CREATE TABLE memory_links (
    source_memory_id    TEXT    NOT NULL REFERENCES memories(id),
    target_memory_id    TEXT    NOT NULL REFERENCES memories(id),
    link_type           TEXT    NOT NULL,
    weight              REAL    NOT NULL CHECK(weight BETWEEN 0 AND 1),
    created_at_ms       INTEGER NOT NULL,
    PRIMARY KEY(source_memory_id, target_memory_id, link_type)
);

CREATE INDEX idx_memory_links_source ON memory_links(source_memory_id);
CREATE INDEX idx_memory_links_target ON memory_links(target_memory_id);

-- The vec0 virtual table for memory embeddings.
-- NOTE: requires the sqlite-vec extension to be loaded.
-- The dimension is set to 512 to match bge-small-zh-v1.5 (pipeline §2.3).
-- Tests that use a different dimension must create their own vec table.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    embedding float[512]
);
