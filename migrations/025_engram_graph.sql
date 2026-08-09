-- Migration 025: append-only typed directed ENGRAM graph substrate.

CREATE TABLE engram_nodes (
    node_id         TEXT PRIMARY KEY,
    node_type       TEXT NOT NULL CHECK(node_type IN ('trace', 'entity', 'category', 'community')),
    conversation_id TEXT NOT NULL,
    created_at_ms   INTEGER NOT NULL CHECK(created_at_ms >= 0)
);

CREATE TABLE engram_edges (
    edge_id          TEXT PRIMARY KEY,
    src              TEXT NOT NULL REFERENCES engram_nodes(node_id),
    dst              TEXT NOT NULL REFERENCES engram_nodes(node_id),
    rel_type         TEXT NOT NULL CHECK(rel_type IN (
                         'semantic', 'entity', 'category', 'spatial', 'episodic',
                         'preference', 'evidence', 'revision', 'temporal_forward'
                     )),
    support          REAL NOT NULL CHECK(support > 0),
    action           TEXT NOT NULL,
    valid_from_ms    INTEGER NOT NULL CHECK(valid_from_ms >= 0),
    valid_to_ms      INTEGER CHECK(valid_to_ms IS NULL OR valid_to_ms >= valid_from_ms),
    source_event_id  TEXT,
    trust_weight     REAL NOT NULL CHECK(trust_weight BETWEEN 0 AND 1),
    likelihood_ratio REAL NOT NULL CHECK(likelihood_ratio > 0),
    UNIQUE(src, dst, rel_type, action, source_event_id)
);

CREATE INDEX idx_engram_nodes_conversation
    ON engram_nodes(conversation_id, created_at_ms);
CREATE INDEX idx_engram_edges_src_active
    ON engram_edges(src, rel_type, valid_to_ms);
CREATE INDEX idx_engram_edges_dst_active
    ON engram_edges(dst, rel_type, valid_to_ms);
CREATE INDEX idx_engram_edges_event
    ON engram_edges(source_event_id);
