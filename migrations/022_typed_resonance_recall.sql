-- Migration 022: typed trace topology and auditable resonance recall.

ALTER TABLE trace_links RENAME TO trace_links_legacy;

CREATE TABLE trace_links (
    source_trace_id     TEXT    NOT NULL REFERENCES traces(id),
    target_trace_id     TEXT    NOT NULL REFERENCES traces(id),
    link_type           TEXT    NOT NULL CHECK(link_type IN (
                                'semantic', 'temporal', 'affective', 'entity', 'causal'
                        )),
    weight              REAL    NOT NULL CHECK(weight BETWEEN 0 AND 1),
    created_at_ms       INTEGER NOT NULL,
    PRIMARY KEY(source_trace_id, target_trace_id, link_type),
    CHECK(source_trace_id <> target_trace_id)
);

INSERT OR REPLACE INTO trace_links (
    source_trace_id, target_trace_id, link_type, weight, created_at_ms
)
SELECT source_trace_id,
       target_trace_id,
       CASE WHEN link_type = 'temporal-forward' THEN 'temporal' ELSE link_type END,
       weight,
       created_at_ms
FROM trace_links_legacy
WHERE link_type IN (
    'semantic', 'temporal', 'temporal-forward', 'affective', 'entity', 'causal'
);

DROP TABLE trace_links_legacy;

CREATE INDEX idx_trace_links_source ON trace_links(source_trace_id);
CREATE INDEX idx_trace_links_target ON trace_links(target_trace_id);
CREATE INDEX idx_trace_links_type ON trace_links(link_type, source_trace_id);

CREATE TABLE resonance_recall_audits (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    query_digest            TEXT    NOT NULL,
    situation_mode          TEXT    NOT NULL,
    hop_budget              INTEGER NOT NULL CHECK(hop_budget >= 1),
    edge_gains_json         TEXT    NOT NULL,
    candidate_audits_json   TEXT    NOT NULL,
    selected_trace_ids_json TEXT    NOT NULL,
    background_median       REAL    NOT NULL CHECK(background_median >= 0),
    emergence_ratio         REAL    NOT NULL CHECK(emergence_ratio > 0),
    null_mass               REAL    NOT NULL CHECK(null_mass BETWEEN 0 AND 1),
    conservation_residual   REAL    NOT NULL,
    emerged                 INTEGER NOT NULL CHECK(emerged IN (0, 1)),
    created_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_resonance_recall_conversation
ON resonance_recall_audits(conversation_id, created_at_ms DESC);
