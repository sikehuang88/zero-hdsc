-- Migration 007: append-only self-belief versions and normalized evidence.
-- Rollback: restore from the pre-migration database backup. SQLite cannot
-- remove these columns in place without rebuilding self_beliefs.

ALTER TABLE self_beliefs ADD COLUMN lineage_id TEXT;
ALTER TABLE self_beliefs ADD COLUMN previous_id TEXT REFERENCES self_beliefs(id);
ALTER TABLE self_beliefs ADD COLUMN cause_event_id TEXT REFERENCES events(id);
ALTER TABLE self_beliefs ADD COLUMN change_reason TEXT NOT NULL DEFAULT 'legacy migration';
ALTER TABLE self_beliefs ADD COLUMN model TEXT;
ALTER TABLE self_beliefs ADD COLUMN prompt_version TEXT;
ALTER TABLE self_beliefs ADD COLUMN candidate_since_ms INTEGER;
ALTER TABLE self_beliefs ADD COLUMN activated_at_ms INTEGER;

UPDATE self_beliefs
SET lineage_id = id,
    change_reason = 'legacy migration from version ' || version,
    version = 1,
    candidate_since_ms = created_at_ms,
    activated_at_ms = CASE
        WHEN status IN ('active', 'challenged', 'revised') THEN created_at_ms
        ELSE NULL
    END
WHERE lineage_id IS NULL OR candidate_since_ms IS NULL;

CREATE UNIQUE INDEX idx_self_beliefs_lineage_version
    ON self_beliefs(lineage_id, version);
CREATE INDEX idx_self_beliefs_lineage
    ON self_beliefs(lineage_id, version DESC);
CREATE INDEX idx_self_beliefs_previous
    ON self_beliefs(previous_id);
CREATE INDEX idx_self_beliefs_cause_event
    ON self_beliefs(cause_event_id);

CREATE TABLE self_belief_evidence (
    belief_version_id   TEXT    NOT NULL REFERENCES self_beliefs(id) ON DELETE CASCADE,
    event_id            TEXT    NOT NULL REFERENCES events(id),
    relation            TEXT    NOT NULL CHECK(relation IN ('supports', 'contradicts')),
    recorded_at_ms      INTEGER NOT NULL,
    PRIMARY KEY(belief_version_id, event_id, relation)
);

INSERT OR IGNORE INTO self_belief_evidence (
    belief_version_id, event_id, relation, recorded_at_ms
)
SELECT belief.id, CAST(evidence.value AS TEXT), 'supports', belief.updated_at_ms
FROM self_beliefs AS belief
CROSS JOIN json_each(belief.evidence_json) AS evidence
JOIN events AS source_event
  ON source_event.id = CAST(evidence.value AS TEXT);

INSERT OR IGNORE INTO self_belief_evidence (
    belief_version_id, event_id, relation, recorded_at_ms
)
SELECT belief.id, CAST(evidence.value AS TEXT), 'contradicts', belief.updated_at_ms
FROM self_beliefs AS belief
CROSS JOIN json_each(belief.counterevidence_json) AS evidence
JOIN events AS source_event
  ON source_event.id = CAST(evidence.value AS TEXT);

CREATE INDEX idx_self_belief_evidence_event
    ON self_belief_evidence(event_id, relation);
