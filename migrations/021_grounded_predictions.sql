-- Migration 021: machine-verifiable predictions and calibration metrics

CREATE TABLE predictions (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    dedup_key               TEXT    NOT NULL UNIQUE,
    claim_text              TEXT    NOT NULL,
    claim_kind              TEXT    NOT NULL CHECK(claim_kind IN (
                                    'user_behavior', 'preference', 'schedule', 'world_fact'
                            )),
    verifier_kind           TEXT    NOT NULL CHECK(verifier_kind IN (
                                    'event_match', 'external_truth', 'tool_result'
                            )),
    verifier_spec_json      TEXT    NOT NULL,
    stated_confidence       REAL    NOT NULL CHECK(stated_confidence BETWEEN 0 AND 1),
    base_rate_prior         REAL    NOT NULL CHECK(base_rate_prior BETWEEN 0 AND 1),
    source_event_ids_json   TEXT    NOT NULL DEFAULT '[]',
    source_trace_ids_json   TEXT    NOT NULL DEFAULT '[]',
    created_at_ms           INTEGER NOT NULL,
    resolve_after_ms        INTEGER NOT NULL,
    expires_at_ms           INTEGER NOT NULL,
    status                  TEXT    NOT NULL DEFAULT 'pending' CHECK(status IN (
                                    'pending', 'resolved_true', 'resolved_false',
                                    'expired', 'unverifiable'
                            )),
    resolved_at_ms          INTEGER,
    resolved_by_event_id    TEXT    REFERENCES events(id),
    brier_contribution      REAL    CHECK(brier_contribution BETWEEN 0 AND 1),
    CHECK(expires_at_ms > resolve_after_ms)
);

CREATE INDEX idx_prediction_due
ON predictions(status, resolve_after_ms);

CREATE INDEX idx_prediction_calibration
ON predictions(conversation_id, status, stated_confidence);
