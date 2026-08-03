-- Migration 015: evidence-gated reflective learning and outcome evaluation

CREATE TABLE reflection_runs (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    correlation_id          TEXT    NOT NULL,
    dedup_key               TEXT    NOT NULL UNIQUE,
    trigger_kind            TEXT    NOT NULL,
    priority_score          REAL    NOT NULL CHECK(priority_score BETWEEN 0 AND 1),
    score_components_json   TEXT    NOT NULL DEFAULT '{}',
    status                  TEXT    NOT NULL DEFAULT 'planned',
    source_episode_id       TEXT    REFERENCES offline_episodes(id),
    source_artifact_id      TEXT    REFERENCES offline_artifacts(id),
    source_event_ids_json   TEXT    NOT NULL DEFAULT '[]',
    source_trace_ids_json   TEXT    NOT NULL DEFAULT '[]',
    reflection_text         TEXT,
    critique_text           TEXT,
    critic_score            REAL    CHECK(critic_score BETWEEN 0 AND 1),
    model                   TEXT,
    prompt_version          TEXT,
    error                   TEXT,
    started_at_ms           INTEGER NOT NULL,
    completed_at_ms         INTEGER
);

CREATE INDEX idx_reflection_run_status
ON reflection_runs(conversation_id, status, started_at_ms DESC);

CREATE TABLE learning_proposals (
    id                      TEXT    PRIMARY KEY,
    run_id                  TEXT    NOT NULL REFERENCES reflection_runs(id),
    conversation_id         TEXT    NOT NULL,
    dedup_key               TEXT    NOT NULL UNIQUE,
    proposal_type           TEXT    NOT NULL,
    status                  TEXT    NOT NULL DEFAULT 'candidate',
    title                   TEXT    NOT NULL,
    content                 TEXT    NOT NULL,
    payload_json            TEXT    NOT NULL DEFAULT '{}',
    rationale               TEXT    NOT NULL,
    confidence              REAL    NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    critic_score            REAL    NOT NULL CHECK(critic_score BETWEEN 0 AND 1),
    target_kind             TEXT,
    target_id               TEXT,
    rollback_json           TEXT    NOT NULL DEFAULT '{}',
    version                 INTEGER NOT NULL DEFAULT 1,
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    applied_at_ms           INTEGER
);

CREATE INDEX idx_learning_proposal_run
ON learning_proposals(run_id, created_at_ms);

CREATE INDEX idx_learning_proposal_status
ON learning_proposals(conversation_id, status, updated_at_ms DESC);

CREATE TABLE learning_proposal_evidence (
    proposal_id             TEXT    NOT NULL REFERENCES learning_proposals(id),
    reference_kind          TEXT    NOT NULL,
    reference_id            TEXT    NOT NULL,
    relation                TEXT    NOT NULL,
    provenance_kind         TEXT    NOT NULL,
    trust_weight            REAL    NOT NULL CHECK(trust_weight BETWEEN 0 AND 1),
    likelihood_ratio        REAL    NOT NULL CHECK(likelihood_ratio > 0),
    content_hash            TEXT    NOT NULL,
    excerpt                 TEXT,
    recorded_at_ms          INTEGER NOT NULL,
    PRIMARY KEY (proposal_id, reference_kind, reference_id)
);

CREATE INDEX idx_learning_evidence_reference
ON learning_proposal_evidence(reference_kind, reference_id);

CREATE TABLE consolidation_decisions (
    id                      TEXT    PRIMARY KEY,
    proposal_id             TEXT    NOT NULL REFERENCES learning_proposals(id),
    decision_version        INTEGER NOT NULL,
    decision                TEXT    NOT NULL,
    gate_score              REAL    NOT NULL CHECK(gate_score BETWEEN 0 AND 1),
    criteria_json           TEXT    NOT NULL DEFAULT '{}',
    reason                  TEXT    NOT NULL,
    rules_version           TEXT    NOT NULL,
    model                   TEXT,
    prompt_version          TEXT,
    event_id                TEXT    REFERENCES events(id),
    created_at_ms           INTEGER NOT NULL,
    UNIQUE (proposal_id, decision_version)
);

CREATE INDEX idx_consolidation_decision_proposal
ON consolidation_decisions(proposal_id, created_at_ms);

CREATE TABLE behavior_experiments (
    id                      TEXT    PRIMARY KEY,
    conversation_id         TEXT    NOT NULL,
    proposal_id             TEXT    NOT NULL REFERENCES learning_proposals(id),
    dedup_key               TEXT    NOT NULL UNIQUE,
    status                  TEXT    NOT NULL DEFAULT 'planned',
    hypothesis              TEXT    NOT NULL,
    policy_key              TEXT    NOT NULL,
    baseline_json           TEXT    NOT NULL DEFAULT '{}',
    treatment_json          TEXT    NOT NULL DEFAULT '{}',
    rollback_json           TEXT    NOT NULL DEFAULT '{}',
    success_criteria_json   TEXT    NOT NULL DEFAULT '{}',
    min_observations        INTEGER NOT NULL CHECK(min_observations >= 1),
    result_score            REAL    CHECK(result_score BETWEEN -1 AND 1),
    conclusion              TEXT,
    started_at_ms           INTEGER,
    due_at_ms               INTEGER NOT NULL,
    completed_at_ms         INTEGER,
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_behavior_experiment_due
ON behavior_experiments(conversation_id, status, due_at_ms);

CREATE TABLE outcome_observations (
    id                      TEXT    PRIMARY KEY,
    experiment_id           TEXT    NOT NULL REFERENCES behavior_experiments(id),
    dedup_key               TEXT    NOT NULL UNIQUE,
    source_event_id         TEXT    REFERENCES events(id),
    observation_kind        TEXT    NOT NULL,
    payload_json            TEXT    NOT NULL DEFAULT '{}',
    normalized_score        REAL    NOT NULL CHECK(normalized_score BETWEEN -1 AND 1),
    confidence              REAL    NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    prediction_error        REAL    NOT NULL CHECK(prediction_error BETWEEN 0 AND 1),
    observed_at_ms          INTEGER NOT NULL,
    created_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_outcome_observation_experiment
ON outcome_observations(experiment_id, observed_at_ms);
