-- Migration 004: Identity and goals

CREATE TABLE self_beliefs (
    id                      TEXT    PRIMARY KEY,
    claim                   TEXT    NOT NULL,
    confidence              REAL    NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    status                  TEXT    NOT NULL DEFAULT 'candidate',
    version                 INTEGER NOT NULL,
    evidence_json           TEXT    NOT NULL DEFAULT '[]',
    counterevidence_json    TEXT    NOT NULL DEFAULT '[]',
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL
);

CREATE INDEX idx_self_beliefs_status ON self_beliefs(status);

CREATE TABLE goals (
    id                  TEXT    PRIMARY KEY,
    owner               TEXT    NOT NULL,
    title               TEXT    NOT NULL,
    motive              TEXT    NOT NULL,
    success_criteria    TEXT    NOT NULL,  -- JSON array
    priority            REAL    NOT NULL,
    progress            REAL    NOT NULL DEFAULT 0.0,
    status              TEXT    NOT NULL DEFAULT 'proposed',
    due_at_ms           INTEGER,
    next_action_at_ms   INTEGER,
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_goals_status ON goals(status);
CREATE INDEX idx_goals_owner ON goals(owner);
CREATE INDEX idx_goals_next_action ON goals(next_action_at_ms);

CREATE TABLE goal_steps (
    id              TEXT    PRIMARY KEY,
    goal_id         TEXT    NOT NULL REFERENCES goals(id),
    action          TEXT    NOT NULL,
    result          TEXT,
    source_event_id TEXT    REFERENCES events(id),
    created_at_ms   INTEGER NOT NULL
);

CREATE INDEX idx_goal_steps_goal ON goal_steps(goal_id);
