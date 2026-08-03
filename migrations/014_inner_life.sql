-- Migration 014: contextual waiting and persistent inner-loop heartbeat state

CREATE TABLE inner_loop_states (
    conversation_id          TEXT    PRIMARY KEY,
    version                  INTEGER NOT NULL,
    mode                     TEXT    NOT NULL,
    reply_expectation        REAL    NOT NULL CHECK(reply_expectation BETWEEN 0 AND 1),
    concern                  REAL    NOT NULL CHECK(concern BETWEEN 0 AND 1),
    curiosity                REAL    NOT NULL CHECK(curiosity BETWEEN 0 AND 1),
    connection_pressure      REAL    NOT NULL CHECK(connection_pressure BETWEEN 0 AND 1),
    uncertainty              REAL    NOT NULL CHECK(uncertainty BETWEEN 0 AND 1),
    offline_readiness        REAL    NOT NULL CHECK(offline_readiness BETWEEN 0 AND 1),
    wait_started_at_ms       INTEGER,
    wait_deadline_at_ms      INTEGER,
    last_user_event_id       TEXT    REFERENCES events(id),
    last_agent_event_id      TEXT    REFERENCES events(id),
    last_heartbeat_at_ms     INTEGER NOT NULL,
    transition_reason       TEXT    NOT NULL,
    updated_at_ms            INTEGER NOT NULL
);

CREATE INDEX idx_inner_loop_mode
ON inner_loop_states(mode, updated_at_ms);
