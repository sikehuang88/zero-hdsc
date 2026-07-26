-- Migration 001: Core schema
-- Creates the foundational tables for the event store, idempotency,
-- schema migration tracking, and configuration metadata.

-- ---------------------------------------------------------------------------
-- Migration tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    applied_at  TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- Events — the immutable fact ledger (pipeline §8.2)
-- ---------------------------------------------------------------------------

CREATE TABLE events (
    id                  TEXT    PRIMARY KEY,
    correlation_id      TEXT    NOT NULL,
    conversation_id     TEXT    NOT NULL,
    actor               TEXT    NOT NULL,
    event_type          TEXT    NOT NULL,
    source_kind         TEXT    NOT NULL,
    content             TEXT    NOT NULL,
    parent_event_id     TEXT    REFERENCES events(id),
    channel             TEXT,
    channel_message_id  TEXT,
    content_hash        TEXT    NOT NULL,
    metadata_json       TEXT    NOT NULL DEFAULT '{}',
    created_at_ms       INTEGER NOT NULL,
    UNIQUE(channel, channel_message_id)
);

CREATE INDEX idx_events_correlation ON events(correlation_id);
CREATE INDEX idx_events_conversation ON events(conversation_id);
CREATE INDEX idx_events_created_at ON events(created_at_ms);
CREATE INDEX idx_events_actor ON events(actor);
CREATE INDEX idx_events_type ON events(event_type);

-- ---------------------------------------------------------------------------
-- Idempotency — deduplicates channel message delivery (pipeline §7.2 step 2)
-- ---------------------------------------------------------------------------

CREATE TABLE idempotency_keys (
    key             TEXT    PRIMARY KEY,
    correlation_id  TEXT    NOT NULL,
    result_json     TEXT    NOT NULL DEFAULT '{}',
    created_at_ms   INTEGER NOT NULL
);

-- ---------------------------------------------------------------------------
-- Configuration metadata — records the embedding model + dim at creation time
-- so that startup can detect mismatches early (pipeline §8.1 step 8).
-- ---------------------------------------------------------------------------

CREATE TABLE config_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    set_at  TEXT NOT NULL
);
