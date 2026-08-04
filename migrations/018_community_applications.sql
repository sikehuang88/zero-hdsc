CREATE TABLE community_applications (
    id                  TEXT PRIMARY KEY,
    application_code    TEXT NOT NULL UNIQUE,
    track               TEXT NOT NULL CHECK(track IN ('developer', 'tester')),
    email               TEXT NOT NULL COLLATE NOCASE,
    display_name        TEXT NOT NULL,
    focus               TEXT NOT NULL,
    profile_url         TEXT,
    platform            TEXT,
    notes               TEXT NOT NULL DEFAULT '',
    metadata_json       TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(metadata_json)),
    status              TEXT NOT NULL DEFAULT 'reserved'
                        CHECK(status IN ('reserved', 'reviewing', 'invited', 'closed')),
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL,
    UNIQUE(track, email)
);

CREATE INDEX idx_community_applications_track_time
ON community_applications(track, created_at_ms, id);

CREATE INDEX idx_community_applications_status
ON community_applications(track, status, updated_at_ms DESC);
