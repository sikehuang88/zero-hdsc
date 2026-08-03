CREATE TABLE emotion_frames (
    id                  TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL,
    correlation_id      TEXT NOT NULL UNIQUE,
    query_event_id      TEXT NOT NULL REFERENCES events(id),
    primary_emotion     TEXT NOT NULL,
    primary_intensity   REAL NOT NULL CHECK(primary_intensity BETWEEN 0.0 AND 1.0),
    valence             REAL NOT NULL CHECK(valence BETWEEN -1.0 AND 1.0),
    arousal             REAL NOT NULL CHECK(arousal BETWEEN 0.0 AND 1.0),
    inhibition          REAL NOT NULL CHECK(inhibition BETWEEN 0.0 AND 1.0),
    certainty           REAL NOT NULL CHECK(certainty BETWEEN 0.0 AND 1.0),
    action_tendency     TEXT NOT NULL,
    frame_json          TEXT NOT NULL CHECK(json_valid(frame_json)),
    created_at_ms       INTEGER NOT NULL
);

CREATE INDEX idx_emotion_frames_conversation_time
    ON emotion_frames(conversation_id, created_at_ms DESC);

CREATE TABLE voice_performance_segments (
    id                  TEXT PRIMARY KEY,
    emotion_frame_id    TEXT NOT NULL REFERENCES emotion_frames(id) ON DELETE CASCADE,
    conversation_id     TEXT NOT NULL,
    correlation_id      TEXT NOT NULL,
    sequence            INTEGER NOT NULL CHECK(sequence >= 0),
    recipe              TEXT NOT NULL,
    intensity           REAL NOT NULL CHECK(intensity BETWEEN 0.0 AND 1.0),
    rate                REAL NOT NULL CHECK(rate BETWEEN 0.72 AND 1.18),
    performance_json    TEXT NOT NULL CHECK(json_valid(performance_json)),
    created_at_ms       INTEGER NOT NULL,
    UNIQUE(emotion_frame_id, sequence)
);

CREATE INDEX idx_voice_performance_correlation
    ON voice_performance_segments(correlation_id, sequence);
