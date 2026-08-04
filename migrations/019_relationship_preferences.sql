CREATE TABLE IF NOT EXISTS relationship_preferences (
    profile_id                 TEXT PRIMARY KEY,
    venom_intensity            INTEGER NOT NULL CHECK(venom_intensity BETWEEN 0 AND 100),
    support_mode               TEXT NOT NULL CHECK(support_mode IN ('adaptive', 'listen', 'comfort', 'solve', 'challenge')),
    proactive_frequency        TEXT NOT NULL CHECK(proactive_frequency IN ('off', 'low', 'balanced', 'high')),
    quiet_hours_enabled        INTEGER NOT NULL CHECK(quiet_hours_enabled IN (0, 1)),
    quiet_hours_start          TEXT NOT NULL,
    quiet_hours_end            TEXT NOT NULL,
    updated_at_ms              INTEGER NOT NULL CHECK(updated_at_ms >= 0)
);

INSERT OR IGNORE INTO relationship_preferences (
    profile_id,
    venom_intensity,
    support_mode,
    proactive_frequency,
    quiet_hours_enabled,
    quiet_hours_start,
    quiet_hours_end,
    updated_at_ms
) VALUES ('primary', 58, 'adaptive', 'balanced', 1, '23:00', '08:00', 0);
