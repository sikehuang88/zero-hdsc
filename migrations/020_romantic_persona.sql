CREATE TABLE IF NOT EXISTS romantic_persona_profiles (
    profile_id                 TEXT PRIMARY KEY,
    character_name             TEXT NOT NULL,
    owner_address              TEXT NOT NULL,
    relationship_stage         TEXT NOT NULL CHECK(relationship_stage IN ('established', 'committed', 'long_term')),
    affection_style            TEXT NOT NULL CHECK(affection_style IN ('restrained', 'balanced', 'expressive')),
    care_style                 TEXT NOT NULL CHECK(care_style IN ('attentive', 'practical', 'protective')),
    conflict_style             TEXT NOT NULL CHECK(conflict_style IN ('direct_repair', 'soft_repair', 'cooldown_repair')),
    teasing_intensity          INTEGER NOT NULL CHECK(teasing_intensity BETWEEN 0 AND 100),
    vulnerability_openness     INTEGER NOT NULL CHECK(vulnerability_openness BETWEEN 0 AND 100),
    custom_notes               TEXT NOT NULL,
    version                    INTEGER NOT NULL CHECK(version >= 1),
    updated_at_ms              INTEGER NOT NULL CHECK(updated_at_ms >= 0)
);

INSERT OR IGNORE INTO romantic_persona_profiles (
    profile_id,
    character_name,
    owner_address,
    relationship_stage,
    affection_style,
    care_style,
    conflict_style,
    teasing_intensity,
    vulnerability_openness,
    custom_notes,
    version,
    updated_at_ms
) VALUES (
    'primary',
    '清雪',
    '你',
    'committed',
    'balanced',
    'attentive',
    'direct_repair',
    62,
    48,
    '',
    1,
    0
);
