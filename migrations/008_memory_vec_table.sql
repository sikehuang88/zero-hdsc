-- Migration 008: retry creation of the optional sqlite-vec table.
--
-- Migration 002 can be applied while sqlite-vec is unavailable. Keeping this
-- table in a dedicated migration lets startup defer only this statement and
-- retry it after the extension becomes available.

CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    embedding float[512]
);
