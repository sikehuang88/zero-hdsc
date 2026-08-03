-- Migration 006: add stable sqlite-vec row mapping.
--
-- Migration 002 was applied before vec_rowid was introduced. Keep the old
-- migration immutable and upgrade both existing and fresh databases here.

ALTER TABLE memories ADD COLUMN vec_rowid INTEGER;

CREATE INDEX IF NOT EXISTS idx_memories_vec_rowid ON memories(vec_rowid);
