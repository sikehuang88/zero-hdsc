-- Migration 010: optional sqlite-vec index for the SSA trace space.
-- Kept separate so startup can defer and retry it when sqlite-vec is absent.

CREATE VIRTUAL TABLE IF NOT EXISTS trace_vec USING vec0(
    embedding float[512]
);
