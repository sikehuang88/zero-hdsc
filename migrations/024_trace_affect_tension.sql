-- Migration 024: retain the third affect axis required by warped retrieval.

ALTER TABLE traces
ADD COLUMN tension REAL NOT NULL DEFAULT 0.0 CHECK(tension BETWEEN 0 AND 1);
