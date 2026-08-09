-- Migration 023: persist observational warped free-energy recall diagnostics.

ALTER TABLE resonance_recall_audits
ADD COLUMN warped_shadow_json TEXT;
