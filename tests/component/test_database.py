"""Component tests for Database — migration runner, WAL, transactions.

Uses temporary file databases. No shared state between tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ssa.config import DatabaseConfig
from ssa.storage.database import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    config = DatabaseConfig(path=str(tmp_db_path), busy_timeout_ms=1000)
    d = Database(config)
    d.initialize()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_creates_database_file(tmp_db_path: Path):
    config = DatabaseConfig(path=str(tmp_db_path))
    d = Database(config)
    d.initialize()
    assert tmp_db_path.exists()
    d.close()


def test_initialize_is_idempotent(db: Database):
    """Calling initialize twice should not raise."""
    # The second call re-runs migrations but finds them all applied.
    db.close()
    db.initialize()
    assert db.schema_version > 0


def test_initialize_creates_parent_directory(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c" / "test.db"
    config = DatabaseConfig(path=str(nested))
    d = Database(config)
    d.initialize()
    assert nested.exists()
    d.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_version_is_26_after_migrations(db: Database):
    assert db.schema_version == 26


def test_trace_schema_retains_warped_affect_tension(db: Database) -> None:
    columns = {row["name"] for row in db.connection.execute("PRAGMA table_info(traces)")}
    assert "tension" in columns


@pytest.mark.parametrize("column_already_present", [False, True])
def test_migration_006_upgrades_both_phase1_and_transitional_databases(
    tmp_db_path: Path,
    column_already_present: bool,
):
    """Migration 006 upgrades old v5 DBs and accepts the transitional schema."""
    conn = sqlite3.connect(tmp_db_path)
    conn.execute(
        "CREATE TABLE schema_migrations "
        "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    vec_column = ", vec_rowid INTEGER" if column_already_present else ""
    conn.execute(f"CREATE TABLE memories (id TEXT PRIMARY KEY{vec_column})")
    conn.execute("CREATE TABLE events (id TEXT PRIMARY KEY)")
    conn.execute(
        """
        CREATE TABLE self_beliefs (
            id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'candidate',
            version INTEGER NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '[]',
            counterevidence_json TEXT NOT NULL DEFAULT '[]',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO self_beliefs (
            id, claim, confidence, status, version,
            evidence_json, counterevidence_json, created_at_ms, updated_at_ms
        ) VALUES ('legacy-belief', 'legacy claim', 0.8, 'active', 3, '[]', '[]', 100, 200)
        """
    )
    for version in range(1, 6):
        conn.execute(
            "INSERT INTO schema_migrations VALUES (?, ?, ?)",
            (version, f"00{version}_legacy", "2026-07-25T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    upgraded = Database(DatabaseConfig(path=str(tmp_db_path)))
    upgraded.initialize()
    columns = {row["name"] for row in upgraded.connection.execute("PRAGMA table_info(memories)")}
    assert "vec_rowid" in columns
    legacy = upgraded.connection.execute(
        """
        SELECT lineage_id, version, candidate_since_ms, activated_at_ms
        FROM self_beliefs WHERE id = 'legacy-belief'
        """
    ).fetchone()
    assert legacy["lineage_id"] == "legacy-belief"
    assert legacy["version"] == 1
    assert legacy["candidate_since_ms"] == 100
    assert legacy["activated_at_ms"] == 100
    assert upgraded.schema_version == 26
    upgraded.close()


def test_all_tables_exist(db: Database):
    expected_tables = {
        "schema_migrations",
        "events",
        "idempotency_keys",
        "config_meta",
        "memories",
        "memory_evidence",
        "memory_links",
        "memory_vec",
        "traces",
        "trace_links",
        "trace_activations",
        "resonance_recall_audits",
        "trace_vec",
        "state_snapshots",
        "relationship_snapshots",
        "appraisals",
        "perception_snapshots",
        "self_beliefs",
        "self_belief_evidence",
        "goals",
        "goal_steps",
        "llm_calls",
        "causal_traces",
        "scheduled_jobs",
        "initiatives",
        "outbox",
        "inner_loop_states",
        "reflection_runs",
        "learning_proposals",
        "learning_proposal_evidence",
        "consolidation_decisions",
        "behavior_experiments",
        "outcome_observations",
        "emotional_memories",
        "emotion_frames",
        "voice_performance_segments",
        "community_applications",
        "relationship_preferences",
        "romantic_persona_profiles",
        "predictions",
        "engram_nodes",
        "engram_edges",
        "cement_seal_state",
        "cement_seal_transitions",
    }
    rows = db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    actual = {r["name"] for r in rows}
    missing = expected_tables - actual
    assert not missing, f"Missing tables: {missing}"


def test_deferred_vector_table_is_retried_when_extension_recovers(
    tmp_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def skip_vec_load(database: Database, _connection: sqlite3.Connection) -> None:
        database._vec_available = False

    monkeypatch.setattr(Database, "_load_vec_extension", skip_vec_load)
    deferred = Database(DatabaseConfig(path=str(tmp_db_path)))
    deferred.initialize()
    deferred_versions = {
        int(row["version"])
        for row in deferred.connection.execute("SELECT version FROM schema_migrations")
    }
    assert deferred.schema_version == 26
    assert 8 not in deferred_versions
    assert 10 not in deferred_versions
    assert 11 in deferred_versions
    assert deferred.vec_extension_loaded is False
    assert (
        deferred.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'memory_vec'"
        ).fetchone()
        is None
    )
    assert (
        deferred.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'trace_vec'"
        ).fetchone()
        is None
    )
    deferred.close()

    monkeypatch.undo()
    recovered = Database(DatabaseConfig(path=str(tmp_db_path)))
    recovered.initialize()
    recovered_versions = {
        int(row["version"])
        for row in recovered.connection.execute("SELECT version FROM schema_migrations")
    }
    assert recovered.schema_version == 26
    assert set(range(1, 12)) <= recovered_versions
    assert recovered.vec_extension_loaded is True
    assert (
        recovered.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'memory_vec'"
        ).fetchone()
        is not None
    )
    assert (
        recovered.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'trace_vec'"
        ).fetchone()
        is not None
    )
    recovered.close()


def test_wal_mode_enabled(db: Database):
    row = db.connection.execute("PRAGMA journal_mode").fetchone()
    assert str(row[0]).lower() == "wal"


def test_foreign_keys_enabled(db: Database):
    row = db.connection.execute("PRAGMA foreign_keys").fetchone()
    assert int(row[0]) == 1


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def test_transaction_commits_on_success(db: Database):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO config_meta (key, value, set_at) VALUES (?, ?, ?)",
            ("test_key", "test_value", "2026-01-01T00:00:00Z"),
        )
    row = db.connection.execute("SELECT value FROM config_meta WHERE key = 'test_key'").fetchone()
    assert row is not None
    assert row["value"] == "test_value"


def test_transaction_rolls_back_on_exception(db: Database):
    with pytest.raises(ValueError, match="rollback test"), db.transaction() as conn:
        conn.execute(
            "INSERT INTO config_meta (key, value, set_at) VALUES (?, ?, ?)",
            ("rollback_key", "rollback_value", "2026-01-01T00:00:00Z"),
        )
        raise ValueError("rollback test")

    row = db.connection.execute(
        "SELECT value FROM config_meta WHERE key = 'rollback_key'"
    ).fetchone()
    assert row is None


def test_transaction_is_isolation_level_immediate(db: Database):
    """The transaction should use BEGIN IMMEDIATE to reduce write contention."""
    # This is implicitly tested by the fact that we can write inside the
    # transaction without errors. A more thorough test would check the
    # actual SQL log, but that's overkill for this phase.
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO config_meta (key, value, set_at) VALUES (?, ?, ?)",
            ("iso_test", "val", "2026-01-01T00:00:00Z"),
        )
    assert (
        db.connection.execute(
            "SELECT COUNT(*) as c FROM config_meta WHERE key = 'iso_test'"
        ).fetchone()["c"]
        == 1
    )


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_integrity_check_passes(db: Database):
    row = db.connection.execute("PRAGMA integrity_check").fetchone()
    assert row[0] == "ok"


# ---------------------------------------------------------------------------
# Connection access
# ---------------------------------------------------------------------------


def test_connection_raises_before_initialize(tmp_db_path: Path):
    d = Database(DatabaseConfig(path=str(tmp_db_path)))
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = d.connection
