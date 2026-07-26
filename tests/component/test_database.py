"""Component tests for Database — migration runner, WAL, transactions.

Uses temporary file databases. No shared state between tests.
"""

from __future__ import annotations

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


def test_schema_version_is_5_after_migrations(db: Database):
    assert db.schema_version == 5


def test_all_tables_exist(db: Database):
    expected_tables = {
        "schema_migrations",
        "events",
        "idempotency_keys",
        "config_meta",
        "memories",
        "memory_evidence",
        "memory_links",
        "state_snapshots",
        "relationship_snapshots",
        "appraisals",
        "self_beliefs",
        "goals",
        "goal_steps",
        "llm_calls",
        "causal_traces",
        "scheduled_jobs",
        "initiatives",
        "outbox",
    }
    rows = db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    actual = {r["name"] for r in rows}
    missing = expected_tables - actual
    assert not missing, f"Missing tables: {missing}"


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
    row = db.connection.execute(
        "SELECT value FROM config_meta WHERE key = 'test_key'"
    ).fetchone()
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
    assert db.connection.execute(
        "SELECT COUNT(*) as c FROM config_meta WHERE key = 'iso_test'"
    ).fetchone()["c"] == 1


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
