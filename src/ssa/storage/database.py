"""Database abstraction — SQLite connection, WAL, migrations, and transactions.

Pipeline §8.1 initialization sequence:
1. Create database parent directory.
2. Open connection, enable foreign_keys.
3. Enable WAL journal mode.
4. Set busy_timeout.
5. Check schema_migrations.
6. Execute missing migrations in a single transaction.
7. (sqlite-vec extension load is deferred to M02 when we add the vec table.)
8. Verify vector dimensions.
9. Lightweight integrity check.
10. Output schema version and database path.

Pipeline §7.3 transaction boundary rule: never hold a SQLite transaction
open while waiting on LLM or embedding calls.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ssa.config import DatabaseConfig
from ssa.domain.events import Event

_ADD_COLUMN_RE = re.compile(
    r"^ALTER\s+TABLE\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"ADD\s+COLUMN\s+(?P<column>[A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)


class MigrationError(RuntimeError):
    """Raised when a migration fails or schema is in an unexpected state."""


class Database:
    """Thin wrapper around `sqlite3.Connection` with WAL + migration support."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._conn: sqlite3.Connection | None = None
        self._vec_available: bool = False
        self._vec_table_pending: bool = False

    @property
    def config(self) -> DatabaseConfig:
        return self._config

    @property
    def path(self) -> Path:
        return Path(self._config.path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_vec_extension(self, conn: sqlite3.Connection) -> None:
        """Try to load the sqlite-vec extension (pipeline §8.1 step 7).

        If the extension is available, sets `_vec_available = True` so that
        the migration runner creates the `memory_vec` virtual table.
        If not available, `_vec_available` remains False and the vec table
        creation is skipped gracefully.
        """
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._vec_available = True
        except Exception:
            # Extension not available — retrieval will be limited.
            self._vec_available = False

    def initialize(self) -> None:
        """Open the connection and run pending migrations.

        Idempotent: safe to call on every startup.
        """
        self._vec_available = False
        self._vec_table_pending = False
        db_path = self.path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,  # autocommit mode; we manage txns manually
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        self._conn = conn

        # §8.1 steps 2-4
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self._config.busy_timeout_ms}")

        # §8.1 step 7: load sqlite-vec extension (before migrations).
        self._load_vec_extension(conn)

        # §8.1 steps 5-6: run migrations
        self._run_migrations(conn)

        # §8.1 step 9: lightweight integrity check
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result and result[0] != "ok":
            raise MigrationError(f"Integrity check failed: {result[0]}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        return self._conn

    @property
    def schema_version(self) -> int:
        row = self.connection.execute("SELECT MAX(version) as v FROM schema_migrations").fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection in a managed transaction.

        Uses BEGIN IMMEDIATE to reduce write contention. If an exception
        occurs, the transaction is rolled back. On success, it commits.

        IMPORTANT (pipeline §7.3): do NOT call LLM or embedding APIs
        inside this context manager.
        """
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Discover and apply migration files from the migrations/ directory.

        NOTE: `executescript()` implicitly commits, so it cannot be used
        inside a managed transaction. Instead we split each migration file
        into individual statements and execute them one by one. This keeps
        the migration atomic and allows rollback on failure.

        Statements that require the sqlite-vec extension (e.g.
        `CREATE VIRTUAL TABLE ... USING vec0`) are detected and skipped
        if the extension is not loaded. The caller can check
        `self.vec_extension_loaded` to determine vector availability.
        """
        migrations_dir = _migration_directory()
        if migrations_dir is None:
            raise MigrationError("Migration resources are missing from the installation")

        migration_files = sorted(
            migrations_dir.glob("*.sql"),
            key=lambda p: int(p.stem.split("_")[0]),
        )

        applied: set[int] = set()
        # Ensure schema_migrations table exists (idempotent with 001_core.sql).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                applied_at  TEXT    NOT NULL
            )
            """
        )
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        applied = {int(r["version"]) for r in rows}

        for mf in migration_files:
            version = int(mf.stem.split("_")[0])
            if version in applied:
                continue

            sql = mf.read_text(encoding="utf-8")
            name = mf.stem
            now = datetime.now(UTC).isoformat()

            statements = _split_sql_statements(sql)

            conn.execute("BEGIN IMMEDIATE")
            try:
                executed_statement = False
                deferred_vec_statement = False
                for stmt in statements:
                    stmt_stripped = stmt.strip()
                    if not stmt_stripped:
                        continue
                    # Skip vec0 virtual table creation if extension not loaded.
                    if "USING vec0" in stmt_stripped and not self._vec_available:
                        self._vec_table_pending = True
                        deferred_vec_statement = True
                        continue
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as exc:
                        if not self._is_satisfied_add_column(conn, stmt_stripped, exc):
                            raise
                    executed_statement = True

                # A migration containing only deferred vec0 DDL stays pending
                # so a later startup can retry it after the extension loads.
                if deferred_vec_statement and not executed_statement:
                    conn.execute("COMMIT")
                    continue
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, now),
                )
                conn.execute("COMMIT")
            except Exception as exc:
                conn.execute("ROLLBACK")
                raise MigrationError(f"Migration {name} failed: {exc}") from exc

            applied.add(version)

    @staticmethod
    def _is_satisfied_add_column(
        conn: sqlite3.Connection,
        statement: str,
        error: sqlite3.OperationalError,
    ) -> bool:
        """Treat an already-present ADD COLUMN as an idempotent upgrade.

        Migration 006 repairs databases created while migration 002 briefly
        contained vec_rowid. Those databases already have the column, while
        older phase-1 databases need it added. Only this narrowly verified
        duplicate-column case is ignored; every other SQL error is raised.
        """
        if "duplicate column name" not in str(error).lower():
            return False
        match = _ADD_COLUMN_RE.match(statement)
        if match is None:
            return False
        table = match.group("table")
        column = match.group("column")
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return any(str(row["name"]).lower() == column.lower() for row in rows)

    @property
    def vec_extension_loaded(self) -> bool:
        """Whether the sqlite-vec extension is available."""
        return self._vec_available and not self._vec_table_pending


def _migration_directory() -> Path | None:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parent.parent / "migrations",
        module_path.parent.parent.parent.parent / "migrations",
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


class EventRepository(Protocol):
    """Append-only event store (pipeline §11.2)."""

    def append(self, event: Event) -> Event: ...

    def get(self, event_id: str) -> Event | None: ...

    def find_by_channel_message(self, channel: str, message_id: str) -> Event | None: ...


__all__ = ["Database", "EventRepository", "MigrationError"]


def _split_sql_statements(sql: str) -> list[str]:
    """Split a multi-statement SQL string into individual statements.

    Strips ``--`` line comments, respects single-quoted string literals,
    and splits on semicolons. Sufficient for our DDL migration files.
    """
    statements: list[str] = []
    current: list[str] = []
    in_string = False
    lines = sql.splitlines()
    for line in lines:
        # Strip line comments (-- ...). A '--' inside a string literal is
        # rare in DDL, so we only check when not in_string.
        if not in_string:
            # Find '--' not inside a string.
            comment_pos = _find_comment_start(line, in_string)
            if comment_pos >= 0:
                line = line[:comment_pos]

        if not line.strip() and not in_string:
            continue

        i = 0
        while i < len(line):
            ch = line[i]

            if ch == "'":
                if not in_string:
                    in_string = True
                elif i + 1 < len(line) and line[i + 1] == "'":
                    # Escaped quote.
                    current.append(ch)
                    current.append(line[i + 1])
                    i += 2
                    continue
                else:
                    in_string = False

            current.append(ch)

            if ch == ";" and not in_string:
                stmt = "".join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []

            i += 1

        # End of line — add newline if we're mid-statement.
        if current and not in_string:
            current.append("\n")

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)

    return statements


def _find_comment_start(line: str, in_string: bool) -> int:
    """Find the position of ``--`` comment start, or -1 if none.

    Only detects comments outside of single-quoted strings.
    """
    if in_string:
        return -1
    i = 0
    while i < len(line):
        if line[i] == "'":
            # Skip to closing quote.
            i += 1
            while i < len(line):
                if line[i] == "'":
                    if i + 1 < len(line) and line[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if line[i] == "-" and i + 1 < len(line) and line[i + 1] == "-":
            return i
        i += 1
    return -1
