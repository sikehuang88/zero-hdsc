"""Initialize the SSA database.

Usage:
    uv run python scripts/init_db.py [--env development|evaluation|production]

Creates the database file, runs all migrations, and prints a summary.
"""

from __future__ import annotations

import sys

from ssa.config import Environment, load_settings
from ssa.storage.database import Database


def main() -> int:
    env_arg = (
        sys.argv[sys.argv.index("--env") + 1] if "--env" in sys.argv else "development"
    )
    env = Environment(env_arg)

    settings = load_settings(env)
    db = Database(settings.database)
    db.initialize()

    # Count tables.
    conn = db.connection
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [r[0] for r in tables]

    print(f"Database:     {db.path}")
    print(f"Environment:  {env.value}")
    print(f"Schema:       v{db.schema_version}")
    print(f"WAL mode:     {conn.execute('PRAGMA journal_mode').fetchone()[0]}")
    print(f"Foreign keys: {conn.execute('PRAGMA foreign_keys').fetchone()[0]}")
    print(f"Tables ({len(table_names)}):")
    for name in table_names:
        print(f"  - {name}")

    # Record embedding config in config_meta for startup verification.
    conn.execute(
        "INSERT OR REPLACE INTO config_meta (key, value, set_at) VALUES (?, ?, ?)",
        ("embedding_model", settings.embedding.model, "init"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO config_meta (key, value, set_at) VALUES (?, ?, ?)",
        ("embedding_dim", str(settings.embedding.dim), "init"),
    )
    conn.commit()

    print(f"\nEmbedding:    {settings.embedding.model} ({settings.embedding.dim}d)")
    print("Database initialized successfully.")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
