"""Component tests for SqliteEventRepository — append, get, dedup.

Pipeline §11.5 acceptance:
- 1000 events write and read back.
- Duplicate channel messages produce only one Event.
- Transaction mid-exception leaves no partial state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa.clock import FrozenClock
from ssa.config import DatabaseConfig
from ssa.domain.enums import Actor, SourceKind
from ssa.domain.events import Event, IncomingSignal, compute_content_hash, normalize_signal
from ssa.ids import SequentialIdGenerator
from ssa.storage.database import Database
from ssa.storage.event_repository import DuplicateEventError, SqliteEventRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> tuple[SqliteEventRepository, Database, SequentialIdGenerator, FrozenClock]:
    db = Database(DatabaseConfig(path=str(tmp_path / "test.db")))
    db.initialize()
    conn = db.connection
    r = SqliteEventRepository(conn)
    ids = SequentialIdGenerator(prefix="evt")
    clock = FrozenClock(start_ms=1_700_000_000_000)
    yield r, db, ids, clock
    db.close()


def make_user_signal(
    content: str = "hello",
    *,
    channel: str = "cli",
    channel_message_id: str | None = None,
    conversation_id: str = "conv-1",
) -> IncomingSignal:
    return IncomingSignal(
        actor=Actor.USER,
        signal_type="user.message",
        content=content,
        channel=channel,
        channel_message_id=channel_message_id,
        conversation_id=conversation_id,
    )


def make_event(
    ids: SequentialIdGenerator,
    clock: FrozenClock,
    content: str = "hello",
    *,
    channel: str = "cli",
    channel_message_id: str | None = None,
) -> Event:
    signal = make_user_signal(
        content, channel=channel, channel_message_id=channel_message_id
    )
    return normalize_signal(
        signal,
        event_id=ids.new(),
        correlation_id=ids.new().replace("evt", "corr"),
        now_ms=clock.now_ms(),
        source_kind=SourceKind.USER_OBSERVED,
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_append_and_get(repo):
    r, _db, ids, clock = repo
    event = make_event(ids, clock, content="hello world", channel_message_id="msg-1")
    r.append(event)

    fetched = r.get(event.id)
    assert fetched is not None
    assert fetched.id == event.id
    assert fetched.content == "hello world"
    assert fetched.actor == Actor.USER
    assert fetched.source_kind == SourceKind.USER_OBSERVED
    assert fetched.event_type == "user.message"


def test_append_preserves_metadata(repo):
    r, _db, ids, clock = repo
    signal = make_user_signal("test", channel_message_id="msg-1")
    signal.metadata = {"source": "telegram", "lang": "zh"}
    event = normalize_signal(
        signal,
        event_id=ids.new(),
        correlation_id="corr-1",
        now_ms=clock.now_ms(),
        source_kind=SourceKind.USER_OBSERVED,
    )
    r.append(event)

    fetched = r.get(event.id)
    assert fetched is not None
    assert fetched.metadata["source"] == "telegram"
    assert fetched.metadata["lang"] == "zh"


def test_get_returns_none_for_missing(repo):
    r, *_ = repo
    assert r.get("nonexistent") is None


def test_count(repo):
    r, _db, ids, clock = repo
    assert r.count() == 0
    for i in range(5):
        e = make_event(ids, clock, content=f"msg-{i}", channel_message_id=f"msg-{i}")
        r.append(e)
    assert r.count() == 5


def test_recent(repo):
    r, _db, ids, clock = repo
    for i in range(10):
        e = make_event(ids, clock, content=f"msg-{i}", channel_message_id=f"m-{i}")
        r.append(e)
        clock.advance_ms(1000)

    recent = r.recent(limit=3)
    assert len(recent) == 3
    # Most recent first.
    assert recent[0].content == "msg-9"
    assert recent[2].content == "msg-7"


# ---------------------------------------------------------------------------
# Idempotency / dedup
# ---------------------------------------------------------------------------


def test_duplicate_channel_message_raises(repo):
    r, _db, ids, clock = repo
    e1 = make_event(ids, clock, content="hello", channel_message_id="tg-123")
    r.append(e1)

    e2 = make_event(ids, clock, content="hello again", channel_message_id="tg-123")
    with pytest.raises(DuplicateEventError, match="already exists"):
        r.append(e2)


def test_find_by_channel_message(repo):
    r, _db, ids, clock = repo
    e = make_event(ids, clock, content="hello", channel_message_id="tg-456")
    r.append(e)

    found = r.find_by_channel_message("cli", "tg-456")
    assert found is not None
    assert found.id == e.id

    assert r.find_by_channel_message("cli", "nonexistent") is None


def test_different_channels_same_message_id_ok(repo):
    r, _db, ids, clock = repo
    e1 = make_event(ids, clock, content="hello", channel_message_id="123")
    e1 = e1.model_copy(update={"channel": "telegram"})
    r.append(e1)

    e2 = make_event(ids, clock, content="hello", channel_message_id="123")
    e2 = e2.model_copy(update={"channel": "cli"})
    # Should not raise — different channels.
    r.append(e2)
    assert r.count() == 2


def test_null_channel_message_id_allows_duplicates(repo):
    """Events without channel_message_id should not trigger the UNIQUE constraint."""
    r, _db, ids, clock = repo
    e1 = make_event(ids, clock, content="hello", channel_message_id=None)
    e2 = make_event(ids, clock, content="hello again", channel_message_id=None)
    r.append(e1)
    r.append(e2)
    assert r.count() == 2


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_find_by_correlation(repo):
    r, _db, ids, clock = repo
    signal = make_user_signal("hello", channel_message_id="m-1")
    e = normalize_signal(
        signal,
        event_id=ids.new(),
        correlation_id="corr-abc",
        now_ms=clock.now_ms(),
        source_kind=SourceKind.USER_OBSERVED,
    )
    r.append(e)

    results = r.find_by_correlation("corr-abc")
    assert len(results) == 1
    assert results[0].id == e.id


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


def test_content_hash_is_sha256(repo):
    r, _db, ids, clock = repo
    e = make_event(ids, clock, content="hello world", channel_message_id="m-1")
    r.append(e)

    fetched = r.get(e.id)
    assert fetched is not None
    assert fetched.content_hash == compute_content_hash("hello world")
    assert len(fetched.content_hash) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Scale: 1000 events
# ---------------------------------------------------------------------------


def test_write_1000_events(repo):
    """Pipeline §11.5: 1000 events write and restart test."""
    r, _db, ids, clock = repo
    for i in range(1000):
        e = make_event(
            ids,
            clock,
            content=f"message number {i}",
            channel_message_id=f"msg-{i:04d}",
        )
        r.append(e)
        clock.advance_ms(1)

    assert r.count() == 1000

    # Read back a few random ones.
    all_recent = r.recent(limit=1000)
    assert len(all_recent) == 1000
    # Most recent should have the highest timestamp.
    timestamps = [e.created_at_ms for e in all_recent]
    assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Transaction rollback leaves no partial state
# ---------------------------------------------------------------------------


def test_append_in_transaction_rollback(db_repo_factory):
    """Pipeline §11.5: transaction mid-exception leaves no partial state."""
    r, db, ids, clock = db_repo_factory

    # Write one event successfully.
    e1 = make_event(ids, clock, content="first", channel_message_id="m-1")
    r.append(e1)

    # Attempt to write in a transaction that then fails.
    e2 = make_event(ids, clock, content="second", channel_message_id="m-2")
    with pytest.raises(ValueError, match="simulated failure"), db.transaction():
        r.append(e2)
        raise ValueError("simulated failure")

    # Only the first event should exist.
    assert r.count() == 1
    assert r.get(e2.id) is None


# ---------------------------------------------------------------------------
# Restart persistence
# ---------------------------------------------------------------------------


def test_events_survive_restart(tmp_path: Path):
    """Pipeline §11.5: events persist across database reopen."""
    db_path = str(tmp_path / "persist.db")
    config = DatabaseConfig(path=db_path)

    # Write.
    db1 = Database(config)
    db1.initialize()
    ids = SequentialIdGenerator()
    clock = FrozenClock()
    r1 = SqliteEventRepository(db1.connection)
    for i in range(50):
        e = make_event(ids, clock, content=f"msg {i}", channel_message_id=f"m-{i}")
        r1.append(e)
    db1.close()

    # Reopen.
    db2 = Database(config)
    db2.initialize()
    r2 = SqliteEventRepository(db2.connection)
    assert r2.count() == 50
    db2.close()


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------


def test_normalize_strips_nul_chars():
    signal = make_user_signal("hello\x00world", channel_message_id="m-1")
    event = normalize_signal(
        signal,
        event_id="e-1",
        correlation_id="c-1",
        now_ms=1000,
        source_kind=SourceKind.USER_OBSERVED,
    )
    assert "\x00" not in event.content
    assert event.content == "helloworld"


def test_normalize_preserves_newlines():
    signal = make_user_signal("line1\nline2\n", channel_message_id="m-1")
    event = normalize_signal(
        signal,
        event_id="e-1",
        correlation_id="c-1",
        now_ms=1000,
        source_kind=SourceKind.USER_OBSERVED,
    )
    assert "\n" in event.content


def test_event_to_jsonl(repo):
    r, _db, ids, clock = repo
    e = make_event(ids, clock, content="hello", channel_message_id="m-1")
    r.append(e)
    fetched = r.get(e.id)
    assert fetched is not None
    jsonl = fetched.to_jsonl()
    assert isinstance(jsonl, str)
    assert "hello" in jsonl
    assert "user" in jsonl
    assert "user_observed" in jsonl


# ---------------------------------------------------------------------------
# Fixtures for transaction test
# ---------------------------------------------------------------------------


@pytest.fixture
def db_repo_factory(tmp_path: Path):
    db = Database(DatabaseConfig(path=str(tmp_path / "tx_test.db")))
    db.initialize()
    r = SqliteEventRepository(db.connection)
    ids = SequentialIdGenerator(prefix="evt")
    clock = FrozenClock()
    yield r, db, ids, clock
    db.close()
