"""Tests for ID generators."""

from ssa.ids import IdGenerator, SequentialIdGenerator, UuidIdGenerator


def test_uuid_generator_is_an_id_generator():
    g = UuidIdGenerator()
    assert isinstance(g, IdGenerator)


def test_uuid_generator_produces_unique_ids():
    g = UuidIdGenerator()
    ids = {g.new() for _ in range(1000)}
    assert len(ids) == 1000


def test_uuid_generator_produces_hex_strings():
    g = UuidIdGenerator()
    s = g.new()
    assert isinstance(s, str)
    # uuid4().hex is 32 hex chars, no dashes.
    assert len(s) == 32
    int(s, 16)  # must be valid hex


def test_sequential_generator_is_an_id_generator():
    g = SequentialIdGenerator()
    assert isinstance(g, IdGenerator)


def test_sequential_generator_produces_ordered_ids():
    g = SequentialIdGenerator()
    ids = [g.new() for _ in range(5)]
    assert ids == ["seq-000001", "seq-000002", "seq-000003", "seq-000004", "seq-000005"]


def test_sequential_generator_custom_prefix():
    g = SequentialIdGenerator(prefix="evt")
    assert g.new() == "evt-000001"
    assert g.new() == "evt-000002"


def test_sequential_generator_independent_instances():
    a = SequentialIdGenerator()
    b = SequentialIdGenerator()
    assert a.new() == "seq-000001"
    assert b.new() == "seq-000001"
    assert a.new() == "seq-000002"
