"""Tests for Clock protocol implementations."""

from ssa.clock import Clock, FrozenClock, SystemClock


def test_system_clock_is_a_clock():
    c = SystemClock()
    assert isinstance(c, Clock)


def test_system_clock_returns_positive_ms():
    c = SystemClock()
    assert c.now_ms() > 0


def test_system_clock_advances_over_time():
    c = SystemClock()
    t0 = c.now_ms()
    # Two calls should be monotonic (or equal on very fast machines).
    assert c.now_ms() >= t0


def test_frozen_clock_is_a_clock():
    c = FrozenClock()
    assert isinstance(c, Clock)


def test_frozen_clock_starts_at_given_value():
    c = FrozenClock(start_ms=1_000_000)
    assert c.now_ms() == 1_000_000


def test_frozen_clock_advance():
    c = FrozenClock(start_ms=1000)
    assert c.advance_ms(500) == 1500
    assert c.now_ms() == 1500
    assert c.advance_ms(0) == 1500


def test_frozen_clock_set():
    c = FrozenClock(start_ms=1000)
    c.set_ms(2_000_000)
    assert c.now_ms() == 2_000_000


def test_frozen_clock_does_not_move_implicitly():
    c = FrozenClock(start_ms=42)
    for _ in range(100):
        assert c.now_ms() == 42


def test_frozen_clock_rejects_negative_start():
    import pytest

    with pytest.raises(ValueError):
        FrozenClock(start_ms=-1)


def test_frozen_clock_rejects_negative_advance():
    import pytest

    c = FrozenClock()
    with pytest.raises(ValueError):
        c.advance_ms(-1)


def test_frozen_clock_rejects_negative_set():
    import pytest

    c = FrozenClock()
    with pytest.raises(ValueError):
        c.set_ms(-1)
