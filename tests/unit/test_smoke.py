"""Smoke test — verifies the package imports and reports its version."""

import ssa


def test_package_imports():
    """`import ssa` must not raise and must expose a version string."""
    assert ssa is not None


def test_version_is_string():
    assert isinstance(ssa.__version__, str)
    assert len(ssa.__version__) > 0


def test_version_format():
    parts = ssa.__version__.split(".")
    assert len(parts) >= 2, "version should be at least major.minor"
