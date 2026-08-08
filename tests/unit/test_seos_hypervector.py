"""Tests for the bit-packed SEOS hypervector kernel."""

from __future__ import annotations

import numpy as np
import pytest

from ssa.hdsc.hypervector import (
    PackedHypervector,
    bind,
    bipolar_dot,
    bundle,
    hamming_distance,
    permute,
    semantic_random,
    similarity,
    unbind,
)


def test_bipolar_round_trip() -> None:
    values = np.asarray([1, -1, 1, 1, -1, -1, 1, -1, 1], dtype=np.int8)
    packed = PackedHypervector.from_bipolar(values)
    np.testing.assert_array_equal(packed.to_bipolar(), values)
    assert packed.byte_length == 2


def test_16384_dimensions_use_2048_bytes() -> None:
    vector = semantic_random(16_384, semantic_role="test", input_digest="content")
    assert vector.byte_length == 2_048


def test_binding_is_reversible() -> None:
    left = semantic_random(128, semantic_role="left", input_digest="a")
    right = semantic_random(128, semantic_role="right", input_digest="b")
    bound = bind(left, right)
    assert unbind(bound, right) == left
    identity = bind(left, left)
    np.testing.assert_array_equal(identity.to_bipolar(), np.ones(128, dtype=np.int8))


def test_similarity_matches_bipolar_hamming_identity() -> None:
    left = PackedHypervector.from_bipolar([1, 1, -1, -1])
    right = PackedHypervector.from_bipolar([1, -1, -1, 1])
    assert hamming_distance(left, right) == 2
    assert bipolar_dot(left, right) == 0
    assert similarity(left, right) == pytest.approx(0.0)
    assert similarity(left, left) == pytest.approx(1.0)


def test_permutation_round_trip() -> None:
    vector = semantic_random(256, semantic_role="input", input_digest="payload")
    encoded = permute(vector, semantic_role="temporal-role")
    restored = permute(encoded, semantic_role="temporal-role", inverse=True)
    assert restored == vector


def test_bundle_is_order_invariant_and_deterministic() -> None:
    left = semantic_random(128, semantic_role="left", input_digest="a")
    right = semantic_random(128, semantic_role="right", input_digest="b")
    first = bundle((left, right), semantic_role="memory")
    second = bundle((right, left), semantic_role="memory")
    third = bundle((left, right), semantic_role="memory")
    assert first == second == third


def test_bundle_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bundle((), semantic_role="empty")


def test_padding_bits_are_validated() -> None:
    with pytest.raises(ValueError, match="padding"):
        PackedHypervector(1, b"\x80")
