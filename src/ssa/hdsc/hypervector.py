"""Bit-packed bipolar hypervectors for the SEOS representation family."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

_NAMESPACE = "hdsc-seos-hypervector-v1"
_POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def _byte_length(dimension: int) -> int:
    return (dimension + 7) // 8


def _semantic_rng(
    semantic_role: str,
    content_digests: Sequence[str],
) -> np.random.Generator:
    """Derive randomness only from semantic role and stable content digests."""

    if not semantic_role.strip():
        raise ValueError("semantic_role must not be empty")
    if any(not digest.strip() for digest in content_digests):
        raise ValueError("content digests must not be empty")
    payload = ":".join((_NAMESPACE, semantic_role, *content_digests))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True, slots=True)
class PackedHypervector:
    """Immutable 1-bit bipolar vector.

    A zero bit represents +1 and a one bit represents -1. Bipolar binding is
    therefore byte-wise XOR.
    """

    dimension: int
    data: bytes

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if len(self.data) != _byte_length(self.dimension):
            raise ValueError("packed byte length does not match dimension")
        unused = len(self.data) * 8 - self.dimension
        if unused:
            mask = ((1 << unused) - 1) << (8 - unused)
            if self.data[-1] & mask:
                raise ValueError("unused padding bits must be zero")

    @property
    def byte_length(self) -> int:
        return len(self.data)

    @property
    def digest(self) -> str:
        payload = self.dimension.to_bytes(8, "big") + self.data
        return hashlib.sha256((_NAMESPACE + ":content:").encode("ascii") + payload).hexdigest()

    def as_uint8(self) -> NDArray[np.uint8]:
        return cast(NDArray[np.uint8], np.frombuffer(self.data, dtype=np.uint8))

    def to_bipolar(self) -> NDArray[np.int8]:
        bits = np.unpackbits(self.as_uint8(), bitorder="little")[: self.dimension]
        result: NDArray[np.int8] = cast(
            NDArray[np.int8],
            (1 - 2 * bits.astype(np.int8)).astype(np.int8, copy=False),
        )
        return cast(NDArray[np.int8], result)

    @classmethod
    def from_bipolar(cls, values: Sequence[int] | np.ndarray) -> PackedHypervector:
        array = np.asarray(values, dtype=np.int8)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("bipolar values must be a non-empty one-dimensional vector")
        if not bool(np.all((array == -1) | (array == 1))):
            raise ValueError("bipolar values must contain only -1 and +1")
        bits = (array < 0).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        return cls(int(array.size), packed.tobytes())


def semantic_random(
    dimension: int,
    *,
    semantic_role: str,
    input_digest: str,
) -> PackedHypervector:
    """Create a reproducible hypervector without structural seed material."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = _semantic_rng(semantic_role, (input_digest,))
    bits = rng.integers(0, 2, size=dimension, dtype=np.uint8)
    return PackedHypervector.from_bipolar(1 - 2 * bits.astype(np.int8))


def _same_dimension(left: PackedHypervector, right: PackedHypervector) -> None:
    if left.dimension != right.dimension:
        raise ValueError("hypervectors must share one dimension")


def bind(left: PackedHypervector, right: PackedHypervector) -> PackedHypervector:
    """Bind two bipolar vectors; XOR is self-inverse in this encoding."""

    _same_dimension(left, right)
    data = np.bitwise_xor(left.as_uint8(), right.as_uint8()).tobytes()
    return PackedHypervector(left.dimension, data)


def unbind(bound: PackedHypervector, key: PackedHypervector) -> PackedHypervector:
    return bind(bound, key)


def hamming_distance(left: PackedHypervector, right: PackedHypervector) -> int:
    _same_dimension(left, right)
    xor = np.bitwise_xor(left.as_uint8(), right.as_uint8())
    return int(_POPCOUNT[xor].sum(dtype=np.int64))


def bipolar_dot(left: PackedHypervector, right: PackedHypervector) -> int:
    return left.dimension - 2 * hamming_distance(left, right)


def similarity(left: PackedHypervector, right: PackedHypervector) -> float:
    return bipolar_dot(left, right) / left.dimension


def _permutation(dimension: int, semantic_role: str) -> NDArray[np.int64]:
    shape_digest = hashlib.sha256(f"dimension:{dimension}".encode("ascii")).hexdigest()
    rng = _semantic_rng(f"permutation:{semantic_role}", (shape_digest,))
    return cast(NDArray[np.int64], rng.permutation(dimension))


def permute(
    vector: PackedHypervector,
    *,
    semantic_role: str,
    inverse: bool = False,
) -> PackedHypervector:
    """Apply a role-stable permutation independent of graph structure."""

    indexes = _permutation(vector.dimension, semantic_role)
    if inverse:
        indexes = np.argsort(indexes)
    values = vector.to_bipolar()[indexes]
    return PackedHypervector.from_bipolar(values)


def bundle(
    vectors: Iterable[PackedHypervector],
    *,
    semantic_role: str,
) -> PackedHypervector:
    """Bundle vectors by majority vote with content-stable tie breaking."""

    items = tuple(vectors)
    if not items:
        raise ValueError("bundle requires at least one hypervector")
    dimension = items[0].dimension
    if any(item.dimension != dimension for item in items):
        raise ValueError("all bundled hypervectors must share one dimension")
    votes: NDArray[np.int32] = np.zeros(dimension, dtype=np.int32)
    for item in items:
        votes += item.to_bipolar().astype(np.int32)
    result = np.sign(votes).astype(np.int8)
    ties = np.flatnonzero(votes == 0)
    if ties.size:
        input_digests = tuple(sorted(item.digest for item in items))
        rng = _semantic_rng(f"bundle-tie:{semantic_role}", input_digests)
        tie_bits = rng.integers(0, 2, size=ties.size, dtype=np.uint8)
        result[ties] = 1 - 2 * tie_bits.astype(np.int8)
    return PackedHypervector.from_bipolar(result)
