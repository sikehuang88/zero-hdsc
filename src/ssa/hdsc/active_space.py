"""HDSC-H2 bounded active-space shadow state.

The append-only archive is deliberately absent from the behavior-state hash.
H2 keeps a fixed-capacity probability measure over semantic clusters plus a
null reservoir. It is a C0-M stability candidate, not a production retriever.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np

from ssa.hdsc.transport import HDSCInvariantError

MODEL_ID: Final = "hdsc-h2-bounded-active-shadow"
CLAIM_LEVEL: Final = "C0-M-bounded-active-state"
CERTIFICATE_SCOPE: Final = "local-fixed-candidate-cluster-capacity-support"
MODELED_SCALE: Final = "macro-coarse-grained"
MICROSTATE_STATUS: Final = "unmodeled"
TRANSPORT_MODEL: Final = "identity-nonexpansive"
SUPPORT_METRIC: Final = "max(normalized-embedding-l2,raw-score-source-distance,prior-state-tv)"

CertificateStatus = Literal["pass", "abstain", "fail"]


@dataclass(frozen=True, slots=True)
class H2Config:
    """Versioned limits for the bounded active-space update."""

    semantic_partition_id: str = "external-semantic-partition-v1"
    capacity: int = 16
    candidate_top_k: int = 8
    retention: float = 0.80
    injection_rate: float = 0.15
    hysteresis: float = 0.02
    score_lipschitz_bound: float = 1.0
    gain_margin: float = 0.05
    mass_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not self.semantic_partition_id.strip():
            raise ValueError("semantic_partition_id must not be empty")
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 1 <= self.candidate_top_k <= self.capacity:
            raise ValueError("candidate_top_k must be in [1, capacity]")
        if not 0.0 <= self.retention < 1.0:
            raise ValueError("retention must be in [0, 1)")
        if not 0.0 <= self.injection_rate <= 1.0:
            raise ValueError("injection_rate must be in [0, 1]")
        if self.retention + self.injection_rate > 1.0:
            raise ValueError("retention + injection_rate must be <= 1")
        if self.hysteresis < 0.0 or not math.isfinite(self.hysteresis):
            raise ValueError("hysteresis must be finite and non-negative")
        if self.score_lipschitz_bound <= 0.0 or not math.isfinite(self.score_lipschitz_bound):
            raise ValueError("score_lipschitz_bound must be finite and positive")
        if not 0.0 < self.gain_margin < 1.0:
            raise ValueError("gain_margin must be in (0, 1)")
        if self.mass_tolerance <= 0.0 or not math.isfinite(self.mass_tolerance):
            raise ValueError("mass_tolerance must be finite and positive")

    @property
    def config_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class H2GainBounds:
    """Empirical Lipschitz bounds used by the conditional small-gain gate."""

    response_signal: float | None = None
    response_memory: float | None = None
    user_feedback: float | None = None
    proposal_signal: float | None = None
    proposal_response: float | None = None
    proposal_memory: float | None = None
    user_gain_source: str = "unmeasured"


@dataclass(frozen=True, slots=True)
class H2Candidate:
    """One proposed archive observation for the shadow active state."""

    trace_id: str
    semantic_fingerprint: str
    cluster_id: str
    cluster_revision: str
    prototype: tuple[float, ...]
    relevance: float
    novelty: float
    cluster_assignment_margin: float | None = None


@dataclass(frozen=True, slots=True)
class H2ActiveCluster:
    """One bounded behavior-state atom."""

    cluster_id: str
    cluster_revision: str
    prototype: tuple[float, ...]
    prototype_hash: str
    mass: float

    @property
    def key(self) -> str:
        return _cluster_key(self.cluster_id, self.cluster_revision)


@dataclass(frozen=True, slots=True)
class H2State:
    """Fixed-capacity active measure; archive size is intentionally excluded."""

    step: int
    clusters: tuple[H2ActiveCluster, ...]
    null_mass: float
    config_hash: str

    @classmethod
    def initial(cls, config: H2Config) -> H2State:
        return cls(step=0, clusters=(), null_mass=1.0, config_hash=config.config_hash)

    @property
    def state_hash(self) -> str:
        """Hash only the behavior measure and its semantic/configuration meaning."""
        return _state_hash(self)

    @property
    def replay_hash(self) -> str:
        """Hash the measure plus its chronological replay position."""
        return _replay_hash(self)


@dataclass(frozen=True, slots=True)
class H2Audit:
    """Certificate inputs and invariant evidence for one shadow update."""

    model_id: str
    claim_level: str
    modeled_scale: str
    microstate_status: str
    transport_model: str
    shadow_only: bool
    config_hash: str
    semantic_partition_id: str
    previous_state_hash: str
    next_state_hash: str
    previous_replay_hash: str
    next_replay_hash: str
    archive_trace_count: int
    capacity: int
    active_before: int
    active_after: int
    candidate_count: int
    unique_cluster_count: int
    duplicate_count: int
    proposal_count: int
    proposal_total_score: float
    proposal_normalization_gain: float | None
    proposal_boundary_gap: float
    capacity_boundary_gap: float
    cluster_assignment_margin: float | None
    proposal_tie_count: int
    support_churn: float
    replacement_count: int
    dropped_mass: float
    duplicate_suppressed_mass: float
    null_mass_before: float
    null_mass_after: float
    total_mass: float
    mass_residual: float
    minimum_mass: float
    retention: float
    injection_rate: float
    local_memory_contraction_bound: float
    semantic_jump_bound: float
    proposal_certified_radius: float
    capacity_certified_radius: float
    certified_radius: float
    small_gain_matrix: tuple[tuple[float, float], tuple[float, float]] | None
    small_gain_spectral_radius: float | None
    small_gain_margin: float | None
    response_signal_gain: float | None
    response_memory_gain: float | None
    user_feedback_gain: float | None
    proposal_signal_gain: float | None
    proposal_response_gain: float | None
    proposal_memory_gain: float | None
    user_gain_source: str
    certificate_scope: str
    support_metric: str
    certificate_status: CertificateStatus
    certificate_reason: str


@dataclass(frozen=True, slots=True)
class H2Result:
    """Next active state and its complete shadow audit."""

    state: H2State
    proposal_mass: tuple[tuple[str, float], ...]
    audit: H2Audit


@dataclass(frozen=True, slots=True)
class _ClusterProposal:
    key: str
    cluster_id: str
    cluster_revision: str
    prototype: tuple[float, ...]
    score: float
    member_count: int


def bounded_active_shadow_step(
    state: H2State,
    candidates: Sequence[H2Candidate],
    config: H2Config,
    *,
    gains: H2GainBounds | None = None,
    archive_trace_count: int = 0,
) -> H2Result:
    """Advance the bounded shadow state without changing production behavior."""
    _validate_state(state, config)
    if archive_trace_count < 0:
        raise HDSCInvariantError("archive_trace_count must be non-negative")
    validated = [_validate_candidate(candidate, config.mass_tolerance) for candidate in candidates]
    dimensions = {len(cluster.prototype) for cluster in state.clusters}
    dimensions.update(len(candidate.prototype) for candidate in validated)
    if len(dimensions) > 1:
        raise HDSCInvariantError("all active and candidate prototypes must share one dimension")
    proposals, duplicate_count, duplicate_suppressed_mass = _collapse_candidates(validated)
    ranked = sorted(proposals, key=lambda item: (-item.score, item.key))
    positive = [proposal for proposal in ranked if proposal.score > 0.0]
    selected = positive[: config.candidate_top_k]
    proposal_mass = _proposal_distribution(selected)
    proposal_total_score = sum(proposal.score for proposal in selected)
    proposal_normalization_gain = (
        len(selected) * config.score_lipschitz_bound / proposal_total_score
        if proposal_total_score > 0.0
        else None
    )

    proposal_boundary_gap, proposal_tie_count = _proposal_boundary(
        ranked,
        config.candidate_top_k,
        config.mass_tolerance,
    )
    proposal_certified_radius = proposal_boundary_gap / (2.0 * config.score_lipschitz_bound)
    cluster_assignment_margin = _cluster_assignment_margin(validated)

    previous_by_key = {cluster.key: cluster for cluster in state.clusters}
    prototype_by_key = {cluster.key: cluster.prototype for cluster in state.clusters}
    metadata_by_key = {
        proposal.key: (proposal.cluster_id, proposal.cluster_revision, proposal.prototype)
        for proposal in proposals
    }
    for key, (_cluster_id, _revision, prototype) in metadata_by_key.items():
        if key in prototype_by_key and not _prototype_equal(
            prototype_by_key[key], prototype, config.mass_tolerance
        ):
            raise HDSCInvariantError("candidate prototype conflicts with active cluster revision")
        prototype_by_key[key] = prototype

    mixed: dict[str, float] = {
        key: config.retention * cluster.mass for key, cluster in previous_by_key.items()
    }
    null_mass = config.retention * state.null_mass + (
        1.0 - config.retention - config.injection_rate
    )
    if proposal_mass:
        for key, mass in proposal_mass.items():
            mixed[key] = mixed.get(key, 0.0) + config.injection_rate * mass
    else:
        null_mass += config.injection_rate

    inactive_mass = sum(mass for mass in mixed.values() if mass <= config.mass_tolerance)
    null_mass += inactive_mass
    positive_mixed = [(key, mass) for key, mass in mixed.items() if mass > config.mass_tolerance]
    ranked_mass = sorted(
        positive_mixed,
        key=lambda item: (
            -_capacity_priority(
                item[0],
                item[1],
                previous_by_key,
                config.hysteresis,
            ),
            item[0],
        ),
    )
    kept = ranked_mass[: config.capacity]
    dropped = ranked_mass[config.capacity :]
    dropped_mass = sum(mass for _key, mass in dropped)
    null_mass += dropped_mass

    kept_keys = {key for key, _mass in kept}
    previous_keys = set(previous_by_key)
    union = kept_keys | previous_keys
    support_churn = len(kept_keys ^ previous_keys) / max(1, len(union))
    replacement_count = min(len(kept_keys - previous_keys), len(previous_keys - kept_keys))
    swapped_mass = sum(mass for key, mass in kept if key not in previous_keys) + sum(
        cluster.mass for key, cluster in previous_by_key.items() if key not in kept_keys
    )

    next_clusters: list[H2ActiveCluster] = []
    for key, mass in sorted(kept, key=lambda item: item[0]):
        if mass < -config.mass_tolerance:
            raise HDSCInvariantError("active update produced negative mass")
        if key in metadata_by_key:
            cluster_id, revision, prototype = metadata_by_key[key]
        elif key in previous_by_key:
            previous = previous_by_key[key]
            cluster_id, revision, prototype = (
                previous.cluster_id,
                previous.cluster_revision,
                previous.prototype,
            )
        else:
            raise HDSCInvariantError("active cluster metadata is missing")
        next_clusters.append(
            H2ActiveCluster(
                cluster_id=cluster_id,
                cluster_revision=revision,
                prototype=prototype,
                prototype_hash=_prototype_hash(prototype),
                mass=max(0.0, mass),
            )
        )

    total_mass = null_mass + sum(cluster.mass for cluster in next_clusters)
    mass_residual = 1.0 - total_mass
    if abs(mass_residual) > config.mass_tolerance:
        raise HDSCInvariantError("bounded active update violated total mass")
    null_mass += mass_residual
    if null_mass < -config.mass_tolerance:
        raise HDSCInvariantError("bounded active update produced negative null mass")
    null_mass = max(0.0, null_mass)

    next_state = H2State(
        step=state.step + 1,
        clusters=tuple(next_clusters),
        null_mass=null_mass,
        config_hash=config.config_hash,
    )
    capacity_boundary_gap = _capacity_boundary(
        ranked_mass,
        config.capacity,
        previous_by_key,
        config.hysteresis,
    )
    capacity_source_gain = config.retention + config.injection_rate * (
        proposal_normalization_gain or 0.0
    )
    capacity_certified_radius = (
        capacity_boundary_gap / (2.0 * capacity_source_gain) if capacity_source_gain > 0.0 else 0.0
    )
    certified_radius = min(
        proposal_certified_radius,
        capacity_certified_radius,
        cluster_assignment_margin if cluster_assignment_margin is not None else 0.0,
    )
    gain_matrix, gain_radius, gain_slack, certificate_status, certificate_reason = (
        _small_gain_certificate(
            gains,
            config,
            proposal_boundary_gap=proposal_boundary_gap,
            capacity_boundary_gap=capacity_boundary_gap,
            cluster_assignment_margin=cluster_assignment_margin,
            certified_radius=certified_radius,
        )
    )
    all_masses = [null_mass, *(cluster.mass for cluster in next_clusters)]
    audit = H2Audit(
        model_id=MODEL_ID,
        claim_level=CLAIM_LEVEL,
        modeled_scale=MODELED_SCALE,
        microstate_status=MICROSTATE_STATUS,
        transport_model=TRANSPORT_MODEL,
        shadow_only=True,
        config_hash=config.config_hash,
        semantic_partition_id=config.semantic_partition_id,
        previous_state_hash=state.state_hash,
        next_state_hash=next_state.state_hash,
        previous_replay_hash=state.replay_hash,
        next_replay_hash=next_state.replay_hash,
        archive_trace_count=archive_trace_count,
        capacity=config.capacity,
        active_before=len(state.clusters),
        active_after=len(next_clusters),
        candidate_count=len(validated),
        unique_cluster_count=len(proposals),
        duplicate_count=duplicate_count,
        proposal_count=len(selected),
        proposal_total_score=proposal_total_score,
        proposal_normalization_gain=proposal_normalization_gain,
        proposal_boundary_gap=proposal_boundary_gap,
        capacity_boundary_gap=capacity_boundary_gap,
        cluster_assignment_margin=cluster_assignment_margin,
        proposal_tie_count=proposal_tie_count,
        support_churn=support_churn,
        replacement_count=replacement_count,
        dropped_mass=dropped_mass,
        duplicate_suppressed_mass=duplicate_suppressed_mass,
        null_mass_before=state.null_mass,
        null_mass_after=null_mass,
        total_mass=null_mass + sum(cluster.mass for cluster in next_clusters),
        mass_residual=1.0 - (null_mass + sum(cluster.mass for cluster in next_clusters)),
        minimum_mass=min(all_masses),
        retention=config.retention,
        injection_rate=config.injection_rate,
        local_memory_contraction_bound=config.retention,
        semantic_jump_bound=min(2.0, 2.0 * swapped_mass),
        proposal_certified_radius=proposal_certified_radius,
        capacity_certified_radius=capacity_certified_radius,
        certified_radius=certified_radius,
        small_gain_matrix=gain_matrix,
        small_gain_spectral_radius=gain_radius,
        small_gain_margin=gain_slack,
        response_signal_gain=(gains.response_signal if gains is not None else None),
        response_memory_gain=(gains.response_memory if gains is not None else None),
        user_feedback_gain=(gains.user_feedback if gains is not None else None),
        proposal_signal_gain=(gains.proposal_signal if gains is not None else None),
        proposal_response_gain=(gains.proposal_response if gains is not None else None),
        proposal_memory_gain=(gains.proposal_memory if gains is not None else None),
        user_gain_source=(gains.user_gain_source if gains is not None else "unmeasured"),
        certificate_scope=CERTIFICATE_SCOPE,
        support_metric=SUPPORT_METRIC,
        certificate_status=certificate_status,
        certificate_reason=certificate_reason,
    )
    return H2Result(
        state=next_state,
        proposal_mass=tuple(sorted(proposal_mass.items())),
        audit=audit,
    )


def state_total_variation(left: H2State, right: H2State) -> float:
    """Return total-variation distance over cluster keys plus the null slot."""
    left_mass = {cluster.key: cluster.mass for cluster in left.clusters}
    right_mass = {cluster.key: cluster.mass for cluster in right.clusters}
    keys = left_mass.keys() | right_mass.keys()
    absolute = abs(left.null_mass - right.null_mass) + sum(
        abs(left_mass.get(key, 0.0) - right_mass.get(key, 0.0)) for key in keys
    )
    return 0.5 * absolute


def _validate_state(state: H2State, config: H2Config) -> None:
    if state.config_hash != config.config_hash:
        raise HDSCInvariantError("state config hash does not match H2 configuration")
    if state.step < 0:
        raise HDSCInvariantError("state step must be non-negative")
    if len(state.clusters) > config.capacity:
        raise HDSCInvariantError("state exceeds configured active capacity")
    if not math.isfinite(state.null_mass) or state.null_mass < -config.mass_tolerance:
        raise HDSCInvariantError("state null mass must be finite and non-negative")
    keys: set[str] = set()
    dimensions: set[int] = set()
    total = state.null_mass
    for cluster in state.clusters:
        if cluster.key in keys:
            raise HDSCInvariantError("state contains duplicate cluster revisions")
        keys.add(cluster.key)
        _required(cluster.cluster_id, "cluster_id")
        _required(cluster.cluster_revision, "cluster_revision")
        prototype = _validate_prototype(cluster.prototype, config.mass_tolerance)
        dimensions.add(len(prototype))
        if cluster.prototype_hash != _prototype_hash(prototype):
            raise HDSCInvariantError("state prototype hash mismatch")
        if not math.isfinite(cluster.mass) or cluster.mass < -config.mass_tolerance:
            raise HDSCInvariantError("state cluster mass must be finite and non-negative")
        total += cluster.mass
    if abs(total - 1.0) > config.mass_tolerance:
        raise HDSCInvariantError("state mass must sum to one")
    if len(dimensions) > 1:
        raise HDSCInvariantError("state prototypes must share one dimension")


def _validate_candidate(candidate: H2Candidate, tolerance: float) -> H2Candidate:
    _required(candidate.trace_id, "trace_id")
    _required(candidate.semantic_fingerprint, "semantic_fingerprint")
    _required(candidate.cluster_id, "cluster_id")
    _required(candidate.cluster_revision, "cluster_revision")
    if not 0.0 <= candidate.relevance <= 1.0 or not math.isfinite(candidate.relevance):
        raise HDSCInvariantError("candidate relevance must be finite and in [0, 1]")
    if not 0.0 <= candidate.novelty <= 1.0 or not math.isfinite(candidate.novelty):
        raise HDSCInvariantError("candidate novelty must be finite and in [0, 1]")
    if candidate.cluster_assignment_margin is not None and (
        candidate.cluster_assignment_margin < 0.0
        or not math.isfinite(candidate.cluster_assignment_margin)
    ):
        raise HDSCInvariantError(
            "candidate cluster_assignment_margin must be finite and non-negative"
        )
    prototype = _validate_prototype(candidate.prototype, tolerance)
    return H2Candidate(
        trace_id=candidate.trace_id,
        semantic_fingerprint=candidate.semantic_fingerprint,
        cluster_id=candidate.cluster_id,
        cluster_revision=candidate.cluster_revision,
        prototype=prototype,
        relevance=candidate.relevance,
        novelty=candidate.novelty,
        cluster_assignment_margin=candidate.cluster_assignment_margin,
    )


def _validate_prototype(prototype: Sequence[float], tolerance: float) -> tuple[float, ...]:
    if not prototype:
        raise HDSCInvariantError("candidate prototype must not be empty")
    values = tuple(float(value) for value in prototype)
    if not all(math.isfinite(value) for value in values):
        raise HDSCInvariantError("candidate prototype must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > tolerance:
        raise HDSCInvariantError("candidate prototype must have unit norm")
    return values


def _collapse_candidates(
    candidates: Sequence[H2Candidate],
) -> tuple[list[_ClusterProposal], int, float]:
    fingerprints: dict[str, str] = {}
    grouped: dict[str, list[H2Candidate]] = {}
    for candidate in candidates:
        key = _cluster_key(candidate.cluster_id, candidate.cluster_revision)
        previous_key = fingerprints.get(candidate.semantic_fingerprint)
        if previous_key is not None and previous_key != key:
            raise HDSCInvariantError("semantic fingerprint maps to multiple cluster revisions")
        fingerprints[candidate.semantic_fingerprint] = key
        grouped.setdefault(key, []).append(candidate)

    proposals: list[_ClusterProposal] = []
    duplicate_count = 0
    duplicate_suppressed_mass = 0.0
    for key, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda item: (-(item.relevance * item.novelty), item.trace_id),
        )
        representative = ordered[0]
        for member in members[1:]:
            if not _prototype_equal(representative.prototype, member.prototype, 1e-10):
                raise HDSCInvariantError("one cluster revision contains conflicting prototypes")
        scores = [member.relevance * member.novelty for member in members]
        score = max(scores)
        duplicate_count += max(0, len(members) - 1)
        duplicate_suppressed_mass += max(0.0, sum(scores) - score)
        proposals.append(
            _ClusterProposal(
                key=key,
                cluster_id=representative.cluster_id,
                cluster_revision=representative.cluster_revision,
                prototype=representative.prototype,
                score=score,
                member_count=len(members),
            )
        )
    return proposals, duplicate_count, duplicate_suppressed_mass


def _proposal_distribution(selected: Sequence[_ClusterProposal]) -> dict[str, float]:
    total = sum(proposal.score for proposal in selected)
    if total <= 0.0:
        return {}
    return {proposal.key: proposal.score / total for proposal in selected}


def _proposal_boundary(
    ranked: Sequence[_ClusterProposal],
    top_k: int,
    tolerance: float,
) -> tuple[float, int]:
    if not ranked:
        return 0.0, 0
    positive_count = sum(proposal.score > 0.0 for proposal in ranked)
    selected_count = min(positive_count, top_k)
    if selected_count == 0:
        return 0.0, sum(proposal.score <= tolerance for proposal in ranked)
    inside = ranked[selected_count - 1].score
    if selected_count < top_k and len(ranked) > selected_count:
        return 0.0, sum(proposal.score <= tolerance for proposal in ranked)
    outside = ranked[selected_count].score if len(ranked) > selected_count else 0.0
    gap = max(0.0, inside - outside)
    tie_count = sum(abs(proposal.score - inside) <= tolerance for proposal in ranked)
    return gap, tie_count if gap <= tolerance else 0


def _cluster_assignment_margin(candidates: Sequence[H2Candidate]) -> float | None:
    if not candidates or any(
        candidate.cluster_assignment_margin is None for candidate in candidates
    ):
        return None
    return min(
        candidate.cluster_assignment_margin
        for candidate in candidates
        if candidate.cluster_assignment_margin is not None
    )


def _capacity_priority(
    key: str,
    mass: float,
    incumbents: dict[str, H2ActiveCluster],
    hysteresis: float,
) -> float:
    return mass + (hysteresis if key in incumbents else 0.0)


def _capacity_boundary(
    ranked_mass: Sequence[tuple[str, float]],
    capacity: int,
    incumbents: dict[str, H2ActiveCluster],
    hysteresis: float,
) -> float:
    if not ranked_mass:
        return 0.0
    if len(ranked_mass) <= capacity:
        return min(mass for _key, mass in ranked_mass)
    inside_key, inside_mass = ranked_mass[capacity - 1]
    outside_key, outside_mass = ranked_mass[capacity]
    inside = _capacity_priority(inside_key, inside_mass, incumbents, hysteresis)
    outside = _capacity_priority(outside_key, outside_mass, incumbents, hysteresis)
    return max(0.0, inside - outside)


def _small_gain_certificate(
    gains: H2GainBounds | None,
    config: H2Config,
    *,
    proposal_boundary_gap: float,
    capacity_boundary_gap: float,
    cluster_assignment_margin: float | None,
    certified_radius: float,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]] | None,
    float | None,
    float | None,
    CertificateStatus,
    str,
]:
    if proposal_boundary_gap <= config.mass_tolerance:
        return None, None, None, "abstain", "proposal support has no positive local margin"
    if capacity_boundary_gap <= config.mass_tolerance:
        return None, None, None, "abstain", "capacity support has no positive local margin"
    if cluster_assignment_margin is None or cluster_assignment_margin <= config.mass_tolerance:
        return None, None, None, "abstain", "cluster assignment margin is unmeasured or zero"
    if certified_radius <= config.mass_tolerance:
        return None, None, None, "abstain", "combined support radius is below tolerance"
    if gains is None:
        return None, None, None, "abstain", "closed-loop gain bounds are unmeasured"
    if gains.user_gain_source.strip().casefold() in {"", "unmeasured", "unknown"}:
        return None, None, None, "abstain", "user-feedback gain source is unmeasured"
    raw = (
        gains.response_signal,
        gains.response_memory,
        gains.user_feedback,
        gains.proposal_signal,
        gains.proposal_response,
        gains.proposal_memory,
    )
    if any(value is None for value in raw):
        return None, None, None, "abstain", "closed-loop gain bounds are incomplete"
    values = tuple(float(value) for value in raw if value is not None)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        return None, None, None, "abstain", "closed-loop gains are non-finite or negative"
    a, b, user, proposal_signal, proposal_response, proposal_memory = values
    try:
        c = config.injection_rate * proposal_signal
        d = config.retention + config.injection_rate * proposal_memory
        e = config.injection_rate * proposal_response
        matrix = ((user * a, user * b), (c + e * a, d + e * b))
    except OverflowError:
        return None, None, None, "abstain", "closed-loop gain composition overflowed"
    if any(not math.isfinite(value) for row in matrix for value in row):
        return None, None, None, "abstain", "closed-loop gain composition overflowed"
    try:
        eigenvalues = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None, None, None, "abstain", "closed-loop spectral solve did not converge"
    if not np.all(np.isfinite(eigenvalues)):
        return None, None, None, "abstain", "closed-loop spectral result is non-finite"
    radius = float(np.max(np.abs(eigenvalues)))
    threshold = 1.0 - config.gain_margin
    slack = threshold - radius
    if radius < threshold:
        return matrix, radius, slack, "pass", "conditional small-gain margin is positive"
    return matrix, radius, slack, "fail", "conditional small-gain margin is exhausted"


def _cluster_key(cluster_id: str, revision: str) -> str:
    return f"{cluster_id}\x1f{revision}"


def _prototype_hash(prototype: Sequence[float]) -> str:
    payload = json.dumps(list(prototype), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _prototype_equal(left: Sequence[float], right: Sequence[float], tolerance: float) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= tolerance for a, b in zip(left, right, strict=True)
    )


def _state_hash(state: H2State) -> str:
    payload = {
        "clusters": [
            {
                "key": cluster.key,
                "prototype_hash": cluster.prototype_hash,
                "mass": cluster.mass,
            }
            for cluster in sorted(state.clusters, key=lambda item: item.key)
        ],
        "null_mass": state.null_mass,
        "config_hash": state.config_hash,
    }
    return _stable_hash(payload)


def _replay_hash(state: H2State) -> str:
    return _stable_hash({"step": state.step, "state_hash": state.state_hash})


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _required(value: str, name: str) -> None:
    if not value.strip():
        raise HDSCInvariantError(f"{name} must not be empty")


__all__ = [
    "CERTIFICATE_SCOPE",
    "CLAIM_LEVEL",
    "MICROSTATE_STATUS",
    "MODELED_SCALE",
    "MODEL_ID",
    "SUPPORT_METRIC",
    "TRANSPORT_MODEL",
    "H2ActiveCluster",
    "H2Audit",
    "H2Candidate",
    "H2Config",
    "H2GainBounds",
    "H2Result",
    "H2State",
    "bounded_active_shadow_step",
    "state_total_variation",
]
