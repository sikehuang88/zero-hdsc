# HDSC-H2 Bounded Active Shadow

Date: 2026-07-26  
Software: 0.3.0  
Model: `hdsc-h2-bounded-active-shadow`  
Claim level: C0-M computational invariants  
Serving status: shadow only, no prompt effect

## Delivered Theory

H2 separates the append-only replay archive from the bounded state that may
eventually mediate behavior:

```text
archive_(t+1) = archive_t append episode_t
mu_t = null_mass * delta_null + sum(cluster_mass[c] * delta_prototype[c])
active_cluster_count <= capacity
total_mass(mu_t) = 1
```

The update is:

```text
g[c] = max(relevance * novelty for candidate in cluster c)
q_t = normalized top-K positive g, else delta_null
mu_hat = beta * mu_t + alpha * q_t + (1-beta-alpha) * delta_null
mu_(t+1) = project_overflow_to_null(mu_hat)
```

- Overflow mass enters the null reservoir; retained mass is not renormalized.
- Exact duplicates cannot multiply a same-batch cluster proposal.
- Across turns, a currently active versioned bucket has zero innovation; an
  evicted bucket may re-enter. No archive-sized fingerprint registry is claimed.
- `state_hash` identifies the behavior measure. `replay_hash` additionally
  includes chronological step.
- Semantic partition identity includes embedding model/dimension, SimHash bits,
  codebook version, and deterministic hyperplanes.

## Local Certificate

The certificate scope is
`local-fixed-candidate-cluster-capacity-support`. It audits:

- SimHash distance to the nearest assignment hyperplane;
- positive/zero/top-K proposal score margin;
- capacity margin under the same `mass + incumbent_hysteresis` priority used by
  selection;
- proposal normalization gain `m * L_score / sum(selected_scores)`;
- the joint user-agent-memory small-gain spectral radius.

The combined support radius is defined only in the declared max-product metric
over normalized-embedding distance, raw-score-source distance, and prior-state
TV. The capacity component divides its margin by
`2 * (beta + alpha * proposal_normalization_gain)`.

A missing or zero margin, an under-tolerance radius, incomplete or unattributed
gain bounds, or non-finite gain data yields `abstain`. Hard selection is not
presented as globally continuous.

## Runtime Integration

- Legacy `legacy-ssa-a0` remains the only serving activation engine.
- H2 advances after a complete episode is committed.
- Startup rebuilds H2 by chronological archive replay.
- `state.step == archive_count` is required before the snapshot can show
  `passed`.
- A missing vector, semantic-partition mismatch, repository exception, or H2
  invariant error is contained in the shadow audit and does not abort the
  persisted turn or alter legacy prompt construction.
- TUI status separates archive/view counts, serving activation, H2 active mass,
  local gate, and closed-loop gate.
- `80x30` uses a complete 14-row compact map; `120x42` retains the complete
  18-row map, list, and trace detail.

## Verification

```text
uv run --no-sync pytest --cov=ssa --cov-report=term -q
  -> 386 passed, 89% total coverage
  -> active_space.py 93%, trace_space_service.py 96%

uv run --no-sync ruff check .
  -> All checks passed

uv run --no-sync mypy --strict src/ssa
  -> Success: no issues found in 45 source files

uv lock --check
  -> 142 packages resolved, lock current

git diff --check
  -> no whitespace errors

uv build
  -> hdsc-0.3.0.tar.gz and hdsc-0.3.0-py3-none-any.whl

isolated wheel: hdsc doctor --database :memory:
  -> 10 packaged migrations, schema v10, sqlite-vec loaded, status ok
```

## Research Boundary

This milestone proves software capacity, mass, replay, duplicate, isolation,
and local certificate contracts. It reports no user-feedback gain measurement,
behavioral effect, identity continuity result, or physical thermodynamic result.
H1 transport and H2 active state remain separate; their composition and any
prompt-path admission belong to HDSC-H3.
