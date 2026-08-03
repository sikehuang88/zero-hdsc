# HDSC Cross-Disciplinary Validation Framework

Date: 2026-07-26
Status: Normative research gate

## 1. Name And Scope

The project is named **Hyperdimensional Space Computing (HDSC)**, Chinese:
**超维度空间计算**.

HDSC is a research program for computation over persistent high-dimensional
trace geometry. It succeeds the Stateless Space Activation prototype. The
current one-hop propagation implementation is frozen as `legacy-ssa-a0` for
replay, ablation, and falsification.

HDSC does not claim that software activation scores are physical energy. It
also does not claim equivalence to established Hyperdimensional Computing
(HDC) or Vector Symbolic Architectures (VSA) merely because embeddings are
high-dimensional.

## 2. Claim Levels

Every paper, metric, UI label, and code module must identify its claim level.

| Level | Scale | Permitted claim | Required evidence |
|---|---|---|---|
| C0-M | Macro | Computational invariant of a coarse variable | Proof plus property tests |
| C0-micro | Micro | Pathwise invariant for a complete microstate | Complete state definition, fixed randomness, transition tests |
| C1-M | Macro | Thermodynamic structural analogy | Conservation/dissipation proof, symmetric transport, entropy-production audit |
| C2 | Physical | Physical thermodynamic model | Calibrated mapping to joules, hardware telemetry, uncertainty analysis, replicated measurement |

`legacy-ssa-a0` is C0-M only. HDSC-H1 targets C1-M for a fixed graph. HDSC-H2
targets C0-M bounded-state and conditional small-gain claims. None of these
labels grants a microstate-path claim. C2 requires
measured physical energy and cannot be inferred from token counts, activation
scores, embedding norms, or metaphorical organism state.

## 3. Ontology

The following quantities must remain distinct:

- semantic affinity: dimensionless retrieval evidence;
- activation mass: a conserved computational budget at C0/C1;
- information entropy: a distributional property;
- organism state: a bounded behavioral control variable;
- physical energy: measured in joules;
- temperature: defined only after specifying a physical or formal capacity;
- importance: write-time epistemic/behavioral salience, not automatically heat
  capacity;
- freshness: a temporal prior, not stored thermal energy.

No equation may substitute one of these quantities for another without an
explicit dimensional map and validation protocol.

### 3.1 Macro-Micro Closure Boundary

Let `X_t` be the complete microstate and let `C` be a many-to-one coarse-graining
map. HDSC-H1 models only:

```text
u_t = C(X_t)
```

A closed macro transition `F_delta` is justified only when every pair of
microstates with the same macrostate has the same conditional coarse future:

```text
C(x) = C(x')
=> E[C(X_(t+delta)) | X_t=x] = E[C(X_(t+delta)) | X_t=x']
```

This lumpability/closure condition has not been established for embeddings,
graph construction, ranking ties, asynchronous events, provider hidden state,
or LLM sampling. When it fails, the macro equation requires unresolved forcing
and memory, for example a generalized Langevin form:

```text
du/dt = -(L + kappa*I)u
        + integral_0^t M(t-s)u(s) ds
        + eta(t)
```

`M` is a learned memory kernel and `eta` is unresolved microstate forcing. No
fluctuation-dissipation relation is asserted until both quantities are measured
and a physical temperature is defined. A positive component `u_i > 0` or a
nonzero perturbation alone does not establish unpredictability. Path divergence
requires evidence such as a positive Lyapunov exponent, a stochastic transition
kernel, or non-identifiability under the coarse map.

Required micro/macro diagnostics:

- ensemble replay from identical `u` with varied hidden states and seeds;
- conditional variance of `C(X_(t+delta))` given `u_t`;
- residual autocorrelation after fitting the macro transition;
- Chapman-Kolmogorov/lumpability tests for the proposed macro state;
- finite-time Lyapunov estimates under controlled perturbations;
- separate uncertainty bands for source allocation, graph construction, and LLM output.

## 4. Provisional HDSC-H1 Kernel

This section defines a candidate to test, not an approved production model.

For every trace, compute source affinity over the complete trace set:

```text
g_i = positive_kernel(cos(query, vector_i), theta)^p
      * freshness_i^alpha
      * importance_i^beta
```

Choose the top `m` source traces after full scoring and allocate a fixed source
budget:

```text
q_i = Q0 * g_i / sum(g_j for j in source_set)
sum(q) = Q0
```

Build a non-negative symmetric conductance matrix `K`, then:

```text
D_ii = sum_j K_ij
L = D - K
u_diffused(t) = exp(-t * L) * q
u_final(t) = exp(-kappa * t) * u_diffused(t)
```

`q` is computational activation mass, not physical energy. `t` is a formal
diffusion duration and `kappa` is a dimensionless environment-leak rate. The
leak term is expressed exponentially so repeated steps form a semigroup; a
free-standing `loss_ratio` would not compose consistently.

Required identities:

```text
K = K.T
L = L.T
L @ 1 = 0
u_diffused >= 0
u_final >= 0
sum(u_diffused) = Q0
sum(u_final) = exp(-kappa * t) * Q0
dissipated = (1 - exp(-kappa * t)) * Q0
```

The kernel uses uniform formal capacity initially. Importance remains a source
prior. A heterogeneous capacity matrix enters only after experiments show that
it improves prediction without double-counting importance.

For large sparse graphs, implementation should use a Krylov `expm_multiply`
operation or an M-matrix implicit solver, not materialize a dense matrix
exponential.

### 4.1 HDSC-H2 Bounded Active State

H2 separates the growing replay archive `H_t` from the fixed-capacity active
measure `mu_t`:

```text
H_(t+1) = H_t append episode_t
mu_t = null_mass * delta_null + sum(cluster_mass[c] * delta_prototype[c])
active_cluster_count <= capacity
sum(mu_t) = 1
```

Candidates are assigned to a fixed semantic codebook independent of archive
size. Each `(cluster_id, revision)` receives its maximum candidate score rather
than a sum. The proposal and update are:

```text
g_c = max(relevance_j * novelty_j for j in cluster c)
q_t = normalized top-K positive cluster scores, else delta_null
mu_hat = beta * mu_t + alpha * q_t + (1-beta-alpha) * delta_null
mu_(t+1) = capacity_projection_to_null(mu_hat)
```

Required constraints:

```text
0 <= beta < 1
0 <= alpha
beta + alpha <= 1
active_cluster_count <= capacity
all mass >= 0
total mass = 1
```

Capacity overflow moves to the null reservoir and is not redistributed over
surviving clusters. The runtime derives novelty from whether the versioned
semantic bucket is currently active, so chronological replay needs no
unarchived appraisal value. It intentionally keeps no archive-sized fingerprint
registry; an evicted bucket may later re-enter.

The local certificate audits three discontinuities: distance from normalized
embeddings to the nearest fixed-codebook hyperplane, the positive/top-K score
margin (including the zero-score entry boundary), and the capacity margin
computed with the same `mass + incumbent_hysteresis` priority used for
selection. On a fixed proposal support with total positive score `S` and `m`
members, score normalization has TV gain at most `m * L_score / S`. The capacity
radius divides its raw margin by twice `beta + alpha * normalization_gain`.
The reported radius belongs only to the declared max-product metric over
normalized-embedding distance, raw-score-source distance, and prior-state TV.
A missing/zero margin, an under-tolerance combined radius, incomplete or
unattributed closed-loop gains, or a non-finite gain yields `abstain`. H2 is
C0-M and does not inherit H1's C1-M label.

### 4.2 HDSC-H1D Directed Nonreversible Transport

H1 is retained only as a reversible symmetric reference. A directed topology
must not be replaced by `(R + R.T) / 2`, because that creates reverse edges and
removes net currents. H1D uses `R[target, source] >= 0` and the column generator:

```text
Q = R - diag(column_sum(R))
1.T @ Q = 0
P(t) = exp(t * Q)
u(t) = P(t) @ u(0)
```

Required computational properties are positivity, mass conservation, semigroup
composition, permutation equivariance, and L1 non-expansion. Symmetric-H1
Dirichlet, double-stochastic, uniform-stationary, and raw-Shannon conclusions do
not transfer to H1D.

With a validated positive stationary distribution `pi`, H1D checks contraction
of `KL(p || pi)`. A finite entropy-production diagnostic additionally requires
reciprocal transition support. One-way edges trigger abstention until an open
reservoir or hidden reverse-channel model is supplied. Physical thermodynamic
status remains uncalibrated.

## 5. Thermodynamics Gate

### 5.1 First-Law Structure

For a closed propagation step, internal transfer must cancel pairwise. For an
open step with a uniform environment sink:

```text
delta_internal = input_budget - dissipated_budget - exported_work
```

For HDSC-H1, `exported_work = 0` and the explicit sink gives the dissipated
term above. No token count, activation score, or embedding norm is silently
converted to joules.

Acceptance metrics:

- conservation residual below `1e-10` in float64 reference tests;
- non-negative node mass within numerical tolerance;
- source, sink, and work terms recorded separately;
- graph growth does not increase the per-query source budget.

### 5.2 Second-Law Structure

For symmetric pure diffusion, the Dirichlet energy must not increase:

```text
V(u) = 0.5 * u.T @ L @ u
dV/dt <= 0
```

The normalized pure-diffusion operator must be doubly stochastic on each
connected component, permitting a non-decreasing Shannon entropy test. Entropy
tests are diagnostics, not substitutes for physical entropy at C2.

### 5.3 Reciprocity

Reversible H1 transport must satisfy symmetric conductance. Directed topology
is routed to H1D's Markov generator and must document non-equilibrium currents.
It cannot be presented as passive heat conduction. Missing reverse support
requires explicit reservoir/hidden-channel accounting before finite entropy
production is claimed.

### 5.4 Numerical Stability

- exact reference solution on small graphs;
- sparse solver agreement within declared tolerance;
- permutation invariance;
- disconnected-component isolation;
- stable behavior for singleton, duplicate, hub, chain, ring, and adversarial
  graphs;
- no NaN, Inf, negative mass, or unbounded amplification.

## 6. Hyperdimensional-Computing Gate

Established HDC/VSA commonly evaluates binding, bundling, permutation,
superposition capacity, cleanup memory, noise robustness, and compositional
decoding. HDSC must either implement and benchmark such operations or state
that it is a high-dimensional spatial system rather than classical HDC.

Required comparisons:

- dense embedding retrieval;
- sparse lexical retrieval;
- graph RAG;
- HDC/VSA representation where applicable;
- legacy SSA-A0;
- HDSC-H1 diffusion with identical trace snapshots.
- HDSC-H1D directed transport on identical multi-relation snapshots.

## 7. Information-Theory Gate

- measure source-distribution entropy and post-diffusion entropy separately;
- report mutual information between activated traces and response claims;
- quantify duplicate-trace amplification;
- perform cluster-level source budgeting when repeated paraphrases dominate;
- separate retrieval diversity from sampling diversity;
- detect information leakage across conversations and source classes.

## 8. Cognitive And Memory Gate

The trace substrate, consolidated memory, relationship state, and self-beliefs
must remain distinct. Evaluation must cover:

- episodic versus semantic retrieval;
- temporal contiguity and semantic association;
- interference, reconsolidation, and contradiction;
- source monitoring and false-memory rate;
- identity hysteresis and counterevidence;
- behavior under reset, replay, and model replacement.

Human-likeness claims require preregistered behavioral experiments and cannot
be inferred from visually plausible activation maps.

## 9. Dynamical-Systems Gate

- prove boundedness of every persistent state variable;
- identify fixed points and possible attractors;
- construct a Lyapunov function or provide counterexamples;
- test perturbation sensitivity and long-horizon drift;
- separate stochastic LLM variance from deterministic substrate dynamics;
- measure feedback gain from belief to action to new evidence.

### 9.1 Closed-Loop And Growing-Space Gate

The append-only archive and the behavior-driving state are separate objects:

```text
archive: H_(t+1) = H_t append episode_t
active:  z_(t+1) = project_fixed_budget(
           (1-alpha) * transport(L_t, z_t) + alpha * inject(episode_t)
         )
```

Archive cardinality may grow linearly while active support, total activation
mass, per-cluster budget, and per-turn influence remain bounded. Acceptance
requires:

- a fixed active-node or active-cluster capacity independent of turn count;
- cluster-normalized source allocation so duplicate traces do not multiply a
  semantic region's budget;
- a declared upper bound on one-turn Laplacian change;
- hysteresis at active-set entry/exit boundaries;
- separate contraction targets for episodic state and evidence gates for
  identity state;
- paired interventions through both paths: previous response to next user
  signal, and previous response to archive/graph to next response;
- distributional sensitivity metrics for discrete stochastic text rather than
  a literal derivative of token strings;
- long-horizon tests reporting finite-time amplification, active-set churn,
  graph spectral-gap drift, duplicate amplification, and response-distribution
  distance.

The literal partial derivative of `r_t` with respect to `r_(t-1)` is zero when
`s_t`, active memory, and randomness are held fixed, because the response
function has no explicit previous-response argument. Historical influence is a
total derivative through two mediators:

```text
previous response -> next user signal -> next response
previous response -> written episode -> active memory/graph -> next response
```

If response sensitivity is bounded by gains `a,b`, user feedback by `u`, and
memory update by `c,d,e`, the joint perturbation matrix is:

```text
M = [[u*a,     u*b],
     [c+e*a, d+e*b]]
```

The time-invariant small-gain gate is `spectral_radius(M) < 1`. Because user
feedback is unknown and time-varying, every stability statement must declare
its assumed user-gain envelope and report empirical joint-gain estimates.

For a positive-semidefinite Laplacian, the Frechet sensitivity of the heat
operator satisfies `||D exp(-tau*L)[Delta L]||_2 <= tau ||Delta L||_2`.
This bounds the transport contribution only. Source selection, graph updates,
top-K discontinuities, user feedback, and generation remain separate measured
gains. A nonzero derivative denotes influence; amplification requires a gain
above one or a positive finite-time Lyapunov estimate.

## 10. Causal And Empirical Gate

Every claimed behavioral effect requires an intervention:

```text
same model + same sampling + same input + changed trace kernel
```

Minimum ablations:

- no trace context;
- semantic top-K only;
- legacy SSA-A0;
- HDSC-H1 without loss;
- HDSC-H1 without diffusion;
- shuffled links;
- randomized importance;
- frozen versus growing trace space.

Report confidence intervals, effect sizes, seed sensitivity, negative results,
and failure cases. A visually compelling single conversation is not evidence.

## 11. Promotion Process

1. Freeze `legacy-ssa-a0` fixtures, prompts, and outputs.
2. Preserve HDSC-H1 as the reversible fixed-graph C1-M reference.
3. Route directed multi-relation topology to H1D without symmetrization.
4. Implement HDSC-H2 behind `hdsc-h2-bounded-active-shadow`.
5. Pass H1/H1D transport properties and H2 capacity, mass, duplicate, permutation,
   contraction, zero-entry, cluster-boundary, normalization-gain,
   hysteresis-adjusted-capacity, and small-gain certificate properties.
6. Replay historical conversations and verify identical H2 state after restart;
   require `state.step == archive_count` and one compatible vector per trace.
7. Run H1D/H2 after episode writes; exclude them from serving retrieval and prompts.
8. Measure user-feedback and proposal gains with shared-randomness paired runs.
9. Compare legacy, H1, H1D, H2, and no-memory baselines on cloned archives.
10. Conduct a documented cross-disciplinary review.
11. Promote by explicit configuration only after every blocking gate passes.

Until promotion, the TUI must display the active engine identifier and label
legacy activation as an experimental baseline.
