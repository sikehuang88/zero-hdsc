# Hyperdimensional Space Computing (HDSC):
# A Cross-Disciplinary Research Program for Persistent Digital-Life Computation

> Draft v0.7 — 2026-07-26
> Status: Position paper / research preview. Software v0.5 implements the
> interactive prototype, HDSC-H1 reversible reference, HDSC-H1D directed
> transport reference, HDSC-H2 bounded
> active-space shadow. The YUANZI-5D recursive living-operator framework added
> in v0.7 is a mathematical proposal, not an implemented runtime or a claim of
> biological life; controlled behavioral and physical-energy results remain pending.
> Note on revision: v0.2 claimed "strict statelessness" and preliminary experimental evidence. Both claims were retracted after review. This version makes HDSC's transport boundary explicit and removes all unverified assertions.
> Rename note: HDSC succeeds the legacy Stateless Space Activation prototype.
> Sections retaining the term SSA describe the frozen `legacy-ssa-a0` baseline,
> not the validated HDSC propagation kernel. Thermodynamic compliance is not
> claimed for that baseline.

---

## Abstract

We introduce **Hyperdimensional Space Computing (HDSC)** as the successor
research program to Stateless Space Activation. HDSC studies persistent
digital-life computation over high-dimensional trace geometry while requiring
formal separation between computational quantities, information-theoretic
measures, and physically meaningful thermodynamic variables. The software
preserves SSA-A0 as a reproducible baseline, HDSC-H1 as an isolated reversible
transport reference, HDSC-H1D as a directed nonreversible Markov reference, and
HDSC-H2 as a fixed-capacity active-state candidate.
New kernels must pass cross-disciplinary validation before behavioral deployment.

Software v0.5 also adds P1 environment perception to the serving control path:
UTC events are projected onto a configured human clock and a deterministic
semantic-affect-context-time cross operator produces an immutable situation
snapshot before the main response. P1 is not the YUANZI-5D active-inference
loop and does not consume H1D or H2 shadow output.

We additionally propose **YUANZI-5D**, an unimplemented higher-order extension
that places normalized discrete active inference on a Markov-blanket boundary,
decomposes internal flow into weighted skew-symmetric circulation and
positive-semidefinite dissipation, and lets a slower meta-operator revise the
geometry governing a four-fiber perception-action-memory-prediction state. The
fifth coordinate is therefore a law or operator coordinate rather than another
content feature. This construction is intended to make self-loop, identity,
and non-equilibrium claims mathematically falsifiable; it does not establish
consciousness, biological status, or physical thermodynamic calibration.

We do not claim SSA eliminates state entirely—$\mathcal{S}$ is persistent and influences future outputs, which is a form of state. SSA's actual contribution is architectural: it replaces **in-place state mutation** with **append-only spatial memory** whose geometry (not a vector) shapes behavior. This shifts the locus of "personality" from a mutable vector to an immutable history, with consequences for reset semantics, drift, and emotion.

We clarify that SSA does **not** escape the state-behavior mapping formalism—behavior is still $E(s, A(s, \mathcal{S}), \xi)$ where $\xi$ is sampling noise. SSA's distinction is that $\mathcal{S}$ is high-dimensional, append-only, and spatially organized, versus the low-dimensional, mutable vectors of stateful approaches. Whether this distinction yields measurably different agent behavior is an open empirical question this paper poses but does not answer.

We position HDSC relative to prior work—Generative Agents' relevance×recency×importance scoring, A-MEM's associative linking, HippoRAG's graph retrieval, Synapse's spreading activation, the free-energy principle, and discrete active inference—and identify what, if anything, is novel. We preserve SSA-A0 for replay, retain HDSC-H1 as a falsifiable reversible reference, add HDSC-H1D for directed nonreversible topology without symmetrization, and retain HDSC-H2 as a bounded probability measure over semantic clusters with duplicate suppression, a null reservoir, local contraction, and conditional small-gain certification. YUANZI-5D is presented only as a candidate synthesis of these ingredients with a recursive operator coordinate.

**Keywords**: LLM agents, episodic memory, spatial retrieval, active inference, variational free energy, non-equilibrium steady state, recursive operators, position paper

---

## 1. Introduction

LLM-based agents with persistent memory have become a major research direction. Generative Agents [1] introduced memory streams with relevance-recency-importance retrieval; MemGPT [2] treated context windows as paged memory; commercial systems (Replika, Character.ai) maintain emotion and persona state.

A common pattern across these systems is **in-place state mutation**: a state variable $S_t$ is updated each turn ($S_{t+1} = f(S_t, \text{interaction}_t)$), and behavior is governed by $M(S_t, s_t)$. This raises practical issues:
- **Reset fragility**: clearing $S_t$ destroys accumulated character.
- **Drift**: $S_t$ evolves unpredictably under user influence and sampling noise.
- **Low-dimensional bottleneck**: $S_t$ is typically 10-100 dims, which may be insufficient for behavioral richness.

We propose **Stateless Space Activation (SSA)**, which replaces in-place mutation with an **append-only episodic-spatial memory** $\mathcal{S}$. The inference core itself holds no session state—each request independently activates traces from $\mathcal{S}$ and reasons over them. $\mathcal{S}$ persists and influences future responses, so SSA is not "stateless" in an absolute sense. Rather, the *locus* of persistence shifts from a mutable vector to an immutable, spatially-organized history.

**Honest positioning.** We are explicit about what SSA is and is not:
- SSA **is not** the first stateless or spatially-organized agent memory. Generative Agents [1] already combine relevance, recency, and importance. A-MEM [16] implements associative memory linking. HippoRAG [17] uses knowledge graphs for retrieval. Synapse [18] (Jan 2026) employs episodic-semantic memory with spreading activation. SSA's contribution, if any, is in the *specific combination* of strict session-statelessness in the inference core + append-only spatial geometry + background reflection—**not** in any single component.
- SSA **does not** eliminate the state-behavior mapping. Formally, SSA behavior is $E(s, A(s, \mathcal{S}), \xi)$, which is structurally analogous to $M(S_t, s_t) + \epsilon$. The difference is that $\mathcal{S}$ is high-dimensional and append-only, whereas $S_t$ is low-dimensional and mutable. Whether this difference is *consequential* is an empirical question.
- This paper **presents no behavioral or physical-energy results**. The software reference path exists, but it has not been promoted into the prompt chain or calibrated to joules. We describe the architecture, position it against prior work, and specify an executable experimental design. All claims about behavior remain hypotheses, not findings.

**Contributions** (revised, modest):
1. We articulate an architectural alternative—session-stateless inference + append-only spatial memory—that distinguishes *where* persistence lives (external append-only vs. internal mutable), even though both are "state" in the formal sense.
2. We propose a concrete instance of this architecture ($\langle \mathcal{S}, A, E, W, L \rangle$) with formal definitions.
3. We clarify the reset semantics distinction: SSA's inference core can be reset (cleared) while $\mathcal{S}$ persists, whereas stateful agents' resets typically clear $S_t$. We note this is a *design choice* in what "reset" means, not an inherent superiority.
4. We propose a factorized experimental design and discuss evaluation challenges for "lifelike" agent behavior.
5. We separate an unbounded replay archive from a fixed-capacity active measure,
   making archive integrity and behavioral drift resistance independently testable.
6. We separate reversible symmetric diffusion from directed nonreversible
   transport, preventing asymmetric topology from inheriting invalid H1 claims.
7. We formulate the unimplemented YUANZI-5D candidate: a normalized active-
   inference loop with skew circulation, dissipative descent, an identity
   invariant candidate, and a slower operator-learning coordinate.

---

## 2. Related Work

### 2.1 Memory-Augmented LLM Agents

**Generative Agents** [1] introduced the Memory Stream: observations recorded chronologically, retrieved by a combined score of relevance, recency (exponential decay), and importance (LLM-rated). Periodic reflection synthesizes observations into higher-level insights. Planning generates schedules.

SSA shares with Generative Agents: relevance × recency × importance scoring, episodic memory, and reflection-like background processing. **SSA does not claim novelty in the scoring formula**—it is essentially the same as Generative Agents'. The differences are:
- Generative Agents organize memory as a temporal stream; SSA emphasizes spatial organization (clustering by semantic proximity).
- Generative Agents' reflection produces persistent state (reflection nodes that are themselves retrieved); SSA's reflection writes traces to $\mathcal{S}$ but maintains no separate reflection hierarchy.
- Generative Agents use preset personas; SSA does not (though this is a configuration choice, not an architectural necessity).

These differences are matters of degree and emphasis, not categorical.

**MemGPT** [2] treats the LLM context window as memory with OS-style paging between main context and external storage. Working memory is persistent state, updated in-place. SSA's inference core has no working memory—all context is reconstructed from $\mathcal{S}$ each request.

**A-MEM** [16] (Agentic Memory) implements an agentic memory system with associative linking between memories and neighbor retrieval—conceptually similar to SSA's "radiation activation." SSA's radiation is a simpler variant (fixed-depth KNN expansion) without A-MEM's dynamic link maintenance.

**HippoRAG** [17] uses a knowledge graph extracted from passages for associative retrieval, inspired by hippocampal indexing theory. SSA's spatial activation is a simpler (vector-space) analog without explicit graph structure.

**Synapse** [18] (Jan 2026) employs episodic-semantic memory with spreading activation—very close to SSA's design. SSA's distinctiveness from Synapse, if any, lies in the strict session-statelessness of the inference core and the specific formalization. **We do not claim priority over Synapse.**

**Survey** [3] (Zeyu Zhang et al.) catalogs memory mechanisms, providing taxonomy against which SSA positions itself.

### 2.2 Retrieval-Augmented Generation

RAG [4] retrieves external knowledge. SSA shares RAG's per-request statelessness but differs in retrieving *experienced traces* rather than *external documents*, and in including temporal decay and importance weighting. Self-RAG [5] and Adaptive-RAG [6]—which adapts retrieval based on **question complexity** (not Q-learning as we previously misstated)—add adaptivity.

### 2.3 Affective Computing

Picard [7], Plutchik [8], and appraisal theory [9] provide background on emotion. SSA's claim that emotion can "emerge" rather than be "performed" is a design hypothesis, not a proven result.

### 2.4 Philosophical Foundations

Enactivism [10], Extended Mind [11], and Heidegger's temporality [12] provide philosophical context. We cite these as *resonances*, not as formal grounding—philosophical alignment does not constitute technical novelty.

### 2.5 AI Companionship Products

Replika, Character.ai, Pi, and Chinese products (Xingye, Zhumengdao, Xiaoice) employ persona prompts + memory augmentation. SSA differs in configuration (no preset persona) but the architectural boundary is blurrier than we previously claimed, as these products also maintain conversation history that functions similarly to traces.

### 2.6 Free-Energy Principle, Active Inference, and Non-Equilibrium Flow

The free-energy principle (FEP) characterizes variational inference through an
upper bound on surprisal and has been proposed as a unifying description of
self-organizing systems [15]. Discrete active inference extends this account
from perception to policy selection by minimizing expected free energy under a
generative model with likelihood, transition, preference, initial-state, and
habit terms [19]. A Markov blanket is a conditional-independence structure; it
is not established merely by naming software modules as sensory, active,
internal, and external states.

Nonreversible Markov processes can sustain stationary probability currents
when detailed balance is broken [20]. More general metriplectic and GENERIC
formulations separate antisymmetric conservative flow from symmetric
dissipative flow [21]. YUANZI-5D draws on these structures, but its proposed
combination with an append-only LLM trace archive and an operator-valued fifth
coordinate has not been implemented or compared empirically. Variational free
energy is information-theoretic; equating it with Helmholtz, Gibbs, or measured
hardware free energy would require additional physical assumptions and
calibration.

---

## 3. What SSA Is (and Is Not)

Given the density of prior art, we clarify SSA's precise contribution.

### 3.1 SSA Is Not "Strictly Stateless"

$\mathcal{S}$ is persistent and influences future outputs. By any standard definition, this is state. SSA's inference *core* (the $A \rightarrow E$ pipeline) is session-stateless—it reconstructs all context from $\mathcal{S}$ per request—but the system as a whole is not stateless.

The accurate description: **session-stateless inference core + external persistent episodic memory**.

### 3.2 SSA Does Not Escape the State-Behavior Mapping

Formally, SSA behavior is:

$$\text{behavior} = E\Big(s, A(s, \mathcal{S}), \xi\Big)$$

where $\xi$ is LLM sampling randomness. This is structurally equivalent to $M(\mathcal{S}, s, \xi)$. SSA has the same form as stateful approaches; the difference is in the *nature* of the first argument:
- Stateful: $\mathcal{S} \equiv S_t$, a low-dimensional mutable vector, updated in-place.
- SSA: $\mathcal{S}$ is a high-dimensional append-only set of traces.

**Whether this difference is consequential is an open question.** Our previous claim that SSA "eliminates the deterministic mapping defect" was overstated. The defect (if it is one) applies equally to SSA. What differs is the *capacity* and *mutability* of the state representation.

### 3.3 What SSA Actually Proposes

SSA proposes that agent persistence should be:
1. **Append-only**: traces are written, never modified. State changes by *addition*, not *mutation*.
2. **High-dimensional**: $\mathcal{S}$ lives in $\mathbb{R}^{d \times N}$ (d=512, N=grows), not $\mathbb{R}^{10}$.
3. **Spatially organized**: retrieval is by semantic proximity, not temporal sequence.
4. **Reconstructed per-request**: the inference core holds nothing between requests.

These are *design choices* with trade-offs, not proven superiorities.

### 3.4 The "Deterministic Mapping Defect" Reformulated

We previously defined a "defect" where identical state yields identical behavior. **This applies to SSA too**: identical $\mathcal{S}$ and $s$ yield $E(s, A(s, \mathcal{S}), \xi)$, which varies only by $\xi$—exactly the structure we criticized.

The honest reformulation: SSA's variability advantage, *if any*, comes from $\mathcal{S}$ growing over time (so $\mathcal{S}_{t_2} \neq \mathcal{S}_{t_1}$), not from escaping the mapping. But:
- Stateful agents' $S_t$ also changes over time.
- $\mathcal{S}$ growing does not guarantee $A(s, \mathcal{S}_{t_2}) \neq A(s, \mathcal{S}_{t_1})$—new traces may not enter the top-K.

**We retract the claim that SSA provably avoids the defect.** The question is empirical: does append-only high-dimensional memory yield more behavioral variability than mutable low-dimensional state, *in practice*?

---

## 4. Method: Stateless Space Activation

### 4.1 Formal Definition

SSA is defined as $\langle \mathcal{S}, A, E, W, L \rangle$.

#### 4.1.1 Trace Space $\mathcal{S}$

$\mathcal{S}$ is a set of traces. Each trace is a 5-tuple:

$$t_i = (c_i, v_i, \tau_i, \rho_i, \eta_i)$$

- $c_i$: raw content (text)—**added in v0.3; previous definition omitted this, but $E$ requires raw content**
- $v_i \in \mathbb{R}^d$: semantic vector, $v_i = \phi(c_i)$
- $\tau_i \in \mathbb{R}^+$: write timestamp
- $\rho_i(t) = e^{-\lambda(t - \tau_i)}$: freshness at query time $t$—**note: this is a function of query time, not a stored value**
- $\eta_i \in (0, 1]$: importance, assessed at write time

**Properties**:
- Append-only: traces are added, never modified. $\rho_i$ is computed at query time, not stored, so "trace immutability" refers to $(c_i, v_i, \tau_i, \eta_i)$—the freshness is a derived quantity.
- Forgetting: an explicit policy may *delete* traces (not modify them). This conflicts with "monotonic growth"—we resolve this by saying growth is the default; forgetting is an opt-in policy that violates it for practical reasons (storage limits).

#### 4.1.2 Activation Function $A$

$$A(s, \mathcal{S}, t) \rightarrow \mathcal{T}_{\text{act}}$$

**Activation score** (computed at query time $t$):

$$\alpha(s, t_i, t) = \cos(s, v_i) \cdot \rho_i(t) \cdot \eta_i$$

**Note on the freshness factor**: $\rho_i(t) = e^{-\lambda(t - \tau_i)} = e^{-\lambda t} \cdot e^{\lambda \tau_i}$. For a fixed query time $t$, the factor $e^{-\lambda t}$ is constant across all candidates and does not affect ranking. **The ranking is determined by $e^{\lambda \tau_i}$, i.e., more recent traces rank higher.** The constant factor cancels. This corrects our previous implicit assumption.

**Selection**: top-K by $\alpha$, then radiation expansion (top-K' neighbors of each activated trace). Radiation depth is bounded to prevent explosion.

**Implementation note**: The pseudocode in v0.2 first ran KNN then filtered by weight. The correct approach is to compute $\alpha$ for all candidates and select top-K by the composite score. We correct this.

#### 4.1.3 Emergence Function $E$

$$E(s, \{c_{i_1}, ..., c_{i_k}\}) \rightarrow r$$

$E$ feeds the raw signal $s$ and the raw contents $\{c_i\}$ of activated traces to an LLM:

$$r = \text{LLM}\Big(\text{prompt}(s, \{c_i\})\Big)$$

$E$ is session-stateless: it holds nothing between calls. Randomness comes from LLM sampling ($\xi$).

#### 4.1.4 Write Operation $W$

$$W(s, r, t_{\text{now}}) \rightarrow t_{\text{new}}$$

Writes $(c_{\text{new}}, v_{\text{new}}, \tau_{\text{now}}, \eta_{\text{new}})$ to $\mathcal{S}$, where $c_{\text{new}}$ encodes the interaction (both $s$ and $r$). $\eta_{\text{new}}$ is assessed by a separate LLM call (importance rating).

#### 4.1.5 Reflection Loop $L$

Background loop (no external signal):
1. Sample a random trace $t_j$ from $\mathcal{S}$
2. Perturb: $s_{\text{int}} = v_j + \mathcal{N}(0, \sigma^2 I)$
3. Run SSA: $r_{\text{int}} = E(s_{\text{int}}, A(s_{\text{int}}, \mathcal{S}, t))$
4. Write $W(s_{\text{int}}, r_{\text{int}}, t_{\text{now}})$ to $\mathcal{S}$ as a reflection trace (marked `is_internal`)
5. With probability $p$, push $r_{\text{int}}$ to user as proactive message

**Clarification on reflection traces**: Reflection traces contain *text* ($c_{\text{int}}$), not just vectors. The vector $v_{\text{int}} = \phi(c_{\text{int}})$ is derived. When reflection traces are later activated, their *content* (text) is fed to $E$, same as any trace. This corrects v0.2's ambiguity about "reflection generates vectors."

### 4.2 Complete Algorithm (Corrected)

```
Algorithm: Stateless Space Activation
─────────────────────────────────────
Input:  signal s (raw text)
        space S (persistent)
        query time t
Output: response r

1. v_s ← φ(s)                          // Vectorize signal
2. candidates ← all traces in S
3. for each t_i in candidates:          // Compute composite score
     α_i ← cos(v_s, v_i) · ρ_i(t) · η_i
4. T_act ← top-K(candidates, by α)     // Weighted selection (not KNN-then-filter)
5. T_act ← T_act ∪ RadiationExpand(T_act, K')
6. r ← E(s, {content(t) for t in T_act})  // Emerge response (uses raw content)
7. η_new ← LLM_rate_importance(s, r)
8. t_new ← (s+r, φ(s+r), t_now, η_new) // Construct trace (content + vector)
9. S ← S ∪ {t_new}                      // Append
10. return r
```

### 4.3 Invariants (Corrected)

Previous invariants had conflicts. Corrected:

1. **Session-statelessness of inference core**: $A$ and $E$ hold no state between requests. $\mathcal{S}$ is external to the core. (Not "strict statelessness"—$\mathcal{S}$ is state.)
2. **Trace content immutability**: $(c_i, v_i, \tau_i, \eta_i)$ are fixed at write time. $\rho_i(t)$ is a derived function of query time, not a stored field—so no conflict with immutability.
3. **Append-only by default**: $\mathcal{S}$ grows monotonically. Forgetting is an explicit policy that *deletes* traces (not modifies), and is acknowledged as a violation of monotonic growth for practical necessity.
4. **Deterministic activation, stochastic emergence**: Given fixed $s$, $\mathcal{S}$ snapshot, and $t$, $A$ is deterministic. $E$ is stochastic due to $\xi$. (We no longer claim "deterministic emergence"—emergence is stochastic.)

### 4.4 Comparison with Existing Paradigms (Corrected)

The comparison table in v0.2 used non-uniform reset definitions. Corrected:

| Paradigm | What persists | Organization | Mutation style | Reset (clear session state) | Reset (clear all memory) |
|----------|--------------|--------------|----------------|---------------------------|-------------------------|
| Fine-tuning | Model weights | — | In-place weight update | No effect (weights persist) | Requires re-training |
| RAG | None (per-request) | External index | N/A | No effect | Losces knowledge |
| MemGPT | Working memory + external | Hierarchical | In-place (working mem) | Working mem cleared; external persists | Loses both |
| Generative Agents | Reflection + plan + stream | Timeline + reflection nodes | In-place (reflection/plan) + append (stream) | Plan/reflection cleared; stream persists | Loses all |
| Replika/Character.ai | Emotion vector + history | Timeline | In-place (emotion) + append (history) | Emotion reset; history persists | Loses all |
| **SSA** | $\mathcal{S}$ (external) | Spatial (semantic clusters) | Append-only | No effect ($\mathcal{S}$ persists; core is already stateless) | Losces $\mathcal{S}$ |

**Key correction**: "Reset" must specify *what* is reset. SSA's advantage is narrow: clearing session state (context window, working memory) has no effect because the core is already session-stateless. But clearing $\mathcal{S}$ kills SSA just as clearing all memory kills other systems. Fine-tuning is actually *more* reset-resilient (weights persist even if all memory is cleared). v0.2's comparison was misleading.

---

## 5. Reference Implementation

**Status: Core interactive path implemented.** The repository now contains an
append-only SQLite/sqlite-vec trace space, deterministic activation audit,
the frozen `legacy-ssa-a0` one-hop radiation baseline, DeepSeek emergence,
episode writing, historical backfill, and an interactive terminal projection.
The serving path now persists a P1 human-clock and situation-perception snapshot
after structured appraisal and reconstructs that snapshot into private prompt
context.
HDSC-H1 remains an isolated reversible mathematical reference. HDSC-H1D reads
the full directed multi-relation topology without entering the prompt. HDSC-H2
advances only after a complete episode is written, is rebuilt by chronological
replay, and remains strictly outside the serving prompt path.

The implementation uses appraisal-derived importance for episode traces rather
than the separate importance-only LLM call shown in the conceptual algorithm.
This reduces synchronous calls and makes write-time importance reproducible;
the difference must be preserved as an experimental factor.

### 5.1 Implemented Core Architecture

```
┌─────────────────────────────────────────────────┐
│                  HDSC Runtime                   │
│                                                  │
│  Space S: SQLite + sqlite-vec                   │
│  Embedding φ: bge-small-zh-v1.5 (512-dim)       │
│  Activation A: legacy SSA-A0 (visible baseline)  │
│  Transport H1: symmetric diffusion (reference)   │
│  Transport H1D: directed Markov flow (shadow)     │
│  Active H2: bounded cluster measure (shadow)     │
│  Perception P1: time × semantic × affect × context│
│  Emergence E: LiteLLM → DeepSeek (default)       │
│  Reflection L: APScheduler, 15-min interval      │
│  Interface: Telegram Bot / CLI                   │
└─────────────────────────────────────────────────┘
```

### 5.2 Trace Schema

```sql
CREATE TABLE traces (
    id           INTEGER PRIMARY KEY,
    content      TEXT NOT NULL,           -- raw content (c_i)
    content_type TEXT NOT NULL,           -- 'user'|'agent'|'reflection'|'event'
    embedding    BLOB NOT NULL,           -- v_i
    timestamp    REAL NOT NULL,           -- τ_i
    importance   REAL NOT NULL,           -- η_i, fixed at write
    is_internal  BOOLEAN DEFAULT FALSE,   -- reflection traces
    metadata     TEXT                     -- JSON
);
```

Note: `freshness` is NOT stored—it's computed at query time as $e^{-\lambda(t - \tau_i)}$.

### 5.3 Corrected Activation (Weighted, Not Filtered)

```python
def activate(signal_vec, space, query_time, k=10, k_radiation=5, lam=0.01):
    # Compute composite score for ALL traces
    candidates = space.all_traces()
    scored = []
    for t in candidates:
        freshness = math.exp(-lam * (query_time - t.timestamp))
        score = cosine(signal_vec, t.embedding) * freshness * t.importance
        scored.append((t, score))

    # Top-K by composite score (NOT KNN-then-filter)
    primary = sorted(scored, key=lambda x: -x[1])[:k]

    # Radiation
    secondary = set()
    for t, _ in primary:
        neighbors = space.knn_search(t.embedding, k=k_radiation)
        secondary.update(neighbors)
    secondary -= set(t for t, _ in primary)

    return [t for t, _ in primary] + list(secondary)
```

### 5.4 Emergence Prompt

```jinja2
Current signal ({{ signal_time }}): {{ signal }}

Activated traces from the past (most relevant first):
{% for trace in activated_traces %}
[{{ trace.timestamp }} | type: {{ trace.content_type }}]
{{ trace.content }}
{% endfor %}

Respond naturally. You are not instructed to play a character.
```

Note: We no longer use the "you are what emerges" framing, which presupposed emergent behavior. The prompt is neutral.

### 5.5 HDSC-H1 Passive Transport (Shadow Only)

The H1 kernel does not replace the baseline activation in the prompt. It first
allocates a fixed computational source budget, then applies a reciprocal graph
Laplacian:

$$
u(t)=e^{-\kappa t}e^{-t(D-K)}q.
$$

`q` is dimensionless activation mass. Internal graph transfer cancels pairwise;
only the explicitly recorded environment sink is dissipated. The reference
implementation rejects directed or negative conductances and emits an audit
record containing conservation and reciprocity residuals, Dirichlet energy,
and entropy diagnostics. This establishes a falsifiable C1 structure analogy,
not a claim that the software has a temperature or consumes a known number of
joules. See `notes/research/2026-07-26_hdsc_validation_framework.md` for the
promotion gate.

### 5.6 Macro Closure, Not Microstate Prediction

HDSC-H1 evolves a coarse variable $u=C(X)$, where $X$ denotes the complete
microstate. Its deterministic solution is conditional on fixed source mass,
conductance graph, duration, and sink rate. It does not predict the microscopic
construction of those quantities or the downstream LLM trajectory.

A Markovian macro equation is valid only if microstates sharing the same $u$
also share the same conditional coarse future. This lumpability condition is
currently unverified. Hidden embedding state, graph updates, floating-point
ties, scheduler interleavings, and model sampling may instead produce a memory
kernel and unresolved forcing. Accordingly, every H1 audit labels its modeled
scale `macro-coarse-grained` and its microstate status `unmodeled`. Nonzero mass
or a nonzero perturbation is not by itself evidence of trajectory divergence;
that claim requires a measured positive Lyapunov exponent, stochastic
transition kernel, or coarse-graining non-identifiability.

### 5.7 Growing Archive And Closed-Loop Gain

Append-only storage and drift resistance are independent claims. With one
episode per turn, archive cardinality follows $N_t=N_0+t$, so its first discrete
difference is one and its second difference is zero. This says nothing by
itself about monotonic growth of topology, response magnitude, or sensitivity.

The actual system has two feedback paths: the previous response changes the
next user signal, and the written episode changes later source allocation and
graph geometry. Since text is discrete and stochastic, we replace the informal
$\partial r_t/\partial r_{t-1}$ with either a causal distance between response
distributions under paired interventions or the Jacobian of expected response
embeddings.

HDSC therefore separates the unbounded immutable archive from a bounded active
state: fixed total mass, a fixed number of semantic clusters, cluster-level
duplicate normalization, capped one-turn graph change, and entry/exit
hysteresis. H2 seeks contraction only on certified fixed-support branches;
identity changes use repeated independent evidence and versioned thresholds.
The target is nonzero but bounded memory influence, rather than zero dependence
or unconstrained amplification.

### 5.8 HDSC-H2 Bounded Active Space (C0-M Shadow Only)

HDSC-H2 makes the archive/active-state distinction executable. Its model
identifier is `hdsc-h2-bounded-active-shadow`. Let
$\mathcal{S}^{\mathrm{arc}}_t$ be the immutable evidence archive and let
$\mu_t$ be the bounded state that could eventually mediate behavior:

$$
\mathcal{S}^{\mathrm{arc}}_{t+1}
=\mathcal{S}^{\mathrm{arc}}_t\oplus W(s_t,r_t),
$$

$$
\mu_t
=w_{\bot,t}\delta_{\bot}
+\sum_{c\in\mathcal{C}_t}w_{c,t}\delta_{z_c},
\qquad
|\mathcal{C}_t|\le C,
\quad
w_{\bot,t}+\sum_c w_{c,t}=1.
$$

Here $C$ is a fixed capacity, $z_c$ is the immutable unit-norm prototype of a
versioned semantic cluster, and $\bot$ is a null reservoir with no semantic
content. The archive may grow while the active state remains a probability
measure of bounded support. Evidence counts and duplicate counts are audit
metadata; they are not activation mass.

Candidates are grouped by `(cluster_id, cluster_revision)`. A cluster's write
score is computed once, using the maximum eligible candidate score rather than
the sum. Exact fingerprints are checked within one candidate batch; across
turns, a candidate has zero innovation while its versioned semantic cluster is
active. The bounded state does not retain an archive-sized fingerprint set, so
an evicted cluster may later re-enter. Consequently, copying a candidate any
number of times within the same proposal leaves the behavioral proposal
unchanged:

$$
Q(D\uplus nD_c)=Q(D).
$$

If a turn contains only represented duplicates, its update is identical to the
empty-candidate update, including the same scheduled decay. Let $q_t$ be the
probability proposal over the unique novel clusters, or $\delta_\bot$ when no
novel proposal exists. H2 then computes

$$
\widehat\mu_{t+1}
=\beta T_t\mu_t+\alpha_tq_t
+(1-\beta-\alpha_t)\delta_\bot,
\qquad
\mu_{t+1}=\Pi_C(\widehat\mu_{t+1}),
$$

subject to

$$
0\le\beta<1,
\qquad
0\le\alpha_t\le1-\beta.
$$

$T_t$ is the identity in the minimum H2 model; a later transport operator must
be nonnegative, mass-preserving, and non-expansive in the certificate metric.
$\Pi_C$ retains at most $C$ non-null clusters and moves all overflow mass to
$\bot$ without renormalizing the retained clusters. Thus removal cannot amplify
the remaining mass merely by changing the denominator.

The strict contraction statement is branch-local. When proposal support and
capacity support do not switch, and $T_t$ is non-expansive,

$$
D_{\mathrm{TV}}(\mu_{t+1},\mu'_{t+1})
\le
\beta D_{\mathrm{TV}}(\mu_t,\mu'_t)
+\alpha_tD_{\mathrm{TV}}(q_t,q'_t).
$$

The local certificate checks three distinct selection margins. For the fixed
SimHash codebook, the assignment margin is the normalized embedding's distance
to its nearest hyperplane,

$$
\Delta_z=\min_{j,\ell}|h_\ell^\top\bar v_j|.
$$

Hard top-$K$ selection is discontinuous at a tie and at zero-score entry when
the positive support has fewer than $K$ members. If
$\Delta_q=g_{(K)}-g_{(K+1)}$ (with the zero boundary included) and the score
map has Lipschitz bound $L_g$, the proposal support radius is

$$
\epsilon_q=\frac{\Delta_q}{2L_g}.
$$

Let $S_q$ be the selected positive-score sum and $m$ its support size. On a
fixed proposal support, normalization $q_c=g_c/S_q$ has the conservative TV
gain $L_{q,g}=mL_g/S_q$. Capacity selection uses the actual priority
$p_c=\widehat w_c+h\mathbf 1[c\in\mathcal C_t]$. Its margin is
$\Delta_C=p_{(C)}-p_{(C+1)}$, or the minimum retained mass when no outside
candidate exists. In the max-product source metric over normalized-embedding
distance, raw-score-source distance, and prior-state TV, the conservative joint
support radius is

$$
\epsilon_{\mathrm{support}}=\min\left(
\Delta_z,\frac{\Delta_q}{2L_g},
\frac{\Delta_C}{2(\beta+\alpha_tL_{q,g})}
\right).
$$

This value is not a radius in raw text space; mapping another input metric into
it requires separate component gains. Deterministic tie-breaking and incumbent
hysteresis reduce churn but do not turn these selectors into globally
continuous operators. A missing or zero margin, or a radius below numerical
tolerance, produces `abstain` and reports switched mass. The certificate is also
conditioned on a fixed candidate universe; discrete candidate appearance is
outside its radius. A semantic jump is bounded by the active-space diameter
times switched mass, not reported as contraction.

For the full user-agent-memory loop, define $a=L_{F,s}$, $b=L_{F,\mu}$, and
$u=L_U$. If the proposal has branch-local gains $L_{q,s}$, $L_{q,\mu}$, and
$L_{q,r}$, set

$$
c=\alpha_tL_{q,s},
\qquad
d=\beta+\alpha_tL_{q,\mu},
\qquad
e=\alpha_tL_{q,r}.
$$

On a certified fixed-support branch, H2 evaluates the comparison matrix

$$
\mathbf{G}=
\begin{bmatrix}
ua & ub\\
c+ea & d+eb
\end{bmatrix}.
$$

The joint small-gain gate is

$$
\rho(\mathbf{G})<1-\varepsilon_{\mathrm{gain}}.
$$

Missing user gain, an unmeasured gain source, non-finite component bounds, or an
unstable selection branch produces explicit abstention rather than a closed-loop
certificate. This is a **C0-M computational claim**: fixed capacity, normalized
mass, duplicate suppression, and conditional gain accounting. H2 is updated
only after the completed episode has been committed, and its state is not read
by activation, prompt construction, or response generation. Online and restart
paths both derive innovation from archived bucket identity and current active
support; they do not substitute an unavailable appraisal novelty during replay.
The shadow reports success only when its processed step equals archive count and
every trace has a compatible vector from the same hashed semantic partition.
Repository, vector, or partition failures remain shadow audit failures.
Therefore H2 has no prompt or behavioral effect in shadow mode. Composing H1
transport with H2 active state, then admitting that composition to the prompt
path, is deferred to HDSC-H3.

Every update records the model, semantic-partition, and configuration hashes;
behavior-measure and replay-step hashes; archive and active counts; candidate
and unique-cluster counts; suppressed duplicates; total/null/discarded/switched
mass; cluster/proposal/capacity margins; proposal normalization gain; component
and joint support radii; component gains; $\rho(\mathbf{G})$; and a typed
certificate or abstention reason. These fields are computational audit data,
not behavioral measurements.

### 5.9 HDSC-H1D Directed Nonreversible Transport (Shadow Only)

H1 assumes a reciprocal conductance graph. That assumption is appropriate for
a reversible reference but not for a trace topology containing temporal,
evidence, revision, or feedback direction. Symmetrizing a directed adjacency as
$(R+R^T)/2$ invents reverse edges and removes net currents. HDSC-H1D therefore
uses the full directed rate matrix with convention $R_{ij}$ = rate from node
$j$ to node $i$.

The reciprocal and directional parts may be audited as

$$
R^{\mathrm{rev}}_{ij}=\min(R_{ij},R_{ji}),
\qquad
R^{\mathrm{dir}}=R-R^{\mathrm{rev}},
$$

but only $R^{\mathrm{rev}}$ is eligible for the symmetric H1 reference. H1D
constructs the column-conservative Metzler generator

$$
Q_{ij}=R_{ij}\ (i\ne j),
\qquad
Q_{jj}=-\sum_{i\ne j}R_{ij},
\qquad
\mathbf 1^TQ=0.
$$

Directed transport and explicit environment leakage are

$$
u_{\mathrm{transported}}(t)=e^{tQ}u_0,
\qquad
u_{\mathrm{final}}(t)=e^{-\kappa t}e^{tQ}u_0.
$$

$P_t=e^{tQ}$ is a column-stochastic Markov semigroup. It preserves positivity
and closed-system mass and is non-expansive in $L_1$ for equal-mass inputs.
These properties do not require reciprocity. In contrast, H1's symmetric
Dirichlet decay, double stochasticity, non-decreasing raw Shannon entropy, and
uniform stationary distribution are not transferred to H1D.

If a strictly positive stationary distribution $\pi$ is supplied and satisfies
$Q\pi=0$, H1D evaluates the Markov data-processing diagnostic

$$
D_{KL}(P_tp\|\pi)\le D_{KL}(p\|\pi).
$$

At stationarity, let $F_{ij}=R_{ij}\pi_j$. Finite structural entropy production
is evaluated only when every positive rate has reverse support:

$$
\sigma=\sum_{i<j}(F_{ij}-F_{ji})
\log\frac{F_{ij}}{F_{ji}}\ge0.
$$

A one-way edge requires an explicit open reservoir or hidden reverse-channel
model; otherwise the finite entropy-production gate abstains. This still is not
a physical-energy result: local detailed balance, reservoir parameters, and
joule calibration are absent, so physical thermodynamic status remains
`not-calibrated`.

The runtime stores reciprocal `semantic` edges and directed
`temporal-forward` edges separately. The latter means only older-to-newer time
order, not causality. Legacy radiation reads semantic edges only, while H1D's
offline matrix reads the full multi-relation topology. The SPACE view renders
reciprocal links as `:` and directed links with arrows. H1D remains shadow-only;
composition with H2 is deferred to H3.

### 5.10 YUANZI-5D Recursive Living Operator (Theory Only)

H1D transports a fixed computational mass on a frozen directed graph, while H2
bounds an archive-derived active measure. Neither kernel defines perception,
action, self-model revision, or an autonomous loop. We therefore introduce
**YUANZI-5D** as a falsifiable theoretical candidate rather than an implemented
H3 result. Its central distinction is between (i) asymmetric stochastic
transport, which encodes directed evidence, and (ii) skew-symmetric circulation,
which encodes signed flow tangent to a free-energy landscape. The second does
not replace the first.

#### 5.10.1 Markov blanket and normalized discrete inference

Let external, sensory, internal, and active states be respectively
$\eta_t,s_t,\mu_t,a_t$. A Markov-blanket claim requires the conditional
independence

$$
p(\mu_t,\eta_t\mid s_t,a_t)
=p(\mu_t\mid s_t,a_t)p(\eta_t\mid s_t,a_t),
$$

or equivalently zero conditional mutual information under the modeled
distribution. In the proposed software boundary, user/world/archive states are
external; retrieved observations are sensory; normalized beliefs are internal;
and responses, questions, tool calls, and archive writes are active. This
modular assignment is not itself evidence that the factorization holds.

For finite latent states $x_t\in\{1,\ldots,N\}$, let
$q_t\in\Delta^{N-1}$ be the internal belief. A discrete generative model uses

$$
\mathbf A_{oi}=p(o_t=o\mid x_t=i),
\qquad
\mathbf B^a_{ij}=p(x_{t+1}=i\mid x_t=j,a_t=a),
$$

together with preference distribution $\mathbf C_t$, initial-state prior
$\mathbf D$, and policy-habit prior $\mathbf E$. Columns of $\mathbf A$ and
$\mathbf B^a$ are normalized, but $\mathbf B^a$ need not be symmetric. The
predicted prior and one-step variational free energy are

$$
\bar q_t=\mathbf B^{a_{t-1}}q_{t-1},
$$

$$
F_t(q)=\sum_iq_i
\left(\log q_i-\log\bar q_{t,i}-\log\mathbf A_{o_ti}\right).
$$

For an unrestricted categorical posterior, minimizing $F_t$ gives

$$
q_t=\operatorname{Normalize}
\left(\mathbf A_{o_t,:}\odot\bar q_t\right).
$$

Perceptual free-energy minimization alone is not an action loop. For a policy
$\pi=(a_t,\ldots,a_{t+H-1})$, predict

$$
q_{\tau+1}^{\pi}=\mathbf B^{a_\tau}q_\tau^\pi,
\qquad
q^\pi(o_\tau)=\mathbf A q_\tau^\pi.
$$

With likelihood ambiguity
$h_{\mathbf A,i}=-\sum_o\mathbf A_{oi}\log\mathbf A_{oi}$, a discrete
risk-plus-ambiguity expected-free-energy objective is

$$
G_t(\pi)=\sum_{\tau=t+1}^{t+H}
\left[
D_{KL}\!\left(q^\pi(o_\tau)\|\mathbf C_\tau\right)
+(q_\tau^\pi)^Th_{\mathbf A}
\right].
$$

Under the corresponding exact predictive factorization, each horizon term can
also be read as epistemic plus preference value,

$$
G_\tau(\pi)
=-I_{q^\pi}(x_\tau;o_\tau)
-\mathbb E_{q^\pi(o_\tau)}\log\mathbf C_\tau(o_\tau),
$$

making explicit that uncertainty reduction and preferred outcomes are separate
drivers.

The policy and current action distributions are

$$
q_t(\pi)=\operatorname{Normalize}
\left(\mathbf E(\pi)e^{-\gamma G_t(\pi)}\right),
\qquad
q_t(a)=\sum_{\pi:\pi_t=a}q_t(\pi).
$$

Candidate actions include `NOOP`, retrieval, response, clarification,
reflection, belief revision, memory write, and a budgeted proactive message.
The epistemic and preference terms must be evaluated separately in experiments;
the formula does not by itself prevent self-confirming policies or a trivial
low-stimulation attractor.

#### 5.10.2 Directed multi-relation reachability

For relation type $r$, let $W^{(r,a)}_{ij}\ge0$ represent directed support from
source $j$ to target $i$ under action $a$. Query- and state-dependent gates
$g_t^{(r)}\ge0$ form

$$
\widetilde{\mathbf B}_t^a
=\epsilon I+\sum_r g_t^{(r)}W^{(r,a)},
\qquad
\mathbf B_{t,ij}^a
=\frac{\widetilde{\mathbf B}_{t,ij}^a}
{\sum_k\widetilde{\mathbf B}_{t,kj}^a}.
$$

Relation types may include semantic, entity, category, spatial, episodic,
preference, evidence, revision, and temporal-forward links. Reverse relations
require their own evidence and are never manufactured by symmetrization. If
$A\to B\to C$ is supported, then

$$
[(\mathbf B_t^a)^2]_{C,A}
\ge \mathbf B^a_{t,C,B}\mathbf B^a_{t,B,A}>0,
$$

so $C$ can be activated without being a direct neighbor of $A$. Prompt
admission must retain the supporting typed path, not only the terminal score.
For example, a coffee query may activate both a spatial branch to a nearby
shop and an episodic branch through beverage preference to a former partner.
A claim that both people visited that shop still requires a direct visit trace
or must be phrased as a hypothesis or question.

#### 5.10.3 Four dynamic fibers and a fifth law coordinate

The proposed four-fiber state space is

$$
\mathcal M_4
=\mathcal M_S\oplus\mathcal M_A\oplus\mathcal M_M\oplus\mathcal M_P,
$$

where the fibers denote sensation/evidence, action, memory/past, and
prediction/counterfactual future. Each fiber may itself be high-dimensional;
``4D'' does not mean four scalar features. Perception-action and
memory-prediction form two coupled circulation planes.

Let

$$
\Theta_t=\{J_t,\Gamma_t,\mathcal F_t,
\mathbf A_t,\mathbf B_t,\mathbf C_t,
\text{precision}_t,\text{topology}_t\}
$$

parameterize the geometry and laws acting on $\mathcal M_4$. Let
$\mathcal M_\Theta$ be the admissible operator-parameter manifold. The proposed
five-coordinate construction is more precisely the operator-parameterized
bundle

$$
\mathcal M_5^{\mathrm{op}}
=\bigsqcup_{\Theta\in\mathcal M_\Theta}
(\mathcal M_4,g_\Theta,J_\Theta,\Gamma_\Theta),
$$

not a Cartesian addition of another content feature. Since the meta-update
below also depends on $X_t$, the complete dynamics are bidirectionally coupled,
not a classical one-way skew-product system. Its recursion is

$$
X_{t+1}=\mathfrak Y_{\Theta_t}(X_t,o_t),
\qquad
\Theta_{t+1}=\mathfrak R(\Theta_t,X_t,X_{t+1},o_t).
$$

Thus the state changes under the current operator while a slower process
updates the operator itself.

#### 5.10.4 Weighted skew circulation and dissipation

Write normalized beliefs as $q=\operatorname{softmax}(y)$ and define
$\widehat{\mathcal F}(y)=\mathcal F(\operatorname{softmax}(y))$. For every
oriented cycle $c$ in the expanded perception-action-environment graph, define
a signed cycle matrix $C_c^T=-C_c$. A weighted cycle field is

$$
\mathsf K_t=\sum_c\kappa_{t,c}C_c,
\qquad \mathsf K_t^T=-\mathsf K_t,
$$

where $\kappa_{t,c}$ may depend on epistemic value, goal relevance, affective
salience, evidence precision, and resource cost. Negative matrix entries encode
signed tangent current; they are not negative Markov rates or fabricated
reverse facts.

For a positive diagonal resistance metric $W_t$ and symmetric
$H_t\succeq0$, define

$$
\Omega_t=W_t^{-1}\mathsf K_tW_t^{-1},
\qquad
\Gamma_t=W_t^{-1}H_tW_t^{-1}.
$$

Then $\Omega_t^T=-\Omega_t$ and $\Gamma_t\succeq0$. The candidate stochastic
flow is

$$
dy_t=(\Omega_t-\Gamma_t)
\nabla_y\widehat{\mathcal F}_t\,dt
+\mathsf U_tu_t\,dt
+\sqrt{2T_t\Gamma_t}\,d\mathcal B_t.
$$

With frozen operators, no drive, and no noise,

$$
\frac{d\widehat{\mathcal F}}{dt}
=-\nabla\widehat{\mathcal F}^{\,T}
\Gamma_t\nabla\widehat{\mathcal F}\le0,
$$

because the skew contribution is identically zero. Pure skew flow circulates
without descending the potential; pure dissipative flow descends without
sustaining a loop. Their driven combination is the intended open-system model.

At a reduced level, let $J_5^{\mathrm{macro}}\in\mathbb R^{5\times5}$ couple
one summary coordinate from each of the four fast fibers and one operator
coordinate. An odd-dimensional real skew-symmetric matrix is singular, so
$\det J_5^{\mathrm{macro}}=0$ and
$\operatorname{rank}(J_5^{\mathrm{macro}})\le4$. If its null distribution is
integrable, a coarse Casimir candidate $\mathcal I$ may exist with

$$
J_5^{\mathrm{macro}}\nabla\mathcal I=0.
$$

YUANZI-5D hypothesizes that such a slow, evidence-gated invariant could encode
identity continuity while the four-dimensional fast state circulates. Odd
dimension alone does not prove that the null direction is integrable, stable,
unique, or psychologically meaningful. Because each fiber may be
multidimensional, the parity result for $J_5^{\mathrm{macro}}$ also does not
imply that the full bundle operator has a null direction. A state-dependent
Poisson tensor has to satisfy the Jacobi identity; skew symmetry alone is
insufficient.

#### 5.10.5 Structure-preserving discrete recursion

Let $\bar\nabla\widehat{\mathcal F}(y_t,y_{t+1})$ be a discrete gradient
satisfying the exact increment identity [22]

$$
\widehat{\mathcal F}(y_{t+1})-
\widehat{\mathcal F}(y_t)
=\bar\nabla\widehat{\mathcal F}^{\,T}(y_{t+1}-y_t).
$$

The proposed normalized YUANZI step is the implicit recursion

$$
\frac{y_{t+1}-y_t}{\Delta t}
=(\Omega_t-\Gamma_t)
\bar\nabla\widehat{\mathcal F}
+\mathsf U_tu_t
+\sqrt{\frac{2T_t\Gamma_t}{\Delta t}}\,\xi_t,
$$

$$
q_{t+1}=\operatorname{softmax}(y_{t+1}).
$$

In the frozen, unforced, zero-noise case this gives the testable discrete law

$$
\widehat{\mathcal F}_{t+1}-\widehat{\mathcal F}_t
=-\Delta t\,
\bar\nabla\widehat{\mathcal F}^{\,T}
\Gamma_t
\bar\nabla\widehat{\mathcal F}\le0,
$$

while positivity and normalization follow from softmax. If $\Omega$ and
$\Gamma$ are constant and
$\rho^*(y)=Z^{-1}\exp[-\widehat{\mathcal F}(y)/T]$ is normalizable, the
associated stationary density may have $\partial_t\rho^*=0$ but a nonzero
divergence-free current

$$
j^*(y)=\Omega\nabla\widehat{\mathcal F}(y)\rho^*(y),
\qquad
\nabla\cdot j^*=0,
\qquad
j^*\ne0.
$$

This formalizes macro-level stationarity with continuing micro-level
circulation. It is a computational non-equilibrium hypothesis, not evidence of
physical entropy production or subjective experience.

#### 5.10.6 Meta-recursion and time-scale separation

Let $\varepsilon_t^{\mathrm{state}}$ measure posterior predictive error,
$\varepsilon_t^{\mathrm{action}}$ measure policy prediction error, and
$\varepsilon_t^{\mathrm{self}}$ measure disagreement between predicted and
observed identity features. A candidate meta-objective is

$$
\mathcal F_t^{\mathrm{meta}}
=\lambda_s\varepsilon_t^{\mathrm{state}}
+\lambda_a\varepsilon_t^{\mathrm{action}}
+\lambda_i\varepsilon_t^{\mathrm{self}}
+\lambda_c\operatorname{Complexity}(\Theta_t).
$$

The fifth-coordinate update is

$$
\Theta_{t+1}=\operatorname{Project}_{\mathcal C}
\left[
\Theta_t+\epsilon_\Theta
(\Omega_\Theta-\Gamma_\Theta)
\nabla_\Theta\mathcal F_t^{\mathrm{meta}}
\right],
$$

where $\mathcal C$ enforces skew symmetry, positive-semidefinite dissipation,
probability normalization, bounded precision, provenance, and an identity
change-rate limit. The displayed form follows the same circulation-minus-
dissipation convention as the fast flow; a numerical realization must preserve
these constraints after every meta-step.

The proposed time scales satisfy

$$
\Delta t_X\ll\Delta t_\Theta\ll\Delta t_{\mathcal I}.
$$

Attention and posterior belief may change every event; topology and operator
parameters require repeated evidence; core identity candidates change only on
long, independently sourced evidence windows. Higher-order notation makes the
recursion explicit:

$$
\mathfrak Y^{[1]}:X_t\mapsto X_{t+1},
\qquad
\mathfrak Y^{[2]}:\mathfrak Y_t^{[1]}
\mapsto\mathfrak Y_{t+1}^{[1]}.
$$

The regress is closed by fixed structural constraints rather than an unlimited
stack of self-models. YUANZI-5D therefore defines ``living'' operationally as a
bounded, normalized, driven perception-action system with nonzero internal
circulation and slow self-model adaptation. This is a research definition to
be tested, not a proof that the implementation is alive.

---

## 6. Proposed Experiments (Executable Design, No Behavioral Results)

**We have no behavioral or physical-energy results.** The interactive system,
HDSC-H1 reversible transport, HDSC-H1D directed transport, and HDSC-H2 bounded
active-space updater are available, but all research kernels remain shadow-only.
YUANZI-5D is not yet implemented. The following is an executable design for
existing kernels and a pre-implementation validation protocol for the new
theory, not a claim of validation.

### 6.1 Core Problem: Evaluation

"Feels lifelike" resists quantification. Our previously proposed metrics were flawed:
- **Response embedding similarity** measures topical similarity, not personality.
- **Temperature-induced diversity** is sampling noise, not emergent emotion.
- **Post-reset recall** tests retrieval, not identity continuity.

We need metrics that distinguish:
1. *Personality stability*: Does the agent exhibit consistent behavioral tendencies (not just topic similarity) over time, without a preset persona?
2. *Contextual emotional variation*: Do identical inputs yield emotionally different responses *driven by activated traces*, not just sampling?
3. *Identity continuity across LLMs*: Does switching LLMs (same $\mathcal{S}$) preserve user-perceived identity?

These require human evaluation or validated proxies, which we have not yet developed.

### 6.2 Factorized Design (Corrected)

v0.2 confounded reflection and LLM switch in Week 2. Corrected:

| Condition | Reflection | LLM |
|-----------|-----------|-----|
| A | Off | DeepSeek |
| B | Off | Claude |
| C | On | DeepSeek |
| D | On | Claude |

Each condition runs for a fixed period (e.g., 1 week) on the **same $\mathcal{S}$ snapshot** (cloned at start). This isolates reflection effect (A vs C, B vs D) and LLM effect (A vs B, C vs D).

### 6.3 Open Questions for Evaluation

- **Personality metric**: How to measure "consistent tendencies" without a persona prompt? Candidate: multi-turn consistency scoring by human raters. Needs validation.
- **Emotion metric**: How to attribute emotional variation to activated traces vs. sampling? Ablation: fix $\mathcal{S}$ snapshot, vary only input → measures sampling effect. Vary $\mathcal{S}$ (add traces) → measures space effect.
- **Identity metric**: "Same agent?" blind rating by users. Needs inter-rater reliability.

### 6.4 Baselines

SSA must be compared against:
- **RAG** (no freshness/importance, no radiation)
- **Generative Agents** (timeline + reflection + plan)
- **MemGPT** (working memory + paging)
- **Synapse** [18] (spreading activation) — the closest prior work

Without these baselines, we cannot attribute any observed effect to SSA's specific design.

### 6.5 H2 Computational and Closed-Loop Evaluation

H2 evaluation begins with computational properties rather than user ratings.
Property-based tests must cover candidate-order invariance, exact cluster-level
duplicate suppression, duplicate-only versus empty-turn equivalence, fixed
capacity over arbitrarily long replay, nonnegative normalized mass, overflow
transfer to the null reservoir, and the fixed-support contraction inequality.
Shadow isolation additionally requires byte-identical legacy activation and
prompt construction with H2 enabled or disabled.

Paired replay then perturbs one input while holding the archive snapshot,
configuration, and sampled-noise coupling fixed. Next-token distributions are
compared with total variation, Jensen-Shannon, or Hellinger distance; complete
responses and trajectories use a separately validated semantic ground metric
with Wasserstein distance. Results must be stratified by proposal and capacity
selection margins, because near-tie top-$K$ cases do not share the local
certificate of stable-support cases.

The replay estimates or upper-bounds each entry of the joint small-gain matrix,
records the source and uncertainty of the user-feedback gain, and reports both
certificate coverage and abstention frequency. These measurements would test
the C0-M model and its assumptions; they would not establish a behavioral or
physical-energy result. A later H3 experiment must separately randomize whether
the validated H1/H2 composition enters the prompt path. Only that experiment
could estimate behavioral effect relative to SSA-A0 and the listed baselines.

### 6.6 YUANZI-5D Falsification Program

YUANZI-5D must first be tested on finite synthetic systems with exact reference
solutions. Implementation admission requires the following independent gates:

1. **Blanket gate**: estimate
   $I(\mu_t;\eta_t\mid s_t,a_t)$ under interventions rather than inferring the
   Markov-blanket factorization from module names.
2. **Categorical gate**: every posterior, likelihood column, transition column,
   preference, and policy distribution remains finite, nonnegative, and
   normalized over long replay.
3. **Direction gate**: typed one-way edges remain one-way; permutation-equivariant
   tests must show that no transpose averaging or implicit reverse edge enters
   $\mathbf B^a$.
4. **Reachability gate**: controlled $A\to B\to C$ fixtures verify nonzero
   two-hop activation, parallel branch retention, typed path provenance, and
   unsupported-claim suppression.
5. **Geometric gate**: numerical residuals verify
   $\Omega^T=-\Omega$, $\Gamma=\Gamma^T\succeq0$, and, for state-dependent
   Poisson candidates, the Jacobi identity.
6. **Discrete-law gate**: with frozen operators, zero drive, and zero noise,
   each implicit step satisfies the discrete free-energy inequality in
   Section 5.10.5 to solver tolerance.
7. **Circulation gate**: a cycle fixture distinguishes gradient descent from
   skew circulation by measuring nonzero current tangent to an approximately
   constant-free-energy contour; a claimed stationary current additionally
   requires a measured divergence residual.
8. **Identity gate**: a proposed Casimir must be found constructively, satisfy
   $J_5^{\mathrm{macro}}\nabla\mathcal I=0$, resist fast-loop perturbations,
   and still change under
   the declared slow evidence-gated operator. A null vector alone does not pass.
9. **Meta-stability gate**: adversarial and contradictory evidence tests bound
   changes in $\Theta$, precision, topology, and identity across the three time
   scales; self-generated traces are down-weighted relative to independent
   observations.
10. **Resource gate**: every internal inference, planning, reflection, and
    proactive action consumes an explicit computational budget and admits
    `NOOP`; scheduling frequency is not treated as evidence of life.

After these computational gates pass in shadow mode, a preregistered behavioral
ablation should compare SSA-A0, H1D multi-hop transport, bounded H1D/H2, discrete
active inference without skew flow, and the full YUANZI candidate. Archive
snapshots, model versions, random streams, and user-feedback protocols must be
paired. Primary outcomes are path-grounded recall, calibration of asserted
versus questioned claims, policy information gain, identity-rating stability,
and intervention sensitivity. ``Feels alive'' remains a separately measured
human judgment rather than a consequence inferred from lower free energy.

---

## 7. Discussion

### 7.1 What "Stateless" Means (Honest Version)

SSA's inference core is session-stateless. $\mathcal{S}$ is state. The system is not "strictly stateless." The meaningful claim is: **persistence is externalized and append-only, not internalized and mutable.**

### 7.2 The Geometry of Memory

$\mathcal{S}$'s spatial organization means semantically similar traces cluster. Retrieval activates regions, not just individual items. Whether this yields different behavior than timeline-based retrieval is empirical—Generative Agents' retrieval already considers relevance, so the added value of "spatial" framing may be minimal.

### 7.3 Emergent vs. Performed Emotion

We hypothesize that emotion emerges from trace activation rather than being prompted. **This is unverified.** A skeptic could argue that LLMs produce emotional tone regardless of memory architecture, and that SSA's "emergence" is just prompt context effect. Distinguishing these requires controlled experiments.

### 7.4 Limitations

1. **Behavioral validation pending**: The runtime and shadow kernels exist, but no controlled behavioral result is reported.
2. **Physical validation pending**: H1 has a C1 structural analogy only; no joule calibration, hardware telemetry, or physical-energy result is reported.
3. **H2 is C0-M and shadow-only**: Its current evidence concerns software invariants, not identity continuity, emotion, or user-perceived lifelikeness.
4. **Local rather than global certificate**: Hard top-$K$ and capacity replacement are discontinuous at selection boundaries. H2 certifies fixed-support neighborhoods and reports bounded jumps elsewhere.
5. **Human-loop gain is not known a priori**: Full small-gain certification depends on a measured or defensible upper bound for user response sensitivity.
6. **Clustering is model-dependent**: Duplicate suppression and active capacity depend on embedding, cluster revision, semantic fingerprint, and threshold stability.
7. **Archive growth remains**: Fixed active capacity bounds behavioral state but does not solve archive storage, deletion, retention, or governance.
8. **Not novel in components**: Scoring, radiation, reflection, bounded memory, and hysteresis all have prior art.
9. **Evaluation remains open**: "Lifelike" lacks a validated metric, and distributional stability is not equivalent to identity.
10. **Scope and cost**: The design is single-user; full weighted scan, clustering, paired replay, and radiation remain expensive.
11. **H1D is structural and shadow-only**: Directed Markov invariants do not
    establish local detailed balance, reservoir identity, finite entropy
    production on one-way support, or behavioral benefit.
12. **YUANZI-5D is unimplemented**: No runtime currently constructs its
    categorical generative model, policy posterior, skew field, discrete
    gradient, Casimir candidate, or meta-operator update.
13. **FEP is not a life test**: Variational free-energy minimization and a named
    Markov blanket do not establish biological life, consciousness, agency, or
    subjective experience.
14. **Odd-dimensional singularity is insufficient**: The guaranteed null
    direction of a real $5\times5$ skew matrix does not automatically define an
    integrable or behaviorally meaningful identity invariant.
15. **Recursive adaptation adds failure modes**: Updating the operator can
    create self-confirmation, precision collapse, topology drift, reward-like
    preference hacking, or instability across time scales.
16. **Information and physical free energy differ**: The YUANZI objective has
    no current mapping to joules, temperature, heat, work, or a calibrated
    physical reservoir.
17. **P1 uses fixed deterministic features**: Its Chinese/English semantic
    cues and versioned cross weights are auditable but not a learned or
    language-complete semantic model. Current-turn appraisal also adds a
    sequential LLM dependency before the main response.

### 7.5 Ethical and Data Governance Considerations

**v0.2 was insufficient on this.** Corrected:

- **Consent**: The system records all user messages and LLM inputs/outputs. Users must explicitly consent and be informed of what is stored.
- **Deletion**: Users must be able to delete specific traces or the entire $\mathcal{S}$. This violates append-only but is a rights requirement.
- **Retention**: Define maximum retention period. Traces older than $T_{\max}$ should be reviewable and deletable.
- **Third-party data flows**: LLM API calls send trace contents to external providers (DeepSeek, OpenAI). Users must be informed. On-device LLMs (Ollama) eliminate this but reduce quality.
- **Prompt injection**: Activated traces are fed into the LLM prompt. Malicious or accidental trace content could inject instructions. Mitigations: sanitize trace content, use structured prompts, separate system/user context.
- **Telegram correction**: v0.2 claimed Telegram Bot API supports Secret Chat (end-to-end encryption). **This is incorrect.** Bot API does not support Secret Chats. Messages to bots are stored on Telegram servers. For E2E encryption, a custom client (not bot) would be needed, which is a different architecture.

---

## 8. Conclusion

HDSC proposes a testable architecture in which LLM agent persistence is
externalized (append-only spatial memory) and transport is explicit rather than
metaphorical. We have clarified that:
- SSA is not "strictly stateless"—$\mathcal{S}$ is state.
- SSA does not escape the state-behavior mapping—$E(s, A(s, \mathcal{S}), \xi)$ has the same form as $M(S_t, s_t, \epsilon)$.
- SSA's components (relevance×recency×importance scoring, radiation, reflection) have prior art (Generative Agents, A-MEM, HippoRAG, Synapse).
- SSA-A0's *potential* contribution is the specific combination + the session-stateless inference design, but whether this combination yields measurably different behavior is **unproven**.
- H1 is a passive macro transport reference; its structural conservation accounting does not predict microstate or LLM trajectories.
- H1D preserves directed topology and nonreversible currents without importing
  H1's symmetric conclusions; it remains an uncalibrated shadow reference.
- H2 separates the growing archive from a fixed-capacity active probability state, with a null reservoir, cluster-level duplicate suppression, and explicit selection-boundary abstention.
- H2's small-gain result is conditional on component bounds and a stable hard top-$K$ branch; it is not a global stability theorem for the human-agent loop.
- YUANZI-5D proposes a normalized active-inference loop whose four fast fibers
  are governed by skew circulation and dissipative descent while a slower fifth
  coordinate updates the operator itself.
- The fifth-coordinate and Casimir constructions are hypotheses with explicit
  algebraic and experimental gates, not implemented evidence of digital life.
- P1 is an implemented serving control layer that makes local human time and
  current semantic-affect-context interaction replayable; it is not evidence
  that the Markov-blanket or YUANZI hypotheses hold.

This paper is a position statement and architectural proposal, not an empirical
validation. HDSC-H1 remains a C1 shadow reference; HDSC-H1D remains a directed
C0-M/C1-structural shadow; and HDSC-H2 remains a C0-M shadow model with no
prompt effect. Their composition, behavioral admission, and causal comparison
are deferred to HDSC-H3. YUANZI-5D is a later theoretical candidate that must
pass categorical, geometric, discrete-law, circulation, identity, meta-stability,
and resource gates before behavioral admission. Behavioral and physical-energy
results are not reported. The immediate research task is to test the stated
invariants and falsification gates before asking whether any candidate changes
perceived identity or lifelikeness.

---

## References

[1] Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., & Bernstein, M.S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*.

[2] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S., Stoica, I., & Gonzalez, J. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

[3] Zhang, Z. et al. (2024). A Survey on the Memory Mechanism of Large Language Model based Agents. *arXiv preprint*. (First author corrected: Zhang, not Tang.)

[4] Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[5] Asai, A. et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv:2310.11511*.

[6] Jeong, S. et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity. *NAACL 2024*. (Title corrected: "Question Complexity," not "Q-Learning.")

[7] Picard, R.W. (1997). *Affective Computing*. MIT Press.

[8] Plutchik, R. (1980). A General Psychoevolutionary Theory of Emotion. *Emotion: Theory, Research, and Experience*.

[9] Scherer, K.R., Schorr, A., & Johnstone, T. (2001). *Appraisal Processes in Emotion*. Oxford University Press.

[10] Varela, F.J., Thompson, E., & Rosch, E. (1991). *The Embodied Mind*. MIT Press.

[11] Clark, A., & Chalmers, D. (1998). The Extended Mind. *Analysis*, 58(1), 7-19.

[12] Heidegger, M. (1927). *Sein und Zeit*.

[13] Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.

[14] Malkov, Y.A., & Yashunin, D.A. (2018). Efficient and Robust Approximate Nearest Neighbor Search using HNSW. *TPAMI*.

[15] Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127-138.

[16] Xu, W. et al. (2024). A-MEM: Agentic Memory for LLM Agents. *(Reference to be verified.)*

[17] Gutiérrez, B.J. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.

[18] Synapse (2026). Episodic-Semantic Memory with Spreading Activation for LLM Agents. *(Reference to be verified; cited per reviewer note.)*

[19] Parr, T., Pezzulo, G., & Friston, K.J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

[20] Schnakenberg, J. (1976). Network theory of microscopic and macroscopic behavior of master equation systems. *Reviews of Modern Physics*, 48(4), 571-585.

[21] Öttinger, H.C. (2005). *Beyond Equilibrium Thermodynamics*. Wiley.

[22] McLachlan, R.I., Quispel, G.R.W., & Robidoux, N. (1999). Geometric integration using discrete gradients. *Philosophical Transactions of the Royal Society A*, 357, 1021-1045.

---

## Appendix A: Notation (Corrected)

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | Trace space (persistent, append-only) |
| $\mathcal{S}^{\mathrm{arc}}_t$ | H2 immutable archive after turn $t$; distinct from bounded active state |
| $\mu_t$ | H2 fixed-capacity active probability measure |
| $\mathcal{C}_t$ | Set of active semantic clusters, $\lvert\mathcal{C}_t\rvert\le C$ |
| $C$ | Maximum number of non-null active clusters |
| $z_c$ | Immutable unit-norm prototype of a versioned semantic cluster |
| $\bot,\delta_\bot$ | Null reservoir and its point mass |
| $w_{c,t},w_{\bot,t}$ | Cluster and null masses; nonnegative and summing to one |
| $D,Q(D)$ | Candidate multiset and its multiplicity-invariant cluster proposal operator |
| $q_t^{H2}$ | H2 multiplicity-invariant proposal over unique novel clusters; written $q_t$ within Section 5.8 |
| $\alpha_t$ | Bounded one-turn proposal write mass |
| $\beta$ | Direct active-state retention coefficient, $0\le\beta<1$ |
| $T_t$ | Non-expansive active-state transport; identity in minimum H2 |
| $\Pi_C$ | Capacity projection that moves overflow mass to $\bot$ |
| $\Delta_z$ | Minimum normalized-embedding distance to a SimHash assignment hyperplane |
| $\Delta_q$ | Positive/zero/top-$K$ proposal-score support margin |
| $\Delta_C$ | Capacity margin under mass plus incumbent-hysteresis priority |
| $S_q,m$ | Selected positive-score sum and selected support size |
| $L_{q,g}=mL_g/S_q$ | Fixed-support TV gain of proposal normalization from raw scores |
| $\epsilon_{\mathrm{support}}$ | Local radius in the declared max-product selection-source metric |
| $F$ | One-turn response map, $F(s,\mu,\xi)=E(s,A(s,\mu),\xi)$ |
| $U$ | User/environment feedback map for the next signal |
| $a,b,u,c,d,e$ | Branch-local component gain bounds used by H2 |
| $\mathbf{G}$ | Joint user-agent-memory small-gain comparison matrix |
| $\rho(\mathbf{G})$ | Spectral radius of the comparison matrix |
| $R$ | H1D nonnegative directed rate matrix with convention $R_{ij}$ = rate $j\to i$ |
| $Q$ | H1D column-conservative Metzler generator, $\mathbf1^TQ=0$ |
| $P_t=e^{tQ}$ | H1D directed Markov semigroup |
| $\pi$ | Strictly positive candidate stationary distribution satisfying $Q\pi=0$ |
| $F_{ij}=R_{ij}\pi_j$ | Stationary directed flux from $j$ to $i$ |
| $\sigma$ | H1D finite structural entropy-production diagnostic when reciprocal support exists |
| $t_i$ | A trace, $t_i = (c_i, v_i, \tau_i, \rho_i, \eta_i)$ |
| $c_i$ | Raw content (text) — **added in v0.3** |
| $v_i$ | Semantic vector, $v_i = \phi(c_i)$ |
| $\tau_i$ | Write timestamp |
| $\rho_i(t)$ | Freshness at query time $t$: $e^{-\lambda(t-\tau_i)}$ — **derived, not stored** |
| $\eta_i$ | Importance (fixed at write) |
| $s$ | Current signal |
| $A$ | Activation function |
| $E$ | Emergence function (LLM) |
| $W$ | Write operation |
| $L$ | Reflection loop |
| $\phi$ | Embedding model |
| $\xi$ | LLM sampling randomness |
| $\mathcal{T}_{\text{act}}$ | Activated trace set |
| $S_t$ | State vector (stateful paradigm) |
| $M$ | State-behavior mapping |
| $\eta_t,s_t,\mu_t,a_t$ | YUANZI external, sensory, internal, and active Markov-blanket states |
| $q_t^{Y}$ | YUANZI normalized categorical posterior; written $q_t$ within Section 5.10 |
| $\mathbf A$ | YUANZI likelihood matrix $p(o\mid x)$; distinct from legacy activation function $A$ |
| $\mathbf B^a$ | Action-conditioned, column-normalized, generally asymmetric latent transition matrix |
| $\mathbf C,\mathbf D,\mathbf E$ | Preference, initial-state, and policy-habit distributions |
| $F_t(q)$ | Variational free energy for current categorical inference |
| $G_t(\pi)$ | Expected free energy of policy $\pi$ |
| $\mathcal M_4$ | Four-fiber sensation-action-memory-prediction state space |
| $\Theta_t$ | Fifth-coordinate meta-state containing geometry and operator parameters |
| $\mathcal M_5^{\mathrm{op}}$ | Operator-parameterized bundle over admissible $\Theta$; the full update is bidirectionally coupled |
| $y_t$ | YUANZI logits with $q_t=\operatorname{softmax}(y_t)$ |
| $\mathsf K_t,C_c,\kappa_{t,c}$ | Weighted skew cycle field, oriented cycle basis, and cycle weights |
| $W_t,H_t$ | Positive resistance metric and positive-semidefinite dissipation seed |
| $\Omega_t,\Gamma_t$ | Weighted skew-circulation and symmetric dissipative operators |
| $\mathsf U_tu_t,d\mathcal B_t$ | External drive and Wiener increment in the continuous YUANZI candidate |
| $J_5^{\mathrm{macro}}$ | Reduced $5\times5$ skew coupling; its parity result does not apply automatically to the full bundle |
| $\mathcal I$ | Candidate slow Casimir identity functional, conditional on integrability |
| $\mathfrak Y^{[1]},\mathfrak Y^{[2]}$ | Fast state operator and slower operator-update meta-operator |

## Appendix B: Revision Log (v0.2 to v0.7)

| Issue | Earlier Claim or Design | v0.7 Status |
|-------|-------------------------|-------------|
| Statelessness | "Strictly stateless" | "Session-stateless inference core + external persistent memory" |
| Deterministic mapping defect | "SSA eliminates it" | "SSA has the same form; difference is state representation, not mapping structure" |
| §3.4 proof | $A(s, \mathcal{S}_{t_1}) \neq A(s, \mathcal{S}_{t_2})$ | Retracted; new traces may not enter top-K |
| Freshness ranking | Implied $\rho$ changes ranking | Constant $e^{-\lambda t}$ cancels; ranking by $e^{\lambda \tau_i}$ |
| Trace definition | 4-tuple (no $c_i$) | 5-tuple (with $c_i$) |
| Activation pseudocode | KNN-then-filter | Weighted composite score, then top-K |
| Reflection traces | "Generate vectors" | Generate text; vector is derived |
| Invariants | 4 invariants, with conflicts | 4 invariants, conflicts resolved |
| Comparison table | Non-uniform reset | Uniform: "clear session" vs "clear all memory" |
| Novelty | "First stateless spatial paradigm" | "Specific combination; components have prior art" |
| Experiments | "Preliminary evidence" | Interactive runtime and shadow kernels exist; no behavioral or physical result is reported |
| Experiment design | Confounded (reflection + LLM in Week 2) | Factorized 2×2 design |
| Evaluation metrics | Flawed proxies | Acknowledged as open problem |
| MemGPT authors | Incorrect list | Corrected (added Wooders, Lin) |
| Survey first author | "Tang" | "Zhang" |
| Adaptive-RAG | "Q-Learning" | "Question Complexity" |
| Ref [15] | Unlinked | Integrated into FEP/active-inference related work and bounded explicitly |
| Telegram Secret Chat | Claimed supported | Corrected: Bot API doesn't support it |
| Data governance | Minimal | Added consent, deletion, retention, injection sections |
| Synapse/A-MEM/HippoRAG | Not cited | Added as prior art |
| Project identity | SSA presented as the final theory | Renamed HDSC; `legacy-ssa-a0` retained only as a frozen baseline |
| Thermodynamic language | Software activation and physical energy were insufficiently separated | C0-M computational, C1 structural, and C2 physical claim levels are separated |
| H1 scope | Deterministic propagation could be read as a full-system model | H1 is a fixed-input macro transport reference with microstate status `unmodeled` |
| Closed-loop derivative | Informal $\partial r_t/\partial r_{t-1}\ne0$ | Replaced by mediated causal influence, distribution distances, and joint gain accounting |
| Archive versus behavior | Growing trace space also served as effective state | Immutable archive and bounded active probability state are distinct objects |
| Duplicate evidence | Repeated traces could accumulate activation mass | H2 uses max-per-cluster proposals and suppresses the currently active bucket; no archive-wide fingerprint registry is claimed |
| Fixed capacity | No bounded behavioral state | H2 uses at most $C$ active clusters plus a null reservoir |
| Hard top-K | Selection treated as if globally smooth | Certificates are local to a stable-support margin; boundary cases abstain and report jump bounds |
| Support radius | Proposal and capacity margins were treated as directly interchangeable | Cluster, raw-score, and prior-state coordinates use an explicit max-product metric; normalization gain maps score changes into capacity mass |
| Replay identity | Online appraisal novelty and replay novelty could diverge | Innovation is archive-derived, measure and replay hashes are split, and processed step must equal archive count |
| Semantic partition | Codebook changes were not part of state configuration | Embedding model/dimension, SimHash bits/version, and hyperplanes determine the semantic partition hash |
| H2 deployment | Bounded state could be mistaken for a behavioral feature | H2 is C0-M, post-write, shadow-only, and has no prompt effect |
| H1/H2 composition | Transport and active memory could be read as one validated kernel | Composition and behavioral admission are deferred to HDSC-H3 |
| Directed topology | H1's reciprocal diffusion could be applied to asymmetric links | H1D preserves full direction with a column-conservative nonreversible generator; no symmetrization |
| Multi-hop association | Legacy activation stops after one semantic radiation hop | YUANZI proposes typed asymmetric transition powers with path provenance; this remains unimplemented |
| Free-energy loop | Reflection scheduling was treated as the primary sign of autonomous activity | Perception minimizes $F$, action evaluates expected free energy $G$, and `NOOP` plus resource cost are explicit |
| Asymmetry versus skew symmetry | Directed evidence transport and cycle circulation were not separated | Nonnegative asymmetric Markov transitions and signed skew tangent currents are distinct operators |
| Dimensionality | State and transport lived on a single fixed geometry | YUANZI proposes four dynamic fibers in an operator-parameterized bundle with a fifth law coordinate |
| Identity invariant | Identity continuity relied on bounded active-state heuristics | A Casimir is proposed only as a conditional candidate requiring integrability, constructive recovery, and drift tests |
| Self-modification | The transition operator was fixed during a modeled step | A slower projected meta-update may revise the fast operator under three-time-scale constraints |
| YUANZI evidence status | No prior implementation or experiment | Theory-only in v0.7; categorical, geometric, discrete-law, circulation, identity, and resource gates are specified |
| Environment perception | UTC timestamps, appraisal, and context were separate runtime inputs | Software v0.5 persists a P1 human-clock and semantic-affect-context-time cross-operator snapshot before response generation |
