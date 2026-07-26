# Stateless Space Activation (SSA):
# Session-Stateless Inference with Episodic-Spatial Memory for LLM Agents

> Draft v0.3 — 2026-07-25
> Status: Position paper / research preview. No empirical results yet. System unimplemented.
> Note on revision: v0.2 claimed "strict statelessness" and preliminary experimental evidence. Both claims were retracted after review. This version repositions SSA's contribution and removes all unverified assertions.

---

## Abstract

We present **Stateless Space Activation (SSA)**, an architecture for LLM-based agents that separates a *session-stateless inference core* from an *external episodic-spatial memory*. Unlike stateful agents whose behavior is governed by an in-place-updated state vector (working memory, reflection hierarchy, emotion parameters), SSA's inference core holds no state across requests: each response is produced by activating a region of a persistent trace space $\mathcal{S}$ and feeding the activated traces to an LLM.

We do not claim SSA eliminates state entirely—$\mathcal{S}$ is persistent and influences future outputs, which is a form of state. SSA's actual contribution is architectural: it replaces **in-place state mutation** with **append-only spatial memory** whose geometry (not a vector) shapes behavior. This shifts the locus of "personality" from a mutable vector to an immutable history, with consequences for reset semantics, drift, and emotion.

We clarify that SSA does **not** escape the state-behavior mapping formalism—behavior is still $E(s, A(s, \mathcal{S}), \xi)$ where $\xi$ is sampling noise. SSA's distinction is that $\mathcal{S}$ is high-dimensional, append-only, and spatially organized, versus the low-dimensional, mutable vectors of stateful approaches. Whether this distinction yields measurably different agent behavior is an open empirical question this paper poses but does not answer.

We position SSA relative to prior work—Generative Agents' relevance×recency×importance scoring, A-MEM's associative linking, HippoRAG's graph retrieval, and Synapse's spreading activation—and identify what, if anything, is novel. We propose a factorized experimental design (reflection on/off × LLM A/B) and discuss why evaluating "lifelike" behavior requires new metrics beyond embedding similarity.

**Keywords**: LLM agents, episodic memory, spatial retrieval, agent architecture, position paper

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
- This paper **presents no experimental results**. The system is unimplemented. We describe the architecture, position it against prior work, and propose an experimental design. All claims about behavior are hypotheses, not findings.

**Contributions** (revised, modest):
1. We articulate an architectural alternative—session-stateless inference + append-only spatial memory—that distinguishes *where* persistence lives (external append-only vs. internal mutable), even though both are "state" in the formal sense.
2. We propose a concrete instance of this architecture ($\langle \mathcal{S}, A, E, W, L \rangle$) with formal definitions.
3. We clarify the reset semantics distinction: SSA's inference core can be reset (cleared) while $\mathcal{S}$ persists, whereas stateful agents' resets typically clear $S_t$. We note this is a *design choice* in what "reset" means, not an inherent superiority.
4. We propose a factorized experimental design and discuss evaluation challenges for "lifelike" agent behavior.

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

## 5. Implementation (Design, Not Yet Built)

**Status: Unimplemented.** The following is the intended design. `src/` and `experiments/` are empty.

### 5.1 Intended Architecture

```
┌─────────────────────────────────────────────────┐
│                  SSA Runtime                     │
│                                                  │
│  Space S: SQLite + sqlite-vec                   │
│  Embedding φ: bge-small-zh-v1.5 (512-dim)       │
│  Activation A: weighted KNN + radiation          │
│  Emergence E: LiteLLM → DeepSeek (default)       │
│  Reflection L: APScheduler, 15-min interval      │
│  Interface: Telegram Bot / CLI                   │
└─────────────────────────────────────────────────┘
```

### 5.2 Intended Trace Schema

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

---

## 6. Proposed Experiments (Design Only, No Results)

**We have no experimental results.** The system is unimplemented. The following is a proposed design.

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

---

## 7. Discussion

### 7.1 What "Stateless" Means (Honest Version)

SSA's inference core is session-stateless. $\mathcal{S}$ is state. The system is not "strictly stateless." The meaningful claim is: **persistence is externalized and append-only, not internalized and mutable.**

### 7.2 The Geometry of Memory

$\mathcal{S}$'s spatial organization means semantically similar traces cluster. Retrieval activates regions, not just individual items. Whether this yields different behavior than timeline-based retrieval is empirical—Generative Agents' retrieval already considers relevance, so the added value of "spatial" framing may be minimal.

### 7.3 Emergent vs. Performed Emotion

We hypothesize that emotion emerges from trace activation rather than being prompted. **This is unverified.** A skeptic could argue that LLMs produce emotional tone regardless of memory architecture, and that SSA's "emergence" is just prompt context effect. Distinguishing these requires controlled experiments.

### 7.4 Limitations

1. **Unimplemented**: No system, no data.
2. **Not novel in components**: Scoring, radiation, reflection all have prior art.
3. **Evaluation undefined**: "Lifelike" lacks validated metrics.
4. **Single-user design**: Multi-user unaddressed.
5. **Cost**: Full weighted scan + radiation is expensive; ANN with composite scoring is non-trivial.
6. **Forgetting**: Append-only growth is unsustainable; forgetting policies are undefined.

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

SSA proposes that LLM agent persistence should be externalized (append-only spatial memory) rather than internalized (mutable state vector), with a session-stateless inference core. We have clarified that:
- SSA is not "strictly stateless"—$\mathcal{S}$ is state.
- SSA does not escape the state-behavior mapping—$E(s, A(s, \mathcal{S}), \xi)$ has the same form as $M(S_t, s_t, \epsilon)$.
- SSA's components (relevance×recency×importance scoring, radiation, reflection) have prior art (Generative Agents, A-MEM, HippoRAG, Synapse).
- SSA's *potential* contribution is the specific combination + the session-stateless inference design, but whether this combination yields measurably different behavior is **unproven**.

This paper is a position statement and architectural proposal, not an empirical validation. The system is unimplemented; experiments are proposed but not conducted. We invite the community to test whether the design choices SSA proposes—append-only vs. mutable, spatial vs. temporal, session-stateless core vs. stateful core—make a measurable difference.

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

[15] Friston, K. (2010). The Free-Energy Principle. *Nature Reviews Neuroscience*. (Note: cited in v0.2's related_work.md but not referenced in paper body. Retained for completeness; will integrate or remove in next revision.)

[16] Xu, W. et al. (2024). A-MEM: Agentic Memory for LLM Agents. *(Reference to be verified.)*

[17] Gutiérrez, B.J. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.

[18] Synapse (2026). Episodic-Semantic Memory with Spreading Activation for LLM Agents. *(Reference to be verified; cited per reviewer note.)*

---

## Appendix A: Notation (Corrected)

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | Trace space (persistent, append-only) |
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

## Appendix B: Revision Log (v0.2 → v0.3)

| Issue | v0.2 Claim | v0.3 Correction |
|-------|-----------|-----------------|
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
| Experiments | "Preliminary evidence" | "No results; system unimplemented" |
| Experiment design | Confounded (reflection + LLM in Week 2) | Factorized 2×2 design |
| Evaluation metrics | Flawed proxies | Acknowledged as open problem |
| MemGPT authors | Incorrect list | Corrected (added Wooders, Lin) |
| Survey first author | "Tang" | "Zhang" |
| Adaptive-RAG | "Q-Learning" | "Question Complexity" |
| Ref [15] | Unlinked | Noted; will integrate or remove |
| Telegram Secret Chat | Claimed supported | Corrected: Bot API doesn't support it |
| Data governance | Minimal | Added consent, deletion, retention, injection sections |
| Synapse/A-MEM/HippoRAG | Not cited | Added as prior art |
