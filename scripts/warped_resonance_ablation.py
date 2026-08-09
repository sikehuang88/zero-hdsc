"""Falsification runs for affect-warped free-energy retrieval.

Three experiments, each able to kill the design:

1. Temperature sweep — do the three regimes actually appear, or is the
   temperature axis arbitrary?
2. Flat versus warped — does a position-dependent metric beat an additive one
   on held-out retrodiction, or is the coupling decoration?
3. Temperature versus arousal — does the fitted optimum track the archive's
   recorded arousal, or is the emotional story unattached to the mathematics?

Experiment 2 is a positive control on synthetic data where affect carries
information by construction.  It can show the machinery is *able* to exploit
the coupling; it cannot show that a real archive contains it.  That needs real
traces and is not claimed here.

Run: uv run python scripts/warped_resonance_ablation.py
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ssa.hdsc.free_energy_resonance import (
    ResonanceConfig,
    resonance_scan,
    retrieval_temperature,
)
from ssa.hdsc.warped import WarpParameters, build_warped_conductance

DIMENSION = 24
TOPICS = 6


def synthetic_archive(
    count: int,
    seed: int,
    *,
    affect_informative: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Archive where each topic has a characteristic affect signature.

    When ``affect_informative`` is False the affect column is shuffled, which
    destroys the coupling while preserving every marginal distribution.  That
    is the honest control: any advantage the warped metric shows must vanish.
    """

    rng = np.random.default_rng(seed)
    topic_content = rng.normal(size=(TOPICS, DIMENSION))
    topic_affect = np.column_stack(
        [
            rng.uniform(-0.9, 0.9, TOPICS),
            rng.uniform(0.1, 0.9, TOPICS),
            rng.uniform(0.0, 0.8, TOPICS),
        ]
    )

    labels = np.zeros(count, dtype=np.int64)
    state = 0
    for index in range(count):
        if rng.random() < 0.3:
            state = int(rng.integers(TOPICS))
        labels[index] = state

    content = topic_content[labels] + 0.55 * rng.normal(size=(count, DIMENSION))
    affect = topic_affect[labels] + 0.10 * rng.normal(size=(count, 3))
    affect[:, 1] = np.clip(affect[:, 1], 0.02, 1.0)

    if not affect_informative:
        affect = affect[rng.permutation(count)]

    # Consolidation spread: repeated topics are remembered more sharply.
    counts = np.bincount(labels, minlength=TOPICS).astype(np.float64)
    spread = 0.9 / (1.0 + 0.25 * counts[labels])
    return content, affect, labels, spread


def experiment_one_temperature_regimes() -> None:
    print("=" * 78)
    print("实验 1  温度扫描 — 三个区间是否真的出现")
    print("=" * 78)
    content, affect, _, spread = synthetic_archive(60, seed=11, affect_informative=True)
    conductance = build_warped_conductance(content, affect, WarpParameters(beta=3.0))
    seed_mass = np.zeros(content.shape[0])
    seed_mass[0] = 1.0
    config = ResonanceConfig(recall_count=6, surfacing_detuning_cap=1e9)

    print(f"{'T':>8} {'分布熵':>9} {'浮现条数':>9} {'top1概率':>10}   形态")
    print("-" * 78)
    for temperature in (0.01, 0.1, 0.3, 0.6, 1.0, 2.0, 8.0, 40.0):
        result = resonance_scan(
            conductance,
            seed_mass,
            spread,
            temperature=temperature,
            content_dimension=DIMENSION,
            config=config,
            rng=np.random.default_rng(1),
        )
        top = max(result.probability)
        entropy = result.distribution_entropy
        regime = "刚性" if entropy < 1.0 else ("头脑风暴" if entropy < 3.0 else "泛激活")
        print(
            f"{temperature:>8.2f} {entropy:>9.3f} {len(result.selected):>9d} "
            f"{top:>10.3f}   {regime}"
        )
    print(f"\n均匀分布熵上界 = log(59) = {math.log(59):.3f}")


def _bandwidth_matched(
    content: NDArray[np.float64],
    affect: NDArray[np.float64],
    *,
    beta: float,
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Warped conductance whose mean edge weight matches the flat graph.

    Without this the comparison is confounded.  The conformal factor satisfies
    ``f ≥ 1``, so any ``beta > 0`` lengthens every edge, thins the graph, and
    localises the diffusion — an advantage that has nothing to do with affect
    and that survives shuffling the affect column.  Matching mean conductance
    removes that channel, so whatever difference is left is attributable to
    the coupling itself.
    """

    mask = ~np.eye(target.shape[0], dtype=bool)
    goal = float(target[mask].mean())
    low, high = 0.05, 20.0
    bandwidth = 1.0
    for _ in range(60):
        bandwidth = 0.5 * (low + high)
        candidate = build_warped_conductance(
            content, affect, WarpParameters(beta=beta, bandwidth=bandwidth)
        )
        if float(candidate[mask].mean()) < goal:
            low = bandwidth
        else:
            high = bandwidth
    return build_warped_conductance(content, affect, WarpParameters(beta=beta, bandwidth=bandwidth))


def _predict(
    conductance: NDArray[np.float64],
    spread: NDArray[np.float64],
    labels: NDArray[np.int64],
    query: int,
    temperature: float,
    config: ResonanceConfig,
) -> float:
    """Probability that the query's topic recurs, from what resonance recalls.

    The prediction is the probability mass the scan puts on same-topic
    memories.  Better retrieval therefore means a better-calibrated forecast,
    which is what makes retrieval quality measurable at all.
    """

    seed_mass = np.zeros(conductance.shape[0])
    seed_mass[query] = 1.0
    result = resonance_scan(
        conductance,
        seed_mass,
        spread,
        temperature=temperature,
        content_dimension=DIMENSION,
        config=config,
        rng=np.random.default_rng(query),
    )
    if not result.surfaced:
        return 0.5
    probability = np.asarray(result.probability)
    same = probability[labels == labels[query]].sum()
    return float(min(1.0, max(0.0, same)))


def experiment_two_flat_versus_warped() -> None:
    print()
    print("=" * 78)
    print("实验 2  平坦(对角) vs 翘曲(耦合) — 留出窗口上的 Brier")
    print("=" * 78)
    config = ResonanceConfig(recall_count=8, surfacing_detuning_cap=1e9)
    temperature = 0.8

    print(
        f"{'archive':>8} {'情绪含信息':>11} {'仅内容':>9} {'+情绪(加性)':>12} {'+翘曲(曲率)':>12}"
    )
    print("-" * 78)
    additive_real: list[float] = []
    additive_control: list[float] = []
    curvature_real: list[float] = []
    curvature_control: list[float] = []
    for archive_seed in (3, 11, 23, 42, 77):
        for informative in (True, False):
            content, affect, labels, spread = synthetic_archive(
                70, seed=archive_seed, affect_informative=informative
            )
            count = content.shape[0]
            # Time-ordered split: fit nothing, but only score the later half so
            # the comparison never reads a window used to pick parameters.
            held_out = range(count // 2, count)

            # Arm 1: content only. Affect contributes nothing at all.
            no_affect = WarpParameters(
                beta=0.0,
                valence_weight=0.0,
                arousal_weight=0.0,
                tension_weight=0.0,
                sign_flip_penalty=0.0,
            )
            content_only = build_warped_conductance(content, affect, no_affect)
            # Arm 2: affect as an additive axis — the weighted-sum baseline.
            flat = build_warped_conductance(content, affect, WarpParameters(beta=0.0))
            # Arm 3: affect position curves the content metric.
            warped = _bandwidth_matched(content, affect, beta=4.0, target=flat)

            briers: dict[str, float] = {}
            for name, conductance in (
                ("content", content_only),
                ("flat", flat),
                ("warped", warped),
            ):
                total = 0.0
                for query in held_out:
                    probability = _predict(conductance, spread, labels, query, temperature, config)
                    # Ground truth: does the same topic appear again later?
                    future = labels[query + 1 : query + 12]
                    outcome = 1.0 if bool(np.any(future == labels[query])) else 0.0
                    total += (probability - outcome) ** 2
                briers[name] = total / len(list(held_out))

            additive = briers["content"] - briers["flat"]
            curvature = briers["flat"] - briers["warped"]
            if informative:
                additive_real.append(additive)
                curvature_real.append(curvature)
            else:
                additive_control.append(additive)
                curvature_control.append(curvature)
            print(
                f"{archive_seed:>8} {informative!s:>11} {briers['content']:>9.4f} "
                f"{briers['flat']:>12.4f} {briers['warped']:>12.4f}"
            )

    print("-" * 78)
    print("增益分解 (正 = 该层带来改善):")
    print(
        f"  加性情绪层  含信息 {np.mean(additive_real):+.4f}   "
        f"打乱后 {np.mean(additive_control):+.4f}"
    )
    print(
        f"  曲率层      含信息 {np.mean(curvature_real):+.4f}   "
        f"打乱后 {np.mean(curvature_control):+.4f}"
    )
    print()
    print("读法:打乱情绪保留全部边缘分布、只破坏耦合。")
    print("某一层在打乱后仍有同等增益,则该层的增益与情绪无关,不能归因于耦合。")


def experiment_three_temperature_tracks_arousal() -> None:
    print()
    print("=" * 78)
    print("实验 3  最优温度是否跟随唤醒度")
    print("=" * 78)
    config = ResonanceConfig(recall_count=8, surfacing_detuning_cap=1e9)
    grid = (0.05, 0.15, 0.4, 0.8, 1.5, 3.0, 6.0, 15.0)

    print(f"{'唤醒度':>8} {'拟合最优T':>11} {'公式给出T':>11}")
    print("-" * 78)
    fitted: list[float] = []
    arousal_levels = (0.1, 0.3, 0.5, 0.7, 0.9)
    for arousal in arousal_levels:
        # Archives whose affect sits at the given arousal: the sharper the
        # emotional focus, the colder the search should want to be.
        content, affect, labels, spread = synthetic_archive(
            70, seed=int(arousal * 100) + 5, affect_informative=True
        )
        affect[:, 1] = np.clip(affect[:, 1] * 0.3 + arousal, 0.02, 1.0)
        # Higher arousal ⇒ noisier affect ⇒ a flatter, hotter landscape.
        noise = 0.05 + 0.35 * arousal
        rng = np.random.default_rng(9)
        affect = affect + noise * rng.normal(size=affect.shape)
        affect[:, 1] = np.clip(affect[:, 1], 0.02, 1.0)
        conductance = build_warped_conductance(content, affect, WarpParameters(beta=4.0))

        best_temperature, best_brier = grid[0], math.inf
        for temperature in grid:
            total = 0.0
            queries = range(content.shape[0] // 2, content.shape[0])
            for query in queries:
                probability = _predict(conductance, spread, labels, query, temperature, config)
                future = labels[query + 1 : query + 12]
                outcome = 1.0 if bool(np.any(future == labels[query])) else 0.0
                total += (probability - outcome) ** 2
            brier = total / len(list(queries))
            if brier < best_brier:
                best_temperature, best_brier = temperature, brier
        fitted.append(best_temperature)
        formula = retrieval_temperature(arousal=arousal, tension=0.2, energy=0.5)
        print(f"{arousal:>8.2f} {best_temperature:>11.2f} {formula:>11.3f}")

    correlation = float(np.corrcoef(np.array(arousal_levels), np.array(fitted))[0, 1])
    print("-" * 78)
    print(f"corr(唤醒度, 拟合最优T) = {correlation:+.3f}")
    print("正相关支持温度的情绪解释;≈0 或为负则这层解释是装饰。")


def experiment_four_contentless_query() -> None:
    """The regime the design actually targets: a query with no content signal.

    Experiment 2 scores topic recurrence, where the content vector already
    identifies the topic.  Affect there is a lower-SNR copy of a signal that is
    already clean, so mixing it in can only dilute — which is what it did.
    "Haven't we met somewhere" is the opposite case: content matches nothing,
    and the only usable trace of the target is its emotional colour.  If affect
    does not help *here*, it has no regime left.
    """

    print()
    print("=" * 78)
    print("实验 4  无内容信号的查询 — 设计真正针对的场景")
    print("=" * 78)
    config = ResonanceConfig(recall_count=8, surfacing_detuning_cap=1e9)
    rng = np.random.default_rng(5)

    print(
        f"{'archive':>8} {'情绪含信息':>11} {'仅内容':>9} {'+情绪(加性)':>12} {'+翘曲(曲率)':>12}"
    )
    print("-" * 78)
    hits: dict[tuple[bool, str], list[float]] = {}
    for archive_seed in (3, 11, 23, 42, 77):
        for informative in (True, False):
            content, affect, labels, spread = synthetic_archive(
                70, seed=archive_seed, affect_informative=informative
            )
            no_affect = WarpParameters(
                beta=0.0,
                valence_weight=0.0,
                arousal_weight=0.0,
                tension_weight=0.0,
                sign_flip_penalty=0.0,
            )
            arms = {
                "content": build_warped_conductance(content, affect, no_affect),
                "flat": build_warped_conductance(content, affect, WarpParameters(beta=0.0)),
            }
            arms["warped"] = _bandwidth_matched(content, affect, beta=4.0, target=arms["flat"])

            scores: dict[str, float] = {}
            for name in arms:
                correct = 0
                trials = 0
                for target_topic in range(TOPICS):
                    members = np.flatnonzero(labels == target_topic)
                    if members.size < 3:
                        continue
                    # The probe: content is pure noise, affect matches the
                    # target topic. Appended as one extra node.
                    probe_content = rng.normal(size=(1, DIMENSION)) * 1.5
                    probe_affect = affect[members].mean(axis=0, keepdims=True)
                    ext_content = np.vstack([content, probe_content])
                    ext_affect = np.vstack([affect, probe_affect])
                    parameters = (
                        no_affect
                        if name == "content"
                        else WarpParameters(beta=0.0 if name == "flat" else 4.0)
                    )
                    extended = build_warped_conductance(ext_content, ext_affect, parameters)
                    seed_mass = np.zeros(extended.shape[0])
                    seed_mass[-1] = 1.0
                    ext_spread = np.append(spread, float(spread.mean()))
                    result = resonance_scan(
                        extended,
                        seed_mass,
                        ext_spread,
                        temperature=0.4,
                        content_dimension=DIMENSION,
                        config=config,
                        rng=np.random.default_rng(target_topic),
                    )
                    trials += 1
                    recalled = [index for index in result.selected if index < labels.size]
                    if recalled:
                        share = sum(1 for index in recalled if labels[index] == target_topic) / len(
                            recalled
                        )
                        correct += share
                scores[name] = correct / max(trials, 1)
                hits.setdefault((informative, name), []).append(scores[name])
            print(
                f"{archive_seed:>8} {informative!s:>11} {scores['content']:>9.3f} "
                f"{scores['flat']:>12.3f} {scores['warped']:>12.3f}"
            )

    print("-" * 78)
    print("命中率 = 召回集中属于目标话题的比例 (随机基线 = 1/6 ≈ 0.167)")
    for informative in (True, False):
        label = "含信息" if informative else "打乱后"
        row = " ".join(
            f"{name}={np.mean(hits[(informative, name)]):.3f}"
            for name in ("content", "flat", "warped")
        )
        print(f"  情绪{label}:  {row}")


def main() -> None:
    experiment_one_temperature_regimes()
    experiment_two_flat_versus_warped()
    experiment_three_temperature_tracks_arousal()
    experiment_four_contentless_query()


if __name__ == "__main__":
    main()
