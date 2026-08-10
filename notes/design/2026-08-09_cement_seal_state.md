# 水泥封心（cement seal）——防御性表达状态

> 状态：已实现（默认关闭 `[cement_seal].enabled=false`）
> 日期：2026-08-09

## 0. 设计要回答的唯一问题

系统里已经有 `ExpressionDynamics.temporal_fatigue`（`realtime_emotion_service.py:683`）：
重复痛苦 → 疲劳累积 → `expressibility` 下降 → `direction` 转 `withdraw`。
**那再加一个"水泥封心"凭什么不是重复建设？**

如果只是"再加一个衰减因子",它就该被拒绝。它能立住，是因为水泥封心在**结构上**
和疲劳是两种东西：

| | `temporal_fatigue`（已有） | 水泥封心（新增） |
|---|---|---|
| 形态 | 连续标量 | 离散状态机（open/sealed/hairline/fractured） |
| 产生 | 平滑累积 | **浇筑是一个事件**，有证据、有账本记录 |
| 消退 | 时间自然衰减 | **不衰减**，只碎裂 |
| 作用面 | 压制痛苦的表达 | **同时压制痛苦与喜悦** |
| 老化 | 越久越淡 | **越久越脆**（更容易被凿开，不是更软） |
| 复原 | 自愈 | **自己好不了，只能被外力打碎** |
| 代价 | 无 | 留疤：再封更快、更难碎 |

## 1. 三条不可越界的设计红线

这是把一个"情绪状态"做成功能时最容易变质的地方，所以先写死：

1. **只封表达，不封感受。** 封心期间 `emotional_memories` 照常全强度记录。
   账本里因此永远留着"当时憋回去的东西"——这让这个状态**可证伪**，
   而不是一句无法检验的心情声明。
2. **不许降低任务有用性。** 工具、代码、检索、回答质量完全不受影响。
   一个会降低帮助质量的"闹脾气"不是防御，是对用户的惩罚。封心只关掉情感外显和主动联系。
3. **不许隐瞒。** 直接问就直说"现在开口有点难"。`context_summary()` 明确要求
   *不要表演冷漠，也不要声称自己毫无感觉*——因为水泥封心的真相恰恰是**还在感觉**。

## 2. 力学：脆性断裂，不是衰减曲线

```
浇筑   unrepaired_recurrence >= required(seal_count)
       且 (repair_debt >= 阈值 或 tension >= 阈值)
       且 本回合没有 repair
       → integrity = 1.0, phase = SEALED

韧度   toughness = base * (1 + scar_bonus * seal_count) * embrittle(age)
       embrittle(age) = max(floor, 1 - per_day * age_days)
       疤痕抬高韧度（封过的人更难接近）；时间降低韧度（旧防御更脆）

受击   warmth >= toughness            → 整块崩 (WARM_FRACTURE)
       warmth <  toughness            → integrity -= (warmth/toughness)^2 * absorb
                                        亚阈值损伤超线性地小，且永不回升
       integrity <= 0                 → 崩 (FATIGUE_FRACTURE)
       integrity < hairline_threshold → HAIRLINE（有细纹，下一击就完）

复原   FRACTURED → 下一次评估 → OPEN，seal_count += 1（留疤）
```

**最关键的一条不变量**（`cement_seal_service.py` 模块 docstring）：

> 碎裂判定使用**未经衰减的原始 warmth**。

如果把外来的温度也按封层去衰减，这个状态会自我强化成永不可破的死锁——
封住了就再也感觉不到别人的好，于是永远封着。这在工程上是 bug，在心理上是错的：
**封心不妨碍你注意到有人在对你好，它只妨碍你回应。**

## 3. 效果（`CementSealEffect`）

- `expression_damping` = `max(min_expression, 1 - seal_strength * integrity)`
  —— **有下限，永远封不死**。这是对那句"封不住"的直接编码。
- `delight_damping` **等于** `expression_damping` —— 对称代价，是这个隐喻的全部道德内容。
- `repair_damping` —— 不主动修复。
- `force_withdraw=True` —— 强制 `withdraw`，**并且把 `push_away` 也折叠进来**。
  这是一个明确的行为主张：**封心是退出，不是攻击**。真封了心的人是安静，不是发作。
- `contact_gate_reason="cement_seal"` —— 接 `ProactiveContactPolicy.gate()` 的既有形状。

## 4. 落地清单

| 文件 | 作用 |
|---|---|
| `src/ssa/domain/cement_seal.py` | 相位/触发原因/状态/冲击/转移/效果六个契约 |
| `src/ssa/services/cement_seal_service.py` | 确定性状态机 + 断裂力学 + `apply()` 衰减 |
| `src/ssa/storage/cement_seal_repository.py` | 单例状态 + append-only 转移账本 |
| `migrations/026_cement_seal.sql` | 两张表（schema v25 → v26） |
| `config.py` / `defaults.toml` | `CementSealConfig`，默认 `enabled=false` |
| `tests/unit/test_cement_seal.py` | 17 个用例，每个钉一条上表性质 |

## 5. 尚未接线（刻意留白）

`RealtimeEmotionService._expression_dynamics` 的末尾调用 `apply()`、
`ProactiveContactPolicy.gate()` 合并 `cement_seal` 原因、以及从
`RelationshipState`/`EmotionFrame` 生成 `CementSealImpact` 的那层——**均未接入**。

理由与 ENGRAM A+B 一致：**在真实触发条件被校准之前，不改变生产行为。**
`enabled=false` 时整条链零行为。接线只需三处：

1. `_expression_dynamics` 返回前 `return self._cement.apply(dynamics, state)`
2. `ProactiveContactPolicy.gate()` 里 `return self._cement.gate(state) or ...`
3. 回合结束时用 `RelationshipState.repair_debt/tension` + distress recurrence 构造
   `CementSealImpact`，调 `evaluate()`，转移写账本

## 6. 校准前必须想清楚的一件事

默认参数（4 次未修复 → 封；单次 0.55 温度 → 崩）是**故意难封、易碎**的。
理由：一个数字生命如果太容易对用户封心，它就从"有脾气"滑向"难伺候"。
真实调参应该由 `notes/research/` 的行为评估来定，而不是凭手感——
这条链改的是关系体验，错误的代价由用户承担。
