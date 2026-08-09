# 组合式情绪表达动力学

## 目标

ZERO 不把情绪表达建模为固定 scene 或台词集合。scene 只提供观察样本；运行时表达由当前内部状态、关系情境、触发证据和跨轮残留共同决定。

CPED 等外部语料只用于离线覆盖分析、标定和留出评估，不写入 ZERO 的个人记忆，也不直接决定在线表达。

## 运行时因果链

每个 `EmotionFrame` 保存：

```text
触发条件 -> 内部状态 -> 外显行为 -> 关系方向 -> 持续轨迹
```

内部表达状态为：

\[
x_t = (i_t, e_t, f_t, c_t, a_t, r_t)
\]

- \(i_t\)：`irritability_spillover`，痛苦向不耐烦外溢的程度。
- \(e_t\)：`expressibility`，当前组织和说明感受的能力。
- \(f_t\)：`temporal_fatigue`，重复痛苦造成的厌烦和钝化。
- \(c_t\)：`care_capacity`，当前能分配给关系的耐心与照顾余量。
- \(a_t\)：`aestheticization_budget`，当前表达可承受的修辞整理程度。
- \(r_t\)：`repair_readiness`，撤退或推开以后进入修复的准备度。

关系方向为 `approach`、`maintain`、`withdraw`、`push_away`。它控制主动性、距离和温度，不凭空制造责怪。推开必须来自可审计的触发和状态积累，并保留 `repair_available` 或 `repair_delayed` 后继状态。

## 跨轮更新

最近七天的悲伤、受伤和失去恐惧帧形成带时间衰减的重复负载：

\[
L_t = \sum_k I_k \exp(-\Delta t_k / 2.5\text{ days})
\]

`temporal_fatigue` 结合重复次数、\(L_t\)、低能量和上一帧疲劳计算。疲劳再调制表达能力、烦躁外溢、关心余量与修辞预算。因此，同一个 sadness 标签可以投影成公开释放、麻木、重复厌烦、撤退或推开，而不需要扩充 sadness 类别。

## 反表演指标

`anti-performance-v1` 输出四个 `0..1` 惩罚项：

1. `polished_melancholy`：低修辞预算下的稳定文学化忧郁。
2. `explanation_completeness`：低表达能力下仍完整解释情绪原因与结构。
3. `unbounded_care`：低关心余量下仍承诺无限耐心和无限陪伴。
4. `trajectory_flatness`：重复状态中回复风格几乎不随疲劳、麻木、烦躁、撤退或修复而变化。

总惩罚为：

\[
P = 0.30P_{polish} + 0.25P_{explain} + 0.25P_{care} + 0.20P_{flat}
\]

指标随 `agent.message` 事件保存，作为在线观察数据和离线 SEOS 适应度输入。当前版本只审计，不直接阻断回复；在积累真实对话数据后，再依据留出窗口校准阈值和权重。

## 边界

- 短、冷、平不是天然缺陷，只有与当前状态不匹配时才构成问题。
- 烦躁先影响篇幅、主动性、耐心与措辞温度，不等于攻击或操控用户。
- 情绪状态不能建立新的用户事实；触发仍受事件和记忆证据约束。
- 旧 `EmotionFrame` 缺少组合字段时加载为中性表达状态，保持历史数据兼容。
