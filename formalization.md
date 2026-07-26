# Stateless Space Activation (SSA) — 形式化定义

> 版本: 0.1 draft
> 日期: 2026-07-25
> 状态: 起草中

---

## 1. 核心命题

传统 LLM agent 的记忆架构基于"状态持久化"——存在一个不断更新的内部状态向量 $S_t$，系统行为由 $S_t$ 决定。这种范式导致：
- 状态漂移（性格不可控地变化）
- 重置悖论（重置即"杀死"了 agent）
- 情绪表演化（情绪是生成文本的属性，而非涌现属性）

SSA 提出：**放弃状态持久化，将"过去"重组为共现的空间几何，让"此刻"通过激活函数从空间中涌现。**

形式化地，SSA 定义为一个三元组 $\langle \mathcal{S}, A, E \rangle$：

- $\mathcal{S}$：痕迹空间（Trace Space）
- $A$：激活函数（Activation Function）
- $E$：涌现函数（Emergence Function）

---

## 2. 形式化定义

### 2.1 痕迹空间 $\mathcal{S}$

$\mathcal{S}$ 是一个高维语义空间，由历史交互留下的痕迹（traces）填充。

**痕迹** $t_i \in \mathcal{S}$ 定义为五元组（v0.3 修订：增加了 $c_i$ 原始内容）：

$$t_i = (c_i, v_i, \tau_i, \rho_i, \eta_i)$$

其中：
- $c_i$：原始内容（文本）—— v0.2 遗漏此项，但 $E$ 需要原始内容
- $v_i \in \mathbb{R}^d$：语义向量，由嵌入模型 $\phi$ 从痕迹内容生成：$v_i = \phi(c_i)$
- $\tau_i \in \mathbb{R}^+$：时间戳，痕迹写入时刻
- $\rho_i(t) = e^{-\lambda(t - \tau_i)}$：新鲜度，**查询时计算的函数，不是存储值**
- $\eta_i \in (0, 1]$：重要性权重，由写入时的 LLM 自评定

> **v0.3 修正**：$\rho_i$ 是查询时间的函数，不是存储字段。因此"痕迹不可变"指 $(c_i, v_i, \tau_i, \eta_i)$ 固定，$\rho_i$ 是派生量，不与不可变性冲突。

**空间几何**：痕迹在 $\mathcal{S}$ 中的"位置"由 $v_i$ 决定。语义相近的痕迹自然聚类，形成"区域"。空间形状 = 痕迹分布密度。

**关键性质**：
- 痕迹只追加，不修改、不删除（append-only）
- 空间无时间轴，时间是痕迹的属性而非组织方式
- 空间可被部分索引（ANN 索引），但本质是连续拓扑

### 2.2 激活函数 $A$

$$A: (s, \mathcal{S}) \rightarrow \mathcal{T}_{\text{act}} \subseteq \mathcal{S}$$

输入：
- 信号 $s \in \mathbb{R}^d$（当前输入的语义向量）
- 空间 $\mathcal{S}$

输出：
- 被激活的痕迹集合 $\mathcal{T}_{\text{act}} = \{t_{i_1}, ..., t_{i_k}\}$

**激活机制**：

$$A(s, t_i) = \underbrace{\cos(s, v_i)}_{\text{语义相似度}} \cdot \underbrace{\rho_i(t)}_{\text{新鲜度}} \cdot \underbrace{\eta_i}_{\text{重要性}}$$

取 top-K 邻居作为激活集。

**辐射激活**（Radiation Activation，SSA 的扩展）：
被激活的痕迹不仅自身参与涌现，还会"辐射"激活其空间邻居，形成一片"亮区"。这模拟了联想记忆——说到"小王"时，跟小王相关的事都浮起来。

形式化：

$$\mathcal{T}_{\text{act}}^{(1)} = \text{topK}\big(A(s, t_i)\big)$$
$$\mathcal{T}_{\text{act}}^{(2)} = \bigcup_{t \in \mathcal{T}_{\text{act}}^{(1)}} \text{topK}'\big(A(v_t, t_i)\big)$$
$$\mathcal{T}_{\text{act}} = \mathcal{T}_{\text{act}}^{(1)} \cup \mathcal{T}_{\text{act}}^{(2)}$$

### 2.3 涌现函数 $E$

$$E: (s, \mathcal{T}_{\text{act}}) \rightarrow r$$

输入：
- 当前信号 $s$（原始形式，不仅是向量）
- 被激活的痕迹集合 $\mathcal{T}_{\text{act}}$（原始内容形式）

输出：
- 响应 $r$（文本、行为、消息）

**机制**：LLM 推理。$E$ 是纯函数，无内部状态：

$$r = \text{LLM}\Big(\text{prompt}(s, \text{content}(\mathcal{T}_{\text{act}}))\Big)$$

**关键性质**：
- 相同输入 → 相同输出（理论上是确定性的，但 LLM 的采样温度引入随机性，这是 feature 不是 bug——河流的湍流）
- 无状态 = 无"性格向量"、无"情绪向量"、无"对话历史指针"

### 2.4 写入操作 $W$

响应完成后：

$$W: (s, r, \tau_{\text{now}}) \rightarrow t_{\text{new}} \in \mathcal{S}$$

新痕迹 $t_{\text{new}}$ 被追加到空间。**注意：写入的是痕迹，不是状态。** 空间形状改变了，但没有"状态"被更新。

### 2.5 内心独白循环 $L$

后台定时循环，无需外部信号：

$$L: \emptyset \rightarrow (s_{\text{internal}}, r_{\text{internal}}, t_{\text{new}})$$

机制：
1. 生成内部信号 $s_{\text{internal}}$：随机选取空间中已有痕迹的向量 + 随机扰动
2. 走标准 SSA 流程：$r_{\text{internal}} = E(s_{\text{internal}}, A(s_{\text{internal}}, \mathcal{S}))$
3. 产物写入空间：$W(s_{\text{internal}}, r_{\text{internal}}, \tau_{\text{now}})$
4. 概率性触发主动消息：以概率 $p$ 将 $r_{\text{internal}}$ 发送给用户

**意义**：这是 SSA "活着"的机制。即使无外部输入，空间仍在被遍历、新痕迹仍在生成、空间形状仍在演化。

---

## 3. SSA 完整算法

```
Algorithm: Stateless Space Activation
─────────────────────────────────────
Input:  signal s (user message or internal trigger)
        space S (persistent trace space)
Output: response r

1. v_s ← φ(s)                      // 信号向量化
2. T_act ← A(v_s, S)               // 激活痕迹
3. T_act ← RadiationExpand(T_act)  // 辐射激活
4. r ← E(s, T_act)                 // 涌现响应
5. t_new ← (v_s, τ_now, ρ_now, η)  // 构造新痕迹
6. S ← S ∪ {t_new, (r content)}    // 写入空间
7. return r
```

---

## 4. SSA 的不变量（v0.3 修订：解决冲突）

> v0.2 的 4 个不变量彼此冲突。修订如下。

1. **推理核会话无状态**（非"严格无状态"）：$A$ 和 $E$ 不跨请求持有任何变量。$\mathcal{S}$ 是外部的、持久的——这是状态，不是"无状态"。准确表述：**会话状态推理核 + 外部持久化情景记忆**。

2. **痕迹内容不可变**：$(c_i, v_i, \tau_i, \eta_i)$ 写入后固定。$\rho_i(t)$ 是查询时派生的函数值，不存储，不与不可变性冲突。

3. **默认 append-only**：$\mathcal{S}$ 默认只增不减。遗忘策略可*删除*痕迹（非修改），但这是对单调增长的**有意违反**，需显式触发。不再声称"单调增长"是不变量——它是默认行为，可被遗忘策略覆盖。

4. **激活确定，涌现随机**：给定 $s$、$\mathcal{S}$ 快照、查询时间 $t$，$A$ 确定性。$E$ 因 $\xi$（LLM 采样）而随机。**不再声称"确定性涌现"**——涌现是随机的。

---

## 5. 状态-行为映射：SSA 与状态派的关系（v0.3 修订）

> **v0.2 声称 SSA"消除了确定性映射缺陷"，此主张已被撤回。** SSA 行为可写成 $E(s, A(s, \mathcal{S}), \xi)$，与状态派 $M(S_t, s_t, \epsilon)$ 结构等价。差异在状态表示的性质，不在映射结构。

### 5.1 状态派范式

$$\text{behavior}_t = M(S_t, s_t, \epsilon)$$

$S_t$：低维可变状态向量（10-100 维），in-place 更新。

### 5.2 SSA 范式

$$\text{behavior} = E\Big(s, A(s, \mathcal{S}, t), \xi\Big)$$

$\mathcal{S}$：高维 append-only 痕迹集合（512 维 × N 条），不 in-place 修改。

### 5.3 结构等价性

两者都是 $f(\text{state}, \text{input}, \text{noise})$ 形式。SSA 的 $\xi$（温度采样）与状态派的 $\epsilon$（外部噪声）在形式上没有区别。

### 5.4 实质差异（如有，待验证）

SSA 与状态派的差异不在映射结构，在**状态表示**：

| 维度 | 状态派 $S_t$ | SSA $\mathcal{S}$ |
|------|-------------|-------------------|
| 维度 | 10-100 | 512 × N |
| 更新方式 | in-place mutation | append-only |
| 组织 | 时间线/分层 | 语义空间 |
| 可变性 | 每次更新 | 只增不改 |

**这种差异是否产生可测量的行为差异，是经验问题，未经验证。**

### 5.5 撤回的证明

v0.2 §3.4 声称 $A(s, \mathcal{S}_{t_1}) \neq A(s, \mathcal{S}_{t_2})$ 当 $t_1 \neq t_2$。**此证明有反例**：新写入的痕迹若激活分低于 top-K 阈值，不进入激活集，$A$ 不变。此外，新鲜度 $\rho_i(t) = e^{-\lambda(t - \tau_i)}$ 对所有候选共享因子 $e^{-\lambda t}$，固定空间快照下不改变排序。

**结论：SSA 不保证"相同输入永不重现"。** 可变性来自 $\xi$ + $\mathcal{S}$ 增长（概率性影响 top-K），不来自形式上的证明。

---

## 6. 跟现有范式的对比（v0.3 修订：统一 reset 定义）

| 范式 | 持久化什么 | 组织方式 | 变更方式 | 清除会话状态 | 清除全部记忆 |
|------|-----------|----------|----------|-------------|-------------|
| Fine-tuning | 模型权重 | — | in-place 权重更新 | 无影响（权重持久） | 需重新训练 |
| RAG | 无（每请求） | 外部索引 | N/A | 无影响 | 丢失知识 |
| MemGPT | 工作记忆 + 外部 | 分层 | in-place（工作记忆）| 工作记忆清；外部保留 | 全丢 |
| Generative Agents | 反思+规划+流 | 时间线+反思节点 | in-place（反思/规划）+ append（流）| 反思/规划清；流保留 | 全丢 |
| Replika/Character.ai | 情绪向量+历史 | 时间线 | in-place（情绪）+ append（历史）| 情绪重置；历史保留 | 全丢 |
| **SSA** | $\mathcal{S}$（外部）| 语义聚类 | append-only | **无影响（核心本就会话无状态）** | 丢 $\mathcal{S}$ |

**修正说明**：v0.2 的比较表对"reset"定义不统一。SSA 在"清除会话状态"时有优势（核心本就会话无状态），但在"清除全部记忆"时与其他系统一样会"死"。Fine-tuning 实际上在 reset 韧性上更强（权重持久）。SSA 的 reset 优势是**窄范围的**。

---

## 7. 待形式化的问题

- [x] 状态-行为确定性映射缺陷 → 已在 §5 形式化
- [x] 跟现有范式的对比 → 已在 §6 完成，含确定性映射缺陷列
- [ ] 辐射激活的深度控制（如何避免激活爆炸？衰减函数？最大深度？）
- [ ] 重要性 $\eta$ 的自评定机制（LLM as judge 的可靠性？校准方法？）
- [ ] 空间拓扑的度量（如何量化"空间形状"？密度图？聚类系数？）
- [ ] 遗忘策略（何时删除痕迹？基于密度？基于重要性衰减？基于访问频率？）
- [ ] 涌现的"情绪"如何检测和量化？（LLM as emotion classifier？人工标注？）
- [ ] 评估指标设计（"像活的"怎么测？连续性得分公式？）
- [ ] 辐射激活的收敛性证明（多深会饱和？是否存在不动点？）
- [ ] 空间形状的稳定性分析（在什么条件下空间形状收敛？）
- [ ] 多用户扩展（$\mathcal{S}$ 是否每用户独立？共享空间如何处理？）
- [ ] 遗忘的形式化（遗忘是删除痕迹还是降低 $\eta$？两种方案的 trade-off？）

---

## 参考文献（待补全）

[1] Park, J.S., et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023.
[2] Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." arXiv 2024.
[3] ...
