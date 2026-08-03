# 超维度空间计算（HDSC）— 形式化重构入口

> 版本: 0.5 draft
> 日期: 2026-07-25
> 状态: legacy SSA-A0 已冻结；HDSC 形式化重构中

> 本文第 1 节之后保存旧 SSA 的形式定义，用于复现实验与反例分析，
> 不代表 HDSC 新内核。新的跨学科主张等级、热力学约束和晋级门禁见
> [`notes/research/2026-07-26_hdsc_validation_framework.md`](notes/research/2026-07-26_hdsc_validation_framework.md)。

---

## 1. HDSC 当前核心命题

超维度空间计算（Hyperdimensional Space Computing, HDSC）研究持久化高维
痕迹几何上的计算。它不是对“数字生命”或物理热力学的命名包装：软件激活
质量、信息熵、行为状态和物理能量是四种不同量，不得互相代换。

HDSC 的首个可计算参考内核是 `hdsc-h1-c1-shadow`。它只处理可逆对称图，把语义检索得到的源预算
$q$ 放入带非负对称电导 $K$ 的图上，并使用组合一致的扩散和环境泄漏：

$$
D_{ii}=\sum_j K_{ij},\qquad L=D-K,
$$
$$
u_{\mathrm{diffused}}(t)=e^{-tL}q,\qquad
u_{\mathrm{final}}(t)=e^{-\kappa t}u_{\mathrm{diffused}}(t).
$$

这里 $u$ 是无量纲的计算激活质量，$t$ 是形式扩散时长，$\kappa$ 是环境
泄漏率。由 $K=K^T\ge0$ 可得 $L=L^T$、$L\mathbf 1=0$；参考实现通过谱分解
计算小图精确解，并记录非负性、质量守恒残差、互易残差、Dirichlet 量和
Shannon 熵诊断。内部扩散不产生或销毁质量，只有显式环境项产生账本中的耗散：

$$
\sum u_{\mathrm{diffused}}=Q_0,\quad
\sum u_{\mathrm{final}}=e^{-\kappa t}Q_0,\quad
Q_{\mathrm{dissipated}}=(1-e^{-\kappa t})Q_0.
$$

这使 HDSC-H1 具备 C1“热力学结构类比”，但不构成 C2 物理热力学模型；C2
必须增加硬件功耗遥测、焦耳标定、测量不确定度和重复实验。

旧 Stateless Space Activation（SSA）保留为 `legacy-ssa-a0` 回放基线。它的
一跳 radiation 不得使用 HDSC-H1 的守恒或热力学措辞。HDSC 定义为一组带版本
的空间、传输和生成算子，而不是把旧公式换一个名字。

v0.4 将真实拓扑拆为互易分量与有向剩余分量。`hdsc-h1d-directed-markov-shadow`
使用非可逆 Markov 生成元处理完整有向图，H1 仅保留为可逆对称参考。当前活动态
理论仍为 `hdsc-h2-bounded-active-shadow`；H1D 与 H2 都是 shadow，不参与生成
prompt。

---

## 2. Legacy SSA-A0 形式化定义

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

## 8. HDSC-H1 C1 Shadow 内核与晋级边界

### 8.1 源预算

给定查询与每个痕迹的相似度 $s_i$、新鲜度 $f_i$ 和重要性 $\eta_i$，先在
完整集合上计算：

$$
g_i=\big[\max(s_i-\theta,0)\big]^p f_i^\alpha\eta_i^\beta.
$$

取全局前 $m$ 个正权痕迹，固定注入预算 $Q_0$：

$$
q_i=Q_0\frac{g_i}{\sum_{j\in M}g_j},\qquad \sum_iq_i=Q_0.
$$

如果没有正证据，参考实现显式报出“无源证据”，不偷偷制造激活质量。
重复节点只能重新分配这一个预算，不得因为图规模增长而放大总量。

### 8.2 形式不变量

对任意有限无向图，参考内核要求：

1. $K_{ij}\ge0$ 且 $K=K^T$（被动互易传导）。
2. $L=D-K$ 对称半正定，$L\mathbf1=0$。
3. $e^{-tL}$ 非负且双随机，故闭系统质量守恒，Shannon 熵不下降。
4. $V(u)=\frac12u^TLu$ 在纯扩散下不增加。
5. 断连分量之间没有质量交换；节点置换不改变物理结果。
6. 环境泄漏单独计入 $Q_{\mathrm{dissipated}}$，满足一阶账本。

这些是计算级和结构级测试，不是把 $u$ 宣称为焦耳。代码位于
`src/ssa/hdsc/transport.py`，性质测试位于
`tests/property/test_hdsc_h1_transport.py`；模型标识和声明等级必须随审计
结果一起保存。

### 8.3 晋级流程

HDSC-H1 只能按以下顺序进入主提示链路：形式不变量测试、历史回放、shadow
模式、多基线盲测、跨学科评审、显式配置晋级。当前运行时仍使用
`legacy-ssa-a0`，因此 H1 的结果不会改变回复。

### 8.4 宏观闭合与微观不可辨识性

令完整微观状态为 $X_t\in\Omega$，宏观激活质量只是一个多对一投影：

$$u_t=C(X_t).$$

HDSC-H1 当前证明的是给定固定 $q$、$K$、$t$ 时宏观算子
$e^{-\kappa t}e^{-tL}$ 的性质。它没有给出微观轨迹证明。要让宏观过程闭合，
任意满足 $C(x)=C(x')$ 的两个微观状态都必须满足：

$$
\mathbb E[C(X_{t+\Delta})\mid X_t=x]
=\mathbb E[C(X_{t+\Delta})\mid X_t=x'].
$$

如果该 lumpability 条件不成立，宏观变量 $u_t$ 不是充分状态，演化式需要
记忆核与未解析扰动：

$$
\dot u(t)=-(L+\kappa I)u(t)
+\int_0^t M(t-s)u(s)\,ds+\eta(t).
$$

因此，$u_i>0$ 或微扰 $\delta>0$ 本身不是不可预测性的充分条件。固定图上的
线性 H1 算子满足谱指数 $-(\lambda_i+\kappa)\le0$，宏观差异是收缩的；微观
不可预期性需要通过正 Lyapunov 指数、随机转移核或粗粒化非辨识性来证明。
当前审计必须标记 `macro-coarse-grained` 与 `microstate-unmodeled`。

### 8.5 追加档案与闭环敏感性

必须把不可变档案与真正驱动行为的活动态分开。令：

$$
e_t=W(s_t,r_t),\qquad
\mathcal H_{t+1}=\mathcal H_t\oplus e_t,
$$

其中 $\mathcal H_t$ 是审计档案。若每轮恰好追加一个 episode，则：

$$
N_t=|\mathcal H_t|=N_0+t,\qquad
\Delta N_t=1,\qquad \Delta^2N_t=0.
$$

因此单调成立的是痕迹基数，不是拓扑距离、响应幅度或各阶导数。连通分量、
谱隙、聚类密度和 top-K 集合都可能非单调变化。

人机系统的完整闭环应写为：

$$
s_{t+1}=U(r_t,h_t,\nu_t),
$$
$$
z_{t+1}=\Pi_{\Delta_Q^B}\left[(1-\alpha)e^{-\tau L_t}z_t
+\alpha B(e_t)\right],
$$
$$
r_t\sim P_\theta\!\left(\cdot\mid s_t,R(s_t,z_t),\xi_t\right).
$$

$z_t$ 是最多 $B$ 个语义簇、总质量固定为 $Q$ 的活动态；$\Pi$ 是到该有界
单纯形的投影。原始档案继续追加，但不直接等同于行为状态。$B(e_t)$ 必须按
语义簇分配预算，同簇重复痕迹共享一个簇预算，防止重复写入放大影响。

由于文本 $r_t$ 是离散随机变量，$\partial r_t/\partial r_{t-1}$ 只是一种
直觉记号。正式测量使用响应分布的干预距离：

$$
I_{t-1\rightarrow t}=D\!\left(
P(r_t\mid\operatorname{do}(e_{t-1}=a)),
P(r_t\mid\operatorname{do}(e_{t-1}=a'))
\right),
$$

或使用响应嵌入期望 $y_t=\mathbb E[\phi(r_t)]$ 的 Jacobian。它至少包含两条
路径：$r_{t-1}\rightarrow s_t\rightarrow r_t$ 的对话反馈，以及
$r_{t-1}\rightarrow e_{t-1}\rightarrow(q_t,L_t)\rightarrow r_t$ 的空间反馈。

更严格地，令 $F(s,m,\xi)=E(s,A(s,m),\xi)$。在固定 $s_t,m_t,\xi_t$ 的偏导
语义下，原式没有显式的 $r_{t-1}$，因此：

$$
\left.\frac{\partial r_t}{\partial r_{t-1}}\right|_{s_t,m_t,\xi_t}=0.
$$

历史影响应写成沿中介路径的总导数。对可微代理变量，有：

$$
\frac{d r_t}{d r_{t-1}}
=(E_s+E_aA_s)U_r+E_aA_mG_r
+E_\xi\frac{d\xi_t}{dr_{t-1}},
$$

其中 $U_r$ 是用户反馈路径，$G_r$ 是记忆写入与活动态更新路径。独立采样时
最后一项为零；共享随机数实验则用它分离随机方差和初值敏感性。多步影响由
联合状态 Jacobian 的乘积决定，而不是由某一个单步导数决定。

对于图扰动 $\Delta L$，矩阵指数的 Frechet 导数为：

$$
D e^{-\tau L}[\Delta L]
=-\tau\int_0^1e^{-(1-a)\tau L}\Delta L e^{-a\tau L}\,da.
$$

当 $L\succeq0$ 时：

$$
\left\|D e^{-\tau L}[\Delta L]\right\|_2
\le \tau\|\Delta L\|_2.
$$

所以扰动控制依赖 $\alpha$、$\tau$、活动容量 $B$、单轮图变化上限
$\|\Delta L_t\|_2\le\varepsilon_L$ 和生成分布的经验增益。`top-K` 在排序并列
边界上不连续，那里不使用形式导数，而使用配对干预、ensemble replay 与
Wasserstein/total-variation 距离。

最终稳定性目标不是令历史影响为零，而是把记忆回路限制为有界增益：

$$
\|\Delta z_{t+1}\|
\le\rho\|\Delta z_t\|+\alpha L_B\|\Delta e_t\|
+\tau Q\varepsilon_L,
\qquad 0\le\rho<1.
$$

把响应、用户反馈和活动态更新的局部增益分别记为：

$$
d(r_t,r_t')\le a\,d(s_t,s_t')+b\,d(z_t,z_t'),
$$

以及用户增益 $u$、活动态直接增益 $c,d$、响应回写增益 $e$，则联合扰动满足：

$$
\begin{bmatrix}
d(s_{t+1},s_{t+1}')\\
d(z_{t+1},z_{t+1}')
\end{bmatrix}
\le
\underbrace{\begin{bmatrix}
ua & ub\\
c+ea & d+eb
\end{bmatrix}}_{M_t}
\begin{bmatrix}
d(s_t,s_t')\\
d(z_t,z_t')
\end{bmatrix}.
$$

在时不变上界下，小增益门禁为 $\rho(M)<1$。用户反馈 $U$ 是未知且时变的，
因此系统给出的是带用户增益假设的条件稳定性，并通过共享随机数的配对轨迹
持续估计实际增益。非零影响只表示存在因果路径；持续蝴蝶效应需要联合
Jacobian 乘积的最大 Lyapunov 指数为正。

对于离散随机文本，定义分布闭环增益：

$$
\eta_t=\sup_{x\ne x'}
\frac{W_d(P_t(\cdot\mid x),P_t(\cdot\mid x'))}{d(x,x')},
\qquad
\lambda_W=\limsup_T\frac1T\sum_{t<T}\log\eta_t.
$$

$\lambda_W<0$ 表示分布收缩，$\lambda_W>0$ 才支持持续扰动放大的判断。

情景活动态采用收缩更新；身份锚点采用多源证据、迟滞阈值和版本化更新。
“记忆可追加”与“行为抗漂移”必须分别证明，前者不会自动推出后者。

---

## 9. HDSC-H2 有界活动空间

### 9.1 双层状态

H2 将系统拆成增长型审计档案与固定容量活动测度：

$$
\mathcal H_{t+1}=\mathcal H_t\oplus e_t,
$$

$$
\mu_t=w_{\bot,t}\delta_\bot+
\sum_{c\in C_t}w_{c,t}\delta_{z_c},\qquad
|C_t|\le C,\quad w_{\bot,t}+\sum_cw_{c,t}=1.
$$

$\mathcal H_t$ 保存可重放历史；$\mu_t$ 才是 H2 的行为活动态。archive count
和 replay step 不进入行为 `state_hash`；包含步号的独立 `replay_hash` 用于审计
顺序。$\bot$ 是显式空槽，用于吸收自然衰减和容量溢出，避免删除低质量簇后
重新归一化放大剩余簇。

### 9.2 固定语义分区与重复抑制

当前工程使用与 archive 数据量无关的固定 16 位超平面码本，把单位 embedding
映射到稳定 bucket。每个 bucket 的原型由码本直接给出，不使用随历史增长而
漂移的在线均值。embedding model、维度、位数、码本版本与超平面内容共同形成
`semantic_partition_id`，并进入 config hash。候选按 `(cluster_id, revision)`
合并：

$$
g_c=\max_{j:c_j=c}\left(\operatorname{relevance}_j
\operatorname{novelty}_j\right).
$$

使用 `max` 而不是 `sum`，故同簇候选复制 $n$ 次不增加簇预算。top-K 正分候选
形成提议分布 $q_t$；没有正提议时 $q_t=\delta_\bot$。精确 fingerprint 只用于
同一候选批次内的冲突检查。跨轮的新颖度由“该版本化 bucket 当前是否仍在活动
态”确定：活动簇内的新候选不追加质量，已被容量投影淘汰的簇以后可以重新进入。
因此 H2 不维护随 archive 增长的永久 fingerprint 集合。

### 9.3 有界更新

H2 v0.3 先采用恒等非扩张传输，隔离活动态问题与 H1 图传播问题：

$$
\widehat\mu_{t+1}
=\beta\mu_t+\alpha q_t+(1-\beta-\alpha)\delta_\bot,
$$

$$
\mu_{t+1}=\Pi_C(\widehat\mu_{t+1}),\qquad
0\le\beta<1,\quad 0\le\alpha,\quad \beta+\alpha\le1.
$$

$\Pi_C$ 只保留排序最高的 $C$ 个非空簇，并把全部溢出质量移入 $\bot$，不对
保留项二次放大。incumbent hysteresis 只参与支持集排序，不修改实际质量。

由此得到以下 C0-M 性质：

1. $|C_t|\le C$，与 archive 长度无关。
2. 所有质量非负、有限且总和恒为 1。
3. 同轮同簇复制不改变下一活动态。
4. 重复候选的零新颖度更新与空候选更新相同。
5. 固定支持、相同提议时，总变差距离满足
   $D_{TV}(\mu_{t+1},\mu'_{t+1})\le\beta D_{TV}(\mu_t,\mu'_t)$。

### 9.4 局部证书与主动弃权

H2 的局部证书同时检查三种离散边界。固定 SimHash 码本的分簇间隔为：

$$
\Delta_z=\min_{j,\ell}|h_\ell^\top \bar v_j|,
$$

即候选单位 embedding 到最近超平面的距离。硬 top-K 的提议边界差为：

$$
\Delta_q=s_{(K)}-s_{(K+1)}.
$$

当正分候选不足 $K$ 时还必须检查从 0 分进入支持的边界，不以常数哨兵替代。
若分数映射的局部 Lipschitz 上界为 $L_s$，提议支持半径为：

$$
\varepsilon_q=\frac{\Delta_q}{2L_s}.
$$

令选中提议的正分总和为 $S_q=\sum_{c\in Q}g_c$，支持大小为 $m$。固定支持上，
$q_c=g_c/S_q$ 对原始分数源度量的保守总变差增益为：

$$
L_{q,g}=\frac{mL_s}{S_q}.
$$

该项显式处理“小分母放大”：$S_q$ 很小时，即使 top-K 名次不变，归一化后的
质量也可能高度敏感。容量选择使用与实现完全相同的优先级
$p_c=\widehat w_c+h\mathbf 1[c\in C_t]$，而不是原始质量。令
$\Delta_C=p_{(C)}-p_{(C+1)}$；没有外部候选时用最小保留质量作为到零支持的
间隔。若先验状态 TV 与原始分数源距离都由同一个 max-product 半径
$\varepsilon$ 控制，则投影前质量扰动不超过
$(\beta+\alpha L_{q,g})\varepsilon$，故容量支持半径取：

$$
\varepsilon_C=
\frac{\Delta_C}{2(\beta+\alpha L_{q,g})}.
$$

`certified_radius` 只属于如下明确的 max-product 支持度量：三个坐标分别是单位
embedding 的 $L_2$ 距离、原始分数源距离和先验活动态的总变差。因而其联合
半径为：

$$
\varepsilon_{support}=\min\left(
\Delta_z,\frac{\Delta_q}{2L_s},
\frac{\Delta_C}{2(\beta+\alpha L_{q,g})}
\right).
$$

它不是原始用户文本空间中的半径；把它换算到其他输入度量必须另外给出分量增益。
任一间隔缺失、为零或出现并列，用户增益或其测量来源缺失，或增益非有限时，
证书状态为 `abstain`。在全部闭环增益齐备时，使用 §8.5 的联合矩阵 $M$；只有
$\rho(M)<1-\varepsilon_{gain}$ 才产生 `conditional-pass`。该结论不覆盖新候选
离散出现、SimHash 换桶或容量支持切换后的全局行为。

### 9.5 工程状态

- 纯内核：`src/ssa/hdsc/active_space.py`。
- 性质测试：`tests/property/test_hdsc_h2_active_space.py`。
- episode 写入后推进，启动时按 archive 时间顺序重放；新颖度只由 archive 中
  可恢复的 bucket 与当前活动态导出，不依赖未归档的 appraisal 临时值。
- `state.step == archive_count` 才允许显示 `passed`；缺向量、分区不匹配或读取
  异常只形成 shadow failure，不中断 serving。
- TUI 分别显示 serving engine、shadow engine、archive/view 数量、active mass、
  local gate 与 closed-loop gate。
- H2 状态只保存在内存并由 archive 重放恢复；尚未写入 legacy activation 表。
- serving prompt、legacy 检索结果和 LLM 请求在 H2 开关前后保持一致。

H1 与 H2 的组合需在固定活动支持上单独验证后形成后续 H3；v0.3 不把两项局部
证明拼接成全局稳定性声明。

---

## 10. HDSC-H1D 非可逆有向传导

### 10.1 拒绝伪对称化

真实痕迹拓扑是多关系图。语义相似度可形成互易边，但时间顺序、证据支持、
修订和反馈关系具有方向。令 $R_{ij}\ge0$ 表示从节点 $j$ 流向节点 $i$ 的速率，
即工程约定 `rates[target, source]`。将有向矩阵替换为
$(R+R^T)/2$ 会同时产生两个错误：为单向边凭空制造反向通道，并消除原系统的
净概率流。

只允许把拓扑分解为诊断量：

$$
R^{\mathrm{rev}}_{ij}=\min(R_{ij},R_{ji}),\qquad
R^{\mathrm{dir}}=R-R^{\mathrm{rev}}.
$$

HDSC-H1 只适用于 $R^{\mathrm{rev}}$ 这类对称可逆参考。完整拓扑必须交给
`hdsc-h1d-directed-markov-shadow`，不得先对称化再复用 H1 结论。

### 10.2 有向 Markov 生成元

H1D 使用连续时间主方程。生成元 $Q$ 定义为：

$$
Q_{ij}=R_{ij}\quad(i\ne j),\qquad
Q_{jj}=-\sum_{i\ne j}R_{ij}.
$$

因此 $Q$ 的非对角元素非负，且列和为零：

$$
\mathbf1^TQ=0.
$$

对列向量活动质量 $u$，有向传导和显式环境泄漏为：

$$
u_{\mathrm{transported}}(t)=e^{tQ}u_0,\qquad
u_{\mathrm{final}}(t)=e^{-\kappa t}e^{tQ}u_0.
$$

$P_t=e^{tQ}$ 是列随机 Markov 半群。由此得到不依赖对称性的 C0-M 性质：

1. $P_t\ge0$，非负源不会产生负质量。
2. $\mathbf1^TP_t=\mathbf1^T$，闭系统总质量守恒。
3. $P_{t+s}=P_tP_s$，连续时间步可组合。
4. 对任意等质量分布 $p,q$，
   $\|P_tp-P_tq\|_1\le\|p-q\|_1$。
5. 环境泄漏仍单独满足
   $Q_{in}=Q_{out}+Q_{dissipated}$。

这些性质来自 Metzler/Markov 结构，不要求 $Q=Q^T$，也不要求均匀稳态。

### 10.3 非平衡信息与热力学门禁

对非对称图，H1 的普通 Dirichlet 单调性、双随机性、Shannon 熵不下降和均匀
稳态结论全部停止使用。若提供严格正的稳态分布 $\pi$，满足 $Q\pi=0$，则可用
Markov 数据处理不等式：

$$
D_{KL}(P_tp\|\pi)\le D_{KL}(p\|\pi).
$$

令稳态通量 $F_{ij}=R_{ij}\pi_j$。若每条正向边都有正的反向边，有限的结构级
熵产生率可写为：

$$
\sigma=\sum_{i<j}(F_{ij}-F_{ji})
\log\frac{F_{ij}}{F_{ji}}\ge0.
$$

若存在 $R_{ij}>0$ 且 $R_{ji}=0$，标准有限熵产生需要额外的开放环境或隐藏反应
通道。H1D 此时对 `finite_entropy_production_gate` 输出 `abstain`，而不是把单向
软件边包装成被动热传导。即使结构门禁通过，焦耳、温度和物理熵仍需局部详细
平衡、储库参数和硬件标定；当前 `physical_thermodynamic_status` 固定为
`not-calibrated`。

### 10.4 工程状态

- 内核：`src/ssa/hdsc/directed_transport.py`。
- 性质测试：`tests/property/test_hdsc_h1d_directed_transport.py`。
- `semantic` 关系仍作为互易分量写入；`temporal-forward` 只按旧痕迹到新痕迹
  的方向写入，不声称因果。
- H1D 离线矩阵读取全部关系；legacy radiation 只读取 `semantic`，所以新增有向
  关系不改变 serving prompt。
- SPACE 图用 `:` 表示互易语义边，用方向箭头表示非语义有向边。
- H1D 与 H2 的组合、闭环增益和 prompt-path 晋级仍属于后续 H3。

---

## 11. P1 环境与事态感知算子

P1 模型标识为 `hdsc-p1-environment-perception`，算子版本为
`semantic-affect-context-time-v1`。它是当前 serving 控制层，不替换
`legacy-ssa-a0` 检索，也不等同于 YUANZI-5D 主动推断。

### 11.1 数字时间到人类时钟

数据库继续只保存 UTC epoch milliseconds。对配置时区 $Z$，时间感知算子为：

$$
\mathcal T_Z(t,t_e,t_u)
=\left(
t_{UTC},t_Z,d_Z,h_Z,q_Z,
\sin\frac{2\pi h_Z}{24},
\cos\frac{2\pi h_Z}{24},
t-t_e,t-t_u,g_t
\right),
$$

其中 $d_Z$ 是星期，$h_Z$ 是带分钟和秒的小数小时，$q_Z$ 是 quiet-hours
指示量，$t_e,t_u$ 分别是上一事件和上一用户事件时间，$g_t$ 是
`first_contact / continuous / resumed / long_gap` 离散间隔。时区使用 IANA
`zoneinfo`，因此 UTC 存储与本地人类语境保持分离。

### 11.2 四类输入状态

语义状态为

$$
s_t=(s_q,s_d,s_{self},s_{bio},s_{rel},s_{time},s_{gap})\in[0,1]^7,
$$

分别表示提问、行动请求、自我披露、自传指涉、关系指涉、时间指涉和信息缺口。
情绪状态直接来自已验证的 appraisal：

$$
e_t=(v,e_+,e_-,a,u,1-c,i),
$$

其中 $v\in[-1,1]$ 是 valence，$a,u,c,i$ 分别是 arousal、urgency、
certainty 和综合强度。语境状态为

$$
c_t=(r_s,r_c,m_d,f_c,\Delta_{topic},c_h,k,tension,u_r,n_{turn}),
$$

包含痕迹支持、语义连续性、记忆密度、对话新鲜度、主题切换、语境一致性、
关系亲密度、张力、未解决压力和近期回合数。所有派生值不升级为用户事实。

### 11.3 交叉算子

当前确定性实现显式计算五个跨域乘积：

$$
\chi_t=\left(
s_{self}i,
s_{bio}r_s,
e_-\max(tension,u_r),
g_{pressure}(1-f_c),
s_d\,u\,controllability
\right).
$$

它们依次表示语义-情绪、语义-记忆语境、情绪-关系语境、时间-连续性和
行动准备度。令 $z_t$ 为基于 $(s_t,e_t,c_t,\mathcal T_t,\chi_t)$ 的版本化
固定权重 logits，则八类事态分布为：

$$
p_t(m)=\operatorname{softmax}\left(\frac{z_t(m)}{\tau_P}\right),
$$

$$
m\in\{answer,act,clarify,support,repair,reminisce,explore,acknowledge\}.
$$

主事态和响应姿态为：

$$
m_t^*=\arg\max_m p_t(m),
\qquad
h_t=H\left(p_t,e_t,c_t,\mathcal T_t\right)\in[0,1]^5,
$$

其中 $h_t$ 的五个分量是 warmth、directness、exploration、caution 和
temporal sensitivity。$h_t$ 只作为 soft control 进入私有 prompt；响应仍必须
遵守来源与不确定性约束。

### 11.4 执行与审计合同

每轮顺序固定为：

```text
append user event
-> legacy trace activation
-> structured appraisal
-> P1 time/semantic/affect/context cross operator
-> immutable perception snapshot
-> reconstruct private prompt from stored snapshot
-> main response
-> state/relationship update and episode write
```

感知记录存入迁移 011 的 `perception_snapshots`，包含 query event、conversation、
correlation、模型 ID、算子版本、完整输入状态、交叉项、归一化模式分布和响应
姿态。后续 prompt 读取该快照而不是按当前时钟重算历史，因此缓存前缀和因果重放
保持一致。性质测试要求时区映射、跨午夜 quiet hours、间隔分类、模式归一化、
确定性、数据库往返和历史 prompt 重建全部通过。

---

## 12. 语境等待与内在心跳算子

这一层不把“进程仍在运行”等同于“用户仍在电脑前”，也不把沉默直接解释为
拒绝、危险或离开。它只维护一个可重放的结构化潜状态，不保存或生成自由文本
形式的私密思维链。当前算子版本记为 `contextual-inner-loop-v1`。

### 12.1 回复期待与动态等待窗口

在 agent 完成第 $t$ 次回复后，从已持久化的事态快照、关系状态和双方文本中构造
特征：

$$
\phi_t=(q_a,support,affect,relation,uncertainty,pause,closure,tension),
$$

其中 $q_a$ 表示 agent 是否留下明确问题，`pause` 表示“等一下、晚点回来”等
暂停承诺，`closure` 表示“先这样、晚安、再见”等结束线索。初始回复期待为：

$$
Q_t^0=\sigma(w_Q^T\phi_t+b_Q).
$$

等待时间不是固定常数，而是：

$$
T_t=\operatorname{clip}\left(
T_0\exp(w_T^T\phi_t+\gamma Q_t^0),
T_{min},T_{max}
\right).
$$

因此，“我晚点回来继续”与 agent 留下的问题会延长等待；明确结束语会缩短等待。
配置仍给出 $T_{min}$、$T_0$、$T_{max}$，防止语义噪声把截止时间放大到无界。

### 12.2 黑盒潜状态递推

令内部控制向量为：

$$
x_t=(Q_t,C_t,K_t,L_t,U_t,R_t)\in[0,1]^6,
$$

分别表示回复期待、担忧、好奇、连接压力、不确定性和离线准备度。心跳间隔为
$\Delta t$，每个有半衰期的分量使用：

$$
D_i(\Delta t)=2^{-\Delta t/h_i},
$$

$$
x_{t+1}^{(i)}=
\operatorname{clip}_{[0,1]}\left[
D_i x_t^{(i)}+(1-D_i)g_i(\phi_t,\tau_t,o_t,r_t)
\right].
$$

这里 $\tau_t$ 是等待进度，$o_t=\max(0,t-t_{deadline})$ 是逾期时间，
$r_t$ 是当前 organism/relationship 状态。确定性 serving 版本取 $\xi_t=0$；若以后
加入扰动，只能使用有界、可记录、可重放的 $K\xi_t$，不得把随机漂移写成用户
事实。

等待截止后的回复期待按半衰期衰减：

$$
Q_{t+1}=Q_t2^{-\Delta o/h_Q}.
$$

无回复带来的“意外量”与逾期 hazard 为：

$$
S_t=Q_t(1-e^{-2\tau_t}),\qquad
H_t=1-e^{-o_t/T_t}.
$$

担忧的目标值只由不确定性、关系张力、连接压力和 $S_t$ 组合；它不是对用户动机
的断言。离线准备度为：

$$
R_t=\sigma(
b_R+1.9H_t+1.2(1-Q_t)+0.55E_t+0.35A_t-1.25C_t
).
$$

### 12.3 模式门控与滞回

模式集合为：

$$
M_t\in\{engaged,waiting,offline,quiet\_rest\}.
$$

转移顺序是：

1. 新用户事件无条件进入 `engaged`，清除等待窗口。
2. agent 回复后进入 `waiting`，建立 $Q_t^0$ 和动态 deadline。
3. 能量低于下限进入 `quiet_rest`。
4. 仅当 $t\ge t_{deadline}$ 且 $R_t\ge\theta_{on}$ 时进入 `offline`。
5. 已在 `offline` 时使用较低的 $\theta_{off}<\theta_{on}$ 保持状态，避免阈值附近
   高频抖动。

`agency.offline_cycle`、`initiative.evaluate` 和 `contact.evaluate` 都读取这一模式门，
不再各自猜测用户是否离开。沉默的多种解释仍由 contact hypothesis 分布维护，
担忧值不会直接触发无限消息；quiet hours、cooldown、daily limit 和 contact
message bound 继续作为下游硬门禁。

### 12.4 后台心跳合同

`inner.heartbeat` 默认每 60 秒运行并 upsert `inner_loop_states` 的单行当前状态，
因此不会每天制造 1440 条快照。只有模式发生变化时才追加
`lifecycle.inner_transition` 事件。`last_heartbeat_at_ms` 是“进程当时存活”的证据，
不是“用户当时在场”的证据。TUI 与 worker 都停止后没有心跳、潜状态递推或离线
行动；重启只观察实际时钟间隔，不回填停机期间不存在的心理活动。

---

## 13. 持续自我反思与学习巩固架构

当前实现把一次离线 artifact 视为 Generator 输出，把后续
`reflection.schedule` 视为独立 Critic 和学习提案入口。这样避免同一条痕迹再次调用
工具或重复生成外部观察，同时让“反思文本”和“长期改变”拥有不同的审计边界。

### 13.1 四层循环

```text
L0 inner heartbeat
  -> engaged / waiting / offline / quiet_rest
L1 reflection schedule
  -> artifact evidence bundle / priority / critic
L2 consolidation
  -> memory candidate / belief revision / goal adjustment / policy proposal
L3 outcome evaluation
  -> user-observed outcome / confirm / revise / rollback
```

反思优先级为：

$$
P_i=\operatorname{clip}\left(
b+0.22S_i+0.18U_i+0.20G_i+0.16N_i+0.16R_i-0.08C_i
\right),
$$

其中 $S_i$ 为意外度，$U_i$ 为未解决压力，$G_i$ 为目标相关性，$N_i$ 为新颖度，
$R_i$ 为关系显著性，$C_i$ 为能量和重复成本。只有 $P_i\ge\theta_P$ 的
completed offline artifact 才会创建 `reflection_run`。

### 13.2 独立证据批判

对每个 run，批判器不重新生成用户事实，而计算：

$$
K_i=0.30\,coverage_i+0.30\,trust_i
    +0.25\,consistency_i+0.15\,falsifiability_i.
$$

`coverage` 检查事件、痕迹或 artifact 是否存在，`trust` 由代码根据
`SourceKind` 赋值，`consistency` 检查 episode 完成状态和错误，
`falsifiability` 检查反思是否留下可检验问题。$K_i$ 不是模型自报置信度，不能由
模型直接覆盖。

### 13.3 提案与巩固门禁

提案集合为：

$$
Y_i\in\{episode\_note,memory\_candidate,belief\_revision,
goal\_adjustment,policy\_proposal\}.
$$

只有满足：

$$
K_i\ge\theta_K\land |E_i|\ge n_{min}
$$

才可进入 `approved`。`memory_candidate` 必须经过记忆去重、冲突检测和 embedding
写入；`belief_revision` 必须经过现有 self-belief append-only 版本链；
`goal_adjustment` 必须经过目标类型路由；`policy_proposal` 只能创建
`behavior_experiment`，不能直接改代码、系统配置或人格常量。

外部供应方数据的 provenance 保持 `provider_reported`，不会自动变成用户事实、
关系事实或自我信念。模型推断长期记忆的 confidence 在写入点再次限制为 $0.6$。

### 13.4 行为实验和结果归因

策略实验将 treatment 作为私有软控制，并保存 baseline、rollback、成功标准和
截止时间。结果只从可观测用户事件计算，沉默不构成负反馈：

$$
O_e=\operatorname{clip}\left(
\frac{1}{|U_e|}\sum_{u\in U_e}signal(u),-1,1
\right).
$$

其中 `signal` 只识别明确正向或负向反馈，未知文本为零。$O_e\ge0.2$ 才能
`confirmed`；$O_e\le-0.2$ 进入 `rolled_back`；其他结果结束实验但不保留策略。
所有 outcome observation 追加写入，并通过 proposal evidence 反向连接到原始提案。

### 13.5 持久化与崩溃恢复

迁移 015 的六类记录分别保存 run、proposal、proposal evidence、decision、
behavior experiment 和 outcome observation。稳定 dedup key 保证重启不重放；
模型/embedding 调用在事务外，短 savepoint 只提交结构化结果。每个 proposal 使用
版本 CAS，每个 decision 追加版本，每个 experiment 使用状态门禁；因此部分失败会
停留在可重试的候选状态，不会把半写入结果伪装成已学习。

---

## 参考文献（待补全）

[1] Park, J.S., et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023.
[2] Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." arXiv 2024.
[3] ...
