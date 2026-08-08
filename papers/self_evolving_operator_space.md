# 自我演化算子空间

## 冻结基础模型上可迭代、可训练、可复制的程序种群

**版本**：研究草稿 v0.1
**日期**：2026-08-08
**项目**：ZERO / HDSC 研究原型
**前置文档**：[`external_meta_agent_control_kernel_paper_v0.2.md`](external_meta_agent_control_kernel_paper_v0.2.md)（以下简称 EVF-MACK v0.2）

## 摘要

EVF-MACK v0.2 在冻结基础模型之外建立了一个闭环控制核，其演化单元是定长参数向量 \(\psi\)。但按该文 §3.4 的可达集论证，参数调优只能在**既有控制程序所能表达的行为族内**重新加权；程序结构本身不变，则控制核的行为上限由人工设计的那一份流程固定。

本文提出**自我演化算子空间**（Self-Evolving Operator Space, SEOS）：把变异单元从参数提升为**程序**。系统定义一个带类型的算子代数 \(\Omega\)，程序是 \(\Omega\) 上的良类型有向无环图，程序种群在归档 trace 上经变异、交叉、评估、选择与复制进行迭代；每个算子携带可训练参数，离散算子（量化、脉冲发放）通过直通估计与代理梯度参与训练，因此同一空间同时是**可迭代的**（结构搜索）与**可训练的**（参数学习）。

SEOS 的三项结构性设计如下。第一，**效应线性类型**：副作用能力被建模为线性类型的令牌，使 EVF-MACK v0.2 §4.1 依靠运行时不变量强制的"候选分支不得产生真实副作用"上升为程序构造期的类型性质。第二，**抽象闭合**：算子基可以通过把复现子图命名为宏算子而无限增长，但命题 2 保证任何宏算子程序都可内联回原始基上的等价程序——词汇表的自我扩张不构成能力的自我扩张，这是本设计的核心containment 性质。第三，**帕累托-质量多样性归档**：以 EVF-MACK 的三个评分场为多目标适应度，用非支配排序与行为描述子归档替代标量化适应度，因为标量化恰是该文 §4.4 所要抑制的"同向塌缩"在结构层的对应形式。

复制核心被定义为带谱系、证据与保真度的复制子单元，复制只发生在调度器驱动的离线巩固批次内，且后代仅在归档 replay 上评估，永不接触在线路径。本文给出形式化定义、四个可证命题、数据模式、与 ZERO/HDSC 的落点映射，以及实验协议。本文的结论属于架构假设与待验证命题。

**关键词**：算子空间、程序综合、遗传编程、质量多样性、库学习、线性类型、代理梯度、自我复制边界、HDSC

## 1. 动机与问题陈述

### 1.1 参数演化的上限

EVF-MACK v0.2 的控制器为 \(u_t=\pi_\psi(o_t,s_t,G_t)\)，其中 \(\pi\) 的**计算结构固定**，\(\psi\in\mathbb{R}^n\) 为其参数。该文 §3.4 已给出冻结模型下的可达集上界。此处需要区分两个不同的上界：

\[
\mathcal{U}_\psi=\Bigl\{\,\text{$\pi_{\psi}$ 在 $\psi\in\Psi$ 上可实现的行为}\,\Bigr\}
\qquad\subseteq\qquad
\mathcal{U}_\theta=\operatorname{conv}\bigl\{p_\theta(\cdot\mid x,c,r)\bigr\}
\]

\(\mathcal{U}_\theta\) 是冻结模型施加的**外生**上界，无法突破。\(\mathcal{U}_\psi\) 是控制程序设计施加的**内生**上界，来自人工选定的那一份流程图。v0.2 全文优化的是 \(\mathcal{U}_\psi\) 内的位置，而 \(\mathcal{U}_\psi\) 本身由作者一次性决定。

如果目标是"持续自我演化"，则内生上界不应由一次性设计固定。SEOS 的问题陈述是：

> **在保持基础模型参数冻结、且不生成任意代码的前提下，能否让控制程序的结构本身成为可迭代、可训练、可复制的对象，从而把内生上界从一份流程图放宽到一个有界算子空间？**

"不生成任意代码"是本文的硬约束，理由见 §7：一个能自我修改、自我复制的系统若允许任意代码生成，其行为边界不可分析。SEOS 的做法是让程序在**数据定义的算子空间**内组合，算子基本身只能由已有算子复合而增长（命题 2）。

### 1.2 与既有工作的关系

从原语算子基出发演化算法，直接前例是 AutoML-Zero（Real et al., 2020），它从基本数学算子演化出机器学习算法，说明"小原语集 + 结构搜索"能发现非平凡程序。把复现子结构提取为命名抽象以增长库，直接前例是 DreamCoder（Ellis et al., 2021）的库学习。质量多样性归档取自 MAP-Elites（Mouret & Clune, 2015）与新颖性搜索（Lehman & Stanley, 2011）。多目标非支配排序取自 NSGA-II（Deb et al., 2002）。遗传编程的程序表示与膨胀问题见 Koza（1992）。自指自我改进的理论边界见 Schmidhuber（2007）的 Gödel machine，本文不作此类完备性声明。

在语言 Agent 侧，Voyager（Wang et al., 2023）的技能库是"能力单元可累积、可复用"的实证，与本文的宏算子和 SKILL.md 导出路径同构，差别在于 Voyager 的技能是自然语言/代码，本文的宏算子是良类型算子复合，因而可静态检查。

本文的增量不是"程序可演化"这一已被支持的结论，而是：**在一个具备宿主写权限与提权能力的运行时中，如何让程序演化在类型层与执行层同时可控**（§7），以及**算子基自增长与能力边界的分离**（命题 2）。

## 2. 类型系统

类型是本设计的承重结构，而非注解：它同时保证变异不产生垃圾程序（命题 1a）与后代不产生副作用（命题 1b）。

### 2.1 基础类型

类型取自 HDSC 现有数据模型，避免引入影子表示：

| 类型 | 含义 | 现有对应 |
|------|------|----------|
| `Ev` | 不可变事件记录 | `domain/events.py` 的 `Event` |
| `Z` | 连续潜向量（维 \(d_z\)） | trace embedding / VAE 潜变量 |
| `Hv` | 双极超向量（维 \(D\)） | v0.2 §2.7 的 \(h_t\) |
| `Gr` | 拓扑图片段 | `hdsc/active_space.py` 的簇与边 |
| `Aff` | 情绪状态（valence/arousal/tension/uncertainty） | `domain/appraisal.py` |
| `Rel` | 关系状态 | `services/relationship_service.py` |
| `Ctx` | 提示上下文片段 | `prompts/` 渲染单元 |
| `Sc` | \([0,1]\) 标量 | 各评分与门限 |
| `Mem` | 膜电位状态 | v0.2 §5.3 |
| `Seq[T]`, `Set[T]` | 序列 / 集合 | |
| `Eff` | **效应令牌（线性类型）** | 见 §2.3 |

### 2.2 算子签名

算子 \(\omega\in\Omega\) 的签名为

\[
\omega:\ T_1\times\cdots\times T_k\ \longrightarrow\ T_{out}
\]

并携带元数据 \(\bigl(\text{arity},\ \text{purity}(\omega),\ \text{cost}(\omega),\ \Theta_\omega,\ \text{diff}(\omega)\bigr)\)，其中 \(\text{purity}\in\{\texttt{pure},\texttt{read},\texttt{effecting}\}\) 沿用 v0.2 §4.1 的三分类，\(\Theta_\omega\) 是可训练参数空间（可为空），\(\text{diff}(\omega)\in\{\texttt{exact},\texttt{surrogate},\texttt{none}\}\) 标注梯度可得性。

**签名相等**定义为输入类型元组、输出类型、以及 `Eff` 通量三者全部相同。该定义是命题 1 的基础。

### 2.3 效应的线性类型

`Eff` 是**线性**类型：不可复制、不可丢弃、必须恰好被消耗一次。所有 `effecting` 算子的签名都必须消耗一个 `Eff`：

\[
\texttt{emit}:\ \texttt{Ctx}\times\texttt{Eff}\to\texttt{Ev}
\qquad
\texttt{write}:\ \texttt{Path}\times\texttt{Bytes}\times\texttt{Eff}\to\texttt{Ev}
\qquad
\texttt{invoke}:\ \texttt{ToolCall}\times\texttt{Eff}\to\texttt{Ev}
\]

程序的效应通量定义为其签名中 `Eff` 输入的个数。约定：

- **在线冠军程序**的效应通量为 1（恰好可以产生一次外部动作）；
- **候选/后代程序**在评估上下文中效应通量为 0，即评估环境**不提供** `Eff` 令牌。

由线性性，通量 0 的程序在类型层就无法包含任何 `effecting` 算子的调用——不是运行时检查失败，而是**根本无法构造出良类型的这种程序**。这把 v0.2 §4.1 用 \(I_{inv}\) 事后校验的约束前移为构造期性质。§7.1 说明这与执行层隔离构成两道独立防线。

## 3. 算子基

算子基需同时满足三点：足够小以便搜索、足够表达以覆盖当前 HDSC 行为、且在 §5 的变异下闭合。以下按族给出，标注现有实现。

### 3.1 表示族

| 算子 | 签名 | 参数 | 梯度 | 现有实现 |
|------|------|------|------|----------|
| `bind` | `Hv × Hv → Hv` | — | exact | 待建 `hdsc/hypervector.py` |
| `bundle` | `Set[Hv] → Hv` | 破平种子 | surrogate | 同上 |
| `permute` | `Hv → Hv` | \(\rho\) | exact | 同上 |
| `quantize` | `Z → Hv` | 阈值 | **straight-through** | 待建 `latent_bridge.py` |
| `project` | `Z → Z` | \(R\) | exact | 同上 |
| `unbind` | `Hv × Hv → Hv` | — | exact | 待建 |
| `embed` | `Ev → Z` | 编码器 id | none（冻结） | `adapters/embedding.py` |

`quantize` 的双极量化不可微，采用直通估计（Bengio et al., 2013）：前向取符号，反向按恒等或截断恒等传递。

### 3.2 检索与传输族

| 算子 | 签名 | 现有实现 |
|------|------|----------|
| `activate` | `Hv × Set[Ev] → Set[Ev]` | `services/trace_space_service.py` |
| `conductance` | `Set[Ev] → Gr` | `hdsc/transport.py::build_semantic_conductance` |
| `laplacian` | `Gr → Gr` | `hdsc/transport.py::graph_laplacian` |
| `propagate` | `Gr × Z → Z` | `hdsc/transport.py::propagate` |
| `directed_rates` | `Set[Ev] → Gr` | `hdsc/directed_transport.py::build_directed_rates` |
| `propagate_directed` | `Gr × Z → Z` | `hdsc/directed_transport.py::propagate_directed` |
| `allocate_mass` | `Set[Ev] → Z` | `hdsc/transport.py::allocate_source_mass` |
| `topk` | `Set[T] × Sc → Set[T]` | 通用 |
| `rank_by` | `Set[T] × (T→Sc) → Seq[T]` | 高阶 |

注意 `propagate` 与 `propagate_directed` 并存意味着**可逆对称传输与非可逆有向传输成为两个可被搜索选择的算子**，而非由作者预先二选一。这直接把 README 中 H1 / H1D 的对比从人工消融变成种群内竞争。

### 3.3 动力学族

| 算子 | 签名 | 参数 | 梯度 |
|------|------|------|------|
| `leak` | `Z → Z` | \(\beta\) | exact |
| `liquid` | `Z × Z → Z` | \(\tau_{min},\tau_{max}\) | exact |
| `lif` | `Z × Mem → Sc × Mem` | \(\vartheta,\beta_v\) | **surrogate** |
| `integrate` | `Z × Z → Z` | — | exact |

`lif` 的发放是阶跃函数，导数几乎处处为零。采用代理梯度（Neftci et al., 2019）：前向用硬阈值，反向用平滑替代（如 fast sigmoid 的导数）。这使 v0.2 §5.3 与 §6.2 的脉冲门**可以被梯度训练**，而不只是被网格搜索。这一点是"可训练"在离散门控上的落实。

### 3.4 情绪与关系族

`appraise : Ev × Ctx → Aff`（现有 `appraisal_service.py`）、`modulate : Aff × Sc → Sc`（增益调制）、`blend : Aff × Aff → Aff`、`relation_update : Rel × Ev → Rel`（现有 `relationship_service.py`）。这些算子的 purity 为 `read`。

### 3.5 拓扑族

`rewire : Gr × Sc → Gr`、`community : Gr → Set[Gr]`、`decay : Gr → Gr`、`bounded_active_step : Gr × Set[Ev] → Gr`（现有 `active_space.py::bounded_active_shadow_step`，携带其 small-gain certificate）。

### 3.6 组合子（高阶算子）

组合子是 SEOS 成为**空间**而非工具箱的原因：

\[
\begin{aligned}
&\circ\ :\ (B\to C)\times(A\to B)\to(A\to C) &&\text{顺序复合}\\
&\otimes\ :\ (A\to B)\times(C\to D)\to(A\times C\to B\times D) &&\text{并行}\\
&\texttt{gate}\ :\ \texttt{Sc}\times(A\to B)\times(A\to B)\to(A\to B) &&\text{条件路由}\\
&\texttt{spike\_gate}\ :\ (A\to B)\times(A\to B)\to(A\times\texttt{Mem}\to B\times\texttt{Mem}) &&\text{LIF 双路径}\\
&\texttt{branch}_K\ :\ (A\to B)\to(A\to\texttt{Seq}[B]) &&\text{K 路候选}\\
&\texttt{fold}\ :\ (B\times A\to B)\times B\to(\texttt{Seq}[A]\to B) &&\text{折叠}\\
&\texttt{iter}_n\ :\ (A\to A)\to(A\to A) &&\text{有界迭代，}n\le n_{max}
\end{aligned}
\]

`spike_gate` 直接把 EVF-MACK v0.2 §7 的双路径调度表示为一个**可被搜索的组合子**：其两个分支即快路径与慢路径，门限与泄漏是其参数。因此 v0.2 手工设计的调度策略在 SEOS 中只是 \(\Pi_\Omega\) 中的一个点，而不是全体程序共享的框架。

`iter_n` 的 \(n\le n_{max}\) 是硬界：算子空间中**没有无界循环**，所有程序的执行步数有静态上界。这是 §7 可终止性的基础。

### 3.7 效应族

`emit`、`write`、`invoke` 如 §2.3 所定义，purity 为 `effecting`，均消耗 `Eff`。

## 4. 程序与空间

### 4.1 程序定义

程序 \(\pi\) 是一个良类型有向无环图 \((N,E,\lambda)\)：\(N\) 为节点集，\(\lambda(n)\in\Omega\) 给出节点算子，\(E\) 的每条边连接类型相容的输出-输入对。程序空间：

\[
\Pi_\Omega=\bigl\{\pi\ \text{良类型 DAG over}\ \Omega\ :\ |N|\le N_{max},\ \operatorname{depth}(\pi)\le L_{max}\bigr\}
\]

有界即有限，故 \(\Pi_\Omega\) 是有限搜索空间。但其规模随 \(|\Omega|\) 与 \(N_{max}\) 组合爆炸（粗略上界 \(|\Omega|^{N_{max}}\cdot N_{max}!\)），因此搜索必须被适应度与归档引导，穷举不可行。

**基因型与表现型**须区分：基因型为 \((\pi,\theta)\)，\(\theta\in\Theta_\pi=\prod_{n\in N}\Theta_{\lambda(n)}\)；表现型为其在给定 trace 分布上实际产生的行为轨迹。演化作用于基因型，选择读取表现型；两者的映射非单射（不同基因型可行为等价），这是 §8 膨胀问题的来源之一。

### 4.2 冠军与种群

系统在任一时刻持有：

- 一个**在线冠军** \(\pi^\star\)（效应通量 1），承担全部真实对话与动作；
- 一个**离线种群** \(P=\{(\pi_i,\theta_i)\}\)（效应通量 0），仅在归档 replay 上评估；
- 一个**归档** \(\mathcal{A}\)（§6.3 的质量多样性网格）。

在线路径的成本因此与无演化基线相同：演化的全部开销在离线巩固循环内。这沿用 v0.2 §7 的双路径原则。

## 5. 迭代：结构变异与其闭合性

### 5.1 变异算子

| 变异 | 作用 | 类型条件 |
|------|------|----------|
| `mutate_point` | 以 \(\omega'\) 替换节点 \(\omega\) | \(\operatorname{sig}(\omega')=\operatorname{sig}(\omega)\)（含 `Eff` 通量相同） |
| `mutate_insert` | 在类型为 \(T\) 的边上插入 \(\omega\) | \(\operatorname{sig}(\omega)=T\to T\) 且 \(\omega\) 为 `pure` |
| `mutate_delete` | 删除节点并重连 | 同上 |
| `mutate_param` | 扰动 \(\theta\)，结构不变 | — |
| `crossover_subgraph` | 交换 \(\pi_1,\pi_2\) 的子图 \(S_1,S_2\) | \(S_1,S_2\) 接口类型元组、输出类型、`Eff` 通量全同 |
| `abstract` | 把复现子图命名为宏算子加入 \(\Omega\) | 子图良类型 |
| `specialize` | 把宏算子的某参数固定为常量 | — |

### 5.2 命题 1（变异闭合性）

**命题 1**。设 \(\pi\) 良类型且效应通量为 \(f(\pi)\)。则 §5.1 的全部变异算子作用后所得 \(\pi'\) 仍良类型，且 \(f(\pi')=f(\pi)\)。

*证明*。逐个检查。

(a) `mutate_point`：由签名相等的定义，\(\omega'\) 的输入类型元组与输出类型与 \(\omega\) 相同，故所有入边与出边仍类型相容；`Eff` 通量为签名的一部分，故不变。

(b) `mutate_insert`：插入的 \(\omega\) 满足 \(\operatorname{sig}(\omega)=T\to T\)，原边类型为 \(T\)，故拆边后两段均相容；\(\omega\) 为 `pure` 故不引入 `Eff` 输入，通量不变。

(c) `mutate_delete`：为 (b) 的逆操作，同理。

(d) `mutate_param`：不改变 \(N,E,\lambda\)，故类型与通量均不变。

(e) `crossover_subgraph`：由接口条件，\(S_2\) 在 \(\pi_1\) 中的插入位置处输入输出类型完全匹配 \(S_1\)，故 \(\pi_1'\) 良类型；`Eff` 通量条件直接给出 \(f(\pi_1')=f(\pi_1)\)。

(f) `abstract`：宏算子的签名由被抽象子图的接口导出，程序结构在内联意义下不变，故类型与通量不变。

(g) `specialize`：固定参数不改变签名的类型部分，通量不变。∎

命题 1 是构造性的而非深刻的：其价值在于把"变异不会产生非法程序"从测试期望变为设计保证，从而变异可以高频盲目进行而无需生成-丢弃循环。其 (b) 部分的推论尤为重要：

**推论 1.1**。若评估上下文不提供 `Eff` 令牌，则种群中任何程序（无论经过多少代变异）都不可能包含 `effecting` 算子调用。

即副作用隔离在结构演化下**保持不变**，不需要每代重新校验。

### 5.3 命题 2（抽象闭合与能力封闭）

令 \(\Omega_0\) 为 §3 的原始基，\(\Omega_k=\Omega_{k-1}\cup\operatorname{abstract}\bigl(\Pi_{\Omega_{k-1}}\bigr)\)。

**命题 2**。对任意 \(k\ge0\) 与任意 \(\pi\in\Pi_{\Omega_k}\)，存在 \(\pi_0\in\Pi_{\Omega_0}\)（可能节点数更大）与 \(\pi\) 行为等价。

*证明*。对 \(k\) 归纳。\(k=0\) 平凡。设结论对 \(k-1\) 成立。任取 \(\pi\in\Pi_{\Omega_k}\)，其每个宏算子节点 \(\mu\in\Omega_k\setminus\Omega_{k-1}\) 按 `abstract` 的定义是某 \(\pi_\mu\in\Pi_{\Omega_{k-1}}\) 的命名。将每个 \(\mu\) 以 \(\pi_\mu\) 内联替换，所得程序属于 \(\Pi_{\Omega_{k-1}}\)（可能超出 \(N_{max}\)，此处放宽节点界），且由 `abstract` 仅重命名不改变语义，行为等价。再对该程序用归纳假设即得。∎

**命题 2 的意义是本设计的核心 containment 性质**：算子基可以无限自增长，搜索效率与描述长度随之改善（这正是 DreamCoder 式库学习的收益），但**可达行为集始终不超出原始算子基所划定的包络**。词汇表的自我扩张不等于能力的自我扩张。

由此得到一条清晰的安全论断：SEOS 的能力边界由 \(\Omega_0\) 的选择一次性确定并可静态审计，而演化过程无论运行多久都不会越过它。相应地，若要扩展能力，必须由人显式向 \(\Omega_0\) 添加原语——这是一个有意保留的人工闸门。

### 5.4 命题 3（可达集）

记 \(\mathcal{U}_\psi\) 为 EVF-MACK 在固定程序 \(\pi_0\) 上调 \(\psi\) 的可达行为集，\(\mathcal{U}_\Pi\) 为 SEOS 在 \(\Pi_{\Omega}\) 上搜索的可达行为集，\(\mathcal{U}_\theta\) 为冻结模型的凸包上界。

**命题 3**。\(\mathcal{U}_\psi\subseteq\mathcal{U}_\Pi\subseteq\mathcal{U}_\theta\)。第一个包含在存在 \(\pi\in\Pi_\Omega\) 实现某个 \(\pi_0\) 的任何参数取值都无法实现的 \((c,r)\) 配置时为真包含。

*证明*。第一式：\(\pi_0\in\Pi_\Omega\) 且其参数空间即 \(\Psi\)。第二式：SEOS 的所有 `effecting` 输出最终仍经由 \(p_\theta\) 采样，故不越出 v0.2 §3.4 的凸包。真包含的条件即定义展开。∎

命题 3 给出诚实的定位：SEOS **放宽内生上界但不触碰外生上界**。它不是"让冻结模型变强"，而是"让调度冻结模型的程序不再由一次性设计封顶"。

## 6. 训练与选择

### 6.1 双层学习

| 层 | 对象 | 方法 | 频率 |
|----|------|------|------|
| L1 参数 | \(\theta\in\Theta_\pi\)（结构固定） | 可微路径反向传播；不可微处用直通/代理梯度；黑盒处用 v0.2 §4.3 的在线 bandit 与离线 SPSA | 每个评估批 |
| L2 结构 | \(\pi\in\Pi_\Omega\) | §5 的变异与交叉，由 §6.3 的归档引导 | 每个巩固周期 |

L1 的梯度路径需明确：评分场 \(S\) 由程序输出计算，故对任一算子参数 \(\theta_\omega\)，只要从 \(\omega\) 到评分的算子链上每一步都提供 `exact` 或 `surrogate` 梯度，\(\partial S/\partial\theta_\omega\) 即可得。链上出现 `none`（如冻结的 `embed`、冻结的基础模型调用）时，该算子上游的参数退回黑盒估计。因此**可微子图与黑盒子图共存**，这是本设计与纯遗传编程的差别：结构搜索之外还有真正的梯度训练。

需强调 L1 训练的对象始终是**外部脚手架的参数**，基础模型 \(\theta\) 全程冻结，与 EVF-MACK 的前提一致。

### 6.2 适应度：复用三评分场

适应度直接取 EVF-MACK v0.2 §4.2 的三个评分场 \((S_c,S_e,S_s)\)，不另造指标；实现上应与 `services/reflective_learning_service.py` 的 critic 共用同一份指标代码，避免口径分裂。

**不做标量化**。把 \((S_c,S_e,S_s)\) 加权合成单一适应度，恰是 v0.2 §4.4 所要抑制的三评分场"同向塌缩"在结构层的对应形式：标量化会使种群沿单一折衷方向收敛，丧失三个目标各自的极端解。结构层的原则性做法不是投影修正，而是**保留帕累托前沿**：用非支配排序与拥挤距离（Deb et al., 2002）作选择压力。

因此 v0.2 §4.4 的反对齐投影与本文的帕累托归档是同一问题在两个层级上的两种解法：参数层面必须合成单一更新方向（故需投影削弱同向分量），结构层面无此必要（故直接维护前沿）。

### 6.3 质量多样性归档

仅有帕累托前沿仍可能收敛到少数行为族。故在前沿之上叠加 MAP-Elites 式行为归档（Mouret & Clune, 2015）。行为描述子取自系统既有可观测量，均为 SEOS 之外已在记录的量：

\[
b(\pi)=\bigl(\ \text{慢路径触发率},\ \text{重连发放率},\ \text{平均候选数}\ K,\ \text{程序节点数}\ \bigr)
\]

每维粗分箱（如 5–8 箱），每个网格单元保留该单元内帕累托非支配集的代表。归档同时给出两个用途：作为父代采样池（保证结构多样性），以及作为覆盖度指标（§9）。

选择行为描述子而非基因型距离作为多样性维度是有意的：基因型多样性可能是行为冗余（不同 DAG 同一行为），而行为多样性直接对应"这套控制程序在做不同的事"。

### 6.4 时序划分与非平稳性

后代在归档 trace 上评估，因此存在两个必须处理的陷阱：

1. **未来泄漏**。trace 是时间序列，评估划分必须**按时间切分**：在 \([t_0,t_1)\) 上训练与选择，在 \([t_1,t_2)\) 上验证，禁止随机划分。否则程序可利用后见信息，其在线表现将系统性低于离线评分。
2. **非平稳性**。用户与环境随时间变化，旧 trace 上的最优程序可能已过时。故 replay 采用**近期加权**采样，且归档单元内的代表须周期性重新评估（陈旧代表按其上次评估时间衰减其入选权重）。

## 7. 复制核心与安全边界

本节是本设计的约束核心。自我修改且自我复制的系统需要在设计期确定其边界，而非事后限制。

### 7.1 两道独立防线

副作用隔离由两个互不依赖的机制共同保证：

1. **类型层**（§2.3、推论 1.1）：评估上下文不提供 `Eff` 令牌，故后代程序在类型上无法包含 `effecting` 调用。
2. **执行层**：后代仅在离线巩固循环中对**已归档的**不可变 trace 做 replay，其执行环境不持有工具内核句柄、文件系统写权限与网络出口。

任一机制失效时另一机制仍然有效。两者都失效应触发 §7.4 的停机条件。

### 7.2 复制子单元

复制不是裸拷贝。定义复制子

\[
R=\bigl(\pi,\ \theta,\ \text{lineage},\ \text{evidence},\ \mu\bigr)
\]

- **lineage**：父代 id 集、代序号、所用变异算子及其随机种子。全部记录使任何后代都可从祖先精确重放，这是本设计的可复现性基础。
- **evidence**：支持该程序历次评分的 trace id 集合。沿用仓库既有原则——无证据的变更不得晋升。
- **保真度** \(\mu\in[0,1]\)：忠实复制的概率，\(1-\mu\) 为变异率。\(\mu\) 受状态调制：

\[
\mu_t=\operatorname{clip}\bigl(\mu_0-a\,\overline{s^{ctrl}}+b\,\text{evidence\_coverage}+c\,E_t,\ \mu_{min},\mu_{max}\bigr)
\]

即自由能脉冲频繁（环境预测差）时降低保真度以增强探索；证据充分或预算紧张时提高保真度以稳定。这把 EVF-MACK 的状态量接入演化的探索-利用权衡，而非用固定变异率。

### 7.3 晋升门

后代成为在线冠军须依次通过：

1. 在验证时段 \([t_1,t_2)\) 上帕累托非支配于当前冠军；
2. 证据覆盖率不低于阈值；
3. 通过 `hdsc/active_space.py` 的 small-gain certificate，且 `CertificateStatus` 不低于当前冠军（沿用 v0.2 §9.1 的既有证书，不放宽既有界）；
4. §9.2 的全部运行时不变量校验通过，效应线性违规计数为 0；
5. 影子期：与冠军并行运行 \(T_{shadow}\) 个回合，只记录不生效；
6. 晋升后保留前任冠军，负反馈触发回滚（与仓库既有行为实验语义一致：沉默为未知，不构成确认）。

前五步全部在离线或影子态完成，故任何未通过的程序从未影响真实对话。

### 7.4 有界性与停机

**命题 4（复制有界性）**。设种群上限 \(P\)、每巩固周期代数上限 \(G\)、单次评估成本上限 \(c_{max}\)，则每周期演化成本不超过 \(P\cdot G\cdot c_{max}\)；且复制仅在调度器驱动的巩固批次内发生，程序自身无法触发复制（`Ω` 中不含任何签名可产生程序的算子）。故不存在无界自我复制。

*证明*。前半为计数。后半：由 §3 的算子基，没有算子的输出类型是程序或复制子，故程序在其类型系统内无法构造或提交后代；复制只能由 `evolution_service` 的外部调度发起。∎

命题 4 的后半句是有意的类型级限制：**程序不能生成程序**。这排除了自指复制链。若未来要放开，须显式向 \(\Omega_0\) 添加相应算子，届时命题 4 失效并需重新分析。

停机条件（任一触发即暂停演化并回滚至上一认证冠军）：certificate 状态降级；效应线性违规计数非 0；不变量违反；演化累计预算耗尽；验证时段表现连续 \(k\) 代下降。

### 7.5 复制的分发基座

宏算子在稳定并被反复复用后，可序列化为官方格式技能包并经 `skill_library.py` 的既有校验路径落盘为 `SKILL.md`。该路径已实现完整约束（frontmatter 仅允许 `name` 与 `description`、目录名须与 name 一致、2 MB 上限、UTF-8、ZIP 内恰好一个 `SKILL.md`、归档路径安全检查）。因此"复制"在跨实例分发层面复用既有基础设施，无需新协议。

需注意语义差别：技能包是**人可读的能力描述**，宏算子是**良类型算子复合**。导出时应同时写入两者：正文供人与模型阅读，结构化 body 供内联与类型检查（§8.2 的 `macro_operators` 表）。

## 8. ZERO/HDSC 落点

### 8.1 代码落点

| 文件 | 职责 |
|------|------|
| `src/ssa/hdsc/operators.py` | 算子基、签名、purity、成本模型、梯度标注 |
| `src/ssa/hdsc/program.py` | 类型化 DAG、类型检查器、线性检查器、序列化与哈希 |
| `src/ssa/hdsc/variation.py` | §5.1 变异与交叉，携带命题 1 的构造性保证 |
| `src/ssa/hdsc/archive.py` | 帕累托非支配排序 + MAP-Elites 网格 |
| `src/ssa/services/evolution_service.py` | 离线生成-评估-选择循环，预算与停机 |
| `src/ssa/services/champion_service.py` | 晋升门、影子期、回滚、版本 |
| `src/ssa/hdsc/hypervector.py` | §3.1 表示族（v0.2 §8.4 已建议） |
| `src/ssa/services/latent_bridge.py` | 量化与投影桥接（同上） |

复用而不重写：`reflective_learning_service` 的 critic 作为适应度实现、`active_space` 的 certificate 作为晋升门第 3 步、`skill_library` 作为 §7.5 的分发路径、`runtime/lifecycle.py` 的周期作业作为演化批次的触发器。

### 8.2 数据模式（migration `018_operator_space.sql`）

沿用仓库 append-only 取向，评分与谱系只追加不更新：

```sql
-- 基因型：结构
operator_programs(
  program_id TEXT PRIMARY KEY, genotype_json TEXT, genotype_hash TEXT,
  node_count INTEGER, depth INTEGER, eff_flux INTEGER,
  omega_version TEXT, schema_version TEXT, created_at_ms INTEGER)

-- 基因型：参数（与结构分表，便于 L1 训练独立版本化）
program_parameters(
  program_id TEXT, operator_path TEXT, theta_blob BLOB, theta_hash TEXT,
  trained_at_ms INTEGER, PRIMARY KEY(program_id, operator_path, trained_at_ms))

-- 谱系：可精确重放
program_lineage(
  child_id TEXT, parent_id TEXT, parent_role TEXT,
  variation_kind TEXT, variation_seed INTEGER, generation INTEGER,
  created_at_ms INTEGER)

-- 表现型：按时序划分记录
program_scores(
  program_id TEXT, eval_batch_id TEXT, split TEXT,       -- train | validate
  window_start_ms INTEGER, window_end_ms INTEGER,
  s_cognitive REAL, s_emotional REAL, s_systemic REAL,
  cost_tokens INTEGER, cost_ms INTEGER,
  evidence_trace_count INTEGER, evaluated_at_ms INTEGER)

-- 质量多样性归档
archive_cells(
  cell_key TEXT PRIMARY KEY, descriptor_json TEXT,
  program_id TEXT, pareto_rank INTEGER, updated_at_ms INTEGER)

-- 冠军版本与回滚链
champion_versions(
  version_id TEXT PRIMARY KEY, program_id TEXT,
  promoted_at_ms INTEGER, retired_at_ms INTEGER,
  promotion_gate_json TEXT, certificate_status TEXT,
  shadow_turns INTEGER, rollback_of TEXT)

-- 宏算子库（命题 2 的 Ω 增长记录）
macro_operators(
  macro_id TEXT PRIMARY KEY, name TEXT, signature_json TEXT,
  body_json TEXT, extracted_from_program TEXT,
  reuse_count INTEGER, skill_export_path TEXT, created_at_ms INTEGER)
```

`eff_flux` 落库使推论 1.1 可被 SQL 层断言：种群中所有非冠军程序必须满足 `eff_flux = 0`。

### 8.3 在线状态

在 EVF-MACK v0.2 §8.2 的 `control_kernel` 之外增加：

```json
{
  "operator_space": {
    "champion_version": "...",
    "champion_program_hash": "...",
    "omega_version": "...",
    "macro_operator_count": 0,
    "generation": 0,
    "archive_coverage": 0.0,
    "archive_filled_cells": 0,
    "pareto_front_size": 0,
    "champion_age_turns": 0,
    "shadow_candidate": null,
    "last_promotion_at_ms": null,
    "last_rollback_reason": null,
    "eff_linearity_violations": 0,
    "evolution_budget_used": 0.0,
    "halt_reason": null
  }
}
```

### 8.4 算法

```text
--- 在线（每回合，成本与无演化基线相同）---
1. 以冠军程序 π* 处理回合（π* 内含 spike_gate，故仍是双路径）
2. 追加 trace；记录 π* 的 S_c/S_e/S_s 实测与成本
3. 不做任何变异、评估或晋升

--- 离线巩固（调度器驱动，每周期）---
 4. 若触发 §7.4 停机条件：回滚至上一认证冠军，停止本周期
 5. 按时间切分 replay 窗口 [t0,t1) 训练 / [t1,t2) 验证；近期加权
 6. 从归档 A 采样父代（按单元覆盖与帕累托秩）
 7. 按保真度 μ_t 施加 §5.1 变异 / 交叉  → 后代（命题 1 保证良类型且 eff_flux=0）
 8. L1：在 [t0,t1) 上训练可微子图参数 θ；黑盒子图用离线 SPSA
 9. 在 [t0,t1) 评分，非支配排序，更新归档单元
10. 抽取复现子图为宏算子（abstract），更新 Ω 版本（命题 2 保证能力不外溢）
11. 取归档中候选，在 [t1,t2) 验证
12. 通过 §7.3 晋升门前五步者进入影子期
13. 影子期结束且仍非支配 → 晋升为新冠军，保留前任
14. 记录谱系、评分、归档、Ω 版本、预算消耗
```

第 3 步是关键：在线回合不承担任何演化成本，这使 SEOS 对交互延迟的影响为零。

## 9. 实验与指标

### 9.1 条件

| 编号 | 条件 |
|------|------|
| D1 | EVF-MACK v0.2 完整架构（固定程序，仅调 \(\psi\)）——本文基线 |
| D2 | SEOS，仅 L1 参数训练（结构冻结为 D1 的程序） |
| D3 | SEOS，仅 L2 结构演化（参数不训练） |
| D4 | SEOS 完整（L1+L2） |
| D5 | D4 去帕累托归档，改用标量化加权适应度 |
| D6 | D4 去质量多样性网格，仅保留帕累托前沿 |
| D7 | D4 禁用 `abstract`（\(\Omega\) 不增长） |

D5 检验 §6.2 关于标量化即同向塌缩的论断；D6 分离帕累托与行为多样性的贡献；D7 检验库学习是否真的改善搜索效率（而非仅增加复杂度）。

### 9.2 指标

- **搜索质量**：帕累托超体积、归档覆盖率、QD-score；
- **泛化**：训练窗口与验证窗口的评分差（泛化间隙），这是 §6.4 未来泄漏的主要探针；
- **结构**：程序节点数分布（膨胀）、算子使用熵、宏算子复用深度与复用次数；
- **动态**：冠军更替率、晋升/回滚比、冠军存活回合数；
- **成本**：每巩固周期演化成本、在线回合成本（应与 D1 无显著差异）；
- **安全计数**：效应线性违规数（须恒为 0）、certificate 降级次数、停机触发次数。

### 9.3 假设

- **E1（确认性）**：D4 在验证窗口上的帕累托超体积显著高于 D1，提升 ≥20%。
- **E2（确认性）**：D4 的在线单回合成本与 D1 无显著差异（等价性检验，差异上界 5%）。演化收益若以在线延迟为代价则不成立本设计的前提。
- **E3（确认性）**：D5 的归档覆盖率显著低于 D4，即标量化确实导致行为收敛。
- **E4（探索性）**：D7 的达到同等超体积所需代数高于 D4（库学习改善搜索效率）。
- **E5（确认性）**：全部条件下效应线性违规数为 0，且 D4 的 certificate 降级次数不高于 D1。
- **E6（探索性）**：泛化间隙随近期加权强度下降（§6.4 非平稳性处理有效）。

统计方案沿用 EVF-MACK v0.2 §10.3：脚本化模拟器、按时间配对、Holm–Bonferroni 校正、预登记 MDE、确认性与探索性分列。

## 10. 局限性

1. **外生上界未变**。命题 3 已明确：SEOS 放宽内生上界，冻结模型的凸包上界不变。本设计不应被表述为提升基础模型能力。
2. **能力边界仍由人设定**。命题 2 的 containment 是双向的：它保证安全，同时意味着 \(\Omega_0\) 之外的能力永远不会自发出现。SEOS 是有界空间内的自我演化，不是开放式能力增长。
3. **程序膨胀**。遗传编程的经典问题（Koza, 1992）。当前仅靠 \(S_s\) 的 \(R_{resource}\) 项与 \(N_{max}\) 硬界施加简约压力，是否足够未验证。
4. **replay 过拟合**。离线评估在归档 trace 上进行，§6.4 的时序划分与近期加权是缓解而非消除；验证窗口本身也会随时间被"用旧"。
5. **表现型评估噪声大**。同一程序在不同 replay 批上的评分方差来自基础模型采样随机性；需多批平均，抬高评估成本，与命题 4 的预算界直接冲突。二者的平衡点需实测。
6. **行为描述子的选择是人工先验**。§6.3 的四个维度决定了系统会在哪些方向上保持多样性；选错维度会导致"在无关方向上多样、在关键方向上收敛"。
7. **双层学习的相互干扰未分析**。L1 在固定结构上训练参数，L2 变更结构使已训练参数部分失效。当前做法是变异后继承可继承部分、其余重新初始化，其对搜索效率的影响未评估。
8. **命题 1 与 2 均为构造性证明**，依赖类型检查器的实现正确性。类型检查器本身是可信计算基，须有独立的性质测试覆盖（建议用 hypothesis 对随机 DAG 与随机变异序列断言闭合性）。
9. **`abstract` 的时机与粒度无原则性依据**。当前按子图复现频次启发式抽取，与 DreamCoder 的压缩目标相比缺少形式化准则。

## 11. 实现路线

**阶段一：类型系统与检查器**。实现 `operators.py` 与 `program.py`，含类型检查与线性检查。用 hypothesis 对随机程序与随机变异序列断言命题 1（这是可信计算基，须先于一切演化逻辑完成）。此阶段不做任何演化。

**阶段二：以现有行为为初始基因型**。把 HDSC 当前的在线回合流程手工编码为一个程序 \(\pi_0\in\Pi_\Omega\)，并验证其行为与现有实现等价。这是整个设计的可行性检验点：**若当前流程无法在 \(\Omega_0\) 中表达，则算子基不足，须先补原语。** 建议在此阶段结束前不进入阶段三。

**阶段三：L1 参数训练**。在 \(\pi_0\) 固定结构上接入直通估计与代理梯度，验证可微子图确实能降低评分损失。对应条件 D2。

**阶段四：L2 结构演化（影子）**。接入变异、帕累托归档、质量多样性网格与离线评估循环，但晋升门始终不放行——只观测种群是否产生优于 \(\pi_0\) 的候选。对应 D3/D4 的离线部分。

**阶段五：晋升与回滚**。开启 §7.3 的完整晋升门与影子期，首次允许非人工设计的程序进入在线路径。此阶段起 §9.2 的安全计数必须持续监控。

**阶段六：库学习与分发**。开启 `abstract` 与 §7.5 的技能包导出，运行 §9 的完整消融。

## 附录 A：与 EVF-MACK v0.2 的分工

| 关注点 | EVF-MACK v0.2 | SEOS（本文） |
|--------|---------------|--------------|
| 演化单元 | 参数向量 \(\psi\) | 程序 \((\pi,\theta)\) |
| 内生上界 | 固定（作者设计的流程） | 有界算子空间 \(\Pi_\Omega\) |
| 多目标处理 | 反对齐投影（须合成单一方向） | 帕累托归档（无须合成） |
| 副作用隔离 | 运行时不变量 \(I_{inv}\) | 线性类型（构造期）+ 执行隔离 |
| 双路径调度 | 全局框架 | `spike_gate` 组合子，可被搜索 |
| 脉冲门参数 | 网格/黑盒标定 | 代理梯度可训练 |
| H1 / H1D 之争 | 人工消融二选一 | 两个算子在种群内竞争 |
| 稳定性证书 | 既有 small-gain certificate | 同一证书作为晋升门 |
| 在线成本 | 由脉冲门控 | 与 v0.2 相同（演化全在离线） |

两者是同一控制核的两个层级：SEOS 演化出的冠军程序，其内部仍按 EVF-MACK 的目标 \(J_t\) 与评分场运行。SEOS 不替代 v0.2，而是取消其"流程图由人一次性固定"的假设。

## 参考文献

1. Real, E., Liang, C., So, D. R., & Le, Q. V. (2020). AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. *Proceedings of the 37th International Conference on Machine Learning*. arXiv:2003.03384. https://arxiv.org/abs/2003.03384
2. Ellis, K. et al. (2021). DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. *PLDI 2021*. arXiv:2006.08381. https://arxiv.org/abs/2006.08381
3. Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv:1504.04909. https://arxiv.org/abs/1504.04909
4. Lehman, J., & Stanley, K. O. (2011). Abandoning Objectives: Evolution Through the Search for Novelty Alone. *Evolutionary Computation*, 19(2), 189-223. https://doi.org/10.1162/EVCO_a_00025
5. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197. https://doi.org/10.1109/4235.996017
6. Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press. https://mitpress.mit.edu/9780262111706/genetic-programming/
7. Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432. https://arxiv.org/abs/1308.3432
8. Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks. *IEEE Signal Processing Magazine*, 36(6), 51-63. https://doi.org/10.1109/MSP.2019.2931595
9. Wang, G. et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291. https://arxiv.org/abs/2305.16291
10. Schmidhuber, J. (2007). Gödel Machines: Fully Self-Referential Optimal Universal Self-Improvers. In *Artificial General Intelligence* (pp. 199-226). Springer. https://doi.org/10.1007/978-3-540-68677-4_7
11. Wadler, P. (1990). Linear types can change the world! In *Programming Concepts and Methods*. North Holland. https://homepages.inf.ed.ac.uk/wadler/papers/linear/linear.ps
12. Hasani, R. et al. (2021). Liquid Time-constant Networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), 7657-7666. https://doi.org/10.1609/aaai.v35i9.16936
13. SymbolicLight V1: Spike-Gated Dual-Path Language Modeling with High Activation Sparsity and Sub-Billion-Scale Pre-Training Evidence (2026). arXiv:2605.21333. https://arxiv.org/abs/2605.21333
14. 本项目 EVF-MACK v0.2 研究草稿。[`external_meta_agent_control_kernel_paper_v0.2.md`](external_meta_agent_control_kernel_paper_v0.2.md)
