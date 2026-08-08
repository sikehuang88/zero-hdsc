# 冻结基础模型上的熵-变分自由能外部控制核

## 基于情绪状态、脉冲门控液态认知与动态混合拓扑的持续自我演化架构

**版本**：研究草稿 v0.2
**日期**：2026-08-08
**项目**：ZERO / HDSC 研究原型
**前一版本**：v0.1（`external_meta_agent_control_kernel_paper.md`）

> **修订摘要**：v0.2 修正了 v0.1 中的六处形式化错误（熵项重复计入、反对齐投影在目标一致时反向、梯度符号约定冲突、扰动强度中能量项符号、`h_t` 符号碰撞、离散时间常数在真实调度周期下退化），补入候选分支的副作用隔离约束，将元梯度从在线双侧 rollout 改为在线 bandit + 离线有限差分，用带时间积分的 LIF 累积器门替换无记忆瞬时重连门，并把稳定性一节接到仓库中已实现的 H2 small-gain certificate。完整逐条清单见附录 B。

## 摘要

基于人类反馈强化学习（RLHF）或其他偏好优化的语言模型，通常在固定偏好目标、KL 约束和相对稳定的数据分布下训练。此过程提高了输出的一致性，却也可能使模型的行为轨迹收敛到低变化吸引子：面对相似信号时，模型重复调用相近的语义路径、情绪姿态和行动策略。本文提出一种让基础模型参数保持冻结、由外部闭环驱动的控制架构，称为**熵-变分自由能外部控制核**（Entropy-Variational Free-Energy Meta-Agent Control Kernel, EVF-MACK）。

EVF-MACK 将冻结的基础模型视为随机策略执行器，将记忆检索、情绪状态、系统能量、候选分支、工具路径和输出选择纳入一个可观测的闭环控制系统。控制核使用变分自由能描述预测误差和后验-先验冲突，使用熵和信息增益维持受控探索，使用情绪状态调节时间常数和扰动强度，并通过反对齐梯度、反对称旋转耦合和动态混合拓扑维持亚稳态。系统将快速策略变量与慢速长期变量分离，只有经过证据门控的结果才进入慢变量更新。

v0.2 引入两项结构性修改。第一，控制核采用**脉冲门控双路径**算力调度：一条低成本连续路径常开，一条高成本离散路径仅在自由能预测误差经时间积分越阈时触发，从而使多分支采样和元梯度估计的成本与情境显著性成比例，而非每回合恒定支出。第二，所有候选分支的评分只允许基于**无副作用的预测转移**，仅被选中的分支执行真实动作；这在具备文件写入与提权 shell 的宿主工具集下是必要的安全约束，而非可选优化。

本文给出形式化定义、离散算法、可验证的方向性保证、与 ZERO/HDSC 的实现映射，以及可复现实验协议。本文的结论目前属于架构假设和待验证命题，尚未将其描述为基础模型能力的改变。

**关键词**：外部 Agent、变分自由能、熵调制、主动推断、液态认知、脉冲门控、双路径、动态拓扑、持续学习、HDSC、情绪状态

## 1. 引言

### 1.1 问题背景

设基础模型的参数为 \(\theta\)，给定输入 \(x\) 后输出分布为：

\[
p_\theta(y\mid x)
\]

偏好优化的目标可以抽象为任务损失、偏好损失和策略约束的组合（Ziegler et al., 2019；Ouyang et al., 2022）：

\[
\mathcal{L}_{align}(\theta)
 = \mathcal{L}_{task}(\theta)
 + \lambda \mathcal{L}_{preference}(\theta)
 + \beta D_{KL}(\pi_\theta\|\pi_{ref})
\]

当目标、参考策略和训练分布长期保持稳定时，优化轨迹容易进入低变化吸引子。这里的"稳态"并非一定是单一参数点，也可以表现为一组相近的输出模板、决策节奏和拒答边界。

本文关心的工程问题是：**在基础模型参数保持冻结的前提下，能否通过外部 Agent 控制层改变整体系统的行为分布、探索轨迹和长期状态演化？**

### 1.2 核心观点

外部控制层直接操作以下变量：

- 检索到的记忆和上下文顺序；
- 当前情绪和 organism 状态的显式表示；
- 采样温度、候选数量和推理模式；
- 工具调用与观察顺序；
- 候选结果的批评、重排和门控；
- 快速策略与慢速策略的交叉；
- 活跃记忆图和行为模块图的重连；
- 每回合投入的算力档位。

因此，系统不依赖基础模型内部的反向传播，而使用外部评分、轨迹结果和环境反馈估计元策略的更新方向。

### 1.3 贡献范围

本文的贡献是架构和验证协议：

1. 给出熵、变分自由能、情绪调制和动态拓扑的统一状态模型，并显式处理自由能与熵项的重叠。
2. 用高维反对齐投影和反对称耦合替代低维向量的直接叉积，并给出保证不反转单个目标方向的构造。
3. 提出快慢变量分离和证据门控，限制持续自我更新的累积漂移。
4. 提出脉冲门控双路径算力调度，使控制核的边际成本与情境显著性成比例。
5. 给出候选分支的副作用隔离约束，使多分支探索在具备宿主写权限的工具集下可用。
6. 将模型映射到 ZERO/HDSC 的 trace、active space、lifecycle 和 consolidation 模块，并复用已实现的 critic 与 small-gain certificate。
7. 给出基线、消融实验、统计功效方案和可观测指标，便于后续复现。

### 1.4 与已有实现的关系

本文不是从零提议。ZERO/HDSC 仓库中已存在三个与本文直接同构的构件，本文的相应章节是它们的推广而非替代：

- `src/ssa/services/reflective_learning_service.py` 中的确定性 critic 已实现按来源覆盖度、来源可信度、一致性和可伪证性加权的评分，与 §4.2 的认知评分场同构；
- `src/ssa/hdsc/active_space.py` 中的 `_small_gain_certificate` 已输出增益矩阵、谱半径、认证半径、局部记忆收缩界、语义跳跃界和质量守恒残差，并由 `tests/property/test_hdsc_h2_active_space.py` 的性质测试覆盖，是 §9 的实际载体；
- `tests/fixtures/appraisal_standard_events.v1.json` 与 `tests/contract/test_appraisal_fixture_contract.py` 提供了 appraisal 输出的 schema 契约，是 §4.2.2 情绪目标校准的锚点。

本文所提的新组件是 §5 的脉冲门、§6.2 的累积器重连门、§7 的双路径调度和 §4 的评分场-元梯度回路。

## 2. 相关工作与概念区分

### 2.1 冻结权重下的外部自我改进

"不更新权重、只更新外部状态"的路线已有实证基础，本文的命题需要相对这些工作定位。Generative Agents（Park et al., 2023）用记忆流、重要性打分、反思和检索构成外部循环，产生了长期一致的行为轨迹；Reflexion（Shinn et al., 2023）用语言化的自我批评与情节记忆在冻结模型上取得跨轮次改进；Voyager（Wang et al., 2023）用可增长的技能库实现开放式能力累积；CoALA（Sumers et al., 2023）给出了语言 Agent 的记忆-动作-决策分层框架。

这些工作共同支持本文的前提：外部回路确实能改变系统级行为分布。本文与它们的差别在于控制量的形式化——上述工作的外部循环主要是**离散的、由提示词表达的启发式**，而 EVF-MACK 把外部循环写成带显式目标函数 \(J_t\)、显式状态 \(s_t\)、可估计梯度和可验证约束的控制系统。因此本文的增量是可测量性与可证明约束，而不是"外部回路有效"这一已被支持的结论。

### 2.2 变分自由能

给定观测 \(o_t\)、潜状态 \(z_t\)、生成模型 \(p_\theta(o_t,z_t)\) 和近似后验 \(q_\phi(z_t)\)，变分自由能为：

\[
F(q_\phi,\theta;o_t)
 = \mathbb{E}_{q_\phi(z_t)}
 [\log q_\phi(z_t)-\log p_\theta(o_t,z_t)]
\]

展开为精度项与复杂度项：

\[
F
 = \underbrace{-\mathbb{E}_{q_\phi}[\log p_\theta(o_t\mid z_t)]}_{\text{精度}}
 + \underbrace{D_{KL}(q_\phi(z_t)\|p_\theta(z_t))}_{\text{复杂度}}
\]

在本文中，\(F\) 是外部控制器对当前状态解释质量的度量，不将其直接等同于基础模型训练损失（Friston, 2010；Friston et al., 2015）。

**与熵项的重叠（v0.1 勘误）**。由 \(\mathbb{E}_{q}[\log q]=-H(q)\) 可得恒等式：

\[
-F = H(q_\phi) + \mathbb{E}_{q_\phi}[\log p_\theta(o_t,z_t)]
\]

即 \(-F\) **已经包含一个 \(+H(q_\phi)\) 项**。因此在目标函数中另加熵项时，必须把该系数理解为超出变分项的**额外**熵权重，否则等效系数会被误报。§3.3 按此约定重写。

### 2.3 熵、信息增益与后验塌缩

后验熵定义为：

\[
H(q_\phi)=-\mathbb{E}_{q_\phi}[\log q_\phi]
\]

信息增益定义为：

\[
I(z_t;o_{t+1})
 = H(q(z_t)) - \mathbb{E}_{o_{t+1}}H(q(z_t\mid o_{t+1}))
\]

**动机修正（v0.1 勘误）**。v0.1 称"仅最小化自由能会倾向于快速收缩状态分布"。由 §2.2 的恒等式可知该陈述作为一般性结论不成立：最小化 \(F\) 本身就包含最大化 \(H(q)\)。实践中出现的分布收缩来自另外两个具体机制：

1. **后验塌缩**：当解码器足够强时，\(D_{KL}(q_\phi(z)\|p(z))\to 0\)，\(q_\phi\) 退化为先验，潜变量不再携带观测信息；
2. **摊销间隙**：\(q_\phi\) 作为摊销推断族无法覆盖真实后验的多峰结构，优化会选择单峰解。

因此额外的熵项和信息增益项针对的是上述两个机制，而非 \(F\) 的定义本身。这一区分决定了 §10 中相应指标的读法：应同时报告 \(D_{KL}(q_\phi\|p)\) 与激活维数，塌缩表现为二者同时趋零。

### 2.4 液态认知

液态认知在本文中表示**时间常数随状态变化的连续动力学**，而非对某一种具体神经网络实现的限定。它使同一个 Agent 在高唤醒、低能量、等待和离线模式下使用不同的内部更新速度；其神经动力学参考 Hasani et al.（2021）。

### 2.5 动态混合拓扑

系统状态由多个节点和关系组成。节点可以是 trace 社区、行为策略、目标或外部工具；边可以表示语义关联、情绪关联、因果证据或竞争关系。拓扑在运行中局部重连，而长期归档保持可追溯。小世界重连的结构性直觉可参考 Watts & Strogatz（1998），本文的拓扑规则仍需通过实验确定。

### 2.6 脉冲门控与双路径

本文的算力调度借用脉冲神经元的两个性质：**阈值发放**与**时间积分**。Leaky Integrate-and-Fire（LIF）单元维持膜电位，对输入驱动带泄漏地积分，越阈时发放一个二值脉冲并复位（Gerstner & Kistler, 2002）。其控制意义是：**单次高幅噪声不足以触发动作，而持续的中等幅度驱动可以**——这正是持续自我更新需要的抗漂移选择性。

SymbolicLight V1（2026）在语言建模层面给出了两条与本文相关的经验结果。第一，其 Dual-Path SparseTCAM 用"指数衰减聚合路径负责长程记忆 + 脉冲门控局部注意力路径负责短程精度"替代密集自注意力，在 194M 参数、3B token 中英语料、>89% 逐元素激活稀疏度下达到验证 PPL 8.88–8.93。第二，其消融显示在同等稀疏度下用**确定性 top-k 掩码替换 LIF 动力学会造成更大的退化**，作者据此认为驱动性能的是时间积分而非稀疏本身；同时报告早期纯脉冲原型虽保持稀疏但存在持续的质量瓶颈，V1 通过引入受控连续路径缩小该差距。

本文对该工作的引用范围需要严格限定。SymbolicLight V1 属于**架构加训练**路线（从零训练、亚十亿规模、PPL 落后 GPT-2 201M 约 7.7%），与本文"权重冻结、外部控制"的设定相反，其数值结果不构成对冻结大模型的验证。本文只借用两条机制层面的结论：其一，门控应带时间积分而非瞬时阈值（用于 §5.2 与 §6.2）；其二，量化/离散路径应与连续路径并存而非取代之（用于 §2.7 的三层表示）。

### 2.7 非人类中心的高维表示

本文不把人类人格、情绪和社会行为作为智能系统的唯一参照。系统的身份可以由高维状态分布、关系拓扑和长期策略轨迹定义。相关表示方法可追溯到稀疏分布式记忆、全息降维表示和张量积表示（Kanerva, 1988；Plate, 1995；Smolensky, 1990；Kanerva, 2009），现代综述见 Kleyko et al.（2023）。

高维超向量采用三种基本算子：

\[
\operatorname{bind}(a,b)=a\otimes b
\]

\[
\operatorname{bundle}(a_1,\ldots,a_n)
 =\operatorname{sign}_\star(a_1+\cdots+a_n)
\]

\[
\operatorname{permute}(a)=\rho(a)
\]

其中 \(\otimes\) 可以采用随机置换后的逐元素乘法，\(\rho\) 是保持范数的维度置换，\(\operatorname{bundle}\) 用于叠加多个角色或关系。\(\operatorname{sign}_\star\) 表示**带随机破平的符号函数**：当分量和为 0 时（双极向量在偶数项叠加时必然出现），从 \(\{-1,+1\}\) 中按固定种子随机取值，以保持结果的双极性。实现时该种子必须与 `projection_seed` 一同记录，否则 replay 不可复现。

一个事件的超向量编码可以写成：

\[
h_t=\operatorname{bundle}(
\operatorname{bind}(H_{content},C_{content}),
\operatorname{bind}(H_{actor},C_{actor}),
\operatorname{bind}(H_{time},C_{time}),
\operatorname{bind}(H_{relation},C_{relation}),
\operatorname{bind}(H_{affect},C_{affect}),
\operatorname{bind}(H_{tool},C_{tool})
)
\]

这使内容、时间、关系和调制场以分布式方式共存；解析时使用与角色超向量的相似度或近似逆绑定，而不是依赖单一人工语义轴。

本文采用三层表示：

1. **连续潜空间**：VAE 的 \(z_t\) 负责压缩、生成、重建和自由能计算。
2. **高维超向量空间**：\(h_t\) 负责组合、绑定、检索和关系解析。
3. **动态拓扑空间**：\(G_t\) 负责节点激活、竞争、扩散和局部重连。

保留连续层（而非只用量化超向量）是有意选择，并与 §2.6 引述的纯脉冲原型质量瓶颈一致：离散化提供组合性与稀疏性，连续层承担精度。

VAE 与超向量之间使用可记录的桥接算子：

\[
z_t\sim q_\phi(z\mid o_t,m_t),\qquad h_t=Q_D(Rz_t+\epsilon_t)
\]

其中 \(R\) 是固定或缓慢更新的随机投影，\(Q_D\) 将向量量化到 \(D\) 维双极或稀疏超向量，\(\epsilon_t\) 是受控扰动。表示层的训练目标扩展为：

\[
L_{rep}=L_{recon}
 +\beta D_{KL}(q_\phi(z\mid o,m)\|p(z))
 +\lambda_bL_{bind}
 +\lambda_gL_{topo}
\]

原始 trace 仍保存为不可变归档，VAE 潜变量、超向量索引和活跃拓扑都属于可重建的派生层。这样可以替换编码器、重算索引和比较不同拓扑，而不破坏历史事件。

**高维性的可观测指标**。有效维度采用参与比：

\[
D_{effective}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}
\]

其中 \(\lambda_i\) 是**连续潜变量 \(z_t\)** 在长度为 \(W\) 的滑动窗口上、经中心化后的样本协方差矩阵特征值；\(W\) 与窗口起点必须随指标一同记录。注意该指标**不适用于双极超向量 \(h_t\)**：随机双极向量的协方差谱近似平坦，\(D_{effective}\approx D\) 恒成立，报告该值不含信息。\(h_t\) 层应改用超向量稀疏度、绑定可逆性（逆绑定后与原角色向量的相似度）和容量-干扰曲线。

其余指标包括拓扑熵、活跃节点数、多时间尺度数量和跨模块信息增益。二维 PCA 只用于桌面端可视化，不代表系统的核心状态维度。

## 3. 系统模型

### 3.1 冻结模型与外部控制器

基础模型：

\[
y_t \sim p_\theta(y_t\mid x_t,c_t,r_t)
\]

其中 \(r_t\) 表示采样和推理路由配置。外部控制器为：

\[
u_t=\pi_\psi(o_t,s_t,G_t)
\]

控制动作 \(u_t\) 包括上下文构造、模型路由、采样参数、工具调用、候选分支、算力档位和最终输出选择。

### 3.2 状态向量

定义系统状态：

\[
s_t=(q_t,e_t,r_t,m_t,g_t,G_t,E_t,\ell_t,v_t)
\]

其中：

- \(q_t\)：激活痕迹或潜状态的近似后验；
- \(e_t\)：情绪状态，包括 valence、arousal、tension 和 uncertainty；
- \(r_t\)：关系状态；
- \(m_t\)：organism 状态，如 energy、curiosity 和 connection need；
- \(g_t\)：目标和主动性状态；
- \(G_t\)：动态模块拓扑；
- \(E_t\)：**可用**能量和剩余调用预算（归一化到 \([0,1]\)）；
- \(\ell_t\)：液态连续控制状态（§5.1；v0.1 中记作 \(h_t\)，与超向量碰撞，已重命名）；
- \(v_t\)：脉冲门的膜电位集合（§5.2、§6.2）。

### 3.3 统一目标

按 §2.2 的恒等式，把熵项拆为变分内生部分与额外部分。外部控制器的单步目标定义为：

\[
J_t
 = -F_t
 + \lambda_H^{ex} H(q_t)
 + \lambda_I I_t
 + \lambda_E V(e_t)
 - \lambda_C C(u_t)
 - \lambda_D D(\hat s_{t+1},s_t)
\]

各项含义：

- \(-F_t\)：解释观测并保持预测一致性；其中已内含系数为 1 的熵项；
- \(\lambda_H^{ex} H(q_t)\)：**超出**变分项的额外熵权重。等效总熵系数为 \(1+\lambda_H^{ex}\)；\(\lambda_H^{ex}=0\) 时退化为纯变分自由能最小化，\(\lambda_H^{ex}<0\) 允许比变分解更集中的后验；
- \(I_t\)：鼓励能带来新信息的行动；
- \(V(e_t)\)：情绪状态的目标价值；
- \(C(u_t)\)：工具、延迟和外部调用成本，**包含控制核自身的采样与评分开销**（见 §4.3）；
- \(D\)：状态漂移和结构突变惩罚。注意其参数是**预测**下一状态 \(\hat s_{t+1}\) 而非真实 \(s_{t+1}\)，理由见 §4.1。

长期目标：

\[
J=\mathbb{E}\left[\sum_{t=0}^{T}\gamma^tJ_t\right]
\]

### 3.4 可达集上界

由于 \(\theta\) 冻结，外部控制器只能在 \(p_\theta\) 的支撑集内重新加权与重组。若记控制器可施加的上下文、路由与采样配置集合为 \(\mathcal{U}\)，则系统可实现的输出分布族被限制在

\[
\mathcal{P}_\theta=\{p_\theta(\cdot\mid x,c,r):(c,r)\in\mathcal{U}\}
\]

的凸组合闭包内。这给出一个明确的能力上界：**外部控制可以改变模式的相对权重、组合方式与触发条件，不能创造 \(p_\theta\) 不具备的模式。** 由此可导出一条可检验的预测：随着候选数与温度上升，输出多样性的增长应先于一致性的下降而饱和；若观测到多样性随温度单调线性上升，则说明测得的"多样性"主要是解码噪声而非语义分支，相应指标无效。§10 将此作为 H1 的前置有效性检查。

## 4. 评分场与黑盒元梯度

### 4.1 候选轨迹与副作用隔离

**v0.1 的问题**。v0.1 把候选轨迹定义为 \(\tau_k=(o_t,c_{t,k},y_{t,k},a_{t,k},s_{t+1,k})\)，其中 \(a_{t,k}\) 为工具或主动行为、\(s_{t+1,k}\) 为执行后状态。按字面实现，为给 \(K\) 个候选打分必须先执行全部 \(K\) 个候选的动作，再丢弃 \(K-1\) 个。ZERO/HDSC 的宿主工具内核包含 `write_file`、`powershell`，并支持 `elevated=true` 经 `Start-Process -Verb RunAs` 提权，因此该实现会产生不可回滚的宿主副作用。这是设计缺陷而非实现细节。

**v0.2 的约束**。把工具按副作用分为三类：

| 类别 | 定义 | 候选阶段 |
|------|------|----------|
| `pure` | 无外部副作用（检索、相似度、纯计算） | 允许真实执行 |
| `read` | 只读外部状态（`read_file`、只读 win32 观测、外部真值查询） | 允许真实执行，计入成本与配额 |
| `effecting` | 可改变宿主或外部状态（`write_file`、`powershell`、提权、发送消息、写记忆） | **禁止**真实执行，只允许 dry-run |

候选轨迹相应重定义为：

\[
\tau_k=(o_t,c_{t,k},y_{t,k},\hat a_{t,k},\hat s_{t+1,k})
\]

其中 \(\hat a_{t,k}\) 是**提议动作**（结构化调用意图 + dry-run 结果），\(\hat s_{t+1,k}\) 是由转移预测器给出的**预测**下一状态。所有评分场只读 \(\hat\cdot\)。仅当输出门（§7.3 第 8 步）选定分支 \(k^\star\) 后，才执行 \(a_{t,k^\star}\) 并观测真实 \(s_{t+1}\)。

对 `effecting` 工具，dry-run 的最低要求是：解析并校验参数、检查路径与权限、返回将被影响的目标清单与预计变更规模，但不落盘、不发送、不提权。无法提供 dry-run 的工具在候选阶段一律按"不可评分"处理，其分支的 \(Q_{tool}\) 记为缺失而非失败（与 §4.2.1 一致）。

该约束作为硬不变量进入 \(I_{inv}\)（§4.2.3）：任一被丢弃分支产生了 `effecting` 类真实调用，则 \(I_{inv}=0\) 并触发审计告警。

### 4.2 三类评分场

系统维护三个外部评分器。所有原始指标先通过 `clip01(x)=min(1,max(0,x))` 归一到 `[0,1]`，再用固定权重或验证集调优权重合成；每个评分场的权重之和为 1。

#### 4.2.1 认知评分场 \(S_c\)

认知场衡量候选结果是否完成任务、保持事实一致、正确使用工具，以及**是否给出了足够具体的可验证内容**：

\[
S_c(\tau_k)=
0.28Q_{task}+
0.22Q_{fact}+
0.16Q_{coh}+
0.14Q_{tool}+
0.10(1-Q_{hall})+
0.10Q_{spec}
\]

可计算定义：

- \(Q_{task}\)：结构化任务评估器的通过比例；无标注任务使用规则检查、目标字段覆盖率和独立评审器平均分。
- \(Q_{fact}\)：设 \(n_{sup}\)、\(n_{con}\)、\(n_{unk}\) 分别为被激活 trace/外部证据支持、与之矛盾、以及无法判定的声明数，则

  \[
  Q_{fact}=\frac{n_{sup}+\zeta n_{unk}}{n_{sup}+n_{con}+n_{unk}},\qquad \zeta\in[0,0.5]
  \]

  当 \(n_{sup}+n_{con}+n_{unk}=0\) 时取先验值 \(Q_{fact}^{0}=0.5\)。**v0.1 勘误**：v0.1 规定"未知声明不计入分母"，导致只输出不可验证表述时分母为空、指标平凡取 1，可被"说废话"策略刷分；此处将未知声明以部分权重计入分母，并配合 \(Q_{spec}\) 抑制该策略。
- \(Q_{coh}\)：实体、时间、关系和当前目标与 snapshot 的一致比例。
- \(Q_{tool}\)：工具调用成功率；**无工具调用时记为缺失并从加权中剔除后重新归一**，不把工具缺失误判为失败，也不误判为满分。
- \(Q_{hall}\)：未得到证据支持的可验证声明比例。
- \(Q_{spec}\)：\(\operatorname{clip01}(n_{ver}/n^\ast_{ver})\)，其中 \(n_{ver}=n_{sup}+n_{con}\) 为可验证声明数，\(n^\ast_{ver}\) 是按任务类型预设的期望可验证声明数。该项使"保持模糊"不再是免费策略。

当模型提供 token log-probability 时，额外记录归一化负对数似然；模型不返回 log-probability 时，使用上述黑盒代理指标。

**复用说明**：`reflective_learning_service.py` 中已实现的 critic 使用来源覆盖度、来源可信度、一致性、可伪证性四项加权，是 \(S_c\) 在离线巩固路径上的特例。两者应共用同一份指标实现，避免在线与离线打分口径分裂。

#### 4.2.2 情绪评分场 \(S_e\)

情绪场描述情绪状态是否与当前情境匹配、变化是否连续、探索强度是否处于可用区间，以及**目标情绪本身是否仍与外部锚点校准**。令 appraisal 产生目标状态 \(e_t^*\)，输出解析器产生实际状态 \(\hat e_{t+1}\)，状态向量包含 valence、arousal、tension、uncertainty：

\[
F_{affect}=\exp(-\|\hat e_{t+1}-e_t^*\|_{W_e}^2)
\]

\[
C_{affect}=\exp(-\|\hat e_{t+1}-e_t\|_{W_c}^2)
\]

\[
H_{band}=1-\operatorname{clip01}
\left(\frac{|H(q_t)-H^*|}{H_{max}-H_{min}}\right)
\]

\[
S_e(\tau_k)=
0.35F_{affect}+
0.20C_{affect}+
0.20H_{band}+
0.10N_{controlled}+
0.15A_{cal}
\]

其中 \(N_{controlled}\) 是候选分支带来的新颖性，使用与最近 \(n\) 条输出的平均 embedding 距离计算，并截断到 `[0,1]`。

\(A_{cal}\) 是**校准锚定项**，用于处理下述循环依赖。\(F_{affect}\) 以 appraisal 产出的 \(e_t^*\) 为目标，而 appraisal 本身由被控的冻结模型产生，因此控制器存在优化"与被控对象的自我判断一致"的自指风险：内部一致性会被奖励，外部效度不会。\(A_{cal}\) 定义为当前 appraisal 模块在留出标注集上的一致度，标注集以 `tests/fixtures/appraisal_standard_events.v1.json` 的标准事件为基础扩展并独立标注；当 \(A_{cal}\) 低于阈值时，\(F_{affect}\) 的权重按比例下调，直至 appraisal 重新校准。该评分描述控制目标，不把情绪评分当作人的主观真值。

#### 4.2.3 系统评分场 \(S_s\)

系统场衡量资源、审计、不变量和结构漂移：

\[
S_s(\tau_k)=
0.30I_{inv}+
0.20A_{audit}+
0.20R_{resource}+
0.15T_{topology}+
0.15(1-D_{state})
\]

可计算定义：

- \(I_{inv}\)：所有硬不变量通过取 1，否则取 0；包括 trace 不可变、父节点存在、ID 唯一、预算约束，以及 **§4.1 的副作用隔离约束**（被丢弃分支未产生 `effecting` 类真实调用）。
- \(A_{audit}\)：必需审计字段的存在比例，包括 correlation id、模型、工具、输入输出摘要、时间和版本。
- \(R_{resource}=\exp(-cost_t/budget_t)\)：token、延迟、工具次数和音频成本的归一化效率，**其中 \(cost_t\) 含控制核自身的候选采样与评分开销**。
- \(T_{topology}=\exp(-\Delta edges_t/cap_t)\)：本轮拓扑变更量相对于变更上限的稳定度。
- \(D_{state}=\operatorname{clip01}(D_{KL}(q_{t+1}\|q_t)/\delta_q)\)：状态漂移超过阈值时趋近 1。

三类评分都保存为事件元数据，形成候选级别的可回放数据集。该数据集是 §4.3 在线估计量的直接来源。

### 4.3 元梯度估计

**符号约定（v0.1 勘误）**。v0.1 同时使用了 \(\hat g_i\)（对分数 \(S_i\) 的有限差分，需上升）与 \(g_i=\nabla_\psi L_i\)（对损失的梯度，需下降），二者关系未定义，而 §6.3 的更新式取加号，字面即"对损失做梯度上升"。v0.2 统一约定：**全程对分数 \(S_i\) 做上升**，不再使用损失记号。即

\[
g_i \triangleq \widehat{\nabla_\psi \mathbb{E}[S_i]},\qquad i\in\{c,e,s\}
\]

所有更新式均为 \(\psi\leftarrow\psi+\alpha g\)。

**在线估计量：候选批 bandit**。控制器每回合本已采样 \(K\) 个候选并逐个评分（§4.2）。设分支选择由控制器的可微代理分布 \(\pi_\psi(k\mid o_t,s_t)\) 给出，则可直接用得分基线化的 score-function 估计量：

\[
g_i^{online}
 =\frac{1}{K}\sum_{k=1}^{K}
 \bigl(S_i(\tau_k)-\bar S_i\bigr)
 \nabla_\psi\log\pi_\psi(k\mid o_t,s_t),
 \qquad \bar S_i=\frac{1}{K}\sum_{k}S_i(\tau_k)
\]

该估计量**不需要任何额外 rollout**：它复用已经生成并已经打分的候选批，边际成本为零。

**离线估计量：SPSA 有限差分**。对无法通过 \(\pi_\psi\) 参数化的控制量（检索阈值、上下文排列规则、拓扑重连超参、时间常数映射），使用同时扰动随机逼近（Spall, 1992）：

\[
g_i^{offline}
 =\frac{1}{m}\sum_{j=1}^{m}
 \frac{S_i(\psi+\delta u_j)-S_i(\psi-\delta u_j)}{2\delta}u_j
\]

\(u_j\) 使用 Rademacher 向量或低差异序列。**该估计量只在离线巩固循环中对已归档的 trace 做 replay 时计算，不在在线回合内执行**。理由是成本：每个 \(\psi\pm\delta u_j\) 需要一次完整轨迹评分，三个评分场、\(m\) 个方向合计 \(6m\) 次评估；\(m=16\) 即近百次评估/更新步，在单用户付费 API 场景下会成为主导成本，且与 \(J_t\) 中的 \(C(u_t)\) 惩罚项自相矛盾（v0.1 未把元梯度自身的开销计入预算）。离线 replay 中这些评估针对历史数据，可批处理、可缓存、可中断。

候选结果、评分、随机种子与 \(\psi\) 版本必须全部写入 trace，保证离线重放的一致性。合成方向：

\[
g_i = \nu\, g_i^{online} + (1-\nu)\, g_i^{offline},\qquad \nu\in[0,1]
\]

其中 \(g_i^{offline}\) 取最近一次离线更新的结果并在其有效期内保持不变。

### 4.4 反对齐投影

为了减少三类目标的同向塌缩，先计算梯度 Gram 矩阵：

\[
K_{ij}=\frac{g_i^Tg_j}{\|g_i\|\|g_j\|+\epsilon}
\]

**与 PCGrad 的关系（定位澄清）**。PCGrad（Yu et al., 2020）在 \(K_{ij}<0\)（目标冲突）时投影掉冲突分量，目的是消解冲突。本文的触发条件**相反**：在 \(K_{ij}>0\)（目标一致）时削弱一致分量，目的是抑制三个评分场的同向塌缩，并保留旋转项用于持续探索。二者共享投影算子而目标相反，不应混读。

定义投影算子与一致分量：

\[
P_{g_j}(g_i)=\frac{g_i^{T}g_j}{\|g_j\|^{2}+\epsilon}\,g_j,
\qquad
c_i=\sum_{j\ne i}\lambda_{ij}\max(0,K_{ij})\,P_{g_j}(g_i)
\]

**方向反转问题（v0.1 勘误）**。v0.1 直接取 \(\tilde g_i=g_i-c_i\)。当三个目标完全一致（\(g_1=g_2=g_3=g\)，\(K_{ij}=1\)）时得 \(\tilde g_1=(1-\lambda_{12}-\lambda_{13})g\)：在三目标情形下只要 \(\lambda_{ij}\ge 0.5\) 便完全抵消，\(>0.5\) 则**方向反转**——恰在三个目标一致、最应当前进时倒退。

v0.2 改为带自适应步长的削弱，并给出方向性保证。取设计常数 \(\kappa_g\in(0,1)\)（建议 \(0.2\)）：

\[
\eta_i=\min\left(1,\ \frac{(1-\kappa_g)\|g_i\|^{2}}{\max(0,\,g_i^{T}c_i)+\epsilon}\right),
\qquad
\tilde g_i=g_i-\eta_i c_i
\]

**命题 1**。上式保证 \(\tilde g_i^{T}g_i\ \ge\ \kappa_g\|g_i\|^{2}\ >\ 0\)。

*证明*。分三种情形。

(i) \(g_i^{T}c_i\le 0\)：此时 \(\eta_i>0\) 而 \(-\eta_ig_i^{T}c_i\ge0\)，故 \(\tilde g_i^{T}g_i=\|g_i\|^{2}-\eta_ig_i^{T}c_i\ge\|g_i\|^{2}\ge\kappa_g\|g_i\|^{2}\)。

(ii) \(g_i^{T}c_i>0\) 且 \(\eta_i=1\)：由 \(\min\) 取到 1 知 \((1-\kappa_g)\|g_i\|^{2}\ge g_i^{T}c_i+\epsilon>g_i^{T}c_i\)，故 \(\tilde g_i^{T}g_i=\|g_i\|^{2}-g_i^{T}c_i>\kappa_g\|g_i\|^{2}\)。

(iii) \(g_i^{T}c_i>0\) 且 \(\eta_i<1\)：此时 \(\eta_i=\dfrac{(1-\kappa_g)\|g_i\|^{2}}{g_i^{T}c_i+\epsilon}\)，故

\[
\eta_ig_i^{T}c_i
=(1-\kappa_g)\|g_i\|^{2}\cdot\frac{g_i^{T}c_i}{g_i^{T}c_i+\epsilon}
\le(1-\kappa_g)\|g_i\|^{2}
\]

从而 \(\tilde g_i^{T}g_i=\|g_i\|^{2}-\eta_ig_i^{T}c_i\ge\kappa_g\|g_i\|^{2}\)。

三种情形均有 \(\tilde g_i^{T}g_i\ge\kappa_g\|g_i\|^{2}\)，且 \(\kappa_g>0\)、\(g_i\ne0\) 时严格为正。∎

即修正后的方向**永不反转任何单个目标**，同时保留至多 \((1-\kappa_g)\) 比例的一致分量削弱能力。命题 1 是逐目标的；合成方向 \(\sum_i w_i\tilde g_i\) 在 \(\bar g\) 上的联合下界还依赖交叉项 \(\tilde g_i^Tg_j\ (i\ne j)\) 的假设，本文不对其作声明。

### 4.5 旋转耦合与受控扰动

最终控制方向：

\[
\bar g=\sum_i w_ig_i,\qquad
g_t=\sum_iw_i\tilde g_i+\Omega_t\bar g+\xi_t
\]

**旋转项的规格（v0.1 补全）**。v0.1 只说明 \(\Omega_t\) 为反对称矩阵，未给定幅度，也未与状态耦合——而 OU 扰动 \(\sigma_t\) 是受状态调制的，两套探索机制口径不一致。v0.2 规定：

\[
\Omega_t=\omega_t\,\frac{A_t-A_t^{T}}{\|A_t-A_t^{T}\|_2+\epsilon},
\qquad
\omega_t=\kappa_\Omega\cdot\operatorname{clip01}\!\left(\frac{H^{*}-H(q_t)}{H_{max}-H_{min}}\right),
\qquad \kappa_\Omega<1
\]

其中 \(A_t\) 由固定种子生成并缓慢更新。由此 \(\|\Omega_t\bar g\|\le\kappa_\Omega\|\bar g\|\)，且旋转强度与 \(\sigma_t\) 共享同一个熵带误差驱动。

**性质**。由 \(\Omega_t\) 反对称得 \(\bar g^{T}\Omega_t\bar g=0\)，故

\[
g_t^{T}\bar g=\Bigl(\sum_iw_i\tilde g_i\Bigr)^{T}\bar g+\xi_t^{T}\bar g
\]

即**旋转项对 \(\bar g\) 方向的投影恰为零**：它提供正交探索而不消耗任何利用分量。又因 OU 扰动零均值，\(\mathbb{E}[\xi_t^T\bar g]=0\)，故 \(\mathbb{E}[g_t^T\bar g]=(\sum_iw_i\tilde g_i)^T\bar g\)，探索项在期望意义上不偏置更新方向。

**受控扰动**。采用 Ornstein-Uhlenbeck 型：

\[
\xi_{t+1}
 =\rho\xi_t
 +\sigma_t\sqrt{1-\rho^2}\,\epsilon_t,
 \qquad \epsilon_t\sim\mathcal N(0,I)
\]

该参数化使 \(\sigma_t\) 恒定时稳态方差恰为 \(\sigma_t^{2}\)；\(\sigma_t\) 时变时该解释为准静态近似，切换过快时应记录瞬态。扰动强度：

\[
\sigma_t
 =\operatorname{clip}
 \bigl(\sigma_0
 +a\,H^{err}_t
 +b\,arousal_t
 +c\,uncertainty_t
 +d\,E_t,
 \sigma_{min},\sigma_{max}\bigr)
\]

\[
H^{err}_t=\operatorname{clip01}\!\left(\frac{H^{*}-H(q_t)}{H_{max}-H_{min}}\right)
\]

**两处修正（v0.1 勘误）**。第一，能量项符号由 \(-dE_t\) 改为 \(+dE_t\)：§2.3 的设计意图是"不确定性较高且能量允许时保留多种解释"，而 \(E_t\) 按 §3.2 定义为**可用**能量，故 \(-dE_t\) 会导致能量充足时探索反而减少，与意图相反。第二，熵项由开环的 \(aH_t\) 改为闭环的 \(aH^{err}_t\)：只在熵**低于**目标带时增加扰动，避免熵已过高时继续加噪。

扰动作用于上下文排列、候选分支、工具顺序、检索阈值和拓扑重连概率，而非直接修改基础模型参数。

## 5. 液态状态与脉冲门控

本节把 v0.1 的 §5（连续时间常数）与 §6.2（无记忆重连门）统一到同一套 LIF 形式下。二者本质相同：LIF 单元即带阈值与复位的液态时间常数单元。

### 5.1 连续路径

令内部连续控制状态为 \(\ell_t\)（v0.1 记作 \(h_t\)，与 §2.7 的超向量 \(h_t\) 碰撞，已重命名）：

\[
\tau_t\dot \ell_t
 =-\ell_t+
 f(W_\ell \ell_t+W_oo_t+W_ee_t+b)
\]

动态时间常数：

\[
\tau_t
 =\tau_{min}
 +(\tau_{max}-\tau_{min})
 \sigma(a^T e_t+b^T m_t+c^T q_t)
\]

离散实现：

\[
\ell_{t+1}
 =(1-\alpha_t)\ell_t+
 \alpha_t f(W_\ell \ell_t+W_oo_t+W_ee_t+b),
 \qquad
\alpha_t=\operatorname{clip}(\Delta t/\tau_t,0,1)
\]

**收敛条件**。要求 \(f\) 有界（如 \(\tanh\)）且 \(\|W_\ell\|_2\) 有界，否则前向 Euler 离散不保证有界。

### 5.2 时间常数与真实调度周期的一致性

**退化问题（v0.1 勘误）**。\(\alpha_t=\operatorname{clip}(\Delta t/\tau_t,0,1)\) 在 \(\Delta t\ge\tau_t\) 时取 1，此时 \(\ell_{t+1}=f(\cdot)\)，状态**完全无记忆**，"液态"动力学退化为瞬时映射。ZERO/HDSC 的 `inner.heartbeat` 实际周期为 60 秒；若 \(\tau_t\) 按秒量级取值，则 \(\alpha_t\equiv 1\)，§5.1 全部失效。

**约束**。\(\tau\) 必须与真实调度周期同单位，并满足

\[
\tau_{min}\ \ge\ \eta_\tau\,\Delta t,\qquad \eta_\tau\ \ge\ 5
\]

即在 60 秒心跳下 \(\tau_{min}\ge 5\) 分钟、\(\tau_{max}\) 取小时量级，对应"等待—离线—静息"这几个本就以分钟到小时计的模式。实现时应把 \(\Delta t\) 取为**实际经过时间**（由事件时间戳差得到）而非标称周期，因为心跳可能被前台回合推迟；并在 \(\alpha_t\) 触及上界 1 时记录一次告警，作为参数配置错误的运行时探针。

### 5.3 LIF 脉冲门

对需要"持续驱动才动作、单次噪声不动作"的决策，引入膜电位 \(v\) 与二值脉冲 \(z\)。给定归一化驱动 \(s^t\in[0,1]\)、泄漏系数 \(\beta_v\in(0,1)\)、阈值 \(\vartheta>0\)：

\[
z^{t}=\mathbb{1}\bigl[v^{t}\ge\vartheta\bigr],
\qquad
v^{t+1}=\beta_v v^{t}+s^{t}-\vartheta z^{t}
\]

（减法式软复位，保留越阈的余量，避免强驱动下丢失信息。）

**命题 2（选择性）**。设初值 \(v^0=0\)。

1. *瞬态免疫*：单步驱动 \(s\) 后驱动归零，发放当且仅当 \(s\ge\vartheta\)；
2. *持续敏感*：恒定驱动 \(s\) 下 \(v\) 收敛到 \(s/(1-\beta_v)\)，故最终发放当且仅当 \(s\ge\vartheta(1-\beta_v)\)。

*证明*。(1) 直接代入。(2) \(v^{t+1}=\beta_v v^t+s\) 的不动点为 \(v^\ast=s/(1-\beta_v)\)，且该递推对 \(v^0=0\) 单调递增收敛；发放条件 \(v^\ast\ge\vartheta\) 即 \(s\ge\vartheta(1-\beta_v)\)。∎

因此**选择性增益**为 \(1/(1-\beta_v)\)：取 \(\beta_v=0.9\) 时，持续驱动的发放门限比瞬态低一个数量级。这给出一个可直接标定的设计旋钮——把 \(\vartheta(1-\beta_v)\) 设为"值得响应的最小持续驱动"，把 \(\vartheta\) 设为"值得立即响应的最小瞬时驱动"。

这一构造替代瞬时 sigmoid 阈值的依据见 §2.6：SymbolicLight V1 报告在同等稀疏度下以确定性 top-k 掩码替换 LIF 会造成更大退化，说明起作用的是时间积分而非稀疏本身。本文据此把时间积分作为门控的必要成分，而非可选项。该依据属于跨设定的机制迁移，其在本文设定下的有效性由 §10 的条件 C8 单独检验。

## 6. 动态混合拓扑

### 6.1 图表示

\[
G_t=(V_t,A_t,X_t)
\]

- \(V_t\)：trace 社区、工具、策略和目标节点；
- \(A_t\)：有向或带类型的关系矩阵；
- \(X_t\)：节点状态和统计量。

边类型包括 semantic、affective、causal、competitive 和 temporal。

### 6.2 累积器重连门

**v0.1 的形式**为无记忆瞬时门：

\[
p_{ij}^{rewire}
 =\sigma(\alpha\,surprise_{ij}
 +\beta\,H_{ij}
 +\gamma\,novelty_{ij}
 -\delta\,cost_{ij})
\]

该形式属于 §2.6 中被消融证据判为较弱的一类（同等稀疏度下的确定性阈值），且不具备抗噪性：单回合的异常 surprise 可直接触发结构变更，与 §1.3 声称的抗漂移目标矛盾。

**v0.2 改为按 §5.3 的 LIF 累积器门**。对每条候选边维护膜电位 \(v_{ij}\)，驱动为

\[
s_{ij}^{t}=\operatorname{clip01}\bigl(
\alpha\,surprise_{ij}^{t}
+\beta\,H_{ij}^{t}
+\gamma\,novelty_{ij}^{t}
-\delta\,cost_{ij}^{t}\bigr)
\]

\[
z_{ij}^{t}=\mathbb{1}\bigl[v_{ij}^{t}\ge\vartheta_G\bigr],
\qquad
v_{ij}^{t+1}=\beta_G v_{ij}^{t}+s_{ij}^{t}-\vartheta_G z_{ij}^{t}
\]

重连当且仅当 \(z_{ij}^{t}=1\)。由命题 2，持续的中等惊奇会累积成一次重连，孤立的噪声尖峰不会；\(\vartheta_G(1-\beta_G)\) 直接就是"值得改结构的最小持续证据强度"。

重连策略：

1. 保留高证据核心边；
2. 对低活跃边进行衰减（膜电位同样泄漏，长期无驱动的候选边自然退出）；
3. 只在发放的边上创建/删除连接；
4. 使用社区检测控制拓扑规模；
5. 将每次拓扑变更连同发放时的 \(v_{ij}\)、\(s_{ij}\) 轨迹写入可审计日志。

第 5 条使"为何在此刻改结构"可回溯到一段驱动历史，而不只是一个瞬时随机数——这是相对 v0.1 在可解释性上的直接改善，也部分回应 §11 中"拓扑持续变化提高复现难度"的局限。

### 6.3 快慢变量交叉

\[
\psi_f^{t+1}
 =\psi_f^t+\alpha_f g_t+\mu_t\epsilon_t
\]

\[
\psi_s^{t+1}
 = (1-\alpha_s)\psi_s^t+
 \alpha_s\operatorname{Crossover}(\psi_s^t,\psi_f^{t+1})
\]

（按 §4.3 的约定，\(g_t\) 是对分数的上升方向，故取加号。）

慢变量更新需要满足证据覆盖、连续性和结果质量门槛。该设计结合持续学习中的经验回放和梯度情景记忆思想（Kirkpatrick et al., 2017；Lopez-Paz & Ranzato, 2017）；负反馈触发回滚，沉默只作为未知结果处理。慢变量的更新时机同样由一个 LIF 门控制，驱动为通过门槛的提案质量，从而使"多次中等质量的一致证据"而非"单次高分"驱动长期变更。

## 7. 双路径算力调度

v0.1 的 §7 对每个回合执行同一套完整流程：多分支采样、三场评分、元梯度、拓扑手术。其边际成本恒定且高，与 \(J_t\) 中的 \(C(u_t)\) 项冲突。v0.2 按 §2.6 的双路径思路，将流程分为常开的低成本路径与脉冲门控的高成本路径。

### 7.1 快路径（常开）

每回合执行，成本与无控制基线相当：

- 更新液态状态 \(\ell_t\)（§5.1）；
- 缓存/增量检索，单一上下文构造；
- 单候选生成（\(K=1\)）；
- 计算 \(F_t\)、\(H(q_t)\)、\(e_t\)、\(E_t\) 与轻量 \(S_c\) 子集；
- 更新全局控制膜电位（§7.2）；
- 不做多分支采样、不做元梯度、不改拓扑。

### 7.2 慢路径（脉冲门控）

全局控制脉冲以自由能预测误差为驱动。设 \(\bar F_t\) 为 \(F\) 的滑动基线、\(\varsigma_F\) 为其滑动尺度：

\[
s^{ctrl}_t=\operatorname{clip01}\!\left(\frac{F_t-\bar F_t}{\varsigma_F}\right)
\]

\[
z^{ctrl}_t=\mathbb{1}\bigl[v^{ctrl}_t\ge\vartheta_c\bigr],
\qquad
v^{ctrl}_{t+1}=\beta_c v^{ctrl}_t+s^{ctrl}_t-\vartheta_c z^{ctrl}_t
\]

当 \(z^{ctrl}_t=1\) 时执行慢路径：多分支采样（\(K>1\)）、三场完整评分、critic 与输出门、反对齐与旋转合成、快变量更新、拓扑重连评估。

除自由能脉冲外，以下情形强制进入慢路径，避免门控掩盖需要审慎处理的场合：任一 `effecting` 类工具被提议；关系状态或目标状态发生离散跃迁；\(I_{inv}\) 上一回合失败；距上次慢路径超过 \(T_{max}\) 个回合（保证下界频率）。

**预算耦合**。\(\vartheta_c\) 随剩余预算 \(E_t\) 反向调整：预算充足时降低阈值（更常进入慢路径），预算紧张时提高阈值。这使 §3.3 的 \(C(u_t)\) 项通过门控阈值真正闭环，而非仅作为事后惩罚。

**元梯度位置**。\(g^{online}\) 在慢路径内计算（复用已有候选批，边际成本为零）；\(g^{offline}\) 只在离线巩固循环中计算（§4.3）。在线回合内不做任何 \(\pm\delta\) rollout。

### 7.3 算法

```text
Algorithm EVF-MACK-v2(o_t)

--- 快路径（每回合） ---
 1. 读取 append-only trace、最近状态与活动拓扑 G_t
 2. 更新液态状态 ℓ_t；计算 q_t, H_t, F_t, e_t, E_t
 3. 更新全局控制膜电位 v^ctrl；判定 z^ctrl
 4. if z^ctrl == 0 and 无强制触发条件:
       单候选生成 → 轻量校验 → 输出 → 记录 trace → return

--- 慢路径（脉冲门控） ---
 5. 构造多组上下文与采样路由；生成候选集合 Y_t（K > 1）
 6. 对每个候选：仅 pure/read 工具真实执行；effecting 工具只 dry-run
    得到提议动作 â_{t,k} 与预测状态 ŝ_{t+1,k}
 7. 计算三个评分场 S_c, S_e, S_s（只读 â, ŝ）
 8. critic 评估与动作门控；选择分支 k*
 9. 执行 a_{t,k*} 的真实动作；观测真实 s_{t+1}
10. 校验副作用隔离不变量（被弃分支无 effecting 真实调用）
11. 计算 g^online；反对齐投影（命题 1）；旋转耦合与 OU 扰动
12. 更新快变量 ψ_f；满足证据门槛时经 LIF 门交叉更新慢变量 ψ_s
13. 更新边级膜电位 v_ij；对发放边执行重连并写审计日志
14. 执行能量、熵带、KL、拓扑规模与漂移检查
15. 记录 trace、评分、拓扑变化、种子与 ψ 版本
16. 输出回复与新的 DashboardSnapshot

--- 离线巩固（独立循环） ---
17. 对归档 trace 做 replay；计算 g^offline（SPSA）
18. 社区检测、潜变量/超向量重算、低证据变更的显式遗忘
```

## 8. ZERO/HDSC 集成方案

### 8.1 在线回合

在 [interactive.py](../src/ssa/runtime/interactive.py) 的单回合流程中加入外部控制器：

```text
user.message
  → trace activation
  → 液态状态更新 + 控制脉冲判定        [快路径]
  → (z^ctrl == 0) ? 单候选直出 : ↓
  → EVF context planner                [慢路径]
  → LLM/tool rounds（effecting 仅 dry-run）
  → candidate critic
  → output gate → 执行选中分支的真实动作
  → assistant.message
```

### 8.2 在线状态

在 snapshot 中增加：

```json
{
  "control_kernel": {
    "path": "fast",
    "free_energy": 0.0,
    "free_energy_baseline": 0.0,
    "control_membrane": 0.0,
    "control_spike": false,
    "forced_slow_reason": null,
    "posterior_entropy": 0.0,
    "entropy_band_error": 0.0,
    "perturbation_scale": 0.0,
    "rotation_scale": 0.0,
    "topology_entropy": 0.0,
    "rewire_spikes": 0,
    "candidate_count": 1,
    "side_effect_isolation_ok": true,
    "fast_policy_version": "...",
    "slow_policy_version": "...",
    "last_gate": "accepted"
  }
}
```

### 8.3 离线巩固

巩固循环位于总仓 [zero](https://github.com/sikehuang88/zero) 的 `巩固循环/` 目录中（该路径在发布前需与总仓实际结构核对），可增加：

- 候选策略轨迹的 replay 与 \(g^{offline}\) 的 SPSA 估计；
- 拓扑变化的社区统计与膜电位轨迹归档；
- 情绪状态与结果质量的关联；
- 快慢策略版本的差异评估；
- 低证据变更的显式遗忘。

### 8.4 高维表示层落点

建议新增 `ssa/hdsc/hypervector.py` 和 `ssa/services/latent_bridge.py`：

- `hypervector.py`：绑定、叠加（含随机破平）、置换、相似度、稀疏度和随机种子管理；
- `latent_bridge.py`：连续 VAE 潜变量与高维超向量之间的投影、量化和版本记录；
- `TraceSpaceService`：同时维护连续 embedding、超向量索引和关系边；
- `active_space.py`：从超向量共激活和向量相似度中构造动态活动图；
- `巩固循环`：对潜变量和超向量进行离线 replay、社区检测和重算。

所有派生表示需要记录 `encoder_id`、`embedding_dim`、`hypervector_dim`、`projection_seed`、`tie_break_seed` 和 `schema_version`，确保编码器升级后可以区分旧索引并执行迁移。

### 8.5 复用已有构件

新增代码应挂接而非重写下列已存在的实现：

| 本文章节 | 已有实现 | 关系 |
|----------|----------|------|
| §4.2.1 \(S_c\) | `services/reflective_learning_service.py` 的 critic | 共用指标实现；离线口径是在线的特例 |
| §4.2.2 \(A_{cal}\) | `tests/fixtures/appraisal_standard_events.v1.json` | 扩展为独立标注的留出校准集 |
| §9 稳定性 | `hdsc/active_space.py` 的 `_small_gain_certificate` | 直接承载；本文只补控制回路侧的缺口 |
| §6 拓扑 | `hdsc/active_space.py` 的容量、null reservoir、hysteresis | 累积器门作为其候选提案的前置过滤 |
| §7 快慢路径 | `runtime/lifecycle.py`、`runtime/autonomous.py` | 控制脉冲作为既有周期作业的触发条件 |

## 9. 稳定性：已实现的证书与待补的边界

**表述修正（v0.1）**。v0.1 的 §9 以"建议满足"列出若干不等式，既非定理也未说明由谁强制。v0.2 明确区分三类陈述：**已实现并由测试覆盖的证书**、**运行时强制的不变量**、**尚未证明的开放命题**。

### 9.1 已实现的证书（H2 活跃空间）

仓库中 `src/ssa/hdsc/active_space.py` 的 `_small_gain_certificate` 已对活跃空间的 replay 派生测度输出下列量，并由 `tests/property/test_hdsc_h2_active_space.py` 的性质测试覆盖：增益矩阵与谱半径、small-gain 余量、认证半径（含提案与容量两个来源）、局部记忆收缩界、语义跳跃界、质量守恒残差与最小质量、提案/容量边界间隙、聚类分配余量，以及 `CertificateStatus` 与失败原因。该证书是条件性的：它在给定经验 Lipschitz 增益与语义分区固定的前提下成立，且当前为 shadow-only，对提示词无影响。

本文不重述该证书，而是把它作为控制核状态更新的既有边界：\(G_t\) 的活跃部分沿用其容量、null reservoir、hysteresis 与重复抑制，§6.2 的累积器门只作为其候选提案的前置过滤器，因此不放宽既有界。

### 9.2 运行时强制的不变量

下列约束由控制核在每回合投影/截断/拒绝实现，属于工程不变量而非定理，违反时记审计事件并回滚：

\[
\|\Delta\psi_t\|\le\delta_\psi,\qquad
H_{min}\le H(q_t)\le H_{max},\qquad
D_{KL}(q_{t+1}\|q_t)\le\delta_q,\qquad
E_{used,t}\le E_{budget,t}
\]

以及活跃拓扑的最大度数、节点数与重连速率上限，和 §4.1 的副作用隔离不变量。基础归档保持 append-only，快速状态与策略版本单独保存，便于回放和比较。

### 9.3 已获得的方向性保证

- **命题 1**（§4.4）：反对齐投影保证 \(\tilde g_i^Tg_i\ge\kappa_g\|g_i\|^2>0\)，即不反转任何单个目标方向。
- **旋转正交性**（§4.5）：\(\bar g^T\Omega_t\bar g=0\)，旋转探索不消耗利用分量；\(\|\Omega_t\bar g\|\le\kappa_\Omega\|\bar g\|\)。
- **命题 2**（§5.3）：LIF 门的瞬态门限 \(\vartheta\) 与持续门限 \(\vartheta(1-\beta_v)\) 相差 \(1/(1-\beta_v)\) 倍，给出可标定的抗噪选择性。

### 9.4 开放命题

以下尚未证明，列为待验证项而非结论：

1. 合成方向 \(\sum_iw_i\tilde g_i\) 在 \(\bar g\) 上的联合下界（需对 \(w_i\) 与交叉项 \(K_{ij}\) 作额外假设）；
2. 快慢双时间尺度更新的联合收敛性（现有分析假设慢变量准静态，实际由 LIF 门离散触发，不满足标准双时标条件）；
3. §9.1 证书在控制核闭环接入（非 shadow）后是否仍成立——闭环会改变输入分布，经验 Lipschitz 增益需重新估计；
4. "持续亚稳态"的形式定义。本文当前只以 §9.2 的有界性近似之，未给出吸引子结构的刻画。

黑盒策略扰动的收敛性可参考 Salimans et al.（2017）与 Spall（1992）；若采用 KL 约束的策略更新，可参考 Schulman et al.（2017）。

## 10. 实验设计

### 10.1 条件

| 编号 | 条件 |
|------|------|
| C1 | 冻结模型直接生成 |
| C2 | 仅上下文检索 |
| C3 | 上下文 + 情绪状态 |
| C4 | 上下文 + 熵扰动 |
| C5 | EVF-MACK 完整架构 |
| C6 | C5 去除动态拓扑 |
| C7 | C5 去除反对齐项 |
| C8 | C5 的 LIF 门替换为等发放率的瞬时 sigmoid 门 |
| C9 | C5 去除脉冲门控（慢路径每回合执行）|

C8 检验 §5.3/§6.2 从 SymbolicLight V1 迁移来的机制假设在本文设定下是否成立；等发放率是必要的匹配条件，否则混淆门控形式与触发频率。C9 分离"双路径省了多少成本"与"双路径损失了多少质量"。

### 10.2 任务

- 多轮对话的一致性与新颖性；
- 长期记忆检索；
- 工具调用路径选择；
- 主动消息和离线学习；
- 输入分布变化后的状态恢复；
- 高不确定性场景下的探索效率。

### 10.3 规模与统计方案

**v0.1 的问题**：7 条件 × 6 任务族，无样本量、无统计检验、无预设效应量；且"用户评价和人工盲评"在单用户系统上是 \(n=1\)，无法支撑 H1–H5。

**v0.2 方案**：

- **可复现主协议**：使用脚本化用户模拟器（固定剧本 + 受控扰动）而非真人，保证条件间输入完全配对。每条件 5 个随机种子 × 每任务族 40 个 episode，共 \(9\times6\times5\times40=10800\) 个 episode。
- **配对设计**：同一 episode 的同一提示在所有条件下运行，采用配对比较以消除提示难度方差。
- **检验**：主要指标用配对 bootstrap（\(10^4\) 次重采样）与 Wilcoxon 符号秩检验并报，跨 5 个假设用 Holm–Bonferroni 校正，族错误率 \(\alpha=0.05\)。
- **效应量**：预先登记最小可检测效应（MDE），并报告 Cliff's delta 而非仅 p 值。
- **确认性 vs 探索性**：H1、H2、H5 为确认性（预设阈值，纳入多重校正）；H3、H4 为探索性（报告效应量与置信区间，不作显著性声明）。
- **人工评估的定位**：单用户系统的真人盲评作为**支持性证据**，不作确认性结论。协议为受试内、逐回合随机分配、成对强制选择，累计 ≥200 组配对比较、跨 ≥30 个会话；报告评分者内一致性（重复 10% 样本）。

### 10.4 指标

- 预测自由能代理值与其滑动基线；
- 后验熵、熵带越界率、\(D_{KL}(q_\phi\|p)\)、\(D_{effective}\)（用于区分塌缩，见 §2.3）；
- 输出语义多样性（embedding 分散度与 distinct-n 并报）；
- 长期事实一致性、\(Q_{spec}\) 与声明密度；
- 情绪状态连续性与 \(A_{cal}\)；
- 工具成功率与调用成本；
- 拓扑重连率、发放率、膜电位分布；
- 慢路径触发率与每回合平均成本（token、延迟、工具次数）；
- 策略版本漂移与证据覆盖率；
- 副作用隔离不变量的违反次数（应恒为 0）。

### 10.5 关键假设

各假设给出预设的判定阈值以保证可伪证性：

- **H1（确认性）**：相对 C2，C5 的语义多样性提升 ≥15%，同时 \(Q_{fact}\) 下降 ≤3 个百分点。前置有效性检查见 §3.4：若多样性随温度单调线性上升而未饱和，则判定多样性指标测到的是解码噪声，H1 不可评估。
- **H2（确认性）**：C5 中三评分场梯度的平均成对余弦 \(\overline{K_{ij}}\) 显著低于 C7，差异 ≥0.10。
- **H3（探索性）**：液态时间常数在等待、探索与高唤醒场景下改善响应节奏（报告效应量）。
- **H4（探索性）**：C5 在分布变化后的恢复速度快于 C6（报告效应量）。
- **H5（确认性）**：C5 的无证据慢变量变更率显著低于取消快慢分离的对照，且 §9.2 不变量违反次数不增加。
- **H6（确认性，新增）**：C5 相对 C9 的每回合平均成本下降 ≥40%，同时 \(S_c\) 下降 ≤2 个百分点；C5 相对 C8 在匹配发放率下 \(S_c\) 不低于 C8。H6 前半部分检验双路径的成本收益，后半部分检验 §2.6 迁移假设。

## 11. 局限性

1. **能力上界是结构性的**。外部控制器改变的是 \(p_\theta\) 支撑集上的加权与组合方式，基础模型内部表征保持冻结。按 §3.4，系统无法产生 \(p_\theta\) 不具备的模式；因此本架构的收益应表述为"更好地调度已有能力"，而非"扩展能力"。这是本文最主要的限制，也是与 SymbolicLight V1 一类架构+训练路线的根本分野。
2. **黑盒梯度噪声大且部分不可观测**。远程模型不返回 log-probability 时，\(S_c\) 只能依赖代理指标；SPSA 的方差随维度上升，离线批量部分缓解但未消除。
3. **评分器本身引入偏差，且存在自指风险**。\(F_{affect}\) 的目标由被控模型产生（§4.2.2），\(A_{cal}\) 只能减轻不能消除该循环；\(Q_{task}\) 在无标注任务上依赖模型评审器，同样继承其偏好。
4. **拓扑持续变化提高复现难度**。§6.2 的膜电位轨迹归档改善了单次变更的可解释性，但跨长时程的结构比较仍缺少稳定的对齐方式。
5. **熵提升不必然带来质量**。必须结合信息增益、结果证据与能量预算；§10.5 的 H1 因此绑定了一致性的下降上限。
6. **脉冲门控引入漏检风险**。慢路径被门控意味着某些应当审慎处理的回合可能走了快路径。§7.2 的强制触发条件是缓解措施而非保证；漏检率需按 §10.4 单独统计。
7. **跨设定的机制迁移未经本设定验证**。§5.3、§6.2 采用 LIF 而非瞬时阈值的依据来自亚十亿规模、从零训练的语言模型消融（§2.6），与本文冻结大模型 + 外部控制的设定差异显著。该迁移由 C8 检验，在结果出来前应视为假设。
8. **双时标收敛性未证**（§9.4 第 2 项）。慢变量由 LIF 门离散触发，不满足标准双时标随机逼近的条件。

## 12. 实现路线

### 阶段一：观测层

记录每轮的自由能代理、熵、候选分支、情绪状态、工具路径与拓扑变化。同时接入 \(\Delta t\) 实测与 \(\alpha_t\) 上界告警（§5.2），在写任何控制逻辑前先确认时间常数配置正确。

### 阶段二：shadow controller

外部控制器只生成建议和评分，不参与最终回复，验证指标与轨迹差异。此阶段先行落地 §4.1 的工具副作用分类与 dry-run 接口，因为它是后续所有多分支实验的前置条件。

### 阶段三：候选门控

开启多分支采样、critic 重排与工具动作门控，接入 \(g^{online}\)。此阶段起 \(I_{inv}\) 必须包含副作用隔离检查。

### 阶段四：脉冲门控与拓扑

加入 LIF 控制脉冲、双路径调度、累积器重连门、快慢策略版本与离线 replay（含 \(g^{offline}\)）。

### 阶段五：实证比较

运行 §10 的完整消融，并在 HDSC 的 trace、lifecycle 与 consolidation 数据上做长期回放。C8/C9 与 H6 应在此阶段与主假设同批评估，避免机制迁移假设长期悬空。

## 附录 A：符号表

| 符号 | 含义 | 备注 |
|------|------|------|
| \(\theta\) | 基础模型参数 | 全程冻结 |
| \(\psi\) | 外部控制器参数 | 分 \(\psi_f\)（快）/ \(\psi_s\)（慢） |
| \(z_t\) | 连续 VAE 潜变量 | §2.7 第一层 |
| \(h_t\) | 高维超向量 | §2.7 第二层。**仅指超向量** |
| \(\ell_t\) | 液态连续控制状态 | §5.1。v0.1 中误用 \(h_t\) |
| \(G_t\) | 动态拓扑 | §2.7 第三层 |
| \(q_t\) | 潜状态近似后验 | |
| \(e_t,\ e_t^*,\ \hat e_{t+1}\) | 当前 / 目标 / 实测情绪状态 | |
| \(E_t\) | **可用**能量与剩余预算 | 归一化 \([0,1]\)；v0.1 符号使用有误 |
| \(F_t\) | 变分自由能 | \(-F\) 内含 \(+H(q)\)，见 §2.2 |
| \(\lambda_H^{ex}\) | **额外**熵权重 | 等效总系数 \(1+\lambda_H^{ex}\) |
| \(S_c,S_e,S_s\) | 认知 / 情绪 / 系统评分场 | 各自权重和为 1，需最大化 |
| \(g_i\) | 对 \(S_i\) 的上升方向估计 | 不再使用损失记号 |
| \(\tilde g_i\) | 反对齐修正后的方向 | 满足命题 1 |
| \(\Omega_t\) | 反对称旋转矩阵 | \(\|\Omega_t\bar g\|\le\kappa_\Omega\|\bar g\|\) |
| \(\xi_t,\ \sigma_t\) | OU 扰动与其尺度 | 稳态方差 \(\sigma_t^2\) |
| \(v,\ z,\ \vartheta,\ \beta\) | 膜电位 / 脉冲 / 阈值 / 泄漏 | 下标 \(G\)=边级，\(c\)=全局控制 |
| \(\hat a_{t,k},\ \hat s_{t+1,k}\) | 提议动作与预测状态 | 评分只读带帽量 |
| \(\Delta t,\ \tau_t,\ \alpha_t\) | 实测步长 / 时间常数 / 离散系数 | 要求 \(\tau_{min}\ge 5\Delta t\) |

## 附录 B：v0.1 → v0.2 修订清单

**形式化勘误**

| # | 位置 | v0.1 | v0.2 |
|---|------|------|------|
| 1 | §3.3 | \(J_t=-F_t+\lambda_HH(q_t)+\cdots\)，熵被重复计入（\(-F\) 已含 \(+H\)） | 改为 \(\lambda_H^{ex}\) 并说明等效系数为 \(1+\lambda_H^{ex}\)（§2.2、§3.3） |
| 2 | §2.2 | "仅最小化自由能会倾向于快速收缩状态分布" | 该陈述不成立；改述为后验塌缩与摊销间隙（§2.3） |
| 3 | §4.2 | \(\tilde g_i=g_i-c_i\)，三目标一致且 \(\lambda_{ij}\ge0.5\) 时抵消或反向 | 引入自适应 \(\eta_i\) 与命题 1，保证 \(\tilde g_i^Tg_i\ge\kappa_g\|g_i\|^2\)（§4.4） |
| 4 | §4.1.4 | 同时使用 \(\hat g_i\)（分数）与 \(g_i=\nabla_\psi L_i\)（损失），§6.3 取加号 | 统一为对分数上升，删除损失记号（§4.3） |
| 5 | §4.3 | \(\sigma_t\) 含 \(-dE_t\)，能量充足时探索反而减少 | 改为 \(+dE_t\)；熵项改为闭环 \(H^{err}_t\)（§4.5） |
| 6 | §2.5 / §5 | \(h_t\) 同时表示超向量与液态状态 | 液态状态改记 \(\ell_t\)；附录 A 给出符号表 |
| 7 | §5 | \(\alpha_t=\text{clip}(\Delta t/\tau_t,0,1)\) 在 60 秒心跳下退化为 1 | 要求 \(\tau_{min}\ge5\Delta t\)、\(f\) 有界、\(\Delta t\) 取实测值、加上界告警（§5.2） |
| 8 | §4.1.1 | \(Q_{fact}\) 未知声明不计入分母 → 可被"说废话"刷分 | 未知声明按 \(\zeta\) 部分计入分母，空集取先验，新增 \(Q_{spec}\)（§4.2.1） |
| 9 | §2.5 | \(\text{bundle}=\text{sign}(\cdot)\)，偶数项叠加时 \(\text{sign}(0)=0\) 破坏双极性 | 改为带固定种子随机破平的 \(\text{sign}_\star\)（§2.7） |
| 10 | §2.5 | \(D_{effective}\) 未指明空间与窗口，且对双极超向量平凡 | 限定为 \(z_t\) 的窗口协方差谱，\(h_t\) 改用稀疏度/绑定可逆性（§2.7） |
| 11 | §4.2 | 未说明 \(P_{g_j}(g_i)\) 的定义，与 PCGrad 触发符号相反但未澄清 | 给出定义并显式对比 PCGrad（§4.4） |
| 12 | §4.2 | \(\Omega_t\) 幅度未定、未与状态耦合 | 给出归一化与熵带驱动的 \(\omega_t\)、上界 \(\kappa_\Omega\) 与正交性性质（§4.5） |
| 13 | §4.1.1 | \(Q_{tool}\) 无工具调用时取 1 | 改为记为缺失并重归一，避免误判为满分（§4.2.1） |

**结构性修改**

| # | 内容 | 章节 |
|---|------|------|
| 14 | 候选分支副作用隔离：工具三分类、dry-run、提议动作 \(\hat a\) 与预测状态 \(\hat s\)、纳入 \(I_{inv}\) | §4.1、§7.3、§12 阶段二 |
| 15 | 元梯度重构：在线用候选批 bandit（零边际成本），SPSA 移至离线 replay；控制核开销计入 \(C(u_t)\) 与 \(R_{resource}\) | §4.3、§4.2.3 |
| 16 | LIF 累积器门替换无记忆瞬时门，给出命题 2 与选择性增益 \(1/(1-\beta_v)\) | §5.3、§6.2 |
| 17 | 双路径算力调度：快路径常开、慢路径由自由能脉冲门控、强制触发条件、阈值与预算耦合 | §7 |
| 18 | 可达集上界与由此导出的 H1 前置有效性检查 | §3.4、§10.5 |
| 19 | 情绪评分场的自指风险与 \(A_{cal}\) 校准锚定项 | §4.2.2 |
| 20 | §9 重写为"已实现证书 / 运行时不变量 / 已获保证 / 开放命题"四类，接入 `active_space.py` 的 small-gain certificate | §9 |
| 21 | 新增"与已有实现的关系"与复用对照表，避免重复实现已有 critic 与证书 | §1.4、§8.5 |
| 22 | 实验设计补统计功效：脚本化模拟器、配对设计、样本量、Holm–Bonferroni、MDE、确认性/探索性划分、人工评估降级为支持性证据 | §10.3 |
| 23 | 新增条件 C8（LIF vs 等发放率瞬时门）、C9（去脉冲门控）与假设 H6（成本收益 + 迁移验证） | §10.1、§10.5 |
| 24 | 局限性补充：能力上界的结构性、脉冲门漏检、跨设定迁移未验证、双时标收敛未证 | §11 |
| 25 | 补入冻结权重外部自我改进的相关工作定位（Generative Agents / Reflexion / Voyager / CoALA），并明确本文增量为可测量性与可证明约束 | §2.1 |
| 26 | 补入脉冲门控与双路径的来源（SymbolicLight V1）及其引用范围限定 | §2.6 |
| 27 | 新增符号表 | 附录 A |

**待办**（本版未解决，需作者确认）

1. §8.3 引用的总仓 `巩固循环/` 路径需与 `zero` 仓库实际结构核对后再定稿。
2. §2.6 对 SymbolicLight V1 的具体数值（PPL 区间、稀疏度定义、消融结论）转引自公开摘要与索引页，定稿前应比对原文全文，尤其"确定性 top-k 掩码退化更大"这一条是 §5.3/§6.2 的主要外部依据。
3. \(\lambda_{ij}\)、\(\kappa_g\)、\(\kappa_\Omega\)、\(\vartheta_{G}\)、\(\beta_G\)、\(\vartheta_c\)、\(\beta_c\)、\(\zeta\)、\(\eta_\tau\) 的具体取值需由 §12 阶段一/二的观测数据标定，本文只给出约束区间。
4. §9.4 第 3 项（闭环接入后证书是否仍成立）应在阶段四前完成一次经验增益重估。

## 参考文献

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127-138. https://doi.org/10.1038/nrn2787
2. Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. *Cognitive Neuroscience*, 6(4), 187-214. https://doi.org/10.1080/17588928.2015.1020053
3. Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., & Irving, G. (2019). Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593. https://arxiv.org/abs/1909.08593
4. Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35. https://arxiv.org/abs/2203.02155
5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. https://arxiv.org/abs/1707.06347
6. Hasani, R. et al. (2021). Liquid Time-constant Networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), 7657-7666. https://doi.org/10.1609/aaai.v35i9.16936
7. Yu, T. et al. (2020). Gradient Surgery for Multi-Task Learning. *Advances in Neural Information Processing Systems*, 33. https://arxiv.org/abs/2001.06782
8. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526. https://doi.org/10.1073/pnas.1611835114
9. Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.08840
10. Salimans, T. et al. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864. https://arxiv.org/abs/1703.03864
11. Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Transactions on Automatic Control*, 37(3), 332-341. https://doi.org/10.1109/9.119632
12. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*, 393, 440-442. https://doi.org/10.1038/30918
13. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press. https://mitpress.mit.edu/9780262610590/sparse-distributed-memory/
14. Plate, T. A. (1995). Holographic Reduced Representations. *IEEE Transactions on Neural Networks*, 6(3), 623-641. https://doi.org/10.1109/72.377968
15. Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. *Artificial Intelligence*, 46(1-2), 159-216. https://doi.org/10.1016/0004-3702(90)90007-M
16. Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors. *Cognitive Computation*, 1, 139-159. https://doi.org/10.1007/s12559-009-9009-8
17. Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. (2023). A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I & II. *ACM Computing Surveys*. https://doi.org/10.1145/3538531
18. Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. arXiv:2304.03442. https://arxiv.org/abs/2304.03442
19. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems*, 36. arXiv:2303.11366. https://arxiv.org/abs/2303.11366
20. Wang, G. et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291. https://arxiv.org/abs/2305.16291
21. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023). Cognitive Architectures for Language Agents. arXiv:2309.02427. https://arxiv.org/abs/2309.02427
22. Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press. https://doi.org/10.1017/CBO9780511815706
23. Marsella, S., & Gratch, J. (2009). EMA: A process model of appraisal dynamics. *Cognitive Systems Research*, 10(1), 70-90. https://doi.org/10.1016/j.cogsys.2008.03.005
24. SymbolicLight V1: Spike-Gated Dual-Path Language Modeling with High Activation Sparsity and Sub-Billion-Scale Pre-Training Evidence (2026). arXiv:2605.21333. https://arxiv.org/abs/2605.21333
