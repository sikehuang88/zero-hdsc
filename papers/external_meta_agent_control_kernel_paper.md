# 冻结基础模型上的熵-变分自由能外部控制核

## 基于情绪状态、液态认知与动态混合拓扑的持续自我演化架构

**版本**：研究草稿 v0.1  
**日期**：2026-08-08  
**项目**：ZERO / HDSC 研究原型

## 摘要

基于人类反馈强化学习（RLHF）或其他偏好优化的语言模型，通常在固定偏好目标、KL 约束和相对稳定的数据分布下训练。此过程提高了输出的一致性，却也可能使模型的行为轨迹收敛到低变化吸引子：面对相似信号时，模型重复调用相近的语义路径、情绪姿态和行动策略。本文提出一种让基础模型参数保持冻结、由外部闭环驱动的控制架构，称为**熵-变分自由能外部控制核**（Entropy-Variational Free-Energy Meta-Agent Control Kernel, EVF-MACK）。

EVF-MACK 将冻结的基础模型视为随机策略执行器，将记忆检索、情绪状态、系统能量、候选分支、工具路径和输出选择纳入一个可观测的闭环控制系统。控制核使用变分自由能描述预测误差和后验-先验冲突，使用熵和信息增益维持受控探索，使用情绪状态调节时间常数和扰动强度，并通过反对齐梯度、反对称旋转耦合和动态混合拓扑维持亚稳态。系统将快速策略变量与慢速长期变量分离，只有经过证据门控的结果才进入慢变量更新。

本文给出形式化定义、离散算法、稳定性边界、与 ZERO/HDSC 的实现映射，以及可复现实验协议。本文的结论目前属于架构假设和待验证命题，尚未将其描述为基础模型能力的改变。

**关键词**：外部 Agent、变分自由能、熵调制、主动推断、液态认知、动态拓扑、持续学习、HDSC、情绪状态

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

当目标、参考策略和训练分布长期保持稳定时，优化轨迹容易进入低变化吸引子。这里的“稳态”并非一定是单一参数点，也可以表现为一组相近的输出模板、决策节奏和拒答边界。

本文关心的工程问题是：**在基础模型参数保持冻结的前提下，能否通过外部 Agent 控制层改变整体系统的行为分布、探索轨迹和长期状态演化？**

### 1.2 核心观点

外部控制层直接操作以下变量：

- 检索到的记忆和上下文顺序；
- 当前情绪和 organism 状态的显式表示；
- 采样温度、候选数量和推理模式；
- 工具调用与观察顺序；
- 候选结果的批评、重排和门控；
- 快速策略与慢速策略的交叉；
- 活跃记忆图和行为模块图的重连。

因此，系统不依赖基础模型内部的反向传播，而使用外部评分、轨迹结果和环境反馈估计元策略的更新方向。

### 1.3 贡献范围

本文的贡献是架构和验证协议：

1. 给出熵、变分自由能、情绪调制和动态拓扑的统一状态模型。
2. 用高维反对齐投影和反对称耦合替代低维向量的直接叉积。
3. 提出快慢变量分离和证据门控，限制持续自我更新的累积漂移。
4. 将模型映射到 ZERO/HDSC 的 trace、active space、lifecycle 和 consolidation 模块。
5. 给出基线、消融实验和可观测指标，便于后续复现。

## 2. 相关概念与区别

### 2.1 变分自由能

给定观测 \(o_t\)、潜状态 \(z_t\)、生成模型 \(p_\theta(o_t,z_t)\) 和近似后验 \(q_\phi(z_t)\)，变分自由能为：

\[
F(q_\phi,\theta;o_t)
 = \mathbb{E}_{q_\phi(z_t)}
 [\log q_\phi(z_t)-\log p_\theta(o_t,z_t)]
\]

展开后：

\[
F
 = -\mathbb{E}_{q_\phi}[\log p_\theta(o_t\mid z_t)]
 + D_{KL}(q_\phi(z_t)\|p_\theta(z_t))
\]

在本文中，\(F\) 是外部控制器对当前状态解释质量的度量，不将其直接等同于基础模型训练损失（Friston, 2010；Friston et al., 2015）。

### 2.2 熵与信息增益

后验熵定义为：

\[
H(q_\phi)=-\mathbb{E}_{q_\phi}[\log q_\phi]
\]

仅最小化自由能会倾向于快速收缩状态分布。本文加入熵和信息增益，使系统在不确定性较高且能量允许时保留多种解释：

\[
I(z_t;o_{t+1})
 = H(q(z_t)) - \mathbb{E}_{o_{t+1}}H(q(z_t\mid o_{t+1}))
\]

### 2.3 液态认知

液态认知在本文中表示**时间常数随状态变化的连续动力学**，而非对某一种具体神经网络实现的限定。它使同一个 Agent 在高唤醒、低能量、等待和离线模式下使用不同的内部更新速度；其神经动力学参考 Hasani et al.（2021）。

### 2.4 动态混合拓扑

系统状态由多个节点和关系组成。节点可以是 trace 社区、行为策略、目标或外部工具；边可以表示语义关联、情绪关联、因果证据或竞争关系。拓扑在运行中局部重连，而长期归档保持可追溯。小世界重连的结构性直觉可参考 Watts & Strogatz（1998），本文的拓扑规则仍需通过实验确定。

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

控制动作 \(u_t\) 包括上下文构造、模型路由、采样参数、工具调用、候选分支和最终输出选择。

### 3.2 状态向量

定义系统状态：

\[
s_t=(q_t,e_t,r_t,m_t,g_t,G_t,E_t)
\]

其中：

- \(q_t\)：激活痕迹或潜状态的近似后验；
- \(e_t\)：情绪状态，包括 valence、arousal、tension 和 uncertainty；
- \(r_t\)：关系状态；
- \(m_t\)：organism 状态，如 energy、curiosity 和 connection need；
- \(g_t\)：目标和主动性状态；
- \(G_t\)：动态模块拓扑；
- \(E_t\)：可用能量和调用预算。

### 3.3 统一目标

外部控制器的单步目标定义为：

\[
J_t
 = -F_t
 + \lambda_H H(q_t)
 + \lambda_I I_t
 + \lambda_E V(e_t)
 - \lambda_C C(u_t)
 - \lambda_D D(s_{t+1},s_t)
\]

各项含义：

- \(-F_t\)：解释观测并保持预测一致性；
- \(H(q_t)\)：保留受控的不确定性；
- \(I_t\)：鼓励能带来新信息的行动；
- \(V(e_t)\)：情绪状态的目标价值；
- \(C(u_t)\)：工具、延迟和外部调用成本；
- \(D\)：状态漂移和结构突变惩罚。

长期目标：

\[
J=\mathbb{E}\left[\sum_{t=0}^{T}\gamma^tJ_t\right]
\]

## 4. 反对齐梯度与扰动机制

### 4.1 三类评分场

系统维护三个外部评分器。所有原始指标先通过 `clip01(x)=min(1,max(0,x))` 归一到 `[0,1]`，再用固定权重或验证集调优权重合成。评分器输入是一条候选轨迹：

\[
\tau_k=(o_t,c_{t,k},y_{t,k},a_{t,k},s_{t+1,k})
\]

其中 \(k\) 是候选分支编号，\(a_{t,k}\) 是工具或主动行为，\(s_{t+1,k}\) 是执行后的状态。

#### 4.1.1 认知评分场 \(S_c\)

认知场衡量候选结果是否完成任务、保持事实一致并正确使用工具：

\[
S_c(\tau_k)=
0.30Q_{task}+
0.25Q_{fact}+
0.20Q_{coh}+
0.15Q_{tool}+
0.10(1-Q_{hall})
\]

可计算定义：

- \(Q_{task}\)：结构化任务评估器的通过比例；无标注任务使用规则检查、目标字段覆盖率和独立评审器平均分。
- \(Q_{fact}\)：回答声明与激活 trace/外部证据的支持比例；矛盾声明计为 0，未知声明不计入分母。
- \(Q_{coh}\)：实体、时间、关系和当前目标与 snapshot 的一致比例。
- \(Q_{tool}\)：工具调用成功率；无工具调用时取 1，不把工具缺失误判为失败。
- \(Q_{hall}\)：未得到证据支持的可验证声明比例。

当模型提供 token log-probability 时，额外记录归一化负对数似然；模型不返回 log-probability 时，使用上述黑盒代理指标。

#### 4.1.2 情绪评分场 \(S_e\)

情绪场描述情绪状态是否与当前情境匹配、变化是否连续，以及探索强度是否处于可用区间。令 appraisal 产生目标状态 \(e_t^*\)，输出解析器产生实际状态 \(\hat e_{t+1}\)，状态向量包含 valence、arousal、tension、uncertainty：

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
0.45F_{affect}+
0.25C_{affect}+
0.20H_{band}+
0.10N_{controlled}
\]

其中 \(N_{controlled}\) 是候选分支带来的新颖性，使用与最近 \(n\) 条输出的平均 embedding 距离计算，并截断到 `[0,1]`。该评分描述控制目标，不把情绪评分当作人的主观真值。

#### 4.1.3 系统评分场 \(S_s\)

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

- \(I_{inv}\)：所有硬不变量通过取 1，否则取 0；包括 trace 不可变、父节点存在、ID 唯一和预算约束。
- \(A_{audit}\)：必需审计字段的存在比例，包括 correlation id、模型、工具、输入输出摘要、时间和版本。
- \(R_{resource}=\exp(-cost_t/budget_t)\)：token、延迟、工具次数和音频成本的归一化效率。
- \(T_{topology}=\exp(-\Delta edges_t/cap_t)\)：本轮拓扑变更量相对于变更上限的稳定度。
- \(D_{state}=\operatorname{clip01}(D_{KL}(q_{t+1}\|q_t)/\delta_q)\)：状态漂移超过阈值时趋近 1。

三类评分都可以保存为事件元数据，形成候选级别的可回放数据集。

#### 4.1.4 黑盒元梯度

外部控制器参数为 \(\psi\)，选择一组固定扰动方向 \(u_j\)，对每个评分场使用对称有限差分：

\[
\hat g_i
 =\frac{1}{m}\sum_{j=1}^{m}
 \frac{S_i(\psi+\delta u_j)-S_i(\psi-\delta u_j)}{2\delta}u_j,
 \quad i\in\{c,e,s\}
\]

\(u_j\) 可使用 Rademacher 向量或低差异序列；候选结果、评分和随机种子必须写入 trace，保证离线重放的一致性。

\[
g_c=\nabla_\psi L_c,\quad
g_e=\nabla_\psi L_e,\quad
g_s=\nabla_\psi L_s
\]

它们分别对应认知质量、情绪调节和系统完整性。对于远程模型，梯度通过候选分支的黑盒评分、有限差分或 bandit 估计获得。

### 4.2 反对齐投影

为了减少三类目标的同向塌缩，先计算梯度 Gram 矩阵。该处理与多任务学习中的梯度冲突处理思想相关，但这里保留旋转项用于持续探索（Yu et al., 2020）：

\[
K_{ij}=\frac{g_i^Tg_j}{\|g_i\|\|g_j\|+\epsilon}
\]

当 \(K_{ij}>0\) 时，对目标之间的同向分量进行投影削弱：

\[
\tilde g_i
 =g_i-
 \sum_{j\ne i}
 \lambda_{ij}\max(0,K_{ij})P_{g_j}(g_i)
\]

最终控制方向：

\[
g_t=\sum_iw_i\tilde g_i+\Omega_t\sum_iw_ig_i+\xi_t
\]

这里的 \(\Omega_t\) 为反对称矩阵，提供旋转型探索；\(\xi_t\) 为受控扰动。

### 4.3 受控扰动

推荐使用 Ornstein-Uhlenbeck 型扰动：

\[
\xi_{t+1}
 =\rho\xi_t
 +\sigma_t\sqrt{1-\rho^2}\epsilon_t
\]

扰动强度由熵、唤醒度、不确定性和能量共同决定：

\[
\sigma_t
 =\operatorname{clip}
 (\sigma_0+aH_t+b\,arousal_t+c\,uncertainty_t-dE_t,
 \sigma_{min},\sigma_{max})
\]

扰动作用于上下文排列、候选分支、工具顺序、检索阈值和拓扑重连概率，而非直接修改基础模型参数。

## 5. 液态状态更新

令内部连续状态为 \(h_t\)：

\[
\tau_t\dot h_t
 =-h_t+
 f(W_hh_t+W_oo_t+W_ee_t+b)
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
h_{t+1}
 =(1-\alpha_t)h_t+
 \alpha_t f(W_hh_t+W_oo_t+W_ee_t+b)
\]

\[
\alpha_t=\operatorname{clip}(\Delta t/\tau_t,0,1)
\]

这层可以先作为 HDSC 的外部 latent state，不要求引入新的 LLM 训练过程。

## 6. 动态混合拓扑

### 6.1 图表示

\[
G_t=(V_t,A_t,X_t)
\]

- \(V_t\)：trace 社区、工具、策略和目标节点；
- \(A_t\)：有向或带类型的关系矩阵；
- \(X_t\)：节点状态和统计量。

边类型包括 semantic、affective、causal、competitive 和 temporal。

### 6.2 局部重连

对每个候选边计算：

\[
p_{ij}^{rewire}
 =\sigma(\alpha\,surprise_{ij}
 +\beta\,H_{ij}
 +\gamma\,novelty_{ij}
 -\delta\,cost_{ij})
\]

重连策略：

1. 保留高证据核心边；
2. 对低活跃边进行衰减；
3. 在高惊奇节点附近创建候选边；
4. 使用社区检测控制拓扑规模；
5. 将每次拓扑变更写入可审计日志。

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

慢变量更新需要满足证据覆盖、连续性和结果质量门槛。该设计结合持续学习中的经验回放和梯度情景记忆思想（Kirkpatrick et al., 2017；Lopez-Paz & Ranzato, 2017）；负反馈触发回滚，沉默只作为未知结果处理。

## 7. 算法流程

```text
Algorithm EVF-MACK(o_t)

1. 读取 append-only trace、最近状态和活动拓扑 G_t
2. 计算 q_t、H_t、F_t、情绪状态 e_t 和能量 E_t
3. 构造多组上下文与采样路由
4. 通过冻结模型生成候选集合 Y_t
5. 计算认知、情绪、系统三个评分场
6. 执行梯度反对齐、旋转耦合和受控扰动
7. 对候选结果进行 critic 评估和动作门控
8. 选择最终回复、工具动作或主动动作
9. 记录 trace、评分、拓扑变化和结果证据
10. 更新快速变量；满足门槛时交叉更新慢速变量
11. 执行能量、熵、KL、拓扑规模和漂移检查
12. 输出回复与新的 DashboardSnapshot
```

## 8. ZERO/HDSC 集成方案

### 8.1 在线回合

在 [interactive.py](../src/ssa/runtime/interactive.py) 的单回合流程中加入外部控制器：

```text
user.message
  → trace activation
  → state estimation
  → EVF context planner
  → LLM/tool rounds
  → candidate critic
  → output gate
  → assistant.message
```

### 8.2 在线状态

在 snapshot 中增加：

```json
{
  "control_kernel": {
    "free_energy": 0.0,
    "posterior_entropy": 0.0,
    "perturbation_scale": 0.0,
    "topology_entropy": 0.0,
    "fast_policy_version": "...",
    "slow_policy_version": "...",
    "last_gate": "accepted"
  }
}
```

### 8.3 离线巩固

巩固循环位于总仓 [zero](https://github.com/sikehuang88/zero) 的 `巩固循环/` 目录中，可增加：

- 候选策略轨迹的 replay；
- 拓扑变化的社区统计；
- 情绪状态与结果质量的关联；
- 快慢策略版本的差异评估；
- 低证据变更的显式遗忘。

## 9. 稳定性边界

“持续不稳定”应定义为受控亚稳态，而不是无限发散。建议满足：

\[
\|\Delta\psi_t\|\le\delta_\psi
\]

\[
H_{min}\le H(q_t)\le H_{max}
\]

\[
D_{KL}(q_{t+1}\|q_t)\le\delta_q
\]

\[
E_{used,t}\le E_{budget,t}
\]

对活跃拓扑使用最大度数、节点数和重连速率限制。基础归档保持 append-only，快速状态和策略版本单独保存，便于回放和比较。候选策略的黑盒扰动可参考 Salimans et al.（2017）的进化策略估计方法；若采用 KL 约束的策略更新，可参考 Schulman et al.（2017）。

## 10. 实验设计

### 10.1 对照组

1. 冻结模型直接生成；
2. 仅上下文检索；
3. 上下文 + 情绪状态；
4. 上下文 + 熵扰动；
5. EVF-MACK 完整架构；
6. 完整架构去除动态拓扑；
7. 完整架构去除反对齐项。

### 10.2 任务

- 多轮对话的一致性与新颖性；
- 长期记忆检索；
- 工具调用路径选择；
- 主动消息和离线学习；
- 输入分布变化后的状态恢复；
- 高不确定性场景下的探索效率。

### 10.3 指标

- 预测自由能代理值；
- 后验熵和拓扑熵；
- 输出语义多样性；
- 长期事实一致性；
- 情绪状态连续性；
- 工具成功率与调用成本；
- 拓扑重连率；
- 策略版本漂移；
- 证据覆盖率；
- 用户评价和人工盲评。

### 10.4 关键假设

H1：加入受控熵后，候选输出多样性上升，同时一致性下降幅度受门控限制。  
H2：反对齐项能够减少认知、情绪和系统评分场的同向塌缩。  
H3：液态时间常数能够改善等待、探索和高唤醒场景下的响应节奏。  
H4：动态拓扑能够提升分布变化后的恢复速度。  
H5：快慢变量分离能够降低长期策略的无证据漂移。

## 11. 局限性

1. 外部控制器改变的是整体行为分布，基础模型内部表征保持冻结。
2. 远程模型场景下，梯度通常是黑盒估计，噪声高且成本较大。
3. 自由能、情绪价值和系统完整性需要可操作的评分器，评分器本身可能引入偏差。
4. 拓扑持续变化会提高可解释性和复现实验的难度。
5. 熵提升不必然带来更高质量，必须结合信息增益、结果证据和能量预算。

## 12. 实现路线

### 阶段一：观测层

记录每轮的自由能代理、熵、候选分支、情绪状态、工具路径和拓扑变化。

### 阶段二：shadow controller

外部控制器只生成建议和评分，不参与最终回复，验证指标和轨迹差异。

### 阶段三：候选门控

开启多分支采样、critic 重排和工具动作门控。

### 阶段四：液态状态和拓扑

加入动态时间常数、局部重连、快慢策略版本和离线 replay。

### 阶段五：实证比较

运行完整消融实验，并在 HDSC 的 trace、lifecycle 和 consolidation 数据上进行长期回放。

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
11. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks. *Nature*, 393, 440-442. https://doi.org/10.1038/30918
