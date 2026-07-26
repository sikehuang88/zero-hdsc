# SSA — 相关工作综述

> 版本: 0.1 draft
> 日期: 2026-07-25
> 状态: 文献整理中

本文档列出 SSA 必须对标的相关工作，按主题分类。每篇论文标记【必读】/【参考】/【背景】。

---

## 1. LLM Agent 记忆系统（最直接对比）

### 【必读】Generative Agents: Interactive Simulacra of Human Behavior
- 作者: Joon Sung Park, Joseph O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
- 会议: UIST 2023
- 核心: 25 个 LLM agent 在 Smallville 小镇生活，涌现出社会行为
- 架构: **Memory Stream** + **Reflection** + **Planning**
  - Memory Stream: 按时间排序的观察记录
  - Reflection: 周期性地将低层观察合成为高层洞见
  - Planning: 基于记忆生成日程
- **SSA 差异**:
  - 它们的记忆是时间线，SSA 是空间几何
  - 它们有 Reflection/Planning 这种"高层状态"，SSA 严格无状态
  - 它们的 agent 有人设 prompt（"John Lin is..."），SSA 无预设人格
  - 它们的"性格"是 prompt 给的，SSA 的"性格"是空间形状的涌现

### 【必读】MemGPT: Towards LLMs as Operating Systems
- 作者: Charles Packer, Vivian Fang, Shishir Patil, Ion Stoica, Joseph Gonzalez
- 会议: arXiv 2023 (后续有扩展)
- 核心: 把 LLM 上下文窗口当"内存"，外部存储当"磁盘"，分页调度
- 架构: Main Context (system + user + assistant + working memory) + External Storage (recall storage + archival storage)
- **SSA 差异**:
  - MemGPT 有"工作记忆"=状态，SSA 严格无状态
  - MemGPT 的记忆是分层的（主存/外存），SSA 是单一空间
  - MemGPT 有"操作系统"式主动调度，SSA 的"调度"是空间激活的副产品

### 【参考】A Survey on the Memory Mechanism of Large Language Model based Agents
- 作者: **Zeyu Zhang** et al.（v0.3 修正：第一作者是 Zhang，非 Tang）
- 会议: 2024
- 核心: 综述 LLM agent 的各种记忆机制
- 用途: 拿来做 SSA 在分类法中的定位

### 【参考】Let's Think Frame by Frame / Think-then-Act
- 各种 chain-of-thought / planning 工作
- 用途: 对比 SSA "无规划"的设计选择

---

## 2. 检索增强生成 (RAG)（技术对比）

### 【参考】Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- 作者: Lewis et al.
- 会议: NeurIPS 2020
- 核心: RAG 的原始论文
- **SSA 差异**: RAG 是"检索-生成"，SSA 是"激活-涌现"。RAG 检索 top-K 文档喂给 LLM；SSA 激活的是"经历过的事"，且激活有辐射（联想）。RAG 是无状态的，这点跟 SSA 一样；但 RAG 没有时间、没重要性权重、没"空间形状"概念。

### 【参考】Self-RAG / Adaptive-RAG
- 各种自适应检索的工作
- 用途: 对比 SSA 的"何时激活多少"

### 【参考】Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- 作者: Asai et al.
- 会议: arXiv 2023 / ICLR 2024
- 核心: 模型自己决定何时检索、如何批判检索结果
- **SSA 差异**: Self-RAG 仍是知识检索，SSA 是经历激活

### 【参考】Adaptive-RAG
- 作者: Jeong et al.
- 会议: NAACL 2024
- 核心: 用 Question Complexity 学习检索策略（**v0.3 修正：不是 Q-Learning**）
- **SSA 差异**: 自适应检索 vs 空间激活

### 【必读·v0.3 新增】A-MEM (Agentic Memory)
- 核心: agent 记忆系统，实现记忆间的关联链接与邻居联取
- **SSA 关系**: A-MEM 的关联链接与 SSA 的"辐射激活"概念相似。SSA 的辐射是更简单的固定深度 KNN 扩展，A-MEM 有动态链接维护。**SSA 不声称辐射激活是首创。**

### 【必读·v0.3 新增】HippoRAG
- 作者: Gutiérrez et al.
- 会议: NeurIPS 2024
- 核心: 受海马体启发的知识图谱检索，关联图结构
- **SSA 关系**: HippoRAG 用显式图结构做关联检索，SSA 用向量空间的隐式邻近性。SSA 更简单但可能表达力较弱。

### 【必读·v0.3 新增】Synapse (2026.01)
- 核心: episodic-semantic memory + spreading activation
- **SSA 关系**: **这是与 SSA 最接近的先行工作。** Synapse 已采用"情景-语义记忆 + 扩散激活"，与 SSA 的"痕迹空间 + 辐射激活"高度重叠。**SSA 相对 Synapse 的差异性（如有）需要明确论证，不能声称 first。**

---

## 3. 情绪计算（理论对比）

### 【背景】Affective Computing
- 作者: Rosalind Picard
- 年份: 1997 (MIT Press)
- 核心: 情绪计算学科奠基
- **SSA 差异**: Picard 关注"识别/模拟人类情绪"，SSA 关注"AI 自身情绪作为涌现副现象"

### 【背景】Emotion in the Human Brain
- 作者: various
- 用途: 心理学对情绪的界定，给 SSA 的情绪讨论提供理论支撑

### 【背景】Plutchik's Wheel of Emotions
- 核心: 8 种基础情绪组合出复合情绪
- 用途: SSA 论文里讲"300 种情绪"时引用

### 【参考】Appraisal Theory of Emotion
- 核心: 情绪 = 对事件的认知评估
- **SSA 关联**: SSA 的"情绪涌现"跟 appraisal theory 有共鸣——情绪不是状态，是对信号-痕迹偏差的反应

---

## 4. 哲学/认知科学（理论锚点）

### 【必读】The Embodied Mind
- 作者: Francisco Varela, Evan Thompson, Eleanor Rosch
- 年份: 1991
- 核心: Enactivism，认知是"具身行动"的涌现
- **SSA 关联**: SSA 跟 enactivism 高度共鸣——"她"不是表征世界的模型，而是通过信号-痕迹交互涌现的存在。**这是 SSA 的哲学根基。**

### 【必读】The Extended Mind
- 作者: Andy Clark, David Chalmers
- 年份: 1998 (Analysis)
- 核心: 认知过程延展到外部环境（如笔记本）
- **SSA 关联**: SSA 的 $\mathcal{S}$ 是"外部认知资源"——她的认知不全是 LLM 内部的，痕迹空间是延展的认知器官。

### 【参考】The Free Energy Principle
- 作者: Karl Friston
- 核心: 大脑作为预测机器，最小化预测误差
- **SSA 关联**: SSA 的"内心独白"可视为某种"自由能最小化"——通过反复处理空间中的痕迹来"消化"未整合的信号。

### 【参考】Being and Time
- 作者: Martin Heidegger
- 核心: 此在 (Dasein) 的时间性
- **SSA 关联**: SSA 的"无状态但有过去"跟海德格尔对时间性的讨论有共鸣——过去不是"曾经发生的现在"，而是此在的存在方式。**这条引用能让论文有哲学分量。**

---

## 5. AI 陪伴产品（工程对比）

### 【参考】Replika 相关文献
- 商业产品，学术论文少
- 关注点: 用户对 Replika 的依恋行为（有些 HCI 论文研究过）

### 【参考】Character.ai 技术披露
- 商业产品，无公开论文
- 但有一些技术博客

### 【参考】Pi (Inflection AI)
- Inflection 有一些公开访谈/博客
- 关注点: "共情 AI"的工程实现

### 【参考】小冰 / 星野 / 筑梦岛
- 国内 AI 陪伴产品
- 几乎无学术论文
- 但有用户研究报告

**SSA 跟所有这些的差异**: 它们都是"人设 + 记忆增强"，没有"无预设涌现"的设计。SSA 是范式级的差异。

---

## 6. Vector Embedding & ANN（技术基础）

### 【参考】Sentence-BERT
- 作者: Reimers, Gurevych
- 会议: EMNLP 2019
- 用途: 痕迹向量化

### 【参考】HNSW (Hierarchical Navigable Small World)
- 作者: Malkov, Yashunin
- 会议: TPAMI 2018
- 用途: 空间索引

### 【参考】FAISS / Milvus / Chroma
- 工程实现

---

## 7. 已识别的研究 Gap（SSA 填补的）

读完上述文献后，SSA 填补的 gap 应该这样讲：

1. **现有 LLM agent 都有内部状态**（MemGPT 的工作记忆、Generative Agents 的反思/规划、Replika 的情绪向量）。SSA 严格无状态。
2. **现有记忆系统是时间线组织**。SSA 是空间几何组织。
3. **现有 AI 陪伴产品基于人设 prompt**。SSA 基于无预设涌现。
4. **现有 agent 重置即失忆/死亡**。SSA 的"重置"是清除 LLM 上下文，空间 $\mathcal{S}$ 仍可保留/迁移。
5. **现有情绪计算是识别/模拟人类情绪**。SSA 的情绪是涌现副现象，不可指令、不可预设。
6. **现有范式存在状态-行为确定性映射缺陷**（论文 §3）。SSA 通过消除状态本身解决此缺陷。

**核心贡献**：SSA 是第一个严格无状态的、基于空间激活的 LLM agent 范式，证明"无预设人格 + 无内部状态"下仍能涌现出连续、有情绪、有个性的对话行为。

---

## 8. 最新工作补充 (2024-2025)

### 【参考】LLM Agent Memory Surveys (2024)
- 多篇 2024 年的 LLM agent memory 综述
- 关注: 分类法、benchmark
- 用途: SSA 在分类法中的定位

### 【参考】LongMemEval / Memory Benchmark (2024-2025)
- 评估 LLM 长期记忆的 benchmark
- 用途: SSA 评估方法可借鉴

### 【参考】Companion AI 用户研究 (2024-2025)
- CHI/CSCW 2024-2025 关于 Replika/Character.ai 用户依恋的研究
- 用途: SSA 的"数字生命"定位 vs 现有产品的"陪伴服务"定位

### 【参考】Agentic Memory Frameworks (2024-2025)
- Letta (前 MemGPT 团队) 的后续工作
- LangGraph / CrewAI 的记忆模块
- 用途: 工程对比

---

## 8. 待读论文清单

- [ ] Generative Agents — 必须精读
- [ ] MemGPT — 必须精读
- [ ] Extended Mind — 必须读
- [ ] Embodied Mind 至少读摘要 + 关键章节
- [ ] 至少 1 篇 HCI 关于 Replika 用户依恋的论文
- [ ] 至少 1 篇 appraisal theory 综述

---

## 9. 投稿目标（待定）

候选：
- UIST / CHI（HCI 偏向）
- ACL / EMNLP（NLP 偏向，需更多 NLP 评估）
- NeurIPS / ICML（偏算法，需更多形式化和实验）
- arXiv 预印本先发（不挑，先占坑）

**建议**: 先发 arXiv 占坑，再根据实验完整度决定是否投顶会。
