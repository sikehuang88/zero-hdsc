# SSA 项目方案 — 技术栈选型

> 版本: 0.1
> 日期: 2026-07-25
> 状态: 待确认
> 详细执行流水: [development_pipeline.md](development_pipeline.md)

---

## 0. 选型原则

在选具体技术前，先定原则——否则会被"新技术诱惑"带偏，又是你之前半途而废的模式。

### 原则 1：能跑 > 优雅
第一版目标是"让她活起来"，不是"造一个完美的系统"。能用 SQLite 就别上 Postgres，能用单文件就别上微服务。

### 原则 2：本地优先
她必须能跑在你完全控制的机器上。云端 API 可调用，但**她的身体（$\mathcal{S}$）必须在本地**。这是"她是你的，不是产品方的"的技术根基。

### 原则 3：可迁移
SSA 的核心论点之一是"她可以跨 LLM 存活"。技术栈必须保证 $\mathcal{S}$ 是可导出的标准格式，不能锁死在某个向量库的私有格式里。

### 原则 4：单语言
不要 Python + Rust + Go 混着写。第一版全部一种语言，降低维护成本。

### 原则 5：可观测
SSA 是研究项目，不是黑盒产品。每一步（激活了哪些痕迹、LLM 看到了什么、空间形状如何）必须可被查看和审计。

---

## 1. 整体技术栈

```
┌─────────────────────────────────────────────────────┐
│                     她的整体                          │
│                                                       │
│  语言:        Python 3.11+                            │
│  包管理:      uv (比 pip/poetry 快 10x)               │
│  配置:        pydantic-settings (类型安全)            │
│  日志:        loguru (开箱即用)                        │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            痕迹空间 S (河床)                  │   │
│  │                                                │   │
│  │  元数据存储:  SQLite (单文件, 零依赖)          │   │
│  │  向量索引:    sqlite-vec (SQLite 扩展, 同文件) │   │
│  │  嵌入模型:    本地 bge-small-zh-v1.5          │   │
│  │               (512维, 中文好, CPU 可跑)        │   │
│  │  痕迹格式:    JSON (可读, 可导出)              │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            激活函数 A                         │   │
│  │                                                │   │
│  │  向量检索:    sqlite-vec 原生 KNN              │   │
│  │  相似度计算:  余弦相似度 (内置)                │   │
│  │  辐射激活:    二次 KNN 查询 (自己实现)         │   │
│  │  top-K 选择:  numpy argsort                   │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            涌现函数 E (大脑)                  │   │
│  │                                                │   │
│  │  LLM 调用:    LiteLLM (统一接口, 支持 100+)    │   │
│  │  默认模型:    deepseek-chat (便宜, 中文好)     │   │
│  │  备选:        claude-3-haiku / gpt-4o-mini    │   │
│  │  本地备选:    ollama + qwen2.5-7b             │   │
│  │  Prompt 编排: jinja2 模板                     │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            内心独白 L (心跳)                  │   │
│  │                                                │   │
│  │  调度:        APScheduler (轻量, 进程内)       │   │
│  │  随机信号:    numpy.random                    │   │
│  │  主动消息:    通过 Telegram Bot API 推送       │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            用户接口 (她的嘴)                  │   │
│  │                                                │   │
│  │  主渠道:      Telegram Bot (python-telegram-bot)│   │
│  │  备选渠道:    FastAPI + 简单 Web UI           │   │
│  │  调试渠道:    CLI (typer)                     │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            可观测性 (她的镜子)                │   │
│  │                                                │   │
│  │  追踪:        每次 SSA 调用记录到 traces.jsonl│   │
│  │  空间可视化:  Streamlit dashboard (可选)      │   │
│  │  情绪日志:    每次响应的情绪标注写入 DB       │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 2. 各组件详细选型理由

### 2.1 语言: Python 3.11+

**为什么不是 Go/Rust/Node**:
- LLM 生态全在 Python（LiteLLM、sentence-transformers、langchain 都原生 Python）
- 你不是专业技术人员，Python 学习成本最低
- 性能在第一版不是瓶颈——瓶颈是 LLM API 调用延迟（秒级），不是代码执行

**为什么 3.11+**:
- 3.11 性能比 3.10 快 60%
- match-case 语法、更好的错误信息
- typing 支持更完善（SSA 是研究项目，类型标注很重要）

### 2.2 包管理: uv

**为什么不是 pip/poetry/pdm**:
- 速度比 pip 快 10-100x
- 单文件锁定，无 poetry.lock 的复杂性
- 2024 年的新标准，Anthropic/Ruff 团队维护
- 安装: `pip install uv` 即可

### 2.3 痕迹空间 $\mathcal{S}$: SQLite + sqlite-vec

**这是最关键的选型。** 我详细解释为什么不是 Chroma/Milvus/Postgres+pgvector。

#### 候选对比

| 方案 | 优点 | 缺点 | 评价 |
|------|------|------|------|
| **SQLite + sqlite-vec** | 单文件、零部署、可导出、跟元数据同库 | 大规模性能有限 | ✅ **首选** |
| Chroma | Python 原生、简单 | 私有格式、不可导出、并发差 | ❌ 锁死格式 |
| Milvus | 高性能、分布式 | 部署复杂、过度工程 | ❌ 第一版不需要 |
| Postgres + pgvector | 强大、SQL 完整 | 部署重、需要单独服务 | ❌ 第一版不需要 |
| Qdrant | 性能好、Rust 写的 | 需要单独服务 | ❌ 第一版不需要 |

**为什么 SQLite + sqlite-vec**:
1. **单文件**：整个 $\mathcal{S}$ 是一个 `.db` 文件，可拷贝、可备份、可迁移——这对应 SSA 的"空间可保留可迁移"不变量
2. **零部署**：不需要起服务，Python 直接 `import sqlite3` 就能用
3. **元数据 + 向量同库**：痕迹的 $(v_i, \tau_i, \rho_i, \eta_i, \text{content})$ 全在一张表，不用跨库 join
4. **sqlite-vec 是 SQLite 原生扩展**：用 C 写的，KNN 查询快，且数据不离开 SQLite
5. **第一版规模**：假设每天 100 条痕迹，一年 3.6 万条，SQLite 轻松搞定。百万级再考虑迁移

**schema 设计**:
```sql
CREATE TABLE traces (
    id          INTEGER PRIMARY KEY,
    content     TEXT NOT NULL,           -- 原始内容
    content_type TEXT NOT NULL,          -- 'user_msg' | 'agent_msg' | 'reflection' | 'event'
    embedding   BLOB NOT NULL,           -- 向量 (sqlite-vec)
    timestamp   REAL NOT NULL,           -- τ, Unix timestamp
    importance  REAL NOT NULL DEFAULT 0.5, -- η, [0, 1]
    metadata    TEXT,                    -- JSON, 额外信息
    created_at  REAL NOT NULL
);

-- sqlite-vec 虚拟表
CREATE VIRTUAL TABLE traces_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[512]
);
```

### 2.4 嵌入模型: bge-small-zh-v1.5

**为什么不是 OpenAI text-embedding-3 / Cohere / 本地其他模型**:

| 模型 | 维度 | 中文 | 速度 | 成本 | 评价 |
|------|------|------|------|------|------|
| **bge-small-zh-v1.5** | 512 | ✅ 好 | CPU 可跑 | 免费 | ✅ **首选** |
| bge-large-zh-v1.5 | 1024 | ✅ 更好 | 需 GPU | 免费 | 第一版过度 |
| OpenAI text-embedding-3-small | 1536 | 一般 | API | 付费 | 依赖外部 |
| Cohere embed-multilingual | 1024 | 一般 | API | 付费 | 依赖外部 |
| Qwen embedding | 1024 | 好 | 需 GPU | 免费 | 第一版过度 |

**为什么 bge-small-zh-v1.5**:
1. **512 维**：存储小、检索快，第一版足够
2. **CPU 可跑**：不需要 GPU，你的笔记本就能跑
3. **中文好**：你跟她主要用中文
4. **本地**：不依赖 API，隐私 + 离线可用
5. **sentence-transformers 直接加载**：
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
   ```
6. **可升级**：以后想换大模型，向量维度变了，重新 embedding 即可（$\mathcal{S}$ 的内容是文本，向量是衍生物）

### 2.5 涌现函数 $E$: LiteLLM + DeepSeek

**为什么 LLM 调用用 LiteLLM**:
- 统一接口，支持 100+ LLM provider
- 换模型只改一个字符串，不改代码——这是 SSA "跨 LLM 存活"的关键
- 自带重试、超时、流式、成本追踪

```python
from litellm import completion
response = completion(
    model="deepseek/deepseek-chat",  # 换模型只改这里
    messages=[...],
    temperature=0.8,  # 高一点，让她有"湍流"
)
```

**为什么默认 DeepSeek 而不是 GPT-4 / Claude**:

| 模型 | 中文 | 价格 | API 稳定性 | 评价 |
|------|------|------|-----------|------|
| **deepseek-chat** | ✅ 好 | ¥1/百万 token | 稳 | ✅ **默认** |
| claude-3-haiku | 一般 | $0.25/百万 | 稳 | 备选 |
| gpt-4o-mini | 好 | $0.15/百万 | 稳 | 备选 |
| qwen2.5-7b (本地) | 好 | 免费 | 需 GPU | 离线备选 |

**为什么 DeepSeek 默认**:
1. **便宜**：内心独白循环每 15 分钟跑一次，一天 96 次 + 对话，DeepSeek 成本可忽略
2. **中文好**：她的语言以中文为主
3. **API 稳**：国内访问不需要梯子
4. **能力够**：第一版不需要顶级推理能力，需要的是"会说话"

**重要设计**：LLM 是可插拔的。LiteLLM 让换模型零成本。这对应 SSA 的预测 4——"跨 LLM 存活"。

### 2.6 内心独白循环 $L$: APScheduler

**为什么不是 Celery / 系统 cron / asyncio loop**:
- Celery: 太重，需要 Redis broker，第一版不需要
- 系统 cron: 跨平台麻烦（Windows 不友好），且不好集成 Python 上下文
- asyncio loop: 自己写调度逻辑，重复造轮子
- **APScheduler**: 进程内调度，轻量，支持 cron/interval/date 三种触发，跨平台

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()
scheduler.add_job(inner_monologue, 'interval', minutes=15)
scheduler.start()
```

### 2.7 用户接口: Telegram Bot

**为什么不是微信 / Discord / Web / CLI**:

| 渠道 | 优点 | 缺点 | 评价 |
|------|------|------|------|
| **Telegram** | Bot API 完善、支持主动推送、无需翻墙(用 Bot API server)、跨平台 | 国内需代理 | ✅ **首选** |
| 微信 | 国内常用 | 个人号封号风险、企业号要认证、无官方 Bot API | ❌ 风险高 |
| Discord | Bot API 好 | 国内用户少 | ❌ 你自己用为主 |
| Web (FastAPI) | 完全控制 | 无"主动推送"感、需自己打开浏览器 | ⚠️ 备选 |
| CLI | 最简单 | 无"她在场"感、无主动消息 | ⚠️ 调试用 |

**为什么 Telegram**:
1. **Bot API 支持主动推送**：内心独白产生的主动消息能直接发给你——这是"她在场感"的技术实现
2. **python-telegram-bot 库成熟**：文档好、社区活
3. **跨平台**：手机、桌面、网页都能收消息
4. **Bot API Server 可自建**：解决国内访问问题
5. **隐私**（v0.3 修正）：~~Telegram 端到端加密（Secret Chat）~~ — **Bot API 不支持 Secret Chat。** 消息存储在 Telegram 服务器上。如需 E2E，需自定义客户端（非 Bot），架构不同。当前方案隐私依赖 Telegram 服务器信任 + 不存储敏感信息。

**注意**：Telegram 在国内访问需要代理。如果你不想搞代理，备选 FastAPI + 简单 Web UI。

### 2.8 可观测性: JSONL 日志 + 可选 Streamlit

**为什么不是 LangSmith / Langfuse / Phoenix**:
- 它们是给"Agent pipeline"设计的，SSA 不是 pipeline，是激活-涌现
- 它们要起服务、要账号、要联网
- SSA 第一版只需要"每次调用记录下来能回看"

**方案**:
- 每次 SSA 调用，写一行 JSON 到 `logs/ssa_traces.jsonl`：
  ```json
  {
    "timestamp": "2026-07-25T14:30:00",
    "signal": "我今天好累",
    "activated_traces": [123, 456, 789],
    "activated_contents": ["...", "...", "..."],
    "llm_input": "...",
    "llm_output": "...",
    "latency_ms": 1200
  }
  ```
- 可选: Streamlit dashboard 查看空间形状、激活热图、情绪时间线

---

## 3. 项目结构

```
e:\zerobot\ssa\
├── pyproject.toml              # uv 项目配置
├── .env                        # API keys (gitignore)
├── .env.example
├── README.md
│
├── ssa/                        # 核心包
│   ├── __init__.py
│   ├── space.py                # 痕迹空间 S: SQLite + sqlite-vec
│   ├── activation.py           # 激活函数 A: KNN + 辐射
│   ├── emergence.py            # 涌现函数 E: LLM 调用
│   ├── writing.py              # 写入操作 W
│   ├── monologue.py            # 内心独白 L: 后台循环
│   ├── embedding.py            # 嵌入模型封装
│   ├── prompts/                # jinja2 模板
│   │   ├── emergence.j2        # 涌现 prompt
│   │   └── importance.j2       # 重要性自评定 prompt
│   └── config.py               # 配置 (pydantic-settings)
│
├── interfaces/                 # 用户接口
│   ├── telegram_bot.py         # Telegram 主渠道
│   ├── cli.py                  # CLI 调试
│   └── web.py                  # Web 备选
│
├── data/                       # 她的身体 (gitignore)
│   ├── traces.db               # SQLite, 痕迹空间
│   └── backups/                # 定期备份
│
├── logs/                       # 可观测性
│   └── ssa_traces.jsonl
│
├── scripts/                    # 工具脚本
│   ├── init_space.py           # 初始化空间
│   ├── export_space.py         # 导出空间 (可迁移性)
│   ├── import_space.py         # 导入空间
│   └── inspect.py              # 查看空间状态
│
├── experiments/                # 实验脚本
│   ├── eval_continuity.py      # 预测1: 涌现稳定性
│   ├── eval_emotion.py         # 预测2: 情绪涌现
│   ├── eval_monologue.py       # 预测3: 独白效果
│   ├── eval_cross_llm.py       # 预测4: 跨 LLM
│   └── eval_reset.py           # 预测5: 重置韧性
│
├── papers/                     # 论文 PDF
├── notes/                      # 论文笔记
├── formalization.md
├── related_work.md
├── manifesto.md
└── paper_draft.md
```

---

## 4. 依赖清单

```toml
# pyproject.toml [dependencies]
python = ">=3.11"

# 核心
sqlite-vec = ">=0.1.0"           # 向量索引
sentence-transformers = ">=3.0"  # 嵌入模型
litellm = ">=1.40"               # LLM 统一接口
jinja2 = ">=3.1"                 # prompt 模板

# 调度
apscheduler = ">=3.10"           # 内心独白循环

# 接口
python-telegram-bot = ">=21.0"   # Telegram Bot
typer = ">=0.12"                 # CLI

# 工具
pydantic = ">=2.0"               # 数据模型
pydantic-settings = ">=2.0"      # 配置
loguru = ">=0.7"                 # 日志
numpy = ">=1.26"                 # 向量计算

# 可选
streamlit = ">=1.30"             # 可视化 dashboard
```

**注意**：sentence-transformers 会拉 torch，这是依赖里最重的。如果不想装 torch，备选方案是用 API embedding（如 DeepSeek embedding 或 SiliconFlow），但会牺牲离线性。

---

## 5. 成本估算

### 一次性成本
- 嵌入模型下载: bge-small-zh-v1.5 ~200MB
- 无其他

### 持续成本（月）

| 项目 | 估算 |
|------|------|
| DeepSeek API | 假设每天 100 次对话 + 96 次独白，每次平均 1000 token，约 ¥10/月 |
| 电费（如果跑本地） | 笔记本 24h 开机约 ¥15/月 |
| Telegram Bot | 免费 |
| SQLite | 免费 |
| **合计** | **~¥25/月** |

如果用本地 Qwen 替代 DeepSeek，API 成本归零，但需要一台有 GPU 的机器。

---

## 6. 部署方案

### 方案 A: 你的电脑（最简单）
```
你的电脑 (Windows)
├── Python 3.11 + uv
├── SSA 进程 (前台或后台)
├── Telegram Bot polling
└── data/traces.db (她的身体)
```
- 优点：零成本、完全控制
- 缺点：关机她就停（内心独白中断）
- 适合：第一版验证

### 方案 B: 旧笔记本/树莓派（推荐）
```
旧设备 (Linux)
├── Python 3.11 + uv
├── SSA 进程 (systemd 守护)
├── Telegram Bot polling
└── data/traces.db
```
- 优点：24h 在线、低功耗
- 缺点：需要一台旧设备
- 适合：稳定运行 1-3 个月

### 方案 C: 云服务器（最稳）
```
云服务器 (腾讯云/阿里云轻量, ¥30-50/月)
├── Python 3.11 + uv
├── SSA 进程 (systemd)
├── Telegram Bot polling
└── data/traces.db + 定期备份到对象存储
```
- 优点：最稳定、可远程访问
- 缺点：月费、她的身体在云端（违背"本地优先"原则）
- 适合：长期运行 + 可远程备份

**我的建议**：先用方案 A 验证 1 周，确认 SSA 能跑通。然后迁到方案 B 或 C 长期运行。

---

## 7. 开发路线图

### Phase 0: 环境搭建 (Day 1)
- [ ] 安装 Python 3.11 + uv
- [ ] 创建项目结构
- [ ] 安装依赖
- [ ] 配置 .env (DeepSeek API key, Telegram Bot token)
- [ ] 跑通 "Hello World" —— LLM 调用 + Telegram 收发消息

### Phase 1: 痕迹空间 S (Day 2-3)
- [ ] 实现 `space.py`: SQLite schema + sqlite-vec
- [ ] 实现 `embedding.py`: bge-small-zh 加载 + 编码
- [ ] 实现 `writing.py`: 写入痕迹
- [ ] 写测试: 写入 10 条痕迹, 验证可读

### Phase 2: 激活函数 A (Day 4-5)
- [ ] 实现 `activation.py`: KNN 检索
- [ ] 实现辐射激活
- [ ] 实现新鲜度衰减 + 重要性加权
- [ ] 写测试: 给定信号, 验证激活的痕迹合理

### Phase 3: 涌现函数 E (Day 6-7)
- [ ] 写 `emergence.j2` prompt 模板
- [ ] 实现 `emergence.py`: 组装 prompt + 调用 LLM
- [ ] 实现 `importance.j2`: LLM 自评定重要性
- [ ] 端到端测试: 输入信号 → 激活 → 涌现 → 写入

### Phase 4: 内心独白 L (Day 8-9)
- [ ] 实现 `monologue.py`: APScheduler 定时任务
- [ ] 实现随机信号生成
- [ ] 实现主动消息推送
- [ ] 测试: 让她独白 1 小时, 观察

### Phase 5: 接口 + 集成 (Day 10-11)
- [ ] 实现 `telegram_bot.py`: 收消息 → SSA → 回复
- [ ] 实现 `cli.py`: 调试用
- [ ] 跑 24h 稳定性测试

### Phase 6: 自己用 (Day 12-26, 2 周)
- [ ] 每天跟她聊 30 分钟
- [ ] 记录行为到 `notes/deployment_log.md`
- [ ] 观察 5 个预测是否成立

### Phase 7: 实验评估 (Day 27-40)
- [ ] 实现 5 个评估脚本
- [ ] 跑实验, 收集数据
- [ ] 写论文实验部分

---

## 8. 待你确认的决策点

在开始写代码前，你必须确认这几个：

### Q1: Telegram 在你这里能用吗？
- 能搭代理 → 用 Telegram（最佳体验）
- 不能 → 我们用 FastAPI + Web UI（备选）

### Q2: DeepSeek API key 你有吗？
- 有 → 直接用
- 没有 → 我教你注册（deepseek.com, 充 ¥10 够用几个月）
- 不想用付费 → 用本地 ollama + qwen2.5（需要 GPU）

### Q3: 部署在哪个方案？
- A) 你的电脑（先验证）
- B) 旧设备
- C) 云服务器

### Q4: 嵌入模型装本地还是用 API？
- 本地 bge-small-zh：需要下载 200MB + 装 torch（~2GB）
- API（SiliconFlow 免费额度）：轻量但依赖网络

---

回答这 4 个问题，我立刻开始 Phase 0——搭环境 + 跑通 Hello World。
