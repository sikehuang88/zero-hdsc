# DeepSeek V4 API 调研与 SSA 适配约定

- 调研日期：2026-07-25
- 文档语言：DeepSeek 官方中文文档
- 适配范围：OpenAI Chat Completions 接口、LiteLLM 传输层、SSA 任务路由
- 官方入口：<https://api-docs.deepseek.com/zh-cn/>

本文记录当前 DeepSeek V4 API 合同及 SSA 的实现选择。官方规格变化时，先更新
本文的参数表与合同测试，再调整生产配置。

## 1. 结论摘要

1. 官方模型 ID 是 `deepseek-v4-flash` 与 `deepseek-v4-pro`。SSA 经 LiteLLM
   调用时使用 `deepseek/deepseek-v4-flash` 与
   `deepseek/deepseek-v4-pro`，前面的 `deepseek/` 是 provider 路由前缀，
   发给 DeepSeek 的模型语义没有变化。
2. 两个模型的上下文长度均为 1M tokens，最大输出均为 384K tokens
   （393,216）。一次调用仍需满足输入与输出总长度的上下文限制。
3. 两个模型均支持思考和非思考模式，官方默认值是
   `thinking.type=enabled`。思考强度仅取 `high` 或 `max`。
4. 思考模式下，`temperature`、`top_p`、`presence_penalty`、
   `frequency_penalty` 均无调节效果。SSA 在思考请求中省略采样参数。
5. 标准 endpoint 是 `https://api.deepseek.com`。strict function tools、
   assistant 对话前缀续写以及 FIM 使用 `https://api.deepseek.com/beta`。
6. JSON Output 要求 `response_format={"type":"json_object"}`，同时 system
   或 user prompt 必须出现 `json` 字样。官方记录了偶发空 `content` 的情形，
   因此 SSA 对空响应和非法 JSON 做有界重试。
7. Tool Calls 仅接受 function，单次最多 128 个。思考模式发生工具调用后，
   后续请求必须完整回传对应 assistant 消息的 `reasoning_content`。
8. `user_id` 参与内容安全、KV Cache 和调度隔离，字符集为
   `[A-Za-z0-9_-]`，长度 1 到 512，字段中不放个人信息。
9. `deepseek-chat` 与 `deepseek-reasoner` 已于北京时间
   2026-07-24 23:59 弃用。SSA 将这些旧别名视为配置错误，不依赖兼容映射。

## 2. 模型与路由

| 维度 | V4 Flash | V4 Pro |
|------|----------|--------|
| 官方模型 ID | `deepseek-v4-flash` | `deepseek-v4-pro` |
| SSA/LiteLLM ID | `deepseek/deepseek-v4-flash` | `deepseek/deepseek-v4-pro` |
| 上下文 | 1M | 1M |
| 最大输出 | 384K | 384K |
| 思考模式 | enabled / disabled，官方默认 enabled | enabled / disabled，官方默认 enabled |
| reasoning effort | high / max | high / max |
| JSON Output | 支持 | 支持 |
| function Tool Calls | 支持 | 支持 |
| 官方并发上限 | 每账号 2500 | 每账号 500 |
| 2026-07-25 官方价格 | 命中输入 0.02 元、未命中输入 1 元、输出 2 元/百万 tokens | 命中输入 0.025 元、未命中输入 3 元、输出 6 元/百万 tokens |

价格和并发配额属于运营数据，部署前以官方价格与限速页的实时内容为准。

### SSA 路由策略

| SSA 工作负载 | 模型 | 模式 | 原因 |
|--------------|------|------|------|
| Memory extraction | V4 Flash | disabled，temperature 0.3 | 高频、结构化 JSON、低延迟优先 |
| Appraisal | V4 Flash | disabled，temperature 0.3 | 高频、结构化 JSON、可用确定性规则兜底 |
| Self-belief / identity evidence review | V4 Pro | enabled，effort high，无采样参数 | 低频、跨证据推理、审慎性优先 |
| 后续复杂 Agent 链 | V4 Pro | enabled，effort max | 官方对复杂 Agent 场景的建议 |

全局默认 `thinking_mode=disabled` 代表普通主路由；具体服务可在每个
`LLMRequest` 上覆盖。官方 API 的默认 enabled 不应替代项目中的显式设置。

## 3. Endpoint 与鉴权

| 用途 | Base URL | SSA 行为 |
|------|----------|----------|
| 常规 Chat Completions、JSON、普通 tools | `https://api.deepseek.com` | 显式传给 LiteLLM |
| strict tools、assistant prefix | `https://api.deepseek.com/beta` | 检测到对应请求后自动选择 |
| FIM Completion | `https://api.deepseek.com/beta` | 官方能力，当前 HDSC Chat adapter 未暴露 FIM 接口 |
| Anthropic 格式 | `https://api.deepseek.com/anthropic` | 当前 SSA 未走这条传输路径 |

API key 从 `SSA_LLM_API_KEY` 读取；`DEEPSEEK_API_KEY` 是兼容来源。
密钥只进入环境变量或本地 `.env`，doctor 只显示 configured/not set。

## 4. Chat Completions 请求参数

| 参数 | 官方合同 | SSA 处理 |
|------|----------|----------|
| `model` | `deepseek-v4-flash` 或 `deepseek-v4-pro` | 校验 V4 白名单并归一化 LiteLLM provider 前缀 |
| `messages` | 至少一条；支持 system、user、assistant、tool | Pydantic 消息模型保留 tool call、tool result、reasoning 与 prefix 字段 |
| `thinking` | `{"type":"enabled"}` 或 `{"type":"disabled"}`；默认 enabled | 通过 `extra_body` 原样发送，避免中间层丢失语义 |
| `reasoning_effort` | `high` 或 `max`；普通请求默认 high，复杂 Agent 可自动为 max | 思考模式下通过 `extra_body` 原样发送 |
| `max_tokens` | 最大 384K，且受 1M 总上下文限制 | 配置校验 1..393216；按任务使用更小上限 |
| `response_format` | `text` 或 `json_object`，默认 text | `json_schema` 存在时发送 `json_object` 并在本地解析校验 |
| `temperature` | 0..2，默认 1 | 仅非思考模式发送；与 `top_p` 二选一 |
| `top_p` | 0..1，默认 1 | 仅非思考模式发送；与 `temperature` 二选一 |
| `stop` | 单个字符串或最多 16 个字符串 | 请求有值时透传 |
| `stream` | SSE 增量，末尾 `data: [DONE]` | 当前 adapter 使用非流式完成接口 |
| `stream_options.include_usage` | stream=true 时在结束前追加完整 usage chunk | 当前非流式路径不使用 |
| `tools` | 最多 128 个，仅 `type=function` | 校验工具类型、函数对象和函数名 |
| `tool_choice` | 通用接口支持 `none`、`auto`、`required` 或指定函数 | 非思考模式完整支持；思考模式仅接受缺省/`auto`/`none` |
| `logprobs` | 返回输出 token 对数概率 | 显式请求时透传 |
| `top_logprobs` | 0..20，要求 `logprobs=true` | 请求模型层校验组合后透传 |
| `user_id` | `[A-Za-z0-9_-]{1,512}` | 通过 `extra_body` 发送稳定的非个人标识 |
| `frequency_penalty` | deprecated，传入后无效果 | 省略 |
| `presence_penalty` | deprecated，传入后无效果 | 省略 |
| `seed` | 当前官方 Chat schema 未列出 | 显式传入时拒绝，避免产生虚假的可复现承诺 |

### 消息字段

- system/user 消息使用文本 `content`。
- assistant 的 `content` 可为 null，例如只返回 tool calls 的轮次。
- assistant 可携带 `reasoning_content`、`tool_calls` 和 Beta `prefix`。
- tool 消息必须包含 `tool_call_id` 与结果 `content`。
- prefix completion 要求最后一条消息角色为 assistant 且 `prefix=true`，并走
  Beta endpoint。

## 5. 思考模式合同

OpenAI 格式的请求示意：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "..."}],
  "reasoning_effort": "high",
  "thinking": {"type": "enabled"}
}
```

OpenAI Python SDK 要把 `thinking` 放进 `extra_body`。SSA 对 LiteLLM 采用相同
策略，并把 `reasoning_effort` 也放入 `extra_body`，从而保留 V4 的 high/max
强度，而不是只让中间层把它解释成开关。

兼容映射由 DeepSeek 服务端定义：`low`、`medium` 映射到 `high`，`xhigh`
映射到 `max`。SSA 配置只接受官方原生值 `high|max`，使实验记录保持明确。

响应中的思维链位于 `message.reasoning_content`，最终答案位于
`message.content`：

- 普通多轮对话未发生工具调用时，旧 `reasoning_content` 在下一轮会被忽略，
  SSA 可只保留最终内容。
- 某轮发生工具调用时，assistant 的 `reasoning_content` 与 `tool_calls` 必须作为
  同一条历史消息完整回传。遗漏会得到 HTTP 400。
- 2026-07-26 真实 API 验证显示，思考模式会以 HTTP 400 拒绝 `required` 和
  指定函数形式的 `tool_choice`；缺省/`auto` 与 `none` 可用。
- `usage.completion_tokens_details.reasoning_tokens` 用于记录本次思考 token。

## 6. JSON Output

启用条件与处理顺序：

1. 请求发送 `response_format={"type":"json_object"}`。
2. system 或 user prompt 明确出现 `json` 字样，并提供期望结构或示例。
3. `max_tokens` 为完整 JSON 留出空间；`finish_reason=length` 时按截断响应处理。
4. 收到内容后先判空，再做 JSON parse，最后做领域 Pydantic/schema 校验。
5. 空 `content` 或非法 JSON 最多按 `json_retry_count` 重试；超过上限后进入该
   业务服务的显式降级或错误路径。

`response_format=json_object` 保证 JSON 语法，不代表业务字段、范围、证据 ID
一定符合领域约束，因此服务层校验仍是必需步骤。

## 7. Tool Calls 与 strict 模式

普通 tools 的约束：

- `tools` 最多 128 个，并且 `type` 只取 `function`。
- 函数名匹配 `[A-Za-z0-9_-]{1,64}`。
- `tool_choice` 缺省时，无 tools 等价于 none；有 tools 等价于 auto。
- 模型生成的 `function.arguments` 是 JSON 字符串，执行工具前仍要解析、校验并
  只传入 schema 声明的参数。

strict tools 的附加约束：

1. endpoint 使用 `https://api.deepseek.com/beta`。
2. 请求中的每一个 function 均设置 `strict=true`，禁止混用 strict 与普通函数。
3. 每个 object 的所有 properties 都列入 `required`。
4. 每个 object 设置 `additionalProperties=false`。
5. 官方列出的 strict JSON Schema 类型为 object、string、number、integer、
   boolean、array、enum、anyOf。

SSA 检测任一 strict function 后验证全量一致性并切换 Beta endpoint。思考模式
与非思考模式均可使用 strict tools。

## 8. `user_id`、KV Cache 与请求隔离

DeepSeek 的上下文硬盘缓存对所有 API 用户默认开启，无需单独开关。后续请求
完整匹配已落盘的缓存前缀单元时产生命中。缓存是尽力而为，构建通常为秒级，
未继续使用的缓存一般在数小时到数天内清理。

`user_id` 同时用于：

- 业务用户的内容安全隔离；
- KV Cache 隔离与隐私管理；
- 调度与并发隔离。

HDSC 当前是单用户系统，使用稳定的 `hdsc-primary`。部署迁移时保持该值稳定，
才能持续获得同一隔离域内的前缀复用；该值不编码姓名、邮箱、手机号等信息。

缓存观测字段：

| 字段 | 含义 |
|------|------|
| `usage.prompt_cache_hit_tokens` | 输入中命中上下文缓存的 tokens |
| `usage.prompt_cache_miss_tokens` | 输入中未命中的 tokens |
| `usage.prompt_tokens` | hit + miss |
| `usage.completion_tokens` | 最终输出及思考相关 completion tokens |
| `usage.completion_tokens_details.reasoning_tokens` | 思维链 tokens |

`LLMResponse.cache_hit_ratio` 以 `hit / (hit + miss)` 计算本次请求的缓存命中率；
provider 未返回缓存 token 统计时为 `0.0`。

提高命中率的实现规则是固定 system prompt、schema 序列化和历史消息的稳定前缀，
把每轮新增、变化最大的数据集中放到最后的 user 消息。所有 system 消息必须组成
连续的消息开头区段；adapter 保持调用方顺序，不做可能改变语义的自动重排，并拒绝
出现在 user、assistant 或 tool 之后的 system 消息。缓存只复用输入计算，不会固定
输出；非思考模式的采样随机性仍按请求参数生效。

## 9. 响应与停止原因

adapter 需要保留以下响应字段用于业务判断和观测：

- `message.content`、`message.reasoning_content`、`message.tool_calls`；
- `choice.logprobs.content` 与 `choice.logprobs.reasoning_content`；
- `finish_reason`：`stop`、`length`、`content_filter`、`tool_calls`、
  `insufficient_system_resource`；
- `model` 与 `system_fingerprint`；
- completion、prompt、total、cache hit/miss、reasoning token 统计。

`insufficient_system_resource` 表示推理资源不足导致生成中断。SSA 将它作为
短暂 provider 故障按 `retry_count` 做有界重试，而不是把部分内容当作成功答案。
LiteLLM 1.93 会把这个原生值映射为 `stop`，同时把原值保存在
`choice.provider_specific_fields.native_finish_reason`；adapter 在解析 JSON 前恢复
原生停止原因，避免把资源中断的半截 JSON 误报为普通解析错误。

## 10. HTTP 错误处理矩阵

| HTTP | 官方含义 | SSA 分类/动作 |
|------|----------|---------------|
| 400 | 请求体格式错误；也可能是 thinking tool turn 缺少 reasoning content | invalid request，修正本地消息合同 |
| 401 | API key 错误 | authentication，停止当前调用并报告密钥配置状态 |
| 402 | 余额不足 | insufficient balance，直接暴露独立错误类型 |
| 422 | 请求参数错误 | invalid request，记录 provider 消息与模型 |
| 429 | 速率或并发达到上限 | rate limited，使用传输层退避/重试策略 |
| 500 | 服务内部故障 | provider unavailable，有界重试 |
| 503 | 服务负载过高 | provider unavailable，有界重试 |

DeepSeek 可能在等待推理期间向非流式连接发送空行，向流式连接发送
`: keep-alive` 注释；自定义 HTTP 解析器要忽略这些保活内容。请求 10 分钟后仍
未开始推理时，服务端会关闭连接。SSA 当前默认客户端 timeout 为 120 秒，避免
后台任务长期占用 worker。

## 11. 配置合同

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `SSA_LLM_API_KEY` / `DEEPSEEK_API_KEY` | 空 | DeepSeek API secret，前者优先 |
| `SSA_LLM_MODEL` | `deepseek/deepseek-v4-flash` | 普通主模型 |
| `SSA_LLM_REASONING_MODEL` | `deepseek/deepseek-v4-pro` | 复杂推理模型 |
| `SSA_LLM_BASE_URL` | `https://api.deepseek.com` | 官方 OpenAI endpoint |
| `SSA_LLM_BETA_BASE_URL` | `https://api.deepseek.com/beta` | strict/prefix endpoint |
| `SSA_LLM_TEMPERATURE` | `0.8` | 非思考默认温度 |
| `SSA_LLM_TOP_P` | 未设置 | 与 temperature 二选一 |
| `SSA_LLM_MAX_TOKENS` | `1024` | 每请求默认输出预算 |
| `SSA_LLM_TIMEOUT_SECONDS` | `120` | transport timeout |
| `SSA_LLM_RETRY_COUNT` | `2` | provider/resource 重试上限 |
| `SSA_LLM_JSON_RETRY_COUNT` | `1` | JSON 空响应/解析失败额外重试 |
| `SSA_LLM_THINKING_MODE` | `disabled` | 普通主路由默认模式 |
| `SSA_LLM_REASONING_EFFORT` | `high` | 思考路由默认强度 |
| `HDSC_DEEPSEEK_USER_ID` | `hdsc-primary` | 安全、缓存、调度隔离标识 |

配置优先级为 CLI override > 环境变量/`.env` > 环境 TOML > defaults TOML。
`hdsc doctor` 显示合并后的模型、endpoint、思考配置、LiteLLM 安装状态与密钥
存在状态，适合在部署前检查实际值；该检查不发送 API 请求。

## 12. LiteLLM 1.93 兼容层约定

SSA 锁定的最低版本为 LiteLLM 1.93.0，并针对以下中间层差异建立合同：

- DeepSeek provider 未显式传 base URL 时会落到 `/beta`，因此每次请求都显式传
  标准或 Beta endpoint。
- 该版本会把 `reasoning_effort` 解释为思考开关，但不保留 V4 原生的 `high|max`
  强度；SSA 通过 `extra_body` 原样发送 `thinking`、`reasoning_effort` 和
  `user_id`。
- 内置模型元数据仍报告 8192 最大输出且 `supports_reasoning=false`，与 V4 官方
  的 384K 输出和思考能力不一致；SSA 不使用这些字段做 V4 上限判断。
- LiteLLM 对未知 `insufficient_system_resource` 做标准化时会变成 `stop`；SSA
  读取 `native_finish_reason` 恢复官方语义。
- 通用 `exclude_defaults` 序列化会递归删除历史 tool call 的
  `type="function"`，并删除 assistant 的 `content=null`。SSA 使用显式消息
  serializer 保留这两个必需字段。

已用 LiteLLM 1.93.0 的真实 `ModelResponse` 类型完成无网络验证，覆盖资源中断
重试、JSON 解析、缓存 token、reasoning token 和两组 logprobs。

### 真实 API 冒烟验证（2026-07-26）

| 路径 | 结果 |
|------|------|
| V4 Flash + thinking disabled | 官方 endpoint 鉴权成功，正常返回并解析 cache hit/miss |
| V4 Pro + thinking high | 正常返回最终答案、`reasoning_content` 与 reasoning token |
| V4 Pro thinking tool 两段调用 | 首段返回 tool call；完整回传 reasoning/tool history 后第二段正常结束，并产生 KV Cache 命中 |
| thinking + named/required tool_choice | 服务端返回 HTTP 400；adapter 已改为发送前校验 |

冒烟请求均使用极小输出预算，日志和文档未记录 API key。

## 13. 验收清单

- 主模型与 reasoning 模型均通过 V4 白名单校验，旧别名触发配置错误。
- 每次 transport 调用显式携带 API key、base URL、timeout 和 retry count。
- high/max 经 `extra_body` 到达 DeepSeek，请求测试不只断言 LiteLLM 顶层参数。
- thinking enabled 请求没有 temperature/top_p。
- JSON prompt 包含 `json`，空 content 与非法 JSON 的重试次数有合同测试。
- strict tools 全量 strict，走 Beta endpoint；普通 tools 走官方标准 endpoint。
- thinking tool history 保留 `reasoning_content`。
- usage 解析覆盖 cache hit/miss 与 reasoning tokens。
- logprobs 同时保留最终内容与 reasoning content 的 token 概率。
- 400、401、402、422、429、500、503 以及
  `insufficient_system_resource` 均有确定的错误映射测试。
- 线上指标按 model、purpose、finish reason、cache hit ratio、reasoning tokens、
  latency 和 retry count 分组，避免只观察总成功率。

## 14. 官方资料

- [DeepSeek API 文档入口](https://api-docs.deepseek.com/zh-cn/)
- [DeepSeek V4 发布说明](https://api-docs.deepseek.com/zh-cn/news/news260424)
- [模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
- [Chat Completions 参数](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)
- [思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)
- [JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode)
- [Tool Calls 与 strict](https://api-docs.deepseek.com/zh-cn/guides/tool_calls)
- [上下文硬盘缓存](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)
- [限速与 user_id 隔离](https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit)
- [错误码](https://api-docs.deepseek.com/zh-cn/quick_start/error_codes)
- [对话前缀续写](https://api-docs.deepseek.com/zh-cn/guides/chat_prefix_completion)
- [FIM Completion](https://api-docs.deepseek.com/zh-cn/guides/fim_completion)
- [列出模型](https://api-docs.deepseek.com/zh-cn/api/list-models)
