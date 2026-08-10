# OpenCode 委派集成落地

本次落地遵循 `open集成.md` 的边界：OpenCode core 不修改，HDSC 通过工具注册、HTTP 客户端、Broker 和外部插件接入。

## 已完成

- `OpencodeConfig` 默认关闭，支持 `HDSC_OPENCODE_*` 环境覆盖。
- `opencode_fire` 在并发许可内创建 session，并用 `prompt_async` 投递后立即返回 job；后台按 GET `/message` 轮询，`opencode_check` 读取状态，`opencode_export` 导出有界摘要。
- `request_timeout_seconds` 只约束单次 HTTP 请求；job 总时长由 `job_timeout_seconds` 独立约束，不再把长任务截断在一次 POST 内。
- job 元数据持久化到 `job_store_path`；进程重启时运行中 job 明确标为 `aborted`，终态记录按 `job_retention_seconds` 回收。
- job/status 保留 `conversation_id` 与 assistant `message_id`，供后续 session-to-trace 回写绑定使用。
- 只有 `enabled=true` 且存在 `OPENCODE_SERVER_PASSWORD` 时才注册能力。
- HTTP 请求使用 Basic Auth，输出受 `poll_max_output_chars` 和 ToolKernel 双重截断。
- Tool audit 只保存参数/输出哈希、job/session/state/provider 等元数据，不保存原始 diff。
- Web Gateway 在配置 `HDSC_BROKER_TOKEN` 时挂载 `/api/opencode-broker`：
  - `GET /context`：返回来源明确的派生上下文。
  - `POST /exec`：复用 coding workspace 路径边界执行命令。
  - `POST /write`：复用边界并支持新文件创建。
  - `POST /session-writeback`：追加不可变的 session 完成事件。
- `E:\zerobot\opencode\third_party\opencode` 保存 provider、权限和 `zero-bridge` 插件资产。

## 启用顺序

1. 启动 HDSC Gateway，并设置 `HDSC_BROKER_TOKEN`。
2. 设置 `HDSC_OPENCODE_ENABLED=true`、`OPENCODE_SERVER_PASSWORD` 和 `OPENCODE_SERVER_USERNAME`。
3. 启动仅监听本机的 `opencode serve`，确认端口与 `HDSC_OPENCODE_BASE_URL` 一致。
4. 将 `third_party/opencode` 资产复制到目标 OpenCode 工作区，设置同一个 `HDSC_BROKER_TOKEN`。
5. 用 `opencode_fire` 提交有界任务，再用 `opencode_check` 验证结果；完成事件会进入 HDSC 账本。

默认配置保持关闭，因此没有服务和密码时现有工具注册表零行为变化。
