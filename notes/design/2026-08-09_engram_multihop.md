# ENGRAM 多跳靶子 A+B

ENGRAM 实现 `hdsiV7.md` §5.10.2 的类型化有向转移幂，不替换现有 `legacy-ssa-a0` 单跳路径。

## 已落地

- `EngramRelation` 固定为论文九类关系；反向关系必须有独立 `EngramEdge`，系统不对称化。
- `engram_nodes` / `engram_edges` 使用 SQLite append-only substrate，边保留 `valid_from/to` 双时序与证据字段。
- `propagate_engram` 采用 `matrix[target][source]` 列随机约定，支持 `truncated_power`、`ppr`、`bounded_active`。
- 每个 `EngramActivation` 保留 `hops`、typed `path` 和 `is_assertable`。
- typed path 的平行边选择按实际 `support * relation_gate * trust * likelihood`，不是裸 support；`hops` 是保留传输路径长度，不承诺最短图距离。
- query seed 对应的旧 trace 会被强制并入有界图；未知 seed 被过滤并返回空结果，避免向量窗口边界造成硬失败。
- 多跳结果只作为召回证据；只有 seed 或带直接证据事件的单跳路径可断言，多跳路径必须降级为假设/疑问。
- `EngramActivationService` 可从现有 traces/trace_links 幂等投影图，再以向量召回分数作为 `pi_0`。
- `[engram].enabled=false` 默认关闭，保留 SSA-A0 消融臂。

## 尚未启用

抽取入图、在线 prompt admission、社区摘要和 `engram_query` 工具属于后续 C-E 阶段；在证据门控与 path-grounded recall 评估接入前不改变生产检索路径。
