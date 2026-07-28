# 知应 AI Agent 工具调用与 MCP 实施计划

> 目标：将当前多角色问答系统升级为具备工具调用、任务状态、人工审批和标准 MCP 接入能力的多 Agent 业务协同系统。  
> 原则：保持现有 API 可用，分阶段实施，每一步都可独立测试、验收和回滚。  
> 更新日期：2026-07-28

## 总体路线

```text
整理现有工具层
  → 增加业务工具
  → 让 Agent 自主调用工具
  → 增加多 Agent 结果综合
  → 增加任务状态与人工审批
  → 建立标准 MCP Server
  → 接入 MCP Client
  → 完善评测、监控和前端展示
```

## 第一阶段：整理现有工具层

### 目标

消除当前自定义 `mcp/` 目录与标准 MCP SDK 之间的概念混淆，同时保留已有的缓存、超时、熔断、降级和统计能力。

### 主要任务

- 将 `mcp/tool_manager.py` 移动到 `tooling/tool_runtime.py`。
- 将 `MCPToolManager` 重命名为 `ToolRuntime`。
- 将 `mcp/knowledge_base.py` 移动到 `tooling/legacy_knowledge_base.py`。
- 更新项目内所有导入路径、类型名和测试。
- 保持 `/chat`、`/search`、`/knowledge/*` 的外部 API 契约不变。
- 为后续安装官方 `mcp` Python SDK 清理命名空间。

### 验收标准

- [ ] 原有 58 个测试全部通过。
- [ ] 项目内不再将本地工具框架称为 MCP。
- [ ] 工具缓存、超时、熔断、fallback 和统计行为不变。
- [ ] API 请求和响应格式不变。
- [ ] `git diff --check` 通过。

## 第二阶段：增加 Mock 业务数据和工具

### 目标

建立可以脱离 LLM 独立运行的业务工具层，为 Agent Tool Calling 和 MCP 接入提供稳定基础。

### 首批工具

| 工具 | 类型 | 执行规则 |
|---|---|---|
| `search_knowledge` | 只读 | 自动执行 |
| `query_order` | 只读 | 自动执行 |
| `query_payment` | 只读 | 自动执行 |
| `query_service_status` | 只读 | 自动执行 |
| `create_ticket` | 低风险写入 | 自动执行并记录事件 |
| `create_refund_draft` | 中风险写入 | 只创建草稿 |
| `execute_refund` | 高风险写入 | 必须人工审批 |

### 数据模型

- `Order`
- `PaymentRecord`
- `ServiceStatus`
- `Ticket`
- `RefundDraft`
- `ToolExecution`
- `ApprovalRequest`

### 主要任务

- 使用 Pydantic 为每个工具定义输入和输出 Schema。
- 使用内存仓库或 SQLite 构建可重复的 Mock 数据。
- 为工具定义风险等级、允许调用的 Agent 和超时配置。
- 为所有写操作增加 `idempotency_key`。
- 统一工具成功、失败、重试和不确定结果的返回结构。
- 增加参数校验、权限校验和审计记录。

### 验收标准

- [ ] 每个工具可以脱离 Agent 单独调用。
- [ ] 每个工具都有严格的输入输出 Schema。
- [ ] 非法订单号、金额和状态转换会被拒绝。
- [ ] 重复调用相同幂等键不会重复执行写操作。
- [ ] 工具异常不会直接泄露内部堆栈或敏感数据。
- [ ] 所有工具具有单元测试。

## 第三阶段：给 Agent 增加 Tool Calling

### 目标

让 Agent 根据用户问题自主选择工具、生成参数、读取结果，并决定继续调用工具或生成最终回答。

### 新执行流程

```text
用户消息
  → Agent 获取允许使用的工具
  → 模型选择工具和参数
  → ToolRuntime 校验并执行
  → 工具结果返回模型
  → 模型继续调用工具或输出最终结果
```

### Agent 工具权限

| Agent | 允许使用的工具 |
|---|---|
| GeneralAgent | 知识检索、查询工单、创建工单 |
| TechnicalAgent | 知识检索、服务状态、技术工单 |
| BillingAgent | 订单查询、支付查询、退款草稿 |
| BillingAgent + 审批上下文 | 执行退款 |

### 主要任务

- 为 `BaseAgent` 增加受控 Tool Calling 循环。
- 按 Agent 类型绑定允许使用的工具。
- 设置最大工具调用步数，建议初始值为 5。
- 设置单工具超时和单请求总超时。
- 拒绝模型调用未授权工具。
- 将工具结果作为明确分区的可信上下文返回模型。
- 对工具失败、超时和参数错误提供可解释 fallback。
- 记录每次工具选择、参数、结果和耗时。
- 将固定的 API 层知识检索逐步迁移为 Agent 可选择的工具，同时保留兼容开关。

### 验收场景

用户输入：

```text
订单 A123 被重复扣款了。
```

预期轨迹：

```text
BillingAgent
  → query_order
  → query_payment
  → 发现重复支付
  → create_refund_draft
  → 返回等待审批的处理结果
```

### 验收标准

- [ ] Agent 能根据问题正确选择工具。
- [ ] 工具参数通过 Schema 校验。
- [ ] Agent 不能调用未授权工具。
- [ ] 工具循环达到步数上限时安全停止。
- [ ] 工具超时不会导致整个服务永久阻塞。
- [ ] 工具结果会影响 Agent 的最终回答。
- [ ] 工具轨迹可记录、查询和测试。

## 第四阶段：增加多 Agent 结果综合

### 目标

取消多个 Agent 回复的直接字符串拼接，增加结构化输出、事实合并、冲突检测和统一行动决策。

### 统一结果模型

```text
AgentWorkResult
├── confirmed_facts
├── assumptions
├── tool_executions
├── recommended_actions
├── unresolved_questions
├── requires_approval
└── requires_human
```

### 新协作流程

```text
                  ┌→ TechnicalAgent ─┐
用户问题 → 路由器 ┤                   ├→ SynthesisNode → 统一回复
                  └→ BillingAgent ───┘
```

### 综合节点职责

- 合并重复事实。
- 区分已确认事实、模型推测和待核实信息。
- 检测不同 Agent 的结论冲突。
- 汇总工具调用及执行结果。
- 确定后续行动和责任 Agent。
- 判断是否需要人工审批或升级。
- 生成一份统一、清晰的最终回复。

### 冲突处理规则

- 实时业务系统结果优先于知识库内容。
- 工具结果优先于模型推测。
- 财务结论必须以订单和支付数据为准。
- 不同可信数据源冲突时停止写操作。
- 无法自动解决的冲突进入人工审核。

### 验收标准

- [ ] 复合问题只返回一份统一回答。
- [ ] 回复能够区分事实和推测。
- [ ] 重复信息会被合并。
- [ ] 冲突不会被静默忽略。
- [ ] 部分 Agent 失败时仍能返回降级结果。
- [ ] 综合节点的输入输出具有独立测试。

## 第五阶段：增加任务状态和人工审批

### 目标

让业务问题在生成回复后继续被跟踪，支持暂停、审批、恢复、执行和关闭。

### 状态机

```text
new
  → processing
  → waiting_user
  → waiting_approval
  → executing
  → resolved
  → closed
```

异常状态：

```text
failed
cancelled
reopened
```

### 持久化对象

- `tickets`
- `ticket_events`
- `approval_requests`
- `tool_executions`
- `ticket_assignments`

### 新增接口

```text
POST /tickets
GET  /tickets/{ticket_id}
GET  /tickets/{ticket_id}/events
POST /tickets/{ticket_id}/approve
POST /tickets/{ticket_id}/reject
POST /tickets/{ticket_id}/resume
POST /tickets/{ticket_id}/close
```

### 主要任务

- 使用 SQLite 完成第一版，后续可迁移到 PostgreSQL。
- 使用代码校验合法状态转换。
- 为任务增加版本号或乐观锁。
- 每次状态变化写入不可变事件记录。
- 高风险工具调用前创建审批请求。
- 审批通过后从安全节点恢复执行。
- 为工具执行结果保存幂等键和外部引用号。
- 服务重启后能够恢复待审批任务。

### 验收标准

- [ ] 每次状态变化都有事件记录。
- [ ] 非法状态转换会被拒绝。
- [ ] 待审批任务可以暂停和恢复。
- [ ] 审批拒绝后不会执行高风险工具。
- [ ] 重复恢复不会重复退款。
- [ ] 服务重启后任务状态不会丢失。
- [ ] 一个会话可以关联一张或多张任务单。

## 第六阶段：建立标准 MCP Server

### 目标

使用官方 MCP Python SDK 将业务工具通过标准协议独立暴露，不再依赖主应用直接导入工具函数。

### 推荐目录

```text
mcp_servers/
├── business_server.py
├── knowledge_tools.py
├── order_tools.py
├── technical_tools.py
└── ticket_tools.py
```

第一版可以只建立一个统一的 `business_server.py`。

### MCP Server 能力

- 使用官方 Python SDK 创建 MCP Server。
- 通过 `list_tools` 返回工具及其 Schema。
- 通过 `call_tool` 执行业务工具。
- 返回结构化成功和错误结果。
- 支持 stdio 用于本地开发。
- 支持 Streamable HTTP 用于服务化部署。
- 在服务端执行身份、权限和风险校验。
- 记录工具调用审计和运行指标。

### 主要任务

- 增加官方 `mcp` Python SDK 依赖并锁定版本。
- 使用 FastMCP 注册只读工具。
- 先完成 `query_order`、`query_payment` 和 `query_service_status`。
- 再接入工单和退款草稿工具。
- 高风险写工具继续受审批服务控制。
- 使用 MCP Inspector 验证工具发现和调用。
- 为 MCP Server 增加 Docker 服务和健康检查。

### 验收标准

- [ ] MCP Server 可以独立启动。
- [ ] MCP Inspector 能发现全部工具。
- [ ] Inspector 可以成功调用只读工具。
- [ ] 主应用不直接导入 MCP Server 内部 handler。
- [ ] MCP Server 返回标准结构化结果。
- [ ] MCP Server 停止时主应用能够明确降级。

## 第七阶段：增加 MCP Client

### 目标

让 Agent 通过标准 MCP Client 发现和调用远程工具，同时保留本地工具模式作为开发与回退路径。

### 客户端流程

```text
连接 MCP Server
  → initialize
  → list_tools
  → 转换为模型可用的工具 Schema
  → Agent 选择工具
  → call_tool
  → 返回结构化结果
```

### 配置

```env
ZHIYING_TOOL_BACKEND=local
ZHIYING_MCP_SERVER_URL=http://mcp-server:8000/mcp
ZHIYING_MCP_CONNECT_TIMEOUT=5
ZHIYING_MCP_CALL_TIMEOUT=30
```

支持模式：

| 模式 | 用途 |
|---|---|
| `local` | 单元测试、快速开发、MCP 故障回退 |
| `mcp` | 标准 MCP Server 调用 |

### 主要任务

- 建立 MCP Client 生命周期管理。
- 完成连接初始化和工具发现。
- 将 MCP Tool Schema 转换为模型工具格式。
- 使用 `call_tool` 执行工具。
- 增加连接复用、超时、重试和重连。
- 缓存工具列表并处理工具变更。
- 处理工具名称冲突和结果格式转换。
- 在日志中隐藏 Token、用户敏感数据和工具敏感参数。
- 对 MCP 调用保留现有缓存、熔断和 fallback 能力。

### 验收标准

- [ ] 相同 Agent 用例在 `local` 和 `mcp` 模式下都能通过。
- [ ] MCP 模式不再直接调用本地业务 handler。
- [ ] MCP 断连能够触发明确降级。
- [ ] 工具列表能够动态发现。
- [ ] MCP 调用次数、失败率和延迟可观测。
- [ ] Client 关闭时能够正确释放连接资源。

## 第八阶段：完善评测、监控和前端展示

### 目标

证明 Agent 工具调用与 MCP 接入真实有效，并让完整业务过程可以被用户和面试官直观看到。

### 新增监控指标

- Agent 工具选择正确率。
- 工具参数正确率。
- 平均工具调用步数。
- 非法工具调用次数。
- MCP 连接失败率。
- MCP 工具成功率和 P50/P95。
- 任务完成率。
- 转人工率。
- 审批等待时间。
- 重复执行拦截次数。
- 单请求模型调用次数和 Token 成本。

### 扩展评测集

至少覆盖：

```text
单一技术问题
单一账单问题
技术与账单复合问题
订单不存在
支付记录冲突
工具超时
MCP Server 不可用
信息缺失
审批通过
审批拒绝
重复执行退款
Prompt Injection
任务关闭后重开
```

目标规模：至少 100 条固定测试数据。

### 评测内容

- 工具选择准确率。
- 工具参数准确率。
- 工具轨迹准确率。
- 非法工具调用率。
- 最终任务完成率。
- Agent 路由准确率。
- 多 Agent 综合质量。
- MCP 与本地工具模式的一致性。
- 有无工具调用的回答质量对比。
- 有无查询改写、重排的消融实验。
- 延迟、Token 和调用成本对比。

### 前端改造

- 展示工具调用轨迹。
- 展示多个 Agent 的执行状态。
- 展示任务状态时间线。
- 增加审批和拒绝按钮。
- 展示 MCP Server 在线状态。
- 展示最终处理结果和执行依据。
- 标记超时、fallback 和人工升级。
- 移除与项目无关的个人推广链接。

### 推荐演示场景

用户输入：

```text
订单 A123 重复扣款 299 元，同时登录提示 401。
```

完整演示：

```text
创建任务
  → TechnicalAgent 查询服务与账号状态
  → BillingAgent 查询订单和支付流水
  → 综合节点确认两个独立问题
  → 创建退款草稿
  → 人工点击批准
  → MCP 工具执行 Mock 退款
  → 任务状态变为 resolved
  → 时间线展示完整审计记录
```

### 验收标准

- [ ] 固定评测集不少于 100 条。
- [ ] 能评测正确工具轨迹。
- [ ] 能对比纯回答与工具增强 Agent。
- [ ] Prometheus 指标实际产生数据。
- [ ] 前端可以查看完整执行时间线。
- [ ] 演示场景可以端到端重复运行。
- [ ] README 包含架构、启动方式、评测结果和演示截图。

## 实施优先级

如果以大厂 AI 应用或 Agent 实习为目标：

```text
必须完成：第一至第五阶段
明显加分：第六至第七阶段
展示完善：第八阶段
```

## 推荐 PR 拆分

```text
PR 1：工具层重命名与基础 Schema
PR 2：Mock 业务数据和工具
PR 3：Agent Tool Calling
PR 4：多 Agent 结构化输出与综合节点
PR 5：任务状态、事件和人工审批
PR 6：标准 MCP Server
PR 7：MCP Client、灰度与降级
PR 8：评测、监控、前端和文档
```

## 最终目标

最终系统应形成以下闭环：

```text
用户提出业务问题
  → 多 Agent 分类和协作
  → Agent 自主选择受控工具
  → MCP 执行业务操作
  → 任务状态持久化
  → 高风险操作等待人工审批
  → 审批后恢复执行
  → 结果通知用户
  → 全过程可观测、可评测、可审计
```

项目最终定位：

> 基于多 Agent、工具调用、标准 MCP 和人工审批的业务任务协同平台。

## 参考资料

- [Model Context Protocol Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Python SDK Client 文档](https://py.sdk.modelcontextprotocol.io/client/)
