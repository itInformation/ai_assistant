# 企业级 AI 知识助手

一个面向企业知识检索与智能问答场景的 AI Agent 项目。项目以阿里云百炼大模型为默认模型服务，围绕 Prompt Engineering、RAG、Tool Calling、ReAct Agent、LangGraph 工作流和可观测性构建，并采用可替换的接口设计支持后续生产化演进。

> 当前进度：Phase 1 — 工程基础已完成

## 快速开始

项目要求 Python 3.11—3.13，并使用 uv 管理依赖：

```bash
cp .env.example .env
uv sync --all-groups
uv run ai-assistant
```

首次运行不需要 API Key。CLI 会加载类型安全配置、初始化结构化日志，并输出
应用健康信息。真实 DashScope 调用将在 Phase 2 接入。

常用质量检查命令：

```bash
uv run ruff check .
uv run black --check .
uv run pytest
```

配置优先从环境变量读取，本地可通过 `.env` 覆盖。`.env` 已被 Git 忽略，
可提交的变量示例位于 `.env.example`。

## 项目目标

本项目不是一次性 Demo，而是一个可测试、可维护、可扩展的企业级 AI 应用工程，主要目标包括：

- 支持 PDF、Word、Markdown 企业文档的摄取与解析。
- 构建基于阿里百炼 Embedding、Milvus Lite、Retriever 和 Rerank 的完整 RAG 链路。
- 实现 Weather、Search、Database 等工具及统一工具注册机制。
- 实现带 Memory、Planning 和 Tool Calling 的 ReAct Agent。
- 使用 LangGraph 编排 Planner、Retriever、Tool、Reviewer、Answer 工作流。
- 记录 Prompt、Response、Token Usage、Tool Trace、Latency 和 Error Log。
- 通过 FastAPI 提供标准 REST API 与 Swagger 文档。
- 使用抽象接口隔离模型、Embedding、向量数据库和外部工具供应商。

## 功能范围

### Prompt Engineering

- System Prompt 与 Prompt Template
- Few-shot 示例
- 结构化 JSON 输出
- Output Parser
- Prompt 版本管理与调试

### RAG

```text
文档加载 → 文档解析 → Chunk → Embedding → Milvus
       → Retriever → Rerank → LLM → Answer
```

### Agent 与工作流

```text
用户问题
   ↓
Planner → Retriever → Tool → Reviewer → Answer
```

进阶版本将演进为 Supervisor、Research Agent、Tool Agent 和 Summary Agent 组成的多 Agent 协作流程。

### 可观测性

- 请求链路 ID
- Prompt 与模型响应日志
- 工具调用轨迹
- 端到端与节点级耗时
- Token 用量
- 异常与重试日志

## 非功能性要求

- 高内聚、低耦合，遵循 SOLID 和 Clean Architecture 思想。
- 所有公共类和函数包含类型注解与 Docstring。
- 业务模块具有单元测试，关键链路具有集成测试。
- 使用 Ruff、Black 和 pytest 保证代码质量。
- 密钥只从环境变量读取，不进入版本控制。
- 外部依赖通过适配器封装，便于替换和测试。
- 本地开发使用 Milvus Lite，生产环境可迁移至 Milvus Standalone、Distributed 或 Zilliz Cloud。

## 技术选型

| 领域 | 技术 | 选择原因 |
| --- | --- | --- |
| 语言 | Python 3.11+ | AI 生态成熟，适合模型、RAG 与 Agent 工程 |
| 依赖管理 | uv + pyproject.toml | 解析和安装速度快，统一环境与项目元数据 |
| 配置 | pydantic-settings | 类型安全、环境变量校验、适合分环境配置 |
| LLM | 阿里云百炼 DashScope | 满足项目约束，支持通义千问及流式输出 |
| Embedding | 百炼 Embedding | 中文语义能力与模型服务保持一致 |
| 向量数据库 | Milvus Lite | 本地零服务依赖，并保持向生产 Milvus 的迁移路径 |
| RAG 编排 | LangChain 组件 | 提供文档与模型生态适配，但核心业务保留自有抽象 |
| Agent 工作流 | LangGraph | 显式状态、节点和条件边，适合复杂流程与可恢复执行 |
| API | FastAPI | 类型驱动、异步友好、自动生成 OpenAPI/Swagger |
| 重试 | tenacity | 声明式退避与重试策略 |
| 测试 | pytest | Python 工程事实标准，插件生态完整 |
| 质量 | Ruff + Black | 快速静态检查与一致格式化 |
| 可观测性 | structlog + 自定义 Trace | 结构化日志便于检索，并可平滑接入 OpenTelemetry |

### 框架使用边界

- LangChain 用于复用成熟的文档和模型生态能力，不让领域逻辑直接依赖其具体实现。
- LangGraph 用于有状态、可分支、可检查点的 Agent 工作流；普通线性流程保持为简单 Python 服务。
- 不引入 LlamaIndex，避免在同一阶段叠加功能重合的 RAG 框架和认知成本。

## 整体架构

```text
┌─────────────────────────────────────────────┐
│ API / CLI                                   │
├─────────────────────────────────────────────┤
│ Application                                 │
│ RAG Service │ Agent Service │ Ingestion     │
├─────────────────────────────────────────────┤
│ Domain                                      │
│ Models │ Ports │ Prompt │ Workflow State    │
├─────────────────────────────────────────────┤
│ Infrastructure                              │
│ DashScope │ Milvus │ Tools │ Logging        │
└─────────────────────────────────────────────┘
```

依赖方向由外向内：基础设施实现领域端口，领域与应用层不依赖具体厂商 SDK。这个结构与 Java 后端中的接口、Service、Repository/Adapter 分层类似。

## 规划目录

```text
.
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
├── data/
├── docs/
├── logs/
├── milvus/
├── scripts/
├── src/
│   └── enterprise_ai_assistant/
│       ├── api/
│       ├── agent/
│       ├── config/
│       ├── core/
│       ├── embedding/
│       ├── graph/
│       ├── llm/
│       ├── memory/
│       ├── models/
│       ├── observability/
│       ├── prompt/
│       ├── rag/
│       ├── tools/
│       └── vectorstore/
└── tests/
    ├── unit/
    └── integration/
```

## 开发 RoadMap

- Phase 0：需求、架构、技术选型、目录与 Git 规划。
- Phase 1：uv 工程、配置、日志、质量工具和最小 Demo。
- Phase 2：DashScope LLM 抽象、聊天、流式、超时与重试。
- Phase 3：Prompt 模板、Few-shot、结构化输出和版本管理。
- Phase 4：Embedding 接口与百炼实现。
- Phase 5：向量存储接口与 Milvus Lite 实现。
- Phase 6：文档摄取、Chunk、Retriever、Rerank 和 RAG 回答。
- Phase 7：工具抽象、注册中心及三个业务工具。
- Phase 8：ReAct、Memory、Planning 和 Tool Calling Agent。
- Phase 9：LangGraph 基础版与进阶版工作流。
- Phase 10：FastAPI 服务与完整可观测性。
- Phase 11：容器化、部署、性能优化、文档及面试材料。

## 初步关键设计

### Chunk 策略

初始候选值为 800 个中文字符、120 字符 overlap。它在语义完整性、召回粒度和模型上下文成本之间取得可解释的平衡。Phase 6 将通过文档类型、检索命中率和回答评测验证，而不是把经验值直接当作最终值。

### 检索策略

第一阶段采用向量 TopK 召回，再通过 Rerank 精排。向量检索负责高召回，Rerank 负责更准确地比较问题与候选段落，避免把全部候选直接交给 LLM 造成噪声和 Token 浪费。

### 供应商抽象

LLM、Embedding、VectorStore、Reranker 和 Tool 都定义稳定协议。DashScope 与 Milvus 作为适配器实现，业务层只依赖协议，后续切换 OpenAI、DeepSeek、Claude 或其他向量库时无需重写核心流程。

## Git 管理规划

采用小步提交和 Conventional Commits：

- `chore: initialize project`
- `feat: add dashscope llm client`
- `feat: add prompt engineering module`
- `feat: implement embedding adapter`
- `feat: implement milvus vector store`
- `feat: implement rag pipeline`
- `feat: add agent tools`
- `feat: implement react agent`
- `feat: add langgraph workflow`
- `feat: expose api and observability`
- `docs: complete project documentation`

每阶段提交前运行：

```bash
uv run ruff check .
uv run black --check .
uv run pytest
git status
```

提交动作由项目所有者确认后执行，避免自动提交未经 Review 的变更。

## 安全与配置原则

- `.env`、`data/`、`logs/`、`milvus/` 和本地缓存不会进入 Git。
- `.env.example` 只保留变量名和无敏感信息的示例。
- Database Tool 默认仅允许参数化只读查询。
- Search 与 Weather Tool 设置超时、重试、速率限制和结果长度上限。
- Prompt 日志支持敏感信息脱敏，生产环境不默认记录完整文档内容。

## 当前状态

Phase 0 已完成项目边界、技术选型、整体架构、目录规划、RoadMap 和 Git 策略。

Phase 1 已完成：

- 建立 uv、`pyproject.toml` 与 `src` layout 工程。
- 使用 pydantic-settings 实现环境变量与 `.env` 类型安全配置。
- 使用 structlog 实现开发环境可读日志和生产环境 JSON 日志。
- 提供 `ai-assistant` 最小 CLI Demo，不依赖外部服务或真实 API Key。
- 配置 Ruff、Black、pytest 与覆盖率报告，并为配置、日志、应用和 CLI
  提供单元测试。

下一阶段是 Phase 2：定义 LLM 端口并接入 DashScope，支持普通聊天、流式输出、
超时和重试。
