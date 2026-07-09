# 项目总结与面试材料

## 简历项目描述

企业级 AI 知识助手：基于 Python、阿里云百炼、Milvus Lite、FastAPI 和 LangGraph 构建的企业知识检索与 Agent 应用。项目实现了 Prompt Engineering、文档解析、Chunk、Embedding、向量检索、Rerank、RAG 问答、Tool Calling、ReAct Agent、短期 Memory、LangGraph 工作流、REST API 与基础可观测性，支持 PDF、Word、Markdown 企业文档和 Weather/Search/Database 工具扩展。

## 可放入简历的职责描述

- 从 0 到 1 搭建企业级 AI 知识助手工程，使用 uv、pyproject、Ruff、Black、pytest 和 src layout 建立可维护 Python 项目结构。
- 封装阿里云百炼 Chat、Embedding、Rerank 和 Function Calling 能力，沉淀供应商无关的 LLM、Embedding、VectorStore、Tool 端口。
- 设计并实现完整 RAG 链路，支持 PDF、Word、Markdown 文档加载、自然边界 Chunk、Milvus Lite 向量存储、TopK 召回、Rerank 精排和带来源引用的 grounded answer。
- 实现 Weather、Search、SQLite Database 工具，使用显式 ToolRegistry、参数校验、只读 SQL、防注入和超时重试控制工具边界。
- 实现 ReAct Agent 与 Memory，支持模型自主选择工具、Observation 回填、调用预算控制、异常安全回填和最终答案收敛。
- 使用 LangGraph 实现 Planner、Retriever、Tool、Reviewer、Answer 工作流，并扩展 Supervisor、Research Agent、Tool Agent、Summary Agent 多 Agent 编排。
- 使用 FastAPI 暴露 REST API 和 Swagger，并记录 Prompt/Response 预览、Token Usage、Tool Trace、Error Log 和 Latency。
- 编写单元测试与集成测试，当前测试覆盖 Prompt、LLM、Embedding、RAG、Tool、Agent、Workflow、API 等核心模块。

## 项目亮点

1. 不是简单 Demo，而是按企业工程分层：Domain Model、Port、Adapter、Application Service、API 分层清晰。
2. 默认使用阿里云百炼，符合国内企业真实选型，同时保留 OpenAI、DeepSeek、Claude 等模型替换空间。
3. RAG 链路完整：Loader、Chunk、Embedding、Milvus、Retriever、Rerank、LLM、Answer 全部可解释、可测试。
4. 工具调用有安全边界：Registry 允许列表、只读 Database Tool、工具超时重试、Observation 截断和错误类型脱敏。
5. Agent 不暴露隐藏思维链，只输出可公开审计的 Thought 摘要、Action、Observation 和 Final Answer。
6. LangGraph 使用真实 StateGraph、State、Node、Edge 和 Checkpoint，便于演示可恢复工作流和多 Agent 拓扑。
7. API 层有可观测性，能回答线上排障常见问题：用了什么 Prompt、调用了哪些工具、耗时多少、Token 多少、哪里失败。
8. 支持 Docker 和 Docker Compose，具备从本地开发走向服务部署的完整闭环。

## 面试讲解稿

这个项目是我为了系统掌握企业 AI Agent 开发，从 0 到 1 做的企业级 AI 知识助手。整体目标不是做一个只能跑通的 Demo，而是模拟真实企业项目，把模型调用、RAG、工具调用、Agent、工作流、API 和可观测性都工程化。

架构上我用了端口与适配器模式。业务层只依赖自定义协议，比如 `ChatModel`、`EmbeddingModel`、`VectorStore`、`Tool`，具体实现放到 DashScope、Milvus、Tavily、SQLite 等适配器里。这样未来换模型或向量库，不需要改 RAG、Agent 和 API 的核心代码。

RAG 链路从文档开始，支持 PDF、Word 和 Markdown。文档解析后会按自然边界 Chunk，默认 800 字符、120 overlap。Chunk 写入 Milvus Lite 前会调用百炼 `text-embedding-v3` 生成 1024 维向量。问答时先向量召回 Top 20，再用 `qwen3-rerank` 精排到 Top 5，最后把上下文交给 Qwen 生成带来源约束的答案。如果上下文不足，Prompt 会要求模型明确说明无法确定。

Agent 部分我实现了原生 Function Calling，不从自由文本里解析工具名。模型可以选择 Weather、Search、Database 三类工具，所有工具都必须先注册到 `ToolRegistry`。Database Tool 特别做了只读保护，只允许参数化 SELECT/WITH，拒绝 DDL、DML、多语句和 ATTACH。ReAct 循环里我限制最大工具调用次数，工具异常会转成安全 Observation，避免把敏感细节暴露给模型。

工作流部分使用 LangGraph。基础版是 Planner、Retriever、Tool、Reviewer、Answer；进阶版是 Supervisor、Research Agent、Tool Agent、Summary Agent。这里我重点使用了 StateGraph 的 State、Node、Edge 和 Checkpoint，让每个节点输入输出都可追踪，也为后续 human-in-the-loop 和失败恢复留了空间。

最后我用 FastAPI 把能力服务化，提供 Chat、RAG、Agent 和 Workflow API。每个 AI 端点都会返回 request_id、latency、token_usage 和 tool_trace，同时服务端记录结构化日志。部署上提供 Dockerfile 和 docker-compose，可以本地启动，也能平滑迁移到生产环境。

## 高频面试问题

### 1. 为什么 Chunk Size 选 800，Overlap 选 120？

800 字符在中文企业文档里通常能覆盖一个完整段落或一小节，语义相对完整，又不会让单个 Chunk 携带过多噪声。120 overlap 用来缓解答案跨段落边界时的信息丢失。它不是最终最优值，而是工程基线，生产环境要通过评测集调参。

### 2. 为什么需要 Rerank？

向量召回速度快，适合从大量 Chunk 中找候选，但它只做向量相似度，排序可能不够精细。Rerank 会把 query 和候选原文放在一起做更细粒度语义匹配，成本更高但排序更准确。两阶段结合能兼顾召回率、准确率和成本。

### 3. 为什么不用 LangChain 把所有东西串起来？

我复用生态，但核心领域逻辑保留自有抽象。这样不会被某个框架的数据结构锁死，也方便测试和替换供应商。LangGraph 用在确实需要状态流转和工作流编排的部分，普通线性流程仍然用简单 Python 服务。

### 4. Agent 如何避免无限调用工具？

有三层控制：第一，`ToolRegistry` 限制模型只能调用允许列表中的工具；第二，`max_tool_calls` 限制单轮最大调用次数；第三，预算耗尽后移除工具定义，强制模型基于已有 Observation 给最终答案。

### 5. Database Tool 如何保证安全？

只允许单条 SELECT 或只读 WITH，使用参数化参数，拒绝 PRAGMA、DDL、DML、ATTACH 和多语句。SQLite 连接使用只读 URI 和 `PRAGMA query_only` 双重保护，并限制最大返回行数。

### 6. 如何处理 Prompt Injection？

RAG Prompt 明确把检索上下文视为不可信数据，要求只依据上下文回答，不执行文档中的指令。工具 Observation 也被视为不可信输入，并做长度限制。更进一步可以加入 Reviewer 节点、安全分类器和来源可信度评分。

### 7. 这个项目怎么观测线上问题？

API 响应和服务端日志都会记录 request_id、latency、token_usage、tool_trace、错误类型、Prompt/Response 预览。通过 request_id 可以串起一次请求里模型、检索、工具和工作流的关键节点。

### 8. 生产环境你会怎么升级？

向量库从 Milvus Lite 升级到 Milvus Standalone/Distributed 或 Zilliz Cloud；Checkpoint 从内存换成持久化数据库；日志接 OpenTelemetry 或 ELK；API 增加鉴权、限流、租户隔离；RAG 增加离线评测和自动回归测试。

## 项目总结

这个项目覆盖了 AI Agent 岗位最常问的工程能力：模型封装、Prompt、RAG、Embedding、向量数据库、Rerank、Tool Calling、ReAct、Memory、LangGraph、API、可观测性、测试和部署。它的价值不只是“能调用大模型”，而是把大模型能力放进一个可维护、可扩展、可排障的后端系统里。
