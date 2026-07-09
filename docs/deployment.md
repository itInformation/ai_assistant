# 部署说明

本文档用于 Phase 11 项目交付，覆盖本地运行、Docker 运行、配置、安全检查和生产化演进建议。

## 本地开发

```bash
cp .env.example .env
uv sync --all-groups
uv run ai-assistant serve
```

启动后访问：

- Health Check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Docker 单容器

```bash
cp .env.example .env
docker build -t enterprise-ai-assistant:latest .
docker run --env-file .env \
  -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/logs:/app/logs" \
  -v "$PWD/milvus:/app/milvus" \
  enterprise-ai-assistant:latest
```

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Compose 默认把 `data/`、`logs/`、`milvus/` 挂载到容器中，便于保存 SQLite 数据库、日志文件和 Milvus Lite 文件。

## 必填配置

最小可用配置：

```env
DASHSCOPE_API_KEY=你的百炼 Key
MILVUS_URI=/app/milvus/knowledge.db
API_HOST=0.0.0.0
API_PORT=8000
```

如果需要 Search Tool，还要配置：

```env
TAVILY_API_KEY=你的 Tavily Key
```

如果需要 Database Tool，请准备只读 SQLite 文件，并设置：

```env
DATABASE_PATH=/app/data/assistant.db
```

## 生产化检查清单

- 密钥只通过环境变量或密钥管理系统注入，不进入镜像和 Git。
- 生产环境建议把 `APP_ENV=production`，日志输出 JSON。
- 将 Milvus Lite 替换为 Milvus Standalone、Distributed 或 Zilliz Cloud。
- 为 API 增加鉴权、限流、审计和租户隔离。
- 将 `InMemoryObservabilityRecorder` 替换为 OpenTelemetry、ELK、Prometheus 或云日志服务。
- 为 LangGraph Checkpoint 替换持久化后端，例如 Postgres 或 SQLite Checkpointer。
- 为 RAG 建立离线评测集，持续评估召回率、重排质量、答案忠实度和拒答率。

## 常用运维命令

```bash
docker compose ps
docker compose logs -f ai-assistant
docker compose restart ai-assistant
docker compose down
```

## 回滚策略

镜像使用不可变版本号发布，例如 `enterprise-ai-assistant:0.1.0`。如果新版本异常：

1. 保留旧镜像和旧 `.env`。
2. 停止当前容器。
3. 使用上一版本镜像重新启动。
4. 检查 `/health`、核心 API、日志和外部模型调用。

Milvus Lite 文件和 SQLite 文件通过 volume 挂载，应用回滚不会自动回滚数据；生产环境应对数据文件或远端数据库建立独立备份策略。
