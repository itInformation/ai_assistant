"""Tests for the Phase 10 FastAPI application and observability."""

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from enterprise_ai_assistant.api.app import create_app
from enterprise_ai_assistant.api.container import ApiServiceOverrides
from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.models import (
    AgentAnswer,
    AgentToolCall,
    AgentTrace,
    ChatMessage,
    ChatResponse,
    RagAnswer,
    RagSource,
    RetrievalCandidate,
    TokenUsage,
    ToolResult,
    WorkflowAnswer,
    WorkflowPlan,
    WorkflowReview,
    WorkflowStep,
)
from enterprise_ai_assistant.observability import InMemoryObservabilityRecorder


class FakeChatModel:
    """Return deterministic chat responses for API tests."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",
    ) -> ChatResponse:
        return ChatResponse(
            content=f"回复: {messages[-1].content}",
            model="fake-chat",
            finish_reason="stop",
            request_id="provider-1",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


class FailingChatModel:
    """Raise a stable error so the API can record error logs."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",
    ) -> ChatResponse:
        raise RuntimeError("provider unavailable")


class FakeRagService:
    """Return one grounded answer."""

    async def answer(self, question: str) -> RagAnswer:
        return RagAnswer(
            answer="员工年假需要在系统中提前申请。",
            sources=(
                RagSource(
                    reference=1,
                    chunk_id="chunk-1",
                    source="handbook.md",
                    chunk_index=0,
                    score=0.92,
                    content="年假需提前申请。",
                    metadata={"department": "hr"},
                ),
            ),
            retrieved_count=3,
            model="fake-rag",
            usage=TokenUsage(prompt_tokens=4, completion_tokens=5, total_tokens=9),
        )


class FakeAgent:
    """Return one answer with one public tool trace."""

    async def answer(self, question: str) -> AgentAnswer:
        return AgentAnswer(
            answer="北京天气晴, 适合散步。",
            traces=(
                AgentTrace(
                    thought="需要查询天气。",
                    action=AgentToolCall(
                        id="call-1",
                        name="weather",
                        arguments={"location": "北京"},
                    ),
                    observation=ToolResult(
                        tool_name="weather",
                        content="北京晴 24°C",
                        data={"temperature_c": 24},
                    ),
                ),
            ),
            model="fake-agent",
            total_prompt_tokens=6,
            total_completion_tokens=7,
        )


@dataclass(slots=True)
class FakeWorkflow:
    """Return a deterministic workflow result for either graph mode."""

    mode: str

    async def run(self, question: str) -> WorkflowAnswer:
        return WorkflowAnswer(
            question=question,
            answer=f"{self.mode} 工作流回答。",
            plan=WorkflowPlan(
                goal="回答问题",
                retrieval_query=question,
                tool_name="weather",
                tool_arguments={"location": "北京"},
            ),
            review=WorkflowReview(passed=True, feedback="证据足够。"),
            retrieved=(
                RetrievalCandidate(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="公司政策片段",
                    source="policy.md",
                    chunk_index=0,
                    vector_score=0.9,
                    metadata={},
                ),
            ),
            tool_result=ToolResult(
                tool_name="weather",
                content="北京晴",
                data={"location": "北京"},
            ),
            steps=(WorkflowStep(node="planner", summary="完成规划"),),
            checkpoint_thread_id="thread-1",
        )


def build_test_client(
    *,
    chat_model: Any | None = None,
    recorder: InMemoryObservabilityRecorder | None = None,
) -> tuple[TestClient, InMemoryObservabilityRecorder]:
    """Build a TestClient with all external providers replaced."""

    event_recorder = recorder or InMemoryObservabilityRecorder()
    app = create_app(
        settings=Settings(_env_file=None, app_env="testing"),
        services=ApiServiceOverrides(
            chat_model=chat_model or FakeChatModel(),
            rag_service=FakeRagService(),  # type: ignore[arg-type]
            agent=FakeAgent(),  # type: ignore[arg-type]
            basic_workflow=FakeWorkflow("basic"),  # type: ignore[arg-type]
            advanced_workflow=FakeWorkflow("advanced"),  # type: ignore[arg-type]
        ),
        recorder=event_recorder,
    )
    return TestClient(app), event_recorder


def test_health_and_openapi_are_available() -> None:
    """API Demo should expose health and Swagger/OpenAPI metadata."""

    client, _ = build_test_client()
    with client:
        health = client.get("/health")
        openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert openapi.status_code == 200
    assert "/api/v1/chat" in openapi.json()["paths"]


def test_chat_endpoint_returns_answer_and_observability() -> None:
    """Chat endpoint should expose response, token usage, and request id."""

    client, recorder = build_test_client()
    with client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好"},
            headers={"x-request-id": "req-chat"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-chat"
    assert payload["answer"] == "回复: 你好"
    assert payload["observability"]["request_id"] == "req-chat"
    assert payload["observability"]["token_usage"]["total_tokens"] == 5
    assert recorder.events[-1].operation == "chat"
    assert recorder.events[-1].prompt_preview == "你好"


def test_rag_endpoint_returns_sources() -> None:
    """RAG endpoint should return grounded sources and token usage."""

    client, recorder = build_test_client()
    with client:
        response = client.post("/api/v1/rag", json={"question": "年假怎么申请?"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["retrieved_count"] == 3
    assert payload["sources"][0]["source"] == "handbook.md"
    assert payload["observability"]["token_usage"]["total_tokens"] == 9
    assert recorder.events[-1].operation == "rag"


def test_agent_endpoint_returns_tool_trace() -> None:
    """Agent endpoint should include safe public tool traces."""

    client, recorder = build_test_client()
    with client:
        response = client.post("/api/v1/agent", json={"question": "北京天气?"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["traces"][0]["tool"] == "weather"
    assert payload["observability"]["tool_trace"][0]["observation"] == "北京晴 24°C"
    assert recorder.events[-1].tool_trace[0]["tool"] == "weather"


def test_workflow_endpoint_runs_selected_graph() -> None:
    """Workflow endpoint should return checkpoint and node trace."""

    client, recorder = build_test_client()
    with client:
        response = client.post(
            "/api/v1/workflow",
            json={"question": "给出建议", "mode": "advanced"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["answer"] == "advanced 工作流回答。"
    assert payload["checkpoint_thread_id"] == "thread-1"
    assert payload["steps"][0]["node"] == "planner"
    assert recorder.events[-1].operation == "workflow.advanced"


def test_endpoint_errors_are_recorded_without_leaking_detail() -> None:
    """Provider failures should become 500 responses and error events."""

    client, recorder = build_test_client(chat_model=FailingChatModel())
    with client:
        response = client.post("/api/v1/chat", json={"message": "你好"})

    assert response.status_code == 500
    assert response.json()["detail"] == "API operation failed"
    assert recorder.events[-1].error_type == "RuntimeError"
    assert recorder.events[-1].error_message == "provider unavailable"
