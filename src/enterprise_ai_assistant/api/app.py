"""FastAPI application factory and REST endpoints."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, TypeVar
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from enterprise_ai_assistant import __version__
from enterprise_ai_assistant.api.container import ApiContainer, ApiServiceOverrides
from enterprise_ai_assistant.api.schemas import (
    AgentApiResponse,
    AgentRequest,
    ChatApiResponse,
    ChatRequest,
    ObservabilitySchema,
    RagApiResponse,
    RagRequest,
    RagSourceSchema,
    TokenUsageSchema,
    ToolTraceSchema,
    WorkflowApiResponse,
    WorkflowRequest,
    WorkflowStepSchema,
)
from enterprise_ai_assistant.app import build_application_info
from enterprise_ai_assistant.config import Settings, get_settings
from enterprise_ai_assistant.core import configure_logging, get_logger
from enterprise_ai_assistant.models import (
    AgentAnswer,
    ChatMessage,
    ChatResponse,
    RagAnswer,
    TokenUsage,
    ToolResult,
    WorkflowAnswer,
)
from enterprise_ai_assistant.observability import (
    InMemoryObservabilityRecorder,
    ObservabilityEvent,
    ObservabilityRecorder,
)
from enterprise_ai_assistant.observability.events import preview_text

T = TypeVar("T")


def create_app(
    *,
    settings: Settings | None = None,
    services: ApiServiceOverrides | None = None,
    recorder: ObservabilityRecorder | None = None,
) -> FastAPI:
    """Create a configured FastAPI app with injectable services for tests."""

    resolved_settings = settings or get_settings()
    configure_logging(
        resolved_settings.log_level,
        json_logs=resolved_settings.app_env in {"staging", "production"},
    )
    container = ApiContainer(resolved_settings, overrides=services)
    event_recorder = recorder or InMemoryObservabilityRecorder()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.container = container
        app.state.recorder = event_recorder
        yield
        await container.close()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="企业级 AI 知识助手 REST API",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[..., Awaitable[Any]],
    ):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = _elapsed_ms(start)
            get_logger(component="api").exception(
                "http_request_failed",
                request_id=request_id,
                path=request.url.path,
                latency_ms=latency_ms,
            )
            raise
        response.headers["x-request-id"] = request_id
        get_logger(component="api").info(
            "http_request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=_elapsed_ms(start),
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return service health and metadata without external provider calls."""

        return asdict(build_application_info(resolved_settings))

    @app.post("/api/v1/chat", response_model=ChatApiResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatApiResponse:
        """Generate a direct model answer."""

        async def operation() -> ChatResponse:
            return (
                await _container(request)
                .chat_model()
                .chat([ChatMessage(role="user", content=payload.message)])
            )

        response, observability = await _observe(
            request,
            operation_name="chat",
            prompt=payload.message,
            operation=operation,
            response_text=lambda item: item.content,
            usage=lambda item: item.usage,
        )
        return ChatApiResponse(
            answer=response.content,
            model=response.model,
            finish_reason=response.finish_reason,
            provider_request_id=response.request_id,
            observability=observability,
        )

    @app.post("/api/v1/rag", response_model=RagApiResponse)
    async def rag(payload: RagRequest, request: Request) -> RagApiResponse:
        """Answer using the enterprise knowledge base."""

        async def operation() -> RagAnswer:
            return await _container(request).rag_service().answer(payload.question)

        response, observability = await _observe(
            request,
            operation_name="rag",
            prompt=payload.question,
            operation=operation,
            response_text=lambda item: item.answer,
            usage=lambda item: item.usage,
        )
        return RagApiResponse(
            answer=response.answer,
            model=response.model,
            retrieved_count=response.retrieved_count,
            sources=[RagSourceSchema(**asdict(source)) for source in response.sources],
            observability=observability,
        )

    @app.post("/api/v1/agent", response_model=AgentApiResponse)
    async def agent(payload: AgentRequest, request: Request) -> AgentApiResponse:
        """Run the ReAct Agent and expose its safe tool trace."""

        async def operation() -> AgentAnswer:
            return await _container(request).agent().answer(payload.question)

        response, observability = await _observe(
            request,
            operation_name="agent",
            prompt=payload.question,
            operation=operation,
            response_text=lambda item: item.answer,
            usage=_agent_usage,
            tool_trace=_agent_tool_trace,
        )
        return AgentApiResponse(
            answer=response.answer,
            model=response.model,
            traces=_agent_tool_trace(response),
            observability=observability,
        )

    @app.post("/api/v1/workflow", response_model=WorkflowApiResponse)
    async def workflow(
        payload: WorkflowRequest,
        request: Request,
    ) -> WorkflowApiResponse:
        """Run the selected LangGraph workflow."""

        async def operation() -> WorkflowAnswer:
            workflow_service = _container(request).workflow(payload.mode)
            return await workflow_service.run(payload.question)

        response, observability = await _observe(
            request,
            operation_name=f"workflow.{payload.mode}",
            prompt=payload.question,
            operation=operation,
            response_text=lambda item: item.answer,
            tool_trace=_workflow_tool_trace,
        )
        return WorkflowApiResponse(
            answer=response.answer,
            mode=payload.mode,
            checkpoint_thread_id=response.checkpoint_thread_id,
            steps=[WorkflowStepSchema(**asdict(step)) for step in response.steps],
            observability=observability,
        )

    return app


async def _observe(
    request: Request,
    *,
    operation_name: str,
    prompt: str,
    operation: Callable[[], Awaitable[T]],
    response_text: Callable[[T], str],
    usage: Callable[[T], TokenUsage | None] | None = None,
    tool_trace: Callable[[T], list[ToolTraceSchema]] | None = None,
) -> tuple[T, ObservabilitySchema]:
    """Run an API operation and emit a sanitized observability event."""

    start = time.perf_counter()
    request_id = _request_id(request)
    prompt_preview = preview_text(
        prompt,
        max_chars=_settings(request).observability_preview_chars,
    )
    try:
        result = await operation()
    except Exception as exc:
        latency_ms = _elapsed_ms(start)
        _recorder(request).record(
            ObservabilityEvent(
                request_id=request_id,
                operation=operation_name,
                latency_ms=latency_ms,
                prompt_preview=prompt_preview,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )
        raise HTTPException(status_code=500, detail="API operation failed") from exc

    latency_ms = _elapsed_ms(start)
    response_preview = preview_text(
        response_text(result),
        max_chars=_settings(request).observability_preview_chars,
    )
    token_usage = usage(result) if usage is not None else None
    trace = tool_trace(result) if tool_trace is not None else []
    _recorder(request).record(
        ObservabilityEvent(
            request_id=request_id,
            operation=operation_name,
            latency_ms=latency_ms,
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            token_usage=token_usage,
            tool_trace=tuple(trace_item.model_dump() for trace_item in trace),
        )
    )
    return result, ObservabilitySchema(
        request_id=request_id,
        latency_ms=latency_ms,
        prompt_logged=True,
        response_logged=True,
        token_usage=_token_usage_schema(token_usage),
        tool_trace=trace,
    )


def _container(request: Request) -> ApiContainer:
    return request.app.state.container


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _recorder(request: Request) -> ObservabilityRecorder:
    return request.app.state.recorder


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid4().hex))


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _token_usage_schema(usage: TokenUsage | None) -> TokenUsageSchema | None:
    if usage is None:
        return None
    return TokenUsageSchema(**asdict(usage))


def _agent_usage(answer: AgentAnswer) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=answer.total_prompt_tokens,
        completion_tokens=answer.total_completion_tokens,
        total_tokens=answer.total_prompt_tokens + answer.total_completion_tokens,
    )


def _agent_tool_trace(answer: AgentAnswer) -> list[ToolTraceSchema]:
    traces = []
    for trace in answer.traces:
        observation = trace.observation
        if isinstance(observation, ToolResult):
            observation_text = observation.content
        else:
            observation_text = observation
        traces.append(
            ToolTraceSchema(
                tool=trace.action.name,
                arguments=trace.action.arguments,
                observation=observation_text,
            )
        )
    return traces


def _workflow_tool_trace(answer: WorkflowAnswer) -> list[ToolTraceSchema]:
    if answer.tool_result is None or answer.plan.tool_name is None:
        return []
    return [
        ToolTraceSchema(
            tool=answer.plan.tool_name,
            arguments=answer.plan.tool_arguments,
            observation=answer.tool_result.content,
        )
    ]


app = create_app()
