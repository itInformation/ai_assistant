"""Pydantic request and response models for the REST API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenUsageSchema(BaseModel):
    """Token usage exposed in API responses."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolTraceSchema(BaseModel):
    """One public tool trace entry."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str


class ObservabilitySchema(BaseModel):
    """Request-level observability summary returned to API callers."""

    request_id: str
    latency_ms: float
    prompt_logged: bool
    response_logged: bool
    token_usage: TokenUsageSchema | None = None
    tool_trace: list[ToolTraceSchema] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Chat endpoint request body."""

    message: str = Field(min_length=1)


class ChatApiResponse(BaseModel):
    """Chat endpoint response body."""

    answer: str
    model: str
    finish_reason: str | None = None
    provider_request_id: str | None = None
    observability: ObservabilitySchema


class RagRequest(BaseModel):
    """RAG endpoint request body."""

    question: str = Field(min_length=1)


class RagSourceSchema(BaseModel):
    """One RAG source chunk returned by the API."""

    reference: int
    chunk_id: str
    source: str
    chunk_index: int
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagApiResponse(BaseModel):
    """RAG endpoint response body."""

    answer: str
    model: str
    retrieved_count: int
    sources: list[RagSourceSchema]
    observability: ObservabilitySchema


class AgentRequest(BaseModel):
    """Agent endpoint request body."""

    question: str = Field(min_length=1)


class AgentApiResponse(BaseModel):
    """Agent endpoint response body."""

    answer: str
    model: str
    traces: list[ToolTraceSchema]
    observability: ObservabilitySchema


class WorkflowRequest(BaseModel):
    """LangGraph workflow endpoint request body."""

    question: str = Field(min_length=1)
    mode: Literal["basic", "advanced"] = "basic"


class WorkflowStepSchema(BaseModel):
    """One LangGraph node step."""

    node: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class WorkflowApiResponse(BaseModel):
    """LangGraph workflow endpoint response body."""

    answer: str
    mode: Literal["basic", "advanced"]
    checkpoint_thread_id: str | None
    steps: list[WorkflowStepSchema]
    observability: ObservabilitySchema
