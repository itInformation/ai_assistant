"""Tests for Phase 9 LangGraph workflows."""

import asyncio
from collections.abc import Mapping, Sequence

from enterprise_ai_assistant.models import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    JSONValue,
    ResponseFormat,
    RetrievalCandidate,
    ToolResult,
    ToolSpec,
)
from enterprise_ai_assistant.tools import ToolRegistry
from enterprise_ai_assistant.workflow import (
    AdvancedLangGraphWorkflow,
    BasicLangGraphWorkflow,
)


class FakeWorkflowModel:
    """Return deterministic planner, reviewer, and answer responses."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> ChatResponse:
        system = messages[0].content
        if response_format == "json_object" and (
            "Planner" in system or "Supervisor" in system
        ):
            return ChatResponse(
                content=(
                    '{"goal":"回答北京出行建议",'
                    '"retrieval_query":"北京 企业 出行 政策",'
                    '"tool_name":"weather",'
                    '"tool_arguments":{"location":"北京"}}'
                ),
                model="fake",
            )
        if response_format == "json_object" and "Reviewer" in system:
            return ChatResponse(
                content='{"passed":true,"feedback":"证据和天气足够回答。"}',
                model="fake",
            )
        return ChatResponse(content="北京适合短时间户外散步。", model="fake")

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> Sequence[ChatChunk]:
        """Streaming is not used by workflow tests."""

        return ()


class FakeRetriever:
    """Return one deterministic enterprise knowledge chunk."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        search_filter: object | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        self.queries.append(query)
        return (
            RetrievalCandidate(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="北京办公室访客需要提前预约。",
                source="policy.md",
                chunk_index=0,
                vector_score=0.91,
                metadata={},
            ),
        )


class FakeWeatherTool:
    """Return deterministic weather output for workflow tests."""

    spec = ToolSpec(
        name="weather",
        description="Fake weather.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    )

    async def invoke(
        self,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        return ToolResult(
            tool_name="weather",
            content=f"{arguments['location']} 晴 24°C",
            data={"location": arguments["location"], "temperature_c": 24},
        )

    async def close(self) -> None:
        """Satisfy the tool lifecycle."""


def test_basic_workflow_runs_all_required_nodes() -> None:
    """The baseline graph should execute Planner/Retriever/Tool/Reviewer/Answer."""

    retriever = FakeRetriever()
    workflow = BasicLangGraphWorkflow(
        chat_model=FakeWorkflowModel(),
        retriever=retriever,  # type: ignore[arg-type]
        tools=ToolRegistry([FakeWeatherTool()]),
        retrieval_top_k=3,
    )

    result = asyncio.run(workflow.run("北京今天适合散步吗?", thread_id="test-thread"))

    assert [step.node for step in result.steps] == [
        "planner",
        "retriever",
        "tool",
        "reviewer",
        "answer",
    ]
    assert result.checkpoint_thread_id == "test-thread"
    assert result.plan.tool_name == "weather"
    assert result.tool_result is not None
    assert result.tool_result.data["temperature_c"] == 24
    assert retriever.queries == ["北京 企业 出行 政策"]
    assert result.answer == "北京适合短时间户外散步。"


def test_advanced_workflow_uses_multi_agent_node_names() -> None:
    """The advanced graph should expose Supervisor/Research/Tool/Summary agents."""

    workflow = AdvancedLangGraphWorkflow(
        chat_model=FakeWorkflowModel(),
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        tools=ToolRegistry([FakeWeatherTool()]),
    )

    result = asyncio.run(workflow.run("北京今天适合散步吗?"))

    assert [step.node for step in result.steps] == [
        "supervisor",
        "retriever",
        "tool",
        "summary_agent",
    ]
    assert result.review.passed is True
    assert result.answer == "北京适合短时间户外散步。"


def test_workflow_falls_back_to_retrieval_only_when_tool_is_not_allowed() -> None:
    """Planner output must not call tools outside the registry allow-list."""

    workflow = BasicLangGraphWorkflow(
        chat_model=FakeWorkflowModel(),
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        tools=ToolRegistry(),
    )

    result = asyncio.run(workflow.run("北京今天适合散步吗?"))

    assert result.plan.tool_name is None
    assert result.tool_result is None
    assert result.steps[2].summary == "Planner 未选择工具"
