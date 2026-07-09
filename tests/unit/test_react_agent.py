"""Tests for bounded ReAct orchestration."""

import asyncio
from collections.abc import Mapping, Sequence

from enterprise_ai_assistant.agent import ConversationMemory, ReActAgent
from enterprise_ai_assistant.models import (
    AgentMessage,
    AgentModelResponse,
    AgentToolCall,
    JSONValue,
    ToolResult,
    ToolSpec,
)
from enterprise_ai_assistant.tools import ToolRegistry


class ScriptedModel:
    """Return deterministic Agent turns and record complete histories."""

    def __init__(self, *responses: AgentModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[tuple[AgentMessage, ...], tuple[ToolSpec, ...]]] = []

    async def chat_with_tools(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolSpec],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentModelResponse:
        self.requests.append((tuple(messages), tuple(tools)))
        return self.responses.pop(0)


class WeatherTool:
    """Return deterministic weather observations."""

    spec = ToolSpec(
        name="weather",
        description="Get weather.",
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
            content="北京当前 30°C",
            data={"location": arguments["location"], "temperature_c": 30},
        )

    async def close(self) -> None:
        """Satisfy the tool lifecycle."""


def response(
    content: str,
    *calls: AgentToolCall,
) -> AgentModelResponse:
    """Build one scripted model response."""

    return AgentModelResponse(
        message=AgentMessage(
            role="assistant",
            content=content,
            tool_calls=tuple(calls),
        ),
        model="fake-agent",
        prompt_tokens=10,
        completion_tokens=5,
    )


def test_agent_calls_tool_observes_and_finishes() -> None:
    """A complete ReAct turn should preserve an auditable trace."""

    call = AgentToolCall(
        id="call-1",
        name="weather",
        arguments={"location": "北京"},
    )
    model = ScriptedModel(
        response("需要查询实时天气。", call),
        response("根据 Weather Tool, 北京当前 30°C。"),
    )
    agent = ReActAgent(
        model=model,
        tools=ToolRegistry([WeatherTool()]),
    )

    result = asyncio.run(agent.answer("北京天气怎么样?"))

    assert result.answer.endswith("北京当前 30°C。")
    assert result.traces[0].action == call
    assert result.traces[0].observation.data["temperature_c"] == 30
    second_messages = model.requests[1][0]
    assert second_messages[-1].role == "tool"
    assert second_messages[-1].tool_call_id == "call-1"
    assert result.total_prompt_tokens == 20


def test_agent_memory_is_included_in_next_turn() -> None:
    """Completed answers should become bounded context for follow-up questions."""

    memory = ConversationMemory()
    model = ScriptedModel(
        response("第一轮回答"),
        response("第二轮回答"),
    )
    agent = ReActAgent(
        model=model,
        tools=ToolRegistry(),
        memory=memory,
    )

    asyncio.run(agent.answer("第一问"))
    asyncio.run(agent.answer("追问"))

    second_request = model.requests[1][0]
    assert [message.content for message in second_request[-3:]] == [
        "第一问",
        "第一轮回答",
        "追问",
    ]


def test_agent_recovers_from_unknown_tool() -> None:
    """Unknown model-selected tools should become safe observations."""

    call = AgentToolCall(id="call-1", name="invented", arguments={})
    model = ScriptedModel(
        response("尝试工具。", call),
        response("该工具不可用, 无法获取信息。"),
    )
    agent = ReActAgent(model=model, tools=ToolRegistry())

    result = asyncio.run(agent.answer("调用不存在的工具"))

    assert result.traces[0].observation == "工具执行失败: ToolNotFoundError"
    assert "ToolNotFoundError" in model.requests[1][0][-1].content


def test_agent_forces_final_answer_after_tool_budget() -> None:
    """The final model call must receive no tools after budget exhaustion."""

    first = AgentToolCall(id="call-1", name="weather", arguments={"location": "北京"})
    second = AgentToolCall(id="call-2", name="weather", arguments={"location": "上海"})
    model = ScriptedModel(
        response("查询北京。", first),
        response("继续查询上海。", second),
        response("基于已有结果给出最终回答。"),
    )
    agent = ReActAgent(
        model=model,
        tools=ToolRegistry([WeatherTool()]),
        max_tool_calls=2,
    )

    result = asyncio.run(agent.answer("比较天气"))

    assert len(result.traces) == 2
    assert model.requests[-1][1] == ()
