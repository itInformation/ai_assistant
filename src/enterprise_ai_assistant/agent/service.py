"""Bounded ReAct orchestration over native model Tool Calling."""

import json

from enterprise_ai_assistant.agent.exceptions import AgentExecutionError
from enterprise_ai_assistant.agent.memory import ConversationMemory
from enterprise_ai_assistant.agent.prompts import AGENT_SYSTEM_PROMPT
from enterprise_ai_assistant.llm import ToolCallingModel
from enterprise_ai_assistant.models import (
    AgentAnswer,
    AgentMessage,
    AgentModelResponse,
    AgentToolCall,
    AgentTrace,
    ToolResult,
)
from enterprise_ai_assistant.tools import ToolError, ToolRegistry


class ReActAgent:
    """Plan, call allow-listed tools, observe results, and answer."""

    def __init__(
        self,
        *,
        model: ToolCallingModel,
        tools: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_tool_calls: int = 5,
        max_observation_chars: int = 6_000,
        max_answer_tokens: int = 1_200,
    ) -> None:
        if max_tool_calls <= 0 or max_observation_chars <= 0 or max_answer_tokens <= 0:
            raise ValueError("Agent execution limits must be positive")
        self._model = model
        self._tools = tools
        self._memory = memory or ConversationMemory()
        self._max_tool_calls = max_tool_calls
        self._max_observation_chars = max_observation_chars
        self._max_answer_tokens = max_answer_tokens

    async def answer(self, question: str) -> AgentAnswer:
        """Run one bounded Agent turn and persist only the completed dialogue."""

        if not question.strip():
            raise ValueError("Agent question must not be empty")
        messages = [
            AgentMessage(role="system", content=AGENT_SYSTEM_PROMPT),
            *(await self._memory.snapshot()),
            AgentMessage(role="user", content=question),
        ]
        traces: list[AgentTrace] = []
        prompt_tokens = 0
        completion_tokens = 0
        last_model = ""

        while len(traces) < self._max_tool_calls:
            response = await self._call_model(messages, with_tools=True)
            last_model = response.model
            prompt_tokens += response.prompt_tokens or 0
            completion_tokens += response.completion_tokens or 0
            messages.append(response.message)
            if not response.message.tool_calls:
                return await self._finish(
                    question,
                    response,
                    traces,
                    prompt_tokens,
                    completion_tokens,
                )

            for call in response.message.tool_calls:
                if len(traces) >= self._max_tool_calls:
                    messages.append(
                        AgentMessage(
                            role="tool",
                            content='{"error":"tool_call_budget_exhausted"}',
                            tool_call_id=call.id,
                        )
                    )
                    continue
                observation, tool_message = await self._execute_tool(call)
                traces.append(
                    AgentTrace(
                        thought=self._thought_summary(response, call),
                        action=call,
                        observation=observation,
                    )
                )
                messages.append(tool_message)

        # Once the budget is exhausted, remove tools and force a grounded
        # summary from observations already present in the conversation.
        response = await self._call_model(messages, with_tools=False)
        last_model = response.model or last_model
        prompt_tokens += response.prompt_tokens or 0
        completion_tokens += response.completion_tokens or 0
        if response.message.tool_calls:
            raise AgentExecutionError("model requested tools after budget exhaustion")
        return await self._finish(
            question,
            response,
            traces,
            prompt_tokens,
            completion_tokens,
            model=last_model,
        )

    async def _call_model(
        self,
        messages: list[AgentMessage],
        *,
        with_tools: bool,
    ) -> AgentModelResponse:
        return await self._model.chat_with_tools(
            messages,
            self._tools.specs() if with_tools else (),
            temperature=0.1,
            max_tokens=self._max_answer_tokens,
        )

    async def _execute_tool(
        self,
        call: AgentToolCall,
    ) -> tuple[ToolResult | str, AgentMessage]:
        try:
            result = await self._tools.invoke(call.name, call.arguments)
            payload = json.dumps(
                {
                    "summary": result.content,
                    "data": result.data,
                    "metadata": result.metadata,
                },
                ensure_ascii=False,
            )
            observation: ToolResult | str = result
        except ToolError as exc:
            # Return a stable error category so the model can recover without
            # receiving credentials, SQL internals, or provider response bodies.
            payload = json.dumps(
                {"error": type(exc).__name__},
                ensure_ascii=False,
            )
            observation = f"工具执行失败: {type(exc).__name__}"
        bounded_payload = payload
        if len(payload) > self._max_observation_chars:
            bounded_payload = json.dumps(
                {"truncated_observation": payload[: self._max_observation_chars]},
                ensure_ascii=False,
            )
        return observation, AgentMessage(
            role="tool",
            content=bounded_payload,
            tool_call_id=call.id,
        )

    async def _finish(
        self,
        question: str,
        response: AgentModelResponse,
        traces: list[AgentTrace],
        prompt_tokens: int,
        completion_tokens: int,
        *,
        model: str | None = None,
    ) -> AgentAnswer:
        answer = response.message.content.strip()
        if not answer:
            raise AgentExecutionError("model returned an empty final answer")
        await self._memory.remember(question, answer)
        return AgentAnswer(
            answer=answer,
            traces=tuple(traces),
            model=model or response.model,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
        )

    @staticmethod
    def _thought_summary(
        response: AgentModelResponse,
        call: AgentToolCall,
    ) -> str:
        summary = response.message.content.strip()
        if not summary:
            return f"需要调用 {call.name} 获取外部信息。"
        return summary[:300]
