"""DashScope chat adapter implemented through its OpenAI-compatible API."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from enterprise_ai_assistant.config import Settings
from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
)
from enterprise_ai_assistant.models import (
    AgentMessage,
    AgentModelResponse,
    AgentToolCall,
    ChatChunk,
    ChatMessage,
    ChatResponse,
    ResponseFormat,
    TokenUsage,
    ToolSpec,
)

_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class DashScopeChatModel:
    """Chat model backed by Alibaba Cloud Model Studio (DashScope)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Create a DashScope adapter with bounded timeout and retry behavior."""

        if not api_key.strip():
            raise LLMConfigurationError("DASHSCOPE_API_KEY is required")
        if timeout_seconds <= 0:
            raise LLMConfigurationError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise LLMConfigurationError("max_retries must not be negative")

        self._model = model
        self._max_retries = max_retries
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            # Tenacity owns the retry policy so one failure cannot trigger
            # nested SDK retries and unexpectedly multiply request latency.
            max_retries=0,
        )
        self._logger = get_logger(provider="dashscope", model=model)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> ChatResponse:
        """Generate a complete chat response."""

        self._validate_messages(messages)
        try:
            response = await self._create_with_retry(
                messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except APIStatusError as exc:
            raise self._provider_error(exc) from exc
        except _RETRYABLE_ERRORS as exc:
            raise self._provider_error(exc) from exc

        if not isinstance(response, ChatCompletion):
            raise LLMProviderError("DashScope returned an unexpected response type")
        if not response.choices:
            raise LLMProviderError("DashScope returned no response choices")

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            request_id=response.id,
            usage=self._map_usage(response.usage),
        )

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: ResponseFormat = "text",
    ) -> AsyncIterator[ChatChunk]:
        """Yield incremental chunks without replaying already emitted content."""

        self._validate_messages(messages)
        try:
            stream = await self._create_with_retry(
                messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            # Only connection establishment is retried. Once a chunk has been
            # emitted, replaying the request could duplicate visible content.
            async for item in stream:
                # Usage-only terminal chunks can legitimately contain no choice.
                if not isinstance(item, ChatCompletionChunk) or not item.choices:
                    continue
                choice = item.choices[0]
                yield ChatChunk(
                    content=choice.delta.content or "",
                    model=item.model,
                    finish_reason=choice.finish_reason,
                    request_id=item.id,
                    usage=self._map_usage(item.usage),
                )
        except APIStatusError as exc:
            raise self._provider_error(exc) from exc
        except _RETRYABLE_ERRORS as exc:
            raise self._provider_error(exc) from exc

    async def chat_with_tools(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolSpec],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentModelResponse:
        """Use DashScope native Function Calling for one Agent turn."""

        if not messages:
            raise ValueError("at least one agent message is required")
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [self._map_agent_message(message) for message in messages],
            "extra_body": {"enable_thinking": False},
        }
        if tools:
            # ReAct handles dependent calls serially and keeps the trace ordered.
            request["parallel_tool_calls"] = False
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            request["tool_choice"] = "auto"
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        try:
            response = await self._request_with_retry(request)
        except APIStatusError as exc:
            raise self._provider_error(exc) from exc
        except _RETRYABLE_ERRORS as exc:
            raise self._provider_error(exc) from exc
        if not isinstance(response, ChatCompletion) or not response.choices:
            raise LLMProviderError("DashScope returned no Agent response choices")
        choice = response.choices[0]
        try:
            calls = tuple(
                self._map_tool_call(call) for call in (choice.message.tool_calls or ())
            )
            message = AgentMessage(
                role="assistant",
                content=choice.message.content or "",
                tool_calls=calls,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                "DashScope returned an invalid Agent message"
            ) from exc
        usage = self._map_usage(response.usage)
        return AgentModelResponse(
            message=message,
            model=response.model,
            finish_reason=choice.finish_reason,
            request_id=response.id,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    async def _create_with_retry(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        response_format: ResponseFormat,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": stream,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format == "json_object":
            request["response_format"] = {"type": "json_object"}
        if stream:
            request["stream_options"] = {"include_usage": True}

        retrying = AsyncRetrying(
            # max_retries excludes the first request, hence the extra attempt.
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            reraise=True,
            before_sleep=self._log_retry,
        )
        async for attempt in retrying:
            with attempt:
                return await self._client.chat.completions.create(**request)
        raise AssertionError("retry loop completed without a result")

    async def _request_with_retry(self, request: dict[str, Any]) -> Any:
        """Execute a non-streaming request with the shared bounded policy."""

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            reraise=True,
            before_sleep=self._log_retry,
        )
        async for attempt in retrying:
            with attempt:
                return await self._client.chat.completions.create(**request)
        raise AssertionError("retry loop completed without a result")

    @staticmethod
    def _map_agent_message(message: AgentMessage) -> dict[str, Any]:
        mapped: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            mapped["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            mapped["tool_call_id"] = message.tool_call_id
        return mapped

    @staticmethod
    def _map_tool_call(call: Any) -> AgentToolCall:
        arguments = json.loads(call.function.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        return AgentToolCall(
            id=str(call.id),
            name=str(call.function.name),
            arguments=arguments,
        )

    def _log_retry(self, retry_state: Any) -> None:
        self._logger.warning(
            "llm_request_retry",
            attempt=retry_state.attempt_number,
            error_type=type(retry_state.outcome.exception()).__name__,
        )

    def _provider_error(self, error: Exception) -> LLMProviderError:
        self._logger.error(
            "llm_request_failed",
            error_type=type(error).__name__,
        )
        return LLMProviderError(f"DashScope request failed: {type(error).__name__}")

    @staticmethod
    def _validate_messages(messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise ValueError("at least one chat message is required")

    @staticmethod
    def _map_usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


def create_dashscope_chat_model(settings: Settings) -> DashScopeChatModel:
    """Build the default DashScope adapter from application settings."""

    return DashScopeChatModel(
        api_key=settings.dashscope_api_key or "",
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_chat_model,
        timeout_seconds=settings.dashscope_timeout_seconds,
        max_retries=settings.dashscope_max_retries,
    )
