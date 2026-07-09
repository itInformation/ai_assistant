"""Application service that combines prompts, models, and output parsers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from enterprise_ai_assistant.llm import ChatModel
from enterprise_ai_assistant.models import ChatMessage, ChatResponse
from enterprise_ai_assistant.prompt.parser import StructuredOutputParser
from enterprise_ai_assistant.prompt.template import PromptTemplate

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StructuredPromptResult(Generic[OutputModelT]):
    """Validated business data together with raw response metadata."""

    value: OutputModelT
    response: ChatResponse


class PromptService:
    """Render prompts and invoke a provider-independent chat model."""

    def __init__(self, chat_model: ChatModel) -> None:
        """Create a prompt service for one chat model adapter."""

        self._chat_model = chat_model

    async def run(
        self,
        template: PromptTemplate,
        variables: Mapping[str, object],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Render and execute a free-text prompt."""

        return await self._chat_model.chat(
            template.render(**variables),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def run_structured(
        self,
        template: PromptTemplate,
        variables: Mapping[str, object],
        parser: StructuredOutputParser[OutputModelT],
        *,
        temperature: float | None = 0,
    ) -> StructuredPromptResult[OutputModelT]:
        """Execute JSON Mode and validate the result against a business schema."""

        messages = list(template.render(**variables))
        messages = self._add_output_instructions(
            messages,
            parser.format_instructions(),
        )
        response = await self._chat_model.chat(
            messages,
            temperature=temperature,
            # Alibaba Cloud recommends not bounding tokens in JSON Mode:
            # truncation would turn an otherwise valid object into invalid JSON.
            max_tokens=None,
            response_format="json_object",
        )
        return StructuredPromptResult(
            value=parser.parse(response.content),
            response=response,
        )

    @staticmethod
    def _add_output_instructions(
        messages: list[ChatMessage],
        instructions: str,
    ) -> list[ChatMessage]:
        if messages and messages[0].role == "system":
            system = messages[0]
            messages[0] = ChatMessage(
                role="system",
                content=f"{system.content}\n\n{instructions}",
            )
        else:
            messages.insert(
                0,
                ChatMessage(role="system", content=instructions),
            )
        return messages
