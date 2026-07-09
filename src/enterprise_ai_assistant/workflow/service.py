"""LangGraph workflows that compose planning, retrieval, tools, and review."""

from __future__ import annotations

import json
from collections.abc import Mapping
from operator import add
from typing import Annotated, Any, NotRequired, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

try:  # LangGraph renamed MemorySaver to InMemorySaver in newer releases.
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - compatibility branch for older releases.
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from enterprise_ai_assistant.llm import ChatModel
from enterprise_ai_assistant.models import (
    ChatMessage,
    JSONValue,
    RetrievalCandidate,
    ToolResult,
    WorkflowAnswer,
    WorkflowPlan,
    WorkflowReview,
    WorkflowStep,
)
from enterprise_ai_assistant.rag import VectorRetriever
from enterprise_ai_assistant.tools import ToolRegistry
from enterprise_ai_assistant.workflow.prompts import (
    build_answer_messages,
    build_planner_messages,
    build_reviewer_messages,
    build_supervisor_messages,
)


class WorkflowState(TypedDict):
    """Shared state passed through every LangGraph node."""

    question: str
    plan: NotRequired[WorkflowPlan]
    retrieved: NotRequired[tuple[RetrievalCandidate, ...]]
    tool_result: NotRequired[ToolResult | None]
    review: NotRequired[WorkflowReview]
    answer: NotRequired[str]
    steps: Annotated[list[WorkflowStep], add]


class BasicLangGraphWorkflow:
    """Planner -> Retriever -> Tool -> Reviewer -> Answer workflow."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        retriever: VectorRetriever,
        tools: ToolRegistry,
        retrieval_top_k: int = 5,
        max_answer_tokens: int = 1_200,
    ) -> None:
        """Compile the graph once so every run uses the same checked topology."""

        if not 1 <= retrieval_top_k <= 20:
            raise ValueError("retrieval_top_k must be between 1 and 20")
        self._chat_model = chat_model
        self._retriever = retriever
        self._tools = tools
        self._retrieval_top_k = retrieval_top_k
        self._max_answer_tokens = max_answer_tokens
        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph().compile(checkpointer=self._checkpointer)

    async def run(
        self,
        question: str,
        *,
        thread_id: str | None = None,
    ) -> WorkflowAnswer:
        """Execute the graph and return a typed, auditable result."""

        if not question.strip():
            raise ValueError("workflow question must not be empty")
        checkpoint_thread_id = thread_id or f"workflow-{uuid4().hex}"
        state = await self._graph.ainvoke(
            {"question": question, "steps": []},
            {"configurable": {"thread_id": checkpoint_thread_id}},
        )
        return _state_to_answer(state, checkpoint_thread_id)

    def _build_graph(self) -> StateGraph:
        """Build the fixed Phase 9 baseline graph topology."""

        graph = StateGraph(WorkflowState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("reviewer", self._reviewer_node)
        graph.add_node("answer", self._answer_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "tool")
        graph.add_edge("tool", "reviewer")
        graph.add_edge("reviewer", "answer")
        graph.add_edge("answer", END)
        return graph

    async def _planner_node(self, state: WorkflowState) -> dict[str, object]:
        tool_names = tuple(spec.name for spec in self._tools.specs())
        system, user = build_planner_messages(state["question"], tool_names)
        response = await self._chat_model.chat(
            [ChatMessage("system", system), ChatMessage("user", user)],
            temperature=0.1,
            response_format="json_object",
        )
        plan = _parse_plan(response.content, state["question"], tool_names)
        return {
            "plan": plan,
            "steps": [
                WorkflowStep(
                    node="planner",
                    summary=f"规划检索: {plan.retrieval_query}",
                    data={"tool_name": plan.tool_name},
                )
            ],
        }

    async def _retriever_node(self, state: WorkflowState) -> dict[str, object]:
        plan = state["plan"]
        retrieved = await self._retriever.retrieve(
            plan.retrieval_query,
            top_k=self._retrieval_top_k,
        )
        return {
            "retrieved": retrieved,
            "steps": [
                WorkflowStep(
                    node="retriever",
                    summary=f"召回 {len(retrieved)} 个知识库 Chunk",
                    data={"top_k": self._retrieval_top_k},
                )
            ],
        }

    async def _tool_node(self, state: WorkflowState) -> dict[str, object]:
        plan = state["plan"]
        if plan.tool_name is None:
            return {
                "tool_result": None,
                "steps": [WorkflowStep(node="tool", summary="Planner 未选择工具")],
            }
        result = await self._tools.invoke(plan.tool_name, plan.tool_arguments)
        return {
            "tool_result": result,
            "steps": [
                WorkflowStep(
                    node="tool",
                    summary=f"调用工具 {plan.tool_name}: {result.content}",
                    data={"tool_name": plan.tool_name},
                )
            ],
        }

    async def _reviewer_node(self, state: WorkflowState) -> dict[str, object]:
        system, user = build_reviewer_messages(
            state["question"],
            state.get("retrieved", ()),
            state.get("tool_result"),
        )
        response = await self._chat_model.chat(
            [ChatMessage("system", system), ChatMessage("user", user)],
            temperature=0,
            response_format="json_object",
        )
        review = _parse_review(response.content)
        return {
            "review": review,
            "steps": [
                WorkflowStep(
                    node="reviewer",
                    summary=review.feedback,
                    data={"passed": review.passed},
                )
            ],
        }

    async def _answer_node(self, state: WorkflowState) -> dict[str, object]:
        system, user = build_answer_messages(
            state["question"],
            state["plan"],
            state.get("retrieved", ()),
            state.get("tool_result"),
            state["review"].feedback,
        )
        response = await self._chat_model.chat(
            [ChatMessage("system", system), ChatMessage("user", user)],
            temperature=0.2,
            max_tokens=self._max_answer_tokens,
        )
        return {
            "answer": response.content,
            "steps": [WorkflowStep(node="answer", summary="生成最终回答")],
        }


class AdvancedLangGraphWorkflow(BasicLangGraphWorkflow):
    """Supervisor -> Research Agent -> Tool Agent -> Summary Agent workflow."""

    def _build_graph(self) -> StateGraph:
        """Build the advanced multi-agent graph topology."""

        graph = StateGraph(WorkflowState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("research_agent", self._retriever_node)
        graph.add_node("tool_agent", self._tool_node)
        graph.add_node("summary_agent", self._summary_node)
        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "research_agent")
        graph.add_edge("research_agent", "tool_agent")
        graph.add_edge("tool_agent", "summary_agent")
        graph.add_edge("summary_agent", END)
        return graph

    async def _supervisor_node(self, state: WorkflowState) -> dict[str, object]:
        tool_names = tuple(spec.name for spec in self._tools.specs())
        system, user = build_supervisor_messages(state["question"], tool_names)
        response = await self._chat_model.chat(
            [ChatMessage("system", system), ChatMessage("user", user)],
            temperature=0.1,
            response_format="json_object",
        )
        plan = _parse_plan(response.content, state["question"], tool_names)
        return {
            "plan": plan,
            "review": WorkflowReview(
                passed=True,
                feedback="进阶工作流由 Summary Agent 汇总 Research 与 Tool 结果。",
            ),
            "steps": [
                WorkflowStep(
                    node="supervisor",
                    summary=f"分配 Research/Tool/Summary 任务: {plan.goal}",
                    data={"tool_name": plan.tool_name},
                )
            ],
        }

    async def _summary_node(self, state: WorkflowState) -> dict[str, object]:
        update = await self._answer_node(state)
        return {
            "answer": update["answer"],
            "steps": [WorkflowStep(node="summary_agent", summary="汇总多 Agent 结果")],
        }


def _parse_plan(
    content: str,
    question: str,
    allowed_tools: tuple[str, ...],
) -> WorkflowPlan:
    """Parse planner JSON and fall back to a safe retrieval-only plan."""

    data = _load_json_object(content)
    tool_name = data.get("tool_name")
    if not isinstance(tool_name, str) or tool_name not in allowed_tools:
        tool_name = None
    arguments = data.get("tool_arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    return WorkflowPlan(
        goal=_string_or_default(data.get("goal"), question),
        retrieval_query=_string_or_default(data.get("retrieval_query"), question),
        tool_name=tool_name,
        tool_arguments=_json_object(arguments),
    )


def _parse_review(content: str) -> WorkflowReview:
    """Parse reviewer JSON and default to a conservative warning."""

    data = _load_json_object(content)
    return WorkflowReview(
        passed=bool(data.get("passed", False)),
        feedback=_string_or_default(data.get("feedback"), "Reviewer 未返回明确反馈。"),
    )


def _load_json_object(content: str) -> dict[str, Any]:
    """Load one JSON object, tolerating markdown fences from weaker models."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _string_or_default(value: object, default: str) -> str:
    """Return a non-empty string or a known safe default."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _json_object(value: Mapping[str, object]) -> dict[str, JSONValue]:
    """Keep only JSON-serializable tool arguments in planner output."""

    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError:
        return {}
    return dict(value)  # type: ignore[return-value]


def _state_to_answer(
    state: WorkflowState,
    checkpoint_thread_id: str,
) -> WorkflowAnswer:
    """Convert raw LangGraph state into the public result model."""

    return WorkflowAnswer(
        question=state["question"],
        answer=state.get("answer", "工作流未生成最终回答。"),
        plan=state.get(
            "plan",
            WorkflowPlan(goal=state["question"], retrieval_query=state["question"]),
        ),
        review=state.get(
            "review",
            WorkflowReview(passed=False, feedback="工作流未执行 Reviewer。"),
        ),
        retrieved=state.get("retrieved", ()),
        tool_result=state.get("tool_result"),
        steps=tuple(state.get("steps", [])),
        checkpoint_thread_id=checkpoint_thread_id,
    )
