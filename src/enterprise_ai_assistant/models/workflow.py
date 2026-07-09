"""Auditable result models returned by LangGraph workflows."""

from dataclasses import dataclass, field

from enterprise_ai_assistant.models.rag import RetrievalCandidate
from enterprise_ai_assistant.models.tool import ToolResult
from enterprise_ai_assistant.models.vectorstore import JSONValue


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One completed LangGraph node with the decision it made."""

    node: str
    summary: str
    data: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    """Planner output shared by the basic and advanced graphs."""

    goal: str
    retrieval_query: str
    tool_name: str | None = None
    tool_arguments: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowReview:
    """Reviewer judgment before the final answer is generated."""

    passed: bool
    feedback: str


@dataclass(frozen=True, slots=True)
class WorkflowAnswer:
    """Final workflow answer with all inspectable intermediate artifacts."""

    question: str
    answer: str
    plan: WorkflowPlan
    review: WorkflowReview
    retrieved: tuple[RetrievalCandidate, ...] = ()
    tool_result: ToolResult | None = None
    steps: tuple[WorkflowStep, ...] = ()
    checkpoint_thread_id: str | None = None
