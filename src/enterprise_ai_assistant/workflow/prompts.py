"""Prompt builders used by LangGraph planning, review, and answer nodes."""

from enterprise_ai_assistant.models import RetrievalCandidate, ToolResult, WorkflowPlan


def build_planner_messages(
    question: str, tool_names: tuple[str, ...]
) -> tuple[str, str]:
    """Return system and user prompts for the workflow planner."""

    tools = ", ".join(tool_names) if tool_names else "无可用工具"
    system = (
        "你是企业级 AI 知识助手的 LangGraph Planner。"
        "请把用户问题拆成检索意图和最多一个工具调用。"
        "只返回 JSON, 字段为 goal、retrieval_query、tool_name、tool_arguments。"
        "tool_name 只能是可用工具之一或 null。"
    )
    user = f"用户问题: {question}\n可用工具: {tools}"
    return system, user


def build_supervisor_messages(
    question: str,
    tool_names: tuple[str, ...],
) -> tuple[str, str]:
    """Return prompts for the advanced supervisor node."""

    tools = ", ".join(tool_names) if tool_names else "无可用工具"
    system = (
        "你是 Supervisor, 负责为多 Agent 工作流分配任务。"
        "请输出 JSON: goal、retrieval_query、tool_name、tool_arguments。"
        "Research Agent 负责知识库检索, Tool Agent 负责工具调用, Summary Agent 汇总。"
    )
    user = f"用户问题: {question}\n可用工具: {tools}"
    return system, user


def build_reviewer_messages(
    question: str,
    candidates: tuple[RetrievalCandidate, ...],
    tool_result: ToolResult | None,
) -> tuple[str, str]:
    """Return prompts for reviewing whether evidence is enough to answer."""

    system = (
        "你是企业知识助手的 Reviewer。"
        "请判断检索证据和工具结果是否足以回答问题。"
        "只返回 JSON, 字段为 passed(boolean)、feedback(string)。"
    )
    user = (
        f"问题: {question}\n"
        f"检索证据:\n{format_candidates(candidates)}\n"
        f"工具结果:\n{format_tool_result(tool_result)}"
    )
    return system, user


def build_answer_messages(
    question: str,
    plan: WorkflowPlan,
    candidates: tuple[RetrievalCandidate, ...],
    tool_result: ToolResult | None,
    review_feedback: str,
) -> tuple[str, str]:
    """Return prompts for grounded final answer generation."""

    system = (
        "你是企业级 AI 知识助手。请基于知识库检索结果和工具结果回答。"
        "不要编造来源; 证据不足时说明无法确定, 并给出下一步建议。"
    )
    user = (
        f"问题: {question}\n"
        f"计划目标: {plan.goal}\n"
        f"Reviewer 反馈: {review_feedback}\n"
        f"知识库证据:\n{format_candidates(candidates)}\n"
        f"工具结果:\n{format_tool_result(tool_result)}"
    )
    return system, user


def format_candidates(candidates: tuple[RetrievalCandidate, ...]) -> str:
    """Render retrieved chunks in a compact, citation-friendly format."""

    if not candidates:
        return "无"
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"[{index}] source={candidate.source} chunk={candidate.chunk_index} "
            f"score={candidate.vector_score:.4f}\n{candidate.content}"
        )
    return "\n\n".join(lines)


def format_tool_result(result: ToolResult | None) -> str:
    """Render a tool observation without exposing provider internals."""

    if result is None:
        return "未调用工具"
    return f"{result.tool_name}: {result.content}"
