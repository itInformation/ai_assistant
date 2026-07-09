"""Stable exceptions raised by the ReAct Agent."""


class AgentError(RuntimeError):
    """Base class for failures at the Agent application boundary."""


class AgentExecutionError(AgentError):
    """Raised when the Agent cannot produce a valid final answer."""
