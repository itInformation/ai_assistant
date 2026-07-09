"""Stable exceptions raised by the tool layer."""


class ToolError(RuntimeError):
    """Base class for tool failures safe to handle at application boundaries."""


class ToolConfigurationError(ToolError):
    """Raised when a tool cannot run with the current configuration."""


class ToolInputError(ToolError):
    """Raised when invocation arguments violate the tool contract."""


class ToolProviderError(ToolError):
    """Raised when an external provider or database operation fails."""


class ToolNotFoundError(ToolError):
    """Raised when a caller requests an unregistered tool."""
