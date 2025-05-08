"""Custom exceptions for the agents module."""


class StructuredError(Exception):
    """Raised when the agent response cannot be structured into the expected format."""

    def __init__(
        self,
        message: str = "Could not structure the agent response into the expected format",
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
