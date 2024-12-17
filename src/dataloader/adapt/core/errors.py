import typing as typ


class CantHandleError(ValueError):
    """Raised when input data can't be handled."""

    def __init__(
        self,
        row: dict[str, typ.Any],
        reason: str = "",
        **kwargs,
    ) -> None:
        """Initialize the error."""
        row_ = {k: type(v).__name__ for k, v in row.items()}
        message = f"Could not find an adapter for row `{row_}`. "
        message += f"Reason: {reason}"
        super().__init__(message, **kwargs)
