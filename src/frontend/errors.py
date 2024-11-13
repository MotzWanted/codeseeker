import json


class BaseExceptionError(Exception):
    """Base exception."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        msg_obj = json.dumps({"message": self.message, "code": self.code})
        return f"event: error\ndata: {msg_obj}\n\n"
