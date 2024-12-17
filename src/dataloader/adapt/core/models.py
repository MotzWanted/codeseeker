import typing as typ

import pydantic


class FactGenerationModel(pydantic.BaseModel):
    """Corpus model."""

    id: str = pydantic.Field(
        ...,
        description="The unique identifier for the corpus element.",
    )
    transcript: str = pydantic.Field(..., description="The input text.")
    facts: list[str] = pydantic.Field(..., description="The summary segments of the text.")

    @pydantic.field_validator("id", mode="before")
    @classmethod
    def _validate_id(cls: typ.Type["FactGenerationModel"], value: str | int) -> str:
        """Validate the ID."""
        if isinstance(value, int):
            return str(value)
        return value
