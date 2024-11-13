import pydantic


class Segment(pydantic.BaseModel):
    """A text chunk."""

    text: str
    start: int
    end: int
    extras: str = ""

    # Makes the model immutable which will automatically generate __hash__ and __eq__ allowing instances to be hashed.
    model_config = pydantic.ConfigDict(frozen=True)

    @pydantic.field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        v = v.strip()
        if v:
            return v
        raise ValueError("Text cannot be empty")
