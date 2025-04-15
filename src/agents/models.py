import numpy as np
import pydantic
import typing as typ


class CodeSpace(pydantic.BaseModel):
    """Class representing a set of candidate codes."""

    candidate_codes: list[str] = pydantic.Field(..., description="The alphanumeric ids of a larger set of codes.")
    codes: list[str] = pydantic.Field(..., description="The alphanumeric ids of the a set of codes.")
    response: str | None = pydantic.Field(None, description="The response from the model.")


class Alignment(pydantic.BaseModel):
    """Class representing the extraction of entities in list of segments."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    indexes: list[int] = pydantic.Field(
        ..., description="A list of indices of the segment elements that align with the query element."
    )
    matrix: np.ndarray = pydantic.Field(
        ..., description="A sparse matrix of alignment scores between corpus and query elements."
    )
    probabilities: np.ndarray | None = pydantic.Field(..., description="Token log probabilities.")
    response: str | None = pydantic.Field(default="", description="The response from the model.")
    extras: dict | None = pydantic.Field(default={}, description="Additional information.")

    @pydantic.field_validator("matrix", "probabilities", mode="before")
    @classmethod
    def ensure_float64(cls, value: np.ndarray) -> np.ndarray:
        return value.astype(np.float32)

    @pydantic.field_validator("indexes", mode="before")
    @classmethod
    def validate_indices(cls: type["Alignment"], v: typ.List[int]) -> typ.List[int]:
        """Validate labels."""
        if len(v) == 0 or (len(v) > 1 and 0 in v):
            v = [0]
        v = list(set(v))
        return v


class AlignmentSingleton(pydantic.BaseModel):
    """Class representing the document indices that substantiates a given fact."""

    ids: typ.List[int] = pydantic.Field(
        ...,
        description="A list of ids that support the hypothesis. If no documents supports the hypothesis, return a zero list.",  # noqa: E501
    )

    @pydantic.field_validator("ids", mode="before")
    @classmethod
    def validate_indices(cls: type["AlignmentSingleton"], v: typ.List[int]) -> typ.List[int]:
        """Validate labels."""
        if len(v) == 0 or (len(v) > 1 and 0 in v):
            v = [0]
        v = list(set(v))
        return v
