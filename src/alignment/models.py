import numpy as np
import pydantic
import typing as typ


class Alignment(pydantic.BaseModel):
    """Class representing the extraction of entities in list of segments."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    indexes: list[list[int]] = pydantic.Field(
        ..., description="A list of indices of the segment elements that align with the query element."
    )
    matrix: np.ndarray = pydantic.Field(
        ..., description="A sparse matrix of alignment scores between corpus and query elements."
    )
    probabilities: np.ndarray | None = pydantic.Field(..., description="Token log probabilities.")
    extras: dict | None = pydantic.Field(default=None, description="Additional information.")

    @pydantic.field_validator("matrix", "probabilities", mode="before")
    @classmethod
    def ensure_float64(cls, value: np.ndarray) -> np.ndarray:
        return value.astype(np.float32)


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
