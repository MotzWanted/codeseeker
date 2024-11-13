import typing as typ

import pydantic

from dataloader.adapters.base import Adapter
from dataloader.loaders.base import DatasetLoader


class DatasetOptionsDiff(pydantic.BaseModel):
    """Preprocessing options diff."""

    prep_map_kws: None | dict[str, typ.Any] = None
    subset_size: None | int = None

    # validators
    _validate_prep_map_kws = pydantic.field_validator("prep_map_kws", mode="before")


class DatasetOptions(pydantic.BaseModel):
    """Preprocessing options."""

    prep_map_kws: dict[str, typ.Any] = pydantic.Field(
        default_factory=dict,
        description="Kwargs for `datasets.map(...)`.",
    )
    size: None | int = pydantic.Field(
        default=None,
        description="Take a subset of the dataset.",
    )
    n_shots: None | int = pydantic.Field(default=None, description="Number of fewshot samples to extract.")

    # validators
    _validate_prep_map_kws = pydantic.field_validator("prep_map_kws", mode="before")

    def __add__(self: typ.Self, other: None | DatasetOptionsDiff) -> typ.Self:
        """Add two options."""
        if other is None:
            return self
        attrs = other.model_dump(exclude_none=True)
        return type(self)(**{**self.model_dump(), **attrs})


class DatasetConfig(pydantic.BaseModel):
    """Defines a dataset."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True, from_attributes=True)

    identifier: str = pydantic.Field(  # type: ignore | auto-lowercase
        ...,
        description="Name of the dataset",
    )
    name_or_path: str | DatasetLoader = pydantic.Field(
        ...,
        description="Path to the dataset loader (overrides `name`)",
    )
    subset: str | None = pydantic.Field(
        default=None,
        description="A list of subset names to load.",
    )
    split: str | None = pydantic.Field(
        default=None,
        description="Dataset split (train, etc.)",
    )
    adapter: type[Adapter] = pydantic.Field(
        ...,
        description="Adapter to use for the dataset.",
    )
    options: DatasetOptions = pydantic.Field(
        default_factory=DatasetOptions,  # type: ignore
        description="Loading/preprocessing options.",
    )
