import os
import pathlib
import typing as typ

import datasets
import pydantic
from datasets import fingerprint
from pydantic_settings import SettingsConfigDict


CACHE_DIR = str(
    pathlib.Path(os.environ.get("CACHE_DIR", "~/.cache/docgen")).expanduser()
)
DATASETS_CACHE_PATH = str(pathlib.Path(CACHE_DIR, "datasets"))


@typ.runtime_checkable
class DatasetLoader(typ.Protocol):
    """A dataset loader."""

    def __call__(
        self, subset: None | str = None, split: None | str = None, **kws: typ.Any
    ) -> datasets.DatasetDict | datasets.Dataset:
        """Load a dataset."""
        ...


class DatasetOptions(pydantic.BaseModel):
    """Preprocessing options."""

    prep_map_kws: dict[str, typ.Any] = pydantic.Field(
        default_factory=dict,
        description="Kwargs for `datasets.map(...)`.",
    )
    subset_size: None | int = pydantic.Field(
        default=None,
        description="Take a subset of the dataset.",
    )
    seed: int = pydantic.Field(
        default=0,
        description="Seed for reproducibility.",
    )
    adapter: None | str = pydantic.Field(
        default=None, description="Adapter for the dataset."
    )
    model_config = SettingsConfigDict(arbitrary_types_allowed=True, extra="forbid")


class DatasetConfig(pydantic.BaseModel):
    """Defines a dataset."""

    identifier: str = pydantic.Field(  # type: ignore | auto-lowercase
        ...,
        description="Name of the dataset",
    )
    name_or_path: str | DatasetLoader = pydantic.Field(
        ...,
        description="Path to the dataset loader (overrides `name`)",
    )
    subsets: list[str] = pydantic.Field(
        default_factory=list,
        description="A list of subset names to load.",
    )
    split: str | None = pydantic.Field(
        None,
        description="Dataset split (train, etc.)",
    )
    trust_remote_code: bool = pydantic.Field(
        default=True,
        description="Trust remote code.",
    )
    options: DatasetOptions = pydantic.Field(
        default_factory=DatasetOptions,  # type: ignore
        description="Loading/preprocessing options.",
    )
    kwargs: dict[str, typ.Any] = pydantic.Field(
        default_factory=dict,
        description="Additional kwargs for the dataset loader.",
    )

    model_config = SettingsConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid", from_attributes=True
    )

    def __hash__(self) -> int:
        """Hash the object based on its name and split."""
        return hash(self.fingerprint())

    @pydantic.computed_field
    def fingerprint(self) -> str:
        """Return the hexdigest of the hash."""
        data = self.model_dump()
        if not isinstance(self.name_or_path, str):
            self.name_or_path = fingerprint.Hasher.hash(self.name_or_path)
        return fingerprint.Hasher.hash(data)

    @pydantic.field_validator("options", mode="before")
    @classmethod
    def _validate_options(
        cls: type[typ.Self], v: None | dict[str, typ.Any] | DatasetOptions
    ) -> dict[str, typ.Any] | DatasetOptions:
        if isinstance(v, DatasetOptions):
            return v

        return DatasetOptions(**(v or {}))

    @pydantic.field_validator("subsets", mode="before")
    @classmethod
    def _validate_subsets(
        cls: type[typ.Self], value: None | str | list[str]
    ) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]

        return [str(x) for x in value]
