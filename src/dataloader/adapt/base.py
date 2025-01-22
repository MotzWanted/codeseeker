import abc
import hashlib
import random
import typing as typ

import datasets
import pydantic

from dataloader.base import DatasetOptions
from segmenters.base import Segmenter

Im = typ.TypeVar("Im", bound=pydantic.BaseModel)
Om = typ.TypeVar("Om", bound=pydantic.BaseModel)
DictStrKey: typ.TypeAlias = dict[str, typ.Any]


class BaseTrainingModel(pydantic.BaseModel):
    """Fewshot model."""

    aid: str
    segments: str
    targets: list[str]
    classes: dict[str, str] = pydantic.Field(..., description="Look up table for classes.")

    @pydantic.field_validator("targets", mode="after")
    def order_target_indices(cls, v: list[int]) -> list[int]:
        """Order the target indices from smallest to largest."""
        if len(set(v)) < len(v):
            raise ValueError(f"The target indices must be unique: {v}")
        return sorted(v)

    @pydantic.model_validator(mode="after")
    def validate_codes_and_classes(self):
        """Validate the codes and classes."""
        for target in self.targets:
            if target not in self.classes:
                raise ValueError(f"The target {target} is not in the classes.")
        return self

    def parse_targets(self, shuffle: bool = False, seed: int = 42) -> str:
        """Parse the targets."""
        keys_list = list(self.classes.keys())
        if shuffle:
            random.shuffle(keys_list, seed=seed)
        return f"{','.join(str(keys_list.index(i)+1) for i in self.targets)}"

    def decode_targets(self, indexes: list[int]) -> list[str]:
        """Decode the targets."""
        return [self.classes[index] for index in indexes]


class BaseInferenceModel(pydantic.BaseModel):
    """Alignment model."""

    aid: str = pydantic.Field(..., description="The alignment identifier.")
    note_type: str | None = pydantic.Field(default=None, description="The note type.")
    note_subtype: str | None = pydantic.Field(default=None, description="The note subtype.")
    classes: list[str] = pydantic.Field(..., description="The classes to constrain the output space.")
    segments: list[str] = pydantic.Field(..., description="The segments to align.")
    targets: list[list[int]] = pydantic.Field(default=[[]], description="The target indices point to class indices.")
    fewshots: list[BaseTrainingModel] | None = pydantic.Field(
        default=None, description="Fewshots to include in prompt."
    )
    index2code: dict[str, str] = pydantic.Field(..., description="Look up table for classes.")

    @pydantic.field_validator("segments", mode="before")
    @classmethod
    def validate_no_empty_strings(cls, v: list[str]) -> list[str]:
        """Validate that the list of strings do not contain empty strings."""
        if "" in v:
            raise ValueError("The list of strings must not contain empty strings.")
        return v

    @pydantic.field_validator("targets", mode="before")
    def validate_target_indices(cls, v: list[list[int]]) -> list[list[int]]:
        """Validate the target indices and sort indices from smallest to largest."""
        if 0 in v and len(v) > 1:
            raise ValueError("The zero index must be the only index if it is present.")
        return [sorted(inner_list) for inner_list in v]

    @pydantic.model_validator(mode="after")
    def validate_targets_and_shuffle_fewshots(self):
        """Validate the labels."""
        if self.targets:
            max_value = max(max(inner_list) for inner_list in self.targets)
            if max_value > len(self.classes):
                raise ValueError("The maximum value in the labels must be less than the number of entities.")

        if self.fewshots is None:
            return self
        seed = int(hashlib.sha256(self.aid.encode("utf-8")).hexdigest(), 16) % (2**32)
        random.seed(seed)
        random.shuffle(self.fewshots)
        return self


class AsDict:
    """A callable that converts a pydantic model to a dict."""

    def __init__(
        self, fn: typ.Callable[[DictStrKey, DatasetOptions], pydantic.BaseModel], options: DatasetOptions
    ) -> None:
        self.fn = fn
        self.options = options

    def __call__(self, x: DictStrKey) -> DictStrKey:
        """Call the inner functions and dump to dict."""
        m = self.fn(x, self.options)
        return m.model_dump()


class Adapter(typ.Generic[Im, Om], abc.ABC):
    """Adapter for alignment instances associated with multiple queries."""

    input_model: type[Im]
    output_model: type[Om]

    def __init__(self, segmenter: Segmenter) -> None:
        self.segmenter = segmenter
        super().__init__()

    @classmethod
    def can_handle(cls, row: dict[str, typ.Any]) -> bool:
        """Can handle."""
        try:
            cls.input_model(**row)
            return True
        except pydantic.ValidationError:
            return False

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> Om:
        """Placeholder for translating a row."""
        raise NotImplementedError(f"{cls.__name__} does not implement `translate_row`")

    @classmethod
    def translate_dset(cls, dset: datasets.Dataset, options: DatasetOptions, **kwargs: typ.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset.map(
            AsDict(cls.translate_row, options),
            remove_columns=dset.column_names,
            desc=f"Adapting dataset using {cls.__name__}",
            **kwargs,
        )

    @classmethod
    def translate(
        cls: type["Adapter"],
        x: dict[str, typ.Any] | datasets.Dataset | datasets.DatasetDict,
        options: DatasetOptions,
        map_kwargs: dict | None = None,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Translate a row, dataset or dataset dict."""
        map_kwargs = map_kwargs or {}
        if isinstance(x, datasets.Dataset):
            return cls.translate_dset(x, options, **map_kwargs)
        if isinstance(x, datasets.DatasetDict):
            return datasets.DatasetDict({k: cls.translate_dset(v, options, **map_kwargs) for k, v in x.items()})  # type: ignore
        if isinstance(x, dict):
            return cls.translate_row(x).model_dump()  # type: ignore

        raise TypeError(f"Cannot adapt input of type `{type(x)}`")
