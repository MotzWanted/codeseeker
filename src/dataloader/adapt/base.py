import abc
import typing as typ
import uuid

import datasets
import pydantic

from dataloader.base import DatasetOptions

Im = typ.TypeVar("Im", bound=pydantic.BaseModel)
Om = typ.TypeVar("Om", bound=pydantic.BaseModel)
DictStrKey: typ.TypeAlias = dict[str, typ.Any]


class CodeModel(pydantic.BaseModel):
    """Model for an ICD code."""

    name: str
    description: str
    chapter_id: str
    etiology: bool
    manifestation: bool
    inclusion_term: list[str] = pydantic.Field(
        default=[], description="Inclusion terms."
    )
    excludes1: list[str] = pydantic.Field(default=[], description="Excludes1 codes.")
    excludes2: list[str] = pydantic.Field(default=[], description="Excludes2 codes.")
    code_first: list[str] = pydantic.Field(default=[], description="Code first.")
    code_also: list[str] = pydantic.Field(default=[], description="Code also.")
    use_additional_code: list[str] = pydantic.Field(
        default=[], description="Use additional code."
    )


class GuidelinesModel(pydantic.BaseModel):
    """Model for guidelines."""

    code: str
    notes: list[str] = pydantic.Field(
        default=[], description="Notes on a code or category."
    )
    includes: list[str] = pydantic.Field(default=[], description="Includes codes.")
    excludes1: list[str] = pydantic.Field(default=[], description="Excludes1 codes.")
    excludes2: list[str] = pydantic.Field(default=[], description="Excludes2 codes.")
    use_additional_code: list[str] = pydantic.Field(
        default=[], description="Use additional code."
    )
    code_first: list[str] = pydantic.Field(default=[], description="Code first.")
    code_also: list[str] = pydantic.Field(default=[], description="Code also.")
    inclusion_term: list[str] = pydantic.Field(
        default=[], description="Inclusion terms."
    )
    assignable: bool = pydantic.Field(
        default=False, description="Is the code assignable?"
    )


class EvidenceSpan(pydantic.BaseModel):
    code: str
    locations: list[tuple[int, int]]


class BaseModel(pydantic.BaseModel):
    """Base model for a note instance"""

    aid: str | uuid.UUID | None = pydantic.Field(default_factory=uuid.uuid4)
    note: str
    note_type: str | None = None
    targets: list[str] | None = None
    evidence_spans: list[EvidenceSpan] | None = None

    model_config = pydantic.ConfigDict(extra="allow")


# class LegacyBaseModel(pydantic.BaseModel):
#     """Fewshot model."""

#     aid: str
#     note: str
#     targets: list[str]

#     @pydantic.field_validator("classes", mode="after")
#     def order_classes(cls, v: list[CodeModel]) -> list[int]:
#         """Order the target indices from smallest to largest."""
#         return sorted(v, key=lambda x: x.name)

#     def parse_targets(self) -> list[str]:
#         """Parse the targets."""
#         target_idexes = [idx for idx, code in enumerate(self.classes, start=1) if code.name in self.targets]
#         return target_idexes

#     def decode_targets(self, indexes: list[int]) -> list[str]:
#         """Decode the targets."""
#         if isinstance(self.classes, str):
#             self.classes = typ.cast(dict[str, str], ast.literal_eval(self.classes))
#         if not isinstance(self.classes, dict):
#             raise TypeError(f"Classes must be a dict, not {type(self.classes)}")
#         return [self.classes[str(index)] for index in indexes]


class AsDict:
    """A callable that converts a pydantic model to a dict."""

    def __init__(
        self,
        fn: typ.Callable[[DictStrKey, DatasetOptions], pydantic.BaseModel],
        options: DatasetOptions,
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
    def translate_dset(
        cls, dset: datasets.Dataset, options: DatasetOptions, **kwargs: typ.Any
    ) -> datasets.Dataset:
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
