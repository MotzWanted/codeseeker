import abc
import typing as typ

import datasets
import pydantic

Im = typ.TypeVar("Im", bound=pydantic.BaseModel)
Om = typ.TypeVar("Om", bound=pydantic.BaseModel)
DictStrKey: typ.TypeAlias = dict[str, typ.Any]


class AsDict:
    """A callable that converts a pydantic model to a dict."""

    def __init__(self, fn: typ.Callable[[DictStrKey], pydantic.BaseModel]) -> None:
        self.fn = fn

    def __call__(self, x: DictStrKey) -> DictStrKey:
        """Call the inner functions and dump to dict."""
        m = self.fn(x)
        return m.model_dump()


class Adapter(typ.Generic[Im, Om], abc.ABC):
    """An adapter for a dataset."""

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
    def translate(
        cls: type["Adapter"],
        x: dict[str, typ.Any] | datasets.Dataset | datasets.DatasetDict,
        map_kwargs: dict | None = None,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Translate a row, dataset or dataset dict."""
        map_kwargs = map_kwargs or {}
        if isinstance(x, datasets.Dataset):
            return cls.translate_dset(x, **map_kwargs)
        if isinstance(x, datasets.DatasetDict):
            return datasets.DatasetDict({k: cls.translate_dset(v, **map_kwargs) for k, v in x.items()})  # type: ignore
        if isinstance(x, dict):
            return cls.translate_row(x).model_dump()  # type: ignore

        raise TypeError(f"Cannot adapt input of type `{type(x)}`")

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any]) -> Om:
        """Placeholder for translating a row."""
        raise NotImplementedError(f"{cls.__name__} does not implement `translate_row`")

    @classmethod
    def translate_dset(cls, dset: datasets.Dataset, **kwargs: typ.Any) -> datasets.Dataset:
        """Translating a dataset."""
        return dset.map(
            AsDict(cls.translate_row),
            remove_columns=dset.column_names,
            desc=f"Adapting dataset using {cls.__name__}",
            **kwargs,
        )
