import typing as typ

import datasets
from dataloader.adapt.base import Adapter
from dataloader.adapt.adapters import KNOWN_ADAPTERS, get_adapter_by_name
from dataloader.base import DatasetOptions

from .utils import get_first_row

T = typ.TypeVar("T")


def find_adapter(
    row: dict[str, typ.Any], verbose: bool = False
) -> None | typ.Type[Adapter]:
    """Find an adapter for a row."""
    for v in KNOWN_ADAPTERS:
        if v.can_handle(row):
            return v

    return None


class CantHandleError(ValueError):
    """Raised when input data can't be handled."""

    def __init__(
        self,
        row: dict[str, typ.Any],
        reason: str = "",
        **kwargs,
    ) -> None:
        """Initialize the error."""
        row_ = {k: type(v).__name__ for k, v in row.items()}
        message = f"Could not find an adapter for row `{row_}`. "
        message += f"Reason: {reason}"
        super().__init__(message, **kwargs)


def transform(
    data: datasets.Dataset | datasets.DatasetDict,
    options: DatasetOptions,
    verbose: bool = False,
) -> datasets.Dataset | datasets.DatasetDict:
    """Translate a HuggingFace daatset."""
    row = get_first_row(data)

    if options.adapter:
        adapter = get_adapter_by_name(options.adapter)
    else:
        adapter: None | typ.Type[Adapter] = find_adapter(row, verbose=verbose)
    if adapter is None:
        raise CantHandleError(row, reason="No matching `Adapter` could be found.")
    return adapter.translate(data, options, map_kwargs=options.prep_map_kws)
