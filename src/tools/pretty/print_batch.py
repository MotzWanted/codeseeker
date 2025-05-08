import functools
import math
import typing as typ
from numbers import Number

import numpy as np
import pydantic
import rich
import rich.console
import rich.syntax
import rich.table
import rich.tree
import torch
from loguru import logger
from typing_extensions import Self, Type
from tools.misc.config import flatten_dict

_PPRINT_PREC = 2


def _smart_str(x: Number) -> str:
    if isinstance(x, float):
        return f"{x:.{_PPRINT_PREC}e}"
    if isinstance(x, int):
        return f"{x}"
    if isinstance(x, complex):
        return f"{x.real:.{_PPRINT_PREC}e} + {x.imag:.{_PPRINT_PREC}e}j"

    return str(x)


class Properties(pydantic.BaseModel):
    """Defines a set of displayable properties for a given object."""

    py_type: str
    dtype: None | str = None
    shape: None | str = None
    device: None | str = None
    mean: None | str = None
    min: None | str = None
    max: None | str = None
    n_nans: None | int = None

    @pydantic.field_validator("py_type", mode="before")
    @classmethod
    def _cast_py_type(cls: Type[Self], value: typ.Any) -> str:  # noqa: ANN401
        if value is None:
            return "None"
        if isinstance(value, type):
            return value.__name__

        return type(value).__name__


@functools.singledispatch
def infer_properties(x: typ.Any) -> Properties:  # noqa: ANN401
    """Base function for inferring properties of an object."""
    return Properties(py_type=type(x))


@infer_properties.register(torch.Tensor)
def _(x: torch.Tensor) -> Properties:
    """Infer properties of a torch tensor."""
    x_float = x.detach().float()
    return Properties(
        py_type=type(x),
        shape=str(list(x.shape)),
        dtype=str(x.dtype),
        device=str(x.device),
        mean=f"{_smart_str(x_float.mean().item())}",
        min=f"{_smart_str(x.min().item())}",
        max=f"{_smart_str(x.max().item())}",
        n_nans=int(torch.isnan(x).sum().item()),
    )


@infer_properties.register(np.ndarray)
def _(x: np.ndarray) -> Properties:
    """Infer properties of a numpy array."""
    x_float: np.ndarray = x.astype(np.float32)
    return Properties(
        py_type=type(x),
        shape=str(x.shape),
        dtype=f"np.{x.dtype}",
        device="-",
        mean=f"{_smart_str(x_float.mean())}",
        min=f"{_smart_str(np.min(x))}",
        max=f"{_smart_str(np.max(x))}",
        n_nans=int(np.isnan(x).sum()),
    )


@infer_properties.register(Number)
def _(x: Number) -> Properties:
    """Infer properties of a number."""
    return Properties(
        py_type=type(x),
        dtype="-",
        shape="-",
        device="-",
        min="-",
        max="-",
        mean=f"{_smart_str(x)}",
    )


@infer_properties.register(dict)
def _(x: dict) -> Properties:
    """Infer properties of a number."""
    return Properties(
        py_type=type(x),
        dtype="-",
        shape="-",
        device="-",
        min="-",
        max="-",
        mean="-",
    )


@infer_properties.register(list)
@infer_properties.register(set)
@infer_properties.register(tuple)
def _(x: list | set | tuple) -> Properties:
    """Infer properties of a list, set or tuple."""
    try:
        arr = np.array(x)
        shape = str(arr.shape)
    except ValueError:
        shape = f"[{len(x)}, ?]"

    n_nans = sum(1 for y in _iter_leaves(x) if y is None)
    leaves_types = list({type(y) for y in _iter_leaves(x)})
    try:
        leaves_mean = np.mean(list(_iter_leaves(x)))
        leaves_min = min(_iter_leaves(x))
        leaves_max = max(_iter_leaves(x))
    except Exception:
        leaves_mean = "-"
        leaves_min = "-"
        leaves_max = "-"

    def _format_type(x: type) -> str:
        if x == type(None):
            return "None"

        return str(x.__name__)

    leaves_types_ = [_format_type(t) for t in leaves_types]
    if len(leaves_types_) == 1:
        leaves_types_ = leaves_types_[0]

    return Properties(
        py_type=type(x),
        dtype=f"py.{leaves_types_}",
        shape=shape,
        min=str(leaves_min),
        max=str(leaves_max),
        device="-",
        mean=str(leaves_mean),
        n_nans=n_nans,
    )


BASE_STYLES = {
    "torch": "bold cyan",
    "np": "bold green",
    "py": "bold yellow",
}

BASE_DEVICE_STYLES = {
    "cpu": "bold yellow",
    "cuda": "bold magenta",
}


def _format_dtype(x: str) -> str:
    """Format a dtype as a string."""
    if "." not in x:
        return f"[white]{x}[/]"
    type_, dtype_str = x.split(".")
    style = BASE_STYLES.get(type_, "")
    dtype_str = f"[{style}]{dtype_str}[/]"
    return f"[white]{type_}.[/white]{dtype_str}"


def _format_py_type(x: str) -> str:
    """Format a python type as a string."""
    style = {
        "list": BASE_STYLES["py"],
        "tuple": BASE_STYLES["py"],
        "set": BASE_STYLES["py"],
        "dict": BASE_STYLES["py"],
        "None": "bold red",
        "Tensor": BASE_STYLES["torch"],
        "ndarray": BASE_STYLES["np"],
    }.get(x, "white")
    return f"[{style}]{x}[/]"


def _format_device(device: str) -> str:
    """Format a device sas a string."""
    if device is None:
        return "-"
    if device.strip() == "-":
        return _default_formatter(device)
    if ":" in device:
        device, idx = device.split(":")
        idx = f"[bold white]:{idx}[/bold white]"
    else:
        idx = ""
    style = BASE_DEVICE_STYLES.get(device, None)
    if style is None:
        return f"{device}{idx}"

    return f"[{style}]{device}[/{style}]{idx}"


def _default_formatter(x: typ.Any) -> str:  # noqa: ANN401
    """Default formatter."""
    x = str(x)
    if x.strip() == "-":
        return f"[white]{x}[/]"

    return x


_FORMATTERS = {
    "dtype": _format_dtype,
    "py_type": _format_py_type,
    "device": _format_device,
}


def _format_field(field_name: str, field_value: typ.Any) -> str:  # noqa: ANN401
    """Apply a formatter to a field value based on its name."""
    formatter = _FORMATTERS.get(field_name, _default_formatter)
    return formatter(field_value)


def pprint_batch(
    batch: dict[str, typ.Any],
    idx: (
        None | list[int]
    ) = None,  # used here to enable compatibility with `datasets.map()`  # noqa: ARG001
    console: None | rich.console.Console = None,
    header: None | str = None,
    footer: None | str = None,
    sort_keys: bool = False,
    **kwargs: typ.Any,
) -> dict:
    """Pretty print a batch of data."""
    table = rich.table.Table(
        title=header, show_header=True, header_style="bold magenta"
    )
    fields = list(Properties.model_fields.keys())
    table.add_column("key", justify="left", style="bold cyan")
    for key in fields:
        table.add_column(key, justify="center")

    # Convert dict to flat dict
    flat_batch = flatten_dict(batch)
    flat_batch_keys = (
        sorted(flat_batch.keys()) if sort_keys else list(flat_batch.keys())
    )
    for key in flat_batch_keys:
        value = flat_batch[key]
        try:
            props = infer_properties(value)
            attrs = {f: getattr(props, f) for f in fields}
            table.add_row(key, *[_format_field(k, v) for k, v in attrs.items()])
        except Exception as e:
            logger.warning(
                f"Error while inferring properties for `{key}={value}` : {e}"
            )
            table.add_row(key, *["[red]ERROR[/red]"])

    if console is None:
        console = rich.console.Console()

    console.print(table)
    if footer is not None:
        console.print(footer)
    return {}


def _iter_leaves(x: typ.Iterable) -> typ.Iterable:
    for i in x:
        if isinstance(i, (list, tuple, set)):
            yield from _iter_leaves(i)
        else:
            yield i


def _safe_yaml(section: str) -> str:
    """Escape special characters in a YAML section."""
    section = section.replace(": ", r"\:")
    section = section.encode("unicode_escape").decode("utf-8")
    return section


def _cleanup_scores(sort_scores: list[float]) -> list[float]:
    non_nan_scores = [x for x in sort_scores if not math.isnan(x)]
    min_score = min(non_nan_scores) if len(non_nan_scores) > 0 else 0.0
    sort_scores = [min_score - 1 if math.isnan(x) else x for x in sort_scores]
    return sort_scores
