from collections import Counter
import functools
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
import transformers
from loguru import logger
from rich import table as rich_table
from typing_extensions import Self, Type


def flatten_dict(node: typ.Mapping[typ.Any, dict | typ.Any], sep: str = ".") -> dict[str, typ.Any]:
    """Flatten a nested dictionary. Keys are joined with `sep`."""
    output = {}
    for k, v in node.items():
        if isinstance(v, typ.Mapping):
            for k2, v2 in flatten_dict(v, sep=sep).items():
                output[f"{k}{sep}{k2}"] = v2
        else:
            output[k] = v
    return output


def human_format_nb(num: int | float, precision: int = 2, base: float = 1000.0) -> str:
    """Converts a number to a human-readable format."""
    magnitude = 0
    while abs(num) >= base:
        magnitude += 1
        num /= base
    # add more suffixes if you need them
    q = ["", "K", "M", "B", "T", "P"][magnitude]
    return f"{num:.{precision}f}{q}"


def pprint_class_balance(targets: list[list[int]]) -> str:
    def classify_target(target: list[int]) -> tuple[str, int]:
        if target == [0]:
            return ("negative", 1)  # Special case for zero lists
        else:
            num_positive_elements = len(target)
            return ("positive", num_positive_elements)

    class_counts = Counter(classify_target(target) for target in targets)
    total = len(targets)
    return ", ".join(f"{label}={np.round(count/total,1)}%" for label, count in sorted(class_counts.items()))


def pprint_parameters_stats(
    model: torch.nn.Module | transformers.PreTrainedModel,
    header: None | str = None,
    console: None | rich.console.Console = None,
) -> None:
    """Print the fraction of parameters for each `dtype` in the model."""
    dtype_counts = {}
    dtype_trainable_counts = {}
    total_params = 0
    total_trainable_params = 0

    for param in model.parameters():
        dtype = param.dtype
        num_params = param.numel()
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + num_params
        total_params += num_params

        if param.requires_grad:
            dtype_trainable_counts[dtype] = dtype_trainable_counts.get(dtype, 0) + num_params
            total_trainable_params += num_params

    # Compute the fraction of (trainable) parameters for each dtype
    dtype_fractions = {dtype: count / total_params for dtype, count in dtype_counts.items()}
    dtype_trainable_fractions = {
        dtype: dtype_trainable_counts.get(dtype, 0) / count for dtype, count in dtype_counts.items()
    }

    # Create a table using rich
    table = rich_table.Table(
        show_header=True,
        title=header,
    )
    table.add_column("Dtype", style="bold cyan")
    table.add_column("Number of Parameters", style="green")
    table.add_column("Fraction of Parameters", style="yellow")
    table.add_column("Fraction of Trainable Parameters", style="magenta")

    for dtype in dtype_counts:
        count = dtype_counts[dtype]
        fraction = dtype_fractions[dtype]
        trainable_fraction = dtype_trainable_fractions[dtype]
        table.add_row(
            str(dtype),
            human_format_nb(count),
            f"{fraction:.2%}",
            f"{trainable_fraction:.2%}",
        )

    # Add a row for total parameters
    if len(dtype_counts) > 1:
        table.add_section()
        table.add_row(
            "Total",
            human_format_nb(total_params),
            f"{1.0:.2%}",
            f"{total_trainable_params/total_params:.2%}",
        )

    console = console or rich.console.Console()
    console.print(table)


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
    def _cast_py_type(cls: Type[Self], value: typ.Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, type):
            return value.__name__

        return type(value).__name__


@functools.singledispatch
def infer_properties(x: typ.Any) -> Properties:
    """Base function for inferring properties of an object."""
    return Properties(py_type=type(x))


@infer_properties.register(torch.Tensor)
def _(x: torch.Tensor) -> Properties:
    """Infer properties of a torch tensor."""
    x_float = x.detach().float()
    return Properties(
        py_type=type(x),  # type: ignore
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
        py_type=type(x),  # type: ignore
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
        py_type=type(x),  # type: ignore
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
        py_type=type(x),  # type: ignore
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


def _default_formatter(x: typ.Any) -> str:
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


def _format_field(field_name: str, field_value: typ.Any) -> str:
    """Apply a formatter to a field value based on its name."""
    formatter = _FORMATTERS.get(field_name, _default_formatter)
    return formatter(field_value)


def pprint_batch(
    batch: dict[str, typ.Any],
    idx: None | list[int] = None,  # used here to enable compatibility with `datasets.map()`
    console: None | rich.console.Console = None,
    header: None | str = None,
    footer: None | str = None,
    sort_keys: bool = False,
    **kwargs: typ.Any,
) -> dict:
    """Pretty print a batch of data."""
    table = rich.table.Table(title=header, show_header=True, header_style="bold magenta")
    fields = list(Properties.model_fields.keys())
    table.add_column("key", justify="left", style="bold cyan")
    for key in fields:
        table.add_column(key, justify="center")

    # Convert dict to flat dict
    flat_batch = flatten_dict(batch)
    flat_batch_keys = sorted(flat_batch.keys()) if sort_keys else list(flat_batch.keys())
    for key in flat_batch_keys:
        value = flat_batch[key]
        try:
            props = infer_properties(value)
            attrs = {f: getattr(props, f) for f in fields}
            table.add_row(key, *[_format_field(k, v) for k, v in attrs.items()])
        except Exception as e:
            logger.warning(f"Error while inferring properties for `{key}={value}` : {e}")
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
