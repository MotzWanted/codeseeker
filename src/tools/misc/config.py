import typing as typ
from collections.abc import Mapping

T = typ.TypeVar("T")


def flatten_dict(
    node: Mapping[typ.Any, dict | typ.Any], sep: str = "."
) -> dict[str, typ.Any]:
    """Flatten a nested dictionary. Keys are joined with `sep`.

    Example:
    ```
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    ```
    """
    output = {}
    for k, v in node.items():
        if isinstance(v, Mapping):
            for k2, v2 in flatten_dict(v, sep=sep).items():
                output[f"{k}{sep}{k2}"] = v2
        else:
            output[k] = v
    return output
