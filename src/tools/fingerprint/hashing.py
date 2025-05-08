import functools
import json

import datasets
import torch
from tools.misc.tensor_tools import serialize_tensor


def _hash_partial(hasher: datasets.fingerprint.Hasher, value: functools.partial) -> str:
    """The default hash of `functools.partial`."""
    data = {
        "cls": value.__class__,
        "func": value.func,
        "args": value.args,
        "keywords": value.keywords,
    }

    hashed = {k: hasher.hash(v) for k, v in data.items() if k not in {"keywords"}}
    hashed["keywords"] = {k: hasher.hash(v) for k, v in data["keywords"].items()}  # type: ignore
    return hasher.hash(hashed)


def _fingerprint_torch_module(
    hasher: datasets.fingerprint.Hasher, value: torch.nn.Module
) -> str:
    """Fingerprint a `torch.nn.Module`."""
    for k, v in value.state_dict().items():
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)
    return hasher.hexdigest()


def fingerprint_torch_module(value: torch.nn.Module) -> str:
    """Fingerprint a `torch.nn.Module`."""
    hasher = datasets.fingerprint.Hasher()
    return _fingerprint_torch_module(hasher, value)


def _hash_dataset(_: datasets.fingerprint.Hasher, value: datasets.Dataset) -> str:
    return value._fingerprint


def _hash_dataset_dict(
    hasher: datasets.fingerprint.Hasher, value: datasets.DatasetDict
) -> str:
    values = {key: value[key]._fingerprint for key in value}
    return hasher.hash(json.dumps(values))
