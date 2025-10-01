import typing

import datasets
import numpy as np

from dataloader import adapt
from dataloader.base import DatasetConfig, DatasetLoader


def _take_subset(
    dset: datasets.Dataset | datasets.DatasetDict, subset_size: None | int
) -> datasets.Dataset | datasets.DatasetDict:
    """Take a subset of the dataset."""
    if subset_size is None:
        return dset

    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(0)

    if isinstance(dset, datasets.Dataset):
        # Directly select a subset using random sampling
        subset_indices = rng.choice(len(dset), size=subset_size, replace=False)
        return dset.select(subset_indices)

    # Recursively apply to DatasetDict
    return datasets.DatasetDict(
        {k: _take_subset(v, subset_size) for k, v in dset.items()}
    )


def combine_datasets(
    inputs: list[datasets.Dataset | datasets.DatasetDict],
) -> datasets.Dataset | datasets.DatasetDict:
    """Combine a list of datasets into a single dataset."""
    if isinstance(inputs, (datasets.DatasetDict, dict)):
        return combine_datasets(list(inputs.values()))
    if isinstance(inputs, datasets.Dataset):
        return inputs
    if isinstance(inputs, list):
        inputs = [combine_datasets(d) for d in inputs]  # type: ignore
        return datasets.concatenate_datasets(inputs)  # type: ignore

    raise TypeError(f"Unexpected type `{type(inputs)}`")


def _load_one_dataset(
    name_or_path: str | DatasetLoader,
    name: str | None = None,
    subset: str | None = None,
    split: str | None = None,
    trust_remote_code: bool = True,
    **kws: typing.Any,
) -> datasets.Dataset | datasets.DatasetDict:
    if isinstance(name_or_path, str):
        data = datasets.load_dataset(
            name_or_path,
            name,
            subset,
            split=split,
            trust_remote_code=trust_remote_code,
            **kws,
        )
        if isinstance(data, (datasets.IterableDataset, datasets.IterableDatasetDict)):
            raise NotImplementedError(f"`{type(data)}` not supported.")

        return data

    try:
        return name_or_path(subset=subset, split=split, **kws)
    except Exception as e:
        raise RuntimeError(
            f"Failed to use `{name_or_path}` as a callable following the `{DatasetLoader}` protocol."
        ) from e


def _load_dataset_from_config(
    config: DatasetConfig, **kws: typing.Any
) -> datasets.Dataset | datasets.DatasetDict:
    """Load the dataset, process it according to the prompt template and return a HF dataset."""
    subsets = config.subsets or [None]
    loaded_subsets = [
        _load_one_dataset(config.name_or_path, subset, split=config.split, **kws)
        for subset in subsets
    ]
    if len(loaded_subsets) == 1:
        return loaded_subsets[0]
    return combine_datasets(loaded_subsets)


def load_dataset(
    config: DatasetConfig,
    **kws: dict[str, typing.Any],
) -> datasets.Dataset | datasets.DatasetDict:
    """Load a dataset."""
    dset = _load_dataset_from_config(config, **config.kwargs)
    dset = _take_subset(dset, config.options.subset_size)
    return adapt.transform(dset, options=config.options, verbose=False)
